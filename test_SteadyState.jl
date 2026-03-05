# test_SteadyState.jl — end-to-end test of the steady-state solver.
#
# Builds the KS model from YAML, runs get_SteadyState, and verifies that
#   (a) the Newton solver converged (residual norm < ε)
#   (b) the aggregate variables satisfy the equilibrium conditions manually

using Pkg
Pkg.activate(@__DIR__)
using Random

include("GeneralStructures.jl")
include("KrusellSmith.jl")
include("ModelParser.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")
include("SteadyState.jl")
include("SteadyStateJacobian.jl")

println("=== Building model from YAML ===")
mod = build_model_from_yaml("KrusellSmith.yaml")
println("  n_v      = $(mod.compspec.n_v)")
println("  n_endog  = $(mod.compspec.n_endog)")
println("  max_lag  = $(mod.compspec.max_lag)")
println("  max_lead = $(mod.compspec.max_lead)")
println()

# asm = SSAssembler(mod)
# p = Float64[get(mod.ss_initial.guesses, k, 1.0) for k in asm.free_keys]
# xvls = get_xVals(asm, p)


println("=== Running get_SteadyStates ===")
t0 = time()
ss_initial, ss_ending = get_SteadyStates(mod)
elapsed = time() - t0
println("  Elapsed: $(round(elapsed; digits=2)) s")
println()

println("=== Initial steady-state variable values ===")
for (k, v) in pairs(ss_initial.vars)
    println("  $k = $(round(v; digits=6))")
end
println()

println("=== Checking equilibrium conditions manually (initial SS) ===")
@unpack α, δ = mod.params
Z  = ss_initial.vars.Z
KS = ss_initial.vars.KS
KD = ss_initial.vars.KD
r  = ss_initial.vars.r
w  = ss_initial.vars.w
Y  = ss_initial.vars.Y

println("  Y   = Z * KS^α           →  $(round(Z * KS^α; digits=6))  (model: $(round(Y; digits=6)))")
println("  r+δ = α*Z*KS^(α-1)       →  $(round(α*Z*KS^(α-1); digits=6))  (model: $(round(r+δ; digits=6)))")
println("  w   = (1-α)*Z*KS^α       →  $(round((1-α)*Z*KS^α; digits=6))  (model: $(round(w; digits=6)))")
println("  KS  = KD                  →  KS=$(round(KS; digits=6))  KD=$(round(KD; digits=6))")
println()

println("=== Residual norm at initial SS ===")
all_keys = var_names(mod)
max_lag  = mod.compspec.max_lag
max_lead = mod.compspec.max_lead
T_pad    = 1 + max_lag + max_lead
n_v      = mod.compspec.n_v

xMat_ss = zeros(n_v, T_pad)
for col in 1:T_pad
    for (i, k) in enumerate(all_keys)
        xMat_ss[i, col] = ss_initial.vars[k]
    end
end

resid = mod.residuals_fn(xMat_ss, mod.params)
println("  residual vector = $resid")
println("  residual norm   = $(norm(resid))")
println()

if norm(resid) < 10 * mod.compspec.ε
    println("PASS: steady state residuals are within tolerance.")
else
    println("FAIL: steady state residuals exceed tolerance.")
end

# ─────────────────────────────────────────────────────────────────────────────
# Test: flatten_policies / unflatten_policies are strict inverses.
#
# These closures live inside getIntdJacobians and cannot be called directly, so
# we replicate the same logic here to verify correctness before they are used in
# the Jacobian computation.
# ─────────────────────────────────────────────────────────────────────────────
println("=== Testing flatten_policies / unflatten_policies ===")

T_test      = mod.compspec.T
het_keys    = vars_of_type(mod, :heterogeneous)
n_agg       = length(het_keys)
Tv          = n_total(mod.heterogeneity)
policy_size = size(first(values(ss_initial.policies)))

# Build a test policy_seqs: each variable gets T-1 distinct random matrices.
rng = (k, t) -> randn(policy_size)
policy_seqs = NamedTuple{het_keys}(
    Tuple([rng(k, t) for t in 1:T_test-1] for k in het_keys))

# ── replicate flatten_policies ──────────────────────────────────────────
function flatten_policies(ps::NamedTuple)
    return vcat([vcat([vec(mat) for mat in seq]...) for seq in values(ps)]...)
end

# ── replicate unflatten_policies ─────────────────────────────────────────
function unflatten_policies(a_flat)
    n_per_var = Tv * (T_test - 1)
    seqs = ntuple(n_agg) do k
        offset = (k - 1) * n_per_var
        [reshape(a_flat[offset + (i-1)*Tv + 1 : offset + i*Tv], policy_size)
         for i in 1:T_test-1]
    end
    return NamedTuple{het_keys}(seqs)
end

a_flat      = flatten_policies(policy_seqs)
policy_back = unflatten_policies(a_flat)

# Check 1: flat vector has the expected length
expected_len = n_agg * Tv * (T_test - 1)
@assert length(a_flat) == expected_len (
    "flatten length mismatch: got $(length(a_flat)), expected $expected_len")

# Check 2: round-trip recovers identical matrices for every variable and period
all_match = all(
    policy_seqs[k][t] == policy_back[k][t]
    for k in het_keys, t in 1:T_test-1)
@assert all_match "flatten→unflatten round-trip failed: matrices differ"

# Check 3: keys are in the same order
@assert keys(policy_back) == keys(policy_seqs) (
    "key order mismatch after unflatten: $(keys(policy_back)) vs $(keys(policy_seqs))")

println("  PASS: flatten/unflatten are exact inverses ($(n_agg) variable(s), " *
        "$(T_test-1) periods, policy size $policy_size)")

# ─────────────────────────────────────────────────────────────────────────────
# Test: Steady-state Jacobian columns match full-model JVPs.
#
# getSteadyStateJacobian decomposes dF/dx into direct (JDI) and indirect
# (JFI * JBI) parts and assembles the full block-Toeplitz Jacobian. We verify
# selected columns of the result against JVPs of the complete pipeline:
#
#   fullPipelineFunc(xVec_endog) =
#     Residuals(assemble_full_xMat(x, ForwardIteration(BackwardIteration(x))))
#
# Each JVP is computed independently via ForwardDiff (through BackwardIteration's
# EGM loop) and provides a model-agnostic reference column.
#
# Note: each JVP call runs a full BackwardIteration with Dual numbers and is
# therefore slow (~same cost as one steady-state solve). 7 columns is
# sufficient to give confidence without making the test suite prohibitive.
#
# Test indices: 1, 2 (boundary), 3 random interior, n-1, n (boundary).
# ─────────────────────────────────────────────────────────────────────────────
println("=== Computing steady-state Jacobian ===")
t_jac = time()
J_computed = getSteadyStateJacobian(ss_initial, mod)
println("  Elapsed: $(round(time() - t_jac; digits=2)) s")
println("  Size: $(size(J_computed))")
println("  Non-zeros: $(nnz(J_computed))")
println()

println("=== Testing Jacobian columns against full-model JVPs ===")

@unpack T, n_endog = mod.compspec

endog_keys    = vars_of_type(mod, :endogenous)
exog_keys     = vars_of_type(mod, :exogenous)

# SS endogenous path: all periods pinned at ss_initial values.
xVec_endog = repeat(Float64[ss_initial.vars[k] for k in endog_keys], T - 1)

# Constant exogenous paths (same convention as getSteadyStateJacobian).
exog_paths_ss = NamedTuple{exog_keys}(
    Tuple(fill(Float64(ss_initial.vars[k]), T - 1) for k in exog_keys))

# Full pipeline: BackwardIteration → ForwardIteration → assemble → Residuals.
# This is the function whose Jacobian getSteadyStateJacobian approximates.
function fullPipelineFunc(xVec_e::AbstractVector)
    policy_seqs = BackwardIteration(xVec_e, exog_paths_ss, mod, ss_initial)
    agg_seqs    = ForwardIteration(policy_seqs, mod, ss_initial)
    padded_xMat = assemble_full_xMat(xVec_e, agg_seqs, exog_paths_ss,
                                     mod, ss_initial, ss_initial)
    return Residuals(padded_xMat, mod)
end

n = n_endog * (T - 1)

# Choose 7 column indices: first 2, 3 random interior, last 2.
Random.seed!(42)
interior_cols = rand(3:n-2, 3)
test_indices  = [1, 2, interior_cols..., n-1, n]

println("  n = $n   (n_endog=$(n_endog), T-1=$(T-1))")
println("  Testing columns: $test_indices")
println()

tol      = 1e-5   # absolute tolerance on column error
all_pass = true

for idx in test_indices
    e_i = spzeros(n)
    e_i[idx] = 1.0

    t0           = time()
    jvp_col      = JVP(fullPipelineFunc, xVec_endog, e_i)
    computed_col = Vector(J_computed[:, idx])
    elapsed_col  = round(time() - t0; digits=2)

    err      = norm(jvp_col - computed_col)
    rel_err  = norm(jvp_col) > 1e-14 ? err / norm(jvp_col) : err
    status   = err < tol ? "PASS" : "FAIL"
    all_pass = all_pass && (err < tol)

    println("  col $(lpad(idx,4)): ||JVP - J_col|| = $(round(err; digits=8))  " *
            "(rel=$(round(rel_err; digits=6)))  $(status)  [$(elapsed_col)s]")
end

println()
if all_pass
    println("PASS: all $(length(test_indices)) Jacobian columns match full-model JVPs.")
else
    println("FAIL: one or more Jacobian columns deviate beyond tolerance $tol.")
end
