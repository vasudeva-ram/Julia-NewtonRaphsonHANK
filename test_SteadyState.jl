# test_SteadyState.jl — end-to-end test of the steady-state solver.
#
# Builds the KS model from YAML, runs get_SteadyState, and verifies that
#   (a) the Newton solver converged (residual norm < ε)
#   (b) the aggregate variables satisfy the equilibrium conditions manually

using Pkg
Pkg.activate(@__DIR__)

include("GeneralStructures.jl")
include("KrusellSmith.jl")
include("ModelParser.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")
include("SteadyState.jl")

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
