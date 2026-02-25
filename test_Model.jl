# Test script: build a SequenceModel from KrusellSmith.yaml and inspect it.
#
# KrusellSmith.jl is included first because backward_capital and
# steadystate_capital (referenced by name in the YAML) live there for now.
# The YAML parser will also include ks_model_functions.jl automatically.

using Pkg
Pkg.activate(@__DIR__)

include("GeneralStructures.jl")
include("KrusellSmith.jl")
include("ModelParser.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")

# ── Build the model from YAML (function file included automatically) ──────────
mod = build_model_from_yaml("KrusellSmith.yaml")

# ── Inspect the result ────────────────────────────────────────────────────────
println("=== SequenceModel ===")
println("  var_names : ", var_names(mod))
println("  equations : ", mod.equations)
println()

println("=== compspec ===")
println("  T        = $(mod.compspec.T)")
println("  ε        = $(mod.compspec.ε)")
println("  dx       = $(mod.compspec.dx)")
println("  n_v      = $(mod.compspec.n_v)  (total vars)")
println("  n_endog  = $(mod.compspec.n_endog)  (endogenous only)")
println("  max_lag  = $(mod.compspec.max_lag)")
println("  max_lead = $(mod.compspec.max_lead)")
println()

println("=== params ===")
for (k, v) in pairs(mod.params)
    println("  $k = $v  ($(typeof(v)))")
end
println()

println("=== heterogeneity ===")
for (name, dim) in pairs(mod.heterogeneity)
    println("  $name:")
    println("    type       = $(dim.dim_type)")
    println("    n          = $(dim.n)")
    println("    grid[1:3]  = $(round.(dim.grid[1:min(3,end)]; digits=4))")
    println("    transition = $(isnothing(dim.transition) ? "nothing" :
                               "$(size(dim.transition)) matrix")")
    println("    policy_var = $(dim.policy_var)")
end
println()

println("=== variables ===")
for (name, var) in pairs(mod.variables)
    println("  $name  ($(var.var_type))")
    println("    description    = $(var.description)")
    println("    backward_fn    = $(isnothing(var.backward_fn)    ? "nothing" : typeof(var.backward_fn))")
    println("    steadystate_fn = $(isnothing(var.steadystate_fn) ? "nothing" : typeof(var.steadystate_fn))")
    println("    seq_fn         = $(isnothing(var.seq_fn)         ? "nothing" : typeof(var.seq_fn))")
end
println()

println("=== steady states ===")
println("  initial.fixed   = $(mod.ss_initial.fixed)")
println("  initial.guesses = $(mod.ss_initial.guesses)")
println("  ending.fixed    = $(mod.ss_ending.fixed)")
println("  ending.guesses  = $(mod.ss_ending.guesses)")
println()

println("=== residuals_fn ===")
println("  type: $(typeof(mod.residuals_fn))")

# ── Smoke test: evaluate residuals at a padded constant matrix ────────────────
# The compiled residuals function expects T_pad = (T-1) + max_lag + max_lead
# columns. It slices to the valid T-1 middle columns internally, returning
# n_eq × (T-1) residuals.
T_val    = mod.compspec.T
n_v      = mod.compspec.n_v
max_lag  = mod.compspec.max_lag
max_lead = mod.compspec.max_lead
T_pad    = (T_val - 1) + max_lag + max_lead

xMat_padded = ones(n_v, T_pad)
print("  residuals_fn smoke test ($(n_v)×$(T_pad) padded xMat) ... ")
try
    r        = mod.residuals_fn(xMat_padded, mod.params)
    expected = length(mod.equations) * (T_val - 1)
    println("returned $(length(r))-element vector (expected $expected) ✓")
catch e
    println("ERROR: $e")
end
