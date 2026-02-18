# Test script: build a SequenceModel from ModelFile.toml and inspect it

using Pkg
Pkg.activate(@__DIR__)

include("GeneralStructures.jl")
include("KrusellSmith.jl")
include("ModelParser.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")

# Function objects must be passed in — TOML only documents the names
agg_vars = (KD = (backward = backward_capital, forward = agg_capital),)

# Build the model
mod = build_model_from_toml("ModelFile.toml", agg_vars)

# Inspect the result
println("=== SequenceModel ===")
println("varXs:        ", mod.varXs)
println("equations:     ", mod.equations)
println()
println("=== compspec ===")
println("  T  = $(mod.compspec.T)")
println("  ε  = $(mod.compspec.ε)")
println("  dx = $(mod.compspec.dx)")
println("  n_v = $(mod.compspec.n_v)")
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
    println("    grid[1:3]  = $(dim.grid[1:min(3,end)])")
    println("    transition = $(isnothing(dim.transition) ? "nothing" : "$(size(dim.transition)) matrix")")
end
println()
println("=== agg_vars ===")
for (name, spec) in pairs(mod.agg_vars)
    println("  $name: backward=$(spec.backward), forward=$(spec.forward)")
end
println()
println("=== residuals_fn ===")
println("  type: ", typeof(mod.residuals_fn))

# Quick smoke test: evaluate residuals at a constant matrix
T = mod.compspec.T
n_v = mod.compspec.n_v
xMat = ones(n_v, T - 1)
try
    r = mod.residuals_fn(xMat, mod.params)
    println("  residuals_fn returned $(length(r))-element vector (expected $(length(mod.equations) * (T-1)))")
catch e
    println("  residuals_fn error: $e")
end
