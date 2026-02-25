# ─────────────────────────────────────────────────────────────────────────────
# ks_model_functions.jl — Krusell-Smith model function file
#
# This file is referenced by KrusellSmith.yaml as `function_file` and is
# auto-included by `build_model_from_yaml` before the model is constructed.
# Any function named in the YAML (backward_function, ss_function, seq_function,
# grid_function) must be defined here or in a file included earlier in the
# include chain.
#
# Functions expected by KrusellSmith.yaml:
#   backward_capital    — EGM backward step           (defined in KrusellSmith.jl)
#   steadystate_capital — steady-state policy iterator (defined in KrusellSmith.jl)
#   exogenousZ          — AR(1) path generator for Z   (defined below)
#
# Built-in grid functions (defined in GeneralStructures.jl, always available):
#   double_exponential          — wealth grid
#   rouwenhorst_discretization  — productivity discretization
# ─────────────────────────────────────────────────────────────────────────────


"""
    exogenousZ(T::Int; ρ::Float64 = 0.9, σ::Float64 = 0.1) -> Vector{Float64}

Generates a T-period path for aggregate productivity Z starting from the
steady state (Z₀ = 1.0) via an AR(1) process:

    Z_t = ρ · Z_{t-1} + σ · √(1-ρ²) · ε_t,   ε_t ~ N(0,1)

Returns a Vector of length T.
"""
function exogenousZ(T::Int; ρ::Float64 = 0.9, σ::Float64 = 0.1)
    Z = ones(T)
    for t in 2:T
        Z[t] = ρ * Z[t-1] + σ * sqrt(1 - ρ^2) * randn()
    end
    return Z
end
