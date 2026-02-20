# ─────────────────────────────────────────────────────────────────────────────
# GeneralStructures.jl — Core data structures, grid construction, and utilities
#
# This file defines the model-agnostic infrastructure for the sequence-space
# HANK solver. It contains:
#
# 1. Heterogeneity dimensions: HeterogeneityDimension struct and build_dimension
#    factory for constructing grids/transition matrices from TOML config.
# 2. Model structure: SequenceModel (the complete model specification).
#    Parameters are stored as a NamedTuple (no ModelParams struct).
# 3. Time-shift operators: shift_lag, shift_lead for lag/lead notation in
#    compiled equilibrium equations.
# 4. Grid construction: make_DoubleExponentialGrid, get_RouwenhorstDiscretization.
# 5. Linear algebra utilities: vectorize_matrices, Vec2Mat, JVP,
#    RayleighQuotient, approximate_inverse_ilu.
#
# Model-specific code (e.g., SteadyState struct, exogenous shock processes)
# lives in separate files (e.g., KrusellSmith.jl).
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra, SparseArrays, DataFrames, UnPack, NLsolve, BenchmarkTools, Interpolations
using Zygote, ForwardDiff, IncompleteLU, IterativeSolvers, TOML


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heterogeneity dimensions
# ─────────────────────────────────────────────────────────────────────────────

"""
    HeterogeneityDimension

Represents one dimension of household heterogeneity (e.g., wealth, productivity, age).
Stored as values in a NamedTuple inside `SequenceModel.heterogeneity`, where the
key provides the dimension name (e.g., `model.heterogeneity.wealth`).

Fields:
- `dim_type::Symbol`: `:endogenous`, `:exogenous`, or `:deterministic`
- `n::Int`: number of grid/state points along this dimension
- `grid::Vector{Float64}`: discretized grid values (length `n`)
- `transition::Union{Matrix{Float64}, Nothing}`: transition matrix (`n × n`) for
   exogenous dimensions; `nothing` for endogenous dimensions
- `policy_var::Union{Symbol, Nothing}`: for endogenous dimensions, the name of the
   aggregated variable (in `model.agg_vars`) whose policy matrices are used to
   construct this dimension's transition matrix via Young's (2010) method.
   E.g., `:KD` for the wealth dimension in the KS model. `nothing` for exogenous
   and deterministic dimensions.
"""
struct HeterogeneityDimension
    dim_type::Symbol
    n::Int
    grid::Vector{Float64}
    transition::Union{Matrix{Float64}, Nothing}
    policy_var::Union{Symbol, Nothing}
end


"""
    n_total(heterogeneity::NamedTuple) -> Int

Returns the total number of states in the discretized heterogeneity space,
computed as the product of all dimension sizes. For example, with 200 wealth
points and 7 productivity points, `n_total` returns 1400.
"""
n_total(heterogeneity::NamedTuple) = prod(d.n for d in values(heterogeneity))


"""
    build_dimension(config::Dict{String, Any}) -> HeterogeneityDimension

Factory function that constructs a `HeterogeneityDimension` from a TOML config
dictionary. Dispatches on the `"type"` key:

- `"endogenous"`: builds a grid using the method specified in `"grid_method"`
  (currently supports `"DoubleExponential"` and `"Uniform"`)
- `"exogenous"`: discretizes a stochastic process using the method specified in
  `"discretization"` (currently supports `"Rouwenhorst"` for AR(1) processes)
- `"deterministic"`: builds a uniform grid over `"bounds"`

# Example
```julia
config = Dict("type" => "endogenous", "grid_method" => "DoubleExponential",
              "n" => 200, "bounds" => [0.0, 200.0])
dim = build_dimension(config)
```
"""
function build_dimension(config::Dict{String, Any})
    dim_type = Symbol(config["type"])

    if dim_type == :endogenous
        method = config["grid_method"]
        n = Int(config["n"])
        bounds = config["bounds"]
        policy_var = haskey(config, "policy_var") ? Symbol(config["policy_var"]) : nothing

        if method == "DoubleExponential"
            grid = collect(make_DoubleExponentialGrid(Float64(bounds[1]), Float64(bounds[2]), n))
        elseif method == "Uniform"
            grid = collect(range(Float64(bounds[1]), Float64(bounds[2]), length=n))
        else
            error("Unknown grid method: $method")
        end

        return HeterogeneityDimension(dim_type, n, grid, nothing, policy_var)

    elseif dim_type == :exogenous
        disc = config["discretization"]
        n = Int(config["n"])

        if disc == "Rouwenhorst"
            ρ = Float64(config["ρ"])
            σ = Float64(config["σ"])
            Π, _, z = get_RouwenhorstDiscretization(n, ρ, σ)
            return HeterogeneityDimension(dim_type, n, z, Π, nothing)
        else
            error("Unknown discretization method: $disc")
        end

    elseif dim_type == :deterministic
        n = Int(config["n"])
        bounds = config["bounds"]
        grid = collect(range(Float64(bounds[1]), Float64(bounds[2]), length=n))
        return HeterogeneityDimension(dim_type, n, grid, nothing, nothing)

    else
        error("Unknown dimension type: $dim_type")
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model structures
# ─────────────────────────────────────────────────────────────────────────────

"""
    ComputationalSpec

Fixed computational parameters required by every model. These define the
solver's discretization of the sequence-space problem and are always present
regardless of the economic model being solved.

Fields:
- `T::Int`: number of transition periods (sequence length)
- `ε::Float64`: convergence tolerance for iterative solvers
- `dx::Float64`: step size for numerical differentiation
- `n_v::Int`: number of aggregate variables (derived from `varXs`)
"""
struct ComputationalSpec
    T::Int
    ε::Float64
    dx::Float64
    n_v::Int
end


"""
    SequenceModel{F, A, H, P}

Complete specification of a heterogeneous agent model for the sequence-space solver.

Type parameters:
- `F`: type of the compiled residuals function
- `A`: type of the aggregated variables NamedTuple
- `H`: type of the heterogeneity NamedTuple
- `P`: type of the parameters NamedTuple

Fields:
- `varXs`: tuple of all aggregate variable names (endogenous + exogenous),
   e.g., `(:Y, :KS, :KD, :r, :w, :Z)`
- `equations`: equilibrium equation strings in immutable order,
   e.g., `("Y = Z * KS(-1)^α", ...)`
- `compspec`: `ComputationalSpec` — solver configuration (T, ε, dx, n_v)
- `params`: NamedTuple of economic parameters (used in equations,
   e.g., `α`, `β`, `δ`). Constructed dynamically from TOML, so different
   models can have different parameters without changing any struct definitions.
- `residuals_fn`: compiled function `(xMat::AbstractMatrix, params) -> Vector`
   produced by `compile_residuals`
- `agg_vars`: NamedTuple mapping aggregated variable symbols to
   `(backward, forward)` function pairs,
   e.g., `(KD = (backward = backward_capital, forward = agg_capital),)`
- `heterogeneity`: NamedTuple of `HeterogeneityDimension` objects,
   e.g., `(wealth = HeterogeneityDimension(...), productivity = HeterogeneityDimension(...))`
"""
struct SequenceModel{F, A, H, P}
    varXs::Tuple{Vararg{Symbol}}
    equations::Tuple{Vararg{String}}
    compspec::ComputationalSpec
    params::P
    residuals_fn::F
    agg_vars::A
    heterogeneity::H
end


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-shift operators
# ─────────────────────────────────────────────────────────────────────────────

"""
    shift_lag(x::AbstractVector, i::Int) -> AbstractVector

Shifts a time-series vector `x` backward by `i` periods. The first `i` entries
are filled with `x[1]` (steady-state boundary condition). Used by compiled
equations for lag notation: `KS(-1)` compiles to `shift_lag(xMat[2, :], 1)`.
"""
function shift_lag(x::AbstractVector, i::Int)
    return vcat(fill(x[1], i), x[1:end-i])
end


"""
    shift_lead(x::AbstractVector, i::Int) -> AbstractVector

Shifts a time-series vector `x` forward by `i` periods. The last `i` entries
are filled with `x[end]` (terminal steady-state boundary condition). Used by
compiled equations for lead notation: `C(+1)` compiles to `shift_lead(xMat[k, :], 1)`.
"""
function shift_lead(x::AbstractVector, i::Int)
    return vcat(x[i+1:end], fill(x[end], i))
end


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grid construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_DoubleExponentialGrid(amin::Float64, amax::Float64, n_a::Int64) -> StepRangeLen

Produces a double-exponential grid of asset holdings on `[amin, amax]` with
`n_a` points. Compared to a uniform grid, the double-exponential grid places
more points near the origin, providing greater precision for the savings
decisions of low-wealth households where policy function nonlinearities are
most prevalent.

The transformation is: `a = amin + exp(exp(u) - 1) - 1` where `u` is uniform
on `[0, log(1 + log(1 + amax - amin))]`.
"""
function make_DoubleExponentialGrid(amin::Float64,
    amax::Float64,
    n_a::Int64)

    𝕌 = log(1 + log(1 + amax - amin))
    𝕌grid = range(0, 𝕌, n_a)
    agrid = amin .+ exp.(exp.(𝕌grid) .- 1) .- 1

    return agrid
end


"""
    get_RouwenhorstDiscretization(n::Int64, ρ::Float64, σ::Float64) -> (Π, D, z)

Discretizes an AR(1) process using the Rouwenhorst (1995) method. Returns:
- `Π`: `n × n` transition matrix
- `D`: stationary distribution vector (length `n`)
- `z`: state-space grid, normalized so that `E[z] = 1`

The Rouwenhorst method is preferred over Tauchen (1986) for highly persistent
processes (large `ρ`). See Kopecky and Suen (2009) for details.

Note: calls `invariant_dist(Π)` from ForwardIteration.jl to compute the
stationary distribution.
"""
function get_RouwenhorstDiscretization(n::Int64,
    ρ::Float64,
    σ::Float64)

    p = (1 + ρ) / 2

    Π = [p 1-p; 1-p p]

    for i = 3:n
        Π_old = Π
        Π = zeros(i, i)
        Π[1:i-1, 1:i-1] += p * Π_old
        Π[1:i-1, 2:end] += (1-p) * Π_old
        Π[2:end, 1:i-1] += (1-p) * Π_old
        Π[2:end, 2:end] += p * Π_old
        Π[2:i-1, 1:end] /= 2
    end

    D = invariant_dist(Π)

    α = 2 * (σ / sqrt(n - 1))
    z = exp.(α * collect(0:n-1))
    z = z ./ sum(z .* D)

    return Π, D, z
end


# ─────────────────────────────────────────────────────────────────────────────
# 6. Linear algebra utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    vectorize_matrices(matrices::Vector{<:Matrix}) -> Vector

Converts a vector of matrices `[M1, M2, ...]` into a single flat vector
`[vec(M1); vec(M2); ...]`. Used for stacking policy or distribution matrices
across time periods into a single vector for the solver.
"""
function vectorize_matrices(matrices::Vector{<:Matrix})
    n, m = size(matrices[1])
    result = similar(matrices[1], n*m, length(matrices))
    for i in 1:length(matrices)
        result[:, i] = vec(matrices[i])
    end
    return [result...]
end


"""
    Vec2Mat(vec::Vector{Float64}, n::Int64, m::Int64) -> Vector{Matrix}

Reshapes a flat vector into a vector of `n × m` matrices. The number of
matrices is inferred as `length(vec) / (n * m)`. Inverse of `vectorize_matrices`.
"""
function Vec2Mat(vec::Vector{Float64}, n::Int64, m::Int64)
    T = Int(length(vec) / (n * m))
    kmat = reshape(vec, (n, m, T))
    return [kmat[:, :, i] for i in 1:T]
end


"""
    JVP(func::Function, primal::AbstractVector, tangent::AbstractVector) -> SparseVector

Computes the Jacobian-vector product (JVP) of `func` at point `primal` in
direction `tangent` using forward-mode automatic differentiation (ForwardDiff).
Returns `J(primal) * tangent` as a sparse vector.

This is the core operation for the Boehl (2024) Newton-Raphson iteration,
where directional derivatives are computed without forming the full Jacobian.
"""
function JVP(func::Function,
    primal::AbstractVector,
    tangent::AbstractVector)

    g(t) = func(primal + t * tangent)
    res = ForwardDiff.derivative(g, 0.0)

    return sparse(res)
end


"""
    RayleighQuotient(M, z) -> Float64

Computes the Rayleigh quotient `z'Mz / z'z` of matrix `M` and vector `z`.
Used in the Newton-Raphson solver for step-size adaptation.
"""
function RayleighQuotient(M, z)
    return dot(z, M * z) / dot(z, z)
end


"""
    approximate_inverse_ilu(iluJ, n::Int) -> SparseMatrixCSC

Approximates the inverse of a matrix using its ILU (Incomplete LU) factorization.
Solves `J * x = e_i` for each standard basis vector `e_i` to build the inverse
column by column. Used as a preconditioner for iterative solvers.
"""
function approximate_inverse_ilu(iluJ, n)
    Jinv = spzeros(n, n)
    Iden = sparse(I, n, n)

    for i in 1:n
        e_i = Iden[:, i]
        x_i = iluJ \ e_i
        Jinv[:, i] = x_i
    end

    return Jinv
end
