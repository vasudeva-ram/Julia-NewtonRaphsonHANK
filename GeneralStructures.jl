# ─────────────────────────────────────────────────────────────────────────────
# GeneralStructures.jl — Core data structures, grid construction, and utilities
#
# This file defines the model-agnostic infrastructure for the sequence-space
# HANK solver. It contains:
#
# 1. Heterogeneity dimensions: HeterogeneityDimension struct.
# 2. Steady-state specification: SteadyStateSpec struct (fixed values + guesses).
# 3. Model structure: SequenceModel + Variable (the complete model specification).
#    Parameters are stored as a NamedTuple (no ModelParams struct).
# 4. Out-of-the-box grid functions: double_exponential, rouwenhorst_discretization
#    (callable by name from YAML model files; wrappers over the primitives below).
# 5. Sequence-space assembly helpers: generate_exog_paths, assemble_full_xMat.
# 6. Time-shift operators: shift_lag, shift_lead for lag/lead notation in
#    compiled equilibrium equations.
# 7. Grid construction primitives: make_DoubleExponentialGrid, get_RouwenhorstDiscretization.
# 8. Linear algebra utilities: vectorize_matrices, Vec2Mat, JVP,
#    RayleighQuotient, approximate_inverse_ilu.
#
# Model-specific code (e.g., SteadyState struct, exogenous shock processes)
# lives in separate files (e.g., KrusellSmith.jl).
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra, SparseArrays, DataFrames, UnPack, NLsolve, BenchmarkTools, Interpolations
using Zygote, ForwardDiff, IncompleteLU, IterativeSolvers


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
    SteadyStateSpec{F, G}

Specification for a single steady state of the model. Holds two NamedTuples:
- `fixed::F`: variables pinned to known values (e.g., exogenous Z = 1.0)
- `guesses::G`: initial guesses for the free endogenous variables that the
  Newton-Raphson steady-state solver will search over (e.g., r = 0.08, w = 1.7)

Type parameters `F` and `G` capture the exact NamedTuple types so that
`SequenceModel` can be fully concrete.
"""
struct SteadyStateSpec{F, G}
    fixed::F    # NamedTuple of pinned values,      e.g. (Z = 1.0,)
    guesses::G  # NamedTuple of initial guesses,    e.g. (r = 0.08, w = 1.7)
end



# ─────────────────────────────────────────────────────────────────────────────
# 3. Model structures
# ─────────────────────────────────────────────────────────────────────────────

"""
    Variable{B, S}

Describes a single aggregate variable in the model. Carries both metadata
(name, type, description) and — for heterogeneous variables — the functions
needed to compute it from household policies.

Type parameters:
- `B`: concrete type of `backward_fn` (`typeof(backward_capital)` or `Nothing`)
- `S`: concrete type of `steadystate_fn` (`typeof(steadystate_capital)` or `Nothing`)

These type parameters ensure that calls to `var.backward_fn(...)` are statically
dispatched — no dynamic dispatch through an abstract `Function` type. Since each
`Variable{B,S}` stored in `model.variables` is fully concrete, and `SequenceModel`
captures the entire `variables` NamedTuple type in its `V` parameter, the compiler
sees exact function types throughout the backward/forward iteration hot loops.

Fields:
- `name::Symbol`: variable identifier, e.g. `:KD`
- `var_type::Symbol`: one of `:endogenous`, `:exogenous`, or `:heterogeneous`
  - `:endogenous`    — free variable in the Newton search at steady state
  - `:exogenous`     — pinned by `[InitialSteadyState]` / `[ExogenousPaths]` in TOML
  - `:heterogeneous` — aggregated from the household distribution; requires
                       `backward_fn` and `steadystate_fn`
- `description::String`: human-readable label, e.g. `"Capital demand"`
- `backward_fn::B`: for `:heterogeneous` variables, the EGM backward-step function
  `(xVals, currentpolicy, model) -> policy_matrix`; `nothing` otherwise
- `steadystate_fn::S`: for `:heterogeneous` variables, the steady-state policy
  iterator `(xVals, model) -> policy_matrix`; `nothing` otherwise
"""
struct Variable{B, S, Q}
    name::Symbol
    var_type::Symbol
    description::String
    backward_fn::B      # for :heterogeneous: EGM step (xVals, policy, model) → policy; else Nothing
    steadystate_fn::S   # for :heterogeneous: SS iterator (xVals, model) → policy; else Nothing
    seq_fn::Q           # for :exogenous: path generator (T, ...) → Vector; else Nothing
end

"""
    Variable(name, var_type, description)

Convenience constructor for `:endogenous` variables with no associated
functions. Produces `Variable{Nothing, Nothing, Nothing}`.
"""
Variable(name::Symbol, var_type::Symbol, description::String) =
    Variable(name, var_type, description, nothing, nothing, nothing)

"""
    Variable(name, var_type, description, backward_fn, steadystate_fn)

Convenience constructor for `:heterogeneous` variables (no seq_fn).
Produces `Variable{B, S, Nothing}`.
"""
Variable(name::Symbol, var_type::Symbol, description::String, backward_fn, steadystate_fn) =
    Variable(name, var_type, description, backward_fn, steadystate_fn, nothing)


"""
    var_names(model::SequenceModel) -> Tuple{Vararg{Symbol}}

Returns the ordered tuple of all aggregate variable names in the model,
matching the row ordering of `xMat`.
"""
var_names(model) = keys(model.variables)


"""
    vars_of_type(model::SequenceModel, t::Symbol) -> Tuple{Vararg{Symbol}}

Returns the names of all variables whose `var_type` equals `t`.
`t` should be one of `:endogenous`, `:exogenous`, or `:heterogeneous`.
"""
vars_of_type(model, t::Symbol) =
    Tuple(k for (k, v) in pairs(model.variables) if v.var_type == t)


"""
    ComputationalSpec

Fixed computational parameters required by every model. These define the
solver's discretization of the sequence-space problem and are always present
regardless of the economic model being solved.

Fields:
- `T::Int`: number of transition periods (sequence length); the economy is at
  `ss_initial` at t=0 and at `ss_ending` at t=T. The Newton search covers the
  T-1 intermediate periods t=1,...,T-1.
- `ε::Float64`: convergence tolerance for iterative solvers
- `dx::Float64`: step size for numerical differentiation
- `n_v::Int`: total number of aggregate variables across all types
  (endogenous + heterogeneous + exogenous); equals the number of rows in the
  full `xMat` passed to the compiled residuals function.
- `n_endog::Int`: number of `:endogenous` variables only. The Newton search
  vector has dimension `n_endog × (T-1)`.
- `max_lag::Int`: maximum lag depth across all compiled equations (e.g. 1 for
  `KS(-1)`, 3 for `KS(-3)`). The padded xMat prepends `max_lag` copies of the
  initial SS column so that `shift_lag` always reads the correct boundary.
- `max_lead::Int`: maximum lead depth across all compiled equations. Appends
  `max_lead` copies of the ending SS column for `shift_lead` boundary.
"""
struct ComputationalSpec
    T::Int
    ε::Float64
    dx::Float64
    n_v::Int
    n_endog::Int
    max_lag::Int
    max_lead::Int
end


"""
    SequenceModel{F, H, P, V, SI, SE}

Complete specification of a heterogeneous agent model for the sequence-space solver.

Type parameters:
- `F`:  type of the compiled residuals function
- `H`:  type of the heterogeneity NamedTuple
- `P`:  type of the parameters NamedTuple
- `V`:  type of the variables NamedTuple
- `SI`: type of the initial `SteadyStateSpec`
- `SE`: type of the ending `SteadyStateSpec`

Fields:
- `variables`: NamedTuple mapping each variable symbol to a `Variable` object,
   e.g., `(Y = Variable(:Y, :endogenous, …), KD = Variable(:KD, :heterogeneous, …), …)`.
   The NamedTuple ordering defines the row ordering of `xMat` throughout the solver.
   Use `var_names(model)` to get the ordered symbol tuple and `vars_of_type(model, t)`
   to filter by `:endogenous`, `:exogenous`, or `:heterogeneous`.
- `equations`: equilibrium equation strings in immutable order,
   e.g., `("Y = Z * KS(-1)^α", ...)`
- `compspec`: `ComputationalSpec` — solver configuration (T, ε, dx, n_v)
- `params`: NamedTuple of economic parameters (e.g., `α`, `β`, `δ`). Constructed
   dynamically from the YAML file so different models need no struct changes.
- `residuals_fn`: compiled function `(xMat::AbstractMatrix, params) -> Vector`
   produced by `compile_residuals`
- `ss_initial`: `SteadyStateSpec` for the initial steady state — `fixed` holds pinned
   values (e.g., `Z = 1.0`); `guesses` holds Newton starting values (e.g., `r = 0.08`)
- `ss_ending`: `SteadyStateSpec` for the ending steady state. Equals `ss_initial` when
   only one steady state is specified (transitory shock).
- `heterogeneity`: NamedTuple of `HeterogeneityDimension` objects,
   e.g., `(wealth = HeterogeneityDimension(...), productivity = HeterogeneityDimension(...))`
"""
struct SequenceModel{F, H, P, V, SI, SE}
    variables::V
    equations::Tuple{Vararg{String}}
    compspec::ComputationalSpec
    params::P
    residuals_fn::F
    ss_initial::SI
    ss_ending::SE
    heterogeneity::H
end


# ─────────────────────────────────────────────────────────────────────────────
# 4. Out-of-the-box grid functions (referenced by name in YAML model files)
# ─────────────────────────────────────────────────────────────────────────────

"""
    double_exponential(; n, grid_min, grid_max) -> Vector{Float64}

Out-of-the-box endogenous grid function for YAML model specifications.
Constructs a double-exponential asset grid of `n` points on `[grid_min, grid_max]`.
Thin wrapper over `make_DoubleExponentialGrid`.

Endogenous dimension convention: returns a single `Vector{Float64}`.
"""
function double_exponential(; n, grid_min, grid_max)
    return collect(Float64, make_DoubleExponentialGrid(Float64(grid_min), Float64(grid_max), Int(n)))
end


"""
    rouwenhorst_discretization(; n, ρ, σ) -> (Vector{Float64}, Matrix{Float64})

Out-of-the-box exogenous grid function for YAML model specifications.
Discretizes an AR(1) process via Rouwenhorst (1995). Thin wrapper over
`get_RouwenhorstDiscretization`.

Exogenous dimension convention: returns a **2-tuple `(grid, Π)`** where
- `grid` is the length-`n` state-space vector (normalized so E[z] = 1)
- `Π` is the `n × n` row-stochastic transition matrix
"""
function rouwenhorst_discretization(; n, ρ, σ)
    Π, _, z = get_RouwenhorstDiscretization(Int(n), Float64(ρ), Float64(σ))
    return z, Π   # (grid, transition_matrix) — grid first, matrix second
end


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequence-space assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_exog_paths(model::SequenceModel, T::Int) -> NamedTuple

Calls the `seq_fn` of every `:exogenous` variable in `model` to generate a
length-`T` path for each. Returns a NamedTuple keyed by the exogenous variable
names, e.g. `(Z = [1.0, 0.98, ...],)`.

`T` should equal `model.compspec.T - 1` (the number of transition periods).

Raises an error for any exogenous variable that lacks a `seq_fn`.
"""
function generate_exog_paths(model::SequenceModel, T::Int)
    exog_keys = vars_of_type(model, :exogenous)
    paths = map(exog_keys) do key
        var = model.variables[key]
        isnothing(var.seq_fn) &&
            error("Exogenous variable '$key' has no seq_fn. " *
                  "Specify a seq_function in the YAML.")
        var.seq_fn(T)
    end
    return NamedTuple{exog_keys}(paths)
end


"""
    assemble_full_xMat(xVec_endog, agg_seqs, exog_paths, model,
                       ss_start, ss_end) -> Matrix

Assembles the padded `n_v × T_pad` matrix required by the compiled residuals
function, where `T_pad = (T-1) + max_lag + max_lead`.

## Column layout

| Columns           | Content                                        |
|-------------------|------------------------------------------------|
| `1:max_lag`       | Initial SS boundary — all rows = `ss_start.vars` |
| `max_lag+1:max_lag+T-1` | Transition path (T-1 periods)          |
| `max_lag+T:T_pad` | Ending SS boundary — all rows = `ss_end.vars`  |

## Row layout (matches `var_names(model)` ordering)

Within the transition columns, rows are filled from three sources:
- `:endogenous` rows — from `xVec_endog` reshaped to `(n_endog × T-1)`
- `:heterogeneous` rows — from `agg_seqs[varname][t]` (forward-iteration aggregates)
- `:exogenous` rows — from `exog_paths[varname][t]`

## AD compatibility

The SS boundary columns are assigned from `Float64` values (with zero
ForwardDiff partials), so gradients flow only through `xVec_endog` and
`agg_seqs` (which themselves depend on `xVec_endog`). This is the intended
behaviour: the solver differentiates with respect to the endogenous sequence,
not the fixed boundary conditions.

`ss_start` and `ss_end` are expected to have a `.vars::NamedTuple` field
keyed by `var_names(model)` (satisfied by the `SteadyState` struct).
"""
function assemble_full_xMat(xVec_endog::AbstractVector,
                             agg_seqs::NamedTuple,
                             exog_paths::NamedTuple,
                             model::SequenceModel,
                             ss_start,
                             ss_end)
    @unpack T, n_v, n_endog, max_lag, max_lead = model.compspec
    T_pad = (T - 1) + max_lag + max_lead   # total columns in padded matrix

    TF   = eltype(xVec_endog)
    xMat = zeros(TF, n_v, T_pad)

    all_keys   = var_names(model)
    endog_keys = vars_of_type(model, :endogenous)
    het_keys   = vars_of_type(model, :heterogeneous)
    exog_keys  = vars_of_type(model, :exogenous)

    # Precompute row indices for each variable group
    endog_rows = [findfirst(==(k), all_keys) for k in endog_keys]
    het_rows   = [findfirst(==(k), all_keys) for k in het_keys]
    exog_rows  = [findfirst(==(k), all_keys) for k in exog_keys]

    # ── Initial SS boundary columns (1:max_lag) ───────────────────────────────
    # Assigned as Float64 → implicit convert to TF with zero derivatives.
    for col in 1:max_lag
        for row in 1:n_v
            xMat[row, col] = ss_start.vars[all_keys[row]]
        end
    end

    # ── Ending SS boundary columns (max_lag+T : T_pad) ────────────────────────
    for col in (max_lag + T):T_pad
        for row in 1:n_v
            xMat[row, col] = ss_end.vars[all_keys[row]]
        end
    end

    # ── Transition columns (max_lag+1 : max_lag+T-1) ─────────────────────────
    xMat_endog = reshape(xVec_endog, n_endog, T - 1)   # n_endog × (T-1)

    for t in 1:T-1
        col = max_lag + t

        for (j, row) in enumerate(endog_rows)
            xMat[row, col] = xMat_endog[j, t]
        end

        for (j, key) in enumerate(het_keys)
            xMat[het_rows[j], col] = agg_seqs[key][t]
        end

        for (j, key) in enumerate(exog_keys)
            xMat[exog_rows[j], col] = exog_paths[key][t]
        end
    end

    return xMat
end


# ─────────────────────────────────────────────────────────────────────────────
# 6. Time-shift operators
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
# 7. Grid construction primitives
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
# 8. Linear algebra utilities
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
