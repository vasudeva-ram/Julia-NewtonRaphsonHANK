# ─────────────────────────────────────────────────────────────────────────────
# KrusellSmith.jl — Model-specific code for the Krusell-Smith (1998) model
#
# This file contains structures and functions that are specific to the
# Krusell-Smith heterogeneous agent model. As the solver becomes more
# generic, model-specific code should live in files like this one, separate
# from the general solver infrastructure.
#
# Currently contains:
# - SteadyState struct (stores steady-state variables, policies, distribution)
# - exogenousZ! (AR(1) shock process for aggregate productivity)
# ─────────────────────────────────────────────────────────────────────────────


"""
    SteadyState

Stores the steady-state solution of the model. Contains the steady-state
values of all aggregate variables, the household policy functions, the
transition matrix, and the stationary distribution over household states.

Fields:
- `vars`: NamedTuple of steady-state variable values, ordered to match
   `model.varXs` (e.g., `(Y=1.0, KS=3.5, r=0.02, w=0.9, Z=1.0)`)
- `policies`: NamedTuple of policy matrices, one per aggregated variable,
   ordered to match `model.agg_vars` (e.g., `(KD = Matrix{Float64},)`)
- `Λ`: sparse transition matrix for the distribution (from `DistributionTransition`)
- `D`: stationary distribution vector (length `n_a * n_e`)
"""
struct SteadyState
    vars::NamedTuple
    policies::NamedTuple
    Λ::SparseMatrixCSC{Float64,Int64}
    D::Vector{Float64}
end


"""
    exogenousZ!(namedXvars::NamedTuple, model::SequenceModel)

Simulates one step of an AR(1) process for the aggregate productivity shock Z.
Expects `namedXvars` to contain fields `Z`, `ρ`, and `σ`.

Returns the updated value of Z.

Note: This is KS-specific. A generic exogenous process handler will
eventually replace this, driven by the `[ExogenousPaths]` TOML section.
"""
function exogenousZ!(namedXvars::NamedTuple,
    model::SequenceModel)

    @unpack Z, ρ, σ = namedXvars
    Z = ρ * Z + σ * sqrt(1 - ρ^2) * randn()
    return Z
end


"""
    ValueFunction(currentpolicy, namedvars, model)

Computes the implied state (endogenous grid) from the current policy function
using the Euler equation. Maps `currentpolicy` → `impliedstate`.
Note: this is model-specific (Krusell-Smith).
"""
function ValueFunction(currentpolicy, namedvars, model::SequenceModel)
    @unpack policygrid, shockmat, Π, policymat = model
    @unpack β, γ = model.params
    @unpack r, w = namedvars

    # Calculate the consumption matrix
    cprimemat = ((1 + r) .* policymat) + (w .* shockmat) - currentpolicy
    exponent = -1 * γ
    eulerlhs = β * (1 + r) * ((cprimemat .^ exponent) * Π')
    cmat = eulerlhs .^ (1 / exponent)

    # Interpolate the policy function to the grid
    impliedstate = (1 / (1 + r)) * (cmat - (w .* shockmat) + policymat)
    return impliedstate
end


"""
    backward_capital(x::Vector{Float64},
    currentpolicy::Matrix{Float64},
    model::SequenceModel)

Performs one step of the Endogenous Gridpoint method (Carroll 2006).
Note that this step is model specific, and the model specific elements are defined in 
the ValueFunction() function.
The function takes the current savings policy function and calculates the
policy function of HHs on the savings grid.
"""
function backward_capital(xVals, # (n_v x 1) vector of endogenous variable values
    currentpolicy, # current policy function guess
    model::SequenceModel)

    # Unpack objects
    @unpack policymat = model
    namedvars = NamedTuple{model.varXs}(xVals)

    impliedstate = ValueFunction(currentpolicy, namedvars, model)
    TF = eltype(currentpolicy)
    griddedpolicy = Matrix{TF}(undef, size(policymat))

    for i in 1:model.params.n_e
        linpolate = extrapolate(interpolate((impliedstate[:,i],), policymat[:,i], Gridded(Linear())), Flat())
        griddedpolicy[:,i] = linpolate.(policymat[:,i])
    end

    return griddedpolicy
end


"""
    steadystate_capital(xVals, model::SequenceModel)

KS-specific steady-state function for the `KD` aggregated variable.
Iterates `backward_capital` from a zero initial guess until the policy
function converges (sup-norm < `model.compspec.ε`), holding aggregate
variable values fixed at `xVals`.

This is the steady-state counterpart to `backward_capital` (one EGM step)
and `agg_capital` (forward aggregation). It is stored in `agg_vars` under
the `steadystate` key and called by `get_SteadyState` to obtain the
household savings policy at the steady state.

Grid dimensions are read from `model.heterogeneity` rather than `model.params`,
so no `n_a`/`n_e` fields are required in the parameter NamedTuple.
"""
function steadystate_capital(xVals, model::SequenceModel)
    @unpack ε = model.compspec
    n_a = model.heterogeneity.wealth.n
    n_e = model.heterogeneity.productivity.n

    guess    = zeros(n_a, n_e)
    newguess = zeros(n_a, n_e)
    tol = 1.0

    while ε < tol
        guess    = newguess
        newguess = backward_capital(xVals, guess, model)
        tol      = norm(newguess - guess)
    end

    return newguess
end


"""
    BackwardSteadyState(varNs, model::SequenceModel)

Applies the Endogenous Gridpoint method to find the steady-state policies.
Iterates `backward_capital` until convergence.

Deprecated: prefer `steadystate_capital`, which reads grid dimensions from
`model.heterogeneity` and is registered in `model.agg_vars`.
"""
function BackwardSteadyState(varNs, model::SequenceModel)
    return steadystate_capital(varNs, model)
end

