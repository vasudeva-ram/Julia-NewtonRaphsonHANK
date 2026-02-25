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


"""
    ValueFunction(currentpolicy, namedvars, model)

Computes the implied state (endogenous grid) from the current policy function
using the Euler equation. Maps `currentpolicy` → `impliedstate`.
Note: this is model-specific (Krusell-Smith).

Grid objects are read from `model.heterogeneity` at call time, so no
pre-built matrices are stored on the model struct.
"""
function ValueFunction(currentpolicy, namedvars, model::SequenceModel)
    n_a       = model.heterogeneity.wealth.n
    n_e       = model.heterogeneity.productivity.n
    grid      = model.heterogeneity.wealth.grid
    prod_grid = model.heterogeneity.productivity.grid
    Π         = model.heterogeneity.productivity.transition
    policymat = repeat(grid, 1, n_e)          # n_a × n_e; each col = wealth grid
    shockmat  = repeat(prod_grid', n_a, 1)    # n_a × n_e; each row = productivity grid

    @unpack β, γ = model.params
    @unpack r, w = namedvars

    # Calculate the consumption matrix
    cprimemat = ((1 + r) .* policymat) + (w .* shockmat) - currentpolicy
    exponent  = -1 * γ
    eulerlhs  = β * (1 + r) * ((cprimemat .^ exponent) * Π')
    cmat      = eulerlhs .^ (1 / exponent)

    # Endogenous grid: implied current wealth that rationalises currentpolicy
    impliedstate = (1 / (1 + r)) * (cmat - (w .* shockmat) + policymat)
    return impliedstate
end


"""
    backward_capital(xVals, currentpolicy, model::SequenceModel)

Performs one step of the Endogenous Gridpoint method (Carroll 2006).
`xVals` is a length-`n_v` vector of all aggregate variable values at the
current time step (must include at least `:r` and `:w`). `currentpolicy` is
the next-period savings policy matrix (n_a × n_e).

Returns the savings policy matrix on the wealth grid for the current period.

Note: this step is model-specific; the EGM core is in `ValueFunction`.
"""
function backward_capital(xVals,           # length-n_v vector of variable values at time t
                          currentpolicy,   # next-period policy matrix (n_a × n_e)
                          model::SequenceModel)
    n_e  = model.heterogeneity.productivity.n
    grid = model.heterogeneity.wealth.grid

    namedvars = NamedTuple{var_names(model)}(Tuple(xVals))

    impliedstate  = ValueFunction(currentpolicy, namedvars, model)
    TF            = eltype(currentpolicy)
    policymat     = repeat(grid, 1, n_e)
    griddedpolicy = Matrix{TF}(undef, size(policymat))

    for i in 1:n_e
        linpolate = extrapolate(
            interpolate((impliedstate[:, i],), policymat[:, i], Gridded(Linear())),
            Flat())
        griddedpolicy[:, i] = linpolate.(policymat[:, i])
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

