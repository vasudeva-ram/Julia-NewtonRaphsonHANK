# KrusellSmith.jl — model-specific functions for Krusell-Smith (1998).
#
# Auto-included by build_model_from_yaml. Contains:
#   exogenousZ    — AR(1) path generator for aggregate productivity Z
#   ValueFunction — one EGM step: F: ∂V/∂a' → (Value=∂V/∂a, KD=a')


"""
    exogenousZ(T::Int; ρ::Float64 = 0.9, σ::Float64 = 0.1) -> Vector{Float64}

Generates a T-period path for aggregate productivity Z starting from Z₀ = 1
via the AR(1) process Z_t = ρ·Z_{t-1} + σ·√(1-ρ²)·ε_t, ε_t ~ N(0,1).
"""
function exogenousZ(T::Int; ρ::Float64 = 0.9, σ::Float64 = 0.1)
    Z = ones(T)
    for t in 2:T
        Z[t] = ρ * Z[t-1] + σ * sqrt(1 - ρ^2) * randn()
    end
    return Z
end


"""
    ValueFunction(value_next, xVals, model) -> NamedTuple

One step of the Endogenous Grid Method (Carroll 2006) for the KS household
problem. Maps the next-period marginal value ∂V_{t+1}/∂a' (n_a × n_e matrix)
to the current-period marginal value and savings policy:

    F : ∂V_{t+1}/∂a' → (Value = ∂V_t/∂a,  KD = a'(a,e))

## Algorithm

1. Euler equation: c_t = (β · E_{e'|e}[∂V_{t+1}/∂a'])^{-1/γ}
2. Implied current wealth on the endogenous grid: a = (c_t + a' - w·e) / (1+r)
3. Interpolate savings policy a'(·,e) onto the exogenous wealth grid.
4. Enforce borrowing constraint: a' ≥ borrow_cons.
5. Marginal value: ∂V_t/∂a = (1+r) · c_t^{-γ}

AD-compatible: TF is promoted from eltype(xVals) and eltype(value_next),
so ForwardDiff dual numbers flow through cleanly.
"""
function ValueFunction(value_next, xVals, model::SequenceModel)
    n_a       = model.heterogeneity.wealth.n
    n_e       = model.heterogeneity.productivity.n
    grid      = model.heterogeneity.wealth.grid
    prod_grid = model.heterogeneity.productivity.grid
    Π         = model.heterogeneity.productivity.transition
    policy_a  = repeat(grid, 1, n_e)        # n_a × n_e; each col = wealth grid
    labor_mat = repeat(prod_grid', n_a, 1)  # n_a × n_e; each row = productivity grid

    @unpack β, γ, borrow_cons = model.params
    namedvars = NamedTuple{var_names(model)}(Tuple(xVals))
    @unpack r, w = namedvars

    TF = promote_type(eltype(xVals), eltype(value_next))

    # Step 1: expected marginal value → consumption on the endogenous grid
    cmat = (β .* (value_next * Π')) .^ (-1 / γ)

    # Step 2: implied current wealth for each (a', e) pair
    impliedstate = (1 / (1 + r)) .* (cmat .- (w .* labor_mat) .+ policy_a)

    # Step 3: interpolate onto the exogenous wealth grid
    griddedpolicy = Matrix{TF}(undef, n_a, n_e)
    for i in 1:n_e
        knots = impliedstate[:, i]
        vals  = policy_a[:, i]
        linpolate = extrapolate(
            interpolate((knots,), vals, Gridded(Linear())),
            Flat())
        griddedpolicy[:, i] = linpolate.(policy_a[:, i])
    end

    # Step 4: enforce borrowing constraint
    griddedpolicy = max.(griddedpolicy, borrow_cons)

    # Step 5: consumption and marginal value on the exogenous grid
    c_grid        = (1 + r) .* policy_a .+ (w .* labor_mat) .- griddedpolicy
    value_current = (1 + r) .* (c_grid .^ (-γ))

    return (Value = value_current, KD = griddedpolicy)
end
