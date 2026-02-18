# Description: This file contains the functions to obtain the steady state of the model.
# For right now, the steady state is obtained using the trust-region method.
#TODO: The Boehl (2021) method for the steady state to be implemented in the future.


"""
    get_SteadyState(model::SequenceModel;
    guess::Union{NamedTuple, nothing}=nothing)

Function to obtain the steady state using the trust-region method. 
"""
function get_SteadyState(model::SequenceModel;
    guess = nothing) # initial guess for the steady state values

    @unpack policygrid, Π = model
    
    # Define main function such that Fₓ : x → z
    function Fx(xVals::Vector{Float64}) #TODO: annotate to support dual numbers and float64 vectors
        a = BackwardSteadyState(xVals, model) # get steady state policies
        Λ = DistributionTransition(a, model) # get transition matrix
        D = invariant_dist(Λ') # get invariant distribution
        z = ResidualsSteadyState(xVals, a, D, model) # get residuals
        
        return z
    end
    
    # Initialize steady state guess
    x̅ = isnothing(guess) ? rand(length(model.varNs)) : collect(values(guess))
    tol = 1.0
    ε = model.compspec.ε
    
    # find steady state solution (trust region method)
    sol = nlsolve(Fx, x̅)
    x = sol.zero
    
    # Build the steady state policies and distribution
    vars = NamedTuple{model.varXs}(x)
    raw_policies = BackwardSteadyState(x, model)
    policies = NamedTuple{keys(model.agg_vars)}(ntuple(_ -> raw_policies, length(model.agg_vars)))
    Λss = DistributionTransition(raw_policies, model)
    dist = invariant_dist(Λss')

    return SteadyState(vars, policies, Λss, dist)
end


# Test Run

function test_SteadyState()
    varXs = (:Y, :KS, :r, :w, :Z)
    equations = (
        "Y = Z * KS(-1)^α",
        "r + δ = α * Z * KS(-1)^(α-1)",
        "w = (1-α) * Z * KS(-1)^α",
        "KS = KD",
    )

    # Computational specs
    compspec = ComputationalSpec(150, 1e-9, 0.0001, length(varXs))

    # Economic parameters as NamedTuple
    params = (β = 0.98, γ = 1.0, δ = 0.025, α = 0.11)

    # Compile equations
    param_names = Set(keys(params))
    union!(param_names, Set([:T, :ε, :dx, :n_v]))
    residuals_fn = compile_residuals(collect(equations), varXs, param_names)

    # Build heterogeneity dimensions
    wealth_config = Dict("type" => "endogenous", "grid_method" => "DoubleExponential",
                         "n" => 200, "bounds" => [0.0, 200.0])
    prod_config = Dict("type" => "exogenous", "discretization" => "Rouwenhorst",
                       "n" => 7, "ρ" => 0.966, "σ" => 0.283)
    heterogeneity = (wealth = build_dimension(wealth_config),
                     productivity = build_dimension(prod_config))

    agg_vars = (KD = (backward = backward_capital, forward = agg_capital),)
    mod = SequenceModel(varXs, equations, compspec, params, residuals_fn, agg_vars, heterogeneity)

    # Obtain steady state
    ss = get_SteadyState(mod, guess = (Y = 1.0, KS = 1.0, r = 0.02, w = 0.1, Z = 1.0))

    return mod, ss
end


"""
    SingleRun(ss::SteadyState,
    model::SequenceModel)

Runs the entire sequence of functions for a single run of the model.
    That is, it calculates the residuals for a given steady state and model.
"""
function SingleRun(ss::SteadyState,
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.compspec
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.vars)...], T-1)

    policy_seqs = BackwardIteration(xVec, model, ss)
    xMat = ForwardIteration(xVec, policy_seqs, model, ss)
    zVals = Residuals(xMat, model)

    return zVals
end


function backFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
    a_seq = BackwardIteration(x_Vec, mod, stst)
    return a_seq 
end


function directJVPJacobian(mod, 
    stst)

    @unpack T, n_v = mod.params
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.vars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        policy_seqs = BackwardIteration(x_Vec, mod, stst)
        xMat = ForwardIteration(x_Vec, policy_seqs, mod, stst)
        zVals = Residuals(xMat, mod)
        return zVals
    end

    for i in 1:n_v
        # dirJacobian[:,n - n_v + i] = JVP(fullFunction, xVec, idmat[:, n - n_v + i])
        dirJacobian[:,i] = JVP(fullFunction, xVec, idmat[:,i])
    end

    return dirJacobian
end


function directNumJacobian(mod, 
    stst)

    @unpack T, n_v = mod.params
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.vars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        policy_seqs = BackwardIteration(x_Vec, mod, stst)
        xMat = ForwardIteration(x_Vec, policy_seqs, mod, stst)
        zVals = Residuals(xMat, mod)
        return zVals
    end

    fullX = fullFunction(xVec)

    for i in 1:n_v
        xDiff = xVec + (1e-4 * idmat[:,i])
        dirJacobian[:,i] = fullFunction(xDiff) - fullX
    end

    return dirJacobian ./ 1e-4
end





