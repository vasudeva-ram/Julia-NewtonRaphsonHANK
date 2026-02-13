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
    ε = model.Params.ε
    
    # find steady state solution (trust region method)
    sol = nlsolve(Fx, x̅)
    x = sol.zero

    # Build the steady state policies and distribution
    ssVars = NamedTuple{model.varXs}(x)
    policies = BackwardSteadyState(x, model)
    Λss = DistributionTransition(policies, model)
    dist = invariant_dist(Λss')

    return SteadyState(ssVars, policies, Λss, dist)
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
    sig = 0.5 * sqrt(1 - (0.966^2))
    params = ModelParams(0.98, 1.0, sig, 0.966, 0.025, 0.11,
                         0.0001, [0.0, 200.0], 200, 7, length(varXs), 150, 1e-9)
    residuals_fn = compile_residuals(collect(equations), varXs)
    policygrid = make_DoubleExponentialGrid(params.gridx[1], params.gridx[2], params.n_a)
    Π, _, shockgrid = get_RouwenhorstDiscretization(params.n_e, params.ρ, params.σ)
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)

    mod = SequenceModel(varXs, equations, params, residuals_fn, policygrid, shockmat, policymat, Π)

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
    @unpack T, n_v = model.Params
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1)
    Zexog = ones(T-1)
    
    a_seq = BackwardIteration(xVec, model, ss)
    KD = ForwardIteration(a_seq, model, ss)
    zVals = Residuals(xVec, KD, Zexog, model)

    return zVals
end


function backFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
    a_seq = BackwardIteration(x_Vec, mod, stst)
    return a_seq 
end


function directJVPJacobian(mod, 
    stst)

    @unpack T, n_v = mod.Params
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.ssVars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, mod, stst)
        KD = ForwardIteration(a_seq, mod, stst)
        zVals = Residuals(x_Vec, KD, Zexog, mod)
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

    @unpack T, n_v = mod.Params
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.ssVars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, mod, stst)
        KD = ForwardIteration(a_seq, mod, stst)
        zVals = Residuals(x_Vec, KD, Zexog, mod)
        return zVals
    end
    
    fullX = fullFunction(xVec)

    for i in 1:n_v
        xDiff = xVec + (1e-4 * idmat[:,i])
        dirJacobian[:,i] = fullFunction(xDiff) - fullX
    end
    
    return dirJacobian ./ 1e-4
end





