include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")


function getDirectJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = T * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T)
    idmat = sparse(1.0I, n, n)
    Zexog = ones(T)
    JDI = zeros(n, n_v)

    # Define backward and forward functions
    function fullFunc(xVec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(xVec, model, ss)
        KD = ForwardIteration(a_seq, model, ss)
        zVals = Residuals(xVec, KD, Zexog, model)
        return zVals
    end

    # Obtain Jacobians using JVPs and VJPs
    for i in 1:n_v
        JDI[:,i] = JVP(fullFunc, xVec, idmat[:, 745 - n_v + i])
    end

    
    return JDI
end


function getIntdJacobians(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.CompParams
    n = T * n_v
    nJ = n_a * n_e * T
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T) # (n_v * T)-dimensional vector
    a_vec = repeat(vec(ss.ssPolicies), T)
    idmat = sparse(1.0I, n, n)
    Zexog = ones(T)
    JBI = spzeros(nJ, n_v)
    JFI = spzeros(n_v, nJ)

    # Define backward and forward functions
    function backFunc(xVec::AbstractVector) # (n_v * T)-dimensional vector
        a_seq = BackwardIteration(xVec, model, ss)
        return a_seq 
    end

    function forwardFunc(aseq) # (n_a x n_e x T) x 1 vector
        KD = ForwardIteration(aseq, model, ss)
        zVals = Residuals(xVec, KD, Zexog, model)
        return zVals
    end

    # Obtain Jacobians using JVPs and VJPs
    idx = (T-1) * n_v # index for the (T-1)^{th} period
    for i in 1:n_v
        JBI[:,i] = JVP(backFunc, xVec, idmat[:, idx - n_v + i])
    end

    for i in 1:n_v
        JFI[i,:] = VJP(forwardFunc, a_vec, idmat[:, idx - n_v + i])
    end
    
    return JBI, JFI
end


function getFinalJacobian(JBI, # jacobian of the backward iteration
    JFI, # jacobian of the forward iteration
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.CompParams
    n_r = n_a * n_e

    # Final Jacobian
    JacobianFinal = [sparse(zeros(n_v, n_v)) for _ in 1:T, _ in 1:T] # Initialize
    for t in 1:T
        for s in 1:T
            JacobianFinal[t,s] = JFI[:,(t-1)*n_r + 1:t*n_r] * JBI[(s-1)*n_r + 1:s*n_r,:]
        end
    end

    return JacobianFinal
end


"""
    get_SteadyState(model::SequenceModel;
    guess::Union{NamedTuple, nothing}=nothing)

Function to obtain the steady state using the Newton-Raphson method. 
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
    ε = model.CompParams.ε
    
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
    sig = 0.5 * sqrt(1 - (0.966^2))
    modpars = ModelParams(0.98, 1.0, sig, 0.966, 0.025, 0.11)
    compars = ComputationalParams(0.0001, [0.0, 200.0], 200, 7, length(varXs), 150, 1e-9)
    policygrid = make_DoubleExponentialGrid(compars.gridx[1], compars.gridx[2], compars.n_a)
    Π, _, shockgrid = get_RouwenhorstDiscretization(compars.n_e, modpars.ρ, modpars.σ)
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)
        
    mod = SequenceModel(varXs, compars, modpars, policygrid, shockmat, policymat, Π)

    # Obtain steady state
    ss = get_SteadyState(mod, guess = (Y = 1.0, KS = 1.0, r = 0.02, w = 0.1, Z = 1.0))

    return mod, ss
end


function SingleRun(ss::SteadyState,
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = T * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T)
    Zexog = ones(T)
    
    a_seq = BackwardIteration(xVec, model, ss)
    KD = ForwardIteration(a_seq, model, ss)
    zVals = Residuals(xVec, KD, Zexog, model)

    return zVals
end


# Some testing functions
mod, stst = test_SteadyState();
zVals = SingleRun(stst, mod);
jbi, jfi = getIntdJacobians(stst, mod);
