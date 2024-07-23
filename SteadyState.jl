include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")


function getDirectJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1)
    idmat = sparse(1.0I, n, n)
    Zexog = ones(T-1)
    JDI = zeros(n, n_v)

    # Obtain steady state a_vec
    a_vec = repeat(vec(ss.ssPolicies), T-1)

    # Define backward and forward functions
    function fullFunc(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        KD = ForwardIteration(a_vec, model, ss)
        zVals = Residuals(x_Vec, KD, Zexog, model)
        return zVals
    end

    k = n - (2*n_v)
    # Obtain Jacobians using JVPs and VJPs
    for i in 1:n_v
        JDI[:,i] = JVP(fullFunc, xVec, idmat[:, k+i]) # taking the derivative of X_{T-2}   
    end

    return (dFbydXlead = JDI[k-n_v+1:k,:], dFbydX = JDI[k+1:k+n_v,:], dFbydXlag = JDI[k+n_v+1:end,:])
end


function getIntdJacobians(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.CompParams
    n = (T-1) * n_v
    nJ = n_a * n_e * (T-1)
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1) # (n_v * T-1)-dimensional vector
    a_vec = repeat(vec(ss.ssPolicies), T-1)
    idmat = sparse(1.0I, n, n)
    Zexog = ones(T-1)
    JBI = spzeros(nJ, n_v)
    JFI = spzeros(n_v, nJ)

    # Define backward and forward functions
    function backFunc(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, model, ss)
        return a_seq 
    end

    function forwardFunc(aseq) # (n_a x n_e x T-1) x 1 vector
        KD = ForwardIteration(aseq, model, ss)
        zVals = Residuals(xVec, KD, Zexog, model)
        return zVals
    end

    # Obtain Jacobians using JVPs and VJPs
    for i in 1:n_v
        JBI[:,i] = JVP(backFunc, xVec, idmat[:, n - n_v + i])
    end

    for i in 1:n_v
        JFI[i,:] = VJP(forwardFunc, a_vec, idmat[:, n - n_v + i])
    end
    
    return JBI, JFI
end


function getJacobianHelper(JBI, # jacobian of the backward iteration
    JFI, # jacobian of the forward iteration
    JDI, # the direct jacobian 
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.CompParams
    n_r = n_a * n_e

    # Obtain helper base 
    JacobianHelper = [sparse(zeros(n_v, n_v)) for _ in 1:T-1, _ in 1:T-1] # Initialize
    for t in 1:T-1
        for s in 1:T-1
            JacobianHelper[t,s] = JFI[:,(t-1)*n_r + 1:t*n_r] * JBI[(s-1)*n_r + 1:s*n_r,:]
        end
    end

    # Add the direct Jacobian
    @unpack dFbydX, dFbydXlag, dFbydXlead = JDI
    JacobianHelper[T-1, T-1] += dFbydX
    JacobianHelper[T-2, T-1] += dFbydXlag
    JacobianHelper[T-1, T-2] += dFbydXlead

    return JacobianHelper
end


"""
    getFinalJacobian(JacobianHelper, # helper Jacobian
    JDI, # direct Jacobian
    model::SequenceModel)

Applies the recursion over the block components of the
helper matrix to obtain the final Jacobian.
#TODO: do I need to add the dfbydxlead to the first block?
"""
function getFinalJacobian(JacobianHelper, # helper Jacobian
    JDI, # direct Jacobian
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams

    # Obtain the final Jacobian
    J̅ = [sparse(zeros(n_v, n_v)) for _ in 1:T-1, _ in 1:T-1] # Initialize
    
    for t in 1:T-1
        for s in 1:T-1
            if s<2 || t<2
                J̅[s,t] = sparse(zeros(n_v, n_v)) + JacobianHelper[T-s,T-t]
            else
                J̅[s,t] = J̅[s-1,t-1] + JacobianHelper[T-s,T-t]
            end
        end
    end

    # adjust the J̅[1,1] entry #TODO: does dfbydxlead need to be added also?
    J̅[1,1] = J̅[1,1] + JDI.dFbydXlag

    return J̅
end


"""
    getConsolidatedJacobian(J̅, # final Jacobian
    model::SequenceModel)

Given a jacobian that is a matrix of matrices, consolidates
the Jacobian into a single matrix.
"""
function getConsolidatedJacobian(J̅, # final Jacobian
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = (T-1) * n_v

    # Consolidate the Jacobian
    J̄ = spzeros(n, n)
    for t in 1:T-1
        for s in 1:T-1
            J̄[(s-1)*n_v + 1:s*n_v, (t-1)*n_v + 1:t*n_v] = J̅[s,t]
        end
    end

    return J̄
    
end


"""
    getSteadyStateJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

Single function to obtain all the intemediate Jacobians needed for the 
    jacobian of the steady state, and then consolidates them into the final Jacobian.
"""
function getSteadyStateJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Obtain Jacobians
    JDI = getDirectJacobian(ss, model)
    JBI, JFI = getIntdJacobians(ss, model)

    # Obtain helper Jacobian
    JacobianHelper = getJacobianHelper(JBI, JFI, JDI, model)
    matrixJacobian = getFinalJacobian(JacobianHelper, JDI, model)
    J̅ = getConsolidatedJacobian(matrixJacobian, model)

    return J̅
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


"""
    SingleRun(ss::SteadyState,
    model::SequenceModel)

Runs the entire sequence of functions for a single run of the model.
    That is, it calculates the residuals for a given steady state and model.
"""
function SingleRun(ss::SteadyState,
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1)
    Zexog = ones(T-1)
    
    a_seq = BackwardIteration(xVec, model, ss)
    KD = ForwardIteration(a_seq, model, ss)
    zVals = Residuals(xVec, KD, Zexog, model)

    return zVals
end


# Some testing functions
mod, stst = test_SteadyState();
zVals = SingleRun(stst, mod);
jbi, jfi = getIntdJacobians(stst, mod);
jdi = getDirectJacobian(stst, mod);
jhelper = getJacobianHelper(jbi, jfi, jdi, mod);
jfinal = getFinalJacobian(jhelper, mod);


function backFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
    a_seq = BackwardIteration(x_Vec, mod, stst)
    return a_seq 
end


function directJVP(mod, 
    stst)

    @unpack T, n_v = mod.CompParams
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.ssVars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n_v)

    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, mod, stst)
        KD = ForwardIteration(a_seq, mod, stst)
        zVals = Residuals(x_Vec, KD, Zexog, mod)
        return zVals
    end
    
    for i in 1:n_v
        # dirJacobian[:,i] = JVP(fullFunction, xVec, idmat[:, n - n_v + i])
        dirJacobian[:,i] = JVP(fullFunction, xVec, idmat[:,i])
    end
    
    return dirJacobian
end