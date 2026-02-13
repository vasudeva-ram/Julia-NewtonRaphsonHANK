# Description: This file contains the functions to obtain the Jacobian of the steady state
# of the model. The Jacobian is used in the solution of the model, and is calculated by
# taking the derivative of the residuals with respect to the endogenous variables.
# Specifically, this implements the methodology described in Boehl(2021)


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
    getDirectJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

Obtains the first element in the jacobian calculation: 
    ∂zᵢ/∂xⱼ = ∂zᵢ/∂xⱼ + ∑ₖ ∂fᵢ/∂aₖ * ∂aₖ/∂xⱼ
    where zᵢ is the residual, xⱼ is the j-th endogenous variable value, and aₖ is the policy function.
"""
function getDirectJacobian(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.Params
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1)
    idmat = sparse(1.0I, n, n)
    exogZ = ones(T-1)
    JDI = zeros(n, n_v)

    # Obtain steady state a_vec
    a_vec = repeat(vec(ss.ssPolicies), T-1)

    # Define backward and forward functions
    function fullFunc(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        KD = ForwardIteration(a_vec, model, ss)
        zVals = Residuals(x_Vec, KD, exogZ, model)
        return zVals
    end

    k = n - (2*n_v)
    # Obtain Jacobians using JVPs and VJPs
    for i in 1:n_v
        JDI[:,i] = JVP(fullFunc, xVec, idmat[:, k+i]) # taking the derivative of X_{T-2}   
    end

    return (dFbydXlead = JDI[k-n_v+1:k,:], dFbydX = JDI[k+1:k+n_v,:], dFbydXlag = JDI[k+n_v+1:end,:])
end


"""
    getIntdJacobians(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

TBW
"""
function getIntdJacobians(ss::SteadyState, # should be the ending steady state (at time T)
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.Params
    n = (T-1) * n_v
    nJ = n_a * n_e * (T-1)
    
    # Initialize vectors
    xVec = repeat([values(ss.ssVars)...], T-1) # (n_v * T-1)-dimensional vector
    a_vec = repeat(vec(ss.ssPolicies), T-1)
    idmat = sparse(1.0I, n, n)
    exogZ = ones(T-1) #TODO: exogZ should be part of the SteadyState object
    JBI = spzeros(nJ, n_v)
    JFI = spzeros(n_v, nJ)

    # Define backward and forward functions
    function backFunc(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, model, ss)
        return a_seq 
    end

    function forwardFunc(aseq) # (n_a x n_e x T-1) x 1 vector
        KD = ForwardIteration(aseq, model, ss)
        zVals = Residuals(xVec, KD, exogZ, model)
        # zVals = ResidualsSteadyState(xVec, aseq, ss.ssD, model)
        return zVals
    end

    # Obtain Jacobians using JVPs and VJPs

    # apply forward mode differentiation
    for i in 1:n_v
        JBI[:,i] = JVP(backFunc, xVec, idmat[:, n - n_v + i])
    end

    # obtain the pullback function
    _, pullback = Zygote.pullback(forwardFunc, a_vec)
    
    # apply reverse mode differentiation
    for i in 1:n_v
        JFI[i,:] = sparse(pullback(idmat[:, n - n_v + i])[1])
    end
    
    return JBI, JFI
end


function getJacobianHelper(JBI, # jacobian of the backward iteration
    JFI, # jacobian of the forward iteration
    JDI, # the direct jacobian 
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v, n_a, n_e = model.Params
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
    @unpack T, n_v = model.Params

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
    @unpack T, n_v = model.Params
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

