# Description: This file contains the functions that implement the backward iteration
# algorithm. The algorithm starts from the steady state at time period T and iterates
# backwards to time period 1. The function returns the sequence of sparse transition
# matrices which will be used in determining the evolution of the distribution in the
# Forward Iteration algorithm.


"""
    backward_capital(xVals, current_policy, model::SequenceModel)

User-provided backward function for the KS model's capital aggregation.
Performs one EGM step to obtain the savings policy at a given time period.
This is a thin wrapper around BackwardStep.
"""
backward_capital(xVals, current_policy, model::SequenceModel) = BackwardStep(xVals, current_policy, model)


"""
    BackwardIteration(xVec, model::SequenceModel, end_ss::SteadyState)

Performs the Backward Iteration algorithm for all aggregated variables.
For each aggregated variable in `model.agg_vars`, calls its backward function
to produce a sequence of T-1 disaggregated policy matrices (iterating from T back to 1).

Returns a NamedTuple mapping each aggregated variable name to its policy sequence
(a Vector of T-1 matrices), e.g., `(KD = [mat1, mat2, ..., mat_{T-1}],)`.
"""
function BackwardIteration(xVec, # (n_v x T-1) vector of variable values
    model::SequenceModel,
    end_ss::SteadyState) # has to be the ending steady state (i.e., at time period T)

    # Reorganize main vector
    T = model.Params.T
    TF = eltype(xVec)
    xMat = transpose(reshape(xVec, (model.Params.n_v, T-1))) # make it `T-1 x n_v` matrix

    # For each aggregated variable, run its backward function over T-1 periods
    policy_seqs = map(model.agg_vars) do spec
        # Initialize with terminal steady state policy
        # TODO: use spec-specific steady state policy once SteadyState stores multiple policies
        policies = fill(Matrix{TF}(undef, size(end_ss.ssPolicies)), T)
        policies[T] = end_ss.ssPolicies

        # Iterate backwards from T-1 to 1
        for i in 1:T-1
            policies[T-i] = spec.backward(xMat[T-i,:], policies[T+1-i], model)
        end

        return policies[1:T-1]
    end

    return policy_seqs
end


"""
    ValueFunction(currentpolicy, namedvars, model)

Computes the implied state (endogenous grid) from the current policy function
using the Euler equation. Maps `currentpolicy` → `impliedstate`.
Note: this is model-specific (Krusell-Smith).
"""
function ValueFunction(currentpolicy, namedvars, model::SequenceModel)
    @unpack policygrid, shockmat, Π, policymat = model
    @unpack β, γ = model.Params
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
    BackwardStep(x::Vector{Float64},
    currentpolicy::Matrix{Float64},
    model::SequenceModel)

Performs one step of the Endogenous Gridpoint method (Carroll 2006).
Note that this step is model specific, and the model specific elements are defined in 
the ValueFunction() function.
The function takes the current savings policy function and calculates the
policy function of HHs on the savings grid.
"""
function BackwardStep(xVals, # (n_v x 1) vector of endogenous variable values
    currentpolicy, # current policy function guess
    model::SequenceModel)

    # Unpack objects
    @unpack policymat = model
    namedvars = NamedTuple{model.varXs}(xVals)

    impliedstate = ValueFunction(currentpolicy, namedvars, model)
    TF = eltype(currentpolicy)
    griddedpolicy = Matrix{TF}(undef, size(policymat))

    for i in 1:model.Params.n_e
        linpolate = extrapolate(interpolate((impliedstate[:,i],), policymat[:,i], Gridded(Linear())), Flat())
        griddedpolicy[:,i] = linpolate.(policymat[:,i])
    end

    return griddedpolicy
end


"""
    BackwardSteadyState(sv::DataFrame,
    model::SequenceModel)

Applies the Endogenous Gridpoint method to find the steady state policies of 
the households.
In essence, performs iteration on `BackwardStep()` until convergence is reached.
"""
function BackwardSteadyState(varNs, # supports both Float64 and dual number vectors
    model::SequenceModel) # has to be the ending steady state (i.e., at time period T)

    # Unpack parameters
    @unpack n_a, n_e, ε, T = model.Params
    newguess = guess = zeros(n_a, n_e)
    tol = 1.0

    # Perform backward iteration till convergence
    while ε < tol
        guess = newguess
        newguess = BackwardStep(varNs, guess, model)
        tol = norm(newguess - guess)
    end

    return newguess
end


# Testing function

function FD_LI()
    x = rand(10)
    sort!(x)
    y = x .+ 1.0
    
    function inter(a)
        res = zeros(size(a))        
        linpolate = extrapolate(interpolate((x,), y, Gridded(Linear())), Flat())
        res = linpolate.(a)
        return res
    end
        
    z = x .+ 0.3
    J = jacobian(inter, z)

    return J
end
