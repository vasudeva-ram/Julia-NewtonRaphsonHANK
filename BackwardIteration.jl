
"""
    BackwardIteration(sv::NamedTuple,
    model::SequenceModel,
    end_ss::SteadyState)

Performs one full iteration of the Backward Iteration algorithm. The algorithm
starts from the steady state at time period T and iterates backwards to time
period 1. The function returns the sequence of sparse transition matrices
which will be used in determining the evolution of the distribution in the Forward
Iteration algorithm.
"""
function BackwardIteration(xVec::AbstractVector, # n_v x (T-1) vector of variable values
    model::SequenceModel,
    end_ss::SteadyState) # has to be the ending steady state (i.e., at time period T)

    # Reorganize main vector
    T = model.CompParams.T
    xMat = transpose(reshape(xVec, (model.CompParams.n_v, T-1))) # make it (T-1) x n_v matrix
    
    # Initialize savings vector
    a_seq = fill(Matrix{Float64}(undef, size(end_ss.ssPolicies)), T)
    a_seq[T] = end_ss.ssPolicies

    # Perform backward Iteration
    for i in 1:T-1
        a_seq[T-i] = BackwardStep(xMat[T-i,:], a_seq[T+1-i], model)
    end
    
    return a_seq
end


"""
    BackwardStep(x::Vector{Float64},
    currentpolicy::Matrix{Float64},
    model::SequenceModel)

Performs one step of the Endogenous Gridpoint method (Carroll 2006). 
Note that this step needs to be model specific. This is just the implementation 
for the Krussell-Smith model. 
The function takes the current savings policy function and calculates the
"""
function BackwardStep(xVals::AbstractVector, # (n_v x 1) vector of endogenous variable values
    currentpolicy::Matrix{Float64}, # current policy function guess 
    model::SequenceModel)

    # Unpack objects
    @unpack policygrid, shockmat, Π, policymat = model
    @unpack β, γ = model.ModParams
    namedvars = NamedTuple{model.varXs}(xVals)
    @unpack r, w = namedvars

    # Calculate the consumption matrix
    cprimemat = ((1 + r) .* policymat) + (w .* shockmat) - currentpolicy
    exponent = -1 * γ
    eulerlhs = β * (1 + r) * ((cprimemat .^ exponent) * Π')
    cmat = eulerlhs .^ (1 / exponent)

    # Interpolate the policy function to the grid
    impliedstate = (1 / (1 + r)) * (cmat - (w .* shockmat) + policymat)
    griddedpolicy = Matrix{Float64}(undef, size(policymat))

    for i in 1:model.CompParams.n_e
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
function BackwardSteadyState(varNs::Vector{Float64}, #TODO: annotate to support dual numbers and float64 vectors
    model::SequenceModel) # has to be the ending steady state (i.e., at time period T)

    # Unpack parameters
    @unpack n_a, n_e, ε, T = model.CompParams
    newguess = guess = zeros(n_a, n_e)
    tol = 1.0

    # Perform backward iIteration till convergence
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
