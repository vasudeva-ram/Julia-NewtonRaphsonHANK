
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
function BackwardIteration(sv::DataFrame,
    model::SequenceModel,
    end_ss::SteadyState) # has to be the ending steady state (i.e., at time period T)

    # Unpack parameters
    T = model.CompParams.T
    
    # Initialize savings vector
    y_seq = fill(Matrix{Any}(undef, size(end_ss.policies.saving)), T)
    y_seq[T] = end_ss.policies.saving

    # Perform backward Iteration
    for i in 1:T-1
        y_seq[T-i] = BackwardStep(sv[T-i], y_seq[T+1-i], model)
    end
    
    return y_seq
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
function BackwardStep(varN, #TODO: annotate to support dual numbers and float64 vectors
    currentpolicy, # current policy function guess 
    model::SequenceModel)

    # Unpack objects
    @unpack policygrid, shockmat, Π, policymat = model
    @unpack β, γ = model.ModParams
    namedvars = NamedTuple{model.varNs}(varN)
    @unpack r, w = namedvars

    # Calculate the consumption matrix
    cprimemat = ((1 + r) .* policymat) + (w .* shockmat) - currentpolicy
    exponent = -1 * γ
    eulerlhs = β * (1 + r) * ((cprimemat .^ exponent) * Π')
    cmat = eulerlhs .^ (1 / exponent)

    # Interpolate the policy function to the grid
    impliedstate = (1 / (1 + r)) * (cmat - (w .* shockmat) + policymat)
    griddedpolicy = zeros(size(impliedstate))

    for i in axes(impliedstate, 2)
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
function BackwardSteadyState(varNs, #TODO: annotate to support dual numbers and float64 vectors
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