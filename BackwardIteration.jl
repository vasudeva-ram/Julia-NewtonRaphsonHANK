
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
    T = model.ComputationalParams.T
    
    # Initialize savings vector
    y_seq = fill(Matrix{Float64}(undef, size(end_ss.policies.saving)), T)
    y_seq[T] = end_ss.policies.saving

    # Perform backward Iteration
    for i in 1:T-1
        y_seq[T-i] = BackwardStep(sv[T-i], y_seq[T+1-i], model)
    end
    
    return y_seq
end


"""
    BackwardStep(sv::NamedTuple,
    currentpolicy::Matrix{Float64},
    model::SequenceModel)

Performs one step of the Endogenous Gridpoint method (Carroll 2006). 
Note that this step needs to be model specific. This is just the implementation 
for the Krussell-Smith model. 
The function takes the current savings policy function and calculates the
"""
function BackwardStep(sv::DataFrame,
    currentpolicy::Matrix{Float64},
    model::SequenceModel)

    # Unpack objects
    @unpack policygrid, shockmat, Π, policymat = model
    @unpack β, γ = model.params
    r = sv.r
    w = sv.w

    # Calculate the consumption matrix
    cprimemat = ((1 + r) .* policymat) + (w .* shockmat) - currentpolicy
    eulerlhs = β * (1 + r) * ((cprimemat .^ (-γ)) * Π')
    cmat = eulerlhs .^ (-1 / γ)

    # Interpolate the policy function to the grid
    impliedstate = (1 / (1 + r)) * (cmat - (w .* shockmat) + policymat)
    griddedpolicy = Matrix{Float64}(undef, size(impliedstate))

    for i in axes(impliedstate, 2)
        linpolate = Interpolations.linear_interpolation(impliedstate[:,i], 
                    policymat[:,i], extrapolation_bc = Interpolations.Flat())
                    griddedpolicy[:,i] = linpolate.(policymat[:,i])
    end

    return griddedpolicy
end



