# Description: This file contains the functions that implement the backward iteration
# algorithm. The algorithm starts from the steady state at time period T and iterates
# backwards to time period 1. The function returns the sequence of sparse transition
# matrices which will be used in determining the evolution of the distribution in the
# Forward Iteration algorithm.


"""
    BackwardIteration(xVec, model::SequenceModel, end_ss::SteadyState)

Performs the Backward Iteration algorithm for all heterogeneous variables.
For each `:heterogeneous` variable in `model.variables`, calls its `backward_fn`
to produce a sequence of T-1 disaggregated policy matrices (iterating from T back to 1).

Returns a NamedTuple mapping each heterogeneous variable name to its policy sequence
(a Vector of T-1 matrices), e.g., `(KD = [mat1, mat2, ..., mat_{T-1}],)`.
"""
function BackwardIteration(xVec, # (n_v x T-1) vector of variable values
    model::SequenceModel,
    end_ss::SteadyState) # has to be the ending steady state (i.e., at time period T)

    # Reorganize main vector
    T = model.compspec.T
    TF = eltype(xVec)
    xMat = transpose(reshape(xVec, (model.compspec.n_v, T-1))) # make it `T-1 x n_v` matrix

    # For each heterogeneous variable, run its backward_fn over T-1 periods
    agg_keys = vars_of_type(model, :heterogeneous)
    seqs = map(agg_keys) do varname
        var = model.variables[varname]
        ss_policy = end_ss.policies[varname]

        # Initialize with terminal steady state policy
        policies = fill(Matrix{TF}(undef, size(ss_policy)), T)
        policies[T] = ss_policy

        # Iterate backwards from T-1 to 1
        for i in 1:T-1
            policies[T-i] = var.backward_fn(xMat[T-i,:], policies[T+1-i], model)
        end

        return policies[1:T-1]
    end
    policy_seqs = NamedTuple{agg_keys}(seqs)

    return policy_seqs
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
