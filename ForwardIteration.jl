# Description: This file contains the functions that are used to perform the forward iteration
# algorithm. The algorithm starts from the optimal policy functions of the households and
# calculates the sequence of aggregated variable values, using the evolution of the
# distribution.


"""
    agg_capital(policy_seq, model::SequenceModel, ss::SteadyState)

User-provided forward/aggregation function for the KS model's capital demand.
Takes a sequence of T-1 savings policy matrices, evolves the distribution forward
using `DistributionTransition`, and computes aggregate capital demand at each period
via `dot(policy, distribution)`.

Returns a Vector of length T-1 with aggregated capital demand values.
"""
function agg_capital(policy_seq, model::SequenceModel, ss::SteadyState)
    D = ss.ssD # initial distribution is the *starting* steady state distribution

    KD = map(1:length(policy_seq)) do i
        a = vec(policy_seq[i])
        Λ = DistributionTransition(a, model)
        D = Λ * D
        return dot(a, D)
    end

    return KD
end


"""
    ForwardIteration(xVec, policy_seqs::NamedTuple, model::SequenceModel, ss::SteadyState)

Takes the variable vector `xVec` and the NamedTuple of policy sequences from
`BackwardIteration`, and assembles the full `n_v × (T-1)` matrix needed by `Residuals`.

For each aggregated variable in `model.agg_vars`, calls its forward function
to compute the aggregated time series and overwrites the corresponding row in `xMat`.
Non-aggregated variable rows are left as-is from `xVec`.

Returns the completed `n_v × (T-1)` matrix ready for `Residuals(xMat, model)`.
"""
function ForwardIteration(xVec, # (n_v x T-1) vector of all variable values
    policy_seqs::NamedTuple, # output of BackwardIteration
    model::SequenceModel,
    ss::SteadyState) # has to be the starting steady state

    @unpack T, n_v = model.Params
    xMat = reshape(copy(xVec), (n_v, T-1))

    # For each aggregated variable, compute its values and place in the correct row
    for (varname, spec) in pairs(model.agg_vars)
        idx = findfirst(==(varname), model.varXs)
        agg_values = spec.forward(policy_seqs[varname], model, ss)
        xMat[idx, :] .= agg_values
    end

    return xMat
end


"""
    DistributionTransition(policy, # savings policy function
    model::SequenceModel)

Implements the Young (2010) method for constructing the transition matrix Λ.
    Takes a policy function and constructs the transition matrix Λ
    by composing it with the exogenous transition matrix Π.
"""
function DistributionTransition1(policy, # savings policy function
    model::SequenceModel)
    
    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.Params

    n_m = n_a * n_e
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = eltype(policy)[]

    for col in eachindex(policy)
        m = findfirst(x->x>=policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, n_e))
            append!(Vs, 1.0 .* Π[j,:])
        else
            append!(Is, (m-1) .+ Jbases)
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, 2*n_e))
            w = (policy[col] - policygrid[m-1]) / (policygrid[m] - policygrid[m-1])
            append!(Vs, (1.0 - w) .* Π[j,:])
            append!(Vs, w .* Π[j,:])
        end
    end

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end


"""
    DistributionTransition(policy, 
        model::SequenceModel)

Implements the Young (2010) method for constructing the transition matrix Λ.
    Takes a policy function and constructs the transition matrix Λ
    by composing it with the exogenous transition matrix Π.
NOTE: Implements it in a way that avoids array mutation to accommodate
    Zygote differentiation.
"""
function DistributionTransition2(policy, 
    model::SequenceModel)

    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.Params

    n_m = n_a * n_e
    Jbases = [(ne - 1) * n_a for ne in 1:n_e]

    Is = Int64[]
    Js = Int64[]
    Vs = eltype(policy)[]

    for col in eachindex(policy)
        m = findfirst(x -> x >= policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            Is = vcat(Is, m .+ Jbases)
            Js = vcat(Js, fill(col, n_e))
            Vs = vcat(Vs, 1.0 .* Π[j, :])
        else
            Is = vcat(Is, (m - 1) .+ Jbases, m .+ Jbases)
            Js = vcat(Js, fill(col, 2 * n_e))
            w = (policy[col] - policygrid[m - 1]) / (policygrid[m] - policygrid[m - 1])
            Vs = vcat(Vs, (1.0 - w) .* Π[j, :], w .* Π[j, :])
        end
    end

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end


"""
    invariant_dist(Π::AbstractMatrix;
    method::Int64 = 1,
    ε::Float64 = 1e-9,
    itermax::Int64 = 50000,
    initVector::Union{Nothing, Vector{Float64}}=nothing,
    verbose::Bool = false
    )

Calculates the invariant distribution of a Markov chain with transition matrix Π.
"""
function invariant_dist(Π)

    ΠT = Π' # transpose

    # https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8
    D = [1; (I - ΠT[2:end, 2:end]) \ Vector(ΠT[2:end,1])]

    return D ./ sum(D) # return normalized to sum to 1.0
end

DistributionTransition = DistributionTransition2