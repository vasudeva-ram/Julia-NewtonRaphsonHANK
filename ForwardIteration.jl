
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
function invariant_dist(Π::Union{Matrix{Float64}, Adjoint{Float64, SparseMatrixCSC{Float64, Int64}}})

    ΠT = Π' # transpose

    # https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8
    D = [1; (I - ΠT[2:end, 2:end]) \ Vector(ΠT[2:end,1])]

    return D ./ sum(D) # return normalized to sum to 1.0
end


"""
    ForwardIteration(a_seq, # sequence of T-savings policy functions
    model::SequenceModel,
    ss::SteadyState)

Takes the optimal steady-state policy functions of the households and calculates the
sequence of aggregate capital demand values. There are two steps involved here:
(1) the function fD: A -> D takes the sequence of policy function and obtains the 
    sequence of distributions, and
(2) the function fKD: D -> KD takes the sequence of distributions and obtains the 
    sequence of aggregate capital demand values.
"""
function ForwardIteration(a_seq, # sequence of T-1 savings policy functions
    model::SequenceModel,
    ss::SteadyState)
    
    # setting up the Distributions vector
    @unpack T, n_a, n_e = model.CompParams
    Tv = n_a * n_e
    D = ss.ssD # initial distribution is the starting steady state distribution
    
    # Perform forward iteration and construct KD functionally
    KD = map(1:T-1) do i
        a = a_seq[(i-1)*Tv + 1:i*Tv]
        Λ = DistributionTransition(a, model)
        D = Λ * D
        return dot(a, D)
    end

    return KD
end


"""
    DistributionTransition(policy, # savings policy function
    model::SequenceModel)

Implements the Young (2010) method for constructing the transition matrix Λ.
    Takes a policy function and constructs the transition matrix Λ
    by composing it with the exogenous transition matrix Π.
"""
function DistributionTransition2(policy, # savings policy function
    model::SequenceModel)
    
    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.CompParams

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


function DistributionTransition(policy, model::SequenceModel)
    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.CompParams

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