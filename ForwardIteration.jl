
"""
    ForwardIteration(y_seq::Vector{Matrix{Float64}},
    model::SequenceModel,
    ss::SteadyState)

Performs one full iteration of the Forward Iteration algorithm. The algorithm
starts from the steady state at time period 1 and iterates forward to time
period T, calculating the transition matrices and returning the sequence of distributions.
"""
function ForwardIteration(y_seq, # sequence of savings policy functions
    model::SequenceModel,
    ss::SteadyState) # has to be the starting (period 1) steady state distribution
    
    # setting up the Distributions vector
    T = length(y_seq)
    D_seq = zeros(T)
    D_seq[1] = ss.D

    # Perform forward iteration
    for i in 2:T
        Λ = DistributionTransition(y_seq[i-1], model.policygrid, model.Π)
        D_seq[i] = Λ * D_seq[i-1] #TODO: check if tranposing is correct
    end
    
    return D_seq
end


"""
    DistributionTransition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64})

Calculates the transition matrix implied by the exogenous shock process and 
the savings policy function following Young (2010).
"""
function DistributionTransition(savingspf, # savings policy function
    policygrid, # savings grid
    Π) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
    n_a, n_e = size(savingspf)
    n_m = n_a * n_e
    policy = vcat(savingspf...)
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]

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
    invariant_dist(Π::AbstractMatrix;
    method::Int64 = 1,
    ε::Float64 = 1e-9,
    itermax::Int64 = 50000,
    initVector::Union{Nothing, Vector{Float64}}=nothing,
    verbose::Bool = false
    )

Calculates the invariant distribution of a Markov chain with transition matrix Π.
"""
function invariant_dist(Π::AbstractMatrix)

    ΠT = Π' # transpose

    # https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8
    D = [1; (I - ΠT[2:end, 2:end]) \ Vector(ΠT[2:end,1])]

    return D ./ sum(D) # return normalized to sum to 1.0
end
