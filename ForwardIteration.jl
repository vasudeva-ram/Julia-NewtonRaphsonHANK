
"""
    ForwardIteration(y_seq::Vector{Matrix{Float64}},
    model::SequenceModel,
    ss::SteadyState)

Performs one full iteration of the Forward Iteration algorithm. The algorithm
starts from the steady state at time period 1 and iterates forward to time
period T, calculating the transition matrices and returning the sequence of distributions.
"""
function ForwardIteration(y_seq::Vector{Matrix{Float64}},
    model::SequenceModel,
    ss::SteadyState) # has to be the starting (period 1) steady state distribution
    
    # setting up the Distributions vector
    T = length(y_seq)
    D_seq = Array{Vector{Float64}}(undef, T)
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
function DistributionTransition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64}) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
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

