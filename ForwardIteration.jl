
"""
    ForwardIteration(a_seq::Vector{Float64},
    model::SequenceModel,
    ss::SteadyState)

Performs one full iteration of the Forward Iteration algorithm. The algorithm
starts from the steady state at time period 1 and iterates forward to time
period T, calculating the transition matrices and returning the sequence of distributions.
"""
function ForwardIteration(a_seq::Vector{Matrix{TF}}, # sequence of T-savings policy functions
    model::SequenceModel,
    ss::SteadyState) where TF # has to be the starting (period 1) steady state distribution
    
    # setting up the Distributions vector
    T = model.CompParams.T
    KD = Vector{TF}(undef, T)
    
    # initial iteration
    D = ss.ssD
    KD[1] = dot(vec(a_seq[1]), D)

    # Perform forward iteration
    for i in 2:T
        Λ = DistributionTransition(a_seq[i-1], model.policygrid, model.Π)
        D = Λ * D
        a = vec(a_seq[i])
        KD[i] = dot(a, D)
    end

    return KD[1:end-1]
end


"""
    DistributionTransition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64})

Calculates the transition matrix implied by the exogenous shock process and 
the savings policy function following Young (2010).
"""
function DistributionTransition(savingspf::Matrix{TF}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::AbstractMatrix) where TF # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
    n_a, n_e = size(savingspf)
    n_m = n_a * n_e
    policy = vcat(savingspf...)
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = TF[]

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


# function DistributionTransition(savingspf::AbstractMatrix, # savings policy function
#     policygrid::AbstractVector, # savings grid
#     Π::AbstractMatrix) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
#     n_a, n_e = size(savingspf)
#     n_m = n_a * n_e
#     policy = vcat(savingspf...)
#     Jbases = [(ne -1)*n_a for ne in 1:n_e]
#     Is = Int64[]
#     Js = Int64[]
#     Vs = Float64[]

#     for col in eachindex(policy)
#         tempIs = Int64[]
#         tempJs = Int64[]
#         tempVs = Float64[]
#         m = findfirst(x->x>=policy[col], policygrid)
#         j = div(col - 1, n_a) + 1
#         if m == 1
#             tempIs = m .+ Jbases
#             tempJs = fill(col, n_e)
#             tempVs = 1.0 .* Π[j,:]
#         else
#             tempIs = vcat((m-1) .+ Jbases, m .+ Jbases)
#             tempJs = fill(col, 2*n_e)
#             w = (policy[col] - policygrid[m-1]) / (policygrid[m] - policygrid[m-1])
#             tempVs = vcat((1.0 - w) .* Π[j,:], w .* Π[j,:])
#         end
#         Is = vcat(Is, tempIs)
#         Js = vcat(Js, tempJs)
#         Vs = vcat(Vs, tempVs)
#     end

#     Λ = sparse(Is, Js, Vs, n_m, n_m)

#     return Λ
# end


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
