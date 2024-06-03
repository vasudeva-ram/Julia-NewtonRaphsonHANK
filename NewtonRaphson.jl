# Implementing Boehl (2024) ``HANK on Speed" methodology
include("KrussellSmith.jl")
using DataFrames, UnPack
import ForwardDiff: derivative, Zygote: pullback


function EquilibriumResiduals(sv::DataFrame, # has to be a (T-1) x n_v DataFrame
    model::AiyagariModel)

    # Unpack parameters
    @unpack δ, α = model.params
    
    # generate lagged variables
    KS_lag = sv.KS
    KS_lag[2:end] = sv.KS[1:end-1]

    # obtain aggregate capital demand
    KD = get_KDemand(sv, model)

    residuals = [
        sv.Y .- KS_lag.^α,
        sv.r .+ δ .- (α .* (KS_lag.^(α-1))),
        sv.w .- ((1-α) .* (KS_lag.^α)),
        sv.KS .- KD
        ]
    
    return residuals
end


function BackwardIteration(sv::DataFrame,
    model::AiyagariModel,
    end_ss::SteadyState) # has to be the steady state at time period T

    y_seq = get_SavingsSequence(sv, model, end_ss)
    Λ_seq = get_ΛSequence(y_seq, model, end_ss)

    return y_seq, Λ_seq
end


function get_SavingsSequence(sv::DataFrame, # has to be a (T-1) x n_v DataFrame
    model::AiyagariModel,
    end_ss::SteadyState) # has to be the steady state at time period T

    # Unpack parameters
    T = model.params.T
    
    # Create price tuples to perform EGM
    price_tuples = collect(zip(sv.r, sv.w))
    
    # Initialize savings vector
    y_seq = fill(Matrix{Float64}(undef, size(end_ss.policies.saving)), T)
    y_seq[T] = end_ss.policies.saving

    # Perform backward Iteration
    for i in 1:T-1
        prices = Prices(price_tuples[T-i]...)
        cmat = consumptiongrid(prices, 
                            model.policymat, 
                            model.shockmat, 
                            y_seq[T+1-i], 
                            model.Π, 
                            model.params)
        y_seq[T-i] = policyupdate(prices, 
                            model.policymat, 
                            model.shockmat, 
                            cmat)
    end
    
    return y_seq
end


function get_ΛSequence(y_seq::Vector{Matrix{Float64}},
    model::AiyagariModel,
    end_ss::SteadyState) # has to be the steady state at time period T

    # setting up the Λso vector
    T = model.params.T
    Λ_seq = Array{SparseMatrixCSC{Float64,Int64}}(undef, T)
    Λ_seq[T] = end_ss.Λ
    
    for i in 1:T-1
        Λ_seq[T-i] = distribution_transition(y_seq[T-i], 
                            model.policygrid, 
                            model.Π)
    end

    return Λ_seq
end


function ForwardIteration(Λ_seq::Vector{SparseMatrixCSC{Float64,Int64}},
    D_ss::Vector{Float64}) # has to be the starting (period 1) steady state distribution

    # setting up the Distributions vector
    T = length(Λ_seq)
    D_seq = Array{Vector{Float64}}(undef, T)
    D_seq[1] = D_ss

    # Perform forward iteration
    for i in 2:T
        D_seq[i] = Λ_seq[i-1] * D_seq[i-1] #TODO: check if tranposing is correct
    end
    
    return D_seq
end


function JVP(func::Function, 
    primal::Vector{Float64}, 
    tangent::Vector{Float64})

    g(t) = func(primal + t*tangent)
    return ForwardDiff.derivative(g, 0.0)
end


function VJP(func::Function, 
    primal::Vector{Float64}, 
    cotangent::Vector{Float64})

    _, func_back = pull_back(func, primal)
    vjp_result, = func_back(cotangent)

    return vjp_result
end