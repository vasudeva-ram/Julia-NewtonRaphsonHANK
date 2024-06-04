# General Structures

struct ModelParams{TF<:Float64}
    β::TF # discount factor
    γ::TF # coefficient of relative risk aversion
    σ::TF # standard deviation of the shock process
    ρ::TF # persistence of the shock process
    δ::TF # depreciation rate
    α::TF # share of capital in production
end


struct ComputationalParams{TF<:Float64, TI<:Int64}
    dx::TF # size of infinitesimal shock for numerical differentiation
    gridx::Vector{TF} # [a_min, a_max] bounds for the savings grid
    n_a::TI # number of grid points for the savings grid
    n_e::TI # number of grid points for the shock grid
    T::TI # number of periods for the transition path
end


function EquilibriumResiduals(sv::NamedTuple, 
    model::AiyagariModel)

    # Unpack parameters
    @unpack δ, α = model.params
    @unpack Y, KS, r, w = sv
    
    # generate lagged variables
    KS_lag = KS
    KS_lag[2:end] = KS[1:end-1]

    # obtain aggregate capital demand
    KD = get_KDemand(sv, model)

    residuals = [
        Y .- KS_lag.^α,
        r .+ δ .- (α .* (KS_lag.^(α-1))),
        w .- ((1-α) .* (KS_lag.^α)),
        KS .- KD
        ]
    
    return residuals
end



