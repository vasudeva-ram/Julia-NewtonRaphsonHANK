

"""
    ResidualsSteadyState(x::Vector{Float64},
    a::Matrix{Float64},
    D::Vector{Float64},
    model::SequenceModel)

Returns the residuals in the steady state, given variables
values, savings policies and a distribution.
"""
function ResidualsSteadyState(varN, # endogenous variable values
    varX, # exogenous variable values
    a, # matrix of savings policies
    D, # steady state distribution
    model::SequenceModel)
    
    namedNvars = NamedTuple{model.varNs}(varN)
    namedXvars = NamedTuple{model.varXs}(varX)
    @unpack Y, KS, r, w = namedNvars
    @unpack Z = namedXvars
    
    # Initialize vectors
    residuals = zeros(length(varN))
    
    # Calculate aggregated variables
    a = vcat(a...)
    KD = a' * D
    
    # Unpack parameters
    @unpack α, δ = model.ModParams
    
    # Calculate residuals
    residuals = [
        Y .- (Z .* (KS.^α)),
        r .+ δ .- (α .* Z .* (KS.^(α-1))),
        w .- ((1-α) .* Z .* (KS.^α)),
        KS .- KD
        ]
    
    return residuals
end



"""
    Residuals(x::Vector{Float64}, 
    model::SequenceModel)

Returns the residuals in the dynamics, given variable
values for the entire sequence of T periods.
"""
function Residuals(x::Vector{Float64}, 
    model::SequenceModel)

    # Unpack parameters
    @unpack δ, α = model.params
    @unpack Y, KS, r, w = sv
    
    # generate lagged variables
    KS_lag = KS
    KS_lag[2:end] = KS[1:end-1]

    # obtain aggregate capital demand
    KD = get_KDemand(x, model)

    residuals = [
        Y .- KS_lag.^α,
        r .+ δ .- (α .* (KS_lag.^(α-1))),
        w .- ((1-α) .* (KS_lag.^α)),
        KS .- KD
        ]
    
    return residuals
end

