

"""
    ResidualsSteadyState(x::Vector{Float64},
    a::Matrix{Float64},
    D::Vector{Float64},
    model::SequenceModel)

Returns the residuals in the steady state, given variables
values, savings policies and a distribution.
"""
function ResidualsSteadyState(xVals::AbstractVector, # vector of variable values
    a::AbstractMatrix, # matrix of savings policies
    D::AbstractVector, # steady state distribution
    model::SequenceModel)
    
    namedXvars = NamedTuple{model.varXs}(xVals)
    @unpack Y, KS, r, w, Z = namedXvars
    
    # Initialize vectors
    residuals = zeros(length(xVals))
    
    # Calculate aggregated variables
    KD = vcat(a...)' * D

    # set exogenous variable steady state value 
    #TODO: allow user to indicate this in main file
    Zexog = 1.0
    
    # Unpack parameters
    @unpack α, δ = model.ModParams
    
    # Calculate residuals
    residuals = [
        Y .- (Z .* (KS.^α)),
        r .+ δ .- (α .* Z .* (KS.^(α-1))),
        w .- ((1-α) .* Z .* (KS.^α)),
        KS .- KD, # capital market clearing
        Z .- Zexog # exogenous variable equality
        ]
    
    return vcat(vcat(residuals'...)...)
end



"""
    Residuals(x::Vector{Float64}, 
    model::SequenceModel)

Returns the residuals in the dynamics, given variable
values for the entire sequence of T periods.
"""
function Residuals(xVec, # (n_v x T-1) vector of all endogenous variable values
    KD, # T-1 sequence of capital demand values
    Zexog, # exogenous variable values
    model::SequenceModel)

    # Unpack parameters
    @unpack δ, α = model.ModParams
    @unpack n_v, T = model.CompParams

    xMat = transpose(reshape(xVec, (n_v, :))) # reshape to (T-1) x n_v matrix
    namedXvecs = NamedTuple{model.varXs}(Tuple([xMat[:,i] for i in 1:n_v]))
    @unpack Y, KS, r, w, Z = namedXvecs
    
    # generate lagged and exogenous variables
    KS_lag = [KS[1]; KS[1:end-1]]

    # Initialize residuals
    residuals = [
        Y .- (Z .* (KS_lag.^α)),
        r .+ δ .- (α .* Z .* (KS_lag.^(α-1))),
        w .- ((1-α) .* Z .* (KS_lag.^α)),
        KS .- KD,
        Z .- Zexog # exogenous variable equality
        ]
        
    transRes = [(i)' for i in residuals]

    return vec(reduce(vcat, transRes))
end



function get_KDemand(a_seq::Vector{Matrix{Float64}},
    D_seq::Vector{Vector{Float64}},
    model::SequenceModel)

    # Unpack parameters
    @unpack α = model.ModParams

    # Initialize vector
    KD = Zygote.Buffer(D_seq[1], length(a_seq), )

    for i in 1:length(a_seq)
        a = vcat(a_seq[i]...)
        KD[i] = a' * D_seq[i]
    end

    res = copy(KD)
    
    return res[1:end-1]
end


# MWE for the function
function ffunc(x::Vector{Float64})

    # first reshape
    xMat = reshape(x, (:, 2))
    
    # apply function
    res = Zygote.Buffer(zeros(Float64, 2, 2), 2, )
    res = [
        xMat[:,1] .+ xMat[:,2],
        xMat[:,1] .- xMat[:,2]
        ] # return value is a vector of two vectors

    # return vcat(vcat(res'...)...)
    return copy(reduce(vcat, reduce(vcat, res')))
end