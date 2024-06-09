# Imports and Uses
using LinearAlgebra, SparseArrays, DataFrames, UnPack, NLsolve, BenchmarkTools, Interpolations
import ForwardDiff: jacobian 
import ForwardDiff: Dual
import Zygote: gradient

#NOTE: The following steady-state struct is specific to the Krussell-Smith model only.
struct SteadyState
    ssVars::NamedTuple # steady state values of the variables
    ssPolicies::Matrix{Float64} # steady state savings policies
    ssΛ::SparseMatrixCSC{Float64,Int64} # steady state transition matrix
    ssD::Vector{Float64} # steady state stationary distribution
end

struct ModelParams{TF<:Float64}
    β::TF # discount factor
    γ::TF # coefficient of relative risk aversion
    σ::TF # standard deviation of the income shock process
    ρ::TF # persistence of the shock process
    δ::TF # depreciation rate
    α::TF # share of capital in production
end

struct ComputationalParams{TF<:Float64, TI<:Int64}
    dx::TF # size of infinitesimal shock for numerical differentiation
    gridx::Vector{TF} # [a_min, a_max] bounds for the savings grid
    n_a::TI # number of grid points for the savings grid
    n_e::TI # number of grid points for the shock grid
    n_v::TI # number of variables in the model
    T::TI # number of periods for the transition path
    ε::TF # convergence/ tolerance criterion
end

struct SequenceModel
    varNs::Tuple{Vararg{Symbol}} # tuple of *aggregate* endogenous variable names only
    varXs::Tuple{Vararg{Symbol}} # tuple of *aggregate* exogenous variable names only
    CompParams::ComputationalParams # parameters determining computational structure of model
    ModParams::ModelParams # parameters determining agents' economic behavior of model
    policygrid # grid of possible savings positions #TODO: eliminate either policygrid or policymat
    shockmat # n_a x n_e matrix of shock values
    policymat # n_a x n_e matrix of savings values
    Π # transition matrix for the shock process
end

# General functions
"""
    make_DoubleExponentialGrid(amin::Float64, 
    amax::Float64, 
    n_a::Int64)

Produces a double-exponential grid of asset holdings.
Compared to a uniform grid, the double-exponential grid is more dense around the origin.
This provides more precision for the asset holdings of the poorest households,
    where nonlinearities are most prevalent.
"""
function make_DoubleExponentialGrid(amin::Float64, 
    amax::Float64, 
    n_a::Int64)
    
    # Find maximum 𝕌 corresponding to amax
    𝕌 = log(1 + log(1 + amax- amin))

    # Create the uniform grid
    𝕌grid = range(0, 𝕌, n_a)

    # Transform the uniform grid to the double-exponential grid
    agrid = amin .+ exp.(exp.(𝕌grid) .- 1) .- 1

    return agrid
end


"""
    get_RouwenhorstDiscretization(n::Int64, # dimension of state-space
    ρ::Float64, # persistence of AR(1) process
    σ::Float64)

Discretizes an AR(1) process using the Rouwenhorst method.
See Kopecky and Suen (2009) for details: http://www.karenkopecky.net/Rouwenhorst_WP.pdf
Better than Tauchen (1986) method especially for highly persistent processes.
"""
function get_RouwenhorstDiscretization(n::Int64, # dimension of state-space
    ρ::Float64, # persistence of AR(1) process
    σ::Float64) # standard deviation of AR(1) process

    # Construct the transition matrix
    p = (1 + ρ)/2
    
    Π = [p 1-p; 1-p p]
    
    for i = 3:n
        Π_old = Π
        Π = zeros(i, i)
        Π[1:i-1, 1:i-1] += p * Π_old
        Π[1:i-1, 2:end] += (1-p) * Π_old
        Π[2:end, 1:i-1] += (1-p) * Π_old
        Π[2:end, 2:end] += p * Π_old
        Π[2:i-1, 1:end] /= 2
    end

    # Obtain the stationary distribution
    #TODO: should Π be transposed here? What does Rouwenhorst return? 
    #SOLVED: No, Π should not be transposed here; it gets transposed (correctly) within the invariant_dist function  
    D = invariant_dist(Π) 

    # Construct the state-space
    α = 2 * (σ/sqrt(n-1))
    z = exp.(α * collect(0:n-1))
    z = z ./ sum(z .* D) # normalize the distribution to have mean of 1
    
    return Π, D, z

end


