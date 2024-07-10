# Imports and Uses
using LinearAlgebra, SparseArrays, DataFrames, UnPack, NLsolve, BenchmarkTools, Interpolations
using Zygote, ForwardDiff, ReverseDiff

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

struct ComputationalParams
    dx::Float64 # size of infinitesimal shock for numerical differentiation
    gridx::Vector{Float64} # [a_min, a_max] bounds for the savings grid
    n_a::Int64 # number of grid points for the savings grid
    n_e::Int64 # number of grid points for the shock grid
    n_v::Int64 # number of variables in the model
    T::Int64 # number of periods for the transition path
    ε::Float64 # convergence/ tolerance criterion
end

struct SequenceModel
    varXs::Tuple{Vararg{Symbol}} # tuple of all *aggregate* variable names only (exog. + endog.)
    CompParams::ComputationalParams # parameters determining computational structure of model
    ModParams::ModelParams # parameters determining agents' economic behavior of model
    policygrid::Vector{Float64} # grid of possible savings positions #TODO: eliminate either policygrid or policymat
    shockmat::Matrix{Float64} # n_a x n_e matrix of shock values
    policymat::Matrix{Float64} # n_a x n_e matrix of savings values
    Π::Matrix{Float64} # transition matrix for the exogenous shock process
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


"""
    vectorize_matrices(matrices::Vector{Matrix{Float64}})

Converts a vector of matrices, `VM = [M1, M2, M3 ...]` into a single vector
`V = [vec(M1), vec(M2), vec(M3) ...]`.
"""
function vectorize_matrices(matrices::Vector{<:Matrix})
    n, m = size(matrices[1])
    result = similar(matrices[1], n*m, length(matrices))
    for i in 1:length(matrices)
        result[:, i] = vec(matrices[i])
    end
    return [result...]
end


function Vec2Mat(vec::Vector{Float64}, n::Int64, m::Int64)
    T = Int(length(vec)/(n*m))
    kmat = reshape(vec, (n, m, T))
    return [kmat[:, :, i] for i in 1:T]
end


"""
    JVP(func::Function, 
    primal::Vector{Float64}, 
    tangent::AbstractArray)

Returns the Jacobian-Vector Product (JVP) of a function.
Given a function `func`, a primal point `primal`, and a tangent vector `tangent`,
the JVP is given by `Jᵀ * tangent`, where `J` is the Jacobian of `func` evaluated 
at the point `primal`.
"""
function JVP(func::Function, 
    primal::Vector{Float64}, 
    tangent::AbstractArray)

    g(t) = func(primal + t*tangent)
    res = ForwardDiff.derivative(g, 0.0)
    
    return res
end


function JVP(func::Function, 
    primal::Vector{Vector{Float64}}, 
    tangent::Vector{Vector{Float64}})

    g(t) = func(primal + t*tangent)
    res = ForwardDiff.derivative(g, 0.0)
    
    return res
end


function testfx(x)
    return x[1]^2 + x[2]^2
end


"""
    VJP(func::Function, 
    primal::Vector{Float64}, 
    cotangent::AbstractArray)

Returns the Vector-Jacobian Product (VJP) of a function.
Given a function `func`, a primal point `primal`, and a cotangent vector `cotangent`,
the VJP is given by `cotangent * J`, where `J` is the Jacobian of `func` evaluated
at the point `primal`.
"""
function VJP(func::Function, 
    primal::Vector{Matrix{Float64}}, 
    cotangent::SparseVector{Float64, Int64})

    _, func_back = pullback(func, primal)
    vjp_result, = func_back(cotangent)

    return vjp_result
end


"""
    RayleighQuotient(J̅_inv::Matrix{Float64},
    Λxy::Vector{Float64},
    y::Vector{Float64})

Calculates the Rayleigh Quotient of a matrix `J̅_inv` and vectors `Λxy` and `y`.
In our implementation, `J̅_inv` is the inverse of the Jacobian matrix evaluated 
at the steady state, `Λxy` is the Jacobian-Vector Product (JVP) of the `Residuals` 
function evaluated against the next guess, `y`, as the tangent vector.
"""
function RayleighQuotient(J̅_inv::Matrix{Float64},
    Λxy::Vector{Float64},
    y::Vector{Float64})

    return (y' * J̅_inv * Λxy) / (y' * y)
end


function VJP_RD(func::Function, 
    primal, 
    cotangent)

    function pullback_f(x)
        return dot(cotangent, func(x))
    end
    
    tape = ReverseDiff.JacobianTape(pullback_f, primal)
    compiled_tape = ReverseDiff.compile(tape)
    
    vjp_result = similar(primal)
    ReverseDiff.jacobian!(vjp_result, compiled_tape, primal)

    return vjp_result
end
