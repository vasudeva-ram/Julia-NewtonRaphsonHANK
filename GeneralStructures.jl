# Imports and Uses
using LinearAlgebra, SparseArrays, DataFrames, UnPack, NLsolve, BenchmarkTools, Interpolations
using Zygote, ForwardDiff, IncompleteLU, IterativeSolvers, TOML

#NOTE: The following steady-state struct is specific to the Krussell-Smith model only.
struct SteadyState
    ssVars::NamedTuple # steady state values of the variables
    ssPolicies::Matrix{Float64} # steady state savings policies
    ssΛ::SparseMatrixCSC{Float64,Int64} # steady state transition matrix
    ssD::Vector{Float64} # steady state stationary distribution
end

struct ModelParams{TF<:Float64}
    # Economic parameters
    β::TF # discount factor
    γ::TF # coefficient of relative risk aversion
    σ::TF # standard deviation of the income shock process
    ρ::TF # persistence of the shock process
    δ::TF # depreciation rate
    α::TF # share of capital in production
    # Computational parameters
    dx::TF # size of infinitesimal shock for numerical differentiation
    gridx::Vector{Float64} # [a_min, a_max] bounds for the savings grid
    n_a::Int64 # number of grid points for the savings grid
    n_e::Int64 # number of grid points for the shock grid
    n_v::Int64 # number of variables in the model
    T::Int64 # number of periods for the transition path
    ε::TF # convergence/ tolerance criterion
end

struct SequenceModel{F, A}
    varXs::Tuple{Vararg{Symbol}} # tuple of all *aggregate* variable names only (exog. + endog.)
    equations::Tuple{Vararg{String}} # equilibrium equations in immutable order
    Params::ModelParams # all model parameters (economic + computational)
    residuals_fn::F # compiled residuals function: (xMat, params) -> Vector
    agg_vars::A # NamedTuple mapping aggregated var symbols → (backward, forward) function pairs
    policygrid::Vector{Float64} # grid of possible savings positions #TODO: eliminate either policygrid or policymat
    shockmat::Matrix{Float64} # n_a x n_e matrix of shock values
    policymat::Matrix{Float64} # n_a x n_e matrix of savings values
    Π::Matrix{Float64} # transition matrix for the exogenous shock process
end


"""
    shift_lag(x::AbstractVector, i::Int)

Shifts a time series vector `x` back by `i` periods.
The first `i` entries are filled with `x[1]` (boundary condition).
"""
function shift_lag(x::AbstractVector, i::Int)
    return vcat(fill(x[1], i), x[1:end-i])
end


"""
    shift_lead(x::AbstractVector, i::Int)

Shifts a time series vector `x` forward by `i` periods.
The last `i` entries are filled with `x[end]` (boundary condition).
"""
function shift_lead(x::AbstractVector, i::Int)
    return vcat(x[i+1:end], fill(x[end], i))
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
    primal::AbstractVector, 
    tangent::AbstractVector)

    g(t) = func(primal + t*tangent)
    res = ForwardDiff.derivative(g, 0.0)
    
    return sparse(res)
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
# function VJP(func::Function, 
#     primal::AbstractVector, 
#     cotangent::SparseVector)

#     function pullback_f(x)
#         return dot(cotangent, func(x))
#     end
    
#     tape = ReverseDiff.JacobianTape(pullback_f, primal)
#     compiled_tape = ReverseDiff.compile(tape)
    
#     vjp_result = similar(primal)
#     ReverseDiff.jacobian!(vjp_result, compiled_tape, primal)

#     return sparse(vjp_result)
# end


# function compile_tape(func::Function, 
#     primal::AbstractVector)

#     tape = ReverseDiff.JacobianTape(func, primal)
#     compiled_tape = ReverseDiff.compile(tape)
#     return compiled_tape
# end




"""
    RayleighQuotient(M, z)

Computes the Rayleigh quotient of a matrix `M` and a vector `z`.
Inputs should be AbstractMatrix and AbstractVector.
"""
function RayleighQuotient(M, z)

    return dot(z, M*z) / dot(z, z)
end


function exogenousZ!(namedXvars::NamedTuple, 
    model::SequenceModel)

    @unpack Z, ρ, σ = namedXvars
    Π = model.Π
    Z = ρ * Z + σ * sqrt(1 - ρ^2) * randn()
    return Z
end


# Function to approximate the inverse using ILU
function approximate_inverse_ilu(iluJ, n)
    Jinv = spzeros(n, n)  # Initialize the inverse matrix
    Iden = sparse(I, n, n)  # Identity matrix

    # Solve J̅ * x = e_i for each column e_i of the identity matrix
    for i in 1:n
        e_i = Iden[:, i]
        x_i = iluJ \ e_i
        Jinv[:, i] = x_i
    end

    return Jinv
end


macro unpack_all(q)
    quote
        
    end
end