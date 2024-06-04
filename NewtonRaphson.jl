# Implementing Boehl (2024) ``HANK on Speed" methodology
using DataFrames, UnPack
import ForwardDiff: derivative
import Zygote: pullback


struct HankModel
    params::Parameters
    varnames::Vector{Symbol}
    policygrid::Vector{Float64}
    shockmat::Matrix{Float64}
    Π::Matrix{Float64}
    policymat::Matrix{Float64}
end


function VarSequences(df::DataFrame) # dataframe consisting of T-1 rows and n_v columns
    field_names = map(Symbol, names(df))
    field_values = [Vector(df[!, name]) for name in field_names]
    return (; (field_names .=> field_values)...)
end



function JVP(func::Function, 
    primal::Vector{Float64}, 
    tangent::Vector{Float64})

    g(t) = func(primal + t*tangent)
    return derivative(g, 0.0)
end


function VJP(func::Function, 
    primal::Vector{Float64}, 
    cotangent::Vector{Float64})

    _, func_back = pull_back(func, primal)
    vjp_result, = func_back(cotangent)

    return vjp_result
end


function RayleighQuotient(J̅_inv::Matrix{Float64},
    Λxy::Vector{Float64},
    y::Vector{Float64})

    return (y' * J̅_inv * Λxy) / (y' * y)
end


function y_Iteration(J̅_inv::Matrix{Float64},
    x::Vector{Float64}, # evaluation point (primal)
    y_init::Vector{Float64}; # initial guess for y (tangent)
    α::Float64=0.5,
    γ::Float64=0.5,
    ε = 1e-9)

    # Initialize iteration
    y = y_new = y_init
    
    while ε < norm(y_old - y_new)
        y = y_new
        α_old = α

        Fx = EquilibriumResiduals(sv, x)
        Λxy = JVP(EquilibriumResiduals, x, y_old)
        α = min(α_old, γ / abs(RayleighQuotient(J̅_inv, Λxy, y)))
        
        y_new = y + α * J̅_inv * (Fx - Λxy)
    end
    
    return y_new
end


function NewtonRaphson(x_0::Vector{Float64}, # initial guess for x
    J̅_inv::Matrix{Float64}; # inverse of the steady-state Jacobian
    ε = 1e-9)

    x = x_new = x_0
    y = zeros(length(x))


    while ε < norm(x_old - x_new)
        x = x_new
        tangent = y_Iteration(J̅_inv, x, y)
        x_new = x - tangent
    end

    return x_new
end

