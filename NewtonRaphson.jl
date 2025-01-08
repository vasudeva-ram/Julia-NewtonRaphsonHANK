# Description: Newton-Raphson algorithm for solving for the equilibrium of the model
# based on Boehl(2021)

function NewtonRaphsonHANK(x_0::Vector{Float64}, # initial guess for x
    J̅::SparseMatrixCSC, # inverse of the steady-state Jacobian
    precond::SparseMatrixCSC, # incomplete LU factorization of J̅
    mod::SequenceModel,
    stst::SteadyState, #TODO: assumes starting and ending steady states are the same!!!!!!!
    Zexog::Vector{Float64}; # exogenous variable values
    ε = 1e-9) # tolerance level

    @unpack T = mod.CompParams
    
    x_new = x_0
    x = zeros(length(x_0))
    y = zeros(length(x))
    y_new = zeros(length(x))
    
    while ε < norm(x - x_new)
        y = x - x_new
        y_new = y_Iteration(J̅, precond, x_new, y, Zexog, mod, stst)
        x = x_new
        x_new = x - y_new
    end

    return x_new
end


function y_Iteration(J̅::SparseMatrixCSC,
    precond::SparseMatrixCSC,
    x::Vector{Float64}, # evaluation point (primal)
    y_init::Vector{Float64}, # initial guess for y (tangent)
    Zexog::Vector{Float64}, # exogenous variable values
    mod::SequenceModel,
    stst::SteadyState; #TODO: assumes starting and ending steady states are the same!!!!!!!
    α::Float64=1.5,
    γ::Float64=1.5,
    ε = 1e-9)

    # define full function
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, mod, stst)
        KD = ForwardIteration(a_seq, mod, stst)
        zVals = Residuals(x_Vec, KD, Zexog, mod)
        return zVals
    end
    
    # Initialize iteration
    y = y_init
    y_old = ones(length(y))
    Λxy = zeros(length(y))
    M = zeros(length(y))
    R = zeros(length(y))
    Fx = fullFunction(x)

    
    while ε < norm(y - y_old)
        # obtain rayleigh quotients
        Λxy = JVP(fullFunction, x, y)
        IterativeSolvers.gmres!(R, J̅, Fx - Λxy, Pr=precond) # restarted GMRes to get J̅⁻¹ * (F(x) - Λ(x,y))
        IterativeSolvers.gmres!(M, J̅, Λxy, Pr=precond) # restarted GMRes to get J̅⁻¹ * Λ(x,y)
        
        # update α's
        α_old = α
        ray = dot(y, M) / dot(y, y)
        α = min(α_old, γ / abs(ray))
        
        # update y
        y_old = y
        y = y_old + (α * R)
    end
    
    return y
end


function alphaUpdate(α::Float64,
    γ::Float64,
    x::Vector{Float64},
    y::Vector{Float64}
    )
    
    return
end


# Testing functions

function test_Vec2Mat_JVP(x::Vector{<:Real})
    return [x * x', x * x' + Matrix(1.0I, length(x), length(x))] 
end

function test_Vec2Vec_JVP(x::Vector{<:Real})
    return append!(vec(x * x'), vec(x * x' + Matrix(1.0I, length(x), length(x))))
end


function VarSequences(df::DataFrame) # dataframe consisting of T-1 rows and n_v columns
    field_names = map(Symbol, names(df))
    field_values = [Vector(df[!, name]) for name in field_names]
    return (; (field_names .=> field_values)...)
end

