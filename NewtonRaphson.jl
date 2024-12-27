# Implementing Boehl (2024) ``HANK on Speed" methodology


function VarSequences(df::DataFrame) # dataframe consisting of T-1 rows and n_v columns
    field_names = map(Symbol, names(df))
    field_values = [Vector(df[!, name]) for name in field_names]
    return (; (field_names .=> field_values)...)
end



function y_Iteration(J̅::SparseMatrixCSC,
    x::Vector{Float64}, # evaluation point (primal)
    y_init::Vector{Float64}, # initial guess for y (tangent)
    Zexog::Vector{Float64}, # exogenous variable values
    mod::SequenceModel,
    stst::SteadyState; #TODO: assumes starting and ending steady states are the same!!!!!!!
    α::Float64=0.5,
    γ::Float64=0.5,
    ε = 1e-9)

    # define full function
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(x_Vec, mod, stst)
        KD = ForwardIteration(a_seq, mod, stst)
        zVals = Residuals(x_Vec, KD, Zexog, mod)
        return zVals
    end
    
    # Initialize iteration
    y_old = zeros(length(y_init))
    y_new = y_init
    
    while ε < norm(y_old - y_new)
        y = y_new
        α_old = α

        Fx = fullFunction(x)
        Λxy = JVP(fullFunction, x, y_old)
        M = cg(J̅, Λxy) # conjugate gradient to get J̅⁻¹ * Λ(x,y)
        α = min(α_old, γ / abs(RayleighQuotient(M, y_old)))
        R = cg(J̅, Fx - Λxy) # conjugate gradient to get J̅⁻¹ * (F(x) - Λ(x,y))
        
        y_new = y + α * R
    end
    
    return y_new
end


function NewtonRaphson(x_0::Vector{Float64}, # initial guess for x
    J̅::SparseMatrixCSC, # inverse of the steady-state Jacobian
    mod::SequenceModel,
    stst::SteadyState; #TODO: assumes starting and ending steady states are the same!!!!!!!
    ε = 1e-9) # tolerance level

    @unpack T = mod.CompParams
    
    x = zeros(length(x_0))
    x_new = x_0
    y = zeros(length(x))
    Zexog = [0.85^t for t in 1:T-1]
    Zexog = 1.0 .+ Zexog

    while ε < norm(x - x_new)
        x = x_new
        tangent = y_Iteration(J̅, x, y, Zexog, mod, stst)
        x_new = x - tangent
    end

    return x_new
end



# Testing functions

function test_Vec2Mat_JVP(x::Vector{<:Real})
    return [x * x', x * x' + Matrix(1.0I, length(x), length(x))] 
end

function test_Vec2Vec_JVP(x::Vector{<:Real})
    return append!(vec(x * x'), vec(x * x' + Matrix(1.0I, length(x), length(x))))
end


