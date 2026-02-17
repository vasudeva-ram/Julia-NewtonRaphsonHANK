# Description: Newton-Raphson algorithm for solving for the equilibrium of the model
# based on Boehl(2021)

function NewtonRaphsonHANK(x_0::Vector{Float64}, # initial guess for x
    J̅::SparseMatrixCSC, # inverse of the steady-state Jacobian
    mod::SequenceModel,
    stst::SteadyState, #TODO: assumes starting and ending steady states are the same!!!!!!!
    Zexog::Vector{Float64}; # exogenous variable values
    precond::Union{SparseMatrixCSC, Nothing}=nothing, # preconditioner for J̅ if available
    ε = 1e-9) # tolerance level

    @unpack T = mod.Params
    
    x = x_0
    y = x_0
    y_new = ones(length(x))
    i = 1

    while (ε < norm(y)) & (i < 100)
        y_new = y_Iteration(J̅, x, y, Zexog, mod, stst)
        x = x - y_new
        y = y_new
        
        i += 1
        println("Iteration: $i, norm(y): $(norm(y))")
    end

    return x
end


function y_Iteration(J̅::SparseMatrixCSC,
    x, # evaluation point (primal)
    y0, # search direction (dual)
    Zexog, # exogenous variable values
    mod::SequenceModel,
    stst::SteadyState; #TODO: assumes starting and ending steady states are the same!!!!!!!
    precond::Union{SparseMatrixCSC, Nothing}=nothing,
    α::Float64=1.0,
    γ::Float64=1.5,
    ε = 1e-9)

    # define full function
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        policy_seqs = BackwardIteration(x_Vec, mod, stst)
        xMat = ForwardIteration(x_Vec, policy_seqs, mod, stst)
        zVals = Residuals(xMat, mod)
        return zVals
    end
    
    # Initialize iteration
    y = y0
    y_old = ones(length(y))
    Λxy = zeros(length(y))
    M = ones(length(y))
    R = ones(length(y))
    Fx = fullFunction(x)
    i = 1
    
    while ε < norm(y - y_old)
        # obtain rayleigh quotients
        Λxy = JVP(fullFunction, x, y)
        IterativeSolvers.gmres!(R, J̅, Fx - Λxy) # restarted GMRes to get J̅⁻¹ * (F(x) - Λ(x,y))
        IterativeSolvers.gmres!(M, J̅, Λxy) # restarted GMRes to get J̅⁻¹ * Λ(x,y)
        
        # update α's
        α_old = α
        ray = dot(y, M) / dot(y, y)
        # α = min(α_old, γ / abs(ray))
        α = 0.5
        
        # update y
        y_old = y
        y = y_old + (α * R)

        i += 1
        if Base.mod(i, 10) == 0
            println("Iteration: $i, α: $α, norm(y - y_old): $(norm(y - y_old)), ray: $(ray)")
        end
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

