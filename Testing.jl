using AutoGrad, Zygote

function fexample(x)
    output = [
        x[1]^3 + x[2]^2 + x[3] + 2*x[4],
        7*x[1] + x[2] + x[3]^2 + x[4]^3,
        x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    ]

    return output
end


function VJP_Old(func::Function, 
    primal::Vector{Float64}, 
    cotangent::Vector{Float64})

    _, func_back = Zygote.pullback(func, primal)
    vjp_result, = func_back(cotangent)

    return vjp_result
end


function VJP_new(func::Function, 
    primal::Vector{Float64}, 
    cotangent::Vector{Float64}) 
    
    x = AutoGrad.Param(primal)
    temp = @diff dot(cotangent, func(x))
    
    return AutoGrad.grad(temp, x)
end


function create_pullback(func, x)
    # Create a tape that records operations on x
    tape = ReverseDiff.JacobianTape(func, x)
    
    # Create a compiled version of the tape for efficiency
    compiled_tape = ReverseDiff.compile(tape)

    # Define the pullback function
    function pullback(Δ)
        # Compute the gradient using ReverseDiff
        return ReverseDiff.jacobian!(compiled_tape, Δ)
    end

    # Return the pullback function
    return pullback
end

function VJP_RD(func::Function, 
    primal::Vector{Float64}, 
    cotangent::Vector{Float64})

    function pullback_f(x)
        return dot(cotangent, func(x))
    end
    
    tape = ReverseDiff.JacobianTape(pullback_f, primal)
    compiled_tape = ReverseDiff.compile(tape)
    
    vjp_result = similar(primal)
    ReverseDiff.jacobian!(vjp_result, compiled_tape, primal)

    return vjp_result
end



prim = [1.0, 2.0, 3.0, 4.0]
cotan = [1.0, 2.0, 3.0]

jac, = Zygote.jacobian(fexample, prim)


vjp_a = cotan' * jac
vjp_b = VJP_Old(fexample, prim, cotan)
vjp_c = VJP_new(fexample, prim, cotan)
vjp_d = VJP_RD(fexample, prim, cotan)