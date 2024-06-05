using SparseArrays

struct SteadyState
    vars::Vector{NamedTuple}
    policies::NamedTuple
    D::Vector{Float64}
end


function VarTuple(x::NamedTuple)
    return collect(values(x))
end


function JacobianBI(x̅::Vector{Float64}, # vector of steady state values
    func::Function, # should be the backward iteration function defined as Fₐ : x → a
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.ComputationalParams
    n = (T-1) * n_v
    
    # Initialize vectors
    ss_Vector = repeat(x̅, T-1)
    I = sparse(1.0I, n, n)
    JBI = Vector{Matrix{Float64}}(undef, n_v)

    for i in 1:n_v
        ∂a = JVP(func, ss_Vector, I[:, n - n_v + i])
        JBI[i] = transpose(∂a)
    end

    return JBI
end


function JacobianFI(x̅::Vector{Float64}, # vector of steady state values
    func::Function, # should be the composition of forward iteration function and aggregation, i.e., Fₓ o F_d : x → z 
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.ComputationalParams
    n = (T-1) * n_v
    
    # Initialize vectors
    ss_Vector = repeat(x̅, T-1)
    I = sparse(1.0I, n, n)
    JFI = Vector{Matrix{Float64}}(undef, n_v)

    for i in 1:n_v
        ∂a = VJP(func, ss_Vector, I[:, n - n_v + i])
        JFI[i] = transpose(∂a)
    end

    return JFI
end
