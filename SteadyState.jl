include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")


function JacobianBI(x̅::Vector{Float64}, # vector of steady state values
    func::Function, # should be the backward iteration function defined as Fₐ : x → a
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
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
    @unpack T, n_v = model.CompParams
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


"""
    get_SteadyState(model::SequenceModel;
    guess::Union{NamedTuple, nothing}=nothing)

Function to obtain the steady state using the Newton-Raphson method. 
"""
function get_SteadyState(model::SequenceModel,
    ssVarXs; # exogenous variable values
    guess = nothing) # initial guess for the steady state values

    @unpack policygrid, Π = model
    
    # Define main function such that Fₓ : x → z
    function Fx(varN) #TODO: annotate to support dual numbers and float64 vectors
        a = BackwardSteadyState(varN, model) # get steady state policies
        Λ = DistributionTransition(a, policygrid, Π) # get transition matrix
        D = invariant_dist(Λ') # get invariant distribution
        z = ResidualsSteadyState(varN, ssVarXs, a, D, model) # get residuals
        
        return z
    end
    
    # Initialize steady state guess
    x̅ = isnothing(guess) ? rand(length(model.varNs)) : collect(values(guess))
    x = x̅
    tol = 1.0
    ε = model.CompParams.ε

    # implement Newton-Raphson method
    while ε < tol
        x = x̅
        J = jacobian(Fx, x)
        x̅ = x - (inv(J) * Fx(x))
        tol = norm(x̅ - x)
    end

    # sol = nlsolve(Fx, x̅)
    # x = sol.zero

    # Build the steady state policies and distribution
    ssVars = NamedTuple{model.varNs}(x)
    policies = BackwardSteadyState(x, model)
    Λss = DistributionTransition(policies, model.policygrid, model.Π)
    dist = invariant_dist(Λss')

    return SteadyState(ssVars, policies, Λss, dist)
end


# Test Run

function test_SteadyState()
    varNs = (:Y, :KS, :r, :w)
    varXs = (:Z,)
    sig = 0.5 * sqrt(1 - (0.966^2))
    modpars = ModelParams(0.98, 1.0, sig, 0.966, 0.025, 0.11)
    compars = ComputationalParams(0.0001, [0.0, 200.0], 200, 7, length(varNs), 300, 1e-9)
    policygrid = make_DoubleExponentialGrid(compars.gridx[1], compars.gridx[2], compars.n_a)
    Π, _, shockgrid = get_RouwenhorstDiscretization(compars.n_e, modpars.ρ, modpars.σ)
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)
        
    mod = SequenceModel(varNs, varXs, compars, modpars, policygrid, shockmat, policymat, Π)

    # Obtain steady state
    ss = get_SteadyState(mod, [1.0], guess = (Y = 1.0, KS = 1.0, r = 0.02, w = 0.1))

    return ss
end
