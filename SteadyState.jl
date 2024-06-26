include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")


function JacobianBI(end_ss::SteadyState, # ending steady state
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = (T-1) * n_v
    start_ss = end_ss
    
    function backFunc(xVec::AbstractVector) # (n_v * T-1)-dimensional vector
        a_seq = BackwardIteration(xVec, model, end_ss)
        KD = ForwardIteration(a_seq, model, start_ss)
        zVals = Residuals(xVec, KD, model)
        return zVals
    end

    # Initialize vectors
    ss_Vector = repeat([values(end_ss.ssVars)...], T-1)
    
    idmat = sparse(1.0I, n, n)
    JBI = Vector{AbstractVector}(undef, n_v)

    for i in 1:n_v
        JBI[i] = JVP(backFunc, ss_Vector, idmat[:, n - n_v + i])
    end

    return JBI
end


function JacobianFI(a_seq::Vector{Matrix{Float64}}, # vector of steady state values
    start_ss::SteadyState, # starting steady state
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.CompParams
    n = (T-1) * n_v
    
    # Initialize vectors
    ss_xVec = repeat([values(start_ss.ssVars)...], T-1)
    idmat = sparse(1.0I, n, n)
    JFI = Vector{Vector{Float64}}(undef, n_v)
    
    function forwardFunc(xVec::Vector{Float64})
        KD = ForwardIteration(a_seq, model, start_ss)
        zVals = Residuals(xVec, KD, model)
        return zVals
    end
    
    for i in 1:n_v
        JFI[i] = VJP(forwardFunc, ss_xVec, idmat[:, n - n_v + i])
    end

    return JFI
end


"""
    get_SteadyState(model::SequenceModel;
    guess::Union{NamedTuple, nothing}=nothing)

Function to obtain the steady state using the Newton-Raphson method. 
"""
function get_SteadyState(model::SequenceModel;
    guess = nothing) # initial guess for the steady state values

    @unpack policygrid, Π = model
    
    # Define main function such that Fₓ : x → z
    function Fx(xVals::Vector{Float64}) #TODO: annotate to support dual numbers and float64 vectors
        a = BackwardSteadyState(xVals, model) # get steady state policies
        Λ = DistributionTransition(a, policygrid, Π) # get transition matrix
        D = invariant_dist(Λ') # get invariant distribution
        z = ResidualsSteadyState(xVals, a, D, model) # get residuals
        
        return z
    end
    
    # Initialize steady state guess
    x̅ = isnothing(guess) ? rand(length(model.varNs)) : collect(values(guess))
    tol = 1.0
    ε = model.CompParams.ε
    
    # find steady state solution (trust region method)
    sol = nlsolve(Fx, x̅)
    x = sol.zero

    # Build the steady state policies and distribution
    ssVars = NamedTuple{model.varXs}(x)
    policies = BackwardSteadyState(x, model)
    Λss = DistributionTransition(policies, model.policygrid, model.Π)
    dist = invariant_dist(Λss')

    return SteadyState(ssVars, policies, Λss, dist)
end


# Test Run

function test_SteadyState()
    varXs = (:Y, :KS, :r, :w, :Z)
    sig = 0.5 * sqrt(1 - (0.966^2))
    modpars = ModelParams(0.98, 1.0, sig, 0.966, 0.025, 0.11)
    compars = ComputationalParams(0.0001, [0.0, 200.0], 200, 7, length(varXs), 150, 1e-9)
    policygrid = make_DoubleExponentialGrid(compars.gridx[1], compars.gridx[2], compars.n_a)
    Π, _, shockgrid = get_RouwenhorstDiscretization(compars.n_e, modpars.ρ, modpars.σ)
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)
        
    mod = SequenceModel(varXs, compars, modpars, policygrid, shockmat, policymat, Π)

    # Obtain steady state
    ss = get_SteadyState(mod, guess = (Y = 1.0, KS = 1.0, r = 0.02, w = 0.1, Z = 1.0))

    return mod, ss
end


function forwardFunc(aseqvec::AbstractVector)
    KD = ForwardIteration(aseqvec, mod, stst)
    zVals = Residuals(xVec, KD, mod)
    return zVals
end

function bi(xVec)
    a_seq = BackwardIteration(xVec, mod, stst)
    return a_seq
end

# Some testing functions
mod, stst = test_SteadyState();
#JBI = JacobianBI(stst, mod);
xVec = repeat([values(stst.ssVars)...], mod.CompParams.T-1);
# a_seq = BackwardIteration(xVec, mod, stst);
# a_seq = [convert(Matrix{Float64}, mat) for mat in a_seq];
# jbi = JacobianBI(stst, mod);
# fi = ForwardIteration(a_seq, mod, stst);
# jfi = JacobianFI(a_seq, stst, mod);

