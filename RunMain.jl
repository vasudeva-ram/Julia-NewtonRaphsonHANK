include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")
include("SteadyState.jl")
include("SteadyStateJacobian.jl")
include("NewtonRaphson.jl")


function solveModel(mod::SequenceModel, 
    stst::SteadyState,
    j̅::SparseMatrixCSC)
    
    T = mod.CompParams.T

    # steady state values
    xVec = repeat([values(stst.ssVars)...], T-1)

    # Exogenous variable
    shock = [0.8^t for t in 1:T-1];
    Zexog = 1.0 .+ shock
    
    # Obtain the solution
    z̅ = NewtonRaphsonHANK(xVec, j̅, mod, stst, Zexog);
    
    return z̅
end




# Some testing functions
mod, stst = test_SteadyState();
# zVals = SingleRun(stst, mod);
# jbi, jfi = getIntdJacobians(stst, mod);
# jdi = getDirectJacobian(stst, mod);
# jhelper = getJacobianHelper(jbi, jfi, jdi, mod);
# jfinal = getFinalJacobian(jhelper, jdi, mod);
# jcons = getConsolidatedJacobian(jfinal, mod);
T = mod.CompParams.T;
J = getSteadyStateJacobian(stst, mod);
# iluJ = ilu(J, τ=0.01); # incomplete LU factorization
# precond = (iluJ.L + sparse(I, size(iluJ.L))) * iluJ.U';
# dropzeros!(precond);
# Jinv =  approximate_inverse_ilu(iluJ, size(J̅)[1]); # approximate inverse
xVec = repeat([values(stst.ssVars)...], T-1); # steady state values
# Exogenous variable
shock = [0.8^t for t in 1:T-1];
Zexog = 1.0 .+ shock;
# Obtain the solution
x_fin = NewtonRaphsonHANK(xVec, J, mod, stst, Zexog);


xFinMat = reshape(x_fin, (mod.CompParams.n_v, T-1)) # reshape to (n_v, T-1) matrix
namedXfins = NamedTuple{mod.varXs}(Tuple([xFinMat[i, :] for i in 1:mod.CompParams.n_v]))

for key in keys(namedXfins)
    plot(namedXfins[key], collect(1:T-1), label="$(key)")
end
    

