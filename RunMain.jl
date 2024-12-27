include("GeneralStructures.jl")
include("ForwardIteration.jl")
include("BackwardIteration.jl")
include("Aggregation.jl")
include("SteadyState.jl")
include("SteadyStateJacobian.jl")


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
zVals = SingleRun(stst, mod);
# jbi, jfi = getIntdJacobians(stst, mod);
# jdi = getDirectJacobian(stst, mod);
# jhelper = getJacobianHelper(jbi, jfi, jdi, mod);
# jfinal = getFinalJacobian(jhelper, jdi, mod);
# jcons = getConsolidatedJacobian(jfinal, mod);
j̅ = getSteadyStateJacobian(stst, mod);



