# Description: This file contains the functions that calculate the residuals
# in the dynamics and steady state of the model. The residuals are used in the
# solution of the model, and are calculated by comparing the model's equations
# to the actual values of the variables.


"""
    Residuals(xMat::AbstractMatrix, model::SequenceModel)

Generic residuals function that evaluates the model's compiled equations
against the variable matrix `xMat` (n_v × T-1).

All variables (endogenous, exogenous, aggregated) should be rows of `xMat`,
ordered to match `model.varXs`. Returns a vector of residuals ordered as:
all equations at t=1, all equations at t=2, ... (column-major of k × (T-1)).

This is the main residuals function in the pipeline:
    BackwardIteration → ForwardIteration → Residuals
"""
function Residuals(xMat::AbstractMatrix, model::SequenceModel)
    return model.residuals_fn(xMat, model.params)
end


