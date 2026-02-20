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


"""
    ResidualsSteadyState(xVals::AbstractVector,
    a::AbstractMatrix,
    D::AbstractVector,
    model::SequenceModel)

Returns the residuals in the steady state, given variable values, savings
policies and a stationary distribution.

#TODO: still hardcodes KS equations — replace with compiled residuals once
       compile_residuals supports scalar (non-sequence) evaluation.
"""
function ResidualsSteadyState(xVals::AbstractVector, # vector of variable values
    a::AbstractMatrix, # matrix of savings policies
    D::AbstractVector, # steady state distribution
    model::SequenceModel)

    namedXvars = NamedTuple{model.varXs}(xVals)
    @unpack Y, KS, r, w, Z = namedXvars

    # Initialize vectors
    residuals = zeros(length(xVals))

    # Calculate aggregated variables
    KD = vcat(a...)' * D

    # set exogenous variable steady state value
    #TODO: allow user to indicate this in ModelFile.toml
    Zexog = 1.0

    # Unpack parameters
    @unpack α, δ = model.params

    # Calculate residuals
    residuals = [
        Y .- (Z .* (KS.^α)),
        r .+ δ .- (α .* Z .* (KS.^(α-1))),
        w .- ((1-α) .* Z .* (KS.^α)),
        KS .- KD, # capital market clearing
        Z .- Zexog # exogenous variable equality
        ]

    return vcat(vcat(residuals'...)...)
end
