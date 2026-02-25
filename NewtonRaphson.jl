# Description: Newton-Raphson algorithm for solving for the equilibrium of the model
# based on Boehl (2021)


"""
    NewtonRaphsonHANK(x_0, J̅, exog_paths, mod, ss_initial, ss_ending; ε) -> Vector

Outer Newton-Raphson loop for the HANK sequence-space solver.

## Arguments

- `x_0::Vector{Float64}`: initial guess, length `n_endog × (T-1)` (endogenous
  variable sequence only; heterogeneous and exogenous variables are not part of
  the search vector).
- `J̅::SparseMatrixCSC`: approximate inverse of the steady-state Jacobian
  (used as a preconditioner inside `y_Iteration`).
- `exog_paths::NamedTuple`: exogenous variable sequences generated once before
  the loop via `generate_exog_paths(mod, T-1)`. Kept fixed across iterations.
- `mod::SequenceModel`: the model.
- `ss_initial`: starting steady state (at t=0); used as the left boundary in
  `assemble_full_xMat` and as the initial distribution for `ForwardIteration`.
- `ss_ending`: ending steady state (at t=T); used as the right boundary and as
  the terminal policy condition for `BackwardIteration`. For a transitory shock
  pass the same `SteadyState` object for both.
- `ε`: convergence tolerance (default `1e-9`).
"""
function NewtonRaphsonHANK(x_0::Vector{Float64},
                            J̅::SparseMatrixCSC,
                            exog_paths::NamedTuple,
                            mod::SequenceModel,
                            ss_initial,
                            ss_ending;
                            ε = 1e-9)
    x = x_0
    y = x_0
    i = 1

    while (ε < norm(y)) && (i < 100)
        y = y_Iteration(J̅, x, y, exog_paths, mod, ss_initial, ss_ending)
        x = x - y
        i += 1
        println("Iteration: $i, norm(y): $(norm(y))")
    end

    return x
end


"""
    y_Iteration(J̅, x, y0, exog_paths, mod, ss_initial, ss_ending; ...) -> Vector

Inner fixed-point iteration for the search direction `y` (Boehl 2021 method).

The full model function `F(x)` is defined as the composition:
1. `BackwardIteration(x, exog_paths, mod, ss_ending)` → policy sequences
2. `ForwardIteration(policy_seqs, mod, ss_initial)` → aggregated sequences
3. `assemble_full_xMat(x, agg_seqs, exog_paths, mod, ss_initial, ss_ending)` → padded matrix
4. `Residuals(padded_xMat, mod)` → residual vector

JVPs `Λ(x, y) = J(x)·y` are computed via forward-mode AD through this
full composition. The search direction update uses:

    y ← y + α · J̅⁻¹ · (F(x) − Λ(x, y))
"""
function y_Iteration(J̅::SparseMatrixCSC,
                     x,
                     y0,
                     exog_paths::NamedTuple,
                     mod::SequenceModel,
                     ss_initial,
                     ss_ending;
                     precond::Union{SparseMatrixCSC, Nothing} = nothing,
                     α::Float64 = 1.0,
                     γ::Float64 = 1.5,
                     ε = 1e-9)

    function fullFunction(x_Vec::AbstractVector)
        policy_seqs = BackwardIteration(x_Vec, exog_paths, mod, ss_ending)
        agg_seqs    = ForwardIteration(policy_seqs, mod, ss_initial)
        padded_xMat = assemble_full_xMat(x_Vec, agg_seqs, exog_paths,
                                         mod, ss_initial, ss_ending)
        return Residuals(padded_xMat, mod)
    end

    # Initialise iteration
    y       = y0
    y_old   = ones(length(y))
    Λxy     = zeros(length(y))
    M       = ones(length(y))
    R       = ones(length(y))
    Fx      = fullFunction(x)
    i       = 1

    while ε < norm(y - y_old)
        Λxy = JVP(fullFunction, x, y)
        # Solve J̅·R = F(x) − Λ(x,y)  and  J̅·M = Λ(x,y)  via restarted GMRes
        IterativeSolvers.gmres!(R, J̅, Fx - Λxy)
        IterativeSolvers.gmres!(M, J̅, Λxy)

        # Step-size update (α stub — TODO: implement adaptive rule from Boehl)
        ray = dot(y, M) / dot(y, y)
        α   = 0.5   # TODO: replace with alphaUpdate

        y_old = y
        y     = y_old + (α * R)

        i += 1
        if Base.mod(i, 10) == 0
            println("y_Iteration $i: α=$α  ‖y−y_old‖=$(norm(y - y_old))  ray=$ray")
        end
    end

    return y
end


function alphaUpdate(α::Float64, γ::Float64, x::Vector{Float64}, y::Vector{Float64})
    # TODO: implement adaptive α from Boehl (2021) §3
    return α
end


# ─────────────────────────────────────────────────────────────────────────────
# Testing utilities
# ─────────────────────────────────────────────────────────────────────────────

function test_Vec2Mat_JVP(x::Vector{<:Real})
    return [x * x', x * x' + Matrix(1.0I, length(x), length(x))]
end

function test_Vec2Vec_JVP(x::Vector{<:Real})
    return append!(vec(x * x'), vec(x * x' + Matrix(1.0I, length(x), length(x))))
end

function VarSequences(df::DataFrame)
    field_names  = map(Symbol, names(df))
    field_values = [Vector(df[!, name]) for name in field_names]
    return (; (field_names .=> field_values)...)
end
