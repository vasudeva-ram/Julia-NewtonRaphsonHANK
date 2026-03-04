# BackwardIteration.jl — EGM backward pass along the transition path.
#
# Starting from the terminal marginal value at t=T (taken from the ending
# steady state), iterates backward to t=1 by calling model.value_fn at each
# period, collecting the sequence of policy matrices for all heterogeneous
# variables.


"""
    BackwardIteration(xVec_endog, exog_paths::NamedTuple,
                      model::SequenceModel, ss_end) -> NamedTuple

Backward iteration over the T-1 transition periods.

## Arguments

- `xVec_endog`: flat vector of length `n_endog × (T-1)`. Reshaped internally
  to `(n_endog × T-1)` where column `t` holds endogenous values at period `t`.
- `exog_paths`: NamedTuple mapping each `:exogenous` variable name to a
  length-(T-1) vector (from `generate_exog_paths`).
- `model`: the `SequenceModel`.
- `ss_end`: ending steady state. Must expose `.value` (terminal marginal value
  matrix) and `.vars` (aggregate SS values used to fill heterogeneous rows of
  xVals, which `value_fn` ignores but the signature requires).

## Algorithm

1. Initialise `value = ss_end.value` (terminal ∂V/∂a at t = T).
2. For t = T-1 down to 1, assemble `xVals_t` (length-n_v) and call
   `model.value_fn(value, xVals_t, model)` to get current-period `Value`
   and all policy matrices.
3. Store policy matrices; pass `Value` to the next (earlier) period.

## Returns

NamedTuple mapping each `:heterogeneous` variable name to a `Vector{Matrix}`
of length T-1, e.g. `(KD = [mat_1, …, mat_{T-1}],)`.

## AD compatibility

When `xVec_endog` carries ForwardDiff dual numbers, `value` is promoted to
`Matrix{Dual}` from the first backward step. The terminal `ss_end.value`
(Float64) is converted with zero partials so gradients flow only through the
endogenous sequence.
"""
function BackwardIteration(xVec_endog,
                           exog_paths::NamedTuple,
                           model::SequenceModel,
                           ss_end)
    @unpack T, n_v, n_endog = model.compspec
    TF = eltype(xVec_endog)

    # Reshape endogenous sequence: n_endog × (T-1), column t = period t values
    xMat_endog = reshape(xVec_endog, n_endog, T - 1)

    all_keys   = var_names(model)
    endog_keys = vars_of_type(model, :endogenous)
    het_keys   = vars_of_type(model, :heterogeneous)
    exog_keys  = vars_of_type(model, :exogenous)

    # Precompute row indices (avoid repeated findfirst inside the hot loop)
    endog_rows = [findfirst(==(k), all_keys) for k in endog_keys]
    exog_rows  = [findfirst(==(k), all_keys) for k in exog_keys]
    het_rows   = [findfirst(==(k), all_keys) for k in het_keys]

    # Pre-fetch ending SS values for heterogeneous rows (value_fn ignores these
    # but needs a concrete value for the signature)
    het_ss_vals = [ss_end.vars[k] for k in het_keys]

    function get_xvals_at_t(t::Int)
        xVals = Vector{TF}(undef, n_v)
        for (j, row) in enumerate(endog_rows)
            xVals[row] = xMat_endog[j, t]
        end
        for (j, row) in enumerate(exog_rows)
            xVals[row] = exog_paths[exog_keys[j]][t]
        end
        for (j, row) in enumerate(het_rows)
            xVals[row] = het_ss_vals[j]
        end
        return xVals
    end

    # Terminal marginal value from the ending steady state (zero AD partials)
    value = Matrix{TF}(ss_end.value)

    # Allocate policy sequence storage: one Vector{Matrix{TF}} per het variable
    seqs_data = [Vector{Matrix{TF}}(undef, T - 1) for _ in het_keys]

    for i in 1:T-1
        t      = T - i
        result = model.value_fn(value, get_xvals_at_t(t), model)

        # Validate return structure on the first call only
        if i == 1
            result isa NamedTuple ||
                error("BackwardIteration: value_fn must return a NamedTuple " *
                      "(got $(typeof(result)))")
            :Value in keys(result) ||
                error("BackwardIteration: value_fn return must have a :Value key " *
                      "(got keys: $(keys(result)))")
            for k in het_keys
                k in keys(result) ||
                    error("BackwardIteration: value_fn return is missing key :$k " *
                          "(got keys: $(keys(result)))")
            end
        end

        value = result.Value
        for (j, varname) in enumerate(het_keys)
            seqs_data[j][t] = result[varname]
        end
    end

    return NamedTuple{het_keys}(Tuple(seqs_data))
end
