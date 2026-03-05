# Description: This file contains the functions to obtain the steady state of the model.
# Uses a modified Newton-Raphson method with the Moore-Penrose pseudoinverse.



"""
    SteadyState

Stores the steady-state solution of the model. Contains the steady-state
values of all aggregate variables, the household policy functions, the
transition matrix, and the stationary distribution over household states.

Fields:
- `vars`: NamedTuple of steady-state variable values, keyed by `var_names(model)`
   (e.g., `(Y=1.0, KS=3.5, r=0.02, w=0.9, Z=1.0)`)
- `policies`: NamedTuple of policy matrices, one per `:heterogeneous` variable
   (e.g., `(KD = Matrix{Float64},)`)
- `Λ`: column-stochastic joint transition matrix (endogenous × exogenous Kronecker)
- `D`: stationary distribution vector (length `n_endog_states * n_exog_states`)
"""
struct SteadyState{VT}
    vars::NamedTuple
    policies::NamedTuple
    Λ::SparseMatrixCSC{Float64,Int64}
    D::Vector{Float64}
    value::VT    # steady-state marginal value ∂V/∂a (n_a × n_e); terminal condition for backward iteration
end


"""
    SSAssembler{M}

Encapsulates the variable-role logic and padded-matrix assembly for the
steady-state Newton solve. Constructed once per model; callable as a function.

Precomputes the time-invariant exogenous Kronecker factor `Λ_exog` at
construction so that successive Newton iterations only need to compose it with
an updated `Λ_endog` — avoiding O(n²) Kronecker products on every call.

## Variable roles
- **Free** (`free_keys`): `:endogenous` vars NOT listed in `ss_initial.fixed`.
  These are the Newton search variables — the components of `p_vec`.
- **Pinned** (`ss_initial.fixed`): exogenous vars and any endogenous vars the
  user wants held fixed. Assigned with zero ForwardDiff derivatives.
- **Heterogeneous**: computed from `model.value_fn` (inner VFI) → `make_endogenous_transition` → `invariant_dist` → `dot`.

## Calling convention

    asm(p_vec::AbstractVector) -> Matrix{eltype(p_vec)}

Returns an `n_v × T_pad` padded matrix (all columns identical at SS) suitable
for the compiled residuals function. Use `get_xVals(asm, p_vec)` to obtain just
the length-`n_v` vector without tiling.
"""
struct SSAssembler{M, SS}
    model::M
    ss_spec::SS                             # SteadyStateSpec for this solve (initial or ending)
    all_keys::Tuple{Vararg{Symbol}}
    free_keys::Tuple{Vararg{Symbol}}
    n_free::Int
    Λ_exog::SparseMatrixCSC{Float64, Int}  # time-invariant exogenous Kronecker factor
    endog_dim::HeterogeneityDimension       # the (single supported) endogenous dimension
    n_exog::Int                             # total number of exogenous states
end

"""
    SSAssembler(model::SequenceModel, ss_spec::SteadyStateSpec) -> SSAssembler

Derives variable roles from `ss_spec.fixed` and precomputes `Λ_exog`.
`ss_spec` is either `model.ss_initial` or `model.ss_ending`.
"""
function SSAssembler(model::SequenceModel, ss_spec)
    pin_keys  = keys(ss_spec.fixed)
    all_keys  = var_names(model)
    free_keys = Tuple(k for k in vars_of_type(model, :endogenous) if !(k in pin_keys))

    endog_dims = [(n, d) for (n, d) in pairs(model.heterogeneity) if d.dim_type == :endogenous]
    exog_dims  = [(n, d) for (n, d) in pairs(model.heterogeneity) if d.dim_type == :exogenous]
    length(endog_dims) == 1 ||
        error("SSAssembler: exactly one endogenous heterogeneity dimension is currently supported")
    endog_dim = endog_dims[1][2]
    n_exog    = isempty(exog_dims) ? 1 : prod(d.n for (_, d) in exog_dims)

    # Precompute the time-invariant exogenous Kronecker factor (Float64, model-fixed).
    # Computed once here; avoids repeating the Kronecker product on every Newton call.
    Λ_exog = spdiagm(0 => ones(Float64, endog_dim.n))
    for (_, dim) in exog_dims
        Λ_exog = kron(sparse(copy((dim.transition::Matrix{Float64})')), Λ_exog)
    end

    return SSAssembler(model, ss_spec, all_keys, free_keys, length(free_keys),
                       Λ_exog, endog_dim, n_exog)
end


"""
    get_xVals(asm::SSAssembler, p_vec::AbstractVector) -> (xVals, ss_value)

Returns the full length-`n_v` aggregate variable vector for iterate `p_vec`,
plus the converged steady-state marginal value matrix.

The heterogeneous variable rows are filled by running an inner VFI loop
on `model.value_fn` until the `Value` field converges. The initial guess
for the marginal value is the endogenous grid repeated across exogenous states
(model-agnostic). The distribution D is computed from the converged policy.

Returns a tuple `(xVals, ss_value)` where:
- `xVals`: length-n_v vector (carries ForwardDiff partials when p_vec is Dual)
- `ss_value`: converged marginal value matrix (same element type as xVals)
"""
function get_xVals(asm::SSAssembler, p_vec::AbstractVector)
    @unpack model, ss_spec, all_keys, free_keys, Λ_exog, endog_dim, n_exog = asm
    n_v   = model.compspec.n_v
    ε     = model.compspec.ε
    T_num = eltype(p_vec)
    xVals = zeros(T_num, n_v)

    # Free endogenous — carry ForwardDiff partials
    for (i, k) in enumerate(free_keys)
        xVals[findfirst(==(k), all_keys)] = p_vec[i]
    end

    # Pinned variables — Float64 constant, zero partials
    for (sym, val) in pairs(ss_spec.fixed)
        xVals[findfirst(==(sym), all_keys)] = val
    end

    # ── Inner VFI loop ────────────────────────────────────────────────────────
    # Initial guess: ones matrix.  A constant marginal value makes `impliedstate`
    # strictly increasing in a' on the first EGM call (since cmat is then constant
    # and a' is the wealth grid), avoiding non-monotone-knot errors at startup.
    value     = ones(eltype(xVals), endog_dim.n, n_exog)
    result_nt = model.value_fn(value, xVals, model)
    for _ in 1:10_000
        value_new = result_nt.Value
        tol_vfi   = maximum(abs.(ForwardDiff.value.(value_new) .-
                                 ForwardDiff.value.(value)))
        value     = value_new
        tol_vfi < ε && break
        result_nt = model.value_fn(value, xVals, model)
    end

    # ── Aggregate heterogeneous variables against the stationary distribution ─
    het_keys        = vars_of_type(model, :heterogeneous)
    policy_var_name = endog_dim.policy_var
    Λ_endog         = make_endogenous_transition(result_nt[policy_var_name], endog_dim, n_exog)
    D               = invariant_dist((Λ_exog * Λ_endog)')

    for varname in het_keys
        xVals[findfirst(==(varname), all_keys)] = dot(vec(result_nt[varname]), D)
    end

    return xVals, result_nt.Value
end


"""
    (asm::SSAssembler)(p_vec::AbstractVector) -> Matrix

Assembles the padded `n_v × T_pad` matrix from `p_vec`. All columns are
identical (steady-state convention). The valid-period slice in the compiled
residuals function returns exactly `n_eq` residuals for one SS period.
"""
function (asm::SSAssembler)(p_vec::AbstractVector)
    @unpack model = asm
    @unpack n_v, max_lag, max_lead = model.compspec
    T_pad = 1 + max_lag + max_lead
    xVals, _ = get_xVals(asm, p_vec)    # discard ss_value; only needed at final extraction
    return reshape(repeat(xVals, T_pad), n_v, T_pad)
end


"""
    find_ss(model::SequenceModel, ss_spec, label::String) -> SteadyState

Finds a single steady state of the model specified by `ss_spec` using a
Newton-Raphson outer loop over the free endogenous variables. An inner VFI
loop (inside `get_xVals`) iterates `model.value_fn` at each outer step until
the marginal value converges.

`label` is a human-readable string (e.g. `"initial"`, `"ending"`) used only
in progress messages and warnings.
"""
function find_ss(model::SequenceModel, ss_spec, label::String)
    asm  = SSAssembler(model, ss_spec)
    F(p) = Residuals(asm(p), model)
    p    = Float64[get(ss_spec.guesses, k, 1.0) for k in asm.free_keys]

    ε        = model.compspec.ε
    z        = F(p)
    iter     = 0
    max_iter = 100
    while norm(z) > ε && iter < max_iter
        println("  [$label] Iteration $iter: residual norm = $(norm(z))")
        J    = ForwardDiff.jacobian(F, p)
        step = J \ z
        η      = 1.0
        z_norm = norm(z)
        safe_eval(q) = try F(q) catch; fill(Inf, length(z)) end
        p_new  = p .- η .* step
        z_new  = safe_eval(p_new)
        while !isfinite(norm(z_new)) || norm(z_new) > z_norm
            η    /= 2
            η > 1e-8 || break
            p_new = p .- η .* step
            z_new = safe_eval(p_new)
        end
        p = p_new
        z = z_new
        iter += 1
    end
    iter == max_iter &&
        @warn "find_ss [$label]: did not converge in $max_iter iterations " *
              "(residual norm: $(norm(z)))"

    # Extract final SS values and the converged marginal value
    xVals_raw, ss_value_raw = get_xVals(asm, p)
    xVals_ss = Float64.(xVals_raw)
    ss_value = Float64.(ss_value_raw)
    vars     = NamedTuple{asm.all_keys}(Tuple(xVals_ss))

    # One more value_fn call with Float64 inputs to get clean Float64 policies
    het_keys  = vars_of_type(model, :heterogeneous)
    result_ss = model.value_fn(ss_value, xVals_ss, model)
    policies  = NamedTuple{het_keys}(Tuple(result_ss[k] for k in het_keys))

    Λ_endog = make_endogenous_transition(
        policies[asm.endog_dim.policy_var], asm.endog_dim, asm.n_exog)
    Λss = asm.Λ_exog * Λ_endog
    D   = invariant_dist(Λss')

    return SteadyState(vars, policies, Λss, D, ss_value)
end


"""
    get_SteadyStates(model::SequenceModel) -> (SteadyState, SteadyState)

Finds both the initial and ending steady states of the model by calling
`find_ss` for each `SteadyStateSpec`. Returns a tuple `(ss_initial, ss_ending)`.

If `model.ss_initial === model.ss_ending` (transitory shock), only one solve
is performed and the same `SteadyState` object is returned for both.
"""
function get_SteadyStates(model::SequenceModel)
    println("=== Finding initial steady state ===")
    ss_initial = find_ss(model, model.ss_initial, "initial")

    # If both specs are the same object, skip the second solve
    if model.ss_initial === model.ss_ending
        println("=== Initial = ending steady state (transitory shock) ===")
        return ss_initial, ss_initial
    end

    println("=== Finding ending steady state ===")
    ss_ending = find_ss(model, model.ss_ending, "ending")

    return ss_initial, ss_ending
end

"""
    SingleRun(ss_initial::SteadyState, ss_ending::SteadyState,
              model::SequenceModel) -> Vector

Runs the complete forward pass (backward iteration → forward iteration →
residuals) from the initial steady state, using a constant sequence equal to
`ss_initial.vars` as the starting guess for all endogenous variables.

Generates exogenous paths via `model.variables[key].seq_fn`, so results are
stochastic for random exogenous processes.
"""
function SingleRun(ss_initial::SteadyState,
                   ss_ending::SteadyState,
                   model::SequenceModel)
    @unpack T, n_endog = model.compspec

    endog_keys  = vars_of_type(model, :endogenous)
    xVec_endog  = repeat(Float64[ss_initial.vars[k] for k in endog_keys], T - 1)

    exog_paths  = generate_exog_paths(model, T - 1)
    policy_seqs = BackwardIteration(xVec_endog, exog_paths, model, ss_ending)
    agg_seqs    = ForwardIteration(policy_seqs, model, ss_initial)
    padded_xMat = assemble_full_xMat(xVec_endog, agg_seqs, exog_paths,
                                     model, ss_initial, ss_ending)
    return Residuals(padded_xMat, model)
end


"""
    directJVPJacobian(mod, ss_initial, ss_ending) -> SparseMatrixCSC

Computes the first `n_endog` columns of the sequence-space Jacobian using
forward-mode JVPs (one JVP per endogenous variable at t=1). Useful for
debugging and AD validation.
"""
function directJVPJacobian(mod, ss_initial, ss_ending)
    @unpack T, n_endog = mod.compspec
    n       = n_endog * (T - 1)
    idmat   = sparse(1.0I, n, n)

    endog_keys = vars_of_type(mod, :endogenous)
    xVec_endog = repeat(Float64[ss_initial.vars[k] for k in endog_keys], T - 1)
    exog_paths = generate_exog_paths(mod, T - 1)

    dirJacobian = spzeros(n_endog * (T - 1), n_endog)

    function fullFunction(x_Vec::AbstractVector)
        policy_seqs = BackwardIteration(x_Vec, exog_paths, mod, ss_ending)
        agg_seqs    = ForwardIteration(policy_seqs, mod, ss_initial)
        padded_xMat = assemble_full_xMat(x_Vec, agg_seqs, exog_paths,
                                         mod, ss_initial, ss_ending)
        return Residuals(padded_xMat, mod)
    end

    for i in 1:n_endog
        dirJacobian[:, i] = JVP(fullFunction, xVec_endog, idmat[:, i])
    end

    return dirJacobian
end


"""
    directNumJacobian(mod, ss_initial, ss_ending) -> SparseMatrixCSC

Computes the first `n_endog` columns of the sequence-space Jacobian via
finite differences. Useful as a reference for validating `directJVPJacobian`.
"""
function directNumJacobian(mod, ss_initial, ss_ending)
    @unpack T, n_endog = mod.compspec
    n       = n_endog * (T - 1)
    idmat   = sparse(1.0I, n, n)

    endog_keys = vars_of_type(mod, :endogenous)
    xVec_endog = repeat(Float64[ss_initial.vars[k] for k in endog_keys], T - 1)
    exog_paths = generate_exog_paths(mod, T - 1)

    dirJacobian = spzeros(n, n_endog)

    function fullFunction(x_Vec::AbstractVector)
        policy_seqs = BackwardIteration(x_Vec, exog_paths, mod, ss_ending)
        agg_seqs    = ForwardIteration(policy_seqs, mod, ss_initial)
        padded_xMat = assemble_full_xMat(x_Vec, agg_seqs, exog_paths,
                                         mod, ss_initial, ss_ending)
        return Residuals(padded_xMat, mod)
    end

    fullX = fullFunction(xVec_endog)

    for i in 1:n_endog
        xDiff = xVec_endog + (1e-4 * idmat[:, i])
        dirJacobian[:, i] = fullFunction(xDiff) - fullX
    end

    return dirJacobian ./ 1e-4
end
