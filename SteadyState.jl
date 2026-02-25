# Description: This file contains the functions to obtain the steady state of the model.
# Uses a modified Newton-Raphson method with the Moore-Penrose pseudoinverse.



"""
    SteadyState

Stores the steady-state solution of the model. Contains the steady-state
values of all aggregate variables, the household policy functions, the
transition matrix, and the stationary distribution over household states.

Fields:
- `vars`: NamedTuple of steady-state variable values, ordered to match
   `model.varXs` (e.g., `(Y=1.0, KS=3.5, r=0.02, w=0.9, Z=1.0)`)
- `policies`: NamedTuple of policy matrices, one per aggregated variable,
   ordered to match `model.agg_vars` (e.g., `(KD = Matrix{Float64},)`)
- `Λ`: sparse transition matrix for the distribution (from `DistributionTransition`)
- `D`: stationary distribution vector (length `n_a * n_e`)
"""
struct SteadyState
    vars::NamedTuple
    policies::NamedTuple
    Λ::SparseMatrixCSC{Float64,Int64}
    D::Vector{Float64}
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
- **Heterogeneous**: computed from `steadystate_fn → Λ → invariant_dist → dot`.

## Calling convention

    asm(p_vec::AbstractVector) -> Matrix{eltype(p_vec)}

Returns an `n_v × T_pad` padded matrix (all columns identical at SS) suitable
for the compiled residuals function. Use `get_xVals(asm, p_vec)` to obtain just
the length-`n_v` vector without tiling.
"""
struct SSAssembler{M}
    model::M
    all_keys::Tuple{Vararg{Symbol}}
    free_keys::Tuple{Vararg{Symbol}}
    n_free::Int
    Λ_exog::SparseMatrixCSC{Float64, Int}  # time-invariant exogenous Kronecker factor
    endog_dim::HeterogeneityDimension       # the (single supported) endogenous dimension
    n_exog::Int                             # total number of exogenous states
end

"""
    SSAssembler(model::SequenceModel) -> SSAssembler

Derives variable roles from `ss_initial.fixed` and precomputes `Λ_exog`.
"""
function SSAssembler(model::SequenceModel)
    pin_keys  = keys(model.ss_initial.fixed)
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

    return SSAssembler(model, all_keys, free_keys, length(free_keys),
                       Λ_exog, endog_dim, n_exog)
end


"""
    get_xVals(asm::SSAssembler, p_vec::AbstractVector) -> Vector

Returns the full length-`n_v` aggregate variable vector for iterate `p_vec`,
without tiling to a padded matrix. Useful for extracting final SS values after
convergence without constructing the full `n_v × T_pad` matrix.

AD-compatible: when `p_vec` carries ForwardDiff dual numbers, derivatives flow
through the free and heterogeneous rows; pinned rows have zero partials.
"""
function get_xVals(asm::SSAssembler, p_vec::AbstractVector)
    @unpack model, all_keys, free_keys, Λ_exog, endog_dim, n_exog = asm
    n_v   = model.compspec.n_v
    T_num = eltype(p_vec)
    xVals = zeros(T_num, n_v)

    # Free endogenous — carry ForwardDiff partials
    for (i, k) in enumerate(free_keys)
        xVals[findfirst(==(k), all_keys)] = p_vec[i]
    end

    # Pinned variables — Float64 constant, zero partials
    for (sym, val) in pairs(model.ss_initial.fixed)
        xVals[findfirst(==(sym), all_keys)] = val
    end

    # The distribution D is determined by the savings policy of the endogenous dimension.
    # Compose the precomputed Λ_exog with the updated Λ_endog to get the full transition.
    # TODO: when multiple endogenous dimensions are supported, compose one Λ_endog per
    # endogenous dim into a joint transition before computing D.
    policy_var_name = endog_dim.policy_var
    policy = model.variables[policy_var_name].steadystate_fn(xVals, model)
    Λ_endog = make_endogenous_transition(policy, endog_dim, n_exog)
    D = invariant_dist((Λ_exog * Λ_endog)')

    # Aggregate all heterogeneous variables against the shared distribution D.
    # Reuse the already-computed policy for policy_var_name to avoid a redundant call.
    for (varname, var) in pairs(model.variables)
        var.var_type == :heterogeneous || continue
        p = (varname === policy_var_name) ? policy : var.steadystate_fn(xVals, model)
        xVals[findfirst(==(varname), all_keys)] = dot(vec(p), D)
    end

    return xVals
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
    xVals = get_xVals(asm, p_vec)
    return reshape(repeat(xVals, T_pad), n_v, T_pad)
end


"""
    get_SteadyState(model::SequenceModel) -> SteadyState

Computes the steady state of the model using a modified Newton-Raphson iteration:

    p_{i+1} = p_i - J† z_i

where `p` is the vector of free endogenous variables, `z = F(p)` are the
equilibrium residuals at the steady-state padded matrix, and `J†` is computed
via Julia's `\\` (least-squares / exact solution for square systems).
Initial guesses come from `ss_initial.guesses` in the YAML, defaulting to 1.
"""
function get_SteadyState(model::SequenceModel)
    asm  = SSAssembler(model)
    F(p) = Residuals(asm(p), model)

    p = Float64[get(model.ss_initial.guesses, k, 1.0) for k in asm.free_keys]

    ε        = model.compspec.ε
    z        = F(p)
    iter     = 0
    max_iter = 100
    while norm(z) > ε && iter < max_iter
        J = ForwardDiff.jacobian(F, p)
        p = p .- J \ z
        z = F(p)
        iter += 1
    end
    iter == max_iter &&
        @warn "get_SteadyState: did not converge in $max_iter iterations (residual norm: $(norm(z)))"

    # Extract full SS values — get_xVals avoids the wasteful T_pad tiling
    xVals_ss = Float64.(get_xVals(asm, p))
    vars     = NamedTuple{asm.all_keys}(Tuple(xVals_ss))

    # Policy matrices at the SS
    het_keys      = vars_of_type(model, :heterogeneous)
    policies_list = map(het_keys) do varname
        model.variables[varname].steadystate_fn(xVals_ss, model)
    end
    policies = NamedTuple{het_keys}(Tuple(policies_list))

    # Λss uses the endogenous dimension's policy_var (savings policy drives the distribution).
    # TODO: when supporting multiple endogenous dimensions, compose multiple Λ_endog matrices
    # into a joint steady-state transition before computing D.
    Λ_endog = make_endogenous_transition(
        policies[asm.endog_dim.policy_var], asm.endog_dim, asm.n_exog)
    Λss = asm.Λ_exog * Λ_endog
    D   = invariant_dist(Λss')

    return SteadyState(vars, policies, Λss, D)
end


"""
    test_SteadyState() -> (SequenceModel, SteadyState)

Builds the KS model from YAML and runs `get_SteadyState`. Useful for
interactive testing of the steady-state solver.
"""
function test_SteadyState()
    mod = build_model_from_yaml("KrusellSmith.yaml")
    ss  = get_SteadyState(mod)
    return mod, ss
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
