# Description: This file contains the functions to obtain the steady state of the model.
# Uses a modified Newton-Raphson method with the Moore-Penrose pseudoinverse.


"""
    get_SteadyState(model::SequenceModel; initial_guess=nothing) -> SteadyState

Computes the steady state of the model using a modified Newton-Raphson iteration:

    p_{i+1} = p_i - J† z_i

where J† is the Moore-Penrose pseudoinverse (implemented via Julia's `\\` on the
tall/square Jacobian from ForwardDiff).

## Variable roles at steady state

| Role               | How determined                                      |
|--------------------|-----------------------------------------------------|
| `:exogenous`       | Pinned from `model.ss_pin_vals` (TOML `[InitialSteadyState]`) |
| `:heterogeneous`   | Computed: `steadystate_fn` → policy, then distribution aggregate |
| `:endogenous` (pinned) | Pinned from `model.ss_pin_vals`                |
| `:endogenous` (free)   | Newton search variables                        |

The internal function `F : R^{n_free} → R^{n_eq}` assembles the full variable
vector, evaluates the compiled residuals on an `n_v × 1` matrix (steady-state
trick: lags/leads collapse to the current value), and returns the residual vector.

`initial_guess` can be a NamedTuple with values for the free endogenous variables,
e.g. `(Y=1.0, KS=1.0, r=0.02, w=0.1)`. If `nothing`, defaults to `ones(n_free)`.
"""
function get_SteadyState(model::SequenceModel; initial_guess = nothing)
    ε   = model.compspec.ε
    n_v = model.compspec.n_v

    # Variables pinned from [InitialSteadyState]: both exogenous and any pinned endogenous
    pin_keys = keys(model.ss_pin_vals)

    # Free endogenous variables: :endogenous vars NOT listed in ss_pin_vals
    free_keys = Tuple(k for k in vars_of_type(model, :endogenous) if !(k in pin_keys))
    n_free    = length(free_keys)

    all_keys = var_names(model)   # ordered symbol tuple, matches xMat rows

    # ─────────────────────────────────────────────────────────────────────────
    # assemble_xMat: builds an (n_v × 1) matrix from the free-variable vector.
    #
    # 1. Insert free endogenous values (from Newton iterate p_vec).
    # 2. Insert pinned values (from model.ss_pin_vals) — zero derivatives.
    # 3. For each :heterogeneous variable, call its steadystate_fn to get the
    #    household policy, build the transition matrix, find the stationary
    #    distribution, and aggregate.
    # The n_v × 1 output feeds directly into Residuals, exploiting the lag/lead
    # identity at steady state (shift_lag on a length-1 vector is a no-op).
    # ─────────────────────────────────────────────────────────────────────────
    function assemble_xMat(p_vec::AbstractVector)
        T_num = eltype(p_vec)
        xVals = zeros(T_num, n_v)

        # Free endogenous variables
        for (i, k) in enumerate(free_keys)
            idx = findfirst(==(k), all_keys)
            xVals[idx] = p_vec[i]
        end

        # Pinned variables (exogenous + any pinned endogenous); value converted to T_num
        for (sym, val) in pairs(model.ss_pin_vals)
            idx = findfirst(==(sym), all_keys)
            xVals[idx] = val   # implicit convert Float64 → T_num (zero derivatives)
        end

        # Aggregated (heterogeneous) variables: policy iteration → distribution → dot
        for (varname, var) in pairs(model.variables)
            var.var_type == :heterogeneous || continue
            policy = var.steadystate_fn(xVals, model)
            Λ      = make_ss_transition(policy, model)
            D      = invariant_dist(Λ')          # Λ is column-stochastic; pass Λ'
            idx    = findfirst(==(varname), all_keys)
            xVals[idx] = dot(vec(policy), D)
        end

        return reshape(xVals, n_v, 1)
    end

    # F : R^{n_free} → R^{n_eq}
    F(p_vec) = Residuals(assemble_xMat(p_vec), model)

    # Initial guess for free endogenous variables
    p = if isnothing(initial_guess)
        ones(Float64, n_free)
    else
        Float64[initial_guess[k] for k in free_keys]
    end

    # Modified Newton: p_{i+1} = p_i - J† z_i
    # J is (n_eq × n_free); J \ z is the least-squares / exact solution.
    z        = F(p)
    iter     = 0
    max_iter = 1000
    while norm(z) > ε && iter < max_iter
        J = ForwardDiff.jacobian(F, p)
        p = p .- J \ z
        z = F(p)
        iter += 1
    end
    iter == max_iter &&
        @warn "get_SteadyState: did not converge in $max_iter iterations (residual norm: $(norm(z)))"

    # Build SteadyState from converged values
    xVals_final = vec(assemble_xMat(p))
    vars        = NamedTuple{all_keys}(Tuple(Float64.(xVals_final)))

    het_keys = vars_of_type(model, :heterogeneous)
    policies_list = map(het_keys) do varname
        model.variables[varname].steadystate_fn(Float64.(xVals_final), model)
    end
    policies = NamedTuple{het_keys}(Tuple(policies_list))

    # Transition matrix and stationary distribution from the first heterogeneous variable
    first_policy = policies[het_keys[1]]
    Λss = make_ss_transition(first_policy, model)
    D   = invariant_dist(Λss')

    return SteadyState(vars, policies, Λss, D)
end


# Test Run

function test_SteadyState()
    # Build variables NamedTuple: endogenous → heterogeneous → exogenous
    variables = (
        Y  = Variable(:Y,  :endogenous,    "Output"),
        KS = Variable(:KS, :endogenous,    "Capital supply"),
        r  = Variable(:r,  :endogenous,    "Interest rate"),
        w  = Variable(:w,  :endogenous,    "Wages"),
        KD = Variable(:KD, :heterogeneous, "Capital demand (aggregated from HH)",
                      backward_capital, steadystate_capital),
        Z  = Variable(:Z,  :exogenous,     "Productivity"),
    )

    equations = (
        "Y = Z * KS(-1)^α",
        "r + δ = α * Z * KS(-1)^(α-1)",
        "w = (1-α) * Z * KS(-1)^α",
        "KS = KD",
    )

    # Computational specs
    compspec = ComputationalSpec(150, 1e-9, 0.0001, length(variables))

    # Economic parameters as NamedTuple
    params = (β = 0.98, γ = 1.0, δ = 0.025, α = 0.11)

    # Compile equations
    param_names = Set(keys(params))
    union!(param_names, Set([:T, :ε, :dx, :n_v]))
    residuals_fn = compile_residuals(collect(equations), keys(variables), param_names)

    # Build heterogeneity dimensions
    wealth_config = Dict("type" => "endogenous", "grid_method" => "DoubleExponential",
                         "n" => 200, "bounds" => [0.0, 200.0], "policy_var" => "KD")
    prod_config = Dict("type" => "exogenous", "discretization" => "Rouwenhorst",
                       "n" => 7, "ρ" => 0.966, "σ" => 0.283)
    heterogeneity = (wealth = build_dimension(wealth_config),
                     productivity = build_dimension(prod_config))

    # Pinned steady-state values from [InitialSteadyState] (Z is exogenous)
    ss_pin_vals = (Z = 1.0,)

    mod = SequenceModel(variables, equations, compspec, params, residuals_fn, ss_pin_vals, heterogeneity)

    # Obtain steady state (initial_guess for free endogenous variables only; Z is pinned)
    ss = get_SteadyState(mod, initial_guess = (Y = 1.0, KS = 1.0, r = 0.02, w = 0.1))

    return mod, ss
end


"""
    SingleRun(ss::SteadyState,
    model::SequenceModel)

Runs the entire sequence of functions for a single run of the model.
    That is, it calculates the residuals for a given steady state and model.
"""
function SingleRun(ss::SteadyState,
    model::SequenceModel)

    # Unpack parameters
    @unpack T, n_v = model.compspec
    n = (T-1) * n_v
    
    # Initialize vectors
    xVec = repeat([values(ss.vars)...], T-1)

    policy_seqs = BackwardIteration(xVec, model, ss)
    xMat = ForwardIteration(xVec, policy_seqs, model, ss)
    zVals = Residuals(xMat, model)

    return zVals
end


function backFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
    a_seq = BackwardIteration(x_Vec, mod, stst)
    return a_seq 
end


function directJVPJacobian(mod, 
    stst)

    @unpack T, n_v = mod.compspec
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.vars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        policy_seqs = BackwardIteration(x_Vec, mod, stst)
        xMat = ForwardIteration(x_Vec, policy_seqs, mod, stst)
        zVals = Residuals(xMat, mod)
        return zVals
    end

    for i in 1:n_v
        # dirJacobian[:,n - n_v + i] = JVP(fullFunction, xVec, idmat[:, n - n_v + i])
        dirJacobian[:,i] = JVP(fullFunction, xVec, idmat[:,i])
    end

    return dirJacobian
end


function directNumJacobian(mod, 
    stst)

    @unpack T, n_v = mod.compspec
    n = (T - 1) * n_v
    idmat = sparse(1.0I, n, n)
    xVec = repeat([values(stst.vars)...], T-1)
    Zexog = ones(T-1)
    dirJacobian = spzeros(n, n)
    
    function fullFunction(x_Vec::AbstractVector) # (n_v * T-1)-dimensional vector
        policy_seqs = BackwardIteration(x_Vec, mod, stst)
        xMat = ForwardIteration(x_Vec, policy_seqs, mod, stst)
        zVals = Residuals(xMat, mod)
        return zVals
    end

    fullX = fullFunction(xVec)

    for i in 1:n_v
        xDiff = xVec + (1e-4 * idmat[:,i])
        dirJacobian[:,i] = fullFunction(xDiff) - fullX
    end

    return dirJacobian ./ 1e-4
end





