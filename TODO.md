# TODO — Julia Newton-Raphson HANK Solver

Items here are deferred work that doesn't block current progress.
Roughly ordered by priority within each section.

---

## YAML Validation (`ModelParser.jl` / `build_model_from_yaml`)

- [ ] Require `steady_states.initial` to be present; raise a clear error if missing
- [ ] Require that every endogenous dimension specifies a `policy_var`
- [ ] Check that every `policy_var` named in a dimension matches a `:heterogeneous` variable
- [ ] Check that every `:heterogeneous` variable has both `backward_function` and `ss_function`
- [ ] Check that every `:exogenous` variable has a `seq_function` (or at least warn if absent)
- [ ] Warn if `steady_states.ending` is missing but the model has distinct initial/ending Z values
- [ ] Validate that all variable names referenced in `equations` are declared in `variables`
- [ ] Validate that all parameter names referenced in `equations` are declared in `parameters.model`
- [ ] Check that dimension `params.n` is a positive integer for all dimensions
- [ ] Check grid bounds are ordered (grid_min < grid_max) for endogenous dimensions

---

## `KrusellSmith.jl` — Fix broken field references (blocks steady-state solve)

- [ ] `ValueFunction`: `model.policygrid` → `model.heterogeneity.wealth.grid`
- [ ] `ValueFunction`: `model.policymat` → `repeat(grid, 1, n_e)`
- [ ] `ValueFunction`: `model.shockmat` → `repeat(prod_grid', n_a, 1)`
- [ ] `ValueFunction`: `model.Π` → `model.heterogeneity.productivity.transition`
- [ ] `backward_capital`: `model.varXs` → `var_names(model)`
- [ ] `backward_capital`: `model.params.n_e` → `model.heterogeneity.productivity.n`

---

## `SteadyStateJacobian.jl` — Fix broken field references

- [ ] All uses of `model.agg_vars` → `vars_of_type(model, :heterogeneous)`
- [ ] All uses of `model.params.n_a`, `model.params.n_e` → `model.heterogeneity` dims
- [ ] Resolve open TODOs on `dfbydxlead` in `getFinalJacobian` (lines ~179, ~201)
- [ ] Test AD compatibility of `flatten_policies`/`unflatten_policies`

---

## `NewtonRaphson.jl`

- [ ] Remove assumption that initial and ending steady states are identical
  (TODOs on lines ~7 and ~37)
- [ ] Implement `alphaUpdate` (currently a stub)

---

## Longer-term / design

- [ ] **Multi-output backward functions**: currently each `backward_function` is assumed
  to return the policy for exactly the one `:heterogeneous` variable it is attached to.
  In richer models a single household solve returns multiple policy functions at once
  (e.g., both consumption *and* labor). Design needed:
  - Allow a `backward_function` to be shared across several `:heterogeneous` variables,
    or alternatively return a NamedTuple of policies keyed by variable name.
  - YAML syntax suggestion: a `backward_function` listed on one variable can name the
    other variables whose policies it also produces (a `coproduces` field, or a shared
    `backward_group`).
  - `BackwardIteration.jl` and `SteadyState.jl` both iterate over heterogeneous vars
    independently; they would need to be aware of which variables are co-produced so the
    backward function is called only once per period and its outputs routed to the right
    policy sequences.
- [ ] Move all KS-specific functions out of `KrusellSmith.jl` into `ks_model_functions.jl`
  so that `KrusellSmith.jl` only contains the `SteadyState` struct
- [ ] Support multiple endogenous heterogeneity dimensions in `ForwardIteration`
  (currently raises an error)
- [ ] Connect `seq_fn` on exogenous variables to the actual exogenous path used in
  `NewtonRaphsonHANK` (currently path is passed manually as `Zexog`)
