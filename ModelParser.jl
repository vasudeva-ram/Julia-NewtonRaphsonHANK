# ─────────────────────────────────────────────────────────────────────────────
# ModelParser.jl — YAML parsing, equation compilation, and model construction
#
# This file handles the translation from user-facing model specification
# (a .yaml file) to the internal Julia representation (SequenceModel).
# It contains two layers:
#
# 1. Equation compilation: transform_expr, compile_residuals — a
#    metaprogramming system that converts string equations like
#    "Y = Z * KS(-1)^α" into compiled Julia functions that operate on
#    the n_v × (T-1) variable matrix. This is done at model construction
#    time via `eval`, so the resulting function is regular Julia code that
#    is fully compatible with automatic differentiation (ForwardDiff, Zygote).
#
# 2. Model construction: build_model_from_yaml — the main entry point that
#    reads a YAML file, auto-includes the model's function file, builds
#    heterogeneity dimensions by calling user-specified (or built-in) grid
#    functions, compiles equations, and returns a complete SequenceModel.
#
# Grid function contract (enforced with descriptive errors):
#   - endogenous dimension: fn(; params...) → Vector{Float64} of length n
#   - exogenous dimension:  fn(; params...) → (Vector{Float64}, Matrix{Float64})
#                           i.e. (grid of length n, n×n transition matrix)
# ─────────────────────────────────────────────────────────────────────────────

using YAML


# ─────────────────────────────────────────────────────────────────────────────
# Equation compilation: string equations → compiled residuals function
# ─────────────────────────────────────────────────────────────────────────────

# Arithmetic operators that should be broadcast for element-wise operations
const BROADCAST_OPS = Set([:+, :-, :*, :/, :^])


"""
    transform_expr(expr, var_indices::Dict{Symbol,Int}, param_names::Set{Symbol})

Recursively walks a Julia AST and transforms it for vectorized evaluation
over the time dimension. The transformation rules are:

- **Variables** (symbols in `var_indices`): `KS` → `xMat[2, :]` (row slice)
- **Lag notation**: `KS(-1)` → `shift_lag(xMat[2, :], 1)` (Julia parses
  `KS(-1)` as `Expr(:call, :KS, -1)`, which we intercept)
- **Lead notation**: `KS(+1)` → `shift_lead(xMat[2, :], 1)`
- **Parameters** (symbols in `param_names`): `α` → `params.α`
- **Arithmetic ops** (`+`, `-`, `*`, `/`, `^`): converted to broadcasting
  versions (`.+`, `.-`, etc.) so they operate element-wise over time
- **Other functions** (e.g., `log`, `exp`): arguments are transformed but
  the function call itself is left unchanged
- **Numbers and unknown symbols**: passed through unchanged
"""
function transform_expr(expr, var_indices::Dict{Symbol,Int}, param_names::Set{Symbol})
    # Numbers pass through unchanged
    if expr isa Number
        return expr

    # Symbols: variable references, parameter references, or Julia built-ins
    elseif expr isa Symbol
        if haskey(var_indices, expr)
            idx = var_indices[expr]
            return :(xMat[$idx, :])
        elseif expr in param_names
            return :(params.$expr)
        else
            return expr  # Julia built-in (e.g., log, exp, pi)
        end

    # Expressions: function calls, operators, etc.
    elseif expr isa Expr
        if expr.head == :call
            func = expr.args[1]

            # Check for VAR(±i) lag/lead notation:
            # Julia parses KS(-1) as Expr(:call, :KS, -1)
            if func isa Symbol && haskey(var_indices, func) && length(expr.args) == 2
                lag_val = expr.args[2]
                if lag_val isa Integer
                    idx = var_indices[func]
                    if lag_val < 0
                        return :(shift_lag(xMat[$idx, :], $(abs(lag_val))))
                    elseif lag_val > 0
                        return :(shift_lead(xMat[$idx, :], $lag_val))
                    else
                        return :(xMat[$idx, :])
                    end
                end
            end

            # Broadcast arithmetic operators
            if func isa Symbol && func in BROADCAST_OPS
                broadcast_func = Symbol(".", func)
                transformed_args = [transform_expr(a, var_indices, param_names) for a in expr.args[2:end]]
                # Fold n-ary operators into nested binary calls
                if length(transformed_args) == 1
                    # Unary (e.g., negation)
                    return Expr(:call, broadcast_func, transformed_args[1])
                else
                    result = Expr(:call, broadcast_func, transformed_args[1], transformed_args[2])
                    for i in 3:length(transformed_args)
                        result = Expr(:call, broadcast_func, result, transformed_args[i])
                    end
                    return result
                end
            end

            # Other function calls (e.g., log, exp, sqrt): transform arguments only
            transformed_args = [func; [transform_expr(a, var_indices, param_names) for a in expr.args[2:end]]]
            return Expr(:call, transformed_args...)
        else
            # Other expression types (blocks, parens, etc.): recurse into args
            new_args = [transform_expr(a, var_indices, param_names) for a in expr.args]
            return Expr(expr.head, new_args...)
        end
    else
        return expr
    end
end


"""
    detect_max_lag_lead(equations::Vector{String},
                        var_syms) -> (max_lag::Int, max_lead::Int)

Walks the ASTs of all `equations` and returns the maximum lag depth and the
maximum lead depth found across every variable reference.

For example, given `"Y = Z * KS(-1)^α"` the function returns `(1, 0)` since
the deepest lag is 1 and there are no leads. For `"C(+2) = r(-3) * Y"` it
would return `(3, 2)`.

These values determine how many boundary-condition columns must be prepended
(initial SS, depth `max_lag`) and appended (ending SS, depth `max_lead`) to
the padded xMat passed to the compiled residuals function.
"""
function detect_max_lag_lead(equations::Vector{String}, var_syms)
    var_set  = Set(var_syms)
    max_lag  = Ref(0)
    max_lead = Ref(0)

    function walk(expr)
        expr isa Expr || return
        if expr.head == :call
            func = expr.args[1]
            # VAR(±i) pattern: Expr(:call, :VAR, lag_integer)
            if func isa Symbol && func in var_set &&
               length(expr.args) == 2 && expr.args[2] isa Integer
                lag_val = expr.args[2]
                if lag_val < 0
                    max_lag[]  = max(max_lag[],  abs(lag_val))
                elseif lag_val > 0
                    max_lead[] = max(max_lead[], lag_val)
                end
                return  # no need to recurse into a terminal lag/lead node
            end
        end
        for a in expr.args
            walk(a)
        end
    end

    for eq_str in equations
        parts = split(eq_str, "="; limit=2)
        length(parts) == 2 || continue
        for part in parts
            walk(Meta.parse(strip(String(part))))
        end
    end

    return max_lag[], max_lead[]
end


"""
    compile_residuals(equations::Vector{String}, var_syms::Tuple{Vararg{Symbol}},
                      param_names::Set{Symbol})

Compiles a vector of equation strings into a single Julia function
`(xMat::AbstractMatrix, params) -> Vector` that evaluates the residuals.

Each equation has the form `"LHS = RHS"`. The residual for each equation
is `LHS .- RHS`, evaluated element-wise over the time dimension (columns
of `xMat`).

## Padded-matrix convention

The compiled function expects `xMat` to have `T_pad = (T-1) + max_lag + max_lead`
columns, where the first `max_lag` columns are initial-SS boundary values and
the last `max_lead` columns are ending-SS boundary values. The function
computes residuals over all `T_pad` columns, then slices to the valid middle
range `(max_lag+1):(T_pad - max_lead)`, returning exactly `n_eq × (T-1)`
values. This ensures that `shift_lag` and `shift_lead` always read the correct
steady-state boundary rather than an arbitrary transition value.

`max_lag` and `max_lead` are detected automatically from the equation ASTs via
`detect_max_lag_lead` and are baked into the compiled function as closure
constants — the caller only needs to pass a correctly sized padded `xMat`.

## Compilation pipeline
1. Parse each equation string into a Julia AST via `Meta.parse`
2. Transform the AST via `transform_expr` (variable → row slice, params → field access, etc.)
3. Assemble all transformed equations + the valid-range slice into a single
   anonymous function body
4. `eval` the function at construction time → returns a regular Julia function

The resulting function is ordinary compiled Julia code (no runtime `eval`),
so it is fully compatible with ForwardDiff and Zygote for automatic differentiation.

Variables are identified by matching symbols against `var_syms` (pass `var_names(model)`
or `keys(model.variables)` to get the ordered symbol tuple from a built model).
Parameters are identified by matching against the explicitly provided `param_names`.

The output vector is ordered: all equations at t=1, all equations at t=2, ...
(column-major vectorization of a k × (T-1) residual matrix, after slicing).
"""
function compile_residuals(equations::Vector{String}, var_syms::Tuple{Vararg{Symbol}},
                           param_names::Set{Symbol})
    var_indices = Dict(sym => i for (i, sym) in enumerate(var_syms))

    # Detect lag/lead depths and bake as closure constants
    max_lag, max_lead = detect_max_lag_lead(equations, var_syms)
    slice_lo = max_lag + 1          # first valid column index (1-based)

    # Generate a unique symbol for each equation's residual
    residual_syms = [Symbol("_r_", i) for i in 1:length(equations)]

    # Build the function body: one assignment per equation
    body_exprs = Expr[]

    for (i, eq_str) in enumerate(equations)
        parts = split(eq_str, "="; limit=2)
        if length(parts) != 2
            error("Equation must contain exactly one '=': $eq_str")
        end

        lhs_parsed = Meta.parse(strip(String(parts[1])))
        rhs_parsed = Meta.parse(strip(String(parts[2])))

        lhs_transformed = transform_expr(lhs_parsed, var_indices, param_names)
        rhs_transformed = transform_expr(rhs_parsed, var_indices, param_names)

        # _r_i = LHS .- RHS  (over all T_pad columns)
        push!(body_exprs, :($(residual_syms[i]) = $(Expr(:call, :.-, lhs_transformed, rhs_transformed))))
    end

    # Slice each residual to the valid middle range, then stack column-major.
    # slice_lo and max_lead are closure constants baked in at compile time.
    trans_exprs = [:($(s)[$slice_lo:(size(xMat, 2) - $max_lead)]') for s in residual_syms]
    push!(body_exprs, :(return vec(reduce(vcat, [$(trans_exprs...)]))))

    fn_body = Expr(:block, body_exprs...)

    fn_expr = :(function(xMat::AbstractMatrix, params)
        $fn_body
    end)

    return eval(fn_expr)
end


# ─────────────────────────────────────────────────────────────────────────────
# Full model construction from YAML
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_model_from_yaml(file_path::String) -> SequenceModel

Constructs a complete `SequenceModel` from a YAML specification file.

Steps performed:
1. Parse the YAML file.
2. `include` the model's `function_file` (e.g. `ks_model_functions.jl`), making
   all model-specific Julia functions available by name.
3. Build heterogeneity dimensions by calling each dimension's `grid_function`
   with the `params` dict as keyword arguments. Return values are validated:
   - endogenous: must return a `Vector` of length `n`
   - exogenous:  must return a 2-tuple `(grid::Vector, Π::Matrix)`, both of
                 size consistent with `n`
4. Build `Variable` objects (endogenous → heterogeneous → exogenous). Functions
   for heterogeneous and exogenous variables are looked up by name in `Main`.
5. Build `ComputationalSpec` from `parameters.computational` (defaults: T=150,
   ε=1e-6, dx=1e-8 if not provided).
6. Build `params` NamedTuple from `parameters.model`.
7. Compile equilibrium equations via `compile_residuals`.
8. Parse `steady_states.initial` and (optionally) `steady_states.ending` into
   `SteadyStateSpec` objects. If only `initial` is present, `ss_ending = ss_initial`
   (transitory shock assumption).
9. Return a `SequenceModel`.

# Example
```julia
mod = build_model_from_yaml("KrusellSmith.yaml")
```
"""
function build_model_from_yaml(file_path::String)
    yaml = YAML.load_file(file_path)
    dir  = dirname(abspath(file_path))

    # ── 0. Include the model function file ────────────────────────────────────
    func_file = yaml["file"]["function_file"]
    include(joinpath(dir, func_file))

    # ── 1. Parse parameters ───────────────────────────────────────────────────
    # Economic (model) parameters
    model_params_list = yaml["parameters"]["model"]
    pnames  = Tuple(Symbol(p["name"]) for p in model_params_list)
    pvalues = Tuple(_parse_number(p["value"]) for p in model_params_list)
    params  = NamedTuple{pnames}(pvalues)

    # Computational parameters (with defaults)
    defaults = Dict("T" => 150, "ε" => 1e-6, "dx" => 1e-8)
    comp_list = get(get(yaml, "parameters", Dict()), "computational", [])
    cs = Dict(p["name"] => p["value"] for p in comp_list)
    T  = Int(get(cs, "T",  defaults["T"]))
    ε  = Float64(get(cs, "ε",  defaults["ε"]))
    dx = Float64(get(cs, "dx", defaults["dx"]))

    # ── 2. Build heterogeneity dimensions ─────────────────────────────────────
    dims_raw   = yaml["dimensions"]
    dim_names  = Tuple(Symbol(d["name"]) for d in dims_raw)
    dim_values = Tuple(_build_dimension_from_yaml(d) for d in dims_raw)
    heterogeneity = NamedTuple{dim_names}(dim_values)

    # ── 3. Build Variable objects: endogenous → heterogeneous → exogenous ─────
    vars_section = yaml["variables"]

    endog_vars = [
        Variable(Symbol(v["name"]), :endogenous, get(v, "description", ""))
        for v in get(vars_section, "endogenous", [])
    ]

    hetero_vars = map(get(vars_section, "heterogeneous", [])) do v
        sym  = Symbol(v["name"])
        desc = get(v, "description", "")
        bfn  = _lookup_fn(v["backward_function"])
        ssfn = _lookup_fn(v["ss_function"])
        Variable(sym, :heterogeneous, desc, bfn, ssfn, nothing)
    end

    exog_vars = map(get(vars_section, "exogenous", [])) do v
        sym  = Symbol(v["name"])
        desc = get(v, "description", "")
        seq  = haskey(v, "seq_function") ? _lookup_fn(v["seq_function"]) : nothing
        Variable(sym, :exogenous, desc, nothing, nothing, seq)
    end

    all_var_list  = [endog_vars..., hetero_vars..., exog_vars...]
    all_var_names = Tuple(v.name for v in all_var_list)
    variables     = NamedTuple{all_var_names}(Tuple(all_var_list))
    n_endog       = length(endog_vars)

    # ── 4. Compile equations ──────────────────────────────────────────────────
    equations   = Tuple(String(e) for e in yaml["equations"])
    param_names = Set{Symbol}(pnames)
    union!(param_names, Set([:T, :ε, :dx, :n_v]))
    max_lag, max_lead = detect_max_lag_lead(collect(equations), all_var_names)
    residuals_fn = compile_residuals(collect(equations), all_var_names, param_names)

    compspec = ComputationalSpec(T, ε, dx, length(variables), n_endog, max_lag, max_lead)

    # ── 5. Parse steady states ────────────────────────────────────────────────
    ss_section = yaml["steady_states"]
    ss_initial = _parse_ss_spec(ss_section["initial"])
    ss_ending  = haskey(ss_section, "ending") ?
                     _parse_ss_spec(ss_section["ending"]) : ss_initial

    return SequenceModel(variables, equations, compspec, params,
                         residuals_fn, ss_initial, ss_ending, heterogeneity)
end


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _parse_number(v) -> Int or Float64

Converts a YAML scalar to a Julia number. Integer YAML values become `Int`;
floating-point values become `Float64`.
"""
_parse_number(v::Integer)      = Int(v)
_parse_number(v::AbstractFloat) = Float64(v)
_parse_number(v)                = v   # fallback (e.g. strings, though unusual)


"""
    _lookup_fn(name::String) -> Function

Looks up a Julia function by name in `Main`. Throws a descriptive error if the
function is not defined — typically because the function file was not included
or contains a typo.
"""
function _lookup_fn(name::String)
    sym = Symbol(name)
    isdefined(Main, sym) ||
        error("Function '$name' not found in scope. " *
              "Check that it is defined in the function_file specified in your YAML.")
    obj = getfield(Main, sym)
    obj isa Function ||
        error("'$name' is defined but is not a Function (got $(typeof(obj))).")
    return obj
end


"""
    _parse_ss_spec(spec_dict) -> SteadyStateSpec

Converts a YAML steady-state sub-dictionary (with `fixed` and `guesses` keys)
into a `SteadyStateSpec`.
"""
function _parse_ss_spec(spec_dict)
    fixed_dict = get(spec_dict, "fixed",   Dict())
    guess_dict = get(spec_dict, "guesses", Dict())

    fixed_keys  = Tuple(Symbol(k) for k in keys(fixed_dict))
    fixed_vals  = Tuple(Float64(v) for v in values(fixed_dict))
    guess_keys  = Tuple(Symbol(k) for k in keys(guess_dict))
    guess_vals  = Tuple(Float64(v) for v in values(guess_dict))

    return SteadyStateSpec(
        NamedTuple{fixed_keys}(fixed_vals),
        NamedTuple{guess_keys}(guess_vals),
    )
end


"""
    _build_dimension_from_yaml(dim_dict) -> HeterogeneityDimension

Builds a `HeterogeneityDimension` from a single YAML dimension entry.

Calls the dimension's `grid_function` with its `params` as keyword arguments,
then validates the return value against the declared dimension type:
- `:endogenous` — grid function must return a `Vector` of length `n`
- `:exogenous`  — grid function must return a 2-tuple `(grid, Π)` with
                  `grid` a length-`n` Vector and `Π` an `n×n` Matrix

Raises a descriptive error on any shape/type mismatch so the user knows
exactly what their grid function returned vs. what was expected.
"""
function _build_dimension_from_yaml(dim_dict)
    dim_type   = Symbol(dim_dict["type"])
    dim_name   = dim_dict["name"]
    fn_name    = dim_dict["grid_function"]
    params_raw = dim_dict["params"]
    n          = Int(params_raw["n"])
    policy_var = haskey(dim_dict, "policy_var") ? Symbol(dim_dict["policy_var"]) : nothing

    # Look up grid function (built-ins are in Main from GeneralStructures.jl)
    grid_fn = _lookup_fn(fn_name)

    # Call with keyword arguments from the params sub-dict
    kwargs = Dict(Symbol(k) => v for (k, v) in params_raw)
    result = grid_fn(; kwargs...)

    if dim_type == :endogenous
        # ── Validate: must return a 1-D Vector of length n ────────────────────
        result isa AbstractVector ||
            error("Grid function '$fn_name' for endogenous dimension '$dim_name' " *
                  "must return a Vector, got $(typeof(result)).\n" *
                  "Endogenous grid functions should return a single grid vector.")
        length(result) == n ||
            error("Grid function '$fn_name' for endogenous dimension '$dim_name': " *
                  "expected $n grid points (params.n = $n), got $(length(result)).")

        return HeterogeneityDimension(:endogenous, n,
                                      collect(Float64, result), nothing, policy_var)

    elseif dim_type == :exogenous
        # ── Validate: must return a 2-tuple (grid::Vector, Π::Matrix) ─────────
        (result isa Tuple && length(result) == 2) ||
            error("Grid function '$fn_name' for exogenous dimension '$dim_name' " *
                  "must return a 2-tuple (grid, transition_matrix), " *
                  "got $(typeof(result)).\n" *
                  "Exogenous grid functions should return (grid_vector, transition_matrix).")

        grid, Π = result

        grid isa AbstractVector ||
            error("First element (grid) from '$fn_name' for exogenous dimension " *
                  "'$dim_name' must be a Vector, got $(typeof(grid)).")
        length(grid) == n ||
            error("Grid from '$fn_name' for exogenous dimension '$dim_name': " *
                  "expected $n points (params.n = $n), got $(length(grid)).")

        Π isa AbstractMatrix ||
            error("Second element (transition matrix) from '$fn_name' for exogenous " *
                  "dimension '$dim_name' must be a Matrix, got $(typeof(Π)).")
        size(Π) == (n, n) ||
            error("Transition matrix from '$fn_name' for exogenous dimension '$dim_name': " *
                  "expected $(n)×$(n), got $(size(Π)).")

        return HeterogeneityDimension(:exogenous, n,
                                      collect(Float64, grid), Matrix{Float64}(Π), nothing)

    else
        error("Unknown dimension type '$dim_type' for dimension '$dim_name'. " *
              "Expected 'endogenous' or 'exogenous'.")
    end
end
