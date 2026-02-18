# ─────────────────────────────────────────────────────────────────────────────
# ModelParser.jl — TOML parsing, equation compilation, and model construction
#
# This file handles the translation from user-facing model specification
# (ModelFile.toml) to the internal Julia representation (SequenceModel).
# It contains three layers:
#
# 1. TOML utilities: ParseTOML, DictToNamedTuple — generic helpers for
#    reading TOML files into Julia data structures.
#
# 2. Equation compilation: transform_expr, compile_residuals — a
#    metaprogramming system that converts string equations like
#    "Y = Z * KS(-1)^α" into compiled Julia functions that operate on
#    the n_v × (T-1) variable matrix. This is done at model construction
#    time via `eval`, so the resulting function is regular Julia code that
#    is fully compatible with automatic differentiation (ForwardDiff, Zygote).
#
# 3. Model construction: build_model_from_toml — the main entry point that
#    reads a TOML file, builds heterogeneity dimensions, compiles equations,
#    and returns a complete SequenceModel ready for solving.
#
# Legacy functions (SplitExpressions, ParseExpressions) have been removed.
# They are superseded by the equation compilation system.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# TOML utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParseTOML(file_path::String) -> Dict{String, NamedTuple}

Parses a TOML file and converts each top-level section into a NamedTuple.
Returns a dictionary mapping section names (strings) to NamedTuples.

Note: This is a generic utility. For full model construction from TOML,
use `build_model_from_toml` instead.
"""
function ParseTOML(file_path::String)
    toml_data = TOML.parsefile(file_path)
    for (key, value) in toml_data
        toml_data[key] = DictToNamedTuple(value)
    end
    return toml_data
end


"""
    DictToNamedTuple(dict::Dict{String, Any}) -> NamedTuple

Converts a `Dict{String, Any}` into a NamedTuple with Symbol keys.
Used internally by `ParseTOML` to make TOML sections accessible via
dot syntax (e.g., `section.fieldname`).
"""
function DictToNamedTuple(dict::Dict{String, Any})
    return NamedTuple{Tuple(Symbol(k) for k in keys(dict))}(values(dict))
end


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
    compile_residuals(equations::Vector{String}, varXs::Tuple{Vararg{Symbol}},
                      param_names::Set{Symbol})

Compiles a vector of equation strings into a single Julia function
`(xMat::AbstractMatrix, params) -> Vector` that evaluates the residuals.

Each equation has the form `"LHS = RHS"`. The residual for each equation
is `LHS .- RHS`, evaluated element-wise over the time dimension (columns
of `xMat`).

The compilation pipeline:
1. Parse each equation string into a Julia AST via `Meta.parse`
2. Transform the AST via `transform_expr` (variable → row slice, params → field access, etc.)
3. Assemble all transformed equations into a single anonymous function body
4. `eval` the function at construction time → returns a regular Julia function

The resulting function is ordinary compiled Julia code (no runtime `eval`),
so it is fully compatible with ForwardDiff and Zygote for automatic differentiation.

Variables are identified by matching symbols against `varXs`.
Parameters are identified by matching against the explicitly provided `param_names`.

The output vector is ordered: all equations at t=1, all equations at t=2, ...
(column-major vectorization of a k × (T-1) residual matrix).
"""
function compile_residuals(equations::Vector{String}, varXs::Tuple{Vararg{Symbol}},
                           param_names::Set{Symbol})
    var_indices = Dict(sym => i for (i, sym) in enumerate(varXs))

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

        # _r_i = LHS .- RHS
        push!(body_exprs, :($(residual_syms[i]) = $(Expr(:call, :.-, lhs_transformed, rhs_transformed))))
    end

    # Stack residuals: transpose each into a row, vcat into k×(T-1) matrix, then vec
    trans_exprs = [:($(s)') for s in residual_syms]
    push!(body_exprs, :(return vec(reduce(vcat, [$(trans_exprs...)]))))

    fn_body = Expr(:block, body_exprs...)

    fn_expr = :(function(xMat::AbstractMatrix, params)
        $fn_body
    end)

    return eval(fn_expr)
end


# ─────────────────────────────────────────────────────────────────────────────
# Full model construction from TOML
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_model_from_toml(file_path::String, agg_vars) -> SequenceModel

Constructs a complete `SequenceModel` from a TOML specification file.

Reads all sections of the TOML and performs the following steps:
1. Builds heterogeneity dimensions from `[Dimensions.*]` sections via `build_dimension`
2. Collects variable names from `[EndogenousVariables]` and `[ExogenousVariables]`
3. Builds `ComputationalSpec` from `[ComputationalSpecs]` section (T, ε, dx).
   If the section or individual fields are missing, defaults are used
   (T=150, ε=1e-6, dx=1e-8) and a message is printed.
4. Constructs a `params` NamedTuple from `[Parameters]` (economic params only)
5. Compiles equilibrium equations from `[Equations]` via `compile_residuals`
6. Assembles everything into a `SequenceModel`

The `agg_vars` argument must be provided directly as a NamedTuple of
`(backward, forward)` function pairs, since Julia functions cannot be
referenced by name in TOML. The TOML `[AggregatedVariables]` section
documents which functions are expected, but the actual function objects
must be passed in by the caller.

# Example
```julia
agg_vars = (KD = (backward = backward_capital, forward = agg_capital),)
mod = build_model_from_toml("ModelFile.toml", agg_vars)
```
"""
function build_model_from_toml(file_path::String, agg_vars)
    toml = TOML.parsefile(file_path)

    # 1. Build heterogeneity dimensions as a NamedTuple
    dims_dict = toml["Dimensions"]
    dim_names = Tuple(Symbol(k) for k in keys(dims_dict))
    dim_values = Tuple(build_dimension(v) for v in values(dims_dict))
    heterogeneity = NamedTuple{dim_names}(dim_values)

    # 2. Build variable tuples
    endog_vars = Tuple(Symbol(k) for k in keys(toml["EndogenousVariables"]))
    exog_vars = Tuple(Symbol(k) for k in keys(toml["ExogenousVariables"]))
    varXs = (endog_vars..., exog_vars...)

    # 3. Build ComputationalSpec from [ComputationalSpecs] (with defaults)
    defaults = Dict("T" => 150, "ε" => 1e-6, "dx" => 1e-8)
    csdict = get(toml, "ComputationalSpecs", Dict{String,Any}())
    if isempty(csdict)
        println("Note: [ComputationalSpecs] section not found in $file_path — using defaults (T=$(defaults["T"]), ε=$(defaults["ε"]), dx=$(defaults["dx"]))")
    end
    function _get_cs(name)
        if haskey(csdict, name)
            return isa(csdict[name], Dict) ? csdict[name]["value"] : csdict[name]
        else
            println("Note: '$name' not specified in [ComputationalSpecs] — using default $(defaults[name])")
            return defaults[name]
        end
    end
    compspec = ComputationalSpec(
        Int(_get_cs("T")),
        Float64(_get_cs("ε")),
        Float64(_get_cs("dx")),
        length(varXs)
    )

    # 4. Build params NamedTuple from [Parameters] (economic params only)
    pdict = toml["Parameters"]
    pnames = Tuple(Symbol(k) for k in keys(pdict))
    pvalues = Tuple(isa(v["value"], Integer) ? Int(v["value"]) : Float64(v["value"])
                    for v in values(pdict))
    params = NamedTuple{pnames}(pvalues)

    # 5. Compile equations (pass all param names — economic + compspec — so compiler
    #    knows what's a param vs. a variable)
    equations = Tuple(toml["Equations"]["equilibrium"])
    param_names = Set{Symbol}(pnames)
    union!(param_names, Set([:T, :ε, :dx, :n_v]))
    residuals_fn = compile_residuals(collect(equations), varXs, param_names)

    # 6. Construct SequenceModel
    return SequenceModel(varXs, equations, compspec, params, residuals_fn, agg_vars, heterogeneity)
end



