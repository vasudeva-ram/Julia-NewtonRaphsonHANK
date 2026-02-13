


"""
    ParseTOML(file_path::String)

Parses the TOML file at the given file path
and returns a dictionary  of NamedTuples which will be used in the
model solution.
"""
function ParseTOML(file_path::String)
    toml_data = TOML.parsefile(file_path)
    for (key, value) in toml_data
        # println(value)
        toml_data[key] = DictToNamedTuple(value)
    end
    return toml_data
end


"""
    DictToNamedTuple(dict::Dict{String, Any})

Converts a dictionary into a NamedTuple that has Symbols for keys
    (instead of strings).
"""
function DictToNamedTuple(dict::Dict{String, Any})
    return NamedTuple{Tuple(Symbol(k) for k in keys(dict))}(values(dict))
end



function SplitExpressions(expressions::Array{String})
    lhs_expressions = []
    rhs_expressions = []
    for expression in expressions
        expression = replace(expression, "+" => ".+")
        expression = replace(expression, "-" => ".-")
        expression = replace(expression, "*" => ".*")
        expression = replace(expression, "/" => "./")
        expression = replace(expression, "^" => ".^")
        lhs, rhs = split(expression, "=")
        push!(lhs_expressions, lhs)
        push!(rhs_expressions, rhs)
    end
    return lhs_expressions, rhs_expressions
end


function ParseExpressions(expressions::Array{String},
    variables::NamedTuple{String, Vector{Float64}},
    params::NamedTuple{String, Float64})

    @unpack (; variables...)
    @unpack (; params...)
    lhs, rhs = SplitExpressions(expressions)

    results = []
    for expression in expressions
        # Replace variable names with their corresponding values
        for (variable, value) in variables
            expression = replace(expression, variable => string(value))
        end
        # Evaluate the expression
        result = eval(Meta.parse(expression))
        push!(results, result)
    end
    return results
end


# ─────────────────────────────────────────────────────────────────────────────
# Equation compilation: string equations → compiled residuals function
# ─────────────────────────────────────────────────────────────────────────────

# Arithmetic operators that should be broadcast for element-wise operations
const BROADCAST_OPS = Set([:+, :-, :*, :/, :^])


"""
    transform_expr(expr, var_indices::Dict{Symbol,Int}, param_names::Set{Symbol})

Recursively walks a Julia expression tree and transforms it so that:
- Variable names become `xMat[idx, :]` (row slices of the input matrix)
- Lag notation `VAR(-i)` becomes `shift_lag(xMat[idx, :], i)`
- Lead notation `VAR(+i)` becomes `shift_lead(xMat[idx, :], i)`
- Parameter names become `params.name`
- Arithmetic operators (`+`, `-`, `*`, `/`, `^`) become broadcasting versions
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
    compile_residuals(equations::Vector{String}, varXs::Tuple{Vararg{Symbol}})

Compiles a vector of equation strings into a single Julia function
`(xMat::AbstractMatrix, params) -> Vector` that evaluates the residuals.

Each equation has the form `"LHS = RHS"`. The residual for each equation
is `LHS .- RHS`, evaluated element-wise over the time dimension.

Variables are identified by matching against `varXs`.
Parameters are identified by matching against `fieldnames(ModelParams)`.
Lag/lead notation `VAR(-i)` / `VAR(+i)` is supported.

The output vector is ordered: all equations at t=1, all equations at t=2, ...
(column-major vectorization of a k × (T-1) residual matrix).
"""
function compile_residuals(equations::Vector{String}, varXs::Tuple{Vararg{Symbol}})
    var_indices = Dict(sym => i for (i, sym) in enumerate(varXs))
    param_names = Set(fieldnames(ModelParams))

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
