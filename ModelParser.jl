


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
