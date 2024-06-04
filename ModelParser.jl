

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


function ParseExpressions(expressions::Array{String}, variables::Dict{String, Float64})
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
