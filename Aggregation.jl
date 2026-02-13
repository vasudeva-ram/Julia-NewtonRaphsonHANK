# Description: This file contains the functions that calculate the residuals
# in the dynamics and steady state of the model. The residuals are used in the
# solution of the model, and are calculated by comparing the model's equations
# to the actual values of the variables.


"""
    Residuals(x::Vector{Float64}, 
    model::SequenceModel)

Returns the residuals in the dynamics, given variable
values for the entire sequence of T periods.
"""
function Residuals(xVec, # (n_v x T-1) vector of all endogenous variable values
    KD, # T-1 sequence of capital demand values
    exogZ, # exogenous variable values
    model::SequenceModel)

    # Unpack parameters
    @unpack δ, α = model.Params
    @unpack n_v, T = model.Params

    xMat = reshape(xVec, (n_v, T-1)) # reshape to (n_v, T-1) matrix
    namedXvecs = NamedTuple{model.varXs}(Tuple([xMat[i, :] for i in 1:n_v]))
    @unpack Y, KS, r, w, Z = namedXvecs
    
    # generate lagged and exogenous variables
    KS_lag = [KS[1]; KS[1:end-1]]

    # Initialize residuals
    residuals = [
        Y .- (Z .* (KS_lag.^α)),
        r .+ δ .- (α .* Z .* (KS_lag.^(α-1))),
        w .- ((1-α) .* Z .* (KS_lag.^α)),
        KS .- KD,
        Z .- exogZ, # exogenous variable equality
        ]
        
    transRes = [(i)' for i in residuals]

    return vec(reduce(vcat, transRes))
end


"""
    Residuals(xMat::AbstractMatrix, model::SequenceModel)

Generic residuals function that evaluates the model's compiled equations
against the variable matrix `xMat` (n_v × T-1).

All variables (endogenous, exogenous, aggregated) should be rows of `xMat`,
ordered to match `model.varXs`. Returns a vector of residuals ordered as:
all equations at t=1, all equations at t=2, ... (column-major of k × (T-1)).
"""
function Residuals(xMat::AbstractMatrix, model::SequenceModel)
    return model.residuals_fn(xMat, model.Params)
end


"""
    ResidualsSteadyState(x::Vector{Float64},
    a::Matrix{Float64},
    D::Vector{Float64},
    model::SequenceModel)

Returns the residuals in the steady state, given variables
values, savings policies and a distribution.
"""
function ResidualsSteadyState(xVals::AbstractVector, # vector of variable values
    a::AbstractMatrix, # matrix of savings policies
    D::AbstractVector, # steady state distribution
    model::SequenceModel)
    
    namedXvars = NamedTuple{model.varXs}(xVals)
    @unpack Y, KS, r, w, Z = namedXvars
    
    # Initialize vectors
    residuals = zeros(length(xVals))
    
    # Calculate aggregated variables
    KD = vcat(a...)' * D

    # set exogenous variable steady state value 
    #TODO: allow user to indicate this in main file
    Zexog = 1.0
    
    # Unpack parameters
    @unpack α, δ = model.Params
    
    # Calculate residuals
    residuals = [
        Y .- (Z .* (KS.^α)),
        r .+ δ .- (α .* Z .* (KS.^(α-1))),
        w .- ((1-α) .* Z .* (KS.^α)),
        KS .- KD, # capital market clearing
        Z .- Zexog # exogenous variable equality
        ]
    
    return vcat(vcat(residuals'...)...)
end





# MWE for the function
function ffunc(x::Vector{Float64})

    # first reshape
    xMat = reshape(x, (:, 2))
    
    # apply function
    res = Zygote.Buffer(zeros(Float64, 2, 2), 2, )
    res = [
        xMat[:,1] .+ xMat[:,2],
        xMat[:,1] .- xMat[:,2]
        ] # return value is a vector of two vectors

    # return vcat(vcat(res'...)...)
    return copy(reduce(vcat, reduce(vcat, res')))
end



"""
    equation_residuals(equations::Vector{String}, vars::NamedTuple, params::NamedTuple)

Evaluate a set of string-based mathematical equations over multiple data points and parameters,
returning a vector of residuals (LHS - RHS for each equation at each row).

# Arguments
- `equations`: A vector of strings, each representing an equation with an "=" sign, e.g. `["x + y = z", "y^2 = a"]`.
- `vars`: A named tuple with variable names as keys and `Vector{Float64}` values.
           Each variable vector is assumed to have the same length.
- `params`: A named tuple with parameter names as keys and `Float64` values (scalars).

# Returns
- A `Vector{Float64}` containing the residuals. Its length is
  `length(equations) * (length of variable vectors)`.
"""
function equation_residuals(equations::Vector{String}, 
    vars::NamedTuple, 
    params::NamedTuple)
    
    # Helper function to evaluate an expression (Expr) in a "local" environment constructed from a dictionary.
    # It effectively creates a `let ... end` block that binds each key in `env` to its corresponding value.
    function eval_in_env(expr::Expr, env::Dict{Symbol,Any})
        # Construct an expression of the form:
        # let
        #   var1 = env[var1]
        #   var2 = env[var2]
        #   ...
        #   <original expr>
        # end
        let_block = :($(Expr(:block)))
        for (k, v) in env
            push!(let_block.args, :($(k) = $v))
        end
        # Now push the expression we want to evaluate
        push!(let_block.args, expr)
        # Wrap it all in a `let ... end` block
        let_expr = :(let
            $let_block
        end)
        return eval(let_expr)
    end

    # 1) Extract the length N of each variable vector (assuming all are the same size).
    N = length(first(values(vars)))  # e.g., if `x = [1.0, 2.0, 3.0]` then N = 3

    # 2) Prepare a result vector. We’ll store residuals in order:
    #    eq1(row1), eq2(row1), ..., eqM(row1), eq1(row2), eq2(row2), ...
    M = length(equations)
    residuals = Vector{Float64}(undef, M * N)

    idx = 1
    for i in 1:N
        # For each row, build a dictionary of local variable bindings for evaluation.
        env = Dict{Symbol, Any}()

        # Add parameters (scalars) to env
        for (pname, pval) in pairs(params)
            env[pname] = pval
        end

        # Add variables for the i-th row to env
        for (vname, vvals) in pairs(vars)
            env[vname] = vvals[i]
        end

        # Evaluate each equation, compute LHS - RHS, and store in residuals vector
        for eq in equations
            eq_parts = split(eq, "=")
            if length(eq_parts) != 2
                error("Each equation string must contain exactly one '=' sign.")
            end

            lhs_str, rhs_str = strip(eq_parts[1]), strip(eq_parts[2])
            lhs_expr = Meta.parse(lhs_str)
            rhs_expr = Meta.parse(rhs_str)

            lhs_val = eval_in_env(lhs_expr, env)
            rhs_val = eval_in_env(rhs_expr, env)

            residuals[idx] = lhs_val - rhs_val
            idx += 1
        end
    end

    return residuals
end
