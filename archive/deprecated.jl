# archive/deprecated.jl — functions removed from the main solver files.
#
# These are retained for reference only; none are called by the current pipeline.
# Source of each function is noted in the section header.


# ─── From KrusellSmith.jl ─────────────────────────────────────────────────────

# One-step AR(1) updater for Z. The pipeline uses exogenousZ (the path
# generator) instead; this mutating variant was never wired in.
function exogenousZ!(namedXvars::NamedTuple, model)
    @unpack Z, ρ, σ = namedXvars
    Z = ρ * Z + σ * sqrt(1 - ρ^2) * randn()
    return Z
end


# ─── From GeneralStructures.jl ────────────────────────────────────────────────

# Flatten a vector of matrices into a single vector [vec(M1); vec(M2); ...].
function vectorize_matrices(matrices::Vector{<:Matrix})
    n, m = size(matrices[1])
    result = similar(matrices[1], n*m, length(matrices))
    for i in 1:length(matrices)
        result[:, i] = vec(matrices[i])
    end
    return [result...]
end

# Inverse of vectorize_matrices: reshape a flat vector into Vector{Matrix{n×m}}.
function Vec2Mat(vec::Vector{Float64}, n::Int64, m::Int64)
    T = Int(length(vec) / (n * m))
    kmat = reshape(vec, (n, m, T))
    return [kmat[:, :, i] for i in 1:T]
end


# ─── From BackwardIteration.jl ────────────────────────────────────────────────

# Smoke test that ForwardDiff can differentiate through Interpolations.jl's
# gridded linear interpolation. Not part of the solver.
function FD_LI()
    x = rand(10)
    sort!(x)
    y = x .+ 1.0

    function inter(a)
        linpolate = extrapolate(interpolate((x,), y, Gridded(Linear())), Flat())
        return linpolate.(a)
    end

    z = x .+ 0.3
    J = jacobian(inter, z)
    return J
end


# ─── From ForwardIteration.jl ─────────────────────────────────────────────────

# Original joint transition matrix builder (version 1).
# Superseded by the factored make_endogenous_transition + Kronecker approach.
# References deleted model fields (model.policygrid, model.Π, model.params.n_a/n_e).
function DistributionTransition1(policy, model)
    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.params
    n_m    = n_a * n_e
    Jbases = [(ne - 1)*n_a for ne in 1:n_e]
    Is = Int64[]; Js = Int64[]; Vs = eltype(policy)[]
    for col in eachindex(policy)
        m = findfirst(x -> x >= policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            append!(Is, m .+ Jbases); append!(Js, fill(col, n_e)); append!(Vs, 1.0 .* Π[j,:])
        else
            append!(Is, (m-1) .+ Jbases); append!(Is, m .+ Jbases)
            append!(Js, fill(col, 2*n_e))
            w = (policy[col] - policygrid[m-1]) / (policygrid[m] - policygrid[m-1])
            append!(Vs, (1.0 - w) .* Π[j,:]); append!(Vs, w .* Π[j,:])
        end
    end
    return sparse(Is, Js, Vs, n_m, n_m)
end

# Version 2 (vcat instead of append!). Same caveats as DistributionTransition1.
function DistributionTransition2(policy, model)
    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.params
    n_m    = n_a * n_e
    Jbases = [(ne - 1) * n_a for ne in 1:n_e]
    Is = Int64[]; Js = Int64[]; Vs = eltype(policy)[]
    for col in eachindex(policy)
        m = findfirst(x -> x >= policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            Is = vcat(Is, m .+ Jbases); Js = vcat(Js, fill(col, n_e)); Vs = vcat(Vs, 1.0 .* Π[j, :])
        else
            Is = vcat(Is, (m - 1) .+ Jbases, m .+ Jbases)
            Js = vcat(Js, fill(col, 2 * n_e))
            w = (policy[col] - policygrid[m - 1]) / (policygrid[m] - policygrid[m - 1])
            Vs = vcat(Vs, (1.0 - w) .* Π[j, :], w .* Π[j, :])
        end
    end
    return sparse(Is, Js, Vs, n_m, n_m)
end

DistributionTransition = DistributionTransition2


# Convenience wrapper: build the full steady-state joint transition matrix from
# a single policy matrix. Duplicates what find_ss does inline using asm.Λ_exog;
# not called by the current pipeline.
function make_ss_transition(policy_mat, model::SequenceModel)
    endog_dims = [(name, dim) for (name, dim) in pairs(model.heterogeneity) if dim.dim_type == :endogenous]
    exog_dims  = [(name, dim) for (name, dim) in pairs(model.heterogeneity) if dim.dim_type == :exogenous]
    n_endog    = prod(d.n for (_, d) in endog_dims)
    n_exog     = prod(d.n for (_, d) in exog_dims)
    length(endog_dims) == 1 ||
        error("make_ss_transition: exactly one endogenous dimension supported")
    endog_dim = endog_dims[1][2]
    Λ_exog = spdiagm(0 => ones(Float64, n_endog))
    for (_, dim) in exog_dims
        Π_T = copy((dim.transition::Matrix{Float64})')
        Λ_exog = kron(sparse(Π_T), Λ_exog)
    end
    Λ_endog = make_endogenous_transition(policy_mat, endog_dim, n_exog)
    return Λ_exog * Λ_endog
end
