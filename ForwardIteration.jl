# Description: This file contains the functions that implement the forward iteration
# algorithm. Starting from the household policy sequences produced by BackwardIteration,
# it evolves the distribution of agents forward through time and aggregates each
# endogenous variable's policy into a time series of aggregate values.
#
# Convention:
#   - Endogenous dimensions are "fast" (inner, row index of policy matrices).
#   - Exogenous dimensions are "slow" (outer, column index of policy matrices).
#   - For the KS model: wealth is fast (n_a rows), productivity is slow (n_e cols).
#   - The vectorised state is j = (e-1)*n_a + a, so wealth varies fastest.
#
# The full per-period transition matrix is built in two steps:
#   1. make_endogenous_transition  — block-diagonal Young's matrix (wealth only)
#   2. Kronecker composition       — compose with Π' to scatter across exog states
# This separation keeps the two mechanisms explicit and independently testable.


"""
    make_endogenous_transition(policy_mat, dim::HeterogeneityDimension, n_exog::Int)

Constructs the block-diagonal endogenous transition matrix for a single endogenous
heterogeneity dimension using Young's (2010) method.

`policy_mat` is an (n_endog × n_exog) matrix where entry [a, e] is the next-period
value of the endogenous state for a household currently at endogenous state `a` and
exogenous state `e`.

Returns a sparse (n_endog * n_exog) × (n_endog * n_exog) **column-stochastic** matrix
that is block-diagonal with n_exog blocks of size n_endog × n_endog. Block e captures
P(a' | a, e) via linear interpolation on `dim.grid`, **holding e fixed**. Exogenous
transitions are NOT applied here; they are composed separately in `ForwardIteration`.

Boundary convention:
- policy below grid[1]    → all mass placed on the first grid point
- policy above grid[end]  → all mass placed on the last grid point
"""
function make_endogenous_transition(policy_mat, dim::HeterogeneityDimension, n_exog::Int)
    n_a  = dim.n
    grid = dim.grid
    n_m  = n_a * n_exog

    Is = Int64[]
    Js = Int64[]
    Vs = eltype(policy_mat)[]

    for e in 1:n_exog
        for ia in 1:n_a
            col = (e - 1) * n_a + ia       # vectorised index of current state (a=ia, e=e)
            p   = policy_mat[ia, e]         # next-period endogenous state

            # First grid index m where grid[m] ≥ p (searchsortedfirst is O(log n_a))
            m = searchsortedfirst(grid, p)

            if m == 1
                # At or below grid minimum — all mass on first point
                push!(Is, (e - 1) * n_a + 1)
                push!(Js, col)
                push!(Vs, one(eltype(policy_mat)))
            elseif m > n_a
                # Above grid maximum — all mass on last point
                push!(Is, (e - 1) * n_a + n_a)
                push!(Js, col)
                push!(Vs, one(eltype(policy_mat)))
            else
                # Linear interpolation between grid[m-1] and grid[m]
                w = (p - grid[m - 1]) / (grid[m] - grid[m - 1])
                push!(Is, (e - 1) * n_a + m - 1)
                push!(Js, col)
                push!(Vs, one(eltype(policy_mat)) - w)
                push!(Is, (e - 1) * n_a + m)
                push!(Js, col)
                push!(Vs, w)
            end
        end
    end

    return sparse(Is, Js, Vs, n_m, n_m)
end


"""
    ForwardIteration(xVec, policy_seqs::NamedTuple, model::SequenceModel, ss::SteadyState)

Generic forward iteration: evolves the distribution of agents over all T-1 periods
and computes the time path of each aggregated variable.

## State-space ordering

Dimensions are partitioned by type (preserving `model.heterogeneity` NamedTuple order):
- Endogenous dims first → **fastest** (inner) indices
- Exogenous dims after  → **slower** (outer) indices, in NamedTuple order

With one endogenous dim (n_a) and K exogenous dims (n_e1, n_e2, …, n_eK):

```
state index j = (eK-1)·n_a·n_e1·…·n_e(K-1) + … + (e2-1)·n_a·n_e1 + (e1-1)·n_a + a
```

Equivalently, using a flattened exogenous index `e_flat` (column-major over e1, …, eK):
```
e_flat = (eK-1)·n_e1·…·n_e(K-1) + … + (e2-1)·n_e1 + e1
j      = (e_flat - 1)·n_a + a
```

**Policy matrix convention**: `policy_seqs[varname][t]` must be `(n_a × n_exog)` with
columns indexed by `e_flat` in the same ordering. For a single exogenous dim this is
just `(n_a × n_e)` with natural column ordering, matching `backward_capital`'s output.

## Algorithm (per period t)

1. **Endogenous transition** — builds block-diagonal Young's matrix `Λ_endog` of size
   `(n_a·n_exog) × (n_a·n_exog)` via `make_endogenous_transition`. There are `n_exog`
   blocks of size `n_a × n_a`, one per flattened exogenous state `e_flat`.

2. **Exogenous transition** `Λ_exog` — prebuilt once (time-invariant) as a nested
   Kronecker product. Each exogenous dim's row-stochastic Π must be transposed to
   obtain the column-stochastic factor. Built iteratively: each successive dim wraps
   around the outside, becoming the next slowest index:
   ```
   Λ_exog = kron(Π_eK', kron(…, kron(Π_e1', I_{n_a})))
   ```

3. **Composition** — `Λ_t = Λ_exog * Λ_endog`.

4. **Distribution update** — `D = Λ_t * D`, starting from `ss.D`.

5. **Aggregation** — for each `varname` in `model.agg_vars`, writes
   `dot(vec(policy_t), D)` into the corresponding row of `xMat`. `vec(policy_t)`
   flattens `(n_a × n_exog)` column-major, matching the state-vector ordering.

Returns the completed `n_v × (T-1)` matrix.

Note: multiple endogenous dimensions are not yet supported (raises an error).
Multiple exogenous dimensions are fully supported via the Kronecker composition.
"""
function ForwardIteration(xVec,
    policy_seqs::NamedTuple,
    model::SequenceModel,
    ss::SteadyState)

    @unpack T, n_v = model.compspec
    xMat = reshape(copy(xVec), (n_v, T-1))

    # Partition heterogeneity dimensions by type, preserving NamedTuple order.
    endog_dims = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :endogenous]
    exog_dims  = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :exogenous]

    n_endog = prod(d.n for (_, d) in endog_dims)
    n_exog  = prod(d.n for (_, d) in exog_dims)

    # Multiple endogenous dimensions not yet supported
    if length(endog_dims) != 1
        error("ForwardIteration: exactly one endogenous dimension is currently supported " *
              "(got $(length(endog_dims)))")
    end
    endog_dim = endog_dims[1][2]

    # Build the time-invariant exogenous Kronecker factor.
    #
    # State ordering: a is fastest; e1 (first exog in heterogeneity) is next;
    # eK (last exog) is slowest. Each successive dim wraps around the outside
    # of the Kronecker product, becoming the next slowest index.
    #
    # Π matrices from Rouwenhorst are row-stochastic (Π[e,e'] = P(e→e')).
    # For D_new = Λ*D_old (column-stochastic convention) we need Π' (transpose).
    #
    # After the loop: Λ_exog = kron(Π_eK', kron(…, kron(Π_e1', I_{n_a})))
    Λ_exog = spdiagm(0 => ones(Float64, n_endog))   # start: identity for endog states
    for (_, dim) in exog_dims
        Π_T = copy((dim.transition::Matrix{Float64})')  # column-stochastic, materialised
        Λ_exog = kron(sparse(Π_T), Λ_exog)             # new dim becomes outermost (slowest)
    end

    D = ss.D  # initial distribution (starting steady state)

    for t in 1:T-1
        # Endogenous Young's transition for this period.
        # policy_seqs[endog_dim.policy_var][t] is (n_a × n_exog), columns indexed
        # by e_flat = (eK-1)·n_e1·…·n_e(K-1) + … + e1 (matching Λ_exog ordering).
        Λ_endog = make_endogenous_transition(policy_seqs[endog_dim.policy_var][t],
                                             endog_dim, n_exog)

        # Compose full transition and evolve distribution
        D = Λ_exog * Λ_endog * D

        # Aggregate: dot(vec(policy_t), D) where vec flattens (n_a × n_exog)
        # column-major — matching state index j = (e_flat-1)*n_a + a.
        for (varname, _) in pairs(model.agg_vars)
            idx = findfirst(==(varname), model.varXs)
            xMat[idx, t] = dot(vec(policy_seqs[varname][t]), D)
        end
    end

    return xMat
end


"""
    DistributionTransition(policy, model::SequenceModel)

Constructs the full joint transition matrix Λ using Young's (2010) method,
composing the endogenous wealth transition with the exogenous productivity
transition in a single pass.

Legacy function retained for use in `get_SteadyState` (computing the
stationary distribution). Use `make_endogenous_transition` + Kronecker
composition for new code.
"""
function DistributionTransition1(policy, # savings policy function
    model::SequenceModel)

    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.params

    n_m = n_a * n_e
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = eltype(policy)[]

    for col in eachindex(policy)
        m = findfirst(x->x>=policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, n_e))
            append!(Vs, 1.0 .* Π[j,:])
        else
            append!(Is, (m-1) .+ Jbases)
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, 2*n_e))
            w = (policy[col] - policygrid[m-1]) / (policygrid[m] - policygrid[m-1])
            append!(Vs, (1.0 - w) .* Π[j,:])
            append!(Vs, w .* Π[j,:])
        end
    end

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end


function DistributionTransition2(policy,
    model::SequenceModel)

    @unpack policygrid, Π = model
    @unpack n_a, n_e = model.params

    n_m = n_a * n_e
    Jbases = [(ne - 1) * n_a for ne in 1:n_e]

    Is = Int64[]
    Js = Int64[]
    Vs = eltype(policy)[]

    for col in eachindex(policy)
        m = findfirst(x -> x >= policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            Is = vcat(Is, m .+ Jbases)
            Js = vcat(Js, fill(col, n_e))
            Vs = vcat(Vs, 1.0 .* Π[j, :])
        else
            Is = vcat(Is, (m - 1) .+ Jbases, m .+ Jbases)
            Js = vcat(Js, fill(col, 2 * n_e))
            w = (policy[col] - policygrid[m - 1]) / (policygrid[m] - policygrid[m - 1])
            Vs = vcat(Vs, (1.0 - w) .* Π[j, :], w .* Π[j, :])
        end
    end

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end


"""
    invariant_dist(Π::AbstractMatrix) -> Vector{Float64}

Calculates the invariant distribution of a Markov chain with transition matrix Π.
"""
function invariant_dist(Π)

    ΠT = Π' # transpose

    # https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8
    D = [1; (I - ΠT[2:end, 2:end]) \ Vector(ΠT[2:end,1])]

    return D ./ sum(D) # return normalized to sum to 1.0
end

DistributionTransition = DistributionTransition2
