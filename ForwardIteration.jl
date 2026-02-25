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
    ForwardIteration(policy_seqs::NamedTuple, model::SequenceModel,
                     ss_initial) -> NamedTuple

Generic forward iteration: evolves the distribution of agents over all T-1
transition periods and aggregates each `:heterogeneous` variable's policy into
a time series of scalar aggregate values.

## Arguments

- `policy_seqs`: NamedTuple from `BackwardIteration` — maps each heterogeneous
  variable name to a `Vector{Matrix}` of length T-1.
- `model`: the `SequenceModel`.
- `ss_initial`: the starting steady state; must expose `.D::Vector{Float64}`
  (the initial stationary distribution at `t = 0`).

## Returns

A NamedTuple mapping each `:heterogeneous` variable name to a `Vector` of
length T-1 containing the sequence of aggregate values, e.g.
`(KD = [0.32, 0.31, …],)`. These are the "aggregated sequences" consumed by
`assemble_full_xMat` to fill the heterogeneous rows of the padded xMat.

## State-space ordering

Dimensions are partitioned by type (preserving `model.heterogeneity` order):
- Endogenous dims → **fastest** (inner) indices
- Exogenous dims  → **slower** (outer) indices, in NamedTuple order

With one endogenous dim (n_a) and K exogenous dims (n_e1, …, n_eK):
```
j = (e_flat - 1)·n_a + a,   e_flat = column-major index over e1,…,eK
```

**Policy matrix convention**: `policy_seqs[varname][t]` must be `(n_a × n_exog)`
with columns indexed by `e_flat` (matches `backward_capital`'s output for a
single exogenous dim).

## Algorithm (per period t)

1. **Endogenous transition** — block-diagonal Young's matrix `Λ_endog` of size
   `(n_a·n_exog)²` via `make_endogenous_transition`.
2. **Exogenous transition** `Λ_exog` — time-invariant Kronecker product (built
   once); each exogenous dim's row-stochastic Π is transposed to column-stochastic.
3. **Composition** — `Λ_t = Λ_exog * Λ_endog`.
4. **Distribution update** — `D_t = Λ_t * D_{t-1}`, starting from `ss_initial.D`.
5. **Aggregation** — `dot(vec(policy_t), D_t)` for each heterogeneous variable.

Note: multiple endogenous dimensions are not yet supported (raises an error).
Multiple exogenous dimensions are fully supported via the Kronecker composition.

## AD compatibility

When `policy_seqs` carries `ForwardDiff.Dual` element types, `D` is
initialised from `ss_initial.D` (Float64) but immediately promoted to Dual
on the first matrix-vector multiply. The element type of `agg_seqs` matches
the element type of the policy matrices.
"""
function ForwardIteration(policy_seqs::NamedTuple,
                          model::SequenceModel,
                          ss_initial)
    @unpack T = model.compspec

    # Partition heterogeneity dimensions by type, preserving NamedTuple order.
    endog_dims = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :endogenous]
    exog_dims  = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :exogenous]

    n_endog_states = prod(d.n for (_, d) in endog_dims)
    n_exog_states  = prod(d.n for (_, d) in exog_dims)

    length(endog_dims) == 1 ||
        error("ForwardIteration: exactly one endogenous dimension is currently " *
              "supported (got $(length(endog_dims)))")
    endog_dim = endog_dims[1][2]

    # Build the time-invariant exogenous Kronecker factor (always Float64).
    # Π matrices from Rouwenhorst are row-stochastic; transpose → column-stochastic.
    # After the loop: Λ_exog = kron(Π_eK', kron(…, kron(Π_e1', I_{n_a})))
    Λ_exog = spdiagm(0 => ones(Float64, n_endog_states))
    for (_, dim) in exog_dims
        Π_T    = copy((dim.transition::Matrix{Float64})')
        Λ_exog = kron(sparse(Π_T), Λ_exog)
    end

    het_keys = vars_of_type(model, :heterogeneous)

    # Infer element type from the first policy matrix so the output vector
    # carries Dual numbers when backward iteration produced them.
    TF = eltype(policy_seqs[het_keys[1]][1])

    # Initial distribution from starting SS, promoted to TF (zero partials).
    D = Vector{TF}(ss_initial.D)

    agg_data = [Vector{TF}(undef, T - 1) for _ in het_keys]

    for t in 1:T-1
        # Endogenous Young's transition for period t
        Λ_endog = make_endogenous_transition(policy_seqs[endog_dim.policy_var][t],
                                             endog_dim, n_exog_states)

        # Evolve distribution
        D = Λ_exog * Λ_endog * D

        # Aggregate each heterogeneous variable: E_D[policy_t]
        for (j, varname) in enumerate(het_keys)
            agg_data[j][t] = dot(vec(policy_seqs[varname][t]), D)
        end
    end

    return NamedTuple{het_keys}(Tuple(agg_data))
end


"""
    make_ss_transition(policy_mat, model::SequenceModel) -> SparseMatrixCSC

Builds the full joint (column-stochastic) transition matrix for the steady-state
distribution from a single time-invariant policy matrix.

This is the same Kronecker + Young's composition used inside `ForwardIteration`,
extracted here so `get_SteadyState` can compute the stationary distribution
without constructing a full T-1 period path.

AD-compatible: when `policy_mat` contains ForwardDiff dual numbers the returned
matrix carries dual element types, propagating derivatives into `invariant_dist`.

Call `invariant_dist(make_ss_transition(policy, model)')` to get the stationary
distribution (pass the transpose because `invariant_dist` expects a row-stochastic
input, while our convention is column-stochastic Λ).
"""
function make_ss_transition(policy_mat, model::SequenceModel)
    endog_dims = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :endogenous]
    exog_dims  = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :exogenous]

    n_endog = prod(d.n for (_, d) in endog_dims)
    n_exog  = prod(d.n for (_, d) in exog_dims)

    length(endog_dims) == 1 ||
        error("make_ss_transition: exactly one endogenous dimension supported " *
              "(got $(length(endog_dims)))")
    endog_dim = endog_dims[1][2]

    # Build time-invariant exogenous Kronecker factor (always Float64)
    Λ_exog = spdiagm(0 => ones(Float64, n_endog))
    for (_, dim) in exog_dims
        Π_T = copy((dim.transition::Matrix{Float64})')
        Λ_exog = kron(sparse(Π_T), Λ_exog)
    end

    Λ_endog = make_endogenous_transition(policy_mat, endog_dim, n_exog)
    return Λ_exog * Λ_endog
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
Π should be **row-stochastic** (rows sum to 1). For a column-stochastic Λ, call
`invariant_dist(Λ')`.

Uses the linear-system trick from:
https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8

For ForwardDiff dual-number inputs, dispatch goes to the specialised overload below,
which avoids propagating duals through SuiteSparse via the Sherman-Morrison formula.
"""
function invariant_dist(Π)
    ΠT = Π'
    M  = I - ΠT[2:end, 2:end]
    b  = Vector(ΠT[2:end, 1])
    D  = vcat(one(eltype(b)), M \ b)
    return D ./ sum(D)
end


"""
    invariant_dist(Π::AbstractMatrix{<:ForwardDiff.Dual}) -> Vector{<:ForwardDiff.Dual}

ForwardDiff-compatible overload that avoids propagating dual numbers through
SuiteSparse (which only handles `AbstractFloat` element types).

## Algorithm

**Primal** — strips dual values and calls the Float64 sparse solver (identical to
the non-Dual method).

**Tangent** — differentiates the stationarity equation `(I − Λ) D = 0, 1ᵀD = 1`
analytically via the Implicit Function Theorem. For the k-th partial direction:

    (I − Λ) ΔD = ΔΛ D₀     (IFT tangent equation)
    1ᵀ ΔD = 0              (differentiated normalisation)

Eliminating `ΔD[1] = −sum(ΔD[2:end])` from the second equation and substituting
`u = Λ[2:end, 1]` turns this into a rank-1 update of the primal matrix
`M = I − Λ[2:end, 2:end]`:

    (M + u 1ᵀ) ΔD[2:end] = (ΔΛ D₀)[2:end]

Solved via the Sherman-Morrison formula with two sparse backsolves (M factorised
once and reused across all N partials):

    ΔD[2:end] = y1 − y2 · sum(y1) / (1 + sum(y2))
        y1 = Mfac \\ rhs_k      (rhs_k = (ΔΛ_k D₀)[2:end], O(nnz) SpMV)
        y2 = Mfac \\ u          (= unnormalised primal tail; already available)

Note: `1 + sum(y2) = s`, the primal normalisation factor, so no extra work.

**Complexity**: O(nnz · N) for the SpMV pass + N sparse backsolves with the
cached factorisation. No densification; memory scales with nnz, not n².
"""
function invariant_dist(Π::AbstractMatrix{<:ForwardDiff.Dual})
    ET  = eltype(Π)
    Tag = ForwardDiff.tagtype(ET)
    V   = ForwardDiff.valtype(ET)
    N   = ForwardDiff.npartials(ET)

    # ΠT = Π' = Λ (column-stochastic).
    # For the canonical call invariant_dist(Λ') with Λ::SparseMatrixCSC{Dual},
    # Π is Adjoint{Dual,SparseMatrixCSC{Dual}} and ΠT is SparseMatrixCSC{Dual}.
    ΠT = Π'

    if ΠT isa SparseMatrixCSC
        # ── Sparse path: IFT + Sherman-Morrison (O(nnz · N)) ──────────────────

        # Extract Float64 values in one pass over nzval, preserving sparsity.
        ΠT0 = SparseMatrixCSC(ΠT.m, ΠT.n,
                              copy(ΠT.colptr), copy(ΠT.rowval),
                              ForwardDiff.value.(ΠT.nzval))

        # Primal: Float64 sparse solve (mirrors the non-Dual method exactly).
        M    = I - ΠT0[2:end, 2:end]
        u    = Vector(ΠT0[2:end, 1])
        Mfac = factorize(M)   # factorise once; reused for all N backsolves
        y2   = Mfac \ u       # unnormalised primal tail (= M⁻¹ u)
        s    = 1.0 + sum(y2)  # normalisation factor; also = 1 + 1ᵀ y2 (S-M denom.)
        D0   = vcat(1.0, y2) ./ s

        n = length(D0)

        # Build rhs_all ((n−1) × N) with one pass over the non-zero structure.
        # rhs_all[:, k] = (ΔΛ_k · D₀)[2:end]  where ΔΛ_k has the same sparsity as ΠT.
        rhs_all = zeros(V, n - 1, N)
        for col in 1:ΠT.n
            d0_col = D0[col]
            for idx in ΠT.colptr[col]:ΠT.colptr[col+1]-1
                row = ΠT.rowval[idx]
                row >= 2 || continue          # rhs only needs rows 2:end
                ps = ForwardDiff.partials(ΠT.nzval[idx])
                for k in 1:N
                    @inbounds rhs_all[row - 1, k] += ps[k] * d0_col
                end
            end
        end

        # Batch backsolve: N right-hand sides with the same factorisation.
        Y1 = Mfac \ rhs_all   # (n−1) × N

        # Sherman-Morrison correction: ΔD_tail = y1 − y2 · sum(y1) / s
        ΔD = Matrix{V}(undef, n, N)
        for k in 1:N
            y1k     = @view Y1[:, k]
            ΔD_tail = y1k .- y2 .* (sum(y1k) / s)
            ΔD[1, k]      = -sum(ΔD_tail)   # from 1ᵀΔD = 0
            ΔD[2:end, k] .= ΔD_tail
        end

    else
        # ── Dense path: hard error ─────────────────────────────────────────────
        # The ForwardDiff overload only supports SparseMatrixCSC inputs.
        # Densifying an n×n dual matrix requires O(n²) memory, which is
        # prohibitive for large transition matrices (e.g. 50,400×50,400 ≈ 20 GiB).
        #
        # If you reach this branch, ensure the transition matrix is built via
        # make_ss_transition (which always returns SparseMatrixCSC), not by
        # calling Matrix(Λ) or any function that materialises a dense copy.
        n = size(ΠT, 1)
        mem_gib = round(n^2 * 8 / 2^30; digits = 1)
        error("invariant_dist (ForwardDiff overload): received a $(n)×$(n) dense " *
              "Dual matrix (~$(mem_gib) GiB if densified). Only SparseMatrixCSC " *
              "inputs are supported. Build your transition matrix via " *
              "make_ss_transition, which always returns a SparseMatrixCSC.")
    end

    # Assemble output as Vector{Dual}
    return [ForwardDiff.Dual{Tag,V,N}(
                D0[i],
                ForwardDiff.Partials{N,V}(ntuple(k -> ΔD[i, k], Val(N))))
            for i in 1:n]
end

DistributionTransition = DistributionTransition2
