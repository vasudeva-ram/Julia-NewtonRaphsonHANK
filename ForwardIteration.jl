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
    transition_step(policy_mat, D_prev, Λ_exog, endog_dim::HeterogeneityDimension,
                    n_exog::Int) -> Vector

One period of distribution evolution:

    D_new = Λ_exog * (Λ_endog(policy_mat) * D_prev)

where `Λ_endog` is the block-diagonal Young's (2010) transition matrix built from
`policy_mat` via `make_endogenous_transition`. The two matrix-vector products are
computed sequentially — never forming the n_m × n_m Kronecker product — so that
the custom `rrule` below can compute the pullback without ever materialising a
dense outer-product cotangent.
"""
function transition_step(policy_mat, D_prev, Λ_exog,
                         endog_dim::HeterogeneityDimension, n_exog::Int)
    Λ_endog = make_endogenous_transition(policy_mat, endog_dim, n_exog)
    return Λ_exog * (Λ_endog * D_prev)
end


"""
    ChainRulesCore.rrule(::typeof(transition_step), ...)

Reverse-mode rule for `transition_step`. Avoids materialising any n_m × n_m
dense matrix in either the forward or backward pass.

## Forward pass

Reproduces the primal computation and additionally records the interpolation
bracket index `m` for each household state (ia, e) into `bucket` (n_a × n_exog
integer matrix). `bucket` and `Λ_endog` are captured by the pullback closure.

## Pullback

Given cotangent `ΔD_new` (length-n_m vector), returns:

1. `u = Λ_exog' * ΔD_new`  — one sparse MVM, O(nnz(Λ_exog))
2. `ΔD_prev = Λ_endog' * u`  — one sparse MVM, O(nnz(Λ_endog)) ≈ O(2 n_m)
3. `Δpolicy[ia,e]` via the chain rule through piecewise-linear interpolation:

       Δpolicy[ia,e] = D_prev[col] · (u[row_hi] − u[row_lo]) / Δgrid

   where `col = (e-1)·n_a + ia`, `row_hi/lo = (e-1)·n_a + m / m-1`, and
   `Δgrid = grid[m] − grid[m-1]`. At clamped boundary states the gradient
   is zero (constant extrapolation has zero derivative).

No outer products or dense n_m × n_m matrices are ever formed. All operations
are O(n_m). Memory overhead per time step: O(n_m) for `bucket` and `Λ_endog`.
"""
function ChainRulesCore.rrule(::typeof(transition_step),
                               policy_mat, D_prev, Λ_exog,
                               endog_dim::HeterogeneityDimension, n_exog::Int)
    n_a  = endog_dim.n
    grid = endog_dim.grid
    n_m  = n_a * n_exog

    # ── Forward: build Λ_endog, recording which bracket each policy falls in ──
    bucket = Matrix{Int}(undef, n_a, n_exog)
    Is = Int[];  Js = Int[];  Vs = Float64[]

    for e in 1:n_exog
        for ia in 1:n_a
            col = (e - 1) * n_a + ia
            p   = policy_mat[ia, e]
            m   = searchsortedfirst(grid, p)
            bucket[ia, e] = m

            if m == 1
                push!(Is, (e-1)*n_a + 1);    push!(Js, col); push!(Vs, 1.0)
            elseif m > n_a
                push!(Is, (e-1)*n_a + n_a);  push!(Js, col); push!(Vs, 1.0)
            else
                w = (p - grid[m-1]) / (grid[m] - grid[m-1])
                push!(Is, (e-1)*n_a + m - 1); push!(Js, col); push!(Vs, 1.0 - w)
                push!(Is, (e-1)*n_a + m);     push!(Js, col); push!(Vs, w)
            end
        end
    end

    Λ_endog = sparse(Is, Js, Vs, n_m, n_m)
    D_new   = Λ_exog * (Λ_endog * D_prev)

    function transition_pullback(ΔD_new)
        # 1. u = Λ_exog' * ΔD_new  (sparse MVM)
        u = Λ_exog' * ΔD_new

        # 2. ΔD_prev = Λ_endog' * u  (sparse MVM; ≈ 2 n_m non-zeros in Λ_endog)
        ΔD_prev = Λ_endog' * u

        # 3. Δpolicy: chain rule through piecewise-linear interpolation.
        #    Interior brackets only; clamped boundaries have zero derivative.
        Δpolicy = zeros(n_a, n_exog)
        for e in 1:n_exog
            for ia in 1:n_a
                m = bucket[ia, e]
                if 1 < m <= n_a
                    col   = (e - 1) * n_a + ia
                    Δgrid = grid[m] - grid[m - 1]
                    Δpolicy[ia, e] = D_prev[col] *
                        (u[(e-1)*n_a + m] - u[(e-1)*n_a + m - 1]) / Δgrid
                end
            end
        end

        # Λ_exog is a Float64 constant (precomputed) → ZeroTangent.
        # endog_dim and n_exog are structural, not differentiable → NoTangent.
        return NoTangent(), Δpolicy, ΔD_prev, ZeroTangent(), NoTangent(), NoTangent()
    end

    return D_new, transition_pullback
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
with columns indexed by `e_flat` (matches the output of `BackwardIteration` for
a single exogenous dimension).

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
    #
    # TODO: Λ_exog is identical every call and depends only on model.heterogeneity.
    # When the full Newton-Raphson solver is implemented, precompute Λ_exog once
    # (analogous to SSAssembler.Λ_exog) and pass it in, rather than rebuilding it
    # for each of the O(T²) ForwardIteration calls per Newton step.
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
        # Evolve distribution: D_t = Λ_exog * Λ_endog(policy_t) * D_{t-1}
        # Uses transition_step so that Zygote's pullback (via the custom rrule)
        # never materialises a dense outer-product cotangent.
        D = transition_step(policy_seqs[endog_dim.policy_var][t], D,
                            Λ_exog, endog_dim, n_exog_states)

        # Aggregate each heterogeneous variable: E_D[policy_t]
        for (j, varname) in enumerate(het_keys)
            agg_data[j][t] = dot(vec(policy_seqs[varname][t]), D)
        end
    end

    return NamedTuple{het_keys}(Tuple(agg_data))
end


"""
ChainRulesCore rrule for ForwardIteration.

Zygote uses this rrule instead of tracing through the ForwardIteration body,
which contains setindex! mutations. The forward pass mirrors ForwardIteration
exactly, but also stores the per-step distribution states and the pullback
closure from each transition_step call. The backward pass is a reverse-time
loop that propagates the cotangent on `agg_seqs` back to `policy_seqs`.

## Backward pass

Given cotangent `Δagg::NamedTuple` on the output (one Float64 per het var per
period), the reverse-time loop (t = T-1 downto 1) does:

  1. Aggregation cotangent on D_t:
       ΔD += Δagg[k][t] * vec(policy[k][t])   for each het var k

  2. Direct gradient of policy[k][t] from dot-product aggregation:
       Δpolicy[k][t] += Δagg[k][t] * reshape(D_t, policy_size)

  3. Backprop through transition_step (using the stored pullback closure):
       (_, Δpolicy_endog, ΔD_prev, ...) = ts_pullback[t](ΔD)
       Δpolicy[endog_var][t] += Δpolicy_endog
       ΔD = ΔD_prev   (propagate cotangent on D_{t-1})
"""
function ChainRulesCore.rrule(::typeof(ForwardIteration),
                               policy_seqs::NamedTuple,
                               model::SequenceModel,
                               ss_initial)
    @unpack T = model.compspec

    endog_dims = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :endogenous]
    exog_dims  = [(name, dim) for (name, dim) in pairs(model.heterogeneity)
                  if dim.dim_type == :exogenous]
    n_endog_states = prod(d.n for (_, d) in endog_dims)
    n_exog_states  = prod(d.n for (_, d) in exog_dims)
    length(endog_dims) == 1 ||
        error("ForwardIteration rrule: exactly one endogenous dimension supported")
    endog_dim = endog_dims[1][2]

    Λ_exog = spdiagm(0 => ones(Float64, n_endog_states))
    for (_, dim) in exog_dims
        Π_T    = copy((dim.transition::Matrix{Float64})')
        Λ_exog = kron(sparse(Π_T), Λ_exog)
    end

    het_keys  = vars_of_type(model, :heterogeneous)
    endog_var = endog_dim.policy_var
    endog_idx = findfirst(==(endog_var), het_keys)

    # ── Forward pass: run the loop, storing distributions and pullback closures ─
    D = Vector{Float64}(ss_initial.D)
    n_m = length(D)
    D_seq        = Vector{Vector{Float64}}(undef, T)   # D_seq[t] = D_{t-1}
    ts_pullbacks = Vector{Function}(undef, T - 1)
    agg_data     = [Vector{Float64}(undef, T - 1) for _ in het_keys]

    for t in 1:T-1
        D_seq[t] = copy(D)
        D_new, pb = ChainRulesCore.rrule(transition_step,
                                          policy_seqs[endog_var][t], D,
                                          Λ_exog, endog_dim, n_exog_states)
        D = Vector{Float64}(D_new)
        ts_pullbacks[t] = pb
        for (j, varname) in enumerate(het_keys)
            agg_data[j][t] = dot(vec(policy_seqs[varname][t]), D)
        end
    end
    D_seq[T] = copy(D)

    result = NamedTuple{het_keys}(Tuple(agg_data))

    function ForwardIteration_pullback(Δagg)
        policy_size = size(policy_seqs[het_keys[1]][1])

        ΔD = zeros(Float64, n_m)
        Δpolicy_data = [[zeros(Float64, policy_size) for _ in 1:T-1]
                        for _ in het_keys]

        for t in T-1:-1:1
            D_t = D_seq[t+1]   # D_t: distribution after transition step t

            # Aggregation: agg[k][t] = dot(vec(policy[k][t]), D_t)
            for (j, varname) in enumerate(het_keys)
                ΔD               .+= Δagg[varname][t] .* vec(policy_seqs[varname][t])
                Δpolicy_data[j][t] .+= Δagg[varname][t] .* reshape(D_t, policy_size)
            end

            # Backprop through transition_step using the stored pullback closure
            grads = ts_pullbacks[t](ΔD)
            # grads[1]=NoTangent, grads[2]=Δpolicy_mat, grads[3]=ΔD_prev
            if endog_idx !== nothing
                Δpolicy_data[endog_idx][t] .+= grads[2]
            end
            ΔD = Vector{Float64}(grads[3])
        end

        Δpolicy_seqs = NamedTuple{het_keys}(
            Tuple(Δpolicy_data[j] for j in 1:length(het_keys))
        )

        return (NoTangent(), Δpolicy_seqs, NoTangent(), NoTangent())
    end

    return result, ForwardIteration_pullback
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

