# SteadyStateJacobian.jl — Boehl (2024) sequence-space Jacobian at the SS.
#
# The full residual map F(x) = Residuals(assemble(x, ForwardIteration(BackwardIteration(x))))
# is decomposed by the chain rule into three components:
#
#   JDI  — direct Jacobian:    dF/dx  with policies held fixed at SS
#   JBI  — backward Jacobian:  d(policy seqs)/dx  (ForwardDiff JVPs)
#   JFI  — forward Jacobian:   dF/d(policy seqs)  (Zygote VJPs)
#
# The combined Jacobian is block-Toeplitz at SS (time-translation invariance).
# Only one block column is computed (by perturbing x at one interior period) and
# the full structure is recovered by the recursion in getFinalJacobian.
#
# Direct blocks (dFbydX, lag blocks, lead blocks) are added at the corner and
# along the edges of the JacobianHelper matrix; the Toeplitz recursion propagates
# them to the correct diagonals of the final Jacobian.
#
# Sparsity: no dense n_m x n_m matrix is ever formed.  The VJP pullback through
# ForwardIteration uses transition_step's custom rrule (ForwardIteration.jl),
# which keeps all operations as O(n_m) sparse MVMs.
#
# Square-system invariant: Newton-Raphson requires n_eq == n_endog.  This is
# enforced by a @assert in getSteadyStateJacobian.  Throughout this file we use
# n_endog everywhere (not a separate n_eq) since they are always identical.


"""
    getSteadyStateJacobian(ss::SteadyState, model::SequenceModel) -> SparseMatrixCSC

Top-level entry point. Builds constant SS exogenous paths, calls the three
component Jacobian functions, assembles the helper matrix, applies the
block-Toeplitz recursion, and returns the consolidated sparse Jacobian
of size n_endog*(T-1) x n_endog*(T-1).

`ss` should be the ending steady state (the point around which the
transition path is linearised).

Asserts that length(model.equations) == model.compspec.n_endog, i.e. the
system is square, as required for Newton-Raphson.
"""
function getSteadyStateJacobian(ss::SteadyState, model::SequenceModel)
    @unpack T = model.compspec
    @assert length(model.equations) == model.compspec.n_endog (
        "System is not square: $(length(model.equations)) equations but " *
        "$(model.compspec.n_endog) endogenous variables. " *
        "Newton-Raphson requires n_eq == n_endog.")

    endog_keys    = vars_of_type(model, :endogenous)
    exog_keys     = vars_of_type(model, :exogenous)
    het_keys      = vars_of_type(model, :heterogeneous)

    # Computed once here and passed to all sub-functions to avoid duplication.
    xVec_endog    = repeat(Float64[ss.vars[k] for k in endog_keys], T - 1)
    exog_paths_ss = NamedTuple{exog_keys}(
        Tuple(fill(Float64(ss.vars[k]), T - 1) for k in exog_keys))
    agg_seqs_ss   = NamedTuple{het_keys}(
        Tuple(fill(Float64(ss.vars[k]), T - 1) for k in het_keys))

    JDI = getDirectJacobian(ss, xVec_endog, agg_seqs_ss, exog_paths_ss, model)
    JBI, JFI = getIntdJacobians(ss, xVec_endog, exog_paths_ss, model)

    JacobianHelper = getJacobianHelper(JBI, JFI, JDI, model)
    matrixJacobian = getFinalJacobian(JacobianHelper, JDI, model)
    return getConsolidatedJacobian(matrixJacobian, model)
end


"""
    getDirectJacobian(ss, xVec_endog, agg_seqs_ss, exog_paths_ss, model)
        -> (blocks, k)

Computes the direct Jacobian: the sensitivity of the residuals to the
endogenous sequence when policy sequences are held fixed at their SS values.

## Perturbed period

Uses a symmetric buffer k = max(max_lag, max_lead) and perturbs period
p = T-1-k. This guarantees:
  - p + k >= p + max_lag: all lag effects land within the transition path
  - p - k <= p - max_lead: all lead effects land within the transition path

By the block-Toeplitz property, all blocks extracted at period p are identical
to those that would be obtained at any other interior period.

## Extracted blocks

Always extracts 2k+1 consecutive n_endog x n_endog matrices stored in `blocks`,
a 1-indexed Vector where:

  blocks[j] = dz_{p + delta} / dx_p,   delta = j - 1 - k

  j = 1    -> delta = -k  (deepest lead slot; zero if max_lead < k)
  j = k+1  -> delta =  0  (contemporaneous / direct block)
  j = 2k+1 -> delta = +k  (deepest lag slot; zero if max_lag < k)

Slots where |delta| > max_lag (lag side) or |delta| > max_lead (lead side) are
zero by construction — no residual at that distance responds to the perturbation.
The uniform 2k+1 length simplifies downstream loops.

## Edge placement in JacobianHelper

The blocks are placed along the edges of the (T-1) x (T-1) JacobianHelper array:

  delta = 0  -> corner JacobianHelper[T-1, T-1]         (direct block)
  delta > 0  -> column JacobianHelper[T-1-delta, T-1]   (right edge, stepping up)
  delta < 0  -> row    JacobianHelper[T-1, T-1+delta]   (top edge, stepping left)

Zero blocks produce no-op additions. The Toeplitz recursion in getFinalJacobian
propagates each non-zero edge entry to the correct off-diagonal of the final
Jacobian.
"""
function getDirectJacobian(ss::SteadyState, xVec_endog::Vector{Float64},
                           agg_seqs_ss, exog_paths_ss, model::SequenceModel)
    @unpack T, n_endog, max_lag, max_lead = model.compspec

    n = n_endog * (T - 1)

    function fullFunc(xVec_e::AbstractVector)
        padded_xMat = assemble_full_xMat(xVec_e, agg_seqs_ss, exog_paths_ss,
                                         model, ss, ss)
        return Residuals(padded_xMat, model)
    end

    k     = max(max_lag, max_lead)   # symmetric buffer
    p     = T - 1 - k               # perturbed transition period (1-indexed)
    k_e   = (p - 1) * n_endog       # offset of period p in xVec_endog
    k_res = (p - k - 1) * n_endog   # offset of period p-k in residuals (first slot)

    @assert p >= 1 && p + k <= T - 1 (
        "Perturbed period p=$p is out of range for T=$T, k=$k.")

    idmat   = sparse(1.0I, n, n)
    JDI_raw = spzeros((T - 1) * n_endog, n_endog)
    for i in 1:n_endog
        JDI_raw[:, i] = JVP(fullFunc, xVec_endog, idmat[:, k_e + i])
    end

    # Extract 2k+1 consecutive slices starting at k_res.
    # blocks[j] corresponds to offset delta = j - 1 - k.
    # Slots where |delta| > max_lag or |delta| > max_lead are zero by construction.
    blocks = [JDI_raw[k_res + (j-1)*n_endog + 1 : k_res + j*n_endog, :]
              for j in 1:2k+1]

    return (blocks = blocks, k = k)
end


"""
    getIntdJacobians(ss, xVec_endog, exog_paths_ss, model) -> (JBI, JFI)

Computes the two intermediate Jacobians needed for the indirect
(policy-mediated) part of the full sequence-space Jacobian.

### JBI — backward Jacobian  (nJ x n_endog, sparse)

d(flattened policy sequences)/dx_{T-1} via ForwardDiff JVPs through
BackwardIteration. Perturbs each of the n_endog variables at the last
transition period. By time-translation invariance the resulting column blocks
characterise the full Toeplitz structure.

### JFI — forward Jacobian  (n_endog x nJ, sparse)

dz_{T-1}/d(flattened policy sequences) via a single Zygote pullback
through ForwardIteration -> assemble_full_xMat -> Residuals, seeded with
each of the n_endog unit vectors at the last residual period.
The custom rrule for transition_step ensures no dense n_m x n_m matrix
is materialised during the pullback.

nJ = n_agg * Tv * (T-1) where Tv = n_total(model.heterogeneity).

### Flat policy layout

flatten_policies and unflatten_policies are strict inverses assuming the
key order of policy_seqs matches vars_of_type(model, :heterogeneous). The
layout is: all T-1 time slices for variable 1 (each flattened column-major),
then all T-1 slices for variable 2, etc. See test_SteadyState.jl for a
concrete round-trip verification.

### Sparsity

The seed vectors passed to the Zygote pullback have length n_endog*(T-1) ~= 600
for KS — harmless as sparse or dense. The returned gradient `grad` is a dense
Vector of length nJ (flat policy sequences); for large grids this can be O(10s
of MB) but is not an n_m x n_m matrix. sparse(grad) is called immediately to
recover sparsity before storing in JFI.
"""
function getIntdJacobians(ss::SteadyState, xVec_endog::Vector{Float64},
                          exog_paths_ss, model::SequenceModel)
    @unpack T, n_endog = model.compspec
    Tv    = n_total(model.heterogeneity)
    n     = n_endog * (T - 1)
    n_res = n_endog * (T - 1)    # == n since n_eq == n_endog (asserted in caller)
    n_agg = length(vars_of_type(model, :heterogeneous))
    nJ    = n_agg * Tv * (T - 1)

    policy_size = size(first(values(ss.policies)))

    a_vec = vcat([repeat(vec(pol), T - 1) for pol in values(ss.policies)]...)

    idmat_e   = sparse(1.0I, n,     n)      # seeds for JBI JVPs
    idmat_res = sparse(1.0I, n_res, n_res)  # seeds for JFI pullback

    JBI = spzeros(nJ, n_endog)
    JFI = spzeros(n_endog, nJ)

    # Flatten NamedTuple of policy sequences -> single vector (for AD).
    # Layout: [var1_t1_flat | var1_t2_flat | ... | var1_t(T-1)_flat | var2_t1_flat | ...]
    # Key order follows values(policy_seqs), which must match vars_of_type(model, :heterogeneous).
    function flatten_policies(policy_seqs::NamedTuple)
        return vcat([vcat([vec(mat) for mat in seq]...)
                     for seq in values(policy_seqs)]...)
    end

    # Inverse of flatten_policies under the same key-order assumption.
    function unflatten_policies(a_flat)
        n_per_var = Tv * (T - 1)
        seqs = ntuple(n_agg) do k
            offset = (k - 1) * n_per_var
            [reshape(a_flat[offset + (i-1)*Tv + 1 : offset + i*Tv], policy_size)
             for i in 1:T-1]
        end
        return NamedTuple{vars_of_type(model, :heterogeneous)}(seqs)
    end

    # backFunc: xVec_endog -> flat policy sequences (ForwardDiff-compatible)
    function backFunc(xVec_e::AbstractVector)
        policy_seqs = BackwardIteration(xVec_e, exog_paths_ss, model, ss)
        return flatten_policies(policy_seqs)
    end

    # forwardFunc: flat policy sequences -> residuals (endogenous x held at SS)
    function forwardFunc(a_flat)
        policy_seqs = unflatten_policies(a_flat)
        agg_seqs    = ForwardIteration(policy_seqs, model, ss)
        padded_xMat = assemble_full_xMat(xVec_endog, agg_seqs, exog_paths_ss,
                                         model, ss, ss)
        return Residuals(padded_xMat, model)
    end

    # JBI: JVP through backFunc, perturbing last-period endogenous vars
    for i in 1:n_endog
        JBI[:, i] = JVP(backFunc, xVec_endog, idmat_e[:, n - n_endog + i])
    end

    # JFI: Zygote VJP through forwardFunc, seeding last-period residuals.
    # idmat_res[:, j] is a SparseVector of length n_endog*(T-1); passed directly
    # (no Vector() conversion) since Zygote pullback accepts AbstractVector.
    # No n_m x n_m matrix forms inside because transition_step has a custom rrule.
    _, pullback = Zygote.pullback(forwardFunc, a_vec)
    for i in 1:n_endog
        grad       = pullback(idmat_res[:, n_res - n_endog + i])[1]
        JFI[i, :] = sparse(grad)
    end

    return JBI, JFI
end


"""
    getJacobianHelper(JBI, JFI, JDI, model::SequenceModel)
        -> Matrix{SparseMatrixCSC}

Assembles the (T-1) x (T-1) array of n_endog x n_endog blocks that combines
the indirect Jacobian (JFI * JBI) with the direct blocks from JDI.

### Indirect blocks

JacobianHelper[t, s] = JFI[:, t-th slice] * JBI[s-th slice, :]

captures the indirect effect: policy set at period s influences residuals at t.

### Direct blocks

The direct blocks from JDI are placed along the edges of JacobianHelper:

  delta = 0     (direct):    JacobianHelper[T-1,       T-1      ] += blocks[j]
  delta > 0     (lag-delta): JacobianHelper[T-1-delta, T-1      ] += blocks[j]
  delta < 0     (lead-delta):JacobianHelper[T-1,       T-1+delta] += blocks[j]

The Toeplitz recursion in getFinalJacobian propagates each edge entry to the
correct off-diagonal of the final Jacobian:

  - Corner [T-1, T-1] -> main diagonal of J̄
  - Right column [T-1-delta, T-1] -> delta-th lower diagonal of J̄
  - Top row [T-1, T-1-|delta|]   -> |delta|-th upper diagonal of J̄

### Efficiency note

The double loop performs O(T^2) small matrix products (n_endog x n_r times
n_r x n_endog). A future optimisation can exploit the diagonal Toeplitz
structure to reduce this to O(T) unique products.
"""
function getJacobianHelper(JBI, JFI, JDI, model::SequenceModel)
    @unpack T, n_endog = model.compspec
    n_r  = n_total(model.heterogeneity)

    JacobianHelper = [spzeros(n_endog, n_endog) for _ in 1:T-1, _ in 1:T-1]

    # Indirect blocks: O(T^2) products
    for t in 1:T-1
        for s in 1:T-1
            JacobianHelper[t, s] = JFI[:, (t-1)*n_r + 1 : t*n_r] *
                                   JBI[(s-1)*n_r + 1 : s*n_r, :]
        end
    end

    # Direct blocks: placed at the corner and along the edges.
    # blocks[j] corresponds to offset delta = j - 1 - k (symmetric around j=k+1).
    # Zero blocks (where |delta| > max_lag or |delta| > max_lead) are no-ops.
    k = JDI.k
    for j in 1:2k+1
        delta = j - 1 - k   # delta < 0 = lead, 0 = direct, > 0 = lag
        if delta == 0
            JacobianHelper[T-1,       T-1      ] += JDI.blocks[j]
        elseif delta > 0
            JacobianHelper[T-1-delta, T-1      ] += JDI.blocks[j]
        else  # delta < 0 (lead)
            JacobianHelper[T-1,       T-1+delta] += JDI.blocks[j]
        end
    end

    return JacobianHelper
end


"""
    getFinalJacobian(JacobianHelper, JDI, model::SequenceModel)
        -> Matrix{SparseMatrixCSC}

Applies the block-Toeplitz recursion to recover the full (T-1) x (T-1) array
of n_endog x n_endog Jacobian blocks from the helper matrix.

### Recursion

  J̄[s, t] = JacobianHelper[T-s, T-t]              if s < 2 or t < 2
  J̄[s, t] = J̄[s-1, t-1] + JacobianHelper[T-s, T-t]   otherwise

### Boundary corrections

The Toeplitz recursion is derived from the infinite-horizon limit. In the
finite-horizon system the boundary values x_0 = x^{SS} (left) and x_T = x^{SS}
(right) are pinned, which breaks the Toeplitz structure near the boundaries.

Left boundary (x_0 pinned):
  J̄[1,1] += blocks[k+2]   (the delta=+1 lag block)
  Empirically verified for max_lag=1 by comparing against a full JVP of the
  complete model. At t=1, x_0 appears as lag-1 in z_1 but is fixed, so its
  missing column's contribution is absorbed here.

  TODO: for max_lag > 1, additional boundary corrections at J̄[s,1] for
  s=2..max_lag may be needed. Verify empirically by comparing to a full JVP
  on a model with max_lag=2.

Right boundary (x_T pinned):
  TODO: by symmetry with the left boundary, models with max_lead > 0 may need
  J̄[T-1, T-1] += blocks[k]  (the delta=-1 lead block).  Verify empirically.
"""
function getFinalJacobian(JacobianHelper, JDI, model::SequenceModel)
    @unpack T, n_endog = model.compspec

    J̅ = [spzeros(n_endog, n_endog) for _ in 1:T-1, _ in 1:T-1]

    for t in 1:T-1
        for s in 1:T-1
            if s < 2 || t < 2
                J̅[s, t] = JacobianHelper[T-s, T-t]
            else
                J̅[s, t] = J̅[s-1, t-1] + JacobianHelper[T-s, T-t]
            end
        end
    end

    # Left boundary correction (empirically verified for max_lag=1).
    # blocks[k+2] is the delta=+1 slot (lag-1), which is non-zero whenever max_lag >= 1.
    # For max_lag=0 (purely forward-looking model) k=0 and there is no lag-1 slot.
    k = JDI.k
    if k >= 1
        J̅[1, 1] += JDI.blocks[k + 2]
    end

    # Right boundary correction placeholder (unverified; see TODO above).
    # if k >= 1
    #     J̅[T-1, T-1] += JDI.blocks[k]   # delta=-1 slot (lead-1)
    # end

    return J̅
end


"""
    getConsolidatedJacobian(J̅, model::SequenceModel) -> SparseMatrixCSC

Stacks the (T-1) x (T-1) array of n_endog x n_endog blocks into a single
sparse n_endog*(T-1) x n_endog*(T-1) matrix.

Block J̄[s, t] maps to rows (s-1)*n_endog+1 : s*n_endog and
columns (t-1)*n_endog+1 : t*n_endog of the output.
"""
function getConsolidatedJacobian(J̅, model::SequenceModel)
    @unpack T, n_endog = model.compspec

    J̄ = spzeros((T-1)*n_endog, (T-1)*n_endog)
    for t in 1:T-1
        for s in 1:T-1
            J̄[(s-1)*n_endog+1 : s*n_endog, (t-1)*n_endog+1 : t*n_endog] = J̅[s, t]
        end
    end

    return J̄
end
