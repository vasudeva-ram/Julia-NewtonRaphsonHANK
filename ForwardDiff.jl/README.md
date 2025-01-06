# IMPORTANT: Vasu's note
This is the version of ForwardDiff.jl that is currently used in evaluating forward mode autodifferentiation.
It differs from the release branch version `0.10.38` in that it contains a [cherry-picked](https://stackoverflow.com/questions/9339429/what-does-cherry-picking-a-commit-with-git-mean) merge of [PR #481](https://github.com/JuliaDiff/ForwardDiff.jl/pull/481) from the master DEV branch of the ForwardDiff module.

**For future Vasu and/or Vasu-replacements**: The use of this version of ForwardDiff is pinned, which means that `Pkg.update()` will not replace this version with future ForwardDiff versions.
If #481 ever makes it from the master DEV branch to release, you can unpin this and update as needed.

## More details on this version
For the implementation of the Newton Raphson method in solving Het-agent models, I use both ForwardDiff and Zygote packages for forward mode and reverse mode automatic differentiation. However, in evaluating the Jacobian of the model, I run into a problem.

The problem has two elements. First, the jacobian is stored as a sparse matrix for speed and efficiency purposes.
Second, the jacobian is evaluated at the steady state where the residuals of the model are approximately zero.
In the case of the exogenous variables, it is precisely zero.
This poses a problem because ForwardDiff evaluates the sparsity structure of the steady state before evaluating the gradient, which results in the derivative of the exogenous variable being evaluated as 0.0 when it should be non-zero.
This problem is fixed by Pull Request #481 of the ForwardDiff module.

Identifying this problem was hard enough, but addressing it is further complicated by the fact that the pull request is only implemented in the DEV branch of the module.
I tried to use the DEV branch directly (instead of the release branch), but this ended up having conflicts with the current version of Zygote.
I tried using ReverseDiff instead, but ran into the same problem.
Thus, as a temporary solution, I have cherry picked the updates implemented in #481 and implemented them in this version of ForwardDiff.jl.
This version is pinned so that future releases of ForwardDiff do not replace this.
In the future, this will have to be revisited and the 48-updated version of the ForwardDiff release would have to be used.



[![CI](https://github.com/JuliaDiff/ForwardDiff.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/ForwardDiff.jl/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=master)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/ForwardDiff.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/ForwardDiff.jl/dev)

# ForwardDiff.jl

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff generally outperform non-AD algorithms (such as finite-differencing) in both speed and accuracy.

Here's a simple example showing the package in action:

```julia
julia> using ForwardDiff

julia> f(x::Vector) = sin(x[1]) + prod(x[2:end]);  # returns a scalar

julia> x = vcat(pi/4, 2:4)
4-element Vector{Float64}:
 0.7853981633974483
 2.0
 3.0
 4.0

julia> ForwardDiff.gradient(f, x)
4-element Vector{Float64}:
  0.7071067811865476
 12.0
  8.0
  6.0

julia> ForwardDiff.hessian(f, x)
4×4 Matrix{Float64}:
 -0.707107  0.0  0.0  0.0
  0.0       0.0  4.0  3.0
  0.0       4.0  0.0  2.0
  0.0       3.0  2.0  0.0
```

Functions like `f` which map a vector to a scalar are the best case for reverse-mode automatic differentiation,
but ForwardDiff may still be a good choice if `x` is not too large, as it is much simpler.
The best case for forward-mode differentiation is a function which maps a scalar to a vector, like this `g`:

```julia
julia> g(y::Real) = [sin(y), cos(y), tan(y)];  # returns a vector

julia> ForwardDiff.derivative(g, pi/4)
3-element Vector{Float64}:
  0.7071067811865476
 -0.7071067811865475
  1.9999999999999998

julia> ForwardDiff.jacobian(x) do x  # anonymous function, returns a length-2 vector
         [sin(x[1]), prod(x[2:end])]
       end
2×4 Matrix{Float64}:
 0.707107   0.0  0.0  0.0
 0.0       12.0  8.0  6.0
```

See [ForwardDiff's documentation](https://juliadiff.org/ForwardDiff.jl/stable) for full details on how to use this package.
ForwardDiff relies on [DiffRules](https://github.com/JuliaDiff/DiffRules.jl) for the derivatives of many simple function such as `sin`.

See the [JuliaDiff web page](https://juliadiff.org) for other automatic differentiation packages.

## Publications

If you find ForwardDiff useful in your work, we kindly request that you cite [the following paper](https://arxiv.org/abs/1607.07892):

```bibtex
@article{RevelsLubinPapamarkou2016,
    title = {Forward-Mode Automatic Differentiation in {J}ulia},
   author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
  journal = {arXiv:1607.07892 [cs.MS]},
     year = {2016},
      url = {https://arxiv.org/abs/1607.07892}
}
```
