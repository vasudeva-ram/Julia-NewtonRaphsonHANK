This repository is work in progress.
It attempts to implement the methodology in Gregor Boehl's 2024 working paper, [HANK on Speed: Robust Nonlinear Solutions using Automatic Differentiation](https://gregorboehl.com/live/hank_speed_boehl.pdf).
In short, the method can be used to find the perfect-foresight solution to a HANK model using the Newton-Raphson method in the sequence space.
The usual problem with naively implementing the Newton-Raphson method in HANK models is that the Jacobian is extremely expensive to calculate, making the iterative process very inefficient.

The methodology presented in this working paper uses innovations in the automatic differentiation (AD) to overcome this issue.
While the Jacobian itself is expensive to calculate even with AD, AD can compute an intermediary object called a Jacobian-Vector Product (or JVP) very cheaply. 
Boehl's method provides a very clever way to iteratively use the JVPs to approximate the Jacobian-related values that the Newton-Raphson method needs.
Combined with other advances made in solving HANK models in the sequence space, finding the perfect-foresigh solutions even for very large models is impressively fast.

Gregor Boehl has his own Python-based repository that implements this methodology extremely efficiently at [EconPizza](https://econpizza.readthedocs.io/en/stable/content.html). 
EconPizza is rigorously tested and impressively optimized using [JAX](https://jax.readthedocs.io/en/latest/jit-compilation.html). 
The implementation is super general: it attempts to make solving HANK models as easy as [Dynare](https://www.dynare.org/) makes solving RANK models.
So if you're looking for code to help solve your model easily, that's the repository you should be headed to.

The purpose of this repository is two-fold: (1) provide a _Julia_-based implementation of the basic elements of Boehl's method, and (2) provide a teaching tool consisting of the basic nuts-and-bolts of this method without the complizations that arise when one tries to optimize and generalize the code.
(Of course, the real reason is to help me learn how to implement this procedure, but I guess I need an excuse for making this repository public).

More to come.
