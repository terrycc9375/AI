Nuclear Norm Regularization for Deep Learning

Christopher Scarvelis
MIT CSAIL
scarv@mit.edu

Justin Solomon
MIT CSAIL
jsolomon@mit.edu

Abstract

Penalizing the nuclear norm of a function’s Jacobian encourages it to locally behave
like a low-rank linear map. Such functions vary locally along only a handful of
directions, making the Jacobian nuclear norm a natural regularizer for machine
learning problems. However, this regularizer is intractable for high-dimensional
problems, as it requires computing a large Jacobian matrix and taking its SVD. We
show how to efficiently penalize the Jacobian nuclear norm using techniques tailor-
made for deep learning. We prove that for functions parametrized as compositions
f = g ◦h, one may equivalently penalize the average squared Frobenius norms of
Jg and Jh. We then propose a denoising-style approximation that avoids Jacobian
computations altogether. Our method is simple, efficient, and accurate, enabling
Jacobian nuclear norm regularization to scale to high-dimensional deep learning
problems. We complement our theory with an empirical study of our regularizer’s
performance and investigate applications to denoising and representation learning.

1
Introduction

Building models that adapt to the structure of their data is a key challenge in machine learning. As
real-world data typically concentrates on low-dimensional manifolds, a good model f should only
be sensitive to changes to its inputs along the data manifold. One may encourage this behavior by
regularizing f so that its Jacobian Jf[x] has low rank. This causes f to locally behave like a low-rank
linear map and therefore be locally constant in directions that are in the kernel of Jf[x].

How should we regularize f so that its Jacobians have low rank? Directly penalizing rank(Jf[x])
during training is challenging, as the rank function is not differentiable. In light of this, the nuclear
norm ∥Jf[x]∥∗is an appealing alternative: Being the L1 norm of a matrix’s singular values, it is the
tightest convex relaxation of the rank function, and as it is differentiable almost everywhere, it can be
included in standard deep learning pipelines.

One critical flaw in this strategy is its computational cost. The nuclear norm of a matrix is the sum
of its singular values, so to penalize ∥Jf[x]∥∗in training, one must (1) compute the Jacobian of a
typically high-dimensional map f : Rn →Rm, (2) take the singular value decomposition (SVD) of
this m × n matrix, (3) sum its singular values, and (4) differentiate through each of these operations.
The combined cost of these operations is prohibitive for high-dimensional data. Consequently, nuclear
norm regularization has yet to be widely adopted by the deep learning community.

This work shows how to efficiently penalize the Jacobian nuclear norm ∥Jf[x]∥∗using techniques
tailor-made for deep learning. We first show that parametrizing f as a composition f = g ◦h
– a feature common to all deep learning pipelines – allows one to replace the expensive nuclear
norm ∥Jf[x]∥∗with two squared Frobenius norms ∥Jh[x]∥2
F and ∥Jg[h(x)]∥2
F , which admit an
elementwise computation that avoids a costly SVD. We prove that the resulting problem is exactly
equivalent to the original problem with the nuclear norm penalty. We in turn approximate this
by a denoising-style objective that avoids the Jacobian computation altogether. Our method is

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
simple, efficient, and accurate – both in theory and in practice – and enables Jacobian nuclear norm
regularization to scale to high-dimensional deep learning problems.

We complement our theoretical results with an empirical study of our regularizer’s performance on
synthetic data. As the Jacobian nuclear norm has seldom been used as a regularizer in deep learning,
we propose applications of our method to unsupervised denoising, where one trains a denoiser given
a dataset of noisy images without access to their clean counterparts, and to representation learning.
Our work makes the Jacobian nuclear norm a feasible component of deep learning pipelines,
enabling users to learn locally low-rank functions unencumbered by the heavy cost of naïve Jacobian
nuclear norm regularization.

2
Related work

Nuclear norm regularization.
As penalizing the nuclear norm in matrix learning problems en-
courages low-rank solutions, nuclear norm regularization (NNR) has been widely used throughout
machine learning. Rennie and Srebro [2005] propose NNR for collaborative filtering, where one
attempts to predict user interests by aggregating incomplete information from a large pool of users.
Candès et al. [2011] introduce robust PCA, which decomposes a noisy matrix into low-rank and
sparse parts and uses nuclear norm regularization to learn the low-rank part. Cabral et al. [2013], Dai
et al. [2014] use NNR to regularize the ill-posed structure-from-motion problem, which recovers a
3D scene from a set of 2D images.

A line of work beginning with Candès and Recht [2012] takes inspiration from compressed sensing and
studies the conditions under which a low-rank matrix can be perfectly recovered from a small sample
of its entries via nuclear norm minimization. This work was followed by Candès and Tao [2010],
Keshavan et al. [2009], Recht [2011], which progressively sharpen the bounds on the number of
samples required for exact recovery. In parallel, Cai et al. [2010] propose singular value thresholding
(SVT), a simple algorithm for approximate nuclear norm minimization that avoids solving a costly
semidefinite program as with earlier algorithms. However, SVT still requires computing a singular
value decomposition in each iteration, which is an onerous requirement for large matrices. Motivated
by this challenge, Rennie and Srebro [2005] show how to convert a nuclear norm-regularized matrix
learning problem into an equivalent non-convex problem involving only squared Frobenius norms.
Our work generalizes their method to non-linear learning problems.

Jacobian regularization in deep learning.
Sokoli´c et al. [2017], Varga et al. [2017], Hoffman et al.
[2020] penalize the spectral and Frobenius norms of a neural net’s Jacobian with respect to its inputs
to improve classifier generalization, particularly in the low-data regime. Similarly, Jakubovitz and
Giryes [2018] fine-tune neural classifiers with Jacobian Frobenius norm regularization to improve
adversarial robustness. Unlike our work, these papers do not consider nuclear norm regularization.

Neural ODEs [Chen et al., 2018] parametrize functions as solutions to initial value problems with
neural velocity fields. Ensuring that the learned dynamics are well-conditioned minimizes the number
of steps required to accurately solve these ODEs. To this end, Finlay et al. [2020] penalize the squared
Frobenius norm of the velocity field’s Jacobian and observe a tight relationship between the value of
this norm and the step size of an adaptive ODE solver. Kelly et al. [2020] extend this approach by
regularizing higher-order derivatives as well.

Finally, it has been observed as early as Webb [1994], Bishop [1995] that training a neural net on
noisy data approximately penalizes the squared Frobenius norm of the network Jacobian at training
data. Inspired by this observation, Vincent et al. [2008, 2010] propose denoising autoencoders for
representation learning, and Rifai et al. [2011] propose directly penalizing the squared Frobenius
norm of the encoder Jacobian to encourage robust latent representations. Alain and Bengio [2014]
show that an autoencoder trained with a penalty on the squared Frobenius norm of its Jacobian learns
the score of the data distribution for small regularization values. Recently, Kadkhodaie et al. [2024]
employ a similar analysis of the denoising objective to study the generalization of diffusion models.

Denoising via singular value shrinkage.
While a full survey of the denoising literature is out
of scope (see e.g. Elad et al. [2023]), we highlight a handful of works that employ singular value
shrinkage (SVS) to denoise low-rank data corrupted by isotropic noise given a noisy data matrix.
SVS denoises a noisy data matrix Y by applying a shrinkage function ϕ to its singular values σd.

2


---Page Break---
This function shrinks small singular values of Y while leaving larger singular values untouched.
Shabalin and Nobel [2010] study the conditions under which it suffices to rescale the noisy data’s
singular values while preserving its singular vectors. Gavish and Donoho [2014] study the optimal
hard thresholding policy, which sets all singular values below a threshold to 0 while preserving the
rest. Nadakuditi [2014] considers general shrinkage policies whose error is measured by the squared
Frobenius distance between the true and denoised data matrix. Gavish and Donoho [2017] then study
optimal shrinkage policies under a larger class of error measures. All of these methods assume that
the clean data matrix is low-rank and hence that the data globally lies in a low-dimensional subspace.
Under the manifold hypothesis, real-world data such as images typically lie on low-dimensional
manifolds and hence locally lie in low-dimensional subspaces. Inspired by this observation, we use
our method in Section 5 to learn an image denoiser given a dataset of exclusively noisy images.

3
Method

In this section, we introduce our method for Jacobian nuclear norm regularization. We first prove
that for functions parametrized as compositions f = g ◦h (which include all non-trivial neural
networks), one may replace the expensive nuclear norm penalty ∥Jf[x]∥∗in a learning problem with
the average of two squared Frobenius norms 1

2
 
∥Jh[x]∥2
F + ∥Jg[h(x)]∥2
F

and obtain an equivalent
learning problem. These squared Frobenius norms admit an elementwise computation that avoids
a costly SVD in computing ∥Jf[x]∥∗. As the Jacobian computation is itself costly for large neural
nets, we use a first-order Taylor expansion and Hutchinson’s trace estimator [Hutchinson, 1989]
to estimate the Frobenius norm terms. Computing a loss with our regularizer costs as few as two
additional evaluations of g and h per sample, allowing our method to scale to large neural networks
with high-dimensional inputs and outputs.

3.1
Preliminaries

Any matrix A ∈Rm×n admits a singular value decomposition (SVD) A = UΣV ⊤, where U ∈
Rm×m and V ∈Rn×n are orthogonal matrices, and Σ ∈Rm×n is a diagonal matrix storing the
singular values σi ≥0 of A. The rank of A is equal to its number of non-zero singular values.

The nuclear norm ∥A∥∗of a matrix A ∈Rm×n is the sum of its singular values: ∥A∥∗= P

i σi.
As σi ≥0 by definition, ∥A∥∗is also the L1 norm of its vector of singular values. Just as L1
regularization steers learning problems towards solutions with many zero entries, nuclear norm
regularization steers matrix learning problems towards low-rank solutions with many zero singular
values. For this reason, nuclear norm regularization has seen widespread use in problems ranging
from collaborative filtering [Rennie and Srebro, 2005, Candès and Recht, 2012] to robust PCA
[Candès et al., 2011] to structure-from-motion [Cabral et al., 2013, Dai et al., 2014].

However, ∥A∥∗is costly to compute because it requires a cubic-time SVD. Furthermore, even efficient
algorithms for nuclear norm minimization such as Cai et al. [2010]’s singular value thresholding
require one SVD per iteration, which is prohibitive for very large matrices A. Motivated by this
computational challenge, Rennie and Srebro [2005, Lemma 1] state that one can compute ∥A∥∗by
solving a non-convex optimization problem:

∥A∥∗=
min
U,V :UV ⊤=A
1
2
 
∥U∥2
F + ∥V ∥2
F

.
(1)

For completeness, we prove this identity in Appendix A.1. Using this result, Rennie and Srebro
[2005] show that the following problems are equivalent:

min
A∈Rm×n ℓ(A) + η∥A∥∗=
min
U∈Rm×r

V ∈Rn×r
ℓ(UV ⊤) + η

2
 
∥U∥2
F + ∥V ∥2
F

,
(2)

where r = min (m, n) and ℓis a generic differentiable loss function. As 1

2∥U∥2
F and its derivative
∇U 1

2∥U∥2
F = U can be computed elementwise without a costly SVD, the RHS objective in (2) is
amenable to gradient-based optimization.

3


---Page Break---
3.2
Our key result

Equation (2) enables the use of simple and efficient gradient-based methods for learning a low-rank
linear map by parametrizing the matrix A as a composition A = UV ⊤of linear maps U and V ⊤. In
deep learning, one is interested in learning non-linear functions that are parametrized by compositions
of simpler functions. Such functions f are differentiable almost everywhere, so they are locally
well-approximated by linear maps specified by their Jacobians Jf[x].

Encouraging the learned function to have a low-rank Jacobian is a natural prior: It corresponds to
learning a function that locally behaves like a low-rank linear map. Such functions vary locally along
only a handful of directions and are constant in the remaining directions. When the training data is
supported on a low-dimensional manifold, these directions correspond to the tangents and normals,
respectively, to the data manifold. One may implement this low-rank prior on Jf by solving the
following optimization problem:

inf
f:Rn→Rm
E
x∼D(Ω) [ℓ(f(x), x) + η∥Jf[x]∥∗] ,
(3)

where ℓis a generic differentiable loss function and D(Ω) is a data distribution supported on Ω⊆Rn.
If f is parametrized as a neural network and n, m are large, this problem is costly to optimize via
stochastic gradient descent, as Jf[x] ∈Rm×n and computing the subgradient of ∥Jf[x]∥∗requires a
cubic-time SVD. In fact, simply storing Jf[x] in memory is often intractable for large n, m, which is
typical when f is an image-to-image map. For example, if f is a denoiser operating on 1024 × 1024
RGB images, its inputs are 3 × 1024 × 1024 = 3,145,728-dimensional, and Jf[x] occupies nearly
40 TB of memory.

To address these challenges, we first prove a theorem generalizing (2) to non-linear functions. We then
show how to avoid computing Jf[x] altogether using a first-order Taylor expansion and Hutchinson’s
estimator. Our primary contribution is the following result:

Theorem 3.1 Let D(Ω) be a data distribution supported on a compact set Ω⊆Rn with measure µ
that is absolutely continuous with respect to the Lebesgue measure on Ω. Let ℓ∈C1(Rm × Rn) be
a continuously differentiable loss function. Then,

inf
f∈C∞(Ω)
E
x∼D(Ω) [ℓ(f(x), x) + η∥Jf[x]∥∗]

=
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
E
x∼D(Ω)

h
ℓ(g(h(x)), x) + η

2
 
∥Jg[h(x)]∥2
F + ∥Jh[x]∥2
F
i
.
(4)

On the left-hand side, we learn a function f : Rn →Rm given fixed input and output dimensions
n, m. On the right-hand side, we learn functions h : Rn →Rd and g : Rd →Rm with n, m fixed
but optimize over the inner dimension d. We prove this theorem in Appendix A.2 and sketch the
proof below. Theorem 3.1 shows that by parametrizing f as a composition of g and h – a feature
common to all deep learning pipelines – one may learn a locally low-rank function without computing
expensive SVDs during training.

Proof sketch.
We denote the left-hand side objective by EL(f) and its inf by (L); we denote the
right-hand side objective by ER(g, h) and its inf by (R). We prove that (L) ≤(R) and (R) ≤(L).

(L) ≤(R) is the easy direction. The basic observation is that if f = g ◦h, then Jf[x] =
Jg[h(x)]Jh[x] by the chain rule. Equation (1) then implies that

∥Jf[x]∥∗=
min
U,V :UV ⊤=Jf[x]
1
2
 
∥U∥2
F + ∥V ∥2
F

≤1

2
 
∥Jg[h(x)]∥2
F + ∥Jh[x]∥2
F

.

(R) ≤(L) is the hard direction. The proof strategy is as follows:

1. We begin with a function fm ∈C∞(Ω) such that EL(fm) is arbitrarily close to its inf over
C∞(Ω). We use fm to construct parametric families of affine functions gz
m, hz
m whose
composition is a good local approximation to fm in a neighborhood of z ∈Ω, both pointwise
and in terms of the contributions to ER(gz
m, hz
m) and EL(fm), resp., due to x ∈Ωnear z.

4


---Page Break---
2. We then stitch together these local approximations to form a sequence of global approxima-
tions gk
m, hk
m to fm. These functions are piecewise affine and hence not regular enough to
lie in C∞(Ω) as required by the right-hand side of Equation (4).

3. Finally, mollifying the piecewise affine functions gk
m, hk
m yields a minimizing sequence of
C∞(Ω) functions gk
m,ϵ, hk
m,ϵ such that ER(gk
m,ϵ, hk
m,ϵ) approaches the inf of EL.

While Theorem 3.1 shows how to regularize a learning problem with the Jacobian nuclear norm
without a cubic-time SVD, the Jacobian computation incurs a quadratic time and memory cost, which
remains heavy for high-dimensional learning problems. To mitigate this issue, the following section
shows how to approximate the ∥Jg[h(x)]∥2
F and ∥Jh[x]∥2
F terms in (4).

3.3
Estimating the Jacobian Frobenius norm

When f : Rn →Rm is a function between high-dimensional spaces, Jf[x] is an m × n matrix that
is costly to compute and to store in memory. Previous works employing Jacobian regularization for
neural networks have noted this issue and proposed stochastic approximations based on Jacobian-
vector products (JVP) against random vectors [Varga et al., 2017, Hoffman et al., 2020]. As JVPs
may be costly to compute for large neural nets, we propose an alternative stochastic estimator that
requires only evaluations of f and analyze its error:

Theorem 3.2 Let f : Rn →Rm be continuously differentiable. Then,

σ2∥Jf[x]∥2
F =
E
ϵ∼N(0,σ2I)


∥f(x + ϵ) −f(x)∥2
2

+ O(σ2).
(5)

Similar results appear in the ML literature as early as Webb [1994]. Our proof in Appendix A.3 relies
on a first-order Taylor expansion and Hutchinson’s trace estimator; a similar proof is given by Alain
and Bengio [2014]. In practice, we obtain accurate approximations to ∥Jf[x]∥2
F by using a small
noise variance σ2 and rescaling the expectation in (5) by
1
σ2 to compensate. In Section 4, we also
show that a single noise sample ϵ ∼N(0, σ2I) suffices in practice.

Using this efficient approximation, we obtain the following regularizer:

R(x; f) =
1
2σ2
E
ϵ∼N(0,σ2I)


∥g(h(x) + ϵ) −g(h(x))∥2
2 + ∥h(x + ϵ) −h(x)∥2
2

,
(6)

where f = g ◦h. In practice, one may use a single draw of ϵ ∼N(0, σ2I) per training iteration
while maintaining good performance on downstream tasks; see e.g. the results in Section 5. In this
case, our regularizer R(x; f) costs merely two additional function evaluations, enabling it to scale to
large neural networks acting on high-dimensional data. In Section 4, we show that parametrizing fθ
as a neural net and solving

inf
fθ:Rn→Rm
E
x∼D(Ω) [ℓ(fθ(x), x) + ηR(x; fθ)]
(7)

yields good approximations to the solution to (3) for problems where exact solutions are known. We
then propose two applications of Jacobian nuclear norm regularization in Section 5.

4
Validation

In this section, we empirically validate our method on a special case of (3) for which closed-form
solutions are known. We consider the following problem:

inf
f:Rn→R

Z

Rn

1

2∥f(x) −τ(x)∥2
2 + η∥Jf[x]∥∗


dx,
(8)

where τ : Rn →R is the indicator function of the unit ball in Rn. As f is a scalar-valued function
in this problem, Jf[x] is a vector, and ∥Jf[x]∥∗= ∥∇f(x)∥2. This is an instance of the celebrated

5


---Page Break---
Rudin-Osher-Fatemi (ROF) model for image denoising [Rudin et al., 1992]. Meyer [2001, p. 36]
shows that the exact solution to (8) given this target function τ(x) is f(x) := (1 −nη)τ(x). This is
a rescaled indicator function of the unit ball.

We parametrize f as a multilayer perceptron (MLP) fθ and solve (8) along with the problem using
our regularizer:

inf
fθ:Rn→R
E
x∼D(Ω)

1

2∥fθ(x) −τ(x)∥2
2 + ηR(x; fθ)

,
(9)

where fθ = gθ ◦hθ. We approximate the integral over Rn by Monte Carlo integration over a box Ω
centered at the origin. We experiment with n = 2 and n = 5 and in each case depict results for a
small and a large regularization value η. We track the objective values of problems (8) and (9) and
show that they converge to the same value, as predicted by Theorem 3.1. We also track the absolute
error of both problem’s solutions across training iterations and plot solutions to each problem at
convergence for the 2D case. We give full implementation details in Appendix B.1.

(a) True solution
to (8), η = 0.1

(b) Neural so-
lution to (8)

(c) Neural so-
lution to (9)

(d)
True
solution to (8),
η = 0.25

(e) Neural so-
lution to (8)

(f) Neural solu-
tion to (9)

Figure 1: Comparison of exact and neural solutions to Problems (8) and (9) with n = 2 and η = 0.1
(first three plots) and η = 0.25 (last three plots). The x−and y−axes represent the inputs to fθ, and
colors denote function values. Solving (9) recovers an accurate approximation to the true solution for
both values of η while requiring no Jacobian nuclear norm computations.

(a) n = 2, η = 0.1
(b) n = 2, η = 0.25

(c) n = 5, η = 0.01
(d) n = 5, η = 0.05

Figure 2: Mean absolute error of neural so-
lutions to (8) (blue) and (9) (orange). Our
regularizer obtains solutions with accuracy
comparable to directly penalizing the Jaco-
bian nuclear norm.

(a) n = 2, η = 0.1
(b) n = 2, η = 0.25

(c) n = 5, η = 0.01
(d) n = 5, η = 0.05

Figure 3: Log-objective values for (8) (blue)
and (9) (orange) across training iterations. As
predicted by Theorem 3.1, both problems con-
verge to nearly identical objective values.

Figure 1 depicts the exact solution to (8) for n = 2 and two values of η, along with neural solutions
to the same problem (8) and the problem with our regularizer (9). Solving our Jacobian-free problem
(9) with 10 draws of ϵ per training iteration yields accurate solutions to the ROF problem for both
values of η, with our problem yielding slightly diffuse transitions across the boundary of the unit disc.

6


---Page Break---
Figure 2 confirms the accuracy our our method’s solutions, which attain absolute error comparable to
the neural solutions to (8).

Figure 3 depicts the objective values for Problems (8) and (9) on log scale across training iterations.
As predicted by Theorem 3.1, both converge toward nearly identical objective values; the larger gap
in Figure 3(c) is an artifact of the loss magnitudes being smaller and the plot being on log scale.

5
Applications

Unsupervised denoising.
In this section, we apply our regularizer R(x; fθ) to unsupervised
denoising. We learn a denoiser fθ that maps noisy images x + ϵ to clean images x given a training
set of noisy images without their corresponding clean images. We motivate the use of our regularizer
via a connection with denoising by singular value shrinkage, and demonstrate that our unsupervised
denoiser nearly matches the performance of a denoiser trained on clean-noisy image pairs.

Singular value shrinkage.
A line of work beginning with Shabalin and Nobel [2010] studies
denoising by singular value shrinkage (SVS). These works seek to recover a low-rank data matrix
X ∈RD×N of unknown rank r given only a single matrix Y = X + σϵZ of clean data X corrupted
by iid white noise Z. Since the clean data is low-rank, the components of Y corresponding to its
small singular values contain mostly noise, so SVS denoises Y by applying a shrinkage function ϕ to
its singular values σd. This function shrinks small singular values of Y while leaving larger singular
values untouched. For convenience, we denote the denoised matrix by ϕ(Y ).

Gavish and Donoho [2017] show that under certain assumptions on the noise Z and data X, one can
derive the optimal shrinker ϕ that asymptotically minimizes ∥X −ϕ(Y )∥2
F . In Appendix A.4, we
show that this optimal shrinker is also the solution to the following problem:

min
A∈RD×D
1
2N ∥AY −Y ∥2
F + η∥A∥∗,
(10)

where we set η = σ2
ϵ to be equal to the noise variance. This problem is a special case of the following
instance of Problem (3)

inf
fθ:RD→RD
E
y∼D(Ω)

1

2∥fθ(y) −y∥2
2 + η∥Jfθ[y]∥∗


(11)

when D(Ω) is an empirical distribution over N noisy training samples and fθ is restricted to be
a linear map. Just as (10) yields an optimal shrinker for denoising low-rank data which globally
lies in a low-dimensional subspace, we conjecture that solving (11) yields an effective denoiser for
manifold-supported data such as images, which locally lie near low-dimensional subspaces – even
when trained on noisy images. We test this conjecture by solving (11) using a neural denoiser fθ. To
make this problem tractable, we replace ∥Jfθ[y]∥∗with our regularizer R(y; fθ), which we compute
using a single draw of ϵ per training iteration.

Experiments.
We train our denoiser by solving (11) with D(Ω) being the empirical distribution
over 288k noisy images from the Imagenet training set [Russakovsky et al., 2015]. Consequently,

PSNR (dB) ↑
Imagenet
CBSD68
Method
σ = 1
σ = 2
σ = 1
σ = 2
BM3D
21.26 ± 2.81
18.71 ± 2.33
19.42 ± 1.88
16.77 ± 1.30
Ours
23.10 ± 3.12
21.05 ± 2.85
21.08 ± 2.04
19.10 ± 1.80
N2N
23.12 ± 3.05
21.21 ± 3.02
20.37 ± 1.71
19.37 ± 1.89
Supervised
23.37 ± 3.25
21.39 ± 2.97
21.62 ± 2.28
19.54 ± 1.95

Table 1: Denoiser performance via average PSNR on held-out images. Our method performs nearly
as well as a supervised denoiser, despite being trained exclusively on highly corrupted data without
access to clean images.

7


---Page Break---
(a) Ground truth (b) Noisy, σ = 1
(c) BM3D
(d) Ours
(e) N2N
(f) Supervised

(g) Ground truth (h) Noisy, σ = 2
(i) BM3D
(j) Ours
(k) N2N
(l) Supervised

Figure 4: Denoiser performance comparison on held-out image corrupted by Gaussian noise with
σ = 1 (first row) and σ = 2 (second row). Our method performs nearly as well as a supervised
denoiser, despite being trained exclusively on highly corrupted data.

our denoiser does not see any clean images during training. The clean images’ channel intensities
lie in [−1, 1], and we corrupt them with Gaussian noise with standard deviations σ ∈{1, 2}. We set
η = σ2 when solving (11). We parametrize the denoiser fθ = gθ ◦hθ as a Unet [Ronneberger et al.,
2015], letting hθ and gθ be its downsampling and upsampling blocks, resp.

We benchmark our method against a supervised denoiser trained with the usual MSE loss ∥fθ(x +
ϵ) −x∥2
2 on clean-noisy pairs, Lehtinen et al. [2018]’s Noise2Noise (N2N) method, which requires
access to independent noisy copies of each ground truth image during training, and BM3D, a classical
unsupervised denoiser [Dabov et al., 2007]. We implement the supervised and N2N denoisers
using the same Unet architecture as our denoiser, and train them on the same dataset with the same
hyperparameters. We evaluate the denoisers via their average peak signal-to-noise ratio (PSNR)
across the CBSD68 dataset [Martin et al., 2001] and across 100 randomly-drawn noisy images
from the Imagenet validation set, randomly cropped to 256 × 256. We provide full details for this
experiment in Appendix B.2.

We report each denoiser’s performance in Table 1 and include 1-sigma error bars computed across
the test images. Despite being trained exclusively on highly corrupted images, our denoiser nearly
matches the performance of an ordinary supervised denoiser at both noise levels and performs
comparably to Noise2Noise, which requires independent noisy copies of each ground truth image
during training.

Figure 5: Jacobian singular
values of supervised denoiser
(blue) and our denoiser (or-
ange) evaluated at a noisy
held-out image with σ = 2.

We further illustrate the comparison in Figure 4. All neural meth-
ods recover substantially more fine detail than the classical BM3D
denoiser, particularly at the larger noise level σ = 2. Notably, our
method performs nearly as well as a supervised denoiser, despite
being trained exclusively on highly corrupted data.

We also demonstrate the sparsity-inducing effect of our regularizer
on the singular values of Jfθ in Figure 5, where we plot the Jaco-
bian singular values of our denoiser and a supervised denoiser at a
randomly-drawn validation image corrupted with σ = 2 Gaussian
noise. We normalize the singular values so that each Jacobian’s
largest singular value is 1 and depict the singular values on log scale.
As expected, our denoiser’s Jacobian singular values decay more
rapidly than those of the supervised denoiser at the same point.

These results show that our regularizer (6) can be used to construct
a tractable non-linear generalization of Gavish and Donoho [2017]’s
optimal shrinker that performs nearly as well as a supervised denoiser
on image denoising tasks, despite being trained exclusively on highly
corrupted images.

8


---Page Break---
Figure 6: Traversals along Jacobian singular
vectors of our unregularized encoder in latent
space. These traversals edit the colors of the
outputs but not other meaningful attributes.

Figure 7: Traversals along Jacobian singular
vectors of our regularized encoder in latent
space. Our regularizer enables these traver-
sals to edit the subject’s facial expressions.

Representation learning.
We now apply our method to unsupervised representation learning. We
train a deterministic autoencoder consisting of an encoder fθ and a decoder gϕ on the CelebA dataset
[Liu et al., 2015] and approximately penalize the Jacobian nuclear norm ∥Jfθ[x]∥∗of the encoder at
training data x ∈D(Ω) using our regularizer R(x; fθ). This encourages the encoder to locally behave
like a low-rank linear map whose image is low-dimensional. One may interpret this as a deterministic
autoencoder with locally low-dimensional latent spaces. We demonstrate that the left-singular vectors
of the encoder Jacobian Jfθ[x] are semantically meaningful directions of variation about training data
in latent space. We provide full experimental details in Appendix B.3.

To demonstrate our autoencoder’s ability to learn meaningful representations, we select an arbitrary
training point x and traverse the latent space of our regularized autoencoder and an unregularized
baseline along rays of the form z = fθ(x) + αud
θ(x), where ud
θ(x) is the d-th left-singular vector of
the encoder Jacobian Jfθ[x]. These left-singular vectors form a basis for the image of Jfθ[x] and
approximate a basis for the tangent space of the latent manifold at fθ(x). We depict decoded images
along this traversal for our regularized autoencoder in Figure 7 and for the baseline in Figure 6.

Figure 8: A β-VAE’s latent traversals
recover meaningful directions of varia-
tion, but the decoded images are highly
diffuse.

The traversals in Figure 7 are semantically meaningful.
For instance, a traversal along the first singular vector
edits the tint of the decoded image, and a traversal along
the second singular vector edits the facial expression of
the image’s subject. The traversals of the unregularized
autoencoder’s latent space in Figure 6 edit the colors of the
decoded image but are unable to control other attributes.

We also follow Higgins et al. [2017] and visualize traver-
sals along latent variables of a β-VAE at the same training
point x in Figure 8. While the β-VAE is able to discover
meaningful directions of variation, the decoded images
are highly diffuse, as is typical of VAEs. In contrast, our
autoencoder’s reconstructions retain finer details. We conjecture that our model’s improved capac-
ity results from our autoencoder’s ability to learn a locally low-dimensional latent space without
constraining its global structure.

6
Conclusion

The Jacobian nuclear norm ∥Jf[x]∥∗is a natural regularizer for learning problems, where it steers
solutions towards having low-rank Jacobians. Such functions are locally sensitive to changes in
their inputs in only a few directions, which is an especially desirable prior for data that is supported
on a low-dimensional manifold. However, computing ∥Jf[x]∥∗naïvely requires both evaluating
a Jacobian and taking its SVD; the combined cost of these operations is prohibitive for the high-
dimensional maps f that often arise in deep learning.

9


---Page Break---
Our work resolves this computational challenge by generalizing a surprising result (2) from matrix
learning to non-linear learning problems. As they rely on parametrizing the learned function f = g◦h
as a composition of functions g and h, our methods are tailor-made for deep learning, where such
parametrizations are ubiquitous. We anticipate that the deep learning community will discover
additional applications of Jacobian nuclear norm regularization to make use of our efficient methods.

As an efficient implementation of our regularizer relies on estimating the squared Jacobian Frobenius
norm using Hutchinson’s trace estimator, some error is inevitable. This error manifests itself in slightly
diffuse boundaries in the solutions to the Rudin-Osher-Fatemi problem in Figure 1. However, we do
not find this error problematic for high-dimensional applications such as unsupervised denoising as
in Section 5. Future implementations of our method may employ more accurate estimators of the
squared Jacobian Frobenius norm for applications where accuracy is of paramount concern.

Acknowledgments and Disclosure of Funding

The MIT Geometric Data Processing Group acknowledges the generous support of Army Research
Office grants W911NF2010168 and W911NF2110293, of National Science Foundation grant IIS-
2335492, from the CSAIL Future of Data program, from the MIT–IBM Watson AI Laboratory, from
the Wistron Corporation, and from the Toyota–CSAIL Joint Research Center.

Christopher Scarvelis acknowledges the support of the Natural Sciences and Engineering Research
Council of Canada (NSERC), funding reference number CGSD3-557558-2021.

References

G. Alain and Y. Bengio. What regularized auto-encoders learn from the data-generating distribution.
The Journal of Machine Learning Research, 15(1):3563–3593, 2014.

C. M. Bishop. Training with noise is equivalent to tikhonov regularization. Neural Computation, 7
(1):108–116, 1995. doi: 10.1162/neco.1995.7.1.108.

R. Cabral, F. De la Torre, J. P. Costeira, and A. Bernardino. Unifying nuclear norm and bilinear factor-
ization approaches for low-rank matrix decomposition. In Proceedings of the IEEE international
conference on computer vision, pages 2488–2495, 2013.

J.-F. Cai, E. J. Candès, and Z. Shen. A singular value thresholding algorithm for matrix completion.
SIAM Journal on Optimization, 20(4):1956–1982, 2010. doi: 10.1137/080738970. URL https:
//doi.org/10.1137/080738970.

E. Candès and B. Recht.
Exact matrix completion via convex optimization.
Commun. ACM,
55(6):111–119, jun 2012. ISSN 0001-0782. doi: 10.1145/2184319.2184343. URL https:
//doi.org/10.1145/2184319.2184343.

E. J. Candès and T. Tao. The power of convex relaxation: near-optimal matrix completion. IEEE
Trans. Inf. Theor., 56(5):2053–2080, may 2010. ISSN 0018-9448. doi: 10.1109/TIT.2010.2044061.
URL https://doi.org/10.1109/TIT.2010.2044061.

E. J. Candès, X. Li, Y. Ma, and J. Wright. Robust principal component analysis? J. ACM, 58(3), jun
2011. ISSN 0004-5411. doi: 10.1145/1970392.1970395. URL https://doi.org/10.1145/
1970392.1970395.

R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud. Neural ordinary differential equations.
Advances in neural information processing systems, 31, 2018.

K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. Image denoising by sparse 3-d transform-domain
collaborative filtering. IEEE Transactions on Image Processing, 16(8):2080–2095, 2007. doi:
10.1109/TIP.2007.901238.

Y. Dai, H. Li, and M. He. A simple prior-free method for non-rigid structure-from-motion factoriza-
tion. International Journal of Computer Vision, 107:101–122, 2014.

10


---Page Break---
M. Elad, B. Kawar, and G. Vaksman. Image denoising: The deep learning revolution and be-
yond—a survey paper.
SIAM Journal on Imaging Sciences, 16(3):1594–1654, 2023.
doi:
10.1137/23M1545859. URL https://doi.org/10.1137/23M1545859.

C. Finlay, J.-H. Jacobsen, L. Nurbekyan, and A. M. Oberman. How to train your neural ode. arXiv
preprint arXiv:2002.02798, 2, 2020.

T. Fritz. Nuclear norm as minimum of frobenius norm product. MathOverflow. URL https:
//mathoverflow.net/q/298009. URL:https://mathoverflow.net/q/298009 (version: 2018-04-
18).

M. Gavish and D. L. Donoho. The optimal hard threshold for singular values is 4/
√

3. IEEE
Transactions on Information Theory, 60(8):5040–5053, 2014. doi: 10.1109/TIT.2014.2323359.

M. Gavish and D. L. Donoho.
Optimal shrinkage of singular values.
IEEE Transactions on
Information Theory, 63(4):2137–2152, 2017.

I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerch-
ner. beta-VAE: Learning basic visual concepts with a constrained variational framework. In
International Conference on Learning Representations, 2017. URL https://openreview.net/
forum?id=Sy2fzU9gl.

J. Hoffman, D. A. Roberts, and S. Yaida. Robust learning with jacobian regularization, 2020. URL
https://openreview.net/forum?id=ryl-RTEYvB.

M. Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing
splines. Communication in Statistics- Simulation and Computation, 18:1059–1076, 01 1989. doi:
10.1080/03610919008812866.

D. Jakubovitz and R. Giryes.
Improving dnn robustness to adversarial attacks using jacobian
regularization. In Proceedings of the European conference on computer vision (ECCV), pages
514–529, 2018.

Z. Kadkhodaie, F. Guth, E. P. Simoncelli, and S. Mallat. Generalization in diffusion models arises
from geometry-adaptive harmonic representations. In The Twelfth International Conference on
Learning Representations, 2024. URL https://openreview.net/forum?id=ANvmVS2Yr0.

J. Kelly, J. Bettencourt, M. J. Johnson, and D. K. Duvenaud. Learning differential equations that are
easy to solve. Advances in Neural Information Processing Systems, 33:4370–4380, 2020.

R. H. Keshavan, S. Oh, and A. Montanari. Matrix completion from a few entries. In 2009 IEEE
International Symposium on Information Theory, pages 324–328, 2009. doi: 10.1109/ISIT.2009.
5205567.

J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2noise:
Learning image restoration without clean data. In International Conference on Machine Learning
(ICML), volume 80, pages 2971–2980, March 2018.

Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In Proceedings of
International Conference on Computer Vision (ICCV), December 2015.

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on
Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and
its application to evaluating segmentation algorithms and measuring ecological statistics. In
Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, volume 2,
pages 416–423 vol.2, 2001. doi: 10.1109/ICCV.2001.937655.

R. Mazumder, T. Hastie, and R. Tibshirani. Spectral regularization algorithms for learning large
incomplete matrices. Journal of Machine Learning Research, 11(80):2287–2322, 2010. URL
http://jmlr.org/papers/v11/mazumder10a.html.

11


---Page Break---
Y. Meyer. Oscillating Patterns in Image Processing and Nonlinear Evolution Equations: The Fifteenth
Dean Jacqueline B. Lewis Memorial Lectures. American Mathematical Society, USA, 2001. ISBN
0821829203.

R. Nadakuditi. Optshrink: An algorithm for improved low-rank signal matrix denoising by optimal,
data-driven singular value shrinkage. Information Theory, IEEE Transactions on, 60:3002–3018,
05 2014. doi: 10.1109/TIT.2014.2311661.

B. Recht. A simpler approach to matrix completion. J. Mach. Learn. Res., 12(null):3413–3430, dec
2011. ISSN 1532-4435.

J. D. M. Rennie and N. Srebro. Fast maximum margin matrix factorization for collaborative prediction.
In Proceedings of the 22nd International Conference on Machine Learning, ICML ’05, page
713–719, New York, NY, USA, 2005. Association for Computing Machinery. ISBN 1595931805.
doi: 10.1145/1102351.1102441. URL https://doi.org/10.1145/1102351.1102441.

S. Rifai, P. Vincent, X. Muller, X. Glorot, and Y. Bengio. Contractive auto-encoders: Explicit
invariance during feature extraction. In Proceedings of the 28th international conference on
international conference on machine learning, pages 833–840, 2011.

R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis
with latent diffusion models, 2021.

O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image
segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015:
18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18,
pages 234–241. Springer, 2015.

L. I. Rudin, S. Osher, and E. Fatemi. Nonlinear total variation based noise removal algorithms.
Physica D: nonlinear phenomena, 60(1-4):259–268, 1992.

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy,
A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recog-
nition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015. doi:
10.1007/s11263-015-0816-y.

A. Shabalin and A. Nobel. Reconstruction of a low-rank matrix in the presence of gaussian noise.
Journal of Multivariate Analysis, 118, 07 2010. doi: 10.1016/j.jmva.2013.03.005.

J. Sokoli´c, R. Giryes, G. Sapiro, and M. R. D. Rodrigues. Robust large margin deep neural networks.
Trans. Sig. Proc., 65(16):4265–4280, aug 2017. ISSN 1053-587X. doi: 10.1109/TSP.2017.2708039.
URL https://doi.org/10.1109/TSP.2017.2708039.

M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoor-
thi, J. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low
dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020.

D. Varga, A. Csiszárik, and Z. Zombori. Gradient regularization improves accuracy of discriminative
models. arXiv preprint arXiv:1712.09936, 2017.

P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features
with denoising autoencoders. In Proceedings of the 25th international conference on Machine
learning, pages 1096–1103, 2008.

P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, and L. Bottou. Stacked denoising
autoencoders: Learning useful representations in a deep network with a local denoising criterion.
Journal of machine learning research, 11(12), 2010.

A. Webb. Functional approximation by feed-forward networks: a least-squares approach to general-
ization. IEEE Transactions on Neural Networks, 5(3):363–371, 1994. doi: 10.1109/72.286908.

K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte. Plug-and-play image restoration
with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44
(10):6360–6376, 2021.

12


---Page Break---
A
Proofs

A.1
Proof of Equation (1)

We draw heavy inspiration from a proof by Fritz that appeared on MathOverflow. We will prove that

∥A∥∗=
min
U,V :UV ⊤=A
1
2
 
∥U∥2
F + ∥V ∥2
F

.

First suppose that U, V are matrices such that UV ⊤= A. By the matrix Hölder inequality,

∥A∥∗= ∥UV ⊤∥∗≤∥U∥F · ∥V ∥F .

By the arithmetic mean-geometric mean (AM-GM) inequality,

∥U∥F · ∥V ∥F =
q

∥U∥2
F · ∥V ∥2
F ≤1

2
 
∥U∥2
F + ∥V ∥2
F

.

Combining these results, we obtain

∥A∥∗= ∥UV ⊤∥∗≤
inf
U,V :UV ⊤=A
1
2
 
∥U∥2
F + ∥V ∥2
F

.

To see that the inf is attained at ∥A∥∗, take the compact SVD A = UAΣV ⊤
A and set U :=
UA
√

Σ, V := VA
√

Σ. Then clearly UV ⊤= A, and

1
2
 
∥U∥2
F + ∥V ∥2
F

= 1

2


∥UA
√

Σ∥2
F + ∥VA
√

Σ∥2
F

= 1

2


∥
√

Σ∥2
F + ∥
√

Σ∥2
F


= 1

2

 r
X

i=1
σi +

r
X

i=1
σi

!

= ∥A∥∗

This proves Equation (1). An alternative proof using somewhat more complex methods is given in
Mazumder et al. [2010]. ■

A.2
Proof of Theorem 3.1

Let D(Ω) be a data distribution with measure µ supported on a compact set Ω⊆Rn that is absolutely
continuously with respect to the Lebesgue measure λ(Ω), and let ℓ∈C1(Rm) be a continuously
differentiable loss function. Let C∞(Ω) denote the set of infinitely differentiable functions on Ω. We
will show that:

inf
f∈C∞(Ω)
E
x∼D(Ω) [ℓ(f(x), x) + η∥Jf[x]∥∗]

=
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
E
x∼D(Ω)

h
ℓ(g(h(x)), x) + η

2
 
∥Jg[h(x)]∥2
F + ∥Jh[x]∥2
F
i
.
(12)

We denote the left-hand side objective by EL(f) and its inf by (L); we denote the right-hand side
objective by ER(g, h) and its inf by (R). We prove that (L) ≤(R) and (R) ≤(L).

A.2.1
(L) ≤(R)

This is the easy direction. The basic observation is that if we parametrize f = g ◦h, then Jf[x] =
Jg[h(x)]Jh[x], and Srebro’s identity (1) tells us that

∥Jf[x]∥∗=
min
U,V :UV ⊤=Jf[x]
1
2
 
∥U∥2
F + ∥V ∥2
F

≤1

2
 
∥Jg[h(x)]∥2
F + ∥Jh[x]∥2
F


13


---Page Break---
The rest of the proof is book-keeping.

We first verify that C∞(Ω) is closed under composition by showing that:

C∞(Ω) = {g ◦h : h ∈C∞(Ω), g ∈C∞(h(Ω))} .
(13)

The ⊆inclusion is straightforward: given any f ∈C∞(Ω), just choose h ≡f and g ≡Id; these
functions are clearly in the correct classes. The ⊇inclusion follows from the fact that the composition
of C∞functions is C∞by the chain rule.

This yields:

inf
f∈C∞(Ω) EL(f) =
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
EL(g ◦h).

We now use Srebro’s identity as follows:

inf
h∈C∞(Ω)
g∈C∞(h(Ω))
EL(g ◦h) =
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
E
x∼D(Ω) [ℓ(g(h((x)), x) + η∥J(g ◦h)[x]∥∗]

=
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
E
x∼D(Ω) [ℓ(g(h((x)), x) + η∥Jg[h(x)]Jh[x]∥∗]

≤
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
E
x∼D(Ω)

h
ℓ(g(h((x)), x) + η

2
 
∥Jg[h(x)]∥2
F + ∥Jh[x]∥2
F
i

=
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
ER(g ◦h).

Consequently,

inf
f∈C∞(Ω) EL(f) ≤
inf
h∈C∞(Ω)
g∈C∞(h(Ω))
ER(g ◦h)

and hence (L) ≤(R).

A.2.2
(R) ≤(L)

This is the hard direction. The proof strategy is as follows:

1. We begin with a function fm ∈C∞(Ω) such that EL(fm) is arbitrarily close to its inf over
C∞(Ω). We use fm to construct parametric families of functions gz
m, hz
m whose composition
is a good local approximation to fm in a neighborhood of z ∈Ω, both pointwise and in
terms of the energy contributions due to x ∈Ωnear z. This step relies crucially on our
ability to construct optimal solutions to the RHS problem in (1).

2. We then stitch together these local approximations to form a sequence of global approxima-
tions gk
m, hk
m to fm. These functions are piecewise affine and hence not regular enough to
lie in C∞(Ω) as required by the right-hand side of Equation (12).

3. Finally, mollifying the piecewise affine functions gk
m, hk
m yields a minimizing sequence of
C∞(Ω) functions gk
m,ϵ, hk
m,ϵ such that ER(gk
m,ϵ, hk
m,ϵ) approaches the inf of EL.

Local approximations to fm.
To begin, let fm ∈C∞(Ω) be a function that attains EL(fm) ≤
inff∈C∞(Ω) EL(f) + 1

m. Fix some z ∈Ω, take the thin SVD of Jfm[z] = Um(z)Σm(z)Vm(z)⊤,
and use it to define two parametric families of affine functions:

hz
m(x) =
p

Σm(z)Vm(z)⊤x,

14


---Page Break---
and

gz
m(y) = Um(z)
p

Σm(z)y + fm(z) −Jfm[z]z.

These functions satisfy two key properties:

gz
m (hz
m(x)) = fm(z) + Jfm[z](x −z) = fm(x) + Rz
m(x),
(14)

where ∥Rz
m(x)∥2 ∈O(∥x −z∥2
2) by Taylor’s theorem, and

η
2
 
∥Jgz
m[hz
m(x)]∥2
F + ∥Jhz
m[x]∥2
F

= η

2


∥
p

Σm(z)∥2
F + ∥
p

Σm(z)∥2
F

= η∥Jfm[z]∥∗. (15)

Using (14), we obtain

ℓ(gz
m(hz
m(x)), x) = ℓ(fm(x) + Rz
m(x), x).
(16)

The continuity of ℓensures that we can make ℓ(fm(x) + Rz
m(x), x) arbitrarily close to ℓ(fm(x), x)
by making ∥x −z∥2 sufficiently small.

Furthermore,

∥Jfm[z]∥∗= ∥Jfm[x] + Jfm[z] −Jfm[x]∥∗≤∥Jfm[x]∥∗+ ∥Jfm[z] −Jfm[x]∥∗,
(17)

and as fm ∈C∞, Jfm is a continuous function, so we can make ∥Jfm[z] −Jfm[x]∥∗arbitrarily
small by making ∥x −z∥sufficiently small.

The compositions of these functions gz
m, hz
m are good local approximations to fm in a neighborhood
of z, both pointwise (by (14)) and in terms of the energy contributions arising from x ∈Ωnear z (by
(16) and (17)).

Global piecewise affine approximations to fm.
We now stitch together these local approximations
to form a sequence of global approximations gk
m, hk
m to fm.

For each k, fix a set of points Zk = {zi}N(k)
i=1 and use this set to partition Ωinto Voronoi regions Vi.
Choose the centroids zi such that maxx∈Vi ∥x −zi∥2 ≤ϵk, for ϵk > 0 sufficiently small to ensure
that |∥Jfm[zi]∥∗−∥Jfm[x]∥∗| + |ℓ(fm(x) + Rzi
m(x), x) −ℓ(fm(x), x)| < 1

k for all x ∈Vi and
for all regions Vi. (The compactness of Ωand the uniform continuity of ∥Jfm∥∗on Ωand ℓon its
domain ensures that we can always find a finite set of centroids with this property.)

Then let gk
m, hk
m be piecewise affine functions such that gk
m(x) := gzi
m(x) and hk
m(x) := hzi
m(x) for
all x ∈int(Vi). For all points x on a Voronoi boundary, define gk
m, hk
m by averaging over the interiors
of the Voronoi cells incident on the boundary. This yields:

15


---Page Break---
E
x∼D(Ω)

h
ℓ(gk
m(hk
m(x)), x) + η

2
 
∥Jgk
m[hk
m(x)]∥2
F + ∥Jhk
m[x]∥2
F
i

=
E
x∼D(Ω)





N(k)
X

i=1
[ℓ(fm(x) + Rzi
m(x)), x) + η∥Jfm[zi]∥∗] · 1 [x ∈Vi]





≤
E
x∼D(Ω)





N(k)
X

i=1


ℓ(fm(x), x) + η∥Jfm[x]∥∗+ 1

k


· 1 [x ∈Vi]





=
E
x∼D(Ω)


ℓ(fm(x), x) + η∥Jfm[x]∥∗+ 1

k



= EL(fm) + 1

k

≤
inf
f∈C∞(Ω) EL(f) + 1

m + 1

k

and hence ER(gk
m, hk
m) ≤inff∈C∞(Ω) EL(f) + 1

m + 1

k.

Mollifying the piecewise affine approximations.
The functions gk
m, hk
m constructed in the previous
section are piecewise affine and hence not regular enough to lie in C∞(Ω) as required by (R). We
now mollify these functions to yield a minimizing sequence of C∞(Ω) functions gk
m,ϵ, hk
m,ϵ such
that ER(gk
m,ϵ, hk
m,ϵ) approaches inff∈C∞(Ω) EL(f).

We mollify gk
m, hk
m by convolving them against the standard mollifiers (infinitely differentiable and
compactly supported on B(0, ϵ)) to yield a sequence of C∞(Ω) functions gk
m,ϵ, hk
m,ϵ. We need to
show that

ER(gk
m,ϵ, hk
m,ϵ) ≤ER(gk
m, hk
m) + ψ(ϵ; m, k)

for some error ψ(ϵ; m, k) that vanishes as ϵ →0 for any m, k. We proceed by individually controlling
the terms in ER(gk
m,ϵ, hk
m,ϵ).

Controlling the error in
E
x∼D(Ω)


ℓ(gk
m,ϵ(hk
m,ϵ(x)), x)

.
We first show that it suffices to prove that

∥gk
m,ϵ ◦hk
m,ϵ −gk
m ◦hk
m∥L1(Ω,µ) →0 for any m, k. (Recall that µ is the measure associated with the
data distribution D(Ω).)

Note that as ℓ∈C1(Rm) and the image of the compact set Ωunder the piecewise affine function
gk
m ◦hk
m is bounded, ℓis L-Lipschitz on gk
m(hk
m(Ω)). For any sequence of functions fn converging
to f in L1, we then have:



Z

Ω
ℓ(fn(x), x)dµ −
Z

Ω
ℓ(f(x), x)dµ
 ≤
Z

Ω
|ℓ(fn(x), x) −ℓ(f(x), x)| dµ

≤
Z

Ω
L∥fn(x) −f(x)∥2dµ

= L
Z

Ω
∥fn(x) −f(x)∥2dµ

= L∥fn −f∥L1(Ω,µ)
→
ϵ→0 0

It therefore suffices to show that ∥gk
m,ϵ ◦hk
m,ϵ −gk
m ◦hk
m∥L1(Ω,µ) →0 for any m, k. To this end, we
will use the bound

16


---Page Break---
∥gk
m,ϵ◦hk
m,ϵ−gk
m◦hk
m∥L1(Ω,µ) ≤∥gk
m,ϵ◦hk
m,ϵ−gk
m,ϵ◦hk
m∥L1(Ω,µ)+∥gk
m,ϵ◦hk
m−gk
m◦hk
m∥L1(Ω,µ)

and show that each of the RHS terms goes to 0 as ϵ →0.

We begin by controlling ∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥L1(Ω,µ). The obvious approach is to use the fact
that gk
m,ϵ is Lipschitz on the bounded domain hk
m(Ω) and that ∥hk
m,ϵ −hk
m∥L1(Ω,µ) →0 by standard
properties of mollifiers. However, this naïve approach fails because the Lipschitz constant of gk
m,ϵ
increases as ϵ →0: We need to bound it in terms of a quantity independent of ϵ. We will instead
use the fact that the un-mollified function gk
m is piecewise affine and hence locally Lipschitz on the
interior of each affine region.

Before proceeding with the remainder of the proof, we make a key observation. Given a Voronoi
partition of the compact domain Ωinto N(k) cells Vi, we constructed gk
m, hk
m as piecewise affine
functions such that gk
m(x) := gzi
m(x) and hk
m(x) := hzi
m(x) for all x ∈int(Vi) – and for all points
x on a Voronoi boundary, we defined gk
m, hk
m by averaging over the interiors of the Voronoi cells
incident on the boundary. The affine functions gzi
m, hzi
m were constructed to have the following
properties:

gz
m (hz
m(x)) = fm(z) + Jfm[z](x −z) = fm(x) + Rz
m(x),
(18)

where ∥Rz
m(x)∥2 ∈O(∥x −z∥2
2) by Taylor’s theorem, and

η
2
 
∥Jgz
m[hz
m(x)]∥2
F + ∥Jhz
m[x]∥2
F

= η

2


∥
p

Σm(z)∥2
F + ∥
p

Σm(z)∥2
F

= η∥Jfm[z]∥∗. (19)

At any x on the interior of a Voronoi cell, gk
m and hk
m are equal to some affine functions gzi
m, hzi
m, so by
(19), η

2
 
∥Jgk
m[hk
m(x)]∥2
F + ∥Jhk
m[x]∥2
F

= η∥Jfm[zi]∥∗< +∞. In particular, ∥Jgk
m[hk
m(x)]∥2
F
and ∥Jhk
m[x]∥2
F must both be well-defined and finite at this x. This can only happen if hk
m(x) lies on
the interior of a Voronoi cell, since otherwise Jgk
m[hk
m(x)] would be undefined. Hence hk
m maps the
interior of Voronoi cells to the interior of Voronoi cells; the contrapositive is that if hk
m(x) lies on a
Voronoi boundary, then x also lies on a Voronoi boundary. It follows that the preimage of the Voronoi
boundaries under hk
m is a set of Lebesgue measure zero. Under the absolute continuity hypothesis,
this is also a set of µ-measure zero. We use this fact extensively throughout the remainder of the
proof.

For any m, k, let Sk
m ⊆Ωbe the set of Voronoi boundaries from the partition we constructed
earlier; this Voronoi partition depends on m, k but not on ϵ. We then begin by stripping the Voronoi
boundaries from Ωto obtain Ω(m, k) := Ω\ Sk
m. This only removes a set of µ-measure 0 from Ω
and hence doesn’t impact the remaining convergence results. Given ϵ, we then partition Ω(m, k) into
two subsets:

• Let Ω0(m, k, ϵ) ⊆Ω(m, k) be the set of x ∈Ω(m, k) such that d(x, Sk
m) > ϵ. (We use
d(p, S) to denote the distance from the point p from the set S.)

• Let Ω1(m, k, ϵ) be its complement in Ω(m, k): the set of x ∈Ω(m, k) such that d(x, Sk
m) ≤
ϵ.

Ω0(m, k, ϵ) will be our good set, and Ω1(m, k, ϵ) will be a bad set whose measure converges to 0 as
ϵ →0. We decompose ∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥L1(Ω,µ) as follows:

∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥L1(Ω,µ) =
Z

Ω0(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ

+
Z

Ω1(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ

17


---Page Break---
We begin by showing that ∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2 = 0 for all x ∈Ω0(m, k, ϵ). To this
end, first recall that the standard mollifier supported on B(0, ϵ) is defined as follows:

ηϵ(y) =

(
C(ϵ) exp

1
∥y

ϵ ∥2
2−1

for ∥y∥2 ≤1

0
for ∥y∥2 > 1
(20)

where C(ϵ) > 0 is chosen to ensure that ηϵ integrates to 1. Now, if x ∈Ω0(m, k, ϵ), then d(x, Sk
m) >
ϵ and hence B(x, ϵ) is entirely contained in the Voronoi cell Vi containing x. (Note that x cannot
lie on a Voronoi boundary, as we have stripped the µ-measure zero set of Voronoi boundaries Sk
m
from Ωbefore constructing Ω0(m, k, ϵ) and Ω1(m, k, ϵ).) On this ball B(x, ϵ), the linearity of
hk
m(y) :=
p

Σm(zi)Vm(zi)⊤y and the rotational symmetry of the mollifier ηϵ yields:

hk
m,ϵ(x) =
Z

B(0,ϵ)
hk
m(y)ηϵ(x −y)dy
(21)

=
Z

B(0,ϵ)

p

Σm(zi)Vm(zi)⊤yηϵ(x −y)dy
(22)

=
p

Σm(zi)Vm(zi)⊤
Z

B(0,ϵ)
yηϵ(x −y)dy

|
{z
}
=x

(23)

=
p

Σm(zi)Vm(zi)⊤y
(24)

= hk
m(x).
(25)

Hence for all x ∈Ω0(m, k, ϵ), hk
m,ϵ(x) = hk
m(x), and it follows that gk
m,ϵ(hk
m,ϵ(x)) = gk
m,ϵ(hk
m(x))
and therefore ∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2 = 0. We conclude that

Z

Ω0(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ = 0.

To bound the second term, note that the following inequalities hold µ-almost everywhere on Ω:

∥gk
m,ϵ(hk
m,ϵ(x))−gk
m,ϵ(hk
m(x))∥2 ≤2
sup
y∈hkm(Ω(m,k))
∥gk
m,ϵ(y)∥2 ≤2
sup
y∈hkm(Ω(m,k))
∥gk
m(y)∥2 =: M(m, k).

(26)

The second inequality can be derived using Jensen’s inequality by first showing the following for all
y ∈hk
m(Ω(m, k)):

∥gk
m,ϵ(y)∥2 = ∥
Z

B(y,ϵ)
gk
m(z) ηϵ(z)dz
| {z }
:=dηϵ(z)

∥2

≤
|{z}
Jensen

Z

B(y,ϵ)
∥gk
m(z)∥2dηϵ(z)

≤
sup
y∈hkm(Ω(m,k))
∥gk
m(y)∥2

Z

B(y,ϵ)
dηϵ(z)

|
{z
}
=1
=
sup
y∈hkm(Ω(m,k))
∥gk
m(y)∥2,

which implies that

18


---Page Break---
sup
y∈hkm(Ω(m,k))
∥gk
m,ϵ(y)∥2 ≤
sup
y∈hkm(Ω(m,k))
∥gk
m(y)∥2.

Returning to (26), we can bound the integral over the bad set
R

Ω1(m,k,ϵ) ∥gk
m,ϵ(hk
m,ϵ(x)) −
gk
m,ϵ(hk
m(x))∥2dµ as follows:

Z

Ω1(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ ≤M(m, k) · µ(Ω1(m, k, ϵ))

where µ(Ω1(m, k, ϵ)) is the measure of Ω1(m, k, ϵ). We will now show that µ(Ω1(m, k, ϵ)) →0,
which will imply

Z

Ω1(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ →
ϵ→0 0,

and therefore allow us to conclude that

∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥2
L1(Ω,µ) →
ϵ→0 0.

Proving that µ(Ω1(m, k, ϵ)) →0.
Recall that:

Ω1(m, k, ϵ) :=

x ∈Ω(m, k) : d(x, Sk
m) ≤ϵ
	
,
(27)

where d(x, Sk
m) denotes the distance from the point x to the union of Voronoi boundaries Sk
m. Note
that this set is a union of cylinders of radius ϵ centered at the Voronoi boundaries Sk
m. As Sk
m has
Lebesgue measure 0, the Lebesgue measure of the cylinders B(m, k, ϵ) also goes to 0 as the radius
ϵ →0. The absolute continuity of µ then implies that µ(Ω1(m, k, ϵ)) →0 as well.

Using these results, we obtain

Z

Ω1(m,k,ϵ)
∥gk
m,ϵ(hk
m,ϵ(x)) −gk
m,ϵ(hk
m(x))∥2dµ →
ϵ→0 0,

and therefore conclude that

∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥2
L1(Ω,µ) →
ϵ→0 0.

This completes the proof that ∥gk
m,ϵ ◦hk
m,ϵ −gk
m,ϵ ◦hk
m∥L1(Ω,µ) →0 as ϵ →0.

Controlling the second term ∥gk
m,ϵ ◦hk
m −gk
m ◦hk
m∥L1(Ω,µ) is easier. Indeed, gk
m,ϵ ◦hk
m →
gk
m ◦hk
m pointwise a.e. as ϵ →0 by standard properties of mollifiers. Furthermore, the sequence
gk
m,ϵ ◦hk
m is dominated almost everywhere by the function identically equal (componentwise) to
maxx∈Ω(m,k) gk
m(hk
m(x)); this function is in L1(Ω, µ) because Ωis a compact domain. It follows
from the dominated convergence theorem that ∥gk
m,ϵ ◦hk
m −gk
m ◦hk
m∥L1(Ω,µ) →0 as ϵ →0.

We have shown that each of the following RHS terms goes to 0 as ϵ →0:

∥gk
m,ϵ◦hk
m,ϵ−gk
m◦hk
m∥L1(Ω,µ) ≤∥gk
m,ϵ◦hk
m,ϵ−gk
m,ϵ◦hk
m∥L1(Ω,µ)+∥gk
m,ϵ◦hk
m−gk
m◦hk
m∥L1(Ω,µ),

which allows us to conclude that ∥gk
m,ϵ ◦hk
m,ϵ −gk
m ◦hk
m∥L1(Ω,µ) →0 and consequently


E
x∼D(Ω)


ℓ(gk
m,ϵ(hk
m,ϵ(x)), x)

−
E
x∼D(Ω)


ℓ(gk
m(hk
m(x)), x)
 →
ϵ→0 0

19


---Page Break---
as ϵ →0, for any m, k. It remains to prove similar convergence results for the remaining terms in
ER.

Controlling the error in
η
2
E
x∼D(Ω)


∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F + ∥Jhk
m,ϵ[x]∥2
F

.
We now show that
R

Ω∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F dµ →
R

Ω∥Jgk
m[hk
m(x)]∥2
F dµ via the dominated convergence theorem (DCT).
First note that Ω(m, k) and Ωonly differ by a set of µ-measure zero (the Voronoi boundaries Sk
m), so
we can equivalently prove

Z

Ω(m,k)
∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F dµ →
ϵ→0

Z

Ω(m,k)
∥Jgk
m[hk
m(x)]∥2
F dµ.

This allows us to avoid points x such that x or hk
m(x) lie on Voronoi boundaries; these are problematic
because if hk
m(x) lies on a Voronoi boundary, then Jgk
m[hk
m(x)] is undefined.

First note that as gk
m, hk
m are piecewise affine and the mollifiers are compactly supported, for any
given x ∈Ω(m, k), one can choose ϵ > 0 sufficiently small so that gk
m,ϵ[hk
m,ϵ(x)] = gk
m[hk
m(x)] and
hence ∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F = ∥Jgk
m[hk
m(x)]∥2
F .

In particular, for any x ∈Ω(m, k), let ϵ1 < d(x, Sk
m) and ϵ2 < d(hk
m(x), Sk
m); these can both be
> 0 because d(x, Sk
m) > 0, d(hk
m(x), Sk
m) > 0 for x ∈Ω(m, k). Then define ϵ := min{ϵ1, ϵ2}.
(21) shows why choosing ϵ1 < d(x, Sk
m) implies hk
m,ϵ(x) = hk
m(x); similar arguments hold show
that ϵ2 < d(hk
m(x), Sk
m) implies gk
m,ϵ(hk
m(x)) = gk
m(hk
m(x)). We then have gk
m,ϵ[hk
m,ϵ(x)] =
gk
m,ϵ[hk
m(x)] = gk
m[hk
m,ϵ(x)] as desired.

Hence for this choice of ϵ, Jgk
m,ϵ[hk
m,ϵ(x)] = Jgk
m[hk
m(x)] and consequently ∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F =
∥Jgk
m[hk
m(x)]∥2
F . (Recall that Jgk
m[hk
m(x)] is well-defined for x ∈Ω(m, k).) It follows that
∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F →∥Jgk
m[hk
m(x)]∥2
F pointwise on Ω(m, k) and pointwise µ-ae on Ω.

Furthermore, another argument via Jensen’s inequality shows that ∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F is dominated
by the function Ak
m that is identically equal to supx∈Ω(m,k) ∥Jgk
m[hk
m(x)]∥2
F ; this function is inte-
grable because Ωis compact.

The DCT then lets us conclude that ∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F →∥Jgk
m[hk
m(x)]∥2
F in L1 and hence that

Z

Ω
∥Jgk
m,ϵ[hk
m,ϵ(x)]∥2
F dµ →
ϵ→0

Z

Ω
∥Jgk
m[hk
m(x)]∥2
F dµ.

The same argument also allows us to conclude that

Z

Ω
∥Jhk
m,ϵ(x)]∥2
F dµ →
ϵ→0

Z

Ω
∥Jhk
m(x)]∥2
F dµ.

We have by now shown that

ER(gk
m,ϵ, hk
m,ϵ) ≤ER(gk
m, hk
m) + ψ(ϵ; m, k)

for some ψ(ϵ; m, k) →0 as ϵ →0. Combining this with our earlier results, we get

ER(gk
m,ϵ, hk
m,ϵ) ≤ER(gk
m, hk
m) + ψ(ϵ; m, k) ≤
inf
f∈C∞(Ω) EL(f) + 1

m + 1

k + ψ(ϵ; m, k).

As we can make 1

m + 1

k +ψ(ϵ; m, k) arbitrarily small by first choosing m, k sufficiently large and then
choosing ϵ(m, k) > 0 to make ψ(ϵ; m, k) sufficiently small, we can finally conclude that (R) ≤(L)
as desired.

This completes the proof of the theorem. ■

20


---Page Break---
A.3
Proof of Theorem 3.2

We will show that

σ2∥Jf[x]∥2
F =
E
ϵ∼N(0,σ2I)


∥f(x + ϵ) −f(x)∥2
2

+ O(σ2).

Since f : Rn →Rm is continuously differentiable, Taylor’s theorem states that:

f(x + ϵ) = f(x) + Jf[x]ϵ + R(x + ϵ),

where ∥R(x + ϵ)∥2 ∈O(∥ϵ∥2
2). Rearranging, taking squared Euclidean norms, and expanding the
square, we obtain:

∥Jf[x]ϵ∥2
2 = ∥f(x + ϵ) −f(x) −R(x + ϵ)∥2
2
= ∥f(x + ϵ) −f(x)∥2
2 −2 · ⟨f(x + ϵ) −f(x), R(x + ϵ)⟩+ ∥R(x + ϵ)∥2
2
|
{z
}
∈O(∥ϵ∥4
2)

≤∥f(x + ϵ) −f(x)∥2
2 + 2 ∥R(x + ϵ)∥2
|
{z
}
∈O(∥ϵ∥2
2)

· ∥f(x + ϵ) −f(x)∥2
|
{z
}
∈O(∥ϵ∥2)

+O(∥ϵ∥4
2)

= ∥f(x + ϵ) −f(x)∥2
2 + O(∥ϵ∥3
2)

= ∥f(x + ϵ) −f(x)∥2
2 + O(∥ϵ∥2
2).

Hutchinson’s trace estimator implies that for any matrix A, ∥A∥2
F = Eϵ∼N(0,I)[∥Aϵ∥2
2]. In particular,

σ2∥Jf[x]∥2
F =
E
ϵ∼N(0,σ2I)


∥Jf[x]ϵ∥2
2


=
E
ϵ∼N(0,σ2I)


∥f(x + ϵ) −f(x)∥2
2 + O(∥ϵ∥3
2)


=
E
ϵ∼N(0,σ2I)


∥f(x + ϵ) −f(x)∥2
2

+ O(E∥ϵ∥2
2
| {z }
=σ2n

)

=
E
ϵ∼N(0,σ2I)


∥f(x + ϵ) −f(x)∥2
2

+ O(σ2),

which completes the proof of the result. ■

A.4
Optimal shrinkage via nuclear norm regularization

Let X ∈RD×N be a low-rank matrix of clean data and Y = X + σϵZ be a matrix of data corrupted
by iid white noise Z. In this appendix, we show that the solution to

min
A∈RD×D
1
2N ∥AY −Y ∥2
F + η∥A∥∗,
(28)

coincides with Gavish and Donoho [2017]’s optimal shrinker for the squared Frobenius norm loss
when η is set to the noise variance σ2
ϵ and as the “aspect ratio” β := d

n →0.

A∗is optimal for problem (28) iff 0 ∈∂ϕ(A∗). Using well-known results from convex optimization,
this condition is equivalent to:

1
Nη (Y −A∗Y )Y ⊤∈∂∥· ∥∗(A∗).
(29)

Furthermore, the subgradient of the nuclear norm is:

21


---Page Break---
∂∥· ∥∗(A) =

UAV ⊤
A + W : U ⊤
A W = 0, WVA = 0, σmax(W) ≤1
	
,
(30)

where A = UAΣAV ⊤
A is an SVD of A. We will show that the solution to (28) is

A∗= UΓU ⊤,
(31)

where Y = UΣV ⊤is an SVD of the noisy data matrix, and

Γd =

(
1 −Nη

σ2
d ,
σd ≥√Nη

0,
σd ≤√Nη.
(32)

The idea is to use the SVD Y = UΣV ⊤and the ansatz A∗= UΓU ⊤to rewrite the LHS of the
inclusion (29) as follows:

1
Nη (Y −A∗Y )Y ⊤= U
 1

Nη (I −Γ)Σ2

U ⊤.

We then express the middle diagonal term as follows:

1
Nη (I −Γ)Σ2 = IT +
1
Nη (I −Γ)Σ2 −IT

where IT is the identity matrix with all columns ≥T set to zero (i.e. an orthogonal projection matrix
onto the first T coordinates). This then yields

U
 1

Nη (I −Γ)Σ2

U ⊤= UT U ⊤
T + U
 1

Nη (I −Γ)Σ2 −IT


U ⊤,

where UT = UIT is U with all columns ≥T set to 0, and T is the first index such that σT ≤√Nη.
As the optimal Γ in (32) sets all entries corresponding to singular values σd ≤√Nη to zero, we
can in fact rewrite A∗= (UIT )Γ(UIT )⊤. It follows that UT U ⊤
T = UIT (UIT )⊤is also of the form
UAV ⊤
A for a valid SVD of A∗(the UIT can serve as both left- and right-singular vectors).

We have therefore expressed the LHS of the inclusion (29) as:

1
Nη (Y −A∗Y )Y ⊤= UAV ⊤
A + W

for UAV ⊤
A = UT U ⊤
T and W = U

1
Nη(I −Γ)Σ2 −IT

U ⊤. This W satisfies all of the conditions
in (30).

Furthermore, if we apply this optimal A∗to the data matrix Y , we obtain A∗Y = UΓΣV ⊤, where

(ΓΣ)d =

(
σ2
d−Nη

σd
,
σ2
d ≥Nη
0,
σ2
d ≤Nη
(33)

If we set η to be equal to the noise variance σ2
ϵ , then this agrees exactly with the optimal shrinker for
the case β := D

N →0 from Gavish and Donoho [2017] under the same noise model Y = X + σϵZ.

B
Experimental details

B.1
Validation experiments: Rudin-Osher-Fatemi (ROF) problem

Architecture.
In all ROF experiments, we parametrize fθ = gθ ◦hθ, where gθ and hθ are both
two-layer MLPs with 100 hidden units. We apply a Fourier feature mapping [Tancik et al., 2020] to

22


---Page Break---
the input coordinates x before passing them through hθ. We use ELU activations in both neural nets
and find the use of differentiable non-linearities to be crucial for obtaining accurate solutions.

Training details.
We train all neural models using the AdamW optimizer [Loshchilov and Hutter,
2019] at a learning rate of 10−4 for 100,000 iterations with a batch size of 10,000. In the n = 2 case,
we integrate over the box [−10, 10]2, and in the n = 5 case, we integrate over the box [−2, 2]5.

We employ a warmup strategy for solving our problem (9). We first train our neural nets at η = 0.05
in the n = 2 case and η = 0.01 in the n = 5 case for 10,000 iterations, and then increase η by 0.05
and 0.01, respectively, each 10,000 iterations until we reach the desired value of η. We then continue
training until we reach 100,000 total iterations.

Each training run for (8) takes approximately 2 hours, and each training run for (9) takes approxi-
mately 45 minutes on a single V100 GPU.

B.2
Denoising

Architecture.
Our architecture for all denoising models is based on the UNet implemented in the
Github repository for Zhang et al. [2021]. Each model is of the form f = g ◦h, where h consists
of the head, downsampling blocks, and body block of the Unet, and g consists of a repeated body
block, the upsampling blocks, and the tail. We replace all ReLU activations with ELU but leave the
remainder of the architecture unchanged.

Training details.
All neural models are trained on 288,049 images from the ImageNet Large Scale
Visual Recognition Challenge 2012 training set [Russakovsky et al., 2015] which we randomly crop
and rescale to 128 × 128. The code for loading and pre-processing this training data is borrowed
from Rombach et al. [2021].

We train all neural models using the AdamW optimizer [Loshchilov and Hutter, 2019] for 2 epochs
at a learning rate of 10−4 then for a final epoch with learning rate 10−5. Each denoising model takes
approximately 5 hours to train on a single V100 GPU.

The training objective for our denoiser is (11) with η = σ. The training objective for the supervised
denoiser is the usual MSE loss:

inf
fθ:RD→RD
E
x∼D(Ω)
ϵ∼N(0,I)

1

2∥fθ(x + σϵ) −x∥2
2


,

where D(Ω) now denotes the empirical distribution over clean training images. The training objective
for the Noise2Noise denoiser is:

inf
fθ:RD→RD
E
x∼D(Ω)
ϵ1,ϵ2∼N(0,I)

1

2∥fθ(x + σϵ1) −(x + σϵ2)∥2
2


.

Note that this requires access to independent noisy copies of the same clean image during training.

Evaluation details.
We evaluate each denoiser by measuring their average peak signal-to-noise
ratio (PSNR) in decibels (dB) on 100 randomly-drawn images from the ImageNet validation set,
randomly cropped to 256 × 256. We corrupt each held-out image with Gaussian noise with the
same standard deviation that the respective models were trained on (σ ∈{1, 2}) and denoise them
using each neural model along with BM3D [Dabov et al., 2007], a popular classical baseline for
unsupervised denoising.

B.3
Representation learning

We use the β-VAE implementation from the AntixK PyTorch-VAE repo and use the default hyperpa-
rameters (in particular, we set β = 10) but set the latent dimension to 32, as we find that this yields
more meaningful latent traversals. Training this β-VAE takes approximately 30 minutes on a single
V100 GPU.

23


---Page Break---
We describe the architecture and training details for our regularized and unregularized autoencoder
which we use to generate the latent traversals in Figures 6 and 7. Training this autoencoder with the
de

Deterministic autoencoder architecture.
Our autoencoder operates on 256 × 256 images from the
CelebA dataset. To reduce the memory and compute costs of our autoencoder, we perform a discrete
cosine transform (DCT) using the torch-dct package and keep only the first 80 DCT coefficients.
We then pass these coefficients into our autoencoder.

Our deterministic autoencoder consists of an encoder fθ followed by a decoder gϕ. The encoder fθ
is parametrized as a two-layer MLP with 10,000 hidden units; the latent space is 700-dimensional.
The decoder gϕ consists of a two-layer MLP with 10,000 hidden units and 3 ∗80 ∗80 = 19200
output dimensions, followed by an inverse DCT, and finally a UNet. We use the same UNet as in the
denoising experiments described in Appendix B.2.

Training details.
We train our autoencoders with the following objective:

inf
fθ,gϕ
E
x∼D(Ω)

1

2∥gϕ(fθ(x)) −x∥2
2 + ηRx; (fθ)

,
(34)

where D(Ω) is the CelebA training set [Liu et al., 2015].

This is a standard deterministic autoencoder objective, with our regularizer approximating the
Jacobian nuclear norm ∥Jfθ[x]∥∗of the encoder fθ.

We train the unregularized autoencoder with η = 0, and the regularized autoencoder with η = 0.5. In
both cases, we train on the CelebA training set for 4 epochs using the AdamW optimizer [Loshchilov
and Hutter, 2019] with a learning rate of 10−4. Training these autoencoders takes approximately 4
hours each on a single V100 GPU.

Generating latent traversals.
To generate the latent traversals for our autoencoder, we draw a
point x from the training set, compute the encoder Jacobian Jfθ[x], and take its SVD to obtain
Jfθ[x] = UΣV ⊤. We then take the first 5 left-singular vectors (i.e. the first 5 columns of U) and
compute z = fθ(x) + αud
θ(x), where ud
θ(x) denotes the d-th column of U. Here α denotes a scalar
coefficient; it ranges over [−20000, 20000] for the unregularized autoencoder and [−2000, 2000] for
the regularized autoencoder.

To generate the latent traversals for the β-VAE, we encode the same training point x and replace the
d-th latent coordinate with an equispaced traversal of [−3, 3] for d ∈{1, 2, 11}; these are the first
three meaningful latent traversals for this training point.

24


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Our main claim is to have proven Theorem 3.1 – which we do in Appendix
A.2. The simplicity of our method is apparent – one can implement our regularizer in a
few lines of code – and its efficiency follows directly from eliminating the need to compute
SVDs or Jacobian matrices. We demonstrate our method’s accuracy in Section 4.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Our method’s primary limitation is that the Hutchinson-based estimator of the
squared Jacobian Frobenius norm introduced in Section 3.3 introduces error that manifests
itself as somewhat diffuse boundaries in the 2D experiment in Section 4. We discuss this
limitation in Section 6.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

25


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: We include full proofs of all theoretical results in Appendix A. We also include
a proof sketch for our primary result (Theorem 3.1) in the main body of the paper; see
Section 3.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Our primary contribution is our regularizer (6), which can be straightforwardly
implemented from the formula in a few lines of code. We have included full experimental
details in Appendix B and also attached code for our experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).

26


---Page Break---
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We have uploaded a zip file with our submission that includes code for training
our models and running our experiments.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: We include full experimental details in Appendix B.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In Table 1, we report 1-sigma error bars for the PSNR attained by each denoiser
across the held-out images.

Guidelines:

27


---Page Break---
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: We report runtimes for experiments and training runs throughout Appendix B.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: Our work conforms with the Code of Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: This is a primarily theoretical paper; we do not anticipate a direct path towards
negative applications arising from this work.

28


---Page Break---
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: Our method does not have a high risk of misuse.
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We cite the datasets, pre-existing algorithms, and code libraries used for our
experiments throughout our work.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

29


---Page Break---
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justification: We do not release new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing or research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?

Answer: [NA]

Justification: Our work does not require IRB approval.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

30


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

31


---Page Break---
