Improving Adaptivity via Over-Parameterization in
Sequence Models

Yicheng Li
Department of Statistics and Data Science
Tsinghua University, Beijing, China
liyc22@mails.tsinghua.edu.cn

Qian Lin ∗
Department of Statistics and Data Science
Tsinghua University, Beijing, China
qianlin@tsinghua.edu.cn

Abstract

It is well known that eigenfunctions of a kernel play a crucial role in kernel
regression. Through several examples, we demonstrate that even with the same
set of eigenfunctions, the order of these functions significantly impacts regression
outcomes. Simplifying the model by diagonalizing the kernel, we introduce an
over-parameterized gradient descent in the realm of sequence model to capture the
effects of various orders of a fixed set of eigen-functions. This method is designed
to explore the impact of varying eigenfunction orders. Our theoretical results show
that the over-parameterization gradient flow can adapt to the underlying structure of
the signal and significantly outperform the vanilla gradient flow method. Moreover,
we also demonstrate that deeper over-parameterization can further enhance the
generalization capability of the model. These results not only provide a new
perspective on the benefits of over-parameterization and but also offer insights into
the adaptivity and generalization potential of neural networks beyond the kernel
regime.

1
Introduction

In recent years, the remarkable success of neural networks in a wide array of machine learning
applications has spurred a search for theoretical frameworks capable of explaining their efficacy
and efficiency. One such framework is the Neural Tangent Kernel (NTK) theory (see, e.g., Jacot
et al. [2018], Allen-Zhu et al. [2019]), which has emerged as a pivotal tool for understanding the
dynamics of neural network training in the infinite-width limit. The NTK theory posits that the
training dynamics of wide neural networks can be closely approximated by a kernel gradient descent
method with the corresponding NTK, elucidating their convergence behaviors during gradient descent
and shedding light on their generalization capabilities. Parallel to this, an extensive literature on
kernel regression (see, e.g., Bauer et al. [2007], Yao et al. [2007]) has studied its generalization
properties, showing its minimax optimality under certain conditions and providing insights into the
bias-variance trade-off. Thus, one can almost fully understand the generalization properties of neural
networks in the NTK regime by analyzing the kernel regression method.

However, the application of NTK theory to analyze neural networks, while invaluable, essentially
frames the problem within a traditional statistical method by a fixed kernel. The NTK analysis, by its
reliance on the fixed kernel approximation, can not entirely account for the adaptability and flexibility
exhibited by neural networks, particularly those of finite width that deviate from the theoretical
infinite-width limit [Woodworth et al., 2020]. Moreover, empirical evidence [Wenger et al., 2023,
Seleznova and Kutyniok, 2022] also suggests that the assumption of a constant kernel during training,
a cornerstone of NTK analysis, may not hold in practical scenarios where the network architecture or

∗Corresponding author Qian Lin also affiliates with Beijing Academy of Artificial Intelligence, Beijing,
China

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
initialization conditions foster a dynamic evolution of the kernel. Also, Gatmiry et al. [2021] showed
the benefits brought by the adaptivity of the kernel on a three-layer neural network. These results
underscore the need for a more nuanced understanding of neural network training dynamics, one that
considers the intricate interplay between network architecture, initialization, and the optimization
process beyond the simplifications of NTK theory.

Recently, another branch of research has focused on the over-parameterization nature of neural
networks beyond the NTK regime, exploring how over-parameterization can lead to implicit regu-
larization and even improve the generalization. In terms of training dynamics, studies (Hoff [2017],
Gunasekar et al. [2017], Arora et al. [2019a], Kolb et al. [2023], etc.) in this domain have revealed
that over-parameterized models, particularly those trained with gradient descent and its variants,
exhibit biases towards simpler, more generalizable functions, even in the absence of explicit reg-
ularization terms. Moreover, in terms of generalization, recent works [Vaškeviˇcius et al., 2019,
Zhao et al., 2022, Li et al., 2021a] have shown that in the setting of high dimensional linear regres-
sion, over-parameterized models with proper initialization and early stopping can achieve minimax
optimal recovery under certain conditions. These results underscore the potential and benefits of
over-parameterized models that go beyond the traditional statistical paradigms.

In this work, we will incorporate the insights from the kernel regression and the over-parameterization
theory to investigate how over-parameterization can improve generalization and also adaptivity under
the non-parametric regression framework. As a first step towards this direction, we will focus on the
sequence model, which is an approximation of a wide spectrum of non-parametric models including
kernel regression. We will show that, by dynamically adapting to the underlying structure of the signal
during the training process, over-parameterization method with gradient descent can significantly
improve the generalization properties compared with the fixed-eigenvalues method. We believe that
our results provide a new perspective on the benefits of over-parameterization and offer insights into
the adaptivity and generalization properties of neural networks beyond the NTK regime.

1.1
Our contributions

Limitations of the (fixed) kernel regression.
In this work, we first investigate the limitations of
the (fixed) kernel regression method by specific examples, illustrating that the traditional kernel
regression method suffers from the misalignment between the kernel and the truth function. We
show that even when the eigen-basis of the kernel is fixed, the associated eigenvalues, particularly
their alignment with the truth function’s coefficients in the eigen-basis, can significantly affect the
generalization properties of the method.

Advantages of over-parameterized gradient descent.
Focusing on the alignment between the
kernel’s eigenvalues and the truth signal (the truth function’s coefficients), we consider the sequence
model and introduce an over-parameterization method (8) that can dynamically adjust the eigenvalues
during the learning process. We show that with proper early-stopping, the over-parameterization
method can achieve nearly the oracle convergence rate regardless of the underlying structure of the
signal, significantly outperforming the vanilla fixed-eigenvalues method when the misalignment is
severe. In addition, the over-parameterization method is also adaptive by its universal choice of the
stopping time, which is independent of the signal’s structure.

Benefits of deeper parameterization.
Moreover, we also consider deeper over-parameterization
(14) and explore how depth affects the generalization properties of the over-parameterization method.
Our results show that adding depth can further ease the impact of the initial choice of the eigenvalues,
thus improving the generalization capability of the model. We also provide numerical experiments to
validate our theoretical results in Section C.

1.2
Notations

We denote by ℓ2 =
n
(aj)j≥1 | P

j≥1 a2
j < ∞
o
the space of square summable sequences. We write
a ≲b if there exists a constant C > 0 such that a ≤Cb and a ≍b if a ≲b and b ≲a, where the
dependence of the constant C on other parameters is determined by the context.

2


---Page Break---
2
Limitations of Fixed Kernel Regression

Let us consider the non-parametric regression problem given by y = f ∗(x) + ε, where ε is the
noise with mean zero and variance σ2, x ∈X and X is the input space with µ being a probability
measure supported on X. The function f ∗(x) represents the unknown regression function we aim
to learn. Suppose we are given samples {(xi, yi)}n
i=1, drawn i.i.d. from the model. We denote
X = (x1, . . . , xn)⊤and Y = (y1, . . . , yn)⊤.

Let k : X × X →R be a continuous positive definite kernel and Hk be its associated reproducing
kernel Hilbert space (RKHS). The well-known Mercer’s decomposition [Steinwart and Scovel, 2012]
of the kernel function k gives

k(x, y) =

∞
X

j=1
λjej(x)ej(y),
(1)

where (ej)j≥1 is an orthonormal basis of L2(X, dµ), and (λj)j≥1 are the eigenvalues of k in

descending order. Moreover, we can introduce the feature map Φ(x) = (λ

1
2
j ej(x))j≥1 : X →ℓ2 (as
a column vector) such that k(x, x′) = ⟨Φ(x), Φ(x′)⟩. With the feature map, a function f ∈Hk can
be represented as f(x) = ⟨Φ(x), β⟩ℓ2 for some β ∈ℓ2.

Defining the empirical loss as L =
1
2n
Pn
i=1(yi −f(xi))2, we can consider an estimator ft =
⟨Φ(x), βt⟩ℓ2 governed by the following gradient flow on the feature space

˙βt = −∇βL = 1

n

n
X

i=1
(yi −⟨Φ(xi), βt⟩ℓ2)Φ(xi),
where
β0 = 0.
(2)

This kernel gradient descent (flow) estimator corresponds to neural networks at infinite width limit by
the celebrated neural tangent kernel (NTK) theory [Jacot et al., 2018, Allen-Zhu et al., 2019].

An extensive literature [Yao et al., 2007, Lin et al., 2018, Li et al., 2024a] has studied the generalization
performance of such kernel gradient descent estimator. From the Mercer’s decomposition, we can
further introduce interpolation spaces for s ≥0 as

[Hk]s :=
n ∞
X

j=1
βjλ

s
2
j ej
 (βj)j≥1 ∈ℓ2o
,
(3)

which is equipped with the norm ∥f∥[Hk]s = ∥β∥ℓ2 for f = P∞
j=1 βjλ

s
2
j ej. Particularly, the
interpolation space [Hk]1 corresponds to the RKHS Hk itself. Then, assuming the eigenvalue decay
rate λj ≍j−γ, the standard results (see, e.g., Yao et al. [2007], Li et al. [2024a]) in kernel regression
state that the optimal rate of convergence under the source condition f ∗∈[Hk]s with ∥f ∗∥[Hk]s ≤1

is n−
sγ
sγ+1 . However, since the interpolation space [Hk]s is defined via the eigen-decomposition of
the kernel, the generalization performance of kernel regression methods is ultimately related to the
eigen-decomposition of the kernel and the decomposition of the target function under the basis, so the
performance is intrinsically limited by the relation between the target function and the kernel itself.
In other words, the choice of the kernel could affect the performance of the method. To demonstrate
this quantitatively, let us consider the following examples.
Example 2.1 (Eigenfunctions in common order). It is well known that kernels possessing certain
symmetries, such as dot-product kernels on the sphere or translation-invariant periodic kernels on the
torus, share the same set of eigenfunctions (such as the spherical harmonics or the Fourier basis). If
we consider a fixed set of eigenfunctions {ej}j≥1 and a given truth function f ∗, for two kernels k1
and k2 with eigenvalue decay rates λj,1 ≍j−γ1 and λj,2 ≍j−γ2 respectively, it follows that:

f ∗∈[Hk1]s1 ⇐⇒f ∗∈[Hk2]s2
for
γ1s1 = γ2s2.

Given that the convergence rate is dependent solely on the product sγ, the convergence rates relative
to the two kernels will be identical.

Example 2.1 seems to show that when the eigenfunctions are fixed, kernel regression methods yield
similar performance across different kernels. However, it’s important to note that this similarity is due

3


---Page Break---
to both kernels having the same eigenvalue decay order, which aligns with the predetermined order
of the basis. In fact, if the eigenvalue decay order of a kernel deviates from that of the true function,
even if the eigenfunction basis remain the same, it can lead to significantly different convergence
rates. Let us consider the following example to illustrate this point.
Example 2.2 (Low-dimensional structure). Consider translation-invariant periodic kernels on the
torus Td = [−1, 1)d with the uniform distribution. Then, their eigenfunctions are given by the
Fourier basis ϕm(x) = exp(iπ ⟨m, x⟩), m ∈Zd. Within this basis, a target function f ∗(x) can be
represented as:

f ∗=
X

m∈Zd
fmϕm(x).

Assuming f ∗exhibits a low-dimensional structure, specifically f ∗(x) = g(x1, . . . , xd0) for some
d0 < d, and considering g belongs to the Sobolev space Ht(Td0) of order t, the coefficients fm can
be shown to simplify to:

fm =
gm1,
m = (m1, 0), m1 ∈Zd0,
0,
otherwise.

Let us now consider two translation-invariant periodic kernels k1 and k2 given in terms of their
eigenvalues: k1 is given by λm,1 = (1 + ∥m∥2)−r for some r > d/2, whose RKHS is the full-
dimensional Sobolev space Hr(Td); k2 is given by λm,2 = (1 + ∥m∥2)−r for m = (m1, 0) and
λm,2 = 0 otherwise. Then, the function f ∗belongs to both [Hk1]s and [Hk2]s for s = t/r. After
reordering the eigenvalues in descending order, the decay rates for the two kernels are identified
as γ1 = 2r/d and γ2 = 2r/d0. Thus, the convergence rates with respect to the two kernels are
respectively:
2t
2t + d
and
2t
2t + d0
.

Therefore, we see that when d is significantly larger than d0, the convergence rate for the second
kernel notably surpasses that of the first.

This example illustrates that the eigenvalues can significantly impact the learning rate, even when
the eigenfunctions are the same. In the scenario presented, the second kernel benefits from the
low-dimensional structure of the target function by focusing only on the relevant dimensions, whereas
the first one suffers from the curse of dimensionality since it considers all dimensions. The key point
to take away from this example is the alignment between the kernel and the target function. To
generalize this example, we can consider the following example where the order of the eigenvalues
does not align with the order of the target function’s coefficients.
Example 2.3 (Misalignment). Let us fix a set of the eigenfunctions (ej)j≥1 and expand the truth
function as f ∗= P

j≥1 θ∗
j ej. Note that by giving (ej)j≥1, we already defined an order of the basis
in j, but coefficients θ∗
j of the truth function are not necessarily ordered by j. Suppose that an index

sequence ℓ(j) gives the descending order of
θ∗
ℓ(j)
. Then we can characterize the misalignment by

the difference between ℓ(j) and j. Specifically, we assume that
θ∗
ℓ(j)
 ≍j−(p+1)/2
and
ℓ(j) ≍jq
for
p > 0, q ≥1,
(4)

where larger q indicates a more severe misalignment. In terms of eigenvalues, let us consider
λj,1 ≍j−γ, which is in the order of j, while λℓ(j),2 ≍j−γ, which is in the order of ℓ(j). Then, the
convergence rates with the two sequences of coefficients are respectively
p
p + q
and
p
p + 1.

Therefore, the convergence rates can differ greatly if the misalignment is significant, namely when q
is large.

From Example 2.2 and Example 2.3, we find that it is beneficial that the eigenvalues of the kernel
align with the structure of the target function. However, one can hardly choose the proper kernel a
priori, especially when the structure of the target function is unknown, so the fixed kernel regression
can be limited by the kernel itself and be unsatisfactory. Motivated by these examples, we would like
to explore the idea of an “adaptive kernel approach,” where the kernel can be learned from the data.

4


---Page Break---
3
Adapting the Eigenvalues by Over-parameterization in the Sequence Model

Motivated by the examples in the last section, as a first step toward the adaptive kernel approach, we
consider adapting the eigenvalues of the kernel with eigenfunctions fixed. To simplify the analysis,
we would like to the following sequence model, which captures the essences of many statistical
models [Brown et al., 2002, Johnstone, 2017].

The sequence model
Let us consider the sequence model [Johnstone, 2017]

zj = θ∗
j + ξj,
j ≥1
(5)

where (zj)j≥1 is the observation, θ∗= (θ∗
j )j≥1 ∈ℓ2 is a sequence of unknown truth parameters
and ξj, j ≥1 are (not necessarily independent) ϵ2-sub-Gaussian random variables with mean zero
and variance at most ϵ2. For any estimator ˆθ = (ˆθj)j≥1, the generalization error is measured
by R(ˆθ; θ∗) = P∞
j=1(ˆθj −θ∗
j )2. Under the asymptotic framework, we are often interested in
the behavior of the generalization error as ϵ →0. Here, we note that the connection between
non-parametric regression and the sequence model yields ϵ2 ≍n−1.

To see the connection between the sequence model and the non-parametric regression model, we first
write the gradient flow (2) in the RKHS in the matrix form as

˙βt = −∇βL = −1

nΦ(X)Φ(X)⊤βt + 1

nΦ(X)y,

where the feature matrix Φ(X) = (Φ(x1), . . . , Φ(xn))∞×n and y = (y1, . . . , yn)⊤. Now, since the
eigenfunctions (ej)j≥1 are fixed, intuitively, the gradient flow can be diagonalized in the eigen-basis
since 1

nΦ(X)Φ(X)⊤≈Λ = diag(λ1, λ2, . . . ) and the noise components are approximately normal
with variance σ2/n by the central limit theorem. Thus, we reach the sequence model. We refer to
Subsection B.1 for a more detailed explanation of the connection between the sequence model and
the kernel regression model.

Regarding the power series expansion (3) in RKHS, for a sequence (λj)j≥1 of descending positive

numbers (e.g., λj = j−γ), we can consider similarly the parameterization θj = λ

1
2
j βj, j ≥1 in ℓ2.
Since (λj)j≥1 corresponds to the eigenvalues of the kernel in the kernel regression, here we also
refer to (λj)j≥1 as the “eigenvalues” with a little abuse of terminology.

With the component-wise loss function Lj(θj) = 1

2(θj −zj)2, we can apply a gradient descent
(gradient flow) with early stopping to derive a component-wise estimator ˆθj. If we directly param-

eterize θj = λ

1
2
j βj with only βj trainable, we obtain the vanilla gradient descent method, which
is just the diagonalized version of the kernel gradient descent. The estimator is simply given by
ˆθj = (1 −e−λjt)zj, where t is the stopping time, and its generalization error is easily computed as

ER(ˆθGF; θ∗) = B2
GF(t; θ∗) + ϵ2VGF(t) =

∞
X

j=1

 
e−λjtθ∗
j
2 + ϵ2
∞
X

j=1

 
1 −e−λjt2 .
(6)

We note here that these quantities also correspond to generalization error in the (fixed) kernel
regression setting [Li et al., 2024a]. In particular, under the setting of (4) and λj ≍j−γ, by choosing
t ≍ϵ−2qγ

p+q , we obtain the convergence rate ϵ

2p
p+q , which is far from optimal if q is large.

3.1
Over-parameterized gradient descent

By the discussion in the previous section, we find it essential to adjust the eigenvalues beyond the
fixed ones (λj)j≥1. Inspired by the over-parameterization nature of neural networks, we can also
consider over-parameterization with gradient descent in our sequence model to train the eigenvalues:
Replacing λ1/2
j
with trainable parameter aj, let us parameterize

θj = ajβj,
(7)

5


---Page Break---
where aj aims to learn the eigenvalues and βj aims to learn the signal. We consider the following
gradient flow (simultaneously for each component j):

˙aj = −∇ajLj,
˙βj = −∇βjLj,

aj(0) = λ1/2
j
,
βj(0) = 0.
(8)

Here, (λj)j≥1 serves as the initial eigenvalues, while the trainable parameters (aj)j≥1 are updated to
adjust the eigenvalues during the training process.

To state our results with the most generality, let us introduce the following quantities on the target
parameter sequence θ∗:

Jsig(δ) :=

j :
θ∗
j
 ≥δ
	
,
Φ(δ) := |Jsig(δ)|,
Ψ(δ) =
X

j /∈Jsig(δ)
(θ∗
j )2.
(9)

The quantity Φ(δ) measures the number of significant components in the target parameter sequence
θ, while Ψ(δ) measures the contribution of the insignificant components, which are commonly
considered in the literature on the sequence model [Johnstone, 2017]. For the concrete setting of (4),
it is easy to show that

Φ(δ) ≍δ−
2
p+1 ,
Ψ(δ) ≍δ

2p
p+1 .
(10)

Moreover, we make the following assumption on the span of the significant components.
Assumption 1. There exists constants κ ≥0 and Csig > 0 such that

max Jsig(δ) ≤Csigδ−κ,
∀δ > 0.
(11)

Assumption 1 says that the span of the significant components, namely those with
θ∗
j
 ≥δ, grows at
most polynomially in 1/δ. This assumption is mild and holds for many practical settings, such as
cases considered in Example 2.3 (κ =
2q
p+1 for the first kernel). In other perspective, it imposes a
mild condition on the misalignment between the ordering of the truth signal and the ordering of the
eigenvalues, where κ measures the misalignment between the ordering of θj and the ordering of j
itself. Then, the following theorem characterizes the generalization error of the resulting estimator.
Theorem 3.1. Consider the sequence model (5) under Assumption 1. Fix λj ≍j−γ for some γ > 1
and let ˆθOp be the estimator given by the gradient flow (8) stopped at time t. Then, there exists some
constants B1, B2 > 0 such that when B1ϵ−1 ≤t ≤B2ϵ−1, we have

ER(ˆθOp, θ∗) ≲ϵ2 h
Φ(ϵ) + ϵ−1/γi
+ Ψ (ϵ ln(1/ϵ))
as
ϵ →0.
(12)

3.2
Towards deeper over-parameterization

Let us further introduce deeper over-parameterization by adding extra D-layers:

θj = ajbD
j βj
(13)

and consider the gradient flow

˙aj = −∇ajLj,
˙bj = −∇bjLj,
˙βj = −∇βjLj,

aj(0) = λ1/2
j
,
bj(0) = b0 > 0,
βj(0) = 0,
(14)

where b0 is the common initialization of all bj.
We remark here if one considers the over-
parameterization θj = ajbj,1 · · · bj,Dβj with the same initialization bj,k = b0, k = 1, . . . , D, then
bj,k’s remain to be the same by symmetry, so this is equivalent to our parameterization θj = ajbD
j βj.
The following theorem presents an upper bound for the generalization error by this deeper over-
parameterized gradient flow.
Theorem 3.2. Consider the sequence model (5) under Assumption 1. Fix λj ≍j−γ for some γ > 1
and let ˆθOp,D be the estimator given by the gradient flow (14) stopped at time t. Then, by choosing
b0 ≍ϵ
1
D+2 , there exists some constants B1, B2 > 0 such that when B1ϵ−1 ≤bD
0 t ≤B2ϵ−1, we
have

ER(ˆθOp,D, θ∗) ≲ϵ2 h
Φ(ϵ) + ϵ−
2
D+2
1
γ
i
+ Ψ (ϵ ln(1/ϵ))
as
ϵ →0.
(15)

6


---Page Break---
3.3
Discussion of the results

Benefits of Over-parameterization
Theorem 3.1 and Theorem 3.2 demonstrate the advantage of
over-parameterization in the sequence model. Compared with the vanilla fixed-eigenvalues gradient
descent method, the over-parameterized gradient descent method can significantly improve the
generalization performance by adapting the eigenvalues to the truth signal. For a more concrete
example, if we consider the setting of (4), plugging (10) yields the following corollary.
Corollary 3.3. Consider the over-parameterized gradient descent in (8) (setting D = 0) or (14).
Suppose (4) holds and λj ≍j−γ for γ > 1 and γ ≥p+1

D+2. Then, by choosing b0 ≍ϵ
1
D+2 (if D ̸= 0)

and t ≍ϵ−2D+2

D+2 , we have

ER(ˆθOp,D, θ∗) ≲ϵ

2p
p+1 (ln(1/ϵ))

2p
p+1
as
ϵ →0.
(16)

In comparison, the vanilla gradient flow method yields the rate ϵ

2p
p+q .

Ignoring the logarithmic factor, Corollary 3.3 shows that the over-parameterized gradient descent
method can achieve a nearly optimal rate ϵ

2p
p+1 , while the vanilla gradient descent method only
achieves the rate ϵ

2p
p+q . The improvement is significant when q is large, which corresponds to the case
that the misalignment between the ordering of the truth signal and the ordering of the eigenvalues
is severe. Moreover, if we return to the low-dimensional regression function in Example 2.2 with
the isotropic kernel k1, we can see that while the vanilla gradient descent method suffers from the
curse of dimensionality with the rate
2t
2t+d, the over-parameterization leads to the dimension-free rate
2t
2t+d0 . Therefore, the over-parameterization significantly improves the generalization performance.

Learning the eigenvalues
To further investigate how the eigenvalues are adapted by over-
parameterized gradient descent, we present the following proposition.
Proposition 3.4. Given the same conditions as in Theorem 3.2 or Theorem 3.1 (with D = 0 and
bD
j = 1 for Theorem 3.1), the term learning the eigenvalues aj(t)bD
j (t) is non-decreasing in t for
each j. Moreover, letting δ ∈(0, 1), when ϵ is small enough, the following holds at time t chosen as
in Theorem 3.1 or Theorem 3.2:

• Signal component: There exist constants C, c > 0 such that for any component satisfying
θ∗
j
 ≥Cϵ ln(1/ϵ), it holds with probability at least 1 −δ that

aj(t)bD
j (t) ≥c
θ∗
j

D+1
D+2 .
(17)

• Noise component: There exist constants c, C, C′ > 0 such that, for any component where
θ∗
j
 ≤ϵ and λj ≤cϵ
2
D+2 , it holds with probability at least 1 −δ that

aj(t)bD
j (t) ≤Cλ

1
2
j ϵ
D
D+2 = C′aj(0)bD
j (0).
(18)

From this proposition, we can see that for the signal components, the eigenvalues are learned to be
at least a constant times a certain power of the truth signal magnitude. Thus, over-parameterized
gradient descent adjusts the eigenvalues to match the truth signal as expected. In the case of noise
components, although the eigenvalues are still increasing due to the training process, the eigenvalues
do not exceed the initial values by some constant factor, provided that λj is relatively small. This
finding suggests that over-parameterized gradient descent effectively adapts eigenvalues to the truth
signal while mitigating overfitting to noise. We remark that when λj is relatively large, the method
still tends to overfit the noise components, contributing an extra ϵ−
2
D+2
1
γ term in the generalization
error, but this term becomes negligible for large γ. Moreover, we also remark that there is a ln(1/ϵ)
gap between the signal and noise components. This is because the signal and the noise can not be
distinguished for the components in the middle.

Adaptive choice of the stopping time
A notable advantage of the over-parameterized gradient
descent method is its adaptivity. Consider the scenario described by (4), vanilla gradient descent
requires the selection of a stopping time t ≍ϵ−(2qγ)/(p+q) to achieve the optimal convergence

7


---Page Break---
rate. However, this choice of stopping time critically depends on the unknown parameters p and
q of the truth parameter, posing a significant challenge in practical applications. In contrast, the
over-parameterized gradient descent only need to choose the stopping time as t ≍ϵ−2D+2

D+2 , which
does not rely on the unknown truth parameters, while still achieving the nearly optimal convergence
rate. This independence from the truth parameters allows the over-parameterization approach to
adaptively accommodate any truth parameter structure by employing a fixed stopping time, without
the need for prior knowledge about the truth function’s properties.

Effect of the depth
The results in Theorem 3.2 also show that deeper over-parameterization can
further improve the generalization performance. In the two-layer over-parameterization, the extra
term ϵ−1/γ in Theorem 3.1 emerges due to the limitation of the adapting large eigenvalues. With the
introduction of depth, namely adding extra D layers to the model with proper initialization, this term
can be improved to ϵ−
2
D+2
1
γ in Theorem 3.2. This improvement suggests that the depth can refine the
model’s sensitivity to eigenvalue adaptation, enabling a more nuanced adjustment to the underlying
signal structure. This finding underscores the importance of model depth in enhancing the learning
process, providing also theoretical evidence for the empirical success of deep learning models.

Comparison with previous works
We will compare our results with the existing literature [Zhao
et al., 2022, Li et al., 2021a, Vaškeviˇcius et al., 2019] on the generalization performance of over-
parameterized gradient descent in the following aspects:

• Problem settings: While the existing literature [Zhao et al., 2022, Li et al., 2021a, Vaške-
viˇcius et al., 2019] investigate the realms of high-dimensional linear regression, focus-
ing on implicit regularization and sparsity, the present study dives into kernel regression
and its approximation by Gaussian sequence models, emphasizing the adaptivity of over-
parameterization to the underlying signal’s structure, a leap towards understanding model
complexity beyond mere regularization. Moreover, while the literature primarily focuses on
the setting of strong signal, weak signal and noise separation, we consider the more general
setting of the sequence model with arbitrary signal and noise components.

• Over-parameterization setup: The existing work Zhao et al. [2022] considers the over-
parameterization setup by the two-layer Hadamard product θ = a⊙b where the initialization
is the same for each component that a(0) = α1 and b(0) = 0. In comparison, our
work considers initializing the eigenvalues aj(0) = λ1/2
j
differently for each component.
Moreover, we extend the over-parameterization to deeper models by adding extra D layers.
Although Vaškeviˇcius et al. [2019] and the subsequent work Li et al. [2021a] also consider the
deeper over-parameterization, their over-parameterization is in the form of θ = u⊙D −v⊙D
with u(0) = v(0) = α1. Unfortunately, though being easy to analysis because of the
homogeneous initialization, this setup could not bring insights into the learning of the
eigenvalues, which is the key to our results. Furthermore, the analysis for Theorem 3.2
involves the interplay between the differently initialized aj and bj, so our analysis is more
involved than the existing works. We also remark that although we only consider the gradient
flow in the analysis, the results can be extended to the gradient descent with proper learning
rates.

• Interpretation of the over-parameterization:
The previous works view the over-
parameterization mainly as a mechanism for implicit regularization, while our work provides
a novel perspective that over-parameterization adapts to the structure of the truth signal by
learning the eigenvalues. Our theory also aligns with the neural network literature [Yang
and Hu, 2022, Ba et al., 2022], where over-parameterization with gradient descent is known
to be beneficial in learning the structure of the target function.

• Connection to sparse recovery: Our results can be phrased for the setting of high dimen-
sional regression with sparsity. Taking a sparse signal (θ∗
j )j≥1, e.g., θ∗
j = 1 for j ∈S,
|S| = s and θ∗
j = 0 for j /∈S, we find that Φ(ϵ) = s and Ψ(ϵ) = 0. Consequently, ignoring
the extra error term, the resulting rate obtained by Theorem 3.1 or Theorem 3.2 is ˜O(s/n)
(ignoring the logarithmic factor). This rate coincides with the minimax rate for sparse
recovery in high-dimensional regression.

8


---Page Break---
3.4
Proof outline

In this subsection, we will provide an outline of the proof of Theorem 3.1 and Theorem 3.2. For the
detailed proof, we refer to Section D for the analysis of the gradient flow equation and Section E for
the generalization error.

Equation analysis
The proof of Theorem 3.1 and Theorem 3.2 relies on the analysis of the gradient
flow (8) and (14) for each component j. For notation simplicity, we will suppress the index j in the
following discussion. Firstly, the symmetry of the equation allows us to consider only the case z > 0.
Then, one can find that
d
dta2 = d

dtβ2 = 1

D
d
dtb2 = 2θ(z −θ),

so we get the conservation quantities a2(t) ≡β2(t) + λ and b2(t) ≡Dβ2(t) + b2
0.

Consequently, for the case D = 0, we can obtain the explicit gradient flow of θ:

˙θ =
q

a4
0 + 4θ2(z −θ),
θ(0) = 0.

Since
p

a4
0 + 4θ2 can be bounded by a multiple of a2
0 + 2θ, we can consider the other equation
˙θ = (a2
0 + 2θ)(z −θ), which admits a closed-form solution.

For the case D ≥1, the equation is more complicated. We will apply a multiple stage analysis
concerning both the effect of a(t) and b(t).

The generalization error
In terms of the generalization error, we first separate the noise case when
|ξj| ≥
θ∗
j
/2 and the signal case when |ξj| <
θ∗
j
/2. For the noise case, we apply the analysis of
the equation to show that θj(t) is bounded roughly by λj for our choice of t. Moreover, the fact that
λj is summable ensures that error of these noise components does not sum up to infinity. On the
other hand, for the signal case, if
θ∗
j
 ≥ϵ ln(1/ϵ), we can show that our choice of t allows θj(t) to
exceed z/2 and converge to z close enough, so the error in these components is only caused by the
random noise and sum up to ϵ2Φ(ϵ). In addition, the remaining signal components contribute to the
error term Ψ(ϵ ln(1/ϵ)). Summing up these two terms, we can obtain the desired generalization error
bound.

4
Numerical Experiments

In this section, we provide some numerical experiments to validate the theoretical results. For more
detailed numerical experiments, please refer to Section C.

We approximate the gradient flow equation (22) and (30) by discrete-time gradient descent and
truncate the sequence model to the first N terms for some very large N. We consider the settings as
in Corollary 3.3 that θ∗is given by (4) for some p > 0 and q ≥1. We set ϵ2 = n−1, where n can be
regarded as the sample size, and consider the asymptotic generalization error rates as n grows.

We first compare the generalization error rates between vanilla gradient descent and over-
parameterized gradient descent (OpGD) in Figure 1 on page 10. The results show that the over-
parameterized gradient descent can achieve the optimal generalization error rate, while the vanilla
gradient descent suffers from the misalignment caused by q > 1 and thus has a slower convergence
rate. Moreover, with a logarithmic least-squares fitting, we find that the resulting generalization error
rates also match the theoretical results in Corollary 3.3 (0.5 for OpGD and 0.33 for vanilla GD).

Additionally, we investigate the evolution of the eigenvalue terms aj(t)bD
j (t) over time t as discussed
in Proposition 3.4. The results are shown in Figure 2 on page 10. We find that the eigenvalue terms
can indeed adapt to the underlying structure of the signal: for large signals, the eigenvalue terms
approach the signals as the training progresses, while for small signals, the eigenvalue terms do not
increase significantly. Moreover, we find that deeper over-parameterization reduces the fluctuations
of the eigenvalue terms for the noise components, and thus improves the generalization performance
of the model.

In summary, the numerical experiments validate our theoretical results and provide insights into the
adaptivity and generalization properties of the over-parameterized gradient descent method.

9


---Page Break---
2000
10000
n (logarithmic)

1.5

1.4

1.3

1.2

1.1

1.0

0.9

log10  generalization error 

Generalization Errors; optimal rate=0.500

Vanilla GD
log Err
0.374log n + 0.29

OpGD
log Err
0.533log n + 0.65

2000
10000
n (logarithmic)

1.5

1.4

1.3

1.2

1.1

1.0

0.9

log10  generalization error 

Generalization Errors; optimal rate=0.500

Vanilla GD
log Err
0.341log n + 0.21

OpGD
log Err
0.499log n + 0.55

Figure 1: Comparison of the generalization error rates between vanilla gradient descent and over-
parameterized gradient descent (OpGD). We set p = 1 and q = 2 for the truth parameter θ∗, and
γ = 1.5 for the left column and γ = 3 for the right column. For each n, we repeat 64 times and plot
the mean and the standard deviation.

100
120
140
160
180
200
index j

2.5

2.0

1.5

1.0

log10 magnitude

t=0

100
120
140
160
180
200
index j

t=150

100
120
140
160
180
200
index j

t=300

Model 
= ab  (D = 1)

Figure 2: The evolution of the trainable eigenvalues aj(t)bD
j (t) over the time t across components
j = 100 to 200 for D = 1. The blue line shows the eigenvalues and the black marks show the
non-zero signals scaled according to Proposition 3.4. For the settings, we set p = 1, q = 2 and γ = 2.

5
Conclusion and Future Work

In this work, we studied the generalization properties of over-parameterized gradient descent in the
context of sequence models. We showed that the over-parameterization method can adapt to the
underlying structure of the signal and significantly outperform the vanilla fixed-eigenvalues method.
These results provide a new perspective on the benefits of over-parameterization and offer insights
into the adaptivity and generalization properties of neural networks beyond the kernel regime.

However, there are also limitations of this work and many interesting directions for future research.
For example, one can directly consider the over-parameterization in the kernel regression by replacing
the feature map Φ(x) = (λ1/2
j
ej(x))j≥1 with the learnable one Φ(x; a) = (ajej(x))j≥1, where

aj’s are learnable parameters initialized by aj(0) = λ1/2
j
. However, the analysis would be more
challenging since now the components are mutually coupled in the gradient flow dynamics.

Perhaps one of the most interesting directions is to study how the over-parameterization method
can also learn the eigenfunctions of the kernel during the training process, which leads to the truly
“adaptive kernel method”. We believe that future studies on this topic will provide a deeper theoretical
understanding of the success of neural networks in practice.

10


---Page Break---
Acknowledgments and Disclosure of Funding

Qian Lin’s research was supported in part by the National Natural Science Foundation of China
(Grant 92370122, Grant 11971257).

We thank the anonymous reviewers and area chairs for their valuable comments and suggestions.
Their feedback helped us improve the quality of the paper.

References

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-
parameterization, June 2019. URL http://arxiv.org/abs/1811.03962.

Ingo Steinwart (auth.) Andreas Christmann. Support Vector Machines. Information Science and
Statistics. Springer-Verlag New York, New York, NY, 1 edition, 2008. ISBN 0-387-77242-1
0-387-77241-3 978-0-387-77241-7 978-0-387-77242-4. doi: 10.1007/978-0-387-77242-4.

Sanjeev
Arora,
Nadav
Cohen,
Wei
Hu,
and
Yuping
Luo.
Implicit
regularization
in
deep
matrix
factorization.
Advances
in
Neural
Information
Processing
Sys-
tems,
32,
2019a.
URL
https://proceedings.neurips.cc/paper/2019/hash/
c0c783b5fc0d7d808f1d14a6e9c8280d-Abstract.html.

Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of
optimization and generalization for overparameterized two-layer neural networks. In International
Conference on Machine Learning, pages 322–332. PMLR, 2019b.

Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang.
On exact computation with an infinitely wide neural net. In Advances in Neural Information
Processing Systems, volume 32. Curran Associates, Inc., 2019c. URL https://proceedings.
neurips.cc/paper/2019/hash/dbc4d84bfcfe2284ba11beffb853a8c4-Abstract.html.

Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. High-
dimensional asymptotics of feature learning: How one gradient step improves the representation,
May 2022.

Frank Bauer, Sergei Pereverzev, and Lorenzo Rosasco. On regularization algorithms in learning
theory. Journal of complexity, 23(1):52–72, 2007. doi: 10.1016/j.jco.2006.07.001.

Alberto Bietti and Francis Bach. Deep equals shallow for relu networks in kernel regimes. arXiv
preprint arXiv:2009.14397, 2020.

Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in
kernel regression and wide neural networks. In Proceedings of the 37th International Conference
on Machine Learning, pages 1024–1034. PMLR, November 2020. URL https://proceedings.
mlr.press/v119/bordelon20a.html.

Lawrence D. Brown, T. Tony Cai, Mark G. Low, and Cun-Hui Zhang. Asymptotic equivalence theory
for nonparametric regression with random design. The Annals of Statistics, 30(3):688–707, 2002.
ISSN 0090-5364. doi: 10.1214/aos/1028674838.

Andrea Caponnetto and Ernesto De Vito.
Optimal rates for the regularized least-squares al-
gorithm.
Foundations of Computational Mathematics, 7(3):331–368, 2007.
doi: 10.1007/
s10208-006-0196-8.

Yunlu Chen, Yang Li, Keli Liu, and Feng Ruan. Kernel learning in ridge regression "automatically"
yields exact low rank solution, October 2023.

Hung-Hsu Chou, Carsten Gieshoff, Johannes Maly, and Holger Rauhut. Gradient descent for
deep matrix factorization: Dynamics and implicit bias towards low rank, August 2023. URL
http://arxiv.org/abs/2011.13772.

Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová. Generalization error rates
in kernel regression: The crossover from the noiseless to noisy regime. Advances in Neural
Information Processing Systems, 34:10131–10143, 2021.

11


---Page Break---
Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes
over-parameterized neural networks. arXiv preprint arXiv:1810.02054, 2018.

Jianqing Fan, Zhuoran Yang, and Mengxin Yu. Understanding implicit regularization in over-
parameterized single index model, November 2021.
URL http://arxiv.org/abs/2007.
08322.

Simon-Raphael Fischer and Ingo Steinwart. Sobolev norm learning rates for regularized least-
squares algorithms. Journal of Machine Learning Research, 21:205:1–205:38, 2020. URL https:
//www.semanticscholar.org/paper/248fb62f75dac19f02f683cecc2bf4929f3fcf6d.

Khashayar Gatmiry, Stefanie Jegelka, and Jonathan Kelner. Optimization and Adaptive Generalization
of Three layer Neural Networks. In International Conference on Learning Representations, October
2021. URL https://openreview.net/forum?id=dPyRNUlttBv.

Amnon Geifman, Abhay Yadav, Yoni Kasten, Meirav Galun, David Jacobs, and Basri Ronen. On
the similarity between the Laplace and neural tangent kernels. In Advances in Neural Information
Processing Systems, volume 33, pages 1451–1461, 2020.

S. Gunasekar, B. Woodworth, S. Bhojanapalli, B. Neyshabur, and N. Srebro. Implicit regularization
in matrix factorization. In Advances in Neural Information Processing Systems, volume 2017-
December, pages 6152–6160, 2017.

Peter D. Hoff. Lasso, fractional norm and structured sparse estimation using a Hadamard product
parametrization. Computational Statistics & Data Analysis, 115:186–198, November 2017. ISSN
0167-9473. doi: 10.1016/j.csda.2017.06.007.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-
Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31.
Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/file/
5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf.

Iain M. Johnstone. Gaussian estimation: Sequence and wavelet models. 2017.

Chris Kolb, Christian L. Müller, Bernd Bischl, and David Rügamer. Smoothing the edges: A general
framework for smooth optimization in sparse regularization using hadamard overparametrization.
arXiv preprint arXiv:2307.03571, 2023.

Masayoshi Kubo, Ryotaro Banno, Hidetaka Manabe, and Masataka Minoji. Implicit Regularization
in Over-parameterized Neural Networks, March 2019. URL http://arxiv.org/abs/1903.
01997.

Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-
Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models
under gradient descent. In Advances in Neural Information Processing Systems, volume 32.
Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/hash/
0d1a9651497a38d8b1c3871c84528bd4-Abstract.html.

Daniel LeJeune and Sina Alemohammad. An adaptive tangent feature perspective of neural networks,
August 2023. URL http://arxiv.org/abs/2308.15478.

Jiangyuan Li, Thanh V. Nguyen, Chinmay Hegde, and Raymond K. W. Wong. Implicit sparse
regularization: The impact of depth and early stopping, October 2021a. URL http://arxiv.
org/abs/2108.05574.

Yicheng Li, Haobo Zhang, and Qian Lin. On the asymptotic learning curves of kernel ridge regression
under power-law decay. In Thirty-Seventh Conference on Neural Information Processing Systems,
2023a. URL https://openreview.net/forum?id=E4P5kVSKlT.

Yicheng Li, Haobo Zhang, and Qian Lin. On the saturation effect of kernel ridge regression.
In International Conference on Learning Representations, February 2023b. URL https://
openreview.net/forum?id=tFvr-kYWs_Y.

12


---Page Break---
Yicheng Li, Weiye Gan, Zuoqiang Shi, and Qian Lin. Generalization error curves for analytic spectral
algorithms under power-law decay, January 2024a. URL http://arxiv.org/abs/2401.01599.

Yicheng Li, Zixiong Yu, Guhan Chen, and Qian Lin. On the eigenvalue decay rates of a class
of neural-network related kernel functions defined on general domains. Journal of Machine
Learning Research, 25(82):1–47, 2024b. ISSN 1533-7928. URL http://jmlr.org/papers/
v25/23-0866.html.

Zhiyuan Li, Yuping Luo, and Kaifeng Lyu. Towards resolving the implicit bias of gradient descent
for matrix factorization: Greedy low-rank learning, April 2021b. URL http://arxiv.org/abs/
2012.09839.

Zhiyuan Li, Tianhao Wang, and Sanjeev Arora. What Happens after SGD Reaches Zero Loss? –A
Mathematical Framework, July 2022. URL http://arxiv.org/abs/2110.06914.

Junhong Lin, Alessandro Rudi, L. Rosasco, and V. Cevher. Optimal rates for spectral algorithms with
least-squares regression over Hilbert spaces. Applied and Computational Harmonic Analysis, 48:
868–890, 2018. doi: 10.1016/j.acha.2018.09.009.

Neil Mallinar, James B. Simon, Amirhesam Abedsoltan, Parthe Pandit, Mikhail Belkin, and Preetum
Nakkiran. Benign, tempered, or catastrophic: A taxonomy of overfitting, July 2022.

Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, and Daniel Soudry. Implicit bias of the
step size in linear diagonal neural networks. In International Conference on Machine Learning,
pages 16270–16295. PMLR, 2022.

Noam Razin, Asaf Maman, and Nadav Cohen. Implicit Regularization in Tensor Factorization, June
2021. URL http://arxiv.org/abs/2102.09972.

Mariia Seleznova and Gitta Kutyniok. Analyzing finite neural networks: Can we trust neural tangent
kernel theory? In Mathematical and Scientific Machine Learning, pages 868–895. PMLR, 2022.
URL https://proceedings.mlr.press/v145/seleznova22a.html.

Ingo Steinwart and C. Scovel. Mercer’s Theorem on General Domains: On the Interaction between
Measures, Kernels, and RKHSs. 2012. doi: 10.1007/S00365-012-9153-3.

Ingo Steinwart, Don R Hush, Clint Scovel, et al. Optimal Rates for Regularized Least Squares Re-
gression. In COLT, pages 79–93, 2009. URL http://www.learningtheory.org/colt2009/
papers/038.pdf.

Tomas Vaškeviˇcius, Varun Kanade, and Patrick Rebeschini. Implicit regularization for optimal sparse
recovery, September 2019. URL http://arxiv.org/abs/1909.05122.

Loucas Pillaud Vivien, Julien Reygner, and Nicolas Flammarion. Label noise (stochastic) gradient
descent implicitly solves the Lasso for quadratic parametrisation.
In Proceedings of Thirty
Fifth Conference on Learning Theory, pages 2127–2159. PMLR, June 2022. URL https://
proceedings.mlr.press/v178/vivien22a.html.

Jonathan Wenger, Felix Dangel, and Agustinus Kristiadi. On the disconnect between theory and
practice of overparametrized neural networks, September 2023. URL http://arxiv.org/abs/
2310.00137.

Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan,
Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In
Proceedings of Thirty Third Conference on Learning Theory, pages 3635–3673. PMLR, July 2020.
URL https://proceedings.mlr.press/v125/woodworth20a.html.

Greg Yang and Edward J. Hu. Feature learning in infinite-width neural networks, July 2022.

Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning.
Constructive Approximation, 26:289–315, August 2007. doi: 10.1007/s00365-006-0663-2.

Chulhee Yun, Shankar Krishnan, and Hossein Mobahi. A unifying view on implicit bias in training
linear neural networks, September 2021. URL http://arxiv.org/abs/2010.02501.

13


---Page Break---
Haobo Zhang, Yicheng Li, and Qian Lin. On the optimality of misspecified spectral algorithms,
March 2023.

Peng Zhao, Yun Yang, and Qiao-Chu He. High-dimensional linear regression via implicit regu-
larization. Biometrika, 109(4):1033–1046, November 2022. ISSN 0006-3444, 1464-3510. doi:
10.1093/biomet/asac010.

14


---Page Break---
Contents

1
Introduction
1

1.1
Our contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

1.2
Notations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

2
Limitations of Fixed Kernel Regression
2

3
Adapting the Eigenvalues by Over-parameterization in the Sequence Model
5

3.1
Over-parameterized gradient descent . . . . . . . . . . . . . . . . . . . . . . . . .
5

3.2
Towards deeper over-parameterization . . . . . . . . . . . . . . . . . . . . . . . .
6

3.3
Discussion of the results
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
7

3.4
Proof outline
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

4
Numerical Experiments
9

5
Conclusion and Future Work
10

References
11

A Additional related works
17

B
Supplementary Technical Details
18

B.1
The connection between RKHS and the sequence model
. . . . . . . . . . . . . .
18

B.2
The examples in Section 2
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

B.2.1
Example 2.1
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

B.2.2
Example 2.2
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

B.2.3
Example 2.3
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

C Detailed Numerical Experiments
21

C.1
Experiments beyond the sequence model . . . . . . . . . . . . . . . . . . . . . . .
22

C.2
Testing eigenvalue misalignment on real-world data . . . . . . . . . . . . . . . . .
22

D Analysis of the gradient flow
26

D.1 The two-layer parameterization . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

D.2
Deeper parameterization
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27

E
Main proofs
31

E.1
Proof of Theorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31

E.2
Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

E.3
The absolute error term . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35

E.4
Proof of Proposition 3.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35

F
Auxiliary results
37

15


---Page Break---
NeurIPS Paper Checklist
38

16


---Page Break---
A
Additional related works

In this section, we will provide additional related works and further discussions.

Regression with fixed kernel
The regression problem with a fixed kernel has been well studied
in the literature. It has been shown that with proper regularization, kernel methods can achieve the
minimax optimal rates under various conditions [Caponnetto and De Vito, 2007, Steinwart et al.,
2009, Lin et al., 2018, Fischer and Steinwart, 2020, Zhang et al., 2023]. Recently, a sequence of
works provided more refined results on the generalization error of kernel methods [Li et al., 2023b,
Bordelon et al., 2020, Cui et al., 2021, Mallinar et al., 2022, Li et al., 2023a, 2024a].

The NTK regime of neural networks
Over-parameterized neural networks are connected to kernel
methods through the neural tangent kernel (NTK) theory proposed by Jacot et al. [2018], which
shows that the dynamics of the neural network at infinite width limited can be approximated by a
kernel method with respect to the corresponding NTK. The theory was further developed by many
follow-up works [Arora et al., 2019c,b, Du et al., 2018, Lee et al., 2019, Allen-Zhu et al., 2019]. Also,
the properties on the corresponding NTK have also been studied [Geifman et al., 2020, Bietti and
Bach, 2020, Li et al., 2024b].

Over-parameterization as Implicit Regularization
There has been a surge of interest in under-
standing the role of over-parameterization in deep learning. One perspective is that over-parameterized
models trained by gradient-based methods can expose certain implicit bias towards simple solutions,
which include linear models [Hoff, 2017, Vaškeviˇcius et al., 2019, Zhao et al., 2022, Li et al., 2021a],
matrix factorization [Gunasekar et al., 2017, Arora et al., 2019a, Li et al., 2021b, Razin et al., 2021,
Chou et al., 2023], linear networks [Yun et al., 2021, Nacson et al., 2022] and neural networks [Kubo
et al., 2019, Woodworth et al., 2020]. Moreover, variants of gradient descent such as stochastic
gradient descent are also shown to have implicit regularization effects [Li et al., 2022, Vivien et al.,
2022]. However, most of these works focus only on the optimization process and the final solution,
but the generalization performance is not well understood.

Generalization Guarantees for Over-parameterized Models
Being the most related to our work,
a few works provided generalization guarantees for over-parameterized models, which only include
linear models [Zhao et al., 2022, Li et al., 2021a, Vaškeviˇcius et al., 2019] and single index model [Fan
et al., 2021]. In detail, the two parallel works [Zhao et al., 2022, Vaškeviˇcius et al., 2019] studied
the high-dimensional linear regression problem under sparse settings and showed that a two-layer
diagonal over-parameterized model with proper initialization and early stopping can achieve minimax
optimal recovery under certain conditions. The subsequent work [Li et al., 2021a] obtained similar
results for multi-layer diagonal over-parameterized models.

The adaptive kernel perspective
The idea of an adaptive kernel has appeared in a few recent
works in various forms [Chen et al., 2023, Gatmiry et al., 2021, LeJeune and Alemohammad, 2023,
Yang and Hu, 2022, Ba et al., 2022], which is also known as “feature learning”. Notably, Gatmiry
et al. [2021] showed the benefits brought by the adaptivity of the kernel on a three-layer neural
network, which is similar to our work in the adaptive kernel perspective. However, our work and
theirs consider different aspects of the adaptive kernel: while they considered an adaptive kernel
space in the form of G⊙K∞around the NTK space, we consider an eigenvalue-parameterized kernel
space with fixed eigen-basis. We believe that these various results, including ours, will contribute to a
better understanding of the generalization properties of over-parameterized models as well as neural
networks.

17


---Page Break---
B
Supplementary Technical Details

B.1
The connection between RKHS and the sequence model

Diagonalized kernel gradient flow as sequence model
Moreover, this connection can also be seen
directly from the following. The Mercer’s decomposition of the RKHS H associated with the kernel
k also provides a series representation of the RKHS:

H =






∞
X

j=1
ajλ

1
2
j ei
 (aj)j≥1 ∈ℓ2




,
(19)

where we denote by ℓ2 =
n
(aj)j≥1 | P∞
j=1 a2
j < ∞
o
. Therefore, by introducing the feature

mapping Φ : X →ℓ2 defined by

Φ(x) = (λ

1
2
j ej(x))j≥1,
(20)

we establish a one-to-one correspondence between a function f ∈H in the RKHS and a vector
β ∈ℓ2 in feature space via f(x) = ⟨Φ(x), β⟩ℓ2 for f ∈H and β ∈ℓ2. Moreover, it is convenient to
consider Φ(x) as column vectors and also define the feature matrix Φ(X) = (Φ(x1), . . . , Φ(xn)) for
X = (x1, . . . , xn). Then, the gradient flow (2) in the feature space ℓ2 writes

˙β(t) = −∇βL = −1

nΦ(X)Φ(X)⊤β(t) + 1

nΦ(X)y.
(21)

Intuitively, since the j, l-th entry

 1

nΦ(X)Φ(X)⊤


j,l
=
λ

1
2
j λ

1
2
l
n

n
X

i=1
ej(xi)el(xi)

the law of large numbers implies that 1

nΦ(X)Φ(X)⊤converges to the diagonal operator Λ =
diag(λ1, λ2, . . . ); moreover, since j-th entry

 1

nΦ(X)y


j
=
λ

1
2
j
n

n
X

i=1
f(xi)ej(xi) +
λ

1
2
j
n

n
X

i=1
ej(xi)εi,

the central limit theorem suggest that it can be approximated by λ

1
2
j zj, where zj = θj + ξj and ξj is
a normal random variable with mean zero and variance σ2/n. Therefore, with these approximations,
the equation can be diagonalized as

˙βj(t) = −λjβj + λ

1
2
j zj = −λj(βj(t) −λ
−1

2
j
zj).

Moreover, we can rewrite the representation f(x) = ⟨Φ(x), β⟩ℓ2 into

f =

∞
X

j=1
λ

1
2
j βjej =

∞
X

j=1
θjej,
θj = λ

1
2
j βj.

Then, in terms of θ, we have

˙βj(t) = λ

1
2
j ˙βj(t) = −λj(θj −zj).

This is exactly the vanilla gradient flow in the sequence model in Section 3.

Furthermore, we can consider the parameterized feature map

Φa(x) = (ajej(x))j≥1 ,

where a = (aj)j≥1. We can consider similar gradient flow in the feature space with both β and a
trainable. Then, with similar diagonalizing argument, we can show that the corresponding gradient
flow in the sequence model is just the over-parameterized gradient flow in (7). Similar correspondence
can be established for the multi-layer models (13).

18


---Page Break---
B.2
The examples in Section 2

B.2.1
Example 2.1

The deduction in Example 2.1 is straightforward. The series expansion (3) can be written as

[Hk]s =




f =
X

j≥1
fjej |
X

j≥1
f 2
j λ−s
j
< ∞




.

We recall that λj,1 ≍j−γ1 and λj,2 ≍j−γ2. Let f ∗= P

j≥1 f ∗
j ej. Then, for ℓ= 1, 2,

f ∗∈[Hkℓ]sℓ⇐⇒
X

j≥1
(f ∗
j )2λ−sℓ
j,1 < ∞⇐⇒
X

j≥1
(f ∗
j )2jγℓsℓ< ∞

Consequently,

f ∗∈[Hk1]s1 ⇐⇒f ∗∈[Hk2]s2
for
γ1s1 = γ2s2.

B.2.2
Example 2.2

We justify the claims in Example 2.2. Let us consider the torus Td = [−1, 1)d and the uniform
measure µ on Td (namely µ(Td) = 1). Let us recall that the multidimensional Fourier basis is given
by ϕm(x) = exp(iπ ⟨m, x⟩) for m ∈Zd.

The Sobolev space Hs(Td) is defined via the Fourier coefficients as

Hs(Td) =




f ∈L2(Td) |
X

m∈Zd
|fm|2(1 + ∥m∥2)s < ∞




,

which is equipped with the inner product (as thus the induced norm)

⟨f, g⟩Hs(Td) =
X

m∈Zd
fmgm(1 + ∥m∥2)s.

Now, we briefly show that Hs(Td) is an RKHS when s > d/2. It suffices to show that f 7→f(x) is
bounded for each x ∈Td [Andreas Christmann, 2008]. Using the boundedness of ϕm, we have

X

m∈Zd
|fmϕm| ≤
X

m∈Zd
|fm| ≤



X

m∈Zd
|fm|2(1 + ∥m∥2)s





1
2 

X

m∈Zd
(1 + ∥m∥2)−s





1
2

Now,

X

m∈Zd
(1 + ∥m∥2)−s ≲
Z

x∈Rd(1 + |x|2)−sdx ≲
Z ∞

0
(1 + r2)−srd−1dr.

Since the last integral is finite when s > d/2, we find that there is a constant C such that
X

m∈Zd
|fmϕm| ≤C∥f∥Hs(Td).

Therefore, the series expansion f(x) = P

m∈Zd fmϕm(x) converges absolutely and uniformly,
and thus |f(x)| ≤P

m∈Zd |fmϕm| ≤C∥f∥Hs(Td), showing that f 7→f(x) is bounded for each
x ∈Td.

Moreover, it is easy to see from the Mercer’s decomposition (1) and the power series expansion
(3) that the kernel of Hs(Td) is given by k(x, x′) = P

m∈Zd(1 + ∥m∥2)−sϕm(x)ϕm(x′), so its
eigenvalues are λm = (1 + ∥m∥2)−s.

19


---Page Break---
To determine the eigenvalue decay rate of λm = (1 + ∥m∥2)−s after reordering them in decreas-
ing order, it suffices to determine the count # {m : λm > δ}: the eigenvalue decay rate is β if
# {m : λm > δ} ≍δ−1/β, see, e.g., Proposition A.1 in Li et al. [2024a]. We have

# {m : λm > δ} = #
n
m : (1 + ∥m∥2)−s > δ
o

≍Vol(
n
x ∈Rd : (1 + |x|2)−s > δ
o
)

≍Vol(
n
x ∈Rd : |x| < δ−1

2s
o
) ≍δ−d

2s .

We consider a function f ∗(x) = g(x1, . . . , xd0) with low-dimensional structure. Let us denote by
x≤d0 = (x1, . . . , xd0) and x>d0 = (xd0+1, . . . , xd) for simplicity. Then, the Fourier coefficients of
f ∗are given by

fm = ⟨f ∗, ϕm⟩L2(T,dµ)

= 2−d
Z

Td0×Td−d0
g(x1, . . . , xd0) exp(iπ ⟨m≤d0, x≤d0⟩) · exp(iπ ⟨m>d0, x>d0⟩)dx≤d0dx>d0

= 2−d0
Z

Td0
g(x1, . . . , xd0) exp(iπ ⟨m≤d0, x≤d0⟩)dx≤d0

· 2−(d−d0)
Z

Td−d0
exp(iπ ⟨m>d0, x>d0⟩)dx>d0

= gm≤d0 · 1{m>d0=0},

so we show that

fm =
gm≤d0,
m = (m≤d0, 0), m≤d0 ∈Zd0,
0,
otherwise.

We recall that g ∈Ht(Td0), so
X

m≤d0∈Zd0

gm≤d0


2
(1 + ∥m≤d0∥2)t < ∞.

To determine the smoothness of f ∗on [Hk1]s and [Hk2]s, following (3), we compute
X

m∈Zd
|fm|2 h
(1 + ∥m∥2)ris
=
X

m=(m≤d0,0),m≤d0∈Zd0

gm≤d0


2 h
(1 + ∥m≤d0∥2)ris

X

m≤d0∈Zd0

gm≤d0


2
(1 + ∥m≤d0∥2)rs,

so f ∗belongs to [Hk1]s and [Hk2]s for s = t/r

B.2.3
Example 2.3

We recall that f ∗= P

j≥1 θ∗
j ej,
θ∗
l(j)
 ≍j−(p+1)/2
and
ℓ(j) ≍jq
for
p > 0, q ≥1,

where ℓ(j) gives the descending order of
θ∗
j
. To compute the relative smoothness of f ∗w.r.t.
λj,1 ≍j−γ, we compute
X

j≥1

θ∗
j
2λ−s
j
=
X

j≥1

θ∗
l(j)

2
λ−s
l(j) ≍
X

j≥1
j−(p+1)(ℓ(j))γs ≍
X

j≥1
j−(p+1)jqγs ≍
X

j≥1
j−1−(p−qγs),

so we have s < p/(qγ) (but arbitrarily close) and the corresponding generalization error rate is
sγ
sγ+1 <
p
p+q. The generalization error rate w.r.t. λl(j),2 ≍j−γ can be computed similarly.

20


---Page Break---
C
Detailed Numerical Experiments

In this section, we provide numerical experiments to validate the theoretical results. The codes are
provided in the supplementary material. We approximate the gradient flow equation (22) and (30) by
discrete-time gradient descent with sufficiently small step size. Moreover, we truncate the sequence
model to the first N terms for some very large N. We consider the settings as in Corollary 3.3 that
θ∗is given by (4) for some p > 0 and q ≥1. We set ϵ2 = n−1, where n can be regarded as the
sample size, and consider the asymptotic performance of the generalization error as n grows. For the
stopping time, we choose the oracle one that minimizes the generalization error for each method. We
first compare the generalization error rates between vanilla gradient descent and over-parameterized
gradient descent (OpGD) in Figure 3 on page 21. The results show that the over-parameterized
gradient descent can achieve the optimal generalization error rate, while the vanilla gradient descent
suffers from the misalignment caused by q > 1 and thus has a slower convergence rate. Moreover,
with a logarithmic least-squares fitting, we find that the resulting generalization error rates are
consistent with the theoretical results in Corollary 3.3 (0.5 for OpGD and 0.33 for vanilla GD); the
oracle stopping times for the over-parameterized gradient descent also match the theoretical value
(0.5).

2000
10000
n (logarithmic)

1.5

1.4

1.3

1.2

1.1

1.0

0.9

log10  generalization error 

Generalization Errors; optimal rate=0.500

Vanilla GD
log Err
0.374log n + 0.29

OpGD
log Err
0.533log n + 0.65

2000
10000
n (logarithmic)

2.0

2.5

3.0

3.5

log10 min gen step

Oracle stopping time

Vanilla GD
log t
1.03log n +
0.53

OpGD
log t
0.56log n +
0.21

2000
10000
n (logarithmic)

1.5

1.4

1.3

1.2

1.1

1.0

0.9

log10  generalization error 

Generalization Errors; optimal rate=0.500

Vanilla GD
log Err
0.341log n + 0.21

OpGD
log Err
0.499log n + 0.55

2000
10000
n (logarithmic)

2

3

4

5

6

log10 min gen step

Oracle stopping time

Vanilla GD
log t
1.17log n + 2.02

OpGD
log t
0.51log n + 0.26

Figure 3: Comparison of the generalization error rates between vanilla gradient descent and over-
parameterized gradient descent (OpGD). We set p = 1 and q = 2 for the truth parameter θ∗. The
left and right columns show respectively the generalization error and the orcale stopping time with
respect to n. For the upper row, we set the eigenvalue decay rate γ = 1.5; for the lower row, we set
γ = 3. For each n, we repeat 64 times and plot the mean and the standard deviation.

Furthermore, we investigate the generalization performance of over-parameterized gradient descent
(also with deeper parameterization) for different settings of the truth parameter θ∗, the eigenvalue
decay rate γ and the depth D. The results are reported in Table C on page 22. We find that the
generalization error rates are in general consistent with the theoretical results in Corollary 3.3, while

21


---Page Break---
p = 0.6 (r∗= 0.37)
p = 1 (r∗= 0.5)
p = 3 (r∗= 0.75)
γ
q = 1
q = 1.5
q = 2
q = 1
q = 1.5
q = 2
q = 1
q = 1.5
q = 2
1.1
0.38
0.40
0.50
0.50
0.45
0.48
0.78
0.69
0.67
2
0.36
0.41
0.52
0.49
0.46
0.50
0.80
0.73
0.72
3
0.36
0.41
0.52
0.48
0.46
0.50
0.76
0.73
0.74
Table 1: Convergence rates of the over-parameterized gradient descent (8) under different settings of
the truth parameter p, q and the eigenvalue decay rate γ, where r∗is the ideal convergence rate. The
convergence rate is estimated by the logarithmic least-squares fitting of the generalization error with
n ranging from 2000, 2200, . . . , 4000, where the generalization error is the mean of 256 repetitions.

p = 0.6 (r∗= 0.37)
p = 1 (r∗= 0.5)
p = 3 (r∗= 0.75)
γ
q = 1
q = 1.5
q = 2
q = 1
q = 1.5
q = 2
q = 1
q = 1.5
q = 2

D = 1

1.1
0.36
0.40
0.52
0.49
0.46
0.49
0.79
0.73
0.72
2
0.36
0.40
0.52
0.48
0.46
0.50
0.76
0.73
0.73
3
0.30
0.32
0.38
0.45
0.41
0.44
0.75
0.73
0.74

D = 2

1.1
0.34
0.39
0.49
0.46
0.44
0.48
0.76
0.74
0.75
2
0.35
0.40
0.51
0.47
0.45
0.49
0.74
0.73
0.73
3
0.36
0.40
0.51
0.48
0.46
0.50
0.74
0.73
0.73
Table 2: Convergence rates of the over-parameterized gradient descent (14) with D = 1 and D = 2.
The settings are the same as in Table C on page 22.

there are some fluctuations due to the finite sample size. Comparing the results for γ = 1.1 across
depth D = 0, D = 1 and D = 2, we see that the method with D = 0 has the slowest convergence rate,
while the method with D = 2 has the fastest convergence rate, justifying that deeper parameterization
can improve the generalization performance. In summary, the numerical experiments validate our
theoretical results.

Additionally, we investigate the evolution of the eigenvalue terms aj(t)bD
j (t) over time t as discussed
in Proposition 3.4. The results are shown in Figure 4 on page 23. We find that the eigenvalue terms
can indeed adapt to the underlying structure of the signal: for large signals, the eigenvalue terms
approach the signals as the training progresses, while for small signals, the eigenvalue terms do not
increase significantly. Moreover, we find that deeper over-parameterization reduces the fluctuations
of the eigenvalue terms for the noise components, and thus improves the generalization performance
of the model.

C.1
Experiments beyond the sequence model

We also explore the adaptivity of the over-parameterized gradient descent beyond the sequence
model. Let us consider the diagonal adaptive kernel method by parameterizing the feature map
with Φ(x; a) = (ajej(x))j≥1 introduced in Section 5. We use the setting in Example 2.2 where
the eigenfunctions are the trigonometric functions. In particular, we consider the truth function
f ∗(x) = sin(7.5πx1) with x = (x1, x2) ∈R2. We present the generalization error curve of a single
trial and also the generalization error rates with respect to the sample size n in Figure 5 on page 24.
The result also shows the benefit of over-parameterization in adapting to the underlying structure of
the signal.

C.2
Testing eigenvalue misalignment on real-world data

In this section, we provide additional experiments to test the eigenvalue misalignment phenomenon
on real-world data. Recalling Example 2.2 and Example 2.3, we know that the misalignment happens
when the order of the eigenvalues of the kernel mismatches the order of coefficients of the truth
function. Therefore, to test the misalignment, we compute the coefficients of the regression function
over the eigen-basis of the kernel and examine whether the coefficients decay in the order given by the
kernel. For the eigen-basis, we use the multidimensional Fourier basis (the trigonometric functions)
considered in Example 2.2, where the order is given by the descending order of λm = (1+∥m∥2)−r.

22


---Page Break---
0
50
100
150
200
Time t

10
1

100

log10 error

D=0

0
200
400
600
800
1000
Time t

D=1

0
500
1000
1500
2000
Time t

D=3

Generalization error

100
120
140
160
180
200
index j

2.25

2.00

1.75

1.50

1.25

1.00

0.75

0.50

log10 magnitude

t=0

100
120
140
160
180
200
index j

t=50

100
120
140
160
180
200
index j

t=100

Model 
= a  (D = 0)

100
120
140
160
180
200
index j

2.5

2.0

1.5

1.0

log10 magnitude

t=0

100
120
140
160
180
200
index j

t=150

100
120
140
160
180
200
index j

t=300

Model 
= ab  (D = 1)

100
120
140
160
180
200
index j

3.0

2.5

2.0

1.5

1.0

log10 magnitude

t=0

100
120
140
160
180
200
index j

t=350

100
120
140
160
180
200
index j

t=700

Model 
= ab3  (D = 3)

Figure 4:
The generalization error as well as the evolution of the eigenvalue terms aj(t)bD
j (t)
over the time t. The first row shows the generalization error of three parameterizations D = 0, 1, 3
with respect to the training time t. The rest of the rows show the evolution of the eigenvalue terms
aj(t)bD
j (t) over the time t. For presentation, we select the index j = 100 to 200. The blue line
shows the eigenvalue terms and the black marks show the non-zero signals scaled according to
Proposition 3.4. For the settings, we set p = 1, q = 2 and γ = 2.

23


---Page Break---
100
101
102
103
104

Training Time (log)

10
1

Test Error (log)

Generalization error vs. training time

Fixed kernel
Diag adapt
Diag adapt (D=1)
Diag adapt (D=2)

160
190
220
250
280
310
340
370
400
n (logarithmic)

1.4

1.2

1.0

0.8

0.6

0.4

log10  generalization error 

Generalization Error vs. n

Fixed kernel
log Err
0.31log n + 0.40

Diag adapt
log Err
0.65log n + 0.27

Figure 5: Comparison of the generalization error between the fixed kernel gradient method and the
diagonal adaptive kernel method. The left figure shows the generalization error curve of a single trial.
The right figure shows the generalization error rates with respect to the sample size n.

We consider the two real-world datasets: “California Housing” and “Concrete Compressive Strength”.
We compute the empirical inner product of the regression function with the Fourier basis functions
up to a certain order. Then, we plot the coefficients in the order given by the kernel. The results are
shown in Figure 6 on page 25. The figures show that the empirical coefficients exhibit significant
spikes. Also, among the coefficients, only very few components have large magnitudes, indicating
the sparse structure of the regression function. Together, these results suggest that the eigenvalues of
the kernel are misaligned with the truth function in these datasets.

24


---Page Break---
0.0

0.5

1.0

1.5

2.0

Fourier Coefficients in the Kernel's Order

0
20000 40000 60000 80000 100000120000140000
0.0

0.5

1.0

1.5

2.0

Fourier Coefficients in the Descending Order

0

10

20

30

Fourier Coefficients in the Kernel's Order

0
2000
4000
6000
8000
10000 12000 14000

10

20

30

Fourier Coefficients in the Descending Order

Figure 6: The empirical coefficients of the regression function over the Fourier basis for the "California
Housing" dataset (upper) and "Concrete Compressive Strength" dataset (lower). Note that we take
different numbers of Fourier basis functions for the two datasets for better visualization.

25


---Page Break---
D
Analysis of the gradient flow

D.1
The two-layer parameterization

Let us consider the gradient flow considered in (8), where we remove the subscript j for notational
simplicity. Let L = 1

2(θ −z)2 and θ = aβ. We are interested in the gradient flow:

˙a = −∇aL = β(z −θ),
˙β = −∇βL = a(z −θ),

a(0) = λ
1
2 > 0,
β(0) = 0.

(22)

Symmetry of the solution
We can find that the solution of the equation for z < 0 can be obtained
by simply a(t), −β(t) for the positive signal case of −z > 0. Therefore, we only need to consider the
case of z > 0. In this case, it is obvious that a(t), β(t) and θ(t) are all non-negative and increasing.

Gradient flow of θ
Now we notice that
1
2
d
dta2 = 1

2
d
dtβ2 = aβ(z −θ),

so

a2(t) −β2(t) ≡a2(0) −β2(0) = λ.
(23)

Using this conservation quantity, we can prove the following estimations:

θ = aβ =
p

λ + β2 · β ≥β2,

θ = aβ = a
p

a2 −λ ≤a2.
(24)

Moreover, the derivative of θ writes
˙θ = a ˙β + ˙aβ = (a2 + β2)(z −θ).

Using a2 + β2 =
p

(a2 −β2)2 + 4a2β2 =
√

λ2 + 4θ2, we conclude the follow explicit equation
for θ:
˙θ =
p

λ2 + 4θ2(z −θ).
(25)

Then, we have the following approximation of the solution.
Lemma D.1. Let us consider the gradient flow (25) and

d
dt
˜θ = (λ + 2|˜θ|)(z −˜θ),
˜θ(0) = 0.
(26)

Then we have

0 ≤˜θ(t/
√

2) ≤θ(t) ≤˜θ(t) ≤z
if
z ≥0;

0 ≥˜θ(t/
√

2) ≥θ(t) ≥˜θ(t) ≥z
if
z ≤0.
(27)

Moreover, (26) is solved by

˜θ(t) = λ(E −1)

2|z| + λE z,
E = exp((2|z| + λ)t).
(28)

Proof. It suffices to consider the case z ≥0. It is easy to see from the gradient flow (22) that both
a(t), β(t) are non-negative. Then, using the elementary inequality

1
√

2(λ + 2x) ≤
p

λ2 + 4x2 ≤λ + 2x,

we have
1
√

2(λ + 2x)(z −x) ≤
p

λ2 + 4x2(z −x) ≤(λ + 2x)(z −x).

Then, the comparison principal in ordinary differential equation yields (27). The verification of (28)
is straightforward.

26


---Page Break---
D.2
Deeper parameterization

Now let us consider the deeper parameterization of the form

θ = abDβ,
(29)

where a, b, β are all trainable parameters, and the gradient flow

˙a = −∇aL = bDβ(z −θ),
˙b = −∇bL = DabD−1β(z −θ),
˙β = −∇βL = abD(z −θ),

a(0) = λ
1
2 > 0,
b(0) = b0 > 0,
β(0) = 0.

(30)

Main idea
We provide the main idea of analyzing the gradient flow (30) here. Using the conser-
vation quantities, we can first focus on the equation of β. For the initial stage when t is relatively
small, β(t) only grows linearly in t. Next, when β(t) exceeds a certain threshold depending on the
initialization (and also the interplay between λj and b0), β(t) grows exponentially in t, provided that
θ(t) ≤z/2. Thus, we can upper bound the hitting time of θ(t) to z/2. Finally, when θ(t) ≥z/2, we
consider directly the equation of θ and show that θ(t) converges to z exponentially fast.

Symmetry of the solution
Similar to the two-layer case, we can find that the solution of the
equation for z < 0 can be obtained by negating the sign of β(t) for the positive signal case. Therefore,
we will focus on the case of z ≥0 where a(t), b(t), β(t) and thus θ(t) are all non-negative and
non-decreasing.

Conservation quantities
Similarly, we have

1
2
d
dta2 =
1
2D
d
dtb2 = 1

2
d
dtβ2 = θ(z −θ).
(31)

so

a =
 
λ + β2 1

2 ,
b =
 
b2
0 + Dβ2 1

2 .
(32)

Using these conservation quantities, we can prove the following estimations in terms of β:

min(λ
1
2 , β) ≤a ≤
√

2 max(λ
1
2 , β),

min(b0,
√

Dβ) ≤b ≤
√

2 max(b0,
√

Dβ).
(33)

The evolution of θ
It is direct to compute that

˙θ = ˙abDβ + aDbD−1˙bβ + abD ˙β

=

(bDβ)2 + (DabD−1β)2 + (abD)2
(z −θ)

= θ2(a−2 + Db−2 + β−2)(z −θ).

(34)

Auxiliary notations
Let us introduce

T (1) = inf
n
t ≥0 : β(t) ≥λ
1
2
o
,
T (2) = inf
n
t ≥0 : β(t) ≥b0/
√

D
o
,

T esc = min(T (1), T (2)),
T sig = inf {t ≥0 : θ(t) ≥z/2} .
(35)

Lemma D.2 (Noise case). For the gradient flow (30), we have the initial estimation

|β(t)| ≤2
D+1

2 λ
1
2 bD
0 |z|t,

|θ(t)| ≤2D+1λb2D
0 |z|t,
for
t ≤min(T (1), T (2)),
(36)

where

T (1) =

2
D+1

2 bD
0 |z|
−1
,
T (2) =

2
D+1

2 √

Dλ
1
2 bD−1
0
|z|
−1
.
(37)

27


---Page Break---
Moreover, if λ
1
2 ≤b0/
√

D, then

|β(t)| ≤λ
1
2 exp

2
D+1

2 bD
0 |z|(t −T (1))+
,

|θ(t)| ≤2
D+1

2 λbD
0 exp

2
D+3

2 bD
0 |z|(t −T (1))+
,
for
t ≤T (1,2),
(38)

where

T (1,2) =

1 + ln
b0
√

Dλ
1
2


T (1).
(39)

Proof. It suffices to consider the case z > 0. Recalling (30) and θ ≥0, we have

˙β ≤abDz.

Using the upper bound in (33), when t ≤T esc, namely when β(t) ≤λ
1
2 and
√

Dβ(t) ≤b0, we have

˙β ≤(
√

2λ
1
2 )(
√

2b0)Dz = 2
D+1

2 λ
1
2 bD
0 z,

implying that

β(t) ≤2
D+1

2 λ
1
2 bD
0 zt,
for
t ≤T esc.

Therefore, we get the lower bound

T esc ≥min(T (1), T (2)),

where T (1) and T (2) are defined by (37) in the lemma. Combining this again with the upper bound
that θ = abDβ ≤2
D+1

2 a0bD
0 β when t ≤T esc, we prove (36).

For the second part, we consider the case λ
1
2 ≤b0/
√

D. In this case, we have T (1) ≤T (2) and
thus T (1) ≥T (1) from the above argument. Now, when t ∈[T (1), T (2)], we turn to the following
equation:

˙β ≤(
√

2β)(
√

2b0)Dz = 2
D+1

2 bD
0 zβ,
for
t ∈[T (1), T (2)],

which yields

β(s + T (1)) ≤β(T (1)) exp

2
D+1

2 bD
0 zs

= λ
1
2 exp

2
D+1

2 bD
0 zs

,
for
s ∈[0, T (2) −T (1)].

Comparing β(s + T (1)) with b0/
√

D gives

T (2) −T (1) ≥
h
2
D+1

2 bD
0 z
i−1
ln
b0
√

Dλ
1
2 = T (1) ln
b0
√

Dλ
1
2 ,

so

T (2) ≥T (1) + T (1) ln
b0
√

Dλ
1
2 ≥T (1)

1 + ln
b0
√

Dλ
1
2


= T (1,2).

Therefore, the comparison principal yields

β(t) ≤λ
1
2 exp

2
D+1

2 bD
0 |z|(t −T (1))+
for
t ≤T (1,2),

where we notice that the bound also holds for t ≤T (1) ≤T (1) since at that time β(t) ≤λ
1
2 . Finally,
(38) is obtained by using θ = abDβ ≤2
D+1

2 bD
0 β2 when t ∈[T (1), T (2)], while the bound also holds
for t ≤T (1).

Lemma D.3 (Signal case). For the gradient flow (30), we have:

• If λ
1
2 ≤b0/
√

D, then

T sig ≤2(bD
0 |z|)−1



1 +

 

ln (D−D/2z/2)
1
D+2

λ
1
2

!+

,
(40)

28


---Page Break---
• If λ
1
2 ≥b0/
√

D, then

T sig ≤2
√

Dλ
1
2 bD−1
0
|z|
−1  
1 + R+
,
(41)

where

R =

(
ln (D|z|/2)
1
D+2
b0
,
D = 1,
1
D−1,
D > 1.

Moreover, we have

|z −θ(t)| ≤1

2|z| exp

−1

4D
D
D+2 |z|

2D+2

D+2 (t −T sig)

,
for
t ≥T sig.
(42)

Proof. It suffices to consider the case z > 0. To provide an upper bound of the signal time T sig, we
observe that the lower bound in (33) implies a sufficient condition for θ ≥z/2 that

β ≥

D−D/2z/2

1
D+2
=⇒
θ ≥1

2z.
(43)

We first consider case that λ
1
2 ≤b0/
√

D. Let us define T
(1) := 2(bD
0 z)−1 and suppose that
T sig ≥2(bD
0 z)−1, otherwise the statement (40) already holds. Then, we first have

˙β = abD(z −θ) ≥1

2λ
1
2 bD
0 z,
for
t ≤T sig.
(44)

This implies that

β(t) ≥1

2λ
1
2 bD
0 zt,
for
t ≤T sig,

and thus

T (1) ≤T
(1) ≤T sig.

Now, for t ∈[T (1), T sig], we use the alternative bound a ≥β to obtain

˙β ≥1

2βbD
0 z,
for
t ∈[T (1), T sig],

giving that

β(s + T (1)) ≥λ
1
2 exp
1

2bD
0 zs

,
for
s ∈[0, T sig −T (1)].

Comparing it with (43), we obtain

T sig −T (1) ≤2(bD
0 z)−1 ln (D−D/2z/2)
1
D+2

λ
1
2
= T
(1) ln (D−D/2z/2)
1
D+2

λ
1
2
,

which, together with the upper bound of T (1), proves (40).

The case that λ
1
2 ≥b0/
√

D is very similar. We define T
(2) := 2(
√

Dλ
1
2 bD−1
0
z)−1 and suppose that

T sig ≥T
(2). Using the estimation (44) again, we get

T (2) ≤T
(2) ≤T sig.

Now, when t ∈[T (2), T sig], we use b ≥
√

Dβ to obtain

˙β ≥1

2λ
1
2 D
D

2 βDz,
for
t ∈[T (2), T sig].

This implies that for s ≤T sig −T (2),

β(s + T (2)) ≥
b0
√

D
exp
1

2λ
1
2 D
D

2 zs

,
if
D = 1,

29


---Page Break---
β(s + T (2)) ≥

(D−1/2b0)−(D−1) −D −1

2
λ
1
2 D
D

2 zs
−
1
D−1
,
if
D > 1.

Consequently, comparing it with (43) gives

T sig −T (2) ≤2

λ
1
2 D
D

2 z
−1
ln (Dz/2)
1
D+2

b0
= T
(2) ln (Dz/2)
1
D+2

b0
,
if
D = 1,

T sig −T (2) ≤
2
D −1

√

Dλ
1
2 bD−1
0
z
−1
=
1
D −1T
(2),
if
D > 1.

Finally, let us consider the convergence stage when t ≥T sig. Since it is always true that θ ≤z, using
the lower bounds in (32), we have

z ≥θ = abDβ ≥β · D
D

2 βD · β = D
D

2 βD+2,

implying that β ≤D−
D
2(D+2) z
1
D+2 . Now, plugging this into (34) and noticing that θ ≥z/2 when
t ≥T sig, we derive

˙θ = θ2(a−2 + Db−2 + β−2)(z −θ)

≥θ2β−2(z −θ)

≥1

4z2D
D
D+2 z−
2
D+2 (z −θ)

= 1

4D
D
D+2 z
2D+2

D+2 (z −θ).

Therefore, we have

z −θ(s + T sig) ≤(z −θ(T sig)) exp

−1

4D
D
D+2 z
2D+2

D+2 s

= 1

2 exp

−1

4D
D
D+2 z
2D+2

D+2 s

,

30


---Page Break---
E
Main proofs

Notations
For notation simplicity, we will use C, c to denote generic positive constants that may
change from line to line.

E.1
Proof of Theorem 3.1

We recall that

ER(ˆθ; θ∗) = E

∞
X

j=1
(ˆθj −θ∗
j )2.

Let us define the signal event

Sj =

ω : |ξj| < 1

2

θ∗
j


,
S∁
j =

ω : |ξj| ≥1

2

θ∗
j


.
(45)

Then, on Sj we have 1

2
θ∗
j
 ≤|zj| ≤3

2
θ∗
j
, while on S∁
j we have |zj| ≤3|ξj|. Then, we decompose

(ˆθj −θ∗
j )2 = (ˆθj −θ∗
j )21Sj + (ˆθj −θ∗
j )21S∁
j .

Moreover, when the signal is significant, we use

(ˆθj −θ∗
j )21Sj ≤2(ˆθj −zj)21Sj + 2(zj −θ∗
j )21Sj = 2(ˆθj −zj)21Sj + 2ξ2
j 1Sj.

On the other hand, when the noise is dominating, we apply

(ˆθj −θ∗
j )21S∁
j ≤2ˆθ2
j1S∁
j + 2(θ∗
j )21S∁
j .

Summing over j and taking the expectation, we have

R(ˆθOp; θ∗) = E

∞
X

j=1
(ˆθj −θ∗
j )2 = E

∞
X

j=1
(ˆθj −θ∗
j )21Sj + E

∞
X

j=1
(ˆθj −θ∗
j )21S∁
j

≤2E

∞
X

j=1
(ˆθj −zj)21Sj + 2E

∞
X

j=1
ξ2
j 1S∁
j + 2E

∞
X

j=1
ˆθ2
j1S∁
j + 2E

∞
X

j=1
(θ∗
j )21S∁
j

= 2E

∞
X

j=1

h
ξ2
j 1Sj + (θ∗
j )21S∁
j

i
(46)

+ 2E

∞
X

j=1
ˆθ2
j1S∁
j
(47)

+ 2E

∞
X

j=1
(ˆθj −zj)21Sj.
(48)

Now, the first term (46), representing the absolute error, is controlled by Proposition E.1 that

E

∞
X

j=1

h
ξ2
j 1Sj + (θ∗
j )21S∁
j

i
≤4

ϵ2Φ(ϵ) + Ψ(ϵ)

.

Therefore, we focus on the remaining two terms and obtain the estimations (49) and (51) in the
following.

The noise term
The term (47) represents the extra error caused by the estimator when the noise
dominates. Applying Lemma D.1, we obtain
ˆθj
 = |θj(t)| ≤
˜θ(t)
 = λj(Ej −1)

2|zj| + λjEj
|zj| ≤1

2λj exp ((6|ξj| + λj) t)
on
S∁
j .

31


---Page Break---
Let us choose J = min {j ≥1 : λj ≤ϵ} ≍ϵ−1/γ. Then, since t ≤B2ϵ−1, we have (|ξj| + λj)t ≤
B2(|ξj|/ϵ + 1) and thus

E
X

j≥J
ˆθ2
j1S∁
j ≤C
X

j≥J
λ2
jE exp(C(|ξj|/ϵ) + C) ≤C
X

j≥J
λ2
j ≤CJ−(2γ−1) ≤Cϵ2−1/γ,

where we notice that E exp(C(|ξj|/ϵ) + C) is uniformly bounded by some constant for all j since

each |ξj|/ϵ is 1-sub-Gaussian. On the other hand, using the obvious bound
ˆθj
 ≤3|ξj| on S∁
j , we
obtain

E
X

j<J
ˆθ2
j1S∁
j ≤C
X

j<J
Eξ2
j ≤Cϵ2J ≤Cϵ2−1/γ.

Combining two terms, we conclude that

E

∞
X

j=1
ˆθ2
j1S∁
j ≤Cϵ2−1/γ.
(49)

The signal term
The term (47) represents the approximation error when the signal is significant.
We apply Lemma D.1 again to derive
ˆθj −zj
 ≤
˜θ(t/
√

2) −zj
 =
2|zj| + λj
2|zj| + λj exp
 
(2|zj| + λj)t/
√

2
|zj|.

Using the fact that 1

2
θ∗
j
 ≤|zj| ≤3

2
θ∗
j
 on Sj, we derive that

(ˆθj −zj)21Sj ≤C
(
θ∗
j
 + λj)2θ2
j
λ2
j exp
 
(2
θ∗
j
 + λj)t/
√

2
.
(50)

Let us define ν = ϵ ln(1/ϵ) ≥ϵ. Recalling (9) and using Assumption 1, we have

j ≤max Jsig(ϵ) ≤Cϵ−κ,
for
j ∈Jsig(ν) ⊆Jsig(ϵ).

Now, if we take t ≥B1ϵ−1 for some constant B1, since
θ∗
j
 ≥ν for j ∈Jsig(ν), we also have

1
√

2

θ∗
j
t ≥
1
√

2tϵ ln(1/ϵ) ≥cB1 ln(1/ϵ),

and thus when j ∈Jsig(ν),

ln
h
λ2
j exp
θ∗
j
t/
√

2
i
= 2 ln λj + 1
√

2

θ∗
j
t ≥cB1 ln(1/ϵ) −C ln j ≥(cB1 −C) ln(1/ϵ).

Consequently, as long as B1 is large enough, we have

λ2
j exp
θ∗
j
t/
√

2

≥1
when
j ∈Jsig(ν).

Therefore, plugging this into (50), we get

(ˆθj −zj)21Sj ≤C
(
θ∗
j
 + λj)2θ2
j
λ2
j exp
 
(2
θ∗
j
 + λj)t/
√

2
 ≤C exp

−(
θ∗
j
 + λj)t/(
√

2)

(
θ∗
j
 + λj)2θ2
j,

and hence
X

j∈Jsig(ν)
E(ˆθj −zj)21Sj ≤
X

j∈Jsig(ν)
exp

−(
θ∗
j
 + λj)t/(
√

2)

(
θ∗
j
 + λj)2θ2
j

≤C

∞
X

j=1


(
θ∗
j
 + λj)t
−2 (
θ∗
j
 + λj)2θ2
j

= Ct−2
∞
X

j=1
θ2
j ≤Cϵ2.

32


---Page Break---
On the other hand, with the trivial bound
ˆθj −zj
 ≤|zj|, the remaining terms are bounded by
X

j /∈Jsig(ν)
E(ˆθj −zj)21Sj ≤
X

j /∈Jsig(ν)
(θ∗
j )2 = Ψ(ν).

Therefore, we conclude that

E

∞
X

j=1
(ˆθj −zj)21Sj ≤Cϵ2 + Ψ (ϵ ln(1/ϵ)) .
(51)

E.2
Proof of Theorem 3.2

Following the same argument as the proof in the previous section, we introduce the events Sj and S∁
j
in (45) and decompose the error as in (46), (47) and (48). Then, we will focus on the last two terms
and derive the estimations (53) and (54) in the following. We recall that b0 ≍ϵ
1
D+2 and we choose t
such that B1ϵ−1 ≤bD
0 t ≤B2ϵ−1 for some constants B1, B2 > 0 that will be determined later.

Here, we also note that the in correspondence to the component-wise gradient flow considered in
Subsection D.2, the initializations are given by λ = λj and b0,j = b0 in (30). Now, since λj ≍j−γ,
the index

J = min
n
j ≥1 : λ1/2
j
≤b0,j/
√

D
o
≍b−2/γ
0
.
(52)

The noise term
To apply the bounds in Lemma D.2, let us denote the event

Aj =




ω : 3 · 2
D+1

2 bD
0 |ξj|t ≤ln
b0

λ

1
2
j
√

D




.

Then, since |zj| ≤3|ξj| on S∁
j , we have t ≤T (1,2)
j
on Aj, where T (1,2)
j
is defined via (39). Then, for
j > J, applying (38) yields

ˆθ2
j1S∁
j ∩Aj ≤2D+1b2D
0 λ2
j exp

2
D+5

2 bD
0 |zj|t

1S∁
j ∩Aj

≤2D+1b2D
0 λ2
j exp

2
D+5

2 B2|zj|/ϵ

1S∁
j ∩Aj

≤Cb2D
0 λ2
j exp(C|ξj|/ϵ)1S∁
j ∩Aj

where we use bD
0 t ≤B2ϵ−1 in the last inequality. Consequently,

E
X

j>J
ˆθ2
j1S∁
j ∩Aj ≤Cb2D
0
X

j>J
λ2
jE exp(C|ξj|/ϵ) ≤Cb2D
0
X

j>J
λ2
j

≤Cb2D
0 J−(2γ−1) ≤Cb2(D+2−1/γ)
0
,

where in the second inequality we notice that E exp(C|ξj|/ϵ) is uniformly bounded since each |ξj|/ϵ
is 1-sub-Gaussian.

On the other hand, noticing bD
0 t ≤B2ϵ−1 again, we have

j ∈A∁
j
=⇒
CbD
0 t|ξj| ≥ln
b0

λ

1
2
j
√

D

=⇒
|ξj|/ϵ ≥cB−1
2
ln
b0

λ

1
2
j
√

D
= cB−1
2
ln
 
b2
0λ−1
j /D

.

Hence, using Lemma F.1 with the sub-Gaussian property of ξj and noticing that b2
0λ−1
j /D ≥1 when
j > J, we obtain

E
X

j>J
ˆθ2
j1S∁
j ∩A∁
j ≤C
X

j>J
E

ξ2
j 1

|ξj|/ϵ ≥cB−1
2
ln
 
b2
0λ−1
j /D
	

33


---Page Break---
≤Cϵ2 X

j>J
exp

−c

cB−1
2
ln
 
b2
0λ−1
j /D
2

≤Cϵ2 X

j>J
exp

−c

ln
 
b2
0jγ2

≤Cϵ2
Z ∞

J
exp

−c

ln
 
b2
0xγ2
dx

≤Cϵ2b−2/γ
0

Z ∞

c
exp

−c [ln(yγ)]2
dy,
y = b2/γ
0
x

≤Cϵ2b−2/γ
0
≤Cϵ2ϵ−
2
D+2
1
γ .

Finally, using the bound
ˆθj
 ≤3|ξj| again, the remaining terms are bounded by

E
X

j≤J
ˆθ2
j1S∁
j ≤ϵ2J ≤Cϵ2b−2/γ
0
≤Cϵ2ϵ−
2
D+2
1
γ .

In summary, we conclude that

E

∞
X

j=1
ˆθ2
j1S∁
j ≤Cϵ2ϵ−
2
D+2
1
γ .
(53)

The signal term
We will apply the bound in Lemma D.3. Let us denote

Jrec =
n
j : t ≥2T sig
j
o
∩Jsig(ϵ).

Then, when j ∈Jrec and Sj holds, (42) and the fact 1

2
θ∗
j
 ≤|zj| ≤3

2
θ∗
j
 imply

|θj −zj| ≤1

2|zj| exp

−1

4D
D
D+2 z
2D+2

D+2 (t −T sig
j
)

≤C
θ∗
j
 exp

−c
θ∗
j

2D+2

D+2 t

.

Consequently,

E
X

j∈Jrec
(ˆθj −zj)21Sj ≤C
X

j∈Jrec
(θ∗
j )2 exp

−c
θ∗
j

2D+2

D+2 t

≤C
X

j∈Jrec
(θ∗
j )2
θ∗
j

2D+2

D+2 t
−D+2

D+1

= C
X

j∈Jrec
t−D+2

D+1 ≤Ct−D+2

D+1 |Jsig(ϵ)| ≤Cϵ2Φ(ϵ),

where we use exp(−cx) ≤Cx−D+2

D+1 in the second inequality.

Let us define ν = ϵ ln(1/ϵ) ≥ϵ. We claim that J∁
rec ⊆Jsig(ν)∁on Sj as long as bD
0 t ≥B1ϵ−1 for

some large constant B1. Then, using the obvious bound
ˆθj −zj
 ≤|zj| ≤3

2
θ∗
j
 on Sj, we have

E
X

j∈J∁
rec

(ˆθj −zj)21Sj ≤
X

j∈Jsig(ν)∁
(θ∗
j )2 = Ψ(ν).
(54)

To prove the claim, we show that Jsig(ν) ⊆Jrec on Sj as long as B1 is large enough. Recalling (9)
and using Assumption 1, for j ∈Jsig(ν) ⊆Jsig(ϵ), we have

j ≤max Jsig(ϵ) ≤Cϵ−κ.

Now, we show that t ≥2T sig
j
for j ∈Jsig(ν) on Sj for different cases in Lemma D.3.

• If λ1/2
j
≤b0/
√

D, we have (40) and thus

t ≥2T sig
j
⇐=
bD
0 |zj|t ≥1 +

 

ln (D−D/2z/2)
1
D+2

λ1/2
j

!+

34


---Page Break---
⇐=
B1

2 |θj|ϵ−1 ≥C ln
 θ∗
j
λ−1
j

+ C

⇐=
B1ϵ ln(1/ϵ)ϵ−1 ≥Cγ ln j + C
⇐=
B1 ln(1/ϵ) ≥Cκ ln(1/ϵ) + C.

• If λ1/2
j
≥b0/
√

D, (41) gives

t ≥2T sig
j
⇐=
√

Dλ1/2
j
bD−1
0
|zj|t ≥1 + R+
j ,

where

Rj =

(
ln (D|zj|/2)
1
D+2
b0
,
D = 1,
1
D−1,
D > 1.

So for both D = 1 and D > 1, we have similarly

t ≥2T sig
j
⇐=
1
2

√

Dλ1/2
j
bD−1
0
θ∗
j
t ≥C ln
 
b−1
0

+ C

⇐=
1
2|θj|bD
0 t ≥C ln(1/ϵ) + C

⇐=
B1 ln(1/ϵ) ≥C ln(1/ϵ) + C.

Therefore, for both cases, we have t ≥2T sig
j
as long as B1 is large enough. This finishes the proof of
the claim.

E.3
The absolute error term

The following proposition connect the absolute error term with the ideal risk in Johnstone [2017].
Proposition E.1. For the sequence model (5), recalling the signal events (45) and the quantities (9),
we have

E

∞
X

j=1

h
ξ2
j 1Sj + θ2
j1S∁
j

i
≤4

∞
X

j=1
min(ϵ2, θ2
j) = 4

ϵ2Φ(ϵ) + Ψ(ϵ)

.
(55)

Proof. It is straightforward to see that

ξ2
j 1Sj + θ2
j1S∁
j = ξ2
j 1{2|ξj|<|θj|} + θ2
j1{2|ξj|≥|θj|} ≤min(4ξ2
j , θ2
j),

so

E
h
ξ2
j 1Sj + θ2
j1S∁
j

i
≤E min(4ξ2
j , θ2
j) ≤4E min(ξ2
j , θ2
j) ≤4 min(ϵ2, θ2
j).

Summing over j yields the inequality. The last equality follows from the definition of Φ(ϵ) and
Ψ(ϵ).

E.4
Proof of Proposition 3.4

The fact that aj(t)bD
j (t) is non-decreasing follows from the analysis of the gradient flow in
Subsection D.1 and Subsection D.2. For δ ∈(0, 1), let us choose C large enough such that
P {|ξj| ≤Cϵ} ≥1 −δ for any fixed j.

The case D = 0
For the signal component where
θ∗
j
 ≥2Cϵ ln(1/ϵ), we have |zj| ≥1

2
θ∗
j
 with
high probability. Then, we follow the analysis of the signal term in Subsection E.1 and obtain that

|θj(t) −zj|2 ≤C(θ∗
j )2t−2 ≤1

4(θ∗
j )2,

provided that ϵ is small enough. This implies that

|θj(t)| ≥1

4

θ∗
j
.

35


---Page Break---
Now, the second inequality in (24) implies that |θj(t)| ≤a2
j(t). Consequently, we conclude that

aj(t) ≥1

2
θ∗
j
1/2.

For the noise component where
θ∗
j
 ≤ϵ, we have |zj| ≤(C + 1)ϵ with high probability. Moreover,
since λj ≍j−γ, we have J = min {j ≥1 : λj ≤ϵ} ≍ϵ−1/γ. Following the similar analysis of the
noise term in Subsection E.1, when j ≥Cϵ−1/γ for some C > 0, we have

|θj(t)| ≤λj exp(C(|zj| + λj)t) ≤λj exp(Cϵt) ≤Cλj.

Then, the first inequality in (24) gives |θj(t)| ≥β2
j (t), so we have |βj(t)| ≤Cλ1/2
j
. Finally,

aj(t) =
q

λj + β2
j (t) ≤
p

λj + Cλj ≤Cλ1/2
j
.

The case D ≥1
For the signal component where
θ∗
j
 ≥2Cϵ ln(1/ϵ), we still have we have
|zj| ≥1

2
θ∗
j
 with high probability. Now, from the analysis of the signal term in Subsection E.2,
we have t ≥2T sig
j
. Moreover, investigating the proof of Lemma D.3, we see that the analysis in
Subsection E.2 actually shows that

|βj(t)| ≥c|zj|

1
D+2 ≥c
θ∗
j

1
D+2 .

Consequently, (32) implies that

aj(t)bD
j (t) ≥|βj(t)|D+1 ≥c
θ∗
j

D+1
D+2 .

For the noise component where
θ∗
j
 ≤ϵ, we also have |zj| ≤(C + 1)ϵ with high probability. Now,
we have

|zj|bD
0 t ≤(C + 1)ϵbD
0 t ≤C0,

for some constant C0. Following the similar analysis of the noise term in Subsection E.2, we can
choose j ≥Cϵ−
2
D+2
1
γ such that

1 + ln
b0
λ1/2
j
√

D
≥C0.

Now, this condition guarantees that t ≤T (1,2) defined in Lemma D.2, so Lemma D.2 gives

|β(t)| ≤λ

1
2
j exp

CbD
0 |zj|(t −T (1))+
≤λ

1
2
j exp
 
CbD
0 |zj|t

≤Cλ

1
2
j .

Combining it with the upper bound in (32) and noticing t ≤T (1,2) yield

aj(t)bD
j (t) ≤2
D+1

2 |βj(t)|bD
0 ≤Cϵ
D
D+2 λ

1
2
j .

36


---Page Break---
F
Auxiliary results

Lemma F.1. Suppose X is σ2-sub-Gaussian, namely, P {|X| ≥t} ≤2 exp
 
−1

2σ2 t

for t ≥0. Then
for M ≥0, we have the tail bound

EX21 {|X| ≥M} ≤4σ2 exp

−1

4σ2 M 2

.
(56)

Proof. Using integration by parts, we have

EX21 {X ≥M} = 2
Z ∞

0
rP {|X| ≥max(M, r)} dr

≤4
Z ∞

0
r exp

−1

2σ2 max(M 2, r2)

dr

=
 
M 2 + 2σ2
exp

−M 2

2σ2



≤4σ2 exp

−1

4σ2 M 2

.

Lemma F.2. Suppose that (θj)j≥1 satisfies
θl(j)
 ≍j−(p+1)/2 for some p > 0 and |θj| = 0
otherwise, where l(j) is a sequence of indices. Defining Φ(δ) and Ψ(δ) as in (9), we have

Φ(δ) ≍δ−
2
p+1 ,
Ψ(δ) ≍δ

2p
p+1 .

Proof. First, from the definition of Φ(δ) and Ψ(δ), we see that they do not depend on ordering of
the indices and zero values of θj. Therefore, we can assume that l(j) = j. Then, assuming that
c1j−(p+1)/2 ≤|θj| ≤C1j−(p+1)/2, we have

Φ(δ) = |{j : |θj| ≥δ}| ≤

n
j : C1j−(p+1)/2 ≥δ
o ≤(δ/C1)−
2
p+1 .

Moreover,

Ψ(δ) =

∞
X

j=1
|θj|21 {|θj| < δ}

=
X

j>Φ(δ)
|θj|2 ≤C2
1
X

j>Φ(δ)
j−(p+1)

≤C2
1CΦ(δ)−p ≤C′δ

2p
p+1

for some constant C′ > 0. The lower bound of them can be obtained similarly.

37


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The main claims are supported by our main theorems as well as the numerical
experiments.
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
Justification: We have discussed the limitations in the last section. The assumptions are also
explained and justified.
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
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

38


---Page Break---
Answer: [Yes]

Justification: The proofs are provided in the appendix.

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

Justification: The related codes are provided in the supplementary material.

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
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

39


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: The related codes are provided in the supplementary material.
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
Justification: The related codes are provided in the supplementary material.
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
Justification: Error bars are reported in Figure 3 on page 21.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)

40


---Page Break---
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

Justification: The experiments can be done by a 64 CPU core laptop with 32 GB memory in
one day.
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
Justification: We follow the NeurIPS Code of Ethics.
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
Justification: It is mainly a theory paper, so there is no societal impact of the work
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

41


---Page Break---
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

Justification: It is mainly a theory paper.

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

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.

42


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

43


---Page Break---
