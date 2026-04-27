The High Line: Exact Risk and Learning Rate Curves
of Stochastic Adaptive Learning Rate Algorithms

Elizabeth Collins-Woodfin
McGill University
elizabeth.collins-woodfin@mail.mcgill.ca

Inbar Seroussi
Tel-Aviv University
inbarser@tauex.tau.ac.il

Begoña García Malaxechebarría
University of Washington
begogar9@uw.edu

Andrew W. Mackenzie
McGill University
andrew.mackenzie@mail.mcgill.ca

Elliot Paquette
McGill University
elliot.paquette@mcgill.ca

Courtney Paquette∗
McGill University & Google DeepMind
courtney.paquette@mcgill.ca

Abstract

We develop a framework for analyzing the training and learning rate dynamics on a
large class of high-dimensional optimization problems, which we call the high line,
trained using one-pass stochastic gradient descent (SGD) with adaptive learning
rates. We give exact expressions for the risk and learning rate curves in terms of
a deterministic solution to a system of ODEs. We then investigate in detail two
adaptive learning rates – an idealized exact line search and AdaGrad-Norm – on
the least squares problem. When the data covariance matrix has strictly positive
eigenvalues, this idealized exact line search strategy can exhibit arbitrarily slower
convergence when compared to the optimal fixed learning rate with SGD. Moreover
we exactly characterize the limiting learning rate (as time goes to infinity) for line
search in the setting where the data covariance has only two distinct eigenvalues.
For noiseless targets, we further demonstrate that the AdaGrad-Norm learning
rate converges to a deterministic constant inversely proportional to the average
eigenvalue of the data covariance matrix, and identify a phase transition when the
covariance density of eigenvalues follows a power law distribution. We provide
our code for evaluation at https://github.com/amackenzie1/highline2024.

1
Introduction

In deterministic optimization, adaptive stepsize strategies, such as line search (see [40], therein),
AdaGrad-Norm [59], Polyak stepsize [48], and others were developed to provide stability and improve
efficiency and adaptivity to unknown parameters. While the practical benefits for deterministic
optimization problems are well-documented, much of our understanding of adaptive learning rate
strategies for stochastic algorithms are still in their infancy.

There are many adaptive learning rate strategies used in machine learning with many design goals.
Some are known to adapt to stochastic gradient descent (SGD) gradient noise while others are
robust to hyper-parameters (e.g., [4, 63]). Theoretical results for adaptive algorithms tend to focus on
guaranteeing minimax-optimal rates, but this theory is not engineered to provide realistic performance

∗Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
10
2
10
1
100
101

SGD Iterations/d

100

Learning rate

100

6 × 10
1

Risk

AdaGrad-Norm Least Squares

d = 256
d = 1024
d = 4096
d = 16384
Theory, learn. rate
Theory, risk

10
1
100
101

SGD Iterations/d

100

9.2 × 10
1

9.3 × 10
1

9.4 × 10
1

9.5 × 10
1

9.6 × 10
1

9.7 × 10
1

9.8 × 10
1

9.9 × 10
1

Learning rate

10
4

10
3

10
2

10
1

100

Risk

AdaGrad-Norm Logistic Regression

d = 16
d = 32
d = 64
d = 128
Theory, learn. rate
Theory, risk

Figure 1: Concentration of learning rate and risk for AdaGrad-Norm on least squares with label
noise ω = 1 (left) and logistic regression with no noise (right). As dimension increases, both risk and
learning rate concentrate around a deterministic limit (red) described by our ODE in Theorem 2.1.
The initial risk increase (left) suggests the learning rate started too high, but AdaGrad-Norm adapts.
Our ODEs predict this behavior. See Sec. H for simulation details.

comparisons; indeed many adaptive algorithms are minimax-optimal, and so more precise statements
are needed to distinguish them. For instance, the exact learning rates (or rate schedules) to which these
strategies converge are unknown, nor their dependence on the geometry of the problem. Moreover,
we often do not know how these adaptive stepsizes compare with well-tuned constant or decaying
fixed learning rate SGD, which can be viewed as a cost associated with selecting the adaptive strategy
in comparison to tuning by hand.

In this work, we develop a framework for analyzing the exact dynamics of the risk and adaptive
learning rate strategies for a wide class of optimization problems that we call high-dimensional linear
(high line) composite functions. In this class, the objective function takes the form of an expected
risk R : Rd →R over high-dimensional data (a, ϵ) ∼D ⊂Rd × R of a function f : R3 →R
composed with the linear functions ⟨X, a⟩, ⟨X⋆, a⟩. That is, we seek to solve

min
X∈Rd

n
R(X)
def
= E a,ϵ[f(⟨a, X⟩, ⟨a, X⋆⟩, ϵ)]
for
(a, ϵ) ∼D, X⋆∈Rdo
.
(1)

We suppose a ∼N(0, K) where K ∈Rd×d is the covariance matrix. We train (1) using (one-pass)
stochastic gradient descent with adaptive learning rates, gk (SGD+AL). Our main goal is to give
a framework for better2 performance analysis of these adaptive methods. We then illustrate this
framework by considering two adaptive learning rate algorithms on the least squares problem3, the
results of which appear in Table 1: exact line-search (idealistic) (Sec. 3) and AdaGrad-Norm (Sec. 4).
We expect other losses and adaptive learning rates can be studied using this approach.

Main contributions.
Performance analysis framework. We provide an equivalence of R(Xk) and
learning rate gk under SGD+AL to deterministic functions R(t) and γt via solving a deterministic
system of ODEs (see Section 2), which we then analyze to show how the covariance spectrum
influences the optimization. See Figure 1. As the dimension d of the problem grows, the learning
curves of R(Xk) become closer to R(t) and the curves concentrate around R(t) with probability
better than any inverse power of d (See Theorem 2.1).

Greed can be arbitrarily bad in the presence of strong anisotropy (that is, Tr(K)/d ≪Tr(K2)/d).
Our analysis reveals that exact line search, which is to say optimally decreasing the risk at each step,
can run arbitrarily slower than the best fixed learning rate for SGD on a least squares problem when
λmin
def
= λmin(K) > C > 0. The best fixed stepsize (least squares problem) is (Tr(K)/d)−1 or the
inverse of the average eigenvalue, see Polyak stepsize [48]. Line search, on the other hand, converges
to a fixed stepsize of order λmin/(Tr(K2)/d). It can be that λmin/(Tr(K2)/d) ≪(Tr(K)/d)−1
making exact line search substantially underperform Polyak stepsize. We further explore this and, in
the case where d-eigenvalues of K take only two values λ1 > λ2 > 0, we give an exact expression
as a function of λ1 and λ2 for the limiting behavior of γt as t →∞(See Fig. 5).

2More realistic, in that it deals with high-dimensional anisotropic loss geometries and more precise, in that it
can distinguish minimax optimal algorithms as more-or-less performant.
3We extend some results to the general strongly convex setting.

2


---Page Break---
Table 1: Summary of adaptive learning rates results on the least squares problem. We summarize
our results for line search and AdaGrad-Norm under various assumptions on the covariance matrix
K. We denote λmin the smallest non-zero eigenvalue of K and Tr(K)

d
the average eigenvalue. Power
law(δ, β) assumes the eigenvalues of K, {λi}d
i=1, follow a power law distribution, that is, for 0 <
β < 1, λi ∼(1 −β)λ−β1(0,1) for all 1 ≤i ≤d and ⟨X0 −X⋆, ωi⟩2 ∼λ−δ
i
where {ωi}d
i=1 are
eigenvectors of K (see Prop 4.4). For ∗(see Prop. 4.2), requires a good initialization on b, η.

Learning rate
K assumption
Limiting γ∞
Convergence rate

AdaGrad-Norm(b, η)
(see Sec. 4)
λmin > C
γt ≍
η2

b
η + 1

4d Tr(K)∥X0−X⋆∥2
log(R)∗≍−λminγ∞t

AdaGrad-Norm(b, η)
Power law
(see Sec. 4)

β + δ < 1
γt ≍δ,β 1
R(t) ≍δ,β tβ+δ−2

β + δ = 1
γt ≍δ,β
1
log(t+1)
R(t) ≍δ,β

t
log(t+1)
−1

1 < β + δ < 2
γt ≍δ,β t−1+
1
β+δ
R(t) ≍δ,β t−
2
β+δ +1

Exact line search,
idealized
(see Sec. 3)
λmin > C
γt ≍
λmin
Tr(K2)/d
log(R) ≍−λminγ∞t

Polyak stepsize
(see Sec. 3)
λmin > C
γt =
1
Tr(K)/d
log(R) ≍−λminγ∞t

AdaGrad-Norm selects the optimal step-size, provided it has a warm start. In the absence of label
noise and when the smallest eigenvalue of K satisfies λmin > C > 0, the learning rate converges to
a deterministic constant that depends on the average condition number (like in Polyak) and scales
inversely with Tr(K)

d
∥X0 −X⋆∥2. Therefore it attains automatically the optimal fixed stepsize in
terms of the covariance without knowledge of Tr(K), but pays a penalty in the constant, namely
∥X0 −X⋆∥2. If one knew ∥X0 −X⋆∥2 then by tuning the parameters of AdaGrad-Norm one
might achieve performance consistent with Polyak; this also motivates more sophisticated adaptive
algorithms such as DoG [29] and D-Adaptation [18], which adaptively compensate and/or estimate
∥X0 −X⋆∥2.

AdaGrad-Norm can use overly pessimistic decaying schedules on hard problems. Consider power law
behavior for the spectrum of K and the signal X⋆. This is a natural setting as power law distributions
have been observed in many datasets [60]. Here the learning rate and asymptotic convergence of
K undergo a phase transition. For power laws corresponding to easier optimization problems, the
learning rate goes to a constant and the risk decays at t−α1. For harder problems, the learning rate
decays like t−η1 and the risk decays at a different sublinear rate t−α2. See Table 1 and Sec. 4 for
details.

Notation. Define R+ = [0, ∞). We say an event holds with overwhelming probability, w.o.p., if
there is a function ω : N →R with ω(d)/ log d →∞so that the event holds with probability at
least 1 −e−ω(d). We let 1A(x) be the indicator function of the set A where it is 1 if x ∈A and 0
otherwise. For a matrix A ∈Rm×d, we use ∥A∥to denote the Frobenius norm and ∥A∥op to denote
the operator-2 norm. If unspecified, we assume that the norm is the Frobenius norm. For normed
vector spaces A, B with norms ∥· ∥A and ∥· ∥B, respectively, and for α ≥0, we say a function
F : A →B is α-pseudo-Lipschitz with constant L if for any A, ˆA ∈A, we have

∥F(A) −F( ˆA)∥B ≤L∥A −ˆA∥A(1 + ∥A∥α
A + ∥ˆA∥α
A).

We write f(t) ≍g(t) if there exist absolute constants C, c > 0 such that c · g(t) ≤f(t) ≤C · g(t)
for all t. If the constants depend on parameters, e.g., α, then we write ≍α.

Related work.
Some notable adaptive learning rates in the literature are AdaGrad-Norm [32, 59, 61],
RMSprop [28], stochastic line search, stochastic Polyak stepsize [35], and more recently DoG [29]
and D-Adaptation [18]. In this work, we introduce a framework for analyzing these algorithms, and
we strongly believe it can be used to analyze many more of these adaptive algorithms. We highlight
below a nonexhaustive list of related work.

3


---Page Break---
AdaGrad-Norm.
AdaGrad, introduced by [19, 36], updates the learning rate at each iteration
using the stochastic gradient information. The single stepsize version [32, 59, 61], that depends
on the norm of the gradient, (see Table 2 for the updates), has been shown to be robust to input
parameters [34]. Several works have shown worst-case convergence guarantees [21, 33, 57, 59]. A
linear rate of O(exp(−κT)) is possible for µ-strongly convex, L-smooth functions (κ is the condition
number µ/L). In [62] (similar idea in [61]), the authors show for strongly convex, smooth stochastic
objectives (with additional assumptions) that the AdaGrad-Norm learning rate exhibits a two stage
behavior – a burn in phase and then when it reaches the smoothness constant it self-stablizes.

Stochastic line search and Polyak stepsizes. Recently there has been renewed interest in studying
stochastic line search [20, 42, 54] and stochastic Polyak stepsize (and their variants) [7, 26, 27, 30,
35, 39, 41, 49]. Much of this research focuses on worst-case convergence guarantees for strongly
convex and smooth functions (see e.g., [35]) and designing practical algorithms. In [53], the authors
provide a bound on the learning rate for Armijo line search in the finite sum setting with a rate of
Lmax/avg. µ where avg. µ is the avg. strong convexity and Lmax is the max. Lipschitz constant
of the individual functions. In this work, we consider a slightly different problem. We work with
the population loss and we note that the analogue to Lmax for us would require that the samples a
satisfy ∥aaT ∥op ≤Lmax for all a; this fails to hold for a ∼N(0, K). Moreover, Lmax could be
much worse than E [∥aaT ∥op].

Deterministic dynamics of stochastic algorithms in high-dimensions. The literature on deter-
ministic dynamics for isotropic Gaussian data has a long history [9, 10, 50, 51]. These results
have been rigorously proven and extended to other models under the isotropic Gaussian assump-
tion [1, 2, 6, 16, 17, 23, 58]. Extensions to multi-pass SGD with small mini-batches [46] as well
as momentum [31] have also been studied. Other high-dimensional limits leading to a different
class of dynamics also exist [11–13, 22, 37]. Recently, significant contributions have been made
in understanding the effects of a non-identity data covariance matrix on the training dynamics
[5, 14, 15, 24, 25, 64]. The non-identity covariance modifies the optimization landscape and affects
convergence properties, as discussed in [15]. This work extends the findings of [15] to stochastic
adaptive algorithms, exploring the effect of non-identity covariance within these algorithms. Notably,
Theorem 1.1 from [15] is restricted to deterministic learning rate schedules, limiting its applicability
in many practical scenarios. In contrast, our Theorem 2.1 accommodates stochastic adaptive learning
rates, aligning with widely used algorithms in practice.

1.1
Model Set-up

We suppose that a sequence of independent samples {(ak, yk)} drawn from a distribution D ⊂Rd×R
is provided where yk is the target. The target yk is a function of some random label noise ϵk ∈R and
the input feature ak dotted with a ground truth signal X⋆∈Rd, ⟨ak, X⋆⟩. Therefore, the distribution
of the data is only determined by the input feature and the noise, i.e., the pair (a, ϵ). In particular, we
assume (a, ϵ) follows a distributional assumption.

Assumption 1 (Data and label noise). The samples (a, ϵ) ∼D are normally distributed: ϵ ∼
N(0, ω2) where ω ∈R, and a ∼N(0, K), with a covariance matrix K ∈Rd×d that is bounded in
operator norm independent of d; i.e., ∥K∥op ≤C. Furthermore, a and ϵ are independent.

For a, X, X⋆∈Rd, ϵ ∈R, and a function f : R3 →R, we seek to minimize an expected risk
function R : Rd →R, which we refer to as the high-dimensional linear composite4, of the form

R(X)
def
= E a,ϵ[Ψ(X; a, ϵ)]
for
(a, ϵ) ∼D,
and
Ψ(X; a, ϵ) = f(⟨a, X⟩, ⟨a, X⋆⟩, ϵ).
(2)

In what follows, we use the matrix W = [X|X⋆] ∈Rd×2 that concatenates X and X⋆, and we shall
let B = B(W) = W T KW be the covariance matrix of the Gaussian vector (⟨a, X⟩, ⟨a, X⋆⟩).

Assumption 2 (Pseudo-lipschitz f). The function f : R3 →R is α-pseudo-Lipschitz with α ≤1.

By assumption, R(X) involves an expectation over the correlated Gaussians ⟨a, X⟩and ⟨a, X⋆⟩. We
can express this as R(X)
def
= h(B) for some well-behaved function h : R2×2 →R.

4Note that d need not be large to define this, but the structure allows us to consider d as a tunable parameter.
Moreover, as we increase d, the analysis we do will be more meaningful.

4


---Page Break---
Table 2: Two adaptive learning rates considered in detail. The stochastic adaptive
learning rate, gk, is the learning rate directly used in the update for SGD whereas the
deterministic, γt, is the deterministic equivalent of gk after scaling.

Algorithm
General update
Least squares

AdaGrad-
Norm(b, η)
b0 = b × d

gk
b2
k = b2
k−1 + ∥∇Ψ(Xk−1)∥2;
gk−1 = d ×
η
|bk|
same

γt
η
q

b2+ Tr(K)

d
R t
0 I(B(s)) ds

η
q

b2+ 2Tr(K)

d
R t
0 R(s) ds

Exact line
search
(idealized)

gk
∥∇R(Xk)∥2

Tr(∇2R(Xk)K)

d
Ea,ϵ[(f ′(⟨a,X⟩;⟨a,X⋆⟩,ϵ))2]

∥∇R(Xk)∥2

2Tr(K2)

d
R(Xk)

γt
arg min
γ
dR(t)

Pd
i=1 λ2
i D2
i (t)
2Tr(K2)R(t)

Assumption 3 (Risk representation). There exists a function h : R2×2 →R such that h(B) = R(X)
is differentiable and satisfies
∇XR(X) = Ea,ϵ∇XΨ(X; a, ϵ).
Furthermore, h is continuously differentiable and its derivative ∇h is α-pseudo-Lipschitz for some
0 ≤α ≤1, with constant L(∇h).

The final assumption is the well-behavior of the Fisher information matrix of the gradients. The first
coordinate of f is special, as the optimizer must be able to differentiate it. Thus, we treat f(x, x⋆, ϵ)
as a function of a single variable with two parameters: f(x, x⋆, ϵ) = f(x; x⋆, ϵ) and denote the
(almost everywhere) derivative with respect to the first variable as f ′.

Assumption 4 (Fisher matrix). Define I(B)
def
= Ea,ϵ[(f ′(⟨a, X⟩; ⟨a, X⋆⟩, ϵ))2] where the function
I : R2×2 →R. Furthermore, I is α-pseudo-Lipschitz with constant L(I) for some α ≤1.

A large class of natural regression problems fit within this framework, such as logistic regression and
least squares (see [15, Appendix B]). We also note that Assumptions 3 and 4 are nearly satisfied for
L-smooth objectives f (see Lemma B.1), and a version of the main theorem holds under just this
assumption (albeit with a weaker conclusion).

1.2
Algorithmic set-up

We apply one-pass or streaming SGD with an adaptive learning rate gk (SGD+AL) to solve
minX∈Rd R(X), (2). Let X0 ∈Rd be an initial vector (random or non-random). Then SGD+AL
iterates by selecting a new data point (ak+1, ϵk+1) such that ak+1 ∼N(0, K) and ϵk+1 ∼N(0, ω2)
and makes the update

Xk+1 = Xk−gk

d ·∇XΨ(Xk; ak+1, ϵk+1) = Xk−gk

d f ′(⟨ak+1, Xk⟩; ⟨ak+1, X⋆⟩, ϵk+1)ak+1, (3)

where gk > 0 is a learning rate (see assumptions below)5. To perform our analysis, we place the
following assumption on the initialization X0 and signal X⋆.
Assumption 5 (Initialization and signal). The initialization point X0 and the signal X⋆are bounded
independent of d, that is, max{∥X0∥, ∥X⋆∥} ≤C for some C independent of d.

Adaptive learning rate.
Our analysis requires some mild assumptions on the learning rate. To this
end, we define a learning rate function γ : R+ × D([0, ∞)) × D([0, ∞)) × D([0, ∞)) →R+ by6

gk
def
= γ(k, Nk(d × ·), Gk(d × ·), Qk(d × ·)), for k ∈N, where for any t ≥0,

(Nk(t), Gk(t), Qk(t))
def
= 1{t<k}

(Wt)T Wt, 1

d∥∇XΨ(Xt; at+1, ϵt+1)∥2, R(Xt)

.
(4)

5Note that cases where Tr(K2)

d
= o(d) can lead to dynamics that converge to full-batch gradient flow. While

our theorem specifically addresses the scenario where the intrinsic dimension, Dim(K)
def
= Tr(K)/∥K∥op,
satisfies Dim(K) = Θ(d), other cases, such as Dim(K) = o(d), may require different learning rate scalings.
6D([0, ∞)) is the càdlàg function class on [0, ∞).

5


---Page Break---
In this definition, for functions taking integer arguments, we extend them to real-valued inputs by
first taking the floor function of its argument. Note that the adaptive learning rates can depend on the
whole history of stochastic iterates (Nk), gradients (Gk), and risk (Qk) via this definition.

We also define a conditional expectation version of Gk where the filtration Fk = σ(X⋆, X0, . . . , Xk):

Gk(t)
def
= 1{t<k}(·)1

dE[∥∇XΨ(Xt; at+1, ϵt+1)∥2|Ft]
for t ≥0.

With this, we impose the following learning rate condition.
Assumption 6 (Learning rate). The learning rate function γ : R+ × D([0, ∞)) × D([0, ∞)) ×
D([0, ∞)) →R is α-pseudo-Lipschitz with constant L(γ) (independent of d) in D([0, ∞)) ×
D([0, ∞)) × D([0, ∞)). Moreover, for some constant C = C(γ) > 0 independent of d and δ > 0,

E [|γ(k, f, Gk(d ×·), q) −γ(k, f, Gk(d ×·), q)| | Fk] ≤Cd−δ(1 + ∥f∥α
∞+ ∥q∥α
∞)
w.o.p.
(5)

Finally, γ is bounded, i.e., there exists a constant ˆC = ˆC(γ) > 0 independent of d so that

γ(k, f, g, q) ≤ˆC(1 + ∥f∥α
∞+ ∥q∥α
∞+ ∥g∥α
∞).
(6)

The inequality (5) ensures that the learning rate concentrates around the mean behavior of the
stochastic gradients. Many well-known adaptive stepsizes satisfy (4) and Assumption 6 including
AdaGrad-Norm, DoG, D-Adaptation, and RMSProp (see Table 2, Sec. A, and Sec. C.3).

2
Deterministic dynamics for SGD with adaptive learning rates

Intuition for deriving dynamics:
The risk R(X) and Fisher matrix can be evaluated solely in
terms of the covariance matrix B. Thus, to know the evolution of the risk over time, it would suffice
to know the evolution of B. Alas, except in the isotropic case where K is a multiple of the identity,
the evolution of B is not autonomous (i.e., its time evolution depends on other unknown variables).
However, if we let (λi, ωi) be the eigenvalues and corresponding orthonormal eigenvectors of K, we
can consider projections Vi(Xk) = d · W T
k ωiωT
i Wk, and it turns out that these behave autonomously.

Example: Least Squares.
One canonical example of (2) is least squares, where we aim to recover
the target X⋆given noisy observations ⟨a, X⋆⟩+ ϵ. In this case, the least squares problem is

min
X∈Rd

n
R(X) = 1

2E a,ϵ[(⟨a, X −X⋆⟩−ϵ)2] = 1

2ω2 + 1

2(X −X⋆)T K(X −X⋆)
o
.
(7)

The pair of functions h (Assumption 3) and I (Assumption 4) can be evaluated simply:

h(B(W)) = 1

2I(B(W)) = 1

2(X −X⋆)T K(X −X⋆) + 1

2ω2.

The deterministic dynamics for the risk R(t) in this case can be simplified to:

R(t) = 1

2(X0 −X⋆)T Ke−2K
R t
0 γs ds(X0 −X⋆) + 1

2ω2 + 1

d
R t
0 γ2
sTr(K2e−2K
R t
s γτ dτ)R(s) ds.

This is a convolution Volterra equation with a convergence threshold of γt <
2d
TrK [14, 44, 46, 47].

In the noiseless label case (i.e., ϵ = 0), the risk is given by R(t) =
1
2d
Pd
i=1 λiD2
i (t). Using the
ODEs in (9), we get the following deterministic equivalent ODE for the D2
i ’s:

d
dtD2
i (t) = −2γtλiD2
i (t) + 2γ2
t λiR(t).
(8)

We will perform a deep analysis of the dynamics of the learning rate on least squares (7), which will
generalize to settings where the outer function f is strongly convex (see D.1).

Deterministic dynamics.
To derive deterministic dynamics, we make the following change to
continuous time by setting

k iterations of SGD = ⌊td⌋,
where t ∈R is the continuous time parameter.

This time change is necessary, as when we scale the size of the problem, more time is needed to
solve the underlying problem. This scaling law scales SGD so all training dynamics live on the same

6


---Page Break---
space. One can solve a smaller d problem and scale it to recover the training dynamics of the larger
problem.7

We now introduce a coupled system of differential equations, which will allow us to model the
behaviour of our learning algorithms. For the ith (λi, ωi)-eigenvalue/eigenvector of K, set

Vi(t)
def
=
V11,i(t)
V12,i(t)
V12,i(t)
V22,i(t)


and averaging over i, B(t)
def
= 1

d

d
X

i=1
λiVi(t).

The Vi(t) and B(t) are deterministic continuous analogues of Vi(Xtd) and B(Xtd) respectively.
Define the following continuous analogues

∇h(B(t))
def
=
H1,t
H2,t
H2,t
H3,t


, N (t)
def
= 1

d

d
X

i=1
Vi(t), R(t)
def
= h(B(t)),
I (t)
def
= I(B(t)),

and finally γt
def
= γ(t, 1{·≤t}N (·), Tr(K)

d
1{·≤t}I (·), 1{·≤t}R(·)).

We now introduce a system of coupled ODEs for each (λi, ωi)-eigenvalue/eigenvector pair of K

dV11,i(t) = −2λiγt (V11,i(t)H1,t + H1,tV11,i(t) + V12,i(t)H2,t + H2,tV12,i(t)) + λiγ2
t I (t),
dV12,i(t) = −2λiγt (H1,tV12,i(t) + H2,tV22,i(t))
(9)
with the initialization of Vi(0) given by Vi(X0). We finally state the deterministic dynamics for the
risk and learning rate.
Theorem 2.1. Under Assumptions 1, 2, 3, 4, 5, 6, then for any ε ∈(0, 1

2) and any T > 0

sup
0≤t≤T


R(X⌊td⌋)
g⌊td⌋


−
R(t)
γt

  < d−ε,
w.o.p.
(10)

The same statements hold comparing W T
tdWtd to N (t) and W T
tdKWtd to B(t).

In fact, we can derive deterministic dynamics for a large class of statistics which are linear combina-
tions of V (t) and functions thereof (See Theorem B.1, and Corollary B.1).

One important corollary is a deterministic limit for the distance to optimality, D2(Xk) = ∥Xk−X⋆∥2,
which is a quadratic form of W T
k Wk and hence covered by Thm. 2.1. The equivalent deterministic
dynamics are

D2(t) = 1

d

d
X

i=1
D2
i (t) = 1

d

d
X

i=1
(V11,i(t) −2V12,i(t) + V22,i(t)),
(11)

where D2
i (t) corresponds D2
i (Xk)
def
= d × (⟨Xk −X⋆, ωi⟩)2.

3
Idealized Exact Line Search and Polyak Stepsize

In this section, we consider two classical idealized algorithms – exact line search and Polyak stepsize.
In deterministic optimization, these learning rate strategies are chosen so that the function value
(exact line search) or distance to optimality (Polyak) produces the largest decrease in function value
(resp. distance to optimality) at the next iteration. For stochastic algorithms, we can ask this to
hold for the deterministic equivalent to the risk R(t) (resp. distance to optimality, D(t)) since we
know that SGD is close to these deterministic equivalents. Thus, the question is: what choice of
learning rate decreases the R(t) (exact line search) and/or D(t) (Polyak stepsize)? We will restrict to
least squares in this section – see Appendix F.1 and F.2 for general functions as well as proofs for
least squares. These are idealized algorithms because we can not implement them as they require
distributional knowledge of a or X⋆. Despite this, they provide a basis for more practical algorithms.

7Note that, holding time fixed, we perform O(d) gradient updates for a problem of dimension d. For the
problems considered here, this scaling leads to consistent dynamics, but there do exist related problems where a
different scaling is more appropriate. For example, under random initialization, to capture the escape of phase
retrieval from the high-dimensional saddle, O(d log d) iterations are needed; see for example [56].

7


---Page Break---
Figure 2: Comparison for Exact Line Search and Polyak Stepsize on a noiseless least squares
problem. The left plot illustrates the convergence of the risk function, while the right plot depicts
the convergence of the quotient γt/ λmin(K)

1
d Tr(K2) for Polyak stepsize and exact line search. Both plots
highlight the implication of equation (13) in high-dimensional settings, where a broader spectrum
of K results in λmin(K)

1
d Tr(K2) ≪
1
1
d Tr(K), indicating slower risk convergence and poorer performance

of exact line search (unmarked) as it deviates from the Polyak stepsize (circle markers) . The gray
shaded region demonstrates that equation (13) is satisfied. See Appendix H for simulation details.

Polyak Stepsize.
A natural threshold to consider is the largest learning rate such that dD(t) < 0,
which we denote by ¯γD
t . Using the least squares ODE (8), this is precisely

¯γD
t = (2R(t)−ω2)

Tr(K)

d
R(t)
and
¯gD
k = (2R(Xk)−ω2)

Tr(K)

d
R(Xk)
.
(12)

Without label noise, (12) simplifies to ¯γD
t = ¯gD
k =
2
Tr(K)/d, the exact threshold for convergence of
least squares.

A greedy stepsize strategy would maximize the decrease in the distance to optimality at each iteration,
denoted by us as Polyak stepsize, γPolyak
t
∈arg minγ dD(t). In the case of least squares, this is

γPolyak
t
= 1

2 ¯γD
t
and
gPolyak
k
= 1

2¯gD
k .

The latter yields the optimal fixed learning rate (up to absolute constant factors) for a noiseless target
on a least squares problem [35, 43]).8

Exact Line Search.
In the context of risk, using (8) and noting that R(t) =
1
2d
Pd
i=1 λiD2
i (t), we
can find γline
t
∈arg min dR(t); i.e., the greedy learning rate that decreases the risk the most in the
next iteration. We call this exact line search. Expressions for the learning rates are given in Table 2,
(c.f. Appendix F.1 for general losses). Because these come from ODEs, we can use ODE theory to
give exact limiting values for the deterministic equivalent of gline
k .
Proposition 3.1. [Limiting learning rate; line search on noiseless least squares] Consider the
noiseless (ω = 0) least squares problem (7) . Then the learning rate is always lower bounded by

λmin(K)
1
d Tr(K2) ≤γline
t
for all t ≥0.

Moreover, suppose K has only two distinct eigenvalues λ1 > λ2 > 0, i.e., K has d/2 eigenvalues
equal to λ1 eigenvalues and d/2 eigenvalues equal to λ2. Then

λmin(K)
1
d Tr(K2) ≤lim
t→∞γline
t
≤2λmin(K)

1
d Tr(K2).
(13)

For a proof and explicit formula for limt→∞γline
t
, see Section F.2. Hence, being greedy for the risk
in a sufficiently anisotropic setting will badly underperform Polyak stepsize (see Fig. 2).

8The Polyak stepsize we analyze in this paper differs slightly from the "classic" stepsize in the literature, that
is, R(Xk)−R(X∗)

||∇R(Xk)||2 . Rather than using this form, we skip an approximation step in the derivation [27] and use the
exactly optimal form. Both variations of the Polyak stepsize can be analyzed under our assumptions; the choice
was admittedly somewhat arbitrary. (Note that in the case of least squares, the two stepsizes coincide.)

8


---Page Break---
100
102
104
106

SGD iterations

5 × 10
1

6 × 10
1

7 × 10
1

8 × 10
1

Risk

(Noisy) Least Squares AdaGrad, 
=1.0, b = 1.0,  = 2.5, d = 500

100

8.8 × 10
1

9 × 10
1

9.2 × 10
1

9.4 × 10
1

9.6 × 10
1

9.8 × 10
1

Learning rate / t^(-1/2)

SGD
Predicted

100
101
102
103
104
105
SGD iterations

100

2 × 100

3 × 100

4 × 100

learning rate

SGD, =719.69
SGD, =100.00
SGD, =51.79
SGD, =31.62
SGD, =12.33

100
101
102
103
104
105
SGD iterations

100

learning rate

SGD, ||X0
X * ||2=1.0

SGD, ||X0
X * ||2=2.0

SGD, ||X0
X * ||2=4.0

SGD, ||X0
X * ||2=8.0

SGD, ||X0
X * ||2=16.0

Figure 3: Quantities effecting AdaGrad-Norm learning rate. (left): Effect of noise (ω = 1.0) on
risk (left axis) and learning rate (right axis). Depicted is learning rate

asymptotic so it approaches 1. (Center, right):
Noiseless least squares (ω = 0). As predicted in Prop. 4.2, limt→∞γt depends on avg. eig. of K
(Tr(K)/d) and ∥X0 −X⋆∥2 but not κ = λmax/λmin. See Appendix H for simulation details.

4
AdaGrad-Norm analysis

In this section, we analyze the behavior of AdaGrad-Norm learning rate in the least squares setting
(see Sec. D for general strongly convex functions). In the presence of additive noise, the AdaGrad-
Norm learning rate decays like t−1/2, regardless of the data covariance K. In contrast, the model
with no noise exhibits a learning rate that depends on the spectrum of K, as illustrated in Figure
3. The learning rate is bounded below by a constant when λmin(K) > 0 is fixed as d →∞, and
we quantify this lower bound. If the limiting spectral measure of K has unbounded density near 0
(e.g. power law spectrum), then the learning rate can approach zero and we quantify the rate of this
convergence in the least squares setting as a function of spectral parameters.

For least squares with additive noise, the learning rate asymptotic γt ≍η/(b2 + ω2

d Tr(K)t)(1/2) is
the fastest decay that AdaGrad-Norm can exhibit. In contrast, the propositions below concern the
noiseless case where, for various covariance examples, the decay rate of γt changes. This is tightly
connected to whether the risk is integrable or not. In the simple case of identity covariance, we obtain
a closed formula for the trajectory of the integral of the risk and therefore also the learning rate.
Proposition 4.1. In the case of identity covariance (K = Id), the risk solves the differential equation

d
dtR(t) =
η2R(t)
b2+2
R t
0 R(s) ds −
2ηR(t)
√

b2+2
R t
0 R(s) ds,
(14)

The solution
R t
0 R(s) ds approaches (from below) a positive constant which yields a computable
lower bound to which γt will converge. Generalizing this to a broader class of covariance matrices,
we get the next proposition, which captures the dependence of γt on Tr(K).

Proposition 4.2. Suppose 1

d Tr(K) ≤b/η, and that
R ∞
0
R(s)γs ds < ∞with γs as in Table 2
(AdaGrad-Norm for least squares), then γt ≍
1
b
η + η2

4d Tr(K)D2(0).

An analog of Proposition 4.2 for the strongly convex setting appears in Sec. D (see Prop. D.1). We
now consider two cases in which, as d →∞, there are eigenvalues of K arbitrarily close to 0.
Proposition 4.3. Assume that, for some C > 0, the number of eigenvalues of K below C is o(d),
and that ⟨X⋆, ωi⟩= O(d−1/2) for all i, (i.e. X⋆is not concentrated in any eigenvector direction).
Then, with the initialization X0 = 0, there exists some ˜γ > 0 such that γt > ˜γ for all t > 0.

Proposition 4.4. Let K have a spectrum that converges as d →∞to the power law measure
ρ(λ) = (1 −β)λ−β1(0,1), for some β < 19, and suppose that D2
i (0) ∼λ−δ
i
for δ ≥0. Then:

• For 1 > β + δ, there exists ˜γ such that γt ≥˜γ, and R(t) ≍δ,β tβ+δ−2 for all t ≥1.

• For 1 < β + δ < 2, γt ≍δ,β t
−1+
1
β+δ
δ,β
, and R(t) ≍δ,β t−
2
β+δ +1 for all t ≥1.

9Our result can be compared to existing findings for SGD under power-law distributions in [8, 52, 55]. While
these works explore similar assumptions regarding the covariance matrix spectrum, they do not address the
high-dimensional regime with diverging Tr(K), focusing primarily on β > 1.

9


---Page Break---
10
1
100
101
102
103
104
105
106
107
time (t)

10
4

10
2

100

102

104

106

risk

t^(beta + delta - 2)
delta + beta = 0.2
delta + beta = 0.4
delta + beta = 0.6
delta + beta = 0.8
delta + beta = 1.0
delta + beta = 1.2
delta + beta = 1.4
delta + beta = 1.6
delta + beta = 1.8
t^(1-2/(beta + delta))
delta + beta = 2.0

10
5
10
3
10
1
101
103
105
107
time (t)

10
6

10
5

10
4

10
3

10
2

10
1

100

learning rate

constant
delta + beta = 0.2
delta + beta = 0.4
delta + beta = 0.6
delta + beta = 0.8
delta + beta = 1.0
delta + beta = 1.2
delta + beta = 1.4
delta + beta = 1.6
delta + beta = 1.8
delta + beta = 2.0
t^(-1 + 1/(beta+delta))

Figure 4: Power law covariance in AdaGrad Norm on a least squares problem. Ran exact
predictions (ODE) for the risk and learning rate (solid lines). Dashed lines give the predictions
from Prop. 4.4 which match experimental results exactly. Phase transition as δ + β varies. When
δ + β < 1 (green), the learning rate (right) is constant as t →∞. In contrast, when 2 > δ + β > 1
(purple), the learning rate decreases at a rate t−1+1/(β+δ) with δ + β = 1 (white) where the change
occurs. Same phase transition occurs in the sublinear rate of the risk decay (left) (see Prop. 4.4).

• For 1 = β + δ, γt ≍δ,β
1
log(t+1), and R(t) ≍δ,β (
t
log(t+1))−1 for all, t ≥1.

This proposition shows non-trivial decay of the learning rate is dictated by the residuals (distance
to optimality at initialization) and the spectrum of K. We note that δ = 0 corresponds to uniform
contribution of each mode (e.g. X0 normally distributed). As the eigenmodes of the residuals become
more localized, the decay of the learning rate is closer to the behaviour in the presence of additive
noise. Furthermore, the scaling behaviour of the loss is affected by the structure of the AdaGrad-Norm
algorithm (see Fig. 4). Lastly, constant stepsize SGD yields R(t) ≍tβ+δ−2, with no transition
occurring at β + δ = 1.

Proofs of the above propositions, in a slightly more general setting, are deferred to Sec. D.

5
Conclusions and Limitations

This work studies stochastic adaptive optimization algorithms when data size and parameter size are
large, allowing for nonconvex and nonlinear risk functions, as well as data with general covariance
structure. The theory shows a concentration of the risk, the learning rate and other key functions
to a deterministic limit, which is described by a set of ODEs. The theory is then used to derive the
asymptotic behavior of the AdaGrad-Norm and idealized exact line search on strongly convex and
least square problems, revealing the influence of the covariance matrix structure on the optimization.
A potential extension of this work would be to study other adaptive algorithms such as D-adaptation,
DOG, and RMSprop which are covered by the theory. Studying the asymptotic behavior of the risk
and the learning rate may improve our understanding of the performance and scalability of these
algorithms on more realistic data. Another important application of the theory would be to analyze
the ODEs presented here on nonconvex problems.

The current form of the theory is limited to Gaussian data, though many parts of the proof can be
extended easily beyond Gaussian data. The main ODE comparison theorem is also only tuned for
analyzing problem setups where the trace of the covariance is on the order of the ambient dimension;
when the trace of the covariance is much smaller than ambient dimension, other stepsize scalings of
SGD are needed. In addition, the analysis is limited to the streaming stochastic adaptive methods.
We conjecture that a similar deterministic equivalent holds also for multi-pass algorithms at least for
convex problems. This has already been shown in the least square problem for SGD with a fixed
deterministic learning rate [43, 45]. Lastly, numerical simulations on real datasets (e.g., CIFAR-5m)
suggests that the predicted risk derived by our theory matches the empirical risk of multipass SGD
beyond Gaussian data (see for example Figure 6).

10


---Page Break---
Acknowledgments and Disclosure of Funding

E. Collins-Woodfin was supported by Fonds de recherche du Québec – Nature et technologies
(FRQNT) postdoctoral training scholarship and Centre de recherches mathématiques (CRM) Applied
math postdoctoral fellowship. Research of B. García Malaxechebarría was in part funded by NSF
DMS 2023166 (NSF TRIPODS II). Research by E. Paquette was supported by a Discovery Grant
from the Natural Science and Engineering Council (NSERC). C. Paquette is a Canadian Institute for
Advanced Research (CIFAR) AI chair, Quebec AI Institute (MILA) and a Sloan Research Fellow in
Computer Science (2024). C. Paquette was supported by a Discovery Grant from the Natural Science
and Engineering Research Council (NSERC) of Canada, NSERC CREATE grant Interdisciplinary
Math and Artificial Intelligence Program (INTER-MATH-AI), Google research grant, and Fonds
de recherche du Québec – Nature et technologies (FRQNT) New University Researcher’s Start-Up
Program. Additional revenues related to this work: C. Paquette has 20% part-time employment at
Google DeepMind.

References

[1] Luca Arnaboldi, Florent Krzakala, Bruno Loureiro, and Ludovic Stephan. Escaping medi-
ocrity: how two-layer networks learn hard single-index models with SGD. arXiv preprint
arXiv:2305.18502, 2023.

[2] Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro.
From high-
dimensional and mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in
two-layers networks. arXiv preprint arXiv:2302.05882, 2023.

[3] Krishna B Athreya, Peter E Ney, and PE Ney. Branching processes. Courier Corporation, 2004.

[4] Amit Attia and Tomer Koren. SGD with adagrad stepsizes: full adaptivity with high probability
to unknown parameters, unbounded gradients and affine variance. In International Conference
on Machine Learning, pages 1147–1171. PMLR, 2023.

[5] Krishnakumar Balasubramanian, Promit Ghosal, and Ye He. High-dimensional scaling lim-
its and fluctuations of online least-squares SGD with smooth covariance. arXiv preprint
arXiv:2304.00707, 2023.

[6] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. High-dimensional limit theorems
for SGD: Effective dynamics and critical scaling. In Advances in Neural Information Processing
Systems, volume 35, pages 25349–25362, New York, 2022. Curran Associates, Inc.

[7] Leonard Berrada, Andrew Zisserman, and M Pawan Kumar. Training neural networks for and
by interpolation. In International conference on machine learning, pages 799–809. PMLR,
2020.

[8] R. Berthier, F. Bach, and P. Gaillard. Tight Nonparametric Convergence Rates for Stochastic
Gradient Descent under the Noiseless Linear Model. In Advances in Neural Information
Processing Systems (NeurIPS), 2020.

[9] Michael Biehl and Peter Riegler. On-line learning with a perceptron. Europhysics Letters, 28
(7):525, 1994.

[10] Michael Biehl and Holm Schwarze. Learning by on-line gradient descent. Journal of Physics A:
Mathematical and general, 28(3):643, 1995.

[11] B. Bordelon and C. Pehlevan. Learning Curves for SGD on Structured Features. In International
Conference on Learning Representations (ICLR), 2022.

[12] Michael Celentano, Chen Cheng, and Andrea Montanari. The high-dimensional asymptotics of
first order methods with random data. arXiv preprint arXiv:2112.07572, 2021.

[13] Kabir Aladin Chandrasekher, Ashwin Pananjady, and Christos Thrampoulidis. Sharp global
convergence guarantees for iterative nonconvex optimization with random data. Ann. Statist.,
51(1):179–210, 2023. ISSN 0090-5364,2168-8966. doi: 10.1214/22-aos2246. URL https:
//doi.org/10.1214/22-aos2246.

11


---Page Break---
[14] Elizabeth Collins-Woodfin and Elliot Paquette. High-dimensional limit of one-pass SGD on least
squares. Electronic Communications in Probability, 29:1–15, 2024. doi: 10.1214/23-ECP571.

[15] Elizabeth Collins-Woodfin, Courtney Paquette, Elliot Paquette, and Inbar Seroussi. Hitting
the high-dimensional notes: An ODE for SGD learning dynamics on GLMs and multi-index
models. arXiv preprint arXiv:2308.08977, 2023.

[16] Alex Damian, Eshaan Nichani, Rong Ge, and Jason D Lee.
Smoothing the landscape
boosts the signal for sgd:
Optimal sample complexity for learning single index mod-
els.
In Advances in Neural Information Processing Systems, volume 36, pages 752–
784, 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/
02763667a5761ff92bb15d8751bcd223-Paper-Conference.pdf.

[17] Yatin Dandi, Emanuele Troiani, Luca Arnaboldi, Luca Pesce, Lenka Zdeborová, and Florent
Krzakala. The benefits of reusing batches for gradient descent in two-layer networks: Breaking
the curse of information and leap exponents. arXiv preprint arXiv:2402.03220, 2024.

[18] Aaron Defazio and Konstantin Mishchenko. Learning-rate-free learning by d-adaptation. In
International Conference on Machine Learning, pages 7449–7479. PMLR, 2023.

[19] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning
and stochastic optimization. Journal of machine learning research, 12(7), 2011.

[20] Darina Dvinskikh, Aleksandr Ogaltsov, Alexander Gasnikov, Pavel Dvurechensky, Alexander
Tyurin, and Vladimir Spokoiny. Adaptive gradient descent for convex and non-convex stochastic
optimization. arXiv preprint arXiv:1911.08380, 2019.

[21] Matthew Faw, Litu Rout, Constantine Caramanis, and Sanjay Shakkottai. Beyond uniform
smoothness: A stopped analysis of adaptive SGD. In The Thirty Sixth Annual Conference on
Learning Theory, pages 89–160. PMLR, 2023.

[22] Cédric Gerbelot, Emanuele Troiani, Francesca Mignacco, Florent Krzakala, and Lenka Zde-
borová. Rigorous dynamical mean-field theory for stochastic gradient descent methods. SIAM
Journal on Mathematics of Data Science, 6(2):400–427, 2024. doi: 10.1137/23M1594388.
URL https://doi.org/10.1137/23M1594388.

[23] Sebastian Goldt, Madhu Advani, Andrew M Saxe, Florent Krzakala, and Lenka Zdeborová.
Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student
setup. Advances in neural information processing systems, 32, 2019.

[24] Sebastian Goldt, Marc Mézard, Florent Krzakala, and Lenka Zdeborová. Modeling the influence
of data structure on learning in neural networks: The hidden manifold model. Physical Review
X, 10(4):041044, 2020.

[25] Sebastian Goldt, Bruno Loureiro, Galen Reeves, Florent Krzakala, Marc Mézard, and Lenka
Zdeborová. The gaussian equivalence of generative models for learning with shallow neural
networks. In Mathematical and Scientific Machine Learning, pages 426–471, New York, New
York, USA, 2022. PMLR.

[26] Robert M Gower, Aaron Defazio, and Michael Rabbat. Stochastic polyak stepsize with a
moving target. arXiv preprint arXiv:2106.11851, 2021.

[27] Elad Hazan and Sham Kakade. Revisiting the polyak step size. arXiv preprint arXiv:1905.00313,
2019.

[28] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning
lecture 6a overview of mini-batch gradient descent. Cited on, 14(8):2, 2012.

[29] Maor Ivgi, Oliver Hinder, and Yair Carmon. DoG is SGD’s best friend: A parameter-free
dynamic step size schedule. arXiv preprint arXiv:2302.12022, 2023.

[30] Xiaowen Jiang and Sebastian U Stich. Adaptive SGD with polyak stepsize and line-search:
Robust convergence and variance reduction. Advances in Neural Information Processing
Systems, 36, 2024.

12


---Page Break---
[31] Kiwon Lee, Andrew N. Cheng, Courtney Paquette, and Elliot Paquette. Trajectory of Mini-
Batch Momentum: Batch Size Saturation and Convergence in High Dimensions. To Appear in
NeurIPS 2022, art. arXiv:2206.01029, June 2022.

[32] Kfir Levy. Online to offline conversions, universality and adaptive minibatch sizes. Advances in
Neural Information Processing Systems, 30, 2017.

[33] Kfir Y Levy, Alp Yurtsever, and Volkan Cevher. Online adaptive methods, universality and
acceleration. Advances in neural information processing systems, 31, 2018.

[34] Xiaoyu Li and Francesco Orabona. On the convergence of stochastic gradient descent with
adaptive stepsizes. In Proceedings of the Twenty-Second International Conference on Artificial
Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages
983–992, 2019.

[35] Nicolas Loizou, Sharan Vaswani, Issam Hadj Laradji, and Simon Lacoste-Julien. Stochastic
polyak step-size for SGD: An adaptive learning rate for fast convergence. In International
Conference on Artificial Intelligence and Statistics, pages 1306–1314. PMLR, 2021.

[36] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex
optimization. arXiv preprint arXiv:1002.4908, 2010.

[37] F. Mignacco, F. Krzakala, P. Urbani, and L. Zdeborová. Dynamical mean-field theory for
stochastic gradient descent in gaussian mixture classification. In Advances in Neural Information
Processing Systems, volume 33, pages 9540–9550, 2020.

[38] P. Nakkiran, B. Neyshabur, and H. Sedghi. The Deep Bootstrap Framework: Good Online Learn-
ers are Good Offline Generalizers. In International Conference on Learning Representations
(ICLR), 2021.

[39] Angelia Nedi´c and Dimitri Bertsekas. Convergence rate of incremental subgradient algorithms.
Stochastic optimization: algorithms and applications, pages 223–264, 2001.

[40] J. Nocedal and S. J. Wright. Numerical Optimization. Springer New York, 2 edition, 2006.

[41] Antonio Orvieto, Simon Lacoste-Julien, and Nicolas Loizou. Dynamics of SGD with stochastic
polyak stepsizes: Truly adaptive variants and convergence to exact solution. Advances in Neural
Information Processing Systems, 35:26943–26954, 2022.

[42] C. Paquette and K. Scheinberg. A stochastic line search method with expected complexity
analysis. SIAM J. Optim., 30(1):349–376, 2020. ISSN 1052-6234. doi: 10.1137/18M1216250.
URL https://doi.org/10.1137/18M1216250.

[43] C. Paquette, K. Lee, F. Pedregosa, and E. Paquette. SGD in the Large: Average-case Analysis,
Asymptotics, and Stepsize Criticality. In Proceedings of Thirty Fourth Conference on Learning
Theory (COLT), volume 134, pages 3548–3626, 2021.

[44] Courtney Paquette and Elliot Paquette. Dynamics of stochastic momentum methods on large-
scale, quadratic models. In Advances in Neural Information Processing Systems, volume 34,
pages 9229–9240, 2021. URL https://proceedings.neurips.cc/paper/2021/file/
4cf0ed8641cfcbbf46784e620a0316fb-Paper.pdf.

[45] Courtney Paquette and Elliot Paquette. High-dimensional optimization. SIAM Views and News,
20:16pp, December 2022.

[46] Courtney Paquette, Elliot Paquette, Ben Adlam, and Jeffrey Pennington. Homogenization of
SGD in high-dimensions: Exact dynamics and generalization properties. arXiv e-prints, art.
arXiv:2205.07069, May 2022.

[47] Courtney Paquette, Elliot Paquette, Ben Adlam, and Jeffrey Pennington. Implicit regularization
or implicit conditioning? exact risk trajectories of sgd in high dimensions. In Advances in
Neural Information Processing Systems, volume 35, pages 35984–35999, New York, 2022.
Curran Associates, Inc.

13


---Page Break---
[48] Boris T Polyak. Introduction to optimization. 1987.

[49] Michal Rolinek and Georg Martius. L4: Practical loss-based stepsize adaptation for deep
learning. Advances in neural information processing systems, 31, 2018.

[50] David Saad and Sara Solla. Dynamics of on-line gradient descent learning for multilayer neural
networks. In Advances in Neural Information Processing Systems, volume 8. MIT Press, 1995.

[51] David Saad and Sara A Solla. Exact solution for on-line learning in multilayer neural networks.
Physical Review Letters, 74(21):4337, 1995.

[52] A. Varre, L. Pillaud-Vivien, and N. Flammarion. Last iterate convergence of SGD for Least-
Squares in the Interpolation regime. In Advances in Neural Information Processing Systems
(NeurIPS), 2021.

[53] S. Vaswani, A. Mishkin, I. Laradji, M. Schmidt, G. Gidel, and S. Lacoste-Julien. Painless
Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates. In Advances in Neural
Information Processing Systems (NeurIPS), volume 32, pages 3732–3745, 2019.

[54] Sharan Vaswani, Issam Laradji, Frederik Kunstner, Si Yi Meng, Mark Schmidt, and Simon
Lacoste-Julien. Adaptive gradient methods converge faster with over-parameterization (but you
should do a line-search). arXiv preprint arXiv:2006.06835, 2020.

[55] M. Velikanov, D. Kuznedelev, and D. Yarotsky. A view of mini-batch SGD via generating
functions: conditions of convergence, phase transitions, benefit from negative momenta. In
International Conference on Learning Representations (ICLR), 2023.

[56] R. Vershynin. High-dimensional probability: An introduction with applications in data science.
Cambridge University Press, Cambridge, UK, 2018. doi: 10.1017/9781108231596. URL
https://doi.org/10.1017/9781108231596.

[57] Bohan Wang, Huishuai Zhang, Zhiming Ma, and Wei Chen. Convergence of adagrad for
non-convex objectives: Simple proofs and relaxed assumptions. In The Thirty Sixth Annual
Conference on Learning Theory, pages 161–190. PMLR, 2023.

[58] Chuang Wang, Hong Hu, and Yue Lu. A solvable high-dimensional model of GAN. In Advances
in Neural Information Processing Systems, volume 32, New York, 2019. Curran Associates, Inc.

[59] Rachel Ward, Xiaoxia Wu, and Leon Bottou. Adagrad stepsizes: Sharp convergence over
nonconvex landscapes. The Journal of Machine Learning Research, 21(1):9047–9076, 2020.

[60] Alexander Wei, Wei Hu, and Jacob Steinhardt. More than a toy: Random matrix models predict
how real-world neural representations generalize. In International Conference on Machine
Learning, pages 23549–23588. PMLR, 2022.

[61] Xiaoxia Wu, Rachel Ward, and Léon Bottou. Wngrad: Learn the learning rate in gradient
descent. arXiv preprint arXiv:1803.02865, 2018.

[62] Yuege Xie, Xiaoxia Wu, and Rachel Ward. Linear convergence of adaptive stochastic gradient
descent. In Proceedings of the Twenty Third International Conference on Artificial Intelligence
and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 1475–1485,
2020.

[63] Junchi Yang, Xiang Li, Ilyas Fatkhullin, and Niao He. Two sides of one coin: the limits of
untuned SGD and the power of adaptive methods. Advances in Neural Information Processing
Systems, 36, 2024.

[64] Yuki Yoshida and Masato Okada. Data-dependence of plateau phenomenon in learning with
neural network—statistical mechanical analysis. In Advances in Neural Information Processing
Systems, volume 32, New York, 2019. Curran Associates, Inc.

14


---Page Break---
The High Line: Exact Risk and Learning Rate Curves of
Stochastic Adaptive Learning Rate Algorithms

Supplementary material

Broader Impact Statement. The work presented in this paper is foundational research and it is not
tied to any particular application. The set-up is on a simple well-studied high-dimensional linear
composites (e.g., least squares, logistic regression, phase retrieval) with synthetic data and solved
using known algorithms, e.g., AdaGrad-Norm. We present deterministic dynamics for the training
loss and adaptive stepsizes. The results are theoretical and we do not anticipate any direct ethical
and societal issues. We believe the results will be used by machine learning practitioners and we
encourage them to use it to build a more just, prosperous world.

A
SGD adaptive learning rate algorithms and stepsizes

In this section, we write down the explicit update rules for 2 different adaptive stochastic gradient
descent algorithms.

Example: AdaGrad-Norm.
We begin with AdaGrad-Norm (see Algorithm 1). Note by unraveling
the recursion, we have that

gk =
η
q

b2 + 1

d2
Pk
j=0 ∥∇XΨ(Xj; aj+1, ϵj+1)∥2
,
(15)

with the deterministic equivalent (see Section 2 and also C.3) for this learning rate being

γt =
η
q

b2 + Tr(K)

d
R t
0 I(B(s)) ds
.
(16)

In the case of the least squares problem, the quantity I(B(t)) is explicit and

γt =
η
q

b2 + 2Tr(K)

d
R t
0 R(s) ds
.
(17)

Algorithm 1 AdaGrad-Norm
Require: Initialize η > 0, X0 ∈Rd, b ∈R and set b0 = b × d

for k = 1, 2, . . . , do

Generate new sample ak ∼N(0, K), ϵk ∼N(0, ω2);
b2
k ←b2
k−1 + ∥∇XΨ(Xk−1; ak, ϵk)∥2;
gk−1 = d ×
η
|bk|;
▷updating learning rate
Xk ←Xk−1 −gk−1

d ∇XΨ(Xk−1; ak, ϵk);
▷updating step with stochastic gradient
end for

Example: RMSprop-Norm
We consider the "normed" version of RMSprop, that is, where there is
only one learning rate parameter.

We consider Algorithm 2 where we put a factor of the learning into the exponential moving average
for RMSprop. The deterministic equivalent for gk for Alg. 2 (see Section 2) is

γt =
η
q

b2e−αt + Tr(K)

d
R t
0 e−α(t−s)I(B(s)) ds
.
(18)

In the case of the least squares problem, the quantity I(B(t)) is explicit and

γt =
η
q

b2e−αt + 2Tr(K)

d
R t
0 e−α(t−s)R(s) ds
.
(19)

15


---Page Break---
Algorithm 2 RMSprop-Norm, α Exponential Moving Average

Require: Initialize η > 0, X0 ∈Rd, b ∈R and set b0 = d × b, α > 0 exponential moving avg.

g−1 = d × η

b0 ;
for k = 1, 2, . . . , do

Generate new sample ak ∼N(0, K), ϵk ∼N(0, ω2);
b2
k ←α · b2
k−1 + (1 −α)∥∇XΨ(Xk−1; ak, ϵk)∥2;
gk−1 = d ×
η
|bk|;
▷updating learning rate
Xk ←Xk−1 −gk−1

d ∇XΨ(Xk−1; ak, ϵk);
▷updating step with stochastic gradient
end for

B
The Dynamical nexus

In this section, we prove the main theorem on concentration of the risk curves and learning rates.
We shall set some notation. In what follows, we again use W = [X|X⋆] ∈Rd×2. We also use
W + = [W|X0] = [X|X⋆|X0].

We shall also use the shorthand r = ⟨a, W⟩, and x = ⟨a, X⟩so that f(⟨a, X⟩, ⟨a, X⋆⟩; ϵ) =
f(⟨a, W⟩; ϵ) = f(r; ϵ).

We shall let B = B(X) = W T KW be the covariance matrix of the Gaussian vector r. We also
write f ′ for the ∂xf.

B.1
Discussion of the assumptions on f

In this section we show how the assumptions we put on h and I are almost satisfied for L-smooth f.
We say that f is L-smooth if:

∥∇f(r1, ϵ1) −∇f(r2, ϵ2)∥≤L
p

(∥r1 −r2∥2 + ∥ϵ1 −ϵ2∥2),
which we note implies f is α-pseudo Lipschitz with α = 1.
Lemma B.1.
1. There exists a function h : R2×2 →R such that h(B(X)) = R(X) is
differentiable and satisfies
∇XR(X) = Ea,ϵ∇XΨ(X; a, ϵ).
Furthermore, h is continuously differentiable on {B : det B ̸= 0} and its derivative ∇h
satisfies an estimate

∥∇h(B1) −∇h(B2)∥≤(
√

2 + 1)L(f) min{∥B−1
1 ∥op, ∥B−1
2 ∥op}∥B1 −B2∥F .

2. The function I(B) = Ea,ϵ[(f ′(⟨a, X⟩; ⟨a, X⋆⟩, ϵ))2] satisfies an estimate

|I(B1) −I(B2)| ≤L(f)
p

I(B1) + I(B2) min{∥B−1
1 ∥op, ∥B−1
2 ∥op}∥B1 −B2∥F .

Proof. To derive the existence of h, note that
R(X) = E(E(f(⟨a, X⟩, ⟨a, X⋆⟩, ϵ)|ϵ))
is an expectation of a Gaussian vector r = (⟨a, X⟩, ⟨a, X⋆⟩). This vector can be expressed as an
image of an iid Gaussian vector z by representing r =
√

Bz, and hence we have

h(B)
def
= E(E(f(
√

Bz, ϵ)|ϵ)).

As the function f is absolutely continuous with a Lipschitz gradient, we can differentiate under the
integral sign and conclude
∇XR(X) = ∇X E f(⟨a, X⟩, ⟨a, X⋆⟩, ϵ) = E ∇Xf(⟨a, X⟩, ⟨a, X⋆⟩, ϵ).
For the differentiability of h, suppose for the moment that f is C2 with bounded second derivatives.10

Setting Q =
√

B the positive semi-definite square root of B, we have
∂Qijh(Q2) = E(E(∂Qijf(Qz, ϵ)|ϵ)).

10This condition can be removed in a standard way: one creates an fϵ which is an approximation to f formed
by convolving with an isotropic Gaussian of variance ϵ. This is C2 and has bounded second derivatives (as f
was smooth). One then takes the limit as ϵ →0.

16


---Page Break---
Then using the chain rule, and setting ∂if to be the i-th partial derivative of f,
∂Qijh(Q2) = E(E(zj∂if(Qz, ϵ)|ϵ)) = E(E([Qij∂i + Qjj∂j]∂if(Qz, ϵ)|ϵ)),
where we have applied Stein’s Lemma. We conclude when det Q ̸= 0 by the implicit function
theorem that h is differentiable and we have
∂Qijh(Q2) =
X
∂klh∂Qij(Q2)kl =
X

l
(∂ilh)Qjl +
X

k
(∂kjh)Qik.

As a matrix equation, this can be written as
(Dh)Q + Q(Dh) = JQ
where
Jkl = E(E((∂k∂lf)(Qz, ϵ)|ϵ)).
This is a linear equation in Dh. When Q ≻0, we can define

A =
Z ∞

0
e−tQ(JQ)e−tQ dt,

and note

AQ + QA = −
Z ∞

0

d
dt
 
e−tQ(JQ)e−tQ
dt = JQ.

Moreover, the mapping M 7→
R ∞
0
e−tQMe−tQ dt defines a two-sided inverse for M 7→MQ+QM,
and so Dh = A. Note that by symmetry of J, Q, and Dh
JQ = (Dh)Q + Q(Dh) = QJ,
and therefore
(Dh)Q + Q(Dh) = 1

2(JQ + QJ),

and so taking inverses on both sides, Dh = J.

Undoing Stein’s Lemma, we have Q(Dh) = (Dh)Q = M, where Mij = E(E(zj∂if(Qz, ϵ)|ϵ)).
From L-smoothness of f

∥M(Q1) −M(Q2)∥≤L E(∥z∥∥Q1z −Q2z∥) ≤
√

2L∥Q1 −Q2∥F .
Hence
∥Dh(Q2
1) −Dh(Q2
2)∥= ∥Q−1
1 M(Q1) −Q−1
2 M(Q2)∥

≤∥Q−1
1 ∥op∥M(Q1) −Q1Q−1
2 M(Q2)∥

≤∥Q−1
1 ∥op
 
∥M(Q1) −M(Q2)∥+ ∥(Q2 −Q1)Q−1
2 M(Q2)∥

.

Note Q−1
2 M(Q2) = (Dh)(Q2
2) is bounded by L(f), and so we arrive at

∥Dh(Q2
1) −Dh(Q2
2)∥≤(
√

2 + 1)L(f)∥Q−1
1 ∥op∥Q1 −Q2∥F

≤(
√

2 + 1)L(f)∥Q−2
1 ∥op∥Q2
1 −Q2
2∥F .

We note the bound is symmetric in Q1 and Q2, and by density of C2 in space of C1,lip, this holds for
L-smooth f. This concludes the estimates for the derivative of h.

For the Fisher matrix, I(B), from L-smoothness, we have again with Q =
√

B,
I(Q2) = E(E((∂1f(Qz, ϵ))2|ϵ)).
Then
|I(Q2
1) −I(Q2
2)| ≤
E(E((∂1f(Q1z, ϵ))2 −(∂1f(Q2z, ϵ))2|ϵ))
 .
Applying Cauchy-Schwarz and using the L-smoothness of f,

|I(Q2
1) −I(Q2
2)| ≤
q

I(Q2
1) + I(Q2
2) × L(f)∥Q1 −Q2∥F .

This lemma shows that an L-smooth function nearly satisfies Assumption 3 and 4 provided that
∥B−1∥op is bounded. Therefore, our concentration result Theorem B.1 and its Corollaries will hold
provided we add a stopping time. Fix M > 0 and let

ℏM(B)
def
= inf{t > 0 : ∥B−1∥op > M}.
Then the concentration of the risk under SGD to a deterministic function, Theorem B.1, holds with t
replaced with t ∧ℏM(B) ∧ℏM(B). The corollaries of Theorem B.1 also follow under this added
stopping time.

In the next section, we prove this concentration theorem, Theorem B.1.

17


---Page Break---
B.2
Integro-differential equation for S (t, z)

A goal of this paper is to show that quadratic statistics φ : Rd →R applied to SGD converge to a
deterministic function. This argument hinges on understanding the deterministic dynamics of one
important statistic, defined as
S(W, z) = W ⊤R(z; K)W,

applied to W⌊td⌋(SGD updates). Here W = [X|X⋆] and R(z; K) = (K −zId)−1 for z ∈C is the
resolvent of the matrix K. The statistic S(W, z) is valuable because it encodes many other important
quantities including W ⊤q(K)W for all polynomials q. We show that S(W⌊td⌋, z), is close to a
deterministic function (t, z) 7→S (t, z) which satisfies an integro-differential equation.

To introduce the integro-differential equation, recall by Assumptions 3 and 4

R(X) = h ◦B(W)
and
E a,ϵ[f ′(a⊤W)2] = I ◦B(W)
with
B(W) = W ⊤KW,

and α-pseudo-Lipschitz functions h : R2×2 →R differentiable and I : R2×2 →R. It will be
useful, throughout the remaining paper, to express ∇h explicitly as a 2 × 2 matrix, that is,

∇h ∼=
 ∇h11
∇h12
∇h21
∇h22


.

With these recollections, the integro-differential equation is defined below.

Integro-Differential Equation for S (t, z). For any contour Ω⊂C enclosing the eigenval-
ues of K, we have an expression for the derivative of S :

dS (t, ·) = F(z, S (t, ·)) dt
(20)

where F(z, S (t, ·))
def
= −2γt

 −1

2πi

I

Ω
S (t, z) dz

H(B(t))

+ HT (B(t))
 −1

2πi

I

Ω
S (t, z) dz


+ γ2
t
d

 Tr(KR(z; K))I(B(t))
0
0
0


(21)

−γt(S (t, z)(2zH(B(t))) + (2zHT (B(t)))S (t, z)).

Here B(t) = −1

2πi

I

Ω
zS (t, z) dz,
H(B) =
 ∇h11(B)
0
∇h21(B)
0


,

γt is defined in (9), and the initialization is
S (0, z) = W ⊤
0 R(z; K)W0.
(22)

The functions h : R2×2 →R and I : R2×2 →R are defined in Assumption 3 and
Assumption 4, respectively.

We first note that there is an actual solution to the integro-differential equation. This solution is the
same as the ODEs defined in the introduction (see (9)) and proved in [15, Lemma 4.1].
Lemma B.2 (Equivalence to coupled ODEs.). The unique solution of (21) with initial condition (22)
is given by

S (t, z) = 1

d

d
X

i=1

1
λi −z Vi(t).

In this section, we will be working with approximate solutions to the integro-differential equa-
tion (20) (see below for specifics). For working with these solutions, we introduce some no-
tation. We shall always work on a fixed contour Ωsurrounding the spectrum of K, given by
Ω
def
= {z : |z| = max {1, 2∥K∥op}}. We note that this contour is always distance at least 1

2 from the
spectrum of K. We define a norm, ∥· ∥Ω, on a continuous function A : C →R as

∥A∥Ω= max
z∈Ω∥A(z)∥.
(23)

18


---Page Break---
Definition B.1 ((ε, M, T)-approximate solution to the integro-differential equation). For constants
M, T, ε > 0, we call a continuous function S : [0, ∞) × C →R2×2 an (ε, M, T)-approximate
solution of (20) if with

ˆτM(S)
def
= inf

t ≥0 : ∥S(t, ·)∥Ω> M

,

then

sup
0≤t≤(ˆτM∧T )

S(t, ·) −S(0, ·) −
Z t

0
F(·, S(s, ·)) ds

Ω≤ε

and S(0, ·) = W ⊤
0 R(·, K)W0, where W0 = [X0|X⋆] is the initialization of SGD.

We suppress the S in the notation for ˆτM, that is, ˆτM = ˆτM(S), when the function S is clear from the
context.

We are now ready to state and prove one of our main results.

Theorem B.1 (Concentration of SGD and deterministic function S (t, z)). Suppose the risk function
R(X) (2) satisfies Assumptions 2, 3, and 4. Suppose the learning rate satisfies Assumption 6, and the
initialization X0 and hidden parameters X⋆satisfy Assumption 5. Moreover the data a ∼N(0, K)
and label noise ϵ satisfy Assumption 1. Let {W⌊td⌋} be generated from the iterates of SGD. Then
there is an ε > 0 so that for any T, M > 0 and d sufficiently large, with overwhelming probability

sup
0≤t≤T ∧ˆτM(S(W,·))∧ˆτM(S )
∥S(W⌊td⌋, ·) −S (t, ·)∥Ω≤d−ε,
(24)

where the deterministic function S (t, z) solves the integro-differential equation (20).

Proof. By Proposition C.1, for any M and T, we can find a ˜ε > 0 such that the function S(Wtd, z)
is an (d−˜ε, M, T)-approximate solution. (For the deterministic function S , it is an (0, M, T)-
approximate solution by definition.) We now apply the stability result, [15, Prop. 4.1], to conclude
that there exists a ε > 0 such that

sup
0≤t≤T ∧ˆτM
∥S (t, z) −S(Wtd, z)∥Ω≤d−ε,
w.o.p,
(25)

where ˆτM is shorthand for ˆτM(S(W, ·)) ∧ˆτM(S ). The result immediately follows.

Corollary B.1. Suppose the assumptions of Theorem B.1 hold. Let f be an α-pseudo-Lipschitz
function with α ≤1 and let q be a polynomial. Set

φ(X)
def
= f(W T q(K)W),
ϕ(t)
def
= f
 −1

2πi

I

Ω
q(z)S (t, z) dz

,
where S (t, z) solves (20).

Then there is an ε > 0 such that for d sufficiently large, with overwhelming probability,

sup
0≤t≤T
|φ (Xtd) −ϕ(t)| ≤d−ε.

Proof. This is basically equivalent to [15, Corollary 4.2]. The only difference is that [15, Corollary
4.2] requires the boundedness of N ; however, since our function f is α-pseudo-Lipschitz with α ≤1,
this boundedness follows from [15, Proposition 1.2], and the rest of the proof is identical to the one
in [15].

Remark B.1. The learning rate gk, technically, is not a function of W T q(K)W. However, As-
sumption 6 ensures that the learning rate concentrates around a function W T q(K)W. Therefore,
Corollary B.1 applies to the learning rate.

C
SGD-AL is an approximate solution

We introduce a rescaling of time to relate the k-th iteration of SGD to the continuous time parameter
t in the differential equation through the relationship k = ⌊td⌋. Thus, when t = 1, SGD has done
exactly d updates. Since the parameter t is continuous and the iteration counter k (integer) discrete,

19


---Page Break---
to simplify the discussion below, we extend k to continuous values through the floor operation,
Xk
def
= X⌊k⌋. Using the continuous parameter t, the iterates are related by Xtd = X⌊td⌋.

The paper [15] provides a net argument showing that we do not need to work with every z on the
contour Ωdefining the integro-differential equation, but only polynomially many in d. Recall that
Ω= {z : |z| = max{2∥K∥op, 1}}. For a fixed ξ > 0, we say that Ωξ is a d−ξ-mesh of Ωif Ωξ ⊂Ω
and for every z ∈Ωthere exists a ¯z ∈Ωξ such that |z −¯z| < d−ξ. We can achieve this with Ωξ
having cardinality, |Ωξ| = C(|Ω|)dξ.
Lemma C.1 (Net argument, [15], Lemma 5.1). Fix T, M > 0 and let ξ > 0. Suppose Ωξ is a d−ξ

mesh of Ωwith |Ωξ| = C · dξ and positive C > 0. Let the function S(t, z) = S(Wtd, z) satisfy

sup
0≤t≤(ˆτM∧T )
∥S(t, ·) −S(0, ·) −
Z t

0
F(·, S(s, ·)) ds∥Ωξ ≤ε
(26)

with ˆτM = inf{t ≥0 : ∥S(t, ·)∥Ω> M}. Then S is a (ε + C(M, T, ∥K∥op)d−ξ, M, T)-
approximate solution to the integro-differential equation, that is,

sup
0≤t≤(ˆτM∧T )
∥S(t, ·) −S(0, ·) −
Z t

0
F(·, S(s, ·)) ds∥Ω≤ε + C · d−ξ,

where C = C(M, T, ∥K∥op, L(I), L(h)) is a positive constant.

(We prove in Section C.1 that S(t, z) does indeed satisfy inequality (26).) We also cite the following
lemma, which relates two stopping times used throughout this paper.
Lemma C.2 (Stopping time, [15], Lemma 4.2). For a constant C depending on ∥K∥op, we have

C ≤∥S(Wtd, ·)∥Ω

∥Wtd∥2
≤2.

Remark C.1. Fix M > 0 and define the stopping time on ∥Wtd∥, ϑ = ϑM, by

ϑM(Wtd)
def
= inf

t ≥0 : ∥Wtd∥2 > M
	
.

Due to the previous lemma, any stopping time ˆτM defined on ∥S(t, ·)∥Ωcorresponds to a stopping
time ϑ on ∥Wtd∥, that is, for c = C−1, ˆτM ≤ϑcM.

C.1
SGD-AL is an approximated solution

Proposition C.1 (SGD-AL is an approximate solution). Fix a T, M > 0 and 0 < ε < δ/8, where δ
is defined in Assumption 6. Then S(Wtd, z) is a (d−ε, M, T)-approximate solution w.o.p., that is,

sup
0≤t≤(T ∧τM)
∥S(Wtd, z) −S(W0, z) −
Z t

0
F(z, S(Wsd, z)) ds∥Ω≤d−ε.
(27)

Again, the proof is very similar to [15, Prop. 5.2]. The one difference is that the martingales and
error terms are slightly more involved, because of the non-deterministic stepsize we are using. The
remainder of this section, along with section C.2, fills in the details of bounding these lower-order
terms, so that the proof can proceed as in [15].

C.1.1
Shorthand notation

In the following sections, we will be using various versions of the stepsize γ. In order to simplify
notation, we set
γ(Gk) = γ(k, Nk(d × ·), Gk(d × ·), Qk(d × ·)),
γ(Gk) = γ(k, Nk(d × ·), Gk(d × ·), Qk(d × ·)),
γ(Bk) = γ(k, Nk(d × ·), Tr(K)I(Bk(d × ·))/d, Qk(d × ·)).

Further, setting ∆k
def
= f ′(rk)ak+1, define

I1(k)
def
= ∆⊤
k ∇2φ(Xk)∆k/d, I2(k)
def
= Tr(∇2φ(Xk)K)E [f ′(rk)2 | Fk]/d, I3(k)
def
= ∇φ(Xk)⊤∆k.
The normalization here (dividing by d) is chosen so that the I terms are all O(1); this is formally
shown in Lemma C.5.

20


---Page Break---
C.1.2
SGD-AL under the statistic

We follow the approach in [15, Section 5.3] to rewrite the SGD adaptive learning rate update rule as
an integral equation. Considering a quadratic function φ : Rd →R and performing Taylor expansion,
we obtain

φ(Xk+1) = φ(Xk) −γ(Gk)

d
∇φ(Xk)⊤∆k + γ(Gk)2

2d2
∆⊤
k ∇2φ(Xk)∆k.
(28)

We will now relate this equation to its expectation by performing a Doob decomposition, involving
the following martingale increments and error terms:

∆Mgrad
k
(φ)
def
= 1

d
 
−γ(Gk)I3(k) + E

γ(Gk)I3(k)
 Fk

,
(29)

∆MHess
k
(φ)
def
= 1

2d
 
γ(Gk)2I1(k) −E

γ(Gk)2I1(k)
 Fk

,
(30)

E [EHess
k
(φ) | Fk]
def
= 1

2d
 
E

γ(Gk)2I1(k)
 Fk

−γ(Bk)2I2(k)

,
(31)

E [Egrad
k
(φ) | Fk]
def
= 1

d
 
−E

γ(Gk)I3(k)
 Fk

+ γ(Bk)∇φ(Xk)⊤∇R(Xk)

.
(32)

We can then write

φ(Xk+1) = φ(Xk) −γ(Bk)

d
∇φ(Xk)⊤∇R(Xk) + γ(Bk)2

2d2
Tr(∇2φ(Xk)K)E [f ′(rk)2 | Fk]

+ ∆Mgrad
k
(φ) + ∆MHess
k
(φ) + E [EHess
k
(φ) | Fk] + E [Egrad
k
(φ) | Fk].

Extending Xk into continuous time by defining Xt = X⌊t⌋, we sum up (integrate). For this, we
introduce the forward difference

(∆φ)(Xj)
def
= φ(Xj+1) −φ(Xj),

giving us

φ(Xtd) = φ(X0) +

⌊td⌋−1
X

j=0
(∆φ)(Xj)
def
= φ(X0) +
Z t

0
d · (∆φ)(Xsd) ds + ξtd,

where |ξtd| =


Z t

(⌊td⌋−1)/d
d · ∆φ(Xsd) ds
 ≤
max
0≤j≤⌈td⌉{|∆φ(Xj)|}. With this, we obtain the Doob

decomposition for SGD-AL:

φ(Xtd) = φ(X0) −
Z t

0
γ(Bsd)∇φ(Xsd)⊤∇R(Xsd) ds
(33)

+ 1

2d

Z t

0
γ(Bsd)2 Tr(K∇2φ(Xsd))E [f ′(rsd)2 | Fsd] ds

+

⌊td⌋−1
X

j=0
Eall
j (φ),

with
Eall
j (φ) = ∆Mgrad
j
(φ) + ∆MHess
j
(φ)
(34)

+ E [EHess
j
(φ) | Fj] + E [Egrad
j
(φ) | Fj]

+ ξtd(φ).

From here, we can proceed as in [15, Section 5.3] to show that SGD-AL is an (ε, M, T)-approximated
solution.

C.1.3
S(Wtd, z) is an approximate solution

Proof of Proposition C.1. The appropriate stepsize, as a function of Wtd, is

γt = γ(td, Ntd, Tr(K)I(Btd)/d, Qtd).

21


---Page Break---
(Note that N, I and Q can all be found as functions of S(Wtd, ·) using contour integration.) It is
shown in the proof of [15, Proposition 5.2] that given the analogue of (33) for deterministic stepsize,
S(Wtd, ·) satisfies

S (Wtd, z) = S (W0, z) +
Z t

0
F (z, S (Wsd, z)) ds +

⌊td⌋−1
X

i=0
Eall
j (S).

The only terms of (33) that differ in our case are the martingale and error terms. Thus to show
that S(Wtd, ·) is an approximate solution of the integro-differential equation (20) all we need is to
bound the martingales and error terms contained in Eall
j . Let Ω= {z : |z| = max{1, 2∥K∥op}}, as
previously. We thus have that for all z ∈Ω,

sup
0≤t≤T ∧ˆτM

S(Wtd, z) −S(W0, z) −
Z t

0
F(z, S(Wsd, z)) ds
 ≤
sup
0≤t≤T ∧ˆτM
∥Eall
td(S(·, z))∥. (35)

Next, fix a constant ξ > 0. Let Ωξ ⊂Ωsuch that there exists a ¯z ∈Ωξ such that |z −¯z| ≤d−ξ

and the cardinality of Ωξ, |Ωξ| = Cdξ where C > 0 can depend on ∥K∥op. For all z ∈Ω, we note
that ˆτM ≤ϑcM (see Lemma C.2). Consequently, we evaluate the error with the stopped process
W ϑ
td
def
= Wd(t∧ϑ) instead of using ˆτM. By Proposition C.2, the proof of which we have deferred to
Section C.2, we have, for any ˆδ > 0

sup
z∈Ωξ
sup
0≤t≤T ∧ϑcM
∥Eall
dt(S(·, z))∥≤d−δ/4+ˆδ
w.o.p.
(36)

We deduce that

sup
0≤t≤T ∧ˆτM
∥S(Wtd, z) −S(W0, z) −
Z t

0
F(z, S(Wsd, z)) ds∥Ωξ ≤d
ˆδ−δ/4
w.o.p.

An application of the net argument, Lemma C.1, finishes the proof after setting ˆδ = δ/8 and
ξ = δ/8.

C.2
Error bounds

All the martingale and error terms (34) go to 0 as d grows. Formally,

Proposition C.2. Let the function f be defined as in Assumption 2. Let the statistic S : [0, ∞)×C →
R2×2 be defined as
S(t, z) = W ⊤
⌊td⌋R(z; K)W⌊td⌋,
(37)

where W = [X|X⋆]. Then, for any z ∈Ωand T, M, ζ > 0, with overwhelming probability,

sup
0≤t≤T ∧ϑ

Eall
dt (S(·, z))
 ≤d−δ/4+ζ,

where to suppress notation we use ϑ as shorthand for ϑcM, and c is the constant from Lemma C.2.

Proof. This follows from combining Propositions C.3, C.4, C.5, C.6, and C.7.

The remainder of this subsection is devoted to proving these supporting propositions; throughout
these proofs we will work with the stopping time ϑ as defined in the proposition above.

C.2.1
Bounds on the lower order terms in the gradient and hessian

Proposition C.3 (Hessian error term). Let f and S be defined as in Assumption 2 and (37). Then,
for any z ∈Ω, T > 0 and ζ > 0, with overwhelming probability,

sup
0≤t≤T ∧ϑ

⌊td⌋−1
X

k=0

E

EHess
k
(S(·, z)) | Fk
 ≤d−δ/4+ζ.

22


---Page Break---
Proof. For arbitrary z ∈Ωand k ≤(T ∧ϑ)d −1, set φ(X) = Sij(W, z) to be the ij-th entry of the
matrix S(W, z). Then

2d E[EHess
k
(φ) | Fk] = E

γ(Gk)2I1(k) | Fk

−γ(Bk)2I2(k)

= E[(γ(Gk)2 −γ(Gk)2)I1(k) | Fk]

+ (γ(Gk)2 −γ(Bk)2) E[I1(k) | Fk] + γ(Bk)2 E[(I1(k) −I2(k) | Fk]
= E1 + E2 + E3.

We look at |E1| first.

|E1| =
E

(γ(Gk)2 −γ(Gk)2)I1(k) | Fk


≤E
h(γ(Gk)2 −γ(Gk)2)
2 | Fk
i 1

2 · E
I1(k) |2 Fk
 1

2

≤E
h
|γ(Gk) + γ(Gk)|

7
2 |γ(Gk) −γ(Gk)|

1
2 | Fk
i 1

2 · E
I1(k) |2 Fk
 1

2

≤E
h
|γ(Gk) + γ(Gk)|7 | Fk
i 1

4 · E [|γ(Gk) −γ(Gk)| | Fk]

1
4 · E
h
|I1(k)|2 | Fk
i 1

2 .

For the first term, we use (6). We have

E
h
|γ(Gk) + γ(Gk)|7 | Fk
i
≤ˆC(γ) · E
h
|2 + 2∥Nk∥α
∞+ 2∥Qk∥α
∞+ ∥Gk∥α
∞+ ∥Gk∥α
∞|7 | Fk
i
.

All the terms inside the expectation, apart from ∥Gk∥α
∞, are deterministic with respect to Fk and
bounded by a constant independent of d (see Lemma C.6). Since we know from Lemma C.6 that for
any ε > 0, all moments of ∥Gk∥∞are bounded by dε w.o.p., we conclude

E
h
|γ(Gk) + γ(Gk)|7 | Fk
i
≤dε
w.o.p.

For the second term, we use (5). Again, since ∥Nk∥∞and ∥Qk∥∞are bounded due to our stopping
time, we have
E [|γ(Gk) −γ(Gk)| | Fk]

1
4 ≤d−δ/4.

The last term, E
h
|I1(k)|2 | Fk
i 1

2 , is also bounded by a constant (see Lemma C.5), and all together,

we find that |E1| ≤dε−δ/4 with overwhelming probability.

Now let us consider |E2|:

|E2| = |(γ(Gk)2 −γ(Bk)2) E[I1(k) | Fk]| = |γ(Gk) + γ(Bk)| · |γ(Gk) −γ(Bk)| · | E[I1(k) | Fk]|.

The first term is bounded by (6), since Gk and Tr(K)I(Bk)/d are bounded independent of d;
the second term is bounded Cd−1 by Lemma C.9, and the last term is bounded by a constant by
Lemma C.5.

Finally, consider |E3|:
|E3| = γ(Bk)2 · |E [(I1(k) −I2(k) | Fk]| .

By (6), the first term is bounded by ˆC(γ)2(1 + ∥Nk∥α
∞+ ∥Qk∥α
∞+ ∥Tr(K)I(Bk)/d∥α
∞)2. All of
these terms are bounded by a constant independent of d (because of the stopping time.) The second
term satisfies the assumptions of Lemma C.8 with H = ∇2φ(Xk), and is thus bounded by Cd−1.
All together,
2d E[EHess
k
(φ) | Fk] ≤d−δ/4+ε.

Summing up to k = Td and dividing through by 2d, we obtain the desired bound.

Proposition C.4 (Gradient error term). Let f and S be defined as in Assumption 2 and (37). Then,
for any z ∈Ω, ζ > 0 and T > 0, with overwhelming probability,

sup
0≤t≤T ∧ϑ

⌊td⌋−1
X

k=0

E
h
Egrad
k
(S(·, z)) | Fk
i ≤d−δ/4+ζ.

23


---Page Break---
Proof. We have

dE [Egrad
k
| Fk] = −E [γ(Gk)⟨∇φ(Xk), ∆k⟩| Fk] + γ(Bk)⟨∇φ(Xk), ∇R(Xk)⟩
= −E [(γ(Gk) −γ(Gk))I3(k) | Fk] −(γ(Gk) −γ(Bk)E [I3(k) | Fk]
= E1 + E2.

We then have

|E1| ≤E
h
|γ(Gk) −γ(Gk)|2 | Fk
i 1

2 · E
h
|I3(k)|2 | Fk
i 1

2

≤E
h
|γ(Gk) + γ(Gk)|3 | Fk
i 1

4 · E [|γ(Gk) −γ(Gk)| | Fk]

1
4 · E
h
|I3(k)|2 | Fk
i 1

2 .

Just as in the Hessian argument, (6) lets us bound E
h
|γ(Gk) + γ(Gk)|3 | Fk
i 1

4 by dε w.o.p.,

(5) lets us bound E [|γ(Gk) −γ(Gk)| | Fk]

1
4 by d−δ/4 w.o.p., and Lemma C.5 lets us bound

E
h
|I3(k)|2 | Fk
i 1

2 by a constant, giving an overall bound of |E1| ≤d−δ/4+ε.

By the same argument as in the Hessian case, |E2| is bounded by Cd−1; in conclusion,

dE [Egrad
k
| Fk] ≤dε−δ/4.

Summing and dividing through by d, we obtain the desired result with ζ = ε.

Proposition C.5 (Gradient martingale). Let f and S be defined as in Assumption 2 and (37). Then,
for any z ∈Ω, ζ > 0 and T > 0, with overwhelming probability,

sup
0≤t≤T ∧ϑ

Mgrad
⌊dt⌋(S(·, z))
 ≤d−1/2+ζ.

Proof. For notational convenience, set ∆Mk = ∆Mgrad
d(k/d∧ϑ), and Fk = −γ(Gk)I3(k)/d, so that

∆Mk = Fk −E[Fk | Fk].

Set F β
k = Projβ(Fk), that is, ensuring Fk stays in [−β, β]. Then F β
k −E[F β
k | Fk] is in [−2β, 2β],
and so for the martingale Mβ
k with increments ∆Mβ
k = F β
k −E[F β
k | Fk], Azuma’s inequality tells
us that

P

|Mβ
k| ≥t

≤2 exp

 
−t2

2 Pk
i=0(2β)2

!

≤2 exp

−t2

2Td(2β)2


.

Set β = d−1+ζ/2 and t = d−1/2+ζ; this becomes

P

|Mβ
k| ≥d−1/2+ζ
≤2 exp
−dζ

8T


.

However, Mβ
k is not quite the martingale we started with: there is still an error term,

|Mk −Mβ
k| =



k
X

i=0
(Fk −E[Fk | Fk]) −(F β
k −E[F β
k | Fk])



≤

k
X

i=0

Fk −F β
k
 +
E[Fk −F β
k | Fk]
 .

We bound this term in overwhelming probability. We have

P

Fk −F β
k ̸= 0

= P (|Fk| > β)

= P

|γ(Gk)I3(k)/d| > d−1+ζ/2

≤P

γ(Gk) ≥dζ/4
+ P

|I3(k)| ≥dζ/4
.

24


---Page Break---
The second term is superpolynomially small by Lemma C.5; the first term is superpolynomially small
by (6) and (C.6).
E[Fk −F β
k | Fk]
 =
E[(Fk −F β
k )1{|Fk|>β} | Fk]


≤E[(Fk −F β
k )2 | Fk]
1
2 · E[12
{|Fk|>β} | Fk]
1
2

≤4 E[F 2
k | Fk]
1
2 · E[1{|Fk|>β} | Fk]
1
2

≤4d−1 E[γ(Gk)4 | Fk]
1
4 · E[I3(k)4 | Fk]
1
4 · E[1{|Fk|>β} | Fk]
1
2 .
As before, the first and second expectations are bounded by constants, and the last expectation is just
the probability that |Fk| > β, which we have already shown is superpolynomially small. So with
overwhelming probability, we have

|Mk −Mβ
k| =



k
X

i=0
(Fk −E[Fk | Fk]) −(F β
k −E[F β
k | Fk])

 ≤d−1/2+ζ

(any power of d would have worked). Combining the error term and the projected martingale, we
find that, with overwhelming probability,

|Mk| ≤d−1/2+ζ.
We can now take the maximum over k from 0 to Td using a union bound; this does not affect the
overwhelming probability statement.

Proposition C.6 (Hessian martingale). Let f and S be defined as in Assumption 2 and (37). Then,
for any z ∈Ω, ζ > 0 and T > 0, with overwhelming probability,

sup
0≤t≤T ∧ϑ

MHess
⌊td⌋(S(·, z))
 ≤d−1/2+ζ.

Proof. The proof here is basically identical to the previous one. Again, set Fk = γ(Gk)2I1(k)/d
and F β
k = Projβ(Fk), with their associated martingales being Mk = Fk −E[Fk | Fk] and Mβ
k =
F β
k −E[F β
k | Fk]. As before, Azuma’s inequality, with β = d−1+ζ/2, gives us

P(Mβ
k ≥d−1/2+ζ) ≤2 exp

−dζ

8T


.

The error term is also quite similar:

|Mk −Mβ
k| ≤

k
X

i=0
|Fk −F β
k | + | E[Fk −F β
k | Fk]|.

We have
P(Fk −F β
k ̸= 0) ≤P(γ(Gk)2 ≤dζ/4) + P(|I2(k)| ≤dζ/4),
both of which are superpolynomially small by (6) and Lemma C.5. For the expectation, we have

| E[Fk −F β
k | Fk]| ≤4d−1 E[γ(Gk)8 | Fk]
1
4 · E[I1(k)4 | Fk]
1
4 · E[1{|Fk|>β} | Fk]
1
2 ;
this product is superpolynomially small by (6), Lemma C.6, and Lemma C.5. Overall, we have, with
overwhelming probability,
|Mk| ≤d−1/2+ζ.
Taking the supremum, we obtain the desired result.

Proposition C.7 (Integral error term). Let f and S be defined as in Assumption 2 and (37). Then, for
z ∈Ω,
|ξtd(S(·, z))| ≤d−1/2.

Proof. We have, as above,

|ξtd| =


Z t

(⌊td⌋−1)/d
d · ∆φ(Xsd) ds


≤
max
0≤j≤⌈td⌉{|∆φ(Xj)|},

which is bounded by d−1/2 w.o.p. by the boundedness of I1, I2, I3, and γ(Bk).

25


---Page Break---
C.2.2
General bounds

In this section, we make use of the subgaussian norm ∥· ∥ψ2 of a random variable (see [56] for
details.) When it exists, this norm is defined as

∥X∥ψ2 ≍inf
n
V > 0 : ∀t > 0, P(|X| > t) ≤2e−t2/V 2o
.
(38)

In particular, Gaussian random variables have a well-defined subgaussian norm.
Lemma C.3 ([15], Lemma 5.3). There exist constants c, C > 0 such that

c∥W∥2 ≤∥S(W, z)∥Ω≤C∥W∥2,
∥∇XS(W, z)∥Ω≤C∥W∥,
and
∥∇2
XS(W, z)∥Ω≤C.

Lemma C.4 (Preliminary bounds). With f and ∆k defined as above, for ε > 0 and λ ≥0, we have

f ′(rk) ≤dε
w.o.p. and
E[|f ′(rk)|λ | Fk] ≤C(λ),
(39)

∥∆k∥2

d
≤dε
w.o.p. and
E

"∥∆k∥2

d

λ
| Fk

#

≤C(λ).
(40)

Proof of (39) in Lemma C.4. By [15, Lemma 3.4], if function f is α-pseudo-Lipschitz with Lipschitz
constant L(f) (as in (2)) and the noise ϵ is independent of a, then

|f ′(r)| ≤C(α)(L(f))(1 + |r| + |ϵ|)max{1,α}.

Then

|f ′(rk)| ≤C(α)(L(f))(1 + |rk| + |ϵ|)max{1,α}

≤C(α)(L(f))(1 + |X⊤
k ak+1| + |ϵ|)max{1,α}.
(41)

Now, since ak+1 is Gaussian, we can write ak+1 =
√

Kvk, for a standard normal vk. Then we see
that X⊤
k ak+1 = X⊤
k
√

Kvk is a single-variable Gaussian, with variance |X⊤
k KXk| ≤∥Xk∥2·∥K∥op
(bounded independently of d because of the stopping time on Xk). Similarly, ϵ is Gaussian and
independent of ak+1, so the expression (41) is bounded w.o.p. by dε, and

E

C(α)(L(f))(1 + |X⊤
k ak+1| + |ϵ|)max{1,α}λ  Fk


≤C(λ)

for some constant C(λ).

Proof of (40) in Lemma C.4. We can write ak+1 =
√

Kvk, where vk is a standard d-dimensional
normal vector. Then, by Hanson-Wright, we have

P
 ∥ak+1∥2 −E[∥ak+1∥2 | Fk]
 ≥d

= P
 v⊤
k Kvk −E[v⊤
k Kvk | Fk]
 ≥d


≤2 exp

−
cd2

∥K∥2
F + ∥K∥opd



≤2 exp

 

−
cd2

d(∥K∥op + ∥K∥2op

!

≤2 exp (−Cd) .

Now, note that E[v⊤
k Kvk | Fk] = Tr(K) ≤d∥K∥op. Together, we get that ∥ak+1∥2 ≤d1+ϵ with
overwhelming probability. Then

∥∆k∥2

d
= ∥f ′(rk)ak+1∥2

d
= ∥ak+1∥2f ′(rk)2

d
,

which is bounded by d2ε w.o.p. Now for the expectation:

E

"∥∆k∥2

d

λ
| Fk

#

≤E





 
∥
√

Kvk∥2

d

!2λ

| Fk





1
2

· E

f ′(rk)4λ | Fk
 1

2

≤E

"∥K∥op · ∥vk∥2

d

2λ
| Fk

# 1

2
· E

f ′(rk)4λ | Fk
 1

2
(42)

26


---Page Break---
For the first term, we have

E

"∥K∥op · ∥vk∥2

d

2λ
| Fk

#

= ∥K∥2λ
op · E

"∥vk∥2

d

2λ
| Fk

#

≤∥K∥2λ
op · 1

d

d−1
X

i=0
E
h 
∥vi
k∥22λ | Fk
i
(Jensen’s inequality)

= ∥K∥2λ
op · E

∥v0
k∥4λ | Fk

,
(i.i.d. assumption)

where we are using the notation vi
k to refer to the ith component of the vector vk. Now, since v0
k is
just a standard Gaussian, all of its moments are bounded. The second term in (42) is bounded by a
constant by (39), as desired.

Lemma C.5 (Gradient and Hessian bounds). Setting

I1(k)
def
= ∆⊤
k ∇2φ(Xk)∆k/d,
I2(k)
def
= Tr(∇2φ(Xk)K)E [f ′(rk)2 | Fk]/d,

I3(k)
def
= ∇φ(Xk)⊤∆k,

for any ε > 0 and λ ≥0, we have

|I1(k)| ≤dε
w.o.p. and
E

|I1(k)|λ | Fk

≤C(λ),
(43)

|I2(k)| ≤C,
(44)

|I3(k)| ≤dε
w.o.p. and
E

|I3(k)|λ | Fk

≤C(λ).
(45)

Proof of (43) in Lemma C.5. Using the fact that ∥∇2φ(Xk)∥op ≤∥S(Wk, ·)∥Ω,

|∆⊤
k ∇2φ(Xk)∆k|

d
≤∥S(Wk, ·)∥Ω∥∆k∥2

d

≤C∥Wk∥2∥∆k∥2

d
.
(Lemma C.3)

Now, ∥Wk∥is bounded by the stopping time. From Lemma C.4, ∥∆k∥2

d
is bounded by dε w.o.p., and
every moment of this expression is bounded independent of d, as desired.

Proof of(44) in Lemma C.5. We have
Tr(∇2φ(Xk)K)E [f ′(rk)2 | Fk]


d
≤d∥∇2φ(Xk)K∥op · E[f ′(rk)2 | Fk]

d
≤∥∇2φ(Xk)∥op · ∥K∥op · E[f ′(rk)2 | Fk]

≤CM 2 E[f ′(rk)2 | Fk].
(Lemma C.3)

From Lemma C.4, E[f ′(rk)2 | Fk] is bounded by a constant independent of d, as desired.

Proof of (45) in Lemma C.5. We have

|∇φ(Xk)⊤∆k| ≤|∇φ(Xk)⊤ak+1| · |f ′(rk)|.

By Lemma C.3, ∥∇φ(Xk)∥≤C∥Wk∥≤CM (since we are working under a stopping time), and
so ∇φ(Xk)⊤ak+1 is subgaussian (and thus bounded by dε w.o.p.). By (39), f ′(rk) is bounded by dε
w.o.p., and so their product is bounded by d2ε w.o.p., as desired. Now for the expectation:

E

|∇φ(Xk)⊤∆k| | Fk

≤E

|∇φ(Xk)⊤ak+1| · |f ′(rk)| | Fk


≤E

|∇φ(Xk)⊤ak+1|2 | Fk
 1

2 · E

f ′(rk)2 | Fk
 1

2

The first term is bounded by a constant independent of d, since subgaussian moments are bounded.
The second term is bounded by Lemma C.4, completing the proof.

27


---Page Break---
Lemma C.6 (Infinity norm bounds). For Gk, Nk, Qk as defined in 1.2, we have, for any ε, λ > 0,
there exists C > 0 such that,

∥Gk∥∞≤dε
w.o.p. and
E[∥Gk∥λ
∞| Fk] ≤dε
w.o.p.,
(46)
∥Nk∥∞≤C,
∥Qk∥∞≤C,
∥Gk∥∞≤C.
(47)

Proof. The first line, (46), follows from (40). For the first inequality, ∥Gk∥∞= max0≤j≤k
∥∆j∥2

d
,
which are all bounded by dε with overwhelming probability. A union bound tells us that the maximum
is also bounded by dε w.o.p.. For the second inequality,

E [|Gk|λ
∞| Fk] ≤E

"∥∆k∥2

d

λ
| Fk

#

+ E

"

max
0≤j≤k−1

∥∆j∥2

d

λ
| Fk

#

≤E

"∥∆k∥2

d

λ
| Fk

#

+
max
0≤j≤k−1

∥∆j∥2

d

λ

≤dε,
(w.o.p.)

as desired. The second line is more straightforward:

∥Nk∥∞= max
0≤j≤k ∥(W +
j )⊤W +
j ∥.

Now, ∥X⋆∥and ∥X0∥are bounded independent of d, and ∥Xj∥is bounded by cM (because of the
stopping time we are using.) Thus the maximum over j of their inner products are bounded by a
constant. The same thing holds for ∥Qk∥∞:

∥Qk∥∞= max
0≤j≤k R(Xj)

= max
0≤j≤k h(W ⊤
j KWj).

Since the derivative of h is pseudo-Lipschitz, h is continuous, and thus bounded for bounded
arguments. And indeed, the argument to h is bounded:

∥W ⊤
j KWj∥≤∥Wj∥2∥K∥op,

both of which are bounded independent of d. Finally, a similar argument applies to Gk:

∥Gk∥∞= max
0≤j≤k E
∥∆j∥2

d
| Fj


≤max
0≤j≤k C = C

by Lemma C.4.

We now prove a concentration result that closely follows [15, Proposition 5.6].
Lemma C.7 ([15], Lemma 5.2). Suppose v ∈Rd is distributed N(0, Id) and U ∈Rd×2 has
orthonormal columns. Then

v | U ⊤v ∼v −U(U ⊤v) + UU ⊤v,
(48)

where v−U
 
U T v

∼N
 
0, Id −UU T 
and UU T v ∼N
 
0, UU T 
with v−U
 
U T v

independent
of UU T v.
Lemma C.8. For a matrix H = Hk with bounded operator norm, or ∥H∥op < C and E[Hk | Fk] =
Hk, set q(a) = a⊤Ha. Then
E[q(ak+1)f ′(rk)2 | Fk] −Tr(KH) E[f ′(rk)2 | Fk]
 ≤C(H).

Note that the H used here is not the same as the matrix used in the integro-differential equation.

Proof. Many of the computations in this proof are taken directly from [15], but we repeat them
here for completeness. We have Fk = σ({Wi}k
i=0); set ˆ
Fk = σ({Wi}k
i=0, {ri}k
i=0). A simple
calculation shows that

E[q(ak+1)f ′(rk)2 | ˆ
Fk] = E[q(ak+1 −E[ak+1 | ˆ
Fk]) | ˆ
Fk] Eϵ[f ′(rk)2]

+ q(E[ak+1 | ˆ
Fk]) Eϵ[f ′(rk)2].
(49)

28


---Page Break---
To compute the conditional mean E[ak+1 | ˆ
Fk] and covariance (ak+1 −E[ak+1 | ˆ
Fk])(ak+1 −
E[ak+1 | ˆ
Fk])⊤, we use Lemma C.7.
By Assumption 1, we can write ak+1 =
√

Kvk, for
vk ∼N(0, Id).

Now we perform a QR-decomposition on
√

KWk
def
=
QkRk where Qk
∈Rd×2 with or-
thonormal columns and Rk ∈R2×2 is upper triangular (and invertible). Set Πk
def
= QkQT
k . In
distribution,
ak+1 | a⊤
k+1Wk
d=
√

Kvk | RT
k QT
k vk.

As Rk is invertible, by Lemma C.7,

ak+1 | a⊤
k+1Wk
d=
√

Kvk | QT
k vk
d=
√

K
 
vk −Πkvk

+
√

KΠkvk.
(50)

We note that (Id −Πk)vk ∼N(0, Id −Πk) and Πkvk ∼N(0, Πk) with (Id −Πk)vk independent
of Πkvk. From this, we have that

E [ak+1 | ˆFk] =
√

KΠkvk,
where vk ∼N(0, Id).
(51)

Moreover the conditional covariance of ak+1 is precisely

(E [(ak+1 −E [ak+1 | ˆ
Fk])(ak+1 −E [ak+1 | ˆ
Fk])⊤| ˆ
Fk])
(52)

=
√

K(Id −Πk)
√

K,
where Πk = QkQT
k .

Next, using that E[Hk | Fk] = Hk, we expand (49) to get the leading order behavior

E [q(ak+1)f ′(rk)2 | ˆFk] = Tr(HK) Eϵ[f ′(rk)2]

−Tr(H
√

KΠk
√

K) Eϵ[f ′(rk)2]

+ q(
√

KΠkvk) Eϵ[f ′(rk)2].

(53)

Taking the expectation with respect to Fk, we obtain

E [q(ak+1)f ′(rk)2 | Fk] −Tr(HK) E[f ′(rk)2 | Fk] = E[Ek | Fk],
(54)

where the error Ek is defined as

Ek = −Tr(H
√

KΠk
√

K) Eϵ[f ′(rk)2]
(55)

+ q(
√

KΠkvk) Eϵ[f ′(rk)2].
(56)

The proof now turns to bounding the expectation of this error quantity.

| Tr(H
√

KΠk
√

K) E[f ′(rk)2 | Fk]| = | Tr(H
√

KΠk
√

K)| · E[f ′(rk)2 | Fk]

≤∥H∥op∥K∥op| Tr(Πk)| · E[f ′(rk)2 | Fk]

≤∥H∥op∥K∥op · rank(Qk) E[f ′(rk)2 | Fk]

≤2∥H∥op∥K∥op E[f ′(rk)2 | Fk].

By (39), the expectation is bounded by a constant, so this term is overall bounded by a constant. We
move on to the next term in the error:

q(
√

KΠkvk)f ′(rk)2 ≤∥H∥op∥K∥op∥Πkvk∥2f ′(rk)2.

Taking expectations and using Cauchy Schwarz, we obtain

E[q(
√

KΠkvk)f ′(rk)2 | Fk] ≤∥H∥op∥K∥op ·
p

E[∥Πkvk∥4 | Fk] ·
p

E[f ′(rk)4 | Fk].

The first expectation is E[∥Πkvk∥2 | Fk] = ∥Πk∥4
F = 8, and the second is bounded by (39) as before.
We thus conclude that E[Ek | Fk] is bounded by a constant depending on ∥H∥op, completing the
proof.

Lemma C.9. There is a constant C such that

|γ(Gk) −γ(Bk)| ≤Cd−1.

29


---Page Break---
Proof. Using the Lipschitz condition on the stepsize, we have

|γ(Gk) −γ(Bk)|
≤∥Gk −Tr(K)I(Bk)/d∥∞× (1 + 2∥Nk∥α
∞+ ∥Gk∥α
∞+ ∥Tr(K)I(Bk)/d∥α
∞+ 2∥Qk∥α
∞)
≤C∥Gk −Tr(K)I(Bk)/d∥∞
(Lemma C.6)

≤Cd−1 max
0≤j≤k

E[a⊤
j+1aj+1f ′(rj)2 | Fj] −Tr(K) E[f ′(rj)2 | Fj]


≤Cd−1,
(Lemma C.8)

as desired.

C.3
Specific learning rates

In this section, we confirm that AdaGrad-Norm satisfies Assumption 6. In the notation of Assump-
tion 6, we have, for AdaGrad-Norm,

γ(td, f, g, q) =
η
q

b2 +
R ∞
0
g(s) ds
.

Note that this reduces to the discrete stepsize if we plug in g = Gk:

γ(td, f, Gk(d × ·), q) =
η
q

b2 +
R ∞
0
Gk(ds) ds

=
η
r

b2 +
R ∞
0

1{ds≤k}
1
d
Pk
i=0 ∥∇XΨ(Xi; ai+1, ϵi+1)∥21[i,i+1)(ds)

ds

=
η
r

b2 +
R ∞
0

1{u≤k}
1
d2
Pk
i=0 ∥∇XΨ(Xi; ai+1, ϵi+1)∥21[i,i+1)(u)

du

=
η
q

b2 + 1

d2
Pk
i=0 ∥∇XΨ(Xi; ai+1, ϵi+1)∥2
,

which is exactly the discrete version of the AdaGrad-Norm stepsize.

Proposition C.8 (Lipschitz). For functions f, g, q such that f(ds) = g(ds) = q(ds) = 0 for s > t,
the AdaGrad stepsize γ is Lipschitz. That is,

|γ(td, f(d × ·), g(d × ·), q(d × ·)) −γ(td, ˆf(d × ·), ˆg(d × ·), ˆq(d × ·))| ≤C(t, γ)(∥g −ˆg∥∞).

Remark C.2. This is a stronger condition than the α-pseudo Lipschitz one in Assumption 6.

Proof. To show this, we look at the derivative of the AdaGrad stepsize function. Setting F(x) =
η
√

b2+x, we have

|F ′(x)| =
η
2(b2 + x)3/2 ≤
η
2b3

30


---Page Break---
for x ∈[0, ∞). We thus have

|γ(td, f(d × ·), g(d × ·), q(d × ·)) −γ(td, ˆf(d × ·), ˆg(d × ·), ˆq(d × ·))|

=



η
q

b2 +
R ∞
0
g(ds) ds
−
η
q

b2 +
R ∞
0
ˆg(ds) ds



=
F
Z ∞

0
g(ds) ds

−F
Z ∞

0
ˆg(ds) ds


≤
η
2b3



Z ∞

0
g(ds) ds −
Z ∞

0
ˆg(ds) ds


≤
η
2b3



Z t

0
g(ds) ds −
Z t

0
ˆg(ds) ds


≤
η
2b3 (t · ∥g −ˆg∥∞)

≤ηt

2b3 · ∥g −ˆg∥∞,

where we were able to replace the ∞with a t because g(ds) = 0 for s > t. We have thus obtained a
Lipschitz constant
ηt
2b3 depending only on t.

Next we show that the AdaGrad-Norm is bounded.
Proposition C.9 (Boundedness). Suppose γ is AdaGrad-Norm. Then (6), as part of Assumption 6,
holds.

Proof. This is immediate:

γ(td, f, g, q) =
η
q

b2 +
R t
0 g(s) ds
≤η

b .

It remains to show that AdaGrad-Norm satisfies (5) in Assumption 6.
Proposition C.10 (Concentration). Suppose γ is AdaGrad-Norm, with Gk and Gk being defined as
before. Then Equation (5), as part of Assumption 6, holds:

E[|γ(Gk) −γ(Gk)| | Fk] ≤Cd−δ(1 + ∥f∥α
∞+ ∥q∥α
∞).

Proof. Looking to remove the square roots, we have

|γ(Gk) −γ(Gk)| ≤|γ(Gk)2 −γ(Gk)2|
1
2 .

For AdaGrad-Norm, we have

γ(Gk)2 −γ(Gk)2 = η2

1

b2 + 1

d2
Pk
j=0 ∥∆j∥2 −
1

b2 + 1

d2
Pk
j=0 E [∥∆j∥2 | Fj]



≤
η2

d2b4 ·



k
X

j=0
(E [∥∆j∥2 | Fj] −∥∆j∥2)


.
(57)

We now bound the sum above. Set Fi = ∥∆i∥2/d, F β
i = Projβ(Fi), ∆Mi = Fi −E[Fi | Fi], and
∆Mβ
i = F β
i −E[F β
i | Fi]. Then |∆Mβ
i | ∈[−2β, 2β], so Azuma’s inequality gives us

P

|Mβ
k| ≥t

≤2 exp

 

−
−t2

2 Pk
i=0(2β)2

!

,

P

|Mβ
k| ≥d1/2+ε
≤2 exp

−
−d1+2ε

2Td(2dε/2)2


= exp

−dε

8T


.

31


---Page Break---
where we set β = dε/2. This is close to the bound we want: the error is

|Mk −Mβ
k| ≤

k
X

i=0
|Fi −F β
i | + | E[Fi −F β
i | Fi]|.

We have

P(Fi −F β
i ̸= 0) = P(|Fi| > β) = P
∥∆i∥2

d
> dε/2

,

which superpolynomially small by (40). The expectation is similar:

| E[Fi −F β
i | Fi]| = | E[(Fi −F β
i )1{|Fi|>β} | Fi]|

≤E[|Fi −F β
i |2 | Fi]
1
2 · E[1{|Fi|>β} | Fi]
1
2

≤4 E[|Fi|2 | Fi]
1
2 · E[1{|Fi|>β} | Fi]
1
2 .

The first expectation is bounded by a constant independent of d by (40), and the second expectation
is superpolynomially small by the same argument as above. We then have

|Mk −Mβ
k| ≤d1/2+ε

with overwhelming probability (note that this would be true for any power of d, by the definition of
superpolynomially small.) We thus conclude that

|Mk| ≤d1/2+ε

with overwhelming probability. Multiplying by d, we find that


k
X

j=0
(E [∥∆j∥2 | Fj] −∥∆j∥2)


≤d3/2+ε
w.o.p.

Plugging this back into (57), we find that
γ(Gk)2 −γ(Gk)2 ≤
η2

d2b4 d3/2+ε

≤Cd−1/2+ε

with overwhelming probability, and so, taking the square root,

|γ(Gk) −γ(Gk)| ≤Cd−1/4+ε/2
w.o.p,

which is less than d−1/4+ε as d grows (we replaced the constant with an extra factor of dε/2.)
Controlling the expectation via the boundedness of γ, we find that with δ = 1/8,

E[|γ(Gk) −γ(Gk)| | Fk] ≤d−δ
w.o.p.,

as desired.

D
Proofs for AdaGrad-Norm analysis

In this section we provide proofs of the propositions related to AdaGrad-Norm in the least squares
setting as well as the more general strongly convex setting. Statements of the propositions for least
squares examples are found in Section 4.

D.1
Strongly convex setting

In order to derive the limiting learning rate in this case, we need the following assumption and some
standard definitions of strong convexity.
Assumption 7 (Risk and loss minimizer). Suppose that

X⋆∈arg minX

R(X) = E a,ϵ[f(⟨X, a⟩, ⟨X⋆, a⟩), ϵ]
	

exists and has norm bounded independent of d. Then one has,

⟨X⋆, a⟩∈arg minx{f(x, ⟨X⋆, a⟩, ϵ)},
for almost surely a ∼N(0, K) and ϵ.

32


---Page Break---
While at first, this assumption seems quite strong, in fact, in a typical student-teacher setup when
label noise is 0 (i.e., ϵ = 0), where the targets have the same model as the outputs, the assumption is
satisfied. Our goal here is not to be exhaustive, but simply to illustrate that our framework admits a
nontrivial and useful analysis and which gives nontrivial conclusions for the optimization theory of
these problems.

Definition D.1 (ˆL-smoothness of outer function f). A function f : R3 →R that is C1-smooth
(in the first variable) is called ˆL(f)-smooth if the following quadratic upper bound holds for any
x, ˆx, y, z ∈R
f(ˆx, y, z) ≤f(x, y, z) + ⟨f ′(x, y, z), ˆx −x⟩+
ˆL(f)

2 |ˆx −x|2.
(58)

Note that if f ′ =
∂
∂xf(x, y, z) is ˆL(f)-Lipschitz, i.e., |f ′(x, y, z) −f ′(ˆx, y, z)| ≤ˆL(f)|x −ˆx|, then
the inequality (58) holds with constant ˆL. Suppose x⋆∈arg minx{f(x, y, z)} exists. An immediate
consequence of (58) is that

1

2ˆL(f)
|f ′(x, y, z)|2 ≤f(x, y, z) −f(x⋆, y, z) ≤
ˆL(f)

2
|x −x⋆|2.
(59)

Definition D.2 (Restricted Secant Inequality). A function f : R3 →R that is C1-smooth (in
the first variable) satisfies the (µ, θ)–restricted secant inequality (RSI) if, for any x ∈R and
x⋆∈arg minx{f(x)},

⟨x −x⋆, f ′(x)⟩≥
µ|x −x⋆|2,
if max{|x⋆|2, |x −x⋆|2} ≤θ,
0,
otherwise.

If f satisfies the above for θ = ∞, then we say f satisfies the µ–RSI.

Proposition D.1. Let the outer function f : R3 →R be a ˆL(f)-smooth function satisfying the
RSI condition with ˆµ(f) with respect to x ∈R. Suppose X⋆∈argminX{R(X)} exists bounded,
independent of d and Assumption 7 holds and that γ0 = η

b =
2ˆµ(f)

(ˆL(f))2 1

d Tr(K)ζ, for some ζ ∈(0, 1),

and that
R ∞
0
R(s)γs ds < ∞with γs as in Table 2 (AdaGrad-Norm, general formula), then

γ∞≥
γ0η2

1 +
ζ
1−ζ D2(0)
.

Proof. Given the Eq. (87) for the distance to optimality, with (x, x⋆) ∼N(0, B),

d
dtD2(t) = −2γt Ea,ϵ[⟨x −x⋆, f ′(x, x⋆)⟩] + γ2
t
d Tr(K)E a,ϵ[(f ′(x, x⋆))2]

By the RSI (with constant ˆµ(f)) condition on f, we have that

E a,ϵ

⟨x −x⋆, f ′(x, x⋆)⟩

≥ˆµ(f)E a,ϵ[(x −x⋆)2] = 2ˆµ(f)R(t),
(60)

where x = ⟨X, a⟩and x⋆= ⟨X⋆, a⟩and we note that x has t-dependence due to the t-dependence in
B. By ˆL(f)-smoothness,
1

2ˆL(f)
(f ′(x))2 ≤
ˆL(f)

2
(x −x⋆)2.

This implies that
1

2(ˆL(f))2 E a,ϵ

(f ′(x, x⋆)2
≤1

2E a,ϵ

(x −x⋆)2
= R(t).
(61)

Thus by (60) and (61), we have that

d
dtD2(t) ≤−γt


4ˆµ(f) −2(ˆL(f))2 1

d Tr(K)γt


R(t)

Which then yield:

D2(t) ≤D2(0) −2

2ˆµ(f) −(ˆL(f))2 1

d Tr(K)γ0

 Z t

0
R(s)γs ds.

33


---Page Break---
Changing variables u = Γ(t) =
R t
0 γs ds, we have that
R ∞
0
R(t)γt dt =
R ∞
0
r(u) du =
∥r∥1. Rearranging the term in the above equation and taking t →∞. We obtain: ∥r∥1 ≤
D2(0)
(2ˆµ(f)−(ˆL(f))2 1

d Tr(K)γ0), given that
ˆ
2µ(f)
(ˆL(f))2 1

d Tr(K)
> γ0.
Using Lemma D.1, with i(v) =

I(B(Γ−1(v))) = E a,ϵ

(f ′(x, x⋆)2
instead of the risk

γ∞=
η2

b
η + 1

2d Tr(K)
R ∞
0
i(v) dv ≥
η2

b
η + 1

d Tr(K)(ˆL(f))2 R ∞
0
r(v) dv
(62)

≥
η2

b
η + 1

d Tr(K)
(ˆL(f))2D2(0)
(2ˆµ(f)−(ˆL(f))2 1

d Tr(K)γ0)

=
η2

b
η +

1
d Tr(K)(ˆL(f))2

2ˆµ(f)(1−ζ)
D2(0)
.

where the first inequality is by Eq. 61, and the last transition is by taking the initial learning rate to be
γ0 =
2ˆµ(f)

(ˆL(f))2 1

d Tr(K)ζ, for ζ ∈(0, 1).

Lemma D.1. Given γt as in Table 2 (AdaGrad-Norm), defining g(u) = γ(Γ−1(u)), with Γ(t) =
R t
0 γs ds, then g(u) =
η2

b
η + 1

2d Tr(K)
R u
0 i(v) dv with i(v) = I(B(Γ−1(v))).

Proof. Taking the square of both sides of the γt equation in Table 2 (AdaGrad-Norm), changing
variables to u = Γ(t) and rearranging the terms:

b2 + Tr(K)

d

Z u

0

i(v)
g(v) dv =
η2

g(u)2 ,
(63)

such that i(v) = I(B(Γ−1(v))). Taking derivative with respect to u, rearranging terms and integrat-
ing leads to the desired result.

D.2
Least squares setting

To study the effect of the structured covariance matrix and cases in which the problem is not strongly
convex, we will focus on the linear least square problem. In this setting, the continuum limit of the
risk for the AdaGrad-Norm algorithm has the form of a convolutional integral Volterra equation,

R(t) = F(Γ(t)) +
Z t

0
γ2
sK(Γ(t) −Γ(s))R(s) ds
(64)

where Γ(t) :=
R t
0 γs ds with,

F(x)
def
= 1

2d

d
X

i=1
λiD2
i (0)e−2λix,
(65)

K(x)
def
= 1

d

d
X

i=1
λ2
i e−2λix.
(66)

In the following we consider three cases, a strongly convex risk in which the spectrum of the
eigenvalues is bounded from below (section D.2.1). A case in which the spectrum is not bounded
from below as d →∞, but the number of eigenvalues below some fixed threshold is o(d) (section
D.2.2). Finally, power law spectrum supported on [0, 1] with d →∞(section D.2.3).

D.2.1
Proofs for case of fixed d

Proof of Proposition 4.2. Define the composite functions r(u) = R(Γ−1(u)), and g(u) =
γ(Γ−1(u)). Integrating the formula for the risk:

34


---Page Break---
Z t

0
r(u) du =
Z t

0
F(u) du +
Z t

0

Z Γ−1(u)

0
γ2
sK(u −Γ(s))R(s) ds du

=
Z t

0
F(u) du +
Z t

0

Z u

0
K(u −x)r(x)g(x) dx du

≤
Z t

0
F(u) du + γ0

Z t

0
r(x)
Z t

x
K(u −x) du dx

Taking t →∞, we get
∥r∥1 ≤∥F∥1 + γ0∥K∥1∥r∥1.

Using ∥K∥1 =
R ∞
0
K(x) dx < γ−1
0 , and noting that by Eq. (66), and Eq. (65), we have that
∥F∥1 = 1

4D2(0), and ∥K∥1 =
1
2d Tr(K),

∥r∥1 ≤
∥F∥1
1 −γ0∥K∥1
=

1
4D2(0)
1 −γ0

2d Tr(K).

On the hand following Lemma D.3, 1

4D2(0)(1 + γ0

2d Tr(K)) ≤∥r∥1. Therefore, ∥r∥1 ≍1

4D2(0).

Next, rewriting the γt equation in Table 2 (AdaGrad-Norm for least squares) in terms of g(u) (Lemma
D.1), we obtain

g(u) =
η2

b
η + 1

d Tr(K)
R u
0 r(x) dx
(67)

Taking u →∞, and using ∥r∥1 ≍1

4D2(0),

γ∞= g(∞) =
η2

b
η + 1

d Tr(K)∥r∥1
≍
η2

b
η + 1

4d Tr(K)D2(0).
(68)

This then completes the proof.

Remark D.1. We note that, on the Least square problem ˆL(f) = ˆµ(f) = 1, therefore, the bound in
Proposition D.1 yields
η2

b
η +
1
2(1−ζ)
1
d Tr(K)D2(0).

Proof of Proposition 4.1. Using the equation for the distance to optimality (Eq. 8), we can derive an
equation for the integral of the risk (with no target noise) which we denote by g(t) =
R t
0 R(s) ds:

g′′(t) = −γt
X

i
λ2
i D2
i (t) + γ2
t
Tr(K2)

d
g′(t).
(69)

For K = Id, this equation simplifies,

g′′(t) = −2γtg′(t) + γ2
t
Tr(K2)

d
g′(t).
(70)

Plugging in the equation for the AdaGrad-Norm learning rate (Table 2) leads to the desired result.
We note that by using the equation for the learning rate, one can also derive a close equation for the
learning rate itself.

D.2.2
Vanishingly few eigenvalues near 0 as d →∞

We now consider the case where, as d →∞, there are eigenvalues of K arbitrarily close to 0. In
Proposition 4.2 we saw a constant lower bound on γt when d is fixed (and thus there are finitely many
eigenvalues within any fixed distance of 0). This can be extended to the case where we have some
C > 0 such that the number of eigenvalues of K below C is o(d) (see Proposition 4.3).

Proof of Proposition 4.3. Following the structure of the loss, after some time the risk starts to de-
crease, and therefore R(t) ≤R0 for and t ≥0. Using these observations, we obtain a preliminary
lower bound of γt > C1t−1/2 (for t > 0), which enables us to deduce that R(t) is integrable and
finally obtain a constant lower bound for γt. The details of this are below.

35


---Page Break---
For t ≥0 and some C1 > 0,

γt =
η
q

b2 + 2

d Tr(K)
R t
0 R(s)ds
≥
η
q

b2 + 2

d Tr(K)R0t
≥C1t−1/2.
(71)

Next, to show that the risk is integrable, we divide the matrix K into two parts K+, and K−, such
that the eigenvalues of K+ are greater than some αs > 0 and the eigenvalues of K−are smaller than
αs where αs is a decreasing function of s to be determined later. We then have that, following Eq.
(8), and the definition of the risk R(t) =
1
2d
Pd
i=1 λiD2
i (t),

R(t) = R(0) −1

d

d
X

i=1
λ2
i

Z t

0
γsDi(s)ds + 1

d

Z t

0
γ2
s Tr(K2) · R(s)ds
(72)

≤R(0) −
Z t

0
γs(2αs −γs
1
d Tr(K2)) · R(s)ds + 2
Z t

0
γsR2(s) ds

with R2(s) =
1
2d
P

i:λi≤αs λiD2
i (s). Next, choosing αs = γs 1

d Tr(K2), we show that the last term
is of order od(1). By Lemma D.2 ∀i, D2
i (t) ≤max
 
γt1R(t1), D2
i (0)

= c0 where the bound c0
comes from the assumption ⟨X⋆, ωi⟩= O(d−1/2) and the initialization X0 = 0. Therefore,

2
Z t

0
γsR2(s) ds ≤1

d2 Tr(K2)c0

Z t

0
γsNs ds.
(73)

where Ns = Pd
i=1 1λi≤γs 1

d Tr(K2). This implies that, if γsNs = o(d), then 2
R t
0 γsR2(s) ds =
od(1), provided that d is taken to be large before t.

We then have that up to od(1) constant,

R(t) ≤R(0) −1

d Tr(K2)
Z t

0
γ2
s · R(s)ds.
(74)

Using Gronwall’s inequality,

R(t) ≤R(0)e−1

d Tr(K2)
R t
0 γ2
s ds ≤R(0)e−1

d Tr(K2)C2
1t
(75)

where in the last transition we used the lower bound on the learning rate derived in Eq. (71). Thus,
the risk is integrable, i.e. there is some C3 such that
Z t

0
R(s) ds ≤
R(0)
1
d Tr(K2)C2
1
for all t > 0. Finally, we plug this into the formula for γt and conclude that, for all t > 0,

γt ≥
η
r

b2 +

1
d Tr(K)R(0)

1
d Tr(K2)C2
1

.
(76)

Lemma D.2. Assume that the risk is bounded and attains its maximum at time t1. Then, for each i,
we have D2
i (t) ≤max(γt1R(t1), D2
i (0)) for all t ≥0.

Proof. Case 1: Suppose that D2
i (0) ≤γ0R(0). Then, by equation (8),
d
dtD2
i (0) ≥0. However,
since D2
i (t), R(t) are continuous, this equation implies that D2
i (t) ≤γtR(t) for all t and thus
D2
i (t) ≤γt1R(t1) for all t.

Case 2: Suppose that D2
i (0) > γ0R(0). Then, by equation (8), d

dtD2
i (0) < 0. If d

dtD2
i (t) < 0 for
all t, then D2
i (t) ≤D2
i (0) for all t. If at some point d

dtD2
i (t) > 0, this implies D2
i (t) ≤γtR(t) and
we are in Case 1.

In the next section, we consider cases in which the risk is not integrable, an example of such case is
when the spectrum of K is supported on the interval [0, 1] or has power-law behavior near 0.

36


---Page Break---
D.2.3
Power law behavior at d →∞

Non-asymptotic bound for the Convolutional Volterra
In this section, we use the convolutional
Volterra structure of the risk (Eq. (64)) to derive non-asymptotic bounds on the risk, which will
be useful in Section D.2.3 to derive the asymptotic behavior of the risk and the learning rate under
power law assumption on the spectrum of the covariance matrix and the discrepancy from the target
at initialization.

Lemma D.3. Let Γ(t) :=
R t
0 γs ds and let

R(t) = F(Γ(t)) +
Z t

0
γ2
sK(Γ(t) −Γ(s))R(s) ds

where γt, K are monotonically decreasing, with ∥K∥1 < ∞. Then all t,

R(t) ≥F(Γ(t)) +
Z t

0
γ2
sK(Γ(t) −Γ(s))F(Γ(s)) ds

If in addition, there exist ϵ > 0 and T > 0 such that, for all t > T,
Z t

0
K(s)K(t −s) ds ≤2(1 + ϵ)∥K∥1K(t)
and
2∥K∥1(1 + ϵ)γ0 < 1

then for all t

R(t) ≤F(Γ(t)) + C
Z t

0
γ2
sK(Γ(t) −Γ(s))F(Γ(s)) ds

for

C =

K(0)
K(T)(2ϵ + 1) + 2

1
1 −2γ(0)∥K∥1(1 + ϵ).

Proof. The lower bound holds trivially, using R(s) ≥F(Γ(s)). For the upper bound, we start with
the following change of variables:

R(t) = F(Γ(t)) +
Z Γ(t)

0
g(u)K(Γ(t) −u))R(u) du,

with g(u) = γΓ−1(u). Let us define the convolution map

G(f)(Γ) = K ∗(gf)(Γ) =
Z Γ

0
K(Γ −u)g(u)f(u) du.

Next we show that this map is contracting and in particular,

G2(f) = G(G(f))(t) =
Z t

0
K(t −s)G(f)(s)g(s) ds
(77)

=
Z t

0
K(t −s)
Z s

0
K(s −u)g(u)f(u) dug(s) ds

=
Z t

0

Z t

u
K(t −s)K(s −u)g(s) ds

g(u)f(u) du

≤
Z t

0
K∗2(t −u)g(u)2f(u) du

where the third transition is since u < s < t. The last transition is by change of variables and the
assumption that γt is a monotone decreasing function. Consecutive application of the convolution
map will then yield by induction,

Gj(f)(t) ≤
Z t

0
K∗(j)(t −u)g(u)jf(u) du.

37


---Page Break---
Therefore, expanding the loss and using the above upper bound, and denote by q = 2(1 + ε)∥K∥1γ0
such that q < 1,

R(t) = F(t) +

∞
X

j=1
Gj(F)(t)
(78)

≤F(t) +

∞
X

j=1

Z t

0
K∗(j)(t −u)g(u)jF(u) du

≤F(t) +




∞
X

j=0
(2∥K∥1γ0(1 + ε))j −1



C1

Z t

0
K(t −u)g(u)F(u) du

≤F(t) +
q
1 −q C1(K ∗(gF))(t)
(79)

where the third transition is by Lemma D.4, with C1 =
K(0)
K(T )(2ϵ+1) + 1, which then completes the
proof.

Lemma D.4 (Lemma IV.4.7 in [3]). Suppose K is monotonically decreasing, with ∥K∥1 < ∞, and
that there exists T > 0 such that ∀t ≥T, and ϵ ≥0,
Z t

0
K(s)K(t −s) ds ≤2(1 + ϵ)∥K∥1K(t).
(80)

Then,

sup
t≥0

K∗n(t)

K(t)
≤(2∥K∥1(1 + ϵ))n−1

K(0)
K(T)(2ϵ + 1) + 1

(81)

Proof. Define αn = supt≥0
K∗n(t)
K(t)(2∥K∥1)n−1 , trivially α1 = 1. Consider the n + 1 convolution,

K∗(n+1)(t)
K(t)(2∥K∥1)n =
1
K(t)

Z t

0

K(s)K∗n(t −s)

(2∥K∥1)n
ds
(82)

By the assumption of the Lemma, we know that there exists some T > 0 such that for ∀t ≥T
Z t

0

K(s)K(t −s)

2∥K∥1
ds ≤(1 + ϵ)K(t).
(83)

Therefore, if t ≥T, we have

1
K(t)

Z t

0

K(s)K∗n(t −s)

(2∥K∥1)n
ds
(84)

=
Z t

0

K(s)K(t −s)

2∥K∥1

K∗n(t −s)
K(t −s)(2∥K∥1)n−1 ds ≤αn(1 + ϵ)

On the other hand, if t < T,

1
K(t)

Z t

0

K(s)K∗n(t −s)

(2∥K∥1)n
ds ≤K(0)

K(T)
∥K∗n(t)∥1

(2∥K∥1)n ≤
K(0)
K(T)2n
(85)

Taking supremum in Eq. (82), and combining the results of Eq. (85), and Eq. (84), we obtain that,

αn+1 ≤
K(0)
K(T)2n + αn(1 + ϵ)

Solving the above recursion equation,

αn ≤K(0)

K(T)

n−2
X

k=0

1
2n−k−1 (1 + ϵ)k + (1 + ϵ)n−1 =
K(0)
K(T)2n−1
1 −(2(1 + ϵ))n−1

1 −2(1 + ϵ)
+ (1 + ϵ)n−1

≤(1 + ϵ)n−1

K(0)
K(T)(2ϵ + 1) + 1

,

rearranging the terms we arrived at the required result.

38


---Page Break---
Asymptotic analysis of the risk
Here, we consider a family of models with d →∞, for which
the following power law asymptotics assumption is satisfied:
Assumption 8. F(x) ≍x−κ1 and K(x) ≍x−κ2 for x ≥1 with κ1 ≥0, κ2 > 1

Corollary D.1 apply Lemma D.3 in the setting for which F, and K has a power law behavior
asymptotically. It shows that the risk will then be dominated by F only. Corollary D.2 shows the
behavior of the learning rate in this setting. Finally, Lemma D.5 shows that Assumption 8 is a
consequence of a power law spectrum near zero on the eigenvalues of the covariance matrix and a
power law assumption on the projected discrepancy at initialization.
Corollary D.1. Suppose Assumption 8 is satisfied, then R(t) ≍F(Γ(t)).

Proof. Define g(u) = γΓ−1(u) and r(u) = R(Γ−1(u)) and observe that g(u) is a decreasing function.
Then, from the upper bound in Lemma D.3, we have

r(u) ≤F(u) + C
Z u

0
g(v)K(u −v)F(v) dv

= F(u) + C

 Z u/2

0
g(v)K(u −v)F(v) dv +
Z u

u/2
g(v)K(u −v)F(v) dv

!

≤F(u) + C1g(0)

 u

2

−κ2 Z u/2

0
F(v) dv +
u

2

−κ1 Z u

u/2
K(u −v) dv

!

≤F(u) + C2(u−κ2+1−κ1 + u−κ1∥K∥)
= O(F(u)).

(86)

Combining this upper bound with the lower bound from Lemma D.3 and that κ2 > 1, we conclude
that r(u) ≍F(u) and R(t) ≍F(Γ(t)).

Next, we derive the asymptotics of γt. There are three different cases, depending on whether the risk
is integrable, which translates to a threshold with respect to the parameter κ1.
Corollary D.2. Suppose Assumption 8 then the following asymptotics for the learning rate hold:

• For κ1 > 1, there exists ˜γ such that γt ≥˜γ and R(t) ≍t−κ1 for all t ≥0.

• For κ1 < 1, γt ≍t−(1−κ1)/(2−κ1) and R(t) ≍t−
κ1
2−κ1 for all t ≥1.

• For κ1 = 1, γt ≍
1
log(t+1) and R(t) ≍(
t
log(t+1))−κ1 for all t ≥1.

Proof. Using the notations g(u) and r(u) defined above along with the change of variable u = Γ(t),
we get
R t
0 R(s) ds =
R u
0
r(v)
g(v) dv. Combining this with Corollary D.1 and the formula for γt we get

g(u) ≍
η
q

b2 + 2

dTr(K)
R u
0
(1+v)−κ1

g(v)
dv
.

Let I(u) = b2 + 2

dTr(K)
R u
0
(1+v)−κ1

g(v)
dv and observe that g(u) ≍
1
√

I(u) and I′(u) =

2
dTr(K) (1+u)−κ1

g(u)
. Thus, I(u) satisfies
I′(u)
√

I(u) ≍(1 + u)−κ1 so we have

p

I(u) −
p

I(0) ≍
Z u

0
(1 + v)−κ1 dv.

In the case of κ1 > 1, this implies
p

I(u) ≤
p

I(0) + C
R
(1 + v)−κ1 dv. This upper bound on I(u)
gives a corresponding lower bound on g(u) and thus a lower bound on γt.

In the case of κ1 < 1, we have
p

I(u) −
p

I(0) ≍(1 + v)1−κ1 so, for u sufficiently large, g(u) ≍
(1 + u)κ1−1. To recover the asymptotic for γt, we observe that
d
duΓ−1(u) =
1
g(u) ≍(1 + u)1−κ1.
Integrating both sides and changing back to t variables, we get t ≍(1 + Γ(t))2−κ1 (or equivalently

39


---Page Break---
1 + Γ(t) ≍t1/(2−κ1)). Finally, plugging this into the formula for γt and applying Corollary D.1, we
get
γt ≍
η
q

b2 + 2

dTr(K)
R t
0 F(Γ(s)) ds
≍(1 + t)−(1−κ1)/(2−κ1).

In the case of κ1 = 1, we follow a similar procedure as for κ1 < 1 to show that t ≍Γ(t) log(Γ(t))
for sufficiently large t. This implies Γ(t) ≍t/ log(t) which gives the desired result after integration.
The decay rate of the risk is then immediate using Corollary D.1.

Lemma D.5. Let K have a spectrum that converges as d →∞to the power law measure ρ(λ) =

Cλ−β1(0,λmax), with C−1 = λ1−β
max
1−β for some β < 1, and λmax > 0, and suppose that D2
i (0) ∼λ−δ
i ,
then F(t) ≍t−κ1, and K(t) ≍t−κ2, with κ1 = 2−β−δ, and κ2 = 3−β. In addition, K(t) ≍t−κ2,
satisfies Eq. (80).

Proof. Following the definition in Eq. (66), and Eq. (65)

F(x) = 1 −β

2λ1−β
max

Z λmax

0
λ1−β−δe−2λx dλ

=
1 −β

2λ1−β
max(2x)2−β−δ

Z 2λmaxx

0
y1−β−δe−y dy =
1 −β

λ1−β
max23−β−δ
γ(2 −β −δ, 2λmaxx)

x2−β−δ
.

Similarly for K,

K(x) = 1 −β

λ1−β
max

Z λmax

0
λ2−βe−2λx dλ =
1 −β

λ1−β
max23−β
γ(3 −β, 2λmaxx)

x3−β
.

with γ(s, z) =
R z
0 xs−1e−x dx is the incomplete gamma function. For large z, γ(s, z) ≍Γ(s), the
complete gamma function. We therefore obtain κ1 = 2 −β −δ, and κ2 = 3 −β. Next, we show
that K(x) ≍x−κ2 satisfies Eq. (80),
Z t

0
K(s)K(t −s) ds ≤
Z t/2

0
K(t)K(t −s) ds +
Z t

t/2
K(t)K(t −s) ds

≤K(t/2)

 Z t/2

0
K(s) ds +
Z t

t/2
K(t −s) ds

!

≤2K(t/2)∥K∥1

by the power-law assumption for t > T, K(t/2) ≍K(t) which then complete the proof.

Proof of Proposition 4.4. The proof is an immediate application of Corollary D.2 with, κ1 = 2−β−δ
as implied by Lemma D.5.

Remark D.2. This includes the case β = 0, which is the uniform measure on [0, λmax].

E
Polyak Stepsize

The distance to optimality of SGD is measured say by D2(X) = ∥X −X⋆∥2. Let us consider
the deterministic equivalent for the distance to optimality D2(t) in (11). Fixing T > 0 and any
ε ∈(0, 1/2), we have by Theorem 2.1 (see also corollary B.1 which show concentration for large
class of statistics) that sup0≤t≤T |∥X⌊td⌋−X⋆∥2 −D2(t)| ≤d−ε, w.o.p. In this way, if we want to
guarantee that the distance to optimality of SGD decreases, we need dD2(t) < 0 with the maximum
decrease being minγt dD2(t).

As it turns out, the evolution of D2 is particular simple, as it solves the differential equation (derived
from the ODE in (9))

d
dtD2(t) = −2γtA(B(t)) + γ2
t
d Tr(K)I(B(t)),








A(B) = Ea,ϵ[⟨x −x⋆, f ′(x ⊕x⋆)⟩],

I(B) = Ea,ϵ[f ′(x ⊕x⋆)2],
where
(x ⊕x⋆) ∼N(0, B).
(87)

40


---Page Break---
Figure 5: Convergence in Exact Line Search on a noiseless least squares problem. The plot on
the left illustrates the convergence of the risk function, while the center and right plots depict the
convergence of the quotient
Dλ2(t)
Dλ1(t) and the learning rate γt, respectively. Further details and formulas
for the limiting behavior can be found in the Appendix F.2. See Appendix H for simulation details.

The distance to optimality threshold, ¯γD
t , occurs precisely when dD2 < 0. This choice of γ makes
the ODE for the distance to optimality stable. By translating the relevant deterministic quantities in
¯γD
t back to SGD quantities, we get

¯gD
k
def
=
2⟨Xk −X⋆, ∇R(Xk)⟩

Tr(K)

d
E a,ϵ[f ′(⟨Xk, a⟩; ⟨X⋆, a⟩, ϵ)2]
with the deterministic equiv. ¯γD
t =
2A(B(t))

Tr(K)

d
I(B(t))
.

(88)
A greedy learning rate that maximizes the decrease at each iteration is simply given by gPolyak
t
∈
arg min dD2(t). This has a closed form and we call this Polyak stepsize11. Again translating this
back to SGD, we have

Polyak learning rate
gPolyak
k
= 1

2¯gD
k
and
deterministic equivalent
γPolyak
t
= 1

2 ¯γD
t .
(89)

In this context, the Polyak learning rate is impractical because we do not known X⋆. In spite of this,
we can learn some things about this learning rate as it is the natural extension of Polyak learning rate
to SGD.

The quantities A(B) and I(B) in (88) and (89) only depend on the low-dimensional function f
and thus do not carry any covariance K or d dependence. Moreover, under additional assumptions
on the function such as (strong) convexity, we can bound from below A(B)/I(B). Thus, in terms
covariance K and d, the Polyak stepsize gPolyak
k
≍
1
Tr(K)/(d) =
1
avg. eig of K .

In the case of least squares (see (7)), we get

gPolyak
k
= 2R(Xk) −ω2

2Tr(K)

d
R(Xk)
and on a noiseless least squares,
gPolyak
k
=
1

Tr(K)

d
.

The latter gives the best fixed learning rate for a noiseless target on a LS problem (as noted in
[35, 43]).

F
Line Search

F.1
General Line Search

Naturally, one can ask a similar question as in Polyak in the context of line search (i.e., decreasing
risk at each iteration of SGD). First, by the structure of the risk (Assumption 3 and 4),

∥∇R(X)∥2 = m(W T K2W)
and
Tr(∇2R(X)K) = v(K).
(90)

Therefore using (9), we have that the deterministic equivalent for ∥∇R(X)∥2 is M (t) =
1
2
Pd
i=1 m(Vi(t)λ2
i ). In this case, the deterministic equivalent for the risk R satisfies the following
ODE

dR = −γtM (t) dt + γ2
t
d v(K)I(B(t)).
(91)

11This is the idea of Polyak stepsize when the problem is deterministic.

41


---Page Break---
From this, we get an immediate learning rate (stability) threshold for the risk, that is, ¯gR
k is the largest
learning rate for which SGD is guaranteed to decrease at each iteration, i.e., when the deterministic
equivalent of R satisfies dR < 0 or equivalently after translating relevant terms into SGD quantities

risk threshold
¯gR
k =
∥∇R(Xk)∥2

Tr(K∇2R(Xk))

d
I(W T
k KWk)
and deterministic equiv
¯γR
t =
M (t)

v(K)

d I(B(t))
.

(92)
The greediest approach, which we call exact line search, would choose the learning rate such that
γline
t
∈arg minγ dR. In this case, we get

gline
k
= 1

2gR
k
and
deterministic equiv
γline
t
= 1

2γR
t .

F.2
Line Search on least squares

In this section, we provide a proof of Proposition 3.1, but, we show more than this including the exact
limiting value for γt.

Proposition F.1. Consider the noiseless (ω = 0) least squares problem (7) . Then the learning rate
is always lower bounded by
λmin(K)
1
d Tr(K2) ≤γline
t
for all t ≥0.

Moreover, suppose K has only two distinct eigenvalues λ1 > λ2 > 0, i.e., K has d/2 eigenvalues
equal to λ1 eigenvalues and d/2 eigenvalues equal to λ2. In this context, the exact limiting value of
γline
t
is given by

lim
k→∞γline
t
=
2
 
λ2
1 + λ2
2x


(λ1 + λ2x) (λ2
1 + λ2
2),
(93)

where x is the positive real root of the second-degree polynomial

P(x) = λ1λ2(x + 1)(λ2x −λ1) + (λ2 −λ1)3x.
(94)

This leads to
λmin(K)
1
d Tr(K2) ≤lim
t→∞γline
t
≤2λmin(K)

1
d Tr(K2).
(95)

Proof. We establish the inequality

λmin(K)
1
d Tr(K2) ≤γline
t
for all t ≥0

by observing

1
d

d
X

i=1
λ2
i D2
i (t) ≥2λmin(K) 1

2d

d
X

i=1
λiD2
i (t) = 2λmin(K)R(t).

Now let us consider K ∼1

2λ1 + 1

2λ2 for λ1 > λ2 > 0.

We define Dλ(t)
def
= Pd
λi=λ D2
i (t). Utilizing the ODEs in (9), we derive

d
dtDλ(t) = −2γtλDλ(t) + 2γ2
t λ × |{λ = λi}d
i=1| × R(t)

for each distinct eigenvalue λ of K. Here |{λ = λi}d
i=1| is the number of eigenvalues of K that
are equal to λ. It immediately follows by our construction of K that |{λ = λi}d
i=1| = d

2. Thus, we
establish the following system of ODEs
 d

dtDλ1(t) = −2γtλ1Dλ1(t) + dγ2
t λ1R(t)
d
dtDλ2(t) = −2γtλ2Dλ2(t) + dγ2
t λ2R(t)
(96)

where R(t) =
1
2d (λ1Dλ1(t) + λ2Dλ2(t)) and γline
t
=
2(λ2
1Dλ1(t)+λ2
2Dλ2(t))
(λ1Dλ1(t)+λ2Dλ2(t))(λ2
1+λ2
2).

42


---Page Break---
Since Dλ2(t) ≥0 and λ1 > λ2 > 0, we infer that R(t) =
1
2d (λ1Dλ1(t) + λ2Dλ2(t)) ≥
1
2dλ1Dλ1(t) ≥0. The structure of the exact line search algorithm ensures limt→∞R(t) = 0,
hence limt→∞Dλ1(t) = 0. Similarly, we deduce limt→∞Dλ2(t) = 0.

By applying L’Hôpital’s rule and substituting the expressions for γline
t
and R(t) in terms of Dλ1(t)
and Dλ2(t), we derive

lim
t→∞
Dλ2(t)
Dλ1(t) = lim
t→∞
dDλ2(t)
dDλ1(t)

= lim
t→∞
−2γtλ2 Dλ2(t) + dγ2
t λ2R(t)
−2γtλ1 Dλ1(t) + dγ2
t λ1R(t)

= lim
t→∞
−2λ2 Dλ2(t) + dγtλ2R(t)
−2λ1 Dλ1(t) + dγtλ1R(t)

= lim
t→∞
γt λ1λ2

2 Dλ1(t) + λ2Dλ2(t)
 
γt λ2

2 −2


γt λ1λ2

2 Dλ2(t) + λ1Dλ1(t)
 
γt λ1

2 −2


= lim
t→∞
Dλ1(t)2λ3
1λ2 + Dλ1(t)Dλ2(t)(−λ1λ3
2 + λ2
1λ2
2 −2λ3
1λ2) + Dλ2(t)2(−λ4
2 −2λ2
1λ2
2)
Dλ1(t)2(−λ4
1 −2λ2
1λ2
2) + Dλ1(t)Dλ2(t)(−λ3
1λ2 + λ2
1λ2
2 −2λ1λ3
2) + Dλ2(t)2λ1λ3
2

=
λ3
1λ2 + limt→∞
Dλ2(t)
Dλ1(t)(−λ1λ3
2 + λ2
1λ2
2 −2λ3
1λ2) +

limt→∞
Dλ2(t)
Dλ1(t)
2
(−λ4
2 −2λ2
1λ2
2)

(−λ4
1 −2λ2
1λ2
2) + limt→∞
Dλ2(t)
Dλ1(t)(−λ3
1λ2 + λ2
1λ2
2 −2λ1λ3
2) +

limt→∞
Dλ2(t)
Dλ1(t)
2
λ1λ3
2

.

Therefore, limt→∞
Dλ2(t)
Dλ1(t) is the positive real root of the second-degree polynomial

P(x) = λ1λ2(x + 1)(λ2x −λ1) + (λ2 −λ1)3x.
(97)

Solving for x > 0, we derive the explicit formula

lim
t→∞
Dλ2(t)
Dλ1(t)

= λ3
1 −2λ2
1λ2 + 2λ1λ2
2 −λ3
2 +
p

λ6
1 −4λ5
1λ2 + 8λ4
1λ2
2 −6λ3
1λ3
2 + 8λ2
1λ4
2 −4λ1λ5
2 + λ6
2
2λ1λ2
2
.
(98)

Given

γline
t
=
2
 
λ2
1Dλ1(t) + λ2
2Dλ2(t)


(λ1Dλ1(t) + λ2Dλ2(t)) (λ2
1 + λ2
2) =
2

λ2
1 + λ2
2
Dλ2(t)
Dλ1(t)



λ1 + λ2
Dλ2(t)
Dλ1(t)

(λ2
1 + λ2
2)
,
(99)

we have

lim
t→∞γline
t
=
2

λ2
1 + λ2
2 limt→∞
Dλ2(t)
Dλ1(t)



λ1 + λ2 limt→∞
Dλ2(t)
Dλ1(t)

(λ2
1 + λ2
2)
.
(100)

By substituting (98), we get

lim
t→∞γline
t

= λ3
1 + 2λ2
1λ2 + 2λ1λ2
2 + λ3
2 −
p

λ6
1 −4λ5
1λ2 + 8λ4
1λ2
2 −6λ3
1λ3
2 + 8λ2
1λ4
2 −4λ1λ5
2 + λ6
2
(λ2
1 + λ2
2)2
.

(101)
A direct calculation reveals that λ1 > λ2 > 0 implies limt→∞γline
t
≤2λmin(K)

1
d Tr(K2) .

Remark F.1. For the scenario where K has an arbitrary number n of distinct eigenvalues, equation
(13) remains valid. The proof parallels the one outlined above. However, in this case, the expression
for limk→∞gk is given by

lim
k→∞gk =
n
 
λ2
1 + λ2
2x1 + · · · + λ2
nxn−1


(λ1 + λ2x1 + . . . λnxn−1) (λ2
1 + · · · + λ2n),
(102)

where x1, . . . , xn−1 > 0 satisfy a more intricate coupled system of n −1 equations.

43


---Page Break---
G
Examples

Any single index model with α-pseudo Lipschitz (α ≤1) activation function is covered by our
SGD+AL theory. In this section, we provide key learning problems within this family of models.

G.1
Binary logistic regression

We consider a binary logistic regression problem with ϵ = 0 where we are trying to classify two
classes. We will follow a Student-Teacher model, in which there exists a true vector X⋆to be the true
direction such that possible labels are, y =
exp(⟨X⋆,a⟩)
exp(⟨X⋆,a⟩)+1. or 1 −y. In order to classify the data we
minimize the KL-divergence between the label y and our estimate defined by the above formula,

R(X) = E a


−⟨X, a⟩·
exp(⟨X⋆, a⟩)
exp(⟨X⋆, a⟩) + 1 + log (exp(⟨X, a⟩) + 1)

.
(103)

To study the ODE dynamics of SGD in Eq. (9) one needs the deterministic risk h(B), and I(B) =
E a[f ′(⟨X, a⟩, ⟨X⋆, a⟩)2], with B = W T KW. Following the computation in Appendix D example
D.4 in [15] we obtain that

h(B) = −B21E z


exp(√B22 · z)
(1 + exp(√B22 · z))2


+ E w

log(exp(w
p

B11) + 1)

,
(104)

where z, w ∼N(0, 1). The I function can also be computed explicitly by solving the following
Gaussian integral, where we define g(x)
def
=
exp(x)
1+exp(y)

I(B) =
1

2π
p

det(B)

Z

R2(g(x) −g(y))2 exp

−1

2

x
y

T
B−1
x
y

 
dx dy.
(105)

We note that, the logistic regression is (µ, θ)–RSI with µ =
1
ℓe
√

4θ see section 2.2 in [15]. Its Lipschitz

constant is ˆL(f) = 1. Using Proposition D.1 one can derive a lower bound on the limiting learning
of AdaGrad Norm.

For more details and more examples, see [15].

G.2
CIFAR 5m

Finally, we include an example that uses real-world data, that is, the CIFAR 5m dataset [38]. Our
theory does not explicitly deal with non-Gaussian distributions, but we find that the theoretical risk
curves generalize cleanly to that case.

As we are now working with discrete data points rather than a distribution, the learning setup, while
closely analogous to what was presented earlier, has some slight differences.

We start with a subset of the data consisting of n grayscale images, each of which is 32 × 32 pixels,
that is, A ∈Rn×1024. We fill a vector b ∈Rn with the corresponding labels (0 for an image of a
plane, 1 for an image of a car.) We then randomly choose a matrix W ∈R1024×d with i.i.d. Gaussian
entries to generate the features F = relu(AW). We want to use least squares to predict the label
from the features, i.e., find

arg minX∈Rd

(

R(X) := 1

2n ∥FX −b∥2 = 1

2n

n
X

i=1
(fi · X −bi)2
)

,
(106)

where fi is the ith row of F. The SGD we now consider is

Xk+1 = Xk −γk
 
fik+1 · X −bik+1

fik+1,
{ik} iid Unif({1, 2, · · · , n}),
(107)

where γk is the usual AdaGrad-Norm stepsize, as in (15). Our empirical covariance matrix K
(remembering that fi is a row vector) is then

K = E i∈[n],j∈[n]

f ⊤
i fj

= 1

nF ⊤F.
(108)

We now use (64), with the AdaGrad-Norm stepsize, to numerically simulate the SGD loss, which we
then compare to the actual loss. Our theory matches empirical results very closely.

44


---Page Break---
100
101
102
103

SGD Iterations/d

10
1

2 × 10
2

3 × 10
2

4 × 10
2

6 × 10
2

Empirical Risk

CIFAR AdaGrad-Norm Least Squares

n = 2048
n = 4096
n = 8192
n = 16384

Figure 6: Predicting the training dynamics on a real dataset, CIFAR-5m [38], using multi-pass
AdaGrad-Norm. This suggests the theory extends beyond Gaussian data and one-pass. Note that the
curves look significantly different for different n; smaller values of n lead to an overparametrized
problem, allowing least squares to memorize datapoints, whereas for larger n, least squares must
learn a general function mapping images of cars and airplanes to their respective labels.

H
Numerical simulation details

Here we provide more details for the figures that appear in the main paper.

Figure 1:
Concentration learning rate and risk for AdaGrad-Norm on a least squares problem
with label noise ω = 1 (left) and on a logistic regression problem with no label noise (right). For
logistic, see Section G. 30 runs of AdaGrad-Norm with parameters b = 1 and η = 1 for each d;
X⋆∼N(0, Id/d), X0 = 0, and K = Id. The shaded region represents a 90% confidence interval
for the SGD runs. As the dimension increases, the risk and stepsize both concentrate around a
deterministic limit (red). The deterministic limit is described by an ODE in Theorem 2.1. The initial
loss increase in the least squares problem suggesting that the learning rate was initially too high, but
AdaGrad-Norm naturally adapts and still the loss converges. Our ODEs predict this behavior.

Figure 2:
Comparison for Exact Line Search and Polyak Stepsize on a noiseless least squares
problem. The left plot illustrates the convergence of the risk function, while the right plot depicts
the convergence of the quotient γt/ λmin(K)

1
d Tr(K2) for Polyak stepsize and exact line search. Both ODE
theory and SGD results are presented, showing a close agreement between the two approaches.
The covariance matrix K is generated such that the eigenvalues follow the expression λi(K) =
r

d
Pd
i=1(
i
d+1)
−2/s ·

i
d+1
−1/s
,
i = 1, . . . , d, where s > 2 is a constant. As s approaches 2,

the spectrum becomes more spread out, resulting in larger values of 1

d Tr(K2). Larger values of
s correspond to smaller spreads in the spectrum. Additionally, Tr(K)/d = 1 for all s. Both plots
highlight the implication of equation (13) in high-dimensional settings, where a broader spectrum
of K results in λmin(K)

1
d Tr(K2) ≪
1
1
d Tr(K), indicating slower risk convergence and poorer performance

of exact line search (unmarked) as it deviates from the Polyak stepsize (circle markers). The gray
shaded region demonstrates that equation (13) is satisfied.

Figure 3:
Quantities effecting AdaGrad-Norm learning rate. (left): The effect of adding
noise to the targets (ω = 1.0) to the risk (left axis) and learning rate (right axis). Ran AdaGrad-
Norm(b = 1.0, η = 2.5) on least squares problem with d = 500. X0, X⋆∼N(0, Id/d). A single
run of the SGD (solid line purple) matches exactly the prediction (ODE, teal). The shaded region
represents 10 runs of SGD with 90% confidence interval. The learning rate decays at the exact
predicted rate of
η
q

b2+ TrKω2

d
t. Depicted is learning rate

asymptotic so it approaches 1. (center, right): Noiseless

least squares setting (ω = 0). (center): Prop. 4.2 predicts the avg. eig of K (Tr(K)/d) as compared
with λmax effects the limk→∞gk. Indeed, this is true. We varied the κ = λmax/λmin while keeping
the Tr(K)/d and all other parameters fixed. All the learning rates behave identically verifying our

45


---Page Break---
theory about the effect of Tr(K)/d on learning rates. (right): Varying the learning rate of AdaGrad
norm by ∥X0 −X⋆∥2; our predictions (dashed) match and we see the inverse relationship predicted
by Prop. 4.2. See Appendix D for details. Additionally, we did the following.

• Center plot: AdaGrad with b = 0.5, η = 2.5 is run on the least squares problem with
d = 1000 and X0, X⋆∼
1
√

dN(0, I). The covariance matrix K is generated so that the
eigenvalues are

λi(K) =

v
u
u
t

d
Pd
i=1

i
d+1
−2/s ·

i
d + 1

−1/s
,
i = 1, . . . , d.

The constant s > 2. When s is near 2, the spectrum is more spread out, i.e., κ = λmax

λmin is
large. Larger values of s mean smaller the spreads. Moreover Tr(K)/d = 1 for all s. In the
simulations, we used s ∈{2.1, 3.0, 3.5, 4.0, 5.5} and recorded the condition number κ.
• Right plot: Ran AdaGrad with b = 0.5, η = 2.5 on the least squares problem with d = 1000.
X⋆= 0 and X0 ∼
p p

dN(0, I) where p ∈{1, 2, 4, 8, 16}. In this way, ∥X0 −X⋆∥2 = p.

Figure 4:
Power law covariance in AdaGrad Norm on a least squares problem. Generated
covariance K such that the density of eigenvalues are (1 −β)λ−β where β = 0.2 and set X0 = 0.
Choose (X⋆
i )d
i=1 = (λ−δ/2
i
)d
i=1 where λi is the i-th eigenvalue of K and we vary δ ∈(0, 1.8) so that
0 < δ + β ≤2. Setting of Prop. 4.4.

Figure 5:
Convergence in Exact Line Search on a noiseless least squares problem. The plot on
the left illustrates the convergence of the risk function, while the center and right plots depict the
convergence of the quotient
Dλ2(t)
Dλ1(t) and the learning rate γt, respectively. Predictions from ODE
theory are compared with results obtained from SGD, demonstrating close agreement between the
two approaches. Initialization was performed randomly, with X0 ∼N(0, Id/d) and X⋆∼
1
√

d1,
where d = 400. The covariance matrix K has two distinct eigenvalues λ1 = 1 > λ2 > 0, and
was constructed by specifying the spectrum, with λi sampled from a discrete uniform distribution
U{1, λ2} for i = 1, . . . , d = 400, and setting K = diag(λi : i = 1, . . . , 400). Further details and
formulas for the limiting behavior can be found in the Appendix F.2.

Figure 6
Convergence on CIFAR 5m [38]. We train a classifier to distinguish between images
of airplanes and cars. Fix d = 2000. Then for multiple values of n, we run AdaGrad-Norm with
initialization X0 = 0, b = 0.1 and η = 5, randomly sampling a datapoint from F at every step.
Details of the setup can be found in Appendix G.2.

46


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction introduce the class of problems we consider
in this work. We do not exaggerate our claims and state precisely the limitations of our
work, namely ’high-dimensional linear composite setting’ with normally distributed data in
Section 1.1.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clearly state all the assumptions on the model and algorithm in Section 1.1.
When the theory switches to the least squares problem, we indicate clearly in the theorem
statements. For example, see AdaGrad (Section 4) and Line Search (Section 3). We are very
careful in the statements of our theorems/propositions/lemmas/corollaries to include all the
assumptions needed to prove the result.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: The paper contains 2 sections detailing the exact set-up of the problem
class (Section 1.1) as well as the algorithmic set-up (Section 1.2). All the proofs for the
main results are provided in the appendix and we are careful to include all the necessary
assumptions to prove these results.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Either in the captions or in Section H, we state all the parameters, covariances
of the data used, initialization, etc that are needed to reproduce the results. We also intend
to release the code and make it available via github. We also remark that the numerical
simulations are relatively simple to reproduce as they are generated using synthetic data and
the algorithms are already in use.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: In the captions of the figures and/or in Section H, we have provided all the
details to reproduce the experiments. The data is synthetic. We also intend to release the
code.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

47


---Page Break---
Justification: Either in the captions of the figures or in Section H, we state all the parameters,
covariance of the data, initialization, etc that are needed to reproduce the results. We also
intend to release the code and make it available via github. We remark that the numerical
simulations are relatively simple to reproduce as they are generated using synthetic data and
the algorithms are already in use.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: In the figures (especially Figure 1), we provided statistical error bars (shaded
region). Part of our theory is that the training dynamics of SGD with adaptive learning rate
in high-dimensions concentrate. We then predict these training dynamics exactly. In this
sense, we have used statistical tests to illustrate our theory.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: The computing resources are minimal in our case as the models are just least
squares and logistic regression with synthetic data. We explicitly state all the relevant details
for another person to run the experiments. We remark that due to the simplicity of our
simulations, no experiment required extensive compute capabilities.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We have reviewed the code of ethics. We have made our utmost attempt to
adhere to the guidelines provided by NeurIPS. We do not use any human subjects nor any
datasets (privacy is not a concern). We did our best to cite all the relevent related work.
Given that our work is in the foundational research and, in addition, the theory side, it is
difficult to mitigate all the risks as the downstream effects of theory are long. The model is
completely synthetic using the standard SGD algorithm; thus we don’t, to the best of our
knowledge, anticipate any risks.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: We do not anticipate any negative societal impact of the work. The work
is a purely theoretical result on foundational research and it is not tied to any particular
application. The work uses only synthetic data, standard algorithms (e.g., SGD, AdaGrad-
Norm) that have already been used in ML/AI, and models such as least squares and logistics
that are textbook. We have included a paragraph at the beginning of our appendix justifying
this answer to the broader impacts of our work.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risks. The work is purely theoretical and uses only
synthetic data generated from a normal distribution. The models employed are standard
statistical model (e.g., least squares and logistic regression) which are textbook problems.
12. Licenses for existing assets

48


---Page Break---
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We acknowledge via citations that the algorithms we study have been intro-
duced before by others. We are not using any datasets or other assets beyond numpy and
thus do not need to name any license or cite any dataset.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not intend to release any new assets. Neither the model nor the
algorithms are new. We are simply analyzing known algorithms and models in this paper.
We do intend to release code for reproducibility. In particular, we want readers to be able to
generate the deterministic ODEs.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
The work is purely theoretical on a simple model.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The work does not involve crowdsourcing nor research with human subjects.
The data used in this work is generated synthetically.

49


---Page Break---
