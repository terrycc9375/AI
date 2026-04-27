Near-Optimal Streaming Heavy-Tailed Statistical
Estimation with Clipped SGD

Aniket Das∗
Stanford University
aniketd@cs.stanford.edu

Dheeraj Nagaraj
Google DeepMind
dheerajnagaraj@google.com

Soumyabrata Pal∗
Adobe Research
soumyabratap@adobe.com

Arun Sai Suggala
Google DeepMind
arunss@google.com

Prateek Varshney∗
Stanford University
vprateek@stanford.edu

Abstract

We consider the problem of high-dimensional heavy-tailed statistical estimation in
the streaming setting, which is much harder than the traditional batch setting due
to memory constraints. We cast this problem as stochastic convex optimization
with heavy tailed stochastic gradients, and prove that the widely used Clipped-
SGD algorithm attains near-optimal sub-Gaussian statistical rates whenever the
second moment of the stochastic gradient noise is finite. More precisely, with T
samples, we show that Clipped-SGD, for smooth and strongly convex objectives,

achieves an error of
q

Tr(Σ)+√

Tr(Σ)∥Σ∥2 ln(ln(T )/δ)

T
with probability 1 −δ, where
Σ is the covariance of the clipped gradient. Note that the fluctuations (depending
on 1/δ) are of lower order than the term Tr(Σ). This improves upon the current

best rate of
q

Tr(Σ) ln(1/δ)

T
for Clipped-SGD, known only for smooth and strongly
convex objectives. Our results also extend to smooth convex and lipschitz convex
objectives. Key to our result is a novel iterative refinement strategy for martingale
concentration, improving upon the PAC-Bayes approach of Catoni and Giulini [8].

1
Introduction

A fundamental problem in machine learning and statistics is the estimation of an unknown parameter
of a probability distribution, given samples from that distribution. This can be expressed as the
minimization of the expected loss: minx F(x) := Eξ∼P [f(x; ξ)], where x represents the parameter
to be estimated, P is the underlying probability distribution which can only be accessed through
samples, and f(x; ξ) is a function which quantifies the loss incurred at a point ξ by parameter x.
In this paper, we focus on the setting where P is a heavy-tailed distribution for which the extreme
values are more likely than in distributions like the Gaussian, f(·; ·) is convex and the learner only
has access to O(d) memory.

The heavy-tailed statistical estimation problem has received increased attention of late because of
the prevalence of heavy-tailed distributions in many statistical applications dealing with real world
data [19, 49, 57, 23]. The presence of such heavy-tailed distributions can significantly degrade the per-
formance of statistical estimation and testing procedures designed under Gaussian (or sub-Gaussian)
tail assumptions [30, 24, 53, 24]. This has spurred recent research efforts towards developing estima-
tors specifically tailored for heavy-tailed settings (e.g., [10, 44, 16, 36]; see Section 1.2 for a more
detailed literature review). Despite substantial progress on this problem in recent years, much of the

∗Work done while at Google

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
existing work has concentrated on batch learning, where the entire dataset is available upfront, and the
learner can revisit data points multiple times, without memory constraints. However, the streaming
setting, where data arrives sequentially and must be processed with limited memory, is increasingly
pertinent in the era of large-scale models. Consequently, in this work, we focus on understanding
estimators for statistical estimation under heavy-tailed distributions, in the streaming setting.

A popular approach to study heavy-tailed streaming statistical estimation casts it as a stochastic
convex optimization (SCO) problem with heavy-tailed gradients [17, 44, 52, 48] - with Clipped-SGD
as the favored solution due to its simplicity [42]. Indeed, clipping has become a standard component
in the training of modern deep neural networks and thus, the properties of Clipped-SGD have been
studied widely in the literature [1, 56, 38, 48, 52] in various contexts. Specifically, several works
have shown that Clipped-SGD has sub-Exponential or sub-Gaussian tails despite the presence of
heavy tailed noise in the gradient [45, 21, 52, 49]. Despite this progress, the best known rates for
Clipped-SGD with smooth and strongly convex losses, under a bounded 2nd moment assumption on

gradient distribution, are of the order
q

Tr(Σ) ln(1/δ)

T
, where δ is the failure probability [52]. Note that

this is still far from the optimal sub-Gaussian rates of
q

Tr(Σ)+∥Σ∥ln(1/δ)

T
. In this work, we bridge
this gap with a sharper analysis of Clipped-SGD for SCO problems, achieving nearly sub-Gaussian
rates (see Section 1.1). Our approach leverages a novel technique obtained by bootstrapping the
Donsker-Varadhan Variational Principle to Freedman’s inequality, yielding tighter concentration
inequalities for vector martingales compared to those in [8]. This enables us to derive more refined
rates for a variety of settings than a direct application of Freedman’s inequality as in [52].

1.1
Sub-Gaussian Error Guarantees for Statistical Estimation

Mean Estimation
We motivate our style of results with the case of mean estimation. The Central
Limit Theorem (CLT) posits that that the empirical mean of T independent and identically distributed
(i.i.d) random variables with a finite covariance, behaves roughly like the empirical mean of Gaussian
random variables with the same covariance, as T →∞. That is, the empirical mean ˆµ, the true mean

µ and the covariance Σ are such that limT →∞P
√

T∥ˆµ −µ∥>
q

Tr(Σ) + ∥Σ∥2 log( 1

δ )

≤δ.
However, these asymptotic rates need not hold with a practical number of samples. Therefore, recent
works on heavy-tailed high dimensional mean estimation consider algorithms and non-asymptotic
guarantees which move beyond the empirical mean (see [36, 10, 9, 27, 28, 15]). Estimators such as
the clipped mean estimator [8, 55], trimmed mean estimator [45], and the geometric median-of-means

estimator [39, 29] achieve an error of at-most
q

Tr(Σ) log( 1

δ )/T with probability 1 −δ with a finite
covariance assumption. Recent ground breaking works [37, 28, 8, 36] further improve upon these re-

sults to construct estimators which can achieve the CLT convergence rates of C
q

Tr(Σ)+∥Σ∥2 log( 1

δ )/T
for every T and δ. Some of these estimators work under just the assumption that the second moment
is bounded [37, 28, 9] and some even provide a nearly linear time algorithm [15].

General Statistical Estimation In this work, we are interested in the general statistical estimation
problem. Among the various approaches, framing this problem as SCO with heavy-tailed gradients
has gained traction recently (see [52] and references there in). While one obvious candidate is to use
SGD with state-of-the-art optimal mean estimators for robust gradient estimation, such methods can
face significant challenges. First, most optimal mean estimators aren’t designed for the streaming
data setting with batch-size being 1. Second, these estimators can be complex, frequently relying on
semi-definite programming or other demanding techniques. Third, and perhaps most importantly,
they don’t typically provide guarantees on the bias of their estimates. This lack of bias control is
problematic because SGD-style algorithms, even when equipped with accurate gradient estimates,
can perform poorly if those estimates are systematically biased (See [3, Theorem 4], where bias
does not cancel across iterations). Given these challenges, the clipped mean estimator of [8] has
emerged as a popular choice for gradient estimation in SCO, due mainly to is simplicity. Several
recent works analyze the performance of SGD with clipped mean estimator for the gradients (i.e,
Clipped SGD). However, as previously mentioned, the best known analysis for clipped SGD achieves
a sub-optimal rate of
p

Tr(Σ) ln(1/δ)/T, under bounded 2nd moment assumption. In this work, we
improve upon these rates and show that with T samples, clipped-SGD obtains a sharper rate of

Tr(Σ)+√

Tr(Σ)∥Σ∥ln( ln(T )

δ
)
T
with probability 1 −δ, which is closer to the truly sub-Gaussian rates.

2


---Page Break---
Table 1: Sample complexity bounds (for converging to an ϵ approximate solution) of various
algorithms for SCO under heavy tailed stochastic gradients. Results are instantiated for smooth and
strongly convex losses, and for the case where the gradient noise has bounded covariance equal to
the Identity matrix. D1 is the distance of the initial iterate from the optimal solution. For readability,
we ignore the dependence of rates on the condition number. Observe all prior works have d log δ−1
dependence in the sample complexity.

Method
Sample Complexity
Batchsize
Domain

Clipped SGD [21]
d
ϵ

log D2
1
ϵ

log δ−1 + log log D2
1
ϵ

O

d
ϵ log

D2
1
ϵ

log

1
δ log D2
1
ϵ

Unbounded

R-Clipped SGD [21]

d
ϵ + log D2
1
ϵ

log δ−1 + log log D2
1
ϵ

O

d
ϵ log

1
δ log D2
1
ϵ

Unbounded

R-Clipped SSTM [21]

d
ϵ + log D2
1
ϵ

log δ−1 + log log D2
1
ϵ

O

d
ϵ log

1
δ log D2
1
ϵ

Unbounded

RobustGD [45]
O

dΦ

ϵ log Φ

δ

with Φ = log D2
1
ϵ
O

d
ϵ log Φ

δ

Unbounded

proxBoost [14]

d
ϵ + log D2
1
ϵ

log δ−1
O

d
ϵ log 1

δ

Unbounded

restarted-RSMD [40]

d
ϵ + log D2
1
ϵ

log δ−1 + log log D2
1
ϵ

O

d
ϵ

log δ−1 + log log D2
1
ϵ

Bounded

Clipped SGD [52]

d
ϵ + D1
√ϵ

log δ−1
1
Unbounded

Clipped SGD (Ours)
d+
√

d log δ−1

ϵ
+ D1 log2(δ−1 log T )
√ϵ
1
Unbounded

1.2
Related Work

Clipped SGD Clipped SGD and it’s variants have been studied under a variety of settings including
convex, strongly-convex, non-convex losses, with various assumptions on the moments of stochastic
gradients. The estimators of [21, 45, 14, 40] work under the assumption of bounded 2nd moments,
but require O(1/ϵ) batch size, to converge to an ϵ-approximate solution. Consequently, they are
not suitable for streaming setting. The recent work of [52], which is closest to our work, addresses
this issue by analysing Clipped-SGD for batch size 1 for smooth, strongly convex losses. But they
achieve a sub-optimal rate of
p

Tr(Σ) ln(1/δ)/T. These rates are improved in our work (see Table 1
for a detailed comparison). Additionally, our work provides convergence rates for convex objectives
that are not strongly convex. Recent works [48, 46, 41, 34, 13] have studied Clipped-SGD with the
assumption that the stochastic gradient has a finite p-th moment for some p ∈(1, 2]. They derive
fine-grained near optimal results in terms of dependence of T and p (but their dependence on log δ−1
is sub-optimal). In contrast, our work specifically the case considers p = 2 with a focus on improving
the sub-Gaussian dependence in the high probability bounds in these works from Tr(Σ) log(1/δ) and
approaching the truly sub-Gaussian rates for estimation 1.1.

Heavy-tailed Estimation Heavy-tailed estimation has a rich history in statistics and we only review
some of the recent advances. Several recent works have studied the problem of heavy-tailed mean
estimation, and have derived estimators that achieve sub-Gaussian rates under the bounded 2nd
moment assumption [36, 10, 9, 27, 28, 15, 45]. Among these, the works of [15, 32] are particularly
relevant to our work. The algorithm of [15] runs in linear time while requiring O(d log δ−1) memory.
But it is not immediately clear how to use their estimator in the framework of SGD. [32] study the
trimmed mean estimator (an estimator that is closely related to clipped mean estimator, where outliers
are removed instead of being clipped) and show that when T = ω(log3 δ−1), d = ω(log2(δ−1)), the
estimator achieves the optimal rates. We not that our analysis of clipped SGD, when instantiated
for mean estimation, leads to similar rates. But unlike [32] which is primarily focused on mean
estimation, we focus on the more general SCO problem.

Heavy-tailed linear regression has been widely studied, with classical estimators based on Huber
regression [30, 50, 33] known to provide optimal rates when the response variables are heavy-tailed,
but the covariates are light-tailed. Recently, there has been a surge of interest in developing estimators
when both covariates and response variables are heavy tailed [5, 44, 17, 43]. However, most of these
works are in the batch setting. Another line of work has considered streaming algorithms in the
Huber-contamination model, which is a much harder contamination model than heavy-tails [18].
However, these algorithms when adapted to heavy-tailed setting, do not provide optimal rates.

3


---Page Break---
1.3
Contributions

Iteratively Refined Martingale Concentration via PAC Bayes Our key technical result obtains
fine-grained concentration guarantees for vector-valued martingales by using the Donsker-Varadhan
Variational Principle to iteratively refine baseline concentration inequalities. This allows us to sharpen
the PAC Bayes bounds of Catoni and Giulini [8] (and its martingale based extensions like [11]), which
were used to analyze the clipped mean estimator. We believe these iterative refinement arguments
could be of independent interest for developing fine-grained concentration bounds.

Sharp Analysis of Clipped SGD Leveraging these fine-grained concentration results, We perform a
fine-grained analysis of clipped SGD for heavy-tailed SCO problem obtain nearly subgaussian perfor-
mance guarantees in the streaming setting with a batchsize of 1 and O(d) space complexity. In particu-
lar, we demonstrate that the sub-optimality gap after T steps scales as Tr(Σ)+
p

∥Σ∥2Tr(Σ) log(1/δ),
improving upon the best known scaling of Tr(Σ) log(1/δ) obtained by prior works [52] only for
smooth strongly convex problems. To the best of our knowledge, we derive the first such guarantees
for smooth convex and lipschitz convex problems in the streaming setting.

Streaming Heavy Tailed Statistical Estimation We use the above results to develop streaming
estimators for various heavy-tailed statistical estimation problems including heavy-tailed mean
estimation as well as linear, logistic and Least Absolute Deviation (LAD) regression with heavy
tailed covariates, all of which exhibit nearly subgaussian performance. Our mean estimation results
improve upon the previous best known guarantees for trimmed mean based estimators [8, 52, 32]
(either in performance or in generality) For heavy-tailed linear regression under the assumption
of bounded 4th moments for the covariates and bounded 2nd moments for the response, our rates
significantly improve upon that of the previous best known streaming estimator [52]. To the best of
our knowledge, we develop the first known streaming estimators for heavy-tailed logistic regression
and LAD regression which attain nearly subgaussian rates

2
Notation and Organization

We work with Euclidean spaces Rd equipped with the standard inner product ⟨·, ·⟩and the induced
ℓ2 norm ∥· ∥. For any matrix A ∈Rm×n, we use ∥A∥2 to denote its Euclidean operator norm
∥A∥= supx̸=0 ∥Ax∥/∥x∥. For A ∈Rd×d, we denote its trace as Tr(A). For any random vector x, we
denote its covariance matrix as Cov[x]. We use ≲, ≳and ≍to denote ≤, ≥and = respectively, upto
universal multiplicative constants. We use ∇f(x) to denote the gradient of a differentiable function
For any convex function f, we use ∂f(x) to denote an arbitrary subgradient of f at x.

3
Background and Problem Formulation

Our work studies the Stochastic Convex Optimization (SCO) problem, described as follows: Let C
denote a closed convex subset of Rd and let F : C →R be a convex function. We aim to solve:

min
x∈C F(x),
(SCO)

assuming access to a convex projection oracle ΠC and a stochastic gradient oracle, which we define
as follows: Let P denote a probability measure supported on an arbitrary domain Ξ from which we can
draw samples. A stochastic gradient oracle for F is a function g : C × C, which, given a point x ∈C
and a sample ξ ∼P returns an unbiased estimate g(x; ξ) of ∇F(x) i.e., Eξ∼P [g(x; ξ)] = ∇F(x).
If F is nondifferentiable, Eξ∼P [g(x; ξ)] = ∂F(x). Note that we do not assume direct access to
∇F(x), which may be expensive or intractable to compute. Our objective is to (approximately) solve
SCO subject to a constraint on the number of samples we can draw from P.

This is an alternative formulation of the statistical estimation problem by recognizing P as the data
distribution, C as the parameter space and defining the population risk F(x) := Eξ∼P [f(x; ξ)], where
f denotes the sample-level loss function. The associated stochastic gradient oracle is g(x; ξ) :=
∇f(x; ξ), ξ ∼P, which is usually easy to compute. As we shall discuss in Section 5, several
statistical estimation problems such as mean estimation, linear regression, logistic regression and
least absolute deviation regression naturally fit into the SCO framework.

4


---Page Break---
We use n(x; ξ) = g(x; ξ) −∇F(x) to denote the stochastic gradient noise and assume it has finite
second moment, i.e., Σ(x) = Eξ∼P [n(x; ξ)n(x; ξ)T ] exists for every x ∈C. Our results make use
of either of the following assumptions on Σ(x).

Assumption 1 (Bounded Second Moment). The exists a positive semidefinite matrix Σ such that:

Σ(x) ⪯Σ
∀x ∈C
(Bdd. 2nd Moment)

Similar assumption has been made by several prior works [21, 40, 14, 45]. We also consider the
following generalized assumption, which is as a refinement of the one made in Tsai et al. [52].

Assumption 2 (Second Moment with Quadratic Growth). There exist constants α, β ≥0 and
1 ≤deffd such that the following holds for every x ∈C

∥Σ(x)∥2 ≤α∥x −x∗∥2 + β;
Tr(Σ(x)) ≤deff
 
α∥x −x∗∥2 + β

(QG 2nd Moment)

where x∗denotes any arbitrary minimizer of F.

Since we consider streaming statistical estimators that are robust to heavy tailed data, we only assume
the existence of the second moment of the stochastic gradient noise and allow its higher moments to be
infinite. That is, our results hold even when Eξ∼P [| ⟨n(x; ξ), v⟩|2+ϵ] = ∞for every ϵ > 0, v ∈Rd

Our work analyzes SCO under either of the following structural assumptions assumptions on F

Assumption 3 (Convexity). F : Rd →R is a convex function if the following holds for any t ∈[0, 1]

F(tx + (1 −t)y) ≤tF(x) + (1 −t)F(y)
∀x, y ∈Rd
(Convexity)

Assumption 4 (µ-Strong Convexity). F : Rd →R is a µ-strongly convex function for µ ≥0 if the
following holds for every t ∈[0, 1]

F(tx + (1 −t)y) ≤tF(x) + (1 −t)F(y) −t(1 −t) · µ

2 ∥x −y∥2
∀x, y ∈Rd (µ-Strong Convexity)

In addition, we also consider either of the two regularity assumptions on F

Assumption 5 (L-smoothness). F : Rd →R is L-smooth for some L ≥0 if F is continuously
differentiable and satisfies the following:

∥∇F(x) −∇F(y)∥≤L∥x −y∥
∀x, y ∈Rd
(L-smoothness)

Assumption 6 (G-Lipschitzness). F : Rd →R is G-Lipschitz for some G ≥0, i.e., F is continuous
and satisfies the following:

∥F(x) −F(y)∥≤G∥x −y∥
∀x, y ∈Rd
(G-Lipschitzness)

4
Results

Under the Bdd. 2nd Moment and QG 2nd Moment assumptions, streaming algorithms for SCO such as
Stochastic Gradient Descent (SGD) typically convergence bounds guarantees that hold in expectation
[56, 26, 22]. However, high probability guarantees require strong assumptions on the tail behavior
of the stochastic gradients (e.g. boundedness or subgaussianity) [25, 47, 31]. Our work analyzes
SCO under heavy tailed stochastic gradients, which typically exhibit large fluctuations from their
expected value due to its higher order moments being potentially infinite. Clipped SGD mitigates the
large fluctuations typically observed in the heavy tailed stochastic gradient g(x; ξ) by thresholding
its norm as follows. The full algorithm is described in Algorithm 1.

clipΓ(g(x; ξ)) :=
g(x; ξ)
∥g(x; ξ)∥· min{Γ, ∥g(x; ξ)∥}

We now present our performance guarantees for clipped SGD for streaming heavy tailed SCO,
wherein Algorithm 1 is subject to an O(d) memory constraint and can access only one stochastic
gradient sample per iteration. For the remainder, of this section, we use x∗∈C to denote an arbitrary
minimizer of F, which is assumed to always exist, and guaranteed to be unique if F satisfies µ-Strong
Convexity. We use x1 to denote the initialization of Algorithm 1 and let D1 = ∥x −x∗∥.

5


---Page Break---
Algorithm 1 Clipped Stochastic Gradient Descent
Input: Initialization x1, Horizon T, Step Sizes (ηt)t∈[T ], Clipping Level Γ

1: for t ∈[T] do
2:
gt ←g(xt; ξt),
ξt ∼P
3:
xt+1 ←ΠC(xt −ηt · clipΓ(gt))
4: end for
5: Last Iterate : Output xT +1
6: Average Iterate : Output ˆxT = 1

T
PT
t=1 xt

4.1
Smooth Strongly Convex Objectives

Theorems 1 and 2, proved in, Appendix B and C respectively, derive high probability convergence
bounds for smooth and strongly convex objectives with second moment assumption.
Theorem 1 (Smooth Strongly Convex Objectives). Let the L-smoothness, µ-Strong Convexity
and Bdd.
2nd Moment assumptions be satisfied.
Then, for any δ ∈(0, 1/2), the last iterate
of Algorithm 1 run for T ≳ln(ln(d)) iterations with stepsize ηt =
4
µ(t+γ) and clipping level

Γ =
µ
ln(ln(T )/δ)
q

(γ + 1)2D2
1 + (T +γ)

µ2
(Tr(Σ) +
p

Tr(Σ)∥Σ∥2 ln(ln(T )/δ))satisfies the following with
probability at least 1 −δ

∥xT +1 −x∗∥≲γD1

T + γ + 1

µ

s

Tr(Σ) +
p

Tr(Σ)∥Σ∥2 ln(ln(T )/δ)

T + γ
(1)

where γ ≍max{ ∥Σ∥2κ2 ln(ln(T )/δ)2

Tr(Σ)
, κ
3/2 ln(ln(T )/δ), κ ln(ln(T )/δ)2}

We use Theorem 1 to derive sharp rates for streaming heavy tailed mean estimation in Section 5.1 and
the following result to derive sharp rates for streaming heavy tailed linear regression in section 5.2
Theorem 2 (Smooth Strongly Convex Objectives with Quadratic Growth Noise Model). Let Assump-
tions µ-Strong Convexity, L-smoothness and QG 2nd Moment be satisfied and let κ = L/µ. For any
δ ∈(0, 1/2), the last iterate of Algorithm 1 run for T ≳ln(ln(d)) iterations with step-size ηt =
4
µ(t+γ)

and clipping level Γ =
µ
ln(ln(T )/δ)
q

(γ + 1)2D2
1 + β

µ2 · (T + γ)(deff + √deff ln(ln(T )/δ)) satisfies the
following with probability at least 1 −δ

∥xT +1 −x∗∥≲γD1

T + γ + 1

µ

s

β(deff + √deff ln(ln(T )/δ))

T + γ
(2)

where γ ≍max{αdeff

µ2 , α√deff

µ2
ln(ln(T )/δ), κ√α

µ
ln(ln(T )/δ),
√καdeff

µ
ln(ln(T )/δ),

κ
2/3α
1/3d
1/3
eff
µ2/3
ln(ln(T )/δ), κ
3/2 ln(ln(T )/δ), κ ln(ln(T )/δ)2, κ2

deff ln(ln(T )/δ)}

Comparison to Prior Works To the best of our knowledge, the result closest to Theorem 2 is [52,

Theorem 1] which analyzes streaming strongly convex SCO and obtains a ζD1

T +ζ + 1

µ

q

βdeff ln(1/δ)

T +ζ
rate

for ζ ≍αdeff log(1/δ)

µ2
. We note that Theorem 2 obtains a significantly better confidence bound which
is closer to the optimal subgaussian rate compared [52, Theorem 1].

Extra log log T term: Our bounds for the statistical error is of the form 1

µ

q

β(deff+√deff ln(ln(T )/δ))

T +γ
which has an extra log log T factor in the lower order term. This is still sharper than prior works with

bounds of the form 1

µ

q

βdeff ln(1/δ)

T +γ
as long as log log T ≪√deff log( 1

δ ).

4.2
Beyond Strongly Convex Objectives

Moving beyond strong convexity, we present Theorems 3 for smooth convex functions and 4 for
Lipschitz convex function, proved in Appendix D and E respectively. To the best of our knowledge,

6


---Page Break---
these are the first results for streaming heavy-tailed convex SCO that exhibits near-subgaussian
concentration without strong convexity.

Theorem 3 (Smooth Convex Objectives). Let Convexity, L-smoothness and Bdd. 2nd Moment be
satisfied. Then, for any δ ∈(0, 1/2) and T ≥ln(ln(d)), there exists an η ∈(0, 1/2L] such that
the average iterate of Algorithm 1 run for T iterations with step-size ηt = η and clipping level

Γ =

r

T√

∥Σ∥2(√

Tr(Σ)+LD1)
ln(ln(T )/δ)
satisfies the following with probability at least 1 −δ:

F(ˆxT ) −F(x∗) ≲LD2
1
T
+ D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + LD1

ln(ln(T )/δ)

T
+ oT (L, D1, Σ)

where oT (L, D1, Σ) represents terms that are of lower order in T (explicated in Appendix D)
Theorem 4 (Lipschitz Convex Objectives). Let Assumptions Convexity, G-Lipschitzness and Bdd.
2nd Moment be satisfied. Then, for any δ ∈(0, 1/2) and T ≥ln(ln(d)), there exists an η ∈(0, G/
√

T]
such that the average iterate of Algorithm 1 run for T iterations with step-size ηt = η and clipping

level Γ =

r

T√

∥Σ∥2(√

Tr(Σ)+G)
ln(ln(T )/δ)
satisfies the following with probability at least 1 −δ

F(ˆxT ) −F(x∗) ≲D1G
√

T
+ D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + G

ln(ln(T )/δ)

T
+ oT (G, D1, Σ)

where oT (G, D1, Σ) represents terms that are lower order in T (explicated in Appendix E)

Remark: We use Theorem 3 to design the first known streaming estimator for logistic regression with
heavy-tailed covariates in Section 5.3 and Theorem 4 to design the first known streaming estimator
for LAD regression with heavy-tailed covariates in Section 5.4.

Remark:
In Theorems 3 and 4, the leading order term in the error is of the form:

D1

r

Tr(Σ)+√

∥Σ∥2
√

Tr(Σ)+ζ

ln(ln(T )/δ)

T
, where ζ ∈{G, LD1}. Assuming G, D1,
p

Tr(Σ) ≍
√

d,
we note that the term dependent on the confidence level log(1/δ) is lower order compared to Tr(Σ).
To the best of our knowledge, this is the first work which establishes strong confidence bounds in
the setting of SCO without strong convexity. Interestingly, our results also improve the best known
rates for sub-Gaussian gradient noise. To be precise, [35, Theorem 3.1] shows a weaker bound of
q

D2
1(G2+Tr(Σ) log( 1

δ ))/T in the setting of Theorem 4, but when the noise is sub-Gaussian.

5
Applications to Streaming Heavy Tailed Statistical Estimation

5.1
Streaming Heavy-Tailed Mean Estimation

Consider streaming heavy tailed mean estimation with clipped SGD with access to N i.i.d samples
from the distribution P. Let Ξ = C, Eξ∼P [ξ] = m ∈C. We further assume Cov[ξ] ⪯Σ and allow
the higher moments to be infinite. As described in Appendix G.1, this is an SCO problem with the
sample loss f(x; ξ) = 1

2∥x −ξ∥2. The population loss and the stochastic gradient are given by:

F(x) = 1

2∥x −m∥2 + Tr(Covξ∼P [ξ]);
g(x; ξ) = x −ξ

The following result, proved in Appendix G.1 via an application of Theorem 1, shows that the last
iterate of clipped SGD attains near-subgaussian rates for the heavy tailed mean estimation problem
Corollary 1 (Heavy Tailed Mean Estimation). Under the stochastic gradient oracle described above,
implemented using N ≳ln(ln(d)) i.i.d samples ξ1, . . . , ξN ∼P, the last iterate of Algorithm 1 when
run under the parameter settings of Theorem 1 satisfies the following with probability at least 1 −δ

∥xN+1 −m∥≲γ∥x1 −m∥

N + γ
+

s

Tr(Σ) +
p

∥Σ∥2Tr(Σ) ln(ln(N)/δ)

N + γ

where γ ≍ln(ln(N)/δ)2

7


---Page Break---
Comparison to Prior Works The clipped mean estimator of [8] and the clipped-SGD based estimator

in [52] come with a guarantee of the form ∥ˆm −m∥≲
q

Tr(Σ) log( 1

δ )/N with probability 1 −δ.
Our result in Corollary 1 obtains a sharper rate of convergence. In a recent work, Lee and Valiant
[32] showed that the trimmed mean estimator achieves the optimal rate of
p

Tr(Σ)/N when N =
ω(log3 δ−1), d = ω(log2(δ−1)). Our result matches this optimal rate in those settings, but is
considerably more general, as it holds for any N, d.

5.2
Streaming Heavy Tailed Linear Regression

In the current and subsequent sections, we use θ ∈C to denote the parameter of F. Let Ξ = Rd × R.
Given a target parameter θ∗∈C, P defines the following linear model:

x ∼Q, E[x] = 0, E[xxT ] = Σ ≻0;
y = ⟨x, θ∗⟩+ ϵ, E[ϵ|x] = 0, E[ϵ2|x] ≤σ2

In addition, we make the following bounded 4th moment asumption on the covariates x

E[⟨x, v⟩4] ≤C4(E[⟨x, v⟩2])2
∀v ∈Rd

for some numerical constant C4 ≥1. Note that we allow both the covariate x and the target y to be
heavy tailed, assuming only finite moments of upto order 4 for x and order 2 for y. The assumption
E[x] = 0 is only made for ease of presentation and our arguments easily adapt to E[x] ̸= 0. Our
task is to estimate θ∗in a streaming fashion with access to N i.i.d samples from P. As described in
Appendix G.2, we reframe this problem as SCO under the sample loss f(θ; x, y) = 1

2(⟨x, θ⟩−y)2.
The associated population loss F(θ) and the stochastic gradient oracle g(θ; x, y) are given by:

F(θ) = 1

2(θ −θ∗)T Σ(θ −θ∗);
g(θ; x, y) = (⟨x, θ⟩−y)x

Corollary 2 (Heavy Tailed Linear Regression). Under the stochastic gradient oracle described
above, implemented using N ≳ln(ln(d)) i.i.d samples from P, the last iterate of Algorithm 1 when
run under the parameter settings of Theorem 2 satisfies the following with probability at least 1 −δ:

∥θN+1 −θ∗∥≲γ∥θ1 −θ∗∥

N + γ
+
σ
λmin (Σ)

s

Tr(Σ) +
p

∥Σ∥2Tr(Σ) ln(ln(N)/δ)

N + γ

where γ ≍max
n
C4κ2Tr(Σ)

∥Σ∥2
, C4κ2q

Tr(Σ)

∥Σ∥2 ln(ln(N)/δ), κ ln(ln(N)/δ)2o
and κ =
∥Σ∥2
λmin(Σ)

To the best of our knowledge, [52, Corollary 4] is the only other streaming estimator for this problem
with subgaussian-style concentration. Our result above significantly improves upon their rates of

∥θ1−θ∗∥

N+ζ
+
σ
λmin(Σ)

q

∥Σ∥2d ln(1/δ)

N+ζ
with ζ = C4dκ2 ln(1/δ). Furthermore, our result is much closer to
the optimal subgaussian rate and gracefully adapts to the stable rank or effective dimension [32], i.e.,
deff = Tr(Σ)/∥Σ∥, therefore implying significant speedups over [52] in settings where deff ≪d.

5.3
Streaming Heavy Tailed Logistic Regression

Let Ξ = Rd × {0, 1} and given a target parameter θ∗∈C, P denote the following linear-logistic
model:

x ∼Q, E[x] = 0, E[xxT ] ⪯Σ;
y ∼Bernoulli(ϕ(⟨θ∗, x⟩))

where ϕ(t) = (1 + e−t)−1. The covariates x are heavy tailed, with only bounded second moments.
The negative log likelihood of y|x is given by f(θ; x, y) = ln(1 + exp(⟨x, θ⟩)) −y ⟨x, θ⟩. The
objective of the logistic regression problem is to estimate θ∗by minimizing the population-level
negative log likelihood:

F(θ) = Ex,y∼P [ln(1 + exp(⟨x, θ⟩)) −y ⟨x, θ⟩]

which is minimized at θ∗. Here, the stochastic gradient oracle is g(θ; x, y) = ϕ(⟨x, θ⟩)x −yx. The
following result applies Theorem 3 to show that the output of clipped SGD attains near-subgaussian
rates for heavy tailed logistic regression. We refer to Appendix G.3 for the proof.

8


---Page Break---
Corollary 3 (Heavy Tailed Logistic Regression). Under the stochastic subgradient oracle described
above, realized using N ≳ln(ln(d)) i.i.d samples from P, the average iterate of Algorithm 1, when
run under the parameter settings of Theorem 4 satisfies the following with probability at least 1 −δ:

F(ˆθN) −F(θ∗) ≲D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + ∥Σ∥2D1

ln(ln(N)/δ)

N
+ oN(Σ, D1)

where oN(Σ, D1) represents terms that are lower order in N (explicated in Appendix G.3

Note that the standard analysis of SGD, with the assumption that ∥x∥≤R almost surely leads to a

bound of the form [4, Proposition 5]: F(ˆθN) −F(θ∗) ≲
RD1√

log( 1

δ )
√

N

5.4
Streaming Heavy Tailed LAD Regression

Let Ξ = Rd × R. Given a target parameter θ∗∈C, P defines the following linear model:

x ∼Q, E[x] = 0, E[xxT ] ⪯Σ;
y = ⟨x, θ∗⟩+ ϵ, Median(ϵ|x) = 0

We allow both the covariate x and target y to be heavy tailed, assuming only bounded second moments
for x. We do not assume any moment bounds on ϵ|x. The assumption E[x] = 0 is made for the sake
of clarity and can be straightforwardly relaxed. The Least Absolute Deviation (LAD) Regression
problem involves estimating θ by solving SCO with the sample loss f(θ; x, y) = | ⟨x, θ⟩−y|. The
stochastic subgradient oracle and population risk is given by:

g(θ; x, y) = sgn(⟨θ, x⟩−y)x,
F(θ) = E [| ⟨θ −θ∗, x⟩−ϵ|]

where sgn(t) =
t
∥t∥for t ̸= 0 and sgn(0) = 0. The following result, whose full statement and proof
is presented in Appendix G.4, applies Theorem 4 to show that the average iterate of clipped SGD
attains near-subgaussian rates for heavy tailed LAD regression. To the best of our knowledge, this is
the first known streaming estimator for this problem.
Corollary 4 (Heavy Tailed LAD Regression). Under the stochastic subgradient oracle described
above, realized using N ≳ln(ln(d)) i.i.d samples from P, the average iterate of Algorithm 1, when
run under the parameter settings of Theorem 4 satisfies the following with probability at least 1 −δ:

F(ˆθN) −F(θ∗) ≲D1

s

Tr(Σ) +
p

∥Σ∥2Tr(Σ) ln(ln(N)/δ)

N
+ oN(Σ, D1)

where oN denotes terms that are lower order in N (explicated in Appendix G.4)

6
Improved Martingale Concentration via Iterative Refinement

Our results are based on the following concentration result for Rd valued martingales. The proof
appears in Appendix F. Suppose Mt for t = 0, . . . , T is an Rd valued martingale such that M0 = 0
almost surely, the difference sequence vt := Mt −Mt−1 is such that ∥vt∥≤Γ and E[vtv⊺
t |Ft−1] =
Σt almost surely for every t = 1, . . . , T for some Γ > 0. Assume that there exist deterministic
sequences p1, . . . , pT and q1, . . . , qT such that Tr(Σt) ≤qt and ∥Σt∥≤pt almost surely.

Theorem 5. Let ¯q := 1

T
PT
t=1 qt and ¯p := 1

T
PT
t=1 pt. Then, for any δ ∈(0, 1

2):

P(sup
t≤T
∥Mt∥≥g(T, δ)
√

T) ≤δ

Where g(T, δ) = CM
h√¯q + ¯p
√

T
Γ
+
Γ
√

T log( K

δ )
i
and K = ln ln((
√¯qT

Γ
+ 1) log(d + 1)) + CM for
some universal constant CM

To prove this result, we first use Freedman’s inequality [20, 51] to obtain a coarse-grained g0 such that
P(supt ∥Mt∥> g0
√

T) ≤δ. We then iteratively refine this inequality via a PAC Bayesian [8, 11, 12]
argument to show that P(supt ∥Mt∥> gk+1
√

T | Bk) ≤δ, where Bk = {supt ∥Mt∥≤gk
√

T} and
g2
k+1 ≲Tr(Σ) + gk
p

∥Σ2∥log(1/δ). This iterative refinement strategy, proved in Theorem 14 is one
of the main technical contributions of our work, which could be of independent interest. We arrive at
Theorem 5 after K ≈log log(T log d) refinement steps.

9


---Page Break---
Remark Theorem 5 is used to control the influence of the fluctuations introduced by clipped SGD.
To this end, let vt be the centered version of clipΓ(gt), ensuring ∥vt∥≤2Γ almost surely. Suppose

Σt = Σ for some fixed Σ and let Γ =
q

∥Σ∥T/log( K

δ ). Then, with probability 1 −δ: supt≤T ∥Mt∥≲
q

TTr(Σ) + T∥Σ∥log( K

δ ). This is sharper than the supt≤T ∥Mt∥≲
q

TTr(Σ) log( d

δ ) guarantee
implied by the Matrix Freedman inequality [51, Corollary 1.6].

7
Proof Sketch

We sketch our proof technique for the case of smooth convex functions considered in 3. We consider
the SGD iterations x1, . . . , xT with clipped stochastic gradient at time t denoted by clipΓ(gt) =
∇F(xt) + vt + bt. Here, vt is the zero mean ‘variance’ such that E[vt|xt] = 0 and ∥vt∥≤2Γ
almost surely. bt is the non-zero mean ‘bias’ which arises due to clipping. Using the usual analysis
of SGD for convex functions (see for instance [31]), we consider:

∥xt+1−x∗∥2 ≤∥xt−x∗∥2−2ηt[F(xt)−F(x∗)]−2ηt⟨vt+bt, xt−x∗⟩+η2
t ∥∇F(xt)+vt+bt∥2

Considering constant step-sizes, we sum the inequalities for each t to conclude:

1
T

T
X

t=1
F(xt) −F(x∗) ≤
1
2ηT ∥x1 −x⋆∥2 + 1

T

T
X

t=1
⟨vt + bt, xt −x∗⟩

+ 3η

2T

X

t
[∥∇F(xt)∥2 + ∥vt∥2 + ∥bt∥2]
(3)

The ’random’ terms to bound compared to gradient descent here are P

t⟨vt + bt, xt −x∗⟩and
P
t ∥vt∥2 + ∥bt∥2 Lemma 13 shows that ∥xt −x∗∥≤2∥x1 −x∗∥with high probability. Under
this event, we bound P

t⟨vt, xt −x∗⟩using the standard Freeman’s inequality and ∥∇F(xt)∥2 by
using smoothness and the fact that ∇F(x⋆) = 0. The bias of the estimator ∥bt∥is bound using
arguments similar to [8] (see Lemma 4). The main improvement of our method is given by our
method of bounding 1

T
P
t ∥vt∥2. We show by an application of Theorem 5 that 1

T
P
t ∥vt∥2 ≲
Tr(Σ) +
p

Tr(Σ)∥Σ∥2 log( log T

δ
) with probability at-least 1 −δ whenever the clipping factor Γ is
appropriately chosen. Choosing the step size η appropriately gives us the result in Theorem 3.

8
Conclusion and Limitations

Our work obtained nearly subgaussian rates for heavy-tailed SCO using clipped SGD by devel-
oping a fine-grained iterative refinement strategy for martingale concentration. As corollaries, we
obtained state-of-the-art streaming estimators for various heavy tailed statistical problems. We note
Clipped-SGD is widely used to optimize neural networks with highly nonconvex landscapes, which is
currently outside the scope of our work. Nevertheless, we believe our techniques could be useful for
providing sharp high-probability guarantees for non-convex losses. Our bounds are currently of the

form
q

d+
√

d ln(ln(T )/δ)

T
, which is suboptimal compared to the tight subgaussian rate of
q

d+ln(1/δ)

T
.
Further research is required to understand if it is possible to obtain truly subgaussian rates with clipped
mean type estimators. Another notable suboptimality of our result is the ln(ln(T )/δ) dependence on
the confidence level (as opposed to the typical ln(1/δ) scaling). However, this is not a major drawback
as our results continue to significantly outperform prior works unless T ≫eexp(
√

d−1) ln(1/δ) (which
is an impractical regime). This drawback arises due to the ln(ln(T)) iterations of our iterative
refinement technique and we believe it can be removed via more sophisticated martingale concen-
tration arguments. Our work lays the foundation for several interesting avenues for future work
including the analysis of heavy tailed statistical estimation under bounded pth moment assumptions
(for p < 2) and the development of parameter free statistical estimators that do not require knowledge
of problem-dependent parameter such as ∥Σ∥, δ etc. (or their respective upper bounds). Deriving
anytime valid guarantees for clipped SGD using our techniques is also an interesting future direction.

10


---Page Break---
References

[1] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep
learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on
computer and communications security, pages 308–318, 2016.

[2] N. Agarwal, S. Chaudhuri, P. Jain, D. M. Nagaraj, and P. Netrapalli. Online target q-learning
with reverse experience replay: Efficiently finding the optimal policy for linear mdps. In
International Conference on Learning Representations, 2021.

[3] A. Ajalloeian and S. U. Stich. On the convergence of sgd with biased gradients. arXiv preprint
arXiv:2008.00051, 2020.

[4] F. Bach. Adaptivity of averaged stochastic gradient descent to local strong convexity for logistic
regression. The Journal of Machine Learning Research, 15(1):595–627, 2014.

[5] A. Bakshi and A. Prasad. Robust linear regression: Optimal rates in polynomial time. In
Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pages
102–115, 2021.

[6] S. Bubeck. Convex optimization: Algorithms and complexity. 2014. doi: 10.48550/ARXIV.
1405.4980. URL https://arxiv.org/abs/1405.4980.

[7] O. Catoni. Statistical learning theory and stochastic optimization: Ecole d’Eté de Probabilités
de Saint-Flour, XXXI-2001, volume 1851. Springer Science & Business Media, 2004.

[8] O. Catoni and I. Giulini. Dimension-free pac-bayesian bounds for the estimation of the mean of
a random vector. arXiv preprint arXiv:1802.04308, 2018.

[9] Y. Cherapanamjeri, N. Flammarion, and P. L. Bartlett. Fast mean estimation with sub-gaussian
rates. In Conference on Learning Theory, pages 786–806. PMLR, 2019.

[10] Y. Cherapanamjeri, S. B. Hopkins, T. Kathuria, P. Raghavendra, and N. Tripuraneni. Algorithms
for heavy-tailed statistics: Regression, covariance estimation, and beyond. In Proceedings of
the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pages 601–609, 2020.

[11] B. Chugg, H. Wang, and A. Ramdas. Time-uniform confidence spheres for means of random
vectors. arXiv preprint arXiv:2311.08168, 2023.

[12] B. Chugg, H. Wang, and A. Ramdas. A unified recipe for deriving (time-uniform) pac-bayes
bounds. Journal of Machine Learning Research, 24(372):1–61, 2023.

[13] A. Cutkosky and H. Mehta. High-probability bounds for non-convex stochastic optimization
with heavy tails. Advances in Neural Information Processing Systems, 34:4883–4895, 2021.

[14] D. Davis, D. Drusvyatskiy, L. Xiao, and J. Zhang. From low probability to high confidence in
stochastic convex optimization. Journal of machine learning research, 22(49), 2021.

[15] J. Depersin and G. Lecué. Robust sub-gaussian estimation of a mean vector in nearly linear
time. The Annals of Statistics, 50(1):511–536, 2022.

[16] I. Diakonikolas and D. M. Kane. Algorithmic high-dimensional robust statistics. Cambridge
university press, 2023.

[17] I. Diakonikolas, G. Kamath, D. Kane, J. Li, J. Steinhardt, and A. Stewart. Sever: A robust
meta-algorithm for stochastic optimization. In International Conference on Machine Learning,
pages 1596–1606. PMLR, 2019.

[18] I. Diakonikolas, D. M. Kane, A. Pensia, and T. Pittas. Streaming algorithms for high-dimensional
robust statistics. arXiv preprint arXiv:2204.12399, 2022.

[19] J. Fan, W. Wang, and Z. Zhu. A shrinkage principle for heavy-tailed data: High-dimensional
robust low-rank matrix recovery. Annals of statistics, 49(3):1239, 2021.

[20] D. A. Freedman. On tail probabilities for martingales. the Annals of Probability, pages 100–118,
1975.

11


---Page Break---
[21] E. Gorbunov, M. Danilova, and A. Gasnikov. Stochastic optimization with heavy-tailed noise
via accelerated gradient clipping. Advances in Neural Information Processing Systems, 33:
15042–15053, 2020.

[22] R. M. Gower, N. Loizou, X. Qian, A. Sailanbayev, E. Shulgin, and P. Richtárik. Sgd: General
analysis and improved rates. In Proceedings of the 36th International Conference on Machine
Learning, pages 5200–5209. PMLR, 2019.

[23] M. Gurbuzbalaban, U. Simsekli, and L. Zhu. The heavy-tail phenomenon in sgd. In International
Conference on Machine Learning, pages 3964–3975. PMLR, 2021.

[24] F. R. Hampel, E. M. Ronchetti, P. Rousseeuw, and W. A. Stahel. Robust statistics: the approach
based on influence functions. Wiley-Interscience; New York, 1986.

[25] N. J. Harvey, C. Liaw, Y. Plan, and S. Randhawa. Tight analyses for non-smooth stochastic
gradient descent. In Conference on Learning Theory, pages 1579–1613. PMLR, 2019.

[26] E. Hazan and S. Kale. Beyond the regret minimization barrier: Optimal algorithms for stochastic
strongly-convex optimization. Journal of Machine Learning Research, 15:2489–2512, 2014.

[27] S. Hopkins, J. Li, and F. Zhang. Robust and heavy-tailed mean estimation made simple, via
regret minimization. Advances in Neural Information Processing Systems, 33:11902–11912,
2020.

[28] S. B. Hopkins. Mean estimation with sub-gaussian rates in polynomial time. The Annals of
Statistics, 48(2):1193–1213, 2020.

[29] D. Hsu and S. Sabato. Loss minimization and parameter estimation with heavy tails. The
Journal of Machine Learning Research, 17(1):543–582, 2016.

[30] P. J. Huber. Robust estimation of a location parameter. Ann. Math. Statist., 35(4):73–101, 1964.

[31] P. Jain, D. M. Nagaraj, and P. Netrapalli. Making the last iterate of sgd information theoretically
optimal. SIAM Journal on Optimization, 31(2):1108–1130, 2021.

[32] J. C. Lee and P. Valiant. Optimal sub-gaussian mean estimation in very high dimensions. In
13th Innovations in Theoretical Computer Science Conference (ITCS 2022). Schloss-Dagstuhl-
Leibniz Zentrum für Informatik, 2022.

[33] X. Li and Q. Sun. Variance-aware decision making with linear function approximation under
heavy-tailed rewards. Transactions on Machine Learning Research.

[34] Z. Liu and Z. Zhou. Stochastic nonsmooth convex optimization with heavy-tailed noises. arXiv
preprint arXiv:2303.12277, 2023.

[35] Z. Liu, T. D. Nguyen, T. H. Nguyen, A. Ene, and H. Nguyen. High probability convergence
of stochastic gradient methods. In International Conference on Machine Learning, pages
21884–21914. PMLR, 2023.

[36] G. Lugosi and S. Mendelson. Mean estimation and regression under heavy-tailed distributions:
A survey. Foundations of Computational Mathematics, 19(5):1145–1190, 2019.

[37] G. Lugosi and S. Mendelson. Sub-gaussian estimators of the mean of a random vector. The
annals of statistics, 47(2):783–794, 2019.

[38] V. V. Mai and M. Johansson. Stability and convergence of stochastic gradient clipping: Beyond
lipschitz continuity and smoothness. In International Conference on Machine Learning, pages
7325–7335. PMLR, 2021.

[39] S. Minsker. Geometric median and robust estimation in banach spaces. Bernoulli, 21(4):
2308–2335, 2015.

[40] A. V. Nazin, A. S. Nemirovsky, A. B. Tsybakov, and A. B. Juditsky. Algorithms of robust
stochastic optimization based on mirror descent method. Automation and Remote Control, 80
(9):1607–1627, 2019.

12


---Page Break---
[41] T. D. Nguyen, T. H. Nguyen, A. Ene, and H. Nguyen. Improved convergence in high probability
of clipped gradient methods with heavy tailed noise. Advances in Neural Information Processing
Systems, 36:24191–24222, 2023.

[42] R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks.
In International conference on machine learning, pages 1310–1318. Pmlr, 2013.

[43] A. Pensia, V. Jog, and P.-L. Loh. Robust regression with covariate filtering: Heavy tails and
adversarial contamination. arXiv preprint arXiv:2009.12976, 2020.

[44] A. Prasad, A. S. Suggala, S. Balakrishnan, and P. Ravikumar. Robust estimation via robust
gradient estimation. arXiv preprint arXiv:1802.06485, 2018.

[45] A. Prasad, S. Balakrishnan, and P. Ravikumar. A robust univariate mean estimator is all you
need. In International Conference on Artificial Intelligence and Statistics, pages 4034–4044.
PMLR, 2020.

[46] N. Puchkin, E. Gorbunov, N. Kutuzov, and A. Gasnikov. Breaking the heavy-tailed noise barrier
in stochastic optimization problems. In International Conference on Artificial Intelligence and
Statistics, pages 856–864. PMLR, 2024.

[47] A. Rakhlin, O. Shamir, and K. Sridharan. Making gradient descent optimal for strongly convex
stochastic optimization. In Proceedings of the 29th International Coference on International
Conference on Machine Learning, pages 1571–1578, 2012.

[48] A. Sadiev, M. Danilova, E. Gorbunov, S. Horváth, G. Gidel, P. Dvurechensky, A. Gasnikov, and
P. Richtárik. High-probability bounds for stochastic optimization and variational inequalities:
the case of unbounded variance. In International Conference on Machine Learning, pages
29563–29648. PMLR, 2023.

[49] V. Srinivasan, A. Prasad, S. Balakrishnan, and P. K. Ravikumar. Efficient estimators for
heavy-tailed machine learning. 2020.

[50] Q. Sun, W.-X. Zhou, and J. Fan. Adaptive huber regression. Journal of the American Statistical
Association, 115(529):254–265, 2020.

[51] J. Tropp. Freedman’s inequality for matrix martingales. Electronic Communications in Proba-
bility, 16(none):262 – 270, 2011. doi: 10.1214/ECP.v16-1624. URL https://doi.org/10.
1214/ECP.v16-1624.

[52] C.-P. Tsai, A. Prasad, S. Balakrishnan, and P. Ravikumar. Heavy-tailed streaming statistical
estimation. In International Conference on Artificial Intelligence and Statistics, pages 1251–
1282. PMLR, 2022.

[53] J. W. Tukey. Mathematics and the picturing of data. In Proceedings of the International
Congress of Mathematicians, Vancouver, 1975, volume 2, pages 523–531, 1975.

[54] R. Vershynin. High-dimensional probability: An introduction with applications in data science,
volume 47. Cambridge university press, 2018.

[55] H. Wang and A. Ramdas. Catoni-style confidence sequences for heavy-tailed mean estimation.
Stochastic Processes and Their Applications, 163:168–202, 2023.

[56] J. Zhang, S. P. Karimireddy, A. Veit, S. Kim, S. Reddi, S. Kumar, and S. Sra. Why are adaptive
methods good for attention models? Advances in Neural Information Processing Systems, 33:
15383–15393, 2020.

[57] W.-X. Zhou, K. Bose, J. Fan, and H. Liu. A new perspective on robust m-estimation: Finite
sample theory and applications to dependence-adjusted multiple testing. Annals of statistics, 46
(5):1904, 2018.

13


---Page Break---
Contents

1
Introduction
1

1.1
Sub-Gaussian Error Guarantees for Statistical Estimation . . . . . . . . . . . . . .
2

1.2
Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3

1.3
Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
4

2
Notation and Organization
4

3
Background and Problem Formulation
4

4
Results
5

4.1
Smooth Strongly Convex Objectives . . . . . . . . . . . . . . . . . . . . . . . . .
6

4.2
Beyond Strongly Convex Objectives . . . . . . . . . . . . . . . . . . . . . . . . .
6

5
Applications to Streaming Heavy Tailed Statistical Estimation
7

5.1
Streaming Heavy-Tailed Mean Estimation . . . . . . . . . . . . . . . . . . . . . .
7

5.2
Streaming Heavy Tailed Linear Regression
. . . . . . . . . . . . . . . . . . . . .
8

5.3
Streaming Heavy Tailed Logistic Regression . . . . . . . . . . . . . . . . . . . . .
8

5.4
Streaming Heavy Tailed LAD Regression . . . . . . . . . . . . . . . . . . . . . .
9

6
Improved Martingale Concentration via Iterative Refinement
9

7
Proof Sketch
10

8
Conclusion and Limitations
10

A Preliminaries
16

B Analysis for Smooth Strongly Convex Functions
18

B.1
Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

B.2
Proof of Lemma 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

B.3
Proof of Lemma 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

B.4
Proof of Lemma 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

C Analysis for Smooth Strongly Convex Functions Under Quadratic Growth Noise Model 29

C.1
Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

C.2
Proof of Lemma 8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34

C.3
Proof of Lemma 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36

D Analysis for Smooth Convex Functions
41

D.1
Proof of Theorem 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
43

D.2
Proof of Lemma 10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
43

D.3
Proof of Lemma 11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44

14


---Page Break---
D.4
Proof of Lemma 12 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44

D.5
Proof of Lemma 13 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45

E Analysis for Lipschitz Convex Functions
46

E.1
Proof of Lemma 14 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
48

E.2
Proof of Lemma 15 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

E.3
Proof of Lemma 16 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

E.4
Proof of Lemma 17 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

F
Improved Martingale Concentration via PAC Bayes Theory
50

F.1
Proof of Theorem 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51

F.2
Proof of Theorem 10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
55

F.3
Proof of Corollary 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
56

G Applications to Streaming Heavy Tailed Statistical Estimation
56

G.1
Streaming Heavy Tailed Mean Estimation : Proof of Corollary 1 . . . . . . . . . .
56

G.2
Streaming Heavy Tailed Linear Regression : Proof of Corollary 2 . . . . . . . . . .
57

G.3
Heavy Tailed Streaming Logistic Regression : Proof of Corollary 3 . . . . . . . . .
59

G.4
Proof of Corollary 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
61

15


---Page Break---
A
Preliminaries

In this section, we collect some preliminary concentration results which will be used in the future
sections. For the following lemma, we refer to Exercise 2.8.5 in [54].
Lemma 1. Suppose X is a real valued random variable such that |X| ≤Γ almost surely, EX = 0
and EX2 = ν. Then, for any λ ∈R such that |λ| ≤
1
2Γ, the following holds:

E exp(λX) ≤exp(λ2ν)

Consider a Rd valued martingale (Mt)T
t=0 with respect to the filtration (Ft)T
t=0 such that M0 = 0
almost surely. We consider the martingale difference sequence vt := Mt −Mt−1 for t ≥1. Clearly,
we must have:

Mt =

t
X

s=1
vs

Definition 1. We say that the martingale Mt satisfies (g, T, δ) uniform concentration if:

P( sup
0≤t≤T
∥Mt∥> g
√

T) ≤δ

Assume that for fixed Γ > 0 and Σ ∈Rd×d that ∥vs∥≤Γ almost surely and E[vtv⊺
t |Ft−1] =: Σt.
Suppose Tr(Σt) ≤qt and ∥Σt∥2 ≤pt almost surely for some non-random constants pt, qt. We
state a high dimensional version of Freedman’s inequality [20, 51] below which follows from From
Corollary 1.3 of [51], we have

Theorem 6. Suppose Mt satisfies the assumptions above. Let ¯q := 1

T
PT
s=1 qt the following is true:

P( sup
0≤t≤T
∥Mt∥> α) ≤(d + 1) exp(−
α2/2
¯qT + Γα

3
)

That is, for any δ > 0, the martingale (Mt)t≤T obeys (g0(δ), T, δ) uniform concentration, where

g0(δ) =
2Γ
3
√

T log( d+1

δ ) +
q

2¯q log( d+1

δ )

The following inequality is a corollary of Theorem 6.
Lemma 2. Let gt ∈Rd be Ft−1 measurable. Then for some constant c1 > 0, we have:

P(∪T
t=1{|

t
X

s=1
⟨gs, vs⟩| ≥α} ∩s≤t {∥gs∥≤As}) ≤2 exp(−
α2

ΓAα+c1
PT
t=1 ptA2
t )
(4)

Where A = sup1≤t≤T At

In addition, we also use the following scalar version of Freedman’s inequality
Lemma 3 (Freedman’s Inequality). Let h1, h2, . . . , hT be a Ft adapted martingale difference
sequence such that E[ht|Ft−1] = 0, E[h2
t|Ft−1] = σ2
t and ∥ht∥≤τ. Then, for any δ ∈(0, 1), the
following holds with probability at least 1 −δ:

t
X

s=1
hs ≤2

v
u
u
tln(1/δ)

t
X

s=1
σ2s + 2τ ln(1/δ)

The following lemma, which bounds the moments of a clipped random vector, is crucial to our
analysis of the bias and variance of the clipped stochastic gradient.
Lemma 4 (Moments of a Clipped Random Vector). Let z ∈Rd be a random vector sampled from the
distribution P with mean m and covariance matrix S. For any Γ > 0, let ˜z = clipΓ(z), and let ˜m and
˜S denote the mean and covariance of ˜z respectively, i.e., ˜m = E[˜z] and ˜S = E

(˜z −˜m) (˜z −˜m)T 
.
Then, the following hold:

∥˜m −m∥≤

p

∥S∥2

Γ


∥m∥+
p

Tr(S)

+ ∥m∥

Γ2
 
∥m∥2 + Tr(S)


∥˜S∥2 ≤∥S∥2 + ∥m∥2

Γ2
 
∥m∥2 + Tr(S)


Tr(˜S) ≤Tr(S)

16


---Page Break---
Proof. The proof of this lemma uses arguments similar to that of Catoni and Giulini [8]. We first
note that for any x ∈Rd

clipΓ(x) = x · min{1, Γ−1∥x∥}

Γ−1∥x∥
Following the proof of Proposition 2.1 of Catoni and Giulini [8], we observe that for any t > 0:

0 ≤1 −min{1, t}

t
≤inf
p≥1
pptp

(p + 1)p+1

Define θ(x) = min{1,Γ−1∥x∥}

Γ−1∥x∥
∀x ∈Rd. Note that clipΓ(x) = θ(x) · x. From the above inequality,
we note that:

0 ≤1 −θ(x) ≤inf
p≥1
pp

(p + 1)p+1 · ∥x∥p

Γp
(5)

Consider any unit vector e ∈Rd. Then,
⟨e, m −˜m⟩= E [⟨e, z −˜z⟩]
= E [⟨e, z −θ(z)z⟩]
= E [(1 −θ(z)) ⟨e, z −m⟩] + ⟨e, m⟩E [(1 −θ(z))]
≤E [(1 −θ(z))| ⟨e, z −m⟩|] + ∥m∥E [(1 −θ(z))]

≤E

inf
p≥1
pp

(p + 1)p+1 · ∥z∥p| ⟨e, z −m⟩|

Γp


+ ∥m∥E

inf
p≥1
pp

(p + 1)p+1 · ∥z∥p

Γp



where the second step uses the definition of θ(z) and the last step uses equation (5). Now, substituting
p = 1 and p = 2 in the first and second terms of the RHS respectively, we obtain the following:

⟨e, m −˜m⟩≤1

ΓE [∥z∥⟨e, z −m⟩] + ∥m∥

Γ2 E[∥z∥2]

≤1

Γ

p

E[∥x∥2]
q

E[⟨e, z −m⟩2] + ∥m∥

Γ2 E[∥z∥2]

≤

p

∥S∥
Γ
·
p

∥m∥2 + Tr(S) + ∥m∥

Γ2
 
∥m∥2 + Tr(S)


≤

p

∥S∥2

Γ


∥m∥+
p

Tr(S)

+ ∥m∥

Γ2
 
∥m∥2 + Tr(S)


where the second step uses the Cauchy Schwarz inequality and the last step uses the subadditivity of
the square root. It follows that:
∥˜m −m∥= sup
∥e∥=1
⟨e, m −˜m⟩

≤

p

∥S∥2

Γ


∥m∥+
p

Tr(S)

+ ∥m∥

Γ2
 
∥m∥2 + Tr(S)


To bound ∥˜S∥, we first note that for any x ∈Rd, 0 ≤θ(x) ≤1. As before, let e ∈Rd denote
an arbitrary unit vector. We note that E[⟨e, ˜z −m⟩2] = E[⟨e, ˜z −˜m⟩2] + E[⟨e, m −˜m⟩2] ≥
E[⟨e, ˜z −˜m⟩2]. Hence, it follows that,

E
h
⟨e, ˜z −˜m⟩2i
≤E
h
⟨e, ˜z −m⟩2i

≤E
h
(θ(z) ⟨e, z⟩−⟨e, m⟩)2i

= E
h
(θ(z) ⟨e, z −m⟩−(1 −θ(z)) ⟨e, m⟩)2i

≤E
h
θ(z) ⟨e, z −m⟩2i
+ ⟨e, m⟩2 E [(1 −θ(z))]

≤E
h
⟨e, z −m⟩2i
+ ∥m∥2E

inf
p≥1
pp

(p + 1)p+1
∥z∥p

Γp



≤∥S∥2 + ∥m∥2E[∥z∥2]

Γ2

≤∥S∥2 + ∥m∥2

Γ2
 
∥m∥2 + Tr(S)


17


---Page Break---
where the fourth step uses Jensen’s inequality by noting that 0 ≤θ(z) ≤1 and the fifth step uses
equation (5).

Finally, To upper bound Tr(˜S), we note that clipΓ is a contractive mapping as it is the projection
operator onto a convex set (namely the ball of radius Γ in Rd centered at the origin). To this end,

Tr(˜S) = E

∥˜z −˜m∥2
= 1

2Ez1,z2
i.i.d.
∼P

∥clipΓ(z1) −clipΓ(z2)∥2

≤1

2Ez1,z2
i.i.d.
∼P

∥z1 −z2∥2
= Tr(S)

The following result, which is a corollary of Theorem 5, is vital for controlling the error introduced
due to the variance of the stochastic gradients, and is one of the major components of our analysis.
The proof of this result is presented in Appendix F.3

Corollary 5 (PAC Bayesian Inequality for Quadratic Variation). Let v1, . . . , vT be an Rd val-
ued martingale difference sequence adapted to the filtration F1, . . . , FT satisfying E[vs|Fs] =
0, E[vsvT
s |Fs] = Σs and ∥vs∥≤τ almost surely. Let UP(t) := min(T, 2⌈log2 t⌉). Suppose
∥Σs∥2 ≤ps and Tr(Σs) ≤qs for some fixed sequences p1, . . . , pT and q1, . . . , qT . Then, there exists
a universal constant Clower such that whenever T > Clower log((1 +
√¯qT

Γ ) log(d + 1)) such that the
following inequality holds with probability at least 1 −δ, for any δ ∈(0, 1

2):

t
X

s=1
∥vs∥2 ≤CM

UP(t)
X

s=1
qs + CMτ 2 ln( ln(T )

δ
)2 + CMt

τ 2

UP(t)
X

s=1
p2
s
∀t ∈[T]

where CM > 0 is an absolute numerical constant.

B
Analysis for Smooth Strongly Convex Functions

Let deff =
Tr(Σ)

∥Σ∥2 and let K = 4 max{8, CM, ln(T)}. For t ≥1, define the filtration Ft =
σ (x1, gs|1 ≤s ≤t) and F0 = σ(x1). Furthermore, let ∇F(xt) = clipΓ(gt) + bt + vt where
bt = ∇F(xt) −E[clipΓ(gt)|Ft−1] and vt = E[clipΓ(gt)|Ft−1] −clipΓ(gt).
We note that
E[vt|Ft−1] = 0 and

∥vt∥≤∥clipΓ(gt)∥−∥E[clipΓ(gt)|Ft−1]∥
≤∥clipΓ(gt)∥−E[∥clipΓ(gt)∥|Ft−1] ≤2Γ

where the first step follows from the triangle inequality, the second step uses Jensen’s inequality and
the last step uses the definition of clipΓ. Hence vt is an F adapted almost surely bounded martingale
difference sequence. Now, let Dt = ∥xt −x∗∥where x∗is the unique minimizer of F (guaranteed
by strong convexity).Let ηt =
A
t+γ where A ≥1 is a numerical constant and γ ≥Aκ + A −1 is a
constant depending on κ, d and ln(1/δ) which we shall specify later. Note that our choice of γ ensures
that ηt ≤
1
L+µ for t ∈[1 : T] We prove the following recurrence for Dt by using the smoothness and
strong convexity properties of F and by exploiting the choice of the step-size.

Lemma 5 (Recurrence for Dt). The following holds for every t ∈[1 : T]

D2
t+1 ≤
γ + 1

t + γ

2A
D2
1 + A22A+1

µ

t
X

s=1

(s + γ −1)2A−1

(t + γ)2A
⟨bs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥bs∥2 (s + γ)2A−2

(t + γ)2A
+ A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A
⟨vs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥vs∥2 (s + γ)2A−2

(t + γ)2A

18


---Page Break---
Now define RT,δ as follows:

RT,δ = (γ + 1)2D2
1 + (T + γ)∥Σ∥2

µ2


deff +
p

deff ln(K/δ)


It is easy to see that Γ =
µ√

RT,δ
ln(K/δ) . In our proof of Theorem 1, we shall establish that the following
holds with probability at least 1 −δ:

D2
t ≤
CRT,δ
(t + γ −1)2 ∀t ∈[1 : T + 1]

where C > 0 is an absolute numerical constant to be chosen later. To this end, we define the event Et
and the random variables dt, ˜bt, ˜vt as follows for t ∈[1 : T + 1]:

Et =

D2
t ≤
CRT,δ
(t + γ −1)2



dt = (xt −x∗)1 {Et}
˜bt = bt1 {Et}
˜vt = vt1 {Et}

We note that since xt is Ft−1 measurable, so are 1 {Et} , Dt, dt, bt and ˜bt.
Furthermore,
E[˜vt|Ft−1] = E[vt|Ft−1]1 {Et} = 0.

We use the following Lemma to control the bias vector ˜bt
Lemma 6 (Bias Control). The following holds almost surely for every t ∈[1 : T]:

∥˜bt∥≤µ
p

RT,δ

 
1
T + γ +
κ ln(1/δ)
√

C

(t + γ −1)
p

d(T + γ)
+ κ3C
3/2 ln(1/δ)2

(t + γ −1)3
+
κ
√

C ln(1/δ)2

(t + γ −1)(T + γ)

!

We use the following lemma to control the variance vector ˜vt. The proof of this lemma, which uses
Freedman’s inequality and the PAC Bayesian martingale concentration inequality of Corollary 6.
Lemma 7 (Variance Control). The following holds with probability at least 1 −δ uniformly for every
t ∈[T] whenever A ≥3 and γ ≥4 max{κ
4/3C
2/3 ln(ln(T )/δ), κ
√

C ln(ln(T )/δ)
3/2}:

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≤27µRT,δ
√

C

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≤CMµ2RT,δ
 
6 + 3 · 24A−13 + 3 · 24A−17

where CM is the absolute numerical constant defined in Corollary 5.

Equipped with this bound on the bias and the variance, we now present the complete proof as follows:

B.1
Proof of Theorem 1

Proof. Let A ≥3, γ ≥4 max{κ
4/3C
2/3 ln(ln(T )/δ), κ
√

C ln(ln(T )/δ)
3/2}. Now, let E denote the
following event

E = {

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≤27µRT,δ
√

C ∀t ∈[T]

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≤CMµ2RT,δ
 
6 + 3 · 24A−13 + 3 · 24A−17
∀t ∈[T]}

Note that by Lemma 7, P(E) ≥1 −δ. We now claim that P
TT +1
t=1 Et|E

= 1, i.e., conditioned on

the event E, the following holds almost surely for every t ∈[1 : T + 1]

D2
t ≤
CRT,δ
(t + γ −1)2 ∀t ∈[1 : T + 1]

19


---Page Break---
We prove the above claim by induction. Note that the claim is trivially true for t = 1 as RT,δ ≥
(γ + 1)2D2
1. Now, consider any t ∈[1 : T] and suppose the claim holds for some 1 ≤s ≤t.

Recall that by Lemma 5

(t + γ)2D2
t+1 ≤
(γ + 1)2A

(t + γ)2A−2 D2
1 + A22A+1

µ

t
X

s=1

(s + γ −1)2A−1

(t + γ)2A−2
⟨bs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥bs∥2 (s + γ)2A−2

(t + γ)2A−2 + A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨vs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥vs∥2 (s + γ)2A−2

(t + γ)2A−2

Under the induction hypothesis, 1 {Es} = 1 ∀s ∈[t]. Hence, Under the induction hypothesis,

1
n
D2
s ≤
CRT,δ
(s+γ−1)(s+γ−2)
o
= 1 and thus, ds = xs −x∗, bs = ˜bs, vs = ˜vs ∀1 ≤s ≤t.
Substituting this transformation into the above inequality, we obtain the following:

(t + γ)2D2
t+1 ≤
(γ + 1)2A

(t + γ)2A−2 D2
1
|
{z
}
1⃝

+ A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩

|
{z
}
2⃝

+ A24A+1

µ2

t
X

s=1
∥˜vs∥2 (s + γ)2A−2

(t + γ)2A−2
|
{z
}
3⃝

+

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2

D
˜bs, ds
E

|
{z
}
4⃝

+ A24A+1

µ2

t
X

s=1
∥˜bs∥2 (s + γ)2A−2

(t + γ)2A−2
|
{z
}
5⃝

(6)

We now bound each of the terms in the RHS as follows.

Bounding 1
⃝
Since A ≥1 and t ≥1,

1⃝=
(γ + 1)2A

(t + γ)2A−2 D2
1 ≤(γ + 1)2D2
1 ≤RT,δ

Bounding 2
⃝
Since γ and A satisfy the conditions of Lemma 7 and we have conditioned on the
event E, it follows that:

A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≤27A22A+1RT,δ
√

C

Bounding 3
⃝
Since γ and A satisfy the conditions of Lemma 7 and we have conditioned on the
event E, it follows that:

A24A+1

µ2

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≤CM22A+2  
6 + 3 · 24A−13 + 3 · 24A−17
RT,δ

Before controlling terms 4⃝and 5⃝, we note that the following holds for every s ∈[t] by Lemma 6

∥bs∥≤µ
p

RT,δ (B1 + B2 + B3 + B4)

20


---Page Break---
where B1, . . . , B4 are defined as:

B1 =
1
T + γ

B2 =
κ ln(K/δ)
√

C

(s + γ −1)
p

d(T + γ)

B3 = κ3C
3/2 ln(K/δ)2

(s + γ −1)3

B4 =
κ ln(K/δ)2√

C
(s + γ −1)(T + γ)

Bounding 4
⃝
Since 1 {Es} = 1

∥ds∥≤

p

CRT,δ
s + γ −1 ≤2
p

CRT,δ
s + γ
Hence,

A22A+1

µ

t
X

s=1

D
˜bs, ds
E (s + γ)2A−1

(t + γ)2A−2 ≤A22A+2RT,δ
√

C

t
X

s=1

s + γ

t + γ

2A−2
(B1 + B2 + B3 + B4)

We now control the first term
t
X

s=1

s + γ

t + γ

2A−2
B1 =
1
T + γ

t
X

s=1

s + γ

t + γ

2A−2

≤
t
T + γ ≤1

where the first inequality follows from the fact that A ≥1 and s ≤t. We now bound the second term

t
X

s=1

s + γ

t + γ

2A−2
B2 ≤κ
√

C ln(K/δ)
p

d(T + γ)

"
t
X

s=1

s + γ

t + γ

2A−2
1
s + γ −1

#

Setting A ≥3/2 and using the fact that s + γ ≥2, it follows that

t
X

s=1

s + γ

t + γ

2A−2
B2 ≤2κ
√

C ln(K/δ)
p

d(T + γ)

t
X

s=1

(s + γ)2A−3

(t + γ)2A−2

≤2κ
√

C ln(K/δ)
p

d(T + γ)
≤2

where the last inequality follows by setting γ ≥Cκ2

d
· ln(K/δ)2

To control the third term, we set A ≥5/2 and proceed as follows:

t
X

s=1

s + γ

t + γ

2A−2
B3 ≤κ3C
3/2 ln(K/δ)2
t
X

s=1

(s + γ)2A−5

(t + γ)2A−2

≤κ3C
3/2 ln(K/δ)2

(t + γ)2

≤κ3C
3/2 ln(K/δ)2

(γ + 1)2
≤1

where the last inequality follows by setting γ ≥κ
3/2C
3/4 ln(K/δ).

To bound the last term,

t
X

s=1

s + γ

t + γ

2A−2
B4 ≤κC
1/2 ln(K/δ)2

T + γ

t
X

s=1

(s + γ)2A−3

(t + γ)2A−2

≤κC
1/2 ln(1/δ)2

γ + 1
≤1

21


---Page Break---
where the second inequality uses the fact that A ≥3/2 and the last inequality follows by setting
γ ≥κC
1/2 ln(K/δ)2. Putting it all together, it follows that

4⃝≤5A4A+1RT,δ
√

C

by setting γ as follows

γ ≥max
κ2C

d
· ln(K/δ)2, κ
3/2C
3/4 ln(1/δ), κC
1/2 ln(K/δ)2


Bounding 5
⃝
By Lemma 6 and Jensen’s inequality

∥˜bs∥2 ≤4µ2RT,δ
 
B2
1 + B2
2 + B2
3 + B2
4


It follows that

A222A+2

µ2

t
X

s=1
∥˜bs∥2
s + γ

t + γ

2A−2
≤A24A+2RT,δ

t
X

s=1

s + γ

t + γ

2A−2  
B2
1 + B2
2 + B2
3 + B2
4


The first term is controlled as follows using the fact that A ≥1

t
X

s=1

s + γ

t + γ

2A−2
B2
1 =

t
X

s=1

1
(T + γ)2 ≤1

The second term is controlled as

t
X

s=1

s + γ

t + γ

2A−2
B2
2 = 4κ2C ln(K/δ)2

d(T + γ)

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤4κ2C ln(K/δ)2

d(t + γ)(T + γ) ≤1

where the last inequality follows because γ ≥κ
q

C

d ln(K/δ)
For controlling the third term, we set A ≥4 to obtain

t
X

s=1

s + γ

t + γ

2A−2
B2
3 = κ6C3 ln(K/δ)4
t
X

s=1

(s + γ)2A−8

(t + γ)2A−2

≤κ6C3 ln(K/δ)4

(γ + 1)5
≤1

where the last inequality uses the fact that γ ≥κ
6/5C
3/5 ln(K/δ)
4/5 To control the fourth term, we use
the fact that A ≥2 to obtain

t
X

s=1

s + γ

t + γ

2A−2
B2
4 = κ2C ln(K/δ)4

(T + γ)2

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤κ2C ln(K/δ)4

(γ + 1)3
≤1

where the last inequality uses the fact that γ ≥κ
2/3C
1/3 ln(K/δ)
4/3 From the obtained bounds, we
conclude that 5⃝≤A24A+3RT,δ.

Hence, setting A = 4 and γ = 4C max{ ∥Σ∥2κ2 ln(ln(T )/δ)2

Tr(Σ)
, κ
3/2 ln(ln(T )/δ), κ ln(ln(T )/δ)2}, we ob-
tain the following

(t + γ)2D2
t+1 ≤1⃝+ 2⃝+ 3⃝+ 4⃝+ 5⃝

≤RT,δ
h
1 + CM22A+2  
6 + 3 · 24A−13 + 3 · 24A−17
+ A24A+3 +
√

C
 
27A22A+1 + 5A4A+1i

≤RT,δ

262145 + 524288CM + 75776
√

C


≤CRT,δ

22


---Page Break---
where the last inequality is obtained by setting C =
 √262145 + 524288CM + 75776
2. It follows
that

D2
t+1 ≤CRT,δ

(t + γ)2

Thus, we have proved by induction that conditioned on E, D2
t ≤CRT,δ

(t+γ)2 for every t ∈[T + 1]. In
particular, the following holds with probability at least 1 −δ:

D2
T +1 ≤C
 γ + 1

T + γ

2
D2
1 + C∥Σ∥2
 
deff + √deff ln(K/δ)


µ2(T + γ)

≲
 γ + 1

T + γ

2
D2
1 + Tr(Σ) +
p

∥Σ∥2Tr(Σ) ln(ln(T )/δ)
µ2(T + γ)

B.2
Proof of Lemma 5

Let ϵt = bt + vt

D2
t+1 = ∥ΠX (xt −ηt∇F(xt) + ηtϵt) −x∗∥2

≤∥xt −ηt∇F(xt) + ηtϵt∥2

≤D2
t −2ηt ⟨∇F(xt), xt −x∗⟩+ 2ηt ⟨ϵt, xt −x∗⟩+ 2η2
t ∥∇F(xt)∥2 + 2η2
t ∥ϵt∥2

By the coercivity lemma in Bubeck [6] ,

∥∇F(xt)∥2 ≤(L + µ) ⟨∇F(xt), xt −x∗⟩−LµD2
t

It follows that,

D2
t+1 ≤(1 −2η2
t Lµ)D2
t −2ηt[1 −ηt(L + µ)] ⟨∇F(xt), xt −x∗⟩+ 2ηt ⟨ϵt, xt −x∗⟩+ 2η2
t ∥ϵt∥2

≤(1 −2η2
t Lµ)D2
t −2ηt[1 −ηt(L + µ)]µD2
t + 2ηt ⟨ϵt, xt −x∗⟩+ 2η2
t ∥ϵt∥2

≤(1 −2ηtµ −2η2
t µ2)D2
t + 2ηt ⟨ϵt, xt −x∗⟩+ 2η2
t ∥ϵt∥2

≤(1 −2ηtµ)D2
t + 2ηt ⟨ϵt, xt −x∗⟩+ 2η2
t ∥ϵt∥2

where the second inequality follows from the strong monotonicity property of ∇F(x) and the fact
that ηt ≤
1
L+µ since γ ≥Aκ + A −1. Now, substituting ηt =
A
µ(t+γ),

D2
t+1 ≤

1 −2A

t + γ


D2
t +
2A
µ(t + γ) ⟨ϵt, xt −x∗⟩+ 2A2∥ϵt∥2

µ2(t + γ)2
(7)

Since 1 −t ≤e−t ∀t ∈R, we note that ∀s < t:

tY

j=s+1


1 −
2A
j + γ


≤exp



−

t
X

j=s+1

2A
j + γ





≤exp

−2A
Z t+1

s+1

du
u + γ



≤exp

−2A ln
 t + 1 + γ

s + 1 + γ



=
s + 1 + γ

t + 1 + γ

2A

≤22A
s + γ

t + γ

2A

23


---Page Break---
Using the above bound to unroll the recurence (7), we obtain:

D2
t+1 ≤




tY

j=1


1 −
2A
j + γ



D2
1 + 2A

µ

t
X

s=1

⟨ϵs, xs −x∗⟩

(s + γ)




tY

j=s+1


1 −
2A
j + γ





+ 2A2

µ2

t
X

s=1

∥ϵs∥2

(s + γ)2




tY

j=s+1


1 −
2A
j + γ





≤
γ + 1

t + γ

2A
D2
1 + A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A
⟨ϵs, xs −x∗⟩+ A222A+1

µ2

t
X

s=1
∥ϵs∥2 (s + γ)2A−2

(t + γ)2A

Expanding ϵs = bs + vs and using Young’s inequality, we conclude that the following holds for
every t ∈[1 : T]

D2
t+1 ≤
γ + 1

t + γ

2A
D2
1 + A22A+1

µ

t
X

s=1

(s + γ −1)2A−1

(t + γ)2A
⟨bs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥bs∥2 (s + γ)2A−2

(t + γ)2A
+ A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A
⟨vs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥vs∥2 (s + γ)2A−2

(t + γ)2A

B.3
Proof of Lemma 6

Note that by definition of Et
∥∇F(xt)∥1 {Et} ≤LDt1 {Et}

≤L

p

CRT,δ
(t + γ −1)

Recall that Γ =
µ√

RT,δ
ln(K/δ) i.e.
p

RT,δ = γ ln(K/δ)

µ
. Substituting this into the above inequality gives us:

∥∇F(xt)∥1 {Et} ≤κΓ ln(K/δ)
√

C
t + γ −1
(8)

We recall that bt = ∇F(xt) −E[clipΓ(gt)|Ft−1] = E[gt|Ft−1] −E[clipΓ(gt)|Ft−1]. Since
Cov[gt|Ft−1] ⪯Σ by Assumption Bdd. 2nd Moment, we obtain the following bound on ∥bt∥
by an application of Lemma 4

∥bt∥≤∥Σ∥2
√deff
Γ
+ ∥∇F(xt)∥
p

∥Σ∥2
Γ
+ ∥∇F(xt)∥3

Γ2
+ ∥Σ∥2deff∥∇F(xt)∥

Γ2

Since ˜bt = bt1 {Et}, it follows that

∥˜bt∥≤∥Σ∥2
√deff
Γ
|
{z
}
A
⃝

+ ∥∇F(xt)∥1 {Et}
p

∥Σ∥2
Γ
|
{z
}
B
⃝

+ ∥∇F(xt)∥31 {Et}

Γ2
|
{z
}
C
⃝

+ ∥Σ∥2deff∥∇F(xt)∥1 {Et}

Γ2
|
{z
}
D
⃝

Bounding A
⃝
By definition of Γ,

∥Σ∥2
√deff
Γ
= ∥Σ∥2
√deff ln(K/δ)
µ
p

RT,δ

≤(T + γ)∥Σ∥2
√deff ln(K/δ)
µT
p

RT,δ

≤µ
p

RT,δ
(T + γ)

Hence A
⃝≤
µ√

RT,δ
T +γ

24


---Page Break---
Bounding B
⃝
Since RT,δ ≥∥Σ∥2deff(T +γ)

µ2
≥∥Σ∥2T

µ2
,
p

∥Σ∥2 ≤
µ√

RT,δ
√

d(T +γ). Substituting this into

equation (8),

∥∇F(xt)∥1 {Et}
p

∥Σ∥2
Γ
≤κ
√

C ln(K/δ)
t + γ −1
·
µ
p

RT,δ
p

d(T + γ)

Hence, B
⃝≤µ
p

RT,δ ·
κ ln(1/δ)
√

C

(s+γ)√

d(T +γ

Bounding C
⃝
From equation (8),

∥∇F(xt)∥3

Γ2
≤κ3C
3/2Γ ln(1/δ)3

(t + γ −1)3

≤µ
p

RT,δ · κ3C
3/2 ln(1/δ)2

(t + γ −1)3

Hence, C
⃝≤µ
p

RT,δ · κ3C
3/2 ln(1/δ)2

(t+γ−1)3

Bounding D
⃝
Recall that,

∥Σ∥2deff ≤µ2RT,δ

T + γ

∥∇F(xt)∥1 {Et}

Γ
≤κ ln(K/δ)
√

C
(t + γ −1)

Γ = µ
p

RT,δ
ln(K/δ)

It follows that

D
⃝= ∥Σ∥2deff∥∇F(xt)∥1 {Et}

Γ2
≤µ
p

RT,δ ·
κ ln(K/δ)2√

C
(t + γ −1)(T + γ)

Hence,

∥˜bt∥≤µ
p

RT,δ

 
1
T + γ +
κ ln(1/δ)
√

C

(t + γ −1)
p

d(T + γ)
+ κ3C
3/2 ln(1/δ)2

(t + γ −1)3
+
κ
√

C ln(1/δ)2

(t + γ −1)(T + γ)

!

B.4
Proof of Lemma 7

For any s ∈[T], we recall that vs = E [clipΓ(gs)|Fs−1] −clipΓ(gs). Since E[gs|Fs−1] = ∇F(xs)
and Cov[gs|Fs−1] ⪯Σ, we obtain the following from Lemma 4

∥E

vsvT
s |Fs−1

∥2 = ∥Cov [clipΓ(gs)|Fs−1] ∥≤∥Σ∥2 + ∥∇F(xs)∥4

Γ2
+ ∥∇F(xs)∥2Tr(Σ)

Γ2

Tr
 
E

vsvT
s |Fs−1

= Tr (Cov [clipΓ(gs)|Fs−1]) ≤Tr(Σ)

For s ∈[1 : T] define E[˜vs˜vT
s |Fs−1] = ˜Σs. Since 1 {Es} is Fs−1-measurable and ˜vs = vs1 {Es},
it follows that ˜Σs = E

vsvT
s |Fs

1 {Es}. Hence, we conclude the following from the above
inequality

∥˜Σs∥2 ≤∥Σ∥2 + ∥∇F(xs)∥41 {Es}

Γ2
+ ∥∇F(xs)∥2Tr(Σ)1 {Es}

Γ2

Tr(˜Σs) ≤Tr(Σ)
(9)

Now, for s ∈[t], we define hs as follows:

hs = ⟨˜vs, ds⟩(s + γ)2A−1

(t + γ)2A−2

25


---Page Break---
Note that E[hs|Fs−1] = ⟨E[˜vs|Fs−1], ds⟩(s+γ)2A−1

(t+γ)2A−2 = 0. Furthermore, since ∥˜vs∥≤∥vs∥≤2Γ

and ∥ds∥≤
√

CRT,δ
s+γ−1

|hs| ≤2Γ ·

p

CRT,δ
s + γ −1 · (s + γ)2A−1

(t + γ)2A−2

≤4Γ
p

CRT,δ

s + γ

t + γ

2A−2

≤4µRT,δ
√

C
ln(K/δ)
(10)

For s ∈[t], define σ2
s = E[h2
s|Fs−1]. It follows that,

σ2
s = (s + γ)4A−2

(t + γ)4A−4 vT
s ˜Σsvs

≤(s + γ)4A−2

(t + γ)4A−4 ∥vs∥2∥˜Σs∥2

≤4CRT,δ ·
s + γ

t + γ

4A−4
∥˜Σs∥2

≤4CRT,δ

s + γ

t + γ

4A−4 
∥Σ∥2 + ∥∇F(xs)∥4

Γ2
+ ∥∇F(xs)∥2∥Σ∥2deff

Γ2



where the last inequality follows from equation (9) and the fact that deff = Tr(Σ)/∥Σ∥2. We now use
the above inequality to control Pt
s=1 σ2
s ln(K/δ) as follows:

t
X

s=1
σ2
s ln(K/δ) ≤4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4
∥Σ∥2

+ 4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥4

Γ2

+ 4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥2∥Σ∥2deff

Γ2
(11)

We now control each of the three terms in the above inequality as follows

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4
∥Σ∥2 ≤4CRT,δ ln(K/δ)∥Σ∥2t

≤4CtRT,δ ·
µ2RT,δ
(T + γ)√deff
≤4µ2CR2
T,δ
Before controlling the remaining two terms, we recall from (8) in the proof of Lemma ?? that

∥∇F(xs)∥1 {Es} ≤κΓ ln(K/δ)
√

C
s + γ −1

≤2κΓ ln(K/δ)
√

C
s + γ

where Γ =
µ√

RT,δ
ln(K/δ) . It follows that

∥∇F(xs)∥4

Γ2
≤16κ4C2Γ2 ln(K/δ)4

(s + γ)4

= µ2RT,δ · 16κ4C2 ln(K/δ)2

(s + γ)4

26


---Page Break---
Thus, we can control the second term in equation (11) as follows

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥4

Γ2
≤64µ2CR2
T,δ · κ4C2 ln(K/δ)3
t
X

s=1

(s + γ)4A−8

(t + γ)4A−4

≤64µ2CR2
T,δ · κ4C2 ln(K/δ)3

(t + γ)3

≤64µ2CR2
T,δ
where the second inequality follows by setting A ≥2 and the last inequality follows by setting
γ ≥κ
4/3C
2/3 ln(K/δ).

To control the third term in (11), we note that by equation (8) and the definition of RT,δ
∥∇F(xs)∥2∥Σ∥2deff

Γ2
≤4µ2RT,δ ·
κ2C ln(K/δ)2

(T + γ)(s + γ)2

It follows that

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥2∥Σ∥2deff

Γ2
≤16µ2CR2
T,δ · κ2C ln(K/δ)3

T + γ

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4

≤16µ2CR2
T,δ · κ2C ln(K/δ)3

(T + γ)(t + γ)

≤16µ2CR2
T,δ
where the second inequality follows by setting A ≥3/2 and the last inequality follows by setting
γ ≥κ
√

C ln(K/δ)
3/2. Substituting the above bounds into equation (11), we note that

t
X

s=1
σ2
s ln(K/δ) ≤84µ2CRT,δ

Thus, by Freedman’s inequality (Lemma 3), we conclude that the following holds with probability at
least 1 −δ/2 uniformly for every t ∈[T]:

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩=

t
X

s=1
hs ≤2

v
u
u
t

t
X

s=1
σ2s ln(K/δ) + 8µRT,δ
√

C ≤27RT,δ
√

C
(12)

To prove the second inequality of this lemma, we define zs = ˜vs ·

s+γ
t+γ
A−1
for s ∈[t]. Note

that E[zs|Fs−1] = 0 and ∥zs∥≤∥˜vs∥≤2Γ. Define the PSD matrices Gs = E[zszT
s |Fs−1] =

s+γ
t+γ
2A−2 ˜Σs. Recalling that Tr(˜Σs) ≤Tr(Σ) and the bound obtained on ∥˜Σs|2 in equation (9),
we infer the following:

Tr(Gs) ≤
s + γ

t + γ

2A−2
Tr(Σ)

∥Gs∥2 ≤
s + γ

t + γ

2A−2
∥Σ∥2 +
s + γ

t + γ

2A−2 ∥∇F(xs)∥41 {Es}

Γ2

+
s + γ

t + γ

2A−2 ∥∇F(xs)∥2Tr(Σ)1 {Es}

Γ2

Substituting (8) into the bound for ∥Gs∥2, we obtain the following

Tr(Gs) ≤qs =
s + γ

t + γ

2A−2
Tr(Σ)

∥Gs∥2 ≤ps =
s + γ

t + γ

2A−2
∥Σ∥2 + (s + γ)2A−6

(t + γ)2A−2 · 16κ4C2 ln(K/δ)2µ2RT,δ

+ (s + γ)2A−4

(t + γ)2A−2 · 4κ2C ln(K/δ)2∥Σ∥2deff
(13)

27


---Page Break---
By Cauchy Schwarz Inequality,

p2
s ≤3
s + γ

t + γ

4A−4
∥Σ∥2
2 + 3 · (s + γ)4A−12

(t + γ)4A−4 · 256κ8C4 ln(K/δ)4µ4R2
T,δ

+ 3 · (s + γ)4A−8

(t + γ)4A−4 · 16κ4C2 ln(K/δ)4∥Σ∥2
2d2
eff
(14)

Since T ≳ln(ln(d)), K = ln(ln(T)) and qs ≤Tr(Σ) ∀s ∈[T], our choice of Γ ensures that the
conditions of Corollary 5 are satisfied. Hence, by Corollary 5, we conclude that the following holds
with probability 1 −δ/2 uniformly for all t ∈[T]

t
X

s=1
∥zs∥2 ≤4CMΓ2 ln(K/δ) + CM

UP(t)
X

s=1
qs + CMt

4Γ2

UP(t)
X

s=1
p2
s

Simplifying the above using equations (13), (14) and the definition of Γ, we obtain the following
inequality which holds with probability at least 1 −δ/2 uniformly for every t ∈[T]:

t
X

s=1
∥zs∥2 ≤4CMµ2RT,δ + CM

UP(t)
X

s=1

s + γ

t + γ

2A−2
Tr(Σ) + 3CM

4

UP(t)
X

s=1

s + γ

t + γ

4A−4 t ln(K/δ)2∥Σ∥2

µ2RT,δ

+ 3CM

4

UP(t)
X

s=1

(s + γ)4A−12

(t + γ)4A−4 · 256tκ8C4 ln(K/δ)6µ2RT,δ

+ 3CM

4

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4
16tκ4C2 ln(K/δ)6Tr(Σ)2

µ2RT,δ
(15)

We now simplify each term in the above inequality by using the fact that UP(t) ≤min{T, 2t}. To
this end, the second term is simplified as follows by using the fact that A ≥1

UP(t)
X

s=1

s + γ

t + γ

4A−4
Tr(Σ) ≤UP(t)Tr(Σ) ≤µ2RT,δ

We now control the third term as follows using the definition of RT,δ and the fact that A ≥1:

UP(t)
X

s=1

s + γ

t + γ

4A−4 t ln(K/δ)2∥Σ∥2

µ2RT,δ
≤µ2RT,δ ·
tUP(t)
d(T + γ)2

≤µ2RT,δ
To control the fourth term, we use the fact that A ≥3 and note that for s ≤2t, (s + γ) ≤2(t + γ)

UP(t)
X

s=1

(s + γ)4A−12

(t + γ)4A−4 · 256tκ8C4 ln(K/δ)6µ2RT,δ ≤µ2RT,δ28κ8C4 ln(K/δ)6
2t
X

s=1

(s + γ)4A−12

(t + γ)4A−4

≤µ2RT,δ · t224A−3κ8C4 ln(K/δ)6

(t + γ)8

≤µ2RT,δ · 24A−3κ8C4 ln(K/δ)6

(t + γ)6

≤µ2RT,δ24A−15

where the last inequality follows by setting γ ≥4κ
4/3C
4/3 ln(K/δ)

We control the last term by a similar argument

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4
16tκ4C2 ln(K/δ)6Tr(Σ)2

µ2RT,δ
≤µ2RT,δ ·
t
(T + γ)2 · 24κ4C2 ln(K/δ)6
2t
X

s=1

(s + γ)4A−8

(t + γ)4A−4

≤
t2

(T + γ)2(t + γ)4 · 24A−3κ4C2 ln(K/δ)6

≤24A−11µ2RT,δ

28


---Page Break---
where the last inequality follows by setting γ ≥4κ
√

C ln(K/δ)
3/2. Substituting the obtained bounds
into equation (15), we conclude that the following holds with probability at least 1 −δ/2 uniformly
for every t ∈[T]:

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 =

t
X

s=1
∥zs∥2 ≤CMµ2RT,δ
 
6 + 3 · 24A−13 + 3 · 24A−17

The proof is completed via a union bound.

C
Analysis for Smooth Strongly Convex Functions Under Quadratic Growth
Noise Model

Following a convention similar to that of Section B, let K = 4 max{8, CM, ln(T)}. For t ≥1,
define the filtration Ft = σ (x1, gs|1 ≤s ≤t) and F0 = σ(x1). Furthermore, let ∇F(xt) =
clipΓ(gt)+bt+vt where bt = ∇F(xt)−E[clipΓ(gt)|Ft−1] and vt = E[clipΓ(gt)|Ft−1]−clipΓ(gt).
As beforem, we note that E[vt|Ft−1] = 0 and ∥vt∥≤2Γ. Hence vt is an F adapted almost surely
bounded martingale difference sequence. Now, let Dt = ∥xt −x∗∥where x∗is the unique minimizer
of F (guaranteed by strong convexity). We also define Σt = Σ(xt) and note that ∥Σt∥≤αD2
t + β
and Tr(Σt) ≤deff
 
αD2
t + β

. Furthermore Σt is Ft−1 measurable. Let ηt =
A
t+γ where A ≥1 is
a numerical constant and γ ≥Aκ + A −1 is a constant depending on κ, d and ln(1/δ) which we
shall specify later. Note that our choice of γ ensures that ηt ≤
1
L+µ for t ∈[1 : T] An application of
Lemma 5 shows that Dt satisfies the following for every t ∈[1 : T]

D2
t+1 ≤
γ + 1

t + γ

2A
D2
1 + A22A+1

µ

t
X

s=1

(s + γ −1)2A−1

(t + γ)2A
⟨bs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥bs∥2 (s + γ)2A−2

(t + γ)2A
+ A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A
⟨vs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥vs∥2 (s + γ)2A−2

(t + γ)2A

We now define RT,δ as follows:

RT,δ = (γ + 1)2D2
1 + (T + γ)β

µ2


deff +
p

deff ln(K/δ)


It is easy to see that Γ =
µ√

RT,δ
ln(K/δ) . In our proof of Theorem 1, we shall establish that the following
holds with probability at least 1 −δ:

D2
t ≤
CRT,δ
(t + γ −1)2 ∀t ∈[1 : T + 1]

where C > 0 is an absolute numerical constant to be chosen later. To this end, we define the event Et
and the Ft measurable random variables dt, ˜bt, ˜vt as follows for t ∈[1 : T + 1]:

Et =

D2
t ≤
CRT,δ
(t + γ −1)2



dt = (xt −x∗)1 {Et}
˜bt = bt1 {Et}
˜vt = vt1 {Et}

We use the following Lemma to control the bias vector ˜bt
Lemma 8 (Bias Control). The following holds almost surely for every t ∈[1 : T]:

∥˜bt∥≤µ
p

RT,δ

7
X

j=1
Bj

29


---Page Break---
where B1, . . . , B7 are defined as follows:

B1 =
1
T + γ ,

B2 = 4αC
√

d ln(ln(T )/δ)
µ2(s + γ)2
,

B3 = 2κ
√

C ln(ln(T )/δ)

(s + γ)
p

d(T + γ)
,

B4 = 4κC ln(ln(T )/δ)√α

µ(s + γ)2
,

B5 = 8κ3C
3/2 ln(ln(T )/δ)2

(s + γ)3
,

B6 = 2κ
√

C ln(ln(T )/δ)2

(s + γ)(T + γ) ,

B7 = 8ακd ln(ln(T )/δ)2C
3/2

µ2(s + γ)3

We use the following lemma to control the variance vector ˜vt. The proof of this lemma, which uses
Freedman’s inequality and the PAC Bayesian martingale concentration inequality of Corollary 6.

Lemma 9 (Variance Control). The following holds with probability at least 1 −δ uniformly for every

t ∈[T] for A ≥3 and γ ≥4C max{ αdeff

µ2 , α ln(K/δ)

µ2
, κ
4/3 ln(K/δ), κ ln(K/δ)
3/2, κ
2/3d
1/3
eff α
1/3

µ2/3
ln(K/δ)}

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≲34 · µRT,δ
√

C

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≲CM


24A−3 25

4 + 5 · 24A−11 + 5 · 24A−16 + 5 · 24A−13

µ2RT,δ

where CM is the absolute numerical constant defined in Corollary 5.

Equipped with this bound on the bias and the variance, we now present the complete proof as follows:

C.1
Proof of Theorem 2

Proof. Let A ≥3, γ ≥4C max{ αdeff

µ2 , α ln(K/δ)

µ2
, κ
4/3 ln(K/δ), κ ln(K/δ)
3/2, κ
2/3d
1/3
eff α
1/3

µ2/3
ln(K/δ)}.
Now, let E denote the following event

E = {

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≤34 · µRT,δ
√

C ∀t ∈[T]

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≤53 · CMµ2RT,δ ∀t ∈[T]}

Note that by Lemma 9, P(E) ≥1 −δ. We now claim that P
TT +1
t=1 Et|E

= 1, i.e., conditioned on

the event E, the following holds almost surely for every t ∈[1 : T + 1]

D2
t ≤
CRT,δ
(t + γ −1)2 ∀t ∈[1 : T + 1]

We prove the above claim by induction. Note that the claim is trivially true for t = 1 as RT,δ ≥
(γ + 1)2D2
1. Now, consider any t ∈[1 : T] and suppose the claim holds for some 1 ≤s ≤t.

30


---Page Break---
Recall that by Lemma 5

(t + γ)2D2
t+1 ≤
(γ + 1)2A

(t + γ)2A−2 D2
1 + A22A+1

µ

t
X

s=1

(s + γ −1)2A−1

(t + γ)2A−2
⟨bs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥bs∥2 (s + γ)2A−2

(t + γ)2A−2 + A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨vs, xs −x∗⟩

+ A24A+1

µ2

t
X

s=1
∥vs∥2 (s + γ)2A−2

(t + γ)2A−2

Under the induction hypothesis, 1 {Es} = 1 ∀s ∈[t]. Hence, Under the induction hypothesis,

1
n
D2
s ≤
CRT,δ
(s+γ−1)(s+γ−2)
o
= 1 and thus, ds = xs −x∗, bs = ˜bs, vs = ˜vs ∀1 ≤s ≤t.
Substituting this transformation into the above inequality, we obtain the following:

(t + γ)2D2
t+1 ≤
(γ + 1)2A

(t + γ)2A−2 D2
1
|
{z
}
1⃝

+ A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩

|
{z
}
2⃝

+ A24A+1

µ2

t
X

s=1
∥˜vs∥2 (s + γ)2A−2

(t + γ)2A−2
|
{z
}
3⃝

+

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2

D
˜bs, ds
E

|
{z
}
4⃝

+ A24A+1

µ2

t
X

s=1
∥˜bs∥2 (s + γ)2A−2

(t + γ)2A−2
|
{z
}
5⃝

(16)

We now bound each of the terms in the RHS as follows.

Bounding 1
⃝
Since A ≥1 and t ≥1,

1⃝=
(γ + 1)2A

(t + γ)2A−2 D2
1 ≤(γ + 1)2D2
1 ≤RT,δ

Bounding 2
⃝
Since γ and A satisfy the conditions of Lemma 7 and we have conditioned on the
event E, it follows that:

A22A+1

µ

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩≤17A4A+1RT,δ
√

C

Bounding 3
⃝
Since γ and A satisfy the conditions of Lemma 7 and we have conditioned on the
event E, it follows that:

A24A+1

µ2

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 ≤A222A+2CM


24A−3 25

4 + 5 · 24A−11 + 5 · 24A−16 + 5 · 24A−13

RT,δ

Bounding 4
⃝
Since 1 {Es} = 1

∥ds∥≤

p

CRT,δ
s + γ −1 ≤2
p

CRT,δ
s + γ

Hence, by Lemma 8

A22A+1

µ

t
X

s=1

D
˜bs, ds
E (s + γ)2A−1

(t + γ)2A−2 ≤A22A+2RT,δ
√

C

t
X

s=1

s + γ

t + γ

2A−2
7
X

j=1
Bj

31


---Page Break---
We now control the first term
t
X

s=1

s + γ

t + γ

2A−2
B1 =
1
T + γ

t
X

s=1

s + γ

t + γ

2A−2

≤
t
T + γ ≤1

where the first inequality follows from the fact that A ≥1 and s ≤t.

We now control the second term
t
X

s=1

s + γ

t + γ

2A−2
B2 ≤4αC
√

d ln(K/δ)
µ2

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤4αC
√

d ln(K/δ)
µ2(t + γ)
≤1

where the first inequality follows from the fact that A ≥2 and s ≤t and the second inequality
follows by setting γ ≥4αC
√

d ln(K/δ)
µ2
.

We now bound the third term as follows:
t
X

s=1

s + γ

t + γ

2A−2
B3 ≤2κ
√

C ln(K/δ)
p

d(T + γ)

t
X

s=1

(s + γ)2A−3

(t + γ)2A−2

≤2κ
√

C ln(K/δ)
p

d(T + γ)
≤1

where we use the fact that A ≥2 and set γ ≥4κ2 ln(K/δ)2

d
.

We now bound the fourth term as follows:
t
X

s=1

s + γ

t + γ

2A−2
B4 ≤4κC ln(K/δ)√α

µ

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤4κC ln(K/δ)√α

µ(t + γ)
≤1

where A ≥2 and γ ≥4κC ln(K/δ)√α

µ
We now bound the fifth term as follows
t
X

s=1

s + γ

t + γ

2
B5 ≤8κ3C
3/2 ln(K/δ)2
t
X

s=1

(s + γ)2A−5

(t + γ)[2A −2]

≤8κ3C
3/2 ln(K/δ)2

(t + γ)2
≤1

where A ≥3 and γ ≥4κ
3/2C
3/4 ln(K/δ).

We now bound the sixth term as follows
t
X

s=1

s + γ

t + γ

2A−2
B6 ≤2κ
√

C ln(K/δ)2

T + γ

t
X

s=1

(s + γ)2A−3

(t + γ)2A−2

≤2κ
√

C ln(K/δ)2

T + γ
≤1

where A ≥3 and γ ≥2κ
√

C ln(K/δ)2

Finally, we control the seventh term as follows

t
X

s=1

s + γ

t + γ

2A−2
B7 ≤8ακd ln(K/δ)2C
3/2

µ2

t
X

s=1

(s + γ)2A−5

(t + γ)2A−2

≤8ακd ln(K/δ)2C
3/2

µ2(t + γ)2
≤1

32


---Page Break---
where A ≥3 and γ ≥4
√

ακd ln(K/δ)C
3/4

µ
. Putting it all together, it follows that

4⃝≤7A4A+1RT,δ
√

C

by setting γ as follows

γ ≥4C max

(
α
√

d ln(K/δ)

µ2
, κ2 ln(K/δ)2

d
, κ√α ln(K/δ)

µ
, κ
3/2 ln(K/δ), κ ln(K/δ)2,

√

καd ln(K/δ)

µ

)

Bounding 5
⃝
By Lemma 8 and Jensen’s inequality

∥˜bs∥2 ≤7µ2RT,δ

7
X

j=1
B2
j

It follows that

A222A+2

µ2

t
X

s=1
∥˜bs∥2
s + γ

t + γ

2A−2
≤7A222A+2RT,δ

t
X

s=1

s + γ

t + γ

2A−2
7
X

j=1
B2
j

The first term is controlled as follows using the fact that A ≥1

t
X

s=1

s + γ

t + γ

2A−2
B2
1 =

t
X

s=1

1
(T + γ)2 ≤1

The second term is controlled as

t
X

s=1

s + γ

t + γ

2A−2
B2
2 ≤16α2C2d ln(K/δ)2

µ4

t
X

s=1

(s + γ)2A−6

(t + γ)2A−2

≤16α2C2d ln(K/δ)2

µ4(t + γ)3
≤1

where A ≥3 and γ ≥2
4/3α
2/3C
2/3d
1/3 ln(K/δ)
2/3

µ4/3
.

The third term is controlled as

t
X

s=1

s + γ

t + γ

2A−2
B2
3 = 4κ2C ln(K/δ)2

d(T + γ)

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤4κ2C ln(K/δ)2

d(t + γ)(T + γ) ≤1

where the last inequality follows because γ ≥κ
q

C

d ln(K/δ)

The fourth term is controlled as

t
X

s=1

s + γ

t + γ

2A−2
B2
4 ≤16κ2C2 ln(K/δ)2α

µ2

t
X

s=1

(s + γ)2A−6

(t + γ)2A−2

where A ≥3 and γ ≥2
4/3κ
2/3C
2/3 ln(K/δ)
2/3α
1/3

µ2/3
For controlling the fifth term, we set A ≥4 to obtain

t
X

s=1

s + γ

t + γ

2A−2
B2
5 = κ6C3 ln(K/δ)4
t
X

s=1

(s + γ)2A−8

(t + γ)2A−2

≤κ6C3 ln(K/δ)4

(γ + 1)5
≤1

where the last inequality uses the fact that γ ≥κ
6/5C
3/5 ln(K/δ)
4/5

33


---Page Break---
To control the sixth term, we use the fact that A ≥2 to obtain

t
X

s=1

s + γ

t + γ

2A−2
B2
6 = κ2C ln(K/δ)4

(T + γ)2

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2

≤κ2C ln(K/δ)4

(γ + 1)3
≤1

where the last inequality uses the fact that γ ≥κ
2/3C
1/3 ln(K/δ)
4/3

To control the seventh term, we set A ≥4 to obtain the following:

t
X

s=1

s + γ

t + γ

2A−2
B2
6 = 64α2κ2d ln(K/δ)4C3

µ4

t
X

s=1

(s + γ)2A−8

(t + γ)2A−2

≤64α2κ2d ln(K/δ)4C3

µ4(t + γ)5
≤1

where γ ≥
2
6/5α
2/5κ
2/5d
1/5 ln(K/δ)
4/5C
3/5

µ4/5
From the obtained bounds, we conclude that 5⃝≤

49A24A+1RT,δ.

Now, we set A = 4 and γ as follows:

γ = max

(
αd

µ2 , α
√

d ln(K/δ)

µ2
, κ√α ln(K/δ)

µ2
,

√

καd ln(K/δ)

µ
, κ
2/3d
1/3α
1/3 ln(K/δ)
µ
2/3
, κ
3/2 ln(K/δ), κ ln(K/δ)2, κ2 ln(K/δ)

d

)

Under this setting of A and γ, we obtain the following

(t + γ)2D2
t+1 ≤1⃝+ 2⃝+ 3⃝+ 4⃝+ 5⃝

≤RT,δ[1 + A222A+2CM


24A−3 25

4 + 5 · 24A−11 + 5 · 24A−16 + 5 · 24A−13


+ 49A24A+1 + 24A4A+1√

C]

≤RT,δ

802817 + 6946816CM + 98304
√

C


≤CRT,δ

where the second inequality holds due to our choice of A and γ and the last inequality is obtained by
setting C =
 √802817 + 6946816CM + 98304
2. It follows that

D2
t+1 ≤CRT,δ

(t + γ)2

Thus, we have proved by induction that conditioned on E, D2
t ≤
CRT,δ
(t+γ−1)2 for every t ∈[T + 1]. In
particular, the following holds with probability at least 1 −δ:

D2
T +1 ≤C
 γ + 1

T + γ

2
D2
1 + Cβ
 
deff + √deff ln(K/δ)


µ2(T + γ)

C.2
Proof of Lemma 8

Following the same steps as in that of the proof of Lemma 6, we use Lemma 4 and the fact that
Cov[gt|Ft−1] = Σt to obtain:

∥˜bs∥≤∥Σs∥√deff1 {Es}

Γ
|
{z
}
A
⃝

+ ∥∇F(xs)∥
p

∥Σs∥1 {Es}
Γ
|
{z
}
B
⃝

+ ∥∇F(xs)∥31 {Es}

Γ2
|
{z
}
C
⃝

+ ∥Σs∥deff∥∇F(xs)∥1 {Es}

Γ2
|
{z
}
D
⃝

34


---Page Break---
Bounding A
⃝
Note that by Assumption QG 2nd Moment

∥Σs∥21 {Es} ≤(β + αD2
s)1 {Es}

≤β + 4αCRT,δ

(s + γ)2

It follows that

∥Σs∥2
√

d1 {Es}
Γ
≤β√deff ln(K/δ)

µ
p

RT,δ
+ 4αC ln(K/δ)
p

RT,δdeff
µ(s + γ)2

Since β√deff ln(K/δ) ≤µ2RT,δ

T +γ , we obtain

A
⃝= ∥Σ∥s
√

d1 {Es}
Γ
≤µ
p

RT,δ

 
1
T + γ + 4αC ln(K/δ)
p

RT,δdeff
µ2(s + γ)2

!

Bounding B
⃝
Note that by equation (8),

∥∇F(xs)∥1 {Es}

Γ
≤2κ
√

C ln(K/δ)
s + γ

Furthermore, by Assumption QG 2nd Moment and the definition of Es

p

∥Σs∥21 {Es} ≤
p

β + 2
p

αCRT,δ
s + γ

Recalling that β ≤
µ2RT,δ
deff(T +γ),

∥∇F(xs)∥
p

∥Σs∥21 {Es}
Γ
≤2κ
√

C ln(K/δ)µ
p

RT,δ
(s + γ)
p

deff(T + γ)
+ 4κC ln(K/δ)
p

αRT,δ
(s + γ)2

≤µ
p

RT,δ

 
2κ
√

C ln(K/δ)

(s + γ)
p

deff(T + γ)
+ 4κC ln(K/δ)√α

µ(s + γ)2

!

Bounding C
⃝
By equation (8),

∥∇F(xs)∥31 {Es}

Γ2
≤µ
p

RT,δ · 8κ3C
3/2 ln(K/δ)2

(s + γ)3

Bounding D
⃝
Since βd ≤µ2RT,δ

T +γ , it follows tat

∥∇F(xs)∥∥Σs∥2deff1 {Es}

Γ2
≤2κ
√

C ln(K/δ)2

µ
p

RT,δ(s + γ)


βd + 4αCRT,δd

(s + γ)2



≤2κ
√

C ln(K/δ)2µ
p

RT,δ
(s + γ)(T + γ)
+ 8ακd ln(K/δ)2C
3/2p

RT,δ
µ(s + γ)3

≤µ
p

RT,δ

 
2κ
√

C ln(K/δ)2

(s + γ)(T + γ) + 8ακd ln(K/δ)2C
3/2

µ2(s + γ)3

!

Hence, we conclude that

∥˜bt∥≤A
⃝+ B
⃝+ C
⃝+ D
⃝≤µ
p

RT,δ

7
X

j=1
Bj

35


---Page Break---
where B1, . . . , B7 are defined as follows:

B1 =
1
T + γ ,

B2 = 4αC
√

d ln(ln(T )/δ)
µ2(s + γ)2
,

B3 = 2κ
√

C ln(ln(T )/δ)

(s + γ)
p

d(T + γ)
,

B4 = 4κC ln(ln(T )/δ)√α

µ(s + γ)2
,

B5 = 8κ3C
3/2 ln(ln(T )/δ)2

(s + γ)3
,

B6 = 2κ
√

C ln(ln(T )/δ)2

(s + γ)(T + γ) ,

B7 = 8ακd ln(ln(T )/δ)2C
3/2

µ2(s + γ)3

C.3
Proof of Lemma 9

As before, for s ∈[1 : T] define E[˜vs˜vT
s |Fs−1] = ˜Σs. Following the same steps as in that of the
proof of Lemma 7, we use Lemma 4 and the fact that Cov[gt|Ft−1] = Σt to obtain:

∥˜Σs∥2 ≤∥Σs∥21 {Es} + ∥∇F(xs)∥41 {Es}

Γ2
+ ∥∇F(xs)∥2Tr(Σs)1 {Es}

Γ2

≤1 {Et}
 
β + αD2
s

+ ∥∇F(xs)∥41 {Es}

Γ2
+ 1 {Es} ∥∇F(xs)∥2deff

Γ2
 
β + αD2
s


≤β + 4αCRT,δ

(s + γ)2 + ∥∇F(xs)∥41 {Es}

Γ2
+ ∥∇F(xt)∥2deff1 {Es}

Γ2


β + 4αCRT,δ

(s + γ)2



(17)

where the second inequality follows from Assumption QG 2nd Moment and the second inequality
follows by definition of Es
Furthermore, since clipΓ is a convex projection, the following holds:

Tr(˜Σs) ≤Tr(Σs)1 {Es}

≤deff
 
β + αD2
s

1 {Es}

≤βdeff + 4αdCRT,δ

(s + γ)2
(18)

Now, for s ∈[t], we define hs as follows:

hs = ⟨˜vs, ds⟩(s + γ)2A−1

(t + γ)2A−2

Note that E[hs|Fs−1] = 0. Furthermore, since ∥˜vs∥≤2Γ and ∥ds∥≤
√

CRT,δ
s+γ−1

|hs| ≤2Γ ·

p

CRT,δ
s + γ −1 · (s + γ)2A−1

(t + γ)2A−2

≤4Γ
p

CRT,δ

s + γ

t + γ

2A−2

≤4µRT,δ
√

C
ln(K/δ)
(19)

36


---Page Break---
For s ∈[t], define σ2
s = E[h2
s|Fs−1]. It follows that,

σ2
s = (s + γ)4A−2

(t + γ)4A−4 vT
s ˜Σsvs

≤(s + γ)4A−2

(t + γ)4A−4 ∥vs∥2∥˜Σs∥2

≤4CRT,δ ·
s + γ

t + γ

4A−4
∥˜Σs∥2

≤4CRT,δ

s + γ

t + γ

4A−4 
β + 4αCRT,δ

(s + γ)2 + ∥∇F(xs)∥41 {Es}

Γ2
+ ∥∇F(xt)∥2deff1 {Es}

Γ2


β + 4αCRT,δ

(s + γ)2



where the last inequality follows from equation (9) and the fact that deff = Tr(Σ)/∥Σ∥2. We now use
the above inequality to control Pt
s=1 σ2
s ln(K/δ) as follows:

t
X

s=1
σ2
s ln(K/δ) ≤4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4
β

+ 4CRT,δ ln(K/δ)

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4 4αCRT,δ

+ 4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥41 {Es}

Γ2

+ 4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥21 {Es} βdeff

Γ2

+ 4CRT,δ ln(K/δ)

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4
4∥∇F(xs)∥21 {Es} αdCRT,δ

Γ2
(20)

We now control each of the five terms in the above inequality as follows

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4
β ≤4CRT,δ ln(K/δ)βt

≤4CtRT,δ ·
µ2RT,δ
(T + γ)√deff
≤4µ2CR2
T,δ
To control the second term,

4CRT,δ ln(K/δ)

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4 4αCRT,δ ≤16CR2
T,δµ2 αC ln(K/δ)

µ2(t + γ)

≤16CR2
T,δµ2

where the second inequality follows by setting A ≥3/2 and the last inequality follows by setting
γ ≥αC ln(K/δ)

µ2
Before controlling the remaining terms, we recall from (8) in the proof of Lemma 6
that

∥∇F(xs)∥1 {Es} ≤κΓ ln(K/δ)
√

C
s + γ −1

≤2κΓ ln(K/δ)
√

C
s + γ

where Γ =
µ√

RT,δ
ln(K/δ) . It follows that

∥∇F(xs)∥41 {Es}

Γ2
≤16κ4C2Γ2 ln(K/δ)4

(s + γ)4

= µ2RT,δ · 16κ4C2 ln(K/δ)2

(s + γ)4

37


---Page Break---
Thus, we can control the third term in equation (20) as follows

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥41 {Es}

Γ2
≤64µ2CR2
T,δ · κ4C2 ln(K/δ)3
t
X

s=1

(s + γ)4A−8

(t + γ)4A−4

≤64µ2CR2
T,δ · κ4C2 ln(K/δ)3

(t + γ)3

≤64µ2CR2
T,δ

where the second inequality follows by setting A ≥2 and the last inequality follows by setting
γ ≥κ
4/3C
2/3 ln(K/δ).

To control the fourth term in (20), we note that by equation (8) and the definition of RT,δ

∥∇F(xs)∥2deffβ1 {Es}

Γ2
≤4µ2RT,δ ·
κ2C ln(K/δ)2

(T + γ)(s + γ)2

It follows that

4CRT,δ ln(K/δ)

t
X

s=1

s + γ

t + γ

4A−4 ∥∇F(xs)∥2βdeff

Γ2
≤16µ2CR2
T,δ · κ2C ln(K/δ)3

T + γ

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4

≤16µ2CR2
T,δ · κ2C ln(K/δ)3

(T + γ)(t + γ)

≤16µ2CR2
T,δ

where the second inequality follows by setting A ≥3/2 and the last inequality follows by setting
γ ≥κ
√

C ln(K/δ)
3/2.

To control the fifth term in equation (20), we proceed as follows:

4CRT,δ ln(K/δ)

t
X

s=1

(s + γ)4A−6

(t + γ)4A−4
4∥∇F(xs)∥21 {Es} αdCRT,δ

Γ2

≤64µ2CR2
T,δ
αdκ2C2 ln(K/δ)3

µ2

t
X

s=1

(s + γ)4A−8

(t + γ)4A−4

≤64µ2CR2
T,δ · αdκ2C2 ln(K/δ)3

µ2(t + γ)3

≤64µ2CR2
T,δ

where the second inequality follows by setting A ≥2 and the last inequality follows by setting

γ ≥α
1/3d
1/3
eff κ
2/3C
2/3 ln(K/δ)
µ2/3
Substituting the above bounds into equation (20), we note that

t
X

s=1
σ2
s ln(K/δ) ≤164µ2CRT,δ

Thus, by Freedman’s inequality (Lemma 3), we conclude that the following holds with probability at
least 1 −δ/2 uniformly for every t ∈[T]:

t
X

s=1

(s + γ)2A−1

(t + γ)2A−2 ⟨˜vs, ds⟩=

t
X

s=1
hs ≤2

v
u
u
t

t
X

s=1
σ2s ln(K/δ) + 8µRT,δ
√

C ≤34RT,δ
√

C
(21)

To prove the second inequality of this lemma, we define zs = ˜vs ·

s+γ
t+γ
A−1
for s ∈[t]. Note

that E[zs|Fs−1] = 0 and ∥zs∥≤∥˜vs∥≤2Γ. Define the PSD matrices Gs = E[zszT
s |Fs−1] =

s+γ
t+γ
2A−2 ˜Σs. Recalling the bounds obtained on ∥˜Σs∥2 and Tr(˜Σs) in equations (17) and (18), we

38


---Page Break---
infer the following:

Tr(Gs) ≤
s + γ

t + γ

2A−2
Tr(Σs)1 {Es}

≤
s + γ

t + γ

2A−2
βdeff + (s + γ)2A−4

(t + γ)2A−2 4αdeffCRT,δ

∥Gs∥2 =
s + γ

t + γ

2A−2
∥˜Σs∥2

≤
s + γ

t + γ

2A−2
β + (s + γ)2A−4

(t + γ)2A−2 4αCRT,δ +
s + γ

t + γ

2A−2 ∥∇F(xs)∥41 {Es}

Γ2

+
s + γ

t + γ

2A−2 ∥∇F(xs)∥21 {Es} βdeff

Γ2
+ (s + γ)2A−4

(t + γ)2A−2
∥∇F(xs)∥21 {Es} 4αdeffCRT,δ

Γ2

Substituting equation (8) into the bound for ∥Gs∥2, we obtain the following

Tr(Gs) ≤qs =
s + γ

t + γ

2A−2
βdeff + (s + γ)2A−4

(t + γ)2A−2 4αdeffCRT,δ

∥Gs∥2 ≤ps =
s + γ

t + γ

2A−2
β + (s + γ)2A−4

(t + γ)2A−2 · 4αCRT,δ + (s + γ)2A−6

(t + γ)2A−2 · 16κ4C2 ln(K/δ)2µ2RT,δ

≤(s + γ)2A−4

(t + γ)2A−2 · 4βdeffκ2C ln(K/δ)2 + (s + γ)2A−6

(t + γ)2A−2 · 16αdeffRT,δκ2C2 ln(K/δ)2 (22)

By Cauchy Schwarz inequality,

p2
s ≤5
s + γ

t + γ

4A−4
β2 + 5 · (s + γ)4A−8

(t + γ)4A−4 16α2C2R2
T,δ + 5 · (s + γ)4A−12

(t + γ)4A−4 · 256κ8C4 ln(K/δ)4µ4R2
T,δ

+ 5 · (s + γ)4A−8

(t + γ)4A−4 · 16β2d2
effκ4C2 ln(K/δ)4 + 5 · (s + γ)4A−12

(t + γ)4A−4 · 256α2d2
effR2
T,δκ4C4 ln(K/δ)4

(23)

Since T ≳ln(ln(d)), K = ln(ln(T)), our choice of Γ and the definition of RT,δ ensures that the
conditions of Corollary 5 are satisfied. Hence, by Corollary 5, we conclude that the following holds
with probability 1 −δ/2 uniformly for all t ∈[T]

t
X

s=1
∥zs∥2 ≤4CMΓ2 ln(K/δ)2 + CM

UP(t)
X

s=1
Qs + CMt

4Γ2

t
X

s=1
P 2
s

Simplyfing the above using equations (22), (23) and the definition of Γ, we obtain the following:

t
X

s=1
∥zs∥2 ≤4CMµ2RT,δ + CM

UP(t)
X

s=1

s + γ

t + γ

2A−2
βdeff + CM

UP(t)
X

s=1

(s + γ)2A−4

(t + γ)2A−2 · 4αdeffCRT,δ

+ 5CM

4

UP(t)
X

s=1

s + γ

t + γ

4A−4 β2t ln(K/δ)2

µ2RT,δ
+ 5CM

4

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4 · 16α2C2RT,δt ln(K/δ)2

µ2

+ 5CM

4

UP(t)
X

s=1

(s + γ)4A−12

(t + γ)4A−4 · 256κ8C4 ln(K/δ)6tµ2RT,δ

+ 5CM

4

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4 · β2d2
efft
µ2RT,δ
· 16κ4C2 ln(K/δ)6

+ 5CM

4

UP(t)
X

s=1

(s + γ)4A−12

(t + γ)4A−4 · 256κ4C4α2d2
eff ln(K/δ)6RT,δ
µ2
(24)

39


---Page Break---
We now simplify each term in the above inequality by using the fact that UP(t) ≤min{T, 2t}. To
this end, the second term is simplified as follows by using A ≥1

UP(t)
X

s=1

s + γ

t + γ

4A−4
βdeff ≤UP(t)βdeff ≤µ2RT,δ

We now control the third term by noting that for s ≤2t, s + γ ≤2t + γ ≤2(t + γ):

t
X

s=1

(s + γ)2A−4

(t + γ)2A−2 · 4αdeffCRT,δ ≤µ2RT,δ · 22A−2αdeff

µ2

2t
X

s=1

1
(t + γ)2

≤22A−1µ2RT,δ ·
αdt
µ2(t + γ)2

≤22A−3µ2RT,δ

where the last inquality follows by setting γ ≥4αCdeff

µ2
.

We now control the fourth term as follows:

UP(t)
X

s=1

s + γ

t + γ

4A−4 β2t ln(K/δ)2

µ2RT,δ
≤µ2RT,δ ·
UP(t)
d(T + γ)2 ≤µ2RT,δ

We now control the fifth term as follows:

16α2C2RT,δt ln(K/δ)2

µ2

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4 ≤µ2RT,δ · 24A−4α2C2 ln(K/δ)2t

µ4

2t
X

s=1

1
(t + γ)4

≤µ2RT,δ · α2C2 ln(K/δ)224A−5

µ4(t + γ)2

≤24A−9µ2RT,δ

where the last inequality usesγ ≥4αC ln(K/δ)

µ2

We now simplify the sixth term as follows:

UP(t)
X

s=1

(s + γ)4A−12

(t + γ)4A−4 · 256µ2RT,δκ8C4 ln(K/δ)6t ≤µ2RT,δt · 24A−4κ8C4 ln(K/δ)6
2t
X

s=1

1
(t + γ)8

≤µ2RT,δ

(t + γ)6 · 24A−3κ8C4 ln(K/δ)6

≤24A−15µ2RT,δ

where the last inequality follows by setting γ ≥4κ
4/3C
2/3 ln(K/δ).

We control the seventh term as follows:

UP(t)
X

s=1

(s + γ)4A−8

(t + γ)4A−4 · β2d2
efft
µ2RT,δ
· 16κ4C2 ln(K/δ)6 ≤µ2RT,δ · 24κ4C2 ln(K/δ)6t
(T + γ)2

2t
X

s=1

(s + γ)4A−8

(t + γ)4A−4

≤µ2RT,δ · 24A−3κ4C2 ln(K/δ)6

(t + γ)4

≤24A−11µ2RT,δ

where γ ≥4κ
√

C ln(K/δ)
3/2.

We use a similar argument to simplify the final term as follows:

UP(t)
X

s=1

t(s + γ)4A−12

(t + γ)4A−4
· 28α2d2
effRT,δκ4C4 ln(K/δ)6

µ2
≤µ2RT,δ · 24A−3α2d2
effκ4C4 ln(K/δ)6

µ4(t + γ)6

≤24A−15µ2RT,δ

40


---Page Break---
where γ ≥4κ
2/3d
1/3
eff C
2/3 ln(K/δ)
µ2/3
. We now set A ≥3 and γ as follows:

γ ≥4C max{αdeff

µ2 , α ln(K/δ)

µ2
, κ
4/3 ln(K/δ), κ ln(K/δ)
3/2, κ
2/3d
1/3
eff α
1/3

µ
2/3
ln(K/δ)}

Under these parameter settings, we substitute the obtained bounds into equation (15), we conclude
that the following holds with probability at least 1 −δ/2 uniformly for every t ∈[T]:

t
X

s=1

s + γ

t + γ

2A−2
∥˜vs∥2 =

t
X

s=1
∥zs∥2 ≤CMµ2RT,δ


24A−3 25

4 + 5 · 24A−11 + 5 · 24A−16 + 5 · 24A−13


The proof is completed via a union bound.

D
Analysis for Smooth Convex Functions

Let deff = Tr(Σ)

∥Σ∥2 . Following a convention similar to that of Section B, let K = 4 max{8, CM, ln(T)}.
For t ≥1, define the filtration Ft = σ (x1, gs|1 ≤s ≤t) and F0 = σ(x1).
Furthermore,
let ∇F(xt) = clipΓ(gt) + bt + vt where bt = ∇F(xt) −E[clipΓ(gt)|Ft−1] and vt =
E[clipΓ(gt)|Ft−1] −clipΓ(gt). As beforem, we note that E[vt|Ft−1] = 0 and ∥vt∥≤2Γ. Hence vt
is an F adapted almost surely bounded martingale difference sequence. Now, let Dt = ∥xt −x∗∥
where x∗is the minimizer of F considered in the statement of Theorem 3. Using the smoothness and
convexity properties of F, we first prove the following intermediate average iterate guarantee:
Lemma 10 (Intermediate Average Iterate Guarantee). The following holds for η ≤1/2L

F(ˆxT ) −F(x∗) ≤D2
1
2ηT + 1

T

T
X

t=1
⟨bt, xt −x∗⟩+ 1

T

T
X

t=1
⟨vt, xt −x∗⟩+ 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

Define the events Et and the random vectors dt, ˜bt and ˜vt as follows for t ∈[T]:

Et = {Dt ≤2D1}
dt = (xt −x∗)1 {Et}
˜bt = bt1 {Et}
˜vt = vt1 {Et}

We use the following lemma to control the bias

Lemma 11 (Bias Control). For every t ∈[T], ∥˜bt∥≤B where B is defined as follows:

B = ∥Σ∥2
√deff
Γ
+ 2LD1
p

∥Σ∥2
Γ
+ 8L3D3
1
Γ2
+ 2∥Σ∥2deffLD1

Γ2

We use the following lemma to control the varince
Lemma 12 (Variance Control). Let V ≥0 be defined as follows:

V = ∥Σ∥2 + 16L4D4
1
Γ2
+ 4L2D2
1∥Σ∥2deff

Γ2

Then the following holds with probability at least 1 −δ uniformly for every t ∈[T]

t
X

s=1
⟨˜vt, dt⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)

t
X

s=1
∥˜vt∥2 ≤CMg2T

where CM is a numerical constant and g2 is defined as follows

g2 = ∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2

41


---Page Break---
Let E denote the following event

E = {

t
X

s=1
⟨˜vs, ds⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)
∀t ∈[T]

t
X

s=1
∥˜vs∥2 ≤CMg2T
∀t ∈[T]}

We define the constant A as follows:

A = ∥Σ∥2
p

deff + LD1
p

∥Σ∥2 =
p

∥Σ∥2
p

Tr(Σ) + LD1


We now set the clipping level Γ =
q

AT
ln(K/δ). For this choice of Γ, we now obtain the following
bound on B:

B ≤∥Σ∥2
√deff
Γ
+ 2LD1
p

∥Σ∥2
Γ
+ 8L3D3
1
Γ2
+ 2∥Σ∥2deffLD1

Γ2

≤2

r

A ln(K/δ)

T
+ 2LD1 ln(K/δ)

AT
 
∥Σ∥2deff + L2D2
1

= B′
(25)

Similarly, we bound the value of V as follows:

V ≤∥Σ∥2 + 16L4D4 ln(K/δ)

AT
+ 4L2D2∥Σ∥2d ln(K/δ)

AT
= V ′
(26)

Equipped with the above inequality, we then bound the value of g as:

g ≤
p

∥Σ∥2deff + 2Γ ln(K/δ)
√

T
+ V
√

T
2Γ

≤
p

∥Σ∥2deff + 2
p

A ln(K/δ) + V
p

ln(K/δ)

2
√

A

≤
p

∥Σ∥2deff + 2
p

A ln(K/δ) + ∥Σ∥2
p

ln(K/δ)

2
√

A
+ 8L4D4 ln(K/δ)
3/2

A
3/2T
+ 2L2D2∥Σ∥2deff ln(K/δ)
3/2

A
3/2T

≤
p

∥Σ∥2deff + 3
p

A ln(K/δ) + 8L4D4 ln(K/δ)
3/2

A
3/2T
+ 2L2D2∥Σ∥2deff ln(K/δ)
3/2

A
3/2T
= g′
(27)

We prove the following lemma to control the growth of the iterates Dt.

Lemma 13 (Iterate Bound). Let η ≤c min{1/2L, D1/B′T, D1/g′√

T} where c =
1
√8CM+330. Then,
conditioned on the event E, Dt ≤2D1 ∀t ∈[T].

Equipped with the above lemmas, we now present a proof of the following theorem, which is a formal
restatement of Theorem 3

Theorem 7 (Smooth Convex Objectives). Let Convexity, L-smoothness and Bdd. 2nd Moment be
satisfied. Then, for any δ ∈(0, 1/2) and T ≥ln(ln(d)), there exists an η ∈(0, 1/2L] such that
the average iterate of Algorithm 1 run for T iterations with step-size ηt = η and clipping level

Γ =

r

T√

∥Σ∥2(√

Tr(Σ)+LD1)
ln(ln(T )/δ)
satisfies the following with probability at least 1 −δ :

F(ˆxT ) −F(x∗) ≲D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + LD1

ln(ln(T )/δ)

T
+ LD2
1
T

+ LD2
1 ln(ln(T )/δ)

T

s

Tr(Σ) + L2D2
1
∥Σ∥2
+ L2D3
1 ln(ln(T )/δ)
3/2

T
3/2

Tr(Σ) + L2D2
1
∥Σ∥3

1/4

42


---Page Break---
D.1
Proof of Theorem 7

We condition on the event E and let η = c

2 min{ 1

2L, D1

B′T ,
D1
g′√

T } where c =
1
√8CM+330. Note that
this choice of η satisfies the requirements of Lemma 10 and Lemma 13. By Lemma 10, the following
holds:

F(ˆxT ) −F(x∗) ≤D2
1
2ηT + 1

T

T
X

t=1
⟨bt, xt −x∗⟩+ 1

T

T
X

t=1
⟨vt, xt −x∗⟩+ 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

By Lemma 13, 1 {Et} = 1 ∀t ∈[T]. Hence, the following holds.

F(ˆxT ) −F(x∗) ≤D2
1
2ηT + 1

T

T
X

t=1

D
˜bt, dt
E
+ 1

T

T
X

t=1
⟨˜vt, dt⟩+ 2η

T

T
X

t=1
∥˜bt∥2 + 2η

T

T
X

t=1
∥˜vt∥2

≤D2
1
2ηT + 2BD1 + 2ηB2 + 2ηCMg2 + 4D1

r

V ln(K/δ)

T
+ 8D1Γ ln(K/δ)

T

≤D2
1
ηT + 3ηB2T + 2ηCMg2 + 4D1

r

V ln(K/δ)

T
+ 8D1

r

A ln(K/δ)

T
Where the second inequality uses Lemma 11 and the definition of the event E and the third inequality
uses ab ≤a2 + b2/4. For the rest of the proof, we shall use C to denote an absolute numerical constant
whose value can differ at every step. By our choice of the step-size

D2

ηT ≤CLD2
1
T
+ CD1B′ + CD1g′

√

T
3ηB2T ≤CD1B′

2ηCMg2 ≤CD1g′

√

T
Hence, conditioned on the event E, the following holds:

F(ˆxT ) −F(x∗) ≤CLD2
1
T
+ CD1B′ + CD1g′√

T + CD1

r

V ′ ln(K/δ)

T
+ CD1

r

A ln(K/δ)

T
Substituting the values of g′, B′ and V ′, we obtain the following:

F(ˆxT ) −F(x∗) ≤CLD2
1
T
+ CD1

r

A ln(K/δ)

T
+ CLD2
1 ln(K/δ)

T
·
Tr(Σ) + L2D2
1
A



+ CLD2
1 ln(K/δ)

T
·

r

Tr(Σ) + L2D2

A
+ CL2D3
1 ln(K/δ)
3/2

T
3/2
·
Tr(Σ) + L2D2
1
A
3/2



Substituting the value of A, we conclude that the following inequality holds almost surely conditioned
on the event E

F(ˆxT ) −F(x∗) ≲D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + LD1

ln(ln(T )/δ)

T
+ LD2
1
T

+ LD2
1 ln(ln(T )/δ)

T

s

Tr(Σ) + L2D2
1
∥Σ∥2
+ L2D3
1 ln(ln(T )/δ)
3/2

T
3/2

Tr(Σ) + L2D2
1
∥Σ∥3

1/4

The prooof is completed by observing that P(E) ≥1 −δ by Lemma 12 which implies that the above
inequality also holds with probability at least 1 −δ

D.2
Proof of Lemma 10

Proof. Since ΠC is a contractive operator

D2
t+1 = ∥xt+1 −x∗∥2 ≤D2
t −2η ⟨∇F(xt) −bt −vt, xt −x∗⟩+ η2∥∇F(xt) −bt −vt∥

≤D2
t −2η ⟨∇F(xt), xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩+ 2η ⟨vt, xt −x∗⟩

+ 2η2∥∇F(xt)∥+ 4η2∥vt∥2 + 4η2∥bt∥2

43


---Page Break---
By the coercivity property,

−2η ⟨∇F(xt), xt −x∗⟩≤−2η[F(xt) −F(x∗)] −η

L∥∇F(xt)∥2

Substituting this into the recurrence for D2
t+1, we obtain the following:

D2
t+1 ≤D2
t −2η[F(xt) −F(x∗)] + 2η ⟨vt, xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩

+ η(2η −1/L)∥∇F(xt)∥2 + 4η2∥vt∥2 + 4η2∥bt∥2

≤D2
t −2η[F(xt) −F(x∗)] + 2η ⟨vt, xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩+ 4η2∥vt∥2 + 4η2∥bt∥2

where the last inequality uses the fact that η ≤1/2L, Rearranging and taking averages on both sides

T
X

t=1
F(xt) −F(x∗) ≤D2
1
2ηT + 1

T

T
X

t=1
⟨bt, xt −x∗⟩+ 1

T

T
X

t=1
⟨vt, xt −x∗⟩+ 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

Using the above inequality and the convexity of F, we conclude that

F(ˆxT ) −F(x∗) = F

 

1
T

T
X

t=1
xt

!

−F(x∗)

≤1

T

T
X

t=1
F(xt) −F(x∗)

≤D2
1
2ηT + 1

T

T
X

t=1
⟨bt, xt −x∗⟩+ 1

T

T
X

t=1
⟨vt, xt −x∗⟩+ 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

D.3
Proof of Lemma 11

Note that by definition of Et

∥∇F(xt)∥1 {Et} ≤LDt1 {Et} ≤2LD1

We recall that bt = E[gt|Ft−1] −E[clipΓ(gt)|Ft−1]. Since Cov[gt|Ft−1] ⪯Σ by Assumption Bdd.
2nd Moment, we obtain the following bound on ∥bt∥by an application of Lemma 4

∥bt∥≤∥Σ∥2
√deff
Γ
+ ∥∇F(xt)∥
p

∥Σ∥2
Γ
+ ∥∇F(xt)∥3

Γ2
+ ∥Σ∥2deff∥∇F(xt)∥

Γ2

Since ˜bt = bt1 {Et}, it follows that

∥bt∥≤∥Σ∥2
√deff
Γ
+ ∥∇F(xt)1 {Et} ∥
p

∥Σ∥2
Γ
+ ∥∇F(xt)∥31 {Et}

Γ2
+ ∥Σ∥2deff∥∇F(xt)∥1 {Et}

Γ2

≤∥Σ∥2
√deff
Γ
+ 2LD1
p

∥Σ∥2
Γ
+ 8L3D3
1
Γ2
+ 2∥Σ∥2deffLD1

Γ2

D.4
Proof of Lemma 12

For any s ∈[T], we recall that vs = E [clipΓ(gs)|Fs−1] −clipΓ(gs). Since E[gs|Fs−1] = ∇F(xs)
and Cov[gs|Fs−1] ⪯Σ, we obtain the following from Lemma 4

∥E

vsvT
s |Fs−1

∥2 = ∥Cov [clipΓ(gs)|Fs−1] ∥≤∥Σ∥2 + ∥∇F(xs)∥4

Γ2
+ ∥∇F(xs)∥2Tr(Σ)

Γ2

Tr
 
E

vsvT
s |Fs−1

= Tr (Cov [clipΓ(gs)|Fs−1]) ≤Tr(Σ)

For s ∈[1 : T] define E[˜vs˜vT
s |Fs−1] = ˜Σs. Since 1 {Es} is Fs−1-measurable and ˜vs = vs1 {Es},
it follows that ˜Σs = E

vsvT
s |Fs−1

1 {Es}. Hence, we conclude the following from the above

44


---Page Break---
inequality

∥˜Σs∥2 ≤∥Σ∥2 + ∥∇F(xs)∥41 {Es}

Γ2
+ ∥∇F(xs)∥2Tr(Σ)1 {Es}

Γ2

≤∥Σ∥2 + 16L4D4
1
Γ2
+ 4L2D2
1Tr(Σ)
Γ2
= V

Tr(˜Σs) ≤Tr(Σ)
(28)

For s ∈[T], define hs = ⟨˜vs, ds⟩. We note that

|hs| ≤∥˜vs∥· ∥ds∥≤4ΓD1
E [hs|Fs−1] = ⟨E[˜vs|Fs−1], ds⟩= 0

E

h2
s|Fs−1

= dT
s E[˜vs˜vT
s ]ds

= dT
s ˜Σsds
≤∥ds∥2∥˜Σ∥≤4D2
1V

Hence, by Freedman’s inequality (Lemma 3), we conclude that the following holds with probability
at least 1 −δ/2:

t
X

s=1
⟨˜vt, dt⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)
∀t ∈[T]

We now apply Corollary 6 with ps = V , qs = Tr(Σ) and τ = 2Γ to conclude that the following
holds with probability at least 1 −δ/2 uniformly for every t ∈[T]

t
X

s=1
∥˜vs∥2 ≤4CMΓ2 ln(K/δ)2 + CMUP(t)Tr(Σ) + CMtUP(t)V 2

4Γ2

≤4CMΓ2 ln(K/δ)2 + CMTTr(Σ) + CMT 2V 2

4Γ2

≤CMT

∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2


= CMg2T

where

g2 = ∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2

The proof is concluded by a union bound

D.5
Proof of Lemma 13

We prove the claim via induction. Clearly, the claim is true for t = 1. Now, suppose the claim holds
for every s ≤t for some t ∈[T]. Since ΠC is a contractive operator

D2
t+1 = ∥xt+1 −x∗∥2 ≤D2
t −2η ⟨∇F(xt) −bt −vt, xt −x∗⟩+ η2∥∇F(xt) −bt −vt∥

≤D2
t −2η ⟨∇F(xt), xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩+ 2η ⟨vt, xt −x∗⟩

+ 2η2∥∇F(xt)∥+ 4η2∥vt∥2 + 4η2∥bt∥2

By the coercivity property,

−2η ⟨∇F(xt), xt −x∗⟩≤−2η[F(xt) −F(x∗)] −η

L∥∇F(xt)∥2

Substituting this into the recurrence for D2
t+1, we obtain the following:

D2
t+1 ≤D2
t −2η[F(xt) −F(x∗)] + 2η ⟨vt, xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩

+ η(2η −1/L)∥∇F(xt)∥2 + 4η2∥vt∥2 + 4η2∥bt∥2

≤D2
t + 2η ⟨vt, xt −x∗⟩+ 2η ⟨bt, xt −x∗⟩+ 4η2∥vt∥2 + 4η2∥bt∥2

45


---Page Break---
where we use the fact that η ≤1/2L. Now, by the Cauchy Schwarz inequality and the fact that
ab ≤a2 + b2/4 we obtain the following:

2η ⟨bt, xt −x∗⟩≤D2
t
2T + η2T∥bt∥2

It follows that

D2
t+1 ≤

1 + 1

2T


D2
t + 5η2T∥bt∥2 + 4η2∥vt∥2 −2η ⟨vt, xt −x∗⟩

Unrolling the above recursion for t steps and using the fact that (1 + 1/2T)T ≤2, we obtain the
following:

D2
t+1 ≤

1 + 1

2T

T
D2
1 +

t
X

s=1


1 + 1

2T

t−s  
5η2T∥bs∥2 + 4η2∥vs∥2 + 2η ⟨vs, xs −x∗⟩


≤2D2
1 +

t
X

s=1
10η2T∥bs∥2 + 8η2∥vs∥2 −4η ⟨vs, xs −x∗⟩

By the induction hypothesis, 1 {Es} = 1 ∀s ∈[t]. Hence,

D2
t+1 ≤2D2
1 + 10η2T

t
X

s=1
∥˜bs∥2 + 8η2
t
X

s=1
∥˜vs∥2 −4η

t
X

s=1
⟨˜vs, ds⟩

≤2D2
1 + 10η2T 2B2 + 8CMη2g2T + 16ηD1
hp

V t ln(K/δ) + 2Γ ln(K/δ)
i

≤3D2
1 + 10η2T 2B2 + 8CMη2g2T + 64η2 p

V t ln(K/δ) + 2Γ ln(K/δ)
2

≤3D2
1 + 10η2T 2B2 + 8CMη2g2T + 128η2V T ln(K/δ) + 1024Γ2 ln(K/δ)2

where the second inequality follows from the Lemma 11 and the fact that we have conditioned on E.
Note that by definition of g2 and the AM-GM inequality

g2T ≥4Γ2 ln(K/δ)2 + V 2T 2

4Γ2
≥max{4Γ2 ln(K/δ)2, 2V T ln(K/δ)}

It follows that

D2
t+1 ≤3D2
1 + 10η2T 2B2 + 8(CM + 40)η2g2T

≤3D2
1 + 10c2D2
1 + c2(8CM + 320)D2
1
≤4D2
1
where the second inequality uses the definition of η and the fact that B′ and g′ upper bound B
and G respectively by equations (25) and (27) and the last inequality sets c =
1
√8CM+330. Hence,
Dt+1 ≤2D1 which proves the claim by induction.

E
Analysis for Lipschitz Convex Functions

Let deff =
Tr(Σ)

∥Σ∥2 . Since Σ is positive semidefinite, 1 ≤deff ≤d. Moreover, let clipΓ(gt) =
∂F(xt) + bt + vt where bt = E[clipΓ(gt)|Ft] −∂F(xt) represents the bias due to clipping and
E[vt|Ft] = 0. Let Dt = ∥xt −x∗∥where x∗is the minimizer of F considered in the statement
of Theorem 3. Using the smoothness and convexity properties of F, we first prove the following
intermediate average iterate guarantee:
Lemma 14 (Intermediate Average Iterate Guarantee). The following holds for any η > 0

F(ˆxT ) −F(x∗) ≤D2
1
2ηT −1

T

T
X

t=1
⟨bt, xt −x∗⟩−1

T

T
X

t=1
⟨vt, xt −x∗⟩

+ ηG2 + 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

46


---Page Break---
Define the events Et and the random vectors dt as follows for t ∈[T]:
Et = {Dt ≤2D1}
dt = (xt −x∗)1 {Et}
We use the following lemma to control the bias
Lemma 15 (Bias Control). For every t ∈[T], ∥bt∥≤B where B is defined as follows:

B = ∥Σ∥2
√deff
Γ
+ G
p

∥Σ∥2
Γ
+ G3

Γ2 + ∥Σ∥2deffG

Γ2

We use the following lemma to control the varince
Lemma 16 (Variance Control). Let V ≥0 be defined as follows:

V = ∥Σ∥2 + G4

Γ2 + G2∥Σ∥2deff

Γ2

Then the following holds with probability at least 1 −δ uniformly for every t ∈[T]

t
X

s=1
⟨vs, ds⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)

t
X

s=1
∥vs∥2 ≤CMg2T

where CM is a numerical constant and g2 is defined as follows

g2 = ∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2

Let E denote the following event

E = {

t
X

s=1
⟨vs, ds⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)
∀t ∈[T]

t
X

s=1
∥vs∥2 ≤CMg2T
∀t ∈[T]}

Note that by Lemma 16, P(E) ≥1 −δ. We define the constant A as follows:

A = ∥Σ∥2
p

deff + G
p

∥Σ∥2 =
p

∥Σ∥2
p

Tr(Σ) + G


We now set the clipping level Γ =
q

AT
ln(K/δ). For this choice of Γ, we now simplify the expression
for B as follows:

B =

r

A ln(K/δ)

T
+ G
 
∥Σ∥2deff + G2
ln(K/δ)
AT
(29)

Similarly, the expression for V can be simplified as follows

V = ∥Σ∥2 + G2 ln(K/δ)

AT
 
∥Σ∥2deff + G2
(30)

Using the above inequality, we derive the following upper bound for g:

g ≤
p

∥Σ∥2deff + 2Γ ln(K/δ)
√

T
+ V
√

T
2Γ

=
p

∥Σ∥2deff + 2
p

A ln(K/δ) + V
p

ln(K/δ)

2
√

A

=
p

∥Σ∥2deff + 2
p

A ln(K/δ) + ∥Σ∥2
p

ln(K/δ)

2
√

A
+ G2 ln(K/δ)
3/2

A
3/2T

 
∥Σ∥2deff + G2

≤
p

∥Σ∥2deff + 3
p

A ln(K/δ) + G2 ln(K/δ)
3/2

A
3/2T

 
∥Σ∥2deff + G2
= g′
(31)

We also prove the following uniform upper bound on the iterates xt

47


---Page Break---
Lemma 17 (Iterate Bound). Let η ≤c min{D1/BT, D1/g′√

T, D1/G
√

T} where c =
1
√8CM+334. Then,
conditioned on the event E, Dt ≤2D1 ∀t ∈[T].

Equipped with the above lemmas, we now prove the following theorem which is a formal restatement
of Theorem 4

Theorem 8 (Lipschitz Convex Objectives). Let Assumptions Convexity, G-Lipschitzness and Bdd.
2nd Moment be satisfied. Then, for any δ ∈(0, 1/2) and T ≥ln(ln(d)), there exists an η ∈(0, G/
√

T]
such that the average iterate of Algorithm 1 run for T iterations with step-size ηt = η and clipping

level Γ =

r

T√

∥Σ∥2(√

Tr(Σ)+G)
ln(ln(T )/δ)
satisfies the following with probability at least 1 −δ

F(ˆxT ) −F(x∗) ≲D1G
√

T
+ D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + G

ln(K/δ)

T

+ D1G ln(K/δ)

T

s

Tr(Σ) + G2

∥Σ∥2
+ D1G2 ln(1/δ)
3/2

T
3/2

Tr(Σ) + G2

∥Σ∥3

1/4

E.1
Proof of Lemma 14

Proof. Since ΠC is a contractive operator

D2
t+1 = ∥xt+1 −x∗∥2 ≤D2
t −2η ⟨∂F(xt) + bt + vt, xt −x∗⟩+ η2∥∇F(xt) + bt + vt∥

≤D2
t −2η ⟨∂F(xt), xt −x∗⟩−2η ⟨bt, xt −x∗⟩−2η ⟨vt, xt −x∗⟩

+ 2η2∥∂F(xt)∥2 + 4η2∥vt∥2 + 4η2∥bt∥2

≤D2
t −2η[F(xt) −F(x∗)] −2η ⟨bt, xt −x∗⟩−2η ⟨vt, xt −x∗⟩+ 2η2G2 + 4η2∥bt∥2 + 4η2∥vt∥2

where the second inequality follows from the definition of the subgradient and the G lipschitzness of
F. Rearranging and taking averages on both sides

T
X

t=1
F(xt) −F(x∗) ≤D2
1
2ηT −1

T

T
X

t=1
⟨bt, xt −x∗⟩−1

T

T
X

t=1
⟨vt, xt −x∗⟩

+ ηG2 + 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

Using the above inequality and the convexity of F, we conclude that

F(ˆxT ) −F(x∗) = F

 

1
T

T
X

t=1
xt

!

−F(x∗)

≤1

T

T
X

t=1
F(xt) −F(x∗)

≤D2
1
2ηT −1

T

T
X

t=1
⟨bt, xt −x∗⟩−1

T

T
X

t=1
⟨vt, xt −x∗⟩

+ ηG2 + 2η

T

T
X

t=1
∥bt∥2 + 2η

T

T
X

t=1
∥vt∥2

48


---Page Break---
E.2
Proof of Lemma 15

We recall that bt = E[gt|Ft−1] −E[clipΓ(gt)|Ft−1]−. Since Cov[gt|Ft−1] ⪯Σ by Assumption
Bdd. 2nd Moment, we obtain the following bound on ∥bt∥by an application of Lemma 4

∥bt∥≤∥Σ∥2
√deff
Γ
+ ∥∂F(xt)∥
p

∥Σ∥2
Γ
+ ∥∂F(xt)∥3

Γ2
+ ∥Σ∥2deff∥∂F(xt)∥

Γ2

≤∥Σ∥2
√deff
Γ
+ G
p

∥Σ∥2
Γ
+ G3

Γ2 + ∥Σ∥2deffG

Γ2

E.3
Proof of Lemma 16

For any s ∈[T], we recall that vs = E [clipΓ(gs)|Fs−1] −clipΓ(gs). Since E[gs|Fs−1] = ∂F(xs)
and Cov[gs|Fs−1] ⪯Σ, we obtain the following from Lemma 4

∥E

vsvT
s |Fs−1

∥2 = ∥Cov [clipΓ(gs)|Fs−1] ∥≤∥Σ∥2 + ∥∂F(xs)∥4

Γ2
+ ∥∂F(xs)∥2Tr(Σ)

Γ2

≤∥Σ∥2 + G4

Γ2 + G2Tr(Σ)

Γ2

Tr
 
E

vsvT
s |Fs−1

= Tr (Cov [clipΓ(gs)|Fs]) ≤Tr(Σ)

For s ∈[T], define hs = ⟨vs, ds⟩. We note that
|hs| ≤∥vs∥· ∥ds∥≤4ΓD1
E [hs|Fs−1] = ⟨E[vs|Fs−1], ds⟩= 0

E

h2
s|Fs−1

= dT
s E[vsvT
s ]ds
= dT
s Σsds
≤∥ds∥2∥Σs∥≤4D2
1V
Hence, by Freedman’s inequality (Lemma 3), we conclude that the following holds with probability
at least 1 −δ/2:

t
X

s=1
⟨˜vt, dt⟩≤4D1
p

V t ln(K/δ) + 8ΓD1 ln(K/δ)
∀t ∈[T]

We now apply Corollary 6 with ps = V , qs = Tr(Σ) and τ = 2Γ to conclude that the following
holds with probability at least 1 −δ/2 uniformly for every t ∈[T]

t
X

s=1
∥˜vs∥2 ≤4CMΓ2 ln(K/δ)2 + CMUP(t)Tr(Σ) + CMtUP(t)V 2

4Γ2

≤4CMΓ2 ln(K/δ)2 + CMTTr(Σ) + CMT 2V 2

4Γ2

≤CMT

∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2


= CMg2T

where

g2 = ∥Σ∥2deff + 4Γ2 ln(K/δ)2

T
+ V 2T

4Γ2
The proof is concluded by a union bound

E.4
Proof of Lemma 17

We prove the claim via induction. Clearly, the claim is true for t = 1. Now, suppose the claim holds
for every s ≤t for some t ∈[T]. Since ΠC is a contractive operator

D2
t+1 = ∥xt+1 −x∗∥2 ≤D2
t −2η ⟨∂F(xt) + bt + vt, xt −x∗⟩+ η2∥∇F(xt) + bt + vt∥

≤D2
t −2η ⟨∂F(xt), xt −x∗⟩−2η ⟨bt, xt −x∗⟩−2η ⟨vt, xt −x∗⟩

+ 2η2∥∂F(xt)∥2 + 4η2∥vt∥2 + 4η2∥bt∥2

≤D2
t −2η[F(xt) −F(x∗)] −2η ⟨bt, xt −x∗⟩−2η ⟨vt, xt −x∗⟩+ 2η2G2 + 4η2∥bt∥2 + 4η2∥vt∥2

49


---Page Break---
where the second inequality follows from the definition of the subgradient and the G lipschitzness of
F. Now, by the Cauchy Schwarz inequality and the fact that ab ≤a2 + b2/4 we obtain the following:

−2η ⟨bt, xt −x∗⟩≤D2
t
2T + η2T∥bt∥2

It follows that

D2
t+1 ≤

1 + 1

2T


D2
t + 5η2T∥bt∥2 + 2η2G2 + 4η2∥vt∥2 −2η ⟨vt, xt −x∗⟩

Unrolling the above recursion for t steps and using the fact that (1 + 1/2T)T ≤2, we obtain the
following:

D2
t+1 ≤

1 + 1

2T

T
D2
1 +

t
X

s=1


1 + 1

2T

t−s  
5η2T∥bs∥2 + 2η2G2 + 4η2∥vs∥2 −2η ⟨vs, xs −x∗⟩


≤2D2
1 + 4η2G2T +

t
X

s=1
10η2T∥bs∥2 + 8η2∥vs∥2 −4η ⟨vs, xs −x∗⟩

By the induction hypothesis, 1 {Es} = 1 ∀s ∈[t]. Hence,

D2
t+1 ≤2D2
1 + 4η2G2T + 10η2T

t
X

s=1
∥bs∥2 + 8η2
t
X

s=1
∥vs∥2 −4η

t
X

s=1
⟨vs, ds⟩

≤2D2
1 + 4η2G2T + +10η2T 2B2 + 8CMη2g2T + 16ηD1
hp

V t ln(K/δ) + 2Γ ln(K/δ)
i

≤3D2
1 + 4η2G2T + +10η2T 2B2 + 8CMη2g2T + 64η2 p

V t ln(K/δ) + 2Γ ln(K/δ)
2

≤3D2
1 + 4η2G2T + +10η2T 2B2 + 8CMη2g2T + 128η2V T ln(K/δ) + 1024Γ2 ln(K/δ)2

where the second inequality follows from the Lemma 15 and the fact that we have conditioned on E.
Note that by definition of g2 and the AM-GM inequality

g2T ≥4Γ2 ln(K/δ)2 + V 2T 2

4Γ2
≥max{4Γ2 ln(K/δ)2, 2V T ln(K/δ)}

It follows that

D2
t+1 ≤3D2
1 + 4η2G2T + 10η2T 2B2 + 8(CM + 40)η2g2T

≤3D2
1 + 4c2D2
1 + 10c2D2
1 + c2(8CM + 320)D2
1
≤4D2
1

where the second inequality uses the definition of η and the fact that g′ upper bounds g, and the last
inequality sets c =
1
√8CM+334. Hence, Dt+1 ≤2D1 which proves the claim by induction.

F
Improved Martingale Concentration via PAC Bayes Theory

We have the following re-statement of Theorem 5 for the sake of readability.

Theorem 9. Suppose Mt for t = 0, . . . , T is an Rd valued martingale such that M0 = 0 almost surely,
the martingale difference sequence vt := Mt−Mt−1 is such that ∥vt∥≤Γ and E[vtv⊺
t |Ft−1] = Σt
almost surely for every t = 1, . . . , T for some Γ > 0. Assume that there are deterministic sequences
p1, . . . , pT and q1, . . . , qT such that Tr(Σt) ≤qt and ∥Σt∥≤pt almost surely.

Let ¯q := 1

T
PT
t=1 qt and ¯p := 1

T
PT
t=1 pt. Then, for any δ ∈(0, 1

2)

P(sup
t≤T
∥Mt∥≥g(T, δ)
√

T) ≤δ

Where g(T, δ) = C
h√¯q + ¯p
√

T
Γ
+
Γ
√

T log( K

δ )
i
and K = log Θ(log((
√¯qT

Γ
+ 1) log(d + 1)))

50


---Page Break---
Define the event At(g) := {∥Mt∥≤g
√

T} and Bt(g) := ∩t
s=1As. Consider the quantity Nt :=
∥Mt∥2 −Pt
s=1 ∥vs∥2.

Theorem 10. Let δ ∈(0, 1

2) and g = g(T, δ

2) be as defined in Theorem 9. Under the conditions of
Theorem 9, the following inequality holds for some large enough universal constant C.

P

{sup
t≤T
|Nt| > ΓCg
√

T log( 1

δ ) + CνgT 3/2

Γ
} ∩BT (g)

≤δ

The next corollary is a simple consequence of the Theorems 9 and 10.

Corollary 6. Let δ ∈(0, 1

2) and g = g(T, δ

3) be as specified in Theorem 9. Under the conditions of
Theorem 9, the following inequality holds with probability at-least 1 −δ:

T
X

t=1
∥vt∥2 ≤Cg2T

F.1
Proof of Theorem 9

The aim of this section is to prove the sharp concentration result given in Theorem 9. We now consider
the concentration of norms of the martingale ∥Mt∥. Define the event At := {∥Mt∥≤g
√

T} and
Bt = ∩t
s=1As. Let H be any stopping time for the martingale Mt. We have the following inequality
which follows from PAC-Bayes theory (see Equation 5.2.1, Page 159 in [7]).

Theorem 11. Suppose π be any measure over Rd and let M1(Rd) denote the space of all probability
measures over Rd. Let γ > 0 be arbitrary. Then conditioned on BT , with probability at-least 1 −δ,
the following inequality holds:

sup
ρ∈M1(Rd)
Eθ∼ργ

Mmin(H,T ), θ

−KL (ρ||||||π) ≤log

EMEθ∼π
exp(γ⟨Mmin(H,T ),θ⟩)1(BT )

δP(BT )


(32)

We will now bound the exponential moment: EMEθ∼π exp(γ ⟨Mt, θ⟩) whenever π = N(0, I).

Theorem 12. Let h(t) := Pt
s=1 log

1 + γ2

2 qt exp(γ2Γ2) + γ4ptg2T exp(2γ2Γg
√

T)

. Then,

Eθ∼π exp(γ⟨Mt, θ⟩−h(t))1(Bt)

is a supermartingale with respect to the filtration Ft

Proof. Let Σt := E[vtv⊺
t |Ft−1] and νt := ∥Σt∥. First, consider Eθ∼π exp(γ ⟨Mt, θ⟩). By the
properties of the Gaussians, we must have almost surely:

Eθ∼π exp(γ ⟨Mt, θ⟩)1(Bt) = exp( γ2∥Mt∥2

2
)1(Bt)
(33)

Using the fact that ∥Mt∥2 = ∥vt∥2 + 2⟨vt, Mt−1⟩+ ∥Mt−1∥2, we have:

E

exp( γ2∥Mt∥2

2
)1(Bt)
Ft−1


= E
h
exp( γ2∥Mt−1∥2

2
+ γ2∥vt∥2

2
+ γ2 ⟨vt, Mt−1⟩)1(Bt)
i

= E

exp( γ2∥vt∥2

2
+ γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1


exp( γ2∥Mt−1∥2

2
)1(Bt−1)
(34)

51


---Page Break---
We will now bound the quantity: E

exp( γ2∥vt∥2

2
+ γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1


. Using the convex-

ity of x →exp(x), we conclude:

E

exp(γ2∥vt∥2

2
+ γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1



≤E
1

2 exp(γ2∥vt∥2)1(At) + 1

2 exp(2γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1



≤1

2

1 + γ2Tr(Σt) exp(γ2Γ2)

+ E
1

2 exp(2γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1


(35)

In the second step, we have used the fact that exp(γ2∥vt∥2)1(At) ≤1+γ2∥vt∥2 exp(γ2Γ2) almost
surely using the power series expansion of the exp() function. Using the power series expansion of
exp(x), we have:

E

exp(2γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1


≤E

exp(2γ2 ⟨vt, Mt−1⟩)
Ft−1



= 1 + 2γ2E[⟨vt, Mt−1⟩
Ft−1] +
X

k≥2

2kγ2k

k!
E[(⟨vt, Mt−1⟩)kFt−1]

≤1 +
X

k≥2

2kγ2k

k!
E[(⟨vt, Mt−1⟩)2Γk−2∥Mt−1∥k−2Ft−1]

≤1 +
X

k≥2

2kγ2k

k!
⟨Mt−1, ΣtMt−1⟩Γk−2∥Mt−1∥k−2

≤1 +
X

k≥2

2kγ2k

k!
νtΓk−2∥Mt−1∥k ≤1 + 2γ4νt∥Mt−1∥2 exp(2γ2∥Mt−1∥Γ)
(36)

Here, νt = ∥Σt∥op In the second step we have used the fact that E[vt|Ft−1] = 0 and the fact that
⟨vt, Mt−1⟩≤Γ∥Mt−1∥almost surely. Plugging Equation (36) into Equation (35), we conclude:

E

exp(γ2∥vt∥2

2
+ γ2 ⟨vt, Mt−1⟩)1(At)
Ft−1



≤1 + γ2

2 Tr(Σt) exp(γ2Γ2) + γ4νt∥Mt−1∥2 exp(2γ2Γ∥Mt−1∥)
(37)

Using Equation (37) and that under the event Bt−1 we must have ∥Mt−1∥≤g
√

T, we conclude:

E

exp( γ2∥Mt∥2

2
)1(Bt)
Ft−1



≤

1 + γ2

2 qt exp(γ2Γ2) + γ4ptg2T exp(2γ2Γg
√

T)

exp( γ2∥Mt−1∥2

2
)1(Bt−1)

= exp(h(t) −h(t −1)) exp

γ2 ∥Mt−1∥2

2


1(Bt−1)
(38)

Therefore, by induction, we conclude the statement of the theorem.

Theorem 13. For any stopping time H,

EMEθ∼π exp(γ

Mmin(H,T ), θ

)1(BT ) ≤exp(h(T))
(39)

52


---Page Break---
Where h(T) = PT
t=1 log

1 + γ2qt

2
exp(γ2Γ2) + γ4ptg2T exp(2γ2Γg
√

T)


Proof. From Theorem 12 and the optional stopping theorem, we conclude that the following quantity
is a super-martingale:

M exp
t
:= Eθ∼π exp(γ

Mmin(H,t), θ

−h(min(H, t)))1(Bmin(H,t))

Therefore, we have:

Eθ∼π exp(γ

Mmin(H,T ), θ

−h(T))1(BT ) ≤M exp
T
≤EM exp
0
= 1

Combining Theorem 13 and Equation (32), we conclude that the following inequality holds with
probability at-least 1 −δ when conditioned on BT :

sup
ρ∈M1(Rd)
Eθ∼ργ

Mmin(T,H), θ

−KL (ρ||||||π) ≤h(T) + log(
1
δP(BT ))

In the RHS of the inequality above, we replace the supremum over M1 with the supremum over
the set of all probability distributions {N(αξ, I) such that ξ ∈Sd−1, α ≥0}.
We note that
KL (N(αξ, I)||||||π) = α2

2 to conclude that the following inequality holds with probability at-least 1−δ
when conditioned on BT :

sup
α>0
γα∥Mmin(H,T )∥−α2

2 ≤h(T) + log(
1
δP(BT ))

That is:

∥Mmin(H,T )∥≤

s

2h(T) + 2 log(
1
δP(BT ))

γ2

Now, note that by definition,

h(T)

T
= 1

T

T
X

t=1
log

1 + γ2

2 qt exp(γ2Γ2) + γ4ptg2T exp(2γ2Γg
√

T)


≤γ2

2 ¯q exp(γ2Γ2) + γ4¯pg2T exp(2γ2Γg
√

T)
(40)

Therefore, whenever: γ ≤min

1
Γ,
1
2√

Γg
√

T


, we note with probability at-least 1 −δ conditioned

on the event BT :

∥Mmin(H,T )∥≲

r

T ¯q + γ2¯pg2T 2 +
1
γ2 log

1
δP(BT )


We therefore state the following theorem:
Theorem 14. Suppose δ, δ1 ∈(0, 1

2). If Mt satisfies (g, T, δ) uniform concentration for some δ < 1

2.
Then Mt also satisfies (g′, T, δ + δ1) concentration, where

(g′)2 = C

"

¯q + γ2¯pg2T +
log( 1

δ1 )

γ2T

#

for any γ ≤min

1
Γ,
1
2√

Γg
√

T


.

53


---Page Break---
Additionally, suppose g ≥c0 Γ
√

T for some fixed constant c0 > 0, then we have for some constant
Citer(c0):

(g′)2 = Citer(c0)[¯q + g

¯p
√

T
Γ
+
Γ
√

T log( 1

δ1 )

]

Proof. Since δ ≤
1
2, we conclude that P(BT ) ≥
1
2. Given that Mt satisfies (g, T, δ) uniform
concentration. We conclude from the discussion above that for some universal constant C and any

γ ≤min

1
Γ,
1
2√

Γg
√

T


, we have:

sup
H
P(∥Mmin(H,T )∥2 ≥C[T ¯q + γ2¯pg2T 2 +
1
γ2 log

1
δ1


]
BT ) ≤δ1

Picking H to be the stopping time given by H = inf{t ≥0 : ∥Mt∥2 ≥C[T ¯q + γ2¯pg2T 2 +

1
γ2 log

1
δ1


]} where C is the same constant as in the equation above, we conclude:

P(sup
t≤T
∥Mt∥2 ≥C[T ¯q + γ2¯pg2T 2 +
1
γ2 log

1
δ1


]
BT ) ≤δ1

Only in this proof, call the event G := {supt≤T ∥Mt∥2 ≥C[T ¯q + γ2¯pg2T 2 +
1
γ2 log

1
δ1


]}. We
have:

P(G) = P(G ∩BT ) + P(G ∩B∁
T ) ≤P(G|BT ) + P(B∁
T ) ≤δ1 + δ

Whenever g ≥c0 Γ
√

T , we can pick λ =
c1(c0)
√

Γg
√

T and conclude the result.

We now state consider Lemma 11 from [2].

Lemma 18. Suppose α, β ≤0 with α + β > 0. Consider the function f : R+ →R+ given by

f(u) = α + β√u. Then, f has the unique fixed point: u∗:=

β+√

β2+4α
2

2
. For t ∈N, denoting

f (t) to be the t fold composition of f with itself, we have for any u ∈R+:

|f (t)(u) −u∗| ≤β(2−
1
2t−1 )|u −u∗|
1
2t .

We are now ready to prove the main theorem 9

Proof of Theorem 9. It is sufficient to show that there exists K = log Θ(log(ΓTd log( 1

δ )))) such

that Mt obeys (g, T, δ) uniform concentration where g = C max( Γ
√

T , ¯q + ¯p
√

T
Γ
+
Γ
√

T log( K

δ ))

Let K ∈N be any fixed integer. By Theorem 6, we conclude that the martingale Mt is (g0( δ

K ), T, δ

K )
uniformly concentrated. Fix some c0 > 0 and Citer(c0) be as in Theorem 14.

Define the sequence gi :=
p

Citer(c0)¯q +
p

Citer(c0)gi−1G where G = ¯p
√

T
Γ
+
Γ
√

T log( K

δ ).

If g0 ≤c0 Γ
√

T , then the statement of the theorem follows. Suppose there exists K1 ≤K −1

such that gK1 ≤c0Γ
√

T and suppose that it is the first such integer. If K1 = 0, the statement of the

theorem follows from (g0( δ

K ), T, δ

K ) uniform concentration of Mt. Suppose 1 ≤K1 ≤K −1.
Then, min(g0, . . . , gK1−1) ≥c0 Γ
√

T . Then, by Theorem 14, the fact that √x + y ≤√x + √y

and induction, we conclude that Mt obeys (gi, T, (i+1)δ

K
) for every i ≤K1. Thus we conclude the
statement of the theorem.

54


---Page Break---
Suppose such a K1 does not exist. Then, min(g0, . . . , gK−1) ≥c0 Γ
√

T . Then, by Theorem 14, the

fact that √x + y ≤√x + √y and induction, we conclude that Mt obeys (gi, T, (i+1)δ

K
) for every
i ≤K −1. Therefore, it obeys (gK, T, δ) uniform concentration.

Consider the function f in Lemma 18 with α =
p

Citer(c0)¯q and β =
p

Citer(c0)G and let the
corresponding fixed point be denoted by g∗. It is easy to show that the fixed point g∗≲√¯q + G.
After K iterations, we must have:

gK ≤g∗+ (Citer(c0)G1−
1
2K )|g0 −g∗|
1
2K ≲g∗+ (G1−
1
2K )|g0|
1
2K

We can show that picking K = log Θ(log((1 +
√¯qT

Γ ) log d)), and the bound on Γ, we conclude the
result.

F.2
Proof of Theorem 10

Proof of Theorem 10. Recall that Σt := E[vtv⊺
t |Ft−1], νt := ∥Σt∥and Nt := ∥Mt∥2 −
Pt
s=1 ∥vt∥2. Note that νt ≤pt and Tr(Σt) ≤pt almost surely.

Let γ ∈R. Define hN(t) := Pt
s=1 log

1 + 4γ2psg2T exp(2|γ|Γg
√

T)

with empty sum denoting

0. We first show that N exp
t
= exp(γNt −hN(t))1(BT ) is a super martingale with respect to the
filtration Ft for 0 ≤t ≤T. For T ≥t > 1, we have:
E[exp(γNt)1(Bt)|Ft−1] = exp(γNt−1)1(Bt−1)E[exp(2γ⟨vt, Mt−1⟩)1(Bt)|Ft−1]

≤exp(γNt−1)1(Bt−1)E[

∞
X

k=0

1
k!2kγk⟨vt, Mt−1⟩k1(Bt−1)|Ft−1]

= exp(γNt−1)1(Bt−1)E[1(Bt−1) +

∞
X

k=2

1
k!2kγk⟨vt, Mt−1⟩k1(Bt−1)|Ft−1]

≤exp(γNt−1)1(Bt−1)E[1 +

∞
X

k=2

1
k!2k|γ|k⟨vt, Mt−1⟩2Γk−2∥Mt−1∥k−21(Bt−1)|Ft−1]

≤exp(γNt−1)1(Bt−1)E[1 + 4γ2⟨vt, Mt−1⟩2 exp(2|γ|Γ∥Mt−1∥)1(Bt−1)|Ft−1]

≤exp(γNt−1)1(Bt−1)E[1 + 4γ2νt∥Mt−1∥2 exp(2|γ|Γ∥Mt−1∥)1(Bt−1)|Ft−1]

≤exp(γNt−1)1(Bt−1)

1 + 4γ2νtg2T exp(2|γ|Γg
√

T)


= exp(γNt−1 + hN(t) −hN(t −1))1(Bt−1)
(41)

This shows that N exp
t
is a super-martingale. Using the fact that N exp
1
= 1 almost surely, the optional
stopping theorem and the Chernoff bound, we conclude that for any stopping time H, we have for
any α, γ > 0

P({Nmin(T,H) > α} ∩BT ) ≤E[exp(γNmin(T,H) −γα)1(BT )]

≤E[exp(γNmin(T,H) −γα)1(Bmin(T,H))]

≤E[N exp
min(T,H)] exp(hN(T) −γα)

≤exp(hN(T) −γα)
(42)

Taking γ =
1
2Γg
√

T allows us to conclude:

P({Nmin(T,H) > ΓCg
√

T log( 2

δ ) + CνgT 3/2

Γ
} ∩BT ) ≤δ

2

Let α = ΓCg
√

T log( 2

δ ) + CνgT 3/2

Γ
and take H to be the stopping time min(inft{t > 0 : Nt >
α}, T) where infimum of an empty set is taken to be infinity. We note that {supt≤T Nt > α} =
{Nmin(T,H) > α}. We thus conclude:

55


---Page Break---
P({sup
t≤T
Nt > ΓCg
√

T log( 2

δ ) + CνgT 3/2

Γ
} ∩BT ) ≤δ

2

Taking γ negative gives the analogous proof for Nt < −α.

F.3
Proof of Corollary 5

Proof. Consider the set S = {UP(t) : 0 ≤t ≤T}. The, |S| ≤log2(T) + 1. By Corollary 6, we
have for any t0 ∈S, the following is true with probability 1 −
δ
1+log2(T )

t0
X

s=1
∥vs∥2 ≤t0g2(t0,
δ
3(1 + log2(T)))

Therefore, by union bound of the above event over every t0 ∈S, we have with probability 1 −δ:

sup
t0∈S

t0
X

s=1
∥vs∥2 ≤t0g2(t0,
δ
3(1 + log2(T))) ≤0

Now, note that Pt
s=1 ∥vs∥2 ≤PUP(t)
s=1 ∥vs∥2 almost surely for every t ∈[T] since t ≤UP(t).
Therefore, we conclude that with probability at-least 1 −δ, the following holds for all t ∈[T]
simultaneously:

t
X

s=1
∥vs∥2 ≤g2 
UP(t),
δ
3(1+log2(T ))

UP(t)

Using the definition of g(, ) from Theorem 9, we conclude the result.

G
Applications to Streaming Heavy Tailed Statistical Estimation

G.1
Streaming Heavy Tailed Mean Estimation : Proof of Corollary 1

Proof. Recall that for this problem, Ξ = C, Eξ∼P [ξ] = m ∈C and Cov[ξ] ⪯Σ. Consider the
following quadratic loss function f : C →R:

f(x; ξ) = 1

2∥x −ξ∥2,
ξ ∼P

The associated population risk function F is given by

F(x) = 1

2 · Eξ∼P

∥x −ξ∥2
= F(x) = 1

2∥x −m∥2 + Tr(Covξ∼P [ξ])

Note that F is L-smooth and µ-strongly convex with L = µ = 1. Thus, κ = 1. Furthermore, m
is the unique minimizer of F. Hence, solving the streaming heavy tailed mean estimation problem
is equivalent to solving the SCO problem for F. To this end, we consider the following stochastic
gradient oracle:

g(x; ξ) = x −ξ

It is easy to see that Ey[g(x; ξ)] = ∇F(x), i.e., the stochastic gradient estimate is unbiased. The
associated stochastic gradient noise n(x; ξ) is given by

n(x; ξ) = ∇F(x) −∇fy(x) = y −m

We now note that

Σ(x) = E[n(x; ξ)n(x; ξ)T ] = E[(y −m)(y −m)T ] = Tr(Covξ∼P [ξ]) ⪯Σ

Hence, we note that the Bdd. 2nd Moment assumption is satisfied. Hence, the result follows by an
application of Theorem 1

56


---Page Break---
G.2
Streaming Heavy Tailed Linear Regression : Proof of Corollary 2

We use θ ∈C to denote the parameter of F. Recall from Section 5.2 that Ξ = Rd × R, and given a
target parameter θ∗∈C, P defines the following linear model:

x ∼Q, E[x] = 0, E[xxT ] = Σ ≻0;
y = ⟨x, θ∗⟩+ ϵ, E[ϵ|x] = 0, E[ϵ2|x] ≤σ2

In addition, we make the following bounded 4th moment asumption on the covariates x

E[⟨x, v⟩4] ≤C4(E[⟨x, v⟩2])2
∀v ∈Rd

for some numerical constant C4 ≥1. Recall that the sample loss function is given by:

f(θ; x, y) = 1

2 (⟨θ, x⟩−y)2 = 1

2 (⟨θ −θ∗, x⟩−ϵ)2

Using the fact that E[ϵ|x] = 0, E[x] = 0 and E[xxT ] = Σ

F(θ) = 1

2(θ −θ∗)T E[xxT ](θ −θ∗) + E[ϵ2]

= 1

2(θ −θ∗)T Σ(θ −θ∗) + E[ϵ2]

We note that E[ϵ2] ≤σ2 as per our assumption hence F is well defined. Furthermore.

∇F(θ) = Σ(θ −θ∗)

∇2F(θ) = Σ

Thus, the population risk F is L-smooth and µ-strongly convex with L = ∥Σ∥2 and µ = λmin(Σ),
i.e., κ =
∥Σ∥2
λmin(Σ). Furthermore, the unique minimizer of F is θ∗. Hence, κ =
∥Σ∥2
λmin(Σ) the linear
regression task of estimating θ∗is equivalent to solving SCO for the above objective.

The associated stochastic gradient oracle g(θ; x, y) at any θ ∈C is given by:

g(θ; x, y) = ∇f(θ; x, y) = x(⟨θ, x⟩−y) = x (⟨θ −θ∗, x⟩−ϵ)

= xxT (θ −θ∗) −xϵ

We first show that g(θ; x, y)) is indeed an unbiased estimate of ∇F(θ)

E[g(θ; x, y)] = E[xxT ](θ −θ∗) −E[xE[ϵ|x]] = Σ(θ −θ∗) = ∇F(θ)

The associated stochastic gradient noise n(θ; x, y)(θ) is given by

n(θ; x, y)(θ) = g(θ; x, y)(θ) −∇F(x)

=
 
xxT −Σ

(θ −θ∗) −xϵ

Σ(θ) = E[n(θ; x, y)n(θ; x, y)]. For convenience, we use M = xxT −Σ and dθ = θ −θ∗and note
that M is symmetric. It follows that:

Σ(θ) = E
h
(Mdθ −xϵ) (Mdθ −xϵ)T i

= E

MdθdT
θ M

+ E

xxT · E

ϵ2|x

−E[xdT
θ M · E[ϵ|x]] −E[MdθxT · E[ϵ|x]]

⪯E

MdθdT
θ M

+ σ2Σ

where we use the fact that E[ϵ|x] = 0, E[ϵ2|x] ≤σ2 and E[xxT ] = Σ.

We shall now upper bound ∥Σ(θ)∥2. To do so, we define A(θ) = E

MdθdT
θ M

and note that A(θ)
is a PSD matrix since for any v ∈Rd, vT A(θ)v = E

(vT Mdθ)2
≥0. Without loss of generality,

57


---Page Break---
we assume θ ̸= θ∗and observe that

sup
∥v∥=1
E[vT A(θ)v] = sup
∥v∥=1
E[⟨dθ, Mv⟩2]

= ∥dθ∥2 sup
∥v∥=1
E[
D
dθ
∥dθ∥, Mv
E2
]

≤∥dθ∥2
sup
∥v∥=1,∥w∥=1
E[⟨w, Mv⟩2]

= ∥dθ∥2
sup
∥v∥=1,∥w∥=1
E[
 
wT  
xxT −Σ

v
2]

≤∥dθ∥2
sup
∥v∥=1,∥w∥=1
E
h 
⟨w, x⟩· ⟨v, x⟩−wT Σv
2i

≤∥dθ∥2
sup
∥v∥=1,∥w∥=1
2
 
wT Σv
2 + 2E
h
⟨w, x⟩2 ⟨v, x⟩2i

≤2∥dθ∥2
 

∥Σ∥2
2 +
sup
∥v∥=1,∥w∥=1

q

E[⟨w, x⟩4]
q

E[⟨v, x⟩4]

!

≤2∥dθ∥2
 

∥Σ∥2
2 + C4
sup
∥v∥=1,∥w∥=1
E[⟨w, x⟩2] · E[⟨v, x⟩2]

!

≤2∥dθ∥2
 

∥Σ∥2
2 + C4 sup
∥w∥=1
wT Σw · sup
∥v∥=1
vT Σv

!

≤∥dθ∥2 · 2∥Σ∥2(C4 + 1)

where we use the fourth moment assumption on the covariates in the eighth step. Note that the above
bound also holds when θ = θ∗since in that case A(θ) = 0 and dθ = 0. It follows that

∥Σ(θ)∥≤∥A(θ)∥+ σ2∥Σ∥

≤2(C4 + 1)∥Σ∥2∥θ −θ∗∥2 + σ2∥Σ∥

We shall now derive an upper bound for Tr(Σ(θ)) as follows:

Tr(Σ(θ)) = E[∥n(θ; x, y)∥2]

= E

∥Mdθ −xϵ∥2

= E[∥Mdθ∥2] −2E[⟨Mdθ, x⟩E[ϵ|x]] + E[∥x∥2E[ϵ2|x]]

≤E[∥Mdθ∥2] + σ2Tr(Σ)

We now control E[∥Mdθ∥2]. Note that E[∥Mdθ∥2] = 0 if θ = θ∗so we shall now consider the case
when θ ̸= θ∗. To this end, let e1, . . . , ed be an orthonormal basis of Rd such that e1 =
dθ
∥dθ∥.

For the remainder of the proof, we use Σij to denote Σij = eT
i Σej where i, j ∈[d], which implies that
Tr(Σ) = Pd
i=1 Σii. We also note that for any two symmetric matrices B, C, (B−C)2 ⪯2B2+2C2.

58


---Page Break---
Hence,

E[∥Mdθ∥2] = ∥dθ∥2E

∥Me1∥2

= ∥dθ∥2E

eT
1 (Σ −xxT )2e1


≤2∥dθ∥2E

eT
1
 
Σ2 + (xxT )2
e1


≤2∥dθ∥2 
eT
1 Σ2e1 + E
h
⟨e1, x⟩2 ∥x∥2i

≤2∥dθ∥2
 

∥Σ2∥+ E

"

⟨e1, x⟩2
d
X

i=1
⟨ei, x⟩2
#!

≤2∥dθ∥2
 

∥Σ∥2 + E
h
⟨e1, x⟩4i
+

d
X

i=2
E[⟨e1, x⟩2 ⟨ei, x⟩2]

!

≤2∥dθ∥2
 

∥Σ∥2 + E
h
⟨e1, x⟩4i
+

d
X

i=2

r

E
h
⟨e1, x⟩4i
E
h
⟨ei, x⟩4i!

≤2∥dθ∥2
 

∥Σ∥2 + C4E
h
⟨e1, x⟩2i2
+ C4

d
X

i=2
E
h
⟨e1, x⟩2i
E
h
⟨ei, x⟩2i!

≤2∥dθ∥2
 

∥Σ∥2 + C4

d
X

i=1
E
h
⟨e1, x⟩2i
E
h
⟨ei, x⟩2i!

≤2∥dθ∥2
 

∥Σ∥2 + C4(eT
1 Σe1)

d
X

i=1
(eT
i Σei)

!

≤2∥dθ∥2
 

∥Σ∥2 + C4(eT
1 Σe1)

d
X

i=1
Σii

!

≤2∥dθ∥2 (∥Σ∥Tr(Σ) + C4∥Σ∥Tr(Σ))

≤2(C4 + 1)∥Σ∥2Tr(Σ)∥dθ∥2

Clearly, the above bound holds even when θ = θ∗. Hence, we infer that

Tr(Σ(θ)) ≤2(C4 + 1)∥Σ∥2Tr(Σ)∥θ −θ∗∥2 + σ2Tr(Σ)

From these bounds, we can conclude the following

∥Σ(θ)∥≤2(C4 + 1)∥Σ∥2
2∥θ −θ∗∥2 + σ2∥Σ∥

Tr(Σ(θ)) ≤Tr(Σ)

∥Σ∥2


2(C4 + 1)∥Σ∥2
2∥θ −θ∗∥2 + σ2∥Σ∥


Thus, the stochastic gradient oracle satisfies Assumption QG 2nd Moment with α = 2(C4 + 1)∥Σ∥2
2,
β = σ2∥Σ∥and deff = Tr(Σ)/∥Σ∥. Hence, the result follows by an application of Theorem 2

G.3
Heavy Tailed Streaming Logistic Regression : Proof of Corollary 3

Recall from Section 5.4 that Ξ = Rd × {0, 1} and P denotes the following linear-logistic model:

x ∼Q, E[x] = 0, E[xxT ] ⪯Σ;
y ∼Bernoulli(ϕ(⟨θ∗, x⟩))

where ϕ(t) = (1 + e−t)−1. The covariates x are heavy tailed, with only bounded second moments.

The sample-level loss is given by the negative log likelihood of y|x as follows:

f(θ; x, y) = ln(1 + exp(⟨x, θ⟩)) −y ⟨x, θ⟩

The associated population loss and stochastic gradient oracle is given by

F(θ) = Ex,y∼P [ln(1 + exp(⟨x, θ⟩)) −y ⟨x, θ⟩]
g(θ; x, y) = ϕ(⟨x, θ⟩)x −yx

59


---Page Break---
We now compute the gradient and the Hessian of F

∇F(θ) = E

exp(⟨x, θ⟩)
1 + exp(⟨x, θ⟩) · x −ϕ(⟨x, θ∗⟩)x


= E [(ϕ(⟨x, θ⟩) −ϕ(⟨x, θ∗⟩)) x]

∇2F(θ) = E[ϕ′(⟨x, θ⟩)xxT ]

= E[ϕ(⟨x, θ⟩)(1 −ϕ(⟨x, θ⟩))xxT ]

Since 0 ≤ϕ(t) ≤1 for every t ∈R, we note that 0 ⪯∇2F(θ) ⪯E[xxT ] ⪯Σ (as E[x] = 0).
Hence, F is convex and L smooth with L = ∥Σ∥2. Furthermore, since ∇F(θ∗) = 0 and F is convex,
we conclude that θ∗is a minimizer of F.

It is easy to see that E [g(θ; x, y)] = E [(ϕ(⟨x, θ⟩) −ϕ(⟨x, θ∗⟩)) x] = ∇F(θ), i.e., the stochastic
gradient is unbiased. Let n(θ; x, y) denote the stochastic gradient noise, i.e.,:

n(θ; x, y) = g(θ; x, y) −∇F(θ)
= ϕ(⟨x, θ⟩)x −E [ϕ(⟨x, θ⟩)x] + E[ϕ(⟨x, θ∗⟩)x] −yx

We shall now control the stochastic gradient covariance Σ(θ) = E[n(θ; x, y)n(θ; x, y)T ]. To this
end, we define ax(θ) and cx,y(θ) as follows:

ax(θ) = ϕ(⟨x, θ⟩)x −E [ϕ(⟨x, θ⟩)x]
cx,y(θ) = E[ϕ(⟨x, θ∗⟩)x] −yx

We note that E [cx,y(θ)|x] = 0 and E[ax(θ)] = 0. Since nx,y(θ) = ax(θ) + bx,y(θ), it follows that:

Σ(θ) == E[n(θ; x, y)n(θ; x, y)T ] = E

ax(θ)ax(θ)T 
+ E

cx,y(θ)cx,y(θ)T 

We now control each of the terms in the RHS as follows:

E

ax(θ)ax(θ)T 
= E[ϕ(⟨x, θ⟩)2xxT ] −E [ϕ(⟨x, θ⟩)x] E [ϕ(⟨x, θ⟩)x]T

⪯E[ϕ(⟨x, θ⟩)2xxT ]

⪯E[xxT ] ⪯Σ

where we use the fact that ϕ(t) ≤1. Similarly,

E

cx,y(θ)cx,y(θ)T 
= E[y2xxT ] −E[ϕ(⟨x, θ∗⟩)x]E[ϕ(⟨x, θ∗⟩)x]T

⪯E[ϕ(⟨x, θ∗⟩)xxT ]

⪯E[xxT ] ⪯Σ

where we use the fact that E[y2|x] = ϕ(⟨x, θ∗⟩) ≤1. It follows that

Σ(θ) ⪯2Σ

Thus, the stochastic gradient oracle satisfies the Bdd. 2nd Moment assumption. Hence, the stochastic
gradient oracle satisfies the Bdd. 2nd Moment assumption. Thus, the following result, which is a
formal version of Corollary 3, is implied by Theorem 7

Corollary 7 (Heavy Tailed Logistic Regression). Under the stochastic subgradient oracle described
above, realized using N ≳ln(ln(d)) i.i.d samples from P, the average iterate of Algorithm 1, when
run under the parameter settings of Theorem 4 satisfies the following with probability at least 1 −δ:

F(ˆθN) −F(θ∗) ≲D1

v
u
u
tTr(Σ) +
p

∥Σ∥2
p

Tr(Σ) + ∥Σ∥2D1

ln(ln(N)/δ)

N
+ ∥Σ∥2D2
1
N

+ D2
1 ln(ln(N)/δ)

N

q

∥Σ∥2Tr(Σ) + ∥Σ∥3D2
1 + ∥Σ∥
5/4
2 D3
1 ln(ln(N)/δ)
3/2

N
3/2
 
Tr(Σ) + ∥Σ∥2
2D2
1
1/4

60


---Page Break---
G.4
Proof of Corollary 4

Recall from Section 5.4 that Ξ = Rd × R and given a target parameter θ∗∈C, P defines the
following linear model:

x ∼Q, E[x] = 0, E[xxT ] ⪯Σ;
y = ⟨x, θ∗⟩+ ϵ, Median(ϵ|x) = 0

We allow both the covariate x and target y to be heavy tailed, assuming only bounded second
moments for x. We do not assume any moment bounds on ϵ|x. The Least Absolute Deviation (LAD)
Regression problem involves estimating θ by solving SCO with the following sample loss

f(θ; x, y) = | ⟨x, θ⟩−y|

The associated population risk and one possible realization of a stochastic subgradient oracle is given
by:

F(θ) = E [| ⟨θ −θ∗, x⟩−ϵ|]
g(θ; x, y) = sgn(⟨θ, x⟩−y)x

where sgn(t) =
t
∥t∥for t ̸= 0 and sgn(0) = 0. We note that for every (x, y) ∈Rd × R, f(θ; x, y) is
a convex function in θ, and thus, the population risk F is a convex function, whose subgradient is
given by:

∂F(θ) = E [sgn (⟨θ −θ∗, x⟩−ϵ) x]

We now show that F is a Lipschitz function by bounding ∂F(θ) as follows:

∥∂F(θ)∥= ∥E [sgn (⟨θ −θ∗, x⟩−ϵ) x] ∥
≤E [|sgn (⟨θ −θ∗, x⟩−ϵ) | · ∥x∥]

≤
p

E [∥x∥2]

≤
p

Tr(Σ)

where the second step follows from Jensen’s inequality, the third step uses the fact that |sgn(t)| ≤1
and applies the Cauchy Schwarz inequality. Hence, F is G-Lipschitz with G =
p

Tr(Σ). We now
show that ∂F(θ∗) = 0 which would imply that θ∗is a minimizer of F (as F is convex)

∇F(θ∗) = E [sgn(ϵ)x] = E [x · E [sgn(ϵ)|x]] = 0

where we use the fact that E[sgn(ϵ)|x] = 0, because ϵ|x is a continuous random variable with zero
median.

For the stochastic gradient oracle described above, the associated stochastic gradient noise n(θ; x, y)
and its covariance Σ(θ) are given as follows:

n(θ; x, y) = sgn(⟨θ −θ∗, x⟩−ϵ)x −E [sgn(⟨θ −θ∗, x⟩−ϵ)x]

Σ(θ) = E

sgn(⟨θ −θ∗, x⟩−ϵ)2xxT 
−E [sgn(⟨θ −θ∗, x⟩−ϵ)x] E [sgn(⟨θ −θ∗, x⟩−ϵ)x]T

⪯E

sgn(⟨θ −θ∗, x⟩−ϵ)2xxT 

⪯E

xxT 
⪯Σ

Hence, the stochastic gradient oracle satisfies the Bdd. 2nd Moment assumption. Thus, the following
result, which is a formal version of Corollary 4, is implied by Theorem 8

Corollary 8 (Heavy Tailed LAD Regression).

F(ˆθN) −F(θ∗) ≲D1

s

Tr(Σ) +
p

∥Σ∥2Tr(Σ) ln(ln(N)/δ)

N
+ D1Tr(Σ) ln(ln(N)/δ)

N
p

∥Σ∥2
+ D1Tr(Σ)
5/4 ln(ln(N)/δ)
3/2

N
3/2∥Σ∥
3/4

NeurIPS Paper Checklist

1. Claims

61


---Page Break---
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We provide complete mathematical proofs of the claims.
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
Justification:
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
Answer: [Yes]
Justification:
Guidelines:

• The answer NA means that the paper does not include theoretical results.

62


---Page Break---
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
Answer: [NA]
Justification: The paper is purely theoretical.
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

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: Paper do note include experiments requiring code.
Guidelines:

63


---Page Break---
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

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [NA]

Justification:

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
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.

64


---Page Break---
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [NA]
Justification:
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
Justification:
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
Justification: The paper is purely theoretical and we do foresee any societal impact of this
work.
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

65


---Page Break---
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
Justification: purely theoretical work.
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
Justification:
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
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification:

66


---Page Break---
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
Justification:
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
Justification:
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

67


---Page Break---
