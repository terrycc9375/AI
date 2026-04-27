Log-concave Sampling from a Convex Body with a
Barrier: a Robust and Unified Dikin Walk

Yuzhou Gu
New York University
yuzhougu@nyu.edu

Nikki Lijing Kuang
University of California, San Diego
l1kuang@ucsd.edu

Yi-An Ma
University of California, San Diego
yianma@ucsd.edu

Zhao Song
Simons Institute for the Theory of Computing, UC Berkeley
magic.linuxkde@gmail.com

Lichen Zhang
MIT CSAIL
lichenz@csail.mit.edu

Abstract

We consider the problem of sampling from a d-dimensional log-concave distri-
bution π(θ) ∝exp(−f(θ)) for L-Lipschitz f, constrained to a convex body with
an efficiently computable self-concordant barrier function, contained in a ball of
radius R with a w-warm start.
We propose a robust sampling framework that computes spectral approximations to
the Hessian of the barrier functions in each iteration. We prove that for polytopes
that are described by n hyperplanes, sampling with the Lee-Sidford barrier function
mixes within eO((d2 + dL2R2) log(w/δ)) steps with a per step cost of eO(ndω−1),
where ω ≈2.37 is the fast matrix multiplication exponent. Compared to the prior
work of Mangoubi and Vishnoi, our approach gives faster mixing time as we are
able to design a generalized soft-threshold Dikin walk beyond log-barrier.
We further extend our result to show how to sample from a d-dimensional spec-
trahedron, the constrained set of a semidefinite program, specified by the set
{x ∈Rd : Pd
i=1 xiAi ⪰C} where A1, . . . , Ad, C are n × n real symmet-
ric matrices. We design a walk that mixes in eO((nd + dL2R2) log(w/δ)) steps
with a per iteration cost of eO(nω + n2d3ω−5). We improve the mixing time
bound of prior best Dikin walk due to Narayanan and Rakhlin that mixes in
eO((n2d3 + n2dL2R2) log(w/δ)) steps.

1
Introduction

Given a convex body, generate samples from the body according to structured densities is a fundamen-
tal problem in computer science and machine learning. It has extensive applications in constrained
convex optimization [Lovász and Vempala, 2006, Narayanan, 2016], differentially private learn-
ing [Wang et al., 2015, Lin et al., 2024] and online optimization [Narayanan and Rakhlin, 2017]. A
central theme in the theory of sampling is to leverage stochasticity and reduce per iteration costs
without having to proportionally increase the number of iterations. This theme has played out in
continuous optimization for both first and second order methods. For the first order methods, focus is
on reducing the variance of the gradient estimators [Johnson and Zhang, 2013, Shalev-Shwartz and
Zhang, 2013, Defazio et al., 2014]. For the second order methods, matrix sketching is often used to

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
reduce computation and storage related to the Hessian matrix [Lee et al., 2019, Jiang et al., 2020b,
2021, Song and Yu, 2021, Qin et al., 2023].

In Markov chain Monte Carlo (MCMC), much of the progress is on the first order methods. Non-
asymptotic analyses are performed for the stochastic gradient Langevin algorithms and their variance
reduced extensions [Dubey et al., 2016, Raginsky et al., 2017, Brosse et al., 2018, Chatterji et al.,
2018, Zou et al., 2018, Dalalyan and Karagulyan, 2019, Li et al., 2019, Ding and Li, 2021]. For
second order methods, there has been a paucity when it comes to applying the sketching techniques.

In this paper, we focus on sampling from a log-concave distribution constrained to a d-dimensional
convex body K that is described by n constraints, where second order information is proven essential
for capturing the geometry of the convex body and consequently for achieving fast convergence
rates [Narayanan, 2016, Narayanan and Rakhlin, 2017, Chen et al., 2018, Laddha et al., 2020,
Mangoubi and Vishnoi, 2023, 2024]. In particular, we associate the convex body with a self-
concordant barrier function [Nesterov and Nemirovskii, 1994, Renegar, 1988, Vaidya, 1989] and
utilize the Hessian matrix H(x) of the barrier function in the sampling algorithm. We then use the
Hessian matrix to propose samples and compute the probability to accept or reject the proposed
samples. We note that this problem involves subtleties beyond the scope of constrained optimization.
In continuous optimization, which focuses on finding the descent directions, unbiased estimators
with reasonable variance oftentimes suffice [Vaidya, 1989, Lee et al., 2015, Huang et al., 2022]. In
MCMC, on the other hand, the target probability distribution needs to be maintained along the entire
trajectory. This poses significant challenges to speed up the sampling algorithms. In the scenario of
uniform sampling over a polytope, Laddha et al. [2020] shows that for a simple logarithmic barrier1,
an unbiased estimator in fact suffices. However, more complicated barriers such as the Lee-Sidford
barrier [Lee and Sidford, 2014, 2019] can only be approximately computed2, an unbiased estimator
is extremely hard to be obtained. Moreover, for sampling from more sophisticated log-concave
distributions and convex bodies, a more general approach is needed.

We therefore propose to obtain a high precision estimator to the desired acceptance rate with an
improved per step running time and without sacrificing the rapid mixing time. In particular, we ask
the following question:

Can we significantly reduce per iteration cost, while preserving the convergence rate of the
log-concave sampling algorithms over various convex bodies?

We answer the above question in the affirmative. To this end, we present a slew of results regarding
log-concave sampling. For polytopes, we provide a walk that mixes in eO(d2 + dL2R2) steps3 with
per iteration cost eO(ndω−1). Prior state-of-the-art results are due to Mangoubi and Vishnoi [2023,
2024], for which their walks mix in eO(nd + dL2R2) with a per iteration cost4 eO(nnz(A) + d2). Our
walk mixes faster whenever n ≥d. Our result partially resolves an open problem in Mangoubi and
Vishnoi [2023] as they asked whether it’s possible to design a Dikin walk whose mixing time is only
depends on d and independent of L, R. We remove the L and R dependence on the dominating term
d2, hence for the case of sampling from uniform distribution (L = 0), we recover the state-of-art
result of Laddha et al. [2020] which mixes in eO(d2) steps with a per iteration cost eO(ndω−1).

We also consider sampling from convex bodies beyond polytopes. In semidefinite programming
(SDP), one often focuses on the dual program of an SDP, where the constraint set is defined as a
spectrahedron K = {x ∈Rd : Pd
i=1 xiAi ⪰C}5 for n×n symmetric matrices Ai and C [Jiang et al.,
2020a, Huang et al., 2022]. We propose a walk that samples from a log-concave distribution over a
spectrahedron K using the Hessian information of the log-barrier. The walk mixes in eO(nd+dL2R2)
steps. For log-barrier of an SDP, explicitly forming the Hessian matrix takes a prohibitively large

1Logarithmic barriers, or log-barriers have been extensively studied in the context of sampling and convex op-
timization. For a d-dimensional polytope with n constraints, it typically leads to a convergence rate polynomially
depends on n.
2Lee-Sidford barriers are the first polynomial-time computable barriers with complexity parameter depends
only logarithmic on n and in extension, the convergence rate.
3We use eO(·) to hide polylogarithmic dependence on n, d and δ where δ is the TV distance between the
target distribution and our Markov chain µ.
4We use nnz(A) to denote the number of nonzero entries in matrix A.
5We use A ⪰0 to denote A is a positive semidefinite matrix, and A ⪰B to denote A −B ⪰0.

2


---Page Break---
O(nωd + n2dω−1) time. We utilize our robust sampling framework to approximately compute this
Hessian via tensor-based sketching techniques, achieving a runtime of eO(nω + n2d3ω−5). As long
as n ≥d
3ω−4

ω−2 (which is a usual setting for SDP, where d ≪n), this provides a speedup.

1.1
Related works

In this work, we focus on Dikin walk [Kannan and Narayanan, 2012], a refined variant of ball
walk [Lovász and Simonovits, 1993]. Roughly speaking, ball walk progresses by moving to a random
point in the ball centered at the current point with the obvious downside that when the convex body is
flat, ball walk progresses slowly. Dikin walk overcomes this problem by instead moving to a random
point in a good ellipse centered at the current point, in an effort to better utilize the local geometry of
the convex body.

For sampling over polytopes, a number of Dikin walks use the ellipse induced by the log-barrier
functions [Kannan and Narayanan, 2012]. The work of Laddha et al. [2020] shows that uniform
sampling over a polytope can mix in O(νd) steps, where ν is the self-concordant parameter of the
barrier function6. Going beyond uniform distributions, Narayanan and Rakhlin [2017] proposes a
walk that samples from a log-concave distribution on a polytope in eO(ν2d) steps. This bound is
later improved by Mangoubi and Vishnoi [2023], in which they show that for logarithmic barrier, a
mixing time of eO(nd + dL2R2) is attainable, where L is the Lipschitz constant of f and R is the
radius of the ball that contains K. A more recent work by Kook and Vempala [2024] has obtained
better bound when the function f is α-relative strongly-convex and the density π is β-Lipschitz. They
manage to obtain a mixing bound of O(νdβ log(w/δ)). However, their algorithm requires stronger
assumptions on f and hence is incomparable to our result. In most previous works, the focus has
been on improving the mixing time, rather than on per iteration costs.

For any convex body, it is well-known that a universal barrier with ν = d exists [Nesterov and
Nemirovskii, 1989, Lee and Yue, 2021], however it is computationally hard to construct the Hessian
of the universal barrier. In short, the universal barrier requires to compute the volume of the polar set
of a high dimensional body, which is even hard to approximate deterministically [Furedi and Barany,
1986, Nesterov and Nemirovskii, 1994]. The seminal work of Lee and Sidford [Lee and Sidford,

2014] presents a nearly-universal barrier with ν = O(d log5 n) and the Hessian can be approximated
in O(ndω−1) time. The Lee-Sidford barrier was originally designed to solve linear programs in
O(
√

d) iterations, and it has been leveraged lately for Dikin walks with rapid mixing time. For
uniform sampling, Chen et al. [2018] gives an analysis with a walk that mixes in eO(d2.5) steps.
Subsequently, Laddha et al. [2020] improves the mixing to eO(d2) steps. In a pursuit to better leverage
the local geometry, Lee and Vempala [2017] proposes a walk relying on Riemannian metric that
mixes in eO(nd3/4) steps. The mixing rate is later improved to eO(nd2/3) via Riemannian Hamiltonian
Monte Carlo (RHMC) with log-barrier [Lee and Vempala, 2018] and eO(n1/3d4/3) with a mixed
Lee-Sidford and log-barrier [Gatmiry et al., 2024]. In this work, we show that log-concave sampling
can also leverage Lee-Sidford barrier to obtain a mixing time of eO(d2 + dL2R2). In comparison,
the hit-and-run algorithm [Lovász and Vempala, 2007] mixes in eO(d2R2/r2) steps where r is the
radius of the inscribed ball inside K. For the regime where L = O(1) and r = O(1), our walk mixes
strictly faster than that of hit-and-run.

We note that for sampling over more general convex bodies, a recent work [Chen and Eldan, 2022]
proves that for uniform sampling over an isotropic convex body, the mixing time bound is eO(d2/ψ2
d)
where ψd is the KLS constant [Kannan et al., 1995]. However, it is unclear how to generalize their
result to non-isotropic convex bodies or log-concave densities.

1.2
Our results

Our results concern log-concave sampling from polytopes and spectrahedra. For polytopes, we state
the result in its full generality.

6Technically, what they have shown is that if the Hessian is a ν-symmetric self-concordant barrier function,
then the walk mixes in O(νd) steps, and they subsequently prove for various barriers of interest, ν = ν.

3


---Page Break---
Theorem 1.1 (Robust sampling for log-concave distribution over polytopes). Let δ ∈(0, 1) and
R ≥1. Given a constraint matrix A ∈Rn×d with a vector b ∈Rn, let K := {x ∈Rd : Ax ≤b} be
the corresponding polytope. Suppose K is enclosed in a ball of radius R with non-empty interior.
Let f : K →R be an L-Lipschitz, convex function with an evaluation oracle. Finally, let π be the
distribution such that π ∝e−f.

Suppose we are given an initial point from K that is sampled from a w-warm start distribution7 with
respect to π for some w > 0, then there is an algorithm (Algorithm 1) that outputs a point from a
distribution µ where TV(µ, π) ≤δ.

Let g : K →R be a ν-self-concordant barrier function such that in time Cg, a spectral approximation
eHg of the Hessian of g denoted by Hg can be computed satisfying

(1 −ϵ) · Hg ⪯eHg ⪯(1 + ϵ) · Hg

for ϵ = Θ(1/d). Then, Algorithm 1 takes at most

eO((νd + dL2R2) log(w/δ))

Markov chain steps. It uses O(1) function evaluations and an extra Cg + dω time at each step.

Let us pause and make some remarks regarding Theorem 1.1. As long as the Hessian matrix can be
approximately generated with an error inversely depends on d, then our algorithm is guaranteed to
converge. Moreover, if the approximate Hessian can be generated more efficiently, then this directly
implies an improvement of our algorithm. On the mixing time side, Theorem 1.1 is nearly-optimal up
to polylogarithmic factors and the dependence on L and R. In particular, the νd mixing time bound
is also achieved by Laddha et al. [2020] in the case of uniform sampling. We instantiate the meta
result of Theorem 1.1 into the following theorem.

Theorem 1.2 (Robust sampling with nearly-universal barrier). Under the conditions of Theorem 1.1,
let g be the Lee-Sidford barrier with ν = O(d log5 n). Then, we have

Cg = eO(ndω−1).

The algorithm takes at most

eO((d2 + dL2R2) log(w/δ))

Markov chain steps.

The Lee-Sidford barrier [Lee and Sidford, 2014, 2019] is the first polynomial-time computable barrier
with a nearly-optimal self-concordance parameter. Several prior works [Chen et al., 2018, Laddha
et al., 2020] utilize this barrier for uniform sampling. For log-concave sampling, the walk of Kook
and Vempala [2024] requires strong-convexity-like assumption on f in order to attain an eO(d2)
mixing time. Our work is the first to obtain an eO(d2) mixing time for log-concave sampling over
polytopes, when L and R are small.

References
Mixing time
Per iteration cost
Lovász and Vempala [2006]
d2R2/r2
nd
Narayanan and Rakhlin [2017]
d5 + d3L2R2
ndω−1

Mangoubi and Vishnoi [2023]
nd + dL2R2
ndω−1

Mangoubi and Vishnoi [2024]
nd + dL2R2
nnz(A) + d2

This work
d2 + dL2R2
ndω−1

Table 1: Comparison of algorithms for sampling from a log-concave density e−f over a d-dimensional
polytope with n constraints, where f is L-Lipschitz. We omit eO(·) to mixing time and per iteration
cost. We assume the evaluation of f can be done in unit time. The first row is the hit-and-run walk.
Our algorithm has the fastest mixing time among all Dikin walks, and for L = O(1), r = O(1), our
walk mixes faster than hit-and-run.

7We say a distribution ρ is a w-warm start of with respect to a distribution π if supx∈K
ρ(x)
π(x) ≤w.

4


---Page Break---
Our next result is closely related to semidefinite programming. Given the the set K = {x ∈Rd :
Pd
i=1 xiAi ⪰C} for symmetric n × n matrices A1, . . . , Ad, C, we consider sampling from a
log-concave distribution over K. We use the popular log-barrier for K studied in Nesterov and
Nemirovskii [1994], Jiang et al. [2020a], Huang et al. [2022]. To further utilize our robust sampling
framework, we develop a novel approach for generating a spectral approximation of the Hessian.
Compared to Jiang et al. [2020a], Huang et al. [2022] in which the Hessian is maintained in a very
careful way in conjunction with the interior point method, our spectral approximation does not require
to control the changes over iterations and is highly efficient when n ≫d, which is a popular regime
for semidefinite programs.

Theorem 1.3 (Robust sampling for log-concave distribution over spectrahedra). Let δ ∈(0, 1) and
R ≥1. Given a collection of symmetric matrices A1, . . . , Ad ∈Rn×n and a target symmetric matrix
C ∈Rn×n, let K := {x ∈Rd : Pd
i=1 xiAi ⪰C} be the corresponding spectrahedron. Suppose K
is enclosed in a ball of radius R with non-empty interior. Let f : K →R be an L-Lipschitz, convex
function with an evaluation oracle. Finally, let π be the distribution such that π ∝e−f.

Suppose we are given an initial point from K that is sampled from a w-warm start distribution with
respect to π for some w > 0, then there is an algorithm (Algorithm 1) that outputs a point from a
distribution µ where TV(µ, π) ≤δ.

Let g : K →R be the logarithmic barrier function with ν = n. There is an algorithm that uses

eO((nd + dL2R2) log(w/δ))

Markov chain steps. In each step it uses O(1) function evaluations and an extra cost of

eO(nω + n2d3ω−5).

Prior work due to Narayanan and Rakhlin [2017] provides a walk that mixes in eO(n2d3 + n2dL2R2)
steps and each step could be implemented in O(dnω + n2dω−1) time, our algorithm improves upon
both mixing and per iteration cost. On the front of per iteration cost, computing the Hessian of the
log-barrier of K exactly takes time O(dnω + n2dω−1), for the regime where n ≫d the dominating
term is dnω. Note that whenever n ≫d
3ω−4

ω−2 , our algorithm is more efficient than that of exact
computation. This is a common regime in semidefinite program in which the number of constraints d
is small compared to the size of the primal solution.

References
Mixing time
Per iteration cost
Lovász and Vempala [2006]
d2R2/r2
nω + n2d
Narayanan and Rakhlin [2017]
n2d3 + n2dL2R2
nωd + n2dω−1

This work
nd + dL2R2
nω + n2d3ω−5

Table 2: Comparison of algorithms for sampling from a log-concave density e−f over a d-dimensional
spectrahedron with n × n symmetric matrix constraints, where f is L-Lipschitz. We omit eO(·) to
mixing time and per iteration cost. We assume evaluation of f can be done in unit time. The first
row is the hit-and-run walk. Our algorithm mixes faster than Narayanan and Rakhlin [2017] in all
parameter regimes, and for L = O(1), r = O(1), d2R2 ≥n, our walk mixes faster than hit-and-run.

2
Technical Overview

Before diving into our techniques, we first illustrate an informal version of our algorithm. The
algorithm itself is a variant of the Dikin walk [Kannan and Narayanan, 2012] called the soft-threshold
Dikin walk [Mangoubi and Vishnoi, 2023]. Starting with a w-warm start point, the algorithm
approximately computes the Hessian of the barrier function, then proposes a new point z from
a Gaussian with mean at the current point x and variance eΦ(x)−1, where eΦ is a multiple of the
approximate Hessian plus a multiple of the identity. The proposal is then accepted via a Metropolis
filter. Note that compared to the standard Dikin walk framework, our algorithm crucially allows the
Hessian to be approximated with a good precision. This means any improvements on generating
spectral approximation leads to runtime improvement of our algorithm in a black-box fashion.

5


---Page Break---
Algorithm 1 Sampler for log-concave distribution π ∝exp(−f) over convex body with a barrier
function g. A is the description of the convex set, function f is an L-Lipschitz concave function and
g is a ν-self-concordant barrier function, x0 ∈K is a w-warm start point, and δ is the total variance
distance between our Markov chain and the target distribution. Let K ⊆Rd enclosed by a ball of
radius R.

1: procedure SAMPLINGCONVEXBODY(A, f, g, x0, δ)
2:
Initialize x to be a point in K
3:
T ←eO((νd + dL2R2) log(w/δ))
4:
for t = 1 →T do
5:
Sample a point ξ ∼N(0, Id)
6:
// H(x) is the exact Hessian matrix at x, we never explicitly form it
7:
Compute eH(x) where (1 −1/d)H(x) ⪯eH(x) ⪯(1 + 1/d)H(x)
8:
eΦ(x) = d · eH(x) + dL2 · Id
9:
z ←x + eΦ(x)−1/2ξ
10:
if z ∈Int(K) then
11:
// H(z) is the exact Hessian matrix at z
12:
Compute bH(z) such that (1 −1/d)H(z) ⪯bH(z) ⪯(1 + 1/d)H(z)
13:
bΦ(z) = d · bH(z) + dL2 · Id

14:
τ ←
exp(−f(z))·(det(bΦ(z)))1/2·exp(−0.5∥x−z∥2
b
Φ(z))

exp(−f(x))·(det(eΦ(x)))1/2·exp(−0.5∥x−z∥2
e
Φ(x))
15:
// Our walk is lazy, i.e., it only moves to a new point with half probability
16:
accept x ←z with probability 0.5 · min{τ, 1}
17:
else
18:
reject z
19:
end if
20:
end for
21: end procedure

2.1
Soft-threshold Dikin walk for log-concave sampling via a barrier-based argument

For uniform sampling from structured convex bodies such as polytopes, Kannan and Narayanan
[2012] introduce a refined ball walk that utilizes a self-concordant barrier function associated with
the convex body. To extend the Dikin walk framework for log-concave sampling, Narayanan and
Rakhlin [2017] adds an additional coefficient exp(−f(x))

exp(−f(z)) to the acceptance probability of Metropolis

filter. This achieves a sub-optimal mixing time bound of eO(ν2d). By introducing a soft-threshold
regularizer to the Hessian, Mangoubi and Vishnoi [2023] proves that for log-barrier function, it is
possible to approach the optimality with an eO(nd + dL2R2) mixing time. Unfortunately, the method
used by Mangoubi and Vishnoi [2023] is specialized to log-barrier, making it hard to generalize to
other barriers. They show that under proper scaling, the true Hessian H(x) under a sequence of
polytopes converges to the regularized Hessian under the given polytope in the limit. They define the
j-th polytope as

Aj =





A
Id
...
Id



, bj =





b
j · 1d
...
j · 1d





the number of copies of Id and j · 1d is a function of j. While this polytope-based method
works well for log-barrier, it functions poorly for more intricate barriers, such as volumetric
barrier and Lee-Sidford barrier. Take volumetric barrier as an example, recall that Hvol(x) =
A⊤Σ(S(x)−1A)S(x)−2A where Σ(S(x)−1A) is the statistical leverage score [Drineas et al., 2006,
Spielman and Srivastava, 2011]. Leverage score is a numerical quantity that measures how important
a row of a matrix is, compared to other rows. This means that, if one duplicates a row infinitely many
times, the row will be assigned a score of 0. More concretely, suppose we are given an n × d matrix
with n identical rows, then the leverage score of each row is d

n. Taking n →∞, it’s easy to see that
all rows have score 0, and the Hessian of volumetric barrier will zero out the rows contributed to the

6


---Page Break---
infinitely many copies of identities. Thus, for volumetric barrier, extra constraints of the polytope
sequence vanish. On the other hand, the Mangoubi and Vishnoi [2023] construction heavily relies on
the infinitely many occurrences of Id for it to converge to the regularized Hessian.

To circumvent this issue and provide a simpler argument, we develop a proof strategy that is barrier-
based, instead of polytope-based. We example a class of popular barrier functions for polytopes
that can be classified as weighted log-barrier functions [Vaidya, 1989]. Roughly speaking, these
functions take in the form of g(x) = −Pn
i=1 wi log(S(x)i), with w = 1n being the log-barrier,
w = σ(S(x)−1A) being the volumetric barrier [Vaidya, 1989] and if we choose w to be a proper
power of the Lewis weights, then the barrier is precisely the Lee-Sidford barrier [Lee and Sidford,
2019, Laddha et al., 2020]. These weights are stable, meaning that if two points are close in certain
local norm induced by proper ellipses given by these weights, then their weights are also close [Lee
and Sidford, 2019]. This property proves to be particularly useful when proving the mixing rate of
our Dikin walk. In addition, we carve out three sufficient conditions for a weighed log-barrier barrier
function to work for log-concave distribution, and has eO(νd) mixing rate:

• ν-symmetry. The Hessian H(x) is ν-symmetry, that is, for any x ∈K, Ex(1) ⊆K ∩
(2x −K) ⊆Ex(
√

ν) where Ex(r) = {y ∈Rd : (y −x)⊤H(x)(y −x) ≤r2}. We show
that ν = ν for log-barrier, volumetric barrier and Lee-Sidford barrier for polytopes, and
log-barrier for spectrahedra.

• Bounded local norm. The variance term ∥H(x)−1/2 · ∇log det(H(x))∥2
2 ≤d poly log n.
We show this condition holds for all barriers of interest in this paper.
• Convexity of regularized barrier. The function F(x) = log det(H(x) + Id) is convex at
x for any x ∈K. We note that for log-barrier, volumetric barrier and Lee-Sidford barrier, it
has been shown that log det(H(x)) is indeed convex at x. We further prove that this still
holds when a copy of identity is blended in.

We want to highlight that our approach is more robust and generic than that of Mangoubi and Vishnoi
[2023], as it solely depends on the structure of the Hessian matrix. Our argument should be treated
as a generalization of Sachdeva and Vishnoi [2016], in which they utilize the bounded local norm
and convexity of log det(H(x)) for log-barrier, together with the Gaussianity to conclude that the
difference between log det are small. We go beyond log-barrier for polytopes and uniform sampling.

2.2
Approximation preserves the mixing time

The core premise of our algorithm is we allow the Hessian to be approximated, so that each step
can be implemented efficiently. This poses significant challenges in proving mixing, as even if
we have a good approximation of one-step in the Markov chain, the approximate chain does not
necessarily mixes as fast as the original chain. For example, if every step is approximated within
ϵ-TV distance to the original Markov chain, then in T steps, the TV distance between the resulting
distribution under approximate walk and the distribution under original walk could be as large as
Tϵ. This means we have to take ϵ very small (O(T −1
mix)) to guarantee same convergence property
as original chain, which is unacceptable. A related issue is that the stationary distribution under
approximate walk may not be the same as the target distribution. Therefore we need a way to control
the mixing properties of the approximate walk. We resolve this problem by exactly computing the
acceptance probability under approximate walk. That is, after proposing the next step z from x, eΦ(x),

we sample bΦ(z), and accept with probability min{1, π(z) b
Gz(x)
π(x) e
Gx(z)} (where eGx(z) is the probability of
proposing z starting from x under the approximate walk). Recall that in the exact walk, we accept
with probability min{1, π(z)Gz(x)

π(x)Gx(z)} (where Gx(z) is the probability of proposing z stating from x
under the exact walk). If we use this acceptability for the approximate walk, then we only have
one-step approximation guarantee, and will suffer from the problem mentioned previously.

By using this modified acceptance probability, we can prove reversibility of the approximate walk,
and that stationary distribution of the approximate walk is indeed our target distribution. For the
mixing rate, we bound the conductance of the approximate walk, which requires us to prove the
following properties:

(1) The proposed step is accepted with decent probability.

7


---Page Break---
(2) Starting from two points moderately close to each other (i.e., ∥x −z∥Φ(x) ≤ϵ), the
approximate steps taken Px and Pz are close in TV distance (i.e., TV(Px, Pz) ≤1 −ϵ′) for
some absolute constants ϵ, ϵ′ > 0.

We are able to prove these properties by comparison with the original chain. That is, assuming the
original chain satisfies properties (1)(2), then the approximate chain also satisfies the same properties,
as long as the approximation is good enough. This enables us to prove these key properties in a
hassle-free way. To prove these comparison results, we need to prove, for example, eGx(z) ≈Gx(z).
Because both Gx(z) and eGx(z) have nice factorizations

Gx(z) ∝det(Φ(x))1/2 exp(−1

2∥x −z∥2
Φ(x)),

eGx(z) ∝det(eΦ(x))1/2 exp(−1

2∥x −z∥2
eΦ(x)),

we only need to prove det(Φ(x)) ≈det(eΦ(x)) and exp(−1

2∥x −z∥2
Φ(x)) ≈exp(−1

2∥x −z∥2
eΦ(x))

separately. Both of these can be handled via Φ(x) ≈eΦ(x), using our approximation procedure.

2.3
Approximation of Lewis weights for rapid mixing

Given our robust and generic Dikin walk framework, we realize it with the Lee-Sidford barrier whose
complexity ν = d log5 n. This ensures the walk mixes in eO(d2 + dL2R2) steps, improving upon the
log-barrier-based walk of Mangoubi and Vishnoi [2023, 2024] that mixes in eO(nd + dL2R2) steps,
as the log-barrier has complexity ν = n. To implement the Lee-Sidford barrier, it is imperative to
compute the ℓp (for p > 0) Lewis weights, which is defined as the following convex program:

min
M⪰0 −log det M, subject to

n
X

i=1
(a⊤
i Mai)p/2 ≤d,
(1)

and the ℓp Lewis weights is a vector wp ∈Rd where (wp)i = a⊤
i M∗ai and M∗is the optimum of
Program (1). Solving the program exactly is not efficient, and a long line of works [Cohen and Peng,
2015, Lee and Sidford, 2014, 2019, Cohen et al., 2019, Fazel et al., 2022, Jambulapati et al., 2022]
provide fast algorithms that approximate all weights up to (1 ± ϵ)-factor. The Hessian of Lee-Sidford
barrier has the form (up to scaling) HLewis(x) = A⊤S(x)−1Wp(S(x)−1A)1−2/pS(x)−1A where
Wp(S(x)−1A) is the diagonal matrix of ℓp Lewis weights with respect to S(x)−1A. When clear from
context, we use Wp as a shorthand for Wp(S(x)−1A). As Wp could be approximated in eO(ndω−1)
time, we could then perform subsequent operations exactly to form the approximate Hessian. We
instead provide a spectral approximation procedure that runs in eO(nnz(A) + Tmat(d, d3, d)) given
the approximate Lewis weights where Tmat(a, b, c) is the time of multiplying an a × b matrix with
a b × c matrix and Tmat(d, d3, d) ≈d4.2. This is important as even though it doesn’t lead to direct
runtime improvement, if approximate Lewis weights can be computed in eO(nnz(A) + poly(d))
time, then we obtain a per iteration cost upgrade in a black-box fashion. Our approach is based on
spectral sparsification via random sampling: given a matrix B ∈Rn×d, one can construct a matrix
eB ∈Rs×d consists of re-scaled rows of B, such that (1 −ϵ) · B⊤B ⪯eB⊤eB ⪯(1 + ϵ) · B⊤B. If
we sample according to the leverage scores [Drineas et al., 2006, Spielman and Srivastava, 2011], one
can set s = O(ϵ−2d log d). In our case, we set B = W 1/2−1/p
p
S(x)−1A, and perform the sampling.
However, computing leverage score exactly is as expensive as forming the Hessian exactly, which
requires O(ndω−1) time. To speed up the crucial step, we use randomized sketching technique
to reduce the row count from n to poly(d), then approximately estimate all leverage scores with
this small matrix. One can either use the sparse embedding matrix [Nelson and Nguyên, 2013]
to approximate these scores in time O(nnz(A) log n + Tmat(d, ϵ−2d, d)), or use the subsampled
randomized Hadamard transform [Lu et al., 2013] in time O(nd log n + Tmat(d, ϵ−2d, d)). We
remark that approximating leverage scores for subsampling has many applications in numerical linear
algebra, such as low rank approximation [Boutsidis et al., 2016, Song et al., 2017, 2019]. Utilizing
this framework, we provide an algorithm that approximates the Hessian of Lee-Sidford barrier in
time eO(ndω−1) + eO(nnz(A) + Tmat(d, d3, d)), as advertised.

8


---Page Break---
2.4
Approximation of log-barrier Hessian over a spectrahedron

For a spectrahedron induced by the dual semidefinite program, it is also natural to define its log-barrier
with its corresponding Hessian being

Hlog(x) = A(S(x)−1 ⊗S(x)−1)A⊤,

where A ∈Rd×n2 with the i-th row being vec(Ai) for n × n symmetric matrix Ai, and S(x) =
Pd
i=1 xiAi −C where C is also an n×n symmetric matrix. Nesterov and Nemirovskii [1994] shows
that by smartly arranging the organization of Hlog(x), it can be computed exactly in time O(dnω +
dω−1n2). Beyond exact computation, Jiang et al. [2020a], Huang et al. [2022] present algorithms that
approximately maintain the Hessian matrix under low-rank updates. These maintenance approaches
are crafted towards solving semidefinite programs in which the trajectory can be carefully controlled.
For our soft-threshold Dikin walk, though the proposal generates a point that is relatively close
to the starting point, due to its Gaussianity nature, much less structure can be exploited and thus
maintained. Instead, we propose a maintenance-free approach that uses randomized sketching to
generate a spectral approximation of Hlog(x).

To illustrate the algorithm, let’s first define a matrix

B = A(S(x)−1/2 ⊗S(x)−1/2).

It is not hard to see that Hlog(x) = BB⊤. Due to the Kronecker product structure of (S(x)−1/2 ⊗
S(x)−1/2), it is natural to consider sketches for Kronecker product of matrices [Diao et al., 2018,
2019, Ahle et al., 2020, Song et al., 2021]. Following the standard procedure for using sketch to
generate spectral approximation, we can choose a TensorSRHT matrix [Ahle et al., 2020, Song
et al., 2021] T and compute R := (S(x)−1/2 ⊗S(x)−1/2)T then form AR. This is unfortunately,
not efficient enough, as multiplying A with R might be too slow. To further optimize the runtime
efficiency, we use a more intrinsic approach based on the matrix T and B. It is well-known that the
i-th row of B can be first computed as S(x)−1/2AiS(x)−1/2 then vectorize. If we can manage to
apply the sketch in a row-by-row fashion, then fast matrix multiplication can be utilized for even
faster algorithms. We recall that T = P ·(HD1 ⊗HD2) where H is the Hadamard matrix and P is a
row sampling matrix. To compute a row, we can first apply HD1 and HD2 individually to S(x)−1/2

in eO(n2) time since H is a Hadamard matrix. We can then form two matrices X and Y with rows
being the corresponding sampled rows from HD1 and HD2. Finally, we compute XAiY ⊤to form
one row. Using fast matrix multiplication, we can form each row in Tmat(d3, n, n) = O(n2d3(ω−2))
time. Applying this procedure to d rows leads to an overall O(n2d3ω−5) time, which beats the
O(dnω) time as long as n ≫d. Our runtime is somewhat slow when n is not that larger than d, this
is due to we have to set ϵ = O(1/d). For more popular regimes where it suffices to set ϵ = O(1), our
algorithm outperforms exact computation as long as n ≥d.

3
Applications

There are many applications of our Dikin walk, such as differentially private learning, simulated
annealing and regret minimization. For a more comprehensive overview of the applications, we refer
readers to Narayanan and Rakhlin [2017].

Differentially private learning.
Let X m = {x1, . . . , xm} denote an m-point dataset, associate
a convex loss function ℓi to point xi, the learning problem attempts to find an optimal parameter
θ∗∈K that minimizes ℓ(θ) = Pm
i=1 ℓi(θ). We could further enforce privacy constraints to learning:
we say a randomized algorithm M : X m →R is ϵ-DP if for any datasets X, X′ ∈X m differ by a
single point and for any S ⊆R, Pr[M(X) ⊆S] ≤eϵ Pr[M(X′) ⊆S]. The differentially private
learning seeks to solve the learning problem under DP guarantees, and it has been shown that if one
allows for approximate DP, then it can be achieved via the exponential mechansim and sampling
from the distribution π(θ) ∝e−ℓ(θ) [Wang et al., 2015]. If K is a polytope or spectrahedron, and
in addition ℓis L-Lipschitz, one can implement our walk to obtain an eO(d2 + dL2R2) mixing for
polytopes or eO(nd + dL2R2) mixing for spectrahedra. Via a standard reduction from TV distance
bound to infinity distance bound [Mangoubi and Vishnoi, 2022], our walk can also be adapted for a
pure DP guarantee [Lin et al., 2024].

9


---Page Break---
Simulated annealing for convex optimization.
Let f : Rd →R be an L-Lipschitz and convex
function, we consider the problem of minimizing f over K when we can only access the function
value of f. This is also referred to as zeroth-order convex optimization. One could solve the
problem via the simulated annealing framework, in which one needs to sample from the distribution
π(x) ∝e−f(x)/T where T is the temperature parameter [Kalai and Vempala, 2006]. Our walk
can be adapted to the framework for polytopes, it mixes in eO(d2.5 + d1.5L2R2) steps and for
spectrahedra, it mixes in eO(nd1.5 + d1.5L2R2) steps, improving upon prior best algorithms that mix
in min{d5.5 + d3.5L2R2, nd1.5 + d1.5L2R2} steps for polytopes and n2d3.5 + n2d1.5L2R2 steps
for spectrahedra [Narayanan and Rakhlin, 2017, Mangoubi and Vishnoi, 2023].

Online convex optimization.
Consider the following online optimization problem: let K be a
convex set that we could choose our actions from and let ℓ1, . . . , ℓT be a sequence of unknown convex
cost functions with ℓt : K →R. The goal is to design a good strategy that chooses from K so that
the total cost is small, compared to the offline optimal cost, in which one could choose the strategy
after seeing the entire sequence of ℓt’s. The algorithm works as follows: at round t, we choose
a distribution µt−1 supported on K and play the action Yt ∼µt−1. Our goal is to minimize the
expected regret, defined as RegT (U) = E[PT
t=1 ℓt(Yt)−PT
t=1 ℓt(U)] with respect to all randomized
strategies defined by distribution pU. It turns out if one sets st(x) = η Pt
s=1 ℓt(x) where η > 0 is a
step size parameter and sets µt ∝e−st, then this update rule reflects the multiplicative weights update
algorithm [Arora et al., 2012]. Moreover, one could prove a nearly-optimal regret bound of O(
√

T)
if the KL divergence between µ0 and all possible pU’s are bounded. The algorithmic problem of
sampling from µt is equivalent to the log-concave sampling problem we study in this paper, and we
could efficiently generate strategies with good regrest, similar to Narayanan and Rakhlin [2017]. Let
ℓt’s be 1-Lipschitz linear functions, Narayanan and Rakhlin [2017] uses their time-varying Dikin
walk to achieve a regret of O(d2.5√

T), while we could significantly improve this bound to O(d
√

T).

4
Conclusion

In this paper, we design a class of error-robust Dikin walks for sampling from a log-concave and
log-Lipschitz density over a convex body. The key features of our walks are that their mixing time
depends linearly on the complexity of self-concordant barrier function of the convex body, and they
allow computationally intensive quantities to be approximated rather than computed exactly. For
polytopes, our walk mixes in eO(d2 + dL2R2) steps with a per iteration cost of eO(ndω−1), and for
spectrahedra, our walk mixes in eO(nd + dL2R2) steps with a per iteration cost of eO(nω + n2d3ω−5).
For polytopes, our walk is the first successful adaptation of the Lee-Sidford barrier for log-concave
sampling under the minimal assumption that f is Lipschitz, improving upon the mixing of prior
works [Narayanan and Rakhlin, 2017, Mangoubi and Vishnoi, 2023, 2024]. For spectrahedra, we
improve the mixing of Narayanan and Rakhlin [2017] from n2d3 + n2dL2R2 to nd + dL2R2,
for the term that depends on only n and d, we obtain an improvement of nd2, and for the term
that depends on L and R, we shave off the quadratic dependence on n. Moreover, we adapt our
error-robust framework and present a sketching algorithm for approximating the Hessian matrix in
eO(nω + n2d3ω−5) time. Our results have deep implications, as it could be leveraged for differentially
private learning, convex optimization and regret minimization.

While our framework offers the fastest mixing Dikin walk for log-concave sampling, its mixing time
has a rather unsatisfactory dependence on the radius of the bounding box, R. It would be interesting
if one could design a walk that does not depend on R. We also note that we require the density of be
log-Lipschitz rather than Lipschitz. Kook and Vempala [2024] shows that if f in addition satisfies
the relative strongly-convex property, then there exists a walk whose mixing does not depend on R
and one only needs the density of be Lipschitz. Another important direction is to further improve the
mixing rate of sampling over a spectrahedron using barrier functions such as volumetric barrier and
hybrid barrier [Anstreicher, 2000], while maintaining an small per iteration cost. Finally, extending
RHMC to log-concave sampling would be essentially, as it has the potential to give the fastest mixing
walk by utilizing Riemannian metrics instead of Euclidean. As our work is theoretical in nature, we
don’t foresee any potential negative societal impact. It has the potential positive impact to reduce
energy consumption and carbon emission when deployed in practice.

10


---Page Break---
Acknowledgement

We would like to thank Jonathan Kelner, Yin Tat Lee, Santosh Vempala and Yunbum Kook for
valuable discussions and anonymous reviewers for helpful comments. The research is partially
supported by the NSF awards: SCALE MoDL-2134209, CCF-2112665 (TILOS). It is also supported
by the U.S. Department of Energy, the Office of Science, the Facebook Research Award, as well
as CDC-RFA-FT-23-0069 from the CDC’s Center for Forecasting and Outbreak Analytics. Lichen
Zhang is supported by NSF awards CCF-1955217 and DMS-2022448. For more information related
to the paper and adjacent topics, see https://www.youtube.com/@zhaosong2031 and https:
//space.bilibili.com/3546587376650961.

References

Thomas D Ahle, Michael Kapralov, Jakob BT Knudsen, Rasmus Pagh, Ameya Velingker, David P
Woodruff, and Amir Zandieh. Oblivious sketching of high-degree polynomial kernels. In Pro-
ceedings of the Fourteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages
141–160, 2020.

Josh Alman and Virginia Vassilevska Williams. A refined laser method and faster matrix multiplica-
tion. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA), pages
522–539. SIAM, 2021.

Kurt M Anstreicher. The volumetric barrier for semidefinite programming. Mathematics of Operations
Research, 25(3):365–380, 2000.

Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a meta-
algorithm and applications. Theory of Computing, 8(1):121–164, 2012.

Christos Boutsidis, David P Woodruff, and Peilin Zhong. Optimal principal component analysis in
distributed and streaming models. In Proceedings of the forty-eighth annual ACM symposium on
Theory of Computing (STOC), pages 236–249, 2016.

Nicolas Brosse, Alain Durmus, and Eric Moulines. The promises and pitfalls of stochastic gradient
langevin dynamics. Advances in Neural Information Processing Systems, 31, 2018.

Niladri Chatterji, Nicolas Flammarion, Yian Ma, Peter Bartlett, and Michael Jordan. On the theory of
variance reduction for stochastic gradient monte carlo. In International Conference on Machine
Learning, pages 764–773. PMLR, 2018.

Yuansi Chen and Ronen Eldan. Hit-and-run mixing via localization schemes, 2022.

Yuansi Chen, Raaz Dwivedi, Martin J. Wainwright, and Bin Yu. Fast mcmc sampling algorithms on
polytopes. J. Mach. Learn. Res., 2018.

Michael B. Cohen and Richard Peng. Lp row sampling by lewis weights. In Proceedings of the
Forty-Seventh Annual ACM Symposium on Theory of Computing, STOC ’15, 2015.

Michael B Cohen, Ben Cousins, Yin Tat Lee, and Xin Yang. A near-optimal algorithm for ap-
proximating the john ellipsoid. In Conference on Learning Theory, pages 849–873. PMLR,
2019.

Arnak S Dalalyan and Avetik Karagulyan. User-friendly guarantees for the langevin monte carlo
with inaccurate gradient. Stochastic Processes and their Applications, 129(12):5278–5311, 2019.

Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. Saga: A fast incremental gradient method
with support for non-strongly convex composite objectives. Advances in neural information
processing systems, 27, 2014.

Huaian Diao, Zhao Song, Wen Sun, and David Woodruff. Sketching for kronecker product regression
and p-splines. In International Conference on Artificial Intelligence and Statistics, pages 1299–
1308. PMLR, 2018.

11


---Page Break---
Huaian Diao, Rajesh Jayaram, Zhao Song, Wen Sun, and David Woodruff. Optimal sketching
for kronecker product regression and low rank approximation. Advances in neural information
processing systems, 32:4737–4748, 2019.

Zhiyan Ding and Qin Li. Langevin monte carlo: random coordinate descent and variance reduction.
J. Mach. Learn. Res., 22:205–1, 2021.

Petros Drineas, Michael W. Mahoney, and S. Muthukrishnan. Sampling algorithms for l2 regres-
sion and applications. In Proceedings of the Seventeenth Annual ACM-SIAM Symposium on
Discrete Algorithm, SODA ’06, page 1127–1136, USA, 2006. Society for Industrial and Applied
Mathematics.

Ran Duan, Hongxun Wu, and Renfei Zhou. Faster matrix multiplication via asymmetric hashing. In
FOCS, 2023.

Kumar Avinava Dubey, Sashank J Reddi, Sinead A Williamson, Barnabas Poczos, Alexander J Smola,
and Eric P Xing. Variance reduction in stochastic gradient langevin dynamics. Advances in neural
information processing systems, 29, 2016.

Maryam Fazel, Yin Tat Lee, Swati Padmanabhan, and Aaron Sidford. Computing lewis weights to
high precision. In Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms
(SODA), pages 2723–2742. SIAM, 2022.

Z Furedi and I Barany. Computing the volume is difficult. In Proceedings of the Eighteenth Annual
ACM Symposium on Theory of Computing, STOC ’86, page 442–447, New York, NY, USA, 1986.
Association for Computing Machinery.

François Le Gall and Florent Urrutia. Improved rectangular matrix multiplication using powers of the
coppersmith-winograd tensor. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium
on Discrete Algorithms, SODA ’18, page 1029–1046, 2018.

Francois Le Gall. Faster rectangular matrix multiplication by combination loss analysis. In Pro-
ceedings of the Thirty-Fifth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA’24,
2024.

Khashayar Gatmiry, Jonathan Kelner, and Santosh S. Vempala. Sampling polytopes with riemannian
hmc: Faster mixing via the lewis weights barrier. In Shipra Agrawal and Aaron Roth, editors,
Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of
Machine Learning Research, pages 1796–1881. PMLR, 30 Jun–03 Jul 2024.

Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 1990.

Baihe Huang, Shunhua Jiang, Zhao Song, Runzhou Tao, and Ruizhe Zhang. Solving sdp faster: A
robust ipm framework and efficient implementation. In FOCS, 2022.

Arun Jambulapati, Yang P. Liu, and Aaron Sidford. Improved iteration complexities for overcon-
strained p-norm regression. In Proceedings of the 54th Annual ACM SIGACT Symposium on
Theory of Computing, STOC 2022, 2022.

Haotian Jiang, Tarun Kathuria, Yin Tat Lee, Swati Padmanabhan, and Zhao Song. A faster interior
point method for semidefinite programming. In 2020 IEEE 61st annual symposium on foundations
of computer science (FOCS), pages 910–918. IEEE, 2020a.

Haotian Jiang, Yin Tat Lee, Zhao Song, and Sam Chiu-wai Wong. An improved cutting plane method
for convex optimization, convex-concave games, and its applications. In Proceedings of the 52nd
Annual ACM SIGACT Symposium on Theory of Computing, pages 944–953, 2020b.

Shunhua Jiang, Zhao Song, Omri Weinstein, and Hengjie Zhang. A faster algorithm for solving
general lps. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing,
pages 823–832, 2021.

Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance
reduction. Advances in neural information processing systems, 26, 2013.

12


---Page Break---
William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space.
Contemporary mathematics, 26(189-206):1, 1984.

Adam Tauman Kalai and Santosh Vempala. Simulated annealing for convex optimization. Mathemat-
ics of Operations Research, 31(2):253–266, 2006.

R. Kannan, L. Lovász, and M. Simonovits. Isoperimetric problems for convex bodies and a localiza-
tion lemma. Discrete Comput. Geom., 1995.

Ravindran Kannan and Hariharan Narayanan. Random walks on polytopes and an affine interior
point method for linear programming. Mathematics of Operations Research, 37(1):1–20, 2012.

Yunbum Kook and Santosh S. Vempala. Gaussian cooling and Dikin walks: The interior-point
method for logconcave sampling. In Shipra Agrawal and Aaron Roth, editors, Proceedings of
Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning
Research, pages 3137–3240. PMLR, 30 Jun–03 Jul 2024.

Aditi Laddha, Yin Tat Lee, and Santosh Vempala. Strong self-concordance and sampling. In
Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages
1212–1222, 2020.

Beatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection.
Annals of Statistics, pages 1302–1338, 2000.

Yin Tat Lee and Aaron Sidford. Path finding methods for linear programming: Solving linear
programs in õ(sqrt(rank)) iterations and faster algorithms for maximum flow. In 55th IEEE Annual
Symposium on Foundations of Computer Science, FOCS 2014, Philadelphia, PA, USA, October
18-21, 2014, pages 424–433, 2014.

Yin Tat Lee and Aaron Sidford. Solving linear programs with sqrt (rank) linear system solves. arXiv
preprint arXiv:1910.08033, 2019.

Yin Tat Lee and Santosh S. Vempala. Geodesic walks in polytopes. In Proceedings of the 49th
Annual ACM SIGACT Symposium on Theory of Computing, STOC 2017, 2017.

Yin Tat Lee and Santosh S Vempala. Convergence rate of riemannian hamiltonian monte carlo and
faster polytope volume computation. In Proceedings of the 50th Annual ACM SIGACT Symposium
on Theory of Computing, pages 1115–1121, 2018.

Yin Tat Lee and Man–Chung Yue. Universal barrier is n-self-concordant. Mathematics of Operations
Research, 46(3):1129–1148, 2021.

Yin Tat Lee, Aaron Sidford, and Sam Chiu-wai Wong. A faster cutting plane method and its
implications for combinatorial and convex optimization. In Foundations of Computer Science
(FOCS), 2015 IEEE 56th Annual Symposium on, pages 1049–1065. IEEE, 2015.

Yin Tat Lee, Zhao Song, and Qiuyi Zhang. Solving empirical risk minimization in the current matrix
multiplication time. In Conference on Learning Theory, pages 2140–2157. PMLR, 2019.

Zhize Li, Tianyi Zhang, Shuyu Cheng, Jun Zhu, and Jian Li. Stochastic gradient hamiltonian monte
carlo with variance reduction for bayesian inference. Machine Learning, 108(8):1701–1727, 2019.

Yingyu Lin, Yian Ma, Yu-Xiang Wang, Rachel Emily Redberg, and Zhiqi Bu. Tractable MCMC
for private learning with pure and gaussian differential privacy. In The Twelfth International
Conference on Learning Representations, 2024.

László Lovász and Miklós Simonovits. Random walks in a convex body and an improved volume
algorithm. Random structures & algorithms, 4(4):359–412, 1993.

László Lovász and Santosh Vempala. Hit-and-run is fast and fun. preprint, Microsoft Research, 2003.

László Lovász and Santosh Vempala. Simulated annealing in convex bodies and an O∗(n4) volume
algorithm. Journal of Computer and System Sciences, 72(2):392–417, 2006.

13


---Page Break---
László Lovász and Santosh Vempala. The geometry of logconcave functions and sampling algorithms.
Random Structures & Algorithms, 30(3):307–358, 2007.

Yichao Lu, Paramveer Dhillon, Dean P Foster, and Lyle Ungar. Faster ridge regression via the
subsampled randomized hadamard transform. In Advances in neural information processing
systems, pages 369–377, 2013.

Oren Mangoubi and Nisheeth Vishnoi. Sampling from log-concave distributions with infinity-distance
guarantees. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors,
Advances in Neural Information Processing Systems, volume 35, pages 12633–12646. Curran
Associates, Inc., 2022.

Oren Mangoubi and Nisheeth K Vishnoi. Sampling from structured log-concave distributions via a
soft-threshold dikin walk. In Advances in Neural Information Processing Systems, NeurIPS’23,
2023.

Oren Mangoubi and Nisheeth K. Vishnoi. Faster sampling from log-concave densities over polytopes
via efficient linear solvers. In The Twelfth International Conference on Learning Representations,
2024.

Hariharan Narayanan. Randomized interior point methods for sampling and optimization. Ann. Appl.
Probab., 2016.

Hariharan Narayanan and Alexander Rakhlin. Efficient sampling from time-varying log-concave
distributions. The Journal of Machine Learning Research, 18(1):4017–4045, 2017.

Jelani Nelson and Huy L Nguyên. OSNAP: Faster numerical linear algebra algorithms via sparser
subspace embeddings. In 54th Annual IEEE Symposium on Foundations of Computer Science
(FOCS), pages 117–126. IEEE, 2013.

Yurii Nesterov and Arkadii Nemirovskii. Self-concordant functions and polynomial-time methods in
convex programming. USSR Academy of Sciences, 1989.

Yurii Nesterov and Arkadii Nemirovskii. Interior-point polynomial algorithms in convex program-
ming. SIAM, 1994.

Lianke Qin, Zhao Song, Lichen Zhang, and Danyang Zhuo. An online and unified algorithm for
projection matrix vector multiplication with application to empirical risk minimization. In AISTATS,
2023.

Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic
gradient langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory, pages
1674–1703. PMLR, 2017.

James Renegar. A polynomial-time algorithm, based on newton’s method, for linear programming.
Math. Program., 1988.

Sushant Sachdeva and Nisheeth K Vishnoi. The mixing time of the dikin walk in a polytope—a
simple proof. Operations Research Letters, 44(5):630–634, 2016.

Shai Shalev-Shwartz and Tong Zhang. Stochastic dual coordinate ascent methods for regularized loss
minimization. Journal of Machine Learning Research, 14(1), 2013.

Zhao Song and Zheng Yu. Oblivious sketching-based central path method for solving linear program-
ming problems. In 38th International Conference on Machine Learning (ICML), 2021.

Zhao Song, David P Woodruff, and Peilin Zhong. Low rank approximation with entrywise l1-norm
error. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pages
688–701, 2017.

Zhao Song, David P Woodruff, and Peilin Zhong. Relative error tensor low rank approximation.
In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms, pages
2772–2789. SIAM, 2019.

14


---Page Break---
Zhao Song, David Woodruff, Zheng Yu, and Lichen Zhang. Fast sketching of polynomial kernels of
polynomial degree. In International Conference on Machine Learning, pages 9812–9823. PMLR,
2021.

Daniel A Spielman and Nikhil Srivastava. Graph sparsification by effective resistances. SIAM Journal
on Computing, 40(6):1913–1926, 2011.

Joel A Tropp. Improved analysis of the subsampled randomized hadamard transform. Advances in
Adaptive Data Analysis, 3(01n02):115–126, 2011.

Pravin M Vaidya. A new algorithm for minimizing convex functions over convex sets. In 30th Annual
Symposium on Foundations of Computer Science (FOCS), pages 338–343. IEEE Computer Society,
1989.

Santosh Vempala. Geometric random walks: a survey. Combinatorial and computational geometry,
52(573-612):2, 2005.

Yu-Xiang Wang, Stephen Fienberg, and Alex Smola. Privacy for free: Posterior sampling and
stochastic gradient monte carlo. In Francis Bach and David Blei, editors, Proceedings of the 32nd
International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning
Research, pages 2493–2502, Lille, France, 07–09 Jul 2015. PMLR.

Virginia Vassilevska Williams. Multiplying matrices faster than coppersmith-winograd. In Proceed-
ings of the forty-fourth annual ACM symposium on Theory of computing (STOC), pages 887–898.
ACM, 2012.

Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, and Renfei Zhou. New bounds for matrix
multiplication: from alpha to omega. In Proceedings of the Thirty-Fifth Annual ACM-SIAM
Symposium on Discrete Algorithms, SODA’24, 2024.

David P. Woodruff. Sketching as a tool for numerical linear algebra. Foundations and Trends in
Theoretical Computer Science, 10(1–2):1–157, 2014.

Difan Zou, Pan Xu, and Quanquan Gu. Stochastic variance-reduced hamilton monte carlo methods.
In International Conference on Machine Learning, pages 6028–6037. PMLR, 2018.

15


---Page Break---
Appendix

Roadmap.
Since there are many technical details in the appendix, we provide a roadmap. The
appendix can be divided into 5 parts: the first part conveys some preliminary information and states
the sufficient conditions we require barrier functions to have, in Section A and B. The second part
proves the mixing time of the Dikin walk when barrier functions satisfy the conditions, and the proofs
are divided into Section C and D. The next part focuses on the runtime complexity of sampling from
polytopes, including how to generate a spectral approximation of the Hessian, approximate leverage
scores and Lewis weights to high precision, in Section E and how to incorporate these algorithmic
prototypes to implement an efficient sampling algorithm in Section F. We dedicate Section G to study
the log-barrier for sampling from a spectrahedron, as it is relatively less explored before. Finally,
we prove the convexity of the function log det(H(x) + Id) for log-barrier, volumetric barrier and in
extension Lee-Sidford barrier in Section H and I.

A
Preliminaries

For any positive integer, we use [n] to denote the set {1, 2, · · · , n}. For a vector x ∈Rn, we
use ∥x∥2 to denote its ℓ2 norm, i.e., ∥x∥2 := (Pn
i=1 x2
i )1/2. We use ∥x∥1 to denote its ℓ1 norm,
∥x∥1 := Pn
i=1 |xi|. We use ∥x∥∞to denote its ℓ∞norm, i.e., ∥x∥∞:= maxi∈[n] |xi|. For a random
variable X, we use E[X] to denote its expectation. We use Pr[·] to denote probability.

We use 0d to denote a length-d vector where every entry is 0. We use 1d to denote a length-d vector
where every entry is 1. We use Id to denote an identity matrix which has size d × d or simply I when
dimension is clear from context.

For a matrix A, we use A⊤to denote the transpose of matrix A. For a square and non-singular matrix
A, we use A−1 to denote the inverse of matrix A. For a real square matrix A, we say it is positive
definite (PD, i.e., A ≻0) if for all vectors x ∈Rn (except for 0n), we have x⊤Ax > 0. For a real
square matrix A, we say it is positive semi-definite(PSD, i.e., A ⪰0) if for all vectors x ∈Rn, we
have x⊤Ax ≥0. For a square matrix, we use det(A) to denote the determinant of matrix A. For
a matrix A, we use ∥A∥to denote its spectral norm, use ∥A∥F to denote its Frobenius norm, i.e.,
∥A∥F := (Pn
i=1
Pd
j=1 A2
i,j)1/2 and use nnz(A) to denote the number of nonzero entries in A.

Given a function f : K →R, we say it’s convex if for any x, y ∈K, f(x) ≥f(y) + ∇f(y)⊤(x −y).
We say it’s L-Lipschitz if |f(x) −f(y)| ≤L · ∥x −y∥2 for a fixed parameter L > 0.

For two distributions P1 and P2, we use TV(P1, P2) to denote the total variation (TV) distance
between P1 and P2.

For a convex body K ⊆Rd, we use Int(K) to denote the interior of K.

We use ⊗to denote Kronecker product. We use A ⊗S B to denote A ⊗I + I ⊗B. Given a matrix A,
we use vec(A) to denote its vectorization, since we will always apply vec(·) to a symmetric matrix,
whether the vetorization is row-major or column-major doesn’t matter. We use ◦to denote Hadamard
or element-wise product.

For two vectors x, y ∈Rd, we use ⟨x, y⟩= x⊤y to denote the standard Euclidean inner product over
Rd, and for two symmetric matrices A, B ∈Rd×d, we use ⟨A, B⟩= tr[A⊤B] to denote the trace
inner product.

In Section A.1, we present several definitions and notations related to fast matrix multiplication. In
Section A.2, we provide some backgrounds about convex geometry. In Section A.3, we state several
basic probability tools. In Section A.4, we state several basic facts related to trace, and Kronecker
product. In Section A.5, we present the standard notion about self-concordance barrier.

A.1
Fast matrix multiplication

Definition A.1 (Fast matrix multiplication). Given three positive integers a, b, c, we use Tmat(a, b, c)
to denote the time of multiplying an a × b matrix with another b × c matrix.

For convenience, we also define the ω(·, ·, ·) function as follows:

16


---Page Break---
Definition A.2 (Fast matrix multiplication, an alternative notation). Given x, y, z, we use dω(x,y,z) to
denote the time of multiplying a dx × dy matrix with another dy × dz matrix.

Lemma A.3 (Gall and Urrutia [2018]). We have the following bounds of ω(·, ·, ·):

• ω(1, 1, 1) = ω,

• ω(1, 3, 1) = 4.199712 (see Table 3 in Gall and Urrutia [2018]).

Here, ω denotes the exponent of matrix multiplication, currently ω ≈2.373 [Williams, 2012, Gall
and Urrutia, 2018, Alman and Williams, 2021, Duan et al., 2023, Williams et al., 2024, Gall, 2024].

A.2
Convex geometry

We define B(x, R) ⊂Rd as

B(x, R) := {y ∈Rd : ∥y −x∥2 ≤R}.

Define polytope K as

K := {x ∈Rd : Ax ≤b}.

Define spectrahedron K as

K := {x ∈Rd :

d
X

i=1
xiAi ⪰C},

where A1, . . . , Ad, C ∈Rn×n are symmetric matrices.

We will often use the notion of a Dikin ellipsoid:

Definition A.4 (Dikin ellipsoid). We define the Dikin ellipsoid Dθ ⊂Rd as follows

Dθ := {w ∈Rd : w⊤H−1(θ)w ≤1}

where H(θ) ∈Rd×d is the Hessian of self-concordant barrier at θ/

We define the standard cross-ratio distance (see Definition E.1 in Mangoubi and Vishnoi [2023] for
example).

Definition A.5 (Cross-ratio distance). Let u, v ∈K. If u ̸= v, let p, q be two endpoints of the chord
in K which passes through u and v such that the four points lie in the order of p, u, v, q, let

σ(u, v) := ∥u −v∥2 · ∥p −q∥2

∥p −u∥2 · ∥v −q∥2
.

We additionally set σ(u, v) = 0 for u = v.

For two subsets S1, S2 ⊆K, we define

σ(S1, S2) := min{σ(u, v) : u ∈S1, v ∈S2}.

We state the standard isoperimetric inequality for cross-ratio distance.

Lemma A.6 (Lovász and Vempala [2003]). Let π : Rd →R be a log-concave density, with support
on a convex body K. Then for any partition of Rd into measurable sets S1, S2, S3, the induced
measure π∗satisfies

π∗(S3) ≥σ(S1, S2) · π∗(S1) · π∗(S2).

A.3
Probability tools

We state some useful concentration inequalities.

17


---Page Break---
Lemma A.7 (Lemma 1 on page 1325 of Laurent and Massart [2000]). Let X ∼X 2
k be a chi-squared
distributed random variable with k degrees of freedom. Each random variable has zero mean and σ2
variance. Then

Pr[X −kσ2 ≥(2
√

kt + 2t)σ2] ≤exp(−t),

Pr[kσ2 −X ≥2
√

ktσ2] ≤exp(−t).

Lemma A.8 (Matrix Chernoff bound [Tropp, 2011]). Let X1, . . . , Xs be independent copies of a
symmetric random matrix X ∈Rd×d with E[X] = 0, ∥X∥≤γ almost surely and ∥E[X⊤X]∥≤σ2.
Let W = 1

s
P
i∈[s] Xi. For any ϵ ∈(0, 1),

Pr[∥W∥≥ϵ] ≤2d · exp

−
sϵ2

σ2 + γϵ/3


.

A.4
Basic algebra facts

Lemma A.9. Let A and B be m × m matrices. Then we have

• tr[AB] = tr[BA].

• If A is symmetric, then tr[AB] = tr[AB⊤].

• If A and B are PSD, then ⟨A, B⟩≥0, and ⟨A, B⟩= 0 if and only if AB = 0m×m.

• If A ⪰0 and B ⪰C, then ⟨A, B⟩≥⟨A, C⟩.

Lemma A.10. Let A, B, C, D be conforming matrices. Then

• (A ⊗B)(C ⊗D) = AC ⊗BD.

• (A ⊗S B)(C ⊗S D) = 1
2(AC ⊗S BD + AD ⊗S BC).

• (A ⊗B)⊤= A⊤⊗B⊤.

• If A and B are non-singular, then A ⊗B is non-singular, and (A ⊗B)−1 = A−1 ⊗B−1.

• vec(ABC) = (C⊤⊗A) · vec(B).

A.5
Self-concordant barrier

We provide a standard definition about self-concordance barrier,
Definition A.11 (Self-concordant barrier). A real-valued function F : Int(K) →R, is a regular
self-concordant barrier if it satisfies the conditions stated below. For convenience, if x ̸∈Int(K), we
define F(x) = ∞.

1. (Convex, Smooth) F is a convex thrice continuously differentiable function on Int(K).

2. (Barrier) For every sequence of points {xi} ∈Int(K) converging to a point x ̸∈Int(K),
limi→∞f(xi) = ∞.

3. (Differential Inequalities) For all h ∈Rd and all x ∈Int(K), the following inequalities
hold.

(a) D2F(x)[h, h] is 2-Lipschitz continuous with respect to the local norm, which is equiv-
alent to
D3F(x)[h, h, h] ≤2(D2F(x)[h, h])
3
2 .

(b) F(x) is ν-Lipschitz continuous with respect to the local norm defined by F,

∥∇hF(x)∥2
2 ≤ν · ∥h∥2
H(x).

We call the smallest positive integer ν for which this holds, the self-concordance
parameter of the barrier.

18


---Page Break---
B
Sufficient Conditions for Log-concave Sampling

To prove that our algorithm works for log-concave distribution, we state three key assumptions (see
Section B.1) which are the sufficient conditions to prove the mixing rate of log-concave distribution,
for different barrier functions.

B.1
Conditions: ν-symmetry, convexity and bounded local norm

We state three key conditions in order to lower bound the conductance for log-concave sampling.
Assumption B.1. Given a convex body K, let H : K →Rd×d be a self-concordant matrix function,
we assume that

(i) ν-symmetry. For any x ∈K, we have Ex(1) ⊆K ∩(2x −K) ⊆Ex(
√

ν) where
Ex(r) = {y ∈Rd : (y −x)⊤H(x)(y −x) ≤r2}.

(ii) Convexity. Let F : K →R be defined as F(x) = log(det(H(x) + Id)), then F is convex
in x.

(iii) Bounded local norm. Let ∇log(det(H(x))) denote the gradient of log(det(H(x))) in x.
For any x ∈K, we have

∥(H(x))−1/2 · ∇log(det(H(x)))∥2
2 ≤eO(d).

For the ν-symmetry assumption, we will prove that for barriers of the concern (log-barrier, Lee-
Sidford barrier for polytopes and log-barrier for spectrahedra), ν = ν. This characteristic is proved
in Laddha et al. [2020] for polytopes, and we prove that for spectrahedra in Section G. For convexity,
due to the large amount of calculations, we defer to Section H and I. For bounded local norm, the
results for polytopes are similarly shown in Laddha et al. [2020], and we will prove for log-barrier
over spectrahedra.

B.2
Local PSD approximation for any self-concordant matrix function

In this section, we prove a generalization of the matrix function self-concordance to regularized
matrix function. When H is Hessian of the log barrier, this fact was proved in [Mangoubi and Vishnoi,
2023, Lemma E.3]. We generalize this fact to general barriers.
Lemma B.2. Let α ∈(0, 1). Let H : K →R be a self-concordant matrix function and Φ(u) :=
α−1H(u) + η−1Id. For any u, v ∈K such that ∥u −v∥Φ(u) ≤
1
α1/2 we have

(1 −α1/2 · ∥u −v∥Φ(u))2 · Φ(v) ⪯Φ(u) ⪯(1 + α1/2 · ∥u −v∥Φ(u))2 · Φ(v).

Proof. By [Laddha et al., 2020, Lemma 1.1], we have

(1 −∥u −v∥H(u))2 · H(v) ⪯H(u) ⪯(1 + ∥u −v∥H(u))2 · H(v).

Because H(u) ⪯α · Φ(u), we get

(1 −α1/2 · ∥u −v∥Φ(u))2 · H(v) ⪯H(u) ⪯(1 + α1/2 · ∥u −v∥Φ(u))2 · H(v).

Using Φ(u) = α−1H(u) + η−1Id again we get

(1 −α1/2 · ∥u −v∥Φ(u))2 · Φ(v) ⪯Φ(u) ⪯(1 + α1/2 · ∥u −v∥Φ(u))2 · Φ(v).

This completes the proof.

B.3
Bounded local norm for regularized Hessian function

We note that the bounded local norm condition generally holds for standard barrier functions, but
in our core argument, we will instead rely on the bounded local norm of the regularized Hessian
function. We prove that the bounded local norm condition implies the bounded local norm on the
regularized Hessian function.

19


---Page Break---
Lemma B.3. Let H : K →Rd×d be a matrix function, define F(x) = log det(H(x) + L2Id) for
L ∈R, then we have the following inequality:

∥(H(x) + L2Id)−1/2∇F(x)∥2
2 ≤∥H(x)−1/2∇log det(H(x))∥2
2.

Proof. Let H(x) = UΛU ⊤be the eigendecomposition of the matrix H(x), let y = U ⊤∇H(x), then

∥(H(x) + L2Id)−1/2∇F(x)∥2
2 = ∥(H(x) + L2Id)−3/2∇H(x)∥2
2
= ∥U(Λ + L2Id)−3/2U ⊤∇H(x)∥2
2
= ∥(Λ + L2Id)−3/2y∥2
2,

the given condition implies

∥H(x)−1/2∇log det(H(x))∥2
2 = ∥H(x)−3/2∇H(x)∥2
2
= ∥UΛ−3/2U ⊤∇H(x)∥2
2
= ∥Λ−3/2y∥2
2.

As L2 ≥0, we can expand the regularized squared ℓ2 norm as

∥(Λ + L2Id)−3/2y∥2
2 =

d
X

i=1

1
(λi + L2)3 y2
i

≤

d
X

i=1

1
λ3
i
y2
i

= ∥Λ−3/2y∥2
2.

This completes the proof.

C
Key Tools for Robust Sampling

In this section, we list several tools which can be shared in the core proofs of all the main theorems.
This section is organized as follows. In Section C.1, we provide a tool for Gaussian concentration.
In Section C.2, we show that PSD approximation implies the determinant approximation (by losing
a factor of d). One of the major idea in this work is instead of using exact accept probability,
we will only need to use approximate accept probability. In Section C.3, we prove several useful
robust properties for approximate accept probability. In Section C.4, we present a general lemma
for bounding the TV distance between exact process and approximate process. In Section C.5, we
provide some specific choices for parameters and then bound the TV distance. In Section C.6, we
provide a definition which will be used in proof of lower bound on conductance. In Section C.7, we
show how to lower bound the conductance by cross ratio.

C.1
Gaussian concentration lemma

Lemma C.1. Let ξ ∼N(0, Id), then for any t >
√

2d, we have

Pr[∥ξ∥2 ≥t] ≤exp(−(t2 −d)/8).

Proof. We consider the squared ℓ2 norm of ξ: ∥ξ∥2
2 = Pd
i=1 ξ2
i , which is a χ2 distribution with
degree of freedom d. By Lemma A.7, we know that

Pr[∥ξ∥2
2 ≥2
√

dk + 2k2 + d] ≤exp(−k2),

set k2 = (t2 −d)/8, we obtain the desired bound.

20


---Page Break---
C.2
Spectral approximation to determinant approximation

Lemma C.2. Let ϵΦ ∈(0, 1). Given four psd matrices satify the following conditions

(1 −ϵΦ)Φ(x) ⪯eΦ(x) ⪯(1 + ϵΦ)Φ(x),

(1 −ϵΦ)Φ(z) ⪯bΦ(z) ⪯(1 + ϵΦ)Φ(z).

Then we have

(1 −10 · ϵΦ · d) · det(Φ(x))

det(Φ(z)) ≤det(eΦ(x))

det(bΦ(z))
≤(1 + 10 · ϵΦ · d) · det(Φ(x))

det(Φ(z))

Proof. Since eΦ(x) ∈(1 ± ϵΦ)Φ(x), we have the following spectral bound:

(1 −ϵΦ)Id ⪯Φ(x)−1/2eΦ(x)Φ(x)−1/2 ⪯(1 + ϵΦ)Id,

we can take the determinant of the middle term:

det(Φ(x)−1/2eΦ(x)Φ(x)−1/2) = det(Φ(x)−1/2) det(eΦ(x)) det(Φ(x)−1/2)

= det(eΦ(x))

det(Φ(x))
∈det((1 ± ϵΦ)Id)

∈(1 ± ϵΦ)d

∈1 + 10 · ϵΦ · d.

Thus, we complete the proof.

C.3
Approximate accept probability

We start with some definitions:

Definition C.3. We define

bGu(x) := det(bΦ(u))1/2 · exp(−0.5∥u −x∥2
bΦ(u))

Gx(u) := det(Φ(x))1/2 · exp(−0.5∥u −x∥2
Φ(x))

eGx(u) := det(eΦ(x))1/2 · exp(−0.5∥u −x∥2
eΦ(x))

Definition C.4. We define

P accept
x,Φ(x)(u) := E
bΦ(u)


min{1,
bGu(x) exp(−f(u))
Gx(u) exp(−f(x))}


P accept
x,eΦ(x)(u) := E
bΦ(u)


min{1,
bGu(x) exp(−f(u))
eGx(u) exp(−f(x))
}


We will use Px, ePx as a shorthand for the above notations, and we call Px the exact process and ePx
the approximate process.

Lemma C.5. Fix x and eΦ(x). Suppose eΦ(x) is (ϵH, δH)-good approximation to Φ(x). Then

ϵp :=
E
u∼N(x,Φ−1(x))[|P accept
x,Φ(x)(u) −P accept
x,eΦ(x)(u)|] ≤0.001.

Proof. Note that

Gx(u)

eGx(u)
=
(det Φ(x))1/2 exp(−1

2∥x −u∥2
Φ(x))

(det eΦ(x))1/2 exp(−1

2∥x −u∥2
eΦ(x))
.

21


---Page Break---
Because eΦ(x) is a good approximation to Φ(x), by Lemma C.2 we have

(1 −ϵH)d/2 ≤(det Φ(x))1/2

(det eΦ(x))1/2 ≤(1 + ϵH)d/2

and

(1 −ϵH)2 ≤
∥x −u∥2
Φ(x)
∥x −u∥2
eΦ(x)
≤(1 + ϵH)2.

By choosing step size sufficiently small, we have ∥u−x∥2
Φ(x) < d with probability 0.9999. Therefore

Pr
h
|∥x −u∥2
Φ(x) −∥x −u∥2
eΦ(x)| ≤3ϵHd
i
≥0.9999.

Then

Pr
h
1 −3ϵHd ≤
exp(−0.5∥x −u∥2
Φ(x))

exp(−0.5∥x −u∥2
eΦ(x)) ≤1 + 3ϵHd
i
≥0.9999.

For ϵHd small enough, we have

Pr
h
0.9999 ≤
P accept
x,Φ(x)(u)

P accept
x,eΦ(x)(u)
≤1.0001
i
≥0.9998,

and therefore

Pr[|P accept
x,Φ(x)(u) −P accept
x,eΦ(x)(u)| ≤0.0001] ≥0.9998.

So

E
u∼N(x,Φ−1(x))[|P accept
x,Φ(x)(u) −P accept
x,eΦ(x)(u)|] ≤0.9998 · 0.0001 + 0.0002 · 1

≤0.001.

This completes the proof.

C.4
TV distance between exact process and approximate process

Lemma C.6 (TV distance between exact process and approximate process). Let eΦ denote the (1±ϵH)
approximation of Φ. Let δH denote the failure probability. Let ϵp be defined as Lemma C.5. Let Px
denote the exact process. Let ePx denote the approximate process. If x ∈K, then

TV(Px, ePx) ≤δH + ϵp +
p

dϵH.

Proof. We say eΦ(x) is good if it is an (ϵH, δH)-approximation to Φ(x), and eΦ(x) is bad otherwise.
We can upper bound TV(Px, ePx) as follows:

TV(Px, ePx) ≤E[TV(Px,Φ(x), Px,eΦ(x))]

= E[TV(Px,Φ(x), Px,eΦ(x)) | eΦ(x) is bad]

+ E[TV(Px,Φ(x), Px,eΦ(x)) | eΦ(x) is good]

≤Pr[eΦ(x) is bad] + E[TV(Px,Φ(x), Px,eΦ(x)) | eΦ(x) is good]

≤δH + E[TV(Px,Φ(x), Px,eΦ(x)) | eΦ(x) is good].

where the third step follows from TV(·, ·) ≤1, the last step follows from Lemma E.3.

It remains to compute the second term in the above equation. By Pinsker’s inequality, when eΦ(x) is
good, we have

TV(N(x, eΦ−1(x)), N(x, Φ−1(x)))2

22


---Page Break---
≤0.5 · D(N(x, eΦ−1(x))||N(x, Φ−1(x)))

= 0.25 · (tr[eΦ(x)−1/2Φ(x)eΦ(x)−1/2] −d + log det eΦ(x)

det Φ(x))

≤0.25 · ((1 ± ϵH)d −d + log(1 ± ϵH)d)
≤0.5 · dϵH.
(2)

where the first step follows from Pinsker inequality.

Now, we can upper bound TV(Px, ePx) in the following sense,

TV(Px, ePx) ≤δH + E[TV(Px,Φ(x), Px,eΦ(x)) | eΦ(x) is good]

≤δH +
E
u∼N(x,Φ−1(x))
u′∼N(x,eΦ−1(x))
any coupling

[|P accept
x,Φ(x)(u) −P accept
x,eΦ(x)(u)| · 1{u = u′} + 1{u ̸= u′}]

≤δH +
E
u∼N(x,Φ−1(x))[|P accept
x,Φ(x)(u) −P accept
x,eΦ(x)(u)|] + TV(N(x, eΦ−1(x)), N(x, Φ−1(x)))

= δH + ϵp + TV(N(x, eΦ−1(x)), N(x, Φ−1(x)))

≤δH + ϵp +
p

dϵH

The forth step follows from definition of ϵp, the fifth step follows from Eq. (2).

C.5
TV distance between exact and approximate process: instantiation

Lemma C.7. If dϵH < 0.001 and δH < 0.001. Then, we have

TV(Px, ePx) ≤0.01

Proof. We can upper bound it as follows

TV(Px, ePx) ≤δH + ϵp +
p

dϵH
≤0.001 + ϵp + 0.001
≤0.01

where the first step follows from Lemma C.6, the second step follows from choice ϵH, δH, and the
last step follows from Lemma C.5.

C.6
Lower bound on conductance, definition

Definition C.8. Let β ∈(0, 0.1) denote some fixed constant. Let S1 ⊂K and S2 := K\S1 and
π(S1) ≤1/2. We define S′
1 and S′
2 as follows

S′
1 := {x ∈S1 : ePx(S2) ≤β},

S′
2 := {z ∈S2 : ePz(S1) ≤β}.

C.7
Lower bound on conductance, lemma

Lemma C.9 (Lower bound the conductance by cross ratio). Let S′
1 and S′
2 be defined as Definition C.8.
The conductance ϕ satisfies

ϕ ≥1

16β · σ(S′
1, S′
2).

Proof. The proof follows the general format for conductance proofs for geometric Markov chains
(see e.g. Section 5 of Vempala [2005]).

Let S1 ⊆K and let S2 = K\S1. Let S′
1 and S′
2 be defined as Definition C.8. We define S′
3
S′
3 := (K\S′
1)\S′
2.

23


---Page Break---
By Lemma D.12, we have that

Z

x∈S1
π(x) ePx(S2)dx =
Z

x∈S2
π(x) ePx(S1)dx.
(3)

Thus, by Lemma A.6, we have

π⋆(S′
3) ≥σ(S′
1, S′
2)π⋆(S′
1)π⋆(S′
2).
(4)

Case 1. First, we assume that both π⋆(S′
1) ≥1

4π⋆(S1) and π⋆(S′
2) ≥1

4π⋆(S2). In this case we have
Z

S1
ePx(S2)π(x)dx = 1

2

Z

S1
ePx(S2)π(x)dx + 1

2

Z

S2
ePx(S1)π(x)dx

≥β

2 · π⋆(S′
3)

≥β

2 · σ(S′
1, S′
2) · π⋆(S′
1)π⋆(S2)

≥β

4 · σ(S′
1, S′
2) · min(π⋆(S′
1), π⋆(S′
2))

≥β

16 · σ(S′
1, S′
2) · min(π⋆(S1), π⋆(S2)).
(5)

where the first step follows from Eq. (3), the second step follows from Definition C.8, the third step
follows from Eq. (4), the fifth step follows from one of π⋆(S′
1) and π⋆(S′
2) is at least 1/2.

Case 2. Now suppose that instead either π⋆(S′
1) < 1

4π⋆(S1) or π⋆(S′
2) < 1

4π⋆(S2).

Case 2a. If π⋆(S′
1) < 1

4π⋆(S1) then we have
Z

S1
ePx(S2)π(x)dx= 1

2

Z

S1
ePx(S2)π(x)dx + 1

2

Z

S2
ePx(S1)π(x)dx

≥1

2

Z

S1\S′
1
ePx(S2)π(x)dx

≥1

2 · 3

4 · βπ⋆(S1)

≥3

8β min(π⋆(S1), π⋆(S2)).
(6)

where the first step follows from Eq. (3), and the third step follows from Definition C.8.

Case 2b. Similarly, if π⋆(S′
2) < 1

4π⋆(S2) we have
Z

S1
ePx(S2)π(x)dx = 1

2

Z

S1
ePx(S2)π(x)dx + 1

2

Z

S2
ePx(S1)π(x)dx

≥1

2

Z

S2\S′
2
ePx(S1)π(x)dx

≥1

2 · 3

4 · βπ⋆(S2)

≥3

8β min(π⋆(S1), π⋆(S2)).
(7)

where the first step follows from Eq. (3), and the third step follows from Definition C.8.

Therefore, Eq. (5), (6), and (7) together imply that
1
min(π⋆(S1), π⋆(S2))

Z

S1
ePx(S2)π(x)dx ≥β

16 · σ(S′
1, S′
2)
(8)

for every partition S1 ∪S2 = K.

Hence, Eq. (8) implies that

ϕ =
inf
S⊆K:π⋆(S)≤1

2

1
π⋆(S)

Z

S
ePx(K\S)π(x)dx ≥β

16σ(S′
1, S′
2).

24


---Page Break---
D
Correctness for General Barrier Functions with Regularization

In Section D.1, we show how to bound the density ratios. In Section D.2, we show how to bound the
determinant. In Section D.3, we show how to bound the local norm. In Section D.4, we explain how
to bound the TV distance between two exact distributions. In Section D.5, we show how to bound
the TV distance between two approximate processes. In Section D.6, we show how to bound the TV
distance between two Gaussians. In Section D.7, we prove reversibility and stationary distribution. In
Section D.8, we prove a lower bound on cross ratio distance. In Section D.9, we relate the cross ratio
distance to local norm by utilizing ν-symmetry.

Algorithm 2 Our algorithm for sampling from polytope with log-barrier (formal version of Algo-
rithm 1). As the only differences between log-barrier for polytopes and others are the choice of ν and
how to generate the approximate the Hessian, we only present this algorithm.

1: procedure MAIN(A ∈Rn×d, b ∈Rn, δ ∈(0, 1), x0 ∈Rd, α > 0, η > 0)
▷Theorem 1.1
2:
▷K := {x ∈Rd : Ax ≤b}
3:
Let g be the log-barrier for K with ν = n
4:
α ←Θ(1/d)
5:
η ←Θ(1/(dL2))
6:
T ←(να−1 + η−1R2) · log(w/δ)
7:
x ←x0
▷We are given x0 ∈Int(K)
8:
ϵ ←Θ(1)
9:
ϵH ←Θ(ϵ/d)
10:
for t = 1 →T do
11:
Sample a point ξ ∼N(0, Id)
12:
▷Let H denote the Hessian of barrier function g
13:
eH(x) ←SUBSAMPLE(A, b, x, n, d, ϵH)
▷Algorithm 3
14:
▷eH(x) can be written P

i∈S eσiaia⊤
i where |S| = ϵ−2
H d log d
15:
▷The above step takes nnz(A) log n + Tmat(d, ϵ−2
H d, d) time

16:
eΦ(x) ←α−1 eH(x) + η−1Id
17:
z ←x + eΦ(x)−1/2ξ
18:
if z ∈Int(K) then
19:
bH(z) ←SUBSAMPLE(A, b, z, n, d, ϵH)
▷Algorithm 3
20:
bΦ(z) ←α−1 bH(z) + η−1Id
21:
accept x ←z with probability

1
2 · min
n exp(−f(z)) · (det(bΦ(z)))1/2 · exp(−0.5∥x −z∥2
bΦ(z))

exp(−f(x)) · (det(eΦ(x)))1/2 · exp(−0.5∥x −z∥2
eΦ(x))
, 1
o

22:
else
23:
reject z
24:
end if
25:
end for
26:
return x
27: end procedure

D.1
Bounding the density ratio

Lemma D.1 (Bounding the density ratio). Let x ∈Int(K). Suppose f is L-Lipschitz and η ≤
1/(10dL2), then

Pr
z∼N(x,Φ−1(x))

h exp(−f(z))

exp(−f(x)) ≥1

2

i
≥0.999.

If in addition, f is β-smooth and η ≤1/(10dβ), then we further have

Pr
z∼N(x,Φ−1(x))

hexp(−f(x))

exp(−f(z)) ≥1

2

i
≥0.499.

25


---Page Break---
Proof. Proof of Part 1. Since z ∼N(x, Φ(x)−1), we have that

z = x + Φ(x)−1/2ξ

= x + (α−1 · H(x) + η−1Id)−1/2

for some ξ ∼N(0, Id).

Since α−1H(x) + η−1Id ⪰η−1Id, and H(x) and Id are both positive definite, we have that

η · Id ⪰(α−1H(x) + η−1Id)−1
(9)

Thus, we can upper bound ∥z −x∥2 as follows:

∥z −x∥2 = ∥(α−1H(x) + η−1Id)−1/2ξ∥2

=
q

ξ⊤(α−1H(x) + η−1Id)−1ξ

≤
p

ξ⊤ηIdξ
= √η · ∥ξ∥2

where the third step follows from Eq. (9),

Recall that ξ is sampled from N(0, Id), using Lemma C.1, we can show

Pr[∥ξ∥2 > t] ≤exp(−(t2 −d)/8),
∀t >
√

2d.

Combining the above two equations, we have

Pr[∥z −x∥2 > √η
√

20d] ≤exp(−19

8 d) < 0.001.
(10)

Using η ≤1/(80dL2), we have

Pr[∥z −x∥2 > 1/(2L)] < 0.001.
(11)

Since f is L-Lipschitz, then we have

exp(−f(z))
exp(−f(x)) = exp(−(f(z) −f(x))) ≥exp(−L∥z −x∥2).

Therefore,

Pr
z∼N(x,Φ−1(x))

h exp(−f(z))

exp(−f(x)) ≥1/2
i
≥
Pr
z∼N(x,Φ−1(x))

h
exp(−L∥z −x∥2) ≥1/2
i

=
Pr
z∼N(x,Φ−1(x))

h
∥z −x∥2 ≤log(2)/L
i

≥
Pr
z∼N(x,Φ−1(x))

h
∥z −x∥2 ≤1/(2L)
i

≥0.999

where the last inequality holds by Eq. (11).

Proof of Part 2. Moreover, in the setting where f is differentiable and β-smooth, we have that, since
z −x is a multivariate Gaussian random variable,

Pr[(z −x)⊤∇f(x) ≤0] = 1

2.

If (z −x)⊤∇f(x) ≤0, we have that

f(z) −f(x) ≤(z −x)⊤∇f(x) + β∥z −x∥2
2
≤β∥z −x∥2
2.

26


---Page Break---
Therefore,

Pr
z∼N(x,eΦ−1(x))
[ π(z)

π(x) ≥1

2]

≥
Pr
z∼N(x,eΦ−1(x))
[ π(z)

π(x) ≥1

2
and (z −x)⊤∇f(x) ≤0] −
Pr
z∼N(x,eΦ−1(x))
[(z −x)⊤∇f(x) > 0]

≥
Pr
z∼N(x,eΦ−1(x))
[e−β∥z−x∥2
2 ≥1

2] −0.5

=
Pr
z∼N(x,eΦ−1(x))
[∥z −x∥2 ≤

p

log(2)
√β
] −0.5

≥0.999 −0.5
= 0.499,

where the last Inequality holds by Eq. (10) since η ≤
1
20dβ .

Lemma D.2. Let ∥x −z∥Φ(x) < 0.001. Let η ≤0.01/L2. Then we have

|f(x) −f(z)| ≤0.1

Proof. We have

|f(x) −f(z)| ≤L · ∥x −z∥2
≤L · √η · ∥x −z∥Φ(x)
≤0.1

where the second step follows from Φ(x) ⪰η−1Id, the last step follows from η ≤0.01/L2.

D.2
Bounding the determinant

Lemma D.3 (Bounding the determinant). Consider any x ∈Int(K), and ξ ∼N(0, Id). Let
z = x + (Φ(x))−1/2ξ. Then

Pr
ξ

h
log(det(Φ(z))) −log(det(Φ(x))) ≥−0.1
i
≥0.999.

Proof. First, note that

log(det(Φ(z))) −log(det(Φ(x))) = log( det(Φ(z))

det(Φ(x)))

= log( det(d · H(z) + dL2 · Id)

det(d · H(x) + dL2 · Id)))

= log( det(H(z) + L2 · Id)

det(H(x) + L2 · Id)),

thus, it suffices to consider the function F(x) = log(det(H(x) + L2 · Id)) and bound F(z) −F(x).
By Assumption (ii), we know that F(x) is convex.

Thus,

F(z) −F(x) ≥(z −x)⊤∇F(x).

We know that

z = x + (Φ(x))−1/2ξ

where ξ ∼N(0, Id).

Thus,

F(z) −F(x) ≥ξ⊤(Φ(x))−1/2∇F(x)

27


---Page Break---
By property of Gaussian distribution, we know that

ξ⊤(Φ(x))−1/2∇F(x)

is a Gaussian with mean 0 and variance ∥(Φ(x))−1/2∇F(x)∥2
2.

By definition of Φ and Assumption (iii) , we have

∥(H(x) + L2 · Id)−1/2∇F(x)∥2
2 = O(d)

and therefore

∥Φ(x)−1/2∇F(x)∥2
2 = O(1).

Thus, rescaling the function and applying the standard concentration inequality, we complete the
proof.

Lemma D.4 (Bounding the determinant, shifted). Let α < 0.001/d. For fixed x, z ∈K such that
∥z −x∥Φ(x) < 0.001. Then, we have

log(det(Φ(z))) −log(det(Φ(x))) ≥−0.01.

Proof. Let F(x) = log det(H(x) + L2 · Id). Then

log(det(Φ(z))) −log(det(Φ(x))) = F(z) −F(x) ≥(z −x)⊤∇F(x)

where the second step is because F(x) is convex (by Assumption (ii)).

We have

((z −x)⊤∇F(x))2 = ((z −x)⊤Φ(x)1/2 · Φ(x)−1/2∇F(x))2

≤∥(z −x)⊤Φ(x)1/2∥2 · ∥Φ(x)−1/2∇F(x)∥2

≤0.001 · ∥Φ(x)−1/2∇F(x)∥2
≤0.001 · 0.1.

where the second step is by Cauchy-Schwarz, the third step is by assumption ∥z −x∥Φ(x) ≤0.001,
the fourth step is by α < 0.001/d and Assumption (iii).

D.3
Bounding the local norm

Lemma D.5 (Bounding the local norm). Consider any x ∈Int(K), and ξ ∼N(0, Id). Let
η ≤0.001/d. Let z = x + (Φ(x))−1/2ξ. Then

Pr
ξ

h
|∥z −x∥2
Φ(z) −∥z −x∥2
Φ(x)| ≤0.01
i
≥0.999

Proof Sketch. Our proof is a generalization of [Sachdeva and Vishnoi, 2016, Proposition 7], where
they prove the result for non-regularized log-barrier. Note that their proof also works for regularized
barriers because two points are close in the Φ(x) norm indicates that they are close in the H(x) norm.
The only difference is that when computing the Gaussian polynomial as in Sachdeva and Vishnoi
[2016], we need to handle non-uniform weights instead of the uniform weights of log-barrier. This
could also be handled by observing that if two points are close in H(x) norm, then their weights are
close in the sense that ∥wp(x)−1(wp(x) −wp(z))∥∞≤cp where cp < 1 is some small constant that
depends on p (see [Lee and Sidford, 2019, Lemma 34]). One could then check that the argument
of [Sachdeva and Vishnoi, 2016, Proposition 7] is robust under small perturbation to the weights, and
conclude similar conclusions.

Lemma D.6 (Bounding the local norm, shifted). Consider any x ∈Int(K), and ξ ∼N(0, Id). Let
η ≤0.001/d. Let ∥z −x∥Φ(x) < 0.001. Let u = x + (Φ(x))−1/2ξ. Then

Pr
ξ

h
|∥u −z∥2
Φ(u) −∥u −x∥2
Φ(u)| ≤0.1
i
≥0.999

28


---Page Break---
Proof. Consider the following inequalities:

|∥u −z∥2
Φ(u) −∥u −x∥2
Φ(u)|

≤∥u −x∥2
Φ(u) + 2|⟨u −x, x −z⟩Φ(u)|

≤2∥u −x∥2
Φ(x) + 4|⟨u −x, x −z⟩Φ(x)|

= 2∥u −x∥2
Φ(x) + 4|w|

where the second step follows from Lemma B.2, in the last step w ∼N(0, ∥Φ(x)1/2(x −z)∥2
2).

First, using η is small, we can show ∥u −x∥2
Φ(x) < 0.01 with probability 0.9999.

Second, using ∥z −x∥Φ(x) < 0.001 and Gaussian concentration, we can show that

Pr[|w| < 0.01] ≥0.9999.

Put them together and union bound, we obtain the desired result.

Lemma D.7 (Bounding the local norm, shifted). Consider any x ∈Int(K), and ξ ∼N(0, Id). Let
η ≤0.001/d. Let ∥z −x∥Φ(x) < 0.001. Let u = x + (Φ(x))−1/2ξ. We have

Pr[|∥u −z∥2
Φ(z) −∥u −x∥2
Φ(u)| < 0.1] ≥0.998

Proof. Let u′ ∼z + Φ(z)−1/2ξ′ where ξ′ ∼N(0, Id). We have

|∥u −z∥2
Φ(z) −∥u −x∥2
Φ(u)|

≤|∥u −z∥2
Φ(z) −∥u′ −z∥2
Φ(z)| + |∥u′ −z∥2
Φ(z) −∥u′ −z∥2
Φ(u′)|

+ |∥u′ −z∥2
Φ(u′) −∥u −z∥2
Φ(u)| + |∥u −z∥2
Φ(u) −∥u −x∥2
Φ(u)|

So

Pr[|∥u −z∥2
Φ(z) −∥u −x∥2
Φ(u)| > 0.1]

≤Pr[|∥u′ −z∥2
Φ(z) −∥u′ −z∥2
Φ(u′)| > 0.05]
|
{z
}
Lemma D.5
+ Pr[|∥u −z∥2
Φ(u) −∥u −x∥2
Φ(u)| > 0.05]
|
{z
}
Lemma D.6

+ Pr[u ̸= u′]
|
{z
}
Lemma D.10
≤0.001 + 0.001 + 0.001 = 0.003.

where first step is by union bound, second step is by Lemma D.5, Lemma D.6, and by TV(u, u′) ≤
0.001 when ∥x −z∥Φ(x) ≤0.001 (Lemma D.10).

D.4
Bounding TV distance between exact processes

Recall that we define Px(S) = Pr[x ∈S]. We prove a bound between exact distributions of two
points x and z.
Lemma D.8 (TV distance between the exact distribution). For any x, z ∈K such that ∥x−z∥Φ(x) ≤
0.001, we have that

TV(Px, Pz) ≤0.99.

Proof. We define

Gx,Φ(x)(u) := (det(Φ(x)))1/2 · exp(−0.5∥u −x∥2
Φ(x)).

Then

TV(Px,Φ(x), Pz,Φ(z)) = 1 −pmin,

where

pmin :=
E
u∼N(x,Φ−1(x))[min{1, Gz(u)

Gx(u), exp(−f(u))Gu(x)

exp(−f(x))Gx(u), exp(−f(u))Gu(z)

exp(−f(z))Gz(u) · Gz(u)

Gx(u)}].

29


---Page Break---
For convenience, we define

a1 := Gz(u)

Gx(u)

a2 := exp(−f(u))Gu(x)

exp(−f(x))Gx(u)

a3 := exp(−f(u))Gu(z)

exp(−f(z))Gz(u) · Gz(u)

Gx(u)

Let θ ∈(0, 1). We have the following:

TV(Px, Pz) = 1 −pmin
= 1 −E[min{1, a1, a2, a3}]
= 1 −min{1, θ} · Pr[min{a1, a2, a3} ≥θ] −min{a1, a2, a3} · Pr[min{a1, a2, a3} < θ]
≤1 −min{1, θ} · Pr[min{a1, a2, a3} ≥θ]
≤1 −θ · Pr[min{a1, a2, a3} ≥θ]
≤1 −θ · θ0
≤1 −0.01 · 0.95
≤0.99

where the sixth step follows from Eq. (12), and the last step follows from choice of θ and θ0.

Let ai = e−bi. It suffices to prove that for some θ ∈(0, 1), we have

Pr[min{a1, a2, a3} ≥θ] ≥θ0.
(12)

Suppose for some τ ∈[1, 5] we can show that

Pr[b1 < τ] ≥θ1
Pr[b2 < τ] ≥θ2
Pr[b3 < τ] ≥θ3

Then by union bound, we can know that

Pr[max{b1, b2, b3} < τ]

= Pr[min{a1, a2, a3} > e−τ]
= Pr[min{a1, a2, a3} > θ]
≥1 −(1 −θ1) −(1 −θ2) −(1 −θ3)
≥0.95

where the second step follows from e−τ = θ ≥0.01, and the last step follows from θ1 = θ2 = θ3 =
0.99. In the following, we will establish the three bounds of interest.

Part 1. By definition, we have

a1 = Gz(u)

Gx(u)

= (det(Φ(z)))1/2

(det(Φ(x)))1/2 · exp(−0.5∥u −z∥2
Φ(z) + 0.5∥u −x∥2
Φ(x))

= exp(−0.5∥u −z∥2
Φ(z) + 0.5∥u −x∥2
Φ(x) + 0.5 log det(Φ(z)) −0.5 log(det(Φ(x))))

Using Lemma D.7 (on the term 0.5∥u −z∥2
Φ(z) −0.5∥u −x∥2
Φ(u)) and Lemma D.5 (on the term
0.5∥u −x∥2
Φ(u) −0.5∥u −x∥2
Φ(x)), we have

Pr
u [0.5∥u −z∥2
Φ(z) −0.5∥u −x∥2
Φ(u)
|
{z
}
Lemma D.7

+ 0.5∥u −x∥2
Φ(u) −0.5∥u −x∥2
Φ(x)
|
{z
}
Lemma D.5

≤0.1τ] ≥0.999,

30


---Page Break---
From Lemma D.4, we know

0.5 log(det(Φ(x))) −0.5 log(det(Φ(z))) ≤0.1τ.

Thus,

Pr[b1 ≤τ] ≥0.99

Part 2. By definition, for a2 we have

a2 = exp(−f(u))Gu(x)

exp(−f(x))Gx(u)

= exp(−f(u))

exp(−f(x)) · (det(Φ(u)))1/2

(det(Φ(x)))1/2 · exp(−0.5∥x −u∥2
Φ(u) + 0.5∥u −x∥2
Φ(x))

= exp(−f(u) + f(x) + 0.5 log(det(Φ(u))) −0.5 log(det(Φ(x))) −0.5∥x −u∥2
Φ(u) + 0.5∥u −x∥2
Φ(x))

By Lemma D.1, we have

Pr
u [f(u) −f(x) ≤0.1τ] ≥0.999

By Lemma D.3, we have

Pr
u [0.5 log(det(Φ(x))) −0.5 log(det(Φ(u))) ≤0.1τ] ≥0.999

By Lemma D.5, we have

Pr
u [0.5∥x −u∥2
Φ(u) −0.5∥u −x∥2
Φ(x) ≤0.1τ] ≥0.999

Then, combining the above three equations, we can get the following result:

Pr[b2 ≤τ] ≥0.99

Part 3.

a3 = exp(−f(u))Gu(z)

exp(−f(z))Gz(u) · Gz(u)

Gx(u)

= exp(−f(u))

exp(−f(z)) · (det(Φ(u)))1/2

(det(Φ(x)))1/2 · exp(−0.5∥z −u∥2
Φ(u) + 0.5∥u −x∥2
Φ(x))

= exp(−f(u) + f(z) + 0.5 log(det(Φ(u))) −0.5 log(det(Φ(x))) −0.5∥z −u∥2
Φ(u) + 0.5∥u −x∥2
Φ(x))

By Lemma D.1 (on the term f(u) −f(x)) and Lemma D.2 (on the term f(x) −f(z)), we have

Pr
u [f(u) −f(x)
|
{z
}
Lemma D.1

+ f(x) −f(z)
|
{z
}
Lemma D.2

≤0.1τ] ≥0.999.

By Lemma D.3, we have

Pr
u [0.5 log(det(Φ(x))) −0.5 log(det(Φ(u))) ≤0.1τ] ≥0.999.

By Lemma D.5 (on the term 0.5∥u −x∥2
Φ(u) −0.5∥u −x∥2
Φ(x)) and Lemma D.6 (on the term
0.5∥z −u∥2
Φ(u) −0.5∥u −x∥2
Φ(u)), we have

Pr
u [0.5∥z −u∥2
Φ(u) −0.5∥u −x∥2
Φ(u)
|
{z
}
Lemma D.6

+ 0.5∥u −x∥2
Φ(u) −0.5∥u −x∥2
Φ(x)
|
{z
}
Lemma D.5

≤0.2τ] ≥0.999

Thus, combining the above three equations, we obtain the following:

Pr[b3 ≤τ] ≥0.99.

31


---Page Break---
D.5
Bounding TV distance between approximate processes

Lemma D.9 (Robust version of Lemma E.10 in Mangoubi and Vishnoi [2023]). If δH < 0.001,
dα < 0.001, dϵH < 0.001 and for any x, z ∈K, ∥x −z∥Φ(x) < 0.001, then we have

TV( ePx, ePz) ≤0.99.

Proof. We can upper bound TV( ePx, ePy) as follows:

TV( ePx, ePy) ≤TV( ePx, Px) + TV(Px, Py) + TV(Py, ePy)

≤TV( ePx, Px) + 0.9 + TV(Py, ePy)
≤0.01 + 0.9 + 0.01
≤0.99

First step is by triangle inequality. Second step is by Lemma D.8. The third step is by Lemma C.7.

D.6
Bounding TV distance between Gaussians

Lemma D.10 (Lemma E.6 in Mangoubi and Vishnoi [2023]). For any x, z ∈K such that ∥x −
z∥Φ(x) ≤0.001, we have

TV(N(x, Φ−1(x)), N(z, Φ−1(z))) ≤
p

3dα + 1/2 · ∥x −z∥Φ(x).

Further, if αd < 0.001, then we have

TV(N(x, Φ−1(x)), N(z, Φ−1(z))) ≤0.001.

Lemma D.11 (Robust version of Lemma D.10). For any x, z ∈K such that ∥x −z∥Φ(x) ≤
1
4α1/2 ,
with probability at least 1 −1/1000, we have

TV(N(x, eΦ−1(x)), N(z, bΦ−1(z))) ≤
p

2dϵH +
p

3dα + 1/2 · ∥x −z∥Φ(x)

Proof. By triangle inequality,

TV(N(x, eΦ−1(x)), N(z, bΦ−1(z)))

≤TV(N(x, eΦ−1(x)), N(x, Φ−1(x)))

+ TV(N(x, Φ−1(x)), N(z, Φ−1(z)))

+ TV(N(z, Φ−1(z)), N(z, bΦ−1(z))).

Second term is bounded using Lemma D.10. First and third term are bounded using similar ways.
We have

TV(N(x, eΦ−1(x)), N(x, Φ−1(x)))2

≤1

2D(N(x, eΦ−1(x))∥N(x, Φ−1(x)))

= 1

4(tr[eΦ(x)−1/2Φ(x)eΦ(x)−1/2] −d + log det eΦ(x)

det Φ(x))

≤1

4((1 ± ϵH)d −d + log(1 ± ϵH)d)

≤1

2dϵH.

Thus, we complete the proof.

32


---Page Break---
D.7
Reversibility and stationary distribution

For any x ∈K, we define the random variable Zx to be the step taken by the Markov chain in
Algorithm 2 from the point x, that is, set z = x + eΦ(x)−1/2ξ where ξ ∼N(0, Id). If z ∈K, set
Zx = z with probability

1
2 · min
n exp(−f(z)) · (det(bΦ(z)))1/2 · exp(−0.5∥x −z∥2
bΦ(z))

exp(−f(x)) · (det(eΦ(x)))1/2 · exp(−0.5∥x −z∥2
eΦ(x))
, 1
o
.

Else, we set z = x.

We provide a modified version of Proposition E.12 in Mangoubi and Vishnoi [2023] where our
Markov chain differs from theirs.
Lemma D.12 (Reversibility and stationary distribution). For any S1, S2 ⊆K we have that
Z

x∈S1
π(x) ePx(S2)dx =
Z

y∈S2
π(y) ePy(S1)dy.

Proof. Let a, b be two i.i.d. random variables (one can view a and b as random coins to generate the
corresponding sparsifers) such that eΦa,x is a function of a, x and bΦb,y is a function of b, y.

Let pa,x(·) be pdf of N(0, eΦ−1
a,x) and pb,y(·) be the pdf of N(0, bΦ−1
b,y). Because a and b are i.i.d, they
are interchangeable, i.e., (a, b) has the same distribution as (b, a).

Z

x∈S1
π(x) ePx(S2)dx

=
Z

x∈S1
π(x)
Z

a
q(a)
Z

y∈S2
pa,x(y)
Z

b
q(b) · min{ π(y)pb,y(x)

π(x)pa,x(y), 1}db dy da dx

=
Z

a
q(a)
Z

b
q(b)
Z

x∈S1

Z

y∈S2
min{π(y)pb,y(x), π(x)pa,x(y)}dy dx db da

=
Z

a
q(a)
Z

b
q(b)
Z

x∈S1

Z

y∈S2
min{π(y)pa,y(x), π(x)pb,x(y)}dy dx db da

=
Z

y∈S2

Z

a
q(a)
Z

x∈S1

Z

b
q(b) min{π(y)pa,y(x), π(x)pb,x(y)}db dx da dy

=
Z

y∈S2
π(y) ePy(S1)dy.

The first and fifth steps are by definition of ePx. The second and fourth steps are by changing order of
integration. The third step is by interchangeability of a and b.

This proves reversibility and stationary distribution.

D.8
Lower bound on cross ratio distance

Definition D.13. We define

F(n, α, η, R) := 0.1/
p

nα−1 + η−1R2.

For simplicity, we use F to denote F(n, α, β, R).

Lemma D.14. Let S′
1 and S′
2 be defined as Definition C.8. Le F be defined as Definition D.13. We
have

σ(S′
1, S′
2) > 1000F.

Proof. Using Lemma D.17, we have that for any u, v ∈K,

σ(u, v) ≥F · ∥u −v∥Φ(u).

33


---Page Break---
We will try to prove σ(u, v) > 1000F in the following. To do that, we will prove it by making a
contradiction.

Suppose σ(u, v) ≤1000F, then we have to require ∥u −v∥Φ(u) ≤0.001.

Once we had that ∥u −v∥Φ(u) ≤0.001, using Lemma D.9, for any u, v ∈K we have that

TV( ePu, ePv) ≤0.99
(13)

On the other hand, Definition C.8 implies that, for any u ∈S′
1, v ∈S′
2 we have that

TV( ePu, ePv) ≥1 −ePu(S2) −ePv(S1) ≥1 −2β > 0.99.
(14)

where the last step follows from 1 −2β > 0.99.

Thus, we obtain a contradiction, which means that

σ(S′
1, S′
2) ≥σ(u, v) > 1000F.

D.9
Bounding cross ratio distance through ν-symmetry

In Lemma E.2 of Mangoubi and Vishnoi [2023], they only prove the cross ratio distance bound for
log-barrier function. Here, we generalize it to arbitrary barrier function. The key concept our proof
relies on is the notion of ν-symmetry (Assumption (i)).
Lemma D.15. Let H ∈K →Rd×d be a ν-symmetric function, and u, v ∈K. Consider the chord
that passes through u, v and intersects with K. Let p, q be the two endpoints of the chord, with the
order p, u, v, q. Then, we have

∥p −u∥H(u) ≤
√

ν.

Proof. Suppose we can show that p ∈K ∩(2u −K), then it naturally follows that p ∈Eu(
√

ν) and
∥p −u∥H(u) ≤
√

ν.

p ∈K trivially follows by the construction of the chord. To see p ∈2u −K, it is enough to show that
for some y ∈K, we have p = 2u −y. We claim y = 2u −p is such a choice. To see 2u −p ∈K,
we need to show that 2Au −Ap ≤b. We partition the constraints into two sets.

Case 1. Suppose for ai, we have a⊤
i u ≤a⊤
i p, then 2a⊤
i u ≤2a⊤
i p ≤a⊤
i p + bi, therefore,
2a⊤
i u −a⊤
i p ≤bi.

Case 2. Suppose otherwise, a⊤
i u > a⊤
i p, then consider

2a⊤
i u −a⊤
i p −bi = (a⊤
i u −a⊤
i p) + (a⊤
i u −bi)

= (a⊤
i u −bi) −(a⊤
i p −bi)
|
{z
}
d1

−(bi −a⊤
i u)
|
{z
}
d2

.

We shall show that d1 ≤d2 via a geometric argument. Consider the projection of both p and u onto
the hyperplane aix = bi, and the two chords that pass through p and u respectively, orthogonal to
the hyperplane. Let us denote them with lp, lu respectively. Observe that lp and lu are parallel, and
∥lp −lu∥2 = (a⊤
i u −bi) −(a⊤
i p −bi) = d1. It should then be obvious that d1 ≤d2: let proj(u)
denote the orthogonal projection of u onto the hyperplane, then

∥lu −proj(u)∥2
|
{z
}
d2

= ∥lu −lp∥2
|
{z
}
d1

+∥lp −proj(p)∥2

since ∥lp −proj(p)∥2 ≥0, we have that d1 ≤d2, and consequently,

2a⊤
i u −a⊤
i p ≤bi.

This proves that 2u −p ∈K, and we can conclude that p ∈2u −K, which yields our desired
result.

Remark D.16. The above argument can be extended to spectrahedra and more general convex sets.

34


---Page Break---
Let u, v ∈K be two arbitrary points, and consider the chord that passes through u, v with two
endpoints being p, q. Note that p, q ∈∂K.

Lemma D.17. For any u, v ∈K, let H be any matrix function satisfying Assumption (i), (ii) and (iii).
We assume that K is contained in ball of radius R and has nonempty interior.

For any parameter α ∈(0, 1), η ∈(0, 1), we define matrix function Φ(u) := α−1 · H(u) + η−1Id.

Then, we have

σ(u, v) ≥
1
p

2να−1 + η−1R2 · ∥u −v∥Φ(u)

Proof. Without loss of generality, we can assume that

∥p −u∥2 ≤∥u −q∥2.
(15)

We can lower bound σ2(u, v) as follows:

σ2(u, v) = (∥u −v∥2
2 · ∥p −q∥2
2
∥p −u∥2
2 · ∥v −q∥2
2
)

≥max{∥u −v∥2
2
∥p −u∥2
2
, ∥u −v∥2
2
∥v −q∥2
2
}

≥max{∥u −v∥2
2
∥p −u∥2
2
, ∥u −v∥2
2
∥u −q∥2
2
}

≥max{∥u −v∥2
2
∥p −u∥2
2
, ∥u −v∥2
2
∥u −q∥2
2
, ∥u −v∥2
2
∥p −q∥2
2
}

= max{∥u −v∥2
2
∥p −u∥2
2
, ∥u −v∥2
2
∥p −q∥2
2
}

≥1

2
∥u −v∥2
2
∥p −u∥2
2
+ 1

2
∥u −v∥2
2
∥p −q∥2
2

≥1

2
∥u −v∥2
H
∥p −u∥2
H
+ 1

2
∥u −v∥2
2
R2

≥1

2
∥u −v∥2
H
ν
+ 1

2
∥u −v∥2
2
R2

= (u −v)⊤(
1
2να−1 × α−1H +
1
2R2η−1 × η−1Id)(u −v)

≥
1
2να−1 + 2η−1R2 ∥u −v∥2
Φ(u)

where the second step is by ∥p −q∥2 ≥max{∥p −u∥2, ∥v −q∥2}, the third step is by ∥v −q∥2 ≤
∥u−q∥2, the fourth step is by ∥p−q∥2 ≥∥p−u∥2, the fifth step follows from ∥p−u∥2 ≤∥u−q∥2
(see Eq. (15)), sixth step is by max{A, B} ≥0.5 · (A + B), where the seventh step follows from
∥p −q∥2 ≤R and Eq. (16). The eighth step is due to Lemma D.15.

It remains to show Eq. (16). Due to the fact that p, u, v are on the same chord, vector u −v and p −u
are on the same direction. This means we can write u −v = c · (p −u) for some c ∈R, and

∥u −v∥H
∥p −u∥H
= ∥c · (p −u)∥H

∥p −u∥H
= c

= ∥u −v∥2

∥p −u∥2
.
(16)

This completes the proof.

35


---Page Break---
E
Computing Approximate Lewis Weights and Leverage Scores

In Section E.1, we present an algorithm to approximate the Hessian. In Section E.2, we show how to
compute leverage score to low precision. In Section E.3, we show how to compute Lewis weights to
high precision.

E.1
Subsampling to approximate the Hessian

Algorithm 3 Subsampling Algorithm

1: procedure SUBSAMPLE(A ∈Rn×d, b ∈Rn, x, n, d, ϵH ∈(0, 1))
2:
ϵσ ←0.001
3:
m ←ϵ−2
H d log d
4:
Form B ∈Rn×d where i-th row of B is ai · (⟨ai, x⟩−bi)−1

5:
Compute the O(1 ± ϵσ)-approximation to the leverage score of B
6:
Sample a matrix eB ∈Rm×d such that (1 −ϵH)B⊤B ⪯eB⊤eB ⪯(1 + ϵH)B⊤B
7:
return eB⊤eB
8: end procedure

In this section, we provide a sparsification result for H(w) using leverage score sampling.

To better monitor the whole process, it is useful to write H(w) as A⊤S(w)−2A, where A ∈Rn×d

is the constraint matrix and S(w) is a diagonal matrix with S(w)i = a⊤
i w −bi. The sparsification
process is then sample the rows from the matrix S(w)−1A.

Definition E.1. Let B ∈Rn×d be a full rank matrix. We define the leverage score of the i-th row of
B as

σi(B) := b⊤
i (B⊤B)−1bi,

where bi is the i-th row of B.
Definition E.2 (Sampling process). For any w ∈K, let H(w) = A⊤S(w)−2A.
Let pi ≥
β·σi(S−1(w)A)

d
, suppose we sample with replacement independently for s rows of matrix S(w)−1A,
with probability pi of sampling row i for some β ≥1. Let i(j) denote the index of the row sampled in
the j-th trial. Define the generated sampling matrix as

eH(w) := 1

s

s
X

j=1

1
pi(j)

ai(j)a⊤
i(j)
(a⊤
i(j)w −bi(j))2 .

Lemma E.3 (Sample using Matrix Chernoff). Let ϵ, δ ∈(0, 1) be precision and failure probability
parameters, respectively. Suppose eH(w) is generated as in Definition E.2, then with probability at
least 1 −δ, we have

(1 −ϵ) · H(w) ⪯eH(w) ⪯(1 + ϵ) · H(w).

Moreover, the number of rows s = Θ(β · ϵ−2d log(d/δ)).

Proof. The proof will be designing a family of random matrices X.
Let yi
=
(A⊤S(w)−2A)−1/2S(w)−1
i,i · ai be the i-th sampled row and set Yi =
1
pi yiy⊤
i . Let Xi = Yi −Id.
Note that

n
X

i=1
yiy⊤
i =

n
X

i=1
(A⊤S(w)−2A)−1/2S(w)−2
i,i · aia⊤
i (A⊤S(w)−2A)−1/2

= (A⊤S(w)−2A)−1/2(

n
X

i=1
S(w)−2
i,i aia⊤
i )(A⊤S(w)−2A)−1/2

= (A⊤S(w)−2A)−1/2(A⊤S(w)−2A)(A⊤S(w)−2A)−1/2

= Id.
(17)

36


---Page Break---
Also, the norm of yi connects directly to the leverage score:

∥yi∥2
2 = S(w)−1
i,i a⊤
i (A⊤S(w)−2A)−1S(w)−1
i,i ai

= σi(S−1A).
(18)

We use i(j) to denote the index of row that has been sampled during j-th trial.

Unbiased Estimator. Note that

E[X] = E[Y ] −Id

= (

n
X

i=1
pi · 1

pi
yiy⊤
i ) −Id

= 0.

Bound on ∥X∥. To bound ∥X∥, we provide a bound for any ∥Xi∥as follows:

∥Xi∥= ∥Yi −Id∥
≤1 + ∥Yi∥

= 1 + ∥yiy⊤
i ∥
pi

≤1 +
∥yi∥2
2
β · σi(S−1(w)A)

=
d
β · σi(S−1(w)A)∥(S(w)A)i∥2
2 + 1.

= 1 + d

β .

Bound on ∥E[X⊤X]∥. We compute the spectral norm of the covariance matrix:

E[X⊤
i(j)Xi(j)] = Id + E[
yi(j)y⊤
i(j)yi(j)y⊤
i(j)
p2
i
] −2 E[
yi(j)y⊤
i(j)
pi
]

= Id + (

n
X

i=1

σi(S(w)−1A)

pi
yiy⊤
i ) −2Id

≤

n
X

i=1

d
β yiy⊤
i −Id

= ( d

β −1)Id,

the spectral norm is then

∥E[X⊤
i(j)Xi(j)]∥≤d

β −1.

Put things together. Set γ = 1 + d

β and σ2 =
d
β −1, we apply Matrix Chernoff Bound as in
Lemma A.8:

Pr[∥W∥≥ϵ] ≤2d · exp

−
sϵ2

d/β −1 + (1 + d/β)ϵ/3



= 2d · exp(−sϵ2 · Θ(β/d))
≤δ

where we choose s = Θ(β · ϵ−2d log(d/δ)). Finally, we notice that

W = 1

s(

s
X

j=1

1
pi(j)
yi(j)y⊤
i(j) −Id)

37


---Page Break---
= (A⊤S(w)−2A)−1/2(1

s

s
X

j=1

1
pi(j)

ai(j)a⊤
i(j)
(a⊤
i(j)w −bi(j))2 )(A⊤S(w)−2A)−1/2 −Id

= H(w)−1/2 eH(w)H(w)−1/2 −Id.

Therefore, we can conclude the desired result via ∥W∥≥ϵ.

E.2
Computing leverage score

In this section, we provide an algorithm that approximates leverage scores in near-input sparsity time.
Our algorithm makes use of the sparse embedding matrix of Nelson and Nguyên [2013] that has
O(log(n/ϵ)) nonzero entries per column.

Lemma E.4 (Approximate leverage scores). Let A ∈Rn×d, there exists an algorithm that runs in
time eO(ϵ−1(nnz(A) + ϵ−1dω)) that outputs a vector ew2(A) ∈Rn, such that ew2(A) ≈ϵ w2(A).

Proof. We follow an approach of Woodruff [2014], but instead we use a high precision sparse
embedding matrix Nelson and Nguyên [2013] and prove a two-sided bound on leverage score.

Let S be a sparse embedding matrix with r = O(ϵ−2d poly log(d/(ϵδ))) rows and each column has
O(ϵ−1 log(nd/ϵ)) nonzero entries. We first compute SA in time eO(ϵ−1 nnz(A)), then compute the
QR decomposition SA = QR in time eO(ϵ−2dω). Note that R ∈Rd×d hence R−1 can be computed
in O(dω) time.

Now, let G ∈Rd×t matrix with t = O(ϵ−2 log(n/δ)), each entry of G is i.i.d. N(0, 1/t) random
variables. Set qi = ∥e⊤
i AR−1G∥2
2 for all i ∈[n]. We argue qi is a good approximation to (w2(A))i.

First, with failure probability at most δ/n, we have that qi ≈ϵ ∥e⊤
i AR−1∥2
2 via Johnson-Lindenstrauss
lemma [Johnson and Lindenstrauss, 1984]. Now, it suffices to argue that ∥e⊤
i AR−1∥2
2 approximates
∥e⊤
i U∥2
2 well, where U ∈Rn×d is the left singular vectors of A. To see this, first observe that for
any x ∈Rd,

∥AR−1x∥2
2 = (1 ± ϵ) · ∥SAR−1x∥2
2
= (1 ± ϵ) · ∥Qx∥2
2
= (1 ± ϵ) · ∥x∥2
2,

where the last step is due to Q has orthonormal columns. This means that all singular values of AR−1
are in the range [1−ϵ, 1+ϵ]. Now, since U is an orthonormal basis for the column space of A, AR−1
and U has the same column space (since R is full rank). This means that there exists a change of
basis matrix T ∈Rd×d with AR−1T = U. Our goal is to provide a bound on all singular values of
T. For the upper bound, we claim the largest singular value is at most 1 + 2ϵ, to see this, suppose for
the contradiction that the largest singular is larger than 1 + 2ϵ and let v be its corresponding (unit)
singular vector. Since the smallest singular value of AR−1 is at least 1 −ϵ, we have

∥AR−1Tv∥2
2 ≥(1 −ϵ)∥Tv∥2
2
> (1 −ϵ)(1 + 2ϵ)
> 1,

however, recall AR−1T = U, therefore ∥AR−1Tv∥2
2 = ∥Uv∥2
2 = ∥v∥2
2 = 1, a contradiction. One
can similarly establish a lower bound of 1 −2ϵ. Hence, the singular values of T are in the range of
[1 −2ϵ, 1 + 2ϵ]. This means that

∥e⊤
i AR−1∥2
2 = ∥e⊤
i UT −1∥2
2
= (1 ± 2ϵ)∥e⊤
i U∥2
2
= (1 ± 2ϵ)(w2(A))i,

as desired. Scaling ϵ to ϵ/2 yields the approximation result.

Now, regarding the running time of computing qi, note that we can first multiply R−1 with G in time
eO(ϵ−2d2), this gives a matrix of size d × t. Multiplying this matrix with A takes eO(ϵ−1 nnz(A))
time. Hence, the overall time for computing ew2(A) = q ∈Rn is eO(ϵ−1 nnz(A) + ϵ−2dω).

38


---Page Break---
Remark E.5. It suffices to choose ϵ =
1
10 for generating approximate leverage scores and subsam-
pling the target matrix H(x).

E.3
Computing Lewis weights

Before stating the main technical tool for this section, we want to make several remarks regarding the
Lewis weights.

Since its introduction by Cohen and Peng [Cohen and Peng, 2015] as an algorithmic tool for ℓp row
sampling, Lewis weights, as its natural formulation follows from a convex program, is not known to
be solved exactly in polynomial time. All known algorithms [Cohen and Peng, 2015, Lee and Sidford,
2019, Fazel et al., 2022, Cohen et al., 2019, Jambulapati et al., 2022] can only compute ϵ-approximate
Lewis weights.

Definition E.6 (ℓp Lewis weights). Let A ∈Rn×d. The ℓp Lewis weights of A is a vector wp(A) ∈
Rn satisfying

wp(A) = σ(Wp(A)
1
2 −1

p A),

where Wp(A) ∈Rn×n is the diagonal matrix that puts wp(A) on the diagonal, σ : Rn×d →Rn is
the operation that outputs all leverage scores of input matrix.

We are now posed to state a main tool from Lee and Sidford [2019].

Theorem E.7 (Theorem 46 in Lee and Sidford [2019]). Let P = {x : Ax > b} denote the interior of
non-empty polytope for non-degenerate A ∈Rn×d. There is an O(d log5 n)-self concordant barrier
ψ defined by ℓp Lewis weight with p = Θ(log n) satisfying

A⊤
x WxAx ⪯∇2ψ(x) ⪯(q + 1)A⊤
x WxAx

where Ax = diag(Ax −b) and wx is the ℓp Lewis weight of the matrix Ax. Furthermore, we can
compute or update the wx, ∇ψ(x) and ∇2ψ(x) as follows:

• Initial Weight: For any x ∈Rd, one can compute a vector ewx such that (1 −ϵ)wx ≤ewx ≤
(1 + ϵ)wx in O(ndω−1/2 · log3 n log(n/ϵ)) time.

• Update Weight and Compute Gradient/Hessian: Given a vector ewx such that ewx = (1 ±
1
100)wx for any y with ∥x −y∥A⊤
x WxAx ≤
c
log2 n for some small constant c > 0, we can
compute ewy, v and H such that ewy = (1 ± ϵ)wy,

∥v −∇ψ(x)∥∇2ψ(x)−1 ≤ϵ,
and (1 −ϵ)∇2ψ(x) ⪯H ⪯(1 + ϵ)∇2ψ(x)

in O(ndω−1 · log n · log(n/ϵ)) time.

F
Fast Approximate Sampling from Polytopes: Complexity

In this section, we provide the runtime analysis for sampling from polytopes via log-barrier and
Lee-Sidford barrier. The result for log-barrier is in Section F.1 and for Lee-Sidford barrier is in
Section F.2.

F.1
Log-barrier in nearly-linear time

Throughout this section, we let α ∈(0, 1/(105d)) and η ∈(0, 1/(20dL2)).

Lemma F.1. If g is log-barrier function with ν = n, then each iteration of Algorithm 2 can be
implemented in O(nnz(A) + Tmat(d, d3, d)) time plus O(1) calls to the oracle of f.

Proof. We go through each step of Algorithm 2 and add up the time and oracle calls for each step:

We first need to sample a d-dimensional Gaussian random vector ξ ∼N(0, Id), which can be
performed in O(d) time.

39


---Page Break---
At each iteration, by the definition of log-barrier function, we can write the Hessian as follows:

H(x) :=

n
X

j=1

aja⊤
j
(a⊤
j x −bj)2 .

We define define matrix B ∈Rn×d as follows
B := S−1A

where si = (⟨ai, x⟩−bi), ∀i ∈[n]. Then we can write H(x) = B⊤B ∈Rd×d.

For matrix B, we can compute a constant approximation of its leverage score, then we can sample
according to leverage, and generate a diagonal matrix D such that

(1 −ϵ)B⊤B ⪯B⊤DDB ⪯(1 + ϵ)B⊤B.

This step takes eO(nnz(A) + dω(1,3,1)) time.

Computing eΦ(x) can be done in d2 time and computing the vector z

z = x + eΦ(x)−1

2 ξ
requires to invert the matrix Φ(x), taking the square root and multiplying with a vector ξ. Inversion
and square root can be performed in time O(dω) by computing its spectral decomposition, and the
matrix-vector product can be done in time O(d2). Hence, the overall time of this step is O(dω).

To determine whether z ∈K, one can simplify verify Az ≤b, in time O(nnz(A)).

We also need to compute bH(z) and bΦ(z). This is similar to computing eH(x) and eΦ(x).

We then need to compute the determinant det(eΦ(x)) and det(bΦ(z)) which can be done in O(dω)
time via a spectral decomposition.

We then need O(1) oracle queries to the function values of f.

Therefore, adding up the number of arithmetic operations and oracle calls from all the different steps of
Algorithm 2, we get that each iteration of Algorithm 2 can be computed in eO(nnz(A)+Tmat(d, d3, d))
arithmetic operations plus O(1) calls to the oracle of f.

Remark F.2. We note a recent work Mangoubi and Vishnoi [2024] has provided an algorithm that
runs in nnz(A) + d2 time per iteration, by adapting techniques from Laddha et al. [2020]. Their
per iteration cost improvement only works for log-barrier, and they could only manage to obtain an
eO(nd + dL2R2) mixing time. Our framework on the other hand generalizes to other barriers such
as Lee-Sidford barrier, which provides a nearly-optimal mixing rate. We also adapt the framework to
sampling from spectrahedra.

F.2
Lee-Sidford barrier via approximation scheme

Lemma F.3. If g is the Lee-Sidford barrierfunction with ν = d log5 n, then, each iteration of
Algorithm 2 can be implemented in eO(ndω−1) time plus O(1) calls to the oracle of f.

Proof. For Lee-Sidford barrier, it suffices to use H(x) = A⊤
x WxAx as where Wx is the ℓp Lewis
weights for p = Θ(log m). However, as Lewis weights cannot be computed exactly, we shall use the
algorithm of Lee and Sidford [2019] to compute an ϵ-approximation.

To invoke Theorem E.7, we need to make sure that z and x satisfy ∥z −x∥A⊤
x WxAx ≤
c
log2 n. We

can achieve so by scaling down α by a factor of log2 n. This only blows up the convergence by eO(1)
factor, so it is acceptable.

The runtime analysis is then similar to Lemma F.1. Computing the initial Lewis weights takes
eO(ndω−1/2 log(1/ϵ)) time owing to Theorem E.7. As the algorithm requires eO(d2) iterations, this
time can be amortized. For each iteration, it needs to query the Lewis weights data structure that
outputs an ϵ-Lewis weights in eO(ndω−1 log(1/ϵ)) time. As we will choose ϵ as O(1/d), this time
is eO(ndω−1). We can then use this approximate Hessian to progress the algorithm. All subsequent
operations take O(dω) time. Thus, the cost per iteration is eO(ndω−1).

40


---Page Break---
G
Approximate Sampling from a Spectrahedron

We present an algorithm that efficiently and approximately samples from a spectrahedron, which is a
popular convex body utilized by semidefinite programs.

G.1
Definitions

Definition G.1. Let A1, . . . , Ad ∈Rn×n be a collection of symmetric matrices and C ∈Rn×n be
symmetric. We define the corresponding spectrahedron as

K = {x ∈Rd :

d
X

i=1
xiAi ⪰C}.

We define the log-barrier for spectrahedron and its corresponding Hessian.
Definition G.2 (Nesterov and Nemirovskii [1994]). Let K be a spectrahedron described by symmetric
matrices {A1, . . . , Ad} ⊆Rn×n and C ∈Rn×n. The log barrier ϕlog(x) is defined as

ϕlog(x) = −log det(S(x))

and its corresponding Hessian is defined as

Hlog(x) = A(S(x)−1 ⊗S(x)−1)A⊤,

where S(x) := Pd
i=1 xiAi −C ∈Rn×n and A ∈Rd×n2 and the i-th row of A is the vectorization
of Ai ∈Rn×n.

To compute Hlog(x), it is handy to define a matrix B ∈Rd×n2.

Definition G.3. We define a matrix B ∈Rd×n2 as B = A(S(x)−1/2 ⊗S(x)−1/2). Consequently,
Hlog(x) = BB⊤.

We state a useful lemma for computing the matrix B.

Lemma G.4. Let matrix B ∈Rd×n2 be defined as in Def. G.3. Then, the i-th row of B can be
computed as vec(S(x)−1/2AiS(x)−1/2).

Proof. The proof relies on a simple fact of Kronecker product and vectorization:

vec(S(x)−1/2AiS(x)−1/2) = (S(x)−1/2 ⊗S(x)−1/2) vec(Ai),

which is the definition of the i-th row of B.

G.2
n-symmetry of Hlog for spectrahedra

In this section, we prove that Hlog is n-symmetry for spectrahedra.
Lemma G.5. Hlog(x) is n-symmetry, that is, for any x ∈K,

Ex(1) ⊆K ∩(2x −K) ⊆Ex(√n).

Proof. Pick a point y ∈Ex(1), then we know that (y −x)⊤A(S(x)−1 ⊗S(x)−1)A⊤(y −x) ≤1.
We note that the set K ∩(2x −K) can be characterized as the set of all points u ∈Rd such that

d
X

i=1
uiAi ⪰C,

d
X

i=1
(2x −u)iAi ⪰C.

These two conditions imply

−S(x) ⪯

d
X

i=1
(ui −xi)Ai ⪯S(x),

41


---Page Break---
−In ⪯

d
X

i=1
(ui −xi)S(x)−1/2AiS(x)−1/2 ⪯In.

We will prove y satisfies these conditions. Note that y ∈Ex(1) implies that

∥

d
X

i=1
(yi −xi) vec(S(x)−1/2AiS(x)−1/2)∥2
2 ≤1,

∥

d
X

i=1
(yi −xi)S(x)−1/2AiS(x)−1/2∥2
F ≤1

if we let M denote Pd
i=1(yi −xi)S(x)−1/2AiS(x)−1/2, then the above condition implies that
M ⊤M ⪯In, meaning all the eigenvalues of M lie between [−1, 1], thus we have shown that
Ex(1) ⊆K ∩(2x −K).

For the other direction, we know that

−In ⪯

d
X

i=1
(yi −xi)S(x)−1/2AiS(x)−1/2 ⪯In,

and henceforth

(y −x)⊤A(S(x)−1 ⊗S(x)−1)A⊤(y −x)

= ∥

d
X

i=1
(yi −xi)S(x)−1/2AiS(x)−1/2∥2
F

≤∥In∥2
F
= n,

where we use the fact that −In ⪯M ⪯In hence all eigenvalues of M lie between [−1, 1], and the
squared eigenvalues of M have its magnitude at most 1. We then use the squared Frobenius norm is
equivalent to squared ℓ2 norm of singular values of M, as desired.

G.3
Bounded local norm of Hlog(x)

We prove that Hlog(x) has the bounded local norm property.

Lemma G.6. Let Hlog(x) ∈Rd×d be defined as in Def. G.2. Let F(x) = log det(Hlog(x)). Then,
F(x) is convex in x.

Proof. The proof is by observe that the function F(x) is actually the volumetric barrier function
for the SDP spectrahedron. Thus, the function F is convex, for more details, see Nesterov and
Nemirovskii [1994].

Lemma G.7. Let F(x) be defined as in Lemma G.6. Then, we have

∇F(x) =





−tr[P((S(x)−1/2A1S(x)−1/2) ⊗S (S(x)−1/2A1S(x)−1/2))]
−tr[P((S(x)−1/2A2S(x)−1/2) ⊗S (S(x)−1/2A2S(x)−1/2))]
...
−tr[P((S(x)−1/2AmS(x)−1/2) ⊗S (S(x)−1/2AmS(x)−1/2))]





where P = B⊤H(x)−1B.

Proof. For simplicity, we let H to denote Hlog. For each coordinate j ∈[d], we have

∂F
∂xj
= tr[H(x)−1 ∂H(x)

∂xj
]

= tr[H(x)−1A∂(S(x)−1 ⊗S(x)−1)

∂xj
A⊤]

42


---Page Break---
= tr[H(x)−1A(∂S(x)−1

∂xj
⊗S(x)−1 + S(x)−1 ⊗∂S(x)−1

∂xj
)A⊤]

= −tr[H(x)−1A((S(x)−1 ∂S(x)

∂xj
S(x)−1) ⊗S(x)−1 + S(x)−1 ⊗(S(x)−1 ∂S(x)

∂xj
S(x)−1))A⊤]

= −tr[H(x)−1A((S(x)−1AjS(x)−1) ⊗S(x)−1 + S(x)−1 ⊗(S(x)−1AjS(x)−1))A⊤],

examine the following term:

tr[H(x)−1A((S(x)−1AjS(x)−1) ⊗S(x)−1)A⊤]

= tr[A⊤H(x)−1A((S(x)−1Aj) ⊗I)(S(x)−1 ⊗S(x)−1)]

= tr[(S(x)−1/2 ⊗S(x)−1/2)A⊤H(x)−1A(S(x)−1/2 ⊗S(x)−1/2)

· ((S(x)−1/2Aj) ⊗S(x)1/2)(S(x)−1/2 ⊗S(x)−1/2)]

= tr[P((S(x)−1/2Aj) ⊗S(x)1/2)(S(x)−1/2 ⊗S(x)−1/2)]

= tr[P((S(x)−1/2AjS(x)−1/2) ⊗In)],

where P ∈Rn2×n2 is the projection of B⊤, i.e., P = B⊤H(x)−1B. Thus,

∇F(x) =





−tr[P((S(x)−1/2A1S(x)−1/2) ⊗S (S(x)−1/2A1S(x)−1/2))]
−tr[P((S(x)−1/2A2S(x)−1/2) ⊗S (S(x)−1/2A2S(x)−1/2))]
...
−tr[P((S(x)−1/2AmS(x)−1/2) ⊗S (S(x)−1/2AmS(x)−1/2))]



.

This completes the proof.

Lemma G.8. The matrix function H(x) = A(S−1 ⊗S−1)A⊤∈Rd×d has the following bounded
norm property: for any direction h ∈Rd,

∥H(x)−1/2DH(x)[h]H(x)−1/2∥2
F ≤4d∥h∥2
H(x).

Proof. Let xt = x + t · h for some fixed vector h ∈Rd, St = Pm
i=1 xt,iAi −C ∈Rn×n,
At = A(S−1/2
t
⊗S−1/2
t
) ∈Rd×n2, Pt = A⊤
t (AtA⊤
t )−1At ∈Rn2×n2 and Σt = Pn
i=1(e⊤
i ⊗
In)Pt(ei ⊗In) = Pn
i=1(In ⊗e⊤
i )Pt(In ⊗ei). Note that for any matrix M ∈Rn×n, tr[Pt · (M ⊗
In)] = tr[Pt · (In ⊗M)] = tr[Σt · M]. Also note that tr[Σt] = tr[Pt] = d.

Let Ht = A⊤(S−1
t
⊗S−1
t
)A ∈Rd×d. We have

∥H−1/2
t
( ∂

∂tHt)H−1/2
t
∥2
F

= tr[H−1
t
· ( ∂
∂tHt) · H−1
t
· ( ∂
∂tHt)]

= tr[H−1
t
· A⊤∂S−1
t
⊗S−1
t
∂t
A · H−1
t
A⊤∂(S−1
t
⊗S−1
t
)
∂t
A]

= tr[AH−1
t
A⊤· ∂S−1
t
⊗S−1
t
∂t
· AH−1
t
A⊤· ∂(S−1
t
⊗S−1
t
)
∂t
]

= tr[Pt(S−1/2
t
⊗S−1/2
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1/2
t
⊗S−1/2
t
)−1

· Pt(S−1/2
t
⊗S−1/2
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1/2
t
⊗S−1/2
t
)−1]

≤tr[Pt · (S−1/2
t
⊗S−1/2
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1
t
⊗S−1
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1/2
t
⊗S−1/2
t
)−1]

= tr[Pt · (S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
⊗S In)2]

43


---Page Break---
= tr[Pt · ((S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
)2 ⊗S In + (S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
⊗S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
))]

= 2 tr[S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
· Σt · S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
]

+ tr[Pt · (S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
⊗S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
)]

≤4 tr[S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
· Σt · S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
]

≤4∥Σt∥tr[(S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
)2]

≤4 tr[Σt] tr[(S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
)2]

= 4d∥h∥2
Ht,

the fifth step is by for PSD matrix M, we have MPM ⪯M 2 where P is a projection matrix, the
sixth step is by the below derivation, the seventh step is by tr[Pt · (M 2 ⊗S In)] = 2 tr[MΣtM], the
eighth step is by for PSD matrix M, 2M ⊗S In ⪰M ⊗M, the ninth step is by Hölder’s inequality
where ⟨A, B⟩≤∥A∥· ∥B∥s1 for ∥B∥s1 be its Schatten-1 norm. If B is PSD, ∥B∥1 = tr[B], the
tenth step is by for PSD matrix A, ∥A∥≤tr[A].

Note that
∂S−1
t
∂t
= −S−1
t
∂St

∂t S−1
t

= −S−1
t
(

d
X

i=1
hiAi)S−1
t

Thus,

∂S−1
t
⊗S−1
t
∂t
= −S−1
t
(

d
X

i=1
hiAi)S−1
t
⊗S−1
t

−S−1
t
⊗S−1
t
(

d
X

i=1
hiAi)S−1
t

and we have

(S−1/2
t
⊗S−1/2
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1
t
⊗S−1
t
)−1 ∂S−1
t
⊗S−1
t
∂t
(S−1/2
t
⊗S−1/2
t
)−1

= (S−1/2
t
⊗S−1/2
t
)−1(S−1
t
(

d
X

i=1
hiAi)S−1
t
⊗S−1
t
+ S−1
t
⊗S−1
t
(

d
X

i=1
hiAi)S−1
t
)(S−1/2
t
⊗S−1/2
t
)−1

· (S−1/2
t
⊗S−1/2
t
)−1(S−1
t
(

d
X

i=1
hiAi)S−1
t
⊗S−1
t
+ S−1
t
⊗S−1
t
(

d
X

i=1
hiAi)S−1
t
)(S−1/2
t
⊗S−1/2
t
)−1

= (S−1/2
t
(

d
X

i=1
hiAi)S−1/2
t
⊗S In)2,

plug in the above derivation we obtain the desired result. Finally, we send t →0 and conclude the
proof.

Lemma G.9. Let H(x) be the matrix function A(S−1 ⊗S−1)A⊤, then we have

∥H(x)−1/2∇F(x)∥2 ≤eO(d).

44


---Page Break---
Proof. We have

∥H(x)−1/2∇F(x)∥2 =
max
v:∥v∥2=1(H(x)−1/2∇F(x))⊤v

=
max
v:∥v∥2=1 tr[H(x)−1DH(x)[H(x)−1/2v]]

=
max
u:∥u∥H(x)=1 tr[H(x)−1/2DH(x)[u]H(x)−1/2]

≤
max
u:∥u∥H(x)=1

√

d∥H(x)−1/2DH(x)[u]H(x)−1/2∥F

≤2d · ∥u∥H(x)
= 2d,

where the fourth step is by tr[M] ≤
√

d · ∥M∥F for d × d PSD matrix M, the fifth step follows from
Lemma G.8, the last step follows from ∥u∥H(x) = 1.

G.4
Fast Hessian approximation

We need a particular type of sketch for Kronecker product of matrices.
Definition G.10 (TensorSRHT [Song et al., 2021]). The TensorSRHT Π : Rn × Rn →Rs is
defined as S :=
1
√sP · (HD1 ⊗HD2), where each row of P ∈{0, 1}s×n2 contains only one 1 at a
random coordinate and one can view P as a sampling matrix. H is a n × n Hadamard matrix, and
D1, D2 are two n × n independent diagonal matrices with diagonals that are each independently set
to be a Rademacher random variable (uniform in {−1, 1}).
Lemma G.11 (Lemma 2.12 in Song et al. [2021]). Let Π be a TensorSRHT matrix defined in
Definition G.10. If s = O(ϵ−2d log3(nd/(ϵδ))), then for any orthonormal basis U ∈Rn2×d, we
have that with probability at least 1 −δ, the singular values of ΠU lie in the range of [1 −ϵ, 1 + ϵ].

The following result provides an efficient embedding for B.

Lemma G.12. Let B ∈Rd×n2 be defined as in Def. G.3, ϵ ∈(0, 1/10) denote an accuracy parameter
and δ ∈(0, 1/10) denote a failure probability.

Let Π ∈Rs×n2 be a TensorSRHT matrix with s = Θ(ϵ−2d log3(nd/(ϵδ))), then we have

Pr[∥ΠB⊤x∥2 = (1 ± ϵ)∥B⊤x∥2, ∀x ∈Rd] ≥1 −δ.

Moreover, ΠB⊤can be computed in time

O(d · Tmat(n, n, s)).

Proof. The correctness part follows directly from Lemma G.11. It remains to argue for the running
time. We need to unravel the construction of both S and B (see Definition G.2). Recall that
Π =
1
√sP · (HD1 ⊗HD2) and

Π(S(x)−1/2 ⊗S(x)−1/2) vec(Ai)

= 1
√sP · (HD1 ⊗HD2) · (S(x)−1/2 ⊗S(x)−1/2) vec(Ai)

= 1
√sP · (HD1S(x)−1/2 ⊗HD2S(x)−1/2) vec(Ai)

= 1
√sP · vec(HD2S(x)−1/2AiS(x)−1/2D1H⊤)

since P is a row sampling matrix, the product can be computed as follows:

• First compute HD1S(x)−1/2 and HD2S(x)−1/2. Since H is a Hadamard matrix, this step
can be carried out in O(n2 log n) time.

45


---Page Break---
• Applying P to the vector can be interpreted as sampling s coordinates from the ma-
trix HD2S(x)−1/2AiS(x)−1/2D1H⊤. Let (i1, j1), . . . , (is, js) denote the coordinates
sampled by P.
We form two matrices X, Y
∈Rs×n where the k-th row of X
is (HD2S(x)−1/2)ik,∗and the k-th row of Y is (HD1S(x)−1/2)jk,∗.
It is easy to
verify that the (ik, jk)-th entry of XAiY ⊤∈Rs×s is the corresponding entry of
HD2S(x)−1/2AiS(x)−1/2D1H⊤. This step therefore takes Tmat(n, n, s) time.

As we need to apply the second step to d rows, the total runtime is

O(d · Tmat(n, n, s)).

Remark G.13. For comparison, compute B straightforwardly using the Kronecker product identity
takes O(dnω) time. As long as the row count s is independent of n, Lemma G.12 approximately
computes B in time eO(n2 · poly(d)). For the regime where d ≪n, our algorithm is much more
efficient.

H
Convexity of Regularized Log-Barrier Function

In this section, we prove the convexity of log det(Hlog(x) + Id) for both polytopes and spectrahedra.

H.1
Convexity of log det(Hlog(x) + Id): polytopes

We prove that for H(x) = Pn
i=1
aia⊤
i
(a⊤
i x−bi)2 , the Hessian of the log barrier, we have the convexity.

Definition H.1 (Ridge leverage score). Let B ∈Rn×d, we define the λ-ridge leverage score of B as

eσλ,i(B) := b⊤
i (B⊤B + λId)−1bi.

If λ = 1, we abbrieviate as eσi(B).

For convenient of the proof, we also define

eσλ,i,j(B) := b⊤
i (B⊤B + λId)−1bj.

We also define bσλ,i(B) as follows:

bσλ,i(B) := b⊤
i (B⊤B + λId)−2bi.

Fact H.2. Let X be Rd×d be invertible and symmetric. Then we have

∂X−1

∂t
= −X−1 ∂X

∂t X−1.

Lemma H.3. Let X ∈Rd×d be invertible and symmetric. Then,

∂log det X = X−1∂X.

Proof. The proof is by chain rule:

∂log det X =
1
det X ∂det X,

to compute ∂det X, recall the adjugate of X, denoted by adj(X) where X−1 =
1
det X adj(X), and
it remains to show that ∂det X = adj(X), this can be derived using cofactor expansion of a matrix,
which states that fix a row i, we have

det X =

d
X

k=1
Xi,k · adj(X)i,k,

by product rule,

∂det X

∂Xi,j
=

d
X

k=1

∂Xi,k
∂Xi,j
· adj(X)i,k + Xi,k · ∂adj(X)i,k
∂Xi,j

46


---Page Break---
= adj(X)i,j,

where
∂adji,k

∂Xi,j
= 0 for all k since the cofactor of i, k is the principal minor by removing row i and
column k. Therefore,

∂det X = adj(X)

and consequently,

∂log det X =
1
det X ∂det X

=
1
det X adj(X)

= X−1.

The goal of this section is to prove the d × d matrix ∇2F(x) ≻0. We start with computing
∇F(x) ∈Rd.

Lemma H.4. We define sx,i = a⊤
i x −bi for each i ∈[n]. Let Sx denote a diagonal matrix such that
Sx = diag(sx). Let H(x) = A⊤S−2
x A ∈Rd×d. Let F(x) = log det(H(x) + Id), then

∇F(x) = −2

n
X

i=1
eσi(S−1
x A) ai

sx,i
.

Proof. First, we know that

∂s−2
x,i
∂xj
= −2ai,j

s3
x,i
.
(19)

For each j ∈[d], we can write ∂F

∂xj as follows:

∂F
∂xj
= tr[(H(x) + Id)−1 ∂(H(x) + Id)

∂xj
]

= tr[(H(x) + Id)−1 ∂H(x)

xj
]

= tr[(H(x) + Id)−1
n
X

i=1
aia⊤
i
∂s−2
x,i
∂xj
]

= tr[(H(x) + Id)−1
n
X

i=1
−2aia⊤
i
ai,j
s3
x,i
]

= −2

n
X

i=1
tr[ai,j

sx,i

a⊤
i (H(x) + Id)−1ai

s2
x,i
]

= −2

n
X

i=1
eσi(S−1
x A)ai,j

sx,i
.

where the forth step follows from Eq. (19), the fifth step follows from tr[AB] = tr[BA], and the last
step follows from eσ.

We use chain rule:
∂F

∂x =
 ∂F

∂x1
∂F
∂x2
· · ·
∂F
∂xd
⊤

= −2

n
X

i=1
eσi(S−1
x A) ai

sx,i
.

Fact H.5. We have the following partial derivative:

∂(eσi(S−1
x A)s2
x,i)
∂xl
= 2

n
X

j=1
eσ2
i,j(S−1
x A) aj,l

sx,j

47


---Page Break---
Proof. We analyze the partial derivative term

∂
∂xl
a⊤
i (H(x) + Id)−1ai = a⊤
i (∂(H(x) + Id)−1

∂xl
)ai

= a⊤
i (−(H(x) + Id)−1 ∂H(x)

∂xl
(H(x) + Id)−1)ai

= a⊤
i (2(H(x) + Id)−1(

n
X

j=1
aja⊤
j
aj,l
s3
x,j
)(H(x) + Id)−1)ai

= 2

n
X

j=1

(a⊤
i (H(x) + Id)−1aj)2

s2
x,j

aj,l
sx,j

= 2

n
X

j=1
eσ2
i,j(S−1
x A) aj,l

sx,j
(20)

where the second step follows from Fact H.2.

Fact H.6. We have

eσi(S−1
x A) = bσi(S−1
x A) +

n
X

j=1
eσ2
i,j(S−1
x A).

Proof. Note that

(H(x) + Id)−1 = (H(x) + Id)−1(H(x) + Id)(H(x) + Id)−1

= (H(x) + Id)−2 +

n
X

j=1

(H(x) + Id)−1aja⊤
j (H(x) + Id)−1)
s2
x,j
,

therefore,

eσi(S−1
x A) = a⊤
i (H(x) + Id)−1ai
1
s2
x,i

= a⊤
i (H(x) + Id)−2ai

s2
x,i
+

n
X

j=1

(a⊤
i (H(x) + Id)−1aj)2

s2
x,is2
x,j

= bσi(S−1
x A) +

n
X

j=1
eσ2
i,j(S−1
x A)

Thus we complete the proof.

Lemma H.7. Let F(x) = log det(H(x) + Id), then

∇2F(x) ⪰0.

Thus, F(x) is convex.

Proof. Note that

∂
∂xl

∂
∂xk
F(x) = −

n
X

i=1
a⊤
i (H(x) + Id)−1aiai
∂
∂xl
(ai,k

s3
x,i
) + ai,k

s3
x,i

∂
∂xl
a⊤
i (H(x) + Id)−1ai


= 3

n
X

i=1

a⊤
i (H(x) + Id)−1ai

s2
x,i

ai,kai,l

s2
x,i
−ai,k

s3
x,i

∂
∂xl
a⊤
i (H(x) + Id)−1ai.
(21)

Plug in Eq. (20) into Eq. (21), we have

3

n
X

i=1
eσi(S−1
x A)ai,kai,l

s2
x,i
−2

n
X

i=1

n
X

j=1
eσi,j(S−1
x A)2 ai,kaj,l

sx,jsx,i
,

48


---Page Break---
and

3

n
X

i=1
eσi(S−1
x A)aia⊤
i
s2
x,i
−2

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A) aia⊤
j
sx,jsx,i

= 3

n
X

i=1
(bσi(S−1
x A) +

n
X

j=1
eσ2
i,j(S−1
x A))aia⊤
i
s2
x,i
−2

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A) aia⊤
j
sx,jsx,i

= 3

n
X

i=1
bσi(S−1
x A)aia⊤
i
s2
x,i
+ 3

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A)aia⊤
i
s2
x,i
−2

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A) ·
aia⊤
j
sx,jsx,i

= 3

n
X

i=1
bσi(S−1
x A)aia⊤
i
s2
x,i
+

n
X

i=1
eσi(S−1
x A)aia⊤
i
s2
x,i
+ 2

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A) · (aia⊤
i
s2
x,i
−
aia⊤
j
sx,jsx,i
)

:= B1 + B2 + B3.
where the first step follows from Fact H.6, the last step follows from

B1 := 3

n
X

i=1
bσi(S−1
x A)aia⊤
i
s2
x,i
,

B2 :=

n
X

i=1
eσi(S−1
x A)aia⊤
i
s2
x,i
,

B3 := 2

n
X

i=1

n
X

j=1
eσ2
i,j(S−1
x A) · (aia⊤
i
s2
x,i
−
aia⊤
j
sx,jsx,i
).

It is obvious that B1 ≻0 and B2 ≻0.

It is not hard to see that
aia⊤
i
s2
x,i
+ aja⊤
j
s2
x,j
⪰2 aia⊤
j
sx,jsx,i
.

Thus we have B3 ⪰0 and we complete the proof.

H.2
Convexity of log det(Hlog(x) + Id): spectrahedra

In this section, we prove that log det(Hlog(x) + Id) is a convex function on x. Combined with the
bounded variance property, we provide an algorithm that samples from the SDP spectrahedron.

Definition H.8. We define the ridge-projection matrix P ∈Rn2×n2 as

P := (S−1/2 ⊗S−1/2)A⊤(H + Id)−1A(S−1/2 ⊗S−1/2).

We define projection matrix P ∈Rn2×n2 as follows

P := (S−1/2 ⊗S−1/2)A⊤H−1A(S−1/2 ⊗S−1/2).
Remark H.9. It is not hard to see that P is a PSD matrix. Note that the matrix H + Id is PSD, since
H is PSD, therefore (H + Id)−1 is also PSD. Thus, the ridge-projection is PSD, as its two “arms”
are symmetric.

Lemma H.10. Let P ∈Rn2×n2 be the projection matrix and let P ∈Rn2×n2 be the ridge-projection
matrix defined in Def. H.8. Then, we have P ⪯P.

Proof. Let H = UΣU ⊤be its eigendecomposition, we need to show that
(H + Id)−1 ⪯H−1.
Let us expand the LHS:

(UΣU ⊤+ Id)−1 = (U(Σ + Id)U ⊤)−1

= U(Σ + Id)−1U ⊤

⪯UΣ−1U ⊤,

where the last step follows from for any non-negative number λ, (1 + λ)−1 ≤λ−1.

49


---Page Break---
The following lemma is a generalization of Theorem 4.1 of Anstreicher [2000].
Lemma H.11. Let Hlog(x) ∈Rd×d be defined as Definition G.6. Let F(x) = log det(Hlog(x)+Id).
Then, F(x) is convex in x.

Proof. Using standard algebra and chain rule computations, we have

∇2F(x) = 2Q(x) + R(x) −2T(x).

For simplicity, we drop x. The Q, R, T ∈Rd×d can be defined as follows

Qi,j := A(H + Id)−1A⊤· (S−1AiS−1AjS−1 ⊕S−1),

Ri,j := A(H + Id)−1A⊤· (S−1AiS−1 ⊕S−1AjS−1),

Ti,j := A(H + Id)−1A⊤· (S−1AiS−1 ⊕S−1) · A(H + Id)−1A⊤· (S−1AjS−1 ⊕S−1).

Using Lemma H.12, we know that

0 ⪯Q(x) ⪯∇2F(x) ⪯3Q(x)

Thus, F(x) is convex.

Lemma H.12. For any x, if S(x) ≻0, then we have

• Part 1. Q ⪰0;

• Part 2. T ⪰0;

• Part 3. T ⪯1
2(Q + R);

• Part 4. ∇2F(x) ⪰0;

• Part 5. R ⪯Q;

• Part 6. ∇2F(x) ⪯3Q(x).

Proof. Proof of Part 1 and 2. Let ξ ∈Rd and ξ ̸= 0.

Then we have

ξ⊤Qξ =
X

i,j
Qi,jξiξj

= A(H + Id)−1A⊤· (S−1BS−1BS−1 ⊕S−1)

= P · (B
2 ⊕Id)

where B = B(ξ) = Pd
i=1 ξiAi, and B = S−1/2BS−1/2.

Similarly, we have

ξ⊤Rξ = A(H + Id)−1A⊤· (S−1BS−1 ⊗S−1BS−1)

= P · (B ⊗B)

We can rewrite ξ⊤Tξ as follows:

ξ⊤Tξ = A⊤(H + I)−1A · (S−1BS−1 ⊕S−1)A(H + I)−1A⊤· (S−1BS−1 ⊕S−1)

= P · (B ⊕Id)P(B ⊕Id)

It is obvious that Id, P and B
2 are all PSD matrices.

Using Part 3 of Lemma A.9 and Part 6 of Lemma A.10, we have

ξ⊤Qξ ≥0

ξ⊤Tξ ≥0

50


---Page Break---
Since ξ is arbitrary, then we know that

Q ⪰0
T ⪰0

Proof of Part 3. Note that P is a projection matrix, and P ⪯P by Lemma H.10.

(B ⊕Id)P(B ⊕Id) ⪯(B ⊕Id)P(B ⊕Id)

⪯(B ⊕Id)Id(B ⊕Id)

= 1

2((B
2 ⊕Id) + (B ⊗B))

where the last step follows from Part 2 of Lemma A.10.

Applying Part 1 of Lemma A.9, we have

⟨P, (B ⊕Id)P(B ⊕Id)⟩≤1

2⟨P, (B
2 ⊕Id) + (B ⊗B)⟩

The above equation implies the following

ξ⊤Tξ ≤1

2ξ⊤(Q + R)ξ

Note that, here ξ is arbitrary, thus we know that

T ⪯1

2(Q + R)

Proof of Part 4. We have

∇2F(x) = 2Q + R −2T

⪰2Q + R −2 · 1

2(Q + T)

= Q

Proof of Part 5. Let vi be orthonormal eigenvectors of B with corresponding eigenvalues λi.

Then using [Horn and Johnson, 1990, Theorem 4.4.5], B
2 ⊕I has orthonormal eigenvectors vi ⊗vj
with corresponding eigenvalues 1

2(λ2
i + λ2
j), while (see Horn and Johnson [1990]) B ⊗B has the
same eigenvectors vi ⊗vj, with corresponding eigenvalues λiλj.

It then follows from (λi −λj)2 for each i, j that

B
2 ⊕Id ⪰B ⊗B.

Using Part 4 of Lemma A.9, we know

⟨P, B
2 ⊕Id⟩≥⟨P, B ⊗B⟩.

Which is

ξ⊤Qξ ≥ξ⊤Rξ

Since it’s arbitrary ξ, then we have

Q ⪰R

Proof of Part 6. We have

∇2F(x) = 2Q + R −2T
⪯2Q + R
⪯3Q

where the second step follows from T ⪰0, the last step follows from Part 5.

51


---Page Break---
H.3
Discussion

It is well-known that for popular barrier functions, such as volumetric barrier and Lee-Sidford
barrier [Lee and Sidford, 2014, 2019], the function log det H(x) is convex in x. However, given
a PSD matrix function H(x) for which log det H(x) is convex in x, it is generally not true that
log det(H(x) + Id) is convex.
Fact H.13 (Folklore). Suppose that H(x) is a positive semi-definite matrix for x in the domain.
Suppose log det H(x) is convex. It is possible log det(H(x) + I) is not convex.

Proof. If log det(H(x)+I) is convex in x, then for any fixed x ∈K, v ∈Rd, log det(H(x+tv)+Id)
is convex in t (for t sufficiently close to 0).

For t sufficiently closely to 0, define fi(t) := λi(H(x + tv)). Then

log det H(x + tv) =
X

i∈[d]
log fi(t)

and the condition that log det H(x) is convex is equivalent to

∂2

∂t2 log det H(x + tv) =
X

i∈[d]

f ′′
i (t)fi(t) −f ′
i(t)2

fi(t)2
≥0.

Now
log det(H(x + tv) + Id) =
X

i∈[d]
log(fi(t) + 1).

The condition that log det(H(x) + Id) is convex is equivalent to

∂2

∂t2 log det(H(x + tv) + Id) =
X

i∈[d]

f ′′
i (t)(fi(t) + 1) −f ′
i(t)2

(fi(t) + 1)2
≥0.

Suppose d = 2, f1(0) = 1, f ′
1(0) = 0, f ′′
1 (0) = 1, f2(0) = 4, f ′
2(0) = 0, f ′′
2 (0) = −3. Then

∂2

∂t2


t=0 log det H(x + tv) = f ′′
1 (0)
f1(0) + f ′′
2 (0)
f2(0) > 0.

∂2

∂t2


t=0 log det(H(x + tv) + I) =
f ′′
1 (0)
f1(0) + 1 +
f ′′
2 (0)
f2(0) + 1 < 0.

This gives an example for which log det H(x) is convex in a region but log det(H(x) + Id) is
not.

I
Convexity of Regularized Volumetric Barrier Function

In this section, we prove the convexity of log det of the regularized volumetric barrier function. This
is crucial, as the convexity proof for regularized Lee-Sidford barrier is identical up to replacing
leverage score by Lewis weights.

I.1
Definitions

Definition I.1. Let A ∈Rn×d. Let b ∈Rn. For each i ∈[n], we define

sx,i := (a⊤
i x −bi)
sx := Ax −b

For each i ∈[n], we define

σi,i(Ax) := a⊤
x,i(A⊤
x Ax)−1ax,i
For each i ∈[n], for each l ∈[n]

σi,l(Ax) := a⊤
x,i(A⊤
x Ax)−1ax,l
Let Σ = σ∗,∗(Ax) ◦In.

52


---Page Break---
Definition I.2. We define H(x) ∈Rd×d as follows

H(x) := A⊤
x
|{z}
d×n

Σ(Ax)
| {z }
n×n

Ax
|{z}
n×d

Definition I.3. For each j ∈[d], we define Pj(x) ∈Rn×n as follows

Pj(x) := ∂Σ(Ax)

∂xj
|
{z
}
n×n

For each j ∈[d], for each k ∈[d], we define Pj,k(x) ∈Rn×n as follows:

Pj,k(x) := ∂2Σ(Ax)

∂xkxj
|
{z
}
n×n

In the previous sections, we mainly use notation σ∗,∗(Ax). For simplicity, from this section, we will
use notation Q(x) instead.
Definition I.4. We define Q(x) ∈Rn×n as follows

Q(x) := σ∗,∗(Ax)
|
{z
}
n×n

.

Fact I.5. We have

σi,i(Ax) = ⟨σ2
∗,i(Ax), 1n⟩.

Proof. The proof is straightforward from definition.

We list a number of standard calculus results.
Fact I.6. We define the following quantities:

• Let sx = Ax −b ∈Rn;

• Let Sx = diag(sx) ∈Rn×n denote the diagonal matrix by putting sx on the diagonal;

• Let A∗,j denote the j-th column of matrix A ∈Rn×d;

• Let Ax = S−1
x A;

• Let a⊤
x,i denote the i-th row of Ax for each i ∈[n].

Then, we have for each j ∈[d]

• Part 1.
∂sx
∂xj
|{z}
n×1

= A∗,j
|{z}
n×1

• Part 2.
∂s−1
x
∂xj
| {z }
n×1

= −s−2
x
|{z}
n×1

◦A∗,j
|{z}
n×1

• Part 3.
∂s−2
x
∂xj
| {z }
n×1

= −2 s−3
x
|{z}
n×1

◦A∗,j
|{z}
n×1

53


---Page Break---
• Part 4.

∂S−2
x
∂xj
| {z }
n×n

= 2 diag(−s−3
x
◦A∗,j)
|
{z
}
n×n

= 2 diag(−S−3
x A∗,j)

• Part 5.

∂A⊤S−2
x A
∂xj
|
{z
}
d×d

= 2 A⊤
|{z}
d×n
diag(−S−3
x A∗,j)
|
{z
}
n×n

A
|{z}
n×d

• Part 6.

∂A⊤
x Ax
∂xj
= 2A⊤
x diag(−Ax,∗,j)Ax

• Part 7.

∂(A⊤
x Ax)−1

∂xj
= 2(A⊤
x Ax)−1 · A⊤
x diag(Ax,∗,j)Ax · (A⊤
x Ax)−1

• Part 8. For each i ∈[n]

∂ax,i

∂xj
= −Ax,i,j
| {z }
scalar

· ax,i
|{z}
d×1

• Part 9. For each i ∈[n]

∂ax,ia⊤
x,i
∂xj
= −2 · Ax,i,j
| {z }
scalar

· ax,ia⊤
x,i
| {z }
d×d

• Part 10.
∂Ax,i,j

∂xj
| {z }
scalar

= −A2
x,i,j
| {z }
scalar

• Part 11.
∂Ax,i,j

∂xk
| {z }
scalar

= −Ax,i,j
| {z }
scalar

Ax,i,k
| {z }
scalar

• Part 12.
∂Ax,∗,j

∂xj
= −A◦2
x,∗,j

• Part 13.
∂Ax,∗,j

∂xk
= −Ax,∗,j ◦Ax,∗,k

• Part 14.
∂Ax

∂xj
|{z}
n×d

= −diag(Ax,∗,j)
|
{z
}
n×n

Ax
|{z}
n×d

54


---Page Break---
Definition I.7. We define eΣ(Ax) ∈Rn×n and eσ∗,∗(Ax) ∈Rn×n as

eΣ(Ax) := (Ax(H(x) + Id)−1A⊤
x ) ◦In
eσ∗,∗(Ax) := Ax(H(x) + Id)−1A⊤
x

Definition I.8. We define F(x) as follows:

F(x) := log(det(H(x) + Id))

Definition I.9. We define
∂
∂x
∂F
∂x⊤∈Rd×d to be

∂
∂x
∂F
∂x⊤= [ ∂

∂x
∂F
∂x⊤]1 + [ ∂

∂x
∂F
∂x⊤]2

where

[ ∂

∂x
∂F
∂x⊤]1,j,k = + tr[(H(x) + Id)−1 ∂2H(x)

∂xj∂xk
]

[ ∂

∂x
∂F
∂x⊤]2,j,k = −tr[(H(x) + Id)−1 ∂H(x)

∂xk
(H(x) + Id)−1 · ∂H(x)

∂xj
]

I.2
Gradient of σi,j

Fact I.10 (First derivative of leverage score). We define the following quantities:

• For each j ∈[d], let A∗,j ∈Rn denote j-th column of A;

• Let Ax := S−1
x A ∈Rn×d;

• Let Ax,∗,j ∈Rn denote the j-th column of Ax ∈Rn×d;

• Let Σ(Ax) ∈Rn×n denote a diagonal matrix where (i, i)-th entry is σi,i(Ax);

• Let σ◦2
∗,∗(Ax) = σ∗,∗(Ax) ◦σ∗,∗(Ax) ∈Rn×n;

• Let σ∗,i(Ax) ∈Rn denote a column vector of σ∗,∗(Ax) ∈Rn×n.

Then we have for each j ∈[d]

• Part 1. For each i ∈[n],

∂σi,i(Ax)

∂xj
= 2⟨σ∗,i(Ax) ◦σ∗,i(Ax), Ax,∗,j⟩−2σi,i(Ax) · Ax,i,j

• Part 2. For each i ∈[n], l ∈[n]

∂σi,l(Ax)

∂xj
= 2⟨σi,∗(Ax) ◦σl,∗(Ax), Ax,∗,j⟩−σi,l(Ax) · (Ax,i,j + Ax,l,j)

Proof. We know

∂σi,i(Ax)

∂xj
= ∂a⊤
x,i(A⊤
x Ax)−1ax,i
∂xj

= ∂⟨ax,ia⊤
x,i, (A⊤
x Ax)−1⟩
∂xj

= ⟨∂ax,ia⊤
x,i
∂xj
, (A⊤
x Ax)−1⟩+ ⟨ax,ia⊤
x,i, ∂(A⊤
x Ax)−1

∂xj
⟩

For the first term in the above, we have

−2Ax,i,j · ⟨ax,ia⊤
x,i, (A⊤
x Ax)−1⟩= −2Ax,i,j · σi,i(Ax)

55


---Page Break---
For the second term, we have

⟨ax,ia⊤
x,i, ∂(A⊤
x Ax)−1

∂xj
⟩= 2⟨ax,ia⊤
x,i, (A⊤
x Ax)−1 · A⊤
x diag(Ax,∗,j)Ax · (A⊤
x Ax)−1⟩

= 2a⊤
x,i(A⊤
x Ax)−1 · A⊤
x diag(Ax,∗,j)Ax · (A⊤
x Ax)−1ax,i

= 2a⊤
x,i(A⊤
x Ax)−1 · (

n
X

l=1
ax,la⊤
x,lAx,l,j) · (A⊤
x Ax)−1ax,i

= 2

n
X

l=1
σl,i(Ax)2Ax,l,j

= 2⟨σ2
∗,i(Ax), Ax,∗,j⟩.

Similarly, we can prove for ∂σi,l(Ax)

∂xj
.

I.3
Gradient of σ

Lemma I.11. Let σ and Σ be defined as Definition I.1. Then, we have

• Part 1

∂σ∗,i(Ax)

∂xj
= 2 σ∗,∗(Ax)
|
{z
}
n×n

(σ∗,i ◦Ax,∗,j)
|
{z
}
n×1

−2 σ∗,i(Ax)
|
{z
}
n×1

◦Ax,∗,j
| {z }
n×1

• Part 2.

∂σ∗,∗(Ax)

∂xj
= 2 σ∗,∗(Ax)
|
{z
}
n×n

diag(Ax,∗,j)σ∗,∗(Ax) −diag(Ax,∗,j)σ∗,∗(Ax) −σ∗,∗(Ax) diag(Ax,∗,j)

• Part 3.

∂Σ(Ax)

∂xj
|
{z
}
n×n

= 2 diag(σ◦2
∗,∗(Ax)
|
{z
}
n×n

Ax,∗,j
| {z }
n×1

) −2 diag(Σ(Ax)
| {z }
n×n

Ax,∗,j
| {z }
n×1

)

= 2 diag((σ◦2
∗,∗(Ax) −Σ(Ax))Ax,∗,j)

Proof. It follows from Fact I.10.

I.4
Gradient for H(x)

Lemma I.12. Recall the definitions of the following quantities:

• Let H(x) ∈Rd×d be defined as Definition I.2.

• For each j ∈[d], let Pj(x) ∈Rn×n be defined as Definition I.3.

• For each j ∈[d], for each k ∈[d], let Pj,k(x) ∈Rn×n be defined as Definition I.3.

Then, we have

• Part 1. For each j ∈[d]

∂H(x)

∂xj
= −2A⊤
x diag(Σ(Ax)Ax,∗,j)Ax

+ A⊤
x Pj(x)Ax

56


---Page Break---
• Part 2. For each j ∈[d], for each k ∈[d]

∂
∂xk
(∂H(x)

∂xj
) = + 6A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax

−2A⊤
x Pk(x) diag(Ax,∗,j)Ax
−2A⊤
x Pj(x) diag(Ax,∗,k)Ax
+ A⊤
x Pj,k(x)Ax

Proof. Proof of Part 1. We have

∂H(x)

∂xj
= ∂(A⊤
x Σ(Ax)Ax)

∂xj

= (∂A⊤
x
∂xj
) · Σ(Ax)Ax + A⊤
x · (∂Σ(Ax)

∂xj
) · Ax + A⊤
x Σ(Ax) · (∂Ax

∂xj
)

= −Ax diag(Ax,∗,j)Σ(Ax)Ax
+ A⊤
x P(x)jAx
−AxΣ(Ax) diag(Ax,∗,j)Ax
= −2A⊤
x diag(Σ(Ax)Ax,∗,j)Ax
+ A⊤
x Pj(x)Ax

Proof of Part 2. Then, we have

∂
∂xk
(∂H(x)

∂xj
) =
∂
∂xk
(−2A⊤
x Σ(Ax) diag(Ax,∗,j)Ax + A⊤
x Pj(x)Ax)

=
∂
∂xk
(−2A⊤
x Σ(Ax) diag(Ax,∗,j)Ax) +
∂
∂xk
(A⊤
x Pj(x)Ax),
(22)

where the first step follows from Part 1, the second step follows from the sum rule.

For the first term of Eq. (22), we have

∂
∂xk
(−2A⊤
x Σ(Ax) diag(Ax,∗,j)Ax)

= −2 ∂

∂xk
(A⊤
x )Σ(Ax) diag(Ax,∗,j)Ax −2A⊤
x
∂
∂xk
(Σ(Ax)) diag(Ax,∗,j)Ax

−2A⊤
x Σ(Ax) ∂

∂xk
(diag(Ax,∗,j))Ax −2A⊤
x Σ(Ax) diag(Ax,∗,j) ∂

∂xk
(Ax)

= −2 ∂

∂xk
(A⊤
x )Σ(Ax) diag(Ax,∗,j)Ax −2A⊤
x
∂
∂xk
(Σ(Ax)) diag(Ax,∗,j)Ax

−2A⊤
x Σ(Ax) diag( ∂

∂xk
(Ax,∗,j))Ax −2A⊤
x Σ(Ax) diag(Ax,∗,j) ∂

∂xk
(Ax)

= + 2A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax
−2A⊤
x Pk(x) diag(Ax,∗,j)Ax
+ 2A⊤
x Σ(Ax) diag(Ax,∗,j) diag(Ax,∗,k)Ax
+ 2A⊤
x Σ(Ax) diag(Ax,∗,j) diag(Ax,∗,k)Ax,
(23)

For the second term of Eq. (22), we have

∂
∂xk
(A⊤
x Pj(x)Ax)

=
∂
∂xk
(A⊤
x )Pj(x)Ax + A⊤
x
∂
∂xk
(Pj(x))Ax + A⊤
x Pj(x) ∂

∂xk
(Ax)

= −(A⊤
x ) diag(Ax,∗,k)Pj(x)Ax + A⊤
x Pj,k(x)Ax −A⊤
x Pj(x) diag(Ax,∗,k)Ax

57


---Page Break---
= −(A⊤
x ) diag(Ax,∗,k)Pj(x)Ax + A⊤
x Pj,k(x)Ax −A⊤
x diag(Ax,∗,k)Pj(x)Ax
= −2(A⊤
x ) diag(Ax,∗,k)Pj(x)Ax + A⊤
x Pj,k(x)Ax.
(24)

By combining Eq. (22), Eq. (23), and Eq. (24), we have

∂
∂xk
(∂H(x)

∂xj
) = + 2A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax

−2A⊤
x Pk(x) diag(Ax,∗,j)Ax
+ 2A⊤
x Σ(Ax) diag(Ax,∗,j) diag(Ax,∗,k)Ax
+ 2A⊤
x Σ(Ax) diag(Ax,∗,j) diag(Ax,∗,k)Ax
−2A⊤
x diag(Ax,∗,k)Pj(x)Ax
+ A⊤
x Pj,k(x)Ax
= + 6A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax
−2A⊤
x Pk(x) diag(Ax,∗,j)Ax
−2A⊤
x Pj(x) diag(Ax,∗,k)Ax
+ A⊤
x Pj,k(x)Ax.

Thus, we complete the proof.

I.5
Hessian of H(x)

Lemma I.13. Recall the definitions of following quantities:

• Let H(x) := A⊤
x Σ(Ax)Ax ∈Rd×d be as definition I.2.

• Let Q(x) := σ∗,∗(Ax) ∈Rn×n be defined as Definition I.4.

Then, we have

• Part 1. For each j ∈[d] and for each k ∈[d]

∂2H(x)

∂xjxk
= C1 + C2 + C3 + C4 + C5

where we define some local d × d size matrix variables

– C1 = +20A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)Ax
– C2 = −6A⊤
x diag(Q◦2(x)(Ax,∗,k ◦Ax,∗,j))Ax
– C3 = −8A⊤
x diag(Ax,∗,k) diag(Q◦2(x)Ax,∗,j)Ax
– C4 = −8A⊤
x diag(Ax,∗,j) diag(Q◦2(x)Ax,∗,k)Ax
– C5 = +8A⊤
x ((Q(x) diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q(x)) ◦In)Ax

Proof. By Part 2 of Lemma I.12, we can show that

∂2H(x)

∂xjxk
= B1 + B2 + B3 + B4

where

B1 = + 6A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax
B2 = −2A⊤
x Pk(x) diag(Ax,∗,j)Ax
B3 = −2A⊤
x Pj(x) diag(Ax,∗,k)Ax
B4 = + A⊤
x Pj,k(x)Ax

where B2 can be decomposed further as

B2 = −2A⊤
x 2 diag((Q◦2(x) −Σ(Ax))Ax,∗,k) diag(Ax,∗,j)Ax

58


---Page Break---
= −4A⊤
x diag(Q◦2(x)Ax,∗,k −Σ(Ax)Ax,∗,k) diag(Ax,∗,j)Ax
= −4A⊤
x (diag(Q◦2(x)Ax,∗,k) −diag(Σ(Ax)Ax,∗,k)) diag(Ax,∗,j)Ax
= −4A⊤
x diag(Q◦2(x)Ax,∗,k) diag(Ax,∗,j)Ax + 4A⊤
x diag(Σ(Ax)Ax,∗,k) diag(Ax,∗,j)Ax
= −4A⊤
x diag(Q◦2(x)Ax,∗,k) diag(Ax,∗,j)Ax + 4A⊤
x diag(Ax,∗,k)Σ(Ax) diag(Ax,∗,j)Ax,

where the second step follows from simple algebra, the third step follows from the definition of
diag(·), the fourth step follows from simple algebra, and the last step follows from the definition of
diag(·). For B3, consider the following:

B3 = −2A⊤
x · 2 diag((Q◦2(x) −Σ(Ax))Ax,∗,j) diag(Ax,∗,k)Ax
= −4A⊤
x diag(Q◦2(x)Ax,∗,j −Σ(Ax)Ax,∗,j) diag(Ax,∗,k)Ax
= −4A⊤
x (diag(Q◦2(x)Ax,∗,j) −diag(Σ(Ax)Ax,∗,j)) diag(Ax,∗,k)Ax
= −4A⊤
x diag(Q◦2(x)Ax,∗,j) diag(Ax,∗,k)Ax + 4A⊤
x diag(Σ(Ax)Ax,∗,j) diag(Ax,∗,k)Ax
= −4A⊤
x diag(Q◦2(x)Ax,∗,j) diag(Ax,∗,k)Ax + 4A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)Ax,

where the second step follows from simple algebra, the third step follows from the definition of
diag(·), the fourth step follows from simple algebra, and the last step follows from the definition of
diag(·).

Finally for B4,

B4 = A⊤
x (
+ 8(Q(x) diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q(x)) ◦In
−6 diag(Q◦2(x) · (Ax,∗,k ◦Ax,∗,j))

−4 diag(Q◦2(x) · Ax,∗,j) diag(Ax,∗,k)

−4 diag(Q◦2(x) · Ax,∗,k) diag(Ax,∗,j)
+ 6Σ(Ax) diag(Ax,∗,j) diag(Ax,∗,k)
)Ax

Eventually, we can show Hessian is C1 + · · · + C5. The reason is following:

For the term C1, we have

C1 = B1 + B2,2 + B3,2 + B4,5
= (6 + 4 + 4 + 6) · A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,i)

= 20 · A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,i)

where B2,2 is the second term of B2, B3,2 is the second term of B3 and B4,5 is the last term of B4.

For the term C2, we have

C2 = B4,2

For the term C3, we have

C3 = B3,1 + B4,3
= −(4 + 4) · A⊤
x diag(Ax,∗,k) diag(Q◦2(x)Ax,∗,j)Ax
= −8 · A⊤
x diag(Ax,∗,k) diag(Q◦2(x)Ax,∗,j)Ax

For the term C4, we have

C4 = B2,1 + B4,4
= −4A⊤
x diag(Q◦2(x)Ax,∗,k) diag(Ax,∗,j)Ax −4A⊤
x diag(Q◦2(x) · Ax,∗,k) diag(Ax,∗,j)Ax
= −8A⊤
x diag(Q◦2(x)Ax,∗,k) diag(Ax,∗,j)Ax.

For the term C5, we have

C5 = B4,1.

59


---Page Break---
I.6
Gradient and Hessian of F(x)

Lemma I.14. Let F(x) ∈R be defined as Definition I.8. Then, we have

• Part 1.
∂F(x)

∂xj
= tr[(H(x) + Id)−1 · ∂H(x)

∂xj
]

• Part 2
∂
∂xk

∂F(x)

∂xj
= + tr[(H(x) + Id)−1
∂2H
∂xj∂xk
]

−tr[(H(x) + Id)−1 ∂H(x)

∂xk
(H(x) + Id)−1 · ∂H(x)

∂xj
]

Proof. Proof of Part 1. We have

∂F(x)

∂xj
= ∂log(det(H(x) + Id))

∂xj

= tr[(H(x) + Id)−1 ∂(H(x) + Id)

∂xj
]

= tr[(H(x) + I + d)−1(∂H(x)

∂xj
+ ∂Id

∂xj
)]

= tr[(H(x) + Id)−1 · ∂H(x)

∂xj
],

where the first step follows from the definition of F(x) (see Definition I.8), the third step follows
from the basic calculus rule, and the last step follows from ∂Id

∂xj = 0.

Proof of Part 2. We have
∂
∂xk

∂F(x)

∂xj

=
∂
∂xk
(tr[(H(x) + Id)−1 · ∂H(x)

∂xj
])

= tr[ ∂

∂xk
((H(x) + Id)−1) · ∂H(x)

∂xj
] + tr[(H(x) + Id)−1 ·
∂
∂xk
(∂H(x)

∂xj
)]

= tr[−(H(x) + Id)−1 ∂

∂xk
(H(x))(H(x) + Id)−1 · ∂H(x)

∂xj
] + tr[(H(x) + Id)−1 ·
∂
∂xk
(∂H(x)

∂xj
)]

= + tr[(H(x) + Id)−1
∂2H
∂xj∂xk
] −tr[(H(x) + Id)−1 ∂H(x)

∂xk
(H(x) + Id)−1 · ∂H(x)

∂xj
],

where the first step follows from Part 1, the second step follows from the product rule, and the last
step follows from simple algebra.

Using standard algebraic tools in literature [Lee and Sidford, 2019], we can show that,
Lemma I.15. We can rewrite it as follows
∂
∂x
∂F

∂x = 20H1 −6H2 −8H3 −8H4 + 8H5

−4G1 + 8G2 + 8G3 −16G4
and
∂
∂x
∂F

∂x ≻0.

Thus, the function F is convex in x.

60


---Page Break---
I.7
Quantity I for Hessian of F

Lemma I.16. We have

[ ∂

∂x
∂F
∂x⊤]1 = 20H1 −6H2 −8H3 −8H4 + 8H5

where

• H1 = A⊤
x eΣ(Ax)Σ(Ax)Ax

• H2 = A⊤
x diag(Q◦2(x)eΣ(Ax)1n)Ax

• H3 = A⊤
x Q◦2(x)eΣ(Ax)Ax

• H4 = A⊤
x eΣ(Ax)Q◦2(x)Ax

• H5 = A⊤
x (Q(x)eΣ(Ax)Q(x)) ◦Q(x))Ax

Proof. It comes from Pi,j and the following lemma.

Lemma I.17. Let H1 ∈Rd×d be Hessian where each entry is

(H1)j,k = tr[(H(x) + Id)−1A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)Ax]
Then, we have

H1 = A⊤
x eΣ(Ax)Σ(Ax)Ax

Proof. We can rewrite (H1)j,k as follows

(H1)j,k = tr[(H(x) + Id)−1A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)Ax]

= tr[Ax(H(x) + Id)−1A⊤
x diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)]
= tr[eσ∗,∗(Ax) diag(Ax,∗,j)Σ(Ax) diag(Ax,∗,k)]

= A⊤
x,∗,j eΣ(Ax)Σ(Ax)Ax,∗,k,
where the first step follows from the Lemma statement, the second step follows from the fact that
all of (H(x) + Id)−1, diag(Ax,∗,j), Σ(Ax), and diag(Ax,∗,k) are diagonal matrices, the third step
follows from the definition of eσ∗,∗(see Definition I.7), and the last step follows from the definition of
eΣ(Ax) (see Definition I.7).

Thus, we have

H1 = A⊤
x eΣ(Ax)Σ(Ax)Ax.
Lemma I.18. Let H2 be defined as

(H2)j,k = tr[(H(x) + Id)−1A⊤
x diag(Q◦2(x)(Ax,∗,k ◦Ax,∗,j))Ax]
Then, we have

H2 = A⊤
x diag(Q◦2(x)eΣ(Ax)1n)Ax

Proof. We have

(H2)j,k = tr[(H(x) + Id)−1A⊤
x diag(Q◦2(x)(Ax,∗,k ◦Ax,∗,j))Ax]

= tr[eΣ(Ax) diag(Q◦2(x)(Ax,∗,k ◦Ax,∗,j))]

= 1⊤
n eΣ(Ax)Q◦2(x)(Ax,∗,k ◦Ax,∗,j)

= A⊤
x,∗,k diag(Q◦2(x)eΣ(Ax)1n)Ax,∗j
where the first step follows from the Lemma statement, the second step follows from the definition of
eΣ(Ax) (see Definition I.7), the third step follows from the definition of 1⊤
n , and the last step follows
from the definition of Σ(Ax).

Thus, we have

H2 = A⊤
x diag(Q◦2(x)eΣ(Ax)1n)Ax.

61


---Page Break---
Lemma I.19. Let H3 be defined as

(H3)j,k = tr[(H(x) + Id)−1A⊤
x diag(Ax,∗,k) diag(Q◦2(x)(Ax,∗,j))Ax]

Then, we

H3 = A⊤
x Q◦2(x)eΣ(Ax)Ax

Proof. Then, we have

(H3)j,k = tr[(H(x) + Id)−1A⊤
x diag(Ax,∗,k) diag(Q◦2(x)(Ax,∗,j))Ax]

= tr[eΣ(Ax) diag(Ax,∗,k) diag(Q◦2(x)(Ax,∗,j))]

= A⊤
x,∗,k eΣ(Ax)Q◦2(x)Ax,∗,j,

where the first step follows from the Lemma statement, the second step follows from the definition of
eΣ(Ax) (see Definition I.7), and the last step follows from the property of diag(·).

Thus, we have

H3 = A⊤
x Q◦2(x)eΣ(Ax)Ax.

Lemma I.20. Let H4 be defined as

(H4)j,k = tr[(H(x) + Id)−1A⊤
x diag(Ax,∗,j) diag(Q◦2(x)(Ax,∗,k))Ax]

Then, we

H4 = A⊤
x eΣ(Ax)Q◦2(x)Ax

Proof. This proof is very similar to Lemma I.19, so we omit the details here.

Lemma I.21. Let H5 be defined as

(H5)j,k = tr[(H(x) + Id)−1A⊤
x (Q(x) diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q(x)) ◦In)Ax]

Then, we have

H5 = A⊤
x (Q(x)eΣ(Ax)Q(x)) ◦Q(x))Ax

Proof. We have

(H5)j,k = tr[(H(x) + Id)−1A⊤
x (Q(x) diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q(x)) ◦In)Ax]

= tr[eΣ(Ax)(Q(x) diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q(x)) ◦In]

=

n
X

l=1
eΣl,l(Ax)Q∗,l(x)⊤diag(Ax,∗,k)Q(x) diag(Ax,∗,j)Q∗,l(x)

= A⊤
x,∗,k

n
X

l=1
eΣl,l(Ax) diag(Q∗,l(x))Q(x) diag(Q∗,l(x))Ax,∗,j

= A⊤
x,∗,k(

n
X

l=1
eΣl,l(Ax)(Q∗,l(x)Q∗,l(x)⊤◦Q(x)))Ax,∗,j

= A⊤
x,∗,k((Q(x)eΣ(Ax)Q(x)) ◦Q(x))Ax,∗,j,

where the first step follows from the Lemma statement, the second step follows from the definition of
eΣ(Ax) (see Definition I.7), the third step follows from the definition of tr[·], and the last step follows
from the fact that eΣ(Ax) is a diagonal matrix (see Definition I.7).

Thus, we have

H5 = A⊤
x (Q(x)eΣ(Ax)Q(x)) ◦Q(x))Ax.

62


---Page Break---
I.8
Quantity II for Hessian of F

Recall that
∂H
∂xj
= −2A⊤
x diag(Σ(Ax)Ax,∗,j)Ax

+ A⊤
x
∂Σ(Ax)

∂xj
Ax

= −2A⊤
x diag(Σ(Ax)Ax,∗,j)Ax
+ 2A⊤
x diag((Q◦2(x) −Σ(Ax))Ax,∗,j)Ax
= A⊤
x diag(2Q◦2(x) −4Σ(Ax))Ax,∗,j)Ax
Lemma I.22. For the second item, we have

[ ∂

∂x
∂F

∂x ]2 = −4G1 + 8G2 + 8G3 −16G4

• G1 = A⊤
x Q◦2(x)eΣ(Ax)2Q◦2(x)Ax

• G2 = A⊤
x Q◦2(x)eΣ(Ax)2Σ(Ax)Ax

• G2 = A⊤
x Σ(Ax)eΣ(Ax)2Q◦2(x)Ax

• G4 = A⊤
x Σ(Ax)eΣ(Ax)2Σ(Ax)Ax

Proof. The proof follows by combining the following lemmas.

Lemma I.23. Let (G1)j,k
(G1)j,k = tr[(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,k)Ax]
Then, we have
G1 = A⊤
x Q◦2(x)eΣ(Ax)2Q◦2(x)Ax

Proof. We can show
(G1)j,k = tr[(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,k)Ax]

= tr[eΣ(Ax) diag(Q◦2Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,k)]

= tr[eΣ(Ax) diag(Q◦2Ax,∗,j)eΣ(Ax) diag(Q◦2(x)Ax,∗,k)]

= A⊤
x,∗,jQ◦2(x)eΣ(Ax)2Q◦2(x)Ax,∗,k
Thus, we have

G1 = A⊤
x Q◦2(x)eΣ(Ax)2Q◦2(x)Ax.

The proofs of G2, G3, G4 are similar to G1. So we omit the details here.
Lemma I.24. Let (G2)j,k
(G2)j,k = tr[(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Σ(Ax)Ax,∗,k)Ax]
Then, we have
G2 = A⊤
x Q◦2(x)eΣ(Ax)2Σ(Ax)Ax.
Lemma I.25. Let (G3)j,k
(G3)j,k = tr[(H + Id)−1A⊤
x diag(Σ(Ax)Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Q◦2(x)Ax,∗,k)Ax]
Then, we have
G3 = A⊤
x Σ(Ax)eΣ(Ax)2Q◦2(x)Ax.
Lemma I.26. Let (G4)j,k
(G4)j,k = tr[(H + Id)−1A⊤
x diag(Σ(Ax)Ax,∗,j)Ax(H + Id)−1A⊤
x diag(Σ(Ax)Ax,∗,k)Ax]
Then, we have
G4 = A⊤
x Σ(Ax)eΣ(Ax)2Σ(Ax)Ax.

63


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The main result of this paper is a class of Dikin walks for log-concave sampling
from convex bodies with self-concordant barrier functions. The abstract and introduction
reflect the complexity of these walks, and the remainder of the paper is dedicated to prove
these assertions.
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
Justification: The limitations are discussed in the last paragraph of Section 4.
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

64


---Page Break---
Answer: [Yes]
Justification: The assumptions are clearly stated in the statement of the theorems: convex
body contained in a ball of radius R, density is log-Lipschitz with parameter L, the walk
starts from a w-warm starting point. See Theorem 1.1, 1.2 and 1.3. The proofs of Theo-
rem 1.1 and 1.2 results can be found in Section C and D for mixing time, and Section F for
per iteration cost. Proofs of Theorem 1.3 can be found in Section G.
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
Answer: [NA]
Justification: The results in this paper are theoretical with no experiments.
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

65


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justification: The results in this paper are theoretical with no experiments.

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

Answer: [NA]

Justification: The results in this paper are theoretical with no experiments.

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

Justification: The results in this paper are theoretical with no experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

66


---Page Break---
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

Answer: [NA]

Justification: The results in this paper are theoretical with no experiments.

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

Justification: We have carefully read Code of Ethics and the paper is adhered to those
principles.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

Justification: The potential societal impact is discussed in the last paragraph of Section 4.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

67


---Page Break---
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
Justification: The results in this paper are theoretical with no experiments.
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
Justification: The results in this paper are theoretical with no experiments.
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

68


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The results in this paper are theoretical with no experiments.
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
Justification: The results in this paper are theoretical with no experiments.
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
Justification: The results in this paper are theoretical with no experiments.
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

69


---Page Break---
