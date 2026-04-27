Score-based generative models are provably robust:
an uncertainty quantification perspective

Nikiforos Mimikos-Stamatopoulos
Department of Mathematics
Université Côte d’Azur
nmimikos@unice.fr

Benjamin J. Zhang
Division of Applied Mathematics
Brown University
benjamin_zhang@brown.edu

Markos A. Katsoulakis
Department of Mathematics and Statistics
University of Massachusetts Amherst
markos@umass.edu

Abstract

Through an uncertainty quantification (UQ) perspective, we show that score-based
generative models (SGMs) are provably robust to the multiple sources of error
in practical implementation. Our primary tool is the Wasserstein uncertainty
propagation (WUP) theorem, a model-form UQ bound that describes how the L2
error from learning the score function propagates to a Wasserstein-1 (d1) ball
around the true data distribution under the evolution of the Fokker-Planck equation.
We show how errors due to (a) finite sample approximation, (b) early stopping, (c)
score-matching objective choice, (d) score function parametrization expressiveness,
and (e) reference distribution choice, impact the quality of the generative model
in terms of a d1 bound of computable quantities. The WUP theorem relies on
Bernstein estimates for Hamilton-Jacobi-Bellman partial differential equations
(PDE) and the regularizing properties of diffusion processes. Specifically, PDE
regularity theory shows that stochasticity is the key mechanism ensuring SGM
algorithms are provably robust. The WUP theorem applies to integral probability
metrics beyond d1, such as the total variation distance and the maximum mean
discrepancy. Sample complexity and generalization bounds in d1 follow directly
from the WUP theorem. Our approach requires minimal assumptions, is agnostic to
the manifold hypothesis and avoids absolute continuity assumptions for the target
distribution. Additionally, our results clarify the trade-offs among multiple error
sources in SGMs.

1
Introduction

Score-based generative models (SGMs) [1, 2] are highly effective [3], producing high quality samples,
with more stable and less computationally intensive training methods than generative adversarial nets
and normalizing flows [4]. The models are empirically robust to approximations and errors in learning
the score function. While SGM generalization properties have been studied for idealized conditions [5,
6, 7, 8], analyses of their robustness in practical settings remains underexplored. This paper analyzes
SGMs through the regularity theory of nonlinear partial differential equations (PDEs) [9], specifically
Hamilton-Jacobi-Bellman (HJB) equations [10]. Our main result is the Wasserstein uncertainty
propagation (WUP) theorem (Theorem 3.1), a versatile model-form uncertainty quantification (UQ)
bound which we use to theoretically explain the robustness of SGMs to approximation errors in

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
practical implementation. Generalization bounds for integral probability metrics (IPMs), such as the
Wasserstein-1 (d1) and the total variation (TV) distance follow directly from the WUP theorem.

In the context of SGMs, the WUP theorem shows how an L2 neighborhood around the true score
function propagates to an IPM neighborhood around the true data distribution. By relating how
various approximations in SGMs contributes to L2 error with respect to the true score function,
we establish how well the resulting SGM generalizes. Theorem 3.2 shows how the error in the
learned score-function with respect to the explicit score-matching objective, or uniform-in-time L2
error, and the choice of initial condition define the radii of d1 and TV neighborhoods, respectively.
Additionally, Theorem 3.3 addresses the case where the score function is learned from finite data
using the denoising score-matching (DSM) objective and incorporates early stopping.

Our bounds capture trade-offs among the errors. The ability of SGMs to generalize depends on
choices such as the early stopping time and how overtraining to the DSM objective and neglecting
early stopping lead to SGMs that memorize and overfit to the data. Notably, our approach relies
on minimal assumptions on the data distribution, being agnostic to the manifold hypothesis. This
contrasts with existing convergence results which assume the hypothesis [8] or not [5, 7] a priori.
Furthermore, unlike previous work, we obtain our generalization bounds with respect to the d1
distance directly, without appealing to the Girsanov theorem, the Kullback-Leibler divergence, the χ2
divergence, or Pinsker’s inequality [5, 11, 7]. This suggests our bounds may be sharper than those
which bound stronger norms and divergences. A notable feature of our DSM generalization bound is
that the error bound is a computable function of the DSM objective, contrasting with prior results
typically assume the learned score is close to the truth with respect to the ESM objective [5].

The WUP theorem also enables robust uncertainty quantification (UQ) for score-based generative
models, a rare capability in generative models. Robust UQ recognizes that learning complex models
involves multiple sources of uncertainty due to modeling choices and imperfect data. The distribu-
tionally robust perspective [12] quantifies a uncertainty set based on divergences and probability
metrics [13, 14, 15, 16] to measure the impact of model uncertainty around a baseline model. The
drawback is that these sets are typically difficult to find computationally. The WUP theorem is an
example of a robust UQ bound for IPMs that is computable.

1.1
Contributions

• We introduce the use of regularity theory of nonlinear PDEs for analyzing generative flows
[9, 10]. Our key idea is using the Kolmogorov backward equation to study the evolution
of IPMs under a generative flow, yielding generalization bounds. WUP is one particular
application of this idea, and while we use it to study SGMs, this analysis is not limited
to only SGMs. We show that the regularizing properties of the underlying Fokker-Planck
equation imply SGM’s robustness to errors with few assumptions on the data distribution.
An intuitive explanation of our main technical contribution is provided in Section 4.

• The WUP theorem (Theorem 3.1), which we state and prove, describes how an L2 ball
centered around the true score function maps to a neighborhood of IPMs (such as d1, TV,
and MMD [17]). WUP is a model-form uncertainty quantification bound, which maps
uncertainties introduced by modeling choices when practically implementing SGM. The
resulting generalization bounds WUP produces demonstrate that with properly chosen model
parameters, SGM is robust to errors due to score-matching, finite sample approximation,
early stopping time, and choice of the reference distribution. (Theorems 3.2, 3.3, and C.1).

• When applied to SGM, WUP produces generalization bounds under minimal hypotheses
on the data distribution and score function, and explains the impact of implementation
errors has on generalization. Our approach is agnostic to the manifold hypothesis, applying
whether or not the data distribution has a density. Moreover, it is adaptable to additional
assumptions about the target distribution to yield improved generalization bounds.

1.2
Related work

Convergence and generalization of SGMs have been well-studied. Many approaches [5, 7, 18] assume
the target distribution is absolutely continuous with respect to a Gaussian, and obtain generalization
bounds for TV, χ2, and d1 by bounding the KL divergence, a stronger divergence, via the Girsanov
theorem [11, 6, 18], Pinsker’s inequality [5], functional inequalities [7], or other means [11, 19].

2


---Page Break---
While [20] shows that kernel estimators for the score function are minimax optimal distribution
estimators, they make no connections to deep learning-based SGMs, and they assume the target
distribution is sub-Gaussian and is in a Sobolev space. Meanwhile, [6] has comprehensive sample
complexity results that considers neural net approximations in the finite sample regime. Their results,
however, are specialized to distributions in a Besov space, and, similar to other work [5, 11, 7, 19],
rely on bounding the KL divergence, which breakdown under the manifold hypothesis.

Our results are similar to [8] which derives d1 bounds directly, proving SGM convergence under the
manifold hypothesis. However, the analysis assumes a particular discretization of the SGM. In [21],
an uncertainty propagation bound for the Wasserstein-2 distance is proven using a similar approach
to this work. Their work, however, relies on strong, uncheckable assumptions on the score function
and target measure, does not address the errors we study here, and is not extendable to IPMs.

2
Background and notation

Let dimension d ∈N, and terminal time T > 0. Denote Td ⊂Rd to be the unit torus and RTd ⊂Rd
to be the torus of radius R > 0. Let P(Ω) be the space of probability distributions on Ω, where Ω
is Rd or RTd. Let π ∈P(Ω) be the target data distribution, known only through a finite number
of samples {zj}N
j=1 ∼π. Let f : [0, T] × Rd →R be a vector field and σ : R →R be a positive
function. Score-based generative modeling [1, 2] is based on considering two Stochastic Differential
Equations (SDEs for short) whose flow maps are inverses of each other. Define Y (s), X(t) to be the
forward and reverse diffusion processes over time interval s, t ∈[0, T] that evolve according to
dY (s) = −f(T −s, Y (s))ds + σ(T −s)dW(s),
Y (0) ∼π
(1)

dX(t) =

f(t, X(t)) + σ(t)2∇log ηπ(T −t, X(t))

dt + σ(t)dW(t),
X(0) ∼m0,
(2)

where Y (s) ∼ηπ(s, ·), and W(t) is a standard Brownian motion in Rd. Heres, ηπ(s, ·) is the density
of Y (s) at time s where the initial distribution was π. From [22], it is known that if m0 = ηπ(T, ·),
then X(t) ∼ηπ(T −t, ·). SGMs generated new samples from π simulating trajectories of the reverse
process (2) using an approximate score function sθ ≈s = ∇log ηπ. Typically f and σ are chosen
such that ηπ(T, ·) is well-approximated by a normal distribution, which is then used as the initial
distribution.

The score function is learned via samples {zj}N
j=1 from π by minimizing one of several score-
matching objectives. Let sθ be some function where the parameters θ are learned via one of three
score-matching objectives. For a path ρ : [0, T] →P(Ω), we define the functionals

J (ρ, θ) =
Z T

0

Z

Ω
|sθ −∇log ρ|2 dρ(s)ds JL(ρ, θ) =
Z T

0

Z

Ω

 
|sθ|2 + 2∇· sθ

dρ(s)ds,
(3)

where the subscript L highlights the linear dependence of JL on the underlying measure ρ. The
explicit score matching objective (ESM) is JESM(ηπ, θ) = J (ηπ, θ). As evaluation of the true score
function ∇log ηπ is typically inaccessible, the implicit (ISM) JISM(ηπ, θ) = JL(ηπ, θ) [23] or the
denoising (DSM) score-matching objectives [24, 1] are computed in practice

JDSM(ηπ, θ) =
Z T

0

Z

Ω

Z

Ω

sθ −∇log ηx′
2
dηx′(s)dπ(x′)ds.

Here ηx′(s) denotes the probability transition kernel from x′ to x of (1) at time s. The DSM objective
is most frequently used in practice as it does not require computing derivatives of sθ. It does, however,
require the evaluation of ηx′(s) in closed form, which is typically only accessible for linear SDEs.
Remark 2.1 (Choice of domain Ω). While our approach will also produce bounds when Ω= Rd, we
primarily focus on the torus RTd, which is equivalent to a bounded domain with periodic boundary
conditions. This choice ensures that the long-time behavior of the noising process (1) converges
to the uniform distribution on RTd. Therefore, we set f = 0 and σ(t) =
√

2 instead of using the
Ornstein-Uhlenbeck process. We make this choice for mathematical clarity, however our results
generally apply, with minor modifications, to the entire space Rd and for other noising processes.

2.1
Equivalence of score-matching objectives in the finite sample regime

Finite sample approximations of ESM, ISM, and DSM are not generally equivalent. However,
[24, 25], show that DSM becomes equivalent to ESM and ISM in the finite sample regime when

3


---Page Break---
ηπ(s) is replaced with its kernel density estimate ηN(s), where ηN(0) = πN =
1
N
PN
i=1 δzi. The
kernel estimate, however, does not have a well-defined score at s = 0, so the DSM objective is often
integrated only for s ∈[ϵ, T], an example of early-stopping in SGM [1, 26]. In continuous time,
this has the effect of score-matching for the mollified distribution πN,ϵ = πN ⋆ρϵ, where ρϵ is the
transition probability kernel for s = ϵ1. Then it can be shown that

J (ηN,ϵ, θ) = JDSM(ηN,ϵ, θ) = JESM(ηN,ϵ, θ) = JISM(ηN,ϵ, θ) + 4∥∇
p

ηN,ϵ∥2
2.
(4)

3
An uncertainty quantification approach to generalization in SGMs

We describe our UQ approach to generalization in SGMs and overview our main results. Discussion
of the proof methods is deferred to Sections 4 and 5. Our primary goal is to study how practical and
approximation errors made in implementing SGMs translate into errors in the generative distribution
relative to the true distribution.

3.1
Source of errors in score-based generative modeling

We attribute errors of SGM to the following six sources:

• Data distribution is only accessible via samples e1: The target distribution is only known
through a finite set of samples. In practice, the regularity of the score function and data
distribution, such as Lipschitzianity or whether the distribution has a density, is unknowable.

• Choice of score-matching objective e2: In practice, score-matching objectives are approx-
imated via samples. The DSM objective is most frequently used as it avoids computing
derivatives of the score function. While DSM and ISM are equivalent given the exact density
evolution η(x, s), they differ when approximated with finite samples. Previous analysis
typically assumes that the learned score function is close to the true score function within
some L2 distance, i.e., a ball determined by JESM. In contrast to previous work [11, 7, 5], we
show in Theorem B.5 how training through DSM translates into a bound for ESM. Moreover
we prove Proposition B.8 which states that if there exists a neural net that well-approximates
the true score then that the target density is necessarily regular.

• Expressiveness of the score function e3. The expressivity of neural net approximations to
the score function will depend on the particular expressivity of the parametrization.

• Choice of reference distribution e4. With access to the exact score function, the denoising
process (2) produces trajectories that sample from π only if the initial condition starts at
ηπ(·, T). In practice, however, the initial distribution is usually a Gaussian approximation
of ηπ or, in the case of the OU noising process, its corresponding stationary distribution.

• Early stopping of the denoising process e5. In practice, the score-matching objective is
integrated from s ∈[ϵ, T], for small ϵ, instead of s = 0 [1, 26]. This prevents the SGM from
memorizing the training data [27, 28], and is equivalent to running the denoising process
for t ∈[0, T −ϵ]. Early stopping is crucial when the data distribution is supported on
a low-dimensional manifold, where it has no density with respect to Lebesgue measure.
Previous analyses of SGM often adjust ϵ to optimize generalization bounds [6, 20].

• Discretization error e6. The denoising SDE must be solved via a numerical scheme. Previ-
ous work [5, 11, 7, 8] have considered the impact of discretization error on the generalization
abilities of SGM. While our analysis does not consider the discretization error, our approach
can be extended to study discretization error through the use of modified equations [29].

3.2
Model-form uncertainty quantification

Let d be an integral probability metric (IPM) between measures ν1, ν2 ∈P(Ω)

d(ν1, ν2) = sup
ψ∈X

Z
ψ(x)d(ν2 −ν1),
(5)

1Here, ⋆denotes convolution.

4


---Page Break---
for some function space X. We assume X = {ψ : Ω→R, ∥∇ψ∥∞≤1} or {ψ : Ω→R, ∥ψ∥∞≤
1}, corresponding to the Wasserstein-1 (d1) and total variation distances, respectively. Recall that if
ν1, ν2 have densities, then their TV distance is equivalent to the L1 distance between their densities.

Let π be the target data distribution and take the stationary distribution of the noising process on the
torus mg(0) =
1
vol(RTd) as the initial condition. Given a learned score function sθ, let mg(T) be the
generative distribution produced by SGM. We study how IPMs between π and mg(T) relate to errors
from the first five sources of errors discussed above, i.e., we seek bounds of the form

d(mg(T), π) ≤F(e1, e2, e3, e4, e5).
(6)
Our key insight is that for score-based generative modeling, we can derive bounds of the form (6)
via a model-form uncertainty quantification bound for the generative Fokker-Planck equation. The
Wasserstein Uncertainty Propagation theorem formalizes this insight.

3.3
Wasserstein uncertainty propagation theorem

The WUP theorem is a general statement about how L2 neighborhoods in the space of drift functions
map to neighborhoods in P(Ω) defined by IPMs.
Theorem 3.1 (Wasserstein Uncertainty Propagation). Let Ω= RTd or Rd. Let b1, b2 : [0, T] × Ω→
Rd be given with ∥∇b1∥∞< ∞, and m1, m2 ∈P(Ω). If mi for i = 1, 2 are given by

∂tmi −∆mi −div(mibi) = 0,
mi(0) = mi
(7)

then, up to a universal constant C > 0, we have the following:

• If
sup
0≤t≤T
∥(b2 −b1)(t)∥L2(m2(t)) := sup
0≤t≤T

Z

Ω

(b2 −b1)(t, x)
2 m2(t, x)dx
1/2
≤ε1,

then

∥m2(T) −m1(T)∥L1(Ω) ≤C(
p

T∥∇b1∥∞+ 1)
 1
√

T
d1(m1, m2) +
√

Tε1


,
(8)

and

∥m2(T) −m1(T)∥L1(Ω) ≤C(
p

T∥∇b1∥∞+ 1)

∥m1 −m2∥L1(Ω) +
√

Tε1

.
(9)

• For Ω= RTd, if ∥b2 −b1∥L2(m2) :=

 Z T

0

Z

Ω

(b2 −b1)(t, x)
2 m2(t, x)dxdt

!1/2

≤ε2,

then
d1(m2(T), m1(T)) ≤CR
3
2 (1 +
p

∥∇b1∥∞) (d1(m1, m2) + ε2) .
(10)

Equation (8) is a particularly notable result as we bound the TV distance between m1(T) and m2(T)
in terms of a weaker d1 metric between m1 and m2. This is due to the regularizing effects of diffusion
processes, which is showcased in the proof. See Section 4 and Section A.1 for full details.

3.4
Robustness of errors under ESM

We use WUP to produce generalization bounds when the only errors are due to the choice of the
initial condition and the approximation of the score function with respect to the ESM objective.
Theorem 3.2 (ESM bounds). Let π ∈P(Ω) where Ω= RTd for some R > 0. Moreover let enn > 0.
Assume that the learned score function sθ is such that J (ηπ, θ) ≤enn. Then for bθ = sθ(T −t, x)
the generated distribution mg(T) ≈π where

∂tmg −∆mg −div(mgbθ) = 0 in (0, T] × Ω,
mg(0) =
1
vol(RTd) in Ω,
(11)

satisfies

d1(π, mg(T)) ≤CR
3
2 (1 +
p

∥∇sθ∥∞)

Re−ωT

R2 d1

π,
1
vol(RTd)



|
{z
}
Error from choice of reference measure (e4)

+
√enn
| {z }
Score function error (e3)


.

(12)

5


---Page Break---
If the stronger estimate sup
0≤t≤T
∥sθ −∇log(ηπ)∥2
L2(ηπ(t)) ≤enn, holds, then

∥mg(T) −π∥L1(Ω) ≤C(
p

T∥∇sθ∥∞+ 1)

 
R2e−ωT

R2
√

T
d1

π,
1
vol(RTd)


+
p

Tenn

!

.
(13)

Here, the constants C, ω depend only on the dimension d.

Applying WUP for b1 and b2 to be the true and learned score function, respectively, with Proposition
D.3 on the convergence of the noising process to the stationary measure immediately yields the
estimates above. Notice that the TV estimate (13) is comparable to Theorem 2 of [5], which also
assumes a uniform-in-time L2-accurate score function. Again, notice that we are able to bound the
TV distance between mg(T) and π using the weaker d1 distance between π and
1
vol(RTd).

3.5
Robustness of errors under DSM

In practice, the score is learned through samples using the DSM objective with an early stopping
parameter [1, 26]. SGM is effective at producing samples from distributions supported on (or near)
low dimensional manifolds. Our d1 generalization bound describes how early stopping aids in
generalization.
Theorem 3.3. (Pointwise DSM generalization) Let bθ = sθ(T −t, x) and mg : [0, T]×RTd →R be
given by (11). Assume the learned score function is such that JDSM(ηN,ϵ, θ) = J (ηN,ϵ, θ) < enn.
Let 0 < δ < ϵ be such that δ ≤ˆπN,ϵ, δ < πϵ. Then up to a dimensional constant C = C(d) > 0,

d1(π, mg(T)) ≲
√ϵ
|{z}
Early stopping (e5)
+R3/2(1 +
p

∥∇sθ∥∞)

Re−ωT

R2 d1


π,
1
vol(RTd)



|
{z
}
Choice of reference measure (e4)

+
p

e′nn


, (14)

where

p

e′nn ≲
√enn
| {z }
DSM score function error (e3)

+

v
u
u
t


1 + | log(δ)|
√ϵ
+ T∥sθ∥2
C2([0,T ]×Ω)


d1(πN, π)
|
{z
}
Finite sample error (e1)
|
{z
}
Choice of SM objective (e2)

. (15)

The pointwise DSM generalization bound applies to every finite training sample of size N. A crucial
part of this theorem relates the error in the practical DSM objective function to the ESM error needed
to apply the WUP theorem. We connect the DSM objective early stopping to the ESM error with
the mollified distribution πϵ in Lemma B.5. This result is agnostic to the manifold hypothesis for
π, as long as ϵ > 0. Such bounds will blow up if the KL divergence were used instead. In fact,
previous results that use the KL divergence to bound the TV distance [5, 19, 11, 6] will produce
vacuous bounds for the d1 distance under the manifold hypothesis as their d1 generalization bounds
are derived by bounding the KL divergence.
Remark 3.4 (Density lower bound). Similar to [6], our DSM generalization bound relies on a
density lower bound assumption. We note however, that our DSM generalization bound holds for
any random sample {zi}N
i=1 ∼π. This density lower bound assumption can be removed via Jensen’s
inequality if we consider the expected d1 distance between π and mg(T) over random empirical
measures. See Theorem C.1 and its proof in Section C.
Remark 3.5 (Trade-offs and memorization). Theorem 3.3 captures trade-offs when training SGMs
and memorization effects. To minimize the error from early stopping, we can let ϵ →0. However,
empirically, without early stopping, SGMs overfit to the kernel approximation and memorize the
training data [27, 28, 25]. The bound is vacuous when ϵ = 0 regardless of whether the distribution
lies on a low-dimensional manifold. As ϵ →0, training the DSM to be small implies that the
score function must approximate a rough function with large Lipschitz constant, which will increase
the bound. This shows that overtraining the DSM objective may not necessarily yield a better
generative model. Moreover, suppose that πN = π, then as ϵ →0 and enn →0, we have that
d1(π, mg(T)) →0, indicating the model memorizes the training data. Our results corroborate those
of [25].

6


---Page Break---
4
Regularity theory of Hamilton-Jacobi-Bellman PDEs enables uncertainty
quantification in SGMs

A recent result in [30] established connections between generative flows with the theory of PDEs,
more specifically the theory of mean field games. We continue investigating these connections and
showcase how one may study generative modeling algorithms by obtaining stability estimates for the
Fokker-Planck equation. We provide a proof sketch for our Wasserstein Uncertainty Quantification
theorem (Theorem 3.1). Our strategy involves (1) constructing test functions for the IPMs using
the Kolmogorov backward equation, (2) choosing the suitable function space for the terminal data
depending on the desired IPM, and (3) obtaining gradient estimates of the test functions via Bernstein
estimates. The theorem relies on the gradient of the test function remaining bounded for t ∈[0, T),
which is guaranteed by the regularizing properties of diffusion processes. See A.1 for full details
about the proof.

4.1
Kolmogorov backward equation determines suitable test functions

From (7), we aim to compute bounds for d(m1(T), m2(T)) = supψ(x)∈X
R
ψ(m1(T) −m2(T))dx.
The measure λ = m1 −m2 satisfies the PDE
∂tλ −∆λ −div(λb1 + m2(b1 −b2)) = 0 in (0, T) × Ω,
λ(0) = m2 −m1 in Ω.
(16)

Let ϕ : [0, T] × Ω→R be a test function in space and time. We integrate in space and time against
the PDE (16) and integrate by parts to move the derivatives on to ϕ which yields

Z

Ω
λ(T, x)ϕ(T, x) −λ(0, x)ϕ(0, x)dx +
Z T

0

Z

Ω
λ
 
−∂tϕ −∆ϕ + b1 · ∇ϕ

dxdt
(17)

+
Z T

0

Z

Ω
m2∇ϕ · (b1 −b2)dxdt = 0

Notice that if we choose the test function ϕ to satisfy the Kolmogorov backward equation (KBE)
−∂tϕ −∆ϕ + b1 · ∇ϕ = 0 in [0, T) × Ω,
ϕ(T, x) = ψ(x) in Ω
(18)
with terminal condition ψ ∈X, then from (17), we have
Z

Ω
λ(T, x)ψ(x)dx =
Z

Ω
λ(0, x)ϕ(0, x)dx +
Z T

0

Z

Ω
m2(t)∇ϕ(t, x) · (b2 −b1)(t)dxdt.
(19)

The equality above is valid for any terminal condition ψ. Depending on the choice of function space
X for ψ, we obtain bounds on different IPMs. Taking the supremum over X we have

d(m1(T), m2(T)) ≤sup
ψ∈X



Z

Ω
λ(0, x)ϕ(0, x)dx
 + sup
ψ∈X



Z T

0

Z

Ω
m2∇ϕ · (b2 −b1)dxdt

 .
(20)

Recall that ϕ is related to ψ via the KBE (18) To obtain bounds for d, we need to bound the two
terms in (20), which depend on the choice of function space X and assumptions on the drift terms.

4.2
IPM bounds depend on choice of terminal function space and gradient estimates

We split our proof sketch into two parts; the first part focuses on deriving L1 estimates, while the
second derives d1 estimates.

L1 estimates.
We first note that because of the regularizing properties of (18) we can obtain bounds
on
R

Ωλ(0, x)ϕ(0, x)dx with weaker norms on ϕ. Notice that

Z

Ω
λ(0, x)ϕ(0, x)dx ≤∥λ(0)∥Y′∥ϕ(0)∥Y,
(21)

where Y denotes a generic space of functions, and Y′ is its dual. In what follows, we show that
regularizing effects of (18) takes functions in X to functions with more regularity Y such that Y is
compactly embedded in X. This in turn implies that ∥· ∥Y′ is weaker than d and so we are able to
bound the stronger norm d by the weaker norm.

7


---Page Break---
• To prove (8), we d = ∥· ∥L1(Ω), which corresponds with X = {ψ : Ω→R, ∥ψ∥∞≤1}.
Observe that we can take Y = {ψ : Ω→R, ∥∇ψ∥∞≤1}, in which case we obtain
Z

Ω
λ(0, x)ϕ(0, x)dx = ∥∇ϕ(0, x)∥∞

Z

Ω
(m1 −m2)
ϕ(0, x)
∥∇ϕ(0, x)∥∞
dx
(22)

≤d1(m1, m2)∥∇ϕ(0, x)∥∞.

Notice that this bound crucially depends on showing that ϕ(0, x) is Lipschitz.
• To prove (9), we can simply take X = Y, and obtain
Z

Ω
λ(0, x)ϕ(0, x) ≤∥λ(0, x)∥L1(Ω)∥ϕ(0, x)∥∞.
(23)

For (8) and (9), Cauchy-Schwarz on the spatial integral shows the second term can be bounded as
Z T

0

Z

Ω
m2∇ϕ · (b2 −b1)dxdt ≤
Z T

0
∥(b1 −b2)(t)∥L2(m2(t))∥∇ϕ(t, x)∥L2(m2(t))dt
(24)

≤sup
0≤t≤T
∥(b1 −b2)(t)∥L2(m2(t))

Z T

0
∥∇ϕ(t, ·)∥∞dt.

Notice here that it is crucial to produce estimates for ∇ϕ.

Wasserstein-1 (d1) estimates.
To prove (10), we choose X = Y = {ψ : Ω→R, ∥∇ψ∥∞≤1},
the space of 1-Lipschitz functions. We obtain (22) again, except the terminal data ψ also has Lipschitz
constant equal to 1. Applying Cauchy-Schwarz in space and time implies the second term is bounded

Z T

0

Z

Ω
m2∇ϕ · (b2 −b1)dxdt ≤

 Z T

0

Z

Ω
|∇ϕ|2m2dxdt

! 1

2
∥b1 −b2∥L2(m2)
(25)

≤T
sup
0≤t≤T
∥∇ϕ(t, ·)∥∞∥b1 −b2∥L2(m2).

We again see that we require estimates for ∇ϕ, and we need them to be bounded.

4.3
Bernstein estimates from HJB theory provide gradient estimates

To obtain estimates for ∇ϕ, we could differentiate (18) to derive a PDE for ∇ϕ. The resulting
PDE, however, will have ∂t(∇ϕ) grow linearly with ∇ϕ, and so if we apply (reverse) Gronwall’s
inequality, the resulting estimates for ∇ϕ will grow exponentially in time. To avoid this exponential
time dependence, we first perform a Hopf-Cole transform u = −2 log ϕ on (18) to derive the HJB
equation for u [9]

−∂tu −∆u + 1

2|∇u|2 + b · ∇u = 0,
u(T, x) = −2 log(ψ(x)).
(26)

Here we provide an example of classical Bernstein estimates (Proposition D.1 and Corollary D.2) to
obtain bounds for ∇ϕ without using Gronwall’s inequality [10]. The main idea is to derive a PDE
(60) for the function z = 1

2|∇u|2 by taking the gradient of (26) and then taking the inner product
with ∇u. Then by showing that the function w(t, x) = z −Cu for sufficiently large C attains its
maximum at (T, x0), we can show that

z(t, x) ≤w(T, x0) + Cu(t, x) ≤1

2|2∇log ψ|2 + C∥2 log ψ∥∞+ C∥u∥∞.
(27)

Using the maximum principle for (18), we find ∥u∥∞= ∥2 log ψ∥∞, yielding z(t, x) ≤C∥log ψ∥C1.
Assuming that ψ is Lipschitz continuous implies boundedness of z and therefore ∇ϕ for all time. A
similar result holds when ψ ∈L∞only (see (59)). Detailed proofs and related boundes are provided
in Proposition D.1. Applications of the bounds to derive the WUP is provided in Section A.1.
Remark 4.1 (The regularizing role of stochasticity). In our analysis, the stochasticity in SGMs
provides two types of regularizing effects. The first is early stopping, which adds a small amount of
Gaussian noise of the data distribution. This, in effect, is equivalent to running the noising process

8


---Page Break---
for a short amount of time and immediately mollifies the initial empirical distribution so that it has
a smooth density. Second, the Laplacian is the key mechanism that regularizes the test function
in (18) [9], which then allows us to bound, for example, the stronger ∥· ∥L1 norm by the weaker
d1-norm. Recall that in PDE theory, the stochasticity of a generative flow manifests as the Laplacian
operator in the Fokker-Planck and Kolmogorov backward equations [9]. Without the Laplacian the
regularizing the test functions in (18) would not be possible in general and, in fact, we would not
have access to long time behavior results.

5
Proof sketches — Score-based generative models are robust to errors

We now provide sketches of the proofs for the main generalization bounds in Theorems 3.2 and 3.3.
The full proofs are provided in Sections A.2 and A.3, respectively.

5.1
Theorem 3.2: ESM generalization bound

Theorem 3.2 is a generalization bound with respect to error from the score function approximation
with respect to the ESM objective (e3) and the choice of reference measure (e4). Assuming we have
an (uniform-in-time) L2-close approximation of the score function, first apply the WUP Theorem 3.1
with m1 =
1
vol(RTd) and m2 = ηπ(T, ·). The distance between m1 and m2 can be expressed in terms
of π by studying the long time behavior of periodic solutions to the heat equation. Proposition D.3
shows the heat equation is a contraction under d1. Applying it to m1 and m2 yields the desired result.

While Theorem 3.2 has no explicit assumptions on π, the assumption that the approximate score
function sθ is close in L2 to the true score implicitly implies regularity of π. Specifically, if sθ is
Lipschitz (which is true for neural networks in practice) and satisfies J (π, θ) < enn, then π must
have finite entropy. This implies that π necessarily has a density. See Proposition B.8 for the formal
statement and proof.

5.2
Theorem 3.3: Pointwise DSM generalization bound

In practice, the score function is learned via a Monte Carlo approximation of the DSM objective
(2). To avoid overfitting to the kernel estimator and memorizing the dataset [25], the time integral
is taken only over s ∈[ϵ, T] where ϵ is the early stopping parameter. This is equivalent to training
with the ESM objective with the true score replaced with the kernel approximation at time ϵ [20, 27].
To derive generalization bounds of SGMs trained via DSM, we establish the relationships between
(1) the mollified distribution πϵ = Γ(ϵ) ⋆π and the true distribution π, and (2) the DSM objective
JDSM(ηN,ϵ, θ) = J (ηN,ϵ, θ) and the ESM objective with respect to the score function of the
mollified distribution J (ηπϵ, θ). Formally, this strategy involves bounding the following two terms

d1(π, mg(T))
|
{z
}
Generalization bound w.r.t true distribution

≤
d1(π, πϵ)
|
{z
}
Early stopping error (e5)

+
d1(πϵ, mg(T)).
|
{z
}
Generalization bound w.r.t. mollified distribution

(28)

For the early stopping error, we use the regularizing properties of the heat equation to show that
d1(π, πϵ) ≤C√ϵ, where C only depends on the dimension d. This implies that, measured in d1,
early stopping only incurs a nominal C√ϵ error even if the π does not admit a density. This result
would not possible if we were to study generalization error in terms of KL or TV directly.

A bound for the second term is obtained by comparing the ESM objective value between the learned
score function and the true score function of the early stopped distribution, J (ηπϵ, θ) to the DSM
objective J (ηN,ϵ, θ). We present Theorem B.5 and its corollary, which under the assumption that
ηN,ϵ and ηπϵ have a lower bound δ, state that if J (ηN,ϵ, sθ) < enn, then J (ηπϵ, sθ) < e′
nn with

e′
nn = enn + C

1 + | log(δ)|
√ϵ
+
1
√

T
+ T∥sθ∥2
C2([0,T ]×Ω)

d1(πN, π).
(29)

The main idea behind Theorem B.5 is to note that the difference between the ESM and DSM objective
functions can be written as a difference in ISM objective functions plus the entropy difference between
ηN,ϵ and ηπϵ. Details for bounding this term is provided in Proposition B.2 and Lemma B.3.

To arrive at the final result, apply the WUP theorem to derive generalization bounds for d1(πϵ, mg(T))
under the assumption that J (ηπϵ, θ) < e′
nn, along with (29). Finally, combine this result with the
error due to the early stopping d1(π, πϵ). Full details of this proof is provided in Section A.3.

9


---Page Break---
6
Discussion: PDE regularity theory and UQ for generative modeling

Our main contribution is the study of generalization in score-based generative models from the
perspective of uncertainty quantification. The regularity theory of nonlinear PDEs is the key technical
tool that produces our results. We emphasize that the tools we use here can be used generative models
beyond SGMs, and that we have not pushed our analysis to the limits of our tools in SGMs. Moreover,
we also emphasize some downstream UQ applications for SGMs that may be of future interest.

6.1
The significance of the regularizing properties of SGMs

A surprising result of our work is deriving bounds for the stronger L1 distance in terms of the weaker
d1 distance (see (22) and Theorem 3.2). The key insight (Equation (21)) is that the evolution of
observables defined by the KBE (18) regularizes the test function. We have not fully exploited the
regularizing effects of (18) in this paper, as we only focused on L1 and d1 estimates.
Improved bounds in Sobolev spaces Hs.
To illustrate other extensions and choices of Y in
(21), observe that in the trivial case when b1 = b2 = 0, (19) simplifies to
R
λ(T, x)ψ(x)dx =
R
λ(0, x)(Γ(T) ⋆ψ)(x)dx, where Γ is the heat kernel. If ∥ψ∥∞≤1, then Γ ⋆ψ(T) ∈C∞and we
have estimates of the form d1(m1(T), m2(T)) ≤C(s, T, d)∥m1 −m2∥H−s for all s ∈N. When
b1, b2 are not identically zero we still expect such estimates, though they will depend on the regularity
of b1. To highlight the importance of regularizing effects, note that by [31], if π ∈P(Td), then for
the empirical measure πN, we expect d1(π, πN) ≲
1

N
1
d . However if s > d

2, ∥π −πN∥H−s ≲
1
√

N .
This suggests that improved regularity may influence overcoming the curse of dimensionality.

6.2
A connection to likelihood-free inference
Computing expectations with respect to posterior distributions is a key task in Bayesian inference.
Generative modeling, in particular, has a key role in future developments of likelihood-free inference
[32, 33, 34, 35]. For generative models to be trustworthy for inference, they must to be shown to
be robust. The WUP theorem provides error bounds for approximating expectations with respect to
some true unknown distribution, and may be significant for SGMs in likelihood-free inference.

For example, suppose we wish to estimate Eπh for some distribution π and observable h, and an
SGM mg(T) is constructed to approximate the expectation. Bounds of the form (6), such as the
WUP theorem, translate into guarantees on the expectations:
Eπh −Emg(T )h
 ≤d(mg(T), π) ≤F(e1, e2, e3, e4, e5).
(30)
In the context of SGM, the inequalities are a posteriori bounds, meaning they can be computed after
learning the model. Additional regularity on h may yield improved guarantees.

6.3
Enabling distributionally robust optimization (DRO)
Robust UQ methods are based on the perspective that learning any complex model will typically
involve multiple sources of uncertainty due to modeling choices, model reduction or learning from
imperfect data. These uncertainties are not just in parameters but are inherently present in the mathe-
matical model itself and will propagate to any predictions. There is substantial related work in recent
years using a distributional robustness perspective, [12], using divergences or probability metrics and
their variational representations to quantify the impact of model uncertainty around a baseline model
that may be either learned (e.g. a generative model) or could be just an empirical distribution from an
unknown true distribution. The approach can generally be described as quantifying an uncertainty
set around the baseline model that the worst-case distribution belongs to via some neighborhood
defined in terms of a probability divergence [13, 14], a Wasserstein distance [15] or maximum mean
discrepancy [16]. There are, however, drawbacks to each of these approaches: a divergence ball
contains only distributions with the same support as the baseline, while the uncertainty set may be
hard to determine practically. The WUP Theorem is a related robust UQ notion where the uncertainty
ball is in an IPM, e.g. 1-Wasserstein, MMD, or TV. However, it allows us to bypass the robust UQ
for stochastic processes relying on restrictive, path-space probability divergence-based approaches or
Girsanov’s Theorem, [36, 37]. Furthermore, the WUP Theorem bounds use PDE theory to provide a
computable uncertainty set, as we also demonstrate in the case of DSM, see Theorem 3.3.

Acknowledgements

M.K. and B.Z. are partially funded by AFOSR grant FA9550-21-1-0354. M.K. is partially funded by
NSF DMS-2307115 and NSF TRIPODS CISE-1934846.

10


---Page Break---
References

[1] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and
Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv
preprint arXiv:2011.13456, 2020.

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
in Neural Information Processing Systems, 33:6840–6851, 2020.

[3] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis.
Advances in neural information processing systems, 34:8780–8794, 2021.

[4] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma
with denoising diffusion GANs. arXiv preprint arXiv:2112.07804, 2021.

[5] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as
easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv
preprint arXiv:2209.11215, 2022.

[6] Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal
distribution estimators. In International Conference on Machine Learning, pages 26517–26582.
PMLR, 2023.

[7] Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence for score-based generative modeling with
polynomial complexity. Advances in Neural Information Processing Systems, 35:22870–22882,
2022.

[8] Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis.
arXiv preprint arXiv:2208.05314, 2022.

[9] Lawrence C Evans. Partial differential equations, volume 19. American Mathematical Society,
2022.

[10] Hung V Tran. Hamilton–Jacobi equations: theory and applications, volume 213. American
Mathematical Soc., 2021.

[11] Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative
modeling: User-friendly bounds under minimal smoothness assumptions. In International
Conference on Machine Learning, pages 4735–4763. PMLR, 2023.

[12] Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. ArXiv,
abs/1908.05659, 2019.

[13] Kamaljit Chowdhary and Paul Dupuis. Distinguishing and integrating aleatoric and epis-
temic variation in uncertainty quantification. ESAIM: Mathematical Modelling and Numerical
Analysis, 47(03):635–662, 2013.

[14] Jun ya Gotoh, Michael Jong Kim, and Andrew E.B. Lim. Robust empirical optimization is
almost the same as mean–variance optimization. Operations Research Letters, 46(4):448–452,
2018.

[15] Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven distributionally robust optimization
using the Wasserstein metric: performance guarantees and tractable reformulations. Mathemati-
cal Programming, 171(1):115–166, Sep 2018.

[16] Matthew Staib and Stefanie Jegelka. Distributionally robust optimization and generalization in
kernel methods. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and
R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran
Associates, Inc., 2019.

[17] Erich Novak, Mario Ullrich, Henryk Wo´zniakowski, and Shun Zhang. Reproducing kernels of
Sobolev spaces on Rd and applications to embedding constants and tractability. Analysis and
Applications, 16(05):693–715, 2018.

11


---Page Break---
[18] Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri. Score diffusion models without
early stopping: finite Fisher information is all you need. arXiv preprint arXiv:2308.12240,
2023.

[19] Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence of score-based generative modeling for
general dat a distributions. In International Conference on Algorithmic Learning Theory, pages
946–985. PMLR, 2023.

[20] Kaihong Zhang, Heqi Yin, Feng Liang, and Jingbo Liu. Minimax optimality of score-based dif-
fusion models: Beyond the density lower bound assumptions. arXiv preprint arXiv:2402.15602,
2024.

[21] Dohyun Kwon, Ying Fan, and Kangwook Lee. Score-based generative modeling secretly
minimizes the wasserstein distance. Advances in Neural Information Processing Systems,
35:20205–20217, 2022.

[22] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their
Applications, 12(3):313–326, 1982.

[23] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable
approach to density and score estimation. In Uncertainty in Artificial Intelligence, pages
574–584. PMLR, 2020.

[24] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural
computation, 23(7):1661–1674, 2011.

[25] Jakiw Pidstrigach. Score-based generative models detect manifolds. Advances in Neural
Information Processing Systems, 35:35852–35865, 2022.

[26] Yang Song. Score-based generative modeling through stochastic differential equations (sdes) -
PyTorch implementation. https://github.com/yang-song/score_sde_pytorch, 2020.
Accessed: May 2, 2024.

[27] Sixu Li, Shi Chen, and Qin Li. A good score does not lead to a good generative model. arXiv
preprint arXiv:2401.04856, 2024.

[28] Benjamin J. Zhang, Siting Liu, Wuchen Li, Markos A. Katsoulakis, and Stanley J. Osher. Wasser-
stein proximal operators describe score-based generative models and resolve memorization,
2024.

[29] Assyr Abdulle, David Cohen, Gilles Vilmart, and Konstantinos C Zygalakis. High weak order
methods for stochastic differential equations based on modified equations. SIAM Journal on
Scientific Computing, 34(3):A1800–A1823, 2012.

[30] Benjamin J Zhang and Markos A Katsoulakis. A mean-field games laboratory for generative
modeling. arXiv preprint arXiv:2304.13534, 2023.

[31] Nicolas Fournier and Arnaud Guillin. On the rate of convergence in wasserstein distance of the
empirical measure. Probability theory and related fields, 162(3):707–738, 2015.

[32] Ricardo Baptista, Bamdad Hosseini, Nikola B Kovachki, and Youssef Marzouk. Conditional
sampling with monotone GANs: from generative models to likelihood-free inference. arXiv
preprint arXiv:2006.06755, 2020.

[33] Ricardo Baptista, Lianghao Cao, Joshua Chen, Omar Ghattas, Fengyi Li, Youssef M Marzouk,
and J Tinsley Oden. Bayesian model calibration for block copolymer self-assembly: Likelihood-
free inference and expected information gain computation via measure transport. Journal of
Computational Physics, page 112844, 2024.

[34] Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. Conditional
simulation using diffusion schrödinger bridges. In Uncertainty in Artificial Intelligence, pages
1792–1802. PMLR, 2022.

[35] Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical
imaging with score-based generative models. arXiv preprint arXiv:2111.08005, 2021.

12


---Page Break---
[36] Paul Dupuis, Markos A Katsoulakis, Yannis Pantazis, and Petr Plechác. Path-space information
bounds for uncertainty quantification and sensitivity analysis of stochastic dynamics. SIAM/ASA
Journal on Uncertainty Quantification, 4(1):80–111, 2016.

[37] Birrell, Jeremiah, Katsoulakis, Markos A., and Rey-Bellet, Luc. Quantification of model
uncertainty on path-space via goal-oriented relative entropy. ESAIM: M2AN, 55(1):131–169,
2021.

13


---Page Break---
A
Proofs of main results

Throughout the proofs we will make frequent use of the fact that if Γ : [0, ∞) × Ω→R denotes the
heat kernel on Ω= Rd or Ω= RTd, then there exists a dimensional constant C = C(d) > 0 such
that

|∇Γ(t) ⋆g| ≤C ∥g∥∞
√

t .
(31)

Furthermore, recall that constants are subject to change from line to line but always maintain a
variable dependence as in the corresponding statement.

A.1
Theorem 3.1

The proof of Theorem 3.1 follows the strategy outlined in Section 4 and uses the different gradient
estimates from the HJB equations provided in Section D.1.

Proof. Let λ = m1 −m2, which satisfies
∂tλ −∆λ −div(λb1 + m2(b2 −b1)) = 0 in (0, T) × RTd,
λ(0) = m1 −m2 in RTd.
(32)

For a fixed ψ : Ω→R, for which we will specify bounds latter, let ϕ : [0, T] × Ω→R be given by
−∂tϕ −∆ϕ + b1 · ∇ϕ = 0 in [0, T) × Ω,
ϕ(T, x) = ψ(x) in Ω.
(33)

Testing against ϕ in (32) and using (33), we obtain
Z
λ(T, x)ψ(x)dx =
Z
λ(0)ϕ(0, x)dx −
Z T

0

Z
m2(s)∇ϕ(s) · (b2 −b1)(s)dxds.
(34)

• Proof of (8) and (9): Assume that ∥ψ∥∞≤1 which then implies

−1 ≤ϕ ≤1.

Since
R
dλ(t) = 0 for all t ∈[0, T] up to adding a constant we can assume without loss of
generality in (34), that
1 ≤ψ ≤3 and 1 ≤ϕ ≤3.
Then from estimate (63) in Corollary D.2, we have

(T −t)∥∇ϕ(t)∥2
∞≤C(T∥∇b∥∞+ 1)

=⇒∥∇ϕ(t)∥∞≤C

p

T∥∇b1∥∞+ 1
√

T −t
.

Next, we note that
Z
λ(T, x)ψ(x)dx =
Z
λ(0)ϕ(0, x)dx −
Z T

0

Z
m2(s)∇ϕ(s) · (b2 −b1)(s)dxds

≤d1(m1, m2)∥∇ϕ(0)∥∞+
Z T

0

 Z
|∇ϕ(s)|2m2(s)dx
 1

2  Z
m2(s)|b2−b1|2(s)dx
 1

2 ds

≤d1(m1, m2)∥∇ϕ(0)∥∞+ sup
0≤t≤T
∥(b2 −b1)(t)∥L2(m2(t))

Z T

0
∥∇ϕ(s)∥∞ds,

≤C(
p

T∥∇b1∥∞+ 1)
d1(m1, m2)
√

T
+
√

T
sup
0≤t≤T
∥(b2 −b1)(t)∥L2(m2(t))

.

Taking the supremum over ∥ψ∥∞≤1 yields

[∥m2(T) −m1(T)∥L1(Ω) ≤C(
p

T∥∇b1∥∞+ 1)×
d1(m1, m2)
√

T
+
√

T
sup
0≤t≤T
∥(b2 −b1)(t)∥L2(m2(t))


.

14


---Page Break---
Finally, if instead we used the bound
Z
λ(0)ϕ(0, x)dx ≤∥λ(0)∥L1(Ω)∥ϕ∥∞≤C∥m1 −m2∥L1(Ω)

we obtain (9).

• Proof of (10): Let Ω= RTd and consider a ψ with ∥∇ψ∥∞≤1. Without loss of generality
we can assume ψ(0) = R + 1 and thus

ψ(x) = ψ(x) −ψ(0) + ψ(0) ≥−|x −0| + R + 1 ≥−R + R + 1 = 1

and similarly
ψ(x) ≤2R + 1.

Therefore by maximum principle we have that

1 ≤ϕ(t, x) ≤2R + 1,

and so
∥ψ∥C1 = ∥ψ∥∞+ ∥∇ψ∥∞≤C(1 + R).

From (34) we have
Z
λ(T, x)ψ(x)dx ≤d1(m1, m2)∥∇ϕ(0)∥∞+ sup
0≤t≤T
∥∇ϕ(t)∥∞∥b1 −b1∥L2(m2).

Thus by Corollary D.2, estimate (62) we obtain
Z
λ(T, x)ψ(x)dx ≤C
q

(1 + ∥∇b∥∞)∥v∥3
C1(d1(m1, m2) + ∥b1 −b1∥L2(m2))

≤CR
3
2 (1 +
p

∥∇b1∥∞)

d1(m1, m2) + ∥b1 −b1∥L2(m2)

,

which after taking the supremum over ∥∇ψ∥∞≤1 yields (10).

A.2
Theorem 3.2

The proof of Theorem 3.2 follows by an application of Theorem 3.1.

Proof. Let b1 = sθ, m1 =
1
vol(RTd) and b2 = ∇log(ηπ), m2 = ηπ(T). Note that if mi, i = 1, 2
solve
∂tmi −∆mi −div(mibi) = 0 in [0, T] × RTd,
mi(0) = mi in [0, T] × RTd,
(35)

then m1 = mg, m2(t) = ηπ(T −t). Thus, by an application of Theorem 3.1 we obtain

d1(mg(T), π) ≤CR
3
2 (1 +
p

∥∇sθ∥∞)(d1(
1
vol(RTd), ηπ) + ∥sθ −∇log(ηπ)∥L2(ηπ))

≤CR
3
2 (1 +
p

∥∇sθ∥∞)

(Re−ωT

R2 d1

π,
1
vol(RTd)


+ √enn

,

where in the last inequality we used Proposition D.3. The second claim follows again by Theorem
3.1 and the fact that

d1(ν1, ν2) =
sup
∥∇g∥∞≤1

Z

RTd gd(ν1 −ν1) ≤R∥ν2 −ν1∥L1.

15


---Page Break---
A.3
Theorem 3.3

The proof of Theorem 3.3 follows the same strategy as in Theorems 3.1 and 3.2. The main technical
difference is that we only have a bound between our generated score function sθ and the score
function generated by the sample ∇log(ηN,ϵ) of the form
Z T

0

Z
|sθ −∇log(ηN,ϵ)|2dηN,ϵ(s)ds ≤enn

and so we need to produce a bound between sθ and the true score function ∇log(ηϵ). This last step
is shown in Section B and in particular Theorem B.5.

Proof of Theorem 3.3. Consider the function ηπ,ϵ : [0, T] × RTd →R given by
∂tηπ,ϵ −∆ηπ,ϵ = 0 in (0, T) × RTd,
ηπ,ϵ(0) = πϵ in RTd.
(36)

Define the drifts
bθ∗(t, x) := sθ∗(T −t, x)
and
bπϵ(t, x) := ∇log(ηπϵ)(T −t, x),
and let mϵ(t, x) = ηπ,ϵ(T −t, x) which satisfies
∂tmϵ −∆mϵ −div(mϵbπϵ(t, x)) = 0,
mϵ = ηπ,ϵ(T, x).
(37)

Finally, we consider the distribution of our generated sample mg : [0, T] × Ω→R which is given by
(
∂tmg −∆mg −div(mgbθ∗) = 0 in (0, T] × Ω,
mg(0) =
1
vol(RTd) in Ω.
(38)

We have the following

d1(π, mg(T)) ≤d1(π, πϵ)

1 + d1(πϵ, mg(T))

2,
(39)

We will bound each of the terms separately.

1. We recall that πϵ = Γ(ϵ) ⋆π where Γ is the heat kernel. Therefore, by the dual formulation
of d1 we have that

d1(π, πϵ) =
sup
∥∇g∥∞≤1

Z
g(x)d(πϵ −π) =
sup
∥∇g∥∞≤1

Z
(Γ(ϵ) ⋆g(x) −g(x))dπ(x).

Although estimates on (Γ(ϵ) ⋆g(x) −g(x)) for a function ∥∇g∥∞≤1 are classical for the
readers convenience we provide a quick proof. Let v(t) = Γ(t) ⋆g, which solves
∂tv −∆v = 0,
v(0) = g.
(40)

Then for all x ∈Ω,

|v(ϵ, x) −v(0, x)| ≤
Z ϵ

0
|∂tv(s, x)|ds =
Z ϵ

0
|∆v(s, x)|ds

=
Z ϵ

0
|(∇Γ(s) ⊗⋆∇g)(x)|ds ≤C∥∇g∥∞

Z ϵ

0

1
√sds ≤C√ϵ.

Where in the above we used estimate (31). Thus, we have

d1(π, πϵ) ≤C√ϵ,

where C = C(d) > 0.

16


---Page Break---
2. First we note that πϵ = mϵ(T). Thus applying Theorem 3.1 for

m1 = mg, b1 = bθ∗

m2 = mϵ, b2 = bπϵ

yields

d1(mg(T), πϵ) = d1(mg(T), mϵ(T))

≲R
3
2 (1 +
p

∥∇b1∥∞)(d1(mϵ(0),
1
vol(RTd)) +
√

T∥bπϵ −bθ∗∥L2(mϵ)).

By Proposition D.3, we have

d1(mϵ(0),
1
vol(RTd)) = d1(ηπ,ϵ(T),
1
vol(RTd)) ≤CRe−ωt

R2 d1(πϵ,
1
vol(RTd))

≤CR2e−ωt

R2 .

Finally, from Theorem B.5

∥bπϵ −bθ∗∥2
L2(mϵ) = J (ηπϵ, θ∗) ≤e′
nn

= enn + C

1 + | log(δ)|
√ϵ
+
1
√

T
+ T∥sθ∥2
C2([0,T ]×Ω)

d1(πN, π)

Therefore, if T ≥1, putting everything together, we have shown that up to a dimensional constant

d1(π, mg(T)) ≤C

 
√ϵ + R
3
2

1 +
p

∥∇b1∥∞

×

"

R2e−ωT

R2 +

s

T

enn + C

1 + | log(δ)|
√ϵ
+ T∥sθ∥2
C2([0,T ]×Ω)


d1(πN, π)
# !

B
Analysis of Score Matching functionals

In this section, we gather all the technical results regarding the connections between ISM, DSM, and
ESM score matching functionals. For the readers convenience we recall some of our notations. Let
Ω⊂Rd and m0 ∈P(Ω). We will denote by ρm0 : [0, T] × Ω→[0, ∞) the solution of
∂tρm0 −∆ρm0 = 0 in (0, T] × Ω,
ρm0(0) = m0 in Ω,
(41)

and we may drop the superscipt m0 when it is clear from context.

First we state a simple result that justifies the formula

J (ρm0, θ) = JL(ρm0, θ) + 4∥∇√ρm0∥2
2,
(42)

which formally follows by

J (ρ, θ) =
Z T

0

Z 
|sθ|2dρ(s) −2sθ · ∇log(ρ) + |∇log(ρ)|2
dρ(s)ds

= JL(ρ, θ) +
Z T

0

Z |∇ρ|2

ρ
dxds

= JL(ρ) + 4∥∇√ρ∥2
2.

However ∥∇√ρm0∥2 may not be finite, thus we record the exact statement along with some technical
facts for later use in the following Proposition.

17


---Page Break---
Proposition B.1. Let m0 be a probability density in Ωsuch that m0(x) log(m0(x)) ∈L1(Ω) and
ρ : [0, T] × Ω→R be given by (41). Then, the following hold:

1. There exists a universal constant C > 0, such that

sup
t∈[0,T ]
∥ρ(t) log(ρ(t))∥L1(Ω) + ∥∇√ρ∥2
L2([0,T ]×Ω) ≤CT∥m0 log(m0)∥L1(Ω),

and thus (42) holds.

2. 4∥∇√ρ∥2
2 =
R

Ωm0 log(m0) −ρ(T) log(ρ(T))dx.

Proof. Testing (41) against f ′(ρ) for a smooth function f, we obtain
Z
f(ρ(T, x))dx +
Z T

0

Z
f ′′(ρ)|∇ρ|2dxdt =
Z
f(m0(x))dx

and the choice of f(x) = x log(x) yields
Z
ρ(T, x) log(ρ(T, x))dx +
Z T

0

Z |∇ρ|2

ρ
dxdt =
Z
m0(x) log(m0(x))dx,

which shows the second claim.
Finally, if m0 log(m0)
∈
L1 this yields estimates on
sup
t∈[0,T ]
∥ρ(t) log(ρ(t))∥1, ∥∇√ρ∥2
2 since

Z
|ρ(T, x) log(ρ(T, x))|dx =
Z
ρ(T, x) log(ρ(T, x))dx + 2
Z
(ρ(T, x) log(ρ(T, x)))−dx

≤
Z
ρ(T, x) log(ρ(T, x))dx + 2e−1|Ω|,

where we used the fact that x log(x) ≥−1

e.

The rest of the subsection justifies the steps outlined in Subsection 5.2. Namely starting from a
estimate of the form
J (ηN,ϵ, θ) ≤enn

our goal is to obtain an estimate of the form

J (ηϵ, θ) ≤e′
nn.

For this we use formula (42). In particular we first use the linear dependence of JL with respect to
the underlying measure to bound

|JL(ηN,ϵ, θ) −JL(ηϵ)|.

Then, instead of comparing the gradients on the difference

∥∇log(ηN,ϵ)∥2
2 −∥∇log(ηϵ)∥2
2

we use Proposition B.1 from which it is enough to bound the entropy terms. The next two Propositions
provide these results along with some other technical facts.

Proposition B.2. Let T > 0 and πi for i = 1, 2 denote two probability measures in Ωsuch that
∥πi log(πi)∥L1 < ∞and ρi the corresponding solutions to (41). Then, there exists a dimensional
constant C > 0 such that

|JL(ρ2, θ) −JL(ρ1, θ)| ≤CT
sup
t∈[0,T ]
d1(ρ2(t), ρ1(t))∥sθ∥2
C2([0,T ]×Ω)

≤CTd(π1, π2)∥sθ∥2
C2([0,T ]×Ω).

In the above sθ indicates the score function approximation.

18


---Page Break---
Proof. We recall that

JL(ρi, θ) =
Z T

0
Eρi
h1

2|sθ|2 + div(sθ)
i
ds =
Z T

0

Z 1

2|sθ|2 + div(sθ)dρi(s)ds.

Therefore, if

gθ = 1

2|sθ|2 + div(sθ)

|JL(ρ2, sθ) −JL(ρ1, sθ)| =

Z T

0

Z 1

2|sθ|2 + div(sθ)d(ρ2(s) −ρ1(s))ds


=

Z T

0

Z
gθd(ρ2 −ρ1)(s)ds
 ≤sup
0≤t≤T
∥∇gθ∥∞

Z T

0
d1(ρ2(s), ρ1(s))ds

≤T
sup
0≤t≤T
∥∇gθ∥∞sup
0≤t≤T
d1(ρ2(t), ρ1(t)).

Since
∥∇gθ∥∞≤∥sθ∥2
C2
to conclude we need to show that

sup
0≤t≤T
d1(ρ2(t), ρ1(t)) ≤d1(π2, π1).

However this follows from our Lemma 3.1 applied for b1 = b2 = ⃗0 and mi = πi for i = 1, 2.

Lemma B.3. Let πϵ, ˆπN,ϵ, 0 < δ < ϵ be as in Theorem 3.3 with ϵ < 1. There exists a dimensional
constant C = C(d) > 0 such that

d1(ˆπN,ϵ, πϵ) ≤d1(πN, π)
(43)

∥πϵ −ˆπN,ϵ∥L1(Ω) ≤C d1(πN, π)
√ϵ
,
(44)

and

∥log(πϵ)πϵ −log(ˆπN,ϵ)ˆπN,ϵ∥L1(Ω) ≤C(1 + | log(δ)|)d1(πN, π)
√ϵ
.
(45)

Moreover, let C > 0 denote the constant from Proposition D.3 and T ≥R2 large enough such that

2CR−de−ωT

R2 ≤1

2
1
vol(RTd),

then up to a dimensional constant C > 0 we have
Z
log(ηN,ϵ(T))ηN,ϵ(T) −log η(T))η(T)dx ≤C
√

T


1 + d log(R)

d1(π, ˆπN).
(46)

Proof. Let ρN,ϵ, ρϵ : [0, T] × Ω→R, be solutions to
∂tρ −∆ρ = 0 in Ω× (0, ϵ),
ρ(0) = ρ0,
(47)

for ρ0 = π, ρ0 = πN respectively. Since πϵ = Γ(ϵ)⋆π, ˆπN,ϵ = Γ(ϵ)⋆πN we note that πϵ = ρϵ(ϵ, x)
and ˆπN,ϵ = ρϵ,N(ϵ, x). Using the same adjoint method as in the proof of Theorem 3.1 for a function
ψ : Ω→R we consider the function ϕ : [0, ϵ] × Ω→R given by
−∂tϕ −∆ϕ = 0
ϕ(ϵ) = ψ.
(48)

Testing against ϕ in the equation for λ = ρϵ −ρϵ,N yields
Z
(ˆπN,ϵ −πϵ)ψ(x)dx =
Z
ϕ(0, x)d(π −πN)(x).

19


---Page Break---
First we consider ∥ψ∥∞≤1 which from (31), yields

∥∇ϕ(0)∥∞≤∥ψ∥∞
√ϵ
≤C
√ϵ.

Thus, for any ∥ψ∥∞≤1 we obtain
Z
(ˆπN,ϵ −πϵ)ψ(x)dx ≤Cd1(π, πN)
√ϵ
,

which implies

∥ˆπN,ϵ −πϵ∥L1(ω) ≤Cd1(π, πN)
√ϵ
.

Next, by considering Lip(ψ) ≤1 which implies

∥∇ϕ(0)∥∞≤1

we obtain
d1(ˆπN,ϵ, πϵ) ≤d1(π, πN).
For estimate (45) we note that

ˆπN,ϵ log(ˆπN,ϵ) −πϵ log(πϵ)
 =


Z 1

0
(1 + log(sˆπN,ϵ + (1 −s)πϵ))ds(ˆπN,ϵ −π)


≤(1 + | log(δ)|)|ˆπN,ϵ −πϵ|.

Therefore,

∥ˆπN,ϵ log(ˆπN,ϵ)−πϵ log(πϵ)∥L1(Ω) ≤(1+| log(δ)|)∥ˆπN,ϵ −πϵ∥L1 ≤C(1+| log(δ)|)d1(πN, π)
√ϵ
.

For estimate (46) we note that for any convex function h

h(x) −h(y) ≥∇h(y) · (x −y) =⇒h(y) −h(x) ≤∇h(y) · (y −x).

Hence by applying the above to h(x) = x log(x) we obtain
Z
log(ηN,ϵ(T))ηN,ϵ(T) −log η(T))η(T)dx ≤
Z
(1 + log(ηN,ϵ(T)))d(ηN,ϵ(T) −ηϵ(T))

≤∥1 + log(ηN,ϵ(T))∥∞∥ηN,ϵ(T) −ηϵ(T)∥L1.
We now note that for any g : RTd →R, ∥g∥∞≤1
Z
g(x)d(ηN,ϵ −ηϵ)(T) =
Z
Γ(T) ⋆gd(ˆπN,ϵ −πϵ) ≤∥∇Γ(T) ⋆g∥∞d1(ˆπN,ϵ, πϵ)

≤C
√

T
d1(ˆπN,ϵ, πϵ).

Taking the supremum yields

∥ηN,ϵ(T) −ηϵ(T)∥L1 ≤C
√

T
d1(ˆπN,ϵ, πϵ) ≤d1(ˆπN,ϵ, πϵ).

Finally, from Proposition D.3 above applied to ρ = ηN,ϵ −
1
vol(RTd) we have

ηN,ϵ(T, x) ≥−∥ρ(T)∥∞+
1
vol(RTd) ≥1

2
1
vol(RTd)

as well as
ηN,ϵ(T, x) ≤
1
vol(RTd) + ∥ρ(T)∥∞≤2
1
vol(RTd)
by our assumptions on T > 0. Therefore,

∥1 + log(ηN,ϵ(T))∥∞≤C(1 + d log(R))

and the result follows.

20


---Page Break---
Remark B.4. It is worth noting that the empirical sample πN could have been regularized by any
smooth kernel ρ. However the choice of the heat kernel which provides the same result as early
stopping behaves quite nicely with respect to the metric d1 since it is a contraction. This is evident in
Lemma B.3 above where we see that we do not pay any additional cost due to the mollification in
(43).

In fact the above techniques show the following general result regarding ESM to DSM bounds.
Combining the above we can state the following general result, about ESM to DSM bounds.
Theorem B.5. Let π1, π2 ∈P(Ω) denote two probability densities, such that

∥πi log(πi)∥L1(Ω) < ∞for i = 1, 2
(49)

and δ > 0 such that
πi(x) ≥δ.
Define ρi : [0, T] × Ω→R as the solutions to
∂tρi −∆ρi = 0 in (0, T] × Ω,
ρi(0) = πi in Ω.
(50)

Then, up to a dimensional constant C = C(d) > 0
J (ρ2, θ) −J (ρ1, θ)
 ≤C

T∥sθ∥2
C2d1(π1, π2) + (1 + | log(δ)|)∥π2 −π1∥L1

.
(51)

B.1
Proof of Theorem 3.3

Now we can prove Theorem 3.3.

Proof. From Proposition B.1 we obtain

J (ηϵ, sθ∗) = JL(ηϵ, sθ∗) + 2∥∇√ηϵ∥2
2

= J (ηN,ϵ, sθ∗) + 2

∥∇√ηϵ∥2
2 −∥∇
p

ηN,ϵ∥2
2∥

+

JL(ηϵ, sθ∗) −JL(ηN,ϵ, sθ∗)

.

By assumption we have the bound

J (ηN,ϵ, sθ∗) ≤enn.

By Proposition B.1
2

∥∇√ηϵ∥2
2 −∥∇
p

ηN,ϵ∥2
2


=
Z

Ω
πϵ(x) log(πϵ(x))−ˆπN,ϵ(x) log(ˆπN,ϵ(x))dx−
Z

Ω
ηϵ(T) log(ηϵ(T))−ηN,ϵ(T) log(ηN,ϵ(T))dx.

From Lemma B.3 we have that

∥πϵ log(πϵ) −ˆπN,ϵ log(ˆπN,ϵ)∥L1(Ω) ≤C(1 + | log(δ)|)d1(πN, π)
√ϵ
.

Using the same argument and the fact that from maximum principle ηϵ(T), ηN,ϵ ≥δ > 0, we can
show

∥η(T) log(η(T)) −ηN,ϵ log(ηN,ϵ(T))∥L1(Ω) ≤(1 + | log(δ)|)∥ηϵ(T) −ηN,ϵ(T)∥L1(Ω)

≤C
√

T
d1(πϵ, ˆπN,ϵ) ≤C
√

T
d1(π, πN).

Furthermore, by Proposition B.2 we have
JL(ηϵ, sθ∗)−JL(ηN,ϵ, sθ∗)
 ≤CTd1(ˆπN,ϵ, πϵ)∥sθ∥2
C2([0,T ]×Ω) ≤CTd1(πN, π)∥sθ∥2
C2([0,T ]×Ω).

Putting everything together we have

J (η, sθ∗) ≤enn + C

1 + | log(δ)|
√ϵ
+
1
√

T
+ T∥sθ∥2
C2([0,T ]×Ω)


d1(πN, π).

21


---Page Break---
As an immediate Corollary we have the following.

Corollary B.6. Using the same notation as in Section 3.5, if enn > 0 is such that

0 ≤J (ηN,ϵ, θ∗) < enn
(52)

then
0 ≤J (ηπϵ, θ∗) < e′
nn,
(53)

where for a dimensional constant C = C(d) > 0

e′
nn = enn + C

1 + | log(δ)|
√ϵ
+
1
√

T
+ T∥sθ∥2
C2([0,T ]×Ω)

d1(πN, π).
(54)

Remark B.7 (On regularity implications of NN approximation). It is important to note that the norm
of our NN approximation ∥∇sθ∥∞is in fact dependent on ϵ > 0 as we can see from the following

∥sθ∥C1 ≳∥sθ∥L2(ηN,ϵ) ≳∥∇log(ηN,ϵ)∥L2(ηN,ϵ) −√enn

and ∥∇log(ηN,ϵ)∥L2(ηN,ϵ) blows up as ϵ →0.

We finish this subsection of technical results with an observation about the implications on the
existence of a smooth NN-approximation.

Proposition B.8 (ESM implies regularity). Let π, sθ and enn > 0 be as above for Ω= RTd. Assume
moreover, that ∥∇sθ∥∞< ∞. Then, π = π(x) admits a density and in fact ∥π log(π)∥L1(Ω) < ∞.

Proof. Since ∥∇sθ∥∞< ∞, for some constant C = C(R, d, enn) > 0 it holds

2∥∇√ηπ∥2
L2 =
Z T

0

Z
|∇log(ηπ)|2dηπ(s, x)ds ≤C.

Moreover, by the regularizing properties of the diffusion we have
Z
ηπ(T) log(ηπ(T))dx < ∞

and the result follows from Proposition B.1.

C
Average DSM generalization

Using the previous results we can in fact show a result about the average error in Theorem B.5. The
main observation here is that by Jensen’s inequality when we take expectation with respect to the
sample πN we no longer require a lower bound on our densities. However, as we will see we require
a very restrictive assumption on the norm of our score function approximation

Theorem C.1. (Average DSM generalization) Let enn, A > 0 and assume that for each sample ˆπN
from π there exists a sθ∗such that
J(ηN,ϵ, θ∗) ≤enn,

with
∥sθ∗∥C2 ≤A.
(55)

Let mg(T) be the generated distribution from ˆπN (which is also random since ˆπN is random). Let
C > 0 be the dimensional constant appearing in Proposition D.3 and T ≥R2 large enough such
that
2CR−de−ωT

R2 ≤1

2
1
vol(RTd),

then
E
h
d1(π, mg(T))
i
≤CR
3
2 (1 +
√

A)(R2e−ωT

R2 +
p

Te′nn),

where

e′
nn = ϵ + enn +
C

N
1
2d


TRA2 +
1
√

T
(1 + d log(R))

.

22


---Page Break---
Remark C.2. It is important to emphasize in the above that we are assuming that we may pick both
an error enn > 0 and a bound A > 0 on the gradient of sθ globally for all random samples ˆπN.
Moreover, sθ is a weak approximation of ∇log(ηN,ϵ) which although smooth for all ϵ > 0 it does
exhibit a blow-up as ϵ > 0 tends to zero. Therefore, since by triangular inequality up to a constant
C = C(R, d) > 0

CA ≥∥sθ∥C2 ≥∥sθ∥L2(ηN,ϵ) ≥∥D
p

ηN,ϵ∥L2 −enn

the constant A > 0 will also exhibit a blow-up which can make the approximation worse. Therefore
there could be a potential trade-off between NN approximation enn becoming small and the value of
the norm A.

First we require a preliminary lemma.

Lemma C.3. Under the same assumptions as in Theorem C.1 we have that

E
h
J (ηϵ, sN
θ )
i
≤enn + C

N
1
d


TA2 +
1
√

T
(1 + d log(R))

.
(56)

Proof. We note that
J (ηϵ, sθ∗) = JL(ηϵ, sθ∗) + 4∥D√ηϵ∥2
2

= J (ηN,ϵ, sθ∗) + 2

∥D√ηϵ∥2
2 −∥D
p

ηN,ϵ∥2
2∥

+

JL(η, sθ∗) −JL(ηN,ϵ, sθ∗)

.

By assumption we have the bound

J (ηN,ϵ, sθ∗) ≤enn.

By Proposition B.1

2

∥D√ηϵ∥2
2 −∥D
p

ηN,ϵ∥2
2


=
Z

Ω
πϵ(x) log(πϵ(x)) −ˆπN,ϵ(x) log(ˆπN,ϵ(x))dx −
Z

Ω
ηϵ(T) log(ηϵ(T)) −ηN,ϵ log(ηN,ϵ(T))dx.

Taking expectations with respect to the sample ˆπN we get

E
h Z

Ω
πϵ(x) log(πϵ(x))−ˆπN,ϵ log(ˆπN,ϵ)dx
i
=
Z

Ω
πϵ(x) log(πϵ(x))−E
h
ˆπN,ϵ(x) log(ˆπN,ϵ(x))
i
dx

≤
Z

Ω
πϵ(x) log(πϵ(x)) −E
h
ˆπN,ϵ(x)
i
log(E
h
ˆπN,ϵi
)dx = 0.

In the above we used Jensen’s inequality along with the fact that

E
h
ˆπN,ϵi
= πϵ.

Moreover, by Lemma B.3 we have that

E
h
ηN,ϵ log(ηN,ϵ(T)) −
Z

Ω
ηϵ(T) log(ηϵ(T))dx
i
≤C
√

T
(1 + d log(R))E
h
d1(π, ˆπN)
i

≤
C
√

TN
1
d (1 + d log(R)).

Finally, from Proposition B.2 we have

E
hJL(η, sθ∗) −JL(ηN,ϵ, sθ∗)

i
≤CTE
h
∥sN
θ ∥2
C2d1(π, ˆπN)
i
≤CT

N
1
d A2

Thus,

E
h
J (ηϵ, θ∗)
i
≤enn + C

N
1
d


TA2 +
1
√

T
(1 + d log(R))

,

and the result follows.

23


---Page Break---
Proof. By Theorem 3.1 we have that
d1(π, mg(T)) ≤√ϵ + d1(πϵ, mg(T)).
And
d1(mg(T), πϵ) = d1(mg(T), mϵ(T))

≤CR
3
2 (1 +
p

∥Db1∥∞)(d1(mϵ(0),
1
vol(RTd)) +
√

T∥bπϵ −bθ∗∥L2(mϵ))

≤CR
3
2 (1 +
√

A)

R2e−ωT

R2 +
p

TJ (ηϵ, θ∗)

.

Taking expectations and using the Lemma above yields the result.

D
Technical results

D.1
HJB estimates

In this subsection, we gather all the regularity estimates for the HJB equations. The results are for
the most part standard, we include however detailed proofs for the readers convenience. For further
study into the techniques and topics we refer to [9, 10]. Although as mentioned in the introduction
in the current work we assumed σ(t) =
√

2 and f = 0 in the OU processes (1), here we prove the
regularity results for a general non-degenerate diffusion α(t). The addition of f may be incorporated
in the drift b.
Proposition D.1. Let Ω⊂Rd, v : Rd →R and b : [0, T] × Ω→R be given with ∥∇b∥∞< ∞.
Consider the solution u : [0, T] × Ω→R of
(
−∂tu −α(t)∆u + α(t)

2 |∇u|2 + b · ∇u = 0 in [0, T) × Ω,
u(T, x) = v(x) in Ω.
(57)

Assume moreover that for some M > 0

0 < 1

M ≤α(t).

Then up to a universal constant C > 0 the following holds

∥∇u(t)∥2
∞≤C(1 + ∥∇b∥∞M)∥v∥C1.
(58)

(T −t)∥∇u(t)∥2
∞≤CM(T∥∇b∥∞+ 1)∥v∥∞.
(59)

Proof. Let z : [0, T] × Ω→R be given by

z = 1

2|∇u|2.

Differentiating the equation for u in i = 1, · · · , d yields
−∂tui −α(t)∆ui + α(t)∇u · ∇ui + b · ∇ui + bi · ∇u = 0,
and thus multiplying by ui and summing over i, we obtain the following equation for z
−∂tz −α(t)∆z + α(t)|∇2u|2 + b · ∇z + α(t)∇u · ∇z + ⟨∇u∇b, ∇u⟩= 0,
z(T) = 1

2|∇v|2.
(60)

In the above ∇b is the matrix with entries

[∇b]i,j = ∂bi

∂xj
for
b = (b1, · · · , bd).
Let w : [0, T] × Ω→R be given by
w(t, x) = z −Cu
where C > 0 is a constant to be determined later. Assume that the function w achieves its maximum
at (s0, x0) ∈[0, T] × Ω. We will look at the following cases:

24


---Page Break---
• Case 1: Assume that 0 ≤s0 < T and we will show that for C > 0 large enough this leads
to a contradiction. At (s0, x0) we have the optimality conditions

∇z(s0, x0) = C∇u(s0, x0)

∆z(s0, x0) ≤C∆u(s0, x0)
∂tz(s0, x0) ≤C∂tu(s0, x0).
Hence,

C(−∂tu(s0, x0) −α(s0)∆u(s0, x0) + b · ∇u(s0, x0))
≤(−∂tz(s0, x0) −α(s0)∆z(s0, x0) + b · ∇z(s0, x0))

and using the respective equations for u, z we obtain that at (s0, x0)

−C α(s0)

2
|∇u|2 ≤−α(s0)|∇2u|2 −⟨∇u · ∇b, ∇u⟩−α(s0)∇u · ∇z.

Using the optimality conditions yields

α(s0)C

2 |∇u|2 + α(s0)|∇2u|2 ≤−⟨∇u · ∇b, ∇u⟩≤∥∇b∥∞|∇u|2,

which is a contradiction if C ≥2∥∇b∥∞M. Thus for

C0 = 2∥∇b∥∞M

the function
z −C0u
achieves its maximum at some point (T, x0).

• Case 2: Assume that s0 = T. Then, for every (t, x) ∈[0, T] × Ωwe have

z(t, x) ≤w(T, x0) + Cu(t, x) ≤1

2|∇v|2 + C∥v∥∞+ C∥u+∥∞

and by maximum principle we also have that

∥u∥∞≤∥v∥∞.

Thus, up to a universal constant

|∇u(t, x)|2 ≲(1 + ∥∇b∥M)∥v∥C1,

which proves estimate (58).

Now we prove estimate (59). The proof follows the previous strategy only this time we consider the
function
w(t, x) = (T −t)z −Cu,
where again C > 0 will be determined later. Assume that the maximum occurs at some point (s0, x0).

1. Case 1: Assume that s0 < T. Looking again the corresponding optimality conditions and
using the equations for u and z we have at (s0, x0)
Cα(s0)

2
−1

|∇u|2 ≤−(T −s0)⟨∇u · ∇b, ∇u⟩≤T∥∇b∥∞|∇u|2,

which is a contradiction if C ≥2TM∥∇b∥∞+ 2M.

2. Case 2: Assume that s0 = T, then for all (t, x) ∈[0, T] × Ωwe have

z(t, x) ≤w(T, x0) + C0u(t, x) = −C0v(T, x) + C0u(t, x) ≤2C0∥v∥∞.

Thus up to a dimensional constant we have

(T −t)|∇u|2(t, x) ≲M(T∥∇b∥∞+ 1)∥v∥∞.

25


---Page Break---
Corollary D.2. Let Ω⊂Rd, v : Rd →R and b : [0, T] × Ω→R be given with ∥∇b∥∞< ∞.
Given ψ : Ω→R, with ψ ≥1 consider the solution ϕ : [0, T] × Ω→R of
−∂tϕ −α(t)∆ϕ + b · ∇ϕ = 0 in [0, T) × Ω,
ϕ(T, x) = ψ(x) in Ω.
(61)

Assume moreover that for some M > 0

0 < 1

M ≤α(t).

Then up to a dimensional constant C > 0 we have the following estimates

∥∇ϕ(t)∥2
∞≤C∥ψ∥3
C1(1 + ∥∇b∥∞M)
(62)

(T −t)∥∇ϕ(t)∥2
∞≤C∥ψ∥3
∞M(T∥∇b∥∞+ 1)
(63)

Proof. Since ψ ≥1 by the Maximum Principle we have that

1 ≤ϕ(t, x) ≤∥ψ∥∞.

Thus we can define u : [0, T] × Ω→R by the formula

ϕ(t, x) = e−u(t,x)

2
⇐⇒u(t, x) = −2 log(ϕ(t, x)),

We then have

∂tϕ = −1

2∂tuϕ, ∇ϕ = −∇u

2 ϕ, ∆ϕ = |∇u|2

4
ϕ −∆u

2 ϕ,

and so by substituting in the equation for ϕ we obtain
(
−∂tu −α(t)∆u + α(t)

2 |∇u|2 + b · ∇u = 0,
u(T, x) = −2 log(ψ(x)).
(64)

The results will follow by Proposition D.1 for v(x) = −2 log(ψ(x)). First we need the following
estimates
|v(x)| = 2| log(ψ(x))| ≤2 log(∥ψ∥∞) ≤2∥ψ∥∞since ψ ≥1,

|∇v(x)| = 2|∇ψ(x)|

|ψ(x)|
≤2∥∇ψ∥∞

=⇒∥v∥C1 ≤2∥ψ∥C1.

Moreover, we note that

|∇u(t, x)|2 = 4|∇ϕ(t, x)|2

ϕ2
≥4|∇ϕ(t, x)|2

∥ψ∥2∞
.

Therefore, by Proposition D.1 estimate (58) we obtain that up to a dimensional constant C > 0

∥∇ϕ(t)∥2
∞≤C∥ψ∥2
∞(1 + ∥∇b∥∞M)∥ψ∥C1 ≤C∥ψ∥3
C1(1 + ∥∇b∥∞M).

Similarly by Proposition D.1, estimate (59) again up to a dimensional constant C > 0 we obtain

(T −t)∥∇ϕ(t)∥2
∞≤C∥ψ∥3
∞M(T∥∇b∥∞+ 1).

D.2
Long time behavior of periodic heat equation.

The following result is classical and establishes exponential rate of convergence in d1 for solutions
to the heat equation in the uniform distribution on the torus. We provide a proof for the readers
convenience.

26


---Page Break---
Proposition D.3. Let Ω= RTd and mi ∈P(Ω), i = 1, 2 be two probability measures. Define
ρi : [0, ∞) × Ω→[0, ∞) by
∂tρi −∆ρi = 0 in (0, ∞) × Ω,
ρi(0) = mi in Ω.

Then, there exists constants C = C(d) > 0, ω = ω(d) > 0 such that

d1(ρ2(t), ρ1(t)) ≤CRe−ωt

R2 d1(m2, m1) for t ≥0.

In particular by choosing m2 =
1
vol(RTd)

d1(ρ1(t),
1
vol(RTd)) ≤CRe−ωt

R2 d1(m1,
1
vol(RTd)) for t ≥0.
(65)

Proof. First we assume that R = 1 and recall that if Γ(t, x) denotes the heat kernel on the unit torus
Td then for a constant C = C(d) > 0 we have the normalizing effect

|∇Γ(t) ⋆g| ≤C ∥g∥∞
√

t .

Fix ψ : Td →R such that Lip(ψ) ≤1 and ψ(0) = 0. Such a function ψ will also satisfy

∥ψ∥∞≤1.

Fix T > 0 and let ϕ : Td × [0, T] →R be given by
−∂tϕ −∆ϕ = 0 in [0, T) × Td,
ϕ(T, x) = ψ(x).
(66)

By testing against ϕ in the equation satisfied by λ = ρ2 −ρ1 we obtain
Z
ψ(x)d(ρ2(T) −ρ1(T))(x) =
Z
ϕ(0)d(m2 −m1) ≤d1(m2, m1)∥∇ϕ(0)∥∞≤C d1(m2, m1)
√

T
.

Taking the supremum in ψ yields

d1(ρ2(T), ρ1(T)) ≤C
√

T
d1(ρ2(0), ρ1(0)).

Repeating the same argument as above with initial conditions mi = ρi(T), we obtain

d1(ρ2(2T), ρ1(2T)) ≤C
√

T
d1(ρ2(T), ρ1(T)) ≤
 C
√

T

2
d1(m2, m1)

and by induction for all k ∈N we have

d1(ρ2(kT), ρ1(kT)) ≤
 C
√

T

k
d1(m2, m1) = CkT −k

2 d1(m2, m1).

Let θ ≥0 and fix a k ∈N such that

kT ≤θ < (k + 1)T.

Since the heat operator defines a contraction in d1 we have that

d1(ρ2(θ), ρ1(θ)) ≤d1(ρ2(kT), ρ1(kT)) ≤(C
√

T)−kd1(m2, m1).

Moreover, we have that

−k ≤−θ

T + 1

thus
d1(ρ2(θ), ρ1(θ)) ≤(C
√

T)−θ

T +1d1(m2, m1)
and so choosing T large enough so that
C
√

T ≥e

27


---Page Break---
yields the result for R = 1. For a general R > 0, we simply apply the above to the functions
λi : [0, ∞) × Td defined by
λi(t, x) = Rdρi(R2t, Rx)
which yields
d1(λ2(t), λ1(t)) ≤Ce−ωtd1(λ2(0), λ1(0)).
Since

d1(λ2(t), λ1(t)) =
sup
ψ:Td→R,Lip(ψ)≤1

Z

Td ψ(x)Rd(ρ2(tR2, Rx) −ρ1(tR2, Rx))

= sup
ψ

Z

RTd ψ
 u

R


(ρ2(tR2, u) −ρ1(tR2, u))

= R
sup
ψ:RTd→R,Lip(ψ)≤1

Z

RTd ψ(u)(ρ2(tR2, u) −ρ1(tR2, u)) = Rd1(ρ2(R2t), ρ1(R2t))

the result follows.

28


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: All our claims, both formal and descriptive explanations of our theorems are
carefully written to appropriately reflect the analytical results we obtain. Each theoretical
claim is backed by a proof and accompanying supporting arguments in the appendix.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]

Justification: Limitations of the analysis are clear based on the assumptions in each theorem.
We have also discussed throughout where the analysis has not been optimized to exhaustion
relative to the power of the analysis tools we use.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The main three theorems in the paper are carefully formulated and stated.
The proofs are descriptively outlined in the main body, and the full details with supporting
propositions and lemmas are provided in the appendix along with their detailed proofs.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA]
Justification: Our paper does not include experiments.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: Our paper does not include experiments requiring code.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [NA]
Justification: Our paper does not include experiments.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: Our paper does not include experiments.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

29


---Page Break---
Answer: [NA]
Justification: Our paper does not include experiments.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: Yes. The paper is also written to preserve anonymity as much as possible.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [No]

Justification: While our paper involves analysis of generative artificial intelligence algorithm,
there are no immediate pathways a malicious actor could use our analysis to produce more
harmful disinformation or deepfakes. Our analysis primarily exist to understand existing
generative AI algorithms.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This paper is an analysis paper, and poses none of these acute risks.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [NA]
Justification: No individual or entity owns the theory of PDEs.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper does not release new assets.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: This paper does not involve crowdsourcing nor does it involve research with
human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]

30


---Page Break---
