Mirror and Preconditioned Gradient Descent in
Wasserstein Space

Clément Bonet
CREST, ENSAE, IP Paris
clement.bonet@ensae.fr

Théo Uscidda
CREST, ENSAE, IP Paris
theo.uscidda@ensae.fr

Adam David
Institute of Mathematics
Technische Universität Berlin
david@math.tu-berlin.de

Pierre-Cyril Aubin-Frankowski
TU Wien
pierre-cyril.aubin@tuwien.ac.at

Anna Korba
CREST, ENSAE, IP Paris
anna.korba@ensae.fr

Abstract

As the problem of minimizing functionals on the Wasserstein space encompasses
many applications in machine learning, different optimization algorithms on Rd
have received their counterpart analog on the Wasserstein space. We focus here
on lifting two explicit algorithms: mirror descent and preconditioned gradient
descent. These algorithms have been introduced to better capture the geometry of
the function to minimize and are provably convergent under appropriate (namely
relative) smoothness and convexity conditions. Adapting these notions to the
Wasserstein space, we prove guarantees of convergence of some Wasserstein-
gradient-based discrete-time schemes for new pairings of objective functionals
and regularizers. The difficulty here is to carefully select along which curves the
functionals should be smooth and convex. We illustrate the advantages of adapting
the geometry induced by the regularizer on ill-conditioned optimization tasks, and
showcase the improvement of choosing different discrepancies and geometries in a
computational biology task of aligning single-cells.

1
Introduction

Minimizing functionals on the space of probability distributions has become ubiquitous in machine
learning for e.g. sampling [13, 129], generative modeling [53, 86], learning neural networks [36, 90,
107], dataset transformation [4, 63], or modeling population dynamics [23, 120]. It is a challenging
task as it is an infinite-dimensional problem. Wasserstein gradient flows [5] provide an elegant way to
solve such problems on the Wasserstein space, i.e., the space of probability distributions with bounded
second moment, equipped with the Wasserstein-2 distance from optimal transport (OT). These flows
provide continuous paths of distributions decreasing the objective functional and can be seen as analog
to Euclidean gradient flows [111]. Their implicit time discretization, referred to as the JKO scheme
[66], has been studied in depth [1, 26, 95, 111]. In contrast, explicit schemes, despite being easier to
implement, have been less investigated. Most previous works focus on the optimization of a specific
objective functional with a time-discretation of its gradient flow with the Wasserstein-2 metrics. For
instance, the forward Euler discretization leads to the Wasserstein gradient descent. The latter takes
the form of gradient descent (GD) on the position of particles for functionals with a closed-form
over discrete measures, e.g. Maximum Mean Discrepancy (MMD), which can be of interest to
train neural networks [7, 30]. For objectives involving absolutely continuous measures, such as the
Kullback-Leibler (KL) divergence for sampling, other discretizations can be easily computed such
as the Unadjusted Langevin Algorithm (ULA) [106]. This leaves the question open of assessing

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the theoretical and empirical performance of other optimization algorithms relying on alternative
geometries and time-discretizations.

In the optimization community, a recent line of works has focused on extending the methods and
convergence theory beyond the Euclidean setting by using more general costs for the gradient descent
scheme [77]. For instance, mirror descent (MD), originally introduced by Nemirovskij and Yudin
[92] to solve constrained convex problems, uses a cost that is a divergence defined by a Bregman
potential [12]. Mirror descent benefits from convergence guarantees for objective functions that
are relatively smooth in the geometry induced by the (Bregman) divergence [88], even if they do
not have a Lipschitz gradient, i.e., are not smooth in the Euclidean sense. More recently, a closely
related scheme, namely preconditioned gradient descent, was introduced in [89]. It can be seen as a
dual version of the mirror descent algorithm, where the role of the objective function and Bregman
potential are exchanged. In particular, its convergence guarantees can be obtained under relative
smoothness and convexity of the Fenchel transform of the potential, with respect to the objective.
This algorithm appears more efficient to minimize the gradient magnitude than mirror descent [68].
The flexible choice of the Bregman divergence used by these two schemes enables to design or
discover geometries that are potentially more efficient.

Mirror descent has already attracted attention in the sampling community, and some popular algo-
rithms have been extended in this direction. For instance, ULA was adapted into the Mirror Langevin
algorithm [3, 32, 62, 64, 79, 121, 132]. Other sampling algorithms have received their counterpart
mirror versions such as the Metropolis Adjusted Langevin Algorithm [116], diffusion models [82],
Stein Variational Gradient Descent (SVGD) [114], or even Wasserstein gradient descent [113]. Pre-
conditioned Wasserstein gradient descent has been also recently proposed for specific geometries in
[31, 44] to minimize the KL in a more efficient way, but without an analysis in discrete time. All the
previous references focus on optimizing the KL as an objective, while Wasserstein gradient flows
have been studied in machine learning for different functionals such as more general f-divergences
[6, 93], interaction energies [19, 78], MMDs [7, 30, 59, 60, 72] or Sliced-Wasserstein (SW) distances
[15, 18, 45, 86]. In this work, we propose to bridge this gap by providing a general convergence
theory of both mirror and preconditioned gradient descent schemes for general target functionals, and
investigate as well empirical benefits of alternative transport geometries for optimizing functionals on
the Wasserstein space. We emphasize that the latter is different from [9, 67], wherein mirror descent
is defined in the Radon space of probability distributions, using the flat geometry defined by TV or
L2 norms on measures, see Appendix A for more details.

Contributions.
We are interested in minimizing a functional F : P2(Rd) →R ∪{+∞} over
probability distributions, through schemes of the form, for τ > 0, k ≥0,

Tk+1 = argmin
T∈L2(µk)
⟨∇W2F(µk), T −Id⟩L2(µk) + 1

τ d(T, Id),
µk+1 = (Tk+1)#µk,
(1)

with different costs d : L2(µk) × L2(µk) →R+, and in providing convergence conditions. While
we can recover a map ¯T = Tk ◦Tk−1 · · · ◦T1 such that µk = ¯T#µ0, the scheme (1) proceeds by
successive regularized linearizations retaining the Wasserstein structure, since the tangent space to
P2(Rd) at µ is a subset of L2(µ) [96]. This paper is organized as follows. In Section 2, we provide
some background on Bregman divergences, as well as on differentiability and convexity over the
Wasserstein space. In Section 3, we consider Bregman divergences on L2(µ) for the cost in (1),
generalizing the mirror descent scheme to the Wasserstein space. We study this new scheme by
discussing its implementation, and proving its convergence under relative smoothness and convexity
assumptions. In Section 4, we consider alternative costs in (1), that are analogous to OT distances with
translation-invariant cost, extending the dual space preconditioning scheme to the latter space. Finally,
in Section 5, we apply the two schemes to different objective functionals, including standard free
energy functionals such as interaction energies and KL divergence, but also to Sinkhorn divergences
[50] or SW [16, 102] with polynomial preconditioners on single-cell datasets.

Notations. Consider the set P2(Rd) of probability measures µ on Rd with finite second moment
and P2,ac(Rd) ⊂P2(Rd) its subset of absolutely continuous probability measures with respect to
the Lebesgue measure. For any µ ∈P2(Rd), we denote by L2(µ) the Hilbert space of functions
f : Rd →Rd such that
R
∥f∥2dµ < ∞equipped with the norm ∥·∥L2(µ) and inner product ⟨·, ·⟩L2(µ).
For a Hilbert space X, the Fenchel transform of f : X →R is f ∗(y) = supx∈X ⟨x, y⟩−f(x).
Given a measurable map T : Rd →Rd and µ ∈P2(Rd), T#µ is the pushforward measure of µ by

2


---Page Break---
T; and T ⋆µ =
R
T(· −x)dµ(x). For µ, ν ∈P2(Rd), the Wasserstein-2 distance is W2
2(µ, ν) =
infγ∈Π(µ,ν)
R
∥x −y∥2 dγ(x, y), where Π(µ, ν) = {γ ∈P(Rd × Rd), π1
#γ = µ, π2
#γ = ν} with
πi(x1, x2) = xi, is the set of couplings between µ and ν, and we denote by Πo(µ, ν) the set of
optimal couplings. When the optimal coupling is of the form γ = (Id, Tν
µ)#µ with Id : x 7→x
and Tν
µ ∈L2(µ) satisfying (Tν
µ)#µ = ν, we call Tν
µ the OT map. We refer to the metric space
(P2(Rd), W2) as the Wasserstein space. We note S++
d
(R) the space of symmetric positive definite
matrices, and for x ∈Rd, Σ ∈S++
d
(R), ∥x∥2
Σ = xT Σx.

2
Background

In this section, we fix µ ∈P2(Rd) and introduce first the Bregman divergence on L2(µ) along with
the notions of relative convexity and smoothness that will be crucial in the analysis of the optimization
schemes. Then, we introduce the differential structure and computation rules for differentiating a
functional F : P2(Rd) →R along curves and discuss notions of convexity on P2(Rd). We refer
the reader to Appendix B and Appendix C for more details on L2(µ) and the Wasserstein space
respectively. Finally, we introduce the mirror descent and preconditioned gradient descent on Rd.

Bregman divergence on L2(µ).
Frigyik et al. [54, Definition 2.1] defined the Bregman divergence
of Fréchet differentiable functionals. In our case, we only need Gâteaux differentiability. In this paper,
∇refers to the Gâteaux differential, which coincides with the Fréchet derivative if the latter exists.

Definition 1. Let ϕµ : L2(µ) →R be convex and continuously Gâteaux differentiable. The Bregman
divergence is defined for all T, S ∈L2(µ) as dϕµ(T, S) = ϕµ(T)−ϕµ(S)−⟨∇ϕµ(S), T−S⟩L2(µ).

We use the same definition on Rd. The map ϕµ (respectively ∇ϕµ) in the definition of dϕµ above is
referred to as the Bregman potential (respectively mirror map). If ϕµ is strictly convex, then dϕµ is a
valid Bregman divergence, i.e. it is positive and separates maps µ-almost everywhere (a.e.). In partic-
ular, for ϕµ(T) = 1

2∥T∥2
L2(µ), we recover the L2 norm as a divergence dϕµ(T, S) = 1

2∥T −S∥2
L2(µ).
Bregman divergences have received a lot of attention as they allow to define provably convergent
schemes for functions which are not smooth in the standard (e.g. Euclidean) sense [11, 88], and
thus for which gradient descent is not appropriate. These guarantees rely on the notion of relative
smoothness and relative convexity [88, 89], which we introduce now on L2(µ).

Definition 2 (Relative smoothness and convexity). Let ψµ, ϕµ : L2(µ) →R be convex and continu-
ously Gâteaux differentiable. We say that ψµ is β-smooth (respectively α-convex) relative to ϕµ if
and only if for all T, S ∈L2(µ), dψµ(T, S) ≤βdϕµ(T, S) (respectively dψµ(T, S) ≥αdϕµ(T, S)).

Similarly to the Euclidean case [88], relative smoothness and convexity are equivalent to respectively
βϕµ −ψµ and ψµ −αϕµ being convex (see Appendix B.2). Yet, proving the convergence of (1)
requires only that these properties hold at specific functions (directions), a fact we will soon exploit.

In some situations, we need the L2 Fenchel transform ϕ∗
µ of ϕµ to be differentiable, e.g. to
compute its Bregman divergence dϕ∗µ.
We show in Lemma 18 that a sufficient condition to
satisfy this property is for ϕµ to be strictly convex, lower semicontinuous and superlinear, i.e.
lim∥T∥→∞ϕµ(T)/∥T∥L2(µ) = +∞. Moreover, in this case, (∇ϕµ)−1 = ∇ϕ∗
µ. When needed, we
will suppose that ϕµ satisfies this assumption.

Differentiability on (P2(Rd), W2).
Let F : P2(Rd) →R ∪{+∞}, and denote D(F) = {µ ∈
P2(Rd), F(µ) < +∞} the domain of F and D( ˜Fµ) = {T ∈L2(µ), T#µ ∈D(F)} the domain of
˜Fµ defined as ˜Fµ(T) := F(T#µ) for all T ∈L2(µ). In the following, we use the differential struc-
ture of (P2(Rd), W2) introduced in [17, Definition 2.8], and we say that ∇W2F(µ) is a Wasserstein
gradient of F at µ ∈D(F) if for any ν ∈P2(Rd) and any optimal coupling γ ∈Πo(µ, ν),

F(ν) = F(µ) +
Z
⟨∇W2F(µ)(x), y −x⟩dγ(x, y) + o
 
W2(µ, ν)

.
(2)

If such a gradient exists, then we say that F is Wasserstein differentiable at µ [17, 74]. Moreover
there is a unique gradient belonging to the tangent space of P2(Rd) verifying (2) [74, Proposition

3


---Page Break---
2.5], and we will always restrict ourselves without loss of generality to this particular gradient, see
Appendix C.1. The differentiability of F and ˜Fµ are very related, as described in the following
proposition.
Proposition 1. Let F : P2(Rd) →R ∪{+∞} be a Wasserstein differentiable functional on D(F).
Let µ ∈P2(Rd) and ˜Fµ(T) = F(T#µ) for all T ∈D( ˜Fµ). Then, ˜Fµ is Fréchet differentiable, and
for all S ∈D( ˜Fµ), ∇˜Fµ(S) = ∇W2F(S#µ) ◦S.

The Wasserstein differentiable functionals include c-Wasserstein costs on P2,ac(Rd) [74, Propo-
sition 2.10 and 2.11], potential energies V(µ) =
R
V dµ or interaction energies W(µ) =
1
2
RR
W(x −y) dµ(x)dµ(y) for V : Rd →R and W : Rd →R differentiable and with bounded
Hessian [74, Section 2.4]. In particular, their Wasserstein gradients read as ∇W2V(µ) = ∇V
and ∇W2W(µ) = ∇W ⋆µ. However, entropy functionals, e.g. the negative entropy defined as
H(µ) =
R
log
 
ρ(x)

dµ(x) for distributions µ admitting a density ρ w.r.t. the Lebesgue measure, are
not Wasserstein differentiable. In this case, we can consider subgradients ∇W2F(µ) at µ for which
(2) becomes an inequality. To guarantee that the Wasserstein subgradient is not empty, we need ρ to
satisfy some Sobolev regularity, see e.g. [5, Theorem 10.4.13] or [108]. Then, if ∇log ρ ∈L2(µ),
the only subgradient of H in the tangent space is ∇W2H(µ) = ∇log ρ, see [5, Theorem 10.4.17] and
[47, Proposition 4.3]. Then, free energies are functionals that write as sums of potential, interaction
and entropy terms [110, Chapter 7]. It is notably the case for the KL to a fixed target distribution, that
is the sum of a potential and entropy term [129], or the MMD as a sum of a potential and interaction
term [7].

Examples of functionals.
The definitions of Bregman divergences on L2(µ) and of Wasserstein
differentiability enable us to consider alternative Bregman potentials than the L2(µ)-norm mentioned
above. For instance, for V convex, differentiable and L-smooth, we can use potential energies
ϕV
µ (T) := V(T#µ), for which dϕV
µ (T, S) =
R
dV
 
T(x), S(x)

dµ(x) where dV is the Bregman
divergence of V on Rd. Notice that ϕµ(T) = 1

2∥T∥2
L2(µ) is a specific example of a potential energy
where V = 1

2∥· ∥2. Moreover, we will consider interaction energies ϕW
µ (T) := W(T#µ) with
W convex, differentiable, L-smooth, and satisfying W(−x) = W(x); for which dϕW
µ (T, S) =

1
2
RR
dW
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′) (see Appendix I.3). We will also use ϕH
µ (T) =
H(T#µ) with H the negative entropy. Note that Bregman divergences on the Wasserstein space
using these functionals were proposed by Li [80], but only for S = Id and optimal transport maps T.

Convexity and smoothness in (P2(Rd), W2).
In order to study the convergence of gradient
flows and their discrete-time counterparts, it is important to have suitable notions of convexity and
smoothness. On (P2(Rd), W2), different such notions have been proposed based on specific choices
of curves. The most popular one is to require the functional F to be α-convex along geodesics
(see Definition 10), which are of the form µt =
 
(1 −t)Id + tTµ1
µ0


#µ0 if µ0 ∈P2,ac(Rd) and
µ1 ∈P2(Rd), with Tµ1
µ0 the OT map between them. In that setting,
α

2 W2
2(µ0, µ1) = α

2 ∥Tµ1
µ0 −Id∥2
L2(µ0) ≤F(µ1) −F(µ0) −⟨∇W2F(µ0), Tµ1
µ0 −Id⟩L2(µ0).
(3)

For instance, free energies such as potential or interaction energies with convex V or W, or the
negative entropy, are convex along geodesics [110, Section 7.3]. However, some popular functionals,
such as the Wasserstein-2 distance µ 7→1

2W2
2(µ, η) itself, for a given η ∈P2(Rd), are not convex
along geodesics. Instead Ambrosio et al. [5, Theorem 4.0.4] showed that it was sufficient for the
convergence of the gradient flow to be convex along other curves, e.g. along particular generalized
geodesics for the Wasserstein-2 distance [5, Lemma 9.2.7], which, for µ, ν ∈P2(Rd), are of the
form µt =
 
(1 −t)Tµ
η + tTν
η


#η for Tµ
η, T ν
η OT maps from η to µ and ν. Observing that for
ϕµ(T) = 1

2∥T∥2
L2(µ), we can rewrite (3) as αdϕµ0(Tµ1
µ0, Id) ≤d ˜
Fµ0(Tµ1
µ0, Id) and see that being
convex along geodesics boils down to being convex in the L2 sense for S = Id and T chosen as an
OT map. This observation motivates us to consider a more refined notion of convexity along curves.

Definition 3. Let µ ∈P2(Rd), T, S ∈L2(µ) and for all t ∈[0, 1], µt = (Tt)#µ with Tt = (1−t)S+
tT. We say that F : P2(Rd) →R is α-convex (resp. β-smooth) relative to G : P2(Rd) →R along
t 7→µt if for all s, t ∈[0, 1], d ˜
Fµ(Ts, Tt) ≥αd ˜Gµ(Ts, Tt) (resp. d ˜
Fµ(Ts, Tt) ≤βd ˜Gµ(Ts, Tt)).

4


---Page Break---
Notice that in contrast with Definition 2, Definition 3 is stated for a fixed distribution µ and direc-
tions (S, T), and involves comparisons between Bregman divergences depending on µ and curves
(Ts)s∈[0,1] depending on S, T. The larger family of S and T for which Definition 3 holds, the more re-
stricted is the notion of convexity of F −αG (resp. of βG−F) on P2(Rd). For instance, Wasserstein-2
generalized geodesics with anchor η ∈P2(Rd) correspond to considering S, T as all the OT maps
originating from η, among which geodesics are particular cases when taking η = µ (hence S = Id).
If we furthermore ask for α-convexity to hold for all µ ∈P2(Rd) and T, S ∈L2(µ) (i.e., not only OT
maps), then we recover the convexity along acceleration free-curves as introduced in [28, 98, 117].
Our motivation behind introducing Definition 3 is that the convergence proofs of MD and precon-
ditioned GD require relative smoothness and convexity properties to hold only along specific curves.

Mirror descent and preconditioned gradient descent on Rd.
These schemes read respectively as
∇ϕ(xk+1) −∇ϕ(xk) = −τ∇f(xk) [12] and yk+1 −yk = −τ∇h∗ 
∇g(yk)

[89], where the objec-
tives f, g and the regularizers h, ϕ are convex C1 functions from Rd to R. The algorithms are closely
related since, using the Fenchel transform and setting g = ϕ∗and h∗= f, we see that, for y = ∇ϕ(x),
the two schemes are equivalent when permuting the roles of the objective and of the regularizer. For
MD, convergence of f is ensured if f is both 1/τ-smooth and α-convex relative to ϕ [88, Theorem
3.1]. Concerning preconditioned GD, assuming that h, g are Legendre,
 
g(yk)


k converges to the
minimum of g if h∗is both 1/τ-smooth and α-convex relative to g∗with α > 0 [89, Theorem 3.9].

3
Mirror descent

For every µ ∈P2(Rd), let ϕµ : L2(µ) →R be strictly convex, proper and differentiable and assume
that the (sub)gradient ∇W2F(µ) ∈L2(µ) exists. In this section, we are interested in analyzing the
scheme (1) where the cost d is chosen as a Bregman divergence, i.e. dϕµ as defined in Definition 1.
This corresponds to a mirror descent scheme in P2(Rd). For τ > 0 and k ≥0, it writes:

Tk+1 = argmin
T∈L2(µk)
dϕµk (T, Id) + τ⟨∇W2F(µk), T −Id⟩L2(µk),
µk+1 = (Tk+1)#µk.
(4)

Iterates of mirror descent.
In all that follows, we assume that the iterates (4) exist, which is true
e.g. for a superlinear ϕµk, since the objective is a sum of linear functions and of the continuous
ϕµk. In the previous section, we have seen that the second term in the proximal scheme (4) can be
interpreted as a linearization of the functional F at µk for Wasserstein (sub)differentiable functionals.
Now define for all T ∈L2(µk), J(T) = dϕµk (T, Id)+τ⟨∇W2F(µk), T−Id⟩L2(µk). Then, deriving
the first order conditions of (4) as ∇J(Tk+1) = 0, we obtain µk-a.e.,

∇ϕµk(Tk+1) = ∇ϕµk(Id) −τ∇W2F(µk) ⇐⇒Tk+1 = ∇ϕ∗
µk
 
∇ϕµk(Id) −τ∇W2F(µk)

. (5)

Note that for ϕµ(T) = 1

2∥T∥2
L2(µ), the update (5) translates as Tk+1 = Id −τ∇W2F(µk), and
our scheme recovers Wasserstein gradient descent [35, 91]. This is analogous to mirror descent
recovering GD when the Bregman potential is chosen as the Euclidean squared norm in Rd [12]. We
discuss in Appendix D.2 the continuous formulation of (4), showing it coincides with the gradient
flow of the mirror Langevin [3, 130], the limit of the JKO scheme with Bregman groundcosts [104],
Information Newton’s flows [126], or Sinkhorn’s flow [41] for specific choices of ϕ and F.

Our proof of convergence of the mirror descent algorithm will require the Bregman divergence to
satisfy the following property, which is reminiscent of conditions of optimality for couplings in OT.
Assumption 1. For µ, ρ ∈P2,ac(Rd) and ν ∈P2(Rd), setting Tµ,ν
ϕµ = argminT#µ=ν dϕµ(T, Id),
Uρ,ν
ϕρ = argminU#ρ=ν dϕρ(U, Id), the functional ϕµ is such that, for any S ∈L2(µ) satisfying
S#µ = ρ, we have dϕµ(Tµ,ν
ϕµ , S) ≥dϕρ(Uρ,ν
ϕρ , Id).

The inequality in Assumption 1 can be interpreted as follows: the “distance” between ρ and ν is
greater when observed from an anchor µ that differs from ρ and ν. We demonstrate that Bregman
divergences satisfy this assumption under the following conditions on the Bregman potential ϕ.
Proposition 2. Let µ, ρ ∈P2,ac(Rd) and ν ∈P2(Rd). Let ϕµ be a pushforward compatible
functional, i.e. there exists ϕ : P2(Rd) →R such that for all T ∈L2(µ), ϕµ(T) = ϕ(T#µ).
Assume furthermore ∇W2ϕ(µ) and ∇W2ϕ(ρ) invertible (on Rd). Then, ϕµ satisfies Assumption 1.

5


---Page Break---
All the maps ϕV
µ , ϕW
µ and ϕH
µ defined in Section 2 satisfy the assumptions of Proposition 2 under
mild requirements, see Appendix D.1. The proof of Proposition 2 is given in Appendix H.2. It relies
on the definition of an appropriate optimal transport problem

Wϕ(ν, µ) =
inf
γ∈Π(ν,µ) ϕ(ν) −ϕ(µ) −
Z
⟨∇W2ϕ(µ)(y), x −y⟩dγ(x, y),
(6)

and on the proof of existence of OT maps for absolutely continuous measures (see Proposition 15),
which implies Wϕ(ν, µ) = dϕµ(Tµ,ν
ϕµ , Id) with Tµ,ν
ϕµ defined as in Assumption 1. From there, we can
conclude that ϕµ satisfies Assumption 1. We notice that the corresponding transport problem recovers
previously considered objects such as OT problems with Bregman divergence costs [25, 103], but is
strictly more general (as our results pertain to the existence of OT maps), as detailed in Appendix D.1.

We now analyze the convergence of the MD scheme. Under a relative smoothness condition along
curves generated by S = Id and T = Tk+1 solutions of (4) for all k ≥0, we derive the following de-
scent lemma, which ensures that
 
F(µk)


k is non-increasing. Its proof can be found in Appendix H.3
and relies on the three-point inequality [29], which we extended to L2(µ) in Lemma 29.

Proposition 3. Let β > 0, τ ≤
1
β . Assume for all k ≥0, F is β-smooth relative to ϕ along
t 7→
 
(1 −t)Id + tTk+1


#µk, which implies βdϕµk (Tk+1, Id) ≥d ˜
Fµk (Tk+1, Id). Then, for all
k ≥0,

F(µk+1) ≤F(µk) −1

τ dϕµk (Id, Tk+1).
(7)

Assuming additionally the convexity of F along the curves µt =
 
(1 −t)Id + tTµ,ν
ϕµ )#µ, t ∈[0, 1]
and that ϕ satisfies Assumption 1, we can obtain global convergence.

Proposition 4. Let ν ∈P2(Rd), α ≥0. Suppose Assumption 1 and the conditions of Proposition 3
hold, and that F is α-convex relative to ϕ along the curves t 7→
 
(1 −t)Id + tTµk,ν
ϕµk )#µk. Then, for
all k ≥1,

F(µk) −F(ν) ≤
α

(1 −τα)−k −1
Wϕ(ν, µ0) ≤1 −ατ

kτ
Wϕ(ν, µ0).
(8)

Moreover, if α > 0, taking ν = µ∗the minimizer of F, we obtain a linear rate: for all k ≥0,
Wϕ(µ∗, µk) ≤(1 −τα)k Wϕ(µ∗, µ0).

The proof of Proposition 4 can be found in Appendix H.4, and requires Assumption 1 to hold so that
consecutive distances between iterates and the global minimizer telescope. This is not as direct as
in the proofs of [88] over Rd, because the minimization problem of each iteration (4) happens in a
different space L2(µk). We discuss in Section 5 how to verify the relative smoothness and convexity
on some examples. In particular, when both F and ϕ are potential energies, it is inherited from the
relative smoothness and convexity on Rd, and the conditions are similar with those for MD on Rd.
We also note that relative smoothness assumptions along descent directions as stated in Proposition 3
and relative strong convexity along optimal curves between the iterates and a minimizer as stated in
Proposition 4 have been used already in the literature of optimization over measures in very specific
cases, e.g. for descent results for the KL along SVGD [71] or for Sinkhorn convergence in [9]. We
further analyze in Appendix F the convergence of Bregman proximal gradient scheme [11, 123] for
objectives of the form F(µ) = G(µ) + H(µ) with H non smooth; which includes the KL divergence
decomposed as a potential energy plus the negative entropy.

Implementation.
We now discuss the practical implementation of MD on (P2(Rd), W2) as written
in (5). If ϕµ is pushforward compatible, we have ∇ϕµk(Tk+1) = ∇W2ϕ
 
(Tk+1)#µk

◦Tk+1; but
if ∇ϕ∗
µk is unknown, the scheme is implicit in Tk+1. A possible solution is to rely on a root finding
algorithm such as Newton’s method to find the zero of ∇J at each step, which we use in Section 5 for
ϕW
µ as Bregman potential. However, this procedure may be computationally costly and scale badly
w.r.t. the dimension and the number of samples, see Appendix G.1. Nonetheless, in the special case
ϕV
µ (T) =
R
V ◦T dµ with V differentiable, strongly convex and L-smooth, since ∇W2V(µ) = ∇V
and (∇V )−1 = ∇V ∗, the scheme reads as

∀k ≥0, Tk+1 = ∇V ∗◦
 
∇V −τ∇W2F(µk)

,
(9)

6


---Page Break---
and can be implemented on particles, i.e. for ˆµk =
1
n
Pn
i=1 δxk
i , xk+1
i
= ∇V ∗ 
∇V (xk
i ) −
τ∇W2F(ˆµk)(xk
i )

for all k ≥0, i ∈{1, . . . , n}. This scheme is analogous to MD in Rd [12]
and has been introduced as the mirror Wasserstein gradient descent [113]. Moreover, for V = 1

2∥· ∥2
2,
as observed earlier, we recover the usual Wasserstein gradient descent, i.e. Tk+1 = Id−τ∇W2F(µk).
The scheme can also be implemented for Bregman potentials that are not pushforward compatible. For
specific ϕ, it recovers notably SVGD and its variants [83, 84, 114, 131] or the Kalman-Wasserstein
gradient descent [56]. We refer to Appendix D.4 for more details.

4
Preconditioned gradient descent

As seen in Section 2, preconditioned gradient descent on Rd has dual convergence conditions
compared to mirror descent. Our goal is to extend these to (1) and P2(Rd). Let τ > 0, µ ∈P2(Rd)
and h : Rd →R proper and strictly convex on Rd. We consider in this section ϕh
µ(T) =
R
h ◦T dµ
and d(T, Id) = ϕh
µk
 
(Id −T)/τ

τ =
R
h
 
(x −T(x))/τ

τ dµk(x). This type of discrepancy is
analogous to OT costs with translation-invariant ground cost c(x, y) = h(x −y), which have been
popular as they induce an OT map [110, Box 1.12]. Such costs have been introduced e.g. in [39, 70]
to promote sparse transport maps. More generally, for ϕµ strictly convex, proper, differentiable and
superlinear, we have (∇ϕµ)−1 = ∇ϕ∗
µ and the following theory is still valid. For simplicity, we
leave studying more general ϕ for future works. Here, the scheme (1) results in:

Tk+1 = argmin
T∈L2(µk)

Z
h
x −T(x)

τ


τ dµk(x)+⟨∇W2F(µk), T−Id⟩L2(µk), µk+1 = (Tk+1)#µk.

(10)
Deriving the first order conditions similarly to (5) in Section 3, we obtain the following update:

∀k ≥0, Tk+1 = Id −τ(∇ϕh
µk)−1 
∇W2F(µk)

= Id −τ∇h∗◦∇W2F(µk).
(11)

Notice that for h = 1

2∥· ∥2
2 the squared Euclidean norm, ϕh
µ and ϕh∗
µ recover the squared L2(µ) norm,
and schemes (4) and (10) coincide. The scheme (10) is analogous to preconditioned gradient descent
[68, 75, 76, 89, 119], which provides a dual alternative to mirror descent. For the latter, the goal
is to find a suitable preconditioner h∗allowing to have convergence guarantees, or to speed-up the
convergence for ill-conditioned problems. It was recently considered on the Wasserstein space by
Cheng et al. [31] and Dong et al. [44] with a focus on the KL divergence as objective F and for
h = ∥· ∥p
p with p > 1 [31] or h quadratic [44]. Moreover, their theoretical analysis [1] was mostly
done using the continuous formulation. Instead we focus on deriving conditions for the convergence
of the discrete-time scheme (11) for more general functionals objectives.

Convergence
guarantees.
Inspired
by
[89],
we
now
provide
a
descent
lemma
on
 
ϕh∗
µk(∇W2F(µk))


k under a technical inequality between the Bregman divergences of ϕh∗
µk and
˜Fµk for all k ≥0. Additionally, we also suppose that F is convex along the curves generated
by S = Tk+1 and T = Id. This last hypothesis ensures that d ˜
Fµk (Tk+1, Id) ≥0, and thus that
 
ϕh∗
µk(∇W2F(µk))


k is non-increasing. Analogously to the Euclidean case, ϕh∗
µ quantifies the mag-
nitude of the gradient, and provides a second quantifier of convergence leading to possibly different
efficient methods compared to mirror descent [68]. The proof relies mainly on the three-point identity
(see e.g. [54, Appendix B.7] or Lemma 28) and algebra with the definition of Bregman divergences.
Proposition 5. Let β > 0. Assume τ ≤1

β , and for all k ≥0, F convex along t 7→
 
(1 −t)Tk+1 +
tId


#µk and dϕh∗
µk
 
∇W2F(µk+1) ◦Tk+1, ∇W2F(µk)

≤βd ˜
Fµk (Id, Tk+1). Then, for all k ≥0,

ϕh∗
µk+1
 
∇W2F(µk+1)

≤ϕh∗
µk
 
∇W2F(µk)

−1

τ d ˜
Fµk (Tk+1, Id).
(12)

Under an additional assumption of a reverse inequality between the Bregman divergences of ϕh∗
µk and
˜Fµk, and assuming that ϕh∗
µ attains its minimum in 0, we can show the convergence of the gradient
quantified by ϕh∗
µ (see Lemma 21), and the convergence of
 
F(µk)


k towards the minimum of F.

Proposition 6. Let α ≥0 and µ∗∈P2(Rd) be the minimizer of F. Assume the conditions
of Proposition 5 hold, and that for ¯T = argminT,T#µk=µ∗d ˜
Fµk (Id, T),
αd ˜
Fµk (Id, ¯T) ≤

7


---Page Break---
dϕh∗
µk
 
∇W2F(¯T#µk) ◦¯T, ∇W2F(µk)

.
Then, for all k ≥1, since ∇W2F(µ∗) = 0 and

ϕh∗
µk(0) = h∗(0),

ϕh∗
µk
 
∇W2F(µk)

−h∗(0) ≤
α

(1 −τα)−k −1

 
F(µ0) −F(µ∗)

≤1 −τα

τk
 
F(µ0) −F(µ∗)

.

(13)
Moreover, assuming that h∗attains its minimum at 0 and α > 0, F converges towards its minimum
at a linear rate, i.e. for all k ≥0, F(µk) −F(µ∗) ≤(1 −τα)k  
F(µ0) −F(µ∗)

.

The proofs of Proposition 5 and Proposition 6 can be found respectively in Appendix H.5 and
Appendix H.6.

We now discuss sufficient conditions to obtain the inequalities between the Bregman divergences
required in Proposition 5 and Proposition 6. Maddison et al. [89] showed on Rd for a cost h and an
objective function g, that these conditions were equivalent to β-smoothness and α-convexity of the
preconditioner h∗(analogous to ϕ∗
µ) relative to the convex conjugate of the objective g∗(analogous
to ˜F∗
µ). To write the inequalities we assumed as a relative smoothness/convexity property of ϕh∗
µk
w.r.t. ˜F∗
µk, we would need at least to ensure that ˜F∗
µk is differentiable, in order to define its Bregman
divergence according to Definition 1. This can be done e.g. by assuming ˜Fµk strictly convex and
superlinear (see Lemma 18). The latter is true for several examples of functionals F we already
mentioned, such as potential or interaction energies with strongly convex potentials.

Under this assumption, we can show that the inequalities in Proposition 5 and Proposition 6 are
implied by relative smoothness and convexity along suitable curves.

Proposition 7. Let µ ∈P2(Rd) and T ∈L2(µ). Assume ˜F∗
µ is Gâteaux differentiable and define F∗
µ
on t 7→µt = (Tt)#µ as F∗
µ(µt) = ˜F∗
µ(Tt) for Tt = (1 −t)U + tS for all t ∈[0, 1], S, U ∈L2(µ).

If ϕh∗is β-smooth relative to F∗
µ along t 7→
 
(1 −t)∇W2F(µ) + t∇W2F(T#µ) ◦T


#µ. Then,

dϕh∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µ)

≤βd ˜
Fµ(Id, T).
(14)

Likewise, if ϕh∗is α-convex relative to F∗
µ along t 7→
 
(1 −t)∇W2F(µ) + t∇W2F(T#µ) ◦T


#µ,
then
αd ˜
Fµ(Id, T) ≤dϕh∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µ)

.
(15)

In particular, for F a potential energy, the conditions coincide with those of [89] in Rd. We refer to
Appendix E.1 for more details.

5
Applications and Experiments

In this section, we first discuss how to verify the relative convexity and smoothness between func-
tionals in practice. Then, we provide some examples of mirror descent and preconditioned gradient
descent on different objectives. We refer to Appendix G for more details on the experiments1.

Relative convexity of functionals.
To assess relative convexity or smoothness as stated in Defini-
tion 3, we need to compare the Bregman divergences along the right curves. When both functionals ϕ
and F are of the same type, for example potential (respectively interaction) energies, this property is
lifted from the convexity and smoothness on Rd of the underlying potential functions (respectively
interaction kernels) to P2(Rd), see Appendix E.2 for more details. When both ϕ and F are potential
energies, the schemes (4) and (10) are equivalent to parallel MD and preconditioned GD since there
are no interactions between the particles. The conditions of convergence then coincide with the ones
obtained for MD and preconditioned GD on Rd [88, 89]. In other cases, (4) and (10) provide schemes
that are novel to the best of our knowledge.

For functionals which are not of the same type, it is less straightforward. Using equivalent notions
of convexity (see Proposition 13), we may instead compare their Hessians along the right curves,

1The code is available at https://github.com/clbonet/Mirror_and_Preconditioned_Gradient_
Descent_in_Wasserstein_Space.

8


---Page Break---
Time (             )
⌧· iter

c
c
W(µt)
K2

K2
K⌃

2
K4
K⌃

4

t
K⌃

2

Figure 1: (Left) Value of W along the flow for two
difference interaction Bregman potentials, (Middle
and Right) Trajectories of particles to minimize W.

Time (             )
⌧· iter

NEM (ours) 
PFB (ours) 
FB

KL(µt||µ?)

Figure 2: Convergence towards Gaussians
N(0, UDU T ) averaged over 20 covariances,
with U ∼Unif(O10(R)) and D fixed.

see Appendix E.2 for an example between an interaction and a potential energy. We note also that
for the particular case of a functional obtained as a sum F = G + H with ˜Gµ and ˜Hµ convex, since
d ˜
Fµ = d ˜Gµ + d ˜
Hµ, d ˜
Fµ ≥max{d ˜Gµ, d ˜
Hµ}, and thus F is 1-convex relative to G and H. This
includes e.g. the KL divergence which is convex relative to the potential and the negative entropy.

MD on interaction energies.
We first focus on minimizing interaction energies W(µ) =
1
2
RR
W(x −y) dµ(x)dµ(y) with kernel W(z) =
1
4∥z∥4
Σ−1 −1

2∥z∥2
Σ−1, Σ ∈S++
d
(R), whose
minimizer is an ellipsoid [27]. Since the Hessian norm of W can be bounded by a polynomial
of degree 2, following [88, Section 2], W is β-smooth relative to K4(z) = 1

4∥z∥4
2 + 1

2∥z∥2
2 with
β = 4, and W is β-smooth relative to ϕµ(T) = 1

2
RR
K4
 
T(x) −T(y)

dµ(x)dµ(y). Supposing
additionally that the distributions are compactly supported, we can show that W is smooth relative to
the interaction energy with K2(z) = 1

2∥z∥2
2. For ill-conditioned Σ, i.e. for which the ratio between
the largest and smallest eigenvalues is large, the convergence can be slow. Thus, we also propose
to use KΣ
2 (z) = 1

2∥z∥2
Σ−1 and KΣ
4 (z) = 1

4∥z∥4
Σ−1 + 1

2∥z∥2
Σ−1. We illustrate these mirror descent
schemes on Figure 1 and observe the convergence we expect for the ones taking into account Σ.
In practice, since ∇ϕµ(T) = (∇K ⋆T#µ) ◦T, the scheme (5) needs to be approximated using
Newton’s algorithm which can be computationally heavy. Using ϕV
µ (T) =
R
V ◦T dµ with V = KΣ
2 ,
we obtain a more computationally friendly scheme with the same convergence, see Appendix G.2,
but for which the smoothness is trickier to show.

MD on KL.
We now focus on minimizing F(µ) =
R
V dµ + H(µ) for V (x) = 1

2xT Σ−1x with Σ
possibly ill-conditioned, whose minimizer is the Gaussian ν = N(0, Σ), and for which Wasserstein
gradient descent is slow to converge. We study the MD scheme in (4) with negative entropy H as the
Bregman potential (NEM), and compare it on Figure 2 with the Forward-Backward (FB) scheme
studied in [43] and the ideally preconditioned Forward-Backward scheme (PFB) with Bregman
potential ϕV
µ (see (116) in Appendix F). For computational purpose, we restrain the minimization in
(4) over affine maps, which can be seen as taking the gradient over the submanifold of Gaussians
[43, 73]. Starting from N(0, Σ0), the distributions stay Gaussian over the flow, and their closed-form
is reported in (62) (Appendix D.3). We note that this might not be the case for the scheme (4),
and thus that this scheme does not enter into the framework developed in the previous sections.
Nonetheless, it demonstrates the benefits of using different Bregman potentials. We generate 20
Gaussian targets ν on R10 with Σ = UDU T , D diagonal and scaled in log space between 1 and 100,
and U a uniformly sampled orthogonal matrix, and we report the averaged KL over time. Surprisingly,
NEM, which does not require an ideal (and not available in general) preconditioner, is almost as fast
to converge as the ideal PFB, and much faster than the FB scheme.

Preconditioned GD for single-cells.
Predicting the response of cells to a perturbation is a central
question in biology. In this context, as the measuring process is destructive, feature descriptions
of control and treated cells must be dealt with as (unpaired) source µ and target distributions ν.
Following [112], OT theory to recover a mapping T between these two populations has been used
in [21, 22, 23, 39, 48, 69, 122]. Inspired by the recent success of iterative refinement in generative
modeling, through diffusion [61, 115] or flow-based models [81, 85], our scheme (1) follows the idea
of transporting µ to ν via successive and dynamic displacements instead of, directly, with a static
map ¯T. We model the transition from unperturbed to perturbed states through the (preconditioned)

9


---Page Break---
#iters convergence 
F(ˆµ)
F(ˆµ)
F(ˆµ)

F(µ) = SW2

2(µ, ⌫)

#iters convergence 
#iters convergence 

with precond.

without precond.

with precond.

scRNAseq
4i

without precond.
without precond.
without precond.
without precond.
without precond.

F(µ) = S2

",2(µ, ⌫)
F(µ) = ED(µ, ⌫)

Figure 3: Preconditioned GD vs. (vanilla) GD to predict the responses of cell populations to cancer
treatment on 4i (Upper row) and scRNAseq (Lower row) datasets. For each treatment, starting from
the untreated cells µi, we minimize F(µ) = D(µ, νi) with νi the treated cells. The plot is organized
as pairs of columns, each corresponding to optimizing a specific metric, with two scatter plots
displaying points zi = (xi, yi) where (First column) yi is the attained minima F(ˆµ) = D(ˆµ, νi)
with preconditioning and xi that without preconditioning, and (Second column) yi is the number of
iterations to reach convergence with preconditioning and xi that without preconditioning. A point
below the diagonal y = x then refers to an experiment in which preconditioning provides (First
column) a better minima or (Second column) faster convergence. We assign a color to each treatment
and plot three runs, obtained with three different initializations, along with their mean (brighter point).

gradient flow of a functional F(µ) = D(µ, ν) initialized at µ0 = µ, where D is a distributional
metric, and predict the perturbed population via ˆµ = minµ F(µ). We focus on the datasets used
in [21], consisting of cell lines analyzed using (i) 4i [58], and (ii) scRNA sequencing [118]. For
each profiling technology, the response to respectively (i) 34 and (ii) 9 treatments are provided.
As in [21], training is performed in data space for the 4i data and in a latent space learned by the
scGen autoencoder [87] for the scRNA data. We use three metrics: the Sliced-Wasserstein distance
SW2
2 [16], the Sinkhorn divergence S2
ε,2 [50] and the energy distance ED [59, 60, 105], and we
compare the performances when minimizing this functional via preconditioned GD vs. (vanilla)
GD. We measure the convergence speed when using a fixed relative tolerance tol = 10−3, as well
as the attained optimal value F(ˆµ). Note that we follow [21] and additionally consider 40% of
unseen (test) target cells for evaluation, i.e., for computing F(ˆµ) = D(ˆµ, ν). As preconditioner, we
use the one induced by h∗(x) = (∥x∥a
2 + 1)1/a −1 with a > 0, which is well suited to minimize
functionals which grow in ∥x −x∗∥a/(a−1) near their minimum [119]. We set the step size τ = 1
for all the experiments. Then, we tune the parameter a very simply: for a given metric D and a
profiling technology, we pick a random treatment and select a ∈{1.25, 1.5, 1.75} by grid search,
and we generalize the selected a for all the other treatments. Results are described in Figure 3:
Preconditioned GD significantly outperforms GD over the 43 datasets, in terms of convergence speed
and optimal value F(ˆµ). For instance, for D = S2
2,ε, we converge in 10 times less iterations while
providing, on average, a better estimate of the treated population. We also compare our iterative (non
parametric) approach with the use of a static (non parametric) map in Appendix G.4.

6
Conclusion

In this work, we extended two non-Euclidean optimization methods on Rd to the Wasserstein space,
generalizing W2-gradient descent to alternative geometries. We investigated the practical benefits of
these schemes, and provided rates of convergences for pairs of objectives and Bregman potentials
satisfying assumptions of relative smoothness and convexity along specific curves. While these
assumptions can be easily checked is some cases (e.g. potential or interaction energies) by comparing
the Bregman divergences or Hessian operators in the Wasserstein geometry, they may be hard to verify
in general. Different objectives such as the Sliced-Wasserstein distance or the Sinkhorn divergence,
or alternative geometries to the Wasserstein-2 as studied in this work, require to derive specific
computations on a case-by-case basis. We leave this investigation for future work.

10


---Page Break---
Acknowledgments and Disclosure of Funding

Clément Bonet acknowledges the support of the center Hi! PARIS and of ANR PEPR PDE-AI.
Adam David gratefully acknowledges funding by the BMBF 01|S20053B project SALE. Pierre-Cyril
Aubin-Frankowski was funded by the FWF project P 36344-N. Anna Korba acknowledges the support
of ANR-22-CE23-0030.

References

[1] Martial Marie-Paul Agueh. Existence of Solutions to Degenerate Parabolic Equations via the
Monge-Kantorovich Theory. Georgia Institute of Technology, 2002. (Cited on p. 1, 7)

[2] Byeongkeun Ahn, Chiyoon Kim, Youngjoon Hong, and Hyunwoo J Kim. Invertible Monotone
Operators for Normalizing Flows. Advances in Neural Information Processing Systems, 35:
16836–16848, 2022. (Cited on p. 28)

[3] Kwangjun Ahn and Sinho Chewi. Efficient Constrained Sampling via the Mirror-Langevin
Algorithm. Advances in Neural Information Processing Systems, 34:28405–28418, 2021.
(Cited on p. 2, 5, 21, 29, 44)

[4] David Alvarez-Melis and Nicolò Fusi. Dataset Dynamics via Gradient Flows in Probability
Space. In International conference on machine learning, pages 219–230. PMLR, 2021. (Cited
on p. 1)

[5] Luigi Ambrosio, Nicola Gigli, and Giuseppe Savaré. Gradient Flows: in Metric Spaces and in
the Space of Probability Measures. Springer Science & Business Media, 2005. (Cited on p. 1,
4, 23, 25, 38, 53)

[6] Abdul Fatir Ansari, Ming Liang Ang, and Harold Soh. Refining Deep Generative Models via
Discriminator Gradient Flow. In 9th International Conference on Learning Representations,
ICLR, 2021. (Cited on p. 2, 20, 44)

[7] Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum Mean Discrepancy
Gradient Flow. Advances in Neural Information Processing Systems, 32, 2019. (Cited on p. 1,
2, 4, 24)

[8] Hedy Attouch, Giuseppe Buttazzo, and Gérard Michaille. Variational Analysis in Sobolev and
BV Spaces. Society for Industrial and Applied Mathematics, 2014. (Cited on p. 33, 35)

[9] Pierre-Cyril Aubin-Frankowski, Anna Korba, and Flavien Léger. Mirror Descent with Relative
Smoothness in Measure Spaces, with Application to Sinkhorn and EM. Advances in Neural
Information Processing Systems, 35:17263–17275, 2022. (Cited on p. 2, 6, 21)

[10] Heinz H Bauschke and Patrick L Combettes. Convex Analysis and Monotone Operator Theory
in Hilbert Spaces. Springer, 2017. (Cited on p. 33)

[11] Heinz H Bauschke, Jérôme Bolte, and Marc Teboulle. A descent lemma beyond Lipschitz
gradient continuity: first-order methods revisited and applications. Mathematics of Operations
Research, 42(2):330–348, 2017. (Cited on p. 3, 6, 20, 37)

[12] Amir Beck and Marc Teboulle. Mirror Descent and Nonlinear Projected Subgradient Methods
for Convex Optimization. Operations Research Letters, 31(3):167–175, 2003. (Cited on p. 2,
5, 7, 20)

[13] David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational Inference: A Review for
Statisticians. Journal of the American statistical Association, 112(518):859–877, 2017. (Cited
on p. 1)

[14] Clément Bonet, Nicolas Courty, François Septier, and Lucas Drumetz. Efficient Gradient
Flows in Sliced-Wasserstein Space. Transactions on Machine Learning Research, 2022. (Cited
on p. 20)

[15] Clément Bonet, Lucas Drumetz, and Nicolas Courty. Sliced-Wasserstein Distances and Flows
on Cartan-Hadamard Manifolds. arXiv preprint arXiv:2403.06560, 2024. (Cited on p. 2)

[16] Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and Radon
Wasserstein Barycenters of Measures. Journal of Mathematical Imaging and Vision, 51:22–45,
2015. (Cited on p. 2, 10, 43)

11


---Page Break---
[17] Benoît Bonnet. A Pontryagin Maximum Principle in Wasserstein Spaces for Constrained
Optimal Control Problems. ESAIM: Control, Optimisation and Calculus of Variations, 25:52,
2019. (Cited on p. 3, 23)
[18] Nicolas Bonnotte. Unidimensional and Evolution Methods for Optimal Transportation. PhD
thesis, Université Paris Sud-Paris XI; Scuola normale superiore (Pise, Italie), 2013. (Cited on
p. 2, 43)
[19] Siwan Boufadène and François-Xavier Vialard. On the global convergence of Wasserstein
gradient flow of the Coulomb discrepancy. arXiv preprint arXiv:2312.00800, 2023. (Cited on
p. 2)
[20] Yann Brenier. Polar Factorization and Monotone Rearrangement of Vector-Valued Functions.
Communications on pure and applied mathematics, 44(4):375–417, 1991. (Cited on p. 27)
[21] Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia del Castillo, Kjong-Van
Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Ratsch. Learning Single-Cell
Perturbation Responses using Neural Optimal Transport. bioRxiv, 2021. (Cited on p. 9, 10, 43)
[22] Charlotte Bunne, Andreas Krause, and Marco Cuturi. Supervised Training of Conditional
Monge Maps. Advances in Neural Information Processing Systems, 35:6859–6872, 2022.
(Cited on p. 9)
[23] Charlotte Bunne, Laetitia Papaxanthos, Andreas Krause, and Marco Cuturi. Proximal Optimal
Transport Modeling of Population Dynamics.
In International Conference on Artificial
Intelligence and Statistics, pages 6511–6528. PMLR, 2022. (Cited on p. 1, 9)
[24] Martin Burger, Matthias Erbar, Franca Hoffmann, Daniel Matthes, and André Schlichting.
Covariance-modulated optimal transport and gradient flows. arXiv preprint arXiv:2302.07773,
2023. (Cited on p. 21)
[25] Guillaume Carlier and Chloé Jimenez. On Monge’s Problem for Bregman-like Cost Functions.
Journal of Convex Analysis, 14(3):647, 2007. (Cited on p. 6, 21)
[26] J. A. Carrillo, M. DiFrancesco, A. Figalli, T. Laurent, and D. Slepˇcev.
Global-in-time
weak measure solutions and finite-time aggregation for nonlocal interaction equations. Duke
Mathematical Journal, 156(2):229 – 271, 2011. (Cited on p. 1)
[27] José A Carrillo, Katy Craig, Li Wang, and Chaozhen Wei. Primal dual methods for Wasserstein
gradient flows. Foundations of Computational Mathematics, pages 1–55, 2022. (Cited on p. 9,
40)
[28] Giulia Cavagnari, Giuseppe Savaré, and Giacomo Enrico Sodini. A Lagrangian approach to
totally dissipative evolutions in Wasserstein spaces. arXiv preprint arXiv:2305.05211, 2023.
(Cited on p. 5, 26)
[29] Gong Chen and Marc Teboulle. Convergence Analysis of a Proximal-Like Minimization
Algorithm Using Bregman Functions. SIAM Journal on Optimization, 3(3):538–543, 1993.
(Cited on p. 6)
[30] Zonghao Chen, Aratrika Mustafi, Pierre Glaser, Anna Korba, Arthur Gretton, and Bharath K
Sriperumbudur. (De)-regularized Maximum Mean Discrepancy Gradient Flow. arXiv preprint
arXiv:2409.14980, 2024. (Cited on p. 1, 2)
[31] Ziheng Cheng, Shiyue Zhang, Longlin Yu, and Cheng Zhang. Particle-based Variational
Inference with Generalized Wasserstein Gradient Flow. In Thirty-seventh Conference on
Neural Information Processing Systems, 2023. (Cited on p. 2, 7, 21)
[32] Sinho Chewi, Thibaut Le Gouic, Chen Lu, Tyler Maunu, Philippe Rigollet, and Austin
Stromme. Exponential Ergodicity of Mirror-Langevin Diffusions. Advances in Neural Infor-
mation Processing Systems, 33:19573–19585, 2020. (Cited on p. 2, 21, 44)
[33] Sinho Chewi, Tyler Maunu, Philippe Rigollet, and Austin J Stromme. Gradient descent
algorithms for Bures-Wasserstein barycenters. In Conference on Learning Theory, pages
1276–1304. PMLR, 2020. (Cited on p. 53)
[34] Sinho Chewi, Jonathan Niles-Weed, and Philippe Rigollet. Statistical Optimal Transport. arXiv
preprint arXiv:2407.18163, 2024. (Cited on p. 23)
[35] Lenaic Chizat. Sparse Optimization on Measures with Over-parameterized Gradient Descent.
Mathematical Programming, 194(1):487–532, 2022. (Cited on p. 5)

12


---Page Break---
[36] Lenaic Chizat and Francis Bach. On the Global Convergence of Gradient Descent for Over-
parameterized Models using Optimal Transport. Advances in neural information processing
systems, 31, 2018. (Cited on p. 1)

[37] Dario Cordero-Erausquin. Transport inequalities for log-concave measures, quantitative forms,
and applications. Canadian Journal of Mathematics, 69(3):481–501, 2017. (Cited on p. 21)

[38] Marco Cuturi, Laetitia Meng-Papaxanthos, Yingtao Tian, Charlotte Bunne, Geoff Davis, and
Olivier Teboul. Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein.
arXiv Preprint arXiv:2201.12324, 2022. (Cited on p. 43)

[39] Marco Cuturi, Michal Klein, and Pierre Ablin. Monge, Bregman and Occam: Interpretable
Optimal Transport in High-Dimensions with Feature-Sparse Maps. In Proceedings of the
40th International Conference on Machine Learning, volume 202 of Proceedings of Machine
Learning Research, pages 6671–6682. PMLR, 23–29 Jul 2023. (Cited on p. 7, 9)

[40] Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. How to compute
Hessian-vector products? In ICLR Blogposts 2024, 2024. (Cited on p. 39)

[41] Nabarun Deb, Young-Heon Kim, Soumik Pal, and Geoffrey Schiebinger. Wasserstein Mirror
Gradient Flow as the Limit of the Sinkhorn Algorithm. arXiv preprint arXiv:2307.16421,
2023. (Cited on p. 5, 21, 29, 56)

[42] Sylvain Delattre and Nicolas Fournier. On the Kozachenko–Leonenko entropy estimator.
Journal of Statistical Planning and Inference, 185:69–93, 2017. (Cited on p. 44)

[43] Michael Ziyang Diao, Krishna Balasubramanian, Sinho Chewi, and Adil Salim. Forward-
backward Gaussian variational inference via JKO in the Bures-Wasserstein Space. In Interna-
tional Conference on Machine Learning, pages 7960–7991. PMLR, 2023. (Cited on p. 9, 29,
37, 41, 42, 53)

[44] Hanze Dong, Xi Wang, Lin Yong, and Tong Zhang. Particle-based Variational Inference
with Preconditioned Functional Gradient Flow. In The Eleventh International Conference on
Learning Representations, 2023. (Cited on p. 2, 7, 21)

[45] Chao Du, Tianbo Li, Tianyu Pang, Shuicheng Yan, and Min Lin. Nonparametric Generative
Modeling with Conditional Sliced-Wasserstein Flows. In International Conference on Machine
Learning (ICML), 2023. (Cited on p. 2)

[46] Andrew Duncan, Nikolas Nüsken, and Lukasz Szpruch. On the geometry of Stein variational
gradient descent. Journal of Machine Learning Research, 24(56):1–39, 2023. (Cited on p. 21,
32)

[47] Matthias Erbar. The heat equation on manifolds as a gradient flow in the Wasserstein space.
Annales de l’Institut Henri Poincaré, Probabilités et Statistiques, 46(1), February 2010. (Cited
on p. 4)

[48] Luca Eyring, Dominik Klein, Théo Uscidda, Giovanni Palla, Niki Kilbertus, Zeynep Akata,
and Fabian J Theis. Unbalancedness in Neural Monge Maps Improves Unpaired Domain
Translation. In The Twelfth International Conference on Learning Representations, 2024.
(Cited on p. 9)

[49] Xingdong Feng, Yuan Gao, Jian Huang, Yuling Jiao, and Xu Liu. Relative Entropy Gradient
Sampler for Unnormalized Distribution. Journal of Computational and Graphical Statistics,
pages 1–16, 2024. (Cited on p. 20, 44)

[50] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé,
and Gabriel Peyré. Interpolating between Optimal Transport and MMD using Sinkhorn
Divergences. In The 22nd International Conference on Artificial Intelligence and Statistics,
pages 2681–2690. PMLR, 2019. (Cited on p. 2, 10, 43)

[51] Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, and Michael John Hutchinson.
Diffusion Models for Constrained Domains. Transactions on Machine Learning Research,
2023. ISSN 2835-8856. (Cited on p. 44)

[52] Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, and Valentin De Bortoli.
Metropolis Sampling for Constrained Diffusion Models. Advances in Neural Information
Processing Systems, 36, 2024. (Cited on p. 44)

13


---Page Break---
[53] Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel
de Bézenac, Mickaël Chen, and Alain Rakotomamonjy. Unifying GANs and Score-Based
Diffusion as Generative Particle Models. Advances in Neural Information Processing Systems,
36, 2023. (Cited on p. 1)

[54] Bela A Frigyik, Santosh Srivastava, and Maya R Gupta. Functional Bregman divergence. In
2008 IEEE International Symposium on Information Theory, pages 1681–1685. IEEE, 2008.
(Cited on p. 3, 7)

[55] Wilfrid Gangbo and Adrian Tudorascu. On differentiability in the Wasserstein space and well-
posedness for Hamilton–Jacobi equations. Journal de Mathématiques Pures et Appliquées,
125:119–174, 2019. (Cited on p. 24)

[56] Alfredo Garbuno-Inigo, Franca Hoffmann, Wuchen Li, and Andrew M Stuart. Interacting
Langevin Diffusions: Gradient Structure and Ensemble Kalman Sampler. SIAM Journal on
Applied Dynamical Systems, 19(1):412–441, 2020. (Cited on p. 7, 21, 32)

[57] Xin Guo, Johnny Hong, and Nan Yang. Ambiguity set and learning via Bregman and Wasser-
stein. arXiv preprint arXiv:1705.08056, 2017. (Cited on p. 21)

[58] Gabriele Gut, Markus Herrmann, and Lucas Pelkmans. Multiplexed protein maps link sub-
cellular organization to cellular state. Science (New York, N.Y.), 361, 08 2018. (Cited on p.
10)

[59] Johannes Hertrich, Manuel Gräf, Robert Beinert, and Gabriele Steidl. Wasserstein steepest
descent flows of discrepancies with Riesz kernels. Journal of Mathematical Analysis and
Applications, 531(1):127829, March 2024. (Cited on p. 2, 10)

[60] Johannes Hertrich, Christian Wald, Fabian Altekrüger, and Paul Hagemann. Generative Sliced
MMD Flows with Riesz Kernels. In The Twelfth International Conference on Learning
Representations, 2024. (Cited on p. 2, 10, 43)

[61] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. Ad-
vances in neural information processing systems, 33:6840–6851, 2020. (Cited on p. 9)

[62] Ya-Ping Hsieh, Ali Kavis, Paul Rolland, and Volkan Cevher. Mirrored Langevin Dynamics.
Advances in Neural Information Processing Systems, 31, 2018. (Cited on p. 2)

[63] Xinru Hua, Truyen Nguyen, Tam Le, Jose Blanchet, and Viet Anh Nguyen. Dynamic Flows on
Curved Space Generated by Labeled Data. In Edith Elkind, editor, Proceedings of the Thirty-
Second International Joint Conference on Artificial Intelligence, IJCAI-23, pages 3803–3811.
International Joint Conferences on Artificial Intelligence Organization, 8 2023. Main Track.
(Cited on p. 1)

[64] Qijia Jiang. Mirror Langevin Monte Carlo: the Case Under Isoperimetry. Advances in Neural
Information Processing Systems, 34:715–725, 2021. (Cited on p. 2)

[65] Yiheng Jiang, Sinho Chewi, and Aram-Alexandre Pooladian. Algorithms for mean-field
variational inference via polyhedral optimization in the Wasserstein space. arXiv preprint
arXiv:2312.02849, 2023. (Cited on p. 37)

[66] Richard Jordan, David Kinderlehrer, and Felix Otto. The Variational Formulation of the
Fokker–Planck Equation. SIAM journal on mathematical analysis, 29(1):1–17, 1998. (Cited
on p. 1)

[67] Mohammad Reza Karimi, Ya-Ping Hsieh, and Andreas Krause. Sinkhorn Flow: A Continuous-
Time Framework for Understanding and Generalizing the Sinkhorn Algorithm. arXiv preprint
arXiv:2311.16706, 2023. (Cited on p. 2, 21)

[68] Jaeyeon Kim, Chanwoo Park, Asuman Ozdaglar, Jelena Diakonikolas, and Ernest K Ryu.
Mirror Duality in Convex Optimization. arXiv preprint arXiv:2311.17296, 2023. (Cited on p.
2, 7, 20)

[69] Dominik Klein, Théo Uscidda, Fabian Theis, and Marco Cuturi. Entropic (Gromov) Wasser-
stein Flow Matching with GENOT. arXiv preprint arXiv:2310.09254, 2023. (Cited on p.
9)

[70] Michal Klein, Aram-Alexandre Pooladian, Pierre Ablin, Eugène Ndiaye, Jonathan Niles-Weed,
and Marco Cuturi. Learning elastic costs to shape monge displacements, 2023. (Cited on p. 7)

14


---Page Break---
[71] Anna Korba, Adil Salim, Michael Arbel, Giulia Luise, and Arthur Gretton. A Non-Asymptotic
Analysis for Stein Variational Gradient Descent. Advances in Neural Information Processing
Systems, 33:4672–4682, 2020. (Cited on p. 6, 32)
[72] Anna Korba, Pierre-Cyril Aubin-Frankowski, Szymon Majewski, and Pierre Ablin. Kernel
Stein Discrepancy Descent. In International Conference on Machine Learning, pages 5719–
5730. PMLR, 2021. (Cited on p. 2, 24)
[73] Marc Lambert, Sinho Chewi, Francis Bach, Silvère Bonnabel, and Philippe Rigollet. Varia-
tional inference via Wasserstein gradient flows. Advances in Neural Information Processing
Systems, 35:14434–14447, 2022. (Cited on p. 9, 42)
[74] Nicolas Lanzetti, Saverio Bolognani, and Florian Dörfler. First-Order Conditions for Opti-
mization in the Wasserstein Space. arXiv preprint arXiv:2209.12197, 2022. (Cited on p. 3, 4,
23, 48)
[75] Emanuel Laude and Panagiotis Patrinos. Anisotropic Proximal Gradient. arXiv preprint
arXiv:2210.15531, 2022. (Cited on p. 7, 20)
[76] Emanuel Laude, Andreas Themelis, and Panagiotis Patrinos. Dualities for Non-Euclidean
Smoothness and Strong Convexity under the Light of Generalized Conjugacy. SIAM Journal
on Optimization, 33(4):2721–2749, 2023. (Cited on p. 7, 20)
[77] Flavien Léger and Pierre-Cyril Aubin-Frankowski. Gradient Descent with a General Cost.
arXiv preprint arXiv:2305.04917, 2023. (Cited on p. 2)
[78] Lingxiao Li, Qiang Liu, Anna Korba, Mikhail Yurochkin, and Justin Solomon. Sampling with
Mollified Interaction Energy Descent. In The Eleventh International Conference on Learning
Representations, 2023. (Cited on p. 2)
[79] Ruilin Li, Molei Tao, Santosh S Vempala, and Andre Wibisono.
The Mirror Langevin
Algorithm Converges with Vanishing Bias. In International Conference on Algorithmic
Learning Theory, pages 718–742. PMLR, 2022. (Cited on p. 2)
[80] Wuchen Li. Transport Information Bregman Divergences. Information Geometry, 4(2):
435–470, 2021. (Cited on p. 4, 21, 55, 56)
[81] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow
Matching for Generative Modeling. In The Eleventh International Conference on Learning
Representations, 2023. (Cited on p. 9)
[82] Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao. Mirror Diffusion
Models for Constrained and Watermarked Generation. Advances in Neural Information
Processing Systems, 36, 2024. (Cited on p. 2, 44)
[83] Qiang Liu. Stein Variational Gradient Descent as Gradient Flow. Advances in neural informa-
tion processing systems, 30, 2017. (Cited on p. 7, 20, 32)
[84] Qiang Liu and Dilin Wang. Stein Variational Gradient Descent: A General Purpose Bayesian
Inference Algorithm. Advances in neural information processing systems, 29, 2016. (Cited on
p. 7, 20, 32)
[85] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning
Representations, 2023. (Cited on p. 9)
[86] Antoine Liutkus, Umut Simsekli, Szymon Majewski, Alain Durmus, and Fabian-Robert Stöter.
Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and
Diffusions. In International Conference on Machine Learning, pages 4104–4113. PMLR,
2019. (Cited on p. 1, 2)
[87] Mohammad Lotfollahi, F Alexander Wolf, and Fabian J Theis. scGen predicts single-cell
perturbation responses. Nature methods, 16(8):715–721, 2019. (Cited on p. 10)
[88] Haihao Lu, Robert M Freund, and Yurii Nesterov. Relatively Smooth Convex Optimization by
First-Order Methods, and Applications. SIAM Journal on Optimization, 28(1):333–354, 2018.
(Cited on p. 2, 3, 5, 6, 8, 9, 20, 22, 35, 36, 40, 55)
[89] Chris J Maddison, Daniel Paulin, Yee Whye Teh, and Arnaud Doucet. Dual Space Precondi-
tioning for Gradient Descent. SIAM Journal on Optimization, 31(1):991–1016, 2021. (Cited
on p. 2, 3, 5, 7, 8, 20)

15


---Page Break---
[90] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A Mean Field View of the Landscape
of Two-Layer Neural Networks. Proceedings of the National Academy of Sciences, 115(33):
E7665–E7671, 2018. (Cited on p. 1)

[91] Pierre Monmarché and Julien Reygner. Local convergence rates for Wasserstein gradient
flows and McKean-Vlasov equations with multiple stationary solutions.
arXiv preprint
arXiv:2404.15725, 2024. (Cited on p. 5)

[92] Arkadij Semenoviˇc Nemirovskij and David Borisovich Yudin. Problem complexity and method
efficiency in optimization. Wiley, 1983. (Cited on p. 2, 20)

[93] Sebastian Neumayer, Viktor Stein, and Gabriele Steidl. Wasserstein Gradient Flows for
Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces. arXiv preprint
arXiv:2402.04613, 2024. (Cited on p. 2)

[94] Maxence Noble, Valentin De Bortoli, and Alain Durmus. Unbiased constrained sampling with
Self-Concordant Barrier Hamiltonian Monte Carlo. In Thirty-seventh Conference on Neural
Information Processing Systems, 2023. (Cited on p. 44)

[95] Felix Otto. Double degenerate diffusion equations as steepest descent. Citeseer, 1996. (Cited
on p. 1)

[96] Felix Otto. The Geometry of Dissipative Evolution Equations: the Porous Medium Equation.
Communications in Partial Differential Equations, 26(1-2):101–174, 2001. (Cited on p. 2)

[97] Felix Otto and Cédric Villani. Generalization of an inequality by Talagrand and links with
the logarithmic Sobolev inequality. Journal of Functional Analysis, 173(2):361–400, 2000.
(Cited on p. 24)

[98] Guy Parker. Some Convexity Criteria for Differentiable Functions on the 2-Wasserstein Space.
arXiv preprint arXiv:2306.09120, 2023. (Cited on p. 5, 26)

[99] Juan Peypouquet. Convex Optimization in Normed Spaces: Theory, Methods and Examples.
Springer, 2015. (Cited on p. 22, 33, 54, 55)

[100] Gabriel Peyré. Entropic approximation of Wasserstein gradient flows. SIAM Journal on
Imaging Sciences, 8(4):2323–2351, 2015. (Cited on p. 20)

[101] Aram-Alexandre Pooladian and Jonathan Niles-Weed. Entropic estimation of optimal transport
maps. arXiv preprint arXiv:2109.12004, 2021. (Cited on p. 42, 43)

[102] Julien Rabin, Gabriel Peyré, Julie Delon, and Marc Bernot. Wasserstein Barycenter and its
Application to Texture Mixing. In Scale Space and Variational Methods in Computer Vision:
Third International Conference, SSVM 2011, Ein-Gedi, Israel, May 29–June 2, 2011, Revised
Selected Papers 3, pages 435–446. Springer, 2012. (Cited on p. 2, 43)

[103] Cale Rankin and Ting-Kam Leonard Wong. Bregman-Wasserstein Divergence: Geometry and
Applications. arXiv preprint arXiv:2302.05833, 2023. (Cited on p. 6, 21, 29)

[104] Cale Rankin and Ting-Kam Leonard Wong. JKO schemes with general transport costs. arXiv
preprint arXiv:2402.17681, 2024. (Cited on p. 5, 20, 21, 29)

[105] Maria L. Rizzo and Gábor J. Székely. Energy Distance. WIREs Computational Statistics, 8(1):
27–38, 2016. (Cited on p. 10)

[106] Gareth O Roberts and Richard L Tweedie. Exponential convergence of Langevin distributions
and their discrete approximations. Bernoulli, pages 341–363, 1996. (Cited on p. 1)

[107] Grant M Rotskoff and Eric Vanden-Eijnden. Neural networks as interacting particle systems:
Asymptotic convexity of the loss landscape and universal scaling of the approximation error.
stat, 1050:22, 2018. (Cited on p. 1)

[108] Adil Salim and Peter Richtárik. Primal dual interpretation of the proximal stochastic gradient
Langevin algorithm. Advances in Neural Information Processing Systems, 33:3786–3796,
2020. (Cited on p. 4)

[109] Adil Salim, Anna Korba, and Giulia Luise. The Wasserstein Proximal Gradient Algorithm.
Advances in Neural Information Processing Systems, 33:12356–12366, 2020. (Cited on p. 37,

38, 42)

[110] Filippo Santambrogio. Optimal Transport for Applied Mathematicians, volume 55. Springer,
2015. (Cited on p. 4, 7, 23)

16


---Page Break---
[111] Filippo Santambrogio. Euclidean, metric, and Wasserstein gradient flows: an overview.
Bulletin of Mathematical Sciences, 7:87–154, 2017. (Cited on p. 1)
[112] Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh
Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-Transport Analysis
of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming.
Cell, 176(4), 2019. (Cited on p. 9)
[113] Louis Sharrock, Lester Mackey, and Christopher Nemeth. Learning Rate Free Bayesian
Inference in Constrained Domains. Advances in Neural Information Processing Systems, 36,
2024. (Cited on p. 2, 7, 32, 44)
[114] Jiaxin Shi, Chang Liu, and Lester Mackey. Sampling with Mirrored Stein Operators. In
International Conference on Learning Representations, 2022. (Cited on p. 2, 7, 32, 44)
[115] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and
Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In
International Conference on Learning Representations, 2021. (Cited on p. 9)
[116] Vishwak Srinivasan, Andre Wibisono, and Ashia Wilson. Fast sampling from constrained
spaces using the Metropolis-adjusted Mirror Langevin algorithm. In Proceedings of Thirty
Seventh Conference on Learning Theory, volume 247, pages 4593–4635, 2024. (Cited on p. 2,
44)
[117] Ken’ichiro Tanaka.
Accelerated gradient descent method for functionals of probability
measures by new convexity and smoothness based on transport maps.
arXiv preprint
arXiv:2305.05127, 2023. (Cited on p. 5, 26, 38)
[118] Fuchou Tang, Catalin Barbacioru, Yangzhou Wang, Ellen Nordman, Clarence Lee, Nanlan
Xu, Xiaohui Wang, John Bodeau, Brian B Tuch, Asim Siddiqui, et al. mRNA-Seq whole-
transcriptome analysis of a single cell. Nature methods, 6(5):377–382, 2009. (Cited on p.
10)
[119] Salma Tarmoun, Stewart Slocum, Benjamin David Haeffele, and Rene Vidal. Gradient
Preconditioning for Non-Lipschitz smooth Nonconvex Optimization. 2022. (Cited on p. 7, 10,
20)
[120] Antonio Terpin, Nicolas Lanzetti, and Florian Dörfler. Learning Diffusion at Lightspeed.
Advances in Neural Information Processing Systems, 2024. (Cited on p. 1)
[121] Belinda Tzen, Anant Raj, Maxim Raginsky, and Francis Bach. Variational Principles for
Mirror Descent and Mirror Langevin Dynamics. IEEE Control Systems Letters, 7:1542–1547,
2023. (Cited on p. 2)
[122] Théo Uscidda and Marco Cuturi. The Monge Gap: A Regularizer to Learn All Transport Maps.
In International Conference on Machine Learning, pages 34709–34733. PMLR, 2023. (Cited
on p. 9)
[123] Quang Van Nguyen. Forward-backward splitting with Bregman distances. Vietnam Journal of
Mathematics, 45(3):519–539, 2017. (Cited on p. 6)
[124] Cédric Villani. Topics in Optimal Transportation, volume 58. American Mathematical Soc.,
2003. (Cited on p. 24)
[125] Cédric Villani. Optimal Transport: Old and New, volume 338. Springer, 2009. (Cited on p.

51)
[126] Yifei Wang and Wuchen Li. Information Newton’s Flow: Second-Order Optimization Method
in Probability Space. arXiv preprint arXiv:2001.04341, 2020. (Cited on p. 5, 21, 24, 25, 29)
[127] Yifei Wang, Peng Chen, and Wuchen Li. Projected Wasserstein gradient descent for high-
dimensional Bayesian inference. SIAM/ASA Journal on Uncertainty Quantification, 10(4):
1513–1532, 2022. (Cited on p. 20, 44)
[128] Yifei Wang, Peng Chen, Mert Pilanci, and Wuchen Li. Optimal Neural Network Approximation
of Wasserstein Gradient Direction via Convex Optimization. arXiv preprint arXiv:2205.13098,
2022. (Cited on p. 20, 44)
[129] Andre Wibisono. Sampling as optimization in the space of measures: The Langevin dynamics
as a composite optimization problem. In Conference on Learning Theory, pages 2093–3027.
PMLR, 2018. (Cited on p. 1, 4, 31, 42)

17


---Page Break---
[130] Andre Wibisono. Proximal Langevin Algorithm: Rapid Convergence under Isoperimetry.
arXiv preprint arXiv:1911.01469, 2019. (Cited on p. 5, 29)
[131] Lantian Xu, Anna Korba, and Dejan Slepcev. Accurate Quantization of Measures via Interact-
ing Particle-based Optimization. In International Conference on Machine Learning, pages
24576–24595. PMLR, 2022. (Cited on p. 7)
[132] Kelvin Shuangjian Zhang, Gabriel Peyré, Jalal Fadili, and Marcelo Pereyra. Wasserstein
Control of Mirror Langevin Monte Carlo. In Conference on Learning Theory, pages 3814–
3841. PMLR, 2020. (Cited on p. 2)
[133] Zico Zolter, David Duvenaud, and Matt Johnson. Deep Implicit Layers - Neural ODEs, Deep
Equilibirum Models, and Beyond. Neurips 2020 Tutorial, 2020. (Cited on p. 39)

18


---Page Break---
Appendix

The appendix is organized as follows. In Appendix A, we discuss related works. In Appendix B
and Appendix C, we provide mathematical background respectively on L2 and on the Wasserstein
space. In Appendix D, we provide complementary results for the mirror descent scheme on the
Wasserstein space. In Appendix E, we discuss the relative convexity and smoothness between
functionals. In Appendix F, we study the Bregman proximal gradient scheme, deriving a convergence
result and closed-form updates for the Gaussian case. In Appendix G, we provide more details on the
experiments. In Appendix H, we report the proofs of our results. Finally, auxiliary results are given
in Appendix I.

Contents

A Related works
20

B
Background on L2(µ)
21

B.1
Differential calculus on L2(µ) . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

B.2
Convexity on L2(µ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

C Background on Wasserstein space
23

C.1
Wasserstein differentials
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

C.2
Wasserstein Hessians . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

C.3
Convexity in Wasserstein space . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

D Additional results on mirror descent
27

D.1
Optimal transport maps for mirror descent . . . . . . . . . . . . . . . . . . . . . .
27

D.2
Continuous formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

D.3
Derivation in specific settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

D.4
Mirror scheme with non-pushforward compatible Bregman potentials
. . . . . . .
32

E
Relative convexity and smoothness
33

E.1
Relative convexity and smoothness between Fenchel transforms
. . . . . . . . . .
33

E.2
Relative convexity and smoothness between functionals . . . . . . . . . . . . . . .
35

F
Bregman proximal gradient scheme
37

G Additional details on experiments
39

G.1
Implementing the schemes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
39

G.2
Mirror descent of interaction energies
. . . . . . . . . . . . . . . . . . . . . . . .
40

G.3
Mirror descent on Gaussians . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
41

G.4
Single-cell experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42

G.5
Mirror descent on the simplex
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
44

H Proofs
44

H.1
Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44

19


---Page Break---
H.2
Proof of Proposition 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45

H.3
Proof of Proposition 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45

H.4
Proof of Proposition 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46

H.5
Proof of Proposition 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47

H.6
Proof of Proposition 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
48

H.7
Proof of Proposition 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
48

H.8
Proof of Lemma 11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

H.9
Proof of Proposition 12 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50

H.10 Proof of Proposition 13 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50

H.11 Proof of Proposition 24 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51

H.12 Proof of Proposition 25 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52

H.13 Proof of Lemma 26 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
53

H.14 Proof of Proposition 27 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
53

I
Additional results
54

I.1
Three-point identity and inequality . . . . . . . . . . . . . . . . . . . . . . . . . .
54

I.2
Convergence lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
55

I.3
Some properties of Bregman divergences
. . . . . . . . . . . . . . . . . . . . . .
55

J
NeurIPS Paper Checklist
58

A
Related works

Mirror descent on Rd. Mirror descent has been introduced by Nemirovskij and Yudin [92] to solve
convex optimization problems. Its convergence has been first studied under the assumption that the
objective has a Lipschitz gradient, see e.g. [12]. More recently, Bauschke et al. [11], Lu et al. [88]
provided convergence guarantees by assuming relative smoothness and convexity.

Preconditioned gradient descent on Rd. The preconditioned gradient descent has first been studied
by Maddison et al. [89], providing convergence guarantees under assumptions on the smoothness
and convexity of the preconditioner relative to the Legendre transform of the objective. Kim et al.
[68] underlined connections with the mirror descent, and introduced an accelerated version of the
preconditioned gradient descent. Laude and Patrinos [75], Laude et al. [76] studied a generalized
version of this algorithm by minimizing an anisotropic upper bound and supposing anisotropic
smoothness of the objective. In particular, their analysis for the descent lemma is also valid for a
non-convex smooth objective. Tarmoun et al. [119] also studied preconditioned gradient descent
for non-Lipschitz smooth non-convex problems.

Wasserstein Gradient flows with respect to non-Euclidean geometries. Several existing schemes
are based on time-discretizations of gradient flows with respect to optimal transport metrics, but
different than the Wasserstein-2 distance.

To simplify the computation of the backward scheme, Peyré [100] added an entropic regularization
into the JKO scheme while Bonet et al. [14] considered using the Sliced-Wasserstein distance instead.
More recently, Rankin and Wong [104] suggested using Bregman divergences e.g. when geodesic
distances are not known in closed-forms.

The most popular objective in Wasserstein gradient flows is the KL. However, this can be intricate to
compute as it requires the evaluation of the density at each step, which is not known for particles,
and thus requires approximations using kernel density estimators [127] or density ratio estimators
[6, 49, 128]. Restricting the velocity field to a reproducing kernel Hilbert space (RKHS), an update
in closed-form can be obtained, which is given by the SVGD algorithm [83, 84]. This algorithm

20


---Page Break---
can also be seen as using an alternative Wasserstein metric [46]. However, the restriction to RKHS
can hinder the flexibility of the method. This motivated the introduction of new schemes based on
using the Wasserstein distance with a convex translation invariant cost [31, 44]. Particle systems
preconditioned by they empirical covariance matrix have also been recently considered, and can be
seen as discretization of the Kalman-Wasserstein or Covariance Modulated gradient flow [24, 56].

Mirror descent with flat geometry.
The space of probability distributions can be endowed with
different metrics. When endowed with the Fisher-Rao metric instead of the Wasserstein distance, the
geometry becomes very different. Notably, the shortest path between the two distributions is now a
mixture between them. In this situation, the gradient is the first variation. Aubin-Frankowski et al. [9]
studied the mirror descent in this space and notably showed connections with the Sinkhorn algorithm
when the mirror map and the optimized functionals are KL divergences. Karimi et al. [67] extended
the mirror descent algorithm for more general time steps, and notably recovered the “Wasserstein
Mirror Flow” proposed by Deb et al. [41] as a special case.

Bregman divergence on P2(Rd).
Several works introduced Bregman divergences on P2(Rd).
Carlier and Jimenez [25] first studied the existence of Monge maps for the OT problem with Bregman
costs c(x, y) = dV (x, y) and symmetrized Bregman costs c(x, y) = dV (x, y) + dV (y, x). For
Bregman costs, the resulting OT problem was named the Bregman-Wasserstein divergence and its
properties were studied in [37, 57, 103]. The Bregman-Wasserstein divergence has also been used
by Ahn and Chewi [3] to show the convergence of the Mirror Langevin algorithm while Rankin and
Wong [104] studied its JKO scheme with KL objective. Li [80] introduced the notion of Bregman
divergence on Wasserstein space for a geodesically strictly convex F : P2(Rd) →R as

∀µ, ν ∈P2(Rd), dF(µ, ν) = F(µ) −F(ν) −⟨∇W2F(ν), Tµ
ν −Id⟩L2(ν),
(16)

where Tµ
ν is the OT map between ν and µ w.r.t W2. The Bregman divergence used in our work
and as defined in Definition 1 is more general as it allows using more general maps and contains as
special case (16). Li [80] studied properties of this Bregman divergence for different functionals F
and provided closed-forms for one-dimensional distributions or Gaussian, but did not use it to define
a mirror scheme.

Mirror descent on P2(Rd).
Deb et al. [41] defined a mirror flow by using the continuous for-
mulation. They focused on KL objectives with Bregman potential ϕ(µ) = 1

2W2
2(µ, ν) with some
reference measure ν ∈P2(Rd), and defined the flow as the solution of
φ(µt) = ∇W2ϕ(µt)
d
dtφ(µt) = −∇W2F(µt).
(17)

We note that ϕ is pushforward compatible and hence enters our framework. Also related to our work,
Wang and Li [126] studied a Wasserstein Newton’s flow, which, analogously to the relation between
Newton’s method and mirror descent [32], is another discretization of our scheme for ϕ = F. We
clarify the link with the Mirror Descent algorithm we define in this work with the previous continuous
formulation above in Appendix D.2.

B
Background on L2(µ)

B.1
Differential calculus on L2(µ)

We recall some differentiability definitions on the Hilbert space L2(µ) for µ ∈P2(Rd). Let
ϕ : L2(µ) →R. We start by recalling the notions of Gâteaux and Fréchet derivatives.

Definition 4. A function ϕ : L2(µ) →R is said to be Gâteaux differentiable at T if there exists an
operator ϕ′(T) : L2(µ) →R such that for any direction h ∈L2(µ),

ϕ′(T)(h) = lim
t→0
ϕ(T + th) −ϕ(T)

t
,
(18)

and ϕ′(T) is a linear function. The operator ϕ′(T) is called the Gâteaux derivative of ϕ at T and if
it exists, it is unique.

21


---Page Break---
Definition 5. The Fréchet derivative of ϕ at T ∈L2(µ) in the direction h ∈L2(µ), denoted δϕ(T, h),
is defined implicitly by
ϕ(T + th) = ϕ(T) + tδϕ(T, h) + to(∥h∥).
(19)

If ϕ is Fréchet differentiable, then it is also Gâteaux differentiable, and both derivatives agree, i.e. for
all T, h ∈L2(µ), δϕ(T, h) = ϕ′(T)(h) [99, Proposition 1.26].

Moreover, since L2(µ) is a Hilbert space, and δϕ(T, ·) and ϕ′(T) are linear and continuous, if ϕ is
Fréchet (resp. Gâteaux) differentiable, by the Riesz representation theorem, there exists ∇ϕ ∈L2(µ)
such that for all h ∈L2(µ), δϕ(T, h) = ⟨∇ϕ(T), h⟩L2(µ) (resp. ϕ′(T)(h) = ⟨∇ϕ(T), h⟩L2(µ)).

As a brief comment on these notions in the context of convexity, if the subdifferential of a convex f
at x contains a single element then it is the Gâteaux derivative and we have an inequality f(y) ≥
f(x) + ⟨∇f(x), y −x⟩. Instead Fréchet différentiability gives an equality (19) corresponding to a
series expansion.

B.2
Convexity on L2(µ)

Let ϕ : L2(µ) →R be Gâteaux differentiable. We recall that ϕ is convex if for all t ∈[0, 1],
T, S ∈L2(µ),
ϕ
 
(1 −t)T + tS

≤(1 −t)ϕ(T) + tϕ(S),
(20)
which is equivalent by [99, Proposition 3.10] with
∀T, S ∈L2(µ), ϕ(T) ≥ϕ(S) + ⟨∇ϕ(S), T −S)⟩L2(µ) ⇐⇒dϕ(T, S) ≥0.
(21)

We now present equivalent definitions of the relative smoothness and relative convexity, which is the
equivalent of [88, Proposition 1.1].
Proposition 8. Let ψ, ϕ : L2(µ) →R be convex and Gâteaux differentiable functions. The following
conditions are equivalent:

(a1) ψ is β-smooth relative to ϕ

(a2) βϕ −ψ is a convex function on L2(µ)

(a3) If twice Gâteaux differentiable, ⟨∇2ψ(T)S, S⟩L2(µ) ≤β⟨∇2ϕ(T)S, S⟩L2(µ) for all T, S ∈
L2(µ)

(a4) ⟨∇ψ(T) −∇ψ(S), T −S⟩L2(µ) ≤β⟨∇ϕ(T) −∇ϕ(S), T −S⟩L2(µ) for all T, S ∈L2(µ).

The following conditions are equivalent:

(b1) ψ is α-convex relative to ϕ

(b2) ψ −αϕ is a convex function on L2(µ)

(b3) If twice differentiable, ⟨∇2ψ(T)S, S⟩L2(µ) ≥α⟨∇2ϕ(T)S, S⟩L2(µ) for all T, S ∈L2(µ)

(b4) ⟨∇ψ(T) −∇ψ(S), T −S⟩L2(µ) ≥α⟨∇ϕ(T) −∇ϕ(S), T −S⟩L2(µ) for all T, S ∈L2(µ).

Proof. We do it only for the smoothness. It holds likewise for the convexity.

(a1) ⇐⇒(a2):
∀T, S ∈L2(µ), dψ(T, S) ≤βdϕ(T, S)

⇐⇒∀T, S ∈L2(µ), ψ(T) −ψ(S) −⟨∇ψ(S), T −S⟩L2(µ)
≤β
 
ϕ(T) −ϕ(S) −⟨∇ϕ(S), T −S⟩L2(µ)


⇐⇒∀T, S ∈L2(µ), (βϕ −ψ)(S) −⟨∇(βϕ −ψ)(S), T −S⟩L2(µ) ≤(βϕ −ψ)(T).

(22)

For the rest of the equivalences, we apply [99, Proposition 3.10]. Indeed, βϕ−ψ convex is equivalent
to

∀T, S ∈L2(µ), ⟨∇(βϕ −ψ)(T) −∇(βϕ −ψ)(S), T −S⟩L2(µ) ≥0

⇐⇒∀T, S ∈L2(µ), β⟨∇ϕ(T) −∇ϕ(S), T −S⟩L2(µ) ≥⟨∇ψ(T) −∇ψ(S), T −S⟩L2(µ),
(23)

22


---Page Break---
which gives the equivalence between (a2) and (a4). And if ψ and ϕ are twice differentiables, it is
also equivalent to

∀T, S ∈L2(µ), ⟨∇2(βϕ −ψ)(T)S, S⟩L2(µ) ≥0

⇐⇒∀T, S ∈L2(µ), β⟨∇2ϕ(T)S, S⟩L2(µ) ≥⟨∇2ψ(T)S, S⟩L2(µ),
(24)

which gives the equivalence between (a2) and (a3).

C
Background on Wasserstein space

C.1
Wasserstein differentials

We recall the notion of Wasserstein differentiability introduced in [5, 17, 74]. First, we introduce sub-
and super-differentials.
Definition 6 (Wasserstein sub- and super-differential [17, 74]). Let F : P2(Rd) →(−∞, +∞]
lower semicontinuous and denote D(F) = {µ ∈P2(Rd), F(µ) < ∞}. Let µ ∈D(F). Then, a
map ξ ∈L2(µ) belongs to the subdifferential ∂−F(µ) of F at µ if for all ν ∈P2(Rd),

F(ν) ≥F(µ) +
sup
γ∈Πo(µ,ν)

Z
⟨ξ(x), y −x⟩dγ(x, y) + o
 
W2(µ, ν)

.
(25)

Similarly, ξ ∈L2(µ) belongs to the superdifferential ∂+F(µ) of F at µ if −ξ ∈∂−(−F)(µ).

Then, we say that a functional is Wasserstein differentiable if it admits sub and super differentials
which coincide.
Definition 7 (Wasserstein differentiability, Definition 2.3 in [74]). A functional F : P2(Rd) →R
is Wasserstein differentiable at µ ∈P2(Rd) if ∂−F(µ) ∩∂+F(µ) ̸= ∅. In this case, we say that
∇W2F(µ) ∈∂−F(µ) ∩∂+F(µ) is a Wasserstein gradient of F at µ, satisfying for any ν ∈P2(Rd),
γ ∈Πo(µ, ν),

F(ν) = F(µ) +
Z
⟨∇W2F(µ)(x), y −x⟩dγ(x, y) + o
 
W2(µ, ν)

.
(26)

Recall that the tangent space of P2(Rd) at µ ∈P2(Rd) is defined as

TµP2(Rd) = {∇ψ, ψ ∈C∞
c (Rd)} ⊂L2(µ),
(27)

where the closure is taken in L2(µ), see Ambrosio et al. [5, Definition 8.4.1]. Lanzetti et al. [74,
Proposition 2.5] showed that if F is Wasserstein differentiable, then there is always a unique gradient
living in the tangent space, and we can restrict ourselves without loss of generality to this gradient.

Lanzetti et al. [74] further showed that Wasserstein gradients provide linear approximations even if
the perturbations are not induced by OT plans, i.e. differentials are “strong Fréchet differentials”.
Proposition 9 (Proposition 2.6 in [74]). Let µ, ν ∈P2(Rd), γ ∈Π(µ, ν) any coupling and let F :
P2(Rd) →R be Wasserstein differentiable at µ with Wasserstein gradient ∇W2F(µ) ∈TµP2(Rd).
Then,

F(ν) = F(µ) +
Z
⟨∇W2F(µ)(x), y −x⟩dγ(x, y) + o

 sZ
∥x −y∥2
2 dγ(x, y)

!

.
(28)

Under regularity assumptions, the Wasserstein gradient of F can be computed in practice using the
first variation δF

δµ [110, Definition 7.12], which is defined, if it exists, as the unique function (up to a
constant) such that, for χ satisfying
R
dχ = 0,

d
dtF(µ + tχ)

t=0 = lim
t→0
F(µ + tχ) −F(µ)

t
=
Z δF

δµ (µ) dχ.
(29)

Then the Wasserstein gradient can be computed as ∇W2F(µ) = ∇δF

δµ (µ), see e.g. [34, Proposition
5.10].

We now show that we can relate the Fréchet derivative of ˜Fµ(T) := F(T#µ) with the Wasserstein
gradient of F belonging to the tangent space of P2(Rd) at µ.

23


---Page Break---
Proposition 10. Let F : P2(Rd) →R ∪{+∞} be a Wasserstein differentiable functional on D(F).
Let µ ∈P2(Rd) and ˜Fµ(T) = F(T#µ) for all T ∈D( ˜Fµ). Then, ˜Fµ is Fréchet differentiable, and
for all S ∈D( ˜Fµ), ∇˜Fµ(S) = ∇W2F(S#µ) ◦S.

Proof. See Appendix H.1.

A similar formula can be found in Gangbo and Tudorascu [55, Corollary 3.22], however the space
H used there is not L2(µ) but a lifting L2(Ω; Rd) of measures on random variables. They should
not be confused.

C.2
Wasserstein Hessians

A natural object of interest in the context of optimization over the Wasserstein space is the Hessian of
the objective F, which we define below, according to the original definitions of [97, Section 3] and
[124, Chapter 8]. This notion is usually defined along Wasserstein geodesics.
Definition 8 (Chapter 8 in [124]). The Wasserstein Hessian of F, denoted HFµ is an operator over
TµP2(Rd) verifying ⟨HFµv0, v0⟩L2(µ) =
d2
dt2 F(ρt)

t=0 if (ρt, vt)t∈[0,1] is a Wasserstein geodesic
starting at µ.

If µ is absolutely continuous, Wassertein geodesics starting from µ are curves of the form ρt =
(Id + t∇ψ)#µ for ψ ∈C∞
c (Rd). Using this fact, one can compute the Wasserstein Hessians of
Kullback–Leibler divergence [97], Maximum Mean Discrepancy [7] or Kernel Stein Discrepancy [72]
and many other functionals.

However in this work, we are interested in more general curves, which we call acceleration free, i.e.
µt = (S + tv)#µ with S, v ∈L2(µ). Thus, we define analogously the Hessian along such curves.

Definition 9. We define the Hessian operator HFµ,t : L2(µ) →L2(µ) as the operator satisfying for
all t ∈[0, 1],
d2

dt2 F(µt) = ⟨HFµ,tv, v⟩L2(µ),
(30)

where t 7→µt = (S + tv)#µ for S, v ∈L2(µ).

Note that the Hessian HFµ,t is taken at time t but that the vector field v ∈L2(µ) is in the tangent
space at t = 0, hence the discrepancy with Definition 8 besides the fact that we can have S ̸= Id.

Wang and Li [126] derived a general closed form of the Wasserstein Hessian on tangent spaces
through the first variation of F. Here, we extend their formula along any curve µt = (S + tv)#µ
with S, v ∈L2(µ). We first provide a lemma computing the derivative of the Wasserstein gradient.

Lemma 11. Let F
: P2(Rd) →R be twice continuously differentiable and assume that
δ
δµ∇δF

δµ (µ) = ∇δ2F

δµ2 (µ) for all µ ∈P2(Rd). Let µ ∈P2(Rd) and for all t ∈[0, 1], µt = (Tt)#µ
where Tt is differentiable w.r.t. t with dTt

dt ∈L2(µ). Then,

d
dt(∇W2F(µt) ◦Tt)(x) =
Z 
∇y∇x
δ2F

δµ2
 
(Tt)#µ
 
Tt(x), Tt(y)
dTt

dt (y)

dµ(y)

+ ∇2 δF

δµ
 
(Tt)#µ
 
Tt(x)
dTt

dt (x).
(31)

Proof. See Appendix H.8.

This allows us to define a closed-form for HFµ,t.
Proposition 12. Under the same assumptions as in Lemma 11, let µt = (Tt)#µ with Tt = S + tv,
S, v ∈L2(µ), then HFµ,t : L2(µ) →L2(µ) (as defined in Definition 9) is defined for all v ∈L2(µ),
x ∈Rd as:

HFµ,t[v](x) =
Z
∇y∇x
δ2F

δµ2
 
(Tt)#µ
 
Tt(x), Tt(y)

v(y) dµ(y)+∇2 δF

δµ
 
(Tt)#µ
 
Tt(x)

v(x).

(32)

24


---Page Break---
Proof. See Appendix H.9.

While HFµ,t and HFµt differ in general, in some simple cases their relation boils down to composition
with a pushforward. For instance, if S is invertible, we can write µt as a curve starting from S#µ with
a velocity field v ◦S−1, i.e. µt = (Id + tv ◦S−1)#S#µ. Thus, we recover the original definition of
the Wasserstein Hessian at t = 0 as HFµ,0 = HFS#µ. However, in general, this does not need to be
the case.

Similarly, if Tt is invertible for all t, setting vt = v ◦T−1
t , we can write

d2

dt2 F(µt) = ⟨HFµ,tv, v⟩L2(µ)

=
Z
⟨HFµ,t[v](x), v(x)⟩dµ(x)

=
Z
⟨HFµ,t[v]
 
T−1
t (xt)

, vt(xt)⟩dµt(xt)

= ⟨HFµtvt, vt⟩L2(µt).

(33)

The last line is obtained by leveraging the closed form in Proposition 12 and that µt = (Tt)#µ, as
for all x ∈supp(µ),

HFµ,t[v](x) =
Z
∇y∇x
δ2F

δµ2 (µt)
 
Tt(x), Tt(y)

v(y) dµ(y) + ∇2 δF

δµ (µt)
 
Tt(x)

v(x)

=
Z
∇y∇x
δ2F

δµ2 (µt)(Tt(x), yt)vt(yt) dµt(y) + ∇2 δF

δµ (µt)
 
Tt(x)

v(x)

= HFµt[vt]
 
Tt(x)

,

(34)

and thus HFµ,t[v]
 
T−1
t (x)

= HFµt[vt](x).

Here are two examples of F satisfying
δ
δµ∇δF

δµ = ∇δ2F

δµ2 for which Proposition 12 provides an
expression of the Wasserstein Hessian.

Example 1 (Potential energy). Let V(µ) =
R
V dµ with V twice differentiable with bounded Hessian.
Then, we have δV

δµ (µ) = V and δ2V

δµ2 (µ) = 0 (using (29)). Thus, applying Proposition 12, we recover
for µt = (Tt)#µ,
d2

dt2 V(µt) =
Z 
∇2V
 
Tt(x)

v(x), v(x)

dµ(x).
(35)

Example 2 (Interaction energy). Let W(µ) = 1

2
RR
W(x−y) dµ(x)dµ(y) with W symmetric, twice
differentiable and with bounded Hessian. Then, we have for all x, y ∈Rd, δW

δµ (µ)(x) = (W ⋆µ)(x)

and δ2W

δµ2 (µ)(x, y) = W(x −y) (see e.g. [126, Example 7]), and thus applying Proposition 12, for
µt = (Tt)#µ, the operator is

HWµ,t[v](x) = −
Z
∇2W
 
Tt(x) −Tt(y)

v(y) dµ(y) + (∇2W ⋆(Tt)#µ)(Tt(x))v(x),
(36)

and
d2

dt2 W(µt) =
ZZ
⟨∇2W
 
Tt(x) −Tt(y)
 
v(x) −v(y)

, v(x)⟩dµ(y)dµ(x).
(37)

C.3
Convexity in Wasserstein space

We first recall the definition of α-convex functionals [5, Definition 9.1.1].
Definition 10. F is α-convex along geodesics if for all µ0, µ1 ∈P2(Rd),

∀t ∈[0, 1], F(µt) ≤(1 −t)F(µ0) + tF(µ1) −αt(1 −t)

2
W2
2(µ0, µ1),
(38)

where (µt)t∈[0,1] is a Wasserstein geodesic between µ0 and µ1.

25


---Page Break---
If we want to derive the minimal set of assumptions for the convergence of the gradient descent
algorithms on Wasserstein space, we can actually restrict the smoothness and convexity to specific
curves. In the next proposition, we characterize the convexity along one curve. The relative
smoothness or convexity follows by considering the convexity of respectively βG −F or F −αG.

Proposition 13. Let F : P2(Rd) →R be twice continuously differentiable. Let µ ∈P2(Rd),
T, S ∈L2(µ), µt =
 
Tt


#µ for all t ∈[0, 1] where Tt = (1 −t)S + tT. Furthermore, denote for

t1, t2 ∈[0, 1], ˜µt1→t2
t
=
 
(1 −t)Tt1 + tTt2


#µ. Then, the following statement are equivalent:

(c1) For all t1, t2, t ∈[0, 1], F(˜µt1→t2
t
) ≤(1−t)F
 
(Tt1)#µ)+tF
 
(Tt2)#µ

, i.e. F is convex
along t 7→µt.

(c2) For all t1, t2 ∈[0, 1], we have d ˜
Fµ(Tt2, Tt1) ≥0, i.e.

F
 
(Tt2)#µ

−F
 
(Tt1)#µ

−⟨∇W2F
 
(Tt1)#µ

◦Tt1, Tt2 −Tt1⟩L2(µ) ≥0.

(c3) For all t1, t2 ∈[0, 1],

⟨∇W2F
 
(Tt2)#µ

◦Tt2 −∇W2F
 
(Tt1)#µ

◦Tt1, Tt2 −Tt1⟩L2(µ) ≥0.

(c4) For all s ∈[0, 1], d2

dt2 F(µt)

t=s ≥0.

Proof. See Appendix H.10.

As stated in Section 2, if we require the convexity to hold along all curves with S = Id and T the
gradient of some convex function, i.e. an OT map, then F is convex along geodesics. Likewise, if
the convexity holds for all S, T that are gradients of convex functions, then we obtain the convexity
along generalized geodesics.

If we require the convexity and the smoothness to hold along any curve of the form µt =
 
(1 −
t)S + tT)#µ for µ ∈P2(Rd) and T, S ∈L2(µ), then it coincides with the transport convexity
and smoothness recently introduced by Tanaka [117, Definitions 4.1 and 4.5]. As by Proposition 1,
δ ˜Fµ(S, T −S) = ⟨∇W2F(S#µ) ◦S, T −S⟩L2(µ), and thus the convexity of F on such a curve reads
as follows

d ˜
Fµ(T, S) = F(T#µ) −F(S#µ) −⟨∇W2F(S#µ) ◦S, T −S⟩L2(µ) ≥0.
(39)

And for ˜Gµ(T) = 1

2∥T∥L2(µ), the β-smoothness of F relative to G expresses as

d ˜
Fµ(T, S) = F(T#µ)−F(S#µ)−⟨∇W2F(S#µ)◦S, T−S⟩L2(µ) ≤β

2 ∥T−S∥L2(µ) = βd ˜Gµ(T, S).
(40)

This type of convexity is actually a particular case of the notion of convexity along acceleration-free
curves introduced by Parker [98] (also introduced by Cavagnari et al. [28] under the name of total con-
vexity). The latter requires convexity to hold along any curve of the form µt =
 
(1 −t)π1 + tπ2

#γ
with γ ∈Π(µ, ν), µ, ν ∈P2(Rd) and π1(x, y) = x, π2(x, y) = y. The transport convexity of
Tanaka [117] is thus a particular case for couplings obtained through maps, i.e. γ = (T, S)#µ.
Parker [98] notably showed that this notion of convexity is equivalent to the geodesic convexity for
Wasserstein differentiable functionals.

We can also define the strict convexity using strict inequalities in Proposition 13-(c1)-(c2)-(c3), but
not in (c4) as there are counter examples already for real functions. For instance, f : R →R,
defined as f(x) = x4 for all x ∈R, is strictly convex but f ′′(0) = 0. Thus, for F(µ) =
R
f dµ,
choosing µ = δ0 and T0 = Id, by Example 1, we have that d2

dt2 F(µt)

t=0 = f ′′(0)v(0)2 = 0. But
F(µt) = f
 
tT1(0)

< tf
 
T1(0)

since f is strictly convex and thus F is well strictly convex.

Finally, as we defined the relative α-convexity and β-smoothness of F relative to G using Bregman
divergences in Definition 3, we can show that it is equivalent to F −αG and βG −F being convex.

26


---Page Break---
Proposition 14. Let F, G : P2(Rd) →R be two differentiable functionals. Let µ ∈P2(Rd),
T, S ∈L2(µ) and for all t ∈[0, 1], µt = (Tt)#µ with Tt = (1 −t)S + tT. Then, F is α-convex
(resp. β-smooth) relative to G along t 7→µt if and only if F −αG (resp. βG −F) is convex along
t 7→µt.

Proof. By Definition 3, F is α-convex relative to G along t 7→µt if for all s, t ∈[0, 1],
d ˜
Fµ(Ts, Tt) ≥αd ˜Gµ(Ts, Tt). This is equivalent to d ˜
Fµ−α ˜Gµ(Ts, Tt) ≥0, which is equivalent by
Proposition 13 (c2) (and using Proposition 1) to F −αG convex along t 7→µt. The result for the
β-smoothness follows likewise.

D
Additional results on mirror descent

D.1
Optimal transport maps for mirror descent

Let ϕ : P2(Rd) →R be a strictly convex functional along all acceleration-free curves µt = (Tt)#µ,
t ∈[0, 1] with Tt = (1 −t)S + tT for T, S ∈L2(µ). Denote for µ ∈L2(µ), ϕµ(T) = ϕ(T#µ).
Since ϕ is strictly convex along all acceleration-free curves, by Proposition 13, for all T ̸= S ∈L2(µ),
dϕµ(T, S) > 0 and thus ϕµ is strictly convex. Indeed, recall that

∀T, S ∈L2(µ), dϕµ(T, S) = ϕµ(T) −ϕµ(S) −⟨∇ϕµ(S), T −S⟩L2(µ)
= ϕ(T#µ) −ϕ(S#µ) −⟨∇W2ϕ(S#µ) ◦S, T −S⟩L2(µ),
(41)

where we used Proposition 1 for the computation of the gradient.

Let us now define for all µ, ν ∈P2(Rd),

Wϕ(ν, µ) =
inf
γ∈Π(ν,µ) ϕ(ν) −ϕ(µ) −
Z
⟨∇W2ϕ(µ)(y), x −y⟩dγ(x, y).
(42)

This problem encompasses several previously considered objects, as discussed in more detail in
Remark 1. Our motivation for introducing Equation (42) is to prove that for ϕµ verifying the
assumptions of Proposition 2, its associated Bregman divergence dϕµ satisfies the property given in
Assumption 1. First, we can observe that as γ = (T, S)#µ ∈Π(T#µ, S#µ), we have dϕµ(T, S) ≥
Wϕ(T#µ, S#µ). Then, for µ ∈P2,ac(Rd), assuming that ∇W2ϕ(µ) = ∇ϕµ(Id) is invertible,
we can leverage Brenier’s theorem [20], and show in Proposition 15 that the optimal coupling of
Equation (42) is of the form (Tµ,ν
ϕµ , Id)#µ with Tµ,ν
ϕµ = argminT#µ=ν dϕµ(T, Id). Moreover, if
ν ∈P2,ac(Rd), we also have that Tµ,ν
ϕµ is invertible with inverse ¯Tν,µ
ϕν = argminT#ν=µ dϕν(Id, T).

Proposition 15. Let µ ∈P2,ac(Rd), ν ∈P2(Rd) and assume ∇W2ϕ(µ) invertible. Then,

1. There exists a unique minimizer γ of (42). Besides, there exists a uniquely determined
µ-almost everywhere (a.e.) map Tµ,ν
ϕµ : Rd →Rd such that γ = (Tµ,ν
ϕµ , Id)#µ. Finally,
there exists a convex function u : Rd →R such that Tµ,ν
ϕµ = ∇u ◦∇W2ϕ(µ) µ-a.e.

2. Assume further that ν ∈P2,ac(Rd). Then there exists a uniquely determined ν-a.e. map
¯Tν,µ
ϕν : Rd →Rd such that γ = (Id, ¯Tν,µ
ϕν )#ν. Moreover, there exists a convex function
v : Rd →R such that ¯Tν,µ
ϕν = ∇W2ϕ(µ)−1 ◦∇v ν-a.e., and Tµ,ν
ϕµ ◦¯Tν,µ
ϕν = Id ν-a.e. and
¯Tν,µ
ϕν ◦Tµ,ν
ϕµ = Id µ-a.e.

3. As a corollary, Wϕ(ν, µ) = minT#µ=ν dϕµ(T, Id) = minT#ν=µ dϕν(Id, T).

Proof. 1. Observe that problem (42) is equivalent to

inf
γ∈Π(ν,µ)

Z
∥x −∇W2ϕ(µ)(y)∥2
2 dγ(x, y).
(43)

Then, since for any γ ∈Π(ν, µ),
 
Id, ∇W2ϕ(µ)


#γ ∈Π
 
ν, ∇W2ϕ(µ)#µ

, we have

inf
γ∈Π(ν,µ)

Z
∥x −∇W2ϕ(µ)(y)∥2
2 dγ(x, y) ≥
inf
˜γ∈Π
 
ν,∇W2ϕ(µ)#µ


Z
∥x −z∥2
2 d˜γ(x, z).
(44)

27


---Page Break---
Let µ ∈P2,ac(Rd). Since ∇W2ϕ(µ) is invertible, ∇W2ϕ(µ)#µ ∈P2,ac(Rd). By Brenier’s theorem,
there exists a convex function u such that (∇u)#(∇W2ϕ(µ))#µ = ν and the optimal coupling is of
the form ˜γ∗= (∇u, Id)#∇W2ϕ(µ)#µ. Let γ = (∇u ◦∇W2ϕ(µ), Id)#µ ∈Π(ν, µ), then
Z
∥z −˜y∥2
2 d˜γ∗(z, ˜y) =
Z
∥∇u
 
∇W2ϕ(µ)(y)

−∇W2ϕ(µ)(y)∥2
2 dµ(y)

=
Z
∥x −∇W2ϕ(µ)(y)∥2
2 dγ(x, y).
(45)

Thus, since γ ∈Π(ν, µ), γ is an optimal coupling for (42).

2. We symmetrize the arguments. Assuming ν ∈P2,ac(Rd) and ∇ϕµ(Id) = ∇W2ϕ(µ) invertible,
by Brenier’s theorem, there exists a convex function v such that (∇v)#ν = ∇W2ϕ(µ)#µ (and such
that ∇u ◦∇v = Id ν-a.e. and ∇v ◦∇u = Id ∇W2ϕ(µ)#µ-a.e.) and the optimal coupling is of the
form ˜γ∗= (Id, ∇v)#ν. Let γ = (Id, ∇W2ϕ(µ)−1 ◦∇v)#ν ∈Π(ν, µ), then
Z
∥x −z∥2
2 d˜γ∗(x, z) =
Z
∥x −∇v(x)∥2
2 dν(x)

=
Z
∥x −∇W2ϕ(µ)
 
(∇W2ϕ(µ))−1(∇v(x))

∥2
2 dν(x)

=
Z
∥x −∇W2ϕ(µ)(y)∥2
2 dγ(x, y).

(46)

Thus, since γ ∈Π(ν, µ), γ is an optimal coupling for (42). Moreover, noting Tµ,ν
ϕµ = ∇u◦∇W2ϕ(µ)
and ¯Tν,µ
ϕν = ∇W2ϕ(µ)−1◦∇v, we have µ-a.e., ¯Tν,µ
ϕν ◦Tµ,ν
ϕµ = ∇W2ϕ(µ)−1◦∇v◦∇u◦∇W2ϕ(µ) = Id
and ν-a.e., Tµ,ν
ϕµ ◦¯Tν,µ
ϕν = ∇u ◦∇W2ϕ(µ) ◦∇W2ϕ(µ)−1 ◦∇v = Id from the aforementioned
consequences of Brenier’s theorem.

We continue this section with additional results relative to the invertibility of mirror maps, which are
required in Proposition 2. For a potential energy V(µ) =
R
V dµ, since ∇W2V = ∇V , then ∇W2V
is invertible provided ∇V is. This is the case e.g. for V strictly convex. We now state in the two next
lemmas conditions for an interaction energy and for the negative entropy to satisfy the invertibility
requirements.

Lemma 16. Let µ ∈P2(Rd) and let W : Rd →R be even, ϵ-strongly convex for ϵ > 0 and
differentiable. Then, for W(µ) =
RR
W(x −y) dµ(x)dµ(y), ∇W2W(µ) is invertible.

Proof. On one hand, ∇W2W(µ) = ∇W ⋆µ. Moreover, W ϵ-strongly convex is equivalent to

∀x, y ∈Rd, x ̸= y, ⟨∇W(x) −∇W(y), x −y⟩≥ϵ∥x −y∥2
2,
(47)

which implies for all x, y, z ∈Rd, ⟨∇W(x −z) −∇W(y −z), x −y⟩≥ϵ∥x −y∥2
2. By integrating
with respect to µ, it implies

⟨(∇W ⋆µ)(x)−(∇W ⋆µ)(y), x−y⟩=
Z
⟨∇W(x−z)−∇W(y −z), x−y⟩dµ(z) ≥ϵ∥x−y∥2
2.

(48)
Thus, ∇W ⋆µ is ϵ-strongly monotone, and in particular invertible [2, Theorem 1].

Lemma 17. Let µ ∈P2,ac(Rd) such that its density is of the form ρ ∝e−V with V : Rd →R
ϵ-strongly convex for ϵ > 0. Then, for H(µ) =
R
log
 
ρ(x)

dµ(x) with ρ the density of µ w.r.t the
Lebesgue measure, ∇W2H(µ) is invertible.

Proof. Let µ such distribution. Then, ∇W2H(µ) = ∇log ρ = −∇V . Since V is ϵ-strongly convex,
then ∇V is ϵ-strongly monotone and in particular invertible [2, Theorem 1].

We conclude this section with a discussion of (42) with respect to related work.

28


---Page Break---
Remark 1. The OT problem (42) recovers other OT costs for specific choices of ϕ. For instance, for
ϕµ(T) = 1

2∥T∥2
L2(µ), it coincides with the squared Wasserstein-2 distance. And more generally, for
ϕV
µ (T) =
R
V ◦T dµ, since by Lemma 31, for all T, S ∈L2(µ),

dϕV
µ (T, S) =
Z
dV
 
T(x), S(x)

dµ(x),
(49)

where dV is the Euclidean Bregman divergence, i.e. for all x, y ∈Rd, dV (x, y) = V (x) −V (y) −
⟨∇V (y), x −y⟩, Wϕ coincides with the Bregman-Wasserstein divergence [103]

BV (µ, ν) =
inf
γ∈Π(µ,ν)

Z
dV (x, y) dγ(x, y).
(50)

D.2
Continuous formulation

Let ϕ : L2(µ) →R be pushforward compatible and superlinear. Introducing the (mirror) map
φ(µ) = ∇W2ϕ(µ), we can write informally the mirror descent scheme (4) and its continuous-time
counterpart when τ →0 as
φ(µk) = ∇W2ϕ(µk)
φ(µk+1) ◦Tk+1 = φ(µk) −τ∇W2F(µk)
−−−→
τ→0

φ(µt) = ∇W2ϕ(µt)
d
dtφ(µt) = −∇W2F(µt).
(51)

However,
d
dtφ(µt) =
d
dt∇W2ϕ(µt) = Hϕµt(vt) where Hϕµt : L2(µt) →L2(µt) is the Hessian
operator (defined in Appendix C.2) such that d2

dt2 ϕ(µt) = ⟨Hϕµt(vt), vt⟩L2(µt) and vt ∈L2(µt) is
a velocity field satisfying ∂tµt + div(µtvt) = 0. Thus, the continuity equation corresponding to
the Mirror Flow is given by

∂tµt −div
 
µt(Hϕµt)−1∇W2F(µt)

= 0.
(52)

For ϕV
µ as Bregman potential, since HϕV
µ (v) = (∇2V )v (see Appendix C.2), the flow is a solution
of ∂tµt −div
 
µt(∇2V )−1∇W2F(µt)

= 0. For F(µ) = KL(µ||ν) with ν ∝e−U, this coincides
with the gradient flow of the mirror Langevin [3, 130] and with the continuity equation obtained
in [104] as the limit of the JKO scheme with Bregman groundcosts. For ϕ = F, this coincides with
Information Newton’s flows [126]. Note also that Deb et al. [41] defined mirror flows through the
scheme τ →0 of (51), but focused on F(µ) = KL(µ||ν) and ϕ(µ) = 1

2W2
2(µ, η).

D.3
Derivation in specific settings

In this section, we analyze several novel mirror schemes obtained through the use of different Bregman
potential maps in (4), and used in various applications in Section 5. We start by discussing the scheme
with an interaction energy as Bregman potential. Next, we study mirror descent with negative entropy
or KL divergence as Bregman potential. For the last two, we derive closed-forms for the case where
every distribution is Gaussian, which is equivalent to working on the Bures-Wasserstein space, and to
use the gradient on the Bures-Wasserstein space [43]. In particular, this space is a submanifold of
P2,ac(Rd) and the tangent space is the space of affine maps with symmetric linear term, i.e. of the
form T(x) = b + S(x −m) with S ∈Sd(R).

Interaction mirror scheme.
Consider as Bregman potential an interaction energy ϕµ(T) =
1
2
RR
W
 
(T(x) −T(x′)

dµ(x)dµ(x′). The mirror descent scheme (4) is given by

∀k ≥0, (∇W ⋆µk+1) ◦Tk+1 = ∇W ⋆µk −τ∇W2F(µk).
(53)

For the particular case W(x) = 1

2∥x∥2
2, the scheme can be made more explicit as ∇W ⋆µ(x) =
R
∇W(x −y) dµ(y) =
R
(x −y) dµ(y) = x −m(µ) with m(µ) =
R
ydµ(y) the expectation, and
thus (53) translates as

∀k ≥0, xk+1 −m(µk+1) = xk −m(µk) −τ∇W2F(µk), xk ∼µk.
(54)

On one hand, recall from Example 2 that the Hessian of ϕ is given, for µ ∈P2(Rd), v ∈L2(µ), by

∀x ∈Rd, Hϕµ[v](x) = −
Z
v(y) dµ(y) + v(x),
(55)

29


---Page Break---
since ∇2W = Id. On the other hand, the mirror descent scheme (54) can be written as, for all k ≥0,

yk+1 = yk −τ∇W2F(µk)(xk), yk = xk −m(µk), xk ∼µk.
(56)

Passing to the limit τ →0, we get

dyt

dt = −∇W2F(µt)(xt), yt = xt −m(µt), xt ∼µt,
(57)

where dyt

dt = dxt

dt −dm(µt)

dt
. Now, by setting vt(x) = dxt

dt , by integration by part, we have

d
dtm(µt) =
Z
x ∂tµt = −
Z
x · div(µtvt) =
Z
vt(y) dµt(y).
(58)

Combining the latter equation with (55), we obtain as expected that dyt

dt = Hϕµt[vt](x).

Negative entropy mirror scheme.
Consider the negative entropy ϕ(µ) =
R
log
 
ρ(x)

dµ(x)
where dµ(x) = ρ(x)dx and ϕµ(T) = ϕ(T#µ). For such Bregman potential, the mirror scheme (4)
can be written for all k ≥0 as

∇log ρk+1 ◦Tk+1 = ∇log ρk −τ∇W2F(µk).
(59)

In general, this scheme is not tractable. Nonetheless, supposing that µk = N(mk, Σk) for all k ≥0,
the scheme translates as

−Σ−1
k+1(Tk+1(xk) −mk+1) = −Σ−1
k (xk −mk) −τ∇W2F(µk), xk ∼µk.
(60)

• For an objective functional F(µ) = H(µ) + V(µ) with V (x) = 1
2xT Σ−1x, the scheme is

−Σ−1
k+1(xk+1 −mk+1) = −Σ−1
k (xk −mk) −τ
 
−Σ−1
k (xk −mk) + Σ−1xk


= −(1 −τ)Σ−1
k (xk −mk) −τΣ−1xk
= −
 
(1 −τ)Σ−1
k
+ τΣ−1
xk + (1 −τ)Σ−1
k mk.

(61)

Assuming mk = 0 for all k, we obtain the following update rule for the covariance matrices:

Σ−1
k+1 =
 
(1 −τ)Σ−1
k
+ τΣ−1T Σk
 
(1 −τ)Σ−1
k
+ τΣ−1
.
(62)

We illustrate this scheme in Figure 2.

• For F(µ) = H(µ), we obtain

−Σ−1
k+1(Tk+1(xk) −mk+1) = −(1 −τ)Σ−1
k (xk −mk), xk ∼µk.
(63)

Assuming mk = 0 for all k, for τ < 1, we obtain the following update rule for the covariance
matrices:

Σ−1
k+1 = (1 −τ)2Σ−1
k , i.e.,
(64)

Σk+1 =
1
(1 −τ)2 Σk =
1
(1 −τ)2k Σ0 ∼
τ→0 e2τkΣ0.
(65)

The continuous time analog of this scheme is thus µt : t 7→N(0, Σt) with Σt = e2tΣ0 and the
negative entropy decreases along this curve as

H(µt) =
Z
log
 
ρt(x)

dµt(x)

=
Z
log

 
1

(2π)
d
2 √det Σt
e−1

2 xT Σ−1
t
x
!

dµt(x)

= −d

2 log(2π) −1

2 log det
 
e2tΣ0

−1

2Tr

Σ−1
t

Z
xxT dµt(x)


= −d

2 log(2πe) −dt −1

2

d
X

i=1
log(λi),

(66)

30


---Page Break---
where (λi)i denote the eigenvalues of Σ0. This is much faster than the heat flow for which the
negative entropy decreases as [129, Appendix E.2]

H(ρt) = −d

2 log(2πe) −1

2

d
X

i=1
log(λi + 2t),
(67)

with the scheme given by [129, Example 6]

∀k ≥0,
mk+1 = m0
Σk+1 = Σk(Id + τΣ−1
k )2.
(68)

With our notations, the heat flow is the continuous time limit of the scheme (4) for the same objective
F but for a quadratic Bregman potential ϕµ(T) = 1

2∥T∥2
L2(µ) (which recovers the Wasserstein-2
geometry, hence Wasserstein-2 gradient flows).

KL mirror scheme.
Suppose we want to optimize the KL divergence, i.e. a functional of the form
F(µ) = G(µ) + H(µ) where G(µ) =
R
Udµ. Then, a natural choice of Bregman potential is also a
functional of the form ϕ(µ) = Ψ(µ) + H(µ) with Ψ(µ) =
R
V dµ, with U α-convex and β-smooth
relative to V .

In that case, we obtain the smoothness of F relative to ϕ. Recall we denote ˜Fµ(T) = F(T#µ) for
T ∈L2(µ). Then for all T, S ∈L2(µ), we have αd ˜Ψµ(T, S) ≤d ˜Gµ(T, S) ≤βd ˜Ψµ(T, S), hence

d ˜
Fµ(T, S) = d ˜
Hµ(T, S) + d ˜Gµ(T, S) ≤d ˜
Hµ(T, S) + βd ˜Ψµ(T, S) ≤max(1, β)dϕµ(T, S). (69)

Similarly, d ˜
Fµ(T, S) ≥min(1, α)dϕµ(T, S).

We now focus on the case where all measures are Gaussian in order to be able to compute a closed-
form, i.e. U(x) = 1

2(x−m)T Σ−1(x−m), V (x) = 1

2xT Λ−1x and for all k ≥0, µk = N(mk, Σk).
In this case, recall that ∇log µk(x) = −Σ−1
k (x−mk). Then, at each step, the mirror descent scheme
(4) writes for xk ∼µk, k ≥0 as

∇V (xk+1) + ∇log
 
µk+1(xk+1)

= ∇V (xk) + ∇log
 
µk(xk)

−τ
 
∇U(xk) + ∇log
 
µk(xk)


⇐⇒Λ−1xk+1 −Σ−1
k+1(xk+1 −mk+1)

= Λ−1xk −Σ−1
k (xk −mk) −τ
 
Σ−1(xk −m) −Σ−1
k (xk −mk)


⇐⇒(Λ−1 −Σ−1
k+1)xk+1 + Σ−1
k+1mk+1

=
 
Λ−1 −(1 −τ)Σ−1
k
−τΣ−1
xk + (1 −τ)Σ−1
k mk + τΣ−1m.
(70)

Thus, we get for the expectation that

(Λ−1 −Σ−1
k+1)mk+1 + Σ−1
k+1mk+1 =
 
Λ−1 −(1 −τ)Σ−1
k
−τΣ−1
mk(1 −τ)Σ−1
k mk + τΣ−1m

⇐⇒Λ−1mk+1 = (Λ−1 −τΣ−1)mk + τΣ−1m

⇐⇒mk+1 = (Id −τΛΣ−1)mk + τΛΣ−1m.
(71)

We note that the latter update on the means coincides with the forward Euler method in the forward-
backward scheme, see (116) in Appendix F, which uses as Bregman potential ϕ = Ψ. Thus, the
entropy does not affect the convergence towards the mean, which can be done simply by (precondi-
tioned) gradient descent.

For the covariance part, we get

(Λ−1 −Σ−1
k+1)T Σk+1(Λ−1 −Σ−1
k+1)

=
 
Λ−1 −τΣ−1 −(1 −τ)Σ−1
k
T Σk
 
Λ−1 −τΣ−1 −(1 −τ)Σ−1
k

.
(72)

Now, supposing that all matrices commute, we get

Λ−2Σk+1 −2Λ−1 + Σ−1
k+1 = (Λ−1 −τΣ−1)2Σk −2(1 −τ)Λ−1 + 2τ(1 −τ)Σ−1

+ (1 −τ)2Σ−1
k
(73)

⇐⇒Λ−2Σk+1 + Σ−1
k+1 = (Λ−1 −τΣ−1)2Σk + 2τΛ−1 + 2τ(1 −τ)Σ−1 + (1 −τ)2Σ−1
k
⇐⇒Σk+1 + Λ2Σ−1
k+1 = (Id −τΛΣ−1)2Σk + 2τΛ + 2τ(1 −τ)Λ2Σ−1 + (1 −τ)2Λ2Σ−1
k .

31


---Page Break---
Denoting

C = (Id −τΛΣ−1)2Σk + 2τΛ + 2τ(1 −τ)Λ2Σ−1 + (1 −τ)2Λ2Σ−1
k ,
(74)
the update on covariances is equivalent to
Σ2
k+1 −CΣk+1 + Λ2 = 0.
(75)

Thus, Σk+1 = 1

2
 
C ± (C2 −4Λ2)
1
2 
.

D.4
Mirror scheme with non-pushforward compatible Bregman potentials

We study in this Section schemes for which the Bregman potential ϕµ is not pushforward compatible,
and thus for which we cannot apply Proposition 2 and thus Assumption 1 may not hold a priori.
An example of such potential is ϕµ(T) = ⟨T, PµT⟩L2(µ) where Pµ : L2(µ) →L2(µ) is a linear
autoadjoint and invertible operator. Since ∇ϕµ(T) = PµT, taking the first order conditions, we
obtain the following scheme:
∀k ≥0, Tk+1 = Id −P −1
µk ∇W2F(µk).
(76)

In particular, this includes SVGD [71, 83, 84] if we pose P −1
µ T = ιSµT with Sµ : L2(µ) →
H defined as SµT =
R
k(x, ·)T(x)dµ(x) which maps functions from L2(µ) to the reproducing
kernel Hilbert space H with kernel k, and with ι : H →L2(µ) the inclusion operator that is
the adjoint of Sµ [71]. It also includes the Kalman-Wasserstein gradient descent [56] for which
P −1
µ
=
R  
x −m(µ)

⊗
 
x −m(µ)

dµ(x) is the covariance matrix, where m(µ) =
R
x dµ(x).

More generally, for ϕµ(T) =
R
Pµ(V ◦T)dµ, we can recover their mirrored version, including
mirrored SVGD [113, 114], i.e. Tk+1 = ∇V ∗◦
 
∇V −τP −1
µk ∇W2F(µk)

.

Kalman-Wasserstein.
We focus now on a particular choice of linear operator Pµ. Namely, we take
PµT = C(µ)T with C(µ) =
 R  
x −m(µ)

⊗
 
x −m(µ)

dµ(x)
−1 the inverse of the covariance
matrix. In this case, (76) corresponds to the discretization of the Kalman-Wasserstein gradient flow
[56]. We now show that it satisfies Assumption 1. First, let us compute the Bregman divergence
associated to ϕ:

∀T, S ∈L2(µ), dϕµ(T, S) = 1

2⟨T, C(µ)T⟩L2(µ) + 1

2⟨S, C(µ)S⟩L2(µ) −⟨C(µ)S, T⟩L2(µ)

= 1

2
 
⟨T, C(µ)(T −S)⟩L2(µ) + ⟨S −T, C(µ)S⟩L2(µ)


= 1

2∥C(µ)
1
2 (T −S)∥2
L2(µ).

(77)

For γ = (T, S)#µ, we can write

dϕµ(T, S) = 1

2

Z
∥C(µ)
1
2 (x −y)∥2
2 dγ(x, y).
(78)

Moreover, the problem infγ∈Π(α,β)
R
∥C(µ)
1
2 (x −y)∥2
2 dγ(x, y) is equivalent to

inf
γ∈Π(α,β) −
Z
xT C(µ)y dγ(x, y),
(79)

which is a squared OT problem. Thus, it admits an OT map if C(µ) is invertible and µ or ν is
absolutely continuous.

Second point of view.
Another point of view would be to use the linearization with the gradient
corresponding to the associated generalized Wasserstein distance, which is of the form ∇WF(µ) =
P −1
µ ∇W2F(µ) [46, 56], i.e. considering

Tk+1 = argmin
T∈L2(µ)
dϕµ(T, Id) + ⟨∇WF(µ), T −Id⟩L2(µ),
(80)

where we assume that ∇WF(µ) ∈L2(µ). In that case, using the first order conditions,

∇J(Tk+1) = 0 ⇐⇒∇W2ϕ
 
(Tk+1)#µk

◦Tk+1 = ∇W2ϕ(µk) −τP −1
µk ∇W2F(µk).
(81)
Then, for ϕµ satisfying Assumption 1, the convergence will hold under relative smoothness and
convexity assumptions similarly as for the analysis derived in Section 3.

32


---Page Break---
E
Relative convexity and smoothness

E.1
Relative convexity and smoothness between Fenchel transforms

In this Section, we show sufficient conditions to satisfy the inequalities assumed in Proposition 5
and Proposition 6 under the additional assumption that, for all k ≥0, ˜Fµk is superlinear, lower
semicontinuous and strictly convex. In this case, we can show that ˜F∗
µk is Gâteaux differentiable, and
thus we can use the Bregman divergence of ˜F∗
µk.

Lemma 18. Let ϕ : L2(µ) →R be a superlinear, lower semicontinuous and strictly convex function.
Then, ϕ∗is Gâteaux differentiable.

Proof. Fix g ∈L2(µ). Notice that

¯f ∈∂ϕ∗(g) ⇐⇒ϕ∗(g) = ⟨¯f, g⟩−ϕ( ¯f) =
sup
f∈L2(µ)
⟨f, g⟩−ϕ(f).
(82)

So to prove there is a unique element in ∂ϕ∗(g), we just need to show that, setting ϕg(f) := −⟨f, g⟩+
ϕ(f), the problem inff∈L2(µ) ϕg(f) has a unique solution. Under our assumptions, ϕg is lower
semicontinuous and strictly convex. Since ϕ is superlinear, ϕg is coercive, i.e. lim∥f∥→∞ϕg(f) =
+∞. There thus exists a solution [8, Theorem 3.3.4], which is unique by strict convexity. Hence
∂ϕ∗(g) is reduced to a point, which is necessarily the Gâteaux derivative of ϕ∗at g.

This allows us to relate the Bregman divergence of ϕ∗to the Bregman divergence of ϕ.

Lemma 19. Let ϕ : L2(µ) →R be a proper superlinear and strictly convex differentiable function,
then for all T, S ∈L2(µ), dϕ∗ 
∇ϕ(T), ∇ϕ(S)

= dϕ(S, T).

Proof. By [99, Corollary 3.44], we have ϕ∗ 
∇ϕ(T)

= ⟨T, ∇ϕ(T)⟩L2(µ) −ϕ(T) for all T ∈L2(µ)
since ϕ is convex and differentiable. By Lemma 18, ϕ∗is invertible and by [10, Corollary 16.24],
since ϕ is proper, lower semicontinuous and convex, then (∇ϕ)−1 = ∇ϕ∗.

Thus, for all T, S ∈L2(µ),

dϕ∗ 
∇ϕ(T), ∇ϕ(S)

= ϕ∗ 
∇ϕ(T)

−ϕ∗ 
∇ϕ(S)

−⟨∇ϕ∗ 
∇ϕ(S)

, ∇ϕ(T) −∇ϕ(S)⟩L2(µ)
= ϕ∗ 
∇ϕ(T)

−ϕ∗ 
∇ϕ(S)

−⟨S, ∇ϕ(T) −∇ϕ(S)⟩L2(µ)
= ⟨∇ϕ(T), T⟩L2(µ) −ϕ(T) −⟨∇ϕ(S), S⟩L2(µ) + ϕ(S)
(83)

−⟨S, ∇ϕ(T) −∇ϕ(S)⟩L2(µ)
= ϕ(S) −ϕ(T) −⟨∇ϕ(T), S −T⟩L2(µ)
= dϕ(S, T).

Finally, we can relate the relative convexity of ϕ relative to ψ∗by using an inequality between the
Bregman divergences of ϕ and ψ. In particular, we recover the assumptions of Propositions 5 and 6
for ϕh∗
µk that is β-smooth and α-convex relative to ˜F∗
µk.

Proposition 20. Let ϕ, ψ : L2(µ) →R proper, superlinear, strictly convex and differentiable. ϕ
is β-smooth (resp. α-convex) relative to ψ∗if and only if ∀T, S ∈L2(µ), dϕ
 
∇ψ(T), ∇ψ(S)

≤
βdψ(S, T) (resp. dϕ
 
∇ψ(T), ∇ψ(S)

≥αdψ(S, T)).

Proof of Proposition 20. First, suppose that ϕ is β-smooth relative to ψ∗. Then, by definition,

∀T, S ∈L2(µ), dϕ(T, S) ≤βdψ∗(T, S).
(84)

In particular,

dϕ
 
∇ψ(T), ∇ψ(S)

≤βdψ∗ 
∇ψ(T), ∇ψ(S)

= βdψ(S, T),
(85)

using Lemma 19.

33


---Page Break---
On the other hand, suppose for all T, S ∈L2(µ), dϕ
 
∇ψ(T), ∇ψ(S)

≤βdψ(S, T). Then, by first
using Lemma 19 and then the supposed inequality, we have for all T, S ∈L2(µ),

βdψ∗ 
∇ψ(T), ∇ψ(S)

= βdψ(S, T) ≥dϕ
 
∇ψ(T), ∇ψ(S)

.
(86)

Likewise, we can show that ϕ is α-convex relative to ψ if and only if dϕ
 
∇ψ(T), ∇ψ(S)

≥
αdψ(S, T) for all T, S ∈L2(µ).

Links with the conditions of Proposition 5 and Proposition 6.
Proposition 20 allows to translate
the inequality hypothesis of Proposition 5 and Proposition 6. Assume that for all k, ˜Fµk is strictly
convex, differentiable and superlinear. We first note that it implies that Fµk is convex along t 7→
 
(1 −t)Tk+1 + tId


#µk. Moreover, by Lemma 18, ∇˜F∗
µk is differentiable.

Note that this assumption is satisfied, e.g. by ϕµ(T) =
R
V ◦T dµ for V η-strongly convex and
differentiable. Indeed, in this case, ϕµ is also η-strongly convex, and satisfies for all T, S ∈L2(µ),

dϕµ(T, S) = ϕµ(T) −ϕµ(S) −⟨∇ϕµ(S), T −S⟩L2(µ) ≥η

2∥T −S∥2
L2(µ)

⇐⇒ϕµ(T) ≥ϕµ(S) + ⟨∇ϕµ(S), T −S⟩L2(µ) + η

2∥T −S∥2
L2(µ).
(87)

For S = 0, and dividing by ∥T∥L2(µ) the right term diverges to +∞when ∥T∥L2(µ) →+∞, and
thus lim∥T∥L2(µ)→∞ϕµ(T)/∥T∥L2(µ) = +∞, and ϕµ is superlinear.

This assumption is also satisfied for interaction energies ϕW
µ (T) =
RR
W
 
T(x)−T(y)

dµ(x)dµ(y)
with W η-strongly convex, even and differentiable. Indeed, by strong convexity of W in 0, we have
for all x, y ∈Rd,

W
 
T(x) −T(y)

−W(0) −⟨∇W(0), T(x) −T(y)⟩≥η

2∥T(x) −T(y)∥2
2

≥η

2 inf
z∈Rd ∥T(x) −z∥2
2.
(88)

Integrating w.r.t. µ ⊗µ, we get

ϕW
µ (T) −W(0) ≥η

2 inf
z∈Rd

Z
∥T(x) −z∥2
2 dµ(x),
(89)

and dividing by ∥T∥L2(µ), we get that ϕW
µ is superlinear.

For a curve t 7→µt, we define F∗
µ on µt as F∗
µ(µt) := ˜F∗
µ(Tt) with ˜F∗
µ the convex conjugate
of ˜Fµ in the L2(µ) sense. Then, we can apply Proposition 20, and we obtain that the inequality
hypothesis of Proposition 5 is implied by the β-smoothness of ϕh∗relative to F∗
µk along t 7→
 
(1 −t)∇W2F(µk) + t∇W2F(µk+1) ◦Tk+1


#µk since

dϕh∗
µk
 
∇W2F(µk+1) ◦Tk+1, ∇W2F(µk)

≤βd ˜
Fµk (Id, Tk+1)

= βd ˜
F∗
µk
 
∇W2F(µk+1) ◦Tk+1, ∇W2F(µk)

,
(90)

where we used Proposition 1 to compute the gradient ∇˜Fµk(Tk+1) = ∇W2F(µk+1) ◦Tk+1.

Similarly, the condition of Proposition 6

dϕh∗
µk
 
∇W2F(T#µk) ◦T, ∇W2F(µk)

≥αd ˜
Fµk (Id, T)

= αd ˜
F∗
µk
 
∇W2F(T#µk) ◦T, ∇W2F(µk)

(91)

is implied by the α-convexity of ϕh∗relative to F∗
µk along t
7→
 
(1 −t)∇W2F(µk) +
t∇W2F(T#µk) ◦T


#µk.

These results are summarized in Proposition 7 and shown formally in Appendix H.7.

34


---Page Break---
Convergence towards the minimizer in Proposition 6.
We add an additional result justifying the
convergence towards the minimizer in Proposition 6.

Lemma 21. Let (X, τ) be a metrizable topological space, and f : X →R ∪{+∞} be strictly
convex, τ-lower semicontinuous and with one τ-compact sublevel set. Let x0 ∈X be the minimizer
of f and take a sequence (xn)n∈N such that f(xn) →f(x0). Then, (xn)n∈N τ-converges to x0.

Proof. The existence of the minimum is given by [8, Theorem 3.2.2]. For N large enough, (xn)n≥N
lives in the τ-compact sublevel set of f, since x0 belongs to it and f(x0) is minimal. We can then
consider a subsequence τ-converging to some x∗. By τ-lower semicontinuity, we have f(x0) ≤
f(x∗) ≤lim inf f(xσ(n)) = f(x0), so f(x0) = f(x∗) and by strict convexity x0 = x∗. Since
all subsequences of (xn)n≥N converge to x∗and the space is metrizable, (xn)n∈N τ-converges to
x0.

The typical case is when X is a Hilbert space and τ is the weak topology. One could wish to have
strong convergence under a coercivity assumption, however “In infinite dimensional spaces, the
topologies which are directly related to coercivity are the weak topologies” [8, p86]. Nevertheless
Gâteaux differentiability implies continuity, which paired with convexity gives weak lower semiconti-
nuity [8, Theorem 3.3.3]. We cannot hope for convergence of the norm of xn to come for free, as the
weak convergence would then imply the strong convergence.

E.2
Relative convexity and smoothness between functionals

Let U, V : Rd →R be differentiable and convex functions. We recall that V is α-convex relative to
U if [88]
∀x, y ∈Rd, dV (x, y) ≥αdU(x, y).
(92)

Likewise, V is β-smooth relative to U if

dV (x, y) ≤βdU(x, y).
(93)

Relative convexity and smoothness between potential energies.
By Lemma 31, for Bregman
potentials of the form ϕµ(T) =
R
V ◦T dµ, the Bregman divergence can be written as

∀T, S ∈L2(µ), dϕµ(T, S) =
Z
dV
 
T(x), S(x)

dµ(x).
(94)

Thus, leveraging this result, we can show that relative convexity and smoothness of ϕV
µ relative to ϕU
µ
is inherited by the relative convexity and smoothness of V relative to U.

Proposition 22. Let µ ∈P2(Rd), ϕµ(T) =
R
V ◦T dµ and ψµ(T) =
R
U ◦T dµ where V, U :
Rd →R are C1. If V is α-convex (resp. β-smooth) relative to U : Rd →R, then ϕµ is α-convex
(resp β-smooth) relative to ψµ.

Proof. First, observe (Lemma 31) that

∀µ ∈P2(Rd), T, S ∈L2(µ), dϕµ(T, S) =
Z
dV
 
T(x), S(x)

dµ(x).
(95)

Let µ ∈P2(Rd), T, S ∈L2(µ). If V is α-convex relatively to U, we have for all x, y ∈Rd,

dV
 
T(x), S(y)

≥αdU
 
T(x), S(y)

,
(96)

and hence by integrating on both sides with respect to µ,

dϕµ(T, S) ≥αdψµ(T, S).
(97)

Likewise, we have the result for the β-smoothness.

35


---Page Break---
Relative convexity and smoothness between interaction energies.
Similarly, by Lemma 32,
for Bregman potentials obtained through interaction energies, i.e. ϕµ(T) =
1
2
RR
W
 
T(x) −
T(x′)

dµ(x)dµ(x′), then

∀T, S ∈L2(µ), dϕµ(T, S) = 1

2

ZZ
dW
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′).
(98)

It also allows to inherit the relative convexity and smoothness results from Rd.

Proposition 23. Let µ ∈P2(Rd), W, K : Rd →R be symmetric, C1 and convex. Let ϕµ(T) =
1
2
RR
W
 
T(x) −T(x′)

dµ(x)dµ(x′) and ψµ(T) = 1

2
RR
K
 
T(x) −T(x′)

dµ(x)dµ(x′). If W is
α-convex relative to K, then ϕµ is α-convex relatively to ψµ. Likewise, if W is β-smooth relatively
to K, then ϕµ is β-smooth relatively to ψµ.

Proof. We use first Lemma 32 and then that W is α-convex relatively to K:

dϕµ(T, S) = 1

2

ZZ
dW
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′)

≥α

2

ZZ
dK
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′)

= αdψµ(T, S).

(99)

Likewise, we have the result for the β-smoothness.

Thus, in situations where the objective functional and the Bregman potential are of the same type and
either potential energies or interaction energies, we only need to show the convexity and smoothness
of the underlying potentials or interaction kernels. For instance, let V : Rd →R be a twice-
differentiable convex function, such that ∥∇2V ∥op ≤pr(∥x∥2) with pr a polynomial function of
degree r and ∥· ∥op the operator norm. Then, by [88, Proposition 2.1], V is β-smooth relative to h
where for all x ∈Rd, h(x) =
1
r+2∥x∥r+2
2
+ 1

2∥x∥2
2.

Relative convexity and smoothness between functionals of different types.
When the functionals
do not belong to the same type, comparing directly the Bregman divergences is less straightforward
in general. In that case, one might instead leverage the equivalence relations given by Proposition 13
and Proposition 14, and show that βG −F or F −αG is convex in order to show respectively the
β-smoothness and α-convexity of F relative to G. For instance, we can use the characterization
through Hessians, and thus we would aim at showing

d2

dt2 F(µt) ≤β d2

dt2 G(µt),
d2

dt2 F(µt) ≥α d2

dt2 G(µt),
(100)

along the right curve t 7→µt.

For instance, consider an objective functional F(µ) = 1

2
RR
W(x −y) dµ(x)dµ(x′) and another
functional G(µ) =
R
V dµ. Then, by Example 1 and Example 2, we have, for µt = (Tt)#µ and
Tt = S + tv,

d2

dt2 G(µt) =
Z
⟨∇2V
 
Tt(x)

v(x), v(x)⟩dµ(x),
(101)

and

d2

dt2 F(µt) =
ZZ
⟨∇2W
 
Tt(x) −Tt(y)
 
v(x) −v(y)

, v(x)⟩dµ(x)dµ(y).
(102)

36


---Page Break---
To show the conditions of Proposition 3, we need to take S = Id and v = Tk+1 −Id, and to verify
for t = s ∈[0, 1] the inequality, i.e.

d2

dt2 F(µt)

t=s ≤β d2

dt2 G(µt)

t=s
⇐⇒
ZZ
⟨∇2W
 
Ts(x) −Ts(y)
 
v(x) −v(y)

, v(x)⟩dµk(x)dµk(y)

≤β
Z
⟨∇2V
 
Ts(x)

v(x), v(x)⟩dµk(x)

⇐⇒
Z D
v(x),
Z  
∇2W
 
Ts(x) −Ts(y)

−β∇2V
 
Ts(x)

v(x)

−∇2W
 
Ts(x) −Ts(y)

v(y)

d µk(y)
E
dµk(x) ≤0.

(103)

For example, choosing W(x) = 1

2∥x∥2
2, then ∇2W = Id and F is β-smooth relative to G as long as
∇2V ◦Ts ⪰1

β Id for any s ∈[0, 1].

F
Bregman proximal gradient scheme

In this section, we are interested into minimizing a functional F of the form F(µ) = G(µ) + H(µ)
where G is smooth relative to some function ϕ and H is convex on L2(µ). Different strategies can be
used to tackle this problem. For instance, Jiang et al. [65] restrict the space to particular directions
along which H is smooth while Diao et al. [43], Salim et al. [109] use Proximal Gradient algorithms.
We focus here on the latter and generalize the Bregman Proximal Gradient algorithm [11], also known
as the Forward-Backward scheme. It consists of alternating a forward step on G and then a backward
step on H, i.e. for k ≥0,
 Sk+1 = argminS∈L2(µk) dϕµk (S, Id) + τ⟨∇W2G(µk), S −Id⟩L2(µk),
νk+1 = (Sk+1)#µk
Tk+1 = argminT∈L2(νk+1) dϕνk+1(T, Id) + τH(T#νk+1),
µk+1 = (Tk+1)#νk+1.
(104)
The first step of our analysis is to show that this scheme is equivalent to
˜Tk+1 = argminT∈L2(µk) dϕµk (T, Id) + τ
 
⟨∇W2G(µk), T −Id⟩L2(µk) + H(T#µk)


µk+1 = (˜Tk+1)#µk.
(105)

This is true under the condition that µk ∈P2,ac(Rd) implies that νk+1 ∈P2,ac(Rd).

Proposition 24. Let ϕµ be pushforward compatible, µ0 ∈P2,ac(Rd) and assume that if µk ∈
P2,ac(Rd) then νk+1 ∈P2,ac(Rd). Then the schemes (104) and (105) are equivalent.

Proof. See Appendix H.11.

We are now ready to state the convergence results for the proximal gradient scheme.
Proposition 25. Let µ0 ∈P2,ac(Rd), τ ≤
1
β and F(µ) = G(µ) + H(µ). Consider the iterates
of the Bregman proximal gradient scheme (104), equivalently (105). Let k ≥0. Assume ˜Hµk is
convex on L2(µk) and G β-smooth relative to ϕ along t 7→
 
(1 −t)Id + t˜Tk+1


#µk. Then, for all
T ∈L2(µk),

F(µk+1) ≤H(T#µk)+G(µk)+⟨∇W2G(µk), T−Id⟩L2(µk) + 1

τ dϕµk (T, Id)−1

τ dϕµk (T, ˜Tk+1).
(106)
Moreover, for T = Id,

F(µk+1) ≤F(µk) −1

τ dϕµk (Id, ˜Tk+1),
(107)

37


---Page Break---
i.e., the scheme decreases the objective at each iteration. Additionally, let α ≥0, ν ∈P2(Rd)
and suppose that ϕµ satisfies Assumption 1. If G is α-convex relative to ϕ along t 7→
 
(1 −t)Id +
tTµk,ν
ϕµk


#µk, then for all k ≥1,

F(µk) −F(ν) ≤
α

(1 −τα)−k −1
Wϕ(ν, µ0) ≤1 −ατ

kτ
Wϕ(ν, µ0).
(108)

Proof. See Appendix H.12.

We verify now that Proposition 24 can be applied for mirror schemes of interest. Salim et al. [109,
Lemma 2] showed that it holds for the Wasserstein proximal gradient when using some potential
energies, more precisely with ϕ(µ) =
R 1

2∥· ∥2
2 dµ and G(µ) =
R
U dµ with U (strictly) convex. We
extend their result for G(µ) =
R
U dµ and ϕ(µ) =
R
V dµ for V strictly convex and U β-smooth
relative to V .

Lemma 26. Let µ ∈P2,ac(Rd), G(µ) =
R
U dµ, ϕµ(T) =
R
V ◦T dµ with V strongly convex and
U β-smooth relative to V , and T = ∇V ∗◦(∇V −τ∇U). Assume τ < 1

β , then T#µ ∈P2,ac(Rd).

Sketch of the proof. The proof of the lemma is inspired from [109, Lemma 2]. We apply [5, Lemma
5.5.3], which requires to show that T is injective almost everywhere and that | det ∇T| > 0 almost
everywhere. See Appendix H.13 for the full proof.

To apply Proposition 25, we need H to be convex along some curve. We discuss here the convexity of
the negative entropy along acceleration free curves. Let µ ∈P2,ac(Rd), and denote ρ its density w.r.t
the Lebesgue measure. For H(µ) =
R
f
 
ρ(x)

dx where f : R →R is C1 and satisfies f(0) = 0,
limx→0 xf ′(x) = 0 and x 7→f(x−d)xd is convex and non-increasing on R+, then by [117, Theorem
4.2], H is convex along curves µt =
 
(1 −t)S + tT)#µ obtained with S and T with positive definite
Jacobians. This is the case e.g. for f(x) = x log x, for which H corresponds to the negative entropy.

By Remark 3, to be able to apply the three-point inequality (that is necessary to obtain the descent
lemma), we actually only need H to be convex along
 
(1 −t)˜Tk+1 + tId


#µk and along
 
(1 −

t)˜Tk+1 + tTµk,ν
ϕµk


#µk for the convergence.

Gaussian target.
In what follows, we focus on G(µ) =
R
Udµ with U(x) = 1

2(x −m)T Σ−1(x −
m) for Σ ∈S++
d
(R), H the negative entropy and with a Bregman potential of the form ϕ(µ) =
R
V dµ with V (x) = 1

2xT Λ−1x. Moreover, we suppose µ0 = N(m0, Σ0). In this situation, each
distribution µk is also Gaussian, as the forward and backward steps are affine operations.

Assuming the covariances matrices are full rank, ˜Tk+1 is affine and its gradient is invertible. Moreover,
by Proposition 15, Tµk,ν
ϕµk = ∇u ◦∇W2ϕ(µk) for ∇u an OT map between ∇W2ϕ(µk)#µk and ν.
Since each distribution is Gaussian, and ∇W2ϕ(µk)(x) = Λ−1x is affine, it has a positive definite
Jacobian. Thus, using [117, Theorem 4.2], we can conclude that we can apply Proposition 25.

Closed-form for Gaussians.
Let G(µ) =
R
Udµ with U(x) = 1

2(x −m)T Σ−1(x −m), Σ ∈
S++
d
(R), m ∈Rd, and H(µ) =
R
log
 
ρ(x)

dµ(x) for dµ = ρ(x)dx. For the Bregman potential,
we will choose ϕ(µ) =
R
V dµ for V (x) = 1

2⟨x, Λ−1x⟩. Recall that the forward step reads as

Sk+1 = ∇V ∗◦
 
∇V −τ∇W2G(µk)

,
νk+1 = (Sk+1)#µk.
(109)

Since ∇V (x) = Λ−1x, and µk = N(mk, Σk), we obtain for xk ∼µk,

Sk+1(xk) = Λ
 
Λ−1xk −τΣ−1(xk −m)

= xk −τΛΣ−1(xk −m).
(110)

Thus, the output of the forward step is still a Gaussian of the form νk+1 = N(mk+ 1

2 , Σk+ 1

2 ) with
(
mk+ 1

2 = (Id −τΛΣ−1)mk + τΛΣ−1m
Σk+ 1

2 = (Id −τΛΣ−1)T Σk(Id −τΛΣ−1).
(111)

38


---Page Break---
Since ∇V is linear, the output of the backward step stays Gaussian. Moreover, the first order
conditions give

∇V ◦Tk+1 + τ∇log(ρk+1 ◦Tk+1) = ∇V

⇐⇒∀x, Λ−1x = Λ−1Tk+1(x) −τΣ−1
k+1(Tk+1(x) −mk+1)

⇐⇒∀x, x = Tk+1(x) −τΛΣ−1
k+1(Tk+1(x) −mk+1).
(112)

Thus, the output is a Gaussian N(mk+1, Σk+1) with (mk+1, Σk+1) satisfying
(
mk+1 = mk+ 1

2
Σk+ 1

2 = (Id −τΛΣ−1
k+1)T Σk+1(Id −τΛΣ−1
k+1).
(113)

Moreover, if Λ and Σk+1 commute, this is equivalent to

Σ2
k+1 −(2τΛ + Σk+ 1

2 )Σk+1 + τ 2Λ2 = 0,
(114)

which solution is given by

Σk+1 = 1

2
 
Σk+ 1

2 + 2τΛ + (Σk+ 1

2 (4τΛ + Σk+ 1

2 ))
1
2 
.
(115)

To sum up, the update is
(
νk+1 = N
 
(Id −τΛΣ−1)mk + τΛΣ−1m, (Id −τΛΣ−1)T Σk(Id −τΛΣ−1)


µk+1 = N
 
mk+ 1

2 , 1

2(Σk+ 1

2 + 2τΛ + (Σk+ 1

2 (4τΛ + Σk+ 1

2 ))
1
2 )

.
(116)

For Λ = Σ, we call it the ideally preconditioned Forward-Backward scheme (PFB).

G
Additional details on experiments

G.1
Implementing the schemes

In this subsection, we sum up how to implement the different schemes in practice, given a finite
number of particles. In all cases, we first sample x(0)
1 , . . . , x(0)
n
∼µ0, then we apply the scheme to
ˆµ(k)
n
= 1

n
Pn
i=1 δx(k)
i .

Mirror descent.
In general, for ϕ pushforward compatible, one needs to solve at each iteration
k ≥0,
∇W2ϕ(µk+1) ◦Tk+1 = ∇W2ϕ(µk) −τ∇W2F(µk).
(117)

If ϕ(µ) =
R
V dµ with ∇V having an analytical inverse, the scheme can be implemented as

∀k ≥0, ∀i ∈{1, . . . , n}, x(k+1)
i
= ∇V ∗ 
∇V (x(k)
i
) −τ∇W2F(ˆµ(k)
n )(x(k)
i
)

.
(118)

Except for this case, one cannot in general invert ∇W2ϕ(µk+1) ◦Tk+1 directly. A practical
workaround is to solve an implicit problem, see e.g. [133]. Here, we use the Newton-Raphson
algorithm. Suppose we have µk = 1

n
Pn
i=1 δx(k)
i
and we are looking for µk+1 = 1

n
Pn
i=1 δxi. Then,
the scheme is equivalent to

∀j ∈{1, . . . , n}, Gj(x1, . . . , xn) = 0,
(119)

for

Gj(x1, . . . , xn) = ∇W2ϕ

 
1
n

n
X

i=1
δxi

!

(xj) −∇W2ϕ(µk)(x(k)
j ) + τ∇W2F(µk)(x(k)
j ).
(120)

Write G(x1, . . . , xn) =
 
G1(x1, . . . , xn), . . . , Gn(x1, . . . , xn)

. Then, at each step k, we perform
the following Newton iterations, starting from (x(k)
1 , . . . , x(k)
n ):

(x(kℓ+1)
1
, . . . , x(kℓ+1)
n
) = (x(kℓ)
1
, . . . , x(kℓ)
n
)−γ
 
JG(x(kℓ)
1
, . . . , x(kℓ)
n
)
−1G(x(kℓ)
1
, . . . , x(kℓ)
n
). (121)
The Jacobian is of size nd × nd, which does not scale well with the dimension and the number
of samples. We can reduce the complexity of the algorithm by relying on inverse Hessian vector
products, see e.g. [40].

39


---Page Break---
0.0
2.5
5.0
7.5
10.0
Time (
 iterations)

0.16

0.14

0.12

0.10

0.08

(
t)

K(x) = 1
2 x 2
2

K(x) = 1
4 x 4
2 + 1
2 x 2
2

0

2

4

6

8

10
t

Figure 4: (Left) Value of W along the flow for
two difference interaction Bregman potentials,
(Right) Trajectories of particles to minimize W.

0
2
4
6
8
10
12
14
Time (
 Iterations)

10
14

10
10

10
6

10
2

102

KL(
t||
* )

NEM
PFB
FB
PKLM
KLM

Figure 5:
Convergence towards Gaussian
N(0, D) with D diagonal and uniformly sam-
pled on [0, 50]10.

Preconditioned gradient descent.
Plugging the empirical measure in (11), the preconditioned
scheme can be implemented as

∀k ≥0, ∀i ∈{1, . . . , n}, x(k+1)
i
= x(k)
i
−τ∇h∗ 
∇W2F(ˆµ(k)
n )(x(k)
i
)

.
(122)

G.2
Mirror descent of interaction energies

Details of Section 5.
We detail in this Section the first experiment of Section 5. We aim at
minimizing the interaction energy W(µ) = 1

2
RR
W(x−y) dµ(x)dµ(y) for W(z) = 1

4∥z∥4
2−1

2∥z∥2
2.
It is well-known that the stationary solution of its gradient flow is a Dirac ring [27]. Since the
stationary solution is translation invariant, we project the measures to be centered.

We study here two Bregman potentials which are also interaction energies. First, observing that
∇2W(z) = 2zzT +
 
∥z∥2
2 −1

Id, we have for all z,

∥∇2W∥op ≤2∥z∥2
2 + ∥z∥2
2 + 1 = 3∥z∥2
2 + 1 = p2(∥z∥2),
(123)

with p2(t) = 3t2 + 1. Thus, by [88, Remark 2], W is β-smooth relative to K4(z) = 1

4∥z∥4
2 + 1

2∥z∥2
2
with β = 4. Thus, using Proposition 23, ˜
Wµ is β-smooth relative to ϕµ(T) = 1

2
RR
K
 
T(x) −
T(x′)

dµ(x)dµ(x′) for all µ, and we can apply Proposition 3.

Under the additional hypothesis that the measures are compactly supported, and thus there exists
M > 0 such that ∥x∥2
2 ≤M for µ-almost every x, we can also show that W is β-smooth relative to
K2(z) = 1

2∥z∥2
2. Indeed, on one hand, ∇2K = Id and ∇2W(z) = 2zzT +
 
∥z∥2
2 −1

Id. Thus, for
all v, z ∈Rd,

vT ∇2W(z)v = 2⟨z, v⟩2 + (∥z∥2
2 −1)∥v∥2
2 ≤3∥z∥2
2∥v∥2
2 ≤3M∥v∥2
2 = 3MvT ∇2K(z)v. (124)

In Figure 4, we plot the evolution of W along the flows obtained with these two Bregman potential,
starting from µ0 = N(0, 0.252I2) for n = 100 particles, with a step size of τ = 0.1 for 120 epochs.

Ill-conditioned interaction energy.
We also study the minimization of an interaction energy with
an ill-conditioned kernel W(z) = 1

4(zT Σ−1z)2 −1

2zT Σ−1z where Σ ∈S++
d
(R) but is possibly
badly conditioned, i.e. the ratio between the largest and smallest eigenvalues is large. In this
case, the stationary solution becomes an ellipsoid instead of a ring. In our experiments, we take
Σ = diag(100, 0.1). For each scheme, we use µ0 = N(0, 0.252I2), n = 100 particles and a step
size of τ = 0.1.

On Figure 1, we use Bregman potentials which take into account this conditioning, namely we use
KΣ
2 (z) = 1

2zT Σ−1z and KΣ
4 (z) = 1

4(zT Σ−1z)2−1

2(zT Σ−1z), and we observe that the convergence
is much faster compared to the same kernels without preconditioning. For KΣ
2 (z) = 1

2zT Σ−1z, the
scheme becomes

(∇K ⋆µk+1) ◦Tk+1 = ∇K ⋆µk −γ∇W2F(µk)

⇐⇒Σ−1 
Tk+1 −m(µk+1)

= Σ−1 
Id −m(µk)

−γΣ−1(IdT Σ−1Id −1)Id

⇐⇒Tk+1 −m(µk+1) = Id −m(µk) −γ(IdT Σ−1Id −1)Id.

(125)

40


---Page Break---
0
5
10
Time (
 iterations)

0.05

0.08

0.11

0.14

0.17

(
t)

K2
K4

K2
K4

5
0
5

1.0

0.5

0.0

0.5

1.0

K4

1
0
1

K4

0

2

4

6

8

10

12
t

0
5
10
Time (
 iterations)

0.05

0.08

0.11

0.14

0.17

(
t)

GD
PredGD

5
0
5

1.0

0.5

0.0

0.5

1.0

PrecGD

1
0
1

GD

0

2

4

6

8

10

12
t

Figure 6: (Left) Value of W over time and trajectory of particles using K4 and KΣ
4 as interaction
kernels. (Right) Value of W over time and trajectory of particles for the Wasserstein gradient descent
and preconditioned Wasserstein gradient descent (with the ideal preconditioner h∗(x) = 1

2xT Σx).

Thus, we see that Σ−1 has less influence which might explain the faster convergence.

Similarly as in the case without preconditioning, using that ∇2W(z) = 2Σ−1zzT Σ−1 +(zT Σ−1z −
1)Σ−1, we can show that

vT ∇2W(z)v = 2⟨z, v⟩2
Σ−1 + (∥z∥2
Σ−1 −1)∥v∥2
Σ−1 ≤3M∥v∥2
Σ−1 = 3MvT ∇2K(z)v.
(126)

For the sake of comparison, we also report on Figure 6 the trajectories of particles for the use of K4
and KΣ
4 , as well as of the usual Wasserstein gradient descent and the preconditioned Wasserstein
gradient descent obtained with h∗(x) = 1

2xT Σx (which is equivalent to the Mirror Descent with
ϕV
µ as Bregman potential and V (x) = 1

2xT Σ−1x). We observe almost the same trajectories as K2,
which would indicate that the target is also smooth compared to ϕV
µ .

Runtime.
These experiments were run on a personal Laptop with a CPU Intel Core i5-9300H.
For the interaction energy as Bregman potential, running the algorithm with Newton’s method for
n = 100 particles in dimension d = 2 for 120 epochs took about 5mn for K2 and KΣ
2 , and about 1h
for K4 and KΣ
4 .

G.3
Mirror descent on Gaussians

As the mirror descent scheme cannot be computed in closed-form for Bregman potentials which are
not potential energies, and thus are computationally costly, we propose here to restrain ourselves to
the Gaussian setting.

We choose as target distribution ν = N(0, Σ) for Σ a symmetric positive definite matrix in R10×10,
and the functional to be optimized is F(µ) =
R
V dµ + H(µ) with V (x) = 1

2xT Σ−1x. The initial
distribution is always chosen as µ0 = N(0, Id). In all cases, the step size is chosen as τ = 0.01,
and we run the scheme for 1500 iterations. For the target distributions, we sample 20 random
covariances of the form Σ = UDU T with D evenly spaced in log scale between 1 and 100, and
U ∈R10×10 chosen as a uniformly random orthogonal matrix, as in [43], and we report the averaged
KL divergence over iterations in Figure 2. We add on Figure 5 the same experiments with targets
of the form N(0, D) where D is a diagonal matrix on R10×10 sampled uniformly over [0, 50]10.
We compare here the Forward-Backward (FB) scheme of [43], the ideally preconditioned Forward-
Backward scheme (PFB), which uses the closed-form (116) derived in Appendix F with Λ = Σ, and
the Mirror Descent with negative entropy Bregman potential (NEM), whose closed-form was derived
in Appendix D.3, and which we recall:

∀k ≥0, Σ−1
k+1 =
 
(1 −τ)Σ−1
k
+ τΣ−1T Σk
 
(1 −τ)Σ−1
k
+ τΣ−1
.
(127)

We also experiment with the KL divergence as Bregman potential (KLM) and the ideally precon-
ditioned KL divergence (PKLM). We observe that, even though the objective is convex relative to
the Bregman potential, this scheme does not always converge. It might be due to its gradient which
might not always be invertible. We leave further investigations for future works.
Remark 2. We note that using as Bregman potential ϕµ(T) =
R
ψ ◦Tdµ for ψ(x) = 1

2xT Λ−1x is
equivalent to using a preconditioner with h∗(x) = 1

2xT Λx.

Analysis of the convergence.
It is well-known that along the Wasserstein gradient flow of the
KL divergence starting from a Gaussian and with a Gaussian target (Ornstein-Uhlenbeck process),

41


---Page Break---
scRNAseq
4i
4i
4i
scRNAseq
scRNAseq

with precond.
without precond.

entropic map  T"
entropic map  T"
entropic map  T"
entropic map  T"
entropic map  T"
entropic map  T"

F(µ) = SW2

2(µ, ⌫)
F(µ) = S2

",2(µ, ⌫)
F(µ) = SW2

2(µ, ⌫)
F(µ) = S2

",2(µ, ⌫)
F(µ) = ED(µ, ⌫)
F(µ) = ED(µ, ⌫)

Figure 7: Preconditioned GD and (vanilla) GD vs. the entropic map Tε [101] to predict the responses
of cell populations to cancer treatments on 4i and scRNAseq datasets, providing respectively 34 and
9 treatment responses. For each profiling technology and each treatment, we have a pair (µi, νi) of
source (untreated) cells and target (treated) cells. For each pair (µi, νi), with both preconditioned
GD and vanilla GD, we minimize the functional F(µ) = D(µ, νi)–with D a metric–to recover the
effect of the perturbation. In both cases, the prediction is obtained by ˆµi = minµ F(µ). We then fit
an entropic map Tε and predict Tε♯µi. We then compare the objective function values F( ˆµi) and
F(Tε♯ˆµi). A point below the diagonal y = x then refers to an experiment in which (preconditioned)
WGD provides a better estimate of the perturbed population.

the measures stay Gaussian [129]. Thus, the Forward-Backward scheme has Gaussian iterates
at each step [43, 109]. In this work, we also use a linearly preconditioned Forward-Backward
scheme, whose closed-form is derived in (116) (Appendix F). For the Bregman potential, we choose
ϕµ(T) =
R
ψ◦T dµ for ψ(x) = 1

2xT Σx. In this situation, G(µ) =
R
V dµ is 1-smooth and 1-convex
relative to ϕ. Thus, we can apply Proposition 25. We refer to Appendix F for more details on the
convexity of H.

For Bregman potentials whose gradient is not affine, the distributions do not necessarily stay Gaussian
along the flows. Thus, we work on the Bures-Wasserstein space and use the Bures-Wasserstein
gradient, i.e. we project the gradient on the space of affine maps with symmetric linear term, i.e. of
the form T(x) = b + S(x −m) with S ∈Sd(R) [43]. We refer to [43, 73] for more details on this
submanifold. This can be seen as performing Variational Inference. We derive the closed-form of the
different schemes in Appendix D.3.

Even though these procedures do not fit exactly the theory developed in this work, we show the relative
smoothness of F relative to H along the curve µt =
 
(1 −t)Id + tTk+1


#µk under the hypothesis
that the covariances matrices have bounded eigenvalues. Moreover, since d ˜
Fµ = dϕV
µ + d ˜
Hµ ≥d ˜
Hµ,
F is also 1-convex relative to H.

Proposition 27. Let λ > 0, F(µ) =
R
V dµ + H(µ) with V (x) = 1

2xT Σ−1x where Σ ∈S++
d
(R)
and Σ ⪯λId. Suppose that for all k ≥0, (1 −τ)Σk+1Σ−1
k
+ τΣk+1Σ−1 ⪰0. Then, F is smooth
relative to H along µt =
 
(1 −t)Id + tTk+1)#µk where µk = N(0, Σk) with Σk ∈S++
d
(R),
Σk ⪯λId.

Proof. See Appendix H.14.

G.4
Single-cell experiments

First, we provide more details on the experiment on single cells of Section 5. Then, we detail a
second experiment comparing the method with using a static map.

42


---Page Break---
Details on the metrics.
We show the benefits of using the polynomial preconditioner over the
single-cell datasets for different metrics.

• The first one considered is the Sliced-Wasserstein distance [16, 102], defined as

∀µ, ν ∈P2(Rd), SW2
2(µ, ν) =
Z

Sd−1 W2
2(P θ
#µ, P θ
#ν) dλ(θ),
(128)

where Sd−1 = {θ ∈Rd, ∥θ∥2 = 1}, λ denotes the uniform distribution on Sd−1 and for all
θ ∈Sd−1, x ∈Rd, P θ(x) = ⟨x, θ⟩. For F(µ) = 1

2SW2
2(µ, ν), the Wasserstein gradient can be
computed as [18]

∇W2F(µ) =
Z

Sd−1 ψ′
θ
 
P θ(x)

θ dλ(θ),
(129)

where, for t ∈R, ψ′
θ(t) = t −F −1
P θ
#ν
 
FP θ
#µ(t)

with FP θ
#µ the cumulative distribution function of

P θ
#µ. In practice, we compute SW and its gradient using a Monte-Carlo approximation by first
drawing L uniform random directions θ1, . . . , θL.

• The second one considered is the Sinkhorn divergence [50] defined as

∀µ, ν ∈P2(Rd), S2
ε,2(µ, ν) = OTε(µ, ν) −1

2OTε(µ, µ) −1

2OTε(ν, ν),
(130)

with

OTε(µ, ν) =
inf
γ∈Π(µ,ν)

Z
∥x −y∥2
2 dγ(x, y) + εKL(γ||µ ⊗ν),
(131)

the entropic regularized OT. The Wasserstein gradient of S2
ε,2 is simply obtained as the potential [50].

• Finally, we also consider the energy distance, defined as

∀µ, ν ∈P2(Rd), ED(µ, ν) = −
ZZ
∥x −y∥2 d(µ −ν)(x)d(µ −ν)(y).
(132)

To compute its Wasserstein gradient, we use the sliced procedure of [60].

Parameters chosen.
For all the metrics, we fixed the step size at τ = 1. To choose the parameter a
of the preconditioner h∗(x) = (∥x∥a
2 + 1)1/a −1, we ran a grid search over a ∈{1.25, 1.5, 1.75}
for a random treatment, and used it for all the others. In particular, we used for the dataset 4i a = 1.5
for the Sinkhorn divergence and for SW, and a = 1.75 for the energy distance. For the scRNAseq
dataset, we used a = 1.25 for the Sinkhorn divergence and SW, and a = 1.5 for the energy distance.
We note that for the dataset 4i, the data lie in dimension d = 48 and d = 50 for scRNAseq. For all
the metrics, we first sampled 4096 particles from the source (untreated) dataset, and used in average
between 2000 and 3000 samples from the target dataset. For the test value, we also added 40% of
unseen cells following [21]. Note that we reported the results in Figure 3 for 3 different initializations
for each treatment, and reported these results with their mean. We report the results using a fixed
relative tolerance tol = 10−3, i.e. at the first iteration where |F(µk) −F(µk−1)|/F(µk−1) ≤tol,
with a maximum value of iterations of 104. For the Sinkhorn divergence, we chose ε as 10−1 time
the variance of the target. Finally, for SW and the computation of the gradient of the energy distance,
we used a Monte-Carlo approximation with L = 1024 projections.

Comparison to an OT static map.
We now compare the prediction of the response of cells to
a perturbation using Wasserstein gradient descent, with and without preconditioning, to the one
provided by a static estimator, the entropic map Tε [101]. This experiment motivates the use of a
dynamic procedure, iterating multiple steps to map the unperturbed population µ to the perturbed
population ν, instead of a unique static step. We use the proteomic dataset [21] as the one considered
in 3. We use the default OTT-JAX [38] of Tε. The results are shown in Figure 7.

Runtime.
For this experiment, we used a GPU Tesla P100-PCIE-16GB. Depending on the conver-
gence and on the metric considered, each run took in between 30s and 10mn. So in total, it took a
few hundred of hours of computation time.

43


---Page Break---
Initial particles
Final particles MD
Final particles MLD
Groundtruth

0
50
100
150
200
Iterations

24

25

26

27

28

29

(
t)

MD
MLD

Figure 8: (Left) Samples from a Dirichlet posterior distribution for Mirror Descent (MD) and Mirror
Langevin (MLD). (Right) Evolution of the objective averaged over 20 different initialisations.

G.5
Mirror descent on the simplex

We can also leverage the mirror map to perform sampling in constrained spaces. This has received
a lot of attention recently either through mirror Langevin methods [3, 32, 116], diffusion methods
[51, 82], mirror SVGD [113, 114] or other MCMC algorithms [52, 94].

The goal here is to sample from a Dirichlet distribution, i.e. from a distribution ν ∝e−V where
V (x) = −Pd
i=1 ai log(xi) −ad+1 log

1 −Pd
i=1 xi

. To sample from such a distribution, we

minimize the Kullback-Leibler divergence, i.e. F(µ) = KL(µ||ν) =
R
V dµ + H(µ). To stay on
the (open) simplex ∆d = {x ∈Rd+1, xi > 0, Pd+1
i=1 xi < 1}, we use the mirror map ϕ(µ) =
R
ψdµ
with ψ(x) = Pd
i=1 xi log(xi) + (1 −P

i xi) log(1 −P

i xi) for which

∇ψ(x) =



log xi −log
 
1 −
X

j
xj





i

,
∇ψ∗(y) =

 
eyi

1 + P

j eyj

!

i
.
(133)

The scheme here is given by Tk+1 = ∇ψ∗◦(∇ψ −γ∇W2F(µk)), where ∇W2F(µk) = ∇V +
∇log µk, with the density of µk estimated through a Kernel Density Estimator (KDE). We plot on
Figure 8a the results obtained for d = 2, a1 = a2 = a3 = 6 and 100 samples. We also report the
results for the Mirror Langevin Dynamic (MLD) algorithm, which provide iid samples, which are
thus less ordered. We plot the evolution of the KL over iterations on Figure 8b (where the entropy is
estimated using the Kozachenko-Leonenko estimator [42]).

The KDE used here will not scale well with the dimension, however, different methods have been
recently propose to overcome this issue, such as using projection on lower dimensional subspaces
[127], or using neural networks to learn ratio density estimators [6, 49, 128].

H
Proofs

H.1
Proof of Proposition 1

Let µ ∈P2(Rd), S, T ∈D( ˜Fµ), ϵ > 0. Since F is Wasserstein differentiable at S#µ, applying
Proposition 9 at S#µ with ν =
 
S + ϵ(T −S)


#µ and γ =
 
S, S + ϵ(T −S)


#µ ∈Π(S#µ, ν), we

44


---Page Break---
obtain,
˜Fµ
 
S + ϵ(T −S)

= F
 
(S + ϵ(T −S))#µ


= F(S#µ) +
Z
⟨∇W2F(S#µ)(x), y −x⟩dγ(x, y)

+ o

 sZ
∥x −y∥2
2 dγ(x, y)

!

= ˜Fµ(S) + ϵ
Z
⟨∇W2F(S#µ)
 
S(x)

, T(x) −S(x)⟩dµ(x)

+ o

 

ϵ

sZ
∥T(x) −S(x)∥2
2 dµ(x)

!

= ˜Fµ(S) + ϵ⟨∇W2F(S#µ) ◦S, T −S⟩L2(µ) + ϵo(∥T −S∥L2(µ)).
(134)

Thus, δ ˜Fµ(S, T −S) = ⟨∇W2F(S#µ) ◦S, T −S⟩L2(µ). Note that in the third equality we used that
∇W2F(µ) ∈L2(µ).

H.2
Proof of Proposition 2

Let µ, ρ ∈P2,ac(Rd) and ν ∈P2(Rd). Define Tµ,ν
ϕµ
= argminT#µ=ν dϕµ(T, Id), Uρ,ν
ϕρ
=
argminU#ρ=ν dϕρ(U, Id) and let S ∈L2(µ) such that S#µ = ρ. Then, noticing that γ =
(Tµ,ν
ϕµ , S)#µ ∈Π(ν, ρ), we have

dϕµ(Tµ,ν
ϕµ , S) = ϕ
 
(Tµ,ν
ϕµ )#µ

−ϕ(S#ν) −
Z
⟨∇W2ϕ(S#µ)(y), x −y⟩d(Tµ,ν
ϕµ , S)#µ(x, y)

= ϕ(ν) −ϕ(ρ) −
Z
⟨∇W2ϕ(ρ)(y), x −y⟩dγ(x, y)

≥Wϕ(ν, ρ) = dϕρ(Uρ,ν
ϕρ , Id).
(135)

In the last line, we used Proposition 15, i.e. that the optimal coupling is of the form (Uρ,ν
ϕρ , Id)#ρ.

H.3
Proof of Proposition 3

Let Tk+1 = argminT∈L2(µk) τ⟨∇W2F(µk), T−Id⟩L2(µk)+dϕµk (T, Id). Applying the three-point
inequality (Lemma 29) with ψ(T) = τ⟨∇W2F(µk), T −Id⟩L2(µk) which is convex, T0 = Id and
T∗= Tk+1, we get for all T ∈L2(µk),

τ⟨∇W2F(µk), T −Id⟩L2(µk) + dϕµk (T, Id)

≥τ⟨∇W2F(µk), Tk+1 −Id⟩L2(µk) + dϕµk (Tk+1, Id) + dϕµk (T, Tk+1),
(136)

which is equivalent to

⟨∇W2F(µk), Tk+1 −Id⟩L2(µk) + 1

τ dϕµk (Tk+1, Id)

≤⟨∇W2F(µk), T −Id⟩L2(µk) + 1

τ dϕµk (T, Id) −1

τ dϕµk (T, Tk+1).
(137)

By the β-smoothness of ˜Fµk relative to ϕµk, we also have

d ˜
Fµk (Tk+1, Id) = ˜Fµk(Tk+1) −˜Fµk(Id) −⟨∇W2F(µk), Tk+1 −Id⟩L2(µk) ≤βdϕµk (Tk+1, Id)

⇐⇒
˜Fµk(Tk+1) ≤˜Fµk(Id) + ⟨∇W2F(µk), Tk+1 −Id⟩L2(µk) + βdϕµk (Tk+1, Id).
(138)

Moreover, since β ≤1

τ , this inequality implies (by non-negativity of dϕµk ),

˜Fµk(Tk+1) ≤˜Fµk(Id) + ⟨∇W2F(µk), Tk+1 −Id⟩L2(µk) + 1

τ dϕµk (Tk+1, Id).
(139)

45


---Page Break---
Then, using the inequality (137), we obtain for all T ∈L2(µk),

˜Fµk(Tk+1) ≤˜Fµk(Id) + ⟨∇W2F(µk), T −Id⟩L2(µk) + 1

τ dϕµk (T, Id) −1

τ dϕµk (T, Tk+1).
(140)
Observing that ˜Fµk(Tk+1) = F(µk+1) and ˜Fµk(Id) = F(µk), we get

F(µk+1) ≤F(µk) + ⟨∇W2F(µk), T −Id⟩L2(µk) + 1

τ dϕµk (T, Id) −1

τ dϕµk (T, Tk+1).
(141)

Finally, setting T = Id, we obtain the result:

F(µk+1) ≤F(µk) −1

τ dϕµk (Id, Tk+1).
(142)

H.4
Proof of Proposition 4

Let ν ∈P2(Rd), and T = argminT,T#µk=ν dϕµk (T, Id). From the relative convexity hypothesis,
we have

d ˜
Fµk (T, Id) ≥αdϕµk (T, Id)

⇐⇒
˜Fµk(T) −˜Fµk(Id) −⟨∇W2F(µk), T −Id⟩L2(µk) ≥αdϕµk (T, Id)

⇐⇒
˜Fµk(T) −αdϕµk (T, Id) ≥˜Fµk(Id) + ⟨∇W2F(µk), T −Id⟩L2(µk)
⇐⇒F(ν) −αdϕµk (T, Id) ≥F(µk) + ⟨∇W2F(µk), T −Id⟩L2(µk).

(143)

Plugging this into (140), we get

F(µk+1) ≤F(ν) + 1

τ
 
dϕµk (T, Id) −dϕµk (T, Tk+1)

−αdϕµk (T, Id).
(144)

Then, by definition of T, note that dϕµk (T, Id) = Wϕ(ν, µk), and by Assumption 1, we have
dϕµk (T, Tk+1) ≥Wϕ(ν, µk+1), since T#µk = ν and (Tk+1)#µk = µk+1. Thus,

F(µk+1) −F(ν) ≤
1

τ −α

Wϕ(ν, µk) −1

τ Wϕ(ν, µk+1).
(145)

Observing that F(µk) ≤F(µℓ) for all ℓ≤k (by Proposition 3 and non-negativity of dϕ for ϕ
convex) and that Wϕ(ν, µ) ≥0, we can apply Lemma 30 with f = F, c = F(ν) and g = Wϕ(ν, ·),
and we obtain

∀k ≥1, F(µk) −F(ν) ≤
α

1
τ
1
τ −α

k
−1
Wϕ(ν, µ0) ≤

1
τ −α

k
Wϕ(ν, µ0).
(146)

For the second result, from (145), we get for ν = µ∗the minimizer of F, since F(µk+1)−F(µ∗) ≥0,

Wϕ(µ∗, µk+1) ≤(1 −ατ) Wϕ(µ∗, µk) ≤(1 −ατ)k+1 Wϕ(µ∗, µ0).
(147)

46


---Page Break---
H.5
Proof of Proposition 5

Let k ≥0, by the definition of dϕh∗
µk and the hypothesis dϕh∗
µk
 
∇W2F(µk+1)◦Tk+1, ∇W2F(µk)

≤
βd ˜
Fµk (Id, Tk+1) , we have

ϕh∗
µk+1
 
∇W2F(µk+1)

= ϕh∗
µk
 
∇W2F(µk)


+ ⟨∇h∗◦∇W2F(µk), ∇W2F(µk+1) ◦Tk+1 −∇W2F(µk)⟩L2(µk)
+ dϕh∗
µk
 
∇W2F((Tk+1)#µk) ◦Tk+1, ∇W2F(µk)


≤ϕh∗
µk
 
∇W2F(µk)


+ ⟨∇h∗◦∇W2F(µk), ∇W2F(µk+1) ◦Tk+1 −∇W2F(µk)⟩L2(µk)
+ βd ˜
Fµk (Id, Tk+1)

≤ϕh∗
µk
 
∇W2F(µk)


+ ⟨∇h∗◦∇W2F(µk), ∇W2F(µk+1) ◦Tk+1 −∇W2F(µk)⟩L2(µk)

+ 1

τ d ˜
Fµk (Id, Tk+1),
(148)

where we used in the last line that τ ≤1

β and the non-negativity of the Bregman divergence since F
is convex along t 7→
 
(1 −t)Tk+1 + tId


#µk and thus by Proposition 13, d ˜
Fµk (Id, Tk+1) ≥0.

Let T ∈L2(µk). Then, using the three-point identity (Lemma 28) (with S = Id, U = T and
T = Tk+1), and remembering that Tk+1 = Id −τ∇h∗◦∇W2F(µk), we get

d ˜
Fµk (Id, Tk+1) = d ˜
Fµk (Id, T) −d ˜
Fµk (Tk+1, T)

−⟨∇W2F
 
(Tk+1)#µk

◦Tk+1, Id −Tk+1⟩L2(µk)
+ ⟨∇W2F(T#µk) ◦T, Id −Tk+1⟩L2(µk)
= d ˜
Fµk (Id, T) −d ˜
Fµk (Tk+1, T)

+ ⟨∇W2F(T#µk) ◦T −∇W2F(µk+1) ◦Tk+1, Id −Tk+1⟩L2(µk)
= d ˜
Fµk (Id, T) −d ˜
Fµk (Tk+1, T)

+ τ⟨∇W2F(T#µk) ◦T −∇W2F(µk+1) ◦Tk+1, ∇h∗◦∇W2F(µk)⟩L2(µk).
(149)
This is equivalent to

⟨∇h∗◦∇W2F(µk), ∇W2F(µk+1) ◦Tk+1 −∇W2F(µk)⟩L2(µk) + 1

τ d ˜
Fµk (Id, Tk+1)

= 1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T)

+ ⟨∇W2F(T#µk) ◦T −∇W2F(µk), ∇h∗◦∇W2F(µk)⟩L2(µk).
(150)

Then, using the definition of dϕh∗
µk
 
∇W2F(T#µk) ◦T, ∇W2F(µk)

, we obtain

⟨∇h∗◦∇W2F(µk), ∇W2F(µk+1) ◦Tk+1 −∇W2F(µk)⟩L2(µk) + 1

τ d ˜
Fµk (Id, Tk+1)

= 1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T)

−dϕh∗
µk
 
∇W2F(T#µk) ◦T, ∇W2F(µk)

+ ϕh∗
µk
 
∇W2F(T#µk) ◦T

−ϕh∗
µk
 
∇W2F(µk)

.
(151)

Plugging this into (148), we get

ϕh∗
µk+1
 
∇W2F(µk+1)

≤ϕh∗
µk
 
∇W2F(T#µk) ◦T

+ 1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T)

−dϕh∗
µk
 
∇W2F(T#µk) ◦T, ∇W2F(µk)

.
(152)

47


---Page Break---
For T = Id, we get

ϕh∗
µk+1
 
∇W2F(µk+1)

≤ϕh∗
µk
 
∇W2F(µk)

−1

τ d ˜
Fµk (Tk+1, Id).
(153)

H.6
Proof of Proposition 6

Let µ∗∈P2(Rd) be the minimizer of F, k ≥0 and T = argminT∈L2(µk),T#µk=µ∗d ˜
Fµk (Id, T).
First, observe that since µ∗is the minimizer of F, then ∇W2F(µ∗) = 0 (see e.g. [74, Theorem
3.1]), and thus ϕh∗
µk(0) = h∗(0). Moreover, it induces that d ˜
Fµk (Id, T) = F(µk) −F(µ∗) and
d ˜
Fµk (Tk+1, T) = F(µk+1) −F(µ∗).

Therefore, using (152) and the hypothesis αd ˜
Fµk (Id, T) ≤dϕh∗
µk
 
0, ∇W2F(µk)

, we get

ϕh∗
µk+1
 
∇W2F(µk+1)

−h∗(0) ≤1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T) −dϕh∗
µk
 
0, ∇W2F(µk)


≤1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T) −αd ˜
Fµk (Id, T)

=
1

τ −α

d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T)

=
1

τ −α
  
F(µk) −F(µ∗)

−1

τ
 
F(µk+1) −F(µ∗)

.

(154)

Then, applying Lemma 30 with f = ϕh∗
·
◦∇W2F (which satisfies ϕh∗
µk+1
 
∇W2F(µk+1)

≤
ϕh∗
µk
 
∇W2F(µk)

by Proposition 5), c = h∗(0) and g = F(·) −F(µ∗) ≥0, we get

ϕh∗
µk
 
∇W2F(µk)

−h∗(0) ≤
α

1
τ
1
τ −α

k
−1

 
F(µ0) −F(µ∗)

≤

1
τ −α

k
 
F(µ0) −F(µ∗)

. (155)

Concerning the convergence of F(µk), if α > 0 and h∗attains its minimum in 0, then necessarily
ϕh∗
µ (T) ≥h∗(0) for all µ ∈P2(Rd) and T ∈L2(µ). Thus, using (152), we get

0 ≤ϕh∗
µk+1
 
∇W2F(µk+1)

−h∗(0) ≤1

τ d ˜
Fµk (Id, T) −1

τ d ˜
Fµk (Tk+1, T) −dϕh∗
µk
 
0, ∇W2F(µk)


≤1

τ
 
F(µk) −F(µ∗)

−1

τ
 
F(µk+1) −F(µ∗)


−αd ˜
Fµk (Id, T)

=
1

τ −α
  
F(µk) −F(µ∗)

−1

τ
 
F(µk+1) −F(µ∗)

.

(156)

Thus, for all k ≥0,

F(µk+1) −F(µ∗) = (1 −τα)
 
F(µk) −F(µ∗)


≤(1 −τα)k+1  
F(µ0) −F(µ∗)

.
(157)

H.7
Proof of Proposition 7

Let µ ∈P2(Rd). Since ˜F∗
µ is Gâteaux differentiable, we can define its Bregman divergence.

For the first point, ϕh∗is β-smooth relative to F∗
µ along t 7→
 
(1 −t)∇W2F(µ) + t∇W2F(T#µ) ◦
T


#µ. Thus, by applying Definition 3 for s = 1 and t = 0, we have

dϕh∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µ)

≤βd ˜
F∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µ)

.
(158)

48


---Page Break---
Using Lemma 19, we finally obtain

dϕh∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µk)

≤βd ˜
F∗
µ
 
∇W2F(T#µ) ◦T, ∇W2F(µ)


= βd ˜
Fµ
 
Id, T),
(159)

which is the desired inequality.

The second point follows similarly.

H.8
Proof of Lemma 11

Let us define ˜Gµ : L2(µ) × Rd →Rd as for all T ∈L2(µ), x ∈Rd,

˜Gµ(T, x) = ∇W2F(T#µ)(x) =






∂
∂x1
δF

δµ (T#µ)(x)
...
∂
∂xd
δF

δµ (T#µ)(x)




=






˜G1
µ(T, x)
...
˜Gd
µ(T, x)




,
(160)

with for all i, ˜Gi
µ : L2(µ) × Rd →R, ˜Gi
µ(T, x) =
∂
∂xi
δF

δµ (T#µ)(x). Using the chain rule, for all
x ∈Rd,

d ˜Gi
µ
ds
 
Ts, Ts(x)

=

∇1 ˜Gi
µ
 
Ts, Ts(x)

, dTs

ds



L2(µ)
+

∇2 ˜Gi
µ
 
Ts, Ts(x)

, dTs

ds (x)

.

(161)
On one hand, we have ∇2 ˜Gi
µ
 
Ts, Ts(x)

= ∇∂

∂xi
δF

δµ
 
(Ts)#µ
 
Ts(x)

. On the other hand, let
us compute ∇1 ˜Gi
µ(T, x). First, we define the shorthands ˜gx,i
µ (T) = ˜Gi
µ(T, x) =
∂
∂xi
δF

δµ (T#µ)(x)
and gx,i(ν) =
∂
∂xi
δF

δµ (ν)(x). Since ˜gx,i
µ (T) = gx,i(T#µ), applying Proposition 1, we know that

∇1 ˜Gµ(T, x) = ∇˜gx,i
µ (T) = ∇W2gx,i(T#µ) ◦T.

Now, let us compute ∇W2gx,i(ν) = ∇δgx,i

δµ (ν). Let χ be such that
R
dχ = 0, then using the

hypothesis that
δ
δµ∇δF

δµ = ∇δ2F

δµ2 and the definition of gx,i,
Z δgx,i

δµ (ν) dχ =
Z
∂
∂xi

δ2F

δµ2 (ν)(x, y) dχ(y).
(162)

Thus, ∇W2gx,i(ν) = ∇y
∂
∂xi
δ2F

δµ2 (ν)(x, y).

Putting everything together, we obtain

d ˜Gi
µ
ds
 
Ts, Ts(x)

=

∇y
∂
∂xi

δ2F

δµ2
 
(Ts)#µ
 
Ts(x), Ts(·)

, dTs

ds



L2(µ)

+

∇∂

∂xi

δF

δµ
 
(Ts)#µ
 
Ts(x)

, dTs

ds (x)


=
Z 
∇y
∂
∂xi

δ2F

δµ2
 
(Ts)#µ
 
Ts(x), Ts(y)

, dTs

ds (y)

dµ(y)

+

∇∂

∂xi

δF

δµ
 
(Ts)#µ
 
Ts(x)

, dTs

ds (x)

,

(163)

and thus

d
ds
˜Gµ
 
Ts, Ts(x)

=
Z
∇y∇x
δ2F

δµ2
 
(Ts)#µ
 
Ts(x), Ts(y)
dTs

ds (y) dµ(y)

+ ∇2 δF

δµ
 
(Ts)#µ
 
Ts(x)
dTs

ds (x).
(164)

49


---Page Break---
H.9
Proof of Proposition 12

First, recall that by using the chain rule and Proposition 1, d

dtF(µt) = ⟨∇W2F(µt) ◦Tt, dTt

dt ⟩L2(µ).
Thus, since d2Tt

dt2 = 0,

d2

dt2 F(µt) = d

dt


∇W2F(µt) ◦Tt, dTt

dt



L2(µ)

=
 d

dt (∇W2F(µt) ◦Tt) , dTt

dt



L2(µ)
.
(165)

By Lemma 11,

d2

dt2 F(µt) =
ZZ 
∇y∇x
δ2F

δµ2
 
(Tt)#µ
 
Tt(x), Tt(y)
dTt

dt (y), dTt

dt (x)

dµ(y)dµ(x)

+
Z 
∇2 δF

δµ
 
(Tt)#µ
 
Tt(x)
dTt

dt (x), dTt

dt (x)

dµ(x)

=
ZZ 
∇y∇x
δ2F

δµ2
 
(Tt)#µ
 
Tt(x), Tt(y)

v(y), v(x)

dµ(y)dµ(x)

+
Z 
∇2 δF

δµ
 
(Tt)#µ
 
Tt(x)

v(x), v(x)

dµ(x)

=
Z D Z
∇y∇x
δ2F

δµ2
 
(Tt)#µ
 
Tt(x), Tt(y)

v(y) dµ(y)

+ ∇2 δF

δµ
 
(Tt)#µ
 
Tt(x)

v(x), v(x)
E
dµ(x).

(166)

H.10
Proof of Proposition 13

1. (c1) =⇒(c2). Let t > 0, t1, t2 ∈[0, 1],

F(˜µt1→t2
t
) ≤(1 −t)F
 
(Tt1)#µ

+ tF
 
(Tt2)#µ


⇐⇒F
 
˜µt1→t2
t

−F
 
(Tt1)#µ


t
≤F
 
(Tt2)#µ

−F
 
(Tt1)#µ

.
(167)

Passing to the limit t →0 and using Proposition 1, we get ⟨∇W2F
 
(Tt1)#µ

◦Tt1, Tt2 −
Tt1⟩L2(µ) ≤F
 
(Tt2)#µ

−F
 
(Tt1)#µ

.

2. (c2) =⇒(c3). Let t1, t2 ∈[0, 1], then by hypothesis,
⟨∇W2F
 
(Tt1)#µ

◦Tt1, Tt2 −Tt1⟩L2(µ) ≤F
 
(Tt2)#µ

−F
 
(Tt1)#µ


⟨∇W2F
 
(Tt2)#µ

◦Tt2, Tt1 −Tt2⟩L2(µ) ≤F
 
(Tt1)#µ

−F
 
(Tt2)#µ

.
(168)

Summing the two inequalities, we get

⟨∇W2F
 
(Tt2)#µ

◦Tt2 −∇W2F
 
(Tt1)#µ

◦Tt1, Tt2 −Tt1⟩L2(µ) ≥0.
(169)

3. (c3) =⇒(c4). Let t1, t2 ∈[0, 1]. First, we have,
Z 1

0

d2

dt2 F(˜µt1→t2
t
) dt = d

dtF(˜µt1→t2
t
)

t=1 −d

dtF(˜µt1→t2
t
)

t=0
= ⟨∇W2F
 
(Tt2)#µ

◦Tt2 −∇W2F
 
(Tt1)#µ

◦Tt1,

Tt2 −Tt1⟩L2(µ)
≥0.

(170)

Let ϵ ∈(0, 1) and define t 7→νϵ
t = ˜µt1→1
ϵt
the interpolation curve between (Tt1)#µ
and
 
Tt1 + ϵ(T −Tt1)


#µ. Then, noting that Tt1 + ϵ(T −Tt1) = Tt1+ϵ(1−t1), so

50


---Page Break---
νϵ
t = ˜µt1→1
ϵt
= ˜µt1→t1+ϵ(1−t1)
t
and we have that
Z 1

0

d2

dt2 F(νϵ
t) dt ≥0.
(171)

Moreover, by continuity,
d2
dt2 F(νϵ
t) −−−→
ϵ→0
d2
dt2 F
 
(Tt1)#µ

=
d2
dt2 F(µt1). Then, since

t 7→
d2
dt2 F(νϵ
t) is continuous on [0, 1], it is bounded, and we can apply the dominated
convergence theorem. This implies that for all t1 ∈[0, 1],

Hessµt1F = d2

dt2 F(µt)

t=t1 = lim
ϵ→0

Z 1

0

d2

dt2 F(νϵ
t) dt ≥0.
(172)

4. (c4) =⇒(c1). Let t1, t2 ∈[0, 1] and φ(t) = F(˜µt1→t2
t
) for all t ∈[0, 1]. From [125,
Equation 16.5],

∀t ∈[0, 1], φ(t) = (1 −t)φ(0) + tφ(1) −
Z 1

0

d2

dt2 φ(s)G(s, t) ds,
(173)

where G is the Green function defined as G(s, t) = s(1 −t)1{s≤t} + t(1 −s)1{t≤s} ≥0
[125, Equation 16.6]. Then, d2

dt2 F(µt) ≥0 implies that
R 1
0
d2
dt2 φ(s)G(s, t) ds ≥0, and thus

φ(t) = F(˜µt1→t2
t
) ≤(1−t)φ(0)+tφ(1) = (1−t)F
 
(Tt1)#µ

+tF
 
(Tt2)#µ

. (174)

H.11
Proof of Proposition 24

Let J(T) = dϕµk (T, Id) + τ
 
∇W2G(µk), T −Id⟩L2(µk) + H(T#µk)

. Taking the first variation,
we get

∇J(˜Tk+1) = ∇ϕµk(˜Tk+1) −∇ϕµk(Id) + τ
 
∇W2G(µk) + ∇W2H
 
(˜Tk+1)#µk

◦˜Tk+1


= ∇ϕµk(˜Tk+1) + τ∇W2H
 
(˜Tk+1)#µk

◦˜Tk+1 −
 
∇ϕµk(Id) −τ∇W2G(µk)


= ∇ϕµk(˜Tk+1) + τ∇W2H
 
(˜Tk+1)#µk

◦˜Tk+1 −∇ϕµk(Sk+1).
(175)

Thus,
∇J(˜Tk+1) = 0 ⇐⇒˜Tk+1 ∈argmin
T∈L2(µk)
dϕµk (T, Sk+1) + τH(T#µk).
(176)

Now, we aim at showing that ˜Tk+1 = Tk+1 ◦Sk+1 or

min
T∈L2(µk) dϕµk (T, Sk+1) + τH(T#µk) =
min
T∈L2(νk+1) dϕνk+1(T, Id) + τH(T#νk+1).
(177)

First, by the change of variable formula, since ϕµ is pushforward compatible, observe that for
T ∈L2(νk+1), dϕνk+1(T, Id) + τH(T#νk+1) = dϕµk (T ◦Sk+1, Sk+1) + τH
 
(T ◦Sk+1)#µk

.

Since {T ◦Sk+1 | T ∈L2(νk+1)} ⊂L2(µk), we have

min
T∈L2(νk+1) dϕνk+1(T, Id) + τH(T#νk+1) ≥
min
T∈L2(µk) dϕµk (T, Sk+1) + τH(T#µk).
(178)

By assumption, νk+1 ∈P2,ac(Rd). Thus, applying Proposition 15, there exists Tνk+1,µk+1
ϕνk+1
such

that (Tνk+1,µk+1
ϕνk+1
)#νk+1 = µk+1 and Tνk+1,µk+1
ϕνk+1
= argminT,T#νk+1=µk+1 dϕνk+1(T, Id), and thus

dϕνk+1(Tνk+1,µk+1
ϕνk+1
, Id) = Wϕ(µk+1, νk+1).

By contradiction, we suppose that

min
T∈L2(νk+1) dϕνk+1(T, Id) + τH(T#νk+1) > dϕµk (˜Tk+1, Sk+1) + τH
 
(˜Tk+1)#µk

.
(179)

On one hand, we have (Tνk+1,µk+1
ϕνk+1
◦Sk+1)#µk = (Tνk+1,µk+1
ϕνk+1
)#νk+1 = µk+1, and there-

fore H
 
(Tνk+1,µk+1
ϕνk+1
◦Sk+1)#µk

=
H(µk+1)
=
H
 
(˜Tk+1)#µk

.
On the other hand,

(˜Tk+1, Sk+1)#µk ∈Π(µk+1, νk+1), and thus

dϕµk (˜Tk+1, Sk+1) ≥Wϕ(µk+1, νk+1) = dϕνk+1(Tνk+1,µk+1
ϕνk+1
, Id).
(180)

51


---Page Break---
Thus,
min
T∈L2(νk+1) dϕνk+1(T, Id) + τH(T#νk+1) > dϕµk (˜Tk+1, Sk+1) + τH
 
(˜Tk+1)#µk

(181)

≥dϕνk+1(Tνk+1,µk+1
ϕνk+1
, Id) + τH
 
(Tνk+1,µk+1
ϕνk+1
)#νk+1

.

But Tνk+1,µk+1
ϕνk+1
∈L2(νk+1), so this is a contradiction. So, we can conclude that the two schemes are

equivalent, and moreover, ˜Tk+1 = Tνk+1,µk+1
ϕνk+1
◦Sk+1.

H.12
Proof of Proposition 25

Let ψ(T) = τ
 
⟨∇W2G(µk), T −Id⟩L2(µk) + H(T#µk)

. Since ˜Hµk is convex on L2(µk), ψ is
convex, and we can apply the three-point inequality (Lemma 29) and for all T ∈L2(µk),

τ
 
H(T#µk) + ⟨∇W2G(µk), T −Id⟩L2(µk)

+ dϕµk (T, Id)

≥τ
 
H(µk+1) + ⟨∇W2G(µk), ˜Tk+1 −Id⟩L2(µk)

+ dϕµk (˜Tk+1, Id) + dϕµk (T, ˜Tk+1),
(182)

which is equivalent to

H(µk+1) + ⟨∇W2G(µk), ˜Tk+1 −Id⟩L2(µk) + 1

τ dϕµ(˜Tk+1, Id)

≤H(T#µk) + ⟨∇W2G(µk), T −Id⟩L2(µk) + 1

τ dϕµk (T, Id) −1

τ dϕµk (T, ˜Tk+1).
(183)

Since ˜Gµk is β-smooth relatively to ϕµk along t 7→
 
(1 −t)Id + t˜Tk+1


#µk, and τ ≤1

β , we also
have
G(µk+1) ≤G(µk) + ⟨∇W2G(µk), ˜Tk+1 −Id⟩L2(µk) + βdϕµk (˜Tk+1, Id)

≤G(µk) + ⟨∇W2G(µk), ˜Tk+1 −Id⟩L2(µk) + 1

τ dϕµk (˜Tk+1, Id).
(184)

Thus, applying first the smoothness of G and then the three-point inequality, we get for all T ∈
L2(µk),

H(µk+1) + G(µk+1) ≤H(µk+1) + G(µk) + ⟨∇W2G(µk), ˜Tk+1 −Id⟩L2(µk) + 1

τ dϕµk (˜Tk+1, Id)

≤H(T#µk) + G(µk) + ⟨∇W2G(µk), T −Id⟩L2(µk) + 1

τ dϕµk (T, Id)

−1

τ dϕµk (T, ˜Tk+1).
(185)

Now, let ν ∈P2(Rd) and Tµk,ν
ϕµk = argminT,T#µk=ν dϕµk (T, Id), and suppose that ˜Gµk is α-convex

relative to ϕµk along t 7→
 
(1 −t)Id + tT µk,ν
ϕµk


#µk. Thus,

d ˜Gµk (Tµk,ν
ϕµk , Id) ≥αdϕµk (Tµk,ν
ϕµk , Id)

⇐⇒G(ν) −αdϕµk (Tµk,ν
ϕµk , Id) ≥G(µk) + ⟨∇W2G(µk), Tµk,ν
ϕµk −Id⟩L2(µk).
(186)

Plugging this into (185), we get

F(µk+1) ≤H(ν)+G(ν)−αdϕµk (Tµk,ν
ϕµk , Id)+ 1

τ dϕµk (Tµk,ν
ϕµk , Id)−1

τ dϕµk (Tµk,ν
ϕµk , ˜Tk+1). (187)

Now, note that dϕµk (Tµk,ν
ϕµk , Id) = Wϕ(ν, µk) and by Assumption 1, dϕµk (Tµk,ν
ϕµk , ˜Tk+1) ≥
Wϕ(ν, µk+1). Thus,

F(µk+1) −F(ν) ≤
1

τ −α

Wϕ(ν, µk) −1

τ Wϕ(ν, µk+1).
(188)

Using T = Id in (185), we observe that F(µk) ≤F(µℓ) for all ℓ≤k. Moreover, Wϕ(ν, µk) ≥0.
Thus, applying Lemma 30 with f = F, c = F(ν) and g = Wϕ(ν, ·), we obtain

∀k ≥1, F(µk) −F(ν) ≤
α

1
τ
1
τ −α

k
−1
Wϕ(ν, µ0) ≤

1
τ −α

k
Wϕ(ν, µ0).
(189)

52


---Page Break---
H.13
Proof of Lemma 26

First, ∇V ∗is bijective. Thus, we only need to show that h = ∇V −τ∇U is injective. Take
u = V −τU.

Since U is β-smooth relative to V , we have for all x, y,

U(x) ≤U(y) + ⟨∇U(y), x −y⟩+ βdV (x, y),
(190)

which is equivalent to

−U(y) ≤−U(x) + ⟨∇U(y), x −y⟩+ βdV (x, y).
(191)

Moreover, by definition of dV ,

V (y) = V (x) −⟨∇V (y), x −y⟩−dV (x, y).
(192)

Summing the two inequalities, we get

V (y) −τU(y) ≤V (x) −⟨∇V (y), x −y⟩−dV (x, y) −τU(x) + τ⟨∇U(y), x −y⟩
+ τβdV (x, y)
= V (x) −τU(x) −⟨∇V (y) −τ∇U(y), x −y⟩−(1 −τβ)dV (x, y).
(193)

This is equivalent to

u(y) ≤u(x) −⟨∇u(y), x −y⟩−(1 −τβ)dV (x, y),
(194)

and thus with u being (1 −τβ)-convex relative to V (for τβ ≤1). For τβ < 1, it is equivalent to
u −(1 −τβ)V convex, i.e. ⟨∇u(x) −∇u(y), x −y⟩≥(1 −τβ)⟨∇V (x) −∇V (y), x −y⟩≥0.
Since V is strictly convex, ∇u is injective.

Moreover, | det ∇T| = | det
 
∇2V ∗◦(∇V −τ∇U)

det ∇2u| > 0 because on one hand u is
(1 −βτ)-convex relative to V which is strictly convex, and on the other hand, V ∗is also strictly
convex.

To conclude, applying [5, Lemma 5.5.3], T#µ is absolutely continuous with respect to the Lebesgue
measure.

H.14
Proof of Proposition 27

On one hand, H is 1-smooth relative to H, thus we only need to show that µ 7→
R
V dµ is smooth
relative to H. Using Proposition 13, we need to show that

d2

dt2 V(µt) = 1

2

Z
(Tk+1 −Id)T ∇2V (Tk+1 −Id) dµk ≤β d2

dt2 H(µt).
(195)

Recall from (61) that Tk+1(x) =
 
(1−τ)Σk+1Σ−1
k +τΣk+1Σ−1
x+cst, thus ∇Tk+1 is a constant.
Using the computations of [43, Appendix B.2],

d2

dt2 H(µt) = ⟨[∇Tt]−2, ∇Tk+1 −Id⟩.
(196)

Assuming (1 −τ)Σk+1Σ−1
k
+ τΣk+1Σ−1 ⪰0, Tk+1 is the gradient of a convex function and µt is
a Wasserstein geodesic. Thus, by [43],

d2

dt2 H(µt) ≥
1
∥Σµt∥op
∥Tk+1 −Id∥2
L2(µk).
(197)

Moreover, by [33, Lemma 10], µ 7→∥Σµ∥op is convex along generalized geodesics, and thus
Σµt ⪯λId and ∥Σµt∥op ≤λ [43]. Hence, noting σmax(M) the largest eigenvalue of some matrix
M,

d2

dt2 H(µt) ≥1

λ∥Tk+1 −Id∥2
L2(µk) ≥
1
λσmax(∇2V )

Z
(Tk+1 −Id)T ∇2V (Tk+1 −Id)dµk

=
2
λσmax(∇2V )
d2

dt2 V(µt).
(198)

53


---Page Break---
From this inequality, we deduce that

λσmax(∇2V )

2
d ˜
Hµk (Tk+1, Id) = λσmax(∇2V )

2
 
H(µk+1) −H(µk)

−⟨∇W2H(µk), Tk+1 −Id⟩L2(µk)


= λσmax(∇2V )

2

Z
(1 −t) d2

dt2 H(µt) dt

≥
Z
d2

dt2 V(µt)(1 −t) dt

= d˜Vµk (Tk+1, Id).

(199)

So,
d ˜
Fµk (Tk+1, Id) = d˜Vµk (Tk+1, Id) + d ˜
Hµk (Tk+1, Id)

≤

1 + λσmax(∇2V )

2


d ˜
Hµk (Tk+1, Id).
(200)

I
Additional results

I.1
Three-point identity and inequality

In this Section, we derive results which are useful to show the convergence of mirror descent or
preconditioned schemes. Namely, we first derive the three-point identity which we use to show the
convergence of the preconditioned scheme in Proposition 5 as well as the three-point inequality,
which we use for the convergence of the mirror descent scheme in Proposition 3.
Lemma 28 (Three-Point Identity). Let ϕ : L2(µ) →R be Gâteaux differentiable. For all S, T, U ∈
L2(µ), we have

dϕ(S, U) = dϕ(S, T) + dϕ(T, U) + ⟨∇ϕ(T), S −T⟩L2(µ) −⟨∇ϕ(U), S −T⟩L2(µ).
(201)

Proof. Let S, T, U ∈L2(µ), then using the linearity of the Gâteaux differential,

dϕ(S, U) −dϕ(S, T) −dϕ(T, U) = ϕ(S) −ϕ(U) −⟨∇ϕ(U), S −U⟩L2(µ)
−ϕ(S) + ϕ(T) + ⟨∇ϕ(T), S −T⟩L2(µ)
−ϕ(T) + ϕ(U) + ⟨∇ϕ(U), T −U⟩L2(µ)
= −⟨∇ϕ(U), S −U⟩L2(µ) + ⟨∇ϕ(T), S −T⟩L2(µ)
+ ⟨∇ϕ(U), T −U⟩L2(µ)
= ⟨∇ϕ(T), S −T⟩L2(µ) −⟨∇ϕ(U), S −T⟩L2(µ).

(202)

Lemma 29 (Three-Point Inequality). Let µ ∈P2(Rd), T0 ∈L2(µ) and ϕµ : L2(µ) →R convex,
and Gâteaux differentiable. Let ψ : L2(µ) →R be convex, proper and lower semicontinuous.
Assume there exists T∗= argminT∈L2(µ) dϕµ(T, T0) + ψ(T). Then, for all T ∈L2(µ),

ψ(T) + dϕµ(T, T0) ≥ψ(T∗) + dϕµ(T∗, T0) + dϕµ(T, T∗).
(203)

Proof. Denote J(T) = dϕµ(T, T0) + ψ(T). Let T∗= argminT∈L2(µ) J(T), hence 0 ∈∂J(T∗).

Since ϕ and ψ are proper, convex and lower semicontinuous, and T 7→dϕµ(T, T0) is continuous
(since ϕµ is continuous), thus by [99, Theorem 3.30], ∂J(T∗) = ∂ψ(T∗) + ∂dϕµ(·, T0)(T∗).

Moreover, since ϕµ is differentiable, ∂dϕµ(·, T0)(T∗) = {∇Tdϕµ(T∗, T0)} = {∇ϕµ(T∗) −
∇ϕµ(T0)}, and thus ∇ϕµ(T0) −∇ϕµ(T∗) ∈∂ψ(T∗)

Finally, by definition of subgradients and by applying Lemma 28, we get for all T ∈L2(µ),

ψ(T) ≥ψ(T∗) −
 
⟨∇ϕµ(T∗), T −T∗⟩L2(µ) −⟨∇ϕµ(T0), T −T∗⟩L2(µ)


= ψ(T∗) −dϕµ(T, T0) + dϕµ(T, T∗) + dϕµ(T∗, T0).
(204)

54


---Page Break---
Remark 3. Actually we can restrict ψ to be convex along
 
(1 −t)T∗+ tT


#µ. In that case,
dψ(T, T∗) = ψ(T) −ψ(T∗) −⟨φ, T −T∗⟩L2(µ) ≥0 for φ ∈∂ψ(T∗) (by Proposition 13) and we
still have ∂ψ(T∗) + ∂dϕµ(·, T0)(T∗) ⊂∂J(T∗) (see [99, Theorem 3.30]) so that we can conclude.

I.2
Convergence lemma

We first provide a Lemma which follows from [88, Theorem 3.1], and which is useful for the proofs
of Propositions 4, 6 and 25.

Lemma 30. Let f : X →R, g : X →R+ and (xk)k∈N a sequence in X such that for all
k ≥1, f(xk) ≤f(xk−1). Assume that there exists β > α ≥0, c ∈R such that for all k ≥0,
f(xk+1) −c ≤(β −α)g(xk) −βg(xk+1), then

∀k ≥1, f(xk) −c ≤
α

β
β−α
k
−1
g(x0) ≤β −α

k
g(x0).
(205)

Proof. First, observe the f(xk) ≤f(xℓ) for all ℓ≤k. Thus, for all k ≥1,

k
X

ℓ=1


β
β −α

ℓ
·
 
f(xk) −c

≤

k
X

ℓ=1


β
β −α

ℓ 
f(xℓ) −c)


≤

k
X

ℓ=1


β
β −α

ℓ 
(β −α)g(xℓ−1) −βg(xℓ)


= β

k−1
X

ℓ=0


β
β −α

ℓ
g(xℓ) −β

k
X

ℓ=1


β
β −α

ℓ
g(xℓ)

= βg(x0) −β

β
β −α

k
g(xk)

≤βg(x0)
since g ≥0.

(206)

Now, note that
β
Pk
ℓ=1(
β
β−α)
ℓ=
α
(
β
β−α)
k−1 =
α
(1+
α
β−α)
k−1 ≤β−α

k
since

1 +
α
β−α
k
≥1 + k
α
β−α

(by convexity on R+ of x 7→(1 + x)k). Thus,

f(xk) −c ≤
β
Pk
ℓ=1

β
β−α
ℓg(x0) =
α

β
β−α
k
−1
g(x0) ≤β −α

k
g(x0).
(207)

I.3
Some properties of Bregman divergences

We provide in this Section additional results on the Bregman divergences introduced in Section 3.
First, we focus on ϕµ(T) =
R
V ◦T dµ. The following Lemma is akin to [80, Proposition 4] which
shows it only for OT maps.

Lemma 31. Let V : Rd →R convex and ϕµ(T) =
R
V ◦T dµ. Then,

∀T, S ∈L2(µ), dϕµ(T, S) =
Z
dV
 
T(x), S(x)

dµ(x).
(208)

Proof. Let T, S ∈L2(µ), then remembering that ∇W2V(µ) = ∇V , we have

dϕµ(T, S) = ϕµ(T) −ϕµ(S) −⟨∇V ◦S, T −S⟩L2(µ)

=
Z
V ◦T −V ◦S −⟨∇V ◦S, T −S⟩dµ
(209)

=
Z
dV
 
T(x), S(x)

dµ(x).

55


---Page Break---
Next, we focus on ϕµ(T) = 1

2
RR
W
 
T(x) −T(x′)

dµ(x)dµ(x′), and we generalize the result
from [80, Proposition 4].

Lemma 32. Let W : Rd →R even (W(x) = W(−x)), convex and differentiable. Let ϕµ(T) =
1
2
RR
W
 
T(x) −T(x′)

dµ(x)dµ(x′). Then,

∀T, S ∈L2(µ), dϕµ(T, S) = 1

2

ZZ
dW
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′).
(210)

Proof. Let T, S ∈L2(µ), remember that ∇W2W(µ) = ∇W ⋆µ, and thus ∇W2W(S#µ) ◦S =
(∇W ⋆S#µ) ◦S. Thus,

dϕµ(T, S) = ϕµ(T) −ϕµ(S) −⟨(∇W ⋆S#µ) ◦S, T −S⟩L2(µ)

= 1

2

ZZ
W
 
T(x) −T(x′)

dµ(x)dµ(x′) −1

2

ZZ
W
 
S(x) −S(x′)

dµ(x)dµ(x′)

−
Z
⟨(∇W ⋆S#µ)(S(x)), T(x) −S(x)⟩dµ(x).
(211)

Then, note that ∇W(−x) = −∇W(x) and thus the last term can be written as:

Z
⟨(∇W ⋆S#µ)(S(x)), T(x) −S(x)⟩dµ(x)

=
ZZ
⟨∇W
 
S(x) −S(x′)

, T(x) −S(x)⟩dµ(x)dµ(x′)

= 1

2

ZZ
⟨∇W
 
S(x) −S(x′)

, T(x) −S(x)⟩dµ(x)dµ(x′)

+ 1

2⟨∇W
 
S(x′) −S(x)

, T(y) −S(y)⟩dµ(x)dµ(x′)

= 1

2

ZZ
⟨∇W
 
S(x) −S(x′)

, T(x) −S(x)⟩dµ(x)dµ(x′)

−1

2⟨∇W
 
S(x) −S(x′)

, T(x′) −S(x′)⟩dµ(x)dµ(x′)

= 1

2

ZZ
⟨∇W
 
S(x) −S(x′)

, T(x) −T(x′) −
 
S(x) −S(x′)

⟩dµ(x)dµ(x′).

(212)

Finally, we get

dϕµ(T, S) = 1

2

ZZ 
W
 
T(x) −T(x′)

−W
 
S(x) −S(x′)

(213)

−⟨∇W
 
S(x) −S(x′)

, T(x) −T(x′) −
 
S(x) −S(x′)

⟩

dµ(x)dµ(x′)

= 1

2

ZZ
dW
 
T(x) −T(x′), S(x) −S(x′)

dµ(x)dµ(x′).

Now, we make the connection with the mirror map used by Deb et al. [41] and derive the related
Bregman divergence.

Lemma 33. Let ϕµ(T) = 1

2W2
2(T#µ, ρ) for µ, ρ ∈P2,ac(Rd). Then, for all T, S ∈L2(µ), such
that T#µ, S#µ ∈P2,ac(Rd),

dϕµ(T, S) = 1

2∥Tρ
T#µ ◦T−Tρ
S#µ ◦S−(T−S)∥2
L2(µ) +⟨Tρ
S#µ ◦S−S, Tρ
T#µ ◦T−Tρ
S#µ ◦S⟩L2(µ),
(214)
where Tρ
T#µ denotes the OT map between T#µ and ρ.

56


---Page Break---
Proof. Let T, S ∈L2(µ) such that T#µ, S#µ ∈P2,ac(Rd). Remember that ∇W2W2
2(·, ρ) =
Id −Tρ
· , then

dϕµ(T, S) = ϕµ(T) −ϕµ(S) −⟨∇W2ϕ(S#µ) ◦S, T −S⟩L2(µ)

= 1

2W2
2(T#µ, ρ) −1

2W2
2(S#µ, ρ) −⟨(Id −Tρ
S#µ) ◦S, T −S⟩L2(µ)

= 1

2∥Tρ
T#µ ◦T −T∥2
L2(µ) −1

2∥Tρ
S#µ ◦S −S∥2
L2(µ) + ⟨Tρ
S#µ ◦S −S, T −S⟩L2(µ)

= 1

2∥Tρ
T#µ ◦T −T∥2
L2(µ) −1

2∥Tρ
S#µ ◦S −S∥2
L2(µ)

+ ⟨Tρ
S#µ ◦S −S, T −Tρ
S#µ ◦S⟩L2(µ) + ⟨Tρ
S#µ ◦S −S, Tρ
S#µ ◦S −S⟩L2(µ)

= 1

2∥Tρ
T#µ ◦T −T∥2
L2(µ) + 1

2∥Tρ
S#µ ◦S −S∥2
L2(µ)

−⟨Tρ
S#µ ◦S −S, Tρ
S#µ ◦S −T⟩L2(µ)

= 1

2∥Tρ
T#µ ◦T −T∥2
L2(µ) + 1

2∥Tρ
S#µ ◦S −S∥2
L2(µ)

−⟨Tρ
S#µ ◦S −S, Tρ
S#µ ◦S −Tρ
T#µ ◦T⟩L2(µ)
(215)

−⟨Tρ
S#µ ◦S −S, Tρ
T#µ ◦T −T⟩L2(µ)

= 1

2∥Tρ
T#µ ◦T −Tρ
S#µ ◦S −(T −S)∥2
L2(µ)

+ ⟨Tρ
S#µ ◦S −S, Tρ
T#µ ◦T −Tρ
S#µ ◦S⟩L2(µ).

57


---Page Break---
J
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]

Justification: In the abstract and introduction, we claim that we adapt mirror descent and the
preconditioned gradient descent to the Wasserstein space, and that we provide guarantees of
convergence. These results are presented in Section 3 and Section 4 for respectively mirror
descent and preconditioned gradient descent. We also claim that we illustrate advantages of
such schemes on ill-conditioned optimization tasks and to minimize discrepancies in order
to align single-cells, which we do in Section 5.
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
Justification: On the two theoretical sections (Section 3 and Section 4), we state all the
hypotheses to obtain our convergence results. Then, in Section 5, we discuss some couples
of target functional F and Bregman potential/preconditioner ϕ for which we can verify these
assumptions. However, in general, verifying these assumptions is a hard problem, as stated
in the Conclusion, and we leave for future works these investigations.
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

58


---Page Break---
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The full set of assumptions are provided for each theoretical result, along with
a complete proof in Appendix.
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
Justification: The information to reproduce the main experimental results are described in
Section 5 and Appendix G.
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

59


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

Justification:
We provide a part of the code to reproduce the experiment on
Gaussians
and
on
interaction
functionals
in
supplementary
materials
and
in
https://github.com/clbonet/Mirror_and_Preconditioned_Gradient_
Descent_in_Wasserstein_Space.

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

Justification: All the training and test details for the experiments are provided in Appendix G.

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

60


---Page Break---
Justification: The experiment on Gaussians in Figure 2 is run over 20 randomly sampled
objective covariances, and the results are plotted with the standard deviation. For the
single-cell experiment of Figure 3, we reported the results with 3 different initialization
for each treatment, along their mean. For the mirror interaction experiment, the results are
deterministic.

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

Justification: We report in Appendix G the computer resources and approximate runtime for
each experiment. Namely, the Gaussian and mirror interaction experiments were done on a
personal laptop on CPU, and took only few hours to run. The single-cell experiment was
performed on GPU as the data are of higher dimension with about 4000 samples, and took a
few hundred of hours of computational time.

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

Justification: The research conducted in this paper is conform with the NeurIPS Code of
Ethics.

Guidelines:

61


---Page Break---
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [NA]

Justification: This work is mostly theoretical and not tied to particular applications.

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

Justification: [NA]

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

62


---Page Break---
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: The datasets used are properly cited.
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
Justification: [NA]
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
Justification: [NA]
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

63


---Page Break---
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: [NA]
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

64


---Page Break---
