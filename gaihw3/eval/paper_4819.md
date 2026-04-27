Optimal deep learning of holomorphic operators
between Banach spaces

Ben Adcock
Department of Mathematics
Simon Fraser University
Canada

Nick Dexter
Department of Scientific Computing
Florida State University
USA

Sebastian Moraga
Department of Mathematics
Simon Fraser University
Canada

Abstract

Operator learning problems arise in many key areas of scientific computing where
Partial Differential Equations (PDEs) are used to model physical systems. In such
scenarios, the operators map between Banach or Hilbert spaces. In this work, we
tackle the problem of learning operators between Banach spaces, in contrast to the
vast majority of past works considering only Hilbert spaces. We focus on learning
holomorphic operators – an important class of problems with many applications.
We combine arbitrary approximate encoders and decoders with standard feedfor-
ward Deep Neural Network (DNN) architectures – specifically, those with constant
width exceeding the depth – under standard ℓ2-loss minimization. We first identify
a family of DNNs such that the resulting Deep Learning (DL) procedure achieves
optimal generalization bounds for such operators. For standard fully-connected
architectures, we then show that there are uncountably many minimizers of the
training problem that yield equivalent optimal performance. The DNN architectures
we consider are ‘problem agnostic’, with width and depth only depending on the
amount of training data m and not on regularity assumptions of the target operator.
Next, we show that DL is optimal for this problem: no recovery procedure can
surpass these generalization bounds up to log terms. Finally, we present numerical
results demonstrating the practical performance on challenging problems including
the parametric diffusion, Navier-Stokes-Brinkman and Boussinesq PDEs.

1
Introduction

Operator learning is increasingly being investigated for problems arising in computational science
and engineering. These problems are often posed in terms of Partial Differential Equations (PDEs),
which can be viewed as operators mapping function spaces to function spaces. Depending on the
requirements for well-posedness of the PDE, both the input and solution spaces are often Hilbert, or,
more generally, Banach spaces. The aim of operator learning is to efficiently capture the dynamic
behavior of these operators using surrogate models, typically based on Deep Neural Networks
(DNNs). Specifically, we want to learn

F : X →Y,
X ∈X 7→F(X) ∈Y,
(1.1)

where Y is the PDE solution space, X represents the data supplied to the PDE, i.e., possibly multiple
functions describing initial and boundary conditions or forcing terms or, equivalently, a vector of
parameters defining such functions.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Let µ be a probability measure on X. Given noisy training data

{(Xi, F(Xi) + Ei)}m
i=1 ,
(1.2)

where X1, . . . , Xm ∼i.i.d. µ and Ei is noise, a typical operator learning methodology consists of
three objects: an approximate encoder EX : X →RdX , an approximate decoder DY : RdY →Y and
a DNN b
N : RdX →RdY. It then approximates F as

F ≈bF := DY ◦b
N ◦EX .
(1.3)

The encoder and decoder are either specified by the problem, learned separately from data, or
learned concurrently with b
N. The goal, as in all supervised learning problems, is to ensure good
generalization via the learned operator bF from as little training data m as possible.

1.1
Contributions

As noted in, e.g., [16, 66], the theory of deep operator learning is still in its infancy. We contribute
to this growth in the following ways. We consider learning classes of holomorphic operators
(Assumption 2.2), with arbitrary approximate encoders EX and decoders DY. As we explain in §2.3
(see also [52, §5.2], [53, §3.4] and [41]) these operators are relevant in many applications, notably
those involving parametric PDEs. The main contributions of this work are as follows.

1. We consider operators taking values in general Banach spaces. As noted, the vast majority of
existing work (with the notable exception of [16]) considers Hilbert spaces.
2. We consider standard feedforward DNN architectures (constant width, width exceeds depth) and
training procedures (ℓ2-loss minimization).
3. (Theorem 3.1) We construct a family of DNNs such that any approximate minimizer of the
corresponding training problem satisfies a generalization bound that is explicit in the various error
sources: namely, an approximation error, which decays algebraically in the amount of training
data m; encoding-decoding errors, which depend on the accuracy of the learned encoders and
decoders; an optimization error, and; a sampling error, which depends on the noise Ei in (1.2).
4. These DNN architectures are problem agnostic; they depend on m only. In particular, the
architectures are completely independent on the regularity assumptions of target operator.
5. (Theorem 3.2) We show that training problems based on any family of fully-connected DNNs
possess uncountably many minimizers that achieve the same generalization bounds.
6. (Theorems 3.1-3.2) We provide bounds in both the L2
µ- and L∞
µ -norms that hold in high probability,
rather than just expectation.
7. (Theorems 4.1-4.2) We show that the generalization bound is optimal with respect to m: no
learning procedure (not necessarily DL-based) can achieve better rates in m up to log terms.
8. Finally, we present a series of experiments demonstrating the efficacy of DL on challenging
problems such as the parametric diffusion, Navier-Stokes-Brinkman and Boussinesq PDEs, the
latter two of which involve operators whose codomains are Banach, as opposed to Hilbert, spaces.

1.2
Relation to previous work

Approximating an operator between function spaces with training data obtained through numerical
PDEs solves presents a formidable challenge. Nevertheless, in recent years, significant advances
have been made through the development of DL techniques, leading to the field of operator learning
[15, 40, 51, 53, 56, 58, 60, 62, 69, 83, 93, 103]. These approaches often leverage intricate DNN
architectures to approximate the complex mappings inherent in physical modelling scenarios. Many
works have also focused on the practical aspects of operator learning in real-world applications
[13, 24, 35–37, 42, 45, 48, 49, 59, 61, 64, 65, 70, 73, 77, 80, 81, 84, 96, 98–101, 104, 105].

On the theoretical side, universal approximation theorems for operator learning have been developed
in [50, 55, 68, 69] and elsewhere. Such bounds are typically not quantitative in the size of the DNN
needed to achieve a certain error. For this, one typically either restricts to specific operators (e.g.,
certain PDEs) or imposes regularity conditions. One such assumption is Lipschitz regularity – see
[10, 16, 55, 66, 87] and references therein. However, learning Lipschitz operators suffers from a
curse of parametric complexity [54], meaning that algebraic rates may not be achievable. Another

2


---Page Break---
common assumption is holomorphy. While stronger, it is, as noted, very relevant to operator learning
problems involving parametric PDEs. Quantitative approximation results for holomorphic operators
have been shown in [26, 31, 41, 55, 71] and elsewhere.

However, these works do not consider the generalization error, i.e., the error incurred when learning
the approximation (1.3) from the finite training data (1.2). This is particularly important in applica-
tions of operator learning where data is obtained through expensive numerical PDE solves, since such
problems are highly data-starved. Several works have tackled this question from the perspective of
statistical learning theory and nonparametric estimation [16, 55, 66], but only for Lipschitz operators.
As observed in [6, §9.5], this approach generally leads to a best O(m−1/2) decay of the L2
µ-norm
error with respect to m. Theorem 4.1 shows that such a rate is strictly suboptimal for learning the
classes of holomorphic operators we consider. Our generalization bounds in Theorems 3.1-3.2 do not
use such techniques, and yield near-optimal rates in both the L2
µ- and L∞
µ -norms. See also [30] for
some related work in this direction for reduced-order modelling with convolutional autoencoders.

Our work is inspired by recent research on learning holomorphic, Banach-valued functions [2, 5,
6]. We extend both these works, in particular, [5], to learning holomorphic operators. We also
significantly improve the error decay rates in [5] with respect to m and show they can be achieved
using substantially smaller DNNs with standard training (i.e., ℓ2-loss minimization). See Remarks
C.1-C.2. Our theoretical guarantees fall into the category of encoder-decoder-nets [52], which
includes the well-known PCA-Net [10] and DeepONet [68] frameworks. As in other recent works
[16, 30, 55, 66], in Theorems 3.1-3.2 we assume the encoder-decoder pair (EX , DY) in (1.3) have
been learned, and focus on the generalization error when training the DNN b
N.

2
Notation, assumptions, setup and examples

2.1
Notation

Let (X, ∥· ∥X ) and (Y, ∥· ∥Y) be Banach spaces and µ be a probability measure on X. Let
(Y∗, ∥· ∥Y∗) be the dual of Y and B(Y∗) be its unit ball. The Bochner and Pettis Lp-norms of
a (strongly and weakly, respectively) measurable operator F : X →Y are defined as

∥F∥Lp
µ(X;Y) =
Z

X
∥F(X)∥p
Y dµ(X)
1/p

|||F|||Lp
µ(X;Y) =
sup
y∗∈B(Y∗)

Z

X
|y∗(F)|p dµ(X)
1/p
,

respectively, for 1 ≤p < ∞, and analogously for p = ∞(see, e.g., [8, 44]). Notice that
|||F|||Lp
µ(X;Y) ≤∥F∥Lp
µ(X;Y) for 1 ≤p < ∞, while |||F|||L∞
µ (X;Y) = ∥F∥L∞
µ (X;Y).

Throughout this work, ℓp(N), 0 < p ≤∞denotes the standard ℓp space with (quasi-)norm ∥· ∥p.
We also define the monotone ℓp space ℓp
M(N) as the space of all sequences z = (zi)∞
i=1 ∈RN whose
minimal monotone majorant ˜z ∈ℓp(N). Here z = (˜zi)∞
i=1 is defined as ˜zi = supj≥i |zj|.

Given a (componentwise) activation function σ, we consider feedforward DNNs of the form

N : Rn →Rk, z 7→N(z) = AL+1(σ(AL(σ(· · · σ(A0(z)) · · · )))),
(2.1)

where Al : RNl →RNl+1 are affine maps, and N0 = n and NL+2 = k. We define width(N) =
max{N1, . . . , NL+1} and depth(N) = L. We denote a class of DNNs of the form (2.1) with a
fixed architecture (i.e., fixed activation function, depth and widths) as N, and write width(N) =
max{N1, . . . , NL+1} and depth(N) = L.

2.2
Assumptions and setup

Let F : X →Y be the unknown operator we seek to learn and

eEX : X →RdX , eDX : RdX →X,
EY : Y →RdY, DY : RdY →Y

be approximate encoders and decoders for X and Y, respectively. As mentioned, we assume that
these maps have already been learned, and focus on the training of the DNN b
N in (1.3). Our main

3


---Page Break---
results allow for arbitrary encoders and decoders (subject to the assumptions detailed below), and
provide generalization bounds that are explicit in these terms: specifically, they depend on how well
each encoder-decoder pair approximates the respective identity map on X or Y.

In order to formulate the precise notion holomorphy for the operator F, we require the following.
Let D = [−1, 1]N and ϱ be the uniform probability measure on D. Given ρ > 1, we define the
Bernstein ellipse E(ρ) =

(x + x−1)/2 : x ∈C, 1 ≤|x| ≤ρ
	
⊂C, and, for convenience, we let
E(1) = [−1, 1]. Next, for ρ = (ρi)i∈N ≥1, we define the Bernstein polyellipse as the product
E(ρ) = E(ρ1) × E(ρ2) × · · · ⊂CN.
Definition 2.1 (Holomorphic map). Let ε > 0, b ∈ℓ1(N) with b ≥0. A Banach-valued function
f : D →Y is (b, ε)-holomorphic if it is holomorphic in the region

R(b, ε) =
[ n
E(ρ) : ρ ≥1,

∞
X

j=1

 
(ρj + ρ−1
j )/2 −1

bj ≤ε
o
⊂CN,
b = (bj)j∈N.
(2.2)

See, e.g., [17, 85]. As noted in [7] we can, by rescaling b, assume that ε = 1. For convenience, we
define the following unit ball, consisting of all such functions of norm at most one over R(b, 1):

H(b) = {f : D →V (b, 1)-holomorphic : ∥f(x)∥Y ≤1, ∀x ∈R(b, 1)}.
(2.3)

Assumption 2.2. Let D = [−1, 1]N and ϱ be the uniform probability measure on D.
(A.I) There is a measurable mapping ι : X →RN such that pushforward ς := ι♯µ is a quasi-uniform
measure supported on D and ι|supp(µ) : X →ℓ∞(N) is Lipschitz with constant Lι ≥0.
(A.II) The operator F has the form F = f ◦ι, where f ∈H(b) for some b ∈ℓp
M(N) and 0 < p < 1.
(A.III) The map EX := ιdX ◦eDX ◦eEX is measurable (here ιdX : X →RdX is the restriction of ι, i.e.,
ιdX (X) = (ι(X)i)dX
i=1) and the pushforward ˜ς := EX ♯µ is absolutely continuous with respect to ϱ.
(A.IV) The maps DY and EY are linear and bounded.

Now let X1, . . . , Xm ∼i.i.d. µ and consider the training data

{(Xi, Yi)}m
i=1 ⊂(X × Y)m,
where Yi = F(Xi) + Ei ∈Y
(2.4)

and Ei ∈Y represents noise. Let N be a class of DNNs N : RdX →RdY, and define

F ≈bF := DY ◦b
N ◦EX ,
where b
N ∈argmin
N∈N

1
m

m
X

i=1
∥Yi −DY ◦N ◦EX (Xi)∥2
Y.
(2.5)

2.3
Discussion of assumptions

We now discuss (A.I)-(A.IV). In §6 we describe future work on relaxing these assumptions.

(A.I) is a weak assumption. It asserts that there is a Lipschitz map ι under which the pushforward
of µ is a quasi-uniform measure supported in D. As we discuss in Example 2.3, this is notably
the case when µ is the law of some random field with an affine parametrization involving bounded
random variables – a situation that occurs frequently in parametric and stochastic PDE problems.
(A.II) describes the specific holomorphy of the operator F – see Remark 2.4 for details. Note that we
require b ∈ℓp
M(N), not just b ∈ℓp(N). It is known [7] that one cannot learn holomorphic functions
(and hence operators) from finite data if b ∈ℓp(N) only. (A.III) is a relatively weak assumption. In
view of (A.I), we expect it to hold as long as the eDX ◦eEX ≈IX sufficiently well. Finally, (A.IV) is
a standard assumption, which holds for instance in the case of PCA-Net and DeepONet. The former
also enforces the learned encoder eEX to be linear, which is not needed in our setup. Moreover, both
approaches usually only deal with the case where both X and Y are Hilbert spaces.

Example 2.3 (Parametric PDEs) A common operator learning problem involves learning the map

F : a ∈X 7→u(a) ∈Y,
where u(a) satisfies Fau = 0
(2.6)

and Fa specifies a certain PDE depending on a parameter or function a. A standard example is the
elliptic diffusion equation over a domain Ω⊂Rn. Here a = a(x) ∈L∞(Ω) =: X is the diffusion
coefficient and u = u(·; a) is the solution of the PDE

−∇· (a∇u(z; a)) = g, z ∈Ω,
u(z; a) = 0, z ∈∂Ω.
(2.7)

4


---Page Break---
Problems such as (2.6) are ubiquitous in scientific computing, with many applications in engineering,
biology, physics, finance and beyond. In many such applications, it is common to assume that the
measure µ on X is the law of a random field

a(x) = a(·; x) = a0(·) +

∞
X

i=1
cixiϕi(·),
(2.8)

for functions a0, ϕi ∈X, where the xi are random variables and ci ≥0 are scalars that ensure that
a ∈L∞(Ω). Under some mild assumptions, (2.8) is then the Karhunen–Loève (KL) expansion of the
measure µ. See, e.g., [91] (see also [55, §3.5.1]). The xi are typically independent. While in some
settings, they may have infinite support, it is also common in practice to assume they range between
finite maxima and minima. After rescaling, one may therefore assume that x ∈D = [−1, 1]N.

Problems of this type fits into our framework. Suppose that x = (x1, x2, . . .) ∼ϱ. The measure
µ is then given as the pushforward µ = a♯ϱ and f : D →Y is the parametric solution map
f : x ∈D 7→u(·; a(x)) ∈Y. If needed, the map ι can be defined in a number of different ways.
Suppose, for instance, that X is a Hilbert space, e.g., X = L2(Ω), and {ϕi}∞
i=1 is a Riesz system
(this holds, for instance, in the case of a KL expansion, in which case {ϕi}∞
i=1 is an orthonormal
basis). Then {ϕi}∞
i=1 has a unique biorthogonal dual Riesz system {ψi}∞
i=1. We may therefore define
ι : a 7→
 
⟨a −a0, ψi⟩L2(Ω)/ci
∞
i=1. Notice that ι is a bounded linear map and F(X) = f ◦ι(X) =
f(x) for X = a(x) ∼µ. However, evaluating ι is often not required for computations (see §A.1).

This example considers an affine parametrization (2.8) inducing the measure µ. Note that other
parametrizations can be considered. Common examples include the quadratic a(z; x) = a0(z) +
(P∞
i=1 cixiϕi(z))2 and log-transformed a(z, x) = exp (P∞
i=1 cixiϕi(z)) parametrizations [17].

Remark 2.4 (Holomorphy assumption) In the previous example, the operator F stems from the
solution map f : D →Y of a parametric PDE. The regularity of solution maps of parametric PDEs
has been intensively studied, and it is known that many such maps are (b, ε)-holomorphic (hence
the resulting operator satisfies (A.II)). Consider, for instance, the affine diffusion problem (2.7)-(2.8).
Under a mild uniform ellipticity condition, the solution map of the standard weak form of the PDE
f : x ∈D 7→u(a(·; x)) ∈H1
0(Ω) is (b, ε)-holomorphic with b = (bi)∞
i=1 and bi = ci∥ϕi∥L∞(Ω).
See, e.g., [3, Prop. 4.9], as well as §B.3. Similar results are known for other parametric PDEs. This
includes parabolic PDEs, various types of nonlinear, elliptic PDEs, PDEs over parametrized domains,
parametric hyperbolic problems and parametric control problems. See [19] or [3, Chpt. 4] for reviews.

3
Main results I: upper bounds

We now present our first two main results. In these results, given an optimization problem mint f(t),
we say that ˆt is a τ-approximate minimizer for some τ ≥0 if f(ˆt) ≤mint f(t) + τ 2.
Theorem 3.1 (Existence of good DNN architectures). Let m ≥3, δ > 0, 0 < ϵ < 1 and
L = L(m, ϵ) = log4(m) + log(1/ϵ). Then there exists a class N of hyperbolic tangent (tanh)
DNNs N : RdX →RdY depending on m and ϵ only with

width(N) ≲(m/L)1+δ,
depth(N) ≲log(m/L),
(3.1)

such that following holds. Suppose that Assumption 2.2 holds and

dX ≥⌈m/L⌉,
Lι · ∥IX −eDX ◦eEX ∥L2µ(X;X) ≤c · (m/L)−1/2,
(3.2)

where IX : X →X is the identity map and c > 0 is a universal constant. Let X1, . . . , Xm ∼i.i.d. µ
and consider the noisy training data (2.4) with arbitrary noise Ei ∈Y. Then, with probability at
least 1 −ϵ, every τ-minimizer b
N of (2.5), where τ ≥0 is arbitrary, yields an approximation bF that
satisfies

|||F −bF|||L2µ(X;Y) ≲Eapp,2 + EX,2 + EY,2 + Eopt,2 + Esamp,2,
(3.3)

∥F −bF∥L∞
µ (X;Y) ≲Eapp,∞+ EX,∞+ EY,∞+ Eopt,∞+ Esamp,∞,
(3.4)

and, if Y is a Hilbert space,

∥F −bF∥L2µ(X;Y) ≲Eapp,2 + EX,2 + EY,2 + Eopt,2 + Esamp,2.
(3.5)

5


---Page Break---
Here, the approximation error terms Eapp,q, q = 2, ∞, are given by

Eapp,q = aY · C(b, p, ξ) · (m/L)θ+1−1/q−1/p,
(3.6)
where aY = ∥DY ◦EY∥Y→Y, C(b, p, ξ) > 0 depends on b, p and ξ only and θ = 0 if Y is a Hilbert
space (as in (3.5)) or θ = 1/2 otherwise (as in (3.3)-(3.4)). The other terms are given by

EX,2 = aY · Lι ·
p

m/(Lϵ) · ∥IX −eDX ◦eEX ∥L2µ(X;X)

EX,∞= aY · Lι ·
p

m/L ·
p

m/L · ∥IX −eDX ◦eEX ∥L2µ(X;X) + ∥IX −eDX ◦eEX ∥L∞
µ (X;X)


EY,2 = ∥IY −DY ◦EY∥L2
F ♯µ(Y;Y)/√ϵ

EY,∞= ∥IY −DY ◦EY∥L∞
F ♯µ(Y;Y) +
p

m/L · ∥IY −DY ◦EY∥L2
F ♯µ(Y;Y),

(3.7)

where IY : Y →Y is the identity map and, if ∥E∥2
2;Y = Pm
i=1 ∥Ei∥2
Y,

Eopt =
τ + 2−m
q = 2
p

m/Lτ + 2−m
q = ∞,
Esamp,q =

(
∥E∥2;Y/√m
q = 2
∥E∥2;Y/
√

L
q = ∞.
(3.8)

(Proofs of this and all other theorems are in §C-G of the supplemental material.) This theorem shows
that there is a family of tanh DNNs that yield provable bounds for learning holomorphic operators.
The error (3.3)-(3.5) decomposes into an approximation error (3.6), which decays algebraically in
the amount of training data m. Later, in Theorems 4.1-4.2, we show that these rates are optimal when
Y is a Hilbert space, up to log factors. Next, are the encoding-decoding errors (3.7), which depend on
how well the approximate encoder-decoder pairs (eEX , eDX ) and (EY, DY) approximate the identity
maps on X and Y, respectively. Observe that these terms are increasing in m for fixed encoders
and decoders. Therefore, as one expects, the accuracy of the encoder-decoder approximations
eDX ◦eEX ≈IX and DY ◦EY ≈IY should increase with increasing m to ensure decay to zero of the
generalization error as m →∞. The specific terms in (3.7) (for q = 2) are quite standard in operator
learning. See, e.g., [53, 55]. When the encoders and decoders are computed via PCA, as in PCA-Net,
standard bounds can be derived for these terms [53]. For similar analysis in the case of DeepONets,
see [55]. Finally, the error (3.3)-(3.5) involves an optimization error Eopt, which primarily depends
on how accurately the optimization problem (2.5) is solved (i.e., the term τ), and a sampling error
Esamp, which depends on the error in the training data (2.4).

Theorem 3.1 allows Y to be a Banach or a Hilbert space. Overall, when Y is only a Banach space, we
obtain a weaker L2
µ-norm bound involving the Pettis norm (3.3) and, moreover, the approximation
error Eapp,q is worse by a factor of 1/2 than when Y is a Hilbert space. (Note that one can establish
a bound for the Bochner L2
µ-norm error when Y is a Banach space via (3.4) and the inequality
∥· ∥L2µ(X;Y) ≤∥· ∥L∞
µ (X;Y). However, we do not believe the resulting bound is sharp). As we
discuss in Remark D.18, the discrepancies between the two cases stem from the lack of an inner
product structure and, in particular, the absence of Parseval’s identity when Y is a Banach space.

Observe that the DNN architecture in Theorem 3.1 is independent of the smoothness of the operator
being learned. We term such an architecture problem agnostic. This theorem considers tanh activations
only. However, as we discuss in Remark D.11, other activations can be readily used instead. Other key
facets of Theorem 3.1 are the width and depth bounds (3.1). Qualitatively, these agree with empirical
practice: namely, better performing DNNs tend to be wider than they are deep, and relatively shallow
DNNs perform well in practice (see [24, 25] and references therein). We also see this later in §5.

On the other hand, the family N is not fully connected. As we describe in §C.2.1, while the weights
on the final layer can be arbitrary real numbers, the weights and biases in the hidden layers come from
a finite (but large) set: they are handcrafted to approximately emulate certain multivariate orthogonal
polynomials. Since fully-connected DNNs are typically used in practice, Theorem 3.1 is essentially a
theoretical contribution. In our next result, we consider the more practical scenario of fully-connected
DNNs.
Theorem 3.2 (Fully-connected DNN architectures are good). There are universal constants
c1, c2, c3, c4 ≥1 such that the following holds. Let m, δ, ϵ and L be as in Theorem 3.1,

dX ≥c1(m + log(1/ϵ)),
Lι · ∥IX −eDX ◦eEX ∥L2µ(X;X) ≤c(δ) · (m + log(1/ϵ))−1/2,
(3.9)

6


---Page Break---
where c(δ) > 0 depends on δ only, consider any class N of fully-connected DNNs satisfying

(n0, nL+2) = (dX , dY),
N1, . . . , NL+1 ≥c2 · (m + log(1/ϵ)) · (m/L)δ,
L ≥c3 · log(m/L).
(3.10)
Suppose that Assumption 2.2 holds and that the pushforward ς in (A.I) is the tensor-product of a
univariate probability distribution with mean zero and variance ω ≳1. Let X1, . . . , Xm ∼i.i.d. µ
and consider (2.4) with arbitrary Ei ∈Y. Then the following hold with probability at least 1 −ϵ.

(A) Uncountably many ‘good’ minimizers. The problem (2.5) has uncountably many minimizers that
satisfy (3.3) with τ = 0 or (3.5) with τ = 0 if Y is a Hilbert space. They also satisfy (3.4) with
τ = 0 and the modified right-hand side
√

LEapp,∞+LEX,∞+
√

LEY,∞+Eopt,∞+
√

LEsamp,∞.

(B) Good minimizers are stable. Suppose that EX ∈L∞
µ (X; RdX ) and let τo > 0 be arbitrary. Then
there is a neighbourhood of DNN parameters around the parameters of each minimizer in (A) for
which the approximation corresponding to any parameters in this neigbourhood also satisfies the
same bounds as in (A) with τ = τo.

(C) Good minimizers can be far apart in parameter space. For sufficiently large m, there are at least
(m/(c4L))2δm minimizers satisfying the bounds in (A) such that, for any two such minimizers,
their parameters satisfy ∥θ′∥= ∥θ∥and ∥θ′ −θ∥≳1.

This theorem states that DL with fully-connected DNN architectures of sufficient width and depth
(3.10) can succeed, since there are minimizers that yield the optimal bounds of Theorem 3.1. Such
minimizers are uncountably many in number (A), stable to perturbations (B) and many of them
(exponentially in m) have sufficiently distinct and nonvanishing/nonexploding parameters (C). This
theorem does not imply that all minimizers are ‘good’ – an issue we discuss further in §6 – but our
numerical results in §5 suggest that (approximate) minimizers obtained through training do, at least
for the experiments considered, achieve the rates specified in Theorem 3.1.

4
Main results II: lower bounds

We now show that the various approximation errors are nearly optimal. For this, we ignore the
encoding-decoding, optimization and sampling errors and proceed as follows. Let C(X; Y) be the
Banach space of continuous operators. We term an (adaptive) sampling map as any map

L : C(X, Y) →Ym,
F 7→L(F) = (F(Xi))m
i=1 ,
(4.1)

where X1 ∈X, X2 = X2(F(X1)) ∈X potentially depends on the previous evaluation F(X1),
X3 = X3(F(X1), F(X2)) ∈X, and so forth. Next, we term a reconstruction map as any map
R : Ym →L2
µ(X; Y). Given this, we let H(b, ι) = {F = f ◦ι : f ∈H(b)} and define

θm(b) = inf
L,R
sup
F ∈H(b,ι)
|||F −R ◦L(F)|||L2µ(X;Y),
(4.2)

where the infimum is taken over all such L and R. In other words, θm(b) measures how well one can
learn holomorphic operators using arbitrary training data and an arbitrary reconstruction procedure.
Theorem 4.1 (Optimal L2 error rates). Suppose that (A.I) holds. Then, for any 0 < p < 1 there is a
constant c(p) > 0 such that the following hold.

(i) For each m ∈N, there is a b ∈ℓp
M(N), b ≥0, ∥b∥p,M = 1 such that θm(b) ≥c(p) · m1/2−1/p.

(ii) There is a b ∈ℓp
M(N), b ≥0, ∥b∥p,M = 1 such that θm(b) ≥c(p) ·
m1/2−1/p

log2/p(2m), ∀m ∈N.

This theorem shows that the error Eapp,2 in Theorems 3.1-3.2 is optimal, up to log terms, whenever
Y is a Hilbert space: there does not exist a reconstruction map surpasses the rate m1/2−1/p for
learning holomorphic operators. Note that this result applies not only to DL-based procedures, but
any procedure that learns such operators from m samples. Another consequence of this result is that
adaptive sampling, i.e., active learning, is of no benefit. As shown by Theorems 3.1-3.2, the optimal
rate m1/2−1/p can, up to log terms, be achieved through inactive learning, i.e., i.i.d. sampling from µ.

Theorem 4.1 considers L2-norm. For the L∞-norm, we present a somewhat weaker result. Let
˜θm(b) = inf
R {EX1,...,Xm∼µ
sup
F ∈H(b,ι)
∥F −R({Xi, F(Xi)})∥L∞
µ (X;Y)},
(4.3)

7


---Page Break---
where the infimum is taken over all reconstruction maps R : (X × Y)m →L∞
µ (X; Y) only.
Theorem 4.2 (Optimal L∞error rates). Suppose that (A.I) holds and that the pushforward ς is the
tensor-product of a univariate probability distribution with mean zero and variance ω ≳1. Then, for
any 0 < p < 1 there is a constant c(p) > 0 such that the following hold.

(i) For each m ∈N, there is a b ∈ℓp
M(N), b ≥0, ∥b∥p,M = 1 such that ˜θm(b) ≥c(p) · m1−1/p

log(m) .

(ii) There is a b ∈ℓp
M(N), b ≥0, ∥b∥p,M = 1 such that ˜θm(b) ≥c(p) ·
m1−1/p

log2/p+1(2m), ∀m ∈N.

As with the previous theorem, this result asserts that the rate m1−1/p is optimal in the L∞-norm
when Y is a Hilbert space. However, it is strictly weaker than Theorem 4.1 as it only considers
i.i.d. random sampling from µ, as opposed to arbitrary (adaptive) samples. Note that Theorem 4.1 is
an extension of [7, Thm. 4.4]. Theorem 4.2 is new, and is of independent interest since it partially
addresses an open problem of [7] about deriving lower bounds in the L∞
µ -norm, as opposed to just
the L2
µ-norm. See §C.2.3-C.2.4 for more discussion.

5
Numerical experiments

We now present numerical results for DL applied to various different parametric PDE problems, as in
Example 2.3. For a full description of our experimental setup, see §A-B.

Since the main objective of this work is to examine the approximation error, we follow a standard
setup and fix the encoder and decoders for each experiment, so that EX and DY in (1.3) do not change
for different choices of b
N. We also set up our experiments so that encoding-decoding (3.7) and
sampling (3.8) errors are zero. We do this in a standard way. To ensure that EX,q = 0, we truncate
the parametric expansions (2.8) after d terms (henceforth termed the parametric dimension) and
define the encoder EX accordingly. This means we effectively consider a parametric PDE depending
on finitely-many parameters. We use Finite Element Methods (FEMs) to both solve the PDE (for
generating training and testing data) and define the decoder DY (see (A.2)). To ensure that EY,q = 0,
we compute errors with respect to the Bochner L2
µ(X; eY)-norm, where eY = DY(RdY) is the FEM
discretization of Y. In other words, we use the same FEM code to generate test data and compute the
errors as we do to construct the operator approximation bF. See §A.1 for further details.

The DNNs in our experiments are fully-connected and of the form (2.1). We denote by σ L × N
DNN a DNN b
N with activation function σ, width N and depth L. To solve (2.5) we use Adam [47]
with early stopping and an exponentially decaying learning rate. We train our DNN architectures for
60,000 epochs and results are averaged over a number of trials. See §A.2 for further details.

Parametric elliptic diffusion equation. Our first example is the parametric elliptic diffusion equation
(2.7). This PDE arises in many scientific computing applications, such as groundwater flow modelling,
see, e.g., [95]. We describe the full PDE and its FE discretization in §B.3. In our experiments, we
consider both affine (B.1) and log-transformed (B.2) diffusion coefficients. The latter is particularly
useful in the groundwater flow problem as the permeability of various layers of sediment can vary
on logarithmic scales. Differing from most prior work, we consider a novel mixed variational
formulation [32] of (2.7), which has a number of key practical benefits (see §B.3.1). In this case,
Y = L2(Ω) is a Hilbert space. Fig. 1 compares the error versus the amount of training data m for
various DNN architectures for learning the solution map of this PDE in d = 4 and d = 8 parametric
dimensions with these two diffusion coefficients. We observe that architectures with the Exponential
Linear Unit (ELU) or hyperbolic tangent (tanh) activation generally outperform similar architectures
with the Rectified Linear Unit (ReLU) activation (as we discuss in Remark D.11, this difference is
in agreement with our theoretical analysis). Overall, the best performing DNNs appear to roughly
match the plotted rate m−1. As we explain further in §B.3.2, this rate is precisely that predicted by
our theory. In particular, the parametric solution map (recall Remark 2.4) is (b, ε)-holomorphic with
b ∈ℓp
M(N) for any p < 2/3, giving an effective convergent rate m1/2−1/p that is arbitrarily close to
m−1. Another important fact that we observe is that despite the parametric dimension doubling from
4 to 8, there is little change in the error behaviour.

Parametric Navier-Stokes-Brinkman equations. We next consider the parametric Navier-Stokes-
Brinkman (NSB) equations. See §B.4 and (B.14) for the full definition. Here the solution is a pair

8


---Page Break---
101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

Figure 1: Elliptic diffusion equation. Average relative L2
µ(X; eY)-norm error versus m for different DNNs
approximating the solution operator for the elliptic diffusion equation (B.9). The first two plots use the affine
coefficient a1,d (B.1) with d = 4, 8, respectively. The rest use the log-transformed coefficient a2,d (B.2).

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

Figure 2: NSB equations. Average relative L2
µ(X; eY)-norm error versus m for different DNNs approximating
the velocity field u of the NSB problem in (B.14). See Fig. 7 for results for the pressure component p. The
diffusion coefficients a1,d, a2,d and d = 4, 8 are as in Fig. 1.

(u, p), where u is the velocity field and p is the pressure. These equations describe the dynamics of a
viscous fluid flowing through porous media with random viscosity. See, e.g., [28, 43, 46, 94]. We use
a mixed variational formulation [34] to discretize the PDE. This formulation is more sophisticated
that standard variational formulations, but conveys various practical advantages. Unlike the previous
example, it leads to Y being either Y = L4(Ω) for u or Y = L2(Ω) for p. See §B.4.1 for details.
Fig. 2 compares a variety of DNN architectures for approximating the velocity field component
in d = 4 and d = 8 parametric dimensions. Here again we observe the ELU and tanh DNN
architectures outperform similar sized ReLU architectures. We also observe a rate close to m−1. Note
that it is currently unknown whether this or the next example possess the same (b, ε)-holomorphy
guarantee as that of the previous example. Yet we observe the same rate, and therefore conjecture
that such a property does indeed hold in these cases. Similar to the previous example, there is also no
deterioration of the rate when moving from d = 4 to d = 8.

Parametric stationary Boussinesq equation. Our final example is a parametric stationary Boussi-
nesq PDE. See §B.5 and (B.16) for the full definition. Here the solution is a triplet (u, φ, p), where u
is the velocity field, φ is the temperature and p is the pressure of the solution. The Boussinesq model
arises in a variety of engineering, fluid dynamics and natural convection problems where changes in
temperature affect the velocity of a fluid [14, 22, 39]. Similar to the previous example, we consider
a fully mixed variational formulation (see §B.5.1), which leads to Y = L4(Ω) (for u), Y = L4(Ω)
(for φ) or Y = L2
0(Ω) (for p). Fig. 3 provides numerical results. Our observations are in line with
the previous two examples, with the ELU and the smaller tanh networks being most often the best
performers in this problem. Once more, the errors roughly correspond to the rate m−1 and there is no
deterioration with increasing d.

6
Conclusions and limitations

The purpose of this work was to derive near-optimal generalization bounds for learning certain
classes of holomorphic operators that arise frequently in operator learning tasks involving PDEs.
Complementing and extending previous works [26, 31, 41, 55, 71] on the approximation of such
operators via DNNs, we showing sharp algebraic rates of convergence in m, thus confirming that

9


---Page Break---
101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

Figure 3: Boussinesq equation. Average relative L2
µ(X; eY)-norm error versus m for different DNNs approxi-
mating the temperature φ of the Boussinesq problem in (B.16) (see Fig. 9 for u and p). The diffusion coefficients
a1,d, a2,d and d = 4, 8 are as in Fig. 1. In this example, we also consider an additional parametric dependence
in the tensor K = Kd describing the thermal conductivity of the fluid. See §B.5 and (B.17).

such operators can be learned efficiently and without the curse of dimensionality. It is notable that
the sizes of the various DNNs in Theorems 3.1-3.2 also do not succumb to the so-called curse of
parametric complexity [54], since the width and depth bounds are at most algebraic in m.

We end by discussing a number of limitations. First, assumption (A.I) may not hold in some
applications. The domain D can easily be replaced by bounded hyperrectangle through rescaling and
the condition that ς be quasi-uniform relaxed to quasi-ultraspherical (by considering ultraspherical
polynomials). However, it is currently an open problem whether our results can be extended to the
case where µ is Gaussian, in which case ς would typically be a tensor-product Gaussian measure
on RN and the relevant polynomials would be the Hermite polynomials. Second, the reader may
have noticed that the encoder EX defined in (A.III) and used to construct the approximation (2.5)
involves the pair (eEX , eDX ) and the map ιdX . This is a technical requirement – also found in other
theoretical works on operator learning – needed to obtain encoding-decoding errors of the form EX,q,
q = 2, ∞. It is unknown whether it can be relaxed. It is also unknown whether the assumption on ˜ς
in (A.III) can be relaxed. We believe this can be done, at least if the L2
µ-norm in (3.2) is replaced by
the L∞
µ -norm. Whether this is possible without modifying (3.2) is currently unknown.

Third, a limitation of Theorem 3.2 is that it only asserts that some minimizers are ‘good’, not all.
Techniques from statistical learning theory can provide stronger bounds that hold for all minimizers.
Yet, as noted in §1.2, these tools typically produce slower rates of decay in m. Overcoming this
limitation – e.g., by refining these tools for the holomorphic setting or showing that the ‘good’
minimizers can indeed be obtained via standard training – is a topic of future work.

Finally, as noted, our theorems provided worse generalization bounds when Y is a Banach space than
when Y is a Hilbert space. Our numerical results in Figs. 2-3 suggest that this factor is an artefact of
the proofs. Whether it can be removed is an interesting open problem.

10


---Page Break---
Acknowledgments and Disclosure of Funding

BA acknowledges the support of the Natural Sciences and Engineering Research Council of Canada
of Canada (NSERC) through grant RGPIN-2021-611675. ND acknowledges the support of Florida
State University through the CRC 2022-2023 FYAP grant program. The authors would like to thank
Gregor Maier for helpful comments and feedback.

References

[1] B. Adcock and N. Dexter. The gap between theory and practice in function approximation
with deep neural networks. SIAM J. Math. Data Sci., 3(2):624–655, 2021.

[2] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga. Deep neural networks are effective
at learning high-dimensional Hilbert-valued functions from limited data. In J. Bruna, J. S.
Hesthaven, and L. Zdeborová, editors, Proceedings of The Second Annual Conference on
Mathematical and Scientific Machine Learning, volume 145 of Proc. Mach. Learn. Res.
(PMLR), pages 1–36. PMLR, 2021.

[3] B. Adcock, S. Brugiapaglia, and C. G. Webster. Sparse Polynomial Approximation of High-
Dimensional Functions. Comput. Sci. Eng. Society for Industrial and Applied Mathematics,
Philadelphia, PA, 2022.

[4] B. Adcock, N. Dexter, and S. Moraga. Optimal approximation of infinite-dimensional holo-
morphic functions II: recovery from i.i.d. pointwise samples. arXiv:2310.16940, 2023.

[5] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga. Near-optimal learning of Banach-
valued, high-dimensional functions via deep neural networks. Neural Networks (in press),
2024.

[6] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga. Learning smooth functions in high
dimensions: from sparse polynomials to deep neural networks. In S. Mishra and A. Townsend,
editors, Numerical Analysis Meets Machine Learning, volume 25 of Handbook of Numerical
Analysis, pages 1–52. Elsevier, 2024.

[7] B. Adcock, N. Dexter, and S. Moraga. Optimal approximation of infinite-dimensional holo-
morphic functions. Calcolo, 61(1):12, 2024.

[8] C. D. Aliprantis and K. C. Border. Infinite Dimensional Analysis: A Hitchhiker’s Guide.
Springer–Verlag, Berlin, Heidelberg, 3rd edition, 2006.

[9] S. Alnæs, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg, C. Richardson, J. Ring, M. E.
Rognes, and G. N. Wells. The FEniCS Project Version 1.5. Archive of Numerical Software, 3
(100), 2015.

[10] K. Bhattacharya, B. Hosseini, N. B. Kovachki, and A. M. Stuart. Model reduction and neural
networks for parametric PDEs. SMAI J. Comput. Math., 7:121–157, 2021.

[11] D. Boffi, F. Brezzi, and M. Fortin. Mixed Finite Element Methods and Applications. Springer,
Berlin, Heidelberg, 1st edition, 2013.

[12] S. Brugiapaglia, S. Dirksen, H. C. Jung, and H. Rauhut. Sparse recovery in bounded Riesz
systems with applications to numerical methods for PDEs. Appl. Comput. Harmon. Anal., 53:
231–269, 2021.

[13] S. Cai, Z. Wang, L. Lu, T. A. Zaki, and G. E. Karniadakis. DeepM&Mnet: Inferring the
electroconvection multiphysics fields based on operator approximation by neural networks. J.
Comput. Phys., 436:110296, 2021.

[14] C. Cao and J. Wu. Global regularity for the two-dimensional anisotropic Boussinesq equations
with vertical dissipation. Arch. Rational Mech. Anal., 208:985–1004, 2013.

[15] Q. Cao, S. Goswami, and G. E. Karniadakis. LNO: Laplace neural operator for solving
differential equations. arXiv:2303.10528, 2023.

11


---Page Break---
[16] K. Chen, C. Wang, and H. Yang. Deep operator learning lessens the curse of dimensionality
for pdes. arXiv:2301.12227, 2023.

[17] A. Chkifa, A. Cohen, and C. Schwab. Breaking the curse of dimensionality in sparse polyno-
mial approximation of parametric PDEs. J. Math. Pures Appl., 103(2):400–428, 2015.

[18] P. G. Ciarlet. The Finite Element Method for Elliptic Problems. Society for Industrial and
Applied Mathematics, Philadelphia, PA, 2002.

[19] A. Cohen and R. A. DeVore. Approximation of high-dimensional parametric PDEs. Acta
Numer., 24:1–159, 2015.

[20] E. Colmenares, G. N. Gatica, and S. Moraga. A Banach spaces-based analysis of a new
fully-mixed finite element method for the Boussinesq problem. ESAIM Math. Model. Numer.
Anal., 54(5):1525–1568, 2020.

[21] D. D˜ung, V. K. Nguyen, and D. T. Pham. Deep ReLU neural network approximation in
Bochner spaces and applications to parametric PDEs. J. Complexity, 79:101779, 2023.

[22] I. Danaila, R. Moglan, F. Hecht, and S. Le Masson. A newton method with adaptive finite
elements for solving phase-change problems with natural convection. J. Comput. Phys., 274:
826–840, 2014.

[23] J. Daws and C. Webster. Analysis of deep neural networks with quasi-optimal polynomial
approximation rates. arXiv:1912.02302, 2019.

[24] M. De Hoop, D. Z. Huang, E. Qian, and A. Stuart. The cost-accuracy trade-off in operator
learning with neural networks. J. Mach. Learn., 1:299–341, 2022.

[25] T. De Ryck, S. Lanthaler, and S. Mishra. On the approximation of functions by tanh neural
networks. Neural Networks, 143:732–750, 2021.

[26] B. Deng, Y. Shin, L. Lu, Z. Zhang, and G. E. Karniadakis. Approximation rates of DeepONets
for learning operators arising from advection–diffusion equations. Neural Networks, 153:
411–426, 2022.

[27] N. Dexter, H. Tran, and C. Webster. A mixed ℓ1 regularization approach for sparse simultaneous
approximation of parameterized PDEs. ESAIM Math. Model. Numer. Anal., 53:2025–2045,
2019.

[28] Y. Dutil, D. R. Rousse, N. B. Salah, S. Lassue, and L. Zalewski. A review on phase-change
materials: Mathematical modeling and simulations. Renew. Sustain. Energy Rev., 15(1):
112–130, 2011.

[29] S. Foucart and H. Rauhut. A Mathematical Introduction to Compressive Sensing. Appl. Numer.
Harmon. Anal. Birkhäuser, New York, NY, 2013.

[30] N. R. Franco and S. Brugiapaglia. A practical existence theorem for reduced order models
based on convolutional autoencoders. arXiv:2402.00435, 2024.

[31] N. R. Franco, S. Fresca, A. Manzoni, and P. Zunino. Approximation bounds for convolutional
neural networks in operator learning. Neural Networks, 161:129–141, 2023.

[32] G. N. Gatica. A Simple Introduction to the Mixed Finite Element Method. SpringerBriefs Math.
Springer, Cham, Switzerland, 2014.

[33] G. N. Gatica, R. Oyarzúa, R. Ruiz-Baier, and Y. D. Sobral. Banach spaces-based analysis of a
fully-mixed finite element method for the steady-state model of fluidized beds. Comput. Math.
Appl., 84:244–276, 2021.

[34] G. N. Gatica, N. Nuñez, and R. Ruiz-Baier. New non-augmented mixed finite element methods
for the Navier–Stokes–Brinkman equations using Banach spaces. J. Numer. Math., 31(4):
343–373, 2022.

12


---Page Break---
[35] S. Goswami, K. Kontolati, M. D. Shields, and G. E. Karniadakis. Deep transfer operator
learning for partial differential equations under conditional shift. Nat. Mach. Intell., 4(12):
1155–1164, 2022.

[36] S. Goswami, D. S. Li, B. V. Rego, M. Latorre, J. D. Humphrey, and G. E. Karniadakis. Neural
operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms.
J. R. Soc. Interface, 19(193):20220410, 2022.

[37] S. Goswami, A. D. Jagtap, H. Babaee, B. T. Susi, and G. E. Karniadakis. Learning stiff
chemical kinetics using extended deep neural operators. arXiv:2302.12645, 2023.

[38] I. Gühring and M. Raslan. Approximation rates for neural networks with encodable weights in
smoothness spaces. Neural Networks, 134:107–130, 2021.

[39] Z. Guo, B. Shi, and C. Zheng. A coupled lattice BGK model for the Boussinesq equations. Int.
J. Numer. Meth. Fluids, 39:325–342, 2002.

[40] Z. Hao, Z. Wang, H. Su, C. Ying, Y. Dong, S. Liu, Z. Cheng, J. Song, and J. Zhu. GNOT: A
General Neural Operator Transformer for Operator Learning. In International Conference on
Machine Learning, pages 12556–12569. PMLR, 2023.

[41] L. Herrmann, C. Schwab, and J. Zech. Neural and spectral operator surrogates: unified
construction and expression rate bounds. Adv. Comput. Math., 50:72, 2024.

[42] A. A. Howard, M. Perego, G. E. Karniadakis, and P. Stinis. Multifidelity deep operator
networks for data-driven and physics-informed problems. J. Comput. Phys., 493:112462,
2024.

[43] W. R. Hwang and S. G. Advani. Numerical simulations of Stokes–Brinkman equations for
permeability prediction of dual scale fibrous porous media. Phys. Fluids., 22:113101, 2010.

[44] T. Hytönen, J. van Neerven, M. Veraar, and L. Weis. Analysis in Banach Spaces, Volume I:
Martingales and Littlewood–Paley Theory. Ergebnisse der Mathematik und ihrer Grenzgebiete.
3. Folge. Springer, Cham, 2016.

[45] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang. Physics-
informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021.

[46] K. Khadra, P. Angot, S. Parneix, and J. P. Caltagirone. Fictitious domain approach for
numerical modelling of Navier-Stokes equations. Int. J. Numer. Meth. Fluids, 34:651–684,
2000.

[47] D. P. Kingma and J. Ba. Adam: a method for stochastic optimization. arXiv:1412.6980, 2017.

[48] K. Kontolati, S. Goswami, M. D. Shields, and G. E. Karniadakis. On the influence of over-
parameterization in manifold based surrogates and deep neural operators. J. Comput. Phys.,
479:112008, 2023.

[49] S. Koric, A. Viswantah, D. W. Abueidda, N. A. Sobh, and K. Khan. Deep learning operator
network for plastic deformation with variable loads and material properties. Eng. Comput.,
pages 1–13, 2023.

[50] N. Kovachki, S. Lanthaler, and S. Mishra. On universal approximation and error bounds for
Fourier neural operators. J. Mach. Learn. Res., 22(1):13237–13312, 2021.

[51] N. B. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. M. Stuart, and
A. Anandkumar. Neural operator: learning maps between function spaces with applications to
PDEs. J. Mach. Learn. Res., 24(89):1–97, 2023.

[52] N. B. Kovachki, S. Lanthaler, and A. M. Stuart. Operator learning: algorithms and analysis. In
S. Mishra and A. Townsend, editors, Numerical Analysis Meets Machine Learning, volume 25
of Handbook of Numerical Analysis, pages 419–467. Elsevier, 2024.

[53] S. Lanthaler.
Operator learning with PCA-Net: upper and lower complexity bounds.
arXiv:2303.16317, 2023.

13


---Page Break---
[54] S. Lanthaler and A. M. Stuart.
The parametric complexity of operator learning.
arXiv:2306.15924, 2024.

[55] S. Lanthaler, S. Mishra, and G. E. Karniadakis. Error estimates for DeepONets: A deep
learning framework in infinite dimensions. Trans. Math. Appl., 6(1):tnac001, 2022.

[56] S. Lanthaler, Z. Li, and A. M. Stuart. The nonlocal neural operator: universal approximation.
arXiv:2304.13221, 2023.

[57] B. Li, S. Tang, and H. Yu. Better approximations of high dimensional smooth functions by
deep neural networks with rectified power units. Commun. Comput. Phys., 27:379–411, 2020.

[58] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anand-
kumar. Neural operator: graph kernel network for partial differential equations. In ICLR,
2020.

[59] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, A. Stuart, K. Bhattacharya, and A. Anandku-
mar. Multipole graph neural operator for parametric partial differential equations. Advances in
Neural Information Processing Systems, 33:6755–6766, 2020.

[60] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandku-
mar. Fourier neural operator for parametric partial differential equations. In ICLR, 2021.

[61] Z. Li, M. Liu-Schiaffini, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart,
and A. Anandkumar. Learning chaotic dynamics in dissipative systems. Advances in Neural
Information Processing Systems, 35:16768–16781, 2022.

[62] Z. Li, N. B. Kovachki, C. Choy, B. Li, J. Kossaifi, S. P. Otta, M. A. Nabian, M. Stadler,
C. Hundt, K. Azizzadenesheli, and A. Anandkumar. Geometry-informed neural operator for
large-scale 3D PDEs. arXiv:2309.00583, 2023.

[63] S. Liang and R. Srikant. Why deep neural networks for function approximation? In ICLR,
2017.

[64] C. Lin, Z. Li, L. Lu, S. Cai, M. Maxey, and G. E. Karniadakis. Operator learning for predicting
multiscale bubble growth dynamics. J. Chem. Phys., 154(10), 2021.

[65] C. Lin, M. Maxey, Z. Li, and G. E. Karniadakis. A seamless multiscale operator neural network
for inferring bubble dynamics. J. Fluid Mech., 929:A18, 2021.

[66] H. Liu, H. Yang, M. Chen, T. Zhao, and W. Liao. Deep nonparametric estimation of operators
between infinite dimensional spaces. J. Mach. Learn. Res., 25:1–67, 2024.

[67] J. Lu, Z. Shen, H. Yang, and S. Zhang. Deep network approximation for smooth functions.
SIAM J. Math. Anal., 53(5):5465–5506, 2021.

[68] L. Lu, P. Jin, and G. E. Karniadakis. DeepONet: Learning nonlinear operators for iden-
tifying differential equations based on the universal approximation theorem of operators.
arXiv:1910.03193, 2019.

[69] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis. Learning nonlinear operators via
DeepONet based on the universal approximation theorem of operators. Nat. Mach. Intell., 3
(3):218–229, 2021.

[70] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, and G. E. Karniadakis. A comprehen-
sive and fair comparison of two neural operators (with practical extensions) based on fair data.
Comput. Methods Appl. Mech. Engrg., 393:114778, 2022.

[71] C. Marcati and C. Schwab. Exponential convergence of deep operator networks for elliptic
partial differential equations. SIAM J. Numer. Anal., 61(3):1513–1545, 2023.

[72] H. Mhaskar. Neural networks for optimal approximation of smooth and analytic functions.
Neural Comput., 8(1):164–177, 1996.

14


---Page Break---
[73] K. Michałowska, S. Goswami, G. E. Karniadakis, and S. Riemer-Sørensen. Neural opera-
tor learning for long-time integration in dynamical systems with recurrent neural networks.
arXiv:2303.02243, 2023.

[74] S. Moraga. Optimal and efficient algorithms for learning high-dimensional, Banach-valued
functions from limited samples. PhD thesis, Simon Fraser University, 2024.

[75] H. H. Nguyen and V. H. Vu. Normal vector of a random hyperplane. Int. Math. Res. Not.
IMRN, 2018(6):1754–1778, 2018.

[76] F. Nobile, R. Tempone, and C. G. Webster. A sparse grid stochastic collocation method for
partial differential equations with random input data. SIAM J. Numer. Anal., 46(5):2309–2345,
2008.

[77] V. Oommen, K. Shukla, S. Goswami, R. Dingreville, and G. E. Karniadakis. Learning two-
phase microstructure evolution using neural operators and autoencoder architectures. npj
Comput. Mater., 8(1):190, 2022.

[78] J. A. A. Opschoor and C. Schwab. Deep ReLU networks and high-order finite element methods
II: Chebyshev emulation. arXiv:2310.07261, 2023.

[79] J. A. A. Opschoor, C. Schwab, and J. Zech. Exponential ReLU DNN expression of holomorphic
maps in high dimension. Constr. Approx., 55(1):537–582, 2022.

[80] J. D. Osorio, Z. Wang, G. Karniadakis, S. Cai, C. Chryssostomidis, M. Panwar, and R. Hov-
sapian. Forecasting solar-thermal systems performance under transient operation using a
data-driven machine learning approach based on the deep operator network architecture.
Energy Conversion and Management, 252:115063, 2022.

[81] W. Peng, Z. Yuan, Z. Li, and J. Wang. Linear attention coupled fourier neural operator for
simulation of three-dimensional turbulence. Physics of Fluids, 35(1), 2023.

[82] M. Pinkus. N-widths in Approximation Theory. Springer–Verlag, Berlin, 1968.

[83] B. Raoni´c, R. Molinaro, T. Rohner, S. Mishra, and E. de Bezenac. Convolutional neural
operators. In ICLR, 2023.

[84] P. I. Renn, C. Wang, S. Lale, Z. Li, A. Anandkumar, and M. Gharib. Forecasting subcritical
cylinder wakes with Fourier neural operators. arXiv:2301.08290, 2023.

[85] C. Schwab and J. Zech. Deep learning in high dimension: neural network expression rates for
generalized polynomial chaos expansions in UQ. Anal. Appl. (Singap.), 17(1):19–55, 2019.

[86] C. Schwab and J. Zech. Deep learning in high dimension: neural network expression rates for
analytic functions in L2(Rd, γd). SIAM/ASA J. Uncertain. Quantif., 11(1):199–234, 2023.

[87] C. Schwab, A. Stein, and J. Zech. Deep operator network approximation rates for Lipschitz
operators. arXiv:2307.09835, 2023.

[88] M. Stoyanov. User manual: Tasmanian sparse grids. Technical Report ORNL/TM-2015/596,
Oak Ridge National Laboratory, One Bethel Valley Road, Oak Ridge, TN, 2015.

[89] M. Stoyanov, D. Lebrun-Grandie, J. Burkardt, and D. Munster. Tasmanian, 9 2013. URL
https://github.com/ORNL/Tasmanian.

[90] M. K. Stoyanov and C. G. Webster. A dynamically adaptive sparse grid method for quasi-
optimal interpolation of multidimensional functions. Comput. Math. Appl., 71(11):2449–2465,
2016.

[91] A. M. Stuart. Inverse problems: a Bayesian perspective. Acta Numer., 19:451–559, 2010.

[92] S. Tang, B. Li, and Y. Haijun. ChebNet: efficient and stable constructions of deep neural
networks with rectified power units via Chebyshev approximation. Commun. Math. Stat. (In
press), 2024.

15


---Page Break---
[93] T. Tripura and S. Chakraborty. Wavelet neural operator for solving parametric partial differen-
tial equations in computational mechanics problems. Comput. Methods Appl. Mech. Engrg.,
404:115783, 2023.

[94] S. Wang, A. Faghri, and T. L. Bergman. A comprehensive numerical model for melting with
natural convection. Int. J. Heat Mass Transfer., 53(9-10):1986–200, 2010.

[95] C. L. Winter and D. M. Tartakovsky. Groundwater flow in heterogeneous composite aquifers.
Water Resour. Res., 38(8), 2002.

[96] K. Wu, T. O’Leary-Roseberry, P. Chen, and O. Ghattas.
Large-scale Bayesian optimal
experimental design with derivative-informed projected neural network. J. Sci. Comput., 95
(1):30, 2023.

[97] D. Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks,
94:103–114, 2017.

[98] E. Yeung, S. Kundu, and N. Hodas. Learning deep neural network representations for Koopman
operators of nonlinear dynamical systems. In 2019 American Control Conference (ACC),
pages 4832–4839. IEEE, 2019.

[99] M. Yin, E. Ban, B. V. Rego, E. Zhang, C. Cavinato, J. D. Humphrey, and G. Em Karniadakis.
Simulating progressive intramural damage leading to aortic dissection using DeepONet: an
operator–regression neural network. Journal of the Royal Society Interface, 19(187):20210670,
2022.

[100] M. Yin, E. Zhang, Y. Yu, and G. E. Karniadakis. Interfacing finite elements with deep neural
operators for fast multiscale modeling of mechanics problems. Comput. Methods Appl. Mech.
Engrg., 402:115027, 2022.

[101] E. Zappala, A. H. d. O. Fonseca, A. H. Moberly, M. J. Higley, C. Abdallah, J. A. Cardin, and
D. van Dijk. Neural integro-differential equations. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 37, pages 11104–11112, 2023.

[102] J. Zech. Sparse-grid approximation of high-dimensional parametric PDEs. PhD thesis, ETH
Zurich, 2018.

[103] Z. Zhang, L. Wing Tat, and H. Schaeffer. Belnet: basis enhanced learning, a mesh-free neural
operator. Proc. R. Soc. A, 479(2276):20230043, 2023.

[104] M. Zhu, H. Zhang, A. Jiao, G. E. Karniadakis, and L. Lu. Reliable extrapolation of deep neural
operators informed by physics or sparse observations. Comput. Methods Appl. Mech. Engrg.,
412:116064, 2023.

[105] Z. Zou, X. Meng, A. F. Psaros, and G. E. Karniadakis. NeuralUQ: A comprehensive library
for uncertainty quantification in neural differential equations and operators. SIAM Rev., 66(1):
161–190, 2024.

16


---Page Break---
A
Experimental setup

In this section, we describe our experimental setup.

A.1
Formulation of the learning problems

We first explain how all our experiments are formulated as operator learning problems. We do this in
a standard way, by first truncating the random field which generates the measure µ, and then using an
FEM to discretize the output space.

All our examples follow Example (2.3) and involve operators of the form (2.6), where Fa represents
a different type of PDE in each case (specifically, either an elliptic diffusion, Navier-Stokes-Brinkman
or Boussinesq PDE). We consider both affine (2.8) and log-transformed parametrizations of the
random field a(x). Thus, in general we write

a(x) = a(·; x) = g

 

a0(·) +

∞
X

i=1
cixiϕi(·)

!

,
(A.1)

where g : R →R is a (measurable) map. In the affine case g(t) = 1. In the log-transformed case
g(t) = exp(t).

As discussed, since the main objective of this work is to examine the approximation error, we
set up our experiments so that encoding-decoding errors are zero. We do this as follows. First,
we fix a parametric dimension d and truncate the expansion in (A.1) after d terms, giving a map
ad : [−1, 1]d →X and measure µ = ad♯ϱd, where ϱd is the uniform probability measure on [−1, 1]d.
We then define the operator F as F(ad(x)) = f(x) = u(·; ad(x)), where u(·; a) is the solution of
the PDE Fau = 0.

In alignment with our theorems, we focus on the in-distribution performance of the learned approx-
imation bF. This means we define the encoder only on supp(µ), as EX (X) = x when X = ad(x)
with x ∈[−1, 1]d. As a result, the encoding-decoding error EX,q in (3.7) satisfies EX,q = 0.

To perform our simulations, we use FEMs to solve the PDE and discretize the output space Y. Let
{φi}K
i=1 ⊂Y be a FEM basis and set dY = K. Then we define the decoder as

DY : RK →Y,
DY(c) =

K
X

i=1
ciφi
(A.2)

and set eY = DY(RK) as the discretization of Y.

With this in hand, we now describe the simulation of training data in general terms. First, we draw
x1, . . . , xm ∼i.i.d. ϱd. Then, for each training sample xi, we compute Yi by using the FEM to solve
the PDE with parameter Xi = ad(xi). Notice that Yi ∈eY in this setup.

We consider a DNN architecture with input dimension n0 = d and output dimension nL+2 = K.
After training, we evaluate the learned approximation bF = DY ◦b
N ◦EX over supp(µ) as bF(X) =
DY ◦b
N ◦EX (X) = DY ◦b
N(x) for X = ad(x) with x ∈[−1, 1]d. Finally, as noted, we us the same
FEM discretization to generate testing data, which allows us to measure the error with respect to
the L2
µ(X; eY)-norm. This effectively means that the encoding-decoding error EY,q in (3.7) satisfies
EY,q = 0 as well.

A.2
Computational setup for the numerical experiments

In this work, we investigate the trade-off between the accuracy of the learned operator and the number
of samples m used in training. Our methodology is summarized as follows.

(i) Implementation. We use the open-source finite element library FEniCS, specifically version
2019.1.0 [9], and Google’s TensorFlow version 2.12.0. More information about TensorFlow can
be found at https://www.tensorflow.org/.
(ii) Hardware. We train the DNN models in single precision on the Digital Research Alliance of
Canada’s Cedar compute cluster (see https://docs.alliancecan.ca/wiki/Cedar), using

17


---Page Break---
Intel Xenon Processor E5-2683 v4 CPUs with either 125GB or 250GB per node. The setup for each
of our PDEs is as follows. For each experiment we consider training with 14 sets of points of size
m ∈{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500} and for 6 different architectures
(4 x 40 and 10 x 100 with ReLU, ELU, and tanh activations) over two parametric dimensions
(d = 4 and d = 8) and two coefficients (a1,d from (B.1) and a2,d from (B.2)), giving 336 DNNs to
be trained for each trial. For the Poisson PDE and Navier-Stokes-Brinkman PDEs we run 12 trials.
For the Boussinesq PDE we run 8 trials due to the larger problem size. For the Poisson PDE, we
allocate 336 nodes with 1 × 32 core CPUs running 4 threads (totalling 1344 threads, 6.8 GB RAM
per node) each running 3 trials, taking approximately 4 hours and 15 minutes to complete. For the
Navier-Stokes-Brinkman PDE, we use the same setup allocating 9.88 GB RAM per node and the
runs take approximately 9 hours and 13 minutes for each of the two components of the solution
to complete. For the Boussinesq PDE, we allocate 336 nodes with 1 × 32 core CPUs running
4 threads per node (totaling 1344 threads, 10 GB RAM per node) each running 2 trials, taking
approximately 12 hours and 32 minutes for each of the 3 components to complete. Given this, the
total time required to reproduce the results in parallel with the above setup is approximately 60
hours or 2.5 days. Results were stored locally on the cluster and the estimated total space used
to store the data for testing and training and results from computation is approximately 50 GB.
Trained models were not retained due to space limitations on the cluster.
(iii) Choice of architectures and initialization. Based on the strategies in [1], we fix the number of
nodes per layer N and depth L such that the ratio β := L/N is β = 0.5. In addition, we initialize
the weights and biases using the HeUniform initializer from keras setting the seed to the trial
number. We consider the Rectified Linear Unit (ReLU)

σ1(z) := max{0, z},

hyperbolic tangent (tanh)

σ2(z) := ez −e−z

ez + e−z ,

or Exponential Linear Unit (ELU)

σ3(z) =
z
z > 0,
ez −1
z ≤0

activation functions in our experiments.
(iv) Optimizers for training and parametrization. To train the DNNs, we use the Adam optimizer
[47], incorporating an exponentially-decaying learning rate. We train our models for 60,000 epochs
or until converging to a tolerance level of ϵtol = 5 · 10−7 in single precision. In light of the
nonmonotonic convergence behavior observed during the minimization of the nonconvex loss (see,
e.g., [1, 2]), we implement early stopping. More precisely, we save the weights and biases of the
partially trained network once the ratio between the current loss and the last checkpoint loss is
reduced below 1/8, or if the current weights and biases produce the best loss value observed in
training. We then restore these weights after training only if the loss value of the current weights is
larger than that of the saved checkpoint.
(v) Training data and design of experiments. First, we define a ‘trial’ as a complete training run for
a DNN approximating a specific function, initialized as mentioned above.
Following the setup of §A.1, we run several trials solving the problem:

Given training data {(Xi, Yi)}m
i=1 ⊂(X × Y)m, Xi ∼i.i.d. µ, Yi = F(Xi) + Ei ∈Y,

approximate F ∈L2
µ(X; Y).

We generate the measurements Yi using mixed variational formulations of the parametric ellip-
tic, Navier-Stokes-Brinkman and Boussinesq PDEs discretized using FEniCS with input data
Xi. The noise Ei ∈Y encompasses the discretization errors from numerical solution. Further
details of the discretization can be found in §B. Each of our architectures is trained across a
range of datasets with increasing sizes. This involves using a set of training data consisting
of values {(Xi, Yi))}m
i=1, where m denotes the size of the training data and belongs to the set
{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500}. After training we calculate the testing
error for each trial and run statistics across all trials for each dataset.
(vi) Testing data and error metric. The testing data is generated similarly to the training data,
obtaining solutions at different points Xi ∈X for i = 1, . . . , mtest. However, the testing data

18


---Page Break---
{(Xi, Yi = F(Xi)+Ei)}mtest
i=1
is generated using a deterministic high-order sparse grid collocation
method [76]. In particular, we use sparse grid quadrature rules to compute approximations to the
Bochner norms

∥F∥L2µ(X; e
Y) =
Z

X
∥F(X)∥2
e
Y dµ(X))
1/2
≈

 mtest
X

i=1
∥F(Xi)∥2
e
Ywi

!1/2

,

where µ = ad♯ϱd is the pushforward measure defined in (A.1) and wi, i = 1, . . . , mtest are the
quadrature weights. We use these approximations to compute the relative L2
µ(X; eY) error

etest
F
=

Pmtest
i=1
∥F(Xi) −bF(Xi)∥2
e
Ywi
1/2

Pmtest
i=1
∥F(Xi)∥2
e
Ywi
1/2
.

We use a high order isotropic Clenshaw Curtis sparse grid quadrature rule to evaluate etest
F
, as
described in [2]. This method shows superior convergence over Monte Carlo integration to evaluate
the global Bochner error. The sparse grid rule gives mtest points at a level ℓfor d dimensions. We
rely on the TASMANIAN sparse grid toolkit [88–90] for the generation of the isotropic rule to study
the generalization performance of the DNN.
(vii) Visualization. The graphs in Figs. 1-3 show the geometric mean (the main curve) and plus/minus
one (geometric) standard deviation (the shaded region). We use the geometric mean because our
errors are plotted in logarithmic scale on the y-axis. See [3, Sec. A.1] for further discussion about
this choice.

B
Description of the parametric PDEs used in the numerical experiments and
their discretization

In this section, we provide full details of the parametric PDEs considered in our numerical experiments.
We also describe their variational formulations and numerical solution using FEM.

B.1
Parametric coefficients

We consider two parametric coefficients of the form (A.1) in our numerical experiments. Our first
coefficient is

a1(z, x) = 2.62 +

∞
X

j=1
xj
sin(πz1j)

j3/2
,
∀z ∈Ω,
(B.1)

where xj ∈[−1, 1], ∀j. Our second example involves a log-transformed coefficient, which is a
rescaling of an example from [76] of a diffusion coefficient with one-dimensional (layered) spatial
dependence given by

a2(z, x) = exp



1 + x1

√πβ

2

1/2
+

∞
X

j=2
ζjϑj(z)xj



,
∀z ∈Ω,

ζj := (√πβ)1/2 exp

 
−(⌊j/2⌋πβ)2

8

!

,

ϑj(z) :=
sin (⌊j/2⌋πz1/βp)
if j is even
cos (⌊j/2⌋πz1/βp)
if j is odd ,

(B.2)

where xj ∈[−1, 1], ∀j. Here we let βc = 1/8, and βp = max{1, 2βc}, β = βc/βp.

In both cases we consider truncation of the expansion after d terms, giving the map aj,d : [−1, 1]d →
X, with j = 1 corresponding to (B.1) and j = 2 corresponding to (B.2). Our input samples are
then {Xi = aj,d(xi)}m
i=1 ⊂X with xi ∈[−1, 1]d drawn identically and independently from ϱd and
j ∈{1, 2}. Note for the Boussinesq problem we also consider an additional parametric dependence
in the tensor K describing the thermal conductivity of the fluid. See §B.5 and (B.17).

19


---Page Break---
B.2
Relevant spaces

We require several function space definitions for the development of the mixed variation formulations
of the Poisson, Navier-Stokes-Brinkman and Boussinesq PDEs. Let Ω⊂Rn, n ∈{2, 3}, be the
physical domain of a PDE. We write Lp(Ω), 1 ≤p ≤∞, for the Lp-space of scalar-valued functions
with respect to the Lebesgue measure (to avoid confusion with the Bochner space L2
µ(X; Y)). We
denote the standard Sobolev spaces as Ws,p(Ω) for s ∈R and p > 1, and write Hk(Ω) when
p = 2 and s = k. Additionally, we consider the space of traces of functions in H1(Ω), denoted by
H1/2(∂Ω), and its dual, H−1/2(∂Ω) (see, e.g., [11, Sec. 1.2] for further details). We also define the
following closed subspace of H1(Ω):

H1
0(Ω) := C∞
0 (Ω)
∥·∥1,Ω.

Here C∞
0 (Ω)
∥·∥1,Ωdenotes the closure of C∞
0 (Ω) (i.e., the space of C∞(Ω) functions with compact
support) with respect to the norm ∥· ∥1,Ω, which, for any v ∈H1(Ω), is given by

∥v∥1,Ω:=
n
|v|2
1,Ω+ ∥v∥2
L2(Ω)
o1/2
,
where |v|1,Ω:= ∥∇v∥L2(Ω).

For scalar functions u and vector fields v, we use ∇u and div(v) to denote their gradient and
divergence, respectively. For tensor fields σ and τ, represented by (σi,j)n
i,j=1 and (τi,j)n
i,j=1,
respectively, we define div(σ) as the divergence operator div acting along the rows of σ, and we
define the trace and the tensor inner-product as

tr(σ) =

n
X

i=1
σi,i, and τ : σ =

n
X

i,j=1
τi,jσi,j,

respectively. Furthermore, we introduce the notation Lp(Ω) and Lp(Ω) to represent the vectorial and
tensorial counterparts of Lp(Ω), respectively, and H1(Ω) and H1(Ω) for the vectorial and tensorial
counterparts of H1(Ω), respectively. Keeping this in mind, we introduce the Banach spaces

H(divq; Ω) :=
n
v ∈L2(Ω) : div(v) ∈Lq(Ω)
o
,

H(divq; Ω) :=
n
τ ∈L2(Ω) : div(τ) ∈Lq(Ω)
o
(B.3)

with norms

∥v∥H(divq;Ω) := ∥v∥L2(Ω) + ∥div(v)∥Lq(Ω),

∥τ∥H(divq;Ω) := ∥τ∥L2(Ω) + ∥div(τ)∥Lq(Ω) .

The cases of q = 4/3 and q = 2 appear in the mixed variational formulations of the considered PDEs,
and for the latter we simply write H(div; Ω).

Often, under certain conditions, such as incompressibility conditions [33, eq.(2.4)], it is convenient to
define variants of these spaces. For example, we define

L2
tr(Ω) :=
n
τ ∈L2(Ω) : tr(τ) = 0
o
,
(B.4)

which represents the space of integrable functions with zero trace over Ω. Furthermore, given the
decomposition (see, e.g., [32])

H(div4/3; Ω) = H0(div4/3; Ω) ⊕R I ,
(B.5)

we may also consider

H0(div4/3; Ω) :=
n
τ ∈H(div4/3; Ω) :
Z

Ω
tr(τ) = 0
o
,
(B.6)

as the space of elements in H(div4/3; Ω) with zero mean trace. Finally, we define

L2
skew(Ω) = {η ∈L2(Ω) : η + ηt = 0},
(B.7)

and the space of L2(Ω) functions with zero integral over Ωas

L2
0(Ω) =

ν ∈L2(Ω) :
Z

Ω
ν = 0

.
(B.8)

20


---Page Break---
B.3
The parametric diffusion equation

We now describe our first example, which is the parametric elliptic diffusion equation. Let Ω⊂R2

be a bounded Lipschitz domain, ∂Ωbe the boundary of Ω, f ∈L2(Ω) and g ∈H1/2(∂Ω). Given
x ∈[−1, 1]d, we consider the PDE

−div(a(x)∇u(x)) = f,
in Ω
(B.9)
u(x) = g,
on ∂Ω.

Here a(x) = a(·, x) ∈L∞(Ω) =: X is the parametric diffusion coefficient. The terms f and g are
not parametric.

B.3.1
Mixed variational formulation

Our first step in precisely defining the problem is to identify sufficient conditions for the solution map
x 7→u(x) to be well-defined. To do this, we diverge from the standard variational formulation in-
volving the space Y = H1
0(Ω) (see Remark B.3) and instead consider a mixed variational formulation
of (B.9). Using a mixed formulation to study the solution of PDEs offers several benefits over the
standard variational formulation. One key advantage is that it allows us to introduce additional vari-
ables that can be of physical interest. Additionally, mixed formulations can naturally accommodate
different types of boundary conditions and introduce Dirichlet boundary conditions directly into the
formulation rather than imposing them on the search space. For further details on mixed formulations,
we refer to [32] and references within.

Assume that there exists r, M > 0 such that, for all x ∈[−1, 1]d,

0 < r ≤essinfz∈Ωa(z, x) =: amin(x) and amax(x) := esssupz∈Ωa(z, x) ≤M.
(B.10)

Then the problem can be recast as a first-order system: given x ∈[−1, 1]d, find (σ, u)(x) ∈
H(div; Ω) × L2(Ω) such that

da(x)(σ(x), τ) + b(τ, u(x)) = G(τ),
∀τ ∈H(div; Ω),

b(σ, v) = F(v),
∀v ∈L2(Ω).
(B.11)

Here d and b are the bilinear forms defined by

da(x)(σ, τ) =
Z

Ω

σ · τ

a(x) ,
∀(τ, σ) ∈H(div; Ω) × H(div; Ω),

b(τ, v) =
Z

Ω
div(τ)v,
∀(τ, v) ∈H(div; Ω) × L2(Ω)

and the functionals G ∈(H(div; Ω))′ and J ∈(L2(Ω))′ are defined by

J(v) = −
Z

Ω
fv,
∀v ∈L2(Ω) and G(τ) = ⟨γ(τ) · n, g⟩1/2,∂Ω,
∀τ ∈H(div; Ω).
(B.12)

For the experiments in this work, we consider Ω= (0, 1)2 and f = 10. For the boundary condition,
we consider a constant value u(z, x) = 0.5 on the bottom of the boundary (0, 1) × {0}, and zero
boundary conditions on the remainder of the boundary.

B.3.2
Holomorphy assumption

Consider the affine parametrization (B.1). Setting M = 2.7 and observing that


X

j∈N
xj
sin(πz1j)

j3/5


≤
X

j∈N

1
j3/5 ≈2.61238 = 2.62 −r,
(B.13)

for some r < 0.00762, we deduce that (B.10) holds, which makes the mixed variational
formulation (B.11) well defined, i.e., for each x ∈[−1, 1]∞, there exists a unique solution
(σ, u)(x) ∈H(div; Ω) × L2(Ω). Moreover, one can show that the parametric solution map
x 7→(σ, u)(x) is (b, ε)-holomorphic for 0 < ε < 0.00762 and where b = (bi)∞
i=1 is given by
bj = ∥sin(πj·)/j3/2∥L∞(Ω) = j−3/2. See [74, Prop. A.3.2].

21


---Page Break---
Figure 4: The domain Ωand FE mesh for the parametric diffusion equation.

Figure 5: The solution u(x) of the parametric Poisson problem in (B.9) for a given parameter x = (1, 0, 0, 0)⊤

with affine coefficient a1,d and d = 4, using a total of K = 2622 DoF. The left plot shows the solution given by
the FEM solver. The right plot show the ELU 4 × 40 DNN approximation after 60, 000 epochs of training with
m = 500 sample points for training.

In view of this property, this example falls within our theory. Note that b ∈ℓp
M(N) for every p < 2/3.
Thus, we expect a theoretical rate of convergence with respect to the amount of training data that is
arbitrarily close to m1/2−3/2 = m−1. This holomorphy result applies to the affine diffusion (B.1),
not the log-transformed diffusion (B.2). However, we expect that it is possible to extend [74, Prop.
A.3.2] to the latter case.

B.3.3
Finite element discretization

We use so-called conforming Finite Element (FE) discretizations [18, Chp. 3]. Given a number
of Degrees of Freedom (DoF) K, this results in finite-dimensional spaces HK ⊆H(div; Ω) and
QK ⊆L2(Ω). Specifically, we consider a regular triangulation TK of Ωmade up of triangles of
minimum diameter hmin = 0.0844 and maximum diameter hmax = 0.1146. This corresponds to a
total number of DoF K = 2622. See Fig. 4 for an illustration of the FE mesh.

Fig. 5 shows a comparison between a reference solution computed by the FEniCS FEM solver and
the approximation obtained by an ELU 4 × 40 DNN.

B.4
Parametric Navier-Stokes-Brinkman equations

We next consider a parametric model describing the dynamics of a viscous fluid through porous
media. Consider a bounded and Lipschitz physical domain Ω⊆R2. Given x ∈[−1, 1]d, we
consider the incompressible nonlinear stationary Navier-Stokes-Brinkman (NSB) equations: find

22


---Page Break---
u : [−1, 1]d × Ω→R2 and p : [−1, 1]d × Ω→R such that

ηu −λdiv(a(x)e(u(x))) + (u(x) · ∇)u(x) + ∇p(x) = f
in Ω
div(u(x)) = 0
in Ω
(B.14)

u =
uD
on ∂Ωin
0
on ∂Ωwall
(a(x)∇e(u) −pI)ν = 0
on ∂Ωout
Z

Ω
p = 0,

where λ = Re−1 and Re is the Reynolds number, a(x) = a(·; x) ∈X := L∞(Ω) and a :
[−1, 1]d × Ω→R+ is the random viscosity of the fluid, η ∈R+ is the scaled inverse permeability
of the porous media, u is the velocity of the fluid, e(u) = 1

2(∇u + (∇u)t) is the symmetric part
of the gradient, p is the pressure of the fluid and f : Ω→R is an external force independent of the
parameters. Here, the fourth condition imposes a zero normal Cauchy stress

(a(x)∇e(u) −pI)ν = 0

for the output boundary ∂Ωout.In addition, the incompressibility of the fluid imposes the following
compatibility condition on uD:
Z

Γ
uD · n = 0
on ∂Ωin.

The third condition also imposes a no-slip condition on the walls Ωwall [34, eq.(2.3)].

B.4.1
Mixed variational formulation

The analysis of the detailed mixed formulation used for this problem in the nonparametric case can
be found in [34]. Over the last decade, many works have used a mixed formulation employing a
Banach space framework, allowing one to solve different PDEs in continuum mechanics in suitable
Banach spaces. The advantage of this formulation is that no augmentation is required, the spaces are
simpler and closer to the original model, and it allows one to obtain more direct approximations of
the variables of physical interest [34, Sec. 1].

Based on the analysis in [34], the mixed variational formulation of the parametric NSB equations
in (B.14) becomes: given x ∈[−1, 1]d, find (u, t, σ, γ)(x) ∈L4(Ω) × L2
tr(Ω) × H0(div4/3; Ω) ×
L2
skew(Ω) such that

λ
Z

Ω
ai(x)t(x) : s −
Z

Ω
s : σ(x) −
Z

Ω
(u ⊗u)(x) : s
=
0,
Z

Ω
t(x) : τ +
Z

Ω
γ(x) : τ +
Z

Ω
u(x) · div(τ)
=
⟨τn, uD⟩∂Ωin,
Z

Ω
δ : σ(x) +
Z

Ω
v · div(σ(x)) −
Z

Ω
ηu(x) · v
=
Z

Ω
f · v,

(B.15)

for all (v, s, τ, δ) ∈L4(Ω)×L2
tr(Ω)×H0(div4/3; Ω)×L2
skew(Ω). Numerically, the skew-symmetry
of γ is imposed by searching for γ ∈L2(Ω) and setting

γ =

0
γ
−γ
0


.

Moreover, we impose the Neumann boundary condition via a Nietsche method as in [34, Sec. 5.2].
Specifically, we add
κ⟨(σ + u ⊗u)n), τn⟩∂Ωout = 0

to the second equation where κ ≫1 is a large constant (e.g., κ = 104). As usual in this formulation,
the pressure p ∈L2(Ω) can be computed according to the post-processing formula

p = −1

2tr(σ + (u ⊗u)).

23


---Page Break---
Figure 6: The solution (u, p)(x) of the parametric NSB problem (B.14) for a given parameter x = (1, 0, 0, 0)⊤

with affine coefficient a1,d and d = 4, using a total of 1464 DoF for u and 244 DoF for p. The top row shows
the solution given by the FEM solver, and the bottom row shows the ELU 4 × 40 DNN approximation after
60, 000 epochs of training with m = 500 sample points. The left plots show the vector field u. The right plots
show the points of highest pressure p.

Note that above we omitted the term x for simplicity.

In our experiments, we consider approximating solutions to the parametric NSB problem with
λ = 0.1, a scaled inverse permeability of η = 10 + z2
1 + z2
2, an external force f = (0, −1)⊤and
random viscosity aj,d as in (B.1)–(B.2) with j ∈{1, 2}.

We consider the unit square Ω= (0, 1)2 as the domain, an inlet boundary ∂Ωin = (0, 1) × {1}, an
outlet boundary ∂Ωout = {1}×(0, 1) and walls ∂Ωwall = {0}×(0, 1)∪(0, 1)×{0}. For simplicity,
we use the same mesh as that of the previous example. See Fig. 4. On the Neumann boundary ∂Ωout
we consider a zero normal Cauchy stress, a Dirichlet condition uD = (0.0625)−1((z2 −0.5)(1 −
z2), 0) on ∂Ωin and a no-slip velocity on ∂Ωwall.

Fig. 6 provides a comparison between a reference solution of the vector field u and pressure p
computed by the FEniCS FEM solver and the approximation generated by a ELU 4 × 40 DNN.

Remark B.1 (Other auxiliary variables) We report the performance of the DNNs approximating
(u, p)(x) ∈L4(Ω) × L2(Ω). Note that any solver based on the above formulation outputs several
other variables, e.g., (t, σ, γ)(x) ∈L2
tr(Ω)×H0(div4/3; Ω)×L2
skew(Ω). One could also approximate
these auxiliary variables using DNNs. However, we restrict our experiments to (u, p) as these are the
primary variables of interest in the problem.

To conclude this discussion, in Fig. 7 we plot the numerical results for the approximation of the
pressure p in the above problem. This complements Fig. 2, which showed results for the velocity field
u. We once more observe similar results: ELU and tanh DNNs outperform ReLU DNNs, the rate of
convergence appears to be close to O(m−1) and there is no degradation with increasing parametric
dimension d.

B.5
Parametric stationary Boussinesq equation

To recap, in our first example, we considered a mixed formulation of a parametric diffusion equation
that provably satisfies the (b, ε)-holomorphy assumption. Using this formulation, we considered

24


---Page Break---
101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

Figure 7: The same as Fig. 2, except showing results for the pressure p.

problems with nonzero Dirichlet boundary conditions, whereas previous works [2, 19, 27] study
the more restrictive case of homogeneous Dirichlet boundary conditions, where u ∈H1
0(Ω). In
our next example, we studied a more complicated parametric PDE, namely, the parametric NSB
equations. While this example currently lacks a holomorphy guarantee, we observe a convergence
rate that aligns with what we expect. We conjecture that the m−1 rate holds both for this and for even
more complicated problems. To illustrate this claim with an example, we now consider a parametric
coupled partial differential equation in three dimensions (Ω⊂R3) with two random coefficients
affecting different parts of the coupled problem. The nonparametric version of this problem is based
on [20].

Specifically, we consider the Boussinesq formulation in [20] that combines a parametric incompress-
ible Navier–Stokes equation with a parametric heat equation. The parametric dependence affects
both equations. The Navier–Stokes equation is affected by a parametric variable multiplying the
temperature-dependent viscosity, and the equation for heat flow is affected directly by the thermal
conductivity of the fluid. To be more precise, given x ∈[−1, 1]d, our goal is to find the velocity
u : [−1, 1]d × Ω→R2, pressure p : [−1, 1]d × Ω→R and temperature φ : [−1, 1]d × Ω→R of a
fluid such that

−div(2a(x)ϖ(φ(x))e(u(x))) + (u(x) · ∇)u(x) + ∇p(x) = φ(x)g
in Ω,
div(u(x)) = 0
in Ω,
(B.16)
−div(K(x)∇φ(x)) + u(x) · ∇φ(x) = 0
in Ω,
u = uD
on ∂Ω,
φ = φD
on ∂Ω,
Z

Ω
p(x) = 0.

Here g = (0, 0, −1)⊤is a gravitational force and K(x) = K(·; x) ∈L∞(Ω), where K : [−1, 1]d ×
Ω→R3×3 is a parametric uniformly-positive tensor describing the thermal conductivity of the fluid.
It is given explicitly by

K(z, x) =



1.89 +
X

j∈N
xj
sin(πz3j)

j9/5





"exp(−z1)
0
0
0
exp(−z2)
0
0
0
exp(−z3)

#

, ∀z ∈Ω, (B.17)

for x ∈[−1, 1]d. The term ϖ : R →R+ is a temperature-dependent viscosity given by ϖ(φ) =
0.1 + exp(−φ) and the term a(x) = a(·; x) ∈L∞(Ω), where a : [−1, 1]d × Ω→R is a parametric
variable affecting the viscosity of the fluid. As in the previous example e(u) is the symmetric part of
∇u. Note that in this case, we have (a(x), K(x)) ∈X := L∞(Ω) × L∞(Ω).

B.5.1
Fully mixed variational formulation

The complete derivation of a fully-mixed variational formulation for the non-parametric Boussinesq
equation in Banach spaces can be found in [20, Sec. 3.1]. To make the presentation simpler we
rewrite it for the parametric case. Given x ∈[−1, 1]d, find (u, t, σ, φ, ˜t, ˜σ)(x) ∈L4(Ω)×L2
tr(Ω)×

25


---Page Break---
H0(div4/3; Ω) × L4(Ω) × L2(Ω) × H(div4/3; Ω) such that

−
Z

Ω
v · div(σ(x)) + 1

2

Z

Ω
t(x)u(x) · v −
Z

Ω
φ(x)g · v
=
0
Z

Ω
2a(x)ϖ(φ(x))tsym(x) : s −1

2

Z

Ω
(u ⊗u)(x) : s
=
Z

Ω
σ(x) : s
Z

Ω
τ : t(x) +
Z

Ω
u(x) · div(τ)
=
⟨τν, uD⟩Γ

−
Z

Ω
ψ div(˜σ(x)) + 1

2

Z

Ω
ψ(x)u(x) · ˜t
=
0
Z

Ω
K(x)˜t(x) · ˜s −1

2

Z

Ω
φ(x)u(x) · ˜s
=
Z

Ω
˜σ(x) · ˜s
Z

Ω
˜τ · ˜t(x) +
Z

Ω
φ(x) div(˜τ)
=
⟨˜τ · ν, φD⟩Γ
Z

Ω
tr(2σ + u ⊗u)(x)
=
0,

(B.18)

for all (v, s, τ, ψ, ˜σ, ˜τ) ∈L4(Ω) × L2
tr(Ω) × H0(div4/3; Ω) × L4(Ω) × L2(Ω) × H(div4/3; Ω).
Here p ∈L2
0(Ω) can be recovered as

p = −1

6tr(2σ + 2cI + u ⊗u),
where c = −1

6|Ω|

Z

Ω
tr(u ⊗u).
(B.19)

As in the previous example, we omitted the term x from this equation for simplicity. For further
details on this formulation we refer to [20] and references within.

Given x ∈[−1, 1]d, we approximate the solution (u, p, φ)(x) ∈(L4(Ω) × L2
0(Ω) × L4(Ω)) of
(B.18) by using DNNs and study the approximation capabilities as we increase the number of training
samples m. As in the previous example (see Remark B.1), we do not aim to approximate the other
variables (t, σ, ˜t, ˜σ)(x) ∈L2
tr(Ω) × H0(div4/3; Ω) × L2(Ω) × H(div4/3; Ω).

In our experiments, we consider the unit cube Ω= (0, 1)3 as the domain in R3. We consider a nonzero
boundary condition uD = (1, 1, 0) on the bottom face of the cube ∂Ωbottom = (0, 1) × (0, 1) × {0},
and zero on the other faces. We set φD = exp(4(−(z1 −0.5)2 −(z2 −0.5)2)) on ∂Ωbottom and
zero otherwise. For simplicity, we consider the same parametric coefficients a1,d and a2,d given
by (B.1) and (B.2), respectively. See Fig. 8 for an example of the solution (u, p, φ)(x) for a given
x ∈[−1, 1]4.

To conclude this section, we provide a comparison of the performance of the DNN architectures in
approximating the velocity field u and pressure p for the Boussinesq PDE. This complements Fig. 3,
which showed results for the temperature φ. As with φ, the convergence rate for p agrees roughly
with the rate m−1, and does not appear to deteriorate with the parametric dimension d. On the other
hand, the convergence rate for the velocity field u is somewhat slower.

C
Overview of the proofs

In this section, we first introduce additional notation that is needed for the proofs of the main results.
We then give a brief overview of the proofs.

C.1
Additional notation

C.1.1
Lipschitz constants

Let (X, ∥· ∥X ) and (Y, ∥· ∥Y) be Banach spaces, G : X →Y and B ⊆X. We define the Lipschitz
constant as L = Lip(G; B, Y) as the smallest constant L ≥0 such that

∥G(X′) −G(X)∥Y ≤L∥X′ −X∥X ,
∀X, X′ ∈B.

26


---Page Break---
Figure 8: The solution (u, φ, p)(x) to the parametric Boussinesq problem in (B.18) for a given parameter
x = (1, 0, 0, 0)⊤with affine coefficient a1,d and d = 4, using a total of 18, 480 DoF for u and 528 DoF for
both φ and p. The top row shows the solution given by the FEM solver and the bottom row shows the 4 × 40
ELU–DNN approximation after 60, 000 epochs of training with m = 500 sample points. The left plots show
streamlines of the vector field u and their directions indicated with coloured arrows. The middle plots visualize
the temperature distribution inside the cube using coloured spheres, with the hottest region at the centre of the
cube. The right plots illustrate the points of highest pressure p.

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

101
102
10-3

10-2

10-1

100

Figure 9: The same as Fig. 3, except showing results for the velocity field u, where Y = L4(Ω), and pressure p,
where Y = L2
0(Ω).

C.1.2
Sequence spaces

We require some notation for sequences. Let (Z, ∥· ∥Z) be a Banach space, d = N ∪{∞} and write
ν = (νk)d
k=1 for an arbitrary multi-index in Nd
0. If Λ ⊆Nd
0 is a finite or countable set of multi-indices
and 0 < p ≤∞we define the space ℓp(Λ; Z) as the set of all Z-valued sequences c = (cν)ν∈Λ for
which ∥c∥p;Z < ∞, where

∥c∥p;Z =

( P
ν∈Λ ∥cν∥p
Z
1/p
0 < p < ∞,
supν∈Λ ∥cν∥Z
p = ∞.

27


---Page Break---
When (Z, ∥· ∥Z) = (R, | · |), we just write ℓp(Λ) and ∥· ∥p. We also define the following:

|||c|||p;Z =
sup
z∗∈B(Z∗)
∥z∗(c)∥p,
∀c ∈ℓp(Λ; Z),

where z∗(c) = (z∗(cν))ν∈Λ ∈R|Λ| for c = (cν)ν∈Λ. Notice that |||c|||p;Z ≤∥c∥p;Z.

Given a sequence c = (cν)ν∈Λ, we write

supp(c) = {ν ∈Λ : cν ̸= 0} ⊆Λ.

For d ∈N ∪{∞}, we write ej, j = 1, . . . , d, for the canonical basis vectors in Rd. We also write 0
for the multi-index of zeros and 1 for the multi-index of ones. Finally, for multi-indices ν = (νk)d
k=1
and µ = (µk)d
k=1, we write ν ≥µ to mean νk ≥µk, ∀k and likewise for ν > µ.

C.1.3
Weights and weighted sequence spaces

Let d ∈N ∪{∞}, Λ ⊆Nd
0 and w = (wν)ν∈Λ ≥0 be a sequence of nonnegative weights. We define
the weighted cardinality of a set S ⊆Λ as

|S|w =
X

ν∈S
w2
ν.
(C.1)

Given a sequence c = (cν)ν∈Λ we write

∥c∥0,w = |supp(c)|w.

Next, for a Banach space (Z, ∥· ∥Z) and 0 < p ≤2, we define the weighted ℓp
w(Λ; Z) space as the
space of Z-valued sequences c = (cν)ν∈Λ for which ∥c∥p,w;Z < ∞, where

∥c∥p,u;Z =

 X

ν∈Λ
w2−p
ν
∥cν∥p
Z

!1/p

.

Notice that ∥· ∥2,u;Z coincides with the unweighted norm ∥· ∥2;Z.

C.1.4
Legendre polynomials

We write {Pn}n∈N0 for the classical Legendre polynomials on [−1, 1] with normalization Pn(1) = 1.
Since
R 1
−1 |Pn(x)|2 dx = (n + 1/2)−1, we define the orthonormal (with respect to the uniform
probability measure) Legendre polynomials as

ψn(x) =
√

2n + 1Pn(x),
x ∈[−1, 1], n ∈N0.

Let D = [−1, 1]N as before,

F =

ν = (νi)∞
i=1 ∈NN
0 : |supp(ν)| < ∞
	

be the set of multi-indices with finitely-many nonzero terms and define the multivariate Legendre
polynomials as

Ψν(x) =
Y

i∈N
ψνi(yi) ≡
Y

i∈supp(ν)
ψνi(xi),
∀x ∈D, ν = (νi)∞
i=1 ∈F.

Here, the second equality follows from the fact that ψ0 = 1. Then it is known that the set

{Ψν}ν∈F ⊂L2
ϱ(D)
(C.2)

constitutes an orthonormal basis for L2
ϱ(D) [19, §3]. For later use, we also define the sequences

u = (uν)ν∈F,
where uν = ∥Ψν∥L∞
ϱ (D) =
Y

k∈N

√

2νk + 1, ∀ν ∈F,
(C.3)

and
v = (vν)ν∈F,
where vν = u5+ξ
ν
, ∀ν ∈F.
(C.4)
Here ξ ≥0 will be chosen later in the proof.

28


---Page Break---
C.1.5
Miscellaneous

Given an optimization problem mint f(t), we say that ˆt is a (σ, τ)-approximate minimizer for some
σ ≥1 and τ ≥0 if f(ˆt) ≤σ2 mint f(t) + τ 2. (In the main paper, we consider only σ = 1, but for
the proofs it is useful to allow σ > 1).

Finally, for convenience, given X1, . . . , Xm ∈X, we define the semi-norm

|||G|||disc,µ =

v
u
u
t 1

m

m
X

i=1
∥G(Xi)∥2
Y

of an operator G : X →Y and

∥g∥disc,˜ς = |||g ◦EX |||disc,µ =

v
u
u
t 1

m

m
X

i=1
∥g ◦EX (Xi)∥2
Y

of a function g : RN →Y. Here, as in (A.III), ˜ς denotes the pushforward measure EX #µ.

C.2
Overview of the proofs

C.2.1
Theorem 3.1

Theorems 3.1-3.2 are based on polynomials, and specifically, procedures for learning efficient
Legendre polynomial approximations to holomorphic operators. As observed, these results are based
on recent work on learning holomorphic, Banach-valued functions [2, 5, 6]. See also Remark C.1
below.

The proof of Theorem 3.1 involves three mains steps.

(a) Formulation (§D.1) and analysis (§D.2-D.4) of a suitable polynomial learning procedure.
(b) Construction of a family of DNNs that approximately emulates the Legendre polynomials (§D.5).

(c) Analysis of the corresponding training problem (2.5) (§D.6–D.8).

Since our goal in this work is ‘agnostic’ DNN architectures (i.e., independent of the smoothness
of the underlying operator), in step (a) we first define a nonlinear set (D.2) spanned by Legendre
polynomials with nonzero indices in certain sets of bounded weighted cardinality. This is effectively
a form of sparse polynomial approximation, and the analysis of the resulting learning procedure (D.3)
relies heavily on techniques from compressed sensing. In order to bound the encoding-decoding
error, we also require several results on Lipschitz continuity (Lemma D.2) and norm equivalences
(Lemma D.4) for multivariate polynomials.

Step (b) relies on what have now become fairly standard results in DNN approximation theory:
namely, the approximate emulation of orthogonal polynomials via DNNs of given width and depth.
We present such a result in Lemma D.9, then use this to define the DNN family N in (D.19).

We then analyze the DNN training problem (2.5) in step (c). Using the emulation result of step (b), we
first show that any approximate minimizer b
N of (2.5) yields a polynomial that is also an approximate
minimizer of the polynomial training problem (D.3) (Lemma D.12). We may then apply the results
shown in Step (a) to prove a generalization bound (Theorem D.13). Up to this point, we have not
used the holomorphy assumption. We now use this assumption to bound the various best polynomial
approximation errors that arise in the previously-derived generalization bound (§D.7). Finally, in
§D.8 we put all these estimates together to complete the proof.

Remark C.1 Theorem 3.1 is a generalization and improvement of [5, Thms. 4.1 & 4.2], which deals
with the case of learning Banach-valued functions rather than operators. Specifically, the setting of [5]
can be considered a special case of this paper where X = ℓ∞(N) and the encoding error EX,q = 0.
Moreover, Theorem 3.1 improves the main results of [5] in three key ways. First, the the DNN
architectures bounds are much narrower: width(N) ≲(m/L)1+δ versus width(N) ≲m3+log2(m)
in the latter. Second, Theorem 3.1 considers standard training, i.e., ℓ2-loss minimization, whereas [5,
Thms. 4.1 & 4.2] requires regularization. Third, for Banach spaces, the error decay rate with respect

29


---Page Break---
to m is roughly doubled: Eapp,2 = O(m1−1/p) in Theorem 3.1 versus O(m1/2(1−1/p)) in [5, Thm.
4.1]. Finally, Theorem 3.1 also provides bounds in the L∞
µ -norm, whereas the results in [5] only
consider the L2
µ-norm.

C.2.2
Theorem 3.2

The proof of Theorem 3.2 relies of three key steps.

(a) Using a minimizer of the polynomial training problem (D.3) to construct a family of minimizers
of the DNN training problem (2.5).
(b) Analysis of the corresponding minimizers using the previously-derived bound for polynomial
minimizers.
(c) Using the Lipschitz continuity of DNNs to show stability of the DNN minimizer and a permutation
argument to show the existence of many parameters that lead to equally ‘good’ minimizers.

Given any minimizer of the polynomial training problem (D.3), it is straightforward to define a DNN
with the desired generalization bound. Unfortunately, this will generally not be a minimizer of (2.5).
To achieve the aims of step (a) we proceed as follows. First, we note that a DNN will be a minimizer
if the corresponding approximation satisfies bF(Xi) = eYi, where eYi are the closest points to the Yi
from eY = DY(RdY). To achieve this, we take the existing DNN then add on a suitable number of
additional terms corresponding to the first r > m order-one Legendre polynomials (§E.1). We show
that by doing this, we can construct a DNN for which bF(Xi) = eYi (Lemma E.1).

In Step (b), we first bound this DNN minimizer in terms of the polynomial minimizer plus the
contributions of these additional terms (Lemma E.2). The latter involves the minimal singular value
of a certain m × r matrix B, which is the matrix of the linear system that enforces the condition
bF(Xi) = eYi. We bound this minimal singular value in §E.3.

We then use this to complete the proof of part (A) of Theorem 3.2 in §E.4. In this section, we also
complete step (c) to establish parts (B) and (C).

Remark C.2 Like Theorem 3.1, Theorem 3.2 also relies on ideas from [5]. However, [5] does not
address fully-connected DNN architectures. To address this challenge, the proof of Theorem 3.2
involves the technical construction described above.

C.2.3
Theorem 4.1

Theorems 4.1 is based on [7, Thm. 4.4]. The basic idea is to consider a family of affine, holomorphic
operators (F.4). This allows us to lower bound the quantity θm(b) by the so-called Gelfand width
(F.1) of a certain weighted unit ball in a finite-dimensional space. Bounds for such Gelfand widths
are known, and this allows us to derive the corresponding result. The main difference between this
and [7, Thm. 4.4] is the setup leading to the construction in (F.4).

C.2.4
Theorem 4.2

Theorems 4.2 employs similar ideas, but in a more technical manner. We consider a family of linear,
holomorphic operators (G.2), which involves a sum over r groups of m + 1 coefficients. We restrict
to coefficients that lie in the null space of the corresponding sampling operator. Then, through a
series of inequalities, we can lower bound ˜θm(b) by a sum over r terms, each involving the ℓ1-norms
of certain vectors in the null space of the matrix of the corresponding sampling operator (G.3). This
matrix is a subgaussian random matrix. We now use a technical estimate from [75] for vectors in the
null space of subgaussian random matrices, which shows that they cannot be ‘spiky’. Applying this
and a series of further inequalities yields the result.

D
Proof of Theorem 3.1

D.1
Formulation of an approximate polynomial training problem

Let Λ ⊂F with supp(ν) ⊆{1, . . . , dX }, ∀ν ∈Λ, and write N = |Λ|. Let k > 0 and define the set
S = SΛ,k = {S ⊆Λ : |S|v ≤k}.
(D.1)

30


---Page Break---
Both Λ and k will be chosen later in the proof. For any Banach space (Z, ∥· ∥Z) define the space

PS;Z =

(X

ν∈S
cνΨν : cν ∈Z, S ∈S

)

.
(D.2)

Then, given the training data (2.4), consider the problem

min
p∈PS; e
Y

1
m

m
X

i=1
∥Yi −p ◦EX (Xi)∥2
Y,
(D.3)

where eY = DY(RdY). Here and throughout the proofs, we slightly abuse notation: the polynomial
p : RN →R, whereas EX has codomain RdX . However, by construction, p is independent of all
but the first dX variables. Hence we may consider p as a function RdX →R. This aside, if ˆp is an
approximate minimizer of (D.3), then we define the approximation to F as

F ≈bF = ˆp ◦EX .
(D.4)

Notice that p ◦EX = DY ◦P ◦EX , where P : RdX →RdY is a vector-valued polynomial, since, by
(A.IV), the map DY is linear. The idea exploited in this proof is to construct a class of DNNs N such
that (i) all such polynomials P are approximated by members of N and (ii) the polynomial training
problem (D.3) is approximated by the DNN training problem (2.5). The first step in this analysis is
therefore to analyze the polynomial training problem (D.3).

D.2
Supporting lemmas

We require several lemmas. The first relates the L∞
ϱ -norm of a polynomial to its Pettis L2
ϱ-norm.

Lemma D.1 (Nikolskii inequality for polynomials). Let (Z, ∥· ∥Z) be any Banach space and
p = P

ν∈S cνΨν for some finite set S ⊂F, where cν ∈Z. Then

∥p∥L∞
ϱ (D;Z) ≤
p

|S|u|||p|||L2ϱ(D;Z).

Proof. By definition

∥p∥L∞
ϱ (D;Z) = |||p|||L∞
ϱ (D;Z) =
sup
z∗∈B(Z∗)
∥z∗(p)∥L∞
ϱ (D).

Fix z∗∈B(Z∗) and write z∗(p) = P

ν∈S z∗(cν)Ψν. By the triangle inequality and the definition
of the weights (C.3), we have

∥z∗(p)∥L∞
ϱ (D) ≤
X

ν∈S
|z∗(cν)|∥Ψν∥L∞
ϱ (D) =
X

ν∈S
uν|z∗(cν)|.

We now apply the Cauchy–Schwarz inequality, (C.1) and Parseval’s identity to get

∥z∗(p)∥L∞
ϱ (D) ≤
p

|S|u

sX

ν∈S
|z∗(cν)|2 =
p

|S|u∥z∗(p)∥L2
ϱ(D).

Since z∗was arbitrary, we deduce that

∥p∥L∞
ϱ (D;Z) ≤
p

|S|u
sup
z∗∈B(Z∗)
∥z∗(p)∥L2ϱ(D) =
p

|S|u|||p|||L2ϱ(D;Z),

as required.

Next, we require the following bound on the Lipschitz constant of a multivariate polynomial.
Lemma D.2 (Lipschitz continuity for polynomials). Let (Z, ∥· ∥Z) be any Banach space and
suppose that p = P

ν∈S cνΨν for some finite S ⊂F, where cν ∈Z. Then satisfies

Lip(p; B∞(N), Z) ≤1

2

p

|S|v · |||p|||L2ϱ(D;Z),

where B∞(N) = {x ∈RN : ∥x∥∞≤1} is the unit ball of ℓ∞(N).

31


---Page Break---
Proof. Fix z∗∈B(Z∗) and let ˜p = z∗(p) = P
ν∈S z∗(cν)Ψν =: P
ν∈S ˜cνΨν be the correspond-
ing scalar-valued polynomial. Let x, x′ ∈D. Then the mean value theorem gives that

˜p(x′) −˜p(x) =

∞
X

i=1
(x′
i −xi) ∂˜p

∂xi
(tx + (1 −t)x′)

for some 0 ≤t ≤1. Since B∞(N) ≡D, this and the fact that D is convex give that

|˜p(x′) −˜p(x)| ≤∥x′ −x∥∞

∞
X

i=1


∂˜p
∂xi


L∞
ϱ (D)
.
(D.5)

We now consider the terms in the sum separately. Using the definition of the Legendre polynomials
(see §C.1.4), we have
∂˜p
∂xi
=
X

ν∈S
˜cνuνP ′
νi(xi)
Y

j̸=i
Pνj(xj).

The unnormalized Legendre polynomials satisy 1 = Pn(1) = ∥Pn∥L∞([−1,1]) and n(n + 1)/2 =
P ′
n(1) = ∥P ′
n∥L∞([−1,1]). Hence

∂˜p
∂xi


L∞
ϱ (D)
≤
X

ν∈S
|˜cν|uννi(νi + 1)/2.

We deduce that

∞
X

i=1


∂˜p
∂xi


L∞
ϱ (D)
≤
X

ν∈S
|˜cν|uν

∞
X

i=1
νi(νi + 1)/2.

Now, by definition of the weights uν and vν (see (C.3) and (C.4)), we have

(νi + 1) ≤
Y

j∈N
(2νj + 1) = u2
ν

and
∞
X

i=1
νi ≤
Y

j∈N
(2νj + 1) = u2
ν.

Hence
∞
X

i=1
uννi(νi + 1) ≤u5
ν ≤vν.

Here we also used the fact that u ≥1. We now apply this, the Cauchy-Schwarz inequality and
Parseval’s identity to get
∞
X

i=1


∂˜p
∂xi


L∞
ϱ (D)
≤1

2∥˜p∥L2ϱ(D)
p

|S|v.

Substituting this into (D.5) now gives

|˜p(x′) −˜p(x)| ≤1

2∥˜p∥L2ϱ(D)
p

|S|v∥x′ −x∥∞.

We now recall that ˜p = z∗(p) and z∗∈B(Z∗) was arbitrary to get

∥p(x′) −p(x)∥Z =
sup
z∗∈B(Z∗)
|z∗(p(x′) −p(x))|

≤1

2
sup
z∗∈B(Z∗)
∥z∗(p)∥L2ϱ(D)
p

|S|v∥x′ −x∥∞

= 1

2|||p|||L2ϱ(D;Z)
p

|S|v∥x′ −x∥∞,

as required.

32


---Page Break---
We now show that this result can be used to imply a norm equivalence for polynomials. For this we
first require the following lemma. Note that in this and subsequent results, we abuse notation and
write ˜ς for both the measure on [−1, 1]dX defined in (A.III) and the measure on D = [−1, 1]N defined
by tensoring this measure (corresponding to the first dX variables x1, . . . , xdX ) with the uniform
measure on D\[−1, 1]dX (corresponding to the remaining variables xdX +1, xdX +2, . . .).

Lemma D.3 (Closeness of L2 norms). Let (Z, ∥· ∥Z) be any Banach space, ς be the measure defined
in (A.I) and ˜ς be the measure defined in (A.III). Suppose that f ∈L2
ς(D; Z) is Lipschitz continuous
with constant L = Lip(f; B∞(N), Z) < ∞and that f depends only on its first dX variables. Then
f ∈L2
˜ς(D; Z) and

|||f|||L2ς(D;Z) −δ ≤|||f|||L2
˜ς(D;Z) ≤|||f|||L2ς(D;Z) + δ,

where
δ = L · ∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(N)).

Proof. Fix z∗∈B(Z∗), let g = z∗(f) ∈L2
ς(D) and set

a = ∥g∥L2
˜ς(D),
b = ∥g∥L2ς(D).

Notice that

|g(x) −g(x′)| ≤∥z∗∥Z∗∥f(x) −f(x′)∥Z ≤L∥x −x′∥∞,
∀x, x′ ∈B∞(N).

Now, with slight abuse of notation, g(ι(X)) = g(ιdX (X)) due to the assumption on f. Therefore,
using this and the Cauchy–Schwarz inequality,

a2 −b2 =
Z

D
(g(ιdX ◦eDX ◦eEX (X)))2 −(g(ιdX (X)))2 dµ(X)

≤L
Z

D
∥ιdX (X) −ιdX ◦eDX ◦eEX (X)∥∞

|g(ιdX (X))| + |g(ιdX ◦eDX ◦eEX )|

dµ(X)

≤L∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(N)) (a + b) .

We deduce that a −b ≤δ. Since z∗was arbitrary, we get

|||f|||L2
˜ς(D;Z) =
sup
z∗∈B(Z∗)
∥z∗(f)∥L2
˜ς(D) ≤
sup
z∗∈B(Z∗)
∥z∗(f)∥L2ς(D) + δ = |||f|||L2ς(D;Z) + δ,

which gives the upper bound. The same argument applied to b2 −a2 also gives the lower bound.

Lemma D.4 (Norm equivalences for polynomials). Let (Z, ∥· ∥Z) be any Banach space, ς be the
measure defined in (A.I), ˜ς be the measure defined in (A.III) and S ⊂F with supp(ν) ∈{1, . . . , dX },
∀ν ∈S. Suppose that
p

|S|v · ∥ιdX −ιdX ◦eDX ◦eEX ∥L2
µ(X;ℓ∞(N)) ≤c,
(D.6)

for some sufficiently small universal constant c > 0. Then the norm equivalence

|||p|||L2ϱ(D;Z) ≲|||p|||L2
˜ς(D;Z) ≲|||p|||L2ϱ(D;Z)

holds for all p = P

ν∈S cνΨν, where cν ∈Z.

Proof. Combining Lemmas D.2 and D.3, we see that

|||p|||L2ς(D;Z) −δ ≤|||p|||L2
˜ς(D;Z) ≤|||p|||L2ς(D;Z) + δ,

where δ = 1

2|||p|||L2ϱ(D;Z)
p

|S|v∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(N)) ≤c|||p|||L2ϱ(D;Z)/2. By (A.I),
there are constants c1 ≥c2 > 0 such that

c1|||p|||L2ϱ(D;Z) ≤|||p|||L2ς(D;Z) ≤c2|||p|||L2ϱ(D;Z).

We now take c = c1/2.

33


---Page Break---
D.3
Analysis of (D.3)

We now analyze (D.3). Our analysis relies on the following result, which shows an error bound
subject to a certain discrete metric inequality (D.7).

Lemma D.5 (Discrete metric inequality implies error bounds). Let Q : Y →eY = DY(RdY) be a
bounded linear operator and πQ = ∥Q∥Y→Y. Suppose that

∥p −q∥disc,˜ς ≥α max{|||p −q|||L2ϱ(D;Y), |||p −q|||L2
˜ς(D;Y)},
∀p, q ∈PS; e
Y
(D.7)

for some α > 0. Then, for any F ∈L∞
µ (X; Y), p ∈PS; e
Y and q ∈PS;Y, we have

|||F −p ◦EX |||L2µ(X;Y) ≤|||F −Q ◦F|||L2µ(X;Y) + α−1∥p −Q ◦q∥disc,˜ς
+ πQ|||F −q ◦EX |||L2µ(X;Y)

and

|||F −p ◦EX |||L∞
µ (X;Y) ≤|||F −Q ◦F|||L∞
µ (X;Y) +
√

2k/α∥p −Q ◦q∥disc,˜ς
+ πQ∥F −q ◦EX ∥L∞
µ (X;Y).

Proof. By the triangle inequality and properties of Q, we have

|||F −p ◦EX |||L2µ(X;Y)
≤|||F −Q ◦F|||L2µ(X;Y) + |||p ◦EX −Q ◦q ◦EX |||L2µ(X;Y) + |||Q ◦F −Q ◦q ◦EX |||L2µ(X;Y)
≤|||F −Q ◦F|||L2µ(X;Y) + |||p ◦EX −Q ◦q ◦EX |||L2µ(X;Y) + πQ|||F −q ◦EX |||L2µ(X;Y).

Now, since p, Q ◦q ∈PS; e
Y, the second term can be bounded by

|||p ◦EX −Q ◦q ◦EX |||L2µ(X;Y) = |||p −Q ◦q|||L2
˜ς(D;Y) ≤α−1∥p −Q ◦q∥disc,˜ς.
(D.8)

This yields the first result.

For the second result, we once more write

|||F −p ◦EX |||L∞
µ (X;Y) ≤|||F −Q ◦F|||L∞
µ (X;Y) + |||p ◦EX −Q ◦q ◦EX |||L∞
µ (X;Y)
+ πQ|||F −q ◦EX |||L∞
µ (X;Y).

For the second term, we use (A.III) to write

|||p ◦EX −Q ◦q ◦EX |||L∞
µ (X;Y) = |||p −Q ◦q|||L∞
˜ς (D;Y) ≤∥p −Q ◦q∥L∞
ϱ (D;Y).

Notice that p −Q ◦q is a polynomial supported in a set S with |S|u ≤2k. We now apply Lemma
D.1 and (D.7) to obtain

|||p ◦EX −Q ◦q ◦EX |||L∞
µ (X;Y) ≤
√

2k|||p −Q ◦q|||L2
ϱ(D;Y) ≤
√

2k/α|||p −Q ◦q|||disc,˜ς. (D.9)

This gives the result.

Theorem D.6 (Error bound for polynomial minimizers). Let Q : Y →eY = DY(RdY) be a bounded
linear operator, πQ = ∥Q∥Y→Y and suppose that (D.7) holds. Let bF be as in (D.4) for some
(σ, τ)-approximate minimizer ˆp of (D.3). Then

|||F −bF|||L2µ(X;Y) ≤|||F −Q ◦F|||L2µ(X;Y) + σ + 1

α
|||F −Q ◦F|||disc,µ

+ πQ


|||F −q ◦EX |||L2
µ(X;Y) + σ + 1

α
|||F −q ◦EX |||disc,µ



+ τ

α + σ + 1

α√m∥E∥2;Y

34


---Page Break---
and

∥F −bF∥L∞
µ (X;Y) ≤∥F −Q ◦F∥L∞
µ (X;Y) +

√

2k(σ + 1)

α
|||F −Q ◦F|||disc,µ

+ πQ

 

∥F −q ◦EX ∥L∞
µ (X;Y) +

√

2k(σ + 1)

α
|||F −q ◦EX |||disc,µ

!

+

√

2kτ

α
+

√

2k(σ + 1)

α√m
∥E∥2;Y

for all q ∈PS;Y, where E = (Ei)m
i=1 ∈Ym is the (Banach-valued) vector of noise terms.

Proof. We apply the previous lemma with p = ˆp. This gives

|||F −bF|||L2µ(X;Y) ≤|||F −Q ◦F|||L2µ(X;Y) + α−1∥ˆp −Q ◦q∥disc,˜ς
+ πQ|||F −q ◦EX |||L2µ(X;Y)

∥F −bF∥L∞
µ (X;Y) ≤∥F −Q ◦F∥L∞
µ (X;Y) +
√

2k/α∥ˆp −Q ◦q∥disc,˜ς
+ πQ∥F −q ◦EX ∥L∞
µ (X;Y).

(D.10)

Consider the second term. We have

∥ˆp −Q ◦q∥disc,˜ς = |||ˆp ◦EX −Q ◦q ◦EX |||disc,µ ≤|||F −Q ◦q ◦EX |||disc,µ + |||F −ˆp ◦EX |||disc,µ.

Consider the second term of this expression. By the triangle inequality and the facts that ˆp is a
(σ, τ)-minimizer and Q ◦q is feasible, we obtain

|||F −ˆp ◦EX |||disc,µ ≤

v
u
u
t 1

m

m
X

i=1
∥Yi −ˆp ◦EX ∥2
Y +
1
√m∥E∥2;Y

≤σ

v
u
u
t 1

m

m
X

i=1
∥Yi −Q ◦q ◦EX ∥2
Y + τ +
1
√m∥E∥2;Y

≤σ|||F −Q ◦q ◦EX |||disc,µ + τ + σ + 1
√m ∥E∥2;Y

Therefore, we get

∥ˆp −Q ◦q∥disc,˜ς ≤(σ + 1)|||F −Q ◦q ◦EX |||disc,µ + τ + σ + 1
√m ∥E∥2;Y.

We now estimate the first term in this expression as follows:

|||F −Q ◦q ◦EX |||disc,µ ≤|||F −Q ◦F|||disc,µ + |||Q ◦F −Q ◦q ◦EX |||disc,µ
≤|||F −Q ◦F|||disc,µ + πQ|||F −q ◦EX |||disc,µ.

Therefore, we conclude that

∥ˆp −Q ◦q∥disc,˜ς ≤(σ+1)|||F −Q ◦F|||disc,µ+(σ+1)πQ|||F −q ◦EX |||disc,µ+τ + σ + 1
√m ∥E∥2;Y.

Combining this with (D.10) now gives the result.

D.4
Ensuring (D.7) holds with high probability

For the proof of the next lemma and subsequent steps of the proof, we let Λ = {ν1, . . . , νN} and
define the matrix

A =
Ψνj(EX (Xi))
√m

m,N

i,j=1
∈Rm×N.

35


---Page Break---
Lemma D.7. Let 0 < ϵ < 1, 0 < δ < 1, k > 0 and suppose that

m ≥c0 · δ−2 · k · (log(eN) · log2(k/δ) + log(2/ϵ))
(D.11)

for some universal constant c0 > 0. Let T = {c ∈RN : ∥c∥2 = 1, ∥c∥0,v ≤k} and

θ+ = sup
n
E∥Ac∥2
2 : c ∈T
o
,
θ−= inf
n
E∥Ac∥2
2 : c ∈T
o
.

Then
∥Ac∥2
2 ≥(θ−−(1 + θ+)c1δ) ∥c∥2
2,
∀c ∈RN, ∥c∥0,v ≤k.

with probability at least 1 −ϵ, where c1 > 0 is a universal constant.

Proof. The proof is based on [12, Thm. 2.13]. Since ∥c∥1,v ≤
q

∥c∥0,v∥c∥2 ≤
√

k, we see that

T ⊆
n
c ∈RN : ∥c∥1,v ≤
√

k
o
.

Define the random vector X ∈RN by X = (Ψνi(EX (X)))N
i=1 for X ∼µ. Observe that

|⟨X, ej⟩| = |Ψνi(EX (X))| ≤uνi ≤vνi

almost surely, by (A.III) and the definition of the weights. Therefore, by [12, Thm. 2.13], if

m ≥c0 · δ−2 · k · log(eN) · log2(k/(c1δ)),
(D.12)

it holds that

sup
c∈T


1
m

m
X

i=1
|⟨c, Xi⟩|2 −E|⟨c, X⟩|2
 ≤c1δ

1 + sup
c∈T
E|⟨c, X⟩|2


with probability at least 1 −2 exp(−c2δ2m/k). Here c0, c1, c2 > 0 are universal constants. Now
observe that
1
m

m
X

i=1
|⟨c, Xi⟩|2 = ∥Ac∥2
2,
E|⟨c, X⟩|2 = E(∥Ac∥2
2).

Therefore, we have shown that

∥Ac∥2
2 ≥θ−−(1 + θ+)c1δ,
∀c ∈T,

with probability at least 1 −2 exp(−c2δ2m/k), provided m satisfies (D.12). To conclude the result,
we observe that (D.11) implies (D.12), up to a possible change in the universal constant c0. Moreover,
it also implies that
2 exp(−c2δ2m/k) ≤ϵ.
Hence we obtain the result.

Lemma D.8. There exist universal constants c0, c1, c2 > 0 such that the following holds. Suppose
that
m ≥c0 · k · (log(eN) · log2(k) + log(2/ϵ))
(D.13)
and
√

k · ∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(N)) ≤c1.
(D.14)

Then (D.7) holds with probability at least 1 −ϵ and constant α ≥c2.

Proof. We shall apply the previous lemma with k replaced by 2k. First, we estimate θ+ and θ−. Let
c ∈T, with T as defined therein with 2k in place of k. Write p = P

ν∈Λ cνΨν for the corresponding
(scalar-valued) polynomial. Then

∥Ac∥2
2 = 1

m

m
X

i=1
|p ◦EX (Xi)|2,

and therefore
E∥Ac∥2
2 = ∥p∥2
L2
˜ς(D).

36


---Page Break---
Since ∥c∥0,v ≤2k, (D.14) implies (D.6). We now apply Lemma D.4 and the fact that ∥p∥L2ϱ(D) =
∥c∥2 = 1 to get
c3 ≤E∥Ac∥2
2 ≤c4
for universal constants c4 ≥c3 > 0. Since c ∈T was arbitrary to deduce that

c3 ≤θ−≤θ+ ≤c4.
(D.15)

We now show that (D.7) holds with the desired probability. Let p, q ∈PS; e
Y be arbitrary. Then their
difference h = p −q can be expressed as

h =
X

ν∈Λ
dνΨν
(D.16)

where d = (dν)ν∈Λ ∈eYN satisfies

∥d∥0,u ≤∥d∥0,v ≤2k.

Now, this, (D.14) and Lemma D.4 imply that

|||h|||L2
˜ς(D;Y) ≤c4|||h|||L2ϱ(D;Y).

Therefore, in order to prove (D.7), it suffices to show that |||h|||L2ϱ(D;Y) ≲∥h∥disc,˜ς.

Observe that
∥h∥disc,˜ς = ∥Ad∥2;Y ≥|||Ad|||2;Y =
sup
y∗∈B(Y∗)
∥y∗(Ad)∥2,

where we recall that for a vector z = (zi)N
i=1 ∈YN, we write y∗(z) = (y∗(zi))N
i=1 ∈RN. By
linearity, we have y∗(Ad) = Ay∗(d). Therefore, by Lemma D.7,

∥h∥2
disc,˜ς = ∥Ad∥2
2;Y
≥
sup
y∗∈B(Y∗)
∥Ay∗(d)∥2
2

≥
sup
y∗∈B(Y∗)
(θ−−(1 + θ+)c5δ) ∥y∗(d)∥2
2

≥(c3 −(1 + c4)c5δ)
sup
y∗∈B(Y∗)
∥y∗(h)∥2
L2ϱ(D)

= (c3 −(1 + c4)c5δ) |||h|||2
L2ϱ(D;Y).

for some universal constant c5 > 0, provided m satisfies (D.11). We now set δ = c3/(2(1 + c4)c5)
to get
∥h∥2
disc,˜ς ≥c3/2|||h|||2
L2ϱ(D;Y),

provided m satisfies (D.11) with this value of δ. However, this is implied by the condition (D.13).
We deduce the result.

D.5
Construction of the DNN family N

Recall that Λ ⊂F, |Λ| = N is an arbitrary set and S is defined by (D.1). In this and what follows,
we slightly abuse notation and consider a DNN N : Rd →R as a function RN →R which depends
on only the first d variables.
Lemma D.9 (Approximating Legendre polynomials with tanh DNNs). Let Γ ⊂Λ with supp(ν) ⊆
{1, . . . , d}, ∀ν ∈Γ, and m(Γ) = maxν∈Γ ∥ν∥1 < ∞. There exists a fully-connected family No of
tanh DNNs N : Rd →R with

width(No) ≲m(Γ),
depth(No) ≲log(m(Γ)),

such that, for any 0 < δ < 1 and ν ∈Γ, there is a DNN Nν ∈No with

∥Nν −Ψν∥L∞
ϱ (D) ≤δ.
(D.17)

Moreover, the zero network 0 : x 7→0 also belongs to No (trivially, since tanh(0) = 0).

37


---Page Break---
Proof. We follow the argument of [5, Thm. 7.4], which is based on [79, Prop. 2.6]. Let ν ∈Γ. Then
via the fundamental theorem of algebra we can write Ψν(x) as a product of ∥ν∥1 numbers as

Ψν(x) =
Y

i∈supp(ν)

νi
Y

j=1
di(xi −rij).

Here {rij}νi
j=1 are the roots of the univariate Legendre polynomial Pνi. We append m(Γ) −∥ν∥1
ones and write Ψν(x) as a product of exactly m(Γ) numbers. Now define the affine map

Aν : Rd →Rm(Γ),
(D.18)

so that Aν(x) is the vector consisting of the values di(xi −rij) for j = 1, . . . , νi and i ∈supp(ν)
and 1 otherwise. To complete the proof, we need to construct a tanh DNN mapping Rm(Γ) →R that
approximately multiplies these numbers. To do this, we argue as in the proof of [5, Thm. 7.4] to see
that there is a tanh DNN Nν with

width(Nν) ≲m(Γ),
depth(Nν) ≲log(m(Γ))

that satisfies the desired bound (D.17). Since these width and depth bounds are independent of ν, we
deduce the result.

Fix δ > 0, let Γ = ∪S∈SS, d = dX and consider the corresponding family No and DNNs Nν,
ν ∈Γ, asserted by this lemma. For any ν ∈Γ, we have ν ∈S for some S ∈S, and any such S
satisfies |S|v ≤k. Therefore u2(5+ξ)
ν
= v2
ν ≤k for any ν ∈S and we deduce that

∥ν∥1 ≤

d
Y

j=1
(2νj + 1) = u2
ν ≤k1/(5+ξ).

This implies that m(Γ) ≤k1/(5+ξ) and therefore

width(No) ≲k1/(5+ξ),
depth(No) ≲log(k).

With this in hand, we now define the family N of DNNs N : RdX →RdY by

N =




















N = C





Nν1
...
Nν|S|
0
...
0





: S = {ν1, . . . , ν|S|} ∈S, C ∈RdY×⌊k⌋




















.
(D.19)

Here we also use the fact that |S| ≤|S|v ≤k. Notice that this family satisfies

width(N) ≤k1+1/(5+ξ),
depth(N) ≲log(k),
(D.20)

due to the bounds for No.

Now let N ∈N and write C = [c1| · · · |c⌊k⌋], where ci ∈RdY. Then

DY ◦N =

|S|
X

i=1
DY(ci)Nνi =

|S|
X

i=1
ciNνi,
where ci ∈eY.

Therefore, we can associate N with the space of functions ePS; e
Y, where, for any arbitrary Banach
space (Z, ∥· ∥Z),

ePS;Z =

(X

ν∈S
cνNν : cν ∈Z, S ∈S

)

.

We now require the following lemma, which relates the distance between a function in ePS;Z and the
corresponding polynomial in PS;Z.

38


---Page Break---
Lemma D.10 (Discrete norms of polynomials and their approximating DNNs). Let S ⊂F,
(Z, ∥· ∥Z) be any Banach space and suppose that p = P

ν∈S cνΨν, where cν ∈Z. Define

˜p =
X

ν∈S
cνNν.

Then
∥p −˜p∥disc,˜ς ≤∥p −˜p∥L∞
ϱ (D;Z) ≤δ
p

|S||||p|||L2ϱ(D;Z).
(D.21)

Moreover, if Z = eY, S ∈S and (D.7) holds with α satisfying δ
p

|S|u/α < 1 then

|||p|||L2ϱ(D;Y) ≤
1

α −δ
p

|S|
∥˜p∥disc,˜ς.

Proof. By (A.III) and the definition of the Nν we have
∥p −˜p∥disc,˜ς ≤∥p −˜p∥L∞
ϱ (D;Z)
=
sup
z∗∈B(Z∗)
∥z∗(p −˜p)∥L∞
ϱ (D)

≤
sup
z∗∈B(Z∗)

X

ν∈S
|z∗(cν)|∥Ψν −Nν∥L∞
ϱ (D)

≤δ|||c|||1;Z.

We now apply the Cauchy–Schwarz inequality and Parseval’s identity to obtain

∥p −˜p∥disc,˜ς ≤δ
p

|S||||c|||2;Z = δ
p

|S|u|||p|||L2ϱ(D;Z),

which gives the first result.

For the second result, we apply (D.7) with q = 0 to get

|||p|||L2ϱ(D;Y) ≤α−1∥p∥disc,˜ς ≤α−1 
∥p −˜p∥disc,˜ς + ∥˜p∥disc,˜ς

.

Using (D.21) and the fact that δ
p

|S|/α < 1 now gives the result.

Remark D.11 (Other activation functions) As seen in this section, a key step in our proofs is
emulating the polynomials via DNNs of quantifiable width and depth. There is an extensive literature
on this topic. See, e.g., [5, 21, 23, 25, 38, 57, 63, 67, 72, 78, 79, 85, 86, 92, 97] and references
therein. The proof of Lemma D.9 reduces this to the task of emulating the multiplication operation
(x1, . . . , xd) ∈Rd 7→x1 · · · xd ∈R via a DNN. As shown in the proof of [5, Thm. 7.4] (which is
based on [79, Prop. 2.6]), this can in turn be achieved using a binary tree of ⌈log2(d)⌉DNNs that
approximately compute the multiplication of two numbers (x, y) ∈R2 7→xy ∈R. Further, this task
can be achieved via the identity xy = ((x + y)2 −(x −y)2)/4 by using a DNN that approximately
computes the squaring function x ∈R 7→x2 ∈R. To summarize, provide a DNN of quantifiable
width and depth can approximately compute the squaring function, it can also approximately emulate
the multivariate Legendre polynomials.

In view of this, we can adapt our main theorems to various other activation functions without change.
This includes Rectified Polynomial Units (RePUs), where the emulation is, in fact, exact (see, e.g.,
[57, Lem. 2.1]). It also includes the Exponential Linear Unit (ELU) used in our numerical experiments
and many others. See, for instance, Proposition 4.7 of [38] and the ensuing discussion. Rectified
Linear Units (ReLUs) are slightly different, as in this case the depth of the DNN that performs the
approximation multiplication depends on the desired accuracy (see, e.g., [79, Prop. 2.6]). One could
modify Theorem 3.1 to consider ReLU DNNs, with the result being a worse depth bound than that
presented for tanh DNNs.

D.6
Analysis of (2.5)

Lemma D.12 (Approximate minimizers of (2.5) yield approximate minimizers of (D.3)). Suppose
that (D.7) holds, let N be the family of DNNs defined in §D.5 and b
N be any (σ, τ)-approximate
minimizer of (2.5). Let
DY ◦b
N =
X

ν∈S
ˆcνNν ∈ePS; e
Y,

39


---Page Break---
where S ∈S and ˆcν ∈eY, and define

ˆp =
X

ν∈S
ˆcνΨν.

Then ˆp is a (σ′, τ ′)-approximate minimizer of (D.3), where

σ′ ≤σ(1 + δ
√

k/α)

τ ′ ≤τ + σδ
√

k/α

∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y


+ δ
√

k|||ˆp|||L2ϱ(D;Y)
.
(D.22)

Proof. Let p = P

ν∈S cνΨν ∈PS; e
Y be arbitrary and N ∈N be the corresponding DNN so that

DY ◦N = P
ν∈S cνNν. Then by the triangle inequality and the fact that b
N is an approximate
minimizer, we have
v
u
u
t 1

m

m
X

i=1
∥Yi −ˆp ◦EX (Xi)∥2
Y

≤

v
u
u
t 1

m

m
X

i=1
∥Yi −DY ◦b
N ◦EX (Xi)∥
2
Y + ∥ˆp −DY ◦b
N∥disc,˜ς

≤σ

v
u
u
t 1

m

m
X

i=1
∥Yi −DY ◦N ◦EX (Xi)∥2
Y + τ + ∥ˆp −DY ◦b
N∥disc,˜ς

≤σ

v
u
u
t 1

m

m
X

i=1
∥Yi −p ◦EX (Xi)∥2
Y + τ + σ∥p −DY ◦N∥disc,˜ς + ∥ˆp −DY ◦b
N∥disc,˜ς.

We now apply Lemma D.10 to the last two terms, noting that |S| ≤|S|v ≤k since S ∈S, to obtain
v
u
u
t 1

m

m
X

i=1
∥Yi −ˆp ◦EX (Xi)∥2
Y ≤σ

v
u
u
t 1

m

m
X

i=1
∥Yi −p ◦EX (Xi)∥2
Y

+ τ + σδ
√

k|||p|||L2ϱ(D;Y) + δ
√

k|||ˆp|||L2ϱ(D;Y).

Consider the third term. We first apply (D.7) with q = 0 to get

|||p|||L2ϱ(D;Y) ≤α−1∥p∥disc,˜ς = α−1|||p ◦EX |||disc,µ.

We then use the triangle inequality to get

|||p ◦EX |||disc,µ ≤|||F −p ◦EX |||disc,µ + |||F|||disc,µ

≤

v
u
u
t 1

m

m
X

i=1
∥Yi −p ◦EX (Xi)∥2
Y + ∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y.

Therefore, we obtain
v
u
u
t 1

m

m
X

i=1
∥Yi −ˆp ◦EX (Xi)∥2
Y ≤σ(1 + δ
√

k/α)

v
u
u
t 1

m

m
X

i=1
∥Yi −p ◦EX (Xi)∥2
Y + τ

+ σδ
√

k/α

∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y


+ δ
√

k|||ˆp|||L2ϱ(D;Y).

Since p ∈PS; e
Y was arbitrary we get the result.

40


---Page Break---
Theorem D.13 (Error bound for DNN minimizers). Let Q : Y →eY = DY(RdY) be a bounded
linear operator, πQ = ∥Q∥Y→Y and suppose that (D.7) holds with α ≥c0 and α −δ
√

k ≥c1 for
suitable universal constants c0, c1 > 0. Let N be the family of DNNs defined in §D.5 and b
N be any
(σ, τ)-approximate minimizer of (2.5). Then the approximation bF = DY ◦b
N ◦EX satisfies

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + σ|||F −Q ◦F|||disc,µ

+ πQ

|||F −q ◦EX |||L2µ(X;Y) + σ|||F −q ◦EX |||disc,µ


+ τ + σδ
√

k∥F∥L∞
µ (X;Y) +
σ
√m∥E∥2;Y

and
∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

kσ|||F −Q ◦F|||disc,µ

+ πQ

∥F −q ◦EX ∥L∞
µ (X;Y) +
√

kσ|||F −q ◦EX |||disc,µ


+
√

kτ + σδk∥F∥L∞
µ (X;Y) +

√

kσ
√m ∥E∥2;Y

for all q ∈PS;Y.

Proof. Let ˆp be as in Lemma D.12. Then

|||F −bF|||L2µ(X;Y) ≤|||F −ˆp ◦EX |||L2µ(X;Y) + |||ˆp ◦EX −DY ◦b
N ◦EX |||L2µ(X;Y)

∥F −bF∥L∞
µ (X;Y) ≤∥F −ˆp ◦EX ∥L∞
µ (X;Y) + ∥ˆp ◦EX −DY ◦b
N ◦EX ∥L∞
µ (X;Y)
.
(D.23)

For the second term, we have, by (A.III) and Lemma D.10,

|||ˆp ◦EX −DY ◦b
N ◦EX |||L2µ(X;Y) ≤∥ˆp ◦EX −DY ◦b
N ◦EX ∥L∞
µ (X;Y)

= ∥ˆp −DY ◦b
N∥L∞
˜ς (D;Y)

≤∥ˆp −DY ◦b
N∥L∞
ϱ (D;Y)

≤δ
√

k|||ˆp|||L2ϱ(D;Y).

We now apply this, Lemma D.12, Theorem D.6 and the facts that α−1 ≲1 and σ′ ≥1 to obtain

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + σ′|||F −Q ◦F|||disc,µ

+ πQ

|||F −q ◦EX |||L2µ(X;Y) + σ′|||F −q ◦EX |||disc,µ


+ τ ′ + σ′

√m∥E∥2;Y

and
∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

kσ′|||F −Q ◦F|||disc,µ

+ πQ

∥F −q ◦EX ∥L∞
µ (X;Y) +
√

kσ′|||F −q ◦EX |||disc,µ


+
√

kτ ′ +

√

kσ′
√m ∥E∥2;Y

for any q ∈PS;Y, where σ′ and τ ′ are as in (D.22). Due to the various assumptions, these satisfy

σ′ ≲σ,
τ ′ ≲τ + σδ
√

k∥F∥L∞
µ (X;Y) +
σ
√m∥E∥2;Y + δ
√

k|||ˆp|||L2ϱ(D;Y).

We deduce that
|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + σ|||F −Q ◦F|||disc,µ

+ πQ

|||F −q ◦EX |||L2
µ(X;Y) + σ|||F −q ◦EX |||disc,µ


+ τ + σδ
√

k∥F∥L∞
µ (X;Y) +
σ
√m∥E∥2;Y + δ
√

k|||ˆp|||L2ϱ(D;Y)

41


---Page Break---
and

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

kσ|||F −Q ◦F|||disc,µ

+ πQ

∥F −q ◦EX ∥L∞
µ (X;Y) +
√

kσ|||F −q ◦EX |||disc,µ


+
√

kτ + σδk∥F∥L∞
µ (X;Y) +

√

kσ
√m ∥E∥2;Y + δk|||ˆp|||L2ϱ(D;Y).

We now bound the final term. Using Lemma D.10 once more, in combination with the fact that
α −δ
√

k ≳1, we see that

|||ˆp|||L2ϱ(D;Y) ≤
1
α −δ
√

k
∥DY ◦b
N∥disc,˜ς ≲∥DY ◦b
N∥disc,˜ς.

≤

v
u
u
t 1

m

m
X

i=1
∥Yi −DY ◦b
N ◦EX (Xi)∥
2
Y +
1
√m∥Y ∥2;Y

≤σ

v
u
u
t 1

m

m
X

i=1
∥Yi −0∥2
Y + τ +
1
√m∥Y ∥2;Y

≤(1 + σ)

∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y


+ τ.

Here, we also used the fact that b
N is an approximate minimizer in the fourth step, as well as the facts
that the zero network 0 ∈N and that DY is a linear map. Plugging this into the previous expressions
now gives the result.

D.7
Bounding the best polynomial approximation error terms

When σ = 1 (as will be the case when we come to prove Theorem 3.1), the error bounds in Theorem
D.13 involve best polynomial approximation error terms of the form

|||F −q ◦EX |||L2µ(X;Y) + |||F −q ◦EX |||disc,µ,
∥F −q ◦EX ∥L∞
µ (X;Y) +
√

k|||F −q ◦EX |||disc,µ

for arbitrary q ∈PS;Y. By the triangle inequality, these are bounded by

E2(F, q) := |||F −q ◦ι|||L2µ(X;Y) + |||F −q ◦ι|||disc,µ
+ |||q ◦ι −q ◦EX |||L2µ(X;Y) + |||q ◦ι −q ◦EX |||disc,µ,

E∞(F, q) := ∥F −q ◦ι∥L∞
µ (X;Y) +
√

k|||F −q ◦ι|||disc,µ

+ ∥q ◦ι −q ◦EX ∥L∞
µ (X;Y) +
√

k|||q ◦ι −q ◦EX |||disc,µ.

(D.24)

In this section, we construct a suitable polynomial q in the case where (A.II) holds and thereby derive
a bound for these term. We first require the following lemma.
Lemma D.14. Let G ∈L∞
µ (X; Y) and 0 < ϵ < 1. Then the following hold.

(a) With probability at least 1 −ϵ on the draw of the Xi, we have

|||G|||disc,µ ≤∥G∥L2µ(X;Y)/√ϵ.

(b) Suppose that m ≥2r log(2/ϵ) for some r > 0. Then, with probability at least 1 −ϵ on the draw
of the Xi, we have

|||G|||disc,µ ≤
√

2

∥G∥L∞
µ (X;Y)/√r + ∥G∥L2
µ(X;Y)

.

Proof. Observe that the random variable |||G|||2
disc,µ satisfies

E|||G|||2
disc,µ = ∥G∥2
L2µ(X;Y).

42


---Page Break---
For (a), we use Markov’s inequality to get

P

|||G|||disc,µ ≥∥G∥L2µ(X;Y)/√ϵ

≤
E|||G|||2
disc,µ
∥G∥2
L2µ(X;Y)/ϵ
= ϵ,

as required.

The proof of (b) is based on [3, Lem. 7.11]. We repeat it here for convenience. Define the random
variable Zi = ∥G(Xi)∥2
Y and observe that

E(Zi) = EX∼µ∥G(X)∥2
Y = ∥G∥2
L2µ(X;Y) =: a.

Let Xi = Zi −E(Zi) so that

|||G|||2
disc,µ = 1

m

m
X

i=1
Zi = 1

m

m
X

i=1
Xi + a.

Now let b = ∥G∥2
L∞
µ (X;Y) and observe that

Xi ≤Zi ≤b,
−Xi ≤E(Zi) ≤b,
a.s..

We also have
m
X

i=1
E(X2
i ) ≤

m
X

i=1
E(Z2
i ) ≤b

m
X

i=1
E(Zi) = abm.

Therefore, Bernstein’s inequality for bounded random variables (see, e.g., [29, Cor. 7.31]) implies
that

P

 
1
m

m
X

i=1
Xi

 ≥t

!

≤2 exp

−t2m/2

ab + bt/3



for any t > 0. We now set t = a + b/r and notice that
t2m/2
ab+bt/3 ≥3m

5r ≥log(2/ϵ). Therefore,


1
m

m
X

i=1
Xi

 < a + b/r

with probability at least 1 −ϵ. It follows that

|||G|||disc,µ ≤
p

2a + b/r ≤
√

2
√a +
p

b/r


with the same probability. Substituting the values for a and b now gives the result.

Lemma D.15. Let F ∈L∞
µ (X; Y), q ∈PS;Y be arbitrary and m ≥2r log(6/ϵ) for some r > 0
and 0 < ϵ < 1. Then

E2(F, q) ≲∥F −q ◦ι∥L∞
µ (X;Y)/√r + ∥F −q ◦ι∥L2µ(X;Y)

+
p

k/ϵ|||q|||L2ϱ(D;Y)∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(RdX )).

and

E∞(F, q) ≲(1 +
p

k/r)∥F −q ◦ι∥L∞
µ (X;Y) +
√

k∥F −q ◦ι∥L2µ(X;Y)

+ k|||q|||L2ϱ(D;Y)∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(RdX ))

+
√

k(1 +
p

k/r)|||q|||L2ϱ(D;Y)∥ιdX −ιdX ◦eDX ◦eEX ∥L∞
µ (X;ℓ∞(RdX ))

with probability at least 1 −ϵ.

43


---Page Break---
Proof. We apply part (b) of Lemma D.14 with ϵ replaced by ϵ/3 to the term |||F −q ◦ι|||disc,µ and
parts (a) and (b) of Lemma D.14 with ϵ replaced by ϵ/3 to the term |||q ◦ι −q ◦EX |||disc,µ. Using the
union bound, this gives that

|||F −q ◦ι|||disc,µ ≲∥F −q ◦ι∥L2µ(X;Y) + ∥F −q ◦ι∥L∞
µ (X;Y)/√r

|||q ◦ι −q ◦EX |||disc,µ ≲∥q ◦ι −q ◦EX ∥L2µ(X;Y)/√ϵ

|||q ◦ι −q ◦EX |||disc,µ ≲∥q ◦ι −q ◦EX ∥L∞
µ (X;Y)//√r + ∥q ◦ι −q ◦EX ∥L2µ(X;Y)

with probability at least 1 −ϵ. This yields

E2(F, q) ≲∥F −q ◦ι∥L2µ(X;Y) + ∥F −q ◦ι∥L∞
µ (X;Y)/√r

+ ∥q ◦ι −q ◦EX ∥L2µ(X;Y)/√ϵ

E∞(F, q) ≲(1 +
p

k/r)∥F −q ◦ι∥L∞
µ (X;Y) +
√

k∥F −q ◦ι∥L2µ(X;Y)

+ (1 +
p

k/r)∥q ◦ι −q ◦EX ∥L∞
µ (X;Y) +
√

k∥q ◦ι −q ◦EX ∥L2µ(X;Y)

with the same probability. It remains to bound the terms involving q ◦ι −q ◦EX . Using (A.III),
Lemma D.2 and the fact that q depends on its first dX variables only, we see that

∥q ◦ι(X) −q ◦EX (X)∥Y ≤1

2

√

k|||q|||L2ϱ(D;Y)∥ιdX (X) −EX (X)∥∞.

Therefore

∥q ◦ι −q ◦EX ∥L2µ(X;Y) ≲
√

k|||q|||L2ϱ(D;Y)∥ιdX −EX ∥L2µ(X;ℓ∞(N))

∥q ◦ι −q ◦EX ∥L∞
µ (X;Y) ≲
√

k|||q|||L2ϱ(D;Y)∥ιdX −EX ∥L∞
µ (X;ℓ∞(N))

Substituting this into the previous bounds and using the definition of EX from (A.III) now gives the
result.

We are now ready to choose the polynomial q. First, we require the following technical lemma, which
shows the existence of multi-index sets of weighted cardinality k which achieve the desired algebraic
rates of convergence.

Lemma D.16. Let f = P

ν∈F cνΨν satisfy f ∈H(b) for some b ∈ℓp(N) with b ≥0. Then there
are index sets S1, S2 ⊂F with |S1|v, |S2|v ≤k such that

∥c −cS1∥2;Y ≤C(b, p, ξ) · k1/2−1/p,
∥c −cS2∥1,v;Y ≤C(b, p, ξ) · k1−1/p,

where C(b, p, ξ) ≥0 depends on b, p and ξ only.

Proof. We first show that c ∈ℓp
v(F; Y). By definition of H(b) (see (2.3)), f is holomorphic in every
Bernstein polyellipse E(ρ) for which ρ satisfies

ρ ≥1,

∞
X

j=1

 
ρj + ρ−1
j
2
−1

!

bj ≤1.
(D.25)

Using [4, Lem. 5.3] (which is based on [102, Cor. B.2.7]) with α = β = 0, we get that ∥c0∥Y ≤1
and

∥cν∥Y ≤
Y

k∈I(ν,ρ)

ρ−νk+1
k
(ρk −1)2 (νk + 1),
ν ∈F\{0}

for all such ρ, where I(ν, ρ) = supp(ν) ∩{k : ρk > 1}. Define the sequence d0 = 1 and

dν = v2/p−1
ν
· inf






Y

k∈I(ν,ρ)

ρ−νk+1
k
(ρk −1)2 (νk + 1) : ρ satisfies (D.25)




,
ν ∈F\{0}.

44


---Page Break---
Using (C.3)-(C.4), we see that

vν =
Y

k∈supp(ν)
(2νk + 1)(5+ξ)/2

and therefore

dν ≤
Y

k∈I(ν,ρ)

ρ−νk+1
k
(ρk −1)2 (2νk + 1)γ

for all ρ satisfying (D.25) and some γ = γ(p, ξ) ≥0. We now use [4, Lem. 5.4] to deduce that
d = (dν)ν∈F ∈ℓp(N) with ∥d∥p ≤C(b, p, ξ) for some C(b, p, ξ) ≥0 depending on b, p and ξ
only. Returning to c, this gives

∥c∥p
p,v;Y =
X

ν∈F
v2−p
ν
∥cν∥p
Y ≤
X

ν∈F
|dν|p ≤C(b, p, ξ)p.

Hence c ∈ℓp
v(F; Y) with ∥c∥p,v;Y ≤C(b, p, ξ), as required.

The second step involves the application of the weighted Stechkin’s inequality (see [3, Lem. 3.12]).
This gives that

min
n
∥c −cS∥q,v;Y : S ⊂F, |S|v ≤k
o
=: σk(c)q,v;Y ≤∥c∥p,v;Yk1/q−1/p,

for any q ∈(p, 2] and k > 0. Applying this result with q = 2 implies the existence of the set S1
(recall that ∥· ∥2,v;Y = ∥· ∥2;Y) and applying it with q = 1 implies the existence of the set S2.

We now define the set

ΛHCI
n
=




ν = (νk)∞
k=1 ∈F :
Y

k:νk̸=0
(νk + 1) ≤n, νk = 0, k > n




⊂F.
(D.26)

Notice that ΛHCI
n
is isomorphic to an index set in Nn
0 by the natural restriction map.
Lemma D.17. Let k > 0 and suppose that Λ ⊇ΛHCI
n
for some n ∈N. Let f = P
ν∈F cνΨν satisfy
f ∈H(b) for some b ∈ℓp
M(N) with b ≥0. Then there exists an index set S ∈S such that

∥f −fS∥L∞
ϱ (D;Y) ≤C(b, p, ξ) ·

k1−1/p + n1−1/p
,
(D.27)

where fS = P

ν∈S cνΨν. Moreover, if Y is a Hilbert space, then we also have that

∥f −fS∥L2ϱ(D;Y) ≤C(b, p, ξ) ·

k1/2−1/p + n1/2−1/p
.
(D.28)

Here C(b, p, ξ) > 0 is a constant depending on b, p and ξ only.

Proof of Lemma D.17. The previous lemma implies that there exist index sets S1, S2 ⊂F with
|S1|v, |S2|v ≤k/2 such that

∥c −cS1∥2;Y ≤C(b, p, ξ) · k1/2−1/p,
∥c −cS2∥1,v;Y ≤C(b, p, ξ) · k1−1/p.

Now define S = S1 ∪S2 ∩Λ and notice that S ⊆Λ and |S|v ≤|S1|v + |S2|v ≤k. Hence S ∈S.
Since vν ≥uν = ∥Ψν∥L∞
ϱ (D), we have (using [5, Lem. 5.1])

∥f −fS∥L∞
ϱ (D;Y) ≤∥c −cS∥1,u;Y ≤∥c −cS2∥1,v;Y + ∥c −cΛ∥1,u;Y.

Hence, to complete the proof of the first result, we need only show that

∥c −cΛ∥1,u;Y ≤C(b, p) · n1−1/p

for some constant C(b, p) ≥0. First, by construction, Λ ⊇ΛHCI
n
contains every anchored set of size
at most n. See, e.g.. Therefore ∥c −cΛ∥1,u;Y ≤∥c −cS∥1,u;Y for any such set S. The result now
follows from [5, Cor. 8.2].

Now suppose that Y is a Hilbert space. Then Parseval’s identity gives that
∥f −fS∥L2
ϱ(D;Y) = ∥c −cS∥2;Y ≤∥c −cS1∥2;Y + ∥c −cΛ∥2;Y.

As before, it suffices to show that
∥c −cΛ∥2;Y ≤C(b, p) · n1/2−1/p.
This follows from the same approach and [5, Cor. 8.2] once more.

45


---Page Break---
D.8
Final arguments

We are now, finally, ready to prove Theorem 3.1. We first consider the case where Y is a Banach
space, then treat the case where Y is a Hilbert space afterwards.

Proof of Theorem 3.1 when Y is a Banach space. We divide the proof into a series of steps.

Step 1: Setup and DNN width/depth bounds. Let m, δ, ϵ and L be as in the theorem statement. We
may without loss of generality assume that δ ≤1/5. Now let

n =
lm

L

m
≤dX
(D.29)

and
Λ = ΛHCI
n ,

where ΛHCI
n
is as in (D.26). We also set

ξ = 1/δ −5 ≥0
(D.30)

and
k = m

cL,
(D.31)

where c ≥1 is a universal constant that will be chosen in the next step. Finally, let δ = 2−m/m and
N by the tanh DNN family (D.19), where the Nν are as in Lemma D.9 for this Λ and value of δ.

By (D.20) and the definition of ξ and k, we immediately see that

width(N) ≲(m/L)1+δ,
depth(N) ≲log(m/L).

This yields the width and depth bounds (3.1). The rest of the proof is therefore devoted to showing
the error bounds (3.3)-(3.4).

Step 2: Ensuring (D.7) holds with probability at least 1 −ϵ/2. A standard bound (see, e.g., the proof
of Lemma 6.4 in [5]) gives that N = |Λ| satisfies

log(eN) ≤4 log2(en) ≲log2(m).
(D.32)

Here, in the final step we used the fact that m ≥3 and L(m, ϵ) ≥1. This and the fact that c ≥1 also
implies that k ≤m. Therefore, the right-hand side of (D.13) with ϵ replaced by ϵ/2 satisfies

c0 · k · (log(eN) · log2(k) + log(4/ϵ)) ≲k · L(m, ϵ).
(D.33)

Hence, for sufficiently large c ≥1, we deduce that (D.13) holds with ϵ/2. Using (A.I), we see that

∥ιdX −ιdX ◦eDX ◦eEX ∥Lq
µ(X;ℓ∞(N)) ≤Lι∥IX −eDX ◦eEX ∥Lq
µ(X;X),
q = 2, ∞.
(D.34)

Hence (D.14) is implied by (3.2). We conclude from Lemma D.8 that (D.7) holds with probability at
least 1 −ϵ/2 and α ≳1.

Step 3: Error analysis. Let f ∈H(b) be the function asserted by (A.II) and define q = fS as the
polynomial asserted by Lemma D.17 with n as in (D.29). We also observe that

∥F∥L∞
µ (X;Y) = ∥f∥L∞
ς (D;Y) ≲∥f∥L∞
ϱ (D;Y) ≲1,
(D.35)

since f ∈H(b). We now apply Theorem D.13 with σ = 1 and use (D.31), the definition of δ and
(3.8) to see that

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + |||F −Q ◦F|||disc,µ
+ πQE2(F, q) + Eopt,2 + Esamp,2

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

k|||F −Q ◦F|||disc,µ
+ πQE∞(F, q) + Eopt,∞+ Esamp,∞

(D.36)

with probability at least 1 −ϵ/2, where E2(F, q) and E∞(F, q) are as in (D.24), Q : Y →eY =
DY(RdY) is any bounded linear operator and πQ = ∥Q∥Y→Y.

46


---Page Break---
Notice that Parseval’s identity and (D.35) imply that

|||q|||L2ϱ(D;Y) ≤|||f|||L2ϱ(D;Y) ≤∥f∥L∞
ϱ (D;Y) ≲1.

Hence, this, the previous bounds, Lemma D.15 with ϵ replaced by ϵ/4 and the union bound yield

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + |||F −Q ◦F|||disc,µ

+ πQ

∥F −q ◦ι∥L∞
µ (X;Y)/√r + ∥F −q ◦ι∥L2µ(X;Y)


+ πQ
p

k/ϵ∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(RdX ))
+ Eopt,2 + Esamp,2

and

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

k|||F −Q ◦F|||disc,µ

+ πQ

(1 +
p

k/r)∥F −q ◦ι∥L∞
µ (X;Y) +
√

k∥F −q ◦ι∥L2µ(X;Y)


+ πQk∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(RdX ))

+ πQ
√

k(1 +
p

k/r)∥ιdX −ιdX ◦eDX ◦eEX ∥L∞
µ (X;ℓ∞(RdX ))
+ Eopt,∞+ Esamp,∞

with probability at least 1 −3ϵ/4, for any r such that m ≥2r log(24/ϵ). In particular, we may
choose r = k due the definition of k (D.31). We next bound the discrete error |||F −Q ◦F|||disc,µ.
Applying Lemma D.14 with ϵ replaced by ϵ/8 and the union bound, we see that

|||F −Q ◦F|||disc,µ ≲∥F −Q ◦F∥L2µ(X;Y)/√ϵ

|||F −Q ◦F|||disc,µ ≲∥F −Q ◦F∥L∞
µ (X;Y)/√r + ∥F −Q ◦F∥L2µ(X;Y)

with probability at least 1 −ϵ/4, provided m ≥2r log(16/ϵ). In particular, we may take r = k once
more. Substituting this into the previous expressions, setting r = k throughout, using the union
bound once more and recalling (3.7), (D.31) and (D.34), we deduce that

|||F −bF|||L2µ(X;Y) ≲∥F −Q ◦F∥L2µ(X;Y)/√ϵ

+ πQ

∥F −q ◦ι∥L∞
µ (X;Y)/
p

m/L + ∥F −q ◦ι∥L2µ(X;Y)


+ (πQ/aY) · EX,2 + Eopt,2 + Esamp,2

and

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
p

m/L∥F −Q ◦F∥L2µ(X;Y)

+ πQ

∥F −q ◦ι∥L∞
µ (X;Y) +
p

m/L∥F −q ◦ι∥L2µ(X;Y)


+ (πQ/aY) · EX,∞+ Eopt,∞+ Esamp,∞

with probability at least 1 −ϵ. This holds for any bounded linear operator Q : Y →eY = DY(RdY).
We now set Q = DY ◦EY, which is linear and bounded by (A.IV) with πQ = ∥DY ◦EY∥Y→Y = aY
by definition. Using this, we observe that

∥F −Q ◦F∥Lq
µ(X;Y) = ∥IY −DY ◦EY∥Lq
F ♯µ(Y;Y),
q = 2, ∞.

Substituting this into the previous expressions and recalling (3.7) now gives

|||F −bF|||L2µ(X;Y) ≲aY

∥F −q ◦ι∥L∞
µ (X;Y)/
p

m/L + ∥F −q ◦ι∥L2µ(X;Y)


+ EX,2 + EY,2 + Eopt,2 + Esamp,2

∥F −bF∥L∞
µ (X;Y) ≲aY

∥F −q ◦ι∥L∞
µ (X;Y) +
p

m/L∥F −q ◦ι∥L2µ(X;Y)


+ EX,∞+ EY,∞+ Eopt,∞+ Esamp,∞.

(D.37)

47


---Page Break---
Step 4: Bounding the polynomial error terms. It remains to bound the error terms F −q ◦ι in (D.37).
Using (A.I), (A.II) and Lemma D.17, we now notice that

∥F −q ◦ι∥L∞
µ (X;Y) = ∥f −q∥L∞
ς (D;Y) ≲∥f −q∥L∞
ϱ (D;Y) ≲C(b, p, ξ) · (k1−1/p + n1−1/p).

Recall that n ≥k. Therefore, using this, (D.31) and (3.6), we see that

aY

∥F −q ◦ι∥L∞
µ (X;Y)/
p

m/L + ∥F −q ◦ι∥L2µ(X;Y)

≲aY∥F −q ◦ι∥L∞
µ (X;Y)

≲aY · C(b, p, ξ) · (m/L)1−1/p

= Eapp,2

and likewise for the L∞
µ -norm bound. Combining this with (D.37) now completes the proof.

Proof of Theorem 3.1 when Y is a Hilbert space. The two differences in theorem statement when Y
is a Hilbert space are: (i) the L2-norm error is with respect to the stronger Bochner norm, and (ii) the
approximation error terms Eapp,q, q = 2, ∞, are smaller by a factor of 1/2 in the exponent (recall
(3.6)). We treat both issues separately.

For the (i), we commence with the supporting results in §D.2. First, we note that Lemmas D.1 and D.2
also hold in the Bochner L2-norm, since these results already give upper bounds involving the Pettis
L2-norm. Next, we observe that the proof of Lemma D.3 is readily adapted to yield an equivalent
result in the Bochner L2-norm with the same constant δ. The same therefore applies to Lemma D.4.

We next consider the analysis of (D.3) in §D.3. If we replace (D.7) by the condition

∥p −q∥disc,˜ς ≥α max{∥p −q∥L2ϱ(D;Y), ∥p −q∥L2
˜ς(D;Y)},
∀p, q ∈PS; e
Y
(D.38)

then the proof of Lemma D.5 yields the same error bounds, except with the Pettis L2-norm replaced
by the Bochner L2-norm. Theorem D.6 is likewise modified to provide a bound in the Bochner
L2-norm.

Up to this point, we have not used the fact that Y is a Hilbert space. We now need this property.
As in §D.4, the next step is to establish that (D.38) holds with high probability. Lemma D.7 is
unchanged, therefore our focus is on Lemma D.8. We now describe the steps needed to modify
the proof of this lemma to assert (D.38) subject to the same conditions (D.13)-(D.14). First, let
{φi}dY
i=1 be an orthonormal basis of eY and write each coefficient cνi of the function h in (D.16) as
cνi = PdY
j=1 bijφj for scalars bij. Then it is a short exercise to write

∥h∥2
disc,˜ς = ∥Ac∥2
2;Y =

dY
X

j=1
∥Abj∥2
2,

where bj = (bij)N
i=1. Since bij = 0, ∀j, whenever cνi = 0, this vector also satisfies ∥bj∥0,v ≤2k.
Hence, by Lemma D.7 and Parseval’s identity twice,

∥h∥2
disc,˜ς ≥

dY
X

j=1
(θ−−(1 + θ+c5δ)∥bj∥2
2

= (θ−−(1 + θ+c5δ)

N
X

i=1
∥cνi∥2
2

= (θ−−(1 + θ+c5δ)∥h∥2
L2ϱ(D;Y).

We now use the bounds (D.15) (which are unchanged) and set δ = c3/(2(1 + c4)c5) once more to get

∥h∥2
disc,˜ς ≥c3/2∥h∥2
L2
ϱ(D;Y).

This gives the desired result.

This completes the changes needed in order to analyze the polynomial training problem (D.3). We
next consider the DNN training problem (2.5). §D.5 remains unchanged. After reviewing their proofs,

48


---Page Break---
we see that Lemma D.12 and Theorem D.13 both hold with Pettis norms replaced by Bochner norms
whenever (D.38) holds instead of (D.7). Finally, we also observe that Lemma D.15 also holds with
Pettis norms replaced by Bochner norms, both in the bound and in the definition (D.24) of E2(F, q)
and E∞(F, q).

Having completed the changes needed in all the preparatory results, we now follow the same steps as
above in the Banach space case. Steps 1 and 2 are unchanged. For Step 3, we go through and replace
Pettis norms by Bochner norms throughout. Using this, we obtain (D.37), except with the Bochner
norm on the left-hand side in the first inequality, i.e.,

∥F −bF∥L2µ(X;Y) ≲aY

∥F −q ◦ι∥L∞
µ (X;Y)/
p

m/L + ∥F −q ◦ι∥L2µ(X;Y)


+ EX,2 + EY,2 + Eopt,2 + Esamp,2

∥F −bF∥L∞
µ (X;Y) ≲aY

∥F −q ◦ι∥L∞
µ (X;Y) +
p

m/L∥F −q ◦ι∥L2µ(X;Y)


+ EX,∞+ EY,∞+ Eopt,∞+ Esamp,∞.

(D.39)

This concludes the changes needed to address (i). To address (ii), we bound the terms F −q ◦ι.
Using (A.I), (A.II), Lemma D.17, the fact that Y is a Hilbert space and the definitions (D.31) and
(D.29) of k and n, we see that

∥F −q ◦ι∥L∞
µ (X;Y) = ∥f −q∥L∞
ς (D;Y) ≲∥f −q∥L∞
ϱ (D;Y) ≲C(b, p, ξ) · (m/L)1−1/p

and

∥F −q ◦ι∥L2µ(X;Y) = ∥f −q∥L2ς(D;Y) ≲∥f −q∥L2ϱ(D;Y) ≲C(b, p, ξ) · (m/L)1/2−1/p

Substituting this into (D.39) and recalling (3.6) now completes the proof.

Remark D.18 (Differences between the Banach and Hilbert space case) Having seen the proof,
we now summarize these differences as follows. First, the matter of whether the Pettis versus Bochner
norm can be used reduces to the choice of such norm in the discrete metric inequality (D.7). When
Y is a Banach space, we are able to establish this in terms of the Pettis norm subject to a log-linear
scaling between m and k (see Lemma D.8 and (D.13)). However, when Y is also a Hilbert space,
we can establish the stronger version (D.38) of this inequality by exploiting the additional structure.
This, in short, is what leads to the stronger norm bound in this case.

Second, in the Hilbert space case, we get an improved approximation error. This stems from (D.17)
and, specifically, the fact that when Y is a Hilbert space we may use Parseval’s identity in the Bochner
space L2
ϱ(D; Y) to bound the L2
ϱ-norm error term via (D.28). This is not possible when Y is a Banach
space, so we settle for bounding this term via (D.27) instead.

E
Proof of Theorem 3.2

E.1
Setup

As in §D.1, let Λ ⊂F with supp(ν) ⊆{1, . . . , dX }, ∀ν ∈Λ, and write N = |Λ|. Let S be as in
(D.1), r ∈N, r > max{m, k} (its precise value will be chosen later in the proof) and define the set

Γ = Γo ∪
[

S∈S
S ⊂F,
where Γo = {ei : i = 1, . . . , r}.
(E.1)

Finally, let 0 < δ < 1 and consider the family No and the tanh DNNs {Nν}ν∈Γ whose existence is
implied by Lemma D.9. We will specify Λ, k, r and δ later in the proof.

Next, let
ˆp =
X

ν∈S
ˆcνΨν

be any minimizer of (D.3), where |S|v ≤k and define

˜p =
X

ν∈S
ˆcνNν.

49


---Page Break---
Let
B =
1
√r

 
Nej ◦EX (Xi)
m,r
i,j=1 .

Now, eY := DY(RdY) is a finite-dimensional subspace of the Banach space Y. Hence, for any Y ∈Y
there exists a closest point eY ∈eY, i.e., a point satisfying

∥Y −eY ∥Y = inf
n
∥Y −Z∥Y : Z ∈eY
o
.

Given Y1, . . . , Ym, let eY1, . . . , eYm ∈eY be the corresponding closest points. Now define

e =
1
√r


eYi −˜p ◦EX (Xi)
m

i=1 ∈eYm

and

¯p = ˜p +

r
X

i=1
(B†e + yz)iNei,
(E.2)

where z ∈N(B)\{0} and y ∈eY, ∥y∥Y = 1, are arbitrary. Note that such a z exists, since r > m
by assumption. Notice that
¯p =
X

ν∈S∪Γo
¯cνNν

for coefficients ¯cν ∈eY. Therefore, we can write

¯p = DY ◦b
N,
(E.3)

where

b
N = C





Nν1
...
Nν|S∪Γo|
0
...
0





(E.4)

and C ∈RdY×(⌊k⌋+r). Finally, we define the approximation

bF = DY ◦b
N ◦EX
(E.5)

and, for convenience,
b

F = ˆp ◦EX .

E.2
Estimation of the DNN minimizer

Lemma E.1 ( b
N is a minimizer). If B is full rank, then

bF(Xi) = eYi,
∀i = 1, . . . , m.

Therefore, b
N is a minimizer of (2.5).

Proof. Observe that

bF(Xi) = p ◦EX (Xi)

= ˜p ◦EX (Xi) + √r

r
X

i=1
(B)ij(B†e + yz)j

= ˜p ◦EX (Xi) + √r(B(B†e + yz))i
= ˜p ◦EX (Xi) + √r(e)i

= eYi.

50


---Page Break---
Here, in the penultimate step we use the facts that Byz = yBz = 0 since z ∈N(B) and BB† = I
since r ≥m and B is full rank by assumption. This gives the first result.

For the second result, we recall that eYi is a closest point to Yi from eY = DY(RdY). Therefore, for
any DNN N,

1
m

m
X

i=1
∥Yi −DY ◦N ◦EX (Xi)∥2
Y ≥1

m

m
X

i=1
∥Yi −eYi∥
2
Y = 1

m

m
X

i=1
∥Yi −DY ◦b
N ◦EX (Xi)∥
2
Y

as required.

Lemma E.2 (Bounding bF in terms of

b

F). Suppose that (D.7) holds with α ≥c0 and α −δ
√

k ≥c1,
and also that
√r∥ιdX −ιdX ◦eDX ◦eEX ∥L2µ(X;ℓ∞(N)) ≤c2,

where c0, c1, c2 > 0 are suitable universal constants. Then the approximation bF satisfies

|||F −bF|||L2µ(X;Y) ≲|||F −

b

F|||L2µ(X;Y)

+ (1 + δ√r)δ
√

k

1 +
√m
√rσmin(B)

 
∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y



+
√m(1 + δ√r)
√rσmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y +
1
√m∥E∥2;Y





+ (1 + δ√r)∥z∥2

and

∥F −bF∥L∞
µ (X;Y) ≲∥F −

b

F∥L∞
µ (X;Y)

+ (1 + δ)√rδ
√

k

1 +
√m
√rσmin(B)

 
∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y



+
√m(1 + δ)

σmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y +
1
√m∥E∥2;Y





+ (1 + δ)√r∥z∥2.

Proof. By the triangle inequality,

|||F −bF|||L2µ(X;Y) ≤|||F −

b

F|||L2µ(X;Y) + |||

b

F −bF|||L2µ(X;Y)

∥F −bF∥L∞
µ (X;Y) ≤∥F −

b

F∥L∞
µ (X;Y) + ∥

b

F −bF∥L∞
µ (X;Y)
.
(E.6)

Consider the second term. We have

|||

b

F −bF|||L2µ(X;Y) = |||ˆp −DY ◦b
N|||L2
˜ς(D;Y) ≤|||ˆp −˜p|||L2
˜ς(D;Y) + |||q|||L2
˜ς(D;Y)

∥

b

F −bF∥L∞
µ (X;Y) = ∥ˆp −DY ◦b
N∥L∞
˜ς (D;Y) ≤∥ˆp −˜p∥L∞
˜ς (D;Y) + ∥q∥L∞
˜ς (D;Y)
,
(E.7)

where q = ¯p −˜p = Pr
i=1(B†e + yz)iNei. Lemma D.10 and (A.III) give that

|||ˆp −˜p|||L2
˜ς(D;Y) = ∥ˆp −˜p∥L∞
˜ς (D;Y) ≤δ
√

k|||ˆp|||L2ϱ(D;Y).
(E.8)

Now consider the other term in (E.7). Define ˜q = Pr
i=1(B†e + yz)iΨei. Then Lemma D.10 and
(A.III) once more give that

|||q|||L2
˜ς(D;Y) ≤|||˜q|||L2
˜ς(D;Y) + |||q −˜q|||L2
˜ς(D;Y) ≤|||˜q|||L2
˜ς(D;Y) + δ√r|||˜q|||L2ϱ(D;Y)

∥q∥L∞
˜ς (D;Y) ≤∥˜q∥L∞
˜ς (D;Y) + ∥q −˜q∥L∞
˜ς (D;Y) ≤∥˜q∥L∞
˜ς (D;Y) + δ√r|||˜q|||L2ϱ(D;Y)
.

51


---Page Break---
Here, we also used the fact that |Γo| = r. We now apply Lemmas D.1 and D.4 and (A.III) once more
to get

|||q|||L2
˜ς(D;Y) ≲(1 + δ√r)|||˜q|||L2ϱ(D;Y),
∥q∥L∞
˜ς (D;Y) ≲(1 + δ)√r|||˜q|||L2ϱ(D;Y).
(E.9)

We next analyze the term |||˜q|||L2ϱ(D;Y). Let y∗∈Y∗. Since y∗((B†e+yz)i) = (B†y∗(e)+y∗(y)z)i,
we have

y∗(˜q) =

r
X

i=1
(B†y∗(e) + y∗(y)z)iΨei.

Hence, by Parseval’s identity,

|||˜q|||L2ϱ(D;Y) =
sup
y∗∈B(Y∗)
∥y∗(˜q)∥L2(D)

=
sup
y∗∈B(Y∗)
∥B†y∗(e) + y∗(y)z∥2

≤
1
σmin(B)
sup
y∗∈B(Y∗)
∥y∗(e)∥2 +
sup
y∗∈B(Y∗)
|y∗(y)|∥z∥2

=
1
σmin(B)|||e|||2;Y + ∥z∥2.

We now use the definition of e and the inequality |||e|||2;Y ≤∥e∥2;Y to obtain

|||˜q|||L2ϱ(D;Y) ≤
√m
√rσmin(B)

v
u
u
t 1

m

m
X

i=1
∥eYi −˜p ◦EX (Xi)∥
2
Y + ∥z∥2.

We next apply the triangle inequality and Lemma D.10 once more to get

|||˜q|||L2ϱ(D;Y) ≤
√m
√rσmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −ˆp ◦EX (Xi)∥
2
Y + ∥ˆp −˜p∥disc,˜ς



+ ∥z∥2

≤
√m
√rσmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −ˆp ◦EX (Xi)∥
2
Y + δ
√

k|||ˆp|||L2ϱ(D;Y)



+ ∥z∥2.

Combining this with (E.7) and (E.9), we deduce that

|||

b

F −bF|||L2µ(X;Y) ≲(1 + δ√r)

"

δ
√

k

1 +
√m
√rσmin(B)


|||ˆp|||L2ϱ(D;Y) + ∥z∥2

+
√m
√rσmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −ˆp ◦EX (Xi)∥
2
Y +
1
√m∥E∥2;Y





#

and

∥

b

F −bF∥L∞
µ (X;Y) ≲(1 + δ)√r

"

δ
√

k

1 +
√m
√rσmin(B)


|||ˆp|||L2ϱ(D;Y) + ∥z∥2

+
√m
√rσmin(B)





v
u
u
t 1

m

m
X

i=1
∥eYi −ˆp ◦EX (Xi)∥
2
Y +
1
√m∥E∥2;Y





#

.

52


---Page Break---
It remains to bound the term |||ˆp|||L2ϱ(D;Y). Using (D.7) and the fact that ˆp is a minimizer of (D.3) and
that the zero polynomial is feasible for (D.3), we get

|||ˆp|||L2ϱ(D;Y) ≲|||ˆp|||disc,˜ς

≤

v
u
u
t 1

m

m
X

i=1
∥Yi −ˆp ◦EX (Xi)∥2
Y +
1
√m∥Y ∥2;Y

≤
2
√m∥Y ∥2;Y

≤2∥F∥L∞
µ (X;Y) +
2
√m∥E∥2;Y

Combining this with the previous bound and (E.6) now completes the proof.

E.3
Estimation of σmin(B)

Recall that a matrix A ∈Rm×n is a subgaussian random matrix if its entries are i.i.d. subgaussian
random variables with mean zero and variance one (see, e.g., [29, Def. 9.1]). The following result
can be found in, e.g., [29, Ex. 9.3].
Lemma E.3 (Smallest singular value of a subgaussian random matrix). Let A ∈Rm×n be a
subgaussian random matrix and σmin be the smallest singular value of
1
√mA. Then, for all 0 < t < 1,

P

σmin ≤1 −c1
p

n/m −t

≤2 exp(−c2mt2),

where c1, c2 > 0 are universal constants.
Lemma E.4 (Bounding σmin(B)). Suppose that √mδ ≤√ω/8,

3(5+ξ)/2√r

2
∥ιdX −ιdX ◦eDX ◦eEX ∥L∞
µ (X,ℓ∞(N)) ≤
√ω

8 ,
(E.10)

where ω is the variance of the univariate probability measure as specified in Theorem 3.2 and

dX ≥r ≥c (m + log(2/ϵ))
(E.11)

for some universal constant c ≥1. Then, with probability at least 1 −ϵ, the matrix B is full rank and

σmin(B) ≥√ω/4.

Proof. Define the matrices

B′ =
1
√r

 
Ψej ◦EX (Xi)
m,r
i,j=1 ,
B′′ =
1
√r

 
Ψej ◦ιdX (Xi)
m,r
i,j=1 .

Then, since r ≥m,

σmin(B) = inf

∥B⊤d∥2 : d ∈Cm, ∥d∥2 = 1
	

≥σmin(B′) −∥(B −B′)⊤∥2
= σmin(B′) −∥B −B′∥2
≥σmin(B′′) −∥B −B′∥2 −∥B′ −B′′∥2.

Now, for any c ∈Cr,

∥(B −B′)c∥2
2 = 1

r

m
X

i=1




r
X

j=1

 
Nej(xi) −Ψej(xi)

cj





2

≤1

r

m
X

i=1
δ2∥c∥2
1

≤mδ2∥c∥2
2.

53


---Page Break---
We deduce that ∥B −B′∥2 ≤√mδ ≤√ω/8. Hence

σmin(B) ≥σmin(B′′) −√ω/8 −∥B′ −B′′∥2.

Now let c ∈Cr and p = Pr
i=1 ciΨei be the corresponding polynomial. Then, by (A.I), (A.III) and
Lemma D.2,

∥(B′ −B′′)c∥2 =

v
u
u
t1

r

m
X

i=1

p ◦ιdX (Xi) −p ◦ιdX ◦eDX ◦eEX (Xi)

2

≤Lip(p, B∞(N), R)∥ιdX −ιdX ◦eDX ◦eEX ∥L∞
µ (X,ℓ∞(N))

≤3(5+ξ)/2√r

2
∥p∥L2ϱ(D)∥ιdX −ιdX ◦eDX ◦eEX ∥L∞
µ (X,ℓ∞(N))

≤
√ω

8 ∥c∥2.

Here, in the third step we used the fact that |Γo|v = 35+ξr, since uei =
√

3. We deduce that
∥B′ −B′′∥2 ≤√ω/8 and therefore

σmin(B) ≥σmin(B′′) −√ω/4.

It remains to show that σmin(B′′) ≥√ω/2 with high probability. By construction, Ψej(x) =
√

3xj.
Now recall that the pushforward ς is a tensor-product of a univariate probability measure supported
in [−1, 1] with mean zero and variance ω > 0. Therefore

(B′′)⊤=
√

3
√ω
√r A,

where A = (ι(Xj)i/√ω)r,m
i,j=1 ∈Rr×m. By construction, the entries of A are i.i.d. subgaussian
random variables with mean zero and variance one. Hence A is a subgaussian random matrix. We
now apply Lemma E.3. Let t = 1/4 and observe that

r ≥4c2
1m,
r ≥16

c2
log(2/ϵ),

by assumption. Therefore,

P(σmin(B′′) ≤√ω/2) ≤P(σmin(A/√r) ≤1/(2
√

3)) ≤ϵ.

This gives the result.

E.4
Final arguments

We are now ready to complete the proof of Theorem 3.2.

Proof of Theorem 3.2, Statement (A). We divide the proof into a series of steps.

Step 1: Setup. Let m, δ, ϵ and L be as in the theorem statement. We once more assume without loss
of generality that δ ≤1/5. Let n be as in (D.29) and Λ = ΛHCI
n , let ξ be as in (D.30) and

k = m

c1L,
(E.12)

where c1 ≥1 is a constant that will be chosen in the next step. Let

δ = min

2−m/r2, √ω/(8√m)
	
,

where ω is the variance of the univariate probability measure and

r = ⌈c2(m + log(1/ϵ))⌉,
(E.13)

where c2 ≥2 will also be chosen in the next step. Note that dX ≥r > m by assumption. Finally, let
S be as in (D.1), Γ be as in (E.1) and {Nν}ν∈Γ be the corresponding family of tanh DNNs ensured
by Lemma D.9. Finally, let bF be given by (E.5), with DNN b
N as in (E.4).

54


---Page Break---
Step 2: b
N is a minimizer. By construction,

width( b
N) ≲(k + r) · m(Γ),
depth( b
N) ≲log(k).
If ν = ei, then ∥ν∥1 = 1. Hence m(Γ) ≤1 + maxS∈S m(S) ≤1 + k1/(5+ξ) = 1 + kδ (see §D.5).
Using the values of k and r, we see that

width( b
N) ≲(m + log(1/ϵ))(m/L)δ,
depth( b
N) ≲log(m/L).

Therefore, b
N ∈N is feasible for (2.5). Lemma E.1 now implies that b
N is a minimizer, provided B
is full rank. We will show that this holds in the next step.

Step 3: Ensuring that σmin(B) ≥√ω/4 with probability at least 1 −ϵ/4. We seek to use Lemma
E.4. By definition of δ, we have that √mδ ≤√ω/8. Now (D.34), (E.13) and (3.9) imply that
(E.10)-(E.11) hold, the latter with ϵ replaced by ϵ/4. Hence Lemma E.4 implies the result.

Step 4: Ensuring (D.7) holds with probability at least 1 −ϵ/4. This step is very similar to Step 2 of
the proof of Theorem 3.1. The only difference comes in the estimation of N in (D.32), since now
N = |Γ| ≤|Λ| + r. Since m ≥3, we have
log(eN) ≤log(e|Λ|(r + 1)) ≲log2(m) + log(m + log(1/ϵ)).
If m ≤log(1/ϵ) then

log(eN) ≲log2(m) + log(log(1/ϵ)) ≤log2(m) +
p

log(1/ϵ),
where in the second step we use the fact that log(t) ≤
√

t for t > 0. Conversely, if m ≥log(1/ϵ),
then
log(eN) ≲log2(m) ≤log2(m) +
p

log(1/ϵ).
Therefore, (D.33) with ϵ/4 reads

c0 · k · (log(eN) · log2(k) + log(6/ϵ)) ≲k ·

log2(m)

log2(m) +
p

log(1/ϵ)

+ log(1/ϵ)


≲k ·
 
log4(m) + log(1/ϵ)

.

= k · L(m, ϵ).
Hence, due to the definition of k, we get that (D.13) holds once more with probability at least 1 −ϵ/4.
The remainder of this step is identical to Step 2 of the proof of Theorem 3.1.

Step 5: Error analysis. We now apply Lemma E.2 to the approximation bF. Since σmin(B) ≳√ω ≳
1, δ ≤δ√r ≤1 and r ≥m, we deduce that

|||F −bF|||L2µ(X;Y) ≲|||F −

b

F|||L2µ(X;Y) + δ
√

k

∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y



+ ∥z∥2 +

v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y +
1
√m∥E∥2;Y

and

∥F −bF∥L∞
µ (X;Y) ≲∥F −

b

F∥L∞
µ (X;Y) + √rδ
√

k

∥F∥L∞
µ (X;Y) +
1
√m∥E∥2;Y



+ √r∥z∥2 + √m





v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y +
1
√m∥E∥2;Y





with probability at least 1 −ϵ/2, where

b

F = ˆp ◦EX and ˆp is the corresponding minimizer of (D.3).
We now appeal to Theorem D.6 with σ = 1 and τ = 0, recalling that α ≳1 and δ
√

k ≲1, to get that

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + |||F −Q ◦F|||disc,µ

+ πQ

|||F −q ◦EX |||L2
µ(X;Y) + |||F −q ◦EX |||disc,µ


+
1
√m∥E∥2;Y + δ
√

k∥F∥L∞
µ (X;Y)

+ ∥z∥2 +

v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y

55


---Page Break---
and, since δ√r ≲1,

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) +
√

k|||F −Q ◦F|||disc,µ

+ πQ

∥F −q ◦EX ∥L∞
µ (X;Y) +
√

k|||F −q ◦EX |||disc,µ


+ ∥E∥2;Y + √rδ
√

k∥F∥L∞
µ (X;Y)

+ √r∥z∥2 + √m

v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y

for any q ∈PS;Y and linear operator Q : Y →eY = DY(RdY) with πQ = ∥Q∥Y→Y. Consider the
final term. Since eYi ∈eY is the closest point to Yi = F(Xi) + Ei we have

∥eYi −Yi∥Y ≤∥F(Xi) −Q ◦F(Xi)∥Y + ∥Ei∥Y.

We now use this, the fact that ˆp is a minimizer and Q ◦q is feasible in combination with triangle
inequality to get
v
u
u
t 1

m

m
X

i=1
∥eYi −

b

F(Xi)∥
2

Y ≤

v
u
u
t 1

m

m
X

i=1
∥Yi −Q ◦q ◦EX (Xi)∥2
Y + |||F −Q ◦F|||disc,µ +
1
√m∥E∥2;Y

≤|||F −Q ◦q ◦EX |||disc,µ + |||F −Q ◦F|||disc,µ +
2
√m∥E∥2;Y

≤πQ|||F −q ◦EX |||disc,µ + 2|||F −Q ◦F|||disc,µ +
2
√m∥E∥2;Y.

Now let f ∈H(b) be the function asserted by (A.II) and q = fS be the polynomial asserted by
Lemma D.17. Substituting this into the previous expression, recalling (D.35) and using the definition
of δ, we deduce that

|||F −bF|||L2µ(X;Y) ≲|||F −Q ◦F|||L2µ(X;Y) + |||F −Q ◦F|||disc,µ

+ πQE2(F, q) + ∥z∥2 + 2−m + Esamp,2

∥F −bF∥L∞
µ (X;Y) ≲∥F −Q ◦F∥L∞
µ (X;Y) + √m|||F −Q ◦F|||disc,µ

+ πQ eE∞(F, q) + √r∥z∥2 + 2−m + E′
samp,∞

with probability at least 1 −ϵ/2. Here E2(F, q) is as in (D.24), eE∞(F, q) is as in (D.24) with k
replaced by m and E′
samp,∞= ∥E∥2;Y =
√

LEsamp,∞.

Now observe that the first bound is identical to the corresponding bound in (D.36), except with Eopt,2
replaced by ∥z∥2 + 2−m. Following the same arguments as in Step 3 of the proof of Theorem 3.1,
this gives the corresponding bound in (D.37), which is

|||F −bF|||L2µ(X;Y) ≲aY

∥F −q ◦ι∥L∞
µ (X;Y)/
p

m/L + ∥F −q ◦ι∥L2µ(X;Y)


+ EX,2 + EY,2 + ∥z∥2 + 2−m + Esamp,2.
(E.14)

The second bound above is identical to the corresponding bound in (D.36), except with Eopt,∞
replaced by √r∥z∥2+2−m, Esamp,∞and E∞(F, q) replaced by E′
samp,∞and eE∞(F, q), respectively,
and with
√

k replaced by √m. We once more follow the same arguments as in Step 3, with these
changes. This yields the corresponding version of (D.37), which is

∥F −bF∥L∞
µ (X;Y) ≲aY
√

L∥F −q ◦ι∥L∞
µ (X;Y) + √m∥F −q ◦ι∥L2
µ(X;Y)


+ E′
X,∞+ E′
Y,∞+ √r∥z∥2 + 2−m + E′
samp,∞.
(E.15)

Here E′
X,∞= LEX,∞and E′
Y,∞=
√

LEY,∞.

56


---Page Break---
Having done this, we then use the bounds from Step 4 of the proof of Theorem 3.1, to get

|||F −bF|||L2µ(X;Y) ≲Eapp,2 + EX,2 + EY,2 + ∥z∥2 + 2−m + Esamp,2

∥F −bF∥L∞
µ (X;Y) ≲E′
app,∞+ E′
X,∞+ E′
Y,∞+ √r∥z∥2 + 2−m + E′
samp,∞,

where E′
app,∞=
√

LEapp,∞.

Step 6: Existence of uncountably many minimizers. Let z = (zi)r
i=1 ∈N(B)\{0} be any vector
and consider z1 = θ1z and z2 = θ2z for θ1, θ2 ∈[−1, 1] with θ1 ̸= θ2. Then these vectors define
functions ¯p1 and ¯p2 as in (E.2) and DNNs b
N1 and b
N2 as in (E.3). Suppose that b
N1 = b
N2. Then,
since DY is linear, we have that ¯p1 = ¯p2. But then, by definition and the fact that y ∈eY\{0}, we
must have

0 =

r
X

i=1
(z1 −z2)iNei = (θ1 −θ2)

r
X

i=1
ziNei.

Suppose that θ1 > θ2 without loss of generality and let x ∈D be the vector (sign(zi))∞
i=1. Then,
Ψei(x) =
√

3sign(zi) and therefore

0 ≥(θ1 −θ2)
√

3∥z∥1 −δr

≥(θ1 −θ2)
√

3∥z∥2 −δr

= (θ1 −θ2)
√

3∥z∥2 −2−m/r

,

since ∥Nei −Ψei∥L∞
ϱ (D) ≤δ and δ ≤2−m/r2 by definition. We now choose z with ∥z∥2 =
2−m/r ≤2−m. Note that this choice of z does not change the error bound, except for a constant. It
also yields
0 ≥(θ1 −θ2)(
√

32−m −2−m) > 0

which is a contradiction. Hence b
N1 ̸= b
N2. Thus, we have shown that any θ ∈[−1, 1] leads to a
distinct DNN minimizer that satisfies the desired bounds. We get the result.

Step 7: Modifications when Y is a Hilbert space. The modifications required when Y is a Hilbert
space are identical to those needed in the proof of Theorem 3.1 (see §D.8). We omit the details.

Proof of Theorem 3.2, Statement (B). Let N = Nθ : RdX →RdY be a tanh DNN, where θ ∈RD
are the network parameters (weights and biases). Let ∥· ∥(dX ), ∥· ∥(dY) and ∥· ∥(D) be arbitrary
norms on RdX , RdY and RD, respectively. Then, since the activation function is a Lipschitz function,
we have
∥Nθ′(x) −Nθ(x)∥(dY) ≤cθ∥θ′ −θ∥(D)(∥x∥(dX ) + 1),

where cθ > 0 is a constant depending on θ.

Now let b
N = b
N ˆθ and bF = DY ◦b
N ◦EX and consider F = DY ◦N ◦EX for N = Nθ. Since
DY : RdY →Y is linear and therefore bounded, we have

∥bF(X) −F(X)∥Y ≤c ˆθ∥DY∥(RdY ,∥·∥(dY ))→Y∥ˆθ −θ∥(D)

∥EX (X)∥(dX ) + 1

.

We deduce that

||| bF −F|||L2µ(X;Y) ≤∥bF −F∥L∞
µ (X;Y)

≤c ˆθ∥DY∥(RdY ,∥·∥(dY ))→Y∥ˆθ −θ∥(D)

∥EX ∥L∞
µ (X;(RdX ,∥·∥(dX )) + 1

.

Therefore, there exists a neighbourhood around ˆθ for which

||| bF −F|||L2µ(X;Y) ≤∥bF −F∥L∞
µ (X;Y) ≤τo

for all parameters θ in the neighbourhood. The result now follows.

Proof of Theorem 3.2, Statement (C). By construction, the DNN b
N defined in (E.4) contains sub-
networks that compute the DNNs Nei, i = 1, . . . , r, which themselves are approximations to the
Legendre polynomials Ψei. The construction of these subnetworks was described in the proof of

57


---Page Break---
Lemma D.9 as the composition of an affine map defined by the fundamental theorem of algebra and a
tanh DNN that approximately multiplies m(Γ) numbers. In this specific case, we have

Ψei(x) = xi.

Therefore, the corresponding affine map (D.18) Aei : RdX →Rm(Γ) has a bias vector that is
all ones, except for a single entry that has value zero. We deduce that the bias vector b of the
full DNN (E.4) contains a subblock of size rm(Γ) that is all ones, except for r zeroes. There are
 rm(Γ)
r

rearrangements of this subblock, each of which leading to a bias vector b′ with ∥b −b′∥≳1.
Moreover, b′ leads to the same DNN, after permuting the various weight matrices in the corresponding
way. Indeed, if P is a permutation matrix, then σ(W x + b) = P −1σ(P W x + P b), since σ acts
componentwise.

It remains to bound
 rm(Γ)
r

from below. We have
rm(Γ)
r


≥(rm(Γ))r

rr
= m(Γ)r.

By (E.13), we have that r ≥2m. Now, by the definition (E.1) of Γ,

m(Γ) ≥max
S∈S m(S),

where S is as in (D.1) for k as in (E.12) and Λ = ΛHCI
n
as in (D.26) with n as in (D.29). Now consider
the set S = {le1} for some l ∈N. Then

|S|v = v2
ν = (2l + 1)5+ξ = (2l + 1)1/δ.

Therefore, S ∈S provided (2l + 1)1/δ ≤k and l ≤n. Set l = ⌊(kδ −1)/2⌋and observe that l ≤n
since k ≤n and δ ≤1/5 by assumption. Therefore S ∈S. Using the definition of k, we get that

m(Γ) ≥m(S) = l ≥(m/(c4L))δ

for all sufficiently large m. The result now follows.

F
Proof of Theorem 4.1

The proof of Theorem 4.1 will follow as a consequence of the following result. For this, we require
the following notation. Given 0 < p ≤∞, s ∈N and a sequence c = (ci)∞
i=1, we let

σs(c)p = min{∥c −z∥p : z ∈ℓ2, |supp(z)| ≤s},

where supp(z) = {i ∈N : zi ̸= 0} for z = (zi)i∈N ∈RN.
Theorem F.1. For any 0 < p < 1 then term θm(b) defined in (4.2) satisfies

θm(b) ≳σm(b)2,
∀b ∈ℓ1(N), b ≥0, ∥b∥1 ≤1.

As noted, the proof of this theorem is based on [7, Thm. 4.4]. We recap the details as they will also be
needed in the proof of the next result. First, we recall some basic definitions. See [82] or [29, Ch. 10]
for more details. Let K be a subset of a normed space (X, ∥· ∥X ). Then its Gelfand m-width is

dm(K, X) = inf

sup
x∈K∩Lm ∥x∥X , Lm a subspace of X with codim(Lm) ≤m

.
(F.1)

An equivalent representation is

dm(K, X) = inf

(

sup
x∈K∩Ker(A)
∥x∥X , A : X →Rm linear

)

.

The Gelfand width is related to the following quantity:

Em
ada(K, X) = inf

sup
x∈K
∥x −∆(Γ(x))∥X , Γ : X →Rm adaptive, ∆: Rm →X

,
(F.2)

58


---Page Break---
where ∆is an arbitrary (potentially nonlinear) reconstruction map and Γ is an adaptive sampling
map. By this, we mean that

Γ(x) =





Γ1(x)
Γ2(x, Γ1(x))
...
Γm(x, Γ1(x), . . . , Γm−1(x))



,

where Γ1 : X →R is linear and, for i = 2, . . . , m, Γi : X × Ri−1 →R is linear in its first
component.

Proof of Theorem F.1. We proceed in a series of steps.

Step 1: Setup. Define the functions

ϕi(x) =
√

3xi,
x ∈D,
i = 1, 2, . . . .

Notice that these functions form an orthonormal system in L2
ϱ(D). (A.I) implies that these functions
form a Riesz system in L2
ς(D), with Riesz constants that are ≍1. Hence they have a (unique)
biorthogonal Riesz system {ψi}∞
i=1 ⊂L2
ς(D). Now define Φi = ϕi ◦ι and Ψi = ψi ◦ι. Let
G ∈L2
µ(X; R) be arbitrary. Then

∥G∥L2µ(X;R) ≥
⟨G, P∞
i=1⟨G, Ψi⟩L2µ(X;R)Ψi⟩L2µ(X;R)
∥P∞
i=1⟨G, Ψi⟩L2(X;R)Ψi∥L2µ(X;R)
=
P∞
i=1 |⟨G, Ψi⟩L2(X;R)|2

∥P∞
i=1⟨G, Ψi⟩L2(X;R)Ψi∥L2µ(X;R)
.

Consider the denominator. Using (A.II) and fact that the ψi form a Riesz system, we see that


∞
X

i=1
⟨G, Ψi⟩L2(X;R)Ψi


L2µ(X;R)
=



∞
X

i=1
⟨G, Ψi⟩L2(X;R)ψi


L2ς(D)
≲

v
u
u
t

∞
X

i=1
|⟨G, Ψi⟩L2(X;R)|2.

We deduce that

∥G∥L2(X;R) ≳

v
u
u
t

∞
X

i=1
|⟨G, Ψi⟩L2(X;R)|2,
∀G ∈L2
µ(X; R).
(F.3)

Now let b ≥0 with b ∈ℓ1(N) and I ⊂N with |I| = N. Using [7, Lem. 5.2] we see that the function

f = c
X

i∈I
ciyϕi ∈H(b),
(F.4)

for any y ∈Y, ∥y∥Y = 1 and c = (ci)i∈N ⊂RN with |c| ≤b (i.e., |ci| ≤bi, ∀i), where c > 0 is a
universal constant.

Step 2: Reduction to a discrete problem. Let L and R be arbitrary sampling and reconstruction maps
as in (4.2). Following [7, Lem. 5.3], let F = f ◦ι and observe that

F(X) = cy
X

i∈I
ciΦi(X)

and therefore
L(F) = yΓ(c),
where Γ : R|I| →Rm is given by

Γ(z) =





c P

i∈I ziΦi(X1)
...
c P

i∈I ziΦi(Xm)



,

due to (4.1). Notice that Γ is an adaptive sampling map of the form defined above. Now let
y∗∈B(Y∗) be such that |y∗(y)| = ∥y∥Y. Then, by (F.3),

|||F −R ◦L(F)|||2
L2µ(X;Y) ≥∥y∗(F −R ◦L(F))∥2
L2µ(X;R)

≳
X

i∈I
|⟨y∗(F −R ◦L(F)), Ψi⟩L2µ(X;R)|2.

59


---Page Break---
By biorthogonality, ⟨y∗(F), Ψi⟩L2µ(X;R) = c∥y∥Yci = c · ci, which implies that

|||F −R ◦L(F)|||L2µ(X;Y) ≳∥c −∆◦Γ(c)∥2,
(F.5)

where
∆: Rm →RN,
y 7→∆(y) =

⟨y∗(R(y · y)), Ψi⟩L2µ(X;R)/c


i∈I .

Therefore,

sup
F ∈H(b,ι)
|||F −R ◦L(F)|||L2µ(X;Y) ≳inf
Γ,∆
sup
c∈RN
supp(c)⊆I
|c|≤b

∥c −∆◦Γ(c)∥2,

where the infimum is taken over all adaptive sampling maps Γ and reconstruction maps ∆. Since L
and R were arbitrary, we get
θm(b) ≳Eada
m (B(b, I), ℓ2
N),

where B(b, I) = {z ∈R|I| : |zi| ≤bi, i ∈I} and ℓ2
N = (RN, ∥· ∥2).

Step 3: Derivation of the lower bounds. The next step is identical to the proof of Theorem 4.4 in [7].
This gives (F.1).

Proof of Theorem 4.1. We use Theorem F.1. For (i), we let b = (bi)∞
i=1 by defined by

bi = (2m)−1/p, i = 1, . . . , 2m,
bi = 0, i > 2m.

This sequence b ∈ℓp
M(N) with ∥b∥p,M = ∥b∥p = 1. Moreover, we have

σm(b)2 = 2−1/pm1/2−1/p.

For (ii) we let b
=
(bi)∞
i=1 be defined by bi
=
cp(i log2(i))−1/p,
where cp
=
 P∞
i=1 1/(i log2(i))
−1/p. This sequence b ∈ℓp
M(N) with ∥b∥p,M = ∥b∥p = 1. Moreover,

σm(b)2
2 ≥c2
p

2m
X

i=m+1
(i log2(i))−2/p ≥c2
p · m · (2m log2(2m))−2/p,

as required.

G
Proof of Theorem 4.2

Much as in the previous section, the proof of Theorem 4.2 is a consequence of the following result.
Theorem G.1. Suppose that the pushforward ς in (A.I) is a tensor-product of a univariate probability
measure. Then the term ˜θm(b) defined in (4.3) satisfies

˜θm(b) ≳σm(b)1/ log(m),
∀b ∈ℓ1(N), b ≥0, ∥b∥1 ≤1.

Proof of Theorem G.1. We proceed in a similar series of steps to those of the last proof.

Step 1: Setup. Let π : N →N be a bijection that gives a nonincreasing rearrangement of b, i.e.,
bπ(1) ≥bπ(2) ≥· · · . Now, let r ∈N be arbitrary and consider the index set

I = I1 ∪· · · ∪Ir,
Il = {π((l −1)(m + 1) + 1), . . . , π(l(m + 1))}.

Notice that |I| = r(m + 1). Define the matrix

A =
 
(ι(Xi))π(j)
m,r(m+1)
i,j=1

and notice that we may write

A = [A1
· · ·
Ar] ,
where Al =
 
(ι(Xi))π((l−1)(m+1)+j)
m,m+1
i,j=1
.

Let σ be the one-dimensional probability measure associated with ς. Then notice that each Al is a
random matrix whose entries are drawn i.i.d. from σ. Since σ is supported in [−1, 1], we deduce that

60


---Page Break---
the Al are independent subgaussian random matrices with the same distribution. Write γ for this
distribution. Let t1, . . . , tr > 0 and El,tl be the event

El,tl =
n
∃u ∈N(Al) : ∥u∥2 = 1, ∥u∥∞<
p

tl/(m + 1)
o
,
l = 1, . . . , r.
(G.1)

We will make a suitable choice of t1, . . . , tr later.

Step 2: Reduction to a discrete problem. Let

C =







c ∈R|I| : |ci| ≤bi, ∀i ∈I, c =





c1
...
cr



, cl ∈N(Al), l = 1, . . . , r







.

Notice that any c ∈C also satisfies c ∈N(A). Now let f = fc be as in (F.4) (we make the
dependence on c explicit now for convenience). Let x ∈D. Then

fc(x) =
√

3cy
X

i∈I
cixi.
(G.2)

We deduce that
(Fc(Xi))m
i=1 =
√

3cyAc = 0,
where Fc = fc ◦ι. This implies that

∥F −R ◦L(F)∥L∞
µ (X;Y) = ∥F −R({Xi, 0}m
i=1)∥L∞
µ (X;Y),

where, for convenience, we let L : F 7→{Xi, F(Xi)}m
i=1. Therefore,

sup
F ∈H(b,ι)
∥F −R ◦L(F)∥L∞
µ (X;Y) ≥sup
c∈C
∥Fc −R({Xi, 0}m
i=1)∥L∞
µ (X;Y).

Now observe that F0 = 0 and 0 ∈C. Hence

sup
F ∈H(b,ι)
∥F −R ◦L(F)∥L∞
µ (X;Y)

≥max

∥R({Xi, 0}m
i=1)∥L∞
µ (X;Y), sup
c∈C
∥Fc∥L∞
µ (X;Y) −∥R({Xi, 0}m
i=1)∥L∞
µ (X;Y)


.

For a > 0, the function x 7→max{x, a −x} is minimized at x = a/2 and takes value a/2 there. We
deduce that
sup
F ∈H(b,ι)
∥F −R ◦L(F)∥L∞
µ (X;Y) ≥1

2 sup
c∈C
∥Fc∥L∞
µ (X;Y).

Now, by (A.I),

∥Fc∥L∞
µ (X;Y) = ∥fc∥L∞
ς (D;Y) ≳∥fc∥L∞
ϱ (D;Y) =
√

3c
sup
∥x∥∞≤1



X

i∈I
cixi

 =
√

3c∥c∥1.

With this in hand, we conclude that

EX1,...,Xm∼µ
sup
F ∈H(b,ι)
∥F −R({Xi, F(Xi)}m
i=1)∥L∞
µ (X;Y) ≳EA1,...,Ar∼γ sup
c∈C
∥c∥1.

We now use the definition of C to write
EX1,...,Xm∼µ
sup
F ∈H(b,ι)
∥F −R({Xi, F(Xi)}m
i=1)∥L∞
µ (X;Y)

≳

r
X

l=1
EAl∼γ
sup
cl∈N(Al)
|(cl)i|≤bi,∀i∈Il

∥cl∥1.
(G.3)

Step 3: Bounding the expected error. Fix l = 1, . . . , r and suppose the event El,tl defined (G.1)
occurs. Let ul be the corresponding vector and define

cl = bπ(l(m+1))

∥ul∥∞
ul, l = 1, . . . , r.

61


---Page Break---
By construction, we have that cl ∈N(Al) and |(cl)i| ≤bi, ∀i ∈Il. We deduce that

sup
cl∈N(Al)
|(cl)i|≤bi,∀i∈Il

∥cl∥1 ≥bπ(l(m+1))∥ul∥1

∥ul∥∞
.

Now observe that
1 = ∥ul∥2
2 ≤∥ul∥1∥ul∥∞.
Therefore, we get that

sup
cl∈N(Al)
|(cl)i|≤bi,∀i∈Il

∥cl∥1 ≥bπ(l(m+1))

∥ul∥2
∞
≥bπ(l(m+1))(m + 1)

tl

whenever the event El,tl occurs. Using the law of total expectation, we deduce that

EAl∼γ
sup
cl∈N(Al)
|(cl)i|≤bi,∀i∈Il

∥cl∥1 ≥bπ(l(m+1))(m + 1)

tl
P(El,tl)

for any fixed tl > 0. We now appeal to [75, Thm. 1.4]. This shows that

P(Ec
l,tl) ≤c2m2 exp(−tl/c2),
∀tl ≥c1 log(m + 1),

where c1, c2 > 0 are universal constants. We may without loss of generality assume that c2 ≥c1 ≥1.
Now set tl = c2 log(2c2m2) ≥c1 log(m + 1). Hence

P(Ec
l,tl) ≤1/2.

We deduce that P(Et,tl) > 1/2 and therefore

EAl∼γ
sup
cl∈N(Al)
|(cl)i|≤bi,∀i∈Il

∥cl∥1 ≥bπ(l(m+1))(m + 1)

2c2 log(2c2m2) .

Now observe that

bπ(l(m+1))(m + 1) ≥bπ(l(m+1)) + · · · + bπ(l(m+1)+m)

Substituting this into (G.3), we deduce that

EX1,...,Xm∼µ
sup
F ∈H(b,ι)
∥F −R({Xi, F(Xi)}m
i=1)∥L∞
µ (X;Y) ≳
1
log(2m)

r(m+1)+m
X

i=m+1
bπ(i).

Since r was arbitrary, we may take the limit r →∞. We now use the fact that

σm(b)1 = bπ(m+1) + bπ(m+2) + · · · .

to obtain the result.

Proof of Theorem 4.2. Using Theorem G.1, statements (i) and (ii) are derived in exactly the same
way as in the proof of Theorem 4.1

62


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope?
Answer: [Yes]
Justification: We thoroughly discuss the main claims made in the abstract and introduction and the
necessary assumptions to show them. Our main theoretical contributions directly address these
claims. We also have several further remarks after these results to provide further context for our
work. We also provide detailed numerical experiments showing that DNN architectures compatible
with the setup for the theoretical results actually achieve the presented rates of approximation for
challenging operator learning problems posed in Banach spaces in terms of the number of samples
needed to achieve a given tolerance.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We discuss our main assumptions and their relative strengths in detail in a separate
section, §2.3. We also end the paper with a section discussing limitations. See §6.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?
Answer: [Yes]
Justification: We have a detailed discussion of the assumptions needed to show our theoretical
results in §2.2-2.3. We provide further discussion of the results themselves in §3-4 to place the
theoretical advancements in this work in the broader context of the operator learning literature.
We provide full proofs of our results in the supplemental material.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experi-
mental results of the paper to the extent that it affects the main claims and/or conclusions of the
paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We provide all of the necessary source code in the supplemental material. Further-
more, we provide a detailed discussion of the setup for the numerical experiments in §A-B of
the supplemental material. Given the code and description of the experiments, reproducing the
experiments is straightforward.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to
faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes]
Justification: We provide all the necessary software to reproduce our experiments along with
instructions for running the code to generate the results.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters,
how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justification: We specify all of the necessary hyperparameters to obtain our experimental results as
well as the optimizers used for training and the software used to generate our training and testing
data. All of these are open source, no proprietary data was used in this work.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: We plot not just the average error over all of the trials but also the (corrected) sample
standard deviation of the transformed sample with shaded plots to provide an estimate of the
variability in the runs.

63


---Page Break---
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the experi-
ments?
Answer: [Yes]
Justification: The experimental details are provided in the Appendix §A.2. All computational
resources are reported, including the type of workers, memory requirements, storage requirements,
and time of execution.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We have complied with the NeurIPS Code of Ethics in the preparation of this
manuscript. No human subject data was used to generate the results for our numerical experiments
and data-related concerns are not relevant to this work.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?
Answer: [NA]
Justification: This work is primarily foundational, and the examples considered do not directly
impact society.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of
data or models that have a high risk for misuse (e.g., pretrained language models, image generators,
or scraped datasets)?
Answer: [NA]
Justification: This work does not release any data or models that have a high risk for misuse. All
code is open source and no proprietary data was used in this work.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the
paper, properly credited and are the license and terms of use explicitly mentioned and properly
respected?
Answer: [Yes]
Justification: The code submitted in the supplemental material to both generate the data for training
and testing our models and generate the experimental results was written by the authors. No other
code or datasets were used in the production of this work.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: New code and data is included as a zip file as supplemental material.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justification: We do not use crowdsourcing for our experiments and no research was conducted
with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Sub-
jects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals
(or an equivalent approval/review based on the requirements of your country or institution) were
obtained?

64


---Page Break---
Answer: [NA]
Justification: We do not use crowdsourcing for our experiments and no research was conducted
with human subjects.

65


---Page Break---
