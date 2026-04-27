Particle Semi-Implicit Variational Inference

Jen Ning Lim
University of Warwick
Coventry, United Kingdom
Jen-Ning.Lim@warwick.ac.uk

Adam M. Johansen
University of Warwick
Coventry, United Kingdom
a.m.johansen@warwick.ac.uk

Abstract

Semi-implicit variational inference (SIVI) enriches the expressiveness of variational
families by utilizing a kernel and a mixing distribution to hierarchically define the
variational distribution. Existing SIVI methods parameterize the mixing distribution
using implicit distributions, leading to intractable variational densities. As a result,
directly maximizing the evidence lower bound (ELBO) is not possible, so they
resort to one of the following: optimizing bounds on the ELBO, employing costly
inner-loop Markov chain Monte Carlo runs, or solving minimax objectives. In this
paper, we propose a novel method for SIVI called Particle Variational Inference
(PVI) which employs empirical measures to approximate the optimal mixing
distributions characterized as the minimizer of a free energy functional. PVI arises
naturally as a particle approximation of a Euclidean–Wasserstein gradient flow and,
unlike prior works, it directly optimizes the ELBO whilst making no parametric
assumption about the mixing distribution. Our empirical results demonstrate that
PVI performs favourably compared to other SIVI methods across various tasks.
Moreover, we provide a theoretical analysis of the behaviour of the gradient flow
of a related free energy functional: establishing the existence and uniqueness of
solutions as well as propagation of chaos results.

1
Introduction

In Bayesian inference, a quantity of vital importance is the posterior p(x|y) = p(x, y)/
R
p(x, y)dx,
where p(x, y) is a probabilistic model; y denotes the observed data, and x the latent variable. An
ever-present issue in Bayesian inference is that the posterior is often intractable. This is because
the normalizing constant is available only in the form of an integral, and approximation methods
are required. One popular method is variational inference (VI) (Jordan, 1999; Wainwright et al.,
2008; Blei et al., 2017). The essence of VI is to approximate the posterior with a member from a
variational family Q where each element of Q is a distribution qθ (called “variational distribution”)
parameterized by θ. These parameters θ are obtained via minimizing a distance or discrepancy (or an
approximation of it) between the posterior p(·|y) and the variational distribution qθ.

Here, we focus on semi-implicit variational inference (SIVI) (Yin and Zhou, 2018). It enables
a rich variational family by utilizing variational distributions, which we refer to as semi-implicit
distributions (SIDs), defined as

qk,r(x) :=
Z
k(x|z)r(z) dz,
(1)

where k : Rdx × Rdz →R+ is a kernel satisfying
R
k(x|z)dx = 1; r ∈P(Rdz) is the mixing
distribution and P(Rdz) denotes the space of distributions with support Rdz, with its usual Borel
σ-field, with finite second moments: Er[∥z∥2] < ∞. Here, and throughout, we assume that the
distributions and kernels of interest admit densities. SIDs are very flexible (Yin and Zhou, 2018) and
can express complex properties, such as skewness, multimodality, and kurtosis. These properties

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
might be present in the posterior but typical variational families may fail to capture them. There are
various approaches to parameterizing these variational distributions: current techniques utilize neural
networks built on top of existing kernels (e.g., Gaussian kernels) to define more complex kernels
(Titsias and Ruiz, 2019), and/or utilize pushforward distributions (a.k.a., implicit distributions (Husz´ar,
2017)) (Yin and Zhou, 2018). On choosing a parameterization, an approximation to the posterior is
obtained by minimizing the exclusive Kullback-Leibler (KL) divergence. This optimization has the
same solution as minimizing the free energy (or the negative evidence lower bound) defined as

E(k, r) :=
Z
log qk,r(x)

p(x, y) qk,r(dx).
(2)

However, since the integral in qk,r is typically intractable, directly optimizing E is not feasible. As a
result, SIVI algorithms focus on designing tractable objectives by using upper bounds of E (Yin and
Zhou, 2018); expensive Markov Chain Monte Carlo (MCMC) chains to estimate the gradient of E
(Titsias and Ruiz, 2019); and optimizing different objectives such as score matching which results in
min-max objectives (Yu and Zhang, 2023).

In our work, we propose an alternative parameterization for SIDs: kernels are constructed as before
(with parameter space denoted by Θ) whereas the mixing distribution r is obtained by optimizing
over the whole space P(Rdz). We motivate the case for minimizing a regularized version of the free
energy E denoted by Eλ (see Eq. (4)); thus, SIVI can be posed as the following optimization problem:
arg min(θ,r)∈Θ×P(Rdz ) Eλ(θ, r). As a means to solving the SIVI problem, we construct a gradient
flow that minimizes Eλ where the space Θ × P(Rdz) is equipped with the Euclidean–Wasserstein
geometry (Jordan et al., 1998; Kuntz et al., 2023). Via discretization, we obtain a practical algorithm
for SIVI called Particle Variational Inference (PVI) that does not rely upon upper bounds of E,
MCMC chains, or solving minimax objectives.

Our main contributions are as follows: (1) we introduce a Euclidean–Wasserstein gradient flow
minimizing Eλ as means to perform SIVI; (2) we develop a practical algorithm, PVI, which arises as
a discretization of the gradient flow that allows for general mixing distributions; (3) we empirically
compare PVI compared with other SIVI approaches across toy and real-world experiments and find
that it compares favourably; (4) we study the behaviour of the gradient flow of a related free energy
functional to establish existence and uniqueness of solutions (Prop. 8) as well as propagation of chaos
results (Prop. 9).

The structure of this paper is as follows: in Section 2, we begin with a discussion of previous
approaches to parameterizing SIDs and their relationship with one another. Then, in Section 3, we
show how PVI is developed: beginning with designing a well-defined loss functional, the construction
of the gradient flow, and how to obtain a practical algorithm. In Section 4, we study properties of a
related gradient flow; and, in Section 5, we conclude with experiments to demonstrate the efficacy of
our proposal. For sake of brevity, we defer our discussion of related works to App. A.

2
On implicit mixing distributions in SID

This section outlines existing approaches to parameterizing SIDs with implicit distributions and how
these choices affect the resulting variational family. Before we begin, we shall summarize the key
assumptions of SIVI. The kernel k is assumed to be a reparametrized distribution in the sense of
Salimans and Knowles (2013); Kingma and Welling (2014); Ruiz et al. (2016). In other words, the
kernel k is defined by the pair (ϕ, pk) where ϕ : Rdz × Rdx →Rdx and pk ∈P(Rdx) such that
k(·|z) = ϕ(z, ·)#pk Furthermore, to ensure that it admits a tractable density, the map ϵ 7→ϕ(z, ϵ)
is assumed to be a diffeomorphism for all z ∈Rdz with its inverse map written as ϕ−1(z, ·). From
the change-of-variable formula, its density is given as k(·|z) = pk(ϕ−1(z, ·)) | det ∇xϕ−1(z, ·)|. We
sometimes write kϕ,pk to denote the underlying ϕ and pk explicitly. Furthermore, the kernel k is
assumed to be computable and differentiable with respect to both arguments.

Several approaches to the parameterization of SID have been explored in the literature. One can
define the variational family by choosing the kernel and mixing distribution from sets K and R
respectively, i.e, the variational family is Q(K, R) := {qk,r : k ∈K, r ∈R}. Yin and Zhou
(2018) focused on a fixed kernel k with r being a pushforward (or “implicit”) distribution, i.e.,
r ∈{g#pr : g ∈G} =: RG;pr where G is a subset of measurable mappings from the sample space of
pr to Rdz. Thus, the QYiZ-variational family is Q({k}, RG;pr). On the other hand, Titsias and Ruiz

2


---Page Break---
(2019) considered a fixed mixing distribution r with k belonging to some parameterized class K. The
typical example is one in which each kernel is defined by composing an existing kernel kϕ,pk with a
function f ∈F, the result is kf;ϕ,pk(·|z) := kϕ(f(·),·),pk(·|z) = ϕ(f(z), ·)#pk which clearly satisfies
the reparameterization assumption. We denoted this kernel class as KF;ϕ,pk := {kf;ϕ,pk : f ∈F}
and its respective QTR-variational family is Q(KG;ϕ,pk, {r}). In Yu and Zhang (2023), they combine
both parameterization for K and R, i.e., the QYuZ-variational family is Q(KF;ϕ,pk, RG;pr). We note
that this is how the variational family is presented in Yu and Zhang (2023, see Sec. 2) but the authors
used QTR-variational family in experiments, i.e., r was fixed. While QYuZ might seem like it defines a
larger variational family than the other approaches, under these common parameterization practices,
we show that they define the same variational family.

Proposition 1 (QYuZ = QYiZ = QTR). Given a QYuZ-variational family of the form QYuZ :=
Q(KF;ϕ,pk, RG;pr), then there is a QYiZ-variational family and QTR-variational family (i.e., QTR :=
Q(KF◦G;ϕ,pk, {pr}) and QYiZ := Q({kϕ,pk}, RF◦G;pr)) such that QYuZ = QYiZ = QTR.

The proof can be found in App. D. This proposition shows that QYuZ-parameterization defines the
“same” variational family as QYiZ and QTR when we parametrize R with push-forward distributions.
In practice, F and G are parameterized by neural networks hence QYuZ can be viewed as QYiZ or QTR
with a deeper neural networks F ◦G. This simplification is a direct result of using push-forward
distributions. Although this parametrization has shown promise e.g., Goodfellow et al. (2020)),
they have issues with expressivity particularly when distributions are disconnected (Salmona et al.,
2022). In our work, we follow in QYuZ-variational families, but, we avoid the use of push-forward
distributions. Instead, we propose to directly optimize over P(Rdz) and so, our variational family
does not simply reduce to QYiZ or QTR.

3
Particle Variational Inference

In this section, we present our proposed method for SIVI, called particle variational inference (PVI).
Similar to prior SIVI methods, the algorithm utilizes kernels (denoted by kθ) with parameters Θ
which satisfy the assumptions listed in Section 2. One example is kθ ∈KΘ;ϕ,pk where Θ is a function
space induced by a neural network. We slightly abuse the notation Θ to also indicate its corresponding
weight space Rdθ. The novelty of this algorithm is that, for the mixing distribution, we directly
optimize over the space P(Rdz) which loosens the requirement for the neural network in the kernel to
learn complex mappings. The result is a “simpler” optimization procedure and increases expressivity
over existing methods. Thus, the variational parameters of PVI are (θ, r) ∈Θ × P(Rdz) =: M with
its corresponding variational distribution defined as qθ,r :=
R
kθ(·|z)r(z)dz. PVI arises naturally as
a discretization of a gradient flow minimizing a suitably defined free energy on Θ × P(Rdz) endowed
with the Euclidean–Wasserstein geometry (Jordan et al., 1998; Ambrosio et al., 2005; Kuntz et al.,
2023). In Section 3.1, we begin by constructing a suitably defined free energy functional; then, in
Section 3.2, we formulate its gradient flow; finally, in Section 3.3, we construct PVI from its gradient
flow.

3.1
Free energy functional

As with other VI algorithms, we are interested in finding variational parameters that minimize
(θ, r) 7→KL(qθ,r, p(·|y)). This optimization problem can be cast equivalently as:

arg min
(θ,r)∈M
E(θ, r),
where E : M →R : (θ, r) 7→
Z
qθ,r(x) log qθ,r(x)

p(x, y) dx.
(3)

Before we can solve this problem, we must ensure that it is well-posed. In other words, it must admit
minimizers in M. In the following proposition, we outline various properties of E:

Proposition 2. Assume that the evidence is bounded log p(y) < ∞and k is bounded; then we have
that E is (i) lower bounded, (ii) lower semi-continuous (l.s.c.), and (iii) non-coercive.

The proof can be found in App. E.1. Prop. 2 tells us that even though E possesses many of the
properties one looks for in a meaningful minimization functional, it lacks coercivity (in the sense of
Dal Maso (2012, Definition 1.12)): a sufficient property to establish the existence of solutions. The
key to showing non-coercivity is that we can construct a kernel kθ(x|z) that does not depend on z.

3


---Page Break---
At first glance, this issue might seem contrived but we note that this problem is closely related to the
problem of posterior collapse (Lucas et al., 2019; Wang et al., 2021). To address non-coercivity, we
propose to utilize regularization and define the regularized free energy as:

Eλ(θ, r) := Eqθ,r(x)


log qθ,r(x)

p(x, y)


+ Rλ(θ, r),
(4)

where Rλ is a regularizer with parameters λ. In Prop. 3, we show that if Rλ is sufficiently regular,
then the Eλ enjoys better properties than its unregularized counterpart E.
Proposition 3. Under the assumptions of Prop. 2, if Rλ is coercive and l.s.c., then the regularized
free energy Eλ is (i) lower bounded, (ii) l.s.c., (iii) coercive. Hence it admits at least one minimizer in
M.

The proof can be found in App. E.2. From here forward, we shall focus on regularizers of the form
RE
λ : (θ, r) 7→λrKL(r, p0) + λθRθ(θ) where λ = {λr, λθ} are the regularization parameters and p0
is a predefined reference distribution. As long as Rθ is l.s.c., coercive and λθ, λr > 0, the resulting
regularizer RE
λ will also be l.s.c. and coercive. There are many possible choices for p0 and Rθ. For
p0, this regularizes solutions of the gradient flow toward it; as such, in settings where there is some
knowledge or preference about r at hand, we can set p0 to reflect that. In our experiments, we utilize
p0 = N(0, M) where M is a positive definite (p.d.) matrix. As for Rθ, there are also many choices.
In the context of neural networks, one natural choice is Tikhonov regularization 1

2∥· ∥2, resulting in
weight decay (Hanson and Pratt, 1988) for gradient descent (Loshchilov and Hutter, 2019) which
is a popular method for regularizing neural networks. In our experiments, we either use Tikhonov
regularization or its simple variant ∥θ∥2
M := ⟨θ, Mθ⟩.

3.2
Gradient flow

To solve the problem in Eq. (3), we construct a gradient flow that minimizes Eλ. To this end, we
endow the space M with a suitable notion of gradient ∇MEλ(θ, r) := (∇θEλ, ∇rEλ) where ∇θ
and ∇r denotes the Euclidean gradient and Wasserstein-2 gradient (Jordan et al., 1998), respectively.
The latter gradient is given by ∇rEλ(θ, r) := −∇z · (r∇zδrEλ[θ, r]), where ∇z· denotes the
standard divergence operator and δr denotes the first variation which is characterized in the following
proposition.

Proposition 4 (First Variation of Eλ and RE
λ). Assume that Ekθ(X|·)
log qθ,r(X)

p(X,y)
 < ∞for all (θ, r) ∈

M; then the first variation of Eλ is δrEλ = δrE+δrRλ where δrE[θ, r](z) = Ekθ(X|z)
h
log qθ,r(X)

p(X,y)
i
,

and δrRE
λ[θ, r] = λr log r/p0.

The proof can be found in App. E.3. Thus, the (Euclidean–Wasserstein) gradient flow of Eλ is

( ˙θt, ˙rt) = −∇MEλ(θt, rt) ⇐⇒
˙θt = −∇θEλ(θt, rt)
˙rt = −∇rEλ(θt, rt) = ∇z · (rt∇zδrEλ[θt, rt])
(5)

We now establish that the above gradient flow dynamic is contractive and that if a log-Sobolev
inequality Eq. (7) holds, one can also establish exponential convergence. The log-Sobolev inequality is
closely related to Polyak–Łojasiewicz inequality (or gradient dominance condition) and is commonly
assumed in gradient-based systems to obtain convergence (for instance, see Kim et al. (2024)). This
is formally stated in the following proposition and proved in App. E.3.
Proposition 5 (Contracting Gradient Dynamics). The free energy Eλ along the flow Eq. (5) is
non-increasing and satisfies

d
dtEλ(θt, rt) = −∥∇MEλ(θt, rt)∥2 ≤0,
(6)

where ∥∇MEλ(θ, r)∥2 := ∥∇θEλ(θ, r)∥2 + ∥∇zδrEλ[θ, r]∥2
r. Moreover, if a log-Sobolev Inequality
holds for a constant τ ∈R>0, i.e., for all (θ, r) ∈M, we have

Eλ(θ, r) −E∗
λ ≤τ∥∇MEλ(θ, r)∥2,
(7)

where E∗
λ := inf(θ,r)∈M Eλ(θ, r); then we have exponential convergence

Eλ(θt, rt) −E∗
λ ≤exp(−t/τ)(Eλ(θ0, r0) −E∗
λ).

4


---Page Break---
Typically direct simulation of the gradient flow Eq. (5) is intractable as the derivative of the first
variation of RE
λ involves ∇z log rt; instead, it is useful to identify the gradient flow with a McKean–
Vlasov SDE, for which they share the same Fokker–Planck equation. The key distinction is that the
SDE can be simulated without access to this quantity ∇z log rt. This SDE, which we term the PVI
flow, is given by

dθt = −∇θEλ(θt, rt) dt, dZt = b(θt, rt, Zt) dt +
p

2λr dWt, where rt = Law(Zt),
(8)

where the drift is b(θ, r, ·) := −∇zδrE[θ, r] + λr∇z log p0 (with the first variation given in Prop. 4)
and Wt is a dz-dimensional Wiener process. A connection between the Langevin diffusion, i.e.,
dZt = ∇z log p(Zt, y) dt+
√

2dWt, and PVI flow can be observed with the fixed kernel kθ(dx|z) =
δz(dx) and λr = 0, namely, they both satisfy the same Fokker–Planck equation.

3.3
A practical algorithm

Algorithm 1 Particle Variational Inference (PVI)

Input: initialization (θ0, {Z0}M
i=1); regularization parameters {λθ, λr}; step-sizes hθ and hr;
number of Monte Carlo samples L (for Eqs. (11) and (12)); and preconditioner Ψ = (Ψθ, Ψr).
for k = 1 to K do

rM
k−1 ←
1
M
PM
m=1 δZk−1,m
θk ←θk−1 −hθΨθ b∇θEλ(θk−1, rM
k−1)
▷See Eq. (11)
ˆbk ←Z 7→−b∇zδrE[θk, rM
k−1](Z) + λr∇z log p0(Z)
▷See Eq. (12)
for m = 1 to M do

Zk,m ←Zk−1,m + hrΨrˆb(Zk−1,m) + √λrhrΨrηk,m
▷ηk,m ∼N(0, Idz)
end for
end for
return (θK, {ZK,m}M
m=1)

To produce a practical algorithm, we are faced with several practical issues. The first issue we tackle
is the computation of gradients of expectations for which using standard automatic differentiation is
insufficient. The second problem is that these gradients are often ill-conditioned and have different
scales in each dimension. This is tackled using preconditioning resulting in adaptive stepsize. Finally,
to produce computationally feasible algorithms, we show how to discretize the PVI flow in both
space and time. PVI is summarised in Algorithm 1.

Computing the gradients. In the PVI flow, both the drift of the ODE and SDE include a gradient of an
expectation with respect to parameters that define the distribution that is being integrated. Specifically,
the terms that contain these gradients are ∇θEλ and ∇zδrEλ. Fortunately, these gradients can be
rewritten as an expectation (as described in Prop. 6) for which the parameters being differentiated
w.r.t. is only found in the integrand (see derivation in App. G.1).

Proposition 6. If ϕ and k are differentiable, then we have

∇θE(θ, r) =Epk(ϵ)r(z) [(∇θϕθ · [sθ,r −sp])(z, ϵ)] ,
(9)

∇zδrE[θ, r](z) =Epk(ϵ) [(∇zϕθ · [sθ,r −sp])(z, ϵ)] .
(10)

where ∇θϕ ∈Rdθ×dx denotes the Jacobian (∇θϕ)ij = ∂θiϕj (and similarly for ∇zϕ); scores are
sθ,r(z, ϵ) := ∇x log qθ,r(ϕθ(z, ϵ)) (and similarly sp(z, ϵ)) ; and · denotes the usual matrix-vector
multiplication in the sense of M · v : (z, ϵ) 7→M(z, ϵ)v(z, ϵ).

From Eqs. (9) and (10), we can produce Monte Carlo estimators for the gradients, i.e.,

b∇θE(θ, r) := 1

L

L
X

l=1
Ez∼r[(∇zϕθ · [sθ,r −sp])(z, ϵl)],
(11)

b∇zδrE[θ, r] := 1

L

L
X

l=1
(∇zϕθ · [sθ,r −sp])(·, ϵl),
(12)

5


---Page Break---
where {ϵl}L
l=1
i.i.d.
∼
pk. This is an instance of a path-wise Monte-Carlo gradient estimator; a
performant estimator that has been shown empirically to exhibit lower variance than other standard
estimators (Kingma and Welling, 2014; Roeder et al., 2017; Mohamed et al., 2020).

Adaptive Stepsizes. One of the complexities of training neural networks is that their gradient is often
poorly conditioned. As a result, for certain problems, the gradients computed from Eq. (9) and
Eq. (10) can often produce unstable algorithms without careful tuning of the step sizes. When this
occurs, we utilize preconditioners (Staib et al., 2019) to avoid this issue. Let Ψθ : Θ 7→Rdθ×dθ and
Ψr : Rdz 7→Rdz×dz be the precondition for components θ and r respectively, then the resulting
preconditioned gradient flow is given by

dθt = −Ψθ∇θEλ(θt, rt) dt, ∂trt = ∇z · (rtΨr∇zδrEλ[θt, rt]).
(13)

If Ψθ and Ψr are positive definite, then Eλ(θt, rt) remains non-increasing, i.e., Eq. (6) holds. As
before, this Fokker–Planck equation is satisfied by the following Mckean–Vlasov SDE:

dθt = −Ψθ(θt)∇θEλ(θt, rt) dt,
(14)

dZt = [Ψr(Zt)b(θt, rt, Zt) + ∇z · Ψr(Zt)] dt +
p

2λΨr(Zt)dWt.
(15)

where (∇z · Ψr)i = Pdz
j=1 ∂zj[(Ψr)ij] and rt = Law(Zt). The equivalence between Eq. (13) and
Eqs. (14) and (15) is shown in App. G.2. A simple example for the preconditioner allows the θt and
Zt to have different time scales; ultimately, this results in different step sizes. Another more complex
example of preconditioner Ψθ is the RMSProp (Tieleman and Hinton, 2012), and Ψr we utilize a
preconditioner inspired by RMSProp (see App. G.2). As with other related works (e.g., see Li et al.
(2016)), we found that the additional term ∇z · Ψr can be omitted in practice: it has little effect but
incurs a large computational cost.

Discretization in both space and time. To obtain an actionable algorithm, we need to discretize
the PVI flow in both space and time. For the space discretization, we propose to use a particle
approximation for rt, i.e., for a set of particles {Zt,m}M
m=1 with each satisfying Law(Zt,m) = rt, we
utilize the approximation rM
t
:=
1
M
PM
m=1 δZt,m which converges almost surely to rt in the weak
topology as M →∞by the strong law of large numbers and a countable determining class argument
(e.g., see Schmon et al. (2020, Theorem 1.1)). This approximation is key to making the intractable
tractable, e.g., Eq. (1) is approximated by qθ,rM
t
=
1
M
PM
m=1 kθ(x|Zt,m). One obtains a particle
approximation to the PVI flow from the following ODE–SDE:

dθM
t
= −∇θEλ(θM
t , rM
t ) dt, ∀m ∈[M] : dZM
t,m = b(θM
t , rM
t , ZM
t,m) dt +
p

2λr dWt,m,

where [M] := {1, . . . , M}. As for the time discretization, we employ Euler-Maruyama discretization
with step-size h which (using an appropriately defined preconditioner) can be decoupled into different
stepsizes for θt and Zt denoted by hθ and hr respectively.

4
Theoretical analysis

We are interested in the behaviour of the PVI flow (8). However, a key issue in its study is that the
drift in PVI flow might lack the necessary continuity properties to analyze using the existing theory.
In this section, we instead analyze the related gradient flow of the more regular functional

Eγ
λ(θ, r) := Eγ(θ, r) + Rλ(θ, r),
where Eγ(θ, r) = Eqθ,r(x)


log qθ,r(x) + γ

p(x, y)


(16)

for γ > 0. A similar modified flow was also explored in Crucinio et al. (2024) for similar reasons;
they found empirically that, at least when using a tamed Euler scheme, setting γ = 0 did not cause
problems in practice. Similarly, our experimental results for PVI found γ = 0 did not have issues. To
provide an additional measure of confidence in the reasonableness of this regularization and of the
use of this functional as a proxy for Eλ, we establish that the minima of Eγ
λ converge to those of Eλ in
the γ →0 limit.
Proposition 7 (Γ-convergence and convergence of minima). Under the same assumptions as Prop. 3,
we have that Eγ
λ Γ-converges to Eλ as γ →0 (in the sense of Def. B.1). Moreover, we have as an
immediate corollary that

inf
(θ,r)∈M Eλ(θ, r) = lim
γ→0
inf
(θ,r)∈M Eγ
λ(θ, r).

6


---Page Break---
The proof uses techniques from Γ-convergence theory (introduced by De Giorgi, see e.g. De Giorgi
and Franzoni (1975); see Dal Maso (2012) for a good modern introduction) and can be found in
App. F.1. The gradient flow of Eγ
λ, which we term γ-PVI, is given by

dθγ
t = −∇θEγ
λ(θγ
t , rγ
t ) dt, dZγ
t = bγ(θγ
t , rγ
t , Zγ
t ) dt +
p

2λr dWt,
(17)

where rγ
t = Law(Zγ
t ) and bγ(θ, r, ·) = −∇zδrEγ[θ, r] + λr∇z log p0. The derivation follows
similarly to that in Section 3.2, and is omitted for brevity. The use of γ > 0 is crucial for establishing
key regularity conditions in our analysis. We proceed by stating our assumptions.
Assumption 1 (Regularity of the target p, reference distribution p0, and Rθ). We assume that log p(y)
is bounded; and p, p0 and Rθ have Lipschitz gradients with constants Kp, Kp0, KRθ respectively:
there exists some Bp ∈R>0 such that log p(y) ≤Bp; and for any given y there exists a Kp ∈R>0
such that ∥∇x log p(x, y) −∇x log p(x′, y)∥≤Kp∥x −x′∥for all x, x′ ∈Rdx (similarly for p0 and
Rθ).
Assumption 2 (Regularity of k). We assume that the kernel k and its gradient is bounded
and has Kk-Lipschitz gradient; i.e., there exist constants Bk, Kk ∈R>0 such that |kθ(x|z)| +
∥∇(θ,x,z)kθ(x|z)∥≤Bk, and ∥∇xkθ(x|z) −∇xkθ′(x′|z′)∥≤Kk(∥(θ, x, z) −(θ′, x′, z′)∥) hold
for all θ, θ′ ∈Θ, z, z′ ∈Rdz, and x, x′ ∈Rdx,
Assumption 3 (Regularity of ϕ and pk.). We assume that ϕ has Lipschitz gradient and bounded
gradient. In order words, there is aϕ ∈R≥0, bϕ ∈R>0 such that ϕ satisfies ∥∇(θ,z)ϕ(z, ϵ) −
∇(θ,z)ϕθ′(z′, ϵ)∥F ≤(aϕ∥ϵ∥+ bϕ)(∥(θ, z) −(θ′, z′)∥) and ∥∇(θ,z)ϕθ(z, ϵ)∥F ≤(aϕ∥ϵ∥+ bϕ) for
all (θ, z), (θ′, z′) ∈Θ × Rdz, ϵ ∈Rdx, where ∥· ∥F denotes the Frobenius norm. We also assume
that pk has finite second moments.

Assumptions 2 and 3 are intimately connected; under some regularity conditions, one may imply
the other but we shall abstain from this digression for the sake of clarity. Assumptions 2 and 3 are
quite mild and hold for popular kernels such as kθ(x|z) = N(x; µθ(z), Σ) under some regularity
assumptions on µθ and Σ (which we show in App. C). These assumptions are key to establishing that
the drift in Eq. (17) is Lipschitz continuous (see Prop. 11 in the App.), from which, we establish the
existence and uniqueness of the solutions of Eq. (17).
Proposition 8 (Existence and Uniqueness). Under Assumptions 1 to 3, if γ
>
0 and
Epk(ϵ)∥sγ
θ,r(z, ϵ) −sp(z, ϵ)∥is bounded (with sγ
θ,r : (z, ϵ) 7→∇x log(qθ,r ◦ϕθ(z, ϵ) + γ)); then,
given (θ0, r0) ∈M, the solutions to Eq. (17) exists and is unique.

The proof can be found in App. F.2. Under the same assumptions, we can establish an asymptotic
propagation of chaos result that justifies the usage of a particle approximation in place of rγ
t in
Eq. (17).
Proposition 9 (Propagation of chaos). Under the same assumptions as Prop. 8; we have for any
fixed T:

lim
M→∞E sup
t∈[0,T ]

θγ
t −θγ,M
t

2
+ W2
2

(rγ
t )⊗M, qγ,M
t

= 0,

where (rγ
t )⊗M = QM
i=1(rγ
t ); qγ,M
t
= Law({Zγ,M
t,m }M
m=1); θγ,M
t
and Zγ,M
t,m are solutions to

dθγ,M
t
= −∇θEγ
λ(θγ,M
t
, rγ,M
t
) dt, where rγ,M
t
= 1

M

M
X

m=1
δZγ,M
t,m

∀m ∈[M] : dZγ,M
t,m = bγ(θγ,M
t
, rγ,M
t
, Zγ,M
t,m ) dt +
p

2λr dWt,m.

The proof can be found in App. F.3. Having established the existence and uniqueness of the γ-PVI
flow, as well as an asymptotic justification for using particles, we now provide a numerical evaluation
to demonstrate the efficacy of our proposal.

5
Experiments

In this section, we compare PVI against other semi-implicit VI methods. As described in the App. A,
these include unbiased semi-implicit variational inference (UVI) of Titsias and Ruiz (2019), semi-
implicit variational inference (SVI) of Yin and Zhou (2018), and the score matching approach (SM)

7


---Page Break---
of Yu and Zhang (2023). Through experiments, we show the benefits of optimizing the mixing
distribution; we compare the effectiveness of PVI against other SIVI methods on a density estimation
problem on toy examples; and, we compare against other SIVI methods on posterior estimation
tasks for (Bayesian) logistic regression and (Bayesian) neural network. The details for reproducing
experiments as well as computation information can be found in App. H. The code is available at
https://github.com/jenninglim/pvi.

5.1
Impact of the mixing distribution

Figure 1: Comparison of PVI and PVIZero on a bimodal mixture of Gaussians for various kernels.
The plot shows the density qθ,r from PVI and PVIZero as well as the KDE plot of r from PVI
described by 100 particles.

From Prop. 1, it can be said that ultimately current SIVI methods utilize (directly or indirectly) a fixed
mixing distribution whilst PVI does not. We are interested in establishing whether there is any benefit
to optimizing the mixing distribution. Intuitively, the mixing distribution can be utilized to express
complex properties, such as multimodality, which the neural network kernel kθ can then exploit. If the
mixing distribution is fixed, this means that the neural network must learn to express these complex
properties directly—which can be difficult (Salmona et al., 2022). This intuition turns out to hold,
but for the kernel to exploit an expressive mixing distribution, it must be designed well. To illustrate
this, consider the distributions 1

2N(µ, I) + 1

2N(−µ, I) for µ = {1, 2, 4} and following kernels: the
“Constant” kernel N(z, I2); “Push” kernel N(fθ(z), σ2
θI2); “Skip” kernel N(z + fθ(z), σ2
θI2); and
“LSkip” kernel N(Wz + fθ(z), σ2
θI2) where W ∈R2×2;. We compare the results from PVI and
PVIZero (PVI with hr = 0 to result in a fixed r ≈N(0, I2)) to emulate PVI with a fixed mixing
distribution. As µ gets larger, the complexity of the kernel (or the mixing distribution) must grow to
express this (e.g., see (Salmona et al., 2022, Corollary 2)).

Fig. 1 shows the resulting densities and the learnt mixing distribution of PVI and PVIZero for
different kernels and various µ. For the constant kernel, PVI can solve this problem by learning a
complex mixing distribution to express the multimodality. However, for the push kernel, it can be
seen that as µ gets larger PVI and PVIZero suffer from mode collapse which we suspect is due to the
mode-seeking behaviour of using reverse KL and why prior SIVI methods utilized annealing methods
(see Yu and Zhang (2023, Section 4.1)). As a remedy, we utilize a Skip kernel which can be seen
to improve both PVI and PVIZero. In particular, both PVI and PVIZero were able to successfully
express the bimodality in µ = 2; however, PVIZero falls short when µ = 4 while PVI can express
the multimodality by learning a bimodal mixing distribution. Since Skip requires dz = dx, we show
that LSkip (which removes the requirement) exhibits a similar behaviour to Skip.

5.2
Density estimation

We follow prior works (e.g., Yin and Zhou (2018)) and consider three toy examples whose densities
are shown in Fig. 2 (they are given explicitly in App. H.2). In this setting, we use the kernel
kθ(x|z) = N(x; z + fθ(z), σ2
θI) with dz = dx = 2 where fθ(z) is a neural network whose
architecture can be found in App. H.2. As a qualitative measure of performance, Fig. 2 shows
the resulting approximating distribution of PVI which can be seen to be a close match to the
desired distribution. To compare methods quantitatively, we report the (sliced) Wasserstein distance

8


---Page Break---
Figure 2: Contour plots of the densities qθ,r (in blue) against the true densities (in black) for various
toy density estimation problems. We also plot the absolute difference in the density of qθ,r and the
true density, i.e., |qθ,r −p|.

(computed by POT (Flamary et al., 2021)) and the rejection power of a state-of-the-art two-sample
kernel test (Biggs et al., 2023) between the approximating and true distribution in Table 1. The results
reported are the average and standard deviation (from ten independent trials of the respective SIVI
algorithms). In each trial, the rejection rate p is computed from 100 tests and the sliced Wasserstein
distance is computed from 10000 samples with 100 projections. If the variational approximation
matches the distribution, the rejection rate will be at the nominal level of 0.05. It can be seen that
PVI consistently performs better than SIVI across all problems. PVI can achieve a rejection rate
near nominal levels across all problems whilst other algorithms can achieve good performances
on one but not the other. The details regarding how the Wasserstein distance is calculated and the
hyperparameters used can be found in App. H.2.

Problem
PVI
UVI
SVI
SM

Banana
0.060.02/0.170.01
0.070.02/0.110.03
0.130.05/0.310.02
0.390.24/0.240.12
Multimodal
0.050.01/0.050.01
0.650.23/0.160.07
0.130.06/0.080.02
0.140.05/0.100.02
X-Shape
0.060.03/0.070.01
0.230.16/0.100.04
0.110.04/0.120.01
0.150.11/0.110.03

Table 1: This table shows the rejection rate p and average (sliced) Wasserstein distance w for toy
density estimation problems. It is written in the format p/w (lower is better) with the subscripts
showing the standard deviation estimated from 10 independent runs. We indicate in bold when the
rejection rate minus the standard deviation is lower than the nominal level 0.05, and for the algorithm
that achieves the lowest Wasserstein score.

5.3
Bayesian logistic regression

As with others (Yin and Zhou, 2018), we consider a Bayesian logistic regression problem
on the waveform dataset (Breiman and Stone, 1984).
The model is expressed as y | x, w ∼
Bernoulli(Sigmoid(⟨x, w⟩)) with prior x ∼N(0, 0.01−1 × I22) where (y, w) ∈{0, 1} × R21
is the response and covariates, and w := [1, w] is the covariates with appended one for the in-
tercept. The “ground truth” is composed of posterior samples generated from running Markov
chain Monte Carlo (MCMC) samples in Yin and Zhou (2018). We use the kernel kθ(x|z) =
N(Wz + fθ(z), exp( 1

2[Mθ + M ⊤
θ ])) where exp denotes the matrix exponential which ensures posi-
tive definiteness. In Fig. 3, we visually compare certain statistics of the distribution obtained from
MCMC and distribution obtained from SIVI methods. Fig. 3a shows the pair-wise marginal posterior
distributions for three weights x1, x2, x3 chosen at random from MCMC and SIVI approximations;
and in Fig. 3b we compare the correlation coefficients obtained from MCMC against SIVI methods. It
can be seen that PVI obtains an approximation close to MCMC with most other SIVI methods obtain-
ing similar performance levels (with the exception of SM). See App. H.3 for all the implementation
details.

5.4
Bayesian neural networks

Following prior works (e.g., Yu and Zhang (2023)), we compare our methods with other baselines
on sampling the posterior of the Bayesian neural network for regression problems on a range of
real-world datasets. We utilize the LSkip kernel kθ(x|z) = N(x; Wz + fθ(z), σ2
θ(z)Idx). In Table 2,

9


---Page Break---
(a)

(b)

Figure 3: Comparison between SIVI methods and MCMC on Bayesian logistic regression problem.
(a) shows the marginal and pairwise approximations of posterior of the weights x1, x2, x3, and (b)
shows the scatter plot of the correlation coefficient of MCMC (y-axis) vs PVI (x-axis).

we show the root mean squared error on the test set. It can be seen that PVI performs well, or at least
comparable, with other SIVI methods across all datasets. The details regarding the model and other
parameters can be found App. H.4.

Dataset
PVI
UVI
SVI
SM

Concrete (Yeh, 2007)
0.430.03
0.500.03
0.500.04
0.920.06
Protein (Rana, 2013)
0.870.05
0.920.04
0.920.04
1.020.03
Yacht (Gerritsma et al., 2013)
0.130.02
0.180.02
0.170.02
0.980.16
Table 2: Root mean square error (lower is better) for Bayesian neural networks on the test set for
various datasets. Here, we write the results in the form µσ where µ is the average RMS and σ is its
standard error computed over 10 independent trials. We indicate in bold the lowest score.

6
Conclusion, Limitations, and Future Work

In this work, we frame SIVI as a minimization problem of Eλ, and then, as a solution, we study
its gradient flow. Through discretization, we propose a novel algorithm called Particle Variational
Inference (PVI). Our experiments found that PVI can outperform current SIVI methods. At a marginal
increase in computation cost (see App. H) compared with prior methods, PVI can consistently perform
better (or at least comparably in the worst cases considered) which we attribute to not imposing
a particular form on the mixing distribution. This is a key advantage of PVI compared to prior
methods: by not relying upon push-forward mixing distributions and instead using particles, the
mixing distribution can express arbitrary distributions when the number of particles is sufficiently
large. Furthermore, it is not necessary to tune the family of mixing distributions to obtain good results
in particular problems. Theoretically, we study a related gradient flow of Eγ
λ and establish desirable
properties such as the existence and uniqueness of solutions and propagation of chaos results.

The main limitation of our work is that the theoretical results only apply to the case where γ > 0;
yet, our experiments were performed with γ = 0 as this is when Eλ corresponds to the (regularized)
evidence lower bound. In future work, one can address these limitations by reducing this gap.
Furthermore, we found that certain kernels were more amenable than others when exploiting an
expressive mixing distribution (e.g., the skip kernel). The question of designing these kernels for PVI
(or SIVI more generally) is important for future work.

10


---Page Break---
Acknowledgments

JNL gratefully acknowledges the funding of the Feuer International Scholarship in Artificial Intelli-
gence. AMJ acknowledges financial support from the United Kingdom Engineering and Physical
Sciences Research Council (EPSRC; grants EP/R034710/1 and EP/T004134/1) and by United King-
dom Research and Innovation (UKRI) via grant EP/Y014650/1, as part of the ERC Synergy project
OCEAN.

References

Luigi Ambrosio, Nicola Gigli, and Giuseppe Savar´e. Gradient flows: in metric spaces and in the
space of probability measures. Springer Science & Business Media, 2005.

Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum Mean Discrepancy gradient
flow. Advances in Neural Information Processing Systems, 32, 2019.

Felix Biggs, Antonin Schrab, and Arthur Gretton. MMD-Fuse: Learning and combining kernels for
two-sample testing without data splitting. In Advances in Neural Information Processing Systems,
volume 37, 2023.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational inference: A review for statisticians.
Journal of the American Statistical Association, 112(518):859–877, 2017.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal
Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao
Zhang. JAX: composable transformations of Python+NumPy programs. v. 0.3.13, 2018. URL
http://github.com/google/jax.

Andrea Braides. Gamma-convergence for Beginners, volume 22. Clarendon Press, 2002.

L. Breiman and C.J. Stone. Waveform Database Generator (Version 1). UCI Machine Learning
Repository, 1984. DOI: https://doi.org/10.24432/C5CS3C.

Rocco Caprio, Juan Kuntz, Samuel Power, and Adam M Johansen. Error bounds for particle gradient
descent, and extensions of the log-Sobolev and Talagrand inequalities. e-print 2403.02004, ArXiv,
2024.

Ren´e Carmona. Lectures on BSDEs, stochastic control, and stochastic differential games with
financial applications. SIAM, 2016.

Ziheng Cheng, Longlin Yu, Tianyu Xie, Shiyue Zhang, and Cheng Zhang. Kernel Semi-Implicit
Variational Inference. In International Conference on Machine Learning, volume abs/2405.18997,
2024.

Francesca R. Crucinio, Valentin De Bortoli, Arnaud Doucet, and Adam M. Johansen. Solving a
class of Fredholm integral equations of the first kind via Wasserstein gradient flows. Stochastic
Processes and their Applications, 173:104374, 2024. ISSN 0304-4149. URL https://doi.org/
10.1016/j.spa.2024.104374.

Gianni Dal Maso. An Introduction to Γ-convergence, volume 8. Springer Science & Business Media,
2012.

Ennio De Giorgi and Tullio Franzoni. Su un tipo di convergenza variazionale. Atti Accad. Naz. Lincei
Rend. Cl. Sci. Fis. Mat. Nat. (8), 58(6):842–850, 1975. ISSN 0392-7881.

R´emi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aur´elie Boisbunon, Stanislas
Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, L´eo Gautheron,
Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet,
Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and
Titouan Vayer. POT: Python Optimal Transport. Journal of Machine Learning Research, 22(78):
1–8, 2021. URL http://jmlr.org/papers/v22/20-451.html.

11


---Page Break---
Nicolas Fournier and Arnaud Guillin. On the rate of convergence in Wasserstein distance of the
empirical measure. Probability Theory and Related Fields, 162(3):707–738, 2015.

J. Gerritsma, R. Onnink, and A. Versluis. Yacht Hydrodynamics. UCI Machine Learning Repository,
2013. DOI: https://doi.org/10.24432/C5XG7R.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the
ACM, 63(11):139–144, 2020.

Alex Graves. Stochastic backpropagation through mixture density distributions. arXiv preprint
arXiv:1607.05690, 2016.

Stephen Hanson and Lorien Pratt. Comparing biases for minimal network construction with back-
propagation. Advances in Neural Information Processing Systems, 1, 1988.

Roger A Horn and Charles R Johnson. Matrix Analysis. Cambridge University Press, 2012.

Ferenc Husz´ar. Variational inference using implicit distributions. e-print 1702.08235, ArXiv, 2017.

Michael Irwin Jordan. Learning in Graphical Models. MIT press, 1999.

Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the Fokker–Planck
equation. SIAM Journal on Mathematical Analysis, 29(1):1–17, 1998.

Juno Kim, Kakei Yamamoto, Kazusato Oko, Zhuoran Yang, and Taiji Suzuki. Symmetric Mean-
field Langevin Dynamics for Distributional Minimax Problems. In Proceedings of The Twelfth
International Conference on Learning Representations, 2024. URL https://openreview.net/
forum?id=YItWKZci78.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In Yoshua Bengio
and Yann LeCun, editors, 2nd International Conference on Learning Representations, ICLR
2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. URL http:
//arxiv.org/abs/1312.6114.

Anna Korba, Pierre-Cyril Aubin-Frankowski, Szymon Majewski, and Pierre Ablin. Kernel Stein
discrepancy descent. In International Conference on Machine Learning, pages 5719–5730. PMLR,
2021.

Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M Blei. Automatic
differentiation variational inference. Journal of Machine Learning Research, 18(14):1–45, 2017.

Juan Kuntz, Jen Ning Lim, and Adam M. Johansen. Particle algorithms for maximum likelihood
training of latent variable models. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent,
editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics,
volume 206 of Proceedings of Machine Learning Research, pages 5134–5180. PMLR, 25–27 Apr
2023. URL https://proceedings.mlr.press/v206/kuntz23a.html.

Marc Lambert, Sinho Chewi, Francis Bach, Silv`ere Bonnabel, and Philippe Rigollet. Variational
inference via Wasserstein gradient flows. Advances in Neural Information Processing Systems, 35:
14434–14447, 2022.

Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin. Preconditioned stochastic
gradient Langevin dynamics for deep neural networks. In Proceedings of the AAAI conference on
artificial intelligence, volume 30, 2016.

Lingxiao Li, Qiang Liu, Anna Korba, Mikhail Yurochkin, and Justin Solomon. Sampling with
Mollified Interaction Energy Descent. In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.net/forum?id=zWy7dqOcel.

Jen Ning Lim, Juan Kuntz, Samuel Power, and Adam Michael Johansen. Momentum particle
maximum likelihood. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria
Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International
Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research,
pages 29816–29871. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/
lim24b.html.

12


---Page Break---
Ilya Loshchilov and Frank Hutter.
Decoupled weight decay regularization.
In International
Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=
Bkg6RiCqY7.

James Lucas, George Tucker, Roger Grosse, and Mohammad Norouzi. Understanding posterior
collapse in generative latent variable models, 2019. URL https://openreview.net/forum?
id=r1xaVLUYuE.

Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte Carlo Gradient
Estimation in Machine Learning. Journal of Machine Learning Research, 21(132):1–62, 2020.
URL http://jmlr.org/papers/v21/19-346.html.

Warren Morningstar, Sharad Vikram, Cusuh Ham, Andrew Gallagher, and Joshua Dillon. Automatic
differentiation variational inference with mixtures. In International Conference on Artificial
Intelligence and Statistics, pages 3250–3258. PMLR, 2021.

Prashant Rana. Physicochemical Properties of Protein Tertiary Structure. UCI Machine Learning
Repository, 2013. DOI: https://doi.org/10.24432/C5QW3H.

Geoffrey Roeder, Yuhuai Wu, and David K Duvenaud. Sticking the landing: Simple, lower-variance
gradient estimators for variational inference. Advances in Neural Information Processing Systems,
30, 2017.

Francisco R Ruiz, Titsias RC AUEB, David Blei, et al. The generalized reparameterization gradient.
Advances in Neural Information Processing Systems, 29, 2016.

Tim Salimans and David A. Knowles. Fixed-Form Variational Posterior Approximation through
Stochastic Linear Regression. Bayesian Analysis, 8(4):837 – 882, 2013. doi: 10.1214/13-BA858.
URL https://doi.org/10.1214/13-BA858.

Antoine Salmona, Valentin De Bortoli, Julie Delon, and Agnes Desolneux. Can push-forward
generative models fit multimodal distributions?
Advances in Neural Information Processing
Systems, 35:10766–10779, 2022.

Filippo Santambrogio. Optimal transport for applied mathematicians. Birk¨auser, NY, 55(58-63):94,
2015.

Sebastian M Schmon, George Deligiannidis, Arnaud Doucet, and Michael K Pitt. Large-sample
asymptotics of the pseudo-marginal method. Biometrika, 108(1):37–51, 07 2020. ISSN 0006-3444.
doi: 10.1093/biomet/asaa044. URL https://doi.org/10.1093/biomet/asaa044.

Albert N. Shiryaev. Probability. Number 95 in Graduate Texts in Mathematics. Springer, New York,
second edition, 1996.

Matthew Staib, Sashank Reddi, Satyen Kale, Sanjiv Kumar, and Suvrit Sra. Escaping saddle points
with adaptive gradient methods. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings
of Machine Learning Research, pages 5956–5965. PMLR, 09–15 Jun 2019. URL https://
proceedings.mlr.press/v97/staib19a.html.

Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop, coursera: Neural networks for machine
learning. University of Toronto, Technical Report, 6, 2012.

Michalis K. Titsias and Francisco Ruiz. Unbiased implicit variational inference. In Kamalika
Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International
Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning
Research, pages 167–176. PMLR, 16–18 Apr 2019. URL https://proceedings.mlr.press/
v89/titsias19a.html.

Martin J Wainwright, Michael I Jordan, et al. Graphical models, exponential families, and variational
inference. Foundations and Trends® in Machine Learning, 1(1–2):1–305, 2008.

Yixin Wang, David Blei, and John P Cunningham. Posterior collapse and latent variable non-
identifiability. Advances in Neural Information Processing Systems, 34:5443–5455, 2021.

13


---Page Break---
I-Cheng Yeh. Concrete Compressive Strength. UCI Machine Learning Repository, 2007. DOI:
https://doi.org/10.24432/C5PK67.

Mingzhang Yin and Mingyuan Zhou. Semi-implicit variational inference. In Jennifer Dy and
Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning,
volume 80 of Proceedings of Machine Learning Research, pages 5660–5669. PMLR, 10–15 Jul
2018. URL https://proceedings.mlr.press/v80/yin18b.html.

Longlin Yu and Cheng Zhang. Semi-implicit variational inference via score matching. In The Eleventh
International Conference on Learning Representations, 2023. URL https://openreview.net/
forum?id=sd90a2ytrt.

14


---Page Break---
A
Related work

In this section, we outline four areas of related work: semi-implicit variational inference; Euclidean-
Wasserstein gradient flows and Wasserstein-gradient flows in VI; mixture models in VI; and finally,
the link between SIVI and solving Fredholm equations of the first kind.

At the time of writing, there are three algorithms for SIVI proposed: SVI Yin and Zhou (2018), UVI
Titsias and Ruiz (2019), SM Yu and Zhang (2023). Concurrently, Cheng et al. (2024) extended the
SM variant by solving the inner minimax objective and simplified the optimization problem. Each
had their parameterization of SID (as discussed in Section 2), and their proposed optimization method.
SVI relies on optimizing a bound of the ELBO which is asymptotically tight. UVI, like our approach,
optimizes the ELBO by using gradients-based approaches. However, one of its terms is the score
∇x log qθ,r(x) which is intractable. The authors proposed using expensive MCMC chains to estimate
it; in contrast to PVI, this term is readily available to us. For SM, they propose to optimize the Fisher
divergence, however, to deal with the intractabilities the resulting objective is a minimax optimization
problem which is difficult to optimize compared to standard minimization problems.

PVI utilizes the Euclidean–Wasserstein geometry. This geometry and associated gradient flows are
initially explored in the context of (marginal) maximum likelihood estimation by Kuntz et al. (2023)
and their convergence properties are investigated by Caprio et al. (2024). In Lim et al. (2024), the
authors investigated accelerated gradient variants of the aforementioned gradient flow in Euclidean–
Wasserstein geometry. The Wasserstein geometry particularly for gradient flows on probability space
has received much attention with many works exploring different functionals (for examples, see Arbel
et al. (2019); Korba et al. (2021); Li et al. (2023)). In the context of variational inference Lambert
et al. (2022) analyzed VI as a Bures–Wasserstein Gradient flow on the space of Gaussian measures.

PVI is reminiscent of mixture distributions which is a consequence of the particle discretization. Mix-
ture models have been studied in prior works as variational distributions(Graves, 2016; Morningstar
et al., 2021). In Graves (2016), the authors extended the parameterization trick to mixture distribu-
tions; and Morningstar et al. (2021) proposed to utilize mixture models as variational distributions
in the framework of Kucukelbir et al. (2017). Although similar, the mixing distribution assists the
kernel in expressing complex properties of the true distribution at hand (see Section 5.1) which is an
interpretation that mixture distribution lacks.

There is an obvious similarity between SIVI and solving Fredholm equations of the first kind. There is
considerable literature on solving such problems; see Crucinio et al. (2024), which is closest in spirit to
the approach of the present paper, and references therein. In fact, writing p(·|y) =
R ˜k(·|z, θ)r(z)dz.
with ˜k(·|z, θ) ≡kθ(·|z) makes the connection more explicit: essentially, one seeks to solve a
nonstandard Fredholm equation, with the LHS known only up to a normalizing constant, constraining
the solution to be in P(Z) × {δθ : θ ∈Θ}. While Crucinio et al. (2024) develop and analyse a
simple Wasserstein gradient flow to address a regularised Fredholm equation, neither the method nor
analysis can be applied to the SIVI problem because of this non-trivial constraint.

B
Γ-convergence

The following is one of many essentially equivalent definitions of Γ-convergence (see Dal Maso
(2012); Braides (2002) for comprehensive summaries of Γ-convergence). We take as definition the
following (see Dal Maso (2012, Proposition 8.1), Braides (2002, Definition 1.5)):

Definition B.1 (Γ-convergence). Assume that M is a topological space that satisfies the first axiom
of countability. Then a sequence Fγ : M →R is said to Γ-converge to F if:

• (lim-inf inequality) for every sequence (θγ, rγ) ∈M converging to (θ, r) ∈M, we have

lim inf
γ→0 Fγ(θγ, rγ) ≥F(θ, r).

• (lim-sup inequality) for any (θ, r) ∈M, there exists a sequence (θγ, rγ) ∈M, known as a
recovery sequence, converging to (θ, r) which satisfies

lim sup
γ→0
Fγ(θγ, rγ) ≤F(θ, r).

15


---Page Break---
Γ-convergence corresponds, roughly speaking, to the convergence of the lower semicontinuous enve-
lope of a sequence of functionals and, under mild further regularity conditions such as equicoercivity,
is sufficient to ensure the convergence of the sets of minimisers of those functionals to the set of
minimisers of the limit functional.

C
On Assumptions 2 and 3

We shall show that the Gaussian kernel kθ(x|z) = N(x; µθ(z), Σ), i.e.,

kθ(x|z) = (2π)−dx/2det(Σ)−0.5 exp

−1

2(x −µθ(z))T Σ−1(x −µθ(z))

,

where µθ : Rdz 7→Rdx; and Σ ∈Rdx×dx and is positive definite. In this section, we show that
Assumptions 2 and 3 are implied by Assumptions 4 and 5.
Assumption 4. µθ is bounded and Σ is positive definite: there exists Bµ ∈R>0 such that the
following holds for all (θ, z) ∈Θ × Rdz:
∥∇(θ,z)µθ(z)∥F ≤Bµ,

and for any x ∈Rdx \ 0, xT Σx > 0.
Assumption 5. µθ is Lipschitz and has Lipschitz gradient, i.e., there exist constants Kµ ∈R>0 such
that for all (θ, z), (θ′, z′) ∈Θ × Rdz the following hold:
∥µθ(z) −µθ′(z′)∥≤Kµ∥(θ, z) −(θ′, z′)∥,

∥∇(θ,z)µθ(z) −∇(θ,z)µθ′(z′)∥F ≤Kµ∥(θ, z) −(θ′, z′)∥.

C.1
kθ satisfies Assumption 2

In this section, we show that kθ satisfies Assumption 2. We first show the boundedness property then
the Lipschitz property.

Boundedness.
First, we shall show that kθ is bounded.
Clearly, we have kθ(x|z)
∈

0, (2π)−dx/2det(Σ)−0.5
hence |kθ| is bounded as a consequence of Assumption 4. Now to show
that the gradient is bounded ||∇(θ,x,z)kθ(x|z)||, we have the following

∇xkθ(x|z) = −kθ(x|z)Σ−1(x −µθ(z)),

∇zkθ(x|z) = ∇zµθ(z)∇µN(x; µ, σ2Idx) |µθ(z),

∇θkθ(x|z) = ∇θµθ(z)∇µN(x; µ, σ2Idx)

µθ(z) .

Hence, we have ∥∇(x,µ,σ)N(x; µθ(z), σ2Idx)∥< ∞, from Assumption 4 and using the fact the
gradient of a Gaussian density of given covariance w.r.t. µ is uniformly bounded. Thus, we have
shown that kθ satisfies the boundedness property in Assumption 2.

Lipschitz. For kθ, one choice of coupling function and noise distribution is ϕθ(z, ϵ) = Σ
1
2 ϵ + µθ(z)
and pk = N(0, Idx) where Σ
1
2 be the unique symmetric and positive definite matrix with (Σ
1
2 )2 = Σ
(Horn and Johnson, 2012, Theorem 7.2.6); and the inverse map is ϕ−1
θ (z, x) = Σ−1

2 (x −µθ(z)).
Thus, from the change-of-variables formula, we have

∇xkθ(x|z) = ∇x[pk(ϕ−1
θ (z, x))det(∇xϕ−1
θ (z, x))]

= det(∇xϕ−1
θ (z, x))∇x[pk
 
ϕ−1
θ (z, x)

]

= det(Σ−1/2)Σ−1/2∇xpk(ϕ−1
θ (z, x))

= ˜Σ−1

2 ∇xpk(ϕ−1
θ (z, x))

where ˜Σ−1

2 := det(Σ−1/2)Σ−1/2. Thus, we have
∥∇xkθ(x|z) −∇xkθ′(x′|z′)∥

≤∥˜Σ−1

2 ∇xpk(ϕ−1
θ (z, x)) −˜Σ−1

2 ∇xpk(ϕ−1
θ′ (z′, x′))∥

≤∥˜Σ−1

2 ∥F ∥∇xpk(ϕ−1
θ (z, x)) −∇xpk(ϕ−1
θ′ (z′, x′))∥

≤C∥ϕ−1
θ (z, x) −ϕ−1
θ′ (z′, x′)∥,

16


---Page Break---
where C is a constant and we use the following facts: ∥˜Σ−1

2 ∥F ≤|det(Σ−1/2)|∥Σ−1/2∥F < ∞
following from the fact Σ−1/2 is positive definite; pk is a standard Gaussian density function with
Lipschitz gradients; and that the inverse map ϕ−1 is Lipschitz from Assumptions 4 and 5:

∥ϕ−1
θ (z, x) −ϕ−1
θ′ (z′, x′)∥≤∥Σ−1

2 ∥F ∥(x, µθ(z)) −(x′, µθ′(z′))∥≤C′∥(x, θ, z) −(x′, θ′, z′)∥.

Hence, we have shown that kθ satisfies the Lipschitz property of Assumption 2, and so Assumption 2
holds for kθ.

C.2
kθ satisfies Assumption 3

One can compute the gradient as

∇(θ,z)ϕθ(z, ϵ) = ∇(θ,z)µθ(z),

and hence ∥∇(θ,z)ϕθ(z, ϵ)∥F is bounded from Assumption 4. The Lipschitz gradient property is
immediate from Assumption 5.

pk has finite second moments since it is a Gaussian.

D
Proofs in Section 2

Proof of Prop. 1. We start by showing QYuZ = QTR. To this end, we begin by showing the inclusion
QYuZ ⊆QTR, i.e., Q(KF;ϕ,pk, RG,pr) ⊆Q(KF◦G;ϕ,pk, {pr}). Let q ∈Q(KF;ϕ,pk, RG,pr), then
there is some f ∈F and g ∈G such that q = qkf;ϕ,pk ,g#pr. From straight-forward computation, we
have

qkf;ϕ,pk ,g#pr = Ez∼g#pr[kf;ϕ,pk(·|z)]
(a)
= Ez∼pr[kf;ϕ,pk(·|g(z))] ∈Q(KF◦G;ϕ,pk, {pr}),

where (a) follows the law of the unconscious statistician, and the last element-of follows from the
fact that kf;ϕ,pk(·|g(ϵ))) = ϕ(f ◦g(ϵ), ·)#pk ∈KF◦G;ϕ,pk. We can follow the argument above in
reverse to obtain the reverse inclusion. Hence, we have obtained as desired.

That QYuZ = QYiZ, follows in a similar manner, which we shall outline for completeness: let
q ∈Q(KF;ϕ,pk, RG,pr), then

q = qkf;ϕ,pk ,g#pr = Ez∼g#pr[kf;ϕ,pk(·|z)] = Ez∼f◦g#pr[kϕ,pk(·|z)] ∈QYiZ.

One can conclude by applying the same logic in the reverse direction.

E
Proofs in Section 3

E.1
Proof of Prop. 2

Proof of Prop. 2. (E is lower bounded). Clearly, we have

E(θ, r) = KL(qθ,r, p(·|y)) −log p(y) ≥−log p(y),

Hence, we have E(θ, r) ∈[−log p(y), ∞) which is lower bounded by our assumption.

(E is lower semi-continuous). Let (θn, rn)n∈N be such that limn→∞rn = r and limn→∞θn = θ.

We can split the domain of integration, and write E equivalently as

E(θ, r) =
Z
1[1,∞)

p(x, y)

qθ,r(x)


log
qθ,r(x)

p(x, y)



|
{z
}
≤0

qθ,r(x) dx
(18)

+
Z
1[0,1)

p(x, y)

qθ,r(x)


log
qθ,r(x)

p(x, y)



|
{z
}
≥0

qθ,r(x) dx
(19)

We shall focus on the RHS of (18).

17


---Page Break---
Note that we have the following bound
−1[1,∞)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x)


≤max

0, log
 p(x, y)

qθn,rn(x)


qθn,rn(x)

≤max{0, C},

where C is some constant. The last inequality follows from the fact that the evidence is bounded
from above and the kernel is bounded. We can apply Reverse Fatou’s Lemma to obtain

lim sup
n→∞−
Z
1[1,∞)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x) dx

≤
Z
lim sup
n→∞


−1[1,∞)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x)

dx.

Since we have the following relationships

lim sup
n→∞1[1,∞)

 p(x, y)

qθn,rn(x)


≤1[1,∞)

p(x, y)

qθ,r(x)


,

lim
n→∞−log
 p(x, y)

qθn,rn(x)


= −log
p(x, y)

qθ,r(x)


,

lim
n→∞qθn,rn = qθ,r pointwise,

where the first line is from u.s.c. of 1[1,∞); the second line from the continuity of log; the final line
follows from the bounded kernel k assumption and dominated convergence theorem.

Thus, we have that

lim sup
n→∞−
Z
1[1,∞)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x) dx

≤−
Z
1[1,∞)

p(x, y)

qθ,r(x)


log
qθ,r(x)

p(x, y)


qθ,r(x) dx,

Using the fact that lim supn→∞−xn = −lim infn→∞xn, we have shown that

−lim inf
n→∞

Z
1[1,∞)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x) dx

≤−
Z
1[1,∞)

p(x, y)

qθ,r(x)


log
qθ,r(x)

p(x, y)


qθ,r(x) dx.
(20)

Similarly, for the RHS of (19), using Fatou’s Lemma (with varying measure and the set-wise
convergence of qθn,rn) and using the l.s.c. of 1[0,1), we obtain that

lim inf
n→∞

Z
1[0,1)

 p(x, y)

qθn,rn(x)


log
qθn,rn(x)

p(x, y)


qθn,rn(x) dx

≥
Z
1[0,1)

p(x, y)

qθ,r(x)


log
qθ,r(x)

p(x, y)


qθ,r(x) dx.
(21)

Hence, combining the bounds (20) and (21), we have that shown that

lim inf
n→∞E(θn, rn) = lim inf
n→∞

Z
log
qθn,rn(x)

p(x, y)


qθn,rn(x) dx

≥
Z
log
qθ,r(x)

p(x, y)


qθ,r(x) dx ≥E(θ, r).

In other words, E is lower semi-continuous.

(Non-Coercivity) To show non-coercivity, we will show that there exists some level set {(θ, r) :
E(θ, r) ≤β} that is not compact. We do this by finding a sequence contained in the level set that
does not contain a (weakly) converging subsequence.

18


---Page Break---
Consider the sequence Π := (θn, rn)n∈N where θn = θ0; ∥θ0∥< ∞; rn = δn; kθ(x|z) =
N(x; θ, Idx); and p(x|y) = N(x; 0, Idx).
Clearly, we have qθ,r(x) = N(x; θ, Idx) and so
KL(qθ,r, p(·|y)) = 1

2∥θ∥2. Hence, there is a β < ∞such that

E(θn, rn) = KL(qθn,rn, p(·|y)) −log p(y) ≤1

2∥θ0∥2 −log p(y) ≤β.

Thus, we have shown that Π ⊂{(θ, r) : E(θ, r) ≤β}. However, since the support of the elements of
{rn ∈P(Rdz)}n eventually lies outside a ball of radius R for any R < ∞and hence of any compact
set, Π is not tight. Hence, Prokhorov’s theorem (Shiryaev, 1996, p. 318) tells us that, as Π is not
tight, it is not relatively compact. We conclude that, as the level set is not relatively compact, the
functional is not-coercive.

E.2
Proof of Prop. 3

Proof of Prop. 3. (Coercivity) Consider the level set {(θ, r) : Eλ(θ, r) ≤β}, which is contained in a
relatively compact set. To see this, first note that
{(θ, r) : Eλ(θ, r) ≤β} ⊆{(θ, r) : −log p(y) + Rλ(θ, r) ≤β}
⊆{(θ, r) : Rλ(θ, r) ≤β + log p(y)}
By coercivity of Rλ, i.e., the above level set is relatively compact hence Eλ is coercive.

(Lower semi-continuity) Lower semi-continuity (l.s.c.) follows immediately from the l.s.c. of E and
Rλ.

(Existence of a minimizer) The existence of a minimizer follows from Dal Maso (2012, Theorem
1.15) utilizing coercivity and l.s.c. of Eλ.

E.3
Proof of Prop. 4

Recall from Santambrogio (2015, Definition 7.12),
Definition E.1 (First Variation). If p is regular for F, the first variation of F : P(Rdz) →R, if it
exists, is the element that satisfies

lim
ϵ→0
F(p + ϵχ) −F(p)

ϵ
=
Z
δrF[r](z)χ(dz),

for any perturbation χ = ˜p −p with ˜p ∈P(Rdz) ∩L∞
c (Rdz) (see Santambrogio (2015, Notation)).

One can decompose the first variation of Eγ
λ as:

δrEγ
λ[θ, r] = δrEγ[θ, r] + δrRE
λ[θ, r].

where Eγ : (θ, r) 7→
R
log

qθ,r(x)+γ

p(x,y)

qθ,r(dx). Since δrRE
λ[θ, r] = λrδrKL(r|p0), its first varia-
tion follows immediately from standard calculations (Ambrosio et al., 2005; Santambrogio, 2015).
As for δrEγ, we have the following proposition:
Proposition 10 (First Variation of Eγ). Assume that for all (θ, r, z) ∈M × Rdz,

Ekθ(X|z)

log
qθ,r(X) + γ

p(X, y)

 < ∞,

then we obtain

δrEγ[θ, r](z) = Ekθ(X|z)


log
qθ,r(X) + γ

p(X, y)


+
qθ,r(X)
qθ,r(X) + γ


.

Proof. Since qθ,r+ϵχ =
R
kθ(·|z)(r + ϵχ)(z) dz = qθ,r + ϵqθ,χ, we have

Eγ(θ, r + ϵχ) =
Z

X
qθ,r+ϵχ(x) log
qθ,r+ϵχ(x) + γ

p(y, x)


dx

=
Z

X
[qθ,r + ϵqθ,χ](x) log([qθ,r + ϵqθ,χ](x) + γ) dx

−
Z

X
[qθ,r + ϵqθ,χ](x) log p(y, x) dx.

19


---Page Break---
Applying Taylor’s expansion, we obtain (x + ϵy) log(x + ϵy + γ)
=
x log(x + γ) +

ϵy

log(x + γ) +
x
x+γ

+ o(ϵ), we obtain

Eγ(θ, r + ϵχ) =
Z

X
qθ,r(x) log qθ,r(x) + γ

p(y, x)
dx

+ ϵ
Z

X
qθ,χ(x)

log
qθ,r(x) + γ

p(y, x)


+
qθ,r(x)
qθ,r(x) + γ


dx + o(ϵ).

Hence, we obtain

lim
ϵ→
Eγ(θ, r + ϵχ) −Eγ(θ, r)

ϵ
=
Z

X
qθ,χ(x)

log qθ,r(x) + γ

p(y, x)
+
qθ,r(x)
qθ,r(x) + γ


dx

=
Z

X

Z

Z
kθ(x|z)χ(dz)
 
log
qθ,r(x) + γ

p(y, x)


+
qθ,r(x)
qθ,r(x) + γ


dx

(a)
=
Z

Z

Z

X
kθ(x|z)

log
qθ,r(x) + γ

p(y, x)


+
qθ,r(x)
qθ,r(x) + γ


dx

χ(z) dz.

One can then identify the desired result. In (a), we appeal to Fubini’s theorem for the interchange of
integrals whose conditions
Z

Z

Z

X

kθ(x|z)

log
qθ,r(x) + γ

p(y, x)


+
qθ,r(x)
qθ,r(x) + γ


χ(z)
 dxdz < ∞,
(22)

are satisfied by our assumptions. This can be seen from

LHS Eq. (22) ≤
Z

Z
Ekθ(X|z)

log
qθ,r(X) + γ

p(X, y)


+
qθ,r(X)
qθ,r(X) + γ

 |χ(z)|dz ≤0,

where we use our assumption and the fact that χ is absolutely integrable.

Proof of Prop. 5. The result can be obtained from direct computation. We begin
d
dtEλ(θt, rt) =
D
∇θEλ(θt, rt), ˙θt
E
+
Z
δrEλ[θt, rt] ∂trt dz

The second term can be simplified
Z
δrEλ[θt, rt] ∂trt dz =
Z
δrEλ[θt, rt](z)∇z · (rt(z)∇δrEλ[θt, rt](z)) dz

= −
Z
rt∥∇zδrEλ[θt, rt](z)∥2 dz

where the last inequality follows from integration by parts. Hence, the claim holds. If the log-Sobolev
inequality holds, then we have
d
dt [Eλ(θt, rt) −E∗
λ] = −∥∇MEλ[θt, rt]∥≤−1

τ [Eλ(θt, rt) −E∗
λ] .

From Gr¨onwalls inequality, we obtain the desired result.

F
Proofs in Section 4

F.1
Proof of Prop. 7

Proof. We first begin by proving Γ-convergence directly via its definition, i.e., demonstrating that
the liminf inequality holds and establishing the existence of a recovery sequence. The latter follows
from pointwise convergence:
lim
γ→0 Eγ
λ(θ, r) = Eλ(θ, r),

upon taking (θγ, rγ) = (θ, r) for all γ.

The liminf inequality can be seen to follow similarly from the l.s.c. argument in App. E.1.

To arrive at the convergence of minima, we invoke Dal Maso (2012, Theorem 7.8) by using the
fact that Eγ
λ is equi-coercive in the sense of Dal Maso (2012, Definition 7.6). To see that Eγ
λ is
equi-coercive, note that we have Eγ
λ ≥Eλ and Eλ is l.s.c. (from Prop. 3), then applying Dal Maso
(2012, Proposition 7.7).

20


---Page Break---
F.2
Proof of Prop. 8

Proof of Prop. 8. We can equivalently write the γ-PVI flow in Eq. (17) as follows

d(θt, Zt) = ˜bγ(θt, Law(Zt), Zt) dt + σ dWt,
(23)

where σ =
0
0
0
√2λrIdz


, and

˜bγ : Rdθ × P(Z) × Rdz →Rdθ+dz : (θ, r, Z) 7→

−∇θEγ
λ(θ, r)
bγ(θ, r, Z)


.

In App. F.4, we show that under our assumptions the drift ˜bγ is Lipschitz. And under Lipschitz
regularity conditions, the proof follows similarly to Lim et al. (2024) which we shall outline for
completeness.

We begin endowing the space Θ × P(Rdz) with the metric

d((θ, r), (θ′, r′)) =
q

∥θ −θ′∥2 + W2
2(q, q′).

Let Υ ∈C([0, T], Θ × P(Rdz)) and denote Υt = (ϑΥ
t , νΥ
t ) for it’s respective components. Consider
the process that substitutes Υ into (23), in place of the Law(Zt) and θt,

d(θΥ
t , ZΥ
t ) = ˜bγ(ϑΥ
t , νΥ
t , ZΥ
t ) dt + σ dWt.

whose existence and uniqueness of strong solutions are given by Carmona (2016)[Thereom 1.2].

Define the operator

FT : C([0, T], Θ × P(Rdz)) →C([0, T], Θ × P(Rdz)) : Υ →(t 7→(θΥ
t , Law(ZΥ
t )).

Let (θt, Zt) denote a process that is a solution to (23) then the function t 7→(θt, Law(Zt)) is a fixed
point of the operator FT . The converse also holds. Thus, it is sufficient to establish the existence and
uniqueness of the fixed point of the operator FT . For Υ = (ϑ, ν) and Υ′ = (ϑ′, ν′)

∥θΥ
t −θΥ′
t ∥2 + E[∥ZΥ
t −ZΥ′
t ∥]2 = E


Z t

0
˜bγ(ϑs, νs, ZΥ
s ) −˜bγ(ϑ′
s, ν′
s, ZΥ′
s ) ds


2

≤tC
Z t

0

h
E∥ZΥ
s −ZΥ′
s ∥2 + ∥ϑs −ϑ′
s∥2 + W2
1(νs, ν′
s)
i
ds

≤C(t)
Z t

0
[W2
2(νs, ν′
s) + ∥ϑs −ϑ′
s∥2] ds,

where we apply Jensen’s inequality; Cr-inequality; Lipschitz drift of ˜bγ; and Gr¨onwall’s inequality.
The constant C := 3K2
˜b and C(t) := tC exp
  1

2t2C

. Thus, we have

d2(FT (Υ)t, FT (Υ′)t) ≤C(t)
Z t

0
d2(Υs, Υ′
s) ds.

Then, for F k
T denoting k successive composition of FT , one can inductively show that it satisfies

d2(F k
T (Υ)t, F k
T (Υ′)t) ≤(tC(t))k

k!
sup
s∈[0,T ]
d2(Υs, Υ′
s).

Taking the supremum, we have

sup
s∈[0,T ]
d2(F k
T (Υ)s, F k
T (Υ′)s) ≤(TC(T))k

k!
sup
s∈[0,T ]
d2(Υs, Υ′
s).

Thus, for a large enough k, we have shown that F k
T is a contraction and from Banach Fixed Point
Theorem and the completeness of the space (C([0, T], Θ × P(Rdz)), sup d), we have existence and
uniqueness.

21


---Page Break---
F.3
Proof of Prop. 9

Recall, the process defined in Prop. 9:

dθγ,M
t
= −∇θEγ
λ(θγ,M
t
, rγ,M
t
) dt, where rγ,M
t
= 1

M

M
X

m=1
δZγ,M
t,m

∀m ∈[M] : dZγ,M
t,m = bγ(θγ,M
t
, rγ,M
t
, Zγ,M
t,m ) dt +
p

2λr dWt,m.

and γ-PVI (defined in Eq. (17)) augmented with extra particles (in the sense that there are M
independent copies of the Z-process) to facilitate a synchronous coupling argument

dθγ
t = −∇θEγ
λ(θγ
t , Law(Zγ
t,1)) dt,

∀m ∈[M] : dZγ
t,m = bγ(θγ
t , Law(Zγ
t,1), Zγ
t,m) dt +
p

2λr dWt,m.

Proof of Prop. 9. This is equivalent to proving that

E sup
t∈[0,T ]
∥θγ
t −θγ,M
t
∥2

|
{z
}
(a)

+ E sup
t∈[0,T ]

(
1
M

M
X

m=1
∥Zγ
t,m −Zγ,M
t,m ∥2
)

|
{z
}
(b)

= o(1).
(24)

We shall treat the two terms individually. We begin with (a) in (24), where Jensen’s inequality gives:

(a) in (24) = E sup
t∈[0,T ]



Z t

0


∇θEγ
λ(θγ,M
s
, rγ,M
s
) −∇θEγ
λ(θγ
s , rγ
s )

ds


2

≤TE
Z T

0

∇θEγ
λ(θγ,M
t
, rγ,M
s
) −∇θEγ
λ(θγ
s , rγ
s )

2
dt

≤Cθ

Z T

0
E∥θγ
s −θγ,M
s
∥2 + EW2
2(rγ,M
s
, rγ
s ) dt.
(25)

where Cθ := 2TK2
Eγ
λ, we apply Cauchy–Schwarz; and the Cr inequality with the Lipschitz continuity
of ∇θEγ
λ from Prop. 12. Using the Cr inequality again, together with the triangle inequality:

EW2
2(rγ,M
s
, rγ
s ) ≤2EW2
2 (rγ
s , ˆrγ
s ) + 2EW2
2(rγ,M
s
, ˆrγ
s )

≤o(1) + 2

M

M
X

m=1
E∥Zγ
s,m −Zγ,M
s,m ∥2,
(26)

where ˆrγ
s =
1
M
PM
m=1 δZγ
s,m and we use Fournier and Guillin (2015). Note that we also have

∥θγ
s −θγ,M
s
∥2 ≤
sup
s′∈[0,T ]
∥θγ
s′ −θγ,M
s′
∥2,
(27)

1
M

M
X

m=1
∥Zγ
s,m −Zγ,M
s,m ∥2 ≤
sup
s′∈[0,T ]

1
M

M
X

m=1
∥Zγ
s′,m −Zγ,M
s′,m∥2.
(28)

Applying Eq. (26) in Eq. (25) then Eqs. (27) and (28), we obtain

(a) ≤2Cθ

Z T

0
E sup
s∈[0,T ]
∥θγ
s −θγ,M
s
∥2 + E sup
s∈[0,T ]

1
M

M
X

m=1
∥Zγ
s,m −Zγ,M
s,m ∥2 ds + o(1).
(29)

Similarly, for (b) in (24), we have

(b) = E sup
t∈[0,T ]

1
M

M
X

m=1



Z t

0
bγ(θγ,M
s
, rγ,M
s
, Zγ,M
s,m ) −bγ(θγ
s , Law(Zγ
s,1), Zγ
s,m) ds


2

≤CzE
Z T

0
∥θγ,M
s
−θγ
s ∥2 + W2
2(rγ,M
s
, Law(Zγ
s,1)) + 1

M

M
X

m=1
∥Zγ
s,m −Zγ,M
s,m ∥2 ds,

22


---Page Break---
where Cz := 3K2
bγand, as before, we apply Cauchy–Schwarz, Lipschitz and Cr inequalities. Then
from Eqs. (26) to (28), we obtain

(b) ≤CE
Z T

0
sup
s∈[0,T ]
∥θγ,M
s
−θγ
s ∥2 + sup
s∈[0,T ]

1
M

M
X

m=1
∥Zγ
s,m −Zγ,M
s,m ∥2 + o(1)ds.
(30)

Combining Eqs. (29) and (30) and applying Gr¨onwall’s inequality, we obtain

E sup
t∈[0,T ]
∥θγ
t −θγ,M
t
∥2 + E sup
t∈[0,T ]

(
1
M

M
X

m=1
∥Zγ
t,m −Zγ,M
t,m ∥2
)

= o(1).

Taking the limit, we have the desired result.

F.4
The drift in Eq. (17) is Lipschitz

In this section, we show that the drift in the γ-PVI flow in Eq. (17) is Lipschitz.

Proposition 11. Under the same assumptions as Prop. 8; the drift ˜b(A, r) is Lipschitz, i.e., there
exists a constant K˜b ∈R>0 such that:

∥˜bγ(θ, r, z)−˜bγ(θ′, r′, z′)∥≤K˜b(∥(θ, z) −(θ′, z′)∥+W2(r, r′)), ∀θ, θ′ ∈Θ, z, z′ ∈Z, r, r′ ∈P(Z).

Proof. From the definition and using the concavity of √· (which ensures that for any a, b ≥0,
√

a + b ≤√a +
√

b), we obtain

∥˜bγ(θ, r, z) −˜bγ(θ′, r′, z′)∥≤∥∇θEγ
λ(θ, r) −∇θEγ
λ(θ′, r′)∥+ ∥bγ(θ, r, z) −bγ(θ′, r′, z′)∥.

It is established below in Prop. 12 that ∇θEγ
λ satisfies a Lipschitz inequality, i.e., there is some
KEγ
λ ∈R>0 such that

∥∇θEγ
λ(θ, r) −∇θEγ
λ(θ′, r′)∥≤KEγ
λ(∥θ −θ′∥+ W2(r, r′)).

It is established below in Prop. 13 that bγ satisfies a Lipschitz inequality, i.e., there is some Kbγ ∈R>0
such that
∥bγ(θ, r, z) −bγ(θ′, r′, z′)∥≤Kbγ(∥(θ, z) −(θ′, z′)∥+ W2(r, r′)).

Hence, we have obtained as desired with K˜b = KEγ
λ + Kbγ.

Proposition 12. Under the same assumptions as Prop. 8, the function (θ, r) 7→∇θEγ
λ(θ, r) is
Lipschitz, i.e., there exist some constant KEγ
λ ∈R>0 such that

∥∇θEγ
λ(θ, r) −∇θEγ
λ(θ′, r′)∥≤KEγ
λ(∥θ −θ′∥+ W2(r, r′)), ∀(θ, r), (θ′, r′) ∈M.

Proof. From the definition, we have

∇θEγ
λ(θ, r) = ∇θEγ(θ, r) + ∇θRλ(θ, r).

Thus, if both ∇θEγ and ∇θRλ are Lipschitz, then so is ∇θEγ
λ. Since Rλ has Lipschitz gradient (by
Assumption 1), it remains to be shown that ∇θEγ is Lipschitz. From Prop. 6, we have

∇θEγ(θ, r) = Epk(ϵ)r(z)
h
(∇θϕθ · [sγ
θ,r −sp])(z, ϵ)
i
=
Z
∇θϕθ · dp,γ
θ,r (z, ϵ) pk(dϵ)r(dz),

where dp,γ
θ,r (z, ϵ) := sγ
θ,r(z, ϵ) −sp(z, ϵ). Then, applying Jensen’s inequality, we obtain

∥∇θEγ(θ, r) −∇θEγ(θ′, r′)∥

=


Z
p(ϵ)
Z h
∇θϕθ · dp,γ
θ,r (z, ϵ)r(z) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)r′(z)
i
dzdϵ


≤
Z
p(ϵ)


Z h
∇θϕθ · dp,γ
θ,r (z, ϵ)r(z) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)r′(z)
i
dz
 dϵ.
(31)

23


---Page Break---
Focusing on the integrand, we can upper-bound it with


Z h
∇θϕθ · dp,γ
θ,r (z, ϵ)r(z) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)r′(z)
i
dz


(a)
≤


Z
∇θϕθ · dp,γ
θ,r (z, ϵ) [r(z) −r′(z)]dz
 +


Z h
∇θϕθ · dp,γ
θ,r (z, ϵ) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)
i
r′(z)dz


(b)
≤
Z ∇θϕθ · dp,γ
θ,r (z, ϵ)
 |r(z) −r′(z)|dz

+
Z ∇θϕθ · dp,γ
θ,r (z, ϵ) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)
 r′(z)dz.

where in (a) we add and subtract the relevant terms and invoke the triangle inequality, and in (b) we
apply Jensen’s inequality. Plugging this back into Eq. (31), we obtain

∥∇θEγ(θ, r) −∇θEγ(θ′, r′)∥

≤
Z
Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z, ϵ)
 |r(z) −r′(z)|dz
(32)

+
Z
Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z, ϵ) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)
 r′(z)dz,
(33)

where the interchange of integrals is justified from Fubini’s theorem for non-negative functions (also
known as Tonelli’s theorem).

As we shall later show, the two terms have the following upper bounds:

(32) ≤KW1(r, r′), and
(34)

(33) ≤K(∥θ −θ′∥+ W1(r, r′)),
(35)

where K denotes a generic constant; and, upon noting that W1 ≤W2, we obtained the desired result.

Now, we shall verify Eqs. (34) and (35). For the Eq. (34), we use the fact that the map z 7→
Epk(ϵ)
∇θϕθ · dp
θ,r(z, ϵ)
 is Lipschitz then from the dual representation of W1, we obtain the
desired result. To see that the aforementioned map is Lipschitz,
Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z, ϵ)
 −Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z′, ϵ)



(a)
≤Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z, ϵ) −∇θϕθ · dp,γ
θ,r (z′, ϵ)


(b)
≤Epk(ϵ)
∇θϕθ(z, ϵ) · (dp,γ
θ,r (z, ϵ) −dp,γ
θ,r (z′, ϵ))


+ Epk(ϵ)
(∇θϕθ(z, ϵ) −∇θϕθ(z′, ϵ)) · dp,γ
θ,r (z′, ϵ)


(c)
≤Epk(ϵ)
h
∥∇θϕθ(z, ϵ)∥F
dp,γ
θ,r (z, ϵ) −dp,γ
θ,r (z′, ϵ)

i

+ Epk(ϵ)
h
∥∇θϕθ(z, ϵ) −∇θϕθ(z′, ϵ)∥F
dp,γ
θ,r (z′, ϵ)

i

(d)
≤Epk(ϵ) [(aϕ∥ϵ∥+ bϕ)(ad∥ϵ∥+ bd)] ∥z −z′∥

+ Epk(ϵ)
h
(aϕ∥ϵ∥+ bϕ)
dp,γ
θ,r (z′, ϵ)

i
∥z −z′∥,

(e)
≤Epk(ϵ) [(aϕ∥ϵ∥+ bϕ)(ad∥ϵ∥+ bd)] ∥z −z′∥

+ 1

2Epk(ϵ)


(aϕ∥ϵ∥+ bϕ)2 +
dp,γ
θ,r (z′, ϵ)

2
∥z −z′∥,

where (a) we use the reverse triangle inequality; (b) we add and subtract relevant terms and apply
the triangle inequality; (c) we use a property of the matrix norm with ∥· ∥F denoting the Frobenius
norm; (d) we utilize Assumption 3 and the Lipschitz property from Prop. 15; (e) we apply Young’s

24


---Page Break---
inequality. Then, from the fact that

Epk(ϵ) [(aϕ∥ϵ∥+ bϕ)(ad∥ϵ∥+ bd)] < ∞, Epk(ϵ)


(aϕ∥ϵ∥+ bϕ)2 +
dp,γ
θ,r (z′, ϵ)

2
< ∞,
(36)

which holds from the assumption that pk has finite second moments Assumption 3, and from our
assumption that Epk(ϵ)
dp,γ
θ,r (z′, ϵ)
 is bounded. Hence, the map is Lipschitz and so Eq. (34) holds.

As for Eq. (35), we focus on the integrand in Eq. (33)

Epk(ϵ)
∇θϕθ · dp,γ
θ,r (z, ϵ) −∇θϕθ′ · dp,γ
θ′,r′(z, ϵ)


≤Epk(ϵ)
h∇θϕθ · (dp,γ
θ,r −dp,γ
θ′,r′)(z, ϵ)
 +
(∇θϕθ −∇θϕθ′) · dp
θ′,r′(z, ϵ)

i

≤Epk(ϵ)
h
∥∇θϕθ(z, ϵ)∥F
(dp,γ
θ,r −dp,γ
θ′,r′)(z, ϵ)
 + ∥(∇θϕθ −∇θϕθ′)(z, ϵ)∥F
dp,γ
θ′,r′(z, ϵ)

i

≤Epk(ϵ) [(aϕ∥ϵ∥+ bϕ)(ad∥ϵ∥+ bd)] (∥θ −θ′∥+ W1(r, r′))

+Epk(ϵ)
h
(aϕ∥ϵ∥+ bϕ)
dp,γ
θ′,r′(z, ϵ)

i
∥θ −θ′∥,

where, for the last line, we apply Prop. 15 and Assumption 3. Applying Young’s inequality and (36),
we have the desired result.

Proposition 13 (bγ is Lipschitz). Under the same assumptions as Prop. 8, the map bγ is Kbγ-
Lipschitz, i.e., there exists a constant Kbγ ∈R>0 such that the following inequality holds for all
(θ, z, r), (θ′, z′, r′) ∈Θ × Rdz × P(Rdz):

∥bγ(θ, r, z) −bγ(θ′, r′, z′)∥≤Kbγ(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)).

Proof. One can write the drift bγ as follows (can be found in Eq. (43) similarly to Prop. 6), we have

bγ(θ, r, z) = −Epk(ϵ)
h
(∇zϕθ · [sγ
θ,r −sp + Γγ
θ,r])(z, ϵ)
i
+ ∇x log p0(z),

where Γγ
θ,r(z, ϵ) :=
γ∇xqθ,r(ϕθ(z,ϵ))
(qθ,r(ϕθ(z,ϵ))+γ)2 . Hence,

∥bγ(θ, r, z) −bγ(θ′, r′, z′)∥≤∥Epk(ϵ)[(∇zϕθ · [dp,γ
θ,r + Γγ
θ,r])(z, ϵ) −(∇zϕθ′ · [dp
θ′,r′ + Γγ
θ′,r′])(z′, ϵ)]∥

+ ∥∇z log p0(z) −∇z log p0(z′)∥

≤Epk(ϵ)∥(∇zϕθ · dp,γ
θ,r + Γγ
θ,r)(z, ϵ) −(∇zϕθ′ · [dp,γ
θ′,r′ + Γγ
θ′,r′])(z′, ϵ)∥

+ Kp0∥z −z′∥,

where for the last inequality we use Jensen’s inequality and Assumption 1. Since we have

Epk(ϵ)∥(∇zϕθ · [dp,γ
θ,r + Γγ
θ,r)](z, ϵ) −(∇zϕθ′ · [dp,γ
θ′,r′ + Γγ
θ′,r′])(z′, ϵ)∥

(a)
≤Epk(ϵ)∥(∇zϕθ · [dp,γ
θ,r + Γγ
θ,r])(z, ϵ) −∇zϕθ(z, ϵ) · [dp,γ
θ′,r′ + Γγ
θ′,r′](z′, ϵ)∥

+Epk(ϵ)∥∇zϕθ(z, ϵ) · [dp,γ
θ′,r′ + Γγ
θ′,r′](z′, ϵ) −(∇zϕθ′ · [dp,γ
θ′,r′ + Γγ
θ′,r′])(z′, ϵ)∥

(b)
≤Epk(ϵ)∥∇zϕθ(z, ϵ)∥F ∥(dp,γ
θ,r + Γγ
θ,r)(z, ϵ) −(dp,γ
θ′,r′ + Γγ
θ′,r′)(z′, ϵ)∥

+Epk(ϵ)∥∇zϕθ(z, ϵ) −∇zϕθ′(z′, ϵ)∥F ∥(dp,γ
θ′,r′ + Γγ
θ′,r′)(z′, ϵ)∥

(c)
≤Epk(ϵ)(aϕ∥ϵ∥+ bϕ)(∥dp,γ
θ,r (z, ϵ) −dp,γ
θ′,r′(z′, ϵ)∥+ ∥Γγ
θ,r(z, ϵ) −Γγ
θ′,r′(z′, ϵ)∥)
(37)

+Epk(ϵ)(aϕ∥ϵ∥+ bϕ)∥(dp,γ
θ′,r′ + Γγ
θ′,r′)(z′, ϵ)∥∥(θ, z) −(θ′, z′)∥
(38)

where, for (a), we add and subtract the relevant terms and invoke the triangle inequality, in (b) we
use properties of the matrix norm, and in (c) we use the bounded gradient and Lipschitz gradient in
Assumption 3. For Eq. (37); upon using Props. 14 and 15, which are established below, we obtain

Epk(ϵ)(aϕ∥ϵ∥+ bϕ)(∥dp,γ
θ,r (z, ϵ) −dp,γ
θ′,r′(z′, ϵ)∥+ ∥Γγ
θ,r(z, ϵ) −Γγ
θ′,r′(z′, ϵ)∥)

≤Epk(ϵ)(aϕ∥ϵ∥+ bϕ)[(ad + aΓ)∥ϵ∥+ (bd + bΓ)](∥(θ, z) −(θ′, z′)∥+ W1(r, r′)).
(39)

25


---Page Break---
As for the second term, Eq. (38),
Epk(ϵ)(aϕ∥ϵ∥+ bϕ)∥(dp,γ
θ′,r′ + Γγ
θ′,r′)(z′, ϵ)∥

(a)
≤Epk(ϵ)(aϕ∥ϵ∥+ bϕ)[∥dp,γ
θ′,r′(z′, ϵ)∥+ ∥Γγ
θ′,r′(z′, ϵ)∥]

(b)
≤Epk(ϵ)(aϕ∥ϵ∥+ bϕ)[∥dp,γ
θ′,r′(z′, ϵ)∥+ BΓ]

(c)
≤Epk(ϵ)
1
2(aϕ∥ϵ∥+ bϕ)2 + 1

2∥dp,γ
θ′,r′(z′, ϵ)∥2 + BΓ(aϕ∥ϵ∥+ bϕ)
(40)

where (a) follows from the triangle inequality, (b) we use Prop. 14 boundedness of Γ, (c) we apply
Young’s inequality to the first term. Similarly to Eq. (36), from our Assumption 3 and our boundness
assumption of the score, we have as desired. Combining Eq. (39) with the result of plugging Eq. (40)
into Eq. (38), we obtain the result.

Proposition 14 (Γ is Lipschitz and bounded). Under Assumption 2, the map (θ, r, z) 7→Γγ
θ,r(z, ϵ) is
Lipschitz and bounded. (Lipschitz) there is constants aΓ, bΓ ∈R>0 such that following hold:
∥Γγ
θ,r(z, ϵ) −Γγ
θ,r(z, ϵ)∥≤(aΓ∥ϵ∥+ bΓ)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)).

Furthermore, it is bounded
∥Γγ
θ,r(z, ϵ)∥≤BΓ.

Proof. Since Γγ
θ,r = γ∇x log(qθ,r(x)+γ)

qθ,r(x)+γ
, where x := ϕθ(z, ϵ), and x′ := ϕθ′(z′, ϵ), we have:

∥Γγ
θ,r(z, ϵ) −Γγ
θ′,r′(z′, ϵ)∥

≤γ

(qθ′,r′(x′) + γ)∇x log(qθ,r(x) + γ) −(qθ,r(x) + γ)∇x log(qθ′,r′(x′) + γ)

(qθ,r(x) + γ)(qθ′,r′(x′) + γ)



≤1

γ |qθ′,r′(x′) −qθ,r(x)|∥∇x log(qθ,r(x) + γ)∥

+ 1

γ (qθ,r(x) + γ)∥∇x log(qθ,r(x) + γ) −∇x log(qθ′,r′(x′) + γ)∥

≤Bk

γ2 |qθ′,r′(x′) −qθ,r(x)| + (Bk + γ)

γ
∥sγ
θ,r(z, ϵ) −sγ
θ′,r′(z′, ϵ)∥

≤BkKq

γ2
(1 + aϕ∥ϵ∥+ bϕ)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′))

+ Bk + γ

γ2
(as∥ϵ∥+ bs)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)),

where the last inequality follows from applying Prop. 18 and Assumption 3 to the first term and
Prop. 16 to the last term . Hence, we have as desired.

Boundedness follows from the fact that

γ∇xqθ,r(ϕθ(z,ϵ))
(qθ,r(ϕθ(z,ϵ))+γ)2
 ≤1

γ ∥∇xqθ,r(ϕθ(z, ϵ))∥≤Bk

γ .

Proposition 15. Under Assumptions 1 to 3, the map (θ, r, z) 7→sγ
θ,r(z, ϵ) −sp(z, ϵ) =: dp,γ
θ,r (z, ϵ)
satisfies the following: there exist ad, bd ∈R>0 such that for all (θ, r), (θ′, r′) ∈M, and z, z′ ∈Rdz,
we have
∥dp,γ
θ,r (z, ϵ) −dp,γ
θ′,r′(z′, ϵ)∥≤(ad∥ϵ∥+ bd)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)).

Proof. Let x := ϕθ(z, ϵ), and x′ := ϕθ′(z′, ϵ). Then, we have
∥dp,γ
θ,r (z, ϵ) −dp,γ
θ′,r′(z′, ϵ)∥≤∥∇x log p(x, y) −∇x log p(x′, y)∥

+ ∥∇x log(qθ,r(x) + γ) −∇x log(qθ′,r′(x′) + γ)∥

≤Kp∥x −x′∥+ ∥∇x log(qθ,r(x) + γ) −∇x log(qθ′,r′(x′) + γ)∥

≤Kp(aϕ∥ϵ∥+ bϕ)∥(θ, z) −(θ′, z′)∥

+ (as∥ϵ∥+ bs)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)),
where Prop. 16 and Assumptions 1 and 3 are used.

26


---Page Break---
Proposition 16 (s is Lipschitz). Under Assumptions 2 and 3 and γ > 0, the map (θ, r, z) 7→sγ
θ,r(z, ϵ)
satisfies the following: there exist constants as, bs ∈R>0 such that the following inequality holds for
all (θ, r), (θ′, r′) ∈M, and z, z′ ∈Rdz:

∥sγ
θ,r(z, ϵ) −sγ
θ′,r′(z′, ϵ)∥≤(as∥ϵ∥+ bs)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)),

Proof. For brevity, we write x = ϕθ(z, ϵ) and x′ = ϕθ′(z′, ϵ); from the definition, we have

∥sγ
θ,r(z, ϵ) −sγ
θ′,r′(z′, ϵ)∥=

∇xqθ,r (x)
qθ,r (x) + γ −∇xqθ′,r′ (x′)

qθ′,r′ (x′) + γ



(a)
≤

∇xqθ,r (x)
qθ,r (x) + γ −∇xqθ′,r′ (x′)

qθ,r (x) + γ

 +

∇xqθ′,r′ (x′)

qθ,r (x) + γ −∇xqθ′,r′ (x′)

qθ′,r′ (x′) + γ



≤
1
qθ,r (x) + γ ∥∇xqθ,r (x) −∇xqθ′,r′ (x′)∥

+ ∥∇xqθ′,r′ (x′) ∥

1
qθ,r (x) + γ −
1
qθ′,r′ (x′) + γ



(b)
≤1

γ ∥∇xqθ,r (x) −∇xqθ′,r′ (x′)∥+ Bk

γ2 |qθ,r(x) −qθ′,r′(x′)|,
(41)

where (a) we add and subtract the relevant terms and the triangle inequality; (b) we use the fact that
∥∇xqθ′,r′ (x′) ∥= ∥
R
∇xkθ′(x′|z)r′(dz)∥≤Bk (from Cauchy-Schwartz and the boundedness part
of Assumption 2). Now, we deal with the terms individually. For the first term in Eq. (41), we use
the fact that the map (θ, r, z) 7→∇xqθ,r(ϕθ(z, ϵ)) is Kq-Lipschitz from Prop. 17. As for the second
term in Eq. (41), we apply Prop. 18.

Hence, we obtain

∥sγ
θ,r(z, ϵ) −sγ
θ′,r′(z′, ϵ)∥≤
Kgq

γ
+ BkKq

γ2


(∥(θ, x) −(θ′, x′)∥+ W1(r, r′))

≤
Kgq

γ
+ BkKq

γ2


(1 + aϕ∥ϵ∥+ bϕ)(∥(θ, z) −(θ′, z′)∥+ W1(r, r′)),

where we use Assumption 3 for the last line.

Proposition 17. Under Assumption 2, the map (θ, r, x) 7→∇xqθ,r(x) is Lipschitz, i.e., for all ϵ,
there exists a Kgq ∈R>0 such that the following inequality holds for all (θ, r), (θ′, r′) ∈M and
z, z′ ∈Rdz,

∥∇xqθ,r(x) −∇xqθ′,r′(x′)∥≤Kgq(∥(θ, x) −(θ′, x′)∥+ W1(r, r′)).

Proof. From direct computation,

∥∇xqθ,r (x) −∇xqθ′,r′ (x′)∥=


Z
[∇xkθ(x|z)r (z) −∇xkθ′(x′|z)r′ (z)] dz


(a)
≤
Z
∥∇x[kθ(x|z) −kθ′(x′|z)]∥r (z) dz

+
Z
∥∇xkθ′(x′|z)∥|r −r′| (z) dz

(b)
≤Kgq(∥(θ, x) −(θ′, x′)∥+ W1(r, r′)),

where (a) we add and subtract the appropriate terms, apply the triangle inequality and the Cauchy-
Schwarz inequality; (b) for the first term, we use the Lipschitz gradient Assumption 2; and for the
second term, we use the dual representation of W1 with the fact map z 7→∥∇x log kθ(x|z)∥is
Kk-Lipschitz from the reverse triangle inequality and the Lipschitz Assumption 2.

Proposition 18. Under Assumption 2, the map (θ, r, x) 7→qθ,r(x) is Lipschitz, i.e., there exists some
Kq ∈R>0 such that for all (θ, r, x), (θ′, r′, x′) ∈Θ × P(Rdz) × Rdx, we have

|qθ,r(x) −qθ′,r′(x′)| < Kq(∥(θ, x) −(θ′, x′)∥+ W1(r, r′)).

27


---Page Break---
Proof. From direct computation, we have

|qθ,r(x) −qθ′,r′(x′)| ≤|qθ,r(x) −qθ,r′(x)| + |qθ,r′(x) −qθ′,r′(x′)|

≤
Z
|kθ(x|z)||r −r′|(z)dz +
Z
|kθ(x|z) −kθ′(x′|z)|r(z)dz

(a)
≤Kq(W1(r, r′) + ∥(θ, x) −(θ, x′)∥)
(42)

where (a) for the first term, we use the fact that the map z 7→|kθ(x|z)| is Bk-Lipschitz (from
the bounded gradient of Assumption 2), and again the Lipschitz property of k from the same
assumption.

G
Algorithmic details

G.1
Gradient estimator

Proof of Prop. 6. We show the derivation of the estimators in Eq. (10). Eq. (9) will follow similarly.
We have

∇zδrE[θ, r](z) = ∇zEkθ(x|z)


log qθ,r(x)

p(y, x)



= ∇zEϵ∼pk


log qθ,r(ϕθ(z, ϵ))

p(y, ϕθ(z, ϵ))


.

Assuming that ϕθ and pk are sufficiently regular to justify the interchange of differentiation and
integration, we obtain

∇zδrE[θ, r](z) = Eϵ∼pk


∇z log qθ,r(ϕ(z, ϵ))

p(y, ϕ(z, ϵ))


.

To obtain as desired, one can apply the chain rule.

Similarly, one can derive a Monte Carlo gradient estimator for ∇zδrEγ as follows:

∇zδrEγ[θ, r](z) = ∇zEkθ(x|z)


log qθ,r(x) + γ

p(y, x)
+
qθ,r(x)
qθ,r(x) + γ



= ∇zEϵ∼pk


log qθ,r(ϕθ(z, ϵ)) + γ

p(y, ϕθ(z, ϵ))
+
qθ,r(ϕθ(z, ϵ))
qθ,r(ϕθ(z, ϵ)) + γ


.

As before, if ϕθ and pk are sufficiently regular, we obtain

∇zδrEγ[θ, r](z) = Eϵ∼pk


∇z log qθ,r(ϕ(z, ϵ)) + γ

p(y, ϕ(z, ϵ))
+
γ∇qθ,r(ϕθ(z, ϵ))
(qθ,r(ϕθ(z, ϵ)) + γ)2


.
(43)

To obtain as desired, one can apply chain rule.

G.2
Preconditioners

Recall that the preconditioned gradient flow is given by

dθt = −Ψθ
t∇θEλ(θt, rt) dt, ∂trt = ∇z · (rtΨr
t∇zδrEλ[θt, rt]),

where δrEλ[θ, r] = δrE[θ, r] + log r/p0 We can rewrite the dynamics of rt as

∂trt = ∇z · (rtΨr
t∇z[δrE[θt, rt] −log p0 + log rt]),
= ∇z · (rtΨr
t∇z[δrE[θt, rt] −log p0]) + ∇z · (rtΨr
t∇z log rt).

The second term can be written as

∇z · (rtΨr
t∇z log rt) = ∇z · (Ψr
t∇zrt) = ∇z · (∇z · [Ψr
trt]) −∇z · (rt∇z · Ψr
t)

28


---Page Break---
Layers
Size
Input
din
Linear(din, dh), LReLU
dh
Linear(dh,dh), LReLU
dh
Linear(dh,dout),
dout
Table 3: Neural network architecture defined by NN(din, dh, dout).

where (∇· Ψr
t)i = Pdz
j=1 ∂zj[(Ψr
t)ij], and last equality holds since

dz
X

i=1
∂zi






dz
X

j=1
(Ψr
t)ij∂zjrt




=

dz
X

i=1
∂zi






dz
X

j=1

 
∂zj [(Ψr
t)ijrt] −rt∂zj[(Ψr
t)ij]




.

Hence, we have the following dynamics of rt:

∂trt = ∇z · (rt (Ψr
t∇z[δrE[θt, rt] −log p0] −∇z · Ψr
t)) + ∇z · (∇z · [Ψr
trt]))

Examples. Following in the essence of RMSProp (Tieleman and Hinton, 2012), we utilize the
preconditioner defined as follows:

Bk = βBk−1 + (1 −β)Diag(A({∇zδrE[θk, rk](Zm)2}M
m=1))

Ψr
k(Z) = (Bk)−0.5

where Bk ∈Rdz×dz and A is some aggregation function such as the mean or max. The idea is to
normalize by the aggregated gradient of the first variation across all the particles since this is the
dominant component in the drift of PVI. Similarly to RMSProp, it keeps an exponential moving
average of the squared gradient which can then be used in the preconditioner.

H
Experimental details

In this section, we highlight additional details for reproducibility and computation. The code was
written in JAX (Bradbury et al., 2018) and executed on a NVIDIA GeForce RTX 4090.

H.1
Section 5.1

Hyperparameters. For the neural network, we use fθ = NN(2, 128, 2) defined in Table 3, the
number of particles M = 100, dz = 2, K = 1000, hθ = 10−4, hz = 10−2, λr = 10−8, for Ψθ we
use RMSProp and we set Ψr = I2.

Computation Time. Each run took 8 seconds using JIT compilation.

H.2
Section 5.2

In this section, we outline all the experimental details regarding Section 5.2.

Densities. Table 4 shows the densities used in the toy experiments.

Name
Density

Banana
N(z2; z2
1/4, 1)N(z1; 0, 2)

X-Shape
1
2N

0,

2
1.8
1.8
2


+ 1

2N

0,

2
−1.8
−1.8
2



Multimodal
1
8N

2
2


, I

+ 1

8N

−2
−2


, I

+ 1

2N

2
−2


, I

+ 1

4N

−2
2


, I


Table 4: Densities used in toy experiments (see Section 5.2).

Hyperparameters. We set the number of parameter updates and particle steps to be K = 15000,
and dz = 2.

29


---Page Break---
• fθ. We use fθ = NN(2, 512, 2).

• PVI. We use M = 100, λθ = 0, λr = 10−8, hx = 10−2, hθ = 10−4, Ψθ we use the
RMSProp preconditoner, Ψr = Idz, and L = 250.

• SVI. We use K = 50 to estimate the objective (Yin and Zhou, 2018, see Eq. 5) which are
around the values used in (Yin and Zhou, 2018). We utilize RMSProp with step size 10−4,
and r = N(0, Idz). The implicit distribution is set to r = N(0, Idz).

• UVI. For the HMC sampler, we follow in (Titsias and Ruiz, 2019) and use 50 burn-in steps,
with step-size 10−1 and 5 leap-frog steps. We use the RMSProp optimizer with stepsize
10−4 for kθ. The implicit distribution is set to r = N(0, Idz).

• SM. For the “dual” function written as f in the original paper (Yu and Zhang, 2023, see
Algorithm 1) we use NN(2, 512, 2). We utilize RMSProp with decaying learning rate from
10−4 to 10−5 to optimize the kernel kθ, and RMSProp with 10−3 to 10−4 for the dual
function f. The implicit distribution is set to r = N(0, Idz).

Sliced Wasserstein Distance. We report the average sliced Wasserstein distance using 100 projections
computed from 10000 samples from the target and the variational distribution.

Two-Sample Test. We use the MMD-Fuse implementation found in https://github.com/
antoninschrab/mmdfuse.git.

Computation Time. An example run on Banana with JIT compilation, PVI took 42 seconds, UVI
took 10 minutes 36 seconds, SM took 45 seconds, and SVI took 38 seconds.

H.3
Section 5.3

In this section, we outline all the hyperparameters for each method used.

Hyperparameters. We use K = 20000 set dz = 10. For all kernel parameters, we use RMSProp
preconditioner with step size hθ = 10−3 .

• fθ. We use fθ = NN(dz, 512, 22).

• PVI. We use M = 100, λθ = 0, λr = 10−8, hx = 10−2, and for Ψr we use the one
described in App. G.2 with mean as the aggregate function.

• SVI. We use K = 50 to estimate the objective (Yin and Zhou, 2018, see Eq. 5) which are
around the values used in (Yin and Zhou, 2018).

• UVI. For the HMC sampler, we follow in (Titsias and Ruiz, 2019) and use 50 burn-in steps,
with step-size 10−1 and 5 leap-frog steps.

• SM. We were unable to improve the performance of SM with our chosen kernel and instead
used the implementation in https://github.com/longinYu/SIVISM?utm_source=
catalyzex.com to obtain posterior samples with implementation details found in the
code repository and in the paper (Yu and Zhang, 2023).

H.4
Section 5.4

In this section, we outline all the experimental details regarding Section 5.4.

Model. We consider the neural network BNN(dbnn
in , dbnn
h
) defined as fx(o) = W ⊤
2 ReLU(W ⊤
1 o +
b1) + b2 where o ∈Rdbnn
in , x = [vec(W2), b2, vec(W1), b1]⊤, W2 ∈Rdbnn
h
×1, b2 ∈R, W1 ∈
Rdbnn
in ×dbnn
h
, b1 ∈Rdbnn
h
. Given an input-output pair Y := {(Oi, Yi)}B
i=1, the model can be defined
as p(Y , x) = p(Y |x)p(x) where the likelihood is p(Y |x) = QB
i=1 N(Yi; fx(Oi), 0.012) and the
prior is N(x; 0, 25I).

Datasets. For all the datasets, we standardize by removing the mean and dividing by the standard
deviation.

• Protein. For the model, we use BNN(9, 30) which results in the problem having dimension
dx = 331. The dataset is composed of 1600 train examples, 401 test examples.

30


---Page Break---
• Yacht. For the model, we use BNN(6, 10) which results in the problem having dimension
dx = 81. The dataset is composed of 246 train examples and 62 test examples.
• Concrete For the model, we use BNN(8, 10) which results in the problem having dimension
dx = 101. The dataset comprises of 824 training examples and 206 test examples.

Hyperparameters. We use K = 1500 set dz = 10. For all kernel parameters, we use RMSProp
preconditioner with step size hθ = 10−3 that decays to 10−5 following a constant schedule that
transitions every 100 parameters steps.

• fθ, σθ. We use fθ = NN(dz, 512, dx) and σθ = Softplus(NN(dz, 512, dx)) + 10−8 and
they share parameters except for the last layers.
• PVI. We use M = 100, λθ = 0, λr = 10−3, hx = 10−3, and for Ψr we use the one
described in App. G.2 with mean as the aggregate function.
• SVI. We use K = 50 to estimate the objective (Yin and Zhou, 2018, see Eq. 5) which
are around the values used in (Yin and Zhou, 2018). The implicit distribution is set to
r = N(0, Idz).
• UVI. For the HMC sampler, we follow in (Titsias and Ruiz, 2019) and use 50 burn-in steps,
with step-size 10−1 and 5 leap-frog steps. The implicit distribution is set to r = N(0, Idz).
• SM. For the “dual” function written as f in the original paper (Yu and Zhang, 2023, see
Algorithm 1) we use NN(dx, 512, dx) and trained with RMSProp with stepsize 10−2. We
tried a decaying learning schedule to 10−4 but found that this degraded the performance.
We used ReLU activations instead as we found that using leaky ReLUs harmed performance.
The implicit distribution is set to r = N(0, Idz).

Computation Time. For each run in the Concrete dataset with JIT compilation, PVI took 37 seconds,
UVI took approximately 1 minute 40 seconds, SVI took 30 seconds, and SM took 27 seconds.

31


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We clearly define our paper scope and contributions at the end of the introduc-
tion with references to why they are accurate.
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
Justification: We highlight the limitations in Section 6.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
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

32


---Page Break---
Justification: For each result, we outline relevant assumptions which can be found in the
appendix.
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
Justification: These can be found in App. H.
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

33


---Page Break---
Answer: [Yes]

Justification: We provide code for reproducibility.

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

Justification: These can be found in App. H and explanations can be found in Section 5.

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

Justification: In our results, we report the mean and standard deviations from independent
trials which can be found in Tables 1 and 2.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).

34


---Page Break---
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
Justification: We outline the computational resource in App. H.
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
Justification: We conform.
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
Justification: This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none which we feel
must be specifically highlighted here.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

35


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

Justification: We do not release data or models that have a high risk of misuse.

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

Justification: Yes, we reference the dataset used.

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

36


---Page Break---
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
Justification: No human subjects were used.
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
Justification: We did not have human participants.
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

37


---Page Break---
