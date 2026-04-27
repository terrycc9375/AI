Variance estimation in compound decision theory
under boundedness

Subhodh Kotekal
Department of Statistics
University of Chicago
Chicago, IL 60637
skotekal@uchicago.edu

Abstract

The normal means model is often studied under the assumption of a known variance.
However, ignorance of the variance is a frequent issue in applications and basic
theoretical questions still remain open in this setting. This article establishes that
the sharp minimax rate of variance estimation in square error is (log log n/ log n)2
under arguably the most mild assumption imposed for identifiability: bounded
means. The rate-optimal estimator proposed in this article achieves the optimal
rate by estimating O (log n/ log log n) cumulants and leveraging a variational rep-
resentation of the noise variance in terms of the cumulants of the data distribution.
The minimax lower bound involves a moment matching construction.

1
Introduction

Consider the prototypical normal means model in compound decision theory,

Xi
ind
∼N(µi, σ2)
(1)

for 1 ≤i ≤n, where the means µ = (µ1, ..., µn) ∈Rn and the noise level σ > 0 are parameters. As
set out in the groundbreaking work of Robbins [47, 48, 67], the empirical Bayes setting is closely
related,

µi
iid
∼G,

Xi | µi
ind
∼N(µi, σ2),
(2)

where the prior G is treated as an unknown parameter in contrast to a fully Bayesian approach. The
empirical Bayes and compound decision theory literatures, motivated by the advent of scientific
technologies (e.g. microarrays) enabling large-scale, parallel experiments, have witnessed tremendous
methodological successes [18, 15, 67, 27, 19, 20, 44, 26, 24, 62, 65, 12, 21, 34, 55] and deep
theoretical developments [66, 54, 23, 68, 31, 30, 46, 49, 33, 2, 8, 9, 45, 52].

In the Gaussian contexts of (1) and (2), existing theoretical work has largely focused on mean
estimation or hypothesis testing when the variance σ2 is known. The goal is to produce an estimator
or a test and evaluate its performance with respect to the benchmark achieved by an oracle (termed
oracle Bayes in [18, 31]) having access to information not available to the statistician (e.g. the prior
G in (2) or the empirical distribution 1

n
Pn
i=1 δµi in (1)). For estimation under square loss, the oracle
estimator is µ∗
i = T ∗(Xi) where

T ∗(x) = E(µ | X = x) = x + σ2 f ′(x)

f(x)
(3)

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
is the posterior mean, under the prior µ ∼G in (2) or µ ∼
1
n
Pn
i=1 δµi in (1), and is known as
Tweedie’s formula [16]. Here, f denotes the density of the marginal distribution of X and the second
term in (3) is known as the Bayes correction. Tweedie’s formula requires knowledge of the prior in
order to compute f and is thus available only to the oracle.

Broadly, there are two common approaches to mimicking the oracle [17]. Typically, the statistician
either estimates the prior (known as g-modelling) or directly estimates the marginal distribution
f (known as f-modelling), and plugs in the resulting estimate into Tweedie’s formula (3). Both
approaches have been taken in the literature [66, 3, 2, 46, 28, 51], with the nonparametric maxi-
mum likelihood estimator (NPMLE) [33, 31, 34, 65, 51] being an especially popular method for
g-modelling; regret bounds have been established under various assumptions on the class of priors.

Not much of the literature addresses the case of an unknown variance. The dearth of results
is certainly not from want of motivation; estimation of the variance is necessary for mimicking
the oracle estimator (3), construction of confidence intervals, hypothesis testing, and estimation
of the signal-to-noise ratio among other statistical tasks. Rather, variance estimation has been
recognized as a challenging problem in a wide range of high-dimensional and nonparametric models
[61, 50, 35, 13, 11, 60, 29, 36, 4, 42]. A few articles have investigated a heteroskedastic version of
(1) or (2) with unknown variances [40, 41, 25], but all of these articles essentially assume access
to multiple i.i.d. observations per mean, thereby enabling direct estimation of each variance. The
situation in (1) and (2) is quite different in that only one observation per mean is available; it is an
honest sequence setting. Despite its clear importance, the fundamental limit of variance estimation in
(1) and (2) has not been established in the literature.

This article investigates variance estimation from a minimax perspective. We will work in the
compound decision setting (1), and so our results will also directly hold in the empirical Bayes setting
(2). For the development of the minimax theory, the following parameter space will be considered,

Θ(L) := {(µ, σ) ∈Rn × (0, ∞) : ||µ||∞≤1 and σ ≤L} ,
(4)

where L > 0. We will focus on the case where L > 0 is some large universal constant, and we
will notationally suppress it by writing Θ to refer to (4). Furthermore, the choice of 1 in the bound
||µ||∞≤1 is not essential; our results essentially go through when it is replaced by some other
universal constant. The boundedness conditions are imposed to avoid triviality. Without boundedness,
the minimax risk is easily seen to be infinite, inf ˆσ supµ∈Rn,σ>0 E(
ˆσ2 −σ22) = ∞, since the
variance is not identifiable. In the empirical Bayes context (2), the constraint ||µ||∞≤1 corresponds
to the condition that G is supported on [−1, 1]; the class of priors with bounded support is a popular
choice for study in the empirical Bayes literature [46, 37, 3, 31, 38]. This article’s goal is to obtain a
sharp characterization of the minimax rate of variance estimation in (1) under square loss over the
parameter space (4). Appendix E contains the notation used in this article.

1.1
Related work

The marginal distribution of the data in the empirical Bayes model (2) can be written as X1, ..., Xn
iid
∼
G∗N(0, σ2) where ∗denotes convolution. The problem of estimating σ2 is a special case of variance
estimation in a semiparametric convolution model [42, 4, 43], which encompasses generic noise
distributions beyond Gaussian; the Fourier transform of the standardized (i.e. unit-scale) noise is
assumed known. In their article [4], Butucea and Matias impose regularity conditions on G by
way of assumptions on its Fourier transform. Butucea and Matias point out it is essential G is less
smooth than the standardized noise distribution. In the Gaussian case, they assume the Fourier
transform ˆG does not vanish for large frequencies and its modulus does not decay faster than the
Fourier transform of the standard Gaussian distribution ω 7→e−ω2/2. If G may be somewhat
smooth in the sense | ˆG(ω)| ≥ce−α|ω|r for |ω| sufficiently large where 0 < r < 2, α > 0, and
c > 0 is an arbitrary constant, then Butucea and Matias construct a Fourier-based estimator which
achieves |ˆσ2 −σ2|2 ≲(log n)r−2 with high probability. In the rougher case | ˆG(ω)| ≥c|ω|−β for
|ω| sufficiently large where β > 1, the faster rate |ˆσ2 −σ2|2 ≲(log log n/ log n)2 is achieved with
high probability. Matching lower bounds are also obtained.

Though the results of [4] are sharp in their setting, the smoothness assumptions are unappealing and
do not imply anything when G is only assumed to have bounded support. In particular, G being
supported on [−1, 1] does not imply G must fall into one of the two cases considered by [4]. For

2


---Page Break---
example, the Uniform[−1, 1] distribution and the mixture 1

2δ−1 + 1

2δ1 respectively have Fourier
transforms ω 7→sin(ω)/ω and ω 7→cos(ω), both of which have zeros that are arbitrarily large.
Without the condition that G has bounded support, the assumptions of [4] which stipulate | ˆG(w)| is
bounded away from zero for large |ω| are made to ensure identifiability of the variance. However,
identifiability is automatically guaranteed when G is assumed to be supported on [−1, 1]. Though
the assumptions of [4] can be safely abandoned, it is not clear how, if it all, the approach of Butucea
and Matias can be modified without smoothness conditions. Setting this serious issue aside, the keen
reader might intuit from an uncertainty principle that if G has a density g which is not identically zero,
then the Fourier transform of g cannot decay too fast since it is compactly supported. In particular, it
is well known it cannot decay faster than Ce−c|ω|, and so it might be guessed by taking r = 1 in the
results of [4] that log−1 n may be optimal.

Moving away from smoothness conditions and assuming only that G has bounded support, Matias [42]
notices the moment generating function of G∗N(0, σ2) is finite everywhere and is given by M(λ) =
 R
exp(λµ)G(dµ)

exp(λ2σ2/2). Therefore, σ2 is identifiable since limλ→∞
2
λ2 log M(λ) = σ2
due to the bounded support of G. With this observation in hand, Matias forms the empirical
moment generating function ˆ
M(λ) = 1

n
Pn
i=1 eλXi, chooses λ ≍√log n, and defines the estimator
ˆσ2 =
2
λ2 log ˆ
M(λ). Matias establishes the upper bound E(|ˆσ2 −σ2|2) ≲
1
log n, but no matching
lower bound is obtained. Though the use of the empirical moment generating function is clever, the
simpler estimator ˆσ = max1≤i≤n Xi/√2 log n also achieves the same rate |ˆσ2 −σ2|2 ≲
1
log n with
high probability as pointed out by [63].

A different perspective regards the marginal distribution X1, ..., Xn
iid
∼G ∗N(0, σ2) as a Gaussian
mixture model with mixing distribution G. Employing a moment-based approach, Wu and Yang
[63] construct an estimator which achieves |ˆσ2 −σ2|2 ≲k2n−1

k with high probability under the
assumption G is a k-atomic measure supported on [−1, 1] with k ≲log n/ log log n. When k is
a fixed constant, a matching minimax lower bound is obtained. Their result highlights a rapid
deterioration in the convergence rate as k increases. In particular, in the generic case where G
need not be atomic (which can be intuited as k →∞), it might be expected the sharp rate will be
logarithmic. Appendix A further discusses the results of [63].

Variance estimation has also been studied under sparsity or regularity assumptions on the means
[11, 61, 5, 36]. In (1) under the assumption ||µ||0 ≤k < n

2 , an estimator achieving |ˆσ2 −σ2| ≲
σ2 k

n log−1 (1 + k/√n) was constructed and shown to be minimax rate-optimal in [36] (see also
[11, 7]). In the context of (2), faster rates of convergence were obtained by [5] when regularity on the
means was assumed in addition to the sparsity. Fourier-based estimators were employed by all of
[36, 11, 5]. Optimal estimation with Hölder-type regularity and without sparsity assumptions was
obtained by way of kernel smoothing in [61]. These approaches critically exploit the regularity and/or
sparsity, and all appear to fail when only boundedness ||µ||∞≤1 is assumed.

Beyond the sequence setting of (1) and (2), much recent attention has been directed at variance
estimation in the linear regression model Yi = ⟨Xi, β⟩+ σZi where the noise Zi ∼N(0, 1) and
the design Xi ∼N(0, Σ) are independent. If Σ ∈Rp×p is known with bounded operator norm and
||β|| ≲1, then Dicker [14] showed the variance can be estimated at rate |ˆσ2 −σ2|2 ≲n−1 + pn−2
with high probability; notably, consistent estimation is possible even with p ≥n without any sparsity
assumptions on β. Assuming β is k-sparse, [53] (see also [1, 22]) established the upper bound
n−1+((k log p)/n)2. For particular regimes, optimality was established in [60], and the impossibility
of consistent estimation for the dense regime was conjectured in the setting n = o(p) and unknown
Σ. Though some approaches were suggested in [13, 29], variance estimation with unknown Σ was
not settled until the impressive work of Kong and Valiant [35], who showed consistent estimation
with n = o(p) samples is surprisingly possible even when k = p. Assuming ||β|| ≲1 and Σ has
bounded condition number, an estimator can be constructed achieving, with high probability, error
|ˆσ2 −σ2| ≤ϵ with sample complexity n = O(poly(log(1/ϵ))p1−log−1(1/ϵ)). Without a condition
number assumption on Σ, consistent estimation was shown to be possible with sample complexity
n = O(poly(1/ϵ)p1−√ϵ). Moreover, Kong and Valiant [35] show these complexities are essentially
tight. Their estimator is based on a clever polynomial approximation scheme. Though their article
also covers certain covariate distributions which are not Gaussian, the sequence settings of (1) and (2)
are not covered.

3


---Page Break---
1.2
Main contribution

Our main result is a sharp characterization of the minimax rate of variance estimation over (4) in the
compound decision setting (1),

inf
ˆσ
sup
(µ,σ)∈Θ
Eµ,σ
ˆσ2 −σ22
≍
log log n

log n

2
.
(5)

A few remarks are in order.
Remark 1. Though the slow, logarithmic nature of (5) is not surprising given the Gaussian mixture
model results of [63] discussed earlier, it is notable a rate faster than log−1 n is achievable. The set of
means µ in the parameter space (4) have no exploitable structure beyond ||µ||∞≤1, and so intuition
suggests to examine the tails of the observations to estimate σ2. This thinking may further suggest the
least favorable prior might place substantial mass near ||µ||∞≈1, thus suggesting using an extreme
order statistic.

Though natural, this intuition does not fully use the statistician’s knowledge the noise is Gaussian.
Conceptually, if the signal distribution looks very different from a Gaussian, then the signal might
be easily disentangled from the noise, enabling easy variance estimation.1 From a lower bound
perspective, the least favorable prior for the means should resemble the noise. Indeed, our lower
bound argument in Section 3 involves constructing a compactly supported prior which shares a
growing number of moments with a particular Gaussian distribution. For technical reasons elaborated
on in Section 3, it turns out one can only match O(log n/ log log n) moments, which subsequently
yields a lower bound of order (log log n/ log n)2 for variance estimation.

The quantitative constraint on the number of moments which can be matched in the lower bound
construction inspires hope it may be possible to estimate at a rate faster than log−1 n, and it encourages
developing some kind of method-of-moments estimator. Though such a strategy has seen success in
a linear regression setting [35], complications seem to appear in the sequence model (1) as discussed
in Remark 2. Cumulants, which are closely related to moments, have much more convenient
properties, and so we develop a cumulant-based variance estimator which essentially requires
estimating O(log n/ log log n) cumulants, parallelling the lower bound.
Remark 2. Variance estimation in the sequence setting (1) turns out to be harder than in regression.
Consider the linear regression model Yi = ⟨Xi, β⟩+ σZi where β ∈Rp is unknown, and the noise
Zi ∼N(0, 1) is independent of the design Xi ∼N(0, Σ). For comparison with the sequence model
(1), consider the setting p = n and ||Σ|| ∨||β|| ≲1. Without any condition number assumption on
Σ, Kong and Valiant [35] construct an estimator achieving, with high probability, the rate (translating
their sample complexity result to estimation risk) |ˆσ2 −σ2|2 ≲(log log n/ log n)4. Though this
result is not directly comparable to (5) since Kong and Valiant impose the ℓ2-norm constraint ||β|| ≲1
whereas the space (4) imposes an ℓ∞-norm constraint more natural for compound decision theory, it
is helpful for intuition to appreciate how the regression setting differs from the sequence setting to
yield a “faster” rate.

In regression, it is clear the second marginal moment is E(Y 2
1 ) = ⟨β, Σβ⟩+ σ2, and so variance
estimation is (up to parametric rate) equivalent to estimation of ⟨β, Σβ⟩. By changing focus to
this target, σ2 is now viewed as a nuisance. The key advantage in regression is that the covariates
are a source of fresh randomness that is independent of the noise. Since Xi is correlated with Yi
but independent of Zi, it is clear E(Y1⟨X1, X2⟩Y2) = ⟨β, Σ2β⟩, E(Y1⟨X1, X2⟩⟨X2, X3⟩Y3) =
⟨β, Σ3β⟩, and so on. In other words, unbiased estimators of the terms

⟨β, Σkβ⟩
	n
k=2 are easily
constructed due to the availability of the random covariates {Xi}n
i=1. The clever idea of Kong and
Valiant [35] is to reach for polynomial approximation tools and approximate ⟨β, Σβ⟩by the readily
estimable

⟨β, Σkβ⟩
	n
k=2.

The sequence model (1) is equivalent to linear regression with a deterministic orthogonal design,
which is quite different from the random design considered above. The target ||µ||2 is the target
analogous to ⟨β, Σβ⟩. The only accessible randomness in (1) are the responses themselves {Xi}n
i=1.
The marginal moments 1

n
Pn
i=1 E(Xk
i ) = 1

n
Pn
i=1 E((µi + σZi)k) all involve the nuisance σ, thus
precluding an application of Kong and Valiant’s idea. The interesting phenomenon that design

1At a high-level, this is the core spirit of the results in the deconvolution literature (e.g. [4, 42]). However,
we do not adopt a Fourier-based lens due to the issues discussed earlier.

4


---Page Break---
properties can have substantial effects on minimax rates has been noted in the literature in other
settings [11, 6, 59], and we suspect this is the cause for the slower rate (5) for variance estimation in
the sequence model (1).
Remark 3. Given the central position of the Gaussian sequence models (1) and (2) in compound
decision theory and empirical Bayes theory (as well as their status at the core of mathematical
statistics more broadly), the cumulant-based variance estimator we propose in this article is designed
to exploit the Gaussian character of the noise. It turns out incorporating noise information is, in
some sense, essential; in Section 4 it is discussed that noise agnosticism implies the impossibility
of consistent variance estimation. Interestingly, the results of Kong and Valiant [35] in the linear
regression setting allow for an unknown noise distribution.2 There appears to be subtle interplay
between properties of the design and the noise; a careful study remains an open problem.

2
A cumulant-based estimator

As noted in Remark 1, the variance estimator we will propose is cumulant-based. Cumulants, though
perhaps not as familiar, are related to the moments of a random variable. But, in contrast to moments,
they have particularly nice attributes for the purpose of variance estimation in (1) and (2). Some key
properties of cumulants are briefly reviewed before the estimation methodology is developed.

2.1
A brief review of cumulants

Suppose Y is a random variable such that its moment generating function M(λ) := E(eλY ) exists
for all λ ∈R. The cumulant generating function is defined by K(λ) := log M(λ), and it admits the
power series K(λ) = P∞
r=1
κrλr

r!
where κr := K(r)(0) is defined to be the rth cumulant. Though
κ1 and κ2 coincide respectively with the mean and variance of Y , the higher cumulants do not have a
simple relationship with the moments. However, the first r cumulants are determined completely by
the first r moments; likewise, the first r moments are determined completely by the first r cumulants.
In fact, an explicit correspondence is available via Bell polynomials.
Definition 1. The incomplete Bell polynomial Br,l for l ≤r is given by

Br,l(x1, ..., xr−l+1) =
X
r!
j1!j2! · · · jr−l+1!

x1

1!

j1 x2

2!

j2
· · ·

xr−l+1
(r −l + 1)!

jr−l+1

where the sum is taken over all sequences j1, j2, j3, ..., jr−l+1 of nonnegative integers such that
j1 + ... + jr−l+1 = l and j1 + 2j2 + 3j3 + ... + (r −l + 1)jr−l+1 = r.

The correspondence between moments and cumulants is given by

κr =

r
X

l=1
(−1)l−1(l −1)! Br,l(m1, m2, ..., mr−l+1),
(6)

mr =

r
X

l=1
Br,l(κ1, κ2, κ3, ..., κr−l+1),
(7)

where mr = E(Y r) denotes the rth moment of Y . As is evident by its definition, the cumulant
generating function plays nicely with convolutions. The cumulant generating function of the random
variable Y1 + Y2 for independent Y1 and Y2 is given by the sum of the individual cumulant generating
functions. Consequently, the rth cumulant of the sum is the sum of the rth cumulants.

2.2
Identifying the noise variance from the marginal cumulants

In the model (1), the second moment exhibits 1

n
Pn
i=1 X2
i = ||µ||2/n + σ2 + OP (n−1/2) due to the
boundedness of ||µ||∞and σ. It is clear estimation of σ2 is equivalent to estimation of ||µ||2/n up to
a (negligible) parametric slowdown in the rate. Associated with any estimator ˆQ for the quadratic
functional ||µ||2

n
is a corresponding variance estimator ˆσ2 = 1

n
Pn
i=1 X2
i −ˆQ. Hence, it suffices

2Their results also allow for a quite general class of covariate distributions beyond centered multivariate
Gaussians.

5


---Page Break---
to focus attention on quadratic functional estimation. The estimation methodology is more plainly
motivated in the empirical Bayes context (2), so the following discussion will develop ideas in that
setting.

In the context of (2), the goal is to estimate the second moment of G. Let κr and mr denote the rth
cumulant and moment respectively of the marginal distribution G ∗N(0, σ2) of the data. Likewise,
let γr and νr denote the rth cumulant and moment respectively of G. Since the noise is mean zero
and σ is bounded, the mean of G can be estimated at parametric rate, and so it is equivalent (up to a
negligible parametric slowdown) to estimate the variance of G, i.e. γ2.

It is not immediately clear how to identify γ2 from the cumulants κr of the marginal distribution of
the data G ∗N(0, σ2). Consider the moment generating function of N(0, σ2) is λ 7→exp(λ2σ2/2),
and so all the cumulants (except the second) are equal to zero. Therefore, κ1 = γ1, κ2 = γ2 +σ2, and
κr = γr for r ≥3. The upshot is we are able to directly estimate all the cumulants of G, except the
second, from {Xi}n
i=1. Unfortunately, it is impossible to generically reconstruct the second cumulant
from the other cumulants. Indeed, the collection of all centered Gaussians differ only in their second
cumulant (the rest are zero).

It turns out it is possible to identify γ2 by exploiting the boundedness of G’s support. For r ≥2,
define the function Mr : [0, ∞) →R with

Mr(γ) =

r
X

l=1
Br,l(γ1, γ, γ3, ..., γr−l+1)
(8)

where Br,l is an incomplete Bell polynomial (see Definition 1). Note by (7) that Mr gives the rth
moment associated to the cumulant sequence (γ1, γ, γ3, . . .) if it is a valid cumulant sequence. The
following proposition asserts a variational representation of γ2.

Proposition 1. If G is supported on [−1, 1], then γ2 = sup {γ ∈[0, 1] : |Mr(γ)| ≤1 for all r ≥2}.

Proof. For ease of notation, let γ∗= sup {γ ∈[0, 1] : |Mr(γ)| ≤1 for all r ≥2}. It is clear γ2 ≤
γ∗since Mr(γ2) = νr ∈[−1, 1] for all r ≥2. To show the lower bound, fix δ > 0. For any τ 2 ≥δ,
observe that the cumulant sequence of the distribution G ∗N(0, τ 2) is (γ1, γ2 + τ 2, γ3, γ4, ...). Since
G ∗N(0, τ 2) does not have all moments contained in [−1, 1], it follows γ2 + τ 2 is not a feasible
point. Since this holds for all τ 2 ≥δ, it immediately follows γ∗≤γ2 + δ. Since δ > 0 was arbitrary,
we have shown γ∗≤γ2.

Proposition 1 enables recovery of γ2 from the other cumulants. However, it requires certifying all
putative moments {Mr(γ)}∞
r=2 live in [−1, 1], which is a difficult task given only a finite amount of
data. One idea is to approximate γ2 by certifying only that an rth putative moment lies in [−1, 1] for
some large choice of r. Specifically, define for r ≥2,

˜γ2(r) := sup {γ ∈[0, 1] : |Mr(γ)| ≤1} .
(9)

Proposition 2. If r is even, then |˜γ2(r) −γ2| ≤C

r for some universal constant C > 0.

Proof. By (9) and Proposition 1, it follows γ2 ≤˜γ2(r). Therefore, Mr(˜γ2(r)) is the rth moment
of G ∗N(0, ˜γ2(r) −γ2). Since r is even, taking µ ∼G and Z ∼N(0, 1), it follows 1 ≥
|Mr(˜γ2(r))| = E((µ +
p

˜γ2(r) −γ2Z)r) = Pr
l=0
 r
l

E(µr−l)(˜γ2(r) −γ2)l/2E(Zl). If r −l
is odd, then l must be odd since r is even, and so we must have E(Zl) = 0. It thus follows
1 ≥P
l≤r, l even
 r
l

E(|µ|r−l)(˜γ2(r) −γ2)l/2E(Zl) ≥(˜γ2(r) −γ2)r/2E(Zr). Hence, |˜γ2(r) −

γ2| ≤E(Zr)−2/r = π−1/r  
2r/2Γ
  r+1

2
−2/r. It follows from Stirling’s approximation that for
some small universal constant c > 0 whose value may change from instance to instance, we have

Γ
  r+1

2
2/r ≥c
p

π(r −1)
  r−1

2e
 r−1

2 2/r
≥cr. This immediately yields the claimed bound. The
proof is complete.

The methodological strategy is in place; an estimator will be constructed to estimate ˜γ2(r) for a
well-chosen value of r to balance the estimator’s variance with the bias from Proposition 2.

6


---Page Break---
2.3
Methodology

We return to the compound setting of (1); there is now no “ground truth" prior G. However, a
recurring theme of compound decision theory [48, 47] is that it mimics the empirical Bayes theory as
if the prior were the empirical distribution of the means. Note the quadratic functional of interest
||µ||2/n is precisely equal to the second moment of the empirical distribution of the means.

To describe our estimation strategy, some preliminary development is necessary. For even r, set mr
to be the rth moment of the distribution ( 1

n
Pn
i=1 δµi) ∗N(0, σ2). For odd r, set mr = 0. Observe
m1, m2, ... are the moments of the symmetrized distribution
  1

n
Pn
i=1( 1

2δµi + 1

2δ−µi)

∗N(0, σ2).
Thus, we are able to conceptually place ourselves in the context of Section 2.2 by making the choice
of “prior”3 G = 1

n
Pn
i=1( 1

2δµi + 1

2δ−µi). Let us adopt the notation of Section 2.2, namely let κr
denote the rth cumulant of G ∗N(0, σ2), and let γr and νr denote the rth cumulant and moment
respectively of G. As noted in Section 2.2, we have κr = γr for r ̸= 2. Furthermore, note the
quadratic functional of interest ||µ||2/n is exactly γ2.

The strategy is to estimate ˜γ2(r) given by (9) for a carefully chosen value of r. In pursuit of this
strategy, estimators for the cumulants (except the second) will now be constructed. Define the moment
estimators

ˆmr :=
 1

n
Pn
i=1 Xr
i
if r is even,
0
otherwise.
(10)

Cumulant estimators are obtained via plugging in to (6),

ˆγr :=

r
X

l=1
(−1)l−1(l −1)! Br,l( ˆm1, ˆm2, ..., ˆmr−l+1)
(11)

for r ̸= 2. To estimate ˜γ2(r), an estimator of the function Mr given by (8) is needed. Define the
(random) function ˆ
Mr : [0, ∞) →R given by

ˆ
Mr(γ) =

r
X

l=1
Br,l(ˆγ1, γ, ˆγ3, ..., ˆγr−l+1).
(12)

For r ≥2 and ε > 0, define the estimator

ˆγ2(r) := sup
n
γ ∈[0, 1] :
 ˆ
Mr(γ)
 ≤1 + ε
o
(13)

where r, ε are tuning parameters to be chosen. The high-level justification behind ˆγ2(r) lies in the
approximation,
Mr(ˆγ2(r)) ≈ˆ
Mr(ˆγ2(r)) ≈Mr(˜γ2(r)).
(14)

Suppose it could be shown ˆ
Mr(γ) concentrates around Mr(γ) uniformly over γ ∈[0, 1]. If so,
then the first approximation in (14) follows. The Bell polynomial structure is quite convenient
as it facilitates a straightforward proof of the desired uniform concentration (see Appendix B.2);
cumbersome empirical process theory is avoided. Since both ˆγ2(r) and ˜γ2(r) are defined in terms
of supremums in (13) and (9), both ˆ
Mr(ˆγ2(r)) and Mr(˜γ2(r)) will be near one, intuitively yielding
the second approximation in (14). Thus |Mr(ˆγ2(r)) −Mr(˜γ2(r))| is small, and since Mr is an
r
2-degree polynomial4, intuition suggests |ˆγ2(r) −˜γ2(r)| can be controlled via an r

2-degree Taylor
expansion. Combining with the approximation error bound provided by Proposition 2 delivers a
bound on |ˆγ2(r) −γ2|, as the following result states.

Proposition 3. Fix ε > 0 and even r. Let E =
n
supγ∈[0,1]
 ˆ
Mr(γ) −Mr(γ)
 ≤ε
o
where ˆ
Mr and

Mr are given by (12) and (8) respectively. On the event E, we have |ˆγ2(r)−γ2| ≤((r/2)!·2ε)2/r+ C

r
where ˆγ2(r) is given by (13) and C > 0 is a universal constant.

3The reader should understand this choice of “prior" as essentially the same as the usual compound decision-
theoretic choice of the empirical distribution of the means. However, it happens to be more convenient to take
the symmetrized version.
4Note that in the Bell polynomial Br,l(γ1, γ, γ3, ..., γr−l+1), it follows by Definition 1 that the power of γ
is given by j2 which must satisfy 2j2 ≤r.

7


---Page Break---
Proof. It follows from the definition (9) of ˜γ2(r) and Proposition 1 that γ2 ≤˜γ2(r). On the event
E, it follows from the definition of ˆγ2(r) and |Mr(˜γ2(r))| ≤1 that ˜γ2(r) ≤ˆγ2(r). Therefore, on
the event E, it follows by Lemmas 1 and 2 that 1 = Mr(˜γ2(r)) ≤Mr(ˆγ2(r)) ≤ˆ
Mr(ˆγ2(r)) + ε ≤
1 + 2ε. This implies 0 ≤Mr(ˆγ2(r)) −Mr(˜γ2(r)) ≤2ε, and so an application of Lemma 3 yields
0 ≤ˆγ2(r) −˜γ2(r) ≤((r/2)! · 2ε)2/r. Combining this bound with the approximation error bound of
Proposition 2 delivers the desired result.

To obtain an estimation error bound, it remains to investigate the stochastic error, i.e. the typical
magnitude of supγ∈[0,1] | ˆ
Mr(γ) −Mr(γ)|, so that the first approximation in (14) is justified. The
cumulant estimators {ˆγl}1≤l≤r, l̸=2 used to define ˆ
Mr(γ) are constructed by plugging in sample
moment estimators for the even moments. Heuristically, only a slow concentration rate can be
expected because the number of moments r being estimated is large.
Proposition 4. Suppose (µ, σ) ∈Θ and c∗, β > 0 are universal constants. If δ ≥
1
logβ n, then there
exist universal constants Cβ,∗, C, C′ ≥1 depending only on β and c∗such that the following holds.

If 2 ≤r ≤
1
Cβ,∗
log n
log log n and σ2 ≥c∗

r , then Pµ,σ
n
supγ∈[0,1]
 ˆ
Mr(γ) −Mr(γ)
 ≤CeC′r log r

√

nδ

o
≥
1 −δ.

With the stochastic error addressed by Proposition 4, the ingredients are in place to furnish a risk
bound for the variance estimator associated to (13). There is the slight technical point that Proposition
4 applies only when σ2 ≳1

r. This condition may be an artifact of the proof, and the conclusion of
Proposition 4 might continue to hold for small σ. However, Proposition 4 turns out to suffice for the
purpose of variance estimation as we can incorporate a truncation step. The variance estimator is
defined as follows. For an even integer r and a real ε > 0, define

ˆσ2 =

 
1
n

n
X

i=1
X2
i −ˆγ2(r)

!

1n
max1≤i≤n Xi>4√log n

r
o
(15)

where ˆγ2(r) is given by (13).
Theorem 1. Let δ =
1
log2 n. Let Cβ,∗, C, and C′ be the universal constants from Proposition 4
corresponding to c∗= 1 and β = 2. There exists a universal constant C∗≥Cβ,∗such that the

following holds. If r is the largest even integer less than or equal to
1
C∗
log n
log log n and ε = CeC′r log r

√

nδ
,
then

sup
(µ,σ)∈Θ
Eµ,σ
 
|ˆσ2 −σ2|2
≲
log log n

log n

2
,

where ˆσ2 is given by (15).

Since ||µ||∞≤1 implies that max1≤i≤n Xi is typically no larger than 1+
p

2σ2 log n, the truncation

choice5 4
q

log n

r
essentially results in using the trivial estimator ˆσ2 = 0 when σ2 ≲1

r. On the other
hand when σ2 ≳1

r, the stochastic error result of Proposition 4 is meaningful and it can be shown
1
n
Pn
i=1 X2
i −ˆγ2(r) estimates σ2 well.

3
Lower bound

The slow convergence rate of the cumulant-based estimator of Section 2 turns out to be the sharp rate.

Theorem 2. There exist universal constants C, c > 0 such that inf ˆσ sup(µ,σ)∈Θ(2) Pµ,σ{|ˆσ2−σ2|2 ≥
C( log log n

log n )2} ≥c.

The lower bound is proved through a moment matching technique employed in Le Cam’s two point
method. The intuition for the construction follows from appreciating the constraint ||µ||∞≤1.
As discussed, the boundedness constraint is imposed to ensure σ2 is identifiable. Though quite

5The choice of 4 is not crucial and could be replaced by some other constant with no change to the conclusion
of Theorem 1.

8


---Page Break---
elementary, it is conceptually useful for our construction to examine why σ2 is unidentifiable if
no constraints are imposed. Specifically, it can be understood from a lower bound perspective by
examining the reduction to the following Bayesian testing problem, H0 : µ = 0 and σ2 = 1 + τ 2
against H1 : µi ∼N(0, τ 2) and σ2 = 1. Observe that under both hypotheses the data are X1, ..., Xn
are i.i.d. draws from N(0, 1 + τ 2). The hypotheses are indistinguishable yet the variances differ
by τ 2 under H0 and H1. Therefore, any variance estimator incurs square loss of at least τ 4; taking
τ →∞establishes infinite estimation risk is inescapable.

Under the boundedness constraint ||µ||∞≤1, the choice of prior µ1, ..., µn
iid
∼N(0, τ 2) is no longer
feasible. One idea is to find a distribution G supported on [−1, 1] such that the marginal data distribu-
tion G∗N(0, 1) is indistinguishable from N(0, τ 2)∗N(0, 1). This can be achieved by constructing G
to share a large number of moments with N(0, τ 2). To elaborate, if the first 2k−1 moments match and
τ < 1, then (see Theorem 3.3.3 of [64]) we have χ2(G∗N(0, 1) || N(0, τ 2)∗N(0, 1)) ≤
16
√2k−1
τ 4k
1−τ 2 .
A well-known sufficient condition (e.g. see [58]) for the desired indistinguishability is that the χ2-
divergence is at most O(1/n). Hence, we seek a G through moment matching.

How many moments can be matched under the constraint that G be supported on [−1, 1]? Since the
odd moments of a centered Gaussian are all zero and taking G symmetric would thus match the odd
moments, only the even moments require consideration. For k even, the kth moment of N(0, τ 2) is
at least, by Stirling’s approximation, cτ kek log k/2 for some small constant c > 0. On the other hand,
the kth moment of G must lie in [−1, 1]. Therefore, a necessary condition for the kth moments to
match is τ kek log k/2 ≲1. This crude reasoning already shows G can only match at most O(1/τ 2)
moments.

The actual construction of G relies on the technology of Gaussian quadrature. With Proposition
9 and Lemma 8, it can be shown that the k-atomic Gaussian quadrature G = Pk
i=1 wiδτzi,
where {zi}k
i=1 are the roots of the kth Hermite polynomial, is supported on the interval
[−
p

τ 2(4k −4),
p

τ 2(4k −4)] and shares the first 2k −1 moments with N(0, τ 2). A sufficient
condition for G to be supported on [−1, 1] is k ≤1/(4τ 2). Therefore, a properly supported G which
matches Ω
 
1/τ 2
moments of N(0, τ 2) can be constructed, validating that our crude reasoning from
earlier is tight. The problem now boils down to choosing τ 2. Since k ≍τ −2, it follows from the
χ2-divergence bound implied by moment matching that choosing τ 2 ≍log log n

log n
yields the desired 1

n
bound on the χ2-divergence and is precisely the desired separation.

4
Noise agnosticism

The cumulant-based variance estimator proposed in this paper heavily relies on the Gaussian character
of the noise, specifically that all cumulants (except the first- and second-order) of a Gaussian are equal
to zero regardless of the Gaussian’s mean and variance parameters. From a minimax perspective,
it turns out it is a fundamental necessity to exploit information about the noise. Concretely, it can
be shown that consistent variance estimation is impossible if nothing beyond subgaussianity (see
Definition 2) is assumed. The impossibility can be seen by appealing to Le Cam’s two-point method
to reduce the problem to a two-point testing problem and using the following simple construction,

H0 : µi
iid
∼Rademacher (1/2) , σ2 = 1, and ξi
iid
∼Rademacher (1/2) ,

H1 : µ = 0, σ2 = 2, and ξi
√

2
iid
∼Rademacher (1/2) ∗Rademacher (1/2) .
It is clear both noise distributions are 2-subgaussian. Furthermore, the data {Xi}n
i=1 are indepen-
dent and identically distributed according to Rademacher (1/2) ∗Rademacher (1/2) under both
hypotheses, and so it is impossible to distinguish H0 and H1. Since the separation between the
choices of σ2 between the two hypotheses is 1, it follows consistent estimation is impossible. This is
formally stated in the following proposition without proof.
Proposition 5. For a > 0, let Ξa := {P : P is a-subgaussian, has mean 0 and variance 1}. Let
Pµ,σ,Pξ denote the joint distribution of the observations from the sequence model, Xi = µi + σξi for
1 ≤i ≤n where ξi ∼Pξ are i.i.d. Then inf ˆσ sup(µ,σ)∈Θ(2),Pξ∈Ξ2 Eµ,σ,Pξ
 
|ˆσ2 −σ2|2
≳1.

In this sense it is necessary to exploit finer information about the noise, and so our cumulant-based
estimator is designed to exploit the noise’s Gaussian character in (1).

9


---Page Break---
Acknowledgments and Disclosure of Funding

This work was supported in part by NSF Grant ECCS-2216912. The author thanks Chao Gao for
helpful discussions.

References

[1] Bayati, M., Erdogdu, M. A., and Montanari, A. (2013). Estimating LASSO Risk and Noise
Level. In Advances in Neural Information Processing Systems.

[2] Brown, L. D. and Greenshtein, E. (2009). Nonparametric empirical Bayes and compound
decision approaches to estimation of a high-dimensional vector of normal means. Ann. Statist.
37(4):1685–1704.

[3] Brown, L. D., Greenshtein, E., and Ritov, Y. (2013). The Poisson compound decision problem
revisited. J. Amer. Statist. Assoc. 108(502):741–749.

[4] Butucea, C. and Matias, C. (2005). Minimax estimation of the noise level and of the deconvolu-
tion density in a semiparametric convolution model. Bernoulli 11(2):309–340.

[5] Cai, T. T. and Jin, J. (2010). Optimal rates of convergence for estimating the null density and
proportion of nonnull effects in large-scale multiple testing. Ann. Statist. 38(1):100–145.

[6] Carpentier, A., Klopp, O., Löffler, M., and Nickl, R. (2018). Adaptive confidence sets for matrix
completion. Bernoulli 24(4A):2429–2460.

[7] Carpentier, A. and Verzelen, N. (2019). Adaptive estimation of the sparsity in the Gaussian
vector model. Ann. Statist. 47(1):93–126.

[8] Castillo, I. and Mismer, R. (2018). Empirical Bayes analysis of spike and slab posterior
distributions. Electron. J. Stat. 12(2):3953–4001.

[9] Chen, J. (2017). Consistency of the MLE under mixture models. Statist. Sci. 32(1):47–63.

[10] Chen, X., De, A., and Servedio, R. A. (2020). Testing noisy linear functions for sparsity. In
STOC ’20—Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing,
pp. 610–623.

[11] Comminges, L., Collier, O., Ndaoud, M., and Tsybakov, A. B. (2021). Adaptive robust
estimation in sparse vector model. Ann. Statist. 49(3):1347–1377.

[12] Deb, N., Saha, S., Guntuboyina, A., and Sen, B. (2022). Two-component mixture model in the
presence of covariates. J. Amer. Statist. Assoc. 117(540):1820–1834.

[13] Dicker, L. H. (2014). Variance estimation in high-dimensional linear models. Biometrika
101(2):269–284.

[14] Dicker, L. H. and Zhao, S. D. (2016). High-dimensional classification via nonparametric
empirical Bayes and maximum likelihood inference. Biometrika 103(1):21–34.

[15] Efron, B. (2008). Microarrays, empirical Bayes and the two-groups model. Statist. Sci. 23(1):1–
22.

[16] Efron, B. (2011). Tweedie’s formula and selection bias. J. Amer. Statist. Assoc. 106(496):1602–
1614.

[17] Efron, B. (2014). Two modeling strategies for empirical Bayes estimation. Statist. Sci. 29(2):285–
301.

[18] Efron, B. (2019). Bayes, oracle Bayes and empirical Bayes. Statist. Sci. 34(2):177–201.

[19] Efron, B. and Morris, C. (1972). Empirical Bayes on vector observations: an extension of
Stein’s method. Biometrika 59:335–347.

10


---Page Break---
[20] Efron, B. and Morris, C. (1972). Limiting the risk of Bayes and empirical Bayes estimators. II.
The empirical Bayes case. J. Amer. Statist. Assoc. 67:130–139.

[21] Efron, B., Tibshirani, R., Storey, J. D., and Tusher, V. (2001). Empirical Bayes analysis of a
microarray experiment. J. Amer. Statist. Assoc. 96(456):1151–1160.

[22] Fan, J., Guo, S., and Hao, N. (2012). Variance estimation using refitted cross-validation in
ultrahigh dimensional regression. J. R. Stat. Soc. Ser. B. Stat. Methodol. 74(1):37–65.

[23] Ghosal, S. and van der Vaart, A. W. (2001). Entropies and rates of convergence for maximum
likelihood and Bayes estimation for mixtures of normal densities. Ann. Statist. 29(5):1233–1263.

[24] Ignatiadis, N., Saha, S., Sun, D. L., and Muralidharan, O. (2023). Empirical Bayes mean
estimation with nonparametric errors via order statistic regression on replicated data. J. Amer.
Statist. Assoc. 118(542):987–999.

[25] Ignatiadis, N. and Sen, B. (2023). Empirical partially Bayes multiple testing and compound χ2
decisions. ArXiv:2303.02887 [math, stat].

[26] Ignatiadis, N. and Wager, S. (2022). Confidence intervals for nonparametric empirical Bayes
analysis. J. Amer. Statist. Assoc. 117(539):1149–1166.

[27] James, W. and Stein, C. (1960). Estimation with quadratic loss. In Proc. 4th Berkeley Sympos.
Math. Statist. and Prob., Vol. I, pp. 361–379. Univ. California Press, Berkeley-Los Angeles,
Calif.

[28] Jana, S., Polyanskiy, Y., and Wu, Y. (2022). Optimal empirical Bayes estimation for the Poisson
model via minimum-distance methods. ArXiv:2209.01328 [math, stat].

[29] Janson, L., Foygel Barber, R., and Candès, E. (2017). EigenPrism: inference for high dimen-
sional signal-to-noise ratios. J. R. Stat. Soc. Ser. B. Stat. Methodol. 79(4):1037–1065.

[30] Jiang, W. (2020). On general maximum likelihood empirical Bayes estimation of heteroscedastic
IID normal means. Electron. J. Stat. 14(1):2272–2297.

[31] Jiang, W. and Zhang, C.-H. (2009). General maximum likelihood empirical Bayes estimation
of normal means. Ann. Statist. 37(4):1647–1684.

[32] Kamath, G. Bounds on the Expectation of the Maximum of Samples from a Gaussian.

[33] Kiefer, J. and Wolfowitz, J. (1956). Consistency of the maximum likelihood estimator in the
presence of infinitely many incidental parameters. Ann. Math. Statist. 27:887–906.

[34] Koenker, R. and Mizera, I. (2014). Convex optimization, shape constraints, compound decisions,
and empirical Bayes rules. J. Amer. Statist. Assoc. 109(506):674–685.

[35] Kong, W. and Valiant, G. (2018). Estimating Learnability in the Sublinear Data Regime. In
Advances in Neural Information Processing Systems.

[36] Kotekal, S. and Gao, C. (2024). Optimal estimation of the null distribution in large-scale
inference. ArXiv:2401.06350 [math, stat].

[37] Li, J., Gupta, S. S., and Liese, F. (2005). Convergence rates of empirical Bayes estimation in
exponential family. J. Statist. Plann. Inference 131(1):101–115.

[38] Liang, T. (2000). On an empirical Bayes test for a normal mean. Ann. Statist. 28(2):648–655.

[39] Lindsay, B. G. (1989). Moment matrices: applications in mixtures. Ann. Statist. 17(2):722–740.

[40] Lu, M. and Stephens, M. (2016). Variance adaptive shrinkage (vash): flexible empirical Bayes
estimation of variances. Bioinformatics 32(22):3428–3434.

[41] Lu, M. and Stephens, M. (2019). Empirical Bayes estimation of normal means, accounting for
uncertainty in estimated standard errors. ArXiv:1901.10679 [stat].

11


---Page Break---
[42] Matias, C. (2002). Semiparametric deconvolution with unknown noise variance. European
Series in Applied and Industrial Mathematics. Probability and Statistics 6:271–292.

[43] Meister, A. (2006). Density estimation with normal measurement error with unknown variance.
Statist. Sinica 16(1):195–211.

[44] Morris, C. N. (1983). Parametric empirical Bayes inference: theory and applications. J. Amer.
Statist. Assoc. 78(381):47–65.

[45] Polyanskiy, Y. and Wu, Y. (2020). Self-regularizing Property of Nonparametric Maximum
Likelihood Estimator in Mixture Models. ArXiv:2008.08244 [math, stat].

[46] Polyanskiy, Y. and Wu, Y. (2021). Sharp regret bounds for empirical Bayes and compound
decision problems. ArXiv:2109.03943 [cs, math, stat].

[47] Robbins, H. (1951). Asymptotically subminimax solutions of compound statistical decision
problems. In Proceedings of the Second Berkeley Symposium on Mathematical Statistics and
Probability, 1950, pp. 131–148.

[48] Robbins, H. (1956). An empirical Bayes approach to statistics. In Proceedings of the Third
Berkeley Symposium on Mathematical Statistics and Probability, 1954–1955, vol. I, pp. 157–
163.

[49] Saha, S. and Guntuboyina, A. (2020). On the nonparametric maximum likelihood estimator
for Gaussian location mixture densities with application to Gaussian denoising. Ann. Statist.
48(2):738–762.

[50] Shen, Y., Gao, C., Witten, D., and Han, F. (2020). Optimal estimation of variance in nonpara-
metric regression with random design. Ann. Statist. 48(6):3589–3618.

[51] Shen, Y. and Wu, Y. (2022). Empirical Bayes estimation: When does g-modeling beat f-
modeling in theory (and in practice)? ArXiv:2211.12692 [math, stat].

[52] Singh, R. S. (1979). Empirical Bayes estimation in Lebesgue-exponential families with rates
near the best possible rate. Ann. Statist. 7(4):890–902.

[53] Sun, T. and Zhang, C.-H. (2012). Scaled sparse linear regression. Biometrika 99(4):879–898.

[54] Sun, W. and Cai, T. T. (2007). Oracle and adaptive compound decision rules for false discovery
rate control. J. Amer. Statist. Assoc. 102(479):901–912.

[55] Sun, W., Reich, B. J., Cai, T. T., Guindani, M., and Schwartzman, A. (2015). False discovery
control in large-scale spatial multiple testing. J. R. Stat. Soc. Ser. B. Stat. Methodol. 77(1):59–83.

[56] Szegö, G. (1939). Orthogonal Polynomials, volume Vol. 23 of American Mathematical Society
Colloquium Publications. American Mathematical Society, New York.

[57] Talagrand, M. (2021). Upper and Lower Bounds for Stochastic Processes: Decomposition
Theorems, volume 60 of Ergebnisse der Mathematik und ihrer Grenzgebiete. 3. Folge / A Series
of Modern Surveys in Mathematics. Springer International Publishing, Cham.

[58] Tsybakov, A. B. (2009). Introduction to Nonparametric Estimation. Springer Series in Statistics.
Springer, New York.

[59] Verzelen, N. (2012). Minimax risks for sparse regressions: ultra-high dimensional phenomenons.
Electron. J. Stat. 6:38–90.

[60] Verzelen, N. and Gassiat, E. (2018). Adaptive estimation of high-dimensional signal-to-noise
ratios. Bernoulli 24(4B):3683–3710.

[61] Wang, L., Brown, L. D., Cai, T. T., and Levine, M. (2008). Effect of mean on variance function
estimation in nonparametric regression. Ann. Statist. 36(2):646–664.

[62] Wang, W. and Stephens, M. (2021). Empirical Bayes matrix factorization. J. Mach. Learn. Res.
22:Paper No. 120, 40.

12


---Page Break---
[63] Wu, Y. and Yang, P. (2020). Optimal estimation of Gaussian mixtures via denoised method of
moments. Ann. Statist. 48(4):1981–2007.

[64] Wu, Y. and Yang, P. (2020). Polynomial methods in statistical inference: theory and practice.
CIT 17(4):402–586.

[65] Xing, Z., Carbonetto, P., and Stephens, M. (2021). Flexible signal denoising via flexible
empirical Bayes shrinkage. J. Mach. Learn. Res. 22:Paper No. 93, 28.

[66] Zhang, C.-H. (1997). Empirical Bayes and compound estimation of normal means. Statist.
Sinica 7:181–193.

[67] Zhang, C.-H. (2003). Compound decision theory and empirical Bayes methods. Ann. Statist.
31:379–390.

[68] Zhang, F. and Gao, C. (2020). Convergence Rates of Empirical Bayes Posterior Distributions:
A Variational Perspective. ArXiv:2009.03969 [math, stat].

13


---Page Break---
Appendices to “Variance estimation in compound decision theory
under boundedness”

The appendices are organized as follows. Appendix A discusses the connection to Gaussian mixture
model results of [63]. The remaining proofs for the results of Section 2.3 are presented in Appendix
B and the proof of Theorem 2 is presented in Appendix C. Appendix D contains auxiliary definitions
and results. Finally, Appendix E describes the notation used in the main text and the appendices.

A
Relationship to Gaussian mixture model

Viewing the marginal distribution X1, ..., Xn
iid
∼G ∗N(0, σ2) as a Gaussian mixture model, Wu and
Yang [63] assume G is a k-atomic distribution with bounded support and, for k ≲
log n
log log n, show that

the variance estimator furnished by Lindsay’s algorithm [39] achieves |ˆσ2 −σ2|2 ≲k2n−1

k with
high probability. A minimax lower bound of order n−1

k is obtained, establishing optimality for fixed
k.

The main focus of Wu and Yang’s article is estimation of G with respect to the Wasserstein 1-distance.
In the case where the variance σ2 is known, they develop a denoised method of moments (DMM)
estimator and prove it achieves the minimax estimation rate. Notably, their results cover the case
where G has a continuous density (see Theorem 5 in [63]); in this case, G is approximated by a finite
mixture with c
log n
log log n atoms (for some small constant c > 0), and the approximation is estimated by
the DMM estimator.

When σ2 is unknown, Wu and Yang study Lindsay’s algorithm (Algorithm 3 in [63]). However,
their article [63] only addresses the case k ≤c
log n
log log n and no analogue of Theorem 5 is offered; no
error bound for the variance estimator is offered either. For continuous G, a natural idea is to mimic
the known variance case by approximating the continuous G by a ˜k-atomic measure and running
Lindsay’s algorithm with this choice ˜k. One may hope the fact that the data actually come from the
continuous G and not the ˜k-atomic approximation does not pose a problem.

Unfortunately, it appears the analysis in [63] of Lindsay’s algorithm strongly relies on the assumption
that the true data-generating distribution G is an atomic measure. To illustrate, let us examine the
proof on page 1997 in [63]. Denote ˆπ = ˆG ∗N(0, ˆσ2) and π = G ∗N(0, σ2) where ˆG, ˆσ2 are the
outputs of Lindsay’s algorithm. Consider the case σ ≤ˆσ, and denote G′ = ˆG ∗N(0, ˆσ2 −σ2).
Following their proof, though Proposition 3 in [63] cannot now be directly invoked since G is
not ˜k-atomic, one may hope a modification of Proposition 3’s proof might be possible. However,
Proposition 3 relies on Lemma 13 (found in the supplementary material of [63]), which we would
like to apply to the measures ˆG ∗N(0, ˆσ2 −σ2) and G. Though Lemma 13 does not require ˆG to be
an atomic distribution, it does require G to be ˜k-atomic and the proof of Lemma 13 makes essential
use of this assumption. It is not clear whether this can be circumvented in a straightforward manner.
It is an interesting open problem to obtain an analogue of Theorem 5 in the case of unknown variance,
and also to establish whether or not Lindsay’s algorithm can achieve the optimal estimation rate.

B
Upper bound

The high-level justification behind the development of the estimation methodology presented in
Section 2.3 is the approximation (14). As discussed, the two key pieces are the handling of the
stochastic error supγ∈[0,1] | ˆ
Mr(γ) −Mr(γ)| and the approximation error |ˆγ2(r) −˜γ2(r)|, the latter
of which is controlled through a Taylor expansion of Mr. Analytic properties of Mr are presented in
Appendix B.1 and the concentration of ˆ
Mr is presented in Appendix B.2

B.1
Analytic properties of Mr

The following results pertain to the function Mr given by (8) and are made with the context of Section
2.2 in force.

14


---Page Break---
Lemma 1. If r is even, then the function Mr given by (8) is strictly increasing on the interval
(γ2, ∞).

Proof. Observe for γ > γ2 that Mr(γ) gives the rth moment of the distribution G ∗N(0, γ −γ2).
In other words, letting µ ∼G and Z ∼N(0, 1), we have

Mr(γ) = E
  
µ + √γ −γ2Z
r
=

r
X

l=0

r
l


E(µr−l)(γ −γ2)l/2E(Zl).

Consider that if r −l is odd, then l must also be odd since r is even, which implies E(Zl) = 0.
Therefore, we can write

Mr(γ) =
X

0≤l≤r,
l even

r
l


E(|µ|r−l)(γ −γ2)l/2E(Zl).

Hence, Mr is strictly increasing in γ on the interval (γ2, ∞) as claimed.

Lemma 2. If r is even, then Mr(˜γ2(r)) = 1 where Mr is given by (8) and ˜γ2(r) is given by (9).

Proof. Consider Mr is a continuous function as it is a polynomial. Observe Mr(γ2) ≤1 by
Proposition 1. Furthermore, consider since Mr(γ) is the rth moment of G ∗N(0, γ −γ2) for
γ > γ2 it follows limγ→∞Mr(γ) = ∞. Furthermore, since Lemma 1 asserts Mr is monotonically
increasing on (γ2, ∞) and since ˜γ2(r) ≥γ2, it follows by continuity that Mr(˜γ2(r)) = 1.

Lemma 3. If x ≥y ≥γ2 and r is even, then x −y ≤((r/2)! (Mr(x) −Mr(y)))2/r where Mr is
given by (8).

Proof. From the proof of Lemma 1 and taking µ ∼G and Z ∼N(0, 1), we have for γ ≥γ2,

Mr(γ) =
X

0≤l≤r,
l even

r
l


E(|µ|r−l)E(Zl)(γ −γ2)l/2.

For 0 ≤k ≤r

2 and for γ ≥γ2, it follows from even l that

M (k)
r
(γ) =
X

0≤l≤r,
l even,
l/2≥k

r
l


E(|µ|r−l)E(Zl)
(l/2)!
(l/2 −k)!(γ −γ2)l/2−k ≥E(|µ|r−2k).

Since Mr(γ) is an r

2-degree polynomial in γ, Taylor expansion along with the above bound yields

Mr(x) −Mr(y) =

r/2
X

k=1
M (k)
r
(y)(x −y)k

k!
≥(x −y)r/2

(r/2)!
.

Here, we have used that every term in the sum is nonnegative. In particular, M (k)
r
(y) ≥0 and
(x −y)k ≥0 since x ≥y. Rearranging gives the desired result.

B.2
Concentration of ˆ
Mr

As the definition of ˆ
Mr in (12) relies on moment estimators ˆmr given by (10) and cumulant estimators
ˆγr given by (11), we will first collect results for these intermediate estimators. Further, recall from
Section 2.3 that mr = 1

n
Pn
i=1 E(Xr
i ) if r is even and mr = 0 otherwise.

Lemma 4. If ||µ||∞≤1, then Var( ˆmr) ≤4r+σ2r(4r)2r

n
.

15


---Page Break---
Proof. The claim is trivially true for odd r, so consider even r.
Write Xi = µi + σZi
where Zi
iid
∼
N(0, 1).
Observe by independence and |µi|
≤
1, we have Var( ˆmr)
=
Var
  1

n
Pn
i=1 Xr
i

=
1
n2
Pn
i=1 Var(Xr
i ) ≤
1
n2
Pn
i=1 E(X2r
i ). By Jensen’s inequality, we have

X2r
i
= 22r   µi

2 + σZi

2
2r ≤22r−1  
µ2r
i + (σZi)2r
. Taking expectation yields

E(X2r
i ) ≤22r−1(µ2r
i + σ2r(2r −1)!!) ≤22r−1  
1 + σ2r(2r −1)2r−1
≤4r + σ2r(4r)2r,

which yields the desired result.

Corollary 1. If ||µ||∞≤1 and u > 0, then Pµ,σ {max1≤l≤r | ˆml −ml| > u} ≤r4r+r(1∨σ2)r(4r)2r

u2n
.

Proof. The result follows by union bound and Chebyshev’s inequality with Lemma 4.

Proposition 6. Suppose (µ, σ) ∈Θ(L) where Θ(L) is given by (4) and where L > 0 is a universal
constant. There exists a large universal constant C > 0 such that the following holds. If u > 0 such
that Cru

1∧σr ∈(0, 1), then on the event {max1≤l≤r | ˆml −ml| ≤u} we have

max
1≤l≤r,
l̸=2
|ˆγl −γl| ≤(1 + Lrr!) · (r!)2 · (2r)r ·
e
Cr
1∧σr ru −1
 .

Proof. The argument borrows heavily from the proof of Lemma A.4 in [10]. For 1 ≤k ≤r with
k ̸= 2, we have by Definition 1,

|ˆγk −γk|

≤

k
X

l=1
(l −1)!|Bk,l( ˆm1, ..., ˆmk−l+1) −Bk,l(m1, ..., mk−l+1)|

≤

k
X

l=1
(l −1)!
X
k!
j1!j2! · · · jk−l+1!·



 ˆm1

1!

j1  ˆm2

2!

j2
· · ·

ˆmk−l+1
(k −l + 1)!

jk−l+1
−
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1

where the sum is taken over all sequences j1, j2, ..., jk−l+1 of nonnegative integers such that j1 +
... + jk−l+1 = l and j1 + 2j2 + 3j3 + ... + (k −l + 1)jk−l+1 = k. By definition of ˆml given by
(10) and the definition of ml given in Section 2.3, we have ˆml = ml = 0 for odd l. Consequently,
we need only examine terms in the sum such that jl = 0 for all odd l, so let us now fix such a term.

For even l, we have by Jensen’s inequality ml = 1

n
Pn
i=1
1
2E(|µi + σZ|l) + 1

2E(| −µi + σZ|l) ≥

E(|σZ|l) where Z ∼N(0, 1). Since l is even, E(|σZ|l) = σlE(Zl) = σl 2l/2Γ( l+1

2 )
√π
≥(c′σ2l)l/2

for some small universal constant c′ > 0 by Stirling’s approximation. Consider (c′σ2l)l/2 ≥
min1≤q≤r(c′σ2q)q/2 ≥C−r(1 ∧σr) since we take C > 0 to be a sufficiently large universal
constant.

Observe on the event {max1≤l≤r | ˆml −ml| ≤u}, we have
 ˆm1

1!

j1  ˆm2

2!

j2
· · ·

ˆmk−l+1
(k −l + 1)!

jk−l+1

≤
m1 + u

1!

j1 m2 + u

2!

j2
· · ·
mk−l+1 + u

(k −l + 1)!

jk−l+1

≤
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
exp

 k−l+1
X

i=1
ji
u
mi

!

.

To obtain the second line we have used that 0 ≤
  ˆmi

i!
ji ≤
  mi+u

i!
ji for all 1 ≤i ≤k −l + 1 since
jl = 0 for all odd l. We have also used the inequality 1 + x ≤ex to obtain the third line. Since
mi ≥C−r(1 ∧σr) for even i as established earlier, it follows by the identity j1 + ... + jk−l+1 = l

16


---Page Break---
that exp
Pk−l+1
i=1
ji u

mi


≤e
Cr
1∧σr lu. Since j1 +2j2 +3j3 +...+(k −l +1)jk−l+1 = k and ji = 0

for all odd i, it follows by Jensen’s inequality (letting Y ∼G ∗N(0, σ2)) that mj1
1 · · · mjk−l+1
k−l+1 =
Qk−l+1
i=1
E(Y i)ji ≤Qk−l+1
i=1
E(Y iji) = Qk−l+1
i=1
E(Y k· iji

k ) ≤Qk−l+1
i=1
E(Y k)
iji

k
= mk. This
inequality yields
 ˆm1

1!

j1  ˆm2

2!

j2
· · ·

ˆmk−l+1
(k −l + 1)!

jk−l+1

≤
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
+ mk

e
Cr
1∧σr lu −1

.
(16)

Let us now prove an analogous lower bound. Again, since mi ≥C−r(1 ∧σr) for even i and
j1 + ... + jk−l+1 = l, we have
 ˆm1

1!

j1  ˆm2

2!

j2
· · ·

ˆmk−l+1
(k −l + 1)!

jk−l+1

=
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1

·

 
1 −m1 −ˆm1

m1

j1
· · ·

1 −mk−l+1 −ˆmk−l+1

mk−l+1

jk−l+1!

≥
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
·

 
1 −u

m1

j1
· · ·

1 −
u
mk−l+1

jk−l+1!

≥
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
·

1 −
Cr

1 ∧σr u
l

≥
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
·

1 −
Cr

1 ∧σr lu


≥
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
−mk
Cr

1 ∧σr lu

≥
m1

1!

j1 m2

2!

j2
· · ·

mk−l+1
(k −l + 1)!

jk−l+1
−mk

e
Cr
1∧σr lu −1

.
(17)

Here, we have used
Cr
1∧σr u ∈(0, 1), jl = 0 for all odd l, and mi ≥C−r(1 ∧σr) for even i to

conclude

1 −mq−ˆmq

mq

jq
≥

1 −
u
mq

jq
≥0 for all q, thus obtaining the third line. We have also

used
Cr
1∧σr u ∈(0, 1) to obtain the third-to-last line. Therefore, it follows from (16) and (17) that

|ˆγk −γk| ≤

k
X

l=1
(l −1)!
X
k!
j1!j2! · · · jk−l+1! · mk
e
Cr
1∧σr lu −1


≤mk|e
Cr
1∧σr ku −1| · (k!)2 · kk

≤2k(1 + Lkk!)|e
Cr
1∧σr ku −1| · (k!)2 · kk.

The last inequality follows from ||µ||∞≤1 and Jensen’s inequality. We have also bounded the
number of admissible sequences j1, ..., jk−l+1 in the sum by kk by elementary counting since the
constraint ji ≤k must always be satisfied. Maximizing over 1 ≤k ≤r with k ̸= 2 yields the desired
result.

Finally, we are able to state a concentration result about ˆ
Mr. In the statement of Proposition 7, note
the concentration occurs on the same event that the cumulant estimators concentrate around the
true cumulants, which in turn occurs, from Proposition 6, on the same event the moment estimators
concentrate.

17


---Page Break---
Proposition 7. Suppose (µ, σ) ∈Θ(L) where Θ(L) is given by (4) and where L > 0 is a universal

constant. On the event

max1≤l≤r,
l̸=2
|ˆγl −γl| ≤u

, we have

sup
γ∈[0,1]

 ˆ
Mr(γ) −Mr(γ)
 ≤(2r)rr3(r!)(rr ∨ur)u,

where ˆ
Mr and Mr are given by (12) and (8) respectively.

Proof. For γ ∈[0, 1], consider by Definition 1 that
 ˆ
Mr(γ) −Mr(γ)


≤

r
X

l=1

X
r!
j1!j2! · · · jr−l+1!

 γ

2!


j2



Y

1≤a≤r−l+1,
a̸=2

 ˆγa

a!

ja
−
Y

1≤a≤r−l+1,
a̸=2

γa

a!

ja



≤

r
X

l=1

X
r!
j1!j2! · · · jr−l+1! |fj(ˆγ) −fj(γ∗)|

where the inner sum is taken over all sequences j = {j1, ..., jr−l+1} of nonnegative integers such
that j1 + ... + jr−l+1 = l and j1 + 2j2 + 3j3 + ... + (r −l + 1)jr−l+1 = r. Here, we have defined
the function fj : Rr−l+1 →R by fj(x) = Q
1≤a≤r−l+1,
a̸=2

  xa

a!
ja and we have used the notation

ˆγ = (ˆγ1, γ, ˆγ3, ..., ˆγr−l+1) and γ∗= (γ1, γ, γ3, ..., γr−l+1). By Taylor expansion and Holder’s
inequality, we have
|fj(ˆγ) −fj(γ∗)| ≤||∇fj(ξ)||1 · ||ˆγ −γ∗||∞
where ξ is some point on the line segment between ˆγ and γ∗. It is immediate that on the event

max1≤l≤r,
l̸=2
|ˆγl −γl| ≤u

we have ||ˆγ −γ∗||∞≤u. Further consider |ξa| ≤|γa| + u ≤2(|γa| ∨

u) ≤2(aa ∨u) by Lemma 9. A straightforward calculation shows

||∇fj(ξ)||1 ≤
X

1≤a≤r−l+1,
a̸=2

1{ja≥1}ja
|ξa|ja−1

(a!)ja
Y

1≤b≤r−l+1,
b̸∈{2,a}


ξb

b!



jb

≤
X

1≤a≤r−l+1,
a̸=2

1{ja≥1}ja
|2(aa ∨u)|ja−1

(a!)ja
Y

1≤b≤r−l+1,
b̸∈{2,a}


2(bb ∨u)

b!



jb

≤2rr2 (rr ∨ur) ,

where we have used 2(aa ∨u) ≥1, ja ≤r, and Pr−l+1
a=1
aja = r. To summarize, we have shown

 ˆ
Mr(γ) −Mr(γ)
 ≤

r
X

l=1

X
r!
j1!j2! · · · jr−l+1! · (2rr2(rr ∨ur))u ≤(2r)rr3(r!)(rr ∨ur)u

for all γ ∈[0, 1]. As in the proof of Proposition 6, we have bounded the number of admissible
sequences j1, ..., jr−l+1 by rr. The proof is complete.

Proof of Proposition 4. Since (µ, σ) ∈Θ and L > 0 is a universal constant, it follows by Corollary
1 that the event

E =

(

max
1≤l≤r | ˆml −ml| ≤δ−1/2 C1eC′
1r log r
√n

)

has Pµ,σ-probability of at least 1 −δ, where C1, C′
1 > 0 are some universal constants. Furthermore,

it is clear for any 2 ≤r ≤
1
Cβ,∗
log n
log log n, we have δ−1/2 C1eC′
1r log r
√n
≲(logβ/2 n)n

C′
1
Cβ,∗−1

2 , and so

18


---Page Break---
Cβ,∗can be taken sufficiently large to ensure δ−1/2 C1eC′
1r log r
√n
decays at some rate that is polynomial

in n. We will now apply Proposition 6 on the event E with u = δ−1/2 C1eC′
1r log r
√n
. To use Proposition

6, it must be checked
˜
Cr
1∧σr u ∈(0, 1) where ˜C > 0 is the universal constant from Proposition 6.
Consider that σ2 ≥c∗

r , and so

˜Cr

1 ∧σr u ≤

 
˜Cr ∨

 
˜C
√

c∗

!r

rr/2
!

u

≤δ−1/2 C1
√n exp

 
C′
1 + 1

2


r

 

log r + log

 
˜C
√

c∗

!

+ log ˜C

!!

≲(logβ/2 n)n

C′
1+1/2+log( ˜
C2/
√

c∗)
Cβ,∗
−1

2 .

Therefore, taking Cβ,∗sufficiently large (depending on c∗and β) implies
˜
Cr
1∧σr u decays at some rate
that is polynomial in n. It then follows from Proposition 6, the fact that r is at most logarithmic in n,
and the inequality |ex −1| ≤ex for x ∈(0, 1),

max
1≤l≤r,
l̸=2
|ˆγl −γl| ≤(1 + Lrr!) · (r!)2 · (2r)r ·
e

˜
Cr
1∧σr ru −1
 ≤C2eC′
2r log ru

where C2, C′
2 > 0 are some universal constants (potentially depending on c∗). Note C2eC′
2r log ru ≲

(logβ/2 n)n

C′
1+C′
2
Cβ,∗−1

2 , and so again we can take Cβ,∗sufficiently large to ensure C2eC′
2r log ru decays
at some rate that is polynomial in n. It then follows from Proposition 7 that

sup
γ∈[0,1]

 ˆ
Mr(γ) −Mr(γ)
 ≤C3eC′
3r log r ·

C2eC′
2r log ru

= δ−1/2 C1C2C3e(C′
1+C′
2+C′
3)r log r
√n
,

where C3, C′
3 > 0 are some universal constants (potentially depending on c∗). The proof is complete.

B.3
Proof of Theorem 1

Proof of Theorem 1. Fix (µ, σ) ∈Θ. Choose

C∗:= Cβ,∗∨2 (4(C′ + 1)) ∨2 · inf

K > 0 :
4(C′ + 1)

K
−1

· 2K < −2

.

Note C∗is a positive universal constant since Cβ,∗, C, and C′ all are positive universal constants.
The reason for this choice of C∗will become clear later on. To bound the estimation risk, first
consider

Eµ,σ
 
|ˆσ2 −σ2|2
≤σ4Pµ,σ

(

max
1≤i≤n Xi ≤4

r

log n

r

)

(18)

+ Eµ,σ






1
n

n
X

i=1
X2
i −ˆγ2(r) −σ2


2

1n
max1≤i≤n Xi>4√log n

r
o



.
(19)

19


---Page Break---
To bound (18), let Zi = Xi −µi and consider by Lemma 10,

σ4Pµ,σ

(

max
1≤i≤n Xi ≤4

r

log n

r

)

≤σ4Pµ,σ

( max
1≤i≤n Zi −Eµ,σ


max
1≤i≤n Zi

 ≥Eµ,σ


max
1≤i≤n Zi


−4

r

log n

r
−||µ||∞

)

≤2σ4 exp






−


Eµ,σ (max1≤i≤n Zi) −4
q

log n

r
−||µ||∞

2

+
2σ2








≤2σ4 exp



−1

2

 √log n
√π log 2 −σ−1
 

4

r

log n

r
+ 1

!!2

+





≲
log log n

log n

2
(20)

where we have used the result of [32] which gives Eµ,σ(max1≤i≤n Zi) ≥
σ
√π log 2
√log n. We have

also used ||µ||∞≤1 and 1

r ≲log log n

log n . To obtain the result of the theorem, it remains to show (19)

has order at most

log log n

log n
2
. To do so, we split the analysis into two cases.

Case 1: Suppose σ2 < 1

r. Consider by Cauchy-Schwarz inequality,

(19) ≤

v
u
u
u
tEµ,σ






1
n

n
X

i=1
X2
i −ˆγ2(r) −σ2


4

· Pµ,σ

(

max
1≤i≤n Xi > 4

r

log n

r

)

.
(21)

Letting Zi = Xi −µi, consider by Lemma 10 we have

Pµ,σ

(

max
1≤i≤n Xi > 4

r

log n

r

)

≤Pµ,σ

( max
1≤i≤n Zi −Eµ,σ


max
1≤i≤n Zi

 > 4

r

log n

r
−||µ||∞−Eµ,σ


max
1≤i≤n Zi

)

≤2 exp



−1

2σ2

 

4

r

log n

r
−||µ||∞−Eµ,σ


max
1≤i≤n Zi

!2

+



.

Consider Eµ,σ (max1≤i≤n Zi) ≤σ√2 log n ≤
q

2 log n

r
and ||µ||∞≤1. Therefore, it follows

that Pµ,σ


max1≤i≤n Xi > 4
q

log n

r


≤2 exp

−log n

rσ2

≤2

n. With this in hand, it follows from

Eµ,σ
 1

n
Pn
i=1 X2
i −ˆγ2(r) −σ24
≲1 that (21) ≲n−1 ≲

log log n

log n
2
as desired.

Case 2: Suppose σ2 ≥1

r. It is clear (19) ≤Eµ,σ
 1

n
Pn
i=1 X2
i −ˆγ2(r) −σ22
. To bound this

expectation, denote the event E =
n
supγ∈[0,1]
 ˆ
Mr(γ) −Mr(γ)
 ≤ε
o
. Since σ2 ≥
1
r, C∗≥

Cβ,∗, and ε =
CeC′r log r

√

nδ
with C, C′ as in Proposition 4 with the choice c∗= 1, it follows by
Proposition 4 that Pµ,σ(E) ≥1 −δ. From the inequality (a + b)2 ≤2a2 + 2b2, it follows
 1

n
Pn
i=1 X2
i −ˆγ2(r) −σ22 ≤2
 1

n
Pn
i=1 X2
i −(γ2 + σ2)
2 + 2|ˆγ2(r) −γ2|2. Since ˆγ2(r), γ2 ∈

20


---Page Break---
[0, 1] and since ε = CeC′r log r

√

nδ
, an application of Proposition 3 yields

2|ˆγ2(r) −γ2|2 ≤2

 
2Ce(C′+1)r log r

√

nδ

!4/r

+ C′′

r2 + 21Ec

where C′′ > 0 is a universal constant. Consider
 
e(C′+1)r log r

√

nδ

!4/r

= exp
1

r (4(C′ + 1)r log r −2 log n + 4 log log n)


≤exp
1

r

4(C′ + 1)

C∗
log n
log log n log
 1

C∗
log n
log log n


−log n


≤exp
4(C′ + 1)

C∗
−1
 log n

r


.

Note C∗≥2 · 4(C′ + 1) ≥4(C′ + 1), so it immediately follows 4(C′+1)

C∗
−1 < 0. Therefore, it
follows from r ≥
1
2C∗
log n
log log n that
 
e(C′+1)r log r

√

nδ

!4/r

≤exp
4(C′ + 1)

C∗
−1

· 2C∗log log n

≤
1
log2 n

where the second inequality follows from the fact that C∗satisfies, by design, the inequality

4(C′+1)

C∗
−1

· 2C∗≤−2. Therefore, we have shown 2

2Ce(C′+1)r log r

√

nδ

4/r
≲
1
log2 n. To summa-
rize, we have shown

Eµ,σ






1
n

n
X

i=1
X2
i −ˆγ2(r) −σ2


2

≲Var

 
1
n

n
X

i=1
X2
i

!

+ Eµ,σ
 
|ˆγ2(r) −γ2|2

≲1

n +
1
log2 n + 1

r2 + Pµ,σ(Ec)

≲1

n +
1
log2 n + 1

r2 + δ

≍
log log n

log n

2

as desired. The proof is complete.

C
Lower bound

Proof of Theorem 2. As seen in many minimax lower bound arguments of the literature, we proceed
by reducing to a two-point testing problem. Without loss of generality we will assume n is larger
than a sufficiently large universal constant. Let 0 < τ 2 ≤1 and we will choose it later. Set
σ2
0 = 1 + τ 2 and σ2
1 = 1. For a distribution π supported on [−1, 1], denote the induced mixture
Pπ,σ =
R
Pµ,σ π⊗n(dµ). It follows by reverse triangle inequality

inf
ˆσ
sup
(µ,σ)∈Θ(2)
Pµ,σ


|ˆσ2 −σ2| ≥τ 2

2



≥1

2 inf
ˆσ


P0,σ0


|ˆσ2 −σ2
0| ≥τ 2

2


+ Pπ,σ1


|ˆσ2 −σ2
1| ≥τ 2

2



≥1

2 inf
ˆσ


P0,σ0


|ˆσ2 −1| ≤τ 2

2


+ Pπ,σ1


|ˆσ2 −1| > τ 2

2



≥1

2 inf
A {P0,σ0(A) + Pπ,σ1 (Ac)}

= 1

2 (1 −dTV(P0,σ0, Pπ,σ1)) ,
(22)

21


---Page Break---
where the infimum in the penultimate line runs over all events A. By the Neyman-Pearson lemma,
1 −dTV(P0,σ0, Pπ,σ1) is the optimal testing risk for the hypothesis testing problem

H0 : X1, ..., Xn
iid
∼N(0, 1 + τ 2),

H1 : X1, ..., Xn
iid
∼π ∗N(0, 1).

Specifically, we have dTV(P0,σ0, Pπ,σ1) ≤
1
2
p

(1 + χ2 (π ∗N(0, 1) || N(0, 1 + τ 2)))n −1 by
Lemma 5. Hence, it suffices to bound the χ2-divergence to furnish a lower bound for (22).

We now construct π and we will pick τ 2 <
1
16 at the end of the proof to obtain the claimed lower
bound. Let D denote the largest even number smaller than or equal to
1
4τ 2 . Note D ≥2 since
τ 2 < 1

8. Let gD = PD
i=1 wiδzi denote the D-point Gaussian quadrature of N(0, 1). Note that the
atoms {zi}D
i=1 are the zeros of the Dth degree Hermite polynomial (Lemma 6). Further note that
this measure is 1-subgaussian and symmetric about zero as D is even (Lemma 7). Take the prior
distribution π = PD
i=1 wiδτzi. In order for π to be a valid choice, it must be verified π is supported
on [−1, 1]. Lemma 8 gives us that |zi| ≤
√

4D −4 since zi is a zero of the Dth degree Hermite
polynomial. Observe |τzi| ≤
p

τ 2(4D −4) < 1 by our choice of D. Hence, π is supported on
[−1, 1] and thus is a valid choice.

We use a moment matching technique (Proposition 8). To do so, we first verify the first 2D −1
moments of π and N(0, τ 2) match.
For any r ∈{1, ..., 2D −1}, observe since gD is the
D-point Gaussian quadrature of N(0, 1), it follows EY ∼N(0,τ 2)(Y r) = τ rEZ∼N(0,1)(Zr) =
τ rEZ∼gD(Zr) = EY ∼π(Y r). Note both N(0, τ 2) and π are τ-subGaussian. Proposition 8, along
with D ≥
1
4τ 2 −2 ≥2 and τ 2 <
1
16, thus implies

χ2(π ∗N(0, 1) || N(0, τ 2) ∗N(0, 1)) ≤
16τ 4D
√

2D −1 (1 −τ 2) ≤
162

15
√

3 exp

−1

4τ 2 log
 1

τ 2


.

Select τ 2 = log log n

16 log n and observe

χ2(π ∗N(0, 1) || N(0, τ 2) ∗N(0, 1)) ≤
162

15
√

3 exp

 

−4 log n

 

1 −log
 
16−1 log log n


log log n

!!

≤1

n

since n is larger than a sufficiently large universal constant. Therefore, dTV(P0,σ0, Pπ,σ1) ≤

1
2

q 
1 + 1

n
n −1 ≤1

2
√e −1, which, when plugged into (22), yields the lower bound

inf
ˆσ
sup
(µ,σ)∈Θ(2)
Pµ,σ


|ˆσ2 −σ2| ≥log log n

32 log n


≥1

2


1 −1

2

√

e −1

.

The proof is complete.

D
Auxiliary definitions and results

Definition 2. A probability distribution P is said to be a-subgaussian for a > 0 if for X ∼P we
have E (exp (λ(X −E(X)))) ≤e
λ2a2

2
for all λ ∈R.
Lemma 5 (χ2 tensorization [58]). If P = Nn
i=1 Pi and Q = Nn
i=1 Qi are product measures, then
χ2(P || Q) = Qn
i=1(1 + χ2(Pi || Qi)) −1.
Proposition 8 (Theorem 3.3.3 [64]). Suppose ν and ν′ are two symmetric probability distributions
that are ε-subgaussian for ε < 1. If the first D moments of ν and ν′ are equal, then χ2(ν ∗
N(0, 1) || ν′ ∗N(0, 1)) ≤
16
√

D
ε2D+2

1−ε2 .

Proposition 9 (Gaussian quadrature [64]). Suppose µ is a probability measure supported on E ⊂R.
If k ≥1, there exists a k-atomic distribution µk = Pk
i=1 wiδxi supported on E such that for any
polynomial p of degree at most 2k −1 we have
R
p(x) dµ(x) = Pk
i=1 wip(xi).
Lemma 6 (Remark 2.7.2 [64]). If k ≥1, then the k-point Gaussian quadrature of N(0, 1) has its
atoms at the roots of the degree k Hermite polynomial Hek.

22


---Page Break---
Lemma 7 (Lemma 2.7.3 [64]). Suppose k ≥1 and gk is the k-point Gaussian quadrature of N(0, 1).
For j ≥2k, we have EX∼gk(Xj) ≤EZ∼N(0,1)(Zj) when j is even and EX∼gk(Xj) = 0 otherwise.
In particular, gk is 1-subgaussian.

Lemma 8. If k ≥1, then the zeros of the kth Hermite polynomial lie in [−
√

4k −4,
√

4k −4].

Proof. From (6.2.18) on page 120 of [56] the zeros of the kth degree physicist’s Hermite polynomial
Hk(x) = (−1)nex2 dk

dxk e−x2 lie in the interval
h
−

√

2(k−1)
√k+2 ,

√

2(k−1)

k+2
i
. Since k −1 ≤k + 2, it imme-

diately follows that we have the inclusion
h
−

√

2(k−1)
√k+2 ,

√

2(k−1)
√k+2

i
⊂

−
√

2k −2,
√

2k −2

. Since
we have the following correspondence between physicist’s and probabilist’s Hermite polynomials
Hk(x) = 2k/2Hek(
√

2x), it follows the zeros of Hek lie in [−
√

4k −4,
√

4k −4]. The proof is
complete.

Lemma 9. The rth cumulant γr of a distribution G supported on [−1, 1] satisfies |γr| ≤rr.

Proof. Let νk denote the kth moment of G and note |νk| ≤1.
From (6) it follows γr =
Pr
l=1(−1)l−1(l −1)!Br,l(ν1, ν2, ..., νr−l+1). Since the coefficients of the Bell polynomial Br,l
are all positive and |νk| ≤1, it follows |γr| ≤Pr
l=1(l −1)!Br,l(1, 1, ..., 1) = Pr
l=1(l −1)!
r
l
	

where
r
l
	
is a Stirling number of the second kind. Consider (l −1)!
r
l
	
=
1

l · l!
r
l
	
≤lr−1.
Therefore, |γr| ≤Pr
l=1 lr−1 ≤rr as desired.

Lemma 10 (Lemma 2.10.6 [57]). If Z ∼N(0, σ2In), then

P
 max
1≤i≤n Zi −E

max
1≤i≤n Zi

 ≥u

≤2 exp

−u2

2σ2



for u ≥0.

E
Notation

For a, b ∈R the notation a ≲b denotes the existence of a universal constant c > 0 such that a ≤cb.
The notation a ≳b is used to denote b ≲a. Additionally a ≍b denotes a ≲b and a ≳b. The
symbol := is frequently used when defining a quantity or object. Furthermore, we frequently use
a∨b := max(a, b) and a∧b := min(a, b). We generically use the notation 1A to denote the indicator
function for an event A. For two probability measures P and Q on a measurable space (X, A),
the total variation distance is defined as dTV(P, Q) := supA∈A |P(A) −Q(A)|. If P is absolutely

continuous with respect to Q, then the χ2-divergence is defined as χ2(P||Q) :=
R

X

dP
dQ −1
2
dQ.

We will frequently use the same notation for two probability densities p and q. For sequences {ak}∞
k=1
and {bk}∞
k=1, the notation ak = o(bk) denotes limk→∞
ak
bk = 0 and the notation ak = ω(bk) is used
to denote bk = o(ak). For a point x ∈R, the symbol δx denotes the probability measure which places
full probability mass at the point x. The symbol ∗denotes convolution and the same symbol will
be used in the context of the convolution of probability measures as well as functions. Throughout,
iterated logarithms will be used (e.g. expressions like log log n). Without explicitly stating so, we
will take such an expression to be equal to some universal constant if otherwise it would be less than
one. For example, log log n should be understood to be equal to a universal constant when n < ee.

23


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract claims a minimax rate for variance estimation under the assump-
tion of bounded means, which is the setting addressed in the paper.

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

Justification: The paper discusses how the proposed variance estimator relies on the Gaussian
character of the noise.

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

24


---Page Break---
Answer: [Yes]

Justification: Some results are proved directly in the main text, and others are stated
rigorously with proofs deferred (and cross-referenced properly) to the technical appendices.

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

Justification: The paper does not include experiments.

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

25


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: The paper does not include experiments.
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
Justification: The paper does not include experiments.
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
Justification: The paper does not include experiments.
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

26


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
Answer: [NA]
Justification: The paper does not include experiments.
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
Justification: The research in this paper is purely theoretical and, to the best of the author’s
knowledge, poses no harmful societal or individual impact.
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
Justification: There is no societal impact of this work as it is a purely theoretical result about
variance estimation.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

27


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

Justification: The paper does not pose risks for misuse.

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

28


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

29


---Page Break---
