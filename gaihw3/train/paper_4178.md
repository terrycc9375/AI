Information-theoretic Limits of Online Classification
with Noisy Labels

Changlong Wu
Ananth Grama
Wojciech Szpankowski
CSoI, Purdue University
wuchangl@hawaii.edu, {ayg,szpan}@purdue.edu

Abstract

We study online classification with general hypothesis classes where the true labels
are determined by some function within the class, but are corrupted by unknown
stochastic noise, and the features are generated adversarially. Predictions are made
using observed noisy labels and noiseless features, while the performance is mea-
sured via minimax risk when comparing against true labels. The noisy mechanism
is modeled via a general noisy kernel that specifies, for any individual data point, a
set of distributions from which the actual noisy label distribution is chosen. We
show that minimax risk is tightly characterized (up to a logarithmic factor of the hy-
pothesis class size) by the Hellinger gap of the noisy label distributions induced by
the kernel, independent of other properties such as the means and variances of the
noise. Our main technique is based on a novel reduction to an online comparison
scheme of two-hypotheses, along with a new conditional version of Le Cam-Birgé
testing suitable for online settings. Our work provides the first comprehensive
characterization for noisy online classification with guarantees that apply to the
ground truth while addressing general noisy observations.

1
Introduction

Learning from noisy data is a fundamental problem in many machine learning applications. Noise
can originate from various sources, including low-precision measurements of physical quantities,
communication errors, or noise intentionally injected by methods such as differential privacy. In such
cases, one typically learns by training on noisy (or observed) data while aiming to build a model that
performs well on the true (or latent) data. This paper focuses on online learning [20] from noisy
labels, where one receives noiseless, adversarially generated features and corresponding noisy labels
sequentially, and predicts the true labels as the data arrive.

Online learning has been primarily studied in the agnostic setting [1, 19, 7], where one receives the
labels in their plain (noise-free) form and the prediction risk is evaluated on the observed labels.
It is typically assumed that both the features and observed labels are generated adversarially, and
prediction quality is measured via the notion of regret, which compares the actual cumulative risk
incurred by the predictor with the minimal cumulative risk incurred by the best expert in a hypothesis
class. While this approach is mathematically appealing, it does not adequately characterize online
learning scenarios when our goal is to achieve good performance with respect to grand truth data that
may be different from the observed (noisy) ones.

This paper considers an online learning scenario that differs from classical agnostic online learning
in two aspects: (i) we assume that the noisy labels are derived from a (semi-) stochastic mechanism
rather than from pure adversarial selections; (ii) our prediction risk is evaluated on the true labels,
not noisy observations. To better motivate the study of such a scenario, we consider the following
example first introduced by Ben-David et al. [1]:

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Example 1. Let H ⊂{0, 1}X be a finite hypothesis class. Consider the following online learning
game between Nature/Adversary and Learner that is played over a time horizon T. At the start,
Nature fixes a ground truth classifier h ∈H. At each time step t ≤T, Nature adversarially selects
feature xt ∈X and reveals it to the learner. The learner makes a prediction ˆyt based on prior features
xt = {x1, · · · , xt} and noisy labels ˜yt−1 = {˜y1, · · · , ˜yt−1}. Nature then selects a (unknown) noise
parameter ηt ∈[0, η] for some given η (known to learner), and generates 1:
˜yt = Bernoulli(ηt) ⊕yt,
where ⊕denotes binary addition and yt = h(xt) is the true label. It was shown in [1, Thm 15] that
there exists a predictor ˆyT such that:

sup
h∈H,xT ∈X T E

" T
X

t=1
1{ˆyt ̸= h(xt)}

#

≤
log |H|

1 −2
p

η(1 −η)
.
(1)

Note that the risk in (1) is significant, as the error introduced by noise to the true labels increases
linearly as ηT, yet the risk remains independent of the time horizon T. This mirrors the fast rates
known in the PAC learning literature when benign noise is present. Despite its foundational nature,
the understanding of this phenomenon beyond simple Massart’s noise has been largely unexplored.

This paper introduces a novel online learning framework for modeling general noisy mechanisms.
In particular, it encompasses (1) as a very specific instance and provides a clear and comprehensive
characterization of the underlying paradigm. Formally, let Y be the set of true (latent) labels and ˜Y
be the set of noisy (observed) labels, which we assume are finite and of size N, M, respectively. Let
X be the feature space. We model the noisy mechanism by a noisy kernel:

K : X × Y →2D( ˜
Y),
(2)

where D( ˜Y) is the set of all distributions over ˜Y. That is, the kernel K maps each pair (x, y) to a
subset Qx
y := K(x, y) ⊂D( ˜Y) of distributions over ˜Y. Observe that the noisy kernel provides a
compact way of modeling noisy label distributions without explicitly referring to the noise. This is
more convenient for our discussion, as ultimately the statistical information is solely determined by
the noisy label distributions.

For any given H ⊂YX and kernel K, we consider the following robust (noisy) online classification
scenario: Nature first selects h ∈H; at each time step t, Nature chooses (adversarially) xt ∈X and
reveals it to the learner; the learner then makes a prediction ˆyt, based on the features xt and noisy
labels ˜yt−1; an adversary then selects a distribution ˜pt ∈Qxt
h(xt), samples ˜yt ∼˜pt and reveals ˜yt to
the learner. Let Φ and Ψ be the strategies of the learner and Nature/adversary, respectively. The goal
of the learner is to minimize the following expected minimax risk:

˜rT (H, K) = inf
Φ sup
Ψ
E

" T
X

t=1
1{h(xt) ̸= ˆyt}

#

,
(3)

where ˆyt = Φ(xt, ˜yt−1). Note that the adversarial selection of distribution ˜pt from the kernel set
Qxt
h(xt) provides more flexibility for modeling scenarios when the noisy label distribution changes
even with the same true label, such as the Massart’s noise in Example 1. We refer to Section 2 for a
more complete specification of our setting.

1.1
Main Contributions

Our main contributions in this paper establish the fundamental limits of minimax risk in (3) by
providing nearly matching lower and upper bounds across a wide range of hypothesis classes H
and noisy kernels K. Observe that, to allow for non-trivial prediction rules, the induced noisy
label distributions must be statistically distinguishable for distinct true labels. To formalize this
intuition, we define, for any noisy kernel K and feature x ∈X, the Hellinger gap as γH(x) =
infy̸=y′∈Y infp∈Qxy,q∈Qx
y′{H2(p, q)}, where H2(p, q) = PM
m=1(
p

p[m] −
p

q[m])2 is the squared
Hellinger distance. That is, γH(x) measures the minimal squared Hellinger distance of the induced
noisy label distributions over all distinct true labels.

Our main result (see also Theorem 2) can be summarized as follows:

1This is also known as Massart’s noise in the literature.

2


---Page Break---
Theorem 1. Let H ⊂YX be any finite class, and K be any noisy kernel such that infx∈X γH(x) ≥γH
for some γH > 0, and Qx
y ⊂D( ˜Y) is closed and convex for all x, y. Then:

˜rT (H, K) ≤O
log2 |H|

γH


.

Moreover, for any K ∈N and any kernel K with at least log K features x ∈X for which γH(x) ≤γH,

there exists a class H of size K that satisfies: ˜rT (H, K) ≥Ω

log |H|

γH


.

Theorem 1 shows that the Hellinger gap is the right characterization for the minimax risk upto at
most a logarithmic factor. Moreover, the risk bound depends solely on the gap parameter γH and
log |H|, independent of time horizon T, the size of Y and ˜Y, and the properties of noise such as
means and variances. For the bounded Bernoulli noise in Example 1, the set Qx
y corresponds to
Bernoulli distributions with parameters in [0, η] if y = 0 and in [1 −η, 1] if y = 1, leading to the
Hellinger gap γH = 1 −2
p

η(1 −η). This matches the dependency on η in Example 1 2. However,
our result holds for any noisy kernel. For instance, if we shift Qx
0 to Bernoulli distribution with
parameter 0 and Qx
1 with parameters in [1 −2η, 1], then γH = 1 −√2η = Θ(1 −2η). This is tighter
than the dependency on η in Example 1 (for η →1

2), since 1 −2
p

η(1 −η) = Θ((1 −2η)2).

Our main proof technique for establishing Theorem 1 is based on a novel (black box) reduction to an
online comparison scheme of two-hypotheses in H, as demonstrated in Theorem 3. This allows us
to reduce the noisy online classification problem to a hypothesis testing problem, which effectively
decouples the adversarial property of the features from the stochastic property of the noisy labels.
However, due to the adversarial selection of the noisy label distributions, the classical hypothesis
testing techniques does not apply. To resolve this issue, we establish in Theorem 4, a generalization
of the Le Cam-Birgé Test with varying conditional marginals for handling pairwise testing via the
Hellinger gap, which is a result of independent interest.

Tight dependency on log |H|.
Although the lower and upper bounds in Theorem 1 differ by a
log |H| factor, this is compensated by the fact that we are dealing with the most general classes
and kernels. This can be tightened for various special cases. Indeed, for a class H with binary
true labels and arbitrary noisy labels, we demonstrate in Theorem 5 that the minimax risk is upper
bounded by 16 log |H|

γL
, where γL is the L2-gap that substitutes the Hellinger distance with L2-distance
in Theorem 1. This is proved via a novel reduction to online conditional distribution estimation under
L2-distance. Moreover, we demonstrate in Appendix G (Theorem 6) that the (optimal) O( log |H|

γH
)
upper bound holds if |Qx
y| = 1 for all x, y, i.e., the noisy label distribution is determined by data.

1.2
Related Work

Online learning with noisy data was discussed in [6], which specifically considers generalized
linear functions with zero-mean and bounded variance noises. Our work differs in that we focus
on classification instead of regression. Moreover, our noisy model does not require that the noise
be zero-mean. To the best of our knowledge, [1] is the only work that has specifically considered
the classification task, but this was limited to bounded Bernoulli noise. From a technical standpoint,
analogous ideas of pairwise comparison have been considered in differential privacy literature, such
as in [11], but only in batch settings. The reduction to online conditional probability estimation
was also explored in [10] within the context of online decision making. However, a distinguishing
feature of our work is that our conditional probability estimation problem is necessarily misspecified,
as our noisy label distributions are selected adversarially and are unknown a priori to the learner.
Our problem setup is further related to differentially private conditional distribution learning, as in
[26], and robust hypothesis testing, discussed in [17, Chapter 16]. Online conditional probability
estimation has been widely studied, see [18, 3, 2, 4, 25, 24]. Conditional density estimation in the
batch setting has also been extensively studied, see [12] for KL-divergence with misspecification
and [9] for L2 loss. Learning from noisy labels in the batch case was discussed in [16] (see also the
references therein) by leveraging suitably defined proxy losses. There has been a long line of research
on online prediction with adversarial observable labels in the agnostic formulation, see [5, 1, 19, 7].

2To the best of our knowledge, this Hellinger interpretation is not known in literature; the proof in [1] is
based on induction without explaining on how the factor 1 −2
p

η(1 −η) is obtained.

3


---Page Break---
2
Notation and Preliminaries

Let X be a set of features (or instances), Y be a set of labels, and ˜Y be a set of noisy observations. We
assume throughout the paper that |Y| = N and | ˜Y| = M for some integers N, M ≥2. We denote
D( ˜Y) as the set of all probability distributions over ˜Y.

Let H ⊂YX be a hypotheses class and K be a noisy kernel in (2). We consider the following
robust online classification scenario: (1) Nature first selects some h ∈H; (2) At time t, Nature
adversarially selects xt ∈X; (3) Learner predicts ˆyt ∈Y, based on (noisy) history observed thus far
(i.e., xt, ˜yt−1); (4) An adversary then selects ˜pt ∈Qxt
h(xt), and generates a noisy sample ˜yt ∼˜pt.

The goal of the learner is to minimize the cumulative error: PT
t=1 1{h(xt) ̸= ˆyt}.

Note that the cumulative error is a random variable that depends on all the randomness associated
with the game. To remove the dependency on such randomness and to assess the fundamental limits
of the prediction quality, we consider the following two measures 3:

Definition 1. Let H ⊂YX be a set of hypotheses and K : X × Y →2D( ˜
Y) be a noisy kernel. We
denote by Φ the (possibly randomized) strategies of the learner. The expected minimax risk is:

˜rT (H, K) = inf
Φ
sup
h∈H,xT ∈X T QT
KEˆyT

" T
X

t=1
1{h(xt) ̸= ˆyt}

#

,
(4)

where QT
K ≡
sup
˜p1∈Qx1
h(x1)
E˜y1∼˜p1 · · ·
sup
˜pT ∈Q
xT
h(xT )
E˜yT ∼˜pT , and ˆyt ∼Φ(xt, ˜yt−1).

By skolemization [19], the operator QT
K is equivalent to sup˜p E˜yT ∼˜p, where ˜p runs over all (joint)
distributions over ˜YT such that ∀t ∈[T], ∀˜yt−1 ∈˜Yt−1 the conditional marginal ˜p˜yt|˜yt−1 ∈Qxt
h(xt).

Definition 2. Let H, K, and Φ be as in Definition 1. For any δ > 0, the high probability minimax
risk at confidence δ is the minimum quantity Bδ(H, K) ≥0 such that there exists a predictor Φ
satisfying:

sup
h∈H,xT ∈X T ,˜p
Pr˜yT ∼˜p,ˆyT

" T
X

t=1
1{h(xt) ̸= ˆyt} ≥Bδ(H, K)

#

≤δ,
(5)

where ˜p is selected as in the discussion above and ˆyt ∼Φ(xt, ˜yt−1).

Note that the kernel map K is generally known to the learner when constructing the predictor Φ.
However, the induced kernel sets Qxt
h(xt) are not, since they depend on the unknown ground truth
classifier h and adversarially generated features xT . In certain cases, such as Theorem 3 and
Example 4, the kernel map K is also not required to be known.

For any x ∈X and y ∈Y, we denote by Qx
y the set induced by a kernel. We can assume, w.l.o.g.,
that the Qx
ys are convex and closed sets, since the adversary can select an arbitrary distribution from
Qx
ys at each time step, including randomized strategies that effectively sample from a mixture (i.e.,
convex combination) of distributions in Qx
ys.

One must introduce some constraints on the kernel K in order to obtain meaningful results. To do so,
we introduce the following well-separation condition:

Definition 3. Let L : D( ˜Y)2 →R≥0 be any divergence, we say a kernel K is well-separated w.r.t. L at

scale γ > 0, if ∀x ∈X, ∀y, y′ ∈Y with y ̸= y′ we have L(Qx
y, Qx
y′)
def
= infp∈Qx
y,q∈Qx
y′ L(p, q) ≥γ.

Example 2. For any y ∈Y, we specify a canonical distribution py ∈D( ˜Y). A natural noisy kernel
would be to define Qx
y = {p ∈D( ˜Y) : TV(p, py) ≤ϵ}, where TV denotes total variation. In this
case, the kernel is well-separated with the gap γ under total variation if miny̸=y′∈Y TV(py, py′) ≥
γ + 2ϵ. In particular, this subsumes Example 1 if, for y ∈{0, 1}, we define py as the distribution that
assigns probability 1 to y, and take ϵ = η, where the TV-gap equals γ = 1 −2η.

3We assume here the selection of ˜pT and xT are oblivious to the learner’s action for simplicity. This is
equivalent to the adaptive case if the learner’s internal randomness are independent among different time steps
by a standard argument from [5, Lemma 4.1], see also Appendix H.

4


---Page Break---
3
Main Results

We begin by stating our main result of this paper.

Theorem 2. Let H ⊂YX be a finite class of size K, and K be a kernel that is well-separated at
scale γH w.r.t. squared Hellinger divergence (Definition 3). Then, the high probability minimax risk
(Definition 2) with confidence δ > 0 is upper bounded by:

Bδ(H, K) ≤8 log(4K/δ) log K

γH

+ log(2/δ).
(6)

Moreover, for any kernel K such that there exist at least L distinct features x ∈X 4 for which
infy̸=y′∈Y H2(Qx
y, Qx
y′) ≤γH, one can find a class H of size K such that:

˜rT (H, K) ≥Ω
min{L, log K}

γH


.

Observe that, the upper bound holds with high probability and the risk is independent of the time
horizon (i.e., the so-called fast rates known in the PAC-learning literature). Moreover, the bound is in-
dependent of the size of Y and ˜Y. A simple integration argument yields the expected risk upper bound
˜r(H, K) ≤O

log2 K

γH


, which matches the lower bound upto only a log K factor. This demonstrates
that, the Hellinger gap of the induced noisy label distributions is the right characterization for the
minimax risk. Moreover, the Hellinger distance can be transformed from other f-divergences (such
total variation) without depending on the size of ˜Y [17, Chapter 7.6].

Example 3. Let K be the kernel in Example 2. Let λ = miny̸=y′∈Y TV(py, py′). Hence, the kernel
is well-separated with TV-gap λ −2ϵ. Since H2(p, q) ≥TV(p, q)2 [17, Eq. 7.22], the Hellinger
gap is lower bounded by (λ −2ϵ)2. Invoking Theorem 2, we have for any hypothesis class H, the

following risk upper bound holds: Bδ(H, K) ≤O

log |H| log(|H|/δ)

(λ−2ϵ)2

.

The rest of this section is devoted to establishing Theorem 2. Our main proof technique is based on a
novel reduction to pairwise testing of two hypotheses as developed in Section 3.1, along with explicit
testing rules in Section 3.2 based on a novel conditional version of Le Cam-Birgé testing.

3.1
Reduction to Pairwise Comparison: a Generic Approach

We first introduce the following key technical concept. Recall that our robust online classification
problem is completely determined by the tuple (H, K).

Definition 4. A problem (H, K) is said to be pairwise testable with confidence δ > 0 and error
bound C(δ) ≥0 if, for any pair hi, hj ∈H, the sub-problem ({hi, hj}, K) admits a predictor (i.e.,
pairwise tester) Φi,j that achieves cumulative risk ≤C(δ) w.p. ≥1 −δ (see Definition 2).

Clearly, any prediction rule for (H, K) serves as a pairwise testing rule for all the sub-problems
({hi, hj}, K) with hi, hj ∈H. Perhaps surprisingly, we will show in this section that any pairwise
testing rules for the sub-problems can also be converted into a prediction rule for (H, K), incurring
only an additional logarithmic factor on the risk bounds.

To this end, suppose that the tuple (H, K) is pairwise testable and the class H = {h1, · · · , hK} is
finite with size K. Let Φi,j be the testing rule (will be constructed in Section 3.2) for hi, hj with error
bound C(δ) and confidence δ > 0. Let xT , ˜yT be any realization of problem (H, K). We define, for
any hi ∈H and t ∈[T], a surrogate loss vector:

∀j ∈[K], vi
t[j] = 1{Φi,j(xt, ˜yt−1) ̸= hi(xt)}.
(7)

That is, the loss vi
t[j] = 1 if and only if the test Φi,j(xt, ˜yt−1) differs from hi(xt). Given access to
testers Φi,js, our prediction rule for (H, K) is then presented in Algorithm 1.

4This is a very mild assumption. For instance, if the kernel is independent of the features (such as Example 1),
we have L = |X|. The lower bound gives Ω( log K

γH ) as long as |X| ≥log K.

5


---Page Break---
Algorithm 1: Predictor via Pairwise Hypothesis Testing
Input: Class H = {h1, · · · , hK}, pairwise testers Φi,j for i, j ∈[K] and error bound C
Set S1 = {1, · · · , K};
for t = 1, · · · , T do

Receive xt;
Sampling index ˆkt from St uniformly and make prediction: ˆyt = hˆkt(xt);
Receive noisy label ˜yt;
Set St+1 = ∅;
for i ∈St do

Compute li
t = maxj∈[K]
Pt
r=1 vi
r[j], where vi
t[j] is defined in (7);
if li
t ≤C then

Update St+1 = St+1 ∪{i};

Theorem 3. Let H ⊂YX be any hypothesis class of size K and K be any noisy kernel. If (H, K)
is pairwise testable with error bound C(δ) as in Definition 4, then for any δ > 0, the predictor in
Algorithm 1 with C = C(δ/(2K)) achieves the high probability minimax risk (Definition 2):

Bδ(H, K) ≤2(1 + 2C(δ/(2K)) log K) + log(2/δ).
(8)

Sketch of Proof. At a high level, our goal is to identify the ground truth classifier hk∗using the
testing results of Φi,js. Note that pairwise testability implies, w.p. ≥1 −δ, the errors made by
tester Φk,k∗on hk∗is upper bounded by C(δ/2K) for all k ∈[K] simultaneously. However, for any
other pair i, j ̸= k∗, the tester Φi,j does not provide any guarantees, since the samples used to test
hi, hj originate from hk∗and is not realizable for Φi,j. The key technical challenge is to extract the
testing results for Φk,k∗from the other irrelevant tests (i.e., Φi,j with k∗̸∈{i, j}), even when the
k∗is unknown. This is resolved by our definition of li
t in Algorithm 1, which computes for each i
the maximum testing loss over all of its competitors. This ensures that, for the ground truth k∗, the
loss lk∗
t
≤C(δ/2K). While for any other i ̸= k∗, we have li
t ≥Pt
r=1 vi
r[k∗] ≥Pt
r=1 1{hi(xr) ̸=
hk∗(xr)} −C(δ/2K). Therefore, any hypothesis hi for which li
t > C(δ/2K) cannot be the ground
truth. Algorithm 1 then maintains an index set St that eliminates all hi for which li
t > C(δ/2K),
and makes prediction ˆyt = hˆkt(xt) with ˆkt sampling uniformly from St.

To derive the risk bound, we use a potential-based analysis that relates the size of Sts with the
prediction error 1{hk∗(xt) ̸= ˆyt}. The intuition behind the analysis is that if E[1{hk∗(xt) ̸= ˆyt}]
is large, then there will be many elements i ∈St for which hi(xt) ̸= hk∗(xt), and thus the loss li
t
will (potentially) increase. Since Algorithm 1 constructs St+1 by eliminating all i ∈St for which
li
t > C(δ/2K), one can therefore bound the prediction error by the change in the size of Sts. The
key technical challenge here is to control the hypotheses that differ from k∗but for which the tester
Φk,k∗errs, which is resolved by carefully defining a potential function. The claimed upper bound
then follows by a similar argument as [14, Thm 2]. See Appendix B for complete proof.

Note that, the reduction of Theorem 3 is general and does not rely on specific properties of the
kernel K (such as the well-separation condition). It provides a black box reduction that converts any
pairwise testing rule for two-hypotheses to a general online classification rule that introduces only a
logarithmic factor on the risk bounds. This effectively decouples the adversarial property of features
from the stochastic property of the noisy labels.

To understand how Theorem 3 operates, we consider the following example:
Example 4. Let H ⊂{0, 1}X , and K be the bounded Bernoulli noise kernel with parameter η in
Example 1. For any hi, hj ∈H, we construct the following testing rule. We may assume, w.l.o.g.,
that hi(x) ̸= hj(x) for all x ∈X, since any x for which hi(x) = hj(x) do not affect our testing.
Moreover, by relabeling, we can assume that hi(x) = 0 and hj(x) = 1 for all x ∈X. At time step t,
after observing the noisy labels ˜yt−1, we compute ˆµt =
1
t−1
Pt−1
r=1 ˜yr. If ˆµt ≥1

2, the tester predicts
ˆyt = 1; else, it predicts ˆyt = 0. By Azuma’s inequality, the probability of making an error at step t is
upper bounded by e−(1−2η)2(t−1)/2. Thus, for any n ≤T, the probability of making any errors after

6


---Page Break---
step n is upper bounded by P∞
t=n e−(1−2η)2(t−1)/2 ≤e−(1−2η)2n/2

(1−2η)2
. Taking n = 2 log(1/δ(1−2η)2)

(1−2η)2
one can upper bound the probability by δ. Therefore, the tuple (H, K) is pairwise testable with
C(δ) ≤2 log(1/δ(1−2η)2)

(1−2η)2
. Invoking Theorem 3, we have:

Bδ(H, K) ≤O
log |H| log(|H|/δ(1 −2η)2)

(1 −2η)2


.
(9)

Note that the risk bound in (9) recovers the risk in Example 1 up to a logarithmic factor, though it
employs a completely different approach (cf. [1]). Moreover, Example 4 provides the key advantage
that the risk holds with high probability and at a fast rate, which is known to be non-trivial for
cumulative errors (see, e.g., [22, 21]). To our knowledge, it remains unclear whether the approach
proposed in [1] admits a high probability guarantee.

3.2
Proof of Theorem 2: the conditional Le Cam-Birgé Testing

As demonstrated in Section 3.1, the risk of noisy online classification can be reduced to the pairwise
testing of two hypotheses. However, we still need to construct the explicit pairwise testing rules. This
section is devoted to providing a generic testing rule for general kernels.

Let h0 and h1 be any two hypotheses. We may assume, w.l.o.g., that h0(x) ̸= h1(x) for all x ∈X,
since the features for which h0 and h1 agree do not affect the testing. We now provide a more
compact characterization of the kernel without explicitly referring to true labels. Let xT be any
realization of features. For any i ∈{0, 1}, t ∈[T], and kernel K, we write Qxt
i
:= K(xt, hi(xt)).

We define QJ
0 and QJ
1 as the sets of all (joint) distributions over ˜YJ induced by the kernel upto time
step J for h0, h1, respectively. Equivalently, for i ∈{0, 1}, we have p ∈QJ
i if and only if for all
t ∈[J] and ˜yt−1 ∈˜Yt−1, the conditional marginal p˜yt|˜yt−1 ∈Qxt
i .

The pairwise testing of h0, h1 at time step J + 1 is then equivalent to the (robust) hypothesis testing
w.r.t. sets QJ
0 and QJ
1 . This is typically resolved using Le Cam-Birgé testing [17, Chapter 32.2] if the
distributions are of product form. However, this does not hold for our purpose, since the distributions
in QJ
i can have highly correlated marginals. Our main result for addressing this issue is a conditional
version of Le Cam-Birgé testing, as stated in Theorem 4 below. To the best of our knowledge, this
conditional version is novel.
Theorem 4 (conditional Le Cam-Birgé Testing). Let QJ
0 and QJ
1 be the classes induced by a kernel
upto time J as defined above. For any t ≤J, we denote γt = H2(Qxt
0 , Qxt
1 ) and assume that Qxt
i
is
convex for all i ∈{0, 1}. Then, there exists a testing rule ψ : ˜YJ →{0, 1} such that

sup
p∈QJ
0 ,q∈QJ
1


Pr˜yJ∼p[ψ(˜yJ) ̸= 0] + Pr˜yJ∼q[ψ(˜yJ) ̸= 1]
	
≤2

J
Y

t=1
(1 −γt/2) ≤2e−1

2
PJ
t=1 γt.

Sketch of Proof. The proof requires a suitable application of the minimax theorem by expressing
the testing error as a linear function and arguing that the QJ
i s are convex. The error bound is then
controlled by a careful application of the chain-rule of Rényi divergence. See Appendix C.

Theorem 4 immediately implies the following cumulative risk bound:
Proposition 1. Let h0, h1 be any hypotheses, xT be any realization of features and QT
i , Qxt
i
be
defined as above with γt = H2(Qxt
0 , Qxt
1 ). Then, there exists a tester ˆyT such that for all δ > 0,
i ∈{0, 1} and ˜p ∈QT
i , w.p. ≥1 −δ over ˜yT ∼˜p, we have:

T
X

t=1
1{hi(xt) ̸= ˆyt} ≤arg min
n

(

n ∈N :

n
X

t=1
γt ≥2 log(2/δ)

)

.

Proof. Let n∗be the minimal number satisfying the RHS. If t ≤n∗(this can be checked at each time
step t using only xt and K), we predict arbitrarily. If t ≥n∗+ 1, we use the tester ψ in Theorem 4
with J = n∗to produce an index ˆi ∈{0, 1} and make the prediction hˆi(xt) for all following time
steps. That is, we only use the tester at step n∗+ 1 and reuse the same testing result for all following

7


---Page Break---
time steps. By Theorem 4, the probability of making errors after step n∗+ 1 is upper bounded by δ.
Therefore, the cumulative risk is upper bounded by n∗with probability ≥1 −δ.

Proof of Theorem 2. Let h0, h1 ∈H be any two-hypotheses. For any time step t such that h0(xt) ̸=
h1(xt), we have, by the well-separation condition, that the gap γt ≥γH in Proposition 1. Consider
the following testing rule: for any time step t such that h1(xt) = h2(xt), we predict the agreed
label; else, we predict the same way as in Proposition 1. Clearly, we only make errors for the second
case. Invoking Proposition 1 with γt = γH for all t ∈[T], we have n∗≤2 log(2/δ)

γH
. Therefore, the

tuple (H, K) is pairwise testable with C(δ) = 2 log(2/δ)

γH
. The upper bound on classification risk then
follows by Theorem 3. The lower bound follows by Le Cam’s two point method and constructing a
hard hypothesis class using an epoch approach. We refer to Appendix D for the complete details.

Remark 1. Note that our techniques can be easily extended to infinite classes using the covering
techniques from [1, 22]. Moreover, by applying Proposition 1, our results can be extended to
scenarios where the gap parameters γt are not uniformly bounded, such as in the case of Tsybakov-
type noise [8], which would lead to risk bounds that scale sublinearly with T, in contrast to the
constant risk in Theorem 2. We leave the details and extensions for a longer manuscript [23].

4
Tighter Bounds for Binary Labels via L2 Gap

We have demonstrated in Theorem 2 that the minimax risk is tightly characterized by the Hellinger
gap induced by the kernel. However, the dependency on log |H| remains sub-optimal. We show in
this section a tight dependency on log |H| for classes with binary true labels via the L2 gap.
Theorem 5. Let H ⊂{0, 1}X be any finite binary valued class, K be any noisy kernel that is
well-separated at scale γL w.r.t. the L2-distance 5 (Definition 3). Then, the expected minimax risk, as
in Definition 1, is upper bounded by: ˜rT (H, K) ≤16 log |H|

γL
.

We begin with the following simple geometry fact that is crucial to our proof.

Lemma 1. Let Q ⊂D( ˜Y) be a convex and closed set, p be a point outside of Q with γ
def
=
infq∈Q L2(p, q). Denote by q∗∈Q the (unique) point that attains L2(p, q∗) = γ. Then for any
q ∈Q, we have L2(q, p) −L2(q, q∗) ≥L2(p, q∗) = γ.

Proof. By the hyperplane separation theorem, the hyperplane perpendicular to line segment p −q∗
at q∗separates Q and p. Therefore, the degree θ of angle formed by p −q∗−q is greater than π/2.
By the law of cosines, L2(q, p) ≥L2(q, q∗) + L2(q∗, p) = L2(q, q∗) + γ.

Our key idea of proving Theorem 5 is to reduce the robust (noisy) online classification problem to a
suitable conditional distribution estimation problem, as discussed next.

Online conditional distribution estimation.
Let F ⊂D( ˜Y)X be a class of functions mapping
X to distributions in D( ˜Y). Online conditional distribution estimation is a game between Nature
and an estimator that follows the following protocol: (1) at each times step t, Nature selects some
xt ∈X and reveals it to the estimator; (2) the estimator then makes an estimation ˆpt ∈D( ˜Y), based
on xt, ˜yt−1; (3) Nature then selects some ˜pt ∈D( ˜Y), samples ˜yt ∼˜pt and reveals ˜yt to the estimator.
The goal is to find a (deterministic) estimator Φ that minimizes the regret:

RegT (F, Φ) =
sup
f∈F,xT ∈X T QT
" T
X

t=1
L(˜pt, ˆpt) −L(˜pt, f(xt))

#

,
(10)

where ˆpt = Φ(xt, ˜yt−1), QT is the operator specified in Definition 1 by setting Qx
y := D( ˜Y) for all
x, y, and L is any divergence. We emphasize that the distributions ˜pT are not necessarily realizable
by f and are selected completely arbitrarily. This contrasts with the well-specified cases employed
in [10, 4], and is the key that enables us to handle the unknown noisy label distributions.

We now establish the following key technical lemma, see Appendix E for proof.

5Recall that L2(p, q) = ||p −q||2
2
def
= PM
m=1(p[m] −q[m])2.

8


---Page Break---
Lemma 2. Let F ⊂D( ˜Y)X be a finite distribution-valued function class. Then, for the L2 divergence,
there exists an estimator Φ, i.e., the Exponential Weight Average (EWA) algorithm, such that

RegT (F, Φ) ≤4 log |F|.

Moreover, estimation ˆpt is a convex combination of {f(xt) : f ∈F}.

Proof Sketch of Theorem 5. We provide the high level ideas and refer to Appendix F for complete
details. We define the following distribution-valued function class F using hypothesis class H and
noisy kernel K. For any x ∈X, we denote by Qx
0 and Qx
1 the sets of noisy label distributions
corresponding to labels 0 and 1, respectively. Since the kernel K is well-separated at scale γL under
L2 divergence, we have, by the hyperplane separation theorem, that there must be qx
0 ∈Qx
0 and
qx
1 ∈Qx
1 such that L2(qx
0 , qx
1 ) = L2(Qx
0, Qx
1) ≥γL. We now define for any h ∈H the function fh
such that ∀x ∈X, fh(x) = qx
h(x). Let F = {fh : h ∈H} and Φ be the estimator from Lemma 2
with class F and L2 divergence (using xT , ˜yT from the original noisy classification game). Our
classification rule is defined as ˆyt = arg miny{L2(qxt
y , ˆpt) : y ∈{0, 1}}. That is, we predict the
label y so that qxt
y is closer to ˆpt under L2 divergence, where ˆpt = Φ(xt, ˜yt−1).

Let h∗∈H be the underlying true classification function. We have by Lemma 2 that

sup
xT ∈X T QT
K

" T
X

t=1
L2(˜pt, ˆpt) −L2(˜pt, fh∗(xt))

#

≤4 log |F| ≤4 log |H|,
(11)

where QT
K is the operator in Definition 1. Now, our key technical goal is to show that L2(˜pt, ˆpt) −
L2(˜pt, fh∗(xt)) ≥L2(ˆpt, fh∗(xt)) ≥γL

4 1{ˆyt ̸= h∗(xt)} via Lemma 1 and a geometric argument,
as illustrated in the figure below:

Qxt
h∗(xt)
Qxt
1−h∗(xt)

ˆpt

qxt
1−h∗(xt)
fh∗(xt)

˜pt

≥√γL/2

The expected minimax risk bound PT
t=1 1{ˆyt ̸= h∗(xt)} ≤16 log |H|

γL
then follows from (11).

Although both our proofs and those provided in [1] are based on the EWA algorithm, the analysis
and resulting algorithms are fundamentally different. For instance, in [1], the learning rate of EWA
depends on the parameter η, while we set it to 1/4 (see Appendix E). More importantly, our proof
applies to any noisy kernel that satisfies the well-separation condition (including cases where | ˜Y| > 2),
which benefits from our geometric interpretation of the kernels. Interestingly, for the specific setting
investigated in [1] (i.e., Example 1), our result yields the same order up to a constant factor, since
1 −2
p

η(1 −η) = Θ((1 −2η)2) for η ∈[0, 1

2). In general, we have 4γL ≤γH ≤√MγL.

5
Discussion

In this paper, we provide nearly matching lower and upper bounds for online classification with
noisy labels via the Hellinger gap of the induced noisy label distributions. Our approach works for
a wide range of hypothesis classes and noisy mechanisms. We expect our results to have a wide
range of applications, such as online learning under (local) differential privacy constraints and online
denoising tasks involving data derived from (noisy) physical measurements (such as learning from
quantum data [15]). The main open problem remaining is to close the logarithmic gap in Theorem 2
for general kernels. While our work primarily focuses on the information-theoretically achievable
minimax risks, we believe that finding computationally efficient predictors (including oracle-efficient
methods as in [14]) would also be of significant interest.

9


---Page Break---
Acknowledgements

This work was partially supported by the NSF Center for Science of Information (CSoI) Grant
CCF-0939370, and also by NSF Grants CCF-2006440, CCF-2007238, and CCF-2211423.

References

[1] Shai Ben-David, Dávid Pál, and Shai Shalev-Shwartz. Agnostic online learning. In Conference
on Learning Theory, volume 3, 2009.

[2] Alankrita Bhatt and Young-Han Kim. Sequential prediction under log-loss with side information.
In Algorithmic Learning Theory, pages 340–344. PMLR, 2021.

[3] Blair Bilodeau, Dylan Foster, and Daniel Roy. Tight bounds on minimax regret under logarithmic
loss via self-concordance. In International Conference on Machine Learning, pages 919–929.
PMLR, 2020.

[4] Blair Bilodeau, Dylan J Foster, and Daniel M Roy. Minimax rates for conditional density
estimation via empirical entropy. arXiv preprint arXiv:2109.10461, 2021.

[5] N. Cesa-Bianchi and G. Lugosi. Prediction, Learning and Games. Cambridge University Press,
2006.

[6] Nicolo Cesa-Bianchi, Shai Shalev-Shwartz, and Ohad Shamir. Online learning of noisy data.
IEEE Transactions on Information Theory, 57(12):7907–7931, 2011.

[7] Amit Daniely, Sivan Sabato, Shai Ben-David, and Shai Shalev-Shwartz. Multiclass learnability
and the erm principle. J. Mach. Learn. Res., 16(1):2377–2404, 2015.

[8] Ilias Diakonikolas, Daniel M Kane, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis.
Efficiently learning halfspaces with tsybakov noise. In Proceedings of the 53rd Annual ACM
SIGACT Symposium on Theory of Computing, pages 88–101, 2021.

[9] Sam Efromovich. Conditional density estimation in a regression setting. The Annals of Statistics,
35:2504–2535, 2007.

[10] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity
of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.

[11] Sivakanth Gopi, Gautam Kamath, Janardhan Kulkarni, Aleksandar Nikolov, Zhiwei Steven Wu,
and Huanyu Zhang. Locally private hypothesis selection. In Conference on Learning Theory,
pages 1785–1816. PMLR, 2020.

[12] Peter D Grünwald and Nishant A Mehta. Fast rates for general unbounded loss functions: from
erm to generalized bayes. The Journal of Machine Learning Research, 21(1):2040–2119, 2020.

[13] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in
Optimization, 2(3-4):157–325, 2016.

[14] Sham Kakade and Adam T Kalai. From batch to transductive online learning. Advances in
Neural Information Processing Systems, 18, 2005.

[15] Abram Magner and Arun Padakandla. Fat shattering, joint measurability, and pac learnability
of povm hypothesis classes. arXiv preprint arXiv:2308.12304, 2023.

[16] Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning
with noisy labels. Advances in neural information processing systems, 26, 2013.

[17] Yury Polyanskiy and Yihong Wu. Information Theory: From Coding to Learning. Cambridge
University Press, 2022.

[18] Alexander Rakhlin and Karthik Sridharan. Sequential probability assignment with binary
alphabets and large classes of experts. arXiv preprint arXiv:1501.07340, 2015.

10


---Page Break---
[19] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Online learning: Random averages,
combinatorial parameters, and learnability. In Advances in Neural Information Processing
Systems, 2010.

[20] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to
algorithms. Cambridge university press, 2014.

[21] Dirk van der Hoeven, Nikita Zhivotovskiy, and Nicolò Cesa-Bianchi. High-probability risk
bounds via sequential predictors. arXiv preprint arXiv:2308.07588, 2023.

[22] Changlong Wu, Ananth Grama, and Wojciech Szpankowski. Online learning in dynamically
changing environments. In Conference on Learning Theory, pages 325–358. PMLR 195, 2023.

[23] Changlong Wu, Ananth Grama, and Wojciech Szpankowski. Robust online classification: From
estimation to denoising. arXiv preprint arXiv:2309.01698, 2023.

[24] Changlong Wu, Mohsen Heidari, Ananth Grama, and Wojciech Szpankowski. Expected worst
case regret via stochastic sequential covering. arXiv preprint arXiv:2209.04417, 2022.

[25] Changlong Wu, Mohsen Heidari, Ananth Grama, and Wojciech Szpankowski. Precise regret
bounds for log-loss via a truncated bayesian algorithm. In Advances in Neural Information
Processing Systems, volume 35, pages 26903–26914, 2022.

[26] Changlong Wu, Yifan Wang, Ananth Grama, and Wojciech Szpankowski. Learning functional
distributions with private labels. In International Conference on Machine Learning (ICML),
volume 202 of PMLR, pages 37728–37744. PMLR, 23–29 Jul 2023.

[27] Tong Zhang. Mathematical analysis of machine learning algorithms. Cambridge University
Press, 2023.

11


---Page Break---
A
Martingale Concentration Inequalities

In this appendix, we present some standard concentration results for martingales, which will be useful
for deriving high probability guarantees. We refer to [27, Chapter 13.1] for the proofs.

Lemma 3 (Azuma’s Inequality). Let X1, · · · , XT be an arbitrary random process adaptive to some
filtration {Ft}t≤T such that |Xt| ≤M for all t ≤T. Let Yt = E[Xt | Ft−1] be the conditional
expected random variable of Xt. Then for all δ > 0, we have

Pr

" T
X

t=1
Yt <

T
X

t=1
Xt + M
p

(T/2) log(1/δ)

#

≥1 −δ,

and

Pr

" T
X

t=1
Yt >

T
X

t=1
Xt −M
p

(T/2) log(1/δ)

#

≥1 −δ.

The following lemma provides a tighter concentration when Xt ≥0, which can be viewed as an
Martingale version of the multiplicative Chernoff bound.

Lemma 4 ([27, Theorem 13.5]). Let X1, · · · , XT be an arbitrary random process adaptive to some
filtration {Ft}t≤T such that 0 ≤Xt ≤M for all t ≤T. Let Yt = E[Xt | Ft−1] be the conditional
expected random variable of Xt. Then for all δ > 0 we have

Pr

" T
X

t=1
Yt < 2

T
X

t=1
Xt + 2M log(1/δ)

#

≥1 −δ,

and

Pr

" T
X

t=1
Yt > 1

2

T
X

t=1
Xt −(M/2) log(1/δ)

#

≥1 −δ.

Proof. Applying [27, Thm 13.5] with ξt = Xt/M and λ = 1 in the theorem.

B
Proof of Theorem 3

Let hk∗∈H be the underlying true classification function and xT be the realization of features.
We take C = C(δ/2K) in Algorithm 1. By definition of pairwise testability and union bound, we
have w.p. ≥1 −δ/2 over the randomness of ˜yT and the internal randomness of Φk,k∗s that for all
k ∈[K],
T
X

t=1
1{hk∗(xt) ̸= Φk,k∗(xt, ˜yt−1)} ≤C(δ/(2K)).
(12)

Note that for any other {i, j} ̸∋k∗, equation (12) may not hold for predictor Φi,j. However, our
following argument relies only on the guarantees for predictors Φk,k∗, which effectively makes our
pairwise testing realizable.

We now condition on the event defined in (12). Let vk
t with k ∈[K] and t ∈[T] be the surrogate
loss vector, as defined in (7). We observe the following key properties

1. We have for all t ∈[T] that

max
j∈[K]

t
X

r=1
vk∗
r [j] ≤C(δ/(2K));
(13)

2. For any k ̸= k∗, we have for all t ∈[T]:

max
j∈[K]

t
X

r=1
vk
r[j] ≥

t
X

r=1
1{hk(xr) ̸= hk∗(xt)} −C(δ/(2K)).
(14)

12


---Page Break---
The first property is straightforward by the definition of vk
t and (12). The second property holds since
the lower bound is attained when j = k∗.

We now analyze the performance of Algorithm 1. By property (13), we know that k∗∈St for all
t ∈[T], i.e., |St| ≥1. Let Nt = |St|. We define for all t ∈[T] the potential:

Et =
X

k∈St
max

(

0, 2C(δ/(2K)) −

t
X

r=1
1{hk(xr) ̸= hk∗(xr)}

)

.

Clearly, we have Et ≤2C(δ/(2K))Nt. Let Dt = |{k ∈St : hk(xt) ̸= hk∗(xt)}|. We have:

Dt ≤Nt −Nt+1 + Et −Et+1,
(15)

since for any k ∈St such that hk(xt) ̸= hk∗(xt), either k is removed from St+1 (which contributes
at most Nt −Nt+1) or its contribution to Et+1 is decreased by 1 when compared to Et (this is
because by our construction of Algorithm 1 and property (14) once the contributions of k to Et equals
0 it must be excluded from St+1). We have, by definition of ˆyt, that:

E [1{hk∗(xt) ̸= ˆyt}] = Dt

|St| ≤Nt −Nt+1 + Et −Et+1

Nt
.
(16)

By a standard argument [14, Thm 2], we have:

T
X

t=1

Nt −Nt+1

Nt
≤

T
X

t=1

 1

Nt
+
1
Nt −1 + · · · +
1
Nt+1 + 1



≤

K
X

k=1

1
k ≤log K.

Moreover, we observe that

T
X

t=1

Et −Et+1

Nt

(a)
≤2C(δ/(2K))N1 −E2

N1
+

T
X

t=2

Et −Et+1

Nt

(b)
≤2C(δ/(2K))(N1 −N2)

N1

+ 2C(δ/(2K))N2 −E3

N2
+

T
X

t=3

Et −Et+1

Nt

(c)
≤2C(δ/(2K))

T
X

t=1

Nt −Nt+1

Nt

≤2C(δ/(2K)) log K,

where (a) and (b) follow by Et ≤2C(δ/(2K))Nt and Nt ≥Nt+1; (c) follows by repeating the
same argument for another T −1 steps.

Therefore, we conclude

E

" T
X

t=1
1{hk∗(xt) ̸= ˆyt}

#

≤(1 + 2C(δ/(2K))) log K,

where the randomness is on the selection of ˆkt ∼St. Since our selection of ˆkt are independent
(conditioning on St) for different t, and the indicator is bounded by 1 and non-negative, we can invoke
Lemma 4 (second part) to obtain a high probability guarantee of confidence δ/2 by introducing an
extra log(2/δ) additive term. The theorem now follows by a union bound with the event (12).

C
Proof of Theorem 4

We start with an application of the minimax theorem to hypothesis testing 6.

6This result was mentioned in [17, Chapter 32.2], without providing a proof.

13


---Page Break---
Lemma 5. Let P0 and P1 be two sets of distributions over a finite domain Ω. If P0 and P1 are
convex under L1 distance (i.e., total variation), then

min
ϕ : Ω→[0,1]
sup
p0∈P0,p1∈P1
{Eω∼p0[1 −ϕ(ω)] + Eω∼p1[ϕ(ω)]} = 1 −
inf
p0∈P0,p1∈P1 ||p0 −p1||TV.

Moreover, if ϕ∗is the function attains minimal, then the tester ψ∗(ω) = 1{ϕ∗(ω) < 0.5} achieves

sup
p0∈P0,p1∈P1
{Prω∼p0[ψ∗(ω) ̸= 0] + Prω∼p1[ψ∗(ω) ̸= 1]} ≤2(1 −
inf
p0∈P0,p1∈P1 ||p0 −p1||TV).

Proof. Observe that the function ϕ can be viewed as a vector in [0, 1]Ω. Moreover, the distribu-
tions over Ωcan be viewed as vectors in [0, 1]Ωas well. Therefore, we have Eω∼p0[1 −ϕ(ω)] +
Eω∼p1[ϕ(ω)] = ⟨p0, 1 −ϕ⟩+ ⟨p1, ϕ⟩, which is a linear function w.r.t. both (p0, p1) and ϕ. Since the
both P0 × P1 and [0, 1]Ωare convex and [0, 1]Ωis compact, we can invoke the minimax theorem [5,
Thm 7.1] to obtain

min
ϕ : Ω→[0,1]
sup
p0∈P0,p1∈P1
{Eω∼p0[1 −ϕ(ω)] + Eω∼p1[ϕ(ω)]}

=
sup
p0∈P0,p1∈P1
min
ϕ : Ω→[0,1] {Eω∼p0[1 −ϕ(ω)] + Eω∼p1[ϕ(ω)]}

=
sup
p0∈P0,p1∈P1
{1 −||p0 −p1||TV},

where the last equality follows by Le Cam’s two point lemma [17, Theorem 7.7]. Let ϕ∗be the
function attains minimal and ψ∗(ω) = 1{ϕ∗(ω) < 0.5}. We have 1{ψ∗(ω) ̸= i} ≤2(1−i−ϕ∗(ω))
for all i ∈{0, 1}. To see this, for i = 0, we have ψ∗(ω) ̸= 0 only if ϕ∗(ω) < 0.5, thus 1 −ϕ∗(ω) ≥
0.5 (the case for i = 1 follows similarly). Therefore, we have for all p0 ∈P0, p1 ∈P1
Prω∼p0[ψ∗(ω) ̸= 0] + Prω∼p1[ψ∗(ω) ̸= 1] ≤2(Eω∼p0[1 −ϕ∗(ω)] + Eω∼p1[ϕ∗(ω)]).

This completes the proof.

We now establish the following key property, which demonstrates that the distribution classes
constructed in Theorem 4 satisfy the condition of Lemma 5.
Lemma 6. Let QJ
0 and QJ
1 be the sets in Theorem 4. Then QJ
0 and QJ
1 are convex under L1 distance.

Proof. Let p1, p2 ∈QJ
i for i ∈{0, 1} and λ ∈[0, 1]. We need to show that p = λp1 + (1 −λ)p2 ∈
QJ
i as well. For any given t ∈[T], we have

p(˜yt | ˜yt−1) =
λp1(˜yt) + (1 −λ)p2(˜yt)
λp1(˜yt−1) + (1 −λ)p2(˜yt−1)

= λp1(˜yt−1)

p(˜yt−1) p1(˜yt | ˜yt−1) + (1 −λ)p2(˜yt−1)

p(˜yt−1) p2(˜yt | ˜yt−1) ∈Qxt
i

where the last inclusion follows by convexity of Qxt
i
as assumed in Theorem 4. Therefore, we have
p ∈QJ
i by definition of QJ
i .

Now, our main technical part is to bound the total variation TV(QJ
0 , QJ
1 ). The primary challenge
comes from controlling the dependencies of conditional marginals of the distributions. To this end,
we introduce the concept of Renyi divergence. Let p1, p2 be two distributions over the same finite
domain Ω, the α-Renyi divergence is defined as

Dα(p1, p2) =
1
α −1 log Eω∼p2

p1(ω)

p2(ω)

α
.

If p, q are distributions over domain Ω1 × Ω2 and r is a distribution over Ω1, then the conditional
α-Renyi divergence is defined as

Dα(p, q | r) =
1
α −1 log Eω1∼r

" X

ω2∈Ω2
p(ω2 | ω1)αq(ω2 | ω1)1−α
#

.

The following property about Renyi divergence is well known [17, Chapter 7.12]:

14


---Page Break---
Lemma 7. Let p, q be two distributions over Ω1 × Ω2 and p(1) and q(1) be the restrictions of p, q on
Ω1, respectively. Then the following chain rule holds

Dα(p, q) = Dα(p(1), q(1)) + Dα(p, q | r),

where r(ω1) = p(1)(ω1)αq(1)(ω1)1−αe−(α−1)Dα(p(1),q(1)) is a distribution over Ω1.

The following key result bounds the Renyi divergence between QJ
0 and QJ
1 :

Proposition 2. Let QJ
0 and QJ
1 be the sets in Theorem 4. If infp∈Qxt
0 ,q∈Qxt
1 Dα(p, q) ≥ηt holds for
all t ≤J. Then

inf
p∈QJ
0 ,q∈QJ
1
Dα(p, q) ≥

J
X

t=1
ηt.

Proof. We prove by induction on J. The base case for J = 1 is trivial, since Q1
0 = Qx1
0
and
Q1
1 = Qx1
1 . We now prove the induction step with J ≥2. For any pair p ∈QJ
0 and q ∈QJ
1 , we have
by Lemma 7 that Dα(p, q) = Dα(p(1), q(1)) + Dα(p, q | r), where p(1), q(1) are restrictions of p, q
on ˜yJ−1 and r is a distribution over ˜YJ−1. By definition of α-Renyi divergence, we have

Dα(p, q | r) ≥inf
˜yJ−1
1
α −1 log
X

˜yJ∈˜
Y
p(˜yJ | ˜yJ−1)αq(˜yJ | ˜yJ−1)1−α

= inf
˜yJ−1 Dα(p˜yJ|˜yJ−1, q˜yJ|˜yJ−1)

(a)
≥
inf
p∈Q
xJ
0
,q∈Q
xJ
1
Dα(p, q)
(b)
≥ηJ,

where (a) follows since p˜yJ|˜yJ−1 ∈QxJ
0
and q˜yJ|˜yJ−1 ∈QxJ
1
by the definition of QJ
0 and QJ
1 ; (b)
follows by assumption. The result then follows by induction hypothesis Dα(p(1), q(1)) ≥PJ−1
t=1 ηt,
since p(1) ∈QJ−1
0
and q(1) ∈QJ−1
1
.

The following result converts the Renyi divergence based bounds to that with Hellinger divergence.

Proposition 3. Let QJ
0 and QJ
1 be the sets in Theorem 4. If H2(Qxt
0 , Qxt
1 ) ≥γt ≥0 holds for all
t ∈[J]. Then

inf
p∈QJ
0 ,q∈QJ
1
H2(p, q) ≥2

 

1 −

J
Y

t=1
(1 −γt/2)

!

.

Proof. Observe that, for any distributions p, q we have

H2(p, q) = 2(1 −e−1

2 D1/2(p,q)).
(17)

Specifically, for give p ∈QJ
0 and q ∈QJ
1 , we have

1 −H2(p, q)/2 = e−1

2 D1/2(p,q) ≤e−1

2
PJ
t=1 ηt =

J
Y

t=1
e−1

2 ηt ≤

J
Y

t=1
(1 −γt/2),

where ηts are the constants in Proposition 2 and the last inequality follows by e−1

2 ηt ≤1 −γt/2 due
to (17) again. This completes the proof.

Proof of Theorem 4. We have, by Lemma 5, that the testing error is upper bounded by 2(1 −
infp∈QJ
0 ,q∈QJ
1 ||p −q||TV). Fix any such p, q, we have by [17, Equation 7.22] that 1 −||p −q||TV ≤
1 −1

2H2(p, q). The result then follows by Proposition 3.

15


---Page Break---
D
Proof of Theorem 2 (Lower Bound)

We denote L ≤log K with K = |H|, and x1, · · · , xL be L distinct elements in X satisfies the
condition of the theorem. We define for any b ∈{0, 1}L a function hb such that for all i ∈[L],
hb(xi) = yi if b[i] = 0 and hb(xi) = y′
i otherwise, where yi ̸= y′
i ∈Y are the elements that
satisfy infp∈Q
xi
yi ,q∈Q
xi
y′
i
{H2(p, q)} ≤γH. Let H be the class consisting of all such hb. Let qi ∈Qxi
yi
and q′
i ∈Qxi
y′
i be the elements satisfying H2(qi, q′
i) ≤γH. We now partition the features xT into
L epochs, each of length T/L, such that each epoch i has constant feature xi. Let h be a random
function selected uniformly from H. We claim that for any prediction rule ˆyt and any epoch i we
have

Eh,˜yT





(i+1)T/L
X

t=iT/L−1
1{h(xt) ̸= ˆyt}



≥Ω
 1

γH


,
(18)

where ˜yt ∼qi if h(xi) = yi and ˜yt ∼q′
i otherwise. The proposition now follows by counting the
errors for all L epochs.

We now establish (18) using the Le Cam’s two point method. Clearly, for each epoch i, the prediction
performance depends only on the label yi = h(xi), which is uniform over {yi, y′
i} and independent
for different epochs by construction. For any time step j during the ith epoch, we denote by ˜yj−1
and ˜y′j−1 the samples generated from qi and q′
i, respectively. By the Le Cam’s two point method [17,
Theorem 7.7] the expected error at step j is lower bounded by

1 −TV(˜yj−1, ˜y′j−1)

2
≥1 −
p

H2(˜yj−1, ˜y′j−1)(1 −H2(˜yj−1, ˜y′j−1)/4)

2
(19)

where the inequality follows from [17, Equation 7.22]. Note that the RHS of (19) is monotone
decreasing w.r.t. H2(˜yj−1, ˜y′j−1), since H2(p, q) ≤2 for all p, q.

By the tensorization of Hellinger divergence [17, Equation 7.23], we have

H2(˜yj−1, ˜y′j−1) = 2 −2(1 −H2(qi, q′
i)/2)j−1 ≤2 −2(1 −γH/2)j−1,

where the last inequality is implied by H2(qi, q′
i) ≤γH. Using the fact log(1 −x) ≥
−x
1−x, we have
if γH ≤1 and j −1 ≤
1
γH then 2 −2(1 −γH/2)j−1 ≤2(1 −e−1) < 2. Therefore, the RHS of
(19) is lower bounded by an absolute positive constant for all j −1 ≤
1
γH , and hence the expected
cumulative error will be lower bounded by Ω(1/γH) during epoch i. This completes the proof.

E
Proof of Lemma 2

Before presenting a formal proof, we first develop some technical concepts. Let ˜Y be the noisy
label set and D( ˜Y) be the class of distributions over ˜Y. We say a function ℓ: ˜Y × D( ˜Y) →R+ is
α-exp-concave if for any ˜y ∈˜Y, the function e−αℓ(˜y,p) is concave w.r.t. p for some α ∈R≥0.

Proposition 4. The function ℓ(˜y, p) = ||e˜y −p||2
2 is 1/4-Exp-concave, where e˜y denotes distribution
assigning probability 1 on ˜y.

Proof. We have by [13, Lemma 4.2] that a function f is α-Exp-concave if and only if

α∇f(p)∇f(p)T ⪯∇2f(p).

For any q ∈D( ˜Y), we denote f(p) = ||p −q||2
2. We have ∇f(p) = 2(p −q) and ∇2f(p) = 2I,
where I is the identity matrix. Taking any u ∈RJ, we have 1

4⟨u, 2(p −q)⟩2 ≤||u||2
2||p −q||2
2 ≤
2||u||2
2 = 2uTIu, where the first inequality follows by Cauchy-Schwarz inequality and the second
inequality follows by:

||p −q||2
2 =
X

˜y∈˜
Y
(p[˜y] −q[˜y])2 ≤
X

˜y∈˜
Y
max{p[˜y], q[˜y]}2 ≤
X

˜y∈˜
Y
p[˜y]2 + q[˜y]2 ≤2,

since p, q ∈D( ˜Y). This completes the proof.

16


---Page Break---
We now introduce the Exponential Weighted Average (EWA) algorithm and its regret analysis under the
Exp-concave losses, which is mostly standard [5, Chapter 3.3] and we include it here for completeness.
Let F = {f1, · · · , fK} ⊂D( ˜Y)X be a D( ˜Y)-valued function class and ℓ: ˜Y × D( ˜Y) →R≥0 be
α-Exp-concave. The EWA algorithm is presented in Algorithm 2.

Algorithm 2: Exponential Weighted Average (EWA) estimator
Input: Class F = {f1, · · · , fK} and α-Exp-concave loss ℓ
Set w1 = {1, · · · , 1} ∈RK;
for t = 1, · · · , T do

Receive xt;
Make prediction:

ˆpt =
PK
k=1 wt[k]fk(xt)

PK
k=1 wt[k]
.

Receive noisy label ˜yt;
for k ∈[K] do

Set wt+1[k] = wt[k]e−αℓ(˜yt,fk(xt));

Algorithm 2 provides the following regret bound:

Proposition 5 ([5, Proposition 3.1]). Let F ⊂D( ˜Y)X be any finite class and ℓbe an α-Exp-concave
loss. If ˆpt is the estimator in Algorithm 2, then for any xT ∈X T and ˜yT ∈˜YT we have

sup
f∈F

T
X

t=1
ℓ(˜yt, ˆpt) −ℓ(˜yt, f(xt)) ≤log |F|

α
.

Proof of Lemma 2. Let Φ be the EWA estimator as in Algorithm 2 with input class F, loss ℓ(˜y, p)
def
=
L2(e˜y, p) and α = 1/4. Let ˜yT be any realization of the noisy labels. We denote et as the standard
base of RM with value 1 at position ˜yt and zeros otherwise. By 1/4-Exp-concavity of loss ℓ
(Proposition 4) and the regret bound from Proposition 5, we have:

sup
f∈F,xT ∈X T ,˜yT ∈˜
YT

T
X

t=1
L2(et, ˆpt) −L2(et, f(xt)) ≤4 log |F|.
(20)

Note that, this bound holds point-wise w.r.t. any individual xT and ˜yT .

Fix any xT and (joint) distribution ˜p over ˜YT . We denote Et as the conditional expectation on ˜yt over
the randomness of ˜yT ∼˜p conditioning on ˜yt−1 and denote ˜pt as the conditional marginal. By the
elementary identity E[L2(X, p) −L2(X, q)] = L2(E[X], p) −L2(E[X], q) for any random variable
X over D( ˜Y), we have for all t ∈[T] that:

Et

L2(et, ˆpt) −L2(et, f(xt))

= L2(˜pt, ˆpt) −L2(˜pt, f(xt)),

since Et[et] = ˜pt for ˜yt ∼˜pt and ˆpt depends only on ˜yt−1. We now take E˜yT on both sides of (20).
By sup E ≤E sup and the law of total probability (i.e., E˜yT [X1 + · · · + XT ] = E˜yT [E1[X1] + · · · +
ET [XT ]] for any random variables XT ), we have:

sup
f∈F,xT ∈X T sup
˜p
E˜yT ∼˜p

" T
X

t=1
L2(˜pt, ˆpt) −L2(˜pt, f(xt))

#

≤4 log |F|,

where ˜p runs over all (joint) distributions over ˜YT . The lemma then follows by the equivalence
between operators QT
K ≡sup˜p E˜yT when taking the kernel set Qx
y := D( ˜Y) for all x, y (see
the discussion following Definition 1). The last part follows by the fact that the EWA estimator
automatically ensures ˆpt is a convex combination of {f(xt) : f ∈F} for all t ∈[T].

17


---Page Break---
F
Proof of Theorem 5

We define the following distribution valued function class F using hypothesis class H and noisy
kernel K. For any x ∈X, we denote by Qx
0 and Qx
1 the sets of noisy label distributions corresponding
to labels 0 and 1, respectively. Since the kernel K is well-separated at scale γL under L2 divergence,
we have, by the hyperplane separation theorem, that there must be qx
0 ∈Qx
0 and qx
1 ∈Qx
1 such
that L2(qx
0 , qx
1 ) = L2(Qx
0, Qx
1) ≥γL. We now define for any h ∈H the function fh such that
∀x ∈X, fh(x) = qx
h(x). Let F = {fh : h ∈H} and Φ be the estimator from Lemma 2 with class
F and L2 divergence (using xT , ˜yT from the original noisy classification game). Our classification
predictor is as follows:
ˆyt = arg min
y {L2(qxt
y , ˆpt) : y ∈{0, 1}}.
(21)

That is, we predict the label y so that qxt
y is closer to ˆpt under L2 divergence, where ˆpt = Φ(xt, ˜yt−1).

Let h∗∈H be the underlying true classification function and xT be the realization of features. We
have by Lemma 2 and 1/4-Exp-concavity of L2 divergence that 7

QT
K

" T
X

t=1
L2(˜pt, ˆpt) −L2(˜pt, fh∗(xt))

#

≤4 log |F|,
(22)

where QT
K is the operator in Definition 1.

For any time step t, we denote by yt = h∗(xt) the true label. Since qxt
y
∈Qxt
y are the elements
satisfying L2(qxt
0 , qxt
1 ) = L2(Qx1
0 , Qxt
1 ) ≥γL and ˆqt is a convex combination of qxt
0
and qxt
1
(Lemma 2), we have qxt
yt is the closest element in Qxt
yt to ˆpt under L2 divergence. Note that, we also
have ˜pt ∈Qxt
yt . Invoking Lemma 1, we find

L2(˜pt, ˆpt) −L2(˜pt, qxt
yt ) ≥L2(ˆpt, qxt
yt ).
(23)

Denote at = L2(˜pt, ˆpt) −L2(˜pt, fh∗(xt)). We have, by (23) and fh∗(xt) = qxt
yt that at ≥
L2(ˆpt, fh∗(xt)). Therefore:

1. For all t ∈[T], at ≥0, since ∀p, q, L2(p, q) ≥0;
2. If ˆyt ̸= yt, then at ≥γL/4. This is because the event {ˆyt ̸= yt} implies that L2(ˆpt, qxt
yt ) ≥
L2(ˆpt, qxt
1−yt). Hence, L2(ˆpt, fh∗(xt)) = L2(ˆpt, qxt
yt ) ≥γL/4. Here, we used the following
geometric fact:

2
p

L2(ˆpt, qxt
yt ) ≥
p

L2(ˆpt, qxt
yt ) +
q

L2(ˆpt, qxt
1−yt)

=
q

L2(qxt
yt , qxt
1−yt) ≥√γL.

This implies that ∀t ∈[T], at ≥γL

4 1{ˆyt ̸= yt}, therefore:

T
X

t=1
1{ˆyt ̸= yt} ≤4

γL

T
X

t=1
L2(˜pt, ˆpt) −L2(˜pt, fh∗(xt)).

The expected minimax risk now follows from (22).

G
Tight Bounds for Kernel Sets of Size One

In this appendix, we establish an upper bound for the case when the kernel set size |Qx
y| = 1 for all
x, y. This includes, for instance, the case when the parameter ηt is known in Example 1.
Theorem 6. Let H ⊂YX be any finite class and K be any noisy kernel that is well-separated at
scale γH w.r.t. squared Hellinger distance such that |Qx
y| = 1 for all x, y. Then the high probability
minimax risk at confidence δ > 0 is upper bounded by

Bδ(H, K) ≤O
log(|H|/δ)

γH


.

7Since QT
K[F(˜yT )] ≤QT [F(˜yT )] for any kernel K and function F, where QT is the unconstrained operator
in (10).

18


---Page Break---
Proof. Our proof follows a similar path as in the proof of Theorem 5, but replacing the L2 loss with
log-loss. Specifically, for any h ∈H, we define fh(x) = qx
h(x), where qx
h(x) is the unique element in
Qx
h(x). Denote F = {fh : h ∈H}. We run the EWA algorithm (Algorithm 2) over F with α = 1
and ℓbeing the log-loss [10], and produce an estimator ˆpT . The classifier is then given by

ˆyt = arg min
y∈Y{H2(qxt
y , ˆpt)}.

Now, our key observation is that the noisy label distribution ˜pt = fh∗(xt) is well-specified (since
|Qx
y| = 1, the only choice for ˜pt is fh∗(xt)), where h∗is the ground truth classifier. Therefore,
invoking [10, Lemma A.14], we find

Pr

" T
X

t=1
H2(˜pt, ˆpt) ≤log |F| + 2 log(1/δ)

#

≥1 −δ.

We claim that 1{ˆyt ̸= h∗(xt)} ≤
4
γH H2(˜pt, ˆpt). Clearly, this automatically satisfies if ˆyt = h∗(xt).
For ˆyt ̸= h∗(xt), we have H2(qxt
ˆyt , ˆpt) ≤H2(qxt
h∗(xt), ˆpt) = H2(˜pt, ˆpt) by definition of ˆyt. This
implies that

H2(˜p, ˆpt) ≥1

4H2(qxt
ˆyt , qxt
h∗(xt)) ≥γH

4 ,

where the first inequality follows by triangle inequality of Hellinger distance (the factor 1

4 comes
from the conversion form squared Hellinger distance to Hellinger distance), and the second inequality
follows by definition of γH. Therefore, we have w.p. ≥1 −δ that

T
X

t=1
1{ˆyt ̸= h∗(xt)} ≤4

γH

(log |F| + 2 log(1/δ)).

This completes the proof since |H| ≥|F|.

Observe that the key ingredient in the proof of Theorem 6 is the realizability of ˜pt by fh∗(i.e.,
well-specified) due to the property |Qx
y| = 1, which does not hold for general kernels.

H
Adaptive v.s. Oblivious Adversaries

In this appendix, we explain how the guarantees for oblivious adversaries can be extended to adaptive
adversaries. This primarily follows from [5, Lemma 4.1], but needs careful adaptation to fit our needs.
We consider the following abstract treatment: we assume that the adversary performs any operation
Qt at time step t and produces an action zt. For any randomized prediction rule ˆyT , the adaptive risk
can be expressed as

Q1Eˆy1 · · · QT EˆyT

" T
X

t=1
ℓ(zt, ˆyt)

#

.

Assume now that the randomness of ˆyt’s is independent and that ˆyt depends only on zt. We claim
that

Q1Eˆy1 · · · QT EˆyT

" T
X

t=1
ℓ(zt, ˆyt)

#

= QT EˆyT

" T
X

t=1
ℓ(zt, ˆyt)

#

.

We prove the case for T = 2 to demonstrate the ideas; the general case follows by induction. Observe
that

Q1Eˆy1Q2Eˆy2[ℓ(z1, ˆy1) + ℓ(z2, ˆy2)]
(a)
= Q1Eˆy1[ℓ(z1, ˆy1) + Q2Eˆy2ℓ(z2, ˆy2)]

(b)
= Q1[Eˆy1[ℓ(z1, ˆy1)] + Eˆy1Q2Eˆy2ℓ(z2, ˆy2)]

(c)
= Q1[Eˆy1[ℓ(z1, ˆy1)] + Q2Eˆy2ℓ(z2, ˆy2)]

(d)
= Q1Q2[Eˆy1[ℓ(z1, ˆy1)] + Eˆy2[ℓ(z2, ˆy2)]]

(e)
= Q1Q2Eˆy1Eˆy2[ℓ(z1, ˆy1) + ℓ(z2, ˆy2)]

19


---Page Break---
where (a) follows since ℓ(z1, ˆy1) is independent of Q2Eˆy2; (b) follows by the linearity of expectation;
(c) follows by the independence of ˆy1 and ˆy2, since the term Q2Eˆy2ℓ(z2, ˆy2) has nothing to do with
the realization of ˆy1; (d) follows since Eˆy1[ℓ(z1, ˆy1)] is independent of z2; (e) follows by the linearity
of expectation.

Observe that, all the predictors constructed in this paper have independent internal randomness (in
fact, the only place where randomness is introduced is in Algorithm 1); thus, our derived risk bounds
hold for adaptive adversaries as well.

20


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction is accurate to reflect the paper’s contributions.

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

Justification: Limitation is discussed in the Discussion section

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

21


---Page Break---
Justification: The assumptions are accurate and the proofs are complete.
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
Justification: This is a pure theory paper.
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

22


---Page Break---
Answer: [NA]
Justification: This is a pure theory paper.
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
Justification: This is a pure theory paper.
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
Justification: This is a pure theory paper.
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

23


---Page Break---
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
Justification: This is a pure theory paper.
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
Justification: We confirm the paper meets the NeurIPS Code of Ethics.
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
Justification: This is a pure theory paper, we do not see any direct societal impacts.
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

24


---Page Break---
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
Justification: This is a pure theory paper.
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
Justification: This is a pure theory paper.
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

25


---Page Break---
Answer: [NA]
Justification: This is a pure theory paper.
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
Justification: This is a pure theory paper.
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
Justification: This is a pure theory paper.
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

26


---Page Break---
