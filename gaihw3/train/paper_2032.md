Cryptographic Hardness of Score Estimation

Min Jae Song
Paul G. Allen School of Computer Science and Engineering
University of Washington
mjsong32@cs.washington.edu

Abstract

We show that L2-accurate score estimation, in the absence of strong assumptions
on the data distribution, is computationally hard even when sample complexity is
polynomial in the relevant problem parameters. Our reduction builds on the result of
Chen et al. (ICLR 2023), who showed that the problem of generating samples from
an unknown data distribution reduces to L2-accurate score estimation. Our hard-to-
estimate distributions are the “Gaussian pancakes” distributions, originally due to
Diakonikolas et al. (FOCS 2017), which have been shown to be computationally
indistinguishable from the standard Gaussian under widely believed hardness
assumptions from lattice-based cryptography (Bruna et al., STOC 2021; Gupte et
al., FOCS 2022).

1
Introduction

Diffusion models [70, 72, 42, 73] have firmly established themselves as a powerful approach to
generative modeling, serving as the foundation for leading image generation models such as DALL-E
2 [62], Imagen [66], and Stable Diffusion [65]. A diffusion model consists of a pair of forward and
reverse processes. In the forward process, noise drawn from a standard distribution, such as the
standard Gaussian, is sequentially applied to data samples, leading its distribution to a pure noise
distribution in the limit. The reverse process, as the name suggests, reverses the noising process and
takes the pure noise distribution “backward in time” to the original data distribution, thereby allowing
us to generate new samples from the data distribution. A key element in implementing the reverse
process is the score function of the data distribution, which is the gradient of its log density. Since the
data distribution is typically unknown, the score function must be learned from samples [44, 78, 72].

Recent advances in the theory of diffusion models have revealed that the task of sampling, in fact,
reduces to score estimation under minimal assumptions on the data distribution [9, 23, 53, 61, 22, 17,
51, 6]. In particular, Chen et al. [17] have shown that L2-accurate score estimates along the forward
process are sufficient for efficient sampling. Thus, assuming access to an oracle for L2-accurate score
estimation, one can efficiently sample from essentially any data distribution. However, this leaves
open the question of whether score estimation oracles themselves can be implemented efficiently, in
terms of both required sample size and computation, for interesting classes of distributions.

We show that L2-accurate score estimation, in the absence of strong assumptions on the data distribu-
tion, is computationally hard, even when sample complexity is polynomial in the relevant problem
parameters. This establishes a statistical-to-computational gap for L2-accurate score estimation,
which refers to an intrinsic gap between what is statistically achievable and computationally feasi-
ble. Our hard-to-estimate distributions are the “Gaussian pancakes” distributions, which previous
works [31, 13, 39] have shown are computationally indistinguishable from the standard Gaussian un-
der plausible and widely believed hardness assumptions. In fact, “breaking” the hardness of Gaussian
pancakes, by means of an efficient detection or estimation algorithm, has profound implications for
lattice-based cryptography, which is central to the post-quantum cryptography standardization led
by the National Institute of Standards and Technology (NIST) [58]. Building on the sampling-to-

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
score estimation reduction by Chen et al. [17], we show that computationally efficient L2-accurate
score estimation for Gaussian pancakes implies an efficient algorithm for distinguishing Gaussian
pancakes from the standard Gaussian. Thus, while sampling may ultimately reduce to L2-accurate
score estimation under minimal assumptions on the data distribution, score estimation itself requires
stronger assumptions on the data distribution for computational feasibility. It is worth noting that
the presence of statistical-to-computational gaps in L2-accurate score estimation was anticipated by
Chen et al. [17, Section 1.1], who mentioned it without formal statement or proof.

1.1
Main contributions

Our main result is a simple reduction from the Gaussian pancakes problem (i.e., the problem of distin-
guishing Gaussian pancakes from the standard Gaussian) to L2-accurate score estimation. We show
that given oracle access to L2-accurate score estimates along the forward process (Assumption A3),
one can compute a test statistic that distinguishes, with non-trivial success probability, whether the
given score estimates belong to a Gaussian pancakes distribution or the standard Gaussian.

A Gaussian pancakes distribution Pu with secret direction u ∈Sd−1 can be viewed as a “backdoored”
Gaussian. It is distributed as a (noisy) discrete Gaussian along the direction u and as a standard
Gaussian in the remaining d−1 directions (see Figure 1).1 A class of Gaussian pancakes (Pu)u∈Sd−1
is parameterized by two parameters, γ and σ, which govern the spacing and thickness of pancakes,
respectively. For instance, a Gaussian pancakes distribution Pu with spacing γ and thickness
σ ≈0 is essentially supported on the one-dimensional lattice (1/γ)Z along the secret direction
u. The Gaussian pancakes problem, then, is a sequence of hypothesis testing problems indexed by
the data dimension d ∈N in which the goal is to distinguish between samples from a Gaussian
pancakes distribution (with unknown u) and the standard Gaussian distribution N(0, Id) with success
probability slightly better than random guessing (see Section 2.3 for formal definitions). Thus, our
result can be summarized informally as follows.

Theorem 1.1 (Informal, see Theorem 3.1). Let γ(d) > 1, σ(d) > 0 be sequences such that σ ≥
1/poly(d) and the corresponding (γ, σ)-Gaussian pancakes distributions (Pu)u∈Sd−1 all satisfy
TV(Pu, N(0, Id)) > 1/2. Then, there exists a polynomial-time randomized algorithm with access
to a score estimation oracle of L2-error O(1/√log d) that solves the Gaussian pancakes problem.

We emphasize that the hardness of estimating score functions of Gaussian pancakes distributions arises
solely from hardness of learning. The score function of Pu is efficiently approximable by function
classes commonly used in practice for generative modeling, such as residual networks [40]. In
addition, under the scaling σγ = O(1), which includes the cryptographically hard regime, the secret
parameter u can be estimated upto L2-error η via brute-force search over Sd−1 with poly(d, γ, 1/η)
samples (Theorem 4.2). The estimated parameter ˆu in turn enables L2-accurate score estimation
(see Section 4). Our estimator, based on projection pursuit [35, 43], may be of independent interest.

We also analyze properties of Gaussian pancakes using Banaszczyk’s theorems on the Gaussian mass
of lattices [3, 74]. This serves two purposes. Firstly, it allows us to verify that Gaussian pancakes
distributions readily satisfy the assumptions for the sampling-to-score estimation reduction of Chen
et al. [17], namely Lipschitzness of the score functions along the forward process (Assumption A1).
This is necessary as the proof of our main theorem crucially relies on the reduction. Secondly,
Banaszczyk’s theorems provide simple means of analyzing properties of Gaussian pancakes, which
are interesting mathematical objects in their own right. While these theorems are standard tools in
lattice-based cryptography (see e.g., [74, 1]), they are likely less known outside the community.

1.2
Related work

Theory of diffusion models.
Recent advances in the theoretical study of diffusion models have
focused on convergence rates of discretized reverse processes [9, 23, 53, 61, 22, 17, 51, 6]. Of
particular relevance to our work is the result of Chen et al. [17] who showed that L2-accurate
score estimates are sufficient to guarantee convergence rates that are polynomial in all the relevant
problem parameters under minimal assumptions on the data distribution, namely Lipschitz scores
throughout the forward process and finite second moment. Prior studies fell short by requiring strong
structural assumptions on the data distribution, such as a log-Sobolev inequality [50, 82], assuming

1An animated visualization of Gaussian pancakes can be found in [12].

2


---Page Break---
Figure 1: Top: Scatter plot of 2D Gaussian pancakes Pu with secret direction u = (−1/
√

2, 1/
√

2),
spacing γ = 6, and thickness σ ∈{0.01, 0.05, 0.25}. Bottom: Re-scaled probability densities of
Gaussian pancakes (blue) for each σ ∈{0.01, 0.05, 0.25} and the standard Gaussian (black) along u.
For fixed γ, the pancakes “blur into each other” as σ increases.

L∞-accurate score estimates [23], or providing convergence rates that are exponential in the problem
parameters [9, 23, 22]. We note that recent works have made the “minimal” assumptions of Chen et
al. [17] even more minimal by considering early stopped reverse processes and dropping the Lipschitz
score assumption [15, 6]. We refer to the book draft of Chewi [19] for more background.

Gaussian pancakes.
The Gaussian pancakes problem stands out among problems exhibiting
statistical-to-computational gaps due to its versatility and strong hardness guarantees. Initially intro-
duced as hard-to-learn Gaussian mixtures in the work of Diakonikolas et al. [31], which established
their SQ hardness, Gaussian pancakes have been extensively utilized in establishing SQ lower bounds
for various statistical inference problems such as robust Gaussian mean estimation [31] and agnos-
tically learning halfspaces and ReLUs over Gaussian inputs [28]. For further details, we refer to
the textbook by Diakonikolas and Kane [29, Chapter 8]. Gaussian pancakes distributions them-
selves serve as instances of fundamental high-dimensional inference problems such as non-Gaussian
component analysis [8] and Wasserstein distance estimation in the spiked transport model [57].

Bruna et al. [13] initiated the exploration of the cryptographic hardness of Gaussian pancakes.
They showed that assuming hardness of worst-case lattice problems, fundamental to lattice-based
cryptography [56, 60], both the Gaussian pancakes and the closely related continuous learning with
errors (CLWE) problems are hard. Follow-up work by Gupte et al. [39] showed that the learning with
errors (LWE) problem [63], a versatile problem which lies at the heart of numerous lattice-based
cryptographic constructions, reduces to the Gaussian pancakes as well. These cryptographic hardness
results have sparked a wave of recent works showcasing various applications of this newly discovered
property of Gaussian pancakes. Notable examples include planting undetectable backdoors in machine
learning models [38], novel public-key encryption schemes based on Gaussian pancakes [10], and
cryptographic hardness of agnostically learning halfspaces [26, 75].

For additional related work on score estimation and statistical-to-computational gaps, see Section A.

1.3
Future directions

Our work brings together the latest advances from the theory of diffusion models and computational
complexity of statistical inference. This intersection provides fertile ground for future research, some
avenues of which we outline below.

3


---Page Break---
Stronger data assumptions for efficient score estimation.
Our result shows that for any class
of distributions that encompasses hard Gaussian pancakes, computationally efficient L2-accurate
score estimation is impossible. This implies that stronger assumptions on the data, which exclude
Gaussian pancakes, are necessary for efficient score estimation. Finding assumptions that exclude
hard instances while being able to capture realistic models of data is an interesting open problem.

Weaker criteria for evaluating sample quality.
In the context of learning Gaussian pancakes, if
the goal of the sampling algorithm were to merely fool computationally bounded testers that lack
knowledge of the secret u ∈Sd−1, then it could simply generate standard Gaussian samples and
fool any polynomial-time test. Thus, sampling is strictly easier than L2-accurate score estimation if
the criteria for evaluating sample quality is less stringent. This suggests exploring sampling under
different evaluation criteria, such as “discriminators” with bounded compute or memory. For example,
Christ et al. [20] have used weaker notions of sample quality in the context of watermarking large
language models (LLMs). More precisely, they used the notion of computational indistinguishability
to guarantee quality of the watermarked model relative to the original model. Exploring potential
connections to the literature on leakage simulation [45, 18, 76] and outcome indistinguishability [41,
32, 33] is also an interesting future direction, as these areas have addressed related questions for
distributions on finite sets.

Extracting “knowledge” from sampling algorithms.
A key difficulty in directly reducing the
Gaussian pancakes problem to sampling is that the Gaussian pancakes problem is hard for polynomial-
time distinguishers, even with access to exact sampling oracles (see Section 2.3 for more details).2
Thus, any procedure utilizing the learned sampler in a black box manner cannot solve the Gaussian
pancakes problem. This is puzzling since for the algorithm to have “learned” to generate samples
from the given distribution, it ought to possess some non-trivial information about it (e.g., leak
information about the secret parameter u)!

This raises the question: How much “knowledge” can we extract with white box access to the
sampling algorithm? Under reasonable structural assumptions on the sampling algorithm, can we
extract privileged information about the data distribution it simulates, beyond what is obtainable
solely via sample access? Our work provides one such example. White box access to a diffusion
model gives access to its score estimates. These score estimates, which enable efficient solutions to
the Gaussian pancakes problem, constitute privileged knowledge that cannot be learned efficiently
even with unlimited access to bona fide samples.

2
Preliminaries

Notation.
We denote by (at) the sequence a1, a2, a3, . . . indexed by t ∈N. When there is no
natural ordering on the index set (e.g., (au)u∈Sd−1), we interpret it as a set. We write a ≲b or
a = O(b) to mean that a ≤Cb for some universal constant C > 0. The notation a ≳b and a = Ω(b)
are defined analogously. We write a ≍b or a = Θ(b) to mean that a ≲b and a ≳both hold.
We write ∧, ∨to mean logical AND and OR, respectively. We also write a ∧b and a ∨b to mean
min(a, b) and max(a, b), respectively. We denote by Qd the “standard” Gaussian N(0, 1/(2π)Id)
(see Remark 2.3). We omit the subscript d when it is clear from context.

2.1
Background on denoising diffusion probabilistic modeling (DDPM)

We give a brief exposition on denoising diffusion probabilistic models (DDPM) [42], a specific type
of diffusion model, since the main reduction of Chen et al. [17] pertains to DDPMs. This section
closely follows [17, Section 2.1] and [80, Section 3]

Let D be the target distribution defined on Rd. In DDPMs, we begin with the forward process, an
Ornstein-Uhlenbeck (OU) process that converges towards Q = N(0, (1/2π)Id), which is described
by the following stochastic differential equation (SDE). Note that the constant 1/√π in front of
dWt in Eq.(1) is non-standard.3 See Remark 2.3 for an explanation of our unconventional choice of

2Note, however, that algorithms that run in time T can query the sampling oracle at most T times, so the
running time imposes a limit on the number of samples the algorithm can see.
3The usual choice is
√

2, for which the stationary distribution is N(0, Id).

4


---Page Break---
variance for the resulting stationary distribution Q.

dXt = −Xtdt + (1/√π)dWt ,
X0 ∼D0 = D ,
(1)

where Wt is the standard Brownian motion in Rd.

Let Dt be the distribution of Xt along the OU process. It is well-known that for any distribution D,
Dt →Q exponentially fast in various divergences and metrics such as the KL divergence [2]. In this
work, we only consider Gaussian pancakes (Pu) and the standard Gaussian Q. If D = Pu and t > 0,
then the distribution Dt is simply another Gaussian pancakes distribution with a larger thickness
parameter (see Definition 2.5). Meanwhile, if D = Q, then Dt = Q for any t ≥0. We run the OU
process until time T > 0, and then simulate the reverse process, described by the following SDE.

dYt = (Yt + (1/π)∇log DT −t(Yt))dt + (1/√π)dWt ,
Y0 ∼F0 = DT .
(2)

Here, ∇log Dt is called the score function of Dt. Since the target D is not known, in order to
implement the reverse process the score function must be estimated from data samples. Assuming
for the moment that we have exact scores (∇log Dt)t∈[0,T ], if we start the reverse process from
F0 = DT , we have Yt ∼Ft = DT −t for any 0 ≤t ≤T, and ultimately YT ∼FT = D. Thus,
starting from (approximately) pure noise DT ≈Q, the reverse process generates fresh samples from
the target distribution D. We need to make several approximations to algorithmically implement this
reverse process. In particular, we need to approximate DT by Q, discretize the continuous-time SDE,
and approximate scores along the (discretized) forward process. Let h > 0 be the step size for the
SDE discretization and denote N := T/h. Given score estimates (skh)k∈[N], the DDPM algorithm
performs the following update (see e.g., [67, Chapter 4.3]).

yk+1 = ehyk + (1/π)(eh −1)s(N−k)h(yk) +
p

e2h −1zk ,
(3)

where zk ∼Q is an independent Gaussian vector. Note that the only dependence of the reverse
process on the target distribution D arises through the score estimates (skh)k∈[N].

The result of Chen et al. [17] demonstrates polynomial convergence rates of this process to the target
distribution D, assuming access to L2-accurate score estimates (skh)k∈[N] and minimal conditions
on the target D. We refer to Section 3.1 for a formal statement of their assumptions and theorem.

2.2
Lattices and discrete Gaussians

Lattices.
A lattice L ⊂Rd is a discrete additive subgroup of Rd. In this work, we assume all lattices
are full rank, i.e., their linear span is Rd. For a d-dimensional lattice L, a set of linearly independent
vectors {b1, . . . , bd} is called a basis of L if L is generated by the set, i.e., L = BZd where
B = [b1, . . . , bd]. The determinant of a lattice L with basis B is defined as det(L) = | det(B)|.

The dual lattice of a lattice L, denoted by L∗, is defined as

L∗= {y ∈Rd | ⟨x, y⟩∈Z for all x ∈L} .

If B is a basis of L then (BT )−1 is a basis of L∗; in particular, det(L∗) = det(L)−1.

Fourier analysis.
We define the Fourier transform of a function f : Rd →C by

bf(y) =
Z

Rd f(x) exp(−2πi⟨x, y⟩)dx .

The Poisson summation formula offers a valuable tool for analyzing functions defined on lattices.

Lemma 2.1 (Poisson summation formula). For any lattice L ⊂Rd and any function f : Rd →C,4

f(L) = det(L∗) · bf(L∗) ,

where we denote f(S) = P

x∈S f(x) for any set S.

4To be precise, f must satisfy some niceness conditions; this will always hold in our applications.

5


---Page Break---
Discrete Gaussians.
A discrete Gaussian is a discrete distribution whose probability mass function
is given by the Gaussian function. These distributions are closely related to Gaussian pancakes
distributions and their properties will be crucial for our analysis.
Definition 2.2 (Gaussian function). We define the Gaussian function ρs : Rd →R of width s > 0 by

ρs(x) := exp(−π∥x∥2/s2) .

When s = 1, we omit the subscript and simply write ρ. In addition, for any lattice L ⊂Rd, we denote
by ρs(L) = P

x∈L ρs(x) the corresponding Gaussian mass of L.

Remark 2.3 (Non-standard choice of “standard” variance). We refer to Qd = N(0, 1/(2π)Id) as
the “standard” Gaussian. This is indeed the standard choice in lattice-based cryptography because it
simplifies normalization factors that arise from taking Fourier transforms. For instance, it allows us
to simply write bρs = snρ1/s and bρ = ρ for s = 1. To translate these results for the “usual” standard
Gaussian, we can simply replace s with s/
√

2π for each occurrence.

Definition 2.4 (Discrete Gaussian). For any lattice L ⊂Rd, parameter s > 0, and shift t ∈Rd, the
discrete Gaussian DL−t,s is a distribution supported on the coset L −t with probability mass

DL−t,s(x) = ρs(x −t)

ρs(L −t) .

We denote by Aγ the discrete Gaussian of width s = 1 supported on the one-dimensional lattice
(1/γ)Z. The distribution of a Gaussian pancakes distribution Pu along the hidden direction is a
smoothed discrete Gaussian, which we formalize in the following.
Definition 2.5 (Smoothed discrete Gaussian). For any γ > 0, let Aγ be the discrete Gaussian of
width 1 on the lattice (1/γ)Z. We define the σ-smoothed discrete Gaussian Aσ
γ as the distribution of
the random variable y induced by the following process.

y =
1
√

1 + σ2 (x + σz) , where x ∼Aγ and z ∼Q .

Furthermore, the density of Aσ
γ is given by

Aσ
γ(z) =

√

1 + σ2

σρ((1/γ)Z)

X

k∈Z
ρ(k/γ)ρσ/
√

1+σ2
 
z −k/γ
p

1 + σ2
.
(4)

Likelihood ratio of smoothed discrete Gaussians.
Let Aσ
γ be the σ-smoothed discrete Gaussian
on (1/γ)Z. Its likelihood ratio T σ
γ with respect to the standard Gaussian is given by

T σ
γ (z) =

√

1 + σ2

σρ((1/γ)Z)

X

k∈Z
ρσ(z −
p

1 + σ2k/γ) .
(5)

When γ and σ are clear from context, we omit them and simply denote the likelihood ratio by T.

2.3
Gaussian pancakes

We define Gaussian pancakes distributions using the likelihood ratio T σ
γ = Aσ
γ/Q. It is important
to note that our parametrization differs from the one used in previous works [13, 39]. We believe
our parametrization is more convenient as it elucidates a natural partial ordering on the space of
parameters (γ, σ). In addition, there is an explicit mapping between the two different parametrizations,
so computationally hard parameter regimes identified by previous works [13, 39] can readily be
translated into setting. See Remark 2.7 for more details.
Definition 2.6 (Gaussian pancakes). For any d ∈N, spacing and thickness parameters γ, σ > 0, we
define the (γ, σ)-Gaussian pancakes distribution P σ
γ,u with secret direction u ∈Sd−1 by

P σ
γ,u(x) := Q(x) · T σ
γ (⟨x, u⟩) ,

where Q = N(0, (1/2π)Id) and T σ
γ is the likelihood ratio of Aσ
γ with respect to Q. When parameters
γ, σ are clear from context, we omit them in the notation and simply denote the distribution by Pu to
avoid clutter.

6


---Page Break---
Remark 2.7 (Partial ordering on Gaussian pancakes). The smoothed discrete Gaussian Aσ
γ arises
in the OU process for the discrete Gaussian Aγ at time t = log
 √

1 + σ2
. Consequently, for any
fixed γ > 0, there exists a natural partial ordering on the family of Gaussian pancakes parametrized
by (γ, σ), given by (γ, σ1) ≤(γ, σ2) whenever σ1 ≤σ2. This ordering arises from the fact that Aσ1
γ
reduces to Aσ2
γ whenever σ1 ≤σ2 via the OU process starting at t1 = log
 p

1 + σ2
1

and run until
t2 = log
 p

1 + σ2
2

.

Definition 2.8 (Advantage). Let A : X →{0, 1} be any decision rule (i.e., distinguisher). For any
pair of distributions (P, Q) on X, we define the advantage of A by

α(A) :=
P[A(X) = 1] −Q[A(X) = 1]
 .

For a sequence of decision rules (Ad)d∈N and distribution pairs (Pd, Qd)d∈N, we say (Ad) has
non-negligible advantage with respect to (Pd, Qd) if its advantage sequence αd = α(Ad) is a
non-negligible function in d, i.e., a function in Ω(d−c) for some constant c > 0.
Definition 2.9 (Computational indistinguishability). A sequence of distribution pairs (Pd, Qd)
is computationally indistinguishable if no poly(d)-time computable decision rule achieves non-
negligible advantage.

Definition 2.10 (Gaussian pancakes problem). For any sequences γ(d), σ(d) > 0 and n(d) ∈N, the
(γ, σ, n)-Gaussian pancakes problem is to distinguish (Pd, Qd) with non-negligible advantage, where
Pd is the n-sample distribution induced by the following two-stage process: 1) draw u uniformly
from Sd−1, 2) draw n i.i.d. Gaussian pancakes samples x1, . . . , xn ∼P ⊗n
u , and Qd = Q⊗n
d , i.e.,
the distribution of n i.i.d. standard Gaussian vectors.

As will be explained next, the exact number of samples n is irrelevant for most applications due to the
cryptographic hardness of the Gaussian pancakes problem. For certain parameter regimes of (γ, σ),
the problem maintains its computational intractability regardless of the sample size n.

Hardness of Gaussian pancakes.
There is an abundance of evidence demonstrating the hardness
of the Gaussian pancakes problem. This makes it compelling to directly assume that Gaussian
pancakes and the standard Gaussian are computationally indistinguishable (see Definition 2.9) for
certain parameter regimes of (γ, σ). Initial results by Bruna et al. [13, Corallary 4.2] showed that
the Gaussian pancakes problem is as hard as worst-case lattice problems for any parameter sequence
(γ, σ) satisfying γ ≥2
√

d and σ ≥1/poly(d). SQ hardness of the problem has been demonstrated
as well [31, 13, 27]. Perhaps surprisingly, the reduction of Bruna et al. shows that even with unlimited
access to an exact sampling oracle, no polynomial-time algorithm A can achieve non-negligible
advantage on the (γ, σ)-Gaussian pancakes problem. This stems from the fact that the running time
of A naturally restricts the number of samples it can “see”, resolving the apparent mystery.

An important follow-up work by Gupte et al. [39] reduced the well-known LWE problem to the
Gaussian pancakes problem. Assuming sub-exponential hardness of LWE [52], a standard assumption
underlying post-quantum cryptosystems expected to be standardized by NIST, the Gaussian pancakes
problem is hard for any γ ≥(log d)1+ε, where ε > 0 is any constant, and σ ≥1/poly(d) [39,
Section 1.2]. Taken together, these findings strongly support the hardness of Gaussian pancakes for
the specified regimes of (γ, σ). Note, however, that the condition σ ≥1/poly(d) is necessary for
hardness as there exist polynomial-time algorithms, based on lattice basis reduction, for exponentially
small σ [83, 25].

3
Hardness of Score Estimation

Our main result is Theorem 3.1, which presents a reduction from the Gaussian pancakes problem
to L2-accurate score estimation. Since the Gaussian pancakes problem exhibits both cryptographic
and SQ hardness in the parameter regime γ ≥2
√

d and σ ≥1/poly(d) [13, 39], these notions of
hardness extend to the task of estimating scores of Gaussian pancakes. Further details on the hardness
of the Gaussian pancakes problem can be found in Section 2.3.
Theorem 3.1 (Main result). Let γ(d) > 1, σ(d) > 0 be any pair of sequences such that σ ≥
1/poly(d) and the corresponding (sequence of) (γ, σ)-Gaussian pancakes distributions (Pu)u∈Sd−1

7


---Page Break---
satisfies TV(Pu, Qd) > 1/2 for any d ∈N. Then, for any δ ∈(0, 1), there exists a poly(d)·log(1/δ)-
time algorithm with access to a score estimation oracle of L2-error O(1/√log d) that solves the
(γ, σ)-Gaussian pancakes problem with probability at least 1 −δ.

The requirement TV(Pu, Qd) > 1/2 is mild and entirely captures interesting parameter regimes
of (γ, σ) for which cryptographic and SQ hardness of Gaussian pancakes are known. We provide
a sufficient condition in Lemma B.9, which shows that γσ < C for some constant C > 0 ensures
separation in TV distance.

Theorem 3.1 implies that even a score estimation oracle running in time poly(d, 21/ε2), where
ε > 0 is the L2 estimation error bound, implies a poly(d) time algorithm for the Gaussian pancakes
problem. This means that estimating the score functions of Gaussian pancakes to L2-accuracy ε even
in poly(d, 21/ε2) time is impossible under standard cryptographic assumptions.

3.1
Proof outline of Theorem 3.1

Here, we sketch the proof of Theorem 3.1 and defer full details to Section B.1. We first recall the
sampling-to-score estimation reduction of Chen et al. [17] and its required assumptions to illustrate
the main idea behind our reduction. The precise formulation of our idea is given in Lemma 3.3.

Assumptions on data distribution.
The reduction of Chen et al. [17] requires the following
assumptions on the data distribution D over Rd.

A1 (Lipschitz score). For all t ≥0, the score ∇log Dt is L-Lipschitz.
A2 (Finite second moment). D has finite second moment, i.e., m2
2 := Ex∼D[∥x∥2] < ∞.
A3 (Score estimation error). For step size h := T/N and all k = 1, . . . , N,

EDkh[∥skh −∇log Dkh∥2] ≤ε2
score .
Theorem 3.2 ([17, Theorem 2]). Suppose assumptions A1-A3 hold. Let Qd be the standard
Gaussian on Rd and let FT be the output of the DDPM algorithm (Section 2.1) at time T with step
size h := T/N such that h ≲1/L, where L ≥1. Then, it holds that

TV(FT , D) ≲
p

KL(D ∥Qd) · exp(−T)
|
{z
}
convergence of forward process

+
(L
√

dh + Lm2h)
√

T
|
{z
}
discretization error

+
εscore
√

T
|
{z
}
score estimation error
.
(6)

In particular, if m2 ≤d, then T ≍max(log(KL(D ∥Qd)/ε), 1) and h ≍ε2/(L2d) gives

TV(FT , D) ≲ε + εscore · max(
p

log(KL(D ∥Qd)/ε), 1) ,
for N = ˜Θ
L2d

ε2


.

Theorem 3.2 shows that if the unknown data distribution D has Lipschitz scores and satisfies m2 ≤d,
then its ε-accurate score estimates along the discretized forward process (skh)k∈[N] can be used to
compute a certificate of Gaussianity defined as follows.
∆:= max
k∈[N] EQd∥skh(x) + 2πx∥2 .
(7)

Using ∆as a test statistic, we decide D ∈(Pu)u∈Sd−1 if ∆≥τ for some carefully chosen threshold
τ > 0 and D = Qd otherwise. This test statistic is motivated by the observation that the discretized
reverse process depends on the data distribution D solely through its score estimates (see Eq.(3)).
For any sequence of score estimates (st)t∈[0,T ] the reverse process outputs FT that is TV-close to D
provided its L2 error along the forward process (Dt)t∈[0,T ] is small.

We claim that if ∆≤η2 and the score estimates (skh) are ε-accurate for (Dkh), then the output of
the reverse process FT is roughly (ε + η)-close in TV distance to the standard Gaussian. This is
because the standard Gaussian is invariant throughout the OU process, so ∆is, in fact, the L2 score
estimation error bound for the case where the data distribution D is equal to Qd. In other words,
∆≤η2 means that for all k ∈[N], the score estimates (skh) are η-close to −2πx which is the
score function of Qd. Thus, ∆is small only if D is close in TV distance to Qd, which shows that ∆
distinguishes between D = Qd and D ∈(Pu)u∈Sd−1 provided TV(Pu, Qd) > 1/2. The following
lemma formalizes this idea. Theorem 3.1 follows from Lemma 3.3 and Lipschitzness of the score
functions (Lemma B.5).

8


---Page Break---
Lemma 3.3 (Gaussianity testing with scores). For any ε ∈(0, 1) and K ≥2, let D be any dis-
tribution on Rd such that m2 ≤d and KL(D ∥Qd) ≤K with L-Lipschitz score ∇log Dt for
any t ≥0, and let ˜ε ≍ε/
p

log(K/ε) be the L2 score estimation error bound with discretization
parameters T ≍log(K/ε), h ≍ε2/(L2d), and N := T/h so that TV(FT ∥D) ≤ε (via The-
orem 3.2). If (skh)k∈[N] are ˜ε-accurate score estimates for the forward process (Dkh)k∈[N] and
∆= maxk∈[N] EQd∥skh(x) + 2πx∥2, then

TV(D, Qd) ≲ε +
p

∆log(K/ε) .

In particular, if TV(D, Qd) > 1/2 then there exist constants C1, C2 > 0 such that for any score
estimates (skh)k∈[N] of the forward process satisfying maxk∈[N] EDkh∥skh(x)−∇log Dkh(x)∥2 ≤
C1/ log K, it holds
∆≥C2/ log K .

Proof. By Theorem 3.2 and our choice of L2 score estimation error bound ˜ε, we have TV(FT , D) ≤
ε. In addition, the score estimates (st) also satisfy a
√

∆-error bound with respect to the forward
process of Qd, which is invariant with respect to time t. Thus, Theorem 3.2 applied with D = Qd as
the data distribution, discretization parameters T, h, and score estimates (skh) gives TV(FT , Qd) ≲
p

∆log(K/ε). By the triangle inequality, we have

TV(D, Q) ≤TV(FT , D) + TV(FT , Q) ≲ε +
p

∆log(K/ε) .

The second part of the theorem follows from using the assumptions TV(D, Q) > 1/2, K ≥2, and
fixing ε > 0 to a sufficiently small constant, which gives us

1 ≲
p

∆log K .

4
Sample Complexity of Gaussian Pancakes

To establish that there is indeed a gap between statistical and computational feasibility, we demonstrate
a polynomial upper bound on the sample complexity of L2-accurate score estimation for Gaussian
pancakes. In particular, we show that a sufficiently good estimate ˆu of the hidden direction u
is enough (Lemma 4.1). The polynomial sample complexity of score estimation then follows
from Theorem 4.2, which shows that if γ(d) and σ(d) satisfy γσ = O(1), then 1 −⟨ˆu, u⟩2 ≤η2 is
statistically achievable with poly(d, γ, 1/η) samples, albeit through brute-force search over Sd−1.

Our estimator ˆu is based on projection pursuit [35, 43]. We design a functional E : Sd−1 →R of the
form E(v) := Ex∼Pug(⟨x, v⟩) with g : R →R carefully chosen to ensure that E(v1) ≥E(v2) if
and only if |⟨v1, u⟩| ≥|⟨v2, u⟩|. Given such a “monotonic” functional (its empirical counterpart ˆE,
to be precise), our estimator for the secret direction u is essentially
ˆu = arg max
v∈Sd−1
ˆE(v) .

We remark that parameter regime γσ = O(1) in Theorem 4.2 encompasses the cryptographically
hard regime of Gaussian pancakes. It is also worth noting that if min(γ, γσ) = ω(√log d), then
Gaussian pancakes are statistically indistinguishable from Qd by Lemma B.10.

We defer the full proofs of Lemma 4.1 and Theorem 4.2 to Section C.
Lemma 4.1 (Score-to-parameter estimation reduction). For any γ > 1, σ > 0, let Pu be the (γ, σ)-
Gaussian pancakes distribution with secret direction u ∈Sd−1. Given any η ∈(0, 1) and ˆu ∈Sd−1
such that 1 −⟨ˆu, u⟩2 ≤η2, the score estimate ˆs(x) = −2πx + ∇log T σ
γ (⟨x, ˆu⟩) satisfies

Ex∼Pu∥ˆs(x) −s(x)∥2 ≲max(1, 1/σ8) · η2d ,
where s(x) = −2πx + ∇log T σ
γ (⟨x, u⟩) is the true score function of Pu.
Theorem 4.2 (Sample complexity of parameter estimation). For any constant C > 0, given γ(d) >
1, σ(d) > 0 such that γσ < C, estimation error parameter η > 0, and δ ∈(0, 1), there exists a brute-
force search estimator ˆu : Rd×n →Sd−1 that uses n = poly(d, γ, 1/η, 1/δ) samples and achieves
∥ˆu(x1, . . . , xn) −u∥2 ≤η2 with probability at least 1 −δ over i.i.d. samples x1, . . . , xn ∼Pu,
where Pu is the (γ, σ)-Gaussian pancakes distribution with secret direction u ∈Sd−1.

9


---Page Break---
References

[1] D. Aggarwal and N. Stephens-Davidowitz. An improved constant in banaszczyk’s transference
theorem. arXiv preprint arXiv:1907.09020, 2019.

[2] D. Bakry, I. Gentil, M. Ledoux, et al. Analysis and geometry of Markov diffusion operators,
volume 103. Springer, 2014.

[3] W. Banaszczyk. New bounds in some transference theorems in the geometry of numbers.
Mathematische Annalen, 296:625–635, 1993.

[4] A. S. Bandeira, A. Perry, and A. S. Wein. Notes on computational-to-statistical gaps: predictions
using statistical physics. Portugaliae Mathematica, 75(2):159–186, 2018.

[5] B. Barak and D. Steurer. Sum-of-squares proofs and the quest toward optimal algorithms. arXiv
preprint arXiv:1404.5236, 2014.

[6] J. Benton, V. De Bortoli, A. Doucet, and G. Deligiannidis. Nearly d-linear convergence bounds
for diffusion models via stochastic localization. In The Twelfth International Conference on
Learning Representations, 2024.

[7] Q. Berthet and P. Rigollet.
Computational lower bounds for sparse pca.
arXiv preprint
arXiv:1304.0828, 2013.

[8] G. Blanchard, M. Kawanabe, M. Sugiyama, V. Spokoiny, K.-R. Müller, and S. Roweis. In
search of non-gaussian components of a high-dimensional distribution. Journal of Machine
Learning Research, 7(2), 2006.

[9] A. Block, Y. Mroueh, and A. Rakhlin. Generative modeling with denoising auto-encoders and
langevin sampling. arXiv preprint arXiv:2002.00107, 2020.

[10] A. Bogdanov, M. C. Noval, C. Hoffmann, and A. Rosen. Public-key encryption from continuous
lwe. Cryptology ePrint Archive, 2022.

[11] M. Brennan and G. Bresler. Reducibility and statistical-computational gaps from secret leakage.
In Conference on Learning Theory, pages 648–847. PMLR, 2020.

[12] B.
Brubaker.
In
neural
networks,
unbreakable
locks
can
hide
invisible
doors.
Quanta
magazine,
2023.
https://www.quantamagazine.org/
cryptographers-show-how-to-hide-invisible-backdoors-in-ai-20230302/
[Accessed: 2024-03-12].

[13] J. Bruna, O. Regev, M. J. Song, and Y. Tang. Continuous lwe. In Proceedings of the 53rd
Annual ACM SIGACT Symposium on Theory of Computing, 2021.

[14] C. L. Canonne.
A short note on an inequality between kl and tv.
arXiv preprint
arXiv:2202.07198, 2022.

[15] H. Chen, H. Lee, and J. Lu. Improved analysis of score-based generative modeling: User-
friendly bounds under minimal smoothness assumptions. In International Conference on
Machine Learning, pages 4735–4763. PMLR, 2023.

[16] M. Chen, K. Huang, T. Zhao, and M. Wang. Score approximation, estimation and distribution
recovery of diffusion models on low-dimensional data. In International Conference on Machine
Learning, pages 4672–4712. PMLR, 2023.

[17] S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, and A. R. Zhang. Sampling is as easy as learning the
score: theory for diffusion models with minimal data assumptions. In International Conference
on Learning Representations (ICLR), 2023.

[18] Y.-H. Chen, K.-M. Chung, and J.-J. Liao. On the complexity of simulating auxiliary input. In
Annual International Conference on the Theory and Applications of Cryptographic Techniques,
pages 371–390. Springer, 2018.

[19] S. Chewi. Log-concave sampling, 2024.

10


---Page Break---
[20] M. Christ, S. Gunn, and O. Zamir. Undetectable watermarks for language models. arXiv
preprint arXiv:2306.09194, 2023.

[21] A. Daniely, N. Linial, and S. Shalev-Shwartz. From average case complexity to improper
learning complexity. In Proceedings of the forty-sixth annual ACM symposium on Theory of
computing, pages 441–448, 2014.

[22] V. De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis. arXiv
preprint arXiv:2208.05314, 2022.

[23] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion schrödinger bridge with applica-
tions to score-based generative modeling. Advances in Neural Information Processing Systems,
34:17695–17709, 2021.

[24] S. Decatur, O. Goldreich, and D. Ron. Computational sample complexity. In Proceedings of
the tenth annual conference on Computational learning theory, pages 130–142, 1997.

[25] I. Diakonikolas and D. Kane. Non-gaussian component analysis via lattice basis reduction. In
Conference on Learning Theory, pages 4535–4547. PMLR, 2022.

[26] I. Diakonikolas, D. Kane, and L. Ren. Near-optimal cryptographic hardness of agnostically
learning halfspaces and relu regression under gaussian marginals. In International Conference
on Machine Learning, pages 7922–7938. PMLR, 2023.

[27] I. Diakonikolas, D. Kane, L. Ren, and Y. Sun. Sq lower bounds for non-gaussian component
analysis with weaker assumptions. Advances in Neural Information Processing Systems, 36,
2024.

[28] I. Diakonikolas, D. Kane, and N. Zarifis. Near-optimal sq lower bounds for agnostically learning
halfspaces and relus under gaussian marginals. Advances in Neural Information Processing
Systems, 33:13586–13596, 2020.

[29] I. Diakonikolas and D. M. Kane. Algorithmic High-Dimensional Robust Statistics. Cambridge
university press Cambridge, 2023.

[30] I. Diakonikolas, D. M. Kane, V. Kontonis, and N. Zarifis. Algorithms and sq lower bounds
for pac learning one-hidden-layer relu networks. In Conference on Learning Theory, pages
1514–1539. PMLR, 2020.

[31] I. Diakonikolas, D. M. Kane, and A. Stewart. Statistical query lower bounds for robust estimation
of high-dimensional gaussians and gaussian mixtures. In 2017 IEEE 58th Annual Symposium
on Foundations of Computer Science (FOCS), pages 73–84. IEEE, 2017.

[32] C. Dwork, M. P. Kim, O. Reingold, G. N. Rothblum, and G. Yona. Outcome indistinguishability.
In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pages
1095–1108, 2021.

[33] C. Dwork, M. P. Kim, O. Reingold, G. N. Rothblum, and G. Yona. Beyond bernoulli: Generating
random outcomes that cannot be distinguished from nature. In International Conference on
Algorithmic Learning Theory, pages 342–380. PMLR, 2022.

[34] V. Feldman, E. Grigorescu, L. Reyzin, S. S. Vempala, and Y. Xiao. Statistical algorithms and a
lower bound for detecting planted cliques. J. ACM, 64(2), 2017.

[35] J. H. Friedman and J. W. Tukey. A projection pursuit algorithm for exploratory data analysis.
IEEE Transactions on computers, 100(9):881–890, 1974.

[36] D. Gamarnik. The overlap gap property: A topological barrier to optimizing over random
structures. Proceedings of the National Academy of Sciences, 118(41):e2108492118, 2021.

[37] S. Goel, A. Gollakota, Z. Jin, S. Karmalkar, and A. Klivans. Superpolynomial lower bounds
for learning one-layer neural networks using gradient descent. In International Conference on
Machine Learning, pages 3587–3596. PMLR, 2020.

11


---Page Break---
[38] S. Goldwasser, M. P. Kim, V. Vaikuntanathan, and O. Zamir. Planting undetectable backdoors
in machine learning models. arXiv preprint arXiv:2204.06974, 2022.

[39] A. Gupte, N. Vafa, and V. Vaikuntanathan. Continuous lwe is as hard as lwe & applications to
learning gaussian mixtures. arXiv preprint arXiv:2204.02550, 2022.

[40] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–
778, 2016.

[41] U. Hébert-Johnson, M. Kim, O. Reingold, and G. Rothblum. Multicalibration: Calibration for
the (computationally-identifiable) masses. In International Conference on Machine Learning,
pages 1939–1948. PMLR, 2018.

[42] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural
information processing systems, 33:6840–6851, 2020.

[43] P. J. Huber. Projection pursuit. The annals of Statistics, pages 435–475, 1985.

[44] A. Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of
Machine Learning Research, 6(4), 2005.

[45] D. Jetchev and K. Pietrzak. How to fake auxiliary input. In Theory of Cryptography Conference,
pages 566–590. Springer, 2014.

[46] M. Kearns. Efficient noise-tolerant learning from statistical queries. Journal of the ACM
(JACM), 45(6):983–1006, 1998.

[47] D. Kunisky. Spectral Barriers in Certification Problems. PhD thesis, New York University,
2022.

[48] D. Kunisky, A. S. Wein, and A. S. Bandeira. Notes on computational hardness of hypothesis
testing: Predictions using the low-degree likelihood ratio. In Mathematical Analysis, its
Applications and Computation: ISAAC 2019, Aveiro, Portugal, July 29–August 2, pages 1–50.
Springer, 2022.

[49] J. B. Lasserre. Global optimization with polynomials and the problem of moments. SIAM
Journal on optimization, 11(3):796–817, 2001.

[50] H. Lee, J. Lu, and Y. Tan. Convergence for score-based generative modeling with polynomial
complexity. Advances in Neural Information Processing Systems, 35:22870–22882, 2022.

[51] H. Lee, J. Lu, and Y. Tan. Convergence of score-based generative modeling for general data
distributions. In International Conference on Algorithmic Learning Theory, pages 946–985.
PMLR, 2023.

[52] R. Lindner and C. Peikert. Better key sizes (and attacks) for lwe-based encryption. In Topics
in Cryptology–CT-RSA 2011: The Cryptographers’ Track at the RSA Conference 2011, San
Francisco, CA, USA, February 14-18, 2011. Proceedings, pages 319–339. Springer, 2011.

[53] X. Liu, L. Wu, M. Ye, and Q. Liu. Let us build bridges: Understanding and extending diffusion
generative models. arXiv preprint arXiv:2208.14699, 2022.

[54] Z. Ma and Y. Wu. Computational barriers in minimax submatrix detection. The Annals of
Statistics, 43(3), 2015.

[55] S. Mei and Y. Wu. Deep networks as denoising algorithms: Sample-efficient learning of
diffusion models in high-dimensional graphical models. arXiv preprint arXiv:2309.11420,
2023.

[56] D. Micciancio and O. Regev. Lattice-based cryptography. In Post-quantum cryptography, pages
147–191. Springer, 2009.

[57] J. Niles-Weed and P. Rigollet. Estimation of Wasserstein distances in the Spiked Transport
Model. Bernoulli, 28(4):2663 – 2688, 2022.

12


---Page Break---
[58] NIST.
Post-quantum
cryptography.
https://csrc.nist.gov/Projects/
Post-Quantum-Cryptography.

[59] P. A. Parrilo. Structured semidefinite programs and semialgebraic geometry methods in robust-
ness and optimization. California Institute of Technology, 2000.

[60] C. Peikert. A decade of lattice cryptography. Foundations and trends® in theoretical computer
science, 10(4):283–424, 2016.

[61] J. Pidstrigach. Score-based generative models detect manifolds. Advances in Neural Information
Processing Systems, 35:35852–35865, 2022.

[62] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image
generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

[63] O. Regev. On lattices, learning with errors, random linear codes, and cryptography. Journal of
the ACM (JACM), 56(6):1–40, 2009.

[64] S. Roman. The umbral calculus. Pure and Applied Mathematics, 1984.

[65] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis
with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 10684–10695, 2022.

[66] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gon-
tijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion mod-
els with deep language understanding. Advances in neural information processing systems,
35:36479–36494, 2022.

[67] S. Särkkä and A. Solin. Applied stochastic differential equations, volume 10. Cambridge
University Press, 2019.

[68] K. Shah, S. Chen, and A. Klivans. Learning mixtures of gaussians using the ddpm objective.
Advances in Neural Information Processing Systems, 36, 2024.

[69] S. Shalev-Shwartz, O. Shamir, and E. Tromer. Using more data to speed-up training time. In
Artificial Intelligence and Statistics, pages 1019–1027. PMLR, 2012.

[70] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning
using nonequilibrium thermodynamics. In International conference on machine learning, pages
2256–2265. PMLR, 2015.

[71] M. J. Song, I. Zadik, and J. Bruna. On the cryptographic hardness of learning single periodic
neurons. Advances in Neural Processing Systems (NeurIPS), 2021.

[72] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution.
Advances in neural information processing systems, 32, 2019.

[73] Y. Song and D. P. Kingma.
How to train your energy-based models.
arXiv preprint
arXiv:2101.03288, 2021.

[74] N. Stephens-Davidowitz. On the Gaussian Measure Over Lattices. PhD thesis, New York
University, USA, 2017.

[75] S. Tiegel. Hardness of agnostically learning halfspaces from worst-case lattice problems. In
The Thirty Sixth Annual Conference on Learning Theory, pages 3029–3064. PMLR, 2023.

[76] L. Trevisan, M. Tulsiani, and S. Vadhan. Regularity, boosting, and efficiently simulating every
high-entropy distribution. In 2009 24th Annual IEEE Conference on Computational Complexity,
pages 126–136. IEEE, 2009.

[77] R. Vershynin. High-dimensional probability: An introduction with applications in data science,
volume 47. Cambridge university press, 2018.

13


---Page Break---
[78] P. Vincent. A connection between score matching and denoising autoencoders. Neural compu-
tation, 23(7):1661–1674, 2011.

[79] M. J. Wainwright. Constrained forms of statistical minimax: Computation, communication and
privacy. In Proceedings of the International Congress of Mathematicians, pages 13–21, 2014.

[80] A. Wibisono, Y. Wu, and K. Y. Yang. Optimal score estimation via empirical bayes smoothing.
arXiv preprint arXiv:2402.07747, 2024.

[81] Y. Wu and J. Xu. Statistical problems with planted structures: Information theoretical and
computational limits. Information-Theoretic Methods in Data Science, 383:13, 2021.

[82] K. Y. Yang and A. Wibisono. Convergence of the inexact langevin algorithm and score-based
generative models in kl divergence. arXiv preprint arXiv:2211.01512, 2022.

[83] I. Zadik, M. J. Song, A. S. Wein, and J. Bruna. Lattice-based methods surpass sum-of-squares
in clustering. In Conference on Learning Theory, pages 1247–1248. PMLR, 2022.

[84] L. Zdeborová and F. Krzakala. Statistical physics of inference: Thresholds and algorithms.
Advances in Physics, 65(5):453–552, 2016.

[85] Y. Zhang, M. J. Wainwright, and M. I. Jordan. Lower bounds on the performance of polynomial-
time algorithms for sparse linear regression. In Conference on Learning Theory, pages 921–948.
PMLR, 2014.

14


---Page Break---
Appendices

A Additional Related Work
16

B
Proofs for Section 3
16

B.1
Proof of Theorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

B.2
Score functions of Gaussian pancakes . . . . . . . . . . . . . . . . . . . . . . . .
19

B.3
Distance from the standard Gaussian . . . . . . . . . . . . . . . . . . . . . . . . .
21

C Proofs for Section 4
24

C.1
Sample complexity of score estimation . . . . . . . . . . . . . . . . . . . . . . . .
24

C.2
Sample complexity of parameter estimation . . . . . . . . . . . . . . . . . . . . .
25

15


---Page Break---
A
Additional Related Work

Score estimation.
Several works have addressed the statistical question of sample complexity for
various data distributions and function classes used to estimate the score [16, 55, 80]. Wibisono et
al. [80] study score estimation for the class of subgaussian distributions with Lipschitz scores. By
establishing a minimax lower bound exhibiting a curse of dimensionality, they show that the exponen-
tial dependence on dimension is fundamental to this nonparametric distribution class, underscoring
the need for stronger assumptions on the data distribution for polynomial sample complexity. Chen
et al. [16] study neural network-based score estimation and derive finite-sample bounds for data
distributions that lie on a low-dimensional linear subspace, which circumvents the curse of dimension-
ality. Mei and Wu [55] study neural network-based score estimation for graphical models which are
intrinsically high dimensional. Assuming the efficiency of variational inference approximation to the
data-generating graphical model, they show that the score function can be efficiently approximated
by residual networks [40] and learned with polynomially many samples. Efficient score estimation
both in sample and computational complexity has been achieved by Shah et al. [68] for mixtures of
two spherical Gaussians using a shallow neural network architecture that matches the closed-form
expression of the score function.

Statistical-to-computational gaps.
Statistical-to-computational gaps have a rich history in statis-
tics, machine learning, and computer science [7, 85, 54, 69, 24]. Indeed, many high-dimensional
inference problems exhibit gaps between what is achievable statistically (with infinite computational
resources) and what is achievable under limited computation. Notable examples include sparse
PCA [7], sparse linear regression [85, 79], and learning one-hidden-layer neural networks over
Gaussian inputs [37, 30].

Since there are no known reductions from NP-hard problems, the gold standard for computational
hardness, to any average-case problem arising in statistical inference, alternative techniques have
been developed to provide rigorous evidence of hardness. These include proving lower bounds
for restricted classes of algorithms like sum of squares (SoS) [49, 59, 5] and statistical query
(SQ) algorithms [46, 34], which capture spectral, moment, and tensor methods, and reducing from
“canonical” problems believed to be hard on average, such as the planted clique [11] or random
k-SAT [21] problem. We refer the reader to surveys and monographs [84, 4, 81, 36, 48, 47, 29] for a
thorough overview of recent literature and diverse perspectives ranging from computer science and
information theory to statistical physics.

B
Proofs for Section 3

B.1
Proof of Theorem 3.1

Theorem B.1 (Theorem 3.1 restated). Let γ(d) > 1, σ(d) > 0 be any pair of sequences such
that σ ≥1/poly(d) and the corresponding (sequence of) (γ, σ)-Gaussian pancakes distributions
(Pu)u∈Sd−1 satisfies TV(Pu, Qd) > 1/2 for any d ∈N. Then, for any δ ∈(0, 1), there exists a
poly(d) · log(1/δ)-time algorithm with access to a score estimation oracle of L2-error O(1/√log d)
that solves the (γ, σ)-Gaussian pancakes problem with probability at least 1 −δ.

Proof. Let D ∈(Pu) ∪Q be the given data distribution. We first verify that assumptions A1-A3
hold, allowing us to apply the score-based Gaussianity test from Lemma 3.3. For any σ(d), γ(d) > 0
such that σ ≥1/poly(d), the (γ, σ)-Gaussian pancakes satisfy the Lipschitz score condition L ≤
poly(d) (Assumption A1) by Lemma B.5 and the second moment bound m2 ≤d (Assumption A2)
by Lemma B.2. In addition, KL(Pu ∥Q) ≤poly(d) (Lemma B.11), so Lemma 3.3 applies to
D with K = poly(d). Moreover, since TV(Pu, Q) > 1/2, Lemma 3.3 implies that if D ∈
(Pu)u∈Sd−1, there exist universal constants C1, C2 > 0 such that if the score estimates (skh)k∈[N]
satisfy EDkh∥skh(x) −∇log Dkh(x)∥2 ≤C1/ log d, then ∆≥C2/ log d.

Let τ = C2/ log d and η2 = min(τ/4, C1/ log d). Then, we have that with η-accurate score
estimates (skh), we have ∆≤η2 ≤τ/4 if D = Q and ∆≥τ otherwise. Note that η ≍1/√log d
and N = T/h ≍L2d log K ≤poly(d). Our proposed distinguisher A uses a finite-sample estimate
of ∆as a test statistic (Eq.(7)) using η-accurate score estimates (skh)k∈[N] and Nℓi.i.d. standard
Gaussian samples (z(k)
i
)(k,i)∈[N]×[ℓ] as follows. Later, it will be shown that setting the batch size ℓ

16


---Page Break---
to ℓ= poly(d) is sufficient for our distinguisher A.

ˆ∆= max
k∈[N]
ˆ∆(k) ,
where ˆ∆(k) = 1

ℓ

ℓ
X

i=1

skh(z(k)
i
) + 2πz(k)
i
2 .

The distinguisher A decides D = Q if ˆ∆≤τ/2 and D ∈(Pu)u∈Sd−1 otherwise. This procedure
runs in time O(Nℓ) and makes N = poly(d) queries to the score estimation oracle of L2-accuracy
O(1/√log d).

One issue in estimating ∆is that the score estimates (skh) only satisfy the mean guarantee with
respect to the forward process (Dkh), i.e., EDkh∥skh(x) −∇log Dkh(x)∥2 ≤η2. These guarantees
do not necessarily provide control over the concentration of random variables ∥st(z)+2πz∥2 induced
by z ∼Qd. Moreover, if D = Pu, then st(z) may behave erratically for z ∼Qd, taking on large
norms in low density areas between the pancakes, which may deter the estimation of ∆. We handle
this by truncating the score estimates.

Let M > 0 be some large number to be determined later. Define the truncated score ¯s by

¯st(x) =
st(x)
if ∥st(x)∥≤M ,
0
otherwise .

We claim that using the truncated score estimates (¯st) in place of (st) introduces negligible (in data
dimension d) additional L2 score estimation error with respect to the forward process (Dt) compared
to the original score estimates (st). Hence, Lemma 3.3 applies with the uniformly bounded vector
fields (˜skh)k∈[N] as the L2-accurate score estimates for (Dkh)k∈[N]. For any (discretized) time
0 ≤t ≤T and distribution Dt from the forward process,

EDt∥¯st(x) −∇log Dt(x)∥2 = EDt

∥st(x) −∇log Dt(x)∥2 · 1[∥st(x)∥≤M]


+ EDt

∥∇log Dt(x)∥2 · 1[∥st(x)∥> M]

.

The second term on the RHS can be upper bounded by

EDt

∥∇log Dt(x)∥2 · 1[∥st(x)∥> M]


= EDt

∥∇log Dt(x)∥2 · 1[(∥st(x)∥> M) ∧(∥∇log Dt(x)∥> M/2)]


+ EDt

∥∇log Dt(x)∥2 · 1[(∥st(x)∥> M) ∧(∥∇log Dt(x)∥≤M/2)]


≤EDt

∥∇log Dt(x)∥2 · 1[∥∇log Dt(x)∥> M/2]


+ EDt

∥st(x) −∇log Dt(x)∥2 · 1[∥st(x)∥> M]

,
(8)

where in Eq.(8) we used the fact that if ∥st(x)∥> M and ∥∇log Dt(x)∥≤M/2, then
∥∇log Dt(x)∥≤M/2 ≤∥st(x) −∇log Dt(x)∥.

Putting things together, we have

EDt∥¯st(x) −∇log Dt(x)∥2 ≤EDt∥st(x) −∇log Dt(x)∥2

+ EDt

∥∇log Dt(x)∥2 · 1[∥∇log Dt(x)∥> M/2]

.

It remains to bound E[∥∇log Dt(x)∥2 · 1[∥∇log Dt(x)∥> M/2]] which depends only on the
distribution Dt. We choose M ≍(
√

d + 1/σ2). If D = Qd, then Dt = Qd for any t > 0, and
by Cauchy-Schwarz and norm concentration (see e.g., [77, Theorem 3.1.1]), there exists a constant
C > 0 such that

EQd

∥x∥2 · 1[∥x∥> M/2]

≤EQd∥x∥4 · Pr
Qd[∥x∥> M/2] ≤exp(−CM 2) .

On the other hand, if D = Pu, then by Lemma B.4, Eq.(13) and the triangle inequality

∥∇log Pu(x)∥=

x + (T σ
γ )′(⟨x, u⟩)
T σ
γ (⟨x, u⟩) u

 ≤∥x∥+ 8π(1 + 1/σ2) .

17


---Page Break---
Using a similar concentration argument, we have

EPu

∥∇log Pu(x)∥2 · 1[∥∇log Pu(x)∥> M/2]


≲EPu

(∥x∥2 + 1/σ4 + 1) · 1[∥x∥> M/2 −8π(1 + 1/σ2)]


≲exp(−O(M 2)) .
(9)

The OU process applied to Pu only increases the σ parameter, so the upper bound in Eq.(9) holds
uniformly over the forward process (Dkh)k∈[N] if D = Pu. Thus, choosing M ≍(
√

d + 1/σ2) =
poly(d) as the truncation threshold suffices to ensure that for all discretized time steps t = kh, the
L2 score estimation error with respect to the forward process (Dt) introduced by truncating the score
estimate st to ¯st is negligible in d. Therefore, we apply Lemma 3.3 with (˜skh)k∈[N] as the score
estimates for the forward process (Dkh)k∈[N].

Since ∥¯st(z) + 2πz∥2, where z ∼Qd, is a random variable with subexponential norm O(M 2), we
can apply Bernstein’s inequality [77, Corollary 2.8.3] to the i.i.d. sum ˆ∆(k). Thus, for any k ∈[N],
ℓ≍(M 4/ε2) · log(N/δ) Gaussian samples suffice to guarantee accurate estimation of the population
mean ∆(k) with additive error less than ε with probability at least 1 −δ/N. By a union bound,
with probability at least 1 −δ, this holds for all k ∈[N]. Setting ε = τ/8 = Θ(1/ log d) and
recalling that N = poly(d), we have a distinguisher A for the Gaussian pancakes problem that makes
N = poly(d) queries to the score estimation oracle, runs in time Nℓ= poly(d) · log(1/δ), and is
correct with probability at least 1 −δ.

Lemma B.2 (Second moment of Gaussian pancakes). For any γ, σ > 0, and u ∈Sd−1, the
(γ, σ)-Gaussian pancakes Pu satisfies

Ex∼Pu[∥x∥2] ≤d

2π .

Proof. Without loss of generality, we assume u = e1. Then,

Ex∼Pu∥x∥2 = Ex∼Aσ
γ [x2] + (d −1)Ez∼Qz2 = Ex∼Aσ
γ [x2] + d −1

2π
.

In addition,

Ex∼Aσ
γ [x2] = Ex∼AγEz∼N(0,1/(2π))
 
x/
p

1 + σ2 + σz/
p

1 + σ22

=
1
1 + σ2 · Ex∼Aγ[x2] +
σ2

1 + σ2 · 1

2π .

Thus, it suffices to establish an upper bound on Ex∼Aγx2, i.e., the second moment of the discrete
Gaussian on (1/γ)Z. Using the Poisson summation formula (Lemma 2.1) and the fact that the Fourier
transform of x2ρ(x) is (1/(2π) −y2)ρ(y),

Ex∼Aγ[x2] =
1
ρ((1/γ)Z)

X

x∈(1/γ)Z
x2ρ(x)

=
γ
ρ((1/γ)Z)

X

y∈γZ

 1

2π −y2

ρ(y)

= 1

2π ·
γ
ρ((1/γ)Z) · ρ(γZ) −
γ
ρ((1/γ)Z)

X

y∈γZ
y2ρ(y)
(10)

< 1

2π

where in Eq.(10), we used the fact that ρ(γZ) = ρ((1/γ)Z)/γ and that the second term is positive.

Since Ex∼Aσ
γ [x2] is a convex combination of Ex∼Aγ[x2] and 1/(2π), the conclusion follows.

18


---Page Break---
B.2
Score functions of Gaussian pancakes

In this section, we show that score functions of Gaussian pancakes distributions are Lipschitz with
respect to x ∈Rd. The score function of Pu, the (γ, σ)-Gaussian pancakes distribution with secret
direction u ∈Sd−1, admits the following analytical expression.

∇log Pu(x) = −2πx + ∇log T(⟨x, u⟩) = −2πx + (T σ
γ )′(⟨x, u⟩)
T σ
γ (⟨x, u⟩) u .
(11)

We use Banaszczyk’s theorems on the Gaussian mass of lattices [3] to upper bound the Lipschitz
constant of (T σ
γ )′/T σ
γ in terms of σ.

Theorem B.3 ([74, Corollary 1.3.5]). For any lattice L ⊂Rd, parameter s > 0, shift t ∈Rd, and
radius r >
p

d/(2π) · s,

ρs((L −t) \ rBd
2) < exp(−πx2)ρs(L) ,

where x := r/s −
p

d/(2π) and Bd
2 denotes the Euclidean ball in Rd.

Lemma B.4 ([74, Lemma 1.3.10]). For any lattice L ⊂Rd, parameter s > 0, and shift t ∈Rd,

exp(−π · dist(t, L)2/s2) · ρs(L) ≤ρs(L −t) ≤ρs(L) .

Lemma B.5 (Lipschitzness of ∇log Pu). For any γ > 1, σ > 0, s = σ/
√

1 + σ2, and u ∈Sd−1,
the score function of the (γ, σ)-Gaussian pancakes distribution Pu satisfies the Lipschitz condition:

∥∇log Pu(y) −∇log Pu(x)∥≲(1/s4)∥y −x∥
for any x, y ∈Rd ,
(12)

and the likelihood ratio T σ
γ of the σ-smoothed discrete Gaussian Aσ
γ relative to N(0, 1/(2π)) satisfies
the uniform bound:

(T σ
γ )′(z)
T σ
γ (z)

 ≤8π

s2
for any z ∈R .
(13)

Proof. It suffices to analyze the Lipschitz constant of the univariate function (T σ
γ )′/T σ
γ : R →R,
which we denote by f = (T σ
γ )′/T σ
γ since for any x ̸= y ∈Rd,

∥∇log Pu(y) −∇log Pu(x)∥≤2π∥y −x∥+
(f(⟨y, u⟩) −f(⟨x, u⟩))u


≤2π∥y −x∥+ |f(⟨y, u⟩) −f(⟨x, u⟩)|

|⟨y, u⟩−⟨x, u⟩|
· |⟨y −x, u⟩|

≤

 

2π + sup
a,b∈R


f(b) −f(a)

b −a



!

∥y −x∥.

We bound the Lipschitz constant of the function f(z) by demonstrating a uniform upper bound on
the absolute value of its derivative f ′(z) := (d/dz)f(z). For notational convenience, we omit the
parameters γ, σ when denoting the likelihood ratio T. The derivative f ′ is given by

f ′ =
T ′

T

′
= (T ′′)T −(T ′)2

T 2
= T ′′

T −
T ′

T

2
.
(14)

Hence, |f ′(z)| ≤|(T ′′/T)(z)| + |(T ′/T)(z)|2 for any z ∈R. We prove uniform upper bounds on
the two RHS terms, starting with |T ′/T|. Using the definition of the likelihood ratio T (Eq.(5)), we
have

T ′(z)

T(z) = 2π

σ2 ·
P

k∈Z −(z −
√

1 + σ2k/γ)ρσ(z −
√

1 + σ2k/γ)
P
k∈Z ρσ(z −
√

1 + σ2k/γ)

= 2π

σs ·
E
x∼DL−˜z,s[x] ,
(15)

19


---Page Break---
where ˜z = z/
√

1 + σ2, s = σ/
√

1 + σ2, L = (1/γ)Z, and DL−˜z,s is the discrete Gaussian
distribution of width s on the lattice coset L −˜z.

We upper bound E[|x|] via a tail bound on DL−˜z,s. Let r > 0 be any number and denote t :=
r/s −
p

1/(2π). By Theorem B.3 and Lemma B.4,

ρs((L −˜z) \ rB2) < exp(−πt2)ρs(L)

< exp(−πt2) exp(π · dist(˜z, L)2/s2)ρs(L −˜z)

= exp(−π((r/s −
p

1/(2π))2 −1/(2sγ)2))ρs(L −z) ,

where we used the fact that dist(˜z, L) ≤1/(2γ) for any ˜z ∈R in the last line.

Thus, for r/(2s) ≥
p

1/(2π) + 1/(2sγ),

Pr
x∼DL−˜z,s[|x| ≥r] = ρs((L −˜z) \ rB2)

ρs(L −˜z)
≤exp(−πr2/(2s)2) .
(16)

Denote r0 := s
p

2/π + 1/γ. Note that r0 <
p

2/π + 1/γ since s ∈[0, 1). Then,

Ex∼DL−˜z,s|x| =
Z

r
Pr[|x| ≥r]dr ≤r0 +
Z

r>r0
Pr[|x| ≥r]dr

≤r0 +
Z

r>r0
exp(−πr2/(2s)2)dr

≤r0 + 2s

≤(2 +
p

2/π)s + 1/γ .
(17)

Therefore by Eq.(15) and (17), for any z ∈R

T ′(z)

T(z)

 ≤2π

σs((2 +
p

2/π)s + 1/γ) .
(18)

Eq.(13) in the statement of Lemma B.5 follows immediately from the fact that s < min(1, σ) and
γ > 1. Next, we demonstrate a uniform upper bound on T ′′/T. The analytical expression of T ′′/T
is given by

T ′′(z)

T(z) =
2π

σ2

2
·
P

k∈Z(z −
√

1 + σ2k/γ)2ρσ(z −
√

1 + σ2k/γ)
P

k∈Z ρσ(z −
√

1 + σ2k/γ)
−σ2

2π



=
2π

σs

2
E
x∼DL−˜z,s x2 −s2

2π


,

where ˜z = z/
√

1 + σ2, s = σ/
√

1 + σ2, and L = (1/γ)Z.

To uniformly bound |T ′′/T|, we upper bound the second moment of DL−˜z,s. Again, let r0 =
s
p

2/π + 1/γ. Using the tail bound from Eq.(16) and the fact that (a + b)2 ≤2a2 + 2b2 for any
a, b ∈R

E
x∼DL−z,σ x2 =
Z ∞

0
r Pr[|x| ≥r]dr ≤r2
0/2 +
Z

r≥r0
r Pr[|x| ≥r]dr

≤r2
0/2 +
Z

r≥r0
r exp(−πr2/(2s2))dr

≤r2
0/2 + s2/π

≤s2 + 1/γ2 .
(19)

20


---Page Break---
Applying Eq.(17) and Eq.(19) to the expression for f ′ = T ′/T (Eq.(14)) and using the fact that
γ > 1 and s = σ/
√

1 + σ2 ≤min(1, σ), we have

∥f ′∥∞≤∥T ′′/T∥∞+ ∥(T ′/T)2∥∞

≤
2π

σs

2 
(1 + 1/2π)s2 + 1/γ2 + ((2 +
p

2/π)s + 1/γ)2

≤100π2

s4
.

Therefore, ∇log Pu(x) is O(1/s4)-Lipschitz since 2π + ∥f ′∥∞≲1/s4.

B.3
Distance from the standard Gaussian

We now prove lower (Lemma B.9) and upper (Lemma B.10) bounds on the TV distance between
Gaussian pancakes distributions (Pu) and the standard Gaussian Q. We also show that the KL
divergence is upper bounded by poly(d) for Gaussian pancakes with σ ≥1/poly(d) (Lemma B.11).

The following fact reduces the d-dimensional problem of bounding TV(Pu, Qd) to a one-dimensional
problem of bounding TV(Aσ
γ, Q), where Aσ
γ is the σ-smoothed discrete Gaussian on (1/γ)Z and
Q = Q1. Without loss of generality, assume u = e1. Then, by the L1-characterization of the TV
distance, we have

TV(Pu, Qd) = 1

2

Z
|Pu(x) −Qd(x)|dx = 1

2

Z
Qd(x)|T σ
γ (x1) −1|dx

= 1

2

Z
|Aσ
γ(x1) −Q(x1)|dx1

= TV(Aσ
γ, Q) .

Hence, it suffices to demonstrate bounds on TV(Aσ
γ, Q). The same applies to the KL divergence
since KL(Pu ∥Qd) = KL(Aσ
γ ∥Q). We first demonstrate a lower bound on the total variation
distance. To this end, we introduce the periodic Gaussian distribution and its useful properties. The
key lemma is Lemma B.9.

Definition B.6 (Periodic Gaussian distribution). For any one-dimensional lattice L ⊂R, we define
the periodic Gaussian distribution ΨL,s : R →R≥0 as follows.

ΨL,s(z) := 1

s

X

x∈L
ρs(x −z) = ρs(L −z)/s .

We can regard the function ΨL,s as a distribution for the following reason: Let λ1(L) denote the
spacing of L. Then, ΨL,s restricted to [0, λ1(L)] is a probability density since

Z λ1(L)

0
ΨL,s(z)dz = 1

s

Z λ1(L)

0

X

x∈L
ρs(x −z) = 1

s

X

x∈L

Z λ1(L)

0
ρs(x −z)dz = 1

s

Z ∞

−∞
ρs(z)dz = 1 .

Lemma B.7 (Mill’s inequality [77, Proposition 2.1.2]). Let z ∼N(0, 1). Then for all t > 0, we have

P(|z| ≥t) =

r

2
π

Z ∞

t
e−x2/2dx ≤1

t ·

r

2
π e−t2/2 .

Lemma B.8 (Periodic Gaussian density bound [71, Claim I.6]). For any s > 0 and any z ∈[0, 1)
the periodic Gaussian density ΨZ,s : [0, 1) →R≥0 satisfies

|ΨZ,s(z) −1| ≤2(1 + 1/(πs))e−πs2 .

21


---Page Break---
Proof. By the Poisson summation formula (Lemma 2.1),

ΨZ,s(z) = 1

s

X

x∈Z
ρs(x −z)

=
X

y∈Z
e−2πizy · ρ1/s(y)

= 1 +
X

y∈Z\{0}
e−2πizy · ρ1/s(y) .

Since |eia| ≤1 for any a ∈R, we have

|ΨZ,s(z) −1| ≤
X

y∈Z\{0}
|e−2πizy| · ρ1/s(y) ≤
X

y∈Z\{0}
ρ1/s(y)

≤2

e−πs2 +
Z ∞

1
e−πs2t2dt


≤2(1 + 1/(πs))e−πs2 .

Lemma B.9 (TV lower bound). There exists a constant C > 0 such that for any γ > 1 and σ > 0,
if γσ < C, then TV
 
Aσ
γ, N(0, 1/(2π))

> 1/2, where Aσ
γ is the σ-smoothed discrete Gaussian on
(1/γ)Z.

Proof. For ease of notation, we denote N(0, 1/(2π)) by Q. Since TV(Aσ
γ, Q) = supS∈F |Aσ
γ(S) −
Q(S)|, where F is the Borel σ-algebra on R, it suffices to find a measurable set S ⊂R such that
Aσ
γ(S) −Q(S) > 1/2.

Let C = 1/(12√2 log 2) be the constant in the statement of Lemma B.9. Let δ ∈(0, 1/4) be the
smallest number satisfying the condition γσ ≤δ/(3
p

log(1/δ)). Such δ > 0 always exists under the
given assumptions since δ/
p

log(1/δ) is increasing in δ and δ = 1/4 satisfies the condition (thanks
to our choice of C). We claim that the set S defined below witnesses the TV lower bound.

S :=
n
z ∈R | dist(z, L) ≤
σ
√

1 + σ2 ·
p

log(1/δ)
o
,

where L = (1/γ
√

1 + σ2)Z.

We show a lower bound for Aσ
γ(S) and an upper bound for Q(S). Using the mixture form of the
density of Aσ
γ (Eq.(4)) and Mill’s tail bounds for the univariate Gaussian (Lemma B.7), we have that
for each Gaussian component in the mixture Aσ
γ, at least 1 −δ fraction of its probability mass is
contained in S. This is because the component means precisely form the one-dimensional lattice L
and S contains all significant neighborhoods of L. Thus, Aσ
γ(S) ≥1 −δ.

Now we show an upper bound for Q(S). Recall from Definition B.6 the density of the periodic
Gaussian distribution ΨZ,s. Since S is a periodic set, its mass Q(S) is equal to ΨZ,s( ˜S ∩Z), where
s = γ
√

1 + σ2 and

˜S =
n
z ∈R | dist(z, Z) ≤γσ
p

log(1/δ)
o
.

Since s = γ
√

1 + σ2 > 1, by Lemma B.8 for any z ∈[0, 1)

|ΨZ,s(z) −1| ≤4e−πs2 < 1/2 .

Since γσ ≤δ/(3
p

log(1/δ)), it follows that

Q(S) = ΨZ,s( ˜S ∩[0, 1]) ≤
 
1 + 4e−πs2
· 2γσ
p

log(1/δ) < 3γσ
p

log(1/δ) ≤δ .

Therefore,
TV(Aσ
γ, Q) ≥Aσ
γ(S) −Q(S) > 1 −2δ > 1/2 .

22


---Page Break---
We now establish upper bounds on TV(Pu, Qd) via upper bounds on TV(Aσ
γ, Q). Lemma B.10
provides a tighter upper bound when min(γ, γσ) = ω(√log d). In this regime, the sequence
(TV(Pu, Qd))d∈N is negligible in d.
It is worth noting that Lemma B.10 is not tight since
TV(Aσ
γ, Q) →0 as σ →∞, whereas the upper bound only converges to e−πγ2.

On the other hand, Lemma B.11 provides an upper bound on KL(Pu ∥Qd) which is useful when
σ is large. Note that the KL divergence upper bounds the TV distance through Pinsker’s or the
Bretagnolle–Huber inequality (see e.g., [14]).

Lemma B.10 (TV upper bound). For any γ > 1 and σ > 0 such that σ ≥2/γ, the following holds
for the σ-smoothed discrete Gaussian Aσ
γ and Q = N(0, 1/(2π)).

TV(Aσ
γ, Q) ≲e−πs2 ,

where s = γσ/
√

1 + σ2.

Proof. By the L1-characterization of the TV distance,

TV(Aσ
γ, Q) = 1

2

Z
|Aσ
γ(x) −Q(x)|dx = 1

2

Z
Q(x)|T σ
γ (x) −1|dx ,

where the likelihood ratio T σ
γ is given by (see Eq.(5))

T σ
γ (x) =

√

1 + σ2

σρ((1/γ)Z)

X

k∈Z
ρσ(x −
p

1 + σ2k/γ)

=

√

1 + σ2

σρ((1/γ)Z)

X

k∈Z
ργσ/
√

1+σ2(γx/
p

1 + σ2 −k) .

The likelihood ratio is also a re-scaled version of the periodic Gaussian density since

ΨZ,s(t) = ρs(Z −t)

s
= 1

s

X

k∈Z
ρs(k −t) .
(20)

Plugging in s = γσ/
√

1 + σ2 = γ/
p

(1/σ2) + 1 to Eq.(20), we have

T σ
γ (x) =
γ
ρ((1/γ)Z) · ΨZ,s(γx/
p

1 + σ2) .

The assumption σ ≥2/γ implies that s ≥1. Thus, by Lemma B.8

|T σ
γ (x) −1| ≤
γ
ρ((1/γ)Z)

ΨZ,s(γx/
p

1 + σ2) −1
 +

γ
ρ((1/γ)Z) −1


≤
γ
ρ((1/γ)Z) · 2(1 + 1/(πs))e−πs2 +

γ
ρ((1/γ)Z) −1


≤
γ
ρ((1/γ)Z) · 3e−πs2 +

γ
ρ((1/γ)Z) −1
 .
(21)

Since ρ((1/γ)Z) = ργ(Z) = γΨZ,γ(0) and γ > 1, by Lemma B.8 applied to ΨZ,γ,

ρ((1/γ)Z)

γ
−1
 ≤2(1 + 1/(πγ))e−πγ2 < 3e−πγ2 < 1/4 .

We may thus write ρ((1/γ)Z)/γ = 1 + ε for some ε ∈R such that |ε| ≤3e−πγ2 < 1/4. Then,

γ
ρ((1/γ)Z) −1
 =

1
1 + ε −1
 =

ε
1 + ε

 < (4/3)ε .

23


---Page Break---
Applying the above inequalities to Eq.(21), we have that for any x ∈R,

|T σ
γ (x) −1| ≤
γ
ρ((1/γ)Z) · 3e−πs2 +

γ
ρ((1/γ)Z) −1


< (1 + 3e−πγ2) · 3e−πs2 + 4e−πγ2

≤4(e−πs2 + e−πγ2) .

Since 0 < s < γ, it follows that

TV(Aσ
γ, Q) ≤4(e−πs2 + e−πγ2) ≤8e−πs2 .

Lemma B.11 (KL upper bound). For any γ > 0 and σ > 0, the following holds for the σ-smoothed
discrete Gaussian Aσ
γ and Q = N(0, 1/(2π)).

KL(Aσ
γ||Q) ≤log(
p

1 + σ2/σ) ≤
1
2σ2 .

Expressed in terms of the time with respect to the OU process e−t = 1/
√

1 + σ2, for any t > 0

KL(Aσ
γ ∥Q) ≤
e−2t

2(1 −e−2t) .

Proof. By Jensen’s inequality and the fact that for any x ∈R, T σ
γ (x) ≤T σ
γ (0).

KL(Aσ
γ||Q) = Ex∼Aσ
γ [log T σ
γ (x)] ≤log Ex∼Aσ
γ T σ
γ (x)

≤log T σ
γ (0)

= log

√

1 + σ2

σργ(Z) · ρs(Z) ,

where s = γσ/
√

1 + σ2 < γ.

If 0 ≤s1 ≤s2, then ρs1(Z) ≤ρs2(Z). Hence, ρs(Z)/ργ(Z) ≤1. Using the fact that log(1+a) ≤a
for any a > −1, we have

KL(Aσ
γ ∥Q) ≤log(
p

1 + σ2/σ) = (1/2) log(1 + 1/σ2) ≤1/(2σ2) .

The second part of the lemma follows straightforwardly from the relation σ/
√

1 + σ2 =
√

1 −e−2t.
Using the fact that log(1 −a) ≥−a/(1 −a) for any a < 1, for any t > 0 we have

KL(Aσ
γ ∥Q) ≤log(1/
p

1 −e−2t) = −(1/2) log(1 −e−2t) ≤
e−2t

2(1 −e−2t) .

C
Proofs for Section 4

C.1
Sample complexity of score estimation

We show that score estimation reduces to parameter estimation for Gaussian pancakes. Given that
the sample complexity of parameter estimation for Gaussian pancakes is polynomial in the relevant
problem parameters (Theorem 4.2), our reduction implies that the sample complexity of score
estimation is polynomial as well.
Lemma C.1 (Lemma 4.1 restated). For any γ > 1, σ > 0, let Pu be the (γ, σ)-Gaussian pancakes
distribution with secret direction u ∈Sd−1. Given any η ∈(0, 1) and ˆu ∈Sd−1 such that
1 −⟨ˆu, u⟩2 ≤η2, the score estimate ˆs(x) = −2πx + ∇log T σ
γ (⟨x, ˆu⟩) satisfies

Ex∼Pu∥ˆs(x) −s(x)∥2 ≲max(1, 1/σ8) · η2d ,

where s(x) = −2πx + ∇log T σ
γ (⟨x, u⟩) is the score function of Pu.

24


---Page Break---
Proof. For simplicity, we omit super and subscripts of the likelihood ratio T σ
γ and simply denote it
by T. Let ˆu be an estimate satisfying 1 −⟨ˆu, u⟩2 ≤η2 and denote ˆu = ⟨ˆu, u⟩u + w. Note that
w ∈Rd is orthogonal to u and ∥w∥2 = 1 −⟨ˆu, u⟩2 ≤η2. Then, we have

s(x) −ˆs(x) = T ′(⟨x, u⟩)

T(⟨x, u⟩) u −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩ˆu

=

 
T ′(⟨x, u⟩)

T(⟨x, u⟩) −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩⟨u, ˆu⟩

!

u −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩w

By the triangle inequality and the fact that (a + b)2 ≤2a2 + 2b2 for any a, b ∈R, we have

∥s(x) −ˆs(x)∥2 ≤2

 
T ′(⟨x, u⟩)

T(⟨x, u⟩) −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩⟨u, ˆu⟩

!2

+ 2

 
T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

η2

≤4

 
T ′(⟨x, u⟩)

T(⟨x, u⟩) −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

+ 4

 
T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

(1 −|⟨u, ˆu⟩|)2 + 2

 
T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

η2

≲

 
T ′(⟨x, u⟩)

T(⟨x, u⟩) −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

+

 
T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

η2
(22)

By Lemma B.5 and the fact that (1 + σ2)/σ2 ≤2 max(1, 1/σ2), we know that the Lipschitz
constant L of T ′/T satisfies L ≲max(1, 1/σ4). Furthermore, by Eq. (13) in Lemma B.5, we have
∥T ′/T∥∞≲max(1, 1/σ2). Applying these upper bounds to Eq.(22),

∥s(x) −ˆs(x)∥2 ≲

 
T ′(⟨x, u⟩)

T(⟨x, u⟩) −T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

+

 
T ′(⟨x, ˆu⟩)

T(⟨x, ˆu)⟩

!2

η2

≲max(1, 1/σ8)(⟨x, u −ˆu⟩2 + η2)

≤max(1, 1/σ8)(∥x∥2∥u −ˆu∥2 + η2)

≤max(1, 1/σ8) · η2(∥x∥2 + 1) .

Since Ex∼Pu∥x∥2 ≤d by Lemma B.2, it follows that Ex∼Pu∥s(x) −ˆs(x)∥2 ≲max(1, 1/σ8) ·
η2d .

C.2
Sample complexity of parameter estimation

For parameter estimation, we design a contrast function g : R →R such that the (population)
functionals G and E, defined below, are monotonic. For any γ > 0, u ∈Sd−1, and (γ, 0)-Gaussian
pancakes Pu, we define

G(σ) := Ex∼Aσ
γ [g(x)]
(23)

E(v) := Ex∼Pu[g(⟨x, v⟩)] .
(24)

Note that E(v) = G(σ), where σ2 = (1 −⟨u, v⟩2)/⟨u, v⟩2. We choose g so that G(σ) is de-
creasing in σ and E(v) is increasing in ⟨u, v⟩2. We use g = T β
γ for some appropriately chosen
β > 0. The monotonicity property of T β
γ is shown in Lemma C.3. Thus, given two candidate
directions v1, v2, if E(v1) ≥E(v2), then ⟨u, v1⟩2 ≥⟨u, v2⟩2. This suggests the projection pursuit-
based estimator ˆu = arg maxv∈C ˆE(v), where C is an η-net of the parameter space Sd−1 and
ˆE(v) = (1/n) Pn
i=1 T β
γ (⟨xi, v⟩) is the empirical version of E. The main theorem of this section
(Theorem 4.2) shows that n = poly(d, γ, 1/η) samples is sufficient for achieving L2-error η using
the estimator ˆu. We start with a useful fact about σ-smoothed likelihood ratios T σ
γ .
Claim C.2. For any β > 0, σ ≥0, and x ∈R,

Ez∼Q


T β
γ


1
√

1 + σ2 x +
σ
√

1 + σ2 z

= T s
γ(x) ,

25


---Page Break---
where s =
p

(1 + β2)(1 + σ2) −1 and T σ
γ denotes the likelihood ratio of the σ-smoothed discrete
Gaussian Aσ
γ on (1/γ)Z with respect to Q = N(0, 1/(2π)) (see Eq. (5)).

Proof. Let ˜x = x/
√

1 + σ2, ˜σ = σ/
√

1 + σ2, and ck(˜x) = ˜x −
p

1 + β2k/γ. Then,

EQ

T β
γ
 
˜x + ˜σz

=

p

1 + β2

βρ((1/γ)Z)

Z
ρ(z)
X

k∈Z
ρβ


˜x + ˜σz −
p

1 + β2k/γ

dz

=

p

1 + β2

βρ((1/γ)Z)

X

k∈Z

Z
ρβ/√

β2+˜σ2(z −ck(˜x))dz · ρ√

β2+˜σ2(ck(˜x))

=

p

1 + β2
p

β2 + ˜σ2 · ρ((1/γ)Z)

X

k∈Z
ρ√

β2+˜σ2(˜x −
p

1 + β2k/γ) .

Plugging in ˜x = x/
√

1 + σ2 and ˜σ = σ/
√

1 + σ2 gives us

EQ


T β
γ


1
√

1 + σ2 x +
σ
√

1 + σ2 z

=

√

1 + s2

sρ((1/γ)Z)

X

k∈Z
ρs(x −
p

1 + s2k/γ) = T s
γ(x) ,

where s =
p

(1 + β2)(1 + σ2) −1.

Lemma C.3 (Monotonicity). Given any γ > 0 and β > 0, define G(σ) := Ex∼Aσ
γ [T β
γ (x)], where
T σ
γ denotes the likelihood ratio of the σ-smoothed discrete Gaussian Aσ
γ on (1/γ)Z with respect to
N(0, 1/(2π)). Then, for any σ > 0, we have G′(σ) < 0.

Proof. Let T 0
γ (x) = P∞
k=0 α2kh2k(x) be the (formal) Hermite expansion of T 0
γ (x), where (hk)k∈N
form an orthonormal sequence with respect to ⟨·, ·⟩Q. By Claim C.2, for any σ ≥0

T σ
γ (x) =

∞
X

k=0
α2k


1
1 + σ2

k
h2k(x) .
(25)

Using the orthonormality of (hk), for any σ > 0

G(σ) = Ex∼Aσ
γ [T β
γ (x)] = ⟨T β
γ , T σ
γ ⟩Q =

∞
X

k=0
α2
2k ·
1
(1 + β2)k(1 + σ2)k

G′(σ) = −

∞
X

k≥1
α2
2k ·
1
(1 + β2)k ·
2kσ
(1 + σ2)k+1 < 0 .

Lemma C.4 (Non-trivial Hermite coefficient). Let (hk) be the normalized Hermite polynomials with
respect to ⟨·, ·⟩Q. For any γ > 1 such that πγ2 ∈N and ℓ∈N, it holds that
Ex∼Aγh2πℓ2γ2(x)
 ≥
1
√2πℓγ ,

where Aγ is the discrete Gaussian on (1/γ)Z.

Proof. Let (Hk) be the unnormalized Hermite polynomials defined by

Hk(x)ρ(x) = (−1)k dk

dxk ρ(x) .

Using the relation between the Fourier transform and differentiation, we have

F{Hk(x)ρ(x)} = F

(−1)k dk

dxk ρ(x)

= (−2πiy)kρ(y) .

26


---Page Break---
Let hk = ckHk be the normalized Hermite polynomials, where ck =
p

(2π)kk! (see e.g., [64,
Chapter 4.2.1]). By the Poisson summation formula (Lemma 2.1), we have

Ex∼Aγhk(x) =
1
ρ((1/γ)Z)

X

x∈(1/γ)Z
ckHk(x)ρ(x)

=
γck
ρ((1/γ)Z)

X

y∈γZ
(−2πiy)kρ(y) .

We now analyze the maximum among the terms inside the sum. Let f(y) = ykρ(y). Note that

f(y) = ykρ(y) = (2π)k exp(−πy2 + k log y) .

The exponent in the above expression is maximized at 2π(y∗)2 = k. This maximum is indeed
achieved in the sum since we can choose y∗= ℓγ, which is permissible given the assumption
πγ2 ∈N. Hence, the maximum value of f(y) is

f(y∗) = exp(−k/2 + (k/2) log(k/2π)) = (k/2eπ)k/2 .

Plugging in the value ck = 1/
p

(2π)kk! and using the Stirling lower bound k! ≥
√

2πk(k/e)k for
all k ∈N,
Ex∼Aγhk(x)
 ≥
2γ
ρ((1/γ)Z)ck(2π)k(k/2eπ)k/2

=
2γ
ρ((1/γ)Z)ck(2πk/e)k/2

=
2γ
ρ((1/γ)Z) · (k/e)k/2

√

k!

≥
2γ
ρ((1/γ)Z) ·
1
(2πk)1/4 .

In addition, we have that

ρ((1/γ)Z) = γρ(γZ) ≤γρ(Z) ≤2γ .

Plugging in k = 2πℓ2γ2, we therefore have

Ex∼Aγhk(x)
 ≥
1
√2πℓγ .

Corollary C.5 (Non-trivial Hermite coefficient, rounded degree). Let (hk) be the normalized Hermite
polynomials with respect to ⟨·, ·⟩Q. For any γ > 1, ℓ∈N, and k = 2⌊ℓ2πγ2⌋, it holds that

Ex∼Aγhk(x)
 ≥
1
e√2πℓγ ,

where Aγ is the discrete Gaussian on (1/γ)Z.

Proof. Let ℓ2πγ2 −⌊ℓ2πγ2⌋= α. Then, α ∈[0, 1) and k = ℓ2(2πγ2) −2α. Similar to the proof
of Lemma C.4, we apply the Poisson summation formula (Lemma 2.1) as follows.

Ex∼Aγhk(x) =
1
ρ((1/γ)Z)

X

x∈(1/γ)Z
hk(x)ρ(x)

=
γ
ρ((1/γ)Z)

X

y∈γZ
ck(−2πiy)kρ(y) .

27


---Page Break---
As previously shown in Lemma C.4, the function f(y) = ykρ(y) achieves its maximum at (y∗)2 =
k/2π = ℓ2γ2 −α/π. The issue now is that y∗is not necessarily contained in γZ. However, we show
that “rounding up” y∗to sγ is sufficient to establish a non-trivial lower bound. Taking the ratio of
f(y∗) and f(sγ),

log f(sγ)/f(y∗) = k log
ℓγ

y∗


−π(ℓ2γ2 −(y∗)2) ≥−α > −1 .

Hence, f(sγ) > f(y∗)/e. Combining this observation and the proof of Lemma C.4 leads to the
conclusion.

Theorem C.6 (Theorem 4.2 restated). For any constant C > 0, given γ(d) > 1, σ(d) > 0 such that
γσ < C, estimation error parameter η > 0, and δ ∈(0, 1), there exists a brute-force search estimator
ˆu : Rd×n →Sd−1 that uses n = poly(d, γ, 1/η, 1/δ) samples and achieves ∥ˆu(x1, . . . , xn) −
u∥2 ≤η2 with probability at least 1 −δ over i.i.d. samples x1, . . . , xn ∼Pu, where Pu is the
(γ, σ)-Gaussian pancakes distribution with secret direction u ∈Sd−1.

Proof. Without loss of generality, we assume η ≤1/γ. If the given error parameter η is larger than
1/γ, we set η = 1/γ. Let C be any η-net of Sd−1. Our brute-force search estimator is

ˆu = arg max
v∈C
ˆE(v) ,
where ˆE(v) = 1

n

n
X

i=1
T β
γ (⟨xi, v⟩) .

We choose β = 1/√πγ as the contrast function parameter for reasons explained later. The population
limit of ˆE is E (Eq.(24)), and the monotonicity of E with respect to ⟨u, v⟩2 (Lemma C.3) implies
that u = arg maxSd−1 E(v) in the infinite-sample limit. For x ∼Pu, the distribution of ⟨x, v⟩is
Aξ
γ, where ξ2 = (1 + σ2)/⟨u, v⟩2 −1. Let v1, v2 ∈Sd−1 be such that ⟨u, v1⟩2 −⟨u, v2⟩2 = ε2.
Let ξ2
1 = (1 + σ2)/⟨u, v1⟩2 −1 and ξ2
2 = (1 + σ2)/⟨u, v2⟩2 −1. Notice that

1
1 + ξ2
1
−
1
1 + ξ2
2
=
ε2

1 + σ2 .

For ε-far pairs v1, v2 such that ξ1 is sufficiently small, we establish a lower bound on E(v1) −E(v2)
in terms of ε. This implies that if E(v1) and E(v2) are close, then ⟨u, v1⟩2 and ⟨u, v2⟩2 are also
close.

By Claim C.2, for any v ∈Sd−1, we have

E(v) = ⟨T ξ
γ , T β
γ ⟩Q =

∞
X

k=0
α2
2k ·
1
(1 + β2)k(1 + ξ2)k .

Since all the terms in the series are non-negative and monotonically decreasing in ξ, for any k ∈N
and v1, v2 ∈Sd−1 such that ξ1 ≤ξ2, we have

E(v1) −E(v2) ≥
α2
2k
(1 + β2)k


1
1 + ξ2
1

k
−

1
1 + ξ2
2

k

=
α2
2k
(1 + β2)k(1 + ξ2
1)k


1 −1 + ξ2
1
1 + ξ2
2

1 + ξ2
1
1 + ξ2
2

k−1
+
1 + ξ2
1
1 + ξ2
2

k−2
+ · · · + 1


≥
α2
2k
(1 + β2)k(1 + ξ2
1)k

ε2(1 + ξ2
1)
1 + σ2



≥
α2
2k
(1 + β2)k(1 + ξ2
1)k−1


ε2

1 + σ2



We choose k = ⌊πγ2⌋, β2 = 1/k. If ξ2
1 ≤C′/k for some constant C′ > 0 (this will be justified
later), then by Corollary C.5, we have that α2
2k ≥1/(2πe2γ). Thus, using the fact that 1 + t ≤et for
any t ∈R,

α2
2k
(1 + β2)k(1 + ξ2
1)k−1


ε2

1 + σ2


≥
α2
2k
eC′+1


ε2

1 + σ2


≥
1
2πeC′+3γ


ε2

1 + σ2


.

28


---Page Break---
Since σ2 ≤C2/γ2 < C2, it follows that

E(v1) −E(v2) >
1
2π(1 + C2)eC′+3 · ε2

γ .
(26)

Taking the contrapositive, if v1, v2 ∈Sd−1 are such that ξ1 ≤ξ2, ξ2
1 ≤C′/k and E(v1) −E(v2) ≤
ε2/(2π(1 + C2)eC′+3γ), then ⟨u, v1⟩2 −⟨u, v2⟩2 ≤ε2.

Equipped with this result, we revisit our η-net C of Sd−1. Let v∗= arg maxv∈C E(v) be the
population maximizer of E within C. By the monotonicity of E(v) with respect to ⟨u, v⟩2, we have
that 1 −⟨u, v∗⟩2 ≤η2/2. Moreover, since by assumption σ2 ≤C2/γ2 and η2 ≤1/γ2 ≤1, the
corresponding noise level (ξ∗)2 := (1 + σ2)/⟨u, v∗⟩2 −1 satisfies

(ξ∗)2 ≤
1 + σ2

1 −η2/2 −1 = σ2 + η2/2

1 −η2/2 ≤2σ2 + η2 ≤(2C2 + 1)/γ2 .
(27)

Now consider ˆu = arg maxv∈C ˆE(v), the maximizer of the empirical objective. By the triangle
inequality,

E(v∗) −E(ˆu) = (E(v∗) −ˆE(v∗)) + ( ˆE(v∗) −ˆE(ˆu)) + ( ˆE(ˆu) −E(ˆu))

≤|E(v∗) −ˆE(v∗)| + | ˆE(ˆu) −E(ˆu)| .
(28)

We show that both terms in Eq.(28) concentrate. Recall that ˆE(v) = (1/n) Pn
i=1 T β
γ (⟨v, xi⟩) and
that the function T β
γ satisfies (see proof of Lemma B.11)

T β
γ (z) ≤T β
γ (0) =

p

1 + β2

βρ((1/γ)Z) · ρ((
p

1 + β2/βγ)Z) ≤
p

1 + β2/β .

Since β2 = 1/(πγ2), we have T β
γ (z) ≤γ
√

2π for any z ∈R. Thus, for any fixed v ∈Sd−1, ˆE(v)
is a sum of i.i.d. bounded random variables T β
γ . By Hoeffding’s inequality, n ≲(γ3/η4) log(|C|/δ)
samples are sufficient to guarantee that for all v ∈C, |E(v) −ˆE(v)| ≲η2/γ with probability
at least 1 −δ. From standard covering number bounds (e.g., [77, Corollary 4.2.13]), we have
|C| ≤(3/η)d. Hence, n ≲(dγ3/η4)(log(1/η) + log(1/δ)) samples are sufficient for the desired
level of concentration.

It follows that E(v∗) −E(ˆu) ≲η2/γ with probability 1 −δ over the randomness of ˆu. Since v∗
satisfies (ξ∗)2 ≲1/γ2 (see Eq.(27)) and ξ∗≤ξ, where we denote ξ2 = (1 + σ2)/⟨u, ˆu⟩2 −1, the
closeness of E(v∗) and E(ˆu) implies ⟨u, v∗⟩2 −⟨u, ˆu⟩2 ≤η2 due to Eq.(26). Hence,

∥u −ˆu∥2 = 2(1 −⟨u, ˆu⟩2) ≤2η2 + 2(1 −⟨u, v∗⟩2) ≤3η2 .

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: See our main result presented in Section 3.
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
Justification: In Section 1.3, we mention that our hardness result can be circumvented if we
aim for learning not in statistical distance, but against bounded discriminators.
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

30


---Page Break---
Justification: Full theorem statements and proofs are provided in Sections B and C.
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
Justification: This is a purely theoretical paper with no experiments.
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

31


---Page Break---
Answer: [NA]
Justification: This is a purely theoretical paper with no experiments.
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
Justification: This is a purely theoretical paper with no experiments.
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
Justification: This is a purely theoretical paper with no experiments.
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

32


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

Justification: This is a purely theoretical paper with no experiments.

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

Justification: We present purely theoretical results. No human subjects or data were involved
in this work.

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

Justification: We present purely theoretical results on fundamental limitations of machine
learning, so we do not see any potential for having societal impacts.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

33


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

Justification: Our work does not involve any release of data or models.

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

Justification: Our paper does not use any assets.

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

34


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: Our paper does not release any new assets.
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
Justification: Our paper does not involve crowdsourcing nor research with human subjects.
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
Justification: Our paper does not involve crowdsourcing nor research with human subjects.
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

35


---Page Break---
