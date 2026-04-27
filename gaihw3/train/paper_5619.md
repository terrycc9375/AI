Fixed Confidence Best Arm Identification in the
Bayesian Setting

Kyoungseok Jang
Universitá degli Studi di Milano
ksajks@gmail.com

Junpei Komiyama
New York University / RIKEN AIP
junpei@komiyama.info

Kazutoshi Yamazaki
The University of Queensland
k.yamazaki@uq.edu.au

Abstract

We consider the fixed-confidence best arm identification (FC-BAI) problem in the
Bayesian setting. This problem aims to find the arm of the largest mean with a fixed
confidence level when the bandit model has been sampled from the known prior.
Most studies on the FC-BAI problem have been conducted in the frequentist setting,
where the bandit model is predetermined before the game starts. We show that
the traditional FC-BAI algorithms studied in the frequentist setting, such as track-
and-stop and top-two algorithms, result in arbitrarily suboptimal performances in
the Bayesian setting. We also obtain a lower bound of the expected number of
samples in the Bayesian setting and introduce a variant of successive elimination
that has a matching performance with the lower bound up to a logarithmic factor.
Simulations verify the theoretical results.

1
Introduction

In many sequential decision-making problems, the learner repeatedly chooses an arm (option) to play
with and observes a reward drawn from the unknown distribution of the corresponding arm. One of
the most widely-studied instances of such problems is the multi-armed bandit problem [Thompson,
1933, Robbins, 1952, Lai, 1987], where the goal is to maximize the sum of rewards during the rounds.
Since the learner does not know the distribution of rewards, they need to explore the different arms,
and yet, exploit the arms of the most rewarding arms so far. Different from the classical bandit
formulation, there are situations where one is more interested in collecting information rather than
maximizing intermediate rewards. The best arm identification (BAI) is a sequential decision-making
problem in which the learner is only interested in identifying the arm with the highest mean reward.
While the origin of this problem dates back to at least the 1950s [Bechhofer, 1954, Paulson, 1964,
Gupta, 1977], recent work in the field of machine learning reformulated the problem [Audibert et al.,
2010]. In the BAI, the learner needs to pull arms efficiently for better identification. To achieve
efficiency and accuracy, the learner should determine which arm to choose based on the history, when
to stop the sampling, and which arm to recommend as the learner’s final decision.

There are two types of BAI problems depending on the optimization objective. In the fixed-budget
(FB) setting [Audibert et al., 2010], the learner attempts to minimize the probability of error (misiden-
tification of the best arm) given a limited number of arm pulls T. In the fixed confidence (FC) setting
[Jamieson and Nowak, 2014], the learner attempts to minimize the number of arm pulls, subject to a
predefined probability of error δ ∈(0, 1). In this paper, we shall focus on the FC setting, which is
useful when we desire a rigorous statistical guarantee.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Most of the previous BAI studies focus on the frequentist setting, where the bandit model is chosen
adversarially from some hypothesis class beforehand. In this setting, several algorithms, such as Track
and Stop [Kaufmann et al., 2016a] and Top-two algorithms [Russo, 2016, Qin et al., 2017b, Jourdan
et al., 2022], are widely known. These algorithms have an optimal sample complexity, meaning that
they are one of the most sample-efficient algorithms among the class of δ-correct algorithms.

The sample complexity of these algorithms is problem-dependent. To see this, consider the following
example.
Example 1. (A/B/C testing) Consider A/B/C testing of web designs. We have three arms (web
designs) from which we would like to find the largest retention rate via allocating users to web
designs i = 1, 2, 3. If we attempt to find the best arm with confidence δ, we may need a large number
of samples (users) when the suboptimality gap (the gap between the retention rate of the best arm and
the second best arm) is small because in such a case the identification of the best arm is difficult – the
minimum number of samples required is inversely proportional to the square of the suboptimality
gap[Kaufmann et al., 2014]. For example, when comparing the testing of retention rates of (0.9, 0.5,
0.1) with (0.9, 0.89, 0.1), the second case requires around

0.9−0.5
0.9−0.89

2 = 1600 times more samples
compared to the first case.

In practice, the retention rate of 0.89 in the second case may be acceptably good compared to the
optimal retention rate of 0.9, and we may stop exploration at the moment the learner identifies a
reasonably good arm, which is the first or the second arm in this example. This idea is formalized in
several ways. The literature of Ranking and Selection (R&S) usually considers the indifference-zone
formulation [Hong et al., 2021]. In the context of best arm identification, a similar notion of ϵ-best
answer identification has also been considered [Maron and Moore, 1993, Even-Dar et al., 2006,
Gabillon et al., 2012, Kaufmann and Kalyanakrishnan, 2013, Jourdan et al., 2023]. In these settings,
the learner accepts a sub-optimal arm whose means are at most ϵ worse than the mean of the optimal
arm. Other related settings include the good arm identification problem [Kano et al., 2019, Tabata
et al., 2020, Zhao et al., 2023], where the goal is to identify an arm that exceeds the predefined
threshold, and the thresholding bandit problem [Locatelli et al., 2016, Xu et al., 2019], where the
goal is to identify whether the mean of each arm is above or below the threshold. All these problem
settings require an extra parameter, like ϵ or an acceptance threshold, that directly determines the
acceptance level. Even though the algorithm’s performance depends on this parameter, it is often
challenging to determine a reasonable value for it in advance.

In this paper, we study an alternative approach based on the Bayesian setting. In particular, we
consider the prior distribution on the model parameters. We relax the requirement on the correctness
of the best arm identification by using the prior belief. Rather than requiring the frequentist δ-
correctness for any model, we require the learner to have marginalized correctness over the prior
distribution, which we call Bayesian δ-correctness.

We study the fixed confidence BAI (FC-BAI) problem in the Bayesian setting. Our contributions are
as follows.

• First, we find that in the Bayesian setting, the performance of the traditional frequentist
setting-based algorithms, such as Track and Stop and Top-two algorithms, can be arbitrarily
worse (Section 3). This is because frequentist approaches spend too many resources when
the suboptimality gap is narrow.
• Second, we prove that the lower bound of the number of expected samples should attain
at least the order of Ω( L(H)2

δ
) as δ →0 (Section 4). Here L(H) is our novel quantity
that represents the sample complexity with respect to the prior distribution H. This order
is different from the existing lower bound in the frequentist setting1, implying that the
Bayesian setting is essentially different from the frequentist setting.
• Third, we design an algorithm whose expected sample size is upper-bounded by
O( L(H)2

δ
log L(H)

δ
) (Section 5). Our algorithm is based on the elimination algorithm
[Maron and Moore, 1993, Even-Dar et al., 2006, Frazier, 2014], but we add an early stop-
ping criterion to prevent over-commitment of the algorithm for a bandit model with a narrow
suboptimality gap. Our algorithm has a matching upper bound up to the logarithmic factor.

1In fact, marginalizing the frequentist sample complexity over the prior distribution leads to an unbounded
value.

2


---Page Break---
We also conduct simulation to demonstrate that the sample complexity of frequentist algorithms does
indeed diverge in a bandit model with a small suboptimality gap, even in very simple cases (Section
6).

1.1
Related work

To our knowledge, BAI problems studied for the Bayesian setting have been limited to the fixed
budget setting [Komiyama et al., 2023, Atsidakou et al., 2023]. Komiyama et al. [2023] showed that,
in the fixed-budget setting, a simple non-Bayesian algorithm has an optimal simple regret up to a
constant factor, implying that the advantage the learner could get from the prior is small when the
budget is large. This is very different from our fixed-confidence setting, where utilizing the prior
distribution is necessary.

Several FC-BAI algorithms used Bayesian ideas on the structure of the algorithm, although most
of those studies used frequentist settings for measuring the guarantee. The ‘Top-Two’ type of
algorithms are the leading representatives in this direction. The first instance of top-two algorithms,
which is called Top-Two Thompson sampling (TTTS), is introduced in the context of Bayesian best
arm identification. TTTS requires a prior distribution, and Russo [2016] showed that the sample
complexity of posterior convergence of TTTS which is the same as the sample complexity of the
frequentist fixed-confidence best arm identification. Subsequent research analyzed the performance
of TTTS from the frequentists’ viewpoint [Shang et al., 2020]. Later on, the idea of top-two sampling
is then extended into many other algorithms, such as Top-Two Transportation Cost [Shang et al.,
2020], Top-Two Expected Improvement (TTEI, Qin et al. 2017b), Top-Two Upper Confidence bound
(TTUCB, Jourdan and Degenne 2022a). Even though some of the top two algorithms adapt a prior,
they implicitly solve the optimization that is justified in view of frequentist.

Another line of Bayesian sequential decision-making is Bayesian optimization [Srinivas et al., 2010,
Mockus, 2012, Shahriari et al., 2016, Jamieson and Talwalkar, 2016, Frazier, 2018], where the
goal is to find the best arm in Bayesian setting. Note that Bayesian optimization tends to deal with
structured identification, especially for Gaussian processes, and most of the algorithms for Bayesian
optimization do not have specific stopping criteria.

2
Problem setup

We study the fixed confidence best arm identification problem (FC-BAI) in a Bayesian setting. In this
setup, we have k arms in the set [k] := {1, 2, . . . , k} with unknown distribution P = (P1, · · · , Pk)
which is drawn from a known prior distribution at time 0, namely H = (H1, · · · , Hk). The unknown
bandit model Pi is a one-parameter distribution, and P is specified by µ := (µ1, · · · , µk). To
simplify the problem, we will focus on the Gaussian case, where each Pi is a Gaussian distribution
with known variance σ2
i . Each mean of Pi, denoted µi, is drawn from a known prior Gaussian
distribution Hi, which can be written as N(mi, ξ2
i ).

At every time step t = 1, 2, · · · , the forecaster chooses an arm At ∈[k] and observes a reward Xt,
which is drawn independently from PAt. Since we focus on the Gaussian case, Xt ∼N(µAt, σ2
At)
conditionally given At and µAt. After each sampling, the forecaster must decide whether to continue
the sampling process or stop sampling and make a recommendation J ∈[k].

Let Ft = σ(A1, X1, A2, X2, · · · , At, Xt) be the σ-field generated by observations up to time t. The
algorithm of the forecaster π := ((At)t, τ, J) is defined by the following triplet [Kaufmann et al.,
2016a]:

• A sampling rule (At)t, which determines the arm to draw at round t based on the previous
history (each At must be Ft−1 measurable).

• A stopping rule τ, which means when to stop the sampling (i.e., stopping time with respect
to Ft).

• A decision rule J, which determines the arm the forecaster recommends based on his
sampling history (i.e., J is Fτ-measurable).

In FC-BAI, the forecaster aims to recommend arm J that correctly identifies (one of) the best arm(s)
i∗(µ) := arg maxi∈[k] µi with probability at least 1 −δ. Since the case of multiple best arms is of

3


---Page Break---
measure zero under H, we can focus on µ such that i∗(µ) is unique. For the FC-BAI problem in the
Bayesian setting, we use the expected probability of misidentification:

PoE(π; H) := Eµ∼H


P

J ̸= i∗(µ)|Hµ

,
(1)

where Hµ := {µ is the correct bandit model}. Now we formally define the algorithm of interest as
follows:
Definition 1. (Bayesian δ-correctness) For a prior distribution H, an algorithm π = ((At), τ, J)
is said to be Bayesian (H, δ)-correct if it satisfies PoE(π; H) ≤δ. Let Ab(δ, H) be the set of
Bayesian (H, δ)-correct algorithms for the prior distribution H.

The objective of the FC-BAI problem in the Bayesian setting is to find an algorithm π =
((At)t, τ, J) ∈Ab(δ, H) that minimizes Eµ∼H[τ].

Terminology
Define Ni(t) = Pt−1
s=1 1[As = i] as the number of times arm i is pulled before
timestep t. Let hi be the probability density function of Hi. Since we consider Gaussian prior,
hi(µi) := (1/
√

2πξi) exp(−(µi−mi)2/(2ξ2
i )). Let i∗, j∗: Rk →[k] be the best and the second best
arm under the input such that for each µ ∈{x ∈Rk : xi ̸= xj∀i ̸= j}, i∗(µ) = arg maxi∈[K] µi
and j∗(µ) = arg maxi∈[K]\{i∗(µ)} µi.

Let KLi(a∥b) :=
(a−b)2

2σ2
i
represent the KL-divergence between two Gaussian distributions with

equal variances (the variance of the i-th arm σ2
i ) but different means, denoted as a and b. Similarly,
d(a, b) := a log(a/b) + (1 −a) log((1 −a)/(1 −b)) is the KL divergence between two Bernoulli
distributions with means a and b. Throughout this paper, Eµ and Pµ denote the expectation and
probability when the bandit model is fixed as µ ∈Rk, i.e., Eµ = E[·|Hµ] and Pµ = P(·|Hµ). We
will abuse the notation PoE so that for λ ∈Rk, PoE(π; λ) means

PoE(π; λ) := Pλ
 
J ̸= i∗(λ)|Hλ

.

Naturally, PoE(π; H) = Eµ∼H

PoE(π; µ)

.

Lastly, we introduce the constant L(H) that characterizes the sample complexity in the Bayesian
setting.
Definition 2. For each i, j ∈[k], define Lij(H) and L(H) as follows:

L(H) :=
X

i,j∈[k],i̸=j
Lij(H) where Lij(H) :=
Z ∞

−∞
hi(x)hj(x)
Y

s:s∈[k]\{i,j}
Hs(x) dx.

This constant has the following interesting property which we call a volume lemma:
Lemma 1 (Volume Lemma, informal). For ∆∈(0, 1), let

L(H, ∆) := 1

∆Pµ∼H
h
µi∗(µ) −µj∗(µ) ≤∆
i
.

Then, lim∆→0+ L(H, ∆)
=
L(H).
In particular, for ∆
<
L(H)
P

i∈[k]
2(k−1)

ξi
, L(H, ∆)
∈

( 1

2L(H), 2L(H)).

The volume lemma states that the volume of prior where the suboptimality gap is smaller than ∆is
proportional to L(H)∆when ∆is small. We will see in Section 3 that such small-gap cases, which
require a large amount of exploration to identify the best arm, dominate the Bayesian expectation of
the stopping time. Therefore, L(H) defines the Bayesian sample complexity. The formal version of
this lemma, which involves some regularity conditions, is shown in Appendix B.
Remark 1. Here, we elaborate on how the Bayesian sample complexity is defined. As will be shown
in Section 3, for an algorithm to have a finite expected stopping time, it must determine whether the
current instance is difficult or not. In particular, if an algorithm tries to identify even the top-O(δ)
‘hardest instances’2 in the prior , the algorithm cannot achieve the finite expected stopping time. By

2Here the hardness is based on the size of the suboptimality gap. When the suboptimality gap of an instance
is small, it is a harder instance for the BAI algorithm to identify the best arm, as mentioned in Example 1.

4


---Page Break---
Lemma 1, the suboptimality gap of the top-O(δ) hardest instance is given by L(H)∆≈δ, and the
corresponding (frequentist) sample complexity is proportional to ∆−2 = (L(H)/δ)2 [Kaufmann
et al., 2014]. Such instances constitute an O(δ) fraction of the prior, and thus the Bayesian sample
complexity is:

O
(L(H))2

δ2
× δ

= O
(L(H))2

δ


.

3
Limitation of traditional frequentist approaches in the Bayesian setting

Existing BAI studies mainly focused on the Frequentist δ-correct algorithms which are defined as
follows:
Definition 3 (Frequentist δ-correctness). An algorithm π = ((At), τ, J) is said to be frequentist
δ-correct if, for any bandit instance µ ∈Rk such that i∗(µ) is unique, it satisfies PoE(π; µ) ≤δ.
Let Af(δ) be the set of all frequentist-δ-correct algorithms.

For the frequentist δ-correct algorithms, Garivier and Kaufmann [2016] proved a lower bound for the
expected stopping time as follows: for all bandit instance µ∈Rk and for all ((At), τ, J) ∈Af(δ),

Eµ [τ] ≥log(δ−1)T ∗(µ) + o(log(δ−1))
(2)

where T ∗(µ) is a sample complexity function dependent on the bandit instance µ.3 Moreover, many
of the known frequentist δ-correct algorithms achieve asymptotic optimality [Garivier and Kaufmann,
2016, Russo, 2016, Tabata et al., 2023, Qin et al., 2017a], meaning that they are orderwisely tight up
to the lower bound on Eq. (2) as δ →0. However, little is known, or at least discussed, about their
performance in the Bayesian setting.

One can check that a frequentist δ-correct algorithm is also Bayesian δ-correct as well (Af(δ) ⊂
Ab(δ, H) for all H). Naturally, our interest is whether or not the most efficient classes of fre-
quentist δ-correct algorithms, such as Tracking algorithms and Top-two algorithms, are efficient in
Bayesian settings. Somewhat surprisingly, the following theorem states that any δ-correct algorithm
is suboptimal in Bayesian settings.

Theorem 2. For all δ > 0, H and ((At), τ, J) ∈Af(δ), Eµ∼H [τ] = +∞.

Proof of Theorem 2 is found in Appendix C. To illustrate the proof, we will use a two-armed Gaussian
instance as an example.

3.1
Special case - two armed Gaussian case

Here we present one intuitive corollary of the lower bound theorem [Kaufmann et al., 2016a, Garivier
and Kaufmann, 2016, Kaufmann et al., 2016b] that uses a standard information-theoretic technique.
Corollary 3 (Kaufmann et al. 2014). Let δ ∈(0, 1). For any frequentist δ-correct algorithm
((At), τ, J) and for any fixed mean vector µ = (µ1, µ2) ∈R2, Eµ[τ] ≥
d(δ,1−δ)
(µ1−µ2)2 ≥
log
1
2.4δ
(µ1−µ2)2 .

In the frequentist setting, Corollary 3 implies the lower bound of Eµ[τ] = Ω(log(δ−1)/(µ1 −µ2)2),
which is Ω(log(δ−1)) when we view parameters (µ1, µ2) as constants. However, in the Bayesian
setting, the algorithm is given the prior distribution H on µ, and thus the stopping time is marginalized
over H. In particular, limiting our interest to the case of |µ1 −µ2| < ∆for small enough ∆> 0, we
can obtain the following lower bound:

Eµ∼H[τ] ≥Eµ∼H[τ · 1[|µ1 −µ2| ≤∆]]
(Since τ is positive r.v.)
≥Eµ∼H [E[τ|µ] · 1[|µ1 −µ2| ≤∆]]
(Law of total expectation)

≥Eµ∼H


log δ−1

(µ1 −µ2)2 · 1[|µ1 −µ2| ≤∆]

(Corollary 3)

≥log δ−1

∆2
Pµ∼H[|µ1 −µ2| ≤∆] ≥log δ−1

∆2
L(H)

2
∆
(Lemma 1)

3For details about T ∗(µ), a reader may refer to Garivier and Kaufmann [2016].

5


---Page Break---
= Ω
L(H) log δ−1

∆


.

This inequality implies that if we naively use a known frequentist δ-correct algorithm in the Bayesian
setting, the expected stopping time will diverge because we can choose an arbitrarily small ∆. The
case of a small gap is difficult to identify, and the expected stopping time can be very large for such a
case if we aim to identify the best arm for any model.

4
Lower bound

This section will elaborate on the lower bound of the stopping time in the Bayesian setting. Theorem
4 below states that any Bayesian (H, δ)-correct algorithm requires the expected stopping time of at
least Ω( L(H)2

δ
).

Theorem 4. Define σmin = mini∈[k] σ2
i and NV = L(H)2σ2
min ln 2
16e4δ
. Let δ < δL(H) be sufficiently
small.4 Then, for any BAI algorithms π = ((At), τ, J), if Eµ∼H[τ] ≤NV , then PoE(π; H) ≥δ.

In this main body, we will use the two-armed Gaussian bandit model with homogeneous variance
condition (i.e. σ1 = σ2 = σ) for easier demonstration of the proof sketch. Theorem 4, which is
more general in the sense that it can deal with k > 2 arms with heterogeneous variances, is proven in
Appendix D.

Sketch of the proof, for k = 2:
It suffices to show that the following is an empty set:

Ab(δ, H, NV ) := {π ∈Ab(δ, H) : Eµ∼H[τ] ≤NV }.

Assume that Ab(δ, H, NV ) ̸= ∅and choose an arbitrary π ∈Ab(δ, H, NV ). We start from the
following transportation lemma:
Lemma 5 (Kaufmann et al. 2016a, Lemma 1). Let δ ∈(0, 1). For any algorithm ((At), τ, J), any
Fτ-measurable event E, any bandit models µ, λ ∈{(x, y) ∈R2 : x ̸= y} such that i∗(µ) ̸= i∗(λ),

Eµ




2
X

i=1
KLi(µi, λi)Ni(τ)



≥d(Pµ(E), Pλ(E)).

Note that the above Lemma holds for any algorithm, and thus works for any stopping time τ. Now
define ν(µ) as a swapped version of µ ∈R2, which means (ν(µ))1 = µ2, ν(µ)2 = µ1, and let
E = {J ̸= i∗(µ)}, the event that the recommendation of the algorithm is wrong. Substituting λ with
ν(µ) from the above equation of Lemma 5 leads to

Eµ

(µ1 −µ2)2

2σ2
τ

≥d(PoE(π; µ), 1 −PoE(π; ν)) ≥log
2
2.4(PoE(π; µ) + PoE(π; ν)).
(3)

Note that the first inequality comes from the fact that E, the failure event of the bandit model µ, is
exactly a success event of ν(µ) in this two-armed case, and the last inequality is from our modified
lemma (Lemma 11) from Eq. (3) of Kaufmann et al. [2016a]. One can rewrite the above inequality as

PoE(π; µ) + PoE(π; ν)

2
≥1

2.4 exp

 

Eµ


−(µ1 −µ2)2

2σ2
τ
!

.
(4)

We can rewrite the conditions of Ab(δ, H, NV ) as

PoE(π; H) =
Z

µ∈R2 PoE(π; µ) dH(µ) ≤δ
and
Z

µ∈R2 Eµ[τ] dH(µ) ≤NV .
(Opt0)

Using Eq. (4) and with some symmetry tricks, we get V0 ≤PoE(π; H) where

V0 :=
Z

µ∈R2
1
2.4 exp

−(µ1 −µ2)2

2σ2
Eµ[τ]

dH(µ) ≤δ and
Z

µ∈R2 Eµ[τ] dH(µ) ≤NV .

(Opt1)

4In particular, δL(H) is defined in Appendix G.

6


---Page Break---
Algorithm 1 Successive Elimination with Early-Stopping

Input: Confidence level δ, prior H
∆0 :=
δ
4L(H)
Initialize the candidate of best arms A(1) = [K].
t = 1
while True do

Draw each arm in A(t) once. t ←t + |A(t)|.
for i ∈A(t) do

Compute UCB(i, t) and LCB(i, t) from (5).
if UCB(i, t) ≤maxj LCB(j, t) then

A(t) ←A(t) \ {i}.
end if
end for
if |A(t)| = 1 then

Return arm J in A(t).
end if
Compute ˆ∆safe(t) := max
i∈A(t)UCB(i, t) −max
i∈A(t)LCB(i, t).

if ˆ∆safe(t) ≤∆0 then

Return arm J, uniformly sampled from A(t).
end if
end while

Note that on the above two inequalities, only Eµ[τ] is the value that depends on the algorithm π.
Now our main idea is that we can relax these two inequalities to the following optimization problem
by substituting Eµ[τ] to an arbitrary ˜n : R2 →[0, ∞) as follows:

V :=
inf
˜n:R2→[0,∞)

Z

µ∈R2 exp

−(µ1 −µ2)2

2σ2
˜n(µ)

dH(µ) s.t.
Z

µ∈R2 ˜n(µ) dH(µ) ≤NV

(Opt2)

and V ≤2.4V0 ≤2.4δ. Let N := {µ ∈R2 : |µ1 −µ2| < ∆:=
8δ
L(H)}. From the discussions in
Section 3.1, one might notice that N is an important region for bounding Eµ∼H[τ]. We can relax the
above (Opt2) to the following version, which focuses more on N:

V ′ :=
inf
˜n:R2→[0,∞)

Z

µ∈N
exp

−(µ1 −µ2)2

2σ2
˜n(µ)

dH(µ) s.t.
Z

µ∈N
˜n(µ) dH(µ) ≤NV .

(Opt3)

One can prove V ′ ≤V ≤2.4δ. Now, if we notice that function x 7→exp

−(µ1−µ2)2

2σ2
x

is a
convex function, we can use Jensen’s inequality to verify that the optimal solution for (Opt3) is when
˜n =
NV
Eµ∼H[1N]1N , and when we use NV in Theorem 4, one can get:

V ′ ≥
Z

µ∈N
exp

 

−∆2

2σ2 ·

 
NV
Eµ∼H[1N ]

!!

dH(µ)
(by optimality of ˜n =
NV
Eµ∼H[1N])

≥
Z

µ∈N
exp

−∆2 · NV

σ2∆L(H)


dH(µ) ≥exp

−∆· NV

σ2L(H)


· ∆L(H)
(both by Lemma 1)

> 2.4δ,
(by definition of NV and ∆)

which is a contradiction. This means no algorithm satisfies (Opt1), and the proof is completed.

5
Main algorithm

This section introduces our main algorithm (Algorithm 1). In short, our algorithm is a modification
of the elimination algorithm with the incorporation of the indifference zone technique. Define

7


---Page Break---
∆0 :=
δ
4L(H) which satisfies the following condition, thanks to Lemma 1:

Pµ∼H(µi∗(µ) −µj∗(µ) ≤∆0) ≤δ

2.

In each iteration of the while loop of Algorithm 1, the learner selects and observes each arm in the
active set. After drawing each arm once, the algorithm calculates the confidence bounds for each arm
in the active set using the formula as follows: let Conf(i, t) and ˆµi(t) be the confidence width and
the empirical mean of arm i at time t as

Conf(i, t) :=

s

2σ2
i
log(6(Ni(t))2/(( δ2

2K )π2))
Ni(t)
,
ˆµi(t) :=

t−1
X

s=1
Xs1[As = i].

Then the upper and lower confidence bounds of arm i at timestep t, denoted as UCB and LCB
respectively, can be defined in the following manner:

UCB(i, t) := ˆµi(t) + Conf(i, t),
LCB(i, t) := ˆµi(t) −Conf(i, t).
(5)

This confidence bounds ensure that, with high probability, for all t ∈[T] and i ∈[K], µi ∈
(UCB(i, t), LCB(i, t)) (See Lemma 15 in Appendix for details). After calculating UCB and LCB,
the algorithm eliminates arms with UCB smaller than the largest LCB and maintains only arms that
could be optimal in the active set A. Up to this point, it follows the traditional elimination approach.

The main difference in our algorithm lies in the stopping criterion. At the end of each iteration, the
algorithm checks the stopping criterion. Unlike typical elimination algorithms that continue until
only one arm remains, we have introduced an additional indifference condition. This condition arises
when the suboptimality gap is so small that identifying them would require an excessive number of
samples. In such cases, our algorithm stops additional attempts to identify differences between arms
in the active set and randomly recommends one from the active set instead.
Remark 2. In the context of PAC-(ϵ, δ) identification, Even-Dar et al. [2006, Remark 9] introduced
a similar approach. The largest difference is that they use the parameter ϵ as a parameter that defines
the indifference-zone level, whereas our parameter ∆0 is spontaneously derived from the prior H
and the confidence level δ without specifying the indifference-zone.

Theorem 6 describes the theoretical guarantee of the Algorithm 1.

Theorem 6. For δ < 4L(H) · min

L(H)
P

i∈[k]
k−1

ξi
,

mini,j∈[k] ξiLij(H)

2

, Algorithm 1 which

consists of ((At), τ, J) has the expected stopping time upper bound as follows:

Eµ∼H[τ] ≤C · σ2
max
L(H)2

δ
log
L(H)

δ


+ O(log δ−1),
(6)

where C = 320

π2

3 + 1

is a universal constant and σmax = maxi∈[k] σi. Here, O(log δ−1) is a

function of δ and H that is proportional to log δ−1 when we view prior parameters H as constants.
Plus, the strategy defined by Algorithm 1 is in Ab(δ, H).

See Appendix E for the formal proof of Theorem 6.
Remark 3. When we compare the lower bound (Theorem 4) with the upper bound of Algorithm
1 (Theorem 6), we can see the algorithm is near-optimal. If we view σmax/σmin as a constant, the
bounds are tight up to a log L(H)

δ
factor.

Remark 4. The condition δ < 4L(H) · min

L(H)/P

i∈[k]
k−1

ξi , mini,j∈[k](ξiLij(H))2
is only
for cleaner illustration of the regret bound in Theorem 6. The non-asymptotic result, when δ is a
moderately large constant, can be found in Appendix E.1.

Proof sketch of Theorem 6
We summarize the general strategy for the proof as follows. By the
law of total expectation, Eµ∼H[τ] = Eµ∼H
h
Eµ[τ]
i
. Therefore, we first derive a frequentist upper

bound of Eµ[τ], and then marginalize it to obtain the expected Bayesian stopping time.

First, with the confidence bound defined as Eq. (5) we have the following guarantee that the true
means for all arms are in the confidence bound interval with high probability.

8


---Page Break---
Lemma 7. For any fixed µ ∈{v ∈Rk : vi ̸= vj for all i, j ∈[k]}, let X(µ) := {∀i ∈[k] and t ∈
N, µi ∈(LCB(i, t), UCB(i, t))}. Then, Pµ

X(µ)

≥1 −δ2.

Now we can rewrite Eµ∼H [τ] as follows:

Eµ∼H [τ] =Eµ∼H [Eµ[τ]]
(Law of Total Expectation)
=Eµ∼H [Eµ[τ1[X(µ)]]] + Eµ∼H [Eµ[τ1[X(µ)c]]]

=
X

i
Eµ∼H
h
Eµ[Ni(τ)1[X(µ)]]
i
+ Eµ∼H [Eµ[τ1[X(µ)c]]] .
(7)

Let ∆i = ∆i(µ) := (maxs∈[k] µs) −µi and R0(∆) ≈⌈Cσ2
max · log ∆−1

∆2
⌉. For the first term, under
X(µ), we can bound Ni(τ) by R0(max(∆0, ∆i)) (Lemma 17 in Appendix E), and integrate it over
the prior distribution to obtain the leading factor. For the second term, thanks to the indifference
stopping condition ( ˆ∆safe(t) ≤∆0), one can prove that τ is always smaller than R(∆0) (Lemma 14
in Appendix E), which leads to non-leading term.

To check that the expected probability of error is below δ, we have an additional lemma:

Lemma 8 (Probability of dropping i∗(µ)). For any µ0, under Hµ0, X(µ0) ⊂T
t {i∗(µ0) ∈A(t)} .

This lemma means under the event X(µ), the best arm is never dropped. We can also prove that
under the event X(µ), if ∆i(µ) > ∆0, the sub-optimal arm will eventually be dropped before the
algorithm terminates (Lemma 14 in Appendix E). These two facts mean there are only two cases in
which the prediction of Algorithm 1 could be wrong.

• Under X(µ)c, both facts cannot guarantee the correct identification. From Lemma 7,
Pµ[X(µ)c] ≤δ2 for all µ, and thus Pµ∼H[X(µ)c] ≤δ2.

• When ∆i(µ) ≤∆0. From Lemma 1 and the definition of ∆0, the probability of drawing
such µ from the prior is at most δ/2.

Therefore, by union bound, Algorithm 1 has the expected probability of misidentification guarantee
smaller than δ2 + δ/2 < δ.

6
Simulation

We conduct two experiments to demonstrate that the expected stopping times of frequentist δ-correct
algorithms diverge in a Bayesian setting and that the elimination process in Algorithm 1 is necessary
for more efficient sampling. In Tables 1 and 2, each column ‘Avg’, ‘Max’, and ‘Error’ represents the
average stopping time, maximum stopping time, and the ratio of the misidentification, respectively.5
More details of these experiments are in Appendix F.

Frequentist algorithms diverge in Bayesian Setting
We evaluate the empirical performance of
our Elimination algorithm (Algorithm 1) by comparing it with other frequentist algorithms such
as Top-two Thompson Sampling (TTTS) [Russo, 2016] and Top-two UCB (TTUCB) [Jourdan and
Degenne, 2022b].

We design an experiment setup that has k = 2 arms with standard Gaussian prior distribution, which
means mi = 0, ξi = 1 for all i ∈[k]. We set δ = 0.1 and ran N = 1000 Bayesian FC-BAI
simulations to estimate the expected stopping time and success rate.

In Table 1, one can see that the two top-two algorithms exhibit very large maximum stopping time.
This supports our theoretical result in Section 3 that the expected stopping time of Frequentist δ-
correct algorithms will diverge in the Bayesian setting. We did not check the track and stop algorithm
[Garivier and Kaufmann, 2016] because it needs to solve an optimization for each round, but the
fact that the expected stopping time of the track and stop is at least half of the TTTS and TTUCB
for a small δ implies that the performance of track and stop is similar to that of top-two algorithms.
Algorithm 1 shows a significantly smaller average stopping time as well as an average computation
time than that of these algorithms.

5We include the computation time in the Appendix F.3

9


---Page Break---
Table 1: Comparison of two top-two algo-
rithms and Algorithm 1.

AVG
MAX
ERROR

ALG. 1
1.06 × 104
2.35 × 105
1.5%
TTTS
1.56 × 105
1.09 × 108
0.5%
TTUCB
1.95 × 105
1.13 × 108
0%

Table 2: Comparison of Algorithm 1 and the
no-elimination version of it.

AVG
MAX
ERROR

ALG. 1
2.69 × 105
1.66 × 107
0.6%
NOELIM
1.29 × 106
8.25 × 107
0%

Effect of the elimination process
We implemented the modification of Algorithm 1 (denoted as
NoElim) that never eliminates an arm from A(t) 6 In this setup, we have k = 10 arms with standard
Gaussian prior distribution, which means mi = 0, ξi = 1 for all i ∈[k]. We set δ = 0.01 and ran
N = 1000 Bayesian FC-BAI simulations.

As one can check from Table 2, elimination of arms helps the efficient use of samples and reduces
stopping time and computation time.

7
Discussion and future works

We have considered the Gaussian Bayesian best arm identification with fixed confidence. We show
that the traditional Frequentist FC-BAI algorithms do not stop in finite time in expectation, which
implies the suboptimality of such algorithms in the Bayesian FC-BAI problem. We have established
a lower bound of the Bayesian expected stopping time, which is of order Ω( L(H)2

δ
). Moreover, we
have introduced the elimination and early stopping algorithm, which achieves a matching stopping
time up to a polylogarithmic factor of L(H) and δ. We conduct simulations to support our results.

In the future, we will attempt to tighten the logarithmic and

maxi σi

mini σi


2 gap between the lower
and upper bound, extend the indifference zone strategy for other traditional BAI algorithms in the
Bayesian setting, extend our analysis from Gaussian bandit instances to general exponential families,
and design a robust algorithm against misspecified priors.

Acknowledgements

K. Jang acknowledge the financial support from the MUR PRIN grant 2022EKNE5K (Learning
in Markets and Society), the FAIR (Future Artificial Intelligence Research) project, funded by the
NextGenerationEU program within the PNRR-PE-AI scheme, and the the EU Horizon CL4-2022-
HUMAN-02 research and innovation action under grant agreement 101120237, project ELIAS
(European Lighthouse of AI for Sustainability).

J. Komiyama was supported by NYU Stern School of Business Research Scholars Fund no. 10-83004-
BF478.

K. Yamazaki was supported by JSPS KAKENHI grant no. JP20K03758, JP24K06844 and
JP24H00328 and the start-up grant by the School of Mathematics and Physics of the University of
Queensland.

References

Alexia Atsidakou, Sumeet Katariya, Sujay Sanghavi, and Branislav Kveton. Bayesian fixed-budget
best-arm identification, 2023.

Jean-Yves Audibert, Sébastien Bubeck, and Rémi Munos.
Best arm identification in multi-
armed bandits.
In Conference on Learning Theory, pages 41–53, 2010.
URL http://
colt2010.haifa.il.ibm.com/papers/COLT2010proceedings.pdf#page=49.

Robert E Bechhofer. A single-sample multiple decision procedure for ranking means of normal
populations with known variances. The Annals of Mathematical Statistics, pages 16–39, 1954.

6See Appendix F.2 for the pseudocode.

10


---Page Break---
Eyal Even-Dar, Shie Mannor, Yishay Mansour, and Sridhar Mahadevan. Action elimination and
stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of
machine learning research, 7:1079–1105, 2006.

Peter I. Frazier. A fully sequential elimination procedure for indifference-zone ranking and selection
with tight bounds on probability of correct selection. Operations Research, 62(4):926–942, 2014.
doi: 10.1287/opre.2014.1282. URL https://doi.org/10.1287/opre.2014.1282.

Peter I. Frazier. A tutorial on bayesian optimization. CoRR, abs/1807.02811, 2018. URL http:
//arxiv.org/abs/1807.02811.

Victor Gabillon, Mohammad Ghavamzadeh, and Alessandro Lazaric. Best arm identification: A
unified approach to fixed budget and fixed confidence. In Peter L. Bartlett, Fernando C. N. Pereira,
Christopher J. C. Burges, Léon Bottou, and Kilian Q. Weinberger, editors, Advances in Neural
Information Processing Systems 25: 26th Annual Conference on Neural Information Processing
Systems 2012. Proceedings of a meeting held December 3-6, 2012, Lake Tahoe, Nevada, United
States, pages 3221–3229, 2012. URL https://proceedings.neurips.cc/paper/2012/hash/
8b0d268963dd0cfb808aac48a549829f-Abstract.html.

Aurélien Garivier and Emilie Kaufmann. Optimal best arm identification with fixed confidence. In
Conference on Learning Theory, pages 998–1027. PMLR, 2016.

Shanti S. Gupta. Selection and ranking procedures: a brief introduction. Communications in
Statistics - Theory and Methods, 6(11):993–1001, 1977. doi: 10.1080/03610927708827548. URL
https://doi.org/10.1080/03610927708827548.

L Jeff Hong, Weiwei Fan, and Jun Luo. Review on ranking and selection: A new perspective.
Frontiers of Engineering Management, 8(3):321–343, 2021.

Kevin G. Jamieson and Robert D. Nowak. Best-arm identification algorithms for multi-armed
bandits in the fixed confidence setting. In 48th Annual Conference on Information Sciences
and Systems, CISS 2014, Princeton, NJ, USA, March 19-21, 2014, pages 1–6. IEEE, 2014. doi:
10.1109/CISS.2014.6814096. URL https://doi.org/10.1109/CISS.2014.6814096.

Kevin G. Jamieson and Ameet Talwalkar. Non-stochastic best arm identification and hyperparameter
optimization. In Arthur Gretton and Christian C. Robert, editors, Proceedings of the 19th Interna-
tional Conference on Artificial Intelligence and Statistics, AISTATS 2016, Cadiz, Spain, May 9-11,
2016, volume 51 of JMLR Workshop and Conference Proceedings, pages 240–248. JMLR.org,
2016. URL http://proceedings.mlr.press/v51/jamieson16.html.

Marc Jourdan and Rémy Degenne. Non-asymptotic analysis of a ucb-based top two algorithm. CoRR,
abs/2210.05431, 2022a. doi: 10.48550/ARXIV.2210.05431. URL https://doi.org/10.48550/
arXiv.2210.05431.

Marc Jourdan and Rémy Degenne. Non-asymptotic analysis of a ucb-based top two algorithm. arXiv
preprint arXiv:2210.05431, 2022b.

Marc Jourdan, Rémy Degenne, Dorian Baudry, Rianne de Heide, and Emilie Kaufmann. Top two
algorithms revisited. Advances in Neural Information Processing Systems, 35:26791–26803, 2022.

Marc Jourdan, Rémy Degenne, and Emilie Kaufmann. An varepsilon-best-arm identification
algorithm for fixed-confidence and beyond. arXiv preprint arXiv:2305.16041, 2023.

Hideaki Kano, Junya Honda, Kentaro Sakamaki, Kentaro Matsuura, Atsuyoshi Nakamura, and
Masashi Sugiyama. Good arm identification via bandit feedback. Mach. Learn., 108(5):721–745,
2019. doi: 10.1007/S10994-019-05784-4. URL https://doi.org/10.1007/s10994-019-
05784-4.

Emilie Kaufmann and Shivaram Kalyanakrishnan. Information complexity in bandit subset se-
lection. In Shai Shalev-Shwartz and Ingo Steinwart, editors, COLT 2013 - The 26th Annual
Conference on Learning Theory, June 12-14, 2013, Princeton University, NJ, USA, volume 30
of JMLR Workshop and Conference Proceedings, pages 228–251. JMLR.org, 2013.
URL
http://proceedings.mlr.press/v30/Kaufmann13.html.

11


---Page Break---
Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of a/b testing. In
Conference on Learning Theory, pages 461–481. PMLR, 2014.

Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification
in multi-armed bandit models. Journal of Machine Learning Research, 17(1):1–42, 2016a.

Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best arm identification
in multi-armed bandit models, 2016b.

Junpei Komiyama, Kaito Ariu, Masahiro Kato, and Chao Qin. Rate-optimal bayesian simple regret
in best arm identification. Mathematics of Operations Research, Ahead of Print, 2023. doi:
10.1287/moor.2022.0011. URL https://doi.org/10.1287/moor.2022.0011.

Tze Leung Lai. Adaptive treatment allocation and the multi-armed bandit problem. The Annals of
Statistics, 15(3):1091 – 1114, 1987. doi: 10.1214/aos/1176350495. URL https://doi.org/
10.1214/aos/1176350495.

Andrea Locatelli, Maurilio Gutzeit, and Alexandra Carpentier. An optimal algorithm for the thresh-
olding bandit problem. In Maria-Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of
the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA,
June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 1690–1698.
JMLR.org, 2016. URL http://proceedings.mlr.press/v48/locatelli16.html.

Oded Maron and Andrew W. Moore.
Hoeffding races:
Accelerating model selection
search for classification and function approximation.
In Jack D. Cowan, Gerald Tesauro,
and Joshua Alspector, editors, Advances in Neural Information Processing Systems 6,
[7th NIPS Conference, Denver, Colorado, USA, 1993], pages 59–66. Morgan Kauf-
mann, 1993. URL http://papers.nips.cc/paper/841-hoeffding-races-accelerating-
model-selection-search-for-classification-and-function-approximation.

J. Mockus. Bayesian Approach to Global Optimization: Theory and Applications. Mathematics
and its Applications. Springer Netherlands, 2012.
ISBN 9789400909090.
URL https://
books.google.fr/books?id=VuKoCAAAQBAJ.

Edward Paulson. A sequential procedure for selecting the population with the largest mean from
k normal populations. The Annals of Mathematical Statistics, 35(1):174 – 180, 1964. doi:
10.1214/aoms/1177703739. URL https://doi.org/10.1214/aoms/1177703739.

Chao Qin, Diego Klabjan, and Daniel Russo. Improving the expected improvement algorithm. In
Advances in Neural Information Processing Systems, volume 30, pages 5381–5391, 2017a.

Chao Qin, Diego Klabjan, and Daniel Russo. Improving the expected improvement algorithm.
Advances in Neural Information Processing Systems, 30, 2017b.

Herbert Robbins. Some aspects of the sequential design of experiments. Bulletin of the American
Mathematical Society, 58(5):527 – 535, 1952.

Daniel Russo. Simple bayesian algorithms for best arm identification. In 29th Annual Conference on
Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 1417–1418.
PMLR, 23–26 Jun 2016.

Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. Taking the
human out of the loop: A review of bayesian optimization. Proc. IEEE, 104(1):148–175, 2016.
doi: 10.1109/JPROC.2015.2494218. URL https://doi.org/10.1109/JPROC.2015.2494218.

Xuedong Shang, Rianne Heide, Pierre Menard, Emilie Kaufmann, and Michal Valko.
Fixed-
confidence guarantees for bayesian best-arm identification. In International Conference on Artificial
Intelligence and Statistics, pages 1823–1832. PMLR, 2020.

Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger. Gaussian process
optimization in the bandit setting: No regret and experimental design. In Johannes Fürnkranz
and Thorsten Joachims, editors, Proceedings of the 27th International Conference on Machine
Learning (ICML-10), June 21-24, 2010, Haifa, Israel, pages 1015–1022. Omnipress, 2010. URL
https://icml.cc/Conferences/2010/papers/422.pdf.

12


---Page Break---
Koji Tabata, Atsuyoshi Nakamura, Junya Honda, and Tamiki Komatsuzaki. A bad arm existence
checking problem: How to utilize asymmetric problem structure? Mach. Learn., 109(2):327–
372, 2020. doi: 10.1007/S10994-019-05854-7. URL https://doi.org/10.1007/s10994-019-
05854-7.

Koji Tabata, Junpei Komiyama, Atsuyoshi Nakamura, and Tamiki Komatsuzaki. Posterior tracking
algorithm for classification bandits. In Francisco J. R. Ruiz, Jennifer G. Dy, and Jan-Willem van de
Meent, editors, International Conference on Artificial Intelligence and Statistics, 25-27 April
2023, Palau de Congressos, Valencia, Spain, volume 206 of Proceedings of Machine Learning
Research, pages 10994–11022. PMLR, 2023. URL https://proceedings.mlr.press/v206/
tabata23a.html.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of
the evidence of two samples. Biometrika, 25(3/4):285–294, 1933.

Yichong Xu, Xi Chen, Aarti Singh, and Artur Dubrawski. Thresholding bandit problem with both
duels and pulls. CoRR, abs/1910.06368, 2019. URL http://arxiv.org/abs/1910.06368.

Yao Zhao, Connor Stephens, Csaba Szepesvári, and Kwang-Sung Jun.
Revisiting simple re-
gret: Fast rates for returning a good arm. In Andreas Krause, Emma Brunskill, Kyunghyun
Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Confer-
ence on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume
202 of Proceedings of Machine Learning Research, pages 42110–42158. PMLR, 2023. URL
https://proceedings.mlr.press/v202/zhao23g.html.

13


---Page Break---
A
Notation table

Table 3: Major notation
symbol
definition
k
number of the arms
δ
confidence level
µ
means (= (µ1, µ2, . . . , µk))
H
prior distribution of µ
Hi
prior distribution of µi
hi
prior density of µi
mi, ξi
mean and standard deviation of hi
Ni(t)
Pt−1
s=1 1[As = i]
L(H)
See Definition 2
i∗(µ), j∗(µ)
best arm and second best arm
ν(µ)
alternative model where the top-two means of µ are swapped
KLi(·, ·)
KL divergence between two distributions
d(p, q)
KL divergence between two Bernoulli distributions with parameters p and q
Ni(t)
number of draws on arm i before time step t
B
320 maxi∈[k] σ2
i
B0

π2

3 + 1

B
Θi
{µ ∈Rk : i∗(µ) = i}
Θij
{µ ∈Rk : i∗(µ) = i, j∗(µ) = j}
µ\i ( µ\i,j)
the vector projection which omits i-th coordinate (i, j-th, respectively)
H\i ( H\i,j)
the distribution which omits i-th coordinate (i, j-th, respectively)

Table 4: Notations for the lower bound proof, Section D
symbol
definition
R
e−4
˜∆
32e4
L(H)δ
ν(µ)
alternative model where the top-two means are swapped
ni(µ)
Eµ[Ni(τ)]

D0(H)





W(−
1
32 maxi∈[k] ξ3/2
i
)
If maxi∈[k] ξi >
3q

e2
210

1
Otherwise
(W is the Lambert W function.)

D1(H)
mini̸=j

" mi

σ2
i −mj

σ2
j


−1
,

1
2σ2
i +
1
2σ2
j


−2
#

δL(H)
L(H)

32e4 · min

D0(H), D1(H), mini∈[k]
1
4m2
i ,
L(H)
4(k−1) P

i∈[k]
1
ξi



B
Proof of Lemma 1

We will use the following formal version of the volume lemma for the proof. The first result is used
for the upper bound, and the second result is used for the lower bound.

Lemma 9 (Volume Lemma, formal). Let Θij := {µ ∈Rk : i∗(µ) = i, j∗(µ) = j}.

1. For any ∆∈(0, 1), define

Lij(H, ∆) := 1

δ

Z

Θij
1[|µi −µj| ≤∆] dH(µ).

14


---Page Break---
Table 5: Notations for the upper bound proof, Section E
symbol
definition
Conf(i, t)
confidence bound (Section 5)
UCB(i, t), LCB(i, t)
upper and lower confidence bounds (Section 5)
ˆ∆safe(t)
maxi∈A(t) UCB(i, t) −maxi∈A(t) LCB(i, t)
∆0
δ
4L(H)
∆thr
min

log 4
√

k
δπ , 1

B


R0(∆)
B log min(∆,∆thr)−1

min(∆,∆thr)2
T0
kR0(∆0)
∆s(µ)
µi∗(µ) −µs
X(µ)
Event T
i∈[k]
hT∞
t=1 {LCB(i, t) ≤µi}
TT∞
t=1 {UCB(i, t) ≥µi}
i
. (Eq. (24))

Then,

Lij(H, ∆) ∈

Lij(H) −1

ξi
∆, Lij(H) + 1

ξi
∆

.
(8)

In particular, when ∆<
L(H)
P

i∈[k]
k−1

ξi
, L(H, ∆) ≤2L(H).

2. (Volume lemma for the lower bound) For small enough positive real number ∆<

min

1
4 maxi∈[k] m2
i , D0(H)

7 let

L′
ij(H, ∆) := 1

∆

Z

Θij
1[|µi −µj| ≤∆]1[|µi|, |µj| ≤
1
√

∆
] dH(µ).

Then,

L′
ij(H, ∆) ∈

Lij(H) −2

ξi
∆, Lij(H) + 1

ξi
∆

.
(9)

In particular, when ∆
<
min

"

1
4 maxi∈[k] m2
i , D0(H),
L(H)
P

i∈[k]
4(k−1)

ξi

#

, L′(H, ∆)
∈
h
1
2L(H), 2L(H)
i
.

Proof. First, let us prove the upper bound of Eq. (8), i.e. Lij(H, δ) ≤Lij(H) + 1

ξi ∆.

Recall for i, j ∈[k], Θij = {µ : i∗(µ) = i, j∗(µ) = j}. We have
Z

Θij
1 [|µi −µj| ≤∆] dH(µ) =
Z ∞

−∞

Z µj+∆

µj
hi(µi) dµi
Y

k̸=i,j

Z µj

−∞
hk(µk) dµkhj(µj) dµj

≤
Z ∞

−∞


∆max
0≤y≤∆hi(µj + y)
 Y

k̸=i,j

Z µj

−∞
hk(µk) dµkhj(µj) dµj

≤∆
Z ∞

−∞


hi(µj) + e−1/2

ξi
∆
 Y

k̸=i,j

Z µj

−∞
hk(µk) dµkhj(µj) dµj

(by the Lipschitz property of the Gaussian density, e−1/2/ξi is the steepest slope of N(mi, ξ2
i ))

≤∆










Z ∞

−∞
hi(µj)
Y

k̸=i,j

Z µj

−∞
hk(µk) dµkhj(µj) dµj

|
{z
}
=Lij(H)

+
e−1/2

ξi
∆










.

7See Appendix G for the definition of D0(H).

15


---Page Break---
Therefore, we verified the upper bound side of Eq. (8).

For the lower bound, by following the same steps, one can prove
R

Θij 1 [|µi −µj| ≤∆] dH(µ) ≥
(Lij(H) −1

ξi ∆)∆.

For the proof of Eq. (9), by Chernoff’s method we can bound the tail probability as follows:
Z

Θij
1

|µi −mi| >
1
√

∆


dH(µ) <
Z

Rk1

|µi −mi| >
1
√

∆


dH(µ) < 2 exp

 

−
1
2∆ξ2
i

!

.

When |mi| <
1
2
√

∆, then we can change the above inequality as:
Z

Θij
1

|µi| >
1
√

∆


dH(µ) ≤
Z

Rk 1

|µi| >
1
√

∆


dH(µ)

≤
Z

Rk 1

|µi −mi| >
1
2
√

∆


dH(µ) < 2 exp

 

−
1
8∆ξ2
i

!

.

Therefore,
Z

Θij
1[|µi −µj| ≤∆]1[|µi|, |µj| ≤
1
√

∆
] dH(µ) ≥
Z

Θij
1 [|µi −µj| ≤∆] dH(µ)

−
Z

Θij
1

|µi −mi| >
1
2
√

∆


dH(µ) −
Z

Θij
1

|µj −mj| >
1
2
√

∆


dH(µ)

≥(Li(H) −1

ξi
∆)∆−2 exp

 

−
1
8∆ξ2
i

!

−2 exp

 

−
1
8∆ξ2
j

!

≥(Li(H) −2

ξi
∆)∆
(by ∆< D0(H))

for ∆< min

mini∈[k]
1
4m2
i , D0(H)

.

C
Proof of Theorem 2

For ∆> 0, let Θi(∆) := {µ ∈Rk : i∗(µ) = i, µi −µj∗(µ) ≤∆}. From Lemma 9, we have

Pµ∼H
h
i∗(µ) = i, µi −µj∗(µ) ≤∆
i
=
X

j̸=i
Pµ∼H
h
i∗(µ) = i, j∗(µ) = j, µi −µj ≤∆
i

≥
X

j̸=i

1
2Lij(H)∆.

Now, for each µ ∈Rk, let ν : Rk →Rk be a function such that for s ∈[k],

ν(µ)s :=






µi∗(µ)
when s = j∗(µ)
µj∗(µ)
when s = i∗(µ)
µs
Otherwise.

For any µ ∈Rk, let E(µ) = {J ̸= i∗(µ)} and ν = ν(µ). By Lemma 1 in Kaufmann et al. [2016a],
for any frequentist δ-correct algorithm π = ((At)t, τ, J), we have

X

s∈[k]
Eµ[Ns(τ)]KLs(µs||νs) ≥d(Pµ(E(µ)), Pν(E(µ))),
µ ∈Rk.
(10)

For the left side, from the construction of ν,

16


---Page Break---
KLs(µs||νs) :=

( (µi∗(µ)−µj∗(µ))2

2σ2
s
s = i∗(µ), j∗(µ)
0
Otherwise.

Since π ∈Af(δ) and from the definition of E(µ) and ν, Pµ(E(µ)) ≤δ and Pν(E(µ)) ≥1 −δ.

Overall, we can rewrite Eq. (10) to the following simpler form:

Eµ[Ni∗(µ)(τ)](µi∗(µ) −µj∗(µ))2

2σ2
i∗(µ)
+ Eµ[Nj∗(µ)(τ)](µi∗(µ) −µj∗(µ))2

2σ2
j∗(µ)
≥d(δ, 1 −δ)

=⇒Eµ[Ni∗(µ)(τ) + Nj∗(µ)(τ)] ≥2d(δ, 1 −δ) mins∈[k] σ2
s
(µi∗(µ) −µj∗(µ))2
.

Since τ = Pk
s=1 Ns(τ), we can lower bound the expected stopping time when µ is given as follows:

Eµ[τ] ≥Eµ[Ni∗(µ)(τ) + Nj∗(µ)(τ)] ≥2d(δ, 1 −δ) mins∈[k] σ2
s
(µi∗(µ) −µj∗(µ))2
.
(11)

Now, when we compute marginal Eµ[τ] over µ, we have

Eµ∼H[τ] = Eµ∼H
h
Eµ[τ]
i
(Law of total expectation)

≥Eµ∼H
h
Eµ[τ]1Θi(∆)
i

≥Eµ∼H

"
2d(δ, 1 −δ) mins∈[k] σ2
s
(µi∗(µ) −µj∗(µ))2
1Θi(∆)

#

(Eq. (11))

≥Eµ∼H

2d(δ, 1 −δ) mins∈[k] σ2
s
∆2
1Θi(∆)



≥

P
j̸=i Lij(H)

d(δ, 1 −δ) mins∈[k] σ2
s
∆2
=

P
j̸=i Lij(H)

d(δ, 1 −δ) mins∈[k] σ2
s
∆
.

Now since ∆is an arbitrary small positive number, we can conclude that Eµ∼H[τ] diverges.

D
Proof of Theorem 4

In this subsection, we will prove the following theorem:
Theorem 10 (Restatement of Theorem 4). Let δ > 0 be sufficiently small such that δ < δL(H). For
any best arm identification algorithm, if

Z

Rk



X

i∈[k]
ni(µ)



dH(µ) ≤NV ,

then
Z

Rk Pµ[J ̸= i∗(µ)] dH(µ) ≥δ.

Proof. By Lemma 1 in Kaufmann et al. [2016a], for any stopping time τ, we have

X

i∈[k]
Eµ[Ni(τ)]KLi(µi||νi) ≥d(Pµ(E(µ)), Pν(E(µ))),
µ ∈Rk.
(12)

To modify the RHS of Eq. (12), we will use the following lemma:

17


---Page Break---
Lemma 11. For any p, q′, q ∈(0, 1) such that q′ ≤q, we have

ln
1
4(p + q) ≤d(p, 1 −q′).

By using Lemma 11 and the fact that Pν(E(µ)) ≥1−Pν[J ̸= i∗(ν)] by definition, we can transform
(12) into
X

i∈[k]
Eµ[Ni(τ)]KLi(µi||νi) ≥ln
1
4(Pµ[J ̸= i∗(µ)] + Pν[J ̸= i∗(ν)]),
∀µ ∈Rk.
(13)

Note that Pµ(E(µ)) is exactly the error probability, and we are interested in the marginal error
probability Eµ∼H[Pµ(E)]. Let ni(µ) = Eµ[Ni(τ)], and ˜∆is an arbitrary small enough positive
variable which we will define later on Eq. (19). Then, by rearrangement, we can induce the following
inequalities
Z

Rk Pµ[J ̸= i∗(µ)] dH(µ)

=
X

i∈[k]

X

j̸=i

Z

Θij
Pµ[J ̸= i] dH(µ)

= 1

2



X

i∈[k]

X

j̸=i

Z

Θij
Pµ[J ̸= i] dH(µ) +
X

j∈[k]

X

i̸=j

Z

Θji
Pν(µ)[J ̸= j]hi(µj)hj(µi)



Y

s̸=i,j
hs(µs)



dµ





(Symmetry of the Lebesgue measure)

= 1

2



X

i∈[k]

X

j̸=i

Z

Θij

h
Pµ[J ̸= i]hi(µi)hj(µj) + Pν(µ)[J ̸= j]hi(µj)hj(µi)
i


Y

s̸=i,j
hs(µs)



dµ





≥
X

i∈[k]

X

j̸=i

Z

Θij

Pµ[J ̸= i∗(µ)] + Pν[J ̸= i∗(ν)]

2
min (hi(µi)hj(µj), hi(µj)hj(µi)) dµ

(AC + BD ≥A min(C, D) + B min(C, D) = (A + B) min(C, D) for A, B, C, D > 0)

≥
X

i∈[k]

X

j̸=i

Z

Θij

exp (−ni(µ)KLi(µi, µj) −nj(µ)KLj(µj, µi))

8
min

1, hi(µj)hj(µi)

hi(µi)hj(µj)


hi(µi)hj(µj) dµ

(Eq. (13))

≥e−4 X

i∈[k]

X

j̸=i

Z

Θij

exp (−ni(µ)KLi(µi, µj) −nj(µ)KLj(µj, µi))

8
1

"

|µi|, |µj| ≤
1
p

˜∆

#

dH(µ).

(Lemma 12)

For the last inequality, we used the following lemma. The proof for this lemma is found in Subsection
D.1.2.

Lemma 12 (Ratio Lemma). For all a, b ∈R which satisfy |a −b| ≤D1(H) and |a|, |b| ≤
1
√

D1(H)
for some fixed D1(H)8, hi(a)hj(b)

hi(b)hj(a) ≥e−4 for all i, j ∈[k].

In short, we have
Z

RkPµ[J ̸= i∗(µ)] dH(µ)

≥e−4 X

i∈[k]

X

j̸=i

Z

Θij

exp (−ni(µ)KLi(µi, µj) −nj(µ)KLj(µj, µi))

8
1

"

|µi|, |µj| ≤
1
p

˜∆

#

dH(µ)

(14)

8D1(H) := mini̸=j

"
mi

σ2
i −
mj

σ2
j



−1
,

1
2σ2
i +
1
2σ2
j


−2
#

18


---Page Break---
and the following statement is a stronger statement than Theorem 10.

If
R P
i∈[k] ni(µ)

dH(µ) ≤NV , then (RHS of Eq. (14))≥δ.

RHS of Eq. (14) is represented in terms of n := (n1, · · · , nk) : Rk →[0, ∞)k (the expected number
of arm pulls) and does hold for any algorithm given n. To prove the above statement, since ‘set
of all expected number of arm pulls’ is a subset of {˜n : Rk →[0, ∞)}, it suffices to show that the
optimal value V of the following objective

V min :=
inf
˜n:Rk→[0,∞)k V (˜n)
(15)

s.t.
Z

Rk




k
X

s=1
˜ns(µ)



dH(µ) ≤NV

where V (˜n) := e−4 X

i∈[k]

X

j̸=i

Z

Θij

exp (−˜ni(µ)KLi(µi, µj) −˜nj(µ)KLj(µj, µi))

8
1

"

|µi|, |µj| ≤
1
p

˜∆

#

dH(µ)

is greater than δ.

Let ˜Θij := {µ ∈Θij : |µi −µj| ≤˜∆, |µi| ≤
1
√

˜∆, |µj| ≤
1
√

˜∆}. Then, it holds that V min ≥V min
1
where

V min
1
:=
inf
˜n:Rk→[0,∞)k V1(˜n)
(16)

s.t.
X

i∈[k]

X

j̸=i

Z

˜Θij




k
X

s=1
˜ns(µ)



dH(µ) ≤NV

where V1(˜n) := e−4 X

i∈[k]

X

j̸=i

Z

˜Θij

exp (−˜ni(µ)KLi(µi, µj) −˜nj(µ)KLj(µj, µi))

8
dH(µ).

To see this, suppose ˆn is an optimal solution to (15). Then, by the constraint of optimization
problem (15),
R

Rk
P

s∈[k] ˆns(µ)

dH(µ) ≤NV and since ˆn is a collection of positive functions,
P

i,j∈[k]:i>j
R
˜Θij∪˜Θji

Pk
s=1 ˜ns(µ)

dH(µ) ≤NV , which means ˆn satisfies the constraint of (16).

By the minimality, V min
1
≤V1(ˆn), and since exp is a positive function, we have V1(ˆn) ≤V (ˆn) =
V min.

Moreover, V min
1
≥V min
2
holds for

V min
2
:=
inf
˜n:Rk→[0,∞)k V2(˜n)
(17)

s.t.
X

i∈[k]

X

j̸=i

Z

˜Θij




k
X

s=1
˜ns(µ)



dH(µ) ≤NV

where V2(˜n) := e−4 X

i∈[k]

X

j̸=i

Z

˜Θij

exp

−

˜ni(µ) + ˜nj(µ)

˜∆2
2 min(σ2s)s∈[k]



8
dH(µ)

by using the fact that KLi(µi, µj) =
˜∆2

2σ2
i ≤
˜∆2
2 min(σ2s)s∈[k] .

19


---Page Break---
Claim 1. We abuse our notation slightly so that H(E) =
R

E dH(µ) for any Lebesgue measurable
set E, and let ˜Θ = ∪i,j∈[k]:i̸=j ˜Θij (note that all ˜Θij are mutually disjoint except for the measure
zero sets). Then, the following nopt is an optimal solutions to (17)

nopt
s (µ) :=
NV
2H(˜Θ)
1

µ ∈

∪j̸=s ˜Θsj

∪

∪i̸=s ˜Θis

.

Proof. Choose an arbitrary ˜n : Rk →[0, ∞)k which satisfies the constraint of optimization problem
(17). Let ˜N(µ) := P
s∈[k] ˜ns(µ). Now, since the function ρ : x 7→
1
8R exp(−x ·
˜∆2
2 min(σ2s)s∈[k] ) is a
convex and decreasing function, by Jensen’s inequality we can say that

V2(˜n) =
X

i∈[k]

X

j̸=i

Z

˜Θij
ρ

˜ni(µ) + ˜nj(µ)

dH(µ)

≥
X

i∈[k]

X

j̸=i
H(˜Θij)ρ

 
1
H(˜Θij)

Z

˜Θij


˜ni(µ) + ˜nj(µ)

dH(µ)

!

(Jensen’s inequality for each integral on ˜Θij)

= H(˜Θ) ·
X

i∈[k]

X

j̸=i

H(˜Θij)

H(˜Θ)
ρ

 
1
H(˜Θij)

Z

˜Θij


˜ni(µ) + ˜nj(µ)

dH(µ)

!

≥H(˜Θ) · ρ



X

i∈[k]

X

j̸=i

H(˜Θij)

H(˜Θ)
1
H(˜Θij)

Z

˜Θij


˜ni(µ) + ˜nj(µ)

dH(µ)





(Jensen’s inequality)

≥H(˜Θ) · ρ





X

i∈[k]

X

j̸=i

1
H(˜Θ)

Z

˜Θij



X

s∈[k]
˜ns



dH(µ)






(˜ni + ˜nj ≤P

s∈[k] ˜ns and ρ is a decreasing function.)

≥H(˜Θ) · ρ

 
NV
H(˜Θ)

!

.
(Constraint of (17))

Since ˜n is chosen arbitrarily, we can say V min
2
≥H(˜Θ)ρ

NV
H( ˜Θ)


. One can check the above nopt

satisfies the constraint in the optimization problem (17) and also satisfies V2(nopt) = H(˜Θ)ρ( NV

H( ˜Θ)).

Therefore, nopt is an optimal solution of optimization problem (17).

Using Lemma 9, we can get

H(˜Θ) =
X

i̸=j

Z

Θij
1[|µi −µj| ≤˜∆]1[|µi|, |µj| ≤
1
p

˜∆
] dH(µ) = L′
ij(H, ˜∆).
(18)

Now applying Claim 1 and Eq. (18) on Optimization (17) implies the following result:

V min
2
= V2(nopt)
(Claim 1)

=
exp

−
NV
2L′(H, ˜∆)
˜∆2
2 min(σ2s)s∈[k]



8e4
X

i̸=j

Z

Θij
1[|µi −µj| ≤˜∆]1[|µi|, |µj| ≤
1
p

˜∆
] dH(µ)

(Eq. (18))

≥
exp

−
NV ˜∆
2 min(σ2s)s∈[k]L′(H, ˜∆)



8e4
× L′(H, ˜∆) ˜∆.
(Eq. (18))

If V ≤δ is true, then V2 ≤δ, which implies

20


---Page Break---
exp

−
NV ˜∆
2 min(σ2s)s∈[k]L′(H, ˜∆)



8e4
×L′(H, ˜∆) ≤δ ⇐⇒NV ≥2 min(σ2
s)s∈[k]L′(H, ˜∆)

˜∆
ln L′(H, ˜∆)

8e4δ
.

To make this lower bound greater than 0, ln L′(H, ˜∆)

8e4δ
> 1. From Lemma 9, we know that for small
enough ˜∆9, L′(H, ˜∆) ∈[ 1

2L(H), 2L(H)]. Setting

˜∆:= 32e4

L(H)δ,
(19)

we have

NV ≥min(σ2
s)s∈[k]
L(H)2

16e4
ln 2
(20)

which is the inequality we desired.

D.1
Proof of Lemmas

D.1.1
Proof of lemma 11

Proof. It is equivalent to prove p + q ≥1

4 exp(−d(p, 1 −q′)) for any p, q, q′ ∈(0, 1) such that
q′ ≤q. First, if p ≥1/4 or q ≥1/4 it trivially holds, and thus we assume p < 1/4 and q < 1/4. We
have
d(p, 1 −q′)d(p, 1 −q)
(p, q ≤1

4)

≤d(p + q, 1 −(p + q))
(p + q < 1

2)

≤log
1
2.4(p + q)
(Eq.(3) of Kaufmann et al. [2016a])

and transforming this yields

p + q ≥1

2.4e−(d(p,1−q′)).

This completes the proof.

D.1.2
Proof of the Lemma 12

Proof. We have

hi(a)hj(b)
hi(b)hj(a) = exp

 

−(a −mi)2

2σ2
i
+ (b −mi)2

2σ2
i
−(b −mj)2

2σ2
j
+ (a −mj)2

2σ2
j

!

= exp

 

−(a −b)(a + b −2mi)

2σ2
i
−(b −a)(a + b −2mj)

2σ2
j

!

= exp

 

2(a −b)

"
mi

σ2
i
−mj

σ2
j

#

−(a −b)(a + b)

"
1
2σ2
i
+
1
2σ2
j

#!

≥exp

 

−2|a −b|


mi

σ2
i
−mj

σ2
j

 −|(a −b)(a + b)|

"
1
2σ2
i
+
1
2σ2
j

#!

≥exp

 

−2δ


mi

σ2
i
−mj

σ2
j

 −2
√

δ

"
1
2σ2
i
+
1
2σ2
j

#!
1
R′
ij(H, δ).

Define R′(H, δ) = mini̸=j R′
ij(H, δ) which satisfies the condition of the Ratio lemma. For δ such
that

δ < D1(H) := min
i̸=j






mi

σ2
i
−mj

σ2
j



−1

,

"
1
2σ2
i
+
1
2σ2
j

#

−2



,

9 ˜∆< min

D0(H), mini∈[k]
1
4m2
i ,
L(H)
4(k−1) P
i∈[k]
1
ξi


. Check Section G for the condition.

21


---Page Break---
we have R′(H, δ) ≤e4.

E
Proof of Theorem 6

For notational convenience, define ∆s(µ) := µi∗(µ) −µs for s ∈[k], ∆(µ) := mins̸=i∗(µ) ∆s =
∆j∗(µ). Let A(t) be the subset of arms that have not been eliminated at time t.

Since δ <
4L2(H)
P

i∈[k]
k−1

ξi
, ∆0 =
δ
4L(H) ≤
L(H)
P

i∈[k]
k−1

ξi
and we have

Pµ∼H(∆0 ≥∆(µ)) =
X

i̸=j

Z

Θij
1[|µi −µj| ≤∆0] dH(µ)

=L(H, ∆0)∆0 ≤2L(H) · ∆0
(Lemma 9)

≤δ

2.
(21)

Next, we consider the upper bound estimator such that ˆ∆safe(t) ≥∆holds with high probability.
Namely,
UCB(i, t), LCB(i, t) = ˆµi(t) ± Conf(i, t),

Conf(i, t) =

s

2σ2
i
log(6(Ni(t))2/(( δ2

2k)π2))
Ni(t)
,

ˆ∆safe(t) = max
i
UCB(i, t) −max
j
LCB(j, t).

From the definition above, we can calculate how many arm pulls the learner needs to narrow down
the confidence width.
Lemma 13. Define B := 320maxi∈[k] σ2
i and ∆thr := min

log 4
√

k
δπ , 1

B

.
Let R0(∆) :=

B log min(∆,∆thr)−1

min(∆,∆thr)2
. Then, for any ∆∈(0, ∞) and for a timestep t which satisfies Ni(t) ≥R0(∆),
Conf(i, t) ≤∆/4.

Proof of Lemma 13. Only for this part of the proof, let ∆′ := min(∆, ∆thr) for notational conve-
nience. Then,

Conf(i, t) =

s

2σ2
i
log(6(Ni(t))2/(( δ2

2k)π2))
Ni(t)

≤

s

2σ2
i
log(6R0(∆)2/(( δ2

2k)π2))
R0(∆)
(assumption on t)

= ∆′q

2σ2
i

v
u
u
tlog(6(B × log(∆′−1)

∆′2
)2/(( δ2

2k)π2))

B × log(∆′−1)

≤∆′q

2σ2
i

s

log(6( B

∆′3 )2/(( δ2

2k)π2))

Blog(∆′−1)
(log ∆′−1 ≤∆′−1)

= ∆′q

2σ2
i

s

6 log(∆′−1) + 2 log B + log 12k

δ2π2
Blog(∆′−1)

≤∆′q

2σ2
i

r

6
B + 2

B + 2

B
(Definition of ∆thr, ∆thr ≥∆′)

= ∆′
r

20σ2
i
B
≤1

4∆′ ≤1

4∆.

22


---Page Break---
From this lemma, one could induce the following corollary which states that Algorithm 5 always
terminates before a certain timestep:

Corollary 14. Let τ (and γ) be the stopping time (and the last iteration of the while loop in Algorithm
1, respectively) where Algorithm 1 meets the stopping condition. Then, γ is always bounded by
R0(∆0) and τ is uniformly bounded by T0 := k · R0(∆0).

Proof of Corollary 14. Let us assume that τ
> T0 + 1.
Then, by Lemma 13, each i ∈
AT0 satisfies Conf(i, T0) ≤∆0/4.
Let iucb(t) = arg maxi∈A(t) UCB(i, t) and ilcb(t) =
arg maxi∈A(t) LCB(i, t). From definition,

ˆ∆safe(T0) =
max
i∈A(T0) UCB(i, T0) −max
i∈A(T0) LCB(i, T0)

= UCB(iucb(T0), T0) −LCB(ilcb(T0), T0)

= 2Conf(iucb(T0), T0) + 2Conf(ilcb(T0), T0) + LCB(iucb(T0), T0) −UCB(ilcb(T0), T0)

≤∆0 + LCB(iucb(T0), T0) −UCB(ilcb(T0), T0)
≤∆0
(Since both arms survived from the elimination phase)

which implies ˆ∆safe ≤∆0, contradicting τ ≥T0 since the algorithm should be terminated by Line
18 at timestep T0. Therefore, τ ≤T0.

Lemma 14 implies Algorithm 1 always stops before T0 samples. Morever, the following lemma states
that with high probability, true mean µ is in between UCB and LCB for all time steps.

Lemma 15. (Uniform confidence bound) The following holds for all i ∈[k]:

Pµ

" ∞
\

t=1
{LCB(i, t) ≤µi}

#

≥1 −δ2

2k ,
(22)

Pµ

" ∞
\

t=1
{µi ≤UCB(i, t)}

#

≥1 −δ2

2k .
(23)

Proof of Lemma 15. The following derives the upper bound part, Eq. (23). The lower bound is
derived by following the same steps.

Since each arm is independent of each other, Eq. (23) boils down to prove

Pµ




∞
\

s=1
{µi ≤UCB(i, ti(s))}



≥1 −δ2

2k ,

where ti(s) = min{t ∈N : Ni(t) ≥s} and ti(s) = ∞if {t ∈N : Ni(t) ≥s} = ∅. For each event
{µi ≤UCBi(ti(s))}, since each arm pull is independent of each other, by Hoeffding’s inequality
we have

Pµ



1

s

s
X

j=1
(Xi
j −µi) ≤−ϵ



≤exp

 

−sϵ2

2σ2
0

!

for any ϵ > 0. If we set ϵ = Conf(i, ti(s)), we can transform the above inequality to

Pµ
 
UCB(i, ti(s)) ≤µi

≤
δ2

2ks2 · 6

π2 .

By the union bound,

Pµ




∞
\

s=1
{µi ≤UCB(i, ti(s))}



≥1 −

∞
X

s=1
Pµ

µi ≥UCB(i, ti(s))


23


---Page Break---
≥1 −

∞
X

s=1

δ2

2ks2 · 6

π2

≥1 −

∞
X

s=1

δ2

2ks2 · 6

π2 = 1 −δ2

2k ,

and the proof is completed.

Let us define a good event based on Lemma 15 as

X(µ) :=
\

i∈[k]








∞
\

t=1
{LCB(i, t) ≤µi}



\



∞
\

t=1
{UCB(i, t) ≥µi}







.
(24)

We now prove that, under Hµ and under this good event X(µ)

• The best arm i∗(µ) is always in the active arm set A(t) for all t (Lemma 16),

• Each count of the suboptimal arm pull, Ni(T0), is bounded by roughly O( log ∆i
∆2
i ) (Lemma
17).
Lemma 16. Let

X ′(µ) =

∞
\

t=1
{i∗(µ) ∈A(t)} .

Then, under Hµ, X(µ) ⊂X ′(µ) and therefore

Pµ [(X ′(µ))c] ≤δ2.

Proof of Lemma 16. Suppose that event X(µ) occurs under Hµ. Then for all i ∈[k] and for all t,
ˆµi −Conf(i, t) ≤µi and ˆµi + Conf(i, t) ≥µi. Now, for any i ̸= i∗,

LCB(i, t) −UCB(i∗, t) = ˆµi −Conf(i, t) −(ˆµi∗+ Conf(i∗, t))
≤µi + Conf(i, t) −Conf(i, t) −(µi∗−Conf(i∗, t) + Conf(i∗, t))
(Event X occurs)
= µi −µi∗< 0

which means when event X occurs, the optimal arm will never be dropped, and thus X ⊂X ′. By
Lemma 15,
Pµ(X ′(µ)) ≥Pµ(X(µ)) ≥1 −δ2.

Lemma 17. For any i ̸= i∗, under Hµ we have

Ni(T0) > R0
 
max(∆i, ∆0)
	
⊂X(µ)c,
(25)

and therefore, Eµ[Ni(T0)1[X(µ)]] ≤R0
 
max(∆i, ∆0)

.

Proof of Lemma 17. Only for this part of the proof, let Ti := R0
 
max(∆i, ∆0)

for brevity.

When ∆i < ∆0, max(∆i, ∆0) = ∆0, which, combined with Corollary 14, implies that
{Ni(T0) ≥Ti} always holds.

For the case of ∆i > ∆0, suppose that the learner is under the events X(µ) and {i ∈A(Ti)}. Note
that
Conf(a, Ti) ≤∆i

4 ,
∀a ∈A(Ti).

Then,

max
a∈A(Ti) LCBa(Ti) −UCBi(Ti) ≥LCBi∗(Ti) −UCBi(Ti)

24


---Page Break---
= ˆµi∗−Confi∗(Ti) −(ˆµi + Confi(Ti))
≥µi∗−2Confi∗(Ti) −(µi + 2Confi(Ti))
(X occurs)
≥µi∗−µi −∆i = 0.

Therefore, when X occurs, µi should be eliminated after timestep Ti so Ni(T0) ≤Ti.

Lemmas 16 and 17 guarantee the Bayesian δ-correctness of our Algorithm 1.
Theorem 18. (δ-correctness) The Bayesian PoE of Algorithm 1 is at most δ, i.e.,
Z

Rk PoE(µ) dH(µ) ≤δ.
(26)

Proof of Theorem 18. Throughout the proof, we use the following results.

• The probability that µi∗(µ) −µj∗(µ) ≤∆0 is at most δ
2 by Eq. (21).

• The event X fails to hold with probability at most δ2 by Lemma 15.

• When µi∗(µ) −µj∗(µ) > ∆0 and event X occurs, by Lemmas 16 and 17, all suboptimal
arms will be eliminated before Tj∗and only the optimal arm will remain in the set. This
means E(µ) ⊂X(µ) ∪{µi∗(µ) −µj∗(µ) ≤∆0}.

Therefore,

PoE(π; H) = Eµ∼H


P

J ̸= i∗(µ)|Hµ

= Eµ∼H
h
E[1(E(µ))|Hµ]
i

≤Eµ∼H
h
E[1({µi∗(µ) −µj∗(µ) ≤∆0}) + 1(X(µ))|Hµ]
i

≤Eµ∼H
h
1({µi∗(µ) −µj∗(µ) ≤∆0}) + E[1(X(µ))|Hµ]
i

≤δ

2 + δ2 < δ.

Finally, Lemma 19 shows the upper bound of the expected stopping time of our algorithm.

Lemma 19. We have E[τ] ≤B0
L(H)2

δ
log

L(H)

δ

+ O(log δ−1).

Proof. We have

E[τ] =

k
X

s=1
E[Ns(T0)] =

k
X

s=1
E[Eµ[Ns(T0)]]

=

k
X

s=1
E
h
Eµ[Ns(T0)1[X(µ)]]
i
+

k
X

s=1
E
h
Eµ[Ns(T0)1[X c(µ)]]
i

≤

k
X

s=1
E
h
Eµ[Ns(T0)1[X(µ)]]
i
+ T0 · δ2
(Corollary 14 and Lemma 15)

and since ∆0 < ∆thr by assumption,

T0 · δ2 = k · R0(∆0) · δ2 ≤kB · log ∆−1
0
∆2
0
· δ2.
(27)

Therefore, it remains to compute the scale of the first term, Pk
s=1 E
h
Eµ[Ns(T0)1[X(µ)]
i
.

25


---Page Break---
The following evaluates E
h
Eµ[Ni(T0)1[X(µ)]]
i
for each i. For notational convenience, let Ts(µ) :=

Eµ
h
R0(max
 
∆s, ∆0)

1[X(µ)]
i
. Then,

E
h
Eµ[Ns(T0)1[X(µ)]]
i
≤E[Ts(µ)]
(Lemma 17)

=

k
X

i=1
E[Ts(µ)1[µ ∈Θi]]

=

k
X

i=1

Z

µ∈Θi
Ts(µ) dH(µ)
(28)

(Recall that Θi = {µ ∈Rk : µi ≥maxj̸=i µj}).
In the following, we calculate each
R

µ∈Θi Ts(µ) dH(µ). By assumption, ∆0 < ∆thr.

For this part, we define a new notation that for each vector v ∈Rk and i, j ∈[k], v\i (and v\i,j)
is the projection of v to Rk−1 (Rk−2, respectively) by omitting i-th coordinate (i, j-th coordinate,
respectively), v\i := (v1, v2, · · · , vi−1, vi+1, · · · , vk)
|
{z
}
k−1 coordinates

. Similarly, for a k-dimensional distribution H

and i, j ∈[k], H\i (and H\i,j) is the distribution which omits i-th (i, j-th, respectively) coordinate.

Case 1: s ̸= i
In this case, by Lemma 17,

Z

µ∈Θi
Ts(µ) dH(µ) =
Z

µ∈Θi
Ts(µ)(1[µi −µs ≥∆0] + 1[µi −µs < ∆0]) dH(µ)

=
Z

µi∈R

Z µi−∆thr

µs=−∞

Z

µ\i,s∈(−∞,µi)k Ts(µ)hi(µi)hs(µs) dH\i,s dµs dµi

+
Z

µi∈R

Z µi−∆0

µs=µi−∆thr

Z

µ\i,s∈(−∞,µi)k Ts(µ)hi(µi)hs(µs) dH\i,s dµs dµi

+
Z

µi∈R

Z µi

µs=µi−∆0

Z

µ\i,s∈(−∞,µi)k Ts(µ)hi(µi)hs(µs) dH\i,s dµs dµi

≤B · log ∆−1
thr
∆2
thr

+
Z

µi∈R

Z µi−∆0

µs=µi−∆thr

Z

µ\i,s∈(−∞,µi)k B log(µi −µs)−1

(µi −µs)2
hi(µi)hs(µs) dH\i,s dµs dµi

+
Z

µi∈R

Z µi

µs=µi−∆0

Z

µ\i,s∈(−∞,µi)k B log ∆−1
0
∆2
0
hi(µi)hs(µs) dH\i,s dµs dµi

(Lemma 17)

=
Z

µi∈R

Z µi−∆0

µs=−µi−∆thr
B log(µi −µs)−1

(µi −µs)2
hi(µi)hs(µs)




Y

k∈[k]\{i,s}
Hk(µi)



dµs dµi

+ B log ∆−1
0
∆2
0
P(i∗(µ) = i, µi −µs ≤∆0) + B · log ∆−1
thr
∆2
thr

≤B log ∆−1
0

Z

µi∈R
hi(µi)
Z µi−∆0

µs=−∞

1
(µi −µs)2 hs(µs)




Y

k∈[k]\{i,s}
Hk(µi)



dµs dµi

|
{z
}
(A)

26


---Page Break---
+ B log ∆−1
0
∆2
0
P(i∗(µ) = i, µi −µs ≤∆0)
|
{z
}
(Pis)

+B log ∆−1
thr
∆2
thr
.

We first deal with the term (A). The following splits (A) into the sum of two integrals (A1) and
(A2):

(A) =
Z

µi∈R
hi(µi)




Y

k∈[k]\{i,s}
Hk(µi)




Z µi−∆0

µs=−∞

1
(µi −µs)2 hs(µs) dµs dµi

=

⌈
1
2√

∆0 ⌉−1
X

l=1

Z

µi∈R
hi(µi)




Y

k∈[k]\{i,s}
Hk(µi)




Z µi−l∆0

µs=µi−(l+1)∆0

1
(µi −µs)2 hs(µs) dµs dµi

|
{z
}
(A1)

+
Z

µi∈R
hi(µi)




Y

k∈[k]\{i,s}
Hk(µi)




Z µi−⌈
1
2√

∆0 ⌉∆0

µs=−∞

1
(µi −µs)2 hs(µs) dµs dµi

|
{z
}
(A2)

.

For (A1), we have

(A1) ≤

⌈
1
2√

∆0 ⌉−1
X

l=1

Z

µi∈R
hi(µi)




Y

k∈[k]\{i,s}
Hk(µi)




Z µi−l∆0

µs=µi−(l+1)∆0

1
l2∆2
0
hs(µs) dµs dµi

=

⌈
1
2√

∆0 ⌉−1
X

l=1

1
l2∆2
0
P(i∗(µ) = i, µi −µs ∈[l∆0, (l + 1)∆0])

≤

⌈
1
2√

∆0 ⌉−1
X

l=1

1
l2∆2
0
· 2P(i∗(µ) = i, µi −µs ≤∆0)
(Lemma 20)

≤

∞
X

l=1

1
l2∆2
0
· 2Pis

= π2

3∆2
0
Pis.
(29)

Now for (A2), we evaluate the inner integral of (A2):

(Inner −A2) :=
1
√

2πσs

Z µi−⌈
1
2√

∆0 ⌉∆0

−∞

1
(µi −µs)2 exp

 

−(µs −ms)2

2σ2s

!

dµs

=
1
√

2πσs


1
(µi −µs) exp

 

−(µs −ms)2

2σ2s

!µi−⌈
1
2√

∆0 ⌉∆0

−∞

+
1
√

2πσs

Z µi−⌈
1
2√

∆0 ⌉∆0

−∞

µs −ms
σ2s(µi −µs) exp

 

−(µs −ms)2

2σ2s

!

dµs

(partial integration)

≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
√

2πσ3s

Z µi−⌈
1
2√

∆0 ⌉∆0

−∞

µi −ms

µi −µs
−1

exp

 

−(µs −ms)2

2σ2s

!

dµs

(µi > µs)

27


---Page Break---
≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
√

2πσ3s

Z µi−⌈
1
2√

∆0 ⌉∆0

−∞

µi −ms
⌈
1
2√∆0 ⌉∆0
exp

 

−(µs −ms)2

2σ2s

!

dµs

≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+ max



1

σ2s

µi −ms
⌈
1
2√∆0 ⌉∆0
, 0



.

By integrating above over variable i, we have

(A2) ≤
Z

R
hi(µi)




Y

k∈[k]\{i,s}
Hk(µi)








1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+ max



1

σ2s

µi −ms
⌈
1
2√∆0 ⌉∆0
, 0







dµi

≤
Z

R
hi(µi)




1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+ max



1

σ2s

µi −ms
⌈
1
2√∆0 ⌉∆0
, 0







dµi
(Fk(·) ≤1)

=
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
σ2s⌈
1
2√∆0 ⌉∆0

Z

R
hi(µi) max

µi −ms, 0

dµi



≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
σ2s⌈
1
2√∆0 ⌉∆0

Z

R
hi(µi)|µi −ms| dµi



≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
σ2s⌈
1
2√∆0 ⌉∆0

Z

R
hi(µi)
 
|µi −mi| + |mi −ms|

dµi



≤
1
√

2πσs

1
⌈
1
2√∆0 ⌉∆0
+
1
σ2s⌈
1
2√∆0 ⌉∆0
(|mi −ms| + σi
√

2
√π )

(Mean of Half normal distribution is σi
√

2
√π )

=
1
⌈
1
2√∆0 ⌉∆0

"
1
√

2πσs
+ 1

σ2s
(|mi −ms| + σi
√

2
√π )

#

|
{z
}
Sis(H)/2

≤Sis(H)
√∆0
= O(∆−1/2
0
).
(30)

Therefore, by Eq. (29) and Eq. (30),

(A) = (A1) + (A2) = 1

∆2
0
· π2

3 Pis + O(∆−1/2
0
),

and therefore

Z

µ∈Θi
Ts(µ) dH(µ) ≤B log ∆−1
0
∆2
0
Pis

π2

3 + 1

+ O(∆−1/2
0
).
(31)

Case 2: s = i
In this case, let
Z

µ∈Θs
Ts(µ) dH(µ) =
X

j̸=s

Z

µ∈Θsj
Ts(µ) dH(µ)

≤
X

j̸=s

Z

µ∈Θsj
max
i̸=s (Ti(µ)) dH(µ)

(Ns(t) increases only when |At| > 1, so there should be a competitor)

≤
X

j̸=s

Z

µ∈Θsj
max
i̸=s

R0(∆i(µ), ∆0)

dH(µ)
(Lemma 17)

≤
X

j̸=s

Z

µ∈Θsj

h
R0(∆j(µ), ∆0)
i
dH(µ)

28


---Page Break---
and by the same calculation as Case 1, we obtain
Z

µ∈Θsj

h
R0(∆j(µ), ∆0)
i
dµ ≤B log ∆−1
0
∆2
0
Psj

π2

3 + 1

+ O(∆−1/2
0
),

and therefore

Z

µ∈Θs
Ts(µ) dH(µ) ≤B log ∆−1
0
∆2
0

π2

3 + 1
 X

j̸=s
Psj + O(∆−1/2
0
).
(32)

For notational convenience, let B0 = B ·
h
π2

3 + 1
i
. From Eq. (28), Eq. (31), Eq. (32), we get

E[Ns(T0)] = Eµ∼H
h
Eµ[Ns(T0)1[X(µ)]]
i
+ Eµ∼H
h
Eµ[Ns(T0)1[X(µ)c]]
i
(33)

≤

k
X

i=1

Z

µ∈Θi
Ts(µ) dH(µ) + Eµ∼H
h
Eµ[Ns(T0)1[X(µ)c]]
i
(Eq. (28))

≤
X

i̸=s

Z

µ∈Θi
Ts(µ) dH(µ) +
Z

µ∈Θs
Ts(µ) dH(µ) + Eµ∼H
h
Eµ[Ns(T0)1[X(µ)c]]
i

≤B0 log ∆−1
0
∆2
0



X

i̸=s
Pis +
X

j̸=s
Psj



+ O(∆−1/2
0
) + Eµ∼H
h
Eµ[Ns(T0)1[X(µ)c]]
i

(Eq. (31) and (32))

= B0 log ∆−1
0
∆2
0

h
P(i∗(µ) ̸= s, µi∗(µ) −µs ≤∆0) + P(i∗(µ) = s, µs −µj∗(µ) ≤∆0)
i

+ O(∆−1/2
0
) + Eµ∼H
h
Eµ[Ns(T0)1[X(µ)c]]
i
.
(34)

Now, the total stopping time is bounded as follows:

E[τ] = E[

k
X

s=1
Ns(T0)]

= B0 log ∆−1
0
∆2
0





k
X

s=1
P(i∗(µ) ̸= s, µi∗(µ) −µs ≤∆0)

|
{z
}
Psum1

+

k
X

s=1
P(i∗(µ) = s, µs −µj∗(µ) ≤∆0)

|
{z
}
Psum2





+ O(∆−1/2
0
) + E[τ1[X c]].
(Eq. (34))

The final task we have left is bounding (Psum1) and (Psum2). Let us define k∗(µ) as the third best
arm in µ. For (Psum1),

(Psum1) =

k
X

s=1
P(i∗(µ) ̸= s, µi∗(µ) −µs ≤∆0)

=

k
X

s=1
P(j∗(µ) = s, µi∗(µ) −µs ≤∆0) +

k
X

s=1
P(j∗(µ) ̸= s, µi∗(µ) −µs ≤∆0)

≤P(µi∗(µ) −µj∗(µ) ≤∆0) +

k
X

s=1
P(µi∗(µ) −µk∗(µ) ≤∆0)

(When j∗(µ) ̸= s, µk∗(µ) ≥s)

29


---Page Break---
≤δ

2 + k · P(µi∗(µ) −µk∗(µ) ≤∆0)
(Definition of ∆0)

≤δ

2 + O(∆2
0).
(Lemma 21)

For (Psum2),

(Psum2) =

k
X

s=1
P(i∗(µ) = s, µs −µj∗(µ) ≤∆0)

= P(µi∗(µ) −µj∗(µ) ≤∆0)

= P(∆(µ) ≤∆0) ≤δ

2
(Definition of ∆0)

and finally we can conclude

E[τ] ≤B0 log ∆−1
0
∆2
0
δ + O(∆−1/2
0
) + E[τ1[X c]]
(Eq. (34), (Psum1) and (Psum2))

≤B0 log ∆−1
0
∆2
0
δ + O(∆−1/2
0
) + KB log ∆−1
0
∆2
0
· δ2
(Eq.(27))

= B0 log ∆−1
0
∆2
0
δ + O(∆−1/2
0
),
(35)

and the proof is completed.

E.1
Bound that holds for any δ

In this case, we need to change the definition of ∆0 as

∆0 := min

 

max

∆∈(0, 1) : L(H, ∆)∆≤δ

2


, min
i̸=j (ξiLij(H))2
!

.
(36)

Assume that ∆0 < ∆thr. Then, all the proof flows of Section E after Eq. (21) follows accordingly,
and we obtain the following upper bound:

E[τ] ≤B0 log ∆−1
0
∆2
0
δ +
2 P

i̸=j Sij(H)
√∆0
|
{z
}
(35)

+ KB0 log ∆−1
0
∆2
0
δ2

|
{z
}
(27)

+ B0



X

i̸=j̸=k
Qijk



log ∆−1
0

|
{z
}
From (Psum1)

+k2·B0 log ∆−1
thr
∆2
thr
,

(37)
where Sij and Qijk are defined in Eq. (30) and Lemma 21, respectively.

Eq. (37) also holds for the case of ∆0 ≥∆thr. In this case, from the definition of R0, we have

Ts(µ) = Eµ
h
R0(max
 
∆s, ∆0)

1[X]
i
= Eµ

R0(∆thr)1[X]

≤R0(∆thr)Pµ[X] = R0(∆thr),

which implies

E[τ] ≤k · R0(∆thr) + T0 · δ2

= k · R0(∆thr) · (1 + δ2)
(T0 = k · R0(∆0) = R0(∆thr))

= k · B0 log ∆−1
thr
∆2
thr
· (1 + δ2)

≤2k B0 log ∆−1
thr
∆2
thr
≤Eq. (37).

In summary, we obtain Eq. (37).

30


---Page Break---
E.2
Proof of Lemmas

E.2.1
Proof of Lemma 20

Lemma 20. For S + 1 ≤
1
√

δ and for δ ≤
 
ξiLis(H)
2, we have

P(i∗(µ) = i, µi −µs ∈[Sδ, (S + 1)δ]) ≤2P(i∗(µ) = i, µi −µs ≤δ).

Proof. We have
Z

Θi
1 [|µi −µs| ∈[Sδ, (S + 1)δ]] dH(µ) =
Z

Θ\i

Z µs+(S+1)δ

µi=µs+Sδ
hi(µi) dµi dH\i(µ\i)

≤
Z

Θ\i
δ

hi(µs) + e−1/2

ξi
(S + 1)δ

dH\i(µ\i)

(by Lipschitz property of Gaussian, e−1/2/ξi is the steepest slope of N(mi, ξ2
i ))

= δ(

"Z

Θ\i
hi(µs) dH(µ\i)

#

|
{z
}
Lis(H)

+e−1/2

ξi
(S + 1)δ)

≤(Lis(H) + 1

ξi
(S + 1)δ)δ.

When S + 1 <
1
√

δ and
√

δ < Lis(H)ξi, we have

Z

Θi
1 [|µi −µj| ∈[Sδ, (S + 1)δ]] dH(µ) ≤2Lis(H)δ

as intended.

E.2.2
Proof of Lemma 21

We will show that the probability that three or more arms are δ-close is O(δ2). Namely:

Lemma 21. We have P(µi∗(µ) −µk∗(µ) ≤∆0) = O(δ2).

Proof. For any i ̸= j ̸= k ∈[k], we have
Z
1 [|µi −µj|, |µi −µk| ≤δ] dH(µ)

=
Z

µ\jk

Z µi+δ

µj=µi−δ

Z µi+δ

µk=µi−δ
hj(µj)hk(µk) dµk dµj dH\jk(µ\jk)

≤
Z

µ\jk

Z µi+δ

µj=µi−δ

Z µi+δ

µk=µi−δ

 

hj(µi) + e−1/2δ

ξj

!
hk(µi) + e−1/2δ

ξk


dµk dµj dH\jk(µ\jk)

(Lipschitz property of Gaussian)

≤(2δ)2
Z

µ\jk

h
hj(µi)hk(µi) + O(δ)
i
dH\jk(µ\jk)

|
{z
}
=:Qijk(H)

= O(δ2).

F
Experimental details

The code used in the experiments for this paper can be found at the following link: https://
github.com/jajajang/FC_BAI_Bayes.

31


---Page Break---
F.1
Stopping condition

TTTS
We use our theoretical results stated in Section 5 for our stopping criterion. For TTTS, we
use Chernoff’s stopping rule, as Garivier and Kaufmann [2016], Jourdan et al. [2022] did. Here is the
description of how it works: for each arm i, j ∈[k], let

ˆµij(t) :=
Ni(t)
Ni(t) + Nj(t) ˆµi(t) +
Nj(t)
Ni(t) + Nj(t) ˆµj(t),

and define
Zij(t) := Ni(t) · KLi(ˆµi, ˆµij) + Nj(t) · KLj(ˆµj, ˆµij).

Now the stopping time is defined as:

τT T T S := inf{t ∈N : max
a∈[k] min
b∈[k]\a Zab(t) ≥β(t, δ)}

for some threshold function β(t, δ), which is defined by the following proposition of Garivier and
Kaufmann [2016]:

Theorem 22 (Garivier and Kaufmann 2016, Proposition 12). Let µ be an exponential family bandit
model. Let δ ∈(0, 1) and α > 1. There exists a constant C = C(α, k) such that whatever the
sampling strategy, using Chernoff’s stopping rule with the threshold

β(t, δ) = log Ctα

δ

ensures that for all µ, Pµ
 
τ < ∞, J ̸= i∗(µ)

≤δ.

In this theorem, we give an advantage to the stopping time of TTTS by setting α = 1, C = 1.
Theoretically, C(α, k) > 1 and C →∞as α →1+, but we set the threshold smaller than the
theoretical guarantee so that TTTS stops earlier.

TTUCB
We followed the stopping rule of the original paper Jourdan and Degenne [2022b]. Let
CG(x) := minλ∈( 1

2 ,1]
2λ−2λ log(4λ)+log ζ(2λ)−0.5 log(1−λ)+x

λ
where ζ is a Riemann ζ function, and

c(n, δ) := 2CG(1

2 log k −1

δ
) + 4 log(4 + log n

2 ).

Let ˆit := arg maxi∈[k] ˆµi(t), the empirical best arm at step t. The TTUCB algorithm stops when

min
i̸=ˆit

ˆµˆi(t) −ˆµi(t)
q

1
Nˆit(t) +
1
Ni(t)
≥
p

c(t, δ).

When the algorithm stops sampling, the TTUCB algorithm recommends the empirical best arm as its
final suggestion.

Since the computation of CG(x) involves optimization, it is computationally heavy when the number
of samples is excessively large (as our Table 1). Instead, we approximated CG(x) ≈x + log x as
mentioned in Jourdan and Degenne [2022b].

F.2
NoElim algorithm

The NoElim algorithm is shown in Algorithm 2.

F.3
Tables including computation time

For all tables in this section, Comp represents the average computation time (second).

32


---Page Break---
Algorithm 2 No Elimination (NoElim) Algorithm

Input: Confidence level δ, prior H
∆0 :=
δ
4L(H)
Initialize the candidate of best arms A(1) = [k]
t = 1
while True do

Draw each arm in [k] once. {Main Difference with Algorithm 1}.
t →t + |A(t)| ˆ∆safe(t).
for i ∈A(t) do

Calculate UCB(i, t) and LCB(i, t) from (5).
if UCB(i, t) ≤maxj LCB(j, t) then

A(t) ←A(t) \ {i}.
end if
end for
if |A(t)| = 1 then

Return arm J in A(t).
end if
Calculate safe empirical gap
if ˆ∆safe(t) ≤∆0 then

Return arm J which is uniformly sampled from A(t).
end if
end while

Table 6: Extended version of Table 1 with computation time.

AVG
MAX
ERROR
COMP.

ALG. 1
1.06 × 104
2.35 × 105
1.5%
0.17
TTTS
1.56 × 105
1.09 × 108
0.5%
27.6
TTUCB
1.95 × 105
1.13 × 108
0%
5.07

Table 7: Extended version of Table 2 with computation time.

AVG
MAX
ERROR
COMP.

ALG. 1
2.69 × 105
1.66 × 107
0.6%
1.59
NOELIM
1.29 × 106
8.25 × 107
0%
5.5

F.4
Additional experiment results - Multiple arms, different prior mean/variance

In Section 6, we only used k = 2 arms for the simulation of Table 1. We made a brief comparison
between Algorithm 1 and TTUCB in k = 10 arm environment with a prior distribution where prior
means and variances are all different.

• Number of arms K = 10

• Prior means: we sample 10 random numbers from N(0, 1) before the experiment starts, and
set them as prior means. Here is the list of prior means: [-0.053 0.528 -0.332 -0.368 -0.273
0.909 0.418 -1.17 0.873 -0.405]

• Prior variance: we sample 10 random numbers from Unif([0.5, 1.5]) before the experiment
starts, and set them as prior means. Here is the list of prior std: [0.604 1.477 1.163 0.988
0.560 0.513 0.997 1.332 0.828 0.833]

• Instance variance: we sample 10 random numbers from Unif([0.5, 1.5]) before the exper-
iment starts, and set them as prior means. Here is the list of instance std: [1.498, 1.262,
1.485, 0.963, 1.375, 0.969, 1.357, 1.238, 1.088, 0.699]

• Number of experiments: 500

33


---Page Break---
• We stop additional sampling of TTUCB when its number of samples is over 108 because
of the time constraint. This means we gave some ‘advantage’ on TTUCB about expected
stopping time (since it makes TTUCB stop earlier than it should.)

Result
Our algorithm had the average stopping time of 1.1 × 105, while TTUCB had the average
stopping time of 7.1 × 105, again proving the superiority of Algorithm 1 over TTUCB in Bayesian
settings.

Table 8: Multiple arms.

AVG
MAX
ERROR
COMP.

ALG. 1
1.1 × 105
2.58 × 106
1.5%
1.52
TTUCB
7.1 × 105
108(CAPPED)
0.1%
6.22

F.5
Miscellaneous

Computation of ∆0
From Definition 2,

Lij(H) :=
Z ∞

−∞
hi(x)hj(x)
Y

s:s∈[k]\{i,j}
Hs(x) dx.

In Scipy package, there are functions for computing the cumulative function of Gaussian Hs
(scipy.norm.cdf) and hi (scipy.norm.pdf). Scipy package also supports the numerical integration
(scipy.integrate.quad) which we use to numerically compute Lij in our experiments.

Codes
The codes are in the following GitHub repository: https://github.com/jajajang/
FC_BAI_Bayes.

Hardware
We used Python 3.7 as our programming language and Macbook Pro M2 16 inch as our
hardware.

G
Scale restriction on δ for each theorem

The second result of the Lemma 9 is used for the lower bound. For this result to hold, we need the
following two conditions for D1:

• Proof of Lemma 1, second result: For the proof of Eq. (9), we used |mi| ≤
1
2
√

∆.

• Proof of Lemma 1, second result:
For the proof of Eq. (9), we used
1
ξi ∆2
>

2 exp

−
1
8∆ξ2
i


+ 2 exp

−
1
8∆ξ2
j


. To satisfy this condition, ∆< D0(H) where

D0(H) :=





W(−
1
32 maxi∈[k] ξ3/2
i
)
If maxi∈[k] ξi >
3q

e2
210

1
Otherwise
.

Here W is the Lambert W function with the principal branch.

For the lower bound proof, we consider sufficiently small δ subject to the following constraints on
˜∆=
32e4
L(H)δ:

• Two conditions above for Lemma 9.

• For the Lemma 12: ˜∆< D1(H) := mini̸=j

" mi

σ2
i −mj

σ2
j


−1
,

1
2σ2
i +
1
2σ2
j


−2
#

.

• To make L′(H, ˜∆) ˜∆∈( 1
2L(H), 2L(H)), ˜∆≤
L(H)
4P

i∈[k]
k−1

ξi
.

34


---Page Break---
In summary, Theorem 1 holds for any

δ ≤δL := L(H)

32e4 · min

 

D0(H), D1(H), min
i∈[k]
1
4m2
i
,
L(H)
4(k −1)P

i∈[k]
1
ξi

!

.

For the upper bound proof (Theorem 6), we consider δ such that ∆0 =
δ
4L(H) satisfies the following
conditions:

• ∆0 <
L(H)
P

i∈[k]
k−1

ξi
to make L(H, ∆0) · ∆0 ≤2L(H)∆0 by the first result of Lemma 9.

• ∆0 ≤mini,j∈[k],i̸=j(Lijξi)2 for the proof and usage of Lemma 20, and
• ∆0 < ∆thr.

35


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We accurately present the paper’s contributions and scope in the abstract and
introduction.

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

Justification: We present our limitations and possible future works in discussion and future
works.

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

36


---Page Break---
Answer: [Yes]

Justification: We provide the full set of assumptions and problem settings in Section 2 and
Appendix G. Additionally, we include all the proofs in the Appendix. Throughout multiple
revisions, we have ensured the correctness of our proofs.

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
Justification: We provide all the experimental details in Section 6 and Appendix F. Addi-
tionally, we provide our code to facilitate the reproduction of our results.

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

37


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We provide open access to the data and code with sufficient instructions.
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
Justification: We specify all the experimental details in Section 6 and Appendix F.
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
Justification: In Section 6 we provide average stopping time, maximum stopping time, and
the ratio of misidentification to report our result statistically.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

38


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

Answer: [Yes]

Justification: We provide the full information on the computer resource in Appendix F. In
addition, we provide the full table in Appendix F to present our time of execution.

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

Justification: This research conforms with the NeurIPS Code of Ethics.

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

Justification: There is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

39


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
Justification: This work poses no such risks.
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
Justification: This paper does not use existing assets.
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

40


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper does not release new assets.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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

41


---Page Break---
