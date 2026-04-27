Lookback Prophet Inequalities

Ziyad Benomar
ENSAE, Ecole Polytechnique,
FairPlay joint team
ziyad.benomar@ensae.fr

Dorian Baudry
Department of Statistics,
University of Oxford
dorian.baudry@ox.ac.uk

Vianney Perchet
CREST, ENSAE, Criteo AI LAB
Fairplay joint team
vianney.perchet@normalesup.org

Abstract

Prophet inequalities are fundamental optimal stopping problems, where a decision-
maker observes sequentially items with values sampled independently from known
distributions, and must decide at each new observation to either stop and gain the
current value or reject it irrevocably and move to the next step. This model is
often too pessimistic and does not adequately represent real-world online selection
processes. Potentially, rejected items can be revisited and a fraction of their value
can be recovered. To analyze this problem, we consider general decay functions
D1, D2, . . ., quantifying the value to be recovered from a rejected item, depending
on how far it has been observed in the past. We analyze how lookback improves, or
not, the competitive ratio in prophet inequalities in different order models. We show
that, under mild monotonicity assumptions on the decay functions, the problem
can be reduced to the case where all the decay functions are equal to the same
function x 7→γx, where γ = infx>0 infj≥1 Dj(x)/x. Consequently, we focus on
this setting and refine the analyses of the competitive ratios, with upper and lower
bounds expressed as increasing functions of γ.

1
Introduction

Optimal stopping problems constitute a classical paradigm of decision-making under uncertainty
[Dynkin, 1963] Typically, in online algorithms, these problems are formalized as variations of the
secretary problem [Lindley, 1961] or the prophet inequality [Krengel and Sucheston, 1977]. In the
context of the prophet inequality, the decision-maker observes a finite sequence of items, each having
a value drawn independently from a known probability distribution. Upon encountering a new item,
the decision-maker faces the choice of either accepting it and concluding the selection process or
irreversibly rejecting it, with the objective of maximizing the value of the selected item. However,
while the prophet inequality problem is already used in scenarios such as posted-price mechanism
design [Hajiaghayi et al., 2007] or online auctions [Syrgkanis, 2017], it might present a pessimistic
model of real-world online selection problems. Indeed, it is in general possible in practice to revisit
previously rejected items and potentially recover them or at least recover a fraction of their value.

Consider for instance an individual navigating a city in search of a restaurant. When encountering
one, they have the choice to stop and dine at this place, continue their search, or revisit a previously
passed option, incurring a utility cost that is proportional to the distance of backtracking. In another
example drawn from the real estate market, homeowners receive offers from potential buyers. The
decision to accept or reject an offer can be revisited later, although buyer interest may have changed,
resulting in a potentially lower offer or even a lack of interest. Lastly, in the financial domain, an

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
agent may choose to sell an asset at its current price or opt for a lookback put option, allowing them
to sell at the asset’s highest price over a specified future period. To make a meaningful comparison
between the two, one must account for factors such as discounting (time value of money) and the
cost of the option.

1.1
Formal problem and notation

To encompass diverse scenarios, we propose a general way to quantify the cost incurred by the
decision-maker for retrieving a previously rejected value.
Definition 1.1 (Decay functions). Let D = (D1, D2, . . .) be a sequence of non-negative functions
defined on [0, ∞). It is a sequence of decay functions if

(i) D1(x) ≤x for all x ≥0,

(ii) the sequence (Dj(x))j≥1 is non-increasing for all x ≥0,

(iii) the function Dj is non-decreasing for all j ≥1.

In the context of decay functions D, if a value x is rejected, the algorithm can recover Dj(x) after j
subsequent steps. The three conditions defining decay functions serve as fundamental prerequisites
for the problem. The first and second conditions ensure that the recoverable value of a rejected item
can only diminish over time, while the final condition implies that an increase in the observed value x
corresponds to an increase in the potential recovered value. Although the non-negativity of the decay
functions is non-essential, we retain it for convenience, as we can easily revert to this assumption by
considering that the algorithm has a reward of zero by not selecting any item.

The motivating examples that we introduced can be modeled respectively with decay functions
of the form Dj(x) = x −cj where (cj)j≥1 is a non-decreasing positive sequence, Dj(x) = ξjx
with ξj ∼B(pj) and (pj)j≥1 a non-increasing sequence of probabilities, and Dj(x) = λjx with
λ ∈[0, 1]. In one of these examples (housing market), the natural model is to use random decay
functions: the buyer makes the same offer if they are still interested, and offers 0 otherwise. Definition
1.1 can be easily extended to consider this case. However, to enhance the clarity of the presentation,
we only discuss the deterministic case in the rest of the paper. In Appendix D, we explain how all the
proofs and theorems can be generalized to that case.

The D-prophet inequality.
For any decay functions D, we define the D-prophet inequality problem,
where the decision maker, knowing D, observes sequentially the values X1, . . . , Xn, with Xi drawn
from a known distribution Fi for all i ∈[n]. If they decide to stop at some step τ, then instead of
gaining Xτ as in the classical prophet inequality, they can choose to select the current item Xτ and
have its full value, or select any item Xi with i < τ among the rejected ones but only recover a
fraction Dτ−i(Xi) ≤Xi of its value. Therefore, if an algorithm ALG stops at step τ its reward is

ALGD(X1, . . . , Xn) = max{Xτ, D1(Xτ−1), D2(Xτ−2), . . . , Dτ−1(X1)}
=
max
0≤i≤τ−1{Di(Xτ−i)} ,

with the convention D0(x) = x. If the algorithm does not stop at any step before n, then its reward is
ALGD(X1, . . . , Xn) = max1≤i≤n{Di(Xπ(n−i+1))}, which corresponds to τ = n + 1.
Remark 1.1. As in the standard prophet inequality, an algorithm is defined by its stopping time, i.e.,
the rule set to decide whether to stop or not. Hence, if D and D′ are two different sequences of decay
functions, any algorithm for the D-prophet inequality, although its stopping time might depend on the
particular sequence of functions D, is also an algorithm for the D′-prophet inequality. Consider for
example an algorithm ALG with stopping time τ(D) that depends on D. Its reward in the D′-prophet
inequality is ALGD′(X1, . . . , Xn) = max0≤i≤τ−1{D′
i(Xτ(D)−i)}.

Observation order.
Several variants of the prophet inequality problem have been studied, depending
on the order of observations. The standard model is the adversarial (or fixed) order: The instance
of the distributions F1, . . . , Fn is chosen by an adversary, and the algorithm observes the samples
X1 ∼F1, . . . , Xn ∼Fn in this order [Krengel and Sucheston, 1977, 1978]. In the random order
model, the adversary can again choose the distributions, but the algorithm observes the samples in a

2


---Page Break---
uniformly random order. Another setting in which the observation order is no longer important is the
IID model [Hill and Kertz, 1982, Correa et al., 2021b], where all the values are sampled independently
from the same distribution F. The D-prophet inequality is well-defined in each of these different
order models: if the items are observed in the order Xπ(1), . . . , Xπ(n) with π a permutation of [n],
then the reward of the algorithm is ALGD(X1, . . . , Xn) = max0≤i≤τ−1{Di(Xπ(τ−i))}. In this
paper, we study the D-prophet inequality in the three models we presented, providing lower and
upper bounds in each of them.

Competitive ratio.
In the D-prophet inequality, an input instance I is a finite sequence of probability
distributions (F1, . . . , Fn). Thus, for any instance I, we denote by E[ALGD(I)] the expected reward
of ALG given I as input, and we denote by E[OPT(I)] the expected maximum of independent
random variables (Xi)i∈[n], where Xi ∼Fi. With these notations, we define the competitive ratio,
which will be used to measure the quality of the algorithms.
Definition 1.2 (Competitive ratio). Let D be a sequence of decay functions and ALG an algorithm.
We define the competitive ratio of ALG by

CRD(ALG) = inf
I
E[ALGD(I)]

E[OPT(I)] ,

with the infimum taken over the tuples of all sizes of non-negative distributions with finite expectation.

An algorithm is said to be α-competitive if its competitive ratio is at least α, which means that for
any possible instance I, the algorithm guarantees a reward of at least αE[OPT(I)]. The notion
of competitive ratio is used more broadly in competitive analysis as a metric to evaluate online
algorithms [Borodin and El-Yaniv, 2005].

1.2
Contributions

It is trivial that non-zero decay functions D guarantee a better reward compared to the classical
prophet inequality. However, in general, this is not sufficient to conclude that the standard upper
bounds or the competitive ratio of a given algorithm can be improved. Hence, a first key question is:
what condition on D is necessary to surpass the conventional upper bounds of the classical prophet
inequality? Surprisingly, the answer hinges solely on the constant γD, defined as follows,

γD = inf
x>0 inf
j≥1

Dj(x)

x


.
(1)

In the adversarial order model, we demonstrate that the optimal competitive ratio achievable in the
D-prophet inequality is determined by the parameter γD alone. Additionally, in both the random
order and IID models, we demonstrate the essential requirement of γD > 0 for breaking the upper
bounds of the classical prophet inequality. In particular, this implies that no improvement can be
made with decay functions of the form Dj(x) = x−cj with cj > 0, or Dj(x) = λjx with λ ∈[0, 1).
Subsequently, we develop algorithms and provide upper bounds in the D-prophet inequality, uniquely
dependent on the parameter γD. We illustrate them in Figure 1, comparing them with the identity
function γ 7→γ, which is a trivial lower bound.

1.3
Related work

Prophet inequalities.
The first prophet inequality was proven by Krengel and Sucheston [Krengel
and Sucheston, 1977, 1978] in the setting where the items are observed in a fixed order, demonstrating
that the dynamic programming algorithm has a competitive ratio of 1/2, which is the best possible.
It was shown later that the same guarantee can be obtained with simpler algorithms [Samuel-Cahn,
1984, Kleinberg and Weinberg, 2012], accepting the first value above a carefully chosen threshold.
For a more comprehensive and historical overview, we refer the interested reader to surveys on the
problem such as [Lucier, 2017, Correa et al., 2019]. Prophet inequalities have immediate applications
in mechanism design [Hajiaghayi et al., 2007, Deng et al., 2022, Psomas et al., 2022, Makur et al.,
2024], auctions [Syrgkanis, 2017, Dütting et al., 2020], resource management [Sinclair et al., 2023],
and online matching [Cohen et al., 2019, Ezra et al., 2020, Jiang et al., 2021, Papadimitriou et al.,
2021, Brubach et al., 2021]. Many variants and related problems have been studied, including, for
example, the matroid prophet inequality [Kleinberg and Weinberg, 2012, Feldman et al., 2016],

3


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
훾D ∈[0, 1]

1/2

1 −1
푒≈0.632

√
3 −1 ≈0.732

4−log 3
4−2/푒2 ≈0.778

1

[Adversarial] lower and upper bound
[Random and IID] lower bound
[Random] upper bound
[IID] upper bound
훾D ↦→훾D

Figure 1: Lower and upper bounds on the competitive ratio in the D-prophet inequality depending on
γD, in the adversarial order (Thm 4.3), random order (Thm 4.4) and IID (Thm 4.6) models

prophet inequality with advice [Diakonikolas et al., 2021], and variants with fairness considerations
[Correa et al., 2021a, Arsenis and Kleinberg, 2022].

Random order and IID models.
Esfandiari et al. [2017] introduced the prophet secretary problem,
where items are observed in a uniformly random order, and they proved a (1 −1

e)-competitive
algorithm. Correa et al. [2021c] showed later a competitive ratio of 0.669, and Harb [2024] enhanced
it to 0.6724, which currently stands as the best-known solution for the problem. They also proved
an upper bound of
√

3 −1 ≈0.732, which was improved to 0.7254 in [Bubna and Chiplunkar,
2023] then 0.723 in [Giambartolomei et al., 2023]. Addressing the gap between the lower and upper
bound remains an engaging and actively pursued open question. On the other hand, the study of
prophet inequalities with IID random variables dates back to papers such as [Hill and Kertz, 1982,
Kertz, 1986], demonstrating guarantees on the dynamic programming algorithm. The problem was
completely solved in [Correa et al., 2021b], where the authors show that the competitive ratio of the
dynamic programming algorithm is 0.745, thus it constitutes an upper bound on the competitive ratio
of any algorithm, and they give a simpler adaptive threshold algorithm matching it. Another setting
that we do not study in this paper, is the order selection model, where the decision-maker can choose
the order in which the items are observed, knowing their distributions [Chawla et al., 2010, Beyhaghi
et al., 2021, Peng and Tang, 2022].

Beyond the worst-case.
In recent years, there has been increasing interest in exploring ways to
exceed the worst-case upper bounds of online algorithms by providing the decision-maker with
additional capabilities. A notable research avenue is learning-augmented algorithms [Lykouris
and Vassilvtiskii, 2018], which equip the decision-maker with predictions or hints about unknown
variables of the problem. Multiple problems have been studied in this framework, such as scheduling
[Purohit et al., 2018, Lassota et al., 2023, Benomar and Perchet, 2024b], matching [Antoniadis et al.,
2020, Dinitz et al., 2021, Chen et al., 2022], caching [Antoniadis et al., 2023, Chlkedowski et al.,
2021, Christianson et al., 2023], the design of data structures [Kraska et al., 2018, Lin et al., 2022,
Benomar and Coester, 2024], and in particular, online selection problems [Dütting et al., 2021, Sun
et al., 2021, Benomar et al., 2023, Benomar and Perchet, 2024a, Diakonikolas et al., 2021]. More
related to our setting, the ability to revisit items in online selection has been studied in problems
such as the multiple-choice prophet inequality, where the algorithm can select up to k items and its
reward is the maximum selected value [Assaf and Samuel-Cahn, 2000]. This allows for revisiting
up to k items, chosen during the execution, for final acceptance or rejection decisions. Similarly, in
Pandora’s box problem [Weitzman, 1978, Kleinberg et al., 2016] and its variants [Esfandiari et al.,
2019, Gergatsouli and Tzamos, 2022, Atsidakou et al., 2024, Gergatsouli and Tzamos, 2024, Berger
et al., 2024], the decision maker decides the observation order of the items, but a cost ci is paid for
observing each value Xi, with the gain being the maximum observed value minus the total opening
costs. A very recent work investigates a scenario closely related to the lookback prophet inequality

4


---Page Break---
[Ekbatani et al., 2024] where, upon selecting a candidate Xi, the decision-maker has the option to
discard it and choose a new value Xj at any later step j, at a buyback cost of fXi, where f > 0. The
authors present an optimal algorithm for the case when f ≥1, although the problem remains open
for f ∈(0, 1). Other problems were studied in similar settings, such as online matching [Ekbatani
et al., 2022] and online resource allocation [Ekbatani et al., 2023].

2
From D-prophet to the D∞-prophet inequality

Let us consider a sequence D of decay functions. By Definition 1.1, for any x ∈[0, ∞] the sequence
(Dj(x))j≥1 converges, since it is non-increasing and non-negative. Hence, there exists a mapping
D∞such that for any x ≥0, limj→∞Dj(x) = D∞(x). Furthermore, we can easily verify that D∞
is non-decreasing and satisfies D∞(x) ∈[0, x] for all x ≥0.

Thanks to these properties, we obtain that (D∞)j≥1 also satisfies Definition 1.1, and is hence a
valid sequence of decay functions. We thus refer to the corresponding problem as the D∞-prophet
inequality. Since Dj ≥D∞for any j ≥1, it is straightforward that the stopping problem with the
decay functions D∞would be less favorable to the decision-maker. More precisely, for any random
variables X1, . . . , Xn, observation order π, and algorithm ALG with stopping time τ, it holds that

ALGD(X1, . . . , Xn) := max{Xπ(τ), max
i<τ Dτ−i(Xπ(i))} ≥max{Xπ(τ), max
i<τ D∞(Xπ(i))} ,

which corresponds to the output of ALG (with the same decision rule) when all the decay functions
are equal to D∞. Therefore, any guarantees established for algorithms in the D∞-prophet inequality
naturally extend to the D-prophet inequality. However, it remains uncertain whether the D-prophet
inequality can yield improved competitive ratios compared to the D∞-prophet inequality. In the
following, we prove that this is not the case, for all the order models presented in Section 1.

Theorem 2.1. Let D∞be the pointwise limit of the sequence of decay functions D = (Dj)j≥1. Then
for any instance I = (F1, . . . , Fn) of non-negative distributions, it holds in the adversarial and the
random order models that

∀ALG : CRD(ALG) ≤sup
A

E[AD∞(I)]
E[OPT(I)] ,
(2)

where the supremum is taken over all the online algorithms A. In the IID model, the same inequality
holds with an additional O(n−1/3) term, which depends only on the size n of the instance.

The main implication of Theorem 2.1 is the following corollary.

Corollary 2.1.1. In the adversarial order and the random order models, if A∞is an optimal algorithm
for the D∞-prophet inequality, i.e. with maximal competitive ratio, then A∞is also optimal for the
D-prophet inequality. Moreover, it holds that

CRD(A∞) = CRD∞(A∞) .

A direct consequence of this result is that, in the adversarial and the random order models, the
asymptotic decay D∞entirely determines the competitive ratio that is achievable and the upper
bounds for the D-prophet inequality. Therefore, we can restrict our analysis to algorithms designed
for the problem with identical decay function. In the IID model, the same conclusion holds if the
worst-case instances are arbitrarily large, making the additional O(n−1/3) term vanish. This is the
case in particular in the classical IID prophet inequality [Hill and Kertz, 1982].

2.1
Sketch of the proof of Theorem 2.1

While we use different techniques for each order model considered, all the proofs share the same
underlying idea. Given any instance I of non-negative distributions, we build an alternative instance
J such that the reward of any algorithm on I with decay functions D = (Dj)j is at most its reward
on J with decay functions all equal to D∞. To do this, we essentially introduce an arbitrarily large
number of zero values between two successive observations drawn from distributions belonging to
I. Hence, under J, the algorithm cannot recover much more than a fraction D∞(X) for any past
observation X collected from a distribution F ∈I.

5


---Page Break---
In the adversarial case, implementing this idea is straightforward, since nature can build J by directly
inserting m zeros between each pair of consecutive values, and the result is obtained by making m
arbitrarily large. For the random order model, we use the same instance J, but extra steps are needed
to prove that the number of steps between two non-zero values is very large with high probability.

Moving to the IID model, an instance I is defined by a pair (F, n), where F is a non-negative
distribution, and n is the size of the instance. In this scenario, we consider an instance consisting of
m > n IID random variables (Yi)i∈[m], each sampled from F with probability n/m, and equal to
zero with the remaining probability. We again achieve the desired result by letting m be arbitrarily
large compared to n. However, the number of variables sampled from F is not fixed; it follows a
Binomial distribution with parameters (m, n/m). We control this variability by using concentration
inequalities, which causes the additional term O(n−1/3).

3
From D∞-prophet to the γD-prophet inequality

As discussed in Section 2, Theorem 2.1 implies that, for either establishing upper bounds or guarantees
on the competitive ratios of algorithms, it is sufficient to study the D∞-prophet inequality, where all
the decay functions are equal to D∞. The remaining question is then to determine which functions
D∞allow to improve upon the upper bounds of the classical prophet inequality. Before tackling this
question, let us make some observations regarding algorithms in the D∞-prophet inequality.

In the D∞-prophet inequality, it is always possible to have a reward of D∞(maxi∈[n] Xi) by rejecting
all the items and then selecting the maximum by the end. Thus, it is suboptimal to stop at a step i
where Xi ≤D∞(maxj<i Xj). An algorithm respecting this decision rule is called rational.
Lemma 3.1. For any rational algorithm ALG in the D∞-prophet inequality, if we denote by τ its
stopping time, then for any instance I = (F1, . . . , Fn) and Xi ∼Fi for all i ∈[n] we have

ALGD∞(X1, . . . , Xn) = ALG0(X1, . . . , Xn) + D∞
 
max
i∈[n] Xi

1τ=n+1 ,

where ALG0 denotes the reward of the algorithm in the standard prophet inequality. Moreover, the
optimal dynamic programming algorithm in the D∞-prophet inequality is rational.

The best competitive ratio in the D∞-prophet inequality is achieved, possibly among others, by the
optimal dynamic programming algorithm, which is a rational algorithm by the previous Lemma.
Hence, it suffices to prove upper bounds on rational algorithms. We use this observation to prove the
next propositions.

Proposition 3.2. In the D∞-prophet inequality, if infx>0
D∞(x)

x
= 0, then it holds, in any order
model, that
∀ALG : CRD∞(ALG) ≤sup
A
CR0(A) ,
(3)

where the supremum is taken over all the online algorithms A, and CR0 denotes the competitive ratio
in the standard prophet inequality.

Proposition 3.2 implies that if infx>0
D∞(x)

x
= 0, then, in any order model, any upper bound on
the competitive ratios of all algorithms in the classical prophet inequality is also an upper bound on
the competitive ratios of all algorithm in the D∞-prophet inequality. Consequently, for surpassing
the upper bounds of the classical prophet inequality, it is necessary to have, for some γ > 0, that
D∞(x) ≥γx for all x ≥0. Furthermore, the next Proposition allows giving upper bounds in the
D∞-prophet inequality that depend only on infx>0
D∞(x)

x
.
Proposition 3.3. Let γ = infx>0 D∞(x)/x, and 0 < a < b. Consider an instance I of distributions
with support in {0, a, b}, then in any order model and for any algorithm ALG we have that

CRD∞(ALG) ≤sup
A

E[Aγ(I)]
E[OPT(I)] ,

where E[Aγ(I)] is the reward of A if all the decay functions were equal to x 7→γx.

The core idea for proving this proposition is that rescaling an instance, i.e. considering (rXi)i∈[n]
instead of (Xi)i∈[n], has no impact in the classical prophet inequality. However, in the D∞-prophet

6


---Page Break---
inequality, rescaling can be exploited to adjust the ratio D∞(rx)

rx
. By considering instances with
random variables taking values in {0, a, b} almost surely, where a < b, a reasonable algorithm facing
such an instance would never reject the value b. Consequently, the value it recovers from rejected
items is either D∞(0) = 0 or D∞(a). Rescaling this instance by a factor r = s/a and taking the
ratio to the expected maximum, the term D∞(s)

s
appears, with s a free parameter that can be chosen
to satisfy D∞(s)

s
→infx>0
D∞(x)

x
= γD.

As a consequence, if infx>0
D∞(x)

x
= γ, then any upper bound obtained in the γ-prophet inequality
(when the decay functions are all equal to x 7→γx) using instances of random variables (X1, . . . , Xn)
satisfying Xi ∈{0, a, b} a.s. for all i, is also an upper bound in the D∞-prophet inequality.

Implication
Consider any sequence D of decay functions, and define

γD := inf
x>0

D∞(x)

x


= inf
x>0 inf
j≥1

Dj(x)

x


.

For any x > 0 and j ≥1 it holds that Dj(x) ≥γDx, therefore, any guarantees on the competitive
ratio of an algorithm in the γD-prophet inequality are valid in the D-prophet inequality, under any
order model. Furthermore, combining Theorem 2.1 and Proposition 3.3, we obtain that for any
instance I of random variables taking values in a set {0, a, b} it holds that

∀ALG : CRD(ALG) ≤sup
A

E[AγD(I)]
E[OPT(I)] ,

with an additional term of order O(n−1/3) in the IID model. In the particular case where γD = 0,
Proposition 3.3 with Theorem 2.1 give a stronger result, showing that no algorithm can surpass the
upper bounds of the classical prophet inequality. This is true also for the IID model since the instances
used to prove the tight upper bound of ≈0.745 are of arbitrarily large size [Hill and Kertz, 1982].

Therefore, by studying the γ-prophet inequality for γ ∈[0, 1], we can prove upper bounds and lower
bounds on the D-prophet inequality for any sequence D of decay functions.

4
The γ-prophet inequality

We study in this section the γ-prophet inequality, where all the decay functions are equal to x 7→γx,
for some γ ∈[0, 1]. For any algorithm ALG with stopping time τ and random variables X1, . . . , Xn,
if the observation order is π, we use the notation

ALGγ(X1, . . . , Xn) = max{Xπ(τ), γXπ(τ−1), . . . , γXπ(1)} .

and we denote by CRγ(ALG) the competitive ratio of ALG in this setting. In the following, we
provide theoretical guarantees for the γ-prophet inequality.

For each observation order, we first derive upper bounds on the competitive ratio of any algorithm,
depending on γ, using only hard instances satisfying the condition of Proposition 3.3. This would
guarantee that the upper bounds extend to the D-prophet inequality if γD = γ. Then, we design
single-threshold algorithms with well-chosen thresholds depending on γ and the distributions, with
competitive ratios improving with γ. A crucial property of single-threshold algorithms, which we use
to estimate their competitive ratios, is that their reward satisfies

ALGγ(X1, . . . , Xn) = ALG0(X1, . . . , Xn) + γ(max
i
Xi)1(maxi Xi<θ) .
(4)

The additional term appearing due to γ depends only on maxi∈[n] Xi, which is the reward of the
prophet against whom we compete. This property is not satisfied by more general class of algorithms
such as multiple-threshold algorithms, where each observation Xπ(i) is compared with a threshold θi.
Remark 4.1. We only consider instances with continuous distributions in the proofs of lower bounds.
The thresholds θ considered are such that Pr(maxi∈[n] Xi ≥θ) = g(γ, n, π), with g depending on
γ, the order model π and the size of the instance n. Such a threshold is always guaranteed to exist
when the distributions are continuous. However, as in the prophet inequality, the algorithms can
be easily adapted to non-continuous distributions by allowing stochastic tie-breaking. A detailed
strategy for doing this can be found for example in [Correa et al., 2021c].

7


---Page Break---
Before delving into the study of the different models, we provide generic lower and upper bounds,
which depend solely on the bounds of the classical prophet inequality and γ.

Proposition 4.2. In any order model, if α is a lower bound in the classical prophet inequality, and β
an upper bound, then, in the γ-prophet inequality

(i) there exists a trivial algorithm with a competitive ratio of at least max{γ, α},

(ii) the competitive ratio of any algorithm is at most (1 −γ)β + γ.

4.1
Adversarial order

We first consider the adversarial order model, and prove the upper bound of
1
2−γ . Then, we provide a
single-threshold algorithm with a competitive ratio matching this upper bound, hence fully solving
the γ-prophet inequality in this adversarial order model.
Theorem 4.3. In the adversarial order model, the competitive ratio of any algorithm is at most
1
2−γ . Furthermore, there exists a single threshold algorithm with a competitive ratio
1
2−γ : given any
instance (F1, . . . , Fn), this is achieved with the threshold θ satisfying

PrX1∼F1,...,Xn∼Fn(maxi∈[n] Xi ≤θ) =
1
2−γ .

The upper bound in the previous theorem is proved using instances satisfying the condition of
Proposition 3.3. Hence it extends to the D∞- then to the D-prophet inequality, with γ = γD, by
Proposition 3.3 and Theorem 2.1.

4.2
Random order

Consider now that the items are observed in a uniformly random order Xπ(1), . . . , Xπ(n), and
X∗= maxi∈[n] Xi. As for the adversarial model, we first prove an upper bound on the competitive
ratio as a function of γ, and then prove a lower bound for a single-threshold algorithm. However, for
this model, there is a gap between the two bounds, as illustrated in Figure 1.

We first prove an upper bound that depends on γ, matching the upper bound
√

3 −1 of Correa et al.
[2021c] when γ = 0 and equal to 1 when γ = 1. Our single-threshold algorithm has a competitive
ratio of at least (1−1

e) when γ = 0, which is the best competitive ratio of a single threshold algorithm
in the prophet inequality [Esfandiari et al., 2017, Correa et al., 2021c], and equal to 1 for γ = 1.

Theorem 4.4. The competitive ratio of any algorithm ALG in the γ-prophet inequality with random
order satisfies
CRγ(ALG) ≤(1 −γ)3/2(
p

3 −γ −
p

1 −γ) + γ .

Furthermore, denoting by pγ is the unique solution to the equation 1 −(1 −γ)p =
1−p
−log p, the
single-threshold algorithm ALGθ with PrX1∼F1,...,Xn∼Fn(maxi∈[n] Xi ≤θ) = pγ satisfies

CRγ(ALG) ≥1 −(1 −γ)pγ .

Similarly to the adversarial order model, we used instances satisfying the condition of Proposition
3.3 to prove the upper bound, thus it extends to the D-prophet inequality with γ = γD.

While the equation defining pγ cannot be solved analytically, the solution can easily be computed
numerically for any γ ∈[0, 1]. Before moving to the IID case, we propose in the following a more
explicit lower bound derived from Theorem 4.4.
Corollary 4.4.1. In the random order model, the single threshold algorithm with a threshold θ
satisfying Pr(maxi∈[n] Xi ≥θ) =
1/e
1−(1−1/e)γ has a competitive ratio of at least 1 −
(1−γ)/e
1−(1−1/e)γ .

4.3
IID Random Variables

In the classical IID prophet inequality, [Hill and Kertz, 1982] showed that the competitive ratio of
any algorithm is at most ≈0.745. The proof of this upper bound is hard to generalize for the IID
γ-prophet inequality. As an alternative, we prove a weaker upper bound, which equals ≈0.778

8


---Page Break---
for γ = 0 and 1 for γ = 1, and the proof relies on instances of arbitrarily large size satisfying the
condition of Proposition 3.3, hence the upper bound can be extended to the D-prophet inequality.

Subsequently, we present a single-threshold algorithm with the same competitive ratio as the random
order algorithm. However, the proof is different, leveraging the fact that the variables are identically
distributed. More precisely, we introduce a single-threshold algorithm with guarantees that depend
on the size n of the instance, then we show that its competitive ratio is at least that of the algorithm
presented in Theorem 4.4, with equality when n approaches infinity.

Although it might look surprising that the obtained competitive ratio in the IID model is not better
than that of the random-order model, the same behavior occurs in the classical prophet inequality.
Indeed, Li et al. [2022] established that no single-threshold algorithm can achieve a competitive ratio
better than 1 −1/e in the standard prophet inequality with IID random variables, which is also the
best possible with a single-threshold algorithm in the random order. However, considering larger
classes of algorithms, the competitive ratios achieved in the IID model are better than those of the
random order model.

We describe the algorithm and give a first lower bound on its reward depending on the size of the
instance in the following lemma.

Lemma 4.5. Let an,γ be the unique solution of the equation

1
(1−a/n)n −1
   1

a −1

= γ, then for

any IID instance X1, . . . , Xn, the algorithm with threshold θ satisfying Pr(X1 > θ) = an,γ

n
has a
reward of at least
1
an,γ


1 −

1 −an,γ

n

n
E[max
i∈[n] Xi] .

We can prove that the reward presented in the Lemma above is strictly better than that of the random
order model. However, both are asymptotically equal as we show in the following theorem.

Theorem 4.6. The competitive ratio of any algorithm in the IID γ-prophet inequality is at most

U(γ) = 1 −(1 −γ) e2 log(3 −γ) −(2 −γ)

2(2e2 −1) −(3e2 −1)γ .

In particular, U is increasing, U(0) =
4−log 3
2(2−1

e
2) ≈0.778 and U(1) = 1. Furthermore, there exists a

single-threshold algorithm ALGθ satisfying

CRγ(ALGθ) ≥1 −(1 −γ)pγ ,

where pγ is defined in Theorem 4.4.

To prove the upper bound, we used instances satisfying the condition of Proposition 3.3, guaranteeing
that it remains true in the D∞-prophet inequality with γ = γD. On the other hand, Theorem 2.1
ensures that the upper bound extends to the D-prophet inequality, but with an additional O(1/n1/3)
term. The latter does change the result, as we considered instances of arbitrarily large size n →∞.

5
Conclusion

In this paper, we addressed the D-prophet inequality problem, which models a very broad spectrum of
online selection scenarios, accommodating various observation order models and allowing to revisit
rejected candidates at a cost. The problem extends the classic prophet inequality, corresponding to
the special case where all decay functions are zero. The main result of the paper is a reduction from
the general D-prophet inequality to the γ-prophet inequality, where all the decay functions equal to
x 7→γx for some constant γ ∈[0, 1]. Subsequently, we provide algorithms and upper bounds for
the γ-prophet inequality, which remain valid, by the previous reduction, in the D-prophet inequality.
Notably, the proved upper and lower bounds match each other for the adversarial order model, hence
completely solving the problem. Our analysis paves the way for more practical applications of
prophet inequalities, and advances efforts towards closing the gap between theory and practice in
online selection problems.

9


---Page Break---
Limitations and future work

Better upper bounds in the D∞-prophet inequality.
Proposition 3.3 establishes that upper bounds
proved in the γ-prophet inequality using instances of random variables with support in some set
{0, a, b} remain true in the D∞-prophet inequality, hence in the D-prophet inequality by Theorem
2.1. This is enough to establish a tight upper bound in the adversarial order model, but not in the
random order and IID models. An interesting question to explore is if more general upper bounds can
be extended, or not, from the γ- to the D-prophet inequality.

Algorithms for the γ-prophet inequality.
As explained in Section 4, our analysis of the competitive
ratio of single-threshold algorithms relies on the identity (4), which is not satisfied for instance by
multiple-threshold algorithms. In the adversarial order model, we proved that the optimal competitive
ratio 1/(2 −γ) can be achieved with a single-threshold algorithm. However, this is not the case in the
random order or IID models. An interesting research avenue is to study other classes of algorithms in
the γ-prophet inequality.

Acknowledgements

This research was supported in part by the French National Research Agency (ANR) in the framework
of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant DOOM ANR-
23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601). Views and
opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union or the European Research Council Executive Agency. Neither the European Union
nor the granting authority can be held responsible for them.

Dorian Baudry thanks the support of the French National Research Agency: ANR-19-CHIA-02
SCAI, ANR-22-SRSE-0009 Ocean, and ANR-23-CE23-0002 Doom; and of the European Research
Council (GTIR project)

References

Antonios Antoniadis, Themis Gouleakis, Pieter Kleer, and Pavel Kolev. Secretary and online matching
problems with machine learned advice. Advances in Neural Information Processing Systems, 33:
7933–7944, 2020.

Antonios Antoniadis, Joan Boyar, Marek Eliás, Lene Monrad Favrholdt, Ruben Hoeksma, Kim S
Larsen, Adam Polak, and Bertrand Simon. Paging with succinct predictions. In International
Conference on Machine Learning, pages 952–968. PMLR, 2023.

Makis Arsenis and Robert Kleinberg. Individual fairness in prophet inequalities. In Proceedings of
the 23rd ACM Conference on Economics and Computation, pages 245–245, 2022.

David Assaf and Ester Samuel-Cahn. Simple ratio prophet inequalities for a mortal with multiple
choices. Journal of applied probability, 37(4):1084–1091, 2000.

Alexia Atsidakou, Constantine Caramanis, Evangelia Gergatsouli, Orestis Papadigenopoulos, and
Christos Tzamos. Contextual pandora’s box. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 10944–10952, 2024.

Ziyad Benomar and Christian Coester.
Learning-augmented priority queues.
arXiv preprint
arXiv:2406.04793, 2024.

Ziyad Benomar and Vianney Perchet. Advice querying under budget constraint for online algorithms.
Advances in Neural Information Processing Systems, 36, 2024a.

Ziyad Benomar and Vianney Perchet. Non-clairvoyant scheduling with partial predictions. In
Forty-first International Conference on Machine Learning, 2024b.

Ziyad Benomar, Evgenii Chzhen, Nicolas Schreuder, and Vianney Perchet. Addressing bias in online
selection with limited budget of comparisons. arXiv preprint arXiv:2303.09205, 2023.

10


---Page Break---
Ben Berger, Tomer Ezra, Michal Feldman, and Federico Fusco. Pandora’s problem with deadlines.
In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 20337–20343,
2024.

Hedyeh Beyhaghi, Negin Golrezaei, Renato Paes Leme, Martin Pál, and Balasubramanian Sivan.
Improved revenue bounds for posted-price and second-price mechanisms. Operations Research,
69(6):1805–1822, 2021.

Allan Borodin and Ran El-Yaniv. Online computation and competitive analysis. cambridge university
press, 2005.

Brian Brubach, Nathaniel Grammel, Will Ma, and Aravind Srinivasan. Improved guarantees for
offline stochastic matching via new ordered contention resolution schemes. Advances in Neural
Information Processing Systems, 34:27184–27195, 2021.

Archit Bubna and Ashish Chiplunkar. Prophet inequality: Order selection beats random order. In
Proceedings of the 24th ACM Conference on Economics and Computation, pages 302–336, 2023.

Shuchi Chawla, Jason D Hartline, David L Malec, and Balasubramanian Sivan. Multi-parameter
mechanism design and sequential posted pricing. In Proceedings of the forty-second ACM sympo-
sium on Theory of computing, pages 311–320, 2010.

Justin Chen, Sandeep Silwal, Ali Vakilian, and Fred Zhang. Faster fundamental graph algorithms via
learned predictions. In International Conference on Machine Learning, pages 3583–3602. PMLR,
2022.

Jakub Chlkedowski, Adam Polak, Bartosz Szabucki, and Konrad Tomasz .Zolna. Robust learning-
augmented caching: An experimental study. In International Conference on Machine Learning,
pages 1920–1930. PMLR, 2021.

Nicolas Christianson, Junxuan Shen, and Adam Wierman. Optimal robustness-consistency tradeoffs
for learning-augmented metrical task systems. In International Conference on Artificial Intelligence
and Statistics, pages 9377–9399. PMLR, 2023.

Alon Cohen, Avinatan Hassidim, Haim Kaplan, Yishay Mansour, and Shay Moran. Learning to
screen. Advances in Neural Information Processing Systems, 32, 2019.

Jose Correa, Patricio Foncea, Ruben Hoeksma, Tim Oosterwijk, and Tjark Vredeveld. Recent
developments in prophet inequalities. ACM SIGecom Exchanges, 17(1):61–70, 2019.

Jose Correa, Andres Cristi, Paul Duetting, and Ashkan Norouzi-Fard. Fairness and bias in online
selection. In International conference on machine learning, pages 2112–2121. PMLR, 2021a.

José Correa, Patricio Foncea, Ruben Hoeksma, Tim Oosterwijk, and Tjark Vredeveld. Posted price
mechanisms and optimal threshold strategies for random arrivals. Mathematics of operations
research, 46(4):1452–1478, 2021b.

Jose Correa, Raimundo Saona, and Bruno Ziliotto. Prophet secretary through blind strategies.
Mathematical Programming, 190(1-2):483–521, 2021c.

Yuan Deng, Vahab Mirrokni, and Hanrui Zhang. Posted pricing and dynamic prior-independent
mechanisms with value maximizers. Advances in Neural Information Processing Systems, 35:
24158–24169, 2022.

Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, Ali Vakilian, and Nikos Zarifis. Learning
online algorithms with distributional advice. In International Conference on Machine Learning,
pages 2687–2696. PMLR, 2021.

Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, and Sergei Vassilvitskii. Faster
matchings via learned duals. Advances in neural information processing systems, 34:10393–10406,
2021.

Paul Dütting, Thomas Kesselheim, and Brendan Lucier. An o (log log m) prophet inequality for
subadditive combinatorial auctions. ACM SIGecom Exchanges, 18(2):32–37, 2020.

11


---Page Break---
Paul Dütting, Silvio Lattanzi, Renato Paes Leme, and Sergei Vassilvitskii. Secretaries with advice. In
Proceedings of the 22nd ACM Conference on Economics and Computation, pages 409–429, 2021.

Evgenii Borisovich Dynkin. The optimum choice of the instant for stopping a markov process. Soviet
Mathematics, 4:627–629, 1963.

Farbod Ekbatani, Yiding Feng, and Rad Niazadeh. Online matching with cancellation costs. arXiv
preprint arXiv:2210.11570, 2022.

Farbod Ekbatani, Yiding Feng, and Rad Niazadeh. Online resource allocation with buyback: Optimal
algorithms via primal-dual. In Proceedings of the 24th ACM Conference on Economics and
Computation, pages 583–583, 2023.

Farbod Ekbatani, Rad Niazadeh, Pranav Nuti, and Jan Vondrák. Prophet inequalities with cancellation
costs. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1247–
1258, 2024.

Hossein Esfandiari, MohammadTaghi Hajiaghayi, Vahid Liaghat, and Morteza Monemizadeh.
Prophet secretary. SIAM Journal on Discrete Mathematics, 31(3):1685–1701, 2017.

Hossein Esfandiari, MohammadTaghi HajiAghayi, Brendan Lucier, and Michael Mitzenmacher. On-
line pandora’s boxes and bandits. In Proceedings of the AAAI Conference on Artificial Intelligence,
volume 33, pages 1885–1892, 2019.

Tomer Ezra, Michal Feldman, Nick Gravin, and Zhihao Gavin Tang. Online stochastic max-weight
matching: prophet inequality for vertex and edge arrival models. In Proceedings of the 21st ACM
Conference on Economics and Computation, pages 769–787, 2020.

Moran Feldman, Ola Svensson, and Rico Zenklusen. Online contention resolution schemes. In
Proceedings of the twenty-seventh annual ACM-SIAM symposium on Discrete algorithms, pages
1014–1033. SIAM, 2016.

Evangelia Gergatsouli and Christos Tzamos. Online learning for min sum set cover and pandora’s
box. In International Conference on Machine Learning, pages 7382–7403. PMLR, 2022.

Evangelia Gergatsouli and Christos Tzamos. Weitzman’s rule for pandora’s box with correlations.
Advances in Neural Information Processing Systems, 36, 2024.

Giordano Giambartolomei, Frederik Mallmann-Trenn, and Raimundo Saona. Prophet inequalities:
Separating random order from order selection. arXiv preprint arXiv:2304.04024, 2023.

Mohammad Taghi Hajiaghayi, Robert Kleinberg, and Tuomas Sandholm. Automated online mecha-
nism design and prophet inequalities. In AAAI, volume 7, pages 58–65, 2007.

Elfarouk Harb. New prophet inequalities via poissonization and sharding, 2024.

Theodore P Hill and Robert P Kertz. Comparisons of stop rule and supremum expectations of iid
random variables. The Annals of Probability, pages 336–345, 1982.

Zhihao Jiang, Pinyan Lu, Zhihao Gavin Tang, and Yuhao Zhang. Online selection problems against
constrained adversary. In International Conference on Machine Learning, pages 5002–5012.
PMLR, 2021.

Robert P Kertz. Stop rule and supremum expectations of iid random variables: a complete comparison
by conjugate duality. Journal of multivariate analysis, 19(1):88–112, 1986.

Robert Kleinberg and Seth Matthew Weinberg. Matroid prophet inequalities. In Proceedings of the
forty-fourth annual ACM symposium on Theory of computing, pages 123–136, 2012.

Robert Kleinberg, Bo Waggoner, and E Glen Weyl. Descending price optimally coordinates search.
arXiv preprint arXiv:1603.07682, 2016.

Tim Kraska, Alex Beutel, Ed H Chi, Jeffrey Dean, and Neoklis Polyzotis. The case for learned index
structures. In Proceedings of the 2018 international conference on management of data, pages
489–504, 2018.

12


---Page Break---
Ulrich Krengel and Louis Sucheston. Semiamarts and finite values. 1977.

Ulrich Krengel and Louis Sucheston. On semiamarts, amarts, and processes with finite value.
Probability on Banach spaces, 4:197–266, 1978.

Alexandra Anna Lassota, Alexander Lindermayr, Nicole Megow, and Jens Schlöter. Minimalistic
predictions to schedule jobs with online precedence constraints. In International Conference on
Machine Learning, pages 18563–18583. PMLR, 2023.

Bo Li, Xiaowei Wu, and Yutong Wu. Query efficient prophet inequality with unknown iid distributions.
arXiv preprint arXiv:2205.05519, 2022.

Honghao Lin, Tian Luo, and David Woodruff. Learning augmented binary search trees. In Interna-
tional Conference on Machine Learning, pages 13431–13440. PMLR, 2022.

Denis V Lindley. Dynamic programming and decision theory. Journal of the Royal Statistical Society:
Series C (Applied Statistics), 10(1):39–51, 1961.

Brendan Lucier. An economic view of prophet inequalities. ACM SIGecom Exchanges, 16(1):24–47,
2017.

Thodoris Lykouris and Sergei Vassilvtiskii. Competitive caching with machine learned advice. In
International Conference on Machine Learning, pages 3296–3305. PMLR, 2018.

Anuran Makur, Marios Mertzanidis, Alexandros Psomas, and Athina Terzoglou. On the robustness
of mechanism design under total variation distance. Advances in Neural Information Processing
Systems, 36, 2024.

Christos Papadimitriou, Tristan Pollner, Amin Saberi, and David Wajc. Online stochastic max-weight
bipartite matching: Beyond prophet inequalities. In Proceedings of the 22nd ACM Conference on
Economics and Computation, pages 763–764, 2021.

Bo Peng and Zhihao Gavin Tang. Order selection prophet inequality: From threshold optimization to
arrival time design. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science
(FOCS), pages 171–178. IEEE, 2022.

Alexandros Psomas, Ariel Schvartzman Cohenca, and S Weinberg. On infinite separations between
simple and optimal mechanisms. Advances in Neural Information Processing Systems, 35:4818–
4829, 2022.

Manish Purohit, Zoya Svitkina, and Ravi Kumar. Improving online algorithms via ml predictions.
Advances in Neural Information Processing Systems, 31, 2018.

Ester Samuel-Cahn. Comparison of threshold stop rules and maximum for independent nonnegative
random variables. the Annals of Probability, pages 1213–1216, 1984.

Sean R Sinclair, Felipe Vieira Frujeri, Ching-An Cheng, Luke Marshall, Hugo De Oliveira Barbalho,
Jingling Li, Jennifer Neville, Ishai Menache, and Adith Swaminathan. Hindsight learning for mdps
with exogenous inputs. In International Conference on Machine Learning, pages 31877–31914.
PMLR, 2023.

Bo Sun, Russell Lee, Mohammad Hajiesmaili, Adam Wierman, and Danny Tsang. Pareto-optimal
learning-augmented algorithms for online conversion problems. Advances in Neural Information
Processing Systems, 34:10339–10350, 2021.

Vasilis Syrgkanis. A sample complexity measure with applications to learning optimal auctions.
Advances in Neural Information Processing Systems, 30, 2017.

Martin Weitzman. Optimal search for the best alternative, volume 78. Department of Energy, 1978.

13


---Page Break---
A
From D-prophet to the D∞-prophet inequality

In this section, we prove the reduction from the D-prophet to the D∞-prophet inequality problem in
the adversarial and random order models, and the reduction up to an additional O(n−1/3) term in the
IID model. First, we prove Corollary 2.1.1, which is the principal implication of Theorem 2.1.

A.1
Proof of Corollary 2.1.1

Proof. Let us denote A∗,∞the algorithm taking optimal decisions for any instance in the D∞-prophet
inequality (obtained via dynamic programming). Then, by Theorem 2.1 we obtain for the adversarial
and random order models that

sup
ALG
CRD(ALG) ≤inf
I sup
A

E[AD∞(I)]
E[OPT(I)] = inf
I
E[AD∞
∗,∞(I)]
E[OPT(I)] = CRD∞(A∗,∞) = sup
A
CRD∞(A) .

(5)
Since CRD(ALG) ≥CRD∞(ALG) for any algorithm, we deduce that (5) is an equality. If we
consider now any algorithm A∞that is optimal for the D∞-prophet inequality, not necessarily A∗,∞,
then Equation (5) provides

CRD(A∞) ≥CRD∞(A∞) = sup
ALG
CRD(ALG) .

The previous inequality is again an equality, and it implies that ¯A∞is also optimal, in the sense of
the competitive ratio, for the D-prophet inequality, and

CRD(A∞) = CRD∞(A∞) .

A.2
Auxilary Lemma

The efficiency of the proof scheme introduced in Section 2.1 relies on the following key argument: if
(Dj)j≥1 converges pointwise to D∞, then for any algorithm A and any instance I, the output of A
when all the decay functions are equal to Dm converges to its output when all the decay functions are
equal to D∞. If X1, . . . , Xn are the realizations of I observed by A and if σ is the order in which
they are observed, then denoting τ the stopping time of A we can write that

E[ADm(I)] −E[AD∞(I)]
= E[max{Xσ(τ), Dm(max
i<τ Xσ(i))}] −E[max{Xσ(τ), D∞(max
i<τ Xσ(i))}]

≤max
π∈Sn
q∈[n]


E[max{Xπ(q), Dm(max
i<q Xπ(i))}] −E[max{Xπ(q), D∞(max
i<q Xπ(i))}]

,

where Sn is the set of all permutations of [n]. The latter upper bound is independent of σ and A. We
show in the following lemma that it converges to 0 when m →∞.

Lemma A.1. Let Sn be the set of all permutations of [n]. For any fixed instance I = (F1, . . . , Fn),
considering Xi ∼Fi for all i ∈[n], define for all m ≥1

ϵm(I) = max
π∈Sn
q∈[n]


E[max{Xπ(q), Dm(max
i<q Xπ(i))}] −E[max{Xπ(q), D∞(max
i<q Xπ(i))}]

.

If E[maxi∈[n] Xi] < ∞, then lim
m→∞ϵm(I) = 0.

Proof. Let us denote f1, . . . , fn the respective probability density functions of X1, . . . , Xq. For
any q ∈[n] and π ∈Sn, let us define for all m ≥0 the function φπ,q
m
: [0, ∞)q →[0, ∞) by
φπ,q
m (x1, . . . , xq) = max{xπ(q), Dm(maxi<q xπ(i))} −max{xπ(q), D∞(maxi<q xπ(i))}. φπ,q
m is
positive because Dm ≥D∞. The sequence (φπ,q
m )m is non-increasing, converges to 0 pointwise,
and is dominated by (x1, . . . , xq) 7→maxi∈[q] xi, which is integrable with respect to the probability

14


---Page Break---
measure (x1, . . . , xq) 7→Qq
i=1 f(xi). Therefore, using the dominated convergence theorem, we
deduce that limm→∞E[φπ,q
m (X1, . . . , Xq)] = 0. It follows that

lim
m→∞ϵm(I) = lim
m→∞


max
π∈Sn
q∈[n]
E[φπ,q
m (X1, . . . , Xq)]

= 0 .

A.3
Proof of Theorem 2.1

Proof of Theorem 2.1. We provide a separate proof for each of the adversarial order, random order
and IID models.

Adversarial order
Let I = (F1, . . . , Fn) be any instance and Xi ∼Fi for all i ∈[n]. Consider
the instance Im = (Y1, . . . , Ymn), where Ykm ∼Fk for any k ∈[n] and Yi = 0 a.s. for all
i /∈{m, 2m, . . . , mn}. It is clear that no reasonable algorithm would stop at a zero value: if the
current observation is 0 it is preferable to wait for a non-null value, or it would have been preferable
to stop at the previous non-null value. Hence, τ is a multiple of m: τ = ρm for some ρ ∈N⋆. Given
that Dj(0) = 0 for all j and the sequence (Dj)j is non-increasing, we have that

E[ALGD(Im)] = E[max
i≤τ Dτ−i(Yi)]

= E[max
k≤ρ Dτ−km(Ykm)]

= E[max
k≤ρ Dρm−km(Xk)]

= E[max{Xρ, max
k<ρ D(ρ−k)m(Xk)}]

≤E[max{Xρ, max
k<ρ Dm(Xk)}]

≤E[max{Xρ, max
k<ρ D∞(Xk)}] + ϵm(I) ,

where ϵm(I) is defined in Lemma A.1. We can then use that the first right-hand term is the output
of some other algorithm that would choose a stopping time ρ when facing I in the context of the
D∞-prophet inequality. More precisely, consider the algorithm Am which, given any instance
I = (F1, . . . , Fn), simulates the behavior of ALG facing the sequence Im, where at each step
i ∈[mn]

• if i /∈{m, . . . , nm}: ALG observes Yi = 0,

• otherwise, if i = km for some k ∈[n]: Am observes Xk and ALG observes Ykm = Xk

• if ALG stops on Yρm, then Am also stops, and its reward is Xρ.

Am(X1, . . . , Xn) stops at the same value as ALG(Y1, . . . , Ym), their reward in the D∞-prophet
inequality is the same, and since maxi∈[n] Xi = maxi∈[mn] Yi this yields to

CRD(ALG) ≤E[ALGD(Im)]

E[OPT(Im)] ≤E[AD∞
m (I)] + ϵm(I)

E[OPT(I)]
≤sup
A: algo

E[AD∞(I)]
E[OPT(I)] +
ϵm(I)
E[OPT(I)] ,

and taking the limit when m →∞gives the result, by making the second term vanish.

Random order
Let I = (F1, . . . , Fn) be an instance of distributions and Xi ∼Fi for i ∈[n].
Using the notation δ0 for the Dirac distribution in 0, we consider Im = (F1, . . . , Fn, δ0, . . . , δ0)
containing m copies of δ0 so that the observations from this instance always contain at least m null
values. Let Y1, . . . , Ym be a realization of this instance. For simplicity, say that Yi = Xi for i ∈[n],
and Yi = 0 for i > n.

We first show that when m →∞, since the observation order is drawn uniformly at random,
the algorithm observes a large number of zeros between every two random variables drawn from
(F1, . . . , Fn). Let us denote by π the uniformly random order in which the observations are received,

15


---Page Break---
i.e. the algorithms observes Yπ(1), Yπ(2), . . ., and let ℓ≥1 be some positive integer, and t1, . . . , tn
be the increasing indices in which the variables Y1, . . . , Yn are observed, i.e. t1 < . . . < tn and
{t1, . . . tn} = {π−1(1), . . . , π−1(n)}. Therefore, any observation outside {Yπ(t1), . . . , Yπ(t1)} is
zero. Using the notation L = mini∈[n−1] |ti+1 −ti|, we obtain that

Pr(L ≤ℓ) = Pr(∪n−1
i=1 {ti+1 −ti ≤ℓ})

= Pr
 
∪n
k=1 ∪k−1
j=1

|π−1(k) −π−1(j)| ≤ℓ
	

≤n(n −1)

2
Pr(|π−1(1) −π−1(2)| ≤ℓ)

= n(n −1)

2
Pr(
 
∪n+m
k=1 (π−1(1) = k, π−1(2) ∈{k −ℓ, . . . , k + ℓ} \ {k})

≤n(n −1)

2
× (n + m) ×
1
n + m ×
2ℓ
n + m −1

≤n2ℓ

m .

Taking ℓ= √m, we find that Pr(L ≤ℓ) ≤n2/√m. Therefore, for any algorithm ALG, observing
that the reward of ALGD is at most maxi∈[n] Xi a.s., and by independence of maxi∈[n] Xi and L,
we deduce that

E[ALGD(Im)] = E[ALGD(Im)1L>ℓ] + E[ALGD(Im)1L≤ℓ]

≤E[ALGD(Im) | L > ℓ] + E[(max
i∈[n] Xi)1L≤ℓ]

≤E[ALGD(Im) | L > ℓ] + E[max
i∈[n] Xi] n2

√m .
(6)

Let us denote τ the stopping time of ALG and tρ = maxj∈[n]{tj : tj ≤τ} the last time when a
variable (Xj)j∈[n] was observed by ALG. The sequence of functions (Dj)j≥1 is non-increasing,
hence

E[ALGD(Im) | L > ℓ] = E[max
i∈[τ] Dτ−i(Yπ(i)) | L > ℓ]

= E[max
j≤i Dτ−tj(Yπ(tj)) | L > ℓ]
(7)

≤E[max
j≤i Dtρ−tj(Yπ(tj)) | L > ℓ]
(8)

= E[max

Yπ(tρ), max
j<ρ Dtρ−tj(Yπ(tj))
	
| L > ℓ]

≤E[max

Yπ(tρ), max
j<ρ Dℓ(Yπ(tj))
	
] ,
(9)

Equation (7) holds because the only non-zero values up to step τ are (Yπ(tj))j∈[ρ]. Inequality (8) uses
that the sequence (Dj)j≥1 is non-increasing, and (9) uses, in addition to that, the independence of L
and (Yπ(tj))j∈[n]. We now argue that the term E[max

Yπ(tρ), maxj<ρ Dℓ(Yπ(tj))
	
] is the expected
reward of an algorithm in the Dℓ-inequality. Given that π is a uniform random permutation of [n+m]
and by definition of t1, . . . , tn, the application σ : k ∈[n] 7→π(tk) is a random permutation of [n].
Therefore we consider the algorithm Am that receives as input the instance I = (F1, . . . , Fn), then
considers the array u = (1, . . . , 1, 0, . . . , 0) composed of n values equal to 1 and m zero values, and
a uniformly random permutation π of [n + m], then simulates ALGD(Im) as follows: at each step
j ∈[n + m]

• if uπ(j) = 0, then ALG observes the value Yπ(j) = 0,

• if uπ(j) = 1, then Am observes the next value Xσ(k), and ALG observes Yπ(j) = Xσ(k),

• when ALG decides to stop, Am also stops, and its reward is the current value Xσ(k).

With this construction, (Yj)j∈[n+m] is indeed a realization of the instance Im, and Am stops on the
last value sampled from F1, . . . , Fn observed by ALG. Therefore, denoting ρ the stopping time of

16


---Page Break---
Am, and ϵℓ(I) as defined in Lemma A.1, we have

E[ALGD(Im) | L > ℓ] ≤E[max

Yπ(tρ), max
j<ρ Dℓ(Yπ(tj))
	
]

= E[max

Xσ(ρ), max
j<ρ Dℓ(Xσ(tj))
	
]

= E[ADℓ
m (I)]

≤E[AD∞
m (I)] + ϵℓ(I)

≤sup
A:algo
E[AD∞(I)] + ϵℓ(I) .

Taking ℓ= √m and substituting into Equation (6), then observing that E[OPT(I)] = E[OPT(Im)],
gives that

CRD(ALG) ≤E[ALGD(Im)]

E[OPT(Im)] ≤sup
A:algo

E[AD∞(I)]
E[OPT(I)] +
ϵ√m(I)
E[OPT(I)] + n2

√m .

Finally, taking m →∞and using Lemma A.1, we deduce that

CRD(ALG) ≤sup
A:algo

E[AD∞(I)]
E[OPT(I)] ,

which completes the proof for the random order.

IID random variables
For any probability distribution F on [0, ∞) and for any n ≥1 we denote
E[OPT(F, n)] the expected maximum of n independent random variables drawn from F, and for any
algorithm ALG we denote E[ALGD(F, n)] its expected output when given n IID variable sampled
from F as input. The proof of Theorem 2.1 for this last model is much more technical than for
previous models, so we first prove several auxiliary results that we will later use to provide a concise
proof of the last part of the theorem.

Lemma A.2. For any probability distribution F and n ≥1, ∆≥0, we have

E[OPT(F, n + ∆)] ≤

1 + ∆

n


E[OPT(F, n)] .

Proof. We first write

Pr(OPT(F, n + ∆) > x) = 1 −F(x)n+∆

=

1 + F(x)n 1 −F(x)∆

1 −F(x)n


(1 −F(x)n)

=

1 + F(x)n 1 −F(x)∆

1 −F(x)n


Pr(OPT(F, n) > x) ,

and then use that

F(x)∆= e∆log(F (x)) ≥1 + ∆log(F(x)) = 1 −∆

n log(1/F(x)n) ≥1 −∆

n

1 −F(x)n

F(x)n


,

so we directly obtain

F(x)n 1 −F(x)∆

1 −F(x)n ≤∆

n ,

which gives that

Pr(OPT(F, n + ∆) > x) ≤

1 + ∆

n


Pr(OPT(F, n) > x) .

As we consider non-negative random variables, it follows directly that

E[OPT(F, n + ∆)] ≤

1 + ∆

n


E[OPT(F, n)] .

17


---Page Break---
Lemma A.3. Let N ∼B(m, ε) and let n := E[N] = εm, then we have

E[OPT(F, N)1N≥n+n2/3] ≤
6
n1/3 E[OPT(F, n + n2/3)] ,

E[OPT(F, N)] ≤

1 +
3
n1/3


E[OPT(F, n + n2/3)] .

Proof. Let ∆, s > 0 such that ∆≤s. For any k ≥1 let Wk = [s + (k −1)∆, s + k∆). (Wk)k≥1
is a partition of [s, ∞), thus we have

E[OPT(F, N)1N≥s] =

∞
X

k=1
E[OPT(F, N)1N∈Wk]

≤

∞
X

k=1
E[OPT(F, s + k∆)1N∈Wk]

=

∞
X

k=1
E[OPT(F, s + k∆)] Pr(N ∈Wk)

≤

∞
X

k=1


1 + k∆

s


E[OPT(F, s)] Pr(N ∈Wk)

=

 

Pr(N ≥s) + ∆

s

∞
X

k=1
k Pr(N ∈Wk)

!

E[OPT(F, s)] ,

where we used Lemma A.2 in the penultimate inequality. Furthermore, observing that

∞
X

k=1
k Pr(N ∈Wk) =

∞
X

k=1

k−1
X

ℓ=0
Pr(N ∈Wk) =

∞
X

ℓ=0

∞
X

k=ℓ+1
Pr(N ∈Wk) =

∞
X

ℓ=0
Pr(N ≥s + ℓ∆) ,

we obtain, given ∆≤s, that

E[OPT(F, N)1N≥s] ≤

 

Pr(N ≥s) + ∆

s

∞
X

k=0
Pr(N ≥s + k∆)

!

E[OPT(F, s)]
(10)

≤

 

2

∞
X

k=0
Pr(N ≥s + k∆)

!

.
(11)

N is a Binomial random variable with expectation n. Therefore, Chernoff’s inequality gives for any
δ ≥0 that

Pr(N ≥(1 + δ)n) ≤exp

−δ2n

2 + δ


≤exp

−min(δ, δ2)n

3


,

where the second inequality can be derived by treating separately δ < 1 and δ ≥1. In particular, for
any k ≥1, taking δ = k∆

n such that ∆≤n yields

Pr(N ≥n+k∆) ≤exp

−min(k∆, k2∆2/n)

3


≤exp

−k min(∆, ∆2/n)

3


= exp

−k∆2

3n


.

Substituting this Inequality into (11) with s = n + ∆, we obtain

E[OPT(F, N)1N≥n+∆] ≤

 

2

∞
X

k=1
Pr(N ≥n + k∆)

!

E[OPT(F, n + ∆)]

≤

 

2

∞
X

k=1
exp

−k∆2

3n

!

E[OPT(F, n + ∆)]

=
2
exp
  ∆2

3n

−1
E[OPT(F, n + ∆)]

≤6n

∆2 E[OPT(F, n + ∆)] ,

18


---Page Break---
and taking ∆= n2/3 proves the first inequality of the lemma.

Let us move now to the second inequality. We have
E[OPT(F, N)1N<s] ≤E[OPT(F, s)1N<s] = E[OPT(F, s)] Pr(N < s) ,
and thus, using Inequality (10), again with s = n + ∆and ∆= n2/3, it follows that
E[OPT(F, N)] = E[OPT(F, N)1N<s] + E[OPT(F, N)1N≥s]

≤

 

1 + ∆

s

∞
X

k=0
Pr(N ≥s + k∆)

!

E[OPT(F, s)]

≤

 

1 +

∞
X

k=1
Pr(N ≥n + k∆)

!

E[OPT(F, s)]

≤

1 +
3
n1/3


E[OPT(F, n + ∆)] .

Lemma A.4. Let δ1, . . . , δm
iid∼B(ε), and N = Pm
i=1 δi. Denoting by n = E[N] = εm, if n ≥4
then

E[N 2OPT(F, N)] ≤

1 + 8

n3


n2E[OPT(F, n + n2/3)] .

Proof. For all k ∈[m], denote by Nk = Pm
i=k δi. We have that

N 2OPT(F, N) =

 m
X

i=1
δi

!2

OPT(F, N)

=




m
X

i=1
δ2
i + 2
X

i<j
δiδj



OPT(F, N) ,

and observing that δ2
i = δi for all i, we obtain in expectation
E[N 2OPT(F, N)] = mE[δ1OPT(F, δ1 + N2)] + m(m −1)E[δ1δ2OPT(δ1 + δ2 + N3)]
≤mE[δ1OPT(F, 1 + N2)] + m(m −1)E[δ1δ2OPT(2 + N3)]

= mεE[OPT(F, 1 + N2)] + m(m −1)ε2E[OPT(2 + N3)]

= mεE[OPT(F, 1 + N2)] + m2ε2E[OPT(2 + N3)] .
(12)
For j ∈{1, 2}, the proof of Lemma A.3 can be easily adjusted to prove an upper bound
on E[OPT(F, j + Nj+1), by first bounding E[OPT(F, j + Nj+1)1Nj+1≥s] then E[OPT(F, j +
Nj+1)1Nj+1<s]. The concentration arguments remain the same, replacing m by m −j. The
expectation of Nj+1 is ε(m −j) = n −εj, hence we obtain

E[OPT(F, j + Nj+1)] ≤

1 +
3
(n −εj)1/3


E[OPT(F, j + (n −εj) + (n −εj)2/3)]

=

1 +
3
(n −εj)1/3


E[OPT(F, j + n + n2/3)]

≤

1 +
3
(n −2)1/3


E[OPT(F, 2 + n + n2/3)]

≤

1 +
4
n1/3


E[OPT(F, 2 + n + n2/3)] ,

where we used respectively in the last inequalities that j ≤2 and n ≥4. Furthermore, Lemma A.2
gives that

E[OPT(F, j + Nj+1)] ≤

1 +
4
n1/3

 
1 +
2
n + n2/3


E[OPT(F, n + n2/3)]

≤

1 +
6
n1/3


E[OPT(F, n + n2/3)] ,

19


---Page Break---
where the last inequality is true for n ≥4. Finally, substituting into 12 yields

E[N 2OPT(F, N)] ≤(m2ε2 + mε)

1 +
6
n1/3


E[OPT(F, n + n2/3)]

= (n2 + n)

1 +
6
n1/3


E[OPT(F, n + n2/3)]

=

1 + 1

n

 
1 +
6
n1/3


n2E[OPT(F, n + n2/3)]

≤

1 +
8
n1/3


n2E[OPT(F, n + n2/3)] .

Lemma A.5. Let δ1, . . . , δm
iid∼B(ε), N = Pm
i=1 δi, n = E[N] = εm and L = mini̸=j{|i −j| :
δi = 1, δj = 1}, then for any ℓ≥0 we have

E[OPT(F, N)1L≤ℓ] ≤7mℓε2E[OPT(F, n + n2/3)] .

Proof. The random variables N and L are not independent, thus we need to adequately compute the
distribution of L conditional to N. For any ℓ≥0 and k ≥2 we have

Pr(L ≤ℓ, N = s) = Pr

L ≤ℓ,

m
X

i=1
δk = s


= Pr

∪m
i=1 ∪i−1
j=max(1,i−ℓ)
 
δi = δj = 1,

m
X

i=1
δk = s


≤mℓPr
 
δ1 = δ2 = 1,

m
X

i=3
δk = s −2


= mℓ
m −2
s −2


εs(1 −ε)m−s+2,

therefore

Pr(L ≤ℓ| N = s) = Pr(L ≤ℓ, N = s)

Pr(N = s)

≤
mℓ
 m−2
s−2

εs(1 −ε)m−s+2
 m
s

εs(1 −ε)m−s

≤mℓ

 m−2
s−2


 m
s


= mℓs(s −1)

m(m −1)

≤ℓs2

m .

Using this inequality and Lemma A.4, we deduce that
E[OPT(F, N)1L≤ℓ] = E[OPT(F, N) Pr(L ≤ℓ| N, OPT(F, N))]
= E[OPT(F, N) Pr(L ≤ℓ| N)]

≤ℓ

mE[N 2OPT(F, N)]

≤

1 +
8
n1/3

 ℓn2

m E[OPT(F, n + n2/3)]

=

1 +
8
n1/3


mℓε2E[OPT(F, n + n2/3)]

= 7mℓε2E[OPT(F, n + n2/3)] .

20


---Page Break---
where we used that 1 +
8
n1/3 ≤7 for n ≥4.

Using the previous lemmas, we can now prove the theorem. Let m > n ≥1, ∆= n2/3, ε = n/m
and let Q be the probability distribution of a random variable that is equal to 0 with probability 1 −ε,
and drawn from F with probability ε.

Let us consider m i.i.d. variables Y1, . . . , Ym ∼Q, and for each i ∈[m] we denote by δi the
indicator that Yi is drawn from F. Define N = Pm
i=1 δi ∼B(m, ε) the number of random variables
Yi drawn from the distribution F. In the following, we upper bound the competitive ratio of any
algorithm by analyzing its ratio on this particular instance. For this, we first provide a lower bound
on E[OPT(Q, m)] using Lemma A.2, and obtain

E[OPT(F, n −∆)] ≥
1
1 + 2∆

n
E[OPT(F, n + ∆)] ≥

1 −2∆

n


E[OPT(F, n + ∆)] ,

thus we have

E[OPT(Q, m)] = E[OPT(F, N)]
≥E[OPT(F, N)1N≥n−∆]
≥E[OPT(F, n −∆)1N≥n−∆]
= E[OPT(F, n −∆)] Pr(N ≥n −∆)

≥

1 −2∆

n


Pr(N ≥n −∆)E[OPT(F, n + ∆)]

≥

1 −2∆

n −Pr(N < n −∆)

E[OPT(F, n + ∆)]

≥

1 −2n−1/3 −exp(−n1/3/2)

E[OPT(F, n + ∆)]

≥

1 −4n−1/3
E[OPT(F, n + ∆)] ,
(13)

where, for the last three inequalities, we used respectively Bernoulli’s inequality, Chernoff bound,
then e−y ≤1/y.

Then, we upper bound the reward of any algorithm given the instance (Q, m) as input. Let L =
mini̸=j{|i −j| : δi = 1, δj = 1} the smallest gap between two successive variables Yi drawn from
F, and let t1 < . . . < tN the indices for which δi = 1. We have for any algorithm ALG and positive
integer ℓthat

E[ALGD(Q, m)] = E[ALGD(Q, m)1N≥n+∆or L≤ℓ] + E[ALGD(Q, m)1N<n+∆,L>ℓ] .
(14)

Using Lemma A.3 and Lemma A.5, the first term can be bounded as follows

E[ALGD(Q, m)1N≥n+∆or L≤ℓ] ≤E[OPT(F, N)1N≥n+∆or L≤ℓ]
≤E[OPT(F, N)1N≥n+∆] + E[OPT(F, N)1L≤ℓ]

≤
 6

n1/3 + 7mℓε2

E[OPT(F, n + n2/3)] .

Recalling that ε = m/n and taking ℓ= √m, we obtain

E[ALGD(Q, m)1N≥n+∆or L≤ℓ] ≤
 6

n1/3 + 7n2

√m


E[OPT(F, n + n2/3)] .
(15)

21


---Page Break---
Regarding the second term in Equation (14), let τ be the stopping time of ALG and tρ = max{j ≤
τ : δj = 1} the last value sampled from F and observed by ALG before it stops. We have

E[ALGD(Q, m)1N<n+∆,L>ℓ] ≤E[ALGD(Q, m) | N < n + ∆, L > ℓ]
= E[max
i∈[τ] Dτ−i(Yi) | N < n + ∆, L > ℓ]

= E[max
j∈[ρ] Dτ−tj(Yi) | N < n + ∆, L > ℓ]

≤E[max
j∈[ρ] Dtρ−tj(Yi) | N < n + ∆, L > ℓ]

= E[max

Ytρ, max
j<ρ Dtρ−tj(Ytj)
	
| N < n + ∆, L > ℓ]

≤E[max

Ytρ, max
j<ρ Dℓ(Ytj)
	
| N < n + ∆] .

We then prove that the last term is the reward of an algorithm Am in the Dℓ-prophet inequality. Let us
Am be the algorithm that takes as input an instance X1, . . . , Xn+∆−1 of n + ∆IID random variables,
then simulates ALGD(Q, m) | N < n + ∆as follows: let δ1, . . . , δm
iid∼B(n/m) set NA = 0 and
for each i ∈[m]

• if δi = 0: ALG observes the value Yi = 0,

• if δi = 1: increment N, then Am observes the next value Xk, and ALG observes Yi = Xk,

• if NA = n + ∆−1 or ALG stops, then Am also stops.

When ALG decides to stop, the current value observed by Am is Xρ: the last value Ytρ observed by
ALG such that δtρ = 0. Observe that stopping when NA = n + ∆+ 1, is equivalent to letting ALG
observe zero values until the end, and stopping when ALG stops. Hence, the variables Y1, . . . , Ym
have the same distribution as m IID samples from Q conditional to N < n + ∆. Denoting ρ the
stopping time of Am and ϵℓ(F, n + ∆) as defined in Lemma A.1, we deduce that

E[ALGD(Q, m) | N < n + ∆, L > ℓ] ≤E[max

Ytρ, max
j<ρ Dℓ(Ytj)
	
| N < n + ∆]

= E[max

Xρ, max
j<ρ Dℓ(Xj)
	
]

= E[ADℓ
m (F, n + ∆)]

≤E[AD∞
m (F, n + ∆)] + ϵℓ(F, n + ∆) .
(16)

Substituting (15) and (16) in (14), with ℓ= √m, yields

E[ALGD(Q, m)] ≤
 6

n1/3 + 7n2

√m


E[OPT(F, n + ∆)] + E[AD∞
m (F, n + ∆)] + ϵ√m(F, n + ∆) ,

and using Inequality 13, it follows that

CRD(ALG) ≤E[ALGD(Q, m)]

E[OPT(Q, m)]

≤

6
n1/3 + 7n2

√m
1 −
4
n1/3
+ E[AD∞
m (F, n + ∆)] + ϵ√m(F, n + ∆)

(1 −
4
n1/3 )E[OPT(F, n + ∆)]

≤

6
n1/3 + 7n2

√m
1 −
4
n1/3
+
1
1 −
4
n1/3

 
ϵ√m(F, n + ∆)
E[OPT(F, n + ∆)] + sup
A:algo

E[AD∞(F, n + ∆)]
E[OPT(F, n + ∆)]

!

,

22


---Page Break---
taking the limit m →∞and using Lemma A.1 gives

CRD(ALG) ≤

6
n1/3
1 −
4
n1/3
+
1
1 −
4
n1/3

 

sup
A:algo

E[AD∞(F, n + ∆)]
E[OPT(F, n + ∆)]

!

=
6
n1/3 −4 +

1 +
4
n1/3 −4

  

sup
A:algo

E[AD∞(F, n + ∆)]
E[OPT(F, n + ∆)]

!

≤
10
n1/3 −4 + sup
A:algo

E[AD∞(F, n + n2/3)]
E[OPT(F, n + n2/3)] .

where the last inequality holds because E[AD∞(F, n + ∆)] ≤E[OPT(F, n + ∆)] for any algorithm
A. From here, the statement of the theorem can be deduced by observing that, for k = n + n2/3, we
have n ≥(n + n2/3)/2 = k/2, thus n1/3 ≥k1/3/2, and we obtain

CRD(ALG) ≤
20
k1/3 −8 + sup
A:algo

E[AD∞(F, k)]
E[OPT(F, k)]

= sup
A:algo

E[AD∞(F, k)]
E[OPT(F, k)] + O
 1

k1/3


.

B
From D∞-prophet to the γD-prophet inequality

B.1
Proof of Lemma 3.1

Proof. Let ALG be any rational algorithm in the D∞-prophet inequality. If ALG stops at some step
τ ∈[n], then by definition we have that Xτ > D∞(maxj<τ Xj), and thus ALGD∞(X1, . . . , Xn) =
ALG0(X1, . . . , Xn). Otherwise, if it stops at τ = n + 1, then its reward is maxi∈[n] D∞(Xi) =
D∞(maxi∈[n] Xi), because D∞-is non increasing.

On the other hand, let A∗be the optimal dynamic programming algorithm for the D∞-prophet inequal-
ity. At any step i, if Xi < D∞(maxj<i Xj), then stopping at i gives a reward of D∞(maxj<i Xj),
while by rejecting Xi, the final reward is guaranteed to be at least D∞(maxj<i Xj). Thus rejecting
Xi can only increase the reward, it is therefore the optimal decision.

B.2
Proof of Proposition 3.2

Proof. Let us place ourselves in any order model, or in the IID model. Assume that infx>0
D∞(x)

x
=
0, then there exist a sequence (sk)k≥1 such that limk→∞
D∞(sk)

sk
= 0.

Let I = (F1, . . . , Fn) an instance of non-negative random variables with finite expectation, and
Xi ∼Fi for all i ∈[n]. Let ALG be a rational algorithm for the D∞-prophet inequality and let us
denote τ its stopping time. Denoting X∗:= maxi∈[n] Xi and using Lemma 3.1, we have for any
constant C > 0 that that

E[ALGD∞(I)] = E[ALG0(I)1τ≤n] + E[D∞(X∗)1τ=n+1]

≤E[ALG0(I)] + E[D∞(C)] + E[D∞(X∗)1X∗>C]

≤sup
A
E[A0(I)] + E[D∞(C)] + E[X∗1X∗>C] .
(17)

Let k ≥1 a positive integer, M a positive constant, and consider the instance Ik of random
variables (Xk
1 , . . . , Xk
n) with Xk
i ∼sk

M Xi for all i ∈[n]. We have that OPT(Ik) = sk

M OPT(I)
and supA E[A0(Ik)] = sk

M supA E[A0(I)]. Therefore, using Inequality (17) with Ik instead of I and

23


---Page Break---
C = sk

M , then dividing by OPT(Ik) gives that

CRD∞(ALG) ≤E[ALGD∞(Ik)]

E[OPT(Ik)]
≤sup
A

E[A0(I)]
E[OPT(I)] +
D∞(sk)
sk
M E[OPT(I)] + E[ sk

M X∗1X∗>M]
sk
M E[OPT(I)]

= sup
A

E[A0(I)]
E[OPT(I)] +

M
E[OPT(I)]

 D∞(sk)

sk
+ E[X∗1X∗>M]

E[OPT(I)]
,

and taking the limit k →∞, we obtain

CRD∞(ALG) ≤sup
A

E[A0(I)]
E[OPT(I)] + E[X∗1X∗>M]

E[OPT(I)]
,

and since X∗has finite expectation, taking the limit M →∞gives

CRD∞(ALG) ≤sup
A

E[A0(I)]
E[OPT(I)] .

Finally, taking the infimum over all instances, we obtain that

CRD∞(ALG) ≤sup
A
CR0(A) .

As in the proof of Corollary 2.1.1, permuting infI and supA is possible because there is an algorithm
(dynamic programming) achieving the supremum for any instance. By Lemma 3.1, the inequality
above holds for in particular for the optimal dynamic programming algorithm, which has a maximal
competitive ratio. Therefore, the inequality remains true for any other algorithm A, not necessarily
rational.

B.3
Proof of Proposition 3.3

Proof. Let us place ourselves in any order model or in the IID model. Since γ = infx>0
D∞(x)

x
,
there exists a sequence (sk)k≥1 of positive numbers such that limk→∞
D∞(sk)

sk
= γ.

For the random variables X1, . . . , Xn taking values in {0, a, b} a.s., in any order model, it is clear
that the optimal decision when observing a zero value is to reject it, and when observing the value
b is to accept it. Let ALG be an algorithm satisfying this property and let τ be its stopping time.
If τ = n + 1 then necessarily maxi∈[n] Xi ̸= b, and the reward of ALG in that case is D∞(a) if
maxi∈[n] Xi and 0 otherwise. In particular, ALG is rational in the D∞-prophet inequality and we
have by Lemma 3.1 that

E[ALGD∞(I)] = E[ALG0(I)] + E[D∞(max
i∈[n] Xi)1τ=n+1]

= E[ALG0(I)] + D∞(a) Pr(τ = n + 1, max
i∈[n] Xi = a) .
(18)

Consider now the instance Ik of random variables (Xk
1 , . . . , Xk
n) where Xk
i = sk

a Xi for all i ∈[n].
Ik satisfies the same assumptions as I with ak = sk and bk = skb

a , and we have E[OPT(Ik)] =
sk

a E[OPT(I)], E[ALG0(Ik)] ≤sk

a supA E[A0(I)] and (maxi∈[n] Xk
i = ak) ⇐⇒(maxi∈[n] Xi =
a). It follows that

E[ALGD∞(Ik)]

E[OPT(Ik)]
≤supA E[A0(I)]

E[OPT(I)]
+
D∞(sk)
sk

a E[OPT(I)] Pr(τ = n + 1, max
i∈[n] Xi = a)

= supA E[A0(I)]

E[OPT(I)]
+
D∞(sk)

sk


a
E[OPT(I)] Pr(τ = n + 1, max
i∈[n] Xi = a) .

Taking the limit k →∞gives

CRD∞(ALG) ≤supA E[A0(I)] + γa Pr(τ = n + 1, maxi∈[n] Xi = a)

E[OPT(I)]

= supA(E[A0(I)] + E[γ(maxi∈[n] Xi)1τ=n+1])

E[OPT(I)]
.

24


---Page Break---
ALG is also rational in the γ-prophet inequality. Therefore, using Lemma 3.1, we deduce that

CRD∞(ALG) ≤E[ALGγ(I)]

E[OPT(I)] ≤sup
A

E[Aγ(I)]
E[OPT(I)].

This upper bound is true for the optimal dynamic programming algorithm, since it rejects all zeros
and accepts b, therefore the upper bound also holds for any other algorithm.

C
The γD-prophet inequality

C.1
Proof of Proposition 4.2

Proof. For the lower bound, it suffices to consider the following trivial algorithm: if α > γ then run
Aα, and if γ > α then observe all the items then select the one with maximum value.

For the upper bound, let I = (F1, . . . , Fn) be an instance of the problem and Xi ∼Fi for all i,
and let βI := supA
E[A0(I)]
E[OPT(I)]. Let A be any algorithm, τ its stopping time, and Yτ = maxi<τ Xπ(i),
where π is the observation order. With the previous notations, we can write that E[Aγ(I)] =
E[max(Xπ(τ), γYτ)]. For any x, y ≥0, the application γ 7→max(x, γy) is convex on [0, 1], hence
it can be upper bounded by (1 −γ)x + γ max(x, y). Therefore

E[Aγ(I)] ≤(1 −γ)E[Xπ(τ)] + γE[max(Xπ(τ), Yτ)]

≤(1 −γ)E[A0(I)] + γE[OPT(I)]

≤
 
(1 −γ)βI + γ

E[OPT(I)] .

Therefore, CRγ(ALG) ≤((1 −γ)βI + γ). Taking the infimum over all the instances I gives the
result. Indeed, if we denote A∗the optimal dynamic programming algorithm for the standard prophet
inequality, then

inf
I βI = inf
I
E[A0
∗(I)]
E[OPT(I)] = CR0(A∗) ≤β .

C.2
Proofs for the adversarial order model

Proof. We first prove the upper bound, and then analyze the single threshold algorithm proposed in
the theorem.

Upper bound
Let ε ∈(0, 1 −γ), and let a =
1
1−(1−ε)γ (such that 1 + (1 −ε)γa = a). Let X1, X2
the two random variables defined by X1 = a almost surely and

X2 =

1
ε
w.p. ε
0
w.p. 1 −ε
.

Stopping at the first step gives a reward of a, while stopping at the second step gives

E[max(γa, X2)] = ε × 1

ε + (1 −ε) × γa = 1 + (1 −ε)γa = a ,

hence the expected output of any algorithm is exactly equal to a. On the other hand

E[max(X1, X2)] = 1 + (1 −ε)a ,

and we deduce that, for any algorithm ALG for the γ-prophet inequality, we have

CR(ALG) ≤E[ALG(X1, X2)]

E[max(X1, X2)] =
a
1 + (1 −ε)a ,

and this is true for any ε ∈(0, 1 −γ), thus taking ε →0 gives

CR(ALG) ≤

1
1−γ
1 +
1
1−γ
=
1
2 −γ .

25


---Page Break---
Lower bound
Consider an algorithm with an acceptance threshold θ, i.e. that accepts the first value
larger than θ. Let I = (F1, . . . , Fn) be any instance, such that Xi ∼Fi for all i, and let us define
X∗= maxi∈[n] Xi and p = Pr(X∗≤θ). In the classical prophet inequality, if no value is larger
than θ then the reward of the algorithm is zero, and we have the classical lower bound

E[ALG0(I)] ≥(1 −p)θ + pE[(X∗−θ)+],
For the γ-prophet, if no value is larger than θ (i.e X∗≤θ), then the algorithm gains γX∗instead of
0. Therefore, it holds that
E[ALGγ(I)] = E[ALG0(I)] + E[X∗1X∗≤θ]
≥(1 −p)θ + pE[(X∗−θ)+] + γE[X∗1X∗≤θ]
= (1 −p)θ + pE[(X∗−θ)1X∗>θ] + γE[X∗1X∗≤θ]
= (1 −p)θ + p(E[X∗1X∗>θ] −(1 −p)θ) + γE[X∗1X∗≤θ]

= (1 −p)2θ + pE[X∗1X∗>θ] + γE[X∗1X∗≤θ] ,
and observing that

θ = E[θ1X∗≤θ]

p
≥E[X∗1X∗≤θ]

p
,

we deduce the lower bound

E[ALGγ(I)] ≥pE[X∗1X∗>θ] +

γ + (1 −p)2

p


E[X∗1X∗≤θ]

≥min

p, γ + (1 −p)2

p


E[X∗] .

The right term is maximized for p satisfying p = γ + (1−p)2

p
, that leads to

p = γ + (1 −p)2

p
⇐⇒p2 = γp + 1 −2p + p2

⇐⇒p =
1
2 −γ .

Hence, by choosing a threshold θ satisfying Pr(X∗≤θ) =
1
2−γ we obtain a competitive ratio of at
least
1
2−γ .

C.3
Proofs for the random order model

We prove here the upper and lower bounds stated in Theorem 4.4.

C.3.1
Proof of Theorem 4.4

Proof. We first prove the upper bound, and then derive the analysis for single threshold algorithms.

Upper bound
Let a > 0, and let X1, . . . , Xn+1 be independent random variables such that
Xn+1 = a a.s. and for 1 ≤i ≤n

Xi ∼

n
w.p.
1
n2
0
w.p. 1 −
1
n2
.

Any reasonable algorithm skips zero values and stops when observing the value n. The only strategic
decision to make is thus to stop or not when observing Xn+1 = a. By analyzing the dynamic
programming solution ALG⋆we obtain that the optimal decision rule is to skip a if it is observed
before a certain step j, and select it otherwise. The step j corresponds to the time when the expectation
of the future reward is less than a. Let π be the random order in which the variables are observed.
Then, if π−1(n + 1) < j, i.e. if the value a is observed before time j, Xn=1 is not selected. The
output of this algorithm is hence n if at least one random variable equals n, and γa otherwise. This
leads to

E[ALGγ
⋆(X) | π−1(n + 1) < j] = n

1 −

1 −1

n2

n
+ γa

1 −1

n2

n

≤1 + γa ,

26


---Page Break---
where we used the inequality
 
1 −
1
n2
n ≥1 −1

n. On the other hand, if π−1(n) ≥j, then a is
selected if the value n has not been observed before it, hence for any i ≥j we have

E[ALGγ
⋆(X) | π−1(n + 1) = i] = n

 

1 −

1 −1

n2

i−1!

+ a

1 −1

n2

i−1

≤i −1

n
+ a ,

we deduce that

E[ALGγ
⋆(X)] ≤(1 + γa) Pr(π−1(n) ≤j −1) +

n+1
X

i=j

i −1

n
+ a

Pr(π−1(n) = i)

= j −1

n + 1(1 + γa) +
1
n + 1

n+1
X

i=j

i −1

n
+ a


= (1 −(1 −γ)a) j

n −1

2

 j

n

2
+ 1

2 + a + o(1)

≤1 + 2γa + (1 −γ)2a2 + o(1) ,
where the last inequality is obtained by maximizing over j/n. Finally, we directly obtain that

E[max
i
Xi] = n

1 −

1 −1

n2

n
+ a

1 −1

n2

n

= 1 + a + o(1) ,
so for any algorithm ALG we obtain that

CRγ(ALG) ≤CRγ(ALG⋆) ≤1 + 2γa + (1 −γ)2a2

1 + a
.

The function above is minimized for a =
q

3−γ
1−γ −1, which translates to

CRγ(ALG) ≤(1 −γ)3/2(
p

3 −γ −
p

1 −γ) + γ .

Lower bound
We still denote by I = (F1, . . . , Fn) the input instance and Xi ∼Fi for all i ∈[n].
Let ALG be the algorithm with single threshold θ, then it is direct that

ALGγ(I) = ALG0(I) + γX∗1X∗<θ .
(19)

We start by giving a lower bound on E[ALG0]. We use from Correa et al. [2021c] (Theorem 2.1) that
for any x < θ it holds that

Pr(ALG0(I) ≥x) = Pr(ALG0(I) ≥θ) = Pr(X∗≥θ) = 1 −p ,
and for x ≥θ it holds that

Pr(ALG0(I) ≥x) ≥1 −p

−log p Pr(X∗≥x) ,

from which we deduce that

E[ALG0(I)] =
Z ∞

0
Pr(ALG0(I) ≥x)dx

≥(1 −p)θ + 1 −p

−log p

Z ∞

θ
Pr(X∗≥x)dx .
(20)

On the other hand, we obtain that

E[X∗1X∗<θ] =
Z ∞

0
Pr(X∗1X∗<θ ≥x)dx =
Z ∞

0
Pr(x ≤X∗< θ)dx

=
Z θ

0
(Pr(X∗> x) −Pr(X∗≥θ))dx

=
Z θ

0
Pr(X∗> x)dx −(1 −p)θ .
(21)

27


---Page Break---
Using (19), (20) and (21) we deduce that

E[ALGγ(I)] ≥(1 −p)θ + 1 −p

−log p

Z ∞

θ
Pr(X∗≥x)dx + γ
Z θ

0
Pr(X∗≥x)dx −γ(1 −p)θ

= (1 −γ)(1 −p)θ + γ
Z θ

0
Pr(X∗≥x)dx + 1 −p

−log p

Z ∞

θ
Pr(X∗≥x)dx

≥((1 −γ)(1 −p) + γ)
Z θ

0
Pr(X∗≥x)dx + 1 −p

−log p

Z ∞

θ
Pr(X∗≥x)dx

≥min

(1 −γ)(1 −p) + γ , 1 −p

−log p

  Z θ

0
Pr(X∗≥x)dx +
Z ∞

θ
Pr(X∗≥x)dx

!

= min

1 −(1 −γ)p , 1 −p

−log p


E[X∗] .

Finally, choosing p = pγ gives the result.

C.3.2
Proof of Corollary 4.4.1

Proof. For p =
1/e
1−(1−1/e)γ , we have immediately that

1 −(1 −γ)p = 1 −
(1 −γ)/e
1 −(1 −1/e)γ ,

and p ∈[1/e, 1] for any γ ∈[0, 1]. Since the function x 7→(1 −x)/ log(1/x) is concave, we can
lower bound it on [1/e, 1] by x 7→1 −1/e + x−1/e

e−1 , which is the line intersecting it in 1/e and 1.
Therefore we have

1 −p
−log p ≥1 −1/e + p −1/e

e −1
= 1 −
(1 −γ)/e
1 −(1 −1/e)γ .

Finally, using the previous theorem, this choice of p guarantees a competitive ratio of at least

min

1 −(1 −γ)p , 1 −p

−log p


= 1 −
(1 −γ)/e
1 −(1 −1/e)γ .

C.4
Proofs for the IID model

Proof of Lemma 4.5

Proof of Lemma 4.5. Let F be the cumulative distribution function of X1, a > 0 and ALG the
algorithm with single threshold θ such that 1 −F(θ) = a

n. We denote X∗= maxi∈[n] Xi. As in the
previous proofs, we will begin by lower bounding ALG0(F, n). For any i ∈[n], ALG stops at step i
if and only if Xi > θ and all the previous items were rejected, i.e. Xj ≤θ for all j < i. Thus we can
write

E[ALG0(F, n)] = E
h
n
X

i=1
(Xi1Xi>θ)1∀j<i:Xj≤θ
i

=

n
X

i=1
F(θ)i−1E[Xi1Xi>θ]

= 1 −F(θ)n

1 −F(θ) E[X11X1>θ]

= 1 −F(θ)n

a
× nE[X11X1>θ] ,
(22)

28


---Page Break---
where the second equality is true by the independence of the random variables (Xi)i. On the other
hand, we can upper bound E[X∗1X∗>θ] as follows

E[X∗1X∗>θ] ≤Pr(X∗> θ)θ + E[(X∗−θ)+]

≤Pr(X∗> θ)θ + E
h
n
X

i=1
(Xi −θ)+
i

= Pr(X∗> θ)θ + n
 
E[X11X1>θ] −Pr(X1 > θ)θ


= (1 −F(θ)n −a)θ + nE[X11X1>θ] .

Using the definition of θ, the independence of (Xi)i then Bernoulli’s inequality we have that

Pr(X∗> θ) = 1 −F(θ)n = 1 −

1 −a

n

n
≤1 −(1 −n × a

n) = a ,

and observing that θ = E[θ1X∗≤θ]

F (θ)n
≥E[X∗1X∗≤θ]

F (θ)n
, we deduce that

E[X∗1X∗>θ] ≤−

1 −1 −a

F(θ)n


E[X∗1X∗≤θ] + nE[X11X1>θ] .

by substituting into (22), we obtain

E[ALG0(F, n)] ≥1 −F(θ)n

a


E[X∗1X∗>θ] +

1 −1 −a

F(θ)n


E[X∗1X∗≤θ]

.

Finally, the reward in the γ-prophet inequality is

E[ALGγ(F, n)] = E[ALG0(F, n)] + γE[X∗1X∗<θ]

≥1 −F(θ)n

a
E[X∗1X∗>θ] +
1 −F(θ)n

a


1 −1 −a

F(θ)n


+ γ

E[X∗1X∗<θ]

≥min
1 −F(θ)n

a
, 1 −F(θ)n

a


1 −1 −a

F(θ)n


+ γ

E[X∗] .

The equation 1−F (θ)n

a
= 1−F (θ)n

a

1 −
1−a
F (θ)n

+ γ, is equivalent to

1
(1 −a/n)n −1
 1

a −1

= γ ,
(23)

and for any n ≥2 the function a 7→
 
1
(1−a/n)n −1
  1

a −1

is decreasing on (0, 1] and goes from 1
to 0, thus Equation (23) admits a unique solution an,γ, and taking a = an,γ guarantees a reward of

1−F (θ)n

an,γ
E[X∗] = 1−(1−
an,γ

n
)n

an,γ
E[X∗].

Proof of Theorem 4.6

Proof. We first prove the upper bound, and then we give the single-threshold algorithm satisfying the
lower bound.

Upper bound
We consider an instance similar to the one used in the proof of Theorem 4.4. Let
a, x > 0, and let X1, . . . , Xn be IID random variables with the following the distribution F defined
by

X1 ∼






n
w.p.
1
n2
a
w.p. x

n
0
w.p. 1 −x

n −
1
n2
.

A reasonable algorithm would always reject the value 0 and accept the value n. However, if the
algorithm faces an item with value a, it must decide to either accept it, or reject it with a guarantee
of recovering γa at the end. By analyzing the dynamic programming algorithm ALG⋆, we find that
the optimal decision is to reject a if observed before a certain step j, and accept it otherwise. Let us
denote τ the stopping time of ALG⋆. By convention, we write τ = n + 1 to say that no value was
selected by the algorithm, in which case the reward is γ maxi∈[n] Xi.

29


---Page Break---
If τ ≤j −1 then necessarily Xτ = n, because ALG⋆rejects the value a if it is met before step j,
and if τ = n + 1 then maxi∈[n] Xi ∈{0, a}, because otherwise the algorithm would have selected
the value n and stopped earlier. It follows that the expected output of ALG⋆on this instance is

E[ALGγ
⋆(F, n)] =n Pr(τ < j) +

n
X

i=j
E[Xi | τ = i] Pr(τ = i)

+ γa Pr(τ = n + 1, max
i∈[n] Xi = a) .
(24)

Let us now compute the terms above one by one.

Pr(τ < j) = Pr(∃i ∈[j −1] : Xi = n) = 1 −

1 −1

n2

j−1
≤j

n2 ,

where we used Bernoulli’s inequality (1 −1/n2)j−1 ≥1 −j−1

n2
> 1 −
j
n2 . For i ∈{j, . . . , n},
ALG⋆stops at i if Xi ∈{a, n} and if it has not stopped before, i.e. Xk ∈{0, a} for all k < j and
Xk = 0 for all k ∈{j, . . . , i −1}, hence
Pr(τ = i) = Pr(∀k < j : Xk ̸= n and ∀j ≤k ≤i −1 : Xk = 0 and Xi ̸= 0)

=

1 −1

n2

j−1 
1 −x

n −1

n2

i−j
Pr(Xi ̸= 0)

≤

1 −x

n

i−j
Pr(Xi ̸= 0) ,

the second equality is true by independence, and the last inequality holds because 1 −
1
n2 ≤1 and
1 −x

n −
1
n2 ≤1 −x

n. By independence of the variables (Xk)k, we also have that

E[Xi | τ = i] = E[Xi | Xi ̸= 0] =
E[Xi]
Pr(Xi ̸= 0) =
1 + ax
n Pr(Xi ̸= 0) .

Finally, the event (τ = n + 1, maxi∈[n] Xi = a) is equivalent (maxi∈[j−1] Xi = a, ∀k ≥j : Xk =
0). In fact, the algorithm does not stop before n + 1 if and only if Xk ̸= n for all k < j and Xk = 0
for all j ≤k ≤n, and under these conditions, it holds that maxi∈[n] Xi = maxi∈[j−1] Xi. Therefore

Pr(τ = n + 1, max
i∈[n] Xi = a) = Pr( max
i∈[j−1] Xi = a, ∀k ≥j : Xk = 0)

≤Pr( max
i∈[j−1] Xi ̸= 0) Pr(∀k ≥j : Xk = 0)

=

 

1 −

1 −x

n −1

n2

j−1! 
1 −x

n −1

n2

n−j

=
 
1 −e−xj

n + o(1)
 
e−x+ xj

n + o(1)


=
 
e
xj

n −1

e−x + o(1) .
All in all, we obtain by substituting into 24 that

E[ALGγ
⋆(F, n)] ≤j

n +
1 + ax

n


n
X

i=j


1 −x

n

i−j
+ γae−x 
e
xj

n −1

+ o(1)

= j

n +
1 + ax

n

 1 −(1 −x/n)n−j+1

x/n
+ γae−x 
e
xj

n −1

+ o(1)

= j

n +
 1

x + a
  
1 −e−x+ xj

n + o(1)

+ γae−x 
e
xj

n −1

+ o(1)

= j

n −
h  1

x + (1 −γ)a
i
e
xj

n + 1

x + a −γae−x + o(1)

≤max
s>0

n
s −
h  1

x + (1 −γ)a
i
exso
+ 1

x + (1 −γe−x)a + o(1)

= −1

x
 
log(1 + (1 −γ)ax) + 1 −x

+ 1

x + (1 −γe−x)a + o(1)

= −1

x log(1 + (1 −γ)ax) + 1 + (1 −γe−x)a + o(1) .

30


---Page Break---
On the other hand, we have that

Pr(max
i∈[n] Xi = n) = 1 −

1 −1

n2

n
= 1

n + o(1/n) ,

Pr(max
i∈[n] Xi = 0) =

1 −x

n −1

n2


= e−x + o(1) ,

Pr(max
i∈[n] Xi = a) = 1 −Pr(max
i∈[n] Xi = 0) −Pr(max
i∈[n] Xi = n) = 1 −e−x + o(1) ,

therefore, the expected maximum value is

E[max
i∈[n] Xi] = n Pr(max
i∈[n] Xi = n) + a Pr(max
i∈[n] Xi = a)

= 1 +
 
1 −e−x
a + o(1) .

We deduce that

E[ALGγ
⋆(F, n)]
E[maxi∈[n] Xi] ≤−1

x log(1 + (1 −γ)ax) + 1 + (1 −γe−x)a

1 +
 
1 −e−x
a
+ o(1)

= 1 −

1
x log(1 + (1 −γ)ax) −(1 −γ)ae−x

1 +
 
1 −e−x
a
+ o(1) .

Consequently, for any a, x > 0 and for any algorithm ALG we have

CR(ALG) ≤CR(ALG⋆)

≤lim
n→∞
E[ALGγ
⋆(F, n)]
E[maxi∈[n] Xi]

≤1 −log(1 + (1 −γ)ax) −(1 −γ)axe−x

x +
 
1 −e−x
ax
.

In particular, for x = 2 and a = 1−γ/2

1−γ
we find that

CR(ALG) ≤1 −log(3 −γ) −(2 −γ)e−2

2 + 2−γ

1−γ (1 −e−2)

= 1 −(1 −γ) e2 log(3 −γ) −(2 −γ)

2(2e2 −1) −γ(3e2 −1)
= U(γ) .

This proves the upper bound stated in the theorem, and we can verify that it is increasing, and satisfies
U(0) = 4−log 3

4−2/e2 U(1) = 1

Lower bound on the competitive ratio
We will prove that the algorithm presented in Lemma 4.5
has a competitive ratio of at least (1 −(1 −γ)pγ), where pγ, first introduced in Theorem 4.4, is the
unique solution of the equation (1−(1−γ)p) =
1−p
−log p, which is equivalent to
  1

p−1
 
1
log(1/p)−1

=
γ.

Let aγ = −log(pγ). It follows from the definition of pγ that aγ is the unique solution of the equation
(ea −1)( 1

a −1) = γ. For any n ≥2 and x ≥0 we have that (1 −x/n)n ≤e−x, hence, by definition
of an,γ and aγ

1
e−an,γ −1
  1

an,γ
−1

≤

1
(1 −an,γ

n )n −1
  1

an,γ
−1


= γ

=

1
e−aγ −1
  1

aγ
−1

.
(25)

31


---Page Break---
Moreover, the function x 7→(ex −1)(1/x −1) is decreasing on (0, 1). In fact its derivative at any
point x ∈(0, 1) is

d
dx

 1

e−x −1
  1

x −1

=
 1

x −1

ex −ex −1

x2

= 1

x2
 
1 −x2 −(1 −x)ex

= 1 −x

x2
(1 + x −ex) < 0 .

It follows from (25) that aγ ≤an,γ. Finally, given that x 7→1−e−x

x
is non-increasing on (0, 1], we
deduce that
1 −(1 −an,γ

n )n

an,γ
≥1 −e−an,γ

an,γ
≥1 −e−aγ

aγ
.

We deduce that the competitive ratio of the algorithm described in Theorem 4.6 is at least 1−e−aγ

aγ
=

1−pγ
log(1/pγ) = 1 −(1 −γ)pγ.

D
Random decay functions

While we only studied deterministic decay functions in the paper, it is also possible to have scenarios
with random decay functions. Consider for example that rejected items remain available after j steps
with a probability pj, this is modeled by Dj(x) = ξjx with ξj a Bernoulli random variable with
parameter pj. We explain in this section how the definitions and our results extend to this case.

Definition D.1 (Random process). Let X is a non-empty set. A random process o X is a collection
of random variables {Zx}x∈X . Two random processes Z = {Zx}x∈X and Z′ = {Z′
x}x∈X ′ are
independent if any finite sub-process of Z is independent of any sub-process of Z′. For simplicity, let
us say that the random processes {Z1
x}x∈X 1, . . . , {Zm
x }x∈X m are mutually independent if, for any
x1 ∈X1, . . . , xm ∈Xm, the random variables Z1
x1, . . . , Zm
xm are mutually independent.

Definition D.2 (Random decay functions). Let D = (D1, D2, . . .) be a sequence of mutually
independent random processes. We say that D is a sequence of random decay functions if

1. Pr(Dj(x) /∈[0, x]) = 0 for any x ≥0 and j ≥1,

2. j ∈N≥1 7→Pr(Dj(x) ≥a) is non-increasing for any x, a ≥0,

3. x ≥0 7→Pr(Dj(x) ≥a) is non-decreasing for any j ∈N≥1 and a ≥0.

The second condition asserts that the random variable Dj−1(x) has first-order stochastic dominance
over Dj(x). Along with the first condition, reflect that the distributions of the rejected values become
progressively smaller. The last condition indicates that for any integer j ≥1 and non-negative real
numbers x < y, Dj(y) has a first-order stochastic dominance over Dj(x), which means that, as the
value of x increases, so does the potential recovered value after j steps.

The decision-maker
In the D-prophet inequality with deterministic decay functions, we assumed
that the decision-maker has full knowledge of the functions D1, D2 . . .. In the randomized setting,
we assume instead that the decision-maker knows the distributions of the decay functions, i.e. knows
the distribution of the random variables Dj(x) for all x ≥0 and j ≥1. However, they do not
observe their values until they decide to stop. The online selection process is therefore as follows: the
algorithm knows beforehand the distributions of the decay functions, then at each step, it observes
a new item with value Xi, and decides to stop or continue. Once they decide to stop at some time
τ, they observe the values D1(Xτ−1), . . . , Dτ(X1) and then they choose the maximal one. As a
consequence, the stopping time τ is independent of the randomness induced by the decay functions.
As in the deterministic case, the expected reward of any algorithm ALG can be written as

E[ALGD(X1, . . . , Xn)] = E[ max
0≤i≤τ−1{Di(Xτ−i)}] .

32


---Page Break---
The limit decay
A key result in our paper is the reduction of the problem to the case where all the
decay functions are identical, and we prove this reduction by considering the pointwise limit of the
decay functions. In the case of random decay functions, instead of the pointwise convergence, it holds
for all x ≥0 that the random variables (Dj(x))j converge in distribution to some random variable
D∞(x). In fact, for any x ≥0 and a ≥0, the sequence (Pr(Dj(x) ≥a))j≥1 is non-increasing
and non-negative, thus it converges to some constant G(x, a). Given that x 7→Pr(Dj(x) ≥a) is
non-decreasing for any j, we obtain by taking the limit j →∞that x 7→G(x, a) is non-increasing,
and with similar argument we obtain, for any x ≥0, that G(x, a) = 1 for all a ≤0 and G(x, a) = 0
for all a > x. Therefore, a 7→1 −G(x, a) defined the cumulative distribution of a random variable
D∞such that

• x ≥0 7→Pr(D∞(x) ≥a) is non-decreasing for all a ≥0,
• Pr(D∞(x) /∈[0, x]) = 0 for al x ≥0.

Therefore, for all x ≥0, D∞(x) is the limit in distribution of (Dj(x))j, hence a sequence
D′ = (D′
1, D′
2, . . .) of mutually independent random processes such that D′
j(x) ∼D∞(x) for
any j ≥1 and x ≥0 defines a sequence of decay functions.
We say in this case that all
the decay functions are identically distributed as D∞. Moreover, it holds for all x ≥0 that
E[D∞(x)] = limj→∞E[Dj(x)] = infj≥1 E[Dj(x)]

From there, all the proofs of Section 2 can be easily generalized to the case of random decay
functions, and it follows that we can restrict ourselves to studying identically distributed decay
functions. Moreover, Proposition 3.2 can be generalized to the case of random decay functions, and
the necessary condition for surpassing 1/2 becomes infx>0
E[D∞(x)]

x
> 0. Similarly, using that the
stopping time τ of the algorithm is independent of randomness induced by D∞, Proposition 3.3
remains true with γ = infx>0
E[D∞(x)]

x
.

Lower bounds
For establishing lower bounds, observe that, for any random decay functions D, if
we denote Hj(x) = E[Dj(x)] for all x, then H = (H1, H2, . . .) defines a sequence of deterministic
decay functions. Furthermore, for any instance X1, . . . , Xn and any algorithm ALG, it holds that

E[ALGD(X1, . . . , Xn)] = E[ max
0≤i≤τ−1{Di(Xτ−i)}]

= E
h
E[ max
0≤i≤τ−1{Di(Xτ−i)} | τ, X1, . . . , Xn]
i

≥E
h
max
0≤i≤τ−1

E[Di(Xτ−i) | τ, X1, . . . , Xn]
	i

= E
h
max
0≤i≤τ−1{Hi(Xτ−i)}
i

= E[ALGH(X1, . . . , Xn)].

It follows that lower bounds established for deterministic decay functions can be extended to random
decay functions by considering their expectations.

Implications
With the previous observations, both the lower and upper bounds, depending on γD
that we proved in the deterministic D-prophet inequality can be generalized to the random D-prophet
inequality, by taking

γD = inf
x>0 inf
j≥1
E[Dj(x)]

x
.

33


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: In the introduction, present the problem, and state the main contributions of
the paper.
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
Justification: The limitations are discussed in the conclusion.
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

34


---Page Break---
Justification: All the assumptions are clearly stated, and full proofs are in the appendix.
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
Justification: No experimental results.
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

35


---Page Break---
Answer: [NA]
Justification: No experimental results.
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
Justification: No experiments.
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
Justification: No experiments.
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

36


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

Answer:[NA]

Justification: No experiments.

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

Justification: We have reviewed the Code of Ethics, and the paper conforms to it.

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

Justification: The contribution of the paper is theoretical. We do not feel there is major
societal impact that needs to be discussed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

37


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

Justification: The contribution of the paper is mainly theoretical.

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

Justification: We do not use any assets.

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

38


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not provide any new assets.
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
Justification: No such experiments are included in the paper.
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
Justification: No such studies are included in the paper.
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

39


---Page Break---
