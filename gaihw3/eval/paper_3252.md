Optimizing the coalition gain in Online Auctions with
Greedy Structured Bandits

Dorian Baudry1,2,∗
Hugo Richard2,∗
Maria Cherifa2,∗
Clément Calauzènes2

Vianney Perchet2

1 Department of Statistics, University of Oxford. 2

2 Inria Fairplay Joint team, CREST, ENSAE Paris, Criteo AI Lab

Abstract

Motivated by online display advertising, this work considers repeated second-price
auctions, where agents sample their value from an unknown distribution with
cumulative distribution function F. In each auction t, a decision-maker bound
by limited observations selects nt agents from a coalition of N to compete for
a prize with p other agents, aiming to maximize the cumulative reward of the
coalition across all auctions. The problem is framed as an N-armed structured
bandit, each number of player sent being an arm n, with expected reward r(n)
fully characterized by F and p + n. We present two algorithms, Local-Greedy (LG)
and Greedy-Grid (GG), both achieving constant problem-dependent regret. This
relies on three key ingredients: 1. an estimator of r(n) from feedback collected
from any arm k, 2. concentration bounds of these estimates for k within an
estimation neighborhood of n and 3. the unimodality property of r under standard
assumptions on F. Additionally, GG exhibits problem-independent guarantees
on top of best problem-dependent guarantees. However, by avoiding to rely on
confidence intervals, LG practically outperforms GG, as well as standard unimodal
bandit algorithms such as OSUB or multi-armed bandit algorithms.

1
Introduction

The online display advertising has seen remarkable evolution in recent decades [14, 30, 33, 25].
Publishers, who are the suppliers of digital ad space on the internet, sell display spots for ads to
advertisers through real-time bidding in spot auctions, with many of these auctions being conducted
using first or second-price mechanisms [20]. Due to the technological complexity of online advertis-
ing, advertisers usually delegate the task of buying ad placements to demand-side platforms (DSP)
that operate many advertising campaigns. This interaction between DSP and the publisher, can be
simplified as the publisher acting as multiple ad auctions selling ad impressions (online displays),
while the DSP acts as a centralized coalition: at each time step, it determines which campaign(s)
from the coalition participate to the auction to maximize their total gain. The chosen campaign(s)
then compete with others to secure impressions. The primary goal of advertising companies is then
to maximize the cumulative utility: the total value of impressions won minus their costs. This raises
a fundamental question: how many ad campaigns should participate in the auction to optimize the
overall utility? In the interim setting, where the DSP observes current bidder values before deciding,
it’s known that only the highest value bidder should be sent. However, online privacy enhancements
in browsers necessitate ex-ante decisions from DSPs [8], without exact value knowledge. Here, the
problem becomes challenging: choosing a small number of campaigns can make it difficult to secure
impressions, while securing the spot with a large number of bidders inevitably raises the price due

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

∗Equal contribution.
2Corresponding author: dorian.baudry@ox.ac.uk


---Page Break---
to competition. In this paper, this problem is formalized and solved via novel Multi-Armed-Bandit
(MAB) algorithms.

Problem statement
Consider a sequence of T ad impressions sold through second price auctions
(see [20] for a survey). At auction t ∈[T], each participant (bidder) bids on the item based on its own
(stochastic) value for the item. The highest bidder wins the item and pays a price equal to the second
highest bid. The decision maker (the DSP) runs N ∈N∗advertising campaigns forming a coalition.
At time t, two groups of bidders participate: (1) nt ∈[N] bidders from the coalition chosen by the
decision maker ex-ante – without knowing the realization of the bidders’ values – and (2) p ∈N∗
other bidders, that we call the competition. When a bidder from the coalition wins the auction, the
decision maker observes the realized value for the winner (also called winning bid). In the rest of the
paper, the following assumptions about the behavior of bidders is made.

Assumption 1. All bidders are identical, their values are sampled i.i.d. from a distribution supported
on [0, 1] characterized by its cumulative distribution function (c.d.f.) F. All bidders bid their value.

Assuming identical bidders with i.i.d values is a strong but widespread assumption in auction theory
[20, 22], known as the symmetric bidders case. It is particularly relevant in online advertising, notably
in homogeneous impression markets where advertisers compete for similar ad displays due to shared
objectives, target demographics, or placement competition. The bounded support assumption is also
standard, as letting an automated system bid arbitrarily large values is unrealistic. Finally, bidders
bid their value as this is a weakly dominant strategy in this case. Lastly, assuming a known number
of competitors p is frequently seen in auction models (see for instance [20] chapter 3.2.2). Under
Assumption 1, the expected reward received by the decision maker at time t is given by r(nt), where
r is the expected reward function, defined by

r : n ∈[N] 7→r(n) := Ev=(vi)i∈[n+p]∼F ×···×F


(v(1) −v(2))1

argmax
i∈[n+p]
vi ∈[n]

(1)

where v(1) and v(2) are respectively the first and second maximum of v, and [n] is used to abbreviate
{1, . . . , n}. The problem therefore reduces to a MAB where the decision maker chooses arms
n1, . . . , nT ∈[N] sequentially and aims to minimize its cumulative expected regret R(T) defined by

R(T) =
X

t≤T
r(n∗) −r(nt) ,
with
n∗= argmax
n∈[N]
r(n),
(2)

given that privacy constraints from the browser [8] only let the decision maker observes (1) if the
coalition won, (2) the realization of the maximum value when winning.

Related works
Following (2), the problem presented in this paper can be formulated as a Multi-Arm
Bandits (MAB, see [21] for a survey). In MAB, a learner repeatedly selects from a set of actions, or
“arms”, each yielding a reward. The goal is to maximize total rewards by striking a balance between
exploration (sampling various arms to learn their rewards) and exploitation (picking the arms with the
highest anticipated rewards based on collected feedback). While the literature has known a significant
development in the last years ([2, 7, 17], to name a few), the most popular approaches arguably
remain exponential weights algorithms (EXP3, [4]) in adversarial settings, and optimism in face of
uncertainty (UCB, [3]) when rewards are stochastic.

While UCB and EXP3 can both tackle the regret minimization problem presented here, they inevitably
achieve sub-optimal performance due to not using the inherent structure of the expected reward
function. Several types of structure have been explored in the bandit literature, some notable examples
being linear bandits [1], Lipschitz bandits [23], or unimodal bandits [9, 28, 29]. The problem consid-
ered here is novel in the literature of structured bandits, arising from the observability restrictions
coming with privacy-enhancing systems. Still, in the next section we show that unimodality – in this
paper the fact that r admits only one local (hence global) maximum – is in many cases inherited from
this stronger structure. A typical strategy to exploit unimodality - also used in this work - consists in
playing a standard bandit policy (such as UCB) on a well chosen subset of arms (OSUB, [9]).

Last, the use of online learning algorithms to tackle repeated auction problems have been explored
in various contexts ([26, 12, 5, 32, 27, 11]). However, none of these works approach the problem
through the perspective of a coalition of bidders, and are thus not applicable to this setting.

2


---Page Break---
Table 1: Comparison of regret guarantees for different algorithms

Algorithm
Regret upper bound

EXP3 [4]
O(
√

NT)
UCB1[3]
O
P

n∈[N]
log(T )

∆n
∧
√

NT


OSUB [9]
O

log(T )
∆n⋆+1 + log(T )

∆n⋆−1 + P

n∈[N]
∆n log log(T )

∆2


LG (this paper)
˜ON(P
n∈[N]
∆n
∆2 )
GG (this paper)
˜ON(P

n∈B⋆
1
∆n + P

n∈S
∆n
∆2 ) ∧e
O(
p

(log(N) + |B⋆|)T)

Outline and contributions.
Section 3 presents two novel bandit algorithms: LG (Local Greedy)
which is inspired by OSUB[9], and GG (Greedy Grid) which combines Local Greedy and a successive
elimination strategy. Theorem 2 and Theorem 3 provide upper bounds on the regret of LG and GG
respectively, which are summarized in Table 1. Both algorithms achieve problem-dependent regret
independent of T. However, their scaling differs: the regret of LG depends on the worst local gap
∆= minn∈[N] |r(n + 1) −r(n)|, while for GG it only depends on the gaps ∆n = r(n∗) −r(n).
Furthermore, w.h.p. GG only suffers regret for arms in a reference grid S containing O(log(N)) arms
and in a neighborhood B⋆of the optimal arm. All these quantities, as well as the notation e
O and
e
ON (hiding logarithmic factors), are defined in Section 3. These regret upper bounds rely on three
key ingredients presented in Section 2: (1) an estimator of r(n) from feedback collected from any
arm k (2) novel concentration bounds on these estimates for k within an estimation neighborhood
of n (Theorem 1) and (3) the unimodality property of r under standard assumptions on F. Lastly,
Appendix D provides an experimental benchmark comparison of the performance of GG, LG and their
competitors: LG has the lowest expected regret among the algorithms tested. Indeed, LG avoids the
explicit use of the confidence bounds in the algorithm which makes it more practical, even though GG
admits better theoretical guarantees.

2
Estimating the reward function from samples of powers of F

In this part, we put aside the sequential nature of the repeated auction setting that we introduced
and consider the problem of estimating the expected reward as a function of the number of bidders,
given a stream of collected data. We first present a formulation of the expected reward function in
terms of powers of the c.d.f. F. Then, we leverage this formula to introduce power estimates, as a
solution to estimate the expected reward of an arm n ∈[N] from samples collected from an arm
k ∈[N]. Lastly, we discuss the theoretical properties of these estimates, introducing upper and lower
confidence bounds on the expected reward in Theorem 1.

2.1
Properties of the expected reward

The expected reward function r (Eq. (1)) can be expressed as a function of n, p and the c.d.f. F.

Lemma 1. The expected reward function defined in Equation (1) satisfies,

n ∈[N] 7→r(n) = n
Z 1

0
F p+n−1(x) −F p+n(x)dx
(3)

The proof can be found in Appendix A.1 and is based on properties of order statistics.

This particular definition of r(n), which is a product of n and a function that decreases with n,
suggests that r could be unimodal for some choices of F. In the rest of the paper, we restrict ourselves
to distributions that guarantees unimodal reward functions.

Assumption 2. F and p are such that the reward function r in Equation (3) is unimodal

As the next lemma shows, many classical distributions lead to unimodal rewards for all p ∈N.
Lemma 2. Let F be the cumulative distribution function of a Bernoulli, truncated exponential or
Complementary Beta distribution. Then, for any p ∈N∗, r in Equation (3) unimodal.

3


---Page Break---
The proof of Lemma 2 can be found in Appendix A.2. Note that the Complementary Beta distribu-
tions [19], chosen for technical reasons, are similar to Beta distributions and any Beta distribution
can be approached by a Complementary Beta. Furthermore, in Appendix A.3 we present experiments
suggesting that r is unimodal for all p ∈N∗if F is the c.d.f of Beta or Kumaraswamy distributions.
However, we also show in Appendix A.4 that this is not always the case, by providing a counter
example. Nonetheless, we argue that (complementary) beta or truncated exponentials are flexible
models for real world data, so Assumption 2 is reasonable in practice. We furthermore discuss in
Section 3.2 the adaptation of our algorithms if this was not the case.

2.2
Estimation of powers of F

Consider the feedback Wk = (wk,1, . . . , wk,mk) gathered after playing arm k and winning the
auction mk times. Wk, represents the sequence of first values (value of the winning bid) which has
been collected by arm k.

It is well known that the marginal distribution of any order statistic can be expressed as a function
of the c.d.f. F (see Section 2.1 of [10]). The distribution of any element of W k has cumulative
distribution function Fk : x ∈[0, 1] →F k+p(x), which clearly exhibits a one-to-one mapping
between Fk(x) and F(x). Hence, given W k, for any ℓ∈N we can estimate F ℓby

eF ℓ
k+p : x 7→( bFk+p(x))
ℓ
k+p , where bFk+p : x 7→
1
mk

mk
X

j=1
1{wk,j ≤x} (emp. c.d.f. of Wk).
(4)

Estimation of r
Consider any arm n ∈[N]. Following Equation (3), it appears that estimating both
F n+p and F n+p−1 is sufficient to construct an estimate of r(n). According to Equation (4), this can
be done from samples originated from any arm k ∈[N], by using the simple estimate

brk(n) = n
Z 1

0


eF n+p−1
k+p
(x) −eF n+p
k+p (x)

dx .
(5)

Furthermore, it also clear that any convex combination of estimates can become a new estimate,
however in the rest of the paper we focus on simple estimates for simplicity.
Remark 1 (Adaptation to different feedback). A similar procedure can be derived for a setting
where the sequence of second prices would be observed instead. Indeed, their distribution would be
Gk : x ∈[0, 1] 7→(k + p)F(x)k+p−1 −(k + p −1)F(x)k+p, which can lead to a reward estimate
similar to (5) by using a suitable inversion formula. The same can be said for the case where both
first and second prices are observed, with additional complexity because the joint distribution should
be considered since for each auction the first and second price are dependent variables.

2.3
Concentration of estimates of the reward function

We now introduce the first theoretical contribution of this paper: confidence bounds on the deviations
of an empirical estimate brk(n) w.r.t. the true expected reward r(n).

Importance of (relatively) local estimation
In principle, (5) suggests that samples from any arm
k ∈[N] can provide a simple estimate of the reward function of any other arm n ∈[N]. However,
we establish that the position of k w.r.t. n significantly impacts the concentration of brk(n). Intuitively,
the ratio(n + p)/(k + p) determines how the uncertainty on F k+p propagates on the reward after
performing the inversion to obtain an estimate of F n+p. Indeed, considering any i ∈N, if for some
x ∈[0, 1] the deviation F(x)i −bFi(x) is small then a first order approximation provides that

∀j ∈N :
(F(x)i)
j
i −bFi(x)
j
i ≈(F(x)i −bFi(x)) × j

i Fi(x)
j
i −1 .
(6)

Hence, a small error on F(x)i is multiplied by j

i Fi(x)
j
i −1 to obtain the resulting error on F(x)j. For
j ≥i this term can be as large as j/i while for j < i it can be arbitrarily large if F i(x) is very small.
This observation motivates a restriction on the range of arms that can be used to estimate the reward
of a given arm n, that we call its estimation neighborhood. We use the convention that arms smaller
than 1 or greater than N exist but have not collected any sample and have a known reward of 0.

4


---Page Break---
Definition 1 (Estimation neighborhood of an arm n). Assume3 that p ≥4 Then, the estimation neigh-
borhood of n is the range V(n) = [vℓ(n), vr(n)] =

k ∈[N] : k + p ∈
 n+p

2 , 3

2(n + p −1)
	
. We
call vℓ(n) and vr(n) respectively the furthest left and right neighbor of n.

Theorem 1 (Concentration of simple estimates). Consider any n ∈[N] and k ∈V(n). Let brk(n)
be defined according to (5) from mk samples collected by k. Then, there exists some constants βk,n
(depending on n, k, p) and ξk,n,F (additionally depending on F) such that, with probability 1 −δ,

|brk(n) −r(n)| ≤βk,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ n × ξk,n,F




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

.
(7)

Furthermore, the constants admit universal upper bounds for any n, k, p, F. For instance if mk ≥4
it holds that βk,n ≤33 and γk,n,F ≤100.

Proof sketch (see Appendix B for the detailed proof). The first ingredient consists in approximating
the reward formulation (3) by a Riemann sum: for some step size D−1 > 0, it holds that brk(n) −
r(n) =
n
D
PD−1
s=0 E(xs) + errD, with xs = s/D for all s ∈{0, . . . , D −1}. In Lemma 4 we
use elementary properties of F to show that the approximation error satisfies errD ∈[0, nD−1].
Next, we upper and lower bound E(xs) with different concentration bounds according to the value
of Fk,s := F(xs)k+p. More precisely, for any δ ∈(0, 1) the following bounds hold each with
probability at least 1 −δ,













| bFk(xs) −Fk,s| ≤
p

Fk,s ×

r

3 log( 2

δ)
mk
if Fk,s ∈I0 :=
h
3 log(2/δ)

mk
, 1
i
(Chernoff) ,

bFk(xs) ≤6 log(2/δ)

mk
if Fk,s ∈I1 :=

δ
mk , 3 log(2/δ)

mk


(Chernoff) ,

bFk(xs) = 0
if Fk,s ∈I2 :=
h
0,
δ
mk

i
(union bound) .

These results are derived in Lemma 5 from a well-known multiplicative form of the Chernoff bound
for Bernoulli random variables [16]. Then, the analysis consists in using the appropriate bound for
each point s ∈{0, . . . , D −1}. The interval I0 provides the first term in (7), which is dominant in
terms of mk, and we make βk,n fully independent of F by carefully using some properties of the
reward function. The two remaining intervals I1 and I2 provide the second term in (7), and γk,n,F
depend on F through the boundaries of the interval I1. The corresponding factor in ξk,n,F can be
bounded by 1 or estimated in practice (see Appendix B.4).

In Appendix B, we give the expression of βk,n and ξk,n,F and provide in (17) and (19) fully explicit
upper and lower confidence bounds on brk(n), depending on all problem parameters, and that are
much tighter than what the universal constants provided in the theorem suggest. These universal
constants are purely indicative, in order to assess that βk,n and ξk,n,F do not diverge for any value of
the problem parameters. We now provide more high-level comments on the derivation of this result.

Discussion
The proof of Theorem 1 is non-trivial, and the careful usage of the Chernoff bounds that
we introduced is crucial to obtain tight bounds on brk(n) for two reasons. First, it seems necessary
to concentrate estimates from arms k > n (see the discussion below (6)), which are instrumental
to the performance of the bandit algorithms presented in the next section. Secondly, by exhibiting
powers of F, they make βk,n not increasing linearly in n, which is not easy to achieve. Indeed, it is
clear from the analysis that this cost would be inevitable with standard Hoeffding bounds. However,
completely avoiding n seems difficult in general, so our proof provides a way to mitigate its cost
by multiplying it by a higher power of m−1
k , at least m
−2

3
k
(if k + p = 3

2(n + p −1)). This is the
theoretical motivation for the definition of V(n) (Definition 1): while k + p = 2(n + p −1) would
lead to theoretically valid results, it would not ensure that the linear term in n is second-order in mk.

We now conclude this section by exhibiting a condition on F that allows to reduce the scaling of the
confidence bound in n to logarithmic terms.

3This assumption simplifies the presentation but our theoretical results can easily be adapted for p < 4 if
n −1 and n + 1 are included by default in V(n).

5


---Page Break---
Lemma 3 (Improved bound for Lipschitz quantile function). Assume that k ∈V(n) and F −1 is
L-Lipschitz, then there exists an absolute constant ξ such that with probability 1 −δ it holds that

|brk(n) −r(n)| ≤βk,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ ξL log
4⌈n√mk⌉

δ

 


log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

(8)

This result is proved in Appendix B.3, and shows that for some distributions (e.g. “close” to uniform)
the confidence bounds converge relatively fast to standard sub-Gaussian type of bounds, even for
very large n. Whether this result holds in general remains open.

3
Bandit algorithms

3.1
Bandit algorithms: Local-Greedy (LG) and Greedy-Grid (GG)

We now detail the two novel bandit algorithms proposed to tackle the problem presented in Section 1.
Both rely on the use of simple estimates of r(n) (see Section 2) by arms present in its estimation
neighborhood V(n) (see Definition 1) and theoretically motivated by Theorem 1. In this section,
for ease of exposition, we describe algorithms as if feedback was collected at every time steps. In
Appendix C.1, we show that the algorithms and their guarantees only require a slight adaptation when
the feedback is collected only when the auction is won.

3.1.1
Local-Greedy

We first present Local-Greedy (LG), which is a natural adaptation of a standard policy in unimodal
bandits, OSUB [9].The main idea of OSUB is to play UCB locally around a reference arm, and eventually
reach the optimal arm n⋆by gradually moving the reference arm in its direction. With LG, we adapt
this principle to efficiently exploit the structure of the problem considered: at each round t, LG
defines a reference arm ℓt, called leader, but plays greedily in the neighborhood V(ℓt), based on
simple power estimates computed with samples from ℓt only. In addition a sampling requirement,
implemented by a parameter α ∈(0, 1), is used in order to ensure the good concentration of these
estimates. We detail Local-Greedy in Algorithm 1 below.

Algorithm 1 Local Greedy (LG)
Input: exploration parameter α, neighborhoods (V(n))n∈[N] (Definition 1)
Play n1 = 1 and observe w ∼F 1+p ;
▷Initialization
for t ≥2 do

Set ℓt = nt−1, compute (brℓt(n))n∈V(ℓt) (Eq.(5)) ;
▷Compute estimates from the leader
If mt := |{s ∈[t −1], ns = ℓt}| ≤αt: play nt = ℓt ;
▷Linear sampling requirement
Else: play nt ∈argmaxn∈V(ℓt) brℓt(n) ;
▷Greedy play in V(ℓt)
Observe w ∼F nt+p ;
▷Record feedback

High-level properties of LG
First, using Greedy instead of UCB is only possible because of the
structure of the problem: when ℓt is well-explored the estimates of arms in V(ℓt) computed with
samples from ℓt are sufficiently close to the true reward, so that no exploration is needed. The
sampling requirement then guarantees that all greedy plays are made when ℓt is well explored.

A second property is that since |V(ℓt)| grows with ℓt, a sequence of locally optimal moves (best play
in a given neighborhood) allows to reach the optimal arm exponentially fast (in O(log(N)) steps),
which is particularly interesting in practice if N is large. On the other hand, LG might suffer from the
inherent drawback of any “local” policy: identifying a high-rewarding arm in a neighborhood can
take a long time if the reward curve in this area is flat (depending on how small are the “local” gaps).
This problem can be attenuated, but not solved, by adding an initial exploration phase. We propose
Greedy-Grid, presented in the next section, as a way to fully address this issue.

Lastly, requiring only the computation of empirical reward estimates is a strength of Local-Greedy.
Indeed, deriving tighter confidence bounds would improve its analysis, but not the practical imple-
mentation (and performance) of the algorithm.

6


---Page Break---
3.1.2
Greedy-grid

The concept of Greedy-Grid is very intuitive: it plays a Local-Greedy strategy only if it can tell
which segment of the reward function contains the best arm with high probability. To implement this
idea, GG uses a Successive-Elimination procedure [13] on a subset of arms forming a reference grid,
denoted by S.

Reference grid
The grid S is designed so that two of its successive arms belong to their respective
neighborhood (Definition 1), and can hence mutually estimate themselves and all arms in between
(Theorem 1). In particular, the optimal arm can be well-estimated at least by its two closest neighbors
on the grid, so its neighborhood can be “discovered” with high probability simply by sampling the
points in the grid in a round-robin fashion for a sufficiently long time.

Following Definition 1, we construct S := {si}i≥1 recursively: s1 = 1, and for i ≥2 we set
si+1 = max {s ≥si : s ∈[N], s ∈V(si), si ∈V(s)}. We provide an illustrative example below.
Example 1. For N = 2000 and p = 100 the grid is S = {1, 50, 123, 233, 398, 645, 1016, 1572}.

Any arm n ∈[N] admits a left and right “neighbor in the grid”, denoted respectively by vS
l (n)
and vS
r (n) and defined by: vS
l (n) = 0 if n < min S, vS
r (n) = N + 1 if n > max S and
(vS
l (n), vS
r (n)) = argmin(x,y)∈S\{n}: n∈[x,y](y −x) otherwise . We call the "bin" of arm n all
arms between its left and right neighbors: B(n) = {n ∈[N], vS
l (n′) < n < vS
r (n′)}. For simplicity
we use the notation B⋆= B(n⋆)4.

Greedy-Grid
We provide the detailed implementation in Algorithm 2 below, and now describe the
general principle of the algorithm. At each round, it operates in two steps. In the first step, it decides
whether to play arms on the grid S (play the grid, to simplify), or to focus on a specific bin (and, as
we will see, play greedy). This choice depends on an elimination procedure: an arm k in S should
be eliminated for this round if their upper confidence bound (UCB) is smaller than the best lower
confidence bound (LCB) among all other arms. Furthermore, if there exists an eliminated arm whose
index is closer to the index i∗
t of the arm with the best LCB, then the unimodality assumption implies
that k should also be eliminated. The set of arms not eliminated at t is called Ct in Algorithm 2.

To compute the UCB (Un) and LCB (Ln) of an arm n, we elect a leader ℓn which is the arm in
[vS
l (n), vS
r (n)] that was played the most in the last t rounds and then compute the bounds based on
ℓn, using Theorem 1. We show in the proof of Theorem 3 that this procedure ensures that a linear
number of samples in t is used to compute the UCB and LCB of arms in B⋆with high probability.

If at least one arm is not eliminated (Ct is not empty), arms in Ct are played one after the other (Round
Robin). If all arms in the grid are eliminated, GG plays greedily in the bin B(i∗
t ) of the arm with
the highest LCB. The empirical reward of each arm n ∈B(i∗
t ) is computed similarly as Un and Ln
using samples from the leader ℓn. GG then plays the best empirical arm αt times which is the same
sampling requirement as LG.

The careful design of Greedy-Grid prevents the main theoretical drawback of Local-Greedy: since the
algorithm has a very low probability to play in a sub-optimal bin, it almost never pays “local gaps” in
a sub-optimal part of the reward function. However this guarantee comes at a cost: if n⋆is not in the
grid, it will never be played until the confidence intervals shrink “sufficiently” to eliminate the entire
grid. Hence, GG might be more conservative than LG in practice, while offering better theoretical
guarantees. We express this trade-off in the next section.

3.2
Regret upper bounds

We now present the theoretical results obtained for the two algorithms presented in Section 3. We
first establish the regret bounds and sketch their proofs, before discussing and comparing the results.
We introduce some notation, that considerably simplifies the presentation of the results.

Notation: e
O and e
On
For any x > 0, we use the notation e
O(x) to describe a quantity that scales in
x, up to logarithmic terms in x and N (hence the notation is linked to the problem). Furthermore,

4We assume a unique optimal arm for simplicity, but the analysis holds if several successive arms are optimal.

7


---Page Break---
Algorithm 2 Greedy Grid
Input: Grid S, confidence levels (δt)t∈N, sampling parameter α
Play n1 = min S and observe w ∼F n1+p
for t ≥2 do

∀n ∈[N] ℓn = argmaxk∈[vS
l (n),vS
r (n)]

mk
z
}|
{
|{u ∈[t −1], nu = k}| ;
▷Elect leaders

∀n ∈[N], Ln = bLℓn(n, δt) and Un = bUℓn(n, δt) ;
▷Compute UCB (21), LCB (22)
i∗
t = argmaxn∈[N] Ln ;
▷Compute best lower bound index
Ct = {a ∈S : ∀s ∈[N] s. t. a ≤s ≤i∗
t or a ≥s ≥i∗
t , Us ≥Li∗
t } ;
▷Non-elim. grid arms
if nt−1 ∈B(i∗
t ) and mnt−1 ≤αt then
▷Ensure linear sampling for bin plays
Play nt = nt−1
else
▷Play grid if non-empty or greedy in the best LCB’s bin
If Ct = ∅: Play Round Robin on Ct; Else play argmaxn∈B(i∗
t ) ˆrℓn(n)

Observe w ∼F nt+p ;
▷Record feedback

for n ∈[N] we also use e
On as a shorthand notation for e
O(

n6 ∨x
	
∧n2x). This type of constants
emerges from using (7) (Theorem 1) in the analysis. Indeed, we proved that the simple estimate of an
arm n by an arm k ∈V(n) admit sub-Gaussian (“square-root”) confidence intervals, independent of
n, when the sample size of k is larger than Ω(n6).
Theorem 2 (Regret bound for Local-Greedy). Let ∆:= minn∈[N−1] |r(n + 1) −r(n)| (worst local
gap). Under Assumption 2 and with α = (log3/2 N + 1)−1, the regret of LG is upper bounded by a
problem-dependent constant: there exists (Cn)n∈[N]\{n⋆}, each satisfying Cn = e
ON
  ∆n

∆2

, such
that RT ≤P
n∈[N]\n⋆Cn.

Additionally, if the arm set forms a single estimation neighborhood, that is ∀n ∈[N] : V(n) ⊃[N],
then each constant Cn can be refined to e
On
 
∆−1
n

, providing RT = e
O(
√

NT), which holds even
when the reward function is not unimodal.

Proof sketch (see Appendix C.3 for the detailed proof). We start by the case where the arm set forms
a single neighborhood. Since LG is guaranteed that any arm it selects will provide an estimate for
all the other arms, this context is very similar to a full information scenario. This explains why
GG achieves both constant regret depending on the gaps, and a gap-independent bound in
√

NT.
Furthermore, the hidden logarithmic constants come from carefully using Theorem 1 to separate the
linear term in n from the gaps when they are small.

The general case presents an additional complexity. Indeed, it is possible that playing arm n ̸= n⋆is
locally optimal, if n is the best arm in the neighborhood of the current leader: playing n in that context
would not be unlikely. To tackle that scenario, we prove that pulling arm n at time t necessarily
implies a locally sub-optimal play, in some estimation neighborhood, at some point in the past
(maximized by the chosen value of α). We then show that this cannot happen after some deterministic
time w.h.p., leading to constant regret. However, since the sub-optimal play might be any arm the
constant now depends on the worst local gap ∆2.

Theorem 3 (Regret upper bound for Greedy-Grid). Suppose that GG is tuned with confidence level
δt =
1
N 2t3 , and α = 1/4. Then, for any T ∈N it holds that

RT = e
ON

 X

n∈B⋆

1
∆n
+
X

n∈S

log(T)

∆n
∧∆n

 
1{n < n⋆}

∆2
vl(n⋆)
+ 1{n > n⋆}

∆2
vr(n⋆)

!!

.

Additionally, it holds that RT = e
O
p

(K + |B⋆|)T

, for K = ⌊log3/2(N)⌋.

Proof sketch (see Appendix C.4 for the detailed proof). First we prove that, w.h.p., during a linear
time range in t GG either played the grid or in B⋆. Hence, arms n ∈[N]\{S ∪B⋆} are played a
(universal!) constant number of times by GG in expectation. Then, for n ∈S the term in log(T )

∆n
comes
from the standard analysis of UCB [3]; while the constant bound comes from exploiting that after

8


---Page Break---
a constant time n the LCB of n⋆eliminates its neighboors w.h.p., and by extension the entire grid.
Finally, the constant bound n ∈B⋆is derived similarly as the first bound of Theorem 2.

Discussion
First, we show that being able to estimate r(n) from the feedback obtained after playing
an arm k in its estimation neighborhood leads to a regret independent of T for both LG and GG. For the
former, the bound depends in general on the worst local gap ∆, while for the latter only the actual gaps
∆n (with n⋆) are involved. This difference permits to obtain a problem-independent guarantee for GG
for any configuration of p and N. Furthermore, its scaling
p

K + |B∗| ≤
q

2n∗+ ⌊log3/2(N)⌋can

be much smaller than
√

N if n∗is small.

Then, we would like to discuss the impact of the concentration bound presented in Theorem 1 on
the regret of both GG and LG. Indeed, a naive approach with Hoeffding bounds would not allow to
remove n from the first order term of the concentration bound, because of the multiplicative factor n
in the definition of r(n). A feature of our concentration bound is that the linear scaling in n does
not appear in the first order term. Informally, this allows to exhibit terms of order e
ON(∆−1
n ) in the
regret analysis instead of e
O(N 2∆−1
n ), which can be significantly better for small gaps. A remark
here is that the size of the grid in GG could be optimized as a larger grid makes the second order term
in Theorem 1 smaller but is paid linearly in the regret.

We nevertheless highlight some potential for improvement in the analysis of LG. First, the local gaps ∆
in the bound of LG could be replaced by (in spirit, referring to S for simplicity) minn∈[N] |r(vS
l (n))−
r(vS
r (n))|. It is clear though that this gap remains “local” and can be arbitrarily smaller than ∆n for
some arms n ∈[N], so the general interpretation of the results would be unchanged. Second, for
simplicity, the analysis of LG was carried out using the constant upper bound of βk,n and ξk,n,F but a
tighter analysis could lead to a better dependency with respect to N.

We now justify the use of simple estimates in GG and LG. In practice, combining estimates would
allow to use more samples for the estimation. However, this would make the algorithm slower, and
we believe that the sampling requirement implemented in the algorithms makes the use of simple
estimates efficient: potential uniform exploration in a neighborhood is replaced by a focus on a single
arm, but the same quality of information is accrued. Furthermore, from a theoretical perspective
union bounds over the samples collected by each arm might also cost a factor N in the analysis.

Lastly, while GG admits better theoretical guarantees, LG might be more appealing in practice because
it does not require to explicitly compute confidence intervals. This means that the regret bounds
provided for LG are conservative, and might be refined with tighter confidence bounds without
changing the algorithm.

Adaptation for non-unimodal rewards
While LG relies heavily on Assumption 2, GG can be
readily adapted to handle non-unimodal reward functions. This is done by modifying the definition
of the set of non-eliminated grid arms Ct to {s ∈S, Us ≥Li∗
t } in Algorithm 2. In that case,
the algorithm can no longer eliminate arms on the grid based on the elimination of other arms.
This naturally induces that the number of plays of sub-optimal arms is no longer bounded by a
constant. In Theorem 4 (see Appendix), we show that only the O(log(T)) term persists for n ∈S
in Theorem 3, while the problem-independent bound remains unchanged. Although we believe
unimodality is necessary for achieving constant regret, this result demonstrates that, even without that
assumption, GG can still provide the same logarithmic regret guarantees as UCB. However, it does so
on a |S|-armed bandit, rather than an N-armed bandits with |S| = O(log(N)) ≪N for large N.

Experimental results
In Appendix D we present a benchmark of LG, GG, UCB, EXP3 and OSUB
on synthetic data in terms of the expected regret R(T). This benchmark illustrates the strong
performance of LG relative to the other approaches. Although GG offers more robust theoretical
guarantees, particularly with sub-linear problem-independent bounds, LG proves to be more effective
in practice. Several factors may explain this gap between theoretical guarantees and empirical
performance. First, as discussed in the previous section, the worst-case local gap in the analysis of
Local Greedy (Theorem 2) might be overly conservative. This worst-case scenario could occur under
a combination of unfavorable conditions, such as poor initialization far from the optimal arm and a flat
reward function, paired with bad luck in exploration. However, such a scenario is likely rare in practice
and was not encountered in our experiments. Additionally, Local Greedy benefits from scenarios

9


---Page Break---
where it starts playing in the optimal neighborhood only after a few steps, a situation GG cannot exploit
due to its need for sufficient statistical evidence to eliminate all suboptimal neighborhoods. While
GG’s caution leads to stronger theoretical guarantees, this comes at the cost of empirical performance.
Moreover, GG’s results are tied to the tightness of the confidence intervals in Theorem 2, a limitation
that does not apply to LG. An interesting and challenging open problem remains whether LG can be
modified to achieve the same theoretical guarantees as GG without sacrificing its performance. We
leave this question for future work.

4
Conclusion

The bandit problem studied in this work is structured since playing arm n gives a reward r(n)
determined by n, p and the unknown c.d.f F and with probability
n
n+p an observation of a sample of
the distribution with c.d.f F n+p.

While traditional bandit approaches give problem dependent bounds depending on T, algorithms
GG and LG presented in this work have constant problem dependent bounds. Furthermore, GG and
LG avoid a quadratic dependency in N for large T thanks to new concentration bounds introduced
in Theorem 1. Overall, while GG has the best theoretical guarantees, LG has better constants and is
therefore better suited for most practical problems (see the discussion at the end of Section 3 and
experimental results in Appendix D).

Whether an algorithm that has the theoretical guarantees of GG and the practical performance of LG
can be designed is an interesting question. We believe that the main leverage to improve the practical
performance of GG might be to derive tighter concentration bounds. Possible directions to improve
over Theorem 1 might include: further refining the decomposition of the integral in (3) according to
the value of F, further use “empirical” components (depending on estimates of F), or even using
ideas from the proof of the DKW inequality [24] to avoid the union bounds over the points of each
interval in the decomposition. We leave these directions for future work.

To conclude, since in practice, a DSP can launch campaigns through multiple auctions, an interesting
question is whether the current analysis could be extended to the case of A auctions where a play at
time t is (na,t)a∈[A] where P

a∈[A] na,t = N and the reward is P

a∈[A] ra(na,t) with ra determined
by integers pa, na,t and Fa in the same way that r depends on p, nt and F. How to explore each
auction in parallel in an efficient manner and how to handle the case where some auctions must be
assigned zero players are then the main questions to solve.

Acknowledgments

Dorian Baudry thanks the support of the French National Research Agency: ANR-19-CHIA-02
SCAI, ANR-22-SRSE-0009 Ocean, and ANR-23-CE23-0002 Doom; and of the European Research
Council (GTIR project).

Vianney Perchet’s research was supported in part by the French National Research Agency (ANR)
in the framework of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant
DOOM ANR-23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.

References

[1] Y. Abbasi-yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic
bandits.
In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, edi-
tors, Advances in Neural Information Processing Systems, volume 24. Curran Associates,
Inc., 2011. URL https://proceedings.neurips.cc/paper_files/paper/2011/file/
e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf. 2

[2] S. Agrawal and N. Goyal. Analysis of thompson sampling for the multi-armed bandit problem.
In Proceedings of the 25th Annual Conference on Learning Theory, 2012. 2

10


---Page Break---
[3] P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit prob-
lem. Machine Learning, 47:235–256, 2002. URL https://api.semanticscholar.org/
CorpusID:207609497. 2, 3, 8, 33, 36

[4] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit
problem. SIAM Journal on Computing, 32(1):48–77, 2002. doi: 10.1137/S0097539701398375.
URL https://doi.org/10.1137/S0097539701398375. 2, 3, 36

[5] M. Babaioff, Y. Sharma, and A. Slivkins. Characterizing truthful multi-armed bandit mech-
anisms: extended abstract.
In Proceedings of the 10th ACM Conference on Electronic
Commerce, EC ’09, page 79–88, New York, NY, USA, 2009. Association for Comput-
ing Machinery.
ISBN 9781605584584.
doi: 10.1145/1566374.1566386.
URL https:
//doi.org/10.1145/1566374.1566386. 2

[6] D. Baudry, N. Merlis, M. B. Molina, H. Richard, and V. Perchet. Multi-armed bandits with
guaranteed revenue per arm. In International Conference on Artificial Intelligence and Statistics,
2-4 May 2024, Palau de Congressos, Valencia, Spain, Proceedings of Machine Learning
Research, 2024. 28

[7] O. Cappé, A. Garivier, O.-A. Maillard, R. Munos, G. Stoltz, et al. Kullback–leibler upper
confidence bounds for optimal sequential allocation. The Annals of Statistics, 41(3):1516–1541,
2013. 2

[8] P. Chatterjee and I. Sharfi.
Bidding and auction services in the privacy sand-
box.
https://github.com/privacysandbox/protected-auction-services-docs/
blob/main/bidding_auction_services_api.md, 2024. 1, 2

[9] R. Combes and A. Proutière.
Unimodal bandits: Regret lower bounds and optimal al-
gorithms.
ArXiv, abs/1405.5096, 2014.
URL https://api.semanticscholar.org/
CorpusID:15210470. 2, 3, 6, 36

[10] H. David and H. Nagaraja. Order Statistics. Wiley Series in Probability and Statistics. Wiley,
2004. ISBN 9780471654018. URL https://books.google.fr/books?id=bdhzFXg6xFkC.
4, 13

[11] Y. Deng, N. Golrezaei, P. Jaillet, J. C. N. Liang, and V. S. Mirrokni.
Multi-channel au-
tobidding with budget and roi constraints.
ArXiv, abs/2302.01523, 2023.
URL https:
//api.semanticscholar.org/CorpusID:256598278. 2

[12] N. R. Devanur and S. M. Kakade. The price of truthfulness for pay-per-click auctions. In
Proceedings of the 10th ACM Conference on Electronic Commerce, EC ’09, page 99–106, New
York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605584584. doi:
10.1145/1566374.1566388. URL https://doi.org/10.1145/1566374.1566388. 2

[13] E. Even-Dar, S. Mannor, Y. Mansour, and S. Mahadevan. Action elimination and stopping
conditions for the multi-armed bandit and reinforcement learning problems. Journal of machine
learning research, 7(6), 2006. 7

[14] L. Ha. Online advertising research in advertising journals: A review. Journal of Current Issues
and Research in Advertising, 30, 05 2012. doi: 10.1080/10641734.2008.10505236. 1

[15] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau,
E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk,
M. Brett, A. Haldane, J. F. del Río, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard,
T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant. Array programming with
NumPy. Nature, 585(7825):357–362, Sept. 2020. doi: 10.1038/s41586-020-2649-2. URL
https://doi.org/10.1038/s41586-020-2649-2. 35

[16] W. Hoeffding. Probability inequalities for sums of bounded random variables. The collected
works of Wassily Hoeffding, pages 409–426, 1994. 5, 18

[17] J. Honda and A. Takemura. Non-asymptotic analysis of a new bandit algorithm for semi-
bounded rewards. Journal of Machine Learning Research, 16:3721–3756, 2015. 2

11


---Page Break---
[18] J. D. Hunter. Matplotlib: A 2d graphics environment. Computing in Science & Engineering, 9
(3):90–95, 2007. doi: 10.1109/MCSE.2007.55. 35

[19] M. Jones. The complementary beta distribution. Journal of Statistical Planning and Inference,
104(2):329–337, 2002. ISSN 0378-3758. doi: https://doi.org/10.1016/S0378-3758(01)00260-9.
URL https://www.sciencedirect.com/science/article/pii/S0378375801002609.
4

[20] V. Krishna. Auction Theory. Academic Press, 2009. 1, 2

[21] T. Lattimore and C. Szepesvari. Bandit algorithms. 2017. URL https://tor-lattimore.
com/downloads/book/book.pdf. 2

[22] J. Levin. Auction theory. Manuscript available at www. stanford. edu/jdlevin/Econ, 20286,
2004. 2

[23] S. Magureanu, R. Combes, and A. Proutiere. Lipschitz bandits: Regret lower bound and optimal
algorithms. In M. F. Balcan, V. Feldman, and C. Szepesvári, editors, Proceedings of The 27th
Conference on Learning Theory, volume 35 of Proceedings of Machine Learning Research,
pages 975–999, Barcelona, Spain, 13–15 Jun 2014. 2

[24] P. Massart. The tight constant in the dvoretzky-kiefer-wolfowitz inequality. The Annals
of Probability, 18(3):1269–1283, 1990. ISSN 00911798. URL http://www.jstor.org/
stable/2244426. 10

[25] S. Muthukrishnan. Ad exchanges: Research issues. In Workshop on Internet and Network
Economics, 2009. URL https://api.semanticscholar.org/CorpusID:10046036. 1

[26] H. Nazerzadeh, A. Saberi, and R. Vohra. Dynamic cost-per-action mechanisms and applications
to online advertising. WWW ’08, page 179–188, New York, NY, USA, 2008. Association
for Computing Machinery. ISBN 9781605580852. doi: 10.1145/1367497.1367522. URL
https://doi.org/10.1145/1367497.1367522. 2

[27] T. Nedelec, C. Calauzènes, N. El Karoui, and V. Perchet. 2022. 2

[28] S. Paladino, F. Trovò, M. Restelli, and N. Gatti. Unimodal thompson sampling for graph-
structured arms. In S. Singh and S. Markovitch, editors, Proceedings of the Thirty-First AAAI
Conference on Artificial Intelligence, February 4-9, 2017, San Francisco, California, USA.
AAAI Press, 2017. 2

[29] H. Saber, P. M’enard, and O.-A. Maillard. Forced-exploration free strategies for unimodal ban-
dits. ArXiv, abs/2006.16569, 2020. URL https://api.semanticscholar.org/CorpusID:
220265988. 2

[30] A. S. Sayedi-Roshkhar. Real-time bidding in online display advertising. Mark. Sci., 37:553–568,
2018. URL https://api.semanticscholar.org/CorpusID:52277027. 1

[31] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski,
P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman,
N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, ˙I. Polat, Y. Feng,
E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero,
C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, and SciPy 1.0
Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature
Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2. 35

[32] J. Weed, V. Perchet, and P. Rigollet. Online learning in repeated auctions. In V. Feldman,
A. Rakhlin, and O. Shamir, editors, 29th Annual Conference on Learning Theory, volume 49
of Proceedings of Machine Learning Research, pages 1562–1583, Columbia University, New
York, New York, USA, 23–26 Jun 2016. PMLR. URL https://proceedings.mlr.press/
v49/weed16.html. 2

[33] Y. Yuan, J. Li, and R. Qin. A survey on real time bidding advertising. Proceedings of 2014
IEEE International Conference on Service Operations and Logistics, and Informatics, SOLI
2014, pages 418–423, 11 2014. doi: 10.1109/SOLI.2014.6960761. 1

12


---Page Break---
A
Properties of the expected reward function

In this appendix we prove the results presented in Section 2.1 of the paper, and discuss the shape of
the expected reward.

A.1
Proof of Lemma 1

Lemma 1. The expected reward function defined in Equation (1) satisfies,

n ∈[N] 7→r(n) = n
Z 1

0
F p+n−1(x) −F p+n(x)dx
(3)

Proof. Given v = (vi)i∈[n+p] ∼F × · · · × F, we have

r(n) = E

(v(1) −v(2))1

argmax
i∈[n+p]
vi ∈[n]


(i)
= E

(v(1) −v(2))

E

1

argmax
i∈[n+p]
vi ∈[n]


(ii)
=

E

v(1)


−E

v(2)


×
n
n + p

=
n
n + p ×
Z 1

0
P(v(1) > x) −P(v(2) > x)dx

=
n
n + p ×
Z 1

0
P(v(2) ≤x) −P(v(1) ≤x)dx

(iii)
=
n
n + p ×
Z 1

0
((n + p)F n+p−1(x) −(n + p −1)F n+p(x) −F n+p(x))dx

= n
Z 1

0
(F n+p−1(x) −F n+p(x))dx .

The first equality is the definition of r(n) in Equation (1). Equality (i) follows by independence of
the index of the maximum and the value of the maximum and second maximum. This is itself a
consequence of the fact that the values are i.i.d.. Then equality (ii) follows since the distribution
of the index of the maximum is uniform over n + p. This is also a consequence of the fact that the
values are i.i.d. Lastly, equality (iii) follows from [10] (Equation 2.1.3) where for k ∈{1, 2}, it is
shown that

P(v(k) ≤x) =

n+p
X

i=n+p−k+1

n + p
i


(1 −F(x))n+p−iF(x)i,

and the proof is concluded by substitution.

A.2
Proof of Lemma 2

As a preliminary, we formally define the non-usual distributions considered in Lemma 2.

Truncated exponential distribution
Let a > 0 be some parameter. Then, we define a truncated
exponential distribution of parameter a as the distribution with c.d.f. F : x 7→1−e−ax

1−e−a . Hence,
F(0) = 0 and F(1) = 1, and the density of this distribution is the same as the density of the
exponential distribution with same parameter on the segment [0, 1], up to a normalization constant.

Complementary Beta distribution
Lemma 2. Let F be the cumulative distribution function of a Bernoulli, truncated exponential or
Complementary Beta distribution. Then, for any p ∈N∗, r in Equation (3) unimodal.

Proof. We consider each family of distributions separately.

13


---Page Break---
Bernoulli distributions
If F is the c.d.f. of B(q) (a Bernoulli distribution of parameter q), then
r(n) is equal to the probability that exactly one player from the coalition draws a value of 1, and every
other player draw a value of 0. Hence, we obtain that r(n) = nq(1 −q)n+p−1, which is trivially
unimodal and maximized in n⋆=
−1
log(1−q) ∨1, regardless of the size of the competition.

Truncated exponential distributions
Let a > 0 be the parameter of the distribution. Let Q(x)
be the inverse function of F (the quantile function), defined by Q(x) = 1

a log

1
1−x(1−e−a)

=

1
a
P+∞
k=1
xk(1−e−a)k

k
.

Let’s denote by q(x) the derivative of Q(x), denoted by q(x) = P+∞
k=0 λkxk where λk = 1

a(1−e−a)k.
Introducing theses functions allows us to rewrite r(n) as follows,

r(n) = n
Z 1

0
F(v)p+n−1(1 −F(v))dv

= n
Z 1

0
xp+n−1(1 −x)q(x)dx
using F(v) = x

= n
Z 1

0
xp+n−1(1 −x)

 +∞
X

k=0
λkxk
!

dx

=

+∞
X

k=0
λk


n
p + n + k −
n
p + n + k + 1



=
n
n + pλ0 + n

+∞
X

j=1

1
n + p + j (λj −λj−1)

= λ0(1 −
p
n + p) +

+∞
X

j=1
(1 −
p + j
n + p + j )(λj −λj−1)

= λ0

−
p
n + p +

+∞
X

j=1

λj−1 −λj

λ0
|
{z
}
θj

p + j
n + p + j



where the last inequality follows since limj→∞λj = 0. Remark that θj ≥0 since λj is decreasing
and P∞
j=1 θj = 1.

The derivative of r(n) is given by,

r′(n) = λ0

p
(n + p)2 −

+∞
X

j=1
θj
p + j
(n + p + j)2



= λ0

Θp(n) −

∞
X

j=1
θjΘp+j(n)


= λ0Θp(n)

1 −

∞
X

j=1
θjΓp,p+j(n)

where Γp,p+j(n) = Θp+j(n)

Θp(n)

the functions Γp,p+j(n) are non-decreasing hence it is the same for their convex combination. As
sign(r′(n)) = sign(1 −P∞
j=1 θjΓp,p+j(n)) it follows that sign(r′(n)) is decreasing meaning r(n)
is unimodal.

Complementary beta distributions
Using the same change of variable as in the previous proof
(Y = F(X)), we express the reward as follows,

r(n) = n × EY ∼Q

Y n+p−1(1 −Y )

,

where Q denotes the quantile function associated with c.d.f F (i.e. F −1). By definition of F, Y
follows a Beta distribution of parameters (a, b). We can thus compute the expected reward by using

14


---Page Break---
the explicit formula for moments of the Beta distribution,

EX∼B(a,b)[Xn+p−1 −Xn+p] =

n+p−2
Y

k=0

a + k
a + b + k −

n+p−1
Y

k=0

a + k
a + b + k

=

n+p−2
Y

k=0

a + k
a + b + k ×

1 −
a + n + p −1
a + b + n + p −1



=

n+p−2
Y

k=0

a + k
a + b + k ×
b
a + b + n + p −1 .

Thanks to this expression, we prove the unimodality by analyzing the ratio r(n+1)

r(n) , that we first write
as

r(n + 1)

r(n)
= n + 1

n
×
a + n + p −1
a + b + n + p −1 × a + b + n + p −1

a + b + n + p

= n + 1

n
× a + n + p −1

a + b + n + p ,

and then obtain that this ratio is larger than 1 if and only if

(n + 1)(a + n + p −1) ≥n(a + b + n + p) ⇐⇒n(a + n + p) + a + p −1 ≥n(a + n + p) + bn

⇐⇒n ≥a + p −1

b
,

which concludes the proof by showing the unimodality and expressing the value of the critical point.

The proof of Lemma 2 highlights that the unimodality assumption is satisfied as soon as the quantile
function, expressed as a power series, has its coefficients that slowly decrease (indeed, the k-th
coefficient just needs to be smaller than 1 −1

k times the k −1-th one).

Similarly, the second proof technique highlights (up to standard algebraic manipulations) that
unimodularity is guaranteed as soon as the function n 7→1 −E[Xn+p−1]/E[Xn+p−2] is log-
concave.

A.3
Additional discussion on the unimodality of r

In this section, we plot the shape of r(n) for some additional families of distribution that we conjecture
to be unimodal from the plots.

Beta distribution
The following figure, illustrate the unimodal shape of r(n) for different parame-
ters for the Beta distribution and p.

15


---Page Break---
Figure 1: Shape of r(n) when F is Beta

Kumaraswamy distribution
The cumulative distribution is defined by F(x) = 1 −(1 −xa)b for
some parameters (a, b) (we use the notation K(a, b)). The following figure, illustrate the unimodal
shape of r(n) for different parameters of K(a, b) and p.

Figure 2: Shape of r(n) when F is Kumaraswamy distribution

We now provide and discuss an example where Assumption 2 is not satisfied.

A.4
An example of distribution with non-unimodal rewards

Let us consider a discrete distribution supported on {0, 0.5, 1}. The counter-example emerges from
putting all the probability mass in 0.5: let us consider a small ϵ > 0, identify F with {ϵ, 1 −ϵ, 1} and
assume that there is no competition (p = 0). Then, we can verify with (3) that

r(n) = n

2 (ϵn−1 + (1 −ϵ)n−1 −(ϵn + (1 −ϵ)n))

= n

2 (ϵn−1(1 −ϵ) + ϵ(1 −ϵ)n−1)

Consider ϵ = 0.15, we get up to a precision 0.001 the values:
(r(n))7
n=1 = (0.5, 0.255, 0.191, 0.190, 0.197, 0.200, 0.198)
showing that r(n) is not unimodal in this case.

16


---Page Break---
B
Concentration bounds on simple reward estimates

B.1
Auxiliary results

Before presenting the proof of the theorem, we present two auxiliary results that are essential to its
development.

B.1.1
Riemann sum approximation of the expected reward

The first result consists in upper bounding the deviation of a Riemann sum approximation of r(n) (for
some n ∈[N]) with respect to its exact integral formulation. This result is also of practical interest,
since it can prevent computing exact integrals at each step of the algorithms without altering their
theoretical guarantees with an appropriate tuning of the approximation error.

Lemma 4 (Riemann sum approximation of r(n)). Let n ∈[N], D ∈N, and define the grid
(xs)s∈{0,...,D−1} =

0, 1

D, . . . , D−1

D
	
. Then, the expected reward approximation

er(n) = n × 1

D

D−1
X

s=0


F(xs)n+p−1 −F(xs)n+p	

satisfies
|r(n) −er(n)| ≤n

D .

Proof. For any j ∈N we consider

Sj = 1

D

D−1
X

s=0
F j(xs)
as an approximation of
Ij =
Z 1

0
F j(x)dx .

We recall that since F j is a c.d.f., it is monotone, increasing and satisfies F j(0) = 0 and F j(1) = 1.
This means that for any s ∈{0, . . . , D −1}, it holds that ∀x ∈[xs, xs+1], F j(xs) ≤F j(x) ≤
F j(xs+1). The linearity of the integral first provides that

Ij =
Z 1

0
F j(x)dx =

D−1
X

s=0

Z
s+1

D

s
D
F j(x)dx

Then, using this decomposition and the monotony of F j we obtain that

Sj ≤1

D

D−1
X

s=0
F j  s

D


≤
Z 1

0
F j(x)dx ≤1

D

D−1
X

s=0
F j
s + 1

D



≤1

D

D−1
X

s=0
F j  s

D


+ 1

D

D−1
X

k=0


F j
s + 1

D


−F j  s

D



≤1

D

D−1
X

s=0
F j  s

D


+ F j(1) −F j(0)

D

= Sj + 1

D .

Therefore, we obtained that for any j ∈N it holds that Sj ≤Ij ≤Sj + 1

D. We conclude by using
this result after splitting the reward as a difference of two integrals that can be expressed in this form,
respectively with j = n + p −1 and j = n + p.

B.1.2
Chernoff bounds for Bernoulli random variables

In the following lemma, we summarize the different concentration bounds that we use in the proof of
Theorem 1.

17


---Page Break---
Lemma 5 (Mutltiplicative Chernoff bounds). Let bµm be the empirical average of m i.i.d. Bernoulli
random variables X1, . . . , Xm with expectation µ. Then, for any δ > 0, each of the following bounds
holds with probability at least 1 −δ,










|bµm −µ| ≤√µ ×
q

3 log( 2

δ)
m
if µ ∈I0 :=
h
3 log(2/δ)

m
, 1
i
,

µm ≤6 log(2/δ)

m
if µ ∈I1 :=

δ
m, 3 log(2/δ)

m

,

µm = 0
if µ ∈I2 :=

0, δ

m

.

Proof. We first tackle the case µ ∈I2, where the bound is obtained by remarking that P(µm > 0) ≤
P(∃i ∈[m] : Xi = 1) ≤mµ ≤δ. The two other cases are obtained by using the multiplicative form
of the well-known Chernoff bounds [16], that provide that

∀γ > 0,
P(bµm ≥(1 + γ)µ) ≤e−m γ2µ

2+γ
, and

∀γ ∈[0, 1], P(bµm ≤(1 −γ)µ) ≤e−m γ2µ

2
.

When considering γ ≤1 we can further write that

P(|bµm −µ| ≥γµ) ≤2e−m γ2µ

3
.

On the other hand, for γ ≥1 the bound for the lower deviation is trivially 0 while the bound for the
upper deviation can be written as follows,

γ ≥1 ⇒P(bµm ≥(1 + γ)µ) ≤e−m γ2µ

2+γ ≤e−m γ2µ

3γ = e−mµ γ

3 .

Hence, the case separation between the intervals I0 and I1 simply consist in identifying the value µ
for which a probability 1 −δ can be obtained by setting an appropriate γ ∈[0, 1] or for γ > 1 in the

above inequalities. More precisely, inverting the first bound provides γ = µ
−1

2
q

3 log( 2

δ)
m
, which is

valid only if µ
−1

2
q

3 log( 2

δ)
m
≤1 ⇒µ ≥
3 log( 2

δ)
m
. This leads to the first confidence interval when

µ ∈I0. The same procedure for µ ∈I1 leads to γ = µ−1 3 log( 2

δ)
m
, which provides the result stated in
the lemma. This concludes the proof.

B.2
Proof of Theorem 1

Theorem 1 (Concentration of simple estimates). Consider any n ∈[N] and k ∈V(n). Let brk(n)
be defined according to (5) from mk samples collected by k. Then, there exists some constants βk,n
(depending on n, k, p) and ξk,n,F (additionally depending on F) such that, with probability 1 −δ,

|brk(n) −r(n)| ≤βk,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ n × ξk,n,F




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

.
(7)

Furthermore, the constants admit universal upper bounds for any n, k, p, F. For instance if mk ≥4
it holds that βk,n ≤33 and γk,n,F ≤100.

Proof. We build the proof from the Riemann sum approximation of the reward presented in Lemma 4
and the Chernoff bounds presented in Lemma 5. Defining some parameter D ∈N that will be fixed
later, we use the first result to consider the following approximation of the empirical reward estimate
brk(n) by

erk(n) = 1

D

D−1
X

s=0


bFk+p,mk
 s

D

 n+p−1

k+p
−bFk+p,mk
 s

D

 n+p

k+p 
.

Thanks to Lemma 4, we know that brk(n) ∈

erk(n) −n

D, erk(n) + n

D

. For the rest of proof, we

introduce the notation F j
s = F
  s

D
j for any j ∈[N], bF k+p
s
= bFk+p,mk
  s

D

, and bFs,k,n =

18


---Page Break---
bFk+p,mk
  s

D
 n+p

k+p , so that

erk(n) := 1

D

D−1
X

s=0


bFs,k,n−1 −bFs,k,n

,

that we want to relate with

er(n) := 1

D

D−1
X

s=0

 
F n+p−1
s
−F n+p
s

.

We use that each variable bFs,k,n can be expressed as the expectation of mk i.i.d. Bernoulli random
variables of expectation F k+p
s
, since bFk+p,mk =
1
mk
Pmk
j=1 1{Xk,j ≤x}. Hence, we can use the
confidence intervals providing by Lemma 5, according to the value of F k+p
s
. We define two critical
values, corresponding to the switch between the different intervals I0, I1, I2 in the lemma, and their
closest upper point in the discretization grid. The first is

x0,k = F −1
  δ

mk


1
k+p !

,
and s0,k = ⌈Dx0,k⌉,

and we recall that below x0,k it holds that bFk,i,n−1 = 0 with probability larger than 1 −δ. Then, we
define

x1,k := F −1



1 ∧

 

4log
  2

δ


mk

!
1
k+p 

, and s1,k := ⌈Dx1,k⌉.

We remark that we use a multiplicative factor 4 inside of F −1, while Lemma 5 might suggest to use
3. We do that for technical reasons, that we will motivate at one stage of the proof. These terms
depend both on the sample size mk and the confidence level δ, but we omit them in the notation
for simplicity. Then, for fixed values of these constants we decompose the estimator between the
intervals I0 = {s1,k, . . . , D −1}, I1 = {s0,k, . . . , s1,k −1}, and I2 = {0, . . . , s0,k −1}. Note that
for the second interval to be non-empty it must hold that s0,k ≤s1,k −1, that we assume in the
following, otherwise we can just remove this interval from the analysis.

For s ≥s1,k, Lemma 5 guarantees that with probability larger than 1 −δ it holds that

bF k+p
s
∈

(1 −γs)F k+p
s
, (1 + γs)F k+p
s

for γs = F
−k+p

2
s

s

3log
  2

δ


mk
,

while for k ≤s1,k −1 we can use one of the two other bounds provided in the lemma. Using a union
bound, all the confidence intervals hold simultaneously for the points in the sum and in x1,k with
probability larger than 1 −(D + 1)δ, which defines a “good” event

G =

(

∀k ∈I2, bF k+p
s
= 0, ∀k ∈I1, bF k+p
s
≤8log
  2

δ


mk
,

∀k ∈I0, bF k+p
s
∈

(1 −γs)F k+p
s
, (1 + γs)F k+p
s
 o
.
(9)

For the rest of the analysis, we assume that G holds. In particular, in that context there exists D −s1,k
constants (zs)s∈{s1,k,...,D−1} such that ∀s ∈I0, bF k+p
s
= (1 + zs)F k+p
s
and zs ∈[−γs, γs]. We now
upper and lower bound brk(n) using these constants, first writing that under G it holds that

erk(n) = 1

D

D−1
X

s=0


bFs,k,n−1 −bFs,k,n


= 1

D

D−1
X

s=s1,k


bFs,k,n−1 −bFs,k,n

+ 1

D

s1,k−1
X

s=0


bFs,k,n−1 −bFs,k,n


= 1

D

D−1
X

s=s1,k


(1 + zs)

n+p−1

k+p F n+p−1
s
−(1 + zs)

n+p
k+p F n+p
s

+ 1

D

s1,k−1
X

s=s0,k


bFs,k,n−1 −bFs,k,n

,

19


---Page Break---
where we used that all the terms are zero for indices smaller than s0,k. We can thus express brk(n) as
follows,
brk(n) = er(n) + nE0 + nE1 ,
with

E0 := 1

D

D−1
X

s=s1,k
((1 + zs)

n+p−1

k+p
−1)F n+p−1
s
−1

D

D−1
X

s=s1,k
((1 + zs)

n+p
k+p −1)F n+p
s
,
and

E1 := 1

D

s1,k−1
X

s=s0,k


bFs,k,n−1 −bFs,k,n

−

s1,k−1
X

s=0

 
F n+p−1
s
−F n+p
s

,

so we can upper and lower bound bri(n) by upper and lower bounding E0 and E1 separately.

Bounding the individual terms of E0
For any k ≥s1,k we consider the term

E0,s := ((1 + zs)

n+p−1

k+p
−1)F n+p−1
s
−((1 + zs)

n+p
k+p −1)F n+p
s
.

We first re-arrange it in a more convenient way, remarking that

F n+p−1
s
= F n+p−1
s
(Fs + (1 −Fs)) = F n+p
s
+ F n+p−1
s
(1 −Fs).

Using this result, we obtain that

E0,s := ((1 + zs)

n+p−1

k+p
−1)F n+p−1
s
−((1 + zs)

n+p
k+p −1)F n+p
s

= ((1 + zs)

n+p−1

k+p
−1)F n+p−1
s
(Fs + (1 −Fs)) −((1 + zs)

n+p
k+p −1)F n+p
s

= F n+p
s
((1 + zs)

n+p−1

k+p
−1) −(1 + zs)

n+p
k+p + 1) + F n+p−1
s
(1 −Fs)((1 + zs)

n+p−1

k+p
−1) ,

which simplifies to

E0,s = F n+p
s
((1 + zs)

n+p−1

k+p
−(1 + zs)

n+p
k+p )
|
{z
}
E−
0,s

+ F n+p−1
s
(1 −Fs)((1 + zs)

n+p−1

k+p
−1)
|
{z
}
E+
0,s

(10)

We remark that these two terms have opposite sign, E+
0,s having the same sign as zs. We first upper
bound E0,s, starting with the case zs > 0, for which it holds that

zs ≥0 ⇒E0,s ≤E+
0,s ≤((1 + γs)

n+p−1

k+p
−1)
|
{z
}
cs

F n+p−1
s
(1 −Fs) .

The constant ck is explicit from the definition of γs, and the bound holds for any i ∈[N + p] without
restriction. However, if we only consider the case n+p−1

k+p
≤2 then we can further write that

cs ≤(1 + γs)2 −1 ≤2γs + γ2
s ≤3γs ,

since γs ≤1 for the values of s considered. Then, for zs ≤0 we use that

zs ≤0 ⇒E0,s ≤E−
0,s ≤F n+p
s
((1 + zs)

n+p−1

k+p
−(1 + zs)

n+p
k+p )

= F n+p
s
(1 + zs)

n+p−1

k+p

1 −(1 + zs)
1
k+p

.

Using the notation ys = −zs for convenience, we upper bound the last multiplicative term as follows,

(1 −ys)
1
k+p = e

log(1−ys)

k+p
≥1 + log(1 −ys)

k + p

= 1 −
1
k + p log

1 +
ys
1 −ys



≥1 −
1
k + p ×
ys
1 −ys
,

20


---Page Break---
which leads to

ys := −zs ≥0 ⇒E0,s ≤E−
0,s ≤F n+p
s
(1 −ys)

n+p−1

k+p
ys
(k + p)(1 −ys)

= F n+p
s
(1 −ys)

n+p−1

k+p
−1
ys
k + p .

We now remark that when k + p ≤n + p −1 then the bound simply becomes E−
0,s ≤F n+p
s
×
γs
k+p.
However, when k+p > n+p−1 the upper bound is diverging when ys gets close to 1. Since the upper
bound is increasing in γs, and using that n + p −1 ≥2

3(k + p) we obtain that E−
0,s ≤F n+p
s
γs
i(1−γs)
1
3 .

This is the motivation for calibrating the threshold s1,k so that k ≥s1,k ⇒(1 −γs)
1
3 ≥1

2, which is
done by tuning s1,k so that γs1,k ≤7

8 (hence the multiplicative 4 inside of F −1).

We thus obtain that
zs ≤0, k ≥s1,k ⇒E0,s ≤2F n+p
s
γs
k + p ,

which finally leads to

E0,s ≤3γsF n+p−1
s
(1 −Fs)
|
{z
}
if zs≥0

∨2 F n+p
s
γs
k + p
|
{z
}

if zs≤0

.
(11)

We now proceed to lower bound E0,s, using again Equation(10). The proof is similar to the proof of
the upper bound, for the case zs ≥0 we can write that

zs ≥0 ⇒−E0,s ≤−E−
0,s ≤F n+p
s
(1 + zs)

n+p−1

k+p ((1 + zs)
1
k+p −1)

≤F n+p(1 + γs)

n+p−1

k+p
γs
k + p

≤2

n+p−1

k+p F n+p
γs
k + p ,

using the concavity of x 7→(1 + x)
1
k+p and that γs ≤1. Then, for the case zs ≤0 we use that

zs ≤0 ⇒−E0,s ≤−E+
0,s ≤F n+p−1
s
(1 −Fs)

1 −(1 + zs)

n+p−1

k+p


≤F n+p−1
s
(1 −Fs)

1 −(1 −γs)

n+p−1

k+p

.

If n+p−1

k+p
≤2 we furthermore obtain that (1 −γs)

n+p−1

k+p
≥(1 −γs)2 ≥1 −2γs, so

zs ≤0 ⇒−E+
0,s ≤2γsF n+p−1
s
(1 −Fs) .

Combining these results, we can lower bound E0,s as follows,

E0,s ≥−

(
2

n+p−1

k+p γs
k + p
F n+p
s
∨2γsF n+p−1
s
(1 −Fs)

)

,
(12)

where the terms involved in this lower bound are analogous to the terms used in the upper bound up
to some multiplicative constants.

Summary: bounds on E0
We start with the lower bound. Using Equation (12), we obtain that

nE0 ≥−n

D

D−1
X

s=s1,k

(
2

n+p−1

k+p γs
k + p
F n+p
s
∨2γsF n+p−1
s
(1 −Fs)

)

≥−

s

3 log
  2

δ


mk
×





n
D

D−1
X

s=s1,k
2

n+p−1

k+p F
n+p−k+p

2
s

k + p
+ n

D

D−1
X

s=s1,k
2F
n+p−1−k+p

2
s
(1 −Fs)






21


---Page Break---
The first sum can be trivially upper bounded by 2

n+p−1

k+p
n
k+p, which cannot be refined without more
restrictive assumptions on F. For the second term, we use that k + p ≤3

2(n + p) to exhibit the
reward function associated with a number of players n′ := n −n(k+p)

2(n+p) ≥n

4 and a competition of

size p′ := p −p(k+p)

2(n+p) ≥p

4.

n
D

D−1
X

s=s1,k
F
n+p−1−k+p

2
s
(1 −Fs) = n × 1

D

D−1
X

s=s1,k
F n′+p′−1
s
(1 −Fs)

≤n ×
Z 1

0
F(x)
n+p−1

4
(1 −F(x))dx + n

D

= n

n′ × n′
Z 1

0
F(x)n′+p′−1(1 −F(x))dx + n

D

≤
n
n′ + p′ −1 + n

D =
n
n + p −k+p

2
+ n

D ,

where we used that the reward is smaller than the probability that the coalition wins the auction,
which is easily generalized even if i/2 is not integer.

We thus conclude the proof of the lower bound by writing that

nE0 ≥−4

n
2(n + p) −(k + p) + n

2D


+ 2

n+p−1

k+p
−2 ×
n
k + p


×

s

3 log
  2

δ


mk
,
(13)

where the worst scaling for the left-hand term in the maximum is attained in k + p = 3 n+p

2
and
provide 2
n
n+p, while for the right-hand term it is achieved in k + p = n+p−1

2
and provides 2
n
n+p−1.

As we already discussed, the upper bound can be expressed very similarly, remarking that the bound
involving the terms γs/i have to be divided by two, and the other term have to be multiplied by 3/2.
We hence directly obtain that

nE0 ≤6

n
2(n + p) −(k + p) + n

2D


+
n
3(k + p)


×

s

3 log
  2

δ


mk
.
(14)

Bounds on E1
We start by upper bounding the second sum by 0. Under G we hence obtain that

nE1 ≤n

D

X

k∈I1
( bF k+p
s
)

n+p−1

k+p

≤n

D

X

k∈I1


8log(2/δ)

mk

 n+p−1

k+p

≤n

x1,k −x0,k + 1

D

 
8log(2/δ)

mk

 n+p−1

k+p
,

which has a worst possible power of 2/3 when mk ≥8 log(2/δ), corresponding to k + p =
3
2(n + p −1). Replacing x1,k by its expression, we further obtain that

nE1 ≤n

8log(2/δ)

mk

 n+p−1

k+p
×




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

−F −1
  δ

mk


1
k+p !

+ 1

D




.

(15)
For the lower bound on nE1
1, we apply the exact same steps, remarking that the constant 8 at the
very first step can be replaced by 4, since we now use the exact value of F k+p in the upper bound.
Furthermore, we have to remove the term x0,k. We finally obtain

nE1 ≥−n

4log(2/δ)

mk

 n+p−1

k+p
×




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

+ 1

D




.
(16)

22


---Page Break---
Summary: bounds on brk(n)
We conclude this proof by summarizing the results, and exhibiting
the constants introduced in the theorem. First, by combining (13) and (16) we obtain the following
lower bound,

brk(n) ≥r(n) −n

D −n

4log(2/δ)

mk

 n+p−1

k+p
×




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

+ 1

D






−4

n
2(n + p) −(k + p) + n

2D


+ 2

n+p−1

k+p
−2 ×
n
k + p


×

s

3 log
  2

δ


mk
.

Then, by combining (14) and (15) we obtain the following upper bound,

brk(n) ≤r(n) + n

D + 6

n
2(n + p) −(k + p) + n

2D


+
n
3(k + p)


×

s

3 log
  2

δ


mk

+ n

8log(2/δ)

mk

 n+p−1

k+p
×




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

−F −1
  δ

mk


1
k+p !

+ 1

D




.

As a final step, we recall that in this proof 1 −δ is the confidence level of the point estimate in each
of the D points (xs)s∈{0,...,D−1} and x1,k. Hence, to obtain a confidence 1 −δ on the full estimate
brk(n) we need to multiply δ by (D + 1) in the bounds presented above. As a final step, we choose
D + 1 = ⌈n√m⌉, so that the term n

D ≤
1
√mk becomes a second order term in the bounds.

After replacing δ and D by the appropriate values, we hence obtain that

brk(n) ≥r(n) −
1
√mk
−β−
k,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
−n × ξ−
k,n,F




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

. (17)

with

β−
k,n = 4
√

3

n
2(n + p) −(k + p) +
n
2(⌈n√mk⌉−1)


+ 2

n+p−1

k+p
−2 ×
n
k + p


, and

ξ−
k,n,F = 4

n+p−1

k+p




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

+
1
⌈n√mk⌉−1




.
(18)

Symmetrically, we obtain that

brk(n) ≤r(n) +
1
√mk
+ β+
k,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ n × ξ+
k,n,F




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

. (19)

with

β+
k,n = 6
√

3

n
2(n + p) −(k + p) +
n
2(⌈n√mk⌉−1)


+
n
3(k + p)


, and

ξ+
k,n,F = 8

n+p−1

k+p




F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

−F −1
  δ

mk


1
k+p !

+
1
⌈n√mk⌉−1




.

(20)

23


---Page Break---
Hence, we obtain the statement of (7) by choosing βk,n = β−
k,n ∨β−
k,n and ξk,n,F = ξ+
k,n,F ∨ξ−
k,n,F .
Furthermore, it is clear from their expression and the constraint k + p ∈
 n+p

2 , 3 n+p

2

that these two
constants are bounded by by absolute constants. Their expression provided in the theorem comes
from choosing the worst admissible value of k for each of their components. This concludes the
proof of the theorem.

Remark 2 (Improved constants for practical implementations). We can further improve the constants
of the bounds according to the position of i with respect to n + p −1.

• In Equation (11) (upper bound): the constant 3 can be improved to 1 if i ≥n + p −1, while
the constant 2 can be improved to n+p−1

k+p
if i ≤n + p −1.

• In Equation (12) (lower bound): the constant 2 on the right-hand side can be improved to 1
if n + p −1 ≤i.

These improved constants translate easily to the upper and lower bounds presented in (13) and (14).

B.3
Proof of Lemma 3

In this part, we prove the tighter confidence bounds, assuming that the quantile function of the value
distribution is Lipschitz.

Lemma 3 (Improved bound for Lipschitz quantile function). Assume that k ∈V(n) and F −1 is
L-Lipschitz, then there exists an absolute constant ξ such that with probability 1 −δ it holds that

|brk(n) −r(n)| ≤βk,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ ξL log
4⌈n√mk⌉

δ

 


log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

(8)

Proof. As the result indicates, the improvement comes from providing finer upper and lower bound
on the term nE1 in the proof of Theorem 1. We can start the refined analysis from Equations (16) and
(15).

Let us consider first the upper bound. In that case, the main ingredient comes from refining the
upper bound of the difference x1,k −x0,k. To do that, we first provide an upper bound on 1 ∧

4 log( 2

δ)
mk


1
k+p
−

δ
mk


1
k+p . We recall that, as in the previous proof, this δ should be multiplied by

⌈n√mk⌉at the end of the computation. We omit this term for now for simplicity. We consider a first
case where the first term is equal to 1, which leads to

1 −
 δ

mk


1
k+p
= 1 −e−
1
k+p log(
mk

δ )

≤
1
k + p log
mk

δ



≤
1
k + p log
4 log(2/δ)

δ



=
1
k + p ×

log
4

δ


+ log log
2

δ


.

For the alternative case we obtain a similar result,
 
4 log
  2

δ


mk

!
1
k+p
−
 δ

mk


1
k+p
=

 
4 log
  2

δ


mk

!
1
k+p  

1 −

δ
4 log(2/δ)


1
k+p !

≤

 
4 log
  2

δ


mk

!
1
k+p
×
1
k + p ×

log
4

δ


+ log log
2

δ


,

24


---Page Break---
so the two upper bounds simplify to



1 ∨

 
4 log
  2

δ


mk

!
1
k+p 


×
1
k + p ×

log
4

δ


+ log log
2

δ



Next, we use the assumption that F −1 is L-Lipschitz to obtain that

n(x1,k −x0,k) := n



F −1



1 ∧

 
4 log
  2

δ


mk

!
1
k+p 

−F −1
  δ

mk


1
k+p !



≤
Ln
k + p ×

log
4

δ


+ log log
2

δ



≤4 Ln

n + p × log
4

δ


.

By substituting δ by δ/(⌈n√mk⌉) we obtain that the linear dependency in n obtained with the

previous analysis is refined to a log

4⌈n√mk⌉

δ

for the upper bound, which matches the result at this
point.

We now consider the lower bound, and work on refining the upper bound of the term −nE1 in the

proof of Theorem 1. To do that, we consider a new intermediary point x′
0,k = F −1

δ

m×n
k+p
n+p−1


.

For the rest of the proof we get back to the discretized formulation of the error (with generic step
D−1), and upper bound

−nE1 ≤n

D

X

k∈I1
F n+p−1
s

|
{z
}
nE0
1

+ n

D

X

k∈I2
F n+p−1
s

|
{z
}
nE1
1

.

We remark that we can use the upper bound provided for n(x1,k−x0,k) to upper bound nE1
1, obtaining
the same result up to some multiplicative constants. Hence, it remains to upper bound E0
1, for which
additional steps are needed. However, we will simply use the exact same trick as before: we consider
another sub-interval I′
2, for which this time it holds that k ∈I′
2 ⇒F k+p
s
≤
δ

m×n
k+p
n+p−1 . Then, we

have that
nE0
1 = n

D

X

k∈I′
2
F n+p−1
s
+ n

D

X

k∈I2\I′
2
F n+p−1
s

≤n

D

X

k∈I′
2

1
n

 δ

mk

 n+p−1

k+p
+ n

D

X

k∈I2\I′
2

 δ

mk

 n+p−1

k+p

≤|I′
2|
D

 δ

mk

 n+p−1

k+p
+ n × |I2| −|I′
2|
D

 δ

mk

 n+p−1

k+p
.

Just as before, we show that |I0|−|I′
0|
D
cannot be too large if the quantile function is Lipschitz. More
precisely, we obtain that
 δ

mk


1
k+p
−

 
δ

mkn

k+p
n+p−1

!
1
k+p
=
 δ

mk


1
k+p
log(n)
n + p −1 ,

so that we can finally write that

nE0
1 ≤
 δ

mk

 n+p−1

k+p
+ n ×

  δ

mk


1
k+p
×
log(n)
n + p −1 × L + 1

D

!

×
 δ

mk

 n+p−1

k+p

≤
 δ

mk

 n+p−1

k+p
+ n ×

log(n)
n + p −1 × L +
1
n√mk


×
 δ

mk

 n+p−1

k+p
,

25


---Page Break---
which is sufficient to conclude, since the multiplicative constants to m
−n+p−1

k+p
k
are clearly dominated

by log

4⌈n√mk⌉

δ

.

B.4
Empirical UCB and LCB

The UCB and LCB in Equation (17) and Equation (19) depend explicitly on the unknown F via
ξ+
k,n,F and ξ−
k,n,F and therefore cannot be used in the implementation of GG.

Below, we give empirical UCB (bUk(n, δ)) and LCB (bLk(n, δ)) by replacing ξ+
k,n,F and ξ−
k,n,F by

empirical estimates bξ+
k,n and bξ−
k,n:

bUk(n, δ) = brk(n) +
1
√mk
+ β−
k,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
+ n × bξ−
k,n




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

(21)

bLk(n, δ) = brk(n) −
1
√mk
−β+
k,n

v
u
u
tlog

2⌈n√mk⌉

δ


mk
−n × bξ+
k,n




log

2⌈n√mk⌉

δ


mk





n+p−1

k+p

(22)

where β−
k,n and β+
k,n are defined in Equation (18) and Equation (20), and

bξ−
k,n = 4

n+p−1

k+p

(
ˆdk+p + 1
⌈n√mk⌉−1

)

,

bξ+
k,n = 8

n+p−1

k+p

(
ˆdk+p + 1
⌈n√mk⌉−1

)

for
ˆdk+p = inf{d ∈{0, . . . , ⌈n√mk⌉−1}, ˆF k+p
d
≥8log(2⌈n√mk⌉/δ)

mk
}

if the infimum exists and ˆdk+p = 1 otherwise.

It is a corollary of Theorem 1 that bUk(n) and bLk(n) are indeed high probability upper and lower
bounds of the true reward:
Corollary 1 (Explicit upper and lower bounds). It holds that

P(bLk(n, δ) ≤r(n) ≤bUk(n, δ)) ≥1 −δ

Proof. With D = ⌈n√mk⌉and x1,k = F −1
 

1 ∧

4 log
 2⌈n√mk⌉

δ


mk


1
k+p !

the good event G de-

fined in Equation (9)implies that with probability 1 −δ the following event H holds:

H =




∀k ≤⌈Dx1,k⌉, bF k+p
s
≤8
log

2⌈n√mk⌉

δ


mk




.

Under H, and since by definition ˆF k+p
ˆdk+p ≥8 log(2⌈n√mk⌉/δ)

mk
, it holds that

ˆdk+p

D
≥x1,k = F −1




1 ∧




4 log

2⌈n√mk⌉

δ


mk





1
k+p 




This implies that bξ−
k,n ≥ξ−
k,n,F and bξ+
k,n ≥ξ+
k,n,F .

Therefore, we can incorporate in the proof of Theorem 1 the fact that the good event G implies
bξ−
k,n ≥ξ−
k,n,F and bξ+
k,n ≥ξ+
k,n,F and obtain the stated result.

26


---Page Break---
C
Regret analysis of Local-Greedy and Greedy-Grid

C.1
Clarification on the feedback received by the algorithms

In this section, we consider the case where a feedback (in the form of a sample from a power of F) is
gathered only when an auction is won. If this is not the case, the decision-maker only knows that the
coalition lost the auction. Therefore, if at time t, nt agents are assigned to an auction and the auction
is lost, it makes sense to continue assigning nt agents to the auction at time t + 1, in order to gather
the information that the algorithm wanted to obtain. The meta algorithm called CoMAB for coalition
multi-armed bandits described in Algorithm 3 implements this strategy.

Algorithm 3 CoMAB
Init: J0 = ∅, m = 1
Input: Algo
for t = 1 . . . T do

if t > 1 and auction at t −1 is not won then

Play nt = nt−1 ;
▷Play same arm until an auction is won
else

Play nt = Algo(Jm−1) ;
▷When an auction is won play as prescribed by input
Algo
if Auction is won then

Observe wnt a sample from F nt+p
Set Jm = Jm−1 ∪{(wnt, nt, m)} ;
▷Record feedback obtained when winning
Update m = m + 1

Local-Greedy and Greedy-Grid are then defined as CoMAB applied on πLG and πGG for local greedy
and greedy-grid respectively. These policies associate a play nm to an history Jm−1. In Algorithm 1
and Algorithm 2, the policies are called sequentially T times and feedback is observed after each
request. This gives an implicit definition of the policies.

Lemma 6 expresses the regret of CoMAB in function of the behavior of any policy π when feedback
is observed after each request.

Lemma 6 (Regret of CoMAB). Consider a policy π that associate to every Jm−1 a play nπ
m. Define
J0 = ∅and Jm = Jm−1 ∪{wnπ
m, nπ
m, m} where wπ
nm is a sample from a distribution with c.d.f
F nπ
m+p and nπ
m = π(Jm−1). Consider mπ
n(m) the number of times π returns n after m calls of π.

After T iterations, CoMAB based on π has regret:

RT ≤

N
X

n=1
E[mπ
n(T)]p + n

n
(r(n∗) −r(n))

Proof. Call nt the play chosen by CoMAB at time t,

ηt = 1{The auction is won at time t} ,

mn(t) = |{ρ ≤t, nρ = n and ηρ = 1}|

the number of times that n is played and the auction is won up to time t and

Zn,m(t) = |{ρ ≤t, nρ = n and m ≤mn(ρ) < m + 1}|

the number of times that n has been played between the m-th time n won an auction and the m+1-th
time. Note that mn(t) ≤mπ
n(t) since at time t, π has been called at most t times.

27


---Page Break---
The regret of CoMAB satisfies:

RT =

T
X

t=1
E[r(n∗) −r(nt)]

=

N
X

n=1
E

" T
X

t=1
1{nt = n}

#

(r(n∗) −r(n))

=

N
X

n=1
E





mn(T )
X

m=1
Zn,m(T)



(r(n∗) −r(n))

=

N
X

n=1
E





mn(T )
X

m=1
Zn,m(T)



(r(n∗) −r(n))

≤

N
X

n=1
E[mπ
n(T)]p + n

n
(r(n∗) −r(n))

where in the second to last inequality, we used the independence between nt = n and Zn,mt(n)(T)
as nt = n depends only on the history at times t < mt(n) while Zn,mt(n)(T) depends only on times
t ≥mt(n).

C.2
Auxiliary result

Before proving the theorems, we present an auxiliary result from [6] that we to derive upper bounds
that can be recovered explicit in the proof of the theorems. Since the proof is simple, we recall it for
completeness.
Lemma 7 (Lemma 4 from [6]). For any ζ ≥1, the mapping

fζ : x ∈[(ζ + 2)ζ ∨3, ∞) 7→sup

t ∈N :
t
log(t)ζ ≤x


satisfies
fζ(x) ≤(ζ + 2)ζ × log(x)ζx.

Proof. We start by remarking that the function g(x) =
x
log(x)ζ is strictly increasing for all x ≥eζ.
Now, consider a value s = Ax log(x)ζ for some A > 0, such that s ≥3 ∨eζ. By the monotonicity
of
t
(log t)ζ , we have that

t > s ⇒
t
(log(t)ζ >
s
(log(s)ζ = x ×
A log(x)ζ

(log(A) + log(x) + ζ log(log(x)))ζ .

Then, for x ≥A ≥3, it holds that log(A) + log(x) + ζ log(log(x)) ≤(ζ + 2) log(x), so we can
simply choose A = (ζ + 2)ζ to obtain the result.

All that is left is to verify that for this choice, s = (ζ + 2)ζ × log(x)ζx ≥3 ∨eζ, but this clearly
holds for all x ≥3 and ζ > 0.

C.3
Proof of Theorem 2

Theorem 2 (Regret bound for Local-Greedy). Let ∆:= minn∈[N−1] |r(n + 1) −r(n)| (worst local
gap). Under Assumption 2 and with α = (log3/2 N + 1)−1, the regret of LG is upper bounded by a
problem-dependent constant: there exists (Cn)n∈[N]\{n⋆}, each satisfying Cn = e
ON
  ∆n

∆2

, such
that RT ≤P

n∈[N]\n⋆Cn.

Additionally, if the arm set forms a single estimation neighborhood, that is ∀n ∈[N] : V(n) ⊃[N],
then each constant Cn can be refined to e
On
 
∆−1
n

, providing RT = e
O(
√

NT), which holds even
when the reward function is not unimodal.

28


---Page Break---
Proof. First, we denote by ert(n) the reward estimate used for arm n at time t, and by brk,t(n) its value
when the arm used to compute the estimate is fixed to k ∈V(n). The proofs rely on concentration
bounds on ert(n) derived from Theorem 1, with a confidence level δt that will be fixed later. However,
we will use this result with extra care given that the identity of the arm k used to compute the
estimate is a random variable, as well as its sample size mk(t). This issue is tackled with appropriate
union bounds. Furthermore, in order to simplify the presentation we denote by E(m, δ) the maximal
diameter (as a function of k) of the confidence interval provided by Equation (7), defined by a number
of plays m of the arm used to estimate, and by a confidence level δ. More precisely, with notation of
Theorem 1, for any (k, n) ∈[N]2 we write that |brk,t(n) −r(n)| ≤E(mk(t), δt) with probability at
least 1 −δt. Furthermore, E(mk(t), δt) is increasing in δt and decreasing in mk(t). Finally, we use
the notation K = ⌈log3/2(N)⌉, so that α =
1
K+1.

We now prove the first statement of the theorem, by upper bounding the number of plays of each
sub-optimal arm.

Single neighborhood
Consider any sub-optimal arm n. The main ingredient of the proof is to
tackle the forced sampling by using that if n is pulled at time t, then it is either pulled “on purpose”
or due to forced sampling. However, it it forced sampled then it must have been selected “on purpose”
by being the best empirical arm in the neighborhood at some previous point in time. We hence
consider the following good event

Gt = {∀s ∈{t −⌊αt⌋, . . . , t}, (∀k ∈V(n) : mk(s) ≥αt), |brk,s(n) −r(n)| ≤E(mk(s), δs)} .

Using Theorem 1, Gt holds with probability at least 1 −|V(n)|t2δt, where we used a crude union
bound on the values of s, k and mk(t) (t could be replaced by t −⌊α⌋t⌋). Using this result, we first
upper bound the number of plays of n up to horizon T as follows,

E

" T
X

t=1
1{nt = n}

#

≤E

" T
X

t=1
1{∃s ∈{t −⌊αt⌋, . . . , t} : brs(n) ≥brs(n⋆)}

#

≤E

" T
X

t=1
1{∃s ∈{t −⌊αt⌋, . . . , t} : brs(n) ≥brs(n⋆)} 1{Gt}

#

+

T
X

t=1
P( ¯Gt) .

As discussed above, the second term satisfies PT
t=1 P( ¯Gt) ≤P+∞
t=1 |V(n)|tδt. In order to make
it constant, we choose δt =
1
|V(n)|t4 . For the first term, we use that ∀s ∈[αt, t], E(mk(s), δt) ≤
E(α(1 −α)t, δt) and that n can be played only if the two confidence intervals overlap. Hence, we
further obtain that

E

" T
X

t=1
1{nt = n}

#

≤E

" T
X

t=1
1{2E(α(1 −α)t, δt) ≥∆n}

#

+

+∞
X

t=1
|V(n)|t2δt−⌊αt⌋

= E

" T
X

t=1
1{2E(α(1 −α)t, δt) ≥∆n}

#

+

+∞
X

t=1
(1 −α)−4t−2

= tα,n +
π2

6(1 −α)4 ,

= tα,n + π2

6


1 + 1

K

4
,
since α =
1
K + 1,

and for a deterministic constant tα,n defined by

tα,n = sup{t ∈N : 2E(α(1 −α)t, δt) ≥∆n} .

We recall that K = ⌈log3/2(N)⌉is used to simplify the notation. The last step consists in upper
bounding the value of tα,n explicitly according to n and ∆n. Considering some t ≥4 ∨n + 1, from

29


---Page Break---
Theorem 1 we know that there exist universal constants β and ξ, and we obtain that

E(α(1 −α)t, δt) ≤β

v
u
u
u
tlog

2⌈n√

mk(s)⌉
δt



mk(s)
+ n × ξ







log

2⌈n√

mk(s)⌉
δt



mk(s)







2
3

≤β

s

log
 
2t4(n + 1)
√

t|V(n)|


α(1 −α)t
+ n × ξ

 
log
 
2t4(n + 1)
√

t|V(n)|


α(1 −α)t

! 2

3

≤β

s

log (t5(n + 1)2)

α(1 −α)t
+ n × ξ

 
log
 
t5(n + 1)2

α(1 −α)t

! 2

3

≤β

s

7 log (t)
α(1 −α)t + n × ξ
 7 log (t)

α(1 −α)t

 2

3
.

Since α =
1
K+1 it holds that α(1 −α) =
1
K+1
K
K+1 ≥
1
2(K+1). We then get that for some universal
constants β′ and ξ′, it first holds that:

E(α(1 −α)t, δt) ≤

β′√

K + 1 + n(K + 1)
2
3 ξ′ r

log(t)

t
,
(23)

where we bounded

log(t)

t
 2

3 by
q

log(t)

t
. Without this simplification, we also obtain that

E(α(1 −α)t, δt) ≤(K + 1)

2
3
(

β′ + n
log(t)

t

 1

6
ξ′
) r

log(t)

t
.
(24)

The different scaling proposed in the theorem then come from using Lemma 7 on (23) and (24),
taking the minimum between the two (since both bounds are valid simultaneously), and for the
latter splitting cases depending on
t
log(t) ≤n6 being satisfied or not (taking this time the maximum
between the two cases).

We provide the right-hand term of the result using (23), applying Lemma 7 with ζ = 1 and

x =
4
∆2n
×

β′√

K + 1 + n(K + 1)
2
3 ξ′2
,

which leads to tα,n ≤3x log(x). This provides the term O

n2
∆2n


of the result, and constants in the

logarithmic terms can be recovered explicitly by recovering the values of β′ and ξ′.

We then obtain the left-hand term of the result by considering (24). Lemma 7 first provides that

n

log(t)

t
 1

6 ≤1 for t ≥18n6 log(n) (first bound). Still using Lemma 7, this simplification permits
to use ζ = 1 and the threshold

y =
4
∆2n
×

β′√

K + 1 + (K + 1)
2
3 ξ′2
,

and an upper bound of tα,n ≤3y log(y), but only if this term is larger than 18n6 log(n). This
provides the remaining terms of the bound, and again the logarithms can be easily recovered by
computing β′ and ξ′.

This concludes the proof for the problem-dependent in the favorable case where all arms are neighbors,
remarking that these upper bounds just have to be multiplied by ∆n and summed over n ∈[N] to
convert into the regret bound. Furthermore, the problem-independent guarantee can be derived from
taking the minimum between the bound and ∆nT, remarking that the worst case is ∆n = T −1/2 if
we omit the logarithms.

30


---Page Break---
General case
We now provide the regret bound for the general case, where at least some arms do
not include [N] in their neighborhood. We recall that two main ingredients of Local-Greedy are (1)
that the arm nt played in t is the best empirical arm in the neighborhood of V(nt−1), according to
the simple estimates computed with samples from nt−1, and (2) that nt = nt−1 if mnt−1(t) < αt
(forced sampling). Hence, similarly to the previous proof we use that nt is either pulled thanks to
a “greedy play” or because of forced sampling. Furthermore, the two cases can be merged because
forced sampling can only come after nt being pulled because of a greedy play in the recent rounds.
More precisely, we use that

{nt = n} ⊂{∃s ∈[t −⌊αt⌋] : ns = n, ms(ℓs) ≥⌈αs⌉} .
(25)

This argument is at the core of our analysis, but before going further we need to introduce the notion
of locally optimal plays.

Definition 2 (Locally optimal plays and optimal path). Given a reference arm ℓ, playing n ∈V(ℓ) is
locally optimal if n = argmaxk∈V(ℓ) r(k). In that case, n is the best neighbor of ℓ, and we use the
notation n = v+(n).

Furthermore, a sequence of successive locally optimal plays is an optimal path towards n⋆. By
construction of V, an optimal path contains at most K := O(log(N)) sub-optimal arms.

The last fact presented in the definition is trivial: in the worst case the path start at one of the extremes
of the interval [N] and n⋆is at the other. By design of V (Definition 1) we obtain that n⋆is reached
in ⌈log3/2(N)⌉steps at most. The rest of the proof is based on the idea that, when t is large enough,
the algorithm starts following an optimal path with high probability, so a sub-optimal arm can be
played only if it is located on an optimal path from another sub-optimal arm to n⋆. We formalize it
with the following result.

Lemma 8 (Existence of a “recent” sub-optimal play). For any time step t ∈[T] and arm n ̸= n⋆, it
holds that

{nt = n} ⊂At :=

∃s ∈

(1 −α)Kt, t

, l ∈[N], l′ ∈V(l) : brs(l) ≤brs(l′),

ms(l) ≥α(1 −α)Kt
and
r(l) > r(l′)
	
,

Proof. Starting from (25), we first use that either n ̸= v+(ℓs), and in that case playing n is locally
sub-optimal so this event belongs to At, or n = v+(ℓs). Let us now consider this second case: by
definition of the leader, ℓs was played right before the sequence of forced plays of n started, which
must have happened at least as recently as t −⌊αt⌋−1. From that point, the recursion pattern is
clear: ℓs must have been selected in the last ⌊α(s −1)⌋, by either being a locally sub-optimal play
or not. The first case is included in At, why the second requires to add another step in the analysis.
Furthermore, the arm used to estimate ℓs was itself samples at least proportionally to s. Using that
this can happen K times, and that the worst number of steps to look into the past at each step is at
most a fraction (1 −α) of the number in the previous step, we finally obtain that there have been a
sub-optimal greedy play in the last

(1 −α)Kt, t

steps.

Before going further, we justify the tuning α =
1
K+1, by stating that it maximizes α(1 −α)K (used
later in the proof). At this step, we can simplify the notation by remarking that

(1 −α)K =

K
K + 1

K
≥e−1 ,

hence we replace (1 −α)K by e−1 in the rest of the proof.

In words Lemma 8 states that, even if forced sampling slows down the ascension towards n⋆, since
optimal paths contain at most K sub-optimal arm then n⋆is relatively fast to reach from any arm in
[N]. Next, we use this result in the regret analysis by considering its occurrences with the following
event,

Ht =

∀s ∈

e−1t, t

, ∀n ∈[N], ∀k ∈V(n) : mk(s) ≥
e−1

1 + K t,

|brk,s(n) −r(n)| ≤E(mk(s), δs)} .

31


---Page Break---
Then, for any tK ∈N we can upper bound the number of plays of each sub-optimal arm n ∈[N] as
follows,

E

" T
X

t=1
1{nt = n}

#

≤E



X

t≥1
1{At}





≤tK + E



X

t≥tK
1{At, Ht}



+
X

t≥tK
P( ¯Ht) ,

with the slight abuse of notation that E is now defined with the coalition size N and not the local value
of n considered. The rest of the proof is analogous to the simple case, where all arms are in a single
neighborhood. We first choose δt =
1
N 2t4 , and obtain that P

t≥tK P( ¯Ht) ≤P+∞
t=1
1
e−4t2 ≤
π2
6e−4 .

Next, we tune tK large enough so that E
hP

t≥tK 1{At, Ht}
i
= 0. This can be done by choosing

tK = sup

t ∈N : 2E
 e−1

1 + K t, δt)

≤∆

.

where ∆= minn∈[N−1]{|r(n + 1) −r(n)|}.

We then deduce the result by applying the exact same steps as for the upper bound of tα,n, by carefully
replacing δt =
1
|V(n)|t4 by δt =
1
N2t4 , and α(1 −α)t by
e−1
1+K t. Lemma 7 then allows to easily obtain
the desired scaling, and to make the upper bound explicit by substitution. Since K is logarithmic in
N, it is clear that it only contributes to the bound only by logarithmic factors.

C.4
Proof of Theorem 3

Theorem 3 (Regret upper bound for Greedy-Grid). Suppose that GG is tuned with confidence level
δt =
1
N 2t3 , and α = 1/4. Then, for any T ∈N it holds that

RT = e
ON

 X

n∈B⋆

1
∆n
+
X

n∈S

log(T)

∆n
∧∆n

 
1{n < n⋆}

∆2
vl(n⋆)
+ 1{n > n⋆}

∆2
vr(n⋆)

!!

.

Additionally, it holds that RT = e
O
p

(K + |B⋆|)T

, for K = ⌊log3/2(N)⌋.

Proof. As the theorem suggests, we will use different arguments depending on the position of n with
respect to the grid and the optimal bin B⋆. Before that, we introduce the crucial result of this proof:
for each time t large enough, thanks to the design of Greedy-Grid, all arms in optimal bin B⋆are
estimated with a simple estimate computed with a linear number of samples in t.

Following the implementation of GG, at each time t and for each arm n ∈[N], a confidence interval
[LCBt(n), UCBt(n)] is computed so that r(n) ∈[LCBt(n), UCBt(n)] with probability at least 1 −δt.
Similarly to the proof of Theorem 2, we consider a “good event” stating that all confidence intervals
where valid on a given time range before t,

Gt =

∀s ∈
 3t

16


, t

, ∀n ∈[N], rt(n) ∈[LCBt(n), UCBt(n)]

.

It is clear that P+∞
t=1 P( ¯Gt) ≤P+∞
t=1 N 2tδ⌈3t

16⌉, where the second union bound on N comes from
considering the (random) identity of the arm whose samples are used to compute the interval. The
following results proves that, under Gt, there is at least one arm in the bin B⋆that was played a linear
number of times in t.

Lemma 9 (Linear number of plays in B⋆under Gt). Under Gt, there exists an arm n ∈B⋆∪
{vS
ℓ(n⋆), vS
r (n⋆)} satisfying mn(3t/4) ≥
t
4K ∧t

8, where we call K = |S| = ⌊log3/2(N)⌋.

Proof. Gt guarantees that any play during the interval
 t

4

, t

(including those due to forced
sampling), were decided with valid confidence intervals. Indeed, starting at 3t

16 we are sure that all

32


---Page Break---
forced exploration launched before that time is completed in t/4. Furthermore, it is also direct from
the design of the algorithms that, if rs(n) ∈[LCBs(n), UCBs(n)] and Greedy-Grid is not forced to
sample the previous arm it must hold that (1) it is playing the grid, or (2) it is playing an arm in B⋆.
Indeed, no arm from a sub-optimal bin would eliminate its best neighbor.

We consider two cases. First, if an arm n ∈B⋆was played between rounds
 t

4

and
 t

2

. In that case
it has collected at least t

8 samples before t, thanks to forced sampling. In the alternative case, the
grid was played between those these two rounds, which incurs
t
4K plays of argmaxs∈S r(s) since by
G, argmaxs∈S r(s) is not eliminated when the grid is played. Then, notice that argmaxs∈S r(s) ⊂
{n∗, vS
ℓ(n⋆), vS
r (n⋆)}. The result follows by combining the two cases.

Without loss of generality, we assume in the following that K ≥2 (if this is not the case, just replace
K by max(K, 2)). As a direct consequence of Lemma 9, using Theorem 1,under Gt there exists
some constant β and ξ (coming from the bounds of the theorem multiplied by 2
√

4 = 4) such that
the LCB of arm n⋆satisfies

∀s ∈
3t

4 , t

: LCBs(n⋆) ≥r(n⋆) −









β

v
u
u
t

K
log

2⌈N
√

t⌉
δt



t
+ N × ξ



K
log

2⌈N
√

t⌉
δt



t





2
3








.

(26)

To simplify the notation, we denote the right-hand term by E(t) in the rest of the proof. Using this
result, we can now consider all the sub-cases presented in the theorem. We fix a sub-optimal arm
n, and upper bound PT
t=1 1{nt = n, Gt} depending on the position of n with respect to the grid S.
Similarly to what we did in the proof of Theorem 2, we relate the pulls due to forced sampling to
actual decisions by stating that, if nt = n, then there exists a round s between t −⌊t/4⌋and t such
that GG requested a pull of arm n from a “grid play” or a “greedy play”.

Case 1: n ∈S. In that case if n is pulled then, since there is no forced sampling for the arms of the
grid it directly holds that

UCBt(n) ≥max
n′∈[N] LCBt(n′) ≥LCBt(n⋆) ≥r(n⋆) −E(t) .
(27)

On the other hand, under Gt it holds that

UCBt(n) ≤r(n) +







β

v
u
u
tK
log

2⌈N
√

t⌉
δt



mn(t)
+ N × ξ



K
log

2⌈N
√

t⌉
δt



mn(t)





2
3 





|
{z
}
E′(t)

,

so pulling arm n is possible only if E(t) + E′(t, mn(t)) ≥∆n, that we simplify to 2E′(t, mn(t)) ≥
∆n or 2E(t) ≥∆n. Considering the second term leads to a constant problem-dependent bound,
analogous to tα,n in the proof of Theorem 2. Hence, we focus on the first term, that provide the
log(T) bound.

This time, we don’t know if mn(t) is large or not. This explains why we obtain a UCB-like
( e
O(log(T))) upper bound with this technique. We use that

t ≤T ⇒log

t2⌈N
√

t⌉
δt


≤log

 

T 2⌈N
√

T⌉
δT

!

= e
O(log(T)) .

Then, similarly to proof of Theorem 2, we mitigate the asymptotic scaling in N by noticing that if
mn(t) = Ω(N 6) then
N

mn(t)
1
6 simplifies. In that case, we obtain a sub-Gaussian confidence interval,

and similarly to the analysis of UCB [3] we obtain that arm n is only pulled at most e
O

log(T )

∆2
n
∨N 6

times with high probability. This is the first part of the result for n ∈S.

33


---Page Break---
We then use another analysis to derive the constant problem-dependent bound. We remark that, when
the confidence intervals are valid, the best arm between vS
ℓ(n⋆) and vS
r (n⋆) can only be eliminated by
an arm i⋆
t ∈B⋆. By design of the algorithm (exploiting the unimodality assumption), it furthermore
holds that if i⋆
t ∈B⋆and those two arms are eliminated then GG does not play on the grid. Hence, the
constant bound in this case comes from upper bounding the time required for this event to happen
under Gt. If vS
ℓ(n⋆) is not eliminated it must hold that UCBt(vS
l (n⋆)) ≥LCBt(n⋆) (if n ≤n⋆).
Furthermore, Lemma 9 also guarantees that

UCBt(vℓ(n⋆)) ≤r(vℓ(n⋆)) + E(t) ,

therefore the event that vℓ(n⋆) is not eliminated is only possible if 2E(t) ≥∆2
vℓ(n⋆). We can then use
Lemma 7, following the same as in the proof of Theorem 2 right after (23) and (24). When T is large

enough, the derivation provides the scaling e
O

1
∆2
vS
ℓ(n⋆)


. We can then can follow the same steps for

vS
r (n⋆), and obtain e
O

1
∆2
vS
r (n⋆)


. Furthermore, it is clear by analogy with the proof of Theorem 2

that for small values of T the upper bounds have to be multiplied by a N 2 factor. Finally, we can
remark that if n < n⋆the first bound is used, while the second is used for n > n⋆. This concludes
the derivation of the upper bound for n ∈S.

Case 2: n /∈S, B(n) ̸= B⋆. We prove that this case is actually impossible under the good event,
which explains the surprising constant upper bound independent of any gap. Indeed, if n /∈S is
played, then it must hold that its right and left neighbors in the grid are eliminated. Since B(n) ̸= B⋆
then at least one of them has a reward at least as good as any arm in B(n). However, if playing arm n
was possible under Gt it would hold that

∃ℓ∈B(n) :
r(ℓ) ≥LCBt(ℓ)

> max{UCBt(vS
ℓ(n)), UCBt(vS
r (n))}

≥max{r(vS
ℓ(n), r(vS
r (n))} ≥r(ℓ) ,

which is a contradiction due to the strict inequality in the second line. Hence, the number of time
such arm n is played is simply upper bounded by P+∞
t=1 P( ¯Gt), which is (by design) bounded by a
universal constant.

Case 3: n /∈S, B(n) = B⋆We use that nt = n implies that ns = n due to a greedy play at some
round s ∈[3t/4, t]. Under Gt, we can thus directly use (26), and obtain that if 2E(t) ≤∆2
n this event
is not possible anymore. Using the same derivation as in the other cases (involving Lemma 7), we
obtain the upper bound scaling in O

1
∆2n


for T large enough, and by O

N2
∆2n


in general.

C.5
Regret of Greedy-Grid adapted for non-unimodal rewards

In this section we develop the result presented in Section 3, regarding the adaptation of Greedy-Grid in
the case when the reward function is no longer assumed to be unimodal. We recall that the adaptation
consists in simplifying the definition of the set Ct in Algorithm 2 by

Ct = {s ∈S, Us ≥Li∗
t } .

We call the resulting algorithm GG-NU, for Greedy-Grid Non Unimodal, to differentiate it from the
original version of GG introduced in the paper. In the following, we formalize the upper bound of the
regret of GG-NU, and discuss how this result is obtained by adapting the proof of Theorem 3 from the
previous section.
Theorem 4 (Regret of GG-NU). Suppose that GG is tuned with confidence level δt =
1
N 2t3 , and
α = 1/4. Then, for any T ∈N it holds that

RT = e
ON

 X

n∈B⋆

1
∆n
+
X

n∈S

log(T)

∆n

!

.

Additionally, it holds that RT = e
O
p

(K + |B⋆|)T

, for K = ⌊log3/2(N)⌋.

34


---Page Break---
Proof. The proof follows the exact same steps as the proof of Theorem 3 presented in Appendix C.4.
To adapt the arguments to GG-NU, we first remark that it suffices to identify which part of the proof
uses the definition of the set Ct. We then find that this is the case when analyzing the Case 1 in
the proof, namely the regret caused by sub-optimal arms in the grid |S|. More precisely, to obtain
the bound of Theorem 3 for GG we provide two simultaneously valid upper bounds: a logarithmic
(log(T)) upper bound with a reasoning akin to the standard UCB analysis, and then a constant upper
bound that carefully leverages the definition of Ct. We easily verify that the steps for the logarithmic
bound remain valid with the new definition of Ct, while the second bound clearly does not hold. This
completes the adaptation of Theorem 3 (for GG) into Theorem 4 (for GG-NU).

D
Experiments

All the code for the experiments is written in Python. We use Matplotlib for plotting [18], Numpy [15]
for numerical computing and Scipy [31] for scientific and technical computing. Scipy and Numpy
are distributed under the BSD 3-Clause, and Matplotlib is distribution under BSD-style license.
Theses licenses allow free use, modification, and distribution of the library. All the experiments were
conducted on a single standard laptop, with an execution time shorter than 24 hours.

In a first simulation, we consider a coalition of size N = 100 and a competition of size p = 4. At
each timestep t, the algorithm decide a number of bidders nt to send to the auction and the values
of all bidders (coalition and competition) v ∈{0, 1}nt+p are sampled according to B(0.05). With
probability
nt
nt+p, the reward v(1) −v(2) is received and v(1) is observed. The (pseudo) regret at time
t is then computed as the sum of reward obtained up to time t. The simulation above is repeated 20
times with random seeds and the mean value across seeds is reported as the expected regret R(t) in
Figure 3. Error bars represent the first and the last decile.

In this simulation, the parameters are chosen to allow for having a significant number of players
while keeping a gap ∆large enough (about 2 × 10−4) to be able to observe logarithmic regrets for
the baselines. LG practically outperforms other approaches by a large gap. The two algorithms that
ignore the structure (UCB and EXP3) end up exhibiting a worse regret than LG, OSUB and GG, which
is expected. However GG has a much higher regret than LG and only outperform UCB and EXP3 for
horizons greater than 105, when it starts to eliminate points from the logarithmic grid S. Indeed, due
to the explicit use of concentration bounds in the algorithm, which multiplicative constants are not
optimized for practical implementations, GG does not practically reach the constant regret regime in
the horizon of these simulations.

To illustrate the practical performance of LG, we perform additional simulations which are identical
to the first one except for the parameters N, p, and the distribution of value that are set according
to Table 2. The results are plot in Figure 4 where it is shown that LG reaches the constant regret
regime after only a couple hundreds or thousands time steps while the other algorithms are still in the
transient linear regime.

Table 2: Configuration of additional experiments presented in Figure 4.

Position
N
p
Value distribution
n∗
∆

Top
5
2
Beta(a = 0.35, b = 0.63)
3
5 × 10−5

Middle
5
2
Trunc. Exp(0, 1)
3
4 × 10−4

Bottom
20
4
Trunc. Exp(0, 1)
5
3 × 10−5

35


---Page Break---
Figure 3: An empirical illustration of Table 1 with simulations in the following setting: values are
distributed according to B(0.05), N = 100 and p = 4. We benchmark LG and GG (this paper),
OSUB [9], UCB [3] and EXP3 [4] in terms of R(T) computed over 20 trajectories. Error bars represent
the first and last decile.

36


---Page Break---
Figure 4: Illustration of additional experiments. Details of parameters are provided in Table 2.

Complexity analysis
The time-complexity of both GG and LG mainly comes from the computation
of reward estimates. They are computed by replacing the integral in Equation (5) by a Riemann sum

37


---Page Break---
with ⌊N
√

T⌋terms (the reasoning behind the number of terms needed is the same as in the proof
of Theorem 1. Therefore, whenever reward estimates of a neighborhood of size O(N) is needed,
it costs O(N 2√

T) operations. Note that during forced exploration steps, reward estimates are not
needed and therefore the associated cost is not paid. The total time complexity therefore depends on
the number of times reward estimates are needed which itself depends on the trajectory. However, our
algorithms could be modified to guarantee that reward estimates are needed at most O(log(T)), times
for instance by only performing updates at the end of phases of exponentially increasing length. This
would lead to a mean complexity per iteration of O(N log(T)), and similar theoretical guarantees
by a slight adaptation of the analysis. This would reduce the burden of using of incorporating the
structure in the algorithm, compared to the O(N) cost of the baselines (which is even O(1) for
OSUB).

E
Broader Impact

The collection of user data should be carried out with the preservation of user privacy in mind. This
issue is at the forefront of recent, ongoing developments, such as the European Union’s General Data
Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA).

In online advertising, maintaining privacy presents new challenges, as decisions must be made without
complete access to user data.

This paper tackles one such challenge by detailing Multi-Armed Bandit (MAB) algorithms that
Demand Side Platforms (DSPs) can use to determine the number of ad campaigns that should partake
in the repeated auction for ad placements, without the need for prior knowledge of each campaign’s
value. This study is therefore a step towards the realization of practical user privacy. It is important to
note that the assumptions we make require that the actions of the DSP have a limited impact on the
market, which should be carefully verified in practical applications.

38


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The theoretical and experimental results provided in (Theorems 1 to 3 and ap-
pendix D ) correspond to the claims made in the abstract and introduction.
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
Justification: See the discussion at the end of Section 3 and future work directions in
Section 4.
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

39


---Page Break---
Answer: [Yes]
Justification: In Theorems 1 to 3, the assumptions are referenced in the statement of each
theorem, sketch of proofs are provided below each theorem and the detailed proofs are
available in Appendices B, C.3 and C.4 respectively.

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
Justification: Experiments are described in the text in Appendix D and can be reproduced
using the code provided in supplementary material.

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

40


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: yes the paper provide open access to the code (as a zip folder in the supple-
mentary materials) with all the instructions needed to reproduce it .

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

Justification: Yes all the details for the experimental setting are provided by the paper (see
Appendix D).

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

Justification: Error bars are reported and defined in Appendix D.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

41


---Page Break---
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
Justification: Experiments were run on a laptop in less than a day (see also Appendix D).
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
Justification: Yes the paper is conform with the NeurIPS Code of Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer:[Yes]
Justification: The discussion on the Broader Impacts of this work is in Appendix E
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

42


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

Justification: The paper poses no such risks.

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

Justification: The python packages were cited and use open source licenses (see Ap-
pendix D).

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

43


---Page Break---
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
Answer: [Yes]
Justification: The code for the experiments is provided in the supplementary materials as a
zip folder.
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

44


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

45


---Page Break---
