Addressing Bias in Online Selection with Limited
Budget of Comparisons

Ziyad Benomar
ENSAE, Ecole Polytechnique,
FairPlay joint team
ziyad.benomar@ensae.fr

Evgenii Chzhen
CNRS, LMO, Université Paris-Saclay
evgenii.chzhen@universite-paris-saclay.fr

Nicolas Schreuder
CNRS, Laboratoire d’informatique
Gaspard Monge (LIGM/UMR 8049)
nicolas.schreuder@cnrs.fr

Vianney Perchet
CREST, ENSAE, Criteo AI LAB
Fairplay joint team
vianney.perchet@normalesup.org

Abstract

Consider a hiring process with candidates coming from different universities. It is
easy to order candidates with the same background, yet it can be challenging to
compare them otherwise. The latter case requires additional costly assessments,
leading to a potentially high total cost for the hiring organization. Given an assigned
budget, what would be an optimal strategy to select the most qualified candidate?
We model the above problem as a multicolor secretary problem, allowing compar-
isons between candidates from distinct groups at a fixed cost. Our study explores
how the allocated budget enhances the success probability in such settings.

1
Introduction

Online selection is among the most fundamental problems in decision-making under uncertainty,
Multiple problems within this framework can be modeled as variants of the secretary problem
[Dynkin, 1963, Chow et al., 1971], where the decision-maker has to identify the best candidate among
a pool of totally ordered candidates, observed sequentially in a uniformly random order. When a
new candidate is observed, the decision maker can either select them and halt the process or reject
them irrevocably. The optimal strategy is well known and consists of skipping the first 1/e fraction
of the candidates and then selecting the first candidate that is better than all previously observed ones.
This strategy yields a probability 1/e of selecting the best candidate. A large body of literature is
dedicated to the secretary problem and its variants, we refer the interested reader to [Chow et al.,
1971, Lindley, 1961] for a historical overview of this theoretical problem.

In practice, as pointed out by several social studies, the selection processes often do not reflect the
actual relative ranks of the candidates and might be biased with respect to some socioeconomic
attributes [Salem et al., 2022, Raghavan et al., 2020]. To tackle this issue, several works have explored
variants of the secretary problem with noisy or biased observations [Salem and Gupta, 2019, Freij and
Wästlund, 2010]. In particular, Correa et al. [2021a] studied the multi-color secretary problem, where
each candidate belongs to one of K distinct groups, and only candidates of the same group can be
compared. This corresponds for example to the case of graduate candidates from different universities,
where the within-group orders are freely observable and can be trusted using a metric such as GPA,
but inter-group order cannot be obtained by the same metric. This model, however, is too pessimistic,
as it overlooks the possibility of obtaining inter-group orders at some cost, through testing and
examination. Taking this into account, we study the multicolor secretary problem with a budget for
comparisons, where comparing candidates from the same group is free, and comparing candidates

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
from different groups has a fixed cost of 1. We assume that the decision-maker is allowed at most
B comparisons. This budget B represents the amount of time/money that the hiring organization
is willing to invest to understand the candidate’s “true” performance. As in the classical secretary
problem, an algorithm is said to have succeeded if the selected candidate is the best overall, otherwise,
it has failed. The objective is to design algorithms that maximize the probability of success.

1.1
Contributions

The paper studies an extension of the multi-color secretary problem [Correa et al., 2021a], where
comparing candidates from different groups is possible at a cost. This makes the setting more realistic
and paves the way for more practical applications, but also introduces new analytical challenges.

In Section 2, we describe a general class of Dynamic-Threshold (DT) algorithms, defined by distinct
acceptance thresholds for each group, that can change over time depending on the available budget.
First, we examine a particular case where all thresholds are equal, which can be viewed as an
extension of the classical 1/e-strategy. However, the analysis is intricate due to additional factors
such as group memberships, comparison history, and the available budget. By carefully controlling
these parameters, we compute the asymptotic success probability of the algorithm, demonstrating its
extremely rapid convergence to the upper bound of 1/e when the budget increases, hence constituting
a first efficient solution to the problem.

Subsequently, our focus shifts to the case of two groups, where we explore another particular case of
DT algorithms: static double threshold algorithms. These involve different acceptance thresholds
for each group, that depend on the groups’ proportions and the initial budget, but do not vary during
the execution of the algorithm. We prove a recursive formula for computing the resulting success
probability, and we exploit it to establish a closed-form lower bound and compute explicit thresholds.

In the two-group scenario, we also derive the optimal algorithm among those that do not utilize the
history of comparisons, which we refer to as memory-less. We present an efficient implementation
of this algorithm and demonstrate, via numerical simulations, that it belongs to the class of DT
algorithms when the number of candidates is large. Leveraging this insight, we numerically compute
optimal thresholds for the two-group case.

1.2
Related work

The secretary problem
The secretary problem was introduced by Dynkin [1963], who proposed
the 1/e-threshold algorithm, having a success probability of 1/e, which is the best possible. Since
then, the problem has undergone extensive study and found numerous applications, including in
finance [Hlynka and Sheahan, 1988], mechanism design [Kleinberg, 2005, Hajiaghayi et al., 2007],
Nested Rollout Policy Adaptation (NRPA) [Dang et al., 2023], active learning [Fujii and Kashima,
2016], and the design of interactive algorithms [Sabato and Hess, 2016, 2018]. Moreover, the
secretary problem has multiple variants [Karlin and Lei, 2015, Bei and Zhang, 2022, Assadi et al.,
2019, Keller and Geißer, 2015], and has inspired other works, for instance, related to matching
[Goyal, 2022, Dickerson et al., 2019] or ranking [Jiang et al., 2021]. A closely related problem is the
prophet inequality [Krengel and Sucheston, 1977, Samuel-Cahn, 1984], where the decision-maker
sequentially observes values sampled from known distributions, and its reward is the value of the
selected item, in opposite the secretary problem where the reward is binary: 1 if the selected value
is the maximum and 0 otherwise. Prophet inequalities also have many applications [Kleinberg and
Weinberg, 2012, Chawla et al., 2010, Feldman et al., 2014] and have been explored in multiple
variants [Kennedy, 1987, Azar et al., 2018, Bubna and Chiplunkar, 2023, Benomar et al., 2024].

Different information settings
In some practical scenarios, the secretary problem may present a
pessimistic model. Therefore, variants with additional information have been studied. For example,
Gilbert and Mosteller [2006] explored a scenario where candidates’ values are independently drawn
from a known distribution. Other studies have examined potential improvements with other types of
information, such as samples [Correa et al., 2021b] or machine-learned advice [Antoniadis et al., 2020,
Dütting et al., 2021, Benomar and Perchet, 2023]. Conversely, some works more closely aligned
with ours have investigated more constrained settings. Notably, Correa et al. [2021a] introduced the
multi-color secretary problem, where totally ordered candidates belong to different groups, and only
the partial order within each group, consistent with the total order, can be accessed. Under fairness

2


---Page Break---
constraints, they designed an asymptotically optimal strategy for selecting the best candidate. Other
settings with only partial information have been studied as well. For example, Monahan [1980, 1982]
addressed the optimal stopping of a target process when only a related process is observed, and they
designed mechanisms for acquiring information from the target process. However, these works do
not assume a fixed budget and instead consider a penalized version of the problem.

Online algorithms with limited advice
This paper also relates to other works on online algorithms,
where the decision-maker is allowed to query a limited number of hints during execution. The
objective of these analyses is to measure how the performance improves with the number of permitted
hints. Such settings have been studied, for example, in online linear optimization [Bhaskara et al.,
2021], caching, [Im et al., 2022], paging [Antoniadis et al., 2023], scheduling [Benomar and Perchet],
metrical task systems [Sadek and Elias, 2024], clustering [Silwal et al., 2023], and sorting [Bai and
Coester, 2024, Benomar and Coester, 2024]. Another related paper by Drygala et al. [2023] studies a
penalized version of the Bahncard problem with costly hints.

2
Formal problem

We consider a strictly totally ordered set of cardinal N, whose elements will be called candidates. We
assume that these candidates are observed in a uniformly random arrival order (x1, . . . xN), and that
they are partitioned into K groups G1, . . . , Gk. For all t ∈[N], we denote by gt ∈[K] the group of
xt, i.e. xt ∈Ggt, and we assume that {gt}t∈[N] are mutually independent random variables

P(gt = k) = λk,
∀t ∈[N], ∀k ∈[K] ,

where λk > 0 for all k ∈[K] and PK
k=1 λk = 1.

We assume that comparing candidates of the same group is free, while comparing two candidates of
different groups is costly. To address the latter case, we consider that a budget B ≥0 is given for
comparisons and we propose two models: the algorithm can pay a cost of 1 in order to:

1. compare a two already observed candidates xt and xs belonging to different group,
2. determine if the current candidate is the best candidate seen so far among all the groups.

For simplicity, we focus on the second model. However, we explain during the paper how our
algorithms adapt to the first model and the cost they incur.

When a new candidate arrives, the algorithm can choose to select them, halting the process, or it
can choose to skip them, moving on to the next one—hoping to find a better candidate in the future.
Once a candidate has been rejected, they cannot be recalled—the decisions are irreversible. Given the
total number of candidates N, the probabilities (λk)k∈[K] characterizing the group membership, and
a budget B, the goal is to derive an algorithm that maximizes the probability of selecting the best
overall candidate. We refer to the problem as the (K, B)-secretary problem

2.1
Additional notation

For all t < s ∈[N], we denote by xt:s := {xt, . . . , xs}, and for all k ∈[K] we denote by Gk
t:s the
set of candidates of group Gk observed between steps t and s,

Gk
t:s := {xr : t ≤r ≤s and gr = k} = xs:t ∩Gk .

If t = 1, then we lighten the notation Gk
s := Gk
1:s. Let A be any algorithm for the (K, B)-secretary
problem, we define its stopping time τ(A) as the step t when it decides to return the observed
candidate. We will often drop the explicit dependency on A and write τ when no ambiguity is
involved. We will say that A succeeded if the selected candidate xτ is the best among all the
candidates {x1, . . . , xN}. Let us also define, for any step t ≥1, the random variables

rt =

t
X

t′=1
1(xt ≤xt′, gt = gt′)
and
Rt =

t
X

t′=1
1(xt ≤xt′) .
(1)

Both random variables have natural interpretations: given a candidate at time t, rt is its in-group rank
up to time t, while Rt is its overall rank up to time t. Note that the actual values of xt do not play a

3


---Page Break---
role in the secretary problem and we can restrict ourselves to the observations rt, gt, Rt. While the
first two random variables are always available at the beginning of round t, the third one can be only
acquired utilizing the available budget. At each step t ∈[N], the decision-maker observes rt, gt and
can perform one of the following three actions:

1. skip: reject xt and move to the next one;
2. stop: select xt;
3. compare: if the comparison budget is not exhausted, use a comparison to determine if
(Rt = 1)—compare the candidate xt to the best already seen candidates in the other groups;

Furthermore, if a comparison has been used at time t, the algorithm has to perform stop or skip
afterward. We denote respectively by at,1 and at,2 the first and second action made by the algorithm
at step t. Let us also define g∗
t the group to which the best candidate observed until step t belongs,

g∗
t = argmax
k∈[K]
{max Gk
t }
∀t ∈[N] ,

and Bt as the budget available for A at step t

B1 = B
and
Bt = Bt−1 −1(at,1 = compare)
∀t ∈[N] .

In the presence of a non-zero budget, the first time when A decides to make a comparison will be a
key parameter in our analysis of the success probability. We denote it by ρ1(A),

ρ1(A) = min{t ∈[N] : at,1 = compare} .

As with the stopping time, when there is no ambiguity about A, we simply write ρ1.
Remark 2.1. Although we formalized the problem using the variables rt and Rt, the only information
needed at any step t is 1(rt = 1) and 1(Rt = 1), i.e we only need to know if the candidate is the
best seen so far, in its own group and overall. In practice, 1(rt = 1) can be observed by comparing
xt to the best candidate up to t −1 belonging to Ggt, and if this is the case then 1(Rt = 1) can be
observed by comparing xt to the best candidate in the other group.

3
Dynamic threshold algorithm for K groups

In this section, we introduce a general family of Dynamic-threshold (DT) algorithms. A DT algorithm
for the (K, B)-secretary problem is defined by a finite doubly-indexed sequence (αk,b)k∈[K],b≤B
of real numbers in [0, 1], which determines the acceptance thresholds based on the group of the
observed candidate and the available budget. During a run of the algorithm, the thresholds used for
each group dynamically change depending on the evolution of the available budget. We denote this
algorithm by AB 
(αk,b)k∈[K],b≤B

.

Upon the arrival of a new candidate xt, the algorithm observes its group gt = k ∈[K] and its
in-group rank rt, and has an available budget of Bt = b ≥0. If t/N < αk,b or rt = 0, the candidate
is immediately rejected. Otherwise, if t/N ≥αk,b and rt = 1, then the candidate is selected if b = 0;
and if the budget is not yet exhausted (b > 0), then the algorithm pays a unit cost to observe the
variable 1(Rt = 1). If this variable is 1, indicating a favorable comparison, the candidate is selected;
otherwise, it is rejected. A formal description is given in Algorithm 1, and a visual representation for
the case of three groups is provided in Figure 1.

3.1
Single-threshold algorithm for K groups

In this section, we focus on the single-threshold algorithm, a specific case within the family of DT
algorithms, where all thresholds are identical across groups and budgets. Initially, the algorithm
rejects all candidates until step T −1, where T ∈[N] is a fixed threshold. Upon encountering
a new candidate that is the best within its group, if no budget remains, the candidate is selected.
Alternatively, if there is still a budget available, the algorithm utilizes it to determine if the current
candidate is the best among all groups. If that is the case, the candidate is then selected. We denote
by AB
T the single-threshold algorithm with threshold T and budget B. We demonstrate that this
algorithm has an asymptotic success probability converging very rapidly to the upper bound of 1/e.

In this first lemma, we prove a recursion formula on the success probability of the single-threshold
algorithm, with a threshold T = ⌊αN⌋for some α ∈[0, 1].

4


---Page Break---
훼1,0
훼2,0
훼3,0

훼1,1
훼2,1
훼3,1

훼1,2
훼2,2
훼3,2

훼1,3
훼2,3
훼3,3

0.0
0.2
0.4
0.6
0.8
1.0
푡/푁

0

1

2

3

Remaining budget 푏

comp: 푟푡= 1,푔푡= 1

skip: 푔푡∈{2, 3}

comp: 푟푡= 1,푔푡∈{1, 2}

skip: 푔푡= 3
skip
comp: 푟푡= 1

Dynamic Treshold algorithm: A퐵 (훼푘,푏)푘∈[3],푏≤3


Figure 1: Schematic description of a DT algorithm in the case of 3 groups

Algorithm 1: Dynamic-Threshold algorithm AB 
(αk,b)k∈[K],b≤B


Input: Available budget B, thresholds (αk,b)k∈[K],b≤B
Initialization: b = B

1 for t = 1, . . . , N do

2
Receive new observation: (rt, gt)

3
if t ≥⌊αgt,bN⌋and rt = 1 then
// compare in-group

4
if b > 0 then
// check budget

5
Update budget: b ←b −1

6
if Rt = 1 then Return: t
// compare inter-group

7
else Return: t

Lemma 3.1. The success probability of the single threshold algorithm AB
T with threshold T = ⌊αN⌋
and budget B ≥0 satisfies the recursion formula

P(AB
T succeeds) = α −αK

K −1 + 1(B ≥0) (K −1)

N
X

t=T

T K

tK+1 P(AB−1
t+1 succeeds) + O
q

log N

N

.

The proof hinges on analyzing the behavior of the algorithm following the first comparison. After
that comparison, the algorithm halts if Rt = 1, and the success probability can be computed in that
case. Otherwise, if Rt ̸= 1, the candidate is rejected, and the algorithm transitions to a new state at
step t + 1, where the available budget reduces to B −1. Its success probability becomes precisely
that of algorithm AB−1
t+1 , with budget B −1 and threshold t + 1.

The recursion outlined in Lemma 3.1 can be used to calculate the asymptotic success probability of
the single-threshold algorithm AB
⌊αN⌋as the number of candidates N approaches infinity.

Theorem 3.2. The asymptotic success probability of the single threshold algorithm AB
T with threshold
T = ⌊αN⌋and budget B ≥0 is

lim
N→∞P(AB
⌊αN⌋succeeds) =
αK

K −1

B
X

b=0

 
1
αK−1 −

b
X

ℓ=0

log(1/αK−1)ℓ

ℓ!

!

.

In particular,
lim
B→∞lim
N→∞P(AB
⌊αN⌋succeeds) = α log(1/α) .

Note that, for B = ∞, the asymptotic success probability in the previous theorem corresponds to the
success probability of the algorithm with a threshold ⌊αN⌋in the secretary problem. Indeed, with an
unlimited budget, the decision-maker can assess at each step whether the current candidate is the best
so far, and the problem becomes equivalent to the classical secretary problem.

5


---Page Break---
Alternative comparison model.
In the alternative comparison model presented in Section 2, the
single threshold algorithm AB
T can be adapted to guarantee the same success probability at the cost
of K −1 additional comparisons. After the first T candidates are rejected, K −1 comparisons are
made between the maximums from each group to identify the best candidate so far. The algorithm
then keeps track of the best candidate: whenever a new candidate is the best in their group, they
are compared to the current best candidate using a single comparison, and the latter is updated
accordingly. This approach enjoys the same guarantees as in Theorem 3.2, but with a budget of
K + B −1 instead of B.

The next corollary measures how the success probability of the single threshold algorithm, in the
setting with K groups, converges to 1/e as the budget increases.
Corollary 3.2.1. The success probability of the single-threshold algorithm with threshold T = ⌊N/e⌋
and budget B ≥0 satisfies

lim
N→∞P(AB
⌊N/e⌋succeeds) ≥1

e


1 −(K −1)B+1

(B + 1)!


.

In particular, for all ε > 0, if K ≤1 + B+1

e (eε)
1
B+1 , then limN P(AB
⌊N/e⌋succeeds) ≥(1 −ε)/e.

This corollary proves that the success probability of AB
⌊N/e⌋converges very rapidly to the upper
bound 1/e as B increases. However, the convergence becomes slower when K is larger.

Surprisingly, the asymptotic success probability of AB
⌊N/e⌋is not influenced by the proportions
(λk)k∈[K], but only by the number of groups K. This means that the algorithm does not benefit from
the cases where there is a majority group Gk with λk very close to 1, which would make the problem
easier. Indeed, it is always possible to achieve a success probability of maxk∈[K] λk/e by rejecting
all the candidates not belonging to the majority group Gk∗, and using the classical 1/e-rule counting
only elements of Gk∗. This algorithm can be combined with ours by always running the one with
the highest success probability, depending on the available budget, the number of groups, and the
group proportions. The resulting algorithm has a success probability that converges to the upper
bound 1/e both when B increases and when maxk λk converges to 1. Nonetheless, due to the very
fast convergence of the success probability of the single threshold algorithm to 1/e, the improvement
brought by having a majority group is only marginal when the budget is sufficient.

As a consequence, the single threshold algorithm surprisingly constitutes a very efficient solution to
the problem even with moderate values of the budget. Computing the optimal thresholds remains
however an intriguing question, which we explore in the following sections in the case of two groups.

4
The case of two groups

In this section, we delve into the particular case of two groups, and we demonstrate how leveraging
different thresholds for each group can enhance the success probability. Let λ ∈(0, 1) represent
the probability of belonging to group G1, and 1 −λ the probability of belonging to group G2. We
examine the success probability of Algorithm AB(α, β), with threshold ⌊αN⌋for group G1 and
⌊βN⌋for group G2, having a budget of B comparisons. This algorithm is a specific instance of the
DT family, wherein the thresholds depend only on the group, and not on the available budget. We
call it a static double-threshold algorithm.

We assume without loss of generality that α ≤β, and we denote by CN the event

(CN) :
∀t ≥1 : max(||G1
t| −λt| , ||G2
t| −(1 −λ)t|) ≤4
p

t log N .

This event provides control over the group sizes at each step. Lemma A.2 guarantees that CN holds
true with a probability at least 1 −
1
N2 for N ≥4. Furthermore, for all t ∈[N], we denote by
AB
t (α, β) the algorithm with acceptance thresholds max{⌊αN⌋, t} and max{⌊βN⌋, t} respectively
for groups G1 and G2, and we denote by UB
N,t,k the probability

UB
N,t,k = P(AB
t (α, β) succeeds, g∗
t−1 = k | CN) .
(2)

Similar to the analysis of the single-threshold algorithm, we establish in Lemma C.1 a recursion
formula satisfied by (UB
N,t,k)B,t,k, which we later utilize to derive lower bounds on the asymptotic

6


---Page Break---
success probability of AB(α, β). To prove this lemma, we study the probability distribution of the
occurrence time ρ1 of the first comparison made by AB
t (α, β), and we examine the algorithm’s
success probability following it. Essentially, if ρ1 = s, we can compute the probability of stopping
and the corresponding success probability, and the distribution of the state of the algorithm at step
s + 1, which yields the recursion. Using adequate concentration arguments and Lemma C.1, we show
the two following results, giving explicit recursive formulas satisfied by the limit of UB
N,t,k when
N →∞, respectively for k = 2 and k = 1.
Lemma 4.1. For all B ≥0 and w ∈[α, β], the limit φB
2 (α, β; w) = limN→∞UB
N,⌊wN⌋,2 exists,
and it satisfies the following recursion

φB
2 (α, β; w) = −λw log
 
(1 −λ) w

β + λ

+
(1 −λ)βw2

(1 −λ)w + λβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) w2
Z β

w

(1 −λ)2w + λ(2 −λ)u

((1 −λ)w + λu)2u2
φB−1
2
(α, β; u)du .

Moreover, UB
N,⌊wN⌋,2 = φB
2 (α, β; w) + O
q

log N

N

.

Lemma 4.2. For all B ≥0 and w ∈[α, β], the limit φB
1 (α, β; w) = limN→∞UB
N,⌊wN⌋,1 exists,
and it satisfies the following recursion

φB
1 (α, β; w) = λw log
 
1 −λ + λ β

w

+
λwβ2

(1 −λ)w + λβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) λ2w
Z β

w

(u −w)φB−1
2
(α, β; u)
((1 −λ)w + λu)2u du .

Moreover, UB
N,⌊wN⌋,1 = φB
1 (α, β; w) + O
q

log N

N

.

We deduce that the asymptotic success probability of Algorithm AB
⌊wN⌋, conditioned on the event CN,
exists and equals φB
1 (α, β; w) + φB
2 (α, β; w). Additionally, by applying Lemma A.3, we eliminate
the conditioning on CN, thus proving the following theorem.
Theorem 4.3. For all 0 < α ≤β ≤1, The success probability of Algorithm AB(α, β) satisfies

P(AB(α,β) succeeds) −O
q

log N

N


= λα log
  β

α

+ αβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) α
Z β

α

φB−1
2
(α, β; u)du

u2
,

with φB
2 (α, β; ·) defined in Lemma 4.1.

It is possible to use Theorem 4.3 and 4.1 to numerically compute the success probability of AB(α, β).
However, this computation is heavy due the recursion defining φB
2 (α, β; w). Moreover, it is difficult
to prove a closed expression, and even more to compute the optimal thresholds.

By disregarding the term containing φ2(α, β; ·) in the theorem, we derive an analytical lower bound
expressed as a function of the parameters λ, B, α, and β, allowing a more effective threshold selection.
In the subsequent discussion, for all w ∈(0, 1] and B ≥0, we denote by SB(w) the following sum:

SB(w) =

B
X

b=0

 
1
w −

b
X

ℓ=0

log(1/w)ℓ

ℓ!

!

.

Corollary 4.3.1. Assume that λ ≥1/2. Let hB : β 7→min
n
β

e exp

βSB(β)

λ

, β
o
, and ˜αB, ˜βB

the thresholds defined as ˜αB = hB(˜βB), and ˜βB minimizing the mapping

β ∈[0, 1] 7→λh(β) log
 
β
hB(β)

+ hB(β)βSB(β) ,

then the success probability of AB(˜αB, ˜βB) satisfies

lim
N→∞P(AB(˜αB, ˜βB) succeeds) ≥1

e −min

1
e(B + 1)!, ( 4

e −1)λ(1 −λ)

.

7


---Page Break---
Therefore, in contrast to the single-threshold algorithm, the asymptotic success probability of
AB(˜αB, ˜βB) approaches 1/e both when the budget increases and when λ approaches 0 or 1.

4.1
Optimal memory-less algorithm for two groups

In the following, an algorithm is called memory-less if its actions at any step t ∈[N] depend only on
the current observations rt, gt, 1(Rt = 1), the available budget Bt, and the cardinals (|Gk
t−1|)k∈[K].

We use in this section a dynamic programming
approach to determine the optimal memory-less
algorithm, which we denote by A∗.
Unlike previous sections, our analysis here is not
asymptotic. By meticulously examining how var-
ious variables, including the precise number of
candidates observed in each group, influence the
success probability of A∗, we rigorously analyze
its state transitions and corresponding success prob-
abilities to determine optimal actions at each step.
A full description and analysis of the optimal
memory-less algorithm can be found in Section
D. Here, we illustrate its actions through Figure 2.

0
0.2
훼★
0
0.6
0.8
1.0
0

0.2

0.4

0.6

0.8

1.0

|퐺1
푡|/푁

퐵= 0

|퐺1
푡| = 휆푡

0
0.2
훼★
1
0.6
0.8
1.0
0

0.2

0.4

0.6

0.8

1.0
퐵= 1

0
0.2
훼★
2
0.6
0.8
1.0
0

0.2

0.4

0.6

0.8

1.0

Acceptance region for 퐺1

퐵= 2

0
0.2
0.4
훽★
0
0.8
1.0
step 푡/푁

0

0.2

0.4

0.6

0.8

1.0

|퐺1
푡|/푁

0
0.2
훽★
1
0.6
0.8
1.0
step 푡/푁

0

0.2

0.4

0.6

0.8

1.0

0
0.2
훽★
2
0.6
0.8
1.0
step 푡/푁

0

0.2

0.4

0.6

0.8

1.0

Acceptance region for 퐺2

Figure 2: Acceptance region of A∗

Computing optimal thresholds for the DT algorithm.
Upon observing (rt, gt), A∗makes a
decision to accept or reject, where acceptance means stop if B = 0 and compare otherwise,
depending on t, |G1
t|, B, gt. Figure 2 shows its acceptance region (dark green), with N = 500,
λ = 0.7, for B ∈{0, 1, 2} and for all possible values of t ∈[N], |G1
t| ≤t, and gt ∈{1, 2}. The x-
and y-axes display respectively the step t and possible group cardinal |G1
t|, which implies |G2
t|, up to
time t. The latter follows a binomial distribution with parameters (λ, t), which tightly concentrates
around its mean |G1
t| ≈λt (and |G2
t| ≈(1 −λ)t) even for moderate values of t. Consequently, when
N is large, |G1
t| ≈λt, and the acceptance region is solely defined by a threshold at the intersection
of the acceptance region and the line |G1
t| ≈λt. This observation implies that A∗behaves as an
instance of DT algorithms when the number of candidates is large. The corresponding thresholds,
which we denote by (α⋆
b, β⋆
b )b≤B, are necessarily optimal, and can be estimated as the intersection of
the acceptance region for Gk and the line (t, λt) for k ∈{1, 2}.

Alternative comparison model.
In the particular case of two groups, both comparison models
introduced in Section 2 are equivalent, as freely comparing a candidate with the best in their group
and then making one costly comparison with the best candidate from the other group is sufficient to
determine if they are the best so far. Therefore, all the results of the current section regarding the
static double-threshold algorithm and optimal memory-less algorithm remain true in the alternative
comparison model.

5
Numerical experiments

In this section, we confirm our theoretical findings via numerical experiments, and we give further
insight regarding the behavior of the algorithms we presented and how they compare to each other. In
all the empirical experiments of this section, each point is computed over 106 independent trials. The
code used for the experiments is available at github.com/Ziyad-Benomar/Addressing-bias-in-online-
selection-with-limited-budget-of-comparisons.

5.1
Single-threshold algorithm

Using Theorem 3.2, the optimal threshold, for the single-threshold algorithm, can be computed
numerically for fixed K and B. Figure 3 illustrates the optimal threshold and the corresponding
success probability for B ∈{0, . . . , 30} and K ∈{2, 10, 25, 50}. For any K ≥2, as the budget
grows to infinity, the problem becomes akin to the standard secretary problem, leading the optimal

8


---Page Break---
threshold to converge to 1/e. However, as discussed in Corollary 3.2.1, the convergence is slower
when the number of groups K is higher.

0
5
10
15
20
25
30
Budget of comparisons 퐵

1/e

0.5

0.6

0.8

0.95

Optimal single threshold

퐾= 2
퐾= 10
퐾= 25
퐾= 50

0
5
10
15
20
25
30
Budget of comparisons 퐵

0

0.1

0.2

0.3

1/e

Success probability

퐾= 2
퐾= 10
퐾= 25
퐾= 50

Figure 3: Single threshold algorithm: optimal threshold and corresponding success probability

Moreover, Theorem 3.2 reveals that the asymptotic success probability is independent of the prob-
abilities of belonging to each group, and it is equal to a value smaller than 1/e. This indicates a
discontinuity of the success probability at the extreme points of the polygon defining the possible
values of (λk)k∈[K]. Figure 4 illustrates this behavior for the case of two groups, with N = 500
candidates, and B ∈{0, 1, 2}. On the other hand, while our theoretical results study asymptotic
success probabilities, they do not comprehend how the performance of the algorithms varies with
the number of candidates. Figure 5 shows that the success probability is better when the number of
candidates is small, and it decreases to match the asymptotic expression when N →∞, represented
with dotted lines, for K ∈{2, 3, 4}, with B = 3.

0.0
0.2
0.4
0.6
0.8
1.0
Probability 휆∈[0, 1] of belonging to group 퐺1

0.25

0.3

0.35

1/e

퐵= 0
퐵= 1
퐵= 2

Figure 4: Single threshold: success probability
for 2 groups, with N = 500 and λ ∈[0, 1]

50
100
150
200
250
300
Number of candidates 푁

0.34

0.35

0.36

0.37

0.38
퐾= 2
퐾= 3

퐾= 4

Figure 5: Convergence to the asymptotic success
probability, with λk = 1/K for all k ∈[K]

5.2
The case of two groups

In the case of two groups, Figure 3 shows that, even with a very limited budget, the single-threshold
algorithm has a success probability almost indistinguishable from the upper bound 1/e. Consequently,
in the remaining experiments in the two-group scenario, we restrict ourselves to small budgets B ≤3.

Theorem D.4 shows a recursive formula for com-
puting the success probability of the optimal
memory-less algorithm A∗for all N ≥1, B ≥0,
and λ ∈(0, 1). Figure 6 displays this success
probability, in solid lines, for N = 500 and
B ∈{0, 1, 2}, along with the success proba-
bility of the static double-threshold algorithm
AB(˜αB, ˜βB) in dotted lines, where ˜αB, ˜βB are
defined in Corollary 4.3.1. The figure demon-
strates that for B = 0, or λ close to 0.5, Algo-
rithm AB(˜αB, ˜βB) matches the performance of
A∗, despite having a much simpler structure.

0.0
0.2
0.4
0.6
0.8
1.0
Probability 휆∈[0, 1] of belonging to 퐺1

0.26

0.3

0.34

1/푒

퐵=0
퐵=1
퐵=2

Figure 6: Success probability of A∗, and the
lower bound of Corollary 4.3.1

9


---Page Break---
For λ = 0.5, both groups have symmetric roles, and the optimal thresholds ˜αB and ˜βB to choose
in the static-threshold algorithm are identical. Hence, the success probability of AB(˜αB, ˜βB) for
λ = 0.5 is exactly that of the single-threshold algorithm, which is independent of λ (Theorem 3.2).
We deduce from this observation and Figure 6 that having different thresholds for each group yields a
substantial improvement over the single-threshold algorithm when λ is close to 0 or 1.

Finally, to emphasize that the dynamic programming algorithm A∗is equivalent to an instance of DT
algorithm for large N, Figure 7 compares the empirical success probabilities of A∗(dotted lines) and
the DT algorithm (solid lines) with thresholds (α⋆
B, β⋆
B)B≥0, computed as explained in Section 4.1.

20
50
100
150
200
250
300
Number of candidates 푁

0.26

0.28

0.30

0.32

0.34

0.36

0.38

휆= 0.5

퐵=0
퐵=1
퐵=2

20
50
100
150
200
250
300
Number of candidates 푁

0.28

0.30

0.32

0.34

0.36

0.38

휆= 0.7

퐵=0
퐵=1
퐵=2

20
50
100
150
200
250
300
Number of candidates 푁

0.350

0.355

0.360

0.365

0.370

0.375

0.380

0.385

휆= 0.95

퐵=0
퐵=1
퐵=2

Figure 7: Empirical success probabilities of A∗and the DT algorithm with optimal thresholds

For λ ∈{0.5, 0.7, 0.95}, the figure confirms that, despite the intricate structure of the optimal
memory-less algorithm, it does not surpass the performance of the DT algorithm with optimal
thresholds when N is large. Nonetheless, the analysis of the optimal memory-less algorithm is what
enables the numerical computation of the optimal thresholds, as explained previously. Figure 8 shows
the optimal thresholds α⋆
b, β⋆
b for all λ ∈[0.5, 1] and B ∈{0, 1, 2}.

0.5
0.6
0.7
0.8
0.9
1.0
Probability 휆∈[ 1
2, 1] of belonging to 퐺1

1/e

0.4

0.45

0.5
훼★
0 (휆)

훼★
1 (휆)

훼★
2 (휆)

0.5
0.6
0.7
0.8
0.9
1.0
Probability 휆∈[ 1
2, 1] of belonging to 퐺1

1/e

0.6

0.8

1.0

훽★
0 (휆)

훽★
1 (휆)

훽★
2 (휆)

Figure 8: (α⋆
b, β⋆
b ) as functions of λ.

These thresholds are continuous functions of λ, both converging to 1/e very rapidly as the budget
increases. Indeed, for B ≥1, they both become very close to 1/e, as for B = 0, the optimal
thresholds are exactly equal to α⋆
0 = λ exp( 1

λ −2) and β⋆
0 = λ, which correspond to the optimal
thresholds described in Corollary 4.3.1 for B = 0 (See the proof of the corollary).

6
Conclusion and future work

This paper explores more realistic online selection processes, wherein nuanced factors such as
imperfect comparisons and optimal budget utilization are considered. We studied a partially ordered
secretary problem, wherein a constrained budget of comparisons is allowed. Our findings encompass
both asymptotic and non-asymptotic analyses. Specifically, we explore the asymptotic behavior of the
single threshold algorithm with K groups, demonstrating its high efficiency with a non-zero budget.
Furthermore, in the context of two groups, we study the success probability of static double-threshold
algorithms, and we present a non-asymptotic optimal memoryless algorithm. Through numerical
experimentation, we demonstrate that this algorithm behaves as a DT algorithm, and, leveraging this
insight, we show how to numerically compute the optimal DT thresholds. However, a limitation of
the paper is that optimal thresholds are only computed in the case of two groups. Future investigation
could explore methods for numerically or analytically characterizing optimal DT thresholds with an
arbitrary number of groups K.

10


---Page Break---
Acknowledgements

This research was supported in part by the French National Research Agency (ANR) in the framework
of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant DOOM ANR-
23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601). Views and
opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union or the European Research Council Executive Agency. Neither the European Union
nor the granting authority can be held responsible for them.

References

Antonios Antoniadis, Themis Gouleakis, Pieter Kleer, and Pavel Kolev. Secretary and online matching
problems with machine learned advice. Advances in Neural Information Processing Systems, 33:
7933–7944, 2020.

Antonios Antoniadis, Joan Boyar, Marek Eliás, Lene Monrad Favrholdt, Ruben Hoeksma, Kim S
Larsen, Adam Polak, and Bertrand Simon. Paging with succinct predictions. In International
Conference on Machine Learning, pages 952–968. PMLR, 2023.

Sepehr Assadi, Eric Balkanski, and Renato Leme. Secretary ranking with minimal inversions.
Advances in Neural Information Processing Systems, 32, 2019.

Yossi Azar, Ashish Chiplunkar, and Haim Kaplan. Prophet secretary: Surpassing the 1-1/e barrier. In
Proceedings of the 2018 ACM Conference on Economics and Computation, pages 303–318, 2018.

Xingjian Bai and Christian Coester. Sorting with predictions. Advances in Neural Information
Processing Systems, 36, 2024.

Xiaohui Bei and Shengyu Zhang. The secretary problem with competing employers on random
edge arrivals. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages
4818–4825, 2022.

Ziyad Benomar and Christian Coester.
Learning-augmented priority queues.
arXiv preprint
arXiv:2406.04793, 2024.

Ziyad Benomar and Vianney Perchet. Non-clairvoyant scheduling with partial predictions. In
Forty-first International Conference on Machine Learning.

Ziyad Benomar and Vianney Perchet. Advice querying under budget constraint for online algorithms.
In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Ziyad Benomar, Dorian Baudry, and Vianney Perchet. Lookback prophet inequalities. arXiv preprint
arXiv:2406.06805, 2024.

Aditya Bhaskara, Ashok Cutkosky, Ravi Kumar, and Manish Purohit. Logarithmic regret from
sublinear hints. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang,
and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34:
Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December
6-14, 2021, virtual, pages 28222–28232, 2021. URL https://proceedings.neurips.cc/
paper/2021/hash/edb947f2bbceb132245fdde9c59d3f59-Abstract.html.

Archit Bubna and Ashish Chiplunkar. Prophet inequality: Order selection beats random order. In
Proceedings of the 24th ACM Conference on Economics and Computation, pages 302–336, 2023.

Shuchi Chawla, Jason D Hartline, David L Malec, and Balasubramanian Sivan. Multi-parameter
mechanism design and sequential posted pricing. In Proceedings of the forty-second ACM sympo-
sium on Theory of computing, pages 311–320, 2010.

Y.S. Chow, H.E. Robbins, H. Robbins, and D. Siegmund. Great Expectations: The Theory of Optimal
Stopping. Houghton Mifflin, 1971. ISBN 9780395053140. URL https://books.google.fr/
books?id=A6ELhyczVf4C.

11


---Page Break---
Jose Correa, Andres Cristi, Paul Duetting, and Ashkan Norouzi-Fard. Fairness and bias in online
selection. In International conference on machine learning, pages 2112–2121. PMLR, 2021a.

José Correa, Andrés Cristi, Laurent Feuilloley, Tim Oosterwijk, and Alexandros Tsigonias-
Dimitriadis. The secretary problem with independent sampling. In Proceedings of the 2021
ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2047–2058. SIAM, 2021b.

Chen Dang, Cristina Bazgan, Tristan Cazenave, Morgan Chopin, and Pierre-Henri Wuillemin. Warm-
starting nested rollout policy adaptation with optimal stopping. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 37, pages 12381–12389, 2023.

John P Dickerson, Karthik Abinav Sankararaman, Aravind Srinivasan, and Pan Xu. Balancing
relevance and diversity in online bipartite matching via submodularity. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 33, pages 1877–1884, 2019.

Marina Drygala, Sai Ganesh Nagarajan, and Ola Svensson. Online algorithms with costly predictions.
In International Conference on Artificial Intelligence and Statistics, pages 8078–8101. PMLR,
2023.

Paul Dütting, Silvio Lattanzi, Renato Paes Leme, and Sergei Vassilvitskii. Secretaries with advice. In
Proceedings of the 22nd ACM Conference on Economics and Computation, EC ’21, page 409–429,
New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450385541. doi:
10.1145/3465456.3467623. URL https://doi.org/10.1145/3465456.3467623.

Evgenii Borisovich Dynkin. The optimum choice of the instant for stopping a markov process. Soviet
Mathematics, 4:627–629, 1963.

Michal Feldman, Nick Gravin, and Brendan Lucier. Combinatorial auctions via posted prices. In
Proceedings of the twenty-sixth annual ACM-SIAM symposium on Discrete algorithms, pages
123–135. SIAM, 2014.

Ragnar Freij and Johan Wästlund. Partially ordered secretaries. Electronic Communications in
Probability, 15:504–507, 2010.

Kaito Fujii and Hisashi Kashima. Budgeted stream-based active learning via adaptive submodular
maximization. Advances in Neural Information Processing Systems, 29, 2016.

John P. Gilbert and Frederick Mosteller. Recognizing the Maximum of a Sequence, pages 355–
398. Springer New York, New York, NY, 2006. ISBN 978-0-387-44956-2. doi: 10.1007/
978-0-387-44956-2_22. URL https://doi.org/10.1007/978-0-387-44956-2_22.

Mohak Goyal. Secretary matching with vertex arrivals and no rejections. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 36, pages 5051–5058, 2022.

Mohammad Taghi Hajiaghayi, Robert Kleinberg, and Tuomas Sandholm. Automated online mecha-
nism design and prophet inequalities. In AAAI, volume 7, pages 58–65, 2007.

M Hlynka and JN Sheahan. The secretary problem for a random walk. Stochastic Processes and
their applications, 28(2):317–325, 1988.

Sungjin Im, Ravi Kumar, Aditya Petety, and Manish Purohit. Parsimonious learning-augmented
caching. In International Conference on Machine Learning, pages 9588–9601. PMLR, 2022.

Kevin Jamieson, Matthew Malloy, Robert Nowak, and Sébastien Bubeck. lil’ucb: An optimal
exploration algorithm for multi-armed bandits. In Conference on Learning Theory, pages 423–439.
PMLR, 2014.

Zhihao Jiang, Pinyan Lu, Zhihao Gavin Tang, and Yuhao Zhang. Online selection problems against
constrained adversary. In International Conference on Machine Learning, pages 5002–5012.
PMLR, 2021.

Anna Karlin and Eric Lei. On a competitive secretary problem. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 29, 2015.

12


---Page Break---
Thomas Keller and Florian Geißer. Better be lucky than good: Exceeding expectations in mdp
evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.

Douglas P Kennedy. Prophet-type inequalities for multi-choice optimal stopping. Stochastic Processes
and their applications, 24(1):77–88, 1987.

Robert Kleinberg and Seth Matthew Weinberg. Matroid prophet inequalities. In Proceedings of the
forty-fourth annual ACM symposium on Theory of computing, pages 123–136, 2012.

Robert D Kleinberg. A multiple-choice secretary algorithm with applications to online auctions. In
SODA, volume 5, pages 630–631. Citeseer, 2005.

Ulrich Krengel and Louis Sucheston. Semiamarts and finite values. 1977.

D. V. Lindley. Dynamic programming and decision theory. Journal of the Royal Statistical Society.
Series C (Applied Statistics), 10(1):39–51, 1961.
ISSN 00359254, 14679876.
URL http:
//www.jstor.org/stable/2985407.

George E Monahan. Optimal stopping in a partially observable markov process with costly informa-
tion. Operations Research, 28(6):1319–1334, 1980.

George E Monahan. State of the art—a survey of partially observable markov decision processes:
theory, models, and algorithms. Management science, 28(1):1–16, 1982.

Manish Raghavan, Solon Barocas, Jon Kleinberg, and Karen Levy. Mitigating bias in algorithmic
hiring: Evaluating claims and practices. In Proceedings of the 2020 Conference on Fairness,
Accountability, and Transparency, FAT* ’20, page 469–481, New York, NY, USA, 2020. Associa-
tion for Computing Machinery. ISBN 9781450369367. doi: 10.1145/3351095.3372828. URL
https://doi.org/10.1145/3351095.3372828.

Sivan Sabato and Tom Hess. Interactive algorithms: from pool to stream. In Conference on Learning
Theory, pages 1419–1439. PMLR, 2016.

Sivan Sabato and Tom Hess. Interactive algorithms: Pool, stream and precognitive stream. Journal
of Machine Learning Research, 18(229):1–39, 2018.

Karim Ahmed Abdel Sadek and Marek Elias. Algorithms for caching and MTS with reduced number
of predictions. In The Twelfth International Conference on Learning Representations, 2024. URL
https://openreview.net/forum?id=QuIiLSktO4.

Jad Salem and Swati Gupta. Closing the gap: Group-aware parallelization for the secretary problem
with biased evaluations. Available at SSRN 3444283, 2019.

Jad Salem, Deven R Desai, and Swati Gupta. Don’t let ricci v. destefano hold you back: A bias-aware
legal solution to the hiring paradox. arXiv preprint arXiv:2201.13367, 2022.

Ester Samuel-Cahn. Comparison of threshold stop rules and maximum for independent nonnegative
random variables. the Annals of Probability, pages 1213–1216, 1984.

Sandeep Silwal, Sara Ahmadian, Andrew Nystrom, Andrew McCallum, Deepak Ramachandran, and
Mehran Kazemi. Kwikbucks: Correlation clustering with cheap-weak and expensive-strong signals.
In Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing
(SustaiNLP), pages 1–31, 2023.

13


---Page Break---
A
Preliminaries

With the assumption that the group membership of each candidate is a random variable, to compute
the asymptotic success probabilities of the algorithms presented in the paper, it is necessary to use
concentration inequalities to estimate the number of candidates in each group. We use Lemma 3 from
Jamieson et al. [2014] to prove the following.
Lemma A.1 (Jamieson et al. [2014]). Let (Xt)t≥1 be i.i.d. Bernoulli random variables, N a positive
integer, and m > 0 satisfying N m > 8, then it holds with a probability of at least 1 −
25
N2m that

∀t ≥1 :

t
X

s=1
(Xs −E[Xs]) ≤2
p

(m + 1)t log N

Proof. Bernoulli random variables are sub-Gaussian with scale parameter 1/2, hence Lemma 3 from
Jamieson et al. [2014] with ε = 1 guarantees that, with a probability of at least 1 −3(
δ
log 2)2, the
following holds

∀t ≥1 :

t
X

s=1
(Xs −E[Xs]) ≤2

r

t log

log(2t)

δ

.

Consider positive integers T ≤N, m ≥1 such that N m > 8, and δ = 2/N m. Then for all
t ∈{T, . . . , N}
log(2t)

δ
≤2t

δ = N mt ≤N m+1

and we deduce that

P

∀t ≥1 :

t
X

s=1
(Xs −E[Xs]) ≤2
p

(m + 1)t log N


≥P

 

∀t ≥1 :

t
X

s=1
(Xs −E[Xs]) ≤2

r

t log

log(2t)

2/N m
!

≥1 −3(2/ log 2)2

N 2m

≥1 −
25
N 2m .

In the context of the K-group secretary problem, adequately using the previous Lemma and using
union bounds yields a concentration inequality on the number of candidates belonging to each group.
Lemma A.2. Let N ≥max(4, K), and consider the following event

∀k ∈[K], ∀t ≥1 : ||Gk
t | −λkt| ≤4
p

t log N ,

which we denote by CN. Then it holds that P(CN) ≥1 −
1
N2 .

Proof. The previous Lemma with N ≥4, m = 3 and respectively with the Bernoulli random
variables Xt,k,0 = 1(gt = k) and Xt,k,1 = 1 −1(gt = k), gives for all i ∈{0, 1} and k ∈[K] that

P

∀t ≥1 : (−1)i(|Gk
t | −λkt) ≤4
p

t log N

≥1 −25

N 6 ,

and we obtain by a union bound that P(CN) ≥1 −50K

N6 . Given that N ≥max(4, K), it follows that

P(CN) ≥1 −K

N · 50

N 3 · 1

N 2 ≥1 −1

N 2 .

Lemma A.3. Let E be any event, not necessarily independent of CN (defined in Lemma A.2), then

P(E | CN) = P(E) + O(1/N 2) .

14


---Page Break---
Proof. We have by Lemma A.2 that

P(E | CN) = P(E ∩CN)

P(CN)
≤
P(E)
1 −
1
N 2
≤

1 + 2

N 2


P(E) ≤P(E) + 2

N 2 .

Using the same inequality with the complementary event Ec of E gives

P(E | CN) = 1 −P(Ec | CN) ≥1 −

P(Ec) + 2

N 2


= P(E) −2

N 2 ,

which concludes the proof.

Lemma A.4. For any x > 0 and for any B ≥1 we have

0 ≤xex −

B
X

b=0

 

ex −

b
X

ℓ=0

xℓ

ℓ!

!

≤ex
xB+2

(B + 1)!.

Proof. Let x > 0 and B ≥1. First, observe that

∞
X

b=0

 

ex −

b
X

ℓ=0

xℓ

ℓ!

!

=

∞
X

b=0

∞
X

ℓ=b+1

xℓ

ℓ! =

∞
X

ℓ=1

ℓ−1
X

b=0

xℓ

ℓ! =

∞
X

ℓ=1

xℓ

(ℓ−1)! = x

∞
X

ℓ=0

xℓ

ℓ! = xex,

therefore
B
X

b=0

 

ex −

b
X

ℓ=0

xℓ

ℓ!

!

≤xex,

and we have

xex −

B
X

b=0

 

ex −

b
X

ℓ=0

xℓ

ℓ!

!

=

∞
X

b=B+1

∞
X

ℓ=b+1

xℓ

ℓ!
=

∞
X

ℓ=B+2

ℓ−1
X

b=B+1

xℓ

ℓ!
=

∞
X

ℓ=B+2
(ℓ−B −1)xℓ

ℓ!

≤

∞
X

ℓ=B+2

xℓ

(ℓ−1)!
≤x

∞
X

ℓ=B+1

xℓ

ℓ!
≤ex
xB+2

(B + 1)!,

where we used for the last step the classical inequality

∞
X

ℓ=B+1

xℓ

ℓ! = ex −

B
X

ℓ=0

xℓ

ℓ! ≤ex
xB+1

(B + 1)!.

B
Single-Threshold algorithms for K groups

B.1
Proof of Lemma 3.1

Proof. We consider that B ≥1. The case B = 0 is treated separately at the end of the proof. Let CN
the event (∀k ∈[K], ∀t ≥1 : ||Gk
t | −λkt| ≤4√t log N). It holds that

P(AB
T succeeds | CN) =

N
X

t=T

K
X

k=1

K
X

ℓ=1
P(AB
T succeeds, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

=

N
X

t=T

K
X

k=1

K
X

ℓ=1
P(AB
T succeeds, Rt = 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

+

N
X

t=T

K
X

k=1

K
X

ℓ=1
P(AB
T succeeds, Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN) .

(3)

15


---Page Break---
In the following, we will estimate the terms in both sums. We recall that we consider K and (λk)k∈[K]
to be constants, and T = αN + o(1), with α also a constant. All the O terms appearing in the proof
are independent of t, as T ≤t ≤N, they only depend on N, T/N = α + o(1), and the constant
parameters.

For t ∈{T, . . . , N}, if ρ1 = t and Rt = 1, then the Algorithms stops on xt, hence its succeeds if and
only if xt = xmax. The event xt = xmax is independent of the group membership of the candidates,
thus independent of CN, and its probability is 1/N. The event gt = ℓ, however, is not independent
of CN, but Lemma A.3 gives that P(gt = ℓ| CN) = P(gt = ℓ) + O(1/N 2) = λℓ+ O(1/N 2).
Therefore, it holds for all k, ℓ∈[K] and t ∈{T, . . . , N} that

P(AB
T succeeds, Rt = 1,ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)
= P(xt = xmax, ρ1 ≥t, gt = ℓ, g∗
T −1 = k | CN)
= P(xt = xmax)P(gt = ℓ| CN)P(ρ1 ≥t, g∗
T −1 = k | CN)

=
λℓ

N + O(1/N 3)

P(ρ1 ≥t, g∗
T −1 = k | CN) .

Note that, for the single-threshold algorithm, we have the equivalence ρ1 = t ⇐⇒ρ1 ≥t and rt =
1. The event ρ1 ≥t happens if and only if no candidate xs for s ∈{T, . . . , t −1} in any group
m ∈[K] exceeds the best candidate seen up to time T −1 in the same group:

∀t ≥T : (ρ1 ≥t) ⇐⇒(∀m ∈[K] : max Gm
T :t−1 < max Gm
T −1) ,

with the convention max ∅= −∞. Consequently, if ρ1 ≥t, then g∗
T −1 = k means that the best
candidate in all groups until time t −1 belongs to group Gk
T −1. Using that T = Θ(N), this yields

P(AB
T succeeds, Rt = 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

=
λℓ

N + O(1/N 3)

P(ρ1 ≥t and max x1:t−1 ∈Gk
T −1 | CN)

=
λℓ

N + O(1/N 3)

P(max x1:t−1 ∈Gk
T −1 | CN)P(ρ1 ≥t | max x1:t−1 ∈Gk
T −1, CN)

=
λℓ

N + O(1/N 3)
 λk(T −1)

t −1
+ O(1/N 2)

P(ρ1 ≥t | max x1:t−1 ∈Gk
T −1, CN)

=
λℓλkT

Nt
+ O(1/N 3)

P(∀m ∈[K] \ {k} : max Gm
T :t−1 < max Gm
T −1 | CN)

=
λℓλkT

Nt
+ O(1/N 3)
 Y

m̸=k
E
h |Gk
T −1|
|Gk
t−1|

 CN
i

=
λℓλkT

Nt
+ O(1/N 3)
 Y

m̸=k
E
λmT + O(√N log N)

λmt + O(√N log N)

 CN



=
λℓλkT

Nt
+ O(1/N 3)
 Y

m̸=k

T

t + O
q

log N

N


=
λℓλkT

Nt
+ O(1/N 3)
 T K−1

tK−1 + O
q

log N

N


= λℓλk

N (T/t)K + O
q

log N

N3

.
(4)

On the other hand, regarding the terms of the second sum in (3), if ρ1 = t but Rt ̸= 1, the Algorithm
uses a comparison to observe Rt but then skips to the next step t + 1. The budget at step t + 1 is
thus B −1 and the group of the best candidate seen so far remains unchanged. Given that the single
threshold algorithm is memory-less, its state at time t + 1 is fully determined by B −1, g∗
t and the

16


---Page Break---
number of candidates seen in each group so far, which is controlled by CN. We deduce that

P(AB
T succeeds, Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

= P(AB
T succeeds | Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k, CN)
(5)
× P(Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

= P(AB−1
t+1 succeeds | g∗
t = k, CN)P(Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN) ,
(6)

where P(AB−1
N+1 succeeds) = 0. For ℓ= k, the probability P(Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k |
CN) is zero, because if g∗
T −1 = k, ρ1 ≥t and gt = ℓ, then the best candidate up to step T −1
belongs to group Gk, and no candidate xs for s ∈{T, . . . , t −1} is better than the maximum in its
group seen before step T −1, thus if xt belongs to Gk and rt = 1 then necessarily Rt = 1.

For ℓ̸= k, it holds that

P(Rt ̸= 1,ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

= P(ρ1 = t, gt = ℓ, max x1:t ∈Gk
T −1 | CN)

= P(max x1:t ∈Gk
T −1 | CN)P(ρ1 = t, gt = ℓ| max x1:t ∈Gk
T −1, CN)

=
λk(T −1)

t −1
+ O(1/N 2)

P(ρ1 = t, gt = ℓ| max x1:t ∈Gk
T −1, CN)

=
λkT

t
+ O(1/N 2)

P(gt = ℓ| CN)P(max Gℓ
T :t−1 < max Gℓ
T −1 < xt

and ∀m ∈[K] \ {k, ℓ} : max Gm
T :t−1 < max Gm
T −1 | CN)

=
λkT

t
+ O(1/N 2)

(λℓ+ O(1/N 2))E
h
1
|Gℓ
t−1|+1 ·
|Gℓ
T −1|
|Gℓ
t−1|

 CN
i
Y

m/∈{k,ℓ}
E
h |Gm
t−1|
|Gm
T −1|
 CN
i

=
λℓλkT

t
+ O(1/N 2)

E
h
1
|Gℓ
t−1|+1 ·
|Gℓ
T −1|
|Gℓ
t−1|

 CN
i
Y

m/∈{k,ℓ}
E
h |Gm
t−1|
|Gm
T −1|
 CN
i

=
λℓλkT

t
+ O(1/N 2)
  T

λℓt2 + O
q

log N

N3

Y

m/∈{k,ℓ}

T

t + O
q

log N

N


=
λℓλkT

t
+ O(1/N 2)
  T

λℓt2 + O
q

log N

N3
 T K−2

tK−2 + O
q

log N

N


= λkT K

tK+1 + O
q

log N

N3

,

where we used in the last equations the event CN and the assumption T = Θ(N). Therefore,
substituting into (6) gives for all ℓ, k ∈[K] and t ∈{T, . . . , N} that

P(AB
T succeeds,Rt ̸= 1, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

=
λkT K

tK+1 + O
q

log N

N3

1(k ̸= ℓ) P(AB−1
t+1 succeeds | g∗
t = k, CN)

=
λkT K

tK+1 + O
q

log N

N3

1(k ̸= ℓ) P(AB−1
t+1 succeeds, g∗
t = k | CN)
P(g∗
t = k | CN)

=
λkT K

tK+1 + O
q

log N

N3

1(k ̸= ℓ) P(AB−1
t+1 succeeds, g∗
t = k | CN)
λk + O(1/N 2)

=
 T K

tK+1 + O
q

log N

N3

1(k ̸= ℓ) P(AB−1
t+1 succeeds, g∗
t = k | CN) .
(7)

17


---Page Break---
Finally, substituting (4) and (7) into (3), and recalling that all the previous O terms are independent
of t, gives that

P(AB
T succeeds | CN) =

N
X

t=T

K
X

k=1

K
X

ℓ=1

λℓλk

N (T/t)K + O
q

log N

N3


+

N
X

t=T

K
X

k=1

X

ℓ̸=k

 T K

tK+1 + O
q

log N

N3

P(AB−1
t+1 succeeds, g∗
t = k | CN)

=

1 + O
q

log N

N
 
N
X

t=T

K
X

k=1

K
X

ℓ=1

λℓλk

N (T/t)K

+

N
X

t=T

K
X

k=1

X

ℓ̸=k

T K

tK+1 P(AB−1
t+1 succeeds, g∗
t = k | CN)


=

1 + O
q

log N

N
 T K

N

N
X

t=T

1
tK + (K −1)

N
X

t=T

T K

tK+1 P(AB−1
t+1 succeeds | CN)

.

Using Riemann sum properties, we obtain

T K

N

N
X

t=T

1
tK = (T/N)K

N

N
X

t=T

1
(t/N)K = αK
Z 1

α

du
uK + O(1/N) = α −αK

K −1 + O(1/N) ,

and by Lemma A.3 we have for all t ∈{T, . . . , N} and b ≥0 that

P(Ab
t succeeds | CN) = P(Ab
t succeeds) + O(1/N 2) ,

with the O(1/N 2) independent of t. Observing that PN
t=T
T K
tK+1 = 1−αK

K
+ o(1) = O(1), it follows
that

P(AB
T succeeds) =

1 + O
q

log N

N
 α −αK

K −1 + (K −1)

N
X

t=T

T K

tK+1 P(AB−1
t+1 succeeds) + O( 1

N )


= α −αK

K −1 + (K −1)

N
X

t=T

T K

tK+1 P(AB−1
t+1 succeeds) + O
q

log N

N

.

This concludes the proof for B ≥1.

For B = 0, P(A0
t+1 succeeds | CN) can be decomposed as in (3). However, the terms of the second
sum are all zero, because if ρ1 = t then the algorithm stops at t, but since Rt ̸= 1, the selected
candidate is not the best one, and thus the succeeding probability is 0. All the computations regarding
the first sum stay the same, and we obtain

P(A0
T succeeds) = α −αK

K −1 + O
q

log N

N

.

B.2
Proof of Theorem 3.2

Proof. Let α ∈(0, 1] a constant. For all w ∈[α, 1] and B ≥0, we denote by φB(w) the limit
limN→∞P(AB
t succeeds) for t = ⌊wN⌋. We will prove by induction over B that this limit exists
for all w ∈[α, 1], is equal to the expression stated in the theorem, with u instead of α, and satisfies

P(AB
t succeeds) = φB(w) + O
 q

log N

N

, with the O term only depending on α and the other
constants of the problem. In particular, the O term is independent of t. For B = 0, Lemma 3.1 gives
immediately for any w ∈[α, 1] and t = ⌊wN⌋that

P(A0
t succeeds) = w −wK

K −1 + O
q

log N

N

=
wK

K −1


1
wK−1 −1

+ O
q

log N

N

.

18


---Page Break---
The O term depends on t, but using the inequalities α + o(1) ≤t/N ≤1, it can be made only
dependent on α. Let B ≥1 and assume the result is true for B −1. Lemma 3.1 and the induction
hypothesis give for all w ∈[α, 1] and t = ⌊wN⌋, that

P(AB
t succeeds) = w −wK

K −1 + (K −1)

N
X

s=t

tK

sK+1 P(AB−1
s+1 succeeds) + O
q

log N

N


= w −wK

K −1 + (K −1)

N
X

s=t

tK

sK+1


φB−1  s+1

N

+ O
q

log N

N

+ O
q

log N

N


= w −wK

K −1 + (K −1)(t/N)K

N

N
X

s=t

φB−1  s+1

N


(s/N)K+1 + O
q

log N

N

,

where we used that the O term in the induction hypothesis is independent of s and that

N
X

s=t

tK

sK+1 φB−1  s+1

N

≤

N
X

s=T

N K

sK+1 ≤1

N

N
X

s=T

1
(s/N)K+1 = O(1) .

Finally, t/N = w +O(1/N), and φB−1 is, by the induction hypothesis, a continuously differentiable
function on [α, 1], therefore, it holds by convergence properties of Riemann sums that

P(AB
t succeeds) = w −wK

K −1 + (K −1)(wK + O( 1

N ))
Z 1

w

φB−1(u)

uK+1
du + O( 1

N )

+ O
q

log N

N


= w −wK

K −1 + (K −1)wK
Z 1

w

φB−1(u)

uK+1
du + O
q

log N

N

,

where the O term depends on t and the constant parameters. Using that T = ⌊αN⌋≤t ≤N,
the O can be made dependent only on α and the other constant parameters. The limit φB(w) =
limN→∞P(AB
⌊wN⌋succeeds) therefore exists, and is equal to

φB(w) = w −wK

K −1 + (K −1)wK
Z 1

w

φB−1(u)

uK+1
du .

The induction hypothesis gives for all u ∈[α, 1] that

φB−1(u) =
uK

K −1

B−1
X

b=0

 
1
uK−1 −

b
X

ℓ=0

log(1/uK−1)ℓ

ℓ!

!

,

hence

(K −1)wK
Z 1

w

φB−1(u)

uK+1
du =
Z 1

w

B−1
X

b=0

 
1
uK −

b
X

ℓ=0

log(1/uK−1)ℓ

ℓ!u

!

du

= BwK
Z 1

w

du
uK −wK
B−1
X

b=0

b
X

ℓ=0

1
ℓ!

Z 1

w

log(1/uK−1)ℓ

u
du

= BwK

−1
(K −1)uK−1

1

w
−wK
B−1
X

b=0

b
X

ℓ=0

1
ℓ!


−log(1/uK−1)ℓ+1

(K −1)(ℓ+ 1)

1

w

= BwK

K −1


1
wK−1 −1

−wK
B−1
X

b=0

b
X

ℓ=0

log(1/wK−1)ℓ+1

(K −1)(ℓ+ 1)!

= BwK

K −1


1
wK−1 −1

−wK
B
X

b=1

b
X

ℓ=1

log(1/wK−1)ℓ

(K −1)ℓ!

=
wK

K −1

B
X

b=1

 
1
wK−1 −1 −

b
X

ℓ=1

log(1/wK−1)ℓ

ℓ!

!

=
wK

K −1

B
X

b=1

 
1
wK−1 −

b
X

ℓ=0

log(1/wK−1)ℓ

ℓ!

!

,

19


---Page Break---
and it follows that

φB(w) =
wK

K −1

 
1
wK−1 −1 +

B
X

b=1

 
1
wK−1 −

b
X

ℓ=0

log(1/wK−1)ℓ

ℓ!

!!

=
wK

K −1

B
X

b=0

 
1
wK−1 −

b
X

ℓ=0

log(1/wK−1)ℓ

ℓ!

!

.

In particular, this identity is true for w = α, which gives the wanted result.

Infinite budget
Taking the limit for B →∞, we obtain that
lim
B→∞lim
N→∞P(AB
⌊αN⌋succeeds) = lim
B→∞φB(w)

=
αK

K −1

∞
X

b=0

 
1
αK−1 −

b
X

ℓ=0

log(1/αK−1)ℓ

ℓ!

!

=
αK

K −1

∞
X

b=0

∞
X

ℓ=b+1

log(1/αK−1)ℓ

ℓ!

=
αK

K −1

∞
X

ℓ=1

ℓ−1
X

b=0

log(1/αK−1)ℓ

ℓ!

=
αK

K −1

∞
X

ℓ=1

log(1/αK−1)ℓ

(ℓ−1)!

= αK log(1/αK−1)

K −1

∞
X

ℓ=0

log(1/αK−1)ℓ

ℓ!

= αK log(1/α) ·
1
αK−1
= α log(1/α) .

B.3
Proof of Corollary 3.2.1

Proof. Let α ∈(0, 1) and T = ⌊αN⌋. Lemma A.4 gives for all x > 0 that

B
X

b=0

 

ex −

b
X

ℓ=0

xℓ

ℓ!

!

≥xex

1 −
xB+1

(B + 1)!


,

in particular, we obtain for x = log(1/αK−1) that

lim
N→∞P(AB
T succeeds) =
αK

K −1

∞
X

b=0

 
1
αK−1 −

b
X

ℓ=0

log(1/αK−1)ℓ

ℓ!

!

≥
αK

K −1 · log(1/αK−1)

αK−1


1 −log(1/αK−1)B+1

(B + 1)!



= α log(1/α)

1 −(K −1)B+1 log(1/α)B+1

(B + 1)!


.

Taking a threshold T = ⌊N/e⌋gives

lim
N→∞P(AB
⌊N/e⌋succeeds) ≥1

e


1 −(K −1)B+1

(B + 1)!


.

To achieve an asymptotic success probability of at least 1−ε

e
for some ε > 0, using the inequality

m! ≥e
  m

e
m, it suffices that K and B satisfy
(K−1)B+1

e( B+1

e
)B+1 ≤ε, which is equivalent to

K ≤1 + B + 1

e
(eε)
1
B+1 .

20


---Page Break---
C
Static Double-threshold algorithm for two groups

C.1
Recursion lemma

We first prove a recursion satisfied by UB
N,t,k (2), which we use to prove the subsequent results in this
section.

Lemma C.1. For all B ≥0, t ∈{⌊αN⌋, . . . , ⌊βN⌋−1}, and k ∈{1, 2}, UB
N,t,k satisfies

UB
N,t,k = λ

N

⌊βN⌋−1
X

s=t
P(ρ1 ≥s, g∗
t−1 = k | CN)

+ 1(B > 0)

1 −λ

⌊βN⌋−1
X

s=t
P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)UB−1
N,s+1,2

+ P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 ≥βN | CN) + O
  1

N

.

Assuming that ρ1 = s ∈{t, . . . , ⌊βN⌋−1}, the first sum corresponds to the success probability if
Rs = 1 and the algorithm selects the candidate xs. The terms of the second sum represent the success
probability after using a comparison at step s but observing Rt ̸= 1, resulting in the rejection of the
candidate. Therefore, the available budget at step s + 1 is B −1, and necessarily g∗
s = 2, because a
comparison at step s can only occur if gs = 1 by definition of the algorithm. Hence, only the term
UB−1
N,s+1,2 appears in the recursion, not UB−1
N,s+1,1. Finally, the last term represents the probability of
success if no comparison has been made before step ⌊βN⌋

Proof. Let B ≥0. For all t ∈{⌊αN⌋, . . . , ⌊βN⌋−1) and k ∈{1, 2}, it holds that

UB
N,t,k = P(AB
t (α, β) succeeds, g∗
t−1 = k | CN)

=

⌊βN⌋−1
X

s=t
P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 = s | CN)

+ P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 ≥βN | CN)

=

⌊βN⌋−1
X

s=t
P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 = s, Rs = 1 | CN)
(8)

+

⌊βN⌋−1
X

s=t
P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)
(9)

+ P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 ≥βN | CN) .
(10)

For all s ∈{t, . . . , ⌊βN⌋−1}, by definition of Algorithm AB
t (α, β), if ρ1 = s and Rs = 1 then
the candidate xs is selected, and the algorithm succeeds if only if xs = xmax. Moreover, the event
{ρ1 = s} is equivalent to {ρ1 ≥s, gs = 1, rs = 1}, hence the terms in (8) can be written as

P(AB
t (α, β) succeeds, g∗
t−1 = k,ρ1 = s, Rs = 1 | CN)
= P(xs = xmax, g∗
t−1 = k, ρ1 = s, Rs = 1 | CN)
= P(xs = xmax, g∗
t−1 = k, ρ1 ≥s, gs = 1 | CN)

= P(gs = 1 | CN)

N
P(ρ1 ≥s, g∗
t−1 = k | CN)

=
 λ

N + O
  1

N 3

P(ρ1 ≥s, g∗
t−1 = k | CN) ,

21


---Page Break---
where used for that the event {xs = xmax} is independent of the group memberships, thus indepen-
dent of CN, and that it is also independent of {ρ1 ≥s, g∗
t−1 = k}, because a realization of the latter
event is determined only by the groups and relative ranks of the candidates {x1, . . . , xs−1}. For the
last equality, we used Lemma A.3.

Secondly, in the case where ρ1 = s and Rs ̸= 1, if B = 0 then the algorithm selects candidate xs
which is not the best overall, hence its probability of succeeding is zero. If B ≥1, the algorithm
makes a comparison but then rejects the candidate. Moreover, for s ∈[t, βN], if ρ1 = s then
necessarily gs = 1, and having Rs ̸= 1 implies that g∗
s = 2. The success probability of AB
t (α, β)
given that ρ1 = s, Rs ̸= 1 is the same as the success probability of AB−1
s+1 (α, β) given that g∗
s = 2.
Therefore, the terms of (9) satisfy

P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)

= 1(B > 0) P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)P(AB
t (α, β) succeeds | g∗
t−1 = k, ρ1 = s, Rs ̸= 1, CN)

= 1(B > 0) P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)P(AB−1
s+1 (α, β) succeeds | g∗
s = 2, CN)

= 1(B > 0) P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)P(AB−1
s+1 (α, β) succeeds, g∗
s = 2 | CN)
P(g∗s = 2 | CN)

= 1(B > 0) P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)
UB−1
N,s+1,2
1 −λ + O( 1

N 2 ) ,

where we used again Lemma A.3. Given that the O terms are independent of s, We deduce that

UB
N,t,k =
 λ

N + O
  1

N3
 ⌊βN⌋−1
X

s=t
P(ρ1 ≥s, g∗
t−1 = k | CN)

+ 1(B > 0)

1
1 −λ + O
  1

N2
 ⌊βN⌋−1
X

s=t
P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)UB−1
N,s+1,2

+ P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 ≥βN | CN)

= λ

N

⌊βN⌋−1
X

s=t
P(ρ1 ≥s, g∗
t−1 = k | CN)

+ 1(B > 0)

1 −λ

⌊βN⌋−1
X

s=t
P(g∗
t−1 = k, ρ1 = s, Rs ̸= 1 | CN)UB−1
N,s+1,2

+ P(AB
t (α, β) succeeds, g∗
t−1 = k, ρ1 ≥βN | CN) + O
  1

N

.

C.2
Additional lemmas

In the following two lemmas, we compute the probabilities appearing in Lemma C.1 for all s ∈
{t, . . . , ⌊βN⌋−1 and k ∈{1, 2}

Lemma C.2. Let ⌊αN⌋≤t ≤s < ⌊βN⌋, and consider a run of Algorithm AB
t (α, β), then it holds
that

P(ρ1 ≥s, g∗
t−1 = 1 | CN) =
λt
(1 −λ)t + λs + O
q

log N

N


P(ρ1 ≥s, g∗
t−1 = 2 | CN) =
(1 −λ)t2

((1 −λ)t + λs)s + O
q

log N

N

.

Proof. Since there are only 2 groups, the event g∗
t−1 = 1 is equivalent to max G2
t−1 < max G1
t−1.
For s ∈{t, . . . , ⌊βN⌋}, Algorithm AB
t (α, β) only makes a comparison (or stops in the case of
B = 0) at step s only if gs = 1 and rs = 1. Therefore, ρ1 ≥s if and only if no candidate belonging

22


---Page Break---
to G1
t:s−1 surpasses the maximum value observed in G1
t−1
P(ρ1 ≥s, g∗
t−1 = 1 | CN) = P(max G1
t:s−1 < G1
t−1, max G2
t−1 < max G1
t−1 | CN)

= P(max(G1
t:s−1 ∪G2
t−1) < G1
t−1 | CN)

= E

|G1
t−1|
|G1
t−1| + |G1
t:s−1| + |G2
t−1|

 CN



= E

|G1
t−1|
t −1 + |G1
t:s−1|

 CN



=
λt + O(√N log N)
t + λ(s −t) + O(√N log N)

=
λt
(1 −λ)t + λs + O
q

log N

N

.

For the case g∗
t = 2, we obtain
P(ρ1 ≥s, g∗
t−1 = 2 | CN) = P(max G1
t:s−1 < G1
t−1, max G1
t−1 < max G2
t−1 | CN)

= max G1
t:s−1 < G1
t−1 < max G2
t−1 | CN)

= E

|G2
t−1|
|G1
t−1| + |G1
t:s−1| + |G2
t−1| ·
|G1
t−1|
|G1
t:s−1| + |G1
t−1|

 CN



= E

|G2
t−1|
t −1 + |G1
t:s−1| · |G1
t−1|
|G1
s−1|

 CN



=
(1 −λ)t + O(√N log N)
t + λ(s −t) + O(√N log N) · λt + O(√N log N)

λs + O(√N log N)

=
(1 −λ)t2

((1 −λ)t + λs)s + O
q

log N

N

.

Lemma C.3. Let ⌊αN⌋≤t ≤s < ⌊βN⌋, and consider a run of Algorithm AB
t (α, β), then

P(ρ1 = s, Rs ̸= 1, g∗
t−1 = 1 | CN) = λ2(1 −λ)(s −t)t

((1 −λ)t + λs)2s + O
q

log N

N3


P(ρ1 = s, Rs ̸= 1, g∗
t−1 = 2 | CN) = (1 −λ)
 
(1 −λ)2t + λ(2 −λ)s

t2

((1 −λ)t + λs)2s2
+ O
q

log N

N3

.

Proof. For Algorithm AB
t (α, β) and s ∈{t, . . . , ⌊βN⌋−1}, ρ1 = s if and only if xs is the first
element in G1 since step t for which rs = 1, thus
ρ1 = s ⇐⇒gs = 1 and max G1
t:s−1 < max G1
t−1 < xs .
Furthermore, Lemma A.3 gives that P(gs = 1 | CN) = P(gs = 1) + O(1/N) = λ + O(1/N).
Therefore, it holds that
P(ρ1 = s, Rs ̸= 1, g∗
t−1 = 1 | CN)

= P(gs = 1 , max G1
t:s−1 < max G1
t−1 < xs , xs < max G2
s−1 , max G2
t−1 < max G1
t−1 | CN)

= P(gs = 1 | CN)P(max G1
t:s−1 < max G1
t−1 < xs < max G2
t:s−1, max G2
t−1 < max G1
t−1 | CN)

= P(gs = 1 | CN)P(max(G1
t:s−1 ∪G2
t−1) < max G1
t−1 < xs < max G2
t:s−1 | CN)

= (λ + O( 1

N ))E
h
|G2
t:s−1|
|G1
t:s−1|+|G2
t−1|+|G1
t−1|+1+|G2
t:s−1| ·
1
|G1
t:s−1|+|G2
t−1|+|G1
t−1|+1 ·
|G1
t−1|
|G1
t:s−1|+|G2
t−1|+|G1
t−1|

 CN
i

= (λ + O( 1

N ))E
|G2
t:s−1|

s
·
1
t + |G1
t:s−1| ·
|G1
t−1|
t −1 + |G1
t:s−1|

 CN



= (λ + O( 1

N )) (1 −λ)(s −t) + O(√N log N)

s(t + λ(s −t) + O(√N log N)) ·
λt + O(√N log N)
t + λ(s −t) + O(√N log N)

= λ2(1 −λ)(s −t)t

((1 −λ)t + λs)2s + O
q

log N

N3

.

23


---Page Break---
On the other hand, in the case where g∗
t−1 = 2, we obtain

P(ρ1 = s, Rs ̸= 1, g∗
t−1 = 2 | CN)

= P(gs = 1 , max G1
t:s−1 < max G1
t−1 < xs , xs < max G2
s−1 , max G1
t−1 < max G2
t−1 | CN)

= P(gs = 1 | CN)P(max G1
t:s−1 < max G1
t−1 < xs < max G2
s−1 , max G1
t−1 < max G2
t−1 | CN)
= P(gs = 1 | CN)P(a < b < xs < max(c, d) , b < c | CN) ,

where a = max G1
t:s−1, b = max G1
t−1, c = max G2
t−1 and d = max G2
t:s−1. Let us denote by E
the event {a < b < xs < max(c, d)} ∩{b < c}. It holds that

E ∩{c < d} = {a < b < xs < max(c, d)} ∩{b < c} ∩{c < d}
= {a < b < c < xs < d} ∪{a < b < xs < c < d}

= {a < b < c < xs < d} ∪
 
{a < b < xs < c} ∩{c < d}

,

E ∩{d < c} = {a < b < xs < max(c, d)} ∩{b < c} ∩{d < c}
= {a < b < xs < c} ∩{d < c} ,

which yields

E =
 
E ∩{c < d}
 
E ∩{d < c}


= {a < b < c < xs < d} ∪
 
{a < b < xs < c} ∩{c < d}

∪
 
{a < b < xs < c} ∩{d < c}


= {a < b < c < xs < d} ∪{a < b < xs < c} .

The two events above are disjoint, and we have

P(a < b < c < xs < d | CN)

= P(max G1
t:s−1 < max G1
t−1 < max G2
t−1 < xs < G2
t:s−1 | CN)

= E
h
|G2
t:s−1|
|G1
t:s−1|+|G1
t−1|+|G2
t−1|+1+|G2
t:s−1| ·
1
|G1
t:s−1|+|G1
t−1|+|G2
t−1|+1 ·
|G2
t−1|
|G1
t:s−1|+|G1
t−1|+|G2
t−1| ·
|G1
t−1|
|G1
t:s−1|+|G1
t−1|· | CN
i

= E
|G2
t:s−1|

s
·
1
t + |G1
t:s−1| ·
|G2
t−1|
t −1 + |G1
t:s−1| · |G1
t−1|
|G1
s−1|

 CN



= (1 −λ)(s −t) + O(√N log N)

s(t + λ(s −t) + O(√N log N)) ·
(1 −λ)t + O(√N log N)
t + λ(s −t) + O(√N log N) · λt + O(√N log N)

λs + O(√N log N)

= (1 −λ)2(s −t)t2

((1 −λ)t + λs)2s2 + O
q

log N

N3

.

The probability of the second event is

P(a < b < xs < c | CN) = P(max G1
t:s−1 < max G1
t−1 < xs < max G2
t−1 | CN)

= E
h
|G2
t−1|
|G1
t:s−1|+|G1
t−1|+1+|G2
t−1| ·
1
|G1
t:s−1|+|G1
t−1|+1 ·
|G1
t−1|
|G1
t:s−1|+|G1
t−1|

 CN
i

= E

|G2
t−1|
t + |G1
t:s−1| ·
1
|G1
s−1| + 1 · |G1
t−1|
|G1
s−1|

 CN



=
(1 −λ)t + O(√N log N)
t + λ(s −t) + O(√N log N) ·
1
λs + O(√N log N) · λt + O(√N log N)

λs + O(√N log N)

=
(1 −λ)t2

λ((1 −λ)t + λs)s2 + O
q

log N

N 3

.

24


---Page Break---
Finally, Lemma A.3 shows that P(gs = 1 | CN) = λ + O(1/N 2), and we deduce

P(ρ1 = s, Rs ̸= 1, g∗
t−1 = 2 | CN) = λP(E | CN) + O( 1

N2 )

= λP(a < b < c < xs < d | CN) + λP(a < b < xs < c | CN) + O( 1

N2 )

= λ(1 −λ)2(s −t)t2

((1 −λ)t + λs)2s2 +
λ(1 −λ)t2

λ((1 −λ)t + λs)s2 + O
q

log N

N3


=
(1 −λ)t2

((1 −λ)t + λs)2s2 (λ(1 −λ)(s −t) + (1 −λ)t + λs) + O
q

log N

N3


=
(1 −λ)t2

((1 −λ)t + λs)2s2
 
(1 −λ)2t + λ(2 −λ)s

+ O
q

log N

N3

.

Lemma C.4. Let ⌊αN⌋≤t < ⌊βN⌋, and consider a run of Algorithm AB
t (α, β), then

P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 1 | CN)

=
t
βN UB
N,⌊βN⌋,1 +
λ(βN −t)t
((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N

,

P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 2 | CN)

=
t2

((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N

.

Proof. Since Algorithm AB
t (α, β) is memoryless, if it does not stop before step ⌊βN⌋, then its
success probability is the same as that of AB
⌊βN⌋(α, β) = AB
0 (β, β), which has the same threshold
β for both groups, if it is in the same state (g∗
⌊βN⌋−1, |G1
⌊βN⌋|). In all the proof, ρ1 is relative to
Algorithm AB
t (α, β), not AB
0 (β, β). It holds that

P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 1 | CN)

=
X

ℓ∈{1,2}
P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ| CN)

=
X

ℓ∈{1,2}
P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ| CN))

× P(AB
t (α, β) succeeds | ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ, CN)

=
X

ℓ∈{1,2}
P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ| CN)P(AB
0 (β, β) succeeds | g∗
⌊βN⌋−1 = ℓ, CN)

=
X

ℓ∈{1,2}
P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ| CN)
P(AB
0 (β, β) succeeds, g∗
⌊βN⌋−1 = ℓ| CN)

P(g∗
⌊βN⌋−1 = ℓ| CN)

=
X

ℓ∈{1,2}
P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ| CN)
UB
N,⌊βN⌋,ℓ
P(g∗
⌊βN⌋−1 = ℓ) + O( 1

N2 ) ,

25


---Page Break---
where we used Lemma A.3 and the definition (2) of UB
N,s,ℓ. Let us now compute the probability of
the event {ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = ℓ} conditional to CN. For ℓ= 1, we have

P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = 1 | CN)

= P(max G1
t:⌊βN⌋−1 < max G1
t−1 , max G2
t−1 < max G1
t−1 , max G2
⌊βN⌋−1 < max G1
⌊βN⌋−1 | CN)

= P(max x1:⌊βN⌋−1 ∈G1
t−1 | CN)

= E

|G1
t−1|
⌊βN⌋−1

CN



= λt + O(√N log N)

βN + O(1)
= λt

βN + O
q

log N

N

.

For ℓ= 2, we first compute the following

P(ρ1 ≥βN, g∗
t−1 = 1 | CN) = P(max G1
t:⌊βN⌋−1 < max G1
t−1 , max G2
t−1 < max G1
t−1 | CN)

= P(max(G1
t:⌊βN⌋−1 ∪G2
t−1) < max G1
t−1 | CN)

= E

"
|G1
t−1|
|G1
t:⌊βN⌋−1| + |G2
t−1| + |G1
t−1|

 CN

#

=
λt + O(√N log N)
t + λ(βN −t) + O(√N log N)

=
λt
(1 −λ)t + λβN + O
q

log N

N

,

and it follows that

P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = 2 | CN)

= P(ρ1 ≥βN, g∗
t−1 = 1 | CN) −P(ρ1 ≥βN, g∗
t−1 = 1, g∗
⌊βN⌋−1 = 1 | CN)

=
λt
(1 −λ)t + λβN −λt

βN + O
q

log N

N


=
λ(1 −λ)(βN −t)t
((1 −λ)t + λβN)βN + O
q

log N

N

.

All in all, we deduce that

P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 1 | CN)

=
 λt

βN + O
q

log N

N
 UB
N,⌊βN⌋,1
λ + O( 1

N2 ) +
 λ(1 −λ)(βN −t)t

((1 −λ)t + λβN)βN + O
q

log N

N

UB
N,⌊βN⌋,2
1 −λ + O( 1

N 2 )

=
 t

βN + O
q

log N

N

UB
N,⌊βN⌋,1 +

λ(βN −t)t
((1 −λ)t + λβN)βN + O
q

log N

N

UB
N,⌊βN⌋,2

=
t
βN UB
N,⌊βN⌋,1 +
λ(βN −t)t
((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N

.

26


---Page Break---
On the other hand, if g∗
t−1 = 2 and ρ1 ≥βN, then necessarily g∗
⌊βN⌋−1 = 2, because no candidate
in G1 up to step ⌊βN⌋−1 surpasses max G1
t−1, which is less than max G2
t−1. Therefore

P(AB
t (α, β) succeeds, ρ1 ≥βN, g∗
t−1 = 2 | CN)

= P(ρ1 ≥βN, g∗
t−1 = 2 | CN)P(AB
t (α, β) | ρ1 ≥βN, g∗
t−1 = 2, CN)

= P(max G1
t:⌊βN⌋−1 < max G1
t−1 < max G2
t−1 | CN)P(AB
0 (β, β) | g∗
⌊βN⌋−1 = 2, CN)

= E

"
|G2
t−1|
t −1 + |G1
t:⌊βN⌋−1| ·
|G1
t−1|
|G1
⌊βN⌋−1|

 CN

#
P(AB
0 (β, β), g∗
⌊βN⌋−1 = 2 | CN)

P(g∗
⌊βN⌋−1 = 2 | CN)

=
(1 −λ)t + O(√N log N)
t + λ(βN −t) + O(√N log N) ·
λt + O(√N log N)
λβN + O(√N log N) ·
UB
N,⌊βN⌋,2
1 −λ + O( 1

N2 )

=
t2

((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N

.

In the following lemma, we compute the exact limit of UB
N,⌊βN⌋,k when the number of candidates
goes to infinity.
Lemma C.5. For all B ≥0 and k ∈{1, 2},

UB
N,⌊βN⌋,k = λkβ2
B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ O
q

log N

N

.

Proof. By definition (2) of UB
N,t,k, we have

UB
N,t,k = P(AB
⌊βN⌋(α, β) succeeds, g∗
t−1 = k | CN) ,

and AB
⌊βN⌋(α, β) is simply the single-threshold algorithm with threshold βN and budget B. Let
T = ⌊βN⌋. As in the proof of Lemma 3.1, we decompose the success probability of AB
⌊βN⌋as
follows
UB
N,T,k = P(AB
T (α, β) succeeds, g∗
T −1 = k | CN)

=

N
X

t=T

X

ℓ∈{1,2}
P(AB
T (α, β) succeeds, ρ1 = t, gt = ℓ, g∗
T −1 = k | CN)

=

N
X

t=T

X

ℓ∈{1,2}


P(AB
T (α, β) succeeds, ρ1 = t, Rt = 1, gt = ℓ, g∗
T −1 = k | CN)

+ P(AB
T (α, β) succeeds, ρ1 = t, Rt ̸= 1, gt = ℓ, g∗
T −1 = k | CN)

.

The terms appearing in the sums above were computed in the proof of Lemma 3.1. It follows
respectively from (4) and (7), with K = 2, that

P(AB
T (α, β) succeeds, ρ1 = t, Rt = 1, gt = ℓ, g∗
T −1 = k | CN) = λℓλk

N (T/t)2 + O
q

log N

N 3

,

P(AB
T (α, β) succeeds, ρ1 = t, Rt ̸= 1, gt = ℓ, g∗
T −1 = k | CN)

=
T 2

t3 + O
q

log N

N3

1(B > 0, k ̸= ℓ) UB−1
N,t+1,k ,

where the O terms are independent of t, they only depend on β. Therefore,

UB
N,T,k =

1 + O
q

log N

N

N
X

t=T

X

ℓ∈{1,2}

λℓλk

N (T/t)2 + T 2

t3 1(B > 0, k ̸= ℓ) UB−1
N,t+1,k



=

1 + O
q

log N

N

N
X

t=T

λk

N (T/t)2 + T 2

t3 1(B > 0) UB−1
N,t+1,k


.

27


---Page Break---
The first sum can easily be computed

N
X

t=T

λk

N (T/t)2 = λk(T/N)2

N

N
X

t=T

1
(t/N)2

= λk(β2 + O( 1

N ))
Z 1

β

du
u2 + O( 1

N )

= λkβ(1 −β) + O( 1

N ) .

Therefore,

UB
N,T,k =

1 + O
q

log N

N
 
λkβ(1 −β) + 1(B > 0)

N
X

t=T

T 2

t3 UB−1
N,t+1,k + O( 1

N )


= λkβ(1 −β) + 1(B > 0)

N
X

t=T

T 2

t3 UB−1
N,t+1,k + O
q

log N

N

.

Dividing by λk yields

 
λ−1
k UB
N,T,k

= β(1 −β) + 1(B > 0)

N
X

t=T

T 2

t3
 
λ−1
k UB−1
N,t+1,k

+ O
q

log N

N

.

thus the double-indexed sequence (λ−1
k Ub
N,t,k)b,t satisfies the same recursion and initial condition as
(P(Ab
t(0, 0) succeeds))b,t (see proof of Lemma 3.1) with β instead of w and K = 2. Therefore, we
deduce immediately that:

λ−1
k UB
N,T,k = β2
B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ O
q

log N

N

.

C.3
Proof of Lemma 4.1

Proof. Using Lemmas C.1, C.2, C.3 and C.4 for k = 2, we obtain for all t ∈{⌊αN⌋, . . . , ⌊βN⌋−1}

UB
N,t,2 = λ

N

⌊βN⌋−1
X

s=t


(1 −λ)t2

((1 −λ)t + λs)s + O
q

log N

N


+ 1(B > 0)

1 −λ

⌊βN⌋−1
X

s=t

 
(1 −λ)
 
(1 −λ)2t + λ(2 −λ)s

t2

((1 −λ)t + λs)2s2
+ O
q

log N

N3
!

UB−1
N,s+1,2

+
t2

((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N

.

The O terms inside the sums depend on the ratios t/N and s/N, but using that α ≤t/N ≤s/N ≤1,
it can be made only dependent on α and the other constants of the problem. Moreover, Thus we can

28


---Page Break---
write

UB
N,t,2 = λ(1 −λ)t2

N

⌊βN⌋−1
X

s=t

1
((1 −λ)t + λs)s

+ 1(B > 0) t2
⌊βN⌋−1
X

s=t

 
(1 −λ)2t + λ(2 −λ)s


((1 −λ)t + λs)2s2
UB−1
N,s+1,2

+
t2

((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N


= λ(1 −λ)( t

N )2 1

N

⌊βN⌋−1
X

s=t

1
((1 −λ) t

N + λ s

N ) s

N

+ 1(B > 0) ( t

N )2 1

N

⌊βN⌋−1
X

s=t

(1 −λ)2 t

N + λ(2 −λ) s

N
((1 −λ) t

N + λ s

N )2( s

N )2 UB−1
N,s+1,2

+
(t/N)2

((1 −λ) t

N + λβ)β UB
N,⌊βN⌋,2 + O
q

log N

N

.

Taking t = ⌊wN⌋= wN + O(1) and using Riemann sum convergence properties yields

( t

N )2 1

N

⌊βN⌋−1
X

s=t

1
((1 −λ) t

N + λ s

N ) s

N
= w2

N

⌊βN⌋−1
X

s=⌊wN⌋

1
((1 −λ)w + λ s

N ) s

N
+ O
  1

N


= w2
Z β

w

du
((1 −λ)w + λu)u + O
  1

N


=
w
1 −λ

 Z β

w

du

u −
Z β

w

du
(1/λ −1)w + u

!

+ O
  1

N


=
w
1 −λ (−log(w/β) −log(1 −λ + λβ/w)) + O
  1

N


=
−w log
 
(1 −λ) w

β + λ


1 −λ
+ O
  1

N

.
(11)

On the other hand,

(t/N)2

((1 −λ) t

N + λβ)β =
w2

((1 −λ)w + λβ)β + O( 1

N ) ,

and Lemma C.5 gives for k = 2 that φB
2 (α, β; β) exists, its expression is

φB
2 (α, β; β) = (1 −λ)β2
B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

,
(12)

and it satisfies UB
N,⌊βN⌋,2 = φB
2 (α, β; β) + O
q

log N

N

. Consequently,

UB
N,⌊wN⌋,2 = −λw log
 
(1 −λ) w

β + λ


+ 1(B > 0) w2

N

⌊βN⌋−1
X

s=⌊wN⌋

(1 −λ)2w + λ(2 −λ) s

N
((1 −λ)w + λ s

N )2( s

N )2 UB−1
N,s+1,2

+
w2

((1 −λ)w + λβ)β φB
2 (α, β; β) + O
q

log N

N

.
(13)

Using this equality, we will prove by induction over B ≥0 that, for all w ∈[α, β], the limit
φB(α, β; w) := limN→∞UB
N,⌊wN⌋,2 exists, is continuous, and satisfies

UB
N,⌊wN⌋,2 = φB
2 (α, β; w) + O
q

log N

N

.

29


---Page Break---
Initialization. For B = 0 and w ∈[α, β], (13) gives immediately

UB
N,⌊wN⌋,2 = −λw log
 
(1 −λ) w

β + λ

+
w2

((1 −λ)w + λβ)β φB
2 (α, β; β) + O
q

log N

N

. (14)

Induction. Let B ≥1, w ∈[α, β], and assume that UB−1
N,⌊uN⌋,2 = φB
2 (α, β; u) + O
q

log N

N


for all u ∈[α, β], where the O does not depend on u. Using this hypothesis for u = s+1

N , with
s ∈{t, . . . , ⌊βN⌋−1}, along with te continuity of φB−1
2
(α, β; ·) and Riemann sums convergence
properties, yields

w2

N

⌊βN⌋−1
X

s=⌊wN⌋

(1 −λ)2w + λ(2 −λ) s

N
((1 −λ)w + λ s

N )2( s

N )2 UB−1
N,s+1,2

= w2

N

⌊βN⌋−1
X

s=⌊wN⌋

(1 −λ)2w + λ(2 −λ) s

N
((1 −λ)w + λ s

N )2( s

N )2 φB−1
2
(α, β; s+1

N ) + O
q

log N

N


= w2
Z β

w

(1 −λ)2w + λ(2 −λ)u

((1 −λ)w + λu)2u2
φB−1
2
(α, β; u)du + O
q

log N

N

.

Therefore, we deduce by (13) that

UB
N,⌊wN⌋,2 = −λw log
 
(1 −λ) w

β + λ


+ w2
Z β

w

(1 −λ)2w + λ(2 −λ)u

((1 −λ)w + λu)2u2
φB−1
2
(α, β; u)du

+
w2

((1 −λ)w + λβ)β φB
2 (α, β; β) + O
q

log N

N

.

This proves that UB
N,⌊wN⌋,2 = φB
2 (α, β; w) + O
q

log N

N

, where

φB
2 (α, β; w) = −λw log
 
(1 −λ) w

β + λ


+ w2
Z β

w

(1 −λ)2w + λ(2 −λ)u

((1 −λ)w + λu)2u2
φB−1
2
(α, β; u)du

+
w2

((1 −λ)w + λβ)β φB
2 (α, β; β) ,

where the expression of φB
2 (α, β; β) is given in (12).

30


---Page Break---
C.4
Proof of Lemma 4.2

Proof. Using Lemmas C.1, C.3, C.2 and C.4 for k = 2, we obtain for all t ∈{⌊αN⌋, . . . , ⌊βN⌋−1}

UB
N,t,1 = λ

N

⌊βN⌋−1
X

s=t


λt
(1 −λ)t + λs + O
q

log N

N


+ 1(B > 0)

1 −λ

⌊βN⌋−1
X

s=t

 λ2(1 −λ)(s −t)t

((1 −λ)t + λs)2s + O
q

log N

N3

UB−1
N,s+1,2

+
t
βN UB
N,⌊βN⌋,1 +
λ(βN −t)t
((1 −λ)t + λβN)βN UB
N,⌊βN⌋,2 + O
q

log N

N


= λ2(t/N)

N

⌊βN⌋−1
X

s=t

1
(1 −λ) t

N + λ s

N

+ 1(B > 0) λ2(t/N)

N

⌊βN⌋−1
X

s=t

s
N −t

N
((1 −λ) t

N + λ s

N )2 s

N
UB−1
N,s+1,2

+ t/N

β UB
N,⌊βN⌋,1 +
λ(β −t

N ) t

N
((1 −λ) t

N + λβ)β UB
N,⌊βN⌋,2 + O
q

log N

N

.

Consider in the following t = ⌊wN⌋= wN + O(1). Using Riemann sums convergence properties,
we have

λ2(t/N)

N

⌊βN⌋−1
X

s=t

1
(1 −λ) t

N + λ s

N
= λ2w
Z β

w

du
(1 −λ)w + λu + O( 1

N )

= λw [log((1 −λ)w + λu)]β
w + O( 1

N )

= λw log
 
1 −λ + λ β

w

+ O( 1

N ) .

Since Ub
N,s,k ≤1 for all b, s, k, and t = wN + O(1), as in the proof of Lemma 4.1, we obtain

UB
N,⌊wN⌋,1 = λw log
 
1 −λ + λ β

w

+ O( 1

N )

+ 1(B > 0) λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N
UB−1
N,s+1,2 + O( 1

N )

+ w

β UB
N,⌊βN⌋,1 +
λ(β −w)w
((1 −λ)w + λβ)β UB
N,⌊βN⌋,2 + O
q

log N

N


= λw log
 
1 −λ + λ β

w


+ 1(B > 0) λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N
UB−1
N,s+1,2

+ w

β φB
1 (α, β; β) +
λ(β −w)w
((1 −λ)w + λβ)β φB
2 (α, β; β) + O
q

log N

N

,

where we used Lemma C.5 in the last inequality, which guarantees that UB
N,⌊βN⌋,k = φB
k (α, β; β) +

O
q

log N

N

for k ∈{1, 2}, with

φB
k (α, β; β) = λkβ2
B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

.

31


---Page Break---
Denoting by φB(α, β; β) = φB
1 (α, β; β) + φB
2 (α, β; β), i.e. φB(α, β; β) =
1
λk φB
k (α, β; β), we
have

w

β φB
1 (α, β; β) +
λ(β −w)w
((1 −λ)w + λβ)β φB
2 (α, β; β) =
λw

β + λ(1 −λ)(β −w)w

((1 −λ)w + λβ)β


φB(α, β; β)

= λw

β


1 + (1 −λ)(β −w)

(1 −λ)w + λβ


φB(α, β; β)

= λw

β


β
(1 −λ)w + λβ


φB(α, β; β)

=
λw
(1 −λ)w + λβ φB(α, β; β) .

Thus

UB
N,⌊wN⌋,1 = λw log
 
1 −λ + λ β

w

+
λw
(1 −λ)w + λβ φB(α, β; β)

+ 1(B > 0) λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N
UB−1
N,s+1,2 + O
q

log N

N

.
(15)

Now, we will prove by induction over B that UB
N,⌊wN⌋,1 = φB
1 (α, β; w) + O
q

log N

N

for all

w ∈[α, β], with φB
1 (α, β; ·) a continuous function satisfying the recursion stated in the Lemma.

Initialization For B = 0, (15) yields immediately for all w ∈[α, β]

UB
N,⌊wN⌋,1 = λw log
 
1 −λ + λ β

w

+
λw
(1 −λ)w + λβ φB(α, β; β) + O
q

log N

N

.

Induction Let B ≥1, and assume that UB−1
N,⌊uN⌋,1 = φB−1
1
(α, β; u)+O
q

log N

N

for all u ∈[α, β],

and that φB
1 (α, β; ·) is continuous. Consequently, using Riemann sums convergence properties, it
holds for all w ∈[α, β] that

λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N
UB−1
N,s+1,2

= λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N


φB−1
2
(α, β; s+1

N ) + O
q

log N

N


= λ2w

N

⌊βN⌋−1
X

s=⌊wN⌋

s
N −w
((1 −λ)w + λ s

N )2 s

N
φB−1
2
(α, β; s+1

N ) + O
q

log N

N


= λ2w
Z β

w

(u −w)φB−1
2
(α, β; u)
((1 −λ)w + λu)2u du + O
q

log N

N

,

thus, we have by substituting into (15)

UB
N,⌊wN⌋,1 = λw log
 
1 −λ + λ β

w

+
λw
(1 −λ)w + λβ φB(α, β; β)

+ λ2w
Z β

w

(u −w)φB−1
2
(α, β; u)
((1 −λ)w + λu)2u du + O
q

log N

N


= φB
1 (α, β; w) + O
q

log N

N

.

32


---Page Break---
C.5
Proof of Corollary 4.3.1

Proof. Assume that λ ≥1/2. For any thresholds 0 < α ≤β ≤1 Theorem 4.3 yields

lim
N→∞P(AB(α, β) succeeds) ≥λα log
  β

α

+ αβSB(β) .

We will now determine thresholds maximizing this lower bound. For a fixed β, we have

∂
∂α
 
λα log
  β

α

+ αβSB(β)

≥0 ⇐⇒λ log
  β

α

−λ + βSB(β) ≥0

⇐⇒λ log
  α

β

≤βSB(β) −λ

⇐⇒α ≤β

e exp
β

λSB(β)

.

This proves that, for fixed β the lower bound λα log
  β

α

+ αβSB(β) is maximized on [0, β] for
α = min(β, β

e exp( β

λSB(β))) = hB(β). With this choice of α, the optimal choice of β is the one
maximizing the mapping β 7→λhB(β) log
 
β
hB(β)

+ hB(β)βSB(β).

In particular, for B = 0, we obtain ˜α0 = λ exp( 1

λ −2) and ˜β0 = λ. They guarantee an asymptotic
success probability of at least λ2 exp( 1

λ −2). Given that the sequence (SB(w))B is non-decreasing
for all w ∈(0, 1], it holds for all B ≥0 that

lim
N→∞P(AB(˜αB, ˜βB) succeeds) ≥λ˜αB log
  ˜βB

˜αB

+ ˜αB ˜βBSB(˜βB)

= max
α≤β

n
λα log
  β

α

+ αβSB(β)
o

≥max
α≤β

n
λα log
  β

α

+ αβS0(β)
o

= λ˜α0 log
  ˜β0

˜α0

+ ˜α0 ˜β0S0(˜β0)

= λ2 exp( 1

λ −2)

≥1

e −( 4

e −1)λ(1 −λ) .

On the other hand, taking equal thresholds α = β = 1

e then using Corollary 3.2.1 with K = 2 gives

lim
N→∞P(AB(˜αB, ˜βB) succeeds) ≥max
α≤β

n
λα log
  β

α

+ αβSB(β)
o

≥SB(1/e)

e2

≥1

e −
1
e(B + 1)! .

Thus, we deduce that

lim
N→∞P(AB(˜αB, ˜βB) succeeds) ≥1

e −min

1
e(B + 1)!, ( 1

e −1)λ(1 −λ)

.

33


---Page Break---
C.6
Proof of Theorem 4.3

Proof. By Lemmas A.3, 4.1 and 4.2, The success probability of Algorithm A(α, β)B can be written
as

P(AB(α, β) succeeds) = P(AB
⌊αN⌋(α, β) succeeds | CN) + O( 1

N 2 )

= P(AB
⌊αN⌋(α, β) succeeds, g∗
⌊αN⌋−1 = 1 | CN)

+ P(AB
⌊αN⌋(α, β) succeeds, g∗
⌊αN⌋−1 = 2 | CN) + O( 1

N2 )

= UB
N,⌊αN⌋,1 + UB
N,⌊αN⌋,2 + O( 1

N2 )

= φB
1 (α, β; α) + φB
2 (α, β; α) + O
q

log N

N


= λα log
 
1 −λ + λ β

α

+
λαβ2

(1 −λ)α + λβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) λ2α
Z β

α

(u −α)φB−1
2
(α, β; u)
((1 −λ)α + λu)2u du

−λα log
 
(1 −λ) α

β + λ

+
(1 −λ)βα2

(1 −λ)α + λβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) α2
Z β

α

(1 −λ)2α + λ(2 −λ)u

((1 −λ)α + λu)2u2
φB−1
2
(α, β; u)du + O
q

log N

N

,

then, regrouping the terms yields

P(AB(α, β) succeeds)

= λα log
 
1 −λ + λ β

α

−λα log
 
(1 −λ) α

β + λ


+
αβ
(1 −λ)α + λβ (λβ + (1 −λ)α)

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) α
Z β

α

 
λ2(1 −α

u) + α

u
 
(1 −λ)2 α

u + λ(2 −λ)
 φB−1
2
(α, β; u)du
((1 −λ)α + λu)2 + O
q

log N

N

.

Finally, observing that

(1 −λ)2 α

u + λ(2 −λ)

= (1 −λ)2 α2

u2 + 2λ(1 −λ) α

u + λ2

=
1
u2
 
(1 −λ)α + λu
2 ,

we deduce the result

P(AB(α, β) succeeds)

= λα log
  β

α

+ αβ

B
X

b=0

 
1
β −

b
X

ℓ=0

log(1/β)ℓ

ℓ!

!

+ 1(B > 0) α
Z β

α

φB−1
2
(α, β; u)du

u2
+ O
q

log N

N

,

D
Optimal memory-less algorithm for two groups

In this section, we derive an optimal memoryless algorithm employing a dynamic programming
approach. We analyze the state transitions depending on the algorithm’s actions and the associated
success probabilities for each state. Unlike previous sections, our study here is not asymptotic.
Therefore, we do not rely on estimating the number of candidates in each group using concentration
inequalities. Instead, we consider the exact number of candidates in each group as a parameter for
decision-making at each step.

34


---Page Break---
D.1
Memoryless algorithms

One distinctive feature of Dynamic Threshold algorithms is their decision-making process, which
solely depends on the observations at each step and the available budget, without recourse to past
comparison history. We designate algorithms exhibiting this characteristic as memory-less algorithms.
Definition D.1. An algorithm A for the (K, B)-secretary problem is memory-less if its actions at
any step t ∈[N] depend only on the current observations rt, gt, 1(Rt = 1), the available budget Bt,
the cardinals (|Gk
t−1|)k∈[K].

We assume that a memory-less algorithm is aware of the current step t at any time, and knows the
proportions of each group (λk)k∈[K]. However, in our analysis, the knowledge of group proportions
is dispensable since we investigate the asymptotic success probabilities of DT algorithms. Indeed,
for setting thresholds that depend on group proportions, if the smallest threshold is at least ϵ > 0,
regardless of group proportions, it suffices to observe the first ⌊ϵN⌋candidates, then estimate
¯λk = ⌊ϵN⌋−1 P⌊ϵN⌋
t=1 1(gt = k) for all k ∈[K]. The algorithm can choose the thresholds using
(¯λk)k∈[K] instead of (λk)k∈[K]. As the number of candidates tends to infinity, ¯λk becomes arbitrarily
close to λk with high probability, and so do the thresholds, assuming they are continuous functions of
the group proportions. Though this introduces additional intricacies to the proofs, the fundamental
proof arguments and the results remain the same. While Definition D.1 only includes deterministic
algorithms, it can be easily extended to randomized algorithms, by considering the distributions of
the actions instead of the actions themselves.

In the following lemma, we establish that the success probability of a memory-less algorithm, given
the history up to step t −1, is contingent upon only a few parameters, which are the available
budget Bt, the group to which the best-observed candidate belongs g∗t, and the sizes of the groups
(|Gk
t |)k ∈[K]. Collectively, these parameters define the state of a memory-less algorithm at step t,
which entirely determines the success probability of the algorithm starting from that state.
Lemma D.1. For any memory-less algorithm A and t ∈[N], denoting by τ the stopping time of A
and by Ft−1 is the history of the algorithm up to step t −1, i.e. the set of all the observations and
actions taken by the algorithm until step t −1, then

P(A succeeds | τ ≥t, g∗
t , Ft−1) = P(A succeeds | τ ≥t, g∗
t , Bt, (|Gk
t |)k∈[K]) .

Proof. Let A be a memory-less algorithm, and let us denote by τ its stopping time. Conditionally to
the history of the algorithm until step t −1 and to the event {τ ≥t}, the success probability of A
depends on the future observations and the future actions of the algorithm.

Given that the algorithm is memory-less, at any step s ≥t, its actions as,1, as,2 depend on the
observations rt, gt, Rt, the budget Bt and (|Gk
s−1|)k∈[K].

Conditionally to the cardinals of the groups at step t−1, the cardinals (|Gk
s−1|)k∈[K] are independent
of the history Ft−1 because

|Gk
s−1| = |Gk
t−1| +

s−1
X

u=t
1(gu = k)
∀k ∈[K] ,

Moreover, since the candidates are observed in a uniformly random order, and the group memberships
are also i.i.d random variables, then for all s ≥2 distributions of rs, Rs depend only on the cardinals
of each group at step s −1, on gs and g∗
s−1. Also g∗
s is a function of g∗
s−1, gs and 1(Rs = 1):

g∗
s = 1(Rs ̸= 1) g∗
s−1 + 1(Rs ̸= 1) gs ,

and the budget Bs satisfies

Bs = Bs−1 −1(as−1,2 = compare) .

Therefore, Conditionally to the Bt, (|Gk
t−1|)k∈[K], g∗
t−1, the distributions of the observations and of
the algorithm’s actions at any step s ≥t are independent of the history before step t.

At any given step t, a memory-less algorithm has access to the available budget Bt and the number of
previous candidates belonging to each group. In the case of two groups, this information reduces to

35


---Page Break---
(t, Bt, |G1
t−1|), since |G2
t−1| = t −1 −|G1
t−1|. The state of the algorithm, which fully determines its
success probability, is given by the tuple (t, Bt, |G1
t−1|, g∗
t−1). However, g∗
t−1 is not known to the
algorithm, hence it must make decisions relying on the limited information it has, to maximize the
expected success probability, where the expectation is taken over g∗
t−1.

D.2
State transitions

For any memory-less algorithm A, we denote by St(A) its state at step t, which is a tuple (t, b, m, ℓ).
Here, t −1 represents the count of previously rejected candidates, b ≥0 denotes the available
budget, m < t indicates the number of prior candidates from group G1, and ℓ∈{1, 2} is the group
containing the best-seen candidate so far.

To examine the state transitions of the algorithm, it is imperative to first understand the distribution of
the new observations at any given step t, depending on St(A). While the group membership gt of
candidate xt is independent of St(A), both rt and Rt are contingent on it.

Lemma D.2. For any memory-less algorithm A and state (t, m, b, ℓ), denoting by k = 3 −ℓthe
group index different from ℓ, it holds that

P(rt = 1 | St(A) = (t, m, b, ℓ), gt = ℓ) = 1

t

P(rt = 1 | St(A) = (t, m, b, ℓ), gt = k) =
|Gk
t−1| + t
t(|Gk
t−1| + 1)

P(Rt = 1 | St(A) = (t, m, b, ℓ), gt = ℓ, rt = 1) = 1

P(Rt = 1 | St(A) = (t, m, b, ℓ), gt = k, rt = 1) = |Gk
t−1| + 1
|Gk
t−1| + t ,

where

|Gk
t−1| =

m
if k = 1
t −1 −m
if k = 2 .

Proof. If St(A) = (t, m, b, ℓ), then in particular g∗
t−1 = ℓ, i.e. max Gℓ
t−1 > max Gk
t−1, thus

P(rt = 1 | St(A) = (t, m, b, ℓ), gt = ℓ) = P(xt > max Gℓ
t−1 | St(A) = (t, m, b, ℓ), gt = ℓ)
= P(xt > max x1:t−1 | St(A) = (t, m, b, ℓ), gt = ℓ)

= 1

t ,

because the rank of xt among previous candidates is independent of their relative ranks and groups,
thus independent of the state of the algorithm. Moreover, if rt = 1, gt = 1 and g∗
t−1 = ℓ, then xt is
the better than the maximum of Gℓ
t−1, which is the maximum of x1:t−1, thus necessarily Rt = 1,

P(Rt = 1 | St(A) = (t, m, b, ℓ), gt = ℓ, rt = 1) = 1 .

On the other hand, if gt = k ̸= ℓ= g∗
t−1, assume that |Gℓ
t−1| > 0. It holds that

P(rt = 1 | St(A) = (t, m, b, ℓ), gt = k) = P(rt = 1 | g∗
t−1 = ℓ, gt = k, |G1
t−1| = m)

= P(rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1| = m)
P(g∗
t−1 = ℓ| |G1
t−1| = m)
.

We have immediately that

P(g∗
t−1 = ℓ| |G1
t−1| = m) = P(max Gℓ
t−1 > max Gk
t−1 | |G1
t−1| = m) = |Gℓ
t−1|
t −1 ,

36


---Page Break---
and the numerator can be computed as

P(rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1| = m)
(16)

= P(xt > max Gk
t−1, g∗
t−1 = ℓ| |G1
t−1| = m)

= P(xt > max Gk
t−1, max Gℓ
t−1 > max Gk
t−1 | |G1
t−1| = m)

= P(xt > max Gℓ
t−1 > max Gk
t−1 | |G1
t−1| = m)

+ P(max Gℓ
t−1 > xt > max Gk
t−1 | |G1
t−1| = m)

= 1

t · |Gℓ
t−1|
t −1 + |Gℓ
t−1|

t
·
1
|Gk
t−1| + 1

= |Gℓ
t−1|

t


1
t −1 +
1
|Gk
t−1| + 1


,
(17)

which yields

P(rt = 1 | St(A) = (t, m, b, ℓ), gt = k) = t −1

|Gℓ
t−1| · |Gℓ
t−1|

t


1
t −1 +
1
|Gk
t−1| + 1



= 1

t


1 +
t −1
|Gk
t−1| + 1



=
|Gk
t−1| + t
t(|Gk
t−1| + 1) .

Finally,

P(Rt = 1 | St(A) = (t, m, b, ℓ), gt = k, rt = 1) = P(Rt = 1, rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1|)
P(rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1|)
.

We computed the denominator term in (17), and the numerator satisfies

P(Rt = 1, rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1|) = P(Rt = 1, g∗
t−1 = ℓ| gt = k, |G1
t−1|)

= P(xt > max GℓGℓ
t−1 > max Gk
t−1 | |G1
t−1|)

= 1

t · |Gℓ
t−1|
t −1 ,

hence

P(Rt = 1 | St(A) = (t, m, b, ℓ), gt = k, rt = 1) =

|Gℓ
t−1|
t(t−1)
|Gℓ
t−1|

t

1
t−1 +
1
|Gk
t−1|+1



=
1
1 +
t−1
|Gk
t−1|+1

= |Gk
t−1| + 1
|Gk
t−1| + t .

This concludes the proof when |Gℓ
t−1| > 0. If |Gℓ
t−1| = 0, then the same identities remain trivially
true.

Using the previous Lemma, we can fully characterize the possible state transitions of a memory-less
algorithm. First, the values of the parameters |G1
t| and Bt+1 are trivially determined based on the
state St at the beginning of step t, the observations rt and gt, and the actions of the algorithm:

|G1
t| = |G1
t−1| + 1(gt = 1) ,
Bt+1 = Bt −1(at,1 = compare) ,

where at,1 is the action taken by the algorithm, which only depends on the state St since the algorithm
is memory-less.

37


---Page Break---
Regarding g∗
t , if gt = g∗
t−1, then g∗
t = g∗
t−1 remains unchanged with probability 1. However, if
gt ̸= g∗
t−1 and rt = 1, and if the algorithm skips the candidate without making a comparison, then g∗
t
is not deterministic based on the history alone. The probability that g∗
t = gt in this case is precisely
the probability that Rt = 1, computed in Lemma D.2

P(g∗
t = gt | St(A) = (t, m, b, ℓ), gt = k, rt = 1) = |Gk
t−1| + 1
|Gk
t−1| + t .

D.3
Expected action rewards

In the following, we denote by A∗the optimal memory-less algorithm for two groups, and for all
B ≥0, t ∈[N], m < t and ℓ∈{1, 2}, we denote by

VB
t,m,ℓ= P
 
A∗succeeds | τ ≥t, St(A∗) = (t, B, m, ℓ)

,

which is its success probability starting from state (t, B, m, ℓ).

We analyze the expected rewards and state transitions of algorithm A∗given its limited information
access. When the algorithm receives a new observation (rt, gt):

• If rt ̸= 1, the optimal action is to skip the candidate (skip).
• If rt = 1 and Bt = 0, the algorithm either stops or skips the candidate. However, if there is
a positive budget Bt, stopping is suboptimal: it is always better to make a comparison first.
• If the algorithm chooses to make a comparison and observes Rt:
– If Rt ̸= 1, the optimal action is to skip the candidate.
– If Rt = 1, the algorithm must decide whether to skip or stop. However, skipping after
observing Rt = 1 is suboptimal compared to skipping immediately after observing
rt = 1, as the latter conserves the budget.

In summary, any rational algorithm follows these decision rules:

• If (rt ̸= 1) or (rt = 1 and Rt ̸= 1), then skip the candidate.
• If (rt = 1 and Rt = 1), select the candidate.

Therefore, the main non-trivial decision to make is whether to reject or accept a candidate after
observing rt = 1. Consider an algorithm A following these rules. At time t with budget Bt = b and
|G1
t−1| = m, if gt = k and rt = 1, choosing an action a ∈{skip, stop, compare} based on these
rules leads to a new state St+1(A) = F(t, b, m, k, a), which is a random variable depending on g∗
t−1
and Rt. If a ∈{stop, compare} and Rt = 1, then St+1(A) is a final state: success or failure.

With this notation, we define RB
t,m(a) as the reward that A∗expects to gain by playing action a after
observing rt = 1 and gt = k in a state St(A∗) = (t, b, m, ·), where it ignores g∗
t−1

RB
t,m,k(a) = E[P
 
A∗succeeds | St+1(A∗) = F(t, B, m, k, a)

| St(A∗) = (t, B, m, ·), rt = 1, gt = k] .

where the expectation is taken over g∗
t−1 and Rt. The optimal memory-less action at any state
(t, B, m, ℓ), knowing that rt = 1, gt = k, is the one maximizing RB
t,m,k(a).

Lemma D.3. Consider a state St = (t, B, m, ·), and let {k, ℓ} = {1, 2}, Mk = m + 1(k = 1), then

RB
t,m,k(stop) = |Gk
t |
N
,

RB
t,m,k(skip) = |G1
t|
t
VB
t+1,Mk,1 + |G2
t|
t
VB
t+1,M,2 ,

RB
t,m,k(compare) = |Gk
t |
N
+ |Gℓ
t|
t

|Gk
t−1| + 1
|Gk
t−1| + t · t

N +
t −1
|Gk
t−1| + tVB−1
t+1,Mk,ℓ


,

Observe that, conditionally to gt and |G1
t−1|, the cardinals of G1
t−1, G2
t−1, G1
t, G2
t are all known:

|G1
t| = |G1
t−1| + 1(gt = 1) ,
|G2
t| = t −|G1
t|,
|G2
t−1| = t −1 −|G1
t−1| .

38


---Page Break---
D.4
Optimal actions and success probability

Using Lemma D.3 and considering the potential state transitions based on the actions, we establish a
recursion satisfied by (VB
t,m,ℓ)t,B,m,ℓ. We present the result without distinction between the cases
ℓ= 1 and ℓ= 2. For simplicity, let λk = P(gt = k) for k = 1, 2, and define Mk = m + 1(k = 1)
for all m ≥0. Additionally, for all (B, t, m, k), define

δB
k = 1
 
RB
t,m,k(accept) ≥RB
t,m,k(skip)

,

where the action accept corresponds to compare for B > 0 and stop for B = 0.

Theorem D.4. For all t ∈[N], m < t and {k, ℓ} = {1, 2}, the success probability of A∗with zero
budget satisfies the recursion

V0
t,m,ℓ= λℓ

δ0
ℓ
N +

1 −δ0
ℓ
t

V0
t+1,Mℓ,ℓ


+ λk

δ0
k
N + 1−δ0
k
t
V0
t+1,Mk,k +
 
1 −1

t
 
2 −δ0
k −
1
|Gk
t−1|+1


V0
t+1,Mk,ℓ

,

and for B ≥1 it satisfies

VB
t,m,ℓ= λℓ

δB
ℓ
N +
 
1 −δB
ℓ
t

VB
t+1,Mℓ,ℓ


+ λk

δB
k
N +
δB
k
|Gk
t−1|+1
 
1 −1

t

VB−1
t+1,Mk,ℓ+ 1−δB
k
t
VB
t+1,Mk,k +
 
1 −1

t
 
1 −
δB
k
|Gk
t−1|+1

VB
t+1,Mk,ℓ

,

where VB
N+1,m,k = 0 for all B ≥0 m ≤N and k ∈{1, 2}.

Proof. Using the results from Section D.2 and D.3, the actions of A∗and the resulting state transitions
are as follows. If the state of A∗at step t is St(A∗) = (t, B, m, ℓ) for some B ≥1, m < t and
ℓ∈{1, 2}: If gt = ℓ, denoting by Mℓ= m + 1(ℓ= 1), we have

• with probability 1 −1/t: rt = 0, and the algorithm rejects the candidate, transitioning to
the state (t + 1, B, Mℓ, ℓ).

• with probability 1/t: rt = 1, and necessarily Rt = 1, because gt = g∗
t−1 = ℓ.

– If RB
t,m,k(compare) > RB
t,m,k(skip), then the algorithm uses a comparison and
observes Rt = 1, hence accepts the candidate. The success probability in that case is
t/N.
– Otherwise, the candidate is rejected and the algorithm goes to state (t + 1, B, Mℓ, ℓ)

On the other hand, if gt = k ̸= g∗
t−1, then denoting by Mk = m + 1(k = 1), we have

• with probability
|Gk
t−1|(t+1)
t(|Gk
t−1|+1): rt = 0, and the algorithm rejects the candidate, transitioning

to the state (t + 1, B, Mk, ℓ).

• with probability
|Gk
t−1|+t
t(|Gk
t−1|+1): rt = 1

– If RB
t,m,k(compare) > RB
t,m,k(skip), then the algorithm uses a comparison

* with probability
|Gk
t−1|+1
|Gk
t−1|+t : Rt = 1 and the algorithm stops, its success probability

is t/N
* with probability
t−1
|Gk
t−1|+t: Rt = 0, the candidate is rejected, and the algorithm

goes to state (t + 1, B −1, Mk, ℓ)
– Otherwise, the candidate is rejected and

* with probability
|Gk
t−1|+1
|Gk
t−1|+t : the algorithm goes to state (t + 1, B, Mk, k)

* with probability
t−1
|Gk
t−1|+t: the algorithm goes to state (t + 1, B, Mk, ℓ)

39


---Page Break---
In the case of a zero budget, the algorithm compares RB
t,m,k(skip) to RB
t,m,k(compare) instead
of RB
t,m,k(compare). If the algorithm decides to reject the candidate then the same state transition
occurs. However, if the candidate is selected and if gt = g∗
t−1 = ℓthen the success probability is
t/N. If On the other hand, if it is selected and gt = k ̸= g∗
t−1 = ℓthen the probability that the

current candidate is the best overall is
|Gk
t−1|+1
|Gk
t−1|+t × t

N .

All in all, for B = 0, then

(V0
t,m,ℓ| gt = ℓ) = 1

t


δ0
ℓ
t
N + (1 −δ0
ℓ)V0
t+1,Mℓ,ℓ


+

1 −1

t


V0
t+1,Mℓ,ℓ

=

1 −δ0
ℓ
t

V0
t+1,Mℓ,ℓ+ δ0
ℓ
N

(V0
t,m,ℓ| gt = k) =
|Gk
t−1|+t
t(|Gk
t−1|+1)


δ0
k
t(|Gk
t−1|+1)
N(|Gk
t−1|+t) + (1 −δ0
k)
 |Gk
t−1|+1
|Gk
t−1|+t V0
t+1,Mℓ,l +
t−1
|Gk
t−1|+tV0
t+1,Mℓ,ℓ


+
|Gk
t−1|+t
t(|Gk
t−1|+1)V0
t+1,Mk,ℓ

= δ0
k
N + 1−δ0
k
t
V0
t+1,Mk,k +
 
1 −1

t
 
2 −δ0
k −
1
|Gk
t−1|+1


V0
t+1,Mk,ℓ,

and we deduce that

V0
t,m,ℓ= λℓ

1 −δ0
ℓ
t

V0
t+1,Mℓ,ℓ+ δ0
ℓ
N


+ λk

δ0
k
N + 1−δ0
k
t
V0
t+1,Mk,k +
 
1 −1

t
 
2 −δ0
k −
1
|Gk
t−1|+1


V0
t+1,Mk,ℓ

.

For B ≥1, we obtain

(VB
t,m,ℓ| gt = ℓ) = 1

t


δB
ℓ
t
N + (1 −δB
ℓ)VB
t+1,Mℓ,ℓ


+

1 −1

t


VB
t+1,Mℓ,ℓ

=

1 −δB
ℓ
t

VB
t+1,Mℓ,ℓ+ δB
ℓ
N

(VB
t,m,ℓ| gt = k) =
|Gk
t−1|+t
t(|Gk
t−1|+1)


δB
k
 |Gk
t−1|+1
|Gk
t−1|+t VB
t+1,Mk,k +
t−1
|Gk
t−1|+tVB
t+1,Mk,ℓ


+ (1 −δB
k )
 |Gk
t−1|+1
|Gk
t−1|+t V0
t+1,Mk,l +
t−1
|Gk
t−1|+tV0
t+1,Mk,ℓ
 
+
|Gk
t−1|+t
t(|Gk
t−1|+1)VB
t+1,Mk,ℓ

= δB
k
N +
δB
k
|Gk
t−1|+1
 
1 −1

t

VB−1
t+1,Mk,ℓ+ 1−δB
k
t
VB
t+1,Mk,k +
 
1 −1

t
 
1 −
δB
k
|Gk
t−1|+1

VB
t+1,Mk,ℓ,

hence

VB
t,m,ℓ= λℓ

δB
ℓ
N +
 
1 −δB
ℓ
t

VB
t+1,Mℓ,ℓ


+ λk

δB
k
N +
δB
k
|Gk
t−1|+1
 
1 −1

t

VB−1
t+1,Mk,ℓ+ 1−δB
k
t
VB
t+1,Mk,k +
 
1 −1

t
 
1 −
δB
k
|Gk
t−1|+1

VB
t+1,Mk,ℓ

,

which concludes the proof.

Implementing the optimal memory-less algorithm A∗with budget B requires knowing the
(Rb
t,m,k(a))t,b,m,k for a ∈{skip, stop, compare}, which depend themselves on the table
 
Vb
t,m,k


t,b,m,k. Using Lemma D.3 and Theorem D.4, these tables can be computed in a O(BN 2)
time as described in Algorithm 2.

After computing these tables, the optimal memory-less algorithm A∗can be implemented by following
the rational decision rules outlined in Section D.3, and when encountering rt = 1 and needing to
choose between accepting or rejecting the candidate, A∗selects the action that maximizes its expected
reward given the information it has about the current state. A detailed description is provided in
Algorithm 3.

40


---Page Break---
Algorithm 2: (Vb
t,m,k)t,b,m,k and (Rb
t,m,k(a))t,b,m,k for a ∈{skip, stop, compare}

Input: Number of candidates N, available budget B, probability distribution of gt: λ1, λ2
Initialization: Vb
N+1,m,k ←0 for all b ≤B, m ≤N, k ∈{1, 2}

1 for b = 1, . . . , B do

2
for t = N, N −1, . . . , 1 do

3
for m = 0, . . . , t do

4
Compute Rb
t,m,k(a) for k ∈{1, 2} and a ∈{skip, stop, compare} using Lemma
D.3

5
Compute Vb
t,m,k for k ∈{1, 2} using Theorem D.4

6 Return: (Vb
t,m,k)t,b,m,k, (Rb
t,m,k(a))t,b,m,k

Algorithm 3: Optimal memory-less algorithm A∗
Input: Number of candidates N, available budget B, probability distribution of gt: λ1, λ2
Initialization: b ←B, m ←0

1 Compute (Vb
t,m,k)t,b,m,k and (Rb
t,m,k(a))t,b,m,k,a using Algorithm 2

2 for t = 1, . . . , N do

3
Receive new observation (rt, gt)

4
if rt = 1 then

5
if b = 0 and R0
t,m,gt(stop) > R0
t,m,gt(skip) then

6
Return: t

7
if b > 0 and Rb
t,m,gt(compare) > Rb
t,m,gt(skip) then

8
b ←b −1

9
if Rt = 1 then

10
Return: t

11
m ←m + 1(gt = 1)

41


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: In the introduction, we first formally present the problem considered, and state
the main results of the paper.
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
Justification: The limitations are discussed during the paper and in the conclusion.
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

42


---Page Break---
Justification: All the proofs are in the appendix.
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
We explain the experimental settings.
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

43


---Page Break---
Answer: [Yes]

Justification: A link to the code is provided in the experiments section.

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

Justification: The full experiments setting is explained in the experiments section.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [No]

Justification: Most plots show deterministic values (optimal thresholds, visualization of
theoretic results,...)

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

44


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
Answer:[No]
Justification: The runtime is not an important factor in our simulations, we do not feel that
precising computer resources is relevant.
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
Justification: We have reviewed the Code of Ethics.
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
Justification: The paper studies a fundamental theoretic problem. It seems to us that no
particular societal impact should be discussed.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

45


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

46


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not provide any new assets.
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

47


---Page Break---
