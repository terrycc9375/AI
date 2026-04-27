Learning the Expected Core of
Strictly Convex Stochastic Cooperative Games

Nam Phuong Tran
Department of Computer Science
University of Warwick
Coventry, United Kingdom
nam.p.tran@warwick.ac.uk

The Anh Ta
CSIRO’s Data61
Marsfield, NSW, Australia
theanh.ta@csiro.au

Shuqing Shi
Department of Informatics
King’s College London
London, United Kingdom
shuqing.shi@kcl.ac.uk

Debmalya Mandal
Department of Computer Science
University of Warwick
Coventry, United Kingdom
debmalya.mandal@warwick.ac.uk

Yali Du
Department of Informatics
King’s College London
London, United Kingdom
yali.du@kcl.ac.uk

Long Tran-Thanh
Department of Computer Science
University of Warwick
Coventry, United Kingdom
long.tran-thanh@warwick.ac.uk

Abstract

Reward allocation, also known as the credit assignment problem, has been an
important topic in economics, engineering, and machine learning. An important
concept in reward allocation is the core, which is the set of stable allocations
where no agent has the motivation to deviate from the grand coalition. In previous
works, computing the core requires either knowledge of the reward function in
deterministic games or the reward distribution in stochastic games. However, this
is unrealistic, as the reward function or distribution is often only partially known
and may be subject to uncertainty. In this paper, we consider the core learning
problem in stochastic cooperative games, where the reward distribution is unknown.
Our goal is to learn the expected core, that is, the set of allocations that are stable
in expectation, given an oracle that returns a stochastic reward for an enquired
coalition each round. Within the class of strictly convex games, we present an
algorithm named Common-Points-Picking that returns a point in the expected
core given a polynomial number of samples, with high probability. To analyse the
algorithm, we develop a new extension of the separation hyperplane theorem for
multiple convex sets.

1
Introduction

The reward allocation problem is a fundamental challenge in cooperative games that seeks reward
allocation schemes to motivate agents to collaborate or satisfy certain constraints, and its solution
concepts have recently gained popularity within the machine learning literature through its application
in explainable AI (XAI) [17, 28, 13, 31] and cooperative Multi-Agent Reinforcement Learning
(MARL) [29, 12, 30]. In the realm of XAI, designers often seek to understand which factors of
the model contribute to the outputs. Solution concepts such as the Shapley value [17, 28] and the
core [31, 13] provide frameworks for assessing the influence that each factor has on the model’s

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
behavior. In cooperative MARL, these solution concepts offer a framework for distributing team
rewards to individual agents, promoting cooperation among players, and preventing the occurrence of
lazy learner phenomena [29, 12, 30]. A crucial notion of reward allocation is stability, defined as an
allocation scheme wherein no agent has the motivation to deviate from the grand coalition. The set of
stable allocations is called the core of the game.

In the classical setting, the reward function is deterministic and commonly known among all agents,
with no uncertainty within the game. However, assuming perfect knowledge of the game is often
unrealistic, as the outcome of the game may contain uncertainty. This led to the study of stochastic
cooperative games, dated back to the seminal works of [9, 27], where stability can be satisfied either
with high probability, known as the robust core, or in expectation, known as the expected core.
However, in these works, the distribution of stochastic rewards is given, allowing agents to calculate
the reward before the game starts, which is not practical since the knowledge of the reward distribution
may only be partially known to the players. When the distribution of the stochastic reward is unknown,
the task of learning the stochastic core by sequentially interacting with the environment appears
much more challenging. Early attempts [19, 20] studied robust core and expected core learning under
full-information, where full information means that the random rewards of all coalitions are observed
at each round. However, full-information feedback is a strong assumption, which does not hold in
many real-world situations. Instead, agents typically observe the reward of their own coalition only
(e.g., only know the value of their own action - joining a particular coalition in this case), which is
typically known as bandit feedback in the online learning literature.

In our work, we focus on learning the expected core, which circumvents the potential emptiness of
the robust core in many practical cases. Moreover, instead of full-information feedback, where the
stochastic rewards of all coalitions are observed each round, we consider the bandit feedback setting,
where only the stochastic reward of the inquired coalition is observed each round. Given the lack of
knowledge about the probability distribution of the reward function, learning the expected core using
data-driven approaches with bandit feedback is a challenging task.

Against this background, the contribution of this paper is three-fold: (1) We focus on expected
core learning problem with unknown reward function, and propose a novel algorithm called the
Common-Points-Picking algorithm, the first of its kind that is designed to learn the expected
core with high probability. Notably, this algorithm is capable of returning a point in an unknown
simplex, given access to the stochastic positions of the vertices, which can also be used in other
domains, such as convex geometry. (2) We establish an analysis for finite sample performance of the
Common-Points-Picking algorithm. The key component of the analysis revolves around a novel
extension of the celebrated hyperplane separation theorem, accompanied by further results in convex
geometry, which can also be of independent interest. (3) We show that our algorithm returns a point
in expected core with at least 1 −δ probability, using poly(n, log(δ−1)) number of samples.

2
Related Work

Stochastic Cooperative Games.
The study of stochastic cooperative games can be traced back
to at least [9]. The main goal of the allocation scheme is to minimise the probability of objections
arising after the realisation of the rewards. [27, 26] later extended and refined the notion of stability in
stochastic settings. These seminal works require information about the reward distribution to compute
a stable allocation scheme before the game starts. Stochastic cooperative games have also been
studied in a Bayesian setting in a series of papers [5, 7, 8, 6]. These works develop a framework for
Bayesian stochastic cooperative games, where the distribution of the reward is conditioned on a hidden
parameter following a prior distribution. The prior distribution is common knowledge among agents
and can be used to define a new concept of stability called Bayesian core. In contrast to previous
works, our paper focuses on studying scenarios where the reward distribution or prior knowledge is
not disclosed to the principal agent and computing a stable allocation requires a data-driven method.

Learning the Core.
The literature on learning the core through sample-based methods can be
categorised based on the type of core one seeks to evaluate. Two main concepts of the stochastic
core are commonly considered, namely the robust core (i.e. core constraints are satisfied with high
probability) [11, 22, 19] and the expected core (i.e. core constraints are satisfied in expectation)
[10, 20]. The robust core is defined in a manner that allows the core inequalities to be satisfied with
high probability when the reward is realised. However, this definition may lead to the practical issue

2


---Page Break---
of an empty robust core, as illustrated in [10]. To mitigate the potential emptiness of the robust core,
we investigate the learnability of the expected core.

The work most closely related to ours is [20], in which the authors introduce an algorithm designed
to approximate the expected core using a robust optimization framework. In the context of full
information feedback, where rewards for all allocations are revealed each round, the algorithm
demonstrates asymptotic convergence to the expected core. Furthermore, the authors provide an
finite-samples error bound for the distance to the expected core. However, when dealing with bandit
feedback, where the reward of the enquired coalition is returned each round, naively applying the
algorithm of [20] may result in an exponential number of samples in terms of the number of players. In
the bandit feedback setting, as we later establish in Theorem 7 that, in general cooperative games, no
algorithm can learn the expected core with a finite number of samples without additional assumptions.
This highlights the key difference in dealing with bandit feedback compared to full-information
feedback. Given this limitation, we narrow our focus to (strictly) convex games, an important class of
cooperative games where the expected reward function is (strictly) supermodular. By leveraging strict
convexity, we introduce the Common-Points-Picking algorithm, which efficiently returns a point
in the expected core with high probability using only a polynomial number of samples. While [20]
proposed a general algorithm applicable to strictly convex games, we argue that it lacks statistical and
computational efficiency due to the absence of a mechanism to exploit the supermodular structure of
the expected reward function. Further detailed comparison can be found in Appendix E.1.

Learning the Shapley Value.
Another relevant line of research is the Shapley approximation, as
the Shapley value is the barycenter of the core in convex games. Therefore, the approximation error
of the Shapley value is an upper bound on the distance between the approximated Shapley value and
the expected core [4, 18, 24]. It is worth noting that, in contrast to the Shapley approximation method,
our algorithm ensures the return of a point in the core. In comparison, the Shapley approximation
approach can only provide an allocation with a bounded distance from the core.

3
Problem Description

3.1
Preliminaries
Notations.
For k ∈N+, denote [k] as set {1, 2, . . . , k}. For n ∈N+, let En be the n-dimensional
Euclidean space, and let us denote D as the Euclidean distance in En. Denote 1n as the vector
[1, ..., 1] ∈Rn. Denote ⟨·, ·⟩as the dot product. For a set C, we denote C \ x as the set resulting from
eliminating an element x in C. For C ⊂En, let diam(C) := maxx,y∈C D(x, y), and Conv (C)
denote the diameter and the convex hull of C, respectively.

Denote Sn := {ω : [n] →[n] | ω is a bijection} as the permutation group of [n]. For any collection
of permutations P ⊂Sn, we denote ωp, p ∈[|P|], as pth permutation order in P. Let si := (i, i+1)
denote the adjacent transposition between i and i + 1. Given a set C, we denote by M(C) the space
of all probability distributions on C.

Stochastic Cooperative Games.
A stochastic cooperative game is defined as a tuple (N, P),
where N is a set containing all agents with number of agents to be |N| = n, and P =
{PS ∈M([0, 1]) | S ⊆N} is the collection of reward distributions with PS to be the reward distri-
bution w.r.t. the coalition S. For any coalition S ⊆N, we denote µ(S) := Er∼PS[r] as the expected
reward of coalition S. For a reward allocation scheme x ∈Rn, let x(S) := P

i∈S xi as the total re-
ward allocation for players in S. A reward allocation x is effective if x(N) = µ(N). The hyperplane
of all effective reward allocations, denoted by HN, is defined as HN = {x ∈Rn | x(N) = µ(N)}.
The convex stochastic cooperative game can be defined as follows:

Definition 1 (Convex stochastic cooperative game). A stochastic cooperative game is convex if the
expected reward function is supermodular [23], that is,

µ (S ∪{i}) −µ(S) ≥µ (C ∪{i}) −µ(C)) ,
∀i /∈S ∪C and ∀C ⊆S ⊆N.
(1)

Next, we define the notion of strict convexity as follows:

Definition 2 (ς-Strictly convex cooperative game). For some constant ς > 0, a cooperative game
is ς-strictly convex if the expected reward function is ς-strictly supermodular [23], that is, µ is
supermodular and

µ(S ∪{i}) −µ(S) ≥µ(C ∪{i}) −µ(C) + ς,
∀i /∈S ∪C and ∀C ⊆S ⊆N.
(2)

3


---Page Break---
Following [20], we define the expected core as follows:

Definition 3 (Expected core [20]). The core is defined as

E-Core := {x ∈Rn | x(N) = µ(N); x(S) ≥µ(S), ∀S ⊆N}.

That is, the E-Core is the collection of all effective reward allocation schemes x (i.e., schemes where
the total allocation adds up to the expected reward of the grand coalition N - see the definition of
effective reward allocation above), where if any agents deviate from the grand coalition N to form a
smaller team S, regardless of how they allocate the reward, each individual will not receive more
expected reward than if they had stayed in N. Note that, as E-Core ⊂HN, its dimension is at most
(n −1). We say that E-Core is full dimensional whenever its dimension is n −1.

In convex games, each vertex of the core in the convex game is a marginal vector corresponding to a
permutation order [23]. This is a special property of convex games, which plays a crucial role in our
algorithm design. For any ω ∈Sn, define the marginal vector ϕω ∈Rn corresponding to ω, that is,
its ith entry is
ϕω
i := µ(P ω(i)) −µ(P ω(i) \ i),
(3)

where P ω
i = {j | ω(j) ≤ω(i)}. It is known from [23] that if the expected reward function is strictly
supermodular, the E-Core must be full dimensional.

3.2
Problem Setting
In our setting we assume that there is a principal who does not know the reward distribution P. In
each round t, the principal queries a coalition St ⊂N. The environment returns a vector rt ∼PSt
independently of the past. For simplicity, we assume that the agent knows the expected reward of
the grand coalition µ(N). Additionally, we assume that the convexity of the game, that is, µ is
supermodular. Our question is how many samples are needed so that with high probability 1 −δ, the
algorithm returns a point x ∈E-Core. More particularly, we ask whether or not there is an algorithm
that can output a point in the E-Core, with probability at least 1 −δ and the number of queries

T = poly(n, log(δ−1)).
(4)

As well shall show in Theorem 7, if E-Core is not full-dimensional, no algorithm can output a point
in E-Core with finite samples. As such, to guarantee the learnability of the E-Core. From now on in
the rest of this paper, we assume that:

Assumption 4. The game is ς-strictly convex.

Note that strict convexity immediately implies full dimensionality [23], which is not the case with
convexity (refer to Section 5). As we shall show in the next sections, strict convexity is a sufficient
condition allowing the principal to learn a point in E-Core with polynomial number of samples.
Practical examples of strictly convex games can be found in appendix D.

4
Learning the Expected Core

In this section, we propose a general-purpose Common-Point-Picking algorithm that is able to
return a point of unknown full-dimensional simplex, given an oracle that provides a noisy position of
the simplex’s verticies. Under the assumption that the game is strictly convex, we show that, when
applying Common-Points-Picking algorithm to the ς-strictly convex game, it can return a point in
E-Core provided the number of samples is poly(n, ς).

4.1
Geometric Intuition
Given E-Core polytope of dimension (n −1). In deterministic case when the knowledge of the
game is perfect, to compute a point in the core, one can query a marginal vector corresponding
to a permutation order ω ∈Sn [23]. Given that we have uncertainty in the estimation of E-Core,
this approach is no longer applicable. The reason is that for each vertex ϕω, we do not precisely
compute its position. Instead, we only have information on its confidence set C(ϕω, δ), a compact
(n −1)-dimensional set. The confidence set contains ϕω with probability at least (1 −δ), as we will
define shortly in this section.

One approach to overcome the effect of uncertainty is that we can estimate multiple vertices of
the E-Core. Let P ⊂Sn be a collection of permutations, and Q = {ϕωp | ωp ∈P} be the set of

4


---Page Break---
vertices corresponding to P. For brevity, we denote Cp := C(ϕωp, δ), ωp ∈P. In this section, we
assume that the confidence sets contain the true vertices, that is, ϕωp ∈Cp, ∀p ∈[|P|], which can be
guaranteed with high probability. It is clear that Conv (Q) ⊂E-Core, since Q is a subset of vertices
of E-Core. The challenge is that, as the ground truth of the position of the vertex can be any point
in the confidence set, we need to ensure that the algorithm outputs a point in the convex hull of any
collection of points in the confidence sets. A sufficient condition to achieve this is that, given |P|
confidence sets {Cp}p∈[|P|], for each xp ∈Cp,

\

xp∈Cp
p∈[|P|]

Conv
 
{xp}p∈[|P|]

̸= ∅.
(5)

This condition means that there exists a common point among all the convex hulls formed by choosing
any point in confidence sets, xp ∈Cp. We call the above intersection a set of common points. The
reason why it is a sufficient condition is that this set is a subset of a ground-truth simplex, implying
that it is in the E-Core. Formally, we have
\

xp∈Cp
p∈[|P|]

Conv
 
{xp}p∈[|P|]

⊂Conv (Q) ⊂E-Core;
(6)

Algorithm 1 Common Points Picking

1: Input collection of permutation order P =
{ωp}p∈[n].
2: t = 0, ep = 0, Q = ∅
3: while Stopping-Condition (Q, bep) do
4:
ep ←ep + 1;
5:
for p ∈[n] do
6:
for i ∈[n] do
7:
Query P ωp
i
.
8:
Orcale returns rep
 
P ωp
i

←rt.
9:
t ←t + 1.
10:
Computing ˆϕωp
i (ep) as (9).
11:
end for
12:
end for
13:
Assign Q =
n
ˆϕωp(ep)
o

p∈[n]
14:
Compute bep as (10)
15: end while
16: Return x⋆= 1

n
P

ω∈P ˆϕω(ep).

which means that any common point must be in
E-Core. Moreover, the set of common points is
learnable. As we shall show in the next section,
our algorithm can access the set of common
points whenever it is nonempty.

We first state a necessary condition for the num-
ber of vertices of E-Core need to estimate for
(5) can be satisfied:

Proposition 5. Assume that all the confidence
sets are full dimensional, that is, dim(Cp) =
n −1, ∀p ∈[|P|], and suppose that |P| < n,
\

xp∈Cp
p∈[|P|]

Conv
 
{xp}p∈[|P|]

= ∅.
(7)

Proposition 5 implies that one needs to estimate
at least n vertices to guarantee the existence of
a common point.

4.2
Common-Points-Picking Algorithm
As Proposition 5 suggests, we need to estimate at least n vertices. As such, from now on, we assume
that |P| = n. Based on the above intuition, we propose Common-Points-Picking, whose pseudo
code is described in Algorithm 1.

The Common-Points-Picking Algorithm can be described as follows. First, let us explain the
notation. In epoch ep, the variable rep (·) in the algorithm represents the reward value of the enquired
coalition; ˆϕωp(ep) represents the estimated marginal vector w.r.t. ωp. In each epoch ep, assuming that
the stopping condition is not satisfied, the algorithm estimates the marginal vectors corresponding to
the collection of given permutation orders Q =
n
ˆϕωp(ep)
o

p∈[n] (lines 6-10). For each p ∈[n], the

estimation can be done by querying the value of the nested sequence P ωp
1 , P ωp
2 , . . . , P ωp
n
(line 6-7),
then estimating the marginal contribution of each player with respect to the permutation order ωp
(line 10). Next, it calculates the confidence bonus bep of the confidence sets and checks the stopping
condition for the next epoch. The algorithm continues until the stopping condition is satisfied, and
then returns the average of the most recent values of the marginal vectors corresponding to P.

The termination of the Common-Points-Picking algorithm is based on the Stopping-Condition
algorithm
(Algorithm
2),
which
can
be
described
as
follows.
Consider
the
case
where Q
̸=
∅.
For each point xp
∈
Q,
the algorithm attempts to calculate
the separating hyperplane Hp(Q),
that separates xp
from the rest Q \ xp
(line 7).

5


---Page Break---
Algorithm 2 Stopping Condition

1: Input collection of n points Q, confidence
bonus bep.
2: Compute ϵep = 2√nbep.
3: if Q = ∅then
4:
Return FALSE.
5: end if
6: for p ∈[n] do
7:
Computing Hp(Q), i.e.,
compute (vp ∈Rn, cp ∈R) such that.









∥vp∥2
= 1;
⟨vp, x⟩
= cp + ϵep, ∀x ∈Q \ xp.
⟨vp, xp⟩
< cp + ϵep;
⟨vp, 1n⟩
= 0;
(8)

8:
Computing distance:

hp :=
min
x:∥x−xp∥∞<bep cp −⟨vp, x⟩.

9:
if hp < n ϵep then
10:
Return FALSE.
11:
end if
12: end for
13: Return TRUE.

The hyperplane Hp(Q) is defined by two param-
eters (vp ∈Rn, cp ∈R), where vp is one of
its unit normal vectors, together with cp, satis-
fying Eq. (8). Specifically, the second and third
equality in (8) implies that Hp(Q) is parallel
and at a distance of ϵep (toward xp) from the
hyperplane that passes through all the points in
Q \ xp. The fourth equality in (8) guarantees
that Hp(Q) is a subset of HN (as the normal
vector of HN is 1n). After computing Hp(Q),
the algorithm checks whether the distance from
the confidence set Cp to Hp(Q) is large enough
(lines 8-12). The stopping condition checks for
all p ∈[n]; if no condition is violated, then
the algorithm returns TRUE. An example of the
construction of separating hyperplanes in HN,
where n = 3 is depicted in Figure 1.

Note that the input of algorithm P can be any
collection of permutation orders such that |P| =
n. In the next section, we will provide instances
of the collection of permutation orders, in which,
under Assumption 4, the algorithm can output
a point in E-Core with high probability and a
polynomial number of samples.

Remark 6. Most of the computational burden
lies in computing the separating hyperplane
Hp(Q) for each p (line 7), and calculating the
distance between the confidence set Cp and Hp(Q) (line 8) in Stopping Condition. Since all tasks
can be completed within polynomial time w.r.t. n, our algorithm is polynomial.

Figure 1: Set of common points constructed by
separating hyperplanes. The intersection of half-
spaces defined by Hp(Q), ∀p ∈[n], creates a
subset of common points. The common points
are in E-Core, provided that the confidence sets
contain the ground-truth vertices.

There
are
two
challenges
regarding
the
Common-Points-Picking algorithm.
First,
we need to design confidence sets such that they
contain the vertices with high probability, which
can be done easily using Hoeffding’s inequality.
Second, we need to define a stopping condi-
tion that can guarantee the non-emptiness of the
common set and output a point in the common
set with a polynomial number of samples. The
second question is more involved and requires
proving new results in convex geometry, includ-
ing an extension of the hyperplane separation
theorem, as we shall fully explain in Section 5.1.

Confidence Set
To calculate this set we will
use the probability concentration inequality to
obtain a confidence set: First, let rep (∅) =
0, ∀ep > 0, define the empirical marginal vec-
tor w.r.t. permutation ω as ˆϕω ∈Rn at epoch ep
as

ˆϕω
i (ep) = 1

ep

ep
X

s=1
rs (P ω
i ) −rs (P ω
i \ i) .
(9)

By the Hoeffding’s inequality, one has that after ep epochs, ∀ω ∈P, for each i ∈[n], with probability
at least 1 −δ, ϕω must belong to the following set:

C(ϕω, δ) =

x ∈HN



x −ˆϕω
∞≤bep


;
s.t.
bep :=

s

2 log(n ep δ−1)

ep
.
(10)

6


---Page Break---
5
Main Results

Before proceeding to the analysis of Algorithm 1, let us exclude the case where learning a stable
allocation is not possible, thereby emphasizing the need of the strict convexity assumption.

Theorem 7. Suppose that E-Core has dimension k < n −1, for any 0.2 > δ > 0 and with finite
samples, no algorithm can output a point in E-Core with probability at least 1 −δ.

The proof of Theorem 7 employs an information-theoretic argument. In particular, we construct two
game instances with low-dimensional cores, utilising the concept of face games introduced in [14].
The construction of the two games is designed to ensure that, despite the KL distance between their
reward distributions being arbitrarily small, their low-dimensional cores are parallel and maintain a
positive distance from each other. Consequently, their cores have empty intersections. This implies
that, given finite samples, no algorithm can reliably distinguish between the two games. Since these
games lack a mutually stable allocation, if an algorithm fails to differentiate between them, it is likely
to choose the wrong core with a certain probability.

It is worth noting that convex games may have a low-dimensional core, as demonstrated in the
following example.

Example 8. Let µ(S) = |S| for all S ⊆N. It is easy to verify that µ is indeed convex. The marginal
contribution of any player i to any set S ⊆N is

µ(S ∪i) −µ(S) = 1, ∀S ⊂N.
(11)

Therefore, the only stable allocation is 1n, which coincides with the Shapley value. Hence, the core
is one-point set. According to Theorem 7, since the core has a dimension of 0 in this case, it is
impossible to learn a stable allocation with a finite number of samples.

Example 8 suggests that convexity alone does not ensure the problem’s learnability, emphasizing the
requirement for strict convexity.

5.1
On the Stopping Condition

In this subsection, we explain the construction of the stopping condition in Algorithm 2. We will show
that the stopping condition can always be satisfied with the number of samples needed polynomially
w.r.t the width of the ground truth simplex. Intuitively, the confidence sets need to shrink to be
sufficiently small, relative to the width of the simplex, to guarantee the existence of a common point.

To simplify the presentation, we restrict our attention to HN and consider it as En−1. First, we state
a necessary condition for the existence of common points.

Proposition 9. Suppose there is a (n −2)-dimensional hyperplane that intersect with all the interior
of confidence sets Cp, ∀p ∈[n] , then common points do not exist.

Proposition 9 suggests that if the ground truth simplex Conv (Q) is not full-dimensional, then the
common set is empty. In addition, even if Conv (Q) is full-dimensional, but the confident sets are
not sufficiently small, one can also create a hyperplane that intersects with all the confidence sets.
For example, when the intersection of the confidence sets is not empty.

On the other hand, when the confidence sets are well-arranged and sufficiently small, that is, there
does not exist a hyperplane that intersects with all of them, a nice separating property emerges, as
stated in the next theorem. This new result can be considered as an extension of the classic separating
hyperplane theorem [3]. First, let us recap the notion of separation as follows.

Definition 10 (Separating hyperplane). Let C and D be two compact and convex subsets of
En−1. Let H be a hyperplane defined by the tuple (v, c), where v is a unit normal vector and c is
a real number, such that ⟨x, v⟩= c, ∀x ∈H. We say H separates C and D if ⟨x, v⟩> c, ∀x ∈
C; and ⟨y, v⟩< c, ∀y ∈D.

Theorem 11 (Hyperplane separation theorem for multiple convex sets). Assume that {Cp}p∈[n]
are mutually disjoint compact and convex subsets in En−1. Suppose that there does not exist a
(n −2)-dimensional hyperplane that intersects with confidence sets Cp, ∀p ∈[n], then for each
p ∈[n], there exists a hyperplane that separates Cp from S

q̸=p
Cq.

7


---Page Break---
Remark 12 (Non-triviality of Theorem 11). At a first glance, Theorem 11 may appear as a trivial
extension of the classic hyperplane separation theorem due to the following reasoning: Consider
the union of all hyperplanes that intersect S

q̸=p Cq, which trivially contains S

q̸=p Cq. Then, by
assuming that these hyperplanes do not intersect Cp, the separation between Cp and S

q̸=p Cq appears
to follow from the classic separation hyperplane theorem. However, there is a flaw in the above
reasoning: The union of these hyperplanes is not necessarily convex. Therefore, the classic separation
hyperplane theorem cannot be applied directly. Instead, employing Carathéodory’s theorem, we
prove in Theorem 11 by contra-position that if the intersection between Cp and Conv(S

q̸=p Cq) is
non-empty, then we can construct a low-dimensional hyperplane that intersects with all the set.

When those confidence sets are well-separated, we can provide a sufficient condition for that the
common points exist. Let Hp be a separating hyperplane that separate Cp from S
q̸=p Cq. We define
Hp corresponding with tuple (vp, cp). Now, denote Ep =

x ∈En−1 | ⟨vp, x⟩< cp	
as the half
space containing Cp. We have that:

Lemma 13. For any xp ∈Cp, p ∈[n],

\

p∈[n]
Ep ⊆Conv

{xp}p∈[n]

.
(12)

Consequently, if T

p∈[n]
Ep is nonempty, it is the subset of common points.

From Lemma 13, as T

p∈[n] Ep is the subset of any simplex defined by a set of points in the confidence
sets, T

p∈[n] Ep must be either empty or bounded subset of En−1. The key implication here is that
Lemma 13 provides us a method to find a point in the common set. An example of Lemma 13 in E2
is illustrated in Figure 1.

Now, the main question is under what conditions T

p∈[n] Ep is nonempty. Next we show that the
nonemptiness of T

p∈[n] Ep can be guaranteed if the diameter of the confidence sets is sufficiently
small. This establishes a condition for the number of samples required by the algorithm.

Theorem 14. Given a collection of confident set {Cp}p∈[n] and let Q = {xp}p∈[n], for any xp ∈Cp.
For any p ∈[n], denote Hp(Q) as the (n −1)-dimensional hyperplane with constant (vp, cp),
∥vp∥= 1 such that
⟨vp, x⟩= cp + maxq∈[n]\p diam(Cp),
∀x ∈Q \ xp.
⟨vp, xp⟩< cp + maxq∈[n]\p diam(Cp).
(13)

For all p ∈[n], if the following holds

D(Cp, Hp(Q)) > 2n

max
q∈[n]\p diam(Cq)

;
(14)

then there exists a common point. In particular, the point x⋆= 1

n
P

p∈[n] xp is a common point.

Intuitively, Theorem 14 states that if the distance from a confidence set Cp to the hyperplane Hp(Q)
is relatively large compared to the sum of the diameters of all other confidence sets, then the average
of any collection of points in the confidence set must be a common point. As such, Theorem 14
determines the stopping condition for Algorithm 1 and provide us a explicit way to find a common
point, which validates the correctness of Algorithm 1. In particular, Algorithm 2 checks if conditions
(14) are satisfied for the confidence sets in each round. If the conditions are satisfied, then Algorithm
1 stops sampling and returns x⋆as the common point.

Note that while the diameters of confidence sets can be controlled by the number of samples regarding
the marginal vector, D(Cp, Hp(Q)) is a random variable and needs to be handled with care. We show
that there exist choices of n vertices such that the simplex formed by them has a sufficiently large
width, resulting in the stopping condition being satisfied with high probability after poly(n, ς−1)
number of samples.

8


---Page Break---
5.2
Sample Complexity Analysis

Now, we show that, the conditions of Theorem 14 can be satisfied with high probability. The distance
D(Cp, Hp(Q)), p ∈[n] can be lower bounded by the width of the ground-truth simplex, which is
defined as follows:

Definition 15 (Width of simplex). Given n points {x1, ...xn} in Rn, let matrix P = [xi]i∈[n],
we define the matrix of coordinates of the points in P w.r.t. xi as coM(P, i) := [(xj −xi)]j̸=i ∈
Rn×(n−1). Denote σk(M) as the kth singular value of matrix M (with descending order). We define
the width of the simplex whose coordinate matrix is P as follows

ϑ(P) := min
i∈[n] σn−1 (coM(P, i)) .
(15)

Lemma 16. Given n points {x1, ..., xn} in Rn, let M be the matrix corresponding to these points,
assume that 0 < Mij < 1 and ϑ(M) ≥σ, for some constant σ > 0. Let R ∈Rn×n be a perturbation
matrix, such that its entries |Rij| < ϵ/2, ∀(i, j), and 0 < ϵ < σ2/3n3. Let hmin be a smallest
magnitude of the altitude of the simplex corresponding to the matrix M + R. One has that

hmin ≥
p

σ2 −6n3ϵ.
(16)

Lemma 16 guarantees that if the width of the ground truth simplex is relatively large compared to
the diameter of the confidence set, then the heights of the estimated simplex are also large. We now
provide an example of a collection of permutation orders corresponding to a set of vertices as follows.

Proposition 17. Fix any ω ∈Sn, consider the collection of permutation P = {ω, ωs1, . . . , ωsn−1}
and matrix M = [ϕω′]ω′∈P. The width of the simplex that corresponds to M, is upper bounded as
ϑ(M) ≥0.5ςn−3/2.

The vertex set in Proposition 17 comprises one vertex and its (n −1) adjacent vertices. Combining
Lemma 16, Proposition 17 with the stopping condition provided by Theorem 14, we now can
guarantee the sample complexity of our algorithm:

Theorem 18. With the choice of collection of permutation order P as in Proposition 17, and suppose
that Assumption 4 holds. Then, for any δ ∈[0, 1], if the number of samples is bounded by

T = O
n15 log(nδ−1ς−1)

ς4


,
(17)

the Common-Points-Picking algorithm returns a point in E-Core with probability at least 1 −δ.

While the choice of vertices in Proposition 17 achieves polynomial sample complexity, the width of
the simplex decreases with dimension growth, hindering its sub-optimality. An alternative choice
of vertices is those corresponding to cyclic permutation, denoted as Cn ⊂Sn, which have a larger
width in large subsets of strictly convex games (as observed in simulations) but can be difficult to
verify in the worst case. We refer readers to Appendix A.4 for the detail simulation and discussion on
the choice of set of n vertices. Based on this observation, we achieve the sample complexity which
better dependence on n as follows.

Theorem 19. Suppose Assumption 4 holds. Let P = Sn the collection of cyclic permutations, and
denote the coordinate matrix of the corresponding vertices as W. Assume that the width of the
simplex ϑ(W) ≥nς

cW for some cW > 0. Then, for any δ ∈[0, 1],if number of samples is

T = O
n5c4
W log(ncW δ−1ς−1)

ς4


,
(18)

the Common-Points-Picking algorithm returns a point in E-Core with probability at least 1 −δ.

It is worth noting that our algorithm does not require information about the constants of the game
ς, cW ; instead, the number of samples required automatically scales with these constants. This
indicates that our algorithm is highly adaptive and requires fewer samples for benign game instances.

9


---Page Break---
Remark 20 (Comment on sample complexity lower bounds). Deriving a lower bound is indeed
important, but comes with several significant challenges. E.g., one possible direction is to extend
the game instances in Theorem 7. However, there are two key technical issues with this idea: (1)
Modifying the face game instance to ensure strict convexity is challenging; (2) It remains unclear
how to generalize two face-game instances into poly(n) game instances such that their cores do
not intersect and the statistical distance of the reward can be upper bounded, which is crucial for
showing poly(n) dependencies in the lower bound (we refer the reader to appendix E.2 for further
discussions). Given the unresolved challenges, deriving a lower bound remains an open question.

6
Experiment

To illustrate the sample complexity of our algorithm in practice and compare it with our theoretical
upper bound, we have conducted a simulation as described below. Code is available at: https:
//github.com/NamTranKekL/ConstantStrictlyConvexGame.git.

Simulation setting: We generate convex game of n players with the expected reward function f
defined recursively as follows: For each S ⊂N s.t. i /∈S,

f(S ∪{i}) = f(S) + |S| + 1 + 0.9ω.
for some ω sampled i.i.d. from the uniform distribution Unif([0, 1]). We then normalize the value
of the reward function within the range [0, 1]. It is straightforward to verify that the strict convexity
constant is ς ≈0.1/n. From the simulation results in Figure 2 (LHS), we can see that the growth
pattern nearly matches that of the theoretical bound given in Theorem 19, indicating that our
theoretical bound is highly informative.

Moreover, to demonstrate that our algorithm is robust even when the strict convexity assumption is
violated, we ran a simulation where the characteristic function is only convex, i.e., the strict convexity
constant is arbitrarily small, as follows:

f(S ∪{i}) = f(S) + |S| + 1 + ω.

We use the cyclic permutations Cn as the input for the algorithm. In Figure 2 (RHS), one can see that
the number of samples required as n grows is sub-exponential, indicating that our algorithm is robust
when the strict convexity assumption is violated.

Figure 2: Simulation with game of n ∈{2, ..., 10} players, where the strict convexity constant ς is
0.1/n in the LHS and 0 in the RHS.

7
Conclusion and Future Work

In this paper, we address the challenge of learning the expected core of a strictly convex stochastic
cooperative game. Under the assumptions of strict convexity and a large interior of the core, we
introduce an algorithm named Common-Points-Picking to learn the expected core. Our algorithm
guarantees termination after poly
 
n, log(δ−1), ς−1
samples and returns a point in the expected core
with probability (1 −δ). For future work, we will investigate whether the sample complexity of our
algorithm can be further improved by incorporating adaptive sampling techniques into the algorithm,
along with developing a lower bound for the class of games.

10


---Page Break---
References

[1] David Aadland and Van Kolpin. Shared irrigation costs: an empirical and axiomatic analysis.
Mathematical Social Sciences, 35(2):203–218, 1998.

[2] Stefan Ambec and Lars Ehlers. Sharing a river among satiable agents. Games and Economic
Behavior, 64(1):35–50, 2008.

[3] Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press,
2004.

[4] Javier Castro, Daniel Gómez, and Juan Tejada. Polynomial calculation of the shapley value
based on sampling. Computers and Operations Research, 36, 2009. ISSN 03050548. doi:
10.1016/j.cor.2008.04.004.

[5] Georgios Chalkiadakis and Craig Boutilier. Bayesian reinforcement learning for coalition
formation under uncertainty. Proceedings of the Third International Joint Conference on
Autonomous Agents and Multiagent Systems, AAMAS 2004, 3, 2004.

[6] Georgios Chalkiadakis and Craig Boutilier. Sequentially optimal repeated coalition formation
under uncertainty. Autonomous Agents and Multi-Agent Systems, 24, 2012. ISSN 13872532.
doi: 10.1007/s10458-010-9157-y.

[7] Georgios Chalkiadakis, Evangelos Markakis, and Craig Boutilier. Coalition formation under
uncertainty: Bargaining equilibria and the bayesian core stability concept. Proceedings of the
International Conference on Autonomous Agents, 2007. doi: 10.1145/1329125.1329203.

[8] Georgios Chalkiadakis, Edith Elkind, and Michael Wooldridge. Computational aspects of
cooperative game theory. Synthesis Lectures on Artificial Intelligence and Machine Learning,
16, 2011. ISSN 19394608. doi: 10.2200/S00355ED1V01Y201107AIM016.

[9] A. Charnes and Daniel Granot. Coalitional and chance-constrained solutions to n -person
games, ii: Two-stage solutions. Operations Research, 25, 1977. ISSN 0030-364X. doi:
10.1287/opre.25.6.1013.

[10] Xin Chen and Jiawei Zhang. A stochastic programming duality approach to inventory central-
ization games. Operations Research, 57, 2009. ISSN 0030364X. doi: 10.1287/opre.1090.0699.

[11] Xuan Vinh Doan and Tri Dung Nguyen. Robust stable payoff distribution in stochastic coopera-
tive games. Operations Research, 2014.

[12] Jakob N. Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon
Whiteson. Counterfactual multi-agent policy gradients. In 32nd AAAI Conference on Artificial
Intelligence, AAAI 2018, 2018.

[13] Ian Gemp, Marc Lanctot, Luke Marris, Yiran Mao, Edgar Duéñez Guzmán, Sarah Perrin,
Andras Gyorgy, Romuald Elie, Georgios Piliouras, Michael Kaisers, Daniel Hennes, Kalesha
Bullard, Kate Larson, and Yoram Bachrach. Approximating the core via iterative coalition
sampling. In Proceedings of the 23rd International Conference on Autonomous Agents and
Multiagent Systems, 2024.

[14] Julio González-Díaz and Estela Sánchez-Rodríguez. Cores of convex and strictly convex games.
Games and Economic Behavior, 62, 2008. ISSN 08998256. doi: 10.1016/j.geb.2007.03.003.

[15] Ilse C.F. Ipsen and Rizwana Rehman. Perturbation bounds for determinants and characteristic
polynomials. SIAM Journal on Matrix Analysis and Applications, 30, 2008. ISSN 08954798.
doi: 10.1137/070704770.

[16] Robert Kleinberg, Aleksandrs Slivkins, and Eli Upfal. Bandits and experts in metric spaces.
Journal of the ACM, 66, 2019. ISSN 1557735X. doi: 10.1145/3299873.

[17] Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In
Neural Information Processing Systems, 2017.

11


---Page Break---
[18] Sasan Maleki, Long Tran-Thanh, Greg Hines, Talal Rahwan, and Alex Rogers. Bounding the es-
timation error of sampling-based shapley value approximation. arXiv preprint arXiv:1306.4265,
2013.

[19] George Pantazis, Filippo Fabiani, Filiberto Fele, and Kostas Margellos. Probabilistically robust
stabilizing allocations in uncertain coalitional games. arXiv preprint arXiv:2203.11322, 3 2022.

[20] George Pantazis, Barbara Franci, Sergio Grammatico, and Kostas Margellos. Distribution-
ally robust stability of payoff allocations in stochastic coalitional games.
arXiv preprint
arXiv:2304.01786, 2023.

[21] Alexander Postnikov.
Permutohedra,
associahedra,
and beyond.
arXiv preprint
arXiv:math/0507163, 7 2005.

[22] Aitazaz Ali Raja and Sergio Grammatico. Payoff distribution in robust coalitional games on
time-varying networks. IEEE Transactions on Control of Network Systems, 9 2020.

[23] Lloyd S. Shapley. Cores of convex games. International Journal of Game Theory, 1, 1971.
ISSN 00207276. doi: 10.1007/BF01753431.

[24] Grah Simon and Thouvenot Vincent. A projected stochastic gradient algorithm for estimating
shapley value applied in attribute importance. In Lecture Notes in Computer Science, volume
12279 LNCS, 2020. doi: 10.1007/978-3-030-57321-8_6.

[25] Milan Studený and Tomáš Kroupa. Core-based criterion for extreme supermodular functions.
Discrete Applied Mathematics, 206, 2016.

[26] Jeroen Suijs and Peter Borm. Stochastic cooperative games: superadditivity, convexity, and
certainty equivalents. Games and Economic Behavior, 27, 1999. ISSN 08998256. doi:
10.1006/game.1998.0672.

[27] Jeroen Suijs, Peter Borm, Anja De Waegenaere, and Stef Tijs.
Cooperative games with
stochastic payoffs. European Journal of Operational Research, 113, 1999. ISSN 03772217.
doi: 10.1016/S0377-2217(97)00421-9.

[28] Mukund Sundararajan and Amir Najmi. The many shapley values for model explanation. In
Proceedings of the 37th International Conference on Machine Learning, 2020.

[29] Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zam-
baldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore
Graepel. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint
arXiv:1706.05296, 6 2017.

[30] Jianhong Wang, Yuan Zhang, Yunjie Gu, and Tae-Kyun Kim. Shaq: Incorporating shapley
value theory into multi-agent q-learning. Advances in Neural Information Processing Systems,
5 2022.

[31] Tom Yan and Ariel D. Procaccia. If you like shapley then you’ll love the core. In AAAI
Conference on Artificial Intelligence, 2021.

[32] Miguel Ángel Mirás Calvo, Carmen Quinteiro Sandomingo, and Estela Sánchez Rodríguez.
The boundary of the core of a balanced game: face games. International Journal of Game
Theory, 49, 2020. ISSN 14321270. doi: 10.1007/s00182-019-00703-2.

12


---Page Break---
Contents of Appendix

A Preliminary and Convex Game
13

A.1
Proof of Theorem 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13

A.2
E-Core of convex games and Generalised Permutahedra . . . . . . . . . . . . . . .
15

A.3
Proof of Proposition 17 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

A.4 Alternative choice of n vertices of E-Core . . . . . . . . . . . . . . . . . . . . . .
17

B
On the Stopping Condition
20

C Sample Complexity Analysis
23

D Examples of Strictly Convex Games
26

E
Further Discussions
27

E.1
Comparison with Pantazis et al. [20] . . . . . . . . . . . . . . . . . . . . . . . . .
27

E.2
Comment on Lower Bounds
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

A
Preliminary and Convex Game

A.1
Proof of Theorem 7

Here and onwards, we adopt the following notation convention: for real numbers a, b ∈[0, 1],
KL (a, b) represents the KL-divergence KL (p, q) where p, q are probability distributions on {0, 1}
such that p(1) = a, q(1) = b. In other words,

KL (a, b) = a ln
  a

b

+ (1 −a) ln

1−a
1−b

.

Lemma 21 ([16]). For any 0 < ε < y ≤1, KL (y −ε, y) < ε2/y(1 −y).

Before stating the proof of Theorem 7, let us introduce some extra notations. Given a game G =
(N, P), with the expected reward function µ, we define the following.

• HC(G) := {x ∈Rn | x(C) = µ(C)} is the hyperplane corresponding to the effective
allocation w.r.t coalition C.

• E-Core(G) is the expected core of the game G.

• FC(G) := E-Core(G) ∩HN\C(G) is facet of the E-Core corresponding to the coalition C.

We use the following definition of the face games in Theorem 7, introduced by [14].

Definition 22 (Face Game). Given a game G = (N, P) with µ(S) = Er∼PS[r], ∀S ⊂N. For any
C ⊂N, define a face game G(C) = (N, PC) with µFC(S) = Er∼PC
S [r] such that, for any S ⊂N,

µFC(S) = µ((S ∩C) ∪(N \ C)) −µ(N \ C) + µ(S ∩(N \ C)).
(19)

[14] showed that the expected core of G(C) is exactly the facet of E-Core(G) corresponding C, that
is, E-Core(G(C)) = FC(G). As noted in [32], one can decompose the reward function of the face
game as follows. For any S ⊂N, we have that

µFC(S) = µFC(S ∩C) + µFC(S ∩(N \ C)).
(20)

We now proceed the proof of Theorem 7.

13


---Page Break---
Proof of Theorem 7 . Denote the set convex games with Bernolli reward as GB, that is,

GB = {G = (N, P) | P = {PS}S⊆N; PS ∈M({0, 1}), ∀S ⊆N}.

Face-game instances and the distance between their E-Core.
We first define two games, G0
and G1, with a full-dimensional E-Core, such that G1 is a slight perturbation of G0. Next, we define
face games corresponding to G0 and G1 using the perturbed facet. We then show that the distance
between the cores of these two face games is at least some positive number ε > 0.

Define a strictly convex game G0 := (N, P) ∈GB, such that µ0(S) := Er∼PS[r], and assume
that µ0 is ς-strictly supermodular. Now, fix a subset C ⊂N, let define a perturbed game instance
G1 := (N, Q) ∈GB, with µ1(S) := Er∼QS[r] such that
µ1(C) := µ0(C) −ε;
µ1(S) := µ0(S);
∀S ⊂N, S ̸= C;
(21)

for some 0 < ε < ς. It is straightforward that G1 is (ς −ε)-strictly convex. Therefore, E-Core(G0)
and E-Core(G1) are both full-dimensional.

Fixing a coalition C ⊂N, we now construct the face games from G0, G1 as in Definition 22.
Let G0(C) := (N, PC), G1(C) := (N, QC) ∈GB, whose expected rewards µ0
FC and µ1
FC are
defined by applying (19) to µ0 and µ1 respectively. Now, we consider the difference between the
expected reward function of these two games.





|µ1
FC(S) −µ0
FC(S)| = 0
∀S ⊂N \ C
|µ1
FC(S) −µ0
FC(S)| = ε
∀S ⊆C
|µ1
FC(N \ C) −µ0
FC(N \ C)| = ε.
(22)

As one can always decompose the set S = (S ∩C) ∪(S ∩N \ C), by the decomposibility of the
face game (20), we has that

|µ1
FC(S) −µ0
FC(S)| ≤ε, ∀S ⊂N.
(23)

As the core of face game G0(C) and G1(C) lie on the hyperplane corresponding to the coalition
N \ C, and the distance between the hyperplanes of G0 and G1 is ε, which lower bounds the
distance between the expected core of G0(C) and G1(C). In particular, as E-Core(G0(C)) =
FC(G0) and E-Core(G1(C)) = FC(G1), and |µ1(N \ C) −µ0(N \ C)| = ε, which leads to
D(HN\C(G0), HN\C(G1)) = ε, we have that

D (E-Core(G0(C)), E-Core(G1(C))) ≥ε.
(24)

The KL distance and imposibility of learning low-dimensional E-Core.
We show that, with
probability δ ∈(0, 0.2), any learner cannot distinguish between G0(C) and G1(C) given there
are finite number of samples. We use the information-theoretic framework similar which is well
developed within multi-armed bandit literature.

We first upper bound the KL-distance between PC
S , QC
S ,
∀S
⊂
N.
Denote c1
:=
minS⊂N
 
µ0
FC(S)(1 −µ0
FC(S))

> 0, by Lemma 21, we have that

KL
 
PC
S , QC
S

= KL
 
µ0
FC(S), µ1
FC(S)

≤ε2

c1
,
∀S ⊂N.

Define the probability space Ψ = 2N × {0, 1}. Fix any algorithm (possibly randomised) A. At
round t, denote (St, rt) ∈Ψ as the coalition selected by the algorithm and the reward return by the
environment. At round s < t, denote νt
0, νt
1 as the probability distribution over Ψt determined by A
and P, Q accordingly.

14


---Page Break---
We have the following, as stated in the appendix of [16]. For any u < t, one has that,

KL (νu
0 , νu
1 ) =
X

ψu−1∈Ψu−1
νu
0 (ψu) log
νu
0 (ψu | ψu−1)
νu
1 (ψu | ψu−1)



=
X

ψu−1∈Ψu−1
νu
0 (ψu) log
νu
0 (Su | ψu−1)
νu
1 (Su | ψu−1) · νu
0 (ru | Su, ψu−1)
νu
1 (ru | Su, ψu−1)



=
X

ψu−1∈Ψu−1
νu
0 (ψu) log
νu
0 (ru | Su, ψu−1)
νu
1 (ru | Su, ψu−1)



[As the distribution of Su depends only on A, not on the distribution νt
0, νt
1.]

=
X

ψu−1∈Ψu−1

X

Su∈2N

X

ru∈{0,1}
νu
0 (ru | Su, ψu−1) log
νu
0 (ru | Su, ψu−1)
νu
1 (ru | Su, ψu−1)


νu
0 (Su, ψu−1)

=
X

ψu−1∈Ψu−1

X

Su∈2N
KL
 
µ0
FC(Su), µ1
FC(Su)

νu
0 (Su, ψu−1)

≤ε2

c1
.

The last inequality hold because KL
 
µ0
FC(S), µ1
FC(S)

≤ε2

c1 , ∀S ∈2N.

We have that

KL
 
νt
0, νt
1

=

t
X

u=1
KL (νu
0 , νu
1 ) ≤tε2

c1
.
(25)

As we can choose ε to be arbitrarily small, we can choose ε such that KL (νt
0, νt
1) ≤0.1.

Now, define the event E as the event that A outputs a point in E-Core(G0(C)), assume that νt
0(E)
with probability at least 0.8. Note that, as E-Core(G0(C)) ∩E-Core(G1(C)) = ∅, E represents the
event where the algorithm fails to output a stable allocation with the game instance G1(C). We have
that from [16]’s Lemma A.5,

νt
1(E) ≥νt
0(E) exp

−KL (νt
0, νt
1) + 1/e
νt
0(E)


> 0.8 exp

−0.1 + 1/e

0.8


> 0.3.
(26)

As it holds for any t > 0, this means that for any finite number of samples, with probability at least
0.1, the algorithm will output the incorrect point.

Remark 23. Upon closely examining the face game instances in the proof of Theorem 7, we can
see that even if the E-Core is full-dimensional, but the width of the interior of E-Core is arbitrarily
small, it is still not possible to learn a point in E-Core with high probability and finite samples. To see
this, let us create two perturbed game instances of the face game instances such that their E-Core are
full-dimensional but have an arbitrarily narrow interior. The construction can be done by applying
modification of equation (19) with an arbitrarily small constant ζ > 0 on the game G0, G1, as
follows.
µFC(S) = µ((S ∩C) ∪(N \ C)) −µ(N \ C) + µ(S ∩(N \ C)) + ζ.
(27)
Now, as long as ζ < ε/2, meaning that the width of the interior of their E-Core is less than half of
the distance between the original face games, the distance between their E-Core remains positive.
Therefore, as the KL distance between the reward distributions is arbitrarily small but the two games
do not share any common stable allocation, no algorithm can output a stable allocation of the ground
truth game with high probability and finite samples.

A.2
E-Core of convex games and Generalised Permutahedra

Formulating the coordinates of the vertices of the core can be achieved using the connection between
the core of a convex game and the generalised permutahedron. There is an equivalence between

15


---Page Break---
generalised permutahedra and polymatroids; it was also shown in [25] that the core of each convex
game is a generalised permutahedron.

For any ω
∈
Sn, let Iω
=
(ω(1), ..., ω(n)).
The n-permutahedron is defined as
Conv ({Iω | ω ∈Sn}). A generalised permutahedron can be defined as a deformation of the permu-
tahedron, that is, a polytope obtained by moving the vertices of the usual permutohedron so that the
directions of all edges are preserved [21]. Formally, the edge of the core corresponding to adjacent
vertices ϕω, ϕωsi can be written as

ϕω −ϕωsi = kω,i(eω(i) −eω(i+1)),
(28)

Where, kω,i ≥0, and e1, . . . en are the coordinate vectors in Rn. If the game is ς-strictly convex,
kω,i > ς.

A.3
Proof of Proposition 17
We utilise the formulation of edges of the generalized permutahedron as described in Subsection A.2
to calculate the matrix of coordinates for the vertices of E-Core. Based on the matrix of coordinates,
we now state the proof of Proposition 17.

Proof of Proposition 17. As the set of vertices is ϕω and its n −1 neighbors, there are only two
cases to consider. First, we need to consider the matrix created by using ϕω as the reference, that
is coM(M, 1). As the neighbors have the same roles, bounding the width of the matrices using any
neighbor as a reference point can be done identically. Therefore, we will prove the theorem for
coM(M, 2), and the proof for coM(M, i), i ̸= 1 can be done in the same manner. Let us denote

V = coM(M, 1) =





c1
0
0
· · ·
0
0
−c1
c2
0
· · ·
0
0
0
−c2
c3
· · ·
0
0
...
...
...
...
...
...
...
...
...
...
...
...
0
0
0
· · ·
−cn−2
cn−1
0
0
0
· · ·
0
−cn−1





∈Rn×(n−1),
(29)

U = coM(M, 2) =





−c1
−c1
−c1
−c1
· · ·
−c1
−c1
c1
c1 + c2
c1
c1
· · ·
c1
c1
0
−c2
c3
0
· · ·
0
0
0
0
−c3
c4
· · ·
0
0
...
...
...
...
...
...
...
...
...
...
...
...
...
...
0
0
0
0
· · ·
−cn−2
cn−1
0
0
0
0
· · ·
0
−cn−1





∈Rn×(n−1),
(30)

in which each ci > ς.

We will exploit the following norm inequality in the proof. For any A1, . . . , An ∈R, we use the
following inequality (norm 2 vs. norm 1 of vectors)

n
X

i=1
A2
i ≥(Pn
i=1 Ai)2

n
(31)

Consider V.
Consider a unit vector x = (x1, ..., xn−1). We have

V x =





c1x1
−c1x1 + c2x2
−c2x2 + c3x3
· · ·
−cn−2xn−2 + cn−1xn−1
−cn−1xn−1




(32)

16


---Page Break---
Applying the Ineq. (31) for A1 = c1x1, A2 = −c1x1 + c2x2, An−1 = −cn−2xn−2 + cn−1xn−1,
An = −cn−1xn−1 gives

∥V x∥2 ≥c2
1x2
1
n
≥ς2x2
1
n ;

∥V x∥2 ≥c2
1x2
1 + (−c1x1 + c2x2)2 ≥c2
2x2
2
n
≥ς2x2
2
n ;

. . .

∥V x∥2 ≥ς2x2
n−1
n
.

(33)

Therefore,

n∥V x∥2 ≥ς2(x2
1 + · · · + x2
n−1)
n
= ς2

n
(34)

Therefore ∥V x∥≥ς/n, hence σn−1(V ) ≥ς/n.

Consider U.
Similarly, consider a unit vector x = (x1, ..., xn−1). We have

Ux =





−c1(x1 + x2 + ... + xn−1)
c1(x1 + x2 + ... + xn−1) + c2x2
−c2x2 + c3x3
−c3x3 + c4x4
· · ·
−cn−2xn−2 + cn−1xn−1
−cn−1xn−1





(35)

Applying the Ineq. (31) for A1 = c1(x1+x2+...+xn−1), A2 = c1(x1+x2+...+xn−1)+c2x2, A3 =
−c2x2 + c3x3, A4 = −c3x3 + c4x4, . . . , An−1 = −cn−2xn−2 + cn−1xn−1, An = −cn−1xn−1
gives

Note that

∥Ux∥2 ≥ς2(x1 + x2 + ... + xn−1)2

n
;

∥Ux∥2 ≥c2
1(x1 + x2 + ... + xn−1)2 + (c1(x1 + x2 + ... + xn−1) + c2x2)2 ≥c2
2x2
2
n
≥ς2x2
2
n ;

∥Ux∥2 ≥c2
1(x1 + x2 + ... + xn−1)2 + (c1(x1 + x2 + ... + xn−1) + c2x2)2 + (−c2x2 + c3x3)2 ≥ς2x2
3
n ;

. . .

∥Ux∥2 ≥ς2x2
n−1
n
(36)
Therefore, we also have

n∥Ux∥2 ≥ς2((x1 + x2 + ... + xn−1)2 + x2
2 + ... + x2
n−1)
n
≥ς2x2
1
n2
(37)

From that, we have that

2n∥Ux∥2 ≥ς2 x2
1
n2 + x2
2
n + · · · + x2
n−1

n
≥x2
1 + ... + x2
n−1
n2
= ς2

n2 , as ∥x∥= 1
(38)

That is, ∥Ux∥≥
ς2
√

2n3 . Therefore, σn−1(U) ≥
ς2
√

2n3 .

Therefore, we have that ϑ(M) >
ς2
√

2n3 .

A.4
Alternative choice of n vertices of E-Core
In this subsection, we provide an alternative choice of vertices rather than that in Proposition 17.
Recall that, with the choice of vertices in Proposition 17, the lower bound for the width of the simplex

17


---Page Break---
diminishes when the dimension increases. This leads to a large dependence of the sample complexity
on n. To mitigate this, we investigate other choices of n vertices. To see this, we first recall the
equivalence between E-Core and generalized permutahedra as explained in Subsection A.2.

However, even in the case of a simple permutahedron, if the set of vertices is not carefully chosen,
the width of their convex can be proportionally small w.r.t. n, as demonstrated in the next proposition.
In particular, the same choice of vertices as in 17 results in the simplex with diminishing width as
follows.

Proposition 24. Consider a permutahedron, fix ω
∈
Sn, consider the matrix W
=
[ϕω, Iωs1, Iωs2, . . . , Iωsn−1]. The width of the simplex that corresponds to M, is upper bounded as
follows:

ϑ(M) ≤3

n.
(39)

Proof. The coordinate matrix w.r.t. ϕω, that is, coM(M, 1) can be written as follows.

V =





1
0
0
· · ·
0
0
−1
1
0
· · ·
0
0
0
−1
1
· · ·
0
0
...
...
...
...
...
...
...
...
...
...
...
...
0
0
0
· · ·
−1
1
0
0
0
· · ·
0
−1





∈Rn×(n−1)
(40)

Therefore, the Gram matrix is

G := V ⊤V =





2
−1
0
0
0
· · ·
0
0
0
−1
2
−1
0
0
· · ·
0
0
0
0
−1
2
−1
0
· · ·
0
0
0
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
...
0
0
0
. . .
0
0
−1
2
−1
0
0
0
. . .
0
0
0
−1
2





∈R(n−1)×(n−1).
(41)

Note that G is a tridiagonal matrix and also Toeplitz matrix, therefore, its minimum eigenvalues has
closed form as follows

λn−1(G) = 2 + 2 cos
(n −1)π

n


= 2 sin2  π

2n


≤5

n2 ;
(42)

as
sin
  π

2n
 ≤
π
2n. Therefore, ϑ(M) ≤σn−1(V ) =
p

λn−1(G) ≤3

n.

Proposition 24 highlights the challenge of selecting a set of vertices such that the width does not
contract with the increasing dimension, even in the case of a simple permutahedron. Denote Cn ⊂Sn
as the group of cyclic permutations of length n. One potential candidate for such a set of vertices is
the collection corresponding to cyclic permutations Cn, as described in the next proposition.

Proposition 25. Consider the matrix W = [Iω]ω∈Cn. We have that

ϑ(W) ≥n

2 .
(43)

18


---Page Break---
Proof. The form of matrix W is as follows

W =





1
n
n −1
. . .
2
2
1
n
. . .
3
3
2
1
. . .
4
...
...
...
...
...
n −1
n −2
n −3
. . .
n −1
n
n −1
n −2
. . .
1





.
(44)

The coordinate matrix w.r.t. the first column is as follows

V = coM(W, 1) =





n −1
n −2
. . .
1
−1
n −2
. . .
1
−1
−2
. . .
1
...
...
...
...
−1
−2
. . .
1
−1
−2
. . .
−(n −1)





.
(45)

Let u ∈Rn−1 be any unit vector, and let z = V u ∈Rn. We have that

zi −zi+1 = nui.
(46)

Let us consider

4 ∥z∥2 = 4z2
1 + 4z2
2 + · · · + 4z2
n
= 2z2
1 + [(z1 + z2)2 + (z1 −z2)2] + [(z2 + z3)2 + (z2 −z3)2]

+ · · · + [(zn−1 + zn)2 + (zn−1 −zn)2] + 2z2
n
≥(z1 −z2)2 + (z2 −z3)2 + · · · + (zn−1 −zn)2

= n2(u2
1 + u2
2 + · · · + u2
n−1) = n2.

(47)

Therefore, we have that

σn−1(V ) =
min
u:∥u∥=1

s

∥V u∥2

∥u∥2
≥n

2 .
(48)

It is straightforward that if one takes any column of W as a reference column, the resulting coordinate
matrices have identical singular values. In particular, for any i, j ∈[n]

coM(W, i) = P · coM(W, j),

where P is a permutation matrix, thus, their singular values are identical. Therefore, we have that

ϑ(W) ≥n

2 .

As a result, the set of vertices corresponding to cyclic permutations is a sensible choice. In case of a
generalised permutahedron, let us define

W := [ϕω]ω∈Cn.
(49)

As generalised permutahedra are deformations of the permutahedron, we expect that ϑ(W) is
reasonably large for a broad class of strictly convex games. In particular, we consider the class of
strictly convex games in which the width ϑ(W) is lower bounded, as in the following assumption:

Assumption 26. The width of the simplex that corresponds to W in (49) is bounded as follows

ϑ(W) ≥nς

cW
,
(50)

for some constant cW > 0.

19


---Page Break---
Figure 3: cW with n ∈{10, 50, 100, 150, 200, 300, 500, 1000}, ς = 0.1

n , and 20000 trials

These parameters will eventually play a crucial role in determining the number of samples required
using this choice of n permutation orders. Although proving an exact upper bound for cW in all
strictly convex games is challenging, we conjecture that cW is relatively small in a large subset of the
games.

To investigate Assumption 26, we conducted a simulation to compute the constant cW
of the minimum singular value σn−1(M).
For each case where n takes values of
(10, 50, 100, 150, 200, 300, 500, 1000), the simulation consisted of 20000 game trials with
ς = 0.1/n. As depicted in Figure 3, the values of cW tend to be relatively small and highly
concentrated within the interval (0, 30). This observation suggests that for most cases of strictly
convex games, cW remains reasonably small. Consequently, our algorithm exhibits relatively low
sample complexity. Code of the experiment is available at: https://github.com/NamTranKekL/
ConstantStrictlyConvexGame.git.

For each case where n takes values of (10, 50, 100, 150, 200, 300, 500, 1000), the simulation
consisted of 20000 game trials with ς = 0.1/n. As depicted in Figure 3, the values of cW tend to be
relatively small and highly concentrated within the interval (0, 30). This observation suggests that
for most cases of strictly convex games, cW remains reasonably small. The results indicate that cW
tends to be relatively small with high probability, and does not depend on the value of n.

B
On the Stopping Condition

Proof of Proposition 5. For each Cp, choose a point in its interior, denote as xp. As there are at most
n −1 points {xp}p∈[|P|], there exists a (n −2)-dimensional hyperplane H that contains {xp}p∈[|P|].
Let ˜H be a hyperplane parallel to H and let the distance D(H, ˜H) be arbitrary small.

As confidence sets are full-dimensional (n −1), ˜H must also intersect with the interiors of all
confidence sets. Since H and ˜H are parallel, any convex hull of points within H and ˜H cannot
intersect. Therefore, there is no common point.

20


---Page Break---
Proof of Proposition 9. The proof spirit is similar to that of Proposition 5.

Let H be the (n −2)-dimensional hyperplane that intersects with the interiors of all confidence sets.
Let ˜H be a hyperplane parallel to H and let the distance D(H, ˜H) be arbitrary small.

As confidence sets are full-dimensional, ˜H must also intersect with the interiors of all confidence sets.
Since H and ˜H are parallel, any convex hull of points within H and ˜H cannot intersect. Therefore,
there is no common point.

The proof of Theorem 11 is a combination of the classic hyperplane separation theorem and the
following lemma.

Lemma 27. Let {Cp}p∈[n] be mutually disjoint compact and convex subsets in En−1. Suppose there
does not exist a (n −2)-dimensional hyperplane that intersects with all confidence sets Cp, ∀p ∈[n],
then for each p ∈[n]

Cp ∩Conv



[

q̸=p
Cq



= ∅.
(51)

Proof. We prove this lemma by contra-position, that is, if there is Cp such that

Cp ∩Conv



[

p̸=q
Cq



̸= ∅;

then there exist a hyperplane that intersects with all the Cp, ∀p ∈[n].

First, assume there is a point x = Cp ∩Conv

 
S

q̸=p
Cp

!

. By Carathéodory’s theorem, there are at

most n points xk ∈S

q̸=p
Cq such that

x =
X

k∈[n]
αkxk.
(52)

As each xk ∈Cq for some Cq, one can rewrite the equation above as

x =
X

q̸=p

X

k: xk∈Cq
αkxk.
(53)

Furthermore, we can write

X

xk∈Cq
αkxk = ˜αq˜xq,
in which,
˜xq :=

P

k: xk∈Cq αkxk
P

k: xk∈Cq αk
,
and
˜αq :=
X

k: xk∈Cq
αk.
(54)

Since Cq is convex, ˜xq ∈Cq. Substituting (54) into (52), one obtains

x =
X

q̸=p
˜αq˜xq.
(55)

Define H as a hyperplane that passes through all ˜xq, we have that x ∈H.

Second, we now show how to construct a hyperplane that intersects with all Cm, m ∈[n]. Let I be
the set of indices such that Cq ∋˜xq. We have two following cases.

(i) First, if |I| = n −1, then H is the (n −2)-dimensional hyperplane that intersect with all
Cm, m ∈[n].

(ii) Second, if |I| < n−1, for any Cq′ ̸= Cp that does not contain any ˜xq, we choose any arbitrary
point xq′ ∈Cq′. As there are n −1 points of ˜xq and xq′, there exists a hyperplane H that
contains all these points. Furthermore, H must contain x, so it is the (n −2)-dimensional
hyperplane that intersects with all sets Cm, ∀m ∈[n].

21


---Page Break---
Now, we state the proof of Theorem 11.

Proof of Theorem 11. As a result of Lemma 27, we have that for all Cp, ∀p ∈[n],

Cp ∩Conv



[

q̸=p
Cq



= ∅.
(56)

Therefore, by the hyperplane separation theorem, there must exist a hyperplane that separates Cp and

Conv

 
S

q̸=p
Cq

!

.

Proof of Lemma 13. Let us denote ∆n as Conv

{xp}p∈[n]

. As there is no hyperplane of dimen-

sion n −2 go through all the set Cp, the simplex ∆n is (n −1) dimensional. We have that
\

p∈[n]
Ep ⊆∆n ⇐⇒∆c
n ⊆
[

p∈[n]
Ec
p;

where Ec
p is the complement of the set Ep.

We will prove the RHS of the above. Consider ˆx ∈∆c
n, as ∆n is full dimensional, ˆx can be uniquely
written as affine combination of the vertices, that is,

ˆx =
X

p∈[n]
λpxp,
X

p∈[n]
λp = 1.

As ˆx ∈∆c
n, there must exist some λk < 0.

Now, we shall prove ˆx ∈Ec
k. Consider the following,


vk, ˆx

=

*

vk,
X

p∈[n]
λpxp
+

= λk

vk, xk
+
X

p̸=k
λp

vk, xp

> λkck + ck X

p̸=k
λp

= ck

(57)

The above inequality holds since

vk, xk
< ck and λk < 0. Therefore, ˆx ∈Ec
k. This means that

∆c
n ⊆
[

k∈[n]
Ec
k.
(58)

Proof of Theorem 14. Before proceeding the main proof, we show two simple consequences of the
construction of Hp(Q), p ∈[n] and the assumption (14).

Fact 1: Consider p ∈[n], Hp(Q) acts as a separating hyperplane for Cp. To see this, assume that
Hp(Q) is not a separate hyperplane for Cp, then there exists zp ∈Cp such that ⟨vp, zp⟩≥cp. From
(13), we have ⟨vp, xp⟩≤cp + maxq∈[n]\p diam(Cp). Then, there are two cases. First, assume that
⟨vp, xp⟩≤cp. As xp, zp ∈Cp and ⟨vp, zp⟩≥cp, there must exist a point x in the line segment
[xp, zp] such that ⟨vp, x⟩= cp. This means that D(Cp, Hp) = 0, which violates assumption (14).
Second, assume that cp ≤⟨vp, xp⟩≤cp + maxq∈[n]\p diam(Cp). Then, we have that

D(Cp, Hp) ≤D(xp, Hp) = | ⟨vp, xp⟩−cp| ≤max
q∈[n]\p diam(Cq).

22


---Page Break---
This also violates assumption (14). This implies that if (14) is satisfied, Hp(Q) must separate Cp
from ∪q̸=pCq.

Fact 2: The distance from any point in Cq from Hp(Q) is bounded as follows. For x ∈Cq, q ̸= p, we
have that
D(x, Hp(Q)) ≤D(x, xq) + D(xq, Hp(Q)) ≤2 max
q′∈[n]\p diam(Cq′).
(59)

Now, we proceed to the main proof. For the ease of notation, we simply write Hp for Hp(Q).

First, from assumption (14), we has that for any p ∈[n],

D(Cp, Hp) = min
x∈Cp D(x, Hp) = min
x∈Cp |cp −⟨vp, x⟩|.
(60)

We have that
min
x∈Cp D(x, Hp) > 2n max
q̸=p diam(Cq)

≥
X

q∈[n]\p
max
x∈Cq D(x, Hp).
(61)

Second, we shows that how to pick a common point which exists when (61) is satisfied. Let us
choose a collection of points xp ∈Cp, p ∈[n], and define

x⋆= 1

n

X

p∈[n]
xp.

Now, we show that x⋆∈Ep, ∀p ∈[n].

For each p ∈[n], consider Hp. We denote

ζpp := cp −⟨vp, xp⟩> 0;
ζpq := ⟨vp, xq⟩−cp > 0,
q ̸= p.

Note that D(x, Hp) = | ⟨vp, x⟩−cp|. Follows (61), we have that

ζpp ≥min
x∈Cp D(x, Hp) >
X

q∈[n]\p
max
x∈Cq D(x, Hp) ≥
X

q∈[n]\p
ζpq.
(62)

Now, let consider

⟨vp, x⋆⟩= 1

n

X

q∈[n]
⟨vp, xq⟩= 1

n

X

q∈[n]\p
(cp + ζpq) + 1

n(cp −ζpp)

= cp + 1

n



X

q∈[n]\p
ζpq −ζpp



< cp.

(63)

Therefore, x⋆∈Ep. As it is true for all Ep, one has that

x⋆∈
\

p∈[n]
Ep.
(64)

Finally, by Lemma 13, we can conclude that x⋆is a common point.

C
Sample Complexity Analysis

Proof of Lemma 16. Denote ∆as the simplex corresponding to M = [x1, ..., xn], ∆i as the facet
opposite the vertex xi, and hi(∆) is the height of simplex w.r.t. the vertex xi. Denote Volk(C) as the
k-dimensional content of C ⊂En−1, where dim(C) = k. Using simple calculus, one has that

hi(∆) =
1
n −1
Voln−1(∆)
Voln−2(∆i),
(65)

23


---Page Break---
We also denote ˆ∆as the perturbed simplex corresponding to M + R and ˆ∆i as the facet opposite the
to the perturbation of xi.

we bound the height hi( ˆ∆) for all i ∈[n −1]. For the height hn( ˆ∆), one can apply similar reasoning.
Let define the coordinate matrix and the pertubation matrix w.r.t xn as follows

V := coM(M, n),
U := coM(R, n);
(66)

We have that |Uij| < ϵ, ∀i, j. By the definition of width ϑ(M), we have that

σn−1(V ) ≥ϑ(M) ≥σ.
(67)

Let define the Gram matrix and perturbed Gram matrix as follows

G := V ⊤V
ˆG := (V + U)⊤(V + U).
(68)

One has that,
ˆG −G = V ⊤U + U ⊤V + U ⊤U := U.

One has that U ij ≤¯ϵ := 3nϵ, as |Vij| < 1 and |Uij| < ϵ < 1. We also has that ∥U∥2 ≤∥U∥F ≤n¯ϵ.

First step. we bound the quantity |det(G+U)−det(G)|

|det(G)|
. By [15]’s Corollary 2.14, one has that

|det(G + U) −det(G)|

|det(G)|
≤

1 +
∥U∥2
σn−1(G)

n−1
−1 ≤

1 + n¯ϵ

σ2

n
−1.
(69)

As (1 + z)n ≤
1
1−nz when z ∈(0, 1

n) and n > 0. One has that

|det(G + U) −det(G)|

|det(G)|
≤
n2¯ϵ
σ2 −n2¯ϵ,
(70)

when ¯ϵ ≤σ2

n2 , or ϵ ≤
σ2
3n3 . Let us define k := σ2−n2¯ϵ

n2¯ϵ
, one has that

|det(G + U) −det(G)|

|det(G)|
≤1

k .
(71)

It means that

det(G + U) ≥

1 −1

k


det(G)
(72)

Second step. we bound the change in content of the ith facets of the simplex, for i ∈[n −1].
Consider the facet that is opposite to the vertex xi, and denote V (i), Ui as the sub-matrices of V, U
by removing ith column. Denote the Gram matrix

G(i) := V (i)⊤V (i)
ˆG(i) := (V (i) + U(i))⊤(V (i) + U(i))
(73)

Note that one can obtain G(i), ˆG(i) and by removing ith row and column of G(i), ˆG(i) respectively.
Denote U(i) := ˆG(i) −G(i), we has that all entries of U(i) smaller than ¯ϵ.

Moreover, by Singular Value Interlacing Theorem, one has that

σ1(G) ≥σ1(G(i)) ≥σ2(G) ≥σ2(G(i)) ≥· · · ≥σn−2(G(i)) ≥σn−2(G(i)) ≥σn−1(G). (74)

Similarly, one has that

|det(G(i) + U(i)) −det(G(i))|

|det(G(i))|
≤

1 +
∥U(i)∥2
σn−2(G(i))

n−2
−1 ≤

1 + n¯ϵ

σ2

n
−1 ≤1

k .
(75)

24


---Page Break---
It means that

det(G(i) + U(i)) ≤

1 + 1

k


det(G(i)).
(76)

Third step. We bound the height hi corresponding to the vertices xi in this step. For i ∈[n −1] one
has that
Vold( ˆ∆) =
1
(n −1)!

q

det(G + U).

Vold−1( ˆ∆i) =
1
(n −2)!

q

det(G(i) + U(i)).
(77)

Furthermore, by the Eigenvalue Interlacing Theorem, we have det(G)/det(Gi) ≥σn−1(G) ≥σ2.
Putting things together, one has that

hi( ˆ∆) =
1
n −1
Voln−1( ˆ∆)

Voln−2( ˆ∆i)
=

s

det(G + U)
det(G(i) + U(i)) ≥

s

k −1
k + 1
det(G)
det(G(i)) ≥σ

r

σ2 −6n3ϵ

σ2

(78)
We note that, the above holds true for i ∈[n −1].

Fourth Step. Now, we bound the height corresponding to the vertex xn. We can define the
coordination matrix and pertubation matrix w.r.t x1 as follows.

V ′ = coM(M, 1),
U ′ = coM(R, 1).
(79)

Note that, by the definition of the width, we have that

σn−1(V ′) ≥ϑ(M) ≥σ;
(80)

and also, |U ′
ij| ≤ϵ. Similarly, applying Steps 1-3, we also have that

hn( ˆ∆) ≥
p

σ2 −6n3ϵ

Therefore, hi( ˆ∆) ≥
√

σ2 −6n3ϵ holds true for all i ∈[n]. We conclude that

hmin ≥
p

σ2 −6n3ϵ,
(81)

whenever, ϵ ≤
σ2
3n3 .

Proof of Theorem 18. Note that |P| = n. Denote ϵ0 := 2 maxp∈[n] diam(Cp). For any ωp ∈P,
define the event
E = {ϕωp ∈Cp, ∀p ∈[n]}
By the construction of the confidence set, we guarantee that E happen with probability at least
1 −n2δ.

Consider p ∈[n], for any q ∈[n] \ p, let xq be the projection of ϕωq onto Hp, and xp :=
arg minp∈Cp D(x, Hp). We have that

D(xk, ϕωk) ≤ϵ0, ∀k ∈[n].

We need to bound D(xp, Hp) by bounding the minimum height of simplex Conv
 
{xp}p∈[n]

, which
is a pertubation of Conv
 
{ϕωp}p∈[n]

.

Define matrix M = [ϕωp]p∈[n], and ˆ
M = [xp]p∈[n]. Let R := M −ˆ
M be the perturbation matrix,
one has that Rij ≤ϵ0, ∀(i, j). By Lemma 16, we have that

D(xp, Hp) ≥
p

σ2 −12n3ϵ0
(82)

Therefore, for D(xp, Hp) ≥nϵ0 holds , it is sufficient to provide the condition for σ such that
p

σ2 −12n3ϵ0 ≥nϵ0.
(83)

25


---Page Break---
Assuming that ϵ0 < 1, for the condition of Lemma 16 and the above inequality to hold, it is sufficient
to choose

ϵ0 =
σ2

13n3 .

Now, we calculate the upper bound for sample needed. At epoch K, we have that

ϵ0 = 2diam(Cp) ≥4

r

2n log(δ−1)

K

σ = nς

cW
.
(84)

Then we have K = O

n13 log(nδ−1ς−1)

ς4

. As each phase, there are at most n2 queries, then the total
number of sample needed is

T = O
n15 log(nδ−1ς−1)

ς4


(85)

for the algorithm to return a common point, with probability of at least 1 −δ.

Proof of Theorem 19. The proof is identical to that of Theorem 18, with the width of the simplex
bounded by ϑ(W) ≥nς

cW .

D
Examples of Strictly Convex Games

Consider a facility-sharing game (a generalisation of cost-sharing games [2, 1]) where joining
a coalition S would provide each player of that coalition a utility value v(k) where |S| = k,
and they have to pay an average maintenance cost c(k). The expected reward of S is defined
as µ(S) = v(k) −c(k), representing the average utility of its coalitional members. This setting
represents many real-world scenarios, such as:

• University departments together plan to set up and maintain a shared computing lab. The
value of using the lab is the same v(k) = v for each department (e.g., their students can
have access computing facilities), but the maintenance cost c(k) is monotone decreasing and
strictly concave (e.g., the more participate the less the average maintenance cost becomes).
An example to such maintenance cost function is, e.g., c(k) = C1 −C2kα, where α > 1
and C1, C2 are appropriately set constants such that the total maintenance cost kc(k) =
C1k −C2k(α+1) is non-negative and monotone increasing in the [0, n] range (n is the total
number of departments). A coalition S here represents the cooperating departments.

• An international corporate is expanding its international markets via mutual collaborations
with multiple local companies. The cost for each potential local member company to join
the consortium is fixed c (e.g., integration cost), while the benefit they can gain through this
collaboration, v(k), is a strictly monotone convex function in the number of local markets
k (assuming that each local partner is in charge of their own local market), due to the
potential synergies between different markets. An example benefit/utility function is, e.g.,
v(k) = kα with α > 1. The corporate’s task is then to invite local companies to join their
coalition/consortium S.

• (An alternative version of airport games) Airlines decide whether they launch a flight from
a particular airport. The more airlines decide to do so, the higher value v(k) (e.g., more
connection options), and the lower average buy-in cost c(k) (e.g., runway maintenance, staff
cost etc.) each airlines can have. It’s reasonable to assume that v(k) is strictly convex and
monotone increasing (e.g., the number of connecting combinations grows exponentially)
and c(k) is monotone decreasing and strictly concave. Those airlines who decide to invest
into that airport will form a coalition.

For each of the scenarios above, we can see that the expected reward function µ(S) = v(|S|) −c(|S|)
is indeed strictly supermodular. The reason is that v(k) and c(k) are discrete and finite on [0, n], and
thus, we can easily find ς > 0 for which these games also admit ς-strict convexity.

26


---Page Break---
E
Further Discussions

E.1
Comparison with Pantazis et al. [20]
While the algorithm in [20] is proposed for general cooperative games and conceptually applicable to
the class of strictly convex games, we argue that their algorithm is not statistically and computationally
efficient when applied to strictly convex games, due to the absence of a specific mechanism to
leverage the supermodular structure of the expected reward function. In particular, firstly, we argue
that without any modification and with bandit feedback, their algorithm would require a minimum
of Ω(2n) samples. Secondly, although we believe the framework of [20] could be conceptually
applied to strict convex games, significant non-trivial modifications may be necessary to leverage the
supermodular structure of the mean reward function.

Appplying [20] to strictly convex games without any modifications.
We first briefly outline their
algorithmic framework. In this paper, the authors assume that each coalition S ⊂N has access to
a number of samples, denoted as tS > 1. For each coalition S, the empirical mean is denoted as
µtS(S), and a confidence set for the given mean reward is constructed, denoted as,

C(µ(S)) =

ˆµ(S) ∈[0, 1] | |ˆµ(S) −µtS(S)| ≤εtS
	
, for some εtS > 0 .

We note that while the algorithm in [20] constructs the confidence set using Wasserstein distance,
in the case of distributions with bounded support, we can simplify it by using the mean reward
difference. After constructing the confidence set for the mean reward of each coalition, the algorithm
solves the following robust optimization problem:

min
x∈Rn ∥x∥2
2

s.t. x(N) = µ(N)
x(S) ≥sup(C(µ(S)),
∀S ⊂N.

That is, finding the stable allocation for the worst-case scenario within the confidence sets. It is clear
that when directly applying this framework to the bandit setting, each coalition must be queried at
least once, that is tS > 1. This inevitably leads to a complexity of Ω(2n) samples, regardless of the
sampling scheme one employs. In term of computation, with 2n −2 confidence sets for all coalitions
S ⊂N, tabular representation of the confidence set incurs extreme computational cost.

Significant modifications required for [20].
As described above, the algorithm in [20] suffers
from 2n sample complexity, and the main reason is because it requires constructing confidence sets
for the mean reward for all coalitions S ⊂N. As such, if we want to apply their algorithm efficiently
to the bandit setting, we need to address this limitation.

To do so, one may need to develop an approach to design a confidence set for a specific class of
strictly convex games. For instance, we can consider the following approach: Given historical data,
instead of writing a confidence set for each individual coalition, let us define a confidence set for the
mean reward function as follows:

C(µ) =

ˆµ : 2N →[0, 1] | ˆµ ∈[C(µ(S))]S⊂N, ˆµ is strictly supermodular
	
;
(86)

where the confidence set C(µ(S)) could potentially be [0, 1] for some coalition S, as there is no
data available for these coalitions. Let Core(ˆµ) be the core with respect to the reward function ˆµ.
We propose a generalization of the framework from the robust optimization problem to adapt to the
structure of the game as follows.

min
x∈Rn ∥x∥2
2

s.t. x(N) = µ(N)

x ∈
\

ˆµ∈C(µ)
Core(ˆµ).
(87)

That is, we find a stable allocation x for every possible supermodular function within the confidence
set of the reward function.

However, implementing and analyzing this approach may pose significant challenges. The first
challenge lies in constructing a tight confidence set [C(µ(S))]S⊂N such that all functions within

27


---Page Break---
this collection are strictly supermodular. We are not aware of a method to explicitly construct
[C(µ(S))]S⊂N containing only strictly supermodular functions, and we believe this set could poten-
tially be very complicated. To illustrate, consider the scenario where we have samples from two
coalitions, {1} and {1, 2}, with the following empirical means:

µ({1}) = 0.11;
µ({1, 2}) = 0.1

This situation might occurs when the number of samples is insufficient. In such cases, regardless of
the value chosen for the remaining coalition rewards in the function µ(S), µ(S) is not supermodular
(as {1} ⊂{1, 2}, yet µ(1) > µ(1, 2)). Consequently, either the confidence set C(µ(1)) or C(µ(1, 2))
does not contain the empirical mean reward, indicating the highly complicated shape of the confidence
set.

The second challenge is that while computing a stable allocation for a given supermodular reward
function ˆµ is a straightforward task, computing a stable allocation for all supermodular reward
functions in the confidence set C(µ) in a computationally efficient way is an open problem, to the
best of our knowledge.

The discussion above also highlights the key difference between our work and that of [20]: Instead
of explicitly constructing the confidence set of the expected mean reward function to integrate the
supermodular structure for computing a stable allocation, which might be a sophisticated task, we
directly exploit the geometry of the core of strictly convex games. Specifically, in strictly convex
games, each vertex of the core corresponds to a marginal vector with respect to some permutation
orders. Given that one can construct the confidence set of marginal vectors easily, our method is
conceptually and computationally simpler. However, we believe that adopting the more general
framework of robust optimization as presented in [20] is a very interesting, but non-trivial, direction,
and we leave it for future work.

E.2
Comment on Lower Bounds
It is also crucial to develop the lower bound for the class of strict convex game. One promising
direction is to extend the game instances in Theorem 7. However, there could be several technical
issues when it comes to deriving a meaningful lower bound for strictly convex games. The main
problem is twofold: Firstly, not every small perturbation of the face game may result in a strictly
convex game; therefore, careful tailoring to ensure strict convexity is required. Second, to show a
polynomial dependence of sample complexity on n, we need to generalise the two game instances in
Theorem 7 into poly(n) game instances for the information-theoretic argument. It is not clear how to
construct them so that their core has no intersection, and the statistical distance of the reward can be
upper bounded. This can be further detailed as follows:

Firstly, as proven in [14], a face game is only guaranteed to be a convex game, and not all perturbations
of it can result in a strictly convex game. In fact, we believe great care is required to ensure that
the perturbations of the face games are strictly convex, thereby allowing them to be used to derive
lower bounds for strictly convex games. Moreover, even if one can guarantee that the perturbation
of the two face-game instances in Theorem 7 is strictly convex, they can only result in a very loose
lower bound. Particularly, The two game instances in Theorem 7 are originally constructed using two
strictly convex games G0 and G1, whose expected rewards differ in only one coalition, denoted as
C ⊂N. This setup simplifies the computation of the statistical distance of the face-games which are
perturbations of G0 and G1 corresponding to the same coalition C. However, since only two game
instances are used to derive the result of Theorem 7, if we employ fully-dimensional-core perturbed
game instances of them, the resulting lower bound will be independent of the dimension n. In other
words, the finite-sample lower bound can be very loose and does not show any dependence on the
dimension n.

Secondly, generalising the approach to poly(n) game instances is not straightforward, as it requires
choosing poly(n) coalitions to perturb the rewards of the original game G0, not just one coalition.
This results in significant differences in the expected reward functions of the corresponding face
games, as each face game is a perturbation with respect to several different coalitions. Consequently,
upper-bounding the pairwise KL distance between these games is highly nontrivial and would require
sophisticated exploitation of the structure of strictly convex games.

28


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our claims made in the abstract and introduction reflect the paper’s contribu-
tions and scope.

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

Justification: We discussed the limitations of the work in the main paper and the conclusion.

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

29


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: We stated the assumptions in the main paper and provide the proof in the
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

Justification: We described our simulations and provided a link to the code available online.

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

30


---Page Break---
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

Answer: [Yes]

Justification: We provided a link to the code available online.

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

Justification: We described our simulations and provided a link to the code available online.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental
material.

31


---Page Break---
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We described our simulations and statistical significance of the experiments.

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

Justification: We discussed the time complexity of our algorithm.

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

Justification: We have reviewed the NeurIPS Code of Ethics.

32


---Page Break---
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

Justification: This paper is primarily theoretical and may have limited direct impact or
implications.

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

Justification: This paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

33


---Page Break---
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

Justification: We does not use existing assets.

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

Justification: This paper does not release new assets

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

34


---Page Break---
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

35


---Page Break---
