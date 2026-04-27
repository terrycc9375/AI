No-Regret M♮-Concave Function Maximization:
Stochastic Bandit Algorithms and NP-Hardness of
Adversarial Full-Information Setting

Taihei Oki∗
Hokkaido University
Hokkaido, Japan
oki@icredd.hokudai.ac.jp

Shinsaku Sakaue∗
The University of Tokyo and RIKEN AIP
Tokyo, Japan
sakaue@mist.i.u-tokyo.ac.jp

Abstract

M♮-concave functions, a.k.a. gross substitute valuation functions, play a fundamen-
tal role in many fields, including discrete mathematics and economics. In practice,
perfect knowledge of M♮-concave functions is often unavailable a priori, and we
can optimize them only interactively based on some feedback. Motivated by such
situations, we study online M♮-concave function maximization problems, which are
interactive versions of the problem studied by Murota and Shioura (1999). For the
stochastic bandit setting, we present O(T −1/2)-simple regret and O(T 2/3)-regret
algorithms under T times access to unbiased noisy value oracles of M♮-concave
functions. A key to proving these results is the robustness of the greedy algorithm
to local errors in M♮-concave function maximization, which is one of our main
technical results. While we obtain those positive results for the stochastic setting,
another main result of our work is an impossibility in the adversarial setting. We
prove that, even with full-information feedback, no algorithms that run in poly-
nomial time per round can achieve O(T 1−c) regret for any constant c > 0 unless
P = NP. Our proof is based on a reduction from the matroid intersection problem
for three matroids, which would be a novel idea in the context of online learning.

1
Introduction

M♮-concave functions form a fundamental function class in discrete convex analysis [32], and various
combinatorial optimization problems are written as M♮-concave function maximization. In economics,
M♮-concave functions (restricted to the unit-hypercube) are known as gross substitute valuations [19,
13, 25]; in operations research, M♮-concave functions are often used in modeling resource allocation
problems [46, 30]. Furthermore, M♮-concave functions form a theoretically interesting special case
of (DR-)submodular functions that the greedy algorithm can exactly maximize (see, Murota and
Shioura [33], Murota [32, Note 6.21], and Soma [48, Remark 3.3.1]), while it is impossible for the
submodular case [34, 10] and the greedy algorithm can find only approximately optimal solutions [35].
Due to the wide-ranging applications and theoretical importance, efficient methods for maximizing
M♮-concave functions have been extensively studied [33, 46, 30, 20, 39].

When it comes to maximizing M♮-concave functions in practice, we hardly have perfect knowledge of
objective functions in advance. For example, it is difficult to know the exact utility an agent gains from
some items, which is often modeled by a gross substitute valuation function. Similar issues are also
prevalent in submodular function maximization, and researchers have addressed them by developing
no-approximate-regret algorithms in various settings, including stochastic/adversarial environments

∗Equal contribution, alphabetical order.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
and full-information/bandit feedback [49, 14, 43, 54, 16, 36, 37, 38, 51, 55, 11, 40]. On the other
hand, no-regret algorithms for M♮-concave function maximization have not been well studied, despite
the aforementioned importance and practical relevance. Since the greedy algorithm can exactly solve
M♮-concave function maximization, an interesting question is whether we can develop no-regret
algorithms—in the standard sense without approximation—for M♮-concave function maximization.

1.1
Our contribution

This paper studies online M♮-concave function maximization for the stochastic bandit and adversarial
full-information settings. Below are details of our results.

In Section 4, we study the stochastic bandit setting, where we can only observe values of an underlying
M♮-concave function perturbed by sub-Gaussian noise. We first consider the stochastic optimization
setting and provide an O(T −1/2)-simple regret algorithm (Theorem 4.2), where T is the number of
times we can access the noisy value oracle. We then convert it into an O(T 2/3)-cumulative regret
algorithm (Theorem 4.3), where T is the number of rounds, using the explore-then-commit technique.
En route to developing these algorithms, we show that the greedy algorithm for M♮-concave function
maximization is robust to local errors (Theorem 3.1), which is one of our main technical contributions
and is proved differently from related results in submodular and M♮-concave function maximization.

In Section 5, we establish the NP-hardness of no-regret learning for the adversarial full-information
setting. Specifically, Theorem 5.2 shows that unless P = NP, no algorithms that run in polynomial
time in each round can achieve poly(N) · T 1−c regret for any constant c > 0, where poly(N) stands
for any polynomial of N, the per-round problem size. Our proof is based on the fact that maximizing
the sum of three M♮-concave functions is at least as hard as the matroid intersection problem for
three matroids, which is known to be NP-hard.2 We carefully construct a concrete online M♮-concave
function maximization instance that enables reduction from this NP-hard problem. Our high-level
idea, namely, connecting sequential decision-making and finding a common base of three matroids,
might be useful for proving hardness results in other online combinatorial optimization problems.

1.2
Related work

There is a large stream of research on no-regret submodular function maximization. Our stochastic
bandit algorithms are inspired by a line of work on explore-then-commit algorithms for stochastic
bandit problems [37, 38, 11] and by a robustness analysis for extending the offline greedy algorithm
to the online setting [36]. However, unlike existing results for the submodular case, the guarantees of
our algorithms in Section 4 involve no approximation factors. Moreover, while robustness properties
similar to Theorem 3.1 are widely recognized in the submodular case, our proof for the M♮-concave
case substantially differs from them. See Appendix A for a detailed discussion.

Combinatorial bandits with linear reward functions have been widely studied [6, 9, 8, 41], and many
studies have also considered non-linear functions [7, 21, 15, 29]. However, the case of M♮-concave
functions has not been well studied. Zhang et al. [53] studied stochastic minimization of L♮-convex
functions, which form another important class in discrete convex analysis [32] but fundamentally
differ from M♮-convex functions. Apart from online learning, a body of work has studied maximizing
valuation functions approximately from samples to do with imperfect information [2, 3, 4].

Regarding hardness results in online learning, most arguments are typically information-theoretic.
For instance, the minimax regret of hopeless games in partial monitoring is Ω(T) [24, Section 37.2].
By contrast, we establish the NP-hardness of the adversarial full-information online M♮-concave
function maximization, even though the offline M♮-concave function maximization is solvable in
polynomial time. Such a situation is rare in online learning. One exception is the case studied by
Bampis et al. [5]. They showed that no polynomial-time algorithm can achieve sub-linear approximate
regret for some online min-max discrete optimization problems unless NP = RP, even though their
offline counterparts are solvable in polynomial time. Despite the similarity in the situations, the
problem class and proof techniques are completely different. Indeed, while their proof is based on the
NP-hardness of determining the minimum size of a feasible solution, it can be done in polynomial
time for M♮-concave function maximization [47, Corollary 4.2]. They also proved the NP-hardness of

2Note that this fact alone does not immediately imply the NP-hardness of no-regret learning since the learner
can take different actions across rounds and each M♮-concave function maximization instance is not NP-hard.

2


---Page Break---
the multi-instance setting, which is similar to the maximization of the sum of M♮-concave functions.
However, they did not relate the hardness of multi-instance problems to that of no-regret learning.

2
Preliminaries

Let V = {1, . . . , N} be a ground set of size N. Let 0 be the all-zero vector. For i ∈V , let ei ∈RV
denote the ith standard vector, i.e., the ith element is 1 and the others are 0; let e0 = 0 for convenience.
For x ∈RV and S ⊆V , let x(S) = P

i∈S xi. Slightly abusing notation, let x(i) = x({i}) = xi.
For a function f : ZV →R ∪{−∞} on the integer lattice ZV , its effective domain is defined as
dom f := { x ∈ZV : f(x) > −∞}. A function f is called proper if dom f ̸= ∅. We say a proper
function f : ZV →R ∪{−∞} is M♮-concave if for every x, y ∈dom f and i ∈V with x(i) > y(i),
there exists j ∈V ∪{0} with x(j) < y(j) or j = 0 such that the following inequality holds:
f(x) + f(y) ≤f(x −ei + ej) + f(y + ei −ej).
(1)

Similarly, we say f : ZV →R ∪{+∞} is M♮-convex if −f is M♮-concave. If x(V ) ≤y(V ),
M♮-concave functions satisfy more detailed conditions, as follows.
Proposition 2.1 (Corollary of Murota and Shioura [33, Theorem 4.2]). Let f : ZV →R ∪{−∞} be
an M♮-concave function. Then, the following conditions hold for every x, y ∈dom f:
(a) if x(V ) < y(V ), ∃j ∈V with x(j) < y(j), f(x) + f(y) ≤f(x + ej) + f(y −ej) holds.
(b) if x(V ) ≤y(V ), ∀i ∈V with x(i) > y(i), ∃j ∈V with x(j) < y(j), (1) holds.

Let [a, b] = { x ∈ZV : a(i) ≤x(i) ≤b(i) } be an integer interval of a, b ∈(Z ∪{±∞})V and f
be M♮-concave. If dom f ∩[a, b] ̸= ∅, restricting dom f to [a, b] preserves the M♮-concavity [32,
Proposition 6.14]. The sum of M♮-concave functions is not necessarily M♮-concave [32, Note 6.16].
In this paper, we do not assume monotonicity, i.e., x ≤y (element-wise) does not imply f(x) ≤f(y).

2.1
Examples of M♮-concave functions

Maximum-flow on bipartite graphs.
Let (V, W; E) be a bipartite graph, where the set V of N
left-hand-side vertices is a ground set. Each edge ij ∈E is associated with a weight wij ∈R. Given
sources x ∈ZV
≥0 allocated to the vertices in V , let f(x) be the maximum-flow value, i.e.,

f(x) =
max
ξ∈ZE
≥0, y∈ZW
≥0

n P
ij∈E wijξij : ∀i ∈V, P
j:ij∈E ξij = xi; ∀j ∈W, P
i:ij∈E ξij = yj
o
.

This function f is M♮-concave; indeed, more general functions specified by convex-cost flow problems
on networks are M♮-concave [32, Theorem 9.27]. If we restrict the domain to {0, 1}V and regard V
as a set of items, W as a set of agents, and wij ≥0 as the utility of matching an item i with an agent
j, the resulting set function f : {0, 1}V →R≥0 coincides with the OXS valuation function known in
combinatorial auctions [45, 25], which is a special case of the following gross substitute valuation.

Gross substitute valuation.
In economics, an agent’s valuation (a non-negative monotone set
function of items) is said to be gross substitute (GS) if, whenever the prices of some items increase
while the prices of the other items remain the same, the agent keeps demanding the same-priced items
that were demanded before the price change [19, 25]. M♮-concave functions can be viewed as an
extension of GS valuations to the integer lattice [32, Section 6.8]. Indeed, the class of M♮-concave
functions restricted to {0, 1}V is equivalent to the class of GS valuations [13].

Resource allocation.
M♮-concave functions also arise in resource allocation problems [46, 30],
which are extensively studied in the operations research community. For example, given n univariate
concave functions fi : Z →R ∪{−∞} and a positive integer K, a function f defined by f(x) =
Pn
i=1 fi(x(i)) if x ≥0 and x(V ) ≤K and f(x) = −∞otherwise is M♮-concave. More general
examples of M♮-concave functions used in resource allocation are given in, e.g., Moriguchi et al. [30].

More examples can be found in Murota and Shioura [33, Section 2] and Murota [32, Section 6.3]. As
shown above, M♮-concave functions are ubiquitous in various fields. However, those are often difficult
to know perfectly in advance: we may neither know all edge weights in maximum-flow problems,
exact valuations of agents, nor fis’ values at all points in resource allocation. Such situations motivate
us to study how to maximize them interactively by selecting solutions and observing some feedback.

3


---Page Break---
Algorithm 1 Greedy-style procedure with possibly erroneous local updates

1: x0 = 0
2: for k = 1, . . . , K :
3:
Select ik ∈V ∪{0}
▷Standard greedy selects ik ∈arg maxi∈V ∪{0} f(xk−1 + ei).
4:
xk ←xk−1 + eik

2.2
Basic setup

Similar to bandit convex optimization [23], we consider a learner who interacts with a sequence of
M♮-concave functions, f 1, . . . , f T , over T rounds. To avoid incurring f t(x) = −∞, we assume that
dom f 2, . . . , dom f T are identical to dom f 1. We also assume 0 ∈dom f 1 and dom f 1 ⊆ZV
≥0,
which are reasonable in all the examples in Section 2.1. We consider a constrained setting where the
learner’s action x ∈dom f 1 must satisfy x(V ) ≤K. If dom f 1 ⊆{0, 1}V , this is equivalent to the
cardinality constraint common in set function maximization. Let X :=

x ∈dom f 1 : x(V ) ≤K
	

denote the set of feasible actions, which the learner is told in advance. (More precisely, a poly(N)-
time membership oracle of X is given.) Additional problem settings specific to stochastic bandit and
adversarial full-information cases are provided in Sections 4 and 5, respectively.

3
Robustness of greedy M♮-concave function maximization to local errors

This section studies a greedy-style procedure with possibly erroneous local updates for M♮-concave
function maximization, which will be useful for developing stochastic bandit algorithms in Section 4.
Let f : ZV →R ∪{−∞} be an M♮-concave function such that 0 ∈dom f ⊆ZV
≥0, which we want
to maximize under x(V ) ≤K. Let x∗∈arg max{ f(x) : x ∈dom f, x(V ) ≤K } be an optimal
solution. We consider the procedure in Algorithm 1. If f is known a priori and i1, . . . , iK are selected
as in the comment in Step 3, it coincides with the standard greedy algorithm for M♮-concave function
maximization and returns an optimal solution [33]. However, when f is unknown, we may select
different i1, . . . , iK than those selected by the exact greedy algorithm. Given any x ∈ZV and update
direction i ∈V ∪{0}, we define the local error of i at x as

err(i | x) :=
max
i′∈V ∪{0} f(x + ei′) −f(x + ei) ≥0,
(2)

which quantifies how much direction i deviates from the choice of the exact greedy algorithm when x
is given. The following result states that local errors affect the eventual suboptimality only additively,
ensuring that Algorithm 1 applied to M♮-concave function maximization is robust to local errors.

Theorem 3.1. For any i1, . . . , iK ∈V ∪{0}, it holds that f(xK) ≥f(x∗) −PK
k=1 err(ik | xk−1).

Proof. The claim is vacuously true if err(ik | xk−1) = +∞occurs for some k ≤K. Below, we
focus on the case with finite local errors. For k = 0, 1, . . . , K, we define

Yk := { y ∈X : y ≥xk, y(V ) ≤K −k + xk(V ) },

where y ≥xk is read element-wise. That is, Yk ⊆X consists of feasible points that can be reached
from xk by the remaining K −k updates (see Figure 1). Note that x∗∈Y0 and YK = {xK} hold.

To prove the theorem, we will show that the following inequality holds for any k ∈{1, . . . , K}:

max
y∈Yk f(y) ≥max
y∈Yk−1 f(y) −err(ik | xk−1).
(3)

Take yk−1 ∈arg maxy∈Yk−1 f(y) and yk ∈arg maxy∈Yk f(y). If f(yk) ≥f(yk−1), we are done
since err(ik | xk−1) ≥0. Thus, we assume f(yk) < f(yk−1), which implies yk−1 ∈Yk−1 \ Yk.
Then, we can prove the following helper claim by using the M♮-concavity of f.

Helper claim. If yk−1 ∈Yk−1 \ Yk, there exists j ∈V ∪{0} such that yk−1 + eik −ej ∈Yk and

f(xk) + f(yk−1) ≤f(xk −eik + ej) + f(yk−1 + eik −ej).
(4)

Assuming the helper claim, we can easily obtain (3). Specifically, (i) f(yk−1 + eik −ej) ≤f(yk)
holds due to yk−1 + eik −ej ∈Yk and the choice of yk, and (ii) err(ik | xk−1) ≥f(xk−1 + ej) −

4


---Page Break---
i1

i2

xk−1

Yk−1

i1

i2

xk

Yk

i1

i2

Yk

xk

Figure 1: Images of Yk on Z2. The set of integer points in the trapezoid is the feasible region X. Left:
the gray area represents Yk−1 consisting of points reachable from xk−1. Middle: if ik = 0 (case 1),
xk−1 = xk holds and Yk−1 shrinks to Yk, the darker area, since the constraint on y(V ) gets tighter.
Right: if ik = i1 (cases 2 and 3), the area, Yk, reachable from xk = xk−1 + ei1 shifts along ei1.

f(xk−1+eik) = f(xk−eik +ej)−f(xk) holds due to the definition of the local error (2). Combining
these with (4) and rearranging terms imply (3) as follows:

f(yk)
(i)
≥f(yk−1 + eik −ej)
(4)
≥f(yk−1) + f(xk) −f(xk −eik + ej)
(ii)
≥f(yk−1) −err(ik | xk−1).

Given (3), the theorem follows from a simple induction on k = 1, . . . , K. For each k, we will prove

max
y∈Yk f(y) ≥f(x∗) −

k
X

k′=1
err(ik′ | xk′−1).
(5)

The case of k = 1 follows from (3) since x∗∈Y0. If it is true for k −1, (3) and the induction hypoth-
esis imply maxy∈Yk f(y) ≥maxy∈Yk−1 f(y) −err(ik | xk−1) ≥f(x∗) −Pk−1
k′=1 err(ik′ | xk′−1) −
err(ik | xk−1), thus obtaining (5). Since YK = {xK} holds, setting k = K in (5) yields Theorem 3.1.

The rest of the proof is dedicated to proving the helper claim, which we do by examining the following
three cases. The middle (right) image in Figure 1 illustrates case 1 (cases 2 and 3).

Case 1: ik = 0. Due to xk−1 = xk, yk−1 ∈Yk−1 \Yk implies yk−1(V ) = K −(k −1)+xk−1(V ).
Thus, xk(V ) = xk−1(V ) = yk−1(V )−(K−k+1) < yk−1(V ) holds. From Proposition 2.1 (a), there
exists j ∈V with xk(j) < yk−1(j) that satisfies (4). Also, yk−1 ≥xk−1 = xk, xk(j) < yk−1(j),
and (yk−1 −ej)(V ) = K −k + xk−1(V ) = K −k + xk(V ) imply yk−1 −ej ∈Yk.

Case 2: ik ̸= 0 and xk(V ) ≤yk−1(V ). In this case, yk−1 ∈Yk−1 \ Yk implies yk−1 ≥xk−1 and
yk−1 ≱xk = xk−1 + eik, hence xk(ik) > yk−1(ik). From Proposition 2.1 (b), there exists j ∈V
with xk(j) < yk−1(j) that satisfies (4). Since yk−1 ≥xk−1 = xk −eik and xk(j) < yk−1(j), we
have yk−1 + eik −ej ≥xk. Also, we have (yk−1 + eik −ej)(V ) = yk−1(V ) ≤K −(k −1) +
xk−1(V ) = K −k + xk(V ). Therefore, we have yk−1 + eik −ej ∈Yk.

Case 3: ik ̸= 0 and xk(V ) > yk−1(V ). Since yk−1 ∈Yk−1, we have yk−1 ≥xk−1. We also have
yk−1(V ) < xk(V ) = xk−1(V ) + 1. These imply yk−1 = xk−1. Therefore, (4) with j = 0 holds by
equality, and yk−1 + eik = xk ∈Yk also holds.

The robustness property in Theorem 3.1 plays a crucial role in developing stochastic bandit algorithms
in Section 4. Furthermore, the robustness would be beneficial beyond the application to stochastic
bandits since M♮-concave functions often involve uncertainty in practice, as discussed in Section 2.1.
Note that Theorem 3.1 has not been known even in the field of discrete convex analysis and that the
above proof substantially differs from the original proof for the greedy algorithm without local errors
for M♮-concave function maximization [33]. Indeed, the original proof does not consider a set like
Yk, which is crucial in our proof. It is also worth noting that Theorem 3.1 automatically implies the
original result on the errorless case by setting err(ik | xk−1) = 0 for all k. We also emphasize that
while Theorem 3.1 resembles robustness properties known in the submodular case [49, 14, 36, 37,
38, 11], ours is different from them in that it involves no approximation factors and requires careful
inspection of the solution space, as in Figure 1. See Appendix A for a detailed discussion.

5


---Page Break---
4
Stochastic bandit algorithms

This section presents no-regret algorithms for the following stochastic bandit setting.

Problem setting.
For t = 1, . . . , T, the learner selects xt ∈X and observes f t(xt) = f ∗(xt) + εt,
where f ∗: ZV →[0, 1] ∪{−∞} is an unknown M♮-concave function and (εt)T
t=1 is a sequence of
i.i.d. 1-sub-Gaussian noises, i.e., E[exp(λεt)] ≤exp(λ2/2) for λ ∈R.3 Let x∗∈arg maxx∈X f ∗(x)
denote the offline best action. In Theorem 4.2, we will discuss the simple-regret minimization setting,
where the learner selects xT +1 ∈X after the Tth round to minimize the expected simple regret:

sRegT := f ∗(x∗) −E

f ∗(xT +1)

.

Here, the expectation is about the learner’s randomness, which may originate from noisy observations
and possible randomness in their strategy. This is a stochastic bandit optimization setting, where the
learner aims to find the best action without caring about the cumulative regret over the T rounds. On
the other hand, Theorem 4.3 is about the standard regret minimization setting, where the learner aims
to minimize the expected cumulative regret (or the pseudo-regret, strictly speaking):

RegT := T · f ∗(x∗) −E

" T
X

t=1
f ∗(xt)

#

.

In this section, we assume that T is large enough to satisfy T ≥K(N + 2) to simplify the discussion.

Pure-exploration algorithm.
Below, we will use a UCB-type algorithm for pure exploration in the
standard stochastic multi-armed bandit problem as a building block. The algorithm is based on MOSS
(Minimax Optimal Strategy in the Stochastic case) and is known to achieve an O(T −1/2) simple
regret as follows. For completeness, we provide the proof and the pseudo-code in Appendix B.

Proposition 4.1 (Lattimore and Szepesvári [24, Corollary 33.3]). Consider a stochastic multi-armed
bandit instance with M arms and T ′ rounds, where T ′ ≥M. Assume that the reward of the ith arm
in the tth round, denoted by Y t
i , satisfies the following conditions: µi := E[Y t
i ] ∈[0, 1] and Y t
i −µi
is 1-sub-Gaussian. Then, there is an algorithm that, after pulling arms T ′ times, randomly returns
i ∈{1, . . . , M} with µ∗−E[µi] = O(
p

M/T ′), where µ∗:= max{µ1, . . . , µM}.

Given this fact and our robustness result in Theorem 3.1, it is not difficult to develop an O(T −1/2)-
simple regret algorithm; we select ik in Algorithm 1 with the algorithm in Proposition 4.1 consuming
⌊T/K⌋rounds and bound the simple regret by using Theorem 3.1, as detailed below. Also, given
the O(T −1/2)-simple regret algorithm, an O(T 2/3)-regret algorithm follows from the explore-then-
commit technique, as described subsequently. Therefore, we think of these no-regret algorithms
as byproducts and the robustness result in Theorem 3.1 as our main technical contribution on the
positive side. Nevertheless, we believe those algorithms are beneficial since no-regret maximization
of M♮-concave functions have not been well studied, despite their ubiquity as discussed in Section 2.1.

The following theorem presents our result regarding an O(T −1/2)-simple regret algorithm.

Theorem 4.2. There is an algorithm that makes up to T queries to the noisy value oracle of f ∗and
outputs xT +1 such that sRegT = O(K3/2p

N/T).

Proof. Based on Algorithm 1, we consider a randomized algorithm consisting of K phases. Fixing a
realization of xk−1, we discuss the kth phase. We consider the following multi-armed bandit instance
with at most N + 1 arms and ⌊T/K⌋rounds. The arm set is { i ∈V ∪{0} : xk−1 + ei ∈X }, i.e.,
the collection of all feasible update directions; note that the learner can construct this arm set since X
is told in advance. In each round t ∈{1, . . . , ⌊T/K⌋}, the reward of an arm i ∈V ∪{0} is given by
Y t
i = f ∗(xk−1 +ei)+εt, where εt is the 1-sub-Gaussian noise. Let µi = E[Y t
i ] = f ∗(xk−1 +ei) ∈
[0, 1] denote the mean reward of the arm i and µ∗= maxi∈V ∪{0} µi the optimal mean reward. If we
apply the algorithm in Proposition 4.1 to this bandit instance, it randomly returns ik ∈V ∪{0} such
that E[ err(ik | xk−1) | xk−1 ] = µ∗−E[ µik | xk−1 ] = O(
p

KN/T), consuming ⌊T/K⌋queries.

3Restricting the range of f ∗to [0, 1] and the sub-Gaussian constant to 1 is for simplicity; our results extend to
any range and sub-Gaussian constant. Note that any zero-mean random variable in [−1, +1] is 1-sub-Gaussian.

6


---Page Break---
Consider sequentially selecting ik as above and setting xk = xk−1 + eik, thus obtaining x1, . . . , xK.
For any realization of i1, . . . , iK, Theorem 3.1 guarantees f ∗(xK) ≥f ∗(x∗) −PK
k=1 err(ik | xk−1).
By taking the expectations of both sides and using the law of total expectation, we obtain

f ∗(x∗)−E[f ∗(xK)] ≤E

" K
X

k=1
err(ik |xk−1)

#

= E

" K
X

k=1
E[ err(ik |xk−1) | xk−1 ]

#

= O(K
3
2 √

NT).

Thus, xT +1 = xK achieves the desired bound. The number of total queries is K⌊T/K⌋≤T.

We then convert the O(T −1/2)-simple regret algorithm into an O(T 2/3)-regret algorithm by using
the explore-then-commit technique as follows.

Theorem 4.3. There is an algorithm that achieves RegT = O(KN 1/3T 2/3).

Proof. Let ˜T ≤T be the number of exploration rounds, which we will tune later. If we use the algo-

rithm of Theorem 4.2 allowing ˜T queries, we can find x ˜T +1 ∈X with sReg ˜T = O(K3/2
q

N/ ˜T).

If we commit to x ˜T +1 in the remaining T −˜T rounds, the total expected regret is

RegT = E





˜T
X

t=1
f ∗(x∗) −f ∗(xt)



+ (T −˜T) · sReg ˜T ≤˜T + T · sReg ˜T = O( ˜T + TK
3
2
q

N/ ˜T).

By setting ˜T = Θ(KN 1/3T 2/3), we obtain RegT = O(KN 1/3T 2/3).

5
NP-hardness of adversarial full-information setting

This section discusses the NP-hardness of the following adversarial full-information setting.

Problem setting.
An oblivious adversary chooses an arbitrary sequence of M♮-concave functions,
f 1, . . . , f T , where f t : ZV →[0, 1] ∪{−∞} for t = 1, . . . , T, in secret from the learner. Then,
for t = 1, . . . , T, the learner selects xt ∈X and observes f t, i.e., full-information feedback. More
precisely, we suppose that the learner gets free access to a poly(N)-time value oracle of f t by
observing f t since M♮-concave functions may not have polynomial-size representations in general.
The learner aims to minimize the expected cumulative regret:

max
x∈X

T
X

t=1
f t(x) −E

" T
X

t=1
f t(xt)

#

,
(6)

where the expectation is about the learner’s randomness. To simplify the subsequent discussion, we
focus on the case where the constraint is specified by K = N and f 1, . . . , f T are defined on {0, 1}V ;
therefore, the set of feasible actions is X =

x ∈dom f 1 : x(V ) ≤K
	
= {0, 1}V .

For this setting, there is a simple no-regret algorithm that takes exponential time per round. Specifi-
cally, regarding each x ∈X as an expert, we use the standard multiplicative weight update algorithm
to select x1, . . . , xT [26, 12]. Since the number of experts is |X| = 2N, this attains an expected
regret bound of O(
p

T log |X|) ≲poly(N)
√

T, while taking prohibitively long poly(N)|X| ≳2N
time per round. An interesting question is whether a similar regret bound is achievable in polynomial
time per round. Thus, we focus on the polynomial-time randomized learner, as with Bampis et al. [5].
Definition 5.1 (Polynomial-time randomized learner). We say an algorithm for computing x1, . . . , xT
is a polynomial-time randomized learner if, given poly(N)-time value oracles of revealed functions,
it runs in poly(N, T) time in each round, regardless of realizations of the algorithm’s randomness.4

Note that the per-round time complexity can depend polynomially on T. Thus, the algorithm can use
past actions, x1, . . . , xt−1, as inputs for computing xt, as long as the per-round time complexity is
polynomial in the input size. The following theorem is our main result on the negative side.

4While this definition does not cover so-called efficient Las Vegas algorithms, which run in polynomial time
in expectation, requiring polynomial runtime for every realization is standard in randomized computation [1].

7


---Page Break---
Theorem 5.2. In the adversarial full-information setting, for any constant c > 0, no polynomial-time
randomized learner can achieve a regret bound of poly(N) · T 1−c in expectation unless P = NP.5

5.1
Proof of Theorem 5.2

As preparation for proving the theorem, we first show that it suffices to prove the hardness for any
polynomial-time deterministic learner and some distribution on input sequences of functions, which
follows from celebrated Yao’s principle [52]. We include the proof in Appendix C for completeness.
Proposition 5.3 (Yao [52]). Let A be a finite set of all possible deterministic learning algorithms
that run in polynomial time per round and F1:T a finite set of sequences of M♮-concave functions,
f 1, . . . , f T . Let RegT (a, f 1:T ) be the cumulative regret a deterministic learner a ∈A achieves on a
sequence f 1:T = (f 1, . . . , f T ) ∈F1:T . Then, for any polynomial-time randomized learner A and
any distribution q on F1:T , it holds that

max

E

RegT (A, f 1:T )

: f 1:T ∈F1:T 	
≥min

Ef 1:T ∼q

RegT (a, f 1:T )

: a ∈A
	
.

Note that the left-hand side is nothing but the expected cumulative regret (6) of a polynomial-time
randomized learner A on the worst-case input f 1:T . Thus, to prove the theorem, it suffices to show
that the right-hand side, i.e., the expected regret of the best polynomial-time deterministic learner on
some input distribution q, cannot be as small as poly(N) · T 1−c unless P = NP. To this end, we will
construct a finite set F1:T of sequences of M♮-concave functions and a distribution on it.

The fundamental idea behind the subsequent construction of M♮-concave functions is that the matroid
intersection problem for three matroids (the 3-matroid intersection problem, for short) is NP-hard.

3-matroid intersection problem.
A matroid M over V is defined by a non-empty set family B ⊆
2V such that for B1, B2 ∈B and i ∈B1 \ B2, there exists j ∈B2 \ B1 such that B1 \ {i} ∪{j} ∈B.
Elements in B are called bases. We suppose that, given a matroid, we can test whether a given S ⊆V
is a base in poly(N) time. (This is equivalent to the standard poly(N)-time independence testing.)
The 3-matroid intersection problem asks to determine whether three given matroids M1, M2, M3
over a common ground set V have a common base B ∈B1 ∩B2 ∩B3 or not.
Proposition 5.4 (cf. Schrijver [44, Chapter 41]). The 3-matroid intersection problem is NP-hard.

We construct M♮-concave functions that appropriately encode the 3-matroid intersection problem.
Below, for any B ⊆V , let 1B ∈{0, 1}V denote a vector such that 1B(i) = 1 if and only if i ∈B.
Lemma 5.5. Let M be a matroid over V and B ⊆2V its base family. There is a function f :
{0, 1}V →[0, 1] such that (i) f(x) = 1 if and only if x = 1B for some B ∈B and f(x) ≤1 −1/N
otherwise, (ii) f is M♮-concave, and (iii) f(x) can be computed in poly(N) time at every x ∈{0, 1}V .

Proof. Let ∥·∥1 denote the ℓ1-norm. We construct the function f : {0, 1}V →[0, 1] as follows:

f(x) := 1 −1

N min
B∈B∥x −1B∥1
(x ∈{0, 1}V ).

Since 0 ≤∥y∥1 ≤N for y ∈[−1, +1]V , f(x) takes values in [0, 1]. Moreover, minB∈B∥x −1B∥1
is zero if x = 1B for some B ∈B and at least 1 otherwise, establishing (i). Below, we show that f is
(ii) M♮-concave and (iii) computable in poly(N) time.

We prove that τ(x) := minB∈B∥x −1B∥1 is M♮-convex, which implies the M♮-concavity of f. Let
δB : ZV →{0, +∞} be the indicator function of B, i.e., δB(x) = 0 if x = 1B for some B ∈B and
+∞otherwise. Observe that τ is the integer infimal convolution of ∥·∥1 and δB. Here, the integer
infimal convolution of two functions f1, f2 : ZV →R ∪{+∞} is a function of x ∈ZV defined as
(f1 □Z f2)(x) := min{f1(x −y) + f2(y) : y ∈ZV }, and the M♮-concavity is preserved under this
operation [32, Theorem 6.15]. Thus, the M♮-convexity of τ(x) = (∥·∥1 □Z δB)(x) follows from the
M♮-convexity of the ℓ1-norm ∥·∥1 [32, Section 6.3] and the indicator function δB [32, Section 4.1].

Next, we show that τ(x) is computable in poly(N) time for every x ∈{0, 1}V , which implies the
poly(N)-time computability of f(x). As described above, τ is the integer infimal convolution of

5Our result does not exclude the possibility of polynomial-time no-regret learning with an exponential factor
in the regret bound. However, we believe whether poly(N) · T 1−c regret is possible or not is of central interest.

8


---Page Break---
∥·∥1 and δB, i.e., τ(x) = min{∥x−y∥1 +δB(y) : y ∈ZV }. Since the function y 7→∥x−y∥1 is M♮-
convex [32, Theorem 6.15], τ(x) is the minimum value of the sum of the two M♮-convex functions.
While the sum of two M♮-convex functions f1, f2 : ZV →Z≥0 ∪{+∞} is no longer M♮-convex in
general, we can minimize it via reduction to the M-convex submodular flow problem [32, Note 9.30].
We can solve this by querying f1 and f2 values poly(N, log L, log M) times, where L is the minimum
of the ℓ∞-diameter of dom f1 and dom f2 and M is an upper bound on function values [18, 17]. In
our case of f1(y) = ∥x −y∥1 and f2(y) = δB(y), we have L = 1 and M ≤N, and we can compute
f1(y) and f2(y) values in poly(N) time (where the latter is due to the poly(N)-time membership
testing for B). Therefore, τ(x) is computable in poly(N) time, and so is f(x).

Now, we are ready to prove Theorem 5.2.

Proof of Theorem 5.2. Let M1, M2, M3 be three matroids over V and f1, f2, f3 functions defined
as in Lemma 5.5, respectively. Let F1:T be a finite set such that each f t (t = 1, . . . , T) is either f1,
f2, or f3. Let q be a distribution on F1:T that sets each f t to f1, f2, or f3 with equal probability.

Suppose for contradiction that some polynomial-time deterministic learner achieves poly(N) · T 1−c
regret in expectation for the above distribution q. Let T be the smallest integer such that the regret
bound satisfies poly(N) · T 1−c <
T
3N ⇔T > (3Npoly(N))1/c. Note that T is polynomial in N
since c > 0 is a constant. We consider the following procedure.

Run the polynomial-time deterministic learner on the distribution q and obtain xt for t = 1, . . . , T.
If some xt satisfies f1(xt) = f2(xt) = f3(xt) = 1, output “Yes” and otherwise “No.”

If M1, M2, M3 have a common base B ∈B1 ∩B2 ∩B3, we have f1(1B) = f2(1B) = f3(1B) = 1.
On the other hand, if xt ̸= 1B for every B ∈B1∩B2∩B3, E[f t(xt)] ≤1−1

3N holds from Lemma 5.5
and the fact that f t is drawn uniformly from {f1, f2, f3}. Thus, to achieve the poly(N) · T 1−c regret
for T > (3Npoly(N))1/c, the learner must return xt corresponding to some common base at least
once among T rounds. Consequently, the above procedure outputs “Yes.” If M1, M2, M3 have
no common base, none of x1, . . . , xT can be a common base, and hence the procedure outputs
“No.” Therefore, the above procedure returns a correct answer to the 3-matroid intersection problem.
Recall that T is polynomial in N. Since the learner runs in T · poly(N, T) time and we can check
f1(xt) = f2(xt) = f3(xt) = 1 for t = 1, . . . , T in T · poly(N) time, the procedure runs in poly(N)
time. This contradicts the NP-hardness of the 3-matroid intersection problem (Proposition 5.4) unless
P = NP. Therefore, no polynomial-time deterministic learner can achieve poly(N) · T 1−c regret in
expectation. Finally, this regret lower bound applies to any polynomial-time randomized learner on
the worst-case input due to Yao’s principle (Proposition 5.3), completing the proof.

Remark 5.6. One might think that the hardness simply follows from the fact that no-regret learning
in terms of (6) is too demanding. However, similar criteria are naturally met in other problems:
there are efficient no-regret algorithms for online convex optimization and no-approximate-regret
algorithms for online submodular function maximization. What makes online M♮-concave function
maximization NP-hard is its connection to the 3-matroid intersection problem, as detailed in the proof.
Consequently, even though offline M♮-concave function maximization is solvable in polynomial time,
no polynomial-time randomized learner can achieve vanishing regret in the adversarial online setting.

6
Conclusion and discussion

This paper has studied no-regret M♮-concave function maximization. For the stochastic bandit setting,
we have developed O(K3/2p

N/T)-simple regret and O(KN 1/3T 2/3)-regret algorithms. A crucial
ingredient is the robustness of the greedy algorithm to local errors, which we have first established for
the M♮-concave case. For the adversarial full-information setting, we have proved the NP-hardness
of no-regret learning through a reduction from the 3-matroid intersection problem.

Our stochastic bandit algorithms are limited to the sub-Gaussian noise model, while our hardness
result for the adversarial setting comes from a somewhat pessimistic analysis. Overcoming these
limitations by exploring intermediate regimes between the two settings, such as stochastic bandits
with adversarial corruptions [28], will be an exciting future direction from the perspective of beyond
the worst-case analysis [42]. We also expect that our stochastic bandit algorithms have room for

9


---Page Break---
improvement, considering existing regret lower bounds for stochastic combinatorial (semi-)bandits
with linear reward functions. For top-K combinatorial bandits, there is a sample-complexity lower
bound of Ω(N/ε2) for any (ε, δ)-PAC algorithm [41]. Since our O(K3/2p

N/T)-simple regret
bound implies that we can achieve an ε-error in expectation with O(K3N/ε2) samples, our bound
seems tight when K = O(1), while the K factors would be improvable. Regarding the cumulative
regret bound, there is an Ω(
√

KNT) lower bound for stochastic combinatorial semi-bandits [22].
Filling the gap between our O(KN 1/3T 2/3) upper bound and the lower bound is an open problem.
(Since we have assumed T = Ω(KN) in Section 4, our upper bound does not contradict the
lower bound.) We believe that our upper bound is essentially tight considering a recent minimax
regret bound by Tajdini et al. [50] for bandit submodular maximization, which we discuss in detail
below. Regarding the adversarial setting, it will be interesting to explore no-approximate-regret
algorithms. If M♮-concave functions are restricted to {0, 1}V , the resulting problem is a special case
of online submodular function maximization and hence vanishing 1/2-approximate regret is already
possible [43, 16, 36]. We may be able to improve the approximation factor by using the M♮-concavity.

Discussion on the tightness of the O(KN 1/3T 2/3) bound.
As mentioned above, obtaining a
tight regret bound for stochastic bandit M♮-concave maximization is left open. Nevertheless, we
conjecture that our O(KN 1/3T 2/3) bound in Theorem 4.3 is tight unless we admit exponential
factors in K. The rationale behind this conjecture lies in a recent result by Tajdini et al. [50]. They
studied stochastic bandit monotone submodular maximization with a ground set of size N and a
cardinality constraint of K, and they showed that there is a lower bound of

Ω

 

min
i≤K (K −i)N 1/3T 2/3 +

sN −K
i


T

!

on robust greedy regret, which compares the learner’s actual reward with the output of the greedy
algorithm, denoted by Sgr, applied to the underlying true submodular function. This lower bound
implies that the O(KN 1/3T 2/3) regret for stochastic bandit submodular maximization, which can

also be achieved by the explore-then-commit strategy, is inevitable when T is small. The
q N−K
i

T

term can be interpreted as the regret achieved by regrading all
 N−K
i

subsets as arms and using a
UCB-type algorithm. Thus, the lower bound represents the best mix of the two regret terms achieved
by the explore-then-commit and the UCB applied to exponentially many arms.

One can readily observe that the proof of the lower bound by Tajdini et al. [50] does not directly
apply to our stochastic bandit M♮-concave maximization problem. Specifically, the function used
in their proof for obtaining the lower bound is submodular but not M♮-concave. Nevertheless, the
situations of Tajdini et al. [50] and our problem in Section 4, with the domain restricted to {0, 1}V ,
are remarkably similar:

1. Since the greedy algorithm applied to the unknown true M♮-concave function f ∗can find
an optimal solution x∗, we have x∗= Sgr. Therefore, the notion of robust greedy regret in
Tajdini et al. [50] essentially coincides with the standard regret in our case.

2. Both the O(KN 1/3T 2/3) and O
q N−K
i

T

regret bounds discussed above can also be

achieved by the explore-then-commit and UCB strategies, respectively, in our M♮-concave
case, where the former is exactly what our Theorem 4.3 states.

Considering these facts, we expect that we can construct a hard instance of stochastic bandit M♮-
concave maximization similar to Tajdini et al. [50] to establish the same regret lower bound. Thus,
we conjecture that our O(KN 1/3T 2/3) regret bound in Theorem 4.3 is tight in K, N, and T, if we
want to avoid the exponential factor, which generally scales as N K, regardless of the value of T.

Acknowledgements

The authors thank the anonymous reviewers for their valuable feedback, particularly for bringing our
attention to the recent result by Tajdini et al. [50]. This work was supported by JST ERATO Grant
Number JPMJER1903, JST CREST Grant Number JPMJCR24Q2, JST FOREST Grant Number
JPMJFR232L, and JSPS KAKENHI Grant Numbers JP22K17853 and 24K21315.

10


---Page Break---
References

[1]
S. Arora and B. Barak. Computational Complexity: A Modern Approach. Cambridge University
Press, 2009 (cited on page 7).
[2]
M. F. Balcan, F. Constantin, S. Iwata, and L. Wang. Learning valuation functions. In Proceed-
ings of the 25th Annual Conference on Learning Theory, volume 23, pages 4.1–4.24. PMLR,
2012 (cited on page 2).
[3]
E. Balkanski, A. Rubinstein, and Y. Singer. The power of optimization from samples. In
Advances in Neural Information Processing Systems, volume 29, pages 4017–4025. Curran
Associates, Inc., 2016 (cited on page 2).
[4]
E. Balkanski, A. Rubinstein, and Y. Singer. The limitations of optimization from samples.
Journal of the ACM, 69(3):1–33, 2022 (cited on page 2).
[5]
E. Bampis, D. Christou, B. Escoffier, and K. T. NGuyen. Online learning for min-max discrete
problems. Theoretical Computer Science, 930:209–217, 2022 (cited on pages 2, 7).
[6]
N. Cesa-Bianchi and G. Lugosi. Combinatorial bandits. Journal of Computer and System
Sciences, 78(5):1404–1422, 2012 (cited on page 2).
[7]
W. Chen, Y. Wang, and Y. Yuan. Combinatorial multi-armed bandit: general framework and
applications. In Proceedings of the 30th International Conference on Machine Learning,
volume 28, pages 151–159. PMLR, 2013 (cited on page 2).
[8]
A. Cohen, T. Hazan, and T. Koren. Tight bounds for bandit combinatorial optimization. In
Proceedings of the 30th Annual Conference on Learning Theory, volume 65, pages 629–642.
PMLR, 2017 (cited on page 2).
[9]
R. Combes, M. S. Talebi, A. Proutiere, and M. Lelarge. Combinatorial bandits revisited. In
Advances in Neural Information Processing Systems, volume 28, pages 2116–2124. Curran
Associates, Inc., 2015 (cited on page 2).
[10]
U. Feige. A threshold of ln n for approximating set cover. Journal of the ACM, 45(4):634–652,
1998 (cited on page 1).
[11]
F. Fourati, C. J. Quinn, M.-S. Alouini, and V. Aggarwal. Combinatorial stochastic-greedy
bandit. Proceedings of the 38th AAAI Conference on Artificial Intelligence, 38(11):12052–
12060, 2024 (cited on pages 2, 5, 14).
[12]
Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an
application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997 (cited
on page 7).
[13]
S. Fujishige and Z. Yang. A note on Kelso and Crawford’s gross substitutes condition. Mathe-
matics of Operations Research, 28(3):463–469, 2003 (cited on pages 1, 3).
[14]
D. Golovin, A. Krause, and M. Streeter. Online submodular maximization under a matroid
constraint with application to learning assignments. arXiv:1407.1082, 2014 (cited on pages 2,
5, 14).
[15]
Y. Han, Y. Wang, and X. Chen. Adversarial combinatorial bandits with general non-linear
reward functions. In Proceedings of the 38th International Conference on Machine Learning,
volume 139, pages 4030–4039. PMLR, 2021 (cited on page 2).
[16]
N. Harvey, C. Liaw, and T. Soma. Improved algorithms for online submodular maximiza-
tion via first-order regret bounds. In Advances in Neural Information Processing Systems,
volume 33, pages 123–133. Curran Associates, Inc., 2020 (cited on pages 2, 10).
[17]
S. Iwata, S. Moriguchi, and K. Murota. A capacity scaling algorithm for M-convex submodular
flow. Mathematical Programming, 103(1):181–202, 2005 (cited on page 9).
[18]
S. Iwata and M. Shigeno. Conjugate scaling algorithm for Fenchel-type duality in discrete
convex optimization. SIAM Journal on Optimization, 13(1):204–211, 2002 (cited on page 9).
[19]
A. S. Kelso and V. P. Crawford. Job matching, coalition formation, and gross substitutes.
Econometrica, 50(6):1483–1504, 1982 (cited on pages 1, 3).
[20]
R. Kupfer, S. Qian, E. Balkanski, and Y. Singer. The adaptive complexity of maximizing a
gross substitutes valuation. In Advances in Neural Information Processing Systems, volume 33,
pages 19817–19827. Curran Associates, Inc., 2020 (cited on page 1).
[21]
B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari. Combinatorial cascading bandits. In
Advances in Neural Information Processing Systems, volume 28, pages 1450–1458. Curran
Associates, Inc., 2015 (cited on page 2).

11


---Page Break---
[22]
B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari. Tight regret bounds for stochastic com-
binatorial semi-bandits. In Proceedings of the 18th International Conference on Artificial
Intelligence and Statistics, volume 38, pages 535–543. PMLR, 2015 (cited on page 10).
[23]
T. Lattimore. Bandit convex optimisation. arXiv:2402.06535, 2024 (cited on page 4).
[24]
T. Lattimore and C. Szepesvári. Bandit Algorithms. Cambridge University Press, 2020 (cited
on pages 2, 6, 14, 15).
[25]
B. Lehmann, D. Lehmann, and N. Nisan. Combinatorial auctions with decreasing marginal
utilities. Games and Economic Behavior, 55(2):270–296, 2006 (cited on pages 1, 3).
[26]
N. Littlestone and M. K. Warmuth. The weighted majority algorithm. Information and Compu-
tation, 108(2):212–261, 1994 (cited on page 7).
[27]
L. H. Loomis. On a theorem of von Neumann. Proceedings of the National Academy of
Sciences of the United States of America, 32(8):213–215, 1946 (cited on page 15).
[28]
T. Lykouris, V. Mirrokni, and R. Paes Leme. Stochastic bandits robust to adversarial corrup-
tions. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing,
pages 114–122. ACM, 2018 (cited on page 9).
[29]
N. Merlis and S. Mannor. Tight lower bounds for combinatorial multi-armed bandits. In
Proceedings of the 33rd Annual Conference on Learning Theory, volume 125, pages 2830–
2857. PMLR, 2020 (cited on page 2).
[30]
S. Moriguchi, A. Shioura, and N. Tsuchimura. M-convex function minimization by continu-
ous relaxation approach: proximity theorem and algorithm. SIAM Journal on Optimization,
21(3):633–668, 2011 (cited on pages 1, 3).
[31]
R. Motwani and P. Raghavan. Randomized Algorithms. Cambridge University Press, 1995
(cited on page 15).
[32]
K. Murota. Discrete Convex Analysis, volume 10. SIAM, 2003 (cited on pages 1–3, 8, 9).
[33]
K. Murota and A. Shioura. M-convex function on generalized polymatroid. Mathematics of
Operations Research, 24(1):95–105, 1999 (cited on pages 1, 3–5, 14).
[34]
G. L. Nemhauser and L. A. Wolsey. Best algorithms for approximating the maximum of a
submodular set function. Mathematics of Operations Research, 3(3):177–188, 1978 (cited on
page 1).
[35]
G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher. An analysis of approximations for maxi-
mizing submodular set functions—I. Mathematical Programming, 14(1):265–294, 1978 (cited
on page 1).
[36]
R. Niazadeh, N. Golrezaei, J. R. Wang, F. Susan, and A. Badanidiyuru. Online learning via
offline greedy algorithms: applications in market design and optimization. In Proceedings of
the 22nd ACM Conference on Economics and Computation, pages 737–738. ACM, 2021 (cited
on pages 2, 5, 10, 14).
[37]
G. Nie, M. Agarwal, A. K. Umrawal, V. Aggarwal, and C. John Quinn. An explore-then-
commit algorithm for submodular maximization under full-bandit feedback. In Proceedings of
the 38th Conference on Uncertainty in Artificial Intelligence, volume 180, pages 1541–1551.
PMLR, 2022 (cited on pages 2, 5, 14).
[38]
G. Nie, Y. Y. Nadew, Y. Zhu, V. Aggarwal, and C. J. Quinn. A framework for adapting
offline algorithms to solve combinatorial multi-armed bandit problems with bandit feedback.
In Proceedings of the 40th International Conference on Machine Learning, volume 202,
pages 26166–26198. PMLR, 2023 (cited on pages 2, 5, 14).
[39]
T. Oki and S. Sakaue. Faster discrete convex function minimization with predictions: the M-
convex case. In Advances in Neural Information Processing Systems, volume 36, pages 68576–
68588. Curran Associates, Inc., 2023 (cited on page 1).
[40]
S. Pasteris, A. Rumi, F. Vitale, and N. Cesa-Bianchi. Sum-max submodular bandits. In Proceed-
ings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238,
pages 2323–2331. PMLR, 2024 (cited on page 2).
[41]
I. Rejwan and Y. Mansour. Top-k combinatorial bandits with full-bandit feedback. In Pro-
ceedings of the 31st International Conference on Algorithmic Learning Theory, volume 117,
pages 752–776. PMLR, 2020 (cited on pages 2, 10).
[42]
T. Roughgarden. Beyond the Worst-Case Analysis of Algorithms. Cambridge University Press,
2021 (cited on page 9).

12


---Page Break---
[43]
T. Roughgarden and J. R. Wang. An optimal learning algorithm for online unconstrained
submodular maximization. In Proceedings of the 31st Annual Conference on Learning Theory,
volume 75, pages 1307–1325. PMLR, 2018 (cited on pages 2, 10).
[44]
A. Schrijver. Combinatorial Optimization, volume 24. Springer, 2003 (cited on page 8).
[45]
L. S. Shapley. Complements and substitutes in the opttmal assignment problem. Naval Re-
search Logistics Quarterly, 9(1):45–48, 1962 (cited on page 3).
[46]
A. Shioura. Fast scaling algorithms for M-convex function minimization with application to
the resource allocation problem. Discrete Applied Mathematics, 134(1):303–316, 2004 (cited
on pages 1, 3).
[47]
A. Shioura. M-convex function minimization under L1-distance constraint and its application
to dock reallocation in bike-sharing system. Mathematics of Operations Research, 47(2):1566–
1611, 2022 (cited on page 2).
[48]
T. Soma. Submodular and Sparse Optimization Methods for Machine Learning and Communi-
cation. PhD thesis, The University of Tokyo, 2016 (cited on page 1).
[49]
M. Streeter and D. Golovin. An online algorithm for maximizing submodular functions. In
Advances in Neural Information Processing Systems, volume 21, pages 1577–1584. Curran
Associates, Inc., 2008 (cited on pages 2, 5, 14).
[50]
A. Tajdini, L. Jain, and K. Jamieson. Minimax optimal submodular optimization with bandit
feedback. arXiv:2310.18465, 2023 (cited on page 10).
[51]
Z. Wan, J. Zhang, W. Chen, X. Sun, and Z. Zhang. Bandit multi-linear DR-submodular
maximization and its applications on adversarial submodular bandits. In Proceedings of the
40th International Conference on Machine Learning, volume 202, pages 35491–35524. PMLR,
2023 (cited on page 2).
[52]
A. C.-C. Yao. Probabilistic computations: toward a unified measure of complexity. In Pro-
ceedings of the 18th Annual Symposium on Foundations of Computer Science, pages 222–227.
IEEE, 1977 (cited on pages 8, 15).
[53]
H. Zhang, Z. Zheng, and J. Lavaei. Stochastic L♮-convex function minimization. In Advances
in Neural Information Processing Systems, volume 34, pages 13004–13018. Curran Associates,
Inc., 2021 (cited on page 2).
[54]
M. Zhang, L. Chen, H. Hassani, and A. Karbasi. Online continuous submodular maximization:
from full-information to bandit feedback. In Advances in Neural Information Processing
Systems, volume 32, pages 9210–9221. Curran Associates, Inc., 2019 (cited on page 2).
[55]
Q. Zhang, Z. Deng, Z. Chen, K. Zhou, H. Hu, and Y. Yang. Online learning for non-monotone
DR-submodular maximization: from full information to bandit feedback. In Proceedings
of the 26th International Conference on Artificial Intelligence and Statistics, volume 206,
pages 3515–3537. PMLR, 2023 (cited on page 2).

13


---Page Break---
A
Differences of Theorem 3.1 from robustness results in the submodular case

The basic idea of analyzing the robustness is inspired by similar approaches used in online submodular
function maximization [49, 14, 36, 37, 38, 11]. However, our Theorem 3.1 for the M♮-concave case
is fundamentally different from those for the submodular case.

At a high level, an evident difference lies in the comparator in the guarantees. Specifically, we need to
bound the suboptimality compared to the optimal value in the M♮-concave case, while the comparator
is an approximate value in the submodular case.

At a more technical level, we need to work on the solution space in the M♮-concave case, while the
proof for the submodular case follows from analyzing objective values directly. Let us overview the
standard technique for the case of monotone submodular function maximization under the cardinality
constraint, which is the most relevant to our case due to the similarity in the algorithmic procedures.
In this case, a key argument is that in each iteration, the marginal increase in the objective value
is lower bounded by a 1/K fraction of that gained by adding an optimal solution, minus the local
error. That is, regarding f : {0, 1}V →R as a submodular set function, the submodularity implies
f(xk) −f(xk−1) ≥
1
K (f(xk−1 ∨x∗) −f(xk−1)) −err(ik | xk−1), where ∨is the element-wise
maximum. Consequently, by rearranging terms in the same way as the proof of the (1 −1/e)-
approximation, one can confirm that local errors accumulate only additively over K iterations. In
this way, the robustness property directly follows from incorporating the effect of local errors into
the inequality for deriving the (1 −1/e)-approximation in the submodular case. By contrast, in our
proof of Theorem 3.1 for the M♮-concave case, we need to look at the solution space to ensure that
the local update by ik with small err(ik | xk−1) does not deviate much from Yk−1, as highlighted
in (3) (and this also differs from the original proof without errors [33]). After establishing this, we
can obtain the theorem by induction by virtue of the non-trivial design of Yk (k = 0, . . . , K), which
satisfies x∗∈Y0 and YK = {xK}.

B
MOSS for pure exploration in stochastic multi-armed bandit

We overview the MOSS-based pure-exploration algorithm used in Section 4. For more details, see
Lattimore and Szepesvári [24, Chapters 9 and 33].

Let I{A} take 1 if A is true and 0 otherwise, and let log+(x) = log max{1, x}. Given a stochastic
multi-armed bandit instance with M arms and T ′ rounds, we consider an algorithm that randomly
selects arms A1, . . . , AT ′ ∈{1, . . . , M}. For t = 1, . . . , T ′, let Y t be a random variable representing
the learner’s reward in the tth round, ˆτi(t) = Pt
s=1 I{As = i} the number of times the ith arm is
selected up to round t, and ˆµi(t) =
1
ˆτi(t)
Pt
s=1 I{As = i}Y s the empirical mean reward of the ith
arm up to round t. Given these definitions, the MOSS algorithm can be described as in Algorithm 2.

Algorithm 2 MOSS

Input: Bandit instance with M arms and T ′ rounds

1: Choose each arm i ∈{1, . . . , M} during the first M rounds
2: for t = M + 1, . . . , T ′ :

3:
Choose At = arg maxi∈{1,...,M} ˆµi(t −1) +
r

4
ˆτi(t−1) log+
T ′
N ˆτi(t−1)


Let a1, . . . , aT ′ denote the realization of A1, . . . , AT ′, respectively, after running the MOSS algo-
rithm. Then, we set the final output to i ∈{1, . . . , M} with probability
1
T ′
PT ′

t=1 I{at = i}. This
procedure gives an O(
p

M/T ′)-simple regret algorithm, as stated in Proposition 4.1.

Proposition 4.1 (Lattimore and Szepesvári [24, Corollary 33.3]). Consider a stochastic multi-armed
bandit instance with M arms and T ′ rounds, where T ′ ≥M. Assume that the reward of the ith arm
in the tth round, denoted by Y t
i , satisfies the following conditions: µi := E[Y t
i ] ∈[0, 1] and Y t
i −µi
is 1-sub-Gaussian. Then, there is an algorithm that, after pulling arms T ′ times, randomly returns
i ∈{1, . . . , M} with µ∗−E[µi] = O(
p

M/T ′), where µ∗:= max{µ1, . . . , µM}.

14


---Page Break---
Proof. Since the suboptimality of the ith arm, defined by µ∗−µi, is at most 1 for all i ∈{1, . . . , M},
the MOSS algorithm enjoys a cumulative regret bound of RegT ′ := T ′ · µ∗−E
hPT ′

t=1 µAt
i
≤

39
√

MT ′ + M (see Lattimore and Szepesvári [24, Theorem 9.1]). Consider setting the final output
to i ∈{1, . . . , M} with probability
1
T ′
PT ′

t=1 I{at = i}, where at denote the realization of At. Then,
it holds that µ∗−Ei∼p[µi] = RegT ′/T ′ (see Lattimore and Szepesvári [24, Proposition 33.2]). The
right-hand side is at most (39
√

MT ′ + M)/T ′ ≤40
p

M/T ′, completing the proof.

C
Proof of Proposition 5.3

Proposition 5.3 (Yao [52]). Let A be a finite set of all possible deterministic learning algorithms
that run in polynomial time per round and F1:T a finite set of sequences of M♮-concave functions,
f 1, . . . , f T . Let RegT (a, f 1:T ) be the cumulative regret a deterministic learner a ∈A achieves on a
sequence f 1:T = (f 1, . . . , f T ) ∈F1:T . Then, for any polynomial-time randomized learner A and
any distribution q on F1:T , it holds that

max

E

RegT (A, f 1:T )

: f 1:T ∈F1:T 	
≥min

Ef 1:T ∼q

RegT (a, f 1:T )

: a ∈A
	
.

Proof. We use the same proof idea as that of Yao’s principle (see, e.g., Motwani and Raghavan [31,
Section 2.2]). First, note that any polynomial-time randomized learner can be viewed as a polynomial-
time deterministic learner with access to a random tape. Thus, we can take A to be chosen according
to some distribution p on the family, A, of all possible polynomial-time deterministic learners.

Consider an |A| × |F1:T | matrix M, whose entry corresponding to row a ∈A and column f 1:T ∈
F1:T is RegT (a, f 1:T ). For any polynomial-time randomized learner A and any distribution q on
F1:T , it holds that

max

E

RegT (A, f 1:T )

: f 1:T ∈F1:T 	
≥min
p′ max
ef1:T p′Mef 1:T

= max
q′
min
ea
eaMq′

≥min

Ef 1:T ∼q

RegT (a, f 1:T )

: a ∈A
	
,

where p′ and q′ denote probability vectors on A and F1:T , respectively, and ef 1:T and ea denote the
standard unit vectors of f 1:T and a, respectively. The equality is due to Loomis’ theorem [27].

15


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We have clarified our scope (no-regret M♮-concave function maximization)
and contributions (stochastic bandit algorithms and NP-hardness of the adversarial setting).
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
Justification: Section 6 discusses limitations and possible directions for overcoming them.
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

16


---Page Break---
Justification: Assumptions are clearly stated in Section 2.2 and problem-setting paragraphs
in Sections 4 and 5. All theoretical results are followed by complete proofs.
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
Justification: This paper does not include experiments.
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

17


---Page Break---
Answer: [NA]
Justification: This paper does not include experiments requiring code.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run
to reproduce the results. See the NeurIPS code and data submission guidelines (https:
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
Justification: This paper does not include experiments.
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
Justification: This paper does not include experiments.
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

18


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

Justification: This paper does not include experiments.

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

Justification: This paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [No]

Justification: This paper is dedicated to theoretical results. While there are potential societal
consequences of our work, we feel none of them must be specifically discussed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

19


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

Justification: This paper poses no such risks.

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
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a
dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.

20


---Page Break---
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

21


---Page Break---
