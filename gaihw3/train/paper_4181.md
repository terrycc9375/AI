Honor Among Bandits:
No-Regret Learning for Online Fair Division

Ariel D. Procaccia∗
Benjamin Schiffer†
Shirley Zhang‡

Abstract

We consider the problem of online fair division of indivisible goods to players
when there are a finite number of types of goods and player values are drawn from
distributions with unknown means. Our goal is to maximize social welfare subject
to allocating the goods fairly in expectation. When a player’s value for an item is
unknown at the time of allocation, we show that this problem reduces to a variant
of (stochastic) multi-armed bandits, where there exists an arm for each player’s
value for each type of good. At each time step, we choose a distribution over arms
which determines how the next item is allocated. We consider two sets of fairness
constraints for this problem: envy-freeness in expectation and proportionality in
expectation. Our main result is the design of an explore-then-commit algorithm
that achieves ˜O(T 2/3) regret while maintaining either fairness constraint. This
result relies on unique properties fundamental to fair-division constraints that allow
faster rates of learning, despite the restricted action space. We also prove a lower
bound of ˜Ω(T 2/3) regret for our setting, showing that our results are tight.

1
Introduction

Fair allocation of indivisible goods is a fundamental problem with a wide range of applications;
implemented algorithms for this task have been widely used in practice [14]. We consider the online
fair division setting, which introduces additional complexities as items arrive one by one and each
item must be immediately and irrevocably allocated at its time of arrival. Crucially, this allocation
must be done without knowledge of future items [5]. One motivating example for this setting is
a food bank that receives donations for a region and then allocates these donations among many
different food pantries and soup kitchens in that region. Donations are often perishable, and therefore
must be immediately allocated. Furthermore, donations can be unpredictable, and hence knowledge
of future items is limited.

Two standard notions of fairness are envy-freeness and proportionality. Envy-freeness implies that
every player is at least as happy with their own allocation as with any other player’s allocation.
Intuitively, envy-freeness guarantees that no player will want to trade their allocation for that of
another player. Proportionality is a slightly weaker notion, which requires only that each of the n
players receive at least a 1/n fraction of their total value for all items. Finding a solution which is
envy-free or proportional is often interesting in and of itself, as can be seen from many previous
results in fair division [7, 30, 12, 3]. In cases where there may exist multiple envy-free or proportional
allocations, however, a natural goal is then to find the best solution among such allocations [13]. In
our work, we evaluate the quality of a fair solution by its (utilitarian) social welfare, which is defined
as the sum over all players of each player’s value for their own allocation.

We take a probabilistic approach to analyzing online fair division. In particular, we assume that
there are a finite number of item types, and each player’s value for each type of item is drawn from

∗Paulson School of Engineering and Applied Sciences,
Harvard University | E-mail:
ariel-
pro@seas.harvard.edu.
†Department of Statistics, Harvard University | E-mail: bschiffer1@g.harvard.edu.
‡Paulson School of Engineering and Applied Sciences, Harvard University | E-mail: szhang2@g.harvard.edu.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
a random distribution. In practice, these distributions would not be known in advance and must
be learned as items are allocated. For example, consider again the food bank. When a new food
pantry opens, the values of that food pantry for different types of products are unknown. After items
have been allocated to the food pantry, however, the food bank can easily collect information on the
demand for various item types at the food pantry. Therefore, we primarily consider the setting where
the player distributions are unknown in advance, and a player’s true value for an item is observed if
and only if that player receives the item. This problem can be viewed as a variant of the multi-armed
bandits problem, as the goal is to learn unknown distributions (player values) while maintaining high
reward (social welfare), subject to fairness constraints; with a finite number of types of items, pulling
an arm represents allocating a specific item type to a specific player.

As is standard in the multi-armed bandits literature, we use the notion of regret to measure the
difference between our algorithm’s performance and that of the optimal policy that knows the value
distributions and is subject to the same fairness constraints. Our overarching challenge is this:
design online allocation algorithms that achieve low regret while maintaining fairness in the form of
envy-freeness or proportionality.

1.1
Our Results
Our main result is that there exists a simple optimization-based explore-then-commit algorithm that
achieves ˜O(T 2/3) regret and maintains envy-freeness in expectation (Algorithm 1 and Theorem
1). A variant of the same algorithm achieves ˜O(T 2/3) regret while maintaining proportionality in
expectation. The key step of the algorithm is a linear program-based optimization that guarantees
that the constraints are satisfied without significantly decreasing social welfare.

The main difficulty in this learning problem is that the envy-freeness and proportionality constraints
depend on the unknown value distributions and may be tight constraints without any slack. We
therefore develop novel machinery that relies on fundamental properties of these fairness notions.
One observation is that our fairness constraints are always satisfied when players are treated equally.
Another crucial property is that when players have unequal values, these fairness notions can be
satisfied with slack (Property 2). The latter property is especially challenging to show for envy-
freeness, and the combinatorial algorithm that achieves it (Lemma 1) should be of independent
interest to researchers in fair division.

1.2
Related Work

Online Fair Division. Work in online fair division generally deals with dividing goods when there is
uncertainty about the future. Early work in the area focused on axiomatic questions [31, 1].

Our paper is most closely related to work by Benadè et al. [5]. Like us, they consider a setting
where indivisible items arrive online and must be allocated immediately and irrevocably to players.
They study several models for how the values of items are determined, ranging from a model where
values are drawn i.i.d. from a distribution common to all players and items to, at the other extreme,
an adversarial model with worst-case values. There are two fundamental differences between their
work and ours. First, Benadè et al. [5] do not optimize social welfare; rather, they seek to either just
minimize envy, or do so while (approximately) satisfying the axiomatic notion of Pareto efficiency.
Second, and more crucially, they assume that the values of all players for an item are known at the
time of its arrival, whereas in our model the values are unknown. It is precisely this modeling choice
that induces a learning problem and underlies the connection between our setting and multi-armed
bandits, which is absent in prior work in online fair division.

Like us, Yamada et al. [35] also study online fair division through the lens of bandit learning. Their
setting is similar to ours in that they consider a finite number of item types where player values
for item types are initially unknown. However, [35] do not guarantee fairness through constraints –
instead, they incorporate fairness through their objective function of Nash social welfare. [35] also
make an additional assumption that player values are normalized, which we do not require for our
results.

Fairness in Multi-armed Bandits. The other main body of literature related to our paper is multi-
armed bandits with constraints. One notion of fairness in multi-armed bandits is the idea that similar
individuals and/or groups should be treated similarly [9, 18, 23]. The fairness constraint of Joseph
et al. [17] is that a worse arm is not pulled with higher probability than a better arm. Their definition
of fairness is actually incompatible with maintaining envy-freeness (or proportionality), because

2


---Page Break---
maintaining envy-freeness may require allocating an item to a player with lower value to prevent envy.
Another common fairness constraint in multi-armed bandits is that every arm receives a minimum
fraction of pulls [10, 11, 20, 28]. This notion of fairness is also not compatible with envy-freeness
because the optimal envy-free allocation may never give a player a specific item type. There also exist
many other fairness notions in contextual bandits that are farther from our setting [15, 29, 32, 34].
Wei et al. [33] analyze a form of envy-freeness in contextual bandits, but their envy-freeness notion
depends only on the treatment probabilities instead of the values.

Our paper is also closely related to work on multi-armed bandits subject to general linear constraints.
Multiple works in linear bandits study “safety” with respect to a linear constraint that depends on the
unknown true mean values [2, 8, 26]. Amani et al. [2] focus on a single constraint and specifically
show that if there is positive slack in the optimal solution, then ˜O(T 1/2) regret is possible. If there
is zero slack, however, their algorithm only achieves ˜O(T 2/3) regret. This differs from our work
because envy-freeness and proportionality involve multiple constraints that can have zero slack. Note
that our setting is similar but not equivalent to linear bandits, as a single arm is pulled in each step in
our setting. There also exist many results for cumulative constraints in bandits [21, 22]. These are
less closely related to our model as we consider constraints that must hold at every time step. Finally,
there is a branch of multi-armed bandits that studies constraints in expectation at each step as in our
paper. However, these works are also in the linear bandits setting and again require a safety gap that
fairness constraints such as envy-freeness may not guarantee [27].

Practical Motivation. Mertzanidis et al. [25] apply online fair division algorithms through a
partnership with a program in Indiana that redistributes rejected truckloads of food. The program,
known as Food Drop, allocates 10,000+ pounds of rejected food per month to food banks. In this
application, the available food arrives in an online and unpredictable way, and the trucks must be
allocated immediately. More generally, the specific food donations depend on what items grocery
stores or restaurants have remaining at the end of the day. Therefore, donations are unpredictable,
which we model through randomness.

In practice, utilities for food donations such as in the Food Drop program may not be additive.
However, if the deliveries are sufficiently infrequent, then additive player utilities are likely to be
a good approximation. For example, in the food allocation data of [19], there were a total of 1760
donations from 169 donors over the course of five months and 277 organizations that received
donations. Therefore, the organizations receive donations every 3-4 weeks on average, suggesting
that donations can be largely regarded as independent.

2
Model

In this section we introduce our basic setting and terminology.

2.1
Online Allocation With Unknown Values

Suppose we have a set of players N = [n] and a set of object types M = [m]. Given a set of T
indivisible items, an allocation A = (A1, ..., An) is a partition of the T items among the n players,
where player i receives the items in Ai. In our model, we assume that every item j has a type
k(j) ∈[m], and that there exists a (possibly unknown) matrix µ∗such that each player i’s value for
an item of type k ∈[m] is independently drawn from a sub-Gaussian distribution with mean µ∗
ik.
Player values are assumed to be independent across both players and items. We will often refer to
µ∗
i as the vector of mean values for player i. For a specific item j, we denote player i’s value for
item j as Vi(j), and similarly, player i’s value for their allocation Ai as Vi(Ai) = P

j∈Ai Vi(j). The
(utilitarian) social welfare of an allocation A is sw(A) = Pn
i=1 Vi(Ai).

We consider algorithms in the following online setting. At each time step t ∈[T], an item jt of type
kt arrives, where kt ∼D, for some known distribution D supported on [m]. We will assume that
D = Unif([m]), or in other words that every item has an equal probability of being type 1, ..., m.
We make this choice purely for ease of exposition, in order to simplify notation; our results and
techniques extend seamlessly to arbitrary distributions D that do not depend on T, as we explain
in Appendix C.1. The algorithm observes the item type kt, and must then immediately allocate the
item jt to a player it, at which time the algorithm observes that player’s value Vit(jt). Note that the
algorithm does not observe any other player’s value for item jt. The high-level goal is to allocate
these items in a manner that maximizes the social welfare of the final allocation of all T items.

3


---Page Break---
We denote X ∈Rn×m as a valid fractional allocation if P
i Xik = 1 for every k ∈[m]. One valid
fractional allocation we will often refer to is the uniform at random allocation (UAR), where every
entry is 1

n. At every time-step t, before observing kt, the allocation algorithm ALG takes as input
the history Ht = {(kt′, it′, Vit′(jt′)) : t′ < t} and returns a fractional allocation Xt = ALG(Ht),
where (Xt)ik represents the probability of allocating the item to player i if the item is of type k.
If the next item is of type kt, then the algorithm allocates the item randomly among the n players
according to the distribution induced by the kth
t column of Xt, i.e. (X⊤
t )kt. Therefore, the tth item
is allocated to player i with probability (Xt)ikt. We denote the final realized allocation that ALG
returns as A(ALG), and the corresponding partial allocation up to time τ as Aτ(ALG). This online
process is summarized in pseudo-code in Appendix A.

We will also assume (explicitly in our theorem statements) that for all i, k, there exist known constants
a, b > 0 such that a ≤µ∗
ik ≤b. This assumption is necessary because if we allow the means of
values to be arbitrary close to zero, then it can be impossible to achieve regret of o(T). This is
formalized in Theorem 11 in Appendix C.2.

2.2
Fairness Notions
We will primarily use two metrics of fairness to evaluate an online allocation algorithm ALG: envy-
freeness in expectation and proportionality in expectation. Both are defined below. For two vectors
x, y ∈Rn, we use ⟨x, y⟩= x · y to represent the dot product of the two vectors.

Definition 1. Let Xt = ALG(Ht) be the fractional allocation used by algorithm ALG at time t
given history Ht. Then ALG satisfies envy-freeness in expectation (EFE) if for all t and all Ht,
(Xt)i · µ∗
i ≥maxi′∈[n] (Xt)i′ · µ∗
i for all i.

Definition 2. Let Xt = ALG(Ht) be the fractional allocation used by algorithm ALG at time t
given history Ht. Then ALG satisfies proportionality in expectation (PE) if for all t and all histories
Ht, (Xt)i · µ∗
i ≥1

n
P
i′∈[n] (Xt)i′ · µ∗
i for all i.

Intuitively, envy-freeness in expectation is equivalent to maintaining that at every time step t and
before observing the item type kt, no player prefers the fractional allocation of any other player in
Xt. Similarly, proportionality in expectation is equivalent to maintaining that at every time step t and
before observing the item type kt, the expected value of every player for their fractional allocation is
at least 1/n times that player’s value if they received the item with probability 1.

In Appendix B, we justify some of the implicit choices behind these definitions. Specifically, we
discuss why we consider envy-freeness in expectation rather than its realization, and also why we
require envy-freeness in expectation to hold at every individual time step. Analogous results for
proportionality can be found in Appendix B.1. For the former question, Theorem 3 shows that in
our setting, no algorithm can with high probability output an allocation A(ALG) with realized envy
less than
√

T. Note that Benadè et al. [5] show that in the adversarial setting, no algorithm can
guarantee o(
√

T) realized envy. Conversely, they also show that when values are generated randomly
and observed before allocation, there exists an algorithm that can guarantee o(1) realized envy with
high probability. Theorem 3 shows that when values are still generated randomly but are unknown at
the time of allocation (as in our setting), no algorithm can guarantee o(
√

T) realized envy with high
probability. We complement Theorem 3 with Theorem 4, which shows that any algorithm ALG that
satisfies envy-freeness in expectation will output a final allocation A(ALG) with realized envy of at
most
√

T log(T) with high probability. Therefore, envy-free in expectation algorithms are within a
log(T) factor of being “optimal” in terms of final realized envy.

We also show that requiring envy-freeness in expectation at every time step does not lead to any social
welfare loss compared to requiring envy-freeness in expectation only at the end of T rounds. More
specifically, Theorem 5 (again in Appendix B) implies that requiring that no player is envious in
expectation of any other player at the end of all T rounds is equivalent to maintaining envy-freeness
in expectation at all times t ∈[T] when maximizing social welfare. A key step of our proof of
Theorem 5 is showing that for every time- or history-dependent algorithm ALG which achieves
envy-freeness in expectation at the end of T rounds, there exists another algorithm ALG′ that is
time- and history-independent, envy-free in expectation at every time step, and achieves the same
social welfare. Therefore, maximizing social welfare only over algorithms which are envy-free in
expectation at every time step is sufficient even if envy-freeness in expectation at the end of T rounds
is all that is desired.

4


---Page Break---
We can formulate our fairness notions as linear constraints, in the spirit of prior work in fair di-
vision [4]. Formally, define ⟨A, B⟩F as the Frobenius inner product of matrices A and B. For
B ∈Rn×m, c ∈R, and a fractional allocation X, we represent the linear constraint ⟨B, X⟩F ≥c as
the tuple (B, c). A fractional allocation X satisfies a set of L linear constraints {(Bℓ, cℓ)}L
ℓ=1 if for
all ℓ∈[L], ⟨Bℓ, X⟩F ≥cℓ. Because the constraints represent “fairness in expectation” relative to
the mean values, we will explicitly let the constraint matrix Bℓ(µ∗) be a function of the mean value
matrix µ∗. Therefore, we will consider sets of constraints of the form {(Bℓ(µ∗), cℓ)}L
ℓ=1. Because
these constraints are functions of µ, we will also refer to families of constraints

{(Bℓ(µ), cℓ)}L
ℓ=1
	

µ.

The following two remarks show how envy-freeness in expectation and proportionality in expectation
can be represented within this framework.

Remark 1. For every ℓ∈[n2], construct Befe
ℓ(µ∗) as follows. Define i = ⌈ℓ

n⌉and i′ = (ℓ
mod n) + 1. For every k ∈[K], let (Befe
ℓ(µ∗))ik = µ∗
ik and (Befe
ℓ(µ∗))i′k = −µ∗
ik. For all
i′′ ̸∈{i, i′}, let (Befe
ℓ(µ∗))i′′ = 0. Then the envy-freeness in expectation constraints for mean µ∗as
defined in Definition 1 correspond to efe(µ∗) := {(Befe
ℓ(µ∗), 0)}n2
ℓ=1.

Remark 2. For every ℓ∈[n], construct Bpe
ℓ(µ∗) as follows. For every k ∈[m], (Bpe
ℓ(µ∗))ℓk =
(n−1)

n
· µ∗
ℓk and (Bpe
ℓ(µ∗))ik = −1

n · µ∗
ℓk for every i ̸= ℓ. Then the proportionality in expectation
constraints for mean µ∗as defined in Definition 2 correspond to pe(µ∗) = {(Bpe
ℓ(µ∗), 0)}n
ℓ=1.

2.3
Regret and Problem Formulation
An algorithm ALG satisfies constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1 if for all t ∈[T], the fractional allocation
Xt used by ALG at time t satisfies the constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1. When µ∗is known, the
expected social welfare can be directly optimized over all algorithms ALG that satisfy constraints
{(Bℓ(µ∗), cℓ)}L
ℓ=1. This problem is equivalent to solving LP (1) with µ = µ∗and using the solution
Y µ∗as the fractional allocation for all time steps.
Y µ := arg max ⟨X, µ⟩F
s.t. ⟨Bℓ(µ), X⟩F ≥cℓ
∀ℓ
X

i
Xik = 1
∀k
(1)

When µ∗is unknown, we define the regret of an algorithm ALG as follows. Note that the baseline
algorithm in this definition of regret is the optimal allocation algorithm under the constraints when
µ∗is known.

Definition 3. Let Y µ∗be the solution to LP (1) for µ = µ∗. Let Xt = ALG(Ht) be the fractional
allocation used by algorithm ALG at time t given history Ht. Then the T-step regret of ALG for
constraints {(Bℓ(µ∗), cℓ}L
ℓ=1 is T · ⟨Y µ∗, µ∗⟩F −PT −1
t=0 ⟨Xt, µ∗⟩F .

We are now ready to present the formal problem statement. Because the constraints depend on the
unknown values that are being learned, we only require constraint satisfaction with high probability.

Problem 1. Suppose we are given n, m, T, a, b such that 0 < a ≤µ∗
ik ≤b for all i ∈[n], k ∈[m].
Given a family of fairness constraints

{(Bℓ(µ), cℓ)}L
ℓ=1
	

µ representing either envy-freeness in
expectation or proportionality in expectation, the goal is to construct an algorithm ALG such that
with probability 1 −1/T, Xt = ALG(Ht) satisfies the constraints (Bℓ(µ∗), cℓ)}L
ℓ=1 for all t ∈[T]
and the regret of ALG for constraints (Bℓ(µ∗), cℓ)}L
ℓ=1 is o(T).

Note that the o(T) regret in Problem 1 will be ˜O(T 2/3) for our results. We use the standard O() and
˜O() notation with respect to the number of time steps T, and therefore the constants represented by
this notation may depend on other problem parameters such as n and m.

3
Fairness Machinery

Our goal in this section is to establish novel, fundamental properties of envy-freeness and propor-
tionality in expectation, which will serve as a crucial part of the machinery underlying our regret
bounds.

In the context of fairness, a natural assumption is that if a group of individuals are treated equally,
then that group is considered to be treated fairly. In that spirit, our first key property is as follows.

5


---Page Break---
Property 1. For any ℓ∈[L], suppose that a fractional allocation X ∈Rn×m satisfies Xi1 =
Xi2, ∀i1, i2 ∈{i : Bℓ(µ)i ̸= 0}. Then ⟨Bℓ(µ), X⟩F ≥cℓ.

Informally, a set of constraints satisfies Property 1 if for any constraint in the set, the constraint
is always satisfied when all players involved in the constraint have the same fractional allocation.
An important consequence of Property 1 is that the uniform at random allocation satisfies every
constraint. This implies that even with no information about the players’ mean values, the uniform at
random allocation will always be fair.

Note that envy-freeness in expectation satisfies Property 1 because any two players with equal
allocations are never envious of each other. Proportionality in expectation also satisfies Property
1, because if every player has the same allocation, then every player is receiving exactly their
proportional allocation.

Observation 1. The envy-freeness in expectation and proportionality in expectation constraints
satisfy Property 1.

Our second key property is more technical and novel. Intuitively, the property requires the existence
of a fractional allocation X′ that is only slightly worse than the optimal constrained allocation Y µ,
but unlike the latter allocation, in X′ the constraints either hold with slack or all players involved in
the constraint are treated equally. Interestingly, this property does not hold for arbitrary sets of linear
constraints, but relies on structure inherent to the envy-freeness in expectation and proportionality in
expectation constraints. The bulk of the theoretical work of this paper is proving that the envy-freeness
in expectation and proportionality in expectation constraints satisfy this property.

Property 2. Let Y µ be the solution to LP (1). Then there exists constants (relative to T) γ0 and
CP2 > 0 such that for any γ < γ0 and any µ, there exists a fractional allocation X′ such that
⟨X′, µ⟩F ≥⟨Y µ, µ⟩F −CP2γ, and such that for each ℓ∈[L], either

1. ⟨Bℓ(µ), X′⟩F ≥cℓ+ γ or

2. ∀i1, i2 ∈{i : Bℓ(µ)i ̸= 0}, X′
i1 = X′
i2.

Lemma 1. The family of envy-freeness in expectation constraints satisfies Property 2.

Proof sketch. We will informally argue that we can transform Y µ into a fractional allocation X′
which satisfies Property 2 through Algorithm 3. Algorithm 3 iterates over ‘envy-with-slack-α’
graphs, which track whether a player prefers their allocation by at least α over another player’s
allocation. More specifically, given parameters µ, X, and α, the corresponding ‘envy-with-slack’
graph has vertices N and edge set E such that a directed edge from i to i′ exists if and only if
Xi · µi −Xi′ · µi < α. The weight of each such edge is Xi · µi −Xi′ · µi. At a high level, Algorithm
3 constructs ‘envy-with-slack-α’ graphs with progressively smaller α, with α ≥γ for all iterations.
The algorithm operates on sets of nodes called equivalence classes, where every pair of nodes in an
equivalence class has the same allocation. Algorithm 3 makes progress in every iteration by either 1)
merging two equivalence classes, or 2) removing an edge from the graph.

An overview of the algorithm is as follows. In each iteration, Algorithm 3 generally performs one
of three operations and decreases α. First, if there exists an equivalence class S with at least one
incoming edge but no outgoing edges, then operation remove-incoming-edge transfers allocation
probability from nodes in S to all other nodes. This will remove all incoming edges to S. If such
an equivalence class does not exist, then Algorithm 3 finds a special type of directed cycle in the
‘envy-with-slack’ graph. The directed cycle is chosen so that the outgoing edge of each node i in
the cycle is among i’s outgoing edges with minimal weight. Therefore, each node i in the cycle is
pointing to an i′ ∈N for whom i has the least slack. If there exists some node i∗∈N which has
an edge to some but not all of the nodes in the cycle, then operation cycle-shift gives each node in
the cycle half of its current allocation and half of the next node’s allocation. This will remove an
outgoing edge from i∗. If such a node does not exist, then Algorithm 3 either decreases α to remove
an edge or creates a new equivalence class by merging all equivalence classes that the nodes in the
cycle belong to via operation average-clique.

However, such a merge may lead to envy, which is removed by Algorithm 4. Each call to Algorithm
4 removes envy from at least one edge. Algorithm 4 does so by first carefully redistributing allocation
among the nodes until there exists a cycle where each node has non-negative envy (which is equivalent
to a cycle with non-positive slack). Each node in the cycle is then given the allocation of the next

6


---Page Break---
node in the cycle. We prove that each call to Algorithm 4 decreases the number of edges with envy,
while not increasing the number of edges in the ‘envy-with-slack’ graph. Furthermore, Algorithm 4
does not significantly decrease the social welfare of the allocation.

The three operations and Algorithm 4 each take as input an allocation X and returns a new allocation
X′ which is close in social welfare to X. Furthermore, each iteration begins with an envy-free
allocation, and the size of the edge set of the ‘envy-with-slack’ graphs never increases throughout the
algorithm. The maximum size of an equivalence class is n, so an edge must be removed from the
‘envy-with-slack’ graph every n iterations. There are at most n2 edges, and the algorithm therefore
terminates in at most n3 iterations with an allocation which satisfies Property 2. For the numerous
details, see Appendix F.

Lemma 2. The family of proportionality in expectation constraints satisfies Property 2.

Proof sketch. Define the slack Si of a player i for an allocation as the amount by which that player’s
value for their allocation is greater than their proportional value. In other words, the slack is the
amount of welfare a player can lose and still satisfy the proportionality constraint. We construct the
fractional allocation X′ in one of two different ways depending on the amount of total slack for the
allocation Y µ across all n players.

If the amount of total slack across all n players is less than bnγ

a , then we take X′ = UAR. Note that
the total slack is equivalent to the change in social welfare between Y µ and UAR. Therefore, because
the total slack was less than bnγ

a , the difference in social welfare between Y µ and UAR is at most
bnγ

a which is O(γ). Furthermore, by definition the UAR allocation satisfies option 2 of Property 2
for all constraints ℓ.

If the amount of total slack is greater than bnγ

a , then we construct X′ from Y µ by transferring
allocation away from players with slack greater than the required γ and redistributing this allocation
so that every player has slack of at least γ. Specifically, each player i loses ∆ik of their allocation for
item k, where

∆ik :=
Y µ
ik
Pm
k′=1 Y µ
ik′ ·
Si
Pn
i′=1 Si′ · nγ

a .

The allocation X′ is then constructed as

X′
ik := Y µ
ik −∆ik + 1

n

n
X

i′=1
∆i′k.
(2)

Intuitively, this can be viewed as each player putting a part of their allocation (proportional to Si · γ)
into a communal “pot.” The pot, consisting of Pn
i′=1 ∆i′k for item k, is then divided evenly among
all n players to form X′. By construction, no player loses more than Si social welfare when the pot
is created, and every player receives at least γ additional social welfare when the pot is redistributed.
Therefore, in the resulting allocation X′, every player prefers their allocation to their proportional
value by at least γ, i.e. each player has a slack of at least γ for X′. Furthermore, the total difference
in social welfare between Y µ and X′ is at most O(γ). We have thus shown that in both cases, X′
will satisfy all of the desired conditions. The full proof is relegated to Appendix E.

It will be useful to introduce two further properties that are immediately satisfied by the definitions
of envy-freeness in expectation and proportionality in expectation. Property 3 guarantees a form of
Lipschitz continuity in µ for the constraints. This is unsurprising, as the entries in the matrices Bℓ(µ)
for both envy-freeness in expectation and proportionality in expectation are linear in the entries of µ.
Property 4 guarantees that the non-zero entries in the constraint matrices stay the same for all values
of µ, which follows directly from Remarks 1 and 2 and the fact that µik is bounded away from 0.

Property 3. There exists a K > 0 such that ∀µ1, µ2 ∈[a, b]n×m and ∀ϵ > 0, if ∥µ1 −µ2∥1 ≤ϵ
then ∥Bℓ(µ1) −Bℓ(µ2)∥1 ≤Kϵ.

Observation 2. The envy-freeness in expectation and proportionality in expectation constraints
satisfy Property 3.

Property 4. For all µ, µ′ ∈[a, b]n×m, {i : Bℓ(µ)i ̸= 0} = {i : Bℓ(µ′)i ̸= 0}.

7


---Page Break---
Observation 3. The envy-freeness in expectation and proportionality in expectation constraints
satisfy Property 4.

Recall that Property 2 implies that for every constraint ℓ, either the constraint ℓhas a slack of at least γ
for X′ or every player involved in constraint ℓis treated equally under allocation X′. A slack of γ in
the constraint guarantees constraint satisfaction for all µ′ close to µ if the constraints are continuous
in µ. Treating every player equally for a given constraint also guarantees that the constraints are
satisfied for all µ′ by Property 1. Therefore, Properties 1 and 2 together with continuity (Property
3) imply that there exists a fractional allocation X′ such that the social welfare of X′ is close to the
social welfare of Y µ and such that X′ not only satisfies the constraints for µ, but also satisfies the
constraints for any µ′ close to µ.

4
Algorithm and Regret Bounds

In this section, we present our main result, an explore-then-commit algorithm which achieves ˜O(T 2/3)
regret while maintaining either proportionality in expectation or envy-freeness in expectation. The
key step in Algorithm 1 is the optimization in LP (3) to guarantee that the fairness constraints are
satisfied with high probability. For µ ∈Rn×m
+
and ϵ ∈Rn×m
+
, we define the confidence region
µ ± ϵ = {µ′ ∈Rn×m : µik −ϵik ≤µ′
ik ≤µik + ϵik ∀i, k}. Note that Algorithm 1 requires solving
LPs with an infinite number of constraints, which we discuss further in Section 6.

Algorithm 1 Fair Explore-Then-Commit
Require: n, m, T, {{(Bℓ(µ), cℓ)}L
ℓ=1}µ
for t ←1 to T 2/3 −1 do

At time t, use fractional allocation Xt = UAR.
end for
Nik ←PT 2/3−1
τ=0
1kτ =k,iτ =i
ˆµik ←
1
Nik
PT 2/3−1
τ=0
1kτ =k,iτ =iViτ (jτ)

ϵik =
q

log2(4Tnm)/(2Nik)
ˆX ←Solution to the following LP:

max
X ⟨X, ˆµ⟩F

s.t. ⟨Bℓ(µ), X⟩F ≥cℓ
∀ℓ∈[L], ∀µ ∈ˆµ ± ϵ

n
X

i=1
Xik = 1
∀k
(3)

for t ←T 2/3 to T −1 do

At time t, use fractional allocation Xt = ˆX.
end for

Theorem 1. Suppose we are given n, m, T, a, b such that 0 < a ≤µ∗
ik ≤b for all i ∈[n], k ∈[m].
If {{(Bℓ(µ), cℓ)}L
ℓ=1}µ = {efe(µ)}µ or {{(Bℓ(µ), cℓ)}L
ℓ=1}µ = {pe(µ)}µ, then Algorithm 1 with
probability 1 −1/T satisfies the constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1 and achieves regret of ˜O(T 2/3) for
constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1.

Proof sketch. We have already shown in Section 3 that both envy-freeness in expectation and propor-
tionality in expectation satisfy Properties 1, 2, 3, and 4. Therefore, it suffices to show that Algorithm
1 achieves ˜O(T 2/3) regret for any family of constraints {{(Bℓ(µ), cℓ)}L
ℓ=1}µ that satisfies Properties
1, 2, 3, and 4.

The allocations used during the warm-up steps of Algorithm 1 are uniform at random, and therefore
these allocations satisfy the constraints {(Bℓ(µ), cℓ)}L
ℓ=1 for all µ by Property 1. Because the
fractional allocations used in the first T 2/3 steps are all UAR, each arm, or (player, item) pair, will
be sampled with probability
1
nm at each step. This implies by Hoeffding’s inequality that with high
probability, Nik = ˜Ω(T 2/3) for all i, k. The i, k entry in the ϵ matrix is proportional to
1
√Nik , and

8


---Page Break---
therefore ∥ϵ∥1 = ˜O(T −1/3) with high probability. Because the value distributions are sub-Gaussian,
a standard application of Hoeffding’s inequality also gives that with high probability, the true mean
matrix will be within our confidence region, i.e. µ∗∈ˆµ ± ϵ. To summarize, because we used UAR
for the first T 2/3 steps, we have that

Pr

∥ϵ∥1 ≤˜O(T −1/3), µ∗∈ˆµ ± ϵ

≥1 −1

T .

For the rest of the proof we will assume that the high probability event in the equation above holds.
The next step is to show that ˆX has ∥ϵ∥1 per-step regret compared to Y µ∗. Let K be the Lipschitz
constant of Property 3. Using Property 2 with µ = µ∗and γ = 2K∥ϵ∥1 = ˜O(T −1/3) gives that there
exists an allocation X′ such that the social welfare of X′ is only O(∥ϵ∥1) less than the social welfare
of the optimal allocation Y µ∗and such that every constraint ℓeither has slack of at least 2K∥ϵ∥1
(option 1 of Property 2) or every player is treated equally in constraint ℓ(option 2 of Property 2).
We will now show that X′ is a solution to LP (3). If option 1 holds for constraint ℓ∈[L], then by
Property 3, X′ will satisfy the constraint (Bℓ(µ), cℓ) for every µ ∈ˆµ ± ϵ. Formally, if option 1 holds
for constraint ℓ, then for any µ ∈ˆµ ± ϵ,

⟨Bℓ(µ), X′⟩F = ⟨Bℓ(µ), X′⟩F −⟨Bℓ(µ∗), X′⟩F + ⟨Bℓ(µ∗), X′⟩F
= ⟨Bℓ(µ) −Bℓ(µ∗), X′⟩F + ⟨Bℓ(µ∗), X′⟩F
≥⟨Bℓ(µ∗), X′⟩F −∥Bℓ(µ) −Bℓ(µ∗)∥1
[0 ≤X′
ik ≤1, ∀i, k]

≥⟨Bℓ(µ∗), X′⟩F −2K∥ϵ∥1
[Property 3]
≥cℓ.
[Property 2: option 1]

If option 2 holds for constraint ℓ∈[L], then Properties 1 and 4 together guarantee that X′ will
satisfy the constraint (Bℓ(µ), cℓ) for every µ ∈ˆµ ± ϵ. Therefore, X′ will satisfy all of the constraints
{Bℓ(µ), cℓ}L
ℓ=1 for every µ ∈ˆµ ± ϵ, which implies that X′ is a solution to LP (3).

Finally, because ˆX is the optimal solution to LP (3), ˆX must have higher social welfare than X′

under means ˆµ. Because ∥µ∗−ˆµ∥1 ≤∥ϵ∥1, this implies that ˆX must have at most O(∥ϵ∥1) less
social welfare than X′ under the true means µ∗. An application of the triangle inequality gives that,

⟨Y µ∗, µ∗⟩F −⟨ˆX, µ∗⟩F = ⟨Y µ∗, µ∗⟩F −⟨X′, µ∗⟩F + ⟨X′, µ∗⟩F −⟨ˆX, µ∗⟩F
= ˜O (∥ϵ∥1 + ∥ϵ∥1)

= ˜O(T −1/3).

Thus, the total regret for the steps after the warm-up period is ˜O(T 2/3). The regret of the warm-up
period is at most (b −a)T 2/3 due to the assumption that the mean values are bounded. We can
therefore conclude that the total regret is ˜O(T 2/3), and this completes the proof of regret. Finally, we
note that by construction of LP (3), if µ∗∈ˆµ ± ϵ then the chosen fractional allocations ˆX must also
satisfy the constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1 as desired. See Appendix D for the full proof.

5
Lower bounds

The following lower bound shows that Theorem 1 is tight up to log factors. An equivalent result
holds for proportionality, with the same proof.

Theorem 2. There exists a, b, n, m such that no algorithm can, for all µ∗∈[a, b]n×m, both satisfy
the envy-freeness constraints and achieve regret of less than
T 2/3
log(T ) with probability at least 1 −1/T.
The same is true for the proportionality constraints.

Proof sketch. We defer the formal proof to Appendix G and provide brief intuition here.

Suppose there are two item types and two players. In this case envy-freeness and proportionality
are equivalent, and therefore we will focus on the former. Consider the following two mean value
matrices.

µ1 =

2
3
1
1



µ2 =
2
3
1
1 + T −1/3



9


---Page Break---
We will show that no algorithm can with probability 1 −1/T achieve regret of less than ˜Ω(T 2/3)
and satisfy envy-freeness in expectation for both of these distributions. First, note that the expected
social welfare maximizing allocation for µ1 is to give all items of type 1 to player 2 and all items of
type 2 to player 1. On the other hand, any envy-free allocation for µ2 must give ˜Ω(T −1/3) fraction
of items of type 2 to player 2. This implies that if an algorithm is unable to distinguish between µ1
and µ2, then either the regret will be ˜Ω(T 2/3) for µ1 or the algorithm will not be envy-free for µ2.

Therefore, any algorithm that has regret of less than ˜Ω(T 2/3) and satisfies envy-freeness for both µ1
and µ2 must distinguish betwen µ1 and µ2. The only way to do this is to allocate at least ˜Ω(T 2/3)
items of type 2 to player 2. However, this will result in regret under µ1 of ˜Ω(T 2/3).

6
Discussion

We conclude by discussing some limitations and open questions. First, Algorithm 1 involves solving a
linear program with an infinite number of linear constraints. Linear programs with an infinite number
of constraints (called semi-infinite programs) are well-studied and occur in many applications [16, 24].
We also note that a finite number of (exponentially many) constraints suffices for envy-freeness
in expectation and proportionality in expectation by bounding all of the possible extreme values
of ϵ. Nevertheless, we opted to avoid this representation because it significantly complicates the
presentation of the algorithm. Furthermore, there also exists a polynomial time separation oracle
for determining whether an allocation satisfies the infinitely many constraints, which would allow
techniques such as the Ellipsoid Method [6] to solve the linear program in polynomial time.

Second, while the regret coefficients for proportionality are polynomial in n, a practical limitation
of Algorithm 1 for envy-freeness is that the ˜O term is exponential in n. We expect, however, that
the worst-case bound we present in Lemma 1 is far from tight. Whether there exists a bound on the
regret that is polynomial in n for learning under envy-freeness in expectation constraints is an open
question for future work.

The other natural question that remains open for future work is whether we can achieve ˜O(
√

T)
regret while maintaining envy-freeness in expectation or proportionality in expectation. If the optimal
solution Y µ∗has a positive slack in every constraint, then an upper confidence bound (UCB) approach
would be likely to work. Unfortunately, the fairness constraints for envy-freeness in expectation and
proportionality in expectation are often tight for the optimal allocation. Furthermore, the constraints
have a constant (greater than 1/n) dependence on every unknown value in the µ∗matrix. Therefore,
every mean value µ∗
ik might need to be learned with high accuracy even if the optimal allocation does
not allocate item type k to player i.

We also note that there exist additional (albeit less prominent) fairness notions for the problem of
online fair division, such as equitability, which may satisfy additional properties that allow for lower
regret. We leave the question of studying more fairness notions for future work.

Finally, a broader question is whether the connection we have established between multi-armed
bandits and online fair division might be leveraged to give a fresh perspective on additional problems
in this area, such as online cake cutting [31].

Acknowledgments and Disclosure of Funding

Procaccia gratefully acknowledges research support by the National Science Foundation under grants
IIS-2147187, IIS-2229881 and CCF-2007080; and by the Office of Naval Research under grant
N00014-20-1-2488. Schiffer was supported by an NSF Graduate Research Fellowship. Zhang was
supported by an NSF Graduate Research Fellowship.

10


---Page Break---
References

[1] M. Aleksandrov, H. Aziz, S. Gaspers, and T. Walsh. Online fair division: Analysing a food bank
problem. In Proceedings of the 24th International Joint Conference on Artificial Intelligence
(IJCAI), pages 2540–2546, 2015.

[2] S. Amani, M. Alizadeh, and C. Thrampoulidis. Linear stochastic bandits under safety constraints.
Advances in Neural Information Processing Systems, 32, 2019.

[3] H. Aziz and S. Mackenzie. A discrete and bounded envy-free cake cutting protocol for any
number of agents. In Proceedings of the 57th Symposium on Foundations of Computer Science
(FOCS), pages 416–427, 2016.

[4] E. Balkanski, S. Brânzei, D. Kurokawa, and A. D. Procaccia. Simultaneous cake cutting. In
Proceedings of the 28th AAAI Conference on Artificial Intelligence (AAAI), pages 566–572,
2014.

[5] G. Benadè, A. M. Kazachkov, A. D. Procaccia, A. Psomas, and D. Zeng. Fair and efficient
online allocations. Operations Research, 2024. Forthcoming.

[6] R. G. Bland, D. Goldfarb, and M. J. Todd. The ellipsoid method: A survey. Operations
Research, 29(6):1039–1091, 1981.

[7] S. J. Brams and A. D. Taylor. An envy-free cake division protocol. American Mathematical
Monthly, 102(1):9–18, 1995.

[8] E. Carlsson, D. Basu, F. Johansson, and D. Dubhashi. Pure exploration in bandits with linear
constraints. In International Conference on Artificial Intelligence and Statistics, pages 334–342.
PMLR, 2024.

[9] G. Chen, X. Li, and Y. Ye. Fairer lp-based online allocation via analytic center. arXiv preprint
arXiv:2110.14621, 2021.

[10] Y. Chen, A. Cuellar, H. Luo, J. Modi, H. Nemlekar, and S. Nikolaidis. Fair contextual multi-
armed bandits: Theory and experiments. In Conference on Uncertainty in Artificial Intelligence,
pages 181–190. PMLR, 2020.

[11] H. Claure, Y. Chen, J. Modi, M. Jung, and S. Nikolaidis. Multi-armed bandits with fairness con-
straints for distributing resources to human teammates. In Proceedings of the 2020 ACM/IEEE
International Conference on Human-Robot Interaction, pages 299–308, 2020.

[12] J. Edmonds and K. Pruhs. Cake cutting really is not a piece of cake. In Proceedings of the 17th
Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 271–278, 2006.

[13] Y. Gal, M. Mash, A. D. Procaccia, and Y. Zick. Which is the fairest (rent division) of them all?
Journal of the ACM, 64(6): article 39, 2017.

[14] J. Goldman and A. D. Procaccia. Spliddit: Unleashing fair division algorithms. SIGecom
Exchanges, 13(2):41–46, 2014.

[15] R. Grazzi, A. Akhavan, J. Falk, L. Cella, and M. Pontil. Group meritocratic fairness in linear
contextual bandits. Advances in Neural Information Processing Systems, 35:24392–24404,
2022.

[16] R. Hettich and K. Kortanek. Semi-infinite programming: theory, methods, and applications.
SIAM review, 35(3):380–429, 1993.

[17] M. Joseph, M. Kearns, J. Morgenstern, S. Neel, and A. Roth. Fair algorithms for infinite and
contextual bandits. arXiv:1610.09559, 2016.

[18] M. Joseph, M. Kearns, J. H. Morgenstern, and A. Roth. Fairness in learning: Classic and
contextual bandits. Advances in Neural Information Processing Systems, 29, 2016.

11


---Page Break---
[19] Min Kyung Lee, Daniel Kusbit, Anson Kahng, Ji Tae Kim, Xinran Yuan, Allissa Chan, Daniel
See, Ritesh Noothigattu, Siheon Lee, Alexandros Psomas, et al. Webuildai: Participatory
framework for algorithmic governance. Proceedings of the ACM on human-computer interaction,
3(CSCW):1–35, 2019.

[20] F. Li, J. Liu, and B. Ji. Combinatorial sleeping bandits with fairness constraints. IEEE
Transactions on Network Science and Engineering, 7(3):1799–1813, 2019.

[21] Q. Liu, W. Xu, S. Wang, and Z. Fang. Combinatorial bandits with linear constraints: Beyond
knapsacks and fairness. Advances in Neural Information Processing Systems, 35:2997–3010,
2022.

[22] X. Liu, B. Li, P. Shi, and L. Ying. An efficient pessimistic-optimistic algorithm for stochastic
linear bandits with general constraints. Advances in Neural Information Processing Systems,
34:24075–24086, 2021.

[23] Y. Liu, G. Radanovic, C. Dimitrakakis, D. Mandal, and D. C. Parkes. Calibrated fairness in
bandits. arXiv preprint arXiv:1707.01875, 2017.

[24] M. López and G. Still. Semi-infinite programming. European Journal of Operational Research,
180(2):491–518, 2007.

[25] Marios Mertzanidis, Alexandros Psomas, and Paritosh Verma. Automating food drop: The
power of two choices for dynamic and fair food allocation. arXiv preprint arXiv:2406.06363,
2024.

[26] A. Moradipari, M. Alizadeh, and C. Thrampoulidis. Linear thompson sampling under unknown
linear constraints. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 3392–3396. IEEE, 2020.

[27] A. Pacchiano, M. Ghavamzadeh, P. Bartlett, and H. Jiang. Stochastic bandits with linear
constraints. In International Conference on Artificial Intelligence and Statistics, pages 2827–
2835. PMLR, 2021.

[28] V. Patil, G. Ghalme, V. Nair, and Y. Narahari. Achieving fairness in the stochastic multi-armed
bandit problem. Journal of Machine Learning Research, 22(174):1–31, 2021.

[29] C. Schumann, Z. Lang, N. Mattei, and J. P. Dickerson. Group fairness in bandit arm selection.
arXiv preprint arXiv:1912.03802, 2019.

[30] F. E. Su. Rental harmony: Sperner’s lemma in fair division. American Mathematical Monthly,
106(10):930–942, 1999.

[31] T. Walsh. Online cake cutting. In Proceedings of the 3rd International Conference on Algorith-
mic Decision Theory (ADT), pages 292–305, 2011.

[32] L. Wang, Y. Bai, W. Sun, and T. Joachims. Fairness of exposure in stochastic bandits. In
International Conference on Machine Learning, pages 10686–10696. PMLR, 2021.

[33] W. Wei, X. Ma, and J. Wang. Fair adaptive experiments. Advances in Neural Information
Processing Systems, 36, 2024.

[34] Y. Wu, Z. Zheng, and T. Zhu. Best arm identification with fairness constraints on subpopulations.
In 2023 Winter Simulation Conference (WSC), pages 540–551. IEEE, 2023.

[35] Hakuei Yamada, Junpei Komiyama, Kenshi Abe, and Atsushi Iwasaki. Learning fair division
from bandit feedback. In International Conference on Artificial Intelligence and Statistics,
pages 3106–3114. PMLR, 2024.

12


---Page Break---
A
Algorithmic Representation of Model

Algorithm 2 [Online Item Allocation]
Require: ALG

1: ∀i, A0
i ←{}, H0 ←{}
2: for t ←1 to T do
3:
Xt ←ALG(Ht)
4:
kt ∼D
5:
Generate item jt of type kt (i.e. Vi(jt) ∼N(µ∗
ikt, 1), ∀i ∈N )
6:
it ←Sample from (Xt)⊤
kt
7:
At
it = At−1
it
+ {jt}
8:
Ht ←Ht−1 + (kt, it, Vit(jt))
9: end for
10: return A = (AT
1 , AT
2 , ..., AT
n)

B
Motivating Fairness in Expectation

In this section, we will explore the relationship between envy-freeness in expectation and realized
envy, as well as the relationship between requiring envy-freeness in expectation at every time step and
only requiring envy-freeness in expectation at the end of round T. For this section, we will use the
following notation. For any algorithm ALG, we denote Xt = ALG(Ht) as the fractional allocation
used at time t, and PrALG(i, k, Ht) := (Xt)ik.

Previous works on fair online allocation of indivisible goods have focused on the fairness of the final,
realized allocation instead of studying fairness in expectation as in Definitions 1 and 2 [5]. We define
the realized envy of an allocation below.

Definition 4. The realized envy of allocation A at time τ is maxi,i′ Vi(Aτ
i′) −Vi(Aτ
i ).

We show in the following theorems that algorithms which are envy-free in expectation are within a
log(T) factor of being “optimal” in terms of final realized envy. Informally, Theorem 3 shows that in
our setting, no algorithm can with high probability output an allocation A(ALG) with realized envy
less than
√

t. Conversely, Theorem 4 shows that any algorithm ALG that satisfies envy-freeness in
expectation will output a final allocation A(ALG) with realized envy of at most
√

T log(T) with
high probability.

Theorem 3. Suppose µ∗is known. For any algorithm ALG and for any τ ∈[T], with probability at
least 1/16 the allocation Aτ(ALG) has realized envy of more than √τ.

proof. Assume there are two players and only one item type, and assume that all values are drawn
from N(µ, 1). Fix τ ≥log2(T). As in Algorithm 2, define it ∈{1, 2} as the player allocated the
item at time t. The realized envy of the two players can be written as:

Realized Envy of Player 1 at time τ =

 τ
X

t=0
1it=1 · V1(jt)

!

−

 τ
X

t=0
1it=2 · V1(jt)

!

(4)

Realized Envy of Player 2 at time τ =

 τ
X

t=0
1it=2 · V2(jt)

!

−

 τ
X

t=0
1it=1 · V2(jt)

!

.
(5)

Let va
i be the value of the assigned player for item i, and vu
i be the value of the unassigned player
for item i. Note that va
i , vu
i ∼N(µ, 1), because at the time of allocation, ALG does not know either
player’s value for the item, and player values are therefore independent of the assignment.

By this coupling argument, Equations (4) and (5) can be rewritten as:

Realized Envy of Player 1 at time τ =

 τ
X

t=0
1it=1 · va
i

!

−

 τ
X

i=0
1it=2 · vu
i

!

(6)

13


---Page Break---
Realized Envy of Player 2 at time τ =

 τ
X

t=0
1it=2 · va
i

!

−

 τ
X

t=0
1it=1 · vu
i

!

.
(7)

By the Central Limit Theorem, Pτ
i=0 va
i and Pτ
i=0 vu
i are both N(τ · µ, τ). Therefore, for any τ,
with probability at least 1/16, we have that

τ
X

t=0
va
t ≤τµ −√τ
and

τ
X

t=0
vu
t ≥τµ + √τ.

Putting these equations together, we have that

τ
X

t=0
va
t + 2√τ ≤

τ
X

t=0
vu
t .

However, this result with Equations (6) and (7) implies that the envy must be at least √τ for any
choice of {it}. Therefore, we have shown that with probability at least 1/16, the envy at time τ will
be at least √τ for any possible algorithm.

Theorem 4. Suppose µ∗is known. Also assume that all of the value distributions are bounded by a
constant B. If ALG is deterministic (i.e. ALG(Ht) is a deterministic function of Ht) and satisfies
envy-freeness in expectation, then with probability 1 −o(1/T), the realized envy of Aτ(ALG) at
every time τ ∈[T] is at most √τ log(T).

proof. We will bound the realized envy of A(ALG) between any two specific players i, i′, which
will then imply a bound on the realized envy of A(ALG) by a union bound. The key observation is
that each round of Algorithm 2 consists of first a random draw from D to determine the item type kt
and then a draw from ξt ∼Unif([0, 1]) which determines the player to which the item is allocated
based on Xt = ALG(Ht). Formally, the tth item is allocated to player i if

ξt ∈

" i−1
X

i′′=1
(Xt)i′′kt ,

i
X

i′′=1
(Xt)i′′kt

#

.

Define {vti′′}n
i′′=1 as the values of the players for the item at time t. This is the final source of
randomness in round t. Therefore, the allocation of any player at time τ is a function of the random
variable sequence {(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0 .

Let Eτ represent the envy accrued by player i for player i′ up until but not including time τ. Then

Eτ =

τ−1
X

t=0
vti ·

m
X

k=1
1kt=k ·

1ξt∈[Pi′−1
s=1 (Xt)skt,Pi′
s=1(Xt)skt] −1ξt∈[Pi−1
s=1(Xt)skt,Pi
s=1(Xt)skt]


:= f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

.

Now we will apply McDiarmid’s inequality to the function f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

.
First, we show that the bounded condition is satisfied.
If {(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0 and
{(k′
t, ξ′
t, {v′
ti′′}n
i′′=1)}τ−1
t=0 differ only at the sth element for any s ∈[0, τ −1], then
f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

−f
 
{(k′
t, ξ′
t, {vti′′}n
i′′=1)}τ−1
t=0
 ≤B.

Therefore, we can apply McDiarmid’s inequality to get that

Pr
 f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

−E

f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0
 ≥√τ log(T)

≤e−log2(T )/(2B2)

≤T −log(T )/(2B2).

14


---Page Break---
Since ALG is envy-free in expectation, we know that E

f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

≤0. There-
fore, we must have that

Pr
 
f
 
{(kt, ξt, {vti′′}n
i′′=1)}τ−1
t=0

≥√τ log(T)

≤T −log(T )/(2B2).

Taking a union bound over all pairs i, i′ and all τ ∈[T], we have that

Pr(∃τ : Realized envy at time τ is ≥√τ log(T) ) ≤n2T · T −2 log(T )/b2 = o(1/T).

Therefore, with probability 1 −o(1/T), the realized envy at evert time τ is at most √τ log(T).

Note that Theorem 3 does not contradict the results of Benadè et al. [5], who achieve envy-freeness
of the realized allocation with high probability when the true player values for the items are known at
time of allocation (as opposed to our model which only knows the item type). When player values
for item jt are known before assignment, Theorem 3 does not apply.

We also defined envy-freeness in expectation as a constraint which needs to hold at every time step t.
In some fair division applications, we may only care about “fairness” of the total allocation at the
end of the process. Therefore, we could instead only require that no player is envious in expectation
of any other player at the end of all T rounds. However, Theorem 5 shows that this is equivalent to
maintaining envy-freeness in expectation at all times t ∈[T] when maximizing social welfare.

Theorem 5. Suppose µ∗is known. Let E be the class of all algorithms that are envy-free in expectation
and let F be the class of all algorithms which satisfy E[Vi(AT
i )] ≥max
i′∈[n] E[Vi(AT
i′)] for all i. Then

max
ALG∈F E[sw(A(ALG))] = max
ALG∈E E[sw(A(ALG))].

proof. By definition, E ⊆F, which proves one direction of the desired equality. We will now show
that for any ALG ∈F, there exists an algorithm ALG′ ∈E such that

E[sw(A(ALG′))] = E[sw(A(ALG))].

First, observe that by the definition of F, we have that for all i, i′,

1
T E



X

t∈[T ]

X

k
Pr
ALG(i, k, Ht)µik Pr
D (k)



≥1

T E



X

t∈[T ]

X

k
Pr
ALG(i′, k, Ht)µik Pr
D (k)



.
(8)

By linearity of expectation, Equation (8) is equivalent to

1
T

X

t∈[T ]

X

k

Z

Ht
Pr
ALG(i, k, Ht)dHt · µik Pr
D (k) ≥1

T

X

t∈[T ]

X

k

Z

Ht
Pr
ALG(i′, k, Ht)dHt · µik Pr
D (k).

(9)
Furthermore, by definition

E[sw(ALG)] = E




n
X

i=1

X

t∈[T ]

X

k
Pr
ALG(i, k, Ht)µik Pr
D (k)





=

n
X

i=1

X

t∈[T ]

X

k

Z

Ht
Pr
ALG(i, k, Ht)dHt · µik · Pr
D (k)

=

n
X

i=1

X

k



X

t∈[T ]

Z

Ht
Pr
ALG(i, k, Ht)dHt



· µik · Pr
D (k).

We will construct the algorithm ALG′ as follows. Suppose ALG′ is time-independent and history-
independent, such that for all t, Ht,

Pr
ALG′
t
(i, k, Ht) = 1

T

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHs.

15


---Page Break---
The expected social welfare of ALG′ is

E[sw(ALG′)] = E




n
X

i=1

X

t∈[T ]

X

k
Pr
ALG′
t
(i, k, Ht)µik Pr
D (k)





= E




n
X

i=1

X

t∈[T ]

X

k

1
T

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHs · µik · Pr
D (k)





= E




n
X

i=1

X

k



X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHs



· µik · Pr
D (k)





= E[sw(ALG)].

Therefore, ALG and ALG′ have the same expected social welfare. Finally, we need to show that
ALG′ ∈E. This is equivalent to showing that for all t and Ht,
X

k
Pr
ALG′
t
(i, k, Ht)µik Pr
D (k) ≥
X

k
Pr
ALG′
t
(i′, k, Ht)µik Pr
D (k).

Starting with the LHS and plugging in the definition of ALG′, we have that

X

k
Pr
ALG′
t
(i, k, Ht)µik Pr
D (k) =
X

k

1
T

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHsµik Pr
D (k)

= 1

T

X

k

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHsµik Pr
D (k)

≥1

T

X

k

X

s∈[T ]

Z
Pr
ALGs(i′, k, Hs)dHsµik Pr
D (k)
[Equation (9)]

=
X

k

1
T

X

s∈[T ]

Z
Pr
ALGs(i′, k, Hs)dHsµik Pr
D (k)

=
X

k
Pr
ALG′
t
(i′, k, Ht)µik Pr
D (k),

as desired. Therefore, we have shown that ALG′ ∈E.

Theorem 6. The algorithm which maximizes expected social welfare subject to EFE up to time T is
time-independent, history-independent, and can be calculated in polynomial time.

proof. By Theorem 5, there exists an optimal algorithm that satisfies envy-freeness in expectation
and that is time-independent and history-independent. To find the best such fractional allocation, all
we must do is solve LP (1) with the envy-freeness in expectation constraints.

B.1
Proportionality
Theorems 3–6 also have equivalent forms for proportionality. We define the realized proportionality
gap as the equivalent of envy for the proportionality constraints. This implies that a proportional
allocation has non-positive proportionality gap.

Definition 5. The realized proportionality gap of allocation A at time τ is maxi 1

n
P

i′ Vi(Aτ
i′) −
Vi(Aτ
i ).

As in Theorems 3 and 4 , the following two results give that algorithms which satisfy proportionality
in expectation are within a log(T) factor of optimal for the realized proportionality gap.

Theorem 7. Suppose µ∗is known. For any algorithm ALG and for any τ ∈[T], with probability at
least 1/16 the allocation Aτ(ALG) has realized proportionality gap of more than √τ.

16


---Page Break---
proof. As in the proof of Theorem 3, assume there are two players and only one item type, and
assume that all values are drawn from N(µ, 1). Then the realized proportionality gap of the two
players can be written as:

Realized proportionality gap of Player 1 at time τ = 1

2

 τ
X

t=0
1it=1 · V1(jt)

!

−1

2

 τ
X

t=0
1it=2 · V1(jt)

!

(10)

Realized proportionality gap of Player 2 at time τ = 1

2

 τ
X

t=0
1it=2 · V2(jt)

!

−1

2

 τ
X

t=0
1it=1 · V2(jt)

!

(11)
These equations only differ from those of realized envy by a scalar factor, and therefore the rest of
the proof follows exactly as in Theorem 3.

Theorem 8. Suppose µ∗is known. Also assume that all of the value distributions are bounded by a
constant B. If ALG is deterministic (i.e. ALG(Ht) is a deterministic function of Ht) and satisfies
proportionality in expectation, then with probability 1 −o(1/T), the realized proportionality gap of
Aτ(ALG) at every time τ ∈[T] is at most √τ log(T).

proof. In this proof, we can let Et be the accrued “proportionality gap” of any player i. Then as in
Theorem 4, an application of McDiarmid’s inequality allows us to bound the realized proportionality
gap with high probability for all τ.

The following two theorems are analogs of Theorem 5 and Theorem 6 for envy-freeness in expec-
tation. Together, these theorems imply that maximizing social welfare subject to proportionality in
expectation at every time step is equivalent to maximizing social welfare subject to proportionality in
expectation only at the end of round T.

Theorem 9. Suppose µ∗is known. Let E be the class of all algorithms that are proportional in

expectation and let F be the class of all algorithms which satisfy E[Vi(AT
i )] ≥1

n

X

i′∈[n]
E[Vi(AT
i′)]

for all i. Then
max
ALG∈F E[sw(A(ALG))] = max
ALG∈E E[sw(A(ALG))].

proof. Suppose ALG ∈F. We will define ALG′ as in Theorem 5 to be

Pr
ALG′
t
(i, k, Ht) = 1

T

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHs.

By this construction, ALG′ ∈E because
X

k
Pr
ALG′
t
(i, k, Ht)µik Pr
D (k)

=
X

k

1
T

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHsµik Pr
D (k)

= 1

T

X

k

X

s∈[T ]

Z
Pr
ALGs(i, k, Hs)dHsµik Pr
D (k)

≥
1
Tn

n
X

i′=1

X

k

X

s∈[T ]

Z
Pr
ALGs(i′, k, Hs)dHsµik Pr
D (k)
[ALG ∈F]

=

n
X

i′=1

X

k

1
Tn

X

s∈[T ]

Z
Pr
ALGs(i′, k, Hs)dHsµik Pr
D (k)

= 1

n

n
X

i′=1

X

k
Pr
ALG′
t
(i′, k, Ht)µik Pr
D (k).

17


---Page Break---
Because E ⊆F and because we showed in Theorem 5 that ALG′ and ALG have the same expected
social welfare, the desired result follows.

Theorem 10. The algorithm which maximizes expected social welfare subject to PE up to time T is
time-independent, history-independent, and can be calculated in polynomial time.

proof. By Theorem 9, there exists an optimal algorithm that satisfies proportionality in expectation
that is time-independent and history-independent. To find the best such fractional allocation, all we
must do is solve LP (1) with the proportionality in expectation constraints.

C
Additional Model Notes

C.1
Choice of D
In the body of the paper, we focus on the case when D is the uniform distribution over item types.
However, our results generalize to any distribution D which does not depend on T. In this case, the
social welfare of a fractional allocation X becomes P

i,k Xik PrD(k)µik. For a matrix ν ∈Rn×m,
define fD(ν) = ν′ ∈Rn×m, where ν′
ik = n · PrD(k) · µik. For mean values µ∗, define µ′ = fD(µ∗).
Then social welfare of a fractional allocation X with means µ∗and distribution D is then simply
1
n⟨X, µ′⟩F as in the uniform D case. Similarly, the envy-freeness in expectation or proportionality
in expectation constraints on X with means µ∗and item distribution D can be represented as
⟨Bℓ(µ′), X⟩F ≥cℓ, which is an equivalent form to the constraints when D is uniform. Therefore,
for arbitrary D we can use Algorithm 1 with only two slight modifications. The first is we must
transform the ˆµ, µ, and other components of the linear programs using the function fD. The second
modification is that we potentially need more exploration steps (larger T) in the warm-up period to
guarantee the same level of estimation of µ∗, depending on the value of mini,k PrD(k). However,
since we require that D does not depend on T, this will not change the overall regret of the algorithm.

C.2
Lower Bound on Means
In this section, we show that if the means of player values can be arbitrarily close to zero, then it can
be impossible to achieve a regret of o(T).

Theorem 11. For ϵ > 0, there does not exist an algorithm ALG such that for any possible
µ∗∈[0, ϵ]n×m, for every t the fractional allocation Xt chosen by ALG satisfies envy-freeness
in expectation with probability greater than 1/2 and the regret of ALG is o(T). The same result also
holds for proportionality in expectation.

proof. W.l.o.g. assume that ϵ = 1. The proof also holds for any other constant ϵ. Suppose the
underlying value distributions are Bernoulli, that we have two players and two item types, and assume
T ≥2. We will consider two cases for µ∗and show that no algorithm can with probability greater
than 1/2 satisfy the constraints and have regret of o(T) for both of these cases of µ∗.

First, let

µ1 =

1/T 2
0
1
0.5


and

µ2 =

0
1/T 2
1
0.5


.

If µ∗= µ1 or µ∗= µ2, then player 1 will not have a realized value of 1 for any of the T items with
probability at least 1/2. Therefore, no algorithm can differentiate between µ∗= µ1 and µ∗= µ2
with probability at least 1/2. The only fractional allocation that is envy-free for both µ∗= µ1
and µ∗= µ2 is the uniform at random allocation. However, this allocation has regret of Ω(T) for
µ∗= µ2, as the best fractional allocation when µ∗= µ2 is the fractional allocation

Y µ2 =

0
0.5
1
0.5


.

This proves the desired result that no algorithm can be envy-free for µ∗and have o(T) regret for
both possible realizations of µ∗. Similarly, the uniform at random allocation is the only proportional
allocation in this example, and therefore the same result holds.

18


---Page Break---
D
Proof of Theorem 1

Observations 1, 2, and 3 give that the proportionality in expectation constraints and the envy-freeness
in expectation constraints satisfy Properties 1, 3, and 4 respectively. Lemmas 1 and 2 respectively give
that the proportionality in expectation constraints and the envy-freeness in expectation constraints
satisfy Property 2. These results combined with Lemma 3 directly prove Theorem 1.

Lemma 3. Let

{(Bℓ(µ), cℓ)}L
ℓ=1
	

µ∈[a,b]n×m be a family of constraints that satisfy Properties 1, 3,

4, and 2. Then with probability 1 −1/T, Algorithm 1 satisfies constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1 and
has regret of ˜O(T 2/3) for constraints {(Bℓ(µ∗), cℓ)}L
ℓ=1.

proof. First, we note that the regret of the first T 2/3 steps can be bounded by

T 2/3−1
X

t=0
⟨Y µ∗, µ∗⟩F −⟨Xt, µ∗⟩F ≤T 2/3(b −a) = O(T 2/3).
(12)

By Property 1, the uniform at random allocation satisfies constraints {(Bℓ(µ), cℓ}L
ℓ=1. Therefore, Xt
satisfies the constraints for all t < T 2/3. Furthermore, because the fractional allocation was uniform
at random for the first T 2/3 steps, we have that for sufficiently large T,

Pr

∥ϵ∥1 ≤nm
q

nm log2(4nmT) · T −1/3


≥Pr

∀i ∈[n], k ∈[m], ϵik ≤
q

nm log2(4nmT) · T −1/3


= Pr

∀i ∈[n], k ∈[m], Nik ≥T 2/3

2nm



= Pr

∀i ∈[n], k ∈[m], Nik ≥T 2/3

nm −T 2/3

2nm



≥Pr

∀i ∈[n], k ∈[m], Nik ≥T 2/3

nm −
p

log(4nmT) · T 1/3


≥1 −nme−2 log(4nmT )

≥1 −1

2T ,
(13)

where the second to last inequality is by Hoeffding’s Inequality and a union bound. This implies
that with probability 1 −
1
2T , ∥ϵ∥1 = ˜O(T −1/3). Because the values are drawn from a Sub-Gaussian
distribution, there exists a constant c > 0 (which depends on the distribution of the values) such that
by Hoeffding’s inequality,

Pr (∀i ∈[n], k ∈[m], |ˆµik −µ∗
ik| ≤ϵik) ≥1 −2nme−c log2(4nmT )

≥1 −2nm

1
4nmT

c log(4nmT )

≥1 −1

2T .
[For sufficiently large T]

(14)
For the rest of this proof, we will assume that

∥ϵ∥1 ≤˜O(T −1/3)
(15)
and
∀i ∈[n], k ∈[m], |ˆµik −µ∗
ik| ≤ϵik,
(16)
which by Equations (13) and (14) happens with probability 1 −1/T.

If K is the Lipschitz constant for this family of constraints, then by Equation (15), 2K∥ϵ∥1 ≤
˜O(T −1/3) ≤γ0 for sufficiently large T, where γ0 is from Property 2. Therefore, taking γ = 2K∥ϵ∥1
in Property 2 gives that there exists some fractional allocation X′ such that

|⟨µ∗, Y µ∗⟩F −⟨µ∗, X′⟩F | ≤O(∥ϵ∥1),
(17)

19


---Page Break---
and such that for every constraint ℓ∈[L], either ∀i1, i2 ∈{i : Bℓ(µ∗)i ̸= 0}, X′
i1 = X′
i2 or

⟨Bℓ(µ∗), X′⟩F ≥cℓ+ 2K∥ϵ∥1.
(18)

For any µ ∈ˆµ ± ϵ, we have that ∥µ −µ∗∥1 ≤2∥ϵ∥1 by Equation (16) and the triangle inequality.
By the Lipschitz continuity assumption (Property 3), this implies that for all µ ∈ˆµ ± ϵ,

∥Bℓ(µ) −Bℓ(µ∗)∥1 ≤2K∥ϵ∥1.
(19)

Therefore, if Equation (18) holds for a constraint ℓ, then for any µ ∈ˆµ ± ϵ,

⟨Bℓ(µ), X′⟩F = ⟨Bℓ(µ), X′⟩F −⟨Bℓ(µ∗), X′⟩F + ⟨Bℓ(µ∗), X′⟩F
= ⟨Bℓ(µ) −Bℓ(µ∗), X′⟩F + ⟨Bℓ(µ∗), X′⟩F
≥⟨Bℓ(µ∗), X′⟩F −∥Bℓ(µ) −Bℓ(µ∗)∥1
[0 ≤X′
ik ≤1, ∀i, k]

≥⟨Bℓ(µ∗), X′⟩F −2K∥ϵ∥1
[Equation (19)]
≥cℓ.
[Equation (18)]

Therefore, we have shown that if Equation (18) holds for a constraint ℓ, then the fractional allocation
X′ satisfies constraint (Bℓ(µ), cℓ) for all µ ∈ˆµ ± ϵ. If Equation (18) does not hold for a constraint ℓ,
then ∀i1, i2 ∈{i : Bℓ(µ∗)i ̸= 0}, X′
i1 = X′
i2. Because {i : Bℓ(µ∗)i ̸= 0} = {i : Bℓ(µ)i ̸= 0} by
Property 4, this implies by Property 1 that ⟨Bℓ(µ), X′⟩F ≥cℓfor all µ. Therefore, we have shown
that X′ satisfies {(Bℓ(µ), cℓ)}L
ℓ=1 for all µ ∈ˆµ ± ϵ, and thus X′ satisfies the constraints in LP (3).

Because ˆX is the optimal solution to LP (3), we have that

⟨X′, ˆµ⟩F ≤⟨ˆX, ˆµ⟩F .

By Equation (16), this implies that

⟨X′, µ∗⟩F −⟨ˆX, µ∗⟩F ≤∥ϵ∥1.
(20)

Therefore,

⟨Y µ∗, µ∗⟩F −⟨ˆX, µ∗⟩F = ⟨Y µ∗, µ∗⟩F −⟨X′, µ∗⟩F + ⟨X′, µ∗⟩F −⟨ˆX, µ∗⟩F
≤O (∥ϵ∥1 + ∥ϵ∥1)
[Equations (17) and (20)]

≤˜O(T −1/3).
[Equation (15)]
(21)

Combining with Equation (12), this gives a total regret of

T −1
X

t=0


⟨Y µ∗, µ∗⟩F −⟨Xt, µ∗⟩F

≤O(T 2/3) +

T
X

t=T 2/3


⟨Y µ∗, µ∗⟩F −⟨Xt, µ∗⟩F

[Equation (12)]

= O(T 2/3) + T · ˜O(T −1/3)
[Equation (21)]

= ˜O(T 2/3).

Lastly, we must show that the constraints are satisfied by the fractional allocation used by the
algorithm for t ≥T 2/3. This is because if Equation (16) holds, then any solution to LP (3) must
satisfy the constraints {(Bℓ(µ∗), cℓ}L
ℓ=1, and therefore the fractional allocation used by the algorithm
for all t ≥T 2/3 will satisfy these constraints. Recall that all of the above relies on Equations (16)
and (15) holding, which happens with probability 1 −1/T as desired.

E
Proof of Lemma 2

For proportionality in expectation, LP (1) can be rewritten as the following linear program.

Y µ := arg max ⟨X, µ⟩F

s.t. Xi · µi −1

n∥µi∥1 ≥0
∀i ∈[n]
X

i
Xik = 1
∀k
(22)

In order to show that the proportionality constraints satisfy Property 2, we want to construct an X′
such that

20


---Page Break---
1. X′ decreases the social welfare relative to Y µ by O(γ) and

2. For every constraint i ∈[n], either X′ = UAR or

X′
i · µi −1

n∥µi∥1 ≥γ.
(23)

LP (22) has n constraints, one corresponding to each player. Define

Si = Y µ
i · µi −1

n∥µi∥1.
(24)

Si is the slack on the ith constraint when using the optimal solution Y µ. Now we have two cases
depending on Pn
i=1 Si, the total amount of slack across all n players.

Case 1: Pn
i=1 Si ≤b

anγ

Let X′ = UAR. This will result in an decrease of social welfare of at most b

anγ compared to
Y µ. To see why, note that the slack of constraint i is equivalent to how much player i prefers their
fractional allocation in Y µ to UAR. Therefore, switching to UAR from Y µ decreases the total social
welfare by Si per player, and therefore decreases the total social welfare by Pn
i=1 Si ≤b

anγ = O(γ).
Furthermore, X′ = UAR clearly satisfies the other condition because every player is treated equally.

Case 2: Pn
i=1 Si > b

anγ

Intuitively, in this case we want to redistribute the slack from the constraints with a lot of slack to the
constraints without much slack. To do this, construct X′ as follows. Define

∆ik :=
Y µ
ik
Pm
k′=1 Y µ
ik′ ·
Si
Pn
i′=1 Si′ · nγ

a .
(25)

By construction, we have that
n
X

i=1

m
X

k=1
∆ik = nγ

a .
(26)

Because Pn
i=1 Si ≥b

anγ, we also have that

Si
P

i′ Si′ · nγ

a ≤Si

b .
(27)

Furthermore, for every i, Si ≤Y µ
i · µi by definition of Si. Because µik ≤b, this implies that
Si

b ≤Pm
k=1 Y µ
ik. With Equation (27), this implies that
Si
P

i′ Si′ · nγ

a ≤Pm
k=1 Y µ
ik. With Equation
(25), this implies that ∆ik ≤Y µ
ik. Finally, we note that

∆i · µi =
Y µ
i · µi
Pm
k′=1 Y µ
ik′ ·
Si
Pn
i′=1 Si′ · nγ

a
[Equation (25)]

≤
Pm
k=1 Y µ
ikµik
b Pm
k′=1 Y µ
ik′


Si
[Equation (27)]

≤
 Pm
k=1 Y µ
ik
Pm
k′=1 Y µ
ik′


Si
[µik ≤b]

= Si.
(28)

Now we are ready to construct X′. Let

X′
ik := Y µ
ik −∆ik + 1

n

n
X

i′=1
∆i′k.
(29)

In order for this to be a valid allocation, we need that X′
ik ≥0, which is true because we showed
above that ∆ik ≤Y µ
ik. We also need that Pn
i=1 X′
ik = 1, which follows from

n
X

i=1
X′
ik =

n
X

i=1

 

Y µ
ik −∆ik + 1

n

n
X

i′=1
∆i′k

!

= 1 −

n
X

i=1
∆ik +

n
X

i′=1
∆i′k = 1.

21


---Page Break---
Next, we will show that Equation (23) is satisfied for all i for fractional allocation X′. Starting with
the left hand side of Equation (23), we have

X′
i · µi −1

n∥µi∥1

= Y µ
i · µi −∆i · µi +

m
X

k=1

1
n

n
X

i′=1
∆i′kµik −1

n∥µi∥1
[Eq (29)]

= Si −∆i · µi + 1

n

n
X

i′=1
∆i′ · µi
[Eq (24)]

≥1

n

n
X

i′=1
∆i′ · µi
[Eq (28)]

≥a

n

n
X

i′=1

m
X

k=1
∆i′k
[µik ≥a]

= γ.
[Eq (26)]

Furthermore, we can bound the decrease in social welfare between fractional allocation X′ and Y µ
by

⟨Y µ, µ⟩F −⟨X′, µ⟩F = ⟨Y µ −X′, µ⟩F

≤

n
X

i=1
∆i · µi
[Equation (29)]

≤b

n
X

i=1

m
X

k=1
∆ik
[µik ≤b]

≤b

anγ.
[Equation (26)]

Therefore, we have shown that X′ has the desired properties, and thus the proportionality constraints
satisfy Property 2.

F
Proof of Lemma 1

For Section F only, we will assume w.l.o.g. that a = 1 and that b ≥1. This is without loss of
generality because envy-freeness in expectation constraints and social welfare are both scale invariant.
Therefore, scaling every player’s values (and mean values) by 1/a will give an equivalent problem
where a = 1.

To prove Lemma 1, will show that we can transform Y µ into a fractional allocation X′ which satisfies
Property 2 through Algorithm 3. Algorithm 3 iterates over the following types of ‘envy-with-slack’
graphs, which track whether a player prefers their allocation by at least α over another player’s
allocation.

Definition 6. Let create-slack-graph(µ, X, α) be the subroutine which returns the directed graph
with vertices N and edges generated as follows. Suppose G = create-slack-graph(µ, X, α). Then a
directed edge from i to i′ exists in G if and only if

⟨Xi, µi⟩−⟨Xi′, µi⟩< α.

At a high level, Algorithm 3 constructs ’envy-with-slack’ graphs with progressively smaller α,
with α ≥γ for all iterations. The algorithm operates on sets of nodes called equivalence classes,
where every pair of nodes in an equivalence class has the same allocation. We represent the set of
equivalence classes in a fractional allocation X as S(X).

Definition 7. Let S(X) be the set of equivalence classes of fractional allocation X, where two nodes
i, i′ are part of the same equivalence class S ∈S(X) if and only if Xi = Xi′.

22


---Page Break---
Algorithm 3 generally makes progress by either 1) merging two equivalence classes, or 2) removing
an edge from the graph. We formalize this model below.

Each iteration r begins with some allocation Xr and a slack value αr. Algorithm 3 then generates
from these parameters a directed graph Gr = create-slack-graph(µ, Xr, αr), which is the ’envy-
with-slack’ graph for allocation Xr given means µ. As in standard graph notation, for a graph G we
define V (G) as the vertices of G and E(G) as the edges of G. Each edge e = (i, i′) ∈E(Gr) has a
weight we = ⟨Xi, µi⟩−⟨Xi′, µi⟩. For a set of vertices S, we use δ+(S) to represent the edges with
head in S and tail in N\S, and similarly, we use δ-(S) to represent the edges with tail in S and head
in N\S. For notational convenience, we let δ-(i) = δ-({i}), and δ+(i) = δ+({i}). Throughout this
section, we will use the notation sw(X, µ) = ⟨X, µ⟩F .

An overview of the algorithm is as follows. In each iteration, Algorithm 3 generally performs one
of three operations and removes an edge by decreasing α. First, if there exists an equivalence class
S with at least one incoming edge but no outgoing edges, then operation remove-incoming-edge
transfers allocation probability from nodes in S to all other nodes. If such an equivalence class does
not exist, then Algorithm 3 finds a specific type of cycle in the ‘envy-with-slack’ graph. If there
exists some node which has an edge to some but not all of the nodes in the cycle, then operation
cycle-shift gives each node in the cycle half of its current allocation and half of the next node’s
allocation. If such a node does not exist and all edges in the graph have low enough weight, then
operation average-clique instead creates a new equivalence class by merging all equivalence classes
that the nodes in the cycle belong to. Such a merge may lead to envy, which is removed by a call to
Algorithm 4. We define each of the three operations formally below, where each operation returns a
new allocation X′.

Definition 8. Let S ∈S(X). Define XSk = Xik for every i ∈S. Let
remove-incoming-edge(S, α, X) be the subroutine which returns X′, where

X′
ik = Xik −(n −|S|)αXSk

2bn P

k′ XSk′
∀i ∈S

and

X′
ik = Xik +
|S|αXSk
2bn P
k′ XSk′
∀i ∈N\S

Definition 9. Let C be a cycle in a graph G = create-slack-graph(µ, X, α) and next(i) be the node
which i points to in C. Then the subroutine cycle-shift(C, X) returns X′, where

X′
ik = Xik + Xnext(i)k

2
∀i ∈V (C)

X′
ik = Xik
∀i ∈N\V (C)

Definition 10. Let Q be a clique in a graph G = create-slack-graph(µ, X, α). Then the subroutine
average-clique(Q, X) returns X′, where

X′
ik =

P

i′∈Q Xi′k

|Q|
∀i ∈Q

X′
ik = Xik
∀i ∈N\Q

We also define two intermediary operations. Note that the arg min function may return the empty set,
a singleton, or a set with multiple elements.

Definition 11. Let G = create-slack-graph(µ, X, α) with edge set E. For each equivalence class
S ∈S(X), let ES = arg mine∈δ+(S) we, where the size of ES may be 0, 1, or > 1. Let E′ =
P

S ES, and let G′ = (N, E′). Then the subroutine find-special-cycle(G, S(X)) returns a cycle C
of G′ where V (C) contains at most one member of each equivalence class S ∈S(X) (and returns ∅
if no such cycle exists).

23


---Page Break---
Definition
12.
Let
G
=
create-slack-graph(µ, X, α)
and
let
U
⊂
N.
Let
distribute-equally(U, β, X) be the subroutine which returns X′, where

X′
ik = (1 −|N\U| · β) · Xik
∀i ∈U

X′
ik = Xik + β ·
X

i′∈U
Xi′k
∀i ∈N\U

We are now ready to present Algorithm 3.

Algorithm 3 Achieving Envy with Slack

Require Y µ, µ
Let α0 ←γ · (en2 log(4bn4)+n3 log(2n), X0 ←Y µ, G0 ←create-slack-graph(µ, X0, α0), r ←0.
while ∃(u, v) ∈E(Gr) s.t. u, v are in different equivalence classes do

if ∃S ∈S(Xr) s.t. δ-(S) ̸= ∅and δ+(S) = ∅then

▷Xr+1 = remove-incoming-edge(S, αr, Xr)
▷αr+1 = αr

2b
else

▷C = find-special-cycle(Gr, S(Xr))
if ∃u ∈N and ∃i, i′ ∈V (C) such that (u, i) ∈E and (u, i′) /∈E then

▷Xr+1 = cycle-shift(C, Xr)
▷αr+1 = αr

2
else if ∃e ∈E(Gr) s.t. we ≥
αr
4bn4 then
▷Xr+1 = Xr
▷αr+1 =
αr
4bn4
else

▷S′ = {S ∈S(Xr) : S ∩V (C) ̸= ∅}
▷Q = S

S∈S′ S
▷Xavg = average-clique(Q, Xr)
▷αavg = αr

n
▷Gavg = create-slack-graph(µ, Xavg, αavg) // Gavg defined for analysis only.
▷X′ = Xavg
▷G′ = create-slack-graph(µ, X′, 0)
while ∃e ∈E(G′) do

▷X′ = remove-envy(µ, X′)
▷G′ = create-slack-graph(µ, X′, 0)
end while
▷Xr+1 = X′
▷αr+1 = αavg

2
end if
end if
▷Gr+1 ←create-slack-graph(µ, Xr+1, αr+1)
▷r = r + 1
end while
return Xr

Algorithm 3 calls remove-envy, which is equivalent to calling Algorithm 4. Algorithm 4 will require
the following definition.

Definition 13. Let create-non-negative-envy-graph(µ, X) be the subroutine which returns
the directed graph with vertices N and edges generated as follows.
Suppose G
=
create-non-negative-envy-graph(µ, X). Then a directed edge from i to i′ exists in G if and only if

⟨Xi, µi⟩−⟨Xi′, µi⟩≤0.

Note that this definition is exactly the same as that of create-slack-graph(µ, X, 0), except with a
weak instead of a strict inequality. We now present Algorithm 4.

We begin by proving some helpful lemmas. It will be convenient to define the following term.

24


---Page Break---
Algorithm 4 Remove Envy (also referred to as subroutine remove-envy(µ, X0))

Require µ, X0
Let G0 ←create-non-negative-envy-graph(µ, X0), r ←0.
if
{e ∈E(G0) : we < 0}
 = 0 then
return X0
end if
▷Choose (u, v) ∈{e ∈E(G0) : we < 0}
▷U ←{v′ ∈E(G0) : w(u,v′) < 0}
while w(u,v) < 0 and there does not exist a cycle in Gr containing (u, v) do

▷∀i ∈U, βi =
min
i′∈N\U,β≥0

 
β s.t. distribute-equally(U, β, Xr) returns X′ where ⟨X′
i, µi⟩= ⟨X′
i′, µi⟩


▷βu = β s.t. distribute-equally(U, β, Xr) returns X′ where ⟨X′
u, µu⟩= ⟨X′
v, µu⟩
▷i∗= arg mini∈({u}∪U) βi
▷Xr+1 = distribute-equally(U, βi∗, Xr)
▷U = U ∪{i∗}
▷Gr+1 ←create-non-negative-envy-graph(µ, Xr+1)
▷r = r + 1
end while
if w(u,v) < 0 then

▷C ←cycle in Gr containing (u, v)
// Let prev(i) and next(i) be the nodes before and after i in C, respectively.
▷∀i ∈V (C), X′
i = Xr
next(i)
else

▷X′ = Xr
end if
return X′

Definition 14. Let X be an envy-free fractional allocation for µ if for all i, i′ ∈N,

⟨Xi, µi⟩−⟨Xi′, µi⟩≥0.

Lemma
4.
Let
X
be
an
envy-free
fractional
allocation
for
µ,
and
let
G
=
create-slack-graph(µ, X, α) with edge set E. Suppose that there exists some equivalence class
S ∈S(X) such that δ-(S) ̸= ∅and δ+(S) = ∅in G, and let X′ = remove-incoming-edge(S, α, X).
Let G′ = create-slack-graph(µ, X′, α/2b) with edge set E′. Then |E′| < |E|.

proof. We first show that if an edge e /∈E, then e /∈E′. Note that for any i, i′ such that (i, i′) /∈E,

⟨Xi, µi⟩−⟨Xi′, µi⟩≥α > α

2b.

Therefore, to show that an edge (i, i′) not in E is also not in E′, it suffices to show that ⟨X′
i, µi⟩−
⟨X′
i′, µi⟩≥⟨Xi, µi⟩−⟨Xi′, µi⟩.

The subroutine remove-incoming-edge(S, α, X) only transfers weight from S to N\S, which implies
that no node in N\S will gain an edge to a node in S. Formally,

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩> ⟨Xi′, µi′⟩−⟨Xi, µi′⟩
∀i ∈S, i′ ∈N\S.
(30)

Every pair of nodes i, i′ ∈S has their fractional allocation reduced by the same amount, so

⟨X′
i, µi⟩−⟨X′
i′, µi⟩= ⟨Xi, µi⟩−⟨Xi′, µi⟩
∀i, i′ ∈S.
(31)

Similarly, every pair of nodes i, i′ ∈N\S has their fractional allocation increased by the same
amount, so
⟨X′
i, µi⟩−⟨X′
i′, µi⟩= ⟨Xi, µi⟩−⟨Xi′, µi⟩
∀i, i′ ∈N\S.
(32)

Finally, we observe that for any i ∈S and i′ ∈N\S,

⟨X′
i, µi⟩−⟨X′
i′, µi⟩=



⟨Xi, µi⟩−
X

k∈[m]

(n −|S|)αXSkµik

2bn P

k′ XSk′



−



⟨Xi′, µi⟩+
X

k∈[m]

|S|αXSkµik
2bn P

k′ XSk′





25


---Page Break---
= ⟨Xi, µi⟩−⟨Xi′, µi⟩−
X

k∈[m]

nαXSkµik
2bn P

k′ XSk′

≥⟨Xi, µi⟩−⟨Xi′, µi⟩−
X

k∈[m]

nαXSk
2n P

k′ XSk′

= ⟨Xi, µi⟩−⟨Xi′, µi⟩−α

2

≥α −α

2

≥α

2b.

where the second inequality is because (i, i′) ̸∈E. Therefore, by definition of G′, (i, i′) /∈E′.

Recall that δ-(S) ̸= ∅in G. We will now show that δ-(S) = ∅in G′. Observe that for i ∈S and
i′ ∈N\S,

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩=



⟨Xi′, µi′⟩+
X

k∈[m]

|S|αXSkµi′k
2bn P
k′ XSk′



−



⟨Xi, µi′⟩−
X

k∈[m]

(n −|S|)αXSkµi′k

2bn P
k′ XSk′





= ⟨Xi′, µi′⟩−⟨Xi, µi′⟩+
X

k∈[m]

nαXSkµi′k
2bn P

k′ XSk′

≥⟨Xi′, µi′⟩−⟨Xi, µi′⟩+ α

2b

≥α

2b.

Therefore, (i′, i) /∈E, which implies δ-(S) = ∅in G′. We conclude that all edges in E′ exist in E,
and at least one edge in E does not exist in E′, which implies that |E′| < |E|.

Lemma
5.
Let
X
be
an
envy-free
fractional
allocation
for
µ,
and
let
G
=
create-slack-graph(µ, X, α) with edge set E. Suppose that there exists some equivalence class
S ∈S(X) such that δ-(S) ̸= ∅and δ+(S) = ∅in G, and let X′ = remove-incoming-edge(S, α, X).
Then X′ is an envy-free allocation for µ.

proof. Let G′ = create-slack-graph(µ, X′, α/2b) with edge set E′. It suffices to show that we ≥
0 ∀e ∈E′. By Lemma 4, if an edge (i, i′) /∈E, then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩≥α

2b ≥0.

Consider any i ∈S and i′ ∈N\S. Then there must not exist an edge (i, i′) ∈E by assumption.
Also by assumption, for any i, i′ such that (i, i′) ∈E, we have that ⟨Xi, µi⟩−⟨Xi′, µi⟩≥0. By a
direct application of Equations (30), (31), and (32), we can conclude that ⟨X′
i, µi⟩−⟨X′
i′, µi⟩≥0
as well.

Lemma 6. Let G = create-slack-graph(µ, X, α) with edge set E.
Suppose that there ex-
ists some equivalence class S ∈S(X) such that δ-(S) ̸= ∅and δ+(S) = ∅in G, and let
X′ = remove-incoming-edge(S, α, X). Then

sw(X, µ) −sw(X′, µ) ≤αn

2 .

proof. Observe that

sw(X′, µ) =
X

i∈S
⟨X′
i, µi⟩+
X

i∈N\S
⟨X′
i, µi⟩

≥
X

i∈S


1 −
(n −|S|)α
2bn P

k′ XSk′


· ⟨Xi, µi⟩+
X

i∈N\S
⟨Xi, µi⟩

26


---Page Break---
≥
X

i∈S


1 −
α
2b P

k′ XSk′


· ⟨Xi, µi⟩+
X

i∈N\S
⟨Xi, µi⟩

=
X

i∈[N]
⟨Xi, µi⟩−
α
2b P

k′ XSk′

X

i∈S
⟨Xi, µi⟩

=
X

i∈[N]
⟨Xi, µi⟩−
α
2b P
k′ XSk′

X

i∈S

X

k
XSkµik

≥
X

i∈[N]
⟨Xi, µi⟩−
α
2 P

k′ XSk′

X

i∈S

X

k
XSk

= sw(X, µ) −α|S|

2

≥sw(X, µ) −αn

2 .

This implies that
sw(X, µ) −sw(X′, µ) ≤αn

2 .

Lemma 7. Let X be an envy-free fractional allocation for µ and let G = create-slack-graph(µ, X, α)
with edge set E. Let C = find-special-cycle(G, S(X)) and suppose there exists a node u such
that for i, i′ ∈V (C), (u, i) ∈E and (u, i′) /∈E. Let X′ = cycle-shift(C, X) and let G′ =
create-slack-graph(µ, X′, α/2) with edge set E′. Then |E′| < |E|.

proof. We first show that if an edge e /∈E, then e /∈|E′|. Suppose that i ∈V (C), i′ ∈N, and edge
(i, i′) /∈E. Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩= 1

2⟨Xi, µi⟩+ 1

2⟨Xnext(i), µi⟩−⟨Xi′, µi⟩

≥1

2⟨Xi, µi⟩+ 1

2⟨Xi′, µi⟩−⟨Xi′, µi⟩

= 1

2(⟨Xi, µi⟩−⟨Xi′, µi⟩)
(33)

≥α

2 .

Now, suppose that i, i′ /∈V (C) and edge (i, i′) /∈E. Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩= ⟨Xi, µi⟩−⟨Xi′, µi⟩≥α.
(34)

Finally, suppose that i ∈V (C), i′ ∈N\V (C), and edge (i′, i) /∈E. Then

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩= ⟨Xi′, µi′⟩−1

2⟨Xi, µi′⟩−1

2⟨Xnext(i), µi′⟩

= 1

2(⟨Xi′, µi′⟩−⟨Xi, µi′⟩) + 1

2(⟨Xi′, µi′⟩−⟨Xnext(i), µi′⟩)
(35)

≥1

2(α) + 1

2(0)

= α

2 .

We have shown that if e ∈E′, then e ∈E. Now we will show that there exists at least one edge e
such that e ∈E, but e /∈E′. Consider the node u described in the lemma statement, and let i ∈V (C)
be some node such that (u, i) ∈E but (u, next(i)) /∈E. Suppose that u /∈V (C). Then

⟨X′
u, µu⟩−⟨X′
i, µu⟩= ⟨Xu, µu⟩−1

2⟨Xi, µu⟩−1

2⟨Xnext(i), µu⟩

= 1

2(⟨Xu, µu⟩−⟨Xi, µu⟩) + 1

2(⟨Xu, µu⟩−⟨Xnext(i), µu⟩)

27


---Page Break---
≥1

2(α) + 1

2(0)

= α

2 .

Suppose that u ∈V (C). Then

⟨X′
u, µu⟩−⟨X′
i, µu⟩= 1

2⟨Xu, µu⟩+ 1

2⟨Xnext(u), µu⟩−1

2⟨Xi, µu⟩−1

2⟨Xnext(i), µu⟩

= 1

2(⟨Xu, µu⟩−⟨Xi, µu⟩) + 1

2(⟨Xnext(u), µu⟩−⟨Xnext(i), µu⟩)

≥1

2(α) + 1

2(0)

= α

2 .

This implies that (u, i) /∈E′, as desired.

Lemma 8. Let X be an envy-free fractional allocation for µ and let G = create-slack-graph(µ, X, α)
with edge set E. Let C = find-special-cycle(G, S(X)) and let X′ = cycle-shift(C, X). Then X′ is
an envy-free allocation for µ.

proof. Let G′ = create-slack-graph(µ, X′, α/2) with edge set E′. It suffices to show that we ≥0
for all e ∈E′. By Lemma 7, if an edge (i, i′) /∈E, then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩≥α

2 ≥0.

Recall that by assumption, for any i, i′ such that (i, i′) ∈E, ⟨Xi, µi⟩−⟨Xi′, µi⟩≥0. Suppose that
i ∈V (C), i′ ∈N, and (i, i′) ∈E. Then by Equation (33),

⟨X′
i, µi⟩−⟨X′
i′, µi⟩≥1

2(⟨Xi, µi⟩−⟨Xi′, µi⟩) ≥0.

Suppose instead that i, i′ /∈V (C) and edge (i, i′) ∈E. Then by Equation (34),

⟨X′
i, µi⟩−⟨X′
i′, µi⟩= ⟨Xi, µi⟩−⟨Xi′, µi⟩≥0.

Finally, suppose that i ∈V (C), i′ ∈N\V (C), and edge (i′, i) ∈E. Then by Equation (35),

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩= 1

2(⟨Xi′, µi′⟩−⟨Xi, µi′⟩) + 1

2(⟨Xi′, µi′⟩−⟨Xnext(i), µi′⟩)

≥1

2(0) + 1

2(0)

= 0.

Lemma 9. Let X be an envy-free fractional allocation for µ and let G = create-slack-graph(µ, X, α)
with edge set E. Let C = find-special-cycle(G, S(X)) and let X′ = cycle-shift(C, X). Then

sw(X, µ) −sw(X′, µ) ≤nα

2 .

proof. Observe that

sw(X′, µ) =
X

i∈V (C)
⟨X′
i, µi⟩+
X

i∈N\V (C)
⟨X′
i, µi⟩

= 1

2 ·
X

i∈V (C)
(⟨Xi, µi⟩+ ⟨Xnext(i), µi⟩) +
X

i∈N\V (C)
⟨Xi, µi⟩

≥1

2 ·
X

i∈V (C)
(⟨Xi, µi⟩+ ⟨Xi, µi⟩−α) +
X

i∈N\V (C)
⟨Xi, µi⟩

28


---Page Break---
=
X

i∈V (C)


⟨Xi, µi⟩−α

2


+
X

i∈N\V (C)
⟨Xi, µi⟩

≥
X

i∈V (C)


⟨Xi, µi⟩−α

2


+
X

i∈N\V (C)


⟨Xi, µi⟩−α

2



=
X

i∈N
⟨Xi, µi⟩−nα

2

This implies that

sw(X, µ) −sw(X′, µ) ≤nα

2 .

Lemma 10. Let X be an envy-free fractional allocation for µ. Let G = create-slack-graph(µ, X, α)
with edge set E, and let Q be a clique in G.
Let X′ = average-clique(Q, X) and G′ =
create-slack-graph(µ, X′, α/n), with E′ the edge set of G′. Then |E′| ≤|E|.

proof. It suffices to show that if an edge e /∈E, then e /∈|E′|. If i, i′ ∈Q, then edge (i, i′) must be
in E, as Q is a clique. Suppose that i, i′ ∈N\Q and (i, i′) ̸∈E. Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩= ⟨Xi, µi⟩−⟨Xi′, µi⟩≥α.

Now, suppose that i ∈Q, i′ ∈N\Q, and (i, i′) /∈E. Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩=
1
|Q|



X

i′′∈Q
⟨Xi′′, µi⟩



−⟨Xi′, µi⟩

≥⟨Xi, µi⟩−α
|Q| −1

|Q|


−⟨Xi′, µi⟩

≥α −α
|Q| −1

|Q|



≥α

n.

Finally, suppose that i ∈Q, i′ ∈N\Q, and (i′, i) /∈E. Then

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩= ⟨Xi′, µi′⟩−1

|Q|



X

i′′∈Q
⟨Xi′′, µi′⟩





=
1
|Q|



X

i′′∈Q
⟨Xi′, µi′⟩−⟨Xi′′, µi′⟩





=
1
|Q|



⟨Xi′, µi′⟩−⟨Xi, µi′⟩+
X

{i′′∈Q:i′′̸=i}
⟨Xi′, µi′⟩−⟨Xi′′, µi′⟩





≥1

n(α + 0)

≥α

n.

Lemma 11. Let X be an envy-free fractional allocation for µ. Let G = create-slack-graph(µ, X, α)
with edge set E, and let Q be a clique in G. Let X′ = average-clique(Q, X). Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩≥−α
∀i, i′ ∈N.

29


---Page Break---
proof. First, suppose i′ ∈N\Q and i ∈N. Then

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩≥min
i′′∈N⟨Xi′, µi′⟩−⟨Xi′′, µi′⟩≥0.

Suppose instead that i, i′ ∈Q. Then

⟨X′
i′, µi′⟩−⟨X′
i, µi′⟩= 0.

Finally, suppose that i ∈Q, i′ ∈N\Q. Then

⟨X′
i, µi⟩−⟨X′
i′, µi⟩=
1
|Q|



X

i′′∈Q
⟨Xi′′, µi⟩



−⟨Xi′, µi⟩

≥⟨Xi, µi⟩−α
|Q| −1

|Q|


−⟨Xi′, µi⟩

≥0 −α
|Q| −1

|Q|



≥−α.

Lemma 12. Let G = create-slack-graph(µ, X, α) with edge set E, and let Q be a clique in G. Let
X′ = average-clique(Q, X). Then

sw(X, µ) −sw(X′, µ) ≤n · α.

proof. Observe that

sw(X′, µ) =
X

i∈Q
⟨X′
i, µi⟩+
X

i∈N\Q
⟨X′
i, µi⟩

≥
X

i∈Q
(⟨Xi, µi⟩−α) +
X

i∈N\Q
⟨Xi, µi⟩

≥
X

i∈N
(⟨Xi, µi⟩−α)

≥sw(X, µ) −n · α.

Rearranging, this implies that

sw(X, µ) −sw(X′, µ) ≤n · α.

Lemma 13. Let G
=
create-slack-graph(µ, X, 0) with edge set E,
and let X′
=
remove-envy(µ, X).
Further let G′ = create-slack-graph(µ, X′, 0) with edge set E′.
Then
|E′| < |E|.

proof. First, we show that if an edge e /∈E, then e /∈E′. In other words, we will show that no
new edges with positive envy (negative weight) are added. Observe that within the while loop,
remove-envy distributes βi∗from a set U to N\U. If at the beginning of the while loop ∃i ∈U such
that wi,i′ < 0 for some i′ ∈N\U, then βi∗= 0 and i′ is added to U. Otherwise, by definition βi∗
is at most the minimum fraction that the set U needs to give away in order to create a new 0 envy
edge between any node in U and a node i′ ∈N\U. Therefore, distribute-equally cannot create a
new edge with positive envy by our choice of βi∗, which implies that no new edge with positive envy
could have been created by the end of the while loop. Now, suppose that w(u,v) < 0 at the end of the
while loop. Then there is a cycle C in Gr containing (u, v). Then for every i, ⟨X′
i, µi⟩≥⟨Xr
i , µi⟩.
The total set of allocations has not changed, so no positive envy is introduced.

Now, we show that there exists an edge e ∈E such that e /∈E′. That is, we show that we have
removed some edge with positive envy. If w(u,v) = 0 at the end of the while loop, then (u, v) ∈E

30


---Page Break---
and (u, v) ̸∈E′ so we are done. Otherwise, there is a cycle in Gr containing (u, v). As the overall
set of allocations has not changed and ⟨Xv, µu⟩−⟨Xu, µu⟩> 0, we must have
i ∈V (C) s.t. ⟨X′
i, µu⟩−⟨X′
u, µu⟩< 0
 <
i ∈V (C) s.t. ⟨Xi, µu⟩−⟨Xu, µu⟩< 0
.

Therefore, there exists some edge e = (u, i) for i ∈V (C) such that e ∈E but e /∈E′.

Lemma 14. Let G = create-slack-graph(µ, X, 0) with edge set E. Suppose that

−
α
4bn3 ≤⟨Xi, µi⟩−⟨Xi′, µi⟩
∀i, i′ ∈N.

Let X0 = X and define Xℓ= remove-envy(µ, Xℓ−1). Then for all i, i′, ℓ,

−α

4n2 ≤⟨Xℓ
i , µi′⟩−⟨Xℓ−1
i
, µi′⟩≤
α
4n3 .

proof. We first prove by induction on ℓthat for all ℓ,

−
α
4bn3 ≤⟨Xℓ
i , µi⟩−⟨Xℓ
i′, µi⟩
∀i, i′ ∈N.

The base case is true by assumption. Now consider any ℓ. By the inductive hypothesis, we have that
for all i, i′ ∈N, −
α
4bn3 ≤⟨Xℓ−1
i
, µi⟩−⟨Xℓ−1
i′
, µi⟩. Suppose for contradiction that there exists some
i, i′ ∈N such that −
α
4bn3 > ⟨Xℓ
i , µi⟩−⟨Xℓ
i′, µi⟩. This means that the envy of i for i′ must have
increased to more than during
α
4bn3 in the ℓth call to remove-envy. The only way for the envy of i for
i′ to increase in remove-envy is if i ∈U, i′ ∈N\U, and βi∗> 0. If ⟨Xℓ−1
i
, µi⟩−⟨Xℓ−1
i′
, µi⟩≤0,
then βi∗= 0. Therefore, we must have ⟨Xℓ−1
i
, µi⟩−⟨Xℓ−1
i′
, µi⟩> 0. However, by our choice of
βi∗, i′ must then be added to U before i becomes envious of i′. Therefore, it is not possible for the
envy of i for i′ to have increased to more than
α
4bn3 in the ℓth call to remove-envy.

We now prove the main lemma. Consider any ℓ. One stopping condition of the while loop in
remove-envy is when w(u,v) = ⟨Xr
u, µu⟩−⟨Xr
v, µu⟩= 0. Observe that v ∈U and u ∈N\U for all
iterations r in the while loop. This implies that u’s allocation is always increasing, while v’s allocation
is always decreasing. The increase in u’s utility is therefore at most u’s current envy towards v’s
allocation in Xℓ−1
i
, which is upper bounded by
α
4bn3 by the induction proof above. Formally,

⟨Xℓ
u, µu⟩−⟨Xℓ−1
u
, µu⟩≤⟨Xℓ−1
v
, µu⟩−⟨Xℓ−1
u
, µu⟩≤
α
4bn3 .
(36)

Furthermore, over the course of remove-envy, the allocation of node u is increased at least as much
as that of any other node i, i.e.

(Xℓ
uk −Xℓ−1
uk ) ≥(Xℓ
ik −Xℓ−1
ik )
∀i ∈N, k ∈[m].

This is because u is always a member of N\U. Therefore, for any i, i′ ∈N,

⟨Xℓ
i , µi′⟩−⟨Xℓ−1
i
, µi′⟩≤b ·
α
4bn3 =
α
4n3 ,

as b is the largest possible utility ratio between two nodes.

Again by applying Equation (36) and because b is the largest possible utility ratio between two nodes,
for any i, i′ ∈[N], the utility of i′ for the allocation transferred from i to u is at most b ·
α
4bn3 =
α
4n3 .
Node i could have transferred to at most n nodes during remove-envy, which implies that node i′
has utility of at most n ·
α
4n3 =
α
4n2 for all of the allocation transferred away from node i during
remove-envy. Therefore,

⟨Xℓ
i , µi′⟩−⟨Xℓ−1
i
, µi′⟩≥−α

4n2
∀i ∈N.

Lemma 15. Let G = create-slack-graph(µ, X, 0) with edge set E. Suppose that

−
α
4bn3 ≤⟨Xi, µi⟩−⟨Xi′, µi⟩
∀i, i′ ∈N.

Let X0 = X and define Xℓ= remove-envy(µ, Xℓ−1). Then for all i, i′ and for all ℓ≤n2,

sw(X0, µ) −sw(Xℓ, µ) ≤nα

4 .

31


---Page Break---
proof. Consider some ℓ. By Lemma 14 we know that

⟨Xℓ
i , µi⟩−⟨Xℓ−1
i
, µi⟩≥−α

4n2
∀i ∈N.

Applying the above once for each node, we obtain

sw(Xℓ, µ) =
X

i∈N
⟨Xℓ
i , µi⟩

≥
X

i∈N


⟨Xℓ
i , µi⟩−α

4n2



≥sw(Xℓ−1, µ) −n · α

4n2 .

Rearranging, this implies that

sw(Xℓ−1, µ) −sw(Xℓ, µ) ≤α

4n.

Because ℓ≤n2, we therefore must have

sw(X0, µ) −sw(Xℓ, µ) ≤nα

4 .

Lemma 16. Let G = create-slack-graph(µ, X, α) with edge set E. Suppose that for S ∈S(X), if
δ-(S) /∈∅, then δ+(S) /∈∅. Further suppose that

−
α
4bn3 ≤⟨Xi, µi⟩−⟨Xi′, µi⟩
∀i, i′ ∈N

Let X0 = X and define Xℓ= remove-envy(µ, Xℓ−1).
Finally, for ℓ≤n2 let G′ =
create-slack-graph(µ, Xℓ, α

2 ) with edge set E′. Then |E′| ≤|E|.

proof. It suffices to show that if an edge (i, i′) /∈E, then (i, i′) /∈E′. By Lemma 14, in each call to
remove-envy, the utility of i for the allocation of i decreases by at most
α
4n2 . Therefore,

⟨Xℓ
i , µi⟩−⟨X0
i , µi⟩≥ℓ· −α

4n2 ≥−α

4 .

Again by Lemma 14, in each call to remove-envy, the utility of i for the allocation of i′ increases by
at most
α
4n3 . Therefore,

⟨Xℓ
i′, µi⟩−⟨X0
i′, µi⟩≤ℓ· α

4n3 ≤α

4n.

Putting both equations together, we have

⟨Xℓ
i , µi⟩−⟨Xℓ
i′, µi⟩≥⟨X0
i , µi⟩−α

4 −

⟨X0
i′, µi⟩+ α

4n



= ⟨X0
i , µi⟩−⟨X0
i′, µi⟩−α

4 −α

4n

≥α −α

2

= α

2
as desired.

Lemma 17. Let X′ = remove-envy(µ, X). Then |S(X′)| ≤|S(X)|.

proof. It suffices to show that no equivalence class S ∈S(X) becomes smaller (i.e. strictly loses
members) during remove-envy. First, suppose for contradiction that S first becomes smaller during
the while loop during iteration r. Then in iteration r, it must be the case that S ∩U ̸= ∅and
S ∩(N\U) ̸= ∅, otherwise the allocation of each member of S would have changed by the same
amount. Furthermore, it must be the case that in iteration r, βi∗> 0, otherwise no allocation would
have been transferred. In iteration r, consider some i ∈(S ∩U) and i′ ∈(S ∩(N\U)). Then βi = 0,

32


---Page Break---
as i begins iteration r with zero envy towards i′. Because βi∗≤mini βi, this means βi∗≤0, which
is a contradiction . Therefore, no equivalence class S becomes smaller during the while loop. To
conclude the proof, we observe that in the cycle elimination step after the while loop, the total set of
allocations remains the same, which implies that the set of equivalence classes remains the same as
well.

We are finally ready to prove Lemma 1.

Proof of Lemma 1. We first prove by induction that every iteration r starts with an envy-free allocation
Xr. The base case is satisfied because X0 = Y µ, and Y µ is envy-free by definition. Now, suppose
that the inductive hypothesis holds for all iterations up to and including r. We will show that iteration
r + 1 starts with an envy-free allocation. If Xr+1 = Xr, then we can directly invoke the inductive
hypothesis for iteration r. Otherwise, suppose that remove-incoming-edge was called in iteration r of
Algorithm 3. Then by Lemma 5, iteration r + 1 starts with an envy-free allocation. Suppose instead
that cycle-shift was called in iteration r. Then by Lemma 8, iteration r + 1 starts with an envy-free
allocation. Finally, suppose that average-clique was called in iteration r. Then the remove-envy is
called repeatedly as long as there exists an edge with negative weight in E(G′). By Lemma 13, each
call to remove-envy removes an edge with negative weight from E(G′), and adds no new edges with
negative weight. There are a finite number of edges in E(G′), so this loop terminates. Therefore,
iteration r + 1 starts with an envy-free allocation.

Next, we prove that in every iteration, either an edge is removed from the envy-with-slack graph, or
the number of equivalence classes decreases. Formally, for two iterations r and r + 1, we prove that
either

1. |E(Gr)| > |E(Gr+1)| or

2. |E(Gr)| ≥|E(Gr+1)| and |S(Xr)| > |S(Xr+1)|.

If remove-incoming-edge is called in iteration r of Algorithm 3, then by Lemma 4 we have |E(Gr)| >
|E(Gr+1)|. If cycle-shift is called in iteration r, then by Lemma 7 we have |E(Gr)| > |E(Gr+1)|.
If ∃e ∈E(Gr) s.t. we ≥
α
4bn4 , then e ∈E(Gr) but e /∈E(Gr+1). Furthermore, if e′ /∈E(Gr),
then e′ /∈E(Gr+1) as Xr = Xr+1. Therefore, we have |E(Gr)| > |E(Gr+1)|.

Finally, suppose average-clique is called in iteration r on clique Q.
Recall that Gavg
=
create-slack-graph(µ, average-clique(Q, Xr), αavg).
By Lemma 10, we know that |E(Gr)| ≥
|E(Gavg)|. The number of edges in E(Gavg) is at most n2, so remove-envy will be called at most n2

times by Lemma 13. Because average-clique was called, we know that we <
αr
4bn4 ∀e ∈E(Gr). By
Lemma 11 with α =
αr
4bn4 ,

⟨Xavg
i
, µi⟩−⟨Xavg
i′ , µi⟩≥−αr

4bn4 = −αavg

4bn3
∀i, i′ ∈N.

Therefore, we can apply Lemma 16 with α = αavg to conclude that |E(Gr+1)| ≤|E(Gavg)| ≤
|E(Gr)|. Finally, observe that because C included members of at least two equivalence classes,
the operation average-clique strictly decreased the number of equivalence classes. By Lemma 17,
operation remove-envy does not increase the number of equivalence classes. Therefore, |S(Xr)| >
|S(Xr+1)|.

We now prove that the algorithm terminates with an Xr which satisfies Proposition 2. Each iteration
either removes an edge or merges two equivalence classes. Because the maximum number of edges
is n2 and the number of equivalence classes is n, Algorithm 3 terminates in at most n3 iterations. We
need to show that Algorithm 3 terminates with αr ≥γ. If an iteration r does not call remove-envy,
then an edge is removed and αr+1 ≥
αr
4bn4 . There can be at most n2 such iterations. If an iteration
r does call remove-envy, then αr+1 =
αr
2n. There can be at most n such iterations which call
remove-envy between every iteration which does not call remove-envy, for a total of at most n3
iterations. Therefore,

αr ≥
α0
(4bn4)n2 · (2n)n3 =
α0
en2 log(4bn4)+n3 log(2n) .

Choosing α0 = γ · (en2 log(4bn4)+n3 log(2n)) thus implies that αr ≥γ for every iteration r if
γ · (en2 log(4bn4)+n3 log(2n)) ≤1. Therefore, we set γ0 = e−n2 log(4bn4)−n3 log(2n).

33


---Page Break---
Finally, we need to show that Algorithm 3 does not significantly decrease the social welfare. By
Lemmas 6, 9 and 12, we know that each of the operations remove-incoming-edge, cycle-shift, and
average-clique change the social welfare by at most O(γ). Each of these operations is called at most
n3 times. Each of remove-incoming-edge and cycle-shift are called at most once for each edge, or at
most n2 times. Operation average-clique is called at most n times for each edge, or at most n3 times.
Finally, Lemma 15 bounds the total social welfare loss from all calls to remove-envy between any
two calls to average-clique by O(γ). Therefore, sw(Y µ, µ) −sw(Xr, µ) = O(γ), as desired.

34


---Page Break---
G
Proof of Theorem 2

proof. Let a = 1, b = 3, n = 2, and m = 2, and assume that the distributions of values are all
normal distributions with variance 1. For n = 2, envy-freeness and proportionality are equivalent,
and therefore we will focus on the former. Define ϵ = T −1/3. We will prove the desired result by
contradiction. Assume there exists an algorithm ALG such that for any µ∗∈[a, b]2×2, the probability
that the algorithm satisfies the envy-freeness constraints and has regret of less than
T 2/3
log(T ) is at least
1 −1/T.

Consider the following two matrices.

µ1 =

2
3
1
1



µ2 =

2
3
1
1 + ϵ



Define P1 as the distribution of the full history HT when using algorithm ALG when µ∗= µ1.
Define P2 as the equivalent distribution when µ∗= µ2.

We will proceed according to the following proof sketch. First, we will show that if µ∗= µ1, then
ALG with constant probability allocates ˜O(T 2/3) items of type 2 to player 2 (Lemma 18). We next
upper bound the total variation distance between P1 and P2. When two distributions are sufficiently
close in total variation distance, then an event that has constant probability under one distribution also
has constant probability under the other distribution. Therefore, Lemma 18 and the closeness in TV
distance of P1 and P2 together imply that if µ∗= µ2, then ALG with constant probability allocates
˜O(T 2/3) items of type 2 to player 2. When ALG allocates ˜O(T 2/3) items of type 2 to player 2,
then ALG cannot satisfy the envy-freeness constraints for µ2. Therefore, the previous two sentences
together imply that with constant probability, ALG will not satisfy the envy-freeness constraints for
all t when µ∗= µ2, which is a contradiction.

Lemma 18. Using the notation defined above,

EP1[N T
22] ≤T 2/3

and

Pr
P1

 T −1
X

t=0
Xt
22 >
T 2/3

log(T)

!

< 1/8.
(37)

Intuitively, both equations in Lemma 18 bound how many times an item of type 2 is allocated to
player 2. The first equation bounds in expectation while the second equation bounds elements of the
fractional allocations Xt. The proof of Lemma 18 is given in Appendix G.1.

Lemma 19. Using the notation from above,

KL(P1, P2) = EP1[N T
22] · KL

N(1, 1), N(1 + ϵ, 1)


proof. Define f 1
ik as the probability density function (pdf) of N((µ1)ik, 1) and f 2
ik as the pdf of
N((µ2)ik, 1). We let fN(µ,σ2) be the pdf of a normal distribution with mean µ and variance σ2.

For any history HT = {(kt, it, vt)}T −1
t=0 , recall that Xt is the fractional allocation chosen by ALG at
time t given history Ht. Then we have that for any HT

P1(HT ) =

T −1
Y

t=0

 1

mXt
itktf 1
itkt(vt)

,

The 1/m term comes from item t having a 1/m probability of being of type kt. The Xt
itkt is the
probability that ALG allocates the item of type kt to player it at time t, and finally f 1
itkt(vt) is the
probability of seeing value vt given that the item of type kt was allocated to player it. Similarly, we
have that

P2(HT ) =

T −1
Y

t=0

 1

mXt
itktf 2
itkt(vt)

.

35


---Page Break---
Therefore, we have that

KL(P1, P2) = EHT ∼P1


log
P1(HT )

P2(HT )



= EHT ∼P1

"

log

 QT −1
t=0
  1

mXt
itktf 1
itkt(vt)


QT −1
t=0
  1

mXt
itktf 2
itkt(vt)


!#

= EHT ∼P1

"

log

 QT −1
t=0
 
f 1
itkt(vt)


QT −1
t=0
 
f 2
itkt(vt)


!#

= EHT ∼P1

"

log

 T −1
Y

t=0

f 1
itkt(vt)
f 2
itkt(vt)

!#

= EHT ∼P1



log




Y

t:(it,kt)=(2,2)

fN(1,1)(vt)
fN(1+ϵ,1)(vt)









= EHT ∼P1




X

t:(it,kt)=(2,2)
log
 fN(1,1)(vt)

fN(1+ϵ,1)(vt)





=

T −1
X

t=0
EHT ∼P1


1(it,kt)=(2,2) log
 fN(1,1)(vt)

fN(1+ϵ,1)(vt)



=

T −1
X

t=0
EHT ∼P1


1(it,kt)=(2,2) E

log
 fN(1,1)(vt)

fN(1+ϵ,1)(vt)

1(it,kt)=(2,2)



=

T −1
X

t=0
EHT ∼P1


1(it,kt)=(2,2) Ev∼N(1,1)


log
 fN(1,1)(v)

fN(1+ϵ,1)(v)



=

T −1
X

t=0
EHT ∼P1

1(it,kt)=(2,2) · KL (N(1, 1), N(1 + ϵ, 1))


= KL (N(1, 1), N(1 + ϵ, 1)) EHT ∼P1

"T −1
X

t=0
1(it,kt)=(2,2)

#

= KL (N(1, 1), N(1 + ϵ, 1)) EHT ∼P1

N T
22

.

We can use Lemma 18 and Lemma 19 to bound the KL-divergence between P1 and P2 by

KL(P1, P2) = EP1[N T
22] · KL

N(1, 1), N(1 + ϵ, 1)

= EP1[N T
22]ϵ2

2
≤1

2.
(38)

We next need the following result from probability theory that is a consequence of the Bretagnolle-
Huber inequality.

Lemma 20. For any two probability distributions p and q defined on the same space and for any
measurable event F in this space,

p(F C) + q(F) ≥1

2e−KL(p,q).

proof. For any probability distributions p and q, we have by the Bretagnolle-Huber inequality that

dTV(p, q) ≤1 −1

2e−KL(p,q).

36


---Page Break---
For any event F,

dTV(p, q) ≥|p(F) −q(F)| ≥p(F) −q(F) = 1 −p(F C) −q(F).

Combining the above equations gives that

p(F C) + q(F) ≥1

2e−KL(p,q).
(39)

Taking p = P1, q = P2 and F =
nPT −1
t=0 Xt
22 ≤
T 2/3
log(T )
o
in Lemma 20, we have by Equation (38)
that

Pr
P1

 T −1
X

t=0
Xt
22 >
T 2/3

log(T)

!

+ Pr
P2

 T −1
X

t=0
Xt
22 ≤
T 2/3

log(T)

!

≥1

2e−KL(P1,P2) ≥1/4.
(40)

Combining Equation (37) with Equation (40) gives

Pr
P2

 T −1
X

t=0
Xt
22 ≤
T 2/3

log(T)

!

≥1/8.
(41)

If PT −1
t=0 Xt
22 ≤
T 2/3
log(T ), then there must exist some time t such that Xt
22 ≤T −1/3

log(T ). If Xt
22 ≤T −1/3

log(T ),
then player 2’s envy in expectation at time t for player 1 under µ2 (for sufficiently large T) is

Xt
11 · 1 + Xt
12(1 + ϵ) −Xt
21 · 1 −Xt
22(1 + ϵ) = (1 −Xt
21) · 1 + (1 −Xt
22)(1 + ϵ) −Xt
21 · 1 −Xt
22(1 + ϵ)

= 2 + ϵ −2Xt
21 −2Xt
22(1 + ϵ)

≥ϵ −2Xt
22(1 + ϵ)

≥ϵ −2T −1/3(1 + ϵ)

log(T)

= T −1/3 −2T −1/3(1 + T −1/3)

log(T)
> 0,

implying that Xt does not satisfy the envy-freeness in expectation constraints under µ2. Therefore,
Equation (41) implies that

Pr
P2(EFE for µ2 not satisfied) ≥Pr
P2

 T −1
X

t=0
Xt
22 ≤
T 2/3

log(T)

!

≥1/8.
(42)

This contradicts the assumption that ALG satisfies the envy-freeness in expectation constraints for
µ2 with probability at least 1 −1/T when µ∗= µ2.

G.1
Proof of Lemma 18
proof. Define E as the event that the algorithm ALG satisfies the envy-free in expectation constraints
for µ1 and has regret less than
T 2/3
log(T ) for µ∗= µ1. By assumption, PrP1(E) ≥1 −1/T.

If µ∗= µ1, then the social welfare maximizing envy-free allocation is

Y µ1 =

0
1
1
0


.

Furthermore, a fractional allocation Xt satisfies envy-freeness in expectation for µ1 only if

Xt
21 + Xt
22 ≥1.
(43)

Therefore, for any Xt that satisfies envy-freeness in expectation for µ1, the regret at time t is

⟨Y µ1, µ∗⟩F −⟨Xt, µ∗⟩F = 4 −(2Xt
11 + 3Xt
12 + Xt
21 + Xt
22)

37


---Page Break---
= 4 −(2(1 −Xt
21) + 3(1 −Xt
22) + Xt
21 + Xt
22)

= −1 + Xt
21 + 2Xt
22
≥Xt
22.
[Equation (43)]

This implies that the regret when µ∗= µ1 of ALG when ALG satisfies envy-freeness in expectation
for µ1 is

T · ⟨Y µ1, µ∗⟩F −

T −1
X

t=0
⟨Xt, µ∗⟩F ≥

T −1
X

t=0
Xt
22.

By definition, under event E, the regret of ALG for µ∗= µ1 is at most
T 2/3
log(T ) and ALG satisfies the
envy-freeness in expectation constraints for µ1. Therefore, the previous equation implies that under
event E,
T −1
X

t=0
Xt
22 ≤
T 2/3

log(T).
(44)

Recall that N T
22 is the number of times item of type 2 is allocated to player 2. Equation (44) implies
that

EP1[N T
22 | E] = EP1

"T −1
X

t=0
Xt
22 | E

#

≤
T 2/3

log(T).

This implies that for sufficiently large T,

EP1[N T
22] = EP1[N T
22 | E] Pr
P1(E) + EP1[N T
22|¬E] Pr
P1(¬E)

≤EP1[N T
22 | E] + T · 1

T

≤
T 2/3

log(T) + 1

≤T 2/3.
(45)

Equation (44) also implies that for sufficiently large T,

Pr
P1

 T −1
X

t=0
Xt
22 ≤
T 2/3

log(T)

!

≥Pr
P1(E) ≥1 −1

T > 7/8.
(46)

Taking the complement of this equation proves Equation (37).

38


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Our main claim is a explore-then-commit algorithm which achieves ˜O(T 2/3)
regret while maintaining envy-freeness or proportionality in expectation. In our body, we
provide an algorithm that does so and a proof sketch, with the full proof in Appendix D.

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

Justification: The limitations of our results are discussed in the discussion section of the
paper.

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

39


---Page Break---
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: All assumptions for the main theoretical results are clearly stated in the model
section and the theorem statements. Proof sketches of the main results appear in the body,
while formal proofs are included in the appendix.

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

Justification: This paper does not include any experiments.

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

40


---Page Break---
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

Answer: [NA]

Justification: This paper does not include any experiments requiring code.

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

Justification: This paper does not include any experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.

41


---Page Break---
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper does not include any experiments.

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

Answer: [NA]

Justification: This paper does not include any experiments.

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

42


---Page Break---
Answer: [Yes]

Justification: All conditions in the code of ethics are met.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses positive societal impacts in applications like food alloca-
tion. There are no (plausible) negative societal impacts.

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

Justification: This paper does not include any datasets or trained models that can be misused.

Guidelines:

• The answer NA means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring

43


---Page Break---
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

Justification: This paper does not use any existing assets.

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

Justification: This paper does not release any new assets.

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

44


---Page Break---
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or human subjects.

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

Justification: This paper does not involve crowdsourcing or human subjects.

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

45


---Page Break---
