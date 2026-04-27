Efficient Graph Matching
for Correlated Stochastic Block Models

Shuwen Chai
Northwestern University
Evanston, IL 60208
shuwenchai2027@u.northwestern.edu

Miklós Z. Rácz
Northwestern University
Evanston, IL 60208
miklos.racz@northwestern.edu

Abstract

We study learning problems on correlated stochastic block models with two bal-
anced communities. Our main result gives the first efficient algorithm for graph
matching in this setting. In the most interesting regime where the average degree is
logarithmic in the number of vertices, this algorithm correctly matches all but a
vanishing fraction of vertices with high probability, whenever the edge correlation
parameter s satisfies s2 > α ≈0.338, where α is Otter’s tree-counting constant.
Moreover, we extend this to an efficient algorithm for exact graph matching when-
ever this is information-theoretically possible, positively resolving an open problem
of Rácz and Sridhar (NeurIPS 2021). Our algorithm generalizes the recent break-
through work of Mao, Wu, Xu, and Yu (STOC 2023), which is based on centered
subgraph counts of a large family of trees termed chandeliers. A major technical
challenge that we overcome is dealing with the additional estimation errors that
are necessarily present due to the fact that, in relevant parameter regimes, the
latent community partition cannot be exactly recovered from a single graph. As
an application of our results, we give an efficient algorithm for exact commu-
nity recovery using multiple correlated graphs in parameter regimes where it is
information-theoretically impossible to do so using just a single graph.

1
Introduction

The proliferation of network data has highlighted the ubiquity and importance of graph matching in
machine learning, with applications in a variety of domains, including social networks [52, 57], com-
putational biology [61], and computer vision [13, 37]. While the graph matching task—recovering
the latent node alignment between two networks—is known to be NP-hard to solve or even ap-
proximate in general [39, 53], in practice it is often possible to solve it well, such as in the works
cited above. This has motivated an exciting recent line of work studying average-case graph match-
ing [14, 15, 65, 16, 23, 28, 24, 50, 5, 21, 22, 40, 41, 42, 17, 20, 18], focusing on correlated Erd˝os–
Rényi random graphs [57]. These papers culminated in recent breakthrough works which developed
efficient graph matching algorithms in the constant noise regime [41, 42].

However, real-world networks are not modeled well by Erd˝os–Rényi random graphs, which in turn
has motivated a growing line of recent work studying graph matching beyond Erd˝os–Rényi [8, 33, 11,
55, 59, 70, 58, 25, 63, 60, 19, 67, 66, 10]. In particular, an important problem in this vein is to study
graph matching in correlated stochastic block models (correlated SBMs) [36, 55, 35] (see Section 2
for definitions), since community structure is prevalent in many networks and the community recovery
problem is a fundamental inference task that is often a starting point for deeper analyses. Recent
work of Rácz and Sridhar [58] determined the fundamental information-theoretic limits for exact
graph matching in correlated SBMs; however, the underlying algorithm used to achieve this limit is
inefficient (that is, not polynomial time). Rácz and Sridhar [58] posed the open problem of finding an
efficient algorithm for (exact) graph matching whenever this is information-theoretically feasible.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Our main contribution positively resolves this open problem of Rácz and Sridhar [58], giving the
first efficient algorithm for graph matching for correlated SBMs with two balanced communities,
under a condition on the correlation strength that is conjectured to be necessary. Specifically, we give
an efficient algorithm that, in the most interesting regime where the average degree is logarithmic
in the number of vertices, achieves almost exact recovery of the latent matching, whenever the
edge correlation parameter s satisfies s2 > α ≈0.338, where α is Otter’s tree-counting constant.
Moreover, we extend this to an efficient algorithm for exact graph matching whenever this is
information-theoretically possible. See Section 3 and Theorem 1 for details.

In addition, our results on graph matching directly imply novel efficient algorithms and results for
community recovery. Specifically, combining—in a black-box fashion—our (exact) graph matching
algorithm with existing community recovery algorithms, we give an efficient algorithm for exact
community recovery using multiple correlated graphs in parameter regimes where it is information-
theoretically impossible to do so using just a single graph. See Section 3 and Theorem 2 for details.

Our algorithm generalizes the recent breakthrough work of Mao, Wu, Xu, and Yu [42], which is based
on centered subgraph counts of a large family of trees termed chandeliers, to the setting of correlated
SBMs. A major technical challenge that we overcome is dealing with the additional estimation errors
that are necessarily present due to the fact that, in relevant parameter regimes, the latent community
partition cannot be exactly recovered from a single graph, and thus the edge-indicator variables in
the centered subgraph counts cannot be precisely centered. Our technical contributions highlight the
interplay between graph matching and community recovery in ways that are complementary to the
recent work of Gaudio, Rácz, and Sridhar [25].

2
Models and problems

In this section we describe the setting of the paper by introducing the stochastic block model (SBM),
correlated SBMs, and the community recovery and graph matching tasks.

The stochastic block model (SBM) is the canonical probabilistic generative model for a network with
latent community structure. The SBM was first introduced by Holland, Laskey, and Leinhardt [30]
and has been widely studied over the past decades [1]. In general, a SBM may consist of a number
of communities, with distinct vertices connected randomly with a probability that depends on their
community memberships.

In this work, we focus on the simplest setting of the balanced two-community SBM. Given n ∈Z+
and p, q ∈[0, 1], we construct G ∼SBM(n, p, q) as follows. The graph G has n vertices, with vertex
labels given by V = [n] := {1, 2, . . . , n}. Let σ∗= {σ∗(i)}n
i=1 be the vector of community labels,
where each entry σ∗(i) ∈{−1, +1} is drawn independently and uniformly at random. Then, given
the community labels σ∗, for any pair of vertices i ̸= j ∈[n], edge (i, j) is in G with probability
p1{σ∗(i)=σ∗(j)} + q1{σ∗(i)̸=σ∗(j)}. That is, two different vertices are connected with probability p if
they are from the same community and connected with probability q otherwise.

Correlated SBMs are multiple SBMs where the corresponding edge variables are correlated [36,
55, 35]. Specifically, we construct two correlated SBMs (G1, G2) ∼CSBM(n, p, q, s) using a
natural subsampling procedure as follows. Let G ∼SBM(n, p, q) be a parent graph with community
labels σ∗. Next, given G, we construct G1 by random sampling of the edges: each edge of G is
included in G1 with probability s, independently of everything else, and non-edges of G remain
non-edges of G1. We then do the edge sampling independently again to obtain G′
2 in the same way.
The child graphs G1 and G′
2 inherit both the vertex labels (given by [n]) and the community labels σ∗
from the parent graph G. Finally, let π∗be a uniformly random permutation of [n] := {1, 2, . . . , n}
and generate G2 by relabeling the vertices of G′
2 according to π∗(e.g., vertex i in G′
2 is relabeled as
π∗(i) in G2). This last step reflects the fact that in practice often the correspondence between the two
vertex sets is unknown. We denote the adjacency matrices of G1 and G2 as A and B, respectively, and
note that the community labels of the two graphs are σA
∗:= σ∗and σB
∗:= σ∗◦π−1
∗, respectively.
See Figure 1 for an illustration.

Marginally, G1 and G2 are identically distributed SBMs: we have G1, G2 ∼SBM(n, ps, qs).
Moreover, G1 and G2 are correlated. Specifically, for every pair of distinct vertices {i, j}, the edge-
indicator random variables Ai,j and Bπ∗(i),π∗(j) are correlated Bernoulli random variables. A simple
calculation shows that if σ∗(i) = σ∗(j), then the correlation coefficient of Ai,j and Bπ∗(i),π∗(j) is

2


---Page Break---
2

1

6

3

4

5

7

12
11

10

8
9

2

1

6

3

4

5

7

12
11

10

8
9
2 (2)

1 (8)

6 (6)

3 (12)

4 (7)

5 (11)

7 (10)

12 (9)
11 (1)

10 (5)

8 (3)
9 (4)
G ∼SBM(n, p, q)

G1
G′
2 (G2)

π∗= (8, 2, 12, 7, 11, 6, 10, 3, 4, 5, 1, 9)

subsample
subsample

Figure 1: Schematic illustrating two-community correlated SBMs; see the text for details. (Figure
reproduced from [58] with permission.)

equal to ρ+ := s 1−p

1−ps, whereas if σ∗(i) ̸= σ∗(j), then this correlation coefficient is ρ−:= s 1−q

1−qs.
Our focus will be on the sparse setting where p, q = o(1) (as n →∞), in which case both ρ+ and
ρ−are asymptotically (1 −o(1))s, and hence we can regard s as the edge correlation parameter.

Community recovery. The goal of community recovery is to recover the latent community labels σ∗
given some (graph) data, such as a SBM G or correlated SBMs (G1, G2). There are various notions
of community recovery, depending on how close an estimate is to the ground truth σ∗. In this work,
we focus on exact community recovery, defined as follows: an estimator bσ achieves exact community
recovery if limn→∞P(| 1

n
Pn
i=1 bσ(i)σ∗(i)| = 1) = 1. The absolute value is present in the previous
expression since we can only hope to recover the community labels up to a global sign flip; in other
words, our goal is to recover the partition of the graph into two communities. A slightly weaker
notion, which also appears throughout our work, is almost exact community recovery, which holds if
limn→∞P(| 1

n
Pn
i=1 bσ(i)σ∗(i)| = 1 −o(1)) = 1; in other words, this notion tolerates a vanishing
fraction of errors. Further weaker notions include partial recovery and weak recovery; since these are
not the focus here, we refer to [1] for details.

Different parameter regimes give rise to different challenges and different notions of recovery become
most relevant. In the constant average degree regime, that is, when p = a

n and q =
b
n for some
constants a and b, it is impossible to recovery the communities exactly. Prior works [47, 49, 44]
have characterized the information-theoretic threshold and developed efficient algorithms for partial
recovery in this regime. On the other hand, if the vertices have polynomially growing degrees, that is,
when p = n−a+o(1) and q = n−b+o(1) for some constants a, b ∈[0, 1), then community recovery is
easy as long as lim infn→∞|pn/qn −1| > 0 (see [48]).

In this work, we focus on the “bottleneck regime” of logarithmic average degree, which is the bare
minimum for the graph to be connected, and which is when exact community recovery is most
interesting. In most of the paper we assume that p = a log n

n
and q = b log n

n
for some positive
constants a, b. For an SBM with two balanced communities and these parameters, there is a sharp
information-theoretic threshold for exact community recovery, which is given by D+(a, b) = 1, where
D+(a, b) := (√a −
√

b)2/2 (see [48, 2, 4, 1]). This quantity is known as the Chernoff–Hellinger
divergence in the general k-community SBM setting with linear size communities [4] and simplifies
to the above form in our setting. In other words, when D+(a, b) > 1, there exists an estimator bσ
that is computable in polynomial-time and which achieves exact community recovery with high
probability. On the other hand, when D+(a, b) < 1, exact community recovery is impossible, in the
sense that for all estimators bσ we have that limn→∞P(| 1

n
Pn
i=1 bσ(i)σ∗(i)| = 1) = 0. Moreover,
when it is possible to achieve exact community recovery on a single graph, several polynomial-time
algorithms have been studied by previous works (e.g., [48, 2]).

Community recovery and graph matching. What are the information-theoretic limits for exact
community recovery given two correlated SBMs (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s)? This
question was initiated and partially solved by Rácz and Sridhar [58], and subsequently fully solved
by Gaudio, Rácz, and Sridhar [26]. Without yet going into the details, these works highlight the
importance of graph matching, that is, the task of recovering the latent matching π∗given the two
correlated graphs (G1, G2). In brief, when π∗can be perfectly recovered from (G1, G2), then one can
take the union graph G1 ∨π∗G2 of G1 and G2, which is also an SBM, but with a larger edge density,

3


---Page Break---
which makes community recovery easier. This shows that exact community recovery is possible from
(G1, G2) even in parameter regimes where this is impossible from just a single graph G1 (see [58]).

Graph matching. Motivated by the above discussion, we now discuss average-case graph matching
and notions of recovery. In general, suppose that (G1, G2) are two correlated random graphs with
n vertices each and that π∗is the underlying latent vertex matching. The goal of graph matching
is to output an estimator bπ = bπ(G1, G2) that is close to π∗. There are various notions of recovery
depending on how close bπ is to π∗; the two most relevant notions are the following. We say that an
estimator bπ achieves exact graph matching if limn→∞P(bπ = π∗) = 1. We say that an estimator
achieves almost exact graph matching if with high probability there exists a subset I ⊆[n] with
|I| = (1 −o(1))n such that bπ|I = π∗|I, where π|I denotes the restriction of π to I. In words, almost
exact graph matching allows the estimator to make a vanishing fraction of errors.

The graph matching problem has been widely studied, with applications to computer vision [13, 37],
computational biology [61], and social networks [57]. In particular, de-anonymizing social networks
is possible with graph matching algorithms, which implies that anonymity is not equivalent to
privacy [52]. That said, studying the limits of graph matching algorithms—including potential
information-computation gaps, as we shall discuss—can help guide data regulators on when to take
more actions with regards to data protection, in addition to anonymity.

As discussed in the introductory paragraphs, there has been a large body of recent work on average-
case graph matching, both studying correlated Erd˝os–Rényi random graphs [57, 14, 15, 65, 16,
23, 28, 24, 50, 5, 21, 22, 40, 41, 42, 17, 20, 18] and more general models of correlated random
graphs [8, 33, 11, 55, 59, 70, 58, 25, 63, 60, 19, 67, 66, 10]. In particular, Rácz and Sridhar [58]
determined the fundamental information-theoretic limits for exact graph matching in correlated SBMs:
in the logarithmic average degree regime discussed above, this threshold is s2 a+b

2
= 1. However,
this result is information-theoretic, and the authors posed the open problem of finding an efficient
algorithm for exact graph matching, whenever this is information-theoretically possible.

Our main contribution positively resolves this open problem, giving the first efficient algorithm for
graph matching for correlated SBMs with two balanced communities. Our algorithm generalizes the
recent breakthrough work of Mao, Wu, Xu, and Yu [42] that developed an efficient graph matching
algorithm for correlated Erd˝os–Rényi graphs. As an application, our results imply novel efficient
algorithms and results for community recovery. We now turn to describing our results.

3
Main results: graph matching

Our main theorem for graph matching on correlated SBMs is that there exists a polynomial-time
algorithm that can achieve exact matching if s2 > α, where α is Otter’s tree counting constant1 [56].

Theorem 1. Fix constants a ̸= b > 0 and s ∈[0, 1]. Let (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s).
For any ε > 0, if s2 ≥α + ε, then the following holds.

(a) (Almost exact matching) There exists a polynomial-time algorithm that outputs a subset I ∈[n]
and a mapping bπ : I →[n] such that bπ = π∗|I and |I| = (1 −o(1))n with high probability.

(b) (Exact matching) If, in addition, s2(a + b)/2 > 1, then there exists a polynomial-time algorithm
that ouputs a mapping bπ such that limn→∞P(bπ = π∗) = 1.

Several remarks are now in order about the tightness of the main result, an overview of the chandelier
counting algorithm when a = b, and the main challenge of our analysis.

Tightness. This result is tight whenever s2 > α, because s2(a+b)/2 = 1 is the information-theoretic
threshold of exact graph matching given two correlated SBMs (G1, G2) [58]. When s2 < α, it is
conjectured that an information-computation gap exists for the correlated Erd˝os–Rényi graphs [42].
Specifically, by assuming that a = b, it is information-theoretically possible to match correlated
Erd˝os–Rényi graphs exactly if s2a > 1 [14, 15, 65]. However, it is believed hard to find a polynomial-
time algorithm to do this. In our model with SBMs, which is an extension from Erd˝os–Rényi graphs,
it is also likely hard to find a polynomial-time algorithm when s2 < α.

1This constant captures the base of the exponential growth of unlabeled rooted trees: the total number of
unlabeled rooted trees with N vertices is (α + o(1))−N.

4


---Page Break---
(a) s=0.6
(b) s=0.6
(c) s=0.8

Figure 2: Phase diagram for graph matching on (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s). The red
diagonal line depicts a = b, which is an Erd˝os–Rényi graph. Black regions: exact graph matching is
possible and can be done efficiently for each community separately by applying the graph matching
algorithm for correlated Erd˝os–Rényi graphs; Green regions: exact graph matching is possible and
can be done efficiently; Light green regions: exact graph matching is impossible, but almost exact
graph matching is possible and can be done efficiently; Cyan regions: exact graph matching is
possible and can be done efficiently by first recovering the community labels almost exactly; Yellow
regions: exact graph matching is impossible but almost exact graph matching can be done efficiently
by first recovering the community labels almost exactly.

Signed chandelier counts. Our theorem extends from the main theorem in Mao et al. [42], which
proposed a polynomial-time algorithm that matches the correlated Erd˝os–Rényi graphs exactly. The
algorithm has two main steps: First, construct signature vectors si and tj for vertices i ∈[n] in
G1 and j ∈[n] in G2 by the signed subgraph counts of a specially designed graph class—termed
Chandeliers—and calculate the weighted inner product of pairs of signature vectors ⟨si, tj⟩and
match vertices if the inner product value is large enough; Second, use a seeded graph matching
algorithm to boost the almost exact graph matching algorithm to exact graph matching. It is natural to
adapt this algorithm from correlated Erd˝os–Rényi graphs to correlated SBMs but the details present
non-trivial challenges, as we explain below.

Main challenge. The main challenge on correlated SBMs is that signed subgraph counts is no longer
a free lunch. Signed subgraph counts is counting the subgraphs on a centralized adjacency matrix,
which is first proposed by Bubeck et al. [9] and later commonly used to control the variance of
counting statistics. The success of the chandelier counting method relies on the sufficient separation
of the two inner product distributions of true and false vertex correspondence. We want to find a
way that keeps doing the adjacency matrices centralization possible. Recall that we explore the
graph matching motivated by community recovery. Interestingly, the solution to this centralization
problem is now the other way around—using a rough community label estimate to help the graph
matching. Our main technical contributions are first showing that when there are no error occurs in
the community label estimate, the signed chandelier counting can be generalized to correlated SBMs
and then show that when the exact community recovery is not possible, the errors introduced in the
community label estimation, which is polynomial in n, are actually tolerable for the whole algorithm.

The analysis falls in two cases. If sD+(a, b) > 1, then we can achieve exact community recovery
on each of the graphs by applying the community recovery algorithm from [48]. In addition, if
s2 a

2 > 1, 2 then it suffices to look at each community individually. This is easy and follows in a
black-box fashion from [42] (See Section 7). However, on the other side, s2 a

2 < 1, one still needs to
use information the community information. Therefore, we need to go through the whole algorithm
analysis again in this case. Note that the analysis would work for both regimes with no constraint on
s2 a

2 < 1. We plot the black-box regime in black and the non-black-box regime in green in Figure 2.

The second case is even trickier. If sD+(a, b) < 1, by the same algorithm, we can only obtain almost
exact correct community labels bσA and bσB on graph G1 and G2, respectively. We perform adjacency
matrix centralization based on bσA and bσB and show that the error introduced in this step is negligible

2For a SBM(n, a log n

n
, b log n

n
), each community, conditioned on its size N ≈n/2, is an Erd˝os–Rényi graph
G(N, a log n

n
) ≈G(N, a log N

2N
).

5


---Page Break---
(a) s=0.6
(b) s=0.7
(c) s=0.8

Figure 3: Phase diagram for exact community recovery with fixed s on correlated SBMs. Green
regions: exact community recovery is possible from G1 alone and can be done efficiently; Lightgreen
regions: exact community recovery is possible from (G1, G2) but impossible from G1 alone, exact
graph matching can be done efficiently and therefore exact community recovery can be done effi-
ciently; Violet regions: exact community recovery is impossible from G1 alone, impossible from
(G1, G2) if s2( a+b

2 ) + s(1 −s)D+(a, b) < 1 and possible if s2( a+b

2 ) + s(1 −s)D+(a, b) > 1 [25].
It is unknown whether there exists an efficient algorithm for exact community recovery in this regime.

in the sense that the inner-product scores remains sufficiently distinguishable between true pairs
(j = π∗(i)) and fake pairs (j ̸= π∗(i)).

4
Application: community recovery

Once matching up the vertices on two correlated graphs, we can combine the information of them
onto a union graph and then immediately have an application on community recovery. Our result for
community detection is that there exists a polynomial-time algorithm for exact community recovery
on correlated SBMs when the squared edge correlation parameter satisfies s2 > α.

Theorem 2. Fix constants a ̸= b > 0 and s ∈[0, 1]. Let (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s).
For any ε > 0, if
s2 ≥α + ε
and
(1 −(1 −s)2)D+(a, b) > 1,

then, there exists an estimator bσ = bσ(G1, G2) that can be computed in polynomial-time such that
limn→∞P(| 1

n
Pn
i=1 bσ(i)σ∗(i)| = 1) = 1.

Theorem 2 is a direct application of our Theorem 1. The proof mainly follows the Theorem 3.3
in [58], which gives exact community recovery on the union graph of G1 and G2 regarding to the
permutation bπ, G1 ∨bπ G2. The key difference is that we substitute the maximum a posterior estimator
used in the first step with the bπ(G1, G2) output by the algorithm used to prove Theorem 1. Figure 3
is a summary of the phase diagram for community recovery determined by this work along with
previous works [58, 25], focusing on the exact community recovery and efficiency.

Remark
1.
Consider
a
more
general
correlated
SBMs
with
K
correlated
graphs
(G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s, K). Theorem 1 also implies an efficient algorithm
for exact community recovery above the exact graph matching threshold for K correlated SBMs when
D+(a, b) >
1
1−(1−s)K (Theorem 3.6 in [58]).

5
Algorithm and the overview of the proofs

In this section, we define a few key concepts, give a brief overview of the algorithm, and briefly
discuss the proof of Theorem 1 and challenges. See Appendix C for a full version.

Chandelier. An (L, M, K, R, D)-chandelier is a rooted tree with L branches, each of which consists
of a path with M edges (M-wire), followed by a rooted tree with K edges (K-bulb); the K-bulbs are
non-isomorphic to each other, each of them has at most R automorphisms, and the maximum degree
is at most D. For each chandelier H, let K(H) denote the set of bulbs of H.

6


---Page Break---
For a rooted tree T, let aut(H) denote the number of rooted automorphisms of T throughout this
paper. We abbreviate rooted automorphism as automorphism when it is clear that we are applying it
to a chandelier. The number of automorphisms of H is determined by the automorphisms of its bulbs.

Let T denote the family of non-isomorphic (L, M, K, R, D)–chandelier. The family size of chande-
lier is |T | =
 |J |
L

, where J ≡J (K, R, D) denotes the collection of unlabeled rooted trees with K
edges, at most R automorphisms, and maximum degree D. Otter [56] showed that the number of
unlabeled rooted trees with K edges (and no constraint on the automorphisms and vertex degrees) is
|J (K, ∞, ∞)| = (α + o(1))−K, where α ≈0.338. We show that under proper choices of R and D,
we have |J (K, R, D)| = (α + o(1))−K in Section C.

Algorithm overview. Given (G1, G2) ∼CSBM(n, p, q, s). Our algorithm contains mainly three
steps. Firstly, we apply the Algorithm 3 (discussed in Section D.3) by Mossel, Neeman, and Sly [48]
to obtain almost exact community label estimates for each single graph. Secondly, we calculate
the signed chandelier counting [42] based similarity score to give an almost exact graph matching
(Algorithm 1 and Algorithm 4). Lastly, we boost the almost exact matching to exact matching by
extending the seeded graph matching algorithm [42] on Erd˝os–Rényi graphs to SBMs (Algorithm 2).

Subgraph counts. For an arbitrary weighted adjacency matrix M of some adjacency matrix A,
vertex i ∈[n], and a rooted graph H, we define the weighted subgraph counts on M as

Wi,H(M) :=
X

S(i)∼
=H
MS, where MS :=
Y

e∈E(S)
Me,

where S(i) enumerates through all subgraphs on the complete graph Kn, rooted at i and are isomor-
phic to H. When M is the adjacency matrix itself, Wi,H(M) is the usual subgraph count, representing
the number of subgraphs rooted at i in M that are isomorphic to H. When M is the centralized
adjacency matrix A := A−E[A], we call Wi,H(M) a signed subgraph count following [9]. However,
we do not have access to E[A] in many cases. Specifically for SBM, we can estimate E[A] through
estimating the community labels. We define the approximately centralized adjacency matrix regarding
to community label estimate bσ, denoted as A
bσ, entry-wise as A
bσ
i,j = Ai,j −p1bσ[i]=bσ[j] −q1bσ[i]̸=bσ[j].

Given a family H of non-isomorphic rooted graphs, we define the subgraph count signature of vertex
i as W H
i (M) := (Wi,H(M))H∈H.

Similarity score. Given a pair of correlated SBMs (G1, G2), we define the similarity score between
vertex i on graph G1 and vertex j on graph G2 as a weighted inner product between two signatures:

Φij := ⟨W T
i (A), W T
j (B)⟩:=
X

H∈T
aut(H)Wi,H(A)Wj,H(B).

When we do not have access to A and B, we use the approximately centralized adjacency matrices
A
bσ and B
bσ. We define the similarity score in a slightly different notation:

Φbσ
ij := ⟨W T
i (A
bσA), W T
j (B
bσB)⟩=
X

H∈T
aut(H)Wi,H(A
bσA)Wj,H(B
bσB).

Almost exact graph matching. The first part in the analysis is to show that by calculating this
similarity score, with an appropriate thresholding strategy, we can match up (1 −o(1))n vertices
correctly (Theorem 3). The high-level idea is to show that the similarity score distributions are
well-separated between true pairs and fake pairs. We expect the similarity score having the following
properties, under event H := { n

2 −n
3
4 ≤|V +|, |V −| ≤n

2 + n
3
4 } (to be discussed in Section D.2):

• For true pairs j = π(i) :

E[Φbσ
iπ∗(i)1H] > 0,
Var[Φbσ
ij1H] = o

E[Φbσ
iπ∗(i)1H]2
,
(1)

• For fake pairs j ̸= π(i) :

E[Φbσ
ij1H] = o

E[Φbσ
iπ∗(i)1H]

,
Var[Φbσ
ij1H] = o

 
E[Φbσ
iπ∗(i)1H]2

n2

!

.
(2)

7


---Page Break---
Precisely forming bounds for these moments constitutes the main bulk of the paper. We provide
results from Proposition 1 to Proposition 6. Followed by these moment bounds, we have Theorem 3.

Theorem 3. Fix a ̸= b > 0 and s ∈[0, 1]. Let p = a log n

n , q = b log n

n
and (G1, G2) ∼
CSBM(n, p, q, s).
For any ε > 0, suppose s2 ≥α + ε. There exists positive constants
C1, C2, C3, C4, C5 > 0 such that the following holds. Pick K, M, L, N, D as

L = C1

ε ,
K = C2 log n,
M =
C3K
log(ns(p ∧q)),
R = exp(C4K),
D = C5
log n
(log log n)2 .

(3)

Pick an arbitrary c ∈(0, 1) and set µ = |T |nNρNσ2N
eff , where σ2
eff := (
σ2
++σ2
−
2
). Then, Algorithm 1
outputs a set I with size (1 −o(1))n and a mapping bπ such that bπ|I = bπ∗|I with high probability.

Proof challenge. In regime sD+(a, b) > 1, the probability of existing one vertex being classified

incorrectly is vanishing for Algorithm 3. Therefore, with high probability, A
bσA = A. If sD+(a, b) <
1, then the recovered bσ contains errors (polynomial in n), which will cause some edges being
centralized incorrectly and thereby affect the moments calculation.

This phenomenon poses a challenge to the algorithm analysis. We highlight some key points in the
context of second moment calculation. For simplicity, we ignore event H here. From definition,

Var[Φbσ
ij] =
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I


Eσ∗
h
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 | σ∗]
i

−Eσ∗
h
E[A
bσA
S1 B
bσB
S2 | σ∗]
i
Eσ∗
h
E[A
bσA
T1 B
bσB
T2 | σ∗]
i 
.

Let us define the union graph U := S1 ∪S2 ∪T1 ∪T13. If sD+(a, b) > 1, we view A
bσ and B
bσ as
A and B, respectively. E[AS1BS2AT1AT2 | σ∗] ̸= 0 only if there exists no edge e ∈E(U) such
that it occurs only once among (S1, S2, T1, T2). This is because different edges are independent and
centralized conditioned on σ∗.

However, in regime sD+(a, b) < 1, E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2
| σ∗] ̸= 0 even when edges are not

occurring multiple times as we have the expectation of A
bσA
e
can be non-zero when conditioning on
σ∗and an estimate bσA that disagrees with σ∗on the edge type (i.e., for e = (u, v), σ∗(u)σ∗(v) ̸=
bσA(u)bσA(v)). This not only causes this cross-moments calculation being more complicated, but
significantly increasing the possibility of the combinations between S1, S2, T1, and T2.

The most important high-level idea to properly bound the moments is: the cross-moment conditioning
on a specific bσ = (bσA, bσB) would be non-trivial if and only if all edges occurring only once are
centralized incorrectly. Assuming there are z edges occurring once, we show that the probability

that bσ satisfying this property is no greater than n−
z(sD+(a,b)−ε log(a/b)/2)

D
for any ε > 0, by using
the definition of T and Lemma 8 later in Section K and Section L. It turns out that we require

n−
z(sD+(a,b)−ε log(a/b)/2)

D
to be o(
1
logC n) for some positive constant C so that (1) and (2) are satisfied.

Efficient algorithm. Calculating Φbσ
ij exactly (Algorithm 1) takes quasi-polynomial time as searching
for all S ∼= H has time complexity nΘ(N). Algorithm 4 in Section F computes an approximated score
in polynomial time. Specifically, we follow the color coding-based similarity score approximation
from [42]. The basic idea is coloring the stochastic block model using N + 1 colors uniformly at
random. Then, we only do signed counts on vertex sets that are colorful with N + 1 distinct colors.
We show that this is an unbiased estimator of Φbσ
ij and only potentially increase the variance by an
additional constant factor in Section F. The result is stated as in Theorem 4.

Theorem 4. Theorem 3 continues to hold with the color-coding sampled estimation eΦbσ
ij in place of
Φbσ
ij. Moreover, by Algorithm 4, {eΦbσ
ij}i,j∈[n] can be computed in O(nC) for some constant C ≡C(ε)
depending only on ε, where ε is from (3).

3Note that S1 and T1 are rooted at i, while S2 and T2 are rooted at j. For e = (u, v) ∈E(S1), we say it
occurs in T1 if e ∈E(T1) and it occurs in S2 (resp. T2) if (σ∗(u), σ∗(v)) ∈E(S2) (resp. T2).

8


---Page Break---
Exact graph matching. The final step of the algorithm is boosting the almost exact matching to a
exact matching. The key idea is exploring the number of common neighbors.

Define h(x) = x log x −x + 1. Denote Nπ(i, j) as the number of common neighbors of i and j
under correspondence π. The idea is that if i and j form a true correspondence, then, with high
probability Neπ(i, j) ≥γ p2+q2

2
s2(n + 2n
3
4 ), under the nice event H. Therefore, we match up i and j
if they have more common neighbors than this threshold. We give the following guarantee.

Theorem 5. Fix a ̸= b > 0 and s ∈[0, 1]. Let p = a log n

n , q = b log n

n
and (G1, G2) ∼
CSBM(n, p, q, s). Suppose s2( a+b

2 ) ≥1 + ε
and
s2 ≥α + ε, for some ε > 0. Let γ be
the unique solution in (1, ∞) to h(γ) =
3 log n
(n−2)pqs2 . Then, the seeded matching Algorithm 2 with
input bπ and an index set I ⊂[n], |I| = (1 −o(1))n ≥(1 −ε/16)n such that bπ|I = π∗outputs an
exact matching eπ = π∗in O(n3(p + q)2) time with probability 1 −o(1).

Putting these pieces together implies Theorem 1. From Theorem 4, we know that for (G1, G2) ∼
CSBM(n, a log n

n , b log n

n , s), a ̸= b, if s2 ≥α+ε for some ε > 0 then there we can match (1−o(1))n
vertices correctly and efficiently with high probability. Take the returned bπ from Algorithm 4 as input
of Algorithm 2, then Theorem 5 guarantees that the final output eπ = π∗with high probability.

6
Related work

Graph Matching.
Correlated Erd˝os–Rényi graphs were first studied in [57] for social network de-
anonymization. The information-theoretic threshold for partial matching was determined by [24, 28]
and the information-theoretic threshold for exact graph matching was determined by [14, 15, 65].

Mao et al. [41] proposed the first efficient algorithm that achieves exact graph matching for correlated
Erd˝os–Rényi graphs with average degree (1 + ε) log n ≤nq ≤n

1
Θ(log log n) and constant noise. This
algorithm only requires a constant edge correlation (sufficiently close to 1) rather than converging
to 1, which represents a perfect correlation. Mao et al. [42] followed up with an improved efficient
algorithm that achieves exact graph matching for any correlation ρ satisfying ρ2 > α when nq(q +
ρ(1 −q)) ≥(1 + ε) log n. It is conjectured that for random graphs of logarithmic average degree,
ρ2 = α is the computational threshold [42]. Muratori and Semerjian [51] added a small constant
constraint on the maximum vertex degree of a chandelier to improve the runtime, at the expense of
having a slightly larger constant bα as the minimum squared correlation requirement.

In the denser regime where p = n−a+o(1), a ∈(0, 1], Ding and Du [17] established a sharp
information-theoretic threshold for matching a positive fraction of vertices. Ding and Li [20] also
developed an efficient algorithm for exact graph matching whenever the edge correlation is non-
vanishing, which goes beyond the Otter’s tree counting constant.

Several recent works also go beyond correlated Erd˝os–Rényi graphs. Wang et al. [64] studied the
exact graph matching with additional attribute information on vanishing edge correlation.

Closely related to our work, Yang et al. [67] adopted the binary tree counting algorithm [41] to give
an efficient graph matching algorithm for correlated SBMs. However, [67] makes several significant
assumptions (which we do not). For one, the algorithm in [67] assumes that the community labels are
known. This is a strong assumption which may be unrealistic in practice; moreover, this precludes
using graph matching as a tool for improved community recovery. In contrast, we do not assume
that community labels are known; in fact, a significant part of our technical work is devoted to
dealing with the errors arising from estimating the community labels. Moreover, our graph matching
algorithm can be directly applied to improve community recovery, as discussed in Theorem 2 and
Section 3. In addition, [67] makes strong assumptions on the parameters, assuming that (1) the
average degree is at least (log n)1.1, (2) the SBM has at least 3 communities, and (3) the correlation
parameter satisfies s > 1 −ε0 for some unspecified (small) ε0. In contrast, our results hold in the
most interesting regime of logarithmic average degree and the most natural setting of two balanced
communities; moreover, our assumption on s is also weaker.

Community recovery with side information
Beyond correlated SBMs, there are some other mod-
els utilizing side information, from multiple networks [29, 62, 34, 31, 68, 71], additional covariates
[7], or both [45, 38]. Multi-layer SBM is first mentioned in [30], which is generated as following:

9


---Page Break---
first, generate the community labels for all vertices and fix them for all layers; second, form edges
on each layer based on the community labels. Typically, different layers in a multi-layer SBM are
conditionally independent given the shared community labels. In addition, several works [45, 38]
also encode community membership correlated covariates onto each node. Aside from the multi-layer
SBM, Braun and Sugiyama [7] recently studied community detection on a novel variation of SBM
whose edges are attached with vectorial covariates.

7
Discussion and future work

Our main contribution in this paper is to give the first efficient algorithm for exact graph matching
for correlated SBMs with two balanced communities, as well as a rigorous proof of its correctness
(Theorem 1). We also discuss novel applications to community recovery (Theorem 2). At the same
time, our work raises many interesting questions for future research, which we discuss here.

Optimal runtime. While our graph matching algorithm is efficient, it would be desirable to under-
stand the optimal running time that can be achieved. Mao, Rudelson, and Tikhomirov [41] gave
an efficient algorithm for matching correlated Erd˝os–Rényi graphs with runtime n2+o(1); the main
drawback is that this algorithm requires the correlation parameter to satisfy s > 1 −ε0 for some
unspecified (small) ε0 > 0. Nonetheless, it would be interesting to generalize this algorithm to
correlated SBMs and the techniques developed in our work may be useful to do so. In very recent
(and concurrent) work, Muratori and Semerjian [51] gave faster algorithms for matching correlated
Erd˝os–Rényi graphs by introducing a constraint on the maximum degree of a chandelier, at the
expense of strengthening the condition s2 > α to s2 > bα for some bα > α. Exploring the connections
between our work and theirs, and generalizing their ideas to correlated SBMs, are of interest.

Information-computation gap. An important assumption throughout this work is that the correlation
parameter satisfies s2 > α. We believe that this is inherently necessary and that there is no efficient
algorithm (in the logarithmic average degree regime) when s2 < α. At the same time, exact graph
matching is information-theoretically possible whenever s2(a + b)/2 > 1, so there is a conjectured
information-computation gap. This mirrors the conjecture in [42] for Erd˝os–Rényi graphs; see
also [18] for the low-degree hardness results on the correlation detection and [10] for the very recent
low-degree hardness results on testing a pair of correlated stochastic block models against a pair of
independent Erd˝os–Rényi graphs.

Efficient exact community recovery when exact graph matching is not possible. Gaudio, Rácz,
and Sridhar [25] determined the information-theoretic threshold for exact community recovery on
correlated SBMs, in particular showing that there is a regime when this is possible even though (1)
this is impossible with a single graph and also (2) exact graph matching is impossible. It remains
unknown whether this can be done efficiently in this regime. We believe that this is possible, and our
work is an important starting point for this question, yet additional ideas are needed to understand the
subtle interplay between graph matching and community recovery in this regime.

Sparser and denser regimes. Our work focuses on the most interesting regime where the average
degree is logarithmic in n; it is worth understanding other regimes too. In particular, the chandelier
counting algorithm by Mao et al. [42] gives almost exact graph matching whenever the average
degree diverges. In our Theorem 1 we require that the average degree diverges logarithmically for the
corresponding result, so that the error rate for community recovery estimate is polynomially small
in n. It would be interesting to overcome this technical barrier and extend the analysis to this sparser
regime. Denser regimes are easier to understand. A close inspection of our analysis shows that it also
works when the average degree diverges as a (small) polynomial in n; in even denser regimes, the
community partition can be recovered exactly and efficiently whenever lim infn→∞|pn/qn −1| > 0
(see [48]) and then the graph matching algorithm in [42] can be applied in a black-box fashion.

General block models. We focused here on the simplest case of SBMs with two balanced com-
munities. It is of great interest to develop efficient graph matching algorithms in the general block
model with k communities, whenever this is possible. Recently, Yang and Chung [66] determined
the information-theoretic threshold for exact graph matching in the k-community symmetric SBM,
extending the results of Rácz and Sridhar [58]. We conjecture that substituting the community recov-
ery algorithm used in our work with the degree-profiling algorithm by Abbe and Sandon [4] gives an
efficient algorithm for graph matching in this more general setting, assuming again that s2 > α.

10


---Page Break---
Acknowledgements

We thank Julia Gaudio, Anirudh Sridhar, Yihong Wu, and Jiaming Xu for helpful discussions. We
also thank anonymous reviewers for constructive feedback. S.C. was supported in part by the Institute
for Data, Econometrics, Algorithms, and Learning (IDEAL), funded through the National Science
Foundation TRIPODS Phase II program (NSF grant ECCS 2216970).

References

[1] Emmanuel Abbe. Community Detection and Stochastic Block Models: Recent Developments.
Journal of Machine Learning Research, 18(177):1–86, 2018.

[2] Emmanuel Abbe, Afonso S. Bandeira, and Georgina Hall. Exact Recovery in the Stochastic
Block Model. IEEE Transactions on Information Theory, 62(1):471–487, 2016.

[3] Emmanuel Abbe, Jianqing Fan, Kaizheng Wang, and Yiqiao Zhong. Entrywise eigenvector
analysis of random matrices with low expected rank. The Annals of Statistics, 48(3):1452–1474,
2020.

[4] Emmanuel Abbe and Colin Sandon. Community detection in general stochastic block models:
Fundamental limits and efficient algorithms for recovery. In 2015 IEEE 56th Annual Symposium
on Foundations of Computer Science (FOCS), pages 670–688. IEEE, 2015.

[5] Boaz Barak, Chi-Ning Chou, Zhixian Lei, Tselil Schramm, and Yueqi Sheng. (Nearly) Efficient
Algorithms for the Graph Matching Problem on Correlated Random Graphs. In Advances in
Neural Information Processing Systems (NeurIPS), volume 32, pages 9190–9198, 2019.

[6] Terry Beyer and Sandra Mitchell Hedetniemi. Constant Time Generation of Rooted Trees.
SIAM Journal on Computing, 9(4):706–712, 1980.

[7] Guillaume Braun and Masashi Sugiyama. VEC-SBM: Optimal Community Detection with
Vectorial Edges Covariates. In Proceedings of The 27th International Conference on Artificial
Intelligence and Statistics (AISTATS), volume 238 of Proceedings of Machine Learning Research
(PMLR), pages 532–540, 2024.

[8] Karl Bringmann, Tobias Friedrich, and Anton Krohmer. De-anonymization of Heterogeneous
Random Graphs in Quasilinear Time. In Proceedings of the 22nd Annual European Symposium
on Algorithms (ESA), pages 197–208, 2014.

[9] Sébastien Bubeck, Jian Ding, Ronen Eldan, and Miklós Z Rácz. Testing for high-dimensional
geometry in random graphs. Random Structures & Algorithms, 49(3):503–532, 2016.

[10] Guanyi Chen, Jian Ding, Shuyang Gong, and Zhangsong Li. A computational transition for
detecting correlated stochastic block models by low-degree polynomials. Preprint available at
https://arxiv.org/abs/2409.00966, 2024.

[11] Carla-Fabiana Chiasserini, Michele Garetto, and Emilio Leonardi.
Social Network De-
Anonymization Under Scale-Free User Relations. IEEE/ACM Transactions on Networking,
24(6):3756–3769, 2016.

[12] Charles J. Colbourn and Kellogg S. Booth. Linear Time Automorphism Algorithms for Trees,
Interval Graphs, and Planar Graphs. SIAM Journal on Computing, 10(1):203–225, 1981.

[13] Donatello Conte, Pasquale Foggia, Carlo Sansone, and Mario Vento. Thirty years of graph
matching in pattern recognition. International Journal of Pattern Recognition and Artificial
Intelligence, 18(03):265–298, 2004.

[14] Daniel Cullina and Negar Kiyavash. Improved Achievability and Converse Bounds for Erd˝os-
Rényi Graph Matching. In ACM SIGMETRICS, volume 44, pages 63–72, 2016.

[15] Daniel Cullina and Negar Kiyavash. Exact alignment recovery for correlated Erd˝os-Rényi
graphs. Preprint available at https://arxiv.org/abs/1711.06783, 2018.

11


---Page Break---
[16] Daniel Cullina, Negar Kiyavash, Prateek Mittal, and H. Vincent Poor. Partial Recovery of
Erd˝os-Rényi Graph Alignment via k-Core Alignment. In ACM SIGMETRICS Performance
Evaluation Review, volume 48, pages 99–100. ACM, 2020.

[17] Jian Ding and Hang Du. Matching recovery threshold for correlated random graphs. The Annals
of Statistics, 51(4):1718–1743, 2023.

[18] Jian Ding, Hang Du, and Zhangsong Li. Low-Degree Hardness of Detection for Correlated
Erd˝os-Rényi Graphs. Preprint available at https://arxiv.org/abs/2311.15931, 2023.

[19] Jian Ding, Yumou Fei, and Yuanzheng Wang. Efficiently matching random inhomogeneous
graphs via degree profiles. Preprint available at https://arxiv.org/abs/2310.10441,
2023.

[20] Jian Ding and Zhangsong Li. A polynomial-time iterative algorithm for random graph matching
with non-vanishing correlation. Preprint available at https://arxiv.org/abs/2306.00266,
2023.

[21] Jian Ding, Zongming Ma, Yihong Wu, and Jiaming Xu. Efficient random graph matching via
degree profiles. Probability Theory and Related Fields, 179(1):29–115, 2021.

[22] Zhou Fan, Cheng Mao, Yihong Wu, and Jiaming Xu. Spectral Graph Matching and Regularized
Quadratic Relaxations: Algorithm and Theory. In Proceedings of the 37th International
Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning
Research (PMLR), pages 2985–2995, 2020.

[23] Luca Ganassali and Laurent Massoulié. From tree matching to sparse graph alignment. In
Proceedings of the 33rd Conference on Learning Theory (COLT), volume 125 of Proceedings
of Machine Learning Research (PMLR), pages 1633–1665, 2020.

[24] Luca Ganassali, Laurent Massoulié, and Marc Lelarge. Impossibility of Partial Recovery in the
Graph Alignment Problem. In Proceedings of the 34th Conference on Learning Theory (COLT),
volume 134 of Proceedings of Machine Learning Research (PMLR), pages 2080–2102, 2021.

[25] Julia Gaudio, Miklós Z. Rácz, and Anirudh Sridhar. Exact Community Recovery in Correlated
Stochastic Block Models. In Proceedings of the 35th Conference on Learning Theory (COLT),
volume 178 of Proceedings of Machine Learning Research (PMLR), pages 2183–2241, 2022.

[26] Julia Gaudio, Miklós Z. Rácz, and Anirudh Sridhar. Exact community recovery in correlated
stochastic block models. In Proceedings of Thirty Fifth Conference on Learning Theory (COLT),
volume 178 of Proceedings of Machine Learning Research, pages 2183–2241. PMLR, 02–05
Jul 2022.

[27] William M.Y. Goh and Eric Schmutz. Unlabeled Trees: Distribution of the Maximum Degree.
Random Structures & Algorithms, 5(3):411–440, 1994.

[28] Georgina Hall and Laurent Massoulié. Partial recovery in the graph alignment problem. Opera-
tions Research, 71(1):259–272, 2023.

[29] Qiuyi Han, Kevin Xu, and Edoardo Airoldi. Consistent estimation of dynamic and multi-
layer block models. In International Conference on Machine Learning (ICML), volume 37 of
Proceedings of Machine Learning Research (PMLR), pages 1511–1520, 2015.

[30] Paul W. Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. Stochastic blockmodels:
First steps. Social Networks, 5(2):109–137, 1983.

[31] Yaofang Hu and Wanjie Wang.
Network-adjusted covariates for community detection.
Biometrika, page asae011, 2024.

[32] Svante Janson, Tomasz Łuczak, and Andrzej Rucinski. Random Graphs. John Wiley & Sons,
2000.

[33] Nitish Korula and Silvio Lattanzi. An efficient reconciliation algorithm for social networks.
Proceedings of the VLDB Endowment, 7(5):377–388, 2014.

12


---Page Break---
[34] Jing Lei, Kehui Chen, and Brian Lynch. Consistent community detection in multi-layer network
data. Biometrika, 107(1):61–73, 2020.

[35] Vince Lyzinski. Information Recovery in Shuffled Graphs via Graph Matching. IEEE Transac-
tions on Information Theory, 64(5):3254–3273, 2018.

[36] Vince Lyzinski, Daniel L. Sussman, Donniell E. Fishkind, Henry Pao, Li Chen, Joshua T.
Vogelstein, Youngser Park, and Carey E. Priebe. Spectral clustering for divide-and-conquer
graph matching. Parallel Computing, 47:70–87, 2015.

[37] Jiayi Ma, Xingyu Jiang, Aoxiang Fan, Junjun Jiang, and Junchi Yan. Image Matching from
Handcrafted to Deep Features: A Survey. International Journal of Computer Vision, 129:23–79,
2021.

[38] Zongming Ma and Sagnik Nandy. Community Detection with Contextual Multilayer Networks.
IEEE Transactions on Information Theory, 69(5):3203–3239, 2023.

[39] Konstantin Makarychev, Rajsekar Manokaran, and Maxim Sviridenko. Maximum Quadratic
Assignment Problem: Reduction from Maximum Label Cover and LP-Based Approximation
Algorithm. In International Colloquium on Automata, Languages, and Programming (ICALP),
pages 594–604. Springer, 2010.

[40] Cheng Mao, Mark Rudelson, and Konstantin Tikhomirov. Random Graph Matching with
Improved Noise Robustness. In Proceedings of the 34th Conference on Learning Theory
(COLT), volume 134 of Proceedings of Machine Learning Research (PMLR), pages 3296–3329,
2021.

[41] Cheng Mao, Mark Rudelson, and Konstantin Tikhomirov. Exact matching of random graphs
with constant correlation. Probability Theory and Related Fields, 186:327–389, 2023.

[42] Cheng Mao, Yihong Wu, Jiaming Xu, and Sophie H. Yu. Random Graph Matching at Otter’s
Threshold via Counting Chandeliers. In Proceedings of the 55th Annual ACM Symposium on
Theory of Computing (STOC), pages 1345–1356, 2023.

[43] Cheng Mao, Yihong Wu, Jiaming Xu, and Sophie H. Yu. Testing network correlation efficiently
via counting trees. The Annals of Statistics, to appear, 2024+.

[44] Laurent Massoulié. Community detection thresholds and the weak Ramanujan property. In
Proceedings of the 46th Annual ACM Symposium on Theory of Computing (STOC), pages
694–703. ACM, 2014.

[45] Vaishakhi Mayya and Galen Reeves.
Mutual Information in Community Detection with
Covariate Information and Correlated Networks. In 2019 57th Annual Allerton Conference on
Communication, Control, and Computing (Allerton), pages 602–607. IEEE, 2019.

[46] Michael Mitzenmacher and Eli Upfal. Probability and Computing: Randomized Algorithms
and Probabilistic Analysis. Cambridge University Press, 2005.

[47] Elchanan Mossel, Joe Neeman, and Allan Sly. Stochastic block models and reconstruction.
arXiv preprint arXiv:1202.1499, 2012.

[48] Elchanan Mossel, Joe Neeman, and Allan Sly. Consistency thresholds for the planted bisection
model. Electronic Journal of Probability, 21(none):1 – 24, 2016.

[49] Elchanan Mossel, Joe Neeman, and Allan Sly. A proof of the block model threshold conjecture.
Combinatorica, 38(3):665–708, 2018.

[50] Elchanan Mossel and Jiaming Xu. Seeded graph matching via large neighborhood statistics.
Random Structures & Algorithms, 57(3):570–611, 2020.

[51] Andrea Muratori and Guilhem Semerjian.
Faster algorithms for the alignment of sparse
correlated Erd˝os–Rényi random graphs. Preprint available at https://arxiv.org/abs/
2405.08421, 2024.

13


---Page Break---
[52] Arvind Narayanan and Vitaly Shmatikov. De-anonymizing Social Networks. In Proceedings of
the 30th IEEE Symposium on Security and Privacy, pages 173–187. IEEE Computer Society,
2009.

[53] Ryan O’Donnell, John Wright, Chenggang Wu, and Yuan Zhou. Hardness of Robust Graph
Isomorphism, Lasserre Gaps, and Asymmetry of Random Graphs. In Proceedings of the
Twenty-Fifth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1659–1677.
SIAM, 2014.

[54] Christoffer Olsson and Stephan Wagner. Automorphisms of Random Trees. In 33rd Interna-
tional Conference on Probabilistic, Combinatorial and Asymptotic Methods for the Analysis
of Algorithms (AofA 2022), volume 225 of Leibniz International Proceedings in Informatics
(LIPIcs), pages 16:1–16:16. Schloss Dagstuhl-Leibniz-Zentrum für Informatik, 2022.

[55] Efe Onaran, Siddharth Garg, and Elza Erkip. Optimal de-anonymization in random graphs with
community structure. In 2016 50th Asilomar Conference on Signals, Systems and Computers,
pages 709–713. IEEE, 2016.

[56] Richard Otter. The Number of Trees. Annals of Mathematics, 49(3):583–599, 1948.

[57] Pedram Pedarsani and Matthias Grossglauser. On the privacy of anonymized networks. In
Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining (KDD), pages 1235–1243, 2011.

[58] Miklós Z. Rácz and Anirudh Sridhar. Correlated Stochastic Block Models: Exact Graph
Matching with Applications to Recovering Communities. In Advances in Neural Information
Processing Systems (NeurIPS), volume 34, pages 22259–22273, 2021.

[59] Miklós Z. Rácz and Anirudh Sridhar. Correlated randomly growing graphs. The Annals of
Applied Probability, 32(2):1058–1111, 2022.

[60] Miklós Z. Rácz and Anirudh Sridhar. Matching Correlated Inhomogeneous Random Graphs
using the k-core Estimator. In 2023 IEEE International Symposium on Information Theory
(ISIT), pages 2499–2504, 2023.

[61] Rohit Singh, Jinbo Xu, and Bonnie Berger. Global alignment of multiple protein interaction
networks with application to functional orthology detection. Proceedings of the National
Academy of Sciences, 105(35):12763–12768, 2008.

[62] Toni Vallès-Català, Francesco A. Massucci, Roger Guimerà, and Marta Sales-Pardo. Multilayer
Stochastic Block Models Reveal the Multilayer Structure of Complex Networks. Physical
Review X, 6(1):011036, 2016.

[63] Haoyu Wang, Yihong Wu, Jiaming Xu, and Israel Yolou. Random Graph Matching in Geometric
Models: the Case of Complete Graphs. In Proceedings of the 35th Conference on Learning
Theory (COLT), volume 178 of Proceedings of Machine Learning Research (PMLR), pages
3441–3488, 2022.

[64] Ziao Wang, Weina Wang, and Lele Wang. Efficient Algorithms for Attributed Graph Alignment
with Vanishing Edge Correlation. In Proceedings of the 37th Conference on Learning Theory
(COLT), volume 247 of Proceedings of Machine Learning Research (PMLR), pages 4889–4890,
2024.

[65] Yihong Wu, Jiaming Xu, and Sophie H. Yu. Settling the Sharp Reconstruction Thresholds of
Random Graph Matching. IEEE Transactions on Information Theory, 68(8):5391–5417, 2022.

[66] Joonhyuk Yang and Hye Won Chung. Graph Matching in Correlated Stochastic Block Models
for Improved Graph Clustering. In Proceedings of the 2023 59th Annual Allerton Conference
on Communication, Control, and Computing (Allerton), pages 1–8. IEEE, 2023.

[67] Joonhyuk Yang, Dongpil Shin, and Hye Won Chung. Efficient Algorithms for Exact Graph
Matching on Correlated Stochastic Block Models with Constant Correlation. In Proceedings of
the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of
Machine Learning Research (PMLR), pages 39416–39452, 2023.

14


---Page Break---
[68] Xiaodong Yang, Buyu Lin, and Subhabrata Sen. Fundamental limits of community detection
from multi-view data: multi-layer, dynamic and partially labeled block models. Preprint
available at https://arxiv.org/abs/2401.08167, 2024.

[69] Lyudmila Yartseva and Matthias Grossglauser. On the Performance of Percolation Graph
Matching. In Proceedings of the First ACM Conference on Online Social Networks (COSN),
pages 119–130, 2013.

[70] Liren Yu, Jiaming Xu, and Xiaojun Lin. The Power of D-hops in Matching Power-Law Graphs.
Proceedings of the ACM on Measurement and Analysis of Computing Systems, 5(2):1–43, 2021.

[71] Jingnan Zhang, Junhui Wang, and Xueqin Wang. Consistent Community Detection in Inter-
Layer Dependent Multi-Layer Networks. Journal of the American Statistical Association, pages
1–11, 2024.

15


---Page Break---
A
Organization

The rest of this paper is organized as follows. In Section B, we give some notations used throughout.
In Section C, we define the similarity score of a pair of vertices and give the formal proof of Theorem 1
based on Theorem 3 (Almost exact graph matching), Theorem 4 (Efficient algorithm for almost exact
graph matching), and Theorem 5 (Exact graph matching by seeded graph matching). In Section D,
we talk about some preliminaries: tail bounds, nice events, a tree node assigning sub-problem,
an automorphism inequality for trees, and the calculation for cross-moments. These results will
be repeatedly used in the following sections. In Section E, Section F, and Section G, we present
the proofs for Theorem 3, Theorem 4, and Theorem 5, respectively. Section F and Section G are
self-contained, while Section E contains several propositions whose proofs are deferred and which
make up the remainder of the paper.

In Section E, we introduce six additional Propositions to show that under two different cases—
namely, sD+(a, b) ≥1 and sD+(a.b) < 1—the mean and variance of the similarity score are
properly controlled. Specifically, if sD+(a, b) ≥1, then Proposition 1 gives the mean calculation
of the similarity score, while Proposition 2 and Proposition 3 are about the variance calculation of
the similarity score for true pairs and fake pairs of vertices. If sD+(a, b) < 1, then Proposition 4
gives the mean calculation of the similarity score, while Proposition 5 and Proposition 6 are about
the variance calculation of the similarity score for true pairs and fake pairs of vertices. The proofs
of Proposition 1, Proposition 2, Proposition 3, Proposition 4, Proposition 5, and Proposition 6 are
presented in Section H, Section I, Section J, Section K, Section L, and Section M, respectively.

B
Notation

For any graph G = (V, E), we denote E(G) as the edge set and V (G) as the vertex set. We let
e(G) := |E(G)| denote the number of edges of graph G and v(G) := |V (G)| denote the number of
vertices in graph G. We define the excess of graph G as e(G) −v(G), the difference between the
number of edges and the number of vertices of G.

Consider an arbitrary graph where vertices are equipped with two possible community labels
{+1, −1}, we denote V + as the set of vertices with community label +1, V −as the set of ver-
tices with community label −1, N(v) as the set vertices that are neighbors of v.

Let π be a permutation on [n], (G1, G2) ∼CSBM(n, p, q, s). Let A (resp. B) be the adjacency
matrix of G1 (resp. G2). Let A := A−E[A] (resp. B) be the centralized adjacency matrix. We further
define the approximately centralized adjacency matrix with respect to community label estimate bσ
as A
bσA := A −EA, where EA is an n × n matrix whose (i, j)-th entry is p if bσ(i) = bσ(j) and q
otherwise.4

We denote G1 ∨π G2 as the union graph with respect to π, such that (i, j) ∈E(G1 ∨π G2) if and
only if (i, j) ∈E(G1) or (π(i), π(j)) ∈E(G2). We denote G1 ∧π G2 as the intersection graph with
respect to π, such that (i, j) ∈E(G1 ∧π G2) if and only if (i, j) ∈E(G1) and (π(i), π(j)) ∈E(G2).

We denote the variance of in-community edges σ2
+ := sp(1−sp) and the variance of cross-community
edges as σ2
−:= sq(1 −sq). Denote the correlation for in-community edges and cross-community
edges as ρ+ and ρ−, respectively. In addition, we define ρ := ρ++ρ−

2
. Consider the average degree
being logarithmic in the number of vertices, then for some constants a, b > 0, p = a log n

n
, q = b log n

n
,
ρ = (1 + Θ( log n

n ))ρ+ = (1 + Θ( log n

n ))ρ−= (1 + Θ( log n

n ))s.5

Throughout the paper, we use standard asymptotic notation O(·), Ω(·), Θ(·), o(·), ω(·). Any limitation
is for n →∞without special explanations. For real numbers x, y, we define x ∨y := max{x, y}
and x ∧y := min{x, y}. In this paper, log is natural logarithmic function (with base e). Further
notation is introduced in the following section which details the algorithms.

4See Section K for illustrations and more discussions on approximately centralized adjacency matrices.
5For arbitrarily small constant ε > 0, there exists n large enough, such that ρ2 ≥α + ε if and only if
s2 ≥α + ε.

16


---Page Break---
C
Proof overview and proof of Theorem 1

C.1
Chandelier

A line of works convert the graph matching problem from quadratic assumption to linear assignment
by creating a signature vector si for each vertex i ∈[n], followed by calculating the similarity
score Φij = ⟨s(1)
i , s(2)
j ⟩of all possible pairs of signatures on two graphs. Recently, Mao et al. [42]
proposed a special tree family T , chandelier, that shows a result of efficient graph matching under
constant correlation.

Definition 1 ((L, M, K, R)–chandelier[42]). An (L, M, K, R)-chandelier is a rooted tree with L
branches, each of which consists of a path with M edges (M-wire), followed by a rooted tree with
K edges (K-bulb); the K-bulbs are non-isomorphic to each other and each of them has at most R
automorphisms.

In this paper, we give an alternative definition of chandelier with five tuple. The first four parameters
remain the same as (L, M, K, R)–chandelier. The last parameter D stands for the maximum degree
of vertices on this chandelier. We explain the necessity of controlling D in the proof challenge.

Definition 2 ((L, M, K, R, D)–chandelier). An (L, M, K, R, D)-chandelier is a rooted tree with
L branches, each of which consists of a path with M edges (M-wire), followed by a rooted tree
with K edges (K-bulb); the K-bulbs are non-isomorphic to each other, each of them has at most R
automorphisms, and the degree of each vertex is at most D.

For each chandelier H, let K(H) denote the set of bulbs of H. For a rooted tree T, let aut(H)
denote the number of rooted automorphisms of T throughout this paper. We abbreviate rooted
automorphism as automorphism when it is clear that we are applying it to a chandelier. The number
of automorphisms of H is determined by the automorphisms of its bulbs. Because all bulbs are
non-isomorphic to each other,

aut(H) =
Y

B∈K(H)
aut(B).
(4)

Let T denote the family of non-isomorphic (L, M, K, R, D)–chandelier. The family size of chande-
lier is |T | =
 |J |
L

, where J ≡J (K, R, D) denotes the collection of unlabeled rooted trees with K
edges, at most R automorphisms, and maximum degree D.

Otter [56] showed that the number of unlabeled rooted trees with K edges (and no constraint on
the automorphisms and vertex degrees) is |J (K, ∞, ∞)| = (α + o(1))−K, where α ≈0.338. We
show that under proper choices of R and D, we have |J (K, R, D)| = (α + o(1))−K through the
following two Lemmas.

Lemma 1. Let K be the number of vertices on a unlabeled rooted tree, C′ >
1
log(1/α) ≈0.9227,
D ≥C′ log K. As K →∞,

|J (K, ∞, D)|
|J (K, ∞, ∞)| = 1 −o(1).
(5)

Proof. Otter [56] characterized that |J (K,∞,D)|

|J (K,∞,∞)| ≍α−n
D
α−n , where αD is the radius of convergence
for the generating function of the number of unlabeled rooted trees whose maximum vertex degree
less than or equal to D. Goh and Schumutz [27] (Theorem 7) showed the following property:
as D →∞, for some constant C > 0, αD = α + CαD + o(αD). Immediately we can see
that |J (K,∞,D)|

|J (K,∞,∞)| = (1 + O(αD))−K = 1 −o(1) if KαD →0. Let C′ >
1
log(1/α), choosing
D ≥C′ log K satisfies KαD →0.

Lemma 2. Let K be the number of vertices on a unlabeled rooted tree, C be a constant and choose
R = exp(CK). For sufficiently large C and K →∞,

|J (K, R, ∞)|
|J (K, ∞, ∞)| = 1 −o(1).
(6)

17


---Page Break---
Figure 4: A chandelier.

Proof. Olsson and Wagner [54] (Theorem 2) showed a central limit theorem result for the number of
automorphism on unlabeled rooted trees:
1
√

K (log aut(HK) −µK) →N(0, σ2) as K →∞, where
HK is a uniform random unlabeled rooted tree with K edges and µ ≈0.137, σ2 ≈0.197. This
implies that for some constant C > µ and R = exp(CK), aut(HK) < R with high probability.

Putting together Lemma 1 and Lemma 2, and choosing R and D as specified, we have that
|J (K, R, D)| = (1 −o(1))|J (K, ∞, ∞)| = (α + o(1))−K. Let β denote a universal constant
such that |J | ≤βK. We take β = α−1.

C.2
Algorithm overview

Given (G1, G2) ∼CSBM(n, p, q, s). Our algorithm contains mainly three steps. Firstly, we apply
the algorithm by Mossel, Neeman, and Sly [48] to obtain almost exact community label estimates for
each single graph. Secondly, we calculate the signed chandelier counting [42] based similarity score
to give an almost exact graph matching. Lastly, we boost the almost exact matching to exact matching
by extending the seeded graph matching algorithm [42] on Erd˝os–Rényi graphs to stochastic block
models.

Almost exact community recovery. Obtaining community label estimates for both graph G1 and
G2 is the first step of our algorithm. This is necessary for centralizing the adjacency matrices for the
signed subgraph counts afterwards. We expect the following properties from the community recovery
algorithm:

(a) gives almost exact recovery (down to the information-theoretic threshold, which is s2(a + b) = 1
in the correlated SBMs with two balanced communities [58]);
(b) gives an error rate for each vertex of inverse-polynomial;

(c) gives error rates on different vertices that are approximately independent.

The community recovery algorithm in [48] (described as Algorithm 3) has been shown with property
(a) by [48] and property (b) by [25] with an error rate of n−sD+(a,b). In this paper, we show that
property (c) is satisfied (See Lemma 8).

Subgraph counts. For an arbitrary weighted adjacency matrix M of some adjacency matrix A,
vertex i ∈[n], and a rooted graph H, we define the weighted subgraph counts on M as

Wi,H(M) :=
X

S(i)∼
=H
MS, where MS :=
Y

e∈E(S)
Me,
(7)

where S(i) enumerates through all subgraphs on the complete graph Kn, rooted at i and are isomor-
phic to H.

When M is the adjacency matrix itself, Wi,H(M) is the usual subgraph count, representing the
number of subgraphs rooted at i in M that are isomorphic to H. When M is the centralized adjacency
matrix A := A −E[A], we call Wi,H(M) a signed subgraph count following [9]. However, we
do not have access to E[A] in many cases. Specifically for SBM, we can estimate E[A] through
estimating the community labels. We define the approximately centralized adjacency matrix regarding
to community label estimate bσ, denoted as A
bσ, entry-wise as A
bσ
i,j = Ai,j −p1bσ[i]=bσ[j] −q1bσ[i]̸=bσ[j].

18


---Page Break---
Using A
bσ in (7) yields the weighted subgraph counts for approximately centralized adjacency matrix.
We also refer to this as a signed subgraph count, though errors may exist.

Given a family H of non-isomorphic rooted graphs, we define the subgraph count signature of vertex
i as
W H
i (M) := (Wi,H(M))H∈H.
(8)

Similarity score. Given a pair of correlated SBMs (G1, G2), we define the similarity score between
vertex i on graph G1 and vertex j on graph G2 as a weighted inner product between two signatures:

Φij := ⟨W T
i (A), W T
j (B)⟩:=
X

H∈T
aut(H)Wi,H(A)Wj,H(B),
(9)

where T is the family of chandelier.

When we do not have access to the centralized adjacency matrices A and B, we use community label
estimates bσA and bσB for G1 and G2 correspondingly. We define the similarity score with a slightly
different notation:

Φbσ
ij := ⟨W T
i (A
bσA), W T
j (B
bσB)⟩=
X

H∈T
aut(H)Wi,H(A
bσA)Wj,H(B
bσB).
(10)

Almost exact graph matching. The first part in the analysis is to show that by calculating this
similarity score, with an appropriate thresholding strategy, we can match up (1 −o(1))n vertices
correctly (Theorem 3). The high-level idea is to show that the similarity score distributions are
well-separated between true pairs and fake pairs. We expect the similarity score having the following
properties, under event H:

• For true pairs j = π(i) :

E[Φbσ
iπ∗(i)1H] > 0,
Var[Φbσ
ij1H] = o

E[Φbσ
iπ∗(i)1H]2
,

• For fake pairs j ̸= π(i) :

E[Φbσ
ij1H] = o

E[Φbσ
iπ∗(i)1H]

,
Var[Φbσ
ij1H] = o

 
E[Φbσ
iπ∗(i)1H]2

n2

!

.

Precisely forming bounds for these moments constitutes the main bulk of the paper. Aside from
proving the desired properties of the first and second order moments, we follow the color coding-
based similarity score estimation idea from [42] to analyze an efficient algorithm. The basic idea is
to color the vertices of SBMs using N + 1 colors uniformly at random. Then, we only do signed
counts on vertex sets that are colorful with N + 1 colors. We show that this is an unbiased estimator
and only potentially increase the variance by an additional constant factor in Section F. The result is
stated formally as in Theorem 4.

Algorithm 1 Almost Exact Graph Matching for CSBM

Input: Adjacency matrices A and B for (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s), a constant c, and
mean value µ.
Output: A mapping bπ : I →[n].

1: Run community recovery Algorithm 3 [48] on A and B separately and get label vector bσA, bσB.

2: For each pair of vertex i in A
bσA and vertex j in B
bσB, compute their similarity score as in [42]:

Φbσ
ij = ⟨W T
i (A
bσA), W T
j (B
bσB)⟩=
X

H∈T
aut(H)Wi,H(A
bσA)Wj,H(B
bσB).
(11)

3: Let τ = cµ, output I := {i|i ∈[n], ∃j ∈[n], s.t. ΦT
ij ≥τ, and ∀k ∈[n] \ {j}, ΦT
ik < τ}.

Proof challenge. In regime sD+(a, b) > 1, the probability of existing one vertex being classified
incorrectly is vanishing. Therefore, with high probability, A
bσA = A. If sD+(a, b) < 1, then

19


---Page Break---
the recovered bσ contains errors (polynomial in n), which will cause some edges being centralized
incorrectly and thereby affect the moments calculation. For example, let i, j ∈[n] be two vertices on
G1 who has the same community label σ∗(i) = σ∗(j). If only one of i, j is labeled incorrectly by

Algorithm 3, then the expectation of A
bσA
i,j conditioned on σ∗and bσ is p −q.

This phenomenon poses a challenge to the algorithm analysis. We highlight some key points in the
context of second moment calculation. For simplicity, we ignore event H here.

Var[Φbσ
ij] =
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I


Eσ∗
h
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 | σ∗]
i

−Eσ∗
h
E[A
bσA
S1 B
bσB
S2 | σ∗]
i
Eσ∗
h
E[A
bσA
T1 B
bσB
T2 | σ∗]
i 
.

Let us define the union graph U := S1 ∪S2 ∪T1 ∪T16. If sD+(a, b) > 1, we view A
bσ and B
bσ as
A and B, respectively. E[AS1BS2AT1AT2 | σ∗] ̸= 0 only if there exists no edge e ∈E(U) such
that it occurs only once among (S1, S2, T1, T2). This is because different edges are independent and
centralized conditioned on σ∗. Without loss of generality, we assume there exists an edge e ∈E(A)
occur only on S1, thus E[Ae | σ∗] = 0 and also E[AS1BS2AT1AT2 | σ∗] = 0.

However, in regime sD+(a, b) < 1, E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2
| σ∗] ̸= 0 even when edges are not

occurring multiple times as we have the expectation of A
bσA
e
can be non-zero when conditioning on
σ∗and an estimate bσA that disagrees with σ∗on the edge type (i.e., for e = (u, v), σ∗(u)σ∗(v) ̸=
bσA(u)bσA(v)). This not only causes this cross-moments calculation being more complicated, but
significantly increasing the possibility of the combinations between S1, S2, T1, and T2. The maximum
vertex in the union graph grows from 2N to 4N, squaring up the trivial bound on the number of
subgraphs on the complete graph Kn that is isomorphic to U.

The most important high-level idea to properly bound the moments is: the cross-moment conditioning
on a specific bσ = (bσA, bσB) would be non-trivial if and only if all edges occurring only once
are centralized incorrectly. This is because, conditioning on bσ satisfying the above property, the
expectation of Ae takes either p −q or q −p for all e ∈E(U) that occurs only once. Assuming there
are z edges occurring once, we show that the probability that bσ satisfying this property is no greater

than n−
z(sD+(a,b)−ε log(a/b)/2)

D
for any ε > 0. Intuitively, this is saying that to incorrectly centralize z
edges at the same time, we expect the Algorithm 3 to label at least ⌈z

D⌉vertices incorrectly. It turns

out that we require n−
z(sD+(a,b)−ε log(a/b)/2)

D
to be o(
1
logC n) for some positive constant C so that (1)
and (2) are satisfied.
Theorem 3. Fix a ̸= b > 0 and s ∈[0, 1]. Let p = a log n

n , q = b log n

n
and (G1, G2) ∼
CSBM(n, p, q, s).
For any ε > 0, suppose s2 ≥α + ε. There exists positive constants
C1, C2, C3, C4, C5 > 0 such that the following holds. Pick K, M, L, N, D as

L = C1

ε ,
K = C2 log n,
M =
C3K
log(ns(p ∧q)),
R = exp(C4K),
D = C5
log n
(log log n)2 .

(12)

Pick an arbitrary c ∈(0, 1) and set µ = |T |nNρNσ2N
eff , where σ2
eff := (
σ2
++σ2
−
2
). Then, Algorithm 1
outputs a set I with size (1 −o(1))n and a mapping bπ such that bπ|I = bπ∗|I with high probability.

Algorithm 1 takes quasi-polynomial time. Algorithm 4 in Section F computes an approximated score
in polynomial time and satisfies the following Theorem.

Theorem 4. Theorem 3 continues to hold with the color-coding sampled estimation eΦbσ
ij in place of
Φbσ
ij. Moreover, by Algorithm 4, {eΦbσ
ij}i,j∈[n] can be computed in O(nC) for some constant C ≡C(ε)
depending only on ε, where ε is from (3).

Exact graph matching. The final step of the algorithm is boosting the almost exact matching to
a exact matching. The key idea is exploring the number of common neighbors for two unmatched
vertices with regard to the current matching.

6Note that S1 and T1 are rooted at i, while S2 and T2 are rooted at j. For e = (u, v) ∈E(S1), we say it
occurs in T1 if e ∈E(T1) and it occurs in S2 (resp. T2) if (σ∗(u), σ∗(v)) ∈E(S2) (resp. T2).

20


---Page Break---
Define h(x) = x log x −x + 1. Denote Nπ(i, j) as the number of common neighbors of i and j
under correspondence π. In another word, Nπ(i, j) is the number of vertex v ∈I such that v is a
neighbor of i in A and π(v) is a neighbor of j in B. The high-level idea is that if i and j form a true
correspondence, then, with high probability Neπ(i, j) ≥γ p2+q2

2
s2(n + 2n
3
4 ), under the nice event H.
Therefore, we match up i and j if they have more common neighbors than this threshold. In addition,
we can show that all the remaining vertices will be matched up with high probability. Formally, we
summarize the algorithm as Algorithm 2 and the guarantee as Theorem 5.

Theorem 5. Fix a ̸= b > 0 and s ∈[0, 1]. Let p = a log n

n , q = b log n

n
and (G1, G2) ∼
CSBM(n, p, q, s). Suppose

s2(a + b

2
) ≥1 + ε
and
s2 ≥α + ε,

for some ε > 0. Let γ be the unique solution in (1, ∞) to h(γ) =
3 log n
(n−2)pqs2 . Then, the seeded
matching Algorithm 2 with input bπ and an index set I ⊂[n], |I| = (1 −o(1))n ≥(1 −ε/16)n such
that bπ|I = π∗outputs an exact matching eπ = π∗in O(n3(p + q)2) time with probability 1 −o(1).

Algorithm 2 Seeded Graph Matching [42]

Input: Adjacency matrices A and B for (G1, G2) ∼CSBM(n, p, q, s), p = a log n

n , q = b log n

n
for
some a > 0, b > 0. A mapping bπ : I →[n] with |I| = (1 −o(1))n, parameters p, q, s, and
γ ∈(1, ∞) such that h(γ) =
3 log n
(n−2)pqs2 .

1: Let J = I, and eπ = bπ.
2: while there exists i /∈J and j /∈eπ(J) such that Neπ(i, j) ≥γ p2+q2

2
s2(n + 2n
3
4 ) do
3:
Add i to J and let eπ(i) = j.
4: end while

Output: eπ.

C.3
Putting things together: proof of Theorem 1

The proof of Theorem 1 follows from Theorem 4 and Theorem 5.

Proof of Theorem 1. From Theorem 4, we know that for (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s),
a ̸= b, if s2 ≥α + ε for some ε > 0 then there we can match (1 −o(1))n vertices correctly and
efficiently with high probability.

Take the returned bπ from Algorithm 4 as input of Algorithm 2, then Theorem 5 guarantees that the
final output eπ = π∗with probability 1 −o(1). This completes the proof of Theorem 1.

D
Preliminaries

D.1
Tail bounds

Lemma 3 (Chernoff Bound, Theorem 2.1 of [32]). Let X ∼Binom(n, p) be a binomial random
variable. Then, for all t ≥0,

P(X ≥EX + t) ≤exp

−
t2

2(EX + t/3)


,

P(X ≤EX −t) ≤exp

−t2

2EX


.

Lemma 4 (Multiplicative Chernoff Bound, Theorem 4.4 and Theorem 4.5 of [46]). Let X ∼
Binom(n, p) be a binomial random variable, denote µ = np as the mean. Let h(x) = x log x−x+1.
Then, for all γ ∈(1, ∞),
P(X ≥γµ) ≤exp(−µh(γ)).
For γ ∈(0, 1),
P(X ≤γµ) ≤exp(−µh(γ)).

21


---Page Break---
Lemma 5. [48, 25] Suppose that a < b. Let Y ∼Binom(m+, a log n

n
) and Z ∼Binom(m−, b log n

n
)
be independent. If m+ = (1 + o(1)) n

2 , m−= (1 + o(1)) n

2 , then,

P(Y < Z) = n−D+(a,b)+o(1).

For any ε > 0,
P(Y −Z ≤ε log n) ≤n−(D+(a,b)−ε log(a/b)

2
)+o(1).

This result has been proved in Lemma 8 in [3] and Lemma 3.3 in [25].

D.2
Nice events

When (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s), there are some events that happen with high proba-
bility and our following analysis intuitively relies on the happening of these nice events.

• (Balanced Communities) We denote H := { n
2 −n
3
4 ≤|V +|, |V −| ≤n

2 + n
3
4 }. We
observe that |V +| ∼Binom(n, 1

2) and |V −| = n −|V +|. By Chernoff bound,

P(Hc) = P(|V +| ≥n

2 + n
3
4 ) −P(|V +| ≤n

2 −n
3
4 ) ≤
1
e(1−o(1))√n .

• (Reasonable Large Neighborhood) Let γ = max(a, b), we also denote that

F = {∀v ∈[n], |N(v)| ≤100 max(1, γ) log3 n}.

Lemma 6. As n →∞, we have

P(F) ≥1 −n−O(log2 n).

Proof. Let X ∼Binom(n, γ log n

n ). Fix i ∈[n], conditioned on any σ∗, |N(i)| is stochastically
dominated by X. Therefore,

P(|N(i)| ≥γ log3 n | σ∗) ≤P(X ≥γ log3 n) ≤exp

−
(γ log3 n)2

2γ log n + 2/3γ log3 n



≤exp(−γ log3 n) = n−γ log2 n,

where the second line uses Bernstein’s inequality and the third line holds for any n such that
log2 n > 6. From an union bound, we have

P(F) = E[P(F|σ∗)] ≥1 −n−O(log2 n).

D.3
Community recovery

We make a slight change on the choice of the partition number of the community detection algorithm
proposed by Mossel, Neeman, and Sly [48]. This algorithm gives almost exact recovery with
n1−sD+(a,b)+ε| log(a/b)| vertices labeled incorrectly. After community recovery, we need to match
the two communities in G1 and G2 by applying the community matching Algorithm 3.

Consider G ∼SBM(n, a log n

n
, b log n

n
). Define γ = max{a, b}. For any vertex v ∈[n], we define
the signed neighbor counts of v in G as

majG(v) = σ∗(v)
X

u∈N(v)
σ∗(u).

For any ε > 0, define a set of vertices:

Iε(G) := {v ∈[n] : majG(v) ≤ε log n or |N(v)| ≥γ log3 n}.

Previous results (Lemma 5.1 [25], Proposition 4.3 [48]) have shown that Algorithm 3’s correctness
on [n] \ Iε(G) with a different choice of m and the lower bound of |N(v)| in the bad vertices set
Iε(G). In our work, we first demonstrates that Algorithm 3 on input (G, a, b, ε) correctly classifies
all vertices in [n] \ Iε(G).

22


---Page Break---
Algorithm 3 Almost-exact Community Recovery [48]
Input: Adjacency matrix A on n vertices; parameters a, b, ε > 0.
Output: A community label estimate bσ ∈{−1, +1}n.

1: Choose a positive integer m satisfying (log(εm(2 max(a, b) log2 n)−1) −1)ε/2 > 1. Initialize
two empty sets, W+ and W−.
2: Using the spectral method of [3] to find a community partition of [n], denoted as (U+, U−).
3: Partition [n] into {U1, . . . , Um} uniformly at random.
4: for i ∈[m] do
5:
Using the spectral method of [3] to find a community partition (Ui,+, Ui,−) of G{[n] \ Ui}. If
|Ui,+∆U+| ≥n/2, then swap Ui,+ and Ui,−.
6:
For v ∈Ui, insert v into W+ or W−according to its neighborhood majority (resp., minority)
in Ui,+ ∪Ui,−if a > b (resp. a < b).
7: end for
8: For i ∈W+, set bσ(i) = 1, and for i ∈W−, set bσ(i) = −1. Return bσ.

Lemma 7. Algorithm 3 on input (n, a, b, s) classifies all vertices in [n] \ Iε(G) correctly with high
probability.

The proof directly follows from the proof of Proposition 5.1 in [25], with two remarks. First, although
the maximum size of neighbors |N(i)| we consider here is enlarged from 100 max{1, γ} log n
to γ log3 n, we adjust the condition of m accordingly such that the tail bound holds. Secondly,
we need to justify that with high probability, all partitions done by the spectral method are still
almost exactly correct under the new choice of m, which is no longer a constant independent of
n. Theorem 3.2 of [3] showed that the vanilla spectral method achieved optimal error rate, in
the sense that E[ 1

n
Pn
i=1 1{σ∗(i)̸=bσ(i)}] ≤n−(1+o(1))D+(a,b). This implies that with probability
1 −n−(1+o(1))D+(a,b), the spectral method labels all but o(n) vertices correctly. Furthermore, for all
i ∈[n], Ui,+ matches with V + \ Ui and Ui,−matches with V −\ Ui on all but o(n) vertices after
step (5) with high probability.

In this work, we further determine the probability of a set of vertices being in the set Iε(G), which is
a generalization of the result on the P(v ∈Iε) for an arbitrary v ∈[n] (Lemma 5.3 of [25]).

Lemma 8. Given a random graph G ∼SBM(n, sa log n

n
, sb log n

n
) and a fixed subgraph induced by
vertex set S ∈[n].

If |S| = O(log n), then for any ε > 0, δ > 0,

P({∀i ∈V (S), i ∈Iε} ∩H) = O(n−|S|(sD+(a,b)−ε(1+δ)| log(a/b)|) + n−εδ(1−o(1)) log n).

If |S| = o(log n), then for any ε > 0, δ > 0,

P({∀i ∈V (S), i ∈Iε} ∩H) = O(n−|S|(sD+(a,b)−ε(1+δ)| log(a/b)|)).

Proof. The main idea is considering the intersection of the interested event {∀i ∈S, i ∈Iε} ∩H
with G = {∀i ∈S, |NS(i)| ≤εδ log n}, where NS(v) denotes the set of neighbors of v restricted on
the vertex set S.

P({∀i ∈S, i ∈Iε} ∩H) ≤P({∀i ∈S, i ∈Iε} ∩H ∩F ∩G) + P(Gc) + P(Fc).
(13)

Firstly, we give the upper bound of P(Gc | σ∗)1H. Assume that |S| ≥εδ log n without loss of
generality. By using an union bound,

P(G) ≤|S| × E[P(|NS(i)| > εδ log n | σ∗)]

= |S| ×

|S|
X

k=δ log n


|S|
εδ log n


(a log n

n
)εδ log n(1 −b log n

n
)|S|−εδ log n

= n

log log n

log n (Ca log2 n

n
)εδ log n = O(n−εδ(1−o(1)) log n),
(14)

23


---Page Break---
where the second equation holds from the assumption of |S| = O(log n).

Secondly, we study this event E = {∀i ∈S, i ∈Iε} ∩H ∩F.

P(E) = E[P({∀v ∈S, majG(v) ≤ε log n} | σ∗)1H]
= E[P({∀v ∈S, majG[[n]\S](v) + majG[S](v) ≤ε log n} | σ∗)1H]

≤E[P({∀v ∈S, majG[[n]\S](v) ≤ε(1 + δ) log n} | σ∗)1H].

For any v ∈S, majG[[n]\S](v) is the difference of two independent binomial random variables Yv and

Zv, where Yv ∼Binom(|V σ∗(v)
G[[n]\S]|, sa log n

n ), Zv ∼Binom(|V −σ∗(v)
G[[n]\S]|, sb log n

n ). With H happening,

we have |V σ∗(v)
G[[n]\S]| = (1 −o(1)) n

2 , |V −σ∗(v)
G[[n]\S]| = (1 −o(1)) n

2 . Since Yv and Zv do not take into
account v ∈S, they are independent for all v ∈S. Therefore,

P(E) = E[Πv∈SP({Yv −Zv ≤ε(1 + δ) log n} | σ∗)1H]

≤(n−(sD+(a,b)−ε(1+δ)| log(a/b)|

2
)+o(1))|S|

≤n−|S|(sD+(a,b)−ε(1+δ)| log(a/b)|),
(15)

where the second line holds by Lemma 5 and the last line holds for sufficiently large n. We conclude
with by Lemma 6, Inequality (14), and Inequality (15) into Inequality (13).

Remark 2. For arbitrary ε > 0, we can find ε′ > 0 and δ > 0 such that ε′(1 + δ) = ε. For the sake
of convenience, we also denote D+(a, b, s, ε) as sD+(a, b) −ε| log(a/b)| and this is equivalent as
the (ε, δ)-parameterization in Lemma 8. We mainly use the sD+(a, b, s, ε) notation in the analysis
throughout this paper.

D.4
Tree node assigning

Before getting to the first moment calculation of the similarity score, we introduce a sub-problem,
named in-community edge counting for node assignment on trees.

Assume that we have a random graph G with n vertices, which are labeled by a community label
vector σ. We are also given a rooted tree T(i) with N vertices other than the root, where i specifies
the root node of T on G. Planting T(i) onto G has at least
  n
N

possible positions. If G is a stochastic
block model, different planted positions of T would contain different numbers of in-community
edges. We are interested in the distribution of the number of in-community edges when planting T(i)
onto G uniformly at random.

Lemma 9. Let Kn be the complete graph of a stochastic block model G on n vertices. Let σ∗=
{σ∗(i)}n
i=1, σ∗(i) ∈{−1, +1} drawn independently and uniformly at random as the vector of
community labels. Let v ∈[n] be an arbitrary node of Kn. Let T be an arbitrary rooted tree with N
vertices other than the root. Consider a uniformly random injective function τ : V (T) →V (Kn)
such that root r(T) is mapped to v in V (Kn). Define X as the random variable representing the
number of in-community edges in the tree T under random τ. Then,

Bin(N, 1

2 −2n−1

4 ) ≼X | H ≼Bin(N, 1

2 + 2n−1

4 ).

Proof. There are N edges on the tree T. For any (u, v) ∈V (Kn) × V (Kn), we denote X(u,v)
as the indicator random variable of event E ={u and v are from the same community}. Since
1
2 −2n−1

4 ≤P(E | H) ≤1

2 + 2n−1

4 , X(u,v) | H stochastically dominates Bernoulli( 1

2 −2n−1

4 )
and is stochastically dominated by Bernoulli( 1

2 + 2n−1

4 ). Thus, the summation over all edges on
tree T, X | H = P
(u,v)∈E(T ) X(u,v) | H, stochastically dominates Binom(n, 1

2 −2n−1

4 ) and is

stochastically dominated by Binom(n, 1

2 + 2n−1

4 ).

Remark 3. If N = o(n
1
4 ), Lemma 9 implies that with some non-negative integer N1 ≤N, P({X =
N1} ∩H) = (1 + o(1))
  N
N1

( 1

2)N. Because P({X = N1} ∩H) = (1 −o(1))P(X = N1 | H), it

24


---Page Break---
suffices to show both the upper and lower bound on P(X = N1 | H).

P(X = N1 | H) = P(X ≥N1 | H) −P(X ≥N1 + 1 | H)

≤
X

t≥N1

N
t


(1

2 + 2n−1

4 )t(1

2 −2n−1

4 )N−t −
X

t≥N1+1

N
t


(1

2 −2n−1

4 )t(1

2 + 2n−1

4 )N−t

≤(1 + o(1))
 N
N1


(1

2)N +
X

t≥(N−N1)∨(N1+1)

N
t


(1

2)Nf(n, N, t),

where f(n, N, t) = (1+4n−1

4 )t(1−4n−1

4 )N−t −(1−4n−1

4 )t(1+4n−1

4 )N−t. The first inequality
holds because of the stochastic dominance in Lemma 9. The second inequality holds because
  N
N−t

f(n, N, N −t) =
 N
t

f(n, N, t) and thus cancels out every term in the summation indexed
from t = N1 + 1 to t = N −(N1 + 1). For t ≥(N −N1) ∨(N1 + 1), we know that
 N
t

<
  N
N1

.
Also, f(n, N, t) ≤(1 + 4n−1

4 )N −(1 −4n−1

4 )N < O(n−1

4 N) = o( 1

N ) from our assumption on
N. Therefore, summing over t ≥(N −N1) ∧(N1 + 1), we have

P(X = N1 | H) ≤(1 + o(1))
 N
N1


(1

2)N.

From the other direction of stochastic dominance, we have P(X = N1 | H) = P(X ≥N1 |
H) −P(X ≥N1 + 1 | H) ≥(1 −o(1))
  N
N1

( 1

2)N.

Lemma 10. Let Kn be the complete graph of a stochastic block model G on n vertices. Let
σ∗= {σ∗(i)}n
i=1, σ∗(i) ∈{−1, +1} drawn independently and uniformly at random as the vector
of community labels. Let v ∈[n] be an arbitrary node of G. Let {T}t
i=1 be a sequence of arbitrary
rooted tree with N vertices other than the root. Consider a uniformly random injective function
τ : V (T) →V (Kn) such that root r(T1) of the first tree is mapped to v in V (Kn). Also, the root of
i-th tree is mapped to τ(r(Ti)) = τ(u) for a fixed vertex on previous trees u ∈∪i−1
t=1V (Tt) or a fixed
vertex on Kn. Define X as the random variable representing the number of in-community edges in
all the trees under random τ. Then,

Bin(N, 1

2 −2n−1

4 ) ≼X | H ≼Bin(N, 1

2 + 2n−1

4 ).

Proof. Define Xi as the random variable for number of in-community edges on Tree Ti with random
embedding τ. The following holds immediately from Lemma 9,

Bin(NTi, 1

2 −2n−1

4 ) ≼Xi | H ≼Bin(NTi, 1

2 + 2n−1

4 ).

Since X | H = P
i Xi | H,

Bin(N, 1

2 −2n−1

4 ) ≼X | H ≼Bin(N, 1

2 + 2n−1

4 ).

To present the following Lemma 11, we need to introduce a few more definitions: decorated union
graph and decorated edges, which will be explained in more details in the context of chandelier in
Section I.

Decorated Union Graph. Let S1, S2, T1, T2 be four rooted graphs. The union graph is defined as
U := S1 ∪S2 ∪T1 ∪T2. We define the decorated union graph as a two-tuple ˙U := (U, DU), which
is U associating with a decoration set. For each edge,

DU(e) =
The subset of {S1, S2, T1, T2} where eoccurs,
if e ∈E(U),
∅,
otherwise.

We call an edge e ∈E(U) is t-decorated if |DU(e)| = t for t ∈{0, 1, 2, 3, 4}. Decorated union
graph ˙U has one-to-one correspondence with (S1, S2, T1, T2) as we can uniquely determine ˙U given
(S1, S2, T1, T2) and uniquely recover (S1, S2, T1, T2) given ˙U.

25


---Page Break---
Lemma 11 (Asymptotic independence of the counts of i-decorated in-community edges). Let Kn be
the complete graph of a stochastic block model G on n vertices. Consider a connected decorated
union graph ˙UP with d1 1-decorated edges, d2 2-decorated edges, d3 3-decorated edges, and d4
4-decorated edges, rooted at v on the complete graph Kn. Consider a uniformly random injective
function τ : V (T) →V (Kn) such that root r(T) is mapped to v in V (Kn). Define X(i) as the
random variable representing the number of i-decorated in-community edges in ˙UP . Assume that
|V ( ˙UP )| = O(log n) and ˙UP has excess k. If k = −1, then,

P(X(1) = M1, X(2) = M2, X(3) = M3, X(4) = M4 | H)

= (1 ± o(1))P(X(1) = M1 | H)P(X(2) = M2 | H)P(X(3) = M3 | H)P(X(4) = M4 | H)

= (1 + o(1))
 d1
M1

 d2
M2

 d3
M3

 d4
M4


1
2d1+d2+d3+d4 .

Proof. From the assumption k = −1 we have ˙UP is a tree. We can decompose UP to a sequence of
trees as follows: Traverse UP in BFS order and include each maximal connected component with all
edges i-decorated as a subtree.

In addition, we break this sequence of tree into four sequences based on the decoration number:
{T (1)
j
}c
j=1, {T (2)
j
}d
j=1, {T (3)
j
}e
j=1, {T (4)
j
}f
j=1. In the random mapping, the root of each tree should

be mapped to either v or a non-root vertex on the other tree. Let X(i)
j
be the random variable for

the number of in-community edges for the j-th tree of i-decoration. X(i)
j
| H satisfies the following
stochastic dominance:

Bin(|V (T (i)
j )|, 1

2 −2n−1

4 ) ≼X(i)
j
| H ≼Bin(|V (T (i)
j )|, 1

2 + 2n−1

4 ).

Thus, their summation satisfies the following stochastic dominance:

Bin(di, 1

2 −2n−1

4 ) ≼X(i) | H ≼Bin(di, 1

2 + 2n−1

4 ), i = 1, 2, 3, 4.

As each of the series of trees occupy O(log n) vertices, every X(i) conditioned on other X(i′), i′ ̸= i
still satisfies the stochastic dominance:

Bin(di, 1

2 −2n−1

4 ) ≼X(i)|X(i′), H ≼Bin(di, 1

2 + 2n−1

4 ).

Specifically, following Remark 3, P(X(4) = M4 | H) = (1 + o(1))
  d4
M4
 1

2d4 , P(X(3) = M3|X(4) =
M4, H) = (1 + o(1))
  d3
M3
 1

2d3 , P(X(2) = M2|X(3) = M3, X(4) = M4, H) = (1 + o(1))
  d2
M2
 1

2d2 ,
and P(X(1) = M1|X(2) = M2, X(3) = M3, X(4) = M4, H) = (1 + o(1))
  d1
M1
 1

2d1 . Collectively,
these form the asymptotic independence of X(i):

P(X(1) = M1, X(2) = M2, X(3) = M3, X(4) = M4 | H)

= (1 + o(1))
 d1
M1

 d2
M2

 d3
M3

 d4
M4


1
2d1+d2+d3+d4 .

Lemma 11 discusses the case of k = −1. We do not expect the same property holds for k ≥0 but we
have an auxiliary result as in the following Corollary 1.

Corollary 1. (If k ≥0, then ˙UP contains cycles.) By definition, ˙UP can be decomposed into a tree
TP with vP vertices other than the root and additional k + 1 distinct edges Ek+1 = {(uj, vj)}k+1
i=j
fixed. By edges’ decorations, Ek+1 = E(1)∪E(2)∪E(3)∪E(4). Let X(i) still be the random variable
representing the number of i-decorated in-community edges on ˙UN. If k ≥0, let X(i,a) be the counts
of i-decorated in-community edges on TN and X(i,b) be the counts of i-decorated in-community
edges on {ei}k+1
i=1 . From construction, we know X(i) = X(i,a) + X(i,b). Correspondingly, there are
Ai i-decorated edges on tree, Bi = |E(i)|, and di = Ai + Bi, which are all fixed from ˙UP .

Then, P(X(1,a) = a1, X(2,a) = a2, X(3,a) = a3, X(4,a) = a4, X(2,b) = b2, X(1,b) = b1, X(3,b) =
b3, X(4,b) = b4 | H) ≤(1 + o(1))
 A1
a1
 A2
a2
 A3
a3
 A4
a4

1
2A1+A2+A3+A4 .

The proof follows straightforwardly from the proof of Lemma 11.

26


---Page Break---
s
r(T (4)
1
)

s
r(T (3)
1
)

s
r(T (4)
2
)

r(T (2)
3
)

s
r(T (3)
2
)

s
r(T (2)
2
)

s
r(T (4)
3
)

s
r(T (4)
4
)

s
r(T (2)
3
)

Figure 5: Decomposition of a decorated tree into three sequences of trees. Edges that are 2, 3,
4-decorated are painted as red, green, and blue color correspondingly. Roots of each subtree is
marked by larger node and annotated as r(T (i)
j ), where i is the decoration counts and j is the order in
its sequence.

D.5
Inequalities concerning automorphisms

In this subsection, we study how adding edges can change the number of rooted automorphisms of a
rooted tree. With a slight abuse of notation, we denote by aut(T) the number of rooted automorphisms
(i.e., automorphisms which fix the root) of a rooted tree T. We start by quantifying the effect of
adding an additional edge.
Lemma 12. Let T = (V, E, r) be a rooted tree with root r. Let T ′ := (V ∪{u}, E ∪{(u, v)}, r)
be the rooted tree obtained from T by adding a vertex u and the edge (u, v), where v ∈V . The
following bounds hold:

aut(T) ×
1
|V | −1 ≤aut(T ′) ≤aut(T) × |V |,
(16)

where |V | is the number of vertices in T.

Proof. Since T is a rooted tree, there is a natural notion of a parent vertex for every vertex other than
the root. Namely, the parent of a vertex v ̸= r is the neighbor of v which is closest to the root r. We
also define the subtree rooted at v to be the subtree of T induced by all vertices whose shortest path
to r goes through v, rooted at v.

We partition V into equivalence classes according to the following rule: v1 and v2 are in the same
equivalence class if and only if they share the same parent and the subtrees rooted at v1 and v2 are
isomorphic. We denote the resulting partition as {Vi}i∈I and refer to the equivalence classes as
orbits. Note that the root r is always in a single-element orbit and thus the size of each orbit is at
most |V | −1.

Observe that a permutation of the vertices is a rooted automorphism precisely when it maps each
vertex to a vertex in its orbit. Thus we have that

aut(T) =
Y

i∈I
|Vi|!.
(17)

Now consider T ′, which adds a new vertex u to T with an edge connecting u to a vertex in T. By (17),
in order to understand aut(T ′), we need to understand how the orbits and their sizes change due to
the addition of the new vertex and edge. The new vertex u will either join an existing orbit or form its
own one. The parent of u may change orbits, so might the parent of its parent, etc. In other words,
the vertices on the path from u to the root r might change their orbit, but vertices not on this path will
not. In the following, we argue iteratively based on the depth of u in T ′ (i.e., its distance from the
root). When considering the upper bound, we will ignore the possible size decrease of orbits. When
considering the lower bound, we will ignore the possible size increase of orbits.

27


---Page Break---
Assume first that the new vertex u is attached to the root r. If there are no leaves except for u
connecting to r, then u forms a new orbit V|I|+1 whose size is 1, which does not change the number
of rooted automorphisms. Otherwise, without loss of generality, assume that orbit V1 is the set that
contains all leaves connected to r. Then u will join this orbit, so the set of orbits of T ′ is given by
V1 ∪{u} and {Vi}|I|
i=2. Thus, we have that aut(T ′) = aut(T)(|V1| + 1), so in particular

aut(T) ≤aut(T ′) = aut(T)(|V1| + 1) ≤aut(T) × |V |.

Now suppose that u has depth 2 in T ′, and let v(1) denote the parent of u in T ′. Without loss of
generality, assume that v(1) ∈V1. There are again two cases depending on whether or not there
are leaves attached to v(1) in T. Suppose first that there are not any leaves attached to v(1) in T.
Then u has its own orbit (of size 1) in T ′. The orbit of v(1) changes from V1 in T to either a new
orbit or some existing orbit Vj1 in T ′ (for some 2 ≤j1 ≤|I|). In the former case we have that
aut(T ′) = aut(T)/|V1|, while in the latter case we have that

aut(T ′) = aut(T) × |Vj1| + 1

|V1|
.

The desired inequalities thus follow since each orbit has size at most |V | −1. Now suppose that
there are leaves attached to v(1) in T and let V2 denote the equivalence class of these vertices.
Then, u joins the orbit V2 in T ′. The orbit of v(1) again changes from V1 in T to either a new
orbit or some existing orbit Vj1 in T ′ (for some 3 ≤j1 ≤|I|). In the former case we have that
aut(T ′) = aut(T) × (|V2| + 1)/|V1|, while in the latter case we have that

aut(T ′) = aut(T) × (|V2| + 1)(|Vj1| + 1)

|V1|
.

The lower bound follows since |V1| ≤|V | −1. For the upper bound in the latter case, note that
by the definition of orbits, in T ′ there are |Vj1| + 1 nodes at depth 1 who each have |V2| + 1
children that are leaves. This implies that T ′ has at least (|V2| + 1)(|Vj1| + 1) non-root vertices, so
(|V2| + 1)(|Vj1| + 1) ≤(|V | + 1) −1 = |V |.

The general case when u has depth ℓin T ′ is analogous. Let v(ℓ−1) denote the parent of u in T ′, let
v(ℓ−2) denote the parent of v(ℓ−1), etc. Without loss of generality, let Vi denote the orbit of v(i) in T,
for i ∈[ℓ−1]. Suppose that v(ℓ−1) has children that are leaves in T (the other case, when it does not,
is similar and simpler), and let Vℓdenote the equivalence class of these vertices. Then, using similar
observations as above, we obtain the following upper and lower bounds on aut(T ′):

aut(T) ×
1
Qℓ−1
i=1 |Vi|
≤aut(T ′) ≤aut(T) × (|Vℓ| + 1) ×

ℓ−1
Y

i=1
(|Vji| + 1) .

Here, for every i ∈[ℓ−1], either ji ∈[ℓ+ 1, |I|] (which corresponds to v(i) changing from Vi
in T to some existing orbit Vji in T ′) or |Vji| = 0 (which corresponds to v(i) changing from Vi
in T to a new orbit in T ′). To conclude the lower bound, observe (using the definition of orbits)
that T contains a subtree consisting of the root and Qℓ−1
i=1 |Vi| additional vertices, so Qℓ−1
i=1 |Vi| ≤
|V | −1. For the upper bound, observe similarly that T ′ contains a subtree consisting of the root and
(|Vℓ| + 1) × Qℓ−1
i=1 (|Vji| + 1) additional vertices, so (|Vℓ| + 1) × Qℓ−1
i=1 (|Vji| + 1) ≤|V |.

Lemma 12 is tight. Let T be a star rooted at its center (i.e., a tree where all vertices except the root are
connected to the root). Then all permutations of the vertices that fix the root are rooted automorphisms,
so aut(T) = (|V (T)| −1)!. Now let T ′ be a tree obtained from T by adding an additional child to
the root (see Figure 6, tree in the middle). Then aut(T ′) = (|V (T)|)! = aut(T) × |V (T)|. This
shows that the upper bound in Lemma 12 is tight. Now let T ′′ be a tree obtained from T by adding a
child to one of the leaves of T (see Figure 6, tree on the right). The rooted automorphisms of T ′′ are
precisely the permutations of the vertices that permute the neighbors of the root which are leaves, so
aut(T ′′) = (|V (T)| −2)! = aut(T)/(|V (T)| −1). This shows that the lower bound in Lemma 12
is tight.

From Lemma 12, we can derive the following corollary by an iterative argument.

28


---Page Break---
s
s
s

Figure 6: Left: The simplest example of two rooted trees T1 and T2 satisfying
p

aut(T1)aut(T2) >
aut(T1 ∪T2); here T1 is induced by the blue and green edges, T2 is induced by the blue and red
edges, and both trees are rooted at the vertex at the top. Middle: Example that shows that the upper
bound in Lemma 12 is tight, where blue lines and black vertices represent T, and the red vertex is
additionally added with an edge attaching it to the root. Right: Example that shows that the lower
bound in Lemma 12 is tight.

Corollary 2. Let T1 = (V1, E1, r) and T2 = (V2, E2, r) be two trees rooted at the same vertex r.
Define the union T1 ∪T2 := (V1 ∪V2, E1 ∪E2, r) by taking the union of the two vertex sets (both
of which contain r) and the union of the two edge sets, with multiple edges ignored (i.e., if an edge
appears in both E1 and E2, then it appears in E1 ∪E2 exactly once). Suppose that T1 ∪T2 is also a
tree. Let d := |E1△E2| denote the size of the symmetric difference of the edge sets. Then
p

aut(T1)aut(T2) ≤aut(T1 ∪T2) × (2 max{|V1|, |V2|})d.
(18)

Proof. By the assumptions on T1 and T2, the union T1 ∪T2 has (|V1| + |V2| + d)/2 −1 edges.
There are two natural ways that we can think of T1 ∪T2. First, we can start from T1 and add
(|V2| −|V1| + d)/2 new vertices—those that are in V2 but not in V1—one at a time, together with a
new edge for each new added vertex, connecting it to an existing vertex, to obtain T1 ∪T2. With this
perspective, applying the lower bound in (16) from Lemma 12 across each of the (|V2| −|V1| + d)/2
steps, we obtain that

aut(T1) ≤aut(T1 ∪T2)

(|V1|+|V2|+d)/2−2
Y

k=|V1|−1
k.
(19)

On the other hand, we can equally well start from T2 and add (|V1| −|V2| + d)/2 new edges and
vertices to obtain T1 ∪T2. Thus, analogously, (19) also holds with aut(T1) on the left hand side
replaced with aut(T2), and |V1| −1 on the right hand side (the minimum value of k in the product)
replaced with |V2| −1.

Combining these two inequalities, we obtain that

p

aut(T1)aut(T2) ≤aut(T1 ∪T2) ×

(|V1|+|V2|+d)/2−2
Y

k=min{|V1|,|V2|}−1
k.

Since d ≤2(max{|V1, V2|} −1), it follows that (|V1| + |V2| + d)/2 −2 ≤2 max{|V1|, |V2|}, so
all factors in the product above are at most 2 max{|V1|, |V2|}. Note also that d ≥max{|V1, V2|} −
min{|V1, V2|}, so the number of factors in the product above is (max{|V1, V2|} −min{|V1, V2|} +
d)/2 ≤d. Putting these observations together we see that (18) holds.

Remark 4. The leftmost example in Figure 6 provides an example showing that the right hand side
of (18) would no longer be an upper bound without the factor (2 max{|V1|, |V2|})d.

D.6
Auxiliary result: bounds on the cross-moments

In this section, we summarize the upper bound on cross-moments. In Lemma 13, we work on the
regime sD+(a, b) > 1 where we can recover the community label exactly first. In Lemma 14, we
work on the regime sD+(a, b) < 1 where we we have a inverse polynomial fraction of vertices being
labeled incorrectly.

29


---Page Break---
Lemma 13. Let (G1, G2) ∼CSBM(n, p, q, s), p =
a log n

n
, q =
b log n

n
. Denote A, B as the
centralized adjacency matrices of G1 and G2 correspondingly. Let e := (u, v) ∈[n] × [n] be an
arbitrary edge. Define the edge type indicator c(e) = + if e is an in-community edge and c(e) = −
if e is a cross-community edge under σ∗. The following holds for 0 ≤ℓ, m ≤2, 2 ≤ℓ+ m ≤4, and
any σ∗,

βℓ,m(e) := σ−(ℓ+m)
c(e)
E[A
ℓ
eB
m
e | σ∗] ≤












1
(ℓ, m) = (2, 0), (0, 2)
(1 + Θ( log n

n ))ρ
(ℓ, m) = (1, 1)
1
√

s(p∧q)
(ℓ, m) = (2, 1), (1, 2)

1
s(p∧q)
(ℓ, m) = (2, 2).

Proof. Conditioning on a specific true community label vector that satisfies the balanced community
event H, we can apply Lemma 5 from [43] for each case of c(e) = + and c(e) = −. Then, the upper
bound are different in terms of p, q and ρ+, ρ−. When (ℓ, m) = (1, 1), βℓ,m ≤max{|ρ+|, |ρ−|} ≤
(1 + Θ( log n

n ))ρ. When (ℓ, m) = (2, 1) or (1, 2), if c(e) = +, then we have βℓ,m ≤
1
√sp, else

we have βℓ,m ≤
1
√sq. Therefore, βℓ,m ≤
1
√

s(p∧q). The same argument holds for the case when

(ℓ, m) = (2, 2).

Lemma 14. Let (G1, G2) ∼CSBM(n, p, q, s), p = a log n

n
, q = b log n

n
. Denote A
bσA, B
bσB as the
approximately centralized adjacency matrices of G1 and G2 correspondingly. Let e := (u, v) ∈
[n] × [n] be an arbitrary edge. Define the edge type indicator c(e) = + if e is an in-community edge
and c(e) = −if e is a cross-community edge under σ∗. Define ∆:= |sp −sq|. The following holds
for 0 ≤ℓ, m ≤2, 1 ≤ℓ+ m ≤4, and any σ∗, bσ = (bσA, bσB),

ηℓ,m(e) := σ−(ℓ+m)
c(e)
E[A
bσA,ℓ
e
B
bσB,m
e
| σ∗, bσ] ≤


















Θ(
√

∆)
(ℓ, m) = (0, 1), (1, 0)
1 + Θ( log n

n )
(ℓ, m) = (2, 0), (0, 2)
ρ(1 + Θ( log n

n ))
(ℓ, m) = (1, 1)
1
√

s(p∧q)(1 + Θ( log n

n ))
(ℓ, m) = (2, 1), (1, 2)

1
s(p∧q)(1 + Θ( log n

n ))
(ℓ, m) = (2, 2).

Proof. Denote the two vertices on e as u and v. We denote p′ := sp and q′ := sq.

(a) ℓ+ m = 1. Without loss of generality, we consider ℓ= 1 and m = 0.

E[A
bσA,1
e
| σ∗, bσ] =
0,
σ∗(u)σ∗(v) = bσA(u)bσA(v),
±∆,
σ∗(u)σ∗(v) ̸= bσA(u)bσA(v).

When E[A
bσA,1
e
| σ∗, bσ] is non-zero, σ−1
c(e)E[A
bσA,1
e
| σ∗, bσ] = Θ(
√

∆).

(b) ℓ+ m = 2. We first consider the case of ℓ= 2 or m = 2.

We explicitly calculate the expectation on the following two cases. If σ∗(u)σ∗(v) = bσ(u)bσ(v),

then E[A
bσA,2
e
| σ∗, bσ] = p′(1 −p′) = σ2
c(e). If σ∗(u)σ∗(v) ̸= bσ(u)bσ(v), then E[A
bσA,2
e
| σ∗, bσ] =
p′ −2p′q′ + q′2 = σ2
c(e) + ∆2.

In summary, we have

E[A
bσA,2
e
| σ∗, bσ] = E[B
bσB,2
e
| σ∗, bσ] ≤σ2
c(e) + ∆2 = σ2
c(e)(1 + Θ(log n

n
)).

We next consider the case of ℓ= m = 1. If σ∗(u)σ∗(v) = bσ(u)bσ(v), then E[A
bσA,1
e
B
bσB,1
e
|

σ∗, bσ] = ρc(e)σ2
c(e). If σ∗(u)σ∗(v) ̸= bσ(u)bσ(v), then E[A
bσA,1
e
B
bσB,1
e
| σ∗, bσ] = ρc(e)σ2
c(e) + ∆2.

In summary, we have

E[A
bσA,1
e
B
bσB,1
e
| σ∗, bσ] ≤ρc(e)σ2
c(e) + ∆2 = ρσ2
c(e)(1 + Θ(log n

n
))

30


---Page Break---
(c) ℓ+ m = 3. With loss of generality, we assume that this edge connects two vertices from different
communities. Conditioned on correct centralization, we can compute

E[A
bσA,2
e
B
bσB,1
e
| σ∗, bσ] = E[A
2
eBe | σ∗] = qs2(1 −q)(1 −2qs) = σ3
−
ρ−(1 −2q′)
p

q′(1 −q′)
≤σ3
−
1
√q′ ,

which is the same as in Lemma 13. If the centralization is incorrect for both graphs, the expectation
changes to the following:

E[A
bσA,2
e
B
bσB,1
e
| σ∗, bσ] = q′s −p′q′ −2p′q′s + 3p′2q′ −p′3.

From observation, we see that the dominant part in both cases are the same, which is q′s = Θ( log n

n ).
Then, we can show that the difference between two quantities are minor:

E[A
bσA,2
e
B
bσB,1
e
| σ∗, bσ] −E[A
2
eBe | σ∗]
σ3
−
= (p′ −q′)(p′2 −2p′q′ −2q′2 + q′ + 2q′s)

q′(1 −q′)
p

q′(1 −q′)
= Θ(
p

q′).

(20)
The above (20) also holds for the case when e is classified correctly in A or B only.

By also considering the other case, that is, this edge is connects two vertices from the same community
but wrongly centralized according to bσ, we conclude with

E[A
bσA,2
e
B
bσB,1
e
| σ∗, bσ] ≤σ3
c(e)
1
p

s(p ∧q)
(1 + Θ(s(p ∧q))) = σ3
c(e)
1
p

s(p ∧q)
(1 + Θ(log n

n
))

(d) ℓ+ m = 4. With loss of generality, we assume that this edge connects two vertices from different
communities. For the correct centralization, the moment stays the same as in Lemma 13,

E[A
2
eB
2
e | σ∗] = q′s−4q′2s+4q′3s+2q′3 −3q′4 = q′2(1−q′)2 +q′(1−q′)ρ−(1−2q′)2 ≤σ4
−
1
q′ .

If the centralization is incorrect for both graphs, we have

E[A
bσA,2
e
B
bσB,2
e
| σ∗, bσ] = q′s −4p′q′s + 4p′2q′s + 2p′2q′ −4p′3q′ + p′4,

and we can show that the error has the following order

E[A
bσA,2
e
B
bσB,2
e
| σ∗, bσ] −E[A
bσA,2
e
B
bσB,2
e
| σ∗] ≤Θ((log n

n
)2) = Θ(σ4
c(e)).
(21)

When the centralization is incorrect for only one graph, (21) still holds.

By also considering the other case, that is, this edge is connects two vertices from the same community
but wrongly centralized according to bσ, we conclude with

E[A
bσA,2
e
B
bσB,2
e
| σ∗, bσ] ≤Θ(σ4
c(e)) + σ4
c(e)
1
s(p ∧q) ≤σ4
c(e)
1
s(p ∧q)(1 + Θ(log n

n
)).

E
Proof of Theorem 3: almost exact graph matching

Fix constants a ̸= b > 0, s ∈[0, 1]. Throughout the paper, we refer to sD+(a, b) > 1 as Regime I

and sD+(a, b) < 1 as Regime II. We define µ := |T |nNρNσ2N
eff , where σ2
eff =
σ2
++σ2
−
2
. Depending
on the different parameter regimes, the analysis is different. We first present the first and second
moment bounds for both regimes and then prove Theorem 3.

E.1
Regime I: Exact community recovery is possible for a single graph

Proposition 1 (Mean calculation, sD+(a, b) ≥1). Given (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s),
the similarity score satisfies,

E[Φij1H] =
(1 + o(1))µ,
if π∗(i) = j,
0,
if π∗(i) ̸= j.

31


---Page Break---
Proposition 2 (Variance calculation–True pairs, sD+(a, b) > 1). Suppose that j = π∗(i), that
sD+(a, b) > 1, and that

14L2

ρ2(K+M)(|J |) ≤1

2,
22R4(2N + 1)(11β)2(K+M)

n
≤1

2,

4R
4
M (11β)
4K+4M

M
ns(p ∧q)
≤1

2,
1 + 2L2

ns(p ∧q) ≤1

4.
(22)

Then, for any i ∈[n], we have

Var[Φij1H]
E[Φiπ∗(i)1H]2 = O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


.

Proposition 3 (Variance calculation–Fake pairs, sD+(a, b) > 1). Suppose that j ̸= π∗(i), that
sD+(a, b) > 1, and that

4L+4L2L∧(4K+2)(11β)8(K+M)R4(2N + 1)3 ≤n

2 ,
4R
2
M (11β)
4(K+M)

M
ns(p ∧q)
≤1

2.
(23)

Then, for any i ∈[n], we have

Var[Φij1H]
E[Φiπ∗(i)1H]2 = O(
1
|T |ρ2N ).

E.2
Regime II: Exact community recovery is impossible for a single graph

Before getting into the details of proof, let us first give more intuition on the approximately centralized
adjacency matrix.

In this regime, for each element in the adjacency matrix, there are four cases: (1) Correct central-
ization for in-community edges; (2) Incorrect centralization for in-community edges; (3) Correct
centralization for cross-community edges; (4) Incorrect centralization for cross-community edges.
Figure 7 gives an example of how the graphon for balanced 2-community SBM changes under
community label misclassifications.

(a) Regime I: exact recovery is possible.
(b) Regime II: exact recovery is impossible.

Figure 7: Illustration on the impact of incorrect community labels on adjacency matrix centralization
with G ∼SBM(n, 0.3, 0.1). Centralized adjacency matrix A
bσA = A −E[A], where matrix E[A]
takes the value of discretized graphon as above.

Proposition 4 (Mean calculation, sD+(a, b) < 1). Let (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s).
Given (K, L, M, R, D)-Chandelier class T . Assume that D = o(
log n
log log n).

For all i ∈[n] with j = π∗(i), we have

E[Φbσ
iπ∗(i)1H] = (1 + o(1))µ.

32


---Page Break---
For all i ∈[n] with j ̸= π∗(i), we have

E[Φbσ
ij1H] = o(µ).

Proposition 5 (Variance calculation–True pairs, sD+(a, b) < 1). Suppose that j = π∗(i),
sD+(a, b) < 1, L = o(n), and that for some c > 0,

4R
2
M (15β)2 K+M

M
ns(p ∧q)
≤1

2,
30R4(2N + 1)2(15β)4(K+M)

n
≤1

2,

sD+(a, b)

D
≥(log log n)2

log n
,
2NL(4LM)6L ≤logc n.
(24)

Then, for any i ∈[n], we have

Var[Φbσ
ij1H]

E[Φbσ
iπ∗(i)1H]2 = O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


.

Proposition 6 (Variance calculation–Fake pairs, sD+(a, b) < 1). Suppose that j ̸= π∗(i),
sD+(a, b) < 1, N = Θ(log n), D = o(
log n
log log n), and that

4R
2
M (15β)2 K+M

M
ns(p ∧q)
≤1

2,
sD+(a, b)

D
≥(log log n)2

log n
,
(15β)2(K+M)30R4(4n + 1)2

n


(2β)8(K+M)(4LM)4L(4L)! ≤1

2,
(25)

Then, for any i ∈[n], we have

Var[Φbσ
ij1H]

E[Φbσ
iπ∗(i)1H] = O(
1
|T |ρ2N ).

E.3
Putting things together: proof of Theorem 3

Proof of Proposition 3. Firstly, with the following conditions, (22) (23) (24) (25) are implied:

L ≤c1 log n

log log n ∧c6
p

ns(p ∧q),
c2
log(ns(p ∧q)) ≤M

K ≤log ρ2

α
2 log 1

ρ2
,
KL ≥c3 log n

log ρ2

α
,

K + M ≤c4 log n,
R = exp(c5K),
D ≤c7
log n
(log log n)2 .
(26)

for some absolute constants c1, c2, . . . , c7 > 0.

Furthermore, with growing ns(p∧q), we have for j = π∗(i),
Var[Φij1H]
E[Φiπ∗(i)1H]2 = o(1) when sD+(a, b) >

1 and
Var[Φij1H]
E[Φiπ∗(i)1H]2 = o(1) when sD+(a, b) < 1. For any ε > 0 there exists ε′ > 0 such that

s2 ≥α + ε ⇔ρ2 ≥α + ε′. We have for j ̸= π∗(i),
Var[Φij1H]
E[Φiπ∗(i)1H]2 = o( 1

n2 ) when sD+(a, b) > 1

and
Var[Φij1H]
E[Φiπ∗(i)1H]2 = o( 1

n2 ) when sD+(a, b) < 1.

Next, we claim that almost exact recovery is achievable by counting chandeliers. Let τ = cµ, for
arbitrary c ∈(0, 1), where µ = E[Φiπ∗(i)1H]. By Chebyshev’s inequality, the probability that the
similarity score of a fake pair (j ̸= π∗(i)) of vertices exceeding τ, is upper bounded as

P(Φij1H ≥τ) ≤P (|Φij1H −E[Φij1H]| ≥cE[Φij1H]) ≤Var[Φij1H]

c2E[Φii1H]2

= O

1
|T |ρ2N


= o( 1

n2 ).

Next, by the definition of H, we have

P(Φij ≥τ) ≤P(Φij1H ≤c) + P(Hc) = o( 1

n2 ).

33


---Page Break---
Applying union bound over all i ̸= j ∈[n], we have P{∃i ̸= j, Φij ≥τ} = o(1). Thus, for all
possible pairs of vertices (i, j) ∈[n] × [n], Φij < τ with high probability.

We next study the probability that the similarity score of a true pair of vertices falling below the
threshold τ. We even consider a larger set containing F := {i ∈[n] : Φiπ∗(i) < τ},

P(|Φiπ∗(i)1H −µ| > (1 −c)µ) ≤
Var[Φiπ∗(i)1H]
(1 −c)2E[Φiπ∗(i)1H]2

= O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


=: γ = o(1).

Therefore, E[|F|] ≤nγ. Denote I ∈[n] the vertex set this algorithm matches. For every vertex not
in the set F, it is guaranteed to be matched up correctly, so

E[|I|] ≥n −E[|F|] = n(1 −γ).

By Markov’s inequality,

P(|F| >= √γn) ≤
γn
√γn = √γ = o(1).

Therefore, Algorithm 1 matches |I| ≥(1 −√γ)n vertices correctly with high probability.

In Regime II, with Proposition 4, Proposition 5, and Proposition 6, the same argument follows. These
two regimes together complete the proof.

F
Proof of Theorem 4: Efficient algorithm for almost exact graph matching

F.1
Algorithm–Color coding based similarity score estimation

In this section, we re-state and modify the efficient graph matching algorithm for correlated Erd˝os–
Rényi graphs discussed in Section 5 of [42], and then analyze it for correlated SBMs.

Let (G1, G2) be a pair of correlated SBMs with adjacency matrices A and B. Let H be a rooted
connected graph with N + 1 vertices. We now want to approximately count the signed subgraphs
rooted at i ∈[n] on the centralized adjacency matrices. In general, we do not have access to the
centralized adjacency matrices A and B, we would use community label estimates bσA and bσB from
Algorithm 3 to approximately centralize the adjacency matrices as A
bσA and B
bσB.

First, we generate a random coloring µ : [n] →[N + 1], which is assigning every node on A to one
of N + 1—the same as the number of vertices on a chandelier—colors independently and uniformly
at random. For any vertex set V ⊂[n], we define χµ(V ) = 1{∀x,y∈V,x̸=y:µ(x)̸=µ(y)}. We call the
vertex set V being colorful if χµ(V ) = 1. We denote r := P(χµ(V ) = 1) =
(N+1)!
(N+1)N+1 . We define
the approximate signed rooted subgraph count as

Xi,H(A
bσA, µ) :=
X

S(i)∼
=H
χµ(V (S))Πe∈E(S)A
bσA
e .
(27)

Observe that E[Xi,H(A
bσA, µ)] = rWi,H(A
bσA), where Wi,H(A
bσA) is the ground truth of signed

counts as defined in Section C. In another word, Xi,H(A
bσA, µ)/r is an unbiased estimator of

Wi,H(A
bσA).

Let t := ⌈1/r⌉, we repeat the random coloring for t times and then average over the estimates before
taking the inner product of two signed counts vectors. Formally, we generate 2t independent colorings
independently and uniformly at random, denoted as {µa}t
a=1 and {νb}t
b=1. For vertex i in G1 and
vertex j in G2, we define the approximate similarity score as follows

eΦbσ
ij := 1

r2
X

H∈T
aut(H)

 
1

t

t
X

a=1
Xi,H(A
bσA, µa)

!  
1

t

t
X

b=1
Xj,H(B
bσB, νb)

!

.

34


---Page Break---
We have E[eΦbσ
ij|A
bσA, B
bσB] = Φbσ
ij and also E[eΦbσ
ij1H|A
bσA, B
bσB, σ∗] = Φbσ
ij1H also holds as 1H is a
deterministic function on σ∗. We summarize the algorithm as Algorithm 4. There are two additional
steps than Algorithm 2 in [42]: First, in the construction of the chandelier class in this work, we need
to filter out instances with maximum degree greater than a threshold D, which takes O(K) time for
each tree; Second, we need to obtain the community label estimates before estimating the signed
subgraph counts.

Algorithm 4 Efficient Almost Exact Graph Matching Algorithm
Input: Adjacency matrices A and B on n vertices for correlated stochastic block models (G1, G2).

Step 1 - Construct the chandelier class

1: (Rooted tree generation [6]) List all non-isomorphic rooted trees with K edges.
2: (Automorphism constraint [12]) Compute aut(H) for each rooted tree using the automorphism
algorithm for trees.
3: (Maximum degree constraint) Compute maximum degree of vertices.
4: (Chandelier class) Return J as the subset of rooted trees whose number of automorphisms is at
most R and maximum degree is at most D. Construct (K, L, M, R, D)-Chandelier class T .
Step 2 - Estimation of similarity score

(Random Coloring) Generate i.i.d. uniformly random colorings {µa}t
a=1 and {νb}t
b=1, each maps
from [n] to [N + 1].
(Community recovery) Obtaining bσA and bσB for A and B independently by Algorithm 3.
for all (i, j) ∈[n] × [n] do

for all H ∈T do

(Signed counts estimation [43]) Compute {Xi,H(A
bσA, µa)}t
a=1 and {Xj,H(B
bσB, νb)}t
b=1.
end for
end for

Output: The approximate similarity scores {eΦij}i,j∈[n].

When we can obtain the correct centralized adjacency matrices A and B, we replace A
bσA (resp.
B
bσB) with A (resp. B), and everything defined above is still valid. In that case, we denote the
approximate similarity score as eΦ rather than eΦbσ.

F.2
Analysis

The analysis is similar to that in Section 5.1 of [42], while we split the analysis into two regimes.

Under Regime I (sD+(a, b) > 1), we can recover the community labels on G1 and G2 exactly correct
with high probability. We define Γij as an upper bound of Var[Φij] as follows:

Var[Φij1H] ≤
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
|E[AS1BS2AT1BT21H]|

1{S1̸=S2 or T1̸=T2 or V (S1)∩V (T1)̸={i}}1{S1△T1⊂S2∪T2,S2△T2⊂S1∪T1} =: Γij.
(28)

If 1{S1△T1⊂S2∪T2,S2△T2⊂S1∪T1} = 0, then there exists some edge occurring only once among
S1, S2, T1, and T2, and so |E[AS1BS2AT1BT21H]|
=
0.
Therefore, we only look at
the cases where every edge occurs at least two times among these four chandeliers.
If
1{S1̸=S2 or T1̸=T2 or V (S1)∩V (T1)̸={i}} = 0, then S1 = S2, T1 = T2, and S1 has no common ver-
tex with T1 except for the root. In this case, S1 and T1, and S2 and T2 have no common edges, so the
covariance between AS1BT1 and AS2BT2 is always zero.

For Regime II (sD+(a, b) < 1), where we cannot recover the correct centralized adjacency matrices
with high probability. Alternatively, we define Γ′
ij as:

Γ′
ij :=
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H].
(29)

From the definition of variance,
Var[Φbσ
ij1H] ≤Γ′
ij.

35


---Page Break---
Lemma 15. Fix constants a ̸= b > 0, s ∈[0, 1]. Let (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s). The
variance of estimation error has the following upper bound:

Var[(eΦij −Φij)1H] ≤3Γij,
Var[(eΦbσ
ij −Φbσ
ij)1H] ≤3Γ′
ij,

where Γij is defined in (28) and Γ′
ij is defined in (29).

Proof. Regime I: Assume that sD+(a, b) > 1. Define

Yij(µ, ν) :=
X

H∈T
aut(H)Xi,H(A, µ)Xj,H(B, ν),

where µ, ν are two (N + 1)-coloring of the vertices in [n]. Then, we can represent the approximate
similarity score as

eΦij =
1
r2t2

t
X

c=1

t
X

d=1
Yij(µc, νd).
(30)

For any µc, νd, 1 ≤c, d ≤t, Yij(µc, νd)/r2 is an unbiased estimator of Φij given A, B as

E[Yij(µc, νd)1H|A, B, σ∗] = r2 X

H∈T
aut(H)Wi,H(A)Wj,H(B)1H = r2Φij1H.

Note that Xi,H(A, µ) and Xj,H(B, ν) are independent conditioned on A, B. Moreover, note that
{Yij(µc, νd)}1≤c,d≤t are identically distributed. Hence, we have

E[eΦij1H] = E[Eµ,ν[ 1

r2t2

t
X

c=1

t
X

d=1
Yij(µc, νd)1H|A, B, σ∗]] = E[Φij1H].

Next, we bound the variance of eΦij. In particular, we can get

Var[(eΦij −Φij)1H] = Var(E[(eΦij −Φij)1H|A, B, σ∗]) + E[Var((eΦij −Φij)1H|A, B, σ∗)]

= E[Var(eΦij|A, B, σ∗)1H],

because Φij and 1H are fixed conditional on A, B, σ∗. Furthermore, conditioned on A and B, for
any 1 ≤c, d, e, f ≤t, Yij(µc, νd) are independent with Yij(µe, νf) if and only if c ̸= e and d ̸= f.

Var[(eΦij −Φij)1H] ≤
1
r4t4

t
X

c=1

t
X

d=1

t
X

e=1

t
X

f=1
E[Cov(Yij(µc, νd), Yij(µe, νf))|A, B, σ∗)1H].

Applying Lemma 16, we have

Var[(eΦij −Φij)1H] ≤
1
r4t4

t
X

c=1

t
X

d=1

t
X

e=1

t
X

f=1
(r2+1{c̸=e}+1{d̸=f} −r4)Γij

=
1
r2t2
 
t2r2 + 2t3r3 −(t2 + 2t3)r4
Γij

≤3Γij,

where the last inequality holds because t = ⌈1/r⌉, tr ≥1.

Regime II: Assume that sD+(a, b) < 1. With community label estimates bσA and bσB, we define

Y bσ
ij (µ, ν) :=
X

H∈T
aut(H)Xi,H(A
bσA, µ)Xj,H(B
bσB, ν),
(31)

where µ, ν are two (N + 1)–coloring of the vertices in [n]. The proof for Regime I still holds in this

case by replacing A, B, Yij, Φij, eΦij and Γij with A
bσA, B
bσB, Y bσ
ij , Φbσ
ij, eΦbσ
ij and Γ′
ij correspondingly.

36


---Page Break---
Lemma 16 (Extension of Lemma 12 in [42] to correlated SBMs). Fix constants a ̸= b > 0, s ∈
[0, 1]. Let (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s). Fix any 1 ≤c, d, e, f ≤t, and i, j ∈[n]. If
sD+(a, b) > 1, then

E[Cov(Yij(µc, νd), Yij(µe, νf)|A, B, σ∗)1H] ≤(r2+1{d̸=f}+1{c̸=e} −r4)Γij.
(32)

If sD+(a, b) < 1, then

E[Cov(Yij(µc, νd), Yij(µe, νf)|A
bσA, B
bσB, σ∗)1H] ≤(r2+1{d̸=f}+1{c̸=e} −r4)Γ′
ij.
(33)

Proof. We first prove (33) and then explain why it implies (32).

For two independent (N + 1)–colorings µ and ν of the vertices in [n]. From the definition of

Y bσ
ij (µ, ν) (31), E[Y bσ
ij (µ, ν)|A
bσA, B
bσB, σ∗] = r2Φbσ
ij. Then,

Cov(Y bσ
ij (µc, νd), Y bσ
ij (µe, νf)|A
bσA, B
bσB, σ∗)

= E[Y bσ
ij (µc, νd)Y bσ
ij (µe, νf)|A
bσA, B
bσB, σ∗] −r4(Φbσ
ij)2.
(34)

From the definition of Y bσ
ij (µ, ν) in (31) again,

E[Yij(µc, νd)Yij(µe, νf)|A
bσA, B
bσB, σ∗]

= E[
X

H∈T
aut(H)Xi,H(A
bσA, µc)Xj,H(B
bσ, νd)

×
X

I∈T
aut(H)Xi,I(A
bσA, µe)Xj,I(B
bσB, νf)|A
bσA, B
bσB, σ∗].

From (27) and the independence of colorings,

E[Yij(µc, νd)Yij(µe, νf)|A
bσA, B
bσB, σ∗]

= E
 X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
χµc(V (S1))A
bσA
S1

× χνd(V (S2))B
bσB
S2 × χµe(V (T1))A
bσA
T1 × χνf (V (T2))B
bσB
T2 |A
bσA, B
bσB, σ∗



=
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
E[χµc(V (S1))χµe(V (T1))]

× E[χνd(V (S2))χνf (V (T2))]A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 .
(35)

From (34), (35), and (10),

E[Cov(Yij(µc, νd), Yij(µe, νf)|A
bσA, B
bσB, σ∗)1H]

=
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

×

E[χµc(V (S1))χµe(V (T1))]E[χνd(V (S2))χνf (V (T2))] −r4

.

Observe that E[χµc(V (S1))χµe(V (T1))]
≤
r1+1{c̸=e} and E[χνd(V (S2))χνf (V (T2))]
≤
r1+1{d̸=f},

E[Cov(Yij(µc, νd), Yij(µe, νf)|A
bσA, B
bσB, σ∗)1H]

≤
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H](r2+1{d̸=f}+1{c̸=e} −r4)

= Γ′
ij(r2+1{d̸=f}+1{c̸=e} −r4),

37


---Page Break---
and this completes the proof for Regime II.

In Regime I, E[AS1BS2AT1BT21H] ̸= 0 only if {S1 △T1 ⊂S2 ∪T2, S2 △T2 ⊂S1 ∪T1} happens.
In addition, if S1 = S2, T1 = T2, and V (S1) ∩V (T1) = {i}, then E[χµc(V (S1))χµe(V (T1))] =
E[χνd(V (S2))χνf (V (T2))] = r2, because S1 (resp. S2) shares no common edges with T1 (resp. T2)
Therefore,

E[Cov(Yij(µc, νd), Yij(µe, νf)|A, B, σ∗)1H]

≤
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
E[AS1BS2AT1BT21H](r2+1{d̸=f}+1{c̸=e} −r4)

=
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(j)∼
=H

X

T1(i),T2(j)∼
=I
E[AS1BS2AT1BT21H](r2+1{d̸=f}+1{c̸=e} −r4)

× 1{S1̸=S2 or T1̸=T2 or V (S1)∩V (T1)̸={i}}1{S1△T1⊂S2∪T2,S2△T2⊂S1∪T1}
= Γij(r2+1{d̸=f}+1{c̸=e} −r4),

where the first line follows from the proof in Regime II and the last line applies the definition of
Γij (28).

Now, we are ready to prove Theorem 4.

Proof of Theorem 4:
For the first part of this proof, our goal is to show that the estimated score

preserves the asymptotic upper bound on
Var[Φij1H]
E[Φiπ∗(i)1H]2 for Regime I and
Var[Φbσ
ij1H]
E[Φbσ
iπ∗(i)1H]2 for Regime II.

For Regime I (sD+(a, b) > 1), we have

Var[eΦij1H]

E[eΦiπ∗(i)1H]2 =
Var[eΦij1H]
E[Φiπ∗(i)1H]2

=
1
E[Φiπ∗(i)1H]2

h
Var[Φij1H] + Var[(eΦij −Φij)1H] + 2 Cov((eΦij −Φij)1H, Φij1H)
i
,

where the first equality is from the fact that eΦij1H is an unbiased estimator of Φij1H conditioned on
A, B, and σ∗. Further,

Var[eΦij1H]

E[eΦiπ∗(i)1H]2 ≤
1
E[Φiπ∗(i)1H]2

h
Γij + 3Γij + 2E[(eΦij −Φij)Φij1H]
i
≤4
Γij
E[Φiπ∗(i)1H]2 ,

where the first inequality holds from (28), Lemma 15, and E[(eΦij −Φij)1H] = 0, and the second
inequality holds because E[((eΦij −Φij)Φij1H)] = E[E[(eΦij −Φij)|A, B, σ∗]Φij1H] = 0.

The proof of Proposition 2 and Proposition 3 are based on analyzing Γij and thus the upper bounds
on
Var[Φij1H]
E[Φiπ∗(i)1H]2 all hold for
Γij
E[Φiπ∗(i)1H]2 . Therefore, we conclude that for all i ∈[n], if (22) holds,
then

Var[eΦiπ∗(i)1H]

E[eΦiπ∗(i)1H]2 = O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


;

and that for all i, j ∈[n], j ̸= π∗(i), if (23) holds, then

Var[eΦij1H]

E[eΦiπ∗(i)1H]2 = O

1
|T |ρ2N


.

For Regime II (sD+(a, b) < 1), from (29) and Lemma 15,

Var[eΦbσ
ij1H]

E[eΦbσ
ij1H]2 ≤
1
E[Φbσ
iπ∗(i)1H]2

h
Γ′
ij + 3Γ′
ij + 2E[(eΦbσ
ij −Φbσ
ij)Φbσ
ij1H])
i
≤4
Γ′
ij
E[Φbσ
iπ∗(i)1H]2 ,

38


---Page Break---
The proof of Proposition 5 and Proposition 6 are based on analyzing Γ′
ij and thus the upper bounds

on
Var[Φbσ
ij1H]
E[Φbσ
iπ∗(i)1H]2 all hold for
Γ′
ij
E[Φbσ
iπ∗(i)1H]2 . Therefore, we conclude that for all i ∈[n], if (24) holds,

then
Var[eΦbσ
iπ∗(i)1H]

E[eΦbσ
iπ∗(i)1H]2 = O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


;

and that for all i, j ∈[n], j ̸= π∗(i), if (25) holds, then

Var[eΦbσ
ij1H]

E[eΦbσ
iπ∗(i)1H]2 = O

1
|T |ρ2N


.

For the second part of this proof, we are going to show the time complexity of Algorithm 4.

First, we know that step 1-1 costs time O(βK) by the algorithm in [6], step 1-2 takes time O(K) by
the algorithm in [12], and step 1-3 takes time O(K) by enumerating through all edges. In summary,
he total time complexity to generate J is O(K2αK). Afterwards, it takes O(|T |) time to complete
step 1-4, the generation of chandelier class.

The signed counts estimation step takes O(|T |N3Nn2) times as shown in the proof of Proposition 5
in [42]. Then, the total time complexity is

O
 
K2βK + |T |(1 + N(3e)Nn2)

= O
|J |
L


N(3e)Nn2


= O
 
βKL(3e)Nn2
= O
 
(3eβ)Nn2
.

Under condition (3) and for large enough log(ns(p ∧q)) ≥log(2), we have

N = (K + M)L = C′ log n

ε
(1 +
C′′

log(ns(p ∧q))) ≤
C′(1 +
C′′
log 2)

ε
log n,

for some constants C′ and C′′. Hence, there exists some constant C depending only on ε such that
the total time complexity of Algorithm 4 is O(nC).

G
Proof of Theorem 5 (Exact graph matching by seeded graph matching)

For a pair of correlated SBMs (G1, G2) ∼CSBM(n, p, q, s), where p = a log n

n
and q = b log n

n
,
Algorithm 4 efficiently matches (1 −o(1))n vertices correctly with high probability. Our next step is
to finish the matching on those remaining vertices.

To do this, we use the seeded graph matching algorithm7 (Algorithm 2): Starting with an initial
partial matching on at least (1 −ε/16)n vertices which is correct on whatever it matches, we form
new matches between vertex i in G1 and vertex j in G2 if the common neighbors of those two
vertices under the current partial matching sufficiently large. We then update the partial matching
and repeat this rule until we get a complete matching, which will be shown happens with high
probability. Define h(x) = x log x −x + 1. The sufficiently large common neighbors threshold is
set as γ p2+q2

2
s2(n + 2n
3
4 ), where γ ∈(1, ∞) such that h(γ) =
3 log n
(n−2)pqs2 .

Denote Nπ(i, j) as the number of common neighbors of i and j under correspondence π. In another
word, Nπ(i, j) is the number of vertex v ∈I such that v is a neighbor of i in A and π(v) is a neighbor
of j in B. If π = π∗, we also write it as N(i, j).

The following lemma for SBM is an analogy to Lemma 13 in [42], which studied the property of
Erd˝os–Rényi graph.

Lemma 17. Fix ε > 0 and a, b > 0 such that a+b

2
≥1 + ε. Let p = a log n

n , q = b log n

n
and
G ∼SBM(n, p, q). Let I ⊂[n] be a subset of vertices, eG(I, Ic) denotes the number of edges
between vertices in I and vertices in Ic = [n] \ I. With probability 1 −n−ε

8 , for any I such

7Relevant seeded graph matching algorithms also occur in [69, 5, 41].

39


---Page Break---
that |I| ≤
ε
16n, eG(I, Ic) ≥η|I||Ic|p ∧q, where η is the unique solution in (0, 1) such that

h(η) =
(1+ ε

8 ) log n
(p+q)( n

2 −n3/4−ε

16 n). In particular,

η ≥1 −

s

1 + ε

8
(1 + ε)(1 −ε

7).

Proof. If I = ∅, eG(I, Ic) ≥η|I||Ic| p+q

2
holds trivially. Let I be an non-empty set with 1 ≤|I| ≤
ε
16n. We denote k as |I|. The number of edges between I and Ic is the summation of k(n −k)
independent Bernoulli trails. For arbitrary vertex i ∈I and vertex j ∈Ic, if they have the same
community label, then the Bernoulli trail between them has mean p. Otherwise,t he Bernoulli trail has
mean q. We assume that there are N1 trails with mean p and N2 trails with mean q, where N1 +N2 =
k(n −k). We can write out the distribution as eG(I, Ic) ∼Binom(N1, p) + Binom(N2, q).

Under the balanced community event H, n

2 −n3/4 ≤|V +|, |V −| ≤n

2 + n3/4. Assume that there
are k+ vertices in I with label +1 and the remaining k−vertices with label −1.

N1

k = 1

k


k+(|V +| −k+) + k−(|V −| −k−)

≥n

2 −n3/4 −k2
+ + k2
−
k
≥n

2 −n3/4 −ε

16n,

(36)
N2

k = 1

k


k−(|V +| −k+) + k+(|V −| −k−)

≥n

2 −n3/4 −2k1k2

k
≥n

2 −n3/4 −ε

32n. (37)

We are interested in the probability of eG(I, Ic) being less than n|I||Ic|p ∧q:

P(eG(I, Ic) ≤ηk(n−k)p∧q) ≤P(eG(I, Ic) ≤η(N1p+N2q)) ≤exp(−(N1p+N2q)h(η)),

where the first inequality holds because N1 + N2 = k(n −k) and p ∧q ≤p, q and the second
inequality holds because of the multiplicative Chernoff bound (Lemma 4).

Let h(η) =
(1+ ε

8 ) log n
(p+q)( n

2 −n3/4−ε

16 n), we can show that

k(1 + ε

8) log n
N1p + N2q
<
(1 + ε

8) log n
(p + q)( n

2 −n3/4 −ε

16n) ≤
1 + ε

8
(1 + ε)(1 −2n−1/4 −ε

8) <
1 + ε

8
(1 + ε)(1 −ε

7) < 1,

(38)
where the first inequality holds because of (36) and (37), the second inequality holds because
a+b

2
≥1 + ε and the third inequality holds for sufficiently large n. Since h(η) ∈(0, 1), there is an
unique solution of η ∈(0, 1) due to the monotocity (decreasing) of the function h(·).

Applying the fact that (N1p + N2q)h(η) > (1 + ε

8) log n (38) and using an union bound over all
subsets I ⊂[n] with size 1 ≤|I| ≤
ε
16, we have

P(∃I ⊂[n] s.t. 1 ≤|I| ≤ε

16, eG(I, Ic) ≤η|I|(n −|I|)p ∧q)

≤

ε
16
X

k=1
nkP(eG(I, Ic) ≤ηk(n −k)p ∧q) ≤

ε
16
X

k=1
nk−k(1+ ε

8 ) = O(n−ε

8 ).

Because h(x) ≥(x−1)2

2
for x ∈(0, 1), we have η ≥1 −
p

2h(η) > 1 −
q

1+ ε

8
(1+ε)(1−ε

7 ).

Now, we are ready to prove Theorem 5.

Proof of Theorem 5. Firstly, we study the size of common neighbors. For an arbitrary vertex
u ∈[n] in G1 and v ∈[n] in G2 such that v is not the true correspondence of u, we study the number
of common neighbors N(u, v) in the intersection graph corresponding to the true permutation.

Case a: These two vertices come from different communities. If v ̸= π∗(u), σ(u)σ(v) = −1, then we
have N(u, v) ∼Binom(n −2, pqs2). By the multiplicative Chernoff bound (Lemma 4) for Binomial
distributions, for γ ∈(1, ∞), we have

P(N(u, v) ≥γ p2 + q2

2
s2(n + 2n
3
4 )) < P(N(u, v) ≥γ(n −2)pqs2) ≤e−(n−2)pqs2h(γ) = n−3,

40


---Page Break---
where h(γ) =
3 log n
(n−2)pqs2 > 1, γ ∈(1, ∞). The first inequality holds because p2+q2

2
(n + 2n
3
4 ) >
(n −2)pq.

By a union bound over all u, v such that v ̸= π∗(u), σ(u)σ(v) = −1, we have

P{∃v ̸= π∗(u), σ(u)σ(v) = −1, s.t. N(u, v) ≥γ p2 + q2

2
s2n(1 + 2n−1

4 )} = O( 1

n).

Case b: These two vertices come from the same community. If v ̸= π∗(u), σ(u)σ(v) = 1, then we
have N(u, v) ∼Binom(|V σ(v)| −2, p2s2) + Binom(|V −σ(v)|, q2s2) because there are |V σ(v)| −2
vertices left from the same community as u, v and |V −σ(v)| vertices from the different community.
Denote n1 as |V σ(v)| −2 and n2 as |V −σ(v)|, n1 + n2 = n −2.

P{N(u, v) ≥γ p2 + q2

2
s2(n + 2n
3
4 )} < P{N(u, v) ≥γ(n1p2 + n2q2)s2}

≤exp(−(n1p2 + n2q2)s2h(γ)) < n−3.

The first inequality holds because p2+q2

2
(n+2n
3
4 ) > n1p2 +n2q2. The last inequality holds because
3 log n
(n−2)pq >
3 log n

p2+q2

2
n(1−2n−1

4 ) >
3 log n
n1p2+n2q2 for sufficiently large n.

Then, with an union bound over all u, v such that v ̸= π∗(u), σ(u)σ(v) = 1, we have

P{∃v ̸= π∗(u), σ(u)σ(v) = 1, s.t. N(u, v) ≥γ p2 + q2

2
s2n(1 + 2n−1

4 )} = O( 1

n).

Combining the above two cases, we have

P{∃v ̸= π∗(u), N(u, v) ≥γ p2 + q2

2
s2n(1 + 2n−1

4 )} = o(1).
(39)

Secondly, we show that the algorithm is working properly. From (39), we assume that for all
v ̸= π∗(u), N(u, v) < γ p2+q2

2
s2(n + 2n
3
4 ). We want to show eπ = π∗|J in every step of Algorithm 2
by induction. This is true for the initialization of eπ as a base case, from our assumption for Theorem 5.
Suppose that this is true for t-th round of the algorithm, then at the (t + 1)-th round, we have that for
all v ̸= π∗(u),

Neπ(u, v) ≤N(u, v) < γ p2 + q2

2
s2(n + 2n
3
4 ).

Therefore, as the (t + 1)-th round, the algorithm will still not add an fake correspondence. Either the
algorithm terminates, or a new vertex i is added to J such that eπ(i) = i, which preserves the property
eπ = π∗|J.

Next, we show that Algorithm 2 always ends with J = [n] by contradiction. Assume that the
algorithm terminates with |J| < n, then Jc ̸= ∅. Then, by definition, for al i ∈Jc, Neπ(i, π∗(i)) <
γ p2+q2

2
(n + 2n
3
4 ). Therefore,

eG1∧π∗G2(J, Jc) =
X

i∈Jc
Neπ(i,π∗(i)) ≤|Jc|γ p2 + q2

2
s2(n + 2n
3
4 ).
(40)

On the other side, G1 ∧π∗G2 ∼SBM(n, s2p, s2q). If s2 p+q

2
≥(1 + ε) log n

n , then from Lemma 17
(in view of Jc as I), with probability of 1 −O(n−ε

8 ),

eG1∧π∗G2(J, Jc) ≥η|J||Jc|s2 p + q

2
≥η(1 −ε

16)n|Jc|s2(p ∧q).
(41)

To reach a contradiction between (40) and (41), the remaining is to prove that

γ ≤η(1 −ε

16)ns(p ∧q)

p2+q2

2
(n + 2n
3
4 )
.
(42)

41


---Page Break---
We observe that h(γ) =
3 log n
(n−2)pqs2 = Θ(1), while η(1−ε

16 )ns(p∧q)

p2+q2

2
(n+2n
3
4 ) = Θ(
n
log n). Therefore (42) holds

for sufficiently large n.

Finally, we analyze the time complexity of Algorithm 2. For each u ∈[n] to be added into
the seeded set I, we update the number of common neighbors Neπ(i, j) for all i, j ∈I. This step
takes O(degG1(u) degG2(u)), where degG(u) denotes the degree of u in graph G. With probability
1 −o( 1

n2 ), degG1(u) = O(n(p + q)). By summing up for all possible u to be added, the time
complexity of Algorithm 2 is O(n3(p + q)2).

H
Proof of Proposition 1

In this section, we calculate the expectation of similarity score for correlated SBM. Recall that we

have defined σ2
+ := sp(1 −sp), σ2
−:= sq(1 −sq) and that σ2
eff := (
σ2
++σ2
−
2
) throughout this paper.

Proof of Proposition 1. By definition of the similarity score,

E[Φij1H] =
X

H∈T
aut(H)
X

S(i)∼
=H

X

S(j)∼
=H
E


AS(i)BS(j)1H

.

The expectation is zero if S(i) = S(j), which implies considering π∗(i) = j suffices:

E[Φij1H] = (1 + o(1))
X

H∈T
aut(H)
X

S(i)∼
=H
E


AS(i)BS(π∗(i)) | H

.

Define X1 as the random variable for the number of in-community edges of S(i), which takes
possible value from 0 to N. Define |{S(i) : S(i) ∼= H}| as the total number of S rooted at i and
isomorphic to H, we have

E[Φiπ∗(i)1H] = (1 + o(1))
X

H∈T
aut(H)|{S(i) : S(i) ∼= H}|E

E


AS(i)BS(π∗(i)) | X1, H
 

≤(1 + o(1))
X

H∈T
nN
N
X

N1=0
P(X1 = N1 | H)ρNσ2N1
+
σ2N−2N1
−
.

The second inequality holds by |{S(i) : S(i) ∼= H}| = (
n
N)N!
aut(H) = (1 + o(1))
nN
aut(H) since
  n
N

N!
is the number of possible vertex embedding of the chandelier onto the random graph and also
E


AS(i)BS(π∗(i))|{X1 = N1} ∩H

= ρN1
+ ρN−N1
−
σ2N1
+
σ2(N−N1)
−
= (1 + o(1))ρNσ2N1
+
σ2(N−N1)
−
.

Then, according to Lemma 9 and the binomial theorem, we have P(X1 = N1 | H) = (1 +
o(1))
  N
N1

2−N. Therefore,

E[Φiπ∗(i)1H] = (1 + o(1))|T |nNρN
N
X

N1=0

 N
N1

 1

2N σ2N1
+
σ2N−2N1
−

= (1 + o(1))|T |nNρN(σ2
+ + σ2
−
2
)N = (1 + o(1))|T |nNρNσ2N
eff .

I
Proof of Proposition 2

I.1
Decorated union graph and union graph partition

The analysis of the first moment involves two chandeliers, while the second moment analysis requires
four chandeliers. Before delving into the analysis, we introduce the notation for the decorated union
graph and establish a rule for union graph partition. We adopt some notations and definitions from
[42] and further introduce concepts that are particularly useful for correlated stochastic block models.

42


---Page Break---
• (Decorated Graph) For any graph G, define ˙G as a decorated graph ˙G := (G, DG), where
DG is a decoration mapping from the edge set to a decoration set. Define the edge set of
decorated graph E( ˙G) := E(G) and the vertex set of decorated graph V ( ˙G) := V (G).
• (Decorated Union Graph) For any pair of chandeliers H, I ∈T , let S1(i), S2(j) ∼= H and
T1(i), T2(j) ∼= I. The union graph is defined as U = S1 ∪S2 ∪T1 ∪T2. Now, let us define
the proper decoration for ˙U:

DU(e) =
The subset of {S1, S2, T1, T2} where e occurs,
if e ∈E(U),
∅,
if e /∈E(U).
(43)

We call an edge e ∈E(U) t-decorated if |DU(e)| = t, t ∈{0, 1, 2, 3, 4}.

• (Decorated Set Operation) Assume that ˙U = (U, DU) and ˙U ′ = (U ′, DU ′) are two
decorated graphs. We define the union, intersection, and difference operations, from which
the complement and symmetric difference naturally follow.

– Union. ˙U ∪˙U ′ := (U ∪U ′, DU ′′ : DU ′′(e) = DU(e) ∪DU ′(e)).
– Intersection. ˙U ∩U ′ := (U ∩U ′, DU ′′ : DU ′′(e) = DU(e) ∩DU ′(e)).
– Difference. ˙U \ ˙U ′ := (U \ U ′, DU ′′ : DU ′′(e) = DU(e) \ DU ′(e)).
• (Union Graph Partition) Assume that i is the root of union graph. Consider the graph U
with edge (i, a) removed for all neighbors a of i. Let ˙C(i, a) be the connected component
therein that contains a. Let ˙G(i, a) be the graph union of ˙C(i, a) and the edge (i, a).
Then, we divide the set of root neighbors N(i) into the following sets depending on
whether ˙G(i, a) is a tree: NT = {a : (i, a) ∈E(U), ˙G(i, a) is a tree}, NN = {a : (i, a) ∈
E( ˙U)} \ NT . Furthermore, we breakdown NT into two sets depending on whether there are
at least M edges 3-decorated: NM = {a ∈NT , |{e ∈E( ˙G(i, a)) : |DU(e)| ≥3}| ≥M},
NL = NT \ NM, .
Next, we define the decomposition of decorated union graph ˙U:

˙UL :=
[

a∈NL

˙G(i, a),
˙UM :=
[

a∈NM

˙G(i, a),
˙UN := ˙U \ ( ˙UL ∪˙UM).
(44)

If any of these become an empty set, we define it as the graph consisting of the single vertex
i. To provide an intuitive understanding: ˙UN contains those chandelier branches that form
cycles in the union graph, while ˙UL and ˙UM are the collection of those chandelier branches
that do not tangle with other branches from bottom and remain a part of tree rooted at i (or
j) in the union graph. As a characteristic of trees, the decorations on edges within ˙UM and
˙UN are monotonically decreasing, meaning that the decoration set of an edge at depth d ≥1
is always a subset of the decoration set of the preceding edge at depth d −1 that connects to
it. For each node v ∈˙UM that is connected to the root, the connected component ˙G(i, a)
should have at list M vertices that are at least 3-decorated. ˙UL is the union of all connected
components remained that are trees. The definition of ˙UL implies that the branches cannot
be fully 3-decorated, otherwise this branch would fall in ˙UM.

For every union graph, we can decompose the decorated union graph based on its decoration sets.
Specifically, we want to keep track of how many times each vertex appears on the chandeliers that
are rooted at i and the chandeliers that are rooted at j. We present the definition as follows.
Definition 3 (Decorated union graph decomposition by decorations).

Kℓm := (V (U), {e ∈E(U) : ℓ= 1{e∈S1} + 1{e∈T1}, k = 1{e∈S2} + 1{e∈T2}}).
(45)
Definition 4 (Edge counts on 3 and 4-decorated edges). Based on the definition of union graph
partition and Kℓm, we further define

eL := 1

2[e((K22 ∪K21) ∩UL) + e((K22 ∪K12) ∩UL)],
(46)

eM := 1

2[e((K22 ∪K21) ∩UM) + e((K22 ∪K12) ∩UM)],
(47)

eN := 1

2[e((K22 ∪K21) ∩UN) + e((K22 ∪K12) ∩UN)],
(48)

43


---Page Break---
where 2eL (resp. 2eM, 2eN) is the counts of 3-decorated edges and two times the 4-decorated edges
on UL (resp. UM, UN).

We use UL(vL, eL, ℓ) to denote the collection of all possible ˙UL with vL vertices except for i and j,
eL counts of special edges as defined, and ℓedges in K11 ∩˙UL. UM(vM, eM) denotes the family
of all possible ˙UM with vM vertices except for i and j and eL counts of special edges as defined.
UN(vN, eN, k) denotes the family of all possible ˙UN with vN vertices except for i and j, eN counts
of special edges as defined, and excess k.

We further define the weights for of decorated graphs ˙UL, ˙UM, and ˙UN.

Definition 5. Let ˙G be an arbitrary ˙UL, ˙UM, or ˙UN, we define the weight of ˙G with regard to a
chandelier S as
wS( ˙G) :=
Y

B∈K(S),B⊂G
aut(B)
1
2 .
(49)

We set wS( ˙G) = 1 if ˙G contains no bulbs in K(S). We define the weight of ˙G as the multiplication of
its weights over S1, S2, T1, T2:

w( ˙G) := wS1( ˙G)wS2( ˙G)wT1( ˙G)wT2( ˙G).
(50)

We have the following observation with regard to the concepts introduced above:

• (Counting edges) 2(vL + vM + vN + k + 1 + 1{i̸=j}) + 2(eL + eM + eN) = 4N. This
holds because 2(vL + vM + vN + k + 1 + 1{i̸=j}) counts all the edges on union graph
twice and 2(eL + eM + eN) makes up for an additional count for 3-decorated edges and
two counts for 4-decorated edges.

• (Automorphism as decorated graph weights) From the definition of aut(·) and chandelier,

(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 = w( ˙UL)w( ˙UM)w( ˙UN).
(51)

• (Trivial upper bound on the decorated graph weights) Since aut(B) is upper bounded by
R for arbitrary bulb B, this inequality follows from the definition of decorated union graph
weights

w( ˙UL) ≤(
√

R
|NL|)4 = R2|NL|.
(52)

The same holds for ˙UM and ˙UN.

Specific to the moment calculation for correlated stochastic block models, we present the following
two definitions.

Definition 6 (g ˙U(σ+, σ−)). Fix chandeliers (S1, S2, T1, T2) on the complete graph, conditioned on
the ground-truth labels of correlated SBMs, we define a function as follows:

g ˙U(σ+, σ−) := σh(S1)+h(S2)+h(T1)+h(T2)
+
σh(S1)+h(S2)+h(T1)+h(T2)
−
,

where h(·) is the counts of edges connecting two points from the same community, h(·) is the counts
of edges connecting two points from different communities on the complete graph Kn.

Definition 7 (Extension of σ2
eff). We define γ2 :=
 σ4
++σ4
−
2

/σ4
eff and γ1 :=
 σ3
++σ3
−
2

/σ3
eff. Thus,
 σ4
++σ4
−
2
k
= (1 + o(1))
 σ2
++σ2
−
2
2k
γk
2 and
 σ3
++σ3
−
2
k
= (1 + o(1))
 σ2
++σ2
−
2
 3k

2 γk
1.

Remark 5. We observe that γ2 < 2 and γ1 ≤γ2. The first holds simply because σ4
+ + σ4
−≥
2σ2
+σ2
−. By Cauchy–Schwarz inequality, for an arbitrary random variable X, E[X3]E[X2]2 ≤

E[X4]E[X2]3/2. Let X takes σ2
+ and σ2
−uniformly at random, then
E[X3]
E[X2]3/2 ≤
E[X4]
E[X2]2 and thus
γ1 ≤γ2 is implied.

44


---Page Break---
I.2
Proof of the Proposition

In regime sD+(a, b) > 1, we can recover the correct community label with high probability [2, 48, 1].
Therefore we can effectively work on the correct centralized adjacency matrices. Recall that H =
{ n

2 −n
3
4 ≤|V +|, |V −| ≤n

2 + n
3
4 }, as defined in Section D.

Proof of Proposition 2. From the definition of variance,

Var[Φij1H] =
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
Cov(AS1BS21H, AT1BT21H).

If S1 = S2, T1 = T2, and V (S1) ∩V (T1) = {i}, the covariance becomes zero. Also, we know from
Proposition 1 that E[AS1BS2]E[AT1BT2] is either 0 or (1 + o(1))(ρσeff)2N. Thus,

Var[Φij1H] ≤
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[AS1BS2AT1BT21H].

E[AS1BS2AT1BT2] is non-zero if and only if each edge e ∈U is at least 2-decorated. Let Wij
denote the collection of decorated union graph UD, where S1(i), S2(j) ∼= H, T1(i), T2(j) ∼= I for
some H, I ∈T , such that each edge is at least 2-decorated and satisfies S1 ̸= S2 or T1 ̸= T2 or
V (S1) ∩V (T1) ̸= {i}.

Var[Φij1H] ≤
X

˙U∈Wij
(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 E[AS1BS2AT1BT21H].
(53)

Conditioned on the ground-truth labeling σ∗,

g ˙U(σ+, σ−) =
Y

2≤ℓ+m≤4

Y

(u,v)∈Kℓm
σ(ℓ+m)
c(u,v) ,

where c(u, v) = + if σ∗(u) = σ∗(v) and c(u, v) = −if σ∗(u) ̸= σ∗(v).

Conditioned on σ∗, H is determined,

E[AS1BS2AT1BT21H] = E

E[AS1BS2AT1BT2 | σ∗]1H

.
(54)

Conditioned on σ∗, g ˙U(σ+, σ−) is fixed, so as g−1
˙U (σ+, σ−).

g ˙U(σ+, σ−)−1E[AS1BS2AT1BT2 | σ∗] =
Y

2≤ℓ+m≤4

Y

(u,v)∈Kℓm
σ−(ℓ+m)
c(u,v)
E[A
ℓ
uvB
m
uv | σ∗].

We define βℓm(e) as σ−(ℓ+m)
c(u,v)
E[A
ℓ
uvB
m
uv | σ∗], for e = (u, v) ∈Kℓ,m. Thus,

g ˙U(σ+, σ−)−1E[AS1BS2AT1BT2 | σ∗] = g ˙U(σ+, σ−)−1
Y

2≤ℓ+m≤4

Y

(u,v)∈Kℓm
βℓm(e).

Then, we apply Lemma 13 to bound each βℓm(e). Since the upper bound of βℓm(e) only depends on
ℓand m, we have the following:

g ˙U(σ+, σ−)−1E[AS1BS2AT1BT2 | σ∗] ≤| max{ρ+, ρ−}|e(K11)
(sp ∧sq)−e(K22)
p

(sp ∧sq)
(e(K21)+e(K12))

=(1 + o(1))ρe(K11)(sp ∧sq)−2N+e(U),
(55)

where the last equality holds because 2
 1

2(e(K12) + e(K21)) + e(K22)

= (2 −2)e(K20) + (2 −
2)e(K02) +(3 −2)e(K12) + (3 −2)e(K21) + (4 −2)e(K22) = 4N −e(U). From (54) and (55),
and the fact that H happens with high probability, we have

E[AS1BS2AT1BT21H] = (1 + o(1))ρe(K11)(sp ∧sq)−2N+e(U)E [g ˙U(σ+, σ−) | H] .
(56)

45


---Page Break---
It remains to compute E [g ˙U(σ+, σ−) | H]. Here, ˙U is fixed and we assume that there are d2 2-
decorated edges, d3 3-decorated edges, and d4 4-decorated edges without loss of generality. Let
X(2), X(3), and X(4) be the number of in-community edges for 2, 3, and 4-decorated edges on the
complete graph respectively. In ˙U is a tree, we can apply Lemma 10 to determine the distribution of
X(2), X(3), and X(4).

Generally, ˙U contains cycles.
˙U can be decomposed into a tree with an additional set of edges
connecting vertices on the tree. We assume that there are Ai i-decorated edges on the tree and Bi
i-decorated edges in the additional edge set of size e( ˙U) −v( ˙U) + 1, for i ∈{2, 3, 4}. We apply
the Corollary 1 and have that P(X(2,a) = a2, X(3,a) = a3, X(4,a) = a4, X(2,b) = b2, X(3,b) =
b3, X(4,b) = b4 | H) ≤(1 + o(1))
 A2
a2
 A3
a3
 A4
a4

1
2A2+A3+A4 , where X(i,a) is the number of i-
decorated in-community edges on the tree-part of ˙U and X(i,b) is the number of i-decorated in-
community edges among the additional edge set.

Ai and Bi are fixed but summed up to di for each ˙U. ai (bi) takes possible values from 0 to Ai (Bi),
for i ∈[4]. The number of i-decorated in-community edges is ai + bi, and the number of i-decorated
cross-community edges is di −ai −bi = (Ai −ai) + (Bi −bi).

E [g ˙U(σ+, σ−) | H] =

A2
X

a2

A3
X

a3

A4
X

a4

B2
X

b2

B3
X

b3

B4
X

b4
σ2(a2+b2)
+
σ2(d2−a2−b2)
−
σ3(a3+b3)
+
σ3(d3−a3−b3)
−

× σ4(a4+b4)
+
σ4(d4−a4−b4)
−

A2
a2

A3
a3

A4
a4


1
2A2+A3+A4

=
σ2
+ + σ2
−
2

A2 σ3
+ + σ3
−
2

A3 σ4
+ + σ4
−
2

A4

×

B2
X

b2=0
σ2b2
+ σ2(B2−b2)
−

B3
X

b3=0
σ3b3
+ σ3(B3−b3)
−

B4
X

b4=0
σ4b4
+ σ4(B4−b4)
−

≤
σ2
+ + σ2
−
2

d2 σ3
+ + σ3
−
2

d3 σ4
+ + σ4
−
2

d4
2k+1,
(57)

where the last inequality holds because we can upper bound by multiplying a binomial coefficient
 B2
b2
 B3
b3
 B4
b4

and that B2 + B3 + B4 = k + 1. By the definition of γ2 and γ1,

E [g ˙U(σ+, σ−) | H] ≤σ2d2+3d3+4d4
eff
γd3
1 γd4
2 2k+1 ≤σ4N
eff γ2(eL+eM+eN)
2
2k+1.
(58)

The last inequality holds because γ1 < γ2 and d3 + d4 = e(K12) + e(K21) + e(K22) ≤2(eL +
eM + eN).

Denote Wij(v, k) as the subset of Wij that contains all the elements that have exactly v vertices
except for i and j and excess k. By applying the definition of union graph partitions, decorated graph
weights, and (53),

Var[Φij1H] ≤(1 + o(1))
X

k≥−1

X

v
(sp ∧sq)−2N+v+k+1
X

˙U∈Wij(v,k)
ρe(K11)w( ˙UL)w( ˙UM)w( ˙UN)

× σ4N
eff γ2(eL+eM+eL)
2
2k+1,
(59)

For the ˙UL, ˙UM, ˙UN partition, we define

PL(vL, eL, ℓ) :=
X

˙UL∈UL(vL,eL,ℓ)
w( ˙UL),
(60)

PM(vM, eM) :=
X

˙UM∈UL(vM,eM)
w( ˙UM),
(61)

PN(vN, eN, k) :=
X

˙UN∈UN(vN,eN)
w( ˙UN).
(62)

46


---Page Break---
Then, we can write out the upper bound as

Var[Φij1H] ≲
X

k

X

v
(sp ∧sq)−2N+v+k+1
X

vL,vM,vN≥0

X

eL,eM,eN,ℓ≥0
σ4N
eff γ2(eL+eM+eL)
2
2k+1

× ρℓPL(vL, eL, ℓ)PM(vM, eM)PN(vN, eN, k).

Applying Proposition 1, Lemma 18, Lemma 19, and Lemma 20, we can upper bound the variance by:

Var[Φij1H] ≲
X

k

X

v
(sp ∧sq)−2N+v+k+1 X

vi≥0

X

ei,ℓ≥0
ρℓ(11βn)vM R

4eM

M 1{vM≤eM
4K+4M

M
}

× σ4N
eff γ2(eL+eM+eL)
2
× 2nvL(1 + 2L2)eLfet,t1{K+M|eL+vL}1{vL+eL≤2N}

× nvN (11β)vN (22R4(vN + 1)2)k+11{vN≤2(K+M)(k+1)}

Note that vM ≤eM 4K+4M

M
, we have P

vM≥0(11β)vM ≤2(11β)eM
4K+4M

M
. Also, we know that
vN ≤2N, vN ≤2(K + M)(k + 1). Then, P

vN≥0(11β)vM ≤2(11β)2(K+M)(k+1). Thus,

Var[Φij1H] ≲
X

k,v
(sp ∧sq)−2N+v+k+1
X

eL+eM+eN=2N−(v+k+1)
23nv 
R
4
M (11β)
4K+4M

M
eM

× (22R4(2N + 1)(11β)2(K+M))k+1(1 + 2L2)eLσ4N
eff γ2(eL+eM+eL)
2

×
X

vL≥0,l≥0


ρℓfet,t1{K+M|eL+vL}1{vL+eL≤2N}

.

Regarding to the Lemma 4 in [42], if
12L2

ρ2(K+M)(|J |−2L) ≤1

2, then

Var[Φij1H] ≲
X

k,v
(sp ∧sq)−2N+v+k+1
X

vL+vM+vN+eL+eM+eN=2N
23nv 
R
4
M (11β)
4K+4M

M
eM

(22R4(2N + 1)(11β)2(K+M))k+1(1 + 2L2)eLσ4N
eff γ2(eL+eM+eN)
2

8ρ2N

ρ−2eL1eL̸=0 +
12L2

ρ2(K+M)|J | 1eL=0


|T |2.

Note that
12L2

ρ2(K+M)(|J |−2L) ≤1

2 is guaranteed by the first condition in (22)
14L2

ρ2(K+M)(|J |) ≤1

2 for
large enough n.

In the following step, we divide Var[Φij1H] by E[Φiπ∗(i)1H]2 and use the fact that eN = 2N −(v +
k + 1 + eL + eM):

Var[Φij1H]
E[Φiπ∗(i)1H]2 ≤(1 + o(1))26σ4N−4N
eff
X

k≥−1

22R4(2N + 1)(11β)2(K+M)

n

k+1

X

eM≥0

 

γ2
2
R
4
M (11β)
4K+4M

M
ns(p ∧q)

!eM

X

eL≥0


γ2
2
1 + 2L2

ns(p ∧q)

eL 
ρ−2eL1{eL>0} +
12L2

ρ2(K+M)|J |1{eL=0}



2N
X

eM≥0


γ2
2(
n
sp ∧sq )
2N−(v+k+1+eL+eM)
1{eL+eM≤2N−(v+k+1)}

In view of γ2 < 2, the last three conditions in (22) guarantee that
X

k≥−1

22R4(2N + 1)(11β)2(K+M)

n

k+1
≤2,

X

eM≥0

 

γ2
2
R
4
M (11β)
4K+4M

M
ns(p ∧q)

!eM
≤2,
X

eL≥0


γ2
2
1 + 2L2

ns(p ∧q)

eL
≤2.

47


---Page Break---
Also, for sufficiently large n,
γ2
2
ns(p∧q) ≤1

2, such that

2N−k−1
X

v=0
(
γ2
2
ns(p ∧q))2N−(v+k+1+eL+eM) ≤2.

In conclusion,
Var[Φij1H]
E[Φiπ∗(i)1H]2 ≤O
γ2
2(2 + 4L2)
ρ2ns(p ∧q) +
12L2

ρ2(K+M)|J |


= O

L2

ρ2ns(p ∧q) +
L2

ρ2(K+M)|J |


.

I.3
Proof of auxiliary Lemmas

To complete the proof of Proposition 2, it remains to prove Lemma 18, Lemma 19, and Lemma 20.
In this case, those upper bounds have been shown in [42]. We briefly re-state the proof idea.
Lemma 18 (Upper bound of PL(vL, eL, ℓ) (True pairs: j = π∗(i), sD+(a, b) > 1)).

PL(vL, eL, ℓ) ≤(1 + o(1))2nvL(1 + 2L2)eLfet,t1{K+M|eL+vL}1{vL+eL≤2N}.

Proof of Lemma 18. Firstly, we define unlabeled union graph class eU(vL, eL, ℓ) and set of labeled
union graphs isomorphic to ˙UL ∈eU(vL, eL, ℓ) as H( ˙UL).

PL(vL, eL, ℓ) =
X

˙UL∈UL(vL,eL,ℓ)
w( ˙UL) =
X

˙UL∈e
UL(vL,eL,ℓ)
w( ˙UL)|H( ˙UL)|.

The number of ways to label ˙UL is |H( ˙UL)| ≤
  n
vL

vL!
aut( ˙UL) ≤
nvL
aut( ˙UL),
  n
vL

vL! ≤nvL. In addition,

w( ˙UL) ≤aut( ˙UL). This is true because every bulbs are exactly 2-decorated so for the union graph,
the automorphism number coming from the orbits in this bulb is the same as the weights of this bulb.
However, there can be other orbits outside the bulbs, for example, some bulbs are isomorphic so the
vertices on the wires can be in the same orbit, thereby increasing the automorphism number.

From Lemma 7 in [42], we know that

| eUL(vL, eL, l)| ≤2(1 + 2L2)eLfet,t1{K+M|eL+vL}1{vL+eL≤2N},

where fet,t counts the possible structures of chandeliers before merging edges to form a union graph
and et is depending on ℓ.

Combining these pieces together, we derived the upper bound:

PL(vL, eL, ℓ) ≤2nvL(1 + 2L2)eLfet,t1{K+M|eL+vL}1{vL+eL≤2N}.

See Lemma 3 in [42] for more detailed discussion on t, et and fet,t.

Lemma 19 (Upper bound of PM(vM, eM) (True pairs: j = π∗(i), sD+(a, b) > 1)).

PM(vM, eM) ≤(11βn)vM R

4eM

M σ2vM+2eM
eff
γeM
2
1{vM≤eM
4K+4M

M
}.

Proof. We have |NM|M ≤2eM because each connected component ˙G(i, a) contains at least M
edges that are 3 or 4-decorated and NM is the number of neighbors a in this set. This gives the
constraint on vM ≤2eM

M (2K + 2M) and from the definition of w(·), w( ˙UM) ≤R2|NM| ≤R

4eM

M .
Define eUM(vM, eM) as the unlabeled union graph class and H( ˙UM) as the set of labeled ˙UM for
˙UM ∈eUM(vM, eM).

PM(vM, eM) =
X

˙UM∈UM(vM,eM)
w( ˙UM) ≤R

4eM

M
X

˙UM∈e
UM(vM,eM)
|H( ˙UM)|.
(63)

We have |H( ˙UM)| ≤(n)vM . There are at most βvM rooted unlabeled undercoated trees, where
β =
1
(1+o(1))α [56] and at most 11vM decorations for each tree. Thus, | eUM(vM, eM)| ≤(11β)vM .
Finally, we obtain that

PM(vM, eM) ≤(11βn)vM R

4eM

M 1{vM≤(2K+2M) 2eM

M }.

48


---Page Break---
Lemma 20 (Upper bound of PN(vN, eN, k) (True pairs: j = π∗(i), sD+(a, b) > 1)).

PN(vN, eN, k) ≤nvN (11β)vN (11R4(vN + 1)2)k+11{vN≤2(K+M)(2k+2)}.

Proof. From Lemma 2 in [42], we know that |NN| ≤2k + 2 and thus w( ˙UN) ≤R|NN| ≤R4(K+1).
Briefly, this is because the excess is k and whenever two branches tangle with each other, the excess
of this graph increases by one. To maximally involving branches in this k + 1 times of branch tangles,
we never re-tangle a branch after it has tangled with the other branch. In this way, we see that there are
at most 2(k + 1) branches in ˙UN. This immediately gives the condition of vN ≤(2k + 2)(M + K).
Define eUN(vN, eN, k) as the unlabeled union graph class and H( ˙UN) as the set of labeled ˙UN for
˙UN ∈eUN(vN, eN, k).

PN(vN, eN, k) =
X

˙UN∈UN(vN,eN,k)
w( ˙UN) ≤R4(k+1)
X

˙UN∈e
UN(vN,eN,k)
|H( ˙UN)|.

˙UL consists of a tree with vN vertices and additional k + 1 edges connecting vertices on the tree.
What’s more, each edge can be associated with at most 11 possible decoration sets.

| eUN(vN, eN, k)| ≤βvN
 vN+1
2


k + 1


11vN+k+1 ≤(11β)vN (11(vN + 1)2)(k+1).

Therefore,

PN(vN, eN, k) ≤nvN (11β)vN (11R4(vN + 1)2)k+11{vN≤2(K+M)(2k+2)}.

J
Proof of Proposition 3

In this section, we show the Proposition 3 to complete the analysis on variance of similarity score. In
this case, as j ̸= π∗(i), the decorated union graph structure is different from the case when j = π∗(i)
because the roots of S1 and S2 are different on the complete graph.

J.1
Graph partition

The definitions in Section I still apply, except that we need to re-define the union graph partition.

• (Union graph partition) We first decompose ˙U into three edge-disjoint subgraphs.
Specifically, for any neighbor a of i in ˙U, consider the graph ˙U with the edge (i, a) removed
and let ˙C(i, a) be the connected component that contains a. Denote ˙G(i, a) as the union of
˙C(i, a) and the edge (i, a). Let

NT (i) = {a : DU((i, a)) ∩{S1, T1} ̸= ∅, ˙G(i, a) is a tree not containing j},
NN(i) = {a : DU((i, a)) ∩{S1, T1} ̸= ∅} \ NT (i).

Symmetrically, let

NT (j) = {a : DU((j, a)) ∩{S2, T2} ̸= ∅, ˙G(j, a) is a tree not containing i},
NN(j) = {a : DU((j, a)) ∩{S2, T2} ̸= ∅} \ NT (j).

Next, we further decompose ˙UT (i) into two edge-disjoint subtrees and similarly for ˙UT (j).
In particular, define

NM(i) = {a ∈NT (i) : |{e ∈E( ˙G(i, a)) : |De| ≥3}| ≥M},
NL(i) = NT (i) \ NM(i).

Then, we decompose ˙U according to the node set N into two tree-parts on root i (resp. j):

˙UL(i) :=
[

a∈NL(i)
˙G(i, a),
˙UM(i) :=
[

a∈NM(i)
˙G(i, a).

49


---Page Break---
and the non-tree part

˙UN := ˙U \ ( ˙UL(i) ∪˙UM(i) ∪˙UL(j) ∪˙UM(j)).
(64)

For convenience, we denote
˙UL := ˙UL(i) ∪˙UL(j),
˙UM := ˙UM(i) ∪˙UM(j).
(65)

We again assign weights to each of these decorated union graph partitions.

w( ˙UL) := wS1( ˙UL(i))wT1( ˙UL(i))wS2( ˙UL(j))wT2( ˙UL(j)),
(66)

w( ˙UM) := wS1( ˙UM(i))wT1( ˙UM(i))wS2( ˙UM(j))wT2( ˙UM(j)),
(67)

w( ˙UN) := wS1( ˙U \ ˙UT (i))wT1( ˙U \ ˙UT (i))wS2( ˙U \ ˙UT (j))wT2( ˙U \ ˙UT (j)).
(68)

J.2
Proof of the Proposition

Proof of Proposition 3. In the case of j ̸= π∗(i), the excess k of the union graph starts from −2.
This is because the minimum number of edges of a decorated union graph for j ̸= π∗(i) with v
vertices except for i and j is v, when S1, T1 and S2, T2 form two disjoint trees rooted at i and j
respectively. Therefore, k ≥v −v( ˙U) = −2.

The proof of Proposition 2 in Section I holds for j ̸= π∗(i) until (59), which holds with placing k + 1
with k + 2. This is because the decorated union graph can be viewed as two disjoint trees with v + 2
vertices plus k + 2 edges connecting vertices on these two trees. We have,

Var[Φij1H] ≤(1 + o(1))
X

k≥−2

2N−k−2
X

v=0
(sp ∧sq)−2N+v+k+2
X

˙U∈Wij(v,k)
ρe(K11)aut(H)aut(I)

× σ4N
eff γe(K12)+e(K21)+e(K22)
2
2k+2.

We define Pij(v, k) := P
˙U∈Wij(v,k) ρe(K11)aut(H)aut(I)γe(K12)+e(K21)+e(K22)
2
.

Var[Φij1H] ≤(1 + o(1))
X

k≥−2

2N−k−2
X

v=0
(sp ∧sq)−2N+v+k+2σ4N
eff 2k+2Pij(v, k)

Case a: k = −2.
We first consider the special case when k = −2. When k = −2, there are two
disjoint trees in the decorated union graph. Then, it must be the case of S1 = T1, S2 = T2, v = 2N,
and H = I. Therefore, e(K12) = e(K21) = e(K22) = 0. We have,

Pij(2N, −2) ≤
X

H∈T
aut(H)2|S1(i) : S1 ∼= H||S2(j) : S2 ∼= H| = |T |n2N.

Under this parameterization, −2N + v + k + 2 is also zero, we have

Var[Φij1H] ≤(1+o(1))

 

|T |n2Nσ4N
eff +
X

k>−2

2N−k−2
X

v=0
(sp ∧sq)−2N+v+k+2σ4N
eff 2k+2Pij(v, k)

!

.

It turns out that the summation over k > −2 would have the same order as the first special case.

Case b: k > −2.
We enumerate through three parts of the union graph. Recall that e(K12) +
e(K21) + e(K22) ≤2(eL + eM + eN) = 4N −2(v + k + 2), therefore

Pij(v, k) ≤
X

eM

X

vL,vM,vN

X

˙U∈W(v,k)
w( ˙UL)w( ˙UM)w( ˙UN)γe(K12)+e(K21)+e(K22)
2

≤
X

eM
γ4N−2(v+k+2)
2
X

vL,vM,vN
PL(vL)PM(vM, eM)PN(vN, k),
(69)

where PM(vM, eM) is defined as before, while we let PL(vL) and PN(vN, k) have no constraints
on the number of 3 and 4-decorated edges for ˙UL and ˙UN.

50


---Page Break---
We can plug those upper bounds from Lemma 21, Lemma 22, and Lemma 23 into (69):

Pij(v, k) ≤
X

eM
R2 eM

M nv|T |4LL2L∧(4K+2)(6β)4(K+M)−21{eM≤2N−(v+k+2)}

×
X

vL+vM+vN=v
vM(11β)vM β(11β)vN (22R4(vN + 2)2)k+2σ4N
eff γ4N−2(v+k+2)
2

× 1{vM≤2(K+M) 2eM

M }1{vN≤4(K+M)(k+2)}.

Note that vM, vN ≤v ≤2N −k −2 ≤2N + 1 as k ≥−1. Therefore vM(vN + 2)2(k+2) ≤
(2N +1)3(k+2). Applying eM ≤2N −(v+k+2), vN ≤4(K+M)(k+2) and vM ≤2(K+M) eM

M
we can get the following upper bound:

Var[Φij1H]
E[Φiπ∗(i)1H]2 ≤(1 + o(1))
1
|T |ρ2N + (1 + o(1))
1
|T |ρ2N

(

4LL2L∧(4K+2)(6β)4K+4M

×
X

k≥−1

(11β)4(K+M)22R4(2N + 1)3

n

k+2

×

2N−k−2
X

v=0

 

γ2
2
R
2
M (11β)
4(K+M)

M
ns(p ∧q)

!2N−(v+k+2) )

.

From the first condition in (23), because γ2 < 2, we have

2N−k−2
X

v=0

 

γ2
2
R
2
M (11β)
4(K+M)

M
ns(p ∧q)

!2N−(v+k+2)

≤2.

From the second condition in (23), we have (11β)4(K+M)22R4(2N+1)3

n
≤1

2, therefore,

X

k≥−1

(11β)4(K+M)22R4(2N + 1)3

n

k+2
≤2
(11β)4(K+M)22R4(2N + 1)3

n


.

From the second condition in (23), we know that

4LL2L∧(4K+2)(6β)4K+4M22
(11β)4(K+M)22R4(2N + 1)3

n


≤1

2.

In summary,
Var[Φij1H]
E[Φiπ∗(i)1H]2 = O(
1
|T |ρ2N ).

J.3
Proof of auxiliary Lemmas

The following Lemma 21, Lemma 22, and Lemma 23, have been shown by Mao et al. [42]. We
briefly re-state those results for completeness.
Lemma 21 (Upper bound of PL(vL) (Fake pairs: j ̸= π∗(i), sD+(a, b) > 1)).

PL(vL) ≤nvL|T |4LL2L∧(4K+2)(6β)4(K+M)−2.

Proof. Firstly, we introduce unlabeled union graph sets eU(vL, eL) and the set of labeled isomorphic
members as H( ˙UL), for ˙UL ∈U(vL, eL, l). From definition (60),

PL(vL) =
X

˙UL∈UL(vL)
w( ˙UL) =
X

˙UL∈e
UL(vL)
w( ˙UL)|H( ˙UL)|.

As we have repeatedly seen, |H( ˙UL)| ≤
nvL
aut( ˙UL). From the Lemma 10 and Claim 5-(v) in [42]

we have the following takeaways: (1) | eUL(vL)| ≤|T |4LL2L∧(4K+2)(6β)4(K+M)−2; (2) w( ˙UL) ≤
aut( ˙UL(i))aut( ˙UL(j)) = aut( ˙UL). Putting pieces together, we complete the proof.

51


---Page Break---
Lemma 22 (Upper bound of PM(vM, eM) (Fake pairs: j ̸= π∗(i))).

PM(vM, eM) ≤vMR

2eM

M nvM (11β)vM 1{eM≤2N−(v+k+2)}1{vM≤2(K+M) 2eM

M }.

Proof. Define eUN(vM, eM) as the collection of unlabeled decorated union graphs and |H( ˙UM)| be
the set of labeled isomorphic members for each ˙UM ∈eUN(vM, eM).

In ˙UM, there are 2eM edges that are 3 or 4-decorated. Since for each branch connecting to the root,
there are at least M edges on it being at least 3-decorated, |NM(i)| + |NM(j)| ≤2 eM

M . Thus, this
directly gives w( ˙UM) ≤R
1
2 ×2(|NM(i)|+|NM(j)|) ≤R2 eM

M and

PM(vM, eM) =
X

˙UM∈UM(vM,eM)
w( ˙UM) ≤R2 eM

M
X

˙UM∈e
UM(vM,eM)
|H( ˙UM)|.

Since ˙UM(i) and ˙UM(j) are two vertex-disjoint trees with vM edges. There are at most vMβvM
unlabeled non-decorated because we can allocate vM edges to 2 trees in at most vM ways and under
each way the number of rooted unlabeled trees are bounded by βvM . There are at most 11vM ways
of decorating each edge. Therefore, | eUM(vM, eM)| ≤vM(11β)vM . In addition, the number of
labeled isomorphic members |H( ˙UM)| ≤nvM and the number of 3-decorated edges plus two times
of 4-decorated edges, eM, is upper bounded by 4N −2(v + k + 2). Putting things together,

PM(vM, eM) ≤vMR

2eM

M nvM (11β)vM 1{eM≤2N−(v+k+2)}.

Lastly, because each connected component ˙C(i, a) contains at most 2(K + M) −1 edges, for
all a ∈NT (i) (same holds for a ∈NT (j)). This is because there are at most four wires, each
from S1, T1, S2, T2 go through vertex a and every edge is at least 2-decorated. Therefore, vM ≤
(K + M) 2eM

M .

Lemma 23 (Upper bound of PN(vN, k) (Fake pairs: j ̸= π∗(i))).

PN(vN, k) ≤nvN β(11β)vN (11R4(vN + 2)2)k+21{vN≤4(K+M)(k+2)}.

Proof. From the Lemma 9 in [42] we know that |NN(i)| + |NN(j)| ≤4(k + 2). The intuition
behind this bound is that ˙UN can be viewed as a bunch of branches (wires plus bulbs) coming from
two different roots i, j tangled together. Whenever two branches intersect with each other, the excess
grow by one. Since the excess of decorated union graph is k and the starting point is two separate
trees (with excess −2), there must be k + 2 times of intersection between different branches. Each
time of intersection involves at most four branches.

Thus w( ˙UN) ≤R|NN(i)|+|NN(j)| ≤R4(k+2). As usual, we define eUN(vN, k) as the collection of
unlabeled decorated union graphs and |H( ˙UL)| be the set of labeled isomorphic members for each
˙UL ∈eUN(vN, k).

PN(vN, k) =
X

˙UN∈UN(vN,k)
w( ˙UN) ≤R4(k+2)
X

˙UN∈e
UN(vN,k)
|H( ˙UL)|.

By |H( ˙UN)| ≤
  n
vN

vN!
aut( ˙UN) ≤nvN , we have:

PN(vN, k) ≤R4(k+2)| eUN(vN, k)|.

In this part, the total number of unlabeled non-decorated graphs ˙UN with vN + 2 vertices and excess

k is bounded by βvN+1 (
vN +2
2 )
k+1

≤βvN+1(vN + 2)2(k+1). This is because for a unlabeled connected
graph with vN + 2 vertices, we can construct a spanning tree with vN + 1 vertices first and than add
additional k + 1 edges connecting some of the vN + 2 vertices. We can also bound the number of
unlabeled ˙UN as vNβvN  (
vN +2
2 )
k+2

, which is constructing two trees rooted at i and j, with a total of
vN ways of vertex number allocation, and then adding an additional k + 2 edges. The latter bound is
looser. Also, there are at most 11vN+k+2 ways of decoration. Thus,

| eUN(vN, k)| ≤βvN+1(vN + 2)2(k+1)11vN+k+2 ≤βvN+1(vN + 2)2(k+2)11vN+k+2.
Lastly, vN is upper bounded by (K + M)(|NN(i)| + |NN(j)|) by Claim 4 in [42].

52


---Page Break---
K
Proof of Proposition 4

In this section, we prove Proposition 4, which extends the Proposition 1 to the case when we can only
perform almost exact community recovery with a single graph. We denote bσ := (bσA, bσB), which is
the combination of the community label estimates for both graphs.

Proof of Proposition 4. By definition of the similarity score and H,

E[Φbσ
ij1H] =
X

H∈T
aut(H)
X

S(i)∼
=H

X

T (j)∼
=H
E[A
bσA
S B
bσB
T 1H]

= (1 + o(1))
X

H∈T
aut(H)
X

S(i)∼
=H

X

T (j)∼
=H
E
h
E[A
bσA
S B
bσB
T
| σ∗, bσ]1H
i
.

Define C as the edge collection of the intersection graph of S and T: C := E(S) ∩E(T).

E[Φbσ
ij1H]

∼
X

H∈T
aut(H)
X

S(i)∼
=H

X

T (j)∼
=H
E



E[
Y

e∈C

A
bσA
e B
bσB
e
Y

e′∈E(S)\C

A
bσA
e′
Y

e′′∈E(T )\C

B
bσB
e′
| σ∗, bσ]1H





∼
X

H∈T
aut(H)
X

S(i)∼
=H

X

T (j)∼
=H
E
 Y

e∈C
E[A
bσA
e B
bσB
e
| σ∗, bσ]
Y

e′∈E(S)\C
E[A
bσA
e′
| σ∗, bσ]

×
Y

e′′∈E(T )\C
E[B
bσB
e′
| σ∗, bσ]1H


.

We perform case studies for E[Φbσ
ij1H] based on the structure of S, T as a union graph and then later
sum each case up.

(a) S = T, that is, all edges appear in pairs (this case is only possible for i = j).
There are
(1 + o(1))nN/aut(H) labeled union graphs of S1 and T1 satisfying this condition. In this case, every
edge is 2-decorated and it no longer matters whether bσ gives a correct output or not, as we can show
the following upper bound.

E[Φbσ
ij1H](A) = (1 + o(1))
X

H∈T
aut(H)
nN

aut(H)E

"Y

e∈C
E[A
bσA
e B
bσB
e
| σ∗, bσ]1H

#

≤(1 + o(1))|T |nN X

N+
P(ζ = N+)
X

N′
P(ηc+ = Nc+, ηc−= Nc−|ζ = N+)

× (ρ+σ2
+)Nc+(ρ−σ2
−)Nc−(ρ+σ2
+ + ∆2)N+−Nc+(ρ−σ2
−+ ∆2)N−N+−Nc−

≤(1 + o(1))|T |nN
N
X

N+=0


N
N −1

 1

2N (ρσ2
+ + ∆2)N+(ρσ2
−+ ∆2)N−N+

= (1 + o(1))|T |nN(ρσ2
eff + ∆2)N,
where ζ is the number of in-community edges out of the N edges in S, ηc+ is the number of in-
community edges that are centralized incorrectly, ηc−is the number of cross-community edges that
are centralized incorrectly, and ∆:= |p −q|. (i) The first equality holds by definition and counting
cases. (ii) The second inequality holds because there are Nc+(Nc−) in(cross)-community edges
centralized correctly, each of which contributes the same as E[AeBe | σ∗] = (1+o(1))ρ+σ2
+(ρ−σ2
−)

(Lemma 13). For the remaining edges, N+ −Nc+ (N −N+ −Nc−) of them has E[A
bσA
e B
bσB
e
|

σ∗, bσ] = (1 + o(1))(ρ+σ2
+) + E[A
bσA
e
| σ∗, bσ]E[B
bσB
e
| σ∗, bσ] ≤(1 + o(1))(ρ+σ2
+ + ∆2). (iii)
The third inequality holds because ∆2 > 0, Lemma 9 gives the distribution of ζ, and ρ = (1 +
Θ( log n

n ))ρ+, ρ = (1 + Θ( log n

n ))ρ−. (iv) The last equality holds from the binomial theorem.

Observe that ∆2 = (1 + Θ( log n

n ))(ρσ2
eff). Assume that N = O(log n),

E[Φbσ
ij1H](A) = (1 + o(1))|T |nN(ρσeff)N.

53


---Page Break---
s
s
s
s

Figure 8: Left: One possible labeling such that all edges are centralized incorrectly, with the colored
vertices indicating those that are labeled incorrectly and the black vertices indicating those that are
labeled correctly. Right: Another possible labeling such that all edges are centralized incorrectly.

(b) S and T have no common edges, that is, E(S) ∩E(T) = ∅.
We can use the trivial bound on
the labeled S and T as
n2N
aut(H)2 .

E[Φbσ
ij1H](B)

∼
X

H∈T
aut(H)
X

S(i)∼
=H

X

T (j)∼
=H:C=∅
E




Y

e′∈E(S)
E[A
bσA
e′
| σ∗, bσ]
Y

e′′∈E(T )
E[B
bσB
e′
| σ∗, bσ]1H





≲|T |
n2N

aut(H)P(E)∆2N,
(70)

where E denotes the event {bσ : Q
e∈E(S) E[A
bσA
e ] Q
e∈E(T ) E[B
bσB
e
] = Θ(∆e(S)+e(T ))} ∩H, that is,
every edge is centralized correctly and H happens. This inequality is true because conditioned on H
and bσ being correct, E[A
bσA
e
| σ∗, bσ]1H = o(n−D+(a,b,s,ε)), the upper bound of the probability that
one vertex on this edge being labeled incorrectly.

We further denote E′ as {bσA : Q

e∈E(S) E[A
bσA
e ] = Θ(∆e(S))}∩H. It is obvious that P(E) ≤P(E′).
Then, we need to upper bound the probability of bσA giving incorrect centralization for all edges
in S ∪T.

We observe that there are only two situations, that is, no vertices in S that has a neighbor with the
same label correctness as itself. See Figure 8 for an illustration. We denote those two possible
outcomes constraint on S ∪T as bσ1 and bσ2. P(E′) = P({bσA = bσ1} ∩H) + P({bσA = bσ2} ∩H).

By the label correctness, we separate V (S) into two disjoint sets: V (S)bσ
c for the correctly labeled
vertices and V (S)bσ
ic for the incorrectly labeled vertices. The deterministic relationship between
bσ1 and bσ2 is: V (S)bσ1
ic = V (S)bσ2
c . without loss of generality, we assume |V (S)bσ1
c | > |V (S)bσ1
ic |.
Observe that event {bσA = bσ1} (resp. {bσA = bσ2}) is equivalent as saying the set of vertices on odd
(resp. even) levels are labeled incorrectly (falling in the bad vertex set Iε as defined in Section D).

Denote pa,b,s,ε,δ,V := n−V ×D+(a,b,s,ε) + n−εδ(1−o(1)) log n. We can apply Lemma 8,

P({bσA = bσ1} ∩H) ≤(1 −pa,b,s,ε,δ,|V (S)bσ1
c
|)(pa,b,s,ε,δ,|V (S)bσ1
ic |),

P({bσA = bσ2} ∩H) ≤(1 −pa,b,s,ε,δ,|V (S)bσ1
ic |)(pa,b,s,ε,δ,|V (S)bσ1
c
|),

P(E′) ≤2(1 −pa,b,s,ε,δ,|V (S)bσ1
c
|)(pa,b,s,ε,δ,|V (S)bσ1
ic |).

Consider each chandelier has L branches and each has a M-wire (Assume that M = Θ(
log n
log logn) as

in condition (3)). Even if we don’t know the structure of bulbs, we have |V (S)bσ1
ic | = Ω(
log n
log log n),
because at least half of the vertices on wires should be labeled incorrectly.

For some constant c1 > 0 such that |V (S)bσ1
ic | ≥c1
log n
log log n,

P(E) ≤P(E′) ≤O(pa,b,s,ε,δ,|V (S)bσ1
ic |) ≤O(n−D+(a,b,s,ε)c1
log n
log log n ).

54


---Page Break---
By substituting P(E) and ∆2N into (70), we have

E[Φbσ
ij1H](B) ≤O

|T |
n2N

aut(H)n−[D+(a,b,s,ε)]c1
log n
log log n (ρσ2
eff)2N(ν)2N

.

Recall that µ = |T |nN(ρσ2
eff)N. Denote ν2 =
∆2

(ρσ2
eff)2 (= (1 + o(1)) 2(a−b)

ρ(a+b)). For some constant

c2 > 0 such that ρσ2
eff ≤c2
log n

n ,

E[Φbσ
ij1H](B) ≤O

µcN
2 ν2N(log n

n
)NnNn−D+(a,b,s,ε)c1
log n
log log n


= O

µcN
2 ν2N
(log n)N

nc1D+(a,b,s,ε)
log n
log log n


.

For some constant c3 > 0 such that N = c3 log n, as in assumption (3),

= O

µ((c2ν2 log n)c3 log log n

nc1D+(a,b,s,ε)
)

log n
log log n


= o(µ/n2).

(c) S and T have some common edges, that is, E(S) ∩E(T) ̸= ∅.
There are at most [n2N −
nN(n −N)N]/aut(H)2 = o(nN)nN/aut(H)2 cases in the enumeration of S and T.

E[Φbσ
ij1H](C) ≲
X

H∈T
aut(H)

N−1
X

M=1

X

S(i)∼
=H,T (j)∼
=H:C̸=∅,Xd=M
(ρσ2
eff + ∆2)N−M∆2MP(E′′),

where we denote by E′′ the event

{bσ :
Y

e′∈E(S)\C
E[A
bσA
e′ ]
Y

e′′∈E(T )\C
E[B
bσB
e′ ] = Θ(∆)|E(S)\C|+|E(T )\C|} ∩H.

Let Xd denote the number of different edges between S and T under the true permutation. The
structures of S and T such that {Xd = M} happens, P

S(i)∼
=H
P

T (j)∼
=H 1{Xd=M}, is upper

bounded by
nN
aut(H) ×
nM
aut(H) as changing M edges from S to T allows changing at most M vertices
for a tree. Then, we have

E[Φbσ
ij1H](C) ≲|T |nN
N−1
X

M=1
(ρσ2
eff + ∆2)N−MnM∆2MP(E′′)

≲|T |nN
N−1
X

M=1
(ρσ2
eff)N−MnM(ρσ2
eff)2M(2(a −b)

ρ(a + b))2MP(E′′)

= (1 + o(1))µ

N−1
X

M=1
(nρσ2
eff)Mν2MP(E′′).
(71)

We observe the following:
P(E′′) ≤n−D+(a,b,s,ε)(2M/2D).
This inequality holds because there are 2M edges centralized incorrectly (required from event E′′)
in S and T and each incorrect labeling of a vertex can lead to incorrect centralization on at most D
edges. Therefore, at least 2M/2D vertices should be wrongly labeled for E′′ to happen.

Substitute P(E′′) into (71). Assuming that D = o(
log n
log log n) (3),

E[Φbσ
ij1H](C) ≲µ

N−1
X

M=1
(
nρσ2
effν2

n
1
D D+(a,b,s,ε) )M ≲µ

N−1
X

M=1
(
ν2c1 log n

n
1
D D+(a,b,s,ε) )M = o(µ).

In summary,

E[Φbσ
ij1H] =
E[Φbσ
ij1H](A) + E[Φbσ
ij1H](B) + E[Φbσ
ij1H](C) = (1 + o(1))µ
if j = π∗(i),
E[Φbσ
ij1H](B) + E[Φbσ
ij1H](C) = o(µ)
if j ̸= π∗(i).

55


---Page Break---
Remark 6. (Denser regime) If we are not restricting the sparse regime p = a log n

n
and q = b log n

n
, we
are interested in what general p, q conditions are for Lemma 4 to hold. Assume that p∨q = O(n−c(n)),
then the geometric series PN−1
M=1
nρσ2
effν2

n
1
D D+(a,b,s,ε) converges if and only if c(n) > 1 −D+(a,b,s,ε)

D
.

L
Proof of Proposition 5

L.1
Proof of the Proposition

In this section, we analyze the second moment of similarity score. We expect the variance of Φij to
be infinitely small in comparison with the squared expectation of true pair’s similarity score.

Proof of Proposition 5. Recall that S1, T1 are rooted on i and S2, T2 are rooted on j.

Var[Φbσ
ij1H] =
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
Cov(A
bσA
S1 B
bσB
S2 1H, A
bσA
T1 B
bσB
T2 1H)

=
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V1

−
(72)

X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[A
bσA
S1 B
bσB
S2 1H]E[A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V2

(a) Analyzing V2.
We first give the upper bound of the latter part. When analyzing with correct
centralization, we ignore this part as it is non-negative in the correct centralization case. However, in
this case, it is possible to be negative because there can be odd number of edges occurring once and
also being incorrectly centralized.

After factorizing V2,

V2 =



X

H∈T
aut(H)
X

S1,S2∼
=H
E[A
bσA
S1 B
bσB
S2 1H]







X

I∈T
aut(I)
X

T1,T2∼
=I
E[A
bσA
T1 B
bσB
T2 1H]



,

we can see that V2 ≥0, and thus we have Var[Φbσ
ij1H] ≤V1 for j = π∗(i).

(b) Analyzing V11.
The main challenge here is that we do not have the condition that every edge
occurs at least twice in the union graph as in the analysis of Regime I. However, we can put union
graphs into two categories based on whether every edge is at least 2-decorated or not. We keep the
notation of Wij as the collection of decorated union graphs ˙U that are at least 2-decorated and we
decompose (72) as follows

V1 =
X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

=
X

˙U∈Wij
(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V11

+

+
X

˙U /∈Wij
(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V12

.

We first show that V11/µ2 = O

L2
ρ2ns(p∧q) +
L2

ρ2(K+M)|J |

.

56


---Page Break---
The incorrect centralization affects the upper bound of moments as the following:

E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

= E



g ˙U(σ+, σ−)
Y

2≤ℓ+m≤4

Y

(u,v)∈Kℓm
σ−(ℓ+m)
c(u,v)
E[A
bσA,ℓ
uv
B
bσB,m
uv
| σ∗, bσ]1H





= E



g ˙U(σ+, σ−)
Y

2≤ℓ+m≤4
βe(Kℓm)
l,m
1H





≤ρe(K11)(sp ∧sq)−2N+e(U)σ4N
eff (1 + O(log n

n
))e(U)

≤ρe(K11)(sp ∧sq)−2N+e(U)σ4N
eff (1 + O(log n

n
))4N.
(73)

The first two equalities follow from definitions. The third inequality holds because of the upper
bounds of βe(Kℓm)
ℓ,m
from Lemma 14 holds for all bσ.

Then, the structures of the union graph, alone with the assigned weights are bounded the same as in
Section I. This implies that
V11
E[Φiπ∗(i)1H]2 ≤O

L2
ρ2ns(p∧q) +
L2

ρ2(K+M)|J |

under the same condition
as (22).

(c) Analyzing V12.
To conclude
Var[Φbσ
ij1H]
E[Φiπ∗(i)1H]2 ≤
V11+V12+V2
E[Φiπ∗(i)1H]2 = O

L2
ρ2ns(p∧q) +
L2

ρ2(K+M)|J |

,

it remains to show V12 = o

L2
ρ2ns(p∧q) +
L2

ρ2(K+M)|J |

. Lemma 24 gives a even stronger result,

because as assumed in Proposition 5, L = o(log n)), giving
L2
ρ2ns(p∧q) +
L2

ρ2(K+M)|J | ≫n−ε′ for all
ε′ > 0.

Lemma 24. Under the same conditions as Proposition 5, for some ε′ > 0,

V12
E[Φiπ∗(i)1H]2 = o(n−ε′).

Proof. For the expectation inside V1 part, it can be separated as eight sets Kℓm:

E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H] = E



g ˙Ug−1
˙U E[
Y

ℓ∈[2],m∈[2],ℓ+m≥1

Y

e∈Kℓm

A
bσA,ℓ
e
B
bσB,m
e
| σ∗, bσ]1H



,

where g ˙U is the abbreviation for g ˙U(σ+, σ−).

Conditioned on σ and bσ, approximately centered edges are still independent with each other,

E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H] = E



g ˙Ug−1
˙U
Y

ℓ∈[2],m∈[2],ℓ+m≥1

Y

e∈Kℓm
E[A
bσA,ℓ
e
B
bσB,m
e
| σ∗, bσ]1H



,

Lemma 14 gives the upper bound over ηℓ,m := σ−(ℓ+m)
c(e)
E[A
bσA,ℓ
e
B
bσB,m
e
| σ∗, bσ] and thus by
enumerating through the products, we have

g−1
˙U
Y

ℓ∈[2],m∈[2],ℓ+m≥1

Y

e∈Kℓm
E[A
bσA,ℓ
e
B
bσB,m
e
| σ∗, bσ]

≤(1 + Θ(log n

n
))4Nρe(K11)(sp ∧sq)−1

2 (e(K12)+e(K21))−e(K22)P(E)∆z/2,

≤(1 + o(1))ρe(K11)(sp ∧sq)v+k+1−2NP(E)(|a −b|

a ∧b )z/2,

57


---Page Break---
where E := {Every 1-decorated edges are centralized incorrectly} ∩H and z is the number of 1-
decorated edges, namely, z := e(K01) + e(K10). The second inequality holds because −1

2(e(K12) +
e(K21)) −e(K22) = (v + k + 1) −2N −z/2.

According to Lemma 8 and the assumption that the maximum degree of chandelier is no greater than
D, we have for any ε > 0,
P(E) ≤n−D+(a,b,s,ε) z

D .
Putting things together,

E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H] ≤(1+o(1))n−D+(a,b,s,ε) z

D (sp∧sq)v+k+1−2Nρe(K11)E[g ˙U(σ+, σ−) | H].

It remains to calculate E[g ˙U(σ+, σ−) | H].

Similar as in the calculation in Section I, where we only consider ˙U being at least 2-decorated.
Here, we need to generalize it into the case when ˙U has 1-decorated edges. ˙U can be decomposed
into a tree with an additional set of edges connecting vertices on the tree. We assume that there
are Ai i-decorated edges on the tree and Bi i-decorated edges in the additional edge set of size
e( ˙U) −v( ˙U) + 1, for i ∈[4]. We apply the Corollary 1 and have that P(X(1,a) = a1, X(2,a) =
a2, X(3,a) = a3, X(4,a) = a4, X(1,b) = b1, X(2,b) = b2, X(3,b) = b3, X(4,b) = b4 | H) ≤
(1+o(1))
 A1
a1
 A2
a2
 A3
a3
 A4
a4

1
2A1+A2+A3+A4 , where X(i,a) is the number of i-decorated in-community
edges on the tree-part of ˙U and X(i,b) is the number of i-decorated in-community edges among the
additional edge set.

Ai and Bi are fixed but summed up to di for each ˙U. ai (bi) takes possible values from 0 to Ai (Bi),
for i ∈[4]. The number of i-decorated in-community edges is ai + bi, and the number of i-decorated
cross-community edges is di −ai −bi = (Ai −ai) + (Bi −bi).

E [g ˙U(σ+, σ−) | H]

≤(1 + o(1))

A1
X

a1

A2
X

a2

A3
X

a3

A4
X

a4

B1
X

b1

B2
X

b2

B3
X

b3

B4
X

b4
σ(a1+b1)
+
σ2(a2+b2)
+
σ2(d2−a2−b2)
−
σ3(a3+b3)
+
σ3(d3−a3−b3)
−

× σ4(a4+b4)
+
σ4(d4−a4−b4)
−

A1
a1

A2
a2

A3
a3

A4
a4


1
2A1+A2+A3+A4

= (1 + o(1))
σ+ + σ−

2

A1 σ2
+ + σ2
−
2

A2 σ3
+ + σ3
−
2

A3 σ4
+ + σ4
−
2

A4

×

B1
X

b1=0
σb1
+ σ(B1−b1)
−

B2
X

b2=0
σ2b2
+ σ2(B2−b2)
−

B3
X

b3=0
σ3b3
+ σ3(B3−b3)
−

B4
X

b4=0
σ4b4
+ σ4(B4−b4)
−

≤(1 + o(1))
σ+ + σ−

2

d1 σ2
+ + σ2
−
2

d2 σ3
+ + σ3
−
2

d3 σ4
+ + σ4
−
2

d4
2k+1,

where the last inequality holds because the upper bound holds with multiplying a binomial coefficient
 B1
b1
 B2
b2
 B3
b3
 B4
b4

and that P

i Bi = k + 1. By the definition of γ2 and γ1,

E [g ˙U(σ+, σ−) | H] ≤(1 + o(1))
σ+ + σ−

2

d1
σ2d2+3d3+4d4
eff
γd3
1 γd4
2 2k+1.

Define γ0 := ( σ++σ−

2
)/σeff and we can see that γ0 < 1. So, we can upper bound ( σ++σ−

2
)d1 as σd1
eff.
Because γ1 < γ2 and d3 + d4 = e(K12) + e(K21) + e(K22),

E [g ˙U(σ+, σ−) | H] ≤(1 + o(1))σ4N
eff γe(K12)+e(K21)+e(K22)
2
2k+1.

In summary,

E[AS1BS2AT1BT21H] ≤(1 + o(1))n−D+(a,b,s,ε) z

D (sp ∧sq)v+k+1−2Nρe(K11)

× σ4N
eff γe(K12)+e(K21)+e(K22)
2
2k+1.
(74)

58


---Page Break---
After plugging the upper bound on cross-moments to the ratio, it remains to bound the number of
different union graph structures and their corresponding weights. In parallel to Wij, we define Sij as
the collection of decorated union graphs that have at least one 1-decorated edge. Sij(v, k) denotes
those with v + 1 + 1j̸=π∗i vertices and excess k.

V12
E[Φiπ∗(i)1H]2

≤

P4N
v+k+1=0
P
˙U∈Sij(v,k) aut(H)aut(I)ρe(K11)(sp ∧sq)v+k+1−2Nγe(K12)+e(K21)+e(K22)
2
2k+1

(1 + o(1))n2Nρ2N|T |2nzD+(a,b,s,ε)/D

We define the ( ˙UL, ˙UM, ˙UN) partition of decorated union graph as (44) in Section I. We define
UL(vL, z, ℓ) as the collection of ˙UL that has vL vertices, ℓedges belonging to set e(K11), and no
more than z 1-decorated edges. We define UM(vM) as the collection of ˙UM that has vM vertices.
We define UN(vN, k) as the collection of ˙UN that has vN vertices and excess k. We also keep the
notation of eU as the corresponding unlabeled decorated union graph sets. In addition,

PL(vL, z, ℓ) :=
X

˙UL∈UL(vL,z,ℓ)
w( ˙UL),
(75)

PM(vM) :=
X

˙UL∈UL(vM)
w( ˙UM),
(76)

PN(vN, k) :=
X

˙UL∈UL(vN,k)
w( ˙UN).
(77)

From the above partition,

V12
E[Φiπ∗(i)1H]2

≤

P4N
v=N
P4N−v
k+1=0(sp ∧sq)v+k+1−2N2k+1 P

z
P

vL,vM,vN
P

ℓρℓPL(vL, z, ℓ)PM(vM)PL(vN, k)

(1 + o(1))n2Nρ2N|T |2nzD+(a,b,s,ε)/Dγ2(v+k+2)−4N
2
.

We show upper bounds for PL(vL, z, ℓ) in Lemma 25, which is

2N
X

ℓ=0
ρℓPL(vL, z, ℓ) ≤(4N)2z+1nvLL(4LM)6L|T |2ρ2Nρ−z

2 .

The upper bounds for PM and PN trivially follows from Lemma 19 and Lemma 20, with a replace-
ment of 11 to 15 as the possible decorations of each vertex increase by 4 for 1-decoration and a
different maximum value of eM, parameterized by v, k, z.

PM(vM) ≤R

2eM

M nvM (15β)(K+M) 2eM

M 1{eM≤2N−(v+k+2)+z/2}
PN(vN, k) ≤nvN (15β)vN (15R4(vN + 1)2)k+11{vN≤2(K+M)(2k+2)}.

Putting all the pieces together, we have

V12
E[Φiπ∗(i)1H]2 ≤
X

k≥−1

30R4(2N + 1)2(15β)4(K+M)

n

k+1

×

4N−k−1
X

v=N

 
R
2
M (15β)2 K+M

M
ns(p ∧q)
γ2
2

!2N−v−k−1

×

4N
X

z=1∨(2(v+k+1)−4N)



(4N)2R
1
M (15β)
K+M

M
√ρn

D+(a,b,s,ε)

D





z

× 2NL(4LM)6L.

59


---Page Break---
From condition (24), 2NL(4LM)6L ≤log3 n. Also, since with (24) and γ2 < 2,

R
2
M (15β)2 K+M

M
ns(p ∧q)
γ2
2 ≤1

2,
15R4(2N + 1)2(15β)4(K+M)

n
≤1

2.

For v ≤2N −k −1, we have

2N−k−1
X

v=N

 
R
2
M (15β)2 K+M

M
ns(p ∧q)
γ2
2

!2N−v−k−1

≤2.

For the first summation, we always have

X

k≥−1

30R4(2N + 1)2(15β)4(K+M)

n

k+1
≤2.

For the last summation with those additional terms, from condition (24), we have

4N
X

z=1



(4N)2R
1
M (15β)
K+M

M
√ρn

D+(a,b,s,ε)

D





z

× 2NL(4LM)6L = o(n−ε′),

for some ε′ > 0 because nD+(a,b,s,ε)/D is the only term being polynomial.

If v > 2N −k −1, then we know that z > 2(v + k + 1 −2N). The summation over v and z together
is upper bounded by

4N−k−1
X

v=2N−k



R
2
M (15β)2 K+M

M
ns(p ∧q)
γ2
2 × (4N)2R
1
M (15β)
K+M

M
√ρn

D+(a,b,s,ε)

D





v+k+1−2N

2NL(4LM)6L = o(n−ε′),

because, again nD+(a,b,s,ε)/D is the only term being polynomial.

In summary, we have
V12
E[Φiπ∗(i)1H]2 = o(n−ε′) for some ε′ > 0, which completes the proof.

L.2
Proof of auxiliary Lemmas

Lemma 25. For true pairs,

2N
X

ℓ=0
ρℓPL(vL, z, ℓ) ≤(4N)2z+1nvLL(4LM)6L|T |2ρ2Nρ−z

2 .

Proof. We define the unlabeled union graph sets corresponding to U(vL, z, ℓ) as eU(vL, z, ℓ). From
definition (60),

From the definition (75) and Claim 1,

2N
X

ℓ=0
ρℓPL(vL, z, ℓ) ≤

2N
X

ℓ=0
ρℓ
X

˙UL∈e
UL(vL,z,ℓ)

nvLw( ˙UL)

aut( ˙UL)
≤

2N
X

ℓ=0
nvLρℓ| eUL(vL, z, ℓ)|(2K)z.
(78)

Recall that eL = 1

2(e(K12 ∪K22 ∩˙UL) + e(K21 ∪K22 ∩˙UL)). The total number of edges on
S1, T1, S2 and T2 involved in ˙UL is (2(vL +eL)−z) and thus the total number of bulbs on S1, T1, S2
and T2 involved in ˙UL is b := 2(vL+eL)−z

K+M
< 4L. Those b bulbs can be partly or fully overlapped
(namely, tangled) with another stay on their own. From the definition of ˙UL (44), it is impossible
to have three or more bulbs tangling with each other. If two bulbs are tangling with each other, we
put them into a pair. If a bunch of bulbs are all not tangling with any other bulbs, we pair them up
arbitrarily. We denote t1 as the number of pairs of bulbs that have decorations being a subset of
{S1, S2} or {T1, T2}. For all 2-decorated edges among these pairs, they are in K11. Since ˙UL has at
most z 1-decorated edges, we have
ℓ≥t1K −z/2.
(79)

60


---Page Break---
Next, we introduce three types of bulbs. The first type is called effective non-isomorphic bulbs, which
is a selection of bulbs that are not isomorphic to each other and always pair with a bulb that are not of
the same type. The selection is not unique and we take the largest possible set of bulbs satisfying
those rules as the set of effective non-isomorphic bulbs. Fixed the effective bulb set, for bulbs that are
isomorphic to those effective non-isomorphic bulbs, we name them as shadow effective bulbs. For
the remaining bulbs, we name them as non-effective bulbs. We have the following Claim 2:

t1

2 ≤a ≤L + t1

2 .
(80)

From definition, there is at most one non-effective bulb and at most one effective non-isomorphic
bulb in each pair of bulbs, while two shadow bulbs can pair up.

We call those effective non-isomorphic bulbs as effective because when enumerating through the
chandelier structures, we let them having the priority of taking any possible structure from J and
serving the base of that pair. Shadow effective bulbs are named so because they mirror the structure
of effective non-isomorphic bulbs and thus will not increase the union graph richness too much. For
non-effective bulbs, we let them take any possible structures with the constrain that there are at most
z 1-decorated edges in ˙UL.

We denote the number of effective non-isomorphic bulbs as a. For all pairs, we upper bound the
number of different non-isomorphic tangled bulbs as following combinatorial factor
|J |
a

b
a

2N
z/2


(4N)
z
2 ,

where
 |J |
a

comes from the structure of a effective non-isomorphic bulbs,
 b
a

is the upper bound of
choosing a effective non-isomorphic bulbs from b bulbs,
 2N
z/2

is the upper bound on the selection of
which edges on effective non-isomorphic bulbs and shadow effective bulbs are overlapped as there are
at most 2K (< 2N) 2-decorated edges on bulbs if all bulbs are perfectly overlapping with one another,
and (4N)
z
2 bounds the placement of those remaining 1-decorated edges from the non-effective bulbs
as each of them has at most 4N possible vertices to attach to on the union graph.

Since wires cannot tangle with bulbs (otherwise, it is not a tree), we bound the possible structures
separately. There can be at most 4 wires tangling with each other, from top to bottom. We apply a
very loose bound even without using this fact, which is (b −1)!M b−1. This is because there are at
most b wires and we determine the structure of wires on the union graph one by one. When the t-th
wire comes in, it can determine which of the t −1 wires to tangle with and the length of overlap,
from 0 to M.

Putting together (79) and (80) with the combinatorial observations, we have

ρℓ| eUL(vL, z, ℓ)| ≤

2L
X

a=0

4L
X

b=2a

a
X

t1=0
ρt1K−z

2 1{a≤L+ t1

2 }|J |a
b
a


(2N)3z/2(b −1)!M

≤|T |(4N)zρ−z

2

2L
X

t1=0
(|J |

t1

2 ρt1K)

2L
X

a=0

4L
X

b=2a

b
a


(b −1)!M b−1

≤(4LM)6L|T |(4N)zρ−z

2

2L
X

t1=0
(|J |ρ2K)

t1

2

≤L(4LM)6L|T |2ρ2N(4N)zρ−z

2 .
(81)

In the second inequality, we loose the upper bound of t1 from a to 2L and change the or-
der of summation.
In the third inequality, we bound the summation over a and b.
Lastly,
P2L
t1=0(|J |ρ2K)
t1

2 ≤L(|J |ρ2K)L = L|T |ρ2N.

Plugging (81) back to (78), after a summation over ℓ, we complete the proof.

Claim 1. Assume that j = π∗(i). For any arbitrary ˙UL ∈UL(vL, z, ℓ), we have

w( ˙UL) ≤aut( ˙UL)(2K)z.

61


---Page Break---
Proof. Denote all bulbs contained in ˙UL as B1, B2, . . . , Bw. (1) Some of them can be fully overlapped
to form a 2-decorated bulbs in the union graph. (2) Some of them can be partly overlapped. (3) And
the remaining of them are fully 1-decorated. There cannot be three or more bulbs overlapping with
each other thanks to the definition of ˙UL (65).

Each bulb Bi contributes to the w( ˙UL) by aut(Bi) independently from definition (66) and (50).
For any vertex on the bulbs, it will not be at the same orbit as any vertex on the wire, so studying
the automorphism of the overlapped bulbs gives a lower bound on the automorphism of the whole
decorated graph. Since each bulb occurs in at most one overlapped bulb, to prove the claim, it
suffices to examine the relationship between weights and automorphism for each of the three cases
aforementioned.

For i, j ∈[w], if bulbs Bi is partly overlapping with Bj. From Corollary 2,
p

aut(Bi)aut(Bj) ≤
aut(Bi ∪Bj)(2K)|E(Bi)△E(Bj)|. If Bi is fully overlapping with Bj, then
p

aut(Bi)aut(Bj) =
aut(Bi ∪Bj). If Bi is fully 1-decorated, then
p

aut(Bi) ≤aut(Bi).

Denote I1 and I2 as the collections of index pairs that the corresponding bulbs fall in case (1) or (2).
Denote I3 as the collection of indices corresponding to the bulbs falling in case (3). Therefore,

w( ˙UL) ≤
Y

(i,j)∈I1
aut(Bi ∪Bj)(2K)|E(Bi)△E(Bj)|
Y

(i,j)∈I2
aut(Bi ∪Bj)
Y

i∈I3
aut(Bi)

≤aut( ˙UL)(2K)
P

(i,j)∈I |E(Bi)△E(Bj)|.
(82)

The union graph has at least 2 × P

(i,j)∈I |E(Bi)△E(Bj)| 1-decorated edges and we know that
˙UL has at most z 1-decorated edges. Therefore, P

(i,j)∈I |E(Bi)△E(Bj)| ≤z. Substituting this
into (82) completes the proof.

Claim 2. Let a be the number of effective non-isomorphic bulbs and t1 be the number of pairs of
tangled bulbs that are decorated by a subset of either {S1, S2} or {T1, T2}. We have

t1

2 ≤a ≤L + t1

2 .
(83)

Proof. For an arbitrary bulb, it occurs at most one time in S1, S2, T1, T2 each and they are paired up
into two sets, a ≥t1

2 . Assume that there are t′ non-isomorphic bulbs among the t1 pairs, they can all
be assigned as effective non-isomorphic bulbs. Then, without loss of generality, S1, S2 have at most
L −t′

2 bulbs unspecified. By assumption, they cannot pair up with each other, so every one bulb from
S1, S2 remaining will pair up with another bulb from T1, T2. Therefore, the remaining bulbs have at
most L −t′

2 effective non-isomorphic bulbs.

Together, we have a ≤t′ + (L −t′

2 ) ≤L + t1

2 , as t′ ≤t1.

M
Proof of Proposition 6

M.1
Proof of the Proposition

Proof of Proposition 6. Recall that S1 and T1 are rooted on i, S2 and T2 are rooted on j. The
minimum value of k is −2 when the union graph consists of two disconnected trees, S1 ∪T1 and
S2 ∪T2. We use the same notation for different parts of the variance as in Section L.

62


---Page Break---
Var[Φbσ
ij1H] =
X

H,I∈T
aut(H)aut(I)
X

S1(i),S2(i)∼
=H,T1(j),T2(j)∼
=I
Cov(A
bσA
S1 B
bσB
S2 1H, A
bσA
T1 B
bσB
T2 1H)

=
X

U∈Wij
(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V11

+
X

U /∈Wij
(aut(S1)aut(S2)aut(T1)aut(T2))
1
2 E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V12

−

X

H,I∈T
aut(H)aut(I)
X

S1,S2∼
=H,T1,T2∼
=I
E[A
bσA
S1 B
bσB
S2 1H]E[A
bσA
T1 B
bσB
T2 1H]

|
{z
}
V2

.
(84)

Since V2 ≥0, it suffices to bound the first two summations. The same argument in Section L to
bound the V11 for true pairs works for this case with an additional fluctuation coming from using
Lemma 14 to bound the cross moments rather than Lemma 13. We have

V11/µ2 ≤(1 + Θ(log n

n
))4N Var[Φbσ
ij1H]
µ2
|sD+(a,b)>1 = O(
1
|T |ρ2N )

under conditions (23). The additional fluctuation comes from the cross moment bounds.

Lemma 26 shows that V12/µ2 = o(
1
|T |ρ2N ). In summary, we have
Var[Φbσ
ij1H]
E[Φiπ∗(i)1H]2 = O(
1
|T |ρ2N ).

Lemma 26. Under the same conditions as Proposition 6,

V12

µ2 = o(
1
|T |ρ2N ).

Proof. First, we apply the upper bound on E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H] as derived in (74), with replac-
ing k + 1 to k + 2 everywhere as from definition v is the number of vertices except for i and j. When
j ̸= π∗(i), the minimum value of j starts from −2 when the union graph consists of two disjoint
trees.

E[A
bσA
S1 B
bσB
S2 A
bσA
T1 B
bσB
T2 1H] ≤(1 + o(1))n−D+(a,b,s,ε) z

D (sp ∧sq)v+k+2−2Nρe(K11)

× σ4N
eff γe(K12)+e(K21)+e(K22)
2
2k+2.
(85)

After plugging the upper bound on cross-moments to the ratio, it remains to bound the number of
different union graph structures and their corresponding weights.

V12
E[Φiπ∗(i)1H]2

≤

P4N
v+k+2=0
P
˙U∈Sij(v,k) aut(H)aut(I)ρe(K11)(sp ∧sq)v+k+2−2Nγe(K12)+e(K21)+e(K22)
2
2k+2

(1 + o(1))n2Nρ2N|T |2nzD+(a,b,s,ε)/D
.

(a) Case k = −2.
We first consider the special case when k = −2. When k = −2, there are two
disjoint trees in the decorated union graph and all edges are decorated by a subset of either {S1, T1}
or {S2, T2}. Then, e(K11) = e(K12) = e(K21) = e(K22) = 0. Also v ≥2N,

V12 = σ4N
eff
X

˙U∈Sij(v≥2N)
aut(H)aut(I)n−
zD+(a,b,s,ε)

D
≤2σ4N
eff Fij,

where
Fij :=
X

H∈T

X

I∈T :aut(I)≤aut(H)
aut(H)2
X

˙U∈Sij(v≥2N,H,I)
n−
zD+(a,b,s,ε)

D
,

63


---Page Break---
and Sij(v ≥2N, H, I) is the collection of decorated union graphs that have at least 2N + 2 vertices,
excess −2, at least 1 edge 1-decorated, and that S1, S2 ∼= H, T1, T2 ∼= I.

For a specific decorated graph ˙U, we denote t1 (resp. t2) as the number of different edges vetween
S1 and T2 (resp. S2 and T2). There are z = 2(t1 + t2) 1-decorated edges and the remaining edges
are in set K02 or K20. We write out Fij under the summation over t1 and t2.

Fij =
X

H∈T

X

S1,S2∼
=H

X

∃I∈T ,aut(H)>aut(I),T1,T2∼
=I

N
X

t1=0

N
X

t2=0
n−
zD+(a,b,s,ε)

D
1{t1+t2≥1}

=
X

t1+t2≥1

X

H∈T
|S(v ≥2N, H, t1, t2)|n−
zD+(a,b,s,ε)

D
,

where S(v ≥2N, H, t1, t2) collects all the possible decorated union graph that have S1, S2 ∼= H,
T1, T2 ∼= I for some I ∈T such that aut(I) < aut(H), and S1 (resp. S2) differ in t1 (resp. t2)
edges with T1 (resp. T2).

Next, we bound |S(v ≥2N, H, t1, t2)| by the following way: First, enumerate through all S1, S2 ∼=
H on the complete graph with all possible structure, this gives
n2N
aut(H)2 . Then, we choose which

edges that are overlapped with T1 on S1 (resp. overlapped with T2 on S2). This is at most
 N
t1
 N
t2

≤
N t1+t2. After this, we draw t1 + t2 new vertices for T1 and T2 and allow them arbitrarily connecting
edges among those N vertices on its chandelier, which is upper bounded by
 N
2
t1+t2 (ignoring
the constraint that T1 ∼= T2 ∼= I for some chandelier I having less automorphism number than H).
Altogether,

|Sij(v ≥2N, H, I)| ≤
n2N

aut(H)2


N 3

n2D+(a,b,s,ε)/D

t1+t2
.

Therefore,

Fij = n2N|T |
X

t1+t2≥1


N 3

n2D+(a,b,s,ε)/D

t1+t2
.

As assumed in Proposition 6, N = Θ(log n) and D = o(
log n
log log n). Therefore,

V12/µ2 = o(
1
|T |ρ2N ).

(b) Case k > −2.
In general, we define ˙UL, ˙UM, and ˙UN partition the same as (64) and (65).
We also define the weights of each part the same as (66), (67), and (68). We define PL(vL, z) =
P
˙U∈UL(vL,z) w( ˙U). The definition of PM(vM) and PN(vN, k) follow. All union graph class should
not have more than z 1-decorated edges, but specifically we only need this constraint for ˙UL.

Note that e(K12) + e(K21) + e(K22) ≤4N −2(v + k + 1) + z,

V12|k>−2 ≤σ4N
eff

4N
X

v=N

4N−v
X

k+2=1

4N
X

z=1
γ4N−2(v+k+2)
2
2k+2(sp ∧sq)v−2N+k+2

×
X

vL,vM,vN
PL(vL, z)PM(vM)PL(vN, k)

γ2
nD+(a,b,s,ε)/D

z
.

We show the upper bound for ˙UL part in Lemma 27. The upper bound for PM and PN trivially
follows from Lemma 22 and Lemma 23 as they do not use any assumption on edges are all at least
2-decorated, except for the number 11, the possible ways of decoration. So, we change 11 to 15 and
then every thing follows. When z ̸= 0, eM as defined before in an arbitrary ˙UM has maximum value
2N −(v + k + 2) −z/2 (same holds for eN, eL but we do not need to use them in our bound). In

64


---Page Break---
summary,

PL(vL, z) ≤nvL|T |β4K(4LM)4L(4L)!(2β)4(K+M) 
(4N)2β
K
K+M
z

PM(vM) ≤R

2eM

M nvM (15β)(K+M) 2eM

M 1{eM≤2N−(v+k+2)+z/2}

≤nvM 
R
2
M (15β)

2(K+M)

M
2N−v−k−2 
R
1
M (15β)
K+M

M
z

PN(vN, k) ≤nvN β(15β)(K+M)2(k+2)(15R4(vN + 2)2)k+2

≤nvN β

(15β)2(K+M)15R4(4N + 1)2k+2
.

The last inequality holds because vN ≤v ≤4N −(k + 2) and k ≥−1.

Putting every pieces together,

V12|k>−2

µ2
≤
X

k≥−1

(15β)2(K+M)30R4(4n + 1)2

n

k+2 4N−k−2
X

v=N

 
γ2
2(15β)2 K+M

M R
2
M
ns(p ∧q)

!2N−v−k−2

×

4N
X

z=1∨(2N−v−k−2)

 
γ2(4N)2R
1
M (15β)
K+M

M β
K
K+M

n

D+(a,b,s,ε)

D

!z

× β4K+1(2β)4(K+M)(4LM)4L(4L)!

|T |ρ2N
.

From the second condition in (25), we first look at the summation of v from N to 2N −k −2,

4N
X

z=1

 
γ2(4N)2R
1
M (15β)
K+M

M β
K
K+M

n

D+(a,b,s,ε)

D

!z

= o(1).

From the first condition in (25),

2N−k−2
X

v=N

 
γ2
2(15β)2 K+M

M R
2
M
ns(p ∧q)

!2N−v−k−2

≤2.

When v > 2N −k −2, the power 2N −v −k −2 < 0. Observe that z ≥2(v + k + 2) −4N =
z −e(K12) −e(K21) −2e(K22), we have the product of two summations upper bounded by

4N−k−2
X

v=2N−k−1

 
ns(p ∧q)

γ2
2(15β)2 K+M

M R
2
M × γ2(4N)2R
1
M (15β)
K+M

M β
K
K+M

n

D+(a,b,s,ε)

D

!v+k+2−2N

.

This is clearly o(1) because N
= Θ(log n) and from the first and second condition (25),
nD+(a,b,s,ε)/D is the only term being logω(1) n.

From the third condition in (25),

X

k≥−1

(15β)2(K+M)30R4(4n + 1)2

n

k+2
β4K+1(2β)4(K+M)(4LM)4L(4L)!

≤2
(15β)2(K+M)30R4(4n + 1)2

n


β4K+1(2β)4(K+M)(4LM)4L(4L)! ≤1.

Therefore, V12/µ2 = o(
1
|T |ρ2N ).

M.2
Proof of auxiliary Lemmas

Lemma 27. For j ̸= π∗(i),

PL(vL, z) ≤nvL|T |β4K(4LM)4L(4L)!(2β)4(K+M) 
(4N)2β
K
K+M
z
.

65


---Page Break---
Proof. We define the unlabeled union graph sets corresponding to U(vL, z) as eU(vL, z). From the
definition (75) and Claim 3,

PL(vL, z) ≤
X

˙UL∈e
UL(vL,z)

nvLw( ˙UL)

aut( ˙UL)
≤(2K)znvL| eUL(vL, z)|.
(86)

Recall that ˙UL consists of two disjoint trees, one rooted at i and the other one rooted at j. We consider
the branches of chandeliers without overlapping with each other.

Then, we specify two categories of branches. A branch is called an invader if it is rooted at i (resp.
j) but in ˙UL(j) (resp. ˙UL(i)). A branch that is not an invader is called a residence. We observe
that there are at most L + ⌊
z
K+M ⌋+ 4 effective non-isomorphic bulbs among residents (defined
in Section L, the proof of Proposition 5) because of the followings: 1) If branches are perfectly
matched and overlapped, there are at most L pairs of them rooted at i and another L pairs rooted at
j. We define effective non-isomorphic bulbs the same as in Lemma 25. Here we have L effective
non-isomorphic bulbs because S1 ∼= S2, T1 ∼= T2. 2) ⌊
z
K+M ⌋is the maximum number of fully
1-decorated branches in allowed ˙UL, and 3) There are at most 4 invading branches, each of which can
at most fully overlapping with one resident bulb, due to the fact that there cannot be two branches on
the same chandelier passing through the same vertex.

The remaining is to bound |eUL(vL, z)|. We observe that there are at most 4(K + M −1) edges
from invading branches, which might be attaching to at most 4(K + M −1) resident branches. This
is because an invader rooted at j may only have its bulb overlapping with ˙UL(i). In this case, one
invader can overlap with multiple resident branches in ˙UL(i).

|eUL(vL, z)| ≤

|J |
L + ⌊
z
K+M ⌋+ 4

2N
z/2


(4N)
z
2 (4LM)4L(2β)4(K+M−1)(4L)!,
(87)

where
 
|J |
L+⌊
z
K+M ⌋+4

is the structures of all resident branches,
 2N
z/2

is the upper bound of choosing

which edges to be not overlapped on bulbs assume starting from perfect overlapped bulbs, (4N)
z
2 is
the bound for placing the remaining z/2 1-decorated vertices, (4LM)4L is a trivial bound on how
resident branches have their wires tangling with each other, (β)4(K+M−1) is the structure of invading
edges, and lastly 2(K+M−1)(4L)! bounds the different interactions between invading edges and the
resident branches. To understand the quantity 2(K+M−1)(4L)!, this comes from the fact that each
invading edge connected to the root can choose one out of at most 4L resident branch to attach, and
that the following invading edges can choose to stay overlapping with the current resident branch of
leave.

Plugging (87) back into (86) with basic binomial bounds and |J | ≤βK, we complete the proof.

Claim 3. Assume that j ̸= π∗(i). For any ˙UL ∈UL(vL, z),

w( ˙UL) ≤aut( ˙UL)(2K)z.
(88)

Proof. Denote all bulbs contained in ˙UL(i) (resp., j) and are attached to wires rooted at i (resp., j)
as B1, B2, . . . , Bw. Denote all bulbs contained in ˙UL(i) (resp., j) and are attached to wires rooted
at j (resp., i) as T1, T2, . . . , Tm. For those branches rooted at i (resp., j) but connect to j (resp., i),
although they can have their bulbs in ˙UL(j) (resp., ˙UL(i)), they contribute to the weight of non-tree
part ˙UN from definitions (66) and (68).

For an arbitrary bulb Bt in ˙UL(i) ∪˙UL(j), we discuss the following three cases. Without loss of
generality, we assume that Bt is attached to a wire rooted at i.

Firstly, if there exists another bulb Br such that Br and Bt be two bulbs with wires rooted both
at i and overlapping with each other. Then, from Corollary 2, we have that
p

aut(Br)aut(Bt) ≤

aut(Br ∪Bt)(2K)
E(Br)△E(Bt)

2
, because each bulb has size K and the difference between two edge
set is at most K. Secondly, if Br is full 1-decorated, then it contributes to
p

aut(Br) to w( ˙UL) and
aut(Bt) to aut( ˙UL). Thirdly, assume that there is another bulb Bt attached to a wire rooted at j partly

66


---Page Break---
overlapping with Bt, then
p

aut(Bt)aut(Tr) ≤aut(Bt ∪Tr)(2K)|E(B4)△E(Tr)| from Corollary 2.
Note that in this case, full overlap is not possible because this invader branch spends at least one edge
connecting from j to i. The third case can be considered as a generalized version of the first case.

By a product over all overlapping bulbs, we have w( ˙UL) ≤aut( ˙UL)(2K)z because the total number
of edges in the difference sets of overlapping bulbs are upper bounded by z, the number of 1-decorated
edges, and the automorphism number of ˙UL is greater than the product of automorphism number of
all bulbs.

67


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: As stated in the abstract and introduction, the main contribution of the paper is
to give the first efficient algorithm for graph matching in the setting of correlated stochastic
block models with two balanced communities. This is exactly the content of Theorem 1
in Section 3. The paper also claims, as an application, an efficient algorithm for exact
community recovery given correlated stochastic block models. This is exactly the content of
Theorem 2 in Section 4. These results, and their underlying assumptions, are discussed in
detail in Sections 2, 3, 4, and 7.
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
Justification: We discuss the limitations of our work, and possible future work that may
address these, in Section 7.
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

68


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: We clearly define the model and questions that we study in Section 2, and
we fully state our main theoretical results, including all assumptions, in Theorem 1 and
Theorem 2. Theorem 2 follows directly from Theorem 1 and prior work, as explained in the
main text. For Theorem 1, we provide a brief overview of the proof in Section 5, and we
give the full proof in the Appendix.
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
Justification: This paper does not include experimental results. Our main contribution is the
theoretical analysis of a novel efficient graph matching algorithm for correlated SBMs. The
algorithm itself is fully described and hence implementable.
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

69


---Page Break---
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

Justification: This paper does not include experiments.

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
results?s

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

70


---Page Break---
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

Justification: This research conforms with the NeurIPS Code of Ethics in every respect.

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

Justification: We discuss both the potential positive and negative societal impacts of this
work, especially graph matching, in Section 2.

71


---Page Break---
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
Justification: The paper does not involve any data or models that have a high risk for misuse.
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
Justification: Relevant prior work on models (all theoretical) is cited and discussed in detail
throughout the paper, following the norms of the research literature. No code or data is used.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

72


---Page Break---
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
Justification: The paper does not release any new assets.
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
Justification: This paper is a theoretical work and does not involve crowdsourcing nor
research with human subjects.
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
Justification: This paper is a theoretical work and does not involve crowdsourcing nor
research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

73


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

74


---Page Break---
