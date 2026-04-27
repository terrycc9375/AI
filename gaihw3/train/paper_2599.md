Harnessing Multiple Correlated Networks
for Exact Community Recovery

Miklós Z. Rácz
Northwestern University
Evanston, IL 60208
miklos.racz@northwestern.edu

Jifan Zhang
Northwestern University
Evanston, IL 60208
jifanzhang2026@u.northwestern.edu

Abstract

We study the problem of learning latent community structure from multiple cor-
related networks, focusing on edge-correlated stochastic block models with two
balanced communities. Recent work of Gaudio, Rácz, and Sridhar (COLT 2022)
determined the precise information-theoretic threshold for exact community recov-
ery using two correlated graphs; in particular, this showcased the subtle interplay
between community recovery and graph matching. Here we study the natural
setting of more than two graphs. The main challenge lies in understanding how to
aggregate information across several graphs when none of the pairwise latent vertex
correspondences can be exactly recovered. Our main result derives the precise
information-theoretic threshold for exact community recovery using any constant
number of correlated graphs, answering a question of Gaudio, Rácz, and Sridhar
(COLT 2022). In particular, for every K ≥3 we uncover and characterize a region
of the parameter space where exact community recovery is possible using K corre-
lated graphs, even though (1) this is information-theoretically impossible using any
K −1 of them and (2) none of the latent matchings can be exactly recovered.

1
Introduction

Finding communities in networks—that is, groups of nodes that are similar—is one of the fundamen-
tal problems in machine learning. This task is crucially important for understanding the underlying
structure and function of networks across diverse applications, including sociology and biology [23].
The increasing availability of network data sets offers the intriguing possibility of improving com-
munity recovery algorithms by synthesizing information across correlated networks. However, in
many settings the graphs are not aligned—which may happen for a variety of reasons, including
anonymization, missing or erroneous data, or simply the alignment being unknown—which presents
a challenge. Thus graph matching—the task of recovering the latent vertex alignment between
graphs—plays a central role in efforts to integrate data across networks. Our work follows an exciting
recent line of work at the intersection of community recovery and graph matching.

Recently, Rácz and Sridhar [41] initiated the study of community recovery in correlated stochastic
block models (SBMs), focusing on the simplest setting of two correlated graphs with two balanced
communities. They determined the information-theoretic limits for exact graph matching, which
has applications for community recovery. In particular, they uncovered a region of the parameter
space where exact community recovery is possible using two correlated graphs even though it is
information-theoretically impossible to do so using just a single graph. Subsequently, Gaudio, Rácz,
and Sridhar [22] determined the information-theoretic limits for exact community recovery from two
correlated SBMs. This required going beyond exact graph matching and understanding the subtle
interplay between community recovery and graph matching.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Gaudio, Rácz, and Sridhar [22] posed the question of understanding what happens in the case of
more than two graphs, which arises naturally in all the motivating examples. For instance, people
participate in numerous overlapping yet complementary social networks, and only by combining
these can we fully understand and make inferences about society. Similarly, synthesizing information
across protein-protein interaction networks from several related species can aid in inferring protein
functions [44]. The main challenge lies in understanding how to optimally pass information across
three or more graphs.

Our main contribution fully answers this open question by Gaudio, Rácz, and Sridhar [22]. Specifi-
cally, we precisely characterize the information-theoretic threshold for exact community recovery
given K correlated SBMs, for any constant K. This result highlights an intricate phase diagram
and quantifies the value of each additional correlated graph for the task of community recovery. In
particular, for every K ≥3 we uncover and characterize a region of the parameter space where exact
community recovery is possible using K correlated graphs, even though (1) this is impossible using
any K −1 of them and (2) none of the latent matchings can be exactly recovered. See Section 3 and
Theorems 1 and 2 for details.

Along the way, we also precisely characterize the information-theoretic threshold for exact graph
matching given K correlated SBMs, for any constant K. In particular, we uncover and characterize a
region of the parameter space where the latent matching between two correlated SBMs cannot be
exactly recovered given just the two graphs, but it can be exactly recovered given K > 3 correlated
SBMs. See Section 3 and Theorems 3 and 4 for details.

To prove our results, we study the so-called k-core matching between all pairs of graphs. Recent
works have shown the k-core matching to be a flexible and successful tool in a variety of settings for
two correlated graphs [13, 22, 43]. Our main technical contribution is to extend this analysis to more
than two graphs. The main difficulty lies in understanding the size of intersections of “bad sets” for
k-core matchings for different pairs of graphs. We refer to Section 4 for details.

2
Models and questions

The stochastic block model (SBM). The SBM is the most common probabilistic generative model for
networks with latent community structure. First introduced by Holland, Laskey, and Leinhardt [25],
it has garnered considerable attention and research. In particular, it can be employed as a natural
testbed for evaluating and assessing clustering algorithms on average-case networks [1]. The SBM
notably displays sharp information-theoretic phase transitions for various inference tasks, offering a
detailed understanding of when community information can be extracted from network data. The
phase transition thresholds were conjectured by Decelle et al. [15] and were proved rigorously in
several papers [33, 34, 35, 36, 2, 3]. We refer to the survey [1] for a detailed overview of the SBM.

In this paper, we focus on the simplest setting, a SBM with two symmetric communities. Let n
be a positive integer and let p, q ∈[0, 1] be parameters representing probabilities. We construct a
graph G ∼SBM(n, p, q) as follows. The graph G has n vertices, labeled by [n] := {1, 2, 3, . . . , n}.
Each vertex i is assigned a community label σ∗(i) from the set {+1, −1}; these are drawn i.i.d.
uniformly at random across i ∈[n]. Let σ∗:= {σ∗(i)}n
i=1 denote the community label vector.
The vertices are thus categorized into two communities: V + := {i ∈[n] : σ∗(i) = +1} and
V −:= {i ∈[n] : σ∗(i) = −1}. Given the community labels σ∗, the edges of G are drawn
independently between pairs of distinct vertices. If σ∗(i) = σ∗(j), then the edge (i, j) is in G with
probability p; otherwise, it is in G with probability q.

Community recovery. In the community recovery task, an algorithm takes as input the graph G
(without knowing σ∗) and outputs an estimated community labeling bσ. Define the overlap between
the estimated labeling and the ground truth as follows:

ov(σ∗, bσ) := 1

n



n
X

i=1
σ∗(i)bσ(i)

 .

The overlap measures how well the true community labels and the estimated labels of the algorithm
match. Note that ov(σ∗, bσ) ∈[0, 1], where the larger the value is, the better performance the
algorithm has. In particular, the algorithm succeeds in exactly recovering the partition into two
communities (i.e., σ∗= bσ or σ∗= −bσ) if and only if ov(σ∗, bσ) = 1. Our focus in this paper is
achieving this goal, known as exact community recovery.

2


---Page Break---
Figure 1: Schematic showing the construction of multiple correlated SBMs (see text for details).

It is well-known that exact community recovery is most challenging and interesting in the logarithmic
average degree regime [1]. Accordingly, we focus on this regime: in most of the paper we assume that
p = a log n

n
and q = b log n

n
for some constants a, b > 0. In this regime there is a sharp information-
theoretic threshold for exact community recovery [2, 35, 3]. Let D+(a, b) := (√a −
√

b)2/2 denote
the so-called Chernoff-Hellinger divergence. Then the information-theoretic threshold is given by
D+(a, b) = 1.
(2.1)
In other words, if D+(a, b) > 1, then exact recovery is possible (and, in fact, efficiently). That
is, there is a (polynomial-time) algorithm which outputs an estimator bσ with the guarantee that
limn→∞P(ov(σ∗, bσ) = 1) = 1. On the other hand, if D+(a, b) < 1 then exact recovery is
impossible: for any estimator eσ, we have that limn→∞P(ov(σ∗, eσ) = 1) = 0.

Correlated SBMs. The objective of our work is to understand how the sharp threshold for exact
community recovery varies when the input data involves multiple correlated graphs. To do so, we
first define a natural model of multiple correlated SBMs [29, 39, 28] (and see further discussion in
Section 6 about alternative models).

We construct (G1, . . . , GK) ∼CSBM(n, p, q, s) as follows, where the additional parameter s ∈
[0, 1] reflects the degree of correlation between the graphs (and the number of graphs K is dropped
from the notation for ease of readability). First, generate a parent graph G0 ∼SBM(n, p, q)
with community labels σ∗. Subsequently, given G0, construct G′
1, G′
2, . . . , G′
K by independent
subsampling. Specifically, each edge of G0 is included in G′
i with probability s, independently
of everything else, and non-edges of G0 remain non-edges in G′
i. The graphs G′
i inherit both the
vertex labels and the community labels from the parent graph G0. Finally, let π∗
12, . . . , π∗
1K be i.i.d.
uniformly random permutations of [n] and let π∗:= (π∗
12, . . . , π∗
1K). Define G1 := G′
1 and, for all
i ∈{2, . . . , K}, define Gi := π∗
1i(G′
i). In other words, for every i > 1 and j ∈[n], vertex j in G′
i
is relabeled to π∗
1i(j) in Gi. This last relabeling step mirrors the real-world observation that vertex
labels are often unaligned across graphs. This construction is shown in Figure 1.

An important property of the model is that marginally each graph Gi is an SBM. Since the subsampling
probability is s, we have that Gi ∼SBM(n, ps, qs). Thus, it follows from (2.1) that, in the
logarithmic average degree regime where p = a log n

n
and q = b log n

n , the communities can be exactly
recovered from G1 alone precisely when sD+(a, b) = D+(sa, sb) > 1.

The key question in our work is how to improve the threshold by incorporating more information
as K, the number of correlated SBMs, increases. This question was initiated by Rácz and Sridhar [41]
and then solved by Gaudio, Rácz, and Sridhar [22] when K = 2.

An essential observation is that, to go beyond the threshold, one needs to combine information from the
K graphs G1, . . . , GK through graph matching. Then one can exactly recover the community labels
using the combined information, even in regimes where it is information-theoretically impossible to
exactly recover σ∗given up to K −1 graphs.

To be more specific, if π∗were known, then one can reconstruct G′
j from Gj and then combine the
graphs G′
1, . . . , G′
K to obtain the union graph H∗, defined as follows: the edge (i, j) is included in

3


---Page Break---
H∗if and only if (i, j) is included in at least one of G′
1, . . . , G′
K. Note that H∗is also an SBM;
specifically, H∗∼SBM
 
n,
 
1 −(1 −s)K
p,
 
1 −(1 −s)K
q

, so (2.1) directly implies that the
communities can be exactly recovered from the union graph H∗if
 
1 −(1 −s)K
D+(a, b) > 1.

Graph matching. In real applications, the permutations π∗= (π∗
12, . . . , π∗
1K) are often not known.
The arguments above highlight the importance of an intermediate task, known as graph matching:
how can one recover the latent permutations π∗given the graphs (G1, G2, . . . , GK)? While here we
regard graph matching as an important intermediate step, it is of great significance in its own right,
with applications in social network privacy [40], machine learning [10], and more. For two correlated
SBMs, this problem was resolved by Rácz and Sridhar [41], who proved that the information-
theoretic threshold for (pairwise) exact graph matching is s2(a + b)/2 = 1. Note that this is also the
connectivity threshold for the intersection graph of G1 and G′
2 (see [41]). Denote

Tc(a, b) := a + b

2
.
(2.2)

With this notation, the (pairwise) exact graph matching threshold is given by s2Tc(a, b) = 1. This
directly implies (by a union bound) that if s2Tc(a, b) > 1, then π∗can be exactly recovered given
(G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s), for any constant K. By the discussion above,
this also gives a sufficient condition for exact community recovery given (G1, G2, . . . , GK) ∼
CSBM(n, a log n

n , b log n

n , s):

s2Tc(a, b) > 1
and
 
1 −(1 −s)K
D+(a, b) > 1.
(2.3)
We will generalize the exact graph matching result of Rácz and Sridhar [41] and show (see Theorem 3
below) that π∗can be exactly recovered given (G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s) if

s
 
1 −(1 −s)K−1
Tc(a, b) > 1,
(2.4)

which (for K > 2) is weaker than the condition s2Tc(a, b) > 1 implied by [41]. (Moreover,
in Theorem 4 we show that the condition in (2.4) is tight for exact recovery of π∗.)
Thus,
by the discussion above, this gives a sufficient condition for exact community recovery given
(G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s):

s
 
1 −(1 −s)K−1
Tc(a, b) > 1
and
 
1 −(1 −s)K
D+(a, b) > 1.
(2.5)
The interplay between community recovery and graph matching. The condition (2.5) is, however,
not tight. To attain the sharp threshold for exact community recovery given K graphs, we need
to answer the following question: does there exist a parameter regime where exact community
recovery is possible for K graphs, even though (1) exact graph matching is impossible, and (2) exact
community recovery is impossible using only K −1 graphs?

For K = 2 graphs, Gaudio, Rácz, and Sridhar [22] proved that the sharp threshold for exact
community recovery given K correlated SBMs is given by
s2Tc(a, b) + s(1 −s)D+(a, b) > 1
and
 
1 −(1 −s)2
D+(a, b) > 1.
(2.6)

The condition
 
1 −(1 −s)2
D+(a, b) > 1 is necessary due to the work [41]. The first condition
in (2.6) demonstrates the interplay between community recovery and graph matching. To be more
specific, the first term s2Tc(a, b) is the threshold for exact graph matching given (G1, G2), while the
second term s(1 −s)D+(a, b) comes from community recovery.

Our main contribution generalizes this result, determining the exact community recovery threshold
for K ≥3 graphs. If
 
1 −(1 −s)K
D+(a, b) > 1, then the sharp threshold is given by

s
 
1 −(1 −s)K−1
Tc(a, b) + s(1 −s)K−1D+(a, b) > 1.
(2.7)
The condition (2.7) also clearly exhibits the interplay between community recovery and graph
matching. The first term comes from graph matching, while the second term comes from community
recovery, as in the case of K = 2. We refer to Section 3 and Theorems 1 and 2 for details.

Despite the apparent similarity in results, when K ≥3 the situation differs significantly from that of
two graphs. The primary challenge lies in the existence of multiple methods for matching K ≥3
graphs. When K = 2, there is only a single matching that needs to be recovered from G1 and G2.
In contrast, with three or more graphs, the graphs can be matched pairwise, or to some anchor
graph, or potentially in many other ways. Integrating information across different matchings requires
substantial additional effort. We present the formal results in the next section.

4


---Page Break---
3
Results

Our main contributions are to determine the precise information-theoretic thresholds for exact
community recovery and for exact graph matching given K correlated SBMs.

3.1
Threshold for exact community recovery

We first describe the precise information-theoretic threshold for exact community recovery, starting
with the positive direction.
Theorem 1 (Exact community recovery from K correlated SBMs). Fix constants a, b > 0 and
s ∈[0, 1], and let (G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s). Suppose that the following two
conditions both hold:
 
1 −(1 −s)K
D+(a, b) > 1
(3.1)

and
s
 
1 −(1 −s)K−1
Tc(a, b) + s(1 −s)K−1D+(a, b) > 1.
(3.2)

Then exact community recovery is possible. That is, there is an estimator bσ = bσ(G1, G2, . . . , GK)
such that lim
n→∞P (ov (bσ, σ∗) = 1) = 1.

Combined with Theorem 2 below (which shows that Theorem 1 is tight), this result precisely answers
an open problem of Gaudio, Rácz, and Sridhar [22]. The condition (3.1) is required for exact
community recovery for K graphs by [41]. We now focus on the condition (3.2). In the prior
work [22], it is proved that the threshold for exact community recovery for two graphs is given
by (2.6). The primary contribution of Theorem 1 is to go beyond this threshold as the number of
graphs K increases. In particular, this showcases that there exists a regime where (1) it is impossible to
exactly recover σ∗from (G1, G2, . . . , GK−1) alone and (2) any exact graph matching is impossible,
yet one can perform exact recovery of σ∗given (G1, G2, . . . , GK). This requires developing novel
algorithms that integrate information from (G1, G2, . . . , GK) delicately and incorporate multiple
graph matchings carefully.

Here we first provide a detailed discussion of the algorithms for three graphs (G1, G2, G3) ∼
CSBM(n, a log n

n , b log n

n , s), which is the simplest case with intriguing new phenomena and challenges
as mentioned. This avoids complicated notations (which arise for general K) for easier understanding.
The new techniques used for combining multiple matchings and integrating information with three
graphs are subsequently generalized to K > 3 correlated SBMs. The high level idea of the algorithm
for exact community recovery when K = 3 consists of five steps (in the following discussion we
assume a > b; when a < b, change majority to minority everywhere):

1. Obtain an almost exact community labeling of G1.

2. Obtain three pairwise partial almost exact graph matchings bµ12, bµ13, and bµ23 between graph
pairs (G1, G2), (G1, G3), and (G2, G3), respectively.

3. For vertices in G1 that are part of at least two matchings, refine the almost exact community
labeling in Step 1 via majority vote in the (union) graph consisting of edges that appear at
least once in (G1, G2, G3).

4. For vertices in G1 that are part of only bµ12 (resp. bµ13), label them via majority vote of their
neighbors’ labels in the graph consisting of edges that appear only in G1 and not in G2 (resp.
only in G1 and not in G3).

5. For vertices in G1 that are not part of any of the three matchings or only part of bµ23, label
them via majority vote of their neighbors’ labels in G1.

Each step in the algorithm involves abundant technical details. See Section 4 for a detailed overview
of the algorithms and proofs. Note that the threshold (3.2) captures the interplay between community
recovery and graph matching, which we now discuss in more detail.

The first term in (3.2), which is s
 
1 −(1 −s)2
Tc(a, b) for K = 3, comes from graph matching.
In [22] it is shown that for one matching, say bµ12, the best possible almost exact graph matching
makes n1−s2Tc(a,b)+o(1) errors. Here, we show that it is possible to obtain almost exact matchings
bµ12 and bµ13 (namely, these will be k-core matchings; see Section 4 for details) such that the size

5


---Page Break---
of the intersection of the two error sets is n1−s(1−(1−s)2)Tc(a,b)+o(1), which is a smaller power of
n. This quantifies how synthesizing information across graph matchings can reduce errors and this
exponent is precisely what shows up in the first term in (3.2). This observation is important and
relevant for Steps 4 and 5 in the algorithm.

On the other hand, the second term in (3.2), which is s(1 −s)2D+(a, b) for K = 3, comes from
community recovery. In fact, this term arises from the majority votes in Step 5, where we use
only edges in G1. Note that the nodes that are unmatched by bµ12 and bµ13 are, roughly speaking,
the isolated nodes in the intersection graphs of G1 and G2, and G1 and G3, respectively. Thus,
while we use all edges in G1 in this step, the relevant edges are not present in G2 nor in G3,
giving the “effective” factor of s(1 −s)2. By (2.1), the exact community recovery threshold for
SBM(n, s(1 −s)2a log n

n , s(1 −s)2b log n

n ) is s(1 −s)2D+(a, b) = 1, giving the second term in (3.2).

The following impossibility result shows the tightness of Theorem 1.
Theorem 2 (Impossibility of exact community recovery). Fix constants a, b > 0 and s ∈[0, 1], and
let (G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s). Suppose that either
 
1 −(1 −s)K
D+(a, b) < 1
(3.3)

or
s
 
1 −(1 −s)K−1
Tc(a, b) + s(1 −s)K−1D+(a, b) < 1.
(3.4)
Then exact community recovery is impossible. That is, for any estimator eσ = eσ(G1, G2, . . . , GK),
we have that lim
n→∞P (ov (eσ, σ∗) = 1) = 0.

Impossibility of exact community recovery given K graphs under the condition (3.3) is proved
in [41]. Hence, Theorem 2 focuses on proving impossibility for exact community recovery given
K graphs under the condition (3.4). In particular, condition (3.4) reveals a parameter regime where
exact community recovery from (G1, G2, . . . , GK) is impossible, yet, if π∗were known, then exact
community recovery would be possible based on the (correctly matched) union graph.

Theorems 1 and 2 combined give the tight threshold for exact community recovery for general K
correlated SBMs, see (2.7). Fig. 2 exhibits phase diagrams illustrating the results for three graphs.

3.2
Threshold for exact graph matching

The techniques that we develop in order to prove Theorems 1 and 2 also allow us to solve the question
of exact graph matching, that is, exactly recovering π∗= (π∗
12, . . . , π∗
1K) from (G1, G2, . . . , GK).
In the context of correlated SBMs and community recovery, exact graph matching can be thought of
as an intermediate step towards exact community recovery. However, more generally, graph matching
is a fundamental inference problem in its own right; see Section 5 for discussion of related work. We
start with the positive direction in the following theorem.
Theorem 3 (Exact graph matching from K correlated SBMs). Fix constants a, b > 0 and s ∈[0, 1],
and let (G1, G2, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s). Suppose that

s
 
1 −(1 −s)K−1
Tc(a, b) > 1.
(3.5)

Then exact graph matching is possible. That is, there exists an estimator bπ = bπ(G1, G2, . . . , GK) =
(bπ12, . . . , bπ1K) such that limn→∞P (bπ(G1, G2, . . . , GK) = π∗) = 1.

Since the condition in (3.5) is weaker than s2Tc(a, b) > 1 (which is the threshold for exact
graph matching for K = 2, as shown in [41]), Theorem 3 implies that there exists a parameter
regime where bπ12 cannot be exactly recovered from (G1, G2), but bπ can be exactly recovered from
(G1, G2, . . . , GK). In other words, it is necessary to combine information across all graphs in order
to recover bπ (and even just to recover bπ12).

The estimator bπ in Theorem 3 is based on pairwise k-core matchings (see Sec. 4 for further de-
tails). Roughly speaking, for each pairwise k-core matching the number of unmatched vertices is
n1−s2Tc(a,b)+o(1); however, we shall show that the number of vertices which cannot be matched
through some combination of pairwise k-core matchings is n1−s(1−(1−s)K−1)Tc(a,b)+o(1), which is
of smaller order. So, when s
 
1 −(1 −s)K−1
Tc(a, b) > 1, then exact graph matching is possible.

The following impossibility result shows the tightness of Theorem 3.

6


---Page Break---
(a) Fixed s = 0.15.
(b) Fixed s = 0.25.

Figure 2: Phase diagram for exact community recovery for three graphs with fixed s, and a ∈[0, 40],
b ∈[0, 40] on the axes. Green region: exact community recovery is possible from G1 alone; Cyan
region: exact community recovery is impossible from G1 alone, but exact graph matching of G1
and G2 is possible, and subsequently exact community recovery is possible from (G1, G2); Dark
Blue region: exact community recovery is impossible from G1 alone, exact graph matching is also
impossible from (G1, G2), yet exact community recovery is possible from (G1, G2); Pink region:
exact community recovery is impossible from (G1, G2) (even though it would be possible if π∗
12 were
known), yet exact community recovery is possible from (G1, G2, G3); Violet region: exact community
recovery is impossible from (G1, G2, G3) (even though it would be possible from (G1, G2) if π∗
12
were known); Light Green region: exact community recovery is impossible from (G1, G2), but
exact graph matching of graph pairs is possible, and subsequently exact community recovery is
possible from (G1, G2, G3); Grey region: exact community recovery is impossible from (G1, G2),
exact graph matching is also impossible from (G1, G2), but exact graph matching is possible from
(G1, G2, G3), and subsequently exact community recovery is possible from (G1, G2, G3); Yellow
region: exact community recovery is impossible from (G1, G2), exact graph matching is impossible
from (G1, G2, G3), yet exact community recovery is possible from (G1, G2, G3); Orange region:
exact community recovery is impossible from (G1, G2, G3) (even though it would be possible
from (G1, G2, G3) if π∗were known); Red region: exact community recovery is impossible from
(G1, G2, G3) (even if π∗is known). The principal finding of this paper is the characterization of the
Pink, Violet, Orange, Yellow, Grey, and Light Green regions.

Theorem 4 (Impossibility of exact graph matching from K correlated SBMs). Fix constants a, b > 0
and s ∈[0, 1], and let (G1, G2, . . . , GK) ∼CSBM(n, a log n

n
, b log n

n
, s). Suppose that

s
 
1 −(1 −s)K−1
Tc(a, b) < 1.
(3.6)

Then exact graph matching is impossible. That is, for any estimator eπ = eπ(G1, G2, . . . , GK) =
(eπ12, . . . eπ1K) we have that limn→∞P (eπ(G1, G2, . . . , GK) = π∗) = 0.

Theorems 3 and 4 combined give the tight threshold for exact graph matching for general K correlated
SBMs, see (2.4). We note that, in independent and concurrent work [5], Ameen and Hajek derived
the threshold for exact graph matching from K correlated Erd˝os–Rényi random graphs; in other
words, they proved Theorems 3 and 4 in the special case of a = b.

Comparing Theorems 3 and 4 with Theorems 1 and 2, note that there exists a parameter regime where
exact community recovery is possible even though exact graph matching is impossible.

4
Overview of algorithms and proofs

In this section we elaborate on the technical details of the community recovery algorithm, for which
high-level ideas were presented in Section 3. We focus our discussion on the setting of K = 3
graphs, which already captures the main technical challenges; we highlight these and explain how

7


---Page Break---
we overcome them. We subsequently explain the generalization from 3 graphs to K graphs. The
overview of the impossibility proof is discussed as well.

k-core matching. We now define a k-core matching [13, 22, 43], which is used for almost exact
graph matching in Step 2. Given a pair of graphs (G, H) with vertex set [n], for any permutation π,
we have the corresponding intersection graph G ∧π H, where (i, j) is an edge in the intersection
graph if and only if (i, j) is an edge in G and (π(i), π(j)) is an edge in H. The k-core estimator
explores all possible permutations π of [n] to seek a permutation bπ that maximizes the size of the
k-core of the intersection graph G ∧π H; recall that the k-core of a graph is the maximal induced
subgraph for which all vertices have degree at least k. The output of the k-core estimator is then a
partial matching bµ, which is the restriction of bπ to the vertex set of the k-core in G ∧bπ H.

One significant advantage of using k-core matchings is a certain optimality property in terms of
performance. Specifically, if (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s), then the k-core estimator
between G1 and G2 fails to match at most n1−s2Tc(a,b)+o(1) vertices, which is the same order as the
number of singletons of G1 ∧π∗
12 G2 and any graph matching algorithm would fail to match these
singletons [11, 41, 22]. Another significant benefit of utilizing k-core matchings is the correctness of
the k-core estimator for correlated SBMs, as discussed in [22]. The k-core estimator might not be
able to match all vertices under the parameter regime that we are interested in; however, every vertex
that it does match is matched correctly with high probability.

Community recovery subroutines. The high-level summary of the algorithm is as follows. Since
exact community recovery might be impossible in G1 alone, we first obtain an initial estimate which
gives almost exact community recovery in G1, as described in Step 1. In Step 2, we use pairwise
partial k-core matchings with k = 13 to obtain bµ := {bµ12, bµ13, bµ23} (see Fig. 3b), which we will
use to combine information across (G1, G2, G3) to recover communities. Note that each partial
matching bµij only matches a subset of the vertices, denoted as Mij; we denote the set of vertices
not matched by bµij by Fij := [n] \ Mij. Subsequently, we split the vertices into two categories:
“good” vertices and “bad” vertices, where “good” vertices are part of at least two matchings and “bad”
vertices are part of at most one matching (see Fig. 3a). We conduct several majority votes among
“good” and “bad” vertices to do the clean-up after the graph matching phase, where each subroutine
is meticulously executed to disentangle the intricate dependencies among (G1, G2, G3) and bµ.

Exact community recovery for the “good” vertices. The major distinction between being “good”
and being “bad” is that “good” vertices can combine information from all three graphs via their union
graph (which is denser), whereas “bad” vertices cannot. Suppose that vertex i is part of bµ12 and bµ13
(i.e., i ∈M12 ∩M13). We can then identify the union graph G1 ∨bµ12 G2 ∨bµ13 G3, which consists of
edges (i, j) such that (i, j) is an edge in G1 or (bµ12(i), bµ12(j)) is an edge in G2 or (bµ13(i), bµ13(j))
is an edge in G3, and i is part of this union graph. Similarly, if i ∈M12 ∩M23, then i is part
of the union graph G1 ∨bµ12 G2 ∨bµ23◦bµ12 G3 that also integrates information from all three graphs.
On the union graph, we can refine the almost exact community labeling by reclassifying “good”
vertices based on a majority vote among the labels of their neighbors that are also “good”, and this
reclassification will be correct as long as the condition (3.1) holds.

There are many underlying technical challenges and roadblocks in the theoretical analysis. The
key difficulty arises from the structure of the union graph. It is statistically guaranteed that in
G1∨π∗
12 G2∨π∗
13 G3, all vertices have a community label which is the same as the majority community
among their neighbors [35]. However, whether this is also the case for G1 ∨bµ12 G2 ∨bµ13 G3 is
unclear, since the latter graph is only defined on the “good” vertices M12 ∩M13. One would like to
demonstrate that the removal of “bad” vertices does not significantly affect the majority community
among neighbors of “good” vertices. Prior work [22] addressed a similar problem for two graphs by
employing a technique known as Łuczak expansion [27] to F12 to ensure that the vertices inside the
expanded set F12 are only weakly connected to the vertices outside of the expanded set [n] \ F12.
Unfortunately, this method is no longer applicable for correlated SBMs with three or more graphs.
Even though the size of the expanded set F12 is orderwise equal to the size of F12, the size of the
intersection of the expanded sets F12 ∩F13 might not be orderwise equal to the size of F12 ∩F13,
which directly leads to the failure of the algorithm working down to the information-theoretic
threshold. To overcome this challenge, we consider the graph G{[n] \ v} to decouple the dependence
of v being connected to a vertex w and w being part of the k-core. Applying the Łuczak expansion
on such a graph for any given v, and through a union bound, we prove that unmatched vertices are
contained in the set of vertices whose degree is smaller than a constant, with high probability. This

8


---Page Break---
allows us to quantify the size of F12 ∩F13 and meanwhile directly ensure that “good” vertices within
the k-core are only weakly connected with “bad” vertices.

Another hurdle needed to overcome, as stated in [22], concerns the almost exact community recovery
in Step 1 which is subsequently used for majority votes. Therefore, it is of great importance to
guarantee that the incorrectly-classified vertices are not well-connected and do not have a great
impact on majority votes. Consequently, we utilize an algorithm originally developed by Mossel,
Neeman, and Sly [35] which allows us to manage the geometry of the misclassified vertices and
demonstrate that the vertices classified incorrectly are indeed only weakly connected.

Exact community recovery for the “bad” vertices. The remaining step is to label the “bad” vertices.
The “bad” vertices can be further classified into three categories (see Fig. 3a): vertices in F12 ∩F13,
which are only matched by bµ23 or are not matched by any of the three matchings; vertices in
F13 ∩F23 \ F12, which are only matched by bµ12; and vertices in F12 ∩F23 \ F13, which are only
matched by bµ13.

Consider the vertices in F12 ∩F13 (the other cases are similar). First of all, as discussed above, we
show that |F12∩F13| = n1−s(1−(1−s)2)Tc(a,b)+o(1) with high probability. Consider the graph G1\bπ12
G2 \bπ13 G3, which consists of the edges (i, j) in G1 such that (bπ12(i), bπ12(j)) and (bπ13(i), bπ13(j))
are not edges in G2 and G3, respectively. Due to the approximate independence of F12 ∩F13 and
G1 \bπ12 G2 \bπ13 G3, for a vertex i ∈F12 ∩F13 we can calculate the probability of the failure of
the majority vote in the graph G1 \bπ12 G2 \bπ13 G3 in a relatively straightforward manner, giving
n−s(1−s)2D+(a,b)+o(1). The factor s(1 −s)2 arises from the fact that the edges in this graph are
subsampled in G1 and are not subsampled in G2 and G3. Now since a vertex in F12 ∩F13 can have at
most 12 edges outside of F12 in G1 ∧bµ12 G2, and also at most 12 edges outside of F13 in G1 ∧bµ13 G3,
the majority vote for i ∈F12 ∩F13 essentially does not change whether it is performed in G1 or in
G1 \bπ12 G2 \bπ13 G3. Putting all this together, the probability that the majority vote fails is at most:

P(exists a vertex i ∈F12 ∩F13 such that the majority vote fails)

= |F12∩F13|×P(majority vote fails for a vertex) = n1−s(1−(1−s)2)Tc(a,b)−s(1−s)2D+(a,b)+o(1).

Thus, if (3.2) with K = 3 holds, then majority vote will correctly classify all vertices in F12 ∩F13.

Generalization to K graphs. For K graphs, we have
 K
2

pairwise matchings to consider (see
Fig. 3b). We again categorize the vertices as “good” and “bad”. The “good” vertices can integrate in-
formation across all K graphs through the pairwise partial k-core matchings {bµij : i, j ∈[K], i ̸= j},
while “bad” vertices cannot. To illustrate this concept more vividly, for any vertex v, consider a new
“metagraph” MGv on K nodes, defined as follows: there is an edge between i and j in MGv if and
only if v can be matched through bµij (see Fig. 4). If the metagraph MGv is connected, then there
exists a path that can connect all of its K nodes. Equivalently, there exists a set of matchings that
allows us to combine information across all K graphs. Subsequently, we quantify the number of “bad”
vertices to be n1−s(1−(1−s)K−1)Tc(a,b)+o(1). The remaining analysis for K graphs can be derived by
generalizing the analysis for three graphs.

Impossibility proof. As discussed in Section 3, we focus on the proof of (3.4) for impossibility.
We compute the maximum a posterior (MAP) estimator for the communities in G1. We show that,
even with significant additional information provided, including all the correct community labels
in G2, the true matchings π∗
ij for i, j ∈{2, 3, . . . , K}, and most of the true matching π∗
12 except
for singletons in the graph G1 ∧π∗
12 (G2 ∨. . . ∨GK), the MAP estimator fails to exactly recovery
communities with probability bounded away from 0 if (3.4) holds. The proof is adapted from the
MAP analysis in [22]. The difference is that here we are considering K graphs G1, G2, . . . , GK with
different additional information provided for the MAP estimator. Given that the MAP estimator is
ineffective under this regime, all other estimators also fail.

Exact graph matching threshold. The proof of the exact graph matching threshold is implicitly
present in the proof of the exact community recovery threshold. Essentially, since we show that
the number of “bad” vertices is n1−s(1−(1−s)K−1)Tc(a,b)+o(1), the condition (3.5) implies that there
are no “bad” vertices with high probability. Since all vertices are “good”, and “good” vertices can
integrate information across all K graphs, the latent matchings can be recovered exactly. For the
impossibility result we analyze the MAP estimator and show that, even with significant additional
information, including the true matchings {π∗
ij : i, j ∈{2, 3, . . . , K}}, it fails if (3.6) holds.

9


---Page Break---
5
Related work

Our work generalizes—and solves an open question raised by—the work of Gaudio, Rácz, and
Sridhar [22]. Just as [22], our work lies at the interface of the literatures on community recovery and
graph matching1—two fundamental learning problems—which we briefly summarize here.

Community recovery in SBMs. A huge research literature exists on learning latent community
structures in networks, and this topic is especially well understood for the SBM [25, 15, 34, 35, 33,
36, 2, 3, 7, 1]. Specifically, we highlight the work of [2, 35], which identify the precise threshold for
exact community recovery for SBMs with two balanced communities. Our algorithm builds upon
their analysis, taking particular care about dealing with the dependencies arising from the multiple
inexact partial matchings between K correlated graphs.

Graph matching: correlated Erd˝os-Rényi random graphs. The past decade has seen a plethora of
research on average-case graph matching, focusing on correlated Erd˝os-Rényi random graphs [40].
The information-theoretic thresholds for recovering the latent vertex correspondence π∗have been
established for exact recovery [11, 46, 12], almost exact recovery [13], and weak recovery [20, 21,
24, 46, 16]. In parallel, a line of work has focused on algorithmic advances [37, 6, 18, 19, 30, 31, 32],
culminating in recent breakthroughs that developed efficient graph matching algorithms in the
constant noise setting [31, 32]. We particularly highlight the work of Cullina, Kiyavash, Mittal,
and Poor [13], who introduced k-core matchings and showed their utility for partial matching of
correlated Erd˝os–Rényi random graphs. Subsequent work has shown the power of k-core matchings
as a flexible and successful tool for graph matching [22, 43, 4]. Our work both significantly builds
upon these works, as well as further develops this machinery, which may be of independent interest.
We also note the independent and concurrent work of Ameen and Hajek which determined the exact
graph matching threshold for K correlated Erd˝os–Rényi random graphs [5].

Graph matching: beyond correlated Erd˝os-Rényi random graphs. Motivated by real-world
networks, a growing line of recent work studies graph matching beyond Erd˝os-Rényi graphs [8, 26,
9, 39, 42, 49, 41, 22, 45, 43, 17, 48, 47], including for correlated SBMs [29, 39, 28, 41, 22, 48, 47].
The works that are most relevant to ours are [41, 22], which have been discussed extensively above.

6
Discussion and Future Work

Our main contribution highlights the power of integrative data analysis for community recovery, yet
many open questions still remain.

Efficient algorithms. Theorem 1 characterizes when exact community recovery is information-
theoretically possible from K correlated SBMs. Is this possible efficiently (i.e., in time polynomial
in n)? The bottleneck in the algorithm that we use to prove Theorem 1, which makes it inefficient, is
the k-core matching step; the other steps are efficient. Recent breakthrough results have developed
efficient graph matching algorithms for correlated Erd˝os–Rényi random graphs [31, 32], which
promisingly suggest that an efficient algorithm for exact community recovery may indeed exist in
this regime. We refer to [22] for further discussion on this point.

General block models. We focused here on the simplest case of SBMs with two balanced communi-
ties. It would be interesting to extend these results to general block models with multiple communities.
This is understood well in the single graph setting [1] and recent work has also characterized the
threshold for exact graph matching for two correlated SBMs with k symmetric communities [47].

Alternative constructions of correlated graphs. An exciting research direction is to study different
constructions of correlated graph models. For general K, there are many ways that K graphs can
be correlated. In particular, the following is a natural alternative construction of multiple correlated
SBMs. First, generate G0 ∼SBM(n, p, q). Then, independently generate Hi ∼SBM(n, p′, q′)
for i ∈[K]. Construct G′
i := G0 ∨Hi, and finally generate Gi through an independent random
permutation of the vertex indices in G′
i. This construction is equivalent to the one we studied in this
paper for K = 2 and it is different when K ≥3, and investigating it is interesting and valuable.

1We note that graph matching has both positive and negative societal impacts. In particular, it is well
known that graph matching algorithms can be used to de-anonymize social networks, showing that anonymity is
different, in general, from privacy [38]. At the same time, studying fundamental limits can aid in determining
precise conditions when anonymity can indeed guarantee privacy, and when additional safeguards are necessary.

10


---Page Break---
Acknowledgements

We thank Taha Ameen, Julia Gaudio, Elchanan Mossel, and Anirudh Sridhar for helpful discussions.
We also thank anonymous reviewers for constructive feedback.

References

[1] E. Abbe. Community Detection and Stochastic Block Models: Recent Developments. Journal
of Machine Learning Research, 18(177):1–86, 2018.

[2] E. Abbe, A. S. Bandeira, and G. Hall. Exact Recovery in the Stochastic Block Model. IEEE
Transactions on Information Theory, 62(1):471–487, 2016.

[3] E. Abbe and C. Sandon. Community detection in general stochastic block models: Funda-
mental limits and efficient algorithms for recovery. In 2015 IEEE 56th Annual Symposium on
Foundations of Computer Science (FOCS), pages 670–688. IEEE, 2015.

[4] T. Ameen and B. Hajek. Robust Graph Matching when Nodes are Corrupt. Preprint available at

https://arxiv.org/abs/2310.18543, 2023.

[5] T. Ameen and B. Hajek. Exact Random Graph Matching with Multiple Graphs. Preprint
available at https://arxiv.org/abs/2405.12293, 2024.

[6] B. Barak, C.-N. Chou, Z. Lei, T. Schramm, and Y. Sheng. (Nearly) Efficient Algorithms for the
Graph Matching Problem on Correlated Random Graphs. In Advances in Neural Information
Processing Systems (NeurIPS), volume 32, pages 9190–9198, 2019.

[7] C. Bordenave, M. Lelarge, and L. Massoulié. Non-backtracking Spectrum of Random Graphs:
Community Detection and Non-regular Ramanujan Graphs. In Proceedings of the 2015 IEEE
56th Annual Symposium on Foundations of Computer Science (FOCS), pages 1347–1357. IEEE,
2015.

[8] K. Bringmann, T. Friedrich, and A. Krohmer. De-anonymization of Heterogeneous Random
Graphs in Quasilinear Time. In Proceedings of the 22nd Annual European Symposium on
Algorithms (ESA), pages 197–208, 2014.

[9] C.-F. Chiasserini, M. Garetto, and E. Leonardi. Social Network De-Anonymization Under
Scale-Free User Relations. IEEE/ACM Transactions on Networking, 24(6):3756–3769, 2016.

[10] T. Cour, P. Srinivasan, and J. Shi. Balanced Graph Matching. In Advances in Neural Information
Processing Systems (NeurIPS), volume 19, 2006.

[11] D. Cullina and N. Kiyavash. Improved Achievability and Converse Bounds for Erd˝os-Rényi
Graph Matching. In ACM SIGMETRICS, volume 44, pages 63–72, 2016.

[12] D. Cullina and N. Kiyavash. Exact alignment recovery for correlated Erd˝os-Rényi graphs.
Preprint available at https://arxiv.org/abs/1711.06783, 2018.

[13] D. Cullina, N. Kiyavash, P. Mittal, and H. V. Poor. Partial Recovery of Erd˝os-Rényi Graph
Alignment via k-Core Alignment. In ACM SIGMETRICS Performance Evaluation Review,
volume 48, pages 99–100. ACM, 2020.

[14] D. Cullina, K. Singhal, N. Kiyavash, and P. Mittal. On the Simultaneous Preservation of
Privacy and Community Structure in Anonymized Networks. Preprint available at https:
//arxiv.org/abs/1603.08028, 2016.

[15] A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová. Asymptotic analysis of the stochas-
tic block model for modular networks and its algorithmic applications. Physical Review E,
84(6):066106, 2011.

[16] J. Ding and H. Du. Matching recovery threshold for correlated random graphs. The Annals of
Statistics, 51(4):1718–1743, 2023.

11


---Page Break---
[17] J. Ding, Y. Fei, and Y. Wang. Efficiently matching random inhomogeneous graphs via degree
profiles. Preprint available at https://arxiv.org/abs/2310.10441, 2023.

[18] J. Ding, Z. Ma, Y. Wu, and J. Xu. Efficient random graph matching via degree profiles.
Probability Theory and Related Fields, 179(1):29–115, 2021.

[19] Z. Fan, C. Mao, Y. Wu, and J. Xu. Spectral Graph Matching and Regularized Quadratic
Relaxations: Algorithm and Theory. In Proceedings of the 37th International Conference on
Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research (PMLR),
pages 2985–2995, 2020.

[20] L. Ganassali and L. Massoulié. From tree matching to sparse graph alignment. In Proceedings
of the 33rd Conference on Learning Theory (COLT), volume 125 of Proceedings of Machine
Learning Research (PMLR), pages 1633–1665, 2020.

[21] L. Ganassali, L. Massoulié, and M. Lelarge. Impossibility of Partial Recovery in the Graph
Alignment Problem. In Proceedings of the 34th Conference on Learning Theory (COLT),
volume 134 of Proceedings of Machine Learning Research (PMLR), pages 2080–2102, 2021.

[22] J. Gaudio, M. Z. Rácz, and A. Sridhar. Exact Community Recovery in Correlated Stochastic
Block Models. In Proceedings of the 35th Conference on Learning Theory (COLT), volume
178 of Proceedings of Machine Learning Research (PMLR), pages 2183–2241, 2022.

[23] M. Girvan and M. E. J. Newman. Community structure in social and biological networks.
Proceedings of the National Academy of Sciences, 99(12):7821–7826, 2002.

[24] G. Hall and L. Massoulié. Partial recovery in the graph alignment problem. Operations
Research, 71(1):259–272, 2023.

[25] P. W. Holland, K. B. Laskey, and S. Leinhardt. Stochastic blockmodels: First steps. Social
Networks, 5(2):109–137, 1983.

[26] N. Korula and S. Lattanzi. An efficient reconciliation algorithm for social networks. Proceedings
of the VLDB Endowment, 7(5):377–388, 2014.

[27] T. Łuczak. Size and connectivity of the k-core of a random graph. Discrete Mathematics,
91(1):61–68, 1991.

[28] V. Lyzinski. Information Recovery in Shuffled Graphs via Graph Matching. IEEE Transactions
on Information Theory, 64(5):3254–3273, 2018.

[29] V. Lyzinski, D. L. Sussman, D. E. Fishkind, H. Pao, L. Chen, J. T. Vogelstein, Y. Park, and
C. E. Priebe. Spectral clustering for divide-and-conquer graph matching. Parallel Computing,
47:70–87, 2015.

[30] C. Mao, M. Rudelson, and K. Tikhomirov. Random Graph Matching with Improved Noise
Robustness. In Proceedings of the 34th Conference on Learning Theory (COLT), volume 134
of Proceedings of Machine Learning Research (PMLR), pages 3296–3329, 2021.

[31] C. Mao, M. Rudelson, and K. Tikhomirov. Exact matching of random graphs with constant
correlation. Probability Theory and Related Fields, 186:327–389, 2023.

[32] C. Mao, Y. Wu, J. Xu, and S. H. Yu. Random Graph Matching at Otter’s Threshold via Counting
Chandeliers. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing
(STOC), pages 1345–1356, 2023.

[33] L. Massoulié. Community detection thresholds and the weak Ramanujan property. In Proceed-
ings of the 46th Annual ACM Symposium on Theory of Computing (STOC), pages 694–703.
ACM, 2014.

[34] E. Mossel, J. Neeman, and A. Sly. Reconstruction and estimation in the planted partition model.
Probability Theory and Related Fields, 162:431–461, 2015.

[35] E. Mossel, J. Neeman, and A. Sly. Consistency thresholds for the planted bisection model.
Electronic Journal of Probability, 21(none):1 – 24, 2016.

12


---Page Break---
[36] E. Mossel, J. Neeman, and A. Sly. A proof of the block model threshold conjecture. Combina-
torica, 38(3):665–708, 2018.

[37] E. Mossel and J. Xu. Seeded graph matching via large neighborhood statistics. Random
Structures & Algorithms, 57(3):570–611, 2020.

[38] A. Narayanan and V. Shmatikov. De-anonymizing Social Networks. In Proceedings of the 30th
IEEE Symposium on Security and Privacy, pages 173–187. IEEE Computer Society, 2009.

[39] E. Onaran, S. Garg, and E. Erkip. Optimal de-anonymization in random graphs with community
structure. In 2016 50th Asilomar Conference on Signals, Systems and Computers, pages
709–713. IEEE, 2016.

[40] P. Pedarsani and M. Grossglauser. On the privacy of anonymized networks. In Proceedings of
the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
(KDD), pages 1235–1243, 2011.

[41] M. Z. Rácz and A. Sridhar. Correlated Stochastic Block Models: Exact Graph Matching with
Applications to Recovering Communities. In Advances in Neural Information Processing
Systems (NeurIPS), volume 34, pages 22259–22273, 2021.

[42] M. Z. Rácz and A. Sridhar. Correlated randomly growing graphs. The Annals of Applied
Probability, 32(2):1058–1111, 2022.

[43] M. Z. Rácz and A. Sridhar. Matching Correlated Inhomogeneous Random Graphs using the
k-core Estimator. In 2023 IEEE International Symposium on Information Theory (ISIT), pages
2499–2504, 2023.

[44] R. Singh, J. Xu, and B. Berger. Global alignment of multiple protein interaction networks with
application to functional orthology detection. Proceedings of the National Academy of Sciences,
105(35):12763–12768, 2008.

[45] H. Wang, Y. Wu, J. Xu, and I. Yolou. Random Graph Matching in Geometric Models: the
Case of Complete Graphs. In Proceedings of the 35th Conference on Learning Theory (COLT),
volume 178 of Proceedings of Machine Learning Research (PMLR), pages 3441–3488, 2022.

[46] Y. Wu, J. Xu, and S. H. Yu. Settling the Sharp Reconstruction Thresholds of Random Graph
Matching. IEEE Transactions on Information Theory, 68(8):5391–5417, 2022.

[47] J. Yang and H. W. Chung. Graph Matching in Correlated Stochastic Block Models for Im-
proved Graph Clustering. In Proceedings of the 2023 59th Annual Allerton Conference on
Communication, Control, and Computing (Allerton), pages 1–8. IEEE, 2023.

[48] J. Yang, D. Shin, and H. W. Chung. Efficient Algorithms for Exact Graph Matching on
Correlated Stochastic Block Models with Constant Correlation. In Proceedings of the 40th
International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine
Learning Research (PMLR), pages 39416–39452, 2023.

[49] L. Yu, J. Xu, and X. Lin. The Power of D-hops in Matching Power-Law Graphs. Proceedings
of the ACM on Measurement and Analysis of Computing Systems, 5(2):1–43, 2021.

13


---Page Break---
A
Organization

The rest of the paper is structured as follows. First, we elaborate on the recovery algorithm for three
graphs in Section C. Section D includes some useful preliminary propositions, including some nice
properties of almost exact community recovery on G1. Section E discusses the k-core estimator.
After these preparations, we are ready to prove the main theorems in the paper.

Section F proves Theorem 1 for three graphs, where we first validate the accuracy of the community
labels for “good” vertices and then classify the remaining “bad” vertices. Section G presents the
proof of the impossibility result (Theorem 2) for three graphs. Section H discusses the recovery
algorithm for K graphs and provides a general proof for K graphs, with additional arguments on
how to identify “good” and “bad” vertices. Section I discusses the proof of the impossibility result
(Theorem 2) for K graphs. Section J contains the proof of the threshold for exact graph matching
given K graphs, that is, the proofs of Theorems 3 and 4.

B
Notation

We introduce here some notation that will be used in the rest of the paper. In most of the paper we
focus on the setting of K = 3 graphs: (G1, G2, G3) ∼CSBM(n, a log n

n , b log n

n , s), and this is the
setting that we consider here as well.

Let V := [n] = {1, 2, 3, ..., n} denote the vertex set of the parent graph G0, and let V + := {i ∈
[n] : σ∗(i) = +1} and V −:= {i ∈[n] : σ∗(i) = −1} denote the sets of vertices in the two
communities. Let
 [n]
2

:= {(i, j) : i, j ∈[n], i ̸= j} denote the set of all unordered vertex
pairs. Given a community labeling σ ∈{+1, −1}n, we define the set of intra-community vertex
pairs as E+(σ) := {(i, j) ∈
 [n]
2

: σ(i) = σ(j)} and the set of inter-community vertex pairs as
E+(σ) := {(i, j) ∈
 [n]
2

: σ(i) = −σ(j)}. Note that E+(σ) and E−(σ) form a partition of
 [n]
2

.

Let A, B, and C denote the adjacency matrices of G1, G2, and G3, respectively. Let B′ and C′
denote the adjacency matrices of G′
2 and G′
3, respectively. Note that, by construction, we have for all
(i, j) ∈
 [n]
2

that B′
i,j = Bπ∗
12(i),π∗
12(j) and C′
i,j = Cπ∗
13(i),π∗
13(j). Observe that we have the following
probabilities for every (i, j) ∈
 [n]
2

. If a, b, c ∈{0, 1}3 and a + b + c > 0, then

P
  
Aij, B′
ij, C′
ij

= (a, b, c)

=
sa+b+c(1 −s)3−a−b−cp
if σ∗(i) = σ∗(j),
sa+b+c(1 −s)3−a−b−cq
if σ∗(i) ̸= σ∗(j).

Furthermore, we have that

P
  
Aij, B′
ij, C′
ij

= (0, 0, 0)

=
1 −p + (1 −s)3p
if σ∗(i) = σ∗(j),
1 −q + (1 −s)3q
if σ∗(i) ̸= σ∗(j).

C
The recovery algorithm for three graphs

Our recovery algorithm is based on discovering a matching between subsets of two graphs.
Definition C.1. Let Gi and Gj be two graphs in vertex set [n] with adjacency matrix A, B, respec-
tively. The pair (Mij, µij) is a matching between Gi and Gj if

• Mij ∈[n],

• µij : Mij →[n],

• µij is injective.

Given a matching (Mij, µij), here are some related notations. Define Gi ∨µij Gj to be the union
graph, whose vertex set is M, whose vertex index is the same as the vertex index of Gi and whose
edge set is {{ℓ, m} : ℓ, m ∈Mij, Aℓm + Bµij(ℓ),µij(m) ≥1}. In other words, the edges are
those that appear in either Gi or Gj. Conversely, Gi ∧µij Gj represents intersection graph, whose
vertex set is Mij, whose vertex index is the same as the vertex index of Gi and whose edge set is
{{ℓ, m} : ℓ, m ∈Mij, Aℓm = Bµij(ℓ),µij(m) = 1}. In other words, the edges are those that appear

14


---Page Break---
(a) Categorization of vertices for three graphs: ver-
tices in the red regions are “bad” while vertices in
the white regions are “good”.

(b) Graph matchings for three graphs. For gen-
eral K, consider
 K
2

partial graph matchings.

Figure 3: Schematic landscape of partial matchings over three graphs.

in both Gi and Gj. Define Gi \µij Gj to be the graph Gi minus Gj, whose vertex set is Mij, whose
vertex index is the same as the vertex index of Gi and the edges are those only appear on Gi and not
appear in Gj.
Definition C.2. Let Gi and Gj, Gk be three graphs on vertex set [n] with adjacency matrix A, B, C,
respectively. The pair (Mij, µij) is a matching between Gi and Gj, while the pair (Mjk, µjk) is a
matching between Gj and Gk. Denote µjk ◦µij as the composition matching between Gi and Gk,
defined on the vertex set Mij ∩Mjk.

For three graphs, we can define the additional notations in the same manner as in Definition C.1 and
the core concepts remain consistent. Gi∨µij Gj∨µjk◦µij Gk represents the union graph of Gi, Gj, Gk,
whose vertex set is M = Mij ∩Mjk, whose vertex index is the same as the vertex index of Gi
and whose edge set is {{ℓ, m} : ℓ, m ∈M, Aℓm + Bµij(ℓ),µij(m) + Cµjk◦µij(ℓ),µjk◦µij(m) ≥1}. In
other words, the edge set are the edges that appears in at least one graph out of Gi, Gj, Gk. Similarly,
Gi ∧µij Gj ∧µjk Gk represents the intersection graph, the edge set is {{ℓ, m} : ℓ, m ∈M; Aℓm =
Bµij(ℓ),µij(m) = Cµjk◦µij(ℓ),µjk◦µij(m) = 1}. Define Gi ∨µij Gj \µjk Gk be the graph whose edge
set is those edges that appear in either Gi or Gj and not appear in Gk. Similarly, we can define
Gi ∧µij Gi \µjk Gk, Gi \µij (Gi ∨µjk Gk), and Gi \µij (Gi ∧µjk Gk) as well. Note that all the
definitions above are defined on vertex set M and use vertex index in Gi.

Introduce dmin(G) := mini∈[n] d(i), where d(i) is the degree of vertex i.
Definition C.3. A matching (Mij, µij) is a k-core matching of (Gi, Gj) if dmin(Gi ∧µij Gj) ≥k.
A matching (Mij, µij) is called a maximal k-core matching if it involves the greatest number of
vertices, among all k-core matchings.

Algorithm 1 k-core matching

Input: Pair of graphs Gi, Gj on n vertices, k ∈[n].
Output: A matching (c
Mij, bµij) of Gi and Gj.

1: Enumerating all possible matchings, find the maximal k-core matching (c
Mij, bµij) of Gi and Gj.

Let (Mij, µij) be the matching found by Algorithm 1 with k = 13. Mij coincides with the maximal
k-core of Gi ∧π∗
ij Gj, denote it as M ∗
ij while µij coincides with the true permutation π∗
ij, with high
probability (Lemma E.5).

The k-core matching is symmetric, i.e. µij(Mij) = Mji. Note that by Definition C.1, Mij uses the
vertex index of Gi while Mji uses the vertex index of Gj, they are equivalent and exchangeable
through the 1-1 mapping. Now define Fij := [n] \ Mij be the set of vertices which are excluded
from the matching. Note that Fij use the vertice index same as Gi. We define F ∗
ij := [n] \ M ∗
ij be
the set of vertices which are outside the maximal k-core of Gi ∧π∗
ij Gj.

As briefly discussed in Section 4, we start with leveraging the “good” vertices in order to find the
correct communities. The “good” vertices are those which are part of at least two matchings out of
three partial matchings µ12, µ13, µ23. The details are shown in Algorithm 2.

15


---Page Break---
Algorithm 2 Labeling the good vertices

Input: Three graphs G1, G2, G3 on n vertices and three 13-core matchings
(M12, µ12, M13, µ13, M23, µ23), parameters a, b, s, ϵ.
Output: A labeling of (M13 ∩M32) ∪(M12 ∩M13) ∪(M23 ∩M12) given by bσ.

1: Apply [35, Algorithm 1] to the graph G1 and parameters (sa, sb, ϵ), obtaining a label c
σ1.
2: Denote F12 = [n] \ M12, F13 = [n] \ M13, F23 = [n] \ M23.
3: For i ∈M13 ∩M32, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp., minority)
of c
σ1(i) with respect to the graph (G1 ∨µ32◦µ13 G2 ∨µ13 G3){M13 ∩M32} if a > b (resp.,
a < b).
4: For i ∈M12 ∩M23, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp., minority)
of c
σ1(i) with respect to the graph (G1 ∨µ12 G2 ∨µ23◦µ12 G3){M12 ∩M23} if a > b (resp.,
a < b).
5: For i ∈M13 ∩M12, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp., minority)
of c
σ1(i) with respect to the graph (G1 ∨µ12 G2 ∨µ13 G3){M13 ∩M12} if a > b (resp., a < b).
6: Return bσ : (M13 ∩M32) ∪(M12 ∩M13) ∪(M23 ∩M12) →{−1, 1}.

The remaining step is to label the “bad” vertices which cannot utilize the combined information
from three graphs. Hence, we classify the “bad” vertices according to the majority of neighborhood
restricted to the corresponding “good” vertices. The detailed descriptions are shown Algorithm 3.

Algorithm 3 Labeling the bad vertices

Input: Three graphs (G1, G2, G3) on n vertices and three 13-core matching (M12, µ12),
(M13, µ13), and (M23, µ23), parameters a, b, s, a label on the “good” vertices bσ.
Output: A labeling of [n] given by bσ.

1: For i ∈F12 ∩F13, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp., minority)
of bσ(i) with respect to the graph G1(M12 ∩M13 ∪{i}) if a > b. (resp., a < b)
2: For i ∈F23 ∩F13 \ F12, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp.,
minority) of bσ(i) with respect to the graph G1 \µ12 G2(M12 ∩M13 ∪{i}) if a > b (resp., a < b).
3: For i ∈F12 ∩F23 \ F13, set bσ(i) ∈{−1, 1} according to the neighborhood majority (resp.,
minority) of bσ(i) with respect to the graph G1 \µ13 G3(M12 ∩M13 ∪{i}) if a > b (resp., a < b).
4: Return bσ : [n] →{−1, 1}.

The complete exact recovery algorithm is exhibited in Algorithm 4. First, the 13-core matchings are
preformed. Next, the “good” vertices are labeled according to the union graph. Finally, the “bad”
vertices are labeled according to neighborhood labels in G1 or G1 \µ12 G2 or G1 \µ13 G3.

Algorithm 4 Full Community Recovery

Input: Three graphs (G1, G2, G3) on n vertices, k = 13, and ϵ > 0.
Output: A labeling of [n] given by bσ.

1: Apply Algorithm 1 on input (Gi, Gj, k), obtaining a matching (c
Mij, bµij), i ̸= j ∈{1, 2, 3}.
Denote c
M := (c
M13 ∩c
M32) ∪(c
M12 ∩c
M13) ∪(c
M23 ∩c
M12).

2: Apply Algorithm 2 on input (G1, G2, G3, c
M12, c
M23, c
M13, bµ13, bµ12, bµ23), obtaining a labeling
bσ : c
M →{−1, 1}.

3: Apply Algorithm 3 on input (G1, G2, G3, c
M12, c
M23, c
M13, bµ13, bµ12, bµ23, bσ), obtaining a labeling
bσ : [n] →{−1, 1}.
4: Return bσ : [n] →{−1, 1}.

D
Preliminaries

Here we provide some useful preliminary propositions.

16


---Page Break---
D.1
Binomial Probabilities

Lemma D.1. Suppose that a ≥b. Let Y ∼Bin(m+, a log(n)/n) and Z ∼Bin(m−, b log(n)/n)
be independent. If m+ = (1 + o(1))n/2, m−= (1 + o(1))n/2, then for any ϵ > 0,

P(Y −Z ≤ϵ log n) ≤n−D+(a,b)+ϵ log(a/b)/2+o(1).

Proof. Proved by [22, Lemma 3.3].

D.2
A useful construction of three correlated stochastic block models

In this section, we elaborate on an alternative method for constructing three correlated SBMs, which
emphasizes the independent regions of G1, G2 and G3. we detail the construction for three graphs
to maintain reasonable and manageable notation throughout our discussion. The extension of these
ideas to the general case of K graphs follows a similar structure where the key steps and arguments
can be directly applied. This construction is analogous to the construction from [22, Section 3.2],
generalizing the case from two graphs to three graphs.

Firstly, we construct a random partition {Eijk, i, j, k, ∈{0, 1}} of
 [n]
2

. Independently, for each pair
{i, j} ∈
 [n]
2

, we let {i, j} ∈{Eijk} with a probability of (1 −s)3−i−j−ksi+j+k. Subsequently, for
each pair {i, j} ∈
 [n]
2

, an edge is constructed between i and j with probability p if the two vertices
are in the same community, and with probability q if they are in different communities. Graph G1 is
constructed using the edges from ∪j,k∈{0,1},i=1Eijk, while the graph G′
2 is constructed using edges
from ∪i,k∈{0,1},j=1Eijk. Graph G2 is then generated from G′
2 and π∗
12 by relabeling the vertices of
G′
2 according to π∗
12. Similarly, the graph G′
3 is constructed using edges from ∪i,j∈{0,1},k=1Eijk and
G3 is obtained from G′
3 and π∗
13 by relabeling the vertices of G′
3 according to π∗
13. This construction
offers an alternative method for generating multiple correlated SBMs and emphasizes regions of
independence between the multiple graphs. The following lemma D.2 describes the idea formally.
Lemma D.2. The random partition construction of correlated SBMs in Section D.2 is equivalent
to the original construction shown in Figure 1. Moreover, conditioned on π∗:= (π∗
12, π∗
13, π∗
23),
σ∗, and E := {Eijk, i, j, k ∈{0, 1}}, the graphs that are comprised of edges in disjoint Eijk are
mutually independent.

Proof. Firstly we show that the distribution of (Ai,j, Bπ∗
12(i),π∗
12(j), Cπ∗
13(i),π∗
13(j)) is the same under
two constructions. Then by the indepence of vertex pairs, the equivalence follows. In the first
construction,
If a + b + c > 0:

P((Aij, Bπ∗
12(i)π∗
12(j), Cπ∗
13(i)π∗
13(j)) = (a, b, c)|π∗, σ∗)

=

sa+b+c(1 −s)3−a−b−cp
if σ∗(i) = σ∗(j),
sa+b+c(1 −s)3−a−b−cq
if σ∗(i) ̸= σ∗(j).

If a + b + c = 0:

P((Aij, Bπ∗
12(i)π∗
12(j), Cπ∗
13(i)π∗
13(j)) = (0, 0, 0)|π∗, σ∗)

=

1 −p + (1 −s)3p
if σ∗(i) = σ∗(j),
1 −q + (1 −s)3q
if σ∗(i) ̸= σ∗(j).

Under the second construction, if σ∗(i) = σ∗(j):

P((Aij, Bπ∗
12(i)π∗
12(j), Cπ∗
13(i)π∗
13(j)) = (a, b, c)|π∗, σ∗) = P({i, j} ∈Eabc) × p

=

sa+b+c(1 −s)3−a−b−cp
if a + b + c > 0,
1 −p + (1 −s)3p
if a + b + c = 0.

If σ∗(i) ̸= σ∗(j), the joint distribution is the same only with p replaced by q. We can see that
the joint distribution under two constructions is the same. To prove the second part of the lemma,
note that conditioned on π∗, σ∗, and the random partition {Eijk, i, j, k ∈{0, 1}}, the edges in Eijk
form independently. Hence the graphs that are comprised of edges in disjoint Eijk are mutually
independent.

17


---Page Break---
Definition D.3. Define the constant:

sabc :=








s3
(a, b, c) = (1, 1, 1),
s2(1 −s)
(a, b, c) ∈{(0, 1, 1), (1, 0, 1), (1, 1, 0)},
s(1 −s)2
(a, b, c) ∈{(0, 0, 1), (1, 0, 0), (0, 1, 0)},
(1 −s)3
(a, b, c) = (0, 0, 0).

The event F holds if and only if

n/2 −n3/4 ≤|V +|, |V −| ≤n/2 + n3/4,

and the following conditions hold for all a, b, c ∈{0, 1}, i ∈[n]:

sabc(|V σ∗(i)| −n3/4) ≤|{j : j ∈Eabc ∩E+(σ∗(i))}| ≤sabc(|V σ∗(i)| + n3/4),

sabc(|V −σ∗(i)| −n3/4) ≤|{j : j ∈Eabc ∩E−(σ∗(i))}| ≤sabc(|V −σ∗(i)| + n3/4).

Lemma D.4. Define sm := mina,b,c∈{0,1} sabc. We have P(Fc) ≤100n exp(−s2
m
√n
2
).

Proof. Denote G holds if and only if n/2−n3/4 ≤|V +|, |V −| ≤n/2+n3/4. The event G is proved
in Lemma 3.8 in [22]. P(Gc) ≤4e−√n.

Then, look at the remaining condition of event F. Fix i ∈[n], condition on σ∗
1, π∗
12, π∗
13. Note that

k+
abc(i) := |{j : j ∈Eabc ∩E+(σ∗
1(i))}| ∼Bin(|V σ∗(i)| −1, sabc).

By Hoeffding inequality we have

P(|k+
abc(i) −sabc(|V σ∗(i)| −1)| ≥sabcn3/4

2
|π∗
12, π∗
13, σ∗
1)1(G)

≤2 exp(−s2
abcn3/2

2|V σ∗(i)|)1(G) ≤2 exp(−(1 −o(1))s2
abc
√n).

Then by a union bound,

P(∃i ∈[n] : |k+
abc(i) −sabc|V σ∗(i)|| ≥sabcn3/4)

≤

n
X

i=1
P(|k+
abc(i) −sabc(|V σ∗(i)| −1)| ≥sabcn3/4

2
)

≤

n
X

i=1
E[P(|k+
abc(i) −sabc(|V σ∗(i)| −1)| ≥sabcn3/4

2
|π∗
12, π∗
13, σ∗
1)1(G)] + P(Gc)

≤2n exp(−(1 −o(1))s2
abc
√n) + 4 exp(−√n) ≤6n exp(−(1 −o(1))s2
m
√n).

Similarly, we can define k−
abc(i) and through an identical proof we have

P(∃i ∈[n] : |k−
abc(i) −sabc|V −σ∗(i)|| ≥sabcn3/4) ≤6n exp(−(1 −o(1))s2
m
√n).

The conclusion then follows by a union bound.

D.3
Almost exact recovery in a single SBM

Lemma D.5. The algorithm (Algorithm 1, [22]) correctly classifies all vertices in [n] \ Iϵ(G), where
Iϵ(G) := {v ∈[n] : majG(v) ≤ϵ log n or N(v) > 100 max{1, a, b} log n} if a > b.

Proof. Directly proved by [22, Lemma 5.1], adapted from [35, Proposition 4.3].

Lemma D.6. Consider a SBM(n, α log n/n, β log n/n), denote γ = max(α, β). Then for every σ∗
we have
P(∀i ∈[n], |N(i)| ≤100 max(1, γ) log n|σ∗) ≥1 −n−99.

Proof. Directly proved by [22, Lemma 5.2], based on arguments of [35].

18


---Page Break---
Lemma D.7. With the assumption of D+(a, b) < 99, E(|Iϵ(G)|) ≤3n1−D+(a,b)+ϵ| log(a/b)|.

Proof. Proved by [22, Lemma 5.3].

Lemma D.8. If 0 < ϵ <
D+(a,b)
2 log(a/b), then

P(∀i ∈[n], |N(i) ∩Iϵ(G)| ≤2⌈D+(a, b)−1⌉|σ∗) = 1 −o(1).

Proof. Proved by [22, Lemma 5.4].

E
Analysis of the k-core estimator

In this section, we prove two important properties of k-core estimator. Lemma E.4 describes that all
vertices have weak connections with those vertices who are not part of the k-core, in the logarithmic
regime that we are interested in. Lemma E.1 argues that all the vertices with degree larger than a
given constant will be part of the k-core.

Lemma E.1. Fix a, b > 0. Consider the graph G ∼SBM(n, a log n

n
, b log n

n
). For any integer m
satisfying m >
2
a+b, all vertices whose degree is greater than m + k are part of the k-core with high
probability.

Proof. For a given m >
2
a+b, we would like to prove that any vertex v with degree greater than
k + m will not be part of the k-core with probability o(n−1). The lemma then follows by a union
bound.

Isolating vertex v for independence.

For a fixed v ∈[n], consider the graph eG := G{[n]\v}. Now we look at the k-core of eG, denote it by
Ck( eG). Since the degG(v) > k +2/(a+b), we can suppose that degG(v) = m+k, m > 2/(a+b).
If the vertex v is not part of the k-core of G, it must has more than m neighbors who are /∈Ck( eG).
Note that the event w /∈Ck( eG) is independent of the event w ∈NG(v), while the latter event
is stochastically dominated by a binomial distribution with probability ν log n/n, ν = max(a, b).
Hence, by the tower rule,

P(v is not part of k - core in G) ≤P(|{w ∈NG(v) : w /∈Ck( eG)}| > m)
≤E[1Bin(|{w:w /∈Ck( e
G)}|,ν log n/n)>m].
(E.2)

The size of {w : w /∈Ck( eG)} can not be directly quantified. Hence, we would like to find a set U
based on eG such that {w ∈[n] \ v : w /∈Ck( eG)} ⊂U, where we can bound the size of U. Now we
denote µ = |U|ν log n/n, then we have that

E[1Bin(|{w:w /∈Ck( e
G)}|,ν log n/n)>m] ≤E[1Bin(|U|,ν log n/n)>m] ≤E[min
t>0 exp(µ(et −1) −tm)].

(E.3)

To construct U, the idea is motivated by Łuczak expansion in [27]. We consider a modified version
of expanding the set in our setting.

Quantify the set U.

Define U to be the set of vertices with degree at most T in the graph eG. The choice of T would
be specified later. Denote H := {n/2 −n3/4 ≤|V +|, |V −| ≤n/2 + n3/4}. By Lemma D.4,

19


---Page Break---
P(Hc) = o(1/n).

E(|U|) ≤E(|U|1H) + P(Hc)n = E(|U|1H) + o(1)

≤n

T
X

i=0

i
X

j=0

(1 + o(1) n

2 )
j

(1 + o(1) n

2 )
i −j


pj(1 −p)(1−o(1)) n

2 −jqi−j(1 −q)(1−o(1)) n

2 −i+j

≤2n(1 −a log n

n
)(1−o(1)) n

2 (1 −b log n

n
)(1−o(1)) n

2

T
X

i=0

i
X

j=0

(1 + o(1))n

2

i
(ν log n

n
)i

≤2n1−a+b

2 +o(1)
T
X

i=0
(i + 1)
(1 + o(1))ν log n

2

i

≤2(T + 1)2(ν log n

2
)T n1−a+b

2 +o(1) = n1−a+b

2 +o(1).

Consider the situation when 1 −a+b

2
> 0. Now we claim: for any constant W, E[|U|W ] ≤
nW −W a+b

2 +o(1). Suppose for W −1, it is true, then for W:

E[|U|W ] =
X

i1,...,iW
P(i1 ∈U, . . . , iW ∈U)

=
X

i1̸=i2...̸=iW
P(i1 ∈U, . . . , iW ∈U)

+
X

i1̸=i2...̸=iW −1
P(i1 ∈U, . . . , iW −1 ∈U) + . . . +
X

i1∈[n]
P(i1 ∈U)

≤
X

i1̸=i2...̸=iW
P(i1 ∈U, . . . , iW ∈U) + E(|U|W −1)

≤
X

i1̸=i2...̸=iW
P(i1 ∈U, . . . , iW ∈U) + E[|U|W −1].

The remaining thing is to show that P

i1̸=i2...̸=iW P(i1 ∈U, . . . , iW ∈U) ≤nW −W a+b

2 +o(1). We
have
X

i1̸=i2...̸=iW
P(i1 ∈U, . . . , iW ∈U)

≤
X

i1̸=i2...̸=iW
P(∩W
j=1{ij has at most T neighbours in [n] \ i1, ..., iW , v})

=
X

i1̸=i2...̸=iW
P({i1 has at most T neighbours in [n] \ i1, ..., iW , v})W

≤E[|U|]W ≤nW −W a+b

2 +o(1).

The first inequality is because that if i1 ∈U in eG, then i1 has at most T neighbours in [n] \ v, then it
implies that {i1 has at most T neighbours in [n] \ i1, ..., iW , v}. The second inequality is due to the
independence of the events. By induction, the claim follows. By choosing appropriate m′-th moment
method of |U|, we can select a ϵ such that P(|U| ≥n1−ϵ) ≤n−m′(a+b)/2+m′ϵ+o(1) = o(n−m a+b

2 ).
Denote D0 as |U| ≤n1−ϵ, then P(Dc
0) = o(n−m a+b

2 ).

The possibility of the existence of a well-connected small subgraph.

Now we would like to bound the probability of the existence of a well-connected small subgraph.
Define the event

D := {there exists S ∈[n] such that |S| < n1−ϵ and G{S} has at least N|S| edges }.

We would like to bound P(D) = o(n−m(a+b)/2). Let S be a κ-vertex subset of [n]. Let XS
be the indicator variable that is 1 if the subgraph induced by S has at least N|S| edges. Denote

20


---Page Break---
ν = max(a, b), then we have

E[
X

S∈[n],|S|=κ
XS] ≤
X

S∈[n],|S|=κ

 κ
2


Nκ


(ν log n

n
)Nκ ≤
X

S∈[n],|S|=κ
(κeν log n

Nn
)Nκ

≤
n
κ


(κeν log n

Nn
)Nκ ≤((e1+1/Nν log n

N
)N(κ

n)N−1)κ.

The second and the third inequality is because
 n
k

≤( en

k )k. Under the assumption |S| < n1−ϵ,

(e1+1/Nν log n

N
)N(κ

n)N−1 < n−ϵ(N−1)+o(1).

Hence for n sufficiently large we have:

E[
X

S∈[n],|S|<n1−ϵ
XS] ≤

n1−ϵ
X

κ=1
(n−ϵ(N−1)+o(1))κ ≤n−ϵ(N−1)+o(1).

Then by Markov’s inequality, P(D) ≤n−ϵ(N−1)+o(1). If we want to bound P(D) = o(n−m(a+b)/2),
set N > (a+b)m

2ϵ
+ 1. Hence,

P(D) ≤E[
X

S∈[n],|S|<n1−ϵ
XS] ≤n−ϵ(N−1)+o(1) < n−m(a+b)/2.

Identify the expansion set of U.

Now we do the following expansion on U, the expansion process is adapted from Łuczak expansion
first introduced in [27].

1. Define U0 := U.

2. Given Ut, define U 1
t+1 to be the set of those vertices outside Ut which have at least N ′

neighbors in Ut. If U 1
t+1 is non-empty, set Ut+1 = Ut ∪{u}, where u is the first vertex in
U 1
t+1. Otherwise, stop the expansion with the set Ut.

Suppose the expansion ends at the step h, hence we have an increasing sequence {Us}h
s=0. Denote
U := Uh to be the set after expansion.

Now claim that on the event Dc ∩D0, we can choose N1, N ′ > 0, such that |U| ≤N1|U|.
Suppose that |U| > N1|U|, then there exists ℓ> 0 s.t. |Uℓ| = N1|U|. On event D0, there exists
ϵ > 0, |U| ≤n1−ϵ, hence |Uℓ| = N1|U| ≤n1−ϵ+o(1). Denote el as the number of edges in
eG{Uℓ}. Each step in the expanding process, at least N ′ edges are added into the graph, hence
eℓ≥N ′ℓ≥N ′(|Uℓ| −|U|) = (N ′ −N ′

N1 )|Uℓ|. We can choose N ′, N1 such that (N ′ −N ′

N1 ) ≥N.
However on the event Dc, the set |Uℓ| < n1−ϵ cannot have at least N|Uℓ| edges, which is a
contradiction. Therefore, on the event Dc ∩D0, |U| ≤N1|U|. Subsequently,

E[|U|m] ≤N m
1 E[|U|m1Dc∩D0] + nmP(D) + nmP(D0)

≤N m
1 E[|U|m] + o(nm−m(a+b)/2) ≤nm−m a+b

2 +o(1).

Bound the probability of v being in the k-core.

Note that eG{[n] \ U} has minimum degree at least T −N ′. If a vertex i ∈[n] \ U, then i /∈U,
it follows that i has degree at least T in eG. However i can have at most N ′ neighbor in U by
construction of expansion process, so i must have at least T −N ′ neighbors in [n] \ U. We can set
T = k + N ′, then If i ∈[n] \ U, i is part of k-core in eG.

Since the deg(v) ≥m + k, it must has at least m neighbors who are not part of k-core in eG.

21


---Page Break---
Follow the equation (E.2), (E.3), denote µ = |U|ν log n/n, then we have

E[1Bin(|U|,ν log n/n)>m] ≤E[min
t>0 exp(µ(et −1) −tm)] ≤E[eµm] + o(n−1)

= eνm log nm

nm
E[|U|m] + o(n−1) ≤n−(a+b)m/2+o(1) + o(n−1)

= o(n−1).

The second inequality follows by setting t = log(1/µ). This is valid since P(|U| > n1−c) ≤
E[|U|W ]
n(1−c)W ≤n−((a+b)/2−c)W +o(1). We can select 0 < c < a+b

2 , W > 0 such that P(|U| > n1−c) =
o(n−1). On the event |U| ≤n1−c, µ = o(1), 1/µ > 1. The third inequality follows by E[|U|m] ≤
nm−m a+b

2 +o(1) and the last equality is due to m >
2
a+b.

If 1 −(a + b)/2 ≤0, we can directly set m = 1. Similarly, we can prove E[|U|] ≤n1−a+b

2 +o(1),
then through a similar calculation, P( v is not part of k-core in G}|) = o(n−1).

Hence, by a union bound, we can say that all vertices with degree larger than m + k will be part of
k-core with high probability.

Lemma E.4. Fix a, b, ε > 0. Let G ∼SBM(n, a log n

n
, b log n

n
). Then, w.h.p., all vertices have at
most ε log n neighbors who are not part of the k-core.

Proof. For v ∈[n], consider the graph eG := G{[n]\v}. Following the same arguments in Lemma E.1,
we can obtain U. We have

P(|{w ∈NG(v) : w is not part of k-core in G}| > ϵ log n)

≤P(|{w ∈NG(v) : w /∈Ck( eG)}| > ϵ log n)

≤P(|{w ∈NG(v) : w /∈Ck( eG)}| > m) ≤o(n−1).

Based on the proof of Lemma E.1, we can show the lemma follows immediately when n is sufficiently
large. Hence, all vertices have at most ϵ log n neighbors who are not part of the k- core with probability
1 −o(1).

Lemma E.5. Fix constants a, b > 0, s ∈[0, 1]. Let (G1, G2) ∼CSBM(n, a log n

n , b log n

n , s). Let
M ∗be the set of vertices of the 13-core in the graph G1 ∧π∗G2. π∗is the permutation of vertices
from G1 to G2. Let (M1, µ1) be the output of the k-core match of (G1, G2), k = 13. Then
P((M1, µ1) = (M ∗, π∗)) = 1 −o(1).

Remark: We can replace (Mk, µk) by (M ∗
k, π∗
k{M ∗}) in any analysis.

Proof. Proved by [22, Lemma 4.8].

Lemma E.6. Let G ∼SBM(n, a log n

n , b log n

n ) for fixed a, b > 0. Fix k ≥1. With probability
1 −o(1), we have that |F| ≤n1−Tc(a,b)+o(1).

Proof. Proved by [22, Lemma 4.13].

Lemma E.7. Suppose G1, G2, G3 are independently subsampled with probability s from a parent
graph G ∼SBM(n, a log n/n, b log n/n) for a, b > 0. Let F ∗
ij be the set of vertices outside the
k-core of Gi ∧πij Gj(taking vertice index in i) with k = 13. Prove that with probability 1 −o(1),
|F ∗
ij ∩F ∗
jk| ≤n1−(2s2−s3)Tc(a,b)+δ, for any δ > 0.

Proof. |F ∗
12 ∩F ∗
23| = |F ∗
21 ∩F ∗
23| = |F ∗
12 ∩F ∗
13|, by symmetricity of G1, G2, G3.

Define Uij to be the set of vertices with degree at most m + k in the intersection graph Gi ∧πij Gj
which marginally follows SBM(n, as2 log n

n
, bs2 log n

n
), and m is an integer satisfying m >
2
s2(a+b).
Then by Lemma E.1, w.h.p., F ∗
ij ⊂Uij. Hence |F ∗
12 ∩F ∗
23| ≤|U12 ∩U23| with high probability.

22


---Page Break---
It thus remains to bound |U12 ∩U13|. Firstly, we bound the expectation of |U12 ∩U13|:

E[|U12 ∩U13|] =

n
X

v=1
E[1v∈U12∩U13] = nE[1v∈U121v∈U13].

Let D1 denote the degree of vertex v in the graph G1. Let Xa ∼Bin((1 + o(1))n/2, sa log n/n),

and Xb ∼Bin((1 + o(1))n/2, sb log n/n). On the event F, D1
d= Xa + Xb, where F is defined in
Definition D.3, Xa, Xb are independent. Note that by Lemma D.4, P(Fc) = o( 1

n2 ). We have

E[1v∈U121v∈U13] = E[E[1v∈U121v∈U13|D1]] = E





 m+k
X

i=0

D1
i


si(1 −s)D1−i
!2



=

m+k
X

i=0

m+k
X

j=0
C(i, j)E[Di+j
1
(1 −s)2D1].
(E.8)

Here C(i, j) is a constant related to i, j. Now look at E[DL
1 (1−s)2D1]. In our regime, L ≤2(m+k)
are constant. Hence:
E[DL
1 (1 −s)2D11F] = E[(Xa + Xb)L(1 −s)2Xa(1 −s)2Xb1F]

=

K
X

t=0
CtE[Xt
a(1 −s)2Xa1F]E[XL−t
b
(1 −s)2Xa1F].
(E.9)

Here Ct is constant related to t, the second equality is due to the independence of Xa, Xb. Now look
at E[Xt
a(1 −s)2Xa1F].
E[Xt
a(1 −s)2Xa1F] ≤E[Xt
a(1 −s)2Xa]

=

(1+o(1))n/2
X

ℓ=0
ℓt(1 −s)2ℓ(sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

=

(log n)3
X

ℓ=0
ℓt((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

+

(1+o(1))n/2
X

ℓ=(log n)3+1
ℓt((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ.

We can bound the first part:

(log n)3
X

ℓ=0
ℓt((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

≤(log n)3t
(log n)3
X

ℓ=0
((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

≤(log n)3t
(1+o(1))n/2−ℓ
X

ℓ=0
((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

=(log n)3t(1 −(1 −(1 −s)2)sa log n

n
)(1+o(1))n/2 ≤n−(1−(1−s)2)sa/2+o(1)

Then we can bound the second part, note that

(1+o(1))n/2
X

ℓ=(log n)3+1
ℓt((1 −s)2sa log n

n
)ℓ(1 −sa log n

n
)(1+o(1))n/2−ℓ

=

(1+o(1))n/2
X

ℓ=(log n)3+1
ℓt(1 −s)2ℓP(Xa = ℓ) ≤(log n)3t(1 −s)2(log n)3P(Xa > (log n)3)

≤n2(log n)2 log(1−s)+o(1) = o(n−(1−(1−s)2)sa/2+o(1)).

23


---Page Break---
The first inequality is true because ℓt(1 −s)2ℓdecreases when ℓ> (log n)3 for sufficiently large n.
The last equality is true because (log n)2 log(1 −s) < −(1 −(1 −s)2)sa/2 for sufficiently large n.

Hence, by summing up the two parts, E[Xt
a(1 −s)2Xa] ≤n−s(1−(1−s)2)a/2+o(1), similarly we can
proof E[XL−t
b
(1 −s)2Xb] ≤n−s(1−(1−s)2)b/2+o(1). By (E.8) and (E.9),

E[1v∈U121v∈U13] ≤P(F)c + n−s(1−(1−s)2)Tc(a,b)+o(1) = n−s(1−(1−s)2)Tc(a,b)+o(1) + o( 1

n2 ).

By Markov’s inequality we have

P(|U12 ∩U13| ≥log nE[|U12 ∩U13|]) ≤
1
log n = o(1).

Hence we have |F ∗
12 ∩F ∗
23| ≤|U12 ∩U13| ≤log nE[|U12 ∩U13|] ≤n1−s(1−(1−s)2)Tc(a,b)+δ, for
any δ > 0, with probability 1 −o(1).

F
Proof of Theorem 1 for three graphs

F.1
Exact recovery in [n] \ (F12 ∪F23) ∪[n] \ (F12 ∪F13) ∪[n] \ (F13 ∪F23)

Definition F.1. For a vertex i in G, define the quantity majority of i:

majG(i) := |NG(i) ∩V σ∗(i)| −|NG(i) ∩V −σ∗(i)|.

By Lemma D.1, we can directly deduce that if (1 −(1 −s)3)D+(a, b) > 1 + ϵ| log(a/b)|, then for
all i ∈[n] we have that majG1∨π12G2∨π13G3(i) ≥ϵ log n with probability 1 −o(1).

F.2
Exact recovery in [n] \ (F12 ∪F23)

Now suppose that i ∈M12 ∩M23. Look at eG := (G1 ∨µ12 G2 ∨µ23◦µ12 G3)([n] \ {F12 ∪F23}).

Lemma F.2. Suppose that (1 −(1 −s)3)D+(a, b) > 1 + 2ϵ| log(a/b)|. Then with probability
1 −o(1), all vertices in (M12 ∩M23) have an ϵ log n majority in eG{M12 ∩M23}.

Proof. Denote F ∗
ij the set of vertices outside the 13-core of Gi ∧π∗
ij Gj. In light of Lemma E.5 and
its remark, we can replace µij with π∗
ij, Fij with F ∗
ij in Lemma F.2. Where we define

G∗:= G1 ∨π∗
12 G2 ∨π∗
13 G3,

H := G∗{M ∗
12 ∩M ∗
23}.
To bound the neighborhood majority in H, for i ∈M ∗
12 ∩M ∗
23 note that:

majH(i) = σ∗(i)
X

j∈NH(i)
σ∗(j) ≤majG∗(i) + |NG∗(i) ∩{F ∗
12 ∪F ∗
23}|,

majH(i) = σ∗(i)
X

j∈NH(i)
σ∗(j) ≥majG∗(i) −|NG∗(i) ∩{F ∗
12 ∪F ∗
23}|.

To sum up, we have

|majH(i) −majG∗(i)| ≤|NG∗(i) ∩{F ∗
12 ∪F ∗
23}|.
(F.3)

Note that majG∗(i) > 2ϵ log n, i ∈[n] with probability 1−o(1), given that (1−(1−s)3)D+(a, b) >
1 + 2ϵ| log(a/b)| by Lemma D.1. Now we prove that the right hand side of (F.3) can be bounded by
ϵ log n. Look at |NG∗(i) ∩{F ∗
12 ∪F ∗
23}|,

|NG∗(i) ∩{F ∗
12 ∪F ∗
23}| ≤|NG1∧G2(i) ∩(F ∗
12)| + |NG2∧G3(π∗
12(i)) ∩(F ∗
23)|
+ |NG∗\(G1∧G2)(i) ∩(F ∗
12)| + |NG∗\(G2∧G3)(i) ∩(F ∗
23)|.

First, look at |NG1∧G2(i) ∩F ∗
12|, by Lemma E.4, w.h.p.,

|NG1∧G2(i) ∩F ∗
12| < ϵ log n/8.

24


---Page Break---
Similarly, we have w.h.p.

|NG2∧G3(π∗
12(i) ∩F ∗
23)| < ϵ log n/8.

What is left is to bound |NG∗\(G2∧G3)(i) ∩(F ∗
23)|, |NG∗\(G1∧G2)(i) ∩(F ∗
12)|.

Note that conditioned on π∗
12, π∗
13, π∗
23, σ∗, E := {Eijk, i, j, k ∈{0, 1}}, the graph G∗\(G1 ∧G2) is
independent of F ∗
12 by Lemma D.2, since F ∗
12 depends only on G1 ∧G2. Thus we can stochastically
dominate |NG∗\(G1∧G2)(i) ∩F ∗
12| by a Poisson random variable X with mean

λn := ν log n

n
|{j ∈F ∗
12 : {i, j} ∈E100∪E101∪E001∪E010∪E011}| ≤ν log n

n
|F ∗
12|, ν := max(a, b).

For a fixed δ > 0, define an event Z := {|F ∗
12| ≤n1−s2Tc(a,b)+δ}. On Z, we have that λn ≤
n−s2Tc(a,b)+δ+o(1). Hence, for any positive integer m:

P({|NG∗\(G1∧G2)(i)∩(F ∗
12)| ≥m}∩Z) ≤P({X ≥m}∩Z) = E[P(X ≥m|F ∗
12, E, σ∗, π∗)1Z]

≤E[(inf
θ>0 e−θm+λn(eθ−1))1Z] ≤E[eλm
n 1Z] ≤n−m(s2Tc(a,b)−δ−o(1)).

Above, the equality on the second line is due to the tower rule and since Z is measurable with respect
to |F ∗
12|, the inequality on the third line is due to a Chernoff bound; the inequality on the fourth line
follows from setting θ = log(1/λn) (which is valid since λn = o(1) if Z holds). The final inequality
uses the upper bound for λn on Z. Taking a union bound, we have

P({∃i ∈[n], |NG∗\(G2∧G1)(i) ∩F ∗
12| ≥m} ∩Z) ≤n1−m(s2Tc(a,b)−δ−o(1)).

Here if we take m > (s2Tc(a, b))−1 and δ < s2Tc(a, b) −m−1, the probability turns to o(1). Thus,
we can set m = ⌈(s2Tc(a, b))−1⌉+ 1. In light of Lemma E.6, |F ∗
12| ≤n1−s2Tc(a,b)+δ, δ > 0 w.h.p.
Hence, the event Z happens with probability 1 −o(1). Hence we have

P({∀i ∈[n], NG∗\(G2∧G1)(i) ∩F ∗
12| ≤⌈(s2Tc(a, b))−1⌉}) = 1 −o(1).

By an identical proof, we have that
P({∀i ∈[n], NG∗\(G2∧G3)(i) ∩F ∗
23| ≤⌈(s2Tc(a, b))−1⌉}) = 1 −o(1).

Hence we have, with probability 1 −o(1), for i ∈M ∗
12 ∩M ∗
23,
|majH(i) −majG1∨π∗
12G2∨π∗
13G3(i)| < ϵ log n,

and hence with probability 1 −o(1),
majH(i) > ϵ log n.

Then by Lemma E.5, we can replace H with eG, F ∗
ij with Fij, the lemma follows.

Next, prove that each vertex in G2 ∨π∗
23 G3 \π∗
12 G1 has a small number of neighbors in π∗
12(Iϵ(G1))

Lemma F.4. If 0 < ϵ ≤
sD+(a,b)
4| log(a/b)|, then

P(∀i ∈[n], |NG2∨π∗
23G3\π∗
12G1(i) ∩π∗
12(Iϵ(G1))| ≤2⌈(sD+(a, b))−1⌉) = 1 −o(1).

Proof. Since Iϵ(G1) depends on G1 alone, it follows that Iϵ(G1) and G2 ∨π∗
23 G3 \π∗
12 G1 are condi-
tionally independent given π∗, σ∗, E. Hence we can stochastically dominate |NG2∨π∗
23G3\π∗
12G1(i) ∩
π∗
12(Iϵ(G1))| by a Poisson random variable X with mean λn given by
λn := ν log n/n|{j ∈Iϵ(G1) : {i, j} ∈E011 ∪E010 ∪E001}| ≤ν log n/n|Iϵ(G1)|.

Next, define the event Z := {|Iϵ(G1)| ≤n1−sD+(a,b)+2ϵ| log(a/b)|}.

Notice that P(Z) = 1 −o(1) by Lemma D.7 and Markov’s inequality, provided sD+(a, b) < 99.
Following identical arguments as the proof of Lemma F.2, we arrive at
P(∃i ∈[n], |NG2∨π∗
23G3\π∗
12G1(i) ∩π∗
12(Iϵ(G1))| ≥m) = o(1),

when m
>
⌈(sD+(a, b) −2ϵ| log a/b|)−1⌉.
If ϵ
≤
sD+(a,b)
4| log(a/b)|, it suffices to set m
=
2⌈(sD+(a, b))−1⌉+ 1.

25


---Page Break---
Lemma F.5. Suppose that a, b, ϵ > 0 satisfy the following conditions:

(1 −(1 −s)3)D+(a, b) > 1 + 2ϵ| log a/b|,
0 < ϵ ≤sD+(a, b)

4| log a/b|.

With high probability, the algorithm correctly labels all vertices in {i ∈[n] \ (F ∗
12 ∪F ∗
23)}.

Proof. Compare the neighborhood majority in H corresponding to bσ1 with the true majority in H,
where H is defined in Lemma F.2:

|σ∗(i)
X

j∈NH(i)
(bσ1(j) −σ∗(j))| ≤|NH(i) ∩Iϵ(G1)| ≤|NG∗(i) ∩Iϵ(G1)|

≤|NG2∨π∗
23G3\π∗
12G1(i) ∩π∗
12(Iϵ(G1))| + |NG1(i) ∩Iϵ(G1)|

≤2⌈D+(a, b)−1⌉+ 2⌈(sD+(a, b))−1⌉≤ϵ log n/2.

The first inequality uses Lemma D.5 that the set of errors are contained in Iϵ(G1). The last inequality
is due to Lemma D.8, F.4. Notice that majH(i) ≥ϵ log n for i ∈[n] \ (F ∗
12 ∪F ∗
23). Hence,
σ∗(i) P

j∈NH(i) bσ1(j) ≥majH(i) −|σ∗(i) P

j∈NH(i)(bσ1(j) −σ∗(j))| ≥ϵ log n/2 > 0, which
implies that the sign of neighborhood majorities are equal to the truth community label for any
i ∈[n] \ (F ∗
12 ∪F ∗
23), with probability 1 −o(1). Then we can convert H to eG{[n] \ (F12 ∪F23)},
the vertices in [n] \ (F12 ∪F23) are correctly labeled with probability 1 −o(1).

Using an identical proof, we can argue that the algorithm correctly label all vertices in M13 ∩M32
and M12 ∩M13.

F.3
Exact recovery in [n] \ {(M13 ∩M32) ∪(M12 ∩M13) ∪(M23 ∩M12)}

Define M = (M13 ∩M32) ∪(M12 ∩M13) ∪(M23 ∩M12). Denote Fb = (F12 ∩F13) ∪(F12 ∩
F23 \ F13) ∪(F13 ∩F23 \ F12), note that Fb = [n] \ M.
Lemma F.6. Suppose that a, b, ϵ > 0 satisfy the following conditions:

(1 −(1 −s)3)D+(a, b) > 1 + 2ϵ| log a/b|,
0 < ϵ ≤sD+(a, b)

4| log a/b|,

s(1 −(1 −s)2)Tc(a, b) + s(1 −s)2D+(a, b) > 1.

With high probability, the algorithm correctly labels all vertices that are in Fb.

Proof. For i ∈Fb, define Hi := (G1 \π∗
13 G3 \π∗
12 G2){(M12 ∩M13) ∪{i}}. Let Ei be the event
that i has a majority of at most ϵ′ log n in the graph Hi. Let bσ be the labeling after the step. For
bervity, define a “nice” event based on the previous results. Define the event H, which holds if and
only if:

• Fij = F ∗
ij;

• c
Hi = Hi;

• bσ(i) = σ∗(i) for all i ∈M12 ∩M13;

• The event F holds;

• |Fb| ≤n1−(2s2−s3)Tc(a,b)+δ.

By Lemmas E.5, E.7, D.4, F.5, the event H holds with probability 1 −o(1). Furthermore, define
E∗
i := majHi(i) ≤ϵ′ log n, we have:

P(∪i∈[n]({i ∈Fb} ∩Ei)) ≤P((∪i∈[n]({i ∈F ∗
b } ∩E∗
i )) ∩H) + P(Hc)

≤

n
X

i=1
P({i ∈F ∗
b } ∩E∗
i ∩{F ∗
b ≤n1−(2s2−3s3)Tc(a,b)+δ} ∩F) + o(1).

(F.7)

26


---Page Break---
By the tower rule, rewrite the term in the right hand side as:

E[P(E∗
i |π∗, σ∗, E, F ∗
b )1i∈F ∗
b 1{|F ∗
b |≤n1−(2s2−s3)Tc(a,b)+δ}∩F].
(F.8)

Now look at P(E∗
i |π∗, σ∗, E, F ∗
b ). Conditional on E, σ∗, π∗, majHi(i) :
d= Y −Z, where Y, Z are
independent with:

Y ∼Bin(|j ∈M ∗
12 ∪M ∗
13 : {i, j} ∈E100 ∩E+(σ∗)|, a log n/n),

Z ∼Bin(|j ∈M ∗
12 ∪M ∗
13 : {i, j} ∈E100 ∩E−(σ∗)|, b log n/n).

By Definition D.3 of the event F, we know that |j ∈M ∗
12 ∪M ∗
13 : {i, j} ∈E100 ∩E−(σ∗)| =
(1 −o(1))s(1 −s)2n/2 and |j ∈M ∗
12 ∪M ∗
13 : {i, j} ∈E100 ∩E+(σ∗)| = (1 −o(1))s(1 −s)2n/2.

Lemma D.1 implies

P(E∗
i |π∗, σ∗, E, F ∗
b )1i∈F ∗
b 1{|F ∗
b |≤n1−(2s2−s3)Tc(a,b)+δ}∩F ≤n−s(1−s)2D+(a,b)+ϵ′ log(a/b)/2+o(1).

Follow (F.8) and take a union bound, we have

n
X

i=1
P({i ∈F ∗
b } ∩E∗
i ∩{F ∗
b ≤n1−(2s2−s3)Tc(a,b)+δ} ∩F) + o(1)

≤n−s(1−s)2D+(a,b)+ϵ′ log(a/b)/2+o(1)E[|F ∗
b |1F ∗
b ≤n1−(2s2−s3)Tc(a,b)+δ]

≤n1−(2s2−s3)Tc(a,b)−s(1−s)2D+(a,b)+ϵ′ log(a/b)/2+δ+o(1).

Under the condition (2s2−s3)Tc(a, b)+s(1−s)2D+(a, b) > 1, we can choose ϵ′, δ small enough so
that the right hand side is o(1). majHi(i) > ϵ′ log n for i ∈F ∗
b , by Lemma E.5, majc
Hi(i) > ϵ′ log n
for i ∈Fb.

Suppose that i ∈F13∩F12, i has at most 12 neighbors in the graph (G1∧π∗
12 G2){(M12∩M13)∪{i}},
and in the graph (G1 ∧π∗
13 G3){(M12 ∩M13) ∪{i}}. Therefore, i has an at least (ϵ′ log n −24)
majority in G1{(M12 ∩M13) ∪{i}}, with high probability. Then, Algorithm 3 correctly label all
vertices in F13 ∩F12.

Suppose that i ∈F12 ∩F23 \ F13, i has at most 12 neighbors in the graph (G1 ∧π∗
12 G2){(M12 ∩
M13) ∪{i}} Therefore, i has an at least (ϵ′ log n −12) majority in G1 \µ13 G3{(M12 ∩M13) ∪{i}},
with high probability. Hence, Algorithm 3 correctly label all vertices in F12 ∩F23 \ F13.

Suppose that i ∈F13 ∩F23 \ F12, i has at most 12 neighbors in the graph (G1 ∧π∗
13 G3){(M12 ∩
M13)∪{i}}. Therefore, i has an at least (ϵ′ log n−12) majority in G1 \µ12 G2{(M12 ∩M13)∪{i}},
with high probability. Algorithm 3 correctly label all vertices in F13 ∩F23 \ F12.

G
Proof of impossibility for three graphs

In this section we prove that Theorem 2 when the exact community recovery is impossible. The
impossibility under the condition (1 −(1 −s)3)D+(a, b) < 1 has been proved in [41]. Hence we
focus on proving impossibility when

(2s2 −s3)Tc(a, b) + s(1 −s)2D+(a, b) < 1.
(G.1)

To prove it, we study the MAP (maximum a posterior) estimator for the communities in G1. Even
with the additional information provided, including all the correct community labels in G2, the
true matching π∗
23 and most of the true matching π∗
12, the MAP estimator fails to exactly recovery
communities with probability bounded away form 0 if the condition (G.1) holds. The proof is
adapted from the MAP analysis in [22]. The difference is that we are considering three correlated
SBM G1, G2, G3. Since we know the true matching π∗
23, we can consider H := G2 ∨π∗
23 G3 ∼
SBM(n, (1 −(1 −s)2)a log n/n, (1 −(1 −s)2)b log n/n). Denote Rij the singleton in Gi ∧Gj.
Then R = R12 ∧R13 is the singleton set in G1 ∧H.

27


---Page Break---
G.1
Notation

Here we review and introduce some notations in brief.
σ∗
i := the ground truth community labels in Gi, i = 1, 2, 3,
V +
i
:= {j ∈[n] : σ∗
i (j) = +1}, V −
i
:= {j ∈[n] : σ∗
i (j) = −1},
σ∗
2(π∗
12(i)) = σ∗
1(i),
here we have V +
2 = π∗
12(V +
1 ).

G.2
The MAP estimator

First define the singleton set of a permutation π with respect to the adjacency matrices A, B, and C
to be:
R(π, A, B, C) := {i ∈[n] : ∀j ∈[n], Ai,jDπ(i),π(j) = 0},

where Dij := max{Bij, Cπ∗
23(i)π∗
23(j)}. For brevity, write Rπ := R(π, A, B, C).

Definition G.2. Define the set S(π, A, B, C) as followings:

1. i ∈R(π, A, B, C);

2. i is a singleton in G1{Rπ};

3. If j ∈N1(i), π(j) /∈NH(π(Rπ)).

Where A, B, C is the adjacency matrix of G1, G2, G3 respectively and D is the adjacency matrix
in H = G2 ∨π∗
23 G3. Note that D is the adjacency matrix of H, so NH(π(Rπ)) = {i ∈[n] : ∃k ∈
π(Rπ), Di,k = 1}.

Define ¯Rπ := Rπ ∪π−1(NH(π(Rπ))). The condition 2 and 3 in Definition G.2 can be replaced by
Ai,j = 0 for all j ∈¯Rπ. Write R∗= R(π∗
12, A, B, C), S∗= S(π∗
12, A, B, C), and ¯R∗= ¯Rπ∗
12 for
brevity. We study the MAP estimate provided the additional knowledge σ∗
2, π∗
23, and π∗
12{[n] \ S∗}.
Theorem 5. Let A, B, C, σ∗
2, π∗
23, π∗
12{[n] \ S∗}, S∗be given. For i ∈[n] \ S∗, bσMAP (i) =
σ∗
2(π∗
12(i)). For vertices in S∗, the MAP estimator depends on whether a, b is larger:
1. If a > b, then the MAP estimator assigns the label +1 to the vertices corresponding to the largest
|S∗∩V +
1 | values in the collection {maj(i)}i∈S∗and assigns the label -1 to the remaining vertices
in S∗.
2. If a < b, then the MAP estimator assigns the label +1 to the vertices corresponding to the smallest
|S∗∩V +
1 | values in the collection {maj(i)}i∈S∗and assigns the label -1 to the remaining vertices
in S∗.

Then the following corollary prove the potential failure of the MAP estimator.
Corollary G.3. If a > b, there exists i ∈S∗∩V +
1 , j ∈S∗∩V −such that maj(i) < maj(j), then
the MAP estimator fails. Similarly, if a < b, there exists i ∈S∗∩V +
1 , j ∈S∗∩V −such that
maj(i) > maj(j), then the MAP estimator fails.

Proof. Suppose a > b. If the MAP estimator classifies i as +1 correctly. By Theorem 5 the MAP
estimator classifies j as +1 which is wrong. The argument for the case a < b is similar.

G.3
The analysis of the failure of MAP estimator

Definition G.4. The event Gδ holds if and only if

n1−(2s2−s3)Tc(a,b)−δ ≤|R∗∩V +
1 |, |R∗∩V −
1 |, | ¯R∗∩V +
1 |, | ¯R∗∩V +
1 | ≤n1−(2s2−s3)Tc(a,b)+δ.

Lemma G.5. For any fixed δ > 0, P(Gδ) = 1 −o(1).

The proof of this lemma is straightforward but tedious and we defer it to Section G.6.

Now define the variable Wi:

Wi :=
1(i ∈S∗, maj(i) < 0),
i ∈R∗∩V +
1 ,
1(i ∈S∗, maj(i) > 0),
i ∈R∗∩V −
1 .
(G.6)

28


---Page Break---
Denote I be the sigma algebra induced by the random variables

D = max(B, C), π∗
12, σ∗
1, R∗, {Eabc : a, b, c ∈{0, 1}}.

Note that ¯R∗, ¯R∗∩V +
1 , ¯R∗∩V −
1 are I measurable.

Now I’d like to show that P

i∈R∗∩V +
1 Wi > 0, P

i∈R∗∩V −
1 Wi > 0 with high probability, then it
follows that ∃i ∈S∗∩V +
1 , j ∈S∗∩V −
1 such that maj(i) < 0 < maj(j). By Corollary G.3, the
MAP estimator fails. Use the first and second method to analyze P

i∈R∗∩V +
1 Wi, P

i∈R∗∩V −
1 Wi:

Lemma G.7. Fix δ > 0 and denote θ := 1 −(2s2 −s3)Tc(a, b) −s(1 −s)2D+(a, b). We have

E[
X

i∈R∗∩V +
1

Wi|I]1(F ∩Gδ) ≥(1 −n−(2s2−s3)Tc(a,b)+2δ)nθ−δ−o(1)1(F ∩Gδ)

and
E[
X

i∈R∗∩V −
1

Wi|I]1(F ∩Gδ) ≥(1 −n−(2s2−s3)Tc(a,b)+2δ)nθ−δ−o(1)1(F ∩Gδ).

Lemma G.8. Fix δ > 0 and denote θ := 1 −(2s2 −s3)Tc(a, b) −s(1 −s)2D+(a, b)

Var(
X

i∈R∗∩V +
1

Wi|I)1(F ∩Gδ) ≤n2θ−3δ1(F ∩Gδ)

and
Var(
X

i∈R∗∩V −
1

Wi|I)1(F ∩Gδ) ≤n2θ−3δ1(F ∩Gδ).

The proofs of these two lemmas are deferred to Section G.7. Using the lemmas above we can now
prove Theorem 5.

Proof of Theorem 2 when K = 3. Firstly, show that P

i∈R∗∩V +
1 Wi > 0 with high probability. Use
the second moment method, we obtain

P(
X

i∈R∗∩V +
1

Wi > 0|I) ≥
E[P

i∈R∗∩V +
1 Wi|I]2

E[(P

i∈R∗∩V +
1 Wi)2|I]

=
E[P

i∈R∗∩V +
1 Wi|I]2

E[P

i∈R∗∩V +
1 Wi|I]2 + Var(P

i∈R∗∩V +
1 Wi|I)

≥1 −
Var(P

i∈R∗∩V +
1 Wi|I)

E[P

i∈R∗∩V +
1 Wi|I]2 .

Hence for unconditional probability, let δ small enough, δ < min((2s2 −s3)Tc(a, b)/8, θ/4), then

P(
X

i∈R∗∩V +
1

Wi > 0) ≥P({
X

i∈R∗∩V +
1

Wi > 0} ∩F ∩Gδ) = E[P(
X

i∈R∗∩V +
1

Wi > 0|I)1(F ∩Gδ)]

≥E

" 

1 −
Var(P

i∈R∗∩V +
1 Wi|I)

E[P

i∈R∗∩V +
1 Wi|I]2

!

1(F ∩Gδ)

#

≥

(1 −o(1))n2θ−3δ−2(θ−δ)+o(1)
P(F ∩Gδ) = 1 −o(1).

The inequality on the last line is by Lemma G.7, G.8.
The equality in the last line is by
Lemma G.5, D.4. The proof for P(P

i∈R∗∩V −
1 Wi > 0) = 1 −o(1) is identical. Hence, in
light of Corollary G.3, the MAP estimator fails with probability 1 −o(1).

29


---Page Break---
G.4
Analysis of Sπ

In this section we introduce the set Aπ and some properties of Sπ.

First, define the set

A(S∗, π∗
12{[n] \ S∗}) := {π ∈Sn : Sπ = S∗, π([n] \ Sπ) = π∗
12([n] \ S∗)}.

For brevity, sometimes write A∗.
Lemma G.9. For any π ∈A∗, Ai,jBπ(i),π(j)Cπ∗
23(π(i)),π∗
23(π(j)) = Ai,jBπ∗
12(i)π∗
12(j)Cπ∗
13(i),π∗
13(j).
Moreover, if i ∈S∗or j ∈S∗,Ai,jBπ(i),π(j)Cπ∗
23(π(i)),π∗
23(π(j)) = Ai,jBπ∗
12(i)π∗
12(j)Cπ∗
13(i),π∗
13(j) =
0,Ai,jCπ∗
23(π(i)),π∗
23(π(j)) = Ai,jCπ∗
13(i),π∗
13(j) = 0,Ai,jBπ(i),π(j) = Ai,jBπ∗
12(i)π∗
12(j) = 0.

Proof. If
i, j
∈
[n] \ S∗,
then
π(i)
=
π∗
12(i)
and
π(j)
=
π∗
12(j),
hence
Ai,jBπ(i),π(j)Cπ∗
23(π(i)),π∗
23(π(j))
= Ai,jBπ∗
12(i)π∗
12(j)Cπ∗
13(i),π∗
13(j).
If i ∈S∗or j
∈S∗,
then by Definition G.2, i
∈
R∗or j
∈
R∗, hence Ai,jBπ(i),π(j)Cπ∗
23(π(i)),π∗
23(π(j))
=
Ai,jBπ∗
12(i)π∗
12(j)Cπ∗
13(i),π∗
13(j)
=
0, Ai,jCπ∗
23(π(i)),π∗
23(π(j))
=
Ai,jCπ∗
13(i),π∗
13(j)
=
0, and
Ai,jBπ(i),π(j) = Ai,jBπ∗
12(i)π∗
12(j) = 0.

Definition G.10. Let ρ be a permutation of S∗The permutation Pπ,ρ is given by

Pπ,ρ :=
π(i)
i ∈[n] \ S∗,
π(ρ(i))
i ∈S∗.

Lemma G.11. Let ρ be a permutation of S∗. Then Pπ,ρ ∈A∗.

Proof. The proof is identical to [22, Lemma 8.11]]. We only need to change B to B′ = max(B, C)
in the argument.

A useful corollary of Lemma G.11 is that the elements of A∗can be described by permutation of S.
Corollary G.12. We have the following representation

A∗= {Pπ∗,ρ : ρ is a permutation of S∗}.

G.5
Deriving the MAP estimator, proof of Theorem 5

G.5.1
The posterior distribution of π∗
12

First we define

µ+(π)abc :=
X

(π(i),π(j))∈E+(σ∗
2)
1((Ai,j, Bπ(i),π(j), Cπ∗
23(π(i)),π∗
23(π(j))) = (a, b, c)), a, b, c ∈{0, 1},

µ−(π)abc :=
X

(π(i),π(j))∈E−(σ∗
2)
1((Ai,j, Bπ(i),π(j), Cπ∗
23(π(i)),π∗
23(π(j))) = (a, b, c)), a, b, c ∈{0, 1},

ν+(π) :=
X

(π(i),π(j))∈E+(σ∗
2)
Ai,j,

ν−(π) :=
X

(π(i),π(j))∈E−(σ∗
2)
Ai,j.

With these definitions, we can derive an exact expression for the posterior distribution of π∗
12 given
A, B, C, σ∗
2.
Lemma G.13. Let π ∈Sn. There’s a constant D1 = D1(A, B, C, σ∗
2, π∗
23) such that

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23)

=D1

p111p000

p011p100

µ+(π)111 p100

p000

ν+(π) p000p110

p100p010

µ+(π)110 p000p101

p001p100

µ+(π)101

×
q111q000

q011q100

µ−(π)111 q100

q000

ν−(π) q000q110

q100q010

µ−(π)110 q000q101

q001q100

µ−(π)101
.

30


---Page Break---
Proof. The proof is adapted from [22, Lemma 8.13]. By Bayes Rule,

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23) = P(A, B, C|σ∗
2, π∗
12 = π, π∗
23)P(π∗
12 = π|σ∗
2, π∗
23)
P(A, B, C|σ∗
2, π∗
23)
.

In the construction of the multiple Correlated SBM, the permutation π∗
12 is chosen independently
of everything else, including the community labeling σ∗
2 and the permutation π∗
23 . Hence we can
rewrite

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23) = d1(A, B, C, σ∗
2, π∗
23)P(A, B, C|σ∗
2, π∗
23, π∗
12 = π),

where d1(A, B, C, σ∗
2) = (n!P(A, B, C|σ∗
2, π∗
23))−1. Look at P(A, B, C|σ∗
2, π∗= π, π∗
23), note
that the edge formation process in G1, G2 and G3 is mutually independent across all vertex pairs
given σ∗
2, π∗
12, π∗
23. Hence we have

P(A, B, C|σ∗
2, π∗= π, π∗
23) =
Y

ijk∈{0,1}
pµ+(π)ijk
ijk
qµ−(π)ijk
ijk
.

In
particular,
the
sums
P

(π(i),π(j))∈E+(σ∗
2) Bπ(i),π(j)Cπ∗
23π(i),π∗
23π(j),
P

(π(i),π(j))∈E+(σ∗
2) Bπ(i),π(j),
and
P

(π(i),π(j))∈E+(σ∗
2) Cπ∗
23π(i),π∗
23π(j), |E+(σ∗
2)|
are
measurable
with
respect
to
B, C, σ∗
2, π∗
23.
Hence
we
do
not
care
the
relevant
value
and
use
Λ
to
represent.
Now,
for
simple
notations
we
write
P ABC
to
represent
P

(π(i),π(j))∈E+(σ∗
2) Ai,jBπ(i),π(j)Cπ∗
23(π(i)),π∗
23(π(j)),
P AB
to
represent
P

(π(i),π(j))∈E+(σ∗
2) Ai,jBπ(i),π(j),
and
P AC
to
represent
P

(π(i),π(j))∈E+(σ∗
2) Ai,jCπ∗
23(π(i)),π∗
23(π(j)). Then, we can write

µ+(π)011 =
X
(1 −A)BC =
X
BC −µ+(π)111 = Λ −µ+(π)111,

µ+(π)010 =
X
(1 −A)B(1 −C) = Λ −µ+(π)110,

µ+(π)001 =
X
(1 −A)(1 −B)C = Λ −µ+(π)101,

µ+(π)000 =
X
(1 −A)(1 −B)(1 −C) = Λ −µ+(π)100 + µ+(π)101 + µ+(π)110 + µ+(π)111,

µ+(π)100 =
X
A(1 −B)(1 −C) = Λ −µ+(π)111 −µ+(π)101 −µ+(π)110 + ν+(π).

Hence we can write

Y

ijk∈{0,1}
pµ+(π)ijk
ijk
= d+
2

p111p000

p011p100

µ+(π)111 p100

p000

ν+(π) p000p110

p100p010

µ+(π)110 p000p101

p001p100

µ+(π)101
.

Here d+
2 is some constant given the information B, C, σ∗
2, π∗
23 Replicating the arguments for
µ−(π)abc, we have that

Y

ijk∈{0,1}
qµ−(π)ijk
ijk
= d−
2

q111q000

q011q100

µ−(π)111 q100

q000

ν−(π) q000q110

q100q010

µ−(π)110 q000q101

q001q100

µ−(π)101
.

Combining the two equations we prove the statement of lemma with D1 = d1d+
2 d−
2 .

Lemma G.14. There is a constant D2 = D2(A, B, C, σ∗
2, S∗, π∗
23, π∗
12{[n] \ S∗}) such that

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}) = D2(
rp100q000

p000q100
)ν+π−ν−(π)1(π ∈A∗).

Proof. By Bayes Rule we have that

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗})

=P(π∗
12 = π|A, B, C, σ∗
2, π∗
23)P(S∗, π∗
12{[n] \ S∗}|π∗
12 = π, A, B, C, σ∗
2, π∗
23)
P(S∗, π∗
12{[n] \ S∗}|A, B, C, σ∗
2, π∗
23)

=
P(π∗
12 = π|A, B, C, σ∗
2, π∗
23)
P(S∗, π∗
12{[n] \ S∗}|A, B, C, σ∗
2, π∗
23)1(π ∈A∗).

31


---Page Break---
The probability in the denominator is a function of A, B, C, S∗, π∗{[n] \ S∗}, π∗
23, σ∗
2. Furthermore,
by Lemma G.9, µ+(π)111, µ+(π)110, µ+(π)101, µ−(π)111, µ−(π)110, µ−(π)101 are constant over
π ∈A∗. By Lemma G.13, we can write

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}) = d2

p100

p000

ν+(π) q100

q000

ν−(π)
1(π ∈A∗),

where

d2 =
D1
P(S∗, π∗
12{[n] \ S∗}|A, B, C, σ∗
2, π∗
23)

q111q000

q011q100

µ−(π∗
12)111 q000q110

q100q010

µ−(π∗
12)110

×
q000q101

q001q100

µ−(π∗
12)101 p111p000

p011p100

µ+(π)111 p000p110

p100p010

µ+(π)110 p000p101

p001p100

µ+(π)101
.

D1 is the same constant in Lemma G.13, d2 is a constant given A, B, C, σ∗
2, S∗, π∗
23, π∗
12{[n] \ S∗}.
To further simplifies the posterior distribution,

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}) = d2

p100

p000

ν+(π) q100

q000

ν−(π)
1(π ∈A∗)

= d2(
rp100q100

p000q100
)ν−(π)+ν+(π)(
rp100q100

p000q100
)ν+(π)−ν−(π)1(π ∈A∗).

Note that ν−(π) + ν+(π) = P

(i,j)∈(
[n]
2 ) Ai,j that only depends on A. The results follows then

P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}) = D2(
rp100q100

p000q100
)ν+(π)−ν−(π)1(π ∈A∗),

where D2 = d2(
q

p100q100
p000q100 )

P

(i,j)∈([n]
2 ) Ai,j.

G.5.2
The posterior distribution of σ∗

Now we can study the posterior distribution of the community labeling σ∗. For a community partition
X = (X+, X−) of [n] in G1, define the set

B(X) := {π ∈A∗: π(X+) = V +
2 , π(X−) = V −
2 }.

In particular, if σX denotes the community memberships associated with X, the following must hold:

• σX(i) = σ∗
2(π∗
12(i)) for i ∈[n] \ S∗;

• |S∗∩X+| = |S∗∩V +
1 |, |S∗∩X−| = |S∗∩V −
1 |.

The first condition must hold since we know the true vertex correspondence and the true community
labels outside of the S∗. The second condition must hold since the number of vertices of each
community in S∗can be deduced by examining the community labels of π∗(S∗) with respect to σ∗
2.
Lemma G.15. If |B(X)| is not empty, then |B(X)| = |S∗∩X−|!|S∗∩X+|!.

Proof. The proof is almost identical to [22, Lemma 8.15]. Suppose that π0, π1 ∈B(X), by
Corollary G.12, there exists ρ such that π1 = Pπ0,ρ. Claim: if i ∈S∗∩X+, ρ(i) ∈S∗∩X+, if
i ∈S∗∩X−, ρ(i) ∈S∗∩X−. If ∃i ∈S∗∩X+, ρ(i) ∈S∗∩X−, then σ∗
2(π0(i)) = 1, σ∗
2(π1(i)) =
σ∗
2(π0(ρ(i))) = −1. This violates the definition of B(X). The claim is proved. Hence we can
decomposition ρ into two disjoint permutations ρ+, ρ−.ρ+ is a permutation of S∗∩X+ while ρ−
is a permutation of S∗∩X−. Hence |B(X)| = (# of choices of ρ+) × (# of choices of ρ−) =
|S∗∩X+|!|S∗∩X−|! = |S∗∩V +
1 |!|S∗∩V −|!.

Then we look at ν+(π) −ν−(π).
Lemma G.16. For all π ∈B(X), we have that ν+(π) −ν−(π) = D3 + P

i∈S∗maj(i)σX(i) where
D3 is a constant depending on A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗} but not on X.

32


---Page Break---
Proof. Note that σX(i)σX(j) = 1 if (π(i), π(j)) ∈E+(σ∗
2), σX(i)σX(j) = −1 if (π(i), π(j)) ∈
E−(σ∗
2). We have

ν+(π) −ν−(π) =
X

(i,j)∈(
[n]
2 )
σX(i)σX(j)Ai,j

=
X

(i,j)∈{[n]\S∗}
σX(i)σX(j)Ai,j +
X

i,j∈S∗
σX(i)σX(j)Ai,j +
X

i∈S∗,j∈{[n]\S∗}
σX(i)σX(j)Ai,j.

We should note that if (i, j) ∈{[n] \ S∗}, then σX(i)σX(j)Ai,j = σ∗
2(π12(i))σ∗
2(π12(j))Ai,j =
σ∗
2(π∗
12(i))σ∗
2(π∗
12(j))Ai,j. Denote

D3 :=
X

(i,j)∈{[n]\S∗}
σX(i)σX(j)Ai,j.

Clearly D3 depends only on A, σ∗
2, S∗, π∗
12{[n] \ S∗}. If i, j ∈S∗, Ai,j = 0 by Definition G.2.
X

i∈S∗,j∈{[n]\S∗}
σX(i)σX(j)Ai,j =
X

i∈S∗,j∈{[n]\S∗}
σX(i)σ∗
1(j)Ai,j

=
X

i∈S∗
σX(i)
X

j∈{[n]\S∗}
σ∗
1(j)Ai,j

=
X

i∈S∗
σX(i)maj(i).

The last equality is because if i ∈S∗,maj(i) = P

j∈[n] Ai,jσ∗
1(j) = P

j∈{[n]\S∗} Ai,jσ∗
1(j), since
Ai,j = 0 if i, j ∈S∗. Then the statement follows,

ν+(π) −ν−(π) = D3 +
X

i∈S∗
σX(i)maj(i).

Lemma G.17. If B(X) is nonempty, then

P((V +
1 , V +
2 ) = (X+, X−)|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗})

= D4

rp100q100

p000q000

P

i∈S∗σX(i)maj(i)
,

where D4 is a constant depending on A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}, but not on X.

Proof. We have that

P((V +
1 , V +
2 ) = (X+, X−)|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗})

=
X

π∈B(X)
P(π∗
12 = π|A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗})

=
X

π∈B(X)
D2

rp100q000

p000q100

ν+(π)−ν−(π)

=D2

rp100q000

p000q100

D3
X

π∈B(X)

rp100q000

p000q100

P

i∈S∗σX(i)maj(i)

=D2

rp100q000

p000q100

D3
|B(X)|
rp100q000

p000q100

P

i∈S∗σX(i)maj(i)

=D4

rp100q000

p000q100

P

i∈S∗σX(i)maj(i)
,

where D4 = D2
q

p100q000
p000q100

D3
|S∗∩V +
1 ||S∗∩V −
1 |. The statement follows.

33


---Page Break---
Proof of Theorem 5. Given A, B, C, σ∗
2, π∗
23, S∗, π∗
12{[n] \ S∗}, it’s obivious that bσMAP (i) =
σ∗
2(π∗
12(i)) for i ∈[n] \ S∗. For vertices in S∗, note that

p100q000
p000q100
= s(1 −s)2p(1 −(1 −(1 −s)3)q)

s(1 −s)2q(1 −(1 −(1 −s)3)q) = (1 + o(1))p

q = (1 + o(1))a

b .

Thus, by Lemma G.17, if a > b, the MAP estimator maximizes P

i∈S∗σX(i)maj(i) while the MAP
estimator minimizes P

i∈S∗σX(i)maj(i) if a < b. Suppose a > b, while satisfying the condition
|S∗∩X+| = |S∗∩V +|, |S∗∩X−| = |S∗∩V −|, the maximum of P

i∈S∗σX(i)maj(i) is obtained
by seting σX(i) = +1 to the vertices i ∈S∗corresponding to the largest |S∗∩V +
1 | values in the
collection {maj(i)}i∈S∗and assigns the label -1 to the remaining vertices in S∗. The proof is the
same suppose a < b. Then Theorem 5 follows.

G.6
Proof of Lemma G.5

Denote Ei as the event that i is a singleton in G1 ∧π∗
12 H, in other words that i ∈R∗. We
assume that the communities are approximately balanced. More precisely, we assume that the
event G = {n/2 −n3/4 ≤|V +|, |V −| ≤n/2 + n3/4} holds. By Lemma D.4, we have that
P(G) = 1 −o(1).

Conditioning on σ∗
1, if i ∈V +
1 , then we have

P(Ei|σ∗
1)1(G) = (1 −s(1 −(1 −s)2)a log n/n)|V +
1 |−1(1 −s(1 −(1 −s)2)b log n/n)|V −
1 |1(G)

≤exp(−s(2s −s2) log n/n(a(|V +
1 | −1) + b|V −
1 |))1(G)

= exp(−(1 −o(1))(2s2 −s3)Tc(a, b) log n)1(G) = n−(2s2−s3)Tc(a,b)+o(1)1(G).

The first inequality uses the fact 1 −x ≤e−x. Hence

E[|R∗∩V +
1 ||σ∗
1]1(G) =
X

i∈V +
P(Ei|σ∗
1)1(G) ≤|V +
1 |n−(2s2−s3)Tc(a,b)+o(1)1(G)

≤n1−(2s2−s3)Tc(a,b)+o(1).

By Markov’s inequality,

P(|R∗∩V +
1 | ≥n1−(2s2−s3)Tc(a,b)+δ|σ∗
1)1(G) ≤n−δ+o(1) = o(1).

Hence

P(|R∗∩V +
1 | ≥n1−(2s2−s3)Tc(a,b)+δ)

≤P(Gc) + E[P(|R∗∩V +
1 | ≥n1−(2s2−s3)Tc(a,b)+δ|σ∗
1)1(G)] = o(1).

Now we derive a lower bound for |R∗∩V +
1 |. For ϵ = ϵn sufficiently small:

(log(P(Ei|σ∗
1)))1(G) = ((|V +
1 | −1) log(1 −s(1 −(1 −s)2)a log n/n)

+ |V −
1 | log(1 −s(1 −(1 −s)2)b log n/n))1(G)

≥−(1 + ϵ)(2s2 −s3) log n/n(a|V +
1 | + b|V −
1 |)1(G)

= (1 −o(1))(1 + ϵ)(2s2 −s3)Tc(a, b)(log n)1(G).

The first inequality uses that log(1−x) ≥−(1+ϵ)x provided 0 < x < ϵ/(1+ϵ). Setting ϵ = n−0.5,
we have that

E[|R∗∩V +
1 ||σ∗
1]1(G) =
X

i∈V +
P(Ei|σ∗
1)1(G) ≥n1−(2s2−s3)Tc(a,b)−o(1)1(G).

Then we bound the variance of |R∗∩V +
1 |. For i, j ∈V +
1 , i ̸= j:

Cov(1(Ei), 1(Ej)|σ∗
1) = P(EiEj|σ∗
1) −P(Ei|σ∗
1)2

= (1−s(2s−s2)a log n/n)2|V +
1 |−3(1−s(2s−s2)b log n/n)2|V −
1 |(1−(1−s(2s−s2)a log n/n)).

34


---Page Break---
On event G, 2|V +
1 |−3 = (1+o(1))n, 2|V −
1 | = (1+o(1))n. Hence using the equality 1 −x ≤e−x,
we have that

Cov(1(Ei), 1(Ej)|σ∗
1)1(G) ≤s(2s −s2)a log n/n exp(−(1 −o(1))s(2s −s2)(a + b) log n)1(G)

= n−1−(2s2−s3)(a+b)+o(1)1(G).

Then

Var(|R∗∩V +
1 ||σ∗
1)1(G)

= Var(
X

i∈V +
1

1(Ei)|σ∗
1)1(G) =
X

i,j∈V +
1

Cov(1(Ei), 1(Ej)|σ∗
1)1(G)

≤
X

i∈V +
1

P(Ei|σ∗
1)1(G) +
X

i̸=j∈V +
1

Cov(1(Ei), 1(Ej)|σ∗
1)1(G)

≤(n1−(2s2−s3)(a+b)+o(1) + n1−(2s2−s3)(a+b)/2+o(1))1(G) = n1−(2s2−s3)(a+b)/2+o(1)1(G).

By the Paley-Zygmund inequality,

P(|R∗∩V +
1 | ≥n1−(2s2−s3)Tc(a,b)−δ|σ∗
1)1(G)

≥(1 −n−δ+o(1))2 E[|R∗∩V +
1 ||σ∗
1]2

E[|R∗∩V +
1 |2|σ∗
1]1(G)

≥(1 −n−δ+o(1))2(1 −Var[|R∗∩V +
1 ||σ∗
1]
E[|R∗∩V +
1 ||σ∗
1]2 )1(G)

≥(1 −n−δ+o(1))2(1 −n−(1−(2s2−s3)Tc(a,b))+o(1))1(G) = (1 −o(1))1(G).

Together with P(G) = 1 −o(1), we have

P(n1−(2s2−s3)Tc(a,b)−δ ≤|R∗∩V +
1 | ≤n1−(2s2−s3)Tc(a,b)+δ) = 1 −o(1).

The proof for |R∗∩V −
1 | is the same. Now we study | ¯R∗∩V +
1 |. Since R∗⊂¯R∗,we have the lower
bound P(| ¯R∗∩V +
1 | ≥|R∗∩V +
1 | ≥n1−(2s2−s3)Tc(a,b)−δ) = 1 −o(1). For the upper bound,

| ¯R∗∩V +
1 | ≤| ¯R∗| ≤
X

i∈R∗
(1 + |NH(π∗
12(i))|) ≤|R∗|(1 + max
j∈[n] NH(j)).

By Lemma D.6, since H ∼SBM(n, (1 −(1 −s)2)a log n/n, (1 −(1 −s)2)b log n/n), we have
that maxj∈[n] NH(j) ≤100 max a, b(1 −(1 −s)2) log n with probability 1 −o(1). Hence we have
that with high probability

| ¯R∗∩V +
1 | ≤(1 + 100 max a, b(1 −(1 −s)2) log n)|R∗|

≤2(1 + 100 max a, b(1 −(1 −s)2) log n)n1−(2s2−s3)Tc(a,b)+δ

≤n1−(2s2−s3)Tc(a,b)+δ.

G.7
Proof of Lemma G.7, G.8

Firstly, we use some useful conditional independence properties given the sigma algebra I.
Lemma G.18. Let i ∈R∗, conditioned on I, {Aij : {i, j} ∈E100} is a collection of mutually
independent random variables where

Aij ∼
Ber(a log n/n)
σ∗
1(i) = σ∗
1(j),
Ber(b log n/n)
σ∗
1(i) = −σ∗
1(j).

The random variables 1(i ∈S∗) and P

j∈[n]\ ¯
R∗Ai,jσ∗
1(j) are conditionally independent given I.

Proof. Note that the sets R∗, ¯R∗only depend on π∗
12, H, G1 ∧π∗
12 H . G1 \π∗
12 H is comprised of
edges in E100, the graph H is comprised of edges in ∪j+k>0Eijk, the graph G1 ∧π∗
12 H is comprised
of edges in ∪j+k>0E1jk. Thus by lemma D.2, G1 \π∗
12 H is conditionally independent of R∗, ¯R∗

35


---Page Break---
given π∗, σ∗
1 and the partition E. In particular, {Ai,j : {i, j} ∈E100} is conditionally independent of
I.

Note that 1(i ∈S∗) = 1(i ∈R∗)1(Ai,j = 0, ∀j ∈¯R∗). Hence 1(i ∈S∗) is measurable with respect
to the sigma algebra generated by I and the collection C1 := {Ai,j : j ∈¯R∗, {i, j} ∈E100}. On the
other hand, since ¯R∗is I−measurable, P

j∈[n]\ ¯
R∗Ai,jσ∗
1(j) is measurable with respect to the sigma
algebra generated by I and the collection C2 := {Ai,j : j ∈[n] \ ¯R∗, {i, j} ∈E100}. C1 ∩C2 = ∅,
the independence of Ai,j implies that the two random variables are conditionally independent given
I.

Lemma G.19. For any δ > 0, i ∈R∗,it holds for sufficiently large n that

P(i ∈S∗|I)1(Gδ) ≥(1 −n−(2s2−s3)Tc(a,b)+2δ)1(Gδ).

Proof. For i ∈R∗, define the following random sets that are I−measurable:

C+(i) : = {j ∈¯R∗: {i, j} ∈E100 ∩E+(σ∗
1)},

C−(i) : = {j ∈¯R∗: {i, j} ∈E100 ∩E−(σ∗
1)}.

Note that i ∈S∗if and only if i ∈R∗and Ai,j = 0, ∀j ∈C+(i) ∪C−(i) conditioned on I. Hence
by Lemma G.18:

P(i ∈S∗|I) = P(Ai,j = 0, ∀j ∈C+(i) ∪C−(i)|I) = (1 −a log n/n)|C+(i)|(1 −b log n/n)|C(i)|

≥(1 −a|C+(i)| log n/n)(1 −b|C−(i)| log n/n)

≥1 −(a|C+(i)| + b|C−(i)|) log n/n.

The first inequality is because of Bernoulli’s inequality. Note that on event Gδ, |C+(i)|, |C−(i)| ≤
| ¯R∗| ≤2n1−(2s2−s3)Tc(a,b)+δ. Thus

P(i ∈S∗|I)1(Gδ) ≥(1 −2(a + b)n−(2s2−s3)Tc(a,b)+δ log n)1(Gδ).

Since 2(a + b) log n ≤nδ, the lemma follows.

Now, define the random variable

Xi :=

(
P(P

j∈[n]\ ¯
R∗Ai,jσ∗
1(j) < 0|I)
i ∈R∗∩V +
1 ,
P(P

j∈[n]\ ¯
R∗Ai,jσ∗
1(j) > 0|I)
i ∈R∗∩V −
1 .

Then we will study Xi on the event F ∩Gδ.
Lemma G.20. For i ∈R∗,

Xi1(F ∩Gδ) = n−s(1−s)2D+(a,b)+o(1)1(F ∩Gδ).

Proof. Suppose i ∈R∗∩V +
1 . Note that ¯R∗, σ∗
1 are I−measurable. By Lemma G.18:
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j)
d= Y −Z

Y ∼Bin(|{j ∈[n] \ ¯R∗: {i, j} ∈E100 ∩E+(σ∗
1)|, a log n/n),

Z ∼Bin(|{j ∈[n] \ ¯R∗: {i, j} ∈E100 ∩E−(σ∗
1)|, b log n/n).
Where Y, Z are independent. For brevity, suppose Y ∼Bin(y, p), Z ∼Bin(z, q). On the event
F ∩Gδ, the upper bound for y:

y ≤|{j ∈[n] : {i, j} ∈E100 ∩E+(σ∗
1)| ≤s(1 −s)2(n/2 + 2n3/4) = (1 + o(1))s(1 −s)2n/2.

The lower bound for y:

y ≥|{j ∈[n] : {i, j} ∈E100 ∩E+(σ∗
1)| −| ¯R∗|

≥s(1 −s)2(n/2 −2n3/4) −n1−(2s2−s3)Tc(a,b)+δ = (1 −o(1))s(1 −s)2n/2.

36


---Page Break---
Same calculation for z and we can derive:

(1 −o(1))s(1 −s)2n/2 ≤z ≤(1 + o(1))s(1 −s)2n/2.

We can rewrite p, q as p = (1+o(1)as(1−s)2 log(s(1−s)2n)/(s(1−s)2n)), q = (1+o(1)bs(1−
s)2 log(s(1 −s)2n)/(s(1 −s)2n)), hence by Lemma D.1, we have

P(
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j) < 0|I)1(F ∩Gδ) = P(Y −Z < 0|I)1(F ∩Gδ) ≤n−s(1−s)2D+(a,b)+o(1).

The same argument works for i ∈R∗∩V −
1 .

Proof of Lemma G.7. For i ∈R∗∩V +
1 , we have that

E[Wi|I]1(F ∩Gδ) = P(i ∈S∗,
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j) < 0|I)1(F ∩Gδ)

= P(i ∈S∗|I)P(
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j) < 0|I)1(F ∩Gδ).

The first equality holds because Ai,j = 0 if i ∈S∗, j ∈¯R∗. The second equality exits because the
conditional independence in Lemma G.18. By Lemm G.19 and G.20, we have that

E[Wi|I]1(F ∩Gδ) ≥(1 −n−(2s2−s3)Tc(a,b)+2δ)Xi1(F ∩Gδ)

≥(1 −n−(2s2−s3)Tc(a,b)+2δ)n−s(1−s)2D+(a,b)+o(1)1(F ∩Gδ).

Summing over i ∈R∗∩V +, we have that

E[
X

i∈R∗∩V +
Wi|I]1(F ∩Gδ)

≥|R∗∩V +|(1 −n−(2s2−s3)Tc(a,b)+2δ)n−s(1−s)2D+(a,b)+o(1)1(F ∩Gδ)

≥(1 −n−(2s2−s3)Tc(a,b)+2δ)n1−(2s2−s3)Tc(a,b)−s(1−s)2D+(a,b)−δ+o(1)1(F ∩Gδ).

The second inequality is because on Gδ, |R∗∩V +| ≥n1−(2s2−s3)Tc(a,b)−δ. The proof of lower
bound for i ∈R∗∩V −
1 is the same.

Proof of Lemma G.8. Suppose i ̸= j ∈R∗∩V +
1 , we have

E[WiWj|I] = P(i, j ∈S∗,
X

k∈[n]\ ¯
R∗
Ai,kσ∗
1(k) < 0,
X

k∈[n]\ ¯
R∗
Aj,kσ∗
1(k) < 0|I)

≤P(
X

k∈[n]\ ¯
R∗
Ai,kσ∗
1(k) < 0,
X

k∈[n]\ ¯
R∗
Aj,kσ∗
1(k) < 0|I)

= P(
X

k∈[n]\ ¯
R∗
Ai,kσ∗
1(k) < 0|I)P(
X

k∈[n]\ ¯
R∗
Aj,kσ∗
1(k) < 0|I) = XiXj.

The second inequality is because by Lemma G.18, {Ai,k}k∈[n]\ ¯
R∗} and {Aj,k}k∈[n]\ ¯
R∗} are condi-
tionally independent given I. Consider the case i = j,

E[W 2
i |I] = P(i ∈S∗,
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j) < 0|I) ≤Xi.

Summing over i ∈R∗∩V +
1 :

E[(
X

i∈R∗∩V +
Wi)2|I] =
X

i∈R∗∩V +
E[Wi|I] +
X

i̸=j∈R∗∩V +
E[WiWj|I]

≤
X

i∈R∗∩V +
Xi +
X

i̸=j∈R∗∩V +
XiXj ≤
X

i∈R∗∩V +
Xi + (
X

i∈R∗∩V +
Xi)2.

37


---Page Break---
Note that
E[
X

i∈R∗∩V +
1

Wi|I]21(F ∩Gδ) = (
X

i∈R∗∩V +
1

P(i ∈S∗,
X

j∈[n]\ ¯
R∗
Ai,jσ∗
1(j) < 0|I))21(F ∩Gδ)

≥(1 −n−(2s2−s3)Tc(a,b)+2δ)2(
X

i∈R∗∩V +
1

Xi)21(F ∩Gδ).

Hence we have

Var(
X

i∈R∗∩V +
1

Wi|I)1(F ∩Gδ)

=



E[(
X

i∈R∗∩V +
1

Wi)2|I] −E[
X

i∈R∗∩V +
1

Wi|I]2



1(F ∩Gδ)

≤




X

i∈R∗∩V +
1

Xi + (
X

i∈R∗∩V +
1

Xi)2 
1 −(1 −n−(2s2−s3)Tc(a,b)+2δ)2


1(F ∩Gδ)

≤




X

i∈R∗∩V +
1

Xi + 2(
X

i∈R∗∩V +
1

Xi)2n−(2s2−s3)Tc(a,b)+2δ



1(F ∩Gδ)

≤

|R∗∩V +
1 |n−s(1−s)2D+(a,b)+o(1) + 2

|R∗∩V +
1 |n−s(1−s)2D+(a,b)+o(1)2
n−(2s2−s3)Tc(a,b)+2δ

1(F ∩Gδ)

≤(nθ+δ+o(1) + 2n2θ+2δ−(2s2−s3)Tc(a,b)+2δ+o(1))1(F ∩Gδ).

The second inequality uses the fact that 1 −(1 −n−x)2 ≤2n−x. The third inequality exists by
Lemma G.5, G.20. Now, for further simplying the upper bound, note that if δ is small enough
(specifically, δ < min((2s2 −s3)Tc(a, b)/8, θ/4)), then θ + δ + o(1) < 2θ −3δ, and 2θ + 4δ −
(2s2 −s3)Tc(a, b) + o(1) < 2θ −3δ. Hence

Var(
X

i∈R∗∩V +
1

Wi|I)1(F ∩Gδ) ≤n2θ−3δ1(F ∩Gδ).

The proof for Var(P

i∈R∗∩V −
1 Wi|I)1(F ∩Gδ) is the same.

H
Proof of Theorem 1 for K graphs

H.1
Categorization of vertices

We start with the definition of categorizing vertices as “good” and “bad” vertices. To begin with, we
define a metagraph for each vertex. Then we categorize each vertex as “good” and “bad” according
to the connectivity of the metagraph for the vertex.

Definition H.1. Given (G1, . . . ., GK) ∼CSBM(n, a log n

n , b log n

n , s), and
 K
2

partial k-core match-
ings bµ := {bµij : i ̸= j ∈[K]}. For any vertex v, define the following graph matching metagraph
for the vertex v, denoted it as MGv. E(MGv) denotes the edge set in the graph MGv. There are
K nodes in MGv, where node i represents the graph Gi, and an edge exists between (i, j), that
is, (i, j) ∈E(MGv) if and only if vertex v can be matched in the partial matching bµij between Gi
and Gj.

Note that MGv is an undirected graph. This is because of an inherent symmetry in the definition of a
k-core matching, which looks at the k-core of the intersection graph of the two matched graphs. Thus
for k-core matchings, a vertex is matched by bµij if and only if it is matched by bµji. This property
does not necessarily hold for other graph matching algorithms.

Definition H.2. Given (G1, . . . , GK) ∼CSBM(n, a log n

n , b log n

n , s), and
 K
2

partial k-core match-
ings bµ := {bµij : i ̸= j ∈[K]}, we define a vertex v to be “good” if and only if MGv is connected.
Conversely, a vertex v is “bad” if and only if MGv is disconnected.

38


---Page Break---
(a) MGv for a “good” vertex v: MGv is con-
nected.

(b) MGv for a “bad” vertex v: MGv is discon-
nected.

Figure 4: Schematic showing the meta graph MGv when K = 5.

For a “bad” vertex v, since MGv is disconnected, the metagraph has at least two disjointed compo-
nents. Hence, there must exist two sets Γg(v), Γb(v) satisfying Γg(v) ∩Γb(v) = ∅, Γg(v) ∪Γg(b) =
[K] and for any i ∈Γg(v), j ∈Γb(v), (i, j) is not an edge in MGv. In other words, v cannot be
matched for any matching between the graph Gi, i ∈Γg(v) and the graph Gj, j ∈Γb(v). Heuristi-
cally, the definition implies that “bad” vertices cannot utilize the combined information for all K
graphs.

Otherwise,the “good” vertices can utilize the combined information for all K graphs, as shown in
Lemma H.3.

Lemma H.3. For a “good” vertex v defined in Definition H.2, for any two node i,j (represents two
graphs Gi, Gj), there exists a path i := ℓ0 −ℓ1 −ℓ2 . . . −ℓd := j such that v can be matched for
bµℓmℓm+1, m ∈{0, 1, ..., k −1}. Define bπij(v) := bµℓd−1ℓd ◦bµℓd−2ℓd−1 ◦. . . ◦bµℓ1ℓ2 ◦bµℓ0ℓ1(v).

Proof. For a “good” vertex v, since MGv is connected, for any two nodes i, j ∈[K], there exists a
path ψij(v) := {ℓ0 −ℓ1 −. . . −ℓd−1 −ℓd, ℓ0 = i, ℓd = j, (ℓm, ℓm+1) ∈E(MGv)}. We can use
the path ψij(v) to define the bπij(v)

Remark: Note that such a path is not unique. However, the choice of path does not matter whp. By
Lemma E.5, for k-core paritial matching, bµij = π∗
ij, i, j ∈[K], with high probability. Hence, for two
different paths with endpoints being i, j ∈[K], denoted by ψ1
ij, ψ2
ij, we can define bπ1
ij, bπ2
ij based on
the two paths ψ1
ij, ψ2
ij separately. Then, with high probability bπ1
ij = π∗
ij = bπ2
ij.

By Lemma H.3 and its remark, we can have the union graph of K graphs (G1 ∨G2 ∨G3 ∨. . . ∨GK)
for the “good” vertices, using the matchings bπ := (bπ12, bπ13, . . . , bπ1K) where bπ1K is defined in
Lemma H.3. If there are multiple paths existing, pick the shortest one, and break ties in lexicographic
order.

H.2
Exact community recovery algorithm for K graphs

The key steps of the exact community recovery algorithm for K graphs are essentially the same
as the algorithm for three graphs. Based on a given almost exact community recovery bσ1 and
 K
2


k-core matchings pairwise, we divide the vertices into two categories:“good” and “bad” vertices
according to Definition H.1, H.2. Then refine the community label according to the majority votes
for the “good” and “bad” vertices sequentially, to obtain the exact community recovery label under
the given conditions. The full algorithm for exact community recovery is given in Algorithm 5:

H.3
Analysis of the k-core estimator

Here we quantify the size of the“bad” vertices. Suppose that for vertex v, Γg = {G1, G2, . . . , GL}
and Γb = {GL+1, GL+2, . . . , GK}, here 1 < L < K −1, v cannot be matched for all the matchings
between one graph from Γa and another graph from Γb. Apparently, v cannot be matched for
L(K −L) matchings. Heuristically, as the number of mathings that cannot be matched for vertices
increases, the size of such vertices decreases, since every matching would match (1 −o(1))n vertices
and only a very small fraction of vertices that cannot be matched. The following lemma demonstrated
the claim.

39


---Page Break---
Algorithm 5 Community Recovery for K graphs

Input: K graphs (G1, G2, . . . , GK) on n vertices, k = 13, and ϵ > 0.
Output: A labeling of [n] given by bσ.

1: Apply [35, Algorithm 1] to the graph G1 and parameters (sa, sb, ϵ), obtaining a label c
σ1.

2: Apply Algorithm 1 on input (Gi, Gj, k), obtaining a matching (c
Mij, bµij), i ̸= j ∈{1, 2, .., K}.
3: For “good” vertices v, look at the set Ψ := {bµij, i, j ∈[K] : (i, j) ∈E(MGv)}, by Lemma H.3,
we can define the bπ := (bπ12, bπ13, . . . , bπ1K) to obtain the union graph of K graphs based on
the matchings from Ψ, denote it as (G1 ∨G2 . . . ∨GK)Ψ. Denote M := ∩Mij, where (i, j)
satisfying bµij ∈MGv. Set bσ(v) ∈{−1, 1} according to the neighborhood majority (resp.,
minority) of c
σ1(v) with respect to the graph (G1 ∨G2 . . . ∨GK)Ψ{M} if a > b (resp., a < b).
4: For “bad” vertices v, denote ϕ := {j ∈[K] : (1, j) ∈E(MGv)}. Denote M := ∩i∈[K]M1i, set
bσ(v) ∈{−1, 1} according to the neighborhood majority (resp., minority) of bσ(v) with respect to
the graph G1 \j /∈ϕ Gj(M ∪{v}) if a > b (resp., a < b).
5: Return bσ : [n] →{−1, 1}.

Lemma H.4. Suppose that G1, G2, . . . , GK are independently subsampled with probability s from a
parent graph G ∼SBM(n, a log n/n, b log n/n) for a, b > 0. Let F ∗
ij be the set of vertices outside
the k-core of Gi ∧πij Gj with k = 13. For 1 ≤L ≤K −1 and every δ > 0, with probability
1 −o(1) we have that | ∩1≤i≤L,L+1≤j≤K F ∗
ij| ≤n1−s(1−(1−s)K−1)Tc(a,b)+δ.

Proof. Define Uij to be the set of vertices with degree at most m + k in Gi ∧πij Gj, where
m >
2
(a+b)s2 , as defined in Lemma E.1. Then by Lemma E.1, w.h.p. we have F ∗
ij ⊂Uij and
| ∩1≤i≤L,L+1≤j≤K F ∗
ij| ≤| ∩1≤i≤L,L+1≤j≤K Uij|. Now we look at the expectation:

E [| ∩1≤i≤L,L+1≤j≤K Uij] = nE




L
Y

i=1

K
Y

j=L+1
1v∈Uij



.

Let D denote the degree of vertex v in the graph G1 ∨G3 . . . ∨GL.

E




L
Y

i=1

K
Y

j=L+1
1v∈Uij



= E



E




K
Y

j=L+1

L
Y

i=1
1v∈Uij


D









≤E









(m+k)L
X

i=0

D
i


si(1 −s)D−i





K−L



The first equality is by the tower rule. The second inequality is due to two observations. Firstly,
{QL
i=1 1v∈Uij} ⊆{deg(v) ≤L(m+k) in the graph (G1∨G3 . . .∨GL)∧Gj, L+1 ≤j ≤K}. To
be more detailed, if the degree of v is at most m+k in the graph Gi ∧Gj, 1 ≤i ≤L, then the degree
of v is at most L(m + k) in the graph (G1 ∨G3 . . . ∨GL) ∧Gj. The second observation is, given D,
for j1 ̸= j2, the events {deg(v) ≤L(m + k) in the graph (G1 ∨G3 . . . ∨GL) ∧Gj1}, {deg(v) ≤
L(m + k) in the graph (G1 ∨G3 . . . ∨GL) ∧Gj2} are independent.

Similar to the proof of Lemma E.7, let Xa ∼Bin((1 + o(1))n/2, (1 −(1 −s)L)a log n/n), and

Xb ∼Bin((1 + o(1))n/2, (1 −(1 −s)L)b log n/n). On the event F, D1
d= Xa + Xb, where F is
defined in Definition D.3, Xa, Xb are independent. Note that by Lemma D.4, P(Fc) = o( 1

n2 ). We
have

E





 m+k
X

i=0

D
i


si(1 −s)D−i
!K−L



=

(m+k)L
X

i1,i2,...,iK−L=0
C(i1, i2, . . . , iK−L)E[D
PK−L
j=0
ij(1 −s)(K−L)D].
(H.5)

40


---Page Break---
Here C(i1, i2, . . . , iK−L) is a constant given i1, i2, . . . , iK−L. Now look at E[DN(1 −s)(K−L)D].
In our regime, N ≤(m + k)(K −L)L are constant. Hence:

E[DN(1 −s)(K−L)D1F] = E[(Xa + Xb)N(1 −s)(K−L)Xa(1 −s)(K−L)Xb1F]

=

N
X

t=0
CtE[Xt
a(1 −s)(K−L)Xa1F]E[XN−t
b
(1 −s)(K−L)Xa1F].

Here Ct is constant related to t, the second equality is due to the independence of Xa, Xb. Now look
at E[Xt
a(1 −s)(K−L)Xa1F].

E[Xt
a(1 −s)(K−L)Xa1F] ≤E[Xt
a(1 −s)(K−L)Xa]

=

(1+o(1))n/2
X

ℓ=0
ℓt(1 −s)(K−L)ℓ((1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

=

(log n)3
X

ℓ=0
ℓt((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

+

(1+o(1))n/2
X

ℓ=(log n)3+1
ℓt((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ.

Similarly, we can bound the first part:

(log n)3
X

ℓ=0
ℓt((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

≤(log n)3t
(log n)3
X

ℓ=0
((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

≤(log n)3t
(1+o(1))n/2
X

ℓ=0
((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

=(log n)3t(1 −(1 −(1 −s)K−L)(1 −(1 −s)L)sa log n

n
)(1+o(1))n/2

≤n−(1−(1−s)K−L)(1−(1−s)L)a/2+o(1).
Then we can bound the second part, using the similar arguments in Lemma E.7, we can show that

(1+o(1))n/2
X

ℓ=(log n)3+1
ℓt((1 −s)K−L(1 −(1 −s)L)a log n

n
)ℓ(1 −(1 −(1 −s)L)a log n

n
)(1+o(1))n/2−ℓ

= o(n−(1−(1−s)K−L)(1−(1−s)L)a/2+o(1)).

Hence, by summing up the two parts, E[Xt
a(1 −s)(K−L)Xa] ≤n−(1−(1−s)K−L)(1−(1−s)L)a/2+o(1).
This is also true for Xb. Then, E[DN(1 −s)(K−L)D] ≤n−(1−(1−s)K−L)(1−(1−s)L)Tc(a,b)+o(1).

For f(x) = (1 −s)x + (1 −s)K−x, x ∈[1, K −1], the function f(x) obtains its maximum at
x = 1, K−1. Hence n((1−s)K−L−1)((1−(1−s)L)Tc(a,b)+o(1) ≤n−s(1−(1−s)K−1)Tc(a,b)+o(1). Similar
to Lemma E.7, by Markov inequality, the lemma follows immediately.

Lemma H.6. The size of “bad” vertices defined in Definition H.2 can be upper bounded by
n−s(1−(1−s)K−1)Tc(a,b)+o(1).

Proof. By definition H.2, the meta graph MGv for“bad” vertex v are disconnected. Hence, there
exists at least two components of MGv, where there are no edges between the nodes from two
components.

With all the preparations for “bad” vertices and “good” vertices, we are now ready to prove Theorem 1
for general K graphs.

41


---Page Break---
H.4
Exact recovery for “good” vertices

By Lemma D.1, we can directly deduce that if (1 −(1 −s)K)D+(a, b) > 1 + ϵ| log(a/b)|, then for
all i ∈[n] we have that majG1∨G2...∨GK(i) ≥ϵ log n w.h.p.

Now for a “good”vertex i, there exists a corresponding matching set Ψ that i can be matched for all
the matchings in Ψ and there is an union graph eG := (G1 ∨G2 ∨. . . ∨G3) that can be derived from
the matchings in Ψ. Let M be the set of vertices that can be matched for all matchings in Ψ.
Lemma H.7. Suppose that (1 −(1 −s)3)D+(a, b) > 1 + 2ϵ| log(a/b)|. Then with probability
1 −o(1), all the “good” vertices in M have an ϵ log n majority in eG{M}.

Proof. Denote F ∗
ij the set of vertices outside the 13-core of Gi ∧π∗
ij Gj. In light of Lemma E.5 and
its remark, we can replace µij with π∗
ij, Fij with F ∗
ij ,M with M ∗in Lemma F.2. Where we define

G∗:= G1 ∨π∗
12 G2 ∨. . . ∨π∗
1K GK,
H := G∗{M ∗}.

To bound the neighborhood majority in H, for i ∈M ∗we have:

majH(i) = σ∗(i)
X

j∈NH(i)
σ∗(j) ≤majG∗(i) +
X

bµjℓ/∈Ψ
|NG∗(i) ∩F ∗
jℓ|,

majH(i) = σ∗(i)
X

j∈NH(i)
σ∗(j) ≥majG∗(i) −
X

bµjℓ/∈Ψ
|NG∗(i) ∩F ∗
jℓ|.

To sum up, we have
|majH(i) −majG∗(i)| ≤
X

bµjℓ/∈Ψ
|NG∗(i) ∩F ∗
jℓ|.
(H.8)

Note that majG∗(i) > 2ϵ log n, i ∈[n] with probability 1−o(1), given that (1−(1−s)K)D+(a, b) >
1 + 2ϵ| log(a/b)| by Lemma D.1. Now we would like to prove that the right hand side of (H.8) can
be bounded by ϵ log n.

Note that |NG∗(i) ∩F ∗
jℓ| ≤|NGj∧Gℓ(i) ∩(F ∗
jℓ)| + |NG∗\(Gj∧Gℓ)(i) ∩(F ∗
jℓ)|.

First, look at |NGj∧Gℓ(i) ∩(F ∗
jℓ)|, by Lemma E.4, w.h.p.,

|NGj∧Gℓ(i) ∩F ∗
jℓ| < ϵ log n/2K2.

The remaining thing is to bound |NG∗\(Gj∧Gℓ)(i) ∩(F ∗
jℓ)|. Note that conditioned on π∗, σ∗, E :=
{Ei1i2..iK, i1, i2, . . . , iK ∈{0, 1}}, the graph G∗\ (Gj ∧Gℓ) is independent of F ∗
jℓby Lemma D.2,
since F ∗
jℓdepends only on Gj ∧Gℓ. Thus we can stochastically dominate |NG∗\(Gj∧Gℓ)(i) ∩F ∗
jℓ|
by a Poisson random variable X with mean

λn := ν log n

n
|{j ∈F ∗
jℓ: {i, j} ∈G∗\ (Gj ∧Gℓ)| ≤ν log n

n
|F ∗
jℓ|, ν := max(a, b).

For a fixed δ > 0, define an event Z := {|F ∗
jℓ| ≤n1−s2Tc(a,b)+δ}. On Z, λn ≤n−s2Tc(a,b)+δ+o(1).
Hence, for any positive integer m:

P({|NG∗\(Gj∧Gℓ)(i)∩(F ∗
jℓ)| ≥m}∩Z ≤P({X ≥m}∩Z) = E[P(X ≥m|F ∗
jℓ, E, σ∗, π∗)1Z]

≤E[(inf
θ>0 e−θm+λn(eθ−1))1Z] ≤E[eλm
n 1Z] ≤n−m(s2Tc(a,b)−δ−o(1)).

Above, the equality on the second line is due to the tower rule and since Z is measurable with respect
to |F ∗
jℓ|, the inequality on the third line is due to a Chernoff bound; the inequality on the fourth line
follows from setting θ = log(1/λn) (which is valid since λn = o(1) if Z holds). The final inequality
uses the upper bound for λn on Z. Taking a union bound, we have

P({∃i ∈[n], |NG∗\(Gj∧Gℓ)(i) ∩F ∗
jℓ| ≥m} ∩Z) ≤n1−m(s2Tc(a,b)−δ−o(1)).

Here if we take m > (s2Tc(a, b))−1 and δ < s2Tc(a, b) −m−1, the probability turns to o(1). Thus,
we can set m = ⌈(s2Tc(a, b))−1⌉+ 1. In light of Lemma E.6, |F ∗
jℓ| ≤n1−s2Tc(a,b)+δ, δ > 0 w.h.p.
Hence, the event Z happens with probability 1 −o(1). Hence we have

P({∀i ∈[n], NG∗\(Gj∧Gℓ)(i) ∩F ∗
jℓ| ≤⌈(s2Tc(a, b))−1⌉}) = 1 −o(1).

42


---Page Break---
Hence we have, with probability 1 −o(1), for i ∈M ∗

|majH(i) −majG∗(i)| <
X

bµjℓ/∈Ψ
(ϵ log n/2K2 + ⌈(s2Tc(a, b))−1⌉) < ϵ log n,

and hence with probability 1 −o(1),

majH(i) > ϵ log n.

Then by Lemma E.5, we can replace H with eG, F ∗
ij with Fij, the lemma follows.

Next, prove that each vertex in G∗\π∗
12 G1 has a small number of neighbors in Iϵ(G1).

Lemma H.9. If 0 < ϵ ≤
sD+(a,b)
4| log(a/b)|, then

P(∀i ∈[n], |NG∗\π∗
12G1(i) ∩Iϵ(G1)| ≤2⌈(sD+(a, b))−1⌉) = 1 −o(1).

Proof. Since Iϵ(G1) depends on G1 alone, it follows that Iϵ(G1) and G∗\π∗
12 G1 are conditionally
independent given π∗, σ∗, E. Hence we can stochastically dominate |NG∗\π∗
12G1(i) ∩Iϵ(G1)| by a
Poisson random variable X with mean λn given by

λn := ν log n/n|{j ∈Iϵ(G1) : {i, j} ∈G∗\ G1}| ≤ν log n/n|Iϵ(G1)|.

Next, define the event Z := {|Iϵ(G1)| ≤n1−sD+(a,b)+2ϵ| log(a/b)|}.

Notice that P(Z) = 1 −o(1) by Lemma D.7 and Markov’s inequality, provided sD+(a, b) < 99.
Following identical arguments as the proof of Lemma H.7, we arrive at

P(∃i ∈[n], |NG∗\π∗
12G1(i) ∩Iϵ(G1)| ≥m) = o(1)

when m > ⌈(sD+(a, b) −2ϵ| log a/b|)−1⌉.
If ϵ ≤
sD+(a,b)
4| log(a/b)|, then it suffices to set m =
2⌈(sD+(a, b))−1⌉+ 1.

Lemma H.10. Suppose that a, b, ϵ > 0 satisfy the following conditions:

(1 −(1 −s)K)D+(a, b) > 1 + 2ϵ| log a/b|, 0 <ϵ ≤sD+(a, b)

4| log a/b|.

With high probability, the algorithm correctly labels all vertices in {i ∈[n] \ M ∗}.

Proof. Compare the neighborhood majority in H corresponding to c
σ1 with the true majority in H,
where H is defined in Lemma F.2:

|σ∗(i)
X

j∈NH(i)
(c
σ1(j) −σ∗(j))| ≤|NH(i) ∩Iϵ)| ≤|NG∗(i) ∩Iϵ(G1)|

≤|NG∗\G1(i)∩Iϵ(G1)|+|NG1(i)∩Iϵ(G1)| ≤2⌈D+(a, b)−1⌉+2⌈(sD+(a, b))−1⌉≤ϵ log n/2.

The first inequality uses Lemma D.5 that the set of errors are contained in Iϵ(G1). The last inequality is
due to Lemma D.8, H.9. Notice that majH(i) ≥ϵ log n for i ∈M ∗. Hence, σ∗(i) P

j∈NH(i) bσ1(j) ≥
majH(i) −|σ∗(i) P

j∈NH(i)(c
σ1(j) −σ∗(j))| ≥ϵ log n/2 > 0, which implies that the sign of
neighborhood majorities are equal to the truth community label for any i ∈M ∗, with probability
1 −o(1). Then we can convert H to eG{M}, the vertices in M are correctly labeled with probability
1 −o(1).

Using an identical proof, we can argue that the algorithm correctly labels all “good” vertices with
probability 1 −o(1).

43


---Page Break---
H.5
Exact recovery for “bad” vertices

Lemma H.11. Suppose that a, b, ϵ > 0 satisfy the following conditions:

(1 −(1 −s)K)D+(a, b) > 1 + 2ϵ| log a/b|, 0 <ϵ ≤sD+(a, b)

4| log a/b|,

s(1 −(1 −s)K−1)Tc(a, b) + s(1 −s)K−1D+(a, b) > 1.

With high probability, the algorithm correctly labels all “bad” vertices.

Proof. For vertex i that are “bad”, denote ψ := {j ∈[K] : i cannot be matched through bµ1j, j ̸= 1}.
Denote Fb as the vertex set of all the “bad” vertices that have the same ψ with vertex i. Denote
M ∗:= ∩i∈[K]M ∗
1i. define Hi := (G1 \π∗
12 G2 \π∗
13 G3 . . . \π∗
1K GK){M ∪{i}}. Let Ei be the event
that i has a majority of at most ϵ′ log n in the graph Hi. Let bσ be the labeling after the step. For
brevity, define a “nice” event based on the previous results. Define the event H, which holds if and
only if:

• Fij = F ∗
ij;

• bσ(i) = σ∗(i) for all i ∈M ∗;

• The event F holds;

• |Fb| ≤n1−s(1−(1−s)K−1)Tc(a,b)+δ.

By Lemma E.5, H.4, D.4, H.10, the event H holds with probability 1 −o(1). Furthermore, define
E∗
i := majHi(i) ≤ϵ′ log n, we have that

P(∪i∈[n]({i ∈Fb} ∩Ei))

≤P((∪i∈[n]({i ∈F ∗
b } ∩E∗
i )) ∩H) + P(Hc)

≤

n
X

i=1
P({i ∈F ∗
b } ∩E∗
i ∩{F ∗
b ≤n1−s(1−(1−s)K−1)Tc(a,b)+δ} ∩F) + o(1).

By the tower rule, rewrite the term in the right hand side as:

E
h
P (E∗
i | π∗, σ∗, E, F ∗
b ) 1i∈F ∗
b 1{|F ∗
b |≤n1−s(1−(1−s)K−1)Tc(a,b)+δ}∩F
i
.
(H.12)

Now look at P (E∗
i | π∗, σ∗, E, F ∗
b ). Conditional on E, σ∗, π∗, majHi(i) :
d= Y −Z, where Y, Z are
independent with:

Y ∼Bin(|j ∈M ∗: {i, j} ∈E100...0 ∩E+(σ∗)|, a log n/n),

Z ∼Bin(|j ∈M ∗: {i, j} ∈E100...0 ∩E−(σ∗)|, b log n/n).
By the Definition D.3 of the event F, we know that |j ∈M ∗: {i, j} ∈E100...0 ∩E−(σ∗)| =
(1 −o(1))s(1 −s)K−1n/2 and |j ∈M ∗: {i, j} ∈E100..0 ∩E+(σ∗)| = (1 −o(1))s(1 −s)K−1n/2.

Lemma D.1 implies that

P(E∗
i |π∗, σ∗, E, F ∗
b )1i∈F ∗
b 1{|F ∗
b |≤n1−s(1−(1−s)K−1)Tc(a,b)+δ}∩F

≤n−s(1−s)K−1D+(a,b)+ϵ′ log(a/b)/2+o(1).

Follow (H.12) and take a union bound, we have that

n
X

i=1
P({i ∈F ∗
b } ∩E∗
i ∩{F ∗
b ≤n1−s(1−(1−s)K−1)Tc(a,b)+δ} ∩F) + o(1)

≤n−s(1−s)K−1D+(a,b)+ϵ′ log(a/b)/2+o(1)E[|F ∗
b |1F ∗
b ≤n1−s(1−(1−s)K−1]

≤n1−s(1−(1−s)K−1)Tc(a,b)−s(1−s)K−1D+(a,b)+ϵ′ log(a/b)/2+δ.

44


---Page Break---
Under the condition s(1 −(1 −s)K−1)Tc(a, b) + s(1 −s)K−1D+(a, b) > 1, we can choose ϵ′, δ
small enough so that the right hand side is o(1). majHi(i) > ϵ′ log n for i ∈F ∗
b , by Lemma E.5,
majc
Hi(i) > ϵ′ log n for i ∈Fb.

Note that i cannot be matched for all bµ1i, i ∈ψ. Hence i has at most 12 neighbors in the graph
(G1 ∧π∗
1i Gi). Therefore for any i ∈Fb has at least ϵ′ log n −12|ψ| majority in G1 \bµ1j,j /∈ψ Gj{M ∪
{i}} with high probability. Hence, we can correctly label all vertices in Fb with high probability.

Use the same arguments for all types of “bad” vertices, we can correctly label all “bad” vertices.

I
Proof of impossibility for K graphs

We study the MAP (maximum a posterior) estimator for the communities in G1. Even with the
additional information provided, including all the correct community labels in G2, the true matching
π∗
23, π∗
24, . . . , π∗
2K and most of the true matching π∗
12, the MAP estimator fails to exactly recovery
communities with probability bounded away form 0 if the condition (G.1) holds. The proof can
be derived by generalizing proof of impossibility for three graphs. The only difference is that we
are considering K correlated SBM G1, G2, . . . , GK. Since we know the true matching π∗
i,j, i, j ∈
{2, 3, . . . , K}, we can consider H := G2∨G3 . . .∨GK ∼SBM(n, (1−(1−s)K−1)a log n/n, (1−
(1 −s)K−1)b log n/n). Denote Rij the singleton in Gi ∧Gj. Then R = R12 ∧R13 . . . ∧R1K is
the singleton set in G1 ∧H. The proof follows the same arguments with more involved notation, and
hence we omit the details. Here we point out the differences of the proof for K graphs.

Define Rπ
:= R(π, A, B2, ..., BK) := {i ∈[n] : ∀j
∈[n], Ai,jDπ(i)π(j) = 0, D =
max(B2, ...., BK)}, here Bi is the adjacent matrix of Gk. Similar to Definition G.2, we can
define Sπ = S(π, A, B2, .., BK). Let Gδ be the event that the following inequalities all hold:

n1−s(1−(1−s)K−1)Tc(a,b)−δ ≤|R∗∩V +
1 |, |R∗∩V −
1 |, | ¯R∗∩V +
1 |, | ¯R∗∩V +
1 |

≤n1−s(1−(1−s)K−1)Tc(a,b)+δ.

We can prove similar versions of Lemma G.7 and G.8, with (2s2 −s3) replaced by s(1−(1−s)K−1)
and s(1 −s)2 replaced by s(1 −s)K−1. We can have same versions of Lemma G.9, Corollary G.12.
We can similarly define µ+(π)i1i2...iK and µ−(π)i1i2...iK for all i1, . . . , iK ∈{0, 1}, and ν+(π) and
ν−(π). When deriving the posterior distribution of π∗
12, similar to Lemma G.14, the information
of A, B2, . . . , BK, σ∗
2, S∗, π∗
12{[n] \ S∗}, π∗:= {π∗
ij, i, j ∈{2, 3, . . . , K}} are given. Note that
for π ∈A∗, we have that µ+(π)1i2...iK and µ−(π)1i2...iK are constant for all i2, . . . , iK ∈{0, 1}
except for µ+(π)10...0 and µ−(π)10...0. We can derive an analogue of Lemma G.14 with p100
replaced by p100...0, p000 replaced by p000...0 and similar for q. Then we have analogous versions of
Lemmas G.15, G.16, and G.17, with p100 replaced by p100...0, p000 replaced by p000...0, and similar
for q. Note that p100...0q000...0

p000...0q100...0 = (1 + o(1)) a

b . The impossibility proof for Theorem 2 follows.

J
Proofs for exact graph matching

J.1
Exact graph matching for K graphs

Proof. Through the 13-core matching in Algorithm 1, we obtain eπ := {bµij, i, j ∈[K]}, where bµij
is the 13-core matching between the graph Gi and Gj.

Recall Definition H.2, we can directly infer that the "good" vertices are those which can be matched
for K graphs through a path across 13-core estimators eπ. We can define a new estimator bπ for those
“good” vertices using the combination of 13-core estimator through the path that connects all K
graphs. The path is defined as in Lemma H.3. For any “good” vertex, such path exists and we can
define the estimator bπ for that vertex.

Note that, by Lemma E.5, with high probability bµij = π∗
ij. Hence if for all the matched vertices,
they will be matched correctly. If the number of "bad" vertices approaches zero, it indicates that all
vertices are correctly matched. Consequently, exact graph matching can be achieved through the
13-core matching algorithm. By Lemma H.4, H.6, we can quantify the size of “bad” vertices: for
every δ > 0 we have that | ∩1≤i≤L,L+1≤j≤K F ∗
ij| ≤n1−s(1−(1−s)K−1)Tc(a,b)+δ. When 1 −s(1 −

45


---Page Break---
(1 −s)K−1)Tc(a, b) > 1, that is, when the condition (3.5) holds, the number of “bad” vertices goes
to zero when n goes to infinity, with high probability. Thus all vertices can be correctly matched and
exact graph matching for K graphs is possible with high probability.

J.2
Impossibility of exact graph matching for K graphs

Proof. Now we consider the graph matching problem with additional information provided, in-
cluding the true correspondences π∗
23, . . . , π∗
2K and the community label σ∗. Then we obtain
the union graph H := G2 ∨π∗
23 G3 ∨. . . ∨π∗
2K GK. We now prove the impossibility by con-
tradictory. Suppose that there exists an estimator bπ which can exactly match G1, G2, ..., GK,
note that H ∼SBM(n, (1−(1−s)K−1)a log n

n
, (1−(1−s)K−1)b log n

n
).
One key point, is that, we
can subsample H′
2, H′
3, ..., H′
K from H. To be more specific, consider the following parame-

ter: ri1,i2,...,iK = s

PK
j=1 ij (1−s)
K−PK
j=1 ij

1−(1−s)K−1
where i1, i2, . . . , iK ∈{0, 1} and PK
j=1 ij > 0. Here
P ri1,i2,...,iK = 1. Then for any vertex pair (i, j):

1. If (i, j) is not an edge in H, then (i, j) is not an edge in H′
2, H′
3, . . . , H′
K.

2. If (i, j) is an edge in H, with probability ri1,i2,...,iK, (i, j) is an edge in the graphs {H′
ij}
where ij = 1 in ri1,i2,...,iK and (i, j) is not an edge in the graphs {H′
ij} where ij = 0 in
ri1,i2,...,iK.

Following the subsampling described as above, we can simulate H′
2, . . . , H′
K from H. Note that
by construction, (G1, G′
2, G′
3, . . . , G′
K) has the same distribution as (G1, H′
2, . . . , H′
K). Then af-
ter independent permutations, we can obtain (H3, H4, . . . , HK) by relabeling the vertex index in
(H′
3, H′
4, . . . , H′
K). Note that H2 = H′
2. Then (G1, G2, . . . , GK) has the same distribution as
(G1, H2, H3, . . . , HK). Since the estimator bπ can exactly match (G1, G2, . . . , GK) with high proba-
bility, it can also exactly match (G1, H2, H3, . . . , Hk) with high probability. Naturally, it can exactly
match vertices in G1 and H2, since H and H2 share the same vertex index then we can have an estima-
tor that exactly match G1 and H given G1 and H, where G1, H are correlated SBMs, independently
subsampling from the parent graph G with probability s1 = s for G1 and s2 = 1−(1−s)K−1 for H.
However, [14, Theorem 1] proves that suppose (G1, G2) ∼CSBM(n, a log n

n
, b log n

n
, s1, s2) subsam-
pling from G ∼SBM(n, a log n

n
, b log n

n
) with probability s1 and s2, respectively, if s1s2Tc(a, b) < 1,
then exact graph matching between G1 and G2 is impossible. Directly applying [14, Theorem 1] we
have that exact graph matching between G1 and H is impossible if s(1 −(1 −s)K−1) < 1. This is a
contradiction, and hence exact graph matching for G1, . . . , GK is impossible.

46


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: As stated in the abstract and introduction, the main contribution of our paper is
to derive the precise information-theoretic threshold for exact community recovery given K
correlated stochastic block models, for any constant K ≥3. This is exactly the content of
Theorem 1 and Theorem 2 in Section 3 (as discussed in detail in Sections 2 and 3).

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

Justification: We discuss limitations, and possible future work that may address these, in
Section 6.

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

47


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: We clearly define the model and questions that we study in Section 2, and we
fully state our main theoretical results, including all assumptions, in Theorem 1, Theorem 2,
Theorem 3, and Theorem 4. For all theorems, we provide a brief overview of the proof in
Section 4, and we give the full proof in the Appendix.
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
Justification: The paper does not include experiments.
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

48


---Page Break---
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

49


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
Answer: [NA]
Justification: The paper does not include experiments.
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
Justification: The research conducted in the paper conforms, in every respect, with the
NeurIPS Code of Ethics.
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
Justification: The current work is primarily theoretical. Nonetheless, broader societal
impacts, both positive and negative, are discussed in Section 1 and Section 5.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.

50


---Page Break---
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
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

51


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

52


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

53


---Page Break---
