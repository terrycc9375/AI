What do Graph Neural Networks learn?
Insights from Tropical Geometry

Tuan Anh Pham
School of Mathematics
University of Edinburgh
Edinburgh, United Kingdom
tuan.pham@ed.ac.uk

Vikas Garg
YaiYai Ltd and Aalto University
vgarg@csail.mit.edu

Abstract

Graph neural networks (GNNs) have been analyzed from multiple perspectives,
including the WL-hierarchy, which exposes limits on their expressivity to distin-
guish graphs. However, characterizing the class of functions that they learn has
remained unresolved. We address this fundamental question for message passing
GNNs under ReLU activations, i.e., the de-facto choice for most GNNs.
We first show that such GNNs learn tropical rational signomial maps or continuous
piecewise linear functions, establishing an equivalence with feedforward networks
(FNNs). We then elucidate the role of the choice of aggregation and update
functions, and derive the first general upper and lower bounds on the geometric
complexity (i.e., the number of linear regions), establishing new results for popular
architectures such as GraphSAGE and GIN. We also introduce and theoretically
analyze several new architectures to illuminate the relative merits of the feedforward
and the message passing layers, and the tradeoffs involving depth and number of
trainable parameters. Finally, we also characterize the decision boundary for node
and graph classification tasks.

1
Introduction

Message passing has been a cornerstone of machine learning, from inference in graphical models
[1, 2] to embedding of graphs [3, 4, 5, 6, 7]. Message passing neural networks (MPNNs) are easy to
implement, and can handle large scale, heterogeneous, and dynamic real world data effectively; so
continue to be an active area of research [8, 9, 10]. Indeed, several popular GNNs are usually cast
and implemented as MPNNs, see e.g., [11, 12, 13, 14, 15].

Unsurprisingly, GNNs as MPNNs have been analyzed from multiple theoretical perspectives. A
major theme is inspired by connections to the so-called 1-WL (Weisfeiler-Leman) test for group
isomorphism and its higher order extensions, where nodes in a (hyper-)graph repeatedly refine
their colors based on the messages from their neighbors [16]. MPNNs are known to be bounded
in power according to the WL-hierarchy [15, 17, 18], implying their inability to distinguish some
non-isomorphic graphs. Unlike WL that strives to expose what GNNs cannot do, here we seek to
unravel what they can.

Notably, the WL formalism implicitly relies on injective hash functions; in contrast, most successful
GNNs typically use ReLU activations that violate injectivity. Bounding the expressivity of such
GNNs via their injective surrogates is theoretically valid; however, it does not illuminate what they
learn. Indeed, several fundamental questions remain elusive for these practical models; e.g., (a) what
class of functions can they represent, (b) how does their expressivity vary with the choice of message
aggregation and update functions, (c) what complexity tradeoffs (e.g., in terms of the number of

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Theoretical contributions of this work

Characterizing the class of functions learned by ReLU MPNNs:
Equivalence with ReLU FNNs, TRSMs and CPLMs
Proposition 1

Estimating the number of linear regions, and complexity tradeoffs:
First general lower bound for ReLU MPNNs
Theorem 3
First general upper bound for ReLU MPNNs
Theorem 4
Max aggregation has greater geometric complexity than sum
Proposition 5
Recovery of existing upper bounds for FNN and GCN
Corollary 1, 2
New upper bounds for GraphSAGE and GIN
Corollary 3, 4

New ReLU MPNNs and complexity tradeoffs:
New architectures that can all learn CPLMs, and their tradeoffs
Proposition 6

Characterizing the decision boundary:
Decision boundary of ReLU MPNNs for graph classification
Proposition 7
Decision boundary of ReLU MPNNs for node classification
Proposition 8

Figure 1: Overview of our results. Formulating ReLU MPNNs as tropical geometric objects allows
us to shed light on several important aspects where WL falls short.

layers or parameters) exist for models of comparable expressivity, and (d) what decision boundary
emerges for node and graph classification tasks?

We appeal to tropical algebra and geometry to address these questions in the context of ReLU MPNNs.
Specifically, casting these networks as tropical geometric objects, we analyze their ability to represent
different weighted sums of tropical monomials (the basic objects of interest in tropical algebra,
akin to monomials for standard polynomials) at different nodes. We first show that the family of
functions represented by these networks is precisely the family of tropical rational signomial maps
(TRSMs). Since TRSMs are known to be equivalent to ReLU FNNs [19], this reveals an equivalence
between ReLU MPNNs and ReLU FNNs: they represent exactly the same class of functions, namely,
continuous piecewise linear maps (CPLMs).

This equivalence in terms of expressivity does not, however, translate into parity in efficiency, e.g., as
quantified in terms of their respective requirements for the number of layers and trainable parameters
to represent an arbitrary TRSM. Indeed, MPNNs have some strengths and limitations relative to FNNs.
On the positive side, MPNNs benefit from parallel computation and streamlined communication
between the nodes and their neighbors. On the flip side, MPNNs are constrained by permutation-
equivariant layers that employ permutation-invariant aggregation operators [20], which impedes their
ability to represent arbitrary combinations of tropical monomials.

In order to better understand the efficiency of different MPNN architectures, we investigate their
geometric complexity, the number of linear regions that they can distinguish. Towards that end, we
establish the first general upper and lower bounds on the geometric complexity of ReLU MPNNs,
recovering the existing results [19, 21] for FNNs and graph convolutional networks (GCNs) as
special cases. Importantly, we also unravel new results for two of the most prominent MPNNs whose
complexity was previously unknown, namely, GraphSAGE [13] and GIN [15].

A particularly attractive aspect of our bounds is manifested in segregation of the contribution from
different components, such as aggregation and update steps. In particular, they reveal that selecting
coordinate-wise max as the aggregation operator affords greater geometric complexity than sum for
ReLU MPNNs. In order to provide further insights about various complexity tradeoffs, we introduce
and analyze four novel MPNN abstractions. Notably, they can all represent arbitrary TRSMs, i.e., they
are as expressive as ReLU FNNs, but differ in terms of layers as well as total learnable parameters
in their corresponding architectures. Fundamentally, we expose a general trend about the relative
merits of the feedforward and message passing paradigms: fewer layers for MPNNs (particularly
with coordinate-wise max aggregation) but fewer trainable parameters for FNNs.

2


---Page Break---
Finally, we study the decision boundaries for node and graph classification, uncovering the underlying
connections with tropical hypersurfaces; i.e., the set of points where two or more tropical monomials
achieve the same value.

We summarize our key contributions in Figure 1. We now proceed to reviewing some related works.

1.1
Related work

Tropical Geometry and Machine Learning.
Tropical geometry [22, 23] provides tools to study
the algebraic geometry and combinatorics of continuous piecewise linear functions, and finds several
applications (e.g., in optimization). Two seminal works [19, 24] initiated the analysis of deep learning
models via tropical geometry, establishing the link between ReLU FNNs and tropical rational
functions. The connection was then extended to maxout-layers in [25]. The decision boundaries
of FNNs through a tropical lens were studied in [26]. Other aspects of deep neural networks have
also be analyzed [27, 28, 29, 30, 31, 32]. We refer to [33] for a survey on the current use of tropical
geometry in deep learning.

Expressivity and WL.
Much work on GNNs has been inspired by noticing the parallels between
MPNNs and the WL test for isomorphism [15, 34, 35]. Standard MPNNs are no more powerful than
1-WL (equivalently, 2WL), so higher order models that consider tuples of vertices have been proposed
[17, 18]. Several adaptations of WL such as geometric WL [36], cellular WL [37],temporal WL [10],
and persistent WL [38] have been introduced in different contexts. Limited expressivity and symmetry
considerations have led to the design of new MPNN architectures [15, 39, 40, 41, 42, 43], inclusion
of different types of features [39, 44], and integration with topological descriptors [45, 46, 47].
Other notions of expressivity have also been proposed, e.g., equivariant polynomials [48] and
homomorphism counts [49].

Other results about GNNs.
GNNs as MPNNs are also known to be prone to information bottle-
necks [50, 51, 52], oversmoothing and oversquashing [53], and heterophily [54, 55]. Some works
have also established their inability to count substructures, compute graph properties, or learn topol-
ogy [49, 56, 57]; connections to algebraic topology [58], biconnectivity [59], and communication
complexity [60, 61], behavior in overparametrized regimes [62, 63]; power of recursion in counting
substructures [64]; benefits of positional encoding [65]; and ability to generalize and achieve universal
approximation [56, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75].

However, neither expressivity in terms of WL nor these other results characterize the class of functions
learned by GNNs, and in this work we bridge this glaring gap for ReLU MPNNs.

Complexity of Neural Networks.
The complexity of deep neural networks is typically studied in
terms of the number of linear regions [76, 77, 78, 79], and the intricacy of decision boundary [19],
though other notions such as trajectory length [80] have also been considered recently. The number
of linear regions quantifies the flexibility of the function class, thus bounding the number of regions
relates closely to both expressivity and generalization [79]. To our knowledge, only the geometric
complexity upper bounds for GCNs are known in the context of GNNs [21]. We provide the first
general lower and upper bounds for ReLU MPNNs, establishing complexity of popular architectures
such as GraphSage and GIN, and recovering the results for GCNs and FNNs as special cases.

2
Preliminaries

2.1
Message Passing Neural Networks (MPNNs)

A general T-layer MPNN χ is defined as a sequence of layers {φ(t)}T
t=1, and takes as input a
(directed) graph G = (V, E, X) with vertices v ∈V , edges (u, v) ∈E ⊆V × V , and features
X = (XV , XE) consisting of node attributes XV ∈R|V |×d0 and edge attributes XE ∈R|E|×d′
0. It
produces as output a refined embedding for each vertex and edge. Formally, χ acts on X as

χ(X) = φ(T ) ◦... ◦φ(1)(X) , where each φ(t) : R|V |×dt−1 × R|E|×d′
t−1 →R|V |×dt × R|E|×d′
t

is permutation-equivariant [20]. Subsequently, depending on the task, the embedding for each node
produced by χ is fed into another neural network, e.g., η : RdT →Rout for node classification or

3


---Page Break---
regression (and similarly for edges). An additional readout step φReadout : R|V |×dT ×R|E|×d′
T →RdG
is usually required to obtain a single vector from the output of χ for graph-level prediction.

We thus proceed to describing the working of each layer φ(t) that yields a representation h(t)
v
∈Rdt

for each vertex v ∈V and a representation e(t)
uv ∈Rd′
t for each edge (u, v) ∈E. Assuming node
embeddings are updated before edge embeddings (the converse works analogously), we have

h(t)
v
= φ(t)
Update(h(t−1)
v
, m(t)
v ),
where

m(t)
v
= φ(t)
Agg(h(t−1)
v
, {{h(t−1)
u
, e(t−1)
uv
, e(t−1)
vu
|u ∈Ne(v)}}) .

Aggregation functions {φ(t)
Agg} are typically permutation-invariant, and we use an operator □such as
sum, average, or coordinate-wise max/min to combine the messages [15, 81]:

φ(t)
Agg(h(t−1)
v
, {{h(t−1)
u
, e(t−1)
uv
, e(t−1)
vu
}}u∈Ne(v)) = ϕ(t)
1 (□(t)
u∈Ne(v)ϕ(t)
2 (h(t−1)
v
, h(t−1)
u
, e(t−1)
uv
, e(t−1)
vu
)),
(1)
where {ϕ(t)
1 } and {ϕ(t)
2 } are usually implemented as FNNs, Ne(v) denotes the set of neighboring
vertices of v, and {{·}} denotes a multiset. When there is no confusion, we usually write □(t) instead.
Let L(t)
1
and L(t)
2
denote, respectively, the number of layers in ϕ(t)
1
and ϕ(t)
2 . We use nt,ℓ
1 (and nt,ℓ
2 )
to denote the dimension of layer ℓin ϕ(t)
1
(and ϕ(t)
2 ). On the other hand, the update functions take the
form

φ(t)
Update(h(t−1)
v
, m(t)
v ) = σ(t)
Update(W (t)
selfh(t−1)
v
+ W (t)
neighm(t)
v ).

Henceforth, we focus on sum and coordinate-wise max, since coordinate-wise min and average can
be obtained from coordinate-max and sum respectively. We shall also number the vertices in V with
A1, ..., A|V | and edges in E with e1, ..., e|E|, and use the lexical order for edges, i.e.,
euv < eu′v′ ⇔u < u′ or (u = u′ and v < v′) .

2.2
Tropical Algebra

Here we adopt the notation from [19]. Let T = R ∪{−∞} be an extended set of real numbers. We
equip T with two binary operators tropical sum ⊕and tropical multiplication ⊙:

a ⊕b = max{a, b} ;
a ⊙b = a + b ;
a ⊕−∞= −∞⊕a = a ;
a ⊙−∞= −∞⊙a = −∞,

where max and + are the usual operators in R. Thus, (T, ⊕, ⊙) is a semi-ring with additive identity
−∞and multiplicative identity 0. We also define the tropical power x⊙a for each x ∈R:

x⊙a =

(
x ⊙... ⊙x = a · x
if a ∈N
(−x)⊙(−a)
if a ∈Z \ N
,
−∞⊙a =
∞
if a ∈N \ {0}
0
if a = 0
;

where · is the standard product over R, and often abbreviate x⊙a to xa without inducing any confusion.
A tropical monomial in m variables takes the form c⊙xa1
1 ⊙...⊙xam
m for c ∈T and a1, ..., am ∈N,
and is often denoted by multi-index shorthand cxα, where α = (a1, . . . , am) ∈Zm and x =
(x1, . . . , xm) ∈Rm. Note that this may be interpreted as the affine combination x⊤α + c.

A tropical polynomial f(x) = c1xα1 ⊕... ⊕crxαr is a finite tropical sum of tropical monomials,
and amounts to a max over finitely many terms, i.e., f(x) = maxr
i=1{x⊤αi + ci}. Without loss of
generality, we assume αi ̸= αj for i ̸= j since we can combine monomials with the same α into one.
A tropical rational function f ⊘g(x) = f(x) −g(x) is the difference between two tropical
polynomials. A map F : Rm →Rp with each component a tropical polynomial [resp. tropical
rational function] is called a tropical polynomial map [resp. tropical rational map], and belongs to
the set Pol(m, p) [resp. Rat(m, p)].

When we allow αi ∈Rm instead of restricting its components to integers (αi ∈Zm), we obtain a
tropical signomial function/map [resp. tropical rational signomial function/map] instead of a
tropical polynomial/map [resp. tropical rational function/map]. It is known that each tropical rational
signomial map (TRSM) is a continuous piecewise linear map (CPLM) and vice versa [19].

4


---Page Break---
3
Tropical Algebra of MPNNs

In this section, we characterize the class of functions learned by ReLU MPNNs, whose activation
functions for both nodes and edges (i.e., those used for η, ϕ(t)
1 , ϕ(t)
2 , σ(t)
Update, etc.) are of the form:

σ(l)(x) = max{x, t(l)}, where t(l) ∈(R ∪−∞)nl .

In particular, note that σ(l)(x) = x when t(l) = −∞. In contrast, t(l) = 0 purges all negative inputs.

Let FReLU MPNN, FReLU FNN, FCPLM, FTRSM be the set of functions represented by all ReLU MPNNs,
ReLU FNNs, CPLMs and TRSMs respectively. [19] established the equivalence of ReLU FNNs,
CPLMs and TRSMs.
Lemma 1 ([19]). FReLU FNN = FCPLM = FTRSM .

We will now extend this result to establish equivalence with FReLU MPNN. Equating ReLU MPNNs
with CPLMs is rather nuanced, since nodes in each layer share the weights. Therefore, we employ
two reductions showing (1) every ReLU FNN can be cast as a ReLU MPNN, and (2) every ReLU
MPNN, in turn, can be expressed as a TRSM.
Proposition 1. [Equivalence of ReLU MPNNs, ReLU FNNs, TRSMs and CPLMs] FReLU MPNN =
FReLU FNN = FCPLM = FTRSM. In other words, the following families are equivalent (with m =
|V |d + |E|d′ and p = |V |dout + |E|d′
out).

1. ReLU MPNNs χ : R|V |×d × R|E|×d′ →R|V |×dout × R|E|×d′
out;
2. Tropical rational signomial maps (TRSMs) F ⊘G : Rm →Rp ;
3. Continuous piecewise linear maps (CPLMs) ψ : Rm →Rp ;
4. ReLU FNNs ν : Rm →Rp.
Remark 1. In contrast to the WL test, which exposes the limitations of MPNNs via injective hash
functions, Proposition 1 characterizes the class of functions that can be represented by MPNNs
with ReLU activations (that are non-injective). Note, however, that Proposition 1 does not quantify
how effective ReLU MPNNs are in representing CPLMs. Moreover, the equivalence between ReLU
MPNNs and ReLU FNNs (according to Proposition 1) does not explain the observed empirical
discrepancy between ReLU MPNNs and ReLU FNNs in practice. This motivates our subsequent
analysis and results (Section 5 and Table 1), which investigate the benefits of ReLU MPNNs in terms
of both the number of learnable parameters and the number of layers required to represent the same
CPLM.

4
Geometric Complexity of ReLU MPNNs

We now invoke tools from tropical algebraic geometry to study the geometric complexity of ReLU
MPNNs, and provide some insights into the model architecture. Following [19], for a CPLM
f : Rm →Rp, we define its linear degree N(f) to be K, where K is the least number of connected
regions Ωk of Rm such that the restriction f|Ωk is affine. Equivalently, following [82], we can also
define K = N(f) as follows: f : Rm →Rp is a CPLM if f is continuous and there exists a set
{fk : k ∈{1, ..., K}} of affine functions and maximal connected subsets (Ωk)K
k=1 satisfying the
following conditions:

Ωi ∩Ωj = ∅;

K
G

k=1
Ωk = Rm;
f|Ωk = fk.

Similarly, we define the convex degree Nc(f) where we require additionally that Ωk is convex.
We further define Nc(f|m′) as the maximum convex degree across restrictions of f to different
m′-dimensional affine subspaces of Rm, m′ ≤m. We will analyze general upper and lower bounds
on N(χ) for a ReLU MPNN χ. Recall our setting of MPNN layers from Section 2.1 consisting of
ϕ(t)
Agg and ϕ(t)
Update. Note that nt,l
1 and nt,l
2 denote the intermediate dimensions in ϕ(t)
1
and ϕ(t)
2 . Let ˜dt

and ¯dt be the output dimension of ϕ(t)
2
and ϕ(t)
1 , thus n
t,L(t)
2
2
= ˜dt.

To analyze the number of linear region for a ReLU MPNN χ, we make a simplifying assumption that
all input graphs to χ have the same graph structure G. In particular, we denote the sum of degrees
of all vertices in G by D, and the maximum degree by S. A key step in our analysis of geometric

5


---Page Break---
complexity is building a ReLU FNN that can be applied on the vectorized input H(t−1) of all node
embeddings h(t−1)
v
. Using the notation indicated in Equation 1, we form a ReLU FNN Φ(t)
2
that
achieves the same effect as ϕ(t)
2
for every adjacent node embedding. The aggregation operator can be
seen as either a matrix multiplication for sum, or a FNN Φ(t)
3
for max aggregation - the difference
between these two cases will be discussed in Section 4.3. Similarly, we form a ReLU FNN Φ(t)
1
for
ϕ(t)
1 , which in combination with Φ(t)
2
and Φ(t)
3
can be seen as a ReLU FNN Φ(t)
Agg.

Our next proposition is important in the analysis for geometric complexity of χ, as it relates the
geometric complexity of the model up to the t + 1-th layer to that of the t−th layer, and the update as
well as the message components. We define the vectorized concatenated embedding of all the vertices
in the t-th layer to be H(t) ∈R|V |dt and show that the result of t-th message aggregation step can be
written as a result of a FNN Φt
Agg(H(t)). Modifying any aggregation or update component thus will
affect the bound and in particular, the choice of aggregation operator has an impact on the geometric
complexity (which we will discuss in Section 4.3).
Proposition 2. [Recursive formula for geometric complexity]

N(φ(t+1)) ≤Nc(φ(t+1)) ≤Nc(φ(t+1)
Update||V |(dt + ˜dt+1)) Nc(Φ(t+1)
Agg
||V |dt) Nc(φ(t)).
(2)

The ideas and proofs for Proposition 2 build on [19], and the details can be found in the Appendix.

4.1
Lower bound of geometric complexity

We note that the lower bound on the linear degree of χ depends heavily on the choice of weights
and biases; e.g., setting their value to 0 trivially results in N(χ) = 0. Thus, a lower bound on
the geometric complexity (i.e., maximal linear degree) is of greater interest. In this subsection, we
will provide a general lower bound for the maximum number of linear regions for a ReLU MPNN,
building on the work of [76].
Theorem 3. [Lower bound on the maximum number of linear regions] Assume for all t, l, we have

nt,l
1 , nt,l
2 ≥d0 and let nt,l
1,d0 =
j
nt,l
1
d0

kd0
and nt,l
2,d0 =
j
nt,l
2
d0

kd0
then the maximum number of linear
regions of functions computed by any ReLU MPNN is lower bounded by

St0

QT
t=1

QL(t)
1
l=1 nt,l
1,d0
QL(t)
2
l=1 nt,l
2,d0



n
T,L(T )
1
1,d0

d0
X

j=0

dT
j



,

where t0 is the number of MPNN layer having max as aggregation operator and for each layer t, the
index l runs through every layer in ϕ(t)
2 , ϕ(t)
1 .

We now sketch some intuition about this result. We can add to Φ(t)
1
(constructed in Algorithm 3
in the Appendix) an initial layer to calculate □(t) = P, while the aggregation □(t) = max can be
seen as a FNN-layer with rank S max activation, thus we can identify S input regions (indicated in
red). By setting φ(t)
Update(h(t−1)
v
, m(t)
v ) = mv, we can express the whole MPNN χ as a FNN applied
to input X (details in Algorithm 1, 2 and 3 in the Supplementary). We build on the analysis in

[76] to construct intermediate layers that identify
j
nt,l

d0

kd0
input regions. Our procedure amounts to
sequentially folding the input space until the last layer (indicated in blue), and then replicating the
hyperplane arrangement in the last layer (indicated in green).

4.2
Geometric complexity - Aggregation and Update steps

We now provide a general upper bound for the geometric complexity of ReLU MPNNs when all the
weights take integer values - this assumption is mild and holds without loss of generality (details in
the Appendix). We call these models integer-weighted ReLU MPNNs. A result similar to Proposition
1 establishes the equivalence of integer-weight ReLU MPNNs with tropical rational maps.

For our analysis, we require a technical condition that the network “does not shrink” the representation
in the following sense: each intermediate dimension nt,l
1 and nt,l
2 of ϕ(t)
1 and ϕ(t)
2 should be sufficiently

6


---Page Break---
large; and the dimension of the new embedding is at least the dimension of the aggregated message
plus the dimension of the previous embedding. These conditions are standard in the analysis using
tropical geometry, see e.g., [19].
Theorem 4. [Upper bound on the geometric complexity] Let χ : R|V |d0 →R|V |dT be an integer-
weight ReLU MPNN. If φ(t) satisfies the following conditions for all MPNN layer t = 1, .., T

• nt,l
2 ≥
D
|V |dt−1 for all l = 1, ..., L(t)
2 ;

• nt,l
1 ≥dt−1 for all l = 1, ..., L(t)
1 ;

• n
t,L(t)
1
1
+ dt−1 ≤dt;

then the linear degree of H(T ) is at most

T
Y

t=1





L(t)
1 −1
Y

l=1

|V |dt−1
X

i=0

|V |nt,l
1
i





|
{z
}

from Φ(t)
1





L(t)
2 −1
Y

l=1

|V |×dt−1
X

i=0

Dnt,l
2
i





|
{z
}

from Φ(t)
2





|V |( ¯dt+dt−1)
X

i=0

|V |dt
i





|
{z
}

from φ(t)
Update

Nc(□(t)),

where Nc(□(t)) ≤

(
1
if □(t) is sum,
1
2(8S)D ˜dt
if □(t) is coordinate-wise max/min .

We emphasize that, to the best of our knowledge, this the first upper bound for general ReLU message
passing architectures. Furthermore, we recover the upper bounds for FNNs and GCNs (with ReLU
activations and integer-weights) established in [19] and [21] respectively as special cases.

Corollary 1. The linear degree of an integer-weight FNN is at most QT
t=1
Pd
i=0
 dt
i

.

Corollary 2. The linear degree of a GCN χ with T hidden layers, ReLU activation, sum aggregation
and integer weight is at most QT
t=1
P|V |d
i=0
 |V |dt
i

.

On the other hand, we obtain new bounds for popular GNN models, particularly GraphSAGE[13]
and GIN[15]. For GraphSAGE, note that the normalization steps do not change the linear degree.

Corollary 3. Let ˜dt be the output dimension of Aggregatet in GraphSAGE [13]. If dt ≥dt−1 + ˜dt ≥
2dt−1 for all MPNN layers t = 1, ..., T, then the linear degree of integer-weighted ReLU GraphSAGE
described in [13] is upper bounded by

Nc(φT ) . . . Nc(φ1),

where

Nc(φt) ≤

(P2|V |dt−1
i=0
 |V |dt
i

if the aggregation step is mean,
P|V | ˜d−1
i=0
 |D| ˜dt
i
 P|V |( ˜dt+dt−1)
i=0
 |V |dt
i

if the aggregation step is pooling.

Corollary 4. Node embedding of integer-weighted ReLU GIN can be written as [15, Equation 4.1].
Let nt,i be the dimension of each of the intermediate layer in the MLP (t), then its linear degree

Nc(χ) ≤

T
Y

t=1




L(t)−1
Y

l=1

|V |dt
X

i=0

|V |nt,l

i



.

4.3
Coordinate-wise max vs. sum for aggregation

Proposition 5. [Coordinate-wise max has greater geometric complexity than sum]

N(□(t)) = 1
if □(t) = P,
Smin{|V |,D} ˜dt ≤
N(□(t)) ≤min
nP|D|dt+1
i=0
 S2|V | ˜dt
i

, S|V | ˜dt
o
if □(t) = max .

To establish Proposition 5 we again adapt [76]. Note that if D ≥|V | (the graph is not too sparse, and
we have enough messages between the vertices), we have that N(□(t)) = S|V | ˜dt if □(t) = max. In
that case, the geometric complexity will grow polynomially with S (maximum degree).

7


---Page Break---
Remark 2. Interestingly, in [15], the authors point out that if ϕ1 and ϕ2 are injective, then χ is as
powerful as the WL test. Under that assumption, coordinate-wise is less “expressive” than mean,
which is less “expressive” than sum.

In contrast, ϕ1 and ϕ2 are not injective in our case (since ReLU activation is not injective), max is
more “expressive” (as measured by the notion of geometric complexity), thus providing another novel
insight. Here, the connectivity of the graph plays a particularly important role.

5
New ReLU MPNNs architectures and complexity tradeoffs

Note that while both ReLU FNNs and ReLU MPNNs learn TRSMs/CPLMs (Proposition 1), they
might differ vastly in terms of their resource requirements (e.g., the number of layers and parameters).
Therefore, we proceed to comparing the complexity of representing a TRSM under the two paradigms
(we do not need integer-weight assumption in this section). We simplify our analysis by noting that
each component of a TRSM results from the difference of two tropical signomial functions (TSFs)
and these TSFs can be computed in parallel using shared layers. Thus, hereafter, we shall focus on
TSFs, i.e., functions f : Rm →R of the form f(x) = ⊕r
i=1ci ⊙x⊙αi, where ci ∈R, αi ∈Rm.

Our idea is to construct as input a clique (i.e., a fully-connected graph) with m nodes A1, . . . , Am and
distribute the r monomials (almost) evenly such that each node Ai contains r′ =
 r

m

monomials
(padding with zero monomials if m · r′ > r). Our construction makes use of a comparison gadget
which is already introduced in [19] and [83], and introduces a novel selection gadget (details in the
Appendix) which proves to be useful with the permutation equivariance restriction of MPNNs.

The two MPNNs below differ in terms of the way they compare the monomials.

Global comparison: Compare m monomials, one from each Ai, simultaneously using coordinate-
wise max aggregation. Now redistribute (using the selection gadget) the resulting O(r′) maximum
monomials (each coordinate yields one such monomial) evenly across the nodes, and recur until the
maximum across all monomials is obtained.

Local comparison: First employ a recursive procedure to compare r′ monomials assigned to each
Ai locally in order to find a maximum monomial for Ai (breaking ties arbitrarily). Now compare the
maximum monomials across the nodes using a single global comparison described above.

One limitation of both these architectures is that they can make no more than m comparisons at a
time but r can be much larger. Fortunately, a theorem due to [84], also exploited previously by [83],
motivates our construction of a Constant MPNN consisting only a constant number of layers to
represent any TSF. However, the number of learnable parameters in this MPNN can be exponential in
the worst case.

Remark 3 (Nature of MPNN and FNN). Without the equivariance constraint on the message
component of MPNN, FNN is much more efficient at constructing affine combinations or tropical
monomials. On the other hand, thanks to the parallel MP paradigm of MPNN, it is more effective
at comparing these monomials, or in other words, increases the geometric complexity of the model.
This benefit of MPNN can be seen in Theorem 4, where one has to build much bigger Φ(t)
1
and Φ(t)
2
to
reconstruct the result of the parallel ϕ(t)
1
and ϕ(t)
2 .

We therefore proceed to our final algorithm that combines the strengths of these two paradigms.
Hybrid MPNN: We first use a layer of FNN to calculate r tropical monomials and then employ a
single layer of MPNN over an r-clique with coordinate-max to learn f.

Proposition 6. There exist ReLU MPNN algorithms (Local, Global, Constant, and Hybrid that
can learn any TSF f : Rm →R with r monomials. Their respective complexity and tradeoffs are
summarized in Table 1.

We provide details about these algorithms and proofs of their correctness as well as complexity
in the Supplementary. Local architecture: Algorithm 6 and Proposition 16, Global architecture:
Algorithm 7 and Proposition 17 architecture, Constant architecture: Algorithm 8 and Proposition 18,
Hybrid architecture: Algorithm 9 and Proposition 19.

8


---Page Break---
Previously
Message layers
Feedforward layers
Learnable parameters
Deep NN in [19]
None
⌈log2(r)⌉+ 1
O(rm)
Deep NN in [83]
None
⌈log2(m)⌉+ 1
O(rm)
New (in this work)
Local (Algorithm 6)
2
⌈log2(r/m)⌉+ 5
O(rm)
Global (Algorithm 7)
⌈log2(r)⌉+ 1
3 ⌈logm(r)⌉+ 2
O(rm)
Constant (Algorithm 8)
2
7
O(mrm+2)
Hybrid (Algorithm 9)
1
1
O(rm)
Table 1: Complexity of representing any tropical signomial function (TSFs) f : Rm →R consisting
of r tropical monomials with different architectures. One more layer is required to compute any
tropical rational signomial map (TRSM). The four new methods introduced here construct a graph
(based on m and r) and leverage message passing to efficiently compare these monomials.

6
Decision boundary

Lastly, we proceed to characterizing the decision boundary of integer-weighted ReLU MPNNs for
classification. We focus on binary classification tasks with a single output to keep the exposition
transparent and explicit, albeit one can adapt our construction and analysis accordingly to accommo-
date multiple classes and outputs. We analyze the decision boundary for both graph and node/edge
predictions.

We begin with the analysis for graph classification. Recall from Section 2 that we need an additional
readout step φreadout : R|V |×dT × R|E|×d′
T →RdG prior to classification η, i.e.

χ : Rd×|V | × Rd′×|E|
φ(T )◦...◦φ(1)
−−−−−−−−→RdT ×|V | × Rd′
T ×|E|
φReadout
−−−−→RdG
η−→R.

Let γ : R →R be an injective score function. For c ∈R in Im(γ), we define the decision boundary
of χ as B = {z ∈Rm : χ(z) = γ−1(c)}, where m = d|V | + d′|E|. The tropical hypersurface T (f)
is precisely the set of points x where f is not linear; i.e., two or more monomials in f achieve the
value of f at x. We adapt a result from [19, Proposition 6.1] to arrive at the following proposition.

Proposition 7 (Decision boundary for graph classification). Let χ : Rd×|V | × Rd′×|E| →R be a
ReLU MPNN and γ : R →R be an injective score function with c in its range. Then χ can be viewed
as a tropical rational function, f ⊘g and the decision boundary B defined above divides Rm into at
most N(f) connected positive regions and at most N(g) connected negative regions. Furthermore,
B is contained in the tropical hypersurface of a specific tropical polynomial, namely,

B ⊆T (γ−1(c) ⊙g ⊕f) .

However, the characterization for node and edge classification requires a more nuanced analysis. We
focus on node classification, since the treatment for edges is analogous. In this setting the neural
network η : RdT →R is applied to the embedding of each node simultaneously, and a scoring
function γi : R →R is then employed for each node to predict its class based on its score ci.
Generally, different scoring functions γi may be applied, but in practice, we often use the same score
function for all the nodes. Viewed individually, the decision boundary for each vertex is similar
to Proposition 7; so we pursue a more interesting problem, namely, characterizing the decision
boundary of the vertices resulting from a vectorized score function Γ : R|V | →R|V |, [Γ(z)]i = γi(z).
However, we immediately hit a roadblock, since choosing a meaningful order for ci is problematic:
the product or subset order is not total whereas the lexical order relies heavily on the ordering of the
vertices and thus violates permutation equivariance of the MPNN paradigm. Therefore, we introduce
the following notion for the decision boundary.

Definition 1. The decision boundary B is defined as S|V |
i=1 Bi, where Bi = {z ∈Rm : νi(z) =
γ−1
i
(ci)}.

In order to proceed, we need to generalize the analysis to tropical hypersurfaces. Specifically, we
relate B with the tropical hypersurface of the components of the corresponding tropical rational map.
Invoking Proposition 6.1 in [19], we establish the following proposition.

Proposition 8 (Decision boundary for node classification). Let χ : Rd×|V | × Rd′×|E| →R|V | be
a ReLU MPNN and γi : R →R, i = 1, 2, ..., |V | be injective score functions with ci in their range.

9


---Page Break---
Then χ can be viewed as a tropical rational map F ⊘G and its decision boundary is contained in
the tropical hypersurface of a specific tropical polynomial, namely,

B ⊆

|V |
[

i=1
T (γ−1(ci) ⊙Gi ⊕Fi).

Conclusion, Broader Impact, and Limitations

In this paper, we characterize the class of functions learned by ReLU MPNNs through the lens of
Tropical geometry. Thus, beyond previous works that are limited to utilizing tropical geometry in the
context of ReLU FNNs, our analysis expands the scope to a widely employed class of GNNs, laying
the groundwork for further work on the connections with other machine learning models.

We provide both the lower and upper bounds for the number of linear regions of ReLU MPNNs. The
upper bound makes some simplifying assumptions; however, Theorem 4 is still general enough to
recover existing bounds for FNNs, GCNs, and provide new bounds for widely used GNN architectures
such as GraphSAGE and GIN. Our bound is rather analytical; a numerical approach to counting
the number of linear regions can be found in [85]. Adapting the method in [85] for MPNNs, and
comparing to our analytical bounds, is an interesting future direction.

We also show that the max aggregation operator is more expressive than the sum operator in terms
of geometric complexity of ReLU MPNNs (see Remark 2 on how this result contrasts with the
implications of the WL test for injective aggregation functions). We thus showcase the dependence
of expressivity on message aggregation operator, and furthermore, the connectivity of the graph
structure (max and total degree). It remains open whether spectral quantities such as the spectrum of
Laplacian have any effect on the geoemetric complexity of ReLU MPNNs.

The theoretical result on equivalence of class of functions learned by ReLU MPNNs and ReLU FNNs
is usually not reflected in practice, where ReLU MPNNs typically outperform FNNs. This motivates
our results in Section 5, where we consider several ReLU MPNN architectures to represent CPLMs
and compare them to ReLU FNNs. Remark 3, Proposition 6 and Table 1 show that ReLU MPNNs
can be more efficient in terms of both the number of learnable parameters as well as the number of
layers required in order to be able to represent the same CPLMs. We however, sidestepped other
important and practical considerations such as the difference in training time, prediction error and the
role of optimisation algorithm (e.g. SGD/Adam). Our focus here has been purely theoretical, and
we believe that (at least some of) proposed new architectures would benefit from a comprehensive
empirical evaluation. We also characterize the decision boundary for graph classification and node
classification, explaining the difference between the two.

Overall, we hope that this work fosters further research on design and analysis of modern deep
architectures through the fruitful machinery of tropical geometry.

Acknowledgments

VG acknowledges support from the Research Council of Finland for the project “Human-steered
next-generation machine learning for reviving drug design” (grant decision 342077), the Jane and
Aatos Erkko Foundation for the project “Biodesign: Use of artificial intelligence in enzyme design
for synthetic biology” (grant 7001703), and a Saab-WASP initiative (grant 411025). TAP wants to
thank Johanna Immonen, Jannis Halbey, Negar Soltan Mohammadi, Yunseon (Sunnie) Lee, Bruce
Nguyen and Nahal Mirzaie for their discussion, support and wonderful experience during his research
internship at Aalto University in 2022.

References

[1] Judea Pearl. Reverend bayes on inference engines: A distributed hierarchical approach. In
AAAI, pages 133–136, 1982.

[2] D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques.
Adaptive computation and machine learning. MIT Press, 2009.

10


---Page Break---
[3] M. Gori, G. Monfardini, and F. Scarselli. A new model for learning in graph domains. In IEEE
International Joint Conference on Neural Networks (IJCNN), 2005.

[4] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini. The graph neural
network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.

[5] Hanjun Dai, Bo Dai, and Le Song. Discriminative embeddings of latent variable models
for structured data. In Proceedings of the 33rd International Conference on International
Conference on Machine Learning - Volume 48, ICML’16, page 2702–2711. JMLR.org, 2016.

[6] Z. Zhang, F. Wu, and W. S. Lee. Factor graph neural networks. In Advances in Neural
Information Processing Systems (NeurIPS), 2020.

[7] Yuyu Zhang, Xinshi Chen, Yuan Yang, Arun Ramamurthy, Bo Li, Yuan Qi, and Le Song.
Efficient probabilistic logic reasoning with graph neural networks. In International Conference
on Learning Representations (ICLR), 2020.

[8] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, E. Monti, and M. Bronstein. Temporal
graph networks for deep learning on dynamic graphs. In ICML 2020 Workshop on Graph
Representation Learning, 2020.

[9] Y. Wang, Y. Chang, Y. Liu, J. Leskovec, and P. Li. Inductive representation learning in temporal
networks via causal anonymous walks. In International Conference on Learning Representations
(ICLR), 2021.

[10] Amauri Souza, Diego Mesquita, Samuel Kaski, and Vikas Garg. Provably expressive temporal
graph networks. In Neural Information Processing Systems (NeurIPS), 2022.

[11] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for
quantum chemistry. In International Conference on Machine Learning (ICML), 2017.

[12] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks.
In International Conference on Learning Representations (ICLR), 2017.

[13] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In
Advances in Neural Information Processing Systems (NeurIPS), 2017.

[14] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph Attention
Networks. In International Conference on Learning Representations (ICLR), 2018.

[15] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In
International Conference on Learning Representations (ICLR), 2019.

[16] B. Weisfeiler and A. A. Leman. A reduction of a graph to a canonical form and an algebra
arising during this reduction. Nauchno-Technicheskaya Informatsia, 2(9):12–16, 1968.

[17] C. Morris, M. Ritzert, M. Fey, W. L. Hamilton, J. E. Lenssen, G. Rattan, and M. Grohe.
Weisfeiler and leman go neural: Higher-order graph neural networks. In AAAI Conference on
Artificial Intelligence (AAAI), 2019.

[18] H. Maron, H. Ben-Hamu, H. Serviansky, and Y. Lipman. Provably powerful graph networks. In
Advances in Neural Information Processing Systems (NeurIPS), 2019.

[19] Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical geometry of deep neural networks.
In Jennifer G. Dy and Andreas Krause, editors, International Conference on Machine Learning
(ICML), volume 80 of Proceedings of Machine Learning Research, pages 5819–5827. PMLR,
2018.

[20] Stefanie Jegelka. Theory of graph neural networks: Representation and learning, 2022.

[21] Hao Chen, Yu Guang Wang, and Huan Xiong. Lower and upper bounds for numbers of linear
regions of graph convolutional networks, 2022.

[22] D. Maclagan and B. Sturmfels. Introduction to Tropical Geometry. Graduate Studies in
Mathematics. American Mathematical Society, 2015.

11


---Page Break---
[23] M. Joswig. Essentials of Tropical Combinatorics. Graduate Studies in Mathematics. American
Mathematical Society, 2021.

[24] Vasileios Charisopoulos and Petros Maragos. A tropical approach to neural networks with
piecewise linear activations. ArXiv, abs/1805.08749, 2018.

[25] Guido Montúfar, Yue Ren, and Leon Zhang. Sharp bounds for the number of regions of maxout
networks and vertices of minkowski sums. SIAM Journal on Applied Algebra and Geometry,
6(4):618–649, 2022.

[26] Motasem Alfarra, Adel Bibi, Hasan Hammoud, Mohamed Gaafar, and Bernard Ghanem. On the
decision boundaries of neural networks: A tropical geometry perspective. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 45(4):5027–5037, 2023.

[27] Georgios Smyrnis and Petros Maragos. Tropical polynomial division and neural networks,
2019.

[28] Marie-Charlotte Brandenburg, Georg Loho, and Guido Montúfar. The real tropical geometry of
neural networks, 2024.

[29] Matthew Trager, Kathlén Kohn, and Joan Bruna. Pure and spurious critical points: a geometric
study of linear networks, 2020.

[30] Dhagash Mehta, Tianran Chen, Tingting Tang, and Jonathan D. Hauenstein. The loss surface
of deep linear networks viewed through the algebraic geometry lens. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 44(9):5664–5680, 2022.

[31] J. Elisenda Grigsby and Kathryn Lindsey. On transversality of bent hyperplane arrangements
and the topological expressiveness of relu neural networks. SIAM Journal on Applied Algebra
and Geometry, 6(2):216–242, 2022.

[32] Francis Williams, Matthew Trager, Daniele Panozzo, Cláudio T. Silva, Denis Zorin, and Joan
Bruna. Gradient dynamics of shallow univariate relu networks. In Hanna M. Wallach, Hugo
Larochelle, Alina Beygelzimer, Florence d’Alché Buc, Edward A. Fox, and Roman Garnett,
editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural
Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC,
Canada, pages 8376–8385, 2019.

[33] Petros Maragos, Vasileios Charisopoulos, and Emmanouil Theodosis. Tropical geometry and
machine learning. Proceedings of the IEEE, 109(5):728–755, 2021.

[34] Christopher Morris, Yaron Lipman, Haggai Maron, Bastian Rieck, Nils M. Kriege, Martin
Grohe, Matthias Fey, and Karsten Borgwardt. Weisfeiler and leman go machine learning: the
story so far. J. Mach. Learn. Res., 24(1), mar 2024.

[35] Bohang Zhang, Jingchu Gai, Yiheng Du, Qiwei Ye, Di He, and Liwei Wang. Beyond weisfeiler-
lehman: A quantitative framework for GNN expressiveness. In The Twelfth International
Conference on Learning Representations, 2024.

[36] Chaitanya K. Joshi, Cristian Bodnar, Simon V. Mathis, Taco Cohen, and Pietro Liò. On the
expressive power of geometric graph neural networks. In International Conference on Machine
Learning, 2023.

[37] Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Liò, Guido F Montufar,
and Michael Bronstein. Weisfeiler and lehman go cellular: Cw networks. In M. Ranzato,
A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural
Information Processing Systems, volume 34, pages 2625–2640. Curran Associates, Inc., 2021.

[38] B. Rieck, C. Bock, and K. Borgwardt. A persistent weisfeiler-lehman procedure for graph
classification. In International Conference on Machine Learning (ICML), 2019.

[39] Xiyuan Wang and Muhan Zhang. How powerful are spectral graph neural networks. ICML,
2022.

12


---Page Break---
[40] Johannes Gasteiger, Florian Becker, and Stephan Günnemann. Gemnet: Universal direc-
tional graph neural networks for molecules. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S.
Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems,
volume 34, pages 6790–6802. Curran Associates, Inc., 2021.

[41] Yi Liu, Limei Wang, Meng Liu, Yuchao Lin, Xuan Zhang, Bora Oztekin, and Shuiwang Ji.
Spherical message passing for 3d molecular graphs. In International Conference on Learning
Representations, 2022.

[42] Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai,
Gopinath Balamurugan, Michael M. Bronstein, and Haggai Maron. Equivariant subgraph
aggregation networks. In International Conference on Learning Representations, 2022.

[43] Nicolas Keriven and Gabriel Peyré. Universal invariant and equivariant graph neural networks.
In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors,
Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[44] R. Sato, M. Yamada, and H. Kashima. Random features strengthen graph neural networks. In
SIAM International Conference on Data Mining (SDM), 2021.

[45] M. Carrière, F. Chazal, Y. Ike, T. Lacombe, M. Royer, and Y. Umeda. PersLay: A neural
network layer for persistence diagrams and new graph topological signatures. In International
Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

[46] Yuzhou Chen, Baris Coskunuzer, and Yulia Gel. Topological relational learning on graphs.
In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors,
Advances in Neural Information Processing Systems, volume 34, pages 27029–27042. Curran
Associates, Inc., 2021.

[47] Max Horn, Edward De Brouwer, Michael Moor, Yves Moreau, Bastian Rieck, and Karsten
Borgwardt. Topological graph neural networks. In International Conference on Learning
Representations, 2022.

[48] Omri Puny, Derek Lim, Bobak T. Kiani, Haggai Maron, and Yaron Lipman. Equivariant
polynomials for graph neural networks. In Proceedings of the 40th International Conference on
Machine Learning, ICML’23. JMLR.org, 2023.

[49] Z. Chen, L. Chen, S. Villar, and J. Bruna. Can graph neural networks count substructures? In
Advances in Neural Information Processing Systems (NeurIPS), 2020.

[50] Uri Alon and Eran Yahav.
On the bottleneck of graph neural networks and its practical
implications. In International Conference on Learning Representations, 2021.

[51] D. Kreuzer, D. Beaini, W. L. Hamilton, V. Letourneau, and P. Tossou. Rethinking graph
transformers with spectral attention. In Advances in Neural Information Processing Systems
(NeurIPS), 2021.

[52] Ben Chamberlain, James Rowbottom, Maria I Gorinova, Michael Bronstein, Stefan Webb,
and Emanuele Rossi. Grand: Graph neural diffusion. In Marina Meila and Tong Zhang,
editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of
Proceedings of Machine Learning Research, pages 1407–1418. PMLR, 18–24 Jul 2021.

[53] Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and
Michael M. Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature.
In International Conference on Learning Representations, 2022.

[54] Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. Is homophily a necessity for graph neural
networks? In International Conference on Learning Representations, 2022.

[55] Yuelin Wang, Kai Yi, Xinliang Liu, Yu Guang Wang, and Shi Jin. ACMP: allen-cahn message
passing with attractive and repulsive forces for graph neural networks. In The Eleventh Interna-
tional Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.
OpenReview.net, 2023.

13


---Page Break---
[56] V. Garg, S. Jegelka, and T. Jaakkola. Generalization and representational limits of graph neural
networks. In International Conference on Machine Learning (ICML), 2020.

[57] Nima Dehmamy, Albert-Laszlo Barabasi, and Rose Yu. Understanding the representation
power of graph neural networks in learning graph topology. In H. Wallach, H. Larochelle,
A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information
Processing Systems, volume 32. Curran Associates, Inc., 2019.

[58] Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Lio, and
Michael M. Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and
oversmoothing in GNNs. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun
Cho, editors, Advances in Neural Information Processing Systems, 2022.

[59] Bohang Zhang, Shengjie Luo, Liwei Wang, and Di He. Rethinking the expressive power
of GNNs via graph biconnectivity. In The Eleventh International Conference on Learning
Representations, 2023.

[60] A. Loukas. What graph neural networks cannot learn: depth vs width. In International
Conference on Learning Representations (ICLR), 2020.

[61] A. Loukas. How hard is to distinguish graphs with graph neural networks? In Advances in
Neural Information Processing Systems (NeurIPS), 2020.

[62] S. S. Du, K. Hou, R. Salakhutdinov, B. Póczos, R. Wang, and K. Xu. Graph neural tangent
kernel: Fusing graph neural networks with graph kernels. In Advances in Neural Information
Processing Systems (NeurIPS), 2019.

[63] Sanjukta Krishnagopal and Luana Ruiz. Graph neural tangent kernel: Convergence on large
graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan
Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML
2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine
Learning Research, pages 17827–17841. PMLR, 2023.

[64] Behrooz Tahmasebi, Derek Lim, and Stefanie Jegelka. The power of recursion in graph neural
networks for counting substructures. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de
Meent, editors, Proceedings of The 26th International Conference on Artificial Intelligence
and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 11023–11042.
PMLR, 25–27 Apr 2023.

[65] Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.
Graph neural networks with learnable structural and positional representations. In International
Conference on Learning Representations, 2022.

[66] S. Verma and Z.-L. Zhang. Stability and generalization of graph convolutional neural networks.
In International Conference on Knowledge Discovery & Data Mining (KDD), 2019.

[67] R. Liao, R. Urtasun, and R. Zemel. A PAC-bayesian approach to generalization bounds for
graph neural networks. In International Conference on Learning Representations (ICLR), 2021.

[68] K. Xu, J. Li, M. Zhang, S. S. Du, K.-I. Kawarabayashi, and S. Jegelka. What can neural
networks reason about? In International Conference on Learning Representations (ICLR),
2020.

[69] Sohir Maskey, Ron Levie, Yunseok Lee, and Gitta Kutyniok. Generalization analysis of message
passing neural networks on large random graphs. In Alice H. Oh, Alekh Agarwal, Danielle
Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems,
2022.

[70] Pascal Esser, Leena Chennuru Vankadara, and Debarghya Ghoshdastidar. Learning theory can
(sometimes) explain generalisation in graph neural networks. In M. Ranzato, A. Beygelzimer,
Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information
Processing Systems, volume 34, pages 27043–27056. Curran Associates, Inc., 2021.

14


---Page Break---
[71] Haotian Ju, Dongyue Li, Aneesh Sharma, and Hongyang R Zhang. Generalization in graph
neural networks: Improved pac-bayesian bounds on graph diffusion. Artificial Intelligence and
Statistics, 2023.

[72] Yangze Zhou, Gitta Kutyniok, and Bruno Ribeiro. OOD link prediction generalization capabili-
ties of message-passing GNNs in larger test graphs. In Alice H. Oh, Alekh Agarwal, Danielle
Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems,
2022.

[73] K. Xu, M. Zhang, J. Li, S. S. Du, K.-I. Kawarabayashi, and S. Jegelka. How neural networks
extrapolate: From feedforward to graph neural networks. In International Conference on
Learning Representations (ICLR), 2021.

[74] Beatrice Bevilacqua, Yangze Zhou, and Bruno Ribeiro. Size-invariant graph representations for
graph classification extrapolations. In Marina Meila and Tong Zhang, editors, International
Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning
Research, pages 837–851. PMLR, 2021.

[75] Andrew Dudzik and Petar Veliˇckovi´c. Graph neural networks are dynamic programmers. In
ICLR 2022 Workshop on Geometrical and Topological Representation Learning, 2022.

[76] Guido F Montufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of
linear regions of deep neural networks. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence,
and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27.
Curran Associates, Inc., 2014.

[77] Boris Hanin and David Rolnick. Complexity of linear regions in deep networks. In Kamalika
Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference
on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97
of Proceedings of Machine Learning Research, pages 2596–2604. PMLR, 2019.

[78] Huan Xiong, Lei Huang, Mengyang Yu, Li Liu, Fan Zhu, and Ling Shao. On the number
of linear regions of convolutional neural networks. In Proceedings of the 37th International
Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of
Proceedings of Machine Learning Research, pages 10514–10523. PMLR, 2020.

[79] Itay Safran, Gal Vardi, and Jason D. Lee. On the effective number of linear regions in shallow
univariate reLU networks: Convergence guarantees and implicit bias. In Alice H. Oh, Alekh
Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information
Processing Systems, 2022.

[80] Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the
expressive power of deep neural networks. In Doina Precup and Yee Whye Teh, editors, Pro-
ceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings
of Machine Learning Research, pages 2847–2854. PMLR, 06–11 Aug 2017.

[81] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov,
and Alexander J Smola. Deep sets. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach,
R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 30. Curran Associates, Inc., 2017.

[82] Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise
linear neural networks. Journal of Computational and Applied Mathematics, 441:115667, 2024.

[83] Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee. Understanding deep
neural networks with rectified linear units. In International Conference on Learning Represen-
tations, 2018.

[84] Shuning Wang and Xusheng Sun. Generalization of hinging hyperplanes. IEEE Transactions
on Information Theory, 51:4425–4431, 12 2005.

[85] Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and counting
linear regions of deep neural networks. In Jennifer Dy and Andreas Krause, editors, Proceedings
of the 35th International Conference on Machine Learning, volume 80 of Proceedings of
Machine Learning Research, pages 4558–4566. PMLR, 10–15 Jul 2018.

15


---Page Break---
Technical Appendix - Supplementary Material

We now describe in detail all the algorithms introduced in the main paper, as well as proofs for all
our results. We begin with the results from the section on Tropical algebra.

Tropical algebra

Lemma 2. Every ReLU FNN ν : Rm →Rp can be represented by a ReLU MPNN χ : R1×m →
R1×p .

Proof. This is obvious when we set G to be a one-clique (i.e., one vertex graph with a self-edge) and
ϕ(1)
2
= ν, with max or sum aggregation, ϕ(1)
1
= id, W (1)
self = 0 and W (1)
neigh = Idp×p.

Lemma 3. Any ReLU MPNN χ : R|V |×d × R|E|×d′ →R|V |×dout × R|E|×d′
out can be expressed as a
TRSM F ⊘G : Rm →Rp, with m = |V |d + |E|d′ and p = |V |dout + |E|d′
out.

Proof. To prove the proposition, we will show that each layer φ(t) is a TRSM on its input, thus
χ is a TRSM as well. ϕ(t)
2
: Rdt−1 × Rdt−1 × Rd′
t−1 × Rd′
t−1 →R ˜
dt and ϕ(t)
1
: R ˜
dt →R ¯dt are
ReLU FNNs, hence by [19, Proposition 5.6], they are TRSMs. The aggregation operators (sum and
coordinate-wise max for our purpose) are tropical operators pertaining to tropical multiplication
and addition respectively. Thus, each component of φ(t)
Agg is a TRSF. On the other hand, σ(t)
Update is a

ReLU activation, the update function φ(t)
Update, and thus, each MPNN layer φ(t) is also a TRSM on its
input.

Proof of Proposition 1. Note that [19, Corollary 5.3] established the equivalence of (2), (3), (4).
Lemma 2 and Lemma 3, imply (4) ⇒(1), and (1) ⇒(2) respectively and hence we are done.

Bounds on geometric complexity

We here provide more detailed analysis for the Geometric complexity and the proof of Theorem 3 and
Theorem 4. To simplify the analysis, we will assume that we have no edge embedding throughout
MPNN layers - the analysis for the most general case follows the same idea. In this analysis, we also
assume that inputs of the model follows the same graph G. Let S and D be the maximum and total
degree of all vertices of G.

We will consider the input H(t−1) of each message passing layer φ(t) as a stacked vector of edge node
embedding h(t−1)
v
. As we do not want to calculate unused messages between non-adjacency nodes,
with the vectorized form (instead of matrix form), we could consider each degree separately and thus
reduce the dimension of intermediate layers from S|V | to D. Forming the vectorized form requires
assumption of an order between the nodes and the induced lexical order for the edges introduced in
Section 2. However, we can neutralize this effect by multiplying with the appropriate permutation
matrix P. We now outline the changes happen in each layer.

1. The vector H(t) ∈R|V |dt will be the input to φ(t) : R|V |dt−1 →R|V |dt (H(0) = X).
2. We then apply ϕ(t)
2
: R|V |dt−1 × R|V |dt−1 →R|V | ˜dt to every pair of adjacent node embed-
ding Ai and Aj to obtain the message from node Ai to node Aj. The idea is to form a big
FNN Φ(t)
2
applied on the whole vector H(t−1) to obtain the output X(t) ∈RD ˜dt. The FNN
Φ(t)
2
in general depends depends on D and connectedness of the graph.
3. After that, we perform the operator □(t) = □(t)
u∈N(v) (sum or coordinate-wise max/min)

of all the neighboring messages to obtain Y (t) ∈R|V | ˜dt. We will see later that there
is a significant difference in the two cases of aggregation operator and suggest that the
coordinate-wise max/min could indeed increase our geometric complexity.
4. Analogously, we then form Z(t) ∈R|V | ¯dt by forming a bigger network Φ(t)
1
(based on
ϕ(t)
1
: R ˜dt →R ¯dt), applied to Y (t), i.e. Z(t) = Φ(t)
1 (Y (t)).

16


---Page Break---
5. We combine step 2, 3 and 4 to form a big FNN Φ(t)
Agg applied to H(t−1)

6. Lastly, we obtain H(t) ∈R|V |×dt by performing the φ(t)
Update.

We proceed with the first step. The algorithm 1 shows us how to build Φ(t)
2
from ϕ(t)
2
by replicating
the neural network ϕ(t)
2
for neighboring nodes and putting 0 for the nodes that are not adjacent.
Without no edge embeddings, our ϕ(t)
2
only depends on the node embedding h(t−1)
Ai
and h(t−1)
Aj
.

Our notations from now on always indicate that ϕ(t)
1 , ϕ(t)
2
are applied to each (pair of) embeddings
whereas Φ(t)
1 , Φ(t)
2
(Φ(t)
3 ) are applied to the corresponding stacked vector as a whole.

Algorithm 1 Building Φ(t)
2
Input: Stacked vector H(t−1) ∈R|V |dt−1 of embedding of all vertices.
We will only describe the construction for weight, the bias could be done analogously. Let W t,l
2
and
nt,l
2 be the weight and number of output nodes of the ℓ-th layer of ϕ(t)
2
respectively (nt,0
2
= dt−1).

Thus, W t,l
2
∈Rnt,l
2 ×nt,l−1
2
. For the first layer l = 1, let W t,1
2
=
h
W t,1
2,1, W t,1
2,2
i
, where W t,1
2,1, W t,1
2,2 ∈

Rnt,1
2
×dt−1 correspond to the weight of the first dt−1 nodes (of the embedding h(t−1)
Ai
) and last dt−1
nodes (of the h(t−1)
Aj
) respectively.

We now define the weight for Φ(t)
2 . In the first layer of Φ(t)
2 , ˜W t,1
2
∈RDnt(1)
2
×|V |dt−1 by breaking it
into blocks [ ˜W t,1
2 ]κ,κ′ of size nt,1
2
× dt−1. For each edge κ-th (according to the ordering of edges),
assume it is (Ai, Aj), then, we set

[ ˜W t,1
2 ]κ,κ′ =












W t,1
2,1 + W t,1
2,2
if κ′ = i = j
W t,1
2,1
if κ′ = i
W t,1
2,2
if κ′ = j
0
otherwise.

(3)

The remaining layers ℓare easy to build:
let
˜W t,l
2
∈
RDnt,l
2 ×Dnt,l−1
2
and
˜W t,l
2
=



W t,l
2
0
...
0
0
W t,l
2
...
0
0
0
...
W t,l
2



and applied the activation accordingly.

Proposition 9. ReLU FNN Φ(t)
2
in Algorithm 1, applied to H(t−1), will result in the same output as
we apply ϕ(t)
2
separately for neighboring nodes h(t−1)
Ai
, h(t−1)
Aj
and combine the result.

Proof. We first assume that ϕ(t)
2
has only one layer and the action of ReLU activation is applied
coordinate-wise, we can assume that

ϕ(t)
2 (h(t−1)
Ai
, h(t−1)
Aj
) = W (t),1
2

"
h(t−1)
Ai
h(t−1)
Aj

#

= W t,1
2,1h(t−1)
Ai
+ W (t,1
2,2 h(t−1)
Aj
.

There are D edges in total, thus X(t) =




X(t)
1
...
X(t)
D



∈RDnt,2
2
and H(t−1) =




h(t−1)
A1...
h(t−1)
A|V |



. For the κ-th

edge (Ai, Aj), we want X(t)
κ
= ϕ(t)
2 (h(t−1)
Ai
, h(t−1)
Aj
), the formula (3) will yield the desired result.

Then applying the following layers Φ(t)
2
is equivalent to applying the corresponding layers of ϕ(t)
2
on
each result.

Proposition 10 (The matrix to take the sum). If □(t)
u∈N(v) is sum, then

Y (t) = MX(t),

17


---Page Break---
where M is a matrix that generally depends on the adjacency matrix A of G. Thus, it does not change
the linear/convex degree and its effect can be absorbed by the first layer of Φ(t)
1 .

Proof. We first build M ∈R|V | ˜dt×D ˜dt, by breaking it into blocks of size ˜dt × ˜dt. Then each block

Mκ,κ′ =
Id ˜dt× ˜dt
if edge κ′ = (Aκ, Aj) ∈E
0
otherwise
.

It is easy to see that Y (t) = MX(t), since the matrix M is the same as the adjacency matrix A of G
replacing 1 with Id ˜dt× ˜dt.

On the other hand, when the operator □(t) is coordinate-wise max-min, the analysis becomes a little
bit more complicated. In particular, we can create a neural network Φ(t)
3
to represent its effect. In the
Algorithm 2, we want to make use of the comparing architecture in [83]. A minor blockage is the
appearance of the weight 1/2 in the comparing layer. Fortunately, by scaling the input by a factor of
1/2 (which does not change the number of linear regions) and scaling the weights by a factor of 2,
we could solve this problem. We note that Φ(t)
3
depends heavily on the graph G.

Algorithm 2 Building Φ(t)
3
Input: X(t) ∈RD ˜dt.

We first scale all entries in the input by 1/2. Then we create a FNN that resembles the construction in
[83], however, we will not compare two consecutive nodes, but the comparison is done according to
the edges of the graph.

Proposition 11. If □(t) is coordinate-wise max/min, then

Y (t) = Φ(t)
3 (X(t)),

where Φ(t)
3
is built according to Algorithm 2.
Remark 4. Here, we surmise that if we have a coordinate-wise max/min aggregation operator, the
geometric complexity of our model grows exponentially with D and polynomially with S.

After that, we continue with step 4, where we apply the neural network ϕ(t)
1
to form the transformation
of the aggregated message. Similar to step 2, we would want to build Φ(t)
1
that is applied on the
whole Y (t) as a vector. However, this is much easier to form ϕ(t)
1
than that of Φ(t)
2
as

([Φ(t)
1 (Y (t))⊤](i−1) ¯dt+1,...,i ¯dt) = ϕ(t)
1 (Y (t)
(i−1) ˜dt+1,...,i ˜dt)⊤.
(4)

Algorithm 3 Building Φ(t)
1
Input: Y (t) ∈R|V | ˜dt.

Let W t,l
1
∈Rnt,l+1
1
×nt,l
1 and ˜W t,l
1
∈R|V |nt,l+1
1
×|V |nt,l
1 be the weight of the ℓ-th layer of ϕ(t)
2 , Φ(t)
2

respectively. Then ˜W t,l
1
=




W t,l
1
0
...
0
0
W t,l
1
...
0
0
0
...
W t,l
1



and applied the activation accordingly.

Obviously, since each component is applied separately, we have the following proposition.
Proposition 12.

([Φ(t)
1 (Y (t))⊤](i−1) ¯dt+1,...,i ¯dt) = ϕ(t)
1 (Y (t)
(i−1) ˜dt+1,...,i ˜dt)⊤.
(5)

18


---Page Break---
Lower bound on geometric complexity

Proof of Theorem 3. By Proposition 9, Proposition 13 and Proposition 11, we note that each Ag-
gregation step φ(t)
Agg can be written as a FNNs Φ(t)
Agg applied to H(t). Note that □(t) = P can be

absorbed by the first layer of Φ(t)
1
(constructed in Algorithm 3). On the other hand, □(t) = max can
be seen as a max-out layer with rank S). Thus, by [76, Theorem 8 proof], it can identify S regions
of the input. Then if φ(t)
Update(hv, mv) = mv then we can write H(t) = Φ(t)
Agg(H(t−1)). Thus, by [76,
Theorem 4], we have the maximum number of linear region of functions computed by any ReLU
MPNN is lower bounded by

St0

QT
t=1

QL(t)
1
l=1 nt,l
1,d0
QL(t)
2
l=1 nt,l
2,d0



n
T,L(T )
1
1,d0

d0
X

j=0

dT
j



,

Upper bound on geometric complexity

Remark 5. If the neural network ϕ(t)
2
has nt,l
2 output nodes in its ℓ-th hidden layer, then the neural
network Φ(t)
2
have Dnt,l
2 nodes in its ℓ-th layer. Therefore, the number of output nodes and hidden
layer nodes depend on the connectivity of the network.

Proposition 13. If the neural network ϕ(t)
1 , ϕ(t)
2
has nt,l
1 , nt,l
2 number of nodes in its ℓ-th hidden layer
then the neural network Φ(t)
1 , Φ(t)
2
have |V |nt,l
1 , |D|nt,l
1 number of nodes in its ℓ-th layer. Thus, if

• nt,l
2 ≥
D
|V |dt for all l = 1, ..., L(t)
2 ;

• nt,l
1 ≥dt for all l = 1, ..., L(t)
1 ;

then applying [19, Theorem 6.3], we obtain an upper bound for convex degree:

Nc(Φ(t)
Agg) ≤

L(t)
2 −1
Y

l=1

|V |dt
X

i=0

Dnt,l
2
i


Nc(□(t))

L(t)
1 −1
Y

l=1

|V |dt
X

i=0

|V |nt,l
1
i


(6)

convex linear regions, where

N(□(t)) =

(
1
if □(t) = P

O(SD ˜dt)
if □(t) = max .
(7)

Proof. If □(t) is coordinate-wise max/min, then Φ(t)
3
has ⌈log2 S⌉+ 1 layers. For ease of analysis,
we can assume each hidden layer has D ˜dt nodes (this only increase the convex/linear degree) except
for the first layer with at most 2D ˜dt nodes, thus from [19, Theorem 6.3], we have

Nc(Φ(t)
3 ) ≤





D ˜dt
X

i=0

2D ˜dt
i





⌈log2 S⌉
Y

l=1

D ˜dt
X

i=0

D ˜dt
i



= 22D ˜dt−1 · 2D ˜dt⌈log2 S⌉,

as Pn
i=0
 2n
i

= 22n−1. Using Propositions 9, 11, and 13, we can write the result of φ(t)
Agg (i.e. Z(t))

as a result of a combined FNN Φ(t)
1
◦Φ(t)
3
◦Φ(t)
2
applied to H(t−1). Then applying [19, Theorem
6.3] yields the stated result.

Thus, we are left with the final step in φ(t)
Update. If we stack the two vector together, which is now of
dimension |V | ¯dt + |V |dt, we can apply [19, Lemma D.4] to prove the following Corollary.

19


---Page Break---
Corollary 5. Let φ(t)
Update = σ(t)
Update ◦ρ(t)
Update : R|V |( ¯dt+dt) →R|V |dt. If |V |( ¯dt + dt) ≤|V |(dt), i.e.
¯dt + dt ≤dt, then

Nc(σUpdate ◦ρ|( ¯dt + dt)|V |) ≤

|V |( ¯dt+dt)
X

i=0

|V |dt
i


.

The following results together yield an important Proposition that allows us to analyze each component
of φ(t) separately before combining them. The following lemma is trivial, but we provide a proof for
completeness.

Lemma 4. Let F ∈Rat(m, p) and m′ ≤m, Then

Nc(F|m′) ≤Nc(F)
(8)

Proof. We invoke the fact that if F is an affine transformation of Rn and A ⊂Rn is convex, then the
image F[A] is also convex. Thus, a convex linear region in Nc(F|d) is still a convex and linear region
in Nc(F), and vice versa. On the other hand, some convex linear regions in Rn do not intersect the
affine space, resulting in the inequality.

We first recall [19, Theorem D.3]: Let F, G in Rat(m, p) and Rat(m′, m). Let H = (h1, ..., hp) ∈
Rat(m′, p) defined by hi := fi ◦G for i = 1, .., p. Then

N(H) ≤Nc(H) ≤Nc(F|m′) · Nc(G).

Proof of Proposition 2. We note that

χ(t) = φ(t) ◦χ(t−1),
(9)

thus using [19, Theorem D.3], we have

Nc(χ(t)) ≤Nc(φ(t)||V |dt−1) Nc(χ(t−1)).
(10)

Note that φ(t)(H(t−1)) = φ(t)
Update(φ(t)
Agg(H(t−1)), H(t−1)) thus

Nc(φ(t)||V |dt) ≤Nc(φ(t)) ≤Nc(φ(t)
Update|(dt + dt)|V |) Nc(φ(t)
Agg).
(11)

We can further break the terms Nc(φ(t)
Agg) down by noticing that φ(t)
Agg = Φ(t)
1 ◦□(t) ◦Φ(t)
2 , applying
in [19, Theorem D.3] multiple times and Lemma 4.

Thus, bringing all the corollary together and apply [19, Theorem 6.3] recursively, we obtain the
following bound for the geometric complexity:

Proof of Theorem 4. The theorem follows from Proposition 2, which tells us that we could analyze
the model in steps. Thus, bringing the pieces together, we have

T
Y

t=1





L(t)
1 −1
Y

l=1

|V | ˜dt
X

i=0

|V |nt,l
1
i





|
{z
}

from Φ(t)
1





L(t)
2 −1
Y

l=1

|V |×dt−1
X

i=0

Dnt,l
2
i





|
{z
}

from Φ(t)
2





|V |( ¯dt+dt−1)
X

i=0

|V |dt
i





|
{z
}

from φ(t)
Update

Nc(□(t)),

where Nc(□(t)) =

(
1
if □(t) is sum
1
2(8S)D ˜dt
if □(t) is coordinate-wise max/min .

20


---Page Break---
6.1
Consequences of Theorem 4

We first recover the upper bound for FNNs and GCNs (with ReLU activations and integer-weights)
established in [19, Theorem 6] and [21, Theorem 4] respectively as special cases. The one for FNN
is straightforward. On the other hand, the bound for GCNs requires slightly more work.

Proof of Corollary 2. Firstly, note that GCN can be modelled by setting ϕ(t)
1
= Id, □(t) = sum, and
φ(t)
Update = Z(t), where Z(t) is just a stacked vector of m(t)
Ai. With this simplification, we can in fact
have a stronger version of Proposition 2 (going down to input layer):

Nc(χ) ≤

T
Y

t=1
Nc(□(t) ◦Φ(t)
2 ||V |d0)

≤

T
Y

t=1





|V |d
X

i=0

|V |dt
i




(by Proposition 10, □(t) is just linear transformation).

Proof of Corollary 3. In both case, we note that ϕ(t)
1
= id, thus, we can forget the contribution of
Φ1. If the aggregation operator is mean, we can substitute □(t) = P, then ϕ(t)
2
= id and φUpdate
stays the same. Thus, we have

Nc(φt) ≤

2|V |dt−1
X

i=0

|V |dt
i


(12)

If the aggregation operator is pooling, then ϕ(t)
2
is a one-layer FNN, □(t) = max and φUpdate stays
the same. Thus, we have

Nc(φt) ≤





|V | ˜dt−1
X

i=0

|D| ˜dt
i









|V |( ˜dt+dt−1)
X

i=0

|V |dt
i



.
(13)

Proof of Corollary 4. According to [15, Equation 4.1], by adding a self-edge to each node in the
graph, t-th layer of GIN can be written as □(t) = P (or affine transformation - which does not
change the geometric complexity), ϕ(t)
2
= id and ϕ(t)
1
= MLP indicated, thus

Nc(χ) ≤

T
Y

t=1




L(t)−1
Y

l=1

|V |dt
X

i=0

|V |nt,l

i



.

New ReLU MPNNs architectures and complexity tradeoffs

In most of the following architectures, the aggregated message has the following form,

m(t)
Ai = ϕ(t)
1 (□(t)
Aj∈N(Ai)ϕ(t)
2 (h(t−1)
Aj
)),
(14)

i.e., the aggregated message to node Ai depends only on the neighboring nodes Aj (and not Ai itself).

Lemma 5. Let F ⊘G : Rm →Rp be a TRSM. Suppose each component Fi of F can be represented
by an FNN/MPNN with LFi layers, and Gi of G by an FNN/MPNN with LGi layers. Then F ⊘G
can be represented as an L-layer FNN/MPNN with L ≤maxp
i=1{max{LFi, LGi}} + 1.

21


---Page Break---
Proof. The main idea underlying our proof here is to exploit parallelization. Firstly, for FNNs, we can
compute all Fi and Gi in parallel using maxn
i=1{max{LFi, LGi}} layers: we set the block matrix
for weights between the different components to zero (for the components that require fewer layers,
we can add additional dummy layers with their weights set to the identity matrix Id). We then need
just one additional layer to compute the difference Fi ⊘Gi, again in parallel, for all i = 1, .., n.
Similarly, for MPNNs, we construct Fi and Gi in parallel following appropriate stacking of the
weights of the components. In this case, each additional layer t (for the components with fewer
layers) is treated as follows: we set the weights for ϕ(t)
2 , ϕ(t)
1
to zero and W (t)
self to identity. Lastly, to
compute the difference Fi and Gi, we just need one additional MPNN layer where aggregation part
is 0 (ϕ(L)
2
, ϕ(L)
1
= 0; □(L) is sum or coordinate-wise max), and W (L)
self = [Idp×p
−Idp×p].

We provide an overview and detailed algorithms for the gadgets here. The broadcast gadget sets up
an MPNN by replicating its input across nodes of a fully connected graph. It also endows each node
with its ROE. The selection and comparison gadgets are implemented as FFNs: the former is used
for partitioning the monomials across the nodes, whereas the latter determines the larger of its two
input monomials.

6.2
Broadcast gadget

Broadcast gadget. Given an input vector x ∈Rm, we first construct a fully connected graph with m
vertices A1, ..., Am. The feature vector for A1 is constructed as follows: first m coordinates comprise
x, the next coordinate is set to 1 (for bias), and the last m coordinates are set to ROE 1m −e1. The
feature vectors for all other nodes Ai consist of m + 1 zeros followed by 1m −ei. A parameterized
message passing layer with weights determined by a target TSF f is then learned to compute all the r
monomials p1, p2, . . . , pr of f. We provide all the details, including the operators in Algorithm 4. It
can be shown that the embedding of node Ai after broadcasting is

p1, p2, ..., pr, (1m −ei)⊤⊤.

Algorithm 4 Broadcast component: MPNN layer
Build an m-clique with vertices A1, ..., Am and bidirectional edges, and introduce loops with m
self-edges from each vertex Ai to itself.

Prepare node embeddings. For A1, we set its first m coordinates to the input x, the next coordinate to
1 for bias, and then the final m coordinates set to its ROE 1m −e1. For other nodes Ai, we instead
have a vector with first m coordinates set to 0, followed by a single coordinate 0 for bias and finally
their respective ROE 1m −ei.

Finally, we construct a layer of message passing. Specifically, ϕ(1)
2
is a 1-layer FNN with identity

activation, no bias, and has the form of Equation 14. Its weight matrix is

C ∈Rr×(m+1)
0r×m
0m×(m+1)
0m×m


,

where each row Ci,: = [αi, ci] ∈Rm+1.
We use the sum aggregation operator and set

ϕ(1)
1
= Id.
Moreover, φ(1)
Update has identity activation, W (1)
self
=

0r×(m+1)
0r×m
0m×(m+1)
Idm×m


, and

W (1)
neigh = Id(r+m)×(r+m).

The following result follows immediately from Algorithm 4.

Proposition 14.
The embedding of vertex Ai after broadcasting with Algorithm 4 is

p1, p2, ..., pr, (1m −ei)⊤⊤, where pi is the ith affine combination (or tropical monomial) of f.

Furthermore, the Broadcast Algorithm 4 uses in total 2 FNN layers across 1 layer of message passing
and (2m + 1)(r + m) parameters of which (m + 1)(r) = O(rm) are learnable.

6.3
Selection gadget

We now describe the selection gadget that we invoke to distribute r monomials (almost) evenly among
the nodes such that the node Ai gets monomials p¯i = {pk : k mod m = i}. Selection gadget. It is
a FNN with two layers that acts on the embeddings produced by the Broadcast gadget and utilizes

22


---Page Break---
ξ

ζ

1
-1
−∞

−∞

1
−1

(a) Gadget for selecting ξ (Lemma 6)

p1

p2

max{p1, p2}

1

−1

−1

1

1

−1

1

−1

1
2
−1

2
1
2
1
2

(b) Comparison gadget (following [83])

Figure 2: Orange, green, and magenta nodes represent input, hidden (with activation max{·, 0}) and
output (with activation max{·, −∞}) units respectively. (Left) ζ can be used as a control to either
let ξ pass or filter it through the network. Note that in practice a sufficiently small negative weight
can be used instead of −∞. (Right) A gadget that yields the greater of its two inputs as the output.

x1
x2
x3

1
0
1
1

p1
p2
p3
p4

0
0
0

(a) Broadcast gadget

p1
p2
p3
p4

0
1
1

p1
p4

1
−1

1
−1

1
−1

1
−1

−∞

−∞

−∞

−∞

−∞

−∞

−∞

−∞

1
−1

1
−1

1
−1

1
−1

(b) Selection gadget for selecting multiple inputs.

Figure 3: (Left) ϕ2 in Broadcast gadget (shown here for m = 3 and r = 4 for node A1) replicates all
the monomials p1, . . . , pr across the nodes of a graph G (not shown). See Algorithm 4 in Appendix
for weights and other details. (Right) Selection gadget builds on the base gadget from Fig. 2a, and
filters out specific monomials at each node of G, yielding a partition of the monomials across nodes.

the following result (Lemma 6) to filter out monomials such that each vertex Ai is left with only the
monomials p¯i = {pk : k mod m = i}. The coordinates of each ROE serve as the control variables
ζ for this purpose. Please see Algorithm 5 for details.

Lemma 6. Consider any two-dimensional input with coordinates ξ ≥0 and ζ ∈{0, 1}. There exists
a 2-layer NN with ReLU activations (see Figure 2a) that outputs ξ when ζ = 0, and 0 when ζ = 1.

Proof. If ζ = 0, then the values at the two nodes in the hidden layer are ξ and −ξ respectively before
activation, and ξ and 0 after activation. Thus, the final output in this case is ξ.

On the other hand, if ζ = 1, then the two nodes in the hidden layer take values −∞, −∞, respectively
before activation, and thus 0 and 0 after activation. Thus, the final output in this case is 0.

Proposition 15. Node Ai, after selection with Algorithm 5, only contains the monomials p¯i of f.

Proof. Note that this result is just a generalization of Lemma 6, in the sense that we now distribute r
monomials into batches of size m (with each monomial playing the role of ξ) across the m nodes
with coordinates from ROE playing the role of ζ. The two layers return 0 when ζ = 1 and ξ when
ζ = 0. Note that there is only one 0 in the ith position of ROE 1m −ei for Ai, so Ai ends up with all
the monomials p¯i after the second layer.

We are now ready to describe all the models, namely the Local, Global, Constant and Hybrid
algorithms to compute a TSF f : Rm →R.

6.4
Local MPNN and its complexity

Proposition 16. Local MPNN can learn any TSF f : Rm →R with r monomials. It requires in total
⌈log2(r/m)⌉+ 5 FNN layers across 2 layers of MPNN, and O(rm) trainable parameters.

23


---Page Break---
Algorithm 5 Selection component: FNN layer

Input: any vector with ROE. In our case, it is

p1, p2, ..., pr, (1m −ei)⊤⊤

We create a 2-layer neural network as described below.

First layer: no bias, ReLU activation max(z, 0), and a weight matrix of dimensions 2r × (r + m)
with weights

wκκ′ =










1
if κ′ ≤r and κ = 2κ′ −1
−1
if κ′ ≤r and κ = 2κ′

−∞
if κ′ ≥r and κ ≡2(κ′ −r), 2(κ′ −r) −1
mod 2m
0
otherwise.

(15)

Second layer: no bias, identity activation, and a weight matrix of dimensions r′ × 2r with weights

wκκ′ =










1
if κ =
l
κ′
2m
m
and κ′ odd,

−1
if κ =
l
κ′
2m
m
and κ′ even,

0
otherwise.

(16)

Algorithm 6 Local MPNN
Input: m-dimensional input x.

First, we perform a broadcast with Algorithm 4.

We need an additional layer of Message Passing. ϕ(2)
2
will have the form of (14). The first two layers
of ϕ(2)
2
pertain to the selection gadget from Algorithm 5.

We then resort to ⌈log2(r/m)⌉+ 1 layers of local comparison (simultaneously comparing 2 nodes
each time) according to the algorithm in [19] or [83] for ϕ(2)
2 .

□(2) is coordinate-wise max, ϕ(2)
1
= Id, and φ(2)
Update has identity activation with W (2)
self = 0 and

W (2)
neigh = Id.

Proof. As shown in Proposition 14, each node Ai has the embedding

p1, p2, ..., pr, (1m −ei)⊤⊤

after broadcasting. These embeddings are now filtered by the selection gadget, so by Proposition
15, each node Ai is left with r′ = ⌈r

m⌉monomials p¯i = {pk : k mod m = i}. Thus, using
⌈log2(r′)⌉+1 layers of comparison (see, e.g. [83]), we obtain the maximum max p¯i. Max aggregation
(coupled with the fact that we have an m-clique with self loops) helps us compute maxi=1,...,m p¯i =
maxi=1,...,r pi = f(x). Note that the second MPNN-layer φ(2) involves only fixed parameters, and
ϕ(2)
2
comprises ⌈log2(r′)⌉+ 3 FNN layers.

6.5
Global MPNN and its complexity

Proposition 17. Global MPNN can learn any TSF f : Rm →R with r monomials. It requires
in total 3 ⌈logm(r)⌉+ 2 FNN-layers across ⌈logm(r)⌉+ 1 MPNN layers, and O(rm) trainable
parameters.

Proof. As shown in Proposition 14, after the Broadcast component, each node Ai has the embedding

p1, p2, ..., pr, (1m −ei)⊤⊤. Then by Proposition 15, the message from Ai after ϕ(2)
2
(modified
Selection Component) is p¯i = {pk : k mod m = i}, appended by ROE. This vector is of dimension
r′ +m = ⌈r

m⌉+m. Then, max aggregation (coupled with the fact that we have an m-clique with self
loops) helps us compute maxi=1,...,m p1, ..., pm, maxi=1,...,m pm+1, ..., p2m, and so on, and output

24


---Page Break---
Algorithm 7 Global MPNN
First, we perform a broadcast with Algorithm 4.

The second layer of MPNN is designed as follows. ϕ(2)
2
implements the selection gadget from 5 and
uses aggregation of the form specified in (14). However, as a placeholder for the ROE, we append m
rows with all coordinates set to 0 to the weight matrix for the second layer in Algorithm 5.Thus, the
weight matrix is now of dimensions (r′ + m) × 2r.

□(2) is coordinate-wise max and ϕ(2)
1
= Id, φ(2)
Update uses identity activation, W (2)
neigh = Id, and

W (2)
self =

0
0
0
Idm×m


.

We then repeat this process (replacing r with r′ until we are left with only 1 component and the final
maximum f). This requires ⌈logm(r)⌉additional MPNN layers.

the same result for every node Ai besides the ROE which is different for the nodes. Similarly, in
the subsequent MPNN layers, invoking the modified selection component and coordinate-wise max
yields maxi=1,...,r pi = f(x). Note that, except for the first MPNN, all the subsequent MPNN layers
have only fixed, i.e., non-learnable parameters. Thus in total, we would need ⌈logm(r)⌉+ 1 MPNN
layers, including the first Broadcast layer.

6.5.1
Constant MPNN and its complexity

Algorithm 8 Constant MPNN
First, we perform a broadcast with Algorithm 4.

We only need one additional message passing layer. ϕ(2)
2
however needs to be modified since for
each node Ai, we would want to have lzj,i for each j = 1, .., q suggested by Sj instead of l¯i that
Proposition 15 guaranteed.

Specifically, in the first layer of ϕ(2)
2 , we now have 2dr hidden nodes (2d nodes for each li), ReLU
activation, no bias, and a weight matrix ∈R2dr×(r+d) with weights.

wκκ′ =










1
if κ′ ≤k, κ′ =
 κ

2m

and κ odd,
−1
if κ′ ≤k, κ′ =
 κ

2m

and κ even,
−∞
if κ′ > k and κ ≡2(κ′ −k), 2(κ′ −k) −1
mod 2m,
0
otherwise.

(17)

We now choose the linear pieces according to Sj. The second layer of ϕ(2)
2
has 2q hidden nodes,
identity activation, no bias and weights ∈R2q×2dr given by

wκκ′ =


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


1
if κ′ = 2m(zµ,i −1) + 2i −1 for i = 1, .., m and κ = 2µ + 1,
1
if 2m(zµ−1,m+1 −1) + 1 ≤κ′ ≤2mzµ−1,m+1, κ′ odd, and κ = 2µ
−1
if κ′ = 2m(zµ,i −1) + 2i for i = 1, .., m and κ = 2µ + 1,
−1
if 2m(zµ−1,m+1 −1) + 1 ≤κ′ ≤2mzµ−1,m+1, κ′ even, and κ = 2µ,
0
otherwise.

For the next layers in ϕ(2)
2 , we take the maximum of node 2κ + 1 and node 2κ using two layers of the
comparison gadget.

The aggregation function is coordinate-wise max, ϕ(2)
1
= Id, φ(2)
Update uses identity activation and

W (2)
self = 0 and [W (2)
neigh]j = sj.

Proposition 18. The Constant algorithm 8 can learn any TSF f : Rm →R with r monomials. It
requires in total 7 FNN layers across 2 layers of MPNN, and O(mrq) learnable parameters.

25


---Page Break---
Proof. z As before, after broadcasting, each node Ai comprises

p1, p2, ..., pr, (1m −ei)⊤⊤.
Then,
after processing with the first two layers of ϕ(2)
2 ,
the output for node Ai is
pz1,i, pz1,m+1, ..., pzq,i, pzq,m+1
⊤. Thus, after processed by the last two layers, the message from

node Ai is
max{pz1,i, pz1,m+1}, ..., max{pzq,i, pzk,m+1}⊤. With the max operation, the aggregated

message for all nodes is
maxi∈S1 li, ..., maxi∈Sq li
⊤. Finally, with sj filling W (2)
neigh, the resulting
embedding for every node becomes Pq
j=1 sj(maxi∈Sj li), which is by [84, Theorem 1], f(x).

6.5.2
Hybrid Architecture and its complexity

Algorithm 9 Hybrid Architecture
Input: m-dimensional input x

We start with a 1-layer FNN with m input nodes and r output nodes for the monomials. It has identity
activation and its weight matrix W ∈Rr×m where each row Wi,: = [αi ∈Rm] and its bias is the
constants in the tropical monomials, i.e. bi = ci, thus, yielding the output y = [p1
...
pr]T .

After that, we build an r-cliques A1, ..., Ar, and bidirectional edges, and introduce loops with m self-
edges from each vertex Ai to itself. We then put the initial embeddings X = yT = [p1
...
pr]. We
then have ϕ(1)
2 (h(0)
Ai , h(0)
Aj ) = h(0)
Ai and Aggregation operation to be coordinate-wise max, ϕ(2)
1
= Id

to form m(1)
Ai (which is f now), and put φ(1)
Update = m(1)
Ai = f .

The following Proposition immediately follows.
Proposition 19. The Hybrid algorithm 8 can learn any TSF f : Rm →R with r monomials. It
requires in total 1 FNN layer and 1 message passing layer, and O(rm) learnable parameters.

26


---Page Break---
NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and precede the (optional) supplemental material. The checklist does NOT
count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .

• [NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification:

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

Justification:

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.

27


---Page Break---
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

Justification:

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

Justification: Our work is purely theoretical and does not involve any experimental result.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.

28


---Page Break---
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
Answer: [NA]
Justification: Our work is purely theoretical and does not any data and code.
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

29


---Page Break---
Answer: [NA]

Justification: Our work is purely theoretical and does not involve any experiments.

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

Justification: Our work is purely theoretical and does not involve any experiments.

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

Justification: Our work is purely theoretical and does not involve any experiments.

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

30


---Page Break---
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification:

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

Justification: This is no societal impact of the work performed as the work is purely
theoretical.

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

Justification: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

31


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

Justification: The paper does not use existing assets.

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

Justification: The paper does not release new assets

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

32


---Page Break---
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
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

33


---Page Break---
