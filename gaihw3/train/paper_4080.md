Compositional PAC-Bayes:
Generalization of GNNs with persistence and beyond

Kirill Brilliantov
ETH Zürich
kbrilliantov@ethz.ch

Amauri H. Souza
Federal Institute of Ceará
amauriholanda@ifce.edu.br

Vikas Garg
YaiYai Ltd & Aalto University
vgarg@csail.mit.edu

Abstract

Heterogeneity, e.g., due to different types of layers or multiple sub-models, poses
key challenges in analyzing the generalization behavior of several modern archi-
tectures. For instance, descriptors based on Persistent Homology (PH) are being
increasingly integrated into Graph Neural Networks (GNNs) to augment them
with rich topological features; however, the generalization of such PH schemes re-
mains unexplored. We introduce a novel compositional PAC-Bayes framework that
provides a general recipe to analyze a broad spectrum of models including those
with heterogeneous layers. Specifically, we provide the first data-dependent gen-
eralization bounds for a widely adopted PH vectorization scheme (that subsumes
persistence landscapes, images, and silhouettes) as well as PH-augmented GNNs.
Using our framework, we also obtain bounds for GNNs and neural nets with ease.
Our bounds also inform the design of novel regularizers. Empirical evaluations
on several standard real-world datasets demonstrate that our theoretical bounds
highly correlate with empirical generalization performance, leading to improved
classifier design via our regularizers. Overall, this work bridges a crucial gap in
the theoretical understanding of PH methods and general heterogeneous models,
paving the way for the design of better models for (graph) representation learning.

1
Introduction

Topological data analysis (TDA) harnesses tools from algebraic topology to unveil the underlying
shape and structure of data. TDA has recently gained significant traction within machine learning
mainly due to its flagship method: persistent homology (PH) [11], which allows for capturing topolog-
ical invariants (like connected components and loops) of the input domain at multiple scales. In par-
ticular, PH has recently been leveraged as a tool to augment the representational capabilities of graph
neural networks (GNNs) [20, 53, 58], with expressivity gains formally established [22, 44]. Intuitively,
PH furnishes global structural signatures that complement the local nature of GNNs [5, 18, 20, 57].

Understanding the generalization behavior of these models is crucial as it plays a pivotal role in
ensuring their reliability and applicability [41]. In this context, there are two fundamental approaches
to achieving generalization bounds: data-independent and data-dependent [47], each offering unique
insights into the generalization problem. Both these approaches have been investigated to analyze the
generalization ability of GNNs [12, 14, 25, 33, 36, 46, 48, 52, 59]. Data-dependent generalization
bounds evoke particular interest since they are typically much tighter than the agnostic bounds
afforded by, e.g., VC dimension. However, no such bounds have been unearthed for PH methods (i.e.,
learnable vectorization schemes) and, consequently, for GNNs enhanced with PH-based descriptors.

We approach this gap with the first data-dependent generalization bound for classifiers based on a
versatile and widely used vectorization framework for persistence diagrams, namely, PersLay [5].
PersLay leverages extended persistence to effectively represent detailed topological features, and
subsumes commonly used methods such as persistence landscapes [4], images [1], and silhouettes [6].

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Central to our analysis is a novel PAC-Bayes framework (Lemma 2) that provides a general recipe
to analyze the generalization of a broad spectrum of models, including those with heterogeneous
layers and those comprising multiple sub-models. To achieve this, we introduce general conditions
(Equations 6-9) that are satisfied by commonly used learning architectures and, surprisingly, their com-
positions (Section 4). Leveraging Lemma 2, we show how to obtain bounds for heterogeneous MLPs
and GNNs in a straightforward manner (Table 2). Notably, we also establish the first generalization
bounds for GNNs augmented with persistence layers (PersLay).

Table 1: Main theoretical contributions of this work.

Section 3: Generalized PAC-Bayes
General recipe for heterogeneous models
Lm. 2
Applying the recipe to GNNs and MLPs
Tab. 2
New bound for PersLay
Cor. 1

Section 4: Compositional PAC-Bayes
Bound for the composition with MLP
Lm. 3
Bound for two models in parallel
Lm. 4
New bound for GNNs with persistence
Cor. 2

Our exposition focuses on graphs;
however, i) our Lemma 2 can be used
in any domain and ii) our bound for
PersLay considers persistence diagrams
obtained from any non-learnable fil-
tration function, and therefore extends
more generally to input domains beyond
graphs. From a technical perspective,
our approach hinges on contrasting
previous analyses within the PAC-Bayes
framework [9, 37, 38] to extract the
common
structure
encoded
in
the
general conditions of Lemma 2. This
allows us to overcome challenges arising from the heterogeneity of the models we consider.

Our experiments on several standard real-world datasets confirm strong correlation between the
empirical performance and our theoretical bounds. We reinforce the merits of our analysis via
regularized PH-based models informed by our bounds with demonstrable empirical benefits.

Our main contributions are:

(Theoretical, see Table 1) We develop a general recipe for obtaining PAC-Bayes bounds for a
broad class of (possibly heterogeneous) models and their compositions. We also provide the first
data-dependent bounds for PH-based classifiers and combinations of GNNs and PersLay;

(Empirical) We show that the dependence on parameters depicted in our bounds strongly correlates
with the observed performance. We also show that novel regularization schemes based on our
bounds can reduce the generalization gap of PH-augmented GNNs on multiple datasets.

2
Preliminaries

This section overviews GNNs, persistent homology on graphs, and their combination. We also
provide basic notions and results in PAC-Bayes learning, which serve as a background for this work.

Notation. We consider attributed graphs denoted as a tuple G = (V, E, z), where V = {1, 2, ..., n}
is the vertex set, E ⊆V × V is the edge set, and z : V →Rdz assigns to each vertex v ∈V an
attribute (or color) z(v). For convenience, hereafter, we denote the feature vector of v by zv. We
consider classification tasks with input and label spaces X and Y = {1, . . . , K} (K is the number of
classes) and the γ-margin loss lγ : Y × RK →{0, 1} where lγ(y, ˆy) = 1(ˆyy ≤γ + maxj̸=y ˆyj) and
γ ≥0 is the margin parameter. Let S = {(xi, yi)}m
i=1 ⊆X × Y denote a collection of m input/label
pairs sampled i.i.d. from some unknown distribution D. Then, the empirical error of a hypothesis
gw : X →RK with parameters w is defined as ˆLS,γ(gw) = 1/m Pm
i=1 lγ(yi, gw(xi)). Accordingly,
we can define the generalization error as LD,γ(gw) = E(x,y)∼D[lγ(y, gw(x))]. We use ∥· ∥2 to refer
the ℓ2 norm (vectors) and the spectral norm (matrices), and ∥· ∥F to refer to the Frobenius norm.
Also, we denote the set {1, ..., n} by [n]. We provide a notation table in the Appendix (Table 5).

Graph neural networks (GNNs). Message-passing GNNs [15, 55] employ a sequence of message-
passing steps, where each node v aggregates messages from its neighbors N(v) = {u : (v, u) ∈E}
and use the resulting vector to update its own embedding. Starting from z(0)
v
= zv, GNNs recursively
apply
z(ℓ+1)
v
= Updℓ

z(ℓ)
v , Aggℓ({{z(ℓ)
u
: u ∈N(v)}})

∀v ∈V,
(1)

where {{·}} denotes a multiset, Aggℓis an order-invariant function and Updℓis an arbitrary update
function — often a multilayer perceptron (MLP).

2


---Page Break---
Persistence homology (PH) on graphs aims to extract detailed (multiscale) topological features
from graphs. A filtration of a graph G is a finite nested sequence of subgraphs of G, i.e., ∅=
G0 ⊂G1 ⊂... ⊂G — alternatively, clique complexes can also be built at each step (see [2]).
While filtrations can be obtained in different ways [2, 18], a typical choice consists of leveraging a
real-valued filtering function f on the vertices of G (or their features) to compute the vertex level set
Vα = {v : f(v) ≤α} at scale α ∈R. Let Gα be the subgraph of G induced by Vα. By increasing α
from −∞to ∞, we obtain a nested sequence of subgraphs called the sub-level filtration of G induced
by f. The idea of PH is to keep track of the appearance and disappearance of topological features
(e.g., connected components, loops) in a filtration. If a topological feature first appears in Gαb and
disappears in Gαd, then we encode its persistence as a pair (αb, αd); if a feature does not disappear,
then its persistence is (αb, ∞). The collection of all pairs forms a multiset that we call persistence
diagram. We use Dgi(G) to denote the persistence diagram for i-dim topological features of graph G.
For details on PH, we refer to Edelsbrunner and Harer [10], Hensel et al. [17], and Hofer et al. [19].

Persistence layers (PersLay). Carrière et al. [5] introduced a general way to vectorize persistence
diagrams. Given a persistence diagram Dg(G) for an arbitrary dimension, PERSLAY computes

PERSLAY(Dg(G)) = Agg ({{ω(p)φ(p) | p ∈Dg(G)}})
(2)

where Agg is any permutation invariant operation (e.g., minimum, maximum, sum, or kth largest
value), ω : R2 7→R is a weight function for the elements in Dg(G), and φ : R2 7→Rh is the
so-called point transformation. More specifically, given a persistence pair p = [p1, p2]⊤∈R2,
PersLay introduces the triangle point transformation (Λ):

φΛ(p) = [Λp(t1), ..., Λp(th)]⊤
with
Λp(ti) = max{0, p2 −|ti −p1|}, ti ∈R
(3)

the Gaussian point transformation (Γ):

φΓ(p) = [Γp(t1), ..., Γp(th)]⊤
with
Γp(ti) = exp

−∥ti −p∥2
2
2τ 2


, ti ∈R2
(4)

and the line point transformation (Ψ):

φΨ(p) = [Ψp(t1), ..., Ψp(th)]⊤
with
Ψp(ti) = ti,1p1 + ti,2p2 + ti,3, ti ∈R3
(5)

In all of these transformations t1, . . . , th are learnable parameters. Notably, the architectural design
of PERSLAY is quite versatile and accommodates a wide range of traditional persistence diagram vec-
torizations, extending DeepSets [56] and including persistence landscapes [4], persistence silhouette
[6], persistence images [1], and other Gaussian-based kernel approaches [23, 29, 31].

To obtain class predictions, the output of PersLay is typically fed to an MLP with Lipschitz activations.
We refer to this joint model (PersLay followed by MLP) as PersLay Classifier (PC).

Figure 1: GNNs with persistence (parallel mode).

GNNs with persistence. Recently, PH has been
used to boost the expressive power of GNNs. For
instance, Horn et al. [20] and Immonen et al. [22]
leverage node embeddings at each layer of a GNN
to obtain persistence descriptors. This topological
information can be added to GNN’s node embeddings
(as in [20]) or concatenated with GNN’s graph-level
representation (as in [22]) — which we refer to the
parallel mode of integrating PH into GNNs. Figure 1 illustrates the latter (parallel mode) with
persistence diagrams vectorized using PersLay.

PAC-Bayesian analysis adopts a Bayesian approach to the PAC learning framework [30, 37, 38, 50].
The idea consists of placing a prior distribution P over our hypothesis class and then use the training
data to obtain a posterior Q, i.e., the learning process induces a posterior distribution over the
hypothesis class. Importantly, we can leverage the Kullback-Leibler (KL) divergence between Q and
P to bound the difference between the generalization and empirical errors [37]. To compute PAC-
Bayes bounds for models like neural networks, we can i) choose a prior, ii) apply a learning algorithm;
and iii) add random perturbations (from some known distribution) to the learned parameters such that
we ensure tractability of the KL divergence. Following this recipe, Neyshabur et al. [40] introduced
the important result in Lemma 1. Notably, Lemma 1 tells us that if we have prior and posterior
distributions and guarantees that the change of the model’s output due to perturbations over the
learned parameters is small with high probability, we can obtain a generalization bound.

3


---Page Break---
Lemma 1 (Neyshabur et al. [40]). Let gw(x) : X →RK be any model with parameters w,
and let P be any distribution on the parameters that is independent of the training data. For
any w, we construct a posterior Q(w + u) by adding any random perturbation u to w, s.t.,
Pu(maxx∈X |gw+u(x) −gw(x)|∞<
γ
4 ) >
1
2 . Then, for any γ, δ > 0, with probability at
least 1 −δ over an i.i.d. size-m training set S drawn according to D, for any w, we have:

LD,0(gw) ≤ˆLS,γ(gw) + 4

s

DKL(Q(w + u)||P) + log 6m

δ
m −1
.

3
Generalized PAC-Bayes

This section first presents a general procedure for obtaining generalization bounds for heterogeneous
models, i.e., going beyond spectrally-normalized layers and architecture-specific models (as in
[33, 40]). Then, we show how to leverage such a procedure to extend existing bounds in the literature
and to obtain the first generalization bound for PersLay.

Our next result (Lemma 2) applied perturbation-based PAC-Bayes bounds to arbitrary models with
(possibly) non-homogeneous layers. To achieve this generality, we carefully contrasted results in
[33, 40, 48] to identify the conditions (Equations 6, 7, 8, and 9) that are sufficient to subsume the
considered models as well as to extend to a broader class of models. In Section 4, we will also exploit
Lemma 2 in the analysis of different combinations of neural models (e.g., GNNs and PersLay).

General recipe for PAC-Bayesian bounds for heterogeneous models

Lemma 2. Let fw : X 7→RK be a model with parameters w = vec{W1, ..., Wn}. If there
exists T ∈R+ and a sequence (Si)i∈[n] with Si ∈R+ both of which may depend on w, and
parameter-independent C1, C2 ∈R+ and sequence (ηi)i∈[n] with ηi ∈(0, 1] such that:

• the output is bounded by C1T:

sup
x∈X
∥fw(x)∥2 ≤C1T,
(6)

• the output change can be bounded under a small perturbation of the parameters, i.e.,
for all u = vec{U1, ..., Un} with ∥Ui∥2 ≤ηiSi:

sup
x∈X
∥fw+u(x) −fw(x)∥2 ≤C2T

n
X

i=1

∥Ui∥2

Si
(7)

• the following auxiliary conditions hold:

1
n

 n
X

i=1

1
Si

!

≥
1
T 1/n ,
(8)

¯η := min
1≤i≤n ηi ≤C1

2C2
.
(9)

Then, for any γ, δ > 0 with probability at least 1 −δ over the choice of training sets S with
m i.i.d. samples drawn according to some distribution D, we have:

LD,0(fw) ≤ˆLS,γ(fw)+

+ O









v
u
u
u
tmax{1, ∥w∥2
2}T 2
 nP

i=1

1
Si

2
h ln (nh) C2
1 ¯η−2 + log max
n
m

δ ,
m
δC1

o

γ2m








,

(10)

where h is the maximum dimension across the matrices (Wi)i∈[n].

4


---Page Break---
Proof sketch. We build on the main result in [40] by extending it to a broader context. The general
idea involves employing Lemma 1. Following [40], we define the prior distribution P as an isotropic
Gaussian with variance σ2 and the posterior distribution Q as a shifted isotropic Gaussian with the
same variance. To achieve a tighter bound, it is essential to maximize σ (since the KL-divergence
scales as O(1/σ2)); consequently, σ should be determined based on the parameter β. However, since
P must remain independent of the learned weights, we set σ according to an approximation of the
learned weights. Specifically, we define β = T P

i 1/Si and at first consider only w such that β fall
within the range |β −˜β| ≤1/2β for some arbitrary ˜β, an approximation. We then select σ based on
this approximation, ˜β. At this point we can apply Lemma 1 for all w such that β falls into the defined
earlier interval. To account for other values of β, we establish a finite grid across the relevant β values
and choose an appropriate ˜β for each interval on the grid. Finally, a union-bound argument across all
˜β values provides the final result. Although Equation 7 and Lemma 1 have their own constraints on
the random perturbation, the above steps outline a method to set the variance σ that satisfies these
constraints and maintains independence from the learned weights.

Table 2: Application of Lemma 2 to MLPs and GNNs. The detailed proof of the lemma applicability
can be found in the Appendix E and the detailed description of the models in Appendix B. Here we
provide brief description. We consider n-layer multilayer perceptron (MLP) with weights W1, ..., Wn.
After layer i we apply Lipi-Lipschitz activation function for i ∈[n −1]. Every input is contained
in ℓ2-ball of radius B. We consider n-layer GCN with weights W1, ..., Wn. After layer i we
apply Lipi-Lipschitz activation function. Every node feature of the graph is contained in ℓ2-ball of
radius B and the maximum degree of the node is d −1. We denote Lip = Lip1 · ... · Lipn−1. We
consider n-layer (n > 2) MPGNN with weights W1, W2, W3 with activation functions g, ϕ, ρ with
corresponding Lipschitz constants. We denote C = LipϕLipgLipρ∥W2∥, λ = ∥W1∥2∥W3∥2 and
ξ = ((dC)n−1−1)/(dC−1). Comparing to [33] we do not add Lipϕ to ξ and instead of Wl we have W3.

Model (reference)
T
Si
ηi
C1
C2

MLP (Neyshabur et al. [40])
nQ

i=1
∥Wi∥2
∥Wi∥2
1
6n
B Lip
eB Lip

GCN (Liao et al. [33])
nQ

i=1
∥Wi∥2
∥Wi∥2
1
6n
d
n−1

2 B Lip
ed
n−1

2 B Lip

GCN (Sun and Lin [48])
nQ

i=1
∥Wi∥2
∥Wi∥2
1
6n
B Lip
eB Lip

MPGNN, dC ̸= 1 (Liao et al. [33])
λξ
∥Wi∥2*
1
6n
B Lipϕ
eBn Lipϕ

MPGNN, dC = 1 (Liao et al. [33])
λ
∥Wi∥2*
1
6n
Bn Lipϕ
eBn2 Lipϕ

*S2 = min{dC, ∥W2∥2}

Discussion.
We note that Lemma 2 requires choosing values for the variables (Si)i∈[n] and T. In
this regard, one might set Si as the spectral norm of the weight matrix Wi, i.e., Si = ∥Wi∥2, and
make T equal to Q
i Si. In this case, the condition in Equation 8 is satisfied — the geometric mean is
always smaller than or equal to the arithmetic mean. Regarding the variables (ηi)i, a typical choice is
to set ηi = O(1/n). By doing so, our bound implicitly depends on n also through ηi.

The role of Equation 6 is to constraint w to non-trivial parameter spaces. In particular, if T is too
small, the magnitude of the model output might not be sufficient to distinguish different inputs up to
a margin γ. In this case, the model would have large empirical loss. In turn, Equation 7 is directly
associated with the condition in Lemma 1, enabling us to use it.

As discussed, Neyshabur et al. [40] assume spectrally-normalized weight matrices. To avoid this
assumption, we introduce the conditions in Equation 8 and Equation 9, which allow us to pick
perturbations that meet the condition in Equation 7, again justifying the application of Lemma 1.

The bound also includes a somewhat unconventional term, max{1, ∥w∥2
2}. While this technical term
allows for a more concise proof, we note that it does not impose suboptimality. More specifically, in
most real-world cases, the squared norm of w is greater than 1. See Appendix I for a discussion.

5


---Page Break---
Applying Lemma 2 to MLPs and GNNs.
As previously mentioned, using Lemma 2 involves
defining the variables T, (Si)i, (ηi)i, C1, C2 to meet all conditions in Lemma 2’s statement. Typically,
this definition comes naturally from the perturbation analysis of the model. To illustrate the power of
Lemma 2, Table 2 shows how we can apply it using the perturbation analysis for MLPs and GNNs
provided in [33, 40, 48]. Detailed proofs are given in Appendix E. Importantly, we note that we do
not make any additional assumption beyond those in the original papers — we state all assumptions
before each proof in the Appendix for clarity.

Notably, our approach generalizes results by Neyshabur et al. [40] and Liao et al. [33] to MLPs/GCNs
with different activation functions — note that the original works only consider ReLU activations. For
inherently non-homogeneous models like MPGNNs, our method leads to tighter bounds in several
settings. We provide a comparison between our bounds and previous ones in Appendices F and G.

Applying Lemma 2 to PersLay.
Next, Corollary 1 provides a PAC-Bayes bound for PersLay under
fixed filtration functions — see Appendix H for a discussion about filtration functions. To the best of
our knowledge, this is the first generalization result for vectorization schemes of persistence diagrams.
Again, the proof consists of verifying if the requirements of Lemma 2 are met. For readability, we
omit the proof and provide only the constant weighting function case in the main text, proof as well
as the arbitrary weighting function case can be found in Appendix E. The perturbation analysis of
PersLay is also given in the supplementary material (Lemma 10).
Corollary 1 (PersLay with constant weighting function). Let fw : G 7→Rk with w = vec{W (φ)}
be a PersLay where W (φ) denotes the parameters of the point-transformation function, φ. Let Dg be
a mapping from graphs to (extended) persistence diagrams with a fixed filtration function and B such
that max
G∈G
max
p∈Dg(G) ∥p∥2 ≤B, then fw satisfies the requirements of Lemma 2 with:

T = T (φ)
and
S = ∥W (φ)∥2
and
η = 1,

C1 = 2C2
and
C2 = 2A2 max{Lip(φ), C(φ)},
where

A1 = A2 = max
G∈G card(Dg(G))
if Agg = "sum"

A1 = 1, A2 = 3
if Agg = "k-max" or "mean"

and

T φ
Cφ
Lipφ

if φ = Λ
max{1, ∥W (φ)∥2}
B
√

h
1
if φ = Γ
max{1, ∥W (φ)∥2}
√

h
1
τ√e
if φ = Ψ
∥W (φ)∥2
√

3 max{1, B}
max{1, B}

PersLay’s special cases.
Carrière et al. [5] designed PersLay with flexibility in mind to subsume
commonly used persistence vectorization schemes in the literature. Consequently, we can obtain
bounds for these schemes — we provide the values of C2 (divided by 2) which is enough to compare
the schemes since everything else is the same in the constant weighting function case:

Persistence k-landscapes [4]
Persistence Images [1]
Persistence Silhouettes [6]

3 max{B
√

h, 1}
card · max{
√

h,
1
τ√e}
card · max{B
√

h, 1}

where card is the maxG∈G Dg(G). If "images" and "silhouettes" use "sum" as an aggregating func-
tion, then our generalization analysis suggests that "k-landscapes" would have stronger guarantees. If
these schemes use "mean" as an aggregating function, then C2 for "k-landscapes" would be at most
C2 for "silhouettes", and the result of comparison of "k-landscapes" and "images" could be in favor
of both "landscapes" and "images" depending on chosen parameters τ and B.

4
Compositional PAC-Bayes

In this section, we present two lemmas (Lemmas 3 & 4) that allow us to compose models satisfying
Lemma 2 requirements. At the end of the section (Corollary 2), we showcase our framework by

6


---Page Break---
getting generalization bounds for combinations of GNNs and PersLay. For readability, here we
provide informal statements and defer the formal ones to the Appendix (Lemma 5, Lemma 6).

In particular, Lemma 3 establishes that the composition of MLPs with models that satisfy Lemma 2
also satisfy it. As a result, we can derive PAC-Bayes bounds for heterogeneous models that leverage
MLPs using our framework in a straightforward way. This is particularly relevant since deep learning
models often employ learnable feature extractors followed by MLPs as classification heads.
Lemma 3 (Informal; Composition with MLP). Let f be an MLP and g be a model satisfying Lemma 2
requirements, then f ◦g also satisfies Lemma 2 requirements.

In addition, we show in Lemma 5 (Appendix) that this result also extends to an arbitrary number of
models beyond MLPs. In particular, the result holds whenever we can upper bound output deviations
due to perturbations on parameters and inputs, i.e., supx∈X ∥fw+u(x + ∆x) −fw(x)∥2 is bounded.

Our next lemma suggests that models satisfying Lemma 2 requirements are closed under parallel
concatenation. We note that combining two (or more) models in parallel is also a common design
choice in deep learning. For instance, this encompasses persistence-augmented GNNs [22] and
ensemble methods [13].
Lemma 4 (Informal; Models in parallel). Let f1, f2 be two models satisfying Lemma 2 requirements
and g be an aggregating Lipschtiz function. Then, g(f1(·), f2(·)) also satisfies Lemma 2 requirements.

We also provide a generalization of this lemma in the Appendix (Lemma 6) for n > 2 models in
parallel. It leads to tighter bounds than a naive 2-by-2 sequential application of Lemma 4.
Corollary 2 (Informal). By combining the results for MLPs, GCNs, MPGNNs (Table 2) and that for
PersLay (Corollary 1) with Lemma 3 and Lemma 4, we can get generalization bounds on various
compositions of these models. In particular, for GNNs with persistence (see Figure 1), we have

• T for the overall model scales with the product of PersLay’s and GNN’s T variables;

• C1 and C2 of the overall model scale linearly with C1, C2 of PersLay and GNN.

Despite the generality of our results, Corollary 2 demonstrates the benefits of our framework in the
domain of graph representational learning. To the best of our knowledge, this the first result providing
generalization guarantees for graph neural networks combined with persistence vectorization
schemes. Furthermore, our findings can aid practitioners in making informed architectural decisions
to enhance the generalizability of their models. Specifically, in the case of combining PersLay with
GNNs, a tighter bound can be achieved by selecting a PersLay dimension that is considerably smaller
than the GNN dimension. Failing to do so may result in a bound dependency on the width of the
form O(h
√

ln h) rather than O(
√

h ln h). Additionally, we recommend using aggregation functions
such as "mean" or "k-max" instead of "sum" as the latter introduces a term maxG∈G card(G) to
the bound, which may be large in practical scenarios.

5
Related Works

Expressivity and generalization of GNNs. GNNs have achieved state-of-the-art performance
across various applications [16, 28, 45, 51, 54], and have garnered significant attention. Maron et al.
[35], Xu et al. [55] analyzed the representational power of GNNs in terms of the 1-WL test, revealing
theoretical limits on their expressivity. This has motivated a surge of works aiming to go beyond
1-WL with GNNs [e.g., 32]. Regarding generalization, Scarselli et al. [46] provided upper bounds on
the order of growth of VC-dimension for GNNs. Garg et al. [14] presented the first data-dependent
generalization bounds for GNNs via Rademacher complexity. Recently, Morris et al. [39] employed
the WL test alongside VC-dimension to gain insights about the generalization performance of GNNs.
For details about the expressivity and learning of GNNs, we refer to Jegelka [24].

Learning theory and PH. Birdal et al. [3], Dupuis et al. [8] and Chen et al. [7] investigate connections
between learning theory and topological data analysis. In particular, Birdal et al. [3], Dupuis et al. [8]
explored the concept of PH dimension as a complexity measure to analyze generalization. Chen et al.
[7] proposed a topological regularizer to simplify decision boundaries by penalizing non-essential
topological features. In contrast, we apply learning theory to derive data-dependent generalization
bounds for arbitrary heterogeneous layers, specifically targetting persistence-aware GNNs, and intro-
duce a regularizer informed by these bounds to guide the design of robust and generalizable models.

7


---Page Break---
1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

Generalization

ρ: 0.78±0.08

DHFR

Generalization gap
Spectral norm

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

ρ: 0.95±0.02

MUTAG

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

1.5

ρ: 0.89±0.06

PROTEINS

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

ρ: 0.94±0.07

NCI1

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

1.5

ρ: 0.79±0.04

IMDB-BINARY

Figure 2: PersLay classifier: spectral norm vs. generalization gap. Overall, our bound on the
spectral norm of the weights is highly correlated with the generalization gap.

26
28

Width, h

0.0

0.5

1.0

Generalization gap

ρ: 0.91±0.07

DHFR

Empirical gap
Our bound

26
28

Width, h

0.00

0.25

0.50

0.75

1.00
ρ: 0.77±0.12

MUTAG

26
28

Width, h

0.0

0.5

1.0
ρ: 0.95±0.03

PROTEINS

26
28

Width, h

0.0

0.5

1.0
ρ: 0.98±0.02

NCI1

26
28

Width, h

0.0

0.5

1.0
ρ: 0.95±0.04

IMDB-BINARY

Figure 3: PersLay classifier: width vs. generalization gap. The dependence of the empirical gap
on the model width is captured by our bound. We obtain high average correlation for all datasets.

PAC-Bayes. The PAC-Bayes framework [37, 38] allows us to leverage knowledge about learn-
ing algorithms and distributions over the hypothesis set to achieve tighter generalization bounds.
Remarkably, Neyshabur et al. [40] presented a generalization bound for feedforward networks in
terms of the product of the spectral norm of weights using a PAC-Bayes analysis. Liao et al. [33]
provided PAC-Bayes bounds for GNNs, and Sun and Lin [48] enhanced their analysis considering
the adversarial case as well. In a seminal work, Dziugaite and Roy [9] optimized PAC-Bayes bounds
directly to obtain non-vacuous generalization bounds for deep stochastic neural networks.

6
Experiments

To illustrate the practical relevance of our analysis, we now consider the generalization of persistence-
aware models on real-world datasets, and report results for regularized models based on our bounds.
In particular, we conduct two main experiments. The first one aims to analyze how well our bounds
capture generalization gaps as a function of model variables. The second assesses to which extent
a structural risk minimization algorithm that uses our bound on the weights spectral norm improve
generalization compared to empirical risk minimizers. We implemented all experiments using
PyTorch [42], and implementation details are given in the Appendix J. Our code is available at
https://github.com/Aalto-QuML/Compositional-PAC-Bayes.

Datasets and evaluation setup. We use six popular benchmarks for graph classification: DHFR,
MUTAG, PROTEINS, NCI1, IMDB-BINARY, MOLHIV, which are available as part of TUDatasets
[26] and OGB [21]. We use a 80/10/10% (train/val/test) split for all datasets when we perform
model selection. Here, we consider both PersLay Classifiers and GNNs with persistence models
with constant weight functions and Gaussian point transformations. For the experiments with GNNs,
we kept only the larger datasets (and added results for the NCI109 dataset). Regarding filtration
functions, we closely follow [5] and use Heat kernels with parameter values equal to 0.1 and 10.

Dependence on model components. Figure 2 and Figure 4 show the generalization gap (measured
as LD,0 −ˆLS,γ=1) and the bound on the weights spectral norm (T from Lemma 2) over the training
epochs for PersLay Classifier and GNNs with persistence, respectively. To evaluate how well our
bound captures the trend observed in the empirical gap, we compute correlation coefficients between
the two sequences across different seeds and report their mean and standard deviation for each dataset.
Overall, the coefficients are greater than 0.7 in 7 out of 9 experiments, indicating a good correlation.

Figure 3 shows the empirical gap and our estimated bound as a function of the model’s width for the
PersLay classifier. Again, we compute correlation coefficients between the two curves and find they
are highly correlated (with an average correlation above 0.91 on 4 out of 5 datasets). Also, we note
that these curves are obtained at the final training epoch. We report additional results across different
epochs and hyper-parameters in the supplementary material. Again, these results validate that our
theoretical bounds can capture the trend observed in the empirical generalization gap.

8


---Page Break---
1
1000
2000
Epoch

0.0

0.5

1.0

Generalization

ρ: 0.77±0.10

NCI1

Generalization gap
Our bound

1
1000
2000
Epoch

−0.5

0.0

0.5

1.0

ρ: 0.68±0.15

NCI109

1
1000
2000
Epoch

−0.5

0.0

0.5

1.0

ρ: 0.80±0.13

PROTEINS

1
1000
2000
Epoch

−0.5

0.0

0.5

1.0

1.5

ρ: 0.73±0.16

IMDB-BINARY

Figure 4: GNNs with persistence: empirical gap vs. PAC-Bayes bound. Again, there is positive
and high correlation between our bound and the observed generalization gap.

Regularizing PersLay. We compare variants of PersLay trained via ERM (empirical risk minimiza-
tion) and a regularized version with loss given by ˆLS,1+αr
p

n2 ln n∥w∥2
2β2; αr is a hyper-parameter
that balances the influence of the two terms and β = T P

i 1/Si — see proof sketch of Lemma 2
in Section 3. This is similar to a weight-decay regularization approach, with the spectral norm of
weights appearing in β. Here, we consider models with n = 1 or 2, selected via hold-out validation.
We note that PersLay classifier does not use node features, it only exploits graph structures.

Table 3: Comparison of PersLay with and without spectral norm regularization. We report accuracy
statistics (except for MOLHIV, which uses AUROC) computed over five independent runs. In 5 out
of 6 cases, the models using SpecNorm achieve better test results.

Method
DRFH
MUTAG
PROTEINS
IMDB-B
NCI1
MOLHIV
PersLay (ERM)
0.71 ± 0.04
0.88 ± 0.02
0.65 ± 0.03
0.65 ± 0.01
0.68 ± 0.01
0.7005 ± 0.0177
PersLay (w/ SpecNorm)
0.72 ± 0.02
0.94 ± 0.01
0.72 ± 0.01
0.70 ± 0.01
0.68 ± 0.01
0.7185 ± 0.0106

Table 3 reports accuracy results (mean and standard deviations) computed over five runs. Overall,
the regularized approach significantly outperforms the ERM variant despite the use of small-sized
networks. On 5/6 datasets, PersLay with spectral norm regularization is the best-performing model.

Regularizing GNNs with persistence. We now evaluate the impact of using our bound to regularize
different GNNs combined with persistent homology (PersLay) in parallel mode. We consider GCN
[28], GraphSage [16], and GIN [55] architectures. Table 4 reports the test classification error and the
generalization gap on the NCI, NCI109, and PROTEINS datasets — mean and standard deviation
obtained over five independent runs. For our regularizer, we select the optimal penalization factor
αr ∈{1e-5, 1e-6, 1e-7, 1e-8} using the validation set. Overall, the results show that the regularized
methods achieve smaller generalization gaps and slightly lower classification errors. In particular, our
spectral regularizer leads to a significant drop in generalization gap in all experiments.

Table 4: Test classification error (0-1 loss) and generalization gap (LD,0 −ˆLS,γ) for PH-augmented
GNNs. ERM means empirical risk minimizer (no regularization). We denote the best-performing
methods in bold. In almost all cases, employing the method derived from our theoretical analysis
leads to the smallest test errors and generalization gaps.

Test error
Generalization gap

GNN
Dataset
ERM
SPECNORM
ERM
SPECNORM

GCN
NCI
0.22 ± 0.01
0.21 ± 0.02
0.19 ± 0.01
0.01 ± 0.06
NCI109
0.28 ± 0.00
0.28 ± 0.02
0.25 ± 0.00
0.12 ± 0.03
PROTEINS
0.31 ± 0.01
0.27 ± 0.02
0.25 ± 0.02
-0.02 ± 0.11

SAGE
NCI
0.24 ± 0.01
0.21 ± 0.02
0.18 ± 0.04
-0.08 ± 0.05
NCI109
0.26 ± 0.01
0.24 ± 0.01
0.23 ± 0.01
0.05 ± 0.06
PROTEINS
0.27 ± 0.02
0.26 ± 0.02
0.25 ± 0.01
-0.15 ± 0.30

GIN
NCI
0.25 ± 0.01
0.22 ± 0.00
0.23 ± 0.01
0.01 ± 0.06
NCI109
0.24 ± 0.01
0.24 ± 0.03
0.22 ± 0.01
0.00 ± 0.08
PROTEINS
0.29 ± 0.02
0.30 ± 0.03
0.26 ± 0.03
0.07 ± 0.18

9


---Page Break---
7
Conclusion, Broader Implications, and Limitations

We derive the first generalization bounds for neural networks that appeal to persistent homology
for graph learning. Notably, the analyzed framework (PersLay) offers a flexible and general way
to extract vector representations from persistence diagrams. Due to this generality, our analysis
covers several methods available in the literature. The developed framework also allows to analyze
composite models like, GNNs combined with PersLay. Our constructions involve a perturbation
and generalization behavior analysis of non-homogeneous networks in rather general setting, which
poses specific technical challenges.

While we provide valuable insights and methodologies, we would like to underscore the need for
future investigations to delve into PH schemes that encompass parametrized filtration functions.
Nonetheless, while some works showed gains using learnable filtrations [20], others have reported no
benefits and advocated fixed functions instead [5, 34]. Moreover, the tightness of our bounds can
further be improved since there is still considerable gap between empirical results and the theoretical
one. By shedding new light on the generalization of machine learning models based on persistent
homology, we hope to contribute to the community by providing key insights about the limits and
power of these methods, paving the path to further theoretical developments on PH-based neural
networks for graph representation learning.

Acknowledgments

VG acknowledges support from the Research Council of Finland (grant decision 342077) for the
project “Human-steered next-generation machine learning for reviving drug design”, and the Jane
and Aatos Erkko Foundation (grant 7001703) for “Biodesign: Use of artificial intelligence in enzyme
design for synthetic biology”. VG also thanks the Finnish Ministry for Education and Culture for
their support via the “MEC Global Programme pilot USA” initiative. AS acknowledges the support
from the Conselho Nacional de Desenvolvimento Científico e Tecnológico CNPq (404336/2023-0)
and the Silicon Valley Community Foundation through the University Blockchain Research Initiative
(Grant #2022-199610). KB thanks Aalto University for the support during the internship.

References

[1] Henry Adams, Tegan Emerson, Michael Kirby, Rachel Neville, Chris Peterson, Patrick Shipman,
Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori Ziegelmeier. Persistence Images:
A Stable Vector Representation of Persistent Homology. Journal of Machine Learning Research,
18(8):1–35, 2016.

[2] M.E. Aktas, E. Akbas, and A.E Fatmaoui. Persistence homology of networks: methods and
applications. Applied Network Science, 2019.

[3] Tolga Birdal, Aaron Lou, Leonidas Guibas, and Umut ¸Sim¸sekli. Intrinsic Dimension, Persis-
tent Homology and Generalization in Neural Networks. In Advances in Neural Information
Processing Systems (NeurIPS), 2021.

[4] P. Bubenik. Statistical topological data analysis using persistence landscapes. Journal of
Machine Learning Research, 16:77–102, 2015.

[5] Mathieu Carrière, Frédéric Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, and Yuhei Umeda.
PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological
Signatures. In Artificial Intelligence and Statistics (AISTATS), 2020.

[6] Frédéric Chazal, Brittany Terese Fasy, Fabrizio Lecci, Alessandro Rinaldo, and Larry Wasser-
man. Stochastic Convergence of Persistence Landscapes and Silhouettes. In Proceedings of the
Thirtieth Annual Symposium on Computational Geometry, 2014.

[7] Chao Chen, Xiuyan Ni, Qinxun Bai, and Yusu Wang. A Topological Regularizer for Classifiers
via Persistent Homology, October 2018.

[8] Benjamin Dupuis, George Deligiannidis, and Umut ¸Sim¸sekli. Generalization Bounds with
Data-dependent Fractal Dimensions. arXiv e-prints, 2023.

10


---Page Break---
[9] Gintare Karolina Dziugaite and Daniel M. Roy. Computing nonvacuous generalization bounds
for deep (stochastic) neural networks with many more parameters than training data.
In
Conference on Uncertainty in Artificial Intelligence (UAI), 2017.

[10] H. Edelsbrunner and J. Harer. Computational Topology - an Introduction. American Mathemat-
ical Society, 2010.

[11] Herbert Edelsbrunner and John Harer. Persistent homology—a survey. In Jacob E. Goodman,
János Pach, and Richard Pollack, editors, Contemporary Mathematics, volume 453, pages
257–282. American Mathematical Society, 2008.

[12] Pascal Esser, Leena Chennuru Vankadara, and Debarghya Ghoshdastidar. Learning theory can
(sometimes) explain generalisation in graph neural networks. In Advances in Neural Information
Processing Systems (NeurIPS), 2021.

[13] M.A. Ganaie, Minghui Hu, A.K. Malik, M. Tanveer, and P.N. Suganthan. Ensemble deep
learning: A review. Engineering Applications of Artificial Intelligence, 115, 2022.

[14] Vikas Garg, Stefanie Jegelka, and Tommi Jaakkola. Generalization and Representational Limits
of Graph Neural Networks. In International Conference on Machine Learning (ICML), 2020.

[15] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for
quantum chemistry. In International Conference on Machine Learning (ICML), 2017.

[16] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large
graphs. In Conference on Neural Information Processing Systems (NeurIPS), 2017.

[17] Felix Hensel, Michael Moor, and Bastian Rieck. A Survey of Topological Machine Learning
Methods. Frontiers in Artificial Intelligence, 4, 2021. ISSN 2624-8212.

[18] Christoph Hofer, Roland Kwitt, Marc Niethammer, and Andreas Uhl. Deep Learning with
Topological Signatures. In Advances in Neural Information Processing Systems (NeurIPS),
2017.

[19] Christoph Hofer, Florian Graf, Bastian Rieck, Marc Niethammer, and Roland Kwitt. Graph
filtration learning. In International Conference on Machine Learning (ICML), 2020.

[20] M. Horn, E. De Brouwer, M. Moor, Y. Moreau, B. Rieck, and K. Borgwardt. Topological graph
neural networks. In International Conference on Learning Representations (ICLR), 2022.

[21] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele
Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs.
arXiv preprint arXiv:2005.00687, 2020.

[22] Johanna Emilia Immonen, Amauri H. Souza, and Vikas Garg. Going beyond persistent ho-
mology using persistent homology. In Advances in Neural Information Processing Systems
(NeurIPS), 2023.

[23] Jan Reininghaus, Stefan Huber, Ulrich Bauer, and Roland Kwitt. A stable multi-scale kernel
for topological machine learning. 2015 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2015.

[24] Stefanie Jegelka. Theory of graph neural networks: Representation and learning. ArXiv,
abs/2204.07697, 2022.

[25] Haotian Ju, Dongyue Li, Aneesh Sharma, and Hongyang R. Zhang. Generalization in graph
neural networks: Improved pac-bayesian bounds on graph diffusion. In International Conference
on Artificial Intelligence and Statistics, 2023.

[26] Kristian Kersting, Nils M. Kriege, Christopher Morris, Petra Mutzel, and Marion Neu-
mann.
Benchmark data sets for graph kernels, 2016.
URL http://graphkernels.cs.
tu-dortmund.de.

[27] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference
on Learning Representations (ICLR), 2015.

11


---Page Break---
[28] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional
networks. In International Conference on Learning Representations (ICLR), 2017.

[29] Genki Kusano, Yasuaki Hiraoka, and Kenji Fukumizu. Persistence weighted Gaussian kernel
for topological data analysis. In International Conference on Machine Learning (ICML), pages
2004–2013, 2016.

[30] John Langford and John Shawe-Taylor. PAC-Bayes & Margins. In Advances in Neural
Information Processing Systems (NeurIPS), volume 15. MIT Press, 2002.

[31] Tam Le and Makoto Yamada. Persistence Fisher Kernel: A Riemannian Manifold Kernel for
Persistence Diagrams. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[32] P. Li, Y. Wang, H. Wang, and J. Leskovec. Distance encoding: Design provably more powerful
neural networks for graph representation learning. In Advances in Neural Information Processing
Systems (NeurIPS), 2020.

[33] Renjie Liao, Raquel Urtasun, and Richard Zemel. A PAC-Bayesian Approach to Generalization
Bounds for Graph Neural Networks. In International Conference on Learning Representations
(ICLR), 2020.

[34] Yuankai Luo, Lei Shi, and Veronika Thost. Improving self-supervised molecular representation
learning using persistent homology. In Thirty-seventh Conference on Neural Information
Processing Systems, 2023.

[35] H. Maron, H. Ben-Hamu, H. Serviansky, and Y. Lipman. Provably powerful graph networks. In
Advances in Neural Information Processing Systems (NeurIPS), 2019.

[36] Sohir Maskey, Ron Levie, Yunseok Lee, and Gitta Kutyniok. Generalization analysis of message
passing neural networks on large random graphs. In Advances in Neural Information Processing
Systems (NeurIPS), 2022.

[37] David McAllester. Simplified PAC-Bayesian Margin Bounds. In Bernhard Schölkopf and Man-
fred K. Warmuth, editors, Learning Theory and Kernel Machines, Lecture Notes in Computer
Science, pages 203–215, Berlin, Heidelberg, 2003. Springer.

[38] David A. McAllester. PAC-Bayesian model averaging. In Proceedings of the Twelfth Annual
Conference on Computational Learning Theory, COLT ’99, pages 164–170. Association for
Computing Machinery, July 1999.

[39] Christopher Morris, Floris Geerts, Jan Tönshoff, and Martin Grohe. WL meet VC. In Interna-
tional Conference on Machine Learning (ICML), 2023.

[40] Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-Bayesian Approach to
Spectrally-Normalized Margin Bounds for Neural Networks. In International Conference on
Representation Learning (ICLR), 2018.

[41] Vahid Partovi Nia, Guojun Zhang, Ivan Kobyzev, Michael R. Metel, Xinlin Li, Ke Sun, Sobhan
Hemati, Masoud Asgharian, Linglong Kong, Wulong Liu, and Boxing Chen. Mathematical
Challenges in Deep Learning. ArXiv e-prints: 2303.15464, 2023.

[42] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison,
L. Antiga, and A. Lerer. Automatic differentiation in pytorch. In Advances in Neural Information
Processing Systems (NeurIPS - Workshop), 2017.

[43] Giovanni Petri, Martina Scolamiero, Irene Donato, Francesco Vaccarino, Thomas Gilbert,
Markus Kirkilionis, and Gregoire Nicolis. Networks and Cycles: A Persistent Homology
Approach to Complex Networks. Proceedings of the European Conference on Complex Systems
2012. Springer International Publishing, 2013. ISBN 978-3-319-00394-8.

[44] Bastian Rieck. On the Expressivity of Persistent Homology in Graph Learning. arXiv e-prints,
(arXiv:2302.09826), 2023.

[45] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini.
The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.

12


---Page Break---
[46] Franco Scarselli, Ah Chung Tsoi, and Markus Hagenbuchner. The vapnik–chervonenkis
dimension of graph and recursive neural networks. Neural Networks, 108:248–259, 2018.

[47] Milad Sefidgaran and Abdellatif Zaidi. Data-dependent Generalization Bounds via Variable-Size
Compressibility. arXiv e-prints, 2023.

[48] Tan Sun and Junhong Lin. Pac-bayesian adversarially robust generalization bounds for graph
neural network. arXiv e-prints, 2024.

[49] Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations of Compu-
tational Mathematics, 12(4):389–434, August 2012. ISSN 1615-3375, 1615-3383.

[50] L. G. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142,
November 1984. ISSN 0001-0782. doi: 10.1145/1968.1972.

[51] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations
(ICLR), 2018.

[52] S. Verma and Z.-L. Zhang. Stability and generalization of graph convolutional neural networks.
In International Conference on Knowledge Discovery & Data Mining (KDD), 2019.

[53] Yogesh Verma, Amauri H Souza, and Vikas Garg. Topological neural networks go persistent,
equivariant, and continuous. In International Conference on Machine Learning (ICML), volume
235, 2024.

[54] Feng Xia, Ke Sun, Shuo Yu, Abdul Aziz, Liangtian Wan, Shirui Pan, and Huan Liu. Graph
Learning: A Survey. IEEE Transactions on Artificial Intelligence, 2(2):109–127, April 2021.
ISSN 2691-4581. doi: 10.1109/TAI.2021.3076021.

[55] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? In International Conference on Learning Representations (ICLR), 2019.

[56] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov,
and Alexander Smola. Deep sets, 2018.

[57] Qi Zhao and Yusu Wang. Learning metrics for persistence-based summaries and applications
for graph classification. In Advances in Neural Information Processing Systems (NeurIPS),
2019.

[58] Qi Zhao, Ze Ye, Chao Chen, and Yusu Wang. Persistence Enhanced Graph Neural Network. In
International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

[59] Yangze Zhou, Gitta Kutyniok, and Bruno Ribeiro. OOD link prediction generalization capa-
bilities of message-passing GNNs in larger test graphs. In Advances in Neural Information
Processing Systems (NeurIPS), 2022.

13


---Page Break---
A
Notation

Table 5 summarizes the main mathematical symbols and abbreviations used in this work.

Table 5: Summary of notation and abbreviations.

Notation
Description

G = (V, E)
arbitrary graph with vertices V and edges E

Dg(G)
persistence diagram of cardinality card(Dg(G))

ω(·)
arbitrary weight function in the PersLay architecture, R2 7→R

W (ω)
parameter vector (matrix) of the ω (weight) function

h
maximum dimensionality of the parameter vectors (matrices)

φ(·)
arbitrary point transformation, in the PersLay architecture R2 7→Rh

φΛ, φΓ, φΨ
Triangle/Gaussian/Line point transformation

ti
ith parameter value(s) of a point transformation — following Carrière et al. [5]

W (φ)
vector comprising all parameters of the point transformation φ, i.e., W φ = vec({ti}h
i=1)

Agg
arbitrary aggregation function (typically sum, mean or k-max)

PERSLAY
a mapping from persistence diagrams to Rh (Equation 2)

K
number of classes

Lipi
Lipschitz constant of an activation function at ith layer

ψi
non-linear Lipi-Lipschitz activation function before ith layer

MLP
multi-layer perceptron with non-linear Lipi-Lipschitz activation functions

vec(·)
function that converts its input to a single vector

D
data distribution

S, m
S is a training set consisting of m input pairs (x, y)

γ
margin scalar used in the margin-based loss

ˆLS,γ(gw)
empirical error (γ-margin loss) of a hypothesis g (with parameters w) on S

LD,γ(gw)
generalization error (γ-margin loss) of a hypothesis g (with parameters w) on D

ℓ2
L-2 norm

B
the maximum norm of the input to the with respect to ℓ2 norm.

DKL
KL-divergence between two distributions

| · |
absolute value

|| · ||2, || · ||F
matrix (or vector) norm induced by vector 2-norm / Frobenius norm of a matrix

U
perturbation of the model parameter

n, n1, n2
number of parameter matrices (vectors)

k
typically the number of models we compose

T
quantity of weights bounding max norn=m and max perturbation of a model in Lemma 2

Si
correcting quantity appearing in max perturbation bound of a model in Lemma 2

C1
constant appearing in max norm upper-bound of the model output in Lemma 2

C2
constant appearing in max perturbation of a model upper-bound in Lemma 2

ηi, ¯η
maximum allowed ratio of ∥Ui∥2 to ∥Si∥2 / min of ηi in Lemma 2

β, ˜β
T
nP

i=1

1
Si / arbitrary approximation of β appearing in Lemma 2

14


---Page Break---
B
Model descriptions

MLP.

Hk = ψk(WkHk−1)
(k-th layer),
Hn = WnHn−1,
(Final layer),
(11)

where Hk ∈Rhk are the vectors computed after layer k, and Wk ∈Rhk−1×hk, k ∈[n] are the
parameters in the k-th layer, and we let H0 = x. ψk, k ∈[n −1] are some element-wise non-linear
Lipk-Lipschitz functions. Note that we change d (number of layers in Neyshabur et al. [40] notation)
to n to be consistent with the number of parameters notation.

GCN.

Hk = ψk(PGHk−1Wk)
(k-th Graph Convolution Layer),

Hn =
1
|V |1|V |Hn−1Wn
(Readout layer),
(12)

where Hk ∈R|V |×hk, k ∈[n −1] are the node representations in each layer, Hn ∈R1×K is the
readout, and Wk ∈Rhk−1×hk, k ∈[n] are the parameters in the k-th layer. And we let H0 = Z,
where Z is the matrix constructed from node features, zv for v ∈V . The matrix PG ∈R|V |×|V | is
related to the graph structure and ψk(·), k ∈[n−1] are some element-wise non-linear Lipk-Lipschitz
mappings. Practically, for GCN, we take PG as the Laplacian of the graph, defined as ˜D−1/2 ˜A ˜D−1/2,
where ˜A is the adjacency matrix with +1 on the diagonal and ˜D = diag
P|V |
j=1 ˜Aij, i ∈[|V |]

is

the degree matrix of ˜A. Note that we sightly changed (comparing to Liao et al. [33], Sun and Lin
[48]) the notation to be consistent: instead of l, number of layers, we have n, instead of n, number of
nodes, we have |V | and instead of σi we have ψi as activation functions.

MPGNN.

Mk = g(C⊤
outHk−1)
(k-th step Message Computation)
¯
Mk = CinMk
(k-th step Message Aggregation)

Hk = ϕ
 
XW1 + ρ
  ¯
Mk

W2

(k-th step Node State Update)

Hn =
1
|V |1|V |Hn−1W3
(Readout Layer),
(13)

where k ∈[n −1], Hk ∈R|V |×hk are node representations/states and Hn ∈R1×K is the output
representation. Here we initialize H0 = 0. WLOG, we assume ∀k ∈[n −1], Hk ∈R|V |×h and
Mk ∈R|V |×h since h is the maximum hidden dimension. Cin ∈R|V |×|E| and Cout ∈R|V |×|E| are
the incidence matrices corresponding to incoming and outgoing nodes1 respectively. Specifically,
rows and columns of Cin and Cout correspond to nodes and edges respectively. Cin[i, j] = 1 indicates
that the incoming node of the j-th edge is the i-th node. Similarly, Cout[i, j] = 1 indicates that the
outgoing node of the j-th edge is the i-th node. g, ϕ, ρ are nonlinear mappings, e.g., ReLU and Tanh.
Technically speaking, g : Rh →Rh, ϕ : Rh →Rh, and ρ : Rh →Rh operate on vector-states of
individual node/edge. However, since we share these functions across nodes/edges, we can naturally
generalize them to matrix-states, e.g., ˜ϕ : R|V |×h →R|V |×h where ˜ϕ(X)[i, :] = ϕ(X[i, :]). By
doing so, the same function could be applied to matrices with varying size of the first dimension.
For simplicity, we use g, ϕ, ρ to denote such generalization to matrices. We denote the Lipschitz
constants of g, ϕ, ρ under the vector 2-norm as Cg, Cϕ, Cρ respectively. We also assume g(0) = 0,
ϕ(0) = 0, and ρ(0) = 0 and define the percolation complexity as C = CgCϕCρ∥W2∥2 following
Garg et al. [14]. Note that we sightly changed (comparing to Liao et al. [33], Sun and Lin [48]) the
notation to be consistent: instead of l, number of layers, we have n, instead of n, number of nodes,
we have |V | and instead of Wl, parameter matrix, we have W3.

1For undirected graphs, we convert each edge into two directed edges.

15


---Page Break---
C
Proofs of main lemma and composition lemmas

Lemma 2. Let fw : X 7→RK be a model with parameters w = vec{W1, ..., Wn}. If there exists
T ∈R and Si ∈R for i ∈[n] depending on W1, ..., Wn; C1, C2 > 0; ηi ∈(0, 1] for i ∈[n] such
that:

maximum norm of the output is bounded by T and C1:

max
x∈X ∥fw(x)∥2 ≤C1T,
(14)

maximum change of the output with perturbed weights, u = vec{U1, ..., Un}, is bounded if the
perturbation is small, ∥Ui∥2 ≤ηiSi:

max
x∈X ∥fw+u(x) −fw(x)∥2 ≤C2T

 n
X

i=1

∥Ui∥2

Si

!

,
(15)

arithmetic mean of Si inverses is at least the nth root of T inverse:

1
n

 n
X

i=1

1
Si

!

≥
1
T 1/n ,
(16)

minimum ηi is at most the ratio of C1 and 2C2:

¯η := min
1≤i≤n ηi ≤C1

2C2
,
(17)

then for any γ, δ > 0 with probability at least 1 −δ over the choice of training set S of size m
sampled accordingly to some data distribution D we have the bound on the generalization gap:

LD,0(fw) ≤ˆLS,γ(fw)+

+ O









v
u
u
u
tmax{1, ∥w∥2
2}T 2
 nP

i=1

1
Si

2
(h ln nh) C2
1 ¯η−2 + log m

δ max
n
1,
1
C1

o

γ2m








,
(18)

where h is the upper bound for dimensions of Wi.

Proof. First, we note that if w such that C1T ≤γ

2 , then this imply for any i ∈[K]

max
x∈X |fw(x)i| ≤max
x∈X ∥fw(x)∥2 ≤C1T ≤γ

2 ,

where we used Equation 6 and the fact that every coordinate of the vector is at most ℓ2-norm of
the vector. In this case the empirical loss would be equal to one, since no index i can satisfy
fw(x)i ≤γ + max
i̸=j fw(x) and the inequality in Equation 10 becomes trivial since generalization

error can not be greater than 1.

So from now on we consider w such that C1T ≥γ

2 . Moreover we note that:

C1T ≥γ

2 ⇒T ≥
γ
2C1
(19)

We denote

β := T

 n
X

i=1

1
Si

!

.
(20)

Following Neyshabur et al. [40] and Liao et al. [33], we consider the prior P = N(0, σ2I) and
random perturbation u ∼N(0, σ2I). Note that the σ of the prior and the perturbation are the same

16


---Page Break---
and will be set according to β. More precisely, we will set the σ based on some approximation ˜β of
β since the prior cannot depend on any learned weights directly. The approximation ˜β is chosen to be
a cover set which covers the meaningful range of β. For now, let us assume that we have a fixed ˜β
and consider β which satisfies |β −˜β| ≤εβ for some ε > 0. Note that this also implies:

|β −˜β| ≤εβ ⇒β ≤
1
1 −ε
˜β
(21)

|β −˜β| ≤εβ ⇒˜β ≤(1 + ε)β
(22)

This setup is very like the setup in Neyshabur et al. [40] and Liao et al. [33]

Choosing σ.
From Tropp [49] we know that if U ∈Rh×h ∼N(0, σ2I), then

P (∥U∥2 ≥t) ≤2h exp

−t2

2hσ2



So applying it to U1, ..., Un (all of them are smaller than h × h) and t = σ
√

2h ln 4nh we have:

P (||U1||2 ≤σCt & ... & ||Ud||2 ≤σCt) ≥1 −

d
X

i=1
P (||Ui||2 ≥σCt) ≥

≥1 −2dhexp

−σ2C2
t
2hσ2


= 1 −2dh exp

ln
1
4dh


= 1

2

(23)

For simplicity of notation we denote Ct =
√

2h ln 4nh .

Using defined perturbation u = vec{U1, ..., Un} and Equation 23 combined with Equation 7 we get
with probability at least 1

2:

max
x∈X ∥fw+u(x) −fw(x)∥∞≤max
x∈X ∥fw+u(x) −fw(x)∥2 ≤

≤C2T

 n
X

i=1

∥Ui∥2

Si

!

≤

≤C2βσCt ≤

≤C2Ct
σ ˜β
1 −ε

(24)

If

σ ≤γ(1 −ε)

4C2Ct ˜β
,

then the perturbation is bounded by γ

4 . However, we need to satisfy the condition of Equation 7, i.e.
∀i ∈[n] : ∥Ui∥≤ηiSi. Let us find Cσ ≥1 such that:

σ :=
γ(1 −ε)
4C2CσCt ˜β
≤γ(1 −ε)

4C2Ct ˜β
(25)

and that

∀i ∈[n] : σCt ≤ηiSi

17


---Page Break---
σCt = γ(1 −ε)

4C2Cσ ˜β

(i)
≤
γ
4C2Cσβ

(ii)
=

(ii)
=
γηiSi
4C2CσηiSiβ

(iii)
=
γ ηiSi

4C2CσηiT

 

1 + P

k̸=i

Si
Sk

!
(iv)
≤

(iv)
≤
γ ηiSi
4C2CσηiT

(v)
≤
γ ηiSi
4C2Cσηi
γ
2C1
=

= ηiSi
C1
2C2Cσ ηi
.

(26)

(i) comes from Equation 21. (ii) we just multiplied and divided on ηiSi. (iii) we plugged in the
expression for β, Equation 20. (iv) 1 + P

k̸=i

Si
Sk ≥1. (v) comes from Equation 19

We choose
Cσ :=
C1
2C2
min
1≤i≤n ηi
=
C1
2C2 ¯η .
(27)

In this case Cσ ≥1 because of Equation 9 and continuing Equation 26:

σCt ≤ηiSi
C1
2C2Cσηi
= ηiSi
¯η
ηi

(i)
≤ηiSi,

where (i) comes from the fact that ¯η is the minimum among ηis, gives us that ∀i ∈[n] : ∥Ui∥2 ≤ηiSi.

Plugging this into expression for σ, Equation 25, we get:

σ = γ(1 −ε)

4C2Cσ ˜β
= γ(1 −ε)¯η

2C1Ct ˜β
(28)

Getting the bound for weights |β −˜β| ≤1/2β.
Now, we can compute the KL-divergence and get
the bound using Lemma 1 for case when β is around ˜β, i.e. w such that |β −˜β| ≤εβ.

DKL(Q(w + u)||P) = ∥w∥2
2
2σ2

(i)
= ∥w∥2
2
2
4C2
1C2
t ˜β2

γ2(1 −ε)2¯η2

(ii)
≤

(ii)
≤2∥w∥2
2
C2
1C2
t

1+ε
1−ε
2
β2

γ2¯η2

(29)

where (i) comes from σ definition, Equation 28, (ii) comes from the upper-bound on ˜β, Equation 22.

We are ready to put this into Lemma 1:

LD,0(fw) ≤ˆLS,γ(fw) + 4

s

DKL(Q(w + u)||P) + log 6m

δ
m −1
=

= ˆLS,γ(fw) + O







v
u
u
tmax{1, ∥w∥2
2}β2C2
1C2
t

1+ε
1−ε
2
¯η−2 + ln m

δ
γ2m





,

(30)

where we upper-bounded ∥w∥2
2 as max{1, ∥w∥2
2} to simplify further derivations.

Union bound.
Finally, we need to consider multiple choices of ˜β so that for any β, we can bound
the generalization error like Equation 30. In order to do this we (i) find interval of reasonable βs,
(ii) define a covering and upper-bound the number of balls in the covering, (iii) combine bounds for
every ˜β together and get the final result.

18


---Page Break---
If β is too large, the KL-divergence would be too large and generalization gap would be greater than
1 which thereby trivialize the bound since the generalization error cannot be greater than 1.

v
u
u
tmax{1, ∥w∥2
2}β2C2
1C2
t

1+ε
1−ε
2
¯η−2 + ln m

δ
γ2m

(i)
≥

(i)
≥

s

β2C2
1
γ2m¯η2

(ii)
≥
C1β
γ√m,

(31)

where (i) holds since we remove multipliers that are greater than 1 and ln m

δ which is ≥0 and appears
in sum, (ii) comes from the fact that ¯η ≤1.

So, if β ≥γ√m

C1 , then the bound becomes trivial.

On the other hand, if the β is too small, then the empirical loss would be too large since the model
would not be able to classify with margin, γ, as it was shown in Equation 19. Using this equation and
Arithmetic versus Geometric mean we get the following inequality:

β = T

n
X

i=1

1
Si

(i)
≥nT
n−1

n
(ii)
≥n
 γ

2C1

 n−1

n
,
(32)

where (i) comes from Equation 8 and (ii) comes from Equation 19.

Combining lower (Equation 32) and upper (Equation 31) bound on β we get:

n
 γ

2C1

 n−1

n
≤β ≤γ√m

C1
(33)

To satisfy the condition |β −ˆβ| ≤εβ we can take εn

γ
2C1

 n−1

n
as the radius of the covering, C. In

this case let us derive the upper bound for the size of the covering, |C|:

|C| ≤
√m

εn

 γ

2C1

 1

n
≤2√m 1

n

 1

C1

1/n
(34)

Let us consider cases:







1
n

1
C1

1/n
≤
1
C1 ,
C1 < 1

1
n

1
C1

1/n
≤1,
C1 ≥1

First inequality holds because function a1/x

x
is monotonously decreasing on the interval (0, +∞). So,
we can combine:

|C| ≤2√m max

1, 1

C1


(35)

It is left to apply the union bound argument for the events of Equation 30 happening with ˜β taking
value from ith ball. Let us denote such event Ei:

P
 
E1 & ... & E|C|

= 1 −P(∃i : not Ei) ≥1 −

|C|
X

i=1
P(not Ei) ≥1 −|C|δ
(36)

19


---Page Break---
Hence with probability at least 1 −δ for all w we have:

LD,0(fw) ≤ˆLS,γ(fw)+

+ O









v
u
u
u
tmax{1, ∥w∥2
2}T 2
 nP

i=1

1
Si

2
C2
t C2
1 ¯η−2 + log m

δ max
n
1,
1
C1

o

γ2m









Note that we substituted ε with 1

2 and now max{1,
1
C1 } appearing under logarithm comparing to
Equation 30.

Lemma 5 (Generalization of Lemma 3). Let f (1)
w1 : X1 7→Rh1, ..., f (k)
wk : Xk 7→Rhk for k ∈N
with wi = vec{W (i)
1 , ..., W (i)
ni } and h1 = K be some models that we can compose, i.e. f =
f (1)(f (2)(...(f (k))...)). If there exists for j ∈[k] : T (j) depending on W (j)
1 , ..., W (j)
nj ; for j ∈
[j] : S(j)
1 , ..., S(1)
nj ∈R depending on W (j)
1 , ..., W (j)
nj ; η(1)
1 , ..., η(1)
n1 , ..., η(k)
1 , ..., η(k)
nk ∈(0, 1] and
C(1)
1 , ..., C(k)
1
∈R, C(1)
2 , ..., C(k)
2
∈R such that:

maximum output for every model is bounded as follows:

∀j ∈[k], ∀x ∈Xj : ∥f (j)
wj (x)∥2 ≤C(j)
1 T (j)∥x∥2,
(37)

and for j = k we can have weaker condition – the same as in Lemma 2,

maximum perturbation of the output during small perturbation of weights, ∥U (j)
i
∥2 ≤η(j)
i S(j)
i
, for
every model is bounded as follows:

∀j ∈[k], ∀x, ∆x ∈Xj : ∥fwj+uj(x + ∆x) −fwj(x)∥2 ≤

≤C(j)
2 T (j)
 

∥∆x∥2 + ∥x∥2

nj
X

i=1

∥U (j)
i
∥2
∥S(j)
i
∥2

!

,
(38)

and for j = k we can have weaker condition – the same as in Lemma 2,

arithmetic mean of inverses of S(j)
i
is greater than nj-root of T (j):

∀j ∈[k] :
1
nj

nj
X

i=1

1

S(j)
i
≥
 1

T (j)

1/nj
,
(39)

and ¯η(j) := min
i∈[nj] η(j)
i
is upper-bounded by the ration C(j)
1
and C(j)
2 :

∀j ∈[k] :
C(j)
1
2C(j)
2
≥¯η(j),
(40)

we denote n =
kP

j=1
nj, then f meets requirements of Lemma 2 with

20


---Page Break---
T =

k
Y

i=1
T (i)
and
S1:n = S(1)
1 , ..., S(1)
n1 , ..., S(k)
1 , ..., S(k)
nk

η1:n = η(1)
1 , ..., η(1)
n1 , ..., η(k)
1 , ..., η(k)
nk

C1 = max
x∈Xk ∥x∥2 max






k
Y

j=1
C(j)
1 , C · C(ind)
1
C(ind)
2






C2 = max
x∈Xk ∥x∥2C,

where ind = arg min
j∈[k] ¯η(j) and C = max
j∈[k]

jQ

i=1
C(i)
2

kQ

i=j+1
C(i)
1

Proof. We denote as f k:j composition of models from k to j

First we test Equation 6 by plugging in Equation 37 for every model:

max
x∈Xk ∥f k:1(x)∥2 ≤max
x∈Xk C(1)
1 T (1)∥f k:2(x)∥2 ≤

≤max
x∈Xk ∥x∥2

k
Y

j=1
C(i)
1 T (i) ≤C1T
(41)

Next, we test Equation 7 by plugging in Equation 38 and Equation 37 for every model:

max
x∈Xk ∥f k:1
w+u(x) −f k:1
w (x)∥2 ≤

≤max
x∈Xk C(1)
2 T (1)
 

∥f k:2
w+u(x) −f k:2
w ∥2 + ∥f k:2
w (x)∥2

 n1
X

i=1

∥U (1)
i
∥2
S(1)
i

!!

≤

≤max
x∈Xk ∥x∥2

k
Y

j=1
T (j)




k
X

j=1

jY

i=1
C(i)
2

(k)
Y

i=j+1
C(i)
1

nj
X

i=1

∥U (j)
i
∥2
S(j)
2



≤

≤C2T

k
X

j=1

nj
X

i=1

∥U (j)
i
∥2
S(j)
i

(42)

To test Equation 8 we use arithmetic versus geometric mean inequality:

T

k
X

j=1

nj
X

i=1

1

S(j)
i
≥

k
X

j=1
nj

T (j) nj −1

nj
Y

i̸=j
T (i) ≥

≥n




k
Y

j=1





T (j) nj −1

nj
Y

i̸=j
T (i)





nj



1/n

≥

≥nT
n−1

n

(43)

It is left to verify Equation 9. As one can notice
C1
2C2 ≥C(ind)
1
C(ind)
2
≥¯η.

Lemma 6 (Generalization of Lemma 4). Let f (1)
w1 : X 7→Rh1, ..., f (k)
wk : X 7→Rhk with w1 =
vec{W (1)
1
, ..., W (1)
n1 }, ..., wk = vec{W (k)
1
, ..., W (k)
nk } be some models satisfying Lemma 2 conditions.
If Agg : Rh1 × ... × Rhk 7→RK such that:

∀x1, y1 ∈Rh1, ..., xk, yk ∈Rhk : ∥Agg(x1, ..., xk) −Agg(y1, ..., yk)∥2 ≤A

k
X

j=1
∥xj −yj∥2

21


---Page Break---
for some A > 0 and Agg(0, ..., 0) = 0. Also we denote n =
kP

j=1
nj, then Agg(f (1)
w1 (·), ..., f (k)
wk (·))

satisfies Lemma 2 conditions with either:

T = max




T (1), ..., T (k),

k
Y

j=1
T (j)





and
S1:n = S(1)
1 , ..., S(1)
n1 , ..., S(k)
1 , ..., S(k)
nk

η1:n = η(1)
1 , ..., η(1)
n1 , ..., η(k)
1 , ..., η(k)
nk

C1 = A max






k
X

j=1
C(j)
1 , max
j∈[k] C(j)
2
· C(ind)
1
C(ind)
2






C2 = A max
j∈[k] C(j)
2 ,

or with

T = max






k
X

j=1
T (j),

k
Y

j=1
T (j)





and
S1:n = S(1)
1 , ..., S(1)
n1 , ..., S(k)
1 , ..., S(k)
nk

η1:n = η(1)
1 , ..., η(1)
n1 , ..., η(k)
1 , ..., η(k)
nk

C1 = A max

(

max
j∈[k] C(j)
1 , max
j∈[k] C(j)
2
· C(ind)
1
C(ind)
2

)

C2 = A max
j∈[k] C(j)
2 ,

where ind = arg min
j∈[k] ¯η(j).

Proof. First, we test Equation 6 for composite model by applying assumption about Agg and its
Equation 6 for every f (j):

max
x∈X ∥Agg(f (1)
w1 (x), ..., f (k)
wk (x))∥≤A

k
X

j=1
∥f (j)
wj (x)∥2 ≤












A
kP

j=1
T (j) max
j∈[k] C(j)
1

A max
j∈[k] T (j)
kP

j=1
C(j)
1

(44)

To test Equation 7 for composite model we again apply assumption about aggregation function and
its Equation 7 version for every f (j):

max
x∈X ∥Agg(f (1)
w1+u1(x), ..., f (k)
wk+uk(x)) −Agg(f (1)
w1 (x), ..., f (k)
wk (w))∥2 ≤

≤A

k
X

j=1
∥f (j)
wj+uj(x) −f (j)
wj (x)∥2 ≤A

k
X

j=1
C(j)
2 T (j)
nj
X

i=1

∥U (j)
i
∥2
S(j)
i
≤

≤AT max
j∈[k] C(j)
2

k
X

j=1

ni
X

i=1

∥U (j)
i
∥

S(j)
i

(45)

To test Equation 8 for composite model we employ arithmetic vs geometric mean together with
Equation 8 for every model:

T

k
X

j=1

nj
X

i=1

1

S(j)
i
=

k
X

j=1

T
T (j)

nj
X

i=1

T (j)

S(j)
i
≥

k
X

j=1
nj
T
(T (j))1/nj ≥n
T
 
kQ

j=1
T (j)
!1/n ≥nT
n−1

n ,
(46)

22


---Page Break---
where the last inequality comes from the fact that
kQ

j=1
T(j) ≤T

It is left to test Equation 9. As one can notice in either case
C1
2C2 ≥
C(ind)
1
2C(ind)
2
≥¯η

D
Perturbation analysis

Lemma 7. Let w = vec{W1, ..., Wn} be a vector of weight matrices of an n-layer MLP, fw : X 7→
RK. Let ψi be a Lipi-Lipschitz activation function after layer i, for i ∈[n −1]. Then for any
input and input perturbation, x, ∆x ∈X, weight perturbation, u = vec{U1, ..., Un}, and constants,
η1, ..., ηn, such that for i ∈[n] : ∥Ui∥2 ≤ηi∥Wi∥2, we have two inequalities:

∥fw(x)∥2 ≤

 n
Y

i=1
Lipi∥Wi∥2

!

|x|2
(47)

∥fw+u(x + ∆x) −fw(x)∥2 ≤

 n
Y

i=1
1 + ηi

!
n
Y

i=1
Lipi∥Wi∥2

 

|∆x|2 + |x|2

n
X

i=1

∥Ui∥2
∥Wi∥2

!

, (48)

where we denote Lipn = 1.

Proof. The proof follows the Neyshabur et al. [40] with some modifications concerning the ∆x and
Lipi and ηi.

We denote truncated f after i + 1th layer as f (i+1). First, we provide the bound on the norm of the
output after i + 1th layer:

∥f (i+1)
w
(x)∥2
(i)
= ∥Wi+1(ψi(f (i)
w (x))∥2

(ii)
≤∥Wi+1∥2∥ψi(f (i)
w (x))∥2

(iii)
≤

(iii)
≤∥Wi+1∥2Lipi∥f (i)
w (x)∥2,
(49)

where (i) is the definition of MLP, (ii) comes from the definition of operator norm, (iii) comes from
Lipschitzness. Unrolling the recursion of Equation 49 will get us (we denote f (0)
w (x) to be x):

∥f (i+1)
w
(x)∥2 ≤

 i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!

|x|2
(50)

Now let us provide the bound for the perturbation after i + 1th layer. We denote this perturbation as
∆i+1:

∆i+1
(i)
=
f (i+1)
w+u (x + ∆x) −f (i+1)
w
(x)

2

(ii)
=

(ii)
=
(Wi+1 + Ui+1)ψi(f (i)
w+u(x + ∆x)) −Wi+1ψi((f (i)
w (x)))

2

(iii)
=

(iii)
=
(Wi+1 + Ui+1)(ψi(f (i)
w+u(x + ∆x)) −ψi(f (i)
w (x))) + Ui+1ψi(f (i)
w (x))

2

(iv)
≤

(iv)
≤∥Wi+1 + Ui+1∥2∥ψi(f (i)
w+u(x + ∆x)) −ψi(f (i)
w (x))∥2 + ∥Ui+1∥2∥ψi(f (i)
w (x))∥2

(v)
≤

(v)
≤(1 + ηi+1) ∥Wi+1∥2Lipi∆i + ∥Ui+1∥2∥x∥2

iY

k=1
Lipk

iY

k=1
∥Wk∥2
(vi)
=

(vi)
= (1 + ηi+1)∥Wi+1∥2Lipi∆i + |x|2

 i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!
∥Ui+1∥2
∥Wi+1∥2
(51)

23


---Page Break---
where (i) comes from the definition of ∆i+1, (ii) comes from the defintion of an MLP, (iii) add
and substract Ui+1ψi(f (i)
w (x)), (iv) comes from triangle inequality combined with operator norm
definition, (v) comes from the definition of ηi+1, Lipschitzness and Equation 50, (vi) comes from
multiplying and dividing by ∥Wi+1∥2 the second term. Now we unroll the recursion of Equation 51

∆i+1

(i)
≤(1 + ηi+1)∥Wi+1∥2Lipi∆i + ∥x∥2

 i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!
∥Ui+1∥2
∥Wi+1∥2

(ii)
≤

(ii)
≤

 i+1
Y

k=1
1 + ηk

!  i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!

∥∆x∥2 + ∥x∥2

 i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!  i+1
X

k=1

∥Uk∥2
∥Wk∥2

!
(iii)
≤

(iii)
≤

 i+1
Y

k=1
1 + ηk

!  i+1
Y

k=1
∥Wk∥2

!  
iY

k=1
Lipk

!  

∥∆x∥2 + ∥x∥2

i+1
X

k=1

∥Uk∥2
∥Wk∥2

!

,

(52)
where (i) comes from Equation 51, (ii) comes from unrolling the recursion ∆i, (iii) comes from the
fact that 1 + ηk ≥1.

Lemma 8 (GCN perturbation analysis). Let fw : X 7→RK with w = vec{W1, ..., Wn} be a n-layer
GCN model. Let ψi be Lipi activation functions after layer i ∈[n −1]. If u = vec{U1, ..., Un},
B ∈R are such that: ∀i ∈[n] : ∥Ui∥2 ≤
1
n∥Wi∥2 and ∀G ∈X, ∀v ∈G : |zv|2 ≤B, and
∀G ∈X, G is simple and have maximum degree of d, then for any G = (V, E, z) ∈X

|fw+u(G) −fw(G)|2 ≤
1
p

|V |
∥Z∥F ∥PG∥n−1
2

 n
Y

i=1
Lipi∥Wi∥2

!
n
X

i=1

∥Ui∥2
∥Wi∥2
≤

≤eB

 n
Y

i=1
Lipi∥Wi∥2

!
n
X

i=1

∥Ui∥2
∥Wi∥2
, ≤

≤eBd(n−1)/2
 n
Y

i=1
Lipi∥Wi∥2

!
n
X

i=1

∥Ui∥2
∥Wi∥2

(53)

where Z ∈R|V |×dz is the matrix consisting of zv for v ∈V , and PG is a Laplacian of the graph
G which is equal to ˜D−1/2 ˜A ˜D−1/2 where ˜A is an adjacency matrix with +1 on the diagonal and

˜D = diag

 
|V |
P

j=1
˜Aij, i ∈[|V |]

!

and d is maximum degree of a graph in X.

Proof. The proof follows [48] except for the fact that we add Lipi to the bound. This change is
straightforward, however, for the completeness of the picture we provide this proof here. The detailed
description of the architectures can be found in Appendix B.

First we prove two helpful propositions

Proposition 1. For any matrix A ∈Rn×m, B ∈Rm×p, we have,

∥AB∥F ≤∥A∥F ∥B∥2

Proof. Let x⊤
i , a⊤
i be ith row of AB and A respectively, then we have:

∥AB∥2
F ≤
X

i=1
∥x⊤
i |2
2 =

n
X

i=1
∥a⊤
i B∥2
2 ≤

n
X

i=1
∥a⊤
i ∥2
2∥B∥2
2 = ∥A∥2
F ∥B∥2
2

Proposition 2. For any undirected graph G = (V, E), let A ∈R|V |×|V | be the adjacency matrix,
D = diag(D1, . . . , D|V |) be the degree matrix, and d = maxi∈[|V |]{Di} be the maximum degree,

24


---Page Break---
where Di = P|V |
j=1 Aij, i ∈[|V |]. Then we have,

(i) ∥A∥2 ≤d;

(ii)
 ˜D−1

2 ˜A ˜D−1

2

2 ≤1, where ˜A = A + I and ˜D = diag( ˜D1, . . . , ˜D|V |) = diag





|V |
X

j=1
˜Aij, i ∈[|V |]



.

Proof. For (i), by the definition of spectral norm, we have

∥A∥2 = max
∥x∥2=1 x⊤Ax = max
∥x∥2=1

X

(i,j)∈E
xixj ≤max
∥x∥2=1

X

(i,j)∈E

x2
i + x2
j
2
≤max
∥x∥2=1 d
X

i∈V
x2
i = d.

For (ii), let ˜E = E ∪{(i, i)|i ∈V } be the edge set associated with the adjacency matrix ˜A. By the
definition of spectral norm, we have
D−1

2 ˜AD−1

2

2 = max
∥x∥2=1 x⊤
D−1

2 ˜AD−1

2

x = max
∥x∥2=1

X

(i,j)∈˜
E

xixj
q

˜Di ˜Dj
≤

≤max
∥x∥2=1

X

(i,j)∈˜
E

 
x2
i
2 ˜Di
+ x2
j
2 ˜Dj

!

≤max
∥x∥2=1

X

i∈V
x2
i = 1.

Before proving the first inequality we note that
1
√

|V |∥X∥F ≤B and ∥PG∥n−1
2
≤1, so the second

inequality is rather straightforward. The last inequality comes from the fact that d ≥1.

We denote the node representation in the j-th (j ≤l) layer as

f j
w(G) = Hj = ψj(PGHj−1Wj),
j ∈[n −1],

f n
w(G) = Hn =
1
|V |1|V |Hn−1Wn.

Adding perturbation u to the parameter w, that is, for the j-th (j ≤n) layer, the perturbed parameters
are Wj + Uj and denote H′
j = f j
w+u(G), j ∈[n].

Upper Bound on the Node Representation.
For any j < n,

∥Hj∥F = ∥ψj(PGHj−1Wj)∥F ≤
≤Lipj∥PGHj−1Wj∥F ≤

≤Lipj∥PGHj−1∥F ∥Wj∥2 ≤

≤Lipj∥PG∥2∥Hj−1∥F ∥Wj∥2,

where the first inequality holds since ψj is a Lipschitz and ψj(0) = 0, and the second and the last
ones hold by Proposition 1. Then, unrolling the recursion and setting H0 = X, we have

∥Hj∥F ≤∥PG∥j
2∥H0∥F

jY

i=1
Lipi∥Wi∥2 ≤

≤∥Z∥F ∥PG∥j
2

jY

i=1
Lipi∥Wi∥2.

(54)

Upper Bound on the Change of Node Representation.
For any j < |V |,

∥H′
j −Hj∥F = ∥ψj(PGH′
j−1(Wj + Uj)) −ψj(PGHj−1Wj)∥F
≤Lipj∥PGH′
j−1(Wj + Uj) −PGHj−1Wj∥F
(55)

25


---Page Break---
Using the triangle inequality, we have

∥H′
j −Hj∥F ≤Lipj∥PG(Wj + Uj)(H′
j−1 −Hj−1)∥F + ∥PGHj−1Uj∥F .

The first term can be bounded as

∥PG(H′
j−1 −Hj−1)(Wj + Uj)∥F = ∥PG∥2∥H′
j−1 −Hj−1∥F ∥Wj + Uj∥2,

and the second term can be bounded as

∥PGHj−1Uj∥F = ∥PG∥2∥Hj−1∥F ∥Uj∥2.

Therefore,

∥H′
j −Hj∥F ≤Lipj∥PG∥2∥H′
j−1 −Hj−1∥F ∥Wj + Uj∥2 + Lipj∥PG∥2∥Hj−1∥F ∥Uj∥2.
(56)

Unrolling the recursion while simplifying notation as: ∥Hj −H′
j∥F ≤aj−1∥H′
j−1 −Hj−1∥F +bj−1
we get:

∥H′
j −Hj∥F ≤

j−1
X

k=0
bk

 j−1
Y

i=k+1
ai

!

=

=

j−1
X

k=0
Lipk+1∥PG∥∥Hk∥F ∥Uk+1∥2

 j−1
Y

i=k+1
Lipi+1∥PG∥2∥Wi+1 + Ui+1∥2

!

=

=

j−1
X

k=0
∥PG∥j−k
2
∥Hk∥F ∥Uk+1∥2

j−1
Y

i=k
Lipi+1

j−1
Y

i=k+1
∥Wi+1 + Ui+1∥2.

Plugging in Equation 54, we have:

∥H′
j −Hj∥F ≤

jY

i=1
Lipi

j−1
X

k=0
∥PG∥j−k
2

 

∥PG∥k∥Z∥F

k
Y

i=1
∥Wi∥2

!

∥Uk+1∥2

j−1
Y

i=k+1
∥Wi + Ui∥2 ≤

≤∥Z∥F

jY

i=1
Lipi

j−1
X

k=0
∥PG∥j
2
∥Uk+1∥2
∥Wk+1∥2

k+1
Y

i=1
∥Wi∥2

j−1
Y

i=k+1


1 + 1

n


∥Wi∥2 =

= ∥Z∥F ∥PG∥j
2

jY

i=1
∥Wi∥2

jY

i=1
Lipi

j
X

k=1

∥Uk∥2
∥Wk∥2


1 + 1

n

j−k

(57)

Final Bound on the Readout Layer.

∥H′
n −Hn∥2 =

1
|V |1|V |H′
n−1(Wn + Un) −1

|V |1|V |Hn−1Wn


2
=

=

1
|V |1|V |(H′
n−1 −Hn−1)(Wn + Un) + 1

|V |1|V |Hn−1Un


2
≤

≤

1
|V |1|V |(H′
n−1 −Hn−1)(Wn + Un)

2
+

1
|V |1|V |Hn−1Un


2
≤

≤

1
|V |1|V |


2
∥(H′
n−1 −Hn−1)(Wn + Un)∥2 +

1
|V |1|V |


2
∥Hn−1Un∥2 ≤

≤
1
p

|V |
∥H′
n−1 −Hn−1∥F ∥Wn + Un∥2 +
1
p

|V |
∥Hn−1∥F ∥Un∥2,

where in a last inequality we first apply that ∥A∥2 ≤∥A∥F for any matrix A and then use Proposi-
tion 1.

Using Equation 54 and Equation 57, we have:

26


---Page Break---
X sorted by f:
. . .
m(k)
. . .
n(k)
. . .
n(i)
. . .

X sorted by g:
. . .
n(i)
. . .
m(k)
. . .
n(k)
. . .

Figure 5: Case g(n(k)) > g(m(k)), f(n(k)) ≥f(m(k))

X sorted by f:
. . .
m(k)
. . .
n(k)
. . .

X sorted by g:
. . .
n(k)
. . .
m(k)
. . .

Figure 6: Case g(n(k)) > g(m(k)), f(n(k)) ≤f(m(k))

∥H′
n −Hn∥2 ≤
1
p

|V |
∥Wn + Un∥2∥X∥F ∥PG∥n−1
n−1
Y

i=1
Lipi∥Wi∥2

n−1
X

k=1

∥Uk∥2
∥Wk∥2


1 + 1

n

n−k−1
+

+
1
p

|V |
∥Un∥2∥PG∥n−1
2
∥X∥F

n−1
Y

i=1
Lipi∥Wi∥2 ≤

≤
1
p

|V |
∥Z∥F ∥PG∥n−1
"

∥Wn + Un∥2

n−1
Y

i=1
Lipi∥Wi∥2

n−1
X

k=1

∥Uk∥2
∥Wk∥2


1 + 1

n

n−k−1
+ ∥Un∥2

n−1
Y

i=1
Lipi∥Wi∥2

#

=
1
p

|V |
∥Z∥F ∥PG∥n−1
n
Y

i=1
Lipi∥Wi∥2

"
∥Wn + Un∥2

∥Wn∥2

n−1
X

k=1

∥Uk∥2
∥Wk∥2


1 + 1

n

n−k−1
+ ∥Un∥2

∥Wn∥2

#

≤

≤
1
p

|V |
∥Z∥F ∥PG∥n−1
2

n
Y

i=1
Lipi∥Wi∥2

"
1 + 1

n

 n−1
X

k=1

∥Uk∥2
∥Wk∥2


1 + 1

n

n−k−1
+ ∥Un∥2

∥Wn∥2

#

≤

≤
e
p

|V |
∥Z∥F ∥PG∥n−1
2

n
Y

i=1
Lipi∥Wi∥2

n
X

k=1

∥Uk∥2
∥Wk∥2
,

where we set Lipn = 1 for simplicity of notation and last inequality holds since 1 ≤(1 + 1

n)n ≤e.

Our next lemma helps us to upper bound the PERSLAY’s perturbation, when Agg = k- max.
Lemma 9. Let X be an arbitrary finite set and f, g : X 7→R. Then we can say that:

|k- max
x∈X
f(x) −k- max
x∈X
g(x)| ≤3 max
x∈X |f(x) −g(x)|

Proof. Denote n : N 7→X by a function that maps natural number k to an element of X that would
be on kth position in order sorted by f. Denote m as an analogous function but for g. Then, we are
interested in the following expression: |f(n(k)) −g(m(k))|. Let us rewrite it:

|f(n(k)) −g(m(k))| = |f(n(k)) −g(n(k)) + g(n(k)) −g(m(k))| ≤
≤|f(n(k)) −g(n(k))| + |g(n(k)) −g(m(k))| ≤
≤max
x∈X |f(x) −g(x)| + |g(n(k)) −g(m(k))|

Now, the task is to prove that |g(n(k)) −g(m(k))| ≤2 maxx∈X |f(x) −g(x)|

Let us consider four cases:

• g(n(k)) > g(m(k)) and f(n(k)) ≥f(m(k)) (Fig. 5). In this case ∃i ∈N such that
f(n(i)) > f(n(k)) and g(n(i)) < g(m(k)). Indeed, if none of the elements "to the right"

27


---Page Break---
of n(k) moved "to the left" of m(k), then "to the right" of m(k), there are at least n −k + 1
elements; however, there are must be exactly n −k elements.

|g(n(k)) −g(m(k))| = g(n(k)) −g(m(k)) ≤g(n(k)) −g(n(i)) ≤
≤f(n(k)) + (max
x∈X |f(x) −g(x)|) −g(n(i)) <

< f(n(i)) + (max
x∈X |f(x) −g(x)|) −g(n(i)) < 2(max
x∈X |f(x) −g(x)|)

• g(n(k)) > g(m(k)) and f(n(k)) ≤f(m(k)) (Fig. 6)

|g(n(k)) −g(m(k))| = g(n(k)) −g(m(k)) ≤

≤f(n(k)) +

max
x∈X |f(x) −g(x)|

−g(m(k)) ≤

≤f(m(k)) +

max
x∈X |f(x) −g(x)|

−g(m(k)) ≤

≤2

max
x∈X |f(x) −g(x)|


• The rest of the cases can be handled analogously.

Lemma 10 (Perturbation analysis of PersLay). Let fw : G 7→Rk with w = {W (ω), W (φ)} be a
PersLay where W (ω) is a parameter vector (matrix) of weight function, ω, and W (φ) is a parameter
vector (matrix) of point-transformation function, φ. Let Dg be a mapping from graphs to (extended)
persistence diagrams with a fixed filtration function and B such that max
G∈G
max
p∈Dg(G) ∥p∥2 ≤B, then:

max
G∈G |fw(G)|2 ≤A1C(ω)T (ω)C(φ)T (φ)
(58)

and for η(ω) and u = {U (ω), U (φ)} such that ∥U (ω)∥2 ≤η(ω)T (ω), we have:

max
G∈G |fw+u(G) −fw(G)|2 ≤

≤A2 max{C(ω)Lip(φ), C(φ)Lip(ω)}(1 + η(ω))T (ω)T (φ)
∥U (φ)∥2

T (φ)
+ ∥U (ω)∥2

T (ω)


,
(59)

where A1 = max
G∈G card(Dg(G)) if Agg is sum and A1 = 1 if Agg is k-max or mean; A2 =

max
G∈G card(Dg(G)) if Agg is sum or A2 = 3 if Agg is k-max or mean;

(T (φ), C(φ), Lip(φ)) = (max{1, ∥W (φ)∥2},
√

hB, 1)
φ = Λ

(T (φ), C(φ), Lip(φ)) = (max{1, ∥W (φ)∥2},
√

h, τe−1/2)
φ = Γ

(T (φ), C(φ), Lip(φ)) = (∥W (φ)∥2,
√

3 max{1, B}, max{1, B})
φ = L

T (ω) = max{1, ∥W (ω)∥2}; and C(ω), Lip(ω) are such that:

max
G∈G
max
p∈Dg(G) |ωw(p)|2 ≤C(ω)T (ω)
and
max
G∈G
max
p∈Dg(G) |ωw+u(p) −ωw(p)| ≤Lip(ω)∥U (ω)∥2

for any w and u.

Proof. First we prove the inequaliry about maximum output norm.

28


---Page Break---
Maximum output norm.

max
G∈G ∥f(Dg(G))∥2 ≤A1 max
G∈G
max
p∈Dg(G) ∥ωw(p)φw(p)∥2 ≤

≤A1 max
G∈G
max
p∈Dg(G) ωw(p)∥φw(p)∥2 ≤

≤A1
max
G∈G,p∈Dg(G) ωw(p)
max
G∈G,p∈Dg G ∥φw(p)∥2,

(60)

where A1 is max
G∈G card(Dg(G)) if Agg is sum and A1 = 1 if Agg is k-max or mean.

From the Lemma statement we can upper bound ωw(p):

max
G∈G,p∈Dg(G) ωw(p) ≤C(ω)T (ω)

Maximum norm of φ:

φ = Λ.

max
G∈G,p∈Dg(G) |φw(p)i| =
max
G∈G,p∈Dg(G) max{0, p2 −|ti −p1|} ≤p2 ≤B

Hence:

max
G∈G,p∈Dg(G) ∥φw(p)∥2 =

" h
X

i=1
φw(p)2
i

#1/2

≤

" h
X

i=1
b2
#1/2

= B
√

h

φ = Γ.

max
G∈G,p∈Dg(G) |φw(p)i| =
max
G∈G,p∈Dg(G) exp

−|p1 −ti,1|2 + |p2 −ti,2|2

2τ 2


≤1

Hence:

max
G∈G,p∈Dg(G) ∥φw(p)∥2 =

" h
X

i=1
|φw(p)i|2
#1/2

≤

" h
X

i=1
1

#1/2

=
√

h

φ = Ψ.

max
G∈G,p∈Dg(G) |φw(p)i| =
max
G∈G,p∈Dg(G) |p1ti[1] + p2ti[2] + ti[3]| ≤

≤max{B, 1}|ti,1 + ti,2 + ti[3]|

Hence:

max
G∈G,p∈Dg(G) ∥φw(p)∥2 =

" h
X

i=1
|(φw)i|2
#1/2

≤

" h
X

i=1
max{B, 1}2|ti,1 + ti,2 + ti[3]|2
#1/2

≤

≤max{b, 1}

" h
X

i=1
3(t2
i,1 + t2
i,2 + ti[3]2)

#

≤

≤
√

3 max{B, 1}∥vec(W (φ))∥2

Combining all together we get the Equation 58.

Maximum perturbation of the PersLay output.

max
G∈G ∥fw+u(Dg(G)) −fw(Dg(G))∥2 ≤A2
max
G∈G,p∈Dg(G) ∥ωw+u(p)φw+u(p) −ωw(p)φw(p)∥2 ≤

≤A2
max
G∈G,p∈Dg(G) [∥ωw+u(p)(φw+u(p) −φw(p)) + φw(p)(ωw+u(p) −ωw(p))∥2] ≤

≤A2
max
G∈G,p∈Dg(G) [|ωw+u(p)|∥φw+u(p) −φw(p)∥2 + ∥φw(p)∥2|ωw+u(p) −ωw(p)|] ,

29


---Page Break---
where A2 is max
G∈G card(Dg(G)) if Agg is sum, A2 is 3 if Agg is k-max by Lemma 9 and A2 is 1 if

Agg is mean.

By the Lemma statement we have:

max
G∈G,p∈Dg(G) ∥ωw+u∥2 ≤Cω(1 + η(ω))T (ω),

and
max
G∈G,p∈Dg(G) |ωw+u(p) −ωw(p)| ≤Lip(ω)∥U (ω)∥2.

Moreover, from Paragraph about max norm of φ:

max
G∈G,p∈Dg(G) ∥φw(p)∥2 ≤C(φ)T (φ)

Maximum perturbation of φ:

φ = Λ. Since g(x) = |x| is 1-Lipschitz we have that:

max
G∈G,p∈Dg(G) ∥(φw+u(p) −φw(p))i∥≤U (φ)
i

Hence,

max
G∈G,p∈Dg(G) ∥φw+u(p) −φw(p)∥2 ≤

" h
X

i=1
|(φw+u(p) −φw(p))i|2
#1/2

≤

≤

" h
X

i=1
(U (φ)
i
)2
#1/2

= ∥U (φ)∥2

φ = Γ. Suppose g(x, y) = exp

−x2+y2

2τ 2

. Then

∥g(x, y) −g(x + ∆x, y + ∆y)∥2 = |∇g(x′, y′)|2
p

∆x2 + ∆y2 ≤

≤max
x′,y′ |∇g(x′, y′)|2
p

∆x2 + ∆y2

by mean-value theorem for some x′, y′ between x and x + ∆x and y and y + ∆y.

Let us find maximum of the gradient by every coordinate. Since the function is symmetric
we need to do it only for one of the coordinates.

The maximum of the norm of the first coordinate of the gradient is achieving at the point
ti,1 = p1 ± τ, and the gradient value at these points is at most
1
τe1/2 . So we have:

max
G∈G,p∈Dg(G) ∥φw+u(p) −φw(p)∥2 ≤

" h
X

i=1

∥U (φ)
i
∥2
2
τ 2e

#1/2

≤∥vec(U (φ))∥2

τe1/2

φ = Ψ. In this case g(x, y, z) = p1x + p2y + z is max{B, 1}-Lipschitz, so

max
G∈G,p∈Dg(G) ∥φw+u(p)−φw(p)∥2 ≤

" h
X

i=1
max{B, 1}2∥U (φ)
i
∥2
2

#1/2

= max{B, 1}∥vec(U (φ))∥2
2

Combining all together we have:

max
G∈G ∥fw+u(G) −fw(G)∥2 ≤A2
h
(1 + η(ω))C(ω)T (ω)Lip(φ)∥U (φ)∥2 + C(φ)T (φ)Lip(ω)∥U (ω)∥2
i
≤

≤A2T (φ)T (ω)(1 + η(φ)) max{Lip(φ)C(ω), Lip(ω)C(φ)}
∥U (φ)∥2

∥T (φ)∥2
+ ∥U (ω)∥2

∥T (ω)∥2



30


---Page Break---
E
Proofs of corollaries

Corollary 3. Let fw : X 7→RK with w = {W1, ..., Wn} be an n-layer MLP with Lipi-Lipschitz
activation functions ψi, for i ∈[n −1]. Let B ∈R be such that ∀x ∈X : ∥x∥2 ≤B. Then fw
satisfy requirements of Lemma 2 with:

T =

n
Y

i=1
∥Wi∥2
and
∀i ∈[n] : Si = ∥Wi∥2, ηi = 1

6n

C1 = B

n−1
Y

i=1
Lipi
and
C2 = eB

n−1
Y

i=1
Lipi

Proof. To test Equation 6 and Equation 7 we use Lemma 7.

To check Equation 8 we apply arithmetic-geometric mean inequality:

n
Y

i=1
∥Wi∥2

 n
X

i=1

1
∥Wi∥2

!

=

n
X

i=1

Y

j̸=i
∥Wj∥2 ≥

≥n




n
Y

i=1

Y

j̸=i
∥Wj∥2



= n

 n
Y

i=1
Ti

! n−1

n
= nT
n−1

n

(61)

It is left to show that Equation 9 holds:

C1
2C2 1

6n
≥3n

e ≥1
(62)

Corollary 4. Let fw : X 7→Rk with w = {W1, ..., Wn} be a n-layer GCN network with readout
layer. Let ψi for i ∈[n −1] be a Lipi-Lipschitz activation function. Let node feature of any graph
be contained in ℓ2-ball of radius B, i.e. ∀G ∈X : ∥zv∥2 ≤B for every node v and ∀G ∈X, G is
simple and has maximum degree at most d −1. Then fw satisfy requirements of Lemma 2 with:

T =

n
Y

i=1
∥Wi∥2
and
∀i ∈n : Si = ∥Wi∥2, ηi = 1

6n

if using perturbation analysis from Liao et al. [33]

C1 = d
n−1

2 B

n−1
Y

i=1
Lipi
and
C2 = eBd
n−1

2

n−1
Y

i=1
Lipi

or if using perturbation analysis from Sun and Lin [48]

C1 = B

n−1
Y

i=1
Lipi
and
C2 = eB

n−1
Y

i=1
Lipi

Proof. To test Equation 6, Equation 7 we use Lemma 8. We can apply Lemma 8 because ¯η =
1
6n ≤1

n.

To test Equation 8 we use arithmetic vs geometric mean inequality:

n
Y

i=1
∥Wi∥2

 n
X

i=1

1
∥Wi∥2

!

=

n
X

i=1

Y

j̸=i
∥Wj∥2 ≥

≥n




n
Y

i=1

Y

j̸=i
∥Wj∥2



= n

 n
Y

i=1
Ti

! n−1

n
= nT
n−1

n

It is left to test Equation 9

31


---Page Break---
C1
2C2 1

6n
≥3n

e ≥1

Corollary 5. Let fw : X 7→Rk with w = {W1, W2, W3} be n-layer MPGNN (n > 2). Let g, ϕ, ρ
be activations functions with Lipschitz constants: Lipg, Lipϕ, Lipρ. We denote LipgLipϕLipρ∥W2∥2
with C. Let node feature of any graph be contained in ℓ2-ball of radius B, i.e. ∀G = (V, E, z) ∈
X, ∀v ∈V : ∥zv∥2 ≤B and ∀G ∈X, G is simple and has maximum degree at most d −1. Then fw
satisfy Lemma 2 requirements with

if dC ̸= 1:

T = ∥W1∥2∥W3∥2
(dC)n−1 −1

dC −1
,

S1 = ∥W1∥, S2 = min{dC, ∥W2∥2}, S3 = ∥W3∥2,

η1 = η2 = η3 = 1

6n
C1 = BCϕ
and
C2 = eBCϕn

if dC = 1:

T = ∥W1∥2∥W3∥2
S1 = ∥W1∥2, S2 = min{1, ∥W2∥2}, S3 = ∥W3∥2

η1 = η2 = η3 =
1
6(n + 1)

C1 = BCϕ(n + 1)
and
C2 = eBCϕ(n + 1)2

Proof. From [33] (Lemma 3.3) we know that:

max
G∈X ∥fw+u(G) −fw(G)∥2 ≤

(
eB(n + 1)2η∥W1∥2∥W3∥2Cϕ,
dC = 1

eBnη∥W1∥2∥W3∥2Cϕ
(dC)n−1−1

dC−1
,
dC ̸= 1,
(63)

where η = max
n
∥U1∥2
∥W1∥2 , ∥U2∥2

∥W2∥2 , ∥U3∥2

∥W3∥2

o
≤
1
n. In our case ¯η ≤
1
6n, so we can apply these
inequalities. Note that in our notation we have n-layer MPGNN and instead of Wl we have W3 (and
instead of Ul we have U3).

And

max
G∈X ∥fw(G)∥2 ≤

(
B(n −1)Cϕ∥W1∥2∥W3∥2,
dC = 1

BCϕ∥W1∥2∥W3∥2
(dCn−1)−1

dC−1
,
dC ̸= 1
(64)

Eq. (76) and Eq. (68) respectively.

Equation 6 follows from Equation 64.

To test Equation 7 we note that:

η ≤∥U1∥2

∥W1∥2
+ ∥U2∥2

∥W2∥2
+ ∥U3∥2

∥W3∥2
≤∥U1∥2

∥W1∥2
+
∥U2∥2
min{dC, ∥W2∥2} + ∥U3∥2

∥W3∥2
(65)

Now Equation 7 follows from Equation 63 and Equation 65.

To test Equation 8 we employ arithmetic vs geometric mean inequality:

T
 1

S1
+ 1

S2
+ 1

S3


≥
3T
(S1S2S3)1/3 ≥






3
∥W1∥2∥W3∥2
(∥W1∥2∥W3∥2)1/3 ≥3T 2/3,
dC = 1

3
∥W1∥2∥W3∥2
(dC)n−1−1

dC−1
(∥W1∥2dC∥W3∥2)1/3
≥3T 2/3,
dC ̸= 1
,

where in case dC = 1, S2 ≤1 and in case dC ̸= 1, S2 ≤dC and (dC)n−1−1

dC−1
≥dC for n > 2.

32


---Page Break---
To test Equation 9 we provide the following derivation:

C1
2C2
≥

(
l+1
2e(n+1)2 ≥
1
6(n+1),
dC = 1
1
en ≥
1
6n,
dC ̸= 1

Corollary 1. Let fw : G 7→Rk with w = {W (ω), W (φ)} be a PersLay where W (ω) is a parameter
vector (matrix) of weight function, ω, and W (φ) is a parameter vector (matrix) of point-transformation
function, φ. Let Dg be a mapping from graphs to (extended) persistence diagrams with a fixed
filtration function and B such that max
G∈G
max
p∈Dg(G) ∥p∥2 ≤B, then fw satisfy requirements of Lemma 2

with:

T = T (φ)T (ω)
and
(S(φ), S(ω)) = (∥W (φ)∥2, ∥W (ω)∥2)
and
(η(φ), η(ω)) = (1, 1)

C1 = 2 max
n
A1C(ω)C(φ), 2A2 max{C(ω)Lip(φ), C(φ)Lip(ω)}
o

C2 = 2A2 max{C(ω)Lip(φ), C(φ)Lip(ω)},

where A1 = max
G∈G card(Dg(G)) if Agg is sum and A1 = 1 if Agg is k-max or mean; A2 =

max
G∈G card(Dg(G)) if Agg is sum or A2 = 3 if Agg is k-max or mean;

(T (φ), C(φ), Lip(φ)) = (max{1, ∥W (φ)∥2}, B
√

h, 1)
φ = Λ

(T (φ), C(φ), Lip(φ)) = (max{1, ∥W (φ)∥2},
√

h, τe−1/2)
φ = Γ

(T (φ), C(φ), Lip(φ)) = (∥W (φ)∥2,
√

3 max{1, B}, max{1, B})
φ = Ψ

T (ω) = max{1, ∥W (ω)∥2}; and C(ω), Lip(ω) are such that:

max
G∈G
max
p∈Dg(G) |ωw(p)|2 ≤C(ω)T (ω)
and
max
G∈G
max
p∈Dg(G) |ωw+u(p) −ωw(p)| ≤Lip(ω)∥U (ω)∥2

for any w and u.

Proof. To test Equation 6 and Equation 7 we apply Lemma 10. η(ω) in Lemma 10 is arbitrary, so we
can use this lemma with η(ω) = 1 and in this case 1 + η(ω) = 2, so our choice of C2 works.

To test Equation 8 we use arithmetic vs geometric mean inequality:

T (φ)T (ω)

1
S(φ) +
1
S(ω)


≥2
T (φ)T (ω)

(∥W (ω)∥2∥W (φ)∥2)1/3 ≥2
T (ω)T (φ)

(T (ω)T (φ))1/3 = 2T 2/3

It is left to test Equation 9

C1
2C2
≥
2A1C(ω)C(φ) max
n
1, 2A2 max{C(ω)Lip(φ),C(φ)Lip(ω)}

A1C(ω)C(φ)
o

2A2 max{C(ω)Lip(φ), C(φ)Lip(ω)
≥1

F
Comparing bounds with prior works

MLP & GCN
The result is not as tight as in [40] because of the homogenity assumption that they
do. Specificaly, matching our result to theirs, we have B2(h ln nh)¯η−2 the same as B2n2(h ln nh)
(instead of n they have d in their notation). We have log
  m

δ max{1, 1

B }

which is in most cases
better than log m·n

δ . However, the suboptimality of our bound comes from:

n
X

i=1

∥Wi∥2
F
∥Wi∥2
2
≤

n
X

i=1
∥Wi∥2
F

n
X

i=1

1
∥Wi∥2
2
≤max

(

1,

n
X

i=1
∥Wi∥2
F

)  n
X

i=1

1
∥Wi∥2

!2

33


---Page Break---
by Cauchy-Schwarz inequality. The most left expression is the term that is in Neyshabur et al. [40]
and the most right expression is the one we have in our bound.

However, the advantage of our result is that it does not depend on the assumption of all activation
functions being equal to ReLU and that we can compose it with results for other networks to get
bounds for compositions of networks.

MPGNNs
As we discussed in the main text the difference between our bound and the one in Liao
et al. [33] comes from the dependency on the weights.

Their dependency: max
n
ζ−(l+1), (λξ)
l+1

l
o
, where ζ = min{∥W1∥2, ∥W2∥2, ∥Wl∥2}, λ =

∥W1∥2∥Wl∥2, ξ = Cϕ
(dC)l−1−1

dC−1
. Let us show, how our bound depends on λ, ζ and ξ:

ζ−1 = max

1
∥W1∥2
,
1
∥W2∥2
,
1
∥Wl∥2


≥1

3


1
∥W1∥2
+
1
∥W2∥2
+
1
∥Wl∥2


≥

≥min{1, dLipϕLipgLipρ}

3


1
∥W1∥2
+
1
min{dC, ∥W2∥2} +
1
∥Wl∥2



and

ζ−1 = max

1
∥W1∥2
,
1
∥W2∥2
,
1
∥Wl∥2


≤
1
∥W1∥2
+
1
∥W2∥2
+
1
∥Wl∥2
≤

≤
1
∥W1∥2
+
1
min{dC, ∥W2∥2} +
1
∥Wl∥2
.

Note that λξ = T. Now let us analyze two cases:

1. (λξ)
l+1

l
≥
(ζ)−(l+1).
In this case ζ−1
≤
(λξ)1/l and ζ−1λξ
≤
(λξ)
l+1

l
=
max{ζ−(l+1), (λξ)
l+1

l }

2. (ζ)−(l+1)
≥
(λξ)
l+1

l .
In this case ζ−l
≥
λξ and ζ−1λξ
≤
ζ−(l+1)
=
max{ζ−(l+1), (λξ)
l+1

l }

So, in the case when dLipϕLipgLipρ is lower bounded by some constant we can conclude that
asymptotically our bound is not inferior, than the one in Liao et al. [33].

G
Dependency on Model Parameters and Hyperparameters

In the Table 6 we present the dependency of existing results and our result on the model parameters
and hyperparameters.

H
Learnable Filtration Functions

Fixed filtration functions dominate the PH/ML literature. The widespread use of learnable functions
is a relatively recent phenomenon in PH-based ML, and usually runs orders of magnitude slower
compared to non-learnable ones. Arguably, applying non-learnable functions still represents the
mainstream approach in TDA.

Some works have explicitly advocated for fixed filtration functions (with learnable vectorizations)
over learnable filtrations. Filtration functions can come in different flavors; for instance, they can
rely on node degree [18], cliques [43], or node attributes [22]. Some of the popular options are
parameter-free. Also, while some works showed gains using learnable filtrations [20], others have
reported no benefits and adopted fixed functions instead [5, 34]. There is still no consensus about the
significance of the gains associated with learnable filtration in many applications.

Perslay [5] uses fixed filtration functions. Despite the generality of our results, we provide specific
bounds for PersLay, which employs fixed filtration functions.

Our work lays a strong foundation for analyzing learnable filtrations. One way to analyze PH with
learnable filtration schemes could be to get upper bounds on perturbation of outputs in terms of the

34


---Page Break---
Table 6: The dependency of the bound on the model parameters. All models have maximum
width of weight matrices h.
We consider n-layer multilayer perceptron (MLP) with weights
W1, ..., Wn. We consider n-layer GCN with weights W1, ..., Wn. We consider n-layer (n > 2)
MPGNN with weights W1, W2, W3. We consider simple graphs with maximum degree d. We denote
C = LipϕLipgLipρ∥W2∥, λ = ∥W1∥2∥W3∥2 and ξ = ((dC)n−1−1)/(dC−1) and |w|2 is the norm of all
parameters in the model. Comparing to [33] we do not add Lipϕ to ξ and instead of Wl we have W3.
We consider PersLay with one of the classical point-transformation (described in [5]) and weight
function with weights W (φ), W (ω) and ∥w∥2 is the norm of all parameters in the model. We consider
abstract model that satisfy Lemma 2 requirements with w, T, ¯η, Cnorm and Si,

Name (Reference)
Model Parameters
Model Hyperparameters

MLP, [40]

nQ

i=1
∥Wi∥2

nP

i=1

∥Wi∥F/∥Wi∥2
n
p

h ln(nh)

GCN, [33]

nQ

i=1
∥Wi∥2

nP

i=1

∥Wi∥F/∥Wi∥2
n
p

dn−1h ln(nh)

MPGNN, [33]
∥w∥2 max{ζ−(n+1), (λξ)
(n+1)/n}
n
p

h ln(nh)

PersLay
∥w∥2∥W (φ)∥2∥W (ω)∥2

√

h
3/2 ln h

Lemma 2
∥w∥2T
nP

i=1

1/Si
Cnorm¯η−1p

h ln (nh)

filtration function parameters. This would additionally require an analysis of Wasserstein distances
between persistence diagrams obtained with different parameters. We believe that for a specific
class of graphs we can get modified upper bounds for perturbation with respect to filtration function
parameters that would depend on Wasserstein distance of the same order. This additional analysis
could be readily integrated into our framework to get generalization bounds for learnable filtrations.

I
Discussion on the max over 1 and parameters norm.

From the first glance it may seem that introducing max{1, ∥w∥2
2} is suboptimal; however, ∥w∥2
2 is
greater than 1 in many real-world scenarios. For instance, suppose we have parameters within the
range of O(ε); then the squared norm would be greater than 1 if the number of parameters exceeds 1

ε2 .
Considering a typical choice in the Deep Learning literature, ε = 0.01, we need more than 10, 000
parameters. For example, a two-layer MLP with a hidden dimension of 64 (also a typical choice) has
more than 8, 000 parameters, so in practice, ∥w∥2
2 ≥1 is usually true.

J
Implementation details

J.1
Datasets

Table 7 reports summary statistics of the datasets used in this paper.

Table 7: Statistics of the datasets.

Dataset
#graphs
#classes
Avg #nodes
Avg #edges

NCI1
4110
2
29.87
32.30
IMDB-B
1000
2
19.77
96.53
PROTEINS (full)
1113
2
39.06
72.82
MUTAG
188
2
17.93
19.79
DHRF
756
2
42.43
44.54
MOLHIV
41127
2
25.5
27.5
NCI109
4127
2
29.68
32.13

35


---Page Break---
J.2
Models

For the experiments with PersLay Classifier, we closely follow the filtration functions used in [5].
In particular, we use Kernel heat functions with parameters t = 0.1 and t = 10 for the remaining
datasets. Instead of processing each diagram type using separate models, we combine ordinary
and extended diagrams for 0- and 1-dimensional features and apply a single model. We use mean
aggregation function in all experiments, and Gaussian point transformations. For the feedforward
part of PersLay, we apply ReLU activation functions. All models are trained with Adam [27] and
learning rate of 10−3 for 3000 epochs.

For the experiments with GNNs with persistence, we use graph isomorphism networks (GINs) with 2
layers and 64 hidden units. The models were trained for 2000 epochs using the Adam optimizer.

Dependence on model paramaters. Regarding the dependence on the spectral norm of weights, we
reported results for a model with a final MLP (multilayer perceptron) of 2 hidden layers (3 layers
in total) and width of 128 for all layers. The number of parameters of the point transformation was
h = 100. For the experiments on width vs. generalization gap, we used h = 100, and 1 hidden layer
with a varying width in {32, 64, 128, 256, 512}.

Regularizing PersLay. For the experiments regarding ERM and spectral norm regularizers, we
perform model selection for (number of layers) l ∈{2, 3}, and αr ∈{10−3, 10−4, 10−5, 10−6}.
Again, we use Gaussian point transformation, h = 100, and width equals to 128. Our goal was to see
if we could observe gains from the regularized version even for shallow neural networks.

Regularizing GNNs with persistence. Here, we consider GNNs with 64 hidden units of 64
(width) and 2 layers. We set the dimensionality of PersLay parameters equal to 100, Gaussian point
transformation, and mean aggregation function. We apply hold-out model selection with penalty term
αr ∈{10−5, 10−6, 10−7, 10−8} using the validation set.

Hardware. For all experiments, we use Tesla V100 GPU cards and consider a memory budget of
32GB of RAM.

26
28

Width, h

−0.5

0.0

0.5

1.0

Generalization gap

ρ: 0.91±0.04

PROTEINS

Empirical gap
Our bound

26
28

Width, h

0.0

0.5

1.0
ρ: 0.93±0.04

NCI1

26
28

Width, h

0.0

0.5

1.0
ρ: 0.72±0.06

IMDB-BINARY

Figure 7: Width vs. generalization gap for the triangle point transformation.

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

1.5

Generalization

ρ: 0.72±0.07

PROTEINS

Generalization gap
Spectral norm

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

1.5

ρ: 0.84±0.07

NCI1

1
1000
2000
3000
Epoch

−0.5

0.0

0.5

1.0

ρ: 0.69±0.04

IMDB-BINARY

Figure 8: Spectral norm vs. generalization gap for the triangle point transformation.

K
Additional visualizations

Figure 7 and Figure 8 report additional results for the triangle point transformation on the three largest
datasets: PROTEINS, NCI1, and IMDB-BINARY. In particular, Figure 7 shows the dependence of
the generalization on width, while Figure 8 shows the dependence on the spectral norm. Overall, our
bound can capture the trend in the empirical gap and produces high correlation values for all datasets.

36


---Page Break---
NeurIPS Paper Checklist

• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Table 1 maps the results supporting our claims.
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
Justification: The discussion about the Table 2 in the Section 3 contains comparison with
prior works. The discussion after Corollary 1 contains description of the limitation of our
PersLay analysis. Moreover, the conclusion summarizes some of the important limitations.
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

37


---Page Break---
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: Assumptions are clearly stated in the statements, and in the Appendix E and
Appendix C.
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
Justification: Details are provided in the Experiment Section as well as in the Appendix J.
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

38


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We provide a link to our code in Section 6.

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

Justification: Details are provided in the Experiment Section as well as in the Appendix J

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

Justification: Plots count on error bars and tables count on standard deviation.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

39


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

Justification: Appendix J.

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

Justification: Our submission follow the NeurIPS ethical guidelines.

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

Justification: We provide a perspective on broader impacts in Section 7, but do not foresee
any direct negative societal impact.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

40


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
Justification: We do not foresee any direct risk stemming from our work.
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
Justification: All code was made by the authors
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

41


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: No new assets.
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
Justification: No experiments with human subjects.
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
Justification: No experiments with human subjects.
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

42


---Page Break---
