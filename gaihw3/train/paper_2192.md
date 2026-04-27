Differentiable Structure Learning with Partial Orders

Taiyu Ban
Lyuzhou Chen
Xiangyu Wang∗
Xin Wang
Derui Lyu
Huanhuan Chen∗

University of Science and Technology of China
{banty, clz31415, wz520, drlv}@mail.ustc.edu.cn
{sa312, hchen}@ustc.edu.cn

Abstract

Differentiable structure learning is a novel line of causal discovery research that
transforms the combinatorial optimization of structural models into a continuous
optimization problem. However, the ﬁeld has lacked feasible methods to integrate
partial order constraints, a critical prior information typically used in real-world
scenarios, into the differentiable structure learning framework. The main difﬁculty
lies in adapting these constraints, typically suited for the space of total orderings,
to the continuous optimization context of structure learning in the graph space.
To bridge this gap, this paper formalizes a set of equivalent constraints that map
partial orders onto graph spaces and introduces a plug-and-play module for their
efﬁcient application. This module preserves the equivalent effect of partial order
constraints in the graph space, backed by theoretical validations of correctness
and completeness. It signiﬁcantly enhances the quality of recovered structures
while maintaining good efﬁciency, which learns better structures using 90% fewer
samples than the data-based method on a real-world dataset. This result, together
with a comprehensive evaluation on synthetic cases, demonstrates our method’s
ability to effectively improve differentiable structure learning with partial orders.

1
Introduction

Learning directed acyclic graph (DAG) structures from observational data is fundamental for causal
discovery in scientiﬁc research [Opgen-Rhein and Strimmer, 2007; Pearl and others, 2000]. Tra-
ditionally, it has been approached as a combinatorial optimization problem dominated by indepen-
dence tests and score-and-search methods [Heinze-Deml et al., 2018]. Zheng et al. [2018] reformed
it as a continuous optimization problem through a novel characterization of the acyclicity constraint
in a differentiable form. Subsequently, numerous studies have proposed various learner architec-
tures [Yu et al., 2019; Zhu et al., 2019; Zheng et al., 2020], acyclicity characterizations [Yu et al.,
2019; Ng et al., 2022; Bello et al., 2022], and optimization techniques [Wei et al., 2020; Deng et al.,
2023a,b] to advance differentiable structure learning.

In practical scenarios of causal discovery, researchers often possess prior knowledge of ordering,
such as known gene activation sequences in genetics [Olson, 2006], standard treatment sequences
in healthcare management [Denton et al., 2007], and sequential seasonal weather patterns in me-
teorology [Bruffaerts et al., 2018]. Such prior information can be generally formalized as a set
O = {(x, y) | x, y ∈X} of partial orders, where the binary relation (x, y) represents that variable
x precedes y in the ordering, denoted as x ≺y. For traditional score-and-search structure learning
methods, the constraints of partial orders can reduce the space of total orderings in which the search
algorithms2 are usually performed [Teyssier and Koller, 2005]. Thus, the prior partial order informs
the search process to ﬁnd more genuine structures, which is essential for practical causal discovery.

∗Corresponding authors.
2Such algorithms are known as order-based search in the literature, see Appendix A for details.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
However, the application of prior partial orders in the context of differentiable structure learning has
not been explored. The main challenge arises from the inapplicability of the scenario, as the contin-
uous optimization of structures is conducted in the graph space, while partial orders are constraints
in the ordering space, leading to a misalignment of the hypothesis space. In related studies, Deng et
al. [2023a] use a search strategy to solve the constrained optimization problem in the ordering space,
which is not a purely continuous method. Some research applies differentiable structure learning to
dynamic Bayesian network (DBN) structure learning [Pamﬁl et al., 2020; Sun et al., 2023; Yang et
al., 2022], which assumes a strict time-series ordering. This strict order prior can be simply imple-
mented in the graph space by freezing the parameters of certain edges (see discussions in Appendix
B), which signiﬁcantly differs from the general partial orders discussed in this paper.

Despite of the misalignment of hypothesis spaces, the partial order constraint in the ordering space
has an equivalent form in the graph space captured by path prohibitions [Grimmett and Stirzaker,
2015]. This leads to a feasible way to apply partial orders in differentiable structure learning. How-
ever, for a sequential ordering with m variables, there are
(m
2
)
paths to be forbidden in the equivalent
constraint. This complexity makes it impractical to develop constraints on path prohibitions individ-
ually, leading to substantial computational overhead for long sequential orderings.

To address this issue, we propose an efﬁcient approach that augments the acyclicity constraint to
naturally forbid all paths in the equivalent constraint of partial orders O. Concretely, we infer the
transitive reduction of O and divide it into maximal paths to capture all possible sequential orderings.
These paths are then individually added to the adjacency matrix of the acyclicity term, forming an
augmented acyclicity constraint. We prove that adherence to this new constraint is equivalent to
adherence to partial order constraints. Furthermore, this method efﬁciently handles long sequential
orderings, requiring only one factor to describe a sequential ordering regardless of its length. It
is a plug-and-play module that can be easily adapted to various algorithms. Evaluations on both
synthetic and real-world data verify its effectiveness. Contributions are listed:

• To the best of our knowledge, this is the ﬁrst work to discuss the integration of prior con-
straints of partial orders into the continuous optimization of structure learning. We propose
a plug-and-play module enabling the integration of this prior, which is theoretically appli-
cable to all continuous methods in the differentiable structure learning context.
• We address the misalignment between the hypothesis space of differentiable structure learn-
ing and partial order constraints by converting them into an equivalent form of path prohi-
bition constraints. By formalizing the continuous characterization of this equivalent con-
straint, we show its limited practicality in dealing with long sequential orderings.
• To efﬁciently integrate long sequential orderings, we introduce a novel approach to apply
path prohibitions by augmenting the acyclicity constraint with partial orders. We prove the
equivalence of this augmented acyclicity constraint to the adherence to partial orders and
show its efﬁciency in dealing with long sequential orderings.

2
Notations and Preliminaries

Notations
In the following illustrations, we denote Wi,:, W:,j, and Wi,j to represent the ith row,
jth column, and (i, j)th element of a matrix W, respectively. If a matrix symbol includes a subscript,
like Ws, we represent its elements as Ws,i,j. For operations resulting in a matrix, such as W1 + W2,
we denote elements of the resulting matrix as (W1 +W2)i,j. For simplicity, the symbol (i, j) is used
contextually: it can refer to a partial order Xi ≺Xj or to a directed edge (Xi, Xj) in the graph.
Related work and proof of statements can be found in Appendix A and Appendix C, respectively.

Structural equation model
Let G denote a directed acyclic graph (DAG) with d nodes, where
the vertex set V corresponds to a set of random variables X = {X1, X2, . . . , Xd}, and the edge set
E(G) ⊂V × V deﬁnes the causal relationships among the variables. The structural equation model
(SEM) speciﬁes that the value of each variable is determined by a function of its parent variables in
G and an independent noise component:

Xj = fj(PaG
j , zj)
(1)

where PaG
j = {Xi | Xi ∈X, (Xi, Xj) ∈E} denotes the set of parent variables of Xj in G, and
zj represents noise that is independent across different j. Denoting the structure of G as a weighted

2


---Page Break---
adjacent matrix W ∈Rd×d, where Wi,j ̸= 0 equals that (Xi, Xj) ∈E(G), we have:

Xj = fj(W:,j, X, zj)
(2)

Given a set of samples D ∈Rm×d generated from this model with either linear or nonlinear func-
tions in {fj}, we next describe the process of learning the structure of G in a differentiable manner.

Differentiable structure learning
The objective of structure learning is to deduce the DAG struc-
ture represented by the weighted adjacency matrix W ∈Rd×d from the data D generated by a
speciﬁc set of functions f. We deﬁne all parameters characterizing W and f as θ and the graph as
G(W(θ)), and we formalize the optimization problem of structure learning as follows:

min
θ
F(D, fθ(D))
subject to G(W(θ)) ∈DAG
(3)

where F is the score function, such as the least squares F(D, fθ(D)) =
1
2m
∑m
i=1 ∥D −fθ(D)∥2
F
[Loh and Bühlmann, 2014]. For clarity, we emphasize the target W and rewrite it as F(W). Simi-
larly, the symbol G(W(θ)) denoting the graph is simpliﬁed as G(W). The acyclicity degree of the
graph can be characterized by a series of continuous functions for a non-negative matrix B:

h(B) = Trace

( d
∑

i=1
ciBi
)

,
ci > 0
(4)

For this acyclicity characterization, Zheng et al. [2018] use h(B) = Trace(eB)−d, derived from an
inﬁnite power series3 of B. Yu et al. [2019] suggest a polynomial form h(B) = Trace((I + 1

dB)d −
I), and Bello et al. [2022] employ a log-determinant function h(B) = −log det(sI −B) + d log s.
For W ∈Rd×d, it is common to use B = W ◦W to ensure the non-negativity of B. Hence, the
following mentioned h(W) actually refers to h(W ◦W).
Proposition 1. (Theorem 1 in [Wei et al., 2020]). The directed graph of a non-negative adjacency
matrix B is a DAG if and only if h(B) = 0 for any h deﬁned by (4).

According to this result, the constraint G(W) ∈DAG can be implemented by the continuous equal-
ity h(W) = 0. This transformation converts the structure learning problem into a continuous opti-
mization problem with an equality constraint, formulated as:

min
W ∈Rd×d F(W)
subject to h(W) = 0
(5)

Zheng et al. [2018] apply the augmented Lagrangian method to solve this problem, a technique
widely adopted in subsequent studies. Note that our proposed plug-and-play module does not alter
the optimization process; therefore, this aspect is out of scope and not discussed in this paper.

Role of orders in structure learning
Variable ordering plays a crucial role in the combinatorial
optimization of structure learning. The order-based score-and-search method is a critical research
direction in this context. It is founded on the principle that structure learning is no longer NP-hard
when the total ordering of variables is known [Teyssier and Koller, 2005]. These methods search
within the hypothesis space of total orderings to identify the optimal solution within this constraint
[Xiang and Kim, 2013; Raskutti and Uhler, 2018; Squires et al., 2020; Wang et al., 2021; Solus et
al., 2021; Chen et al., 2019; Li and Beek, 2018; Chen et al., 2016]. They beneﬁt from prior partial
orders, which help reduce the space of possible total orderings. However, differentiable structure
learning solves Equation (5) in the DAG space, making it inapplicable to directly use partial orders.
Therefore, we employ an alternative constraint equivalent to partial orders in the following section.

3
Differentiable Structure Learning with Partial Orders

This section introduces the integration of partial order constraints into the continuous DAG opti-
mization framework. First, we convert the constraints from the ordering space into an equivalent
form in the DAG space. Next, we discuss the limitations of direct characterizations of these equiv-
alent constraints. Finally, we present an efﬁcient approach to integrate partial order constraints and
illustrate its theoretical correctness and completeness.

3The terms with powers higher than d are expressible in ﬁnite terms with powers not exceeding d by the
Cayley-Hamilton theorem. Therefore, Trace(eB) −d is also an instance of Equation (4).

3


---Page Break---
3.1
Capture partial orders with path prohibition constraints

To begin with, we formally deﬁne critical concepts related to partial orders.

Deﬁnition 1 (Partial Order). For a set S of variables, a partial order is a binary relation ≺on S
which is a subset of S × S. For all elements x, y, and z in S, the following properties are satisﬁed:

Reﬂexivity: x ≺x for every x in S; Antisymmetry: If x ≺y and y ≺x, then x = y; Transitivity: If
x ≺y and y ≺z, then x ≺z.

For the structure, if x ≺y, then y cannot be the ancestor of x; that is, no directed path exists from y
to x in the graph. Note that while the partial order relation is transitive, the absence of paths is not.
This requires further consideration of the transitive property of orders.

Deﬁnition 2 (Transitive closure). For a set S and a binary relation R ⊆S×S, the transitive closure
of R, denoted by R+, is deﬁned as R+ = ∪∞
n=1 Rn. Rn is deﬁned recursively by: R1 = R,
Rn+1 = R ◦Rn, R ◦T = {(x, z) ∈S × S | ∃y ∈S such that (x, y) ∈S and (y, z) ∈T }.

Remark 1. The transitive closure O+ of a set of partial orders O encompasses all orders either
directly contained in or inferable through transitivity from O.

Now, we consider the following result from graph theory, which is essential for transforming order
constraints into structural constraints.

Proposition 2. There exists at least one topological sort of DAG G that satisﬁes the partial order
set O if and only if, for any order (i, j) in O+, Xj is not an ancestor of Xi in G.

With this statement, the structure learning problem with partial orders O can be implemented by its
equivalent constraint set of path prohibitions, formalized as:

min
W ∈Rd×d F(W)
subject to h(W) = 0, Xj ⇝Xi /∈G(W) for all (i, j) ∈O+
(6)

where Xj ⇝Xi /∈G(W) indicates that no directed path exists from Xj to Xi in G(W). Subse-
quently, we introduce this constraint’s continuous characterization and discuss its limitations.

3.2
Continuous characterization of path prohibitions

This section introduces the continuous characterization of the path prohibition constraint in Equation
(6) and the practical difﬁculties in optimizing it. To clarify the unique challenges when dealing with
ﬂexible partial orders, we begin with the case of total orderings.

Deﬁnition 3 (Total Ordering). A total ordering is a permutation π of all the variables, with π(i)
denoting the index of the variable in the ith position. Xπ(i) precedes Xπ(j) if and only if i < j.

For total ordering, the order relationship between any pair of variables is contained in the transi-
tive closure π+. This property allows for a simple implementation of the constraint of π by edge
prohibition, as illustrated below:

Proposition 3. A graph G is a DAG and satisﬁes total ordering π if and only if edge (u, v) does not
exist in G for all (v, u) ∈π+.

This edge absence constraint on G(W) can be directly implemented by setting the corresponding
parameters in W to zero, resulting in the following formulation:

min
W F(W)
subject to Wπ(i),π(j) = 0 for all i ≥j
(7)

In this case, structure learning becomes an unconstrained optimization problem, as adherence to
total orderings naturally satisﬁes the DAG constraint. This problem can be solved more efﬁciently
than the original problem with the constraint equality h(W) = 0.

Now we consider the ﬂexible partial order constraints. Let O represent a set of partial orders that
do not inherently contain cycles within their transitive closure O+. Merely forbidding edges that
violate O+ is insufﬁcient for compliance, as it is possible to walk from a variable to a preceding
variable in O through another variable whose order with others is not contained in O, such as:

4


---Page Break---
Example 1. For example, consider four nodes 1, 2, 3, 4 with a partial order set O = {(1, 2), (2, 3)}.
We forbid all inverse edges in O+, which are (2, 1), (3, 2), (3, 1). Despite this, directed paths violat-
ing the partial order (1, 2) can still exist, such as the path (2, 4, 1). Such paths can be constructed
by traversing nodes not in O, like node 4 in this case.

Consequently, we must consider the constraint of path prohibitions equivalent to O. According to
the proof to Proposition 1, the following equality can be used for path prohibition constraints.

Proposition 4. No directed path Xi ⇝Xj exists in G(W) if and only if
(∑d
l=1(W ◦W)l)

i,j = 0.

With this statement, we can formalize the optimization problem in Equation (6) as follows:

min
W F(W)
subject to h(W) = 0, p(W, O) = 0
(8a)

p(W, O) =
∑

(i,j)∈O+

( d
∑

l=1
(W ◦W)l
)

j,i

(8b)

Remark 2. A signiﬁcant difﬁculty of the optimization problem formulated in Equation (8a) is its
steep decline in training efﬁciency as the complexity of partial orders increases. The penalty term
p(W, O), as deﬁned by Equation (8b), includes a term for each order in O+, directly impacting
the computational cost for gradient calculations. When dealing with a sequential ordering with m
variables, it introduces
(m
2
)
new terms. Each of these terms demands comparable time for gradient
calculation to the acyclicity term h(W) typically used in current studies. This makes the computa-
tional load impractical for long sequential orderings. Note that the total ordering constraint results
in the most constraint terms in this case, while it can be efﬁciently addressed by Equation (7).

This observation underpins the need to develop a more efﬁcient method to ensure that the structure
learning process remains computationally feasible for long sequential orderings.

3.3
Augmented acyclicity-based partial order characterization

This section introduces an efﬁcient method to characterize partial orders, distinct from directly rep-
resenting the equivalent path prohibitions. We ﬁrst introduce some critical concepts.
Deﬁnition 4 (Transitive Reduction). The transitive reduction O−of a relation O is the smallest
relation such that the transitive closure of O−is equal to the transitive closure of O. Formally,
(O−)+ = O+ and O−is minimal.

The transitive reduction is used to eliminate redundant orders to facilitate calculation efﬁciency.
Below, we provide an example to illustrate transitive reduction alongside transitive closure.
Example 2. For a set of transitive binary relation O = {(1, 2), (2, 3), (1, 3), (3, 4)}, its transitive
closure is O+ = O ∪{(1, 4), (2, 4)}, and its transitive reduction is O−= O \ {(1, 3)}.
Deﬁnition 5. Let G = (V, E) be a graph. A source is a vertex in V with no incoming edges, i.e.,
{v ∈V : deg−(v) = 0}. A sink is a vertex with no outgoing edges, i.e., {v ∈V : deg+(v) = 0}.
Deﬁnition 6 (Maximal Path). Let G = (V, E) be a graph with a node set V and edge set E. A path
p = (v1, . . . , vk) with (vi, vi+1) ∈E is considered a maximal path if v1 is a source, vk is a sink,
and the path is not a proper subsequence of any other path from v1 to vk.
Deﬁnition 7. The transitive closure of a path p = (v1, . . . , vk), denoted as p+, is the set of all
ordered pairs (vi, vj) for 1 ≤i < j ≤k.
Remark 3. For brevity, the following discussions regard the concepts of the directed graph, path,
sequential ordering, and partial order set as an identical type of set whose element is an ordered pair
(i, j), as both node reachability in a graph and the order relationship are transitive. Especially, we
do not distinguish between a partial order set O and the graph G(O) constructed by E(G(O)) =
{(i, j) | (i, j) ∈O}.
Remark 4. We assume that no cycle exists in O+. That is, O is not conﬂicting with itself.

With these deﬁnitions, we formalize the new approach to integrating partial order constraint O into
differentiable structure learning as follows (see Appendix D for detailed implementations):

5


---Page Break---
min
W F(W)
subject to h′(W, O) = 0
(9a)

h′(W, O) =
∑

o∈P(O−)
h(A(W, o))
(9b)

A(W, o) = W + τWo −W ◦Wo
(9c)
Wo,i,j = [(i, j) ∈o]
(9d)

Here, O−is the transitive reduction of O. P(O−) represents the set of all maximal paths of O−).
[P] is the indicator function valuing 1 if condition P holds and 0 otherwise. τ > 0 is a hyper-
parameter used for adjusting the weight in gradient calculation.
Remark 5. Recall that h(W) ≥0 by Equation (4). Then we have that h′(W, O) = 0 is equivalent
to h(A(W, o)) = 0 for o ∈P(O−) by Equation (9b).

Equation (9) can be interpreted as augmenting the original acyclicity constraint h(W) = 0 to a
stronger one h′(W, O) = 0. Speciﬁcally, we use a series of partial order-augmented acyclicity con-
straints h(A(W, o)) = 0 for o in the maximal path set of O−as described in Equation (9b). For
each augmented acyclicity, we add the path o to the adjacency matrix W by A(W, o) as detailed in
Equation (9c). Thus, the acyclicity function h with A(W, o) as input represents a stronger acyclic-
ity constraint. The additional part of this stronger acyclicity accurately captures adherence to the
sequential ordering indicated by o, which can be derived from the following statement.
Lemma 1. A graph G is a DAG and satisﬁes a sequential ordering o = {(p1, p2, · · · , pm)} if and
only if graph G′ is a DAG where E(G′) = E(G) ∪o.

This lemma states the equivalence of h(A(W, o)) = 0 to adherence to the sequential ordering o.
Now consider the following statement.
Lemma 2. For the set P(O−) of all maximal paths of O−, the union of the transitive closures of
these paths is the transitive closure of O: ∪

o∈P(O−) o+ = O+

This lemma states that adherence to all the sequential orderings o indicated by maximal paths in
O−is equivalent to adherence to the complete set O of partial orders. Recall that h′(W, O) = 0 is
equivalent to h(A(W, o)) = 0 for o in P(O−), and h(A(W, o)) = 0 is equivalent to adherence to o.
Hence, we derive that h′(W, O) = 0 is equivalent to adherence to O by Lemma 2, as described in
the following statement (the proof of these statements is provided in Appendix C.1).
Theorem 1. A graph G is a DAG and satisﬁes a set of partial orders O if and only if h′(W, O) = 0
for the function h deﬁned by Equation (4) and h′ deﬁned by Equations (9b), (9c), and (9d).

Theorem 1 shows the correctness and completeness of the equality h′(W, O) = 0 in capturing the
partial order constraint O. More concretely, all prior information of O is fully integrated while no
extra information beyond O is introduced. This is attributed to two critical steps in integrating O
into the acyclicity constraint. Step 1. Split the partial order constraint O into sequential orderings.
Step 2. Ensure that these sequential orderings are maximal paths in O−. If Step 1 is removed4,
and we directly add all the edges in O−to G(W) for augmented acyclicity, h′ will degenerate into
h(A(W, O−)). This introduces extra orders outside of O, as indicated in the following example.
Example 3. Assume that h′(W, O) ≡h(A(W, O−)).
Consider a partial order set O =
{(1, 2), (3, 4)} and a DAG G(W) with edges E(G(W)) = {(2, 3), (4, 1)}. Obviously, G(W) satis-
ﬁes O. Consider the graph constructed by adding edges in O−(where O−= O) to G(W), i.e., the
graph of the matrix A(W, O−). Its edge set is O ∪E(G(W)) = {(1, 2), (2, 3), (3, 4), (4, 1)} and
contains a cycle (1, 2, 3, 4, 1). This makes h′(W, O) ≡h(A(W, O)) ̸= 0 by Proposition 1.

In this case, a legal DAG that satisﬁes O is forbidden by the constraint h′(W, O) = 0, indicating that
extra constraints beyond the prior are introduced. In other words, Step 1 guarantees the necessity of
the constraint equality for partial order O. For Step 2, it guarantees the sufﬁcient adherence to O. If
this step is removed, some order constraints of O can be lost. See the following case.

4Note that Step 2 would also be omitted in the absence of Step 1.

6


---Page Break---
Example 4. Consider a partial order constraint set O = {(1, 2), (2, 3), (4, 2)}. Suppose that it is di-
vided into two sequential orderings o1 = (1, 2, 3) and o2 = (4, 2), where o2 is not the maximal path
of O−. Then consider graph G(W) with edges {(1, 2), (3, 4)}. We have that E(G(A(W, o1))) =
{(1, 2), (2, 3), (3, 4)} and E(G(A(W, o2))) = {(1, 2), (3, 4), (4, 2)}. Both graphs are DAGs satis-
fying h(A) = 0 by Proposition 1. Then we have h′(W, O) ≡h(A(W, o1)) + h(A(W, o2)) = 0.
Even if G(W) satisﬁes this constraint equality, the edge (3, 4) in G(W) still violates the order
(4, 3) ∈O+, derived by the transitive result of (4, 2) and (2, 3).

In this case, an illegal instance that violates O is not forbidden by the constraint h′(W, O) = 0. This
indicates that some orders in O+ are not speciﬁed by h′(W, O) = 0 if Step 2 is omitted.

Remark 6. Now we discuss the complexity of gradient calculation for h′(W, O). Equation (9b)
indicates that this complexity is determined by the number |P(O−)| of maximal paths in O−, rather
than the size |O+| of its transitive closure. For a sequential ordering with m variables, h′ contains
only one factor of h regardless of the value of m. This addresses the impractical computational load
of path prohibition constraints with
(m
2
)
factors as discussed in Remark 2. Note that the computa-
tional complexity of h′(W, O) can increase with multiple sequential orderings, which is evaluated
in the following section.

4
Experiments

We evaluate our module for applying prior partial order (PPO) on linear NOTEARS [Zheng et al.,
2018], NOTEARS-MLP [Zheng et al., 2020], and DAGMA [Bello et al., 2022]. It is named as ‘PPO-
alg-l-p’, where alg is the backbone algorithm, and l,p are settings on partial orders. Representative
results are reported here, and the complete results are available in Appendix E.

Section 4.1 presents the results on synthetic datasets. Section 4.2 discusses the results obtained
using a well-established biological dataset [Sachs et al., 2005]. For computational resources, linear
NOTEARS and DAGMA are executed on a 32-core AMD Ryzen 9 7950X CPU at 4.5GHz, while
NOTEARS-MLP uses an NVIDIA GeForce RTX 3090 GPU, both with a 32GB memory limit. We
conduct ﬁve simulations for each synthetic structure and one simulation for the data and partial order
constraints for each structure.

4.1
Synthetic datasets

Random DAGs are generated using Erdös-Rényi (ER) and scale-free (SF) models with node degrees
in {2, 4} and numbers of nodes d in {20, 30, 50}. For linear SEM, uniformly random weights are
assigned to the weighted adjacency matrix A. Given A, samples are generated by X = AT X +
z, X ∈Rd using noise models {Gaussian (gauss), Exponential (exp)}. Observational samples D ∈
Rn×d are then generated with the sample size n = 4d. For nonlinear cases, uniformly random
weights are assigned to weighted adjacency matrices W1, W2, W3. Based on these matrices, samples
are generated using X = tanh(XW1)+cos(XW2)+sin(XW3)+z with z ∼N(0, 1). The sample
size is set to n = 20d. The parameter τ in Equation (9c) is set to 1.

To mimic real-world prior partial orders, we generate multiple sequential orderings, referred to as
the chain of ordering. Speciﬁcally, we ﬁrst conduct a topological sort on the DAG to derive a total
ordering π. We then randomly select l chains, each denoting a sub-ordering randomly generated
from π. Here, l is the number of chains and m is the size of each chain. We ﬁrst investigate the case
of a single chain of ordering and then the case of multiple chains of orderings. For single-chained
ordering, l = 1 and m is in {0.5d, 0.75d, d} (where d is the number of nodes). For multi-chained
ordering, l is in {1, 2, 3, 5, 10} and m is ﬁxed at 0.5d.

4.1.1
Single-chained ordering

In this experiment, we examine structure learning using single-chained ordering. The results of
Structural Hamming Distance (SHD), F1 score, and run time for linear NOTEARS are illustrated in
Figure 1. Results for NOTEARS-MLP (nonlinear samples) and DAGMA are presented in Figure 2.

Output quality. Our method demonstrates notable superiority over algorithms without prior infor-
mation in terms of output quality in most cases, with the advantage becoming more pronounced

7


---Page Break---
NOTEARS
PPO-NOTEARS-1-50
PPO-NOTEARS-1-75
PPO-NOTEARS-1-100
Method

Figure 1: Structural discovery in terms of SHD (lower is better), F1-score (higher is better) and CPU time
(log10 s) with NOTEARS on linear data. Rows: graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free}
graphs with kd expected edges. Columns: noise types of SEM. Error bars represent standard errors over 5
simulations. Method: PPO-alg-1-p denotes our method with partial order settings l = 1 and m = p%d.

NOTEARS-MLP
PPO-NOTEARS-1-50
PPO-NOTEARS-1-75
PPO-NOTEARS-1-100
Method

SHD

(a) NOTEARS-MLP on nonlinear data

SHD

DAGMA
PPO-DAGMA-1-50
PPO-DAGMA-1-75
PPO-DAGMA-1-100
Method

(b) DAGMA on linear data with exp noise

Figure 2: Structural discovery in terms of SHD (lower is better), F1-score (higher is better) and CPU time
(log10 s) with NOTEARS-MLP and DAGMA on representative cases.

as the number of nodes in the ordering chain increases. This conﬁrms the effectiveness of our ap-
proach in enhancing structure learning quality with prior partial orders. Some degradation cases
may be caused by invalid priors in the simulated ordering. Since the topological sort for a DAG is
not unique, some random ordering chains may not contribute to revealing the most essential parts of
orderings, especially with smaller chain sizes.

Run time. We observe that our method with a single-chained ordering is typically faster than struc-
ture learning without prior. This efﬁciency is due to the effective management of our module for
single-chained orderings. However, in some cases, such as on the SF4 graph with NOTEARS-MLP
and DAGMA, the efﬁciency can be degraded. This indicates that the impact of partial orders on
efﬁciency can vary with different data distributions and backbone algorithms.

4.1.2
Multi-chained ordering

The results for SHD, F1-score, and run time using linear NOTEARS and NOTEARS-MLP with
multi-chained orderings are presented in Figure 3. The output quality demonstrates similar trends to

8


---Page Break---
d=20
d=30
d=50
CPU Time
d=20
d=30
d=50
CPU Time
NOTEARS
PPO-NOTEARS-1-50
PPO-NOTEARS-2-50
PPO-NOTEARS-3-50
Method
PPO-NOTEARS-10-50
PPO-NOTEARS-5-50

SHD

(a) NOTEARS on linear data with gauss noise

SHD

d=20
d=30
d=50
CPU Time
d=20
d=30
d=50
CPU Time
NOTEARS-MLP
PPO-NOTEARS-1-50
PPO-NOTEARS-2-50
PPO-NOTEARS-3-50
Method
PPO-NOTEARS-10-50
PPO-NOTEARS-5-50

(b) NOTEARS-MLP on nonlinear data

Figure 3: Structural discovery in terms of SHD (lower is better), F1-score (higher is better) and CPU time
(log10 s) with linear NOTEARS and NOTEARS-MLP. Method: PPO-alg-l-50 represents our method where l
is the number of chains and the size of chains is m = 0.5d.

those seen with single-chained ordering, with more pronounced improvements over the baselines as
the number of partial order constraint chains increases. The run time dynamics are illustrated with
a curve that reﬂects changes corresponding to the primary inﬂuencing factor: the number of chains.
Initially, run time increases and then decreases as the number of chains grows. This pattern results
from the increasing complexity of gradient calculations for the constraint term h′, which scales with
the number of maximal paths in the partial orders. At ﬁrst, the increasing number of chains leads to
more maximal paths in the partial order, thus delaying time efﬁciency. As the partial order becomes
denser, more chains can be covered by a longer chain, which leads to fewer maximal paths in its
transitive reduction, resulting in better time efﬁciency.

Table 1: Structural discovery in terms of SHD↓, FDR↓, TPR↑and F1↑on sach dataset with various sample
sizes. Linear NOTEARS is used as the backbone algorithm, and PPO-1-m denotes our method using a single-
chained ordering containing m nodes. The best result is highlighted with bold texts.

Method
sach-50
sach-100
sach-500
sach-853
SHD FDR TPR
F1
SHD FDR TPR
F1
SHD FDR TPR
F1
SHD FDR TPR
F1
NOTEARS
25
0.76 0.41 0.30
16
0.65 0.41 0.38
15
0.60 0.35 0.38
12
0.57 0.35 0.39
PPO-1-6
20
0.70 0.35 0.32
15
0.62 0.29 0.33
13
0.50 0.35 0.41
14
0.55 0.29 0.36
PPO-1-8
16
0.59 0.41 0.41
11
0.42 0.41 0.48
12
0.40 0.35 0.44
10
0.30 0.41 0.52
PPO-1-11
12
0.27 0.47 0.57
10
0.00 0.41 0.58
11
0.13 0.41 0.56
11
0.13 0.41 0.56

4.2
Real-world data

The dataset provided by Sachs et al. [2005] consists of continuous measurements of protein and
phospholipid expression levels in human immune system cells. It is frequently used as a benchmark
in graphical models due to its associated consensus network, which includes 11 nodes and 17 edges,
based on experimental annotations recognized by the biological community.

We use experimental data from one of the cells with 853 samples. To mimic varying levels of exper-
imental resources, we selected the ﬁrst s data samples for testing, where s ∈{50, 100, 500, 853}. A
single-chained prior ordering is given involving different numbers of variables in {6, 8, 11}. Linear
NOTEARS serves as the backbone algorithm. The parameter τ in Equation (9c) is set to 3.

The structural evaluation metrics reported in Table 1 include SHD, False Discovery Rate (FDR),
True Positive Rate (TPR), and F1 score. The ﬁndings reveal that NOTEARS with partial orders
discovers more accurate structures than NOTEARS without prior information in most cases. Re-
markably, even with the smallest sample size (50), NOTEARS with partial order constraint (11
nodes) signiﬁcantly outperforms the baseline using the largest sample size (853). This underscores
the efﬁcacy of the proposed partial order constraint-based differentiable structure learning approach
in conserving experimental resources in scientiﬁc research contexts.

9


---Page Break---
5
Discussion

Limitations and future directions
Despite the theoretical correctness and completeness of the
proposed method for integrating partial orders in differentiable structure learning, there are some
practical limitations.

First, the augmented acyclicity constraint h′(W, O) = 0 cannot be strictly satisﬁed during the op-
timization process of the augmented Lagrangian method, as proven by Wei et al. [2020]. This may
result in some order constraints from the prior not being satisﬁed in the output. This issue is inher-
ent to the optimization aspect of differentiable structure learning and may be addressed with more
reﬁned optimization techniques in the future.

Additionally, although the proposed method efﬁciently handles long sequential orderings, its efﬁ-
ciency can be impacted by partial orders with complex structures. This is evident from the experi-
mental results involving multi-chained orderings. We randomly selected multiple ordering chains,
each comprising half of the nodes, forming a complex order structure with considerable maximal
paths. This leads to a sharp increase in the number of constraint terms in the augmented acyclicity
constraint. Fortunately, real-world ordering priors typically do not exhibit such complex structures
and can usually be captured by a few chains. To enhance time efﬁciency in such cases, a more re-
ﬁned characterization method could be explored to reduce computational overhead in the future. We
may focus on improving the gradient calculation of the proposed augmented acyclicity constraint in
the context of multiple sequential orderings. This can be explored by merging the common parts of
the gradient calculation process or developing more efﬁcient characterizations.

Conclusion
This paper enhances the ﬁeld of differentiable structure learning by enabling this
framework to apply priors of partial order constraints. We systematically analyze the related chal-
lenges of applying ﬂexible order constraints and propose a novel and effective strategy to address
them by augmenting the acyclicity constraint. We present a theoretical proof conﬁrming the cor-
rectness and completeness of our strategy. Empirical results highlight the superiority of our method
in improving structure learning with partial order constraints. Results on a well-known real-world
dataset further emphasize its potential in uncovering more accurate causal mechanisms with reduced
experimental resources.

Broader Impact

The proposed method allows researchers across various scientiﬁc ﬁelds to specify ordering priors,
enhancing causal discovery with state-of-the-art differentiable structure learning algorithms from
experimental or observational data. As demonstrated with the real-world Sachs dataset [Sachs et al.,
2005], differentiable structure learning using a proper ordering prior with only 10% of the samples
required by methods without a prior yields signiﬁcantly better structures. This reduction in data
requirements for causal discovery can save experimental resources across many domains.

However, researchers must exercise caution when providing ordering priors. The proposed method
strictly adheres to these priors, and numerous incorrect ordering priors can severely impact the
results. For instance, in social sciences, if incorrect assumptions about the order of socio-economic
events are used as priors, the resulting causal model may be misleading, affecting policy decisions
based on such a model.

Acknowledgements

This research was supported in part by the National Key R&D Program of China (No.
2021ZD0111700), in part by the National Nature Science Foundation of China (No. 62137002,
62176245, 62406302), in part by the Natural Science Foundation of Anhui province (No.
2408085QF195), in part by the Key Research and Development Program of Anhui Province (No.
202104a05020011), in part by the Key Science and Technology Special Project of Anhui Province
(No. 202103a07020002).

10


---Page Break---
References

Kevin Bello, Bryon Aragam, and Pradeep Ravikumar. Dagma: Learning dags via m-matrices and a
log-determinant acyclicity characterization. Advances in Neural Information Processing Systems,
35:8226–8239, 2022.

Nicolas Bruffaerts, Tom De Smedt, Andy Delcloo, Koen Simons, Lucie Hoebeke, Caroline Ver-
straeten, An Van Nieuwenhuyse, Ann Packeu, and Marijke Hendrickx. Comparative long-term
trend analysis of daily weather conditions with daily pollen concentrations in brussels, belgium.
International Journal of Biometeorology, 62:483–491, 2018.

Eunice Yuh-Jie Chen, Yujia Shen, Arthur Choi, and Adnan Darwiche. Learning Bayesian networks
with ancestral constraints. Advances in Neural Information Processing Systems, 29, 2016.

Wenyu Chen, Mathias Drton, and Y Samuel Wang. On causal discovery with an equal-variance
assumption. Biometrika, 106(4):973–980, 2019.

David Maxwell Chickering. Optimal structure identiﬁcation with greedy search. Journal of machine
learning research, 3(Nov):507–554, 2002.

Gregory F Cooper and Edward Herskovits. A bayesian method for constructing bayesian belief
networks from databases. In Uncertainty Proceedings 1991, pages 86–94. Elsevier, 1991.

Chang Deng, Kevin Bello, Bryon Aragam, and Pradeep Kumar Ravikumar. Optimizing notears
objectives via topological swaps. In International Conference on Machine Learning, pages 7563–
7595. PMLR, 2023.

Chang Deng, Kevin Bello, Pradeep Ravikumar, and Bryon Aragam. Global optimality in bivariate
gradient-based dag learning. Advances in Neural Information Processing Systems, 36:17929–
17968, 2023.

Brian Denton, James Viapiano, and Andrea Vogl. Optimization of surgery sequencing and schedul-
ing decisions under uncertainty. Health Care Management Science, 10:13–24, 2007.

Maxime Gasse, Alex Aussem, and Haytham Elghazel. A hybrid algorithm for bayesian network
structure learning with application to multi-label learning. Expert Systems with Applications,
41(15):6755–6772, 2014.

Geoffrey Grimmett and David Stirzaker. Mathematics for computer science. MIT OpenCourse-
Ware, Massachusetts Institute of Technology, 2015. Available at: https://ocw.mit.edu/courses/
6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_session18.pdf.

Christina Heinze-Deml, Marloes H Maathuis, and Nicolai Meinshausen. Causal structure learning.
Annual Review of Statistics and Its Application, 5:371–391, 2018.

Andrew Li and Peter Beek. Bayesian network structure learning with side constraints. In Interna-
tional Conference on Probabilistic Graphical Models, pages 225–236. PMLR, 2018.

Po-Ling Loh and Peter Bühlmann. High-dimensional learning of linear causal networks via inverse
covariance estimation. The Journal of Machine Learning Research, 15(1):3065–3105, 2014.

Ignavier Ng, AmirEmad Ghassami, and Kun Zhang. On the role of sparsity and dag constraints
for learning linear dags. Advances in Neural Information Processing Systems, 33:17943–17954,
2020.

Ignavier Ng, Sébastien Lachapelle, Nan Rosemary Ke, Simon Lacoste-Julien, and Kun Zhang. On
the convergence of continuous constrained optimization for structure learning. In International
Conference on Artiﬁcial Intelligence and Statistics, pages 8176–8198. PMLR, 2022.

Eric N Olson. Gene regulatory networks in the evolution and development of the heart. Science,
313(5795):1922–1927, 2006.

Rainer Opgen-Rhein and Korbinian Strimmer. From correlation to causation networks: a simple
approximate learning algorithm and its application to high-dimensional plant gene expression
data. BMC systems biology, 1:1–10, 2007.

11


---Page Break---
Roxana Pamﬁl, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilgerstorfer, Konstantinos Geor-
gatzis, Paul Beaumont, and Bryon Aragam. Dynotears: Structure learning from time-series data.
In International Conference on Artiﬁcial Intelligence and Statistics, pages 1595–1605. PMLR,
2020.

Judea Pearl et al. Models, reasoning and inference. Cambridge, UK: CambridgeUniversityPress,
19(2):3, 2000.

Garvesh Raskutti and Caroline Uhler. Learning directed acyclic graph models based on sparsest
permutations. Stat, 7(1):e183, 2018.

Karen Sachs, Omar Perez, Dana Pe’er, Douglas A Lauffenburger, and Garry P Nolan.
Causal protein-signaling networks derived from multiparameter single-cell data.
Science,
308(5721):523–529, 2005.

Liam Solus, Yuhao Wang, and Caroline Uhler. Consistency guarantees for greedy permutation-based
causal inference algorithms. Biometrika, 108(4):795–814, 2021.

Peter Spirtes, Clark Glymour, and Richard Scheines. Causation, prediction, and search. MIT press,
2000.

Chandler Squires, Yuhao Wang, and Caroline Uhler. Permutation-based causal structure learning
with unknown intervention targets. In Conference on Uncertainty in Artiﬁcial Intelligence, pages
1039–1048. PMLR, 2020.

Xiangyu Sun, Oliver Schulte, Guiliang Liu, and Pascal Poupart. Nts-notears: Learning nonpara-
metric dbns with prior knowledge. In International Conference on Artiﬁcial Intelligence and
Statistics, pages 1942–1964. PMLR, 2023.

Marc Teyssier and Daphne Koller. Ordering-based search: a simple and effective algorithm for
learning bayesian networks. In Proceedings of the Twenty-First Conference on Uncertainty in
Artiﬁcial Intelligence, pages 584–590, 2005.

Ioannis Tsamardinos, Laura E Brown, and Constantin F Aliferis.
The max-min hill-climbing
bayesian network structure learning algorithm. Machine learning, 65:31–78, 2006.

Xiaoqiang Wang, Yali Du, Shengyu Zhu, Liangjun Ke, Zhitang Chen, Jianye Hao, and Jun Wang.
Ordering-based causal discovery with reinforcement learning. arXiv preprint arXiv:2105.06631,
2021.

Dennis Wei, Tian Gao, and Yue Yu. Dags with no fears: A closer look at continuous optimization for
learning bayesian networks. Advances in Neural Information Processing Systems, 33:3895–3906,
2020.

Jing Xiang and Seyoung Kim. A* lasso for learning a sparse Bayesian network structure for contin-
uous variables. Advances in neural information processing systems, 26, 2013.

Xing Yang, Chen Zhang, and Baihua Zheng. Segment-wise time-varying dynamic bayesian network
with graph regularization. ACM Transactions on Knowledge Discovery from Data, 16(6):1–23,
2022.

Yue Yu, Jie Chen, Tian Gao, and Mo Yu. DAG-GNN: DAG structure learning with graph neural
networks. In International Conference on Machine Learning, pages 7154–7163. PMLR, 2019.

Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. Dags with no tears: Continuous
optimization for structure learning. Advances in Neural Information Processing Systems, 31,
2018.

Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, and Eric Xing. Learning sparse non-
parametric dags. In International Conference on Artiﬁcial Intelligence and Statistics, pages 3414–
3425. PMLR, 2020.

Shengyu Zhu, Ignavier Ng, and Zhitang Chen. Causal discovery with reinforcement learning. In
International Conference on Learning Representations, 2019.

12


---Page Break---
Appendix A
Related Work

A.1
Combinatorial optimization of structure learning

Traditional combinatorial optimization of structure learning includes constraint-based, score-based,
and hybrid methods. Constraint-based methods utilize conditional independence (CI) tests to con-
struct the graph. The most notable example is the PC algorithm (named after its developers, Peter
and Clark), which begins with a complete graph and progressively removes edges between nodes
that are conditionally independent given a set of other variables [Spirtes et al., 2000]. Constraint-
based approaches typically result in a partially directed acyclic graph (PDAG), where some edges
remain undirected due to equivalent DAG conﬁgurations.

Score-based methods employ a scoring function to assess the ﬁt of a DAG model to the observed
data, aiming to identify the graph with the best score. Commonly used scoring functions, such
as BIC, BDeu, and MDL, are decomposable, which facilitates local optimization for each variable.
Notable algorithms in this category include the Greedy Equivalent Search (GES) [Chickering, 2002]
and the K2 algorithm [Cooper and Herskovits, 1991].

A signiﬁcant subset of score-based methods assumes a total ordering of the variables to expedite
ﬁnding the optimal solution within this constraint, followed by a search among different order hy-
potheses to locate the most ﬁtting order. Teyssier and Koller [2005] demonstrated that with a ﬁxed
ordering, the optimal solution could be computed in polynomial time, avoiding the need for DAG
consistency checks. Although searching the order space is computationally intensive, these meth-
ods have outperformed traditional DAG-based searches. Subsequent studies have expanded on this
framework [Xiang and Kim, 2013; Raskutti and Uhler, 2018; Squires et al., 2020; Wang et al., 2021;
Solus et al., 2021; Chen et al., 2019].

Hybrid methods integrate score-based and constraint-based approaches. They streamline the search
for potential parent nodes for each variable by applying CI tests to narrow down the set of candidates.
Examples of hybrid methods include Max-Min Hill Climbing (MMHC) [Tsamardinos et al., 2006]
and H2PC [Gasse et al., 2014].

A.2
Differentiable structure learning

Zheng et al. [2018] introduced NOTEARS, which proposes a continuous characterization of the
DAG constraint. This transforms the structure learning within the structural equation model (con-
tinuous scoring function) into a differentiable constrained optimization problem. The authors used
an augmented Lagrangian method to solve this problem in a linear SEM, observing superior perfor-
mance compared to state-of-the-art structure learning solvers. They further extended this approach
with an MLP learner to address nonlinear SEMs [Zheng et al., 2020]. Subsequent studies have
advanced the ﬁeld of differentiable structure learning by improving acyclicity characterization, de-
veloping more powerful neural network architectures, and enhancing optimization approaches.

Yu et al. [2019] formalized the structure learning task using a graphical neural network (GNN)
and proposed a GNN-based structure learning approach named DAG-GNN. A major advantage of
DAG-GNN is its ability to handle both discrete data (modeled by Bayesian networks) and continu-
ous data (modeled by SEMs). Zhu et al. [2019] introduced a reinforcement learning approach for
structure learning. Yu et al. [2019] used a different polynomial function to characterize acyclicity,
achieving better computational efﬁciency. Bello et al. [2022] proposed DAGMA, which employs a
log-determinant function, showing a stronger constraint on acyclicity. Ng et al. [2020] introduced a
likelihood-based scoring function, demonstrating superiority over the regression-based scoring used
in NOTEARS. Wei et al. [2020] developed a local search post-processing algorithm informed by
the KKT conditions of the constrained optimization problem. Deng et al. [2023a] further intro-
duced a local search differentiable structure learning algorithm based on topological swap, similar
to order-based search in combinatorial structure learning.

Distinct from these studies, this paper aims to enhance the ﬁeld of differentiable structure learning
by integrating more types of prior constraints. Current studies typically address straightforward
prior information such as constraints of edges [Wei et al., 2020] and total ordering [Deng et al.,
2023a]. However, real-world priors are often more ambiguous and less comprehensive. Therefore,

13


---Page Break---
we consider partial orders, which are general enough to represent real-world priors on ordering
relationships that may be contained in experimental settings or existing domain knowledge.

Appendix B
Discussions of Ordering Prior in DBN

This section explores the connection between dynamic Bayesian Networks (DBNs) and the ordering
prior discussed in this paper, and how this prior can be implemented through parameter freezing.
DBNs assume that the observational data are time-series data containing m time slices. Given d
variables, there are m × d nodes in the DBN, where each variable has one "copy" in each time slice.
The inﬂuence patterns among the variables vary according to different assumptions, with a common
premise being the order constraints of the time slices [Pamﬁl et al., 2020; Sun et al., 2023; Yang et
al., 2022]. Speciﬁcally, a variable in a later time slice cannot affect one in an earlier time slice. This
order constraint can be generally described by the following ordering type.

Deﬁnition 8 (Time-series ordering). In a time-series ordering S, the entire set of variables is parti-
tioned into distinct stages. Variables within the same stage do not have a deﬁned ordering relative
to each other, while variables in an earlier stage precede those in a later stage.

In the time-series ordering, the partial order between any two variables in different stages is known.
For a time-series ordering, it can be integrated into the structure learning by parameter freezing.

Theorem 2. A graph G(W) is a DAG and satisﬁes time-series ordering S if and only if h(W) = 0
and Wi,j = 0 for all (j, i) ∈S+.

Proof. The aim is to prove the following equivalence:

h(W) = 0 and Wi,j = 0 for all (j, i) ∈S+ ⇐⇒G(W) ∈DAG and Xi ⇝Xj /∈G(W) for all (j, i) ∈S+.

Necessity (⇐=):

1. If G(W) is a DAG, then by Proposition 1 in the main text, h(W) = 0. 2. If Xi ⇝Xj /∈G(W)
for all (j, i) ∈S+, this implies that there is no directed edge from i to j for any (j, i) ∈S+.
Therefore, Wi,j = 0 for all (j, i) ∈S+.

Thus, if G(W) is a DAG and Xi ⇝Xj /∈G(W) for all (j, i) ∈S+, then h(W) = 0 and Wi,j = 0
for all (j, i) ∈S+.

Sufﬁciency (=⇒):

1. If h(W) = 0, then by Proposition 1 in the main text, G(W) is a DAG. 2. We need to show that
Wi,j = 0 for all (j, i) ∈S+ implies Xi ⇝Xj /∈G(W) for all (j, i) ∈S+.

We prove this by contradiction. Assume that there exists a pair (v, u) ∈S+ such that Xu ⇝Xv ∈
G(W). This means there is a directed path from u to v in G(W).

Let the path be (i0, i1, · · · , iq−1, iq) where i0 = u and iq = v. Since v is in an earlier stage than u, v
precedes u in the partial order S+. Therefore, there must exist at least one pair (ik, ik+1) in the path
such that ik+1 is in an earlier stage than ik. This implies (ik+1, ik) ∈S+, and hence Wik,ik+1 ̸= 0.

However, this contradicts our assumption that Wi,j = 0 for all (j, i) ∈S+. Therefore, our assump-
tion is false, and it must be that Xi ⇝Xj /∈G(W) for all (j, i) ∈S+.

Thus, if h(W) = 0 and Wi,j = 0 for all (j, i) ∈S+, then G(W) is a DAG and Xi ⇝Xj /∈G(W)
for all (j, i) ∈S+.

This result leads to a parameter freezing strategy for applying the time-series ordering into differen-
tiable structure learning:

min
W ∈Rd×d F(W)
subject to h(W) = 0 and Wi,j = 0 for all (j, i) ∈S+
(10)

14


---Page Break---
Appendix C
Proof of Main Results

C.1
Proof of Theorem 1

This section proves the equivalence of the constraint equality h′(W, O) as designed in Equation (9)
to the adherence to the partial order constraint O.
Theorem 1. A graph G is a DAG and satisﬁes a set of partial orders O if and only if h′(W, O) = 0
for the function h deﬁned by Equation (4) and h′ deﬁned by Equations (9b), (9c), and (9d).

Recall the construction of h′: The function
h′(W, O) =
∑

o∈P(O−)
h(A(W, o))

where P(O−) represents the set of maximal paths in O−. A(W, o) returns a adjacent matrix that
represents a graph by adding the path o to the graph G(W), denoted as E(G(A)) = E(G(W)) ∪o.
Given that h(W) ≥0 by Equation (4), we have:

h′(W, O) = 0 ⇐⇒h(A(W, o)) = 0 for o ∈P(O−)
(11)

The proof of Theorem 1 is completed by two statements. We consider the ﬁrst one:
Lemma 1. A graph G is a DAG and satisﬁes a sequential ordering o = {(p1, p2, · · · , pm)} if and
only if graph G′ is a DAG where E(G′) = E(G) ∪o.

Proof. Recall the equivalent constraints of the order constraint in Proposition 4, and we have the
following objective to prove:

G′ ∈DAG for E(G′) = E(G) ∪o ⇐⇒G ∈DAG and Xj ⇝Xi /∈G for (i, j) ∈o+
(12)
(⇐=) We prove the necessity by contradiction. Suppose that the right side holds and G′ is not a
DAG. Consider a cycle (c1, c2, · · · , ck, c1) in G′ under the following two cases.

1) No edge in the cycle is contained in o. Then all these edges in the cycle belong to G by the
condition E(G′) = E(G) ∪o. This conﬂicts with the fact that G is a DAG.

2) If some edges are contained in o and they form a consecutive path (cr1, cr1+1, · · · , cq1). In
this case, the rest part of the cycle (cq1, cq1+1, · · · , cr1) is contained in the graph G. This forms a
directed path from cq1 to cr1 for (cr1, cq1) ∈o+, which conﬂicts with the condition of the right-hand
side of Equality (12) Xj ̸⇝Xi /∈G for (i, j) ∈o+.

3) Some edges are contained in o and do not form a single consecutive path. Then we can represent
them as a set of disjoint paths in sequence:
ro = {(cr1, cr1+1, · · · , cq1), (cr2, cr2+1, · · · , cq2), · · · , (crl, crl+1, · · · , cql)}
where 1 ≤qi < ri+1 ≤k for i ∈{1, 2, · · · , l}. Consider the rest parts of the cycle in G:

rG = {(cq1, cq1+1, · · · , cr2), (cq2, cq2+1, · · · , cp3), · · · , (cql, · · · , ck, c1, · · · , cr1)}
Consider cqi and cri+1 for arbitrary i. Since they are included in o and o is a sequence, we have:

Either (cqi, cri+1) or (cri+1, cqi) is contained in o+
(13)
Recall that all the paths in rG belong to G, then the path (cqi, cqi+1, · · · , cri+1) belongs to G.
Combined with the right-hand condition in (12), we have (cqi, cri+1) /∈o+.
Then we have
(cqi, cri+1) ∈o+ by (13). This conclusion holds for all i ∈{1, 2, · · · , l} (note that rl+1 = r1).
Then we have:

{(cq1, cr2), (cq2, cr3), · · · , (cql, cr1)} ⊂o+
(14)
Recall that paths in ro are contained in o, and we also have:

{(cr1, cq1), (cr2, cq2), · · · , (cr+l, cql)} ⊂o+
(15)
Combining (14) and (15), we derive that a cycle (cr1, cq1, cr2, · · · , cql, cr1) is contained in o+, which
conﬂicts the premise that o is acyclic.

(=⇒) Since E(G) ⊂E(G′) and G is DAG, we have that G is DAG. Since the path o is contained
in G′, we have Xi ⇝Xj ∈G for all (i, j) ∈o+. Then we have that no path Xj ⇝Xi exists in G′
as it will introduce a cycle. The proof is completed.

15


---Page Break---
Now we consider the second statement:
Lemma 2. For the set P(O−) of all maximal paths of O−, the union of the transitive closures of
these paths is the transitive closure of O: ∪

o∈P(O−) o+ = O+

Proof. We have o+ ⊆O+ by that o ⊆O−⊆O. Therefore ∪
o∈P(O−) o+ ⊆O+. Then we prove
O+ ⊆∪
o∈P(O−) o+ by contradiction. Suppose that ∃(i, j) ∈O+ such that (i, j) /∈∪
o∈P(O−) o+.
We have that (i, j) ∈(O−)+ by that (O−)+ = O+. This indicates that a path from i to j exists
in O−. Moreover, we have that this path does not belong to any maximal path of O−by that
(i, j) /∈∪

o∈P(O−) o+. This introduces contradiction since any path in a graph at least belongs to
one of its maximal path (we can extend any path to be a maximal path).

Now we derive the result of Theorem 1 by these results. Consider the result of Lemma 1:

G′ ∈DAG for E(G′) = E(G) ∪o ⇐⇒G ∈DAG and Xj ⇝Xi /∈G for (i, j) ∈o+

The left-hand condition equals h(A(W, o)) = 0. Combining this with Equation (11), we have:

h′(W, O) = 0 ⇐⇒G ∈DAG and Xj ⇝Xi /∈G for (i, j) ∈∪o∈P(O−)o+

With the result ∪o∈P(O−)o+ = O+ from Lemma 3, we have:

h′(W, O) = 0 ⇐⇒G ∈DAG and Xj ⇝Xi /∈G for (i, j) ∈O+

The path absence condition on the right-hand side is equivalent to adherence to O. Hence, we
complete the proof of Theorem 1.

C.2
Proof of the rest statements

Proposition 1. (Theorem 1 in [Wei et al., 2020]). The directed graph of a non-negative adjacency
matrix B is a DAG if and only if h(B) = 0 for any h deﬁned by (4).

Proof. Recall the deﬁnition of h(B):

h(B) = Trace(

d
∑

i=1
ciBi), ci > 0

This function can be understood by interpreting the diagonal elements of the matrix powers Bi,
which represent the weighted i-length directed paths from a variable to itself, essentially, a cycle. A
matrix B represents a DAG if and only if there are no i-length cycles for any i ∈N+, evidenced by
Trace(Bi) = 0. Consequently, Trace(Bi) = 0 must hold for all i ∈N+ if B describes a DAG. The
converse is supported by Lemma 3, which conﬁrms that the absence of cycles ranging from 1-length
to d-length, as characterized by Equation (4), sufﬁciently ensures the acyclicity of the graph.

Lemma 3. Any cyclic graph with d variables must contain cycle(s) with less than d length.

Proof. We prove this lemma by contradiction. Suppose that the cycle with the minimum length
l > d as (i1, i2, · · · , il, i1). Since there are d nodes, there exist at least two indexes in the cycle
iq, ip that refers to the same node, i.e., iq = ip. Suppose that that q < p, then the new cycle
(i1, · · · , iq−1, ip+1, · · · , i1 by cutting the paths from iq to ip has a lower length. This conﬂicts with
that the original cycle has the minimum length.

Proposition 3. A graph G is a DAG and satisﬁes total ordering π if and only if edge (u, v) does not
exist in G for all (v, u) ∈π+.

Proof. Firstly, we formalize this statement according to Proposition 4 as follows:

(u, v) /∈G for (v, u) ∈π+ ⇐⇒G(W) ∈DAG and Xu ⇝Xv /∈G for (v, u) ∈π+

The necessity (⇐=) is straightforward by that Xu ⇝Xv /∈G =⇒(u, v) /∈G. For sufﬁciency
(=⇒), we employ a proof by contradiction. Assume that there exists a path (u1, u2, . . . , uk) in G

16


---Page Break---
that violates π, such that (uk, u1) ∈π+. For ui and ui+1, either (ui, ui+1) ∈π+ or (ui+1, ui) ∈π+
given that the order of any pairwise nodes in contained in π+. Combined with the transitivity of or-
ders, there must be at least one edge (ui, ui+1) ∈E(G) where (ui+1, ui) ∈π+. This contradicts
the condition that (u, v) /∈G for all (v, u) ∈π+. To demonstrate the absence of cycles (DAG
constraint), consider setting uk = u1, which similarly leads to a contradiction under these condi-
tions.

Proposition 2. There exists at least one topological sort of DAG G that satisﬁes the partial order
set O if and only if, for any order (i, j) in O+, Xj is not an ancestor of Xi in G.

Proof. (=⇒) If there exists a topological sort that satisﬁes O, then for any (i, j) ∈O+, Xj is not
an ancestor of Xi in G.

Suppose there exists a topological sort T of G that satisﬁes O. For any (i, j) ∈O+, i must appear
before j in the topological sort T because O+ represents a transitive closure of the order constraints.
If Xj were an ancestor of Xi in G, there would be a path from Xj to Xi. In a topological sort, for
any edge (u, v), u must appear before v. Therefore, if Xj were an ancestor of Xi, Xj would appear
before Xi in the topological sort. However, this would contradict the fact that i appears before j in
the topological sort T (as i is related to j by O+). Thus, Xj cannot be an ancestor of Xi in G.

(⇐=) If for any order (i, j) in O+, Xj is not an ancestor of Xi in G, then there exists a topological
sort of G that satisﬁes O.

Assume for any (i, j) ∈O+, Xj is not an ancestor of Xi in G. This implies there is no directed path
from Xj to Xi for any (i, j) ∈O+). Construct a topological sort of G using a standard topological
sorting algorithm (such as Kahn’s Algorithm or Depth-First Search). During the construction, ensure
that for any (i, j) ∈O, i appears before j in the ordering. Since O+ does not create any cycles
(because Xj is not an ancestor of Xi), the ordering respects the partial order O. The resulting
topological sort T satisﬁes the constraints of O by construction.

Proposition 4. No directed path Xi ⇝Xj exists in G(W) if and only if
(∑d
l=1(W ◦W)l)

i,j = 0.

Proof. ∑d
l=1(W ◦W)l is an instance of the acyclicity function deﬁned in Equation (4). By the
proof of Proposition 1, setting the matrix entry (i, j) to zero effectively blocks all paths of lengths
ranging from 1 to d from Xi to Xj, fulﬁlling the necessary condition for Xi ⇝Xj ̸∈G(W). This
condition is also deemed sufﬁcient due to Lemma 4.

Lemma 4. If a directed path Xi ⇝Xj exists in a graph with d nodes, then a directed path Xi ⇝Xj
of length less than d must also exist.

The proof of Lemma 4 is similar to that of lemma 3.

Appendix D
Implementation of the Proposed Method

This section presents the psudocodes of the implementations of the proposed method.

Algorithm 1 Partial Order Constraint-based Differentiable Structure Learning

Require: Observational data D ∈Rm×d; A set of partial orders O.
Ensure: A DAG G.

1: W ←Solve minW F(W, D) subject to h′(W, O) = 0 by a backbone algorithm ▷h′ is deﬁned
by Algorithm 2.
2: Construct G by adding edges (i, j) where |Wi,j| > γ
▷γ > 0 is the threshold value, which is
set to 0.3 in experiments.
3: return G

17


---Page Break---
Algorithm 2 Augmented Acyclicity Characterization Function h′ : Rd×d →R

Require: A weighted matrix W ∈Rd×d; A set of partial orders O; An acyclicity characterization
function h : Rd×d →R.
1: O−←Derive the transitive reduction of O by Algorithm 4
2: G ←Construct a graph by adding edges (i, j) where (i, j) ∈O−

3: result ←0
4: P ←Find all maximal paths in G by Algorithm 3
5: for each path o in P do
6:
W ′ ←Replace the (i, j)th element of W with τ for each edge (i, j) ∈o
▷τ > 0 is a
hyper-parameter that controls the strength of the order constraints.
7:
result ←result + h(W ′)
8: end forreturn result

Algorithm 3 Find All Maximal Paths in a DAG

Require: A directed acyclic graph G = (V, E)
Ensure: A set of all maximal paths in G

1: function FINDMAXIMALPATHS(G)
2:
Initialize an empty list all_paths
3:
Identify all nodes in G with no incoming edges as start_nodes
4:
for each node u in start_nodes do
5:
Call DFS(u, [])
6:
end for
7:
return all_paths
8: end function
9: function DFS(node, path)
10:
Append node to path
11:
Initialize extensions_found to False
12:
for each v such that there is an edge from node to v do
13:
Set extensions_found to True
14:
Call DFS(v, path)
15:
end for
16:
if not extensions_found then
17:
Add path to all_paths
▷Path is maximal
18:
end if
19:
Remove node from path
▷Backtrack
20: end function

Algorithm 4 Derive Transitive Reduction of a Relation

Require: A set of ordered pairs O representing a relation.
Ensure: Transitive reduction of O.

1: Compute R, the transitive closure of O
2: Initialize T ←O
3: for each pair (i, j) ∈O do
4:
for each pair (k, l) ∈O do
5:
if i ̸= k and j ̸= l then
6:
if (i, k) ∈R and (k, j) ∈R then
7:
Remove (i, j) from T
8:
end if
9:
end if
10:
end for
11: end for
12: return T

18


---Page Break---
Appendix E
Complete Experimental Results

This section presents the complete experimental results and detailed experimental settings. The
hyper-parameters of algorithms include: the threshold γ = 0.3 for the existence of an edge, the
weight λ1 = 0.03 for L1 regularization term, the weight λ2 = 0.01 for L2 regularization term,
the weight τ = 1 in the proposed augmented acyclicity term, the maximum ρ value ρmax = 1016
for augment Lagrangian method. The NOTEARS-MLP uses an MLP with an input dimension d, a
hidden layer of size d × 10, and an output layer of size d, where d is the number of variables. The
evaluation settings are detailed in the caption of each ﬁgure.

The evaluation metrics used in the study include the following: Structural Hamming Distance
(SHD), which counts the number of edges that differ between the recovered structure and the ground
truth; True Positive Rate (TPR), deﬁned as TP/(TP + FN), where TP is the number of correctly
recovered edges and FN is the count of missing or reversed edges; False Discovery Rate (FDR),
calculated as FP/(FP + TP), where FP represents the number of extra edges; and the F1-score,
computed as 2 · P/(P + R), with R = TPR and P = 1 −FDR.

The runtime of the algorithm refers to the time taken to generate the ﬁnal result, speciﬁcally the
time required for the acyclicity term h(·) to reach a sufﬁciently small value (set to 10−8 in the
experiments).

We ﬁrst report two supplementary results regarding 1) the comparison of the pure path absence-
based implementation partial order constraints as shown in Equation (8) and 2) the selection of
hyper-parameter τ in Equation (9c). Following this, we report the supplementary results to those in
the main texts.

Table 2: Comparison of path absence- and augmented acyclicity-based partial ordering constraints for structure
learning in terms of run time t (s) and output quality F1-score.

Prior
Metric
d=20
d=30
d=40
d=50

Na
t / F1
18.3 / 0.47
35.1 / 0.51
44.7 / 0.54
71.6 / 0.49
0.1d
t (s)
19.7 / 107.3
41.1 / 170.9
65.8 / 316.5
117.5 / 6523.0
F1
0.42 / 0.39
0.49 / 0.52
0.56 / 0.54
0.48 / 0.52
0.5d
t (s)
26.2 / 328.3
21.3 / 221.1
41.9 / 271.8
106.7 / 4124.5
F1
0.50 / 0.53
0.60 / 0.62
0.59 / 0.57
0.52 / 0.52
d
t (s)
4.2 / 140.0
7.2 / 56.5
12.1 / 83.0
41.9 / 1475.3
F1
0.71 / 0.71
0.75 / 0.75
0.72 / 0.72
0.70 / 0.70

E.1
Comparison of Path Absence- and Augmented Acyclicity-based Partial Order
Constraints

We implemented Equation (8), which applies straightforward path absence constraints to enforce
partial order, and compared its performance with our augmented acyclicity-based method. The
results, averaged over 3 repetitions, in terms of runtime and F1-score using NOTEARS are shown
in Table 2. The experimental settings include: ER-4 structure, linear data with Gaussian noise, a
sample size of 40, and a single-chained partial order with nodes [0.1d, 0.5d, d]. Results are presented
as Ours/Equation 8, with Prior Na representing the baseline (NOTEARS without prior constraints).

The results demonstrate that the method in Equation (8) achieves comparable output quality to our
main approach, showing that it effectively represents partial order constraints. However, as expected,
Equation (8) exhibits signiﬁcantly slower runtime. Interestingly, the most signiﬁcant slowdown
occurs when fewer variables are included in the ordering than the total variable set. This may be due
to the method’s disjoint integration with the acyclicity term, possibly causing the partial order term
to take longer to reconcile with the acyclicity constraint when the prior information is insufﬁcient.

E.2
Analysis of Selection of Virtual Edge Weight τ

The hyper-parameter τ > 0 serves as the uniform weight for edges in the set P(O−), which are
added solely to the augmented acyclicity term h′(W, O) and not to the data approximation term

19


---Page Break---
Table 3: Variance of different properties in partial order-based structure learning with varying τ values.

τ
0
0.1
0.2
0.3
0.4
0.5
1
2
3
4
5
6
7
8

h
4.E-01
1.E-04
2.E-01
2.E-06
4.E-07
1.E-09
4.E-09
1.E-10
5.E-11
6.E-09
3.E-09
7.E-05
1.E-12
0.E+00
h′
6.E-09
2.E-05
6.E-04
2.E-08
2.E-08
3.E-08
6.E-09
5.E-09
4.E-09
2.E-10
8.E-08
4.E+00
1.E-02
5.E-02
F
9.6
5.E+03
4.E+03
12.9
12.7
16.9
11.7
11.7
11.5
11.7
8.E+02
2.E+04
3.E+04
3.E+04
DAG?



✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
@Edge
53
31
55
61
62
69
71
70
66
67
33
1
0
0
F1
0.27
0.27
0.36
0.61
0.65
0.56
0.68
0.69
0.7
0.69
0.23
0
0
0

F(W). These edges, termed "virtual" edges, may not exist in the actual graph but are included
in the "acy-graph" used to enforce acyclicity constraints. The following analysis highlights the
theoretical impact of different values of τ.

Small τ: When τ is too small, it fails to properly reﬂect the presence of virtual edges in the acy-
graph, leading to inadequate enforcement of prior information. This can result in reversed paths
not being properly excluded, weakening the acyclicity constraint and potentially allowing cycles to
form. Essentially, a very small τ removes the effect of virtual edges in P(O−), diminishing the
strength of the acyclicity constraint and compromising the learning process.

Large τ: On the other hand, if τ is too large, it can disrupt numerical stability by overshad-
owing small weights that signify absent edges.
For example, consider a forbidden cycle p =
W1,2W2,3W3,1, where W1,2 = τ is set to a large value, while W2,3 = r−(indicating edge ab-
sence). This conﬁguration results in ∇W3,1p = τ × r−, making the inﬂuence of W1,2 on W3,1 large
enough to enforce the absence of W3,1, even though the cycle should be considered broken due to
the absence of W2,3. This misinterpretation can lead to the erroneous removal of edges, potentially
producing an empty graph.

We conducted experiments with varying values of τ and observed the real acyclicity loss h, aug-
mented acyclicity loss h′, data approximation loss F, and output metrics such as DAG condition,
edge count, and F1 score. Results for an ER-4 graph with Gaussian noise model, node count d = 20,
sample size n = 40, and edge threshold γ = 0.1 support this theoretical analysis and are presented
in Table 3.

E.3
Supplementary Results

This section presents supplementary results to the main text, providing additional insights across
complete datasets, various data settings, and methods used. The ﬁgures included, Figures 4-8, ex-
pand on the analysis, offering a more comprehensive understanding of the experimental outcomes.

20


---Page Break---
NOTEARS-Linear-SEM

PPO-NOTEARS-1-50

PPO-NOTEARS-1-75

PPO-NOTEARS-1-100

Method

Figure 4: Results of single-chained ordering for NOTEARS with linear SEM. Structural discovery in terms
of SHD (lower is better), F1-score (higher is better), FDR (lower is better), TPR (higher is better) and CPU
time (log10 s). Rows: graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free} graphs with kd expected
edges. Columns: noise types of SEM. Error bars represent standard errors over 5 simulations. Method: PPO-
NOTEARS-1-p represents our method with the ordering of a single chain with size m = p%d.

21


---Page Break---
DAGMA-Linear-SEM

PPO-DAGMA-1-50

PPO-DAGMA-1-75

PPO-DAGMA-1-100

Method

Figure 5: Results of single-chained ordering for DAGMA. Structural discovery in terms of SHD (lower is
better), F1-score (higher is better), FDR (lower is better), TPR (higher is better) and CPU time (log10 s). Rows:
graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free} graphs with kd expected edges. Columns: noise
types of SEM. Error bars represent standard errors over 5 simulations. Method: PPO-DAGMA-1-p represents
our method with the ordering of a single chain with size m = p%d.

22


---Page Break---
NOTEARS-Linear-SEM

PPO-NOTEARS-1-50

PPO-NOTEARS-2-50

PPO-NOTEARS-3-50

Method

PPO-NOTEARS-5-50

PPO-NOTEARS-10-50

NOTEARS-Linear-SEM

PPO-NOTEARS-1-50

PPO-NOTEARS-2-50

PPO-NOTEARS-3-50

Method

PPO-NOTEARS-5-50

PPO-NOTEARS-10-50

d=20
d=30
d=50
CPU Time
d=20
d=30
d=50
CPU Time

Figure 6: Results of multi-chained orderings for NOTEARS with linear SEM. Structural discovery in terms
of SHD (lower is better), F1-score (higher is better), FDR (lower is better), TPR (higher is better) and CPU
time (log10 s). Rows: graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free} graphs with kd expected
edges. Columns: noise types of SEM. Error bars represent standard errors over 5 simulations. Method: PPO-
NOTEARS-l-50 represents our method using the ordering where the chain number is l and the chain size is
m = 0.5d.

23


---Page Break---
NOTEARS-MLP
PPO-NOTEARS-1-50
PPO-NOTEARS-1-75
PPO-NOTEARS-1-100
Method

Figure 7: Results of single-chained Ordering for NOTEARS-MLP. Structural discovery in terms of SHD (lower
is better), F1-score (higher is better), FDR (lower is better), TPR (higher is better) and CPU time (log10 s).
Rows: graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free} graphs with kd expected edges. Error
bars represent standard errors over 5 simulations. Method: PPO-NOTEARS-1-p represents our method with
the ordering of a single chain with size m = p%d.

NOTEARS-MLP
PPO-NOTEARS-1-50
PPO-NOTEARS-2-50

PPO-NOTEARS-3-50
Method

PPO-NOTEARS-5-50
PPO-NOTEARS-10-50

d=20
d=30
d=50
CPU Time
d=20
d=30
d=50
CPU Time

Figure 8: Results of multi-chained orderings for NOTEARS-MLP. Structural discovery in terms of SHD (lower
is better), F1-score (higher is better), FDR (lower is better), TPR (higher is better) and CPU time (log10 s).
Rows: graph types. {ER,SF}-k represents {Erdös-Rényi, scale-free} graphs with kd expected edges. Error
bars represent standard errors over 5 simulations. Method: PPO-NOTEARS-l-50 represents our method using
the ordering where the chain number is l and the chain size is m = 0.5d.

24


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: The main claims made in the abstract and introduction are demonstrated both
theoretically and practically.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reﬂect how
much the results can be expected to generalize to other settings.
• It is ﬁne to include aspirational goals as motivation as long as it is clear that these
goals are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justiﬁcation: The paper lists the limitations of the proposed method in the Appendix.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-speciﬁcation, asymptotic approximations only holding locally). The au-
thors should reﬂect on how these assumptions might be violated in practice and what
the implications would be.
• The authors should reﬂect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reﬂect on the factors that inﬂuence the performance of the ap-
proach. For example, a facial recognition algorithm may perform poorly when image
resolution is low or images are taken in low lighting. Or a speech-to-text system might
not be used reliably to provide closed captions for online lectures because it fails to
handle technical jargon.
• The authors should discuss the computational efﬁciency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to ad-
dress problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be speciﬁcally instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

25


---Page Break---
Answer: [Yes]

Justiﬁcation: The full set of assumptions is provided and the proof of all theoretical results
under these assumptions are provided in Appendix.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theo-
rems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a
short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be comple-
mented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclu-
sions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justiﬁcation: All the critical algorithms of implementation of the proposed method is pro-
vided, and all the critical parameter settings are provided in Appendix.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps
taken to make their results reproducible or veriﬁable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture
fully might sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation,
it may be necessary to either make it possible for others to replicate the model with
the same dataset, or provide access to the model. In general. releasing code and data
is often one good way to accomplish this, but reproducibility can also be provided via
detailed instructions for how to replicate the results, access to a hosted model (e.g., in
the case of a large language model), releasing of a model checkpoint, or other means
that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all sub-
missions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear
how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to re-
produce the model (e.g., with an open-source dataset or instructions for how to
construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case au-
thors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

26


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justiﬁcation: All the codes of implementations and evaluations are provided in the supple-
mentary ﬁle.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so No is an acceptable answer. Papers cannot be rejected simply for not
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
Justiﬁcation: All the detailed hyperparameters are provided in the Appendix.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of
detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropri-
ate information about the statistical signiﬁcance of the experiments?
Answer: [Yes]
Justiﬁcation: Multiple simulations are made and the error bars are reported in the result.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

27


---Page Break---
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
Normality of errors is not veriﬁed.
• For asymmetric distributions, the authors should be careful not to show in tables or
ﬁgures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding ﬁgures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justiﬁcation: Sufﬁcient information on the computer resources is provided in the experi-
ment section, including CPU, GPU and memory.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments
that didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justiﬁcation: No ethics issues are with this paper.

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

Justiﬁcation: A relevant discussion is provided in Appendix.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

28


---Page Break---
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact spe-
ciﬁc groups), privacy considerations, and security considerations.
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
• If there are negative societal impacts, the authors could also discuss possible mitiga-
tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efﬁciency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justiﬁcation: This paper poses no such risks.
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by re-
quiring that users adhere to usage guidelines or restrictions to access the model or
implementing safety ﬁlters.
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
Justiﬁcation: All the datasets and algorithms used in this paper are correctly cited.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the pack-
age should be provided. For popular datasets, paperswithcode.com/datasets has cu-
rated licenses for some datasets. Their licensing guide can help determine the license
of a dataset.

29


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documenta-
tion provided alongside the assets?
Answer: [NA]
Justiﬁcation: This paper does not release new assets.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can
either create an anonymized URL or include an anonymized zip ﬁle.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the pa-
per include the full text of instructions given to participants and screenshots, if applicable,
as well as details about compensation (if any)?
Answer: [NA]
Justiﬁcation: The answer NA means that the paper does not involve crowdsourcing nor
research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Including this information in the supplemental material is ﬁne, but if the main contri-
bution of the paper involves human subjects, then as much detail as possible should
be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
tion, or other labor should be paid at least the minimum wage in the country of the
data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justiﬁcation: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equiva-
lent) may be required for any human subjects research. If you obtained IRB approval,
you should clearly state this in the paper.
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity
(if applicable), such as the institution conducting the review.

30


---Page Break---
