Markov Equivalence and Consistency in
Differentiable Structure Learning

Chang Deng†∗
Kevin Bello†,‡
Pradeep Ravikumar‡
Bryon Aragam†

†Booth School of Business, University of Chicago, Chicago, IL 60637
‡Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA 15213

Abstract

Existing approaches to differentiable structure learning of directed acyclic graphs
(DAGs) rely on strong identifiability assumptions in order to guarantee that global
minimizers of the acyclicity-constrained optimization problem identifies the true
DAG. Moreover, it has been observed empirically that the optimizer may exploit
undesirable artifacts in the loss function. We explain and remedy these issues
by studying the behavior of differentiable acyclicity-constrained programs under
general likelihoods with multiple global minimizers. By carefully regularizing the
likelihood, it is possible to identify the sparsest model in the Markov equivalence
class, even in the absence of an identifiable parametrization. We first study the
Gaussian case in detail, showing how proper regularization of the likelihood defines
a score that identifies the sparsest model. Assuming faithfulness, it also recovers
the Markov equivalence class. These results are then generalized to general models
and likelihoods, where the same claims hold. These theoretical results are validated
empirically, showing how this can be done using standard gradient-based optimizers
(without resorting to approximations such as Gumbel-Softmax), thus paving the
way for differentiable structure learning under general models and losses.

1
Introduction

Directed acyclic graphs (DAGs) are the most common graphical representation for causal models
[48, 60, 50], where nodes represent variables and directed edges represent cause-effect relationships
among variables. We are interested in the problem of structure learning, i.e. learning DAGs from
passively observed data, also known as causal discovery. Our focus will mainly be on score-
based approaches to DAG learning [11, 23], where the structure learning problem is formulated as
optimizing a given score or loss function s(B; X) that measures how well the graph, represented
as an adjacency matrix B ∈{0, 1}p×p, fits the observed data X, constrained to the graphical
structure B being acyclic. This combinatorial optimization problem is generally known to be
NP-complete [10, 12].

Recent advances in score-based methods have introduced a continuous representation of DAGs,
transforming the combinatorial acyclicity constraint into a continuous constraint via a differentiable
function that exactly characterizes DAGs [73]. In this case, the discrete adjacency matrix B ∈
{0, 1}p×p is first relaxed to the space of real matrices, i.e., B ∈Rp×p, and then a differentiable
function h : Rp×p →[0, ∞) is devised so that h(B) = 0 if and only if B is a DAG [73, 4]. This
results in the following optimization problem:
min
B∈Rp×p s(B; X)
subject to
h(B) = 0.
(1)

Considering a differentiable score function s, the differentiable program (1) facilitates the use of
gradient-based optimization techniques along with the use of richer models, such as neural networks,

∗changdeng@chicagobooth.edu

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
for modeling the functional relationships among the variables [74, 67, 41, 29, 44, 28, 76]. One of the
most attractive features of this approach is that it applies to general models, losses, and optimizers, in
contrast to prior work. Moreover, it cleanly separates computational and statistical concerns, so that
each can be studied in isolation, in the same spirit as the graphical lasso [36, 68, 17].

Looking back at the inception of the continuous DAG learning framework by Zheng et al. [73, 74],
however, most developments in this framework have focused on the design of alternative differentiable
acyclicity functions h with better numerical/computational properties [4, 32, 72, 67], placing little
emphasis on which score function to use [41]. In fact, and unfortunately, regardless of the modeling
assumptions, it has become a rather standard practice [67, 74, 4, 13, 32, 28] to simply use the least
squares (LS) loss (a.k.a. “reconstruction loss”) as the score by default, following the original paper
by Zheng et al. [73], despite its known statistical limitations [63, 33, 1].

As a result, Reisach et al. [55] flagged the empirical successes of continuous structure learning
(CSL) methods as largely due to the high agreement between the order of marginal variances of
the nodes and the topological order of the underlying simulated DAGs, a concept they describe
as “varsortability”. Then, Reisach et al. [55] empirically showed that the performance in structure
recovery of CSL methods drops significantly after simple data standardization. More recently, Ng
et al. [42] demonstrated that this phenomenon may not be explained by varsortability, and instead
pointed out that the explanations are due to the score function, albeit without proposing which score
function to use. These observations motivate a deeper consideration of the choice of score.

Unfortunately, despite the fact that several score functions have been proposed for learning Bayesian
networks (such as BIC [23], BDeu [35], and MDL [6]), their application to CSL methods is not well
understood. This paper is precisely concerned with finding a suitable and general score function
with strong statistical properties for CSL methods. That is, our objective is to find a score function
that is: (1) intrinsically differentiable so that it is amenable to gradient-based optimization without
approximations; (2) applicable to general models; (3) scale-invariant; (4) capable of identifying the
sparsest model under proper regularization; and (5) connects nicely with classical concepts from
Bayesian networks such as faithfulness and Markov equivalence classes.

Contributions. The main contribution of our work is to show that a properly regularized, likelihood-
based score function has the five properties outlined above. We begin with Gaussian models to convey
the main ideas, and then discuss generalizations. In more detail:

1. (Section 4) Starting with Gaussian models, we show that using the log-likelihood with a
quasi-MCP penalty (10) as the scoring function leads to optimal solutions of (1) that corre-
spond to the sparsest DAG structure which is Markov to P(X) (Theorem 1). Furthermore,
under the faithfulness assumption, all optimal solutions are the sparsest within the same
Markov equivalence class (Theorem 2).
2. (Section 5) We provide general conditions on the log-likelihood under which similar results
hold for general models (Theorem 4).
3. (Section 4.5) We show that for Gaussian models, the log-likelihood score is scale-invariant.
This means that rescaling or standardizing the data does not change the DAG structure
(Theorem 3), and hence is not susceptible to varsortability.
4. We conduct experiments in multiple settings to evaluate the advantages of using a likelihood-
based scoring method. The findings from these experiments are detailed in in Section 6 and
D. The empirical results support our theoretical claims: The likelihood-based score is robust
and scale invariant.

2
Related work

Most methods for learning DAGs fall into two primary categories: Constraint-based algorithms, which
depend on tests of conditional independence, and score-based algorithms, which aim to optimize a
specific score or loss function. As our focus is on score-based methods, we only briefly mention
classical constraint-based methods [59, 34, 62]. Within the umbrella of score-based methods, the
linear Gaussian models is covered in works such as [1, 2, 19, 20, 37, 49], while studies on linear
non-Gaussian SEMs are found in [33, 57]. Regarding nonlinear SEMs, significant contributions
have been made in additive models [9, 15, 64], additive noise models [24, 49, 39], generalized linear
models [47, 46, 22], and broader nonlinear SEMs [38, 21].

2


---Page Break---
Works that are more directly connected to our research include those developed in the continuous
structure learning (CSL) framework [e.g. 73, 74, 13, 4, 14, 29, 76, 41, 40, 28, 44]. Most of these
papers focus on empirical and computational aspects, and only a few study the theoretical properties
of the CSL framework in (1). These include: [65, 43] studied the optimization and convergence
subtleties of problem (1); [13] studied optimality guarantees for more general types of score functions
and proposed a bi-level optimization method to guarantee local minima; [14] designed an optimization
scheme that converges to the global minimum of the least squares score in the bivariate case. Finally,
among the few works that study score functions under this framework, we note: [41] studied the
properties of the ℓ1-regularized profile log-likelihood, which leads to quasi-equivalent models to the
ground-truth DAG; and the authors in [56] claim that a family of likelihood-based scores reduce
to the least square loss, although this only holds under knowledge of the noise variances [33].
Perhaps most closely related to our work is [8], who proved a similar identifiability result under
the likelihood score. However, they used an ℓ0 regularizer along with the faithfulness assumption,
which leads to an inherently non-differentiable optimization problem that is much simpler to analyze
but requires approximations (e.g. Gumbel-Softmax) to optimize. On the other hand, they also
consider interventional data, which we do not pursue in this work. Extending our results to include
interventional data and interventional Markov equivalence is an important direction for future work.
In contrast to the aforementioned works, we also prove that the log-likelihood has desirable properties
such as being scale invariant, and when regularized by nonconvex and differentiable approximations
of the ℓ0 function, it provably leads to useful solutions that are minimal models and Markov equivalent
to the underlying structure, without assuming faithfulness.

3
Preliminaries

We let G = (V, E) denote a directed graph on p nodes, with vertex set V = [p] := {1, . . . , p} and
edge set E ⊂V × V , where (i, j) ∈E indicates the presence of a directed edge from node i to node
j. We associate each node i ∈V to a random variable Xi, and let X = (X1, . . . , Xp).

Structural equation models (SEMs).
An SEM (X, f, P(N)) over the random vector X =
(X1, . . . , Xp) is a collection of p structural equations of the form:
Xj = fj(X, Nj),
∂kfj = 0 if k /∈PAj,
(2)

where f = (fj)p
j=1 is a collection of functions fj : Rp+1 →R, here N = (N1, . . . , Np) is a vector
of independent noises with distribution P(N), and PAj denotes the set of parents of node j. Here,
∂kfj denotes the partial derivative of fj w.r.t. Xk, which is identically zero when fj is independent
of Xk, i.e. fj(X, Nj) = fj(PAj, Nj). The graphical structure induced by the SEM, assumed to be a
DAG, will be represented by the following p × p weighted adjacency matrix B:
B = B(f),
Bij = ∥∂ifj∥2,
(3)
and we use G(B) to denote the corresponding binary adjacency matrix. For any set B of SEMs, let
G(B) := {G(B(f)) : (X, f, P(N)) ∈B},
(4)
i.e. G(B) is the collection of all the DAGs implied by B. If D is a set of DAGs and B is a set of SEM,
we also abuse notation by writing D = B to indicate D = G(B).

The SEM (2) is general enough to include many well-known models, such as linear SEMs [e.g.,
33, 49], generalized linear models [47, 45, 18], and additive noise models [24, 51], post-nonlinear
models [70, 71] and general nonlinear SEM [38, 21, 26, 74]. To illustrate some of these models:
In linear SEMs we have Xj = fj(PAj) + Nj, where fj is a linear map; in causal additive models
(CAM) we have Xj = P

k∈PAj fj,k(Xk) + Nj, where fj,k is a univariate function; in post-nonlinear
models we have Xj = fj,1(fj,2(PAj) + Nj). In fact, essentially any distribution can be represented
as an SCM of the form (2); see Proposition 7.1 in Peters et al. [50].

Faithfulness and sparsest representations. It is well-known that the DAG G is not always identifi-
able from X, and there is a well-developed theory on what can be identified based on X under certain
assumptions. This leads to the concepts of faithfulness and sparsest representations, which we briefly
recall here; we refer the reader to [60, 48, 50] for details. Let I(P) denote the set of conditional
independence relations implioed by the distribution P, and let I(G) denote the set of d-separations
implied by the graph G. Then P is Markov to G if I(G) ⊂I(P), and faithful to G if I(P) ⊂I(G).
When both conditions hold, i.e. I(P) = I(G), then G is called a perfect map of P. Following
common convention, we will simply call P faithful when I(G) = I(P). When P is faithful to G,
the Markov equivalence class (MEC) of G is identifiable and can be represented by a CPDAG.

3


---Page Break---
Definition 1. For any DAG G, the Markov equivalence class is M(G) = { eG : I( eG) = I(G)}

Since faithfulness may not always hold, there has been progress in understanding what can be identi-
fied under weaker conditions. One approach which we will use is the notion of a sparsest (Markov)
representation (SMR), introduced in [54]. A sparsest representation of P is a Markovian DAG G
that has strictly fewer edges than any other Markovian DAG G′, and such sparsest representation is
unique up to Markov equivalence class. Theorem 2.4 in [54] shows that if P is faithful to G, then G
must be a sparsest representation of P. This notion is closely related to the notion of minimality we
adopt in Definition 2 (cf. Lemma 4 in the Appendix). These ideas can be generalized and weakened
even further; see [31, 30] for details.

Parameters and the negative log-likelihood (NLL). For positive integers m, s, we will use ψ ∈
Ψ ⊆Rm and ξ ∈Ξ ⊆Rs to denote the model parameters for f = (f1, . . . , fp) and N, respectively.2
Then we denote the distribution of X by P(X; ψ, ξ). Let x ∈Rp denote one observation of X.
Given n i.i.d. samples X = (x1, . . . , xn)⊤where xi ∼P(X; ψ, ξ), the negative log-likelihood and
expected negative log-likelihood can be written as:

ℓn(ψ, ξ) = −1

n

n
X

i=1
log P(xi; ψ, ξ),
ℓ(ψ, ξ) = −E[log P(x; ψ, ξ)],
(5)

where the subscript n in ℓn is used to indicate the sample version of the log-likelihood.

Identifiability. Let ψ0 (resp. ξ0) denote the model parameters for the ground truth f 0 (resp. N 0), let
B0 = B0(ψ0) ∈Rp×p denote the induced weighted adjacency matrix, and let G(B0) ∈{0, 1}p×p
denote the induced binary adjacency matrix. For example, in the general linear Gaussian model (6),
ψ = B represents the adjacency matrix, and ξ = Ωdenotes the variance of the Gaussian noise. In
another case, if fj is approximated by a multilayer perceptron (MLP), with Nj as Gaussian noise,
then ψ includes all the parameters of the MLP, while ξ represents the variance of the Gaussian noise.
Additionally, (B)ij = [B(ψ)]ij = ∥i-th column of A(1)
j ∥, where A(1)
j
is the first hidden layer in fj
[74]. Thus, by our definitions, P(X; ψ0, ξ0) is the true distribution. Here, there are two types of
identifiability questions:

1. Parameter identifiability: Is it possible to uniquely determine the parameters (ψ0, ξ0) based
on observations from P(X; ψ0, ξ0)? Formally, is there any ( eψ, eξ) ̸= (ψ0, ξ0), such that
P(X, ψ0, ξ0) = P(X, eψ, eξ) almost surely?
2. Structural identifiability: Is it possible to uniquely determine the DAG G(B0) based on
observations from P(X; ψ0, ξ0)? In other words, is there any ( eψ, eξ) ̸= (ψ0, ξ0) such that
P(X, ψ0, ξ0) = P(X, eψ, eξ) but G(B0) ̸= G(B( eψ)).

In general, parameter identifiability implies structural identifiability since the ability to uniquely
determine parameter values often means that the structure they induce is also identifiable. However,
the converse is not generally true, i.e. structural identifiability does not always imply parameter
identifiability, as different parameter values can lead to the same structure. Classical results on
identifiability of SEMs include: linear SEM with equal variance [33], linear SEM with non-Gaussian
noises [57, 58], causal additive models with Gaussian noises [9], additive models with continuous
noise [51], and post-nonlinear models [70, 71, 25].

In models where parameter identifiability is possible, the population NLL ℓ(ψ, ξ) serves as a natural
choice for the score function because it attains a unique minimum at the true parameters (ψ0, ξ0).
However, this approach is not straightforward for nonidentifiable models, where multiple parameter
sets can induce the same data distribution P(X; ψ0, ξ0), leading to ambiguities in parameter or
structure estimation. In such cases, regularizing the log-likelihood can alleviate this issue. These
regularizers enforce specific characteristics like sparsity, guiding the model towards more meaningful
solutions (e.g. faithful or sparsest), despite the lack of identifiability.

4
General linear Gaussian SEMs: A nonidentifiable model
Although our results apply to general models, we begin by outlining the main idea with one of the
simplest nonidentifiable models, the Gaussian model. Our goal in this section is to theoretically show

2Given that ψ describes all the parameters for the functions f, we will also use B(ψ) to denote B(f) in (3).

4


---Page Break---
how the NLL with nonconvex differentiable regularizers can lead to meaningful solutions such as
minimal-edge models and elements of Markov equivalent classes. We also discuss and prove the
scale invariance of NLL, making it amenable to CSL approaches and addressing concerns raised in
previous work [55, 42]. Then, in Section 5, we extend these results to general models.

4.1
Gaussian DAG models

A linear SEM (B, Ω) over X with independent Gaussian noises N, a special case of (2), is well-
known to be nonidentifiable in terms of parameters and structure [see 2, for discussion]. We write the
model as follows:
X = B⊤X + N,
(6)

where B ∈Rp×p is a matrix of coefficients with G(B) being a DAG, and N ∈Rp is the vector of
independent noises with covariance matrix Ω= diag(ω2
1, . . . , ω2
p).3

Given the model (6) it is easy to see that the distribution P(X) is Gaussian and is fully characterized
by the pair (B, Ω). That is:

X ∼N(0, Σ),
Σ = Σf(B, Ω) := (I −B)−⊤Ω(I −B)−1,
(7)

where Σ is the covariance matrix of X. In the sequel, we use the subscript f to refer to a function. In
this case, Σf denotes a function with arguments (B, Ω) and returns the covariance matrix. Moreover,
we use Θ to denote the corresponding precision matrix (inverse of the covariance matrix):

Θ = Θf(B, Ω) := (I −B)Ω−1(I −B)⊤.
(8)

Let X = (x1, . . . , xn)⊤be n i.i.d. samples of X. Then, let the sample covariance matrix be
bΣ = 1

n
Pn
i=1 xix⊤
i . The sample NLL function is given by:

ℓn(B, Ω) = −1

n log

n
Y

i=1
P(xi; B, Ω) = 1

2 log det Ω−log det(I −B) + 1

2 Tr(bΣΘ(B, Ω)) + const.

The corresponding population NLL function is

ℓ(B, Ω) = −EX log P(X; B, Ω) = 1

2 log det Ω−log det(I −B) + 1

2 Tr(ΣΘ(B, Ω)) + const.

The full derivation can be found in Appendix C.1. Here, it is important to note that the distribution of
X is fully determined by either the precision matrix Θ or the covariance matrix Σ.

4.2
Equivalence and nonidentifiability

Our goal is to identify (B, Ω): Unfortunately, the model is inherently nonidentifiable in terms of both
parameter and structure. This means that multiple pairs (B, Ω) for model (6) can induce the same
data distribution P(X) given in (7), thus resulting also in the same precision matrix Θ. To address
this, we define the equivalence class E(Θ) as the set of all pairs (B, Ω) such that Θf(B, Ω) = Θ.
E(Θ) := {(B, Ω) : Θf(B, Ω) = Θ}.
(9)
It is worth noting that the size of E(Θ) is finite and at most p!, which corresponds to the number of
permutations for p variables [2]. For more comprehensive details on this class, see Appendix C.2.

This ambiguity naturally leads to the question: which pair (B, Ω) should we estimate? Since any pair
would be indistinguishable based only on observational data, a natural objective is to estimate the
“simplest” DAG, for example, a DAG that induces the precision matrix Θ with the smallest number of
edges. In other words, our goal is to estimate the matrix B that has the minimal number of nonzero
entries in the equivalence class. Let sB = |{(i, j) : Bij ̸= 0}|.
Definition 2 (Minimality). (B, Ω) is called a minimal-edge I-map4 in the equivalence class E(Θ)
if sB ≤s e
B, ∀( eB, eΩ) ∈E(Θ). The set of all minimal-edge I-maps in the equivalence class E(Θ) is
referred to as the minimal equivalence class Emin(Θ):
Emin(Θ) = {(B, Ω) : (B, Ω) is minimal-edge I-map, (B, Ω) ∈E(Θ)}.

3In terms of notation given in Section 3, we have parameters ψ = B and ξ = (ω2
1, . . . , ω2
p), and parameter
spaces Ψ = Rp×p, Ξ = Rp
>0.
4This generalizes the classical definition for DAGs [e.g. 63] to refer to the entire model with the distribution
and graph encoded by the matrix B and the error variance Ω.

5


---Page Break---
In the sequel, for brevity, we will often refer to such models as “minimal models”.

Unlike faithfulness, which may not always hold, the minimal equivalence class Emin(Θ) is always
well-defined. Moreover, as detailed in Lemma 4 in the Appendix, Definition 2 is closely related
to the SMR assumption [54]: Under the SMR assumption (and hence also faithfulness) for G, we
have M(G) = Emin(Θ), i.e., Emin(Θ) is the Markov equivalence class of G. However, there could
be multiple pairs (B, Ω) within Emin(Θ). Nevertheless, our goal is to recover one element from
Emin(Θ). The elements in Emin(Θ) not only represent the “simplest” DAG model for X in terms of
edge count, but also bear a deep connection to classical notions such as Markov equivalence. For
example, under faithfulness, all these elements describe the same independence statements.
Lemma 1. Let X follow model (6) with (B0, Ω0) and Θ0 = Θf(B0, Ω0). Assume that P(X) is
faithful to G0 := G(B0). Then M(G0) = Emin(Θ0).

Recall our convention that this means that M(G0) = G(Emin(Θ0)), i.e. the DAG structures contained
in Emin(Θ0) coincide with M(G0). Thus, under the faithfulness assumption, recovering Emin(Θ0)
is the same as recovering the MEC, which is the usual goal in causal discovery. Moreover, we
emphasize that these apply generally: For non-Gaussian X following the model specified in (2), the
same conclusion can be made; see Lemma 3 in the Appendix. Finally, we note that the commonly
used LS loss does not have the same minimizers as the log-likelihood when the noise variances are
different; see Appendix C.3.

4.3
Regularization

In order to distinguish elements in E(Θ) from the minimal elements in Emin(Θ), we need to somehow
account for the number of edges when evaluating the score function. The common approach to this
is to use BIC, or equivalently the ℓ0 penalty. Although both approaches effectively penalize the
number of nonzero entries in B, their non-differentiability makes them unsuitable for differentiable
structure learning. The ℓ1 penalty, while amenable to differentiable approaches,5 is not effective in
precisely counting the number of edges, and also biased in parameter estimation6. To mitigate these
shortcomings alternatives such as the smoothly clipped absolute deviation (SCAD) penalty [16] and
the minimax concave penalty (MCP) [69] have been proposed. We choose to use a reparametrized
version of MCP, termed quasi-MCP, defined as follows:

quasi-MCP:
pλ,δ(t) = λ[(|t| −t2

2δ )1(|t| < δ) + δ

21(|t| > δ)]
(10)

Here, 1(·) is the indicator function; corresponding plot can be found in Appendix C.5. Similar to
MCP, quasi-MCP is a symmetric function that takes on a quadratic form between [0, δ] and remains
constant for values greater than δ. The function is smooth, and for values |t| > δ, it approximates the
behavior of the ℓ0 penalty, thus serving to penalize the number of non-zero coefficients in B.

The score function in (1) can be naturally written as

s(B, Ω; λ, δ, X) = ℓn(B, Ω) + pλ,δ(B)
(11)

where pλ,δ(B) = P

i̸=j pλ,δ(Bij). Then, the optimization problem can be written as

min
B,Ωs(B, Ω; λ, δ, X)
subject to
h(B) = 0, Ω> 0.
(12)

It is worth noting that for any B, the corresponding optimal Ωthat minimizes s(B, Ω; λ, δ, X) can
be easily be expressed in terms of B as Ωf(B) (see Appendix C.1). Therefore, we can always plug
Ωf(B) into (12) to profile out Ω.

4.4
Provably recovering minimal models

Solving problem (12) requires minimizing ℓn(B, Ω) and pλ,δ simultaneously. To study the behavior
of these minimizers, let us define the set of global minimizers,

On,λ,δ = {(B∗, Ω∗) : (B∗, Ω∗) is a minimizer of (12)}.
(13)

5Although ℓ1 is nonsmooth, standard smoothing techniques can be applied to ℓ1 regularizers as in [73, 4].
6See Appendix C.4 for examples.

6


---Page Break---
Ideally, we would like On,λ,δ = Emin(Θ0), however, it is unclear whether there exist values of λ and
δ such that any optimal solution (B∗, Ω∗) lies within Emin(Θ). The following theorem provides an
affirmative answer to this question. In the sequel, we say that a property S(x) holds for all sufficiently
small x > 0 if there is some fixed ϵ > 0 such that for every x ≤ϵ, the property S(x) holds.
Theorem 1. Let X follow model (6) with (B0, Ω0) and Θ0 = Θf(B0, Ω0). Let X be n i.i.d. samples
from P(X), and On,λ,δ be defined as in (13). Then, for all sufficiently small λ, δ > 0 (independent
of n), it holds that P(On,λ,δ = Emin(Θ0)) →1 as n →∞.

In other words, we can always guarantee that On,λ,δ = Emin(Θ0) by taking λ, δ sufficiently small,
which is easily accomplished in practice. In the following, we use the superscript 0 to denote ground
truth parameters. Additionally, we can assume that B0 always belongs to Emin(Θ0), ensuring that our
reference to the ground truth aligns with the simplest or minimal representation within the equivalence
class. Moreover, by Lemma 1, under the faithfulness assumption, Theorem 1 can be interpreted as
recovering the Markov equivalence class M(G0):
Theorem 2. Consider the setup in Theorem 1 and assume additionally that P(X) is faithful to
G0 := G(B0). Then, for all sufficiently small λ, δ > 0 (independent of n), it holds that P(On,λ,δ =
M(G0)) →1 as n →∞.

Theorem 2 indicates with properly chosen hyperparameters, the optimal solution from optimization
(12) will produce a graph that adheres to the same independence statements as G0. This implies that
the structure learned through the optimization process accurately reflect the underlying causal or
conditional independence structure of underlying data generating process.
Remark 1. Although we use quasi-MCP (mainly for its simplicity), it turns out MCP or SCAD can
also be used. See Corollary 1 in Appendix A for details.

4.5
Scale invariance and standardization

It is known that the LS loss is not scale-invariant, i.e. re-scaling the data (and in particular, standard-
izing it) can drastically change the structure [33], a fact which Reisach et al. [55] use to argue that
differentiable DAG learning with the LS Loss is also not scale-invariant. Here we show that by using
a different score—in this case the log-likelihood—fixes this and results in (provable) scale invariance.
Thus, the choice of score function is crucial if certain properties such as scale invariance are desired.
The following result restates the well-known fact that Gaussian DAGs are invariant to re-scaling (i.e.
re-scaling does not change the support for any (B, Ω) ∈E(Θ)) using our notation:
Lemma 2. Let X ∼N(0, Σ), suppose Σ is a positive definite covariance matrix and let Θ := Σ−1,
suppose D is a diagonal matrix with positive diagonal entries. Then G(E(Θ)) = G(E(DΘD)).

Lemma 2 has appealing consequences for standardization. Given raw data X, denote its standardized
version by Z (cf. Appendix C.6). Ideally, structure learning algorithms will output the same structure
whether X or Z is used as input, and Lemma 2 suggests that re-scaling X will not alter the structure
of the DAG that is recovered from optimizing (12). The following theorem formalizes this:
Theorem 3. Under the same setting as Theorem 1, the solutions to (12) are scale-invariant. That is,
for any n ≥0, let

On,λ,δ(X) ={(B∗, Ω∗) : (B∗, Ω∗) is a minimizer of (12) with data X},
On,λ,δ(Z) ={(B∗, Ω∗) : (B∗, Ω∗) is a minimizer of (12) with data Z},

where Z is the standardized version of X. Then, for all sufficiently small λ, δ ≥0 and all n, we have
G(On,λ,δ(X)) = G(On,λ,δ(Z)). Moreover, for all sufficiently small λ, δ > 0 we have

P

G(On,λ,δ(X)) = G(On,λ,δ(Z)) = G(Emin(Θf(B0, Ω0)))

→1
as n →∞.

Thus, even on finite samples, the set of DAG structures G(On,λ,δ(X)) derived from the raw (un-
standardized) data X will always be the same as G(On,λ,δ(Z)), which is derived from standardized
data Z. As a result, standardizing Gaussian data does not affect the recovered DAG structure if the
optimization problem (12) can be solved exactly.
Remark 2. Theorem 3 applies to global optimization of the objective (12). Of course, in practice,
algorithms can get stuck in local optima, but the global solutions (even for finite samples n) will
always be scale invariant.

7


---Page Break---
5
Nonconvex regularized log-likelihood for general models

The results in the previous section are not specific to Gaussian models, although this helps with
interpretability in a familiar setting. We now extend these results from linear Gaussian SEMs to more
general SEMs. Here, we assume that X follows model (2) and the induced distribution is denoted by
P(X; ψ0, ξ0). Let us define the equivalence class E(ψ0, ξ0),

E(ψ0, ξ0) = {(ψ, ξ) : P(x; ψ, ξ) = P(x; ψ0, ξ0), ∀x ∈Rp}.

That is, E(ψ0, ξ0) is a set of pairs (ψ, ξ) that induce the same distribution P(X; ψ0, ξ0). As a result,
any pair (ψ, ξ) within this equivalence class will be a minimizer of the NLL ℓ(ψ, ξ). Analogously to
Definition 2, we can also define the collection of minimal elements in the equivalence class E(ψ0, ξ0).

Definition 3. (ψ, ξ) is called a minimal-edge I-map in the equivalence class E(ψ0, ξ0) if sB(ψ) ≤
sB( e
ψ), ∀( eψ, eξ) ∈E(ψ0, ξ0). We further define

Emin(ψ0, ξ0) = {(ψ, ξ) : (ψ, ξ) is minimal-edge I-map, (ψ, ξ) ∈E(ψ0, ξ0)}.

Here, it is crucial that our concept of minimality concerns sB(ψ), which is the number of nonzero
entries in the weighted adjacency matrix B(ψ), rather than the number of nonzero entries in the
parameter ψ itself. Therefore, sB(ψ) essentially counts the number of edges in the adjacency matrix.

Assumption A. (1) |E(ψ0, ξ0)| is finite. (2) B(ψ) is L-Lipschitz w.r.t. ψ, i.e. ∥B(ψ1)−B(ψ2)∥2

∥ψ1−ψ2∥2
≤L.

Assumption B. For any α such that ℓ(ψ0, ξ0) < α, the level set {(ψ, ξ) : ℓ(ψ, ξ) ≤α} is bounded,
where ℓ(ψ, ξ) is the expected NLL defined in (5).

Assumption A(1) is relatively mild; it requires that the equivalence class contains only finitely many
points. This assumption is satisfied by Gaussian models, generalized linear models with continuous
output [66], binary output [74, 13], and most exponential families. It is also obviously satisfied by
any identifiable model since |E(ψ0, ξ0)| = 1. Assumption A(2) is a mild continuity requirement
on B(ψ). Assumption B simply guarantees that the optimization problem has a minimizer, and is
standard [7]. More discussions about the assumptions are included in Appendix C.7. Without this
type of assumption, score-based learning is not even well-defined.

Similar in spirit to Theorem 1, we can show that by combining the NLL with quasi-MCP for
appropriate λ, δ, solving the following problem, we recover elements of Emin(ψ0, ξ0):

min
ψ∈Ψ,ξ∈Ξ ℓn(ψ, ξ) + pλ,δ(B(ψ))
subject to
h(B(ψ)) = 0,
(14)

where pλ,δ(·) is quasi-MCP defined in (10). Next, define its set of global minimizers.

On,λ,δ = {(ψ∗, ξ∗) : (ψ∗, ξ∗) is minimizer of (14)}.

Theorem 4. Let X follow model (2) with parameters (ψ0, ξ0) and let X be n i.i.d. samples from
P(X; ψ0, ξ0). Under Assumptions A-B, for all sufficiently small λ, δ > 0 (independent of n), it holds
that P(On,λ,δ = Emin(ψ0, ξ0)) →1 as n →∞.

Theorem 5. Under the setting in Theorem 4 and assuming that P(X; ξ0, ψ0) is faithful with respect
to G0 := G(B(ψ0)). Then, for all sufficiently small λ, δ > 0 (independent of n), it holds that
P(On,λ,δ = M(G0)) →1 as n →∞.

6
Experiments

To solve (12) and (14), we employ the augmented Lagrangian algorithm [5] from NOTEARS [73, 74],
modifying their least squares score with ℓ1 penalty into the log-likelihood with MCP (10). We
compare our approach to relevant baselines, e.g. NOTEARS [73], GOLEM [41], DAGMA [4],
VarSort [55], FGES [52] and PC [59]. For our variation of NOTEARS that employs a score function
based on the NLL with MCP, we name it as LOGLL-NOTEARS. The suffixes ‘POPULATION’ and
‘SAMPLE’ denote the use of the population and sample covariance matrix, respectively. Full details of
the experiments are given in Appendix D.

8


---Page Break---
Our primary empirical results are shown in Figures 1 and 2. We use the structural Hamming distance
(SHD) as the main metric to evaluate the difference between the estimated graph and the ground truth
graph. Lower SHD values indicate better estimation accuracy. Given that the model specified in (6)
is nonidentifiable, we compare the CPDAGs of the estimated graph and the ground truth graph.

In Figure 1(a), we observe that using the NLL+MCP achieves the best performance for the different
types of graphs and ranks second best for sparse graphs {ER1, SF1}. In Figure 1(b), standardizing
X significantly impacts the performance of GOLEM, NOTEARS, and DAGMA; the SHD values
are not any better than an empty graph, exactly as predicted by prior theory. The performance of
LOGLL-NOTEARS-SAMPLE and LOGLL-NOTEARS-POPULATION are also affected by standardization,
but these methods remain robust and continue to make meaningful discoveries. It is important to
note that this observation does not contradict our Lemma 2. The challenges arise because solving
the optimization problems (12) and (14) to find global solutions becomes inherently difficult as p
increases.To verify the scale invariance property in Theorem 3, we also conduct experiments on small
graphs and include exact method that solve (12) and (14) to global optimal, see Figure 5.

In Figure 2, we replicate the Figure 1 in [55], providing a more direct comparison between various
methods applied to raw data (X) and standardized data (X standardized). We include VarSort
(referred to as sortnregress in [55]) as a baseline. Notably, for smaller graphs (p = 10), both LOGLL(-
NOTEARS)-SAMPLE and LOGLL(-NOTEARS)-POPULATION exhibit the scale-invariant property
alongside PC and FGES, in alignment with Lemma 2. This contrasts sharply with other methods,
which completely deteriorate. For larger graphs (p = 50), standardizing the data mildly degrades the
performance of LOGLL(-NOTEARS)-SAMPLE and LOGLL(-NOTEARS)-POPULATION. This can be
attributed to the increased complexity of optimization as the size of the graph grows.

In Figure 3, we use a concrete toy example to investigate two key factors in the implementation: (1)
the impact of random initialization, and (2) the upper limit for δ0 that can be applied according to
Theorem 1. We generate 105 initializations Brandom with weight for each edge uniformly sampled
within [−5, 5], and perform optimization using LOGLL-NOTEARS starting from these points. The
“maximal δ" is the theoretical maximum δ0 that ensures the validity of Theorem 1. We computed
the SHD and the distances between the estimated M(Best) and M(B0). The red line in Figure 3
represents the average SHD and distances. The distribution of these 105 estimated SHD and distances
is visualized using dots of varying sizes, where larger dots indicate a higher frequency of points. In
some cases where SHD takes a value of −1, this value is used to indicate that the estimated Best does
not form a valid DAG, which is an artifact of thresholding and affects < 0.5% of models. For the
remaining models, the optimization (12) can typically be solved very close to a globally optimal, and
according to Theorem 2, the SHD should ideally be zero, which is consistent with the figure.

Our results are not limited to the linear model with Gaussian noise. In Appendix E.3, we provide
additional experiments on a logistic model (binary Xj) and neural networks. Further details on the
experimental settings and additional experiments can be found in Appendix D and E.

7
Conclusion

Continuous score-based structure learning is a relative newcomer to the literature on causal structure
learning, which goes back several decades. It has attracted significant attention due to its simplicity
and generality, however, its theoretical properties are often misunderstood. We have sought to
fill in this gap by studying its statistical aspects (to complement ongoing computational studies,
e.g. [41, 65, 4, 13, 14, 43]). To this end, we proposed a fully differentiable score function for
structure learning, composed of log-likelihood and quasi-MCP. We demonstrated that the global
solution corresponds to the sparsest DAG structure that is Markov to the data distribution. Under
mild assumptions, we conclude that all optimal solutions are the sparsest within the same Markov
equivalence class. Additionally, the proposed score is scale-invariant, producing the same structure
regardless of the data scale under the linear Gaussian model. Experimental results validate our theory,
showing that our score provides better and more robust structure recovery compared to other scores.

We hope that this work stimulates further statistical inquiry into the properties of CSL. For example,
we have focused on parametric models, and left extensions to nonparametric models to future work.
Certain assumptions such as the finiteness of the equivalence class and the boundedness of the level
set of the log-likelihood become more interesting in this regime. We have mentioned already that
extensions to richer data types including interventions is an important direction. It would be of great

9


---Page Break---
interest to explore ways to relax our assumptions to expand our statistical understanding of CSL in
broader scenarios.

1

ER
SF

25
50
75
100

0

20

40

60

80

0

25

50

75

d (Number of nodes)

Structural Hamming Distance (SHD)

2

ER
SF

25
50
75
100

0

40

80

120

160

0

50

100

150

d (Number of nodes)

4

ER
SF

25
50
75
100

0

250

500

750

1000

0

100

200

300

400

d (Number of nodes)

method
DAGMA
FGES
GOLEM
LOGLL_NOTEARS_POPULATION
LOGLL_NOTEARS_SAMPLE
NOTEARS
PC
(a) X (generated by Equation (6))

1

ER
SF

25
50
75
100

0

50

100

0

50

100

150

d (Number of nodes)

Structural Hamming Distance (SHD)

2

ER
SF

25
50
75
100

0

100

200

300

400

0

100

200

300

d (Number of nodes)

4

ER
SF

25
50
75
100

0

300

600

900

0

200

400

600

d (Number of nodes)

method
DAGMA
FGES
GOLEM
LOGLL_NOTEARS_POPULATION
LOGLL_NOTEARS_SAMPLE
NOTEARS
PC
(b) standardization of X (generated by Equation (6))

Figure 1: Results in terms of SHD between MECs of estimated graph and ground truth. Lower is
better. Column: k = {1, 2, 4}. Row: random graph types. {ER,SF}-k = {Scale-Free,Erd˝os-Rényi }
graphs with kd expected edges. Here p = {10, 20, 50, 70, 100}, n = 1000.

0
5
10
15
20
25
30
35
Structural Hamming Distance (SHD)

LogLL_population

LogLL_sample

VarSort

DAGMA

GOLEM

NOTEARS

FGES

PC

Method

LogLL
Baseline
Countiuous
Combinatorial
raw
standardized

(a) p = 10, graph =“ER”, k = 2

0
25
50
75
100
125
150
175
200
Structural Hamming Distance (SHD)

LogLL_population

LogLL_sample

VarSort

DAGMA

GOLEM

NOTEARS

FGES

PC

Method

LogLL
Baseline
Countiuous
Combinatorial
raw
standardized

(b) p = 50, graph =“ER”, k = 2

Figure 2: Comparison of raw (orange) vs. standardized (green) data. SHD (lower is better) between
Markov equivalence classes (MEC) of recovered and ground truth graphs for ER-2 graphs with 10
(left) or 50 (right) nodes. In (b), SHD for VarSort with standardized data is omitted due to its average
exceeding 300.

0.0
0.2
0.4
0.6
0.8
1.0
1.0

0.5

0.0

0.5

1.0

1.5

2.0

SHD between MEC

Fork: 0->1,0->2

maximal 
shd

0.0
0.2
0.4
0.6
0.8
1.0

0

1

2

3

Distance

Fork: 0->1,0->2

maximal 
dist

20000

40000

60000

80000

100000

Number of Points

Figure 3: Graph: fork structure X0 →X1 and X0 →X2. For 0 < δ < δ0, the estimated
(Best, Ωest) ∈Emin(Θ0) because SHD and distance are close to 0.

10


---Page Break---
References

[1] Aragam, B., Amini, A. and Zhou, Q. [2019], ‘Globally optimal score-based learning of directed
acyclic graphs in high-dimensions’, Advances in Neural Information Processing Systems 32.

[2] Aragam, B. and Zhou, Q. [2015], ‘Concave penalized estimation of sparse Gaussian Bayesian
networks’, The Journal of Machine Learning Research 16(1), 2273–2328.

[3] Barabási, A.-L. and Albert, R. [1999], ‘Emergence of scaling in random networks’, science
286(5439), 509–512.

[4] Bello, K., Aragam, B. and Ravikumar, P. [2022], ‘Dagma: Learning dags via m-matrices and
a log-determinant acyclicity characterization’, Advances in Neural Information Processing
Systems 35, 8226–8239.

[5] Bertsekas, D. P. [1997], ‘Nonlinear programming’, Journal of the Operational Research Society
48(3), 334–334.

[6] Bouckaert, R. R. [1993], Probabilistic network construction using the minimum description
length principle, in ‘European conference on symbolic and quantitative approaches to reasoning
and uncertainty’, Springer, pp. 41–48.

[7] Boyd, S., Boyd, S. P. and Vandenberghe, L. [2004], Convex optimization, Cambridge university
press.

[8] Brouillard, P., Lachapelle, S., Lacoste, A., Lacoste-Julien, S. and Drouin, A. [2020], ‘Differen-
tiable causal discovery from interventional data’, Advances in Neural Information Processing
Systems 33, 21865–21877.

[9] Bühlmann, P., Peters, J. and Ernest, J. [2014], ‘Cam: Causal additive models, high-dimensional
order search and penalized regression’, The Annals of Statistics 42(6), 2526–2556.

[10] Chickering, D. M. [1996], Learning bayesian networks is np-complete, in ‘Learning from data’,
Springer, pp. 121–130.

[11] Chickering, D. M. [2003], ‘Optimal structure identification with greedy search’, JMLR
3, 507–554.

[12] Chickering, D. M., Heckerman, D. and Meek, C. [2004], ‘Large-sample learning of Bayesian
networks is NP-hard’, Journal of Machine Learning Research 5, 1287–1330.

[13] Deng, C., Bello, K., Aragam, B. and Ravikumar, P. K. [2023], Optimizing notears objectives via
topological swaps, in ‘International Conference on Machine Learning’, PMLR, pp. 7563–7595.

[14] Deng, C., Bello, K., Ravikumar, P. and Aragam, B. [2023], ‘Global optimality in bivariate
gradient-based dag learning’, Advances in Neural Information Processing Systems 36.

[15] Ernest, J., Rothenhäusler, D. and Bühlmann, P. [2016], ‘Causal inference in partially linear
structural equation models: identifiability and estimation’, arXiv preprint arXiv:1607.05980 .

[16] Fan, J. and Li, R. [2001], ‘Variable selection via nonconcave penalized likelihood and its oracle
properties’, Journal of the American statistical Association 96(456), 1348–1360.

[17] Friedman, J., Hastie, T. and Tibshirani, R. [2008], ‘Sparse inverse covariance estimation with
the graphical lasso’, Biostatistics 9(3), 432–441.

[18] Gao, M., Ding, Y. and Aragam, B. [2020], ‘A polynomial-time algorithm for learning nonpara-
metric causal graphs’, Advances in Neural Information Processing Systems 33, 11599–11611.

[19] Ghoshal, A. and Honorio, J. [2017], Learning identifiable gaussian bayesian networks in
polynomial time and sample complexity, in ‘Proceedings of the 31st International Conference
on Neural Information Processing Systems’, pp. 6460–6469.

11


---Page Break---
[20] Ghoshal, A. and Honorio, J. [2018], Learning linear structural equation models in polynomial
time and sample complexity, in ‘Proceedings of the Twenty-First International Conference on
Artificial Intelligence and Statistics’, Vol. 84 of Proceedings of Machine Learning Research,
PMLR, pp. 1466–1475.

[21] Goudet, O., Kalainathan, D., Caillou, P., Guyon, I., Lopez-Paz, D. and Sebag, M. [2018], ‘Learn-
ing functional causal models with generative neural networks’, Explainable and interpretable
models in computer vision and machine learning pp. 39–80.

[22] Gu, J., Fu, F. and Zhou, Q. [2019], ‘Penalized estimation of directed acyclic graphs from discrete
data’, Statistics and Computing 29(1), 161–176.

[23] Heckerman, D., Geiger, D. and Chickering, D. M. [1995], ‘Learning bayesian networks: The
combination of knowledge and statistical data’, Machine learning 20(3), 197–243.

[24] Hoyer, P., Janzing, D., Mooij, J. M., Peters, J. and Schölkopf, B. [2008], ‘Nonlinear causal
discovery with additive noise models’, Advances in neural information processing systems 21.

[25] Immer, A., Schultheiss, C., Vogt, J. E., Schölkopf, B., Bühlmann, P. and Marx, A. [2023],
On the identifiability and estimation of causal location-scale noise models, in ‘International
Conference on Machine Learning’, PMLR, pp. 14316–14332.

[26] Kalainathan, D., Goudet, O., Guyon, I., Lopez-Paz, D. and Sebag, M. [2022], ‘Structural
agnostic modeling: Adversarial learning of causal graphs’, Journal of Machine Learning
Research 23(219), 1–62.

[27] Kingma, D. P. and Ba, J. [2014], ‘Adam: A method for stochastic optimization’, arXiv preprint
arXiv:1412.6980 .

[28] Kyono, T., Zhang, Y. and van der Schaar, M. [2020], ‘Castle: Regularization via auxiliary causal
graph discovery’, Advances in Neural Information Processing Systems 33, 1501–1512.

[29] Lachapelle, S., Brouillard, P., Deleu, T. and Lacoste-Julien, S. [2020], Gradient-based neural
dag learning, in ‘International Conference on Learning Representations’.

[30] Lam, W. Y. [2023], Causal Razors and Causal Search Algorithms, PhD thesis, Carnegie Mellon
University.

[31] Lam, W.-Y., Andrews, B. and Ramsey, J. [2022], Greedy relaxations of the sparsest permutation
algorithm, in ‘Uncertainty in Artificial Intelligence’, PMLR, pp. 1052–1062.

[32] Lee, H.-C., Danieletto, M., Miotto, R., Cherng, S. T. and Dudley, J. T. [2019], Scaling structural
learning with no-bears to infer causal transcriptome networks, in ‘Pacific Symposium on
Biocomputing 2020’, World Scientific, pp. 391–402.

[33] Loh, P.-L. and Bühlmann, P. [2014], ‘High-dimensional learning of linear causal networks via
inverse covariance estimation’, The Journal of Machine Learning Research 15(1), 3065–3105.

[34] Margaritis, D. and Thrun, S. [1999], Bayesian network induction via local neighborhoods, in
‘Proceedings of the 12th International Conference on Neural Information Processing Systems’,
pp. 505–511.

[35] Maxwell Chickering, D. and Heckerman, D. [1997], ‘Efficient approximations for the marginal
likelihood of bayesian networks with hidden variables’, Machine learning 29(2), 181–212.

[36] Meinshausen, N. and Bühlmann, P. [2006], ‘High-dimensional graphs and variable selection
with the lasso’.

[37] Meinshausen, N. and Bühlmann, P. [2006], ‘High-dimensional graphs and variable selection
with the Lasso’, The Annals of Statistics 34(3).

[38] Monti, R. P., Zhang, K. and Hyvärinen, A. [2020], Causal discovery with general non-linear
relationships using non-linear ica, in ‘Uncertainty in artificial intelligence’, PMLR, pp. 186–195.

12


---Page Break---
[39] Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J. and Schölkopf, B. [2016], ‘Distinguishing
cause from effect using observational data: methods and benchmarks’, The Journal of Machine
Learning Research 17(1), 1103–1204.

[40] Moraffah, R., Moraffah, B., Karami, M., Raglin, A. and Liu, H. [2020], ‘Causal adversarial
network for learning conditional and interventional distributions’, arXiv:2008.11376 .

[41] Ng, I., Ghassami, A. and Zhang, K. [2020], ‘On the role of sparsity and dag constraints for
learning linear dags’, Advances in Neural Information Processing Systems 33, 17943–17954.

[42] Ng, I., Huang, B. and Zhang, K. [2024], Structure Learning with Continuous Optimization:
A Sober Look and Beyond, in ‘Proceedings of the Third Conference on Causal Learning and
Reasoning’, PMLR, pp. 71–105.

[43] Ng, I., Lachapelle, S., Ke, N. R., Lacoste-Julien, S. and Zhang, K. [2022], On the convergence
of continuous constrained optimization for structure learning, in ‘International Conference on
Artificial Intelligence and Statistics’, PMLR, pp. 8176–8198.

[44] Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Georgatzis, K., Beaumont, P.
and Aragam, B. [2020], Dynotears: Structure learning from time-series data, in ‘International
Conference on Artificial Intelligence and Statistics’, PMLR, pp. 1595–1605.

[45] Park, G. and Park, H. [2019a], Identifiability of generalized hypergeometric distribution (ghd) di-
rected acyclic graphical models, in ‘The 22nd International Conference on Artificial Intelligence
and Statistics’, PMLR, pp. 158–166.

[46] Park, G. and Park, S. [2019b], ‘High-dimensional poisson structural equation model learning
via \ell_1-regularized regression.’, J. Mach. Learn. Res. 20, 95–1.

[47] Park, G. and Raskutti, G. [2017], ‘Learning quadratic variance function (qvf) dag models via
overdispersion scoring (ods).’, J. Mach. Learn. Res. 18, 224–1.

[48] Pearl, J. [2009], Causality, Cambridge university press.

[49] Peters, J. and Bühlmann, P. [2014], ‘Identifiability of gaussian structural equation models with
equal error variances’, Biometrika 101(1), 219–228.

[50] Peters, J., Janzing, D. and Schölkopf, B. [2017], Elements of causal inference: foundations and
learning algorithms, MIT press.

[51] Peters, J., Mooij, J. M., Janzing, D. and Schölkopf, B. [2014], ‘Causal discovery with continuous
additive noise models’, JMLR .

[52] Ramsey, J., Glymour, M., Sanchez-Romero, R. and Glymour, C. [2017], ‘A million variables
and more: the fast greedy equivalence search algorithm for learning high-dimensional graphical
causal models, with an application to functional magnetic resonance images’, International
journal of data science and analytics 3, 121–129.

[53] Ramsey, J., Zhang, J. and Spirtes, P. L. [2012], ‘Adjacency-faithfulness and conservative causal
inference’, arXiv preprint arXiv:1206.6843 .

[54] Raskutti, G. and Uhler, C. [2018], ‘Learning directed acyclic graph models based on sparsest
permutations’, Stat 7(1), e183.

[55] Reisach, A., Seiler, C. and Weichwald, S. [2021], ‘Beware of the simulated dag! causal
discovery benchmarks may be easy to game’, Advances in Neural Information Processing
Systems 34, 27772–27784.

[56] Seng, J., Zeˇcevi´c, M., Dhami, D. S. and Kersting, K. [2023], Learning large dags is harder
than you think: Many losses are minimal for the wrong dag, in ‘The Twelfth International
Conference on Learning Representations’.

[57] Shimizu, S., Hoyer, P. O., Hyvärinen, A., Kerminen, A. and Jordan, M. [2006], ‘A linear
non-gaussian acyclic model for causal discovery.’, Journal of Machine Learning Research
7(10).

13


---Page Break---
[58] Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y., Washio, T., Hoyer,
P. O., Bollen, K. and Hoyer, P. [2011], ‘Directlingam: A direct method for learning a lin-
ear non-gaussian structural equation model’, Journal of Machine Learning Research-JMLR
12(Apr), 1225–1248.

[59] Spirtes, P. and Glymour, C. [1991], ‘An algorithm for fast recovery of sparse causal graphs’,
Social science computer review 9(1), 62–72.

[60] Spirtes, P., Glymour, C. N., Scheines, R. and Heckerman, D. [2000], Causation, prediction, and
search, MIT press.

[61] Tibshirani, R. [1996], ‘Regression shrinkage and selection via the lasso’, Journal of the Royal
Statistical Society Series B: Statistical Methodology 58(1), 267–288.

[62] Tsamardinos, I., Aliferis, C. F., Statnikov, A. R. and Statnikov, E. [2003], Algorithms for large
scale markov blanket discovery, in ‘FLAIRS conference’, Vol. 2, pp. 376–380.

[63] Van de Geer, S. and Bühlmann, P. [2013], ‘ℓ0-penalized maximum likelihood for sparse directed
acyclic graphs’, The Annals of Statistics 41(2), 536–567.

[64] Voorman, A., Shojaie, A. and Witten, D. [2014], ‘Graph estimation with joint additive models’,
Biometrika 101(1), 85–101.

[65] Wei, D., Gao, T. and Yu, Y. [2020], DAGs with no fears: A closer look at continuous optimization
for learning bayesian networks, in ‘Advances in Neural Information Processing Systems’.

[66] Ye, Q., Amini, A. A. and Zhou, Q. [2024], ‘Federated learning of generalized linear causal
networks’, IEEE Transactions on Pattern Analysis and Machine Intelligence .

[67] Yu, Y., Chen, J., Gao, T. and Yu, M. [2019], Dag-gnn: Dag structure learning with graph neural
networks, in ‘International Conference on Machine Learning’, PMLR, pp. 7154–7163.

[68] Yuan, M. and Lin, Y. [2007], ‘Model selection and estimation in the gaussian graphical model’,
Biometrika 94(1), 19–35.

[69] Zhang, C.-H. [2010], ‘Nearly unbiased variable selection under minimax concave penalty’, The
Annals of Statistics 38(2), 894 – 942.

[70] Zhang, K. and Hyvarinen, A. [2012], ‘On the identifiability of the post-nonlinear causal model’,
arXiv preprint arXiv:1205.2599 .

[71] Zhang, K., Wang, Z., Zhang, J. and Schölkopf, B. [2015], ‘On estimation of functional causal
models: general results and application to the post-nonlinear causal model’, ACM Transactions
on Intelligent Systems and Technology (TIST) 7(2), 1–22.

[72] Zhang, Z., Ng, I., Gong, D., Liu, Y., Abbasnejad, E., Gong, M., Zhang, K. and Shi, J. Q.
[2022], ‘Truncated matrix power iteration for differentiable dag learning’, Advances in Neural
Information Processing Systems 35, 18390–18402.

[73] Zheng, X., Aragam, B., Ravikumar, P. K. and Xing, E. P. [2018], ‘Dags with no tears: Continu-
ous optimization for structure learning’, Advances in neural information processing systems
31.

[74] Zheng, X., Dan, C., Aragam, B., Ravikumar, P. and Xing, E. [2020], Learning sparse non-
parametric dags, in ‘International Conference on Artificial Intelligence and Statistics’, Pmlr,
pp. 3414–3425.

[75] Zheng, Y., Huang, B., Chen, W., Ramsey, J., Gong, M., Cai, R., Shimizu, S., Spirtes, P. and
Zhang, K. [2024], ‘Causal-learn: Causal discovery in python’, Journal of Machine Learning
Research 25(60), 1–8.

[76] Zhu, S., Ng, I. and Chen, Z. [2020], Causal discovery with reinforcement learning, in ‘Interna-
tional Conference on Learning Representations’.

14


---Page Break---
SUPPLEMENTARY MATERIAL
Markov Equivalence and Consistency in
Differentiable Structure Learning

A
Preliminary Technical Results

In this appendix, we include various technical results used to prove the main theorems of the paper.
Proofs can be found in Appendix B.

The following corollary supports Remark 1. In the main paper, we use quasi-MCP (10) as a penalty
in the optimization problems (12) and (14) for simplicity. However, similar conclusions hold when
MCP or SCAD is used as the penalty term.
Corollary 1 (MCP/SCAD). Under the same setting as Theorem 1. Let optimal solutions collection
be

On,λ,a = {(B∗, Ω∗) : (B∗, Ω∗) is minimizer of (12) with pλ,δ(t) replaced by pMCP
λ,a
(t) or pSCAD
λ,a
(t)}

Then, for all sufficiently small λ, a > 0 (independent of n), it holds that On,λ,a = Emin(Θ0) as
n →∞, where MCP pMCP
λ,a
(·) and SCAD pSCAD
λ,a
(·) are defined in Appendix C.5.

The following lemma is a generalization of Lemma 1. Even for the general model, under the
faithfulness assumption, all elements in the minimal equivalence class Emin(ψ0, ξ0) belong to the
same Markov equivalence class, as is the case in the general linear Gaussian model (6).
Lemma 3. Consider that X is generated by (2) with (ψ0, ξ0). Assume that P(X; ξ0, ψ0) is faithful
to G0 := G(B(ψ0)). Then

M(G0) = G(Emin(ψ0, ξ0)) = {G(B(ψ)) : (ψ, ξ) ∈Emin(ψ0, ξ0)}

where B(ψ) is the adjacency matrix implied by the parameterization (ψ, ξ), see (3). M(G0) is the
Markov equivalence class of G0, see Definition 1.

Under the Sparsest Markov representation assumption, all elements in the minimal equivalence class
are also in the same Markov equivalence class. It is important to note that the faithfulness assumption
is stronger than the Sparsest Markov representation assumption. Specifically, if P is faithful with
respect to G, then the pair (G, P) satisfies the Sparsest Markov representation assumption.
Lemma 4. If a pair (G, P(X; B, Ω)) satisfies Sparsest Markov representation (SMR) (see Definition
4), then M(G) = G(Emin(Θ)) = {G(B) : (B, Ω) ∈Emin(Θ)} where M(G) is Markov equivalence
class of G (see Definition 1).

The following lemma provides the formulation for the standardization of X, along with its covariance
and precision matrices.
Lemma 5 (standardization). Let X ∼N(0, Σ), σ2
i := Var(Xi) and D := diag(σ1, . . . , σp). Then
the standardization of X, corresponding covariance matrix and precision matrix can be expressed as

Xstd := D−1(X −EX),
Cov(Xstd) = D−1ΣD−1
[Cov(Xstd)]−1 = DΘD

The following lemma establishes an useful identity that holds for any adjacency matrix of a DAG,
which is used in the derivation of the log-likelihood function for the model in Equation (6).
Lemma 6. If B is adjacency matrix of a DAG, then log det(I −B) = 0.

The following lemma provides a condition under which the optimization problem (12) is well-defined,
ensuring that ℓ(B, Ω) > −∞for any (B, Ω).
Lemma 7. For any (B, Ω), if Ω> 0, then Σ := Σf(B, Ω) is positive definite. Moreover, if X is
generated by Equation (6) with (B0, Ω0), then ℓ(B, Ω) > −∞for any (B, Ω).


---Page Break---
The following lemma is used in the proof of Theorem 1. It justifies that the loss of every element in
A3 is strictly greater than the loss of the ground truth, i.e., (B0, Ω0).
Lemma 8. Under the same setting and notation as in the proof of Theorem 1, see Section B.1. If
for any ( ¯B, ¯Ω) ∈E(Θ0), it holds that dist( ¯B, A3) > 0, there exists α > 0 such that ℓ(B, Ω) −
ℓ(B0, Ω0) > α for all (B, Ω) ∈A3.

B
Detailed Proofs

B.1
Proof of Theorem 1

Proof. It suffices to consider the population case, i.e., ℓn(B, Ω) is replaced by its population counter-
part ℓ(B, Ω). By Lemma 7, we have

ℓ(B, Ω) > −∞

Also, pλ,δ(B) ≥0 for any B. Consequently, optimization problem (12) is well-defined.

By convention, we assume that (B0, Ω0) ∈Emin(Θ0). Now, consider the case where pλ,δ(B0) = 0,
which is equivalent to B0 = 0, since ℓ(B, Ω) ≥ℓ(B0, Ω0) and pλ,δ(B) ≥pλ,δ(B0) for any B.
Therefore, for all λ > 0 and δ > 0, B0 is the unique optimal solution to optimization problem (12),
proving the conclusion.

In the subsequent proof, we assume that |Emin(Θ0)| = 1, that is, Emin(Θ0) = {(B0, Ω0)}. This
assumption simplifies the proof because any element of Emin(Θ0) is indistinguishable based on the
value of ℓ(B, Ω) and the penalty for the chosen (δ, λ), as shown below. Our goal is to identify one
element via optimization problem (12), which significantly simplifies the argument.

First, let us define

δ0 =
τ
1 + ∆
where τ :=
min
(B,Ω)∈E(Θ0)
min
{(i,j)|Bij̸=0} |Bij|
(a)
= min
π
min
{(i,j)|[ e
B0(π)]ij̸=0}

[ eB0(π)]ij


with any ∆> 0. (a) is due to the fact that each element in equivalence class E(Θ0) is one-to-one
associated with eB0(π), see Section C.2 or [2] for detailed discussion. Then, for any λ > 0 and
0 < δ < δ0, consider the set A1 = {(B, Ω) | pλ,δ(B) = pλ,δ(B0)}. For any (B, Ω) ∈A1, we have
(B, Ω) /∈E(Θ0) \ {(B0, Ω0)}. This follows from the fact that

pλ,δ(B) = λδ

2 sB > λδ

2 sB0 = pλ,δ(B0)
∀(B, Ω) ∈E(Θ0) \ {(B0, Ω0)}.

As a consequence, this implies that ℓ(B0, Ω0) < ℓ(B, Ω), ∀(B, Ω) ∈A1. Therefore,

ℓ(B0, Ω0) + pλ,δ(B0) < ℓ(B, Ω) + pλ,δ(B)
∀(B, Ω) ∈A1.

Next, we define A2 = {(B, Ω) | pλ,δ(B) > pλ,δ(B0)}. Since ℓ(B0, Ω0) ≤ℓ(B, Ω), it follows that
for all (B, Ω) ∈A2, the following inequality holds:

ℓ(B0, Ω0) + pλ,δ(B0) < ℓ(B, Ω) + pλ,δ(B)
∀(B, Ω) ∈A2.

Therefore, we need to examine the set A3 = {(B, Ω) | pλ,δ(B) < pλ,δ(B0)}. For (B0, Ω0) to
achieve the minimum value of the score function, it is crucial that the following condition is satisfied:

ℓ(B0, Ω0) + pλ,δ(B0) < ℓ(B, Ω) + pλ,δ(B)
∀(B, Ω) ∈A3.

This condition guarantees that the ground truth parameters (B0, Ω0) correspond to the optimal
solution by comparing their score with any other parameters in the subset A3.

It is important to note that pλ,δ(t) = λp1,δ(t), ∀t. Thus, a necessary and sufficient condition for this
to hold is:

λ <
min
(B,Ω)∈A3
ℓ(B, Ω) −ℓ(B0, Ω0)

p1,δ(B0) −p1,δ(B) .

16


---Page Break---
Note that for all (B, Ω) ∈A3, we have p1,δ(B0) −p1,δ(B) ≤δ

2s0, with equality achieved when
B = 0. Therefore, the denominator on the RHS cannot be arbitrarily large. Moreover, since
(B, Ω) ∈A3, it follows that (B, Ω) /∈E(Θ0), as A3 ∩E(Θ0) = ∅.

We define the distance from ¯B to the set A3 as:

dist( ¯B, A3) =
inf
(B,Ω)∈A3 ∥B −¯B∥2.

For all ( ¯B, ¯Ω) ∈E(Θ0), it turns out that dist( ¯B, A3) must be positive due to the design of δ0, giving:

dist( ¯B, A3) >
min
(B,Ω)∈E(Θ0)
min
{(i,j)|Bij̸=0} |Bij| −δ0

=τ −
τ
1 + ∆=
∆
1 + ∆τ > 0.

By Lemma 8, there exists some α > 0 such that ℓ(B, Ω) −ℓ(B0, Ω0) > α for all (B, Ω) ∈A3.
Consequently, we have:

inf
(B,Ω)∈A3
ℓ(B, Ω) −ℓ(B0, Ω0)

p1,δ(B0) −p1,δ(B) > 0.

Thus, we can define

λ0 =
inf
(B,Ω)∈A3
ℓ(B, Ω) −ℓ(B0, Ω0)

p1,δ(B0) −p1,δ(B) > 0.

In summary, for all 0 < λ < λ0 and 0 < δ < δ0, for any ( bB, bΩ) ∈Emin(Θ0), and for all
(B, Ω) /∈Emin(Θ0), the following holds:

ℓ( bB, bΩ) + pλ,δ( bB) < ℓ(B, Ω) + pλ,δ(B).

This concludes the proof.

B.2
Proof of Theorem 2

Proof. Here, Θf(B0, Ω0) = Θ0. From Theorem 1, we know that when n →∞

On,λ,δ = Emin(Θ0).

Given the additional assumption that p(X) is faithful with respect to G0 := G(B0), by Lemma 1, we
have

M(G0) = Emin(Θ0) = {G(B) : (B, Ω) ∈Emin(Θ0)}.

Note that

{G(B) : (B, Ω) ∈Emin(Θ0)} = {G(B) : (B, Ω) ∈On,λ,δ}.

Thus, we conclude that
M(G0) = On,λ,δ.
as n →∞
This completes the proof.

B.3
Proof of Theorem 3

Proof. In this proof, we use the notation introduced in Section C.6. Note that when X is used in (12),
we essentially compute the sample covariance matrix bΣ based on X as follows:

bΣ = 1

n [X −1n · (bµ1, . . . , bµp)]⊤[X −1n · (bµ1, . . . , bµp)]

and plug it into the negative sample log-likelihood function. The same procedure applies to Z:

bΣstd = 1

nD−1 [X −1n(bµ1, . . . , bµp)]⊤[X −1n(bµ1, . . . , bµp)] D−1

= D−1bΣD−1.

17


---Page Break---
Denote bΘ = (bΣ)−1 and bΘstd = (bΣstd)−1 = DbΘD. For bΘ, by applying Theorem 1, there exist
λraw,n
0
> 0 and δraw,n
0
> 0 such that for any 0 < λ < λraw,n
0
and 0 < δ < δraw,n
0
, we have
On,λ,δ(X) = Emin(bΘ). For DbΘD, we apply Theorem 1 again, and there exist λstd,n
0
> 0 and
δstd,n
0
> 0 such that for any 0 < λ < λstd,n
0
and 0 < δ < δstd,n
0
, we have On,λ,δ(Z) = Emin(DbΘD).

We can select δ0 = min{δraw,n
0
, δstd,n
0
} and λ0 = min{λraw,n
0
, λstd,n
0
} in optimization (12) to ensure
that On,λ,δ(X) = Emin(bΘ) and On,λ,δ(Z) = Emin(DbΘD) hold simultaneously.

By applying Lemma 2, we conclude that:

G(On,λ,δ(X)) = G(Emin(bΘ)) = G(Emin(DbΘD)) = G(On,λ,δ(Z)).

Furthermore, as n →∞, we have bΣ →Σ and bΘ →Θ. Therefore, Emin(bΘ) →Emin(Θ), and
Emin(bΘstd) →Emin(Θstd) as n →∞. Thus,

G(On,λ,δ(X)) = G(Emin(Θ)) = G(On,λ,δ(Z)).
where n →∞

Note that we use n →∞to indicate we consider the result in population level.

B.4
Proof of Theorem 4

Proof. This proof shares many similarities with the proof of Theorem 1. First, when n →∞, we
consider the result at the population level. Thus, ℓn(ψ, ξ) →ℓ(ψ, ξ), and we will focus on ℓ(ψ, ξ) in
the following. As a result, we only work with ℓ(ψ, ξ) instead of ℓn(ψ, ξ).

By convention, we can always assume that (ψ0, ξ0) ∈Emin(ψ0, ξ0).

Now, consider the case where pλ,δ(B(ψ0)) = 0, which implies that B(ψ0) = 0. Since X ∼
P(X, ψ0, ξ0), we have
ℓ(ψ0, ξ0) ≤ℓ(ψ, ξ),
and thus

ℓ(ψ0, ξ0) + pλ,δ(B(ψ0)) ≤ℓ(ψ, ξ) + pλ,δ(B(ψ))
∀(ψ, ξ) ∈Ψ × Ξ, ∀λ > 0, ∀δ > 0.

Therefore, (ψ0, ξ0) is the optimal solution to optimization (12).

As we iterate, we can assume that |Emin(ψ0, ξ0)| = 1, meaning Emin(ψ0, ξ0) = {(ψ0, ξ0)}. This
assumption simplifies the proof because any element in Emin(ψ0, ξ0) is indistinguishable based on
the value of ℓ(ψ, ξ) and the penalty for the chosen parameters (δ, λ). Our goal is to find one element
by solving the optimization problem (14), and this assumption simplifies the argument significantly.

First, we define

δ0 =
τ
1 + ∆
τ :=
min
(ψ,ξ)∈E(ψ0,ξ0)
min
{(i,j):B(ψ)ij̸=0} |B(ψ)|ij.

It is important to note that, under Assumption A (1), since |E(ψ0, ξ0)| is finite, we have τ > 0.

Then, for any λ > 0 and 0 < δ < δ0, consider the set

A1 = {(ψ, ξ) | pλ,δ(B(ψ)) = pλ,δ(B(ψ0))}.

It is clear that for any (ψ, ξ) ∈A1, we must have (ψ, ξ) /∈E(ψ0, ξ0), since for any (ψ, ξ) ∈E(ψ0, ξ0),
we have pλ,δ(B(ψ)) > pλ,δ(B(ψ0)). As a result,

ℓ(ψ0, ξ0) < ℓ(ψ, ξ)
∀(ψ, ξ) ∈A1.

Therefore,

ℓ(ψ0, ξ0) + pλ,δ(B(ψ0)) < ℓ(ψ, ξ) + pλ,δ(B(ψ))
∀(ψ, ξ) ∈A1.

Next, we consider the set A2 = {(ψ, ξ) | pλ,δ(B(ψ)) > pλ,δ(B(ψ0))}, and we know that
ℓ(ψ0, ξ0) ≤ℓ(ψ, ξ). Therefore,

ℓ(ψ0, ξ0) + pλ,δ(B(ψ0)) < ℓ(ψ, ξ) + pλ,δ(B(ψ))
∀(ψ, ξ) ∈A2.

18


---Page Break---
Consequently, we need to check A3 = {(ψ, ξ) | pλ,δ(B(ψ)) < pλ,δ(B(ψ0))}. For (ψ0, ξ0) to be
the minimizer of (14), we require that the following condition holds:

ℓ(ψ0, ξ0) + pλ,δ(B(ψ0)) < ℓ(ψ, ξ) + pλ,δ(B(ψ))
∀(ψ, ξ) ∈A3.

This condition ensures that the ground truth parameters (ψ0, ξ0) correspond to the optimal solution
by comparing their score with that of any other parameters in the subset A3.

It is also worth noting that pλ,δ(t) = λp1,δ(t) for all t. Therefore, a necessary and sufficient condition
for this to hold is:

λ <
inf
(ψ,ξ)∈A3
ℓ(ψ, ξ) −ℓ(ψ0, ξ0)
p1,δ(B(ψ0)) −p1,δ(B(ψ)).

Note that for (ψ, ξ) ∈A3, we have p1,δ(B(ψ0)) −p1,δ(B(ψ)) ≤
δ
2sB(ψ0). Therefore, the de-
nominator on the RHS cannot be arbitrarily large. Moreover, for any 0 < δ < δ0, the following
holds:

∆
1 + ∆τ ≤∥B(ψ0) −B(ψ)∥2 ≤L∥ψ0 −ψ∥2
∀(ψ, ξ) ∈A3.

The second inequality follows from Assumption A (b). As a consequence,

∥(ψ, ξ) −(ψ0, ξ0)∥2 ≥∥ψ0 −ψ∥2 ≥
τ∆
L(1 + ∆).

Thus, we obtain

A3 ⊆{(ψ, ξ) | ∥(ψ, ξ) −(ψ0, ξ0)∥2 ≥
∆τ
L(1 + ∆)} = A4.

First, note that A3 is a nonempty set. Otherwise, the conclusion would hold immediately. Let us
select any ( ¯ψ, ¯ξ) ∈A3 ̸= ∅, and define A5 = {(ψ, ξ) | ℓ(ψ, ξ) ≤ℓ( ¯ψ, ¯ξ)}. Then,

inf
(ψ,ξ)∈A3 ℓ(ψ, ξ) =
inf
(ψ,ξ)∈A3∩A5 ℓ(ψ, ξ) ≥
inf
(ψ,ξ)∈A4∩A5 ℓ(ψ, ξ).

It is important to note that A4 ∩A5 is nonempty, since ( ¯ψ, ¯ξ) ∈A4 ∩A5. By Assumption B
and the properties of ℓ(ψ, ξ), we know that A5 is a bounded and closed set, and A4 is a closed set.
Consequently, A4∩A5 is compact. Furthermore, for all (ψ, ξ) ∈E(ψ0, ξ0), we have (ψ, ξ) /∈A4∩A5.
All of this leads to the following conclusion:

inf
(ψ,ξ)∈A3 ℓ(ψ, ξ) ≥
min
(ψ,ξ)∈A4∩A5 ℓ(ψ, ξ) =
inf
(ψ,ξ)∈A4∩A5 ℓ(ψ, ξ) > ℓ(ψ0, ξ0).

As a result, we define λ0 as follows:

λ0 =
inf
(ψ,ξ)∈A3
ℓ(ψ, ξ) −ℓ(ψ0, ξ0)
p1,δ(B(ψ0)) −p1,δ(B(ψ)) > 0.

B.5
Proof of Theorem 5

Proof. The proof is combination of Lemma 3 and Theorem 4, similar to Proof of Theorem 2.

B.6
Proof of Lemma 1

Before proving the result, we introduce the definitions of the Sparsest Markov representation assump-
tion and restricted faithfulness, along with a few useful theorems.
Definition 4 (Sparsest Markov representation [54]). A pair (G0, P) satisfies the Sparsest Markov
Representation (SMR) assumption if (G0, P) satisfies the Markov property and |G| > |G0| for every
DAG G such that (G, P) satisfies the Markov property and G ̸∈M(G0).

In other words, the SMR assumption asserts that the true DAG G0 is the (unique up to Markov
equivalence) sparsest DAG satisfying the Markov property.

19


---Page Break---
Definition 5 (Restricted-faithfulness [53, 54]). A distribution P satisfies the restricted-faithfulness
assumption with respect to a DAG G if it is Markov to G and following two conditions hold:

• Adjacency-faithfulness: for all (j, k) ∈E and all subsets S ⊂[p]\{j, k} it holds that
Xj ̸⊥⊥Xk | XS

• Orientation-faithfulness: for all triples (j, k, l) with skeleton j −l −k and all subsets
S ⊂[p]\{j, k} such that j is d-connected to k given S it holds that Xj ̸⊥⊥Xk | XS
Theorem 6 ([53]). If a distribution P is faithful to G, then such distribution P also satisfies the
restricted-faithfulness assumption with respect to G.
Theorem 7 (Theorem 2.4 in [54]). Let (G, P) satisfy the Markov property. Then the restricted-
faithfulness assumption implies the SMR assumption.

Proof. First, by Theorem 6, the faithfulness assumption implies the restricted faithfulness assump-
tion. Second, by Theorem 7, the restricted faithfulness assumption implies the Sparsest Markov
representation assumption. Furthermore, note that for any (B, Ω) ∈G(Emin(Θ0)), the distribution
P(X) is Markov to G(B), since (B, Ω) ∈G(E(Θ0)). According to the definition of the Sparsest
Markov representation assumption, all sparsest DAGs that satisfy the Markov property must belong
to the same Markov equivalence class. In our case, this means G(Emin(Θ0)) = M(G0).

B.7
Proof of Lemma 2

Proof. For X ∼N(0, Σ), let ¯Θ = DΘD, and denote the inverse of ¯Θ as ¯Σ = (¯Θ)−1 = D−1ΣD−1.
It follows that ¯X := D−1X ∼N(0, ¯Σ). Now, consider the following least squares regression for
j ∈{1, . . . , p} and S ⊆{1, . . . , p} \ {j}. Let β ∈R|S|. Then the following relationships hold:

βSj = arg min
β E∥Xj −β⊤XS∥2
2 ⇒βSj = Σ−1
SSΣSj

¯βSj = arg min
β E∥¯Xj −¯β⊤¯XS∥2
2 ⇒¯βSj = ¯Σ−1
SS ¯ΣSj

¯Σ−1
SS =([D−1ΣD−1]SS)−1 = DSSΣ−1
SSDSS
¯ΣSj =[D−1ΣD−1]Sj = D−1
SSΣSjD−1
jj
¯βSj =¯Σ−1
SS ¯ΣSj = DSSΣ−1
SSDSSD−1
SSΣSjD−1
jj = DSSβSjD−1
jj

As a consequence, supp(βSj) = supp(¯βSj). Note that for all (B, Ω) ∈E(Θ), we know from Section
C.2 that there exists a π ∈P such that B = eB(π). Moreover, B can be recovered by least squares
regression using X with its topological sort [2, 13] that is consistent with π. For such a π, we can
find a pair ( ¯B, ¯Ω) ∈E(DΘD), where ¯B has the same topological sort as π, and it can be recovered
by least squares regression on ¯X. We have shown that, for the same S, j, supp(βSj) = supp(¯βSj).
Therefore, supp(B) = supp( ¯B).

B.8
Proof of Lemma 3

Proof. The proof is the same as Lemma 1.

B.9
Proof Lemma 4

Proof. This follows directly from the definition of the Sparsest Markov Representation (SMR)
assumption. Since for all (B, Ω) ∈Emin(Θ), G(B) is Markovian to P and G(B) is the Sparsest,
by Definition 2, all G(B) must belong to the same Markov equivalence class by the definition of
SMR.

B.10
Proof of Lemma 5

Proof. Xstd = D−1(X −EX) is based on definition of standardization.

Cov(Xstd) = Cov(D−1(X −EX)) = D−1Cov((X −EX))D−1 = D−1ΣD−1

[Cov(Xstd)]−1 = [D−1ΣD−1]−1 = DΣ−1D = DΘD.

20


---Page Break---
B.11
Proof of Lemma 6

Proof. Detailed proof can be found in [41], Appendix Section D.

B.12
Proof of Lemma 7

Proof. From the definition of Σf(B, Ω):

Σf(B, Ω) := (I −B)−⊤Ω(I −B)−1 = (I −B)−⊤Ω1/2Ω1/2(I −B)−1,

where Ω1/2 = diag(ω1, . . . , ωp). It is clear that Σ(B, Ω) is positive semidefinite, as

x⊤Σ(B, Ω)x = ∥Ω1/2(I −B)−1x∥2
2 ≥0,
∀x ∈Rp.

Next, we just need to show that Ω1/2(I −B)−1x ̸= 0 for all x ̸= 0.

Ω1/2(I −B)−1x ̸= 0 ⇔(I −B)−1x ̸= 0 ⇔x ̸= 0.

Here, ω2
j > 0 for all j, so Ω1/2 is invertible. As (I −B) is a full rank matrix, then (I −B)−1 is also
a full rank matrix, it indicates that Σ(B, Ω) is positive definite matrix.

Since Ω0 > 0, it follows that Σ0 is positive definite. By Lemma 6, we have:

ℓ(B, Ω) =1

2 log det Ω−log det(I −B) + 1

2 Tr(Σ0Θ(B, Ω)) + const.

=1

2 log det Ω+ 1

2 Tr(Σ0Θ(B, Ω)) + const.

≥ℓ(B, Ωf(B)) = ℓ(B)

where Ωf(B) and ℓ(B) are defined in Equations (15) and (17), respectively. The last inequality
follows from Section C.1. Next, we need to prove that ℓ(B) > −∞.

ℓ(B) =1

2 log det diag((I −B)⊤Σ0(I −B)) + const.

=1

2

p
X

j=1
log E∥Xj −B⊤
j X∥2
2 + const.

=1

2

p
X

j=1
log E∥(ej −Bj)⊤X∥2
2 + const.

=1

2

p
X

j=1
log(ej −Bj)⊤Σ0(ej −Bj) + const.

≥1

2

p
X

j=1
log ∥(ej −Bj)∥2
2Λmin(Σ0) + const.

≥1

2

p
X

j=1
log Λmin(Σ0) > −∞

Here, ej ∈Rp is a unit vector with the j-th position equal to 1 and all other positions being zero, and
Λmin(Σ0) is the minimum eigenvalue of Σ0. Since Σ0 is positive definite, we have Λmin(Σ0) > 0.

Because B is the adjacency matrix of a DAG, it follows that Bjj = 0, which implies ∥ej −Bj∥≥
∥ej∥= 1. As a result,

ℓ(B, Ω) ≥ℓ(B) > −∞.

B.13
Proof of Lemma 8

Proof. Note that for a fixed B, the corresponding optimal Ωf(B) = diag((I −B)⊤Σ0(I −B))
is the solution with respect to ℓ(B, Ω). Therefore, without causing confusion, we take Ωf(B) =

21


---Page Break---
diag((I −B)⊤Σ0(I −B)) and consider the log-likelihood as a function of B only, i.e., ℓ(B), for
simpler representation. See Equation (17) in Section C.1 for details. It is clear that 0 ∈A3, so we
define A4 = {B | ℓ(B) ≤ℓ(0)}. Note that ℓ(0) is finite.

ℓ(0) ≥ℓ(B) =

p
X

j=1
log(ej −Bj)⊤Σ0(ej −Bj)

≥

p
X

j=1
log ∥ej −Bj∥2Λmin(Σ0)

= log



(Λmin(Σ0))p
p
Y

j=1
∥ej −Bj∥2



.

This indicates that

p
Y

j=1
∥ej −Bj∥2 ≤
exp(ℓ(0))
(Λmin(Σ0))p

Moreover,

∥ek −Bk∥2 ≤

p
Y

j=1
∥ej −Bj∥2 ≤exp(ℓ(0))

Λp
min
∀k ∈{1, . . . , p}

This implies that Bk must be bounded, and therefore every B in A4 is bounded. It is clear that
arg minB∈A3 ℓ(B) ∈A4. Thus, we need to show that minB∈A3 ℓ(B) = minB∈A3∩A4 ℓ(B) >
ℓ(B0). Define

A5 = { ˘B | dist( ˘B, E(Θ0)) ≥1

2
min
B∈E(Θ0) dist(B, A3)}

It is easy to see that A3 ⊆A5. Then,

min
B∈A3∩A4 ℓ(B) ≥
min
B∈A4∩A5 ℓ(B).

Note that A4 ∩A5 is closed, bounded, and nonempty (0 ∈A4 ∩A5), and ℓ(B) is a continuous
function of B. Consequently, there exists at least one minimizer of ℓ(B) in A4 ∩A5. Combining this
with the fact that B /∈A4 ∩A5 for all B ∈E(Θ0), we conclude that:

min
B∈A3 ℓ(B) ≥
min
B∈A4∩A5 ℓ(B) > ℓ(B0).

B.14
Proof of Corollary 1

Proof. By Theorem 1, we know there exists λ0 > 0 and δ0 > 0. For MCP, it can be transformed into
quasi-MCP, by reparameterization from Section C.5. Then, combining these results together.

0 < λ = λmcp < λ0, 0 < δ = aλmcp < δ0 ⇒0 < λmcp < λ0, 0 < a <
δ0
λmcp

We could simple set a0 : δ0

λ0 and (λmcp)0 = λ0. For SCAD, we just requires the following is satisfied
to satisfies the pattern in the proof of Theorem 1.

0 < aλscad < δ0, 0 < λ2
scad(a + 1)

2
< λ0

One simple choice is to let

a0 = (λscad)0 < min{
p

δ0,
p

λ0, 1}

This completes the proof of Corollary 1.

22


---Page Break---
C
Additional Examples and Details

In this appendix, we provide the additional details of derivations, examples, concepts, and discussions
referenced in the main paper. These include:

• The derivation of the log-likelihood function for the model in Equation (6) (Appendix C.1).
• A brief introduction to the characterization of the equivalence class E(Θ) (Appendix C.2).
• Examples demonstrating that the optimal solution for the least squares loss differs from the
optimal solution of the log-likelihood (Appendix C.3).
• An example illustrating the estimation bias when the ℓ1 penalty is applied (Appendix C.4).
• The formulations for quasi-MCP, MCP, and SCAD (Appendix C.5).
• The standardization of the random variable X and the dataset X (Appendix C.6).
• A detailed discussion of Assumptions A and B (Appendix C.7).

C.1
Log-likelihood of Model (6)

In this subsection, we detail the negative log-likelihood of the model in Equation (6).

ℓn(B, Ω) = −1

n log

n
Y

i=1
f(xi; B, Ω)

= −1

n log

n
Y

i=1

1
(2π)p/2(det Σ(B, Ω))1/2 exp

−x⊤
i Θ(B, Ω)xi

2



=p

2 log 2π + 1

2 log det Σ(B, Ω) + 1

2n

n
X

i=1
x⊤
i Θ(B, Ω)xi

=1

2 log det(I −B)−⊤Ω(I −B)−1 + 1

2 Tr(Θ(B, Ω)(
Pn
i=1 x⊤
i xi
n
)) + const.

=1

2 log det Ω−log det(I −B) + 1

2 Tr(bΣΘ(B, Ω)) + const.

=1

2

p
X

j=1
log w2
j + 1

2 Tr(Ω−1(I −B)⊤bΣ(I −B)) −log det(I −B) + const.

=1

2

p
X

j=1
log w2
j + 1

2

p
X

j=1

((I −B)⊤bΣ(I −B))jj

ω2
j
−log det(I −B) + const.

It is easy to know that for any fix B, the optimal solution of (ω∗
j )2 can be written as:

(ω∗
j )2 = [(I −B)⊤bΣ(I −B)]jj

Therefore, optimal solution Ωf(B) for any fixed B can be written as:

Ωf(B) = diag((I −B)⊤bΣ(I −B))
(15)

Let us define profile sample log-likelihood ℓn(B) as function of B with such optimal Ω(B) plugged
in

ℓn(B) =1

2 log det diag((I −B)⊤bΣ(I −B)) −log det(I −B) + const.

=1

2 log 1

n∥Xj −XBj∥2
2 −log det(I −B) + const.
(16)

Where X = (X1, . . . , Xp) = (x1, . . . , xn)⊤and corresponding profile population log-likelihood

ℓ(B) =1

2 log det diag((I −B)⊤Σ(I −B)) −log det(I −B) + const.

=1

2 log E∥Xj −B⊤
j X∥−log det(I −B) + const.
(17)

23


---Page Break---
C.2
Equivalence class E(Θ)

We provide a brief introduction to the equivalence class E(Θ), which has been extensively studied in
[2]. We adopt the notation from [2], and further details can be found in that work.
Definition 6 (topological sort). a topological sort of a directed graph is an ordering on the nodes,
often denoted by ≺, such that the existence of a directed edge Xk →Xj implies that Xk ≺Xj in
the ordering.

Let P denote the collection of all permutations of the indices {1, . . . , p}. For an arbitrary matrix
A and any π ∈P, let PπA represent the matrix obtained by permuting the rows and columns of A
according to π, such that (PπA)ij = aπ(i)π(j).

A DAG B is said to be compatible with permutation π if PπB is a lower-triangular matrix, which is
equivalent to saying that Xk →Xj in B implies that π−1(k) > π−1(j). Similarly, π is also called
compatible with B.

For any positive definite matrix Θ and π ∈P, the matrix PπΘ represents the same covariance
structure as Θ, up to a reordering of the variables. The Cholesky decomposition of Pπ(Θ) can be
uniquely written as:

PπΘ = (I −L)D−1(I −L)⊤= Θf(L, D),

where L is strictly lower triangular and D is diagonal. By Lemma 8 in [2], the following holds:

PπΘ(L, D) = Θ(PπL, PπD)
∀π ∈P.

Therefore,

Θ = Θf(Pπ−1L, Pπ−1D).

For each π, we define:

eB(π) := Pπ−1L,
eΩ(π) := Pπ−1D.

This suggests that for any π ∈P, there exists a pair ( eB(π), eΩ(π)) ∈E(Θ), where eB(π) can be
uniquely determined based on the permutation π and Θ [2]. It is important to emphasize that different
permutations, π1 ̸= π2, can still result in the same pairs, i.e., ( eB(π1), eΩ(π1)) = ( eB(π2), eΩ(π2)).
Furthermore, this indicates that for any (B, Ω) ∈E(Θ), there exists at least one permutation π such
that (B, Ω) = ( eB(π), eΩ(π)). Moreover, it turns out that the collection of pair of ( eB(π), eΩ(π)) forms
the entire equivalence class E(Θ).
Lemma 9 (Lemma 1, 2). Suppose Σ is a positive definite covariance matrix and Θ = Σ−1. Then,

E(Θ) ={(Pπ−1L, Pπ−1D) : PπΘ = Θf(L, D), π ∈P}

={( eB(π), eΩ(π)) : π ∈P}

This result indicates that the size of E(Θ) is at most p!, which is large but finite.

C.3
LS loss vs. log-likelihood

The following examples show that when the variances are unequal, the LS loss will not in general
have the same minimizers as the log-likelihood. The first example is just Example 1 in [33]. Suppose
(X1, X2) is distributed according to the following linear SEM with unequal variances:

X1 = ϵ1,
X2 = −X1

2 + ϵ2,
ϵ1 ∼N(0, 1),
ϵ2 ∼N(0, 1/4).

Thus

B0 =

0
−1/2
0
0


,
Ω0 =

1
0
0
1/4


,
Σ0 = Σf(B0, Ω0) =

1
−1/2
−1/2
1/2



and also

E(Θ0) = Emin(Θ0) = {B0, B1}

24


---Page Break---
where

B1 =

0
0
−1
0


,
Ω1 =

1
0
0
1


.

Moreover, ℓ(B0, Ω0) = ℓ(B1, Ω1), since both SEM represent the same covariance.

But it turns out that E[∥X −B⊤
1 X∥2] < E[∥X −(B0)⊤X∥2]: More precisely, it is easy to check
that

E[∥X −B⊤
1 X∥2] = Tr((I −B1)⊤Σ0(I −B1)) = 1,

E[∥X −(B0)⊤X∥2] = Tr((I −B0)⊤Σ0(I −B0)) = 5/4,

and moreover B1 is the global minimizer of the LS loss E[∥X −B⊤X∥2]. It follows that when the
variances are different, the log-likelihood and LS loss have different global minimizers.

Similar calculations can be carried out for d > 2, but are tedious owing to the size of E(Θ). For
example, here is an example of an SEM over 3 nodes such that the LS loss has a different set of global
minimizers, but also the LS-global minimizer has more edges than the sparsest Markov representation:

B0 =

 0
0
−3/10
0
0
−2
0
0
0

!

,
Ω0 =

 7
0
0
0
3
0
0
0
2

!

,
Σ0 = Σf(B0, Ω0) =

 7
0
−2
0
3
−5
−2
−5
10

!

.

For this model, LS loss selects the following SEM with 3 edges:

B1 =

 
0
0
0
−1.197
0
−1.589
−0.7532
0
0

!

.

We have B0 ∈Emin(Θ0), but B1 /∈Emin(Θ0).

C.4
Estimation bias under ℓ1

We provide an example showing that when the ℓ1 penalty is applied, the estimation becomes biased.
Therefore, ℓ1 should not be used. Consider the following linear Structural Equation Model (SEM):
X1 = N(0, 1)
X2 = X1 + N(0, σ2)

If the topological sort is known, i.e., X1 →X2, and an ℓ1 penalty is used for minimizing the negative
log-likelihood:

min
a log E[∥X2 −aX1∥2
2] + log E[∥X1∥2
2] + λ|a|

Ideally, we would expect a = 1 to be the minimal solution to the loss function. However, the problem
is equivalent to:

log((1 −a)2 + σ2) + λ|a|

It is clear that a = 1 is not the minimal solution to the loss function, as the derivative at a = 1 is
nonzero for any λ > 0, indicating that the ℓ1 penalty leads to a biased estimator. This bias does not
occur when using MCP or SCAD with appropriate hyperparameters.

C.5
quasi-MCP, MCP and SCAD

We present the formulas for quasi-MCP, MCP, and SCAD, and demonstrate that quasi-MCP and
MCP are equivalent.

quasi-MCP
pλ,δ(t) = λ

|t| −t2

2δ


1(|t| < δ) + δ

21(|t| > δ)


MCP
pMCP
λ,a
(t) = 1(|t| < aλ)

λ|t| −t2

2a


+ 1(|t| ≥λa)λ2a

2

SCAD
pSCAD
λ,a
(t) = λ|t|1(|t| < λ) + 1(λ < |t| < aλ)2aλ|t| −t2 −λ2

2(a −1)
+ 1(|t| ≥λa)λ2(a + 1)

2

25


---Page Break---
3
2
1
0
1
2
3
t

1

0

1

2

p , (t)

Plot of p , (t)

p , (t)

Figure 4: The plot of pλ,δ(t) with λ = 2, δ = 1

It is worth noting that if we set δ as aλ in quasi-MCP, then pλ,aλ(t) = pMCP
λ,a
(t). In another way, if
we set a = δ

λ in MCP, then pMCP
λ, δ

λ
(t) = pλ,a(t). Thus, quasi-MCP and MCP are equivalent to each
other.

C.6
Standardization of X and X

We present the formulas for the standardization of X and the standardization of the corresponding
dataset X.

Let σ2
i := Var(Xi) and D := diag(σ1, . . . , σp). Denote the standardized version of X as Xstd,
which can be expressed as:

Xstd = D−1(X −EX).

For X ∈Rn×p, we can write X = (Xij) = (x1, . . . , xn)⊤= (X1, . . . , Xp), and define sample
average for node j as bµj

bµj = 1

n

n
X

i=1
Xij
∀j ∈[p].

Next, we define the sample variance for node j as

bσ2
j =
1
n −1

n
X

i=1
(Xij −bµj)2.

The diagonal matrix of sample standard deviations is then

bD = diag(bσ1, . . . , bσp).

Finally, we standardize X by subtracting the sample means and scaling by the inverse of bD:

Z = [X −1n · (bµ1, . . . , bµp)] bD−1,

where 1n ∈Rn is an n-dimensional vector with all entries equal to 1.

C.7
Discussion of Assumption A, B

In this subsection, we provide a more detailed discussion of Assumptions A and B.

26


---Page Break---
First, it is important to emphasize that if pλ,δ is replaced by the ℓ0 penalty in (12) or (14), then
Assumptions A and B can be omitted, and all the results still hold. In this case, the proof would
be significantly simplified. However, the use of the differentiable quasi-MCP, in contrast to the ℓ0
penalty, introduces substantial complications, necessitating some additional assumptions that are
fundamentally different from the results in [8]. Specifically, Assumptions A and B are exactly what
is required to make the problem suitable for gradient-based optimization.

Assumption A is not restrictive and is satisfied by all identifiable models, including linear Gaussian
models, generalized linear models with continuous output [66], binary output [74, 4], and most
exponential families. However, the requirement for the finiteness of the equivalence class can be
relaxed. What is truly needed is that the minimal nonzero edge has sufficient “signal,” i.e.,

min
(ψ,ξ)∈E(ψ0,ξ0)
min
{(i,j):B(ψ)ij̸=0} |B(ψ)|ij > 0

This is trivially true when |E(ψ0, ξ0)| is finite. When |E(ψ0, ξ0)| is infinite, each |B(ψ)|ij could be
positive, but it is possible lim inf(ψ,ξ)∈E(ψ0,ξ0) min{(i,j):B(ψ)ij̸=0} |B(ψ)|ij = 0, because |B(ψ)|ij
can be arbitrarily small. The ℓ0 penalty deals with this with its discontinuity at zero, whereas the
continuity of quasi-MPC makes this more challenging. This is the cost of differentiability, which we
argue is worthwhile

Assumption B is a standard assumption in the optimization literature [7] and is generally quite weak.
Moreover, it is almost necessary because quasi-MCP cannot exactly count the number of edges in
B(ψ). The magnitude of the quasi-MCP penalty does not directly reveal the number of edges. This is
the trade-off for replacing the ℓ0 penalty with a fully differentiable sparsity-inducing penalty. Finally,
it is worth noting that this assumption can also be relaxed: what is truly required is that for any ϵ > 0,
there exists δ > 0 such that

ℓ(ψ, ξ) −ℓ(ψ0, ξ0) > δ
for all {(ψ, ξ) | dist((ψ, ξ), E(ψ0, ξ0)) > ϵ}.

In other words, we require a loss gap when (ψ, ξ) is not in E(ψ0, ξ0). This can be inferred from
Assumption B.

The following is the proof when pλ,δ is replaced with the ℓ0 penalty.

Proof when pλ,δ is replaced with ℓ0:

Proof. We can also assume that |Emin(ψ0, ξ0)| = 1, meaning Emin(ψ0, ξ0) = {(ψ0, ξ0)}. In other
words, there is a unique element in the minimal equivalence class. This is because any element in
Emin(ψ0, ξ0) is indistinguishable based on the score function, i.e., the value of ℓ(ψ, ξ) and the penalty
for the number of edges in B(ψ). Our objective is to find this unique element by solving equation
(12) in the paper, which simplifies the proof.

When sB(ψ0) = 0, the result is straightforward:

ℓ(ψ0, ξ0) + sB(ψ0) ≤ℓ(ψ, ξ) + sB(ψ).

Now, let us consider the more general case where sB(ψ0) > 0 and divide the parameter space into
three regions:

A1 = {(ψ, ξ) | sB(ψ) > sB(ψ0)},
A2 = {(ψ, ξ) | sB(ψ) = sB(ψ0)},
A3 = {(ψ, ξ) | sB(ψ) < sB(ψ0)}.

Case 1: Consider A1. Since ℓ(ψ0, ξ0) ≤ℓ(ψ, ξ), the following holds for any λ > 0:

ℓ(ψ0, ξ0) + λsB(ψ0) < ℓ(ψ, ξ) + λsB(ψ)
∀(ψ, ξ) ∈A1.

Case 2: Consider A2. Since |Emin(ψ0, ξ0)| = 1, it follows that for all (ψ, ξ) ∈E(ψ0, ξ0) and
(ψ, ξ) ̸= (ψ0, ξ0), we have sB(ψ) > sB(ψ0). Therefore, for all (ψ, ξ) ∈A2, it holds that ℓ(ψ0, ξ0) <
ℓ(ψ, ξ). Consequently, for any λ > 0:

ℓ(ψ0, ξ0) + λsB(ψ0) < ℓ(ψ, ξ) + λsB(ψ)
∀(ψ, ξ) ∈A2.

Case 3: Consider A3. We need to prove that:

ℓ(ψ0, ξ0) + λsB(ψ0) < ℓ(ψ, ξ) + λsB(ψ)
∀(ψ, ξ) ∈A3.

27


---Page Break---
This is equivalent to showing that there exists a positive λ such that:

λ < ℓ(ψ, ξ) −ℓ(ψ0, ξ0)

sB(ψ0) −sB(ψ)
∀(ψ, ξ) ∈A3.

Since sB(ψ0) is the minimal number of edges in the equivalence class, any (ψ, ξ) ∈A3 corresponds
to B(ψ) with a number of edges strictly less than sB(ψ0). This implies that:

ℓ(ψ, ξ) −ℓ(ψ0, ξ0) > 0
∀(ψ, ξ) ∈A3.

Furthermore, we have 1 ≤sB(ψ0) −sB(ψ) ≤sB(ψ0), which implies that there exists a small but
positive λ that satisfies the inequality.

D
Experiment Details

In this section, we provide all the details about the experiments. These include: (1) the types of
graphs used, (2) the process for generating the samples, (3) the baseline methods we compare against
and where to find the code for these methods, (4) the implementation details of our method and how
to replicate the results, and (5) the metrics used to evaluate the estimation.

D.1
Experimental Setting

In this section, we outline the process for generating graphs and data for Structural Equation Models
(SEMs) in (2). For each model, a random graph G is generated using one of two types of random
graph models: Erd˝os-Rényi (ER) or Scale-Free (SF). The models are specified to have, on average,
kp edges, where k ∈{1, 2, 4}. These configurations are denoted as ERk or SFk, respectively.

• Erd˝os-Rényi (ER), Random graphs whose edges are add independently with equal probability.
We simulated models with p, 2p and 4p edges (in expectation) each, denoted by ER1, ER2,
and ER4 respectively.

• Scale-free network(SF). Network simulated according to the preferential attachment process
[3]. We simulated scale-free network with p, 2p and 4p edges and β = 1, where β is the
exponent used in the preferential attachment process.

Linear SEMs.
Given a random DAG B ∈{0, 1}p×p from one of these two graph models, edge
weights were assigned independently from Unif([−1.5, −0.5] ∪[0.5, 1.5]) to obtain a weight matrix
B ∈Rp×p. Given B, we sampled X = B⊤X + z ∈Rp according to:

• Gaussian noise with unequal variance (Gauss-NV): zi ∼N(0, σ2
i ), i = 1, . . . , p where
σi ∼Unif[0.1, 0.7]

We chose to set σi, the noise variances in our models, to be relatively smaller compared to the settings
used in previous studies such as [73], [41], and [4]. This decision aims to mitigate the potential
exploitation of accumulated variance along the topological sort, as highlighted in [55].

Generalized Linear Model with Binary Output
Given a random DAG B ∈{0, 1}p×p from one
of these two graph models, edge weights were assigned independently from Unif([−1.5, −0.5] ∪
[0.5, 1.5]) to obtain a weight matrix B ∈Rp×p. Given B, we sample Xj according to the following

Xj = Bernoulli(exp(B⊤
j X)/(1 + exp(B⊤
j X)))
j = 1, . . . , p

where Bj is j-th column of B. The corresponding negative log-likelihood function:

s(B; X) = 1

n

p
X

i=1
1⊤
n (log(1n + exp(XB)) −Xi ◦(XB))

where X = (Xij) = (x1, . . . , xn)⊤= (X1, . . . , Xp)

28


---Page Break---
Nonlinear Models with Neural Networks.
We primarily follow the nonlinear setting described in
Zheng et al. [74]. Given G, we simulate the SEM as follows:

Xj = fj(Xpa(j)) + Nj
∀j ∈[p],

where Nj ∼N(0, σ2
i ) and σi ∼Uni[0.1, 1]. Here, fj is a randomly initialized MLP with one
hidden layer of size 100 and sigmoid activation. It is worth noting that the score function used in
nonlinear-NOTEARS [74] is least square loss:

s(f, X) = 1

2n

p
X

i=1
∥xi −bfi(X)∥2,

where each bfi is an MLP with one hidden layer of size 30 and sigmoid activation.

Simulation
We generated random datasets X ∈Rn×p by sampling rows i.i.d. from the models
described above. For each simulation, we produced datasets with n samples across graphs with p
nodes.

• Linear Model: p = {10, 20, 50, 70, 100}, k = {1, 2, 4}, n = 1000 and graph types = {ER,
SF}.

• Generalized Linear Model: p = {10, 20, 40}, k = {1, 2}, n = 10000 and graph types =
{ER, SF}.

• Nonlinear Model: p = {10, 20, 40}, k = {1, 2}, n = 1000 and graph types = {ER, SF}.

For each dataset, we applied several structural learning algorithms, including fast greedy equivalence
search (FGES [52]), constraint-based methods (PC [60]), NOTEARS [73, 74] (using least squares
loss), GOLEM [41] (using NLL with ℓ1 penalty), VarSort [55], causal additive models (CAM
[9]), LOGLL(-NOTEARS/DAGMA)-SAMPLE (utilizing the sample covariance matrix bΣ), LOGLL(-
NOTEARS/DAGMA)-POPULATION (using the population covariance matrix Σ) and exact method
(EXACT-SEARCH). Implementation details are provided in the following paragraph. After running
the algorithms, a post-processing threshold of 0.3 was applied to the estimated matrix Best to prune
small values, following the methodology in [73, 74].

Implementation
The implementation details of baseline are listed below:

• Fast Greedy Equivalence Search (FGES [52]) is based on greedy search and assumes
linear dependency between variables. The implementation is based on the py-tetrad
package, available at https://github.com/cmu-phil/py-tetrad. We use BIC as the
score function with default parameters.

• PC [60] is constraint-based method and based on uses conditional independence induced by
causal relationships to learn those causal relationships. The implementation is based on the
py-tetrad package, available at https://github.com/cmu-phil/py-tetrad. We use
Fisher-Z test with α = 0.5.

• NOTEARS[73, 74] is the continuous DAG learning algorithm using least square loss with ℓ1
regularization. It is implemented in python: https://github.com/xunzheng/notears.

• GOLEM [41] is implemented using Python and TensorFlow.
The code is available
https://github.com/ignavierng/golem.

• VarSort [55] is based on the observation that variances tend to accumulate along the topolog-
ical sort. It uses Lasso [61] to recover the coefficients. The code is implemented in Python
and is available https://github.com/Scriddie/Varsortability.

• DAGMA[4] is a continues DAG learning algorithm with better accuracy and faster computa-
tional speed. It also use least square loss with ℓ1 penalty as NOTEARS. The implementation
is available at https://github.com/kevinsbello/dagma.

• Causal additive model (CAM [9]) learns an addititve SEM by leveraging efficient nonpara-
metric regression techiques and greedy search over edges. The code is implemented in R,
and avaiable at https://rdrr.io/cran/CAM/man/CAM.html

29


---Page Break---
• LOGLL(-NOTEARS/DAGMA)-SAMPLE/POPULATION is our approach, which modifies the
original NOTEARS or DAGMA algorithm by replacing its scoring function. Instead
of using the least squares loss with an ℓ1 penalty, it employs a log-likelihood function
that includes a quasi-MCP penalty, as defined in (10). For LOGLL(-NOTEARS/DAGMA)-

SAMPLE, we use the sample covariance matrix ˆΣ in the score function. In contrast, LOGLL(-
NOTEARS/DAGMA)-POPULATION uses the true covariance matrix Σ as a baseline approach.
In this paper, LOGLL-NOTEARS refers to solving (12) using NOTEARS with NLL and quasi-
MCP as the score function, while LOGLL-DAGMA refers to solving (12) using DAGMA
with NLL and quasi-MCP as the score function.

• EXACT-SEARCH is used to indicate that the optimization problem (12) is solved exactly.
This approach is feasible only for small graphs, where we attempt to calculate all possible
configurations ˜B(π) as defined in Section C.2. These calculations can be performed using
Cholesky Decomposition or Ordinary Least Squares (OLS). The label POPULATION signifies
that the operation is based on the population covariance matrix Σ, while SAMPLE denotes
that it is based on the sample covariance matrix bΣ. The Structural Hamming Distance
(SHD) for EXACT-SEARCH is calculated on an average basis. This involves identifying the
set Mmin(Θ) or Mmin(bΘ), calculating the SHD for each DAG within this set, and then
computing the average SHD.

D.2
Implementation of LogLL(-NOTEARS/DAGMA)-population/sample

Linear model
There are two main challenges in solving (14). The first challenge is that (12) is a
highly nonconvex optimization problem and is sensitive to initialization. If we randomly initialize or
set the initialization to zero, as done in [73, 74, 41], LOGLL-NOTEARS/DAGMA often gets stuck at a
local optimal solution. The second challenge arises from Theorem 1, where we are advised to select
λ and δ such that 0 < λ < λ0 and 0 < δ < δ0. Theoretically, smaller values for λ and δ should be
used to adhere to the theorem’s guidelines, however, in practice, solving the optimization problem
(12) to global optimality is not always feasible.

To address the first challenge, we adopt the approach from Ng et al. [41]. We first run NOTEARS
(with least squares loss and ℓ1 penalty) or DAGMA (with least squares loss and ℓ1 penalty) to obtain
a "good" initialization point. Then, we apply LOGLL(-NOTEARS/DAGMA)-POPULATION/SAMPLE to
obtain the final output.

To address second challenge, we use warm starts. We begin with larger values for λ and δ and
solve (12) using LOGLL-NOTEARS/DAGMA to obtain an initial Best. We then reduce λ and δ by a
factor of γ < 1 and use the previous output as the starting point for the next iteration of LOGLL-
NOTEARS/DAGMA. This process is repeated until the negative log-likelihood ℓn(Best, Ωest) begins to
increase. This iterative approach helps to refine the solutions gradually, ensuring that each step starts
from a potentially better approximation, as formally outlined in Algorithm 1.

In Algorithm 1, we detail the complete implementation of LOGLL(-NOTEARS/DAGMA)-SAMPLE. By
replacing bΣ with Σ and substituting ℓn with ℓ, this algorithm is adapted to the full implementation of
LOGLL(-NOTEARS/DAGMA)-POPULATION.

It turns out that LOGLL-DAGMA outperforms LOGLL-NOTEARS in our experiments. Therefore, we
present the results from LOGLL-DAGMA.

Nonlinear model
We utilize LOGLL-NOTEARS, which uses the same optimization framework in
[74] with replacement of least square loss to negative log-likelihood loss, we keep other hyperparam-
eter unchanged. The score function (negative log-likelihood) we use

sNLL(f, X) = 1

2n

d
X

i=1
log

∥xi −bfi(X)∥2

Here ˆfi is i-th MLP with one hidden layer of size 40 and sigmoid activation.

Standardized data Z
Although it has been shown that the log-likelihood score is scale-invariant
for the linear model with Gaussian noise (see Theorem 3), it was observed that using standardized

30


---Page Break---
Algorithm 1: Full Implementation

Input: Sample covariance bΣ, decay factor γ ∈(0, 1), and λ, δ, initial point (Bin, Ωin), initial
loss ℓin (typically very large)
// initial point (Bin, Ωin) is obtained from NOTEARS or DAGMA

Output: (Best, Ωest)

1 while True do

2
Solve LOGLL-NOTEARS or LOGLL-DAGMA with input (Bin, Ωin), and get output (Bout, Ωout)

3
Calculate ℓn(Bout, Ωout)

4
if ℓin > ℓn(Bout, Ωout) then

5
ℓin ←ℓn(Bout, Ωout)

6
λ ←γλ

7
δ ←γδ

8
(Bin, Ωin) ←(Bout, Ωout)

9
else

10
return (Bin, Ωin)

11
end

12 end

data Z makes solving the optimization problem (12) significantly more challenging. For the LOGLL-
NOTEARS implementation, the LBFGS-B algorithm fails to produce meaningful solutions. As a result,
we replaced LBFGS-B with ADAM [27], an optimizer better suited for handling the difficulties
of standardized data, to solve the subproblem in NOTEARS. Alternatively, directly using LOGLL-
DAGMA is another effective way to address this challenge. Empirically, we find that setting γ = 0.8,
λ = 0.4, and δ = 0.2 usually serves as a good choice for the parameters in our optimization
procedures.

D.3
Metrics

We evaluate the performance of each algorithm with the following three metrics:

• Structure Hamming distance (SHD): A standard benchmark in the structure learning
literature that counts the total number of edges additions, deletions, and reversals needed
to convert the estimated graph into the true graph. Since our model specified in (6) is
unidentifiable, the Structural Hamming Distance (SHD) is calculated with respect to the
completed partially directed acyclic graph (CPDAG) of the ground truth and Best. We utilize
the code from Zheng et al. [75].
• Times: The amount of time the algorithm takes to run, measured in seconds. This metric is
used to evaluate the speed of the algorithms.

31


---Page Break---
E
Additional Results

E.1
Linear Model (SHD)

0
5
10
15
20
25
Structural Hamming Distance (SHD)

Exact_population

Exact_sample

LogLL_population

LogLL_sample

VarSort

DAGMA

GOLEM

NOTEARS

FGES

PC

Method

LogLL
Baseline
Countiuous
Combinatorial
raw
standardized

Figure 5: Structural Hamming Distance (SHD, with lower values indicating better performance)
between Markov equivalence classes (MEC) of recovered and ground truth graphs for ER-2 graphs
with 8 nodes. Here Exact-search is added to illustrate Theorem 3. Standardization does not affect
the DAG structure if the optimization (10) can be solved globally. Both Exact-sample and Exact-
population produce the same DAG structure for raw data X and standardized data Z. When the
population covariance matrix is known, Emin(Θ0) = M(G0), resulting in an SHD of zero. The poor
performance of Exact-sample can be attributed to the lack of thresholding applied to the coefficients
recovered from Ordinary Least Squares (OLS). Since bΣ is only an approximation of Σ, coefficients
derived from OLS based on different permutations π may shift from zero to nonzero, even though
such coefficients might be very small. However, since Exact is impractical for real-world applications,
we use this example primarily for illustrative purposes, and thus no threshold is applied to this
method.

32


---Page Break---
0
2
4
6
8
Structural Hamming Distance (SHD)

LogLL_population

LogLL_sample

VarSort

DAGMA

GOLEM

NOTEARS

FGES

PC

Method

LogLL
Baseline
Countiuous
Combinatorial
raw
standardized

Figure 6: Comparison of raw (orange) vs. standardized (green) data. Structural Hamming Distance
(SHD, with lower values indicating better performance) between Markov equivalence classes (MEC)
of recovered and ground truth graphs for ER-2 graphs with 5 nodes

0
20
40
60
80
100
120
Structural Hamming Distance (SHD)

LogLL_population

LogLL_sample

VarSort

DAGMA

GOLEM

NOTEARS

FGES

PC

Method

LogLL
Baseline
Countiuous
Combinatorial
raw
standardized

Figure 7: Comparison of raw (orange) vs. standardized (green) data. Structural Hamming Distance
(SHD, with lower values indicating better performance) between Markov equivalence classes (MEC)
of recovered and ground truth graphs for ER-2 graphs with 20 nodes

33


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0

1

0

1

2

3

4

5

6

SHD between MEC

0->1,0->2,1->3,2->3

maximal 
shd

0.0
0.2
0.4
0.6
0.8
1.0

0

2

4

6

8

Distance

0->1,0->2,1->3,2->3

maximal 
dist

10000

20000

30000

40000

50000

Number of Points

10000

20000

30000

40000

50000

Number of Points

Figure 8: Graph: structure X0 →X1, X0 →X2, X1 →X3, X2 →X3. For 0 < δ < δ0, the
estimated (Best, Ωest) ∈Emin(Θ0) because SHD and distance are closed to 0.
.

0.0
0.2
0.4
0.6
0.8
1.0

1.0

0.5

0.0

0.5

1.0

1.5

2.0

2.5

3.0

SHD between MEC

V Structure: 0->1,2->1

maximal 
shd

0.0
0.2
0.4
0.6
0.8
1.0

0

1

2

3

4

5

Distance

V Structure: 0->1,2->1

maximal 
dist

20000

40000

60000

80000

100000

Number of Points

20000

40000

60000

80000

100000

Number of Points

Figure 9: Graph: structure X0 →X1, X2 →X1. For 0 < δ < δ0, the estimated (Best, Ωest) ∈
Emin(Θ0) because SHD and distance are closed to 0.
.

34


---Page Break---
E.2
Linear Model (Time)

1

ER
SF

25
50
75
100

0

100

200

300

400

500

0

100

200

300

400

500

d (Number of nodes)

Time (seconds)

2

ER
SF

25
50
75
100

0

100

200

300

400

500

0

200

400

d (Number of nodes)

4

ER
SF

25
50
75
100

0

200

400

600

0

200

400

d (Number of nodes)

method
DAGMA
FGES
GOLEM
LOGLL_NOTEARS_POPULATION
LOGLL_NOTEARS_SAMPLE
NOTEARS
PC

Figure 10: Results in term of Time. Lower is better. Column: k = {1, 2, 4}. Row: random
graph types. {ER,SF}-k = {Scale-Free,Erd˝os-Rényi } graphs with kd expected edges. Here d =
{10, 20, 50, 70, 100}, n = 1000. Standard error is removed for better visualization. It is for different
methods on raw data X

1

ER
SF

25
50
75
100

0

200

400

0

200

400

600

d (Number of nodes)

Time (seconds)

2

ER
SF

25
50
75
100

0

200

400

600

0

200

400

600

d (Number of nodes)

4

ER
SF

25
50
75
100

0

200

400

600

0

200

400

600

d (Number of nodes)

method
DAGMA
FGES
GOLEM
LOGLL_NOTEARS_POPULATION
LOGLL_NOTEARS_SAMPLE
NOTEARS
PC

Figure 11:
Results in term of Time. Lower is better. Column: k = {1, 2, 4}. Row: random
graph types. {ER,SF}-k = {Scale-Free,Erd˝os-Rényi } graphs with kd expected edges. Here d =
{10, 20, 50, 70, 100}, n = 1000. Standard error is removed for better visualization. It is for different
methods on standardized data Z

35


---Page Break---
E.3
Nonlinear Model (SHD)

E.3.1
Neural Network

0
2
4
6
8
10

CAM

L2

LOGLL

d = 10, graph = SF, k = 1

2
4
6
8
10
12

d = 10, graph = ER, k = 1

2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
22.5

d = 10, graph = SF, k = 2

5.0
7.5
10.0
12.5
15.0
17.5
20.0
22.5

d = 10, graph = ER, k = 2

0
5
10
15
20
25

CAM

L2

LOGLL

d = 20, graph = SF, k = 1

5.0
7.5
10.0
12.5
15.0
17.5
20.0
22.5

d = 20, graph = ER, k = 1

10
15
20
25
30
35
40

d = 20, graph = SF, k = 2

15
20
25
30
35
40
45

d = 20, graph = ER, k = 2

10
15
20
25
30
35
40

CAM

L2

LOGLL

d = 40, graph = SF, k = 1

15
20
25
30
35
40
45

d = 40, graph = ER, k = 1

30
40
50
60
70
80

d = 40, graph = SF, k = 2

40
50
60
70
80

d = 40, graph = ER, k = 2

MEC Structural Hamming Distance by Method

raw
standardized

Figure 12: Structural Hamming distance (SHD) between Markov equivalence classes (MEC) of
recovered and ground truth graphs. LOGLL (i.e. LOGLL-NOTEARS) stands for NOTEARS method
with log-likelihood and quasi-MCP, L2 (i.e. NOTEARS) stands for NOTEARS method with least
square and ℓ1.

E.3.2
General Linear Model with Binary Output (Logistic Model)

1

ER
SF

10
20
30
40

4

8

12

16

4

8

12

16

d (Number of nodes)

Structural Hamming Distance (SHD)

2

ER
SF

10
20
30
40

10

20

30

10

20

30

40

d (Number of nodes)

4

ER
SF

10
20
30
40

20

30

40

50

60

70

20

30

40

50

60

d (Number of nodes)

method
NOTEARS
NOTEARS_LOGLL

General Linear Model with Binary Output (Logistic Model)

Figure 13: Structural Hamming distance (SHD) for Logistic Model, Row: random graph types, {SF,
ER}-k= {Scale-Free,Erd˝os-Rényi } graphs. Columns: kd expected edges. NOTEARS_LOGLL (i.e.
LOGLL-NOTEARS) uses log-likelihood with quasi-MCP, NOTEARS use log-likelihood with ℓ1. Error
bars represent standard errors over 10 simulations.

36


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We are confident about this point.

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

Justification: This is the last section of main paper.

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

37


---Page Break---
Justification: We include all the important assumptions.
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
Justification: We provide the detailed discussion in the appendix.
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

38


---Page Break---
Answer: [Yes]
Justification: It is simple adaption of open source code. We will release our the code.
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
Justification: It is included in the experinments section.
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
Justification: We include the error bar to illustrate this point.
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

39


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

Answer: [No]

Justification: We run the algorithm on personal computer. It should be easy to replicate the
results.

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

Answer: [No]

Justification: We mainly focus on the theoretical findings.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

40


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

Justification: This is unrelated to the topic of paper.

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

Justification: We cite the related paper.

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

41


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification:
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
Justification:
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
Justification:
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
