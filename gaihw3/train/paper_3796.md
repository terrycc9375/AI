Small coresets via negative dependence:
DPPs, linear statistics, and concentration

Rémi Bardenet ∗
Univ. Lille, CNRS, Centrale Lille,
UMR 9189 – CRIStAL,
F-59000 Lille, France
remi.bardenet@cnrs.fr

Subhroshekhar Ghosh ∗
Department of Mathematics
National University of Singapore
10 Lower Kent Ridge Road, 119076, Singapore
subhrowork@gmail.com

Hugo Simon-Onfroy ∗
Université Paris-Saclay, CEA, Irfu
Département de Physique des Particules
91191, Gif-sur-Yvette, France
hugo.simon@cea.fr

Hoang Son Tran ∗†
Department of Mathematics
National University of Singapore
10 Lower Kent Ridge Road, 119076, Singapore
hoangson.tran@u.nus.edu

Abstract

Determinantal point processes (DPPs) are random configurations of points with
tunable negative dependence. Because sampling is tractable, DPPs are natural can-
didates for subsampling tasks, such as minibatch selection or coreset construction.
A coreset is a subset of a (large) training set, such that minimizing an empirical
loss averaged over the coreset is a controlled replacement for the intractable min-
imization of the original empirical loss. Typically, the control takes the form of
a guarantee that the average loss over the coreset approximates the total loss uni-
formly across the parameter space. Recent work has provided significant empirical
support in favor of using DPPs to build randomized coresets, coupled with interest-
ing theoretical results that are suggestive but leave some key questions unanswered.
In particular, the central question of whether the cardinality of a DPP-based coreset
is fundamentally smaller than one based on independent sampling remained open.
In this paper, we answer this question in the affirmative, demonstrating that DPPs
can provably outperform independently drawn coresets. In this vein, we contribute
a conceptual understanding of coreset loss as a linear statistic of the (random)
coreset. We leverage this structural observation to connect the coresets problem to
a more general problem of concentration phenomena for linear statistics of DPPs,
wherein we obtain effective concentration inequalities that extend well-beyond
the state-of-the-art, encompassing general non-projection, even non-symmetric
kernels. The latter have been recently shown to be of interest in machine learning
beyond coresets, but come with a limited theoretical toolbox, to the extension
of which our result contributes. Finally, we are also able to address the coresets
problem for vector-valued objective functions, a novelty in the coresets literature.

1
Introduction

Let X = {xi | i ∈J1, nK} be a set of n points in a Euclidean space, called the data set. Let F be a
set of nonnegative functions on X, called queries. Many classical learning problems, supervised or

∗The authors are listed in alphabetical order by their surnames
†Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
unsupervised, are formulated as finding a query f ∗in F that minimizes an additive loss function of
the form
L(f) :=
X

x∈X
µ(x)f(x),
(1)

where µ : X →R+ is a weight function.
Example 1 (k-means). For X ⊂Rd and k ∈N, the goal of k-means clustering is to find a set C∗of
k “cluster centers" by minimizing (1) over

F =

fC : x 7→min
q∈C ∥x −q∥2
2 | C ⊂Rd, |C| = k

.

Here, each query f is indexed by a set of k cluster centers, and the loss (1) is the quantization error.
Example 2 (linear regression). When X = {xi := (yi, zi) | i ∈J1, nK} ⊂Rd+1, linear regression
corresponds to minimizing (1) over

F =

(y, z) 7→(a⊤y + b −z)2 | a ∈Rd, b ∈R
	
.

Penalty terms can be added to each function, to cover e.g. ridge or lasso regression.

In many machine learning applications, the complexity of the corresponding optimization problem
grows with the cardinality n of the dataset. When n ≫1 makes optimization intractable, one is
tempted to reduce the amount of data, using only a tractable number of representative samples. This
is the idea formalized by coresets; we refer to (Bachem, Lucic, and Krause, 2017) for a survey, and
to (Huang, Li, and Wu, 2024; Cohen-Addad, Larsen, Saulpic, Schwiegelshohn, and Sheikh-Omar,
2022) for specific coreset constructions for k-means and Euclidean clustering. An ε-coreset is a
subset S ⊂X, possibly with corresponding weights ω(x), x ∈S, such that

LS(f) :=
X

x∈S
ω(x)f(x)
(2)

is within ε of L(f), uniformly in f ∈F. If the cardinality m of S is significantly smaller than the
intractable size n of the original data set, one has reduced the complexity of the algorithm at a little
cost in accuracy.

Many randomized coreset constructions, where such guarantees are shown to hold with large proba-
bility, are built by drawing elements independently from the data set X (Bachem et al., 2017, Chapter
3). Because a representative coreset should intuitively be made of diverse data points, negative
dependence between the coreset elements has been proposed as an effective possibility to improve
their performance (Tremblay, Barthelmé, and Amblard, 2019). In particular, the authors advocate the
use of Determinantal Point Processes (DPPs), a family of probability distributions over subsets of X
parametrized by an n × n kernel matrix K that enforces diversity, all of this while coming with a
polynomial-time exact sampling algorithm.

Tremblay et al., 2019 give extensive theoretical and empirical justification for the use of DPPs in
randomized coreset construction. In one of their key results, using concentration results in (Pemantle
and Peres, 2011), Tremblay et al., 2019 bound the cardinality of a DPP-based ε-coreset, and their
bound is O(ε−2). However, it is known that the best ε-coresets built with independent samples are
also of cardinality O(ε−2). Thus, the crucial question of whether DPP-based coresets can provide
a strict improvement remained to be settled; given the computational simplicity of independent
schemes, this would be fundamental to justify the deployment of DPP-based methods.

In this paper, we settle this question in the affirmative, demonstrating that for carefully chosen kernels,
DPP-based coresets provably yield significantly better accuracy guarantees than independent schemes;
equivalently, to achieve similar accuracy it suffices to use significantly smaller coresets via DPPs. In
particular, we will show that DPP-based coresets actually can achieve cardinality m = O(ε−2/(1+δ)).
The quantity δ depends on the variance of the subsampled loss under the considered DPP, and some
DPPs yields δ > 0. A cornerstone of our approach is a structural understanding of the coreset loss (2)
as a so-called linear statistic of the random point set S, which enables us to go beyond earlier results
that were based on concentration properties of general Lipschitz functions of a DPP (Pemantle and
Peres, 2011).

In this endeavour, we obtain very widely-applicable concentration inequalities for linear statistics of
DPPs compared to the state of the art; cf. (Breuer and Duits, 2013) that mostly focuses on scalar-
valued statistics for finite rank ensembles on R. In particular, we are able to address all DPPs that

2


---Page Break---
have appeared so far in the ML literature. Specifically, our results are able to handle non-symmetric
kernels and vector-valued linear statistics.

DPPs with non-symmetric kernels have recently been shown to be of significant interest in machine
learning, such as recommendation systems (Gartrell, Brunel, et al., 2019; Gartrell, Han, et al., 2020;
Han et al., 2022), but they come with a limited theoretical toolbox, to which this paper makes a
contribution. On the other hand, vector-valued statistics arise naturally in many learning problems,
including coreset settings such as the gradient estimator in Stochastic Gradient Descent (Bardenet,
Ghosh, et al., 2021). However, the literature on coresets for vector-valued statistics is scarce, and in
this paper we inaugurate their study with effective approximation guarantees via DPPs.

The rest of the paper is organized as follows. Section 2 contains background on DPPs and coresets.
Section 3 contains our contributions. Section 4 provides numerical illustrations. Section 5 contains a
discussion on limitations and future work.

2
Background

We introduce here the two key notions of determinantal point process and coreset, and observe that a
coreset guarantee is a uniform control over specific linear statistics of a point process.

Determinantal point processes.
A point process S on a Polish space X is a random locally finite
subset of X. Given a reference measure µ on X (e.g., the Lebesgue measure if X = Rd or the
counting measure if X is discrete), a point process S is called a DPP (w.r.t. µ) if there exists a
measurable function K : X × X →C such that

E
h X

̸=
f(xi1, . . . , xik)
i
=
Z

X k f(x1, . . . , xk) det[K(xi, xj)]k×k dµ⊗k(x1, . . . , xk),
(3)

where the sum in the LHS ranges over all pairwise distinct k-tuples of the random locally finite subset
S, for all bounded measurable f : X k →R and for all k ∈N. Such a function K is called a kernel
for the DPP S, and µ is called the background measure.

When the ground set X is of finite cardinality n, an equivalent but more intuitive way to define DPPs
is as follows: a random subset S of X is called a DPP if there exists an n × n-matrix K such that
P(T ⊂S) = det[KT ],
∀T ⊂X,
where KT denotes the submatrix of K with rows and columns indexed by T.

In a similar vein to Gaussian processes, all the statistical properties of a DPP are encoded in this
kernel function K and background measure µ. A feature of DPPs with far-reaching implications
for machine learning is that sampling and inference with DPPs are tractable. We refer the reader
to (Hough et al., 2006; Kulesza and Taskar, 2012) for general references. Originally introduced in
electronic optics (Macchi, 1975), they have been turned into generic statistical models for repulsion in
spatial statistics (Lavancier et al., 2014; Biscio and Lavancier, 2017) and machine learning (Kulesza
and Taskar, 2012; Belhadji et al., 2020a; Brunel, 2018; Derezinski and Mahoney, 2019; Derezinski,
Liang, et al., 2020; Gartrell, Brunel, et al., 2019; Ghosh and Rigollet, 2020).
Example 3 (L-ensemble and m-DPP). Let X be a finite set of cardinality n, µ be the counting
measure, and L be a positive semi-definite n × n-matrix. The L-ensemble with parameter L is the
point process S on X such that, for all T ⊂X, P(S = T) ∝det[LT ], where LT is the square
submatrix of L corresponding to the rows and columns indexed by the subset T. It can be shown
that S is a DPP on X with kernel K := L(I + L)−1. In general, the cardinality of S is a random
variable. By conditioning on the event {|S| = m}, we obtain the so-called m-DPPs (Kulesza and
Taskar, 2012).
Example 4 (Multivariate OPE; Bardenet and Hardy, 2020). Let X = Rd and µ be a measure on
Rd having all moments finite, let (pk)k∈Nd be the orthonormal sequence resulting from applying the
Gram-Schmidt procedure to the monomials xk1
1 . . . xkd
d , taken in the graded lexical order. The kernel
K(m)
µ
(x, y) := Pm−1
k=0 pk(x)pk(y) then defines a projection DPP on Rd, called the multivariate
Orthogonal Polynomial Ensemble (OPE) of rank m and reference measure µ.

Multivariate OPEs were used in (Bardenet and Hardy, 2020) as nodes for numerical integration,
leading to a Monte Carlo estimator with mean squared error decaying in m−1−1/d, faster than under

3


---Page Break---
independent sampling. In (Bardenet, Ghosh, and Lin, 2021), the authors investigated the problem
of DPP-based minibatch sampling for Stochastic Gradient Descent (SGD), and exploited a delicate
interplay between a finite dataset and its ambient data distribution to leverage this fast decay for
improved approximation guarantees. In particular, they proposed the following DPP defined on a
(large) finite ground set.
Example 5 (Discretized multivariate OPE; Bardenet, Ghosh, et al., 2021). Let n ∈N and X =
{x1, . . . , xn} ⊂[−1, 1]d. Let q(x)dx be a probability measure on [−1, 1]d. Let K(m)
q
be the
multivariate OPE kernel of rank m with reference measure q(x)dx, as defined in Example 4. Let
˜γ : [−1, 1]d →R+ be a function, assumed to be positive on X, and consider

K(m)
q,˜γ (x, y) :=

s

q(x)
˜γ(x)K(m)
q
(x, y)

s

q(y)
˜γ(y),
x, y ∈[−1, 1]d.

Consider then the n × n matrix ˜K = K(m)
q,˜γ |X×X . ˜K is symmetric and positive semidefinite, and
we let K be the matrix with the same eigenvectors, the m largest eigenvalues replaced by 1, and the
remaining eigenvalues replaced by 0. Then K defines a DPP on X.

Coresets.
Let ε > 0 and X be a set of cardinality n. The classical definition of a coreset is
multiplicative.
Definition 1 (multiplicative coreset). A subset3 S ⊂X is an ε-multiplicative coreset if

∀f ∈F,

LS(f)

L(f) −1
 ≤ε,
(4)

where L and LS are respectively defined in (1) and (2).

An immediate and important consequence of (2) is that the ratio of the minimum value of LS by that
of L is within O(ε) of 1 (Bachem et al., 2017, Theorem 2.1).

One way to satisfy (2) with high probability for a single f is through importance sampling, taking S
to be formed of m > 0 i.i.d. samples from some instrumental density q on X, and taking ω = µ/q in
(2). Langberg and Schulman, 2010 showed that a suitable choice of q actually yields the uniform
guarantee (2). It suffices to take for instrumental pdf q(x) ∝µ(x)s(x), where s upper-bounds the
so-called sensitivity

s(x) ≥sup
f∈F

f(x)
P

y∈X µ(y)f(y),
∀x ∈X.
(5)

For δ > 0, k ≥S2

2ε2 log 2/δ independent draws are then enough to build an ε-multiplicative coreset,
where S = P

x∈X µ(x)s(x); see (Bachem et al., 2017)[Section 2.3]. The tighter the bound (5), the
smaller the size of the coreset. One important limitation is that finding a tight bound is nontrivial.

Although not standard, a natural alternative definition of a coreset is that of an additive coreset.
Definition 2 (additive coreset). A subset S ⊂X is an ε-additive coreset if
1
n |LS(f) −L(f)| ≤ε,
∀f ∈F.
(6)

Note the arbitrary scaling factor 1/n in (6) compared to (2), which we adopt to simplify comparisons
between the two coreset definitions. With an additive coreset, the minimal value of LS is guaranteed
to be within ±nε of the minimal value of L: Similarly to a multiplicative coreset, with ε suitably
small one should be happy to train one’s algorithm only on S.

Coreset guarantee and linear statistics.
Let S be a point process on a finite X = {x1, . . . , xn}.
For a test function φ : X →R, we denote by Λ(φ) := P
x∈S φ(x) the so-called linear statistic of
φ. In a coreset problem, for a query f ∈F, the estimated loss LS(f) in (2) is the linear statistic
Λ(ωf). When S is a DPP with a kernel K on X (w.r.t. the counting measure), we will choose the
weight ω(x) = K(x, x)−1, where for x = xi ∈X, we define K(x, x) to be Kii. By (3), this choice
makes LS(f) an unbiased estimator for L(f). Guaranteeing a coreset guarantee such as (6) with high
probability thus corresponds to a uniform-in-f concentration inequality for the linear statistic Λ(ωf).
This motivates studying the concentration of linear statistics under a DPP, to which we now turn.

3Note that we defined a coreset as a subset and not a sub-multiset of X, thus ignoring multiplicity. This is
because we allow weights in (2), so that repeated items are unnecessary in a coreset.

4


---Page Break---
3
Theoretical results

We first give new results on the concentration of linear statistics under very general DPPs. These
results are of interest in their own right, and should find applications in ML beyond coresets. Next we
examine the implications of the concentration of linear statistics for coresets, showing that a suitable
DPP does yield a coreset size of size o(ε−2), thus beating independent sampling.

Concentration inequalities for linear statistics of DPPs.
We start with Hermitian kernels.

Theorem 1 (Hermitian kernels). Let S be a DPP on a Polish space X with reference measure µ and
Hermitian kernel K. Then for any bounded test function φ : X →R, we have

P(|Λ(φ) −E[Λ(φ)]| ≥ε) ≤2 exp

−
ε2

4A Var [Λ(φ)]


,
∀0 ≤ε ≤2A Var [Λ(φ)]

3∥φ∥∞
,

where A > 0 is a universal constant.

Our Theorem 1 is similar in spirit to a seminal concentration inequality by Breuer and Duits, 2013.
However, their result only applies to DPPs with Hermitian projection kernels of finite rank. We
emphasize that our Theorem 1 is applicable to all Hermitian kernels on general Polish spaces.

In view of recent interest in machine learning on DPPs with non-symmetric kernels, we present
here a concentration inequality for such DPPs. We propose a novel approach to control the Laplace
transform in the non-symmetric case (which can also be applied to the symmetric setting). As a
trade-off, the range for ε becomes a bit smaller. For simplicity, we present the result for a finite
ground set, but the proof applies more generally.
Theorem 2 (Non-symmetric kernels). Let S be a DPP on a finite set X = {x1, . . . , xn} with a
non-symmetric kernel K. Then for any bounded test function φ : X →R, we have

P(|Λ(φ)−E[Λ(φ)]| ≥ε) ≤2 exp

−
ε2

4 Var [Λ(φ)]


, ∀0 ≤ε ≤
Var [Λ(φ)]2

40∥φ∥3∞· max(1, ∥K∥2op) · ∥K∥∗
,

where ∥· ∥op denotes the spectral norm and ∥· ∥∗denotes the nuclear norm of a matrix.
Remark 2.1. For simplicity, we will use the concentration inequality in Theorem 1 from now on.
However, we keep in mind that we always can apply Theorem 2 to deduce analogous results for
non-symmetric kernels.

We conclude with a concentration inequality for linear statistics of vector-valued functions.
Theorem 3 (Vector-valued statistics). Let S be a DPP on a Polish space X with reference measure
µ and Hermitian kernel K. Let Φ = (φ1, . . . , φp)⊤: X →Rp be a vector-valued test function,
and we denote by Λ(Φ), V(Φ) the vectors (Λ(φi))p
i=1 and (Var [Λ(φi)]1/2)p
i=1, respectively. Let
∥x∥2
ω := Pp
i=1 ω2
i |xi|2 be a weighted norm on Rp for some weights ω1, . . . ωp ≥0. Then, for some
universal constant A > 0, we have

P(∥Λ(Φ) −E[Λ(Φ)]∥ω ≥ε) ≤2p exp

−
ε2

4A∥V(Φ)∥2ω


,

for 0 ≤ε ≤2A∥V(Φ)∥ω

3
min1≤i≤p
√

Var[Λ(φi)]

∥φi∥∞
·

DPPs for coresets.
We demonstrate the effectiveness of concentration inequalities for linear
statistics of DPPs in the coresets problem, achieving uniform approximation guarantees over function
classes. To accommodate as many ML settings as possible, we shall consider two natural types of
function classes: vector spaces of functions (A.1) and parametrized function spaces (A.2).

For vector spaces of functions, we assume that

dim spanR(F) = D < ∞for some D.
(A.1)

This assumption covers common situations like linear regression in Example 2, where we observe
that each f ∈F is a quadratic function in (d + 1) variables. Thus the dimension of the linear span of
F is at most (d+1)2 +(d+1)+1. Another popular class of queries, originating in signal processing

5


---Page Break---
problems, is the class of band-limited functions. A function f : Td 7→R (where Td denotes the
d-dimensional torus) is said to be band-limited if there exists B ∈N such that its Fourier coefficients
ˆf(k1, . . . , kd) = 0 whenever there is a kj such that |kj| > B. It is easy to see that the space F of
B-bandlimited functions satisfies dim F ≤(2B + 1)d.

Another common scenario is when F is parametrized by a finite-dimensional parameter space:

F = {fθ : θ ∈Θ}, where Θ is a bounded subset of RD for some D,
(A.2)

∥fθ −fθ′∥∞≤ℓ∥θ −θ′∥for some ℓ> 0, uniformly on Θ.
(A.3)

Conditions (A.2) and (A.3) cover e.g. the k-means problem of Example 1, as well as (non-)linear
regression settings. For k-means, for instance, each query is parametrized by its cluster centers
C = {q1, . . . , qk}, which can be viewed as a parameter (q1, . . . , qk) ∈Rkd.

Finally, with the idea in mind to derive multiplicative coresets from additive ones, we note that since
L(f) is typically of order n (for any f whose effective support covers a positive fraction of the ground
set), it is natural to assume that

1
n|L(f)| ≥c, for some c > 0, uniformly on F.
(A.4)

Theorem 4. Let S be a DPP with a Hermitian kernel K on a finite set X = {x1, . . . , xn} and
m = E[|S|]. Assume that for all i ∈{1, . . . , n}, Kii ≥ρ · m/n for some ρ > 0 not depending on
m, n. Let V ≥supf∈F Var

n−1LS(f)

. Under (A.1) and (A.4),

P

∃f ∈F :
LS(f)

L(f) −1
 ≥ε

≤2 exp

6D −c2ε2

16AV


,
0 ≤ε ≤
4AρmV
3c supf∈F ∥f∥∞
·

Assuming (A.2), (A.3), (A.4) and |S| ≤B · m a.s. for some B > 0, we have

P

∃f ∈F :
LS(f)

L(f) −1
 ≥ε

≤2 exp

CD −D log ε −c2ε2

16AV


, 0 ≤ε ≤
4AρmV
3c supf∈F ∥f∥∞
·

Here A > 0 is a universal constant and C = C(Θ, B, ρ, ℓ, c) > 0 is some constant.

Remark 4.1. For a bounded query f, Var

n−1LS(f)

= O(m−1) for i.i.d. sampling. In compari-
son, sampling with DPPs often yields smaller variance for linear statistics, in O(m−(1+δ)) for some
δ > 0; see Section 3 for an example. Thus, the upper bound for the range of ε for which we could
use our concentration result is O(m−δ). Plugging in ε = m−α for α ≥δ gives the upper bounds
2 exp(6D −C′m1+δ−2α) and 2 exp(CD + αD log m −C′m1+δ−2α) respectively (C and C′ are
some positive constants independent of m and n), which both converge to 0 as m →∞as long as
α < (1 + δ)/2. In other words, the accuracy rate ε can be chosen to be as small as m−1/2−δ′/2, for
any 0 < δ′ < δ, which is strictly smaller than the best accuracy rate m−1/2 of i.i.d. sampling.

Remark 4.2. For i.i.d. sampling S with expected size m, P(x ∈S) = m/n for all x ∈X. For a
DPP S with kernel K, one has P(xi ∈S) = Kii. Thus, assuming that for all i, Kii ≥ρ · m/n for
some ρ > 0 means that every point in the dataset X should have a reasonable chance to be sampled.
This also guarantees that the estimated loss LS(f) = P

x∈S f(x)/K(x, x) will not blow up, where
for x = xi ∈X, we write K(x, x) for Kii.

Remark 4.3. For the parametrized function spaces, the assumption |S| ≤B · m a.s. is not
strictly necessary, and is introduced here only for the sake of simplicity in presenting the results. A
version of Theorem 4 without this assumption will be discussed in Appendix A.4. In fact, we only
need n−1 P

x∈S K(x, x)−1 to be bounded with high probability, which follows from the condition
K(x, x) ≥ρ · m/n and the fact that |S| is highly concentrated around its mean m.

Remark 4.4. However, we remark that the assumption |S| ≤B · m a.s. holds for most kernels of
interest; DPPs with projection kernels being typical and significant examples. In machine learning
terms, it entails that the coresets are not much bigger than their expected size m; whereas in practice,
sampling schemes typically produce coresets of a fixed size (such as with projection DPPs).

Remark 4.5. It is straightforward to derive a version for additive coresets from Theorem 1. In fact,
we will not need assumption (A.4) in the additive setting.

6


---Page Break---
For the coresets problem for vector-valued functions, let F consist of f : X →Rp. For each f ∈F,
we denote by LS(f), L(f) and V(f) the vectors in Rp whose i-coordinates are LS(fi), L(fi) and
Var

n−1LS(fi)
1/2, respectively. Let ∥x∥2
ω := Pp
i=1 ω2
i |xi|2 be a weighted norm on Rp.

Theorem 5. Let S be a DPP as in Theorem 4. Let V ≥supf∈F max1≤i≤p ω2
i Var

n−1LS(fi)

.
Assuming (A.1), then

P

∃f ∈F : 1

n∥LS(f) −L(f)∥ω ≥ε

≤2p exp

6D −c2ε2

16AV


,

where 0 ≤ε ≤4AρmV

3c
· (supf∈F maxi=1,...,p ∥fi∥∞)−1.

Application: Discretized multivariate OPE.
We revisit Example 5. It has been shown in (Bardenet,
Ghosh, et al., 2021) that sampling with the DPP S constructed in this example yields significant
variance reduction for a wide class of linear statistics on X. To be more precise, in their setting,
X = {x1, . . . , xn} is a random data set, where xi’s are i.i.d. samples from a distribution γ with
support inside a d-dimensional hypercube; ˜γ is a density estimator for γ and q(x)dx is a reference
measure on that hypercube. The DPP S is then defined by the kernel K in Example 5 w.r.t. the
empirical measure n−1 P

x∈X δx. We normalize the kernel by setting ˆK := n−1K, so that S is a
DPP with kernel ˆK w.r.t. the counting measure on X. Then, under some mild assumptions on ˜γ and
q(x)dx, with high probability in the data set X, we have Var

n−1LS(f)

= O(m−(1+1/d)) for any
test function f satisfying some mild regularity conditions. For more details, we refer the reader to
(Bardenet, Ghosh, et al., 2021).

The significant reduction on the variance of linear statistics motivates us to apply Theorem 4 to this
setting. We also remark that all assumptions on the kernel in Theorem 4 are satisfied for ˆK with high
probability in the data set X (see Appendix A.6). Let F be a family of test functions on X satisfying
regularity conditions as in Bardenet, Ghosh, et al., 2021. Then we can state:

Theorem 6. For ε = O(m−1/d), w.h.p. in the data set X, we have

PS

∃f ∈F :
LS(f)

L(f) −1
 ≥ε

≤2 exp

6D −C′ε2m1+1/d
,
assuming (A.1) ,

and

PS

∃f ∈F :
LS(f)

L(f) −1
 ≥ε

≤2 exp

CD−D log ε−C′ε2m1+1/d
, assuming (A.2), (A.3),

where PS indicates the randomness only in S and C, C′ > 0 do not depend on m, n.
Remark 6.1. Theorem 6 confirms the discussion in Remark 4.1 for this particular example of DPP.
More precisely, Theorem 6 implies that, with probability tending to 1, sampling with this DPP gives
|LS(f)/L(f) −1| ≤m−( 1

2 + 1

2d )−, ∀f ∈F, where ( 1

2 + 1

2d)−denotes any positive number strictly
smaller than 1

2 + 1

2d. Meanwhile, for i.i.d. sampling, the accuracy rate ε is at best m−1/2.
Remark 6.2. It may be noted that, DPPs being Hilbert space-based models, they interact well with
linear projection based methods. As such, our method can be applied on dimensionally reduced data,
wherein the d in Remark 6.1 can be taken to be the reduced dimension, which is usually quite small.
As such, the improvement in the approximation guarantees is substantial, especially for large scale
problems entailing large m.

4
Experiments

In this section, we compare randomized coresets for the k-means problem of Example 1, on different
datasets. One virtue of k-means as a benchmark is that asymptotically tight upper-bound on sensitivity
(5) can easily be computed (Bachem et al., 2017, Lemma 2.2).

Competing approaches.
We compare 6 different coreset samplers.4 Each sampler takes as input a
finite dataset X, an integer m, and sampler-specific parameters. It returns a random subset S ⊂X of

4The link to a GitHub repository is temporarily hidden for anonymity.

7


---Page Break---
cardinality m. For the associated weight function ω in (2), we always take the inverse of the marginal
probability of inclusion, i.e. ω(x) = 1/P(x ∈S).

The first two baselines use independent sampling. The uniform method returns m samples from
X, uniformly and without replacement, and runs in O(m). The second method, sensitivity, is
specific to the k-means problem. It corresponds to the classical sensitivity-based importance sampling
coreset of Langberg and Schulman, 2010 described in Section 2. It runs in O(nk + nm).

The rest of the methods use negative dependence. The third method, termed G-mDPP, uses an m-DPP
sampler where the likelihood kernel is a Gaussian kernel, with adjustable bandwidth denoted by h.
It is basically Algorithm 1 of Tremblay et al., 2019, except we do not approximate the likelihood
kernel using random features. We prefer avoiding approximations in this paper to isolatedly probe
the benefit of negative dependence, but our choice comes at the cost O(n3) of performing SVD as a
preprocessing, in addition to the usual O(nm2) sampling time. Similarly, we compute the marginal
probabilities of inclusion of m-DPPs exactly, via Equation (205) and Algorithm 7 of Kulesza and
Taskar, 2012. These costly steps will likely be approximated in real data applications; see the
discussion of complexity to Section 5. The fourth method, OPE, is the discretized OPE of Example 5.
We take q to be a product of univariate beta pdfs, with parameters tuned to match the marginal
moments of the dataset, as in (Bardenet, Ghosh, et al., 2021). We take ˜γ to be a kernel density
estimator (KDE) built on X, using the Epanechnikov kernel, with Scott’s bandwidth selection method,
as implemented in the scikit-learn package (Pedregosa et al., 2011). When KDE estimation
is precomputed as in our experiments, the method runs in O(nm2), and O(n2 + nm2) otherwise.
Note that there is no cubic power of n, as one can perform the eigenvalue thresholding in Example 1
by a reduced SVD of the m × n feature matrix (pk(xi)). The fifth method, termed Vdm-DPP, is
Algorithm 2 of Tremblay et al., 2019, which runs in O(nm2). It is an OPE in the sense of Example 4,
but where the reference measure µ is the discrete empirical measure of the dataset. Although we
have no result on how its linear statistics scale, its similarity with the discretized OPE, as well as
its numerical performance in the experiments of Tremblay et al., 2019, make us expect Vdm-DPP to
behave similarly to OPE. The sixth method, stratified, is a stratified sampling baseline limited
to the case where X ⊂[−1, 1]d and X is “well-spread". It partitions [−1, 1]d into a grid of m bins,
and then independently draws one element uniformly in the intersection of X with each bin. It is a
special case of projection DPP, which runs in O(nm) and has obvious pitfalls, like requiring that X
has a non-empty intersection with each bin, which is unlikely to be the case for non-uniformly spread
datasets and high dimensions. Yet, this is a simple solution that one would likely implement to probe
the benefits of negative dependence.

The performance metric.
To investigate the cardinality of a coreset for a given error, we let
QS denote the quantile function of sup |LS(f) −L(f)|/L(f), the supremum over all queries of
the relative error. Intuitively, QS(0.9) = 10−2 means that 90% of the sampled coresets have a
worst case relative error below 10−2. We shall look at how an estimated QS(0.9) varies with m,
especially its slope in log-log plots with respect to m. Now, the set F of all queries for k-means
in combinatorially large, even for small values of k. Therefore, each time we need to evaluate the
supremum of the relative error, we rather uniformly sample without replacement k elements of X,
100 times and independently, and we take the maximum value of the relative error among these
100 values. Moreover, for each method and each coreset size m, the quantile function QS(0.9) is
estimated by an empirical quantile over 100 independent coresets sampled for each value of m.

Results.
We first consider a synthetic dataset of n = 1024 data points, sampled uniformly and
independently in [−1, 1]d; see Figure 1a. We consider d = 2 for demonstration purposes, but we
have observed similar results for other small dimensions. Figure 1b depicts our estimate of QS(0.9)
as a function of the coreset size m, in log-log format. The two i.i.d. baselines decrease as m−1/2, as
expected. The stratified baseline, intuitively well-suited to uniformly-spread datasets, outperforms all
other methods with a m−1 rate, consistent with its known optimal variance reduction (Novak, 1988).
Finally, the m-DPP and the two DPPs also yield a faster decay, eventually outperforming the i.i.d.
baselines as m grows. This is expected for the discretized OPE, as it follows from the theoretical
results from Section 3; but it is interesting to see that the Gaussian m-DPP and the Vdm-DPP seem
to reach a similar m−3/4 fast rate. For the Gaussian m-DPP, however, the performance depends on
the value of the bandwidth of the Gaussian kernel: in Figure 1c, we see that the rate of decay can go
from i.i.d.-like to OPE-like as the bandwidth increases; this is expected from results like (Barthelmé
et al., 2023). Note that the color code of Figure 1c differs from other figures.

8


---Page Break---
−1.0
−0.5
0.0
0.5
1.0
−1.00

−0.75

−0.50

−0.25

0.00

0.25

0.50

0.75

1.00

data, n=1024
OPE, m=169

(a) Uniform dataset and OPE sample

100
101
102

sample size

10−1

100

0.90-quantile sup. rel. error

uniform
sensitivity
OPE
Vdm-DPP
G-mDPP, h=0.1
stratiﬁed

(b) QS(0.9) vs. coreset size m

100
101
102

sample size

10−1

100

0.90-quantile sup. rel. error

uniform
G-mDPP, h=0.01
G-mDPP, h=0.05
G-mDPP, h=0.10
G-mDPP, h=0.20
G-mDPP, h=0.30

(c) QS(0.9) vs. coreset size m

Figure 1: Results for the uniform dataset.

In the uniform dataset of Fig. 1a, the sensitivity function is almost flat, which makes sensitivity
behave like uniform. To give an edge to sensitivity, we now consider the trimodal dataset shown
in Fig. 2a, with an OPE sample superimposed. The performance of sensitivity improves; see
Figure 2b, while the determinantal samplers still outperform the independent ones thanks to a faster
decay. For this dataset, it is not easy to stratify, and we thus do not show results for stratified.
We note that the size of a marker placed at x is proportional to the corresponding weight 1/K(x, x)
in the estimator of the average loss. Equivalently, the marker size is inversely proportional to the
marginal probability of x being included in the DPP sample.

Finally, we consider the classical MNIST dataset, after a PCA of dimension 4. Figure 2c shows again
the faster decay of the performance metric for the two DPPs (OPE and Vdm-DPP), compared to the two
independent methods. However, the advantage progressively disappears as the dimension increases be-
yond 4 (unshown), as expected from the gain in variance of the discretized multivariate OPE, which be-
comes negligible when d ≫1; see Section 5 for suggestions on how to prove a dimension-independent
decay. The source code used in this work is available at github.com/hsimonfroy/DPPcoresets,
where DPP samplers are built upon the Python package DPPy (Gautier et al., 2019).

−1.0
−0.5
0.0
0.5
1.0
−1.00

−0.75

−0.50

−0.25

0.00

0.25

0.50

0.75

1.00
data, n=1023
OPE, m=169

(a) Trimodal dataset and OPE sample

100
101
102

sample size

10−1

100

0.90-quantile sup. rel. error

uniform
sensitivity
OPE
Vdm-DPP
G-mDPP, h=0.1

(b) QS(0.9) vs. coreset size m

100
101
102

sample size

100

0.90-quantile sup. rel. error

uniform
sensitivity
OPE
Vdm-DPP

(c) QS(0.9) vs. coreset size m

Figure 2: Results on other datasets.

5
Discussion

Limitations.
Our paper is a theoretical contribution, and our approach has several limitations
before it can be a practical addition to the coreset toolbox. The improvement over independent
sampling relies on a variance scaling for linear statistics of a particular DPP, which itself relies on
both 1) an Ansatz that the dataset was generated i.i.d. from some pdf γ with a large support, and
2) the availability of a good approximation to γ; see Section 3 and (Bardenet, Ghosh, et al., 2021).
While Item 1) is usually deemed to be reasonable in a wide range of situations, we solve Item 2) by

9


---Page Break---
relying on a kernel density estimator, which is costly to manipulate. Another limitation is that the
improvement over independent sampling is in 1/d and thus progressively vanishes as the dimension
increases. Finally, a classical caveat is that although tractable, sampling a DPP still costs O(nm2),
provided the kernel is available in diagonalized form.

Future work.
The limitations above set up a research program. In particular, an intriguing observa-
tion in our empirical studies is the comparative performance of various DPP-based coreset samplers;
several of them exhibit effective performance. While we have sharp theoretical guarantees for the
discretized OPE-based scheme, obtaining similar guarantees and parameter-tuning protocols for other
samplers, like m-DPPs, will be of great practical interest as they would bypass the need, e.g., for an
approximation to the data-generating mechanism γ. The DPP called Vdm-DPP in Section 4, which is
itself an OPE for a discrete measure, might be a bridge between OPEs and m-DPPs, as Vdm-DPP can
be seen as a limit of Gaussian m-DPPs (Barthelmé et al., 2023). On a more general note, improving
the computational complexity of sampling DPPs remains an active topic, and we should examine
which techniques, e.g. by leveraging low-rank structures, preserve the small coreset property. Any
breakthrough in the complexity of DPP sampling would also have salutary consequences for the
broader program of negative dependence as a toolbox for machine learning. On the dependence of the
rate to the dimension, we propose to investigate the impact of smoothness of the test functions on the
rate: in numerical integration with mixtures of DPPs, smoothness does bring dimension-independent
rates (Belhadji et al., 2020b). Finally, in a more theoretical direction, extending concentration
inequalities for linear statistics beyond the restricted range of ε appearing e.g. in Theorem 1 is a
mathematically challenging problem, with potential learning-theoretic consequences.

Acknowledgments and Disclosure of Funding

RB and HSO acknowledge support from ERC grant Blackjack (ERC-2019-STG-851866) and ANR
AI chair Baccarat (ANR-20-CHIA-0002). SG was supported in part by the MOE grants R-146-000-
250-133, R-146-000-312-114, A-8002014-00-00 and MOE-T2EP20121-0013. HST was supported
by the NUS Research Scholarship.

References

[1]
Olivier Bachem, Mario Lucic, and Andreas Krause. “Practical Coreset Constructions
for Machine Learning”. In: arXiv: Machine Learning (2017). URL: https : / / api .
semanticscholar.org/CorpusID:88517375.
[2]
Rémi Bardenet, Subhroshekhar Ghosh, and Meixia Lin. “Determinantal point processes
based on orthogonal polynomials for sampling minibatches in SGD”. In: Advances in Neural
Information Processing Systems 34 (2021), pp. 16226–16237.
[3]
Rémi Bardenet and Adrien Hardy. “Monte Carlo with Determinantal Point Processes”. In:
Annals of Applied Probability (2020). URL: https://hal.archives-ouvertes.fr/hal-
01311263.
[4]
Simon Barthelmé, Nicolas Tremblay, Konstantin Usevich, and Pierre-Olivier Amblard. “Deter-
minantal point processes in the flat limit”. In: Bernoulli 29.2 (2023), pp. 957–983.
[5]
Ayoub Belhadji, Rémi Bardenet, and Pierre Chainais. “A determinantal point process for
column subset selection”. In: Journal of machine learning research 21.197 (2020), pp. 1–62.
[6]
Ayoub Belhadji, Rémi Bardenet, and Pierre Chainais. “Kernel interpolation with continuous
volume sampling”. In: International Conference on Machine Learning. PMLR. 2020, pp. 725–
735.
[7]
Christophe Ange Napoléon Biscio and Frédéric Lavancier. “Contrast estimation for parametric
stationary determinantal point processes”. In: Scandinavian Journal of Statistics 44.1 (2017),
pp. 204–229.
[8]
Jonathan Breuer and Maurice Duits. “The Nevai condition and a local law of large numbers
for orthogonal polynomial ensembles”. In: Advances in Mathematics 265 (2013), pp. 441–484.
URL: https://api.semanticscholar.org/CorpusID:119731958.
[9]
Victor-Emmanuel Brunel. “Learning signed determinantal point processes through the principal
minor assignment problem”. In: Advances in Neural Information Processing Systems 31 (2018).

10


---Page Break---
[10]
Vincent Cohen-Addad, Kasper Green Larsen, David Saulpic, Chris Schwiegelshohn, and
Omar Ali Sheikh-Omar. “Improved Coresets for Euclidean k-Means”. English. In: Advances in
Neural Information Processing Systems 35 - 36th Conference on Neural Information Processing
Systems, NeurIPS 2022. Ed. by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho,
and A. Oh. Advances in Neural Information Processing Systems. Publisher Copyright: ©
2022 Neural information processing systems foundation. All rights reserved.; 36th Conference
on Neural Information Processing Systems, NeurIPS 2022 ; Conference date: 28-11-2022
Through 09-12-2022. Neural Information Processing Systems Foundation, 2022.
[11]
Michal Derezinski, Feynman Liang, and Michael Mahoney. “Bayesian experimental design
using regularized determinantal point processes”. In: International Conference on Artificial
Intelligence and Statistics. PMLR. 2020, pp. 3197–3207.
[12]
Michal Derezinski and Michael Mahoney. “Distributed estimation of the inverse Hessian by
determinantal averaging”. In: Advances in Neural Information Processing Systems 32. Ed. by
H. Wallach, H. Larochelle, A. Beygelzimer, F. d Alché-Buc, E. Fox, and R. Garnett. Curran
Associates, Inc., 2019, pp. 11401–11411.
[13]
Mike Gartrell, Victor-Emmanuel Brunel, Elvis Dohmatob, and Syrine Krichene. “Learning
nonsymmetric determinantal point processes”. In: Advances in Neural Information Processing
Systems 32 (2019).
[14]
Mike Gartrell, Insu Han, Elvis Dohmatob, Jennifer Gillenwater, and Victor-Emmanuel Brunel.
“Scalable learning and MAP inference for nonsymmetric determinantal point processes”. In:
arXiv preprint arXiv:2006.09862 (2020).
[15]
Guillaume Gautier, Guillermo Polito, Rémi Bardenet, and Michal Valko. “DPPy: DPP Sam-
pling with Python”. In: Journal of Machine Learning Research 20.180 (2019), pp. 1–7. URL:
http://jmlr.org/papers/v20/19-179.html.
[16]
Subhroshekhar Ghosh and Philippe Rigollet. “Gaussian determinantal processes: A new model
for directionality in data”. In: Proceedings of the National Academy of Sciences 117.24 (2020),
pp. 13207–13213.
[17]
Insu Han, Mike Gartrell, Elvis Dohmatob, and Amin Karbasi. “Scalable MCMC sampling
for nonsymmetric determinantal point processes”. In: International Conference on Machine
Learning. PMLR. 2022, pp. 8213–8229.
[18]
J. Ben Hough, Manjunath Krishnapur, Yuval Peres, and Bálint Virág. “Determinantal Processes
and Independence”. In: Probability Surveys 3 (2006). URL: https://doi.org/10.1214%
2F154957806000000078.
[19]
Lingxiao Huang, Jian Li, and Xuan Wu. “On Optimal Coreset Construction for Euclidean (k,z)-
Clustering”. In: Proceedings of the 56th Annual ACM Symposium on Theory of Computing.
STOC 2024. Vancouver, BC, Canada: Association for Computing Machinery, 2024, pp. 1594–
1604. ISBN: 9798400703836. DOI: 10.1145/3618260.3649707. URL: https://doi.org/
10.1145/3618260.3649707.
[20]
Kurt Johansson and Gaultier Lambert. “Gaussian and non-Gaussian fluctuations for meso-
scopic linear statistics in determinantal processes”. In: The Annals of Probability 46.3 (2018),
pp. 1201–1278. DOI: 10.1214/17-AOP1178. URL: https://doi.org/10.1214/17-
AOP1178.
[21]
Alex Kulesza and Ben Taskar. “Determinantal Point Processes for Machine Learning”. In:
Foundations and Trends® in Machine Learning 5.2–3 (2012), pp. 123–286. ISSN: 1935-8237.
DOI: 10.1561/2200000044. URL: http://dx.doi.org/10.1561/2200000044.
[22]
Michael Langberg and Leonard J. Schulman. “Universal epsilon-approximators for integrals”.
In: Proceedings of the 2010 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA).
2010, pp. 598–607. DOI: 10.1137/1.9781611973075.50. eprint: https://epubs.siam.
org/doi/pdf/10.1137/1.9781611973075.50. URL: https://epubs.siam.org/doi/
abs/10.1137/1.9781611973075.50.
[23]
Frédéric Lavancier, Jesper Møller, and Ege Rubak. “Determinantal Point Process Models
and Statistical Inference”. In: Journal of the Royal Statistical Society Series B: Statistical
Methodology 77.4 (Dec. 2014), pp. 853–877. ISSN: 1467-9868. DOI: 10.1111/rssb.12096.
URL: http://dx.doi.org/10.1111/rssb.12096.
[24]
Odile Macchi. “Processus ponctuels et coincidences – Contributions à l’étude théorique des
processus ponctuels, avec applications à l’optique statistique et aux communications optiques”.
PhD thesis. Université Paris-Sud, 1972.

11


---Page Break---
[25]
Odile Macchi. “The Coincidence Approach to Stochastic Point Processes”. In: Advances in
Applied Probability 7.1 (1975), pp. 83–122. ISSN: 00018678. URL: http://www.jstor.
org/stable/1425855 (visited on 05/22/2024).
[26]
Erich Novak. Deterministic and Stochastic Error Bounds in Numerical Analysis. Vol. 1349.
Lecture Notes in Mathematics. Berlin, Heidelberg: Springer, 1988. ISBN: 978-3-540-50368-2.
DOI: 10.1007/BFb0079792. URL: http://link.springer.com/10.1007/BFb0079792.
[27]
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P.
Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher,
M. Perrot, and E. Duchesnay. “Scikit-learn: Machine Learning in Python”. In: Journal of
Machine Learning Research 12 (2011), pp. 2825–2830.
[28]
Robin Pemantle and Yuval Peres. “Concentration of Lipschitz Functionals of Determinantal
and Other Strong Rayleigh Measures”. In: Combinatorics Probability and Computing 23 (Aug.
2011). DOI: 10.1017/S0963548313000345.
[29]
A. Soshnikov. “Determinantal random point fields”. In: Russian Mathematical Surveys 55
(2000), pp. 923–975.
[30]
Nicolas Tremblay, Simon Barthelmé, and Pierre-Olivier Amblard. “Determinantal point pro-
cesses for coresets”. In: Journal of Machine Learning Research 20.168 (2019), pp. 1–70.

12


---Page Break---
A
Appendix / supplemental material

A.1
Proof of Theorem 1

We present here the proof for the general setting, i.e., when X is a Polish space and S is a DPP with a
Hermitian kernel K w.r.t a background measure µ. By abuse of notation, we will also denote by K
the integral operator

K : L2(X, µ) →L2(X, µ)
,
f(x) 7→
Z

X
K(x, y)f(y)dµ(y).

We denote by Ck(φ), k ≥1 the cumulants of Λ(φ), i.e.

log E[etΛ(φ)] =
X

k≥1

Ck(φ)

k!
tk,
for t near 0.

Note that C1(φ) = E[Λ(φ)], C2(φ) = Var [Λ(φ)]. In general, we have the formula (see Johansson
and Lambert, 2018)

Ck(φ) =

k
X

q=1

(−1)q+1

q

X

k1,...,kq≥1
k1+...+kq=k

k!
k1! . . . kq!Tr[Φk1K . . . ΦkqK],
(7)

where Φ : L2(X, µ) →L2(X, µ) is the operator f(x) 7→φ(x)f(x).

By the Macchi-Soshnikov theorem (Macchi, 1972; Soshnikov, 2000), 0 ⪯K ⪯I, and we can write

C2(φ) = Tr[Φ(I −K)ΦK] = Tr[
√

KΦ(I −K)Φ
√

K] = ∥
√

I −KΦ
√

K∥2
HS,
where ∥· ∥HS denotes the Hilbert-Schmidt norm of an operator.
Lemma 1. For k ≥1, we have

∥
√

I −KΦk√

K∥HS ≤k∥φ∥k−1
∞∥
√

I −KΦ
√

K∥HS,
where ∥φ∥∞:= supx∈X |φ(x)|.

Proof. One has

∥
√

I −KΦk√

K∥2
HS
=
Tr[
√

KΦk(I −K)Φk√

K]

=
Tr[Φk(I −K)ΦkK]

=
Tr[Φ2kK] −Tr[ΦkKΦkK]

=
Z
φ(x)2kK(x, x)dµ(x) −
ZZ
φ(x)kK(x, y)φ(y)kK(y, x)dµ(x)dµ(y)

=
Z
φ(x)2k
K(x, x) −
Z
K(x, y)K(y, x)dµ(y)

dµ(x)

+
1
2

ZZ
(φ(x)k −φ(y)k)2K(x, y)K(y, x)dµ(x)dµ(y).

Since 0 ⪯K ⪯I, we have K2 ⪯K, which implies K(x, x) ≥
R
K(x, y)K(y, x)dµ(y) for µ-a.e.
x. Thus,
Z
φ(x)2k
K(x, x) −
Z
K(x, y)K(y, x)dµ(y)

dµ(x)

≤
∥φ∥2k−2
∞

Z
φ(x)2
K(x, x) −
Z
K(x, y)K(y, x)dµ(y)

dµ(x).

On the other hand, by the symmetry of K, we have K(x, y)K(y, x) = |K(x, y)|2 ≥0 for all
x, y ∈X. Note that

|φ(x)k −φ(y)k| = |φ(x) −φ(y)|


k−1
X

j=0
φ(x)jφ(y)k−1−j ≤k∥φ∥k−1
∞|φ(x) −φ(y)|.

Combining all ingredients, we deduce that ∥
√

I −KΦk√

K∥2
HS ≤k2∥φ∥2k−2
∞
∥
√

I −KΦ
√

K∥2
HS,
as desired.

13


---Page Break---
Lemma 2. For k ≥3, we have

|Ck(φ)|

k!
≤
1
√

2π ekk3/2∥φ∥k−2
∞C2(φ).

Proof. We recall the formula (7), observe that

k
X

q=1

(−1)q+1

q

X

k1,...,kq≥1
k1+...+kq=k

k!
k1! . . . kq! = 0.

Then one can write

Ck(φ)
=

n
X

q=1

(−1)q+1

q

X

k1,...,kq≥1
k1+...+kq=k

k!
k1! . . . kq!


Tr[Φk1K . . . ΦkqK] −Tr[ΦkK]


=

n
X

q=2

(−1)q+1

q

X

k1,...,kq≥1
k1+...+kq=k

k!
k1! . . . kq!


Tr[Φk1K . . . ΦkqK] −Tr[ΦkK]

.

For any k1, . . . , kq ≥1 such that k1 + . . . + kq = k, we observe that

|Tr[Φk1K . . . Φkq−2KΦkq−1+kqK] −Tr[Φk1K . . . Φkq−2KΦkq−1KΦkqK]|

=
|Tr[Φk1K . . . Φkq−2KΦkq−1(I −K)ΦkqK]|

=
|Tr[
√

KΦk1K . . . Φkq−2√

K
√

KΦkq−1√

I −K
√

I −KΦkq√

K]|

≤
∥
√

KΦk1K . . . Φkq−2√

K
√

KΦkq−1√

I −K∥HS · ∥
√

I −KΦkq√

K∥HS
≤
∥
√

KΦk1K . . . Φkq−2√

K∥op · ∥
√

KΦkq−1√

I −K∥HS · ∥
√

I −KΦkq√

K∥HS

≤
∥
√

KΦk1K . . . Φkq−2√

K∥op · kq−1kq∥φ∥kq−1+kq−2
∞
∥
√

I −KΦ
√

K∥2
HS
≤
kq−1kq∥φ∥k−2
∞C2(φ),

here we used Lemma 1, the fact that 0 ⪯K ⪯I ,∥Φ∥op = ∥φ∥∞and the ∥· ∥op norm is
submultiplicative. Since kj ≤k for all 1 ≤j ≤q, using a telescoping argument gives

|Tr[Φk1K . . . ΦkqK] −Tr[ΦkK]| ≤qk2∥φ∥k−2
∞C2(φ).

Hence

|Ck(φ)| ≤

k
X

q=2

X

k1,...,kq≥1
k1+...+kq=k

k!
k1! . . . kq!k2∥φ∥k−2
∞C2(φ).

Now observe that for k ≥3

k
X

q=2

X

k1,...,kq≥1
k1+...+kq=k

1
k1! . . . kq! < kk

k! ≤
ek
√

2πk
·

Thus,
|Ck(φ)|

k!
≤
1
√

2π ekk3/2∥φ∥k−2
∞C2(φ).

Combining all ingredients above, one can show that.
Lemma 3. For |t| ≤1/(3∥φ∥∞), we have

| log E[etΛ(φ)] −tE[Λ(φ)]| ≤At2 Var [Λ(φ)] ,

where A > 0 is an universal constant.

14


---Page Break---
Proof. For |t| ≤
1
3∥φ∥∞, we have

| log E[etΛ(φ)] −tE[Λ(φ)]|
=

X

k≥2

Ck(φ)

k!
tk

≤
X

k≥2

|Ck(φ)|

k!
|t|k

≤
|t|2C2(φ)
1

2 +
X

k≥3

1
√

2π ekk3/2∥φ∥k−2
∞|t|k−2

≤
|t|2C2(φ)
1

2 +
1
√

2π e2 X

k≥3
k3/2(e/3)k−2

=
At2 Var [Λ(φ)] ,

where A > 0 is some universal constant.

We can finish the proof of Theorem 1 as follows.

Proof of Theorem 1. Let ε > 0. We have

log P(Λ(φ) −E[Λ(φ)] ≥ε)
≤
inf
t


log E[etΛ(φ)] −tE[Λ(φ)] −tε


≤
inf
t


−tε + t2A Var [Λ(φ)]


where the infimum is taken on t ∈(0, 1/(3∥φ∥∞)].

For 0 ≤ε ≤2A Var[Λ(φ)]

3∥φ∥∞
, choosing

t0 =
ε
2A Var [Λ(φ)] ≤
1
3∥φ∥∞

gives

log P(Λ(φ) −E[Λ(φ)] ≥ε) ≤−
ε2

4A Var [Λ(φ)]

as desired.

A.2
Proof of Theorem 2

Denote by Φ the diagonal matrix Diag(φ) ∈Rn×n. For each t ∈R, we define

Gt := I −exp(tΦ) = −

∞
X

k=1

tk

k!Φk.

By the Campbell formula, we have E[etΛ(φ)] = det[I −GtK], t ∈R. By choosing t ≥0 small
enough such that ∥GtK∥op < 1, one can expand

log det[I −GtK] = −

∞
X

k=1

1
k Tr[(GtK)k].

Observe that ∥Gt∥op ≤e|t|∥φ∥∞−1 ≤2|t|∥φ∥∞for all |t| ≤1/(3∥φ∥∞). From now on, we will
consider t ≥0 such that

0 ≤t∥φ∥∞M ≤1

3,

where M := max(∥K∥op, 1). This choice for t will particularly imply that ∥GtK∥op ≤2/3.

15


---Page Break---
For k = 1, we have

−Tr[GtK]
=

∞
X

p=1

tp

p!Tr[ΦpK]

≤
tE[Λ(φ)] + t2

2 Tr[Φ2K] +
X

p≥3

tp

p!|Tr[ΦpK]|

≤
tE[Λ(φ)] + t2

2 Tr[Φ2K] +
X

p≥3

tp

p!∥φ∥p
∞∥K∥∗

≤
tE[Λ(φ)] + t2

2 Tr[Φ2K] + t3∥φ∥3
∞∥K∥∗.

For k = 2, we have

−1

2Tr[(GtK)2]
=
−1

2

X

p,q≥1

tp+q

p!q! Tr[ΦpKΦqK]

=
−t2

2 Tr[ΦKΦK] −1

2

X

p+q≥3

tp+q

p!q! Tr[ΦpKΦqK]

≤
−t2

2 Tr[ΦKΦK] + 1

2

X

l≥3

tl

l! 2l∥φ∥l
∞∥K∥op∥K∥∗

≤
−t2

2 Tr[ΦKΦK] + t3∥φ∥3
∞∥K∥∗∥K∥op.

For k ≥3, we observe that

|Tr[(GtK)k]| ≤∥(GtK)k∥∗≤∥GtK∥k−3
op ∥Gt∥3
op∥K∥2
op∥K∥∗≤
2

3

k−3
(2t∥φ∥∞)3∥K∥2
op∥K∥∗.

Thus
X

k≥3

1
k |Tr[(GtK)k]| ≤8t3∥φ∥3
∞∥K∥2
op∥K∥∗.

Combining all ingredients, we deduce that

log E[etΛ(φ)]
≤
tE[Λ(φ)] + t2

2 Var [Λ(φ)] + t3∥φ∥3
∞∥K∥∗(1 + ∥K∥op + 8∥K∥2
op)

≤
tE[Λ(φ)] + t2

2 Var [Λ(φ)] + 10t3∥φ∥3
∞∥K∥∗M 2.

Let ε > 0. We have

log P(Λ(φ) −E[Λ(φ)] ≥ε)
≤
inf
t


log E[etΛ(φ)] −tE[Λ(φ)] −tε


≤
inf
t


−tε + t2

2 Var [Λ(φ)] + t3 · 10∥φ∥3
∞∥K∥∗M 2

where the infimum is taken on t ∈(0, 1/(3∥φ∥∞M)].

For 0 ≤ε ≤
Var[Λ(φ)]2

40∥φ∥3∞∥K∥∗M 2 , we choose

t0 =
ε
Var [Λ(φ)] ≤
Var [Λ(φ)]
40∥φ∥3∞∥K∥∗M 2 ≤
2M∥φ∥2
∞∥K∥∗
40∥φ∥3∞∥K∥∗M 2 <
1
3∥φ∥∞M ·

This choice yields

log P(Λ(φ) −E[Λ(φ)] ≥ε) ≤−
ε2

2 Var [Λ(φ)] + t3
0 · 10∥φ∥3
∞∥K∥∗M 2.

16


---Page Break---
Note that

t3
0 · 10∥φ∥3
∞∥K∥∗M 2 =
ε3

Var [Λ(φ)]3 · 10∥φ∥3
∞∥K∥∗M 2 ≤
ε2

4 Var [Λ(φ)]·

This implies

log P(Λ(φ) −E[Λ(φ)] ≥ε) ≤−
ε2

4 Var [Λ(φ)]
as desired.

A.3
Proof of Theorem 3

Proof of Theorem 3. By a scaling argument, it suffices to prove for the case ω1 = . . . = ωp = 1. We
have

P(∥Λ(Φ) −E[Λ(Φ)]∥2 ≥ε)
=
P

p
X

i=1
|Λ(φi) −E[Λ(φi)]|2 ≥ε2

=
P

p
X

i=1
|Λ(φi) −E[Λ(φi)]|2 ≥ε2
p
X

i=1

Var [Λ(φi)]

∥V(Φ)∥2
2



≤

p
X

i=1
P

|Λ(φi) −E[Λ(φi)]| ≥ε

p

Var [Λ(φi)]

∥V(Φ)∥2


·

For each 1 ≤i ≤p, applying Theorem 1 gives

P

|Λ(φi)−E[Λ(φi)]| ≥ε

p

Var [Λ(φi)]

∥V(Φ)∥2


≤2 exp

−
ε2

4A∥V(Φ)∥2
2


, ∀0 ≤ε ≤2A∥V(Φ)∥2
p

Var [Λ(φi)]
3∥φi∥∞
·

The theorem follows.

A.4
Proof of Theorem 4

Using (A.4), we deduce that
LS(f)

L(f) −1
 ≤1

cn|LS(f) −L(f)|,
∀f ∈F.

This implies

P

∃f ∈F :
LS(f)

L(f) −1
 ≥ε

≤P

∃f ∈F : 1

n|LS(f) −L(f)| ≥cε

.

Thus, it suffices to bound the RHS. For each f ∈F, let V ≥Var

n−1LS(f)

, we apply Theorem 1
for the linear statistic LS(f) = Λ(f/K) to obtain

P
 1

n|LS(f) −L(f)| ≥cε

≤2 exp

−c2ε2

4AV


,
∀0 ≤ε ≤
2AnV
3c∥f/K∥∞
·

Using K(x, x) ≥ρm/n, we deduce that the above inequality holds for any 0 ≤ε ≤2AρmV

3c∥f∥∞.

Proof of Theorem 4: Assuming (A.1). We let Fsym := {λf : |λ| ≤1, f ∈F}, and let B be the
convex hull of Fsym. Since B is a symmetric convex body in F, there exists a norm ∥· ∥F in F such
that B is the unit ball in (F, ∥· ∥F).

Define
L(f) := 1

n


LS(f) −L(f)

,
f ∈F,

then it is clear that L(f) is linear in f. Moreover, for any f, g ∈F, one has

|L(f) −L(g)| = |L(f −g)| = ∥f −g∥F ·
L

f −g
∥f −g∥F

 ≤∥f −g∥F sup
h∈B
|L(h)|.

17


---Page Break---
For each δ > 0, let Bδ be a δ-net for (B, ∥· ∥F). By definition of a δ-net, for any f ∈B, there exists
an f0 ∈Bδ such that ∥f −f0∥F ≤δ. Thus, for every f ∈B

|L(f)| ≤|L(f0)| + |L(f) −L(f0)| ≤|L(f0)| + δ sup
h∈B
|L(h)| ≤sup
g∈Bδ
|L(g)| + δ sup
h∈B
|L(h)|.

This implies supf∈B |L(f)| ≤
1
1−δ supf∈Bδ |L(f)|, ∀0 < δ < 1. In particular, choosing δ = 1/2
gives
sup
f∈B
|L(f)| ≤2 sup
f∈B1/2
|L(f)|.

Therefore

P

sup
f∈B
|L(f)| ≥cε

≤P

2 sup
f∈B1/2
|L(f)| ≥cε

= P

∃f ∈B1/2 : 1

n|LS(f) −L(f)| ≥cε/2

.

Let N(B, ∥· ∥F, 1/2) be the 1/2-covering number of B, then

P

∃f ∈B : 1

n|ˆLS(f) −L(f)| ≥cε

≤N(B, ∥· ∥F, 1/2) · 2e−c2ε2/16AV ,

for any

V ≥sup
f∈B
Var
 1

nLS(f)

,
0 ≤ε ≤
4AρmV
3c supf∈B ∥f∥∞
·

Note that for a finite dimensional normed vector space, for 0 < δ < 1, one has

N(B, ∥· ∥F, δ) ≤
3

δ

dim F
.

This implies

P

∃f ∈B : 1

n|LS(f) −L(f)| ≥cε

≤2 exp

6D −c2ε2

16AV


·
(8)

Since F ⊂B, it is clear that

P

∃f ∈F : |LS(f) −L(f)| ≥ncε

≤P

∃f ∈B : |LS(f) −L(f)| ≥ncε

.
(9)

On the other hand, for each f ∈B, there exist 0 ≤t ≤1, |λi| ≤1, fi ∈F, i = 1, 2 such that
f = tλ1f1 + (1 −t)λ2f2. Therefore

Var [LS(f)]1/2
=
Var [LS(tλ1f1) + LS((1 −t)λ2f2)]1/2

≤
Var [LS(tλ1f1)]1/2 + Var [LS((1 −t)λ2f2)]1/2

≤
t Var [LS(f1)]1/2 + (1 −t) Var [LS(f2)]1/2

≤
sup
g∈F
Var [LS(g)]1/2 .

Moreover,

∥f∥∞= sup
x∈X
|f(x)| ≤t sup
x∈X
|f1(x)| + (1 −t) sup
x∈X
|f2(x)| ≤sup
g∈F
∥g∥∞.

Thus,
sup
f∈B
Var [LS(f)] = sup
f∈F
Var [LS(f)]
,
sup
f∈B
∥f∥∞= sup
f∈F
∥f∥∞.
(10)

From (8), (9), (10), the theorem follows.

Proof of Theorem 4: Assuming (A.2), (A.3). We define L(θ) := 1

n(LS(fθ) −L(fθ)), θ ∈Θ. Then

P

∃f ∈F : 1

n|LS(f) −L(f)| ≥cε

= P(∃θ ∈Θ : |L(θ)| ≥cε) = P(sup
θ∈Θ
|L(θ)| ≥cε).

Using (A.3), we have |L(fθ) −L(fθ′)| ≤nℓ∥θ −θ′∥and

|LS(fθ) −LS(fθ′)| ≤ℓ∥θ −θ′∥
 X

x∈S

1
K(x, x)


≤ℓ∥θ −θ′∥nρ−1m−1|S|.
(11)

18


---Page Break---
This implies |L(θ) −L(θ′)| ≤C∥θ −θ′∥a.s., for some constant C depending on B, ρ, ℓ. Let Γ be a
cε
2C -net for Θ, then

sup
θ∈Θ
|L(θ)| ≤sup
θ′∈Γ
|L(θ′)| + cε

2 ·

Thus
P(sup
θ∈Θ
|L(θ)| ≥cε) ≤P(sup
θ′∈Γ
|L(θ′)| ≥cε/2)

We note that |Γ| = O(ε−D). This completes the proof.

Remark 6.3. Without the assumption |S| ≤B · m a.s., one can continue from (11) as follows.
Denote by λ1 ≥. . . ≥λn ≥0 the eigenvalues of K, it is known that |S| =d X1 + . . . + Xn, where
Xi ∼Ber(λi) are independent. Let B > 0, then using a multiplicative Chernoff bound for the sum
of independent Bernoulli variables gives

P(|S| > (B + 1)m) = P

n
X

i=1
Xi > (B + 1)m

≤exp

−
B2

B + 2m

·

By choosing B large, this event will have small probability. Meanwhile, on the event {|S| ≤
(B + 1)m}, we can use exactly the same argument as in the proof above.

A.5
Proof of Theorem 5

Proof of Theorem 5. It suffices to show for the case ω1 = . . . = ωp = 1. For each f ∈F, by
applying Theorem 1 and an union bound argument, we have

P
 1

n∥LS(f) −L(f)∥≥ε

≤2p exp

−c2ε2

4AV


,
∀0 ≤ε ≤
2AρmV
3c maxi ∥fi∥∞
·

Using the same argument as in the proof of Theorem 4 under assumption (A.1) gives the result.

A.6
Proof of Theorem 6

Proof of Theorem 6. We remark that Var

n−1LS(f)

= O(m−(1+1/d)) uniformly for all f ∈F,
w.h.p. in the data set X. Hence, Theorem 6 is a direct application of Theorem 4 with V =
Cm−(1+1/d) for some constant C > 0. As we discussed in the Remark 4.1, the range for ε is
O(m−1/d). Thus, it suffices to check the conditions on ˆK. Since ˆK is a projection of rank m,
|S| = m a.s. Moreover, we have n ˆK(x, x) = K(x, x), which is typically of order m, where we used
an uniform CLT result and an asymptotic for multivariate OPE kernels (see Bardenet, Ghosh, et al.,
2021 for more details).

19


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We claim theoretical results, which are established as theorems in the main
text.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We discuss limitations in Section 5.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: We give our assumptions and results in Section 3.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We give experimental details in Section 4.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We will release code to reproduce the experimental section in a public GitHub
repository upon acceptance. We stress that our contributions are the theoretical results in the
main paper; the code is there only to support our theoretical claims.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: We give details in Section 4.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: Our metric is a quantile of a worst-case error; there is no standard error bar.
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: Our experiments are small-scale and were performed on a personal computer;
we mention this in Section 4.

20


---Page Break---
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]

Justification: It is a theoretical paper, and we cannot see any potential harmful consequence.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: It is a theoretical paper, so societal impact is a long way downstream. A
positive impact we can see in the study of coresets is the possibility of saving energy when
training large models.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risk.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: we properly cite the Python libraries we use.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.

21


---Page Break---
