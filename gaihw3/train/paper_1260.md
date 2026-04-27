Benign overfitting in leaky ReLU networks with
moderate input dimension

Kedar Karhadkar1∗
Erin George1∗
Michael Murray1
Guido Montúfar12
Deanna Needell1
{kedar,egeo,mmurray,montufar,deanna}@math.ucla.edu
1UCLA
2Max Planck Institute for Mathematics in the Sciences
∗Equal contribution

Abstract

The problem of benign overfitting asks whether it is possible for a model to perfectly
fit noisy training data and still generalize well. We study benign overfitting in two-
layer leaky ReLU networks trained with the hinge loss on a binary classification
task. We consider input data that can be decomposed into the sum of a common
signal and a random noise component, that lie on subspaces orthogonal to one
another. We characterize conditions on the signal to noise ratio (SNR) of the model
parameters giving rise to benign versus non-benign (or harmful) overfitting: in
particular, if the SNR is high then benign overfitting occurs, conversely if the
SNR is low then harmful overfitting occurs. We attribute both benign and non-
benign overfitting to an approximate margin maximization property and show that
leaky ReLU networks trained on hinge loss with gradient descent (GD) satisfy
this property. In contrast to prior work we do not require the training data to be
nearly orthogonal. Notably, for input dimension d and training sample size n, while
results in prior work require d = Ω(n2 log n), here we require only d = Ω(n).

1
Introduction

Intuition from learning theory suggests that fitting noise during training reduces a model’s perfor-
mance on test data. However, it has been observed in some settings that machine learning models can
interpolate noisy training data with only nominal cost to their generalization performance (Zhang
et al., 2017; Belkin et al., 2018, 2019), a phenomenon referred to as benign overfitting. Establishing
theory that can explain this phenomenon has attracted much interest in recent years and there is now
a rich body of work on this topic particularly in the context of linear models. However, the study of
benign overfitting in the context of non-linear models, in particular shallow ReLU or leaky ReLU
networks, has additional technical challenges and subsequently is less well advanced.

Much of the effort in regard to theoretically characterizing benign overfitting focuses on showing,
under an appropriate scaling of the dimension of the input domain d, size of the training sample
n, number of corruptions k and number of model parameters p that a model can interpolate noisy
training data while achieving an arbitrarily small generalization error. Such characterizations of benign
overfitting position it as a high dimensional phenomenon1: indeed, the decrease in generalization
error is achieved by escaping to higher dimensions at some rate relative to the other aforementioned
hyperparameters. However, for these mathematical results to be relevant for explaining benign
overfitting as observed in practice, clearly the particular scaling of d with respect to n, k and p needs
to reflect the ratios seen in practice. Although a number of works, which we discuss in Section
1.2, establish benign overfitting results for shallow neural networks, a key and significant limitation
they share is the requirement that the input features of the training data are at least approximately

1We provide a formal definition of benign overfitting as a high dimensional phenomenon in Appendix E.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
orthogonal to one another. To study benign overfitting, these prior works typically assume the input
features consist of a small, low-rank signal component plus an isotropic noise term. Therefore, for
the near orthogonality property to hold with high probability it is required that the input dimension d
scales as d = Ω(n2 log n) or higher. This assumption highly restricts the applicability of these results
for explaining benign overfitting in practice.

In this work we assume only d = Ω(n) and establish both harmful and benign overfitting results for
shallow leaky ReLU networks trained via gradient descent (GD) on the hinge loss. In particular, we
consider n data point pairs (xi, yi) ∈Rd × {±1}, where, for some vector v ∈Sd−1 and scalar γ ∈
[0, 1], the input features are drawn from a pair of Gaussian clusters xi ∼N(±√γv, √1 −γ 1

d(Id −
vvT )) and yi = sign(⟨E[xi], v⟩). The training data is noisy in that k of the n points in the training
sample have their output label flipped. We assume equal numbers of positive and negative points
among clean and corrupt ones. We provide a full description of our setup and assumptions in Section
2. Our proof techniques are novel and identify a new condition allowing for the analysis of benign
and harmful overfitting which we term approximate margin maximization, wherein the norm of the
network parameters is upper bounded by a constant of the norm of the max-margin linear classifier.

1.1
Summary of contributions

Our key results are summarized as follows.

• In Theorem 3.1, we prove that a leaky ReLU network trained on linearly separable data with
gradient descent and the hinge loss will attain zero training loss in finitely many iterations.
Moreover, the network weight matrix W at convergence will be approximately max-margin
in the sense that ∥W ∥= O

∥w∗∥
α√m

, where α is the leaky parameter of the activation
function, m is the width of the network, and w∗is the max-margin linear classifier. We
apply this result to derive generalization bounds for the network on test data.

• In Theorem 3.2, we establish conditions under which benign overfitting occurs for leaky
ReLU networks. If the input dimension d, number of training points n, number of corrupt
points k, and signal strength γ satisfy d = Ω(n) and γ = Ω( 1

k), then the network will
exhibit benign overfitting. We emphasize that existing works on benign overfitting require
d = Ω(n2 log n) to ensure nearly orthogonal data.

• In Theorem 3.3, we find a complementary lower bound for the generalization error to show
that, for gradient descent classifiers, the bound in Theorem 3.2 is tight up to a constant in
the exponent that can depend on α.

• In Theorem 3.4, we find conditions under which non-benign overfitting occurs. If d = Ω(n)
and γ = O( 1

d), then the network will exhibit non-benign overfitting: in particular its
generalization error will be at least 1

8.

1.2
Related work

There is now a significant body of literature theoretically characterizing benign overfitting in the
context of linear models, including linear regression (Bartlett et al., 2020; Muthukumar et al., 2020;
Wu & Xu, 2020; Zou et al., 2021; Hastie et al., 2022; Koehler et al., 2021; Wang et al., 2021a;
Chatterji & Long, 2022; Shamir, 2022), logistic regression (Chatterji & Long, 2021; Muthukumar
et al., 2021; Wang et al., 2021b), max-margin classification with linear and random feature models
(Montanari et al., 2023b,a; Mei & Montanari, 2022; Cao et al., 2021) and kernel regression (Liang
& Rakhlin, 2020; Liang et al., 2020; Adlam & Pennington, 2020). However, the study of benign
overfitting in non-linear models is more nascent.

Homogeneous networks trained with gradient descent and an exponentially tailed loss are known to
converge in direction to a Karush-Kuhn-Tucker (KKT) point of the associated max-margin problem
(Lyu & Li, 2020; Ji & Telgarsky, 2020)2. This property has been widely used in prior works to
prove benign overfitting results for shallow neural networks. Frei et al. (2022) consider a shallow,
smooth leaky ReLU network trained with an exponentially tailed loss and assume the data is drawn
from a mixture of well-separated sub-Gaussian distributions. A key result of this work is, given

2One also needs to assume initialization from a position with a low initial loss.

2


---Page Break---
sufficient iterations of GD, that the network will interpolate noisy training data while also achieving
minimax optimal generalization error up to constants in the exponents. Xu & Gu (2023) extend
this result to more general activation functions, including ReLU, as well as relax the assumptions
on the noise distribution to being centered with bounded logarithmic Sobolev constant, and finally
also improve the convergence rate. George et al. (2023) also study ReLU as opposed to leaky ReLU
networks but do so in the context of the hinge loss, for which, and unlike exponentially tailed losses,
a characterization of the implicit bias is not known. This work also establishes transitions on the
margin of the clean data driving harmful, benign and no-overfitting training outcomes. Frei et al.
(2023) use the aforementioned implicit bias of GD for linear classifiers and shallow leaky ReLU
networks towards solutions that satisfy the KKT conditions of the margin maximization problem to
establish settings where the satisfaction of said KKT conditions implies benign overfitting. Kornowski
et al. (2023) also use the implicit bias results for exponentially tailed losses to derive similar benign
overfitting results for shallow ReLU networks. Cao et al. (2022); Kou et al. (2023) study benign
overfitting in two-layer convolutional as opposed to feedforward neural networks: indeed, whereas in
most prior works data is modeled as the sum of a signal and noise component, in these two works the
signal and noise components are assumed to lie in disjoint patches. The weight vector of each neuron
is applied to both patches separately and a non-linearity, such as ReLU, is applied to the resulting
pre-activation. In this setting, the authors prove interpolation of the noisy training data and derive
conditions on the clean margin under which the network benignly versus harmfully overfits. A follow
up work (Chen et al., 2023) considers the impact of Sharpness Aware Minimization (SAM) in the
same setting. Finally, and assuming d = Ω(n5), Xu et al. (2024) establish benign overfitting results
for a data distribution which, instead of being linearly separable, is separated according to an XOR
function.

We emphasize that the prior work on benign overfitting in the context of shallow neural networks
requires the input data to be approximately orthogonal. Under standard data models studied this
equates to the requirement that the input dimension d versus the size of the training sample n satisfies
d = Ω(n2 log n) or higher. Here we require only d = Ω(n). The weaker dimensionality requirement
requires substantially different proof techniques. George et al. (2023) study a setting most similar to
the one studied here, however, the techniques are very different. In particular, the results presented in
this other work are derived by carefully tracking neuron activation patterns. While in high dimensions
this is feasible due to the near orthogonality of the noise in low dimensions this is far more challenging
as noise vectors can be highly correlated leading to coupling effects.

Finally, we remark that our proof technique for the convergence of GD to a global minimizer in the
context of a shallow leaky ReLU network (Theorem 3.1) is closely related to the proof techniques
used by Brutzkus et al. (2018). While this work does establish a generalization bound, the bound
assumes that population dataset is linearly separable rather than just the training dataset. Hence, it
cannot be applied when the training dataset has label-flipping noise, which is the setting that we are
interested in for benign overfitting.

2
Preliminaries

Let [n] = {1, 2, . . . , n} denote the set of the first n natural numbers. We remark that when using
big-O notation we implicitly assume only positive constants. We use c, C, C1, C2, . . . to denote
absolute constants with respect to the input dimension d, the training sample size n, and the width
of the network m. Note constants may change in value from line to line. Furthermore, when using
big-O notation all variables aside from d, n, k and m are considered constants. However, for clarity
we will frequently make the constants concerning the confidence δ and failure probability ϵ explicit.
Moreover, for two functions f, g : N →N, if we say f = O(g) implies property p, what we mean is
there exists an N ∈N and a constant C such that if f(n) ≤Cg(n) for all n ≥N then property p
holds. Likewise, if we say f = Ω(g) implies property p, what we mean is there exists an N ∈N and
a constant C such that if f(n) ≥Cg(n) for all n ≥N then property p holds. Finally, we use ∥· ∥to
denote the ℓ2 norm of the vector argument or ℓ2 →ℓ2 operator norm of the matrix argument.

2.1
Data model

We study data generated as per the following data model.

3


---Page Break---
Definition 2.1. Suppose d, n, k ∈N, γ ∈(0, 1) and v ∈Sd−1. If (X, ˆy, y, x, y) ∼D(d, n, k, γ, v)
then

1. X ∈Rn×d is a random matrix whose rows, which we denote xi, satisfy xi = √γyiv +
√1 −γni, where ni ∼N(0d, 1

d(Id −vvT )) are mutually i.i.d..

2. y ∈{±1}n is a random vector with entries yi that are mutually independent of one another
as well as the noise vectors (ni)i∈[n] and are uniformly distributed over {±1}. This vector
holds the true labels of the training set.

3. Let B ⊂[n] be any subset chosen independently of y such that |B| = k. Then ˆy ∈{±1}n is
a random vector whose entries satisfy ˆyi ̸= yi for all i ∈B and ˆyi = yi for all i ∈Bc =: G.
This vector holds the observed labels of the training set.

4. y is a random variable representing a test label which is uniformly distributed over {±1}.

5. x ∈Rd is a random vector representing the input feature of a test point and satisfies
x = √γyv + √1 −γn, where n ∼N(0d, 1

d(Id −vvT )) is mutually independent of the
random vectors (ni)i∈[n].

We refer to (X, ˆy) as the training data and (x, y) as the test data. Furthermore, for typographical
convenience we define y ⊙ˆy =: β ∈{±1}n.

To provide some interpretation to Definition 2.1, the training data consists of n points of which
k have their observed label flipped relative to the true label. We refer to v and ni as the signal
and noise components of the i-th data point respectively: indeed, with γ > 0 then for i ∈G
yi⟨xi, v⟩= √γ > 0. The test data is drawn from the same distribution and is assumed not to be
corrupted. We say that the training data (X, ˆy) is linearly separable if there exists w ∈Rd such that

ˆyi⟨w, xi⟩≥1,
for all i ∈[n].

For finite n, this condition is equivalent to the existence of a w with ˆyi⟨w, xi⟩> 0 for all i ∈[n].
We denote the set of linearly separable datasets as Xlin ⊂Rn×d × {±1}n. For a linearly separable
dataset (X, ˆy), the max-margin linear classifier is the unique solution to the optimization problem

arg min
w∈Rd ∥w∥such that ˆyi⟨w, xi⟩≥1 for all i ∈[n].

Observe one may equivalently take a strictly convex objective ∥w∥2 and the constraint set is a
closed convex polyhedron that is non-empty iff the data is linearly separable. The max-margin linear
classifier w∗has a corresponding geometric margin 2/∥w∗∥. When d ≥n and γ > 0, input feature
matrices X from our data model almost surely have linearly independent rows xi and thus (X, ˆy) is
almost surely linearly separable for any observed labels ˆy ∈{±1}n.

2.2
Architecture and learning algorithm

We study shallow leaky ReLU networks with a forward pass function f : R2m×d × Rd →R defined
as

f(W , x) =

2m
X

j=1
(−1)jσ(⟨wj, x⟩),
(1)

where W ∈R2m×d are the parameters of the network, σ : R →R is the leaky ReLU function,
defined as σ(x) = max(x, αx), where α ∈(0, 1] is referred to as the leaky parameter. We remark
that we only train the weights of the first layer and keep the output weights of each neuron fixed.
Although σ is not differentiable at 0, in the context of gradient descent we adopt a subgradient and let
˙σ(z) = 1 for z ≥0 and let ˙σ(z) = α otherwise. The hinge loss ℓ: R →R≥0 is defined as

ℓ(z) = max{0, 1 −z}.
(2)

Again, ℓis not differentiable at zero; adopting a subgradient we define for any j ∈[2m]

∇wjℓ(ˆyf(W , x)) =
(−1)j+1ˆyx ˙σ(⟨wj, x⟩)
ˆyf(W , x) < 1,
0
ˆyf(W , x) ≥1.

4


---Page Break---
The training loss L : R2m×d × Rn×d × Rn →R is defined as

L(W , X, ˆy) =

n
X

i=1
ℓ(ˆyif(W , xi)).
(3)

Let W (0) ∈R2m×d denote the model parameters at initialization. For each t ∈N we define W (t)
recursively as
W (t) = W (t−1) −η∇W L(W (t−1), X, ˆy),
where η > 0 is the step size. Let F(t) ⊆[n] denote the set of all i ∈[n] such that ˆyif(W (t), xi) < 1.
Then equivalently each neuron is updated according to the following rule: for j ∈[2m]

w(t)
j
= GD(W (t−1), η) := w(t−1)
j
+ η(−1)j
X

i∈F(t−1)
ˆyixi ˙σ(⟨w(t−1)
j
, xi⟩).
(4)

For ease of reference we now provide the following definition of the learning algorithm described
above.
Definition 2.2. Let AGD : Rn×d ×{±1}n ×R×R2m×d →R2m×d return AGD(X, ˆy, η, W (0)) =:
W , where the j-th row wj of W is defined as follows: let w(0)
j
be the j-th row of W (0) and

generate the sequence (w(t)
j )t≥0 using the recurrence relation w(t)
j
= GD(W (t−1), η) as defined in
equation 4.

1. If for j ∈[2m], limt→∞w(t)
j
does not exist then we say AGD is undefined.

2. Otherwise we say AGD converges and wj = limt→∞w(t)
j .

3. If there exists a T ∈N such that for all j ∈[2m] w(t)
j
= w(T )
j
for all t ≥T, then we say
AGD converges in finite time.

We often find that all matrices in the set

{AGD(X, ˆy, η, W (0)) : ∀j ∥w(0)
j ∥≤λ}

agree on all relevant properties. In this case, we abuse notation and say that AGD(X, ˆy, η, λ) = W
where W is a generic element from this set.

Finally, in order to derive our results we make the following assumptions concerning the step size
and initialization of the network.
Assumption 1. The step size η satisfies η ≤1/(mn maxi∈[n] ∥xi∥2) and for all j ∈[2m] the

network at initialization satisfies ∥w(0)
j ∥≤√α/(m mini∈[n] ∥xi∥).

Under our data model the input data points have approximately unit norm; therefore these assumptions
reduce to η ≤
C
mn and ∥w(0)
j ∥≤C√α

m .

2.3
Approximate margin maximization

We now introduce the notion of an approximate margin maximizing algorithm, which plays a key
role in deriving our results. Although the primary setting we consider in this work is the learning
algorithm AGD (see Definition 2.2), we derive benign overfitting guarantees more broadly for any
learning algorithm which fits into this category. Recall Xlin denotes the set of linearly separable
datasets (X, ˆy) ∈Rn×d × {±1}n.
Definition 2.3. Let f : Rp × Rd →R denote a predictor function with p parameters. An algorithm
A : Rn×d × Rn →Rp is approximately margin maximizing with factor M > 0 on f if for all
(X, ˆy) ∈Xlin
ˆyif(A(X, ˆy), xi) ≥1 for all i ∈[n]
(5)
and
∥A(X, ˆy)∥≤M∥w∗∥,
(6)
where w∗is the max-margin linear classifier of (X, ˆy). Moreover, if A is an approximate margin
maximizing algorithm we define

|A| = inf{M > 0 : A is approximately margin maximizing with factor M}.
(7)

5


---Page Break---
In the above definition we take the standard Euclidean norm on Rp. In particular if Rp = R2m×d is a
space of matrices we take the Frobenius norm.

3
Main results

In order to prove benign overfitting it is necessary to show that the learning algorithm outputs a model
that correctly classifies all points in the training sample. The following theorem establishes this for
AGD and bounds the margin maximizing factor |AGD|.
Theorem 3.1. Let f : Rp × Rn →R be a leaky ReLU network with forward pass as defined by
equation 1. Suppose the step size η and initialization condition λ satisfy Assumption 1. Then for any
linearly separable data set (X, ˆy) AGD(X, ˆy, η, λ) converges after T iterations, where

T ≤C∥w∗∥2

ηα2m .

Furthermore AGD is approximately margin maximizing on f (Definition 2.3) with

|AGD| ≤
C
α√m.

A proof of Theorem 3.1 can be found in Appendix D.1. Note also by Definition 2.3 that the solution
W = AGD(X, ˆy) for (X, ˆy) ∈Xlin is a global minimizer of the training loss defined in equation 3
with L(W , X, ˆy) = 0. Our approach to proving this result is reminiscent of the proof of convergence
of the perceptron algorithm and therefore is also similar to the techniques used by Brutzkus et al.
(2018).

For training and test data as per Definition 2.1 we provide an upper bound on the generalization error
for approximately margin maximizing algorithms. For convenience we summarize our setting as
follows.
Assumption 2. Setting for proving generalization results.

• f : R2m×d × Rd →R is a shallow leaky ReLU network as per equation 1.

• A : Rn×d×{±1}n →R2m×d is a learning algorithm that returns the weights W ∈R2m×d
of the first layer of f.

• We let v ∈Sd−1 and consider training data (X, ˆy) and test data (x, y) distributed accord-
ing to (X, ˆy, y, x, y) ∼D(d, n, k, γ, v) as per Definition 2.1.

Under this setting we have the following generalization result for an approximately margin maximiz-
ing algorithm A. Note this result requires γ, and hence the signal to noise ratio of the inputs, to be
sufficiently large.
Theorem 3.2. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A is approximately
margin-maximizing (Definition 2.3). If n = Ω
 
log 1

δ

, d = Ω(n), k = O(
n
1+m|A|2 ), and γ = Ω
  1

k


then there is a fixed positive constant C such that with probability at least 1 −δ over (X, ˆy)

P(yf(W , x) ≤0 | X, ˆy) ≤exp

−C ·
d
k(1 + m|A|2)


.

A proof of Theorem 3.2 is provided in Appendix D.2. To comment informally on the relationship
between k and γ, we require γ = Ω(k−1) in order to guarantee that any network which achieves
zero hinge loss does so by focusing on the signal component v rather than the noise components ni.
We use the projection of the model weights onto the signal subspace as a measure of the strength
of the signal the model has learned and derive our generalization results based on this measure. In
Section 4 we provide a proof sketch of this framework in the simpler, linear model setting. Combining
Theorems 3.1, and 3.2 we arrive at the following benign overfitting result for shallow leaky ReLU
networks trained with GD on hinge loss.
Corollary 3.2.1. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A = AGD
where η, λ ∈R>0 satisfy Assumption 1. If n = Ω
 
log 1

δ

, d = Ω(n), k = O(α2n), and γ = Ω
  1

k


then the following hold.

6


---Page Break---
1. The algorithm AGD terminates almost surely after a finite number of updates. If W =
AGD(X, ˆy), then L(W , X, ˆy) = 0.

2. There is a fixed positive constant C such that, with probability at least 1−δ over the training
data (X, ˆy),

P(yf(W , x) ≤0 | X, ˆy) ≤exp

−C · α2d

k


.

We remark that the upper bound is at most exp (−Cd/n) for a different constant C as we assume
k = O(α2n).

If k is large enough, this bound is tight up to constants and factors of α in the exponent. This is given
by the following theorem, proven in Appendix D.2.
Theorem 3.3. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A = AGD where
η, λ ∈R>0 satisfy Assumption 1. If n = Ω(k), d = Ω(n), and k = Ω(log 1

δ + 1

α), then there is a
fixed positive constant C such that with probability at least 1 −δ over (X, ˆy)

P(yf(W , x) ≤0 | X, ˆy) ≥exp

−C · d

αk


.

In addition to this benign overfitting result we also provide the following non-benign overfitting result
for AGD. Note that conversely this result requires γ, and hence the signal to noise ratio of the inputs,
to be sufficiently small.
Theorem 3.4. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A = AGD,
where η, λ ∈R>0 satisfy Assumption 1. If n = Ω(1), d = Ω
 
n + log 1

δ

and γ = O

α3

d

then the
following hold.

1. The algorithm AGD terminates almost surely after finitely many updates. With W =
AGD(X, ˆy), L(W , X, ˆy) = 0.

2. With probability at least 1 −δ over the training data (X, ˆy)

P(yf(W , x) < 0 | X, ˆy) ≥1

8.

A proof of Theorem 3.4 is provided in Appendix D.3.

4
Approximate margin maximization and generalization: insight from linear
models

In this section we outline proofs for the analogues of Theorems 3.2 and 3.4 in the context of linear
models. The arguments are thematically similar and clearer to present. We provide complete proofs
of benign and non-benign overfitting for linear models in Appendix C.

An important lemma is the following, which bounds the largest and n-th largest singular values (σ1
and σn respectively) of the noise matrix N:
Lemma 4.1. Let N ∈Rn×d denote a random matrix whose rows are drawn mutually i.i.d. from
N(0d, 1

d(Id −vvT )). If d = Ω
 
n + log 1

δ

, then there exists constants C1 and C2 such that, with
probability at least 1 −δ,
C1 ≤σn(N) ≤σ1(N) ≤C2.

We prove this lemma in Appendix B using results from Vershynin (2018) and Rudelson & Vershynin
(2009). A consequence of this lemma is that with probability at least 1 −δ, the condition number of
N restricted to span N can be bounded above independently of all hyperparameters. For this reason,
we refer to the noise as being well-conditioned.

Now let w = A(X, ˆy) be the linear classifier returned by the algorithm. Observe that we can
decompose the weight vector into a signal and noise component

w = avv + z,

where z ⊥v and av ∈R. Based on this decomposition the proof proceeds as follows.

7


---Page Break---
1. Generalization bounds based on the SNR:
For test data as per the data model given in
Definition 2.1 we want to bound the probability of misclassification: in particular, we want to bound
the probability that
X := y⟨w, x⟩= √γav +
p

1 −γ⟨n, z⟩≤0.

As the noise is normally distributed, X ∼N
 √γav, 1−γ

d ∥z∥2
and the desired upper bound
therefore follows from Hoeffding’s inequality,

P(X ≤0) ≤exp

−
γda2
v
2(1 −γ)∥z∥2


.

Using Gaussian anti-concentration, we also obtain a lower bound for the probability of misclassifica-
tion:

P(y⟨w, x⟩≤0) ≥max

(
1
2 −

s

dγ
2π(1 −γ)
av
∥z∥, 1

4 exp

−6d

π
γ
1 −γ
a2
v
∥z∥2

)

.

2. Upper bound the norm of the max-margin classifier:
In order to use the approximate max-
margin property we require an upper bound on ∥w∗∥. As by definition ∥w∗∥≤∥˜w∥, it suffices to
construct a vector ˜w that interpolates the data and has small norm. Using that the noise matrix of the
data is well-conditioned with high probability, we achieve this by strategically constructing the signal
and noise components of ˜w. This yields the bound

∥w∗∥≤∥˜w∥≤C min

 r
n
1 −γ ,

s

1
γ +
k
1 −γ

!

,

where the arguments of the min function originate from a small and large γ regime respectively.

3. Lower bound the SNR using the approximate margin maximization property:
Based on step
1 the key quantity of interest from a generalization perspective is the ratio av/∥z∥, which describes
the signal to noise ratio (SNR) of the learned classifier. To lower bound this quantity we first lower
bound av. In particular, if av is small, then the only way to attain zero loss on the clean data is
for ∥z∥to be large. However, under appropriate assumptions on d, n, k and γ this can be shown to
contradict the bound ∥z∥≤∥w∥≤|A|∥w∗∥, and thus av must be bounded from below. A lower
bound on av/∥z∥then follows by again using ∥z∥≤∥w∗∥. Hence we obtain a lower bound for the
SNR and establish benign overfitting.

4. Upper bound the SNR using the zero loss condition:
For the generalization lower bound, we
compute an upper bound for the ratio av/∥z∥rather than a lower bound. Since the model perfectly
fits the training data with margin one,

1 ≤ˆyi⟨w, xi⟩= √γβiav +
p

1 −γˆyi⟨z, ni⟩

for all i ∈[n]. The above inequality implies that √1 −γˆyi⟨ni, z⟩is at least √γav for all corrupt
points. Since the noise is well-conditioned, this gives a lower bound on ∥z∥in terms of av and
hence an upper bound on the SNR av/∥z∥. By the second generalization lower bound in step 1, the
generalization error is bounded below at a similar exponential rate to the upper bound.

5. Upper bound the SNR using the zero loss condition and maximum margin property:
To
prove non-benign overfitting, we again compute an upper bound for the ratio av/∥z∥. We return to
the zero loss condition:

1 ≤ˆyi⟨w, xi⟩= √γβiav +
p

1 −γˆyi⟨z, ni⟩

for all i ∈[n]. If γ is small and |√γav| is large, then ∥w∥will be large, contradicting the approximate
margin maximization. Hence the above inequality implies that √1 −γˆyi⟨ni, z⟩is large for all i ∈[n].
Since the noise is well-conditioned, this can only happen when ∥z∥is large. This gives us a lower
bound on ∥z∥. As before, we can also upper bound av by ∥w∥, giving us an upper bound on the
SNR av/∥z∥. By the first generalization lower bound in step 1, the classifier generalizes poorly and
exhibits non-benign overfitting.

8


---Page Break---
4.1
From linear models to leaky ReLU networks

The proof of benign and non-benign overfitting in the linear case uses the tension between the two
properties of approximate margin maximization: fitting both the clean and corrupt points with margin
versus the bound on the norm. To extend this idea to a shallow leaky ReLU network as per equation 1,
we consider the same decomposition for each neuron j ∈[2m],

wj = ajv + zj,

where aj ∈R and zj ⊥v. In the linear case ±av can be interpreted as the activation of the linear
classifier on ±v respectively: in terms of magnitude the signal activation is the same in either case
and thus we measure the alignment of the linear model with the signal using |av|. For leaky ReLU
networks we define their activation on ±v respectively as A1 = f(W , v) and A−1 = −f(W , −v),
and then define the alignment of the network as Amin = min{A1, A−1}. Considering the alignment
of the network with the noise, then if Z ∈R2m×d denotes a matrix whose j-th row is zj, then
we measure the alignment of the network using ∥Z∥F . As a result, analogous to av/∥z∥, the key
ratio from a generalization perspective in the context of a leaky ReLU network is Amin/∥Z∥F . The
proof Theorems 3.2, 3.3, and 3.4 then follow the same outline as Steps 1-3 above but with additional
non-trivial technicalities.

5
Conclusion

In this work we have proven conditions under which leaky ReLU networks trained on binary
classification tasks exhibit benign and non-benign overfitting. We have substantially relaxed the
necessary assumptions on the input data compared with prior work; instead of requiring nearly
orthogonal data with d = Ω(n2 log n) or higher, we only need d = Ω(n). We achieve this by using
the distribution of singular values of the noise rather than specific correlations between noise vectors.
Our emphasis was on networks trained by gradient descent with the hinge loss, but we establish a
new framework that is general enough to accommodate any algorithm that is approximately margin
maximizing.

There are a few limitations of our results which would be natural questions to address in future
work. While we improve upon existing results in our dependence on the input dimension of the
data, we still require that the training dataset is linearly separable. This leaves open the question
of whether an overparameterized network will perfectly fit the training data and generalize well for
lower dimensional data, or satisfy a similar margin maximization condition. We also focus mainly on
two-layer networks with fixed outer layer weights trained with the hinge loss. It would be interesting
to investigate whether analogous results hold for deeper architectures or different loss functions and
data models.

Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No.
DMS-1928930 and by the Alfred P. Sloan Foundation under grant G-2021-16778, while the authors
EG and DN were in residence at the Simons Laufer Mathematical Sciences Institute (formerly MSRI)
in Berkeley, California, during the Fall 2023 semester. EG and DN were also partially supported
by NSF DMS 2011140. EG was also supported by a NSF Graduate Research Fellowship under
grant DGE 2034835. GM and KK were partly supported by NSF CAREER DMS 2145630 and
DFG SPP 2298 Theoretical Foundations of Deep Learning grant 464109215. GM was also partly
supported by NSF grant CCF 2212520, ERC Starting Grant 757983 (DLT), and BMBF in DAAD
project 57616814 (SECAI).

9


---Page Break---
References

Ben Adlam and Jeffrey Pennington. The neural tangent kernel in high dimensions: Triple descent and
a multi-scale theory of generalization. In Hal Daumé III and Aarti Singh (eds.), Proceedings of
the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine
Learning Research, pp. 74–84. PMLR, 2020. URL https://proceedings.mlr.press/v119/
adlam20a.html.

Peter L. Bartlett, Philip M. Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear
regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070, 2020. URL
https://www.pnas.org/doi/abs/10.1073/pnas.1907378117.

Mikhail Belkin, Siyuan Ma, and Soumik Mandal. To understand deep learning we need to understand
kernel learning. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International
Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp.
541–549. PMLR, 2018. URL https://proceedings.mlr.press/v80/belkin18a.html.

Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning
practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences,
116(32):15849–15854, 2019. URL https://doi.org/10.1073/pnas.190307011.

Alon Brutzkus, Amir Globerson, Eran Malach, and Shai Shalev-Shwartz.
SGD learns over-
parameterized networks that provably generalize on linearly separable data. In International
Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=
rJ33wwxRb.

Yuan Cao, Quanquan Gu, and Mikhail Belkin. Risk bounds for over-parameterized maximum margin
classification on sub-gaussian mixtures. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang,
and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34,
pp. 8407–8418. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/
paper_files/paper/2021/file/46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.pdf.

Yuan Cao, Zixiang Chen, Misha Belkin, and Quanquan Gu. Benign overfitting in two-layer convolu-
tional neural networks. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh
(eds.), Advances in Neural Information Processing Systems, volume 35, pp. 25237–25250. Cur-
ran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/
2022/file/a12c999be280372b157294e72a4bbc8b-Paper-Conference.pdf.

Niladri S. Chatterji and Philip M. Long. Finite-sample analysis of interpolating linear classifiers in
the overparameterized regime. Journal of Machine Learning Research, 22(129):1–30, 2021. URL
http://jmlr.org/papers/v22/20-974.html.

Niladri S. Chatterji and Philip M. Long.
Foolish crowds support benign overfitting.
Journal
of Machine Learning Research, 23(125):1–12, 2022. URL http://jmlr.org/papers/v23/
21-1199.html.

Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, and Quanquan Gu. Why
does sharpness-aware minimization generalize better than SGD? In Thirty-seventh Conference
on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=
3WAnGWLpSQ.

Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural
network classifiers trained by gradient descent for noisy linear data. In Po-Ling Loh and Maxim
Raginsky (eds.), Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of
Proceedings of Machine Learning Research, pp. 2668–2703. PMLR, 2022. URL https://
proceedings.mlr.press/v178/frei22a.html.

Spencer Frei, Gal Vardi, Peter L. Bartlett, and Nathan Srebro. Benign overfitting in linear classifiers
and leaky relu networks from KKT conditions for margin maximization. In Gergely Neu and
Lorenzo Rosasco (eds.), The Thirty Sixth Annual Conference on Learning Theory, 12-15 July 2023,
Bangalore, India, volume 195 of Proceedings of Machine Learning Research, pp. 3173–3228.
PMLR, 2023. URL https://proceedings.mlr.press/v195/frei23a.html.

10


---Page Break---
Erin George, Michael Murray, William Swartworth, and Deanna Needell.
Training shallow
ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?
In
A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances
in Neural Information Processing Systems, volume 36, pp. 35139–35189. Curran Associates,
Inc., 2023.
URL https://proceedings.neurips.cc/paper_files/paper/2023/file/
6e73c39cc428c7d264d9820319f31e79-Paper-Conference.pdf.

Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J. Tibshirani. Surprises in high-
dimensional ridgeless least squares interpolation. The Annals of Statistics, 50(2):949–986, 2022.
URL https://doi.org/10.1214/21-AOS2133.

Ziwei Ji and Matus Telgarsky.
Directional convergence and alignment in deep learning.
In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in
Neural Information Processing Systems, volume 33, pp. 17176–17186. Curran Associates,
Inc., 2020.
URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
c76e4b2fa54f8506719a5c0dc14c2eb9-Paper.pdf.

Frederic Koehler, Lijia Zhou, Danica J. Sutherland, and Nathan Srebro. Uniform convergence of
interpolators: Gaussian width, norm bounds and benign overfitting. In A. Beygelzimer, Y. Dauphin,
P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems,
2021. URL https://openreview.net/forum?id=FyOhThdDBM.

Guy Kornowski, Gilad Yehudai, and Ohad Shamir. From tempered to benign overfitting in ReLU
neural networks. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
URL https://openreview.net/forum?id=LnZuxp3Tx7.

Yiwen Kou, Zixiang Chen, Yuanzhou Chen, and Quanquan Gu. Benign overfitting in two-layer ReLU
convolutional neural networks. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara
Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International
Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp.
17615–17659. PMLR, 2023. URL https://proceedings.mlr.press/v202/kou23a.html.

Tengyuan Liang and Alexander Rakhlin. Just interpolate: Kernel “Ridgeless” regression can gen-
eralize. The Annals of Statistics, 48(3):1329 – 1347, 2020. URL https://doi.org/10.1214/
19-AOS1849.

Tengyuan Liang, Alexander Rakhlin, and Xiyu Zhai. On the multiple descent of minimum-norm
interpolants and restricted lower isometry of kernels. In Jacob Abernethy and Shivani Agarwal
(eds.), Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of
Machine Learning Research, pp. 2683–2711. PMLR, 2020. URL https://proceedings.mlr.
press/v125/liang20a.html.

Kaifeng Lyu and Jian Li. Gradient descent maximizes the margin of homogeneous neural networks.
In International Conference on Learning Representations, 2020. URL https://openreview.
net/forum?id=SJeLIgBKPS.

Song Mei and Andrea Montanari. The generalization error of random features regression: Precise
asymptotics and the double descent curve. Communications on Pure and Applied Mathematics,
75(4):667–766, 2022. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.
22008.

Andrea Montanari, Feng Ruan, Basil Saeed, and Youngtak Sohn. Universality of max-margin
classifiers. arXiv preprint arXiv:2310.00176, 2023a.

Andrea Montanari, Feng Ruan, Youngtak Sohn, and Jun Yan. The generalization error of max-margin
linear classifiers: Benign overfitting and high-dimensional asymptotics in the overparametrized
regime. arXiv preprint arXiv:1911.01544, 2023b.

Vidya Muthukumar, Kailas Vodrahalli, Vignesh Subramanian, and Anant Sahai. Harmless interpo-
lation of noisy data in regression. IEEE Journal on Selected Areas in Information Theory, 1(1):
67–83, 2020. URL https://doi.org/10.1109/JSAIT.2020.2984716.

11


---Page Break---
Vidya Muthukumar, Adhyyan Narang, Vignesh Subramanian, Mikhail Belkin, Daniel Hsu, and Anant
Sahai. Classification vs regression in overparameterized regimes: Does the loss function matter? J.
Mach. Learn. Res., 22(1), 2021. URL http://jmlr.org/papers/v22/20-603.html.

George Pólya. Remarks on computing the probability integral in one and two dimensions. In
Proceedings of the Berkeley Symposium on Mathematical Statistics and Probability, number 1, pp.
63. University of California Press Berkeley, 1949.

Mark Rudelson and Roman Vershynin. Smallest singular value of a random rectangular matrix.
Communications on Pure and Applied Mathematics, 62(12):1707–1739, 2009. URL https:
//doi.org/10.1002/cpa.20294.

Ohad Shamir. The implicit bias of benign overfitting. In Po-Ling Loh and Maxim Raginsky
(eds.), Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings
of Machine Learning Research, pp. 448–478. PMLR, 2022. URL https://proceedings.mlr.
press/v178/shamir22a.html.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science.
Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.
URL https://doi.org/10.1017/9781108231596.

Guillaume Wang, Konstantin Donhauser, and Fanny Yang. Tight bounds for minimum ℓ1-norm
interpolation of noisy data. In International Conference on Artificial Intelligence and Statistics,
2021a. URL https://proceedings.mlr.press/v151/wang22k.html.

Ke Wang, Vidya Muthukumar, and Christos Thrampoulidis. Benign overfitting in multiclass classifi-
cation: All roads lead to interpolation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang,
and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34,
pp. 24164–24179. Curran Associates, Inc., 2021b. URL https://proceedings.neurips.cc/
paper_files/paper/2021/file/caaa29eab72b231b0af62fbdff89bfce-Paper.pdf.

Denny Wu and Ji Xu. On the optimal weighted ℓ2 regularization in overparameterized linear re-
gression. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances
in Neural Information Processing Systems, volume 33, pp. 10112–10123. Curran Associates,
Inc., 2020.
URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
72e6d3238361fe70f22fb0ac624a7072-Paper.pdf.

Xingyu Xu and Yuantao Gu. Benign overfitting of non-smooth neural networks beyond lazy training.
In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent (eds.), Proceedings of The 26th
International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of
Machine Learning Research, pp. 11094–11117. PMLR, 2023. URL https://proceedings.
mlr.press/v206/xu23k.html.

Zhiwei Xu, Yutong Wang, Spencer Frei, Gal Vardi, and Wei Hu. Benign overfitting and grokking
in ReLU networks for XOR cluster data. In The Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.net/forum?id=BxHgpC6FNv.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understand-
ing deep learning requires rethinking generalization. In International Conference on Learning
Representations, 2017. URL https://openreview.net/forum?id=Sy8gdB9xx.

Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, and Sham Kakade. Benign overfitting
of constant-stepsize SGD for linear regression. In Mikhail Belkin and Samory Kpotufe (eds.),
Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of
Machine Learning Research, pp. 4633–4635. PMLR, 2021. URL https://proceedings.mlr.
press/v134/zou21a.html.

12


---Page Break---
Appendix A
Preliminaries on random vectors

Recall that the sub-exponential norm of a random variable X is defined as
∥X∥ψ1 := inf{t > 0: E[exp(|X|/t)] ≤2}
(see Vershynin, 2018, Definition 2.7.5) and that the sub-Gaussian norm is defined as
∥X∥ψ2 := inf{t > 0: E[exp(X2/t2)] ≤2}.

A random variable X is sub-Gaussian if and only if X2 is sub-exponential. Furthermore, ∥X2∥ψ1 =
∥X∥2
ψ2.

Lemma A.1. Let n ∼N(0d, d−1(Id −vvT )) and suppose that Z ∈Rm×d. Then with probability
at least 1 −ϵ,

∥Zn∥≤C∥Z∥F

r

1
d log 1

ϵ .

Proof. Let P = Id −vvT be the orthogonal projection onto span({v})⊥, so that Zn is identi-
cally distributed to d−1/2ZP n′, where n′ has distribution N(0d, Id). Following Vershynin (2018,
Theorem 6.3.2),∥Zn∥−∥d−1/2ZP ∥F

ψ2 =
∥d−1/2ZP n′∥−∥d−1/2ZP ∥F

ψ2
≤C∥d−1/2ZP ∥

≤Cd−1/2∥Z∥∥P ∥

= Cd−1/2∥Z∥

≤Cd−1/2∥Z∥F ,
where we used that P is an orthogonal projection in the fourth line and that the operator norm is by
bounded above by the Frobenius norm in the fifth line. As a result the sub-Gaussian norm of ∥Zn∥
is bounded as
∥Zn∥

ψ2 ≤
∥Zn∥−∥d−1/2ZP ∥F

ψ2 +
∥d−1/2ZP ∥F

ψ2
≤Cd−1/2(∥Z∥F + ∥ZP ∥F )

≤Cd−1/2∥Z∥F ,
where the last line follows from the calculation
∥ZP ∥F = ∥P T ZT ∥F
= ∥P ZT ∥F
≤∥P ∥∥ZT ∥F
= ∥ZT ∥F
= ∥Z∥F .
This implies a tail bound (see Vershynin, 2018, Proposition 2.5.2)

P(∥Zn∥≥t) ≤2 exp

−
dt2

C∥Z∥2
F


,
for all t ≥0.

Setting t = C∥Z∥F
q

1
d log 2

ϵ , the result follows.

Lemma A.2. Let n ∼N(0d, d−1(Id −vvT )) and suppose z ∈span({v})⊥. There exists a C > 0
such that with probability at least 1 −δ

|⟨n, z⟩| ≤C∥z∥

r

1
d log 1

δ .

Furthermore, there exists a c > 0 such that with probability at least 1

2

|⟨n, z⟩| ≥c∥z∥
√

d
.

13


---Page Break---
Proof. Let X = ⟨n, z⟩. Then X is Gaussian with variance

E[X2] = E[zT nnT z]

= zT E[nnT ]z

= d−1zT (Id −vvT )z

= d−1zT z

= d−1∥z∥2.

Note the third line above follows from the fact that z ∈span({v})⊥. Therefore by Hoeffding’s
inequality, for all t ≥0

P(|⟨n, z⟩| ≥t) ≤2 exp

−
dt2

C∥z∥2


.

Setting t = C∥z∥2q

1
d log 2

δ , we obtain

P(|⟨n, z⟩| ≥t) ≤δ,

which establishes the first part of the result.

Since d1/2∥z∥−1X is a standard Gaussian, there exists a constant c such that

P(|d1/2∥z∥−1X| ≥c) ≤1

2.

Rearranging, we obtain

P(|⟨n, z⟩| ≥cd−1/2∥z∥) ≤1

2.

This establishes the second part of the result.

Appendix B
Upper bounding the norm of the max-margin classifier of the
data

Here we establish key properties concerning the data model given in Definition 2.1, our main goal
being to establish bounds on the norm of the max-margin classifier. To this end we first identify
certain useful facts about rectangular Gaussian matrices. In what follows we index the singular values
of any given matrix A in decreasing order as σ1(A) ≥σ2(A) ≥· · · . Furthermore, we denote the
i-th-row of a matrix A as ai.

Lemma B.1. Let G ∈Rn×d be a Gaussian matrix whose entries are mutually i.i.d. with distribution
N(0, 1). If d = Ω
 
n + log 1

δ

, then with probability at least 1 −δ the following inequalities are
simultaneously true.

1. σ1(G) ≤C(
√

d + √n),

2. σn(G) ≥c(
√

d −√n).

Proof. We proceed by upper bounding the probability that each individual inequality does not hold.

1. To derive an upper bound on σ1(G) we use the following fact (see Vershynin, 2018, Theorem
4.4.5). For any ϵ > 0,

P(σ1(G) ≥C1(√n +
√

d + ϵ)) ≤2 exp(−ϵ2).

With ϵ = √n +
√

d and d ≥log 4

δ then

P(σ1(G) ≥2C1(√n +
√

d)) ≤2 exp(−d)

≤δ

2.

14


---Page Break---
2. To derive a lower bound on σn(G) we use the following fact (see Rudelson & Vershynin, 2009,
Theorem 1.1). There exist constants C1, C2 > 0 such that, for any ϵ > 0,

P(σn(G) ≤ϵ(
√

d −
√

n −1)) ≤(C1ϵ)d−n+1 + e−C2d.

Let ϵ =
1
C1e and let d ≥2n +

2 +
1
C2


log 4

δ . Then

P(σn(A) ≤ϵ(
√

d −√n)) ≤exp(−d/2) + exp(−C2d)

≤δ

4 + δ

4

= δ

2.

Hence both bounds hold simultaneously with probability at least 1 −δ.

The next lemma formulates lower and upper bounds on the smallest and largest singular values of a
noise matrix under our data model.

Lemma 4.1. Let N ∈Rn×d denote a random matrix whose rows are drawn mutually i.i.d. from
N(0d, 1

d(Id −vvT )). If d = Ω
 
n + log 1

δ

, then there exists constants C1 and C2 such that, with
probability at least 1 −δ,
C1 ≤σn(N) ≤σ1(N) ≤C2.

Proof. Let H = span({v})⊥∼= Rd−1. Let N ′ : H →Rn be a random matrix whose rows are
drawn mutually i.i.d. from N(0d, IH). Since d = Ω
 
n + log 1

δ

, with probability at least 1 −δ, we
have
c(
√

d −√n) ≤σn(N ′) ≤σ1(N ′) ≤C(
√

d + √n).

We denote the above event by ω. Let J : H →Rd be the inclusion map and let P = Id −vvT . For
any random vector n with distribution N(0d, IH), Jn is a Gaussian random vector with covariance
matrix JJT = P . Therefore, d−1/2N ′JT is a random matrix whose rows are drawn mutually i.i.d.
from N(0d, d−1P ). That is, d−1/2N ′JT is identically distributed to N. For the lower bound on
the n-th largest singular value, if d ≥4n

c2 , then conditional on ω we have

σn(d−1/2N ′JT ) = d−1/2σmin(JN ′T )

≥d−1/2σmin(J)σmin(N ′T )

= d−1/2σmin(N ′T )

= d−1/2σn(N ′)

≥c −
rn

d

≥c

2.

Note here we define σmin to be the smallest singular value of a matrix. In the first line we used JN ′T
is a linear map Rn →Rd and d ≥n, and in the third line we used the fact that J is an inclusion map.
For the upper bound on the largest singular value, if d ≥
n
C2 , then conditional on ω we have

σ1(d−1/2N ′JT ) = d−1/2σ1(JN ′T )

≤d−1/2σ1(N ′T )

= d−1/2σ1(N ′)

≤C +
rn

d
≤2C.

15


---Page Break---
Note here that again we used the fact that J is an inclusion map in the first line. Therefore, if
d = Ω
 
n + log 1

δ


P
 c

2 ≤σn(N) ≤σ1(N) ≤2C

≥P(ω)

≥1 −δ.

The following lemma is useful for constructing vectors in the noise subspace with properties suitable
for bounding the norm of the max-margin solution. We remark that the same approach could be
used in the setting where the noise and signal are not orthogonal by considering the pseudo-inverse
([N, v]T )† instead of N †.

Lemma B.2. Let I ⊆[n] be an arbitrary subset such that |I| = ℓ. In the context of the data
model given in Definition 2.1, assume d = Ω
 
n + log 1

δ

. Then there exists z ∈Rd such that with
probability at least 1 −δ the following hold simultaneously.

1. ˆyi⟨ni, z⟩= 1 for all i ∈I,

2. ˆyi⟨ni, z⟩= 0 for all i /∈I,

3. z ⊥v,

4. C1 ≤∥z∥≤C2.

Proof. Recall that N ∈Rn×d is a random matrix whose rows are selected i.i.d. from N(0d, d−1(Id−
vvT )). By Lemma 4.1, with probability at least 1 −δ we have

c ≤σn(N) ≤σ1(N) ≤C.

Conditioning on this event, we will construct a vector z which satisfies the desired properties. Let
w ∈Rn satisfy wi = ˆyi if i ∈I and wi = 0 otherwise. Let z = N †w, where N † = N T (NN T )−1
is the right pseudo-inverse of N. Then Nz = w. In particular, for i ∈I, ˆyi⟨ni, z⟩= ˆyiwi = 1,
and for i /∈I, ˆyi⟨ni, z⟩= ˆyiwi = 0. This establishes properties 1 and 2. Since N †w is in the span
of the set {ni}i∈[n], it is orthogonal to v. This establishes property 3. Finally, we can bound

∥z∥= ∥N †w∥

≤∥N †∥∥w∥

=
∥w∥
σn(N)

≤∥w∥

c

=

√

ℓ
c
and

∥z∥= ∥N †w∥

≥σn(N †)∥w∥

=
∥w∥
σ1(N)

≥∥w∥

C

=

√

ℓ
C
which establishes property 4.

16


---Page Break---
With Lemma B.2 in place we are now able to appropriately bound the max-margin norm.
Lemma B.3. In the context of the data model given in Definition 2.1, let w∗denote the max-margin
classifier of the training data (X, ˆy), which exists almost surely. If d = Ω
 
n + log 1

δ

then with
probability at least 1 −δ

∥w∗∥≤C

s

1
γ +
k
1 −γ ,

where C > 0 is a constant.

Proof. Under the assumptions stated, the conditions of Lemma B.2 hold with probability at least
1 −δ. Conditioning on this let

w =
1
√γ v +
2
√1 −γ z

where z is the vector constructed in Lemma B.2 with I = B. For any i ∈[n] we therefore have

ˆyi⟨xi, w⟩= βi + 2 ˆyi⟨ni, z⟩.

As a result, for i ∈G

ˆyi⟨xi, w⟩= 1 + 2 ˆyi⟨ni, z⟩= 1,

while for l ∈B

ˆyi⟨xi, w⟩= −1 + 2ˆyi⟨ni, z⟩= 1.

As a result ˆyi⟨xi, w⟩= 1 for all i ∈[n]. Furthermore, observe

∥w∥2 = 1

γ +
4
1 −γ ∥z∥2

≤C
 1

γ +
k
1 −γ



for a universal constant C. To conclude observe ∥w∗∥≤∥w∥by definition of being max-margin.

Lemma B.3 constructs a classifier with margin one using an appropriate linear combination of the
signal vector v and a vector in the noise subspace which classifies all noise components belonging to
bad points correctly. This bound is useful for the benign overfitting setting in which γ is not too small.
However, for small γ, as is the case in the non-benign overfitting setting, this bound behaves poorly
as the only way the construction can fit all the data points is by making the coefficient in front of the
v component large. For the non-benign overfitting setting we therefore require a different approach
and instead fit all data points based on their noise components alone. In particular, the following
bound behaves better than that given in Lemma B.3 when γ approaches 0.
Lemma B.4. In the context of the data model given in Definition 2.1, let w∗denote the max-margin
classifier of the training data (X, ˆy), which exists almost surely. If d = Ω
 
n + log 1

δ

then with
probability at least 1 −δ

∥w∗∥≤C
r
n
1 −γ .

Proof. Applying Lemma B.2 with I = [n] then with probability 1 −δ there exists z ∈Rd such that
∥z∥= Θ(√n), ˆyi⟨ni, z⟩= 1 for all i ∈[n] and z ⊥v. Conditioning on this event, let

w =
1
√1 −γ z.

Then for all i ∈[n],
ˆyi⟨xi, w⟩= ˆyi⟨ni, z⟩= 1.
Furthermore, there exists a constant C > 0 such that

∥w∥≤C
r
n
1 −γ .

To conclude observe ∥w∗∥≤∥w∥by definition of being max-margin.

17


---Page Break---
Appendix C
Linear models

C.1
Sufficient conditions for benign and harmful overfitting

We start by providing a lemma which characterizes the generalization properties of linear classifiers.
Lemma C.1. In the context of the data model given in Definition 2.1, consider the linear classifier
w = avv + z, where av ∈R and ⟨z, v⟩= 0. If av ≥0, then the generalization error can be
bounded as follows:

P(y⟨w, x⟩≤0) ≤exp

−d

2
γ
1 −γ
a2
v
∥z∥2


.

and

P(y⟨w, x⟩≤0) ≥max

(
1
2 −

s

dγ
2π(1 −γ)
av
∥z∥, 1

4 exp

−6d

π
γ
1 −γ
a2
v
∥z∥2

)

.

Proof. Recall from Definition 2.1 that a test pair (x, y) satisfies

x = y(√γv +
p

1 −γn),

where n ∼N(0d, d−1(Id −vvT )) is a random vector. Let X = y⟨w, x⟩, so

X = ⟨avv + z, √γv +
p

1 −γn⟩

= √γav +
p

1 −γ⟨n, z⟩.

Then X is a Gaussian random variable with expectation

E[X] = √γav +
p

1 −γE[⟨n, z⟩]
= √γav
and variance

Var(X) = (1 −γ)Var(⟨n, z⟩)

= (1 −γ)E[zT nnT z]

= 1 −γ

d
zT (Id −vvT )z

= 1 −γ

d
zT z

= (1 −γ)∥z∥2

d
.

By Hoeffding’s inequality, for all t ≥0,

P(X ≤√γav −t) ≤exp

−
t2d
2(1 −γ)∥z∥2


.

Setting t = √γav, we obtain

P(X ≤0) ≤exp

−
γda2
v
2(1 −γ)∥z∥2


,

which establishes the upper bound on the generalization error.

To prove the lower bound, we integrate a standard Gaussian pdf:

P(X ≤0) = P

 
X −√γav
√1 −γ∥z∥/
√

d
≤−

s

γd
1 −γ
av
∥z∥

!

= 1

2 −
1
√

2π

Z 0

−
q

γd
1−γ
av
∥z∥
e−t2/2dt

≥1

2 −
1
√

2π

 s

γd
1 −γ
av
∥z∥

!

.

18


---Page Break---
Another bound can be obtained using the following inequality (Pólya, 1949, (1.5)):

1
√

2π

Z x

0
e−t2/2dt ≤
p

1 −e−2x2/π.

We proceed

P(X ≤0) = 1

2 −
1
√

2π

Z 0

−
q

γd
1−γ
av
∥z∥
e−t2/2dt

= 1

2 −
1
√

2π

Z q

γd
1−γ
av
∥z∥

0
e−t2/2dt

≥1

2 −1

2

s

1 −exp

−
2γda2v
π(1 −γ)∥z∥2



≥1

2 −1

2


1 −1

2 exp

−
2γda2
v
π(1 −γ)∥z∥2



≥1

4 exp

−
2γda2
v
π(1 −γ)∥z∥2


.

The following result establishes benign and non-benign overfitting for linear models and data as
per Definition 2.1, with a phase transition between these outcomes depending on the signal to noise
parameter γ.

Theorem C.2. In the context of the data model described in Section 2, let w∗be a max-margin
linear classifier of the training data. Let A : Rn×d × {±1}n →Rd be a learning algorithm which is
approximately margin-maximizing, Definition 2.3. For δ ∈(0, 1], let d = Ω
 
n + log 1

δ

. Then with
probability at least 1 −δ over the randomness of the training data (X, ˆy), the following hold with ϵ
denoting the generalization error of A(X, ˆy).

(A) If γ
=
Ω(|A|2n−1) and k
=
O(|A|−2n), then exp
 
−C1dk−1
≤
ϵ
≤
exp
 
−C2dk−1|A|−2
for fixed positive constants C1 and C2.

(B) If γ = O(|A|−2d−1), then ϵ ≥1

2 −
p

Cdγ|A|2 for a fixed positive constant C.

Proof. For training data (X, ˆy) let w = A(X, ˆy) be the learned linear classifier. First, recall
xi = √γyiv + √1 −γni for all i ∈[n], ∥v∥= 1, and observe that we can decompose the vector w
as w = avv + z, where av ∈R, and z ⊥v. As a result, for each i ∈[n],

ˆyi⟨xi, w⟩= √γavβi +
p

1 −γˆyi⟨ni, z⟩.
(8)

First we establish (A). As d = Ω
 
n + log 1

δ

, Lemmas 4.1 and B.3 show that with probability at

least 1 −δ

2, ∥N∥2, ∥NG∥2, ∥NB∥≤C and ∥w∗∥≤C
q

1
γ +
k
1−γ . Here NG and NB denote the
matrices formed by taking only the rows of N which satisfy β = 1 and β = −1, respectively. We
denote this event ω and condition on it in all that follows for the proof of (A). As A is approximately
max margin then given equation 8 we have for all i ∈G

1 ≤√γav +
p

1 −γˆyi⟨ni, z⟩.

Suppose that √γav < 1

2. Then the above inequality implies √1 −γˆyi⟨ni, z⟩≥1

2 for all i ∈G.
Squaring and then summing this expression over all i ∈G it follows that

n −k

4
≤(1 −γ)
X

i∈G
|⟨ni, z⟩|2

≤(1 −γ)∥NGz∥2

≤(1 −γ)∥NG∥2∥z∥2.

19


---Page Break---
Since A is approximately margin-maximizing,

n −k

4
≤(1 −γ)C∥z∥2

≤(1 −γ)C∥w∥2

≤C|A|2(1 −γ)∥w∗∥2

≤C|A|2(1 −γ)
 1

γ +
k
1 −γ



≤C|A|2

γ
+ C|A|2k,

which further implies n ≤C|A|2

γ
+ C|A|2k for some other constant C. For this inequality to hold,

either C|A|2

γ
≥n

2 or C|A|2k ≥n

2 . With k ≤
n
4C|A|2 and γ ≥1

k, neither of these can be true and
therefore we conclude that √γav ≥1

2. Then

a2
v
∥z∥2 ≥
1
4γ∥z∥2

≥
1
4γ∥w∥2

≥
1
4γ|A|2∥w∗∥2

≥
C
|A|2γ
1

1
γ +
k
1−γ

≥
C
2|A|2γ
1 −γ

k

for a positive constant C. Letting (x, y) denote a test point pair, then by Lemma C.1 it follows that
for a different constant C

P(y⟨w, x⟩≤0 | ω) ≤exp

−d

2
γ
1 −γ
a2
v
∥z∥2



≤exp

−Cd

|A|2k


.

Hence the generalization error is at most ϵ when ω occurs, which happens with probability at least
1 −δ. This establishes the upper bound of (A).

For the lower bound of (A), since w is a linear classifier,

⟨ni, z⟩≥
1
√1 −γ (1 + av
√γ) ≥av

r
γ
1 −γ

for all i ∈B, from which we conclude |⟨ni, z⟩| ≥av√γ. This implies

∥NBz∥≥aV

s

kγ
1 −γ

Along with ∥NBz∥≤∥NB∥∥z∥≤C∥z∥we conclude

∥z∥≥av

C

s

kγ
1 −γ .

With this bound we then bound

av
∥z∥≤C ·
r1 −γ

kγ .

20


---Page Break---
By Lemma C.1 we then can bound

P(y⟨w, x⟩≤0 | ω) ≥1

4 exp

−6d

π
γ
1 −γ
a2
v
∥z∥2



≥exp

−Cd

k



for a new constant C, provided av is positive. In the last line we can bound d

k below as d ≥n and
k = O(n). If av is negative then the generalization error is at least 1

2, which is still bounded below
by exp(−Cd/k).

We now turn our attention to (B). As d = Ω(n + log 1

δ ), from Lemmas 4.1 and B.4 with probability
at least 1 −δ it holds that ∥NG∥2 ≤C and ∥w∗∥≤
C√n
√1−γ . We denote this event ω′ and condition
on it in all that follows for the proof of (B). In particular,
a2
v + ∥z∥2 = ∥w∥2

≤|A|2∥w∗∥2

≤C|A|2n

1 −γ .
(9)

For all i ∈[n],

1 ≤√γβiav +
p

1 −γ ˆyi⟨ni, z⟩.
For this inequality to hold, either |√γav| ≥1/2 or √1 −γ ˆyi⟨ni, z⟩≥1/2 for all i ∈[n]. If
|√γav| ≥1/2, then with γ ≤
1
4(C+1)|A|2d we have

a2
v ≥2C|A|2d ≥2C|A|2n.
However, from equation 9 we have

2C|A|2n > C|A|2n

1 −γ
≥a2
v

which is a contradiction. Therefore, under the regime specified there exists a C > 0 such that with
γ ≤
1
C|A|2d, we have √1 −γ ˆyi⟨ni, z⟩≥1/2 for all i ∈[n]. Rearranging, squaring and summing
over all i ∈[n] yields

n
4(1 −γ) ≤

n
X

i=1
|⟨ni, z⟩|2 ≤∥Nz∥2 ≤C∥z∥2,

where the final inequality follows from conditioning on ω. Then
a2
v
∥z∥2 ≤∥w∥2

∥z∥2

≤|A|2∥w∗∥2

∥z∥2

≤
C|A|2
n
1−γ
n
1−γ
≤C|A|2,
where the constant C > 0 may vary between inequalities. Letting (x, y) denote a test point pair, by
Lemma C.1 it follows that

P(y⟨w, x⟩≤0 | ω) ≥1

2 −

s

dγ
2π(1 −γ)
av
∥z∥

≥1

2 −

s

Cdγ|A|2

1 −γ

≥1

2 −
p

Cdγ|A|2.

Hence the generalization error is at least 1

2 −
p

Cdγ|A|2 when ω′ occurs, which happens with
probability at least 1 −δ. This establishes (B).

21


---Page Break---
Appendix D
Leaky ReLU Networks

In this section we consider a leaky ReLU network f : R2m × Rd →R with forward pass given by

f(W , x) =

2m
X

j=1
(−1)jσ(⟨wj, xi⟩),

where σ(z) = max(αz, z) for some α ∈(0, 1). For any such network, we may decompose the
neuron weights wj into a signal component and a noise component,

wj = ajv + zj,

where aj ∈R and zj ∈Rd satisfies zj ⊥v. The ratio aj/∥zj∥therefore grows with the alignment
of wj with the signal and shrinks if wj instead aligns more with the noise. Collecting the noise
components of the weight vectors, let Z ∈R(2m)×d be the matrix whose j-th row is zj. In order to
track the alignment of the network as a whole with the signal versus noise subspaces we introduce
the following quantities. Let

A1 = f(W , v) =

2m
X

j=1
(−1)jσ(aj),

A−1 = f(W , −v) =

2m
X

j=1
(−1)j+1σ(−aj)

be referred to as the positive and negative signal activation of the network respectively. Moreover,
define

Amin = min(A1, A−1)

as the worst-case signal activation of the network, and

Alin =

2m
X

j=1
(−1)jaj

as the linearized network activation. To measure the amount of noise the network learns we define

zlin =

2m
X

j=1
(−1)jzj.

D.1
Training dynamics

Theorem 3.1. Let f : Rp × Rn →R be a leaky ReLU network with forward pass as defined by
equation 1. Suppose the step size η and initialization condition λ satisfy Assumption 1. Then for any
linearly separable data set (X, ˆy) AGD(X, ˆy, η, λ) converges after T iterations, where

T ≤C∥w∗∥2

ηα2m .

Furthermore AGD is approximately margin maximizing on f (Definition 2.3) with

|AGD| ≤
C
α√m.

Proof. Our approach is to adapt a classical technique used for the proof of convergence of the
Perceptron algorithm for linearly separable data. This is also the approach adopted by Brutzkus
et al. (2018). The key idea of the proof is to bound in terms of the number of updates both the norm
of the learned vector w as well as its alignment with any linear separator of the data. From the
Cauchy-Schwarz inequality these bounds cannot cross, and this in turn bounds the number of updates
that can occur. Analogously, we track the alignment of W (t) with the max-margin classifier along
with the Frobenius norm of the W (t). To this end denote

G(t) = ∥W (t)
j
∥2
F

22


---Page Break---
and

F(t) =

2m
X

j=1
(−1)j⟨w(t)
j , w∗⟩,

where w∗is a max-margin linear classifier of the dataset.
Recall that F(t) = {i ∈[n] :
ˆyif(W (t), xi) < 1} denotes the number of active data points at training step t. We also define
U(t) = Pt−1
s=0 |F(s)| to be the number of data point updates between iterations 0 and t. First, by
Cauchy-Schwarz

1 ≤⟨w∗, xi⟩≤∥w∗∥· ∥xi∥

for all i ∈[n]. Therefore,

∥w∗∥≥
1
mini∈[n] ∥xi∥.

By Assumption 1, for all j ∈[2m],

∥w(0)
j ∥≤
√α
m mini∈[n] ∥xi∥≤∥w∗∥

αm .
(10)

For all t ≥0, the update rule of GD implies

G(t + 1) =

2m
X

j=1
∥w(t+1)
j
∥2

=

2m
X

j=1


w(t)
j
+ η(−1)j X

i∈F(t)
˙σ(⟨w(t)
j , xi⟩)ˆyixi



2

=

2m
X

j=1
∥w(t)
j ∥2 + 2η

2m
X

j=1

X

i∈F(t)
(−1)j ˙σ(⟨w(t)
j , xi⟩)ˆyi⟨w(t)
j , xi⟩+ η2
2m
X

j=1

X

i,l∈F(t)
˙σ(⟨w(t)
j , xi⟩)⟨ˆyixi, ˆyixℓ⟩

≤

2m
X

j=1
∥w(t)
j ∥2 + 2η

2m
X

j=1

X

i∈F(t)
(−1)j ˙σ(⟨w(t)
j , xi⟩)ˆyi⟨w(t)
j , xi⟩+ 2mη2|F(t)|2 max
i∈[n] ∥xi∥2.

Observe that for all z ∈R, σ(s) = ˙σ(z)z, so can rewrite the second term of the above expression as

2η

2m
X

j=1

X

i∈F(t)
(−1)j ˙σ(⟨w(t)
j , xi⟩)ˆyi⟨w(t)
j , xi⟩= 2η

2m
X

j=1

X

i∈F(t)
(−1)jσ(⟨w(t)
j , xi⟩)ˆyi

= 2η
X

i∈F(t)
ˆyif(W (t), xi)

< 2η
X

i∈F(t)
1

= 2η|F(t)|

where the inequality in the second-to-last line follows as we are summing over F(t), which by
definition consists of the i ∈[n] such that ˆyif(W (t), xi) < 1. As a result we obtain

G(t + 1) ≤

2m
X

j=1
∥w(t)
j ∥2 + 2η|F(t)| + 2mη2|F(t)|2 max
i∈[n] ∥xi∥2

= G(t) + 2η|F(t)| + 2mη2|F(t)|2 max
i∈[n] ∥xi∥2

≤G(t) + 4η|F(t)|,

where the last line follows since

η ≤
1
mn maxi∈[n] ∥xi∥2 ≤
1
|F(t)|m maxi∈[n] ∥xi∥2 .

23


---Page Break---
By equation 10, the initialization satisfies

G(0) =

2m
X

j=1
∥w(0)
j ∥2

≤

2m
X

j=1

∥w∗∥2

α2m2

= 2∥w∗∥2

α2m
So by induction, for all t ≥0

G(t) ≤2∥w∗∥2

α2m
+ 3η

t−1
X

s=0
|F(s)| = 2∥w∗∥2

α2m
+ 3ηU(t).
(11)

Next we find a bound for F(t). For all t ≥0 then by definition of the GD update

F(t + 1) =

2m
X

j=1
(−1)j⟨w(t+1)
j
, w∗⟩

=

2m
X

j=1
(−1)j⟨w(t)
j , w∗⟩+ η

2m
X

j=1

X

i∈F(t)
˙σ(⟨w(t)
j , xi⟩)ˆyi⟨w∗, xi⟩.

Since ˆyi⟨w∗, xi⟩≥1 for all i ∈[n], the above expression is bounded below by

2m
X

j=1
(−1)j⟨w(t)
j , w∗⟩+ η

2m
X

j=1

X

i∈F(t)
˙σ(⟨w(t)
j , xi⟩)ˆyi = F(t) + η

2m
X

j=1

X

i∈F(t)
˙σ(⟨w(t)
j , xi⟩)

≥F(t) + η

2m
X

j=1

X

i∈F(t)
α

≥F(t) + 2ηmα|F(t)|.

Hence unrolling the update for GD for all t ≥0 it follows that

F(t + 1) ≥F(0) + 2ηmα

t−1
X

s=0
|F(s)|.

At initialization, by equation 10 then

F(0) =

2m
X

j=1
(−1)j⟨w(0)
j , w∗⟩

≥−

2m
X

j=1
∥w(0)
j ∥· ∥w∗∥

≥−

2m
X

j=1

∥w∗∥2

αm

= −2∥w∗∥2

α
.

Therefore by induction, for all t ≥0 we have

F(t) ≥−2∥w∗∥2

α
+ 2ηmα

t−1
X

s=0
|F(s)|

= −2∥w∗∥2

α
+ 2ηmαU(t).

24


---Page Break---
Combining our bounds for F(t) and G(t), we obtain

−2∥w∗∥2

α
+ 2ηmαU(t) ≤F(t)

=

2m
X

j=1
(−1)j⟨w(t)
j , w∗⟩

≤∥w∗∥

2m
X

j=1
∥w(t)
j ∥

≤∥w∗∥



2m

2m
X

j=1
∥w(t)
j ∥2





1/2

= ∥w∗∥(2mG(t))1/2

≤∥w∗∥
4∥w∗∥2

α2
+ 6mηU(t)
1/2
.

This implies that either

−2∥w∗∥2

α
+ 2ηmαU(t) ≤0
(12)

or

−2∥w∗∥2

α
+ 2ηmαU(t)
2
≤∥w∗∥2
4∥w∗∥2

α2
+ 6mηU(t)

.
(13)

If (12) holds, then

U(t) ≤∥w∗∥2

ηα2m .

If (13) holds, then rearranging yields

4η2m2α2U(t)2 ≤14∥w∗∥2ηmU(t)

U(t) ≤7∥w∗∥2

2ηα2m .

Therefore, in both cases there exists a constant C such that

U(t) ≤C∥w∗∥2

ηα2m .
(14)

This holds for all t ∈N and therefore
∞
X

t=0
|F(t)| ≤C∥w∗∥2

ηα2m
< ∞.

This implies that there exists s ∈N such that |F(s)| = 0. Let T ∈N be the minimal iteration such
that |F(T )| = 0. Then for all i ∈[n] ˆyif(W (T ), xi) ≥1. So the network achieves zero loss and also
has zero gradient at iteration T. In particular,

T =

T −1
X

t=0
1 ≤

T −1
X

t=0
|F(t)| ≤C∥w∗∥2

ηα2m .

To bound |AGD| we combine equations (14) and (11) to obtain

G(T) ≤2∥w∗∥2

α2m
+ 3ηU(t)

≤2∥w∗∥2

α2m
+ C∥w∗∥2

ηα2m

≤C∥w∗∥2

α2m
.

25


---Page Break---
As a result for all linearly separable datasets (X, ˆy)

∥W ∥F

∥w∗∥=
C
α√m

and therefore
|AGD| ≤
C
α√m
as claimed.

The training dynamics of gradient descent also give us the following result relating the linearization
of the noise component of the network to the noise component of the network itself.
Lemma D.1. Let λ, δ > 0. Suppose that d ≥Ω
 
n + log 1

δ

. In the context of training data (X, ˆy)
sampled under the data model given in Definition 2.1, let W = AGD(X, ˆy, η, λ). Then with
probability at least 1 −δ over the randomness of (X, ˆy)

∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2 ≤C

αm (∥zlin∥+ 2mλ)2 .

Proof. At each iteration of gradient descent,

w(t+1)
j
= w(t)
j
+ η(−1)j
n
X

i=1
b(t)
ij ˆyixi,

where

b(t)
ij =








0
if ˆyif(W (t), xi) ≥1
1
if ˆyif(W (t), xi) < 1 and ⟨w(t)
j , xi⟩≥0
α
otherwise.
.

Let T be the iteration at which gradient descent terminates. Then for each j ∈[2m],

wj = w(T )
j
= w(0)
j
+ η(−1)j
T −1
X

t=0

n
X

i=1
b(t)
ij ˆyixi.

Then the noise component of wj is given by

zj = wj −⟨wj, v⟩v

= w(0)
j
−⟨w(0)
j , v⟩v + η(−1)j
T −1
X

t=0

n
X

i=1
b(t)
ij ˆyi(xi −⟨xi, v⟩v)

= w(0)
j
−⟨w(0)
j , v⟩v + η(−1)j
T −1
X

t=0

n
X

i=1
b(t)
ij ˆyini.

Define
ˆzj = zj −w(0)
j
+ ⟨w(0)
j , v⟩v
and let

ˆzlin =

2m
X

j=1
(−1)j ˆzj,

Then for all j ∈[2n],

(∥ˆzj∥−∥zj∥)2 ≤∥ˆzj −zj∥2

= ∥w(0)
j
−⟨w(0)
j , v⟩v∥2

≤∥w(0)
j ∥2

≤λ2.

26


---Page Break---
Furthermore, if ∥zj∥≤∥ˆzj∥then the above implies ∥ˆzj∥≤∥zj∥+ λ while if ∥zj∥≥∥ˆzj∥then
this inequality holds trivially. As a result,
∥ˆzj∥2 −∥zj∥2 = |(∥ˆzj∥+ ∥zj∥) · (∥ˆzj∥−∥zj∥)|

≤|∥ˆzj∥+ ∥zj∥| · |∥ˆzj∥−∥zj∥)|
≤(2∥zj∥+ λ)(λ).

If ∥zj∥≥∥ˆzj∥then the above implies ∥ˆzj∥2 ≥∥zj∥2 −λ(2∥zj∥+ λ), if ∥zj∥≤∥ˆzj∥this
inequality is trivially true. As a result,

2m
X

j=1
∥ˆzj∥2 ≥

2m
X

j=1
(∥zj∥2 −λ(2∥zj∥+ λ))

=

2m
X

j=1
∥zj∥2 −2λ

2m
X

j=1
∥zj∥−2mλ2

≥

2m
X

j=1
∥zj∥2 −2λ
√

2m




2m
X

j=1
∥zj∥2





1/2

−2mλ2

= ∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2,
(15)

where the third line is an application of Cauchy-Schwarz. Moreover,

∥ˆzlin −zlin∥=



2m
X

j=1
(−1)j(ˆzj −zj)



≤

2m
X

j=1
∥ˆzj −zj∥

≤

2m
X

j=1
λ

≤2mλ,

so

∥ˆzlin∥≥∥zlin∥−2mλ.
(16)

Let N ′ ∈Rd×n to be the matrix whose i-th column is ˆyini, equivalently N ′ = Ndiag(ˆy). Then

ˆzj = η(−1)jN ′cj,

where cj ∈Rn is given by

(cj)i =

T −1
X

t=0
b(t)
ij .

Due to symmetry of the noise distribution then the columns of N ′ are i.i.d. with distribution
N(0d, d−1(Id −vvT )). Therefore by Lemma 4.1 (and the assumptions d = Ω
 
n + log 1

δ

), with
probability at least 1 −δ over the randomness of the training data there exist positive constants C′, C
such that C′ ≤σmin(N ′) ≤σmax(N ′) ≤C. As a result

C′η∥cj∥≤∥ˆzj∥≤Cη∥cj∥.
(17)

We claim that for any j, j′ ∈[2m] and i ∈[n], (cj)i ≥α(cj′)i. Indeed, if ˆyif(W (t), xi) ≥1, then
b(t)
ij = b(t)
ij′ = 0, and if ˆyif(W (t), xi) < 1, then both b(t)
ij and b(t)
ij′ are elements of {α, 1}. This in
particular implies that

⟨cj, cj′⟩≥α⟨cj, cj⟩.

27


---Page Break---
Let us define

clin =

2m
X

j=1
cj.

Then

∥ˆzlin∥2 =



2m
X

j=1
(−1)j ˆzj



2

=



2m
X

j=1
ηN ′cj



2

≤Cη2



2m
X

j=1
cj



2

= Cη2∥clin∥2,
(18)

where we used that ∥N ′∥≤C in the third line. We also have

∥clin∥2 =

2m
X

j=1

2m
X

j′=1
⟨cj, cj′⟩

≥α

2m
X

j=1

2m
X

j′=1
⟨cj, cj⟩

= 2αm

2m
X

j=1
∥cj∥2.
(19)

Finally we combine our bounds for c, z, and ˆz:

∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2 ≤

2m
X

j=1
∥ˆzj∥2

≤Cη2
2m
X

j=1
∥cj∥2

≤Cη2

αm ∥clin∥2

≤C

αm∥ˆzlin∥2

≤C

αm (∥zlin∥+ 2mλ)2 .

Here we applied equations (15) in the first line, (17) in the second line, (19) in the third line, (18) in
the fourth line, and (16) in the fifth line. This establishes both the bounds claimed.

D.2
Benign overfitting

To establish benign overfitting in leaky ReLU networks, we first determine an upper bound on the
generalization error of the model in terms of the signal-to-noise ratio of the network weights.
Lemma D.2. Let ϵ ∈(0, 1). Suppose that

Amin
∥Z∥F
≥C2

s

(1 −γ)m log 1

ϵ
γd
.

Then for test data (x, y) as per Definition 2.1,

P(yf(W , x) ≤0) ≤ϵ.

28


---Page Break---
Proof. Recall that a test point (x, y) satisfies

x = y(√γv +
p

1 −γn),

where n ∼N(0d, 1

d(Id −vvT )). If yf(W , x) ≤0, then

0 ≥yf(W , x)

=

2m
X

j=1
(−1)jyσ(⟨wj, x⟩)

=

2m
X

j=1
(−1)jyσ(⟨ajv + zj, y(√γv +
p

1 −γn⟩)

=

2m
X

j=1
(−1)jyσ(y(√γaj +
p

1 −γ⟨zj, n⟩))

≥

2m
X

j=1
(−1)jyσ(y√γaj) −

2m
X

j=1

p

1 −γ|⟨zj, n⟩|

= √γAy −

2m
X

j=1

p

1 −γ|⟨zj, n⟩|.

When Amin ≥0, this implies that

γA2
min ≤(1 −γ)




2m
X

j=1
|⟨zj, n⟩|





2

≤2m(1 −γ)

2m
X

j=1
|⟨zj, n⟩|2

= 2m(1 −γ)∥Zn∥2

≤2m(1 −γ)∥Z∥2
F ∥n∥2,

where the second inequality is an application of Cauchy-Schwarz. So

P(yf(W , x) ≤0) ≤P

∥Zn∥2 ≥
γA2
min
2m(1 −γ)


.

By Lemma A.1, the above probability is less than ϵ if
r
γ
2m(1 −γ)Amin ≥C∥Z∥F

r

1
d log 1

ϵ ,

or equivalently,

Amin
∥Z∥F
≥C2

s

(1 −γ)m log 1

ϵ
γd
.

We will also need the number of positive labels to be (mildly) balanced with the number of negative
labels.
Lemma D.3. Let δ > 0 and suppose that ℓ= Ω
 
log 1

δ

. Let I ⊆[n] be an arbitrary subset such
that |I| = ℓ. Consider training data (X, y) as per the data model given in Definition 2.1. Then with
probability at least 1 −δ,
ℓ
4 ≤|{i ∈S : yi = 1}| ≤3ℓ

4 .

29


---Page Break---
Proof. For i ∈I let Yi be a random variable taking the value 1 if yi = 1 and 0 if yi = −1. Then the
Yi are i.i.d. Bernoulli random variables with P(Yi = 1) = 1

2. Let

Y =
X

i∈I
Yi = |{i ∈S : yi = 1}|

so that E[Y ] = l

2. By Chernoff’s inequality, for all t ∈(0, 1),

P
Y −ℓ

2

 ≥t ℓ

2


≤2e−Cℓt2.

Setting t = 1

2, we see that ℓ

4 ≤Y ≤3ℓ

4 with probability at least

1 −2 exp

−Cℓ

4


≥1 −δ

when ℓ= Ω
 
log 1

δ

.

We are now able to prove our main benign overfitting result for leaky ReLU networks.

Theorem 3.2. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A is approximately
margin-maximizing (Definition 2.3). If n = Ω
 
log 1

δ

, d = Ω(n), k = O(
n
1+m|A|2 ), and γ = Ω
  1

k


then there is a fixed positive constant C such that with probability at least 1 −δ over (X, ˆy)

P(yf(W , x) ≤0 | X, ˆy) ≤exp

−C ·
d
k(1 + m|A|2)


.

Proof. Since d = Ω(n) = Ω
 
n + log 1

δ

, by Lemma B.3, with probability at least 1 −δ

3 over the
randomness of the data, the max-margin classifier w∗satisfies

∥w∗∥≤C

s

1
γ +
k
1 −γ .

We denote this event by ω1. For s ∈{1, −1}, let Gs denote the set of i ∈G such that ⟨v, xi⟩= s. If
n = Ω
  1

δ

and k = O(n), then |G| = Ω
 
log 1

δ

. Under these assumptions, by Lemma D.3,

|Gs| ≥1

4|G| ≥Cn

for both s ∈{1, −1} with probability at least 1 −δ

3. We denote this event by ω2. For s ∈{1, −1},
let NGs ∈R|Gs|×d be the matrix whose rows are indexed by Gs and are given by the vectors ni
for i ∈Gs. As d = Ω(n) = Ω
 
n + log 1

δ

and the rows of NGs are drawn mutually i.i.d. from
N(0d, d−1(Id −vT )), the following holds by Lemma 4.1. With probability at least 1 −δ

3 over the
randomness of the training data, ∥NGs∥≤C for both s ∈{1, −1}. We denote this event by ω3. Let
ω = ω1 ∩ω2 ∩ω3. By the union bound P(ω) ≥1 −δ. We condition on ω for the remainder of this
proof.

Since W = A(X, ˆy) and A is approximately margin maximizing,

∥W ∥F ≤|A| · ∥w∗∥

≤C|A|

s

1
γ +
k
1 −γ .
(20)

30


---Page Break---
Let s ∈{−1, 1} be such that As = Amin. Since the network attains zero loss, for all i ∈Gs,

1 ≤ˆyif(W , xi)

=

2m
X

j=1
(−1)j ˆyiσ(⟨wj, xi⟩)

=

2m
X

j=1
(−1)jyiσ(⟨ajv + zj, √γyiv +
p

1 −γni⟩)

=

2m
X

j=1
(−1)jyiσ(√γajyi +
p

1 −γ⟨zj, ni⟩)

≤

2m
X

j=1
(−1)jyiσ(√γajyi) +

2m
X

j=1
|
p

1 −γ⟨zj, ni⟩|

= √γ

2m
X

j=1
(−1)jsσ(saj) +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|

= √γAs +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|

= √γAmin +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|.

Hence, we have either √γAs ≥1

2 or √1 −γ P2m
j=1 |⟨zj, ni⟩| ≥1

2 for all i ∈Gs. We consider these
two cases separately.

If √γAmin ≥1

2, then

Amin
∥Z∥F
≥Amin

∥W ∥F

≥
1
2√γ∥W ∥F

≥C
1
√γ|A|
q

1
γ +
k
1−γ

= C
1

|A|
q

1 +
kγ
1−γ

≥C
1

|A| + |A|
q

kγ
1−γ
.

Then by Lemma D.2, the network has generalization error less than ϵ when

1

|A| + |A|
q

kγ
1−γ
≥C

s

(1 −γ)m log 1

ϵ
γd

or equivalently
s

(1 −γ)m log 1

ϵ
γd
+

s

mk log 1

ϵ
d
≤C

|A|.

This is satisfied for ϵ = exp(−C ·
d
k(1+m|A|2)) for some different constant C when γ = Ω( 1

k), which
is true by assumption. So if √γAmin ≥1

2, then the network has generalization error less than ϵ
whenever ω occurs, which happens with probability at least 1 −δ.

31


---Page Break---
Now suppose that √1 −γ P2m
j=1 |⟨zj, ni⟩| ≥1

2 for all i ∈Gs. Squaring both sides of the inequality
and applying Cauchy-Schwarz, we obtain

1
4 ≤(1 −γ)




2m
X

j=1
|⟨zj, ni⟩|





2

≤2m(1 −γ)

2m
X

j=1
|⟨zj, ni⟩|2.

Summing over all i ∈Gs, we obtain

|Gs|

4
≤2m(1 −γ)
X

i∈Gs

2m
X

j=1
|⟨zj, ni⟩|2

= 2m(1 −γ)

2m
X

j=1
∥NGszj∥2,

Applying ω2 and ω3, we obtain the bound

n ≤Cm(1 −γ)

2m
X

j=1
∥NGszj∥2

≤Cm(1 −γ)

2m
X

j=1
∥NGs∥2∥zj∥2

≤Cm(1 −γ)

2m
X

j=1
∥zj∥2

= Cm(1 −γ)∥Z∥2
F
≤Cm(1 −γ)∥W ∥2
F .

Then applying (20),

n ≤Cm(1 −γ)∥W ∥2
F

≤Cm(1 −γ)|A|2
 1

γ +
k
1 −γ



≤Cm|A|2
 1

γ + k

.

This implies that

n ≤Cm|A|2

γ
or

k ≥
Cn
m|A|2 .

Neither of these conditions can occur if γ = Ω
  1

k

and k = O

n
|A|2m

. Thus, in all cases, the

network has generalization error less than exp(−C ·
d
k(1+m|A|2)) when ω occurs, which happens
with probability at least 1 −δ.

We are also able to show the lower bound for the generalization error stated in the main text.
Theorem 3.3. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A = AGD where
η, λ ∈R>0 satisfy Assumption 1. If n = Ω(k), d = Ω(n), and k = Ω(log 1

δ + 1

α), then there is a
fixed positive constant C such that with probability at least 1 −δ over (X, ˆy)

P(yf(W , x) ≤0 | X, ˆy) ≥exp

−C · d

αk


.

32


---Page Break---
Proof. We proceed along the lines of Theorem 3.2. For s ∈{1, −1}, let Bs denote the set of i ∈B
such that ⟨v, xi⟩= s. Note |B| = Ω
 
log 1

δ

. Under these assumptions, by Lemma D.3,

|Bs| ≥1

4|B| ≥Ck

for both s ∈{1, −1} with probability at least 1 −δ

3. We denote this event by ω1. For s ∈{1, −1},
let NBs ∈R|Bs|×d be the matrix whose rows are indexed by Bs and are given by the vectors ni
for i ∈Bs. As d = Ω(n) = Ω
 
k + log 1

δ

and the rows of NBs are drawn mutually i.i.d. from
N(0d, d−1(Id −vT )), the following holds by Lemma 4.1. With probability at least 1 −δ

3 over the
randomness of the training data, ∥NBs∥≤C for both s ∈{1, −1}. We denote this event by ω2. By
Lemma D.1, there is a constant C such that

∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2 ≤C

αm (∥zlin∥+ 2mλ)2 .
(21)

with probability at least 1 −δ

3. We denote this event by ω3. Let ω = ω1 ∩ω2 ∩ω3. By the union
bound P(ω) ≥1 −δ. We condition on ω for the remainder of this proof.

Let s ∈{1, −1} be such that As = max{A1, A−1}. Since the network attains zero loss, for all
i ∈Bs,
1 ≤ˆyif(W , xi)

=

2m
X

j=1
(−1)j ˆyiσ(⟨wj, xi⟩)

=

2m
X

j=1
(−1)jyiσ(⟨ajv + zj, −√γyiv +
p

1 −γni⟩)

=

2m
X

j=1
(−1)jyiσ(−√γajyi +
p

1 −γ⟨zj, ni⟩)

≤

2m
X

j=1
(−1)jyiσ(−√γajyi) +

2m
X

j=1
|
p

1 −γ⟨zj, ni⟩|

= √γ

2m
X

j=1
(−1)j+1sσ(saj) +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|

= −√γAs +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|.

From which we conclude
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩| ≥1 + √γAs ≥√γAs

for all such i. Squaring both sides of the inequality and applying Cauchy-Schwarz, we obtain

γAs ≤(1 −γ)




2m
X

j=1
|⟨zj, ni⟩|





2

≤2m(1 −γ)

2m
X

j=1
|⟨zj, ni⟩|2.

Summing over all i ∈Bs, we obtain

|Bs|γAs ≤2m(1 −γ)
X

i∈Bs

2m
X

j=1
|⟨zj, ni⟩|2

= 2m(1 −γ)

2m
X

j=1
∥NBszj∥2,

33


---Page Break---
Applying ω2 and ω3, we obtain the bound

kγAs ≤Cm(1 −γ)

2m
X

j=1
∥NBszj∥2

≤Cm(1 −γ)

2m
X

j=1
∥NBs∥2∥zj∥2

≤Cm(1 −γ)

2m
X

j=1
∥zj∥2

= Cm(1 −γ)∥Z∥2
F .

For k = Ω( 1

α), this inequality along with Assumption 1 implies that

C∥Z∥2
F ≤∥Z∥2
F + 2λ
√

2m∥Z∥F + 2mλ2

for a different constant C. With the last two inequalities and equation 21, we obtain the bound, for a
new constant C.

kγAs ≤C 1 −γ

α
(∥zlin∥+ 2mλ)2 .

We then apply k = Ω( 1

α) and Assumption 1 again to conclude that for some C,

∥zlin∥≥C

s

kγAsα

1 −γ .

Note that
Alin = A1 + A−1

1 + α
≤2As.

We then bound
Alin
∥zlin∥≤C
As
q

kγAsα

1−γ

≤C
r1 −γ

kγα

for some constant C. Now consider a test point (x, y), which satisfies

x = y(√γv +
p

1 −γn),

where n ∼N(0d, 1

d(Id −vvT )). Since the data distribution is symmetric,

P(yf(W , x) ≤0) ≥1

2P(yf(W , x) ≤0 or −yf(W , −x) ≤0)

≥1

2P(yf(W , x) −yf(W , −x) ≤0).

We see that

yf(W , x) −yf(W , −x) = (1 + α)

yAlin
√γ + ⟨zlin, n⟩
p

1 −γ


By Lemma C.1 we then can bound

P(y⟨w, x⟩≤0 | ω) ≥1

8 exp

−6d

π
γ
1 −γ
A2
lin
∥zlin∥2



≥exp

−Cd

αk



for a new constant C, provided Alin is positive. In the last line we can bound d

k below as d =
Ω(n) = Ω(k). If Alin is negative, then the generalization error is at least 1

4 which is also at least
exp(−Cd/(αk)).

34


---Page Break---
D.3
Non-benign overfitting

In this section we show that leaky ReLU networks trained on low-signal data exhibit non-benign
overfitting. As in the case of benign overfitting, we will rely on a generalization bound which depends
on the signal-to-noise ratio of the network.
Lemma D.4. Let W ∈R2m×d be the first layer weight matrix of a shallow leaky ReLU network
given by equation 1. Suppose (x, y) is a random test point sampled under the data model given in
Definition 2.1. If W is such that Alin ≥0 and

Alin

zlin
= O
r1 −γ

γd



then
P(yf(W , x) < 0) ≥1

8.

Alternatively, if Alin ≤0 then

P(yf(W , x) < 0) ≥1

4.

Proof. By Definition 2.1 (−x, −y) is identically distributed to (x, y), therefore

P(0 > yf(W , x)) = 1

2 (P(0 > yf(W , x)) + P(0 > −yf(W , −x)))

≥1

2P(0 > yf(W , x) ∪0 > −yf(W , −x))

≥1

2P (0 > yf(W , x) −yf(W , −x)) .

Next we compute

yf(W , x) −yf(W , −x)

=

2m
X

j=1
(−1)jy(σ(⟨wj, x⟩) −σ(⟨wj, −x⟩))

= (1 + α)

2m
X

j=1
(−1)jy⟨wj, x⟩

= (1 + α)

2m
X

j=1
(−1)j⟨ajv + zj, √γv +
p

1 −γn⟩

= (1 + α)√γ

2m
X

j=1
(−1)jaj + (1 + α)
p

1 −γ

*

n,

2m
X

j=1
(−1)jzj

+

= (1 + α)√γAlin + (1 + α)
p

1 −γ⟨n, zlin⟩.

The above two calculations imply that

P(0 > yf(W , x)) ≥1

2P(0 > yf(W , x) −yf(W , −x))

= 1

2P(0 > (1 + α)√γAlin + (1 + α)
p

1 −γ⟨n, zlin⟩)

= 1

2P

⟨−n, zlin⟩>
r
γ
1 −γ Alin


.

Suppose that Alin ≥0. As the noise distribution is symmetric ⟨n, zlin⟩
d= ⟨−n, zlin⟩. Therefore,

1
4P

|⟨n, zlin⟩| >
r
γ
1 −γ Alin


= 1

4P

|⟨n, u⟩| >
r
γ
1 −γ
Alin
∥zlin∥


,

35


---Page Break---
where u =
zlin
∥zlin∥is the unit vector pointing in the direction of zlin. Note by construction u ∈
span({v})⊥. If

Alin
∥zlin∥= O
r1 −γ

γd


,

then by Lemma A.2,

P

|⟨n, u⟩| >
r
γ
1 −γ
Alin
∥zlin∥


≥1

2

and therefore

P(0 > yf(W , x)) ≥1

4P

|⟨n, u⟩| >
r
γ
1 −γ
Alin
∥zlin∥


≥1

8.

If Alin < 0, then again by the symmetry of the noise

P(0 > yf(W , x)) ≥1

2P

⟨−n, zlin⟩>
r
γ
1 −γ Alin



≥1

2P (⟨−n, zlin⟩> 0)

= 1

4.

This establishes the result.

Theorem 3.4. Under the setting given in Assumption 2, let δ ∈(0, 1) and suppose A = AGD,
where η, λ ∈R>0 satisfy Assumption 1. If n = Ω(1), d = Ω
 
n + log 1

δ

and γ = O

α3

d

then the
following hold.

1. The algorithm AGD terminates almost surely after finitely many updates. With W =
AGD(X, ˆy), L(W , X, ˆy) = 0.

2. With probability at least 1 −δ over the training data (X, ˆy)

P(yf(W , x) < 0 | X, ˆy) ≥1

8.

Proof. If Alin < 0, then by Lemma D.4,

P(yf(W , x) < 0) ≥1

4.

So it suffices to consider the case Alin ≥0. Since d = Ω
 
n + log 1

δ

, by Lemma B.4, the max-margin
classifier w∗satisfies

∥w∗∥≤C
r
n
1 −γ

with probability at least 1 −δ

3 over the randomness of the input dataset. We denote this event by ω1
and condition on it for the rest of this proof. By Theorem 3.1,

∥W ∥≤C∥w∗∥

α√m

≤C

α

r
n
m(1 −γ).

36


---Page Break---
By Theorem 3.1, the network perfectly fits the training data, so for all i ∈[n], ˆyif(W , xi) ≥1, and
therefore

1 ≤|f(W , xi)|

=



2m
X

j=1
(−1)jσ(⟨wj, xi⟩)



≤

2m
X

j=1
|⟨wj, xi⟩|

=

2m
X

j=1
|⟨ajv + zj, √γyiv +
p

1 −γni⟩|

=

2m
X

j=1
|ajyi
√γ +
p

1 −γ⟨zj, ni⟩|

≤√γ

2m
X

j=1
|aj| +
p

1 −γ

2m
X

j=1
|⟨zj, ni⟩|.

This implies that either 1

2 ≤√γ P2m
j=1 |aj| or 1

2 ≤√1 −γ P2m
j=1 |⟨zj, ni⟩| for all i ∈[n]. We
consider both cases separately.

Suppose that 1

2 ≤√γ P2m
j=1 |aj|. Then squaring both sides and applying Cauchy-Schwarz, we obtain

1
4 ≤γ




2m
X

j=1
|aj|





2

≤2mγ

2m
X

j=1
|aj|2

≤2mγ

2m
X

j=1
∥wj∥2

= 2mγ∥W ∥2
F

≤
Cγn
α2(1 −γ).

This cannot occur if γ = O

α2

n

, and in particular it cannot occur if d = Ω(n) and γ = O

α3

d

.

Now suppose that 1

2 ≤√1 −γ P2m
j=1 |⟨zj, ni⟩| for all i ∈[n]. Squaring both sides and applying
Cauchy-Schwarz, we obtain

1
4 ≤(1 −γ)




2m
X

j=1
∥⟨zj, ni⟩|





2

≤2m(1 −γ)

2m
X

j=1
∥⟨zj, ni⟩∥2.

37


---Page Break---
Summing over all i ∈[n], we obtain

n
4 ≤2m(1 −γ)

n
X

i=1

2m
X

j=1
∥⟨zj, ni⟩∥2

= 2m(1 −γ)

2m
X

j=1
∥Nzj∥2

≤2m(1 −γ)∥N∥2
2m
X

j=1
∥zj∥2

= 2m(1 −γ)∥N∥2∥Z∥2
F .

Recall that d = Ω
 
n + log 1

δ

, and that the rows of N are i.i.d. with distribution N(0d, d−1(Id −
vvT )). So by Lemma 4.1, with probability at least 1 −δ

3 over the randomness of the dataset,
∥N∥≤C. We denote this event by ω2 and condition on it for the rest of this proof. So

∥Z∥2
F ≥
Cn
m(1 −γ).
(22)

Let λ =
√α

m . By Assumption 1, ∥w(0)
j ∥≤λ for all j ∈[2m]. So by Lemma D.1,

∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2 ≤C

αm(∥zlin∥+ 2mλ)2.
(23)

By (22),

∥Z∥F ≥
C√n
p

m(1 −γ)

≥C√n
√m
≥Cλ√nm

≥8λ
√

2m,

where the last line holds if n = Ω(1). Then by (23),

1
2∥Z∥2
F = ∥Z∥2
F −1

4∥Z∥2
F −1

4∥Z∥2
F

≤∥Z∥2
F −2λ
√

2m∥Z∥F −2mλ2

≤C

αm(∥zlin∥+ 2mλ)2

= C

αm(∥zlin∥+ 2√α)2.

Taking the square root of both sides and recalling that α ∈(0, 1) is a constant, we obtain

∥Z∥F ≤C∥zlin∥
√αm
+
C
√m.

This implies that either ∥Z∥F ≤
2C
√m or ∥Z∥F ≤2C∥zlin∥
√αm . The case ∥Z∥F ≤
2C
√m cannot happen,
since by (22),

∥Z∥F ≥
C′√n
p

m(1 −γ)

≥C′√n
√m

≥2C
√m

38


---Page Break---
when n = Ω(1). So we have ∥Z∥F ≤2C∥zlin∥
√αm . Again applying (22), we obtain

∥zlin∥≥C√αm∥Z∥F

≥C√αn
√1 −γ .

So

Alin
∥zlin∥≤C Alin
√1 −γ
√αn

= C√1 −γ
√αn

2m
X

j=1
(−1)jaj

≤C√1 −γ
√αn

√

2m




2m
X

j=1
|aj|2





1/2

≤C
p

m(1 −γ)
√αn




2m
X

j=1
∥wj∥2





1/2

= C∥W ∥F
p

m(1 −γ)
√αn

≤
C
α3/2 .

Here we used that Alin ≥0 and applied Cauchy-Schwarz in the third line. Then by Lemma D.4, if

C
α3/2 ≤O
r1 −γ

γd


,

then

P(yf(W , x) < 0) ≥1

8.

This occurs if γ = O

α3

d

. Hence, in all cases, we have shown that with the appropriate scaling, the

generalization error is at least 1

8 when both ω1 and ω2 occur. This happens with probability at least
1 −δ.

Appendix E
Formalizing benign overfitting as a high dimensional
phenomenon

To formalize benign overfitting as a high dimensional phenomenon we first introduce the notion of a
regime. Informally, a regime is a subset of the hyperparameters Ω∈N4 which describes accepted
combinations of the input data dimension d, the number of points in the training sample n, the number
of corrupt points k and the number of trainable model parameters p.

Definition E.1. A regime is a subset Ω⊂N4 which satisfies the following properties.

1. For any tuple (d, n, k, p) ∈Ωthe number of corrupt points is at most the total number of
points, k ≤n.

2. There is no upper bound on the number of points,

sup
(d,n,k,p)∈Ω
n = ∞.

A non-trivial regime is a regime which satisfies the following additional condition.

39


---Page Break---
3. Define the set of increasing sequences of Ωas Ω∗
=
{(nl, dl, kl, pl)l∈N
⊂
Ωs.t. liml→∞nl = ∞}. For any (nl, dl, kl, pl)l∈N ∈Ω∗it holds that

lim inf
l→∞
kl
nl
> 0.

Intuitively, a regime defines how the four hyperparameters (d, n, k, p) can grow in relation to one
another as n goes to infinity. A non-trivial regime is one in which the fraction of corrupt points in the
training sample is non-vanishing. In order to make a formal definition of benign overfitting as high
dimensional phenomenon we introduce the following additional concepts.

• A learning algorithm A = (Ad,n,p)(d,n,p)∈N3 is a triple indexed sequence of measurable
functions Ad,n,p : Rn×d × Rn →Rp.

• An architecture M = (fd,p)d,p∈N2 is a double indexed sequence of measurable functions
fd,p : Rd × Rp →R.

• A data model D = (Dd,n,k)(d,n,k)∈N3 is a triple indexed sequence of Borel probability
measures Dd,n,k defined over Rn×d × {±1} × Rd × {±1}.

With these notions in place we are ready to provide a definition of benign overfitting in high
dimensions.

Definition E.2. Let (ϵ, δ) ∈(0, 1]2, A be a learning algorithm, M an architecture, D a data model
and Ωa regime. If for every increasing sequence (dl, nl, kl, pl)l∈N ∈Ω∗there exists an L ∈N such
that for all l ≥L with probability at least 1 −δ over (X, ˆy), where (X, ˆy, x, y) ∼Ddl,nl,kl, it
holds that

1. yif(Adl,nl,pl(X, ˆy), xi) > 0 ∀i ∈[nl],

2. P(yf(Adl,nl,pl(X, ˆy), x) ≤0) ≤infW ∈Rnl×dl P(yf(W , x) ≤0) + ϵ,

then the quadruplet (A, M, D, Ω) (ϵ, δ)-benignly overfits. If (A, M, D, Ω) (ϵ, δ)-benignly overfits
for any (ϵ, δ) ∈(0, 1]2 then we say (A, M, D, Ω) benignly overfits.

Analogously, we define non-benign overfitting as follows.

Definition E.3. Let (ϵ, δ) ∈(0, 1]2, A be a learning algorithm, M an architecture, D a data model
and Ωa regime. If for every increasing sequence (dl, nl, kl, pl)l∈N ∈Ω∗there exists an L ∈N such
that for all l ≥L with probability at least 1 −δ over (X, ˆy), where (X, ˆy, x, y) ∼Ddl,nl,kl, it
holds that

1. yif(Adl,nl,pl(X, ˆy), xi) > 0 ∀i ∈[nl],

2. P(yf(Adl,nl,pl(X, ˆy), x) ≤0) ≥infW ∈Rnl×dl P(yf(W , x) ≤0) + ϵ,

then the quadruplet (A, M, D, Ω) (ϵ, δ)-non-benignly overfits. If (A, M, D, Ω) (ϵ, δ)-non-benignly
overfits for any (ϵ, δ) ∈(0, 1]2 then we say (A, M, D, Ω) non-benignly overfits.

One of the key contributions of this paper is proving (ϵ, δ)-benign and non-benign overfitting when
the architecture is a two-layer leaky ReLU network (equation 1), the learning algorithm returns the
inner layer weights of the network by minimizing the hinge loss over the training data using gradient
descent (Definition 2.2), and the regime satisfies the conditions d = Ω(n log 1/ϵ), n = Ω(1/δ),
k = O(n) and p = 2dm for some network width 2m, m ∈N.

Appendix F
Experiments

To further support our theory, we train shallow neural networks on the data model described in
Definition 2.1 and record the numerical results. Scripts to reproduce these experiments can be found
at https://github.com/kedar2/benign_overfitting. These experiments were run on the
CPU of a MacBook Pro M2 with 8GB of RAM. For our first experiment, we investigate the effect of
the ratio d

n on the generalization error of the network. Recall that by Theorem 3.2, the generalization

40


---Page Break---
error is bouned above by exp (−Cn) when d

n = Ω(1). In other words, if d

n is larger than a critical
threshold, then the generalizatione error decays quickly to 0 as n increases. We empirically confirm
this prediction in Figure 1, where we train several networks while varying d

n and n, and estimate
the generalization error for each configuration by averaging over 20 trials. Within each trial, we
trained the inner layer of the network with gradient descent using the hinge loss until the training loss
reached 0. For d

n greater than around 7, the generalization error rapidly decays to 0 as n →∞.

Figure 1: Generalization error of a two-layer leaky ReLU network trained to 0 hinge loss varying
n and d. Parameter settings: α = 0.1, γ = 5/n, m = 64, k = 0.1n, number of trials = 5, size of
validation sample = 1000.

Next, we train a two-layer network varying n and γ (Figure 2), holding constant the ratio d

n. Since γ
controls the signal-to-noise ratio of the data, the generalization error of the learned network decreases
as γ increases. For each value of n, the generalization error falls off steeply as γ reaches a certain
threshold. This threshold decreases as n increases, indicating that the network has higher noise
tolerance as n increases. This is in agreement with our theoretical results where we found that benign
overfitting occurs at the threshold γ = Ω
  1

k

(which is in this case Ω
  1

n

). We also see that the
generalization error for large values of γ is similar across different values of n. This effect is also
predicted by Corollary 3.2.1, since we scale both d and k proportionally to n.

41


---Page Break---
Figure 2: Generalization error of a two-layer leaky ReLU network trained to 0 hinge loss varying
γ and n. Parameter settings: α = 0.1, d = 2n, m = 64, k = 0.1n, number of trials = 10, size of
validation sample = 1000.

42


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The results stated in the abstract and introduction are stated formally in
Section 3 and then proven in Appendix D.
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
Justification: We discuss limitations of the work in the conclusion of the paper.
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

43


---Page Break---
Justification: All necessary assumptions are stated in the theorems. All theorems are proven
in full detail in Appendices C and D, with proof sketches appearing in Section 4.
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
Justification: We describe our experimental setup in Appendix F.
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

44


---Page Break---
Answer: [Yes]
Justification: We provide a link to our code to reproduce our experiments.
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
Justification: We describe our setup in Appendix F.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: Our experiments consist of a heatmap and a line chart with multiple plots.
It was not possible to add error bars without crowding the plots. In our description of
our experimental setup we describe the sample size, from which standard errors can be
computed.
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

45


---Page Break---
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

Justification: We describe the resources used in Appendix F.

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

Justification: The research conducted in the paper did not use data, assets, or human
participants; only studied existing models; and was conducted in accordance to the Code of
Ethics.

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

Justification: This paper is a purely theoretical study explaining behaviors seen in neural
networks in practice. There are no foreseeable societal impacts of this work.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

46


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

Justification: No new model or data is presented in this paper. The paper is a theoretical
study of neural networks.

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

Justification: The paper does not use existing assests.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

47


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

Justification: The paper does not use new assets.

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

48


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

49


---Page Break---
