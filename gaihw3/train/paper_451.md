Generalized Fast Exact Conformalization

Diyang Li
Cornell University
diyang01@cs.cornell.edu

Abstract

Conformal prediction converts nearly any point estimator into a prediction interval
under standard assumptions while ensuring valid coverage. However, the extensive
computational demands of full conformal prediction are daunting in practice, as it
necessitates a comprehensive number of trainings across the entire latent label space.
Unfortunately, existing efforts to expedite conformalization often carry strong
assumptions and are developed speciﬁcally for certain models, or they only offer
approximate solution sets. To address this gap, we develop a method for fast exact
conformalization of generalized statistical estimation. Our analysis reveals that the
structure of the solution path is inherently piecewise smooth, and indicates that
utilizing second-order information of difference equations sufﬁces to approximate
the entire solution spectrum arbitrarily. We provide a uniﬁed view that not only
encompasses existing work but also attempts to offer geometric insights. Practically,
our framework integrates seamlessly with well-studied numerical solvers. The
signiﬁcant speedups of our algorithm as compared to the existing standard methods
are demonstrated across numerous benchmarks.

1
Introduction

In modern algorithmic practice, quantifying uncertainty is crucial for accurate and reliable model
predictions. Conformal prediction [1] serves as a powerful statistical tool that leverages the observed
data to construct prediction intervals containing the outcome with a predeﬁned probability level.
It enjoys model-free coverage guarantee regardless of the underlying distribution of the data. In
recent years, conformal prediction has gained increasing attention from the community of machine
learning [2–5], data mining [6–8] and computer vision [9, 10]. This growing interest is attributed to
the attractive properties that it operates under the assumption of exchangeability, which is a weaker
condition than independence and identical distribution, allowing for a wider range of applications in
real-world scenarios where data may not meet strict statistical assumptions. Meanwhile, conformal
prediction can be combined with almost any existing point estimators, even when the model is
potentially misspeciﬁed [5].

While exhibits appealing properties, the application of conformal prediction often comes at a high
computational cost [11, 12]. Kindly note that in this paper we refer to the full conformal prediction
that does not discard training points as opposed to the split conformal prediction, as the latter involves
only one single ﬁtting. From a numerical perspective, when constructing a conformal prediction
set, one needs to exhaustively search all points (potential candidates) in the label space, where for
each point the learning model needs to be reﬁtted and the conformity score needs to be re-calculated.
In many scenarios like regression, the number of possible candidates is inﬁnite as the latent label
can take an uncountable number of possible values. Conventional conformal prediction works by
a grid-search type method to loop over the label space [13, 14], which discretize the interval of
interest and subsequently solve a sequence of individual optimization subproblems. To improve such
brute-force approach, there have been many efforts in community that devoted to develop better
algorithms for computing the prediction set. A natural idea is to generate the set of all solutions

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: Representative related work, which are instances of generalized parametric estimations.

Model
Reference
Exact
Loss
Regularizer
Constrained
Path Structure

Least Squares
[15]

Quadratic
\

Piecewise Linear
Ridge Regression
[16]

Quadratic
∥w∥2

Piecewise Linear
Empirical Risk Minimization
[17]

Convex
Convex

Piecewise Smooth
Elastic Net
[12]

Quadratic
∥w∥1

Piecewise Linear
Generalized Lasso Regression
[18]

Convex
∥w∥1

Piecewise Linear
General Formulation (ours)
Section 4

PCr
PCr

Piecewise Smooth

indexed by the latent label candidate using numerical continuation (a.k.a. homotopy) method, and we
name this set of optimal solutions as (exact) solution path, as illustrated in Figure 1.





x⊤
1
y1
...
...
x⊤
n
yn

n
x⊤
+1
yn+1





w⋆
(1)

w⋆
(2)

w⋆
(3)

Y

R1

Figure 1:
Diagram of setting on fast exact con-
formalization, where label candidate yn+1 should
loop over the whole latent label space Y and each
possible yn+1 ∈R1 corresponds to one point on
the (red) solution path. Conventional practice is to
reﬁt the model on each new yn+1 while we employ
the path-following algorithm to obtain the whole
solution spectrum within 1 execution.

Despite extensive theoretical and empirical ef-
forts, the understanding of conformalization
path remains rather deﬁcient. For some sim-
ple cases, closed-form characterizations of con-
formal prediction sets are available, such as k-
nearest neighbors, least squares regression [15],
and ridge regression [16] with quadratic loss.
The study by [12] presents an exact solution
path for conformalized Lasso and elastic net
using their statistical property and ℓ1-sparsity
analysis. Investigations into more general objec-
tives as discussed in [17, 18] incorporate linear
interpolation to approximate the intrinsic piece-
wise smooth structure, yielding prediction sets
lacking of ﬁnite-sample calibration. In other
terms, [17, 18] offer only an upper bound for
the approximation error and fail to control the
degree of approximation relative to the optimal
solutions. We summarize these relevant prior studies in Table 1. Given that existing exact algorithms
are either tailored to a select few models, or turn out to be grid-search type approaches that take a
very black-box approach, it prompts the following question:

Can we “open the black-box” by developing a methodology
that better exploits the structure of the path?

We answer this question positively by introducing a differential equation perspective to analyze
the ground truth solution path, which enables us to better reveal and exploit the fundamental path
structure. This more profound understanding enables us to build more generalized conformalization
algorithm and present improved computational guarantees.

1.1
Our contributions

The main contributions brought by this paper are summarized as follows.

Generalizable framework
This study aims to extend the application of fast exact conformalization
into generalized statistical estimation. We relax the assumptions by considering a cost function that is
no longer globally differentiable, but rather piecewise differentiable, while introducing constraints
on the weight vector to ﬁt more statistical models. In our analysis, the Clarke subdifferential of the
objective is derived, and the set of nonsmooth points of path can essentially be described as a level
set of certain smooth functions. By adopting a reparameterization regime, our framework effectively
broadens the scope of several existing algorithms in Table 1 and provides a uniﬁed view of them.

Theoretical insights
We analyze the underlying structure of path through the ﬁrst-order optimality
conditions of the regularized problem, identifying sufﬁcient conditions for a local path to be smooth
around a given point and reveal that the structure of the path is inherently piecewise smooth. More
precisely, if conditions are met, then the path is locally the projection of a higher-dimensional
smooth manifold onto the optimization space, thus offering preliminary geometric intuitions. Our

2


---Page Break---
investigation further suggests that leveraging the second-order information of difference equations
can approximate the solution spectrum arbitrarily.

Practical efﬁciency
Practically, our framework is not only straightforward to implement but also
computationally efﬁcient. With theoretical analysis, we present explicit expressions for the gradient
ﬂows of objective, which is homogenized and well aligned with standard forms used by mainstream
numerical libraries for ordinary differential equations (ODE), thereby easing the programming efforts.
When crossing potential kinks (or nonsmooth points), the computations are facilitated by boundary
conditions pre-set in the numerical solver. Notably, our algorithm eliminates the need for extensive
iterations for computing the entire solution spectrum, contrasting sharply with conventional baselines.
In experiments, we demonstrate the signiﬁcant computational speed advantages of our algorithm over
existing baselines, without compromising on accuracy.

2
Background

Notation
For a set J ⊆Rp, we denote by cl(J ) the closure and by int(J ) the interior of J w.r.t.
the natural topology on Rp. Deﬁne J (i) the i-th element of J and conv(J ) be the convex hull
of J , or conv(J ) = {v : v = Pk
i=1 ˆθiui, ui ∈J , ˆθi ∈R>0, Pk
i=1 ˆθi = 1}. The w(i) is the i-th
element of vector w and I(·) is an indicator function. Let ˙z(t) be the derivative dz(t)

dt
of function
z(t). Let O be the zero matrix and P {·} be the event probability. The sgn(·) is the sign function
sgn(x) =
x
|x|(x ̸= 0) or 0 (x = 0) that applied entrywise.

2.1
Model and assumptions

Given the dataset {xi, yi}n
i=1, where xi ∈Rp is the sample (covariate) and yi is the i-th label
(response) live in label space Y ⊆R1. We consider the generalized parametric estimation problem

w⋆∈arg min
w

n
X

i=1
Li

yi, ηw(xi)

+

m
X

j=1
λjΩj (w)

s.t.
gi (w) = 0,
1 ≤i ≤r,
hj (w) ≤0,
1 ≤j ≤s,

(1)

where ηw is the model prediction function, Li is the loss and Ωj is the regularizer. The gi, hj are
constraints on w, and parameter λj ∈R>0 controls the degree of regularity. In the following, we
will deﬁne the piecewise differentiability and state main assumptions that used throughout this work.
Deﬁnition 1 (PCr Function). Let f : U →R be continuous on the open set U ⊆Rp and
fi : U →R, i ∈{1, . . . , k} be a set of r-times continuously differentiable (or Cr) functions for
r ∈N ∪{∞}. If f(x) ∈{fi(x)}i∈{1,...,k} holds for all x ∈U, then f is an r-times piecewise
continuously differentiable (or PCr) function. The {f1, . . . , fk} is a set of selection functions of f.

When working with PCr-functions in a local sense, it is useful to only consider the selection functions
that have an impact on the local behavior around a given point.
Deﬁnition 2 (Essentially Active Set). Let f : U →R be a PCr function on the open set
U
⊆Rp with a set of selection functions {f1, . . . , fk}.
Denote IK = {1, . . . , k}.
Then
given a x1 ∈U, Ia
f (x1) ≜{i ∈IK : f(x1) = fi(x1)} is called the active set at x1, and the
Ie
f(x1) ≜{i ∈IK : x1 ∈cl ( int ( {x2 ∈U : f(x2) = fi(x2)}))} is the essentially active set at x1.

Assumption 1. We assume that Li and Ωj in (1) are PCr functions each with a set of selection
functions S

k∈Ie
Li{Dk
Li} and S

k∈Ie
Ωj {Dk
Ωj}, respectively.

Assumption 2. We assume that Ωj and Li are non-differentiable at w with multiple active
selection functions, where j ∈{1, . . . , m}, i ∈{1, . . . , n + 1}.
We further assume that
Ia
Ωj(w) ≡Ie
Ωj(w), Ia
Li(w) ≡Ie
Li(w) holds for all w considered and all Ωj, Li in the following.

Assumption 2 ensures that all selection functions we consider are actually relevant for the represen-
tation of Ωj, Li. i.e., it does not matter if we consider the active or the essentially active set in the
underlying optimizing space, which allows for an easier representation of Dk
Ωj and Dk
Li.

3


---Page Break---
2.2
Conformal prediction

Deﬁnition 3 (Symmetrical Algorithm). A deterministic algorithm A : (x1, . . . , xn) →A⋆is
symmetric if for any permutation τ of {1, . . . , n}, A(x1, . . . , xn)
a.s.
= A(xτ(1), . . . , xτ(n)).

Deﬁnition 4 (Conformity Score). The conformity score function A, symmetric in its ﬁrst n inputs, is
deﬁned as A(⃗x1, . . . ,⃗xn;⃗xn+1): R(p+1)×(n+1) →R, where ⃗xi ≜(xi, yi).

Conformal prediction starts from a conventional model ﬁtting stage, followed by the evaluation of
conformity score and the construction of prediction set. The score function A serves as a measure of
deviation or conformity, assessing the extent to which the new input xn+1 aligns with the previously
ﬁtted model. A higher conformity score indicates a better match between xn+1 and the model [1, 14].
For a new instance xn+1 where the prediction region is desired, the conformalization method operates
by assigning a p-value to each latent yn+1 ∈Y, formalized as

ˆpyn+1 = 1 −
1
n + 1

n+1
X

i=1
I (Ai ≥An+1) ,
(2)

where Ai ≜A(⃗x1, ..,⃗xi−1,⃗xi+1, ..,⃗xn+1;⃗xi). Speciﬁcally, in density estimation, A is deﬁned as
ηw⋆(xn+1), where ηw⋆is the density function estimated from the augmented dataset. For regression
tasks, A might be set as −|yn+1 −ηw⋆(xn+1)|, with ηw⋆being the regression function trained by
the dataset including n + 1 samples. Under the assumption of exchangeability among the pairs
{xi, yi}n+1
i=1 , the ˆpyn+1 returned by (2) has been demonstrated to be statistically valid [13, 14]. To
generate the prediction set Γ(·), one thresholds these p-values at a prescribed error level α ∈(0, 1),
resulting in
Γ(xn+1) =

yn+1 : ˆpyn+1 ≥α
	
.
(3)

Theorem 1. [19] Suppose that {xi, yi}n+1
i=1 are exchangeable and the ﬁtting algorithm A is symmetric.
Conformal prediction applied on {xi, yi}n
i=1 ∪{xn+1} outputs a set Γ(·) such that

P

y⋆
n+1 ∈Γ(xn+1)
	
≥1 −α,
(4)

where y⋆
n+1 is the ground truth (n + 1)-th label.

Theorem 1 (a.k.a. coverage guarantee) requires only exchangeability of input data and symmetry
of the conformity score function, which are met by nearly all prevalent model ﬁtting algorithms. In
alignment with the analysis taken in prior research, we treat yn+1 := yn+1(z) as a function of scalar
variable z. We utilize it to facilitate the traversal of yn+1 across the entire label space Y, and compute
the homotopy solution path {w⋆(z) : zmin ≤z ≤zmax} (also shown in Figure 1). We assume that
yn+1(·): [zmin, zmax] →Y is continuously differentiable in terms of z for simplicity.

3
Main results

We present our main results for fast exact conformalization. The discussions here focus on positive
z,1 but the derivation extends easily to include negative values as well, which will be discussed later.

3.1
Surrogate function

Lemma 1. Let f : U →R be a PCr-function on the open set U ⊆Rp and let Cr-functions
{f1, . . . , fk} be a set of selection functions of f. Then for any x ∈U, there exists an open
neighborhood U ′ ⊆U of x on which f is also a continuous selection of {fi : i ∈Ie
f(x)}.

Drawing insights from Lemma 1, we minimize the surrogate function of (1) as

min
w∈Rp Ez (w) :=

n+1
X

i=1
Li (yi, ηw(xi)) +

m
X

j=1
λjΩj(w)+ρ

r
X

i=1
|gi(w)|+ρ

s
X

j=1
max {0, hj(w)} , (5)

1For technicality reasons, we enforce z > 0 even the limit w⋆(0+) ≜limz→0+ w⋆(z) might be well-deﬁned.

4


---Page Break---
where ρ ∈R>0 and yn+1 = yn+1 (z). This deﬁnition of Ez (w) is meaningful regardless of whether
the contributing functions are convex. Denote model estimation terms P

iLi+P

j λjΩj as LM(w|z),
and the minimizer of (5) as w⋆(z). It is interesting to compare Ez (w) to the Lagrangian function

Lz(w) := LM (w|z) +

r
X

i=1
˜λigi(w) +

s
X

j=1
˜µjhj(w),
(6)

which captures the behavior of LM(·) near the optimum. At a constrained minimum w⋆, the
Lagrangian satisﬁes the stationarity condition ∇L(w⋆) = 0; its inequality multipliers ˜µj are nonneg-
ative and satisfy the complementary slackness ˜µjhj(w⋆) = 0. In the penalized (5), one usually takes

ρ > max{|˜λ1|, . . . , |˜λr|, ˜µ1, . . . , ˜µs},
(7)

which creates the favorable circumstances: (i) Lz(w) ≤Ez(w) for all w, (ii) Lz(w) ≤LM(w|z) =
Ez(w) for all feasible w, (iii) Lz(w⋆) = LM(w⋆|z) = Ez(w⋆) with profound consequences.

Theorem 2. Our surrogate function Ez(w) is increasing in ρ. Furthermore, Ez(w) is strictly convex
for one ρ > 0 if and only if it is strictly convex for all ρ > 0. Likewise, it is coercive for one ρ > 0 if
and only if it is coercive for all ρ > 0. Finally, if LM(·) is strictly convex (or coercive), then Ez(w)
for all ρ are strictly convex (or coercive).

Given Theorem 2, several classical results [20] state that minimizing Ez(w) is effective in minimizing
LM(·) subject to the constraints if we choose ρ by (7).

3.2
Conformal path characterization

Deﬁnition 5 (Clarke Subdifferential). For continuous function f : U →R on the open set U ⊆Rp,
let Θ ⊆U be the set of points in which f is not differentiable. The Clarke subdifferential of f at
x ∈U is deﬁned as

∂f(x) ≜conv({φ ∈Rp : ∃{xi}∞
i=1 ∈Rp\Θ with lim
i→∞xi = x, lim
i→∞∇f(xi) = φ}).

Speciﬁcally, if f is continuously differentiable in x, ∂f(x) = {∇f(x)}. While PCr-functions are
generally non-smooth, we can use the Clarke subdifferential to obtain ﬁrst-order optimality of (5),

n
X

i=1

X

k∈Ia
Li(w⋆)

ˆθk
Li(w⋆)∇Dk
Li(yi, ηw⋆(xi))+
X

k∈Ia
Ln+1(w⋆)

ˆθk
Ln+1(w⋆)∇Dk
Ln+1(yn+1(z), ηw⋆(xn+1))≜D′(w⋆),

ρ

r
X

i=1
ˆθgi∇gi(w⋆)+ρ

s
X

j=1
ˆθhj∇hj(w⋆)+

m
X

j=1

X

k∈Ia
Ωj (w⋆)
λj ˆθk
Ωj(w⋆)∇Dk
Ωj (w⋆)+D′(w⋆) = 0, (8)

where ˆθk
Ωj, ˆθk
Li is the k-th auxiliary parameter for convex hull of each Ωj, Li. The (8) is accompanied
by the active sets conditions and the subdifferentials conditions, rewritten in detail as

Dk
Li(w⋆)−Dli
Li(w⋆) = 0, ∀k ∈Ia
Li(w⋆)\{li}, ∀i ∈¯Ie
Li
Dk
Ωj(w⋆)−Drj
Ωj(w⋆) = 0, ∀k ∈Ia
Ωj(w⋆)\{rj}, ∀j ∈¯Ie
Ωj
X

k∈Ia
Li(w⋆)

ˆθk
Li(w⋆)−1 = 0, ˆθk
Li(w⋆) ≥0, 1 ≤i ≤n + 1

X

k∈Ia
Ωj(w⋆)

ˆθk
Ωj(w⋆)−1 = 0, ˆθk
Ωj(w⋆) ≥0, 1 ≤j ≤m

(9)

where rj, li is randomly selected from Ia
Ωj, Ia
Li and being ﬁxed, with coefﬁcients satisfying ˆθgi ∈





{−1}
gi(w) < 0
[−1, 1]
gi(w) = 0
{1}
gi(w) > 0
, ˆθhj ∈






{0}
hj(w) < 0
[0, 1]
hj(w) = 0
{1}
hj(w) > 0
. In this work, we specialize to the case where the

5


---Page Break---
constraint functions gi (1 ≤i ≤r) and hj (1 ≤j ≤s) are afﬁne, i.e., the gradients ∇gi(w), ∇hj(w)
are constant.2 We deﬁne gi and hj as constraint residuals gi(w) := v⊤
i w −di, hj(w) := ω⊤
j w −ej.

We keep track of the following index sets determined by signs of constraint residuals:

NE = {i : gi(w) = v⊤
i w−di < 0}, ZE = {i : gi(w) = v⊤
i w−di = 0}, PE = {i : gi(w) = v⊤
i w−di > 0},

NI = {j : hj(w) = ω⊤
j w−ej < 0}, ZI = {j : hj(w) = ω⊤
j w−ej = 0}, PI = {j : hj(w) = ω⊤
j w−ej > 0}.
(10)

To characterize the path, we further introduce a reparameterization in terms of an auxiliary variable
t ≥0 (thought of as time), whereby for a given T > 0 we introduce functions z(·) : [0, T] →
[zmin, zmax] and ξ(·) : [zmin, zmax] →R such that: ξ(·) is Lipschitz, z(·) is differentiable on (0, T),
and we have ˙z = ξ(z(t)) for all t ∈(0, T). In a slight abuse of notation, we deﬁne the path w.r.t. t
as
n
w⋆(t) ≜w⋆(z(t)) : t ∈[0, T]
o
. To enhance structural clarity, we denote nZ = |ZE ∪ZI| and
represent the certain matrix inversion as
 ˜H(w⋆|z)
U⊤
Z
UZ
OnZ×nZ

−1
=

P(w⋆|z)
Q(w⋆|z)
Q⊤(w⋆|z)
R


,
(11)

where the rows of matrix UZ are the constant differentials, v⊤
i for i ∈ZE and ω⊤
j for j ∈ZI, and

˜H(w⋆|z) ≜

n
X

i=1

X

k∈Ia
Li(w⋆)

ˆθk
Li(w⋆)∇2Dk
Li(yi, ηw⋆(xi)) +

m
X

j=1

X

k∈Ia
Ωj (w⋆)
λj ˆθk
Ωj(w⋆)∇2Dk
Ωj (w⋆)

+
X

k∈Ia
Ln+1(w⋆)

ˆθk
Ln+1(w⋆)∇2Dk
Ln+1(yn+1(z) , ηw⋆(xn+1)) .

(12)

Now we are fully equipped to unveil the structure of homotopy path, as shown in next subsection.

3.3
Piecewise smooth structure & Uniﬁed view

To describe the structure, we deﬁne the kink (non-smooth point) as the point that ˆθk
Ωj, ˆθk
Li hit the
restriction bound in (9), or set in (10) is violated so the entire structure changes.
Theorem 3. Given an optimum (w⋆
0, z0) at t0, and assume that t0 is not a kink, then there exists an
open neighborhood of t0 such that w⋆(t) is a C1 function of t and satisﬁes the following autonomous
system

˙w⋆(t) ≜Υ(w⋆, z) = −
X

k∈Ia
Ln+1(w⋆)

ˆθk
Ln+1(w⋆)ξ(z)·P(w⋆|z)
h
∂z∇Dk
Ln+1(yn+1(z) , ηw⋆(xn+1))
i
,

(13)
where P(w⋆|z) is obtained from the deﬁnition in (11), and the ˆθLi, ˆθΩj can be solved from (8).

The (13) offers an explicit gradient ﬂow of the solution path in underlying optimization space.
Solving this ODE numerically would recover all the optimal parametric solutions, which provide
the conformity scores and one can obtain the required p-values (2) that used for exact conformal
prediction via (3).
Theorem 4. The optimality solution (w⋆(t), z(t)) has a unique trajectory for t ∈(0, T). The
w⋆(t) is continuous if selections Dk
Li(·), Dk
Ωj(·) are convex. Furthermore, if gradients of constraints
{∇gi(w) : ∇gi(w) = 0}∪{∇hj(w) : ∇hj(w) = 0} are afﬁnely independent at the solution w⋆(z)
over an open neighborhood of z, then the coefﬁcient paths ˆθgi, ˆθhj are unique and continuous at z.
Theorem 5. On a optimality path with set conﬁguration (10), the coefﬁcients for constraints satisﬁes

rZ ≜
 ˆθZE(w⋆)
ˆθZI(w⋆)


= −Q(w⋆)
1

ρD′(w⋆) + u⊤
¯
Z


,
(14)

2In principle a similar algorithm can be developed for the general convex constraint where the hj are relaxed
to convex, but that is beyond the scope of current paper.

6


---Page Break---
where Q(w⋆) is from (11), and

u⊤
¯
Z := −
X

i∈NE
vi +
X

i∈PE
vi +
X

j∈PI
ωj.

Although Theorem 3, 4, and 5 are highly technical and may difﬁcult to grasp on ﬁrst glance, they lay
the groundwork for practical application, as illustrated later in Section 4. Speciﬁcally, Theorem 4
ensures continuity, which is fundamental for the numerical ODE solvers. Theorem 5, on the other
hand, provides a rule for handling constraints.

Theorem 6. Suppose that Dk
Li(·) is µ-strongly convex for µ ≥0, Dk
Ωj(·) is σ-strongly convex for σ >
0, and P(w⋆|z), ∂zDk
Ln+1(·), ∂z∇Dk
Ln+1(·) are all locally ℓ-Lipschitz continuous. Suppose further
that ξ(·) is Lipschitz continuous on [zmin, zmax] and satisﬁes |ξ(z)| ≤¯C for all z ∈[zmin, zmax].
Then, it holds that Υ(·, ·) deﬁned in (13) is uniformly ℓΥ-Lipschitz continuous with ℓΥ = ¯Cℓ2 +
2 ¯
Cℓ
(n+1)µ+Pm
j=1 λjσ for any z ∈[zmin, zmax] when the active selections IL, IΩare ﬁxed.

The essence of Theorem 6 lies in its suggestion that the dynamics of w⋆(t) is piecewise continuous,
i.e., w⋆(t) maintains smoothness between two adjacent kinks. By considering speciﬁc choices
of ξ(·) and yn+1(z), our system (13) generalizes some previously studied methodologies in fast
conformalization. First, consider the scenario with an equally spaced discretization of the interval
[0, T], namely tk = k · h′ for some ﬁxed step-size h′ > 0. Thus, the sequence zk := z(tk) is
approximately given by zk+1 ≈zk + h′ · ξ(zk). Intuitively, the choice of ξ(·) controls the dynamic
of z(·) and generalizes some previously considered sequences {zk} for the problem (5). For example,
letting ξ(z) := 1 we recover the arithmetic sequence in [12] and letting ξ(z) := −z we recover the
geometric sequence in [17].

4
Fast conformalization algorithm

The complete algorithm on the fast exact conformalization is outlined in the plate referred as
Algorithm 1.

Algorithm 1 Fast Exact Conformalization Algorithm
Input: Training data {xi, yi}n
i=1, new covariate xn+1, range [zmin, zmax], initial solution w⋆
0, regu-
larization strength {λj}m
j=1, miscoverage level α ∈(0, 1).
1: \\ Full Path Generation
2: z ←0, set NE, PE, PI and all IL, IΩby w⋆
0.
3: while 0 ≤z ≤zmax do
4:
while partitions NE, PE, PI, IL, IΩare met do
5:
Calculate ˜H(w⋆|z), rZ as in (12), (14).
6:
Solve ODE system (13).
7:
end while
8:
Update NE, PE, PI, IL, IΩby index violator(s).
9: end while
10: z ←0.
11: while zmin ≤z ≤0 do
12:
Repeat
the
above
procedure
analogously
for
negative
values
of
z,
obtaining
{w⋆(z) : zmin ≤z ≤0}.
13: end while
14: \\ Conformal Set Generation
15: for i = 1 to n + 1 do
16:
Calculate conformity score path Ai for i-th sample.
17: end for
18: Calculate the path of ˆpyn+1 by (2).
19: Γ(xn+1) ←

yn+1 : ˆpyn+1 ≥α
	
.
Output: Conformal prediction set Γ(xn+1).

7


---Page Break---
Table 2: Numerical results for average empirical coverage, average length of the conformal prediction
set, and the total number of kinks observed. One standard error is given in the parenthesis following
the average number.

Dataset
Model (Parameter)
Grid1/Grid2
SCP
Exact
# Kinks
Coverage
Length
Coverage
Length
Coverage
Length

fried
NLS(\)
0.81(0.003)
5.82(0.53)
0.81(0.007)
5.55(0.94)
0.80(0.009)
5.30(0.72)
6
cadata
NLS(\)
0.82(0.007)
5.54(0.12)
0.82(0.007)
5.63(0.94)
0.80(0.003)
5.35(0.02)
2
delta
NLS(\)
0.81(0.008)
5.63(0.16)
0.81(0.002)
5.79(0.53)
0.80(0.002)
5.26(0.27)
5
cadata
GFM(λ1 =0.03)
0.82(0.001)
11.64(0.13)
0.81(0.004)
11.38(0.36)
0.80(0.003)
11.34(0.27)
7
cadata
GFM(λ1 =0.02)
0.83(0.008)
10.72(0.91)
0.81(0.004)
10.39(0.97)
0.80(0.001)
10.13(0.21)
5
elevator
GFM(λ1 =0.03)
0.83(0.003)
11.09(0.58)
0.81(0.007)
11.34(0.97)
0.80(0.002)
10.68(0.81)
9
fried
IGR(λ1 =0.1, λ2 =0.02)
0.81(0.002)
19.14(0.78)
0.81(0.001)
19.69(0.96)
0.81(0.002)
19.05(0.19)
12
cadata
IGR(λ1 =0.1, λ2 =0.02)
0.82(0.003)
19.99(0.33)
0.80(0.002)
19.16(0.29)
0.80(0.002)
19.14(0.82)
13
elevator
IGR(λ1 =0.2, λ2 =0.05)
0.82(0.003)
19.99(0.85)
0.81(0.001)
19.53(0.53)
0.81(0.005)
19.48(0.75)
7

2097
4194
6291
8388
10485
12582
14679
16776
Size of dataset (delta)

0

5

10

15

20

25

30

Running Time of NLS (s)

Grid1
Grid2
Exact

900
1800
2700
3600
4500
5400
6300
7200
Size of dataset (fried)

0

5

10

15

20

25

30

Running Time of GFM (s)

Grid1(
1 = 0.01)
Grid1(
1 = 0.03)

Grid2(
1 = 0.01)

Grid2(
1 = 0.03)

Exact(
1 = 0.01)
Exact(
1 = 0.03)

4500
9000
13500
18000
22500
27000
31500
36000
Size of dataset (fried)

0

5

10

15

20

25

Running Time of IGR (s)

Grid1(
1, 2 = 0.1, 0.02)

Grid1(
1, 2 = 0.2, 0.03)

Grid2(
1, 2 = 0.1, 0.02)

Grid2(
1, 2 = 0.2, 0.03)

Exact(
1, 2 = 0.1, 0.02)

Exact(
1, 2 = 0.2, 0.03)

Figure 2: The running time under different sizes of datasets.

4.1
On solving (11)

To avoid the direct computation of matrix inversion (11), our practical algorithm involves sweep
operator [22, 23]. Suppose A is an p × p symmetric matrix, sweeping on the k-th diagonal entry
Akk ̸= 0 of A results in a matrix bA with entries

ˆAkk = −1

Akk ,
ˆAik = Aik

Akk (i ̸= k) ,
ˆAkj = Akj

Akk (j ̸= k) ,
ˆAij = Aij −AikAkj

Akk
(i, j ̸= k) . (15)

Since the sweeping (15) preserves symmetry, all operations can be performed solely on either the
lower or upper-triangular part of A to ease the computational burden [24]. To begin with, we initiate

with a sweeping tableau as
h
˜H(w⋆|z)
*
UZ
O

i
, and further sweeping of diagonal entries of block ˜H

yields M ≜
h M11
*
M21
M22

i
. Then we reinitialize our new tableau in the form of
h −M22
M21
*
−M11

i
,

and further sweeping of diagonal entries of block M22 makes
h R
Q⊤(w⋆|z)
*
P(w⋆|z)

i
. Compared to direct

inversion, it also decreases a O(p2 + n2
Z) storage space.

4.2
Efﬁciency

Algorithm 1 shows a very favourable behavior empirically, and converges remarkably faster than
the standard grid-search type algorithm. We argue that this observation is actually quite natural.
Indeed, our algorithm can follow the ground truth solution path, and the numerical integration process
is fully deterministic, which avoids large ﬂuctuations between the iteration steps like stochastic
gradient descent. Therefore, the solving process of Algorithm 1 exhibits a more stable character being
completely deterministic and has no extensive loops, which explains the much faster convergences
observed in practice. In contrast, the standard grid-search type method would cost N-times the
original batch iterative algorithms, where N is the number of grid points.

5
Numerical experiments

We provide experimental results on real-world benchmarks to validate our derived algorithm. All
experiments presented in this study were conducted on a workstation running the Ubuntu 18.04
operating system, equipped with Intel Xeon Gold 5218R CPU×64 and 60.9 GB of RAM. We
integrate a system of ordinary differential equations using lsoda from the FORTRAN library, where an

8


---Page Break---
interface for SciPy is available using the odepack. The concrete parameter settings of ODE solver
are shown in the Table 3, wherein the numerical solver exploit the Runge-Kutta method of order 4 or
5. The parameterizers are set to yn+1(z) = 4z, ξ(z) = 1, respectively.

Table 3: List of the key parameters in the numerical solver.

Parameters
Descriptions
Values
rtol
allowed relative error in the solution
1e-6
atol
allowed absolute error in the solution
1.49e-8
tcrit
vector of critical points
set by known / explicit kinks
h0
initial step size for the integration
0.02*(zmax −zmin)
hmax
maximum absolute step size allowed
0.1*(zmax −zmin)
hmin
minimum absolute step size allowed
1e-7
mxstep
maximum number of steps allowed for each point
400
mxordn
maximum order to be allowed for the non-stiff method
12
mxords
maximum order to be allowed for the stiff method.
5
ixpr
extra printing at method switches
True
mxhnil
maximum number of messages printed
15
tfirst
the required order of the ﬁrst two arguments
True
full_output
return a dictionary of optional outputs
True

Model
For evaluation, here we employ 3 speciﬁc forms of (1), i.e., Nonnegative Least Squares
(NLS) [25], Graph-guided Fused Model (GFM) [26], and Inverse Gaussian Regression (IGR) [27].

42
84
126
168
210
252
294
336
Number of grids (fried dataset)

0

5

10

15

20

25

30

Running Time of GFM (s)

Grid1(
1 = 0.03)

Grid1(
1 = 0.05)

Grid2(
1 = 0.03)

Grid2(
1 = 0.05)

Exact(
1 = 0.03)

Exact(
1 = 0.05)

Figure 3: The training time against
grid numbers.

Conformalization
A conformal prediction set with target
coverage level 0.8 (α = 0.2) is calculated for each sample
in testing set using each of the 4 methods, i.e., the standard
grid-point evaluation method (Grid1) [1], grid-point method
with warm-restart strategy (Grid2) [28], the split conformal
prediction method (SCP) [4], and our exact conformalization
method in Algorithm 1 (Exact). We use the conformity
score function Ai = −|yi −ηw⋆(xi)|. Conventionally, the
interval [ymin
n+1, ymax
n+1] (part of the input) can be chosen sim-
ply as [y[1], y[n]], where y[1] ≤y[2] ≤· · · ≤y[n] are the
order statistics of the response variable. In experiments
we set the search range even more conservatively, enlarg-
ing the sample range by 50% of length [ymin
n+1, ymax
n+1] :=

y[1] −0.25(y[n] −y[1]), y[n] + 0.25(y[n] −y[1])

.

0
1
2
3
0.0

0.2

0.4

0.6

0.8

1.0

Probability

0.971

0.027 0.002 0.0

# kinks

7

6

5

4

Running time (log(s))

=1e-2

100
200
300
400
500
k

7

6

5

4
=2e-2 

λ1

λ1

Figure 4: The histogram of kink numbers and the run-
ning time of Algorithm 1 against k, where k lives in
yn+1 := k · z(k ∈N).

Dataset
Our experiments were con-
ducted using real-world datasets. We em-
ploy real-world datasets from OpenML
[29] and UCI repository [30] in simula-
tions. We randomly partition the dataset
into training set, testing set, and calibra-
tion set (used in SCP) with 70%, 10%, and
20% of the total samples. To facilitate op-
timization, we have standardized the entire
original dataset by removing the mean and
scaling to unit variance for the features, and
adjusting the mean of all labels to 0.

Setup
Our central claim is twofold, en-
compassing both accuracy and efﬁciency.
We ﬁrst report the average empirical coverage, average length of the prediction set in Table 2. Re-
garding running efﬁciency, we present average training time per dataset in Figure 2 while varying the
scale of training set. In Figure 3, we compare the training times when different grid numbers are used
in Grid1 and Grid2. We further plot the histogram of kink numbers, and the running time against
various yn+1(·) in Figure 4.

Results & Analysis
From Table 2 we observe that all these methods provide valid and nearly
perfect coverage. The grid and exact method give similar lengths, where the slight difference is due

9


---Page Break---
to the rounding between neighboring grid points. The SCP produces wider intervals due to a less
efﬁcient use of data. Given Figure 2, our exact method is much faster than the baselines, with same
solid performance. We can also learn from the Figure 3 that the time of Algorithm 1 still compares
favorably against the grid-search type method, even when the grid is sparse. Figure 4 found that
there exists tiny number of kinks in majority runnings, which offers hope for the future expansion of
our algorithm, and indicates that the choice of yn+1 can make a difference in efﬁciency, as it will
determine the solving interval and the total query times of gradient.

6
Conclusion

In this work, we present a uniﬁed framework and an elaborate algorithm with statistical analysis for
fast exact conformalization regarding generalized parametric estimation. We illustrate the strong and
competitive performance of proposed methods in a series of benchmarks.

In future work, a potential direction is to consider scenarios where labels are multidimensional, such as
multi-task learning, in which the label space Y would be indexed by multiple independent parameters
(z1, . . . , zK). Under such conditions, the homotopy solution path would extend to a solution surface,
and our ODE system could be reformulated as a corresponding system of partial differential equations.
Additionally, compared to some previous work, Algorithm 1 offers increased speed but at the cost of
higher memory requirements. It necessitates storing all training samples throughout the process for
gradient queries. We believe it would be interesting to use recent advancements like the Kronecker-
factored approximate method to potentially enhance the memory scalability.

Acknowledgement

I would like to thank the anonymous reviewers for their valuable comments and suggestions to
improve the presentation of this paper.

References

[1] Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine
Learning Research, 9(3), 2008.

[2] Victor Chernozhukov, Kaspar Wüthrich, and Yinchu Zhu. Distributional conformal prediction.
Proceedings of the National Academy of Sciences, 118(48):e2107794118, 2021.

[3] Osbert Bastani, Varun Gupta, Christopher Jung, Georgy Noarov, Ramya Ramalingam, and Aaron
Roth. Practical adversarial multivalid conformal prediction. Advances in Neural Information
Processing Systems, 35:29362–29373, 2022.

[4] Matteo Fontana, Gianluca Zeni, and Simone Vantini. Conformal prediction: a uniﬁed review of
theory and new challenges. Bernoulli, 29(1):1–23, 2023.

[5] Anastasios N Angelopoulos, Stephen Bates, et al. Conformal prediction: A gentle introduction.
Foundations and Trends® in Machine Learning, 16(4):494–591, 2023.

[6] Jonathan Alvarsson, Staffan Arvidsson McShane, Ulf Norinder, and Ola Spjuth. Predicting with
conﬁdence: using conformal prediction in drug discovery. Journal of Pharmaceutical Sciences,
110(1):42–49, 2021.

[7] Margaux Zaffran, Olivier Féron, Yannig Goude, Julie Josse, and Aymeric Dieuleveut. Adaptive
conformal predictions for time series. In International Conference on Machine Learning, pages
25834–25866. PMLR, 2022.

[8] Chen Xu and Yao Xie. Conformal prediction for time series. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2023.

[9] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan, and Jitendra Malik. Uncer-
tainty sets for image classiﬁers using conformal prediction. In International Conference on
Learning Representations, 2020.

10


---Page Break---
[10] Paul Melki, Lionel Bombrun, Boubacar Diallo, Jérôme Dias, and Jean-Pierre Da Costa. Group-
conditional conformal prediction via quantile regression calibration for crop and weed classiﬁ-
cation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
614–623, 2023.

[11] Harris Papadopoulos, Vladimir Vovk, and Alexander Gammerman. Regression conformal
prediction with nearest neighbours. Journal of Artiﬁcial Intelligence Research, 40:815–840,
2011.

[12] Jing Lei. Fast exact conformalization of the lasso using piecewise linear homotopy. Biometrika,
106(4):749–764, 2019.

[13] Jing Lei, James Robins, and Larry Wasserman. Distribution-free prediction sets. Journal of the
American Statistical Association, 108(501):278–287, 2013.

[14] Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J Tibshirani, and Larry Wasserman.
Distribution-free predictive inference for regression. Journal of the American Statistical Associ-
ation, 113(523):1094–1111, 2018.

[15] Geoffrey S Watson. Linear least squares regression. The Annals of Mathematical Statistics,
pages 1679–1699, 1967.

[16] Evgeny Burnaev and Vladimir Vovk. Efﬁciency of conformalized ridge regression. In Confer-
ence on Learning Theory, pages 605–622. PMLR, 2014.

[17] Eugene Ndiaye and Ichiro Takeuchi. Computing full conformal prediction set with approximate
homotopy. Advances in Neural Information Processing Systems, 32, 2019.

[18] Etash Kumar Guha, Eugene Ndiaye, and Xiaoming Huo. Conformalization of sparse generalized
linear models. In International Conference on Machine Learning, pages 11871–11887. PMLR,
2023.

[19] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random
world, volume 29. Springer, 2005.

[20] Andrzej Ruszczynski. Nonlinear optimization. Princeton university press, 2011.

[21] David Avis and Komei Fukuda. A pivoting algorithm for convex hulls and vertex enumeration of
arrangements and polyhedra. In Proceedings of the seventh annual symposium on Computational
geometry, pages 98–104, 1992.

[22] James H Goodnight. A tutorial on the sweep operator. The American Statistician, 33(3):
149–158, 1979.

[23] Roderick JA Little and Donald B Rubin. Statistical analysis with missing data, volume 793.
John Wiley & Sons, 2019.

[24] Kenneth Lange, J Chambers, and W Eddy. Numerical analysis for statisticians, volume 2.
Springer, 1999.

[25] Rasmus Bro, Sijmen De Jong, and Sijmen De Jong. A fast non-negativity-constrained least
squares algorithm. Journal of Chemometrics: A Journal of the Chemometrics Society, 11(5):
393–401, 1997.

[26] Xi Chen, Qihang Lin, Seyoung Kim, Jaime G Carbonell, and Eric P Xing. An efﬁcient proximal
gradient method for general structured sparse learning. stat, 1050, 2010.

[27] Venkata Seshadri. The inverse Gaussian distribution: statistical theory and applications, volume
137. Springer Science & Business Media, 2012.

[28] Parikshit Ram.
On the optimality gap of warm-started hyperparameter optimization.
In
International Conference on Automated Machine Learning, pages 12–1. PMLR, 2022.

[29] Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. Openml: networked science
in machine learning. ACM SIGKDD Explorations Newsletter, 15(2):49–60, 2014.

11


---Page Break---
[30] Arthur Asuncion and David Newman. Uci machine learning repository, 2007.

[31] Fuzhen Zhang. The Schur complement and its applications, volume 4. Springer Science &
Business Media, 2006.

[32] Dimitri Bertsekas, Angelia Nedic, and Asuman Ozdaglar. Convex analysis and optimization,
volume 1. Athena Scientiﬁc, 2003.

[33] Michael Ulbrich. Nonsmooth newton-like methods for variational inequalities and constrained
optimization problems in function spaces. Habilitation, Technical University of Munich, Munich,
2002.

[34] Stefan Scholtes. Introduction to piecewise differentiable equations. Springer, 2012.

[35] Walter Gautschi. Numerical analysis. Springer Science & Business Media, 2011.

[36] Jack Levine and HM Nahikian. On the construction of involutory matrices. The American
Mathematical Monthly, 69(4):267–272, 1962.

12


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: See Section 1 and Section 3.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reﬂect how
much the results can be expected to generalize to other settings.
• It is ﬁne to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justiﬁcation: See Section 6.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-speciﬁcation, asymptotic approximations only holding locally). The authors
should reﬂect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reﬂect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reﬂect on the factors that inﬂuence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efﬁciency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be speciﬁcally instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

13


---Page Break---
Justiﬁcation: See Section 2, Section 3 and Appendix.
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
Justiﬁcation: The contribution is primarily a new statistical framework. See Section 4 and
Section 5.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or veriﬁable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might sufﬁce, or if the contribution is a speciﬁc model and empirical evaluation, it may
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

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

14


---Page Break---
Answer: [No]
Justiﬁcation: This paper does not release new assets.
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
Justiﬁcation: See Section 5.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?
Answer: [Yes]
Justiﬁcation: See Section 5.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.

15


---Page Break---
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not veriﬁed.
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
Justiﬁcation: See Section 5.
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
Justiﬁcation: We have reviewed the NeurIPS Code of Ethics.
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
Justiﬁcation: [NA]
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact speciﬁc
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to

16


---Page Break---
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
feedback over time, improving the efﬁciency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justiﬁcation: [NA]
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety ﬁlters.
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
Justiﬁcation: Licenses are available referring to the provided links.
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

17


---Page Break---
Answer: [NA] .
Justiﬁcation: This paper does not release new assets.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip ﬁle.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justiﬁcation: [NA]
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is ﬁne, but if the main contribu-
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
Justiﬁcation: [NA]
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

18


---Page Break---
