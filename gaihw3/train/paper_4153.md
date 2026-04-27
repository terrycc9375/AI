Optimal Algorithms for Online Convex Optimization
with Adversarial Constraints

Abhishek Sinha, Rahul Vaze
School of Technology and Computer Science
Tata Institute of Fundamental Research
Mumbai 400005, India
abhishek.sinha@tifr.res.in, rahul.vaze@gmail.com

Abstract

A well-studied generalization of the standard online convex optimization (OCO)
framework is constrained online convex optimization (COCO). In COCO, on every
round, a convex cost function and a convex constraint function are revealed to the
learner after it chooses the action for that round. The objective is to design an online
learning policy that simultaneously achieves a small regret while ensuring a small
cumulative constraint violation (CCV) against an adaptive adversary interacting
over a horizon of length T. A long-standing open question in COCO is whether an
online policy can simultaneously achieve O(
√

T) regret and ˜O(
√

T) CCV without
any restrictive assumptions. For the first time, we answer this in the affirmative
and show that a simple first-order policy can simultaneously achieve these bounds.
Furthermore, in the case of strongly convex cost and convex constraint functions,
the regret guarantee can be improved to O(log T) while keeping the CCV bound
the same as above. We establish these results by effectively combining adaptive
OCO policies as a blackbox with Lyapunov optimization - a classic tool from
control theory. Surprisingly, the analysis is short and elegant.

1
Introduction

Online convex optimization (OCO) is a standard framework for modelling and analyzing a broad
family of online decision problems under uncertainty. In the OCO problem, on every round t, an
online policy first selects an action xt from a closed and convex admissible set (a.k.a. decision set)
X. Then the adversary reveals a convex cost function ft, resulting in a cost of ft(xt). The goal of an
online policy is to choose an admissible action sequence {xt}T
t=1 so that its cumulative cost is not
much larger than that of any fixed admissible action chosen in hindsight. In particular, the objective
is to minimize the static regret defined below

RegretT ≡sup
{ft}T
t=1
sup
x⋆∈X
RegretT (x⋆), where RegretT (x⋆) ≡
T
∑
t=1
ft(xt) −
T
∑
t=1
ft(x⋆).
(1)

The term static refers to using a fixed benchmark, specifically only one action x⋆throughout the
horizon of length T.

In this paper, we consider a generalization of the standard OCO framework. In this problem, on
every round t, the online policy first chooses an admissible action xt ∈X, and then the adversary
chooses a convex cost function ft ∶X →R and k constraints of the form gt,i(x) ≤0, i ∈[k], where
gt,i ∶X →R is a convex function for each i ∈[k]1. Since gt,i’s are revealed after the action xt is

1Notations: For any natural number n, we define [n] ≡{1, 2, . . . , n}. For any real number z, we define
(z)+ ≡max(0, z).

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
chosen, an online policy need not necessarily take feasible actions on each round, and the obvious
metric of interest in addition to (1) is the total cumulative constraint violation (CCV) V(T) defined as

CCVT ≡V(T) =
k
max
i=1 Vi(T)
where
Vi(T) =
T
∑
t=1
(gt,i(xt))+.
(2)

Let X ⋆be the feasible set consisting of all admissible actions that satisfy all constraints gt,i(x) ≤
0, i ∈[k],t ∈[T]. Under the standard assumption that X ⋆is not empty, the goal is to design an
online policy to simultaneously achieve a small regret (1) with respect to any admissible benchmark
x⋆∈X ⋆and a small CCV (2). We refer to this problem as the constrained OCO (COCO). The
assumption X ⋆≠∅will be relaxed in Section 3 for the Online Constraint Satisfaction (OCS) problem
where the cost functions are set to zero, and the objective is to minimize just the CCV.

COCO arises in many applications, including online portfolio optimization with risk constraints,
resource allocation in cloud computing with time-varying demands, pay-per-click online ad markets
with budget constraints [Liakopoulos et al., 2019], online recommendation systems, dynamic pricing,
revenue management, robotics and path planning problems, and multi-armed bandits with fairness
constraints [Sinha, 2024a]. The necessity for revealing the constraints sequentially may also arise,
e.g., in communication-limited settings, where it might be infeasible to reveal all constraints defining
the feasible set at a time (e.g., combinatorial auctions). See Section 4 for an application of the COCO
framework in fraud detection which involves binary classification with a highly-imbalanced dataset.

1.1
Related Work

Unconstrained OCO:
In a seminal paper, Zinkevich [2003] showed that for solving (1), the
ubiquitous projected online gradient descent (OGD) policy achieves an O(
√

T) regret for convex cost
functions with uniformly bounded sub-gradients. A number of follow-up papers proposed adaptive
and parameter-free versions of OGD [Hazan et al., 2007, Orabona and Pál, 2018]. See Orabona
[2019], Hazan [2022] for textbook treatments of the OCO framework and associated algorithms.

Constrained OCO (COCO): (A) Time-invariant constraints: A number of papers considered
COCO with time-invariant constraints, i.e., gt,i = gi,∀t [Yuan and Lamperski, 2018, Jenatton et al.,
2016, Mahdavi et al., 2012, Yi et al., 2021]. These works assume that the functions gi’s are known to
the policy a priori. However, they allowed the policy to remain infeasible on any round to avoid the
costly projection step of the vanilla projected OGD policy. Their main objective was to design an
efficient policy (avoiding the explicit projection step) with a small regret and CCV.

(B) Time-varying constraints: Solving the COCO problem when the constraint functions, i.e., gt,i’s,
change arbitrarily with time t is more challenging. In this case, except for Neely and Yu [2017] and
Liakopoulos et al. [2019], most of the prior works construct some Lagrangian function and then
update the primal and dual variables [Yu et al., 2017, Sun et al., 2017, Yi et al., 2023]. However, the
performance bounds obtained with this approach remain suboptimal. Both Neely and Yu [2017] and
Liakopoulos et al. [2019] use the drift-plus-penalty (DPP) framework introduced by Neely [2010]
to solve the constrained problem under various assumptions. In particular, Neely and Yu [2017]
proposed a DPP-based policy for COCO upon assuming the Slater’s condition, i.e., gt,i(x⋆) < −η,
for some η > 0 ∀i,t. Clearly, this condition precludes the important case of non-negative constraint
functions (e.g., constraint functions of the form max(0,gt(x))). Furthermore, the bounds obtained
upon assuming Slater’s condition depend inversely with the Slater’s constant η (usually hidden under
the big-Oh notation). Since η could be arbitrarily small, these bounds could be arbitrarily loose.
Liakopoulos et al. [2019] extended Neely and Yu [2017]’s result by considering a weaker form of the
feasibility assumption without assuming Slater’s condition. Furthermore, although these DPP-based
results are interesting, they have not been able to provide improved regret or CVV bounds when the
cost functions ft’s are strongly convex because of the linearization step inherent in this approach.

In a recent paper, Guo et al. [2022] considered COCO and obtained the best-known prior results
without assuming Slater’s condition. However, in addition to yielding sub-optimal bounds, their
policy is quite computationally intensive since it requires solving a convex optimization problem on
each round. Compared to this, all policies proposed in this paper take only a single gradient-descent
step and perform only one Euclidean projection on each round. Please refer to Table 1 for a brief
summary of the results and Section A.5 in the Appendix for a qualitative comparison. The COCO
problem has been considered in the dynamic setting as well [Chen and Giannakis, 2018, Cao and
Liu, 2018, Vaze, 2022, Liu et al., 2022] where the benchmark x⋆in (1) is replaced by x⋆
t that is also

2


---Page Break---
Reference
Regret
CCV
Complexity per round
Assumptions

Mahdavi et al. [2012]
O(
√

T)
O(T
3/4)
Projection
Time-invariant constraints
Jenatton et al. [2016]
O(T max(β,1−β))
O(T 1−β/2)
Projection
Time-invariant constraints
Sun et al. [2017]
O(
√

T)
O(T
3/4)
Bregman Projection
-
Neely and Yu [2017]
O(
√

T)
O(
√

T)
Conv-OPT
Slater condition
Yuan and Lamperski [2018]
O(T max(β,1−β))
O(T 1−β/2)
Projection
Time-invariant constraints
Yu and Neely [2020]
O(
√

T)
O(1)
Conv-OPT
Slater & Time-invariant constraints
Yi et al. [2021]
O(T max(β,1−β))
O(T (1−β)/2)
Conv-OPT
Time-invariant constraints
Yi et al. [2022]
O(T β)
O(T 1−β/2)
Projection
Strongly convex cost
Guo et al. [2022]
O(
√

T)
O(T
3/4)
Conv-OPT
-
Guo et al. [2022]
O(log T)
O(√T log T)
Conv-OPT
Strongly convex cost
Yi et al. [2023]
O(T max(β,1−β))
O(T 1−β/2)
Conv-OPT
-
Yi et al. [2023]
O(log(T))
O(√T log T)
Conv-OPT
Strongly convex cost
This paper
O(
√

T)
O(
√

T log T)
Projection
-
This paper
O(log T)
O(√T log T)
Projection
Strongly convex cost
This paper
O(log T)
O( log T

α )
Projection
Strongly convex cost, RegretT ≥0,

Table 1: Summary of the results on COCO. Unless stated otherwise, we assume arbitrary time-varying convex
constraints and convex cost functions. In the above table, 0 ≤β ≤1 is an adjustable parameter, α is the strong
convexity parameter of the strongly convex cost functions. Conv-OPT refers to solving a constrained convex
optimization problem on each round. Projection refers to the Euclidean projection operation on the convex set
X. For typical convex sets (e.g., Euclidean box, probability simplex), projection operations are substantially
more efficient than solving a constrained convex optimization problem.

allowed to change its actions over time. However, we focus our attention on achieving the optimal
performance bounds for the static version. A special case of COCO is the ONLINE CONSTRAINT
SATISFACTION (OCS) problem that does not involve any cost function, i.e., ft = 0, ∀t, and the only
object of interest is the CCV. The OCS problem becomes especially interesting in the setting where
the feasible set may be empty.

1.2
Our Contributions

In this paper, we consider both COCO and OCS problems and make the following contributions.

1. We propose an efficient first-order policy that simultaneously achieves O(
√

T) regret and
O(
√

T log T) CCV for the COCO problem. Our result breaks the long-standing O(T
3/4)
barrier for the CCV and matches the lower bound (derived in Theorem 3, previously missing
from the literature) up to a logarithmic term. For strongly convex cost functions, the regret
guarantee is improved to O(log T) while keeping the CCV bound the same as above. Under
an additional assumption that the regret is non-negative, we obtain a further improved
logarithmic CCV bound in the strongly convex setting (see Table 1).
2. We additionally consider a special case of the COCO problem, called Online Constraint
Satisfaction (OCS), under relaxed feasibility assumptions and obtain sub-linear CCV bounds.
3. On the algorithmic side, our policy simply runs an adaptive first-order OCO algorithm as
a blackbox on a specially constructed convex surrogate cost function sequence. On every
round, the policy needs to compute only two gradients and an Euclidean projection. This is
way more efficient compared to the policies proposed in the previous works [Guo et al., 2022,
Neely and Yu, 2017], which need to solve expensive convex optimization problems on each
round while yielding sub-optimal bounds. Furthermore, in the special case of time-invariant
constraints, our results yield an efficient first-order OCO policy with competitive regret and
CCV bounds [Mahdavi et al., 2012, Jenatton et al., 2016, Yi et al., 2021].
4. Our results are obtained by introducing a crisp and elegant potential function-based algorith-
mic technique for simultaneously controlling the regret and the CCV. In brief, the regret and
CCV bounds are derived from a single inequality that arises from plugging in off-the-shelf
adaptive regret bounds in a new regret decomposition result (Eqn. (6)). This new analytical
technique might also be of independent interest.

3


---Page Break---
5. Finally, in Section 4, we evaluate the practical performance of our algorithm in the online
credit card fraud detection problem with a highly imbalanced dataset.

2
The Constrained OCO (COCO) Problem

2.1
Assumptions

We now state the assumptions considered in this paper. These assumptions are standard in literature
on the COCO problem [Guo et al., 2022, Yi et al., 2021, Neely and Yu, 2017].
Assumption 1 (Convexity). The cost function ft ∶X ↦R and the constraint function gt,i ∶X ↦R
are convex for all t ≥1,i ∈[k]. The admissible set (a.k.a. the decision set or the action set) X ⊆Rd
is closed and convex and has a finite Euclidean diameter D.
Assumption 2 (Lipschitzness). All cost functions {ft}t≥1 and the constraint functions {gt,i}i∈[k],t≥1’s
are G-Lipschitz. In other words, for any x,y ∈X, we have
∣ft(x) −ft(y)∣≤G∣∣x −y∣∣, ∣gt,i(x) −gt,i(y)∣≤G∣∣x −y∣∣, ∀t ≥1,i ∈[k].

Unless specified otherwise, the norm ∣∣⋅∣∣will refer to the standard Euclidean norm and ∇f will
refer to an arbitrary subgradient of a convex function f. Assumption 2 implies that the ℓ2-norm
of the (sub)gradients of the cost and constraint functions are uniformly upper-bounded by G over
the admissible set X. Finally, we make the following feasibility assumption about the constraint
functions.
Assumption 3 (Feasibility). There exists a feasible action x⋆∈X s.t. gt,i(x⋆) ≤0,∀t,i. The feasible
set X ⋆is defined to be the set of all feasible actions. The feasibility assumption implies that X ⋆≠∅.

The feasibility assumption distinguishes the cost functions from the constraint functions and is
commonly assumed in the literature [Guo et al., 2022, Neely and Yu, 2017, Yu and Neely, 2016,
Yuan and Lamperski, 2018, Yi et al., 2023, Liakopoulos et al., 2019]. In Section 3, we will consider
a constraint-only variant of the problem where the feasibility assumption (Assumption 3) will be
relaxed. See Appendix A.1 for a brief discussion on the assumptions.

Remarks:
On each round, multiple constraints of the form gt,i(x) ≤0,i ∈[k] can be replaced
by a single new constraint gt(x) ≤0 where the constraint function gt is defined to be the pointwise
maximum of the given constraints, i.e., gt(x) ≡maxk
i=1 gt,i(x),x ∈X. It is easy to verify that if
each of the constraint functions {gt,i}k
i=1 satisfies the above assumptions, then the constraint function
gt defined above also satisfies the assumptions. Hence, throughout this section and without loss of
generality, we will assume that only one constraint function is revealed on each round. That being
said, under the relaxed feasibility assumption in Section 3, this trick does not work and there we will
need to consider the full set of k constraint functions.

2.2
Online Policy for COCO

Recall that compared to the standard OCO problem where the only objective is to minimize the Regret
[Hazan, 2022], in COCO, our objective is twofold: to simultaneously control the Regret and the CCV.
See Section A.2 in the Appendix for preliminaries on the OCO problem and some standard results
which will be useful in our analysis. In the following, we propose a Lyapunov function-based policy
that yields the optimal Regret and CCV bounds for the COCO problem. Although for simplicity, we
assume that the horizon length T is known, we can use the standard doubling trick for an unknown T.

2.3
Design and Analysis of the Algorithm

To simplify the analysis, we pre-process the cost and constraint functions on each round as follows.

Pre-processing: On every round, we first clip the negative part of the constraint function to zero by
passing it through the standard ReLU unit. Then, we scale both the cost and constraint functions by a
positive factor β, which will be determined later. In other words, we work with the pre-processed
inputs ˜ft ←βft, ˜gt ←β(gt)+. Hence, the pre-processed functions are βG-Lipschitz and ˜gt ≥0,∀t.

In the following, we derive the Regret and CCV bounds for the pre-processed functions. The bounds
for the original problem are obtained upon scaling the results back by β−1 in the final step.

4


---Page Break---
Algorithm 1 Online Policy for COCO

1: Input: Sequence of convex cost functions {ft}T
t=1 and constraint functions {gt}T
t=1, G = a
common Lipschitz constant, T = Horizon length, D = Euclidean diameter of the admissible set
X, PX (⋅) = Euclidean projection operator on the set X
2: Parameter settings:

1. Convex cost functions: β = (2GD)−1,V = 1,λ =
1
2
√

T ,Φ(x) = exp(λx) −1.

2. α-strongly convex cost functions: β = 1,V = 8G2 ln(T e)

α
,Φ(x) = x2.

3: Initialization: Set x1 ∈X arbitrarily, Q(0) = 0.
4: for each t = 1 ∶T do
5:
Play xt, observe ft,gt, incur a cost of ft(xt) and constraint violation of (gt(xt))+

6:
˜ft ←βft, ˜gt ←β max(0,gt).
7:
Q(t) = Q(t −1) + ˜gt(xt).
8:
Compute (sub)gradient ∇t = ∇ˆft(xt), where the surrogate function ˆft is defined in Eqn. (5)
9:
xt+1 = PX (xt −ηt∇t), where

ηt =

⎧⎪⎪⎪⎨⎪⎪⎪⎩

√

2D
2
√

∑t
τ=1 ∣∣∇τ ∣∣2
2
,
for convex costs (AdaGrad stepsizes)

1
∑t
s=1 Hs ,
for strongly convex costs (Hs= strong convexity parameter of fs,s ≥1)

10: end for each

2.3.1
Defining the Surrogate Cost Functions

Let Q(t) denote the CCV for the pre-processed constraints up to round t. Clearly, Q(t) satisfies the
simple recursion Q(t) = Q(t −1) + ˜gt(xt),t ≥1, with Q(0) = 0. Recall that one of our objectives
is to make Q(t) small. Towards this, let Φ ∶R+ ↦R+ be any non-decreasing differentiable convex
potential (Lyapunov) function such that Φ(0) = 0. Using the convexity of Φ(⋅), we have

Φ(Q(t))
≤
Φ(Q(t −1)) + Φ′(Q(t))(Q(t) −Q(t −1))
=
Φ(Q(t −1)) + Φ′(Q(t))˜gt(xt).
(3)

Hence, the change (drift) of the potential function Φ(Q(t)) on round t can be upper bounded as

Φ(Q(t)) −Φ(Q(t −1)) ≤Φ′(Q(t))˜gt(xt).
(4)

Recall that, in addition to controlling the CCV, we also want to minimize the cumulative cost
∑T
t=1 ft(xt) (which is equivalent to the regret minimization). Inspired by the stochastic drift-plus-
penalty framework of Neely [2010], we combine these two objectives to a single objective of
minimizing a sequence of surrogate cost functions { ˆft}T
t=1 which are obtained by taking a positive
linear combination of the drift upper bound (4) and the cost function. More precisely, we define

ˆft(x) ∶= V ˜ft(x) + Φ′(Q(t))˜gt(x), t ≥1.
(5)

In the above, V is a suitably chosen non-negative parameter to be determined later. In brief, the
proposed policy for COCO, described in Algorithm 1, simply runs an adaptive OCO policy on the
surrogate cost function sequence { ˆft}t≥1, with a specific choice of the potential function Φ(⋅), the
parameter V , and step-size sequence {ηt}t≥1, as dictated by the following analysis.

2.3.2
The Regret Decomposition Inequality

Let x⋆∈X ⋆be any feasible action guaranteed by Assumption (3). Plugging in the definition of
surrogate costs (5) into the drift inequality (4), and using the fact that gτ(x⋆) ≤0,∀τ ≥1, we have

Φ(Q(τ)) −Φ(Q(τ −1)) + V ( ˜fτ(xτ) −˜fτ(x⋆)) ≤ˆfτ(xτ) −ˆfτ(x⋆), ∀τ ≥1.

Summing the above inequalities for rounds 1 ≤τ ≤t, and using the fact that Φ(0) = 0, we obtain

Φ(Q(t)) + V Regrett(x⋆) ≤Regret′
t(x⋆), ∀x⋆∈X ⋆,
(6)

where Regrett on the LHS and Regret′
t on the RHS of (6) refer to the regret for learning the pre-
processed cost functions { ˜ft}t≥1 and the surrogate cost functions { ˆft}t≥1 respectively. We will use

5


---Page Break---
the following upper bound on the ℓ2-norm of the (sub)gradients Gt of the surrogate cost function ˆft
defined in Eqn. (5):

Gt ≡∣∣∇ˆft(xt)∣∣
(a)
≤V ∣∣∇˜ft(xt)∣∣+ Φ′(Q(t))∣∣∇˜gt(xt)∣∣
(b)
≤βG(V + Φ′(Q(t)),
(7)
where in (a), we have used the triangle inequality for ℓ2 norms and in (b), we have used the fact that
all pre-processed functions are βG-Lipschitz.

2.3.3
Convex Cost and Convex Constraint Functions

We now apply the regret decomposition inequality (6) to the case of convex cost and convex constraint
functions. Let us choose the regret-minimizing OCO subroutine for the surrogate cost functions to be
the OGD policy with adaptive step sizes (a.k.a. AdaGrad) described in part 1 of Theorem 6 in the
Appendix (see Algorithm 1). Plugging in the adaptive regret bound (24) on the RHS of (6), setting
β = (2GD)−1, and using Eqn. (7), we arrive at the following inequality valid for any t ≥1 ∶

Φ(Q(t)) + V Regrett(x⋆) ≤

¿
Á
Á
À

t
∑
τ=1
(Φ′(Q(τ)))
2 + V
√

t.
(8)

In deriving the above result, we have utilized simple algebraic inequalities (x + y)2 ≤2(x2 + y2)
and
√

a + b ≤√a +
√

b,a,b ≥0. Now recall that the sequence {Q(t)}t≥1 is non-negative and non-
decreasing as ˜gt ≥0. Furthermore, the derivative Φ′(⋅) is non-decreasing as the function Φ(⋅) is
assumed to be convex. Hence, bounding all terms in the summation on the RHS of (8) from above by
the last term, we arrive at the following inequality for any feasible x⋆∈X ⋆∶

Φ(Q(t)) + V Regrett(x⋆) ≤Φ′(Q(t))
√

t + V
√

t.
(9)
The simplified regret decomposition inequality (9) constitutes the key step for the subsequent analysis.

∎Performance Analysis

An exponential Lyapunov function:
We now derive the Regret and CCV bounds for the proposed
policy (Algorithm 1) by choosing Φ(⋅) to be the exponential Lyapunov function: Φ(x) ≡exp(λx)−1,
where the parameter λ ≥0 will be fixed later. Clearly, the function Φ(⋅) satisfies the required
conditions for a Lyapunov function - it is a non-decreasing and convex function with Φ(0) = 0.

Bounding the Regret:
With the above choice for the Lyapunov function Φ(⋅), Eqn. (9) implies
that for any feasible x⋆∈X ⋆and for any t ∈[T], we have

exp(λQ(t)) −1 + V Regrett(x⋆) ≤λexp(λQ(t))
√

t + V
√

t.
Transposing the first term on the above inequality to the RHS and dividing throughout by V , we have:

Regrett(x⋆) ≤
√

t + 1

V + exp(λQ(t))

V
(λ
√

t −1).
(10)

Choosing any λ ≤
1
√

T , the last term in the above inequality becomes non-positive for any t ∈[T].
Hence, for any x⋆∈X ⋆, we have the following regret bound

Regrett(x⋆) ≤
√

t + 1

V . ∀t ∈[T].
(11)

Bounding the CCV:
Since all pre-processed cost functions are βG = (2D)−1-Lipschitz, we
trivially have Regrett(x⋆) = ∑t
s=1( ˜fs(xs) −˜fs(x⋆)) ≥−Dt

2D ≥−t

2. Hence, from Eqn. (10), we have
that for any λ <
1
√

T and any t ∈[T] ∶

exp(λQ(t))

V
(1 −λ
√

t) ≤2t + 1

V
Ô⇒Q(t) ≤1

λ ln 1 + 2V t

1 −λ
√

t
.
(12)

Choosing λ =
1
2
√

T ,V = 1, and scaling the bounds back by β−1 ≡2GD, we arrive at our main result.

Theorem 1. For the COCO problem with adversarially chosen G-Lipschitz cost and constraint
functions, Algorithm 1, with β = (2GD)−1,V = 1,Φ(x) = exp(
x
2
√

T )−1, yields the following Regret
and CCV bounds for any horizon length T ≥1 ∶

Regrett ≤2GD(
√

t + 1), ∀t ∈[T], CCVT ≤4GD ln(2(1 + 2T))
√

T.
In the above, D denotes the Euclidean diameter of the closed and convex admissible set X.

6


---Page Break---
2.3.4
Strongly Convex Cost and Convex Constraint Functions

We now consider the setting where each of the cost functions ft,t ≥1, is α-strongly convex for
some α > 0. The constraint functions gt’s are assumed to be convex as before and not necessarily
strongly convex. In this case, we choose the regret-minimizing OCO subroutine for the surrogate
cost functions to be the OGD algorithm with the step-size sequence as given in part 2 of Theorem
6 in the Appendix (see Algorithm 1). Since the cost functions are known to be α-strongly convex,
each of the surrogate cost functions (5) is V α-strongly convex. Hence, using the bound from Eqn.
(7), choosing the scaling parameter to be β = 1, and simplifying the generic regret bound given by
Eqn. (25), we obtain the following regret bound for learning the surrogate cost functions { ˆfs}s≥1:

Regret′
t(x⋆) ≤V G2

α
(1 + ln(t)) + G2

αV

t
∑
τ=1

(Φ′(Q(τ)))2

τ
, x⋆∈X.
(13)

In the above, we have used the standard bound for the Harmonic sum: ∑t
τ=1
1
τ ≤1 + ln(t), as well
as the fact that (a + b)2 ≤2(a2 + b2). Substituting the bound (13) into the regret decomposition
inequality (6), and using the non-decreasing property of the sequence {Q(τ)}τ≥1 and the derivative
Φ′(⋅), we obtain

Φ(Q(t)) + V Regrett(x⋆) ≤V G2

α
(1 + ln(t)) + G2

αV (1 + ln(t))(Φ′(Q(t)))
2, ∀x⋆∈X ⋆,∀t. (14)

Finally, choosing Φ(⋅) as the quadratic Lyapunov function, i.e., Φ(x) ≡x2, we arrive at the following
result for strongly convex cost and convex constraint functions.

Theorem 2. For the COCO problem with adversarially chosen α-strongly convex, G-Lipschitz
cost functions and G-Lipschitz convex constraint functions, Algorithm 1, with β = 1,V
=
8G2 ln(T e)

α
,Φ(x) = x2, yields the following Regret and CCV bounds for any horizon length T ≥1 ∶

Regrett(x⋆) ≤G2

α (1 + ln(t)), CCVt = O(

√

tlog T

α
),∀x⋆∈X ⋆, ∀t ∈[T].

Furthermore, if the worst-case regret is non-negative in some round t (i.e., supx⋆∈X ⋆Regrett(x⋆) ≥0),
then the CCV can be further improved to CCVT = O( log T

α ) while keeping the regret bound the same.

Please refer to Appendix A.6 for the proof of Theorem 2.

Remarks: The second part of the theorem is surprising because it says that when the regret is
non-negative, a stronger logarithmic CCV bound holds for not necessarily strongly convex constraints.
In Appendix A.7, we give example of an interesting class of adversaries, called convex adversary, for
which the non-negative regret assumption holds true in the OCO setting.

2.4
Lower Bounds

We now show that under Assumptions 1, 2, and 3, the regret and the CCV of any online policy
for the COCO problem for T rounds are both lower bounded by Ω(
√

T) provided the problem is
high-dimensional. Recall that if the constraint function gt = 0,∀t, then the COCO problem reduces
to the standard OCO problem, and Ω(
√

T) is a well-known regret lower bound for OCO [Hazan,
2022, Theorem 10]. In this case, we trivially have CCV = 0. The main challenge in proving a lower
bound for COCO is simultaneously bounding both the regret and CCV. Prior work does not give any
simultaneous lower bounds since the standard adversarial inputs used to derive the lower bound of
Hazan [2022] do not satisfy the feasibility assumption (Assumption 3). We derive the lower bound by
constructing a sequence of cost and constraint functions that satisfy Assumption 3 in a d-dimensional
Euclidean box of unit diameter.

Theorem 3. Under Assumptions 1, 2, and 3, for any choice of the horizon length T and online policy,
there exists a problem instance with dimension d ≥T where min(RegretT ,CCVT ) = Ω(
√

T).

In high-dimensional problems where d ≫T, the above lower bound matches with the upper bound
given in Theorem 1. The proof of Theorem 3 can be found in Appendix A.4.

7


---Page Break---
3
The Online Constraint Satisfaction Problem (OCS)

In this section, we study a special case of the COCO problem, which involves only constraint
functions and no cost functions. The OCS problem arises in many practical settings, including the
multi-task learning problem (see Section A.3 in the Appendix for a brief discussion). In Section
A.8 in the Appendix, we also establish a connection between the OCS problem and the well-studied
Convex Body Chasing problem [Argue et al., 2019]. The setup is similar to the COCO setting –
on every round t ≥1, an online policy selects an action xt from a closed, bounded, and convex
admissible set X ⊆Rd. After observing the current action xt, the adversary chooses k constraints of
the form gt,i(x) ≤0,i ∈[k], where each gt,i ∶X ↦R is a convex function. Let I be any sub-interval
of the horizon [1,T]. The cumulative constraint violation (CCV) V(T) for the OCS problem is
defined as the maximum signed cumulative constraint violation in any sub-interval:

V(T) =
k
max
i=1 Vi(T), where Vi(T) = max
I⊆[1,T ] ∑
t∈I
gt,i(xt), 1 ≤i ≤k.
(15)

The objective is to design an online learning policy so that V(T) is as small as possible. It is worth
noting that in the OCS problem, we consider a soft constraint violation metric maxI ∑t∈I gt,i(xt)
instead of the hard violation metric ∑T
t=1(gt,i(xt))+ as in COCO. This allows for compensating
the infeasibility on one round with strict feasibility on other rounds. In contrast with the COCO
setting, without Assumption 3, running a no-regret policy on the pointwise maximum of the constraint
functions no longer works as the CCV of any fixed benchmark could grow linearly with T. In the
OCS problem, we relax the feasibility assumption (Assumption 3), and consider the following two
distinct alternatives instead.

1. S-feasibility:
Here, we assume that there is an admissible action x⋆∈X that satisfies the
aggregate constraints over any interval of S rounds. However, unlike Liakopoulos et al. [2019], which
also considers the same assumption, the value of the parameter S is not necessarily known to the
policy a priori. Towards this end, we define the set of all S-feasible actions XS as below:
XS = {x⋆∈X ∶∑
τ∈I
gτ,i(x⋆) ≤0,∀sub-intervals I ⊆[1,T], ∣I∣= S,∀i ∈[k]}.
(16)

We now replace Assumption 3 with the following weaker version:
Assumption 4 (S-feasibility). XS ≠∅for some 1 ≤S ≤T.

Clearly, Assumption 4 is weaker than Assumption 3 as X ⋆⊆XS,∀S ≥1. Note that even when
the individual constraint functions satisfy S-feasibility, their pointwise maximum need not satisfy
S-feasibility. Hence, unlike COCO under Assumption 3, this problem cannot be solved by simply
running a no-regret policy on the pointwise maximum of the constraints.

2. PT -constrained adversary
In this case, we drop any feasibility assumption altogether. As a
consequence, any static admissible benchmark x⋆∈X also incurs a CCV.
Definition 1. An adversary is called PT -constrained if its minimum static CCV is PT F, i.e.,
1
F minx⋆∈X maxI⊆[T ],i ∑t∈I gt,i(x⋆) = PT , where F is a normalizing factor denoting the maxi-
mum absolute value of the constraint functions within the compact admissible set X.

As before, the value of PT is not necessarily known to the policy a priori.

3.1
Designing an OCS Policy with a Quadratic Lyapunov function

We define a process Q(t) = (Qi(t),i ∈[k]),t ≥1, which tracks the CCV:

Qi(t) = (Qi(t −1) + gt,i(xt))
+, Qi(0) = 0, t ≥1, ∀i ∈[k].
(17)
Notably, in contrast to COCO, we do not clip the constraint functions in the above recursion.
Expanding Eqn. (17), which is also known as the queueing recursion or the Lindley process [Asmussen,
2003, pp. 92], and using the definition in Eqn. (15), we have the following relation for all i ∈[k] ∶

Vi(T) ≡
T
max
t=1 max(0,
t−1
max
τ=0

t
∑
s=t−τ
gs,i(xl)) =
T
max
t=1 Qi(t).
(18)

Equation (18) indicates that to control the CCV (15), it is sufficient to control the Q(t) process.
Similar to the COCO problem, we combine the classic Lyapunov method with adaptive no-regret
OCO policies to control the Q(t) process.

8


---Page Break---
A Quadratic Lyapunov function:
We consider the quadratic potential function Φ(Q(t)) ≡
∑k
i=1 Q2
i (t),t ≥1. Since ((x)+)2 = xx+,∀x ∈R, from Eqn. (17), we have

Q2
i (t)
=
(Qi(t −1) + gt,i(xt))Qi(t) = Qi(t −1)Qi(t) + Qi(t)gt,i(xt),

(a)
≤
1
2Q2
i (t) + 1

2Q2
i (t −1) + Qi(t)gt,i(xt), ∀i ∈[k].
(19)

where in (a), we have used the AM-GM inequality. Rearranging Eqn. (19), the change of the potential
function Φ(Q(t)) on round t can be upper bounded as follows

Φ(Q(t)) −Φ(Q(t −1)) =
k
∑
i=1
(Q2
i (t) −Q2
i (t −1)) ≤2
k
∑
i=1
Qi(t)gt,i(xt).
(20)

Similar to (5), we now define a surrogate cost function ˆft ∶X ↦R as a linear combination of the
constraint functions with the coefficients given by the vector Q(t), i.e.,

ˆft(x) ≡2
k
∑
i=1
Qi(t)gt,i(x).
(21)

Clearly, the surrogate cost function ˆft(⋅) is convex since the coefficients Qi(t)’s are non-negative
and the constraint functions are convex. Our OCS policy, described below, simply runs a regret-
minimizing adaptive OCO subroutine on the surrogate cost function sequence (21).

The OCS policy (Algorithm 2): Pass the surrogate cost functions { ˆft}t≥1 to the AdaGrad algorithm
which enjoys a data-dependent regret as given in part 1 of Theorem 6 in the Appendix (Eqn. (24)).

Algorithm 2 Online Policy for OCS

1: Input: Sequence of convex constraint functions {gt,i}i∈[k],t≥1, a closed and convex admissible
set X with a finite Euclidean diameter D, PX (⋅) = Euclidean projection operator on the set X
2: Output: Sequence of admissible actions {xt}t≥1
3: Initialization: Set x1 ∈X arbitrarily, Qi(0) = 0, ∀i ∈[k].
4: for each each round t ≥1 do
5:
Play xt, observe the constraint functions {gt,i}i∈[k] revealed by the adversary.
6:
[Update Q(t)] Qi(t) = (Qi(t −1) + gt,i(xt))+,i ∈[k].
7:
[Compute a subgradient] ∇t ≡∇ˆft(xt) = 2∑k
i=1 Qi(t)∇gt,i(xt).

8:
[AdaGrad step] Compute the next action xt+1 = PX (xt −ηt∇t), where ηt =
√

2D
2
√

∑t
τ=1 ∣∣∇τ ∣∣2
2
.

9: end for each

3.2
Performance Bounds

Theorem 4. Under Assumptions 1, 2, and 4, Algorithm 2 achieves the following CCV bound for the
OCS problem: V(T) = O(max(
√

ST,S)).
Theorem 5. Under Assumptions 1 and 2, Algorithm 2 achieves the following CCV bound for the
OCS problem for any PT -constrained adversary as given in Definition 1:

V(T) = O(P
1/3
T T
2/3) + O(
√

T).

Trivially, we have S ≤T and PT ≤T. In the non-trivial case where either S or PT increases
sub-linearly with the horizon length T, the above theorems yield sublinear CCV bounds.

4
Experiments: Credit Card Fraud Detection

Classification with a highly imbalanced dataset:
We first formulate the credit card fraud detection
problem in the COCO framework. Assume that we receive a sequence of d-dimensional feature
vectors {zt}t≥1 and the corresponding binary labels {yt}t≥1 for a sequence of credit card transactions,
where each transaction can either be legitimate (label = 0) or fraudulent (label = 1). The problem

9


---Page Break---
0.2
0.4
0.6
0.8
1.0
False Positive Rate

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

1.00

True Positive Rate

Area under the ROC curve= 0.92

ROC curve

Figure 1: ROC curve obtained by varying λ

0
50000
100000
150000
200000
250000
Rounds

0

50

100

150

200

250

300

350

CCV

Variation of CCV

Figure 2: Typical variation of the CCV with time

is to predict the label ˆyt for each transaction zt before its true label yt ∈{0,1} is revealed. Typically,
legitimate transactions outnumber fraudulent transactions by orders of magnitude. Since the goal
is to detect any fraudulent transactions (even at the cost of a few false alarms), maximizing the
classification accuracy alone is insufficient due to the significant class imbalance. We propose the
following reformulation for this problem within the COCO framework.
Formulation:
Let ˆyt(zt,x) be the likelihood of class 1 for the feature zt, given by a parameterized
model with parameter x. Hence, the log-likelihood L(t) of the data on round t can be expressed as:
L(t) = yt log(ˆyt(zt,x)) + (1 −yt)log(1 −ˆyt(zt,x)).
We train the model by maximizing the sum of log-likelihoods for legitimate transactions, subject to
the constraint that all fraudulent transactions have a likelihood value close to 1 (i.e., the sum of the
log-likelihoods of the fraudulent transactions remains close to zero):

max
x

T
∑
t=1
(1 −yt)log(1 −ˆyt(zt,x)), s.t.
T
∑
t=1
yt log(ˆyt(zt,x)) ≥0.
(22)

The above problem (22) can be immediately recognized to be an instance of COCO with the following
cost and constraint functions:
ft(x) ≡−(1 −yt)log(1 −ˆyt(zt,x)), gt(x) ≡−yt log(ˆyt(zt,x)), t ≥1.
In our experiments, we consider the common scenario in which the likelihoods are modeled by the
output of a feedforward neural network. Note that the feasibility assumption (Assumption 3) is
naturally satisfied as the overparameterized neural network models are known to perfectly fit the data
[Belkin et al., 2019]. However, in this case, the functions ft and gt are generally non-convex.
Experiments:
We experiment with a publicly available credit card transaction dataset [Dal Pozzolo
et al., 2014]. This highly imbalanced dataset contains only 492 frauds (∼0.17%) out of 284,807
reported transactions. Each data point has Din = 30 features and binary labels. We choose a simple
network architecture with a single hidden layer containing H = 10 hidden nodes and sigmoid non-
linearities. Unlike previous algorithms, our algorithm is especially suitable for training neural network
models as it only needs to compute the gradients (via backward pass) and evaluate the functions (via
forward pass). Initially, all weights are independently sampled from a standard normal distribution.
The network is then trained using Algorithm 1 on a quad-core CPU with 8 GB RAM. The projection
operation corresponds to L2-normalization. The code has been publicly released [Sinha, 2024b].
Results:
Given the severe class imbalance, the area under the ROC curve, which plots the True
Positive Rate (TPR) against the False Positive Rate (FPR), is an appropriate metric to evaluate any
prediction algorithm for this problem. By varying the hyperparameter λ, we obtain the ROC curve
shown in Figure 1. The area under the ROC curve is computed to be ≈0.92, which is an excellent
score (cf. ideal score = 1.0), notwithstanding the fact that, unlike the standard resampling-based
techniques, the algorithm learns in an entirely online fashion starting from random initialization.
Figure 2 illustrates the expected sublinear variation of CCV during one of the algorithm runs.
5
Conclusion
In this paper, we proposed efficient online policies for the COCO problem with optimal performance
bounds. We also derived sublinear CCV bounds for the OCS problem under a set of relaxed
assumptions. Our analysis is streamlined, leveraging Lyapunov theory and adaptive regret bounds for
the standard OCO problem. In the future, exploring dynamic regret bounds and a bandit extension of
the COCO problem would be interesting.

10


---Page Break---
6
Acknowledgement

This work was supported by the Department of Atomic Energy, Government of India, under project no.
RTI4001 and by a Google India faculty research award. The first author was also partially supported
by a US-India NSF-DST collaborative grant coordinated by IDEAS-Technology Innovation Hub
(TIH) at the Indian Statistical Institute, Kolkata. The authors gratefully acknowledge comments from
the anonymous reviewers, which substantially improved the quality of the presentation.

References

Nikolaos Liakopoulos, Apostolos Destounis, Georgios Paschos, Thrasyvoulos Spyropoulos, and
Panayotis Mertikopoulos. Cautious regret minimization: Online optimization with long-term
budget constraints. In International Conference on Machine Learning, pages 3944–3952. PMLR,
2019.

Abhishek Sinha. BanditQ - Fair Bandits with Guaranteed Rewards. In Uncertainty in Artificial
Intelligence. PMLR, 2024a.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In
Proceedings of the 20th international conference on machine learning (icml-03), pages 928–936,
2003.

Elad Hazan, Alexander Rakhlin, and Peter Bartlett. Adaptive online gradient descent. Advances in
neural information processing systems, 20, 2007.

Francesco Orabona and Dávid Pál. Scale-free online learning. Theoretical Computer Science, 716:
50–69, 2018.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213,
2019.

Elad Hazan. Introduction to online convex optimization. MIT Press, 2022.

Jianjun Yuan and Andrew Lamperski.
Online convex optimization for cumulative constraints.
Advances in Neural Information Processing Systems, 31, 2018.

Rodolphe Jenatton, Jim Huang, and Cédric Archambeau. Adaptive algorithms for online convex
optimization with long-term constraints. In International Conference on Machine Learning, pages
402–411. PMLR, 2016.

Mehrdad Mahdavi, Rong Jin, and Tianbao Yang. Trading regret for efficiency: online convex
optimization with long term constraints. The Journal of Machine Learning Research, 13(1):
2503–2528, 2012.

Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Tianyou Chai, and Karl Johansson.
Regret and
cumulative constraint violation analysis for online convex optimization with long term constraints.
In International Conference on Machine Learning, pages 11998–12008. PMLR, 2021.

Michael J Neely and Hao Yu. Online convex optimization with time-varying constraints. arXiv
preprint arXiv:1702.04783, 2017.

Hao Yu, Michael Neely, and Xiaohan Wei. Online convex optimization with stochastic constraints.
Advances in Neural Information Processing Systems, 30, 2017.

Wen Sun, Debadeepta Dey, and Ashish Kapoor. Safety-aware algorithms for adversarial contextual
bandit. In International Conference on Machine Learning, pages 3280–3288. PMLR, 2017.

Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Yiguang Hong, Tianyou Chai, and Karl H Johansson.
Distributed online convex optimization with adversarial constraints: Reduced cumulative constraint
violation bounds under slater’s condition. arXiv preprint arXiv:2306.00149, 2023.

Michael J Neely. Stochastic network optimization with application to communication and queueing
systems. Synthesis Lectures on Communication Networks, 3(1):1–211, 2010.

11


---Page Break---
Hengquan Guo, Xin Liu, Honghao Wei, and Lei Ying. Online convex optimization with hard con-
straints: Towards the best of two worlds and beyond. Advances in Neural Information Processing
Systems, 35:36426–36439, 2022.

Tianyi Chen and Georgios B Giannakis. Bandit convex optimization for scalable and dynamic iot
management. IEEE Internet of Things Journal, 6(1):1276–1286, 2018.

Xuanyu Cao and KJ Ray Liu. Online convex optimization with time-varying constraints and bandit
feedback. IEEE Transactions on automatic control, 64(7):2665–2680, 2018.

Rahul Vaze. On dynamic regret and constraint violations in constrained online convex optimization.
In 2022 20th International Symposium on Modeling and Optimization in Mobile, Ad hoc, and
Wireless Networks (WiOpt), pages 9–16, 2022. doi: 10.23919/WiOpt56218.2022.9930613.

Qingsong Liu, Wenfei Wu, Longbo Huang, and Zhixuan Fang. Simultaneously achieving sublinear
regret and constraint violations for online convex optimization with time-varying constraints. ACM
SIGMETRICS Performance Evaluation Review, 49(3):4–5, 2022.

Hao Yu and Michael J Neely. A low complexity algorithm with o(
√

T) regret and o(1) constraint
violations for online convex optimization with long term constraints. Journal of Machine Learning
Research, 21(1):1–24, 2020.

Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Tianyou Chai, and H Karl. Regret and cumulative con-
straint violation analysis for distributed online constrained convex optimization. IEEE Transactions
on Automatic Control, 2022.

Hao Yu and Michael J Neely. A low complexity algorithm with o(
√

T) regret and o(1) constraint vio-
lations for online convex optimization with long term constraints. arXiv preprint arXiv:1604.02218,
2016.

CJ Argue, Sébastien Bubeck, Michael B Cohen, Anupam Gupta, and Yin Tat Lee. A nearly-linear
bound for chasing nested convex bodies. In Proceedings of the Thirtieth Annual ACM-SIAM
Symposium on Discrete Algorithms, pages 117–122. SIAM, 2019.

Søren Asmussen. Applied probability and queues, volume 2. Springer, 2003.

Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning
practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences,
116(32):15849–15854, 2019.

Andrea Dal Pozzolo, Olivier Caelen, Yann-Ael Le Borgne, Serge Waterschoot, and Gianluca Bon-
tempi. Learned lessons in credit card fraud detection from a practitioner perspective. Expert
systems with applications, 41(10):4915–4928, 2014. The dataset is available for download at
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/.

Abhishek Sinha. Source code for “Optimal Algorithms for Online Convex Optimization with
Adversarial Constraints"; A. Sinha, R. Vaze. https://github.com/abhishek-sinha-tifr/
COCO, 2024b.

Max Hopkins, Daniel M. Kane, Shachar Lovett, and Gaurav Mahajan. Realizable learning is all you
need. In Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings
of Machine Learning Research, pages 3015–3069. PMLR, 02–05 Jul 2022.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and
stochastic optimization. Journal of machine learning research, 12(7), 2011.

Jacob Abernethy, Chansoo Lee, Abhinav Sinha, and Ambuj Tewari. Online linear optimization via
smoothing. In Conference on Learning Theory, pages 807–823, 2014.

Pooria Joulani, Andras Gyorgy, and Csaba Szepesvári. Delay-tolerant online convex optimization:
Unified analysis and adaptive-gradient algorithms. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 30, 2016.

12


---Page Break---
Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint
arXiv:1706.05098, 2017.

Ofer Dekel, Philip M Long, and Yoram Singer. Online multitask learning. In International Conference
on Computational Learning Theory, pages 453–467. Springer, 2006.

Keerthiram Murugesan, Hanxiao Liu, Jaime Carbonell, and Yiming Yang. Adaptive smoothed online
multi-task learning. Advances in Neural Information Processing Systems, 29, 2016.

Nikhil Bansal, Martin Böhm, Marek Eliáš, Grigorios Koumoutsos, and Seeun William Umboh.
Nested convex bodies are chaseable. In Proceedings of the Twenty-Ninth Annual ACM-SIAM
Symposium on Discrete Algorithms, pages 1253–1260. SIAM, 2018.

Sébastien Bubeck, Bo’az Klartag, Yin Tat Lee, Yuanzhi Li, and Mark Sellke. Chasing nested convex
bodies nearly optimally. In Proceedings of the Fourteenth Annual ACM-SIAM Symposium on
Discrete Algorithms, pages 1496–1508. SIAM, 2020.

13


---Page Break---
A
Appendix

A.1
Discussion on Assumptions 1, 2 and 3

Assumptions 1 and 2 are standard in the online learning literature. The feasibility assumption
(Assumption 3) is analogous to the realizability assumption in learning theory [Hopkins et al., 2022]
and is commonly used in the COCO literature [Neely and Yu, 2017, Yu and Neely, 2016, Yuan and
Lamperski, 2018, Yi et al., 2023, Liakopoulos et al., 2019]. Assumption 3 requires the existence
of a single admissible action x⋆∈X that satisfies the constraints in every round. Consequently,
all constraint functions are required to be non-positive over a non-empty common subset. This
assumption is weakened in Section 3, Assumption 4, which only requires the existence of a fixed
admissible action x⋆that satisfies the constraints on average. Specifically, Assumption 4 requires
that the sum of the constraint functions evaluated at some admissible x⋆over any interval of length S
is non-positive. Notably, throughout the paper, we do not assume Slater’s condition as it does not
hold in many problems of interest [Yu and Neely, 2016]. As a result, unlike many previous works
[Yu et al., 2017], our bounds are independent of Slater’s constant, which can be problem-dependent.
Furthermore, we do not restrict the sign of either cost or constraint functions, allowing them to take
both positive and negative values.

A.2
Preliminaries on Online Convex Optimization (OCO)

The standard OCO problem can be described as a repeated game between an online policy and an
adversary [Hazan, 2022]. Let X ⊆Rd be a convex decision set, which we refer to as the admissible
set. In each round t ≥1, an online policy selects an action xt ∈X. After the action xt is chosen, the
adversary reveals a convex cost function ft ∶X ↦R. The goal of the online policy is to choose an
admissible action sequence {xt}t≥1 so that its total cost over a horizon of length T is not significantly
larger than the total cost incurred by any fixed admissible action x⋆∈X. More precisely, the objective
is to minimize the static regret, defined as:

RegretT ≡sup
x⋆∈X
RegretT (x⋆), where RegretT (x⋆) ≡
T
∑
t=1
ft(xt) −
T
∑
t=1
ft(x⋆).
(23)

Algorithm 3 Online Gradient Descent (OGD)

1: Input: Non-empty closed convex set X ⊆Rd, sequence of convex cost functions {ft}t≥1, step
sizes η1,η2,...,ηT > 0, Euclidean projection operator PX (⋅) onto the set X
2: Initialization: Set x1 ∈X arbitrarily
3: for each round t ≥1 do
4:
Play xt, observe ft, incur a cost of ft(xt).
5:
Compute a (sub)gradient ∇t ≡∇ft(xt).
6:
Update xt+1 = PX (xt −ηt∇t).
7: end for each

In a seminal paper, Zinkevich [2003] showed that the online gradient descent policy, outlined in
Algorithm 3, run with an appropriately chosen constant step size sequence, achieves a sublinear regret
bound RegretT = O(
√

T) for Lipschitz-continuous convex cost functions. In Theorem 6, we recall
two standard results on further refined data-dependent adaptive regret bounds achieved by the OGD
policy with appropriately chosen adaptive step size sequences.

Theorem 6. Consider the generic OGD policy outlined in Algorithm 3.

1. [Duchi et al., 2011], [Orabona, 2019, Theorem 4.14] Let the cost functions {ft}t≥1 be
convex and the step size sequence be adaptively chosen as ηt =
√

2D
2
√

∑t
τ=1 G2τ ,t ≥1, where

D is the Euclidean diameter of the admissible set X and Gt = ∣∣∇ft(xt)∣∣2,t ≥1. Then
Algorithm 3 achieves the following regret bound:

RegretT ≤
√

2D

¿
Á
Á
À

T
∑
t=1
G2
t.
(24)

14


---Page Break---
The OGD policy with the above adaptive step-size sequence is known as (a variant of) the
AdaGrad policy in the literature [Duchi et al., 2011].

2. [Hazan et al., 2007, Theorem 2.1] Let the cost functions {ft}t≥1 be strongly convex and let
Ht > 0 be the strong convexity parameter2 for the cost function ft. Let the step size sequence
be adaptively chosen as ηt =
1
∑t
s=1 Hs ,t ≥1. Then Algorithm 3 achieves the following regret
bound:

RegretT ≤1

2

T
∑
t=1

G2
t
∑t
s=1 Hs
.
(25)

Similar adaptive regret bounds are known for various other online learning policies as well. For
structured domains, one can use other algorithms such as AdaFTRL [Orabona and Pál, 2018]
which gives better regret bounds for high-dimensional problems. Furthermore, for problems with
combinatorial structures, adaptive oracle-efficient algorithms, e.g., Follow-the-Perturbed-Leader
(FTPL)-based policies, can be employed [Abernethy et al., 2014, Theorem 11]. Our proposed policies
are agnostic to the specific online learning subroutine used for the surrogate OCO problem - what
matters is that the subroutine provides adaptive regret bounds similar to (24) and (25). This flexibility
allows for an immediate extension of our algorithm to a wide range of settings, such as delayed
feedback [Joulani et al., 2016] or combinatorial actions.

A.3
Online Multi-task Learning as an Instance of the OCS Problem

Shared weights

Figure 3: A schematic for the online multi-task learning problem

Consider the problem of online multi-task learning where a single model is trained to perform a
number of related tasks [Ruder, 2017, Dekel et al., 2006, Murugesan et al., 2016]. See Figure 3
for a simplified schematic of the multi-task learning pipeline. In this setup, the action xt naturally
corresponds to the shared weight vector that specifies the common model for all tasks. The loss
function for the jth task on round t is given by the function gt,j(⋅),j ∈[k]. A task is assumed
to be satisfactorily completed (e.g., correct prediction in the case of classification problems) on
any round if the corresponding loss is non-positive. As an example, using linear predictors for
the binary classification problem, the requirement for the jth task on round t can be taken to be
gt,j(xt) ≡⟨zt,j,xt⟩≤0, where zt,j is the feature vector for the jth task. The goal in multi-task
learning is to sequentially update the shared weight vectors {xt}T
t=1 so that all tasks are successfully
completed. Formally, we require that the maximum cumulative loss of each task over any sub-interval
grows sub-linearly. Since the weight vector is shared across the tasks, the above goal would be
impossible to achieve had the tasks not been related to each other [Ruder, 2017]. Theorem 4 and
Theorem 5 give performance bounds for Algorithm 2 under different task-relatedness assumptions.

2The strong convexity of ft implies that ft(y) ≥ft(x) + ⟨∇ft(x), y −x⟩+ Ht

2 ∣∣x −y∣∣2, ∀x, y ∈X, ∀t.

15


---Page Break---
A.4
Proof of Theorem 3

We prove the Theorem via constructing an explicit input sequence for which no online policy can
have better than Ω(
√

T) regret and CCV.

Action space X:
Let d = T. Let X be the d-dimensional cuboid 0 ≤xi ≤
1
√

d, 1 ≤i ≤d. Clearly,
the Euclidean diameter of X is 1.

Input:
For each round we will only consider the case when only one constraint is revealed, i.e.,
k = 1. On round t = 1,...,d, choose the constraint gt to be xt ≤
1
4
√

d or xt ≥
3
4
√

d with equal

probability of 1

2 for x = (x1,...,xd) ∈X. Thus, at round t, only the tth dimension has an effective
constraint. If the chosen gt is xt ≤
1
4
√

d then pick ft = ∣x −
1
4
√

d∣, otherwise pick ft = ∣x −
3
4
√

d∣.

For any online policy A, the expected constraint violation at round t is at least
1
8
√

d. Thus, the overall

expected constraint violation over rounds t = 1,...,d is at least
√

d
8 . Moreover, the expected cost
E[ft(xt)] of A is at least
1
8
√

d for each t = 1,...,d, and the overall cost E[∑T
t=1 ft(xt)] is at least
√

d
8 .

Recall that the choice of input has to satisfy Assumption 3, i.e., X ⋆≠∅. We next demonstrate that
for the prescribed input ∃x⋆∈X ⋆.

Choosing a feasible x⋆:
When gt is such that the constraint is xt ≤
1
4
√

d choose x⋆∈X such

that x⋆
t =
1
4
√

d for t = 1,2,...,d, while if gt is such that xt ≥
3
4
√

d, then choose x⋆
t =
3
4
√

d for
t = 1,2,...,d. Thus, a single vector x⋆satisfies all the revealed constraints. Moreover, with this
choice of x⋆, the overall cost of x, ∑t ft(x⋆), is 0.

Since d = T, we get that for any online policy A its regret is at least Ω(
√

T) and the cumulative
constraint violation is Ω(
√

T).
∎

A.5
Comparison with Previous Works

A.5.1
Neely and Yu [2017], Yu et al. [2017] and Liakopoulos et al. [2019]

Policies proposed by Yu et al. [2017] and Liakopoulos et al. [2019] are almost identical to Neely and
Yu [2017]. The policy proposed in Neely and Yu [2017], however, is highly customized, does not
fully exploit the best guarantees available for the standard OCO problem, and obtains sub-optimal
performance bounds that depend inversely on Slater constant, which is assumed to be strictly positive.
In a nutshell, Neely and Yu [2017] choose the next action xt+1 using the algorithm described below.
For all rounds t ≥1, define the following evolution for Q(t) ∶

Q(t) = (Q(t −1) + gt(xt) + ∇T gt(xt)(xt −xt−1))+, Q(0) = 0.
(26)

The next action is chosen by solving the following quadratic optimization problem:

xt+1 = arg min
x∈X [⟨V ∇T ft(xt) + Q(t)∇gt(xt),x⟩+ α∣∣x −xt−1∣∣2],

where V and α are suitably chosen parameters.

In comparison, we have a different and simpler update rule:

Q(t) = Q(t −1) + (2GD)−1(gt(xt))+, Q(0) = 0.
(27)

We then construct a convex surrogate function ˆft(x) ≡ft(x)+
1
4GD
√

T e

Q(t)
2
√

T (gt(x))+, whose gradient
is then passed directly to the AdaGrad subroutine.

Remarks:
We emphasize that Theorem 1, which shows that it is possible to simultaneously achieve
O(
√

T) regret and ˜O(
√

T) CCV in the convex setting without assuming Slater’s condition, is highly
surprising and unexpected. In fact, Liakopoulos et al. [2019, Section 4] had previously commented
that:

16


---Page Break---
"... On the other hand, the point O(
√

T), O(
√

T) achieved by Neely and Yu [2017] for K = 1 is
not part of our achievable guarantees; we attribute this gap to the stricter Slater assumption studied
by Neely and Yu [2017]."

Theorem 1 squarely falsifies the last conjecture.

A.5.2
Guo et al. [2022]

The policy in Guo et al. [2022] is a slightly modified form of the policy proposed in [Neely and Yu,
2017]. In particular, it chooses the action xt by solving the following quadratic optimization problem
over X ∶

xt = arg min
x∈X [⟨∇ft−1(xt−1),x −xt−1⟩+ Q(t −1)γt−1g+
t−1(x) + αt−1∣∣x −xt−1∣∣2],

where the Q variables are updated as follows:

Q(t) = max(Q(t −1) + γt−1g+
t−1(xt),ηt).

Here αt,ηt,γt are suitably chosen learning rate parameters. Essentially, this policy is trying to find
the local optimum of an augmented Lagrangian under the online information model (ft and gt are
revealed after action xt is chosen). Since their augmented Lagrangian involves the constraint function
gt−1, their policy needs to solve a full-fledged constrained convex optimization problem over the
set X after having full access to the constraint function. In comparison, our policy, rather than
using approximations to Lagrangian and adding regularizers, makes full use of the well-developed
theory for OCO and uses first-order methods that need to compute only a gradient and perform one
Euclidean projection on each round.

A.5.3
Jenatton et al. [2016]

The policy proposed by Jenatton et al. [2016] is based on the idea of primal-dual algorithm for
optimizing the augmented Lagrangian

Lt(λ,x) = ft(x) + λgt(x) −θt

2 λ2,

where θt

2 λ2 is the augmentation term. The primal variable xt and the dual variable λt are updated by
executing projected gradient descent and gradient ascent on the Lagrangian as follows:

xt+1 = PX (xt −ηt∇xLt(xt,λt))

and
λt+1 = (λt + µt∇λLt(xt,λt))+,
where θt,ηt, and µt are parameters to be chosen.

A.6
Proof of Theorem 2

Bounding the CCV:
Choosing Φ(x) = x2 in Eqn. (14), we have for any feasible x⋆∈X ⋆∶

Q2(t) + V Regrett(x⋆) ≤V G2

α
(1 + ln(t)) + 4G2Q2(t)ln(Te)

αV
,
(28)

where, on the last term in the RHS, we have used the fact that t ≤T. Setting V = 8G2 ln(T e)

α
, and
transposing the last term on the RHS to the left, the above inequality yields

Q2(t) + 2V Regrett(x⋆) ≤2V G2

α
(1 + ln(t)).
(29)

Since the cost functions are assumed to be G-Lipschitz (Assumption 2), we trivially have
Regrett(x⋆) = ∑T
t=1(ft(xt) −ft(x⋆)) ≥−GDt. Hence, from Eqn. (29), we obtain

Q2(t) ≤2V GDt + 2V G2

α
(1 + ln(t)) Ô⇒Q(t)
(a)
≤4G

√

GD

α tln(Te) + 4G2 ln(Te)

α
.

where step (a), we have substituted V = 8G2 ln(T e)

α
. Hence, we have the following bound CCVt =

O(
√

t log T

α
).

17


---Page Break---
Bounding the regret:
Using the above choice for the parameter V and the fact that Q2(t) ≥0,
from Eqn. (29), we have

2V Regrett(x⋆) ≤2V G2

α
(1 + ln(t)).

This leads to the following logarithmic bound for regret for any feasible x⋆∈X ⋆∶

Regrett(x⋆) ≤G2

α (1 + ln(t)).
∎

A sharper CCV bound under the non-negative regret assumption:
We now establish an
improved CCV bound when the worst-case regret is non-negative on some round t ≥1. Let
supx⋆∈X ⋆Regrett(x⋆) ≥0 for some round t ≥1. Letting V = 8G2 ln(T e)

α
as above, from Eq. (29) we
have

Q2(t) ≤2V G2

α
(1 + ln(t)) Ô⇒Q(t) = O(lnT

α ),t ∈[T]. ∎

Comment:
From the above proof, it immediately follows that the same conclusion holds even
under the weaker assumption of −RegretT = O( log T

α ).

A.7
Adversaries Ensuring Non-negative Regret

Convex adversary:
An adversary is called convex if for any sequence of action sequence {xt}T
t=1,
the adversary chooses the cost function sequence {ft}T
t=1 such that for any T ≥1, we have

T
∑
t=1
ft(xt) ≥
T
∑
t=1
ft(¯xT ),
(30)

where ¯xT ≡1

T ∑T
t=1 xt. Hence, by definition, a convex adversary guarantees a non-negative regret
with respect to the average action ¯xT for all rounds. In the following, we give two examples of convex
adversaries.

1. Fixed adversary:
An adversary which always selects a fixed convex function f on all rounds is
a convex adversary. In this case, Eqn. (30) holds due to the Jensen’s inequality.

2. Minimax adversary:
Let F denote an arbitrary non-empty set of convex functions defined on
the admissible set X. Consider an adversary M, which, upon seeing the selected action xt, chooses
the worst cost function ft from the set F on round t ∶

ft ∈arg max
f∈F f(xt).

We now show that M is a convex adversary. By definition, for any round τ ∈[T], we have

fτ(xτ) ≥ft(xτ) Ô⇒fτ(xτ) ≥1

T

T
∑
t=1
ft(xτ).

Summing up the above inequalities for each τ ∈[T], we have

T
∑
τ=1
fτ(xτ) ≥
T
∑
t=1

1
T

T
∑
τ=1
ft(xτ)
(a)≥
T
∑
t=1
ft(¯xT ),
(31)

where inequality (a) follows upon applying Jensen’s inequality to each cost function. Eqn. (31) shows
that M is a convex adversary.

P.S. It can be easily seen that Fixed adversary is a special case of Minimax adversary where F = {f}.

18


---Page Break---
A.8
Connection Between OCS and the Convex Body Chasing Problem

A well-studied problem related to the OCS problem is the nested convex body chasing (NCBC)
problem [Bansal et al., 2018, Argue et al., 2019, Bubeck et al., 2020], where at each round t, a
convex set χt ⊆χ is revealed such that χt ⊆χt−1, where χ0 = χ ⊆Rd is a convex, compact, and
bounded set. The objective is to choose xt ∈χt so as to minimize the total movement cost across
rounds C = ∑T
t=1 ∣∣xt −xt−1∣∣2, where x0 ∈χ is some fixed action. In NCBC, action xt is chosen
after the set χt is revealed. This is in contrast to the OCS problem, where xt must be chosen before
the constraints gt,i’s are revealed at round t. Moreover, note that the nested condition χt ⊆χt−1 is
stricter than Assumption 3, which is applicable to the OCS problem. However, as we show next, a
feasible algorithm for NCBC also provides an upper bound on the CCV of the OCS problem under
Assumption 3.

In this reduction, we define χt as the intersection of the first kt convex constraints gτ,i ≤0,1 ≤
τ ≤t,i ∈[k], revealed up to round t for the OCS problem. It is easy to see that χt is convex and
χt ⊆χt−1,∀t. Let xt be the action chosen by an algorithm A for the NCBC problem after the set
χt is revealed. Note that χt ≠∅, thanks to Assumption 3. We now choose yt ∶= xt−1 as the action
for the OCS problem on round t, ensuring that action yt is chosen before the set χt is revealed. The
resulting ith constraint violation for the OCS problem at round t is given by

gt,i(yt)
(a)
≤gt,i(yt) −gt,i(yt+1) ≤G∣∣yt −yt+1∣∣,
where (a) follows from the feasibility of A for NCBC, yt+1 = xt ∈χt and hence gt,i(yt+1) ≤0.
Summing across rounds t = 1,...,T, and taking the max over all the k constraints, we get that the
CCV using A for the OCS is upper bounded by ∑T
t=2 G∣∣yt −yt+1∣∣≤∑T
t=2 G∣∣xt−1 −xt∣∣≤G ⋅CA,
where CA is the movement cost of A for the NCBC problem.

From prior work Bansal et al. [2018], Argue et al. [2019], Bubeck et al. [2020], it is known that for
NCBC, a Steiner point-based algorithm that chooses xt as the Steiner point of χt can achieve CA =
O(√dlog d), where χ ⊂Rd. Thus, the Steiner point-based algorithm (even though computationally
intensive) provides an O(√dlog d) constraint violation for the OCS as well. However, this result
is effective for problems where √dlog d = o(T). Our result efficiently overcomes this hurdle and
provides a bound under weaker feasibility assumptions even beyond √dlog d = o(T) – a setting that
is better motivated in practice for modern deep learning applications which are characteristically
high-dimensional.

A.9
Proof of Theorem 4

Generalized regret decomposition:
Fix any S-feasible benchmark x⋆∈XS, as given by Eqn. (16).
Then, from Eqn. (20), we have

Φ(τ) −Φ(τ −1)
≤
2
k
∑
i=1
Qi(τ)gτ,i(xτ)

=
2
k
∑
i=1
Qi(τ)(gτ,i(xτ) −gτ,i(x⋆)) + 2
k
∑
i=1
Qi(τ)gτ,i(x⋆)

=
ˆfτ(xτ) −ˆfτ(x⋆) + 2
k
∑
i=1
Qi(τ)gτ,i(x⋆).

Summing up the above inequalities from τ = 1 to τ = t, we have

k
∑
i=1
Q2
i (t) = Φ(t) ≤Regret′
t(x⋆) + 2
k
∑
i=1

t
∑
τ=1
Qi(τ)gτ,i(x⋆),
(32)

where Regret′(⋅) refers to the regret of the surrogate costs as before. We now bound the last term by
making use of the S-feasibility of the action x⋆as given by Eqn. (16). Let us now divide the entire
interval [1,t] into disjoint and consecutive sub-intervals {Ij}⌈t/S⌉
j=1 , each of length S (except the last
interval which could be of a smaller length). Let Q⋆
i (j) be the value of the variable Qi(⋅) at the
beginning of the jth interval. We have

t
∑
τ=1
Qi(τ)gτ,i(x⋆) =

⌈t/S⌉
∑
j=1
∑
τ∈Ij
(Qi(τ) −Q⋆
i (j))gτ,i(x⋆) +

⌈t/S⌉
∑
j=1
Q⋆
i (j) ∑
τ∈Ij
gτ,i(x⋆).
(33)

19


---Page Break---
Using the boundedness assumption, let gt,i(x) ≤F,∀x ∈X,t,i. Using the Lipschitzness property of
the queueing dynamics (17) with respect to time, we have

max
τ∈Ij ∣Qi(τ) −Q⋆
i (j)∣≤F(S −1).

Substituting the above bound into Eqn. (33), we obtain

t
∑
τ=1
Qi(τ)gτ,i(x⋆) ≤(1 + t

S )F 2S(S −1) + F(S −1)(Qi(t) + F(S −1)),
(34)

where in the last term, we have used the S-feasibility of the action x⋆in all intervals, except possibly
the last interval. Substituting the bound (34) into Eqn. (32), we arrive at the following extended regret
decomposition inequality:

k
∑
i=1
Q2
i (t)
≤
Regret′
t(x⋆) + 2kF 2St + 2FS
k
∑
i=1
Qi(t) + 4F 2S2k.
(35)

Eqn. (35) leads to the following bound on the cumulative constraint violation.

A.9.1
CCV Bound

We now apply the generalized regret decomposition bound given in (35) to the case of convex
constraint functions. Substituting the regret bound (24) of the AdaGrad policy into Eqn. (35), we
have

k
∑
i=1
Q2
i (t) ≤c1

¿
Á
Á
À

t
∑
τ=1
(
k
∑
i=1
Q2
i (τ)) + c2St + c3S
k
∑
i=1
Qi(t) + c4S2

where the constants c1 ≡O(GD
√

k),c2 = O(kF 2),c3 = O(F),c4 = O(kF 2) are problem-specific
parameters that depend on the bounds on the gradients and the maximum value of the constraint
functions, the number of constraints, and the diameter of the admissible set. Defining Q2(t) ≡
∑i Q2
i (t), we obtain:

Q2(t) ≤c1

¿
Á
Á
À

t
∑
τ=1
Q2(τ) + c2St + c3S
k
∑
i=1
Qi(t) + c4S2.

Since Qi(t) ≤Ft,∀i, the above inequality can be simplified to

Q2(t) ≤c1

¿
Á
Á
À

t
∑
τ=1
Q2(τ) + c′
2St + c4S2, ∀t ≥1,
(36)

where we have defined c′
2 ≡c3kF + c2. To solve the above system of inequalities, note that for each
1 ≤τ ≤t, we have

Q2(τ) ≤c1

¿
Á
Á
À

t
∑
τ=1
Q2(τ) + c′
2St + c4S2.

Summing up the above inequalities for 1 ≤τ ≤t and defining Zt ≡
√

∑t
τ=1 Q2(τ), we obtain

Z2
t
≤
c1tZt + c′
2St2 + c4S2t

i.e., Z2
t
≤
3max(c1tZt,c′
2St2,c4S2t)

i.e., Zt
=
O(max(t,t
√

S,S
√

t)).

Substituting the above bound for Z(t) in Eqn. (36), we have for any t ≥1:

Q2(t)
=
O(max(Zt,St,S2))

i.e., Q(t)
=
O(max(
√

Zt,
√

St,S))

Hence, Qi(t) ≤Q(t)
=
O(max(
√

t,
√

tS1/4,
√

St1/4,
√

St,S))

=
O(max(
√

St,S)), ∀i ∈[k].

The final result follows upon appealing to Eqn. (18).
∎

20


---Page Break---
A.10
Proof of Theorem 5

We will use a similar line of arguments used in the analysis of an S-constrained adversary for a
suitable value of S to be determined later. We start from Eqn. (33), which holds for any value of
the sub-interval length S ≥1 and any arbitrary adversary. Furthermore, from the definition of a
PT -constrained adversary, we know that there exists a benchmark x⋆∈X such that for any interval
Ij and any i ∈[k], we have:
∑
τ∈Ij
gτ,i(x⋆) ≤PT F,

where F is the maximum absolute value of the constraint functions as given in Definition 1. Hence,

⌈t/S⌉
∑
j=1
Q⋆
i (j) ∑
τ∈Ij
gτ,i(x⋆)
≤
PT F

⌈t/S⌉
∑
j=1
Q⋆
i (j)

≤
PT F

S

⌈t/S⌉
∑
j=1
∑
τ∈Ij
(Q⋆
i (j) −Qi(τ)) + PT F

S

t
∑
τ=1
Qi(τ).

Hence, from Eqn. (33), we have that

t
∑
τ=1
Qi(τ)gτ,i(x⋆)
≤
(1 + t

S )F 2S(S −1) + (1 + t

S )PT F 2(S −1) + PT F

S

t
∑
τ=1
Qi(τ)

≤
F 2(S + PT )(S + t) + PT F

S

t
∑
τ=1
Qi(τ).

Substituting the above bound into Eqn. (32), we have that

k
∑
i=1
Q2
i (t) ≤Regret′
t(x⋆) + 2kF 2(S + PT )(S + t) + 2PT F

S

t
∑
τ=1

k
∑
i=1
Qi(τ).

Plugging in the regret bound of the AdaGrad policy for the surrogate cost functions, the above
equation yields

k
∑
i=1
Q2
i (t) ≤GD
√

2k

¿
Á
Á
À

t
∑
τ=1
(
k
∑
i=1
Q2
i (τ)) + 2kF 2(S + PT )(S + t) + 2PT F

S

t
∑
τ=1

k
∑
i=1
Qi(τ).
(37)

Using Cauchy-Schwarz inequality, the last term of the above inequality can be upper bounded by

2PT F
√

kt
S

¿
Á
Á
À

t
∑
τ=1
(
k
∑
i=1
Q2
i (τ)).

Hence, we have the following inequality which holds for any 1 ≤S ≤t and 1 ≤τ ≤t ∶

k
∑
i=1
Q2
i (τ) ≤(GD
√

2k + 2PT F
√

kt
S
)

¿
Á
Á
À

t
∑
τ=1
(
k
∑
i=1
Q2
i (τ)) + 2kF 2(S + PT )(S + t).
(38)

Summing up the above inequalities for 1 ≤τ ≤t and defining Z2
t ≡∑t
τ=1 ∑k
i=1 Q2
i (τ), we have:

Z2
t
≤
(GD
√

2k + 2PT F
√

kt
S
)tZt + 2kF 2t(S + PT )(S + t)

≤
2max((GD
√

2k + 2PT F
√

kt
S
)tZt,2kF 2t(S + PT )(S + t)).

The above inequality implies that

Zt
≤
2max((GD
√

2k + 2PT F
√

kt
S
)t,F
√

kt(S + PT )(S + t))

≤
2max((GD
√

2k + 2PT F
√

kT
S
)T,FT
√

2k(S + PT )),
(39)

21


---Page Break---
where in the last step, we have used the fact that t ≤T and S ≤T. Now, let us choose S ≡P
2/3
T T
1/3.
With the above choice of S, from the above inequality, we have the following bound for Zt ∶

Zt ≤2max((GD
√

2k + 2F
√

kP
1/3
T T
1/6)T,2F
√

kP
1/3
T T
7/6) = O(P
1/3
T T
7/6) + O(T),

where we have used the fact that PT ≤T. Substituting the above bound in (38), we have for any
1 ≤i ≤k and any t ≤T ∶

k
∑
i=1
Q2
i (t)
(a)
= O(P
2/3
T T
4/3) + O(T) + O(P
2/3
T T
4/3) = O(P
2/3
T T
4/3) + O(T),

where in (a), we have used the fact that T ≥S ≥PT in bounding the last term. Hence, we have the
following upper bound on the queue lengths for any 1 ≤t ≤T

∣∣Q(t)∣∣∞≤∣∣Q(t)∣∣2 = O(P
1/3
T T
2/3) + O(
√

T).

The final result follows upon appealing to the relation (18).
∎

22


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We give complete proofs of all claims made in the paper (either in the main
paper or in the Appendix).

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

Justification: We clearly state the assumptions under which our results hold.

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

23


---Page Break---
Answer: [Yes]
Justification: We clearly state the assumptions in the main paper and give complete proofs
of all claims (either in the main paper or in the Appendix)

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

Justification: We have publicly released the code so that it can be used to reproduce the main
experimental results reported in this paper. Link to the codebase, which includes detailed
instructions on how to run the code, has been included in the paper.

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

24


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We have publicly released the code with clear instructions on how to replicate
the experimental results.

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

Justification: Sufficient details have been provided in the paper.

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

Justification: Since the paper deals with worst-case guarantees against adversarial inputs,
statistical guarantees are superfluous. Choice of random seeds used in the experiments have
been clearly specified in the code.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

25


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

Answer: [Yes]

Justification: Compute details have been mentioned in the paper.

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

Justification: This work conforms with the NeurIPS Code of Ethics in every respect.

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

Justification: This is a foundational work on online learning and does not have a direct
societal impact.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.

26


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

Justification: This theoretical paper does not pose any such risks.

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

Justification: The public dataset used in the experiments has been appropriately cited.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

27


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

Answer: [Yes]

Justification: The code we have released is well documented.

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

Justification: [NA]

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

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

28


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

29


---Page Break---
