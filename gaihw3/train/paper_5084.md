Gradient-Free Methods for Nonconvex Nonsmooth
Stochastic Compositional Optimization

Zhuanghua Liu
Department of Computer Science, National University of Singapore
CNRS@CREATE LTD, 1 Create Way, #08-01 CREATE Tower, Singapore 138602
liuzhuanghua9@gmail.com

Luo Luo∗
School of Data Science, Fudan University
Shanghai Key Laboratory for Contemporary Applied Mathematics
luoluo@fudan.edu.cn

Bryan Kian Hsiang Low
Department of Computer Science, National University of Singapore
lowkh@comp.nus.edu.sg

Abstract

Stochastic compositional optimization (SCO) problems are popular in many real-
world applications, including risk management, reinforcement learning, and meta-
learning. However, most of the previous methods for SCO require the smoothness
assumption on both the outer and inner functions, which limits their applications
to a wider range of problems. In this paper, we study the SCO problem in that
both the outer and inner functions are Lipschitz continuous but possibly nonconvex
and nonsmooth. In particular, we propose gradient-free stochastic methods for
finding the (δ, ϵ)-Goldstein stationary points of such problems with non-asymptotic
convergence rates. Our results also lead to an improved convergence rate for the
convex nonsmooth SCO problem. Furthermore, we conduct numerical experiments
to demonstrate the effectiveness of the proposed methods.

1
Introduction

In this paper, we consider the following stochastic compositional optimization (SCO) problem:

min
x∈Rd Φ(x) ≜f(g(x)),
(1)

where the outer and inner functions f : Rm →R and g: Rd →Rm has the form of

f(y) ≜Eξ[F(y; ξ)]
and
g(x) := Eζ[G(x; ζ)],

and the stochastic components F(y; ξ) and G(x; ζ) are Lipschitz continuous but possibly nonconvex
and nonsmooth. Random variables ξ and ζ are independent. Such formulation is popular in many real-
world applications, including risk management [1], statistical learning [2], reinforcement learning [3],
and model agnostic meta-learning [4].

Most of the existing work [2, 5, 6, 7, 8] for nonconvex SCO problem is based on the assump-
tion that both functions f(·) and g(·) are smooth. Unfortunately, many modern machine learning

∗The corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: We present the stochastic zeroth-order complexity of proposed algorithms for solving
nonsmooth stochastic compositional optimization problems.

METHODS
PROBLEM
COMPLEXITY
REFERENCE

GFCOM
NONCONVEX
O(d3.5δ−3ϵ−6)
COROLLARY 4.2

GFCOM+
NONCONVEX
O(d3.5δ−3ϵ−5)
COROLLARY 4.4

WS-GFCOM2
CONVEX
O(d3δ−2.4ϵ−4.8 + d3.5δ−2ϵ−6)
COROLLARY 5.3

WS-GFCOM+
CONVEX
O(d3δ−2.4ϵ−4 + d3.5δ−2ϵ−5)
COROLLARY 5.4

models including deep neural networks do not satisfy the smoothness condition. Ruszczynski [9]
proposed a single time-scale stochastic subgradient method for solving the Problem (1). However,
the author only provided asymptotic convergence analysis for the approach. In recent work, Liu
and Davanloo Tajbakhsh [10], Hu et al. [11] presented the non-asymptotic convergence for the
nonconvex nonsmooth SCO problem, while their analysis requires additional assumptions such as the
weak-convexity and the relative smoothness condition.

The non-smoothness in the Problem (1) implies the classical gradient-based approaches and the
convergence measure in terms of the gradient norm cannot be applied. The Clarke subdifferential [12]
for the Lipshitz continuous functions is a natural extension of gradients for the smooth functions.
However, hard instances have shown that no deterministic or randomized algorithms can find an ϵ-
stationary point with respect to the Clarke subdifferential of a Lipschitz function in finite time [13, 14].
To address this issue, Zhang et al. [13] proposed a refined notion of the (δ, ϵ)-Goldstein stationary
point in terms of the Goldstein δ-subdifferential, which considers the convex hull of the Clarke
subdifferential at points in the δ-neighbourhood [15].

In this paper, we propose a zeroth-order stochastic method called gradient-free compositional op-
timization method (GFCOM) for solving Problem (1) in finite time. In particular, we show that
the GFCOM can find a (δ, ϵ)-Goldstein stationary point of the objective function Φ(·) = f(g(·))
within the stochastic zeroth-order oracle complexity of O(d3.5δ−3ϵ−6). Furthermore, we improve
the GFCOM by using the variance reduction technique [16, 17, 18, 19] to establish a more effi-
cient first-order oracle estimator, leading to the algorithm GFCOM+ which achieves a tighter upper
complexity bound of O(d3.5δ−3ϵ−5). In addition, we study convex nonsmooth SCO problems. In
this regime, prior methods [2, 20, 21] suffer two major limitations: (i) Their convergence analysis
is based on the smoothness condition of the outer function. (ii) Their convergence result is mea-
sured by the sub-optimality of the function value gap, while the non-asymptotic convergence rate
for finding the stationary point has not been studied. We overcome these issues by involving a
warm-start strategy into GFCOM+, which is guarantee to find a (δ, ϵ)-Goldstein stationary point
within O(d3δ−2.4ϵ−4 + d3.5δ−2ϵ−5) stochastic zeroth-order oracle complexity. We summarize the
complexity of proposed methods in Table 1.

2
Related Work

In this section, we review prior work for stochastic compositional optimization and classical noncon-
vex nonsmooth optimization.

2.1
Stochastic Compositional Optimization

In a pioneer work, Wang et al. [2] studied the non-asymptotic convergence of nonconvex smooth
stochastic compositional optimization by proposing the stochastic compositional gradient descent
(SCGD), which contains two sequences of stepsizes for different time scales to update the variable
and track the inner function value, respectively. The authors also heuristically extended their methods
to zeroth-order optimization. Wang et al. [5] proposed an accelerated variant of SCGD using an
extrapolation-smoothing scheme, and Ghadimi et al. [6] proposed a single time-scale approach
to accelerate the convergence further. Additionally, Hu et al. [7], Lin et al. [20], Yuan et al. [22]
incorporated the variance reduction technique into the first-order iteration, achieving a tight stochastic
first-order complexity under the mean-squared smoothness assumption.

2


---Page Break---
Compared with the smooth counterpart, the study of nonsmooth compositional optimization is
relatively scarce. Ruszczynski [9] proposed a single time-scale stochastic subgradient method for
nonconvex nonsmooth SCO problems, while the theoretical analysis only provided the asymptotic
convergence rate. Liu and Davanloo Tajbakhsh [10] introduced the stochastic composition Bregman
gradient method and provided a non-asymptotic convergence analysis under the relative smoothness
condition. Vladarean et al. [23] proposed a Frank-Wolfe algorithm for constrained nonconvex
nonsmooth SCO problems. Their analysis assumes that the outer function is convex but possibly
non-differentiable, and the inner function is smooth. Very recently, Hu et al. [11] studied stochastic
methods for the finite-sum coupled compositional optimization problem. Their convergence rate is
established by assuming both the outer and inner functions are weakly convex, and the outer function
is non-decreasing. In addition, Kalogerias and Powell [24] studied the zeroth-order stochastic
optimization for a specific compositional optimization problem in risk-aware learning.

2.2
Non-Asymptotic convergence Analysis of Nonconvex Nonsmooth Optimization

In this subsection, we present a literature review for classical nonconvex nonsmooth optimization.
The study of this field has a long history [12, 25], but the non-asymptotic convergence analysis
of nonsmooth optimization has only emerged in recent years. Zhang et al. [13] provided the non-
asymptotic complexity analysis of the interpolated normalized gradient descent method to achieve
a (δ, ϵ)-Goldstein stationary point of a Lipschitz function with a nonstandard subgradient oracle.
Davis et al. [26], Tian et al. [27] improved this method by introducing random perturbations in each
iteration to remove the assumptions. Recently, Cutkosky et al. [28] proposed the optimal algorithm
via the reduction from nonconvex nonsmooth optimization to online learning.

The development of non-asymptotic convergence analysis of zeroth-order methods for nonsmooth
optimization was initiated by Nesterov and Spokoiny [29]. Later, Lin et al. [30] proposed gradient-
free methods for this problem by establishing a relationship between the Goldstein δ-subdifferential
and randomized smoothing. Chen et al. [31], Liu et al. [32] improved their results by leveraging the
variance-reduction technique. Kornowski and Shamir [33] obtained a sharper bound by applying
the reduction technique introduced by Cutkosky et al. [28] to the gradient-free setting. Liu et al.
[34], Grimmer and Jia [35] further extends the methodology to the constrained setting. However,
these methods do not apply to the nonconvex nonsmooth SCO Problem (1).

3
Preliminaries

In this section, we first present the notations and assumptions used in this paper, then introduce the
convergence criteria for nonsmooth optimization and the randomized smoothing technique.

3.1
Notations and Assumptions

We use ∥·∥to denote the Euclidean norm of a vector. We define Bδ(x) ≜{y ∈Rd : ∥y −x∥≤δ}
as the Euclidean ball centered at x ∈Rd with a radius δ > 0. We let conv(A) be the convex hull of
the set A. For two given sets A and B, we define A × B as their Cartesian product. In addition, we
denote f ◦g as the function composition such that (f ◦g)(x) ≜f(g(x)).

Throughout this paper, we assume the objective function (1) satisfies the following assumptions.
Assumption 3.1. We assume the stochastic component F(·, ξ) is Lf(ξ)-Lipschitz for any given ξ,
and the stochastic component G(·, ζ) is Lg(ζ)-Lipschitz for any given ζ. That is, it holds

|F(x, ξ) −F(y, ξ)| ≤Lf(ξ) ∥x −y∥
and
∥G(ˆx, ζ) −G(ˆy, ζ)∥≤Lg(ζ) ∥ˆx −ˆy∥,

for any x, y ∈Rd and ˆx, ˆy ∈Rm. We also assume the Lipschitz parameters Lf(ξ) and Lg(ζ)
have bounded second-order moments such that Eξ[Lf(ξ)2] ≤G2
f and Eζ[Lg(ζ)2] ≤G2
g for some
constants Gf, Gg > 0.
Remark 3.2. We can verify that Assumption 3.1 implies the function f(·) is Gf-Lipschitz, and the
function g(·) is Gg-Lipschitz by Jensen’s inequality.
Assumption 3.3. We assume that there exists a constant σ0 as the upper bound on the variance of
the functions G(·, ζ), such that for any x ∈Rd, we have Eζ
h
∥G(x, ζ) −g(x)∥2i
≤σ2
0.

3


---Page Break---
Assumption 3.4. We assume that the composite function Φ(·) ≜(f ◦g)(·) is lower bounded such
that Φ∗≜infx∈Rd Φ(x) > −∞.

3.2
Convergence Criteria for Nonsmooth Functions

We introduce the definitions of the Clarke subdifferential and approximate Clarke stationary points.

Definition 3.5 (Clarke [12]). The Clarke subdifferential of a Lipschitz function f at a point x is
defined as ∂f(x) ≜conv {g : g = limxk→x ∇f(xk)}. Furthermore, we call a point x an ϵ-Clarke
stationary point of f if it holds that min{∥g∥: g ∈∂f(x)} ≤ϵ.

Zhang et al. [13], Kornowski and Shamir [14] showed that no deterministic or randomized algorithm
could find an ϵ-Clarke stationary point in finite time. Consequently, Zhang et al. [13] considered a
refined notion of approximate stationary point in terms of the Goldstein δ-subdifferential.

Definition 3.6 (Zhang et al. [13]). Given a Lipschitz function f : Rd →R and δ > 0, the Goldstein
δ-subdifferential of f at point x ∈Rd is defined as ∂δf(x) := conv(∪y∈Bδ(x)∂f(y)), which is the
convex hull of the Clarke subdifferential at the points in the δ-neighbourhood of x. Additionally,
a point x ∈Rd is called a (δ, ϵ)-Goldstein stationary point of f(·) if it holds that min{∥g∥: g ∈
∂δf(x)} ≤ϵ.

Recent work [13, 26, 28] has shown that it is possible to find a (δ, ϵ)-Goldstein stationary point of
a nonsmooth problem without a composition structure in finite time. However, these theories are
not applicable to nonsmooth SCO. In particular, we can infer from Rademacher’s theorem and
Assumption 3.1 that the composite function Φ(·) is differentiable almost everywhere. Let Q ⊆Rd be
the set on which Φ is differentiable, then Rd \ Q is of measure zero. Recent work assumes they have
access to the unbiased stochastic gradient estimator of the objective function for any x ∈Q. In our
setting, the unbiased gradient estimator of the composite function Φ(x) is JG(x; ζ)∇F(g(x); ξ),
where JG is the Jacobian matrix of the function G(·; ζ). However, such an estimator is hard to obtain
because the function value g(x) is an expectation.

3.3
Randomized Smoothing

The randomized smoothing is a popular technique for nonsmooth analysis [36] and gradient-free
optimization [29]. Concretely, given a Lipschitz function f and a uniform distribution P on a unit ball,
we define its smoothed surrogate as fδ(x) = Eu∼P[f(x + δu)], which has the following properties.

Lemma 3.7 (Lin et al. [30, Proposition 2.3]). Let fδ(x) = Eu∼P[f(x + δu)] where P is a uniform
distribution on a unit ball in ℓ2-norm. Suppose the function f is L-Lipschitz, then we have (a)
|fδ(x)−f(x)| ≤δL; (b) fδ(·) is differentiable everywhere and L-Lipschitz with cL
√

dδ−1-Lipschitz
gradient, where c is some positive constant; (c) ∇fδ(x) ∈∂δf(x) for any x ∈Rd.

Moreover, we consider the following unbiased gradient estimator of the smoothed surrogate func-
tion fδ(·), which can be obtained from two function query oracle calls on points uniformly sampled
from a unit sphere [37].

Lemma 3.8 (Lin et al. [30, Lemma D.1]). Let f(x) = E[F(x; ξ)] be a L-Lipschitz function. We
denote

ι(x; u, ξ) ≜d

2δ (F(x + δu; ξ) −F(x −δu; ξ))u,

where u is uniformly sampled from a distribution on a unit sphere in Rd space.
Then, we
have E[ι(x; u, ξ)] = ∇fδ(x) and E[∥ι(x; u, ξ)∥2] ≤16
√

2πdL2.

4
Algorithms for Nonconvex Nonsmooth SCO

In this section, we propose zeroth-order stochastic algorithms for solving the nonconvex nonsmooth
SCO problem. We also provide non-asymptotic convergence analysis for the proposed methods.

4


---Page Break---
Algorithm 1: GFCOM(x0, η, T, bf, bg)

1 for t = 0, 1, . . . , T −1 do

2
Sample {ξt,i, wt,i}bf
i=1 and {ζt,i}bg
i=1.

3
Generate G(xt ± δwt,j; ζt,i) for every (i, j) ∈[bg] × [bf].

4
Let yt,j =
1
bg
P

i∈[bg] G(xt + δwt,j; ζt,i).

5
Let zt,j =
1
bg
P

i∈[bg] G(xt −δwt,j; ζt,i).

6
Let vt =
1
bf
P

j∈[bf ]
d
2δ(F(yt,j; ξt,j) −F(zt,j; ξt,j))wt,j.

7
Update xt+1 = xt −ηvt.

8 end

9 Return: xR where R is uniformly sampled from [T].

Algorithm 2: GFCOM+(x0, η, T, bf, b′
f, bg, b′
g, m)

1 for t = 0, 1, . . . , T −1 do

2
if t mod m = 0 then

3
Sample {ξt,i, wt,i}bf
i=1 and {ζt,i}bg
i=1.

4
Generate G(xt ± δwt,j; ζt,i) for every (i, j) ∈[bg] × [bf].

5
Let yt,j =
1
bg
P
i∈[bg] G(xt + δwt,j; ζt,i).

6
Let zt,j =
1
bg
P

i∈[bg] G(xt −δwt,j; ζt,i).

7
Let vt =
1
bf
P

j∈[bf ]
d
2δ(F(yt,j; ξt,j) −F(zt,j; ξt,j))wt,j.

8
else

9
Sample {ξt,i, wt,i}
b′
f
i=1 and {ζt,i}
b′
g
i=1.

10
Generate G(xt ± δwt,j; ζt,i) and G(xt−1 ± δwt,j; ζt,i) for every (i, j) ∈[b′
g] × [b′
f].

11
Let yk,j =
1
b′g
P

i∈[b′g] G(xk + δwt,j; ζk,i) for k ∈{t −1, t}.

12
Let zk,j =
1
b′g
P

i∈[b′g] G(xk −δwt,j; ζk,i) for k ∈{t −1, t}.

13
Let qk =
1
b′
f
P

j∈[b′
f ]
d
2δ(F(yk,j; ξt,j) −F(zk,j; ξt,j))wt,j for k ∈{t −1, t}.

14
Let vt = qt −qt−1 + vt−1.

15
end

16
Update xt+1 = xt −ηvt.

17 end

18 Return: xR where R is uniformly sampled from [T].

4.1
The Algorithms

In this subsection, we propose the gradient-free compositional optimization method (GFCOM) and
its accelerated variant GFCOM+. We first introduce the main intuition of the GFCOM. Consider the
following hypothetical zeroth-order gradient estimator

¯vt = 1

bf

X

j∈[bf ]

d
2δ (F(g(xt + δwt,j); ξt,j) −F(g(xt −δwt,j); ξt,j))wt,j,
(2)

where bf > 0 is the mini-batch size of the gradient estimator. By Lemma 3.8, the vector ¯vt is
an unbiased estimator of ∇Φδ(xt). Unfortunately, it is intractable to obtain the function values
g(xt ± δwt,j) because g(·) is an expectation of stochastic component functions G(·; ζ). To remedy
this issue, we introduce auxiliary variables yt,j and zt,j to approximate the inner function values
g(xt + δwt,j) and g(xt −δwt,j), respectively. In particular, the vectors yt,j and zt,j are mini-batch
function estimators defined as follows

yt,j = 1

bg

X

i∈[bg]
G(xt + δwt,j; ζt,i),
and
zt,j = 1

bg

X

i∈[bg]
G(xt −δwt,j; ζt,i),
(3)

5


---Page Break---
where bg > 0 is the mini-batch size of the function estimator. Accordingly, we use these two variables
to replace the function calls of g(·) in the gradient estimator vt of Eq. (2). The complete procedure
of the GFCOM is presented in Algorithm 1.

For the GFCOM+, we leverage the variance reduction technique to approximate ∇Φδ(xt). In
particular, we consider the following hypothetical recursive gradient estimator
¯vt = ¯qt −¯qt−1 + vt−1,
(4)
where ¯qt and ¯qt−1 are mini-batch gradient estimator defined as follows.

¯qk = 1

b′
f

X

j∈[b′
f ]

d
2δ (F(g(xk + δwt,j); ξt,j) −F(g(xk −δwt,j); ξt,j))wt,j,

where b′
f > 0 is the mini-batch size and k ∈{t −1, t}. Compared with the mini-batch gradient esti-
mator (2), the recursive gradient estimator (4) has been shown to achieve a sharper complexity bound
in nonconvex optimization literature [17, 18, 31]. However, the gradient estimator is computationally
intractable due to the unknown function g(·). Similar to the development of Algorithm 1, we define
yt, zt to estimate the inner function values g(xt ± δwt,j). We also introduce variables yt−1, zt−1
to approximate the inner function values g(xt−1 ± δwt,j) at the previous iteration. Then we define
stochastic gradient estimators qt and qt−1 in terms of yt, yt−1, zt and zt−1.

qk = 1

b′
f

X

j∈[b′
f ]

d
2δ (F(yk,j; ξt,j) −F(zk,j; ξt,j))wt,j,

for k ∈{t −1, t}. We replace the minibatch gradient estimator ¯qt and ¯qt−1 in the recursive gradient
estimator ¯vt of Eq. (4) with the refined gradient estimators qt and qt−1. The complete procedure of
GFCOM+ is presented in Algorithm 2.

4.2
Convergence Analysis

In this subsection, we consider the complexity analysis of the proposed algorithms introduced in
Section 4.1. We assume that Φ(x0) −Φ∗≤R, where R > 0 is some constant.

The following theorem shows the convergence rate of solving the Problem (1) with the GFCOM
method presented in Algorithm 1.
Theorem 4.1. Under Assumption 3.1, 3.3 and 3.4, running the GFCOM algorithm (Algorithm 1)
with η ≤δ/(cGfGg
√

d) where c > 0 is some constant, then the output xR satisfies

E
h
∥∇Φδ(xR)∥2i
= O

 
GfGg
√

dR
δT
+
G2
fG2
g
√

d
T
+
dG2
fG2
g
bf
+
d2G2
fσ2
0
δ2bg

!

.
(5)

Using Theorem 4.1 with the parameter setting

T = Θ

 
GfGg
√

dR
δϵ2
+
G2
fG2
g
√

d
ϵ2

!

,
bf = Θ

 
dG2
fG2
g
ϵ2

!

and
bg = Θ

 
d2G2
fσ2
0
δ2ϵ2

!

,

we obtain the following oracle complexity result for Algorithm 1.
Corollary 4.2. Under Assumption 3.1, 3.3 and 3.4, the GFCOM algorithm (Algorithm 1) requires at
most O
 
d3.5G5
fG3
gσ2
0Rδ−3ϵ−6 + d3.5G6
fG4
gσ2
0δ−2ϵ−6
stochastic zeroth-order function query calls
to obtain a (δ, ϵ)-Goldstein stationary point of Φ.

After giving the complexity bound of GFCOM in Corollary 4.2, we now consider the convergence
analysis of GFCOM+. We will show that it enjoys a sharper complexity bound due to the utilization
of the recursive gradient estimator. The following theorem shows the convergence rate of solving
Problem (1) with the GFCOM+ (Algorithm 2).
Theorem 4.3. Under Assumption 3.1, 3.3 and 3.4, running the GFCOM+ algorithm (Algorithm 2)
with η=δ/(2cGfGg
√

d), b′
f =Θ(dGfGgϵ−1) and m=Θ(GfGgϵ−1), then the output xR satisfies

E
h
∥∇Φδ(xR)∥2i
= O

 √

dGfGgR

δT
+

√

dG2
fG2
g
T
+
dG2
fG2
g
bf
+
d2G2
fσ2
0
δ2bg
+
d2G2
fσ2
0
δ2b′g

!

.

6


---Page Break---
Algorithm 3: WS-GFCOM(x0, η0, T0, bg,0, η, T, bf, bg, b′
f, b′
g, m)

1 Let x1 = GFCOM(x0, η0, T0, 1, bg,0).

2 Option I (WS-GFCOM2): x2 = GFCOM(x1, η, T, bf, bg).

3 Option II (WS-GFCOM+): x2 = GFCOM+(x1, η, T, bf, b′
f, bg, b′
g, m).

4 Return: x2.

Using Theorem 4.3 with the parameter setting

T =Θ

 √

dGfGgR

δϵ2
+

√

dG2
fG2
g
ϵ2

!

, bf =Θ

 
dG2
fG2
g
ϵ2

!

, bg =b′
g =Θ

 
d2G2
fσ2
0
δ2ϵ2

!

,

we obtain the following oracle complexity result for Algorithm 2.
Corollary 4.4. Under Assumption 3.1, 3.3 and 3.4, the GFCOM+ algorithm (Algorithm 2) requires
at most O
 
d3.5G4
fG2
gσ2
0Rδ−3ϵ−5 + d3.5G5
fG3
gσ2
0δ−2ϵ−5
stochastic zeroth-order function query
calls to obtain a (δ, ϵ)-Goldstein stationary point of Φ.

For both Theorem 4.1 and 4.3, we take c = 1 according to Lemma 8 of Duchi et al. [36].

4.3
Discussion

In Algorithm 2, we only apply the variance reduction technique to the outer function f(·) to accelerate
our algorithm. In contrast, existing methods [2, 7, 22] for nonconvex smooth SCO also apply the
technique on the inner function estimator to obtain an improved complexity bound. Here we briefly
discuss the cause that leads to such a difference. For smooth optimization, the main intuition
of the variance reduction technique is to establish a connection between the bound of the mean-
square error term E

∥vt −∇Φ(xt))∥2 
and the expected distance of iterates at successive iterations
E[∥xt −xt−1∥2], which diminishes asymptotically. In our algorithm, we exploit the randomized
smoothing with perturbed iterates xt ± δwt,j to approximate the gradient of the smoothed surrogate
function Φδ(xt). If we apply the variance reduction to the inner function estimator, the mean-
square error E[∥vt −∇Φδ(xt))∥2] is bounded by the expected distance of the perturbed iterates at
successive iterations E[∥xt −xt−1 ± δ(wt,j −wt−1,j)∥2], which does not vanish asymptotically.

5
Extensions to Convex Nonsmooth SCO

In this section, we extend the result in Section 4.1 to study the convex nonsmooth SCO problem.
Firstly, we introduce an additional assumption as follows.
Assumption 5.1. We suppose the function f(x) is convex and non-decreasing, G(x; ζ) is convex for
any given ζ, and the solution set X ∗= arg minx∈Rd Φ(x) is non-empty.

From this assumption and Section 3.2.4 by Boyd and Vandenberghe [38], we can deduce that Φ(x) is
a convex function. We will show that stochastic algorithms obtain an improved convergence rate for
the nonsmooth SCO problem with Assumption 5.1. To obtain a (δ, ϵ)-Goldstein stationary point of
the problem, we propose a two-phase gradient-free stochastic method called warm-started GFCOM
(WS-GFCOM) in Algorithm 3. The intuition is that we use the GFCOM method to get a sufficiently
small sub-optimality in the first phase, and then we apply the proposed methods in Section 4.1 to
find the stationary point in the second phase. We remark that using the GFCOM method for the
first phase is justified by the observation that it can achieve optimal convergence rate in terms of the
sub-optimality of function value gap given the access to the exact function value g(x) [39].

5.1
Convergence Analysis

In this subsection, we consider the complexity analysis of the proposed WS-GFCOM method. Let
ˆR ≜minx∈X ∗∥x −x0∥. We characterize the convergence rate of the WS-GFCOM method for the
convex nonsmooth SCO problem at the first phase with the following theorem.

7


---Page Break---
Theorem 5.2. Under Assumption 3.1, 3.3 and 5.1, running the WS-GFCOM algorithm (Algorithm 3)
with parameters η0 = Θ
  ˆR/(GfGg
√dT0)

, T0 = Θ
 
dG2
fG2
g ˆR2ρ−2
, bg,0 = Θ
 
G2
fσ2
0ρ−2
and
δ = Θ
 
ρG−1
f G−1
g

, then the output x1 satisfies E[Φ(x1) −Φ∗] ≤ρ. In addition, the total zeroth-
order stochastic oracle complexity is at most O
 
dG4
fG2
gσ2
0 ˆR2ρ−4
.

Theorem 5.2 implies that the initial suboptimality at the beginning of the second phase is bounded
by ρ. Consequently, the total complexity of WS-GFCOM2 (Algorithm 3 with Option I) is bounded
by O
 
dG4
fG2
gσ2
0 ˆR2ρ−4 + d3.5G5
fG3
gσ2
0ρδ−3ϵ−6 + d3.5G6
fG4
gσ2
0δ−2ϵ−6
. An appropriate choice of
ρ leads to the following oracle complexity of Algorithm 3 with Option I.
Corollary 5.3. Under Assumption 3.1, 3.3 and 5.1, the WS-GFCOM2 algorithm (Algorithm 3
with Option I) requires at most O
 
d3G4.8
f G2.8
g σ2
0 ˆR0.4δ−2.4ϵ−4.8 + d3.5G6
fG4
gσ2
0δ−2ϵ−6
stochastic
zeroth-order function query calls to obtain a (δ, ϵ)-Goldstein stationary point of Φ.

With a similar deduction, we can show that using the GFCOM+ for the second phase can obtain an
improved complexity bound. The following theorem shows the oracle complexity of Algorithm 3
with Option II.
Corollary 5.4. Under Assumption 3.1, 3.3 and 5.1, the WS-GFCOM+ algorithm (Algorithm 3
with Option II) requires at most O
 
d3G4
fG2
gσ2
0 ˆR0.4δ−2.4ϵ−4 + d3.5G5
fG3
gσ2
0δ−2ϵ−5
stochastic
zeroth-order function query calls to obtain a (δ, ϵ)-Goldstein stationary point of Φ.

6
Experiments

We compare the proposed methods GFCOM and GFCOM+ with a Kiefer-Wolfowitz style zeroth-order
baseline method [2, 40]. In particular, the baseline gradient estimator is defined as

vt = 1

bf

X

j∈[bf ]

d
2δ
 
F(yt,j; ξt,j) −F(zt,j; ξ′
t,j)

,

where yt,j and zt,j are function estimators defined in Eq. (3). ξt,j and ξ′
t,j are independent random
variables. We test all the methods on the nonconvex penalized risk-averse portfolio management
problem and a reinforcement learning (RL) problem. We set δ = 0.1 for the GFCOM and GFCOM+
methods.

6.1
Nonconvex Penalized Portfolio Management

We consider the portfolio management problem with capped-ℓ1 regularizer [41]. Let x denote the
investment quantity corresponding to N assets and rt ∈RN denote the returns of N assets at
timestamp t. We can formulate the portfolio management problem as the following nonsmooth
compositional optimization problem

min
x∈RN −1

T

T
X

t=1
⟨rt, x⟩+ 1

T

T
X

i=1

 

⟨rt, x⟩−1

T

T
X

s=1
⟨rs, x⟩

!2

+ β(x),
(6)

where β(x) = λ PN
i=1 min{|xi|, α} and λ, α > 0 are tunable hyperparameters. Specifically, the
inner function G(x; ξ) and outer function F(w; ζ) can be formulated as

G(x; ξ) = [x1, . . . , xN, ⟨rξ, x⟩]⊤

and

F(w; ζ) = −⟨rζ, w[N]⟩+ (⟨rζ, w[N]⟩−wN+1)2 + β(x).

Both random variables ξ and ζ are uniformly sampled from {1, . . . , T}. We choose λ = 10−5 and
α = 2 in our experiments. The goal of the Problem (6) is to maximize the return while controlling
the variance of the portfolio.

We compare all the methods on 6 different portfolio datasets formed on Size and Operating Profitabil-
ity2. For all algorithms, we tune the stepsize among {1 × 10−5, 3 × 10−5, . . . , 1 × 10−3, 3 × 10−3}.

2http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

8


---Page Break---
0
1
2
3
Function Calls
1e8

0

100

200

300

400

500

Loss

Kiefer
GFCOM
GFCOM

0
1
2
3
Function Calls
1e8

0

100

200

300

400

500

Loss

Kiefer
GFCOM
GFCOM

0
1
2
3
Function Calls
1e8

0

100

200

300

400

Loss

Kiefer
GFCOM
GFCOM

(a) Asia Pacific ex Japan OP
(b) Europe OP
(c) Global ex US OP

0
1
2
3
Function Calls
1e8

0

100

200

300

Loss

Kiefer
GFCOM
GFCOM

0.0
0.5
1.0
1.5
2.0
Function Calls
1e8

0

200

400

600

800

Loss

Kiefer
GFCOM
GFCOM

0.0
0.5
1.0
1.5
2.0
Function Calls
1e8

0

200

400

600

Loss

Kiefer
GFCOM
GFCOM

(e) Global OP
(f) Japan OP
(g) North America OP

Figure 1: We present the loss vs. complexity on several portfolio management datasets. The plot of
GFCOM and the Kiefer-Wolfowitz method are overlapped as their performance are close to each
other.

0.0
0.5
1.0
Function Calls
1e7

0.250

0.275

0.300

0.325

0.350

0.375

0.400

Loss

Kiefer
GFCOM
GFCOM

0.00
0.25
0.50
0.75
1.00
Function Calls
1e7

0.250

0.275

0.300

0.325

0.350

0.375

0.400

Loss

Kiefer
GFCOM
GFCOM

0.0
0.5
1.0
Function Calls
1e7

0.250

0.275

0.300

0.325

0.350

0.375

0.400

Loss

Kiefer
GFCOM
GFCOM

(a) n = 400
(b) n = 600
(c) n = 800

Figure 2: For the RL task, we present the loss vs. complexity on datasets with states of different sizes.

We choose the mini-batch size bf = bg = 1000. In addition, we set b′
f = 100, b′
g = 1000 and
m = bf/b′
f = 10 for the GFCOM+ algorithm. Figure 1 shows that the GFCOM+ algorithm converges
much faster than the GFCOM and the baseline method across all datasets.

6.2
Application to Reinforcement Learning

We demonstrate an experiment on RL and verify the effectiveness of the proposed methods on value
function evaluation. Let V π(s) be the value function of a state s under a policy π for all state s ∈S
where |S| = n. Let rs′,s be the reward transition from s′ to s, and γ > 0 is a discounting factor.
Furthermore, we assume that the value of each state can be parameterized as a linear map of some
feature map ψs ∈Rd of the state s such that V π(s) = ⟨ψs, w⟩. Then we formulate the RL problem
as a Bellman residual minimization problem

min
w∈Rd

n
X

s=1
h

 

⟨ψs, w⟩−
X

s′
Pss′(rs,s′ + γ⟨ψs′, w⟩)

!

,

where Pss′ is the probability transition matrix and h(x) = 1−exp(−|x|/σ) is a nonconvex nonsmooth
loss which is more robust to adversarial outliers than the squared loss [42, 43]. Specifically, the inner
function G(x; ξ) and outer function F(w; ζ) can be formulated as

G(w; ξ) = [⟨ψ1, w⟩, r1,ξ1 + γ⟨ψξ1, w⟩, . . . , ⟨ψn, w⟩, rn,ξn + γ⟨ψξn, w⟩]⊤

9


---Page Break---
and

F(z; ζ) = h(z2ζ −z2ζ+1).

In the above formulation, each ξi is uniformly sampled from {Pi1, . . . , Pin} and ζ is uniformly
sampled from {1, . . . , T}. We follow a similar experiment setup by Yuan et al. [22]. Specifically,
we generate a Markov decision process with different numbers of states n ∈{400, 600, 800} and 10
actions at each state. The transition probability matrix is generated from the uniform distribution
from [0, 1]. In addition, the rewards are sampled uniformly from [0, 1]. In terms of hyperparameter
setting, we choose bf = bg = 100 for all algorithms. In addition, we set b′
f = 10, b′
g = 100 and
m = bf/b′
f = 10 for the GFCOM+ algorithm. For other hyperparameters, we use the same setting
for the portfolio management problem. The experimental results in Figure 2 show that the GFCOM+
significantly outperforms other methods.

7
Conclusion

In this work, we propose novel zeroth-order algorithms for nonconvex nonsmooth stochastic compo-
sitional optimization. We present the non-asymptotic convergence rate of the proposed algorithms for
obtaining a (δ, ϵ)-Goldstein point of the problem. Furthermore, we extend our methods with a warm-
start phase to solve the convex nonsmooth SCO problem with improved convergence guarantees. We
conduct numerical experiments on portfolio management and reinforcement learning problems to
demonstrate the effectiveness of the proposed algorithms.

In future work, it is interesting to study the lower bound of the zeroth-order algorithms on nonconvex
nonsmooth SCO. It is also interesting to investigate whether the complexity bound of zeroth-order
algorithms can be further improved.

Acknowledgement

This research/project is supported by the National Research Foundation, Singapore under its AI
Singapore Programme (AISG Award No: AISG2-PhD-2023-08-043T-J). This research is part of
the programme DesCartes and is supported by the National Research Foundation, Prime Minister’s
Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE)
programme. Luo Luo is supported by National Natural Science Foundation of China (No. 62206058),
Shanghai Sailing Program (22YF1402900), Shanghai Basic Research Program (23JC1401000), and
the Major Key Project of PCL under Grant PCL2024A06.

References

[1] Darinka Dentcheva, Spiridon Penev, and Andrzej Ruszczy´nski. Statistical estimation of com-
posite risk functionals and risk optimization problems. Annals of the Institute of Statistical
Mathematics, 69:737–760, 2017.

[2] Mengdi Wang, Ethan X Fang, and Han Liu. Stochastic compositional gradient descent: algo-
rithms for minimizing compositions of expected-value functions. Mathematical Programming,
161:419–449, 2017.

[3] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press,
2018.

[4] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adap-
tation of deep networks. In International conference on machine learning, pages 1126–1135.
PMLR, 2017.

[5] Mengdi Wang, Ji Liu, and Ethan X Fang. Accelerating stochastic composition optimization.
Journal of Machine Learning Research, 18(105):1–23, 2017.

[6] Saeed Ghadimi, Andrzej Ruszczynski, and Mengdi Wang. A single timescale stochastic
approximation method for nested stochastic optimization. SIAM Journal on Optimization, 30
(1):960–979, 2020.

10


---Page Break---
[7] Wenqing Hu, Chris Junchi Li, Xiangru Lian, Ji Liu, and Huizhuo Yuan. Efficient smooth
non-convex stochastic compositional optimization via stochastic recursive gradient descent.
Advances in Neural Information Processing Systems, 32, 2019.

[8] Tianyi Chen, Yuejiao Sun, and Wotao Yin. Solving stochastic compositional optimization is
nearly as easy as solving stochastic optimization. IEEE Transactions on Signal Processing, 69:
4937–4948, 2021.

[9] Andrzej Ruszczynski. A stochastic subgradient method for nonsmooth nonconvex multilevel
composition optimization. SIAM Journal on Control and Optimization, 59(3):2301–2320, 2021.

[10] Yin Liu and Sam Davanloo Tajbakhsh. Stochastic composition optimization of functions
without lipschitz continuous gradient. Journal of Optimization Theory and Applications, 198
(1):239–289, 2023.

[11] Quanqi Hu, Dixian Zhu, and Tianbao Yang. Non-smooth weakly-convex finite-sum coupled
compositional optimization. Advances in Neural Information Processing Systems, 36, 2024.

[12] Frank H. Clarke. Optimization and nonsmooth analysis. SIAM, 1990.

[13] Jingzhao Zhang, Hongzhou Lin, Stefanie Jegelka, Ali Jadbabaie, and Suvrit Sra. Complexity
of finding stationary points of nonsmooth nonconvex functions. In Proc. ICML, pages 11173–
11182, 2020.

[14] Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimization.
Journal of Machine Learning Research, 23(314):1–44, 2022.

[15] AA Goldstein. Optimization of lipschitz continuous functions. Mathematical Programming, 13:
14–22, 1977.

[16] Lam M Nguyen, Jie Liu, Katya Scheinberg, and Martin Takáˇc. Sarah: A novel method for
machine learning problems using stochastic recursive gradient. In International conference on
machine learning, pages 2613–2621. PMLR, 2017.

[17] Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal non-
convex optimization via stochastic path-integrated differential estimator. Advances in neural
information processing systems, 31, 2018.

[18] Zhe Wang, Kaiyi Ji, Yi Zhou, Yingbin Liang, and Vahid Tarokh. Spiderboost and momentum:
Faster variance reduction algorithms. Advances in Neural Information Processing Systems, 32,
2019.

[19] Zhize Li, Hongyan Bao, Xiangliang Zhang, and Peter Richtárik. Page: A simple and optimal
probabilistic gradient estimator for nonconvex optimization. In International conference on
machine learning, pages 6286–6295. PMLR, 2021.

[20] Tianyi Lin, Chengyou Fan, Mengdi Wang, and Michael I Jordan. Improved sample complexity
for stochastic compositional variance reduced gradient. In 2020 American Control Conference
(ACC), pages 126–131. IEEE, 2020.

[21] Yibo Xu and Yangyang Xu. Katyusha acceleration for convex finite-sum compositional opti-
mization. INFORMS Journal on Optimization, 3(4):418–443, 2021.

[22] Huizhuo Yuan, Xiangru Lian, and Ji Liu. Stochastic recursive variance reduction for efficient
smooth non-convex compositional optimization. arXiv preprint arXiv:1912.13515, 2019.

[23] Maria-Luiza Vladarean, Nikita Doikov, Martin Jaggi, and Nicolas Flammarion. Linearization
algorithms for fully composite optimization. In The Thirty Sixth Annual Conference on Learning
Theory, pages 3669–3695. PMLR, 2023.

[24] Dionysios S Kalogerias and Warren B Powell. Zeroth-order stochastic compositional algorithms
for risk-aware learning. SIAM Journal on Optimization, 32(2):386–416, 2022.

[25] Marko M Makela and Pekka Neittaanmaki. Nonsmooth optimization: analysis and algorithms
with applications to optimal control. World Scientific, 1992.

11


---Page Break---
[26] Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A
gradient sampling method with complexity guarantees for lipschitz functions in high and low
dimensions. Advances in neural information processing systems, 35:6692–6703, 2022.

[27] Lai Tian, Kaiwen Zhou, and Anthony Man-Cho So. On the finite-time complexity and practical
computation of approximate stationarity concepts of lipschitz functions. In International
Conference on Machine Learning, pages 21360–21379. PMLR, 2022.

[28] Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth non-
convex optimization through online-to-non-convex conversion. In International Conference on
Machine Learning, pages 6643–6670. PMLR, 2023.

[29] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions.
Foundations of Computational Mathematics, 17(2):527–566, 2017.

[30] Tianyi Lin, Zeyu Zheng, and Michael Jordan. Gradient-free methods for deterministic and
stochastic nonsmooth nonconvex optimization. Advances in Neural Information Processing
Systems, 35:26160–26175, 2022.

[31] Lesi Chen, Jing Xu, and Luo Luo. Faster gradient-free algorithms for nonsmooth nonconvex
stochastic optimization. In International Conference on Machine Learning, pages 5219–5233.
PMLR, 2023.

[32] Chengchang Liu, Chaowen Guan, Jianhao He, and John Lui. Quantum algorithms for non-
smooth non-convex optimization. arXiv preprint arXiv:2410.16189, 2024.

[33] Guy Kornowski and Ohad Shamir. An algorithm with optimal dimension-dependence for
zero-order nonsmooth nonconvex stochastic optimization. arXiv preprint arXiv:2307.04504,
2023.

[34] Zhuanghua Liu, Cheng Chen, Luo Luo, and Bryan Kian Hsiang Low. Zeroth-order methods
for constrained nonconvex nonsmooth stochastic optimization. In Forty-first International
Conference on Machine Learning, 2024.

[35] Benjamin Grimmer and Zhichao Jia. Goldstein stationarity in lipschitz constrained optimization.
arXiv preprint arXiv:2310.03690, page 9, 2023.

[36] John C Duchi, Peter L Bartlett, and Martin J Wainwright. Randomized smoothing for stochastic
optimization. SIAM Journal on Optimization, 22(2):674–701, 2012.

[37] John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for
zero-order convex optimization: The power of two function evaluations. IEEE Transactions on
Information Theory, 61(5):2788–2806, 2015.

[38] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press,
2004.

[39] Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with
two-point feedback. Journal of Machine Learning Research, 18(52):1–11, 2017.

[40] Jack Kiefer and Jacob Wolfowitz. Stochastic estimation of the maximum of a regression
function. The Annals of Mathematical Statistics, pages 462–466, 1952.

[41] Tong Zhang. Analysis of multi-stage convex relaxation for sparse regularization. Journal of
Machine Learning Research, 11(3), 2010.

[42] Chao Qu, Yan Li, and Huan Xu. Non-convex conditional gradient sliding. In international
conference on machine learning, pages 4208–4217. PMLR, 2018.

[43] Zebang Shen, Cong Fang, Peilin Zhao, Junzhou Huang, and Hui Qian. Complexities in
projection-free stochastic non-convex minimization. In The 22nd International Conference on
Artificial Intelligence and Statistics, pages 2868–2876. PMLR, 2019.

12


---Page Break---
The appendix is organized as follows. Section A introduces supporting lemmas that are essential for
the analysis of the proposed gradient-free SCO methods. Section B proves the convergence rate of
the GFCOM method introduced in Section 4. Section C proves the convergence rate of the GFCOM+
method which enjoys a better function oracle call complexity. Section D provides the convergence
analysis of the WS-GFCOM2 and WS-GFCOM+ proposed in Section 5.

A
Supporting Lemmas

Throughout the work, we define the variable gt(x) =
1
bg
P

i∈[bg] G(x; ζt,i) for the GFCOM algo-
rithm, and we denote

gt(x) =

( 1

bg
P

i∈[bg] G(x; ζt,i),
t mod m = 0

1
b′g
P
i∈[b′g] G(x; ζt,i),
Otherwise
(7)

for the GFCOM+ method. Now we introduce an important lemma which is useful for the analysis of
both GFCOM and GFCOM+ algorithms.
Lemma A.1. Under Assumption 3.1 and 3.3, for both Algorithms 1 and 2 it holds that

E[(f ◦g)δ(xt+1) −(f ◦g)δ(xt)]

≤−η

2E
h
∥∇(f ◦g)δ(xt)∥2i
−

 
η
2 −cη2GfGg
√

d
2δ

!

E
h
∥vt∥2i
+ η

2E
h
∥vt−∇(f ◦g)δ(xt)∥2i
.

Proof. From the smoothness of Φδ = (f ◦g)δ, we have

(f ◦g)δ(xt+1) −(f ◦g)δ(xt)

≤⟨∇(f ◦g)δ(xt), xt+1 −xt⟩+ cGfGg
√

d
2δ
∥xt+1 −xt∥2

= −η[⟨∇(f ◦g)δ(xt), vt⟩] + cη2GfGg
√

d
2δ
∥vt∥2

= −η

2 ∥∇(f ◦g)δ(xt)∥2 −

 
η
2 −cη2GfGg
√

d
2δ

!

∥vt∥2 + η

2 ∥vt −∇(f ◦g)δ(xt)∥2 .

Taking expectations on both sides of the inequality, we get the desired result.

B
Convergence Analysis of Algorithm 1

Before giving the analysis of the convergence rate of Algorithm 1, we first present the bound of the
mean-square error term E

∥vt −∇Φδ(xt)∥2 
.
Lemma B.1. Under Assumption 3.1 and 3.3, for Algorithm 1 it holds that

E
h
∥vt −∇Φδ(xt)∥2i
≤
2d2G2
fσ2
0
δ2bg
+
32
√

2πdG2
fG2
g
bf

Proof. Recall that vt =
1
bf
P
j∈[bf ]
d
2δ(F(yt,j; ξt,j) −F(zt,j; ξt,j)), we have

E
h
∥vt −∇Φδ(xt)∥2i

≤2E






vt −1

bf

X

j∈[bf ]

d
2δ (F(g(xt + δwt,j); ξt,j) −F(g(xt −δwt,j); ξt,j))



2



+ 2E







1
bf

X

j∈[bf ]

d
2δ (F(g(xt + δwt,j); ξt,j) −F(g(xt −δwt,j); ξt,j)) −∇(f ◦g)δ(xt)



2



13


---Page Break---
≤2




d2

2δ2bf

X

j∈[bf ]
E
h
∥F(g(xt + δwt,j); ξt,j) −F(gt(xt + δwt,j); ξt,j)∥2i

+
d2

2δ2bf

X

j∈[bf ]
E
h
∥F(g(xt −δwt,j); ξt,j) −F(gt(xt −δwt,j); ξt,j)∥2i




+
32
√

2πdG2
fG2
g
bf

≤
2d2G2
fσ2
0
δ2bg
+
32
√

2πdG2
fG2
g
bf
.

The first inequality is due to ∥a + b∥2 ≤2 ∥a∥2+2 ∥b∥2 for any a, b ∈Rd. The second inequality is
due to Lemma 3.8. The last inequality follows from the Gf-Lipchitzness of f(·) and Assumption 3.3.

We present the formal proof of Theorem 4.1 and Corollary 4.2 below.

B.1
Proof of Theorem 4.1

Proof. If we take η =
δ
cGf Gg
√

d and rearrange the formula in Lemma A.1, we have

η
2E[∥∇(f ◦g)δ(xt)∥2] ≤E[(f ◦g)δ(xt) −(f ◦g)δ(xt+1)] + η

2E[∥vt −∇(f ◦g)δ(xt)∥2].

Sum the above inequality from t = 0 to T −1 and divide both sides by ηT

2 , we have

1
T

T −1
X

t=0
E
h
∥∇(f ◦g)δ(xt)∥2i
≤2cGfGg
√

dE [(f ◦g)δ(x0) −(f ◦g)δ(xT )]

δT

+
32
√

2πdG2
fG2
g
bf
+
2d2G2
fσ2
0
δ2bg
.

In addition, by Lemma 3.7, we have

E [(f ◦g)δ(x0) −(f ◦g)δ(xT )] ≤E[(f ◦g)(x0) −(f ◦g)(xT )] + 2GfGgδ.

Combining the last two inequalities, we get the desired result.

B.2
Proof of Corollary 4.2

Proof. The total zeroth-order oracle calls can be bounded by

O(Tbfbg)

=O

  
GfGg
√

dR
δϵ2
+
G2
fG2
g
√

d
ϵ2

!

·
dG2
fG2
g
ϵ2
·
d2G2
fσ2
0
δ2ϵ2

!

=O

 
d3.5G5
fG3
gσ2
0R
δ3ϵ6
+
d3.5G6
fG4
gσ2
0
δ2ϵ6

!

.

C
Convergence Analysis of Algorithm 2

In this section, we consider the formal proof of the convergence rate of the GFCOM+ method. First,
we introduce the following lemma to bound the mean-square error between the recursive gradient
estimator and the gradient of the surrogate function.

14


---Page Break---
Lemma C.1. Let nt = ⌊t/m⌋m. Under Assumption 3.1 and 3.3, for Algorithm 2 it holds that

E
h
∥vt −∇(f ◦g)δ(xt)∥2i

≤
10d2G2
fG2
gη2

δ2b′
f

t
X

i=nt
E
h
∥vi∥2i
+
10md2G2
fσ2
0
δ2b′
fb′g
+
32
√

2πdG2
fG2
g
bf

+
2cd2G2
fσ2
0
δ2bg
+
2cd2G2
fσ2
0
δ2b′g
.

Proof. By ∥a + b∥2 ≤2 ∥a∥2 + 2 ∥b∥2, we can infer that

E
h
∥vt −∇(f ◦g)δ(xt)∥2i

≤2E
h
∥vt −∇(f ◦gt)δ(xt)∥2i
+ 2E
h
∥∇(f ◦gt)δ(xt) −∇(f ◦g)δ(xt)∥2i
.

To bound the first term of R.H.S., we have

E[∥vt −∇(f ◦gt)δ(xt)∥2]

=E







1
b′
f

X

j∈[b′
f ]

 d

2δ (F(yt,j; ξt,j)−F(zt,j; ξt,j))wt,j−d

2δ (F(yt−1,j; ξt,j)−F(zt−1,j; ξt,j))wt,j



+vt−1 −∇(f ◦gt)δ(xt)∥2i

=E







1
b′
f

X

j∈[b′
f ]

 d

2δ (F(yt,j; ξt,j)−F(zt,j; ξt,j))wt,j−d

2δ (F(yt−1,j; ξt,j)−F(zt−1,j; ξt,j))wt,j



−(∇(f ◦gt)δ(xt) −∇(f ◦gt−1)δ(xt−1))∥2i
+ E
h
∥vt−1 −∇(f ◦gt−1)δ(xt−1)∥2i

≤1

b′2
f

X

j∈[b′
f ]
E

"
d
2δ (F(yt,j; ξt,j)−F(zt,j; ξt,j))wt,j−d

2δ (F(yt−1,j; ξt,j)−F(zt−1,j; ξt,j))wt,j



2#

+ E
h
∥vt−1 −∇(f ◦gt−1)δ(xt−1)∥2i
.

The second equality follows from Lemma 3.8. The last inequality is due to E[∥x −E[x]∥2] ≤
E[∥x∥2]. Observe that

E

"
d
2δ (F(yt,j; ξt,j) −F(zt,j; ξt,j))wt,j −d

2δ (F(yt−1,j; ξt,j) −F(zt−1,j; ξt,j))wt,j



2#

≤5E

d
2δ (F(g(xt + δwt,j); ξt,j) −F(g(xt −δwt,j); ξt,j)−

(F(g(xt−1 + δwt,j); ξt,j) −F(g(xt−1 −δwt,j); ξt,j)))wt,j∥2i

+ 5E

"
d
2δ (F(g(xt + δwt,j); ξt,j) −F(yt,j; ξt,j))wt,j



2#

+ 5E

"
d
2δ (F(g(xt −δwt,j); ξt,j) −F(zt,j; ξt,j))wt,j



2#

+ 5E

"
d
2δ (F(g(xt−1 + δwt,j); ξt,j) −F(yt−1,j; ξt,j))wt,j



2#

+ 5E

"
d
2δ (F(g(xt−1 −δwt,j); ξt,j) −F(zt−1,j; ξt,j))wt,j



2#

15


---Page Break---
≤
5d2G2
fG2
g
δ2
E
h
∥xt −xt−1∥2i
+
5d2G2
fσ2
0
δ2b′g

=
5d2G2
fG2
gη2

δ2
E
h
∥vt−1∥2i
+
5d2G2
fσ2
0
δ2b′g
.

The first inequality is due to ∥a1 + · · · + an∥2 ≤n ∥a1∥2 + · · · + n ∥an∥2. The second inequality is
due to the Lipschitzness of both inner and outer functions with Assumption 3.3. Consequently, one
has

E
h
∥vt −∇(f ◦gt)δ(xt)∥2i

≤
5d2G2
fG2
gη2

δ2b′
f
E
h
∥vt−1∥2i
+
5d2G2
fσ2
0
δ2b′
fb′g
+ E
h
∥vt−1 −∇(f ◦gt−1)δ(xt−1)∥2i

≤
5d2G2
fG2
gη2

δ2b′
f

t
X

i=nt
E
h
∥vi∥2i
+
5md2G2
fσ2
0
δ2b′
fb′g
+
16
√

2πdG2
fG2
g
bf
.

The last inequality follows from Lemma 3.8. In addition, for t mod m = 0 we can bound

E
h
∥∇(f ◦gt)δ(xt) −∇(f ◦g)δ(xt)∥2i

=E

"Eu

d

δ ((f ◦gt)(xt + δu) −(f ◦g)(xt + δu)) u


2#

≤d2

δ2 Eu
h
∥(f ◦gt)(xt + δu) −(f ◦g)(xt + δu)∥2 ∥u∥2i

≤
cd2G2
fσ2
0
δ2bg
.

The last inequality follows from the Lipschitzness of f and Assumption 3.3. Similarly, for t mod m ̸=
0 we can bound

E
h
∥∇(f ◦gt)δ(xt) −∇(f ◦g)δ(xt)∥2i
≤
cd2G2
fσ2
0
δ2b′g
.

Putting everything together, we get the desired bound.

We present the formal proof of Theorem 4.3 and Corollary 4.4 below.

C.1
Proof of Theorem 4.3

Proof. Rearrange the terms in Lemma A.1, sum t from 0 to T −1, and divide both sides by ηT

2 ,

1
T

T −1
X

t=0
E
h
∥∇(f ◦g)δ(xt)∥2i

≤2E [(f ◦g)δ(x0) −2(f ◦g)δ(xT )]

ηT
−1

T

 

1 −cηGfGg
√

d
δ

! T −1
X

i=0
E
h
∥vi∥2i

+ 1

T

T −1
X

t=0
E
h
∥vt −∇(f ◦g)δ(xt)∥2i

≤2E [(f ◦g)δ(x0) −2(f ◦g)δ(xT )]

ηT
−1

T

 

1 −cηGfGg
√

d
δ
−
10md2G2
fG2
gη2

δ2b′
f

! T −1
X

i=0
E
h
∥vi∥2i

+
10md2G2
fσ2
0
δ2b′
fb′g
+
32
√

2πdG2
fG2
g
bf
+
2cd2G2
fσ2
0
δ2bg
+
2cd2G2
fσ2
0
δ2b′g
.

16


---Page Break---
The last inequality follows from Lemma C.1. If we choose hyperparameters as follows

η =
δ
2cGfGg
√

d
,
b′
f = Θ
dGfGg

ϵ


,
m = Θ
GfGg

ϵ


,

Then we can deduce that 1 −cηGf Gg
√

d
δ
−
10md2G2
f G2
gη2

δ2b′
f
≤0. By Lemma 3.7, we have

E [(f ◦g)δ(x0) −(f ◦g)δ(xT )] ≤E[(f ◦g)(x0) −(f ◦g)(xT )] + 2GfGgδ.

Therefore, we obtain the following result

E
h
∥∇Φδ(xR)∥2i
= O

 √

dGfGgR

δT
+

√

dG2
fG2
g
T
+
dG2
fG2
g
bf
+
d2G2
fσ2
0
δ2bg
+
d2G2
fσ2
0
δ2b′g

!

.

C.2
Proof of Corollary 4.4

Proof. The total zeroth-order oracle calls can be bounded by

O(Tb′
fb′
g + Tbfbg/m)

=O

  √

dGfGgR

δϵ2
+

√

dG2
fG2
g
ϵ2

!

· dGfGg
ϵ
·
d2G2
fσ2
0
δ2ϵ2

!

=O

 
d3.5G4
fG2
gσ2
0R
δ3ϵ5
+
d3.5G5
fG3
gσ2
0
δ2ϵ5

!

.

D
Extensions to Convex Nonsmooth Functions

In this section, we present the formal proof of theorems presented in Section 5.

D.1
Proof of Theorem 5.2

Proof. Since G(·; ζ) is convex function, gt(·) is also convex. Using the result of Section 3.2.4 of [38]
and Assumption 5.1, we can deduce that f ◦gt is a convex function. Let x∗= arg minx(f ◦g)δ(x),
then we have

E[(f ◦gt)δ(xt) −(f ◦gt)δ(x∗)]
≤E[⟨∇(f ◦gt)δ(xt), xt −x∗⟩]

=E
 d

2δ (F(gt(xt + δu), ξt) −F(gt(xt −δu), ξt)), xt −x∗


≤1

η0
E [⟨xt −xt+1, xt −x∗⟩]

= 1

2η0
E
h
∥xt −x∗∥2 −∥xt+1 −x∗∥2 + ∥xt −xt+1∥2i

≤1

2η0
E
h
∥xt −x∗∥2 −∥xt+1 −x∗∥2i
+ 8
√

2πdG2
fG2
gη0.

The first inequality follows from the convexity of f ◦gt. The last inequality is due to Lemma 3.8.
Observe that for ∀x ∈Rd,

|(f ◦g)δ(x) −(f ◦gt)δ(x)|
=|Eu[(f ◦g)(x + δu)] −Eu[(f ◦gt)(x + δu)]|
=|Eu[(f ◦g)(x + δu) −(f ◦gt)(x + δu)]|
≤Eu[|(f ◦g)(x + δu) −(f ◦gt)(x + δu)|]

17


---Page Break---
≤c1Gfσ0
p

bg,0
,

where c1 is some constant. The second equality is due to the linearity of expectations. The last
inequality follows from Assumption 3.3 and Lipschitzness of f. Consequently, we have

E[(f ◦g)δ(x1) −(f ◦g)δ(x∗)]

≤
ˆR2

2η0T0
+ 8
√

2πdG2
fG2
gη0 + 2c1Gfσ0
p

bg,0

≤12 ˆRGfGg
√

d
√T0
+ 2c1Gfσ0
p

bg,0
.

By Lemma 3.7, we have

E [(f ◦g)δ(x1) −(f ◦g)δ(x∗)] ≤E[(f ◦g)(x1) −(f ◦g)(x∗)] + 2GfGgδ.

Consequently, one has

E[(f ◦g)δ(x1) −(f ◦g)δ(x∗)]

≤12 ˆRGfGg
√

d
√T0
+ 2c1Gfσ0
p

bg,0
+ 2GfGgδ.

To obtain E[(f ◦g)(x1) −(f ◦g)(x∗)] ≤ρ, we choose

η0 =
ˆR
GfGg
√dT0
,
T0 = Θ

 ˆR2G2
fG2
gd
ρ2

!

,
bg,0 = Θ

 
G2
fσ2
0
ρ2

!

,
δ = Θ

ρ
GfGg


.

D.2
Proof of Corollary 5.3

Proof. The total stochastic function oracle calls can be bounded by

O (Tbfbg + T0bg,0)

=O

 
GfGg
√

d(ρ + GfGgδ)

δϵ2
·
dG2
fG2
g
ϵ2
·
d2G2
fσ2
0
δ2ϵ2
+
d ˆR2G4
fG2
gσ2
0
ρ4

!

=O

 
d3.5G5
fG3
gσ2
0ρ
δ3ϵ6
+
d3.5G6
fG4
gσ2
0
δ2ϵ6
+
d ˆR2G4
fG2
gσ2
0
ρ4

!

=O

 
d3 ˆR0.4G4.8
f G2.8
g σ2
0
δ2.4ϵ4.8
+
d3.5G6
fG4
gσ2
0
δ2ϵ6

!

.

D.3
Proof of Corollary 5.4

Proof. The total stochastic function oracle calls can be bounded by

O
 
Tb′
fb′
g + Tbfbg/m + T0bg,0


=O

 
GfGg
√

d(ρ + GfGgδ)

δϵ2
· dGfGg
ϵ
·
d2σ2
0G2
f
δ2ϵ2
+
d ˆR2G4
fG2
gσ2
0
ρ4

!

=O

 
d3.5G4
fG2
gσ2
0ρ
δ3ϵ5
+
d3.5G5
fG3
gσ2
0
δ2ϵ5
+
d ˆR2G4
fG2
gσ2
0
ρ4

!

=O

 
d3 ˆR0.4G4
fG2
gσ2
0
δ2.4ϵ4
+
d3.5G5
fG3
gσ2
0
δ2ϵ5

!

.

18


---Page Break---
NeurIPS Paper Checklist

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

19


---Page Break---
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
Answer: [Yes]
Justification:
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

20


---Page Break---
Answer: [No]

Justification: We are clearing the code with internal compliance and will release it upon
approval.

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

Justification:

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

Justification:

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

21


---Page Break---
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

Justification:

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

Answer: [NA]

Justification: There are no potential positive or negative societal impacts of this work.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

22


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

Justification:

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

Justification:

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

23


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
Justification: The paper does not involve crowdsourcing or research with human subjects.
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

24


---Page Break---
