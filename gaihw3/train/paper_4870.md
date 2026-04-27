A Globally Optimal Portfolio for m-Sparse Sharpe
Ratio Maximization

Yizun Lin1
Zhao-Rong Lai1∗
Cheng Li1

1Department of Mathematics
College of Information Science and Technology
Jinan University, Guangzhou, China
{linyizun,laizhr}@jnu.edu.cn
licheng@stu2020.jnu.edu.cn

Abstract

The Sharpe ratio is an important and widely-used risk-adjusted return in ﬁnancial
engineering. In modern portfolio management, one may require an m-sparse
(no more than m active assets) portfolio to save managerial and ﬁnancial costs.
However, few existing methods can optimize the Sharpe ratio with the m-sparse
constraint, due to the nonconvexity and the complexity of this constraint. We
propose to convert the m-sparse fractional optimization problem into an equivalent
m-sparse quadratic programming problem. The semi-algebraic property of the
resulting objective function allows us to exploit the Kurdyka-Łojasiewicz property
to develop an efﬁcient Proximal Gradient Algorithm (PGA) that leads to a portfolio
which achieves the globally optimal m-sparse Sharpe ratio under certain conditions.
The convergence rates of PGA are also provided. To the best of our knowledge,
this is the ﬁrst proposal that achieves a globally optimal m-sparse Sharpe ratio with
a theoretically-sound guarantee.

1
Introduction

The Sharpe ratio (SR) [33] is an important and widely-used performance metric in ﬁnance. Suppose
an investing strategy is represented by a portfolio w ∈RN of N assets from a ﬁnancial market.
µ ∈RN and Σ ∈RN×N denote the expected return vector (in excess of the risk-free rate) and its
covariance matrix for the N assets, respectively. It can be seen that w⊤µ and
√

w⊤Σw represent
the expected return and its standard deviation (i.e., risk) for the portfolio w. The original deﬁnition
of SR is given as the follow quotient between return and risk:

S0(w) =
w⊤µ
√

w⊤Σw
.
(1.1)

Ever since the proposal of SR, how to maximize it becomes an attractive research topic. Ordinary
portfolio optimization methods based on either the mean-variance approach [5, 10] or the exponential
growth rate approach [22, 24] can reduce the portfolio risk and increase the portfolio return to some
extent [23], and hence improve the SR. On the other hand, direct SR optimization methods are
also proposed. Hung et al. [18], Yu and Xu [35] consider the SR as a differentiable function of
the portfolio, which can be solved via the augmented Lagrangian method. Pang [29] converts the
SR maximization under the self-ﬁnancing and long-only constraints into a linear complementarity
problem, which can be solved via the Parametric Linear Complementarity Technique (PLCT) and the
principle pivoting algorithm [12]. Note that PLCT requires µi > 0 for at least some asset i in order
to be feasible.

∗Correspondence to: Zhao-Rong Lai

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
In modern portfolio management, it is widely-recognized that the number of selected assets should
be restricted to a manageable size, in order to keep simplicity and save time and ﬁnancial costs.
Managerial strategies provide an approach to achieve this objective, such as the revenue driven
resource allocation [11], the endowment model [15], and selling stocks after market crashes [27].
However, the managerial approaches still require intensive administration and abundant experience
in management and ﬁnance. Hence researchers turn to sparsity models for solutions via the com-
putational approaches. In [10], a Sparse and Stable Markowitz Portfolio (SSMP) is proposed by
imposing ℓ1-regularization on the portfolio. Ao et al. [2] further propose a mean-variance model with
an ℓ1 constraint based on a maximum-Sharpe-ratio estimation. In [24], the exponential growth rate
(EGR) criterion [1, 13, 19, 23] is exploited to develop a Short-term Sparse Portfolio Optimization
(SSPO). Furthermore, a Short-term Sparse Portfolio Optimization with ℓ0-regularization (SSPO-ℓ0)
is developed in [28]. In [25], a nonlinear shrinkage of the covariance matrix is proposed to obtain an
appropriate size of free parameters. Motivated by this strategy, Lai et al. [22] characterize a sparse
structure for covariance estimation to construct a portfolio via the machine learning approach.

The ℓ1-regularization and the ℓ1 constraint cannot control the exact number of selected assets. One
has to tune the sparsity parameter to roughly adjust this number. On the other hand, suppose we
want to select no more than m active assets out of N assets to construct a portfolio, then this can be
exactly represented by the m-sparse (or ℓ0) constraint ∥w∥0 ⩽m, where the ℓ0 norm ∥· ∥0 denotes
the number of nonzero components of a vector. Although many sparsity models are established for
the Markowitz portfolio, few existing methods can optimize the SR (1.1) with the ℓ0 constraint, due
to the nonconvexity and the complexity of this constraint. In addition to the ℓ0 constraint, other
realistic constraints should also be imposed to ensure feasibility. For example, the self-ﬁnancing
constraint represents full re-investment and no external loans; the long-only constraint represents
no short position. If all these constraints are imposed, the whole model becomes even much more
difﬁcult to solve.

To overcome these difﬁculties, we observe that this optimization is essentially an m-sparse fractional
optimization that can be transformed into an equivalent m-sparse quadratic programming. Then
the resulting objective function is semi-algebraic, so that the Kurdyka-Łojasiewicz (KL) property
[3] can be exploited to develop an efﬁcient Proximal Gradient Algorithm (PGA) [30]. It converges
to a portfolio which achieves the globally optimal m-sparse SR under certain conditions. To the
best of our knowledge, this is the ﬁrst proposal that achieves a globally optimal m-sparse SR with a
theoretically-sound guarantee. Our main contributions can be summarized as follows.

1) We propose to directly maximize the SR with the ℓ0 constraint, the self-ﬁnancing constraint
and the long-only constraint on the portfolio. This model aims to obtain a feasible and realistic
portfolio that optimizes the SR with exact sparsity.
2) SR maximization is essentially a fractional optimization. We convert this m-sparse fractional
optimization problem into an equivalent m-sparse quadratic programming problem, which reduces
the difﬁculty of solving it.
3) We observe that the resulting objective function is semi-algebraic, thus exploit the KL property
to develop an efﬁcient PGA that leads to a globally or at least a locally optimal solution of the
m-sparse SR maximization model. The convergence rates of PGA are also provided.

Besides the above contributions, our approach also has several advantages: (i) It can be extended to a
wide range of optimization problems with semi-algebraic objective functions and constraints. (ii)
The actual sparsity is robust to the choice of m. (iii) It needs very little parameter tuning. (iv) It
does not require any external algorithms or commercial optimizers.

2
Related Works and Existing Problems

There are some existing works that indirectly or directly optimize the SR to some extent via the
computational approach. We introduce some examples and then analyze some unsolved problems.

2.1
Ordinary Portfolio Optimization

An intuitive approach is to directly optimize the portfolio, so that the expected return is maximized
and/or the risk is minimized. These methods can be categorized into the mean-variance approach
and the exponential growth rate approach [23]. Let R ∈RT ×N be the sample asset return matrix
with T trading times and N assets, and 1n denotes the vector of n ones. Brodie et al. [10] propose

2


---Page Break---
to impose ℓ1-regularization on the mean-variance model, forming a Sparse and Stable Markowitz
Portfolio (SSMP):

ˆw = argmin
w∈RN

 1

T ∥Rw −ρ1T ∥2
2 + τ∥w∥1


,
s.t.
w⊤ˆµ = ρ, w⊤1N = 1,
(2.1)

where ˆµ := 1

T R⊤1T denotes the column vector of sample mean returns, and τ ⩾0 is the regular-
ization parameter. ∥· ∥2 and ∥· ∥1 denote the ℓ2-norm and the ℓ1-norm, respectively. The quadratic
form 1

T ∥Rw −ρ1T ∥2
2 actually computes the mean squared error between the sample portfolio return
r(t)w (r(t) is the t-th row of R) and the given return level ρ. Equations w⊤ˆµ = ρ and w⊤1N = 1
are the expected return constraint and the self-ﬁnancing constraint, respectively. Model (2.1) can be
approximately solved by a surrogate model [10] and the Least Absolute Shrinkage and Selection
Operator (Lasso) [34]. Goto and Xu [17] also exploit the Lasso to solve a mean-variance model
through sparse hedging restrictions.

Ao et al. [2] propose a maximum-Sharpe-ratio estimated and sparse regression (MAXER) to approach
mean-variance efﬁciency. Assume there are sufﬁcient observations T > (N + 2). MAXER ﬁrst
computes the maximum-Sharpe-ratio estimated regression response ˆrc as follows:

ˆθs = ˆµ⊤ˆΣ−1 ˆµ, ˆθ := (T −N −2)ˆθs −N

T
, ˆrc := σ 1 + ˆθ
p

ˆθ
,
(2.2)

where ˆµ and ˆΣ denote the sample mean and the sample covariance, respectively, σ is the risk
constraint parameter. Then it adopts the Lasso to obtain the portfolio:

ˆw = argmin
w∈RN
1
T ∥Rw −ˆrc1T ∥2
2,
s.t.
∥w∥1 ⩽τ.
(2.3)

Instead of the mean-variance approach, our method takes an essentially different objective that
directly maximizes the SR in (1.1). Besides, our method does not require T > (N + 2).

Based on the exponential growth rate criterion [1, 13, 19, 23], Lai et al. [24] propose to minimize a
kind of negative potential return w⊤ϕ with ℓ1-regularization but without any risk term, forming a
Short-term Sparse Portfolio Optimization (SSPO) model

ˆw = argmin
w∈RN


w⊤ϕ + τ∥w∥1
	
,
s.t.
w⊤1N = 1.
(2.4)

It develops an unconstrained augmented Lagrangian with the existence of a saddle point that can be
solved by the alternating direction method of multipliers (ADMM). Luo et al. [28] further propose
the SSPO-ℓ0 model

ˆw = argmin
w∈∆


w⊤ϕ + τ∥w∥0
	
,
∆:=

w ∈RN w ⩾0N and w⊤1N = 1
	
,
(2.5)

where ∆is the N-dimensional simplex. This simplex constraint w ∈∆is the combination of the
long-only and the self-ﬁnancing constraints. Under this constraint, the ℓ0-regularization problem
(2.5) has a closed-form solution ˜Imin
ϕ
:=
n
i ∈NN
 ϕi ⩽minj∈NN ϕj + ϵ
o
with a tolerance ϵ ⩾0,

where NN := {1, 2, . . . , N}.

On the other hand, Lai et al. [22] propose a rank-one covariance estimator based on the operator
space decomposition, in order to capture the rapidly-changing risk structure in the ﬁnancial market:

X = VJΞU ⊤
J , D = Ξ2 −1

T ΞV ⊤
J 1T 1⊤
T VJΞ, ζ∗
1 =

tr(D)
N(T −1)

−1

2
θ1, ˆΣRO := u1ζ∗
1u⊤
1 ,

where X = R + 1T ×N denotes the price relative matrix and VJΞU ⊤
J is its singular value decompo-
sition (SVD), θ1 and u1 are the largest eigenvalue and its eigenvector, respectively.

Although the above portfolio optimization methods may partly improve the SR, they may not
be competitive to direct SR optimization. Hence direct SR optimization methods should still be
developed and investigated.

3


---Page Break---
2.2
Sharpe Ratio Optimization

Pang [29] proposes to optimize the following SR model:

max
w∈RN S0(w),
s.t.
w ∈∆, Cw ⩽d,
(2.6)

where S0(w) is deﬁned by (1.1), C ∈Rl×N and d ∈Rl form a linear constraint for w. It can be
transformed into the following equivalent parametric linear complementarity problem:





u = −µ + Σw + (C⊤−1Nd⊤)y ⩾0N,
w ⩾0N,
v = −(C −d1⊤
N)w ⩾0l,
y ⩾0l,
u⊤w = v⊤y = 0.
(2.7)

Problem (2.7) can be efﬁciently solved by the principle pivoting algorithm [12], but it requires µi > 0
for at least some asset i in order to be feasible. Moreover, if we aim to construct an m-sparse w,
this approach becomes invalid. In Section 3, we will convert the m-sparse SR optimization into an
equivalent m-sparse quadratic programming, and the latter is still a nonconvex optimization. We
further elaborate a proximal gradient algorithm to obtain a globally or locally optimal SR.

Another viable approach is to consider the SR as a function of the portfolio and directly optimize it
under some realistic constraints. Hung et al. [18] propose the following IPSRM-D model to optimize
the SR:

max
w∈∆


S (w) := w⊤µ + κ1w⊤Uw

w⊤Dw
+ κ2w⊤(1N −w)

,
(2.8)

where U ∈RN×N and D ∈RN×N are upside and downside risk matrices, respectively, w⊤(1N −
w) is a diversiﬁcation term. κ1 and κ2 are hyperparameters that control the strength of upside risk and
diversiﬁcation, respectively. Interested readers can further refer to [35] for some practical estimators
for µ, U and D.

However, S (w) in model (2.8) is different from the original SR S0(w) (1.1) in several signiﬁcant
parts. First, S (w) uses second-order moments w⊤Uw and w⊤Dw as risk metrics, but S0(w)
uses the ﬁrst-order moment
√

w⊤Σw instead. In general, a ﬁrst-order moment is more appropriate
because the expected return w⊤µ should remain in the same order of magnitude as
√

w⊤Σw.
Second, the numerator of S0(w) does not contain any risk term, while the numerator of S (w)
contains w⊤Uw. This may change the meaning of SR as an equilibrium point in the efﬁcient frontier
based on the CAPM theory [32]. These facts may affect the performance of SR optimization.

Another problem is the lack of effective solving algorithms that could really maximize the SR under
constraints. A conventional way is to adopt gradient methods, since S (w) is a differentiable function
when w ̸= 0N. Hung et al. [18], Yu and Xu [35] propose to adopt the augmented Lagrangian method
to optimize (2.8). Though they do not specify which form of Lagrangian models is used, we give the
following one without loss of generality:

L (w, λ) := S (w) + ϱ

2(w⊤1N −1)2 + λ⊤w,
(2.9)

where ϱ

2(w⊤1N −1)2 with hyperparameter ϱ ⩽0 is a regularization term for the self-ﬁnancing
constraint, and λ ∈RN
+ is the dual variable with respect to (w.r.t.) w for the long-only constraint
w ⩾0N. RN
+ denotes the set of all N-dimensional nonnegative vectors. The update scheme is
w(k+1) = w(k) + η1∇wL (w(k), λ(k)),
λ(k+1) = λ(k) −η2∇λL (w(k+1), λ(k)),
(2.10)

where η1, η2 ⩾0 are update step sizes. Note that S is a nonconvex function w.r.t. w, and the
augmented Lagrangian method is a surrogate method that approximates model (2.8). Hence (2.10)
does not necessarily lead to the maximum SR. Worse still, due to the augmented term ϱ

2(w⊤1N −1)2,
(2.10) may not even decrease the objective function S . Moreover, the Lagrangian L (w(k), λ(k))
is increased by the w(k) updates but decreased by the λ(k) updates, hence (2.10) cannot guarantee
convergence to a point (w∗, λ∗) without a thorough investigation of the update scheme. To summarize,
the augmented Lagrangian method and most existing gradient methods cannot guarantee global or
local optimality for model (2.8).

4


---Page Break---
3
PGA for m-Sparse Sharpe Ratio Maximization

In this section, we formulate the maximization problem of SR as a nonconvex fractional optimization
under constraints. Instead of directly solving the proposed model, we develop an efﬁcient proximal
gradient algorithm to solve a simpler surrogate model (subtraction form) that is equivalent to the
original constrained fractional optimization model.

3.1
m-Sparse Sharpe Ratio Maximization Model

In order to retain the risk premium meaning of SR in ﬁnance, we directly maximize the original SR
in (1.1) instead of a variant like (2.8). In the perspective of statistical estimation, suppose we have a
sample asset return (in excess of the risk-free rate) matrix R ∈RT ×N with T trading times and N
assets. Then the original SR (1.1) can be represented by

S(w) :=

1
T 1⊤
T Rw
q

1
T −1∥Rw −( 1

T 1⊤
T Rw)1T ∥2
2 + ϵ∥w∥2
2
,
(3.1)

where ϵ∥w∥2
2 is a regularization term for a positive deﬁnite Qϵ deﬁned in (3.2). The parameter ϵ
can be an arbitrarily-small positive parameter, whose effect on the risk term can be negligible. To
simplify the notation, we deﬁne

p := 1

T R⊤1T , Q :=
1
√

T −1


R −1

T 1T ×T R

and Qϵ := Q⊤Q + ϵI.
(3.2)

Then the maximization of SR under the m-sparse, long-only and self-ﬁnancing constraints is given
by

max
w∈∆
∥w∥0⩽m
S(w) :=
p⊤w
p

w⊤Qϵw
,
(3.3)

where the simplex ∆is deﬁned in (2.5). Note that minimizing −S(w) under the constraint ∥w∥0 ⩽m
is essentially quite different from minimizing the ℓ0-regularization version −S(w) + τ∥w∥0 with
some positive τ. In general, the latter is easier because it incorporates the ℓ0 norm into the objective
function and enlarges the feasible set by dropping the constraint ∥w∥0 ⩽m. We simply call (3.3)
the m-Sparse Sharpe Ratio Maximization (mSSRM) model. In fact, to solve the mSSRM model, it
sufﬁces to solve the following simpler constrained minimization model

min
v⩾0N
∥v∥0⩽m

1

2v⊤Qϵv −p⊤v

.
(3.4)

To see this, we establish the relation between the solutions of these two models in the following
theorem, whose proof is provided in Appendix A.1. We deﬁne the constraint sets in model (3.3) and
(3.4) by

Ω1 := {w ∈∆| ∥w∥0 ⩽m} and Ω:= {v ∈RN| v ⩾0N and ∥v∥0 ⩽m},
(3.5)

respectively. It is obvious that Ω1 ⫋Ω.

Theorem 1 Suppose that there exists some ˜w ∈Ω1 such that p⊤˜w > 0. If ˆv is an optimal solution
of model (3.3), then
p⊤ˆv
ˆv⊤Qϵ ˆv ˆv is an optimal solution of model (3.4). Conversely, if ˆv is an optimal
solution of model (3.4), then
ˆv
ˆv⊤1N is an optimal solution of model (3.3).

Deﬁning the indicator function ιΩby

ιΩ(v) :=
0,
if v ∈Ω;
+∞,
otherwise,
(3.6)

we can rewrite model (3.4) as the following two-term unconstrained minimization model:

min
v∈RN {f(v) + ιΩ(v)} , where f(v) := 1

2v⊤Qϵv −p⊤v.
(3.7)

We then turn to solving model (3.7) instead of the mSSRM model in (3.3).

5


---Page Break---
3.2
Proximal Gradient Algorithm

To develop a proximal gradient algorithm for solving model (3.7), we recall the notion of proximity
operator, and then establish the relation between the solution of model (3.7) and the proximity
characterization (3.8) in Theorem 2. For a proper function ψ : Rn →R, its proximity operator
at x ∈Rn is deﬁned by proxψ(x) := argminu∈Rn
 1

2∥u −x∥2
2 + ψ(u)
	
. We remark that for
function ψ that is nonconvex, proxψ(x) may not be unique. Throughout this paper, the formula
h = proxψ(x) represents h ∈proxψ(x). For v∗∈RN and δ > 0, we denote by B(v∗; δ) the
neighborhood of v∗with radius δ. If there exists some δ > 0 such that f(v∗) ⩽f(v) holds for all
v ∈Ω∩B(v∗; δ), then we say that v∗is a locally optimal solution of model (3.7).

Theorem 2 Let function ιΩand f be deﬁned by (3.6) and (3.7), respectively. If v∗is a globally
optimal solution of model (3.7), then for any α ∈

0,
1
∥Qϵ∥2

i
,

v∗= proxιΩ(v∗−α∇f(v∗)) .
(3.8)

Conversely, we have the following two statements:

(i) If α ⩾1

ϵ and (3.8) holds, then v∗is a globally optimal solution of model (3.7).
(ii) For any α > 0, if (3.8) holds, then v∗is a locally optimal solution of model (3.7).

The proof of Theorem 2 is provided in Appendix A.2. Based on this theorem, the Proximal Gradient
Algorithm (PGA) for solving model (3.7) can be given by the following iterative scheme:

v(k+1) = proxιΩ

v(k) −α∇f(v(k))

, where k ∈N, α > 0, v(0) ∈Ω.
(3.9)

We then compute the closed form of proxιΩ. For a vector v ∈RN, we denote by mv and Jv
pos
the number of positive components and the index set of positive components of v. If mv ⩾m,
then we denote by Jv
m-pos an index set of the m-largest positive components of v. Speciﬁcally,
by letting {vji}i∈NN be an rearrangement of {vj}j∈NN such that vj1 ⩾vj2 ⩾· · · ⩾vjN , then
Jv
m-pos := {j1, j2, . . . , jm}. Throughout this paper, for a given vector v ∈RN, we shall always
compute proxιΩ(v) according to the following proposition. Its proof is given in Appendix A.3.

Proposition 3 Let ιΩbe deﬁned by (3.6), v ∈RN, and deﬁne the index set Jv by

Jv =
Jv
m-pos,
if mv > m;
Jv
pos,
if mv ⩽m.

Then the vector h given by hj =
vj,
if j ∈Jv;
0,
if j ∈NN\Jv satisﬁes that h ∈proxιΩ(v).

3.3
Convergence Analysis of PGA

In this subsection, we delve into the convergence analysis of PGA. We aim to demonstrate that PGA
possesses the capability to converge to a globally optimal solution of model (3.4). The limit point
obtained by PGA can also yield a globally optimal solution of the original model (3.3), under certain
conditions. We also demonstrate the convergence rates of PGA.

Firstly, we introduce a proposition that illustrates the convergence and monotonic decreasing behavior
of the objective function values for the iterative sequence, as well as the vanishing gap between
consecutive iterates. The proof of this proposition is provided in Appendix A.4.

Proposition 4 Let function ιΩand f be deﬁned by (3.6) and (3.7), respectively, and let F := f + ιΩ.
If α ∈

0,
1
∥Qϵ∥2


, then for arbitrary initial vector v(0) ∈RN, the sequence {v(k)}k∈N generated
by PGA satisﬁes the following properties:

(i) v(k) ∈Ω, for all k ∈N;
(ii) F(v(k+1))+a∥v(k+1)−v(k)∥2
2 ⩽F(v(k)) for all k ∈N, where a := 1

2
  1

α −∥Qϵ∥2

> 0;
(iii) limk→∞F(v(k)) exists;

(iv) limk→∞∥v(k+1) −v(k)∥2 = 0.

6


---Page Break---
Though we have established the convergence of {F(v(k))}k∈N and the vanishing gap between
consecutive iterates, further efforts are necessary to rigorously conﬁrm the convergence of the iterative
sequence {v(k)}k∈N. We demonstrate the convergence of {v(k)}k∈N to a local minimizer of function
F and the corresponding convergence rates in the following theorem. In order to maintain consistency
with the original SR maximization model (as outlined in Theorem 1), we further deﬁne sequence
{w(k)}k∈N based on {v(k)}k∈N and conduct an analysis of the convergence rate of {w(k)}k∈N. To
prove the theorem, we need to introduce the notions of subdifferential, semi-algebraic function and
Kurdyka-Łojasiewicz property, along with several technical lemmas. Detailed proofs and relevant
content can be found in Appendix A.5.

Theorem 5 Suppose that there exists some ˜w ∈Ωsuch that p⊤˜w > 0. For arbitrary initial vector
v(0) ∈RN, let {v(k)}k∈N be generated by PGA, and let {w(k)}k∈N be deﬁned by

w(k) :=

(
v(k)

(v(k))⊤1N ,
if (v(k))⊤1N ̸= 0;
0N,
otherwise.

If α ∈

0,
1
∥Qϵ∥2


, then the following statements hold:

(i) {v(k)}k∈N converge to a locally optimal solution v∗of model (3.4) with convergence rates
∥v(k) −v∗∥2 = O(1/
√

k) and |f(v(k)) −f(v∗)| = O(1/k).
(ii) The limit point v∗of {v(k)}k∈N satisﬁes that v∗⩾0N and v∗̸= 0N.
(iii) {w(k)}k∈N converge to w∗:=
v∗

(v∗)⊤1N with convergence rates ∥w(k) −w∗∥2 = O(1/
√

k)

and |S(w(k)) −S(w∗)| = O(1/
√

k), where S is deﬁned in (3.3).

In the remainder of this section, we always let v∗∈Ωbe the locally optimal solution of model (3.7)
that sequence {v(k)}k∈N converges to. We recall that mv∗and Jv∗
pos denote the number of positive
components and the index set of positive components of v∗, respectively. Suppose that there exists
some ˜w ∈Ωsuch that p⊤˜w > 0. Then item (ii) in Theorem 5 together with v∗∈Ωyields that
1 ⩽mv∗⩽m. In fact, v∗is also the globally optimal solution of the convex model

min
v∈ˆΩ

1

2v⊤Qϵv−p⊤v

, where ˆΩ:={v ∈RN| v⩾0N and vj=0 for all j ∈NN\Jv∗
pos}.
(3.10)

Certainly, ˆΩ̸= ∅due to the condition mv∗⩾1. According to the deﬁnition of ˆΩ, it is straightforward
to observe that v∗∈ˆΩ. Furthermore, ˆΩis a closed convex set and ˆΩ⊂Ω. To analyze the relation
between v∗and the original m-sparse Sharpe ratio maximization model (3.3), we deﬁne

ˆΩ1 := {v ∈∆| vj = 0 for all j ∈NN\Jv∗
pos},
(3.11)

where ∆is given by (2.5). It is easy to see that ˆΩ1 ⊂Ω1, where Ω1 deﬁned in (3.5) is the constraint
set of model (3.3). We then have the following theorem, whose proof is provided in Appendix A.6.

Theorem 6 Suppose that there exists some ˜w ∈ˆΩsuch that p⊤˜w > 0, where ˆΩis deﬁned in (3.10),
and let w∗:=
v∗

(v∗)⊤1N . Then the following statements hold:

(i) v∗is the unique globally optimal solution of model (3.10).
(ii) w∗is a globally optimal solution of model max
w∈ˆΩ1
S(w).

(iii) If mv∗= m, then w∗is a locally optimal solution of model (3.3).

Item (iii) in Theorem 6 demonstrates that the limit point of the sequence obtained by PGA can yield
a locally optimal solution of model (3.3). In fact, according to item (i) in Theorem 2, we have the
following theorem that provides sufﬁcient conditions for obtaining a globally optimal solution of
model (3.3), whose proof is provided in Appendix A.7.

Theorem 7 Suppose that there exists some ˜w ∈Ω1 such that p⊤˜w > 0, and let w∗:=
v∗

(v∗)⊤1N . If
one of the following two conditions holds:

(i) mv∗< m;

7


---Page Break---
(ii) mv∗= m and ∇if(v∗) > −ϵ · min{v∗
i |i ∈supp(v∗)} for all i ∈NN\supp(v∗),

then w∗is a globally optimal solution of model (3.3).

Combining Theorem 7 and item (iii) in Theorem 6, we see that the proposed method can obtain
a globally optimal solution of model (3.3) when mv∗< m. Even if this condition does not hold,
we can obtain at least a locally optimal solution. To test validation of PGA’s global optimality, we
conduct a set of simulation experiments, whose details are presented in Appendix A.8. The codes
for the simulation experiments are accessible via the link: https://github.com/linyizun2024/
mSSRM/tree/main/Codes_for_Simulation.

We call the existence of ˜w ∈ˆΩsuch that p⊤˜w > 0 in Theorem 6 the Existence of Positive Expected
Return (EPER) condition. Although the EPER condition is required to guarantee the convergence
of w∗to a locally optimal solution of the original model (3.3), the proposed method is still of high
practical signiﬁcance in the case that the EPER condition does not hold. From the proofs of item
(i) in Theorem 5 and item (i) in Theorem 6, we see that even if the EPER condition does not hold,
the sequence generated by PGA still converges to a locally optimal solution v∗of model (3.4),
which is also the globally optimal solution of model (3.10). In these two models, the objective
function 1

2v⊤Qϵv −p⊤v (subtraction form) w.r.t. v represents risk minus expected return, whose
minimization gives smaller risk and less loss in revenue, even if the expected return is not positive.
For the case that the expected return is not positive, compared with the failure of Sharpe ratio of
fractional form, the globally or locally optimal solution of the subtraction form seems to have more
realistic signiﬁcance. We recall from item (ii) in Theorem 5 that v∗may be equal to 0N if the EPER
condition does not hold. In this case, we shall set w∗= 0N and keep all the wealth in the risk-free
asset to avoid loss in revenue. To close this section, we summarize the whole m-sparse Sharpe ratio
maximization method, which we abbreviate to mSSRM-PGA, in Appendix A.9.

4
Experimental results

Extensive experiments with real-world ﬁnancial data sets are conducted to evaluate the performance
of the proposed mSSRM-PGA. Moreover, we also consider one baseline method: 1/N [14], as well as
9 state-of-the-art methods: IPSRM-D [18], PLCT [29], SSMP [10], MAXER [2], SSPO [24], SPOLC
[22], S1, S2 and S3 [28], as competitors in the experiments. We use 6 real-world monthly benchmark
data sets: FF25, FF25EU, FF32, FF49, FF100 and FF100MEINV to compare different methods.
These data sets are collected from the baseline and commonly-used Kenneth R. French’s Real-world
Data Library2. Details regarding these competitors and data sets are given in Appendix A.10. As
for mSSRM-PGA, we examine three levels of sparsity m = 10, m = 15, m = 20 and set ϵ = 10−3.
The setting of other parameters are presented in Appendix A.9. The codes of mSSRM-PGA are
accessible via the link: https://github.com/linyizun2024/mSSRM/tree/main/Codes_for_
Experiments_in_Paper.

4.1
Results for Sharpe ratios

We adopt the moving-window trading framework in [23] to imitate real-world portfolio management.
For each method, we use the asset returns {r(t)}T
t=1 or the price relatives {x(t) := r(t) + 1N}T
t=1
in the time window t = [1 : T] to update the portfolio ˆw(T +1) for the next trading time. On the
(T + 1)-th time, we compute the portfolio return by ˆr(T +1), ˆ
w = x⊤
(T +1) ˆw(T +1) −1 and then turn
to the next round where the time window moves to t = [2 : (T + 1)] and a new portfolio ˆw(T +2) is
computed. This procedure is repeated till the last trading time T , which yields a return sequence
{ˆr(t), ˆ
w}T
t=1. This sequence can be used to compute the test SR:

c
SR =
(PT
t=T +1 ˆr(t), ˆ
w)/(T −T)
q

(PT
s=T +1(ˆr(s), ˆ
w −(PT
t=T +1 ˆr(t), ˆ
w)/(T −T))2)/(T −T −1)
.

The 1/N strategy does not involve the time window size T. For all other methods, we examine two
conventional settings for the time window size in the ﬁnance industry [2, 17]: T = 60 and T = 120.

2http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

8


---Page Break---
Table 1 shows the (monthly) SRs of the 11 compared methods. Because MAXER requires T >
(N +2), it is unavailable on FF100 and FF100MEINV when T = 60. It is worth noting that the trivial
strategy 1/N outperforms most competitors in most situations. The reason is that 1/N diversiﬁes the
risk over all the assets, which is also an effective risk control approach [14]. However, mSSRM-PGA
outperforms all the competitors including 1/N on all the 6 data sets when T = 60 and on 5 data sets
when T = 120. For example, its SR is more than 70% higher than that of 1/N on FF25EU whether
T = 60 or T = 120. Hence mSSRM-PGA achieves competitive SRs with sparse portfolios, which
saves much managerial cost.

Table 1: Sharpe ratios of different portfolio optimization methods on 6 benchmark data sets.

Strategy
FF25 FF25EU FF32
FF49
FF100 FF100MEINV FF25 FF25EU FF32
FF49
FF100 FF100MEINV
T = 60
T = 120
1/N
0.2276 0.1574 0.2234 0.2057 0.2087
0.2151
0.2276 0.1574 0.2234 0.2057 0.2087
0.2151
SPOLC
0.1452 0.0315 0.1734 0.0752 0.0562
0.1009
0.1545 0.0350 0.1830 0.1291 0.0988
0.1218
SSPO
0.1544 0.0411 0.1181 0.0588 0.0425
0.0872
0.1789 0.0719 0.1557 0.0601 0.0529
0.1109
S1
0.1497 0.0369 0.1169 0.0559 0.0327
0.0879
0.1789 0.0736 0.1525 0.0648 0.0467
0.0999
S2
0.1382 0.0633 0.1225 0.0573 0.0456
0.1034
0.1578 0.0725 0.1438 0.0605 0.0602
0.1203
S3
0.1428 0.0607 0.1238 0.0570 0.0469
0.1100
0.1609 0.0709 0.1463 0.0617 0.0603
0.1215
SSMP
0.1934 0.1596 0.1535 0.1658 0.0883
0.1448
0.1920 0.0849 0.1512 0.1581 0.0573
0.1495
MAXER
0.1825 0.2229 0.1625 0.1581
N/A
N/A
0.1921 0.2379 0.1465 0.1433 0.1351
0.1479
IPSRM-D
0.2239 0.1994 0.1952 0.1436 0.1766
0.1662
0.2439 0.2358 0.2240 0.1410 0.2012
0.1712
PLCT
0.2475 0.2708 0.2600 0.2119 0.2270
0.2220
0.2468 0.2796 0.2577 0.2025 0.2369
0.2279
mSSRM-PGA(m=10) 0.2481 0.2712 0.2612 0.2151 0.2290
0.2217
0.2472 0.2796 0.2592 0.2041 0.2391
0.2271
mSSRM-PGA(m=15) 0.2481 0.2708 0.2615 0.2135 0.2289
0.2232
0.2474 0.2796 0.2592 0.2040 0.2381
0.2293
mSSRM-PGA(m=20) 0.2481 0.2708 0.2615 0.2134 0.2285
0.2234
0.2474 0.2796 0.2592 0.2041 0.2384
0.2292

4.2
Results for Cumulative Wealths

Ordinary investors are also concerned about how much they gain when using an investing strategy.
Without loss of generality, we can set the initial wealth for an investing strategy as S(0) = 1, then the
ﬁnal cumulative wealth can be conveniently computed by S(T ) = QT
t=1(ˆr(t), ˆ
w + 1). The results of
ﬁnal cumulative wealths are shown in Table 2. The two competitors 1/N and PLCT perform well in
general. Nevertheless, mSSRM-PGA achieves the best ﬁnal cumulative wealths on 4 out of the 6
data sets. Besides, it outperforms each competitor on at least 5 out of the 6 data sets. For example,
mSSRM-PGA is about 20% higher than the second best competitor PLCT on FF49 when T = 60
and m = 10. On the data sets where mSSRM-PGA is not the best method, it is still the second best
method. These results indicate that mSSRM-PGA is an effective strategy for pursuing return gain in
a practical perspective.

Table 2: Cumulative wealths of different portfolio optimization methods on 6 benchmark data sets.

Strategy
FF25 FF25EU FF32
FF49
FF100 FF100MEINV FF25 FF25EU FF32
FF49
FF100 FF100MEINV
T = 60
T = 120
1/N
355.98
13.05
424.42 235.48 364.87
428.70
355.98
13.05
424.42 235.48 364.87
428.70
SPOLC
57.53
0.96
169.58
5.44
2.39
14.05
70.46
1.03
259.74 100.49 16.03
36.20
SSPO
129.35
1.22
30.20
1.33
0.89
8.98
286.51
2.67
130.21
1.61
1.74
25.62
S1
100.76
1.08
29.47
1.09
0.54
9.25
265.82
2.78
121.47
2.23
1.27
15.57
S2
66.24
2.17
39.27
1.39
1.15
20.45
130.31
2.73
93.13
1.89
2.67
43.66
S3
70.88
2.01
36.88
1.28
1.20
23.73
129.90
2.61
89.51
1.92
2.62
38.70
SSMP
248.67
13.47
158.98 186.79 10.09
154.27
237.45
3.25
149.65 143.18
2.26
222.35
MAXER
173.39
47.56
200.03 142.31
N/A
N/A
216.94
55.71
117.42 98.85
79.82
188.54
IPSRM-D
398.55
37.25
243.83 69.57 240.12
146.40
567.76
77.79
507.47 50.04 457.86
188.34
PLCT
581.41 126.04 918.62 238.27 471.44
354.70
608.65 148.19 854.83 157.50 552.41
399.48
mSSRM-PGA (m=10) 615.34
126.02 991.89 285.02 527.09
375.75
640.89
147.17 928.19 188.38 635.65
421.97
mSSRM-PGA (m=15) 614.71
125.19 996.32 262.54 522.28
383.44
643.44
147.17 927.21 172.95 597.67
435.01
mSSRM-PGA (m=20) 614.70
125.19 996.23 262.06 515.50
384.65
643.44
147.17 927.16 173.27 603.05
433.15

4.3
Results for Transaction Costs

Cumulative wealth with transaction cost can also be tested to see how the transaction cost inﬂuences
the performance of different methods. We adopt the proportional transaction cost model [8, 26, 21]

Sν=S(0)

T
Y

t=1

"

( ˆw⊤
(t)x(t)) ·

 

1 −ν

2

N
X

i=1
| ˆw(t),i −˜w(t−1),i|

!#

,
˜w(t−1),i= ˆw(t−1),ix(t−1),i

ˆw⊤
(t−1)x(t−1)
,

where ˜w(t−1),i is the evolved portfolio weight of the i-th asset at the end of the (t −1)-th period,
and ν is the bidirectional transaction cost rate. When the cost rate of buying is the same as that

9


---Page Break---
of selling, updating the evolved portfolio ˜w(t−1) as the next portfolio ˆw(t) yields a proportional
transaction cost of ν

2
PN
i=1 | ˆw(t),i −˜w(t−1),i|. Figure 2 in Appendix A.10 shows the ﬁnal cumulative
wealths of different methods as ν varies from 0 to 0.5% with T = 60. mSSRM-PGA outperforms all
other competitors on FF25, FF25EU and FF32 for all ν ∈[0, 0.5%], and on FF100 for ν ⩽0.45%.
mSSRM-PGA is the second best method on FF100MEINV, following 1/N. This is because 1/N
naturally keeps a small trading volume. Note that a manager for a mutual fund with sufﬁcient trades
and capital is able to negotiate for a small enough ν. Thus mSSRM-PGA is applicable to scenarios
with a certain level of transaction cost.

4.4
Sparsity for mSSRM-PGA

In this subsection, we examine the sparsity of the portfolios { ˆw(t)} generated by mSSRM-PGA. The
sparsity can be measured by the cardinality of the support set of ˆw(t): |supp( ˆw(t))|. For each data set
and each setting of m, the mean and the standard deviation (STD) of {|supp( ˆw(t))|} are computed
to provide a general description, shown in Table 3. It indicates that mSSRM-PGA further increases
sparsity compared with the preseted sparsity level m. Moreover, mSSRM-PGA keeps stable sparsity
w.r.t. the change of m. For example, the average sparsity for mSSRM-PGA is about 4.9 when T = 60
(or 4.4 when T = 120) on FF25EU, for all the settings m = 10, 15, 20. As the total number of
assets N increases, mSSRM-PGA gets more advantageous in sparsity. For example, the average
sparsity for mSSRM-PGA is about 8 ∼11 on FF100 and FF100MEINV, compared with N = 100.
It indicates that mSSRM-PGA selects only 8% ∼11% of the assets in the whole asset pool, while
the widely-used 1/N strategy has to maintain the whole asset pool. Therefore, mSSRM-PGA can
save much managerial cost by reducing the proportion of selected assets, while keeping a competitive
performance in SR optimization.

Table 3: Sparsity of the portfolios generated by mSSRM-PGA: |supp( ˆw(t))|.
m
FF25 FF25EU FF32
FF49
FF100 FF100MEINV FF25 FF25EU FF32
FF49
FF100 FF100MEINV
T = 60
T = 120

10 Mean 6.3511 4.8342 6.4159 8.1214 8.0097
7.9175
7.1359 4.4560 7.0825 8.4790 8.3706
8.8722
STD 2.4164 2.1763 2.3654 1.9918 2.2473
2.3915
2.2221 1.4645 2.4464 1.9080 2.3343
2.0117

15 Mean 6.4746 4.9352 7.4286 9.0000 8.9709
9.0825
7.1637 4.4430 7.4919 9.6003 9.8754
10.5906
STD 3.0451 2.3573 3.0590 3.0713 3.3964
3.5906
2.2995 1.4462 2.2567 3.1731 3.6169
3.4845

20 Mean 6.4763 4.9352 7.4692 9.0421 9.1974
9.2994
7.1637 4.4430 7.4919 9.6828 10.0437
10.7621
STD 3.0462 2.3573 3.1734 3.1620 3.8949
4.0125
2.2995 1.4462 2.2567 3.3349 3.9160
3.7460

5
Concluding Remarks

The Sharpe ratio (SR) is a very important measurement for the performance of returns attributable
to risk in ﬁnance. On the other hand, modern portfolio management usually restricts the number of
selected assets to a relatively small size, in order to save managerial and ﬁnancial costs. The m-sparse
(ℓ0) constraint is an exact constraint for a sparse portfolio, but it is nonconvex and complex. Thus
few existing methods can optimize the SR with the m-sparse constraint. In this study, we convert
the m-sparse fractional optimization problem into an equivalent m-sparse quadratic programming
problem. Then we develop an efﬁcient, easy-to-implement and mathematically sound proximal
gradient algorithm to solve this nonconvex problem. We theoretically prove that this algorithm yields
a portfolio that achieves the globally optimal m-sparse Sharpe ratio under certain conditions.

We conduct extensive experiments on 6 real-world monthly benchmark data sets built on the Kenneth
R. French’s widely-used public data library. The numerical results demonstrate that the proposed
mSSRM-PGA improves the SR, compared with 9 state-of-the-art portfolio optimization methods
including SPOLC, SSPO, S1, S2, S3, SSMP, MAXER, IPSRM-D, PLCT and one baseline method 1/N.
For another evaluating metric cumulative wealth, mSSRM-PGA outperforms each competitor on at
least 5 out of the 6 data sets. Besides, mSSRM-PGA can withstand a considerable level of transaction
cost rate. Sparsity experiments indicate that mSSRM-PGA successfully generates portfolios with
stable sparsity, and its advantage increases as the size of the whole asset pool increases. In summary,
the proposed mSSRM-PGA is a promising approach in managing portfolios or other ﬁnancial issues,
which is worth further investigations. A limitation of this research lies in its inability to directly apply
to fractional optimization models featuring nondifferentiable numerator and denominator. Future
work will strive to broaden the theoretical and methodological foundations, ultimately enabling its
application to a broader spectrum of fractional optimization models in machine learning.

10


---Page Break---
Acknowledgements

The authors thank the anonymous reviewers for their constructive comments and valuable suggestions
in improving this paper. This work was supported in part by National Natural Science Founda-
tion of China under Grants 12401120 and 62176103, in part by Guangdong Basic and Applied
Basic Research Foundation under Grants 2021A1515110541 and 2023B1515120064, in part by the
Science and Technology Planning Project of Guangdong under Grant 2023A0505030013, and in
part by the Science and Technology Planning Project of Guangzhou under Grants 2024A04J3940,
2024A04J9896, 202206030007, Nansha District: 2023ZD001 and Development District: 2023GH01.

References

[1] Paul H. Algoet and Thomas M. Cover. Asymptotic optimality and asymptotic equipartition
properties of log-optimum investment. The Annals of Probability, 16(2):876–898, 1988.

[2] Mengmeng Ao, Li Yingying, and Xinghua Zheng. Approaching mean-variance efﬁciency for
large portfolios. The Review of Financial Studies, 32(7):2890–2919, 2019.

[3] Hédy Attouch, Jérôme Bolte, Patrick Redont, and Antoine Soubeyran. Proximal alternating
minimization and projection methods for nonconvex problems: An approach based on the
Kurdyka-Łojasiewicz inequality. Mathematics of Operations Research, 35(2):438–457, 2010.

[4] Hedy Attouch, Jérôme Bolte, and Benar Fux Svaiter. Convergence of descent methods for
semi-algebraic and tame problems: proximal algorithms, forward–backward splitting, and
regularized Gauss–Seidel methods. Mathematical Programming, 137(1):91–129, 2013.

[5] Gah-Yi Ban, Noureddine El Karoui, and Andrew E. B. Lim. Machine learning and portfolio
optimization. Management Science, 64(3):1136–1154, 2018.

[6] Heinz H. Bauschke and Patrick L. Combettes. Convex Analysis and Monotone Operator Theory
in Hilbert Space. Springer, New York, 2nd edition, 2017.

[7] Dimitri P. Bertsekas. Nonlinear Programming. Athena Scientiﬁc, Belmont, MA, 2nd edition,
1999.

[8] Avrim Blum and Adam Kalai. Universal portfolios with and without transaction costs. Machine
Learning, 35(3):193–205, 1999.

[9] Jérôme Bolte, Shoham Sabach, and Marc Teboulle. Proximal alternating linearized minimization
for nonconvex and nonsmooth problems. Mathematical Programming, 146(1):459–494, 2014.

[10] Joshua Brodie, Ingrid Daubechies, Christine De Mol, Domenico Giannone, and Ignace Loris.
Sparse and stable Markowitz portfolios. Proceedings of the National Academy of Sciences of
the United States of America, 106(30):12267–12272, 2009.

[11] Raul O. Chao, Stylianos Kavadias, and Cheryl Gaimon. Revenue driven resource allocation:
Funding authority, incentives, and new product development portfolio management. Manage-
ment Science, 55(9):1556–1569, 2009.

[12] Richard W. Cottle. Monotone solutions of the parametric linear complementarity problem.
Mathematical Programming, 3(1):210–224, 1972.

[13] Thomas M. Cover. Universal portfolios. Mathematical Finance, 1(1):1–29, 1991.

[14] Victor DeMiguel, Lorenzo Garlappi, and Raman Uppal. Optimal versus naive diversiﬁcation:
How inefﬁcient is the 1/N portfolio strategy? The Review of Financial Studies, 22(5):1915–1953,
2009.

[15] Stephen G. Dimmock, Neng Wang, and Jinqiang Yang. The endowment model and modern
portfolio theory. Management Science, 70(3):1554–1579, 2024.

[16] John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra. Efﬁcient projections
onto the ℓ1-ball for learning in high dimensions. In Proceedings of the International Conference
on Machine Learning (ICML), pages 272–279, 2008.

11


---Page Break---
[17] Shingo Goto and Yan Xu. Improving mean variance optimization through sparse hedging
restrictions. The Journal of Financial and Quantitative Analysis, 50(6):1415–1441, 2015.

[18] Kei Keung Hung, Chi Chiu Cheung, and Lei Xu. New Sharpe-ratio-related methods for portfolio
selection. In Proceedings of the IEEE/IAFE/INFORMS 2000 Conference on Computational
Intelligence for Financial Engineering (CIFEr), pages 34–37, 2000.

[19] John L. Kelly. A new interpretation of information rate. The Bell System Technical Journal,
35(4):917–926, 1956.

[20] Min Jeong Kim, Yongjae Lee, Jang Ho Kim, and Woo Chang Kim. Sparse tangent portfolio
selection via semi-deﬁnite relaxation. Operations Research Letters, 44(4):540–543, 2016.

[21] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. Radial basis func-
tions with adaptive input and composite trend representation for portfolio selection. IEEE
Transactions on Neural Networks and Learning Systems, 29(12):6214–6226, 2018.

[22] Zhao-Rong Lai, Liming Tan, Xiaotian Wu, and Liangda Fang. Loss control with rank-one co-
variance estimate for short-term portfolio optimization. Journal of Machine Learning Research,
21(97):1–37, 2020.

[23] Zhao-Rong Lai and Haisheng Yang. A survey on gaps between mean-variance approach
and exponential growth rate approach for portfolio optimization. ACM Computing Surveys,
55(2):1–36, 2023. Article No. 25.

[24] Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang, and Xiaotian Wu. Short-term sparse portfolio
optimization based on alternating direction method of multipliers. Journal of Machine Learning
Research, 19(63):1–28, 2018.

[25] Olivier Ledoit and Michael Wolf. Nonlinear shrinkage of the covariance matrix for portfolio
selection: Markowitz meets Goldilocks. The Review of Financial Studies, 30(12):4349–4388,
2017.

[26] Bin Li, Steven C.H. Hoi, Doyen Sahoo, and Zhi-Yong Liu. Moving average reversion strategy
for on-line portfolio selection. Artiﬁcial Intelligence, 222:104–123, 2015.

[27] Hong Liu and Mark Loewenstein. Market crashes, correlated illiquidity, and portfolio choice.
Management Science, 59(3):715–732, 2013.

[28] Ziyan Luo, Xiaotong Yu, Naihua Xiu, and Xingyuan Wang. Closed-form solutions for short-term
sparse portfolio optimization. Optimization, 71(7):1937–1953, 2022.

[29] Jong-Shi Pang. A parametric linear complementarity technique for optimal portfolio selection
with a risk-free asset. Operations Research, 28(4):927–941, 1980.

[30] Neal Parikh and Stephen Boyd. Proximal algorithms. Foundations and Trends R⃝in Optimization,
1(3):127–239, 2014.

[31] R. Tyrrell Rockafellar and Roger J-B. Wets. Variational Analysis, volume 317. Springer Science
& Business Media, 2009.

[32] William F. Sharpe. Capital asset prices: A theory of market equilibrium under conditions of
risk. Journal of Finance, 19(3):425–442, 1964.

[33] William F. Sharpe. Mutual fund performance. Journal of Business, 39(1):119–138, 1966.

[34] Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal
Statistical Society, 58(1):267–288, 1996.

[35] Xiaohui Yu and Lei Xu. Adaptive improved portfolio Sharpe ratio maximization with diversi-
ﬁcation. In Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural
Networks (IJCNN), pages 472–476, 2000.

12


---Page Break---
A
Appendix

A.1
Proof of Theorem 1

To prove Theorem 1, we need the following lemma.

Lemma 8 Suppose that there exists some ˜w ∈Ω1 such that p⊤˜w > 0. If ˆv is an optimal solution of
model (3.4), then ˆv ̸= 0N and p⊤ˆv = ˆv⊤Qϵˆv > 0.

Proof. Since ˆv is an optimal solution of model (3.4), we know that ˆv ∈Ω. Let w :=
p⊤˜
w
˜
w⊤Qϵ ˜
w ˜w. The
facts ˜w ∈Ω1 and p⊤˜w > 0 imply that w ∈Ω. Then it follows that

1
2 ˆv⊤Qϵˆv −p⊤ˆv ⩽1

2w⊤Qϵw −p⊤w = −1

2
(p⊤˜w)2

˜w⊤Qϵ ˜w < 0.

Hence p⊤ˆv > 1

2 ˆv⊤Qϵˆv ⩾0 and ˆv ̸= 0N. Now by letting v :=
p⊤ˆv
ˆv⊤Qϵ ˆv ˆv, then v ∈Ωand

1
2 ˆv⊤Qϵˆv −p⊤ˆv ⩽1

2v⊤Qϵv −p⊤v = −1

2
(p⊤ˆv)2

ˆv⊤Qϵˆv .
(A.1)

Multiplying both sides of (A.1) by 2ˆv⊤Qϵˆv yields
 
p⊤ˆv −ˆv⊤Qϵˆv
2 ⩽0,

which implies that p⊤ˆv = ˆv⊤Qϵˆv > 0.

Proof of Theorem 1. Let ˆv be an optimal solution of model (3.3). Then ˆv ∈Ω1 and

p⊤ˆv
p

ˆv⊤Qϵˆv
⩾
p⊤˜w
p

˜w⊤Qϵ ˜w
> 0.

Deﬁning ˜v :=
p⊤ˆv
ˆv⊤Qϵ ˆv ˆv, we see that ˜v ∈Ωand

1
2 ˜v⊤Qϵ˜v −p⊤˜v = 1

2
(p⊤ˆv)2

(ˆv⊤Qϵˆv)2 ˆv⊤Qϵˆv −(p⊤ˆv)2

ˆv⊤Qϵˆv = −1

2
(p⊤ˆv)2

ˆv⊤Qϵˆv < 0.
(A.2)

For any u ∈Ωsuch that 1

2u⊤Qϵu −p⊤u < 0, we have p⊤u > 0, u ̸= 0N and ˜u :=
u
u⊤1N ∈Ω1.
Then the fact ˆv is an optimal solution of model (3.3) implies that

(p⊤ˆv)2

ˆv⊤Qϵˆv ⩾(p⊤˜u)2

˜u⊤Qϵ ˜u = (p⊤u)2

u⊤Qϵu.
(A.3)

Note that
1
2u⊤Qϵu −p⊤u + 1

2
(p⊤u)2

u⊤Qϵu =
1
2u⊤Qϵu(u⊤Qϵu −p⊤u)2 ⩾0,

which combined with (A.3) and (A.2) yields

1
2u⊤Qϵu −p⊤u ⩾−1

2
(p⊤u)2

u⊤Qϵu ⩾−1

2
(p⊤ˆv)2

ˆv⊤Qϵˆv = 1

2 ˜v⊤Qϵ˜v −p⊤˜v.

Therefore, ˜v is an optimal solution of model (3.4).

Conversely, let ˆv be an optimal solution of model (3.4). It follows from Lemma 8 that ˆv ̸= 0N and
p⊤ˆv = ˆv⊤Qϵˆv > 0. Thus ˆv =
p⊤ˆv
ˆv⊤Qϵ ˆv ˆv. For any v ∈Ωsuch that p⊤v > 0, we let u :=
p⊤v
v⊤Qϵvv.
Then u ∈Ωand

−1

2
(p⊤ˆv)2

ˆv⊤Qϵˆv = 1

2 ˆv⊤Qϵˆv −p⊤ˆv ⩽1

2u⊤Qϵu −p⊤u = −1

2
(p⊤v)2

v⊤Qϵv .
(A.4)

Now we let ¯v :=
ˆv
ˆv⊤1N . Then ¯v ∈Ω1. Inequality (A.4) yields that

p⊤¯v
p

¯v⊤Qϵ¯v
=
p⊤ˆv
p

ˆv⊤Qϵˆv
⩾
p⊤v
p

v⊤Qϵv
.

Note that Ω1 ⊂Ω. Therefore, ¯v is an optimal solution of model (3.3).

13


---Page Break---
A.2
Proof of Theorem 2

To prove Theorem 2, we ﬁrst investigate the properties of function f in Proposition 9, and then recall
two well-known results as Lemmas 10 and 11. Let ψ be a function from Rn to [−∞, +∞]. Then
ψ is proper if −∞/∈ψ(Rn) and {x ∈Rn| ψ(x) < +∞} ̸= ∅. Let ψ : Rn →R be a proper
function. We say that ψ is convex if for any x, y ∈Rn and any λ ∈(0, 1), ψ(λx + (1 −λ)y) ⩽
λψ(x) + (1 −λ)ψ(y). If there exists β > 0 such that ψ −β

2 ∥· ∥2
2 is convex, then ψ is said to be
β-strongly convex.

Proposition 9 Let f : RN →R be deﬁned in (3.7). Then the following hold:

(i) f is ϵ-strongly convex on RN;

(ii) ∇f is ∥Qϵ∥2-Lipschitz continuous on RN.

Proof. Let ˜f(v) := f(v) −ϵ

2∥v∥2
2 = 1

2v⊤Q⊤Qv −p⊤v, v ∈RN. Since the Hessian matrix
Q⊤Q of ˜f is positive semideﬁnite, we know that ˜f is convex on RN (see Proposition B.4 of [7]).
Thus item (i) holds. The gradient of f is given by ∇f(v) = Qϵv −p. For all x, y ∈RN,
∥∇f(x) −∇f(y)∥2 ⩽∥Qϵ∥2∥x −y∥2, which implies item (ii).

Lemma 10 (Proposition A.24 of [7]) Let function ψ : Rn →R be differentiable with an L-
Lipschitz continuous gradient, where L > 0. Then

ψ(y) −ψ(x) ⩽⟨∇ψ(x), y −x⟩+ L

2 ∥y −x∥2
2

holds for all x, y ∈Rn.

Lemma 11 (Exercise 17.5 of [6]) Let ψ : Rn →R be differentiable and β > 0. Then ψ is β-
strongly convex if and only if

ψ(y) −ψ(x) ⩾⟨∇ψ(x), y −x⟩+ β

2 ∥y −x∥2
2

holds for all x, y ∈Rn.

Proof of Theorem 2. We ﬁrst show that (3.8) holds when v∗is a globally optimal solution of model
(3.7). By the deﬁnition of proximity operator, (3.8) is equivalent to

v∗= argmin
u∈RN ιΩ(u) + 1

2 ∥u −v∗+ α∇f(v∗)∥2
2 ,

that is,

ιΩ(u) + 1

2 ∥u −v∗+ α∇f(v∗)∥2
2 ⩾ιΩ(v∗) + 1

2 ∥α∇f(v∗)∥2
2 , for all u ∈RN.

According to the deﬁnition of ιΩin (3.6) and the fact v∗∈Ω, the above inequality can be simply
rewritten as
⟨∇f(v∗), u −v∗⟩+ 1

2α∥u −v∗∥2
2 ⩾0, for all u ∈Ω.
(A.5)

To prove (3.8), it sufﬁces to show that (A.5) holds. From Proposition 9, we know that f is ϵ-strongly
convex and ∇f is ∥Qϵ∥-Lipschitz continuous on RN. Then Lemma 10 yields that

f(u) −f(v∗) ⩽⟨∇f(v∗), u −v∗⟩+ ∥Qϵ∥2

2
∥u −v∗∥2
2, for all u ∈Ω.
(A.6)

Since v∗is a globally optimal solution of model (3.7), f(u) −f(v∗) ⩾0 for all u ∈Ω, which

together with (A.6) and the fact α ∈

0,
1
∥Qϵ∥2

i
implies (A.5). This proves that (3.8) holds.

Conversely, if α ⩾1

ϵ and (3.8) holds, then we have (A.5). Recall that f is ϵ-strongly convex. It
follows from Lemma 11 that

f(u) −f(v∗) ⩾⟨∇f(v∗), u −v∗⟩+ ϵ

2∥u −v∗∥2
2,

14


---Page Break---
which together with the fact α ⩾1

ϵ and (A.5) implies that f(u) −f(v∗) ⩾0 for all u ∈Ω. Thus
the assertion in item (i) holds.

We then prove item (ii). The fact (3.8) holds implies (A.5). For δ > 0, we deﬁne

˜Ωδ := {u ∈B(v∗; δ)| ⟨∇f(v∗), u −v∗⟩= 0}.

Note that when u tends to v∗, the quadratic term
1
2α∥u −v∗∥2
2 is of higher order inﬁnitesimal than
the linear term |⟨∇f(v∗), u −v∗⟩|. There must be some δ > 0 such that

|⟨∇f(v∗), u −v∗⟩| > 1

2α∥u −v∗∥2
2, for all u ∈B(v∗; δ)\˜Ωδ.
(A.7)

We then show that

⟨∇f(v∗), u −v∗⟩⩾0, for all u ∈

B(v∗; δ)\˜Ωδ

∩Ω.
(A.8)

Otherwise, there exists some ˜u ∈

B(v∗; δ)\˜Ωδ

∩Ωsuch that ⟨∇f(v∗), ˜u −v∗⟩< 0. It follows
from (A.7) that

⟨∇f(v∗), ˜u −v∗⟩+ 1

2α∥˜u −v∗∥2
2 < 0,

which contradicts (A.5). Hence (A.8) holds. This together with the deﬁnition of ˜Ωδ yields that

⟨∇f(v∗), u −v∗⟩⩾0, for all u ∈B(v∗; δ) ∩Ω.
(A.9)

Recall that f is convex and differentiable on RN. According to (A.9) and the ﬁrst order condition for
convexity (Proposition B.3 of [7]),

f(u) −f(v∗) ⩾⟨∇f(v∗), u −v∗⟩⩾0, for all u ∈B(v∗; δ) ∩Ω,

which implies that v∗is a locally optimal solution of model (3.7).

A.3
Proof of Proposition 3

Proof. By the deﬁnitions of ιΩand its proximity operator, we have

proxιΩ(v) = argmin
u∈Ω
∥u −v∥2.

To prove that h ∈proxιΩ(v), it is equivalent to show that

∥h −v∥2
2 ⩽∥u −v∥2
2, for all u ∈Ω.
(A.10)

For any u ∈Ω, there exists an index set Ju ∈NN with m elements such that uj = 0 for all
j ∈NN\Ju. Let Jv
neg be the index set of negative components in v and J′
u := (NN\Ju)∪Jv
neg. Since
u ⩾0N, ∥u −v∥2
2 ⩾P

j∈J′
u v2
j . Let J′
h = NN\Jv. Then Jv
neg ⊂J′
h and ∥h −v∥2
2 = P

j∈J′
p v2
j .
If mv > m, then Jv = Jv
m-pos. We are easy to see from the deﬁnition of Jv
m-pos that

X

j∈J′
u
v2
j −
X

j∈J′
h
v2
j =
X

j∈NN\(Ju∪Jv
neg)
v2
j −
X

j∈NN\(Jv
m-pos∪Jv
neg)
v2
j ⩾0.

If mv ⩽m, then Jv = Jv
pos and P
j∈J′u v2
j −P
j∈J′
h v2
j = P
j∈(NN\Ju)∪Jv
neg v2
j −P
j∈Jv
neg v2
j ⩾0.
Now we conclude from the above two cases that

∥u −v∥2
2 −∥h −v∥2
2 ⩾
X

j∈J′u
v2
j −
X

j∈J′
h
v2
j ⩾0,

that is, (A.10) holds. This completes the proof.

15


---Page Break---
A.4
Proof of Proposition 4

Proof. Item (i) follows from (3.9) and the deﬁnition of proxιΩdirectly. Then we have that ιΩ(v(k)) =
0 for all k ∈N. To prove item (ii), it sufﬁces to show that

f(v(k+1)) + a∥v(k+1) −v(k)∥2
2 ⩽f(v(k)), for all k ∈N.
(A.11)

Note that a = 1

α −∥Qϵ∥2 > 0, since α ∈

0,
1
∥Qϵ∥2


. Let

ϕ(u) := 1

2

u −v(k) + α∇f(v(k))

2

2 + ιΩ(u), u ∈RN.
(A.12)

Then (3.9) implies that ϕ(v(k+1)) ≤ϕ(v(k)), that is,

⟨∇f(v(k)), v(k+1) −v(k)⟩⩽−1

2α∥v(k+1) −v(k)∥2
2, for all k ∈N,
(A.13)

It follows from Lemma 10 that

f(v(k+1)) −f(v(k)) ⩽⟨∇f(v(k)), v(k+1) −v(k)⟩+ ∥Qϵ∥2

2
∥v(k+1) −v(k)∥2
2.
(A.14)

Combining (A.13) and (A.14) yields (A.11). Thus item (ii) holds. Now that F is monotonically
decreasing, according to the monotone convergence theorem, to prove item (iii), it sufﬁces to
show that function F is bounded below on Ω. Solving ∇f(v∗) = 0 gives v∗= Q−1
ϵ p. Since
f is convex and differentiable on RN, f attains the minimum value at Q−1
ϵ p on RN. Hence
f(v) ⩾f
 
Q−1
ϵ p

= −1

2p⊤Q−1
ϵ p for all v ∈RN, which implies that F(v) ⩾−1

2p⊤Q−1
ϵ p for all
v ∈Ω. Therefore, item (iii) holds. Now taking the limit on both sides of the inequality in item (ii)
yields item (iv) immediately. This completes the proof.

A.5
Proof of Theorem 5

In order to prove Theorem 5, it is necessary to review several deﬁnitions and establish several
preliminary results. First, We recall the notions of subdifferentials and critical point. The lower limit
of function ψ at x and the domain of ψ are deﬁned by

lim inf
y→x ψ(y) := lim
δ→0+


inf
y∈B(x;δ) ψ(y)

(A.15)

and
dom ψ := {x ∈Rn| ψ(x) < +∞},
respectively. We say that ψ is lower semicontinuous at x ∈Rn if ψ(x) ⩽lim inf
u→x ψ(u). If ψ is lower
semicontinuous at every x ∈Rn, then ψ is lower semicontinuous on Rn [31].

Deﬁnition 1 (Subdifferentials and critical point) Let ψ : Rn →R be a proper lower semicontinu-
ous function.

(i) For each x ∈dom ψ, the Fréchet subdifferential of ψ at x, written by ˆ∂ψ(x), is the set of
all vectors u ∈Rn which satisfy

lim inf
y→x
y̸=x

ψ(y) −ψ(x) −⟨u, y −x⟩

∥y −x∥2
⩾0.

When x /∈dom ψ, we set ˆ∂ψ(x) = ∅.

(ii) The limiting-subdifferential, or simply the subdifferential of ψ at x ∈dom ψ, written by
∂ψ(x), is deﬁned through the following closure process

∂ψ(x) := {u ∈Rn| ∃xk →x, ψ(xk) →ψ(x) and uk ∈ˆ∂ψ(xk) →u as k →+∞}.

We call an element in ∂ψ(x) subgradient of ψ at x. We say that x is a critical point of ψ if
0n ∈∂ψ(x).

16


---Page Break---
We also recall the following known results about subdifferential from Theorem 8.6, Exercise 8.8 (c)
and Theorem 10.1 of [31], respectively.

Fact 12 For x ∈dom ψ, ˆ∂ψ(x) ⊂∂ψ(x).

Fact 13 Let ψ1 : Rn →R and ψ2 : Rn →R be two proper lower semicontinuous functions and
x ∈Rn. If ψ1 is differentiable on a neighborhood of x and ψ2 is ﬁnite at x, then

∂(ψ1 + ψ2)(x) = ∇ψ1(x) + ∂ψ2(x).

Fact 14 (Fermat’s rule) If x ∈Rn is a local minimizer of ψ, then 0n ∈∂ψ(x).

We shall use Theorem 2.9 of [4], which is recalled as Proposition 15, to prove the convergence of the
PGA. For this purpose, we recall the notions of Kurdyka-Łojasiewicz (KL) property and KL function.

Deﬁnition 2 (KL property) Let ψ : Rn →R be a proper semicontinuous function. We say that ψ
satisﬁes the KL property at ˆx ∈dom ∂ψ if there exist η ∈(0, +∞], a neighborhood U of ˆx and a
continuous concave function ϕ : [0, η) →[0, +∞] such that

(i) ϕ(0) = 0;

(ii) ϕ is continuously differentiable on (0, η) with ϕ′ > 0;

(iii) ϕ′(ψ(x) −ψ(ˆx))·dist(0, ∂ψ(x)) ⩾1 for any x ∈U ∩{x ∈Rn : ψ(ˆx) < ψ(x) <
ψ(ˆx) + η}.

Deﬁnition 3 (KL function) We call a proper lower semicontinuous function ψ : Rn →R KL
function if ψ satisﬁes the KL property at all points in dom ∂ψ.

Proposition 15 Let ψ : Rn →R be a proper lower semicontinuous function. Consider a sequence
{x(k)}k∈N ⊂Rn satisfying the following conditions:

(i) There exists a > 0 such that

ψ(x(k+1)) + a∥x(k+1) −x(k)∥2
2 ⩽ψ(x(k)), for all k ∈N.

(ii) There exist b > 0 and y(k+1) ∈∂ψ(x(k+1)) such that

∥y(k+1)∥2 ⩽b∥x(k+1) −x(k)∥2, for all k ∈N.

(iii) There exist a subsequence {x(kj)}j∈N+ and x∗∈Rn such that

lim
j→∞x(kj) = x∗and
lim
j→∞ψ(x(kj)) = ψ(x∗).

If ψ satisﬁes the KL property at x∗, then

lim
k→∞x(k) = x∗and 0n ∈∂ψ(x∗).

We then focus on verifying that the sequence {v(k)}k∈N generated by PGA satisﬁes all the conditions
in Proposition 15. The satisfaction of item (i) has been shown in Proposition 4. We next consider the
satisfaction of item (ii) in Proposition 15.

Proposition 16 Let {v(k)}k∈N be generated by PGA. Then there exist q(k+1) ∈∂F(v(k+1)) and
b > 0 such that
∥q(k+1)∥2 ⩽b∥v(k+1) −v(k)∥2, for k ∈N.
(A.16)

Proof. Let

q(k+1) := 1

α(v(k) −v(k+1)) + ∇f(v(k+1)) −∇f(v(k)), k ∈N,

17


---Page Break---
and function ϕ be deﬁned by (A.12). We ﬁrst prove that q(k+1) ∈∂F(v(k+1)). It follows from (3.9)
and Fact 14 that 0N ∈∂ϕ(v(k+1)), which together with Fact 13 yields that

v(k) −v(k+1) −α∇f(v(k)) ∈∂ιΩ(v(k+1)), for all k ∈N.

Note that ιΩ= αιΩ. The above inclusion relation can be rewritten as

1
α(v(k) −v(k+1)) −∇f(v(k)) ∈∂ιΩ(v(k+1)), for all k ∈N.
(A.17)

Now combining (A.17) and the fact ∂F(v(k+1)) = ∇f(v(k+1)) + ∂ιΩ(v(k+1)) yields that q(k+1) ∈
∂F(v(k+1)), k ∈N.

We next prove that (A.16) holds. Since ∇f is Qϵ-Lipschitz continuous,

∥q(k+1)∥2 ⩽1

α∥v(k+1) −v(k)∥2 + ∥∇f(v(k+1)) −∇f(v(k))∥2 ⩽b∥v(k+1) −v(k)∥2,

where b :=
  1

α + ∥Qϵ∥2

. This completes the proof.

We then consider the satisfaction of item (iii) in Proposition 15. To this end, we need the following
two lemmas.

Lemma 17 Let {v(k)}k∈N be generated by PGA. If α ∈

0,
1
∥Qϵ∥2


, then {v(k)}k∈N is bounded.

Proof. We let γ := ∥I−αQϵ∥2, and denote by λmax(Qϵ), λmin(Qϵ) the maximum and the minimum
eigenvalues of Qϵ, respectively. Since Qϵ is symmetric positive deﬁnite and α <
1
∥Qϵ∥2 =
1
λmax(Qϵ),
we have γ = 1−α·λmin(Qϵ) ∈(0, 1). From Proposition 3, we are easy to see that ∥proxιΩ(v)∥2 ≤
∥v∥2 for all v ∈RN, which together with (3.9) yields that

∥v(k+1)∥2 ⩽∥v(k) −α∇f(v(k))∥2 = ∥(I −αQϵ)v(k) + αp∥2 ⩽γ∥v(k)∥2 + α∥p∥2,

for all k ∈N. The above inequality implies that

∥v(k+1)∥2 ⩽γk+1∥v(0)∥2 + α∥p∥2

k
X

j=0
γj = γk+1∥v(0)∥2 + α∥p∥2 · 1 −γk+1

1 −γ
,

for all k ∈N. Therefore, {v(k)}k∈N is bounded, since γ ∈(0, 1).

Lemma 18 Let {v(k)}k∈N be generated by PGA. If v∗is an accumulation point of {v(k)}k∈N, then
v∗∈Ω.

Proof. Since v∗is an accumulation point of {v(k)}k∈N, there exists a subsequence {v(kj)}j∈N+ of
{v(k)}k∈N such that lim
j→∞v(kj) = v∗. Note that the set Ωis closed and v(kj) ∈Ωfor all j ∈N+.

Hence v∗∈Ω, which completes the proof.

Proposition 19 Let {v(k)}k∈N be generated by PGA and F := f + ιΩ. If α ∈

0,
1
∥Qϵ∥2


, then

there exist a subsequence {v(kj)}j∈N+ of {v(k)}k∈N and v∗∈Ωsuch that

lim
j→∞v(kj) = v∗and
lim
j→∞F(v(kj)) = F(v∗).

Proof. It follows from Lemma 17 that {v(k)}k∈N is bounded. So there exists a subsequence
{v(kj)}j∈N+ of {v(k)}k∈N converges to some v∗∈RN. It follows from Lemma 18 that v∗∈Ω.
By the continuity of f on RN, we have limj→∞f(v(kj)) = f(v∗). We also know that ιΩ(v∗) =
ιΩ(v(k)) = 0 for all k ∈N. Therefore, limj→∞F(v(kj)) = F(v∗), which completes the proof.

To employ Proposition 15 for the convergence of PGA. We still need to show that F satisﬁes the KL
property at v∗. To this end, we recall the notions of semi-algebraic sets and functions, and recall a
known result in [4, 9] that establishes the relation between semi-algebraic property and KL property
as Lemma 20.

18


---Page Break---
Deﬁnition 4 (Semi-algebraic sets and functions) A subset S ⊂Rn is called real semi-algebraic if
it can be represented by

S =

s[

j=1

t\

i=1
{x ∈Rn| pij(x) = 0, qij(x) < 0} ,
(A.18)

where pij and qij are real polynomial functions for i ∈Nt, j ∈Ns, for some s, t ∈N+. A function
ψ : Rn →R is called semi-algebraic if its graph {(x, ψ(x)) : x ∈dom ψ} is a semi-algebraic
subset of Rn+1.

Lemma 20 Let ψ : Rn →R be a proper lower semicontinuous function. If ψ is semi-algebraic, then
it satisﬁes the KL property at any point of dom ∂ψ := {u ∈Rn| ∂ψ(u) ̸= ∅}.

Proposition 21 Let F := f +ιΩ, where function ιΩand f are deﬁned by (3.6) and (3.7), respectively.
Then dom ∂F = Ωand F is semi-algebraic.

Proof. We ﬁrst prove that dom ∂F = Ω. Note that dom F = Ωis closed, which together with
the deﬁnition of limiting-subdifferential implies that dom ∂F ⊂Ω. For any v ∈Ω, it is easy to
verify that 0N ∈ˆ∂ιΩ(v). Then we see from Fact 12 that 0N ∈∂ιΩ(v). Now by Fact 13, we have
∇f(v) ∈∂F(v), which means that ∂F(v) ̸= ∅, that is, v ∈dom ∂F. Hence Ω⊂dom ∂F. This
proves dom ∂F = Ω.

We then prove that F is semi-algebraic. From the deﬁnition of semi-algebraic function, we are easy
to see that the sum of semi-algebraic functions is still semi-algebraic. It is obvious that function f is
semi-algebraic. To prove that F is semi-algebraic, it sufﬁces to show that ιΩis semi-algebraic. The
graph of ιΩis given by

gra ιΩ=

x ∈RN+1 x1:N ⩾0N, ∥x1:N∥0 ⩽m and xN+1 = 0
	
,
(A.19)

where x1:N := (x1, x2, . . . , xN)⊤. Note that there are K :=
 
N
N−m

combinations to choose an
index set with (N −m) elements out of the set {1, 2, . . . , N}. We denote these index sets with size
(N −m) by J1, J2, . . . , JK, and let ˜Ji := Ji ∪{N + 1}, i ∈NK. Then the graph of ιΩin (A.19)
can be represented by

gra ιΩ=

K
[

j=1







\

i∈˜
Jj


x ∈RN+1 xi = 0
	


\


\

i/∈˜
Jj


x ∈RN+1 −xi ⩽0
	






,

which implies that ιΩis a semi-algebraic function. This completes the proof.

We show in the following proposition that the objective function F := f + ιΩsatisﬁes the KL
property at any accumulation point of sequence {v(k)}k∈N.

Proposition 22 Let {v(k)}k∈N be generated by PGA. If v∗is an accumulation point of {v(k)}k∈N,
then F satisﬁes the KL property at v∗.

Proof. Since v∗is an accumulation point of {v(k)}k∈N, it follows from Lemma 18 that v∗∈Ω. By
Proposition 21, we know that F is semi-algebraic and v∗∈dom ∂F. Thus the desired result follows
from Lemma 20 immediately.

We then show the continuity of the proximity operator proxιΩin the following proposition, which is
also required to prove the convergence of PGA.

Proposition 23 Let {x(k)}k∈N ⊂RN be a sequence converges to some x∗∈Ω, and let
h(k) = proxιΩ(x(k)) for k ∈N and h∗= proxιΩ(x∗) be given according to Proposition 3.
Then lim
k→∞h(k) = h∗.

Proof. Since x∗∈Ω, mx∗⩽m. We ﬁrst consider the case mx∗= 0, that is, x∗
j ⩽0 for all j ∈NN.

Then h∗= 0N. For all ε > 0, we let δ1 :=
ε
√

N . Then there exists K1 ∈N such that x(k)
j
⩽δ1 for

19


---Page Break---
all j ∈NN and k ⩾K1. By the deﬁnition of h(k), we know that 0 ⩽h(k)
j
⩽δ1 for all j ∈NN and
k ⩾K1. Hence
∥h(k) −h∗∥2 = ∥h(k)∥2 ⩽
√

Nδ1 = ε, for all k ⩾K1,

which implies lim
k→∞h(k) = h∗.

We then consider the case 0 < mx∗⩽m. In this case, the set {j ∈NN| x∗
j > 0} is nonempty. For

all ε > 0, we let δ2 := min
n
1
3x∗
min-pos,
ε
√

N

o
, where

x∗
min-pos := min
j∈NN{x∗
j| x∗
j > 0}.

There exists K2 ∈N such that for all k ⩾K2, ∥x(k) −x∗∥2 ⩽δ2, which indicates that

x(k)
j
⩾x∗
j −δ2 ⩾2

3x∗
min-pos > 0, for j ∈Jx∗
pos
(A.20)

and
x(k)
j
⩽x∗
j + δ2 ⩽δ2 ⩽1

3x∗
min-pos, for j ∈NN\Jx∗
pos.
(A.21)

By the fact mx∗⩽m and the deﬁnitions of h(k) and h∗, we can conclude from (A.20) and (A.21)
that for all k ⩾K2,
h(k)
j
= x(k)
j , h∗
j = x∗
j, for j ∈Jx∗
pos
and
0 ⩽h(k)
j
⩽δ2, h∗
j = 0, for j ∈NN\Jx∗
pos.

Thus ∥h(k) −h∗∥2 ⩽
√

Nδ2 ⩽ε, which yields lim
k→∞h(k) = h∗. This completes the proof.

We are now in a position to utilize Proposition 15 and Theorem 2 to prove Theorem 5.

Proof of Theorem 5. We ﬁrst prove item (i). According to Propositions 4, 16, 19 and 22, the
convergence of {v(k)}k∈N to a critical point v∗∈Ωof F := f + ιΩfollows from Proposition 15
immediately. By item (ii) in Theorem 2, to prove that v∗is a locally optimal solution of model (3.7)
(or model (3.4)), it sufﬁces to show that (3.8) holds. Since lim
k→∞v(k) = v∗, the Lipschitz continuity

of ∇f yields that
lim
k→∞


v(k) −α∇f(v(k))

= v∗−α∇f(v∗).

Now by letting k →∞on both side of (3.9) and employing Proposition 23, we obtain (3.8), which
proves the convergence of {v(k)}k∈N to a locally optimal solution v∗of model (3.4).

We then prove the convergence rates of {v(k)}k∈N. Let Φk(v) := ∥v −v(k) + α∇f(v(k))∥2
2,
v ∈RN. It is obvious that Φk is 2-strongly convex, since the Hessian matrix of function Φk −∥· ∥2
2
is positive semideﬁnite. To prove the convergence rate, we ﬁrst show that there exists K ∈N such
that
⟨∇f(v∗), v(k) −v∗⟩⩾0 and ⟨∇Φk(v(k+1)), v∗−v(k+1)⟩⩾0,
(A.22)
for all k ⩾K. From the proof of Theorem 2 in Appendix A.2, we see that (A.9) holds. It has been
shown that limk→∞v(k) = v∗and v(k) ∈Ωfor all k ∈N (see item (i) in Proposition 4). Then there
exists K1 ∈N such that v(k) ∈B(v∗; δ) ∩Ωfor all k ⩾K1, which together with (A.9) implies that
the ﬁrst inequality in (A.22) holds for all k ⩾K1. According to the deﬁnition of Φk and (3.9), we
see that

v(k+1) = argmin
v∈RN
1
2∥v −v(k) + α∇f(v(k))∥2
2 + ιΩ(v) = argmin
v∈Ω
Φk(v).

By a procedure similar to the ﬁrst paragraph of the proof of Theorem 2, we can establish that

v(k+1) = proxιΩ

v(k+1) −α∇Φk(v(k+1))

.

Note that Φk is also strongly convex. Using a proof analogous to that of the ﬁrst inequality in (A.22),
we can establish the existence of K2 ∈N such that the second inequality in (A.22) holds for all
k ⩾K2. By setting K = max{K1, K2}, we conclude that (A.22) holds for all k ⩾K.

20


---Page Break---
It follows from Lemma 11 that

Φk(v∗) ⩾Φk(v(k+1)) + ⟨∇Φk(v(k+1)), v∗−v(k+1)⟩+ ∥v∗−v(k+1)∥2
2,
(A.23)

which together with the second inequality in (A.22) implies that

Φk(v∗) ⩾Φk(v(k+1)) + ∥v∗−v(k+1)∥2
2,

that is,

∥v∗−v(k) + α∇f(v(k))∥2
2 ⩾∥v(k+1) −v(k) + α∇f(v(k))∥2
2 + ∥v∗−v(k+1)∥2
2,
(A.24)

for all k ⩾K. For simplicity of notation, we deﬁne z(k) := v(k+1) −v(k), k ∈N. Expanding
(A.24) and dividing the resulting inequality by 2α yields

⟨∇f(v(k)), z(k)⟩+ 1

2α∥z(k)∥2
2 + 1

2α∥v∗−v(k+1)∥2
2 ⩽1

2α∥v∗−v(k)∥2
2 + ⟨∇f(v(k)), v∗−v(k)⟩,
(A.25)
for all k ⩾K. Recall that (A.14) in the proof of Proposition 4 holds. Since α ∈

0,
1
∥Qϵ∥2


, (A.14)
implies that

f(v(k+1)) −f(v(k)) ⩽⟨∇f(v(k)), z(k)⟩+ 1

2α∥z(k)∥2
2, for all k ∈N.
(A.26)

Combining (A.26) and (A.25), we obtain that

f(v(k+1)) −f(v(k)) + 1

2α∥v∗−v(k+1)∥2
2 ⩽1

2α∥v∗−v(k)∥2
2 + ⟨∇f(v(k)), v∗−v(k)⟩, (A.27)

for all k ⩾K. It follows from the ﬁrst order condition for convexity (Proposition B.3 of [7]) that

f(v∗) ⩾f(v(k)) + ⟨∇f(v(k)), v∗−v(k)⟩,

which together with (A.27) yields

f(v(k+1)) + 1

2α∥v∗−v(k+1)∥2
2 ⩽f(v∗) + 1

2α∥v∗−v(k)∥2
2,

that is,

f(v(k+1)) −f(v∗) ⩽1

2α


∥v(k) −v∗∥2
2 −∥v(k+1) −v∗∥2
2

, for all k ⩾K.

We see from (A.11) that {f(v(k))}k∈N is monotonically decreasing. Then for all j ∈N+,

j

f(v(K+j)) −f(v∗)

⩽

K+j−1
X

i=K


f(v(i+1)) −f(v∗)


⩽1

2α

K+j−1
X

i=K


∥v(i) −v∗∥2
2 −∥v(i+1) −v∗∥2
2


= 1

2α


∥v(K) −v∗∥2
2 −∥v(K+j) −v∗∥2
2

.

Hence
f(v(K+j)) −f(v∗) ⩽
1
2αj ∥v(K) −v∗∥2
2.
(A.28)

Note that ∥v(K) −v∗∥2
2 is a constant. Let k = K + j and C :=
1
α∥v(K) −v∗∥2
2. Then (A.28)
implies that

f(v(k)) −f(v∗) ⩽C · 1

2j ⩽C · 1

k , for j ⩾K.

Thus

|f(v(k)) −f(v∗)| = O
1

k


.
(A.29)

21


---Page Break---
Combining Lemma 11 and the ﬁrst inequality in (A.22), we obtain that

f(v(k)) −f(v∗) ⩾ϵ

2∥v(k) −v∗∥2
2, for all k ⩾K.

This together with (A.29) yields that ∥v(k) −v∗∥2 = O

1
√

k


.

We next prove item (ii). The fact v∗⩾0N follows from (3.8) and Proposition 3 directly. Assume,
to reach a contradiction, that v∗= 0N. Item (i) in this theorem shows that v∗is a locally optimal
solution of model (3.7). Then there exists δ > 0 such that

f(v) ⩾f(v∗) = f(0N) = 0, for all v ∈Ω∩B(v∗; δ).
(A.30)

We recall from the assumption of this theorem that there exists ˜w ∈Ωsuch that p⊤˜w > 0. Let
˜wα := α ˜w, where α > 0. Then ˜wα ∈Ωand p⊤˜wα > 0. Note that when α tends to 0, the quadratic
term 1

2 ˜w⊤
α Qϵ ˜wα is of higher order inﬁnitesimal than the linear term p⊤˜wα. There exists some
sufﬁcient small α > 0 such that ˜wα ∈B(v∗; δ) and

f( ˜wα) = 1

2 ˜w⊤
α Qϵ ˜wα −p⊤˜wα < 0,

which contradicts (A.30). Therefore, v∗̸= 0N.

Lastly, we prove item (iii). Since v∗⩾0N and v∗̸= 0N. There exit ε0 > 0 and K′ ∈N such that
(v∗)⊤1N > ε0 and (v(k))⊤1N > ε0 for all k ⩾K′, and hence w(k) =
v(k)

(v(k))⊤1N for all k ⩾K′.

Note that v∗and {v(k)}k∈N are both bounded. There exist C1 > 0 and C2 > 0 such that

∥v(k)∥2
(v(k))⊤1N
 · |(v∗)⊤1N| ⩽C1 and C1
√

N +

1
(v∗)⊤1N

 ⩽C2.

Then for all k ⩾K′,
w(k) −
v(k)

(v∗)⊤1N


2
⩽

1
(v(k))⊤1N
−
1
(v∗)⊤1N

 · ∥v(k)∥2

=
∥v(k)∥2
(v(k))⊤1N
 · |(v∗)⊤1N| ·
(v(k) −v∗)⊤1N


⩽C1
√

N∥v(k) −v∗∥2,

and hence

∥w(k) −w∗∥2 =
w(k) −
v(k)

(v∗)⊤1N
+
v(k)

(v∗)⊤1N
−w∗

2

⩽C1
√

N∥v(k) −v∗∥2 +

1
(v∗)⊤1N

 ∥v(k) −v∗∥2

⩽C2∥v(k) −v∗∥2.

This implies that ∥w(k) −w∗∥2 = O

1
√

k


. Similarly, we can prove that there exists C3 > 0 such
that
|S(w(k)) −S(w∗)| ⩽C3∥w(k) −w∗∥2, for all k ⩾K′,

which implies that |S(w(k)) −S(w∗)| = O

1
√

k


. This completes the proof.

A.6
Proof of Theorem 6

To prove Theorem 6, we recall Proposition 11.4 of [6] as the following lemma.

Lemma 24 Let ψ : Rn →R be be proper and convex. Then every local minimizer of ψ is a global
minimizer.

22


---Page Break---
Proof of Theorem 6. We ﬁrst prove item (i). Let ιˆΩbe deﬁned by

ιˆΩ(v) :=
0,
if v ∈ˆΩ;
+∞,
otherwise.

Then model (3.10) is equilvalent to min
v∈RN ˆF(v), where ˆF := f + ιˆΩ. Of course, ιˆΩis proper. The

convexity of ˆΩimplies that ιˆΩis convex (see Example 8.3 of [6]). Recall that f is strictly convex,
and hence ˆF is proper and strictly convex. Since v∗is a locally optimal solution of model (3.7) and
ˆΩ⊂Ω, there exists δ > 0 such that

f(u) ⩾f(v∗), for all u ∈B(v∗; δ) ∩ˆΩ.
(A.31)

The fact v∗∈ˆΩgives ιˆΩ(v∗) = 0. Then (A.31) implies that ˆF(u) ⩾ˆF(v∗) for all u ∈B(v∗; δ),
that is, v∗is a local minimizer of ˆF. Now it follows from Lemma 24 that v∗is also a global minimizer
of ˆF. The strict convexity of ˆF implies the uniqueness of its global minimizer.

We next prove item (ii). From item (i) in this theorem and item (ii) in Theorem 5, we see that
v∗∈ˆΩis a globally optimal solution of model (3.10) and v∗̸= 0N. Then we are able to prove that
p⊤v∗= (v∗)⊤Qϵv∗> 0. We omit this proof here since it is very similar to the proof of Lemma 8.
Now we have v∗=
p⊤v∗

(v∗)⊤Qϵv∗v∗. For any v ∈ˆΩsuch that p⊤v > 0, we let u :=
p⊤v
v⊤Qϵvv. Then

u ∈ˆΩand p⊤u > 0. Hence

−1

2
(p⊤v∗)2

(v∗)⊤Qϵv∗= 1

2(v∗)⊤Qϵv∗−p⊤v∗⩽1

2u⊤Qϵu −p⊤u = −1

2
(p⊤v)2

v⊤Qϵv ,

which implies that
p⊤v∗
p

(v∗)⊤Qϵv∗⩾
p⊤v
p

v⊤Qϵv
, for all v ∈ˆΩ.

Since v∗∈ˆΩand v∗̸= 0N, by the deﬁnition of ˆΩ1, we see that w∗∈ˆΩ1. Note that ˆΩ1 ⊂ˆΩ. For
all w ∈ˆΩ1,
p⊤w
p

w⊤Qϵw
⩽
p⊤v∗
p

(v∗)⊤Qϵv∗=
p⊤w∗
p

(w∗)⊤Qϵw∗,

which implies that w∗is a globally optimal solution of model max
w∈ˆΩ1
S(w).

Lastly, we prove item (iii). Note that w∗
j > 0 for all j ∈Jv∗
pos. Let w∗
min-pos :=
min
j∈Jv∗
pos
{w∗
j },

δ := 1

3w∗
min-pos, and let w be any vector in B(w∗; δ) ∩Ω1. Then wj ⩾2δ > 0 for j ∈Jv∗
pos, and
|wj| ⩽δ for j ∈NN\Jv∗
pos. Since w ∈Ω1 and mv∗= m, we conclude that wj = 0 for j ∈NN\Jv∗
pos,
which implies that w ∈ˆΩ1. It has been shown that w∗is an optimal solution of model max
w∈ˆΩ1
S(w).

Therefore, S(w) ⩽S(w∗) for all w ∈B(w∗; δ) ∩Ω1, that is, w∗is a locally optimal solution of
model (3.3). This completes the proof.

A.7
Proof of Theorem 7

Proof. According to Theorem 1 and item (i) in Theorem 2, to prove the desired result, it sufﬁces to
show that

v∗= proxιΩ


v∗−1

ϵ ∇f(v∗)

.
(A.32)

From the proof of Theorem 5, we know that (3.8) holds. According to the computation of proxιΩ
in Proposition 3, to guarantee the validity of (3.8), we have ∇if(v∗) = 0 for all i ∈supp(v∗).
Otherwise, there exists some i0 ∈supp(v∗) such that v∗
i0 ̸= v∗
i0 −α∇i0f(v∗), which together with
Proposition 3 implies that v∗̸= proxιΩ(v∗−α∇f(v∗)), a contradiction to (3.8).

Suppose that mv∗< m. Then we have ∇if(v∗) ⩾0 for all i ∈NN\supp(v∗). Otherwise, there
exists some i1 ∈NN\supp(v∗) such that v∗
i1 −α∇i1f(v∗) > 0. Note that mv∗< m. The operation

23


---Page Break---
of proxιΩwill preserve the positive value v∗
i1 −α∇i1f(v∗) instead of truncating it as 0, which violates
(3.8). In this case, we now have v∗
i −1

ϵ ∇if(v∗) = v∗
i for i ∈supp(v∗) and v∗
i −1

ϵ ∇if(v∗) ⩽0
for i ∈NN\supp(v∗), which imply that (A.32) holds.

Suppose that item (ii) holds. Let δ := min{v∗
i |i ∈supp(v∗)} > 0. For i ∈NN\supp(v∗),
since 1

ϵ ∇if(v∗) > −δ and vi = 0, we have v∗
i −1

ϵ ∇if(v∗) < δ. Note that mv∗= m. The
operation of proxιΩmakes v∗
i −1

ϵ ∇if(v∗) = v∗
i for i ∈supp(v∗) and v∗
i −1

ϵ ∇if(v∗) = 0 for
i ∈NN\supp(v∗), that is, (A.32) holds. This completes the proof.

A.8
Validation of PGA’s Global Optimality Through Simulation Experiments

To test the validation of PGA’s global optimality, we conduct a set of simulation experiments by
considering model (3.7), where Qϵ := Q⊤Q + ϵI. The iterative scheme of PGA for solving this
model is given by (3.9) with α =
0.99
∥Q∥2 .

In the simulation experiments, we set Σ ∈R10×10 by Σij := 0.5|i−j|, and randomly generate a
matrix Q ∈R50×10 from the multivariate normal distribution, with mean vector 010 and covariance
matrix Σ. We set p as a random vector with components that are randomly generated numbers in the
range [−10, 10], and casually set ϵ = 0.001 and the sparsity m = 3.

The direct exhaustive approach enumerates all possible support set conﬁgurations, totaling C3
10 = 120
cases. In each case, we solve a 3-dimension quadratic programming problem. By comparing the
optimal solutions corresponding to these 120 cases, we obtain the exact globally optimal solution of
model (3.7). After that, we can evaluate the optimality of PGA’s convergence.

For each experiment, we performed 500 iterations of PGA. To ensure the robustness of our ﬁndings,
we used three different initializations: 0N, 1N/N and 1N. We repeated the experiments 104 times
for each initialization, with different Q and p in each run. We found that in over 7,200 of the
104 trials, for any of the three initializations, both the normalized error of the iterative sequence
∥v(k) −v∗∥2/∥v∗∥2 and the normalized error of the function value |f(v(k)) −f(v∗)|/|f(v∗)|
were smaller than 10−10. Here v∗denotes the globally optimal solution, and v(k) represents the
iterative sequence at the k-th iteration of PGA. We show the plots of ∥v(k) −v∗∥2/∥v∗∥2 and
|f(v(k)) −f(v∗)|/|f(v∗)| in the following Figure 1, and show in Table 4 these two normalized
errors obtained at 500 iterations of PGA in ten simulation experiments.

0
100
200
300
400
500
Iteration number

0

0.2

0.4

0.6

0.8

0
100
200
300
400
500
Iteration number

0

0.1

0.2

0.3

0.4

0.5

Figure 1: Simulation results of PGA for model (3.7). Left: normalized error of the iterative sequence
versus number of iterations. Right: normalized error of function value versus number of iterations.

Table 4: The normalized errors obtained at 500 iterations of PGA in 10 simulation experiments.

k
1
2
3
4
5
6
7
8
9
10

∥v(k)−v∗∥2

∥v∗∥2
2.45E-16 1.62E-16 3.03E-16 6.23E-01 1.14E-16 7.68E-16 7.41E-16 9.55E-17 1.50E-16 2.52E-16

∥f(v(k))−f(v∗)∥2

∥f(v∗)∥2
0.00
1.48E-16 1.68E-16 1.17E-01
0.00
1.40E-16 4.73E-16 1.36E-16
0.00
0.00

From the simulation experiments, we conclude that the proposed PGA has a high probability (over
72%) of directly converging to a globally optimal solution of model (3.7). This ﬁnding is consistent
with the sufﬁcient conditions for global optimality in Theorem 7.

24


---Page Break---
A.9
Solving Algorithm: mSSRM-PGA

Algorithm A1 mSSRM-PGA
Input: Given the sample asset return matrix R ∈RT ×N and the positive parameter ϵ.
Preparation: Let p =
1
T R⊤1T , Q =
1
√T −1
 
R −1

T 1T ×T R

and Qϵ = Q⊤Q + ϵI. Compute
the largest eigenvalue λ1 of Qϵ, and set α = 0.999

λ1 .
Initialization: Set v(0) = p, tol = 10−5, MaxIter = 104 and k = 0.
repeat
1. v(k+1) = proxιΩ
 
v(k) −α
 
Qϵv(k) −p


2. k = k + 1

until ∥v(k)−v(k−1)∥2

∥v(k−1)∥2
⩽tol or k > MaxIter.

if v(k) ̸= 0N
3. w∗=
v(k)

(v(k))⊤1N
else
4. w∗= 0N
Output: The portfolio w∗.

A.10
Additional Experimental Results

The 1/N strategy rebalances to the equally-weighted portfolio on each trading time. S1, S2 and
S3 are different versions of SSPO-ℓ0 in (2.5), among which S1 is deterministic but S2 and S3 are
randomized. The hyperparameters of these competitors are set according to the original papers.

FF25 contains 25 portfolios developed by BE/ME (book equity to market equity) and investment in
the US market. FF25EU contains 25 portfolios developed by ME and prior return in the European
market. FF32 contains 32 portfolios developed by BE/ME and investment in the US market. FF49
contains 49 industry portfolios in the US market. FF100 contains 100 portfolios developed by ME
and BE/ME, while FF100MEINV contains 100 portfolios developed by ME and investment, both in
the US market. The information of these data sets are given in Table 5.

Table 5: Information of 6 real-world monthly benchmark data sets.

Data Set
Region
Time
Months
Assets
FF25
US
Jul/1971 ∼May/2023
623
25
FF25EU
EU
Nov/1990 ∼May/2023
391
25
FF32
US
Jul/1971 ∼May/2023
623
32
FF49
US
Jul/1971 ∼May/2023
623
49
FF100
US
Jul/1971 ∼May/2023
623
100
FF100MEINV
US
Jul/1971 ∼May/2023
623
100

There is a relaxation approach based on the semi-deﬁnite programming (SDP Relaxation, [20])
that intends to address nearly the same mSSRM model (3.3) of this paper, except for relaxing the
cardinality constraint and the long-only constraint. Therefore, this method fails to control cardinality
exactly and a simplex projection [16] should be implemented to ensure feasibility. Its experimental
results are also provided in Table 6, which are not so good as those of mSSRM-PGA.

Table 6: Final cumulative wealths (CW) and Sharpe Ratios (SR) of SDP Relaxation and mSSRM-PGA
on 6 data sets (T = 60).

Data Set
FF25
FF25EU
FRENCH32
FF49
FF100
FF100MEINV
Strategy
CW
SR
CW
SR
CW
SR
CW
SR
CW
SR
CW
SR
SDP Relaxation
323.76 0.2340
14.25
0.1674 290.24 0.2224 280.46 0.2151
0.51
0.0218 194.09 0.1528
mSSRM-PGA (m=10) 615.34 0.2481 126.02 0.2712 991.89 0.2612 285.02 0.2151 527.09 0.2290 375.75 0.2217
mSSRM-PGA (m=15) 614.71 0.2481 125.19 0.2708 996.32 0.2615 262.54 0.2135 522.28 0.2289 383.44 0.2232
mSSRM-PGA (m=20) 614.70 0.2481 125.19 0.2708 996.23 0.2615 262.06 0.2134 515.50 0.2285 384.65 0.2234

As for practical issues, Table 7 shows the running times for different methods with T = 60, where
mSSRM-PGA achieves competitive computational efﬁciency. Figure 2 shows the ﬁnal cumulative
wealths of different methods as the transaction cost rate ν varies from 0 to 0.5% with T = 60, which
indicates that mSSRM-PGA can withstand considerable levels of transaction cost rates.

25


---Page Break---
Table 7: The average running times (in seconds) of different portfolio optimization models for one
period on 6 data sets.

Data Set
SPOLC
SSPO
S1
S2
S3
SSMP MAXER IPSRM-D PLCT SDP Relaxation mSSRM-PGA
FF25
0.0263
0.0234 5.72E-05 5.63E-05 8.46E-05 0.0122
0.0525
0.0009
0.0020
1.0115
0.0052
FF25EU
0.0222
0.0239 1.70E-05 2.89E-05 2.59E-05 0.0316
0.0588
0.0009
0.0015
0.8178
0.0059
FRENCH32
0.0239
0.0250 2.93E-05 2.81E-05 3.08E-05 0.0075
0.0392
0.0012
0.0021
1.2862
0.0075
FF49
0.0252
0.0458 2.94E-05 3.51E-05 2.82E-05 0.0083
0.0270
0.0029
0.0034
12.3780
0.0114
FF100
0.0306
0.0854 5.38E-05 4.48E-05 4.27E-05 0.0451
0.0458
0.0132
0.0052
24.5852
0.0713
FF100MEINV
0.0296
0.0864 5.08E-05 4.81E-05 4.53E-05 0.0145
0.0152
0.0144
0.0059
23.9911
0.0696

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

100

200

300

400

500

600

700

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(a) FF25

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

20

40

60

80

100

120

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(b) FF25EU

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

100

200

300

400

500

600

700

800

900

1000

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(c) FF32

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

50

100

150

200

250

300

350

400

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(d) FF49

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

50

100

150

200

250

300

350

400

450

500

550

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(e) FF100

0
0.1
0.2
0.3
0.4
0.5
Transaction Cost Rate (%) 

0

100

200

300

400

500

600

700

Cumulative Wealth

1/N
SPOLC
SSPO
S3
S2
S1
SSMP

MAXER
IPSRM-D
PLCP
SDP-relaxation
mSSRM-PGA(m=10)
mSSRM-PGA(m=15)
mSSRM-PGA(m=20)

(f) FF100MEINV
Figure 2: Final cumulative wealths of portfolio optimization methods w.r.t. transaction cost rate ν on
6 benchmark data sets.

26


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justiﬁcation: The whole project with all the codes for reproductions would be made public
if this paper were to be accepted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?

Answer: [Yes]

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

11. Safeguards

27


---Page Break---
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justiﬁcation: This work has no social risk of misuse.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]

28


---Page Break---
