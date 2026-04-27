Stochastic Optimization Algorithms for Instrumental
Variable Regression with Streaming Data

Xuxing Chen♯
Department of Mathematics
University of California, Davis
xuxchen@ucdavis.edu

Abhishek Roy♯
Halıcıo˘glu Data Science Institute
University of California, San Diego
a2roy@ucsd.edu

Yifan Hu
College of Management, EPFL
Department of Computer Science, ETH Zurich
yifan.hu@epfl.ch

Krishnakumar Balasubramanian
Department of Statistics
University of California, Davis
kbala@ucdavis.edu

Abstract

We develop and analyze algorithms for instrumental variable regression by viewing
the problem as a conditional stochastic optimization problem. In the context of
least-squares instrumental variable regression, our algorithms neither require matrix
inversions nor mini-batches and provides a fully online approach for performing
instrumental variable regression with streaming data. When the true model is linear,
we derive rates of convergence in expectation, that are of order O(log T/T) and
O(1/T 1−ι) for any ι > 0, respectively under the availability of two-sample and
one-sample oracles, where T is the number of iterations. Importantly, under the
availability of the two-sample oracle, our procedure avoids explicitly modeling
and estimating the relationship between the independent and the instrumental
variables, demonstrating the benefit of the proposed approach over recent works
based on reformulating the problem as minimax optimization problems. Numerical
experiments are provided to corroborate the theoretical results.

1
Introduction

Instrumental variable analysis is widely used in fields like econometrics, health care [TM17], social
science [Bol12], and online advertisement to estimate the causal effect of a random variable, X,
on an outcome variable, Y , when an unobservable confounder influences both. By identifying an
instrumental variable correlated with the variable X but unrelated to the confounders, researchers
can isolate the exogenous variation in X and estimate a causal relationship between X and Y . In
the context of regression, Instrumental Variable Regression (IVaR) addresses endogeneity issues
when an independent variable is correlated with the error term in the regression model, leveraging an
instrument variable Z such that Y is independent of X|Z. In this paper, we focus on the following
statistical model:

Y = gθ∗(X) + ϵ1
with
X = hγ∗(Z) + ϵ2
(1)

where X ∈Rdx and ϵ1 are correlated and ϵ2 is a centered unobserved noise (independent of Z ∈Rdz),
leading to confounding in the model between X and Y ∈R. Here ϵ1 and ϵ2 are dependent, and
θ∗and γ∗are true parameters for the respective function g and h. Our goal is to design efficient
algorithms that recovers θ∗from the data.

♯XC and AR contributed equally to this work.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Traditionally, IVaR algorithms are based on two-stage estimation procedures, where we first regress
Z and X to obtain an estimator b
X, and then regress b
X and Y , with the essence that b
X is independent
of Y , and thus eliminating the aforementioned endogeneity of the unknown confounder. A vast
literature has devoted to understanding the two-stage approaches [HH05, DFFR11, HLLBT17],
with the parametric two-stage least-squares (2SLS) procedure being the most canonical one [AI95].
The main drawback of this approach is that the second-stage regression problem is affected by the
estimation error from the regression problem corresponding to first stage. In fact, [AP09] call the first
stage regression as “forbidden regression”, due to the concerns in estimating a nuisance parameter.

Considering the squared loss function, [MMLR20] formulate the IVaR problem as a conditional
stochastic optimization problem [HZCH20]:

min
g∈G F(g) := EZEY |Z[(Y −EX|Z[g(X)])2].
(2)

However, [MMLR20] did not solve problem (2) efficiently, and resort to reformulating (2) further
as a minimax optimization problem. Indeed, they mention explicitly in their work that “it remains
cumbersome to solve (2) directly because of the inner expectation”. Then, they leverage the Fenchel
conjugate of the squared loss, leading to a minimax optimization with maximization over a continuous
functional space. Following [DHP+17], [MMLR20] propose to use reproducing kernel Hilbert space
(RKHS) to handle the maximization over continuous functional space. See also [LS18, BKS19,
DLMS20, LCY+20, BKM+23] for similar minimax approaches. The issue with such an approach is
that approximating the dual variable via maximization over continuous functional space inevitably
introduces approximation error. Hence, although there is no explicit nuisance parameter estimation
step like in the two-stage approach, there is an implicit one, which makes the minimax approach less
appealing as an alternate to the two-stage procedures.

In this work, contrary to the claim made in [MMLR20] that problem (2) is cumbersome to solve,
we design and analyze efficient streaming algorithms to directly solve the conditional stochastic
optimization problem in (2). Direct application of methods from [HZCH20] for solving (2) is
possible, yet their approach utilizes nested sampling, i.e., for each sample of Z, [HZCH20] generate
a batch of samples of X from P(X|Z), to reduce the bias in estimating the composition of non-
linear loss function with conditional expectations. Thus their methods are not suitable for the
streaming setting that we are interested in. Considering (2), we first parameterize the function class
G := {g(θ; X) | θ ∈Rdθ}. Now, defining F(g) := F(θ), we observe that the gradient ∇F(θ)
admits the following form

∇F(θ) = EZ[(EX|Z[g(θ; X)] −EY |Z[Y ])∇θEX|Z[g(θ; X)]],
(3)

which implies that one does not need the nested sampling technique to reduce the bias. However, the
presence of product of two conditional expectations EX|Z[g(θ; X)] still causes significant challenges
in developing stochastic estimators of the above gradient in the streaming setting. In this work, we
overcome this challenge and develop two algorithms that are applicable to the streaming data setting
avoiding the need for generating batches of samples of X from P(Z|X).

Contributions. We make the following contributions in this work.

• Two-sample oracles: Our first algorithm leverages the observation that if we have access to
a two-sample oracle that outputs two samples X and X′ that are independent conditioned on
the instrument Z, we can immediately construct an unbiased stochastic gradient estimator of
the gradient in (3). Based on this crucial observation, we propose the Two-Sample One-stage
Stochastic Gradient IVaR (TOSG-IVaR) method (Algorithm 1) that avoids explicitly having to
estimate or model the relationship between Z and X thereby overcoming the “forbidden regression”
problem.. Under standard statistical model assumptions, for the case when g is a linear model,
we establish rates of convergence of order O(log T/T) for the proposed method, where T is the
overall number of iterations; see Theorem 1.
• One-sample oracles: In the case when we do not have the aforementioned two-sample oracle,
we estimate the stochastic gradient in (3) by using the streaming data to estimate one of the
conditional expectations, and the corresponding prediction to estimate the other, resulting in the
One-Sample Two-stage Stochastic Gradient IVaR (OTSG-IVaR) method (Algorithm 2). Assuming
further that the X depends linearly on the instrument Z, we establish a rate of convergence of
order O(1/T 1−ι), for any ι > 0; see Theorem 2.

2


---Page Break---
1.1
Literature Review

IVaR analysis. Instrumental variable analysis has a long history, starting from the early works
by [Wri28] and [Rei45]. Several works considered the aforementioned two-stage procedure for
IVaR; a summary could be found in the work by [AP09]. Nonparametric approaches based on
wavelets, splines, reproducing kernels and deep neural networks could be found, for example,
in the works by [HLLBT17, SSG19, BKS19, MMLR20, MZG+21, XCS+21, ZGG+22, PSF24].
Another popular approach for IVaR is via Generalized Method of Moments (GMM); see, for
example, [CP12, BKS19, DLMS20] for an overview. Such approaches essentially reformulate the
problem as a minimax problem and hence suffer from the aforementioned “forbidden regression”
problem.

Identifiability conditions for IVaR. Several works in the literature have also focused on establishing
the identifiability conditions for IVaR in the parametric and the nonparametric setting. Regardless
of the procedure used, they are invariably based on certain source conditions motivated by
the inverse problems literature (see, for example, [CFR07, CR11, BKM+23]) or the related
problem of completeness conditions, which posits that the conditional expectation operator is
one-to-one [BF17, LCY+20]. Semi-parametric identifiability is also considered recently in the work
of [CPS+23]. Our focus in this work is not focused on the identifiability; for the formulation (2) that
we consider, [MMLR20] provide necessary conditions for identifiability that we adopt.

Stochastic optimization with nested expectations. Recently, much attention in the stochastic
optimization literature has focused on optimizing a nested composition of T expectation functions.
Sample average approximation algorithms in this context are considered in the works of [EN13]
and [HCH20].
Optimal iterative stochastic optimization algorithms for the case of T
= 2
were by derived by [GRW20].
For the general T ≥1 case, [WFL17] provided sub-optimal
rates, whereas [BGN22] derived optimal rates; see also [ZX21] and [CSY21] for related works
under stronger assumptions, and [Rus21] for similar asymptotic results. While the above works
required certain independence assumptions regarding the randomness across the different composi-
tions, [HZCH20, HWX+24] studied the case of T = 2 where the the randomness are generically
dependent. They termed this problem setting as conditional stochastic optimization, which is the
framework that the IVaR problem in (2) falls in. Compared to prior works, for e.g., [GRW20]
and [BGN22], in order to handle the dependency between the levels, [HZCH20] require mini-batches
in each iteration, making their algorithm not immediately applicable to the purely streaming setting.
In this work, we show that despite the problem (2) being a conditional stochastic optimization
problem, mini-batches are not required due the additional favorable quadratic structure available in
IVaR.

Streaming IVaR. [VSH+16, DVB24] analyzed streaming versions of 2SLS in the online1 and
adversarial settings. Focusing on linear models, [VSH+16] provide preliminary asymptotic analysis
assuming access to efficient no-regret learners, while [DVB24] provide regret bounds under the
strong assumption that the instrument is almost surely bounded. Furthermore, our algorithms have
significantly improved per-iteration and memory complexity compared to [DVB24]; see Sections A
and B for details. [CLL+23] developed stochastic optimization algorithms for the GMM formulation
and provide asymptotic analysis. Their algorithm requires access to an offline dataset for initialization
and is hence not fully online. The above works (i) do not focus on avoiding the forbidden regression
problem and (ii) do not view IVaR via the conditional stochastic optimization lens, like we do.

2
Two-sample One-stage Stochastic Gradient Method for IVaR

Recall that our goal is to solve the objective function given in (2). By [MMLR20, Theorem 4], the
optimal solution of (2) gives the true underlying causal relationship under the following assumption.

Assumption 2.1. (Identifiability Assumption)

• The conditional distribution PZ|X is continuous in Z for any value of X.

1Their notion of online is from the literature on online learning [SS+12].

3


---Page Break---
Algorithm 1 Two-sample One-stage Stochastic Gradient-IVaR (TOSG-IVaR)
Input: ♯of iterations T, stepsizes {αt}T
t=1, initial iterate θ1.
1: for t = 1 to T do
2:
Sample Zt, sample independently Xt and X′
t from PX|Zt, and sample Yt from PY |Xt.
3:
Update θt
θt+1 = θt −αt+1(g(θt; Xt) −Yt)∇θg(θt; X′
t).

4: end for
Output: θT .

• The function class G := {g(θ; X) | θ ∈Rdθ} is correctly specified, i.e., it includes the true
underlying relationship between X and Y .

Notice that both assumptions are standard in the IVaR literature [NP03, CP12, MMLR20], and
makes the objective in (2) is the meaningful for IVaR. However, [MMLR20] resort to reformulating
the objective function in (2) as a minimax optimization problem as described in Section 1. While
their original motivation was to avoid two-state estimation procedure and avoid the “forbidden
regression”, their minimax reformulation ends up having to solve a complicated approximation of the
original objective resulting in having to characterize the approximation error which is non-trivial.

Algorithm and Analysis. Our aim in this work is to directly solve the original problem in (2),
leveraging the structure provided by the quadratic loss. Given the gradient formulation in (3), a
natural way to build unbiased gradient estimator is to generate X and X′, two independent samples
of X from the conditional distributions PX|Z, for a given realization of Z and generate one sample
of Y from the conditional distribution PY |X. Then, an unbiased gradient estimator is

v(θ) = (g(θ; X) −Y )∇θg(θ; X′).
(4)

This could be plugged into the standard stochasic gradient descent algorithm, which give us the
Two-sample Stochastic Gradient Method for IVR (TSG-IVaR) method illustrated in Algorithm 1. In
particular, the algorithm never requires estimating (or modeling) the relationship between X and Z as
needed in the two-stage procedure [AP09] and the minimax formulation based procedures [MMLR20,
LS18, BKS19, DLMS20, LCY+20, BKM+23]. Furthermore, this viewpoint not only provides a
novel algorithm for performing IV regression, but also provides a novel data collection mechanism
for the practical implementation of IVaR. In addition, such a two-sample gradient method is not very
restrictive when the instrumental variable Z takes value in a discrete set. In this case, to implement
the two-sample oracle, it is enough simply pick two sets of samples (X, Y, Z) and (X′, Y ′, Z) for
which Z has repeated observations (which is possible when Z is a discrete random variable) from a
pre-collected dataset.

Assumption 2.2. The tuple (Zt, Xt, X′
t, Yt) is independent and identically distributed, across t.

To demonstrate the convergence rate of Algorithm 1, we first consider the case when g is a linear
function, i.e., g(θ; X) = X⊤θ. We make the following assumptions.

Assumption 2.3. Suppose there exists µ > 0 such that EZ
h
EX|Z[X] · EX|Z[X]⊤i
⪰µI.

Assumption 2.4. Let (ϑ1, ϑ2, ϑ3, ϑ4) ∈R4
+. For any Z, X′ and X i.i.d. generated from PZ|X , and
Y generated from PY |X. There exists constants Cx, Cy, Cxx, Cyx > 0 such that

E
h
∥X′X⊤−EX|Z[X]EX|Z[X]⊤∥2i
≤Cxdϑ1
x ,
(5)

E
h
∥Y X′ −EY |Z[Y ]EX|Z[X]∥2i
≤Cydϑ2
x ,
(6)

E
h
∥EX|Z[X] · EX|Z[X]⊤−EZ
h
EX|Z[X] · EX|Z[X]⊤i
∥2i
≤Cxxdϑ3
z ,
(7)

E
h
∥EY |Z[Y ] · EX|Z[X] −EZ
h
EY |Z[Y ] · EX|Z[X]
i
∥2i
≤Cyxdϑ4
z ,
(8)

where ∥· ∥denotes the Euclidean norm and operator norm for a vector and matrix respectively.

4


---Page Break---
The above assumptions are mild moment assumptions required on the involved random variables.
The following result demonstrates that Assumptions 2.3 and 2.4 are naturally satisfied even under
non-linear modeling assumption on (1). We defer its proof to Section D.3.
Lemma 1. Suppose there exist θ∗∈Rdx, γ∗∈Rdz×dx, a non-linear map ϕ : Rdx →Rdx, and a
positive semi-definite matrix Σ ∈Rdz×dz such that

EZ
h
ϕ(γ⊤
∗Z) · ϕ(γ⊤
∗Z)⊤i
⪰µI, E[∥ϕ(γ⊤
∗Z)∥2] = O(dx),

Z ∼N(0, Σ), X = ϕ(γ⊤
∗Z) + ϵ2, Y = θ⊤
∗X + ϵ1, ϵ2 ∼N(0, σ2
ϵ2Idx), ϵ1 ∼N(0, σ2
ϵ1),
(9)

where ϵ1, ϵ2 are independent of Z and

E

ϵ2
1∥ϵ2∥2
≤σ2
ϵ1,ϵ2dx, E

∥ϕ(γ⊤
∗Z) · ϕ(γ⊤
∗Z)⊤−E[ϕ(γ⊤
∗Z) · ϕ(γ⊤
∗Z)⊤]∥2
≤Cdz.
(10)

Then Assumptions 2.3 and 2.4 hold with ϑ1 = ϑ2 = 2 and ϑ3 = ϑ4 = 1. If ϕ is an identity map, then
the conditions involving ϕ become γ⊤
∗Σγ∗⪰µI, tr(γ⊤
∗Σγ∗) = O(dx), E

∥ZZ⊤−Σ∥2
≤Cdz.

The above assumption is standard in the stochastic approximation, statistics and econometrics litera-
ture. It could be further relaxed to Markovian-type dependency assumptions, following techniques
in the works of [DAJJ12, SSY18, Eve23, RBG22]; we leave a detailed examination of the Marko-
vian streaming setup as future work. Under the above assumptions, we have the following result
demonstrating the last-iterate global convergence of Algorithm 1.

Theorem 1. Suppose Assumptions 2.3, 2.4, and 2.2 hold. In Algorithm 1, defining σ2
1 := 2Cxdϑ1
x +
2Cxxdϑ3
z and σ2
2 := Cydϑ2
x + Cyxdϑ4
z , set αt ≡α = log T

µT
≤
µ
µ2+3σ2
1 . Then, we have

E

∥θT −θ∗∥2
≤E

∥θ0 −θ∗∥2

T
+ 3∥θ∗∥2(σ2
1 + σ2
2) log T
µ2T
.

Proof techniques. In the analysis of Theorem 1, the following decomposition (see (18) for the
derivation) plays a crucial role:

θt+1 −θ∗= At + αt+1Bt,

At = θt −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt + αt+1EZ
h
EY |Z[Y ] · EX|Z[X]
i
−θ∗,

Bt = −

X′
tX⊤
t −EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt +

YtX′
t −EZ
h
EY |Z[Y ] · EX|Z[X]
i
,

where At corresponds to deterministic component, and Bt corresponds to the stochastic component
arising due to the use of stochastic gradients. Standard assumptions on the variance of the stochastic
gradient made in the stochastic optimization literature include the uniformly bounded variance
assumption [Lan20] and the expected smoothness condition [KR20]. In the IVaR setup, such
standard assumptions do not hold as θt potentially can be unbounded and thus the gradient estimator
can be unbounded. Hence, we establish our results under natural statistical assumptions arising in the
context of the IVaR problem, which form the main novelty in our analysis. Furthermore, compared
to [MMLR20], notice that we use two samples of X from the conditional distribution PX|Z and
achieve an e
O(1/T) last iterate convergence rate to the global optimal solution, which is the true
underlying causal relationship under Assumption 2.1. In comparison, [MMLR20] only provide
asymptotic convergence result to the optimal solution of an approximation problem.

Additional discussion. It is interesting to explore other losses beyond squared loss (for example to
handle classification setting [CF21]), potentially using the Multilevel Monte Carlo (MLMC) based
stochastic gradient estimators. While [HCH21], develops such algorithms, the main challenge is about
how to avoid mini-batches required in their work leveraging the problem structure in instrumental
variable analysis. Furthermore, in the case when g(θ; X) is parametrized by a non-linear models,
for instance, a neural network, we provide local convergence guarantees under additional stronger
conditions made typically in the stochastic optimization literature.

Assumption 2.5. Let the following assumptions hold:

• Function F(θ) is ℓ-smooth.

5


---Page Break---
• The iterates {θt}T +1
t=1 generated by Algorithm 1 are in a compact set A.
• The random objects X|Z and Y |Z have bounded variance for any Z, i.e., there exist σ > 0 such
that

E

∥X −E [X | Z] ∥2 | Z

≤σ2, E

∥Y −E [Y | Z] ∥2 | Z

≤σ2.

Proposition 1. Suppose Assumptions 2.1, 2.2, and 2.5 hold. Choosing αt ≡α = O

1
√

T


, for
Algorithm 1 we have

min
1≤t≤T E

∥∇F(θt)∥2
= O
 1
√

T


.

The proof of the proposition is immediate. Note that under Assumption 2.5, we can deduce that the
unbiased gradient estimator v(θ) = (g(θ; X) −Y )∇θg(θ; X′) has a bounded variance since

Var(v(θ)) =Var(g(θ; X) −Y )Var(∇θg(θ; X′))

+ Var(g(θ; X) −Y ) (E [∇θg(θ; X′)])2 + Var(∇θg(θ; X′)) (E [g(θ; X) −Y ])2 ≤σ2
v,

where the variance and expectation are taken conditioning on Z and θ, and σv > 0 is a constant that
only depends on σ, function g and the compact set A in Assumption 2.5. Then one can directly follow
the analysis of non-convex stochastic optimization (see, for example, [GL13, Theorem 2.1]) to obtain
Proposition 1. Relaxing the Assumption 2.5 (typically made in the stochastic optimization literature)
with more natural assumptions on the statistical model and obtaining a result as in Theorem 1 for the
non-convex setting is left as future work.

3
One-sample Two-stage Stochastic Gradient Method for IVaR

We now examine designing streaming IVaR algorithm with access to the classical one-sample oracle,
i.e., we observe a streaming set of samples (Xt, Yt, Zt) at each time point t. Note that in this case,
using the same Xt (instead of X′
t) in (4) makes the stochastic gradient estimator biased.

Intuition. Consider the case of linear models, i.e., Y = θ⊤
∗X + ϵ1 with X = γ⊤
∗Z + ϵ2, where
θ∗∈Rdx×1, and γ∗∈Rdz×dx, as also considered in Lemma 1. Recall the true gradient in (3) and
the stochastic gradient estimator of Algorithm 1 in (4). Since we no longer have X′
t, we replace the
term X′
t with the predicted mean of Xt given Zt. Suppose that γ∗is known. We specifically replace
∇θtg(θt; X′
t) = X′
t by E|Zt [Xt] = γ⊤
∗Zt. In such a case, indeed we have an unbiased gradient
estimator:

Et

γ⊤
∗Zt(X⊤
t θt −Yt)

= Et
h
E|Zt [Xt] (E|Zt [Xt]⊤θt −E|Zt [Yt])
i

=Et

γ⊤
∗ZtZ⊤
t γ∗(θt −θ∗)

= γ⊤
∗ΣZγ∗(θt −θ∗) = ∇θF(θt),

where Et [·] is the conditional expectation w.r.t the filtration defined on {γ1, θ1, γ2, θ2, · · · , γt, θt}.

In reality, γ∗is unknown beforehand. Hence, we estimate γ∗using some online procedure and replace
∇θtg(θt; X′
t) by γt⊤Zt instead of γ⊤
∗Zt. It leads to the following updates:

θt+1 = θt −αt+1γ⊤
t Zt(X⊤
t θt −Yt),
γt+1 = γt −βt+1Zt(Z⊤
t γt −X⊤
t ).
(11)

A closer inspection reveals that the updates in (11) can diverge until γt is close enough to γ∗. It is
easy to see this fact from the following expansion of θt+1 −θ∗. We have

θt+1 −θ∗= bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY + αt+1Dtθ∗+ αt+1γt
⊤ξZtγ∗(θt −θ∗)

+ αt+1γt
⊤ξZtγ∗θ∗+ αt+1γt
⊤ξZtYt −αt+1γt
⊤Ztϵ2,t
⊤θt,
(12)

where

ξZt = ΣZ −ZtZ⊤
t ,
ξZtYt = ΣZY −ZtYt,
bQt :=
 
I −αt+1γt
⊤ΣZγ∗

.

However, the matrix γt⊤ΣZγ∗may not be positive semi-definite, even if ΣZ is positive definite. Thus
the negative eigenvalues associated with γt⊤ΣZγ∗might cause the θt iterates to first diverge, before

6


---Page Break---
0
1
2
3
4
5
log (t)

6

4

2

0

2

4

6

8

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)

Figure 1: (11) can initially diverge before converging eventually, leading to a worse performance in
practical settings compared to Algorithm 2. See Appendix C.2 for the experimental setup.
.

Algorithm 2 One-Sample Two-stage Stochastic Gradient-IVarR (OTSG-IVaR)
Input: Stepsizes {αt}t, {βt}t, initial iterates γ1, θ1.

1: for t = 1, 2, · · · do
2:
Sample Zt, sample Xt from PX|Zt, Sample Yt from PY |Xt.
3:
Update

θt+1 = θt −αt+1γ⊤
t Zt(Z⊤
t γtθt −Yt),
(13)

γt+1 = γt −βt+1Zt(Z⊤
t γt −X⊤
t ).
(14)

4: end for

eventually converging as γt gets closer to γ∗. We illustrate this intuition in a simple experiment in
Figure 1. To resolve this issue, we propose Algorithm 2, where we replace g(θt, Xt) = X⊤
t θt with
ZT
t γtθt in (11). With such a modification, in the corresponding decomposition for θt+1 −θ∗(see
(40)), we have bQt =
 
I −αt+1γt⊤ΣZγt

, where the matrix product γt⊤ΣZγt is always positive
semi-definite. Hence, with a properly chosen stepsize αt we could quantify the convergence of θt
to θ∗non-asymptotically. Nevertheless, assuming a warm-start condition on θ0, we also show the
convergence of (11), in Appendix E.3 for completeness.

Algorithm and Analysis. Based on the intuition, we present Algorithm 2. One could interpret the
algorithm as the SGD analogy of the offline 2SLS algorithm [AI95]. It is also related to the framework
of non-linear two-stage stochastic approximation algorithms [DR20, DTSM18, MP06]; albeit the
updates of θt and γt are coupled since both updates use Zt. Furthermore, the dependency between the
randomness between the two stages in the IVaR problem, makes the analysis significantly different
and more challenging from the classical analysis of two-stage algorithms (see below Theorem 2 for
additional details). Finally, while Algorithm 2 is designed for linear models, the intuition behind the
method is also applicable to non-linear models (i.e., between Z and X, and X and Y ). We focus
on linear models in this work in order to derive our theoretical results. A detailed treatment of the
nonlinear case (for which the analysis is significantly nontrivial) is left for future work. We make the
following additional assumptions for the convergence analysis of Algorithm 2.
Assumption 3.1. For some constants Cz, Czy > 0, we have the following bounds on the fourth
moments:

E

∥ΣZ −ZZ⊤∥4
≤Czdϑ5
z ,
E

∥ΣZY −ZY ∥4
≤Czydϑ6
z ,
ϑ := max{ϑ5, ϑ6}.
(15)

Assumption 3.2. There exist constants 0 < µZ ≤λZ < ∞such that µZIdz ⪯ΣZ ⪯λZIdz.

The above conditions are rather mild moment conditions, similar to Assumption 2.4, and could be
easily verified for the linear model setting we consider.
Assumption 3.3. {γt}t is within a compact set of diameter Cγdκ
z for some constants Cγ > 0, κ ≥0.

7


---Page Break---
We emphasize that Assumption 3.3 is only for the uncoupled sequence γt, which is an SGD sequence
for solving a strongly-convex problem. It holds easily in various cases, for example by projecting
the iterates onto any compact sets or a sufficiently large ball containing γ∗. It is also well-known
that, without any projection operations, {γt}t sequence is almost surely bounded [PJ92] under our
assumptions. Finally, similar assumptions routinely appear in the analysis of SGD algorithms in
various related settings; see, for example, [Tse98, GOP19, HS19, NJN19, AYS20, RGP20].

We now present our result on the convergence of {θt}t below in Theorem 2 (see Appendix E.1 for the
proof). In comparison to Theorem 1 (regarding Algorithm 1), we highlight that Theorem 2 provides
an any-time guarantee, as the total number of iterations is not required in advance by Algorithm 2.
Theorem 2. Suppose Assumptions 2.3, 2.2 (without X′
t), 3.1, 3.3, and 3.2 hold.
In Al-
gorithm 2, for any ι
>
0, set αt
=
Cαt−1+ι/2 and βt
=
Cβt−1+ι/2, where Cα
=
min{0.5d−4κ−ϑ/2
z
λ−1
Z C−2
γ , 0.5(∥γ∗∥λZ)−2}, and Cβ = µ2d−1−2κ
z
/128. Then, we have

E

∥θt −θ∗∥2
= O
 1

t1−ι


.

Remark 1. In Theorem 2, we present the step-size choices for the fastest rate of convergence. In
the proof of Theorem 2 (see Appendix E.1), we show that convergence can be guaranteed for a
range of step-sizes given by αt = Cαt−a, βt = Cβt−b, where 1/2 < a, b < 1, b > 2 −2a
with corresponding rate being E

∥θt −θ∗∥2
= O(max{t−b(2−(1−ι/2)−1), t−a log(2/ι −1)}). In
particular, one requires a, b < 1 to ensure (αt −αt+1)/αt = o(αt), and (βt −βt+1)/βt = o(βt),
as is standard in stochastic approximation literature (see, for example, [CLTZ20, PJ92]).

Proof Techniques. The major challenge towards the convergence analysis of {θt}t lies in the
interaction term γtZtZ⊤
t γtθt between γt and θt in (13). This multiplicative interaction term leads to
an involved dependence between the noise in the stochastic gradient updates for the two stages. Such
a dependence has not been considered in existing analysis of non-linear two time-scale algorithms
[MP06, MSB+09, DTSM18, DR20, XL21, WZZ21, Doa22]. In addition, [Doa22] considers the
case when the noise sequence is not only independent of each other but also independent of iterate
locations. Furthermore, they assumes (see their Assumption 3) that the condition in Assumption 2.3
holds for all γ whereas Assumption 2.3 only needs to hold for γ∗, that is much milder. Similarly,
many works (for example, Assumption 1 in [WZZ21], Assumption 2 in [XL21] and Theorem 2 in
[MSB+09]) assume that the iterates of both stages are bounded in a compact set and consequently,
and hence the variance of the stochastic gradients are also uniformly bounded.

In our setting, firstly, the stochastic gradient in (13), evaluated at (θt, γt) is biased:

Et,Zt

γt
⊤Zt(Z⊤
t γtθt −Yt)

= Et,Zt

γt
⊤Zt(Z⊤
t γtθt −Z⊤
t γ∗θ∗)

= Et

γt
⊤ΣZ(γtθt −γ∗θ∗)


=γt
⊤ΣZγt(θt −θ∗) + γt
⊤ΣZ(γt −γ∗)θ∗̸= γ⊤
∗ΣZγ∗(θt −θ∗) = ∇θF(θt).

Furthermore, even under Assumption 3.3, the variance of the stochastic gradient is not (13) uniformly
bounded. Overcoming these issues, in addition to the aforementioned dependence between the noise
in the stochastic gradient updates for the two stages, forms the major novelty in our analysis. We
proceed by noting that if γ∗, ΣZ, and ΣZY were known beforehand, one conduct deterministic
gradient updates, i.e., eθt+1 = eθt −αt+1γ⊤
∗

ΣZγ∗eθt −ΣZY

, to obtain θ∗. By standard results

on gradient descent for strongly convex functions (see, for example, [Nes13]), {eθt}t converges
exponentially fast as stated in Lemma 4. Hence, it remains to show that the trajectory of θt converges
to the trajectory of eθt. That is, defining the sequence δt := θt −eθt, our goal is to establish the
convergence rate of E

∥δt∥2
2

. We first provide an intermediate bound (see Lemma 6) and then
progressively sharpen to a tighter bound (see Lemma 7). In doing so, it is also required to show that
E

∥θt∥4
is bounded, which we prove in Lemma 5. The proof of Lemma 5 is non-trivial and requires
carefully chosen stepsizes satisfying P∞
t=1(α2
t + αt
√βt) < ∞.

4
Numerical Experiments

Experiments for Algorithm 1 (TOSG-IVaR). We first consider the following problem, in which
(Z, X, Y ) is generated via

Z ∼N(0, Idz), X = ϕ(γ⊤
∗Z) + c · (h + ϵx), Y = θ⊤
∗X + c · (h1 + ϵy),

8


---Page Break---
0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(a)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(b)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(c)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(d)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(e)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(f)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(g)

0
200
400
600
800
1000
t

0

2

4

6

8

10

*
2

TOSG-IVaR

(h)

Figure 2: E[∥θt −θ∗∥2] of Algorithm 1 under different settings detailed in Section 4.

where c > 0 is a scalar to control the variance of the noise vector, and h1 is the first coordinate
of h. The noise vectors (or scalar) h, ϵx, ϵy are independent of Z, and we have h ∼N(1dx, Idx),
ϵx ∼N(0, Idx),ϵy ∼N(0, 1). In each iteration, one tuple (X, X′, Y ) is generated and used to update
θt according to Algorithm 1. We set (dx, dz) ∈{(4, 8), (8, 16)}, c ∈{0.1, 1.0}, and ϕ(s) ∈{s, s2}.
We repeat each setting 50 times and report the curves of E[∥θt −θ∗∥2] in Figure 2, where the
expectation is computed as the average of ∥θt −θ∗∥2 of all trials, and the shaded region represents
the standard deviation. The first row and the second row correspond to ϕ(s) = s and ϕ(s) = s2
respectively. Here, c = 0.1 for odd columns and c = 1.0 for even columns. We have (dx, dz) = (4, 8)
for the first two columns and (dx, dz) = (8, 16) for the last two columns. Empirically, we can observe
that our Algorithm 1 performs well across all different settings.

Experiments for Algorithm 2 (OTSG-IVaR). Next, we compare our Algorithm 2 as well as its
variant and Algorithm 1 in [DVB24]. We write “OTSG-IVaR”, “CSO – Eq. (11)” and “[DVB23]” to
represent Algorithm 2, Algorithm 2 with the updates replaced by (11) and Algorithm 1 in [DVB24]
(see Appendix A). We follow simulation settings similar to [DVB24]:

Y = θ⊤
∗X + ν,
X = γ⊤
∗Z + ϵ,
ϵ = σϵN(0, Idx),
ν = ρϵ1 + N(0, 0.25),
(16)

where ϵ1 is the first coordinate of ϵ, θ∗∈Rdx is a unit vector chosen uniformly randomly, and
γ∗∈Rdz×dx where γij = 0 for i ̸= j, and γij = 1 for i = j, i = 1, 2, · · · , dx, and j = 1, 2, · · · , dz.
Here ρ controls the level of endogeneity in the model. We compare the performance of Algorithm 2
with (11), and O2SLS [DVB24] for ρ = 1, 4, and σϵ = 0.5, 1. By varying σϵ we control the
correlation between X and Z. We consider two settings (dx, dz) = (1, 1), and (dx, dz) = (8, 16).
As performance metric, in Figure 3 we plot E

∥θt −θ∗∥2
where the E [·] is approximated by
averaging over 50 trials, and both axes are in log scale (base 10). We also show, in Figure 4, the
convergence of the test Mean Squared Error (MSE) evaluated over 400 test samples to the best
possible test MSE where θ∗and γ∗are known beforehand. For Figures 3 and 4, the first row and
second row corresponds to (dx, dz) = (1, 1) and (dx, dz) = (8, 16) respectively, and σϵ = 0.5 in
odd columns and σϵ = 1.0 in even columns. We have ρ = 1.0 for the first two columns and ρ = 4.0
for the last two columns. We can observe that O2SLS has much larger variance in different settings,
while our algorithms perform consistently well in all settings. We further conduct experiments on
real-world datasets provided in [AE96] and [Rya12]. Due to space limit, we include the numerical
results in Section C.3 of the Appendix.

5
Conclusion

We presented streaming algorithms for least-squares IVaR based on directly solving the associated
conditional stochastic optimization formulation in (2). Our algorithms have several benefits, including
avoidance of mini-batches and matrix inverses. We show that the expected rates of convergences
for the proposed algorithms are of order O(log T/T) and O(1/T 1−ι), for any ι > 0, under the

9


---Page Break---
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

5

4

3

2

1

0

1

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(a)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

5

4

3

2

1

0

1

2

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(b)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

5

4

3

2

1

0

1

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(c)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

5

4

3

2

1

0

1

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(d)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

5

4

3

2

1

0

1

2

3

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(e)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

4

2

0

2

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(f)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

4

2

0

2

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(g)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

4

2

0

2

4

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(h)

Figure 3: Comparison of E[∥θt −θ∗∥2] (log-log scale) for Algorithm 2, Eq. 11 and [DVB24].

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

0

1

2

3

4

5

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(a)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

1

2

3

4

5

6

7

8

9

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(b)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

2

3

4

5

6

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(c)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

3

4

5

6

7

8

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(d)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

0

2

4

6

8

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(e)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

1

2

3

4

5

6

7

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(f)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

2

3

4

5

6

7

8

9

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(g)

0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
log (t)

3

4

5

6

7

8

9

log(Test MSE)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]
Offline

(h)

Figure 4: Comparison of test MSE (log-log scale) for Algorithm 2, Eq. 11 and [DVB24].

availability of two-sample and one-sample oracles, respectively. As future work, it is interesting to
develop streaming inferential methods for IVaR. Leveraging related works for the vanilla SGD [PJ92,
ABE19, SZ22, CLTZ20, ZCW23] to the setting of Algorithms 1 and 2, provides a concrete direction
to establish Central Limit Theorems and develop limiting covariance estimation procedures.

Acknowledgments and Disclosure of Funding

YH was supported as a part of NCCR Automation, a National Centre of Competence (or Excellence)
in Research, funded by the Swiss National Science Foundation (grant number 51NF40_225155). KB
was supported in part by NSF grants DMS-2053918 and DMS-2413426.

References

[ABE19] Andreas Anastasiou, Krishnakumar Balasubramanian, and Murat A Erdogdu. Normal
approximation for stochastic gradient descent via non-asymptotic rates of martingale clt.
In Conference on Learning Theory, pages 115–137. PMLR, 2019. (Cited on page 10.)

[AE96] Joshua Angrist and William N Evans. Children and their parents’ labor supply: Evidence
from exogenous variation in family size, 1996. (Cited on pages 9 and 16.)

10


---Page Break---
[AI95] Joshua D Angrist and Guido W Imbens. Two-stage least squares estimation of average
causal effects in models with variable treatment intensity. Journal of the American
statistical Association, 90(430):431–442, 1995. (Cited on pages 2 and 7.)

[AP09] Joshua D Angrist and Jörn-Steffen Pischke. Mostly harmless econometrics: An empiri-
cist’s companion. Princeton university press, 2009. (Cited on pages 2, 3, and 4.)

[AYS20] Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. SGD with shuffling: optimal rates without
component convexity and large epoch requirements. Advances in Neural Information
Processing Systems, 33:17526–17535, 2020. (Cited on page 8.)

[BF17] Andrii Babii and Jean-Pierre Florens. Is completeness necessary? estimation in non-
identified linear models. arXiv preprint arXiv:1709.03473, 2017. (Cited on page 3.)

[BGN22] Krishnakumar Balasubramanian, Saeed Ghadimi, and Anthony Nguyen. Stochastic
multilevel composition optimization algorithms with level-independent convergence
rates. SIAM Journal on Optimization, 32(2):519–544, 2022. (Cited on page 3.)

[BKM+23] Andrew Bennett, Nathan Kallus, Xiaojie Mao, Whitney Newey, Vasilis Syrgkanis, and
Masatoshi Uehara. Minimax instrumental variable regression and l_2 convergence
guarantees without identification or closedness. arXiv preprint arXiv:2302.05404, 2023.
(Cited on pages 2, 3, and 4.)

[BKS19] Andrew Bennett, Nathan Kallus, and Tobias Schnabel. Deep generalized method of
moments for instrumental variable analysis. Advances in neural information processing
systems, 32, 2019. (Cited on pages 2, 3, and 4.)

[Bol12] Kenneth A Bollen. Instrumental variables in sociology and the social sciences. Annual
review of sociology, 38(1):37–72, 2012. (Cited on page 1.)

[CF21] Samuele Centorrino and Jean-Pierre Florens. Nonparametric instrumental variable
estimation of binary response models with continuous endogenous regressors. Econo-
metrics and Statistics, 17:35–63, 2021. (Cited on page 5.)

[CFR07] Marine Carrasco, Jean-Pierre Florens, and Eric Renault. Linear inverse problems in
structural econometrics estimation based on spectral decomposition and regularization.
Handbook of econometrics, 6:5633–5751, 2007. (Cited on page 3.)

[CLL+23] Xiaohong Chen, Sokbae Lee, Yuan Liao, Myung Hwan Seo, Youngki Shin, and
Myunghyun Song. SGMM: Stochastic approximation to generalized method of mo-
ments. arXiv preprint arXiv:2308.13564, 2023. (Cited on pages 3 and 16.)

[CLTZ20] Xi Chen, Jason D Lee, Xin T Tong, and Yichen Zhang. Statistical inference for model
parameters in stochastic gradient descent. Annals of Statistics, 48(1):251–273, 2020.
(Cited on pages 8, 10, and 20.)

[CP12] Xiaohong Chen and Demian Pouzo. Estimation of nonparametric conditional moment
models with possibly nonsmooth generalized residuals. Econometrica, 80(1):277–321,
2012. (Cited on pages 3 and 4.)

[CPS+23] Yifan Cui, Hongming Pu, Xu Shi, Wang Miao, and Eric Tchetgen Tchetgen. Semi-
parametric proximal causal inference. Journal of the American Statistical Association,
pages 1–12, 2023. (Cited on page 3.)

[CR11] Xiaohong Chen and Markus Reiss. On rate optimality for ill-posed inverse problems in
econometrics. Econometric Theory, 27(3):497–521, 2011. (Cited on page 3.)

[CSY21] Tianyi Chen, Yuejiao Sun, and Wotao Yin. Solving stochastic compositional optimiza-
tion is nearly as easy as solving stochastic optimization. IEEE Transactions on Signal
Processing, 69:4937–4948, 2021. (Cited on page 3.)

[DAJJ12] John C Duchi, Alekh Agarwal, Mikael Johansson, and Michael I Jordan. Ergodic mirror
descent. SIAM Journal on Optimization, 22(4):1549–1578, 2012. (Cited on page 5.)

[DFFR11] Serge Darolles, Yanqin Fan, Jean-Pierre Florens, and Eric Renault. Nonparametric
instrumental regression. Econometrica, 79(5):1541–1565, 2011. (Cited on page 2.)

[DHP+17] Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song. Learning from conditional
distributions via dual embeddings. In Artificial Intelligence and Statistics, pages 1458–
1467. PMLR, 2017. (Cited on page 2.)

11


---Page Break---
[DLMS20] Nishanth Dikkala, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. Minimax esti-
mation of conditional moment models. Advances in Neural Information Processing
Systems, 33:12248–12262, 2020. (Cited on pages 2, 3, and 4.)

[Doa22] Thinh T Doan. Nonlinear two-time-scale stochastic approximation convergence and
finite-time performance. IEEE Transactions on Automatic Control, 2022. (Cited on
page 8.)

[DR20] Thinh Doan and Justin Romberg. Finite-time performance of distributed two-time-scale
stochastic approximation. In Learning for Dynamics and Control, pages 26–36. PMLR,
2020. (Cited on pages 7 and 8.)

[DTSM18] Gal Dalal, Gugan Thoppe, Balázs Szörényi, and Shie Mannor. Finite sample analysis
of two-timescale stochastic approximation with applications to reinforcement learning.
In Conference On Learning Theory, pages 1199–1233. PMLR, 2018. (Cited on pages 7
and 8.)

[DVB23] Riccardo Della Vecchia and Debabrota Basu. Online instrumental variable regression:
Regret analysis and bandit feedback. arXiv preprint arXiv:2302.09357v1, 2023. (Cited
on pages 15 and 16.)

[DVB24] Riccardo Della Vecchia and Debabrota Basu. Stochastic online instrumental vari-
able regression: Regrets for endogeneity and bandit feedback.
arXiv preprint
arXiv:2302.09357v3, 2024. (Cited on pages 3, 9, 10, 15, 16, and 17.)

[EN13] Yuri M Ermoliev and Vladimir I Norkin. Sample average approximation method
for compound stochastic optimization problems.
SIAM Journal on Optimization,
23(4):2231–2263, 2013. (Cited on page 3.)

[Eve23] Mathieu Even. Stochastic gradient descent under Markovian sampling schemes. In
International Conference on Machine Learning, pages 9412–9439. PMLR, 2023. (Cited
on page 5.)

[GL13] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for
nonconvex stochastic programming. SIAM journal on optimization, 23(4):2341–2368,
2013. (Cited on page 6.)

[GOP19] Mert Gurbuzbalaban, Asu Ozdaglar, and Pablo A Parrilo. Convergence rate of incre-
mental gradient and incremental newton methods. SIAM Journal on Optimization,
29(4):2542–2565, 2019. (Cited on page 8.)

[GRW20] Saeed Ghadimi, Andrzej Ruszczynski, and Mengdi Wang. A single timescale stochastic
approximation method for nested stochastic optimization. SIAM Journal on Optimiza-
tion, 30(1):960–979, 2020. (Cited on page 3.)

[HCH20] Yifan Hu, Xin Chen, and Niao He. Sample complexity of sample average approximation
for conditional stochastic optimization. SIAM Journal on Optimization, 30(3):2103–
2133, 2020. (Cited on page 3.)

[HCH21] Yifan Hu, Xin Chen, and Niao He. On the bias-variance-cost tradeoff of stochastic
optimization. Advances in Neural Information Processing Systems, 34:22119–22131,
2021. (Cited on page 5.)

[HH05] Peter Hall and Joel L Horowitz. Nonparametric methods for inference in the presence
of instrumental variables. Annals of statistics, 33(6):2904–2929, 2005. (Cited on page 2.)

[HLLBT17] Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. Deep iv: A
flexible approach for counterfactual prediction. In Proceedings of the 34th International
Conference on Machine Learning-Volume 70, pages 1414–1423. JMLR, 2017. (Cited on
pages 2 and 3.)

[HS19] Jeff Haochen and Suvrit Sra. Random shuffling beats SGD after finite epochs. In
International Conference on Machine Learning, pages 2624–2633. PMLR, 2019. (Cited
on page 8.)

[HWX+24] Yifan Hu, Jie Wang, Yao Xie, Andreas Krause, and Daniel Kuhn. Contextual stochastic
bilevel optimization. Advances in Neural Information Processing Systems, 36, 2024.
(Cited on page 3.)

12


---Page Break---
[HZCH20] Yifan Hu, Siqi Zhang, Xin Chen, and Niao He. Biased stochastic first-order methods
for conditional stochastic optimization and applications in meta learning. Advances in
Neural Information Processing Systems, 33:2759–2770, 2020. (Cited on pages 2 and 3.)
[KR20] Ahmed Khaled and Peter Richtárik. Better theory for SGD in the nonconvex world.
arXiv preprint arXiv:2002.03329, 2020. (Cited on page 5.)
[Lan20] Guanghui Lan. First-order and stochastic optimization methods for machine learning,
volume 1. Springer, 2020. (Cited on page 5.)
[LCY+20] Luofeng Liao, You-Lin Chen, Zhuoran Yang, Bo Dai, Mladen Kolar, and Zhaoran
Wang. Provably efficient neural estimation of structural equation models: An adversarial
approach. Advances in Neural Information Processing Systems, 33:8947–8958, 2020.
(Cited on pages 2, 3, and 4.)
[LS18] Greg Lewis and Vasilis Syrgkanis. Adversarial generalized method of moments. arXiv
preprint arXiv:1803.07164, 2018. (Cited on pages 2 and 4.)
[MMLR20] Krikamol Muandet, Arash Mehrjou, Si Kai Lee, and Anant Raj. Dual instrumental
variable regression. Advances in Neural Information Processing Systems, 33:2710–2721,
2020. (Cited on pages 2, 3, 4, and 5.)
[MP06] Abdelkader Mokkadem and Mariane Pelletier. Convergence rate and averaging of
nonlinear two-time-scale stochastic approximation algorithms. Annals of Applied
Probability, 16(3):1671–1702, 2006. (Cited on pages 7 and 8.)
[MSB+09] Hamid Maei, Csaba Szepesvari, Shalabh Bhatnagar, Doina Precup, David Silver, and
Richard S Sutton. Convergent temporal-difference learning with arbitrary smooth
function approximation. Advances in neural information processing systems, 22, 2009.
(Cited on page 8.)
[MZG+21] Afsaneh Mastouri, Yuchen Zhu, Limor Gultchin, Anna Korba, Ricardo Silva, Matt
Kusner, Arthur Gretton, and Krikamol Muandet. Proximal causal learning with kernels:
Two-stage estimation and moment restriction. In International conference on machine
learning, pages 7512–7523. PMLR, 2021. (Cited on page 3.)
[Nes13] Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87.
Springer Science & Business Media, 2013. (Cited on page 8.)
[NJN19] Dheeraj Nagaraj, Prateek Jain, and Praneeth Netrapalli. SGD without replacement:
Sharper rates for general smooth convex functions. In International Conference on
Machine Learning, pages 4703–4711. PMLR, 2019. (Cited on page 8.)
[NP03] Whitney K Newey and James L Powell. Instrumental variable estimation of nonpara-
metric models. Econometrica, 71(5):1565–1578, 2003. (Cited on page 4.)
[Pap03] Christos H Papadimitriou. Computational complexity. In Encyclopedia of computer
science, pages 260–265. 2003. (Cited on page 15.)
[PJ92] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by
averaging. SIAM journal on control and optimization, 30(4):838–855, 1992. (Cited on
pages 8, 10, and 26.)
[PSF24] Caio Peixoto, Yuri Saporito, and Yuri Fonseca. Nonparametric instrumental variable
regression through stochastic approximate gradients. arXiv preprint arXiv:2402.05639,
2024. (Cited on page 3.)
[RBG22] Abhishek Roy, Krishnakumar Balasubramanian, and Saeed Ghadimi. Constrained
stochastic nonconvex optimization with state-dependent Markov data. Advances in
Neural Information Processing Systems, 35:23256–23270, 2022. (Cited on page 5.)
[Rei45] Olav Reiersøl. Confluence analysis by means of instrumental sets of variables. PhD
thesis, Almqvist & Wiksell, 1945. (Cited on page 3.)
[RGP20] Shashank Rajput, Anant Gupta, and Dimitris Papailiopoulos. Closing the convergence
gap of SGD without replacement. In International Conference on Machine Learning,
pages 7964–7973. PMLR, 2020. (Cited on page 8.)
[Rus21] Andrzej Ruszczynski. A stochastic subgradient method for nonsmooth nonconvex
multilevel composition optimization. SIAM Journal on Control and Optimization,
59(3):2301–2320, 2021. (Cited on page 3.)

13


---Page Break---
[Rya12] Stephen P Ryan. The costs of environmental regulation in a concentrated industry.
Econometrica, 80(3):1019–1061, 2012. (Cited on pages 9 and 16.)
[SS+12] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations
and Trends® in Machine Learning, 4(2):107–194, 2012. (Cited on page 3.)
[SSG19] Rahul Singh, Maneesh Sahani, and Arthur Gretton. Kernel instrumental variable
regression. Advances in Neural Information Processing Systems, 32, 2019. (Cited on
page 3.)
[SSY18] Tao Sun, Yuejiao Sun, and Wotao Yin. On Markov chain gradient descent. Advances in
neural information processing systems, 31, 2018. (Cited on page 5.)
[SZ22] Qi-Man Shao and Zhuo-Song Zhang. Berry–esseen bounds for multivariate nonlinear
statistics with applications to m-estimators and stochastic gradient descent algorithms.
Bernoulli, 28(3):1548–1576, 2022. (Cited on page 10.)
[TM17] Ambuj Tewari and Susan A Murphy. From ads to interventions: Contextual bandits
in mobile health. Mobile health: sensors, analytic methods, and applications, pages
495–517, 2017. (Cited on page 1.)
[Tse98] Paul Tseng. An incremental gradient (-projection) method with momentum term and
adaptive stepsize rule. SIAM Journal on Optimization, 8(2):506–531, 1998. (Cited on
page 8.)
[VSH+16] Arun Venkatraman, Wen Sun, Martial Hebert, J Bagnell, and Byron Boots. Online
instrumental variable regression with applications to online linear system identification.
In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.
(Cited on page 3.)
[WFL17] M. Wang, E. X. Fang, and B. Liu. Stochastic compositional gradient descent: Al-
gorithms for minimizing compositions of expected-value functions. Mathematical
Programming, 161(1-2):419–449, 2017. (Cited on page 3.)
[Wri28] Philip Green Wright. The tariff on animal and vegetable oils. Number 26. Macmillan,
1928. (Cited on page 3.)
[WZZ21] Yue Wang, Shaofeng Zou, and Yi Zhou. Non-asymptotic analysis for two time-scale
tdc with general smooth function approximation. Advances in Neural Information
Processing Systems, 34:9747–9758, 2021. (Cited on page 8.)
[XCS+21] Liyuan Xu, Yutian Chen, Siddarth Srinivasan, Nando de Freitas, Arnaud Doucet,
and Arthur Gretton. Learning deep features in instrumental variable regression. In
International Conference on Learning Representations, 2021. (Cited on page 3.)
[XL21] Tengyu Xu and Yingbin Liang. Sample complexity bounds for two timescale value-
based reinforcement learning algorithms. In International Conference on Artificial
Intelligence and Statistics, pages 811–819. PMLR, 2021. (Cited on page 8.)
[ZCW23] Wanrong Zhu, Xi Chen, and Wei Biao Wu. Online covariance matrix estimation
in stochastic gradient descent.
Journal of the American Statistical Association,
118(541):393–404, 2023. (Cited on page 10.)
[ZGG+22] Yuchen Zhu, Limor Gultchin, Arthur Gretton, Matt J Kusner, and Ricardo Silva. Causal
inference with treatment measurement error: a nonparametric instrumental variable
approach. In Uncertainty in Artificial Intelligence, pages 2414–2424. PMLR, 2022.
(Cited on page 3.)
[ZX21] Junyu Zhang and Lin Xiao. Multilevel composite stochastic optimization via nested
variance reduction. SIAM Journal on Optimization, 31(2):1131–1157, 2021. (Cited on
page 3.)

14


---Page Break---
Appendix

A
Online updates of [DVB24]

For the sake of clarity, we present the O2SLS algorithm proposed in [DVB24, v3]1 in the streaming
format, without any explicit matrix inversions that we used in our experiments:

θt+1 = (I −Utγt
⊤ZtZ⊤
t γt)θt + Utγt
⊤ZtYt
γt+1 = (I −VtZtZ⊤
t )γt + VtZtX⊤
t

Ut+1 = Ut −Utγt⊤ZtZ⊤
t γtUt
1 + Z⊤
t γtUtγt⊤Zt

Vt+1 = Vt −VtZtZ⊤
t Vt
1 + Z⊤
t VtZt
V0 = λ−1Idz,

where Ut, Vt are two additional matrix sequences which tracks the matrix inverse of
Pt
i=1 γ⊤
i ZiZ⊤
i γi, and (λIdz + Pt
i=1 ZtZ⊤
t ) respectively for a user defined parameter λ. As men-
tioned in [DVB24], we choose λ = 0.1. The major difference between O2SLS and Algorithm 2
is that O2SLS takes an online two-stage regression approach to minimize a suitably defined regret
whereas we take a conditional stochastic optimization point of view which requires carefully chosen
step-sizes. In our Algorithm 2, we do not need to explicitly or implicitly do matrix inverse which can
potentially cause stability issues. Furthermore, unlike [DVB24], we neither assume Pt
i=1 ZiZ⊤
i is
invertible for all t nor do we assume that Z is a bounded random variable for our analysis. Finally,
the per-iteration computational complexity and memory requirement of Algorithm 2 is significantly
better than O2SLS; see Section B.

B
Per-iteration Complexities

For the linear case, i.e., the underlying relationship between Z and X as well as X and Y are linear,
Table 1 summarizes the per-iteration memory costs and number of arithmetic operations of the
original O2SLS [DVB23], the updated O2SLS [DVB24] that we provide a matrix form update in
Appendix A, TOSG-IVaR (Alg 1), and OTSG-IVaR (Alg 2) at the t-th iteration.

Notice that the original version of O2SLS [DVB23] has a per-iteration and memory cost dependent
on the iteration number t as it needs to use all the samples accumulated till the iteration t to conduct
an offline 2SLS at each iteration. The updated O2SLS [DVB24] (the algorithm that we compare
to) uses samples obtained at iteration t to perform the update. Although the updated O2SLS avoids
explicit matrix inversion, it is obvious that its arithmetic operations and memory cost per iteration are
larger than our TOSG-IVaR and OTSG-IVaR.

We highlight that the TOSG-IVaR, which uses two samples X and X′ from the conditional distribution
P(X | Z), requires only O(dx) memory and arithmetic operations at each iteration.

For a fair comparison, we assume that two n×n matrices multiplication admits an O(n3) complexity,
i.e., using normal textbook matrix multiplication. We also assume computing the inversion of a n × n
matrix admits an O(n3) complexity. Interested readers may refer to [Pap03] for more details about
faster algorithms with better complexities for matrix operations.

1Note that the streaming algorithm was not present in version 1, i.e., [DVB23, v1].

15


---Page Break---
Table 1: Memory cost and the number of arithmetic operations at iteration t.

Algorithm
Memory cost
Arithmetic Operations

O2SLS [DVB23, v1]
t(dx + dz) + dzdx + dx
O(d3
x + td2
x + tdxdz)

O2SLS [DVB24, v3] (Sec. A)
d2
x + d2
z + dzdx + dx
O(d2
x + d2
z + dzdx)

TOSG-IVaR (our Alg 1)
dx
O(dx)

OTSG-IVaR (our Alg 2)
dxdz + dx
O(min(d2
x, d2
z) + dzdx)

C
Experimental Details

C.1
Compute Resources

All experiments in Section 4 were conducted on a computer with an 11th Intel(R) Core(TM) i7-
11370H CPU. The time and space required to run our experiments are negligible and we anticipate
they can be conducted in almost all computers.

C.2
Experimental Details for Figure 1

In Figure 1, we show an example where the updates (11) may diverge first before converging
eventually and finite time performance can be much worse compared to Algorithm 2. For this
experiment, we choose the model presented in (16) with dx = dz = 1, θ∗= 1, γ∗= −1, ρ = 4, and
σϵ = 1. When initialized at γ0 = 10, and θ0 = 0, the updates in (11) keeps diverging rapidly at first
whereas Algorithm 2 is much more stable. So, by the end of 100, 000 iterations, while Algorithm 2
achieves an error of ≈10−5, (11) achieves ≈104 that is worse than it was at initialization because
(11) has not recovered from the initial divergence phase yet. However, once (11) starts converging,
the convergence rate of (11) is similar to Algorithm 2 as one can see from Figure 1 (also see our
discussion on the convergence of (11) in Section E.3).

C.3
Additional Experiments on Real-World Dataset

In this section we provide experimental results on real-world datasets provided in [AE96] and [Rya12].
The results are included in Figures 5 and 6 respectively. Following the convention in Section 4, we
write “OTSG-IVaR”, “CSO – Eq. (11)” and “[DVB23]” to represent Algorithm 2, Algorithm 2 with
the updates replaced by (11) and Algorithm 1 in [DVB24].

C.3.1
Children and Their Parents’ Labor Supply Data in [AE96]

The
outcome
Y
is
number
of
working
weeks
divided
by
52,
the
regressor
X
is
1(number of children is greater than 2),
and
the
instrumental
variable
Z
is
1(first two siblings are of same sex), where 1(·) is the indicator function.
At each time we
randomly sample a data-point from this dataset without replacement. Since the “true” parameter θ∗is
unknown in real data, we use the offline model parameter estimate as our ground truth, following
[CLL+23]. We include the results in Figure 5, in which we observe that CSO performs similar to
[DVB23] whereas OTSG-IVaR performs much better in terms of convergence speed. Moreover, the
estimation errors of CSO and [DVB23] plateau after ≈10,000 iterations whereas OTSG-IVaR keeps
on improving the estimate over the observed horizon.

C.3.2
U.S. Portland Cement Industry Data in [Rya12]

Response variable Y is log(shipped), and the predictor X is log(price). There are 4 instrumental
variables given by wage in dollars per hour for skilled manufacturing workers, electricity price, coal
price, and gas price. Unlike the previous dataset, we have only 483 data samples here. So, to mimic an
i.i.d. data stream, we divide our training into multiple epochs of length equal to the number of training
data samples, and over each epoch, at each iteration, we sample one data point uniformly randomly
without replacement from the training data to generate the data-stream. Both OTSG-IVaR and CSO
perform much better than [DVB23] in terms of convergence speed. Figure 6(b) is a magnified view
of iterations > 50 to highlight the performance difference between various algorithms.

16


---Page Break---
0
1
2
3
4
5
log (t)

3.0

2.5

2.0

1.5

1.0

0.5

0.0

0.5

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

Figure 5: Comparison of E[∥θt −θ∗∥2] (log-log scale) for Algorithm 2, Eq. 11 and [DVB24].

0
1
2
3
4
log (t)

4

2

0

2

4

6

8

10

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(a)

2.0
2.5
3.0
3.5
4.0
4.5
log (t)

4

2

0

2

4

log(
*
2)

OTSG-IVaR
CSO -- Eq. (11)
[DVB23]

(b)

Figure 6: Comparison of E[∥θt −θ∗∥2] (log-log scale) for Algorithm 2, Eq. 11 and [DVB24].

D
Proofs for Section 2

D.1
Proof of Theorem 1

Proof. We aim to find the optimal θ∗. According to (2), we know

EZ
h
EX|Z[X] · EX|Z[X]⊤i
θ∗= EZ
h
EY |Z[Y ] · EX|Z[X]
i
(17)

The updates in Algorithm 1 can be written as

θt+1 = θt −αt+1(X⊤
t θt −Yt)X⊤
t .

Hence we have

θt+1 −θ∗

=θt −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt + αt+1EZ
h
EY |Z[Y ] · EX|Z[X]
i
−θ∗

−αt+1

X′
tX⊤
t −EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt + αt+1

YtX′
t −EZ
h
EY |Z[Y ] · EX|Z[X]
i
.

(18)

Now we analyze the convergence and variance separately. For the convergence part, we have

θt −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt + αt+1EZ
h
EY |Z[Y ] · EX|Z[X]
i
−θ∗

=

I −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
(θt −θ∗).
(19)

17


---Page Break---
For the variance part we have

E
h
∥X′
tX⊤
t −EZ
h
EX|Z[X] · EX|Z[X]⊤i
∥2i

=E
h
∥X′
tX⊤
t −EX|Zt[X] · EX|Zt[X]⊤+ EX|Zt[X] · EX|Zt[X]⊤−EZ
h
EX|Z[X] · EX|Z[X]⊤i
∥2i

≤2E
h
∥X′
tX⊤
t −EX|Zt[X] · EX|Zt[X]⊤∥2 + ∥EX|Zt[X] · EX|Zt[X]⊤−EZ
h
EX|Z[X] · EX|Z[X]⊤i
∥2i

≤2Cxdϑ1
x + 2Cxxdϑ3
z
=: σ2
1
(20)

Similarly, we have

E
h
∥YtX′
t −EZ
h
EY |Z[Y ] · EX|Z[X]
i
∥2i

=E
h
∥YtX′
t −EY |Zt[Y ] · EX|Zt[X]∥2 + ∥EY |Zt[Y ] · EX|Zt[X] −EZ
h
EY |Z[Y ] · EX|Z[X]
i
∥2i

≤Cydϑ2
x + Cyxdϑ4
z
=: σ2
2.
(21)

Now we know from (18), (19), (20), and (21) that

∥θt+1 −θ∗∥2 = ∥At∥2 + 2αt+1 ⟨At, Bt⟩+ α2
t+1∥Bt∥2.
(22)

where

At =

I −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
(θt −θ∗)

Bt = −

X′
tX⊤
t −EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt +

YtX′
t −EZ
h
EY |Z[Y ] · EX|Z[X]
i
.

This implies

Eθt+1|θt
h
∥θt+1 −θ∗∥2i

=∥

I −αt+1EZ
h
EX|Z[X] · EX|Z[X]⊤i
(θt −θ∗)∥2

+ α2
t+1EXt,X′
t,Yt,Zt|θt
h
∥

X′
tX⊤
t −EZ
h
EX|Z[X] · EX|Z[X]⊤i
θt −

YtX′
t −EZ
h
EY |Z[Y ] · EX|Z[X]
i
∥2i

≤(1 −αt+1µ)2∥θt −θ∗∥2 + 3α2
t+1

σ2
1∥θt −θ∗∥2 + σ2
1∥θ∗∥2 + σ2
2∥θ∗∥2

≤((1 −αt+1µ)2 + 3α2
t+1σ2
1)∥θt −θ∗∥2 + 3α2
t+1σ2
1∥θ∗∥2 + 3α2
t+1σ2
2∥θ∗∥2,
(23)

where the first inequality uses Cauchy-Schwarz inequality, the definition of σ1, σ2 and Assumption
2.4. Choosing αt+1 such that

((1 −αt+1µ)2 + 3α2
t+1σ2
1) ≤1 −αt+1µ ⇔α ≤
µ
µ2 + 3σ2
1
and taking expectation on both sides of (23), we have

E
h
∥θt+1 −θ∗∥2i
≤(1 −αt+1µ)E
h
∥θt −θ∗∥2i
+ 3α2
t+1σ2
1∥θ∗∥2 + 3α2
t+1σ2
2∥θ∗∥2.

Now, we use the following result.

Lemma 2. Suppose we have three sequences {at}∞
t=0, {bt}∞
t=0, {rt}∞
t=0 satisfying

at+1 ≤rtat + bt, rt > 0
(24)

for any t ≥0. Define Rt+1 = Qt
i=0 ri, we have

at+1 ≤Rt+1a0 +

t
X

i=0

Rt+1bi

Ri+1
.

By Lemma 2, we know

E
h
∥θt+1 −θ∗∥2i
≤

tY

i=0
(1 −αiµ)E
h
∥θ0 −θ∗∥2i
+ (3σ2
1∥θ∗∥2 + 3σ2
2∥θ∗∥2)

t
X

i=0
α2
i

tY

j=i+1
(1 −αjµ).

18


---Page Break---
Now if we set αi = α, we know

E
h
∥θt −θ∗∥2i
≤(1 −αµ)tE
h
∥θ0 −θ∗∥2i
+ α2
t
X

i=0
(1 −αµ)i
(3σ2
1∥θ∗∥2 + 3σ2
2∥θ∗∥2)

≤e−tαµE
h
∥θ0 −θ∗∥2i
+ α

µ(3σ2
1∥θ∗∥2 + 3σ2
2∥θ∗∥2)

Choosing α, T such that α = log T

µT
≤
µ
µ2+3σ2
1 , we know

E
h
∥θT −θ∗∥2i
≤
E
h
∥θ0 −θ∗∥2i

T
+ 3∥θ∗∥2(σ2
1 + σ2
2) log T
µ2T
.

D.2
Proof of Lemma 2

Proof. We notice from (24) that for any 0 ≤i ≤t, we have
ai+1
Ri+1
≤ai

Ri
+
bi
Ri+1
.

Taking summation on both sides, we have

at+1
Rt+1
≤a0

R0
+

t
X

i=0

bi
Ri+1
which completes the proof by multiplying Rt+1 on both sides.

D.3
Proof of Lemma 1

Proof. We first notice that Assumption 2.3 holds since

EZ
h
EX|Z[X] · EX|Z[X]⊤i
= EZ
h
ϕ(γ⊤
∗Z) · ϕ(γ⊤
∗Z)⊤i
⪰µI.

For (5) and (6), we have

E
h
∥X′X⊤−EX|Z[X]EX|Z[X]⊤∥2i

=E
h
∥ϵ′
2ϕ(γ⊤
∗Z)⊤+ ϕ(γ⊤
∗Z)ϵ⊤
2 + ϵ′
2ϵ⊤
2 ∥2i

≤3E
h
∥ϵ′
2ϕ(γ⊤
∗Z)⊤∥2 + ∥ϕ(γ⊤
∗Z)ϵ⊤
2 ∥2 + ∥ϵ′
2ϵ⊤
2 ∥2i

=3E
h
∥ϕ(γ⊤
∗Z)ϵ′⊤
2 ϵ′
2ϕ(γ⊤
∗Z)⊤∥+ ∥ϕ(γ⊤
∗Z)ϵ⊤
2 ϵ2ϕ(γ⊤
∗Z)⊤∥+ |ϵ⊤
2 ϵ′
2|2i
= O(d2
x),
(25)

and
E
h
∥Y X′ −EY |Z[Y ]EX|Z[X]∥2i

=E
h
∥X′X⊤θ∗+ ϵ1X′ −EX|Z[X]EX|Z[X]⊤θ∗∥2i

≤2E
h
∥X′X⊤θ∗−EX|Z[X]EX|Z[X]⊤θ∗∥2i
+ 2E
h
ϵ2
1∥ϕ(γ⊤
∗Z) + ϵ′
2∥2i

=O(∥θ∗∥2σ2
ϵ2d2
x + σ2
ϵ1dx + σ2
ϵ1,ϵ2dx),
where the first inequality uses Cauchy-Schwarz inequality, and the second equality uses (9), (10) and
(25). For (7) we have

E
h
∥EX|Z[X] · EX|Z[X]⊤−EZ
h
EX|Z[X] · EX|Z[X]⊤i
∥2i

=E
h
∥ϕ(γ⊤
∗Z)ϕ(γ⊤
∗Z)⊤−E

ϕ(γ⊤
∗Z)ϕ(γ⊤
∗Z)⊤
∥2i
= O(dz)

where the last equality uses (10). Using the above conclusion in (8), we have

E
h
∥EY |Z[Y ] · EX|Z[X] −EZ
h
EY |Z[Y ] · EX|Z[X]
i
∥2i

=E
h
∥EX|Z[X] · EX|Z[X]⊤θ∗−EZ
h
EX|Z[X] · EX|Z[X]⊤θ∗
i
∥2i
= O(∥θ∗∥2dz).

19


---Page Break---
E
Proofs for Section 3

E.1
Proof of Theorem 2

Proof of Theorem 2 . Recall that ξZt, and ξZtYt are the i.i.d. noise sequences

ξZt = ΣZ −ZtZ⊤
t ,
ξZtYt = ΣZY −ZtYt.

Note γ∗, and θ∗can be written as γ∗= Σ−1
Z ΣZX ∈Rdz×dx, and θ∗=
 
γ⊤
∗Σzγ∗
−1 γ⊤
∗ΣZY ∈Rdx
which we are going to use throughout the proof.

To quantify the bias, we use the following bound on E

∥γt −γ∗∥k
2

, k = 1, 2, 4, proved in Lemma
3.2 of [CLTZ20].

Lemma 3. Suppose Assumption 2.2, and Assumption 3.2 hold. Then we have

E

∥γt −γ∗∥k
= O
q

dkzβt
k

for
k = 1, 2, 4.
(26)

We proceed by noting that if γ∗, ΣZ, and ΣZY were known beforehand, one could use the following
deterministic gradient updates to obtain θ∗.

eθt+1 = eθt −αt+1γ⊤
∗

ΣZγ∗eθt −ΣZY

.
(27)

Lemma 4. Let Assumption 2.3 be true. Then, choosing ηk = O(k−a) with 1/2 < a < 1, we have
∥eθt −θ∗∥= O
 
exp(−t1−a)

.

Define the sequence δt := θt −eθt. We will establish the convergence rate of E

∥δt∥2
2

. From (13),
and (27), we have the following expansion of δt+1.

δt+1 = Qtδt + αt+1Dtθt + αt+1(γt −γ∗)⊤ΣZY −αt+1γt
⊤ξZtYt + αt+1γt
⊤ξZtγtθt,
(28)

where

Qt :=(I −αt+1γ⊤
∗ΣZγ∗),

Dt :=γ⊤
∗ΣZγ∗−γt
⊤ΣZγt.

First we will establish an intermediate bound on E

∥δt∥2
. To do so, we will need the following
result which shows that E

∥θt −θ∗∥4
2

is bounded for all t which we prove in Section E.2.

Lemma 5 (Boundedness of fourth moment of ∥θt −θ∗∥). Let the conditions in Theorem 2 be
true. Then, choosing αt, βt such that αt ≤d−4κ−ϑ/2
z
, and P∞
t=1(α2
t + αt
√βt) < ∞, we have
E

∥θt −θ∗∥4
2

is bounded by some constant M > 0.

Lemma 6 (Intermediate bound on E[∥δt∥2
2]). Let the conditions in Theorem 2 be true. We have the
following intermediate bound on E

∥δt∥2
:

E

∥δt∥2
= O

βtd1+2κ
z
+ αt+1d4κ+ϑ/2
z
+
p

dzβt

.
(29)

Proof of Lemma 6. Recall the update for δt+1 obtained in (28).

δt+1 = Qtδt + αt+1Dtθt + αt+1(γt −γ∗)⊤ΣZY −αt+1γt
⊤ξZtYt + αt+1γt
⊤ξZtγtθt.

Then,

∥δt+1∥2
2 =δt
⊤Q2
tδt + α2
t+1∥Dtθt + (γt −γ∗)⊤ΣZY −γt
⊤ξZtYt + γt
⊤ξZtγtθt∥2

+ 2αt+1δt
⊤Qt
 
Dtθt + (γt −γ∗)⊤ΣZY


+ 2αt+1δt
⊤Qt
 
γt
⊤ξZtγtθt −γt
⊤ξZtYt

.

(30)

20


---Page Break---
Then, choosing α1(∥γ∗∥2λZ)2 < 1, using Young’s inequality and Assumption 2.2, from (30) we get,

Et

∥δt+1∥2
2

≤(1 −αt+1µ)∥δt∥2 + 4α2
t+1
 
∥Dtθt∥2 + ∥(γt −γ∗)⊤ΣZY ∥2

+ 4α2
t+1
 
∥γt∥2
2E

∥ξZtYt∥2
2

+ ∥γt∥4
2E

∥ξZt∥2
∥θt∥2

+ 2αt+1δt
⊤Qt
 
Dtθt + (γt −γ∗)⊤ΣZY


≲(1 −αt+1µ)∥δt∥2 + 4α2
t+1
 
∥Dt∥2∥θt∥2 + ∥(γt −γ∗)⊤ΣZY ∥2

+ 4Cα2
t+1

d2κ+ϑ/2
z
+ d4κ+ϑ/2
z
∥θt∥2
+

2αt+1δt
⊤Qt
 
Dtθt + (γt −γ∗)⊤ΣZY

,

where the last inequality follows by Assumption 3.1,and Assumption 3.3.

Now, taking expectation on both sides, we obtain

E

∥δt+1∥2
2

≲(1 −αt+1µ)E

∥δt∥2
+ 4α2
t+1
 
E

∥Dt∥2∥θt∥2
+ E

∥(γt −γ∗)⊤ΣZY ∥2

+ 4Cα2
t+1

d2κ+ϑ/2
z
+ d4κ+ϑ/2
z
E

∥θt∥2

+ 2αt+1

E
h
|δt
⊤QtDtθt|
i
+ E
h
|δt
⊤Qt(γt −γ∗)⊤ΣZY |
i
.

(31)

Now, the following bounds are true:

1. We have that

α2
t+1E

∥Dt∥2∥θt∥2
≤α2
t+1
q

E [∥Dt∥4
2] E [∥θt∥4
2] ≲d1+2κ
z
α2
t+1βt,
(32)

where the first inequality follows by Cauchy-Schwarz inequality, the second inequality follows by
(42), and Lemma 5.

2. Using ΣZY = O(1), and Lemma 3, we get

α2
t+1E

∥(γt −γ∗)⊤ΣZY ∥2
≲dzβtα2
t+1.
(33)

3. We have that

αt+1E
h
|δt
⊤QtDtθt|
i
≤αt+1E [∥δt∥2∥Qt∥2∥Dt∥2∥θt∥2]

≤αt+1µ

16
E

∥δt∥2
+ 4αt+1

µ

q

E [∥Dt∥4
2] E [∥θt∥4
2]

≲αt+1µ

16
E

∥δt∥2
+ 4d1+2κ
z
αt+1βt
µ
,
(34)

where the first inequality follows by Hölder’s inequality, the second inequality follows by Young’s
inequality, Cauchy-Schwarz inequality, and ∥Qt∥2 < 1, and the third inequality follows by (42),
and Lemma 5.

4. Using ∥Qt∥2 < 1, ∥ΣZY ∥2 = O(1), Cauchy-Schwarz inequality, and Lemma 3, we get,

αt+1E
h
|δt
⊤Qt(γt −γ∗)⊤ΣZY |
i

≲αt+1E [∥δt∥2∥γt −γ∗∥2]

≤αt+1
q

E [∥δt∥2
2] E [∥γt −γ∗∥2
2]

≤
√dzβtαt+1

2
+
√dzβtαt+1E

∥δt∥2
2


2
.
(35)

Combining (31), (32), (33), (34), (35), and Lemma 5, we have

E

∥δt+1∥2
2


21


---Page Break---
≲(1 −αt+1µ)E

∥δt∥2
+ 4α2
t+1βtd1+2κ
z
+ 4Cα2
t+1d4κ+ϑ/2
z

+ 2αt+1

µE

∥δt∥2
/16 + 4d1+2κ
z
βt/µ +
p

dzβt/2 +
p

dzβtE

∥δt∥2
2

/2

(36)

≲(1 −7µαt+1/8 + αt+1
p

dzβt)E

∥δt∥2
+ (8αt+1βtd1+2κ
z
/µ + 4Cα2
t+1d4κ+ϑ/2
z
)

+ αt+1
p

dzβt

≲(1 −3µαt+1/4)E

∥δt∥2
+ (8αt+1βtd1+2κ
z
/µ + 4Cα2
t+1d4κ+ϑ/2
z
) + αt+1
p

dzβt.
(37)

In the above, the third inequality follows by choosing βt ≤µ2/(64dz), and αt+1
√dzβt < 1. Then,
from (37), we have

E

∥δt∥2
2

= O

βtd1+2κ
z
+ αt+1d4κ+ϑ/2
z
+
p

dzβt

.

Coming back to the proof of Theorem 2, observe that, we can sharpen the bound in (35) using
Lemma 6 which allows us to avoid the use of Young’s inequality. This leads to the following
improved version of the recursion in (37) using which we can improve the term √dzβt in (29) as
follows:

E

∥δt+1∥2
2

≲(1 −7µαt+1/8)E

∥δt∥2

+ αt+1O

βtd1+2κ
z
+ αt+1d4κ+ϑ/2
z
+
p

αt+1βtd1/2+2κ+ϑ/4
z
+ (βtdz)3/4

=O

βtd1+2κ
z
+ αt+1d4κ+ϑ/2
z
+
p

αt+1βtd1/2+2κ+ϑ/4
z
+ (βtdz)3/4
.

In fact, this trick can be used repeatedly to sharpen the bound even further as shown in Lemma 7.

Lemma 7 (Final improved bound on E[∥δt∥2
2]). Let the conditions in Theorem 2 be true. Then
using Lemma 6, we have,

E

∥δt+1∥2
2


≲O

 

(dzβt)1−2−r−1 +

r
X

i=0


α2−i
t+1βt
1−2−id1+(4κ+ϑ/2−1)2−i
z
+ βt(1 + α2−i
t+1)d1+21−iκ
z
!

,

where r is any non-negative integer.

Proof of Lemma 7. If we have

E

∥δt∥2
= O

αt+1d4κ+ϑ/2
z
+ βtd1+2κ
z
+
p

dzβt

,

then from (35), we have,

E
h
|δt
⊤Qt(γt −γ∗)⊤ΣZY |
i
≲
q

E [∥δt∥2
2] E [∥γt −γ∗∥2
2]

=O
p

αt+1βtd1/2+2κ+ϑ/4
z
+ βtd1+κ
z
+ (dzβt)3/4
.
(38)

Then, similar to (36), we have,

E

∥δt+1∥2
2


≲(1 −αt+1µ)E

∥δt∥2
+ 4α2
t+1βtd1+2κ
z
+ 4Cα2
t+1d4κ+ϑ/2
z

+ 2αt+1

µE

∥δt∥2
/16 + 4d1+2κ
z
βt/µ +
p

αt+1βtd1/2+2κ+ϑ/4
z
+ βtd1+κ
z
+ (dzβt)3/4

≲(1 −7µαt+1/8)E

∥δt∥2

+ αt+1O

 

(dzβt)3/4 +

1
X

i=0


α2−i
t+1βt
1−2−id1+(4κ+ϑ/2−1)2−i
z
+ βt(1 + α2−i
t+1)d1+21−iκ
z
!

=O

 

(dzβt)3/4 +

1
X

i=0


α2−i
t+1βt
1−2−id1+(4κ+ϑ/2−1)2−i
z
+ βt(1 + α2−i
t+1)d1+21−iκ
z
!

.

22


---Page Break---
Now if we repeat this step r number of times (where r is to be set later), by progressive sharpening
we get the following bound.
E

∥δt+1∥2
2


≲O

 

(dzβt)1−2−r−1 +

r
X

i=0


α2−i
t+1βt
1−2−id1+(4κ+ϑ/2−1)2−i
z
+ βt(1 + α2−i
t+1)d1+21−iκ
z
!

.

Coming back to the proof of Theorem 2, we have that by combining Lemma 4, and Lemma 7,

E

∥θt −θ∗∥2
≤2E

∥δt∥2
+ 2E
h
∥eθt −θ∗∥2i

=O

 

(dzβt)1−2−r−1 +

r
X

i=0


α2−i
t+1βt
1−2−id1+(4κ+ϑ/2−1)2−i
z
+ βt(1 + α2−i
t+1)d1+21−iκ
z
!

. (39)

Now, in (39), for some arbitrarily small number ι > 0, choosing

αt = min(0.5d−4κ−ϑ/2
z
λ−1
Z C−2
γ , 0.5(∥γ∗∥2λZ)−2)t−1+ι/2,
βt = µ2d−1−2κ
z
t−1+ι/2/128,

and setting r = ⌈log2 ((ι/2)−1 −1) −1⌉we get,

E

∥θt −θ∗∥2
= O

max

t−1+ι, t−1+ι/2 log((ι/2)−1 −1)

.

E.2
Proof of Lemma 5

Proof. Using the form of θ∗, from (13) we get,

θt+1 −θ∗= bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY + αt+1Dtθ∗+ αt+1γt
⊤ξZtγt(θt −θ∗)

+ αt+1γt
⊤ξZtγtθ∗+ αt+1γt
⊤ξZtYt.
(40)

where bQt :=
 
I −αt+1γt⊤ΣZγt

= Qt + αt+1Dt. Recall that Dt = γ⊤
∗ΣZγ∗−γt⊤ΣZγt. By
Assumption 3.3, we have the following bound on ∥Dt∥2.

∥Dt∥2 = O(λZC2
γd2κ
z ).
(41)

We have the following bound on E

∥Dt∥4
2

by Lemma 3.

E

∥Dt∥4
2

= E

∥(γ∗−γt)⊤ΣZγ∗+ γt
⊤ΣZ(γ∗−γt)∥4
2

= O(d2+4κ
z
β2
t ).
(42)
From (40), we have

∥θt+1 −θ∗∥2
2 ≤(θt −θ∗)⊤bQ2
t(θt −θ∗) + 3α2
t+1∥γt
⊤ξZtγt(θt −θ∗)∥2
2
+ 2αt+1(θt −θ∗)⊤bQt(γt −γ∗)⊤ΣZY

+ 2αt+1(θt −θ∗)⊤bQtDtθ∗+ A1,t + A2,t,
(43)
where
A1,t =α2
t+1
 
∥(γt −γ∗)⊤ΣZY ∥2
2 + ∥Dtθ∗∥2
2
+ 2Σ⊤
ZY (γt −γ∗)Dtθ∗+ 3∥γt
⊤ξZtγtθ∗∥2
2 + 3∥γt
⊤ξZtYt∥2
2

,
(44)
and
A2,t =2αt+1( bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY
+ αt+1Dtθ∗)⊤(γt
⊤ξZtγt(θt −θ∗) + γt
⊤ξZtγtθ∗+ γt
⊤ξZtYt).
Define
A3,t :=3α2
t+1∥γt
⊤ξZtγt(θt −θ∗)∥2
2 + 2αt+1(θt −θ∗)⊤bQt(γt −γ∗)⊤ΣZY

+ 2αt+1(θt −θ∗)⊤bQtDtθ∗+ A1,t + A2,t.
(45)

Then, choosing C2
γd2κ
z λZαt+1 < 1, which ensures ∥bQt∥≤1, we have

∥θt+1 −θ∗∥4
2 ≤∥θt −θ∗∥4
2 + 2(θt −θ∗)⊤bQ2
t(θt −θ∗)A3,t + A2
3,t.
(46)
We now have the following bounds:

23


---Page Break---
1. Using Assumption 3.1, and Assumption 3.3,

α4
t+1E

∥γt
⊤ξZtγt(θt −θ∗)∥4
2

≲d8κ+ϑ
z
α4
t+1E

∥θt −θ∗∥4
2

.
(47)

2. We have that

E
h
((θt −θ∗)⊤bQt(γt −γ∗)⊤ΣZY )2i

≲E

∥θt −θ∗∥2∥γt −γ∗∥2

≤
q

E [∥θt −θ∗∥4
2] E [∥γt −γ∗∥4
2]

≤dzβt
 
1 + E

∥θt −θ∗∥4
2

/2,
(48)

where, the first inequality follows by ∥bQt∥2 = O(1), and ∥ΣZY ∥2 = O(1). The second inequality
follows by Cauchy-Schwarz inequality. The last inequality follows by
√

ab ≤(a + b)/2, and
Lemma 3.

3. We have that

E
h
((θt −θ∗)⊤bQtDtθ∗)2i

≲E

∥θt −θ∗∥2
2∥Dt∥2
2


≤
q

E [∥θt −θ∗∥4
2] E [∥Dt∥4
2]

≲d1+2κ
z
βt
 
1 + E

∥θt −θ∗∥4
2

/2,
(49)

where, the first inequality follows by ∥bQt∥2 = O(1), and ∥θ∗∥2 = O(1). The second inequality
follows by Cauchy-Schwarz inequality. The last inequality follows by
√

ab ≤(a + b)/2, and (42).

4. Using Assumption 3.1, Assumption 3.3, (42), and Lemma 3, we have

E

A2
1,t

= O
 
d8κ+ϑ
z
α4
t+1

.
(50)

.

5. Using Young’s inequality, Assumption 3.1, Assumption 3.3, Lemma 3, ∥ΣZY ∥2 = O(1), ∥θ∗∥2 =
O(1), and (42), we have

E

A2
2,t

≤2α2
t+1E
h
∥bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY + αt+1Dtθ∗∥4
2
i

+ 2α2
t+1E

∥γt
⊤ξZtγt(θt −θ∗) + γt
⊤ξZtγtθ∗+ γt
⊤ξZtYt∥4
2


≲α2
t+1d8κ+ϑ
z
(1 + E

∥θt −θ∗∥4
2

).
(51)

6. Using ∥bQt∥2 = O(1), Assumption 3.1, and Assumption 3.3,

α2
t+1E
h
(θt −θ∗)⊤bQ2
t(θt −θ∗)∥γt
⊤ξZtγt(θt −θ∗)∥2
2
i
≲α2
t+1d4κ+ϑ/2
z
E

∥θt −θ∗∥4
2

.
(52)

7. We have that

αt+1E
h
|(θt −θ∗)⊤bQ2
t(θt −θ∗)(θt −θ∗)⊤bQt(γt −γ∗)⊤ΣZY |
i

≲αt+1E

∥θt −θ∗∥3
2∥γt −γ∗∥2


≤αt+1
 
E

∥θt −θ∗∥4
2
3/4  
E

∥γt −γ∗∥4
2
1/4

≤αt+1
p

dzβt
 
E

∥θt −θ∗∥4
2
3/4

≤3αt+1
√dzβt
4
E

∥θt −θ∗∥4
2

+ αt+1
√dzβt

4
,
(53)

where, the first inequality follows by ∥bQt∥2 = O(1), and ∥ΣZY ∥2 = O(1), the second inequality
follows by Cauchy-Schwarz inequality, the third inequality follows by Lemma 3 and the fourth
inequality follows by Young’s inequality.

24


---Page Break---
8. Similar to (53), we have,

αt+1E
h
|(θt −θ∗)⊤bQ2
t(θt −θ∗)(θt −θ∗)⊤bQtDtθ∗|
i

≤3d1/2+κ
z
αt+1
√βt
4
E

∥θt −θ∗∥4
2

+ d1/2+κ
z
αt+1
√βt
4
.
(54)

9. Using ∥bQt∥2 = O(1), Cauchy-Schwarz inequality, (50), and Young’s inequality,

E
h
(θt −θ∗)⊤bQ2
t(θt −θ∗)A1,t
i

≤E

∥θt −θ∗∥2
2A1,t


≤
q

E [∥θt −θ∗∥4
2] E

A2
1,t


≲d4κ+ϑ/2
z
α2
t+1
 
1 + E

∥θt −θ∗∥4
2

.
(55)

10. By Assumption 2.2, we have,

Et
h
(θt −θ∗)⊤bQ2
t(θt −θ∗)A2,t
i
= 0.
(56)

Now using Jensen’s inequality, and combining (47), (48), (49), (50), and (51), we have,

E

A2
3,t

≤45α4
t+1E

∥γt
⊤ξZtγt(θt −θ∗)∥4
2

+ 20α2
t+1E
h
((θt −θ∗)⊤bQt(γt −γ∗)⊤ΣZY )2i

+ 20α2
t+1E
h
((θt −θ∗)⊤bQtDtθ∗)2i
+ 5E

A2
1,t

+ 5E

A2
2,t


≲α4
t+1dϑ7+8κ
z
E

∥θt −θ∗∥4
2

+ dzα2
t+1βt
 
1 + E

∥θt −θ∗∥4
2


+ α2
t+1d1+2κ
z
βt
 
1 + E

∥θt −θ∗∥4
2

+ d8κ+ϑ
z
α4
t+1 + α2
t+1d8κ+ϑ
z
(1 + E

∥θt −θ∗∥4
2

)

≲α2
t+1d8κ+ϑ
z
 
1 + E

∥θt −θ∗∥4
2

.
(57)

Combining (52), (53), (54), (55), and (56), we get,

E
h
(θt −θ∗)⊤bQ2
t(θt −θ∗)A3,t
i

≲α2
t+1d4κ+ϑ/2
z
E

∥θt −θ∗∥4
2

+ 3αt+1
√dzβt
4
E

∥θt −θ∗∥4
2

+ αt+1
√dzβt

4

+ 3d1/2+κ
z
αt+1
√βt
4
E

∥θt −θ∗∥4
2

+ d1/2+κ
z
αt+1
√βt
4
+ d4κ+ϑ/2
z
α2
t+1
 
1 + E

∥θt −θ∗∥4
2


≲(α2
t+1d4κ+ϑ/2
z
+ αt+1
p

βt+1d1/2+κ
z
)(1 + ∥θt −θ∗∥4
2).
(58)

Combining (46), (57), and (58), we have,

E

∥θt+1 −θ∗∥4
2

≲(1 + α2
t+1d8κ+ϑ
z
+ αt+1
p

βt+1d1/2+κ
z
)
 
1 + E

∥θt −θ∗∥4
2

.
(59)

Now choosing αt, βt such that αt ≤d−4κ−ϑ/2
z
, and P∞
t=1(α2
t+1 + αt+1
p

βt+1) < ∞, we get

E

∥θt −θ∗∥4
2

≤M,
(60)

for some constant 0 ≤M < ∞.

E.3
Comment on the convergence of (11)

We now discuss the convergence properties of the update sequence (11), which we refer to as the
conditional stochastic optimization (CSO) based updates, which we restate below:

θt+1 = θt −αt+1γ⊤
t Zt(X⊤
t θt −Yt),
γt+1 = γt −βt+1Zt(Z⊤
t γt −X⊤
t ).

Similar to (40), for the above updates, we have the following expansion:

θt+1 −θ∗= bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY + αt+1Dtθ∗+ αt+1γt
⊤ξZtγ∗(θt −θ∗)

25


---Page Break---
+ αt+1γt
⊤ξZtγ∗θ∗+ αt+1γt
⊤ξZtYt −αt+1γt
⊤Ztϵ2,t
⊤θt,

where ξZt = ΣZ −ZtZ⊤
t , ξZtYt = ΣZY −ZtYt, bQt :=
 
I −αt+1γt⊤ΣZγ∗

= Qt + αt+1Dt, and
Dt = (γ∗−γt)ΣZγ∗.

Recall that the reason for the initial divergence of the updates in (11) are the potential negative
eigenvalues of γt⊤ΣZγ∗. Here we will show that if γt⊤ΣZγ∗is positive semi-definite or γt is close
enough to γ∗such that the negative eigenvalues (if any) are not too large in absolute values, then the
updates in (11) indeed exhibit the same convergence rate as Algorithm 2.

Assumption E.1. Let either of the following two conditions be true. For all t ≥t0,

1. γtΣZγ∗is positive semidefinite.
2. ∥γt −γ∗∥2 ≲dzβt.

Note that Condition 1 of Assumption E.1 is an idealized condition which is difficult to ensure for all
t in reality. But of course if this is true, then γtΣZγ∗does not have a negative eigenvalue to cause
divergence and the proof then follows exactly like Lemma 5.

Hence, we will focus on the more realistic Condition 2 of Assumption E.1 which holds true almost
surely [PJ92]. Since we are interested in the asymptotic rate of convergence of CSO updates (due to
the requirement of Assumption E.1), we will only concentrate on the iterations t ≥t0. In this case,
the proof steps are similar to Theorem 2 except for two major differences, that we discuss below.

Difference 1: Potential negative definiteness of γt⊤ΣZγ∗:

Under Condition 2, γt⊤ΣZγ∗can indeed be negative definite. In general, if γt⊤ΣZγ∗is negative
definite then that is undesirable as we explain Section 3. In terms of the proof, we can no longer write
(θt −θ∗)⊤bQ⊤
t bQt(θt −θ∗) ≤∥θt −θ∗∥2 (which was possible to do in (43) in the proof of Lemma 5).
Subsequently, (46) breaks down. But we will show that under Condition 2 the negative eigenvalues
are not too large in terms of absolute values. Specifically, we can write,

(θt −θ∗)⊤bQ⊤
t bQt(θt −θ∗)

=(θt −θ∗)⊤(Q2
t + αt+1Q⊤
t Dt + αt+1D⊤
t Qt + α2
t+1D⊤
t Dt)(θt −θ∗)

≤(1 + 2αt+1∥Dt∥)∥θt −θ∗∥2 + α2
t+1∥Dt∥2∥θt −θ∗∥2

≤(1 + 2αt+1
p

dzβt)∥θt −θ∗∥2 + α2
t+1∥Dt∥2∥θt −θ∗∥2.

(61)

The term α2
t+1∥Dt∥2∥θt −θ∗∥2 is of the order of A3,t defined in (45). Now αt+1
√dzβt is small
enough in the sense that we choose the stepsizes such that P∞
t=1(α2
t+1 + αt+1
√βt) < ∞. Using
this one can now show a similar bound as (59) and consequently show E

∥θt −θ∗∥4
is bounded.

Now let us see what happens in the absence of Condition 2. Here one could use the fact (1 +
2αt+1∥Dt∥) ≲(1 + 2Cγαt+1dκ
z ) which is too big. Recall that we want something at least of the
order of αt+1
√βt to show that θt sequence is bounded. One could also try to use the fact that
E [∥Dt∥] is small by Lemma 3. But since Dt and θt are interdependent, one needs to decouple them.
One way to do this would be to use Cauchy-Shwarz inequalityas shown below.

E

∥Dt∥∥θt −θ∗∥2
≤
p

E [∥Dt∥2] E [∥θt −θ∗∥4] ≲
p

dzβtE [∥θt −θ∗∥4].

But that leads to the presence of E

∥θt −θ∗∥4
in (43) which is potentially problematic due to the
fact that on the left-hand side we have E

∥θt+1 −θ∗∥2
.

Difference 2: Presence of additional error term αt+1γt⊤Ztϵ2,t⊤θt:

When comparing (12) with (40), yet another crucial difference is the presence of the term
αt+1γt⊤Ztϵ2,t⊤θt. We will show by the following observations that this error term gets absorbed by
other terms already present in (40) without affecting the convergence rate. Specifically, the following
holds.

26


---Page Break---
1. Using the independence between Z, and ϵ2,t, and by Assumption 2.2, we have,

Et[( bQt(θt −θ∗) + αt+1(γt −γ∗)⊤ΣZY + αt+1Dtθ∗+ αt+1γt
⊤ξZtγ∗(θt −θ∗)

+ αt+1γt
⊤ξZtγ∗θ∗)⊤γt
⊤Ztϵ2,t
⊤θt] = 0.

2. We also have that

α2
t+1Et

(γt
⊤ξZtYt)⊤γt
⊤Ztϵ2,t
⊤θt


=α2
t+1(γt
⊤ΣZγt∥θ∗∥2 + γt
⊤ΣZγtθ⊤
∗(θt −θ∗))

≤α2
t+1(γt
⊤ΣZγt∥θ∗∥2 + ∥γt
⊤ΣZγt(θt −θ∗)∥2 + ∥θ∗∥2)

This shows that the above term is of the same order as A1,t and A3,t defined in (44), and (45).
3. Finally, we have

α2
t+1Et

∥γt
⊤Ztϵ2,t
⊤θt∥2
≲α2
t+1(∥γt∥2∥θt −θ∗∥2 + ∥γt∥2∥θ∗∥2).

So this term is of the order of A3,t as well.

Combining the above facts and following similar procedure as the proof of Theorem 2, one can show
that the CSO updates achieve a similar rate under additional Assumption E.1.

27


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: Sections 2, 3, and 4.
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
Justification: We discuss some limitations of our analysis and some future directions in the
end of Section 2.
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

28


---Page Break---
Justification: Sections 2, 3, and the Appendix.
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
Justification: Sections 4, C.1, and C.2.
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

29


---Page Break---
Answer: [Yes]

Justification: Sections 4, C.1, and C.2. We also provide the code in the main supplemental
material.

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

Justification: Sections 4, C.1, and C.2.

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

Justification: Section 4.

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

30


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
Justification: Sections 4, C.1, and C.2.
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
Justification: Sections 2 and 3. Our research conforms in every respect, with the NeurIPS
Code of Ethics.
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
Justification: Sections 2 and 3. Our work is mainly about theory and algorithm design, so
there is no societal impact of the work performed.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

31


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

Justification: Sections 2 and 3. Our work is mainly about theory and algorithm design, and
poses no such risks.

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

Justification: Sections 4, C.1, and C.2.

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

32


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: Sections 4, C.1, and C.2. The code is in the supplementary material.
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
Justification: Sections 2 and 3. Our paper is mainly about theory and algorithm design, and
does not involve crowdsourcing nor research with human subjects.
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
Justification: Sections 2 and 3. Our paper is mainly about theory and algorithm design, and
does not involve crowdsourcing nor research with human subjects.
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
