Efficient and Sharp Off-Policy Evaluation
in Robust Markov Decision Processes

Andrew Bennett∗
Morgan Stanley
andrew.bennett@morganstanley.com

Nathan Kallus∗
Cornell University
kallus@cornell.edu

Miruna Oprescu∗
Cornell University
amo78@cornell.edu

Wen Sun∗
Cornell University
ws455@cornell.edu

Kaiwen Wang∗
Cornell University
kw437@cornell.edu

Abstract

We study the evaluation of a policy under best- and worst-case perturbations to a
Markov decision process (MDP), using transition observations from the original
MDP, whether they are generated under the same or a different policy. This is
an important problem when there is the possibility of a shift between historical
and future environments, e.g. due to unmeasured confounding, distributional
shift, or an adversarial environment. We propose a perturbation model that allows
changes in the transition kernel densities up to a given multiplicative factor or its
reciprocal, extending the classic marginal sensitivity model (MSM) for single time-
step decision-making to infinite-horizon RL. We characterize the sharp bounds
on policy value under this model – i.e., the tightest possible bounds based on
transition observations from the original MDP – and we study the estimation of
these bounds from such transition observations. We develop an estimator with
several important guarantees: it is semiparametrically efficient, and remains so even
when certain necessary nuisance functions, such as worst-case Q-functions, are
estimated at slow, nonparametric rates. Our estimator is also asymptotically normal,
enabling straightforward statistical inference using Wald confidence intervals.
Moreover, when certain nuisances are estimated inconsistently, the estimator still
provides valid, albeit possibly not sharp, bounds on the policy value. We validate
these properties in numerical simulations. The combination of accounting for
environment shifts from train to test (robustness), being insensitive to nuisance-
function estimation (orthogonality), and addressing the challenge of learning from
finite samples (inference) together leads to credible and reliable policy evaluation.

1
Introduction

Offline policy evaluation (OPE) from historical data is crucial in domains where active, on-policy
experimentation is costly, risky, unethical, or otherwise operationally infeasible. Relevant domains
range from medicine to finance and recommendation systems. However, whenever historical data is
used to study future behavior, there is a concern of non-stationarity – shift between the environment
generating the data (training environment) and the environment in which a policy will be deployed
(test environment). This may occur, e.g., due to general distributional shifts in the environment
over time, unobserved confounding in the observed historical data, or adversarial elements of the
environment (such as other agents) that may react when the agent is deployed. While standard OPE
in offline reinforcement learning (ORL) accounts for the change between the logging and evaluation
policies, it may overlook the fact that the Markov decision process (MDP) too has changed. While

∗Alphabetical order.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
this issue is particularly critical in high-stakes domains, it is broadly appealing to understand how
value shifts across different environments in any application domain.

Robust MDPs [34, 56] model unknown environments by allowing an adversary to choose from any
one environment in a set. Therefore, they offer a natural model for unknown environment shifts by
simply considering all environments to which we could possibly shift. A variety of work addresses
questions such as planning in a known robust MDP [30, 51, 80] as well as online learning [6, 79].
Here we focus on a purely statistical estimation question: given observations of transitions from
some unknown transition kernel, we wish to estimate the worst-case (or best-case) value of a given
evaluation policy in a robust MDP, defined by a set of MDPs whose transition functions are centered
around the observed transition kernel.

This setting captures the previously studied unconfounded robust OPE problem [73], where the
observed transition kernel corresponds to an MDP, and the observed transitions are the result of
applying some logging policy within this MDP. In such cases, the goal is to estimate policy values
that are robust to future changes in the MDP dynamics. However, our setting is more general in that it
also captures problems where the observed transitions are confounded by some unobserved variables,
in which case they do not correspond to observations from the transition kernel of an MDP. In this
case, the robust MDP and the robust policy value estimates are designed to account for worst-case (or
best-case) impact of this confounding bias. In either case, as in ORL, we emphasize that we do not
know the observational MDP, and can only access it via a sample of transitions. Furthermore, even in
the simple case with no unmeasured confounding, in a notable departure from standard ORL, the
problem can be difficult even if the logging and evaluation policies are the same (the usually easy
on-policy setting), since the policy can induce very different visitation distributions in the original
and perturbed MDPs.

Such robust offline evaluation from transition data was considered in recent work [12, 59]. We build
on this recent work by focusing the question of statistically efficient and robust estimation of the sharp
bounds (i.e., the tightest possible given the data). Previous work focused on evaluation using only
the Q-function under the worst-case environment (in some cases under a relaxation of the adversary,
leading to loose bounds). Thus, any error in its estimation translates directly to error in evaluation.
In other words, flexible nonparametric modeling of this function can mean slow rates for estimated
bounds and a lack semiparametric efficiency. Moreover, without a clear understanding of the noise in
the estimates, we cannot add confidence bands to the bounds, leading to bounds that are too tight.

We address these challenges by developing an orthogonalized estimation method that combines
several nuisance functions: the worst-case Q-function, the state-visitation frequency in the worst-case
environment, and a threshold function characterizing the worst-case transition kernel. Our first key
result is that, to first order, our estimator behaves as a sample average using the true values of these
functions without having to estimate them at all, provided we just estimate them at certain slow
nonparametric rates. This ensures we not only have a √n-rate of estimation even when nuisances
are estimated more slowly, but also that our estimator is asymptotically normal. This allows for the
construction of confidence bands on the bounds, providing assurance that the true bound is captured.
We further show that our asymptotic variance is in fact the minimum variance among all regular and
asymptotically linear (RAL) estimators, ensuring semiparametric efficiency. Our second key result is
that even if we do not estimate some of the nuisance functions correctly, we are still consistent to
sharp or valid bounds. That is, even when we are biased due to misestimation of nuisances, our bias
(if any) only enlarges our bounds, so they remain valid. We illustrate these guarantees numerically.
Collectively, these guarantees lend substantial credibility to the bounds generated by our method.

Our contributions are summarized as follows:

1. We provide novel algorithms and analysis for learning robust Q-functions (Section 3) and
robust visitation density ratios (Section 4) under the function approximation setting.
2. We derive the sharp and efficient estimator for the robust policy value, which is optimal in
the local-minimax sense and is the gold standard in semiparametric estimation (Section 5).
3. We empirically validate the efficiency and sharpness of our approach (Section 6).

1.1
Related Works

Unobserved Confounding in Sequential Decision-Making. OPE in robust MDPs is related to OPE
bounds in confounded MDPs, where the behavior policy and the transition kernel are influenced by

2


---Page Break---
unobserved confounders. The constraint Eq. (1) that defines our target robust MDP aligns with the
Marginal Sensitivity Model (MSM) [66] employed in sensitivity analysis for causal inference. Yet,
unlike the MSM, which limits the ratio of policy densities, our approach directly constrains the ratio of
the transition kernels. Our formulation can be viewed as a generalization of the MSM from traditional
two-action no-horizon causal effects (where the constrains coincide) to multi-action infinite-horizon
discounted MDPs, where the next state is the “potential outcome”. In that sense, our model essentially
serves as an outcome-based sensitivity model [10]. This distinction is crucial as it enables our model
to subsume the policy-based MSM in cases where the policy is confounded. Nonetheless, the reverse
does not hold, and the policy-based MSM does not imply a transition kernel-based MSM for A > 2.
This point is further corroborated by [12], who explore the policy-based MSM within confounded
MDPs and obtain non-sharp identification bounds when A > 2. In contrast, our approach yields
sharp identification in general, regardless of the number of actions and without placing assumptions
on the behavior policy, which may or may not be confounded.

[13] also considered an MSM-like model in the transition kernel but their formulation assumes
A = 2. [40] operates under the setting of [12] and required tabular states. We note that all these
works including ours considers i.i.d. confounders at each step, which translates to a robust MDP
with (s, a)-rectangularity and ensures that the worst-case problem is still an MDP rather than a
POMDP. The importance of this assumption was verified by [55], who showed that without it, the
non-memoryless confounder can create exponential-in-horizon changes in value.

Neyman Orthogonality and Semiparametric Efficient Estimation. We leverage a body of research
focusing on learning with nuisances functions (e.g., Q-functions) that we need to estimate from data
but are not the primary target (e.g., policy value). Much of this research [7, 16, 17, 29, 64, 70, among
others] aims to identify Neyman-orthogonal estimators, which are first order orthogonal (insensitive)
to nuisance errors. This literature is tightly linked to the semiparametric efficient estimation literature
since Neyman-orthogonal scores can arise naturally from efficient influence functions [33, 62].
Going beyond the no-horizon causal inference setting, some explore such estimators in off-policy
sequential-decisions contexts [19, 38, 42, 48, 50]. Notably, [39] derive efficient influence functions
and orthogonal estimation in standard, non-robust OPE in infinite-horizon RL, which coincides with
our unconfounded no-uncertainty case (Λ = 1).

Moving beyond point-identified settings, some works explore orthogonality and efficiency for partial
identification and sensitivity analysis. In the causal inference literature, efficient/orthogonal estimation
in the no-horizon setting has been studied extensively under several sensitivity models [10, 18, 24, 58].
Closest to our work is [24] who provide an orthogonal estimator and convergence rates under the
MSM [66], which coincides with our setting under γ = 1. In the sequential setting, [55] considers
confounding at a single time step under the MSM, but their estimator is not orthogonal when the
quantile function is unknown. [12] provide a fitted-Q-iteration learner with an orthogonalized loss
function, but not orthogonal/efficient estimates of worst-case policy value.

2
Preliminaries

We consider an MDP with state space S, action space A, transition kernel P(s′ | s, a), reward
function r(s, a) ∈[0, 1] and initial state distribution d1 ∈∆(S). We do not require S or A to
be finite. We assume r and d1 are known for simplicity, and it is standard to extend our analysis
to when they are unknown. We are given a dataset D of n i.i.d. tuples (si, ai, ri, s′
i) such that
(si, ai) ∼ν, s′
i ∼P(· | s, a) and ri = r(si, ai), where ν is an arbitrary data-generating distribution.
For discount factor γ ∈[0, 1), let the Q function be the discounted cumulative rewards under a policy
π : S →A, Qπ,P (s, a) = Eπ,P [P∞
t=0 γtrt(st, at) | s1 = s, a1 = a]. Similarly, define the value
function as Vπ,P (s) = Qπ,P (s, π), where we use the notation f(s, π) := Ea∼π(s)[f(s, a)] for any
function f : S × A →R.

We are interested in estimating the value of a fixed target policy πt (a.k.a. evaluation policy) in an
unobserved MDP with a feasible perturbed transition kernel U. We say U is a feasible perturbation
of the observed, nominal kernel P if for all s, a, s′: we have

Λ−1(s, a) ≤dU(s′|s,a)

dP (s′|s,a) ≤Λ(s, a)
(1)

where Λ(s, a) ∈[1, ∞) is a sensitivity parameter chosen by the practitioner. On the extremes, Λ = 1
corresponds to no-confounding (i.e., classic OPE setting) and Λ = ∞corresponds to maximal-
confounding (i.e., worst or best outcome). We denote the set of all feasible perturbations of P by

3


---Page Break---
U(P), which is an s, a-rectangular set [51]. We define the best- and worst-case Q functions of πt as

Q+(s, a) := supU∈U(P ) Qπt,U(s, a);
Q−(s, a) := infU∈U(P ) Qπt,U(s, a).
(2)

Thus, the goal of this paper is to estimate the best- and worst-case value of πt at the initial state,

V ±
d1 := (1 −γ)Es1∼d1[V ±(s1)].
(3)

where V ±(s) = Ea∼πt(s)[Q±(s, a)] and the ± symbol signals that an equation should be read twice,
once with ± = + and once with ± = −. For clarity, we focus the discussion in the main text on
estimating the worst-case policy value, V −
d1. We provide a similar analysis for policy values under
best-case perturbations (V +
d1) in Appendix B.

Compared to standard OPE, robust OPE is more challenging since the best- and worst-case transition
kernels U ± are unobserved as our dataset D is generated under P. For example, standard OPE is easy
in the on-policy case i.e., if D were generated by πt, but our problem is still “off-data” and non-trivial.

Discounted Visitation Distributions. For any transition kernel U, define the discounted visitation
distribution of πt under U as: dπt,∞
d1,U (s) := (1 −γ) P∞
h=1 γh−1dπt,h
d1,U(s), where dπt,h
d1,U(s) is the
probability of reaching state s in the Markov chain induced by U and policy πt starting from d1(·).
We use d−,∞as shorthand for dπt,∞
d1,U −, where U −denotes the worst-case kernel in U(P).

Bellman-type Operators. For any function f : S × A →R and transition kernel U, recall the
Bellman operator is defined as TUf(s, a) := r(s, a) + γEU[f(s′, πt) | s, a]. For robust OPE,
we define the following robust analog T +
robf(s, a) := r(s, a) + γ supU∈U(P ) EU[f(s′, πt) | s, a]
and T −
robf(s, a) := r(s, a) + γ infU∈U(P ) EU[f(s′, πt) | s, a]. Moreover, we define JUf(s, a) :=
γEU[f(s′, πt) | s, a] −f(s, a). For any linear operator T , also let T ′ denote its adjoint: that is, for
all f, g ∈L2(ν), ⟨f, T g⟩= ⟨T ′f, g⟩, where ⟨·, ·⟩is the inner product in L2(ν).

CVaR −(s, a)
CVaR + (s, a)

[v(s0) ∣s, a]

β −(s, a)
β + (s, a)

v(s0) ∣s, a

Figure 1: Lower and upper CVaRs and
quantiles (β) of v(s′) | s, a distribution.

Conditional Value-at Risk (CVaR). For a random vari-
able X, its upper/lower CVaRs at level τ ∈[0, 1] is defined
as the average outcome of the upper/lower τ-fraction of
cases, and are formally defined as follows [61]:

CVaR+
τ (X) := min
b∈R

b + τ −1E[(X −b)+]
	
,

CVaR−
τ (X) := max
b∈R

b + τ −1E[(X −b)−]
	
,

where y+ := max(0, y) and y−:= min(0, y) for y ∈R.
The optima are attained at the upper/lower τ-th quantile
of X which we denote as β+
τ (X)/β−
τ (X), i.e.,

CVaR+
τ (X) := β+
τ (X)+τ −1E[(X−β+
τ (X))+], CVaR−
τ (X) := β−
τ (X)+τ −1E[(X−β−
τ (X))−].

If X has a cumulative distribution function (CDF) which is differentiable at β±
τ (X), its CVaRs
simplify to CVaR+
τ (X) = E[X | X ≥β+
τ (X)] and CVaR−
τ (X) = E[X | X ≤β−
τ (X)]. In the
paper, τ will often be set to (Λ + 1)−1 ∈[0, 0.5].

Notations. We use x ≲y to mean that x ≤Cy holds for some universal constant C. The
indicator function I[p] takes value 1 if p is true and 0 otherwise. For a measure µ, we let ∥f∥µ :=
(Eµ|f(X)|2)1/2 denote the L2 norm of f, provided it exists. When µ is clear from context, we
also use ∥f∥p := (E|f(X)|p)1/p to denote the Lp norm of f and ∥f∥p,n := (En|f(X)|p)1/p
to denote the empirical analog. For a data sample of size n, we define the empirical mean as
En[f(X)] = 1

n
Pn
i=1 f(xi). For a nuisance function f, we reserve f ∗as its true value and bf as the
learned value from data. Moreover, we employ + and −to denote functions corresponding to best-
and worst-case bounds, respectively. See Appendix A for a comprehensive notation table.

2.1
Background: Non-robust OPE

We provide a quick primer on the double RL (DRL) estimator for classic OPE in non-robust MDPs
[38], which combines estimates of the Q-function and density ratio w to achieve orthogonality,
double robustness and semiparametric efficiency. This sets the stage for our orthogonal estimator

4


---Page Break---
in Section 5, which generalizes DRL to robust MDPs by incorporating the robust Q-function and
density ratio in the worst-case MDP, as described in Section 3 and Section 4 respectively.

The DRL estimator involves two nuisances: (1) q, for which the oracle (true value) is the Q-function
of the target policy Qπt, and (2) w, for which the oracle is the density ratio of the target policy’s
visitation distribution and the data distribution wπt = dd
πt,∞
d1,P/dν. In this section, let η = (w, q) denote
the DRL nuisances (outside this section, we use η to denote our robust estimator’s nuisances) and let
η⋆= (wπt, Qπt) denote their true values, then the recentered efficient influence function (EIF) of V πt
d1
in non-robust MDPs is given by:
ψDRL(s, a, s′; w, q) = V πt
d1 + w(s, a) · (r(s, a) + γq(s′, πt) −q(s, a)).

The DRL estimator uses cross-fitting to learn nuisances bη[k] on all data excluding the k-th fold Dk,
for k = 1, 2, . . . , K and estimates the OPE value via:
bV DRL
d1
= 1

n
PK
k=1
P

(s,a,s′)∈Dk ψDRL(s, a, s′; bη[k]).

As we will see, this paves the way for the EIF of the robust value (Theorem 5.1) and our orthogonal
estimator (Algorithm 3). There are two main guarantees for DRL: double robustness and semipara-
metric efficiency. Let rw
n and rq
n be rate functions depending on n = |D| such that ∥bq[k]−Qπt∥2 ≤rq
n
and ∥bw[k] −wπt∥2 ≤rw
n . Then, DRL enjoys |bV DRL
d1
−V πt
d1 | ≤Op(n−1/2 + rw
n rq
n), which confers
the algorithm double robustness properties. Moreover, if Σope is the efficiency bound (i.e., minimum
achievable asymptotic variance among RAL estimators in nonparametric models for (s, a, s′)), then
√n(bV DRL
d1
−V πt
d1 )
d→N(0, Σope). We seek similar guarantees for our orthogonal robust estimator.

3
Robust Q-Function Estimation with Fitted-Q Evaluation

In this section, we identify the robust Q-function using the robust Bellman equation and then derive
convergence rates for iteratively minimizing the robust Bellman error.

3.1
Identification of the worst-case Q-function

The robust worst-case Q-function of πt, denoted as Q−, satisfies the robust Bellman equation
Q−(s, a) = T −
robQ−(s, a), ∀s, a since the uncertainty set U(P) factorizes over s, a [34]. While these
equations may seem intractable due to the inf in the definition of T −
rob, [12] showed that T −
rob has a
closed form solution in terms of the CVaR under the observed kernel P.
Lemma 3.1. Set τ(s, a) = (Λ(s, a) + 1)−1. Then, for any q : S × A →R,

T −
robq(s, a) = r(s, a) + γΛ−1(s, a)E[v(s′) | s, a] + γ(1 −Λ−1(s, a)) CVaR−
τ(s,a)[v(s′) | s, a],

where v(s′) = Ea′∼πt(s′)[q(s′, a′)], and E, CVaRτ are under the observed kernel P(· | s, a).

Lemma 3.1 implies that Q−is identified via the following equation of observable distributions:
Q−(s, a) = r(s, a)+γΛ−1(s, a)E[Q−(s′, πt) | s, a]+γ(1−Λ−1(s, a)) CVaR−
τ(s,a)[Q−(s′, πt) | s, a].

Under no confounding (Λ(s, a) = 1), this recovers the classic Bellman equation.

3.2
Estimating the Robust Q-Function with Robust FQE

In this section, we estimate Q−via an iterative fitting algorithm based on fitted Q-evaluation (FQE)
[54]. Our algorithm RobustFQE (Algorithm 1) proceeds for M iterations with two main steps in
each iteration i. First, in Line 5, we estimate the lower-quantile of bvi−1(s′) | s, a. Here, we assume
access to an oracle QR for quantile regression, which is a well-established problem, allowing for
the use of various existing algorithms. Second, in Line 6, we solve the tractable robust Bellman
equation in Lemma 3.1 with the CVaR term estimated by its orthogonal estimating equation with the
learned quantiles [57]. By orthogonally estimating CVaR, we achieve second-order dependence on
the quantile estimation errors from the first step. Next, we minimize the mean squared error using a
general function class, Q ⊂S × A 7→[0, (1 −γ)−1].

To enable convergence guarantees, we make two assumptions. First, we assume that the quantile
regression oracle has a specific convergence rate, which can be guaranteed under certain smoothness
conditions [9, 14, 27, 28, 52, 60, 65]. Distributional RL may also be modified to learn quantiles of
the next state value and have shown benefits in practice [21, 22] and in theory [5, 75, 76, 78].

5


---Page Break---
Algorithm 1 RobustFQE: Iterative fitting for estimating Q−and β−
τ .

1: Input: Number of iterations M, Dataset D of size n, Q-function class Q.
2: Initialize bv−
0 (s′) = 0.
3: for i = 1, 2, . . . , M do
4:
Set Di = D[ni/M : n(i + 1)/M].
5:
On the first half of Di, estimate the τ(s, a) lower quantile of bv−
i−1(s′), s′ ∼P(· | s, a).
Let bβ−
i (s, a) denote the learned lower quantiles from the quantile regression oracle QR.
6:
Using the second half of Di, solve the empirical robust Bellman equation by minimizing
squared prediction error for the pseudo-outcome:

bq−
i ←arg minq∈Q
1
|Di|/2
P

(s,a,s′)∈Di[|Di|/2+1:][(y−(s, a, s′) −q(s, a))2], where

y−(s, a, s′) = r(s, a) + γΛ−1(s, a)bv−
i−1(s′) + γ(1 −Λ−1(s, a))

× (bβ−
i (s, a) + τ −1(s, a)(Ea′∼πt(s′)[(bq−
i (s′, a′) −bβ−
i (s, a))−]).

7: Output: bq−
M, bβ−
M.

Assumption 3.2 (QR Oracle). For any v : S 7→[0, (1 −γ)−1], let the true τ(s, a)-quantile of
v(s′), s′ ∼P(s, a) be denoted by βv
τ (s, a). Given a dataset DQR, we assume QR outputs estimates
bβv with bounded ℓ∞error: for any δ, w.p. 1 −δ, ∥bβq −βq
τ∥∞< errQR(|DQR|, δ).

The second assumption is completeness under the robust Bellman T −
rob. Completeness is a standard
assumption in algorithms based on temporal-difference learning and without it, fitted-Q can diverge
or converge to suboptimal fixed points [45, 68].
Assumption 3.3 (Completeness). For all q ∈Q, we have T −
robq ∈Q.

We note that the current proofs of [12, 59] require a stronger completeness: Tβq ∈Q for all q ∈Q
and feasible β. We circumvent the need for the stronger “all-β” completeness by bounding model
misspecification of least squares regression with second order error in the quantile regression.

Finally, we express our bounds with the critical radius εQ
n , a standard tool for deriving fast rates in
statistics; see Appendix D.2 for a summary. Also, we denote the standard concentrability coefficient
with C−
d1 :=
dd−,∞
µ
/dd1

∞, a standard and necessary quantity for OPE.

Theorem 3.4. Let εQ
n denote the critical radius of Q. Under Assumptions 3.2 and 3.3, RobustFQE
ensures that for any δ ∈(0, 1), w.p. 1 −δ,

∥bq−
M −Q−∥d1 ≲(1 −γ)−2(
q

C−
d1 · εQ
n + err2
QR(n/2M, δ/2M)), and
(1 −γ)Ed1[bv−
M(s1)] −V −
d1
 ≲γM + (1 −γ)−1(
q

C−
d1 · εQ
n + err2
QR(n/2M, δ/2M)).

For parametric classes (e.g., finite or linear), the critical radius converges at the standard e
O(n−1/2)
rate. Due to the orthogonal estimation of CVaR, we benefit from a favorable second-order dependence
on errQR which allows for quantile regression to converge at slower e
O(n−1/4) rates. The main
disadvantage of this direct approach is that it converges at a slow sub-√n rate if εQ
n converges at a
sub-√n, e.g., εQ
n converges at a e
O(n−1/4) rate if Q is nonparametric with metric entropy at most
1/t2 [71]. In Section 5, we present an orthogonal estimator that is both robust to slower rates of Q
and achieves semiparametric efficiency.

4
Robust w-Function Estimation with Minimax Learning

Before we present our orthogonal estimator, we study another essential nuisance function: the robust
visitation density ratio, i.e., the robust w-function [2, 39]. In this section, we first identify the worst-
case transition kernel U −in our uncertainty set U(P). Then, we propose a minimax estimator [69]
for the robust w-function, an important nuisance function for our orthogonal estimator in Section 5.

Identification of U −.
The robust transition kernel U −is defined as the feasible perturbed kernel
that achieves the inf in the robust Bellman equation Q−(s, a) = T −
robQ−(s, a). Let F −(y | s, a) =

6


---Page Break---
Algorithm 2 RobustMIL: Minimax Estimation of w± with a Stabilizer

1: Input: Dataset D, prior stage estimate eζ, function classes W, F, stabilizer weight λ > 0.
2: Define weights ξ−(s, a, s′) := Λ−1(s, a) + (1 −Λ−1(s, a))τ −1(s, a)I[eζ(s, a, s′) ≤0].
3: Output:

bw−= arg min
w∈W
max
f∈F En

w(s, a)(γξ−(s, a, s′)f(s′, πt) −f(s, a)) + (1 −γ)Ed1f(s1, πt)


−λ∥γξ−(s, a, s′; eζ)f(s′, πt) −f(s, a)∥2
2,n
(6)

P(V −(s′) ≤y | s, a) be the next-state pushforward measure of the robust value function V −. Then,
U −is a convex combination of the nominal kernel P and a reweighting of P by an indicator function.

Lemma 4.1. Suppose F −(β−
τ (s, a) | s, a) = τ, where β−
τ (s, a) is the lower τ-th quantile of
F −(· | s, a). Then,

U −(s′ | s, a)/P(s′ | s, a) = Λ−1(s, a) + (1 −Λ−1)τ(s, a)−1I[(V −(s′) −β−
τ (s, a)) ≤0].
(4)

The proof strategy decomposes U −into its nominal and perturbed components, leveraging the primal
solution of CVaRτ [3]; we formalize this in Appendix E.2.

Identification of w−.
Using the identification of U −in Lemma 4.1, we can now identify the robust
w-function based on the Bellman flow equations in the worst-case MDP. The Bellman flow in the
robust MDP is given by d−,∞(s) = (1 −γ)d1(s) + γEes∼d−,∞,ea∼πt(es)U −(s | es, ea). where d−,∞(s)
was defined in Section 2. Thus, the robust visitation density, defined as w−(s) := dd−,∞(s)/dν(s),
satisfies the following moment condition for all f : S 7→R:
E[w−(s)f(s)] = (1 −γ)Ed1[f(s1)] + γE[w−(s, a)Es′∼U −(s,a)[f(s′)]],
(5)

where we relaxed notation and defined w−(s, a) := w(s) · πt(a | s)/ν(a | s). As before, in the
unconfounded base (Λ = 1), this result recovers the classic Bellman flow.

4.1
Estimating w−with Robust Minimax Indirect Learning

We now propose a penalized minimax estimator for w−that generalizes the Minimax Indirect
Learning (MIL) of [69] to our robust MDP setting. Our estimator, RobustMIL (Algorithm 2),
leverages a general function class W ⊂S × A 7→R+ to approximately solve the moment equation
in Eq. (5). It does so by minimizing the difference between the left- and right-hand sides of the
equation across a sufficiently large set of adversaries f in a discriminator class F ⊂S × A 7→R.
Since U −is unknown, we approximate it via Eq. (4) by plugging in a threshold eζ(s, a, s′) in the
indicator function to approximate the true threshold ζ−(s, a, s′) := V −(s′) −β−
τ(s,a)(s, a). This
yields the minimax objective in Eq. (6), where we also allow for an optional regularization of the
adversary’s norm which can be useful for obtaining fast convergence rates.

We make the following assumptions for MIL [69]. The first is a regularity condition that (i) our
function class has bounded outputs and (ii) ζ is continuously distributed around the threshold.
Assumption 4.2 (Regularity). (i) supw∈W∪{w−} ∥w∥∞< ∞; (ii) the marginal CDF of V −(s′) −
β−(s, a), i.e., F(y) = P(V −(s′) −β−
τ(s,a)(s, a) ≤y), is boundedly differentiable around 0.

If next-value distribution is discrete, we can use the discrete form of CVaR and (ii) can be removed.

The second is that the adversary class is rich enough to capture all projected errors under the adjoint
of the operator JU −f(s, a) := γEU −[f(s′, πt) | s, a] −f(s, a).
Assumption 4.3 (w−-realizability and completeness). w−∈W and J ′
U −(W −w−) ⊂F.

We note that Assumption 4.3 is monotone in the function class size and can be satisfied by making
the function class more expressive, e.g., increasing size of the neural net. Our algorithms are also
robust to violations in Assumption 4.3, which we show in Appendix G.

We are now ready to state the main estimation result for w−in terms of the critical radius (Ap-
pendix D.2) of the function class.

7


---Page Break---
Algorithm 3 Orthogonal Estimator for V −
d1
1: Input: Dataset D, number of splits K.
2: for k = 1, 2, . . . , K do
3:
Use data D \ Dk to learn (q−,[k], β−,[k]) with Algorithm 1 and w−,[k] with Algorithm 2
4:
for i = ⌊(k −1)n/K⌋, . . . , ⌊kn/K⌋−1 do ψ−
i = ψ(si, ai, s′
i, bη−)

5: Output: bV −
d1 = 1

n
Pn
i=1 ψ−
i .

Theorem 4.4. Let εW
n denote the maximum critical radii of the following classes:

G1 = {(s, a, s′) 7→(f(s, a) −γf(s′, πt)), f ∈F},

G2 =

(s, a, s′) 7→(w(s, a) −w−(s, a))(γf(s′, πt) −f(s, a)), f ∈F, w ∈W
	
.

Under Assumptions 4.2 and 4.3, RobustMIL ensures that for any δ, w.p. 1 −δ,
J ′
U −( bw −w−)

2 ≲εW
n + ∥eζ−−ζ−∥∞+
p

log(1/δ)/n.

As before, the critical radius εW
n converges at an e
O(n−1/2) rate for parametric classes. Notably, our
bounds degrade linearly w.r.t. the ℓ∞error in eζ−for estimating ζ−. For example, if eζ(s, a, s′) =
bv(s′) −bβ(s, a) where bv, bβ are estimated with RobustFQE, then the ζ-error can be bounded by
O(∥bv−v−∥∞+∥bβ−β−∥∞). We present the full proof in Appendix G, where we also present a more
general result that is robust to misspecifications to realizability and completeness (Assumption 4.3).

5
Orthogonal and Efficient Estimator for Robust Policy Value

In this section, we propose an orthogonal estimator that is robust against errors in the nuisances
(exhibiting only second-order sensitivity), achieves semiparametric efficiency, and enables inference.
Our estimator is based on the efficient influence function (EIF) of V −
d1, which is the canonical gradient
of a statistical estimand [67]. The adoption of EIFs for developing efficient estimators is a broadly
employed technique in causal inference [16, 43] and reinforcement learning [35, 39].

We define the collection of nuisance parameters by η−= (w−, q−, β−). The notation bη indicates
that these functions are estimated from data, while the notation η denotes their true values.
Theorem 5.1 ((Recentered) Efficient Influence Function). The (R)EIF of V −
d1 is given by:

ψ(s, a, s′; η−) = V −
d1 + w−(s, a)
 
r(s, a) + γρ−(s, a, s′; v−, β−) −q−(s, a)

,
where

ρ−(s, a, s′; v−, β−) = Λ(s, a)−1v−(s′) + (1 −Λ(s, a)−1)
 
β−(s, a) + τ −1(v−(s′) −β−(s, a))−

.

Remark 5.2. When Λ = 1, there is no shift in the target environment, and the weight on the CVaR
term is zero. The (R)EIF then reduces to the (R)EIF in [39] for regular OPE with an infinite horizon.
As Λ →∞, the CVaR term becomes predominant, with the quantile β−(s, a) taking extreme values.
This yields the (novel) (R)EIF for the problem in [25], where the expected value term is replaced
solely by a CVaR component in the Bellman equation.

The (R)EIF forms the basis of our orthogonal estimator. First, we note that E[ψ(s, a, s′; η−)] is an
unbiased estimator of V −
d1. Furthermore, the expression for ψ(s, a, s′; η−) depends only on quantities
w−, q−, β−which can be estimated from data. Thus, we can cast the expression E[ψ(s, a, s′; η−)]
as a statistical estimand to be learned from the observed sample. This suggests a natural two-stage
estimator that we summarize in Algorithm 3. In the first stage, we estimate the nuisance parameters bη
from the data with K-fold cross-fitting; in the second stage, these estimates are incorporated into the
(R)EIF expression and we calculate the empirical average using the observed data. We summarize
our procedure in Algorithm 3.

The nuisance estimation is detailed in Sections 3.2 and 4.1. The reliance on the EIF confers
our estimator desirable statistical properties including a second order bias due to the nuisances,
meaning the bias has a product structure with respect to the nuisance errors. Thus, this special
structure orthogonalizes away the dependency on bQ−errors which now only appear in second order.
Furthermore, our estimator is semiparametrically efficient in the sense that under mild consistency
assumptions, it achieves minimum variance among all regular and asymptotically linear (RAL)
estimators. We provide theoretical justifications for these properties in the next section.

8


---Page Break---
5.1
Theoretical Guarantees of the Orthogonal Estimator

We now characterize the theoretical properties of our orthogonal estimator. We consider the K-fold
cross-fitted estimator in Algorithm 3 given by

bV −
d1 = 1

n
PK
k=1
P

(s,a,s′)∈Dk ψ(s, a, s′; bη[k]),

where nuisances bη[k], k ∈[K] are trained on all data excluding the kth fold Dk. The following
theorem outlines the theoretical guarantees of this estimator:

Theorem 5.3 (Efficiency of bV −
d1). Let rw
n,p, rq
n,p, rβ
n,p be functions of the same size n = |D| such
that ∥J ′
U −( bw−,[k] −w)∥p ≤rw
n,p, ∥bq−,[k] −q∥p ≤rq
n,p, and ∥β−,[k] −β∥p ≤rβ
n,p for any k ∈[K].
Furthermore, assume that the regularity conditions in Assumption 4.2 hold. Then:

|bV −
d1 −V −
d1| ≲Op(n−1/2) + Op(rw
n,2rq
n,2 + (rq
n,∞)2 + (rβ
n,∞)2)
(Rates)

Furthermore, if rw
n,2 ∨rq
n,2 = op(1), rw
n,2rq
n,2 = op(n−1/2), rq
n,∞= op(n−1/4), and rβ
n,∞=
op(n−1/4), then bV −
d1 satisfies:

√n(bV −
d1 −V −
d1)
d−→N(0, Σ),
Σ = Var(ψ(s, a, s′; η−)).
(Normality & Efficiency)

Moreover, Σ is the minimum achievable asymptotic variance among RAL estimators in the nonpara-
metric model for (s, a, s′) (the efficiency bound).

We provide the intuition along with a detailed proof in Appendix H. The first part of Theorem 5.3
implies that as long as we estimate the nuisances at rates faster that n−1/4, then we can learn bV −
d1 at
parametric rates. The second part of Theorem 5.3 states that under mild consistency assumptions,
our estimator attains the efficiency bound and is asymptotically normal. That means, for example,
we can construct asymptotically valid lower 95%-confidence bound on bV −
d1 by simply subtracting
1.64 times ˆse = 1

n(PK
k=1
P
(s,a,s′)∈Dk(ψ(s, a, s′; bη[k]) −bV −
d1)2)1/2. Then, we can be sure to have a
bound on the worst-case RL policy value, accounting both for potential environment shift and finite
data. Finally, in Appendix J, we describe two settings when our orthogonal estimator remains valid
even if some nuisances are inconsistent, which is a desirable guarantee for sensitivity analysis [23].

Bringing it all together.
We can instantiate Theorem 5.3 with the nuisance estimators from the
previous sections. First, use RobustFQE to estimate bq−and bβ−, ensuring ∥bq−−Q−∥2 ≤O(εQ
n +
err2
QR). Under smoothness conditions (Lemma D.2), the L2 guarantee for bq−implies an L∞guarantee
for bq−, which also ensures an L∞guarantee for bβ−. This ensures max(∥bq−−Q−∥∞, ∥bβ−−β−∥∞)
is well-controlled. Then, we can set eζ−(s, a, s′) = bq−(s′, πt) −bβ−(s, a) and run RobustMIL for
estimating bw−. By Theorem 4.4, its projected-L2 error is O(εW
n + ∥bq−−Q−∥∞+ ∥bβ−−β−∥∞).
Therefore, the final rate via Theorem 5.3 is O((εQ
n + err2
QR) · εW
n + ∥bq−−Q−∥2
∞+ ∥bβ−−β−∥2
∞).

6
Empirical Evaluation

We now provide a proof-of-concept empirical investigation to validate our theoretical findings. We
experiment with our proposed methodology in a simple synthetic environment. First, we discuss our
environment, followed by our approach for solving for the nuisances functions η−. Then, we provide
empirical results for our orthogonal estimator, and compare its performance to weighted or direct
estimators using the Q−or w−nuisances only. The code for our experiments is open-sourced and
available at https://github.com/CausalML/adversarial-ope/.

Experimental Setup
We consider a synthetic MDP with a one-dimensional state and two actions,
modeled after a simple control problem with non-deterministic dynamics. The task is to estimate the
worst-case policy value V −
d1 of a fixed candidate policy πt, across four different constant values of the
sensitivity parameter: Λ(s, a) ∈{1, 2, 4, 8}.

9


---Page Break---
1
2
4
8
(s, a)

0.2

0.4

0.6

0.8

Estimated Vd1

estimator

Q
W
Orth

Λ
Mean squared error (MSE) to true worst-case policy value

Q
W
Orth
1
.000240 ± .000170
.002722 ± .002266
.000005 ± .000006
2
.004053 ± .001329
.003584 ± .002311
.003244 ± .000600
4
.009799 ± .005172
.013862 ± .009228
.008721 ± .002543
8
.006247 ± .003980
.052643 ± .050839
.005713 ± .002730

Figure 2: Results of our synthetic data experiments. We show results for our three estimators on all
four Λ values, over our 10 experiment replications. Above: Box plot summarizing range of policy
value estimates for each combination of estimator and Λ, with Horizontal red dashed lines showing
the true worst-case policy values V −
d1. Below: Table summarizing the corresponding MSE of these
estimators for the true worst-case policy value, along with one standard deviation errors.

We considered three methods for estimating the robust value V −
d1:

1. Q (RobustFQE): Direct method using the estimated robust quality function bQ−only.
2. W (RobustMIL): Importance-sampling method using the estimated robust density ratio bw−only.
3. Orth: Our orthogonal estimator which combines the former two, as described in Algorithm 3.

We performed 10 replications of our experimental procedure, where for each replication we: (1)
sampled a dataset of 20,000 tuples using a different fixed logging policy πb; (2) fit the nuisance
functions Q−, β−, and w−following the method outlined in Algorithms 1 and 2 for each Λ; and (3)
estimated the corresponding robust policy value V −
d1 for all estimators using the fitted nuisances.

Results
We summarize our results in Fig. 2. We note that all of our estimators are consistently
valid for all values of Λ in our experiment. Notably, Orth consistently has the lowest mean squared
error for the true worst-case policy value. In particular, incorporating the robust importance-sampling
weights improves the RobustFQE estimator Q, even though these importance-sampling weights
by themselves (as in W) are much noisier estimators. This is consistent with our theory that the
orthogonal estimator is semiparametrically efficient and insensitive to errors in the nuisance functions.

Full experimental details, including our MDP, target/logging policies, methodology for computing
the true robust policy values V −
d1, and nuisance estimation, are provided in Appendix K. Finally, we
also performed an empirical evaluation in the real-world medical problem of sepsis management
using the MIMIC-III dataset [36]. We detail these results in Appendix L.

7
Conclusion

We consider the problem of infinite-horizon OPE in RL settings when there can be unknown, but
bounded, shifts in the transition distribution compared to the transition distribution generating the
data. This can arise due to unobserved confounding, where observed transitions do not reflect the
true causal ones, non-stationarity in the environment, or adversarial environments. We propose a
sensitivity model for such transition kernel shifts analogous to the classic MSM for static decision
making, and provide theoretical guarantees for identifying and estimating the sharp (i.e., tightest
possible) bounds on the best/worst-case policy value, as well as the corresponding robust Q-function
and state density ratio functions. Our estimator for the best/worst-case policy value is orthogonal
(insensitive to how the nuisance functions are estimated) and achieves semiparametric efficiency
(attaining the best possible asymptotic variance). Finally, our estimator also supports inference,
ensuring we can derive reliable bounds for the robust policy value even with finite data.

10


---Page Break---
Acknowledgements

We thank the anonymous reviewers for their valuable feedback and insightful suggestions. This
material is based upon work supported by the National Science Foundation under Grant Numbers
1846210, IIS-2154711, CAREER 2339395, and by the U.S. Department of Energy, Office of Science,
Office of Advanced Scientific Computing Research, under Award Number DE-SC0023112.

References

[1] Alekh Agarwal, Yuda Song, Wen Sun, Kaiwen Wang, Mengdi Wang, and Xuezhou Zhang.
Provable benefits of representational transfer in reinforcement learning. In The Thirty Sixth
Annual Conference on Learning Theory, pages 2114–2187. PMLR, 2023.

[2] Philip Amortila, Dylan J Foster, Nan Jiang, Ayush Sekhari, and Tengyang Xie. Harnessing
density ratios for online reinforcement learning. arXiv preprint arXiv:2401.09681, 2024.

[3] Marcus Ang, Jie Sun, and Qiang Yao. On the dual representation of coherent risk measures.
Annals of Operations Research, 262:29–46, 2018.

[4] Jean-Yves Audibert and Alexandre B Tsybakov. Fast learning rates for plug-in classifiers under
the margin condition. arXiv preprint math/0507180, 2005.

[5] Alex Ayoub, Kaiwen Wang, Vincent Liu, Samuel Robertson, James McInerney, Dawen Liang,
Nathan Kallus, and Csaba Szepesvári. Switching the loss reduces the cost in batch reinforcement
learning. International Conference of Machine Learning, 2024.

[6] Kishan Panaganti Badrinath and Dileep Kalathil. Robust reinforcement learning using least
squares policy iteration with provable performance guarantees. In International Conference on
Machine Learning, pages 511–520. PMLR, 2021.

[7] Alexandre Belloni, Victor Chernozhukov, Ivan Fernandez-Val, and Christian Hansen. Program
evaluation and causal inference with high-dimensional data. Econometrica, 85(1):233–298,
2017.

[8] Andrew Bennett, Nathan Kallus, and Miruna Oprescu. Low-rank mdps with continuous action
spaces. arXiv preprint arXiv:2311.03564, 2023.

[9] Pallab K Bhattacharya and Ashis K Gangopadhyay. Kernel and nearest-neighbor estimation of
a conditional quantile. The Annals of Statistics, pages 1400–1415, 1990.

[10] Matteo Bonvini, Edward Kennedy, Valerie Ventura, and Larry Wasserman. Sensitivity analysis
for marginal structural models. arXiv preprint arXiv:2210.04681, 2022.

[11] Haïm Brezis and Petru Mironescu. Where sobolev interacts with gagliardo–nirenberg. Journal
of functional analysis, 277(8):2839–2864, 2019.

[12] David Bruns-Smith and Angela Zhou. Robust fitted-q-evaluation and iteration under sequentially
exogenous unobserved confounders. arXiv preprint arXiv:2302.00662, 2023.

[13] David A Bruns-Smith.
Model-free and model-based policy evaluation when causality is
uncertain. In International Conference on Machine Learning, pages 1116–1126. PMLR, 2021.

[14] Domagoj ´Cevid, Loris Michel, Jeffrey Näf, Nicolai Meinshausen, and Peter Bühlmann. Distri-
butional random forests: Heterogeneity adjustment and multivariate distributional regression.
arXiv preprint arXiv:2005.14458, 2020.

[15] Jonathan Chang, Kaiwen Wang, Nathan Kallus, and Wen Sun. Learning bellman complete
representations for offline policy evaluation. In International Conference on Machine Learning,
pages 2938–2971. PMLR, 2022.

[16] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen,
Whitney Newey, and James Robins. Double/debiased machine learning for treatment and
structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018. doi: 10.1111/ectj.12097.

11


---Page Break---
[17] Victor Chernozhukov, Mert Demirer, Esther Duflo, and Ivan Fernandez-Val. Generic machine
learning inference on heterogeneous treatment effects in randomized experiments, with an
application to immunization in india. Technical report, National Bureau of Economic Research,
2018.

[18] Victor Chernozhukov, Carlos Cinelli, Whitney Newey, Amit Sharma, and Vasilis Syrgkanis.
Long story short: Omitted variable bias in causal machine learning. Technical report, National
Bureau of Economic Research, 2022.

[19] Victor Chernozhukov, Whitney Newey, Rahul Singh, and Vasilis Syrgkanis. Automatic debiased
machine learning for dynamic treatment effects and general nested functionals. arXiv preprint
arXiv:2203.13887, 2022.

[20] Yinlam Chow, Aviv Tamar, Shie Mannor, and Marco Pavone. Risk-sensitive and robust decision-
making: a cvar optimization approach. Advances in neural information processing systems, 28,
2015.

[21] Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos. Implicit quantile networks for
distributional reinforcement learning. In International conference on machine learning, pages
1096–1105. PMLR, 2018.

[22] Will Dabney, Mark Rowland, Marc Bellemare, and Rémi Munos. Distributional reinforce-
ment learning with quantile regression. In Proceedings of the AAAI conference on artificial
intelligence, volume 32, 2018.

[23] Jacob Dorn and Kevin Guo. Sharp sensitivity analysis for inverse propensity weighting via
quantile balancing. Journal of the American Statistical Association, 118(544):2645–2657, 2023.

[24] Jacob Dorn, Kevin Guo, and Nathan Kallus. Doubly-valid/doubly-sharp sensitivity analysis for
causal inference with unmeasured confounding. arXiv preprint arXiv:2112.11449, 2021.

[25] Yihan Du, Siwei Wang, and Longbo Huang. Provably efficient risk-sensitive reinforcement
learning: Iterated cvar and worst path. In The Eleventh International Conference on Learning
Representations, 2022.

[26] Yaqi Duan, Chi Jin, and Zhiyuan Li.
Risk bounds and rademacher complexity in batch
reinforcement learning. In International Conference on Machine Learning, pages 2892–2902.
PMLR, 2021.

[27] Anouar El Ghouch and Marc G Genton. Local polynomial quantile regression with parametric
features. Journal of the American Statistical Association, 104(488):1416–1429, 2009.

[28] Kevin Elie-Dit-Cosaque and Véronique Maume-Deschamps. Random forest estimation of
conditional distribution functions and conditional quantiles. Electronic Journal of Statistics, 16
(2):6553–6583, 2022.

[29] Dylan J Foster and Vasilis Syrgkanis. Orthogonal statistical learning. The Annals of Statistics,
51(3):879–908, 2023.

[30] Vineet Goyal and Julien Grand-Clement. Robust markov decision processes: Beyond rectangu-
larity. Mathematics of Operations Research, 48(1):203–226, 2023.

[31] Oliver Hines, Oliver Dukes, Karla Diaz-Ordaz, and Stijn Vansteelandt. Demystifying statistical
learning based on efficient influence functions. The American Statistician, 76(3):292–304, 2022.

[32] Ronald A Howard and James E Matheson. Risk-sensitive markov decision processes. Manage-
ment science, 18(7):356–369, 1972.

[33] Hidehiko Ichimura and Whitney K Newey. The influence function of semiparametric estimators.
Quantitative Economics, 13(1):29–61, 2022.

[34] Garud N Iyengar. Robust dynamic programming. Mathematics of Operations Research, 30(2):
257–280, 2005.

12


---Page Break---
[35] Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning.
In International Conference on Machine Learning, pages 652–661. PMLR, 2016.

[36] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad
Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii,
a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.

[37] Nathan Kallus. What’s the harm? sharp bounds on the fraction negatively affected by treatment.
Advances in Neural Information Processing Systems, 35:15996–16009, 2022.

[38] Nathan Kallus and Masatoshi Uehara. Double reinforcement learning for efficient off-policy
evaluation in markov decision processes. The Journal of Machine Learning Research, 21(1):
6742–6804, 2020.

[39] Nathan Kallus and Masatoshi Uehara. Efficiently breaking the curse of horizon in off-policy
evaluation with double reinforcement learning. Operations Research, 70(6):3282–3302, 2022.

[40] Nathan Kallus and Angela Zhou. Confounding-robust policy evaluation in infinite-horizon
reinforcement learning. Advances in neural information processing systems, 33:22293–22304,
2020.

[41] Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou. Doubly robust distribu-
tionally robust off-policy evaluation and learning. In International Conference on Machine
Learning, pages 10598–10632. PMLR, 2022.

[42] Edward H Kennedy. Nonparametric causal effects based on incremental propensity score
interventions. Journal of the American Statistical Association, 114(526):645–656, 2019.

[43] Edward H Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects.
arXiv preprint arXiv:2004.14497, 2020.

[44] Amirhossein Kiani, Chris Wang, and Angela Xu. Sepsis world model: A mimic-based openai
gym" world model" simulator for sepsis treatment. arXiv preprint arXiv:1912.07127, 2019.

[45] J Kolter. The fixed points of off-policy td. Advances in Neural Information Processing Systems,
24, 2011.

[46] Navdeep Kumar, Kfir Levy, Kaixin Wang, and Shie Mannor. Efficient policy iteration for robust
markov decision processes via regularization. arXiv preprint arXiv:2205.14327, 2022.

[47] Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Yehuda Levy, and Shie Mannor. Policy
gradient for rectangular robust markov decision processes. In Thirty-seventh Conference on
Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=
NLpXRrjpa6.

[48] Mark J Laan and James M Robins. Unified methods for censored longitudinal data and causality.
Springer, 2003.

[49] Giovanni Leoni. A first course in Sobolev spaces. American Mathematical Soc., 2017.

[50] Greg Lewis and Vasilis Syrgkanis. Double/debiased machine learning for dynamic treatment
effects. In NeurIPS, pages 22695–22707, 2021.

[51] Shie Mannor, Ofir Mebel, and Huan Xu. Robust mdps with k-rectangular uncertainty. Mathe-
matics of Operations Research, 41(4):1484–1509, 2016.

[52] Nicolai Meinshausen and Greg Ridgeway. Quantile regression forests. Journal of machine
learning research, 7(6), 2006.

[53] Volodymyr Mnih.
Playing atari with deep reinforcement learning.
arXiv preprint
arXiv:1312.5602, 2013.

[54] Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of
Machine Learning Research, 9(5), 2008.

13


---Page Break---
[55] Hongseok Namkoong, Ramtin Keramati, Steve Yadlowsky, and Emma Brunskill. Off-policy
policy evaluation for sequential decisions under unobserved confounding. Advances in Neural
Information Processing Systems, 33:18819–18831, 2020.

[56] Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain
transition matrices. Operations Research, 53(5):780–798, 2005.

[57] Tomasz Olma. Nonparametric estimation of truncated conditional expectation functions. arXiv
preprint arXiv:2109.06150, 2021.

[58] Miruna Oprescu, Jacob Dorn, Marah Ghoummaid, Andrew Jesson, Nathan Kallus, and Uri Shalit.
B-learner: Quasi-oracle bounds on heterogeneous causal effects under hidden confounding. In
International Conference on Machine Learning, pages 26599–26618. PMLR, 2023.

[59] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust rein-
forcement learning using offline data. Advances in neural information processing systems, 35:
32211–32224, 2022.

[60] Jeffrey S Racine and Kevin Li. Nonparametric conditional quantile estimation: A locally
weighted quantile kernel approach. Journal of Econometrics, 201(1):72–94, 2017.

[61] R Tyrrell Rockafellar and Stanislav Uryasev. Conditional value-at-risk for general loss distribu-
tions. Journal of banking & finance, 26(7):1443–1471, 2002.

[62] Anton Schick. On asymptotically efficient estimation in semiparametric models. The Annals of
Statistics, pages 1139–1151, 1986.

[63] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[64] Vira Semenova and Victor Chernozhukov. Debiased machine learning of conditional average
treatment effects and other causal functions. The Econometrics Journal, 24(2):264–289, 2021.

[65] Ichiro Takeuchi, Quoc V Le, Timothy D Sears, Alexander J Smola, and Chris Williams.
Nonparametric quantile estimation. Journal of machine learning research, 7, 2006.

[66] Zhiqiang Tan. A distributional approach for causal inference using propensity scores. Journal
of the American Statistical Association, 101(476):1619–1637, 2006.

[67] Anastasios A Tsiatis. Semiparametric theory and missing data, volume 4. Springer, 2006.

[68] John Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function
approximation. Advances in neural information processing systems, 9, 1996.

[69] Masatoshi Uehara, Masaaki Imaizumi, Nan Jiang, Nathan Kallus, Wen Sun, and Tengyang Xie.
Finite sample analysis of minimax offline reinforcement learning: Completeness, fast rates and
first-order efficiency. arXiv preprint arXiv:2102.02981, 2021.

[70] Mark J van der Laan, Sherri Rose, Wenjing Zheng, and Mark J van der Laan. Cross-validated
targeted minimum-loss-based estimation. Targeted learning: causal inference for observational
and experimental data, pages 459–474, 2011.

[71] Aad W Van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 2000.

[72] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48.
Cambridge university press, 2019.

[73] Jie Wang, Rui Gao, and Hongyuan Zha. Reliable off-policy evaluation for reinforcement
learning. Operations Research, 72(2):699–716, 2024.

[74] Kaiwen Wang, Nathan Kallus, and Wen Sun. Near-minimax-optimal risk-sensitive reinforce-
ment learning with cvar. In International Conference on Machine Learning, pages 35864–35907.
PMLR, 2023.

14


---Page Break---
[75] Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, and Wen Sun. The benefits of being
distributional: Small-loss bounds for reinforcement learning. Advances in Neural Information
Processing Systems, 36, 2023.

[76] Kaiwen Wang, Nathan Kallus, and Wen Sun. The central role of the loss function in reinforce-
ment learning. arXiv preprint arXiv:2409.12799, 2024.

[77] Kaiwen Wang, Dawen Liang, Nathan Kallus, and Wen Sun. Risk-sensitive rl with optimized
certainty equivalents via reduction to standard rl. arXiv preprint arXiv:2403.06323, 2024.

[78] Kaiwen Wang, Owen Oertell, Alekh Agarwal, Nathan Kallus, and Wen Sun. More benefits of
being distributional: Second-order bounds for reinforcement learning. International Conference
of Machine Learning, 2024.

[79] Yue Wang and Shaofeng Zou. Online robust reinforcement learning with model uncertainty.
Advances in Neural Information Processing Systems, 34:7193–7206, 2021.

[80] Wolfram Wiesemann, Daniel Kuhn, and Berç Rustem. Robust markov decision processes.
Mathematics of Operations Research, 38(1):153–183, 2013.

[81] Wenhao Xu, Xuefeng Gao, and Xuedong He.
Regret bounds for Markov decision pro-
cesses with recursive optimized certainty equivalents. In Andreas Krause, Emma Brunskill,
Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceed-
ings of the 40th International Conference on Machine Learning, volume 202 of Proceed-
ings of Machine Learning Research, pages 38400–38427. PMLR, 23–29 Jul 2023. URL
https://proceedings.mlr.press/v202/xu23d.html.

15


---Page Break---
Appendices

A
Notations

Table 1: List of Notations

S, A
State and action spaces.
∆(S)
The set of distributions supported by set S.
d1
The initial state distribution.
Λ(s, a)
Tolerance parameter for kernel shift at (s, a). Takes values [1, ∞].
τ(s, a)
τ(s, a) =
1
1+Λ(s,a) ∈[0, 1

2].

V ±, Q±
Robust value and quality functions of the target policy πt.
f(s, π)
f(s, π) := Ea∼π(s)[f(s, a)].
U ±(s′ | s, a)
Robust transition kernel which attains the best- or worst-case value.
TU, T ±
rob
Bellman operator under U and the robust Bellman operators.
JU
JUf(s, a) := γEU[f(s′, πt) | s, a] −f(s, a)
β±
τ (s, a)
The upper τ-th quantile of V +(s′) and lower τ-th quantile of V −(s′), s′ ∼P(s, a).
dπt,∞
d1,U
The γ-discounted average visitation of πt under MDP with transition U starting from d1.
d±,∞
d±,∞= dπt,∞
d1,U ±.
ν(s), ν(s, a)
Data generating distribution. ν(s) marginalizes over actions.
w±
w± = dd±,∞/dν. This is valid both as a function of s or (s, a).
ω(s, a)
ω(s, a) = πt(a|s)

ν(a|s) .

x+, x−
max(0, x), min(0, x) respectively, for x ∈R.
x ≲y
x ≤Cy for some constant C.
En
Empirical average over n samples.
∥f∥p
Lp norm, (E|f(X)|p)1/p.
f ⋆
True (oracle) value of a parameter or function f.
f, ¯f
Putative value of a parameter or function f.
bf
Estimated value of a parameter or function f.

B
Results for Policy Evaluation Under Best-Case Perturbations

In this section, we present analogous results for the best-case perturbation under the uncertainty set,
corresponding to the supremum case of Eq. (2). We derive a similar orthogonal estimator with the
properties outlined in Theorem 5.3, following the same reasoning presented in the main text.

Q+ Identification and Estimation.
We present the results of Lemma 3.1 for T +
rob:

T +
robq(s, a) = r(s, a) + γΛ−1(s, a)E[v(s′) | s, a] + γ(1 −Λ−1(s, a)) CVaR+
τ(s,a)[v(s′) | s, a].

Next, applying Assumption 3.2 and Assumption 3.3 to T +
rob, we derive from Theorem 3.4 for Q−
that:

∥bq+
M −Q+∥d1 ≲(1 −γ)−2(
q

C+
d1 · εQ
n + err2
QR(n/2M, δ/2M)), and
(1 −γ)Ed1[bv+
M(s1)] −V +
d1
 ≲γM + (1 −γ)−1(
q

C+
d1 · εQ
n + err2
QR(n/2M, δ/2M)).

w+ Identification and Estimation.
We first state the identification result for U −as in Lemma 4.1:

U +(s′ | s, a)/P(s′ | s, a) = Λ−1(s, a) + (1 −Λ−1)τ(s, a)−1I[(V +(s′) −β+
τ (s, a)) ≥0].

16


---Page Break---
Algorithm 4 Orthogonal Estimator for V +
d1
1: Input: Dataset D, number of splits K.
2: for k = 1, 2, . . . , K do
3:
Use data D \ Dk to learn (q+,[k], β+,[k]) with Algorithm 1 and w+,[k] with Algorithm 2
4:
for i = ⌊(k −1)n/K⌋, . . . , ⌊kn/K⌋−1 do ψ+
i = ψ(si, ai, s′
i, bη+)

5: Output: bV +
d1 = 1

n
Pn
i=1 ψ+
i .

Then, under Assumption 4.2 and Assumption 4.3 formulated for U +, the minimax rates from
Theorem 4.4 are given by:
J ′
U +( bw −w+)

2 ≲εW
n + ∥eζ+ −ζ+∥∞+
p

log(1/δ)/n.

Orthogonal and Efficient Estimator for V +
d1.
Let the set of nuisance parameters be denoted by
η+ = (w+, q+, β+). Then, the (recentered) efficient influence function (R)EIF (see Theorem 5.1)
for in V +
d1 is formulated as:

ψ(s, a, s′; η+) = V +
d1 + w+(s, a)
 
r(s, a) + γρ+(s, a, s′; v+, β+) −q+(s, a)

,
where

ρ+(s, a, s′; v+, β+) = Λ(s, a)−1v+(s′) + (1 −Λ(s, a)−1)
 
β+(s, a) + τ −1(v+(s′) −β+(s, a))+

.

Using this (R)EIF, the orthogonal estimator for V +
d1 is presented in Algorithm 4. We now restate
Theorem 5.3 for bV +
d1:

Theorem B.1 (Efficiency of bV +
d1). Let rw
n,p, rq
n,p, rβ
n,p be functions of n
=
|D| such that
∥J ′
U +( bw+,[k] −w∗)∥p ≤rw
n,p, ∥bq+,[k] −q∗∥p ≤rq
n,p, and ∥β+,[k] −β∗∥p ≤rβ
n,p for any k ∈[K].
Furthermore, assume that the regularity conditions in Assumption 4.2 hold. Then:

|bV +
d1 −Vd1| ≲Op(n−1/2) + Op(rw
n,2rq
n,2 + (rq
n,∞)2 + (rβ
n,∞)2)
(Rates)

Furthermore, if rw
n,2 ∨rq
n,2 = op(1), rw
n,2rq
n,2 = op(n−1/2), rq
n,∞= op(n−1/4), and rβ
n,∞=
op(n−1/4), then bV +
d1 satisfies:
√n(bV +
d1 −Vd1)
d−→N(0, Σ),
Σ = Var(ψ(s, a, s′; η+)).
(Normality & Efficiency)
Moreover, Σ is the minimum achievable asymptotic variance among RAL estimators in the nonpara-
metric model for (s, a, s′) (the efficiency bound).

C
Additional Related Works

Robust MDPs.
There is a rich literature on Robust MDPs [30, 34, 51, 80] with s, a-rectangular
uncertainty sets, but these foundational works assumed knowledge of the transition kernel. Recently,
learning-based robust MDP algorithms have been proposed for uncertainty sets under the total
variation [47, 59] and more generally Lp balls [46]. These Lp uncertainty sets are additive in nature,
i.e., the adversary adds or subtracts a vector in the ℓp ball to P(· | s, a), whereas our uncertainty set
is multiplicative in nature, i.e., the adversary can multiply or divide a bounded factor and is more
commonly used in causal inference to model unobserved confounding. In the contextual bandit
setting, [41] also derived efficiency bounds for robust OPE where both state distribution and reward
distributions may shift – their work is however restricted to the one-step bandit setting while our full
RL setting is more challenging.

Risk-Sensitive RL.
Risk-sensitive RL is the problem of optimizing the risk measure of cumulative
rewards [32] and is tightly related to robust MDPs [20]. For example, as we proved in Lemma 3.1,
the MSM uncertainty set is indeed equivalent to risk-sensitive RL with the dynamic risk measure
ΛE + (1 −Λ) CVaRτ. We note that efficient online RL algorithms have been proposed for similar
measures [25, 81]. Static risk-sensitive RL also modifies the Bellman equations in an augmented
MDP [74, 77]. Our focus is on deriving the optimal off-policy evaluation estimators for the problem,
which involves a different set of challenges such as deriving the efficiency bound and ensuring
sharpness guarantees even when nuisances are estimated slowly.

17


---Page Break---
D
Additional Technical Details

D.1
Higher Order Norms via Smoothness

For any x ∈R+, define ⌊x⌋as the greatest integer that is strictly less than x, and let x and
{x} = x−⌊x⌋represent the fractional part. Thus, we obtain the distinct decomposition x = ⌊x⌋+{x},
where ⌊x⌋∈N and {x} ∈(0, 1].

Definition D.1 (α-smooth functions). Given α ∈(0, ∞) and X ⊆Rm, f : X →R is an α-smooth
function if (1) the mixed derivatives up to ⌊α⌋-order exist and are bounded; and (2) all ⌊α⌋-order
derivatives are {α}-Hölder continuous [49].

Lemma D.2 (L∞Bound for α-Smooth Functions). Let f : X →R, X ⊆Rm be an α-smooth
function as in Definition D.1. Then, if X is Rm, a half-space or a bounded Lipschitz domain in Rm,
there exists a constant C such the following inequality holds:

∥f∥∞≤C∥f∥

pα
pα+m
p
.

Proof. This lemma is a direct application of the fractional Gagliardo-Nirenberg interpolation inequal-
ity (Theorem 1 in [11]) from the functional analysis literature. For a more comprehensive exposition
on this result, see Appendix A.1 in [8].

D.2
Localized Rademacher Complexity and Critical Radius

Here, we recap the localized Rademacher complexity and critical radius which is a standard complex-
ity measure for obtaining fast rates for squared loss [72]. Let G be a class of functions g : Z →R.
Given n datapoints z1, z2, . . . , zn, the empirical localized Rademacher complexity is:

Rn(ε, G) := Eσ

"

sup
g∈G:∥g∥n≤ε

1
n

n
X

i=1
ϵig(zi)

#

,

where Eσ is expectation over n independent Rademacher random variables σ1, σ2, . . . , σn, i.e.,
Eσ[·] =
1
2n
P

σ∈{−1,1}n[·]. Note that when ε = ∞, there is no localization and Rn(∞, G) reduces
to the vanilla Rademacher complexity. Let C := supg∈G ∥g∥∞be the envelope of G. Then, the
critical radius of G with n, called εn, is the smallest ε that satisfies Rn(ε, G) ≤ε2/C.

Unless otherwise stated, we will posit that G is star-shaped: there exists g0 ∈G such that for all
g ∈G and α ∈[0, 1], we have αg0 + (1 −α)g ∈G. If not, we can replace G by its star-hull, i.e., the
smallest star-shaped set containing G. We will also posit that G is symmetric for simplicity.

The critical radius is a well-studied quantity in statistics [72] and also recently in RL [26, 69].
For example if G has d VC-subgraph dimension, then w.p. 1 −δ, εn ≤O(
p

d log n/n). For
nonparametric models with metric entropy at most 1/tβ, the critical radius can also be bounded by
O(n−1/(max(2+β,2β))) [69], e.g., is O(n−1/4) if β = 2.

E
Proofs for Identification Results

E.1
Identification of robust Q

Lemma 3.1. Set τ(s, a) = (Λ(s, a) + 1)−1. Then, for any q : S × A →R,

T −
robq(s, a) = r(s, a) + γΛ−1(s, a)E[v(s′) | s, a] + γ(1 −Λ−1(s, a)) CVaR−
τ(s,a)[v(s′) | s, a],

where v(s′) = Ea′∼πt(s′)[q(s′, a′)], and E, CVaRτ are under the observed kernel P(· | s, a).

18


---Page Break---
Proof. Consider the uncertainty set in Trob where the constraint on U (Eq. (1)) can be rewritten as:

0 ≤U(s′|s,a)−Λ−1(s,a)P (s′|s,a)

P (s′|s,a)
≤Λ(s, a) −Λ−1(s, a).

Therefore, we can write U(s′ | s, a) = Λ−1(s, a)P(s′ | s, a) + (1 −Λ−1)G(s′ | s, a) where we
define G(s′ | s, a) := U(s′|s,a)−Λ−1(s,a)P (s′|s,a)

1−Λ−1(s,a)
. Thus, the constraints on G are that G(· | s, a) ≪

P(· | s, a) and ∥dG(s′|s,a)

dP (s′|s,a)∥≤Λ(s, a) + 1. Setting τ(s, a) =
1
Λ(s,a)+1, we can apply the primal
form of CVaR [3, 24] to obtain

inf
G≪P :∥dG(·|s,a)

dP (·|s,a) ∥∞≤τ −1(s,a)
EG[f(s′)] = CVaR−
τ(s,a)[f(s′) | s, a].

Therefore, the supremum in Trob can be expressed as Λ−1(s, a) times the expectation under nominal
P and (1 −Λ−1(s, a)) times the above CVaR expression, which finishes the proof of the −case.

For the + case, we can simply use sup instead of inf and upper CVaR instead of lower CVaR.

E.2
Identification of robust kernel and visitation

Lemma 4.1. Suppose F −(β−
τ (s, a) | s, a) = τ, where β−
τ (s, a) is the lower τ-th quantile of
F −(· | s, a). Then,

U −(s′ | s, a)/P(s′ | s, a) = Λ−1(s, a) + (1 −Λ−1)τ(s, a)−1I[(V −(s′) −β−
τ (s, a)) ≤0].
(4)

Lemma E.1. Fix any v : S →R and define the pushforward Fv(y | s, a) = P(v(s′) ≤y | s, a).
Suppose Fv(β±
τ,Fv(·|s,a)(s, a) | s, a) = 1

2 ± ( 1

2 −τ), where β±
τ,Fv is the upper/lower τ-quantile
of Fv. Then, supU∈U(P ) EU[v(s′) | s, a] = Es′∼U +
v (s,a)[v(s′)] and infU∈U(P ) EU[v(s′) | s, a] =
Es′∼U −
v (s,a)[v(s′)], where

U ±
v (s′ | s, a)/P(s′ | s, a) = Λ−1(s, a) + (1 −Λ−1)τ(s, a)−1I[±(v(s′) −β±
τ,Fv(·|s,a)(s, a)) ≥0].

Proof. We start with some intuitions. First, if the CDF of v(s′) is differentiable β+
τ (s, a), then
CVaR+
τ (v(s′) | s, a) = E[v(s′) | f(s′) ≥β+
τ (s, a), s, a] and the result follows immediately from
Lemma 3.1 by noticing that the form of U + exactly recovers the convex combination of expectation
and CVaR. Alternatively, one can use the closed form solution of the primal CVaR as derived in [3]
to obtain the result.

We now provide a formal proof. Fix any s, a and let τ = τ(s, a). Fix any function v(s′) ∈R.
We want to show that the worst-case U + = arg maxU∈U(P ) EU[v(s′) | s, a] has a closed
form expression as shown in line 725.
By the proof of Lemma 3.1 above, we can rewrite
U +(s′
| s, a) = Λ−1(s, a)P(s′
| s, a) + (1 −Λ−1(s, a))G+(s′
| s, a), where G+
=
arg maxG≪P :|dG(·|s,a)/dP (·|s,a)|∞≤τ −1(s,a) EG[v(s′)]. Thus, it suffices to simplify G+. To do so, we
invoke the premise that the CDF of v(s′) is differentiable at β+
τ , i.e. Fv(β+
τ,Fv(s, a) | s, a) = 1 −τ.
This implies that the CVaR is exactly the conditional expectation of the 1 −τ(s, a)-fraction of best
outcomes, i.e. CVaR+
τ (v(s′) | s, a) = E[v(s′) | v(s′) ≥β+
τ (s, a), s, a], which in turn is equal to
τ −1E[v(s′)I[v(s′) ≥β+
τ (s, a)] | s, a]. Thus, G+(s′ | s, a) = τ −1P(s′ | s, a)I[v(s′) ≥β+
τ (s, a)].
This concludes the proof for the + case. The proof for the −case follows identical steps.

F
Proofs for Robust FQE

We prove a more general result with approximate completeness, which shows that Theorem 3.4 is
robust to approximate completeness.

Assumption F.1 (Approximate Completeness). maxq∈Q ming∈Q ∥g −T ±
CVaRq∥ν ≤εQComp.

19


---Page Break---
Theorem F.2. Assume Assumption F.1. Under the same setup as Theorem 3.4, we have
bq±
K −Q±
µ ≲
1
(1 −γ)2 (
q

C±
µ ·
 
εQ
n + εQComp

+ err2
QR(n/2K, δ/2K)),

and
V ±
d1 −(1 −γ)Ed1[bq±
K(s1, πt)]
 ≲γK +
1
1 −γ (
q

C±
µ ·
 
εQ
n + εQComp

+ err2
QR(n/2K, δ/2K)).

Proof. Let U ± denote the worst-case kernel that satisfies V ±
d1 = (1 −γ)Ed1V πt
U ±(s1). Then,

V ±
d1 −(1 −γ)Ed1[bq±
K(s1, πt)] = (1 −γ)Ed1[V πt
U ±(s1) −bqK(s1, πt)]

= Edπ,∞
U± [T πt
U ±bqK(s, a) −bqK(s, a)]
(Lemma F.3)

≤
4
1 −γ
max
k=1,2,...

bqk −T πt
U ±bqk−1

d
πt,∞

U± + γK/2.
(Lemma F.4)

Consider any k = 1, 2, . . . . By definition of U ±, we have
bqk −T πt
U ±bqk−1

d
πt,∞

U±
=
bqk −T ±
β⋆
k bqk−1

d±,∞,
(by def of U ±)

where β⋆
k(s, a) is the true quantile of bvk−1(s′). Denote q⋆
k := T ±
robbqk−1 and let β⋆
k be the true
upper/lower quantile of bqk−1. Recall the population loss function is

Lk(q, β) := E

yβ
k (s, a, s′) −q(s, a)
2

yβ
k (s, a, s′) = r(s, a) + γΛ−1(s, a)bvk−1(s′)

+ γ(1 −Λ−1(s, a))
 
β(s, a) + τ −1(s, a)(bvk−1(s′) −β(s, a))±

.

The empirical loss bLk(q, β) is if E is replaced by En. Note that bqk = arg minq∈Q bLk(q, bβk).

Nonparametric Least Squares with Model Misspecification.
We will directly invoke [72, Theo-
rem 13.13], which gives a fast rate for misspecified least squares with general nonparametric classes.
We now bound the misspecification. Recall that at the k-th iteration, our regression Bayes-optimal is
E[y
bβk
k (s, a, s′) | s, a] = Tbβk bqk−1(s, a). By Lemma H.3, we know this is close to Tβ⋆
k bqk−1(s, a) with
second order errors in β: for any µ, we have
T ±
bβk bqk−1 −T ±
β⋆
k bqk−1

d±,∞
µ
≲∥bβk −β⋆
k∥2
∞.

Finally, by approximate completeness (Assumption F.1), there exists g
∈
Q such that
∥Tβ⋆
k bqk−1(s, a) −g∥≤εQComp. Putting this together: for any k, there exists a g ∈Q such
that

∥g −Tbβk bqk−1(s, a)∥d±,∞
µ
≤∥g −Tβ⋆
k bqk−1(s, a)∥d±,∞
µ
+ ∥Tβ⋆
k bqk−1(s, a) −Tbβk bqk−1(s, a)∥d±,∞
µ

≤
q

C±
µ · εQComp + ∥bβk −β⋆
k∥2
∞.

Therefore, [72, Theorem 13.13] (and concentration of least squares) certifies that:
bqk −Tbβk bqk−1

d±,∞≲
q

C±
µ ·
 
εQComp + εn

+ ∥bβk −β⋆
k∥2
∞.

Therefore, we have proven:
bqk −T ±
β⋆
k bqk−1

d±,∞
µ
≤
bqk −T ±
bβk bqk−1

d±,∞
µ
+
T ±
bβk bqk−1 −T ±
β⋆
k bqk−1

d±,∞
µ

≲
q

C±
µ ·
 
εQComp + εn

+ ∥bβk −β⋆
k∥2
∞.

This concludes the proof.

20


---Page Break---
Lemma F.3 (Performance Difference). For any π, transition kernel P, and function f : S × A →R,
we have

V π
P −Es∼d1[f(s, π)] =
1
1 −γ Edπ,∞
P
[T π
P f(s, a) −f(s, a)].

Proof. See Lemma C.1 of [15].

Lemma F.4 (Unrolling). For any π, transition kernel P, and functions f0, f1, . . . , fK : S × A →R
satisfying f0(s, a) = 0, we have ∥fK −T π
P fK∥dπ,∞
P
≤
4
1−γ maxk=1,2,...∥fk −T π
P fk−1∥dπ,∞
P
+

γK/2.

Proof. See Lemma C.2 of [15].

G
Proofs for Robust Minimax Algorithm

Assumption G.1 (Approximate W-realizability and completeness). Assume the following hold for
W and F:
(A) Approximate realizability: minw∈W∥JU ±(w± −w)∥2 ≤εWReal;
(B) Approximate completeness: maxw∈W minf∈F
f −J ′
U ±(w −w±)

2 ≤εWComp.

We prove a more general result with approximate realizability and completeness, which implies
Theorem 4.4 that is robust to misspecification in its assumptions.

Theorem G.2. Under Assumption G.1 and the same setup as Theorem 4.4, we have

J ′
U ±( bw −w±)

2 ≲εW
n + ∥eζ± −ζ±∥∞+

r

log(1/δ)

n
+ εWReal + εWComp.

Proof. For this proof, we focus on the worst-case kernel P ⋆of the form
P ⋆(s′|s,a)

P (s′|s,a)
=
τ −1(s, a)I[ζ⋆(s, a, s′) ≤0] where ζ⋆(s, a, s′) = V −(s′) −β−(s, a). This corresponds to the
pure CVaR case of T −
rob; the E part is identical to standard non-robust RL so we omit it. The best-case
kernel U + can be handled similarly. Let bP(s′ | s, a) denote our estimated robust kernel, which

satisfies
b
P (s′|s,a)
P (s′|s,a) = τ −1(s, a)I[bζ(s, a, s′) ≤0], where bζ(s, a, s′) is the given prior stage estimate of
ζ⋆(s, a, s′) = V −(s′) −β−(s, a).

The key and only difference between our Algorithm 2 and the MIL algorithm ( bwmil) of [69] is that
our next-state samples are importance weighted with ξ±(s, a, s′), which is the density ratio of the
estimated robust kernel bP(s′ | s, a) and the nominal kernel P(s′ | s, a). Note also that ξ±(s, a, s′) ≤
τ −1(s, a) < ∞, and hence |En[ζ(s, a, s′)f(s′)] −Es,a∼ν,s′∼b
P (s,a)[f(s′)]| ≲
p

log(1/δ)/n w.p.

1 −δ. Therefore, up to O(
p

log(1/δ)/n) errors, our Algorithm 2 can be viewed as MIL applied to
the MDP with kernel bP.

To invoke the result of [69, Theorem 6.1] (in MDP with kernel bP), we need to show that its
assumptions are met by bounding the model misspecification, i.e., Eq. (6) and Appendix C of [69].
Note that these misspecifications are w.r.t. the MDP with kernel bP, since this is the MDP in
which we’re applying Theorem 6.1 of [69]. Specifically, the two errors we need to bound are, (A)
approximate realizability: εA = minw∈W ∥J ′
b
P (w b
P −w)∥2; and (B) approximate completeness:
εB = maxw∈W minf∈F ∥f −J ′
b
P (w −w b
P )∥2 where recall that JP is the linear operator defined as
JP f(s, a) := γEP [f(s′, πt) | s, a] −f(s, a) and J ′
P is the adjoint.

21


---Page Break---
Bounding misspecifications by ∥bζ −ζ⋆∥∞.
Since ζ⋆(s, a, s′) has a marginal CDF that’s boundedly
differentiable around 0 (i.e., (ii) of Assumption 4.2), [37, Lemma 3] implies that ζ⋆(s, a, s′) satisfies
a 1-margin (Definition H.2). Hence, Lemma H.3 and the continuity of ζ⋆(s, a, s′) implies that

Pr

I[bζ(s, a, s′) ≤0] ̸= I[ζ⋆(s, a, s′) ≤0]


= Pr

(I[bζ(s, a, s′) ≤0] ̸= I[ζ⋆(s, a, s′) ≤0]), ζ⋆(s, a, s′) ̸= 0

≲∥bζ −ζ⋆∥∞,

Thus, for any v : S →R,

E
(E b
P −EP ⋆)[v(s′) | s, a]
 ≤E[τ −1(s, a)(I[bζ(s, a, s′) ≤0] ̸= I[ζ⋆(s, a, s′) ≤0]) · |v(s′)|]

≲∥v∥∞· Pr

I[bζ(s, a, s′) ≤0] ̸= I[ζ⋆(s, a, s′) ≤0]


≲∥v∥∞∥bζ −ζ⋆∥∞,

or equivalently
E∥bP(· | s, a) −P ⋆(· | s, a)∥TV ≲∥bζ −ζ⋆∥∞.
(7)

Equipped with Eq. (7), we can now bound the following two types of errors: (i) ⟨f, (TP ⋆−T b
P )g⟩,
and (ii) ⟨w ˆ
P −wP ⋆, h⟩, where f, g : S × A →R and h : S →R, and TP and wP are the Bellman
operator and visitation density of target policy πt in the MDP with kernel P.

For (i):
⟨f, (JP ⋆−J b
P )g⟩
 =
E[f(s, a)
 
γ(EP ⋆−E b
P )[g(s′, πt) | s, a]

]


≤γ∥f∥∞E
(EP ⋆−E b
P )[g(s′, πt) | s, a]


≲γ∥f∥∞∥g(·, πt)∥∞∥bζ −ζ⋆∥∞.

For (ii):

⟨w b
P −wP ⋆, h⟩= E[(w b
P (s) −wP ⋆(s))h(s)]

≤∥h∥∞∥d b
P −dP ⋆∥TV

≤∥h∥∞
γ
1 −γ EdP ⋆∥bP(· | s, a) −P ⋆(· | s, a)∥TV
(Eq. (9))

≲C∥h∥∞
γ
1 −γ E∥bP(· | s, a) −P ⋆(· | s, a)∥TV
(Assumption 4.2(i))

≲C∥h∥∞
γ
1 −γ ∥bζ −ζ⋆∥∞,

where C = ∥ddP ⋆/dν∥∞< ∞.

For approximate realizability (εA): for any w ∈W, we have

∥J ′
b
P (w b
P −w)∥2
≤∥(J b
P −JP ⋆)′(w b
P −w)∥2 + ∥J ′
P ⋆(w b
P −wP ⋆)∥2 + ∥J ′
P ⋆(w⋆−w)∥2
= ⟨w b
P −w, (J b
P −JP ⋆)g1⟩+ ⟨w b
P −wP ⋆, JP ⋆g2⟩+ ∥J ′
P ⋆(w⋆−w)∥2

≲∥bζ −ζ⋆∥∞+ ∥J ′
P ⋆(w⋆−w)∥2
where g1
=
((JP ⋆−J b
P )′(w b
P −w))/∥(JP ⋆−J b
P )′(w b
P −w)∥2, g2
=
(J ′
P ⋆(w b
P −
wP ⋆))/∥J ′
P ⋆(w b
P −wP ⋆)∥2. The last inequality uses (i) and (ii) with the fact that ∥g1∥∞< ∞and
∥g2∥∞< ∞as the w terms are bounded by our premise. Therefore, taking min over w and using
Assumption G.1, we have εA ≲∥bζ −ζ⋆∥∞+ εWReal.

For approximate completeness (εB): for any w ∈W and f ∈F, we have

∥f −J ′
b
P (w −w b
P )∥2
≤∥f −J ′
P ⋆(w −wP ⋆)∥2 + ∥(JP ⋆−J b
P )′(w −wP ⋆)∥2 + ∥J ′
P ⋆(w b
P −wP ⋆)∥2

≲∥f −J ′
P ⋆(w −wP ⋆)∥2 + ∥bζ −ζ⋆∥∞,

22


---Page Break---
for the same reason as εA as the error terms are the same. Thus, εB ≲∥bζ −ζ⋆∥∞+ εWComp.

In sum, we have shown that the misspecification is at most O(∥bζ −ζ⋆∥∞+ εWReal + εWComp).
Therefore, [69, Theorem 6.1 and Appendix C] ensures that w.p. 1 −δ, our learned bw satisfies,
J ′
b
P ( bw −w b
P )

2 ≲εW
n + ∥bζ −ζ⋆∥∞+ εWReal + εWComp +
p

log(1/δ)/n.

Concluding the proof.
The final step is to translate the above guarantee to ∥J ′
P ⋆( bw −wP ⋆)∥2.
The following shows that the switching cost is O(∥bζ −ζ⋆∥∞) as before:

∥J ′
P ⋆( bw −wP ⋆)∥2
≤∥(JP ⋆−J b
P )′( bw −wP ⋆)∥2 + ∥J ′
b
P ( bw −w b
P )∥2 + ∥J ′
b
P (w b
P −wP ⋆)∥2

≲εW
n + ∥bζ −ζ⋆∥∞+ εWReal + εWComp +
p

log(1/δ)/n.

This concludes the proof.

Lemma G.3 (Visitation performance-difference). Let P, U : S →R+ be non-negative measures,
which should be thought of as transitions in a discounted Markov chain. Assume U satisfies
P

s′ U(s′ | s) ≤1. Define dU = (1 −γ) P∞
h=1 γh−1dh
U, where dh
U =
R

s1,s2,...,sh−1 d1(s1)U(s2 |
s1) . . . U(s | sh−1)ds1:h−1. Assume the same for P.

Let F ⊂S →R be a function class that satisfies f ∈F
=⇒
g(s) = Es′∼P (s)[f(s′)] ∈F,
i.e., closed under projection with P. Then, define the integral (probability) metric ∥P −U∥F :=
supf∈F|(EP −EU)[f(s)]|. Then we have,

∥dP −dU∥F ≤
γ
1 −γ EdU ∥P(· | s) −U(· | s)∥F.
(8)

Proof. Recall Bellman’s flow, which is dP (s) = (1 −γ)d1(s) + γEes∼dP P(s | es). Fix any f ∈F.
The initial state distributions cancel, so we have,

|(EdP −EdU )[f(s)]|

=
γEes∼dP Es∼P (·|es)[f(s)] −γEes∼dU Es∼U(·|es)[f(s)]


≤
γEes∼dP Es∼P (·|es)[f(s)] −γEes∼dU Es∼P (·|es)[f(s)]


+
γEes∼dU Es∼P (·|es)[f(s)] −γEes∼dU Es∼U(·|es)[f(s)]


≤γ
(Ees∼dP −Ees∼dU )[Es∼P (·|es)f(s)]
 + γEes∼dU
 
Es∼P (·|es) −Es∼U(·|es)

[f(s)]
.

Thus, taking supremum over F, we have

∥dP −dU∥F
≤γ sup
f∈F

(Ees∼dP −Ees∼dU )[Es∼P (es)f(s)]
 + γEes∼dU sup
f∈F

 
Es∼P (·|es) −Es∼U(·|es)

[f(s)]


= γ∥dP −dU∥F + γEes∼dU ∥P(· | es) −U(· | es)∥F.
(F closed under P-projection)

Rearranging terms finishes the proof.

If F is the class of functions with ∥f∥∞≤1, then this recovers the TV distance, which gives,

∥dP −dU∥TV ≤
γ
1 −γ EdU ∥P(· | s) −U(· | s)∥TV.
(9)

This generalizes Lemma E.3 of [1] to infinite horizon.

23


---Page Break---
H
Proofs and Additional Details for the Orthogonal Estimator

H.1
Intuition for Theorem 5.3

We provide some intuition for the results in Theorem 5.3. Consider the V −bound and let us decouple
the indicator I [v(s′) −β(s, a) ≤0] that appears implicitly in the (v−(s′) −β−(s, a))−notation
of Theorem 5.1. We augment the set of nuisances with ζ(s, a, s′) = v−(s′) −β−(s, a) such that
(v−(s′)−β−(s, a))−= (v−(s′)−β−(s, a))I [ζ(s, a, s′) ≤0]. We state the following lemma (which
we elaborate upon in Lemmas H.4 and H.5 in the Appendix):
Lemma H.1 (Double sharpness with correct ζ⋆). Let E[ψ(s, a, s′; q, w, β, ζ⋆)] be the expectation of
the (R)EIF with an arbitrary nuisance set η = (w, q, β), but where the indicator I[v−(s′) ≤β−(s, a)]
has been replaced with the correct indicator I[ζ⋆(s, a, s′) ≤0]. Then:

V −
d1 = E[ψ(s, a, s′; q, w⋆, β⋆, ζ⋆)] = E[ψ(s, a, s′; q⋆, w, β⋆, ζ⋆)]

This lemma implies that if β−= (β∗)−and ζ = ζ∗, then the estimator bV −
d1 has a property known
as “double-robustness" [43] or “double-sharpness" [24] in q and w, meaning the bias vanishes when
either q or w is consistent. Moreover, the convergence rate would be Op(rw
n,2rq
n,2). This condition
holds provided that β and ζ are correctly specified. However, estimation errors in β introduce
an additional Op
 
(rβ
n,∞)2
term, reflecting that β is first-order optimal for the CVaR component.
Additionally, discrepancies between ζ and ζ∗contribute an extra Op
 
(rq
n,∞)2
to the error. While
this discussion gives some insight into how we achieve the results in Theorem 5.3, we provide a a
rigorous analysis in the next section.

H.2
Preliminaries

For this proof, our focus will be on bV −
d1. The argument for bV +
d1 is analogous, following a symmetric
approach. To improve the clarity of our exposition, we will omit the −and τ indices, assuming their
presence is clear from the context.

For simplicity, we assume that n is a multiple of K such that n = KnK, where nK is the size of a
fold. We let En, Ek denote the empirical averages over the entire sample and the kth fold, respectively.
Recall that we use bη = ( bw, bq, bβ) and η∗= (w∗, q∗, β∗) to denote the estimated and oracle nuisances,
respectively.

We further suppress the dependency on s, a in Λ and τ and we write the ρ term in Theorem 5.1 as

ρ(s, a, s′; v, β) = (1 −λ)v(s′) + λ
 
β(s, a) + τ −1(v(s′) −β(s, a))−

.
(10)

We justify this by noting that the analysis holds regardless of whether λ and τ depend on s, a.
Sometimes, it will be useful to decouple the indicator I [v(s′) −β(s, a) ≤0] implicit in the definition
of ρ. In this case, we augment the set of nuisances with ζ(s, a, s′) = v(s′) −β(s, a) and write ρ as

ρ(s, a, s′; v, β, ζ) = (1 −λ)v(s′) + λ
 
β(s, a) + τ −1(v(s′) −β(s, a))I [ζ(s, a, s′) ≤0]

.
(11)

Similarly define ψ(·; w, q, β, ζ) with the ρ(·; v, β, ζ).

H.3
Auxiliary Lemmas

Definition H.2 (Margin Condition). A function f : X →R of some random variable X is said to
satisfy the margin condition with sharpness α ∈[0, ∞] (or more succinctly, an α-margin) if there
exist a fixed constant c > 0 such that

∀t > 0 : P(0 < |f(X)| ≤t) ≤ctα.

If f(X) is either zero or bounded away from zero almost surely, then f satisfies an infinite margin,
i.e., α = ∞[37, Lemma 2]. If f(X) is continuously distributed in a neighborhood around 0, i.e.,

24


---Page Break---
its CDF is boundedly differentiable on (−ε, 0) ∪(0, ε) for some ε > 0, then f has a 1-margin [37,
Lemma 3].
Lemma H.3 (Margin Guarantees). For any f : X →R satisfying α-margin (Definition H.2),
p ∈[1, ∞], and any g : X →R, the following statements hold for some constant C > 0:

E[(I[g(X) ≤0] −I[f(X) ≤0])f(X)] ≤C∥f −g∥

p(1+α)

p+α
p
,
(12)

P[I[g(X) ≤0] ̸= I[f(X) ≤0], f(X) ̸= 0] ≤C∥f −g∥

pα
p+α
p
,
(13)

where ∥·∥p is the Lp norm and we set ∞t/∞= t in the exponents.

The proof of Eq. (12) for any p ∈[1, ∞] and of Eq. (13) for p = ∞is given in [4, Lemmas 5.1 and
5.2]. The proof of Eq. (13) for p < ∞is given in [37, Lemma 5].
Lemma H.4 (Sharpness with correct q⋆and β⋆).
1
n
P

(s,a,s′)∼D ψ(s, a, s′; w, q, β) is an unbiased
estimator of V ⋆
d1 when q = q⋆, β = β⋆, i.e.,

(1 −γ)Ed1v⋆(s1) = E[ψ(s, a, s′; w, q⋆, β⋆)].

Proof. Since q⋆and β⋆are correct, the robust Bellman equation holds, and so for every s, a,

E

(1 −λ)v⋆(s′) + λ(β⋆(s, a) + τ −1(v⋆(s′) −β⋆(s, a))−) | s, a

= 0.

Thus, multiplying by any w does not change the fact that the debiasing term in ψ has expectation
zero. Since we have v⋆, the first term in ψ is exactly the estimand, which concludes the proof.

Lemma H.5 (Sharpness with correct w∗and ζ∗).
1
n
P

(s,a,s′)∼D ψ(s, a, s′; w, q, β, ζ) is an unbiased
estimator of V ⋆
d1 when w = w⋆, ζ = ζ⋆, i.e.,

(1 −γ)Ed1v⋆(s1) = E[ψ(s, a, s′; q, w⋆, β, ζ⋆)]

Proof. Let P ⋆denote the robust transition kernel and let d⋆denote the robust visitation measure
under π, which satisfies: for all functions f,

Ed⋆[f(s, a)] = (1 −γ)Ed1f(s, π) + γEes,ea∼d⋆,s∼P ⋆(s,a)[f(s, π)].

Since ζ⋆is correct, for any v, s, a, we have

Es′∼P (s,a)

(1 −λ)v(s′) + λ
 
β(s, a) + τ −1(v(s′) −β(s, a))I [ζ⋆(s, a, s′) ≤0]


= Es′∼P (s,a)

(1 −λ)v(s′) + λτ −1v(s′)I [ζ⋆(s, a, s′) ≤0]

(⋆)

= Es′∼P ⋆(s,a)[v(s′)],
(Lemma 4.1)

where in ⋆we used Es′∼P (s,a)

β(s, a)
 
1 −τ −1I [ζ⋆(s, a, s′) ≤0]

= β(s, a)
 
1 −τ −1τ

= 0.
That is, for all function f, we have

(1 −γ)Ed1v(s1) + E[w⋆(s, a)(r(s, a) + γρ(s, a, s′; v, β, ζ⋆) −q(s, a))]

= (1 −γ)Ed1v(s1) + Es,a∼d⋆[r(s, a) + γρ(s, a, s′; v, β, ζ⋆) −q(s, a)]

= Es,a∼d⋆[r(s, a)] + (1 −γ)Ed1v(s1) + Es,a∼d⋆
γEs′∼P ⋆(s,a)[v(s′)] −q(s, a)


= Es,a∼d⋆[r(s, a)]
(robust Bellman flow)
= (1 −γ)Ed1v⋆(s1).

This concludes the proof.

H.4
Proof of Rates

The estimation error is given by:

|bVd1 −V ∗
d1| =


1
K

K
X

k=1
Ek[ψ(s, a, s′; bη[k])] −V ∗
d1

 ≤1

K

K
X

k=1

Ek[ψ(s, a, s′; bη[k])] −V ∗
d1


25


---Page Break---
We wish need to bound
Ek[ψ(s, a, s′; bη[k])] −V ∗
d1
. We have that:
Ek[ψ(s, a, s′; bη[k])] −V ∗
d1
 ≤
Ek[ψ(s, a, s′; bη[k])] −E[ψ(s, a, s′; bη[k])]
 +
E[ψ(s, a, s′; bη[k])] −V ∗
d1


The first term is Op(n−1/2) by the CLT. We are now interested in bounding the second term:

ε(bη) :=
E[ψ(s, a, s′; bη)] −V ∗
d1
.
(14)

where we dropped the [k] indicator without loss of generality. We further decompose ε(bη) into two
error terms, εA and εB, as follows:

ε(bη) =
E
h
ψ(s, a, s′; bq, bw, bβ)
i
−E
h
ψ(s, a, s′; bq, w⋆, bβ, ζ⋆)
i
(Lemma H.5)

≤
E
h
ψ(s, a, s′; bq, bw, bβ)
i
−E
h
ψ(s, a, s′; bq, bw, bβ, ζ⋆)
i
(εA)

+
E
h
ψ(s, a, s′; bq, bw, bβ, ζ⋆)
i
−E
h
ψ(s, a, s′; bq, w⋆, bβ, ζ⋆)
i.
(εB)

Bounding εA: Error from the incorrect indicator ζ.

εA = γλτ −1E bw(s, a)

bv(s′) −bβ(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]


≤Cγλτ −1E

bv(s′) −bβ(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]


(Assumption 4.2)

≲E

bv(s′) −bβ(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]


We break these terms down as follows:

E

bv(s′) −bβ(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]


=E(v⋆(s′) −β⋆(s, a))

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]

(εA
1 )

+ E

bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]

.

(εA
2 )

We first bound εA
1 . Assumption 4.2 implies

P(0 < |v⋆(s′) −β⋆(s, a)| ≤t) ≤c′′t, ∀t ∈[0, c′),
P(|v⋆(s′) −β⋆(s, a)| = 0) = 0,

where c′ < 1 is the min of 1 and the given neighborhood of zero and c′′ ≥1 is the max of 1 and
the bound on the density in that neighborhood. This implies a margin condition with α = 1 and
c = c′′/c′.

We can instantiate the first part of Lemma H.3 with f(X) = v⋆(s′)−β⋆(s, a), g(X) = bv(s′)−bβ(s, a)
and obtain

εA
1 ≲
v⋆(s′) −β⋆(s, a) −bv(s′) + bβ(s, a)


2p
p+1

p

≤∥bv(s′) −v⋆(s′)∥

2p
p+1
p
+
bβ(s, a) −β⋆(s, a)


2p
p+1

p
.

To bound εA
2 , first write
E

bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)

I
h
bv(s′) −bβ(s, a) ≤0
i
−I [v⋆(s′) −β⋆(s, a) ≤0]


≤
bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)

p

· P

I
h
bv(s′) −bβ(s, a) ≤0
i
̸= I [v⋆(s′) −β⋆(s, a) ≤0]
(p−1)/p
.
(Holder’s inequality)

26


---Page Break---
We can bound P

I
h
bv(s′) −bβ(s, a) ≤0
i
̸= I [v⋆(s′) −β⋆(s, a) ≤0]

using the second part of
Lemma H.3 such that

εA
2 ≲
bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)

p

bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)


p−1
p+1

=
bv(s′) −bβ(s, a) −v⋆(s′) + β⋆(s, a)


2p
p+1

p

≤∥bv(s′) −v⋆(s′)∥

2p
p+1
p
+
bβ(s, a) −β⋆(s, a)


2p
p+1

p
.

Putting the εA
1 and εA
2 together, we have

εA ≲∥bv(s′) −v⋆(s′)∥

2p
p+1
p
+
bβ(s, a) −β⋆(s, a)


2p
p+1

p
(when p ∈[1, ∞))

≲∥bv(s′) −v⋆(s′)∥2
∞+
bβ(s, a) −β⋆(s, a)

2

∞.
(when p = ∞)

Bounding εB: Error with correct indicator but wrong nuisances.
Now we focus on bounding
εB.

εB = E
h
ψ(s, a, s′; bq, bw, bβ, ζ⋆)
i
−E
h
ψ(s, a, s′; bq, w⋆, bβ, ζ⋆)
i

= E( bw(s, a) −w⋆(s, a))

r(s, a) + γρ(s, a, s′; bv, bβ, ζ⋆) −bq(s, a)


= E( bw(s, a) −w⋆(s, a))

r(s, a) + γρ(s, a, s′; bv, bβ, ζ⋆) −bq(s, a)


−E( bw(s, a) −w⋆(s, a))(r(s, a) + γρ(s, a, s′; v⋆, β⋆) −q⋆(s, a))
(Lemma H.4)

= E( bw(s, a) −w⋆(s, a))

bq(s, a) −q⋆(s, a) + γ(ρ(s, a, s′; bv, bβ, ζ⋆) −ρ(s, a, s′; v⋆, β⋆))

.

In the Lemma H.4 step, we used

0 = (1 −γ)Ed1v⋆(s1) −E[ψ(s, a, s′; q⋆, bw, β⋆)] = (1 −γ)Ed1v⋆(s1) −E[ψ(s, a, s′; q⋆, w⋆, β⋆)].

Finally, note that

ρ(s, a, s′; bv, bβ, ζ⋆) −ρ(s, a, s′; v⋆, β⋆)

= (1 −λ)(bv(s′) −v⋆(s′)) + λτ −1(bv(s′) −v⋆(s′))I [ζ⋆(s, a, s′) ≤0]

+ λ(bβ(s, a) −β⋆(s, a))
 
1 −τ −1I [ζ⋆(s, a, s′) ≤0]

.

Due to continuity of the CDF of v⋆(s′) at β⋆(s, a) for all s, a, we have Pr(ζ⋆(s′, s, a) ≤0 | s, a) = τ
and so the last term vanishes. Thus, we’re left with a quantity that is at most ≲(bv(s′) −v⋆(s′)).
Therefore,

εB ≲E( bw(s, a) −w⋆(s, a))(JU ±(bq(s, a) −q⋆(s, a)))

≤∥J ′
U ±( bw −w⋆)∥2∥bq −q⋆∥2.
(Holder’s inequality)

Putting everything together, we obtain the desired rates:

|bVd1 −V ∗
d1| ≲Op(n−1/2) + ∥J ′
U ±( bw −w⋆)∥2∥bq −q⋆∥2 + ∥bv −v⋆∥

2p
p+1
p
+
bβ −β⋆

2p
p+1

p

= Op(n−1/2) + Op

rw
n rq
n + (rq
n,p)

2p
p+1 + (rβ
n,p)

2p
p+1

(when p ∈[1, ∞))

≲Op(n−1/2) + ∥J ′
U ±( bw −w⋆)∥2∥bq −q⋆∥2 + ∥bv −v⋆∥2
∞+
bβ −β⋆
2

∞
= Op(n−1/2) + Op
 
rw
n rq
n + (rq
n,∞)2 + (rβ
n,∞)2
.
(when p = ∞)

27


---Page Break---
H.5
Proof of Normality & Efficiency

In this part of the theorem, we let:

eVd1 = 1

K

K
X

k=1
Ek[ψ(s, a, s′; η∗)]

Then, we can write the following equality:
√n(bVd1 −V ∗
d1) = √n(bVd1 −eVd1) + √n(eVd1 −V ∗
d1)
|
{z
}

d−→N(0,Σ)

The second term converges in distribution to N(0, Σ) from the CLT and the fact that ψ is the efficient
influence function. Thus, it remains to show that the first term is op(1). We decompose the first term
as follows:

√n(bVd1 −eVd1) = √n 1

K

n
X

k=1


E[ψ(s, a, s′; bη[k])] −E[ψ(s, a, s′; η∗)]

(15)

+ √n 1

K

n
X

k=1
(Ek −E)[ψ(s, a, s′; bη[k]) −ψ(s, a, s′; η∗)]
|
{z
}
εk

(16)

In Eq. (15), we have that |E[ψ(s, a, s′; bη[k])] −E[ψ(s, a, s′; η∗)]| is bounded as in Eq. (Rates). Given
the theorem’s assumption about the nuisance rates, this term is op(n−1/2) and Eq. (15) is op(1). We
now seek to control the εk term in Eq. (16). Letting Dk represent the samples in the kth fold, we
leverage sample splitting to show that the mean of εk | Dk is 0:

E[εk | Dk] = E[Ek[ψ(s, a, s′; bη[k]) −ψ(s, a, s′; η∗)] −E[ψ(s, a, s′; bη[k]) −ψ(s, a, s′; η∗)] | Dk]
= 0

where we consider bη[k] fixed with respect to the second expectation. The result follows from the fact
that bη[k] does not depend on Dk. Then, we can invoke Chebyshev’s inequality to obtain the following
bound:

P

εk
Var[εk | Dk]1/2 ≥ϵ
Dk


≤1

ϵ2 , ∀ϵ > 0

Thus, we have shown that εk | Dk = Op(Var[εk | Dk]1/2) = Op(n−1/2E[(ψ(s, a, s′; bη[k]) −
ψ(s, a, s′; η∗))2 | Dk]1/2). Here, we used the fact that nK = n/K (the size of Dk) and that K is a
fixed integer that doesn’t grow with n. Moreover, εk has 0 conditional mean.

For the remainder of the analysis, we leave the conditioning on Dk implicit for simplicity. To bound
E[(ψ(s, a, s′; bη[k])−ψ(s, a, s′; η∗))2 | Dk]1/2 = ∥ψ(s, a, s′; bη[k])−ψ(s, a, s′; η∗)∥2, we use similar
notation and techniques as in Appendix H.4:

∥ψ(s, a, s′; bη[k]) −ψ(s, a, s′; η∗)∥2 ≤∥ψ(s, a, s′; bq, bw, bβ) −ψ(s, a, s′; bq, bw, bβ, ζ∗)∥2
(σ1)

+ ∥ψ(s, a, s′; bq, bw, bβ, ζ∗) −ψ(s, a, s′; q∗, w∗, β∗, ζ∗)∥2
(σ2)

where we invoked Cauchy-Schwarz for the L2 norm. We bound σ2 as follows:

σ2 ≤∥ψ(s, a, s′; bq, bw, bβ) −ψ(s, a, s′; q∗, bw, bβ, ζ∗)∥2
(σ2a)

+ ∥ψ(s, a, s′; q∗, bw, bβ, ζ∗) −ψ(s, a, s′; q∗, bw, β∗, ζ∗)∥2
(σ2b)

+ ∥ψ(s, a, s′; q∗, bw, β∗, ζ∗) −ψ(s, a, s′; q∗, w∗, β∗, ζ∗)∥2
(σ2c)

≤∥bv −v∗∥2 + γ(1 −λ)∥bw∥2∥bv −v∗∥2 + γλτ −1∥bw∥2∥bv −v∗∥2 + ∥bw∥2∥bq −q∗∥2
(σ2a)

+ γλ∥bw∥2∥bβ −β∗∥2 + γλτ −1∥bw∥2∥bβ −β∗∥2
(σ2b)

+ ∥bw −w∗∥2
 
∥r∥2 + γ(1 −λ)∥v∗∥2 + γλ∥β∗∥2 + γλτ −1∥v∗−β∗∥2

(σ2c)

28


---Page Break---
Given our rate assumptions, our boundedness assumptions for bw, the implicit boundedness of
q∗, v∗, w∗, β∗, as well as the ordering of the L2 and L∞norms, σ2 is op(1). We now bound the σ1
term:

σ2 = γλτ −1  bw(s, a)(bv(s′) −bβ(s, a))(I[bv(s′) ≤bβ(s, a)] −I[v∗(s′) ≤β∗(s, a)])

2

There are two cases in which the difference of indicators is non-zero:
(
bv(s′) ≤bβ(s, a) and v∗(s′) > β∗(s, a) ⇒I[bv(s′) ≤bβ(s, a)] −I[v∗(s′) ≤β∗(s, a)] = 1
bv(s′) > bβ(s, a) and v∗(s′) ≤β∗(s, a) ⇒I[bv(s′) ≤bβ(s, a)] −I[v∗(s′) ≤β∗(s, a)] = −1

In the first case, bv(s′) −bβ(s, a) ≤0, β∗(s, a) −v∗(s′) < 0 and thus

|(bv(s′)−bβ(s, a))(I[bv(s′) ≤bβ(s, a)]−I[v∗(s′) ≤β∗(s, a)])| ≤|bv(s′)−bβ(s, a)+β∗(s, a)−v∗(s′)|.

In the second case, bv(s′) −bβ(s, a) > 0, β∗(s, a) −v∗(s′) ≤0 and

|(bv(s′)−bβ(s, a))(I[bv(s′) ≤bβ(s, a)]−I[v∗(s′) ≤β∗(s, a)])| ≤|bv(s′)−bβ(s, a)+β∗(s, a)−v∗(s′)|.

Going back to σ1, we have:

σ2 ≤γλτ −1∥bw∥2∥bv(s′) −bβ(s, a) + β∗(s, a) −v∗(s′))∥2

≤γλτ −1∥bw∥2(∥bv −v∗∥2 + ∥bβ −β∗∥2)

By out theorem’s assumptions, this term is also op(1). Putting σ1 and σ2 together, we have that
∥ψ(s, a, s′; bη[k]) −ψ(s, a, s′; η∗)∥2 is op(1) and εk | Dk is op(n−1/2). By the bounded convergence
theorem, this implies that εk is also op(n−1/2). Then, the term in 16 is op(1), which further means
that √n(bVd1 −eVd1) = op(1). Our proof is now complete.

I
Derivation of the Efficient Influence Function

We use the ε-contamination approach of [31] to derive an influence function (IF) for our estimand
V −
d1. The proof for V +
d1 follows symmetrically. We note that since our tangent space is the whole
space as it factorizes in the trivial way (as in [39, Page 54]), the IF we derive is actually the efficient
influence function (EIF).

Let P(s, a, s′) denote the data distribution. Consider the ε-contamination Pε(s, a, s′) = (1 −
ε)P(s, a, s′) + εδ(¯s, ¯a, ¯s′), where δ(¯z) is the dirac delta at ¯z, i.e., δ(¯z) has infinite mass at ¯z and 0
mass elsewhere. Let V −
ε denote the robust value function under the transition kernel Pε(s′ | s, a).
Omitting the ε subscript means ε = 0. The IF of V −
d1 is then given by

d
dε(1 −γ)Ed1V −
ε (s1)|ε=0.

We dedicate the rest of this section towards this goal, which will be obtained in Theorem I.5.

Lemma I.1.

d
dεPε(s′ | s, a)|ε=0 = δ(¯s, ¯a)

P(s, a)(δ(¯s′) −P(s′ | s, a)).

Proof. Use the fact Pε(s′ | s, a) = Pε(s,a,s′)

Pε(s,a)
= (1−ε)P (s,a,s′)+εδ(¯s,¯a,¯s′)

(1−ε)P (s,a)+εδ(¯s,¯a)
and take derivative.

Lemma I.2 (IF of conditional expectation). For any s, a and fε,

d
dεEPε[fε(s′) | s, a]|ε=0 = δ(¯s, ¯a)

P(s, a)(f(¯s′) −EP [f(s′) | s, a]) + EP

 d

dεfε(s′)|ε=0 | s, a

,

where f = f0.

29


---Page Break---
Proof.

d
dεEPε[fε(s′) | s, a]|ε=0 =
X

s′
f(s′) d

dεPε(s′ | s, a)|ε=0 +
X

s′

d
dεfε(s′)|ε=0P(s′ | s, a)

= δ(¯s, ¯a)

P(s, a)(f0(¯s′) −EP [f0(s′) | s, a]) + EP

 d

dεfε(s′)|ε=0 | s, a

,

Lemma I.3 (IF of conditional CVaR). For any τ, s, a and fε,

d
dε CVaRτ,Pε[fε(s′) | s, a]|ε=0 = δ(¯s, ¯a)

P(s, a)
 
βτ(s, a) + τ −1(f(¯s′) −βτ(s, a))−−CVaRτ(f(s′) | s, a)


+ EP


τ −1I [f(s′) ≤βτ(s, a)] d

dεfε(s′)|ε=0 | s, a

,

where f = f0 and βτ(s, a) be the (1 −τ)-th quantile of f(s′), s′ ∼P(s, a).

Proof.

d
dε CVaRPε[fε(s′) | s, a]|ε=0
(17)

= d

dε min
b
EPε

b + τ −1(fε(s′) −b)−| s, a

|ε=0
(18)

= d

dεEPε

βτ(s, a) + τ −1(fε(s′) −βτ(s, a))−| s, a

|ε=0,
(19)

where the last equality is due to Danskin’s theorem and the fact that βτ(s, a) is the maximizer of the
CVaR dual form at ε = 0. Continuing, let gε(s′; s, a) := βτ(s, a) + τ −1(fε(s′) −βτ(s, a))−, so

d
dεEPε[gε(s′; s, a) | s, a]

= δ(¯s, ¯a)

P(s, a)(g(¯s′; s, a) −EP [g(s′, s, a) | s, a]) + EP

 d

dεgε(s′; s, a)|ε=0 | s, a

(Lemma I.2)

= δ(¯s, ¯a)

P(s, a)(g(¯s′; s, a) −CVaRτ(f(s′) | s, a)) + EP


τ −1I [f(s′) ≤βτ(s, a)] d

dεfε(s′)|ε=0 | s, a

.

This concludes the proof.

We now prove the key “one-step forward” lemma.

Lemma I.4 (One-Step Forward). For any state distribution ν(s), we have

Es∼ν

 d

dεV −
ε (s)|ε=0



= ν(¯s)π(¯a | ¯s)

P(¯s, ¯a)
 
r(¯s, ¯a) + γ
 
(1 −λ)V −(¯s′) + λ
 
βτ(¯s, ¯a) + τ −1(V −(¯s′) −βτ(¯s, ¯a))−


−Q−(¯s, ¯a)


+ γEs∼ν


Eπ,P

 
(1 −λ) + λτ −1I

V −(s′) ≤βτ(s, a)
 d

dεV −
ε (s′)|ε=0 | s

.

30


---Page Break---
Proof. For any s1, we have
d
dεV −
ε (s1)

= d

dεEa1∼π(s1)

r(s1, a1) + γ((1 −λ)EPε

V −
ε (s2) | s1, a1

+ λ CVaRτ,Pε

V −
ε (s2) | s1, a1


ε=0

= γEa1∼π(s1)


(1 −λ) d

dεEτ,Pε

V −
ε (s2) | s1, a1

|ε=0 + d

dε CVaRτ,Pε

V −
ε (s2) | s1, a1

|ε=0



= γ(1 −λ)Ea1∼π(s1)

 δ(¯s, ¯a)

P(s1, a1)
 
V −(¯s′) −EP

V −(s2) | s1, a1


+ γ(1 −λ)Ea1∼π(s1)EP

 d

dεV −
ε (s2)|ε=0 | s1, a1



+ γλEa1∼π(s1)

 δ(¯s, ¯a)

P(s1, a1)
 
βτ(s1, a1) + τ −1(V −(¯s′) −βτ(s1, a1))−−CVaRτ(V −(s2) | s1, a1)


+ γλEa1∼π(s1)EP


τ −1I

V −(s2) ≤βτ(s1, a1)
 d

dεV −
π,Pε(s2)

.

Taking expectation over s1 ∼ν, we have

Es∼ν

 d

dεV −
ε (s)|ε=0


= γ ν(¯s)π(¯a | ¯s)

P(¯s, ¯a)


(1 −λ)V −(¯s′) + λ
 
βτ(¯s, ¯a) + τ −1(V −(¯s′) −βτ(¯s, ¯a))−


−
 
(1 −λ)E

V −(s′) | ¯s, ¯a

+ λ CVaRτ(V −(s′) | ¯s, ¯a)


+ γEs∼ν


Eπ,P

 
(1 −λ) + λτ −1I

V −(s′) ≤βτ(s, a)
 d

dεV −
ε (s′)|ε=0 | s

.

Finally recall that V −satisfies the Bellman equation, so
(1 −λ)E

V −(s′) | ¯s, ¯a

+ λ CVaRτ(V −(s′) | ¯s, ¯a) = Q−(¯s, ¯a) −r(¯s, ¯a).
This concludes the proof.

Equipped with our main one-step lemma, we can now unroll it an infinite number of steps to derive
the IF of our estimand.
Theorem I.5 (IF of Estimand). Let us denote
g(¯s, ¯a, ¯s′) := r(¯s, ¯a) + γ
 
(1 −λ)V −(¯s′) + λ
 
βτ(¯s, ¯a) + τ −1(V −(¯s′) −βτ(¯s, ¯a))−

.
Then, we have

Ed1

 d

dεV −
ε (s1)|ε=0


= dπ,∞
rob (¯s, ¯a)

P(¯s, ¯a) g(¯s, ¯a, ¯s′).

Proof. Let dh denote the h-th step visitation in the robust MDP, with transition Prob satisfying
Prob(s′|s,a)

P (s′|s,a) = (1 −λ) + λτ −1I [V −(s′) ≤βτ(s, a)]. Then notice that the final term of Lemma I.4 is
exactly Es∼ν

Eπ,Prob
 d

dεV −
ε (s′)|ε=0 | s

. Therefore,

Ed1

 d

dεV −
ε (s1)|ε=0



= d1(¯s)π(¯a | ¯s)

P(¯s, ¯a)
g(¯s, ¯a, ¯s′) + γEs2∼d2

 d

dεV −
ε (s2)|ε=0



= d1(¯s)π(¯a | ¯s)

P(¯s, ¯a)
g(¯s, ¯a, ¯s′) + γ d2(¯s)π(¯a | ¯s)

P(¯s, ¯a)
g(¯s, ¯a, ¯s′) + γ2Es3∼d3

 d

dεV −
ε (s3)|ε=0


.

Iterating the process, we have

Ed1

 d

dεV −
ε (s1)|ε=0


=

∞
X

h=1
γh−1 dh(¯s)π(¯a | ¯s)

P(¯s, ¯a)
g(¯s, ¯a, ¯s′) = dπ,∞
rob (¯s, ¯a)

P(¯s, ¯a) g(¯s, ¯a, ¯s′),

as desired.

31


---Page Break---
Finally, we can conclude that the IF in Theorem I.5 is in fact the efficient IF (EIF) because it is in the
tangent space, as the tangent space is contains all functions [39].

J
Additional Validity Guarantees for Orthogonal Estimator

Our orthogonal estimator has additional desirable properties such as validity when some nuisances
are misspecified. Specifically, the bounds returned by our orthogonal estimator will be asymptotically
valid, though possibly loose, when some nuisances are inconsistent, i.e., do not converge to the their
true values. Below, we detail conditions under which we achieve validity. To be concise, we focus on
the −case as the + case is symmetric.

Validity with correct Q±.
If bQ = Q±, we obtain valid bounds even if w, β are inconsistent.
Lemma J.1. For any w, β, we have E[ψ(s, a, s′; Q−, β, w)] ≤V −
d1 with equality when β = β−
τ .

Validity with Q = T ±
β Q.
Even if bQ is misspecified, we still have a valid bound if it solves a
Bellman-type equation of the dual CVaR form. For a β : S × A →R, define:

T ±
β f(s, a) := r(s, a) + γΛ−1(s, a)E[f(s′, πt) | s, a]

+ γ(1 −Λ−1(s, a))E

β(s, a) + τ −1(s, a)(f(s′, πt) −β(s, a))± | s, a

.

Lemma J.2. Fix any w, β. If Q±
β = T ±
β Q±
β , then E[ψ(s, a, s′; Q−
β , β, w)] ≤V −
d1.

Remark J.3. Lemmas J.1 and J.2 are dual to each other: in Lemma J.1, the plug-in is consistent while
the debiasing correction errs in the valid direction (i.e., ≥0 for + and ≤0 for −). In Lemma J.2, the
plug-in is valid while the debiasing correction has expectation zero.

J.1
Proofs for validity

Lemma J.1. For any w, β, we have E[ψ(s, a, s′; Q−, β, w)] ≤V −
d1 with equality when β = β−
τ .

Proof.

E[ψ(s, a, s′; Q−, β, w)] ≤(1 −γ)Ed1[V −
β (s1)] + E[w(s, a)
 
Q−(s, a) −T −
CVaRQ−(s, a)

]

= V −
d1 + 0 = V −
d1,

where the inequality comes from the fact that β is sub-optimal for E[β(s, a) + τ −1(V −(s′) −
β(s, a))−]. The same proof applies for Q+.

We now prove Lemma J.2. First, we show that the Tβ perspective gives rise to a dual definition of
Q± (dual to Eq. (2)).
Lemma J.4.

Q+(s, a) = arg minβ:Qβ=T +
β Qβ Qβ(s, a),
Q−(s, a) = arg maxβ:Qβ=T −
β Qβ Qβ(s, a).

Proof. Unroll Q−(s, a) = r(s, a) + γ infU∈U(P ) EU[r(s′, a′) + γ infU∈U(P ) EU[. . . ]], replacing
each infU∈U(P ) with the convex combination of E and CVaR from Lemma 3.1. Then, write
each CVaR using the dual form, i.e., maxβ{β(s, a) + τ −1(s, a)E[(· · · −β(s, a))+]}. By s, a-
rectangularity, the scalar maxβ separates per s, a, so we can pull all the maxes out front as a max
over β(s, a) functions. Note that not all β(s, a) functions have a well-defined infinite sum in this
manner, as Tβ is not always a contraction. The condition Qβ = T −
β Qβ exactly characterizes when
this unrolling is well-defined. Thus, Q−is exactly the minimum Qβ whenever this procedure of
unrolling with β is well-defined. This concludes the proof.

32


---Page Break---
Lemma J.2. Fix any w, β. If Q±
β = T ±
β Q±
β , then E[ψ(s, a, s′; Q−
β , β, w)] ≤V −
d1.

Proof.

E[ψ(s, a, s′; Q−
β , β, w)] = (1 −γ)Ed1[V −
β (s1)] + 0 ≤V −
d1.

The first equality is because the correction term is T −
β Q−
β −Q−
β , which is zero since Q−
β is a fixed
point. The inequality is due to Lemma J.4.

K
Additional Details for Main Experiment

K.1
Environment

We consider a simple MDP with a one-dimensional state space S = [0, 5], a binary action space
A = {0, 1}, reward function

r(s, a) = 26 −s2 −I [a = 1]

26
,

which we note takes values in the range [0, 1], and with transitions given by

P(· | s, a = 0) = UnifClip[s −0.2, s + 1]
P(· | s, a = 1) = UnifClip[0.2s −0.02, s + 0.5] ,

where UnifClip[a, b] denotes a uniform distribution between max(a, 0) and min(b, 5). In addition,
the environment always starts in initial state s0 = 2. Essentially, this is a simple control environment,
where high rewards are obtained by maintaining state as close to zero as possible, the action a = 1 is
a control action that (in expectation) moves the state closer to zero, and which occurs a small reward
cost, and the action a = 0 is a passive action that allows the state to freely drift (with an overall drift
away from zero).

K.2
Target Policy

We focus on estimating the worst-case policy value V −
d1 for the simple threshold-based target policy
πt which takes action a = 1 when s ≥2, and a = 0 whenever s < 2.

K.3
Logging Policy and Data Sampling Procedure

We sample data using an evaluation policy πb which is an ϵ-smoothed threshold policy similar to πt.
Specifically, πb takes action a = 1 when s ≥1.5 with probability 0.95, and takes action a = 0 when
s < 1.5 with probability 0.95. We obtain a dataset {si, ai, s′
i, ri} by first rolling out with πb for 1000
burn-in time steps, and then sampling the tuple (s, a, s′, r) every 10 time steps. For each replication
of our experiment, we sample 10,000 tuples in total.

K.4
Calculation of True Worst-Case Policy Values

A major challenge in studying robust policy value estimation is that, even with ground truth knowledge
of the MDP and/or access to a simulator, it may be intractable to estimate the robust policy values V ±
d1.
Fortunately, the above environment has the desirable property that we can analytically compute the
best/worst-case transition distributions allowed by our sensitivity model, since no matter what policy
πt the agent is acting with, it always strictly prefers transitions to smaller states. In detail, suppose
that for some state, action pair (s, a) we have P(· | s, a) = Unif[x, y], for some 0 ≤x ≤y ≤5].
Then, letting α = 1/(1 + Λ(s, a)), it is easy to verify that the worst case transition kernel is given by

U −(· | s, a) = (1 −Λ−1(s, a))Unif[y −α(y −x), y] + Λ−1(s, a)Unif[x, y] .

33


---Page Break---
That is, the worst case transition kernel is given by a mixture of two uniform distributions. Therefore,
we can easily simulate rollouts with the best/worst case transition kernels, and accurately estimate
the robust policy values. This allows us to validate our methodology in this synthetic environment.
Specifically, for each Λ(s, a) we experiment with, we can compute the corresponding ground truth
V −
d1 up to arbitrary precision via Monte Carlo sampling, by rolling out trajectories with πt in the
adversarial MDP according to the above worst-case transition kernel.

Note as well that if one wanted to estimate the best-case policy value, analogous reasoning would
give us

U +(· | s, a) = (1 −Λ−1(s, a))Unif[x, x + α(y −x)] + Λ−1(s, a)Unif[x, y] .

However, in our experiments we only concern ourselves with worst-case policy value estimation.

K.5
Nuisance Estimation

We instantiate slight variations of Algorithms 1 and 2 using neural nets for the classes Q, B, and
W used for fitting Q−, β−, and w−respectively, and linear sieves for the corresponding critic
class Q that we perform maximization over for the minimax estimation of w−. Specifically, we
grow the linear sieve for the critic class in a data-driven way, as follows: at each step k of the
respective algorithm, we compute the best response qk ∈Q to the previous iterate solution wk ∈W
by optimizing over a neural net class, and then we append this best-response function to the set of
functions in our linear sieve for the corresponding critic class. Full exact nuisance estimation details
necessary for reproducibility will be available in our code release.

K.6
Estimators

We estimate the worst-case policy value using three different estimators:

• Q: Direct estimator given by:

bV −
d1 = bQ−(s1, πt(s1)) ,

where s1 is the deterministic initial state.

• W: Importance sampling-style estimator using ˆw−, which is given by:

bV −
d1 = 1

n

n
X

i=1
bw−(si, ai)bξiri ,

where
bξi = Λ−1 + (1 −Λ−1)(1 + Λ)I
h
bV −(s′
i) ≤bβ−(si, ai)
i
.

• Orth: Our orthogonal estimator using EIF, given by

bV −
d1 = 1

n

n
X

i=1
ψ(si, ai, s′
i; bQ−, bβ−, bw−) .

Note as well that we used a simpler data splitting procedure rather than the cross-fitting procedure
described in Algorithm 3. Specifically, we used the first 10,000 tuples for estimating nuisances, and
the second 10,000 tuples for the final estimators. This was done for the sake of computational ease in
running experiments with many replications, and was performed in the same way for all methods.

In addition, for extra robustness, in each experiment replication we ran the nuisance estimation
pipeline 5 times (on the same fixed sampled dataset), and took the 80th percentile policy value
estimates, since the estimators tend to under-estimate the true policy value by design, with greater
under-estimation when the nuisance estimates are less well optimized.

34


---Page Break---
L
Empirical Investigation on Medical Application

Here, we describe an additional empirical investigation of our methodology on medical data. Specif-
ically, we consider the problem of sepsis management using RL. For all parts of the investigation
described below, fully complete details can be obtained from our code release.

L.1
Motivation of Investigation

Training RL models in simulated environments derived from real-world data is an exciting avenue for
leveraging AI towards critical medical use cases. However, doing this obviously has the downside
that, unless one undergoes the very risky process of training an RL agent online via real medical
interventions, one has to resort to training within simulators, and then has to account for the inevitable
“sim-to-real” gap. Therefore, our robust OPE methodology provides an interesting approach for
estimating worst-case performance of RL models under potential changes in dynamics when moving
to real application.

L.2
RL Environment

Our RL environment is based on the OpenAI Gym sepsis simulator environment of [44]. This RL
environment allows for simulation of dynamic sepsis management, which was created by training a
blackbox ML model to mimic observed transition dynamics from the real-world electronic health
record-based MIMIC-III dataset [36]. This existing sepsis simulator is an episodic environment
that continues until the agent either recovers or dies. It has a 46-dimensional state space containing
various vital measurements, a discrete action space containing 24 possible actions (where an action
is essentially the Cartesian product of some independent base actions). The reward function in
this original simulator gives zero reward whenever an episode has not terminated, a +15 reward at
termination when if the patient survives, or a -15 reward at termination if the patient dies. Please see
[44] and the code release linked therein for additional details.

We built an RL environment for our investigation by creating a simple wrapper around this existing
sepsis simulator, in order to make it fit our setup. In particular, we made the following key changes:

1. We made the environment infinite-horizon, by automatically looping to a new random
starting state for a fresh patient whenever the episode in the base simulator terminates

2. We normalized the reward function so that it lies in range [0, 1], where:

(a) r(s, a) = 0 if patient dies
(b) r(s, a) = 1 if patient recovers and is discharged

(c) r(s, a) = 0.5 if treatment has not terminated for current patient

In addition, for this environment, we perform all experiments with γ = 0.95.

L.3
Policies for Investigation

We constructed RL policies for our empirical investigation by training some deep RL models using
the sepsis simulator environment.

In the case of the behavioral policy πb used to generate the observational offline data, we trained
this policy by running Proximal Policy Optimization (PPO: [63]) over a relatively large (16,000)
number timesteps, in order to emulate a reasonably good “current best practices” model for creating
observational data.

In the case of the target policy πt to be evaluated, we trained this policy using Deep Q Learning
(DQL: [53]), over a relatively small (1,600) number of timesteps, in order to emulate a potentially
risky new candidate model.

35


---Page Break---
Λ
Median Policy Value Estimate
Q
W
Orth
1
.546 ± .003
.386 ± .087
.532 ± .008
2
.454 ± .040
.534 ± .141
.515 ± .036
4
.381 ± .077
.287 ± .106
.338 ± .086
Table 2: Median policy value estimate for sepsis management investigation, for each estimator and
value of Λ over 5 runs of each estimator from random initial seeds. The ± values are given by half
the difference between 80th and 20th percentiles.

L.4
Creating an Offline Dataset

Using our behavioral policy πb which we created as above, we generated a fixed offline dataset
consisting of 20,000 observed tuples of state, action, reward, and next state. Unlike with our main
empirical investigation in the main paper, we did not perform any “thinning” on these sampled tuples
to make them more independent, so that the observed transitions are sequentially correlated as with
real-world medical data.

L.5
Nuisance Estimation

We perform nuisance estimation almost identically as in our main empirical investigation, with the
only change being a slight change to our neural network architectures to better handle the large
discrete action space. Specifically, instead of training neural networks that take state as input and
produce |A| outputs (one per action), we train neural networks that take both state and action as
inputs, using a learnt low-dimensional encoding of the actions, and produce a single output. Please
see our code release for details.

L.6
Estimators

We consider the same three estimators (Q, W, and Orth) as in our main empirical investigation. As
in that investigation, we use these to estimate the worst-case policy value for the given Λ(s, a). In
addition, as in the main experiments, we consider these estimators for various fixed Λ(s, a) that do
not depend on s or a. In this case, we consider Λ ∈{1, 2, 4}, as these reflect a reasonable range of
possible confounding strength for real application.

L.7
Results

Below, in Table 2 we show the estimated policy value for all three estimators for each fixed Λ ∈
{1, 2, 4}. Here, we present the median policy value estimate over 5 runs of our estimators from
random starting seeds after removing outliers.2 In addition, we present a ± spread given by half the
difference between the 80th and 20th percentiles.

Although for this investigation we cannot analytically compute the ground truth “true” adversarial
policy values to evaluate against when Λ > 1, we can still analyze the trends of these estimators and
compare them to those observed in our main synthetic experiment, and we can also compare their
accuracy when Λ = 1.

First, in the case of Λ = 1, we computed the true policy value of πt to be within the range 0.532±0.002
with 95% confidence. This is almost exactly equal to the median Orth estimator, but far outside the
spread of outputs of the Q estimator. That is, although the Q estimator has somewhat lower variance
in outputs over multiple runs for Λ = 1 compared with Orth, it appears to be far more biased.

Next, looking more broadly across all values of Λ, as in our main experiment, the Q and Orth
estimators generally result in similar estimates to each other, and the W estimators are very variable.

2Specifically, we exclude policy value estimates that lie outside the possible range of [0, 1], which occasionally
occur due to bad optimization from the starting seed.

36


---Page Break---
This may reflect the relative difficulty of estimating the w−nuisance function compared with Q−and
β−; although both Orth and W are affected by this difficulty, the Orth estimator has a theoretical
robustness to the errors of these nuisance functions that the W estimator does not, as outlined in our
theory.

We also observe that when Λ = 1 the Q estimator is significantly more stable than Orth, but when
Λ > 1 the stability of Orth is either comparable to or superior to Q. In order to understand this, we
first note that unlike in our main experiments, here the repetitions are re-runs of the estimators with
the same offline sepsis dataset, so these ± spreads reflect potential computational errors rather than
statistical errors. Given this, this pattern of errors could be explained by the fact that when Λ = 1
the Q estimation is extremely simple, reducing to standard FQI, whereas when Λ > 1 it requires a
more complex robust FQI estimation with simultaneous estimation of β−. That is, the difference in
computational difficulty of estimating Orth versus Q may be smaller for Λ > 1.

Overall, although it is hard to definitively compare the accuracy of these estimators for Λ > 1
given a fundamental lack of ground truth, given both a similar pattern of results as in our synthetic
experiments, as well as the far greater accuracy of Orth when Λ = 1, it seems reasonably to believe
based on these results that our proposed Orth estimator may be more reliable than the existing robust
FQI approach of the Q estimator.

Finally, we consider the implication of our results for the problem of learning sepsis management
policies from simulators. Our Orth estimator suggests that there is relatively little sensitivity of
this environment to deviations allowed by Λ = 2, but very significant deviation allowed by Λ = 4.
Indeed, given the reward structure described above, the worst-case results under Λ = 4 imply an
extremely high mortality rate. Whether worst-case deviations of this magnitude are reasonable or not
is unclear, and this is something that requires further investigation for future work on RL for sepsis
management.

37


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes] .
Justification: Yes, we provide complete proofs for our theorems and describe detailed
empirical validation for our proposed algorithms.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] .

Justification: Yes, we discussed where our assumptions may fail and settings not captured
by the current paper, which we believe are directions for future research.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: Yes, we provide full assumptions in the main paper and the complete proofs
are written in the Appendix.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] .

Justification: Yes, our experimental section includes all details needed to reproduce the main
experimental results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes] .
Justification:
Yes, our code is open-sourced at https://github.com/CausalML/
adversarial-ope/.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]
Justification: Yes, please see our experimental section and appendices for all training and
evaluation details.

Guidelines:

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes] .
Justification: Yes, our experiments are replicated over multiple seeds and we report the
confidence intervals.

8. Experiments Compute Resources

38


---Page Break---
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes] .
Justification: Yes, this paper is mostly focused on theory and our experiment is a proof of
concept and can be run on a standard GPU.
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] .
Justification: Yes, we have reviewed the code of ethics and believe our research conforms to
it.
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA] .
Justification: This paper is about foundational research not tied to particular applications so
we do not feel the need to highlight any societal impacts.
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA] .
Justification: This paper is about foundational research not tied to particular applications so
we do not feel the need to highlight any risks for misuse here.
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [NA] .
Justification: The paper does not use any existing assets.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA] .
Justification: The paper does not release new assets.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA] .
Justification: The paper does not involve crowdsourcing nor research with human subjects.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA] .
Justification: The paper does not involve crowdsourcing nor research with human subjects.

39


---Page Break---
