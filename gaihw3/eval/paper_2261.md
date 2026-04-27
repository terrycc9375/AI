Theoretical Investigations and Practical Enhancements
on Tail Task Risk Minimization in Meta Learning

Yiqin Lv
Qi Wang∗
Dong Liang∗
Zheng Xie∗
College of Science, National University of Defense Technology
Changsha, China
Email to: {lvyiqin98,wangqi15,dongliangnudt,xiezheng81}@nudt.edu.cn

Abstract

Meta learning is a promising paradigm in the era of large models and task dis-
tributional robustness has become an indispensable consideration in real-world
scenarios. Recent advances have examined the effectiveness of tail task risk mini-
mization in fast adaptation robustness improvement [1]. This work contributes to
more theoretical investigations and practical enhancements in the field. Specifically,
we reduce the distributionally robust strategy to a max-min optimization problem,
constitute the Stackelberg equilibrium as the solution concept, and estimate the
convergence rate. In the presence of tail risk, we further derive the generalization
bound, establish connections with estimated quantiles, and practically improve the
studied strategy. Accordingly, extensive evaluations demonstrate the significance of
our proposal and its scalability to multimodal large models in boosting robustness.

1
Introduction

The past few years have witnessed a surge of research interest in meta learning due to its great
potential in the academia and industry [2–5]. By leveraging previous experience, such a learning
paradigm can extract knowledge as priors and empower learning models with adaptability to unseen
tasks from a few examples [6].

Nevertheless, the investigation of the robustness needs to be more comprehensive from the task
distribution perspective. In particular, the recently developed large models heavily rely on the few-
shot learning capability and demand robustness of prediction in risk-sensitive scenarios [7]. For
example, when the GPT-like dialogue generation system [8–10] comes into medical consultancy
domains, imprecise answers can cause catastrophic consequences to patients, families, and even
societies in real-world scenarios. In light of these considerations, it is desirable to watch adaptation
differences across tasks when deploying meta learning models and promote task robustness study for
meeting substantial practical demands.

Recently, Wang et al. [1] proposes to increase task distributional robustness via employing the tail
risk minimization principle [11] for meta learning. In circumventing the optimization intractability
in the presence of nonconvex risk functions, a two-stage optimization strategy is adopted as the
heuristic to solve the problem. In brief, the strategy consists of two phases in iteration, respectively:
(i) estimating the risk quantile VaRα [11] with the crude Monte Carlo method [12] in the task space;
(ii) updating the meta learning model parameters from the screened subset of tasks. Such a strategy is
simple in implementation, with an improvement guarantee under certain conditions, and empirically
shows improved robustness when faced with task distributional shifts. Despite these advances, there
remain several unresolved theoretical or practical issues in the field.

∗Correspondence Authors.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Existing limitations. This paper also works on the robustness of fast adaptation in the task space and
tries to fill gaps in [1]. Theoretically, we notice that in [1] (i) there constitutes no notion of solutions,
(ii) it lacks an algorithmic understanding of the two-stage optimization strategy, (iii) the analysis on
generalization capability is ignored in the tail risk of tasks. Empirically, the use of the crude Monte
Carlo might be less efficient in quantile estimates and suffers from a higher approximation error of
the VaRα, degrading the adaptation robustness. These bottlenecks may weaken the versatility of the
two-stage optimization strategy’s use in practice and require more understanding before deployment.

Primary contributions. In response to the above-mentioned concerns, we propose translating
the two-stage optimization strategy for distributionally robust meta learning [1] into a max-min
optimization problem [13]. Intrinsically, this work models the optimization steps as a Stackelberg
game, and task selection and the sub-gradient optimizer work as the leader and follower players in
decision-making, respectively. The theoretical understanding is from two aspects:

1. We constitute the local Stackelberg equilibrium as a solution concept, estimate the convergence
rate, and characterize the asymptotic behavior in learning dynamics.

2. We derive the generalization bound in the presence of the tail task risk, which connects quantile
estimates with fast adaptation capability in unseen tasks.

Meanwhile, the empirical influence of VaRα estimators is examined, and we advance meta learners’
robustness by comprising more accurate quantile estimators.

2
Literature Review

2.1
Meta Learning

Meta learning, or learning to learn, is an increasingly popular paradigm to distill knowledge from
prior experience to unseen scenarios with a few examples [6]. Various meta learning methods have
emerged in the past decade, and this section overviews some dominant families.

The context-based methods mainly use the encoder-decoder structure and represent tasks by latent
variables. Typical ones are in the form of the conditional exchangeable stochastic processes and
learn function distributions, such as neural processes [14], conditional neural processes [15] and their
extensions [16–24]. The optimization-based approaches seek the optimal meta initialization of model
parameters and update models from a few examples. Widely known are model agnostic meta learning
[25] and related variants [26–29], such as MetaCurvature [30], which learns curvature information
and transforms gradients in the inner-loop optimization. The metrics-based methods represent tasks
in geometry and perform well in few-shot image classification [31–33]. For example, MetaOptNet
[34] proposes to learn embeddings under a linear classifier and achieve SOTA few-shot classification
performance. There also exist other methods, e.g., hyper-networks [35, 36], memory-augmented
networks [37] and recurrent models [38].

2.2
Robustness & Generalization

The robustness concept in meta learning attracts recent attention, particularly when deploying
large models in real-world scenarios. Admittedly, previous literature works have investigated the
scenarios where the meta dataset’s input is corrupted [39, 40] or the model parameter is perturbed
[29]. Studies regarding the fast adaptation robustness in task distribution remain limited. Wang
et al. [41] explicitly generates task distribution for robust adaptation. Collins et al. [42] employs
the worst-case optimization for promoting MAML’s robustness to extreme worst cases. With the
help of tail risk minimization, Wang et al. [1] proposes two-stage optimization strategies to robustify
the fast adaptation. This work centers around [1] but stresses more theoretical understandings and
performance improvement points.

As for generalization capability, there are a couple of works in meta learning. Chen et al. [43]
exploits the information theory to derive the bound for MAML’s like methods. From the data splitting
perspective, Bai et al. [44] formulates the theoretical foundation and connects it to optimality. In [45],
an average risk bound is constructed with the bias for improving performance. Importantly, prior
work [1] ignores the generalization analysis, and meta learner’s generalization in tail risk cases has
not been studied in the literature.

2


---Page Break---
3
Preliminaries

General notations. Let p(τ) be the task distribution in meta learning. We respectively express the
task space and the model parameter space as Ωτ and Θ. We denote the complete task set by T and
refer to Dτ as the meta dataset.

For instance, Dτ comprises a collection of data points {(xi, yi)}n+m
i=1
in regression. Dτ is ususally
prepared into the support set DS
τ for skill transfer and the query set DQ
τ to assess adaptation perfor-
mance. Take the conditional neural process [15] as an example, DS
τ = {(xi, yi)}n
i=1 works for task
representation with DQ
τ = {(xi, yi)}n+m
i=1
the all data points to fit in regression.

The meta risk function corresponds to a map ℓ: Dτ × Θ 7→R+, evaluating fast adaptation
performance. Given p(τ) and meta learning model parameters θ, we can induce the cumulative
distribution of the meta risk function value in the real space as Fℓ(l; θ) := P({ℓ(DQ
τ , DS
τ ; θ) ≤l; τ ∈
T , l ∈R+}), but there is no explicit parameterized form for Fℓin practice as Fℓis θ-dependent.

When it comes to the tail risk minimization, we commonly use the conditional value-at-risk (CVaRα)
with the probability threshold α ∈[0, 1). The quantile of our interest is called the value-at-risk
(VaRα) [11] with the definition: VaRα [ℓ(T , θ)] = infl∈R+{l|Fℓ(l; θ) ≥α, τ ∈T }. The resulting
normalized cumulative distribution F α
ℓ(l; θ) is defined as:

F α
ℓ(l; θ) =

(
0,
l < VaRα[ℓ(T , θ)]
Fℓ(l;θ)−α

1−α
,
l ≥VaRα[ℓ(T , θ)].

∀θ ∈Θ, the meta learning operator Mθ defines: Mθ : τ 7→ℓ(DQ
τ , DS
τ ; θ). Accordingly, the tail risk
task subspace Ωα,τ := S

ℓ≥VaRα[ℓ(T ,θ)]

M−1
θ (ℓ)

, with the task distribution constrained in Ωα,τ by
pα(τ; θ). Please refer to Fig. 7 for illustrations of risk concepts.
Assumption 1. To proceed, we retain most assumptions from [1] for theoretical analysis, including:

1. The meta risk function ℓ(DQ
τ , DS
τ ; θ) is βτ-Lipschitz continuous w.r.t. θ;

2. The cumulative distribution Fℓ(l; θ) is βℓ-Lipschitz continuous w.r.t. l, and the normalized
density function pα(τ; θ) is βθ-Lipschitz continuous w.r.t. θ;

3. For arbitrary valid θ ∈Θ and corresponding pα(τ; θ), ℓ(DQ
τ , DS
τ ; θ) is bounded:
supτ∈Ωα,τ ℓ(DQ
τ , DS
τ ; θ) ≤Lmax.

3.1
Risk Minimization Principles

This subsection revisits commonly used risk minimization principles in the meta learning field.

Expected risk minimization. The standard principle is the expected/empirical risk minimization
originated from statistical learning theory [46]. It minimizes meta risk based on the sampling chance
of tasks from the original task distribution:

min
θ∈Θ E(θ) := Ep(τ)
h
ℓ(DQ
τ , DS
τ ; θ)
i
.
(1)

Worst-case risk minimization. Noticing that the worst fast adaptation can be disastrous in some
risk sensitive scenarios, Collins et al. [42] proposes to conduct the worst-case optimization in meta
learning:

min
θ∈Θ max
τ∈T Ew(θ) := ℓ(DQ
τ , DS
τ ; θ).
(2)

However, as observed from experiments in [42], such a principle inevitably sacrifices too much
average performance for gains of worst-case robustness. Meanwhile, it requires a couple of imple-
mentation tricks and specialized algorithms in stabilizing optimization.

Expected tail risk minimization (CVaRα). To balance the average performance and the worst-case
performance, Wang et al. [1] minimizes the expected tail risk, or equivalently CVaRα risk measure:

min
θ∈Θ Eα(θ) := Epα(τ;θ)
h
ℓ(DQ
τ , DS
τ ; θ)
i
.
(3)

3


---Page Break---
Due to no closed form of pα(τ; θ), Wang et al. [1] introduces a slack variable ξ ∈R and reformulates
the objective as follows:

min
θ∈Θ,ξ∈R Eα(θ, ξ) :=
1
1 −α

Z 1

α
vβdβ = ξ +
1
1 −αEp(τ)
h
ℓ(DQ
τ , DS
τ ; θ) −ξ
+i
,
(4)

where
vβ
:=
F −1
ℓ
(β) denotes the quantile statistics and

ℓ(DQ
τ , DS
τ ; θ) −ξ
+
:=
max{ℓ(DQ
τ , DS
τ ; θ) −ξ, 0} is the hinge risk.

The optimization objective involves the integral of quantiles in a continuous interval (α, 1], which is
intractable to precisely parameterize with neural networks. The form in Eq. (4) utilizes the duality
trick [11], enabling tractable sampling from the complete task space.

3.2
Examples & Two-stage Heuristic Strategies

Before delving deeper into the theoretical issues, we first present DR-MAML [1] as an instantiation
to explain the expected tail risk minimization.
Example 1 (DR-MAML [1]). Given p(τ) and vanilla MAML [25], the distributionally robust MAML
within CVaRα can be written as a bi-level optimization problem:

min
θ∈Θ
ξ∈R
ξ +
1
1 −αEp(τ)
h
ℓ(DQ
τ ; θ −λ∇θℓ(DS
τ ; θ)) −ξ
+i
,
(5)

where the gradient update w.r.t. the support set ∇θℓ(DS
τ ; θ) indicates the inner loop with a learning
rate λ. The outer loop executes the gradient updates w.r.t. Eq. (5) and seeks the robust meta
initialization in the parameter space.

Two-stage optimization strategies. Without loss of generality, we further detail the computa-
tional pipelines of Example 1 with two-stage optimization strategies. Note that MAML [25] is an
optimization-based meta learning method, and the implementation is to execute the sub-gradient
descent over a batch of tasks when updating the meta initialization θmeta:

θτi
t = θmeta
t
−λ1∇θℓ(DS
τi; θ), i = 1, . . . , B
(6a)
ˆξ = ˆF −1
MC-B(α),
(6b)

δ(τi) = 1[ℓ(DQ
τi; θτi
t ) ≥ˆξ], i = 1, . . . , B
(6c)

θmeta
t+1 ←θmeta
t
−λ2
h
B
X

i=1
∇θ[δ(τi) · ℓ(DQ
τi; θτi
t )]
i
.
(6d)

Sub-gradient for Meta Parameter Update

Leader's Move (Stage-I):
(1) Risk Distribution Modeling with KDEs
(2) Optimal Subset Selection in the Task
Batch

Follower's Move (Stage-II):

Meta Learning
Fast Adaptation

Meta Task Batch

Risk Density Modeling

Figure 1: Illustration of optimization stages
in distributionally robust meta learning
from a Stackelberg game. Given the DR-
MAML example, the pipeline can be inter-
preted as bi-level optimization: the leader’s
move for characterizing tail task risk and the
follower’s move for robust fast adaptation.

Here, λ1 and λ2 are the inner loop and the outer
loop learning rates, and the subscript t records the
iteration number, with δ(τi) the indicator variable.
ˆFMC-B is the empirical distribution with B Monte
Carlo task samples. δ(τi) = 1 indicates the meta risk
ℓ(DQ
τi; θτi
t ) after fast adaptation falls into the defined
tail risk region, otherwise δ(τi) = 0.

Throughout optimizing DR-MAML, Stage-I includes
the fast adaptation w.r.t. individual task in Eq. (6a),
and the quantile estimate in Eq. (6b). Stage-II ap-
plies the sub-gradient updates to the model parame-
ters in Eq. (6c)/(6d). These two stages repeat until
convergence is achieved.

4
Theoretical Investigations

This section presents theoretical insights into two-stage optimization strategies. We perform analysis
from the algorithmic convergence, the asymptotic tail risk robustness, and the cross-task generalization
capability in meta learning.

4


---Page Break---
...

...

...

...

...

Estimation

Surrogate Function

Optimization

Improvement

Guarantee
Existence & Solution Concept [Proposition 2]

Convergence Rate [Theorem 4.1]

Generalization in Tail Risk [Theorem 4.3]

CDF Approx. Error with KDE [Theorem 4.4]

Stackelberg Game for Meta Learning Robustification

Asymptotic Performance Gap [Theorem 4.2]
Equilibrium

Main Empirical Results [Extensive Evaluation]

Figure 2: The sketch of theoretical and empirical contributions in two-stage robust strategies.
On the left side is the two-stage distributionally robust strategy [1]. The contributed theoretical
understanding is right-down, with the right-up the empirical improvement. Arrows show connections
between components.

4.1
Distributionally Robust Meta Learning as a Stackelberg Game

Implementing the two-stage optimization strategy in meta learning requires first specifying the
stages’ order. The default is the minimization of the risk measure w.r.t. the parameter space after
the maximization of the risk measure w.r.t. the task subspace. Hence, we propose to connect it to
max-min optimization [13] and the Stackelberg game [47].

Max-min optimization. With the pre-assigned decision-making orders, the studied problem can be
characterized as:

max
q(τ)∈Qα min
θ∈Θ F(q, θ) := Eq(τ)
h
ℓ(DQ
τ , DS
τ ; θ)
i
,
(7)

where Qα := {q(τ)|Tq ⊆T ,
R

τ∈Tq p(τ)dτ = 1 −α} constitutes a collection of uncertainty sets [48]
over task subspace Tq, and q(τ) is the normalized probability density over the task subspace. Note
that in the expected tail risk minimization principle, there is no closed form of optimization objective
Eq. (4) as the tail risk is θ-dependent. It is approximately interpreted as the max-min optimization
when applied to the distribution over the uncertainty set Qα.

Proposition 1. The uncertainty set Qα is convex and compact in terms of probability measures.

Practical optimization is achieved via mini-batch gradient estimates and sub-gradient updates with
the task size B in [1]; the feasible subsets correspond to all combinations of size ⌈B ∗(1 −α)⌉. Also,
Eq. (7) is non-differentiable w.r.t. q(τ), leaving previous approaches [49–52] unavailable in practice.

Stackelberg game & best responses. The example computational pipelines in Eq. (6) can be under-
stood as approximately solving a stochastic two-player zero-sum Stackelberg game. Mathematically,
such a game referred to as SG can be depicted as SG := ⟨PL, PF ; {q ∈Qα}, {θ ∈Θ}; F(q, θ)⟩.

Moreover, we translate the two-stage optimization as decisions made by two competitors, which are
illustrated in Fig. 1. The maximization operator executes in the task space, corresponding to the
leader PL in SG with the utility function F(q, θ). The follower PF attempts to execute sub-gradient
updates over the meta learners’ parameters via maximizing −F(q, θ).

The two players compete to maximize separate utility functions in SG, which can be characterized as:

SG : qt = arg max
q∈Qα Eq
h
ℓ(DQ
τ , DS
τ ; θt)
i

|
{z
}
Leader Player

,
θt+1 = arg min
θ∈Θ Eqt
h
ℓ(DQ
τ , DS
τ ; θ)
i
,
|
{z
}
Follower Player
(8)

where the leader player PL specifies the worst case combinations from the uncertainty set Qα, and
the follower PF reacts to the resulting normalized tail risk for increasing fast adaptation robustness.

It is worth noting that the update rules in Eq. (8) are also called best responses of players in game
theory. The above procedures can be deemed the bi-level optimization [53] since the update of the
meta learner implicitly depends on the leader’s last time decision.

5


---Page Break---
4.2
Solution Concept & Properties

The improvement guarantee has been demonstrated when employing two-stage optimization strategies
for minimizing the tail risk in [1]. Furthermore, we claim that under certain conditions, there converges
to a solution for the proposed Stackelberg game SG. The sufficient evidence is:

1. The two-stage optimization [1] results in a monotonic sequence:

Model Updates : · · · 7→{qt−1, θt} 7→{qt, θt+1} 7→· · ·
(9a)
Monotonic Improvement : · · · ≥F(qt−1, θt) ≥F(qt, θt+1) ≥· · · ;
(9b)

2. As ℓ≤Lmax, the objective Eq
h
ℓ(DQ
τ , DS
τ ; θ)
i
≤Lmax naturally holds ∀q ∈Qα and θ ∈Θ.

Built on the boundness of risk functions and the theorem of improvement guarantee, such an
optimization process can finally converge [54]. Then, a crucial question arises concerning the
obtained solution: What is the notion of the convergence point in the game?

To answer this question, we need to formulate the corresponding solution concept in SG. Here, the
global Stackelberg equilibrium is introduced as follows.
Definition 1 (Global Stackelberg Equilibrium). Let (q∗, θ∗) ∈Qα × Θ be the solution. With the
leader q∗∈Qα and the follower θ∗∈Θ, (q∗, θ∗) is called a global Stackelberg equilibrium if the
following inequalities are satisfied, ∀q ∈Qα and ∀θ ∈Θ,

inf
θ′∈Θ F(q, θ′) ≤F(q∗, θ∗) ≤F(q∗, θ).

Proposition 2 (Existence of Equilibrium). Given the Assumption 1, there always exists the global
Stackelberg equilibrium as the Definition 1 for the studied SG.

Nevertheless, the existence of the global Stackelberg equilibrium can be guaranteed; it is NP-hard to
obtain the equilibrium with existing optimization techniques. The same as that in [55], we turn to the
local Stackelberg equilibrium as the Definition 2, where the notion of the local Stackelberg game is
restricted in a neighborhood Q′
α × Θ′ in strategies.
Definition 2 (Local Stackelberg Equilibrium). Let (q∗, θ∗) ∈Qα × Θ be the solution. With the
leader q∗∈Qα and the follower θ∗∈Θ, (q∗, θ∗) is called a local Stackelberg equilibrium for the
leader if the following inequalities hold, ∀q ∈Q′
α,

inf
θ∈SΘ′(q∗) F(q∗, θ) ≥
inf
θ∈SΘ′(q) F(q, θ), where SΘ′(q) := {¯θ ∈Θ′|F(q, ¯θ) ≤F(q, θ), ∀θ ∈Θ′}.

The nature of nonconvex programming comprises the above local optimum, and we introduce
concepts below for further analysis. It can be validated that F(q, θ) is a quasi-concave function w.r.t.
q, meaning that for any positive number l ∈R+, the set {q|q ∈Qα, F(q, θ) > l} is convex in Qα.
As a result, we deduce that there exists an implicit function h(·) : Θ →Qα such that the condition
holds h(θ) = q with q = arg max¯q∈Qα F(¯q, θ). For the implicit function h, along with ∇θF(q, θ),
we make the Assumption below.

Assumption 2. The implicit function h(·) is βh-Lipschitz continuous w.r.t. θ ∈Θ, and ∇θF(q, θ) is
βq-Lipschitz continuous w.r.t. q ∈Qα.

4.3
Convergence Rate & Generalization Bound

Learning to learn scales with the number of tasks, but the optimization process is computationally
expensive [56–60], particularly when large language models are meta learners [3, 61, 62]. In
training distributionally robust meta learners, estimating the convergence rate allows monitoring of
the convergence and designing early stopping criteria to reach a desirable performance, reducing
computational burdens [63]. Consequently, we turn to another question regarding the solution concept:
What is the convergence rate of the two-stage optimization algorithm?

The runtime complexity for the leader’s move can be easily estimated from subset selection, while
the analysis for the follower is non-trivial. Under certain conditions, we can derive the following
convergence rate theorem, where λ is the learning rate in gradient descent w.r.t. θ.

6


---Page Break---
Theorem 4.1 (Convergence Rate for the Second Player). Let the iteration sequence in op-
timization be:
· · · 7→{qt−1, θt} 7→{qt, θt+1} 7→· · · 7→{q∗, θ∗}, with the converged
equilibirum (q∗, θ∗).
Under the Assumption 2 and suppose that ||I −λ∇2
θθF(q∗, θ∗)||2 <
1 −λβqβh, we can have limt→∞
||θt+1−θ∗||2

||θt−θ∗||2
≤1, and the iteration converges with the rate
 
||I −λ∇2
θθF(q∗, θ∗)||2 + λβqβh

when t approaches infinity.

Moreover, after executing the two-stage algorithm T time steps and given learned θmeta
T
, we can
establish a bound on the asymptotic performance gap w.r.t. CVaRα in Theorem 4.2. For expositional
clarity, we simplify ℓ(DQ
τ , DS
τ ; θ∗), ℓ(DQ
τ , DS
τ ; θmeta
T
), VaRα [ℓ(T , θ∗)], and VaRα [ℓ(T , θmeta
T
)] as ℓ∗,
ℓmeta, VaR∗
α, and VaRmeta
α
, respectively.

Theorem 4.2 (Asymptotic Performance Gap in Tail Task Risk). Under the Assumption 1 and given a
batch of tasks {τi}B
i=1, we can have

CVaRα(θmeta
T
) −CVaRα(θ∗) ≤βτ∥θmeta
T
−θ∗∥+ VaR∗
α
1 −α


P(T1) −P(T2)

,
(10)

where T1 = {τ : ℓ∗< VaR∗
α, ℓmeta ≥VaRmeta
α
}, T2 = {τ : ℓ∗≥VaR∗
α, ℓmeta < VaRmeta
α
}.

For sufficiently large T, the first term can be bounded by a small number due to the convergence, and
the second term vanishes since limT →∞ℓmeta = ℓ∗and limT →∞VaRmeta
α
= VaR∗
α, respectively.

Another crucial issue regarding meta learning lies in the fast adaptation capability in unseen cases.
This drives us to answer the following question: How does the resulting meta learner generalize in
the presence of tail task risk?

To this end, we first define R(θ∗) = Epα(τ) [ℓ∗], bR(θ∗) =
1
B
PB
i=1 δ(τi)ℓ(DQ
τi, DS
τi; θ∗), and
bRw(θ∗) = 1

B
PB
i=1
pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗), where τi ∼p(τ). Also note that the support of pα(τ; θ∗)
is within that of p(τ), namely supp(pα(τ; θ∗)) ⊆supp(p(τ)). Then we can induce Theorem 4.3 w.r.t.
the tail risk generalization.

Theorem 4.3 (Generalization Bound in the Tail Risk Cases). Given a collection of task samples
{τi}B
i=1 and corresponding meta datasets, we can derive the following generalization bound in the
presence of tail risk:

R(θ∗) ≤bR(θ∗) +

s

2
 
α
1−αL2max + Vτi∼pα(τ)

ℓ(DQ
τi, DSτi; θ∗)

ln
  1

ϵ


B

+
1
3(1 −α)
Lmax

B


2 ln
1

ϵ


+ 3αB

,

(11)

where the inequality holds with probability at least 1 −ϵ and ϵ ∈(0, 1), V[·] denotes the variance
operation, and Lmax is from the Assumption 1.

In conjunction with the confidence ϵ and a task batch B of significant size, Theorem 4.3 reveals the
generalization bound given the meta-trained parameter θ∗. It is also associated with the variance
Vτi∼pα(τ)[ℓ(DQ
τi, DS
τi; θ∗)]. Besides, we also derive a specific bound in the case of MAML, and
details are attached in Appendix Theorem C.1.

4.4
Practical Enhancements & Implementations

Theorem 4.3 reveals that an accurate estimate of VaRα yields a precise variance (i.e.,
Vτi∼pα(τ)[ℓ(DQ
τi, DS
τi; θ∗)]), leading to more reliable bounds. Accordingly, this section offers
improvements over [1] via utilizing kernel density estimators (KDE) [64] for VaRα’s estimates.
Compared to crude Monte Carlo (MC) methods, KDE can handle arbitrary complex distributions,
capture local statistics well, and smoothen the cumulative function in a non-parametric way.

Specifically, we can construct KDE with a batch of task risk values {ℓ(DQ
τi, DS
τi; θ)}B
i=1:

Fℓ-KDE(l; θ) =
Z l

−∞

1
Bhℓ

B
X

i=1
K
t −ℓ(DQ
τi, DS
τi; θ)
hℓ


dt,
(12)

7


---Page Break---
where K : Rd →R is a kernel function, e.g., the Gaussian kernel, K(x) =
exp(−||x||2/2)
R
exp(−||x||2/2)dx, and
hℓis the smoothing bandwidth. Once the KDE is built, it enables access to the quantile from the
cumulative distribution functions or numeric integrals. The following Theorem 4.4 shows that KDE
serves as a reliable approximation for VaRα.

Theorem 4.4. Let F −1
ℓ-KDE(α; θ) = VaRKDE
α
[ℓ(T , θ)] and F −1
ℓ
(α; θ) = VaRα[ℓ(T , θ)]. Suppose that
K(x) is lower bounded by a constant, ∀x. For any ϵ > 0, with probability at least 1 −ϵ, we can have
the following bound:

sup
θ∈Θ

 
F −1
ℓ-KDE(α; θ) −F −1
ℓ
(α; θ)

≤O

hℓ
√B ∗log B


.
(13)

As implied, one can close the distribution approximation gap by adopting a smaller, more flexible
bandwidth. Additionally, KDE models offer a smooth estimate of the cumulative distribution function
and require no prior assumptions.
Remark 1. In addition to smoothness, flexibility, and distribution agnostic traits, KDE in adoption
can enhance the studied method’s generalization capability. The crude Monte Carlo used in [1]
typically incurs an error of approximately O( 1
√

B) in estimating quantiles [65]. In contrast, that of

KDE is no more than O(
hℓ
√B∗log B) from Theorem 4.4.

5
Empirical Findings

Prior sections mainly focus on the theoretical understanding of two-stage distributionally robust
strategies. This section conducts extensive experiments on a broader range of benchmarks and
examines the improvement tricks, e.g., the use of KDE for quantile estimates, from empirical results.

Benchmarks & baselines. We perform experiments on the few-shot regression, system identification,
image classification, and meta reinforcement learning, where most of them keep setups the same as
prior work [1, 42]. We evaluate the methods from risk minimization principles and corresponding
indicators, including expected/empirical risk minimization (Average), worst-case risk minimization
(Worst), and tail risk minimization (CVaRα).

MAML mainly works as the base meta learner, and we term the KDE-augmented DR-MAML as
DR-MAML+. Then we compare DR-MAML+ with several baselines, including vanilla MAML [25],
TR-MAML [42], DRO-MAML [66] and DR-MAML [1].

Average
Worst
CVaRα

1

2

3

4

5
Sinusoid 5-shot

Average
Worst
CVaRα

0.5

1.0

1.5

2.0

2.5

3.0

3.5
Sinusoid 10-shot

Mean Square Errors

MAML
TR-MAML
DRO-MAML
DR-MAML
DR-MAML+(Ours)

Figure 3: Meta testing performance in sinu-
soid regression problems (5 runs). The charts
report testing mean square errors (MSEs) over
490 unseen tasks [42] with α = 0.7, where
black vertical lines indicate standard error bars.

Average
Worst
CVaRα

0.5

1.0

1.5

2.0
Pendulum 10-shot

Average
Worst
CVaRα

0.5

1.0

1.5

2.0
Pendulum 20-shot

Mean Square Errors

MAML
TR-MAML
DRO-MAML
DR-MAML
DR-MAML+(Ours)

Figure 4: Meta testing performance in Pen-
dulum 10-shot and 20-shot problems (5
runs). Reported are testing MSEs over 529 un-
seen tasks with α = 0.5, where black vertical
lines indicate standard error bars.

5.1
Sinusoid Regression

The goal of the sinusoid regression [25] is to quickly fit an underlying function f(x) = A sin(x −B)
from K randomly sampled data points, and tasks are specified by (A, B). The meta-training and
testing setups are the same as that in [1, 42], where many easy functions with a tiny fraction of
difficult ones are included in the training.

Result & analysis. As illustrated in Fig. 3, we can observe that DR-MAML+ consistently outper-
forms all baselines across average and CVaRα indicators in the 5-shot case. Though the average
performance slightly lags behind DR-MAML in the 10-shot case, DR-MAML+ surpasses other
baselines in both the Worst and CVaRα indicators. This implies that DR-MAML+ exhibits more

8


---Page Break---
robustness in challenging task distributions, e.g., 5-shot case. Furthermore, the standard error asso-
ciated with our method is significantly smaller than others, underscoring the stability of DR-MAML+.

5.2
System Identification

The system identification corresponds to learning a dynamics model from a few collected transitions
in physics systems. Here, we consider the Pendulum system and create diverse dynamical systems
by varying its mass m and length l, with (m, l) ∼U([0.4, 1.6], [0.4, 1.6]). A random policy collects
transitions for meta training, and 10 random transitions work as a support dataset.

Result & analysis. Fig. 4 shows no significant difference between 10-shot and 20-shot cases.
DR-MAML+ dominates the performance across all indicators in both cases. Due to the min-max
optimization, TR-MAML behaves well in the worst-case but sacrifices too much average performance.
Within the studied strategies, DR-MAML+ exhibits an advantage over DR-MAML regarding CVaRα.

5.3
Few-shot Image Classification

We perform few-shot image classification on the mini-ImageNet dataset [67], with the same setup in
[42]. The task is a 5-way 1-shot classification problem. And 64 classes are selected for constructing
meta-training tasks, with the remaining 32 classes for meta-testing.

Table 1: Average 5-way 1-shot classification accuracies in mini-ImageNet with reported
standard deviations (3 runs). With α = 0.5, the best results are in bold.

Eight Meta-Training Tasks
Four Meta-Testing Tasks
Method
Average
Worst
CVaRα
Average
Worst
CVaRα

MAML [25]
70.1±2.2
48.0±4.5
63.2±2.6
46.6±0.4
44.7±0.7
44.6±0.7
TR-MAML [42]
63.2±1.3
60.7±1.6
62.1±1.2
48.5±0.6
45.9±0.8
46.6±0.5
DRO-MAML [66]
67.0±0.2
56.6±0.4
61.6±0.2
49.1±0.2
46.6±0.1
47.2±0.2
DR-MAML [1]
70.2±0.2
63.4±0.2
67.2±0.1
49.4±0.1
47.1±0.1
47.5±0.1
DR-MAML+(Ours)
70.4±0.1
63.8±0.2
67.5±0.1
49.9±0.1
47.2±0.1
48.1±0.1

Result & analysis. In Table 1, methods within a two-stage distributionally robust strategy, namely
DR-MAML and DR-MAML+, show superiority to others across all indicators in both training and
testing scenarios, which is similar to empirical findings in [1]. Interstingly, DR-MAML+ and DR-
MAML are comparable in most scenarios, and we attribute this to the small batch size in training,
which weakens KDE’s quantile approximation advantage.

5.4
Meta Reinforcement Learning

Table 2: Meta testing returns in point robot navigation (4 runs).
The chart reports average return and CVaRα return with α = 0.5.

Method
Average
CVaRα

MAML [25]
-21.1 ± 0.69
-29.2 ± 1.37
DRO-MAML [66]
-20.9 ± 0.41
-29.0 ± 0.66
DR-MAML [1]
-19.6 ± 0.49
-28.9 ± 1.20
DR-MAML+(Ours)
-19.2± 0.44
-28.4± 0.86

Here, we take 2-D point robot
navigation as the meta reinforce-
ment learning benchmark in eval-
uation. The goal is to reach the
target destination with the help of
a few exploration transitions for
fast adaptation, and we retain the
setup in MAML [25]. In meta
testing, we randomly sample 80 navigation goals and examine methods’ navigation performance.

20
30
40
50
60
70
Task Batch Size

0.02

0.04

0.06

Sinusoid 5-shot

MC
KDE

20
30
40
50
60
70
Task Batch Size

0.005

0.010

0.015

0.020

0.025

Sinusoid 10-shot

MC
KDE

Approximation Error

Figure 5: VaRα approximation errors with the
crude MC and KDE. We compute the difference
between the estimated
ˆ
VaRα and the Oracle VaRα
in the absolute value | ˆ
VaRα −VaRα|.

Result & analysis. As reinforcement learning
methods fluctuate fiercely in worst-case indi-
cators, we only report Average and CVaRα re-
turns in Table 2. We observe that using studied
strategies in DR-MAML enhances the returns.
DR-MAML+ benefits from a more reliable quan-
tile estimate and achieves superior performance.
The application of distributional robustness to
reinforcement learning yields improvements in
returns.

9


---Page Break---
5.5
Assessment of Quantile Estimators

With the meta trained model, e.g., DR-MAML+ in sinusoid regression, we collect the testing task
risk values with different task batch sizes to estimate the VaRα from respectively the crude MC and
KDE. As observed from Fig. 5, the VaRα approximation error decreases with more tasks, and the
KDE produces more accurate estimates with a sharper decreasing trend. The above well verifies the
conclusion in Theorem 4.3.

5.6
Empricial Result Summarization

Here, we summarize two points from the above empirical results and associated theorems. (i) From
Theorem 4.2/4.3 and Fig. 3/4/5: the VaRα estimate relates to the reliable generalization bound, and
cumulated tiny approximation errors along iterations potentially result in worse equilibrium. (ii)
From Theorem 4.3/4.4, Remark 1, Fig. 3, and Table 1/2: with the studied strategy, the KDE is a better
choice of task risk distribution modelling than the crude MC in tougher benchmarks, e.g., 5-shot
sinusoid regression, meta-testing mini-ImageNet classification, and point robot navigation.

5.7
Compatibility with Large Models

Average
CVaR

70

75

80

85

90

95

100
tiered-ImageNet

Average
CVaR

75.0

77.5

80.0

82.5

85.0

87.5

90.0

92.5

95.0
ImageNetA

Average
CVaR

80.0

82.5

85.0

87.5

90.0

92.5

95.0

97.5

100.0
ImageNetSketch

Classification Accuracies

CLIP
MaPLe
DR-MaPLe
DR-MaPLe+

Figure 6: Meta testing results on 5-way 1-shot classification accuracies with reported standard
deviations (3 runs). The charts respectively report classification accuracies over 150 unseen tasks.
We further conduct few-shot image classification experiments in the presence of large model. Note
that CLIP [68] exhibits strong zero-shot adaptation capability; hence, we employ "ViT-B/16"-based
CLIP as the backbone to enable few-shot learning in the same way as MaPLe with training setup
N_CTX = 2 and MAX_EPOCH = 30 [69], scaling to large neural networks in evaluation (See
Appendix Section D for details).

Improved Robustness in Evaluation: As illustrated in Fig. 6, DR-MaPLe and DR-MaPLe+
consistently outperform baselines across both average and indicators in cases, demonstrating the
advantage of the two-stage strategy in enhancing the robustness of few-shot learning. DR-MaPLe+
achieves better results as KDE quantiles are more accurate with large batch sizes. These results
confirm the scalability and compatibility of our method on large models.

Learning Efficiency as Limitations: In terms of implementation time and memory cost, we retain
the setup the same as that in [1]: use the same maximum number of meta gradient updates for all
baselines in training processes, which means given α = 0.5, the tail risk minimization principle
requires double task batches to evaluate and screen sub-batches. It can be seen that both DR-MaPLe
and DR-MaPLe+ consume more memories, and the extra training time over MaPLe arises from the
evaluation and sub-batch screening in the forward pass. Such additional computations and memory
costs raise computational and memory efficiency issues for exchanging extra significant robustness
improvement in fast adaptation.

6
Conclusion

To conclude, this paper proposes to understand the two-stage distributionally robust strategy from
optimization processes, define the convergence solution, and derive the generalization bound in the
presence of tail task risk. Extensive experiments validate the studied improvement tricks and reveal
more empirical properties of the studied strategy. We leave computational overhead reduction as a
promising topic for future exploration in robust fast adaptation.

10


---Page Break---
Acknowledgement

This work is funded by National Natural Science Foundation of China (NSFC) with the Number #
62306326. We express particular gratitude to friends who guide large model-relevant experiments.

References

[1] Qi Wang, Yiqin Lv, Yanghe Feng, Zheng Xie, and Jincai Huang. A simple yet effective strategy
to robustify the meta learning paradigm. In Thirty-seventh Conference on Neural Information
Processing Systems, 2023.

[2] Yuanfu Lu, Yuan Fang, and Chuan Shi. Meta-learning on heterogeneous information networks
for cold-start recommendation.
In Proceedings of the 26th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining, pages 1563–1573, 2020.

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[4] Yi Yuan, Gan Zheng, Kai-Kit Wong, and Khaled B Letaief. Meta-reinforcement learning
based resource allocation for dynamic v2x communications. IEEE Transactions on Vehicular
Technology, 70(9):8964–8977, 2021.

[5] Brenden M Lake and Marco Baroni. Human-like systematic generalization through a meta-
learning neural network. Nature, pages 1–7, 2023.

[6] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in
neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44
(9):5149–5169, 2021.

[7] Qi Wang, Yanghe Feng, Jincai Huang, Yiqin Lv, Zheng Xie, and Xiaoshan Gao. Large-scale
generative simulation artificial intelligence: The next hotspot. The Innovation, page 100516,
2023.

[8] Young-Jun Lee, Chae-Gyun Lim, and Ho-Jin Choi. Does gpt-3 generate empathetic dialogues?
a novel in-context example selection method and automatic evaluation metric for empathetic
dialogue generation. In Proceedings of the 29th International Conference on Computational
Linguistics, pages 669–683, 2022.

[9] Chen Tang, Hongbo Zhang, Tyler Loakman, Chenghua Lin, and Frank Guerin. Terminology-
aware medical dialogue generation. In ICASSP 2023-2023 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[10] Bharath Chintagunta, Namit Katariya, Xavier Amatriain, and Anitha Kannan. Medically
aware gpt-3 as a data generator for medical dialogue summarization. In Machine Learning for
Healthcare Conference, pages 354–372. PMLR, 2021.

[11] R Tyrrell Rockafellar, Stanislav Uryasev, et al. Optimization of conditional value-at-risk.
Journal of risk, 2:21–42, 2000.

[12] Dirk P Kroese and Reuven Y Rubinstein. Monte carlo methods. Wiley Interdisciplinary Reviews:
Computational Statistics, 4(1):48–58, 2012.

[13] John M Danskin. The theory of max-min, with applications. SIAM Journal on Applied
Mathematics, 14(4):641–664, 1966.

[14] Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami,
and Yee Whye Teh. Neural processes. arXiv preprint arXiv:1807.01622, 2018.

[15] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray
Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional neural processes.
In International Conference on Machine Learning, pages 1704–1713. PMLR, 2018.

11


---Page Break---
[16] Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard Turner.
Meta-learning probabilistic inference for prediction. In International Conference on Learning
Representations, 2018.

[17] Qi Wang, Marco Federici, and Herke van Hoof. Bridge the inference gaps of neural pro-
cesses via expectation maximization. In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.net/forum?id=A7v2DqLjZdq.

[18] Andrew Foong, Wessel Bruinsma, Jonathan Gordon, Yann Dubois, James Requeima, and
Richard Turner. Meta-learning stationary stochastic process prediction with convolutional
neural processes. Advances in Neural Information Processing Systems, 33:8284–8295, 2020.

[19] Qi Wang and Herke Van Hoof. Doubly stochastic variational inference for neural processes
with hierarchical latent variables. In International Conference on Machine Learning, pages
10018–10028. PMLR, 2020.

[20] Muhammad Waleed Gondal, Shruti Joshi, Nasim Rahaman, Stefan Bauer, Manuel Wuthrich,
and Bernhard Scholkopf. Function contrastive learning of transferable meta-representations. In
Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24
July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages
3755–3765. PMLR, 2021.

[21] Qi Wang and Herke van Hoof. Learning expressive meta-representations with mixture of expert
neural processes. In Advances in neural information processing systems, 2022.

[22] Juho Lee, Yoonho Lee, Jungtaek Kim, Eunho Yang, Sung Ju Hwang, and Yee Whye Teh.
Bootstrapping neural processes. Advances in neural information processing systems, 33:6606–
6615, 2020.

[23] Qi Wang and Herke Van Hoof. Model-based meta reinforcement learning using graph structured
surrogate models and amortized policy search.
In International Conference on Machine
Learning, pages 23055–23077. PMLR, 2022.

[24] Jiayi Shen, Xiantong Zhen, Marcel Worring, et al. Episodic multi-task learning with heteroge-
neous neural processes. arXiv preprint arXiv:2310.18713, 2023.

[25] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adap-
tation of deep networks. In International conference on machine learning, pages 1126–1135.
PMLR, 2017.

[26] Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with
implicit gradients. Advances in neural information processing systems, 32, 2019.

[27] Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting
gradient-based meta-learning as hierarchical bayes. arXiv preprint arXiv:1801.08930, 2018.

[28] Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning.
Advances in neural information processing systems, 31, 2018.

[29] Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, and Tianyi Chen. Sharp-maml: Sharpness-
aware model-agnostic meta learning. In International Conference on Machine Learning, pages
10–32. PMLR, 2022.

[30] Eunbyung Park and Junier B Oliva. Meta-curvature. Advances in neural information processing
systems, 32, 2019.

[31] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning.
Advances in neural information processing systems, 30, 2017.

[32] Kelsey Allen, Evan Shelhamer, Hanul Shin, and Joshua Tenenbaum. Infinite mixture prototypes
for few-shot learning. In International Conference on Machine Learning, pages 232–241.
PMLR, 2019.

12


---Page Break---
[33] Sergey Bartunov and Dmitry Vetrov. Few-shot generative modelling with generative matching
networks. In International Conference on Artificial Intelligence and Statistics, pages 670–678.
PMLR, 2018.

[34] Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with
differentiable convex optimization. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), June 2019.

[35] Jacob Beck, Matthew Thomas Jackson, Risto Vuorio, and Shimon Whiteson. Hypernetworks
in meta-reinforcement learning. In Conference on Robot Learning, pages 1478–1487. PMLR,
2023.

[36] Dominic Zhao, Johannes von Oswald, Seijin Kobayashi, João Sacramento, and Benjamin F
Grewe. Meta-learning via hypernetworks. 4th Workshop on Meta-Learning at NeurIPS 2020,
2020.

[37] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap.
Meta-learning with memory-augmented neural networks.
In International conference on
machine learning, pages 1842–1850. PMLR, 2016.

[38] Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl2:
Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779,
2016.

[39] Ren Wang, Kaidi Xu, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Chuang Gan, and Meng Wang.
On fast adversarial robustness adaptation in model-agnostic meta-learning. In International
Conference on Learning Representations, 2020.

[40] Micah Goldblum, Liam Fowl, and Tom Goldstein. Adversarially robust few-shot learning: A
meta-learning approach. Advances in Neural Information Processing Systems, 33:17886–17895,
2020.

[41] Cheems Wang, Yiqin Lv, Yixiu Mao, Yun Qu, Yi Xu, and Xiangyang Ji. Robust fast adaptation
from adversarially explicit task distribution generation. arXiv preprint arXiv:2407.19523, 2024.

[42] Liam Collins, Aryan Mokhtari, and Sanjay Shakkottai. Task-robust model-agnostic meta-
learning. Advances in Neural Information Processing Systems, 33:18860–18871, 2020.

[43] Qi Chen, Changjian Shui, and Mario Marchand. Generalization bounds for meta-learning:
An information-theoretic analysis. Advances in Neural Information Processing Systems, 34:
25878–25890, 2021.

[44] Yu Bai, Minshuo Chen, Pan Zhou, Tuo Zhao, Jason Lee, Sham Kakade, Huan Wang, and
Caiming Xiong. How important is the train-validation split in meta-learning? In International
Conference on Machine Learning, pages 543–553. PMLR, 2021.

[45] Giulia Denevi, Carlo Ciliberto, Riccardo Grazzi, and Massimiliano Pontil. Learning-to-learn
stochastic gradient descent with biased regularization. In International Conference on Machine
Learning, pages 1566–1575. PMLR, 2019.

[46] Vladimir Vapnik. The nature of statistical learning theory. Springer science & business media,
1999.

[47] Tao Li and Suresh P Sethi. A review of dynamic stackelberg game models. Discrete &
Continuous Dynamical Systems-B, 22(1):125, 2017.

[48] Aharon Ben-Tal, Dick Den Hertog, Anja De Waegenaere, Bertrand Melenberg, and Gijs Rennen.
Robust solutions of optimization problems affected by uncertain probabilities. Management
Science, 59(2):341–357, 2013.

[49] Dimitri P Bertsekas. Nonlinear programming. Journal of the Operational Research Society, 48
(3):334–334, 1997.

13


---Page Break---
[50] Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave
minimax problems. In International Conference on Machine Learning, pages 6083–6093.
PMLR, 2020.

[51] Pierre Loridan and Jacqueline Morgan. Weak via strong stackelberg problem: new results.
Journal of global Optimization, 8:263–287, 1996.

[52] Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters
by implicit differentiation. In International conference on artificial intelligence and statistics,
pages 1540–1552. PMLR, 2020.

[53] Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. Investigating bi-level
optimization for learning and vision from a unified perspective: A survey and beyond. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 44(12):10045–10067, 2021.

[54] Tom M Apostol. Mathematical analysis. 1974.

[55] Tanner Fiez, Benjamin Chasnov, and Lillian Ratliff. Implicit learning dynamics in stackelberg
games: Equilibria characterization, convergence analysis, and empirical study. In International
Conference on Machine Learning, pages 3133–3144. PMLR, 2020.

[56] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for
modern deep learning research. In Proceedings of the AAAI conference on artificial intelligence,
volume 34, pages 13693–13696, 2020.

[57] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel
Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network
training. arXiv preprint arXiv:2104.10350, 2021.

[58] Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish. Scaling laws for
transfer. arXiv preprint arXiv:2102.01293, 2021.

[59] Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He,
Antong Li, Mengshen He, Zhengliang Liu, et al. Summary of chatgpt-related research and
perspective towards the future of large language models. Meta-Radiology, page 100017, 2023.

[60] Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva,
Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. Chatgpt
for good? on opportunities and challenges of large language models for education. Learning
and individual differences, 103:102274, 2023.

[61] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot
learners. In Joint Conference of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Processing,
ACL-IJCNLP 2021, pages 3816–3830. Association for Computational Linguistics (ACL), 2021.

[62] Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Parteek Sharma, Fan Chen, and Lei Jiang.
Llmcarbon: Modeling the end-to-end carbon footprint of large language models. arXiv preprint
arXiv:2309.14393, 2023.

[63] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[64] Mats Rudemo. Empirical choice of histograms and kernel density estimators. Scandinavian
Journal of Statistics, pages 65–78, 1982.

[65] Hui Dong and Marvin K Nakayama. A tutorial on quantile estimation via monte carlo. Monte
Carlo and Quasi-Monte Carlo Methods: MCQMC 2018, Rennes, France, July 1–6, pages 3–30,
2020.

[66] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust
neural networks. In International Conference on Learning Representations, 2020.

14


---Page Break---
[67] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks
for one shot learning. Advances in neural information processing systems, 29, 2016.

[68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning,
pages 8748–8763. PMLR, 2021.

[69] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fa-
had Shahbaz Khan. Maple: Multi-modal prompt learning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 19113–19122, 2023.

[70] Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, and Xiangyang Ji. Supported trust region
optimization for offline reinforcement learning. In International Conference on Machine
Learning, pages 23829–23851. PMLR, 2023.

[71] Hongchang Zhang, Yixiu Mao, Boyuan Wang, Shuncheng He, Yi Xu, and Xiangyang Ji. In-
sample actor critic for offline reinforcement learning. In The Eleventh International Conference
on Learning Representations, 2023.

[72] Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, and Xiangyang Ji.
Supported value
regularization for offline reinforcement learning. Advances in Neural Information Processing
Systems, 36, 2024.

[73] Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. Lectures on stochastic
programming: modeling and theory. Society for industrial Mathematics, 2009.

[74] Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, and Xiangyang Ji. Counterfactual
conservative q learning for offline multi-agent reinforcement learning. Advances in Neural
Information Processing Systems, 36, 2024.

[75] Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Liu Linc,
Yang Feng, Lin Lai, Hongyang Qin, et al. Hokoff: real game dataset from honor of kings and
its offline reinforcement learning benchmarks. Advances in Neural Information Processing
Systems, 36, 2024.

[76] Lorna I Paredes and Chew Tuan Seng. Controlled convergence theorem for banach-valued hl
integrals. Scientiae Mathematicae Japonicae, 56(2):347–358, 2002.

[77] Chi Jin, Praneeth Netrapalli, and Michael Jordan. What is local optimality in nonconvex-
nonconcave minimax optimization? In International conference on machine learning, pages
4880–4889. PMLR, 2020.

[78] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press,
2004.

[79] C Frappier and QI Rahman. On an inequality of s. bernstein. Canadian Journal of Mathematics,
34(4):932–944, 1982.

[80] Rong Liu and Lijian Yang. Kernel estimation of multivariate cumulative distribution function.
Journal of Nonparametric Statistics, 20(8):661–677, 2008.

[81] Srh Larochelle. Optimization as a model for few-shot learning. In International Conference on
Learning Representations, 2017.

[82] Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B Tenen-
baum, Hugo Larochelle, and Richard S Zemel. Meta-learning for semi-supervised few-shot
classification. arXiv preprint arXiv:1803.00676, 2018.

[83] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural
adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 15262–15271, 2021.

15


---Page Break---
[84] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global repre-
sentations by penalizing local predictive power. Advances in Neural Information Processing
Systems, 32, 2019.

[85] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmai-
son, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani,
Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala.
Py-
torch: An imperative style, high-performance deep learning library. In Advances in Neu-
ral Information Processing Systems 32: Annual Conference on Neural Information Pro-
cessing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada,
pages 8024–8035, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/
bdbca288fee7f92f2bfa9f7012727740-Abstract.html.

[86] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu
Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. {TensorFlow}: a system for
{Large-Scale} machine learning. In 12th USENIX symposium on operating systems design and
implementation (OSDI 16), pages 265–283, 2016.

16


---Page Break---
Contents

1
Introduction
1

2
Literature Review
2

2.1
Meta Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

2.2
Robustness & Generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2

3
Preliminaries
3

3.1
Risk Minimization Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3

3.2
Examples & Two-stage Heuristic Strategies . . . . . . . . . . . . . . . . . . . . .
4

4
Theoretical Investigations
4

4.1
Distributionally Robust Meta Learning as a Stackelberg Game . . . . . . . . . . .
5

4.2
Solution Concept & Properties . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6

4.3
Convergence Rate & Generalization Bound
. . . . . . . . . . . . . . . . . . . . .
6

4.4
Practical Enhancements & Implementations . . . . . . . . . . . . . . . . . . . . .
7

5
Empirical Findings
8

5.1
Sinusoid Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8

5.2
System Identification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

5.3
Few-shot Image Classification
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

5.4
Meta Reinforcement Learning
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

5.5
Assessment of Quantile Estimators . . . . . . . . . . . . . . . . . . . . . . . . . .
10

5.6
Empricial Result Summarization . . . . . . . . . . . . . . . . . . . . . . . . . . .
10

5.7
Compatibility with Large Models . . . . . . . . . . . . . . . . . . . . . . . . . . .
10

6
Conclusion
10

A Quick Guide to This Work
19

A.1 Technical Comparison in Robust Fast Adaptation . . . . . . . . . . . . . . . . . .
19

A.2
Significance of Theoretical Understandings
. . . . . . . . . . . . . . . . . . . . .
19

A.3
Meanings of Indicators and Terms . . . . . . . . . . . . . . . . . . . . . . . . . .
20

A.4
Computational Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

A.5
Broader Impact & Future Extensions . . . . . . . . . . . . . . . . . . . . . . . . .
20

B
Pseudo Algorithms
20

C Expressions, Theorems & Proofs
21

C.1
Characterization of Optimization Processes
. . . . . . . . . . . . . . . . . . . . .
21

C.2
Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

C.3
Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

C.4
Proof of Proposition 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

17


---Page Break---
C.5
Proof of Quasi-concavity for F(q, θ) w.r.t. q . . . . . . . . . . . . . . . . . . . . .
24

C.6
Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

C.7
Proof of Theorem 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

C.8
Proof of Theorem 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

C.9
Proof of Theorem 4.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

C.10 Additional Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

D Implementation Details
30

D.1
Benchmark Details & Neural Architectures & Opensource Codes . . . . . . . . . .
30

D.2
Modules in Python
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31

E Additional Experimental Results
32

E.1
Evaluation with Other Robust Meta Learners
. . . . . . . . . . . . . . . . . . . .
32

E.2
Numeric Results in Tables and Histograms . . . . . . . . . . . . . . . . . . . . . .
32

E.3
Sensitivity Analysis to Confidence Level . . . . . . . . . . . . . . . . . . . . . . .
33

E.4
Further Exploration on Adaptation . . . . . . . . . . . . . . . . . . . . . . . . . .
35

F
Computational Platforms & Softwares
35

18


---Page Break---
A
Quick Guide to This Work

This section mainly includes explanations and clarifications on this work.

A.1
Technical Comparison in Robust Fast Adaptation

Table 3: A summary of robust fast adaptation methods. We take MAML as an example, list related
methods, and report their characteristics in literature. We mainly report the statistics according to
whether existing literature works include the generalization analysis and convergence analysis. The
form of meta learner and the robustness type are generally connected.

Principle
Meta Learner
Generalization Convergence
Robustness Type

MAML
minθ∈Θ Ep(τ)
h
ℓ(DQ
τ , DS
τ ; θ)
i

✓
✓
−−

DRO-MAML [66]
maxq(τ)∈Q minθ∈Θ Eq(τ)
h
ℓ(DQ
τ , DS
τ ; θ)
i

✗
✗
Uncertainty Set (Not tail risk)

TR-MAML [42]
minθ∈Θ maxτ∈T ℓ(DQ
τ , DS
τ ; θ)
✓
✓
Worst-Case Task

DR-MAML [1]
minθ∈Θ Epα(τ;θ)
h
ℓ(DQ
τ , DS
τ ; θ)
i

✗
✗
Tail Task Risk

DR-MAML+(Ours) maxq(τ)∈Qα minθ∈Θ Eq(τ)
h
ℓ(DQ
τ , DS
τ ; θ)
i

✓
✓
Tail Task Risk

Primary differences: As far as we know, literature work is quite limited regarding fast adaptation
robustness in the task space. TR-MAML and DR-MAML are the most recent and typical ones
that can handle task distributional shift scenarios well. As reported in Table 3, TR-MAML only
focuses on the worst-case, which considers a bit extreme and rarely occurred cases. DRO-MAML
is a new baseline, where the uncertainty set Q is included for robust fast adaptation, hence there
exists no theoretical analysis. As for the tail task risk, DR-MAML lacks generalization capability and
convergence rate analysis w.r.t. the meta learner. The meta learner in DR-MAML+ is a more specific
instantiation of that in DR-MAML. We claim that these theoretical understanding is necessary in the
presence of the robust fast adaptation due to its potential applications in large models.

Theoretical and empirical insights: In comparison, this work not only contributes to the Stackelberg
game for estimates, but also derives the generalization and the asymptotic performance gap in
iterations based on a normalized but non-differentiable probability density space. Note that we
lean more focus on theoretical understanding and pursuing SOTA performance is not the ultimate
purpose of this work. The connections between different quantile estimators and generalization
bound highlighted in Theorem 4.3/4.4 and Remark 1 reveal the theoretical advantage of KDEs over
crude Monte Carlo methods. This motivates us to replace crude Monte Carlo with KDEs. Such
a replacement as a simple implementation trick is supported by rigorous theoretical analysis. The
empirical results align with theoretical understanding.

In terms of improving the studied strategy, investigations in extensive experiments seem meaningful
for practical implementations, and some non-trivial discoveries together with improvement tricks are
also reported, such as the relationship between quantile estimate errors and adaptation robustness,
the batch size’s influence on several benchmarks, etc.

In this work, the theoretical and empirical parts are connected in an implicit manner. The generaliza-
tion capability is empirically examined from experimental results, and the performance gap between
DR-MAML+ and DR-MAML can be attributed to the difference in generalization bounds. As for the
convergence trait and asymptotic performance, the insight might guide the optimization process in
training large models, such as early stopping criteria design.

A.2
Significance of Theoretical Understandings

As pointed out in [3, 61, 62], large language models are few-shot learners. When a large model, such
as a large decision-making model in the future, comes into practice, fast adaptation robustness can be
a crucial issue as real-world scenarios are indeed risk-sensitive.

This work takes the latest work [1] as an example, and the interest is in the theoretical aspect. Most of
the assumptions in this work are from [1]. The baselines are typical and latest, while the benchmarks

19


---Page Break---
cover diverse downstream tasks. In multimodal few-shot image classification experiments, our
contributed points help guide the development of large models in terms of training and robustness
enhancement. Our investigations also provide insight into robust policy optimization, particularly
when safety is one necessary consideration [70–72].

Shapiro et al.’s book [73] is a comprehensive resource that addresses stochastic modeling and
optimization methods, but it does not explore solution concepts in game theory or define generalization
bounds relevant to meta learning and deep learning. Instead, our work further enriches the stochastic
programming theory in meta learning, connects it to the Stackelberg game, and contributes to tail risk
generalization bounds, convergence rates, asymptotic properties, and so on. Therefore, the solution
concept and the theoretical properties are specific to our meta learning setup, distinctly from the
scope covered by [73].

A.3
Meanings of Indicators and Terms

Illustration of VaRα, CDF and others: Fig. 7 illustrates a typical probability distribution, cumula-
tive distribution, and the resulting mean, VaRα, and CVaRα. Given α ∈[0, 1), VaRα is the α quantile
of the risk distribution. Specially, VaR0.5 coincides with the mean. Upon the definition of VaRα,
CVaRα can be define as CVaRα = Ep(τ)
h
ℓ|ℓ≥VaRα
i
. That is, CVaRα is the expectation of the
risks of the 1 −α tail of the distribution. Relative to the original probability distribution, CVaRα can
be interpreted as a certain distribution shift, which reweighs arbitrary risk exceeding VaRα up to a
coefficient
1
1−α.

Meaning of the asymptotic performance gap: We plot Fig. 8 to display the gap between the CVaRα
value in iterations and that in the convergence. The area difference depicts this gap.

A.4
Computational Complexity

Analyzing computational complexity across all meta-learning methods is inherently challenging
due to the diversity in methodological approaches within the field. Meta-learning encompasses
a wide range of techniques, including gradient-based methods, which rely on iterative updates
to model parameters, and non-parametric methods, which may instead focus on instance-based
learning or kernel-based approaches. Therefore, the space complexity is specific to the meta-learning
method, while this work is agnostic to it. Here, we report the computational complexity for the DR-
MAML+ as O

B(B −α|M|)

while using KDE with the Gaussian kernel, and that of DR-MAML

is O

B(log(B) −α|M|)

.

A.5
Broader Impact & Future Extensions

This paper presents work whose goal is to advance the field of robust meta learning. There are many
potential societal consequences of our work, which we detail as follows.

The fast adaptation robustness is an urgent concern, particularly in large models and risk-sensitive
control. This work provides versatile insights for theoretical analysis and performance improvement
in the presence of tail task risk, and future explorations can be decision-making scenarios, such as
multi-agent policy optimization [74, 75], and computational/memory cost reduction.

B
Pseudo Algorithms

For a better understanding of the game theoretical optimization, we take DR-MAML+ and DR-CNP+
as examples and include the Pseudo Algorithms 1/2 in this section. Particularly, the algorithms
specify the decision-making orders and highlight the use of KDE modules to build task risk value
distributions and estimate the quantile.

20


---Page Break---
Figure 7: Diagram of risk concepts in this work. Here, the x-axis is the task risk value in fast
adaptation given a specific θ. The shadow-lined region illustrates the tail risk with a probability 1 −α
in the probability density. The area of the shadow-lined region after 1 −α normalization corresponds
to the expected tail risk CVaRα.

Figure 8: Illustration of the asymptotic behavior in approximating the equilibrium. Here, the
x-axis is the feasible task risk value in fast adaptation. The dark blue region indicates the histogram of
the task risk values in the local Stackelberg equilibrium (q∗, θ∗). The shallow blue region describes the
histogram of the task risk values at some iterated point (qT −1, θmeta
T
). The sets T1 and T2 respectively
collect the tasks resulting the opposite order.

C
Expressions, Theorems & Proofs

C.1
Characterization of Optimization Processes

Without loss of generality, we can also express the process of solving the studied Stackelberg game
as:

max
q∈Qα F(q, θ∗(q))
s.t. θ∗(q) = arg min
θ∈Θ F(q, θ)
(13a: Leader’s Decision-Making)

min
θ∈Θ F(q, θ),
(13b: Follower’s Decision-Making)

where the optimization w.r.t. (q, θ) is the computation of the best responses for two adversarial
players. As a bi-level optimization in Eq. (14a)/(14b), the exact solution is intractable to obtain in a
theoretical sense, and the two-stage distributionallly robust optimization is a heuristic approach.

Meaning of the obtained equilibrium. Here, we can interpret the obtained solution (q∗, θ∗) from
solving Eq. (7) as follows. Given the follower’s decision θ∗and the induced task risk distribution
Fℓ(l; θ∗), the leader cannot further raise a proposal of a task subset with a probability 1 −α to
degradde the tailed expected performance. And this explains the meaning of robust fast adaptation
solution w.r.t. the tail task risk.

21


---Page Break---
Algorithm 1: Meta-training DR-MAML+ as A Stackelberg Game
Input
:Task distribution p(τ); Confidence level α; Task batch size B; Learning rates: λ1 and
λ2.
Output :Meta-trained model parameter θ.
Randomly initialize the model parameter θ;
while not converged do

Sample a batch of tasks {τi}B
i=1 ∼p(τ);
# The Leader Player’s Decision-Making
for i = 1 to B do

// inner loop via gradient descent as the fast adaptation

Evaluate the gradient: ∇θℓ(DS
τi; θ) in Eq. (5);
Perform task-specific gradient updates:
θi ←θ −λ1∇θℓ(DS
τi; θ);
end
// model the task risk distribution and estimate the quantile

Evaluate performance LB = {ℓ(DQ
τi; θi)}B
i=1;
Estimate VaRα[ℓ(T , θ)] and set ξ = ˆξα in Eq. (5) with kernel density estimators;
Screen the subset L ˆ
B = {ℓ(DQ
ˆτi; θi)}K
i=1 with ˆξα for meta initialization updates;
# The Follower Player’s Decision-Making
Execute outer loop via gradient descent to increase adaptation robustness:
θ ←θ −λ2∇θ
PK
i=1 ℓ(DQ
ˆτi; θi) in Eq. (5);
end

Algorithm 2: Meta Training DR-CNP+ as A Stackelberg Game
Input
:Task distribution p(τ); Confidence level α; Task batch size B; Learning rate λ.
Output :Meta-trained model parameter θ.
Randomly initialize the model parameter θ;
while not converged do

Sample a batch of tasks {τi}B
i=1 ∼p(τ);
# The Leader Player’s Decision-Making
// model the task risk distribution and estimate the quantile

Evaluate performance LB = {ℓ(DQ
τi; z, θi)}B
i=1;
Estimate VaRα[ℓ(T , θ)] ≈ˆξα with kernel density estimators;
Screen the subset L ˆ
B = {ℓ(DQ
ˆτi; z, θ)}K
i=1 with ˆξα for meta initialization updates;
# The Follower Player’s Decision-Making
Execute gradient descent to increase adaptation robustness:
θ ←θ −λ∇θ
PK
i=1 ℓ(DQ
ˆτi; z, θ);
end

C.2
Assumptions

We list all of the assumptions mentioned in this work. These assumptions further serve the demon-
stration of propositions and theorems in the main paper.

Assumption 1. To proceed, we retain most assumptions from [1] for theoretical analysis, including:

1. The meta risk function ℓ(DQ
τ , DS
τ ; θ) is βτ-Lipschitz continuous w.r.t. θ;
2. The cumulative distribution Fℓ(l; θ) is βℓ-Lipschitz continuous w.r.t. l, and the normalized
density function pα(τ; θ) is βθ-Lipschitz continuous w.r.t. θ;
3. For arbitrary valid θ ∈Θ and corresponding pα(τ; θ), ℓ(DQ
τ , DS
τ ; θ) is bounded:
supτ∈Ωα,τ ℓ(DQ
τi, DS
τi; θ) ≤Lmax.

Assumption 2. The implicit function h(·) is βh-Lipschitz continuous w.r.t. θ ∈Θ, and ∇θF(q, θ) is
βq-Lipschitz continuous w.r.t. q ∈Qα.

22


---Page Break---
C.3
Proof of Proposition 1

Proposition 1. The uncertainty set Qα is convex and compact in terms of probability measures.

Proof: We firstly focus on the convexity of Qα. For any {q1 := q1(τ), q2 := q2(τ)} ∈Qα, we
partition these two task spaces with non-zero sampling probability mass respectively as T1 ∪TC and
T2 ∪TC. As displayed in Fig. 9, TC denotes the shared subset task between q1 and q2. Below we
show that λ1q1 + λ2q2 ∈Qα with λ1 + λ2 = 1. This is true because

Z

τ∈Tλ1q1+λ2q2
p(τ)dτ
(15a)

=
Z

τ /∈Tq1∪Tq2
p(τ)dτ +
Z

τ∈TC


λ1p(τ) + λ2p(τ)

dτ +
Z

τ∈T1
λ1p(τ)dτ +
Z

τ∈T2
λ2p(τ)dτ

(15b)

= 0 + λ1

Z

τ∈TC
p(τ)dτ +
Z

τ∈T1
p(τ)dτ

+ λ2

Z

τ∈TC
p(τ)dτ +
Z

τ∈T2
p(τ)dτ

(15c)

= λ1

Z

τ∈Tq1
p(τ)dτ + λ2

Z

τ∈Tq2
p(τ)dτ
(15d)

= 1 −α.
(15e)

We next demonstrate the compactness of Qα. The distance between two distributions ∀{q1, q2} ∈Qα
can be defined as:

dQα(q1, q2) :=
Z

τ∈T

q1(τ) −q2(τ)
dτ.

Since L1 space is a Banach space, the compactness is equivalent to the closedness and Boundedness
of Qα. Considering a sequence {qn(τ) ∈Qα} with the resulting limitation is q∗(τ), following the
Controlled Convergence Theorem [76], we know that

lim
n→∞

Z

τ∈T
pn(τ) −p∗(τ)dτ ≤lim
n→∞

Z

τ∈T

pn(τ) −p∗(τ)
dτ = lim
n→∞dQα(qn, q∗) = 0.
(16)

Due to the symmetry of the distance, we can have limn→∞
R

τ∈T p∗(τ) −pn(τ)dτ ≤0. Thus,

Z

τ∈T
p∗(τ)dτ = lim
n→∞

Z

τ∈T
pn(τ)dτ = 1 −α.

That is, p∗(τ) ∈Qα, indicating that Qα is a closed set. As the boundedness is clear in the studied
problem, this completes the proof of Proposition 1.
■

C.4
Proof of Proposition 2

Proposition 2 (Existence of Equilibrium) Given the Assumption 1, there always exists the global
Stackelberg equilibrium as the Definition 1 for the studied SG.

Proof: Note that Θ is compact as a subspace of the Euclidean space. And it is trivial to see that
F(q, θ) := Eq
h
ℓ(DQ
τ , DS
τ ; θ)
i
is continuous w.r.t. θ ∈Θ as ℓsatisfies the βτ-Lipschitz continuity in
the Assumption 1.

Here we need to show the continuity of F(q, θ) w.r.t. the collection of probability measures or
probability functions Qα. To this end, with ∀θ ∈Θ fixed, We consider two metric spaces (Qα, dQα)
and (L, RL). The map of our interest is g(q) = F(q, ·) : Qα 7→L ⊆R+.

23


---Page Break---
Figure 9: Partition of the task subspace. Here we take two probability measure {q1, q2} ∈Qα
for illustration. T1 ∪TC and T2 ∪TC defines the corresponding task subspaces for q1 and q2 with
non-zero probability mass in the whole space T .

Naturally, we can have the following inequality:
g(q1) −g(q2)
 =
Eq1
h
ℓ(DQ
τ , DS
τ ; θ)
i
−Eq2
h
ℓ(DQ
τ , DS
τ ; θ)
i
(17a)

≤

Z

τ∈TC
[q1(τ) −q2(τ)]ℓ(DQ
τ , DS
τ ; θ)dτ

(17b)

+

Z

τ∈T1
q1(τ)ℓ(DQ
τ , DS
τ ; θ)dτ −
Z

τ∈T2
q2(τ)ℓ(DQ
τ , DS
τ ; θ)dτ

(17c)

≤
Z

τ∈TC

q1(τ) −q2(τ)
ℓ(DQ
τ , DS
τ ; θ)dτ
(17d)

+
Z

τ∈T1

q1(τ) −q2(τ)
ℓ(DQ
τ , DS
τ ; θ)dτ +
Z

τ∈T2

q1(τ) −q2(τ)
ℓ(DQ
τ , DS
τ ; θ)dτ
(17e)

≤3Lmax

Z

τ∈T

q1(τ) −q2(τ)
dτ = 3LmaxdQα(q1, q2),
(17f)

which implies 3Lmax-Lipschitz continuity of g(q) w.r.t. ∀q ∈Qα.

According to the Remark in [77], there always exists the global Stackelberg equilibrium as the
Definition 1 when Qα × Θ is compact and F(q, θ) is continuous. This completes the proof of
Proposition 2.
■

C.5
Proof of Quasi-concavity for F(q, θ) w.r.t. q

It can be validated that F(q, θ) is a quasi-concave function w.r.t. q, meaning that for any positive
number l ∈R+, the set {q|q ∈Qα, F(q, θ) > l} is convex in Qα.

Proof: According to the conventional definition (i.e., the superlevel set is convex [78]), for all
λ1 + λ2 = 1, q1, q2 ∈{q|F(q, θ) > l}, we can have

F(λ1q1 + λ2q2, θ) = Eλ1q1+λ2q2
h
ℓ(DQ
τ , DS
τ ; θ)
i
(18a)

= λ1Eq1
h
ℓ(DQ
τ , DS
τ ; θ)
i
+ λ2Eq2
h
ℓ(DQ
τ , DS
τ ; θ)
i
(18b)

= λ1F(q1, θ) + λ2F(q2, θ)
(18c)
> l.
(18d)

Thus, λ1q1 + λ2q2 ∈{q|F(q, θ) > l} and the superlevel set is convex, implying that F(q, θ) is
quasi-concave w.r.t. q.
■

C.6
Proof of Theorem 4.1

Theorem 4.1 (Convergence Rate for the Second Player) Let the iteration sequence in opti-
mization be:
· · ·
7→
{qt−1, θt}
7→
{qt, θt+1}
7→
· · ·
7→
{q∗, θ∗}, with the converged
equilibirum (q∗, θ∗).
Under the Assumption 2 and suppose that ||I −λ∇2
θθF(q∗, θ∗)||2 <

24


---Page Break---
1 −λβqβh, we can have limt→∞
||θt+1−θ∗||2

||θt−θ∗||2
≤1, and the iteration converges with the rate
 
||I −λ∇2
θθF(q∗, θ∗)||2 + λβqβh

.

Proof: Let the resulting stationary point be [q∗, θ∗], we denote the difference terms by ˆq = q −q∗
and ˆθ = θ −θ∗. Then, according to the optimization step, we can have the following equations:

θt+1 = θt −λ∇θF(qt; θt) =⇒ˆθt+1 = ˆθt −λ∇θF(qt; θt).
(19)

Now we perform the first-order Taylor expansion of the θ related function ∇θF(qt; θ) around θ∗and
can derive:

∇θF(qt; θ) = ∇θF(qt; θ∗) + ∇2
θθF(qt; θ∗)(θ −θ∗) + O(||θ −θ∗||)
(20a)

∇θF(qt; θt) ≃∇θF(qt; θ∗) + ∇2
θθF(qt; θ∗)(θt −θ∗).
(20b)

Then we have the following result with the help of Assumption 2:

∥∇θF(qt; θ∗)∥2 = ∥∇θF(qt; θ∗) −∇θF(q∗; θ∗)∥2
(21a)
= ∥∇θF(h(θt); θ∗) −∇θF(h(θ∗); θ∗)∥2
(21b)
≤βqdQα(h(θt), h(θ∗))
(21c)
≤βqβh∥θt −θ∗∥2.
(21d)

With Eq. (19), Eq. (20) and Eq. (21), we can derive the equation that:

ˆθt+1 = ˆθt −λ∇θF(qt; θt)
(22a)

= ˆθt −λ
h
∇θF(qt; θ∗) + ∇2
θθF(qt; θ∗)ˆθt
i
(22b)

=
h
I −λ∇2
θθF(qt; θ∗)
i
ˆθt −λ∇θF(qt; θ∗)
(22c)

=⇒||ˆθt+1||2 ≤∥I −λ∇2
θθF(qt; θ∗)∥2||ˆθt||2 + λ∥∇θF(qt; θ∗)∥2
(22d)

≤
 
∥I −λ∇2
θθF(qt; θ∗)∥2 + λβqβh

||∥ˆθt∥2
(22e)

Thus, when ∥I −λ∇2
θθF(q∗; θ∗)∥2 < 1 −λβqβh, we have

lim
t→∞
||ˆθt+1||2

||ˆθt||2
≤lim
t→∞∥I −λ∇2
θθF(qt; θ∗)∥2 + λβqβh
(23a)

= ∥I −λ∇2
θθF(q∗; θ∗)∥2 + λβqβh
(23b)
< 1.
(23c)

This completes the proof of Theorem 4.1.
■

C.7
Proof of Theorem 4.2

Theorem 4.2 (Asymptotics in the Tail Risk Cases) Under the Assumption 1 and given a batch of
tasks {τi}B
i=1, we can have

CVaRα(θmeta
T
) −CVaRα(θ∗) ≤βτ∥θmeta
T
−θ∗∥+ VaR∗
α
1 −α


P(T1) −P(T2)

,
(24)

where T1 = {τ : ℓ∗< VaR∗
α, ℓmeta ≥VaRmeta
α
}, T2 = {τ : ℓ∗≥VaR∗
α, ℓmeta < VaRmeta
α
}.

25


---Page Break---
Proof. Given a batch of tasks {τ1, · · · , τB} and according to the definition of CVaRα, we have

CVaRα(θmeta
T
) −CVaRα(θ∗)
(25a)

=
1
1 −α

Z

{τ:ℓmeta≥VaRmeta
α
}
ℓmetap(τ)dτ −
1
1 −α

Z

{τ:ℓ∗≥VaR∗
α}
ℓ∗p(τ)dτ
(25b)

=
Z

τ
ℓmetapα(τ; θmeta
T
)dτ −
Z

τ
ℓ∗pα(τ; θ∗)dτ
(25c)

=
Z

τ


ℓmetapα(τ; θmeta
T
) −ℓ∗pα(τ; θmeta
T
)

dτ +
Z

τ


ℓ∗pα(τ; θmeta
T
) −ℓ∗pα(τ; θ∗)

dτ
(25d)

=
Z

τ


ℓmeta −ℓ∗
pα(τ; θmeta
T
)dτ +
Z

τ
ℓ∗
pα(τ; θmeta
T
) −pα(τ; θ∗)

dτ
(25e)

≤βτ∥θmeta
T
−θ∗∥+
Z

T1∪T2∪T3∪T4
ℓ∗
pα(τ; θmeta
T
) −pα(τ; θ∗)

dτ
(25f)

= βτ∥θmeta
T
−θ∗∥+
Z

T1∪T2
ℓ∗
pα(τ; θmeta
T
) −pα(τ; θ∗)

dτ
(25g)

= βτ∥θmeta
T
−θ∗∥+
Z

T1
ℓ∗pα(τ; θmeta
T
)dτ −
Z

T2
ℓ∗pα(τ; θ∗)dτ
(25h)

= βτ∥θmeta
T
−θ∗∥+
1
1 −α

Z

T1
ℓ∗p(τ)dτ −
1
1 −α

Z

T2
ℓ∗p(τ)dτ
(25i)

≤βτ∥θmeta
T
−θ∗∥+ VaR∗
α
1 −α

Z

T1
p(τ)dτ −VaR∗
α
1 −α

Z

T2
p(τ)dτ
(25j)

= βτ∥θmeta
T
−θ∗∥+ VaR∗
α
1 −αP(T1) −VaR∗
α
1 −αP(T2).
(25k)

In inequality (25f), T1 = {τ : ℓ∗< VaR∗
α, ℓmeta ≥VaRmeta
α
}, T2 = {τ : ℓ∗≥VaR∗
α, ℓmeta <
VaRmeta
α
}, T3 = {τ : ℓ∗< VaR∗
α, ℓmeta < VaRmeta
α
}, T4 = {τ : ℓ∗≥VaR∗
α, ℓmeta ≥VaRmeta
α
}.
Moreover, this inequality holds due to the βτ−Lipschitz continuous of ℓ(Dτ; θ). In Eq. (25g),
pα(τ; θmeta
T
) = pα(τ; θ∗) = 0 when τ ∈T3, and pα(τ; θmeta
T
) = pα(τ; θ∗) = p(τ)

1−α when τ ∈T4.
Thus, we complete the proof of Theorem 4.2.
■

C.8
Proof of Theorem 4.3

Theorem 4.3 (Generalization Bound in the Tail Risk Cases) Given a collection of task samples
{τi}B
i=1 and corresponding meta datasets, we can derive the following generalization bound in the
presence of tail risk:

R(θ∗) ≤bR(θ∗) +

s

2
 
α
1−αL2max + Vτi∼pα(τ)

ℓ(DQ
τi, DSτi; θ∗)

ln
  1

ϵ


B

+
1
3(1 −α)
Lmax

B


2 ln
1

ϵ


+ 3αB

,

(26)

where the inequality holds with probability at least 1 −ϵ and ϵ ∈(0, 1), V[·] denotes the variance
operation, and Lmax is from the Assumption 1.

Proof. R(θ∗) −bR(θ∗) can be decomposed to two parts, i.e.,

R(θ∗) −bR(θ∗) =

R(θ∗) −bRw(θ∗)

+

bRw(θ∗) −bR(θ∗)

.
(27)

For the first part (i.e., R(θ∗) −bRw(θ∗)), we will adopt the Bernstein’s inequality to provide an
upper bound. Regarding pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗) −R(θ∗) as a random variable with respect to τi and
according to Assumption 1, we know that

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗) −R(θ∗) ≤
1
1 −αℓ(DQ
τi, DS
τi; θ∗) −R(θ∗) ≤
1
1 −αLmax.
(28)

26


---Page Break---
Thus, following Bernstein’s inequality [79], we know that

P

 
 1

B

B
X

i=1

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗) −R(θ∗)
 ≥ξ

!

(29a)

= P
 bRw(θ∗) −R(θ∗)
 ≥ξ

(29b)

≤exp



−
Bξ2

2Vτi
h
pα(τi)

p(τi) ℓ(DQ
τi, DSτi; θ∗) −R(θ∗)
i
+ 2

3
1
1−αLmaxξ



,
(29c)

where

Vτi

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗) −R(θ∗)

(30a)

= Vτi

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗)

(30b)

= Eτi

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗)
2
−

Eτi

pα(τi)

p(τi) ℓ(DQ
τi, DS
τi; θ∗)
2
(30c)

=
Z

τ

pα(τ)

p(τ) ℓ(DQ
τ , DS
τ ; θ∗)
2
p(τ)dτ −
Z

τ

pα(τ)

p(τ) ℓ(DQ
τ , DS
τ ; θ∗)p(τ)dτ
2
(30d)

=
Z

τ

pα(τ)

p(τ) ℓ(DQ
τ , DS
τ ; θ∗)2pα(τ)dτ −
Z

τ
ℓ(DQ
τ , DS
τ ; θ∗)pα(τ)dτ
2
(30e)

=
1
1 −α

Z

τ
ℓ(DQ
τ , DS
τ ; θ∗)2pα(τ)dτ −
Z

τ
ℓ(DQ
τ , DS
τ ; θ∗)pα(τ)dτ
2
(30f)

≤

1
1 −α −1
 Z

τ
ℓ(DQ
τ , DS
τ ; θ∗)2pα(τ)dτ + Vτ∼pα(τ)

ℓ(DQ
τ , DS
τ ; θ∗)

(30g)

:=
α
1 −αL2
max + Vτ∼pα(τ).
(30h)

Setting ϵ to match the upper bound in inequality (29c) shows that with probability at least 1 −ϵ, the
following bound holds:

 bRw(θ∗) −R(θ∗)
 ≤

s

2(
α
1−αL2max + Vτ∼pα(τ)) ln
  1

ϵ


B
+ 2Lmax ln
  1

ϵ


3(1 −α)B .
(31)

For the second part (i.e., bRw(θ∗) −bR(θ∗)), we have

bRw(θ∗) −bR(θ∗) = 1

B

B
X

i=1

pα(τi)

p(τi) −δ(τi)

ℓ(DQ
τi, DS
τi; θ∗)
(32a)

≤Lmax

B

B
X

i=1

pα(τi)

p(τi) −δ(τi)

(32b)

= Lmax

B

B
X

i=1

pα(τi)

p(τi) −1

δ(τi)
(32c)

=
α
1 −α
Lmax

B

B
X

i=1
δ(τi).
(32d)

27


---Page Break---
In summary, we can obtain an upper bound of R(θ∗) −bR(θ∗). That is,

R(θ∗) −bR(θ∗) ≤
 bRw(θ∗) −R(θ∗)
 + bRw(θ∗) −bR(θ∗)

≤

s

2(
α
1−αL2max + Vτ∼pα(τ)) ln
  1

ϵ


B
+ 2Lmax ln
  1

ϵ


3(1 −α)B
+
α
1 −α
Lmax

B

B
X

i=1
δ(τi)

≤

s

2(
α
1−αL2max + Vτ∼pα(τ)) ln
  1

ϵ


B
+
1
3(1 −α)
Lmax

B


2 ln
1

ϵ


+ 3αB

.

This completes the proof of Theorem 4.3.
■

C.9
Proof of Theorem 4.4

Theorem 4.4 Let F −1
ℓ-KDE(α; θ) = VaRKDE
α
[ℓ(T , θ)] and F −1
ℓ
(α; θ) = VaRα[ℓ(T , θ)]. Suppose that
K(x) is lower bounded by a constant, ∀x. For any ϵ > 0, with probability at least 1 −ϵ, we can have
the following bound:

sup
θ∈Θ

 
F −1
ℓ-KDE(α; θ) −F −1
ℓ
(α; θ)

≤O

hℓ
√B ∗log B


.
(33)

Proof. For any constant M, we firstly notice that

Pτ1,··· ,τB


sup
θ∈Θ

 
F −1
ℓ-KDE(α; θ) −F −1
ℓ
(α; θ)

≤M

≥1 −ϵ
(34a)

⇔Pτ1,··· ,τB


sup
θ∈Θ

 
F −1
ℓ-KDE(α; θ) −F −1
ℓ
(α; θ)

≥M

≤ϵ
(34b)

⇔Pτ1,··· ,τB


sup
θ∈Θ

 
Fℓ(t −M; θ) −Fℓ-KDE(t; θ)

≥0

≤ϵ,
t = F −1
ℓ-KDE(α; θ).
(34c)

For any θ and t, we have

Pτ1,··· ,τB
 
Fℓ(t −M; θ) −Fℓ-KDE(t; θ) ≥0

≤ϵ
(35a)

⇔Pτ1,··· ,τB
 
Fℓ(t −M; θ) −Fℓ-KDE(t −M; θ) + Fℓ-KDE(t −M; θ) −Fℓ-KDE(t; θ) ≥0

≤ϵ
(35b)

⇔Pτ1,··· ,τB
 
Fℓ(t −M; θ) −Fℓ-KDE(t −M; θ) ≥Fℓ-KDE(t; θ) −Fℓ-KDE(t −M; θ)

≤ϵ
(35c)

⇔Pτ1,··· ,τB
 
Fℓ(t; θ) −Fℓ-KDE(t; θ) ≥Fℓ-KDE(t + M; θ) −Fℓ-KDE(t; θ)

≤ϵ
(35d)

⇔Pτ1,··· ,τB
 
Fℓ(t; θ) −Fℓ-KDE(t; θ) ≥M

Bhℓ


B
X

i=1
K
 t −ℓ(DQ
τi, DS
τi; θ)
hℓ


+ o(M)

≤ϵ (35e)

⇐Pτ1,··· ,τB
 
Fℓ(t; θ) −Fℓ-KDE(t; θ) ≥MKmin

hℓ


≤ϵ,
(35f)

where Kmin is the lower bound of the kernel function K(x), i.e., K(x) ≥Kmin, ∀x.

According to Theorem 3 of [80], we know that for any ϵ > 0, with probability at least 1 −ϵ, the
following inequality holds:

Pτ1,··· ,τB

 

sup
θ∈Θ,t≥0

 
Fℓ(t; θ) −Fℓ-KDE(t; θ)

≥
C
√B ∗log B

!

≤ϵ,
(36)

where C is a constant. Let M =
hℓC
Kmin
√B∗log B. Thus, the Eq. (35f) holds and we complete the proof
of Theorem 4.4.
■

C.10
Additional Theorem

To gain more theoretical insights into a popular meta-learning method—MAML [25], we provide the
following Theorem C.1. Before proceeding, we introduce some notations. During meta-training, a

28


---Page Break---
finite number of task instances are observed by first sampling a task from the distribution p(τ). Each
task Dτi comprises a collection of mi data points {(DS
i,j, DQ
i,j)}mi
j=1, which are distributed over Z
with each data point drawn from a distribution Di. For some risk ℓ, define the family of functions
FZ := {ℓ(θ −λ∇θℓ(DS
τi; θ), DQ
τi) : θ ∈Θ}. For each task Dτi, the Rademacher complexity of F
on mi samples is

Ri
mi(FZ) = E(DS
i,j,DQ
i,j)∼(Di)miEϵ



sup
θ∈Θ

1
mi

mi
X

j=1
ϵjℓ(θ −λ∇θℓ(DS
i,j; θ), DQ
i,j)



,
(37)

where the ϵj’s are Rademacher random variables. Let Fi(θ) = EDiℓ(θ −λ∇θℓ(DS
i,j; θ), DQ
i,j),
ˆFi(θ) =
1
mi
Pmi
j=1 ℓ(θ −λ∇θℓ(DS
i,j; θ), DQ
i,j). Denote by θ∗the optimal model parameter under the
two-stage algorithm. Theorem C.1 provides generalization of the algorithm to new tasks.
Theorem C.1 (Generalization Bound for MAML in the Tail Risk Cases). For a new task τB+1 with
distribution DB+1, if DB+1 = PB
i=1 aiDi, then with probability at least 1 −δ for any δ > 0, we can
have

FB+1(θ∗) ≤max
i
ˆFi(θ∗) +

B
X

i=1



2aiRi
mi(FZ) + ai

s

log (B/δ)

2mi



.
(38)

Proof. The proof consists of two parts. We first explore the generalization to new instances of
previously-seen tasks. Then we solve the generalization to new tasks.

Step 1. For any sample set A = {(DS
i,j, DQ
i,j)}mi
j=1, define Φ(A) = supℓ∈FZ Fi(θ) −ˆFi(θ). Let
A and A′ := {((DS
i,j)′, (DQ
i,j)′)}mi
j=1 be two samples that differ by exactly one point. According
to the fact supx f(x) −supx g(x) ≤supx(f(x) −g(x)), we know that Φ(A′) −Φ(A) ≤
1
mi
due to the difference in exactly one point. Similarly, we can obtain Φ(A) −Φ(A′) ≤
1
m, thus
Φ(A) −Φ(A′)
 ≤1

m. Following McDiarmid’s inequality, for any δ > 0, with probability at least
1 −δ

2, we have

Φ(A) ≤EA[Φ(A)] +

s

log(2/δ)

2mi
.
(39)

We next bound the expectation of the right-hand side of inequality (39) as follows:

EA[Φ(A)] = EA
h
sup
ℓ∈FZ
Fi(θ) −ˆFi(θ)
i

= EA
h
sup
ℓ∈FZ
EA′
h
EA′(Fi(θ)) −ˆFi(θ)
ii
(40)

≤EA,A′
h
sup
ℓ∈FZ
EA′(Fi(θ)) −ˆFi(θ)
i
(41)

= EA,A′
h
sup
ℓ∈FZ

1
mi

mi
X

j=1

 
ℓ(DS
i,j, DQ
i,j) −ℓ((DS
i,j)′, (DQ
i,j)′)
i
(42)

= EA,A′,ϵ
h
sup
ℓ∈FZ

1
mi

mi
X

j=1
ϵj
 
ℓ(DS
i,j, DQ
i,j) −ℓ((DS
i,j)′, (DQ
i,j)′)
i

≤EA,ϵ
h
sup
ℓ∈FZ

1
mi

mi
X

j=1
ϵjℓ(DS
i,j, DQ
i,j)
i
+ EA′,ϵ
h
sup
ℓ∈FZ

1
mi

mi
X

j=1
ϵjℓ((DS
i,j)′, (DQ
i,j)′)
i

= 2EA,ϵ
h
sup
ℓ∈FZ

1
mi

mi
X

j=1
ϵjℓ(DS
i,j, DQ
i,j)
i

= 2Rmi(FZ).

Eq. (40) uses the law of total expectation. Inequality (41) holds by Jensen’s inequality and the
convexity of the supremum function. In Eq. (42), ℓ(DS
i,j, DQ
i,j) := ℓ(θ −λ∇θℓ(DS
i,j; θ), DQ
i,j),

29


---Page Break---
where (DS
i,j, DQ
i,j) ∈A. Following inequality (39), we can know that

Fi(θ) ≤ˆFi(θ) + 2Rmi(FZ) +

s

log(2/δ)

2mi
.
(43)

Step 2. Since the new distribution DB+1 is the convex combination of Di, ∀i = 1, · · · , B, we have
FB+1(θ) = PB
i=1 aiFi(θ). Accordingly, with probability at least 1 −δ over the choice of samples
used to compute ˆF(θ),

FB+1(θ∗) =

B
X

i=1
aiFi(θ∗) ≤

B
X

i=1



ai ˆFi(θ∗) + 2aiRmi(FZ) + ai

s

log(2/δ)

2mi



,
(44)

which yields that

FB+1(θ∗) ≤max
i
ˆFi(θ∗) +

B
X

i=1



2aiRi
mi(FZ) + ai

s

log (2/δ)

2mi



.
(45)

In summary, the two steps complete the proof of Theorem C.1.
■

D
Implementation Details

D.1
Benchmark Details & Neural Architectures & Opensource Codes

Here, we illustrate all meta learning benchmark purposes in Fig. 10, which includes sinusoid
regression, pendulum system identification, few-shot image classification, and meta reinforcement
learning. We no longer run experiments on the Omniglot dataset, as most baselines can achieve SOTA
performance and cannot tell the difference well from the openreview of [1].

（a）Sinusoid Regression
（d）Continuous Control

1
2

support dataset

class:

class:
?
?
?

query dataset

（c）Few-Shot Image Classification
（b）System Identification

Figure 10: Typical meta learning benchmarks in evaluation.

Sinusoid regression: In [1, 42], a lot of easy tasks and limited challenging tasks are sampled for
meta-training, with the tasks from the whole space employed in evaluation. The default range of the
phase parameter is B ∈[0, π], while those of the amplitude are A ∈[0.1, 1.05] for easy tasks and
A ∈[4.95, 5.0] for challenging tasks. Generally, sinusoid functions with larger amplitudes are hard
to adapt from a few support data points. The mean square error works as the risk function to measure
the gap between the predicted value f(x) and the actual value. We set the task batch 50 for 5-shot
and 25 for 10-shot, and the maximum iteration number is 70000. We refer the reader to TR-MAML
and DR-MAML for all of the setups.

We retain the neural architectures [42, 1] in for all MAML like methods. In detail, all methods
take a multilayer perceptron with two hidden layers and 40 ReLU activation units in each layer.
The inner loop is achieved via one stochastic gradient descent step. As for CNP like methods,
please refer to the vanilla set-up in [15] (The Github link is attached here: https://github.com/
google-deepmind/neural-processes).

As the task space is hugh, there is no way to exactly estimate the risk quantile. Hence, the Oracle
quantile in Fig. 5 is roughly computed from the sampled 100 tasks given the pretrained DR-MAML+.
The rationale behind this operation is that increasing the population number in statistics reduces the
quantile estimate bias.

System identification: The pendulum system is a classical environment in the OpenAI gym (en-
vironment details are: https://github.com/openai/gym/blob/master/gym/envs/classic_

30


---Page Break---
control/pendulum.py), and it is an actuated joint with one fixed end. The goal of system iden-
tification for the pendulum system is to predict the state transition given arbitrary actions with
several randomly collected transitions as the support dataset. The observation is a tuple in the form
(cos θ, sin θ, θ′), where θ ∈[π, π]. The action is in the range a ∈[−2.0, 2.0] and the torque is applied
to the pendulum body. The mass m and the length l of the pendulum follows a uniform distribution
(m, l) ∼U([0.4, 1.6] × [0.4, 1.6]), sampled variables configure a Markov decision process as the task.
In each batch, there are 16 tasks, and each task comprises 200 data points. Specifically, 10 few-shot
data points are randomly sampled to enable system identification per task, denoted as 10-shot. For
20-shot cases, the number of data points in support dataset is 20. And the maximum iteration
number is 5000.

For all MAML-like methods, the neural architecture used here is a multilayer perceptron with three
hidden layers of 128 hidden units each and the activation function is ReLU. The learning rate for
both the inner and outer loops is set at 1e-4.

Few-shot image classification: The few-shot image classification is mostly described as an N-way
K-shot classification, where N classes with K-labeled instances for each are considered. The dataset
is organized in the same manner as that in [81, 42, 1]: These include 64 classes for meta-training,
with the rest 36 classes for meta-testing. We generate each task in the way: 8 meta-training tasks from
the class {6, 7, 7, 8, 8, 9, 9, 10} are randomly generated from 64 meta-training classes; the remaining
classes are organized similarly. As a result, each task is constructed from sampling one image from
five classes, corresponding to a 5-way 1-shot problem. The task batch is set 4 with a maximum
number of iterations of 60000 in meta-training.

For all MAML-like methods, the neural architecture used here is a four-layer convolutional neural
network for the mini-ImageNet datasets. The inner loop is achieved via one stochastic gradient
descent step. We refer the reader to TR-MAML and DR-MAML for all of the setups (The Github
link is attached here https://github.com/lgcollins/tr-maml).

Meta reinforcement learning: 2D Navigation is a classical meta reinforcement learning benchmark
where efficient explorations matter. The task in 2D Navigation is to guide the point robot to take
move actions for a purpose of reaching a specific goal location from the step-wise reward. The reward
the agent receives from the environment is based on the distance to the goal, and 20 episodes work as
the support dataset for navigation fast adaptation. In terms of the task distribution, we sample tasks
from a uniform distribution U([−0.5, 0.5] × [−0.5, 0.5]) over goal locations.

As for the neural architecture for policy network set-ups, we refer the reader to vanilla MAML (Github
link is attached here https://github.com/tristandeleu/pytorch-maml-rl) and CAVIA
(The Github link is attached here https://github.com/lmzintgraf/cavia/tree/master/rl).
And trust region policy optimization works for policy optimization.

Table 4: Computational and memory cost in MaPLe relevant experiments.

Method
MaPLe
DR-MaPLe
DR-MaPLe+ (Ours)

Implementation Time
2.1 h
+1.7 h
+1.7 h
Memory Usage
41.57 G
+36.84G
+36.84G

Few-Shot Image Classification with MaPLe [69]: The stochastic gradient descent is the default
optimizer with the learning rate 0.0035, and A6000 GPUs work for computations. We examine tail
task risk minimization effectiveness on three large datasets. The class number split setup in datasets
(class number to train/validate/test) is TieredImageNet (351/97/160) [82], ImagenetA (128/32/40)
[83], and ImagenetSketch (640/160/200) [84]. Table 4 reports the overall training time and memory,
where the vanilla MaPLe serves as the anchor point, and + means additional costs from the two-stage
operation. For details of experimental implementations and setups, feel free to access our code at
https://github.com/lvyiqin/DRMAML.

D.2
Modules in Python

This subsection includes the impelementation of KDE for the studied strategy. Here, the example of
the hinge loss is illustrated as follows.

31


---Page Break---
1 import
numpy as np

2 import
torch

3 from
scipy.stats
import
gaussian_kde

4 from
scipy.optimize
import
brentq

5

6 def loss(batch_loss , confidence_level ):

7
# estimate
the
VaR_alpha
according to kernel
density
estimator

8
kde = gaussian_kde(batch_loss)

9
try:

10
target_func = lambda x: kde. integrate_box_1d (-np.inf , x) -
confidence_level

11
VaR_alpha = brentq(target_func , np.min(batch_loss), np.max(
batch_loss))

12
except
ValueError:

13
x = np.linspace(np.min(batch_loss), np.max(batch_loss), 1000)

14
pdf = kde.evaluate(x)

15
cdf = np.cumsum(pdf) / np.sum(pdf)

16
index = np.argmax(cdf
>= confidence_level )

17
VaR_alpha = x[index]

18

19
# calculate
the meta loss

20
tail_loss = [i - VaR_alpha if (i - VaR_alpha) > 0 else
torch.
tensor (0.).cuda () for i in batch_loss]

21
new_batch_loss = torch.stack(tail_loss).mean ()

22
factor = 1 / (1 - confidence_level )

23
loss_meta = VaR_alpha + factor * new_batch_loss

24
return
loss_meta

Listing 1: The calculation process of CVaRα objective.

E
Additional Experimental Results

Due to the page limit in the main paper, we include additional experiments and corresponding results
in this section.

E.1
Evaluation with Other Robust Meta Learners

In addition to MAML, we apply a similar modification to CNP, which results in TR-CNP, DRO-CNP,
DR-CNP, and DR-CNP+ (DR-CNP with KDE for VaRα estimates). We report the meta testing results
on sinusoid regression and pendulum system identification benchmarks.

As illustrated in Table 5/6, all methods achieve comparable average performance in sinusoid and
pendulum system identification. Regarding CVaRα, DR-CNP’s improvement is relatively marginal
over others except DR-CNP+. Compared to MAML, CNP seems more sensitive to quantile estimate
accuracies when meeting with the studied strategies.

Table 5: MSEs for Sinusoid 5-shot with reported standard deviations (5 runs). With α = 0.7,
the best results are in bold.

Method
Average
Worst
CVaRα
CNP [15]
0.09±0.00
2.71±0.54
0.24±0.01
TR-CNP [42]
0.10±0.01
1.51±0.30
0.22±0.03
DRO-CNP [66]
0.09±0.02
2.54±1.81
0.21±0.05
DR-CNP [1]
0.09±0.01
1.62±0.45
0.20±0.02
DR-CNP+(Ours)
0.08±0.01
1.47±0.90
0.17±0.02

E.2
Numeric Results in Tables and Histograms

As the improving tricks in this work are regarding the quantile estimators, here we particularly include
the quantitative results to show the difference between DR-MAML and DR-MAML+ in Table 7/8.

32


---Page Break---
Table 6: MSEs for Pendulum 10-shot with reported standard deviations (5 runs). With α = 0.5,
the best results are in bold.

Method
Average
Worst
CVaRα
CNP [15]
0.75±0.01
1.51±0.23
0.87±0.02
TR-CNP [42]
0.76±0.00
1.24±0.02
0.85±0.01
DRO-CNP [66]
0.73±0.01
1.51±0.16
0.85±0.01
DR-CNP [1]
0.75±0.01
1.40±0.16
0.86±0.01
DR-CNP+(Ours)
0.72±0.01
1.36±0.07
0.82±0.00

Note that the studied distributionally robust strategy is on the tail risk minimization, and CVaRα
is the direct optimization indicator. As can be seen, DR-MAML+’s performance superiority over
DR-MAML is significant w.r.t. CVaRα values in 5-shot sinusoid regression and four mini-ImageNet
meta-testing tasks. These scenarios are more challenging than others as (i) the context information for
adaptation is limited in 5-shot data points and (ii) the distributional shift is severe in mini-ImageNet
meta-testing phase.

Table 7: Test average mean square errors (MSEs) with reported standard deviations for sinusoid
regression (5 runs). We respectively consider 5-shot and 10-shot cases with α = 0.7. The results
are evaluated across the 490 meta-test tasks, as in [42]. The best results are in bold.

5-shot
10-shot
Method
Average
Worst
CVaRα
Average
Worst
CVaRα
DR-MAML [1]
0.89±0.04
2.91±0.46
1.76±0.02
0.54±0.01
1.70±0.17
0.96±0.01
DR-MAML+(Ours)
0.87±0.02
2.78±0.22
1.65±0.02
0.59±0.02
1.51±0.11
0.95±0.02

Table 8: Average 5-way 1-shot classification accuracies in mini-ImageNet with reported
standard deviations (3 runs). With α = 0.5, the best results are in bold. The higher, the better for
all values.

Eight Meta-Training Tasks
Four Meta-Testing Tasks
Method
Average
Worst
CVaRα
Average
Worst
CVaRα
DR-MAML [1]
70.2±0.2
63.4±0.2
67.2±0.1
49.4±0.1
47.1±0.1
47.5±0.1
DR-MAML+(Ours)
70.4±0.1
63.8±0.2
67.5±0.1
49.9±0.1
47.2±0.1
48.1±0.1

We can attribute the performance differences of the two methods to the cumulative quantile estimation
errors using the crude MC. Even though the quantile estimation error in Fig. 5 difference is tiny
in each step, the cumulative error indeed affects the converged equilibrium a lot. This reflects the
advantage of the KDE’s used in DR-MAML+ when the task batch size cannot be set larger in practice.

We also investigate the task risk value distributions in pendulum system identification. To this end,
we visualize one run testing results for all methods in Fig. 11. It seems DR-MAML+’s result is more
skewed to the left than others.

Fig. 12 displays all methods’ performance w.r.t. the average and CVaRα returns along the meta-
training process. We exclude TR-MAML in visualization due to its worse performance and unstable
training properties. We can find that the DR-MAML exhibits a fast performance rise at the early
stage but its capability to continuously improve diminishes over time. DR-MAML+ consistently
outperforms other baselines in most cases. The above suggests that the KDE module achieves
performance gains over the crude MC when implemented with the two-stage distributionally robust
strategy for meta RL scenarios.

E.3
Sensitivity Analysis to Confidence Level

To reveal the impact of confidence levels on model performance, we perform a sensitivity analysis
with respect to confidence levels. Since only DR-MAML and DR-MAML+ are influenced by the
confidence levels during the distributionally robust optimization across all baselines, we only compare
the performance of the two methods to highlight the differences between them. As shown in Fig.

33


---Page Break---
0.5
1.0
1.5
2.0
Mean Squared Error (MSE)

0

20

40

60

80

100

Average=0.60

CVaRα=0.76

DR-MAML+(Ours)

0.5
1.0
1.5
2.0
Mean Squared Error (MSE)

0

20

40

60

80

100

Average=0.62

CVaRα=0.86

MAML
DR-MAML+(Ours)

0.5
1.0
1.5
2.0
Mean Squared Error (MSE)

0

20

40

60

80

100

120

Average=0.68
CVaRα=0.81

TR-MAML
DR-MAML+(Ours)

0.5
1.0
1.5
2.0
Mean Squared Error (MSE)

0

20

40

60

80

100

Average=0.61

CVaRα=0.82

DRO-MAML
DR-MAML+(Ours)

0.5
1.0
1.5
2.0
Mean Squared Error (MSE)

0

20

40

60

80

100

120

Average=0.61

CVaRα=0.78

DR-MAML
DR-MAML+(Ours)

Figure 11: Histograms of meta-testing performance in system identification. With α = 0.5, we
visualize the comprision results of baselines and our DR-MAML+ in 10-shot prediction. The lower,
the better for Average and CVaRα values.

0
100
200
300
400
500
Steps

−30

−25

−20

−15

−10

Average Return

0
100
200
300
400
500
Steps

−40

−30

−20

−10

CVaR Return

Point Robot Navigation

MAML
DRO-MAML
DR-MAML
DR-MAML+(Ours)

Figure 12: Learning curves for the point robot navigation task. Here, 20 trajectories work as
the support set for adaptation. The curves report the normalized returns and are averaged over four
random seeds, with α = 0.5.

13/14, we can observe that in both sinusoid 5-shot and 10-shot tasks, as the confidence level varies,
DR-MAML+ exhibits more stable performance than DR-MAML, indicating that DR-MAML+ has
a lower sensitivity to confidence levels. It can be illustrated that the crude Monte Carlo used in
DR-MAML is more unstable in terms of quantile estimation than the kernel density estimator used in
DR-MAML+. This can be due to the fact that the crude Monte Carlo method is more likely to get
stuck in the local optimal solution. In addition, it can be seen from Fig. 13/14 that the performance
of our developed DR-MAML+ is better than DR-MAML in most cases. DR-MAML+ exhibits lower
mean squared errors than DR-MAML in the average, worst, and CVaRα indicators, demonstrating
the advantages of more accurate quantile estimation in improving robustness.

0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

0.6

0.7

0.8

0.9

1.0

1.1

1.2

Average

DR-MAML
DR-MAML+(Ours)

0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

2.0

2.5

3.0

3.5

4.0

4.5

5.0

Worst

DR-MAML
DR-MAML+(Ours)

0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

1.0

1.5

2.0

2.5

3.0

CVaR

DR-MAML
DR-MAML+(Ours)

Sinusoid 5-shot

Figure 13: Meta testing performance of DR-MAML and DR-MAML+ with different confidence
level on Sinusoid 5-shot tasks. In the plots, the vertical axis is the MSEs, the horizontal axis is the
confidence level, and the shaded area represents the standard deviation.

34


---Page Break---
0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

0.4

0.6

0.8

1.0

1.2

Average

DR-MAML
DR-MAML+(Ours)

0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

1

2

3

4

Worst

DR-MAML
DR-MAML+(Ours)

0.1
0.3
0.5
0.7
0.9
Conﬁdence Level

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

CVaR

DR-MAML
DR-MAML+(Ours)

Sinusoid 10-shot

Figure 14: Meta testing performance of DR-MAML and DR-MAML+ with different confidence
level on Sinusoid 10-shot tasks. In the plots, the vertical axis is the MSEs, the horizontal axis is the
confidence level, and the shaded area represents the standard deviation.

X

0

1

2

3

4

5

Y

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Z

−2

−1

0

1

2

3

MAML

X

0

1

2

3

4

5

Y

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Z

−1

0

1

2

TR-MAML

X

0

1

2

3

4

5

Y

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Z

0

1

2

3

DRO-MAML

X

0

1

2

3

4

5

Y

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Z

−1

0

1

2

3

4

DR-MAML

X

0

1

2

3

4

5

Y

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Z

−2

0

2

4

6

DR-MAML+

Figure 15: The fast adaptation risk landscape of meta-trained MAML, TR-MAML, DRO-
MAML, DR-MAML and DR-MAML+. The figure illustrates a 5-shot sinusoid regression
example, mapping to the function space f(x) = A sin(x −B). The X-axis and Y -axis represent the
amplitude parameter a and phase parameter b respectively. The plots exhibit testing MSEs on the
Z-axis across random trials of task generation.

E.4
Further Exploration on Adaptation

We demonstrate the adaptation risk landscape of meta-trained MAML [25], TR-MAML [42], DRO-
MAML [66], DR-MAML [1] and our DR-MAML+ in Fig. 15. The adaptation risk landscape shows
the superiority of our method in optimizing within the expected tail risk minimization. Compared to
other methods, DR-MAML+ exhibits smoother and smaller risk profiles, illustrating its robustness
even in challenging tasks.

F
Computational Platforms & Softwares

This work employs Pytorch [85] as the default deep learning toolkit when implementing the developed
methods. As for baselines, TR-MAMAL follows the standard implementation as work [42] and runs
with Tensorflow [86]. Others are implemented with Pytorch. All experimental results are computed
by NVIDIA RTX6000 GPUs and A800 GPUs.

35


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the claims.

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

Justification: We discuss the limitations of the work in Sec 5.7.

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

36


---Page Break---
Justification: Refer to Appendix C.
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
Justification: Refer to Appendix D.
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

37


---Page Break---
Answer: [Yes]
Justification: Refer to Appendix D.
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
Justification: Refer to Appendix D.
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
Justification: Refer to Fig. 3/4/6/12/13/14 and Table 1/2/5/6/7/8.
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

38


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
Answer: [Yes]
Justification: Refer to Appendix F.
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
Justification: We adhere to the NeurIPS Code of Ethics.
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
Justification: Refer to Appendix A.5.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
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

39


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
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The question is not applicable to the paper.
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
Justification: The question is not applicable to the paper.
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

40


---Page Break---
Answer: [NA]
Justification: The question is not applicable to the paper.
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
Justification: The question is not applicable to the paper.
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
Justification: The question is not applicable to the paper.
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

41


---Page Break---
