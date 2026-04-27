BECAUSE:
Bilinear Causal Representation for Generalizable
Offline Model-based Reinforcement Learning

Haohong Lin1, Wenhao Ding1, Jian Chen1, Laixi Shi2, Jiacheng Zhu3, Bo Li4, Ding Zhao1

1CMU,
2Caltech,
3MIT,
4UChicago & UIUC
{haohongl, wenhaod}@andrew.cmu.edu

Abstract

Offline model-based reinforcement learning (MBRL) enhances data efficiency by
utilizing pre-collected datasets to learn models and policies, especially in scenarios
where exploration is costly or infeasible. Nevertheless, its performance often
suffers from the objective mismatch between model and policy learning, resulting in
inferior performance despite accurate model predictions. This paper first identifies
the primary source of this mismatch comes from the underlying confounders
present in offline data for MBRL. Subsequently, we introduce BilinEar CAUSal
rEpresentation (BECAUSE), an algorithm to capture causal representation for both
states and actions to reduce the influence of the distribution shift, thus mitigating
the objective mismatch problem. Comprehensive evaluations on 18 tasks that vary
in data quality and environment context demonstrate the superior performance of
BECAUSE over existing offline RL algorithms. We show the generalizability and
robustness of BECAUSE under fewer samples or larger numbers of confounders.
Additionally, we offer theoretical analysis of BECAUSE to prove its error bound
and sample efficiency when integrating causal representation into offline MBRL.
See more details in our project page: https://sites.google.com/view/be-cause.

1
Introduction

Offline Reinforcement Learning (RL) has shown great promise in learning directly from pre-collected
datasets, especially in scenarios where active interaction is expensive or infeasible [1]. Specifically,
offline model-based reinforcement learning (MBRL) [2, 3, 4], learning policies with an estimated
world model, generally perform better than their model-free counterparts in long-horizon tasks
such as self-driving vehicles [5], robotics [6], and healthcare [7]. However, offline RL suffers from
distribution shift because the rollout data is either sampled from some suboptimal behavior policies
or sampled from slightly different training environments compared to the deployment time [8].

Model Loss

Policy
𝜋(𝑎|𝑠)
Model 
𝑇((𝑠!|𝑠, 𝑎)

Environment

𝑇(𝑠′|𝑠, 𝑎)

Data Buffer

{𝑠, 𝑎, 𝑠!, 𝑟}

Objective Mismatch

Data collection with behavior policy

Model Loss

low model loss ≠ success
low model loss ≈ success

test

Our method

(Unknown)

Success
Failure

Figure 1: The objective mismatch problem.

Although identifying distribution shift issues,
many of the current offline MBRL works fail to
model the shift in environment dynamics, which
is ubiquitous and could cause catastrophic fail-
ure of trained policy at a slightly different de-
ployment stage. Furthermore, since the learn-
ing objectives of the world models and poli-
cies are isolated from each other, a significant
challenge in offline MBRL is objective mis-
match [9, 10] problem (shown in Figure 1): mod-
els that achieve a lower training loss are not nec-
essarily better for control performance. For example, a dynamics model achieve relatively low

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
prediction loss, yet it may not be sufficient to guide the planner to the high-reward region. Previous
works [9, 10] have attempted to reduce such objective mismatch by jointly learning the model and
policy. However, the performance is suboptimal due to the lack of digging into the underlying cause
of the mismatch issue [8].

In this work, we identify that the objective mismatch between model estimation and policy learning
comes from two sources of distribution shift in offline MBRL: (1) shift between the online optimal
policy and offline sub-optimal behavior policies, and (2) shift between the data collection environment
and online testing environments. Unlike humans, who make decisions based on reasoning over task-
relevant factors, models in offline RL memorize correlations without learning the causality. The
sub-optimal behavior policies introduce spurious correlations [11] between actions and states, making
the model memorize specific actions. When online testing environments differ from the data collection
environment, the model could overfit spurious correlations in the state and fail to generalize to unseen
states. Based on the analysis of the above mismatch, our work differs from previous work of causal
model-based RL [12, 13] in that we model causality in both model and policy learning., We aim to
avoid spurious correlation by discovering underlying structures between abstracted states and actions.

To alleviate objective mismatch and generalize well, we introduce the BilinEar CAUSal rEpresenta-
tion (BECAUSE) that integrates the causal representation in both world model learning and planning
of MBRL agents. Inspired by preliminary works that use bilinear MDPs to capture the structural
representation in MBRL [14], we first approximate the causal representation to capture the low-rank
structure in the world model, then use this learned representation to facilitate planning by quantifying
the uncertainty of sampled transition pairs. Consequently, we factorize the spurious correlations and
learn a unified representation for both the world model and planner.

In summary, the contribution of this paper is threefold:

• We formulate offline MBRL into the causal representation learning problem, highlighting the tight
connection between structural causal models and low-rank structures in MDPs. To the best of
our knowledge, this is the first work that systematically reveals the connection between causal
representation learning and Bilinear MDPs.

• We propose BECAUSE, an empirical causal representation framework, based on the above formu-
lation. BECAUSE first learns a causal world model, then fosters the generalizability of offline RL
agents by quantifying the uncertainty of the state transition, which facilitates conservative planning
to mitigate the objective mismatch.

• We provide extensive empirical studies and performance analysis in tasks of multiple domains to
demonstrate the superiority of BECAUSE over existing baselines, which illustrates its potential to
improve the generalizability and robustness of offline MBRL algorithms.

2
Problem Formulation

To alleviate the objective mismatch problem and the degraded performance caused by the spurious
correlation, we first provide our novel formulation of learning the underlying causal structures of
Markov Decision Process (MDP) under the bilinear MDP setting, then introduce the causal discovery
for MDP with confounders.

2.1
Preliminary: MDP and Bilnear MDP

We denote an episodic finite-horizon MDP by M =

S, A, T , H, r
	
, which is composed of state
space S, action space A, a set of transition functions T , planning horizon H and reward function r
associated with task preferences. Without loss of generality in many real-world practices, we assume
that the reward function is bounded by rh ∈[0, 1], ∀h ∈[H]. Specifically, we are interested in a
goal-conditioned reward setting, where ∀g ∈S, r(s, a; g) = 1 if and only if s = g.

Given a policy π and the state-action pair (s, a) ∈S × A, we then define the state-action value
function in the timestep h as Qπ
h(s, a) = Eπ
hPH
i=h ri(si, ai)|sh = s, ah = a
i
, and the value func-

tion V π
h (s) = Eπ
hPH
i=h ri(si, ai)|sh = s
i
. The expectation Eπ here is integrated into randomness

throughout the trajectory, which is essentially induced by the random action of the policy ai ∼π(·|si)

2


---Page Break---
and the time-homogeneous transition dynamics of the environment si+1 ∼T(·|si, ai), ∀i ∈[h, H].
In the offline dataset, the data rollouts can be seen as generated by some (mixed) behavior policy πβ,
resulting in a dataset D with in total n samples {si, ai, s′
i, ri}1≤i≤n.
Definition 1 (Bilinear MDP [14]). For each (s, a) ∈S × A, s′ ∈S, we have the corresponding
feature vector ϕ(·, ·) : R|S| × R|A| →Rd, µ(·) : R|S| →Rd′. With some core matrix M ∈Rd×d′,
we can represent the transition function kernel T(·|·, ·) as

∀s, a, s′ ∈S × A × S, T(s′|s, a) = ϕ(s, a)T Mµ(s′),
(1)

where ϕ(s, a) and µ(s′) are embedding functions that map the original state and action to the latent
space, M is the core matrix that models the transition relationship between the previous timestep
and next timestep in the latent space. Such a linear decomposition in the transition dynamics allows
us to embed structures of the transition model without the loss of general function approximation
capabilities to derive state and action representations.

2.2
Action State Confounded MDP

 𝑠
 𝑠′

 𝑎

 𝑢!

  𝑢!′
 𝑢"

 𝑠
 𝑠′

 𝑎

 𝑢
 𝑠
 𝑠′

 𝑎
 𝑢

State Nodes

Action Nodes

Confounders
Causal Path

Confounder Path

(a) Confounded MDP
(b) SC-MDP
(c) ASC-MDP (Ours)

Figure 2: Comparison of our ASC-MDP with two
existing formulations.

We consider the existence of confounders in
the MDP to represent the offline data collec-
tion process, and define action-state confounded
MDP (ASC-MDP):

Definition 2 (ASC-MDP). Besides the compo-
nents in standard MDPs M =

S, A, T, H, r
	
,
we introduce a set of unobserved confounders
u. In ASP-MDP, confounders are factorized as
u = {uπ, uc}1≤h≤H, where uπ ∈U denotes
the confounders between s and a ∼πβ(s) in-
duced by behavior policies, and uc ∈U denotes
the confounders within the state-action pairs of
the environment transition, that is, the inher-
ent structure between (s, a) and s′. Here we
assume a time-invariant confounder distribution u ∼Pu(·), ∀h ∈[H], which is a common assump-
tion [15, 16, 17]

The resulting causal relationship of ASC-MDP is demonstrated in Figure 2. Originating from
the original MDP, ASC-MDP is different from the Confounded MDP [18] and State-Confounded
MDP (SC-MDP) [19] in that it models both the spurious correlation between the current state s and
the current action a, as well as those between the next state s′ and (s, a). Yet, confounded MDP and
SC-MDP only model part of the possible confounders between states and actions. The factorization
of the confounder in ASC-MDP aligns with the source of spurious correlation in offline MBRL.

3
Proposed Method: BECAUSE

We propose BECAUSE, our core methodology for modeling, learning, and applying our causal
representations for generalizable offline MBRL. Section 3.1 models the basic format of causal
representations and analyzes their properties. Section 3.2 gives a compact way to learn the causal
representation ϕ(s, a) and µ(s′), as well as the core mask estimation M. Section 3.3 utilizes these
learned causal representations in both world model learning and MBRL planning from offline datasets.

3.1
Causal Representation for ASC-MDP

In the presence of a hidden confounder u, we model the confounder behind the transition dynamics
as a linear confounded MDP [18]:

T(s′|s, a, u) = eϕ(s, a, u)T µ(s′),
(2)

where u ∼Pu(·). Inspired by the Bilinear MDP in Definition 1, we decompose eϕ(s, a, u) into a
confounder-aware core matrix M(u) and a feature mapping ϕ(s, a), which factorize the influence of

3


---Page Break---
the confounders. Given the factorization of confounder u = {uc, uπ} in Definition 2, we derive via
d-separation in the graphical model in Figure 2 that s′ ⊥⊥uπ|{s, a, uc}. As a result, we only need to
consider the confounder uc from the environment when decomposing the transition model:

T(s′|s, a, u) = T(s′|s, a, uc) = ϕ(s, a)T M(uc)µ(s′).
(3)

Definition 3 (Construction of causal graph G). In ASC-MDP, G =
0d×d
M
0d′×d
0d′×d′

. for all (sparse)

core matrix M, the causal graph G is bipartite, thus ∀G, G ∈DAG.

Definition 3 reveals the connection between the core matrix M and causal graph G, as is formulated
in the ASC-MDP. To reduce the influence of u and estimate the unconfounded transition model
T(s′|s, a), one way is to identify the causal structures induced by the confounder uc for the transition
dynamics [20]. Existing methods in differentiable causal discovery [21, 22, 23] transform causal
discovery on some causal graph G, into a maximum likelihood estimation (MLE) with regularization:

bG = arg max
G∈DAG
log p(D; ϕ, µ, G) −λ|G|
=⇒
c
M = arg max
M
log p(D; ϕ, µ, M) −λ|M|,
(4)

Since in our case, M is a sub-matrix of the causal graph G. Given the Definition 3, G automatically
satisfies the formulation of ASC-MDP, thus discovering G is essentially estimating the sparse
submatrix M without DAG constraints: c
M = arg maxM log p(D; ϕ, µ, M) −λ|M|. We elaborate
Definition 3 and show the relationship between core matrix M and causal graph G in Appendix A.3.

We also make the assumption that the sparse G and M remain invariant with different environment
confounders in the offline training and online testing.

Assumption 1 (Invariant causal graph). We denote the causal graph G under confounder u as G(u).
The generalization problem that we aim to solve satisfies the invariance in the causal graph G(uc),
where G(uc) = G(u′
c), M(uc) = M(u′
c), uc and u′
c are the confounders in training and testing.
Remark 1. The assumption 1 can also be interpreted as task independence in [24], invariant state
representation [25], or invariant action effect in [26]. See detailed comparison in appendix Table 4.

3.2
Learning Causal Representation from Offline Data

Planning

World Model

…

···

···

···
Uncertainty
start

goal
𝑠, 𝑎

  𝑠′

Offline

Buffer

𝑇"(𝑠!|𝑠, 𝑎)

𝑢

 𝜙(⋅,⋅)

𝜇(⋅)

𝑀)

Uncertainty Quantifier 

𝐸!(s, a)

Break correlation

Figure 3: BECAUSE learns a causality-aware rep-
resentation from the buffer and uses it in both the
world model and uncertainty quantification to ob-
tain a pessimistic planning policy.

We first learn the causal world model T(s′|s, a)
in the presence of confounders u in the offline
datasets. As formulated in ASC-MDP 2, there
are two sets of confounders: uπ and uc. To
estimate an unconfounded transition model and
remove the effect of confounder, we first remove
the impact of uc which comes from the dynam-
ics shift by estimating a batch-wise transition
matrix M(uc), then we apply a reweighting for-
mula to deconfound uπ induced by the behavior
policies and mitigate the model objective mis-
match.

As discussed in Definition 3, we only need to
optimize the part of the parameters of the causal
graph G, i.e. M. Thus, we can remove the con-
straints in (4), then transform the original causal
discovery problem into a regularized MLE prob-
lem as follows:

min
M Lmask(M) = min
M (−log p(D; ϕ, µ, M) + λ|M|)

= min
M
 
E(s,a,s′)∈D∥µT (s′)K−1
µ
−ϕT (s, a)M∥2
2
|
{z
}
World Model Learning

+
λ∥M∥0
| {z }
Sparsity Regularization


.
(5)

where Kµ := P
s′∈S µ(s′)µ(s′)T is an invertible matrix. The derivation of Equation (5) is elaborated
in Appendix A.4. In practice, we use the χ2-test for discrete state and action space and the fast

4


---Page Break---
Conditional Independent Test (CIT) [27] for continuous variables to estimate each entry in the core
matrix M. We regularize the sparsity of M by controlling the p-value threshold in CIT and provide a
more detailed implementation in Appendix C.1.

Estimating the core mask provides a more accurate relationship between state and action represen-
tations, and we further refine the state action representation function ϕ and µ to help capture more
accurate transition dynamics. We optimize them by solving the following problem, according to the
transition model loss and spectral norm regularization [28] to satisfy the regularity constraints of the
feature in Assumption 3:

min
ϕ,µ Lrep(ϕ, µ) = min
ϕ,µ E(s,a,s′)∈D ∥µT (s′)K−1
µ
−ϕT (s, a)M∥2
2 + λϕ∥ϕ∥2 + λµ∥µ∥2.
(6)

The world model learning process is illustrated in Figure 3. The estimation of individual M(uc)
mitigates the spurious correlation brought by uc. To further deal with the spurious correlation in
uπ induced by the behavior policy πβ(a|s, uπ), we utilize the conditional independence property in
the ASC-MDP shown in Equation (3). The following equation shows that the true unconfounded
transition T can be rewritten in the reweighting formulas. This reweighting process in mask M serves
as the soft intervention approach [18, 29] to estimate the treatment effect in the transition function T
in an unconfounded way:

T(s′|s, a) = Epu[ϕ(s, a)T M(uc)µ(s′) · πβ(a|s, uπ)]

Epuπβ(a|s, uπ)

= ϕ(s, a)T
Epu[M(uc)πβ(a|s, uπ)]

Epuπβ(a|s, uπ)


µ(s′) ≜ϕ(s, a)T M(u)µ(s′).
(7)

The derivation of Equation (7) is illustrated in Appendix A.5.
Equation (7) basically shows
a re-weighting process given the empirical estimation of M(uc) in every batch of trajectories:
M(u) = Epu[M(uc)πβ(a|s,uπ)]

Epuπβ(a|s,uπ)
. Compared to the general reweighting strategies in previous MBRL
literatures [18, 29] which reweights the entire value function, this re-weighting process is conducted
only on the estimated matrix, while the representation ϕ(s, a) and µ(s′) are subsequently regularized
by weighted estimation of M. The pipeline of causal world model learning is described in the
first part of the Algorithm 1. We discuss more details of the implementation and experiment in
Appendix C.

3.3
Causal Representation for Uncertainty Quantification

Algorithm 1: BECAUSE Training and Planning
Input: Offline dataset D, causal discovery frequency k
Output: Causal mask M, feature function bϕ, bµ, policy bπ
// Causal world model learning
M0 ←[1]d′×d
for i ∈[K] do

Update bϕn, bµn by Lrep(ϕ, µ) in (6)
if i mod k = 0 then

Update c
Mn with Lmask(M) in (5)
Weighted average M n with (7)
// Uncertainty quantifier learning
Fit Eθ(s, a) with LEBM (8)
Initialize bVH+1(s) = 0, ∀(s, a)
// Pessimistic planning
while h < H do

Estimate the uncertainty with score Eθ(s, a)
Compute Qh(s, a) with (9)
bVh(s, a) = maxa Q(s, a)
a ←arg maxa Q(s, a)
s′, r ←env.step (a, g)

To avoid entering OOD states in the
online deployment, we further design
a pessimistic planner according to the
uncertainty of the predicted trajecto-
ries in the imagination rollout step to
mitigate objective mismatch.

We use the feature embedding from
bilinear causal representation to help
quantify the uncertainty, denoted as
Eθ(s, a). As we have access to the
offline dataset, we learn an Energy-
based Model (EBM) [30, 31] based
on the abstracted state representation
ϕ and core matrix M. A higher out-
put of the energy function Eθ(·, ·) in-
dicates a higher uncertainty in the cur-
rent state as they are visited by the be-
havior policies πβ less frequently. In
practice, the energy-based model usu-
ally suffers from a high-dimensional
data space [32]. To mitigate this over-
head of training a good uncertainty

5


---Page Break---
quantifier, we first embed the state
samples through the abstract representation µ(s′), and the state action pair via ϕ(s, a).

LEBM(θ) = E b
T (·|s,a)Eθ[µ(s+)|ϕ(s, a)] −Eq(s,a)Eθ[µ(s−)|ϕ(s, a)] + λEBM∥θ∥2,
(8)

where µ(s)+ refers to the positive samples from the approximated transition dynamics bT(·|s, a),
and µ(s−) refers to the latent negative samples via the Langevin dynamics [30]. Additionally, we
regularize the parameters of EBM to avoid overfitting issues. We attach more training details and
results of EBMs in Appendix C.2 The learned energy function Eθ(s, a) is used to quantify the
uncertainty based on the offline data.

During the online planning stage, we use the learned EBM to adjust the reward estimation based on
Model Predictive Control (MPC) [33]. At timestep h, we basically subtract the original step return
estimation rh(s, a) by its uncertainty Eθ(s, a):

Qh(s, a) = bQh(s, a) −Eθ(s, a) = rh(s, a) −Eθ(s, a)
|
{z
}
Adjusted Return

+
X

s′∈S
bT(s′|s, a)bVh+1(s′).
(9)

3.4
Theoretical Analysis of BECAUSE

Then we move on to develop the theoretical analysis for the proposed method BECAUSE. Based
on two standard Assumption 2 and 3 on the feature’s existence and regularity, we achieve the
finite-sample complexity guarantee — an upper bound of the suboptimality gap as follows, whose
proof is postponed to Appendix B.
Theorem 1 (Performance guarantee). Consider any 0 < δ < 1 and any initial state es ∈S. Under
the Assumption 2, 3 and that the transition model T is an SCM (defined in 4), for any accuracy level
0 ≤ξ ≤1, with probability at least 1 −δ, the output policy π of BECAUSE (Algorithm 1) based on
the historical dataset D with n = P

(s,a)∈S×A n(s, a) samples generated from a behavior policy πβ
satisfies:

V ∗
1 (es) −V π
1 (es) ≲min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗

"s

log(1/δ)
n(sh, ah) | s1 = es

#

,

where C1, Cs are some universal constants, σ is SCM’s noise level (see Definition 4), and M ∈Rd×d′

is the optimal ground truth sparse transition matrix to be estimated.

The error bound shrinks as the offline sample size n over all state-action pairs increase. It also grows
proportionally to the planning horizon H, SCM’s noise level σ, and the ℓ0 norm of the ground true
causal mask M, which describes the intrinsic complexity of the world model.

Consequently, with Proposition 1 in the Appendix, we can achieve ξ-optimal policy (V ∗
1 (es)−V π
1 (es) ≤
ξ) as long as the historical dataset satisfies the following conditions: ∀0 ≤ξ ≤1,

min
(s,a,h)∈S×A×[H] Eπ⋆
n(sh, ah) | s1 = es

≳
min

C2
1 log2   ∥M∥0

ξ

|S|, C2
sσ2∥M∥0
	
· H2 log(1/δ)

ξ2
.

4
Experiment Results

Figure 4: Three environments used in this paper.

In this section, we conduct a comprehensive em-
pirical evaluation of BECAUSE’s generalization
performance in a diverse set of environments,
covering different decision-making problems in
the grid world, manipulation, and autonomous
driving domains, shown in Figure 4.

4.1
Experiment Setting

Environment Design
We design 18 tasks in 3 representative RL environments in Figure 4. Agents
need to acquire reasoning capabilities to receive higher rewards and achieve goals.

6


---Page Break---
Lift-Random
Lift-Medium
Lift-Expert

Unlock-Random
Unlock-Medium
Unlock-Expert

Crash-Random
Crash-Medium
Crash-Expert

Unlock-Random
Unlock-Random

Unlock-Medium
Unlock-Medium

Unlock-Expert
Unlock-Expert

(a)
(b)
(c)

Figure 5: Results of BECAUSE and baselines in different tasks. (a) Average success rate in distri-
bution and out of distribution. (b) Average success rate w.r.t. ratio of offline samples. (c) Average
success rate w.r.t. spurious level in the environments. We evaluate the mean and standard deviation of
the best performance among 10 random seeds and report task-wise results in Appendix Table 6.

• Lift: Object manipulation environment in RoboSuite [34]. We designed this environment for the
agent to lift an object with a specific color configuration on the table to a desired height. In the
OOD environment Lift-O, there is an injected spurious correlation between the color of the cube
and the position of the cube in the training phase. During the testing phase, the correlation between
color and position is different from training.
• Unlock: We designed this environment for the agent to collect a key to open doors in Minigrid [35].
In the OOD environment Unlock-O, there will be a different number of goals (doors to be opened)
in the testing environments from the training environments.
• Crash: Safety is critical in autonomous driving, which is reflected by the collision avoidance
capability. We consider a risky scenario where an AV collides with a jaywalker because its view is
blocked by another car [36]. We design such a crash scenario based on highway-env [37], where
the goal is to create crashes between a pedestrian and AVs. In the OOD environment Crash-O, the
distribution of reward (number of pedestrians) is different in online testing environments.

For all three different environments, we set a specific subset of the state space as the goal g ∈S, and
the reward is defined as the goal-reaching reward r(s, a, g) = I(r = g). When the episode ends in
the goal state within the task horizon H, the episode is considered a success. We then use the average
success rate as the general evaluation metrics for our BECAUSE and all baselines.

In each environment, we collect three types of offline data: random, medium, and expert based on
the different levels of uπ in the behavior policies. In Unlock environments, we collect 200 episodes
from each level of behavior policies as the offline demonstration data, and the number of episodes is
1,000 in the environments Lift and Crash, which all have continuous state and action space. A more
detailed view of the environment hyperparameters and behavior policy design is in Appendix C.5.

Baselines
We compare our proposed BECAUSE with several offline causal RL or MBRL baselines.
ICIL [38] learns a dynamic-aware invariant causal representation learning to assist a generalizable
policy learning from offline datasets. CCIL [39] conducts a soft intervention in our offline setting
by jointly optimizing policy parameters and masks over the state. MnM [9] unifies the objective of
jointly training the model and policy, which allocates larger weights in the state prediction loss in the
high-reward region. Delphic [40] introduces delphic uncertainty to differentiate between uncertainties
caused by hidden confounders and traditional epistemic and aleatoric uncertainties. TD3+BC [41]
is an offline model-free RL approach that combines the Twin Delayed Deep Deterministic Policy

7


---Page Break---
Gradient (TD3) algorithm with Behavior Cloning (BC) to adopt both the actor-critic framework
and supervised learning from expert demonstrations. MOPO [2] is an offline MBRL approach that
uses flat latent space and count-based uncertainty quantification to maintain conservatism in online
deployment. GNN [42] is a GNN-based baseline using a Relational Graph Convolutional Network to
model the temporal dependency of state-action pairs in the dynamic model with message passing.
CDL [24] uses causal discovery to learn a task-independent world model. Denoised MDP [12] and
IFactor [13] conduct causal state abstraction based on their controllability and reward relevance.
The last three methods are designed for online settings, so we only implement their model learning
objectives. We attach more details of the baseline implementation in Appendix C.6.

4.2
Experiment Results Analysis

We empirically answer the following research questions.

• RQ1: How is the generalizability of BECAUSE in the online environments (which may be unseen)?
Specifically, how does BECAUSE perform under diverse qualities of demonstration data (different
level of uπ), and different environment contexts (different uc)?

• RQ2: How does the design in BECAUSE contribute to the robustness of its final performance
under different sample sizes or spurious levels?

• RQ3: How does BECAUSE scale up to visual RL tasks with image observation input compared to
other visual RL baselines?

• RQ4: How does BECAUSE achieve the aforementioned generalizability by mitigating the objective
mismatch problem in offline MBRL?

For RQ1, in Figure 5(a), we evaluate the success rate in the online environment against different
baselines. The result shows that under different environments and different qualities of behavior
policies πβ (different uπ), BECAUSE consistently achieves the best performance in 8 out of 9 for
both the in-distribution (I) and out-of-distribution (O) for all the demonstration data quality (different
level of uπ). Where O here indicates the tasks under unseen environment with confounder u′
c ̸= uc
different from offline training. Another finding is that model-based approaches generally perform
better than model-free approaches at various levels of offline data, which shows the importance of
world model learning for generalizable offline RL. We attach the detailed results in the Appendix
Table 6, 7 and the causal masks discovered in each environment in Appendix Figure 8 for reference.

For RQ2, we compare different aspects of BECAUSE’s robustness with MOPO without causal
structures [2]. We compare their performance with different ratios of the entire offline dataset and
illustrate the success rates in Figure 5(b). The result shows that, for any selected number of samples,
BECAUSE consistently outperforms MOPO with a clear margin. We also evaluate BECAUSE
performance at higher spurious levels in Figure 5(c). We add up to 8× of the original number of
confounders in the environments to test the robustness of the agent’s performance. BECAUSE
consistently outperforms MOPO and the margin enlarges as the spurious level grows higher.

Table 1: Comparison of visual RL performance.

Tasks
ICIL
IFactor
BECAUSE

Unlock-I-random
0.8±0.8
4.3±1.1
15.7±3.3
Unlock-O-random
1.5±1.8
4.7±1.6
5.9±0.9
Unlock-I-medium
5.3±2.0
30.2±4.1
62.0±4.6
Unlock-O-medium
8.6±4.2
15.4±2.4
71.6±9.1
Unlock-I-expert
8.7±3.4
34.0±4.8
63.7±3.9
Unlock-O-expert
17.1±4.2
16.7±3.1
73.6±19.5

For RQ3, we conduct experiments with
visual inputs in the Unlock environments
with ICIL [38] and IFactor [13]. We param-
eterize the feature encoder as a three-layer
CNN with 128 dimensions hidden size for
all the baselines and our methods. The re-
sults in Table 1 show that BECAUSE can
significantly improve both in-distribution
and out-of-distribution performance under
different quality of behavior policies.

For RQ4, we aim to understand whether BECAUSE achieves higher performance by resolving the
objective mismatch problem. We first collect two groups of trajectories: τpos and τneg, each with
positive reward (success) and negative reward (failure) in Unlock task with sparse goal-reaching
reward. We want to have a model whose loss is informative for discriminating control results,
that is, we wish Lmodel(τpos) < Lmodel(τneg). According to our visualization in Figure 6, in
Unlock-Expert and Unlock-Medium, the ratio of τpos is much higher in BECAUSE than MOPO
among the trajectories with low model loss. In Unlock-Random, the mismatch of the model and

8


---Page Break---
MOPO
BECAUSE

Example Policies

Positive Reward

Negative Reward

Unlock-random
Unlock-medium
Unlock-expert

Figure 6: Evaluation of the difference between the distribution of episodic model loss for success
and failure trajectories. The higher difference indicates a reduction in model mismatch issues. An
example of failure mode is trying to open the door without having the key.

control objective is more significant, since the demonstration is poor in state coverage. MOPO cannot
succeed even when the model loss is low, whereas our methods can. We perform a hypothesis test
with H0 : Lmodel(τpos) < Lmodel(τneg). In BECAUSE, this desired property is more significant
than MOPO attributed to the causal representation we learn, indicating a reduction of objective
mismatch. We attach detailed discussions for the mismatch evaluation in Appendix C.3 and Table 5.

4.3
Ablation Studies

Table 2: The ablation studies between BECAUSE and
its variants. We report the overall Success rate (%) over
9 in-distribution (I) and 9 out-of-distribution (O) tasks,
respectively. Bold is the best.

Variants
BECAUSE Optimism Linear
Full

Overall-I
73.3±4.5
64.4±6.4
57.9±6.1
39.3±6.3
Overall-O
43.0±4.9
32.4±3.3
33.2±5.2
25.2±3.9

We conducted ablation studies with three
variants of BECAUSE and report the aver-
age success rate across nine in-distribution
and nine out-of-distribution tasks in Ta-
ble 2.
The Optimism variant conducts
optimistic planning instead of pessimistic
planning in Equation (9), which uses uni-
form sampling in the planner module. The
Linear variant assumes a full connection
to the causal matrix M, then directly uses
linear MDP to parameterize the dynamics model T, which removes the causal discovery module
in BECAUSE. The Full variant learns from the full batch of data to estimate the causal mask with-
out iterative update. We report the results of the task-wise ablation with confidence interval and
significance in Appendix Table 8 and 9.

5
Related Works

Objective Mismatch in MBRL
The objective mismatch in MBRL [43, 44] refers to the fact that
pure MLE estimation of the world model does not align well with the control objective. Previous
works [29, 45] propose reweighting during model training to alleviate this mismatch, [46] proposes
a goal-aware prediction by redistributing model error according to their task relevance. These works
essentially reweight loss for the entire model training, while our work conducts reweighting just
over the estimated causal mask more efficiently. More recently, [9, 10] proposed a joint training
between the world model and policies. Although joint optimization improves performance, they do
not address the generalizability of the learned model under the distribution shift setting. In the offline
setting, Model-based RL [2, 3, 4, 47] employs model ensemble, pessimistic policy optimization or
value iteration [48, 49], and an energy-based model for planning [50] to quantify uncertainty and
improve test performance. To the best of our knowledge, no previous work explored or modeled the
impact of distribution shift on the objective mismatch problem in MBRL.

9


---Page Break---
Causal Discovery with Confounder
Most of the existing causal discovery methods [51] can
be categorized into constraint-based and score-based. Constraint-based methods [52] start from
a complete graph and iteratively remove edges with statistical hypothesis testing [53, 54]. This
type of method is highly data-efficient but not robust to noisy data. As a remedy, score-based
methods [55, 56] use metrics such as the likelihood or BIC [57] as scores to manipulate edges in
the causal graph. Recently, researchers have extended score-based methods with RL [58], order
learning [59] or differentiable discovery [22, 60, 61]. To alleviate the non-identifiability under
hidden confounders, active intervention methods have been explored [62], aiming to break spurious
correlations in an online fashion. With extra assumptions on confounders, some recent works detect
such correlations [63, 64, 65] so that models can effectively identify elusive confounders.

Causal Reinforcement Learning
Recently, many RL algorithms have incorporated causality
to improve reasoning capability [66] and generalizability. For instance, [67] and [68] explicitly
estimate causal structures with the interventional data obtained from the environment in an online
setting. These structures can be used to constrain the output space [19] or to adjust the buffer
priority [69]. Building dynamic models in model-based RL [24, 70, 71] based on causal graphs
is widely studied. Most existing causal MBRL works focus on estimating the causal world model
by predicting transition dynamics and rewards. Existing methods learn this causal world model
via structural regularization [23, 72, 73], conditional independence test [24, 70, 74], variational
inference [75, 12], counterfactual data augmentation [76, 77], hierarchical skill abstraction [78, 79],
uncertainty quantification [40], reward redistribution [24, 80], causal context modeling [81, 82] and
structure-aware state abstraction [12, 13, 83, 84] based on the controllability and task or reward
relevance. However, the presence of confounders during data collection can skew the learned policy,
making it susceptible to spurious correlations. Deconfounding solutions have been proposed either
between actions and states [39, 85, 86] or among different dimensions of state variables [19, 87].

6
Conclusion

In this paper, we study how to mitigate the objective mismatch problem in MBRL, especially under
the offline settings where distribution shift occurs. We first propose ASC-MDP and the bilinear
causal representation associated with it. Based on the formulation, we proposed how to learn this
causal abstraction by alternating between causal mask learning and feature learning in fitting the
world dynamics. In the planning stage, we applied the learned causal representation to an uncertainty
quantification module based on EBM, which improves the robustness under uncertainty in the
online planning stage.
We theoretically justify BECAUSE’s sub-optimality bound induced by
the sparse matrix estimation problem and offline RL. Comprehensive experiments on 18 different
tasks show that given a diverse level of demonstration as the offline dataset, BECAUSE has better
generalizability than baselines in different online environments, and it robustly outperforms baselines
under different spurious levels or sample sizes. We empirically show that BECAUSE mitigates the
objective mismatch with causal awareness learned from offline data. One limitation of BECAUSE
lies in its simplified assumption of time-homogeneous causal structure, which may not always hold in
long-horizon or non-stationary settings. Besides, the current implementation is still based on vector
observations. It will be interesting to scale up the causal reasoning framework into high-dimensional
observations to discover concept factors in long-horizon visual RL settings.

Acknowledgement

This work is partially supported by the Defense Advanced Research Projects Agency (DARPA) under
Contract No. HR00112320012. The work of L. Shi is supported in part by the Resnick Institute and
Computing, Data, and Society Postdoctoral Fellowship at California Institute of Technology.

10


---Page Break---
References

[1] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning:
Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

[2] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea
Finn, and Tengyu Ma. Mopo: Model-based offline policy optimization. Advances in Neural
Information Processing Systems, 33:14129–14142, 2020.

[3] Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel:
Model-based offline reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F.
Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33,
pages 21810–21823. Curran Associates, Inc., 2020.

[4] Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea
Finn. Combo: Conservative offline model-based policy optimization. Advances in neural
information processing systems, 34:28954–28967, 2021.

[5] Zeyu Zhu and Huijing Zhao. A survey of deep rl and il for autonomous driving policy learning.
IEEE Transactions on Intelligent Transportation Systems, 23(9):14043–14065, 2021.

[6] Aviral Kumar, Anikait Singh, Frederik Ebert, Mitsuhiko Nakamoto, Yanlai Yang, Chelsea Finn,
and Sergey Levine. Pre-training for robots: Offline rl enables learning new tasks from a handful
of trials. arXiv preprint arXiv:2210.05178, 2022.

[7] Shengpu Tang and Jenna Wiens. Model selection for offline reinforcement learning: Practical
considerations for healthcare settings. In Machine Learning for Healthcare Conference, pages
2–35. PMLR, 2021.

[8] Laixi Shi and Yuejie Chi. Distributionally robust model-based offline reinforcement learning
with near-optimal sample complexity. arXiv preprint arXiv:2208.05767, 2022.

[9] Benjamin Eysenbach, Alexander Khazatsky, Sergey Levine, and Russ R Salakhutdinov. Mis-
matched no more: Joint model-policy optimization for model-based rl. Advances in Neural
Information Processing Systems, 35:23230–23243, 2022.

[10] Shentao Yang, Shujian Zhang, Yihao Feng, and Mingyuan Zhou. A unified framework for alter-
nating offline model training and policy learning. Advances in Neural Information Processing
Systems, 35:17216–17232, 2022.

[11] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: founda-
tions and learning algorithms. The MIT Press, 2017.

[12] Tongzhou Wang, Simon S Du, Antonio Torralba, Phillip Isola, Amy Zhang, and Yuandong
Tian. Denoised mdps: Learning world models better than the world itself. arXiv preprint
arXiv:2206.15477, 2022.

[13] Yu-Ren Liu, Biwei Huang, Zhengmao Zhu, Honglong Tian, Mingming Gong, Yang Yu, and Kun
Zhang. Learning world models with identifiable factorization. arXiv preprint arXiv:2306.06561,
2023.

[14] Lin Yang and Mengdi Wang. Reinforcement learning in feature space: Matrix bandit, kernels,
and regret bound. In International Conference on Machine Learning, pages 10746–10756.
PMLR, 2020.

[15] Weitong Zhang, Jiafan He, Dongruo Zhou, Q Gu, and A Zhang. Provably efficient representation
selection in low-rank markov decision processes: from online to offline rl. In Uncertainty in
Artificial Intelligence, pages 2488–2497. PMLR, 2023.

[16] Fan Feng and Sara Magliacane. Learning dynamic attribute-factored world models for efficient
multi-object reinforcement learning. arXiv preprint arXiv:2307.09205, 2023.

[17] Angela Zhou. Reward-relevance-filtered linear offline reinforcement learning. arXiv preprint
arXiv:2401.12934, 2024.

11


---Page Break---
[18] Lingxiao Wang, Zhuoran Yang, and Zhaoran Wang. Provably efficient causal reinforcement
learning with confounded observational data. Advances in Neural Information Processing
Systems, 34:21164–21175, 2021.

[19] Wenhao Ding, Laixi Shi, Yuejie Chi, and Ding Zhao. Seeing is not believing: Robust reinforce-
ment learning against spurious correlation. arXiv preprint arXiv:2307.07907, 2023.

[20] Junzhe Zhang and Elias Bareinboim. Markov decision processes with unobserved confounders:
A causal approach. Technical report, Technical report, Technical Report R-23, Purdue AI Lab,
2016.

[21] Karren Yang, Abigail Katcoff, and Caroline Uhler. Characterizing and learning equivalence
classes of causal dags under interventions. In International Conference on Machine Learning,
pages 5541–5550. PMLR, 2018.

[22] Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and
Alexandre Drouin. Differentiable causal discovery from interventional data. Advances in Neural
Information Processing Systems, 33:21865–21877, 2020.

[23] Zizhao Wang, Xuesu Xiao, Yuke Zhu, and Peter Stone. Task-independent causal state abstraction.
Workshop on Robot Learning: Self-Supervised and Lifelong Learning, NeurIPS, 2021.

[24] Zizhao Wang, Xuesu Xiao, Zifan Xu, Yuke Zhu, and Peter Stone. Causal dynamics learning for
task-independent state abstraction. arXiv preprint arXiv:2206.13452, 2022.

[25] Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau,
Yarin Gal, and Doina Precup. Invariant causal prediction for block mdps. In International
Conference on Machine Learning, pages 11214–11224. PMLR, 2020.

[26] Wenxuan Zhu, Chao Yu, and Qiang Zhang. Causal deep reinforcement learning using observa-
tional data. arXiv preprint arXiv:2211.15355, 2022.

[27] Krzysztof Chalupka, Pietro Perona, and Frederick Eberhardt. Fast conditional independence
test for vector variables with large sample sizes. arXiv preprint arXiv:1804.02747, 2018.

[28] Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generaliz-
ability of deep learning. arXiv preprint arXiv:1705.10941, 2017.

[29] Nathan Lambert, Brandon Amos, Omry Yadan, and Roberto Calandra. Objective mismatch in
model-based reinforcement learning. arXiv preprint arXiv:2002.04523, 2020.

[30] Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models.
Advances in Neural Information Processing Systems, 32, 2019.

[31] Yezhen Wang, Bo Li, Tong Che, Kaiyang Zhou, Ziwei Liu, and Dongsheng Li. Energy-based
open-world uncertainty modeling for confidence calibration. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 9302–9311, 2021.

[32] Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space
energy-based prior model. Advances in Neural Information Processing Systems, 33:21994–
22008, 2020.

[33] Eduardo F Camacho and Carlos Bordons Alba. Model predictive control. Springer science &
business media, 2013.

[34] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Soroush
Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for
robot learning. In arXiv preprint arXiv:2009.12293, 2020.

[35] Maxime Chevalier-Boisvert, Lucas Willems, and Suman Pal. Minimalistic gridworld environ-
ment for openai gym. https://github.com/maximecb/gym-minigrid, 2018.

[36] Zenna Tavares, James Koppel, Xin Zhang, Ria Das, and Armando Solar-Lezama. A language
for counterfactual generative models. In International Conference on Machine Learning, pages
10173–10182. PMLR, 2021.

12


---Page Break---
[37] Edouard Leurent. An environment for autonomous driving decision-making. https://github.
com/eleurent/highway-env, 2018.

[38] Ioana Bica, Daniel Jarrett, and Mihaela van der Schaar. Invariant causal imitation learning for
generalizable policies. Advances in Neural Information Processing Systems, 34:3952–3964,
2021.

[39] Pim De Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning.
Advances in Neural Information Processing Systems, 32, 2019.

[40] Alizée Pace, Hugo Yèche, Bernhard Schölkopf, Gunnar Ratsch, and Guy Tennenholtz. Delphic
offline reinforcement learning under nonidentifiable hidden confounding. In The Twelfth
International Conference on Learning Representations, 2023.

[41] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.
Advances in neural information processing systems, 34:20132–20145, 2021.

[42] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max
Welling. Modeling relational data with graph convolutional networks. In European semantic
web conference, pages 593–607. Springer, 2018.

[43] Amir-massoud Farahmand, Andre Barreto, and Daniel Nikovski. Value-aware loss function for
model-based reinforcement learning. In Artificial Intelligence and Statistics, pages 1486–1494.
PMLR, 2017.

[44] Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algo-
rithmic framework for model-based deep reinforcement learning with theoretical guarantees.
arXiv preprint arXiv:1807.03858, 2018.

[45] Pierluca D’Oro, Alberto Maria Metelli, Andrea Tirinzoni, Matteo Papini, and Marcello Restelli.
Gradient-aware model-based policy search. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 34, pages 3801–3808, 2020.

[46] Suraj Nair, Silvio Savarese, and Chelsea Finn. Goal-aware prediction: Learning to model what
matters. In International Conference on Machine Learning, pages 7207–7219. PMLR, 2020.

[47] Yihao Sun, Jiaji Zhang, Chengxing Jia, Haoxin Lin, Junyin Ye, and Yang Yu. Model-bellman
inconsistency for model-based offline reinforcement learning. In International Conference on
Machine Learning, pages 33177–33194. PMLR, 2023.

[48] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement
learning with linear function approximation. In Conference on Learning Theory, pages 2137–
2143. PMLR, 2020.

[49] Masatoshi Uehara and Wen Sun. Pessimistic model-based offline reinforcement learning under
partial coverage. arXiv preprint arXiv:2107.06226, 2021.

[50] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Learning iterative reasoning
through energy minimization. In International Conference on Machine Learning, pages 5570–
5582. PMLR, 2022.

[51] Clark Glymour, Kun Zhang, and Peter Spirtes. Review of causal discovery methods based on
graphical models. Frontiers in genetics, 10:524, 2019.

[52] Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. Causation, prediction,
and search. MIT press, 2000.

[53] Karl Pearson. X. on the criterion that a given system of deviations from the probable in the case
of a correlated system of variables is such that it can be reasonably supposed to have arisen from
random sampling. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
Science, 50(302):157–175, 1900.

[54] Kun Zhang, Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Kernel-based conditional
independence test and application in causal discovery. arXiv preprint arXiv:1202.3775, 2012.

13


---Page Break---
[55] David Maxwell Chickering. Optimal structure identification with greedy search. Journal of
machine learning research, 3(Nov):507–554, 2002.

[56] Alain Hauser and Peter Bühlmann. Characterization and greedy learning of interventional
markov equivalence classes of directed acyclic graphs. The Journal of Machine Learning
Research, 13(1):2409–2464, 2012.

[57] Andrew A Neath and Joseph E Cavanaugh. The bayesian information criterion: background,
derivation, and applications.
Wiley Interdisciplinary Reviews: Computational Statistics,
4(2):199–203, 2012.

[58] Shengyu Zhu, Ignavier Ng, and Zhitang Chen. Causal discovery with reinforcement learning.
arXiv preprint arXiv:1906.04477, 2019.

[59] Dezhi Yang, Guoxian Yu, Jun Wang, Zhengtian Wu, and Maozu Guo. Reinforcement causal
structure learning on order graph. In Proceedings of the AAAI Conference on Artificial Intelli-
gence, volume 37, pages 10737–10744, 2023.

[60] Nan Rosemary Ke, Olexa Bilaniuk, Anirudh Goyal, Stefan Bauer, Hugo Larochelle, Bernhard
Schölkopf, Michael C Mozer, Chris Pal, and Yoshua Bengio. Learning neural causal models
from unknown interventions. arXiv preprint arXiv:1910.01075, 2019.

[61] Yunzhu Li, Antonio Torralba, Anima Anandkumar, Dieter Fox, and Animesh Garg. Causal
discovery in physical systems from videos. Advances in Neural Information Processing Systems,
33:9180–9192, 2020.

[62] Nino Scherrer, Olexa Bilaniuk, Yashas Annadani, Anirudh Goyal, Patrick Schwab, Bernhard
Schölkopf, Michael C Mozer, Yoshua Bengio, Stefan Bauer, and Nan Rosemary Ke. Learning
neural causal models with active interventions. arXiv preprint arXiv:2109.02429, 2021.

[63] Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. Don’t take the easy way out: Ensemble-
based methods for avoiding known dataset biases. arXiv preprint arXiv:1909.03683, 2019.

[64] Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton. Learning the difference that makes a
difference with counterfactually-augmented data. arXiv preprint arXiv:1909.12434, 2019.

[65] Meike Nauta, Ricky Walsh, Adam Dubowski, and Christin Seifert. Uncovering and correcting
shortcut learning in machine learning models for skin cancer diagnosis. Diagnostics, 12(1):40,
2021.

[66] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Explainable reinforcement
learning through a causal lens. In Proceedings of the AAAI conference on artificial intelligence,
volume 34, pages 2493–2500, 2020.

[67] Suraj Nair, Yuke Zhu, Silvio Savarese, and Li Fei-Fei. Causal induction from visual observations
for goal directed tasks. arXiv preprint arXiv:1910.01751, 2019.

[68] Sergei Volodin, Nevan Wichers, and Jeremy Nixon. Resolving spurious correlations in causal
models of environments via interventions. arXiv preprint arXiv:2002.05217, 2020.

[69] Maximilian Seitzer, Bernhard Schölkopf, and Georg Martius. Causal influence detection for
improving efficiency in reinforcement learning. Advances in Neural Information Processing
Systems, 34, 2021.

[70] Zheng-Mao Zhu, Xiong-Hui Chen, Hong-Long Tian, Kun Zhang, and Yang Yu. Offline
reinforcement learning with causal structured world models. arXiv preprint arXiv:2206.01474,
2022.

[71] Mirco Mutti, Riccardo De Santi, Emanuele Rossi, Juan Felipe Calderon, Michael Bronstein, and
Marcello Restelli. Provably efficient causal model-based reinforcement learning for systematic
generalization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37,
pages 9251–9259, 2023.

14


---Page Break---
[72] Fan Feng, Biwei Huang, Kun Zhang, and Sara Magliacane. Factored adaptation for non-
stationary reinforcement learning.
Advances in Neural Information Processing Systems,
35:31957–31971, 2022.

[73] Tianying Ji, Yongyuan Liang, Yan Zeng, Yu Luo, Guowei Xu, Jiawei Guo, Ruijie Zheng, Furong
Huang, Fuchun Sun, and Huazhe Xu. Ace: Off-policy actor-critic with causality-aware entropy
regularization. arXiv preprint arXiv:2402.14528, 2024.

[74] Jiaheng Hu, Zizhao Wang, Peter Stone, and Roberto Martin-Martin. Elden: Exploration via
local dependencies. arXiv preprint arXiv:2310.08702, 2023.

[75] Wenhao Ding, Haohong Lin, Bo Li, and Ding Zhao. Generalizing goal-conditioned reinforce-
ment learning with variational causal reasoning. Advances in Neural Information Processing
Systems, 35:26532–26548, 2022.

[76] Lars Buesing, Theophane Weber, Yori Zwols, Nicolas Heess, Sebastien Racaniere, Arthur Guez,
and Jean-Baptiste Lespiau. Woulda, coulda, shoulda: Counterfactually-guided policy search. In
International Conference on Learning Representations, 2018.

[77] Silviu Pitis, Elliot Creager, Ajay Mandlekar, and Animesh Garg. Mocoda: Model-based
counterfactual data augmentation. arXiv preprint arXiv:2210.11287, 2022.

[78] Tabitha E Lee, Jialiang Alan Zhao, Amrita S Sawhney, Siddharth Girdhar, and Oliver Kroemer.
Causal reasoning in simulation for structure and transfer learning of robot manipulation policies.
In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 4776–4782.
IEEE, 2021.

[79] Tabitha Edith Lee, Shivam Vats, Siddharth Girdhar, and Oliver Kroemer. Scale: Causal learning
and discovery of robot manipulation skills using simulation. In Conference on Robot Learning,
pages 2229–2256. PMLR, 2023.

[80] Yudi Zhang, Yali Du, Biwei Huang, Ziyan Wang, Jun Wang, Meng Fang, and Mykola Pech-
enizkiy. Interpretable reward redistribution in reinforcement learning: A causal approach.
Advances in Neural Information Processing Systems, 36, 2024.

[81] Biwei Huang, Fan Feng, Chaochao Lu, Sara Magliacane, and Kun Zhang. Adarl: What, where,
and how to adapt in transfer reinforcement learning. In International Conference on Learning
Representations, 2021.

[82] Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis,
Bingqing Chen, and Ding Zhao. What went wrong? closing the sim-to-real gap via differentiable
causal discovery. In Conference on Robot Learning, pages 734–760. PMLR, 2023.

[83] Xiang Fu, Ge Yang, Pulkit Agrawal, and Tommi Jaakkola. Learning task informed abstractions.
In International Conference on Machine Learning, pages 3480–3491. PMLR, 2021.

[84] Biwei Huang, Chaochao Lu, Liu Leqi, José Miguel Hernández-Lobato, Clark Glymour, Bern-
hard Schölkopf, and Kun Zhang. Action-sufficient state representation learning for control with
structural constraints. In International Conference on Machine Learning, pages 9260–9279.
PMLR, 2022.

[85] Zhihong Deng, Zuyue Fu, Lingxiao Wang, Zhuoran Yang, Chenjia Bai, Zhaoran Wang, and Jing
Jiang. Score: Spurious correlation reduction for offline reinforcement learning. arXiv preprint
arXiv:2110.12468, 2021.

[86] Maxime Gasse, Damien Grasset, Guillaume Gaudron, and Pierre-Yves Oudeyer. Using con-
founded data in latent model-based reinforcement learning. Transactions on Machine Learning
Research, 2023.

[87] Junzhe Zhang, Daniel Kumor, and Elias Bareinboim. Causal imitation learning with unobserved
confounders. Advances in neural information processing systems, 33:12263–12274, 2020.

[88] Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In
International Conference on Machine Learning, pages 5084–5096. PMLR, 2021.

15


---Page Break---
[89] Thomas Blumensath and Mike E Davies. Iterative hard thresholding for compressed sensing.
Applied and computational harmonic analysis, 27(3):265–274, 2009.

[90] Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear
inverse problems. SIAM journal on imaging sciences, 2(1):183–202, 2009.

[91] Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement
learning. Advances in neural information processing systems, 21, 2008.

[92] Ryan Tibshirani and Larry Wasserman. Sparsity, the lasso, and friends. Lecture notes from
“Statistical Machine Learning,” Carnegie Mellon University, Spring, 2017.

[93] Patrick E McKnight and Julius Najab. Mann-whitney u test. The Corsini encyclopedia of
psychology, pages 1–1, 2010.

16


---Page Break---
A
Auxiliary Details of BECAUSE Framework

A.1
Notation Summary

We illustrate all the notations used in the main paper and appendix in Table 3.

Table 3: Notations used in this paper and their corresponding meanings.

Notation
Explanation

A, a
Action space, action
S, s
State space, state
r
Reward
γ
Discount factor
T(·|·, ·)
Transition dynamics
h, H
Timestep h, Horizon H
π∗
Optimal policy in the online environments
πβ
Behavior policy generating the offline datasets
V (·)
State value function
Q(·, ·)
Action-state value function
D
Offline datasets
n(s, a)
Number of samples for (s, a) pairs in offline datasets
Bh
Bellman operator

ϕ(·, ·)
Feature representation of state and action
eϕ(·, ·)
Feature of state, action and confounder in Equation (2)
µ(·)
Feature representation of next state
d, d′
Dimensions of features for ϕ(·, ·) and µ(·)
Kµ
Feature matrix expanded from µ in Equation (5)
Cϕ, Cµ, Cβ
Feature regularity
Cs
Sparsity-related constant
κ
Restrictive eigenvalue constant
X
Feature Kronecker product
u, uc, uπ
Confounders
M, M(u)
Binary transition matrix (under certain confounders)
G
Causal graph
c
M, M(u)
Estimated causal matrix
βM, bβM
Optimal / estimated parameters in causal matrix
PAG(·)
Parental node in the causal graph G
ϵ
Exogenous noise in SCM by Definition 4
λ, λϕ, λµ
Spectrum regularizer weight in Equation (6)
σ
Standard deviation of exogenous noise in SCM
ξ
Accuracy level of the policy
K
Iterative update steps in BECAUSE

Eθ
Energy-based model
δ
Level of ‘high probability’
Γ(·, ·)
Uncertainty quantification function
E
δ-uncertainty quantifier set
λEBM
Regularizer weight for the ℓ2 norm in EBM

A.2
Equivalence of the Assumptions with Previous Works

A.3
Derivation of Definition 3

The node of this causal graph G =
0d×d
M
0d×d′
0d×d


contains two groups of entities: (1) The state

action abstraction ϕ(s, a), and (ii) the next state abstraction µ(s′). We denote ϕ(·, ·)(i) as the ith

17


---Page Break---
Table 4: Summary of assumptions and equivalent forms.

Assumption
Base and Equivalent Format

Invariant State Abstraction [25]
T(s′|s, a; u) = T(ϕ(s′)|ϕ(s), a) · pe(u′|s, u)

T(s′|s, a; u) ∝T(ϕ(s′)|ϕ(s), a)

Task Independence [24]
T(ϕ(s′)|ϕ(s), a) = T(s′
C|sC, sR, a)p(s′
R|sR)

T(ϕ(s(1))|ϕ(s(1)), a; u(1)) = T(ϕ(s(2))|ϕ(s(2)), a; u(2))

Invariant Action Effect [26]

Tonline(s′|s, a) = P

u T(s′|s, a, u)ˆp(u|s)

Toffline(s′|s, a) = P

u T(s′|s, a, u)ˆp(u|s)

Tonline(s′|s, a) ∝Toffline(s′|s, a)

Invariant Causal Graph
u(1) ̸= u(2), M(u(1)) = M(u(2))

T(s′|s, a; u(1)) = T(s′|s, a; u(2))

factor in the abstracted state action representations, and µ(·)(j) for the jth factor in the abstracted
state representations.

The source node of all the nodes G is ϕ(s, a)(i), which is the abstracted state-action representation,
and the sink node of all the edges in G is the to µ(s)′(i).

T(s′|s, a) =

ϕ(s, a)T
µ(s′)T  0d×d
M
0d×d′
0d×d

 
ϕ(s, a)
µ(s′)


= ϕ(s, a)T Mµ(s′),
(10)

Therefore, G is a bipartite graph, since there will be no edges between ϕ(s, a)(i), ϕ(s, a)(j), or
µ(s)(i), µ(s)(j).

Consequently, we show that G ∈DAG.

A.4
Derivation of Equation (5)

Definition 4 (Structured Causal Model). An SCM θ := (S, E) consists of a collection S of d
functions [11],
sj := fj(PAG(sj), ϵj), j ∈[d],
(11)

where PAG
j ⊂{s1, . . . , sd}\{sj} are called parents of xj in the Directed Acyclic Graph (DAG) G,
and E = {ϵi}d
i=1 are jointly independent. For instance, in continuous state and action space, we
parameterize the world model with joint Gaussian Distribution, i.e. ϵ ∼N(0, σIdd′).

We then use bilinear MDP to approximate the original likelihood function in Equation (4), i.e.

p(D; ϕ, µ, M) ∝
Y

(s,a,s′)∈D
exp(−∥µT (s′)K−1
µ
−ϕT (s, a)M∥2
2),
(12)

where Kµ := P

s′∈S µ(s′)µ(s′)T is an invertible matrix. Then we can apply an MLE in Equation (5).

In our BECAUSE algorithm, the optimization of the causal world model is conducted by solving
the regularized MLE problem in Equation (13). The biggest difference between BECAUSE and the
offline version of [14, 15] is that it aims to apply ℓ0 regression instead of ridge regression to estimate
matrix M:
Mn = arg max
M
[log p(D; ϕ, µ, M) −λ|M|]

= arg min
M

X

(s,a,s′)∈D
∥µT (s′)K−1
µ
−ϕT (s, a)M∥2
2

|
{z
}
World Model Learning

+
λ∥M∥0
| {z }
Sparsity Regularization

.
(13)

18


---Page Break---
A.5
Proof of Equation (7)

The derivation depends on the following re-weighting formula in [18]:

T(s′|s, a) = EpuT(s′|s, a, u)πβ(a|s, uπ)

Epuπβ(a|s, uπ)
.
(14)

Then we apply equation (14) to the decomposition in Equation (2) and Equation (3), which yields

T(s′|s, a) = Epu

T(s′|s, a, u)πβ(a|s, uπ)


Epuπβ(a|s, uπ)

= Epu

ϕ(s, a)T M(uc)µ(s′)πβ(a|s, uπ)


Epu[πβ(a|s, uπ)]

= ϕ(s, a)T
"
Epu

M(uc)πβ(a|s, uπ)


Epu[πβ(a|s, uπ)]

#

µ(s′)

= ϕ(s, a)T M(u)µ(s′),

(15)

where the last equality holds by letting M(u) := Epu[M(uc)πβ(a|s,uπ)]

Epu[πβ(a|s,uπ)]
.

B
Proof of Theorem 1

In this section, we provide the proof of the sub-optimality upper bound in Theorem 1. We first show
some useful definitions and lemmas in Section B.1. Armed with them, we provide the theoretical
results tailored for the causal discovery setting in Section B.2. Furthermore, we give a detailed proof
of the uncertainty set form in our causal discovery problems in Section B.3.

B.1
Preliminary

In this subsection, we first define the δ-uncertainty quantifier Γ, then we refer to the lemmas in the
previous literature to construct a suboptimality bound based on the defined uncertainty quantifier Γ.

First, we define the Bellman operator Bh, for some value function V : S 7→R, the Bellman operator
can be defined as:
(BhV )(s, a) = E[rh(sh, ah) + V (sh+1)|sh = s, ah = a].
(16)
Similarly, we denote the approximate Bellman operator of the empirical MDP constructed from the
offline dataset D as bBh for any h ∈[H].
Definition 5 (δ-Uncertainty Quantifier). We let {Γh}H
h=1, Γh : S × A 7→R to be a δ-uncertainty
quantifier with respect to data distribution PD if the event:

E =
n
|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)| ≤Γh(s, a), ∀(s, a, h) ∈S × A × [H]
o

satisfies PD(E) ≥1 −δ.

As we consider the offline model learning and planning, we define the model evaluation error at each
step h ∈[H] as

∀(s, a) ∈S × A :
ιh(s, a) = (Bh bVh+1)(s, a) −bQh(s, a),
(17)
where ιh is the error induced by the approximate Bellman operator, especially the transition ker-
nel based on D. We then identify the source of sub-optimality in our offline MBRL setting by
decomposing the sub-optimality error in Lemma 1.
Lemma 1 (Decomposition of Suboptimality [88]).

∀s ∈S :
V ∗
h (s) −V π
h (s) = −

H
X

h′=h
Eπ[ιh(sh′, ah′)|sh = s] +

H
X

h′=h
Eπ∗[ιh′(sh′, ah′)|sh = s]

+

H
X

h′=h
Eπ∗[⟨bQh′(sh′, ·), π∗(·, sh′) −bπ(·, sh′)⟩A|sh = s],

(18)

19


---Page Break---
where π is any learned policy, π∗is the optimal policy that maximizes the cumulative return as below:

π∗= arg max
π
Eπ
h
H
X

h′=1
γh′r(sh′, ah′)|sh
i
.

Based on this decomposition, we will get the basic form of sub-optimality error bound for general
offline RL settings in Lemma 2:

Lemma 2 (Suboptimality in standard MDP [88]). Suppose we have {Γh}H
h=1 as δ-uncertainty
quantifier. Under E defined in Equation (5), the suboptimality error bound by conservative planning
satisfies:

∀s ∈S :
V ∗
h (s) −V π
h (s) ≤2

H
X

h′=h
Eπ∗[Γh′(sh′, ah′)|s1 = s].

The basic form of sub-optimality bound in Lemma 2 involves an uncertainty quantifier Γh, which in
our case will be further replaced by an exact bound in our sparse matrix estimation problem of causal
discovery algorithms.

B.2
Proof of Theorem 1

The main results hold under the following two assumptions:

Assumption 2 (Existence of a core matrix given the feature embedding). For each (s, a) ∈S × A,
feature vectors ϕ(s, a) ∈Rd, µ(s) ∈Rd′ are approximated as a priori. Given a specific confounder
set u, there exists an unknown matrix M(u)∗∈Rd′×d such that,

T(s′|s, a, u) = ϕ(s, a)T M(u)µ(s′).
(19)

Assumption 3 (Feature regularity). We assume feature regularity [14, 15] for the following compo-
nents of the confounded bilinear MDP:

• ∀u, ∥M(u)∥2
F ≤CMd,

• ∀(s, a) ∈S × A, ∥ϕ(s, a)∥2
2 ≤Cϕd,

• ∀s′ ∈R|S|, ∥µT s′∥2 ≤Cµ∥s′∥∞, ∥µK−1
µ ∥2,∞≤C′
µ,

• ∀s, a, s′ ∈S × A × S, ∥ϕ(s, a)µ(s′)T ∥1 ≤Cµ.

where CM, Cϕ, Cµ, C′
µ are some universal constants.

Here, for any matrix X, ∥X∥2,∞:= maxi
qP

j X2
ij represents the operator 2 7→∞norm.

Proof pipeline.
Armed with the above assumptions, we turn to the bilinear MDP setting, which this
work focuses on. We shall develop the finite-sample analysis by specifying the main error term —-
δ-uncertainty quantifier Γ (see Lemma 2) for our time-homogeneous core matrix estimation problem
in the following lemma.

Lemma 3 (Uncertainty bound for Bilinear Causal Representation). Under the Assumption 2, 3 and
that T is an SCM (defined in 4), for the BECAUSE algorithm, for the ξ-optimal policy (V ∗
1 (es) −
V π
1 (es) ≤ξ), ∀0 ≤ξ ≤1, we have the δ-uncertainty set as:

EBECAUSE =
n
|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)|

≲min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	
s

log(1/δ)

n(s, a) , ∀(s, a, h) ∈A × S × [H]
o
,

where C1 is some universal constants.

20


---Page Break---
Armed with the above lemma, we complete the proof of Theorem 1 by showing that

V ∗
1 (es) −V π
1 (es) ≤2

H
X

h′=1
Eπ∗[Γh′(sh′, ah′)|s1 = es]

≲2

H
X

h=1
Eπ∗

"

min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	
s

log(1/δ)
n(sh, ah) | s1 = es

#

(20)
This concludes the proof of Theorem 1.

B.3
Proof of Lemma 3

The key to proving Theorem 1 is to prove Lemma 3. The proof pipeline of Lemma 3 is illustrated
below. In Step 1, we derive the estimation of the causal transition matrix M in BECAUSE as a
sparsity regression problem. In Step 2, we decompose the error terms within δ-uncertainty set into
two parts: (a) error due to the under-explored dataset, (b) error due to optimization error in the
structured causal model. Then we bound both error terms in Step 3 and Step 4, respectively. Finally,
in Step 5, we sum up all the results and derive the form of δ-uncertainty quantifier which will lead to
our final results in Theorem 1.

Step 1: deriving the output model of BECAUSE.
Recalling the original optimization problem in
equation (13) to estimate the core matrix:

c
M = arg max
M
[log p(ϕ, µ, M) −λ|M|]

= arg min
M

X

(s,a,s′)∈D
∥µ(s′)T K−1
µ
−ϕ(s, a)T M∥2
2

|
{z
}
Model Learning

+
λ∥M∥0
| {z }
Sparsity Regularization

.
(21)

This part of derivation aims to transform the above estimation problem into a linear regression
problem, with the regression data pairs (X, T) and some unknown parameters β associated with
mask M to be estimated. Eventually, we’ll derive the representation of each part of βM, X, T, and
eventually reach the following form:

min
βM
X

(si,ai,s′
i)∈D
[∥Tπβ(s′
i | si, ai) −XiβM∥2
2 + λ∥βM∥0].
(22)

We define each component of this target form of ℓ0 regression as follows:

• For unknown parameters βM: We first define βM ∈[0, 1]dd′ as a column dimensional vector
consisting of all the entries in time-homogenous causal matrix M, where βM
i
denotes the i-th
entry of βM. Besides, we define βM
D as the true core matrix given some offline dataset D and
corresponding data pairs Tdata, Xdata that satisfies Tdata = βM
D Xdata + ϵ.
• For dataset D: Recall the transition pairs in the offline dataset D = {si, ai, s′
i}1≤i≤n. Here, n
represents the sample size over certain state-action pairs in the rollout data by some behavior
policy πβ. For simplicity, we denote n ≜n(s, a) in the following derivation, which is mentioned
in Section 2.1.
• For regression target Tπβ: Then, we introduce the following transition targets Tπβ induced by
the offline dataset D sampled with behavior policy πβ:

Tπβ(s′|s, a) : =

P

(si,ai,s′
i)∈D 1(si = s, ai = a, s′
i = s′)
P
(si,ai,s′
i)∈D 1(si = s, ai = a)

=
1
n(s, a)

X

(si,ai,s′
i)∈D
1(si = s, ai = a, s′
i = s′).
(23)

Under the n finite samples in the offline dataset, we assume that Tπβ ∼N(E[Tπβ], σ2In). The
above definition specifies the regression target in the ℓ0 regression problem, and we denote

21


---Page Break---
Tπβ = [Tπβ(s′
1|s1, a1), · · · , Tπβ(s′
n|sn, an)]T ∈Rn as the empirical transition probabilities of
certain transition pairs in the offline data D = {si, ai, s′
i}1≤i≤n.
• For regression data X: Next, we need to specify the data X in the regression problem. We
denote the i-th row of X as the i-th sample in the offline transition pairs Xi ∈D, which is a
vector of Kronecker product between ϕ(si, ai) ∈Rd and normalized µ(s′
i)
Cµ
∈Rd′ (without loss
of generality, we assume Cϕ = 1 and only need to normalize µ(s′
i) by Cµ):

Xi = ϕ(si, ai) ⊗µ(s′
i)
Cµ

= 1

Cµ
[ϕ(si, ai)(1)µ(s′
i)(1), ϕ(si, ai)(1)µ(s′
i)(2), · · · , ϕ(si, ai)(d)µ(s′
i)(d′)]T
(24)

As a result, Xi ∈Rdd′, since there are in all n samples in offline dataset, X ∈Rn×dd′ is the
dataset-dependent matrix with all n rows of samples, and d and d′ are the latent dimension
of ϕ and µ, respectively. Based on the feature regularity criteria in Assumption 3, we have
∥Xi∥2 ≤∥Xi∥1 ≤1, ∥X∥∞≤1.

The prior work [14] estimate the transition kernel of a bilinear MDP using the following ridge
regression:
min
M E(s,a,s′)∈D∥µ(s′)T K−1
µ
−ϕ(s, a)T M∥2
2 + λ∥M∥2.
(25)

In this paper, in order to promote the sparsity of the matrix M, we introduce the ℓ0 regularization
term and arrive at the following optimization problem:

min
βM
X

(si,ai,s′
i)∈D
[∥µ(s′
i)T K−1
µ µ(s′
i) −ϕ(si, ai)T Mµ(s′
i)∥2
2 + λ∥βM∥0]

→min
βM
X

(si,ai,s′
i)∈D
[∥Tπβ(s′
i | si, ai) −XiβM∥2
2 + λ∥βM∥0] =: bβM
D ,
(26)

where we denote the solution associated with the offline dataset D as bβM
D . Here, we use the empirical
version constructed by the finite samples in offline dataset D.

Given the goal-conditioned reward setting, for a single episode s ∼τ, r(s, a; g) = 1 if and only if
s = g, otherwise r(s, a; g) = 0, as is specified in Section 2.1. Since we are essentially predicting the
probabilities (normalized to a sum of 1) of whether the next state is the goal state, i.e. P

s∈S bV (s) = 1.
Therefore, we have ∥bV (·)∥1 ≤1.

As is denoted by Equation (23), for the specific offline dataset collected by some behavior policies πβ,

we have the regression target Tπβ(s′|s, a) =

P

(si,ai,s′
i)∈D 1(si=s,ai=a,s′
i=s′)
P

(si,ai,s′
i)∈D 1(si=s,ai=a)
, and the corresponding

features induced by the dataset Xi = ϕ(si, ai) ⊗µ(s′
i)
Cµ . We have the following equations hold:

Tπβ = XβM
D + ϵ,
(27)

where βD is the true underlying transition mask given the offline dataset D, ϵ ∼N(0, σ · In) is some
exogenous noise of the transition model.

Specifically, for the regression problem, we transform the original trajectory dataset D as [X, Tπβ],
as the representation of transition pairs rolled out by the behavior policy πβ. Similarly, we can
define some ’well-explored dataset’ D∗, which is an infinite dataset rollout by the behavior policy
πβ, D∗= {si, ai, ri}∞
i=0, similar to the definition of regression target T in Equation (23) and
representation data X in Equation (24), we have [X, E[Tπβ]] as the regression pairs with access to
the true transition probability distribution for all the state action pairs (s, a). Here X ∈Rn×dd′ is the
regression data defined in equation (22). Recalling the definition in Equation (23), we can further
build the regression target under well-explored dataset D∗as:

E[Tπβ(s′|s, a)] =

P

(si,ai,s′
i)∈D∗1(si = s, ai = a, s′
i = s′)
P
(si,ai,s′
i)∈D∗1(si = s, ai = a)

= Es′∼T (·|s,a)[1(si = s, ai = a, s′
i = s′)]

(28)

22


---Page Break---
We denote E[Tπβ] =

E[Tπβ(s′
1|s1, a1)], · · · E[Tπβ(s′
n|sn, an)]
T ∈Rn. In practice, with finite
sample size n, we have E[Tπβ(s′|s, a)] = T(s′|s, a) + ϵ = XβM
D∗+ ϵ. In addition, we also introduce
a vector form T ∈R|S||A|×|S| so that ∀(s, a) ∈S × A, we have

T(·|s, a) =

T(s′
1|s, a)
T(s′
2|s, a)
· · ·
T(s′
|S||s, a)
T ∈R|S|,

where the state space is denoted as S = {s′
1, · · · , s|S|} are all possible states. Similar to the
Kronecker product we define for X in equation (26), we define X in a matrix form for any state-action
pair (s, a):

X(·|s, a) =

ϕ(s, a) ⊗µ(s′
1)
Cµ
, · · · ϕ(s, a) ⊗
µ(s′
|S|)

Cµ

T ∈R|S|×dd′.

In addition, we let X(s′|s, a) ∈R1×dd′ denote the s′-th row of X(·|s, a) associated with the state
s′ ∈S. Consequently, the estimated transition kernel can be expressed as follows:

bT(·|s, a) = ϕT (s, a)c
Mµ(·) = X(·|s, a)bβM
D .

Step 2: decomposing the term of interest.
To begin with, recalling the definition of Bellman
operator Bh in Equation (16) and applying Hölder’s inequality, the term of interest for any time step
1 ≤h ≤H and state-action pair (s, a) ∈S × A can be controlled as

|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)| ≤|⟨bT(·|s, a) −T(·|s, a), bVh+1⟩|

≤∥bT(·|s, a) −T(·|s, a)∥∞∥bVh+1∥1

≤∥bT(·|s, a) −T(·|s, a)∥∞,

(29)

where the first inequality is held given our goal-conditioned reward formulation in section 2.1. To
continue, we have

|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)| ≤∥bT(·|s, a) −T(·|s, a)∥∞

= ∥X(·|s, a)bβM
D −X(·|s, a)βM∥∞
(30)

Here, we recall bβM
D represents the parameter vector in the estimated causal masks based on the offline
dataset D sampled by πβ. Similarly, we denote bβM
D∗as the estimated causal mask outputted from
equation (26) based on the infinite dataset D∗generated by the behavior policy πβ. Then, we can
further control equation (30) as

∥X(·|s, a)bβM
D −X(·|s, a)βM∥∞= ∥X(·|s, a)bβM
D −X(·|s, a)bβM
D∗+ X(·|s, a)bβM
D∗−X(·|s, a)βM∥∞

≤∥X(·|s, a)[bβM
D −bβM
D∗]∥∞+ ∥X(·|s, a)[bβM
D∗−βM]∥∞

≤∥X(·|s, a)∥∞∥bβM
D −bβM
D∗∥∞+ ∥X(·|s, a)∥∞∥bβM
D∗−βM∥∞

≤∥bβM
D −bβM
D∗∥∞
|
{z
}
(a)

+ ∥bβM
D∗−βM∥∞
|
{z
}
(b)
(31)
Here the last inequality comes from the fact that

∥X(·|s, a)∥∞= max
i∈|S|

X

j∈[dd′]
|X(·|s, a)ij| = max
i∈|S| ∥X(·|s, a)i∥1

= max
i∈|S| ∥ϕ(s, a)µ(s′
i)T ∥1 ≤Cµ

Cµ
= 1

based on the definition of X in equation (24) and assumption 3. Here (a) comes from the mismatch
error between the demonstrated offline dataset and some optimal rollout datasets. And (b) comes
from the error of the ℓ0 optimization of causal masks given the existence of exogenous noise σ
defined by SCM in Definition 4. We will control them separately in the following.

23


---Page Break---
Step 3: Controlling term (a).
We need to consider the optimization process in the original
regression problem in equation (22) to fully understand the difference between bβM
D and bβM
D∗, where
the only difference is that the latter uses a perfect dataset with infinite samples. The optimization
problem we target (cf. equation (26)) can be solved by the iterative hard thresholding algorithm (IHT)
proposed by [89] IHT offers an iterative solution for the ℓ0 regression problem, armed with a hard
thresholding operator as below:

[gλ(β)]j =
max{0, βj −λ}
if βj > λ
0
if βj ≤λ, j = 1, · · · , dd′.
(32)

We denote bβM
D (i) as the estimated causal mask parameters after i-th iterations with dataset D.
Similarly, we denote bβM
D∗(i) as the estimation after i-th iterations with dataset D∗. We initialize the
graph to be a full graph regardless of the datasets (D∗or D) used in the optimization process, leading
to bβM
D∗(0) = bβM
D (0) = 1 ∈Rdd′.

Recall that X ∈Rn×dd′, Tπβ ∈Rn, E[Tπβ] ∈Rn and bβM
D is [0, 1]dd′ based on the definition in the
original ℓ0 optimization problem in equation (22), equation (23) and equation (28). The update rules
of using either the dataset D or D∗can be written as:

bβM
D (i) = gλ(bβM
D (i −1) + ηXT [Tπβ −X bβM
D (i −1)])

βM
D∗(i) = gλ(bβM
D∗(i −1) + ηXT [E[Tπβ] −X bβM
D∗(i −1)]).
(33)

Note that the difference between the two parameters bβM
D and bβM
D∗essentially relies on the difference
between two pairs of transition kernel estimation datasets [X, E[Tπβ]] and [X, Tπβ]. It is easily
verified that the hard-thresholding operator [gλ(·)]i defined in equation (32) is L = 1-Lipschitz.
According to [90], we can set the learning rate η ≤1

L = 1 here. Using the Lipschitz property, at each
iterative update step i, we can control the difference between the estimated parameters obtained by
using D or D∗as

∥bβM
D (i) −bβM
D∗(i)∥2

= ∥gλ
 bβM
D (i −1) + ηXT [Tπβ −X bβM
D (i −1)]


−gλ
 bβM
D∗(i −1) + ηXT [E[Tπβ] −X bβM
D∗(i −1)]

∥2

≤∥
 bβM
D (i −1) + ηXT [Tπβ −X bβM
D (i −1)]

−
 bβM
D∗(i −1) + ηXT [E[Tπβ] −X bβM
D∗(i −1)]

∥2

≤∥(Idd′ −ηXT X)[bβD(i −1) −bβD∗(i −1)] + ηXT [Tπβ −E[Tπβ]]∥2

≤∥Idd′ −ηXT X∥2∥bβD(i −1) −bβD∗(i −1)∥2 + η∥X∥2∥Tπβ −E[Tπβ]∥2

≤∥bβD(i −1) −bβD∗(i −1)∥2 + η∥Tπβ −E[Tπβ]∥2,
(34)
Here, Idd′ represent the identity matrix of size dd′ × dd′, the last inequality holds based on the
fact that ∥X∥2 ≤1 defined in equation (24). Since η ≤1

L = 1, as a result, ∥Idd′ −ηXT X∥2 =
max(1 −ηλ(XT X)) ≤1. We perform IHT for sufficient K > 0 iterations and output the last step
estimation as our solutions for either the offline dataset D (we use for practical optimization) or
the perfect dataset D∗. In practice, for any dataset such as D and any accuracy level 0 ≤ξ ≤1,
we have the output of IHT gradually converges to the optimal solution bβM
D of the problem in

equation (26). Namely, after at most K ≃log

∥M∥0

ξ

steps [89, Corollary 1], the output satisfies
bβM
D −bβM
D (K)

2 ≤ξ

4. Similarly, based on the perfect dataset D∗, the output of IHT gradually

converges to bβM
D∗at the same rate. Consequently, the term of interest (a) can be bounded recursively
as:

24


---Page Break---
∥bβM
D −bβM
D∗∥∞=
bβM
D (K) −bβM
D∗(K) +

bβM
D −bβM
D (K)

+

bβM
D∗−bβM
D∗(K)

∞

≤
bβM
D (K) −bβM
D∗(K)

∞+ ξ

4 + ξ

4

≤
bβM
D (K) −bβM
D∗(K)

2 + ξ

4 + ξ

4

≤∥bβM
D (0) −bβM
D∗(0)∥2 + Kη∥Tπβ −E[Tπβ]∥2 + ξ

2

= Kη∥Tπβ −E[Tπβ]∥2 + ξ

2

≲η log
 ∥M∥0

ξ

∥Tπβ −E[Tπβ]∥2 + ξ

2,

(35)

where the second inequality holds by recursively applying equation (34) to K, K −1, · · · , 0 iterations,
the last equality is due to the fact that bβM
D (0) = bβM
D∗(0), and the last inequality holds by setting
K = log( ∥M∥2

ξ
).

Now the remaining of the proof will focus on controlling ∥Tπβ −E[Tπβ]∥∞. Recall that we have
defined the regression target in equation (23) and equation (28):

Tπβ(s′|s, a) =
1
n(s, a)

n(s,a)
X

i=1
1(si = s, ai = a, s′
i = s′),

E[Tπβ(s′|s, a)] = Es′∼T (·|s,a)[1(si = s, ai = a, s′
i = s′)].

Proposition 1 (Well-explored dataset). With the offline dataset D of in total n samples with n(s, a)
samples generated independently conditioned on any (s, a). For any 0 < δ < 1, with probability at
least 1 −δ, one has

∀(s, a) ∈S × A :
∥Tπβ(·|s, a) −E[Tπβ(·|s, a)]∥2 ≤Cβ

s

|S|
n(s, a) log
|S||A|

δ


,

for some universal constant Cβ.

The above proposition can be directly proved by applying [91, Lemma 17] over all (s, a) ∈S × A:

max
(s,a)∈S×A ∥Tπβ(·|s, a) −E[Tπβ(·|s, a)]∥1 ≤

s

14|S|
n(s, a) log
2|S||A|

δ


.
(36)

As a result, we can further extend the results in equation (35) to bound the term (a) as follows:

∥bβM
D −bβM
D∗∥∞≤Kη∥Tπβ −E[Tπβ]∥2 + ξ

2

≲ηCβCµ log
 ∥M∥0

ξ

s

|S|
n(s, a) log
|S||A|

δ


+ ξ

2

≤CβCµ log
 ∥M∥0

ξ

s

|S|
n(s, a) log
|S||A|

δ


+ ξ

2

≜C1 log
 ∥M∥0

ξ

s

|S|
n(s, a) log
|S||A|

δ


+ ξ

2,

(37)

where C1 = CβCµ is some constant that is related to the feature regularity, ξ is the level of error
tolerance in the cumulative value returns, in our setting, 0 ≤ξ ≤1.

25


---Page Break---
Step 4: Controlling term (b).
For (b) in Equation (31), which is ∥bβM
D∗−βM∥∞, we are interested
in what is the optimization error given finite well-explored offline dataset D∗. Here the optimization
error mainly originates from the Gaussian noise in the SCM formulation in Definition 4. Yet our
causal discovery module, i.e. an ℓ0 estimator, will always encounter some estimation error induced
by the exogenous noise.

Firstly, we have the bounded relationship between ℓ2 and ℓ∞norm:

∥bβM
D∗−βM∥∞≤∥bβM
D∗−βM∥2,

then we can analyze the error bound for ℓ0 regression in the sense of ℓ2 norm.

The derivation below generally follows the ℓ0 regularized linear regression bound in [92]. Based on
Assumption 2 and 3, there exists M, we denote this optimal solution in the vector form as βM.

Besides the aforementioned optimization error in the iterative thresholding update process, according
to the SCM in Definition 4 and Assumption 2 and Proposition 1, we denote a finite subset of observed
transition probabilities of the well-explored data E[Tπβ] with size n: Tobv ∈Rn. By definition above,
we can then assume the finite-sample regression target T is generated by causal features of transition
pairs (denoted by X = Xπ∗= Xπβ, X ∈Rn×dd′) and the ground-truth causal mask (represented by
βM ∈Rdd′) with the following equation:

Tobv ≜E[Tπβ] = T + ϵ = XβM + ϵ.
(38)

Here ϵ ∼N(0, σIn) is the independent exogenous noise defined in Definition 4. We’ll then use the
above equation to bound term (b) in equation (31).

Specifically, we solved this ℓ0 regression problem with its bounded form as follows:

min
βM ∥Tobv −XβM∥2
2,
s.t.∥βM∥0 ≤s.
(39)

In BECAUSE, we select the sparsity level s ≈∥βM∥0 = ∥M∥0 ≤dd′.

For simplicity, we denote the approximate solution bβM
D∗in Equation (39) as bβM. By the virtue of
optimality of the solution βM in Equation (39), we find that

∥Tobv −XβM∥2
2 ≤∥Tobv −X bβM∥2
2,
(40)

then by expanding and shifting the terms, we can derive the basic inequality for the ℓ0 estimator
above:
∥Tobv −XβM∥2
2 ≤∥(Tobv −XβM) + (XβM −X bβM)∥2
2

(38)
==⇒∥ϵ∥2
2 ≤∥ϵ + (XβM −X bβM)∥2
2

= ∥ϵ∥2
2 + ∥X bβM −XβM∥2
2 + 2⟨ϵ, Xββ∗−X bβM⟩

Since both bβM and βM are s-sparse, the vector bβM −βM is at most 2s-sparse. We denote the set
of all 2s-sparse dd′-dimensional vector set as Tdd′(2s), we denote v = I(bβM −βM = 0) as an
indicator vector, vi = 0 if and only if bβM
i
−βM
i , ∀i ∈[dd′].

By shifting the terms, we use the sub-Gaussian assumption and further use Hölder’s inequality to get
the following results:

1
n(s, a)∥X bβM −XβM∥2
2 ≤
2
n(s, a)⟨XT ϵ, bβM −βM⟩

≤
2
n(s, a)∥bβM −βM∥2
sup
v∈Tdd′(2s)
⟨v, XT ϵ⟩

≤2∥bβM −βM∥2 sup
|S|=2s
∥(XT ϵ)S

n(s, a) ∥2

≤2∥bβM −βM∥2σ

s

2s log(dd′/δ)

n(s, a)
,

(41)

26


---Page Break---
where ϵ ∼N(0, σ2In) is the exogenous noise variable in the SCM in Definition 4.

Then by simply applying restricted eigenvalue (RE) condition, for some κ(S) > 0, we have

κ(S)∥bβM −βM∥2
2 ≤
1
n(s, a)∥X(bβM −βM)∥2
2 ≤2∥bβM −βM∥2σ

s

2s log(dd′/δ)

n(s, a)

Therefore, we have:

∥bβM −βM∥2
2 ≤2σ
√

2s
κ(S)

r

log(dd′/δ)

n
≜Csσ

s

∥M∥0 log(dd′/δ)

n(s, a)
(42)

with probability at least 1 −δ, which bounds term (b) in Equation (31).

Step 5: Summing up the results
Summarizing both bounds for terms (a) and (b) in Equation (31),
we will get the following bounds with probability 1 −δ:

|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)| ≤∥bβM
D∗−βM∥∞+ ∥bβM
D −bβM
D∗∥∞

≤C1 log
 ∥M∥0

ξ

s

|S|
n(s, a) log
|S||A|

δ


+ Csσ
p

∥M∥0

s

log(dd′/δ)

n(s, a)
+ ξ

2

≲min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	
s

log(1/δ)

n(s, a)
(43)

Therefore, we complete the proof of Lemma 3 by showing that for all (s, a) ∈A × A, h ∈[H],

EBECAUSE

(

|(bBh bVh+1)(s, a) −(Bh bVh+1)(s, a)|

≲min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	
s

log(1/δ)

n(s, a)

)
(44)

The final bound of the term (b) includes a dependency of O( 1
√n) and logarithm of dimensionality dd′

in the estimated transition matrix. It also incurs a dependency of error tolerance ξ or SCM’s noise
level σ square root of sparsity level √s =
p

∥M∥0.

So far, we prove the following bound in theorem 1:

V ∗
1 (es) −V π
1 (es) ≤min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗

"s

log(1/δ)
n(sh, ah) | s1 = es

#

,

(45)

In order to achieve ξ-optimal policy such that V ∗
1 (es) −V π
1 (es) ≲ξ, the RHS needs to satisfy:

min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗

"s

log(1/δ)
n(sh, ah) | s1 = es

#

≲ξ
(46)

We first multiply
q

min(s,a,h)∈S×A×[H] Eπ⋆
n(sh, ah) | s1 = es

, and then take the square on both
sides. We have:

min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗





s

log(1/δ) min(s,a,h)∈S×A×[H] n(sh, ah)


n(sh, ah)
| s1 = es





≲ξ
r

min
(s,a,h)∈S×A×[H] Eπ⋆
n(sh, ah) | s1 = es


(47)

27


---Page Break---
Given the definition, LHS in Equation (47) satisfies:

LHS ≲min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗
hp

log(1/δ) | s1 = es
i
(48)

To ensure the satisfaction of ξ-optimal policy, we thus would like the RHS of Equation (47) satisfies:

RHS = ξ
r

min
(s,a,h)∈S×A×[H] Eπ⋆
n(sh, ah) | s1 = es


≳min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	 H
X

h=1
Eπ∗
hp

log(1/δ) | s1 = es
i

= min

C1 log
 ∥M∥0

ξ
p

|S|, Csσ
p

∥M∥0)
	
H
p

log(1/δ)

(49)

Consequently, we can shift the terms and get the sample complexity bound for the ξ-optimal policy,
∀0 ≤ξ ≤1:

min
(s,a,h)∈S×A×[H] Eπ⋆
n(sh, ah) | s1 = es

≳
min

C2
1 log2   ∥M∥0

ξ

|S|, C2
sσ2∥M∥0
	
· H2 log(1/δ)

ξ2
,

(50)

C
Additional Experiments Details

In this section, we provide additional experiment results and algorithm implementation details.

C.1
Implementation of Causal Discovery

We implement the causal discovery primarily based on Equation (5). However, in practice, how to
control the coefficient before the sparsity regularization terms is crucial to the final performance. In
practice, instead of controlling λ, we use p-value as a threshold to determine the following conditional
independence:
ϕ(s, a)(i) ⊥⊥µ(s′)(j) | ϕ(s, a)−(i).
(51)

Here ϕ(·, ·)(i) means that this element is the ith factor in the abstracted state action representation,
similar to µ(·, ·)(j), ϕ(s, a)−(i) means all the other factors in the representation except for the ith
factor.

If the p-value based on the above conditional independence test is less than a threshold, we can
remove the edge by setting Mij = 0. Please refer to the Appendix Table 13 for the selection of
threshold in each environment.

C.2
Training Details of Energy-based Model

We train the EBM according to the margin loss in Equation (8). In practice, we attach Tanh() to the
output layer to clip the unnormalized score between -1 and +1. The energy networks take in both
conditions and samples, then concatenate them together and sent it into MLP encoders. The detailed
hyperparameters of EBM are listed in Table 12.

In the vanilla EBM, people follow the Langevin dynamics to effectively sample the negative samples.
Here, as we discover the causal mask and identify the causal representation in the model learning
stage, we find that we can use these representations in both the energy networks and the sampling
process to get some effective negative samples, which is similar to the practice of augmentation of
causality-guided counterfactual data [77].

The interesting trick we employ here is the way to get our negative samples by mixing the latent
factors from offline data. For example, for a positive sample array

28


---Page Break---
x+ =





µ(s′
1)(1)
µ(s′
1)(2)
· · ·
µ(s′
1)(d′)

µ(s′
2)(1)
µ(s′
2)(2)
· · ·
µ(s′
2)(d′)
· · ·
· · ·
· · ·
· · ·
µ(s′
n)(1)
µ(s′
n)(2)
· · ·
µ(s′
n)(d′)



,
(52)

with conditions:

y =





ϕ(s1, a1)(1)
ϕ(s1, a1)(2)
· · ·
ϕ(s1, a1)(d)

ϕ(s1, a1)(1)
ϕ(s1, a1)(2)
· · ·
ϕ(s1, a1)(d)
· · ·
· · ·
· · ·
· · ·
ϕ(s1, a1)(1)
ϕ(s1, a1)(2)
· · ·
ϕ(s1, a1)(d)



.
(53)

Here, si, ai, s′
i denotes the timestep of the offline samples. ϕ(·, ·)(i) means this element is the ith

factor in the abstracted state action representation, similar to the µ(·, ·)(i)

as we already get the corresponding causal representation µ(s′) that is semantically meaningful,

we can mix the columns to create useful counterfactual negative samples

x−
countefactual =





µ(s′
1)(1)
µ(s′
2)(2)
· · ·
µ(s′
n−1)(d′)

µ(s′
2)(1)
µ(s′
1)(2)
· · ·
µ(s′
n)(d′)
· · ·
· · ·
· · ·
· · ·
µ(s′
n)(1)
µ(s′
2)(2)
· · ·
µ(s′
1)(d′)



.
(54)

These counterfactual negative samples seem to effectively use the causal representation and speed up
the training, as we show in Figure 7.

Random
Causal EBM

T=0                    T=100                  T=200                   T=300                  T=400           

T=500                  T=600                   T=700                  T=800                  T=900                   

Random
Causal EBM

Next State

GT

Figure 7: Comparison of the convergence speed in EBM training. Compared to random negative
samples, our approach enjoys a higher rate of convergence empirically.

C.3
Additional Mismatch Analysis

To evaluate the significance of the objective mismatch effect at three different levels of offline datasets
in Unlock environments, we collect 5,000 episodes in each of the Unlock environments for each
method. Then we evaluate the mismatch via the following two metrics.

• We conduct hypothesis testing via Mann-Whitney U Test [93], with Null hypothesis
H0 : Lmodel(τpos) < Lmodel(τneg),

• To understand the exact difference in model loss between two groups of samples, we compute
their Wasserstein-1 distance in the episodic model loss between the trajectories with positive and
negative rewards, i.e. W1(τpos∥τneg).

We report the results of p-value and the W1 distance of two groups of model loss samples in the
following table.

29


---Page Break---
Table 5: The comparison results of the p-value and the W1 distance (×10−4) between MOPO and
BECAUSE. Bold means the better.

Methods
Unlock-Expert
Unlock-Medium
Unlock-Random
p-value (↓)
W1 Dist (↑)
p-value (↓)
W1 Dist (↑)
p-value (↓)
W1 Dist (↑)

MOPO
6.5 × 10−5
0.7
1.0
1.1
1.0
N.A.
Ours
≈0
3.1
≈0
3.0
0.9
< 0.1

C.4
Additional Experiment Results

We report the results of the task-wise performance of all baselines in the main experiment and variants
in the ablation studies in Table 6, 7, 8, and 9.

Table 6: Success rate (%) for 18 tasks in three different environments. We evaluate the mean and
95% confidence interval given by the t-test of the best performance among 10 random seeds, as well
as the p-value between the overall performance. Bold is the best.

Env
ICIL
CCIL
TD3+BC
MOPO
GNN
CDL
Denoised
IFactor
MnM
Delphic
Ours

Lift-I-R
14.3±9.9
10.0±11.4
9.7±5.4
24.3±2.9
22.1±3.6
33.8±5.0
20.0±4.0
24.0±2.8
16.3±2.8
20.2±3.1
19.2±0.6
Lift-O-R
8.5±4.7
0.0±0.0
1.3±1.0
10.2±1.6
13.3±2.3
16.0±4.7
15.5±3.8
21.2±3.0
14.2±2.2
17.7±2.8
21.4±3.9

Unlock-I-R
3.4±1.1
2.2±0.8
4.39±0.9
21.5±1.9
11.7±2.1
6.6±0.5
6.9±0.7
8.1±1.1
8.6±1.3
14.4±1.1
32.7±2.8
Unlock-O-R
11.6±4.0
13.5±3.7
13.3±3.0
16.6±1.3
12.1±1.5
7.6±1.0
7.0±0.8
8.0±1.1
9.0±1.0
12.2±1.1
27.6±2.0

Crash-I-R
11.4±4.3
19.5±10.5
9.1±6.6
32.7±4.9
11.7±0.8
39.7±4.6
32.0±2.8
31.3±4.4
16.0±2.2
17.4±3.6
59.4±6.1
Crash-O-R
2.8±1.8
5.7±3.0
0.9±0.6
10.0±2.0
3.7±0.4
10.8±1.9
10.8±2.1
11.0±3.6
4.1±0.6
5.4±1.1
19.7±1.4

Lift-I-M
54.0±13.3
44.0±20.8
26.6±14.8
39.8±5.3
27.5±5.3
32.9±5.7
35.3±4.8
41.0±5.7
28.7±1.8
24.0±2.3
59.5±4.4
Lift-O-M
46.8±15.2
20.0±21.9
13.0±1.1
31.9±2.8
25.8±1.8
26.0±4.2
27.0±3.2
30.2±2.7
24.3±2.3
18.3±2.6
32.3±4.9

Unlock-I-M
4.8±0.9
4.7±1.3
5.4±1.0
84.8±5.1
17.5±2.6
29.7±4.4
12.9±1.5
37.7±5.5
37.6±4.9
74.1±1.5
98.0±4.9
Unlock-O-M
12.9±2.8
14.1±2.0
19.2±2.5
39.5±4.7
16.2±1.8
20.5±3.9
10.8±0.8
21.5±2.7
27.0±3.7
51.7±2.3
68.8±1.5

Crash-I-M
35.5±9.9
24.5±12.2
16.3±9.0
47.7±7.3
11.5±1.1
63.5±4.0
63.8±4.0
45.5±5.0
18.4±2.9
58.2±2.2
90.4±1.8
Crash-O-M
11.7±3.5
7.9±4.4
5.6±3.1
17.3±2.3
3.8±0.4
20.0±2.3
20.3±1.6
16.0±2.0
6.3±1.9
22.2±1.7
20.3±1.9

Lift-I-E
73.8±17.0
86.7±15.7
44.3±12.2
82.4±6.7
63.3±0.0
71.0±5.5
74.2±5.5
98.0±3.7
33.3±3.0
53.5±6.3
92.8±1.2
Lift-O-E
54.1±26.1
80.0±18.9
41.6±16.6
72.4±1.9
60.8±0.6
63.1±5.1
64.0±4.4
91.7±5.7
29.0±4.1
49.5±3.1
93.7±5.9

Unlock-I-E
11.6±3.2
13.7±3.1
15.6±2.2
88.8±4.6
15.3±1.6
73.2±2.8
50.3±3.0
59.3±2.6
60.7±2.2
83.2±1.2
97.4±1.0
Unlock-O-E
22.4±5.0
35.4±6.0
41.6±6.2
39.9±4.4
13.8±1.4
41.3±4.1
35.7±2.3
29.5±3.5
38.7±2.0
54.4±1.9
82.1±6.5

Crash-I-E
35.7±9.8
26.8±7.2
26.0±14.4
58.5±4.7
11.2±0.9
63.7±2.8
69.3±3.9
52.0±5.3
10.2±1.1
57.0±1.2
95.3±1.3
Crash-O-E
11.3±3.8
12.3±3.5
6.2±1.8
14.8±2.1
3.5±0.9
18.8±1.8
20.7±2.2
15.6±2.3
3.9±0.6
16.5±1.7
20.7±3.2

Overall-I
27.2
25.8
17.5
53.4
21.0
44.7
40.5
44.1
25.5
44.7
73.3
Overall-O
20.2
19.9
14.6
28.1
17.0
24.9
23.5
27.2
17.4
27.5
43.0

30


---Page Break---
Table 7: p-values of different methods (each has 10 random trials) against BECAUSE in various
environments. Under the significance level 0.05, we mark all the baseline results that are significantly
lower than BECAUSE as green, and the rest as red. We can see that BECAUSE significantly
outperforms 10 baselines in 18 tasks in 91.1% of the experiments (164 out of total 180 pairs of
experiments).

Env
ICIL
TD3+BC
MOPO
GNN
CDL
Denoised
IFactor
CCIL
MnM
Delphic

Lift-I-random
0.001
0.000
0.001
0.000
0.000
0.000
0.001
0.001
0.000
0.000
Lift-O-random
0.000
0.000
0.000
0.001
0.033
0.013
0.464
0.000
0.001
0.052
Unlock-I-random
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Unlock-O-random
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-I-random
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-O-random
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Lift-I-medium
0.198
0.000
0.000
0.000
0.000
0.000
0.000
0.067
0.000
0.000
Lift-O-medium
0.966
0.000
0.438
0.009
0.022
0.032
0.209
0.125
0.003
0.000
Unlock-I-medium
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Unlock-O-medium
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-I-medium
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-O-medium
0.000
0.000
0.018
0.000
0.412
0.500
0.001
0.000
0.000
0.943
Lift-I-expert
0.017
0.000
0.004
0.000
0.000
0.000
0.993
0.203
0.000
0.000
Lift-O-expert
0.004
0.000
0.000
0.000
0.000
0.000
0.297
0.027
0.000
0.000
Unlock-I-expert
0.000
0.000
0.001
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Unlock-O-expert
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-I-expert
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
0.000
Crash-O-expert
0.000
0.000
0.002
0.000
0.133
0.500
0.005
0.000
0.000
0.011

Table 8: Success rate (%) for 18 tasks in three different environments. We evaluate the mean and
95% confidence interval of the test performance among 10 random seeds. Bold means the best.

Env
BECAUSE
BECAUSE-Optimism
BECAUSE-Linear
BECAUSE-Full

Lift-I-random
33.8±5.0
23.2±3.1
16.5±1.6
22.2±6.6
Lift-O-random
21.4±3.9
15.3±1.5
8.9±4.8
18.2±5.3

Unlock-I-random
32.7±2.8
31.2±2.4
20.7±1.7
10.7±0.8
Unlock-O-random
27.6±2.1
26.3±1.7
24.0±2.7
9.3±0.6

Crash-I-random
59.4±6.2
49.9±9.2
54.3±5.4
36.1±7.3
Crash-O-random
19.7±1.4
14.2±0.5
14.8±1.5
10.0±2.2

Lift-I-medium
59.5±4.5
46.8±2.1
24.4±5.3
36.4±6.7
Lift-O-medium
32.3±5.0
24.5±2.5
16.4±2.3
28.9±4.3

Unlock-I-medium
98.0±4.9
92.7±5.8
91.0±1.8
29.9±1.9
Unlock-O-medium
68.8±1.5
58.7±2.2
60.0±2.0
18.7±1.3

Crash-I-medium
90.4±1.8
82.8±9.9
66.7±7.4
60.8±1.8
Crash-O-medium
20.3±1.9
17.5±0.0
15.9±2.8
24.6±0.9

Lift-I-expert
92.8±1.2
68.6±4.7
75.6±10.7
78.1±6.1
Lift-O-expert
93.7±6.0
58.3±7.9
66.1±8.0
71.9±6.9

Unlock-I-expert
97.4±1.0
93.1±1.9
94.0±2.0
29.3±1.3
Unlock-O-expert
82.1±6.6
64.6±3.1
65.8±3.2
20.2±1.7

Crash-I-expert
95.3±1.4
91.0±2.2
77.9±2.8
50.3±7.6
Crash-O-expert
20.7±3.2
12.5±0.0
26.9±6.1
25.0±1.8

Overall-I
73.3
64.4
57.9
39.3
Overall-O
43.0
32.4
33.2
25.2

31


---Page Break---
Table 9: p-values of different methods (each has 10 random trials) against BECAUSE in various
environments. Under the significance level 0.05, we mark all the baseline results that are significantly
lower than BECAUSE as green, and the rest as red. We can see that BECAUSE significantly
outperforms 3 variants in 18 tasks 83.3% of the experiments (45 out of total 54 pairs of experiments).

Env
BECAUSE-Optimism
BECAUSE-Linear
BECAUSE-Full

Lift-I-random
0.001
0.000
0.003
Lift-O-random
0.003
0.000
0.146
Unlock-I-random
0.189
0.000
0.000
Unlock-O-random
0.145
0.015
0.000
Crash-I-random
0.036
0.090
0.000
Crash-O-random
0.000
0.000
0.000
Lift-I-medium
0.000
0.000
0.000
Lift-O-medium
0.004
0.000
0.130
Unlock-I-medium
0.067
0.006
0.000
Unlock-O-medium
0.000
0.000
0.000
Crash-I-medium
0.061
0.000
0.000
Crash-O-medium
0.005
0.005
1.000
Lift-I-expert
0.000
0.003
0.000
Lift-O-expert
0.000
0.000
0.000
Unlock-I-expert
0.000
0.002
0.000
Unlock-O-expert
0.000
0.000
0.000
Crash-I-expert
0.001
0.000
0.000
Crash-O-expert
0.000
0.969
0.990

C.5
Additional Environment Description

We provide a more detailed description of the environments we use in the experiment, as shown in
Table 10.

Lift
The Lift environment based on the robosuite [34] contains 33 dimensions of state space,
including the end effector pose, joint pose, joint velocity, cube pose as well as its relative position,
cube color, and a contact flag. It contains 4 dimensions of hybrid action space that uses Operation
Space Control (OSC) to control the 3D position and the 1D gripper movement. The task is counted
as a success when the assigned block is lifted from the table over 0.1m. The generalization setting
in the Lift environment is to use an unseen combination of position and color during online testing.
This environment can be abstracted into 15 dimensions of factorizable state space and 4 dimensions
of factorizable action space. The causal graph of this environment is recorded in Figure 8(a).

Unlock
The Unlock environments based on the MiniGrid world [35] contain 110 dimensions of
discrete state space, with 3 of 36-dimensional vector inputs representing the current position of the
agent, key, and door in a 6x6 grid world. The rest 2 dimensions in the state space memorize the state
of whether the agent has the key in hand. The action space is also discrete (with eight dimensions)
to determine the movement (up/down/left/right) and the pick-key, open-door actions. An episode
will be counted as a success when the agent holds the key and uses it to open the door in the right
position. The generalization setting in the Unlock environment is to change the position of the door
and increase the number of total goals in the environment. The agent will only successfully finish one
episode by opening all the doors. The causal graph of this environment is recorded in Figure 8(b).

Crash
The Crash environments are based on the Highway environment [37] which contains 22
dimensions of continuous state space, with four vector inputs representing the current position,
velocity, and orientation of the surrounding vehicles and ego vehicles. There are two additional
dimensions of state memorizing the collision type between the ego vehicles and surrounding vehicles
or pedestrians. The 8-dimensional action space is continuous to determine the acceleration in the
x −y directions of the ego and surrounding agents. The generalization of the Crash environment is
to add different numbers of pedestrians that may cause the crash. An episode will only end when the
ego vehicles have a near-miss with both of the pedestrians at the scene. We visualize the causal graph
of this environment in Figure 8(c).

32


---Page Break---
All three environments are visualized in Figure 4. We list their basic configurations in Table 10.

Table 10: Environment configurations used in experiments

Parameters
Environment
Lift
Unlock
Crash

Max step size
30
15
30
State dimension
33
110
22
Action dimension
4
8
8
Action type
Hybrid
Discrete
Hybrid
Intrinsic state rank
15
4
6
Intrinsic action rank
4
3
4

(a) Lift Environment
(b) Unlock Environment
(c) Crash Environment

Figure 8: Underlying causal graph G in all 3 environments with expert demonstration.

C.6
Additional Baseline Information

We collect data on the above 3 different environments, thus forming 9 groups of offline datasets.

Table 11: Bahavior policies used to collect offline data in different environments.

Environment
Behavior
#Episodes
Success Rate
Additional Description

Lift

Random
1000
0.24
Random actions after a few steps of initialization.
Medium
1000
0.60
Random actions before the goal-reaching expert.
Expert
1000
1.00
Query expert policy for all time steps.

Unlock

Random
200
0.21
Random navigation with high randomness
Medium
200
0.46
Targeted searching in goal directions
Expert
200
0.87
Shortest path planning via A∗

Crash

Random
1000
0.14
Fixed ego, random pedestrians
Medium
1000
0.35
Planned ego, random pedestrians
Expert
1000
0.66
Planning in both ego and pedestrians

After collecting the data using scripted policies in different environments, we train all agents as well
as BECAUSE under 10 different random seeds. Then we report the best performance of each trial
and compute the mean and standard deviation over 10 seeds for each task in the Appendix 6.

We refer to the following codebase to implement all the baselines we use:

• Invariant Causal Imitation Learning (ICIL, [38]): https://github.com/ioanabica/Invariant-
Causal-Imitation-Learning, MIT License.
• Causal Confusion Imitation Learning (CCIL, [39]): reference link to the paper.
• TD3 with Behavior Cloning (TD3+BC, [41]): https://github.com/sfujim/TD3_BC, MIT
License.

33


---Page Break---
• Model-based Offline Policy Otimization (MOPO, [2]): https://github.com/junming-
yang/mopo.git, MIT License.
• Relational Graph Neural Network (GNN, [42]): https://github.com/MichSchli/RelationPrediction.git,
MIT License.
• Causal Dynamics Learning (CDL, [24]): https://github.com/wangzizhao/robosuite/tree/cdl,
MIT License.
• Denoised MDP (Denoised, [12]): https://github.com/facebookresearch/denoised_mdp.git,
CC BY-NC 4.0.
• Mismatch No More (MnM, [9]): reference link to the paper.
• World model with identifiable factorization (IFactor, [13]), reference link to the paper.
• Delphic Offline RL (Delphic, [40]): reference link to the paper.

The detailed hyperparameters we use in BECAUSE and other baselines are listed in Table 12 and
Table 13:

C.7
Experiment Support

Our code is available at the anonymous repo: https://anonymous.4open.science/r/BECAUSE-NeurIPS

Computing resources
The experiments are run on a server with 2×AMD EPYC 7542 32-Core
Processor CPU, 2×NVIDIA RTX 3090 graphics and 2×NVIDIA RTX A6000 graphics, and 252 GB
memory. For one single experiment, it takes BECAUSE and other baselines about 1.5 hours with
100, 000 iterations to train the world model and 1, 000, 000 steps to train the energy-based models.

Table 12: Hyper-parameters of models used in experiments of BECAUSE and baselines (Part I)

Models
Parameters
Environment
Lift
Unlock
Crash

BECAUSE

Learning rate
0.0001
0.001
0.0001
Size of data D
15000
4000
15000
Epoch per iteration
20
5
10
Batch size
256
256
256
Planning horizon H
15
10
20
Planning population
1500
100
1000
Reward discount γ
0.99
0.99
0.99
Spectral norm regularizer λϕ
10−4
10−4
10−4

Spectral norm regularizer λµ
10−4
10−4
10−4

Causal discovery pthres
10−8
10−4
10−6

Encoder hiddens
256
64
128
EBM hidden
256
64
128
EBM negative buffer
5000
1000
5000
EBM training steps
1000
1000
1000
EBM regularizer λEBM
10−4
10−4
10−4

MOPO*

MLP hiddens
256
64
128
MLP layers
2
2
2
Ensemble number
5
5
5

CDL*

Initialized mask coef.
1.0
1.0
1.0
MLP hiddens
256
64
128
Sparsity regularizer
0.001
0.001
0.001

GNN*
GNN hiddens
256
64
128
GNN layers
3
1
3

* Use the same planning parameters as BECAUSE.

34


---Page Break---
Table 13: Hyper-parameters of models used in experiments of baselines (Continued)

Models
Parameters
Environment
Lift
Unlock
Crash

ICIL

Learning rate of MINE
0.0001
0.0001
0.0001
MINE hiddens
256
64
128
MLP hiddens
256
64
128
Learning rate of EBM
0.01
0.01
0.01
Size of buffer of EBM
1000
1000
1000
EBM training steps
1000
1000
1000
EBM hiddens
256
64
128
K of langevin rollout
60
60
60
λV ar of langevin rollout
0.01
0.01
0.01

TD3+BC

Learning rate of Critic
0.0003
0.003
0.0003
Critic hiddens
256
64
128
Learning rate of Actor
0.0003
0.001
0.0001
Actor hiddens
256
64
128
Target update rate
0.005
0.001
0.0001
Policy noise
0.2
0.2
0.2
Balance coefficient α
1.0
2.5
2.5

Denoised MDP

x belief size
256
64
128
y belief size
256
64
128
z belief size
0
0
0
x state size
33
110
22
y state size
33
110
22
z state size
0
0
0
embedding size
256
64
128
Learning rate
0.0001
0.001
0.0001

IFactor

All hidden dim
256
64
128
Disentangled prior output size
19
7
10
Learning rate
0.0001
0.001
0.0001

MnM

All hidden dim
256
64
128
Discriminator learning rate
0.0001
0.001
0.0001
Discriminator clip norm
0.25
0.25
0.25

CCIL

Reg weight
0.0001
0.001
0.0001
Initial mask probability
0.95
0.95
0.95
Learning rate
0.0001
0.001
0.0001

Delphic

Ensemble model size
5
5
5
Uncertainty penalty weight
0.0001
0.0001
0.00005
KL weight
0.0001
0.0001
0.00005

35


---Page Break---
C.8
Broader Impact

This work incorporates causality into reinforcement learning methods, which helps humans under-
stand the underlying mechanism of algorithms and check the source of failures. However, the learned
causal world model may contain human-readable private information about the environment and the
dataset. To mitigate this potential negative societal impact, the causal world model should only be
accessible to trustworthy users.

36


---Page Break---
NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and precede the (optional) supplemental material. The checklist does NOT
count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: They are provided in the abstract and introduction.
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
Justification: They are discussed in the conclusion.

37


---Page Break---
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

Justification: They are provided in section 3.4 and Appendix A and B.

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

Justification: We provide necessary implementation details in section 3.2, 3.3 and Ap-
pendix C.

Guidelines:

• The answer NA means that the paper does not include experiments.

38


---Page Break---
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

Answer: [Yes]
Justification: We provide an open access to the code and data collection scripts in the
anonymous link.

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

39


---Page Break---
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: We provide all the experiment settings and details in Appendix C.5 and C.6.
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

Justification: We report the mean and 95% confidence interval (CI) as well as the significance
level (p value) in Table 6, 7, 8 and 9.
Guidelines: We

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
Justification: Provided in Appendix C.7.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.

40


---Page Break---
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: Provided in the OpenReview form.
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
Justification: Provided in Appendix C.8
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
Justification: This paper does not contain pre-trained language models, image generators,
scraped datasets, or similar assets.
Guidelines:

41


---Page Break---
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

Justification: All the baselines are properly cited and introduced in Appendix C.6.

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

Answer: [NA]

Justification: The paper does not release new assets.

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

42


---Page Break---
Justification: The paper does not involve crowdsourcing or research with human subjects.
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
Justification: The paper does not involve study participants.
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

43


---Page Break---
