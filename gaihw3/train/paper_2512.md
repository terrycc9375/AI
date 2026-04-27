Doubly Mild Generalization for Offline
Reinforcement Learning

Yixiu Mao1, Qi Wang1, Yun Qu1, Yuhang Jiang1, Xiangyang Ji1

1Department of Automation, Tsinghua University
myx21@mails.tsinghua.edu.cn, xyji@tsinghua.edu.cn

Abstract

Offline Reinforcement Learning (RL) suffers from the extrapolation error and value
overestimation. From a generalization perspective, this issue can be attributed to the
over-generalization of value functions or policies towards out-of-distribution (OOD)
actions. Significant efforts have been devoted to mitigating such generalization, and
recent in-sample learning approaches have further succeeded in entirely eschewing
it. Nevertheless, we show that mild generalization beyond the dataset can be trusted
and leveraged to improve performance under certain conditions. To appropriately
exploit generalization in offline RL, we propose Doubly Mild Generalization
(DMG), comprising (i) mild action generalization and (ii) mild generalization
propagation. The former refers to selecting actions in a close neighborhood of the
dataset to maximize the Q values. Even so, the potential erroneous generalization
can still be propagated, accumulated, and exacerbated by bootstrapping. In light
of this, the latter concept is introduced to mitigate the generalization propagation
without impeding the propagation of RL learning signals. Theoretically, DMG
guarantees better performance than the in-sample optimal policy in the oracle
generalization scenario. Even under worst-case generalization, DMG can still
control value overestimation at a certain level and lower bound the performance.
Empirically, DMG achieves state-of-the-art performance across Gym-MuJoCo
locomotion tasks and challenging AntMaze tasks. Moreover, benefiting from its
flexibility in both generalization aspects, DMG enjoys a seamless transition from
offline to online learning and attains strong online fine-tuning performance.

1
Introduction

Reinforcement learning (RL) aims to solve sequential decision-making problems and has garnered
significant attention in recent years [53, 67, 74, 63, 12]. However, its practical applications encounter
several challenges, such as risky exploration attempts [20] and time-consuming data collection
phases [35]. Offline RL emerges as a promising paradigm to alleviate these challenges by learning
without interaction with the environment [40, 42]. It eliminates the need for unsafe exploration and
facilitates the utilization of pre-existing large-scale datasets [31, 48, 59].

However, offline RL suffers from the out-of-distribution (OOD) issue and extrapolation error [19].
From a generalization perspective, this well-known challenge can be regarded as a consequence of the
over-generalization of value functions or policies towards OOD actions [47]. Specifically, the potential
value over-estimation at OOD actions caused by intricate generalization is often improperly captured
by the max operation [73]. This over-estimation will propagate to values of in-distribution samples
through Bellman backups and further spread to values of OOD ones via generalization. In mitigating
value overestimation caused by OOD actions, substantial efforts have been dedicated [19, 39, 38, 17]
and recent advancements in in-sample learning have successfully formulated the Bellman target
solely with the actions present in the dataset [37, 85, 92, 88, 21] and extracted policies by weighted

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
behavior cloning [57, 80]. As a result, these algorithms completely eschew generalization and avoid
the extrapolation error. Despite simplicity, this way can not take advantage of the generalization
ability of neural networks, which could be beneficial for performance improvement. Until now, how
to appropriately exploit generalization in offline RL remains a lasting issue.

This work demonstrates that mild generalization beyond the dataset can be trusted and leveraged to
improve performance under certain conditions. For appropriate exploitation of mild generalization, we
propose Doubly Mild Generalization (DMG) for offline RL, comprising (i) mild action generalization
and (ii) mild generalization propagation. The former concept refers to choosing actions in the
vicinity of the dataset to maximize the Q values. However, the mere utilization of mild action
generalization still falls short in adequately circumventing potential erroneous generalization, which
can be propagated, accumulated, and exacerbated through the process of bootstrapping. To address
this, we propose a novel concept, mild generalization propagation, which involves reducing the
generalization propagation while preserving the propagation of RL learning signals. Regarding
DMG’s implementation, this work presents a simple yet effective scheme. Specifically, we blend the
mildly generalized max with the in-sample max in the Bellman target, where the former is achieved
by actor-critic learning with regularization towards high-value in-sample actions, and the latter is
accomplished using in-sample learning techniques such as expectile regression [37].

We conduct a thorough theoretical analysis of our approach DMG in both oracle and worst-case
generalization scenarios. Under oracle generalization, DMG guarantees better performance than the
in-sample optimal policy in the dataset [38, 37]. Even under worst-case generalization, DMG can
still upper bound the overestimation of value functions and guarantee to output a safe policy with a
performance lower bound. Empirically1, DMG achieves state-of-the-art performance on standard
offline RL benchmarks [16], including Gym-MuJoCo locomotion tasks and challenging AntMaze
tasks. Moreover, benefiting from its flexibility in both generalization aspects, DMG can seamlessly
transition from offline to online learning and attain superior online fine-tuning performance.

2
Preliminaries

RL.
The environment in RL is mostly characterized as a Markov decision process (MDP), which
can be represented as a tuple M = (S, A, P, R, γ, d0), comprising the state space S, action space A,
transition dynamics P : S × A →∆(S), reward function R : S × A →[0, Rmax], discount factor
γ ∈[0, 1), and initial state distribution d0 [70]. The goal of RL is to find a policy π : S →∆(A)
that can maximize the expected discounted return, denoted as J(π):

J(π) = Es0∼d0,at∼π(·|st),st+1∼P (·|st,at)

" ∞
X

t=0
γtR(st, at)

#

.
(1)

For any policy π, we define the value function as V π(s) = Eπ [P∞
t=0 γtR(st, at)|s0 = s] and the
state-action value function (Q-value function) as Qπ(s, a) = Eπ [P∞
t=0 γtR(st, at)|s0 = s, a0 = a].

Offline RL.
Distinguished from traditional online RL training, offline RL handles a static dataset
of transitions D = {(si, ai, ri, s′
i)}n−1
i=0 and seeks an optimal policy without any additional data
collection [40, 42]. We use ˆβ(a|s) to denote the empirical behavior policy observed in D, which
depicts the conditional distributions in the dataset [19]. Ordinary approximate dynamic programming
methods minimize temporal difference error, according to the following loss [70]:

LT D(θ) = E(s,a,s′)∼D
h
(Qθ(s, a) −R(s, a) −γ max
a′ Qθ′(s′, a′))2i
,
(2)

where πϕ is a parameterized policy, Qθ(s, a) is a parameterized Q function, and Qθ′(s, a) is a target
Q function whose parameters are updated via Polyak averaging [53].

3
Doubly Mild Generalization for Offline RL

This section discusses the strategy to appropriately exploit generalization in offline RL. In Section 3.1,
we introduce a formal perspective on how generalization impacts offline RL and discuss the issues of

1Our code is available at https://github.com/maoyixiu/DMG.

2


---Page Break---
over-generalization and non-generalization. Subsequently, we propose the DMG concept, comprising
mild action generalization and mild generalization propagation in Section 3.2. Following this, we
conduct a comprehensive analysis of DMG in both oracle generalization (Section 3.3) and worst-case
generalization scenarios (Section 3.4). Finally, we present the practical algorithm in Section 3.5.

3.1
Generalization Issues in Offline RL

Offline RL training typically involves a complex interaction between Bellman backup and generaliza-
tion [47]. Offline RL algorithms vary in backup mechanisms to train the Q function. Here we denote
a generic form of Bellman backup as Tu, where u is a distribution in the action space.

TuQ(s, a) := R(s, a) + γEs′∼P (·|s,a)


max
a′∼u(·|s′) Q(s′, a′)

(3)

During offline training, this backup is exclusively executed on (s, a) ∈D, and the values of (s, a) /∈D
are influenced solely via generalization. A crucial aspect is that (s′, a′) in the Bellman target can
be absent from the dataset D, depending on the choice of u. As a result, Bellman backup and
generalization exhibit an intricate interaction: the backups on (s, a) ∈D impact the values of
(s, a) /∈D via generalization; the values of (s, a) /∈D participates in the computation of Bellman
target, thereby affecting the values of (s, a) ∈D.

This interaction poses a key challenge in offline RL, value overestimation. The potential overestima-
tion of values of (s, a) /∈D, induced by intricate generalization, tends to be improperly captured by
the max operation, a phenomenon known as maximization bias [73]. This overestimation propagates
to values of (s, a) ∈D through backups and further extends to values of (s, a) /∈D via generaliza-
tion. This cyclic process consistently amplifies value overestimation, potentially resulting in value
divergence. The crux of this detrimental process can be summarized as over-generalization.

To address value overestimation, recent advancements in the field have introduced a paradigm
known as in-sample learning, which formulates the Bellman target solely with the actions present
in the dataset [37, 85, 92, 88, 21]. Its effect is equivalent to choosing u in Tu to be exactly ˆβ,
i.e., the empirical behavior policy observed in the dataset. Following in-sample value learning,
policies are extracted from the learned Q functions using weighted behavior cloning [57, 9, 55]. By
entirely eschewing generalization in offline RL training, they effectively avoid the extrapolation
error [19], a strategy we term non-generalization. However, the ability to generalize is a critical
factor contributing to the extensive utilization of neural networks [41]. In this sense, in-sample
learning methods seem too conservative without utilizing generalization, particularly when the offline
datasets do not cover the optimal actions in large or continuous spaces.

3.2
Doubly Mild Generalization

The following focuses on the appropriate exploitation of generalization in offline RL.

We start by analyzing the generalization effect under the generic backup operator Tu. We consider a
straightforward scenario, where Qθ is updated to Qθ′ by one gradient step on a single (s, a) ∈D with
learning rate α. We characterize the resulting generalization effect on any (s, ˜a) /∈D2 as follows.
Theorem 1 (Informal). Under certain continuity conditions, the following equation holds when the
learning rate α is sufficiently small and ˜a is sufficiently close to a:

Qθ′(s, ˜a) = Qθ(s, ˜a) + C1 (TuQθ(s, ˜a) −Qθ(s, ˜a) + C2∥˜a −a∥) + O
 
∥θ′ −θ∥2
(4)

where C1 ∈[0, 1] and C2 is a bounded constant.

The formal theorem and all proofs are deferred to Appendix B.

Note that Eq. (4) is the update of the parametric Q function (Qθ →Qθ′) at state-action pairs
(s, ˜a) /∈D, which is exclusively caused by generalization. If ˜a is within a close neighborhood of
a, then C2∥˜a −a∥is small. Moreover, as C1 ∈[0, 1], Eq. (4) approximates an update towards the
true objective TuQθ(s, ˜a), as if Qθ(s, ˜a) is updated by a true gradient step at (s, ˜a) /∈D. Therefore,

2The interplay between backup and generalization does not involve states out of the dataset (Bellman target
does not contain OOD states), hence we do not consider (˜s, ˜a) /∈D, though the analysis of Q(˜s, ˜a) is similar.

3


---Page Break---
Theorem 1 shows that, under certain continuity conditions, Q functions can generalize well and
approximate true updates in a close neighborhood of samples in the dataset. This implies that mild
generalizations beyond the dataset can be leveraged to potentially pursue better performance. Inspired
by Theorem 1, we define a mildly generalized policy ˜β as follows.

Definition 1 (Mildly generalized policy). Policy ˜β is termed a mildly generalized policy if it satisfies

supp(ˆβ(·|s)) ⊆supp(˜β(·|s)), and
max
a1∼˜β(·|s)
min
a2∼ˆβ(·|s)
∥a1 −a2∥≤ϵa,
(5)

where ˆβ is the empirical behavior policy observed in the offline dataset.

It means that ˜β has a wider support than ˆβ (the dataset), and for any a1 ∼˜β(·|s), we can find
a2 ∼ˆβ(·|s) (in dataset) such that ∥a1 −a2∥≤ϵa. In other words, the generalization of ˜β beyond
the dataset is bounded by ϵa when measured in the action space distance. According to Theorem 1,
there is a high chance that Qθ can generalize well in this mild generalization area ˜β(a|s) > 0.

However, even in this mild generalization area, it is inevitable that the learned value function will incur
some degree of generalization error. The possible erroneous generalization can still be propagated
and exacerbated by value bootstrapping as discussed in Section 3.1. To this end, we introduce an
additional level of mild generalization, termed mild generalization propagation, and propose a novel
Doubly Mildly Generalization (DMG) operator as follows.

Definition 2. The Doubly Mild Generalization (DMG) operator is defined as

TDMGQ(s, a) := R(s, a)+γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
Q(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
Q(s′, a′)

#

(6)

where ˆβ is the empirical behavior policy in the dataset and ˜β is a mildly generalized policy.

Note that in typical offline RL algorithms, extrapolation error and value overestimation caused by
erroneous generalization are propagated through bootstrapping, and the discount factor of this process
is γ. DMG reduces this discount factor to λγ, mitigating the amplification of value overestimation.
On the other hand, in contrast to in-sample methods, DMG allows mild generalization, utilizing the
generalization ability of neural networks to seek better performance, as Theorem 1 suggests that
value functions are highly likely to generalize well in the mild generalization area.

To summarize, the generalization of DMG is mild in two aspects: (i) mild action generalization:
based on the mildly generalized policy ˜β, which generalizes beyond ˆβ, DMG selects actions in a
close neighborhood of the dataset to maximize the Q values in the first part of the Bellman target;
and (ii) mild generalization propagation: DMG mitigates the generalization propagation without
hindering the propagation of RL learning signals by blending the mildly generalized max with the
in-sample max in the Bellman target. This reduces the discount factor through which generalization
propagates, mitigating the amplification of value overestimation caused by bootstrapping.

To support the above claims, we provide a comprehensive analysis of DMG in both oracle and
worst-case generalization scenarios, with particular emphasis on value estimation and performance.

3.3
Oracle Generalization

This section conducts analyses under the assumption that the learned value functions can achieve
oracle generalization in the mild generalization area ˜β(a|s) > 0, formally defined as follows.
Assumption 1 (Oracle generalization). The generalization of learned Q functions in the mild
generalization area ˜β(a|s) > 0 reflects the true value updates according to TDMG.

The mild generalization area ˜β(a|s) > 0 may contain some points outside the offline dataset, and
TDMG might query Q values of such points. This assumption assumes that the generalization at
such points reflects the true value updates according to TDMG. The rationale for such an assumption
comes from Theorem 1, which characterizes the generalization effect of value functions in the mild
generalization area. Now we analyze the dynamic programming properties of the operators TDMG
and TIn, where TIn is the in-sample Q learning operator [37, 88, 21] defined as follows.

4


---Page Break---
Definition 3. The In-sample Q Learning operator [37] is defined as

TInQ(s, a) := R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
Q(s′, a′)

#

(7)

where ˆβ is the empirical behavior policy in the dataset.

Lemma 1. TIn is a γ-contraction operator in the in-sample area ˆβ(a|s) > 0 under the L∞norm.

Following Lemma 1, we denote the fixed point of TIn as Q∗
In, and its induced policy as π∗
In. Here Q∗
In
is known as the in-sample optimal value function [37], which is the value function of the in-sample
optimal policy π∗
In. We refer readers to [37, 38, 88] for more discussions on the in-sample optimality.

Now we present the theoretical properties of DMG for comparison.
Theorem 2 (Contraction). Under Assumption 1, TDMG is a γ-contraction operator in the mild
generalization area ˜β(a|s) > 0 under the L∞norm. Therefore, by repeatedly applying TDMG, any
initial Q function can converge to the unique fixed point Q∗
DMG.

We denote the induced policy of Q∗
DMG as π∗
DMG, whose performance is guaranteed as follows.
Theorem 3 (Performance). Under Assumption 1, the value functions of π∗
DMG and π∗
In satisfy:

V π∗
DMG(s) ≥V π∗
In(s),
∀s ∈D.
(8)

Theorem 3 indicates that the policy learned by DMG can achieve better performance than the
in-sample optimal policy under the oracle generalization condition.

3.4
Worst-case Generalization

This section turns to the analyses in the worst-case generalization scenario, where the learned value
functions may exhibit poor generalization in the mild generalization area ˜β(a|s) > 0. In other words,
this section considers that TDMG is only defined in the in-sample area ˆβ(a|s) > 0 and the learned
value functions may have any generalization error at other state-action pairs. In this case, we use the
notation ˆTDMG to tell the difference.

We make continuity assumptions about the learned Q function and the transition dynamics.
Assumption 2 (Lipschitz Q). The learned Q function is KQ-Lipschitz. ∀s ∼D, ∀a1, a2 ∼A,
|Q(s, a1) −Q(s, a2)| ≤KQ∥a1 −a2∥
Assumption 3 (Lipschitz P). The transition dynamics P is KP -Lipschitz. ∀s, s′ ∼S, ∀a1, a2 ∼A,
|P(s′|s, a1) −P(s′|s, a2)| ≤KP ∥a1 −a2∥

For Assumption 2, a continuous learned Q function is particularly necessary for analyzing value func-
tion generalization and can be relatively easily satisfied using neural networks or linear models [24].
Assumption 3 is also a common assumption in theoretical studies of RL [13, 87, 61].

Now we consider the iteration starting from arbitrary function Q0: ˆQk
DMG = ˆTDMG ˆQk−1
DMG and
Qk
In = TInQk−1
In , ∀k ∈Z+. The possible value of ˆQk
DMG is bounded by the following results.
Theorem 4 (Limited overestimation). Under Assumption 2, the learned Q function of DMG by
iterating ˆTDMG satisfies the following inequality

Qk
In(s, a) ≤ˆQk
DMG(s, a) ≤Qk
In(s, a) + λϵaKQγ

1 −γ
(1 −γk), ∀s, a ∼D, ∀k ∈Z+.
(9)

Since in-sample training eliminates the extrapolation error [37, 92], Qk
In can be considered a relatively
accurate estimate [37]. Therefore, Theorem 4 suggests that DMG exhibits limited value overesti-
mation under the worst-case generalization scenario. Moreover, the bound becomes tighter as ϵa
decreases (milder action generalization) and λ decreases (milder generalization propagation). This is
consistent with our intuitions in Section 3.2.

Finally, we show in Theorem 5 that even under worst-case generalization, DMG guarantees to output
a safe policy with a performance lower bound.

5


---Page Break---
Theorem 5 (Performance lower bound). Let ˆπDMG be the learned policy of DMG by iterating ˆTDMG,
π∗be the optimal policy, and ϵD be the inherent performance gap of the in-sample optimal policy
ϵD := J(π∗) −J(π∗
In). Under Assumptions 2 and 3, for sufficiently small ϵa, we have

J(ˆπDMG) ≥J(π∗) −CKP Rmax

1 −γ
ϵa −ϵD.
(10)

where C is a positive constant.

3.5
Practical Algorithm

This section puts DMG into implementation and presents a simple yet effective practical algorithm.
The algorithm comprises the following networks: policy πϕ, target policy πϕ′, Q network Qθ, target
Q network Qθ′, and V network Vψ.

Policy learning.
Practically, we expect DMG to exhibit a tendency towards mild generalization
around good actions in the dataset. To this end, we first consider reshaping the empirical behavior
policy ˆβ to be skewed towards actions with high advantage values ˆβ∗(a|s) ∝ˆβ(a|s) exp(A(s, a)).
Then we enforce the proximity between the trained policy and the reshaped behavior policy to
constrain the generalization area. We define the generalization set ΠG as follows.

ΠG = {π | KL(ˆβ∗(·|s)∥π(·|s)) ≤ϵ}
(11)

Note that forward KL allows π to select actions outside the support of ˆβ∗, enabling ΠG to generalize
beyond the actions in the dataset. With ΠG defined, the next step is to compute the maximal Q within
ΠG. To accomplish this, we adopt Actor-Critic style training [70] for this part.

max
ϕ
Es∼D,a∼πϕ(·|s)Qθ(s, a),
s.t. πϕ ∈ΠG
(12)

By treating the constraint term as a penalty, we maximize the following objective.

max
ϕ
Es∼D,a∼πϕ(·|s)Qθ(s, a) −νEs∼DKL(ˆβ∗(·|s)∥πϕ(·|s))
(13)

Through straightforward derivations, Eq. (13) is equivalent to the following policy training objective.

Jπ(ϕ) = Es∼D,a∼πϕ(·|s)Qθ(s, a) −νE(s,a)∼D [exp(α(Qθ′(s, a) −Vψ(s))) log πϕ(a|s)]
(14)

where α is an inverse temperature and Qθ′(s, a) −Vψ(s) computes the advantage function A(s, a).

Algorithm 1 DMG

1: Initialize πϕ, πϕ′, Qθ, Qθ′, and Vψ.
2: for each gradient step do
3:
Update ψ by minimizing Eq. (15)
4:
Update θ by minimizing Eq. (16)
5:
Update ϕ by maximizing Eq. (14)
6:
Update target networks: θ′ ←(1−ξ)θ′+
ξθ, ϕ′ ←(1 −ξ)ϕ′ + ξϕ
7: end for

Value learning.
Now we turn to the implementa-
tion of the TDMG operator for training value func-
tions. By introducing the aforementioned policy,
we can substitute maxa∼˜β in TDMG with Ea∼π.
Regarding maxa∼ˆβ in TDMG, any in-sample learn-
ing techniques can be employed to compute the
in-sample maximum [37, 88, 85, 21]. In particular,
based on IQL [37], we perform expectile regression.

LV (ψ) =
E
(s,a)∼D [Lτ
2 (Qθ′(s, a) −Vψ(s))] (15)

where Lτ
2(u) = |τ −1(u < 0)|u2 and τ ∈(0, 1). For τ ≈1, Vψ can capture the in-sample maximal
Q [37]. Finally, we have the following value training loss.

LQ(θ) = E(s,a,s′)∼D


Qθ(s, a) −R(s, a) −γλEa′∼πϕ′Qθ′(s′, a′) −γ(1 −λ)Vψ(s′)
2
(16)

Overall algorithm.
Integrating all components, we present our practical algorithm in Algorithm 1.

6


---Page Break---
4
Discussions and Related Work

Summary of offline RL work from a generalization perspective.
As analyzed above, DMG is
featured in both mild action generalization and mild generalization propagation. Within the actor-
critic framework upon which most offline RL algorithms are built, these two aspects correspond to
the policy and value training phases, respectively. Action generalization concerns whether the policy
training intentionally selects actions beyond the dataset to maximize Q values, while generalization
propagation involves whether value training propagates generalization through bootstrapping. Table 1
presents a clear comparison of offline RL works in this generalization view. The table shows one
representative method of each category and we elaborate on others as follows.

Table 1: Comparison of offline RL work from the generalization perspective.

IQL
AWAC
TD3BC
TD3
DMG (Ours)

Action generalization
none
none
mild
full
mild

Generalization propagation
none
full
full
full
mild

Concerning policy learning, AWR [57], AWAC [55], CRR [80], 10% BC [8], IQL [37], and other
works such as [78, 9, 66, 21, 88] extract policies through weighted or filtered behavior cloning,
thereby lacking intentional action generalization to maximize Q values beyond the dataset. Typical
policy-regularized offline RL methods like TD3BC [17], BRAC [84], BEAR [38], SPOT [83], and
others such as [79, 61, 72] introduce regularization terms to Q maximization objectives to regularize
the trained policy towards the behavior policy and allows mild action generalization. Online RL
algorithms like TD3 [18] and SAC [27] have no constraints and maximize Q values in the entire action
space, corresponding to full action generalization. Regarding value training, in-sample learning
methods including OneStep RL [7], IQL [37], InAC [85], IAC [92], XQL [21], and SQL [88]
completely avoid generalization propagation and accumulation via bootstrapping, whereas typical
offline and online RL approaches allow full generalization propagation through bootstrapping. In the
proposed approach DMG, generalization is mild in both aspects.

Recently, Ma et al. [47] have also drawn attention to generalization in offline RL and the issue of over-
generalization. They mitigate over-generalization from a representation perspective, differentiating
between the representations of in-sample and OOD state-action pairs. Lyu et al. [44] argue that
conventional value penalization like CQL [39] tends to harm the generalization of value functions and
hinder performance improvement. They propose mild value penalization to mitigate the detrimental
effects of value penalization on generalization.

Connection to heuristic blending approaches.
Our approach also relates to the framework of
blending heuristics into bootstrapping [10, 81, 71, 28, 82, 22]. In offline RL, HUBL [22] blends
Monte-Carlo returns into bootstrapping and acts as a data relabeling step, which reduces the degree
of bootstrapping and thereby increases its performance. In contrast, DMG blends the in-sample
maximal values into the bootstrapping operator. DMG does not reduce the discount for RL learning
but reduces the discount for generalization propagation.

For extended discussions on related work, please refer to Appendix A.

5
Experiments

In this section, we conduct several experiments to justify the validity of the proposed method DMG.
Experimental details and extended results are provided in Appendices C and D, respectively.

5.1
Main Results on Offline RL Benchmarks

Tasks.
We evaluate the proposed approach on Gym-MuJoCo locomotion domains and challenging
AntMaze domains in D4RL [16]. The latter involves sparse-reward tasks and necessitates “stitching”
fragments of suboptimal trajectories traveling undirectedly to find a path to the goal of the maze.

Baselines.
Our offline RL baselines include both typical bootstrapping methods and in-sample
learning approaches. For the former, we compare to BCQ [19], BEAR [38], AWAC [55], TD3BC [17],

7


---Page Break---
Table 2: Averaged normalized scores on Gym locomotion and Antmaze tasks over five random seeds.
m = medium, m-r = medium-replay, m-e = medium-expert, e = expert, r = random; u = umaze, u-d =
umaze-diverse, m-p = medium-play, m-d = medium-diverse, l-p= large-play, l-d = large-diverse.

Dataset-v2
BC
BCQ
BEAR
DT
AWAC
OneStep
TD3BC
CQL
IQL
DMG (Ours)

halfcheetah-m
42.0
46.6
43.0
42.6
47.9
50.4
48.3
47.0
47.4
54.9±0.2
hopper-m
56.2
59.4
51.8
67.6
59.8
87.5
59.3
53.0
66.2
100.6±1.9
walker2d-m
71.0
71.8
-0.2
74.0
83.1
84.8
83.7
73.3
78.3
92.4±2.7
halfcheetah-m-r
36.4
42.2
36.3
36.6
44.8
42.7
44.6
45.5
44.2
51.4±0.3
hopper-m-r
21.8
60.9
52.2
82.7
69.8
98.5
60.9
88.7
94.7
101.9±1.4
walker2d-m-r
24.9
57.0
7.0
66.6
78.1
61.7
81.8
81.8
73.8
89.7±5.0
halfcheetah-m-e
59.6
95.4
46.0
86.8
64.9
75.1
90.7
75.6
86.7
91.1±4.2
hopper-m-e
51.7
106.9
50.6
107.6
100.1
108.6
98.0
105.6
91.5
110.4±3.4
walker2d-m-e
101.2
107.7
22.1
108.1
110.0
111.3
110.1
107.9
109.6
114.4±0.7
halfcheetah-e
92.9
89.9
92.7
87.7
81.7
88.2
96.7
96.3
95.0
95.9±0.3
hopper-e
110.9
109.0
54.6
94.2
109.5
106.9
107.8
96.5
109.4
111.5±2.2
walker2d-e
107.7
106.3
106.6
108.3
110.1
110.7
110.2
108.5
109.9
114.7±0.4
halfcheetah-r
2.6
2.2
2.3
2.2
6.1
2.3
11.0
17.5
13.1
28.8±1.3
hopper-r
4.1
7.8
3.9
5.4
9.2
5.6
8.5
7.9
7.9
20.4±10.4
walker2d-r
1.2
4.9
12.8
2.2
0.2
6.9
1.6
5.1
5.4
4.8±2.2

locomotion total
784.2
968.0
581.7
972.6
975.3
1041.2
1013.2
1010.2
1033.1
1182.8

antmaze-u
66.8
78.9
73.0
54.2
80.0
54.0
73.0
82.6
89.6
92.4±1.8
antmaze-u-d
56.8
55.0
61.0
41.2
52.0
57.8
47.0
10.2
65.6
75.4±8.1
antmaze-m-p
0.0
0.0
0.0
0.0
0.0
0.0
0.0
59.0
76.4
80.2±5.1
antmaze-m-d
0.0
0.0
8.0
0.0
0.2
0.6
0.2
46.6
72.8
77.2±6.1
antmaze-l-p
0.0
6.7
0.0
0.0
0.0
0.0
0.0
16.4
42.0
55.4±6.2
antmaze-l-d
0.0
2.2
0.0
0.0
0.0
0.2
0.0
3.2
46.0
58.8±4.5

antmaze total
123.6
142.8
142.0
95.4
132.2
112.6
120.2
218.0
392.4
439.4

and CQL [39]. For the latter, we compare to BC [58], OneStep RL [7], IQL [37], XQL [21], and
SQL [88]. We also include the sequence-modeling method Decision Transformer (DT) [8].

Comparison with baselines.
Aggregated results are displayed in Table 2. On the Gym locomotion
tasks, DMG outperforms prior methods on most tasks and achieves the highest total score. On
the much more challenging AntMaze tasks, DMG outperforms all the baselines by a large margin,
especially in the most difficult large mazes. For detailed learning curves, please refer to Appendix D.3.
According to [56], we also report the results of DMG over more random seeds in Appendix D.2.

Runtime.
We test the runtime of DMG and other baselines on a GeForce RTX 3090. As shown in
Appendix D.1, the runtime of DMG is comparable to that of the fastest offline RL algorithm TD3BC.

5.2
Performance Improvement over In-sample Learning Approaches

Table 3: DMG combined with various in-sample ap-
proaches, showing averaged scores over 5 seeds.

Dataset-v2
XQL (+DMG)
SQL(+DMG)

halfcheetah-m
47.7 →55.3
48.3 →54.5
hopper-m
71.1 →90.1
75.5 →97.7
walker2d-m
81.5 →88.7
84.2 →89.8
halfcheetah-m-r
44.8 →51.1
44.8 →51.8
hopper-m-r
97.3 →102.5
101.7 →101.8
walker2d-m-r
75.9 →90.0
77.2 →95.2
halfcheetah-m-e
89.8 →92.5
94.0 →93.5
hopper-m-e
107.1 →111.1
111.8 →110.4
walker2d-m-e
110.1 →111.3
110.0 →109.6

total
725.3 →792.7
747.5 →804.2

DMG can be combined with various in-
sample learning approaches.
Besides
IQL [37], we also apply DMG to two re-
cent state-of-the-art in-sample algorithms,
XQL [21] and SQL [88]. As shown in Ta-
ble 3 (and Table 2), DMG consistently and
substantially improves upon these in-sample
methods, particularly on sub-optimal datasets
where generalization plays a crucial role in
the pursuit of a better policy. This provides
compelling empirical evidence that the per-
formance of in-sample methods is largely
confined by eschewing generalization beyond
the dataset, while DMG effectively exploits
generalization, achieving significantly im-
proved performance across tasks.

8


---Page Break---
0.00
0.25
0.50
0.75
1.00
0

20

40

60

80

100

Normalized Return

Return

320

330

340

350

Q Value

walker2d-medium-v2

Q value

0.00
0.25
0.50
0.75
1.00
40

60

80

100

Normalized Return

Return

250

255

260

265

270

Q Value

hopper-medium-v2

Q value

0.00
0.25
0.50
0.75
1.00
50

52

54

56

58

60

Normalized Return

Return

480

500

520

540

Q Value

halfcheetah-medium-v2

Q value

0.00
0.25
0.50
0.75
1.00
10

15

20

25

30

Normalized Return

Return
0

50

100

150

Q Value

halfcheetah-random-v2

Q value

Figure 1: Performance and Q values of DMG with varying mixture coefficient λ over 5 random
seeds. The crosses × mean that the value functions diverge in several seeds. As λ increases, DMG
enables stronger generalization propagation, resulting in higher and probably divergent learned Q
values. Mild generalization propagation plays a crucial role in achieving strong performance.

10
1
0.5
0.1 0.05 0.010.001
0

20

40

60

80

100

Normalized Return

Return

300

320

340

360

380

400

Q Value

walker2d-medium-v2

Q value

10
1
0.5
0.1 0.05 0.010.001
0

25

50

75

100

Normalized Return

Return

250

260

270

280

290

300

Q Value

hopper-medium-v2

Q value

10
1
0.5
0.1 0.05 0.010.001
45

50

55

60

Normalized Return

Return
460

480

500

520

540

Q Value

halfcheetah-medium-v2

Q value

10
1
0.5
0.1 0.05 0.010.001
0

10

20

30

40

Normalized Return

Return

0

20

40

60

80

100

Q Value

halfcheetah-random-v2

Q value

Figure 2: Performance and Q values of DMG with varying penalty coefficient ν over 5 random
seeds. As ν decreases, DMG allows broader action generalization, leading to larger learned Q values.
Mild action generalization is also critical for attaining superior performance.

5.3
Ablation Study for Performance and Value Estimation

Mixture coefficient λ.
The mixture coefficient λ controls the extent of generalization propagation.
We fix ν = 0.1 and vary λ ∈[0, 1], presenting the learned Q values and performance on several
tasks in Figure 1. As λ increases, DMG enables increased generalization propagation through
bootstrapping, and the learned Q values become larger and probably diverge. A moderate λ (mild
generalization propagation) is crucial for achieving strong performance across datasets. Under the
same degree of action generalization, mild generalization propagation effectively suppresses value
overestimation, facilitating more stable policy learning.

Penalty coefficient ν.
The penalty coefficient ν regulates the degree of action generalization. We fix
λ = 0.25 and vary ν. The results are shown in Figure 2. As ν decreases, DMG allows broader action
generalization beyond the dataset, which results in higher learned values. Regarding performance, a
moderate ν (mild action generalization) is also crucial for achieving superior performance.

5.4
Online Fine-tuning after Offline RL

Table 4: Online fine-tuning results on AntMaze tasks,
showing normalized scores of offline training and 1M
steps online fine-tuning, averaged over 5 seeds.

Dataset-v2
TD3
IQL
DMG (Ours)

antmaze-u
0.0
89.6 →96.2
92.4 →98.4
antmaze-u-d
0.0
65.6 →62.2
75.4 →89.2
antmaze-m-p
0.0
76.4 →89.8
80.2 →96.8
antmaze-m-d
0.0
72.8 →90.2
77.2 →96.2
antmaze-l-p
0.0
42.0 →78.6
55.4 →86.8
antmaze-l-d
0.0
46.0 →73.4
58.8 →89.0

antmaze total
0.0
392.4 →490.4
439.4 →556.4

Benefiting from its flexibility in both gen-
eralization aspects, DMG enjoys a seam-
less transition from offline to online learn-
ing. This is accomplished through a grad-
ual enhancement of both action generaliza-
tion and generalization propagation. Since
IQL [37] has demonstrated superior online
fine-tuning performance compared to previ-
ous methods [55, 39] in its paper, we follow
the experimental setup of IQL and compare
to IQL. We also train online RL algorithm
TD3 [18] from scratch for comparison. We
use the challenging AntMaze domains [16],
given DMG’s already high offline perfor-
mance in Gym locomotion domains. Results are presented in Table 4. While online training from
scratch fails in the challenging sparse reward AntMaze tasks, DMG initialized with offline pretraining
succeeds in learning near-optimal policies, outperforming IQL by a significant margin. Please refer
to Appendix C.2 for experimental details, and to Appendix D.4 for learning curves.

9


---Page Break---
6
Conclusion and Limitations

This work scrutinizes offline RL through the lens of generalization and proposes DMG, comprising
mild action generalization and mild generalization propagation, to exploit generalization in offline RL
appropriately. We theoretically analyze DMG in oracle and worst-case generalization scenarios, and
empirically demonstrate its SOTA performance in offline training and online fine-tuning experiments.

While our work contributes valuable insights, it also has limitations. The DMG principle is shown to
be effective across most scenarios. However, when the function approximator employed is highly
compatible with a specific task setting, the learned value functions may generalize well in the entire
action space. In such case, DMG may underperform full generalization methods due to conservatism.

Acknowledgment

We thank the anonymous reviewers for feedback on an early version of this paper. This work was
supported by the National Key R&D Program of China under Grant 2018AAA0102801, National
Natural Science Foundation of China under Grant 61827804.

References

[1] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization.
In International conference on machine learning, pages 22–31. PMLR, 2017.

[2] Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline
reinforcement learning with diversified q-ensemble. Advances in neural information processing
systems, 34:7436–7447, 2021.

[3] Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, and
Zhaoran Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning.
In International Conference on Learning Representations, 2022. URL https://openreview.
net/forum?id=Y4cs1Z3HnqL.

[4] Jacob Beck, Risto Vuorio, Evan Zheran Liu, Zheng Xiong, Luisa Zintgraf, Chelsea Finn, and
Shimon Whiteson. A survey of meta-reinforcement learning. arXiv preprint arXiv:2301.08028,
2023.

[5] Avinandan Bose, Simon Shaolei Du, and Maryam Fazel. Offline multi-task transfer rl with
representational penalization. arXiv preprint arXiv:2402.12570, 2024.

[6] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press,
2004.

[7] David Brandfonbrener, Will Whitney, Rajesh Ranganath, and Joan Bruna. Offline rl without
off-policy evaluation. Advances in Neural Information Processing Systems, 34:4933–4946,
2021.

[8] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter
Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning
via sequence modeling. Advances in neural information processing systems, 34:15084–15097,
2021.

[9] Xinyue Chen, Zijian Zhou, Zheng Wang, Che Wang, Yanqiu Wu, and Keith Ross. Bail: Best-
action imitation learning for batch deep reinforcement learning. Advances in Neural Information
Processing Systems, 33:18353–18363, 2020.

[10] Ching-An Cheng, Andrey Kolobov, and Adith Swaminathan. Heuristic-guided reinforcement
learning. Advances in Neural Information Processing Systems, 34:13550–13563, 2021.

[11] Ching-An Cheng, Tengyang Xie, Nan Jiang, and Alekh Agarwal. Adversarially trained actor
critic for offline reinforcement learning. In International Conference on Machine Learning,
pages 3852–3878. PMLR, 2022.

10


---Page Break---
[12] Jonas Degrave, Federico Felici, Jonas Buchli, Michael Neunert, Brendan Tracey, Francesco
Carpanese, Timo Ewalds, Roland Hafner, Abbas Abdolmaleki, Diego de Las Casas, et al.
Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):
414–419, 2022.

[13] Francois Dufour and Tomas Prieto-Rumeau. Finite linear programming approximations of
constrained discounted markov decision processes. SIAM Journal on Control and Optimization,
51(2):1298–1324, 2013.

[14] Francois Dufour and Tomas Prieto-Rumeau. Approximation of average cost markov deci-
sion processes using empirical distributions and concentration inequalities. Stochastics An
International Journal of Probability and Stochastic Processes, 87(2):273–307, 2015.

[15] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adap-
tation of deep networks. In International conference on machine learning, pages 1126–1135.
PMLR, 2017.

[16] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for
deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

[17] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.
Advances in neural information processing systems, 34:20132–20145, 2021.

[18] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error
in actor-critic methods. In International conference on machine learning, pages 1587–1596.
PMLR, 2018.

[19] Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning
without exploration. In International conference on machine learning, pages 2052–2062.
PMLR, 2019.

[20] Javier Garcıa and Fernando Fernández. A comprehensive survey on safe reinforcement learning.
Journal of Machine Learning Research, 16(1):1437–1480, 2015.

[21] Divyansh Garg, Joey Hejna, Matthieu Geist, and Stefano Ermon. Extreme q-learning: Maxent
RL without entropy. In The Eleventh International Conference on Learning Representations,
2023. URL https://openreview.net/forum?id=SJ0Lde3tRL.

[22] Sinong Geng, Aldo Pacchiano, Andrey Kolobov, and Ching-An Cheng. Improving offline RL
by blending heuristics. In The Twelfth International Conference on Learning Representations,
2024. URL https://openreview.net/forum?id=MCl0TLboP1.

[23] Seyed Kamyar Seyed Ghasemipour, Dale Schuurmans, and Shixiang Shane Gu.
Emaq:
Expected-max q-learning operator for simple yet effective offline and online rl. In International
Conference on Machine Learning, pages 3682–3691. PMLR, 2021.

[24] Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree. Regularisation of neural
networks by enforcing lipschitz continuity. Machine Learning, 110:393–416, 2021.

[25] Sven Gronauer and Klaus Diepold. Multi-agent deep reinforcement learning: a survey. Artificial
Intelligence Review, 55(2):895–943, 2022.

[26] Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, and Alois Knoll.
A review of safe reinforcement learning: Methods, theory and applications. arXiv preprint
arXiv:2205.10330, 2022.

[27] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-
policy maximum entropy deep reinforcement learning with a stochastic actor. In International
conference on machine learning, pages 1861–1870. PMLR, 2018.

[28] Ehsan Imani, Eric Graves, and Martha White. An off-policy policy gradient theorem using
emphatic weightings. Advances in neural information processing systems, 31, 2018.

11


---Page Break---
[29] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model:
Model-based policy optimization. Advances in neural information processing systems, 32,
2019.

[30] Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza,
Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement
learning of implicit human preferences in dialog. arXiv preprint arXiv:1907.00456, 2019.

[31] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad
Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii,
a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.

[32] Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad
Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-
based reinforcement learning for atari. arXiv preprint arXiv:1903.00374, 2019.

[33] Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel:
Model-based offline reinforcement learning. Advances in neural information processing systems,
33:21810–21823, 2020.

[34] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014.

[35] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey.
The International Journal of Robotics Research, 32(11):1238–1274, 2013.

[36] Ilya Kostrikov, Rob Fergus, Jonathan Tompson, and Ofir Nachum. Offline reinforcement
learning with fisher divergence critic regularization. In International Conference on Machine
Learning, pages 5774–5783. PMLR, 2021.

[37] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit
q-learning. In International Conference on Learning Representations, 2022. URL https:
//openreview.net/forum?id=68n2s9ZJWF8.

[38] Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-
policy q-learning via bootstrapping error reduction. Advances in Neural Information Processing
Systems, 32, 2019.

[39] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning
for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:
1179–1191, 2020.

[40] Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. Reinforce-
ment learning: State-of-the-art, pages 45–73, 2012.

[41] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444,
2015.

[42] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning:
Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

[43] Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch.
Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural
information processing systems, 30, 2017.

[44] Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu. Mildly conservative q-learning for offline
reinforcement learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun
Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://
openreview.net/forum?id=VYYf6S67pQc.

[45] Xiaoteng Ma, Yiqin Yang, Hao Hu, Qihan Liu, Jun Yang, Chongjie Zhang, Qianchuan Zhao, and
Bin Liang. Offline reinforcement learning with value-based episodic memory. arXiv preprint
arXiv:2110.09796, 2021.

12


---Page Break---
[46] Yecheng Ma, Dinesh Jayaraman, and Osbert Bastani.
Conservative offline distributional
reinforcement learning. Advances in Neural Information Processing Systems, 34:19235–19247,
2021.

[47] Yi Ma, Hongyao Tang, Dong Li, and Zhaopeng Meng. Reining generalization in offline
reinforcement learning via representation distinction. In Thirty-seventh Conference on Neu-
ral Information Processing Systems, 2023. URL https://openreview.net/forum?id=
mVywRIDNIl.

[48] Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman. 1 year, 1000 km: The oxford
robotcar dataset. The International Journal of Robotics Research, 36(1):3–15, 2017.

[49] Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, and Xiangyang Ji. Supported trust region
optimization for offline reinforcement learning. In International Conference on Machine
Learning, pages 23829–23851. PMLR, 2023.

[50] Yixiu Mao, Qi Wang, Chen Chen, Yun Qu, and Xiangyang Ji. Offline reinforcement learning
with ood state correction and ood action suppression. arXiv preprint arXiv:2410.19400, 2024.

[51] Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, and Xiangyang Ji.
Supported value
regularization for offline reinforcement learning. Advances in Neural Information Processing
Systems, 36, 2024.

[52] Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, and Shixiang Gu.
Deployment-efficient reinforcement learning via model-based offline optimization. In In-
ternational Conference on Learning Representations, 2021. URL https://openreview.
net/forum?id=3hGNqpI4WS.

[53] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G
Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

[54] Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Model-based
reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 16(1):1–118,
2023.

[55] Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online
reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.

[56] Andrew Patterson, Samuel Neumann, Martha White, and Adam White. Empirical design in
reinforcement learning. arXiv preprint arXiv:2304.01315, 2023.

[57] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression:
Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[58] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in
neural information processing systems, 1, 1988.

[59] Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Linc Liu,
Junfeng Yang, Lin Lai, Hongyang Qin, et al. Hokoff: Real game dataset from honor of kings
and its offline reinforcement learning benchmarks. In Thirty-seventh Conference on Neural
Information Processing Systems Track on Datasets and Benchmarks, 2023.

[60] Yun Qu, Boyuan Wang, Yuhang Jiang, Jianzhun Shao, Yixiu Mao, Cheems Wang, Chang Liu,
and Xiangyang Ji. Choices are more important than efforts: Llm enables efficient multi-agent
exploration. arXiv preprint arXiv:2410.02511, 2024.

[61] Yuhang Ran, Yi-Chen Li, Fuxiang Zhang, Zongzhang Zhang, and Yang Yu. Policy regularization
with dataset constraint for offline reinforcement learning. In International Conference on
Machine Learning, pages 28701–28717. PMLR, 2023.

[62] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob
Foerster, and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent
reinforcement learning. Journal of Machine Learning Research, 21(178):1–51, 2020.

13


---Page Break---
[63] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Si-
mon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering
atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

[64] Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, and Xiangyang Ji. Counterfactual
conservative q learning for offline multi-agent reinforcement learning. In Thirty-seventh Con-
ference on Neural Information Processing Systems, 2023. URL https://openreview.net/
forum?id=62zmO4mv8X.

[65] Jianzhun Shao, Hongchang Zhang, Yun Qu, Chang Liu, Shuncheng He, Yuhang Jiang, and
Xiangyang Ji. Complementary attention for multi-agent reinforcement learning. In International
Conference on Machine Learning, pages 30776–30793. PMLR, 2023.

[66] Noah Siegel, Jost Tobias Springenberg, Felix Berkenkamp, Abbas Abdolmaleki, Michael
Neunert, Thomas Lampe, Roland Hafner, Nicolas Heess, and Martin Riedmiller. Keep doing
what worked: Behavior modelling priors for offline reinforcement learning. In International
Conference on Learning Representations, 2020. URL https://openreview.net/forum?
id=rke7geHtwH.

[67] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur
Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of
go without human knowledge. nature, 550(7676):354–359, 2017.

[68] Yihao Sun, Jiaji Zhang, Chengxing Jia, Haoxin Lin, Junyin Ye, and Yang Yu. Model-bellman
inconsistency for model-based offline reinforcement learning. In International Conference on
Machine Learning, pages 33177–33194. PMLR, 2023.

[69] Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM
Sigart Bulletin, 2(4):160–163, 1991.

[70] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press,
2018.

[71] Richard S Sutton, A Rupam Mahmood, and Martha White. An emphatic approach to the
problem of off-policy temporal-difference learning. Journal of Machine Learning Research, 17
(73):1–29, 2016.

[72] Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. Revisiting
the minimalist approach to offline reinforcement learning. Advances in Neural Information
Processing Systems, 36, 2024.

[73] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double
q-learning. In Proceedings of the AAAI conference on artificial intelligence, volume 30, 2016.

[74] Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Jun-
young Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster
level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350–354, 2019.

[75] Cheems Wang, Yiqin Lv, Yixiu Mao, Yun Qu, Yi Xu, and Xiangyang Ji. Robust fast adaptation
from adversarially explicit task distribution generation. arXiv preprint arXiv:2407.19523, 2024.

[76] Qi Wang and Herke Van Hoof. Model-based meta reinforcement learning using graph structured
surrogate models and amortized policy search.
In International Conference on Machine
Learning, pages 23055–23077. PMLR, 2022.

[77] Qi Wang, Yiqin Lv, Zheng Xie, Jincai Huang, et al. A simple yet effective strategy to robustify
the meta learning paradigm. Advances in Neural Information Processing Systems, 36, 2024.

[78] Qing Wang, Jiechao Xiong, Lei Han, Han Liu, Tong Zhang, et al. Exponentially weighted
imitation learning for batched historical data. Advances in Neural Information Processing
Systems, 31, 2018.

[79] Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive
policy class for offline reinforcement learning. In The Eleventh International Conference on
Learning Representations, 2023. URL https://openreview.net/forum?id=AHvFDPi-FA.

14


---Page Break---
[80] Ziyu Wang, Alexander Novikov, Konrad Zolna, Josh S Merel, Jost Tobias Springenberg, Scott E
Reed, Bobak Shahriari, Noah Siegel, Caglar Gulcehre, Nicolas Heess, et al. Critic regularized
regression. Advances in Neural Information Processing Systems, 33:7768–7778, 2020.

[81] Albert Wilcox, Ashwin Balakrishna, Jules Dedieu, Wyame Benslimane, Daniel Brown, and Ken
Goldberg. Monte carlo augmented actor-critic for sparse reward deep reinforcement learning
from suboptimal demonstrations. Advances in Neural Information Processing Systems, 35:
2254–2267, 2022.

[82] Robert Wright, Steven Loscalzo, Philip Dexter, and Lei Yu. Exploiting multi-step sample
trajectories for approximate value iteration. In Machine Learning and Knowledge Discovery in
Databases: European Conference, ECML PKDD 2013, Prague, Czech Republic, September
23-27, 2013, Proceedings, Part I 13, pages 113–128. Springer, 2013.

[83] Jialong Wu, Haixu Wu, Zihan Qiu, Jianmin Wang, and Mingsheng Long. Supported policy
optimization for offline reinforcement learning. In Alice H. Oh, Alekh Agarwal, Danielle
Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems,
2022. URL https://openreview.net/forum?id=KCXQ5HoM-fy.

[84] Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement
learning. arXiv preprint arXiv:1911.11361, 2019.

[85] Chenjun Xiao, Han Wang, Yangchen Pan, Adam White, and Martha White. The in-sample soft-
max for offline reinforcement learning. In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.net/forum?id=u-RuvyDYqCM.

[86] Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-
consistent pessimism for offline reinforcement learning. Advances in neural information
processing systems, 34:6683–6694, 2021.

[87] Huaqing Xiong, Tengyu Xu, Lin Zhao, Yingbin Liang, and Wei Zhang. Deterministic policy
gradient: Convergence analysis. In Uncertainty in Artificial Intelligence, pages 2159–2169.
PMLR, 2022.

[88] Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan,
and Xianyuan Zhan. Offline RL with no OOD actions: In-sample learning via implicit value
regularization. In The Eleventh International Conference on Learning Representations, 2023.
URL https://openreview.net/forum?id=ueYYgo2pSSU.

[89] Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. RORL:
Robust offline reinforcement learning via conservative smoothing. In Alice H. Oh, Alekh
Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information
Processing Systems, 2022. URL https://openreview.net/forum?id=_QzJJGH_KE.

[90] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea
Finn, and Tengyu Ma. Mopo: Model-based offline policy optimization. Advances in Neural
Information Processing Systems, 33:14129–14142, 2020.

[91] Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea
Finn. Combo: Conservative offline model-based policy optimization. Advances in neural
information processing systems, 34:28954–28967, 2021.

[92] Hongchang Zhang, Yixiu Mao, Boyuan Wang, Shuncheng He, Yi Xu, and Xiangyang Ji.
In-sample actor critic for offline reinforcement learning. In The Eleventh International Con-
ference on Learning Representations, 2023. URL https://openreview.net/forum?id=
dfDv0WU853R.

[93] Wenxuan Zhou, Sujay Bajracharya, and David Held. Plas: Latent action space for offline
reinforcement learning. In Conference on Robot Learning, pages 1719–1735. PMLR, 2021.

15


---Page Break---
A
Extended Related Work

Model-free offline RL.
In offline RL, a fixed dataset is provided and no further interactions are
allowed [40, 42]. As a result, conventional off-policy RL algorithms suffer from the extrapolation
error due to OOD actions and exhibit poor performance [19]. To address this challenge, various
offline RL algorithms have been developed, primarily categorized into model-free and model-based
approaches. In model-free solutions, value regularization methods introduce conservatism in value
estimation through direct penalization [39, 36, 46, 86, 11, 64, 51], or via value ensembles [2, 3, 89].
Policy constraint approaches enforce proximity between the trained policy and the behavior policy,
either explicitly via divergence penalties [84, 38, 30, 17, 83], implicitly by weighted behavior
cloning [9, 57, 55, 80, 49], or directly through specific parameterization of the policy [19, 23, 93].
Some recent efforts focus on learning the optimal policy within the dataset’s support (known as in-
support or in-sample optimal policy) in a theoretically sound manner [49, 51, 83]. These approaches
are less influenced by the the dataset’s average quality. Another popular branch of algorithms opts for
in-sample learning, which formulates the Bellman target without querying the values of any unseen
actions [7, 45, 37, 85, 92, 88, 21]. Among these, OneStep RL [7] evaluates the behavior policy
via SARSA [70] and performs only one step of constrained policy improvement without off-policy
evaluation. IQL [37] modifies the SARSA update, using expectile regression to approximate an
upper expectile of the value distribution and enables multi-step dynamic programming. Following
IQL, several recent works such as InAC [85], IAC [92], XQL [21], and SQL [88] have developed
different in-sample learning frameworks, further enhancing the performance of in-sample learning
approaches. However, this work shows that the performance of in-sample approaches is confined
by eschewing generalization beyond the offline dataset. In contrast, the proposed approach DMG
utilizes doubly mild generalization to appropriately exploit generalization and achieves significantly
stronger performance across tasks.

Model-based offline RL.
Model-based offline RL methods involve training an environmental
dynamics model, from which synthetic data is generated to facilitate policy optimization [69, 29, 32].
In the context of offline RL, algorithms such as MOPO [90] and MOReL [33] propose to estimate
the uncertainty within the trained model and subsequently impose penalties or constraints on state-
action pairs characterized by high uncertainty levels, thus achieving conservatism in the learning
process. Some model-based approaches incorporate conservatism in a similar way to those model-
free ones. For example, COMBO [91] leverages value penalization, while BREMEN [52] employs
behavior regularization. More recently, MOBILE [68] introduces uncertainty quantification via the
inconsistency of Bellman estimations within a learned dynamics ensemble. SCAS [50] proposes a
generic model-based regularizer that unifies OOD state correction and OOD action suppression in
offline RL. However, typical model-based methods often involve heavy computational overhead [29],
and their effectiveness hinges on the accuracy of the trained dynamics model [54].

Recently, Bose et al. [5] explores multi-task offline RL from the perspective of representation learning
and introduced a notion of neighborhood occupancy density. The neighborhood occupancy density
at a given stata-action pair in the dataset for a source task is defined as the fraction of points in the
dataset within a certain distance from that stata-action pair in the representation space. Bose et al.
[5] use this concept to bound the representational transfer error in the downstream target task. In
contrast, DMG is a wildly compatible idea in offline RL and provides insights into many offline RL
methods. DMG balances the need for generalization with the risk of over-generalization in offline
RL. Generalization to stata-action pairs in the neighborhood of the dataset corresponds to mild action
generalization in the DMG framework.

B
Proofs

In this section, we provide the proofs of all the theories in the paper.

B.1
Proof of Theorem 1

This section presents the formal theorem for the Theorem 1 in the main paper, along with its proof.

We first make several common continuity assumptions for Theorem 1.

16


---Page Break---
Assumption 4 (Lipschitz Q). The learned value function Qθ is KQ-Lipschitz and is upper bounded
by Qmax. ∀s ∼D, ∀a1, a2 ∼A, |Qθ(s, a1) −Qθ(s, a2)| ≤KQ∥a1 −a2∥.
Assumption 5 (Lipschitz Q gradient). The learned value function Qθ is smooth, i.e, has a Kg-
Lipschitz continuous gradient. ∀s ∼D, ∀a1, a2 ∼A, ∥∇θQθ(s, a1) −∇θQθ(s, a2)∥≤Kg∥a1 −
a2∥.
Assumption 6 (Bounded Q and Q gradient). ∀s, a, |Qθ(s, a)| ≤Qmax and ∥∇θQθ(s, a)∥≤gmax.
Assumption 7 (Lipschitz P). The transition dynamics P is KP -Lipschitz. ∀s, s′ ∼S, ∀a1, a2 ∼A,
|P(s′|s, a1) −P(s′|s, a2)| ≤KP ∥a1 −a2∥.
Assumption 8 (Lipschitz R). The reward function R is KR-Lipschitz. ∀s ∼S, ∀a1, a2 ∼A,
|R(s|a1) −R(s, a2)| ≤KR∥a1 −a2∥.

A continuous learned Q function is particularly necessary for the analysis of value function general-
ization. Since we often use neural networks or linear models to parameterize the value function Qθ,
Assumptions 4 and 5 can be relatively easily satisfied [24]. Assumptions 6, 7, and 8 are also common
in the theoretical studies of RL [13, 87, 61] and optimization [6].

Before we start the proof of Theorem 1, we prove the following lemma.
Lemma 2. ∀s ∼D, ∀a1, a2 ∼A, |TuQθ(s, a1) −TuQθ(s, a2)| ≤KT ∥a1 −a2∥. where KT is a
positive bounded constant.

Proof. ∀s ∼D, ∀a1, a2 ∼A,

|TuQθ(s, a1) −TuQθ(s, a2)|

=
R(s, a1) −R(s, a2) + γEs′∼P (·|s,a1)


max
a′∼u(·|s′) Q(s′, a′)

−γEs′∼P (·|s,a2)


max
a′∼u(·|s′) Q(s′, a′)


≤|R(s, a1) −R(s, a2)| + γ
Es′∼P (·|s,a1)


max
a′∼u(·|s′) Q(s′, a′)

−Es′∼P (·|s,a2)


max
a′∼u(·|s′) Q(s′, a′)


= |R(s, a1) −R(s, a2)| + γ



X

s′
(P(s′|s, a1) −P(s′|s, a2))
max
a′∼u(·|s′) Q(s′, a′)



≤|R(s, a1) −R(s, a2)| + γ
X

s′
|(P(s′|s, a1) −P(s′|s, a2))|

max
a′∼u(·|s′) Q(s′, a′)


≤KR∥a1 −a2∥+ γ
X

s′
KP ∥a1 −a2∥Qmax

=(KR + γKP |S|Qmax)∥a1 −a2∥

where the last inequality holds by Assumptions 6, 7, and 8.

Therefore, for any s ∼D, a1, a2 ∼A, it holds that

|TuQθ(s, a1) −TuQθ(s, a2)| ≤KT ∥a1 −a2∥,
(17)

where KT := KR + γKP |S|Qmax is a positive bounded constant.

We restate the scenario analyzed in Theorem 1: Qθ is updated to Qθ′ by one gradient step on a single
state-action pair (s, a) ∈D, which affects the Q-value of an arbitrary state-action pair (s, ˜a) /∈D.
The parameter update is

θ′ = θ + α(TuQθ(s, a) −Qθ(s, a))∇θQθ(s, a)
(18)

where α is the learning rate.

Now we start the proof of Theorem 1 in the main paper.
Theorem 6 (Theorem 1). Under Assumptions 4 to 8, the following equation holds when the learning
rate α is sufficiently small and ˜a is sufficiently close to a:

Qθ′(s, ˜a) = Qθ(s, ˜a) + C1 (TuQθ(s, ˜a) −Qθ(s, ˜a) + C2∥˜a −a∥) + O
 
∥θ′ −θ∥2
(19)

where C1 ∈[0, 1] and C2 ∈[−KQ −KR −γKP |S|Qmax, KQ + KR + γKP |S|Qmax].

17


---Page Break---
Proof. We formalize Qθ′(s, ˜a) by Taylor expansion at the parameter θ:

Qθ′(s, ˜a) = Qθ(s, ˜a) + ∇θQθ(s, ˜a)⊤(θ′ −θ) + O
 
∥θ′ −θ∥2
(20)

By plugging Eq. (18) into Eq. (20), we have

Qθ′(s, ˜a) = Qθ(s, ˜a)+α∇θQθ(s, ˜a)⊤∇θQθ(s, a) (TuQθ(s, a) −Qθ(s, a))+O
 
∥θ′ −θ∥2
(21)

According to Assumption 4 and Lemma 2, it holds that
|Qθ(s, ˜a) −Qθ(s, a)| ≤KQ∥˜a −a∥
(22)
|TuQθ(s, ˜a) −TuQθ(s, a)| ≤KT ∥˜a −a∥
(23)
where KT := KR + γKP |S|Qmax is a positive bounded constant.

Therefore,
|(TuQθ(s, ˜a) −Qθ(s, ˜a)) −(TuQθ(s, a) −Qθ(s, a))|
= |(TuQθ(s, ˜a) −TuQθ(s, a)) + (Qθ(s, a) −Qθ(s, ˜a))|
≤|(TuQθ(s, ˜a) −TuQθ(s, a))| + |(Qθ(s, a) −Qθ(s, ˜a))|
≤KT ∥˜a −a∥+ KQ∥˜a −a∥

As a result,
TuQθ(s, a) −Qθ(s, a) ≤TuQθ(s, ˜a) −Qθ(s, ˜a) + (KQ + KT )∥˜a −a∥
TuQθ(s, a) −Qθ(s, a) ≥TuQθ(s, ˜a) −Qθ(s, ˜a) −(KQ + KT )∥˜a −a∥

Thus we can let
TuQθ(s, a) −Qθ(s, a) = TuQθ(s, ˜a) −Qθ(s, ˜a) + C2∥˜a −a∥,
(24)
where C2 ∈[−KQ −KT , KQ + KT ] is a bounded constant.

Now we shift our focus to α∇θQθ(s, ˜a)⊤∇θQθ(s, a). Let v = ∇θQθ(s, ˜a)−∇θQθ(s, a). According
to the smoothness of Qθ in Assumption 5, it holds that
∥v∥= ∥∇θQθ(s, ˜a) −∇θQθ(s, a)∥≤Kg∥˜a −a∥.
(25)

Therefore,
∇θQθ(s, ˜a)⊤∇θQθ(s, a)

=(∇θQθ(s, a) + v)⊤∇θQθ(s, a)

=∥∇θQθ(s, a)∥2 + v⊤∇θQθ(s, a)

≥∥∇θQθ(s, a)∥2 −∥v∥∥∇θQθ(s, a)∥

≥∥∇θQθ(s, a)∥2 −Kg∥˜a −a∥∥∇θQθ(s, a)∥

Therefore, for sufficiently close ˜a and a such that ∥˜a −a∥≤∥∇θQθ(s, a)∥/Kg, it holds that
α∇θQθ(s, ˜a)⊤∇θQθ(s, a) ≥0.

On the other hand, because ∥∇θQθ∥is bounded by gmax according to Assumption 6, it holds that

α∇θQθ(s, ˜a)⊤∇θQθ(s, a) ≤αg2
max

By choosing a small learning rate α such that α ≤1/g2
max,

α∇θQθ(s, ˜a)⊤∇θQθ(s, a) ≤1

In such cases (sufficiently close ˜a and a, and sufficiently small α), let

C1 := α∇θQθ(s, ˜a)⊤∇θQθ(s, a)
(26)
We have C1 ∈[0, 1].

By plugging Equations (24) and (26) into Equation (21), the following equation holds.
Qθ′(s, ˜a) = Qθ(s, ˜a) + C1 (TuQθ(s, ˜a) −Qθ(s, ˜a) + C2∥˜a −a∥) + O
 
∥θ′ −θ∥2
(27)
where C1 ∈[0, 1], C2 ∈[−KQ −KT , KQ + KT ], and KT = KR + γKP |S|Qmax.

This concludes the proof.

18


---Page Break---
B.2
Proofs under Oracle Generalization

We first restate the several definitions in the main paper.

Definition 4 (Mildly generalized policy, Definition 1). Policy ˜β is termed a mildly generalized policy
if it satisfies

supp(ˆβ(·|s)) ⊆supp(˜β(·|s)), and
max
a1∼˜β(·|s)
min
a2∼ˆβ(·|s)
∥a1 −a2∥≤ϵa,
(28)

where ˆβ is the empirical behavior policy in the offline dataset.

Definition 5 (Definition 2). The Doubly Mildly Generalization (DMG) operator is defined as

TDMGQ(s, a) := R(s, a) + γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
Q(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
Q(s′, a′)

#

(29)
where ˆβ is the empirical behavior policy in the dataset and ˜β is a mildly generalized policy.

Definition 6 (Definition 3). The In-sample Q Learning operator [37] is defined as

TInQ(s, a) := R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
Q(s′, a′)

#

(30)

where ˆβ is the empirical behavior policy in the dataset.

In this subsection, we assume that the learned value function can make oracle generalization in the
mild generalization area ˜β(a|s) > 0, which is formally defined as follows.

Assumption 9 (Oracle generalization, Assumption 1). The generalization of learned Q functions in
the mild generalization area ˜β(a|s) > 0 reflects the true value updates according to TDMG. In other
words, TDMG is well defined in the mild generalization area ˜β(a|s) > 0.

This assumption can be considered reasonable according to the results presented in Theorem 6 above.
In such cases, we can analyze the dynamic programming properties of operators TIn and TDMG.

Before we start the proofs of Lemma 1 and Theorem 2 in the main paper, we prove a lemma.

Lemma 3. For any function f1, f2, any variant x ∈X, the following inequality holds:
max
x∈X f1(x) −max
x∈X f2(x)
 ≤max
x∈X |f1(x) −f2(x)| .
(31)

Proof. Define x1 := argmaxx∈X f1(x) and x2 := argmaxx∈X f2(x).

According to the definition, the following inequality holds:

f1(x2) −f2(x2) ≤f1(x1) −f2(x2) ≤f1(x1) −f2(x1)
(32)

Therefore,
max
x∈X f1(x) −max
x∈X f2(x)


= |f1(x1) −f2(x2)|
≤max {|f1(x2) −f2(x2)| , |f1(x1) −f2(x1)|}
≤max
x∈X |f1(x) −f2(x)|

This concludes the proof of Lemma 3.

Lemma 4 (Lemma 1). TIn is a γ-contraction operator in the in-sample area ˆβ(a|s) > 0 under the
L∞norm.

19


---Page Break---
Proof. Let f1 and f2 be two arbitrary functions.

For all (s, a) s.t. ˆβ(a|s) > 0, we have

|TInf1(s, a) −TInf2(s, a)|

=

R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f1(s′, a′)

#

−R(s, a) −γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

=γ

Es′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

≤γEs′∼P (·|s,a)

"
max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)



#

≤γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
|f1(s′, a′) −f2(s′, a′)|

#

≤γ
max
(s,a): ˆβ(a|s)>0
|f1(s, a) −f2(s, a)|

where the second inequality holds by Lemma 3.

Therefore, in the in-sample area ˜β(a|s) > 0, TIn is a γ-contraction operator under the L∞norm.
This concludes the proof for TIn.

Thus, by repeatedly applying TIn, any initial Q function can converge to the unique fixed point Q∗
In.
We denote its induced policy by π∗
In:

Q∗
In(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
Q∗
In(s′, a′)

#

,
ˆβ(a|s) > 0,
(33)

π∗
In(s) := argmax
a∼ˆβ(·|s)
Q∗
In(s, a).
(34)

Here, Q∗
In is known as the in-sample optimal value function [38, 37], which is the value function of
the in-sample optimal policy π∗
In. We refer readers to [83, 37, 49, 51] for more discussions on the
in-sample or in-support optimality.

Now we start the proof of Theorem 2 in the main paper.
Theorem 7 (Contraction, Theorem 2). Under Assumption 9, TDMG is a γ-contraction operator in
the mild generalization area ˜β(a|s) > 0 under the L∞norm. Therefore, by repeatedly applying
TDMG, any initial Q function can converge to the unique fixed point Q∗
DMG.

Proof. By the oracle generalization assumption (Assumption 9), TDMG is well defined in the mild
generalization area ˜β(a|s) > 0.

Let f1 and f2 be two arbitrary functions. For all (s, a) s.t. ˜β(a|s) > 0, we have

TDMGf1(s, a) −TDMGf2(s, a)

=R(s, a) + γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f1(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
f1(s′, a′)

#

−R(s, a) −γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f2(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

=γλEs′∼P (·|s,a)

"

max
a′∼˜β(·|s′)
f1(s′, a′) −
max
a′∼˜β(·|s′)
f2(s′, a′)

#

+ γ(1 −λ)Es′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

20


---Page Break---
Therefore, for all (s, a) s.t. ˜β(a|s) > 0,

|TDMGf1(s, a) −TDMGf2(s, a)|

≤

γλEs′∼P (·|s,a)

"

max
a′∼˜β(·|s′)
f1(s′, a′) −
max
a′∼˜β(·|s′)
f2(s′, a′)

#

+

γ(1 −λ)Es′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

≤γλEs′∼P (·|s,a)

"
max
a′∼˜β(·|s′)
f1(s′, a′) −
max
a′∼˜β(·|s′)
f2(s′, a′)



#

+ γ(1 −λ)Es′∼P (·|s,a)

"
max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)



#

≤γλEs′∼P (·|s,a)

"

max
a′∼˜β(·|s′)
|f1(s′, a′) −f2(s′, a′)|

#

+ γ(1 −λ)Es′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
|f1(s′, a′) −f2(s′, a′)|

#

≤γλEs′∼P (·|s,a)
max
(s,a): ˜β(a|s)>0
|f1(s, a) −f2(s, a)|

+ γ(1 −λ)Es′∼P (·|s,a)
max
(s,a): ˜β(a|s)>0
|f1(s, a) −f2(s, a)|

=γ
max
(s,a): ˜β(a|s)>0
|f1(s, a) −f2(s, a)|

where the third inequality holds by Lemma 3.

Therefore, in the mild generalization area ˜β(a|s) > 0, TDMG is a γ-contraction operator under the
L∞norm. This concludes the proof.

As a result, by repeatedly applying TDMG, any initial Q function can converge to the unique fixed
point Q∗
DMG. We denote the induced policy of Q∗
DMG by π∗
DMG.

Q∗
DMG(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼˜β(·|s′)
Q∗
DMG(s′, a′)

#

,
˜β(a|s) > 0,
(35)

π∗
DMG(s) := argmax
a∼˜β(·|s)
Q∗
DMG(s, a).
(36)

Before we start the proof of Theorem 3, we prove two lemmas.
Lemma 5. Under Assumption 9, for any function f, the following inequality holds:

TDMGf(s, a) ≥TInf(s, a), ∀(s, a) s.t. ˜β(a|s) > 0.
(37)

Proof. The oracle generalization assumption (Assumption 9) implies that TIn is also well defined in
the mild generalization area ˜β(a|s) > 0. Because supp(ˆβ(·|s)) ⊆supp(˜β(·|s)), TIn requires less
information than TDMG. Therefore, TDMG being well defined in the mild generalization area implies
TIn also being well defined in that area.

According to the definitions, for all (s, a) s.t. ˜β(a|s) > 0,

TDMGf(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
f(s′, a′)

#

(38)

TInf(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f(s′, a′)

#

(39)

21


---Page Break---
Therefore, for all (s, a) s.t. ˜β(a|s) > 0, we have

TDMGf(s, a) −TInf(s, a)

=γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f(s′, a′) −λ
max
a′∼ˆβ(·|s′)
f(s′, a′)

#

≥0

where the last inequality holds because ˜β has a wider support than ˆβ.

Lemma 6. Under Assumption 9, for any function f1, f2 such that f1(s, a)
≥
f2(s, a),
∀(s, a) s.t. ˜β(a|s) > 0, the following inequality holds:

TDMGf1(s, a) ≥TDMGf2(s, a), ∀(s, a) s.t. ˜β(a|s) > 0
(40)

Proof. By Assumption 9, TDMG is well defined in the mild generalization area ˜β(a|s) > 0.

According to the definition, for all (s, a) s.t. ˜β(a|s) > 0,

TDMGf(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
f(s′, a′)

#

(41)

f1 and f2 satisfy
f1(s, a) ≥f2(s, a), ∀(s, a) s.t. ˜β(a|s) > 0.
(42)

Therefore, for all (s, a) s.t. ˜β(a|s) > 0,

TDMGf1(s, a) −TDMGf2(s, a)

=γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f1(s′, a′) −λ
max
a′∼˜β(·|s′)
f2(s′, a′)

#

+ γEs′∼P (·|s,a)

"

(1 −λ)
max
a′∼ˆβ(·|s′)
f1(s′, a′) −(1 −λ)
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

≥0

Now we start the proof of Theorem 3 in the main paper.
Theorem 8 (Performance, Theorem 3). Under Assumption 9, the value functions of π∗
DMG and π∗
In
satisfy:
V π∗
DMG(s) ≥V π∗
In(s),
∀s ∈D.
(43)

Proof. We first prove the following inequality:

(TDMG)kf(s, a) ≥(TIn)kf(s, a), ∀k ∈Z+, ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.
(44)

When k = 1, according to Lemma 5, it holds that

(TDMG)1f(s, a) ≥(TIn)1f(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.

Suppose when k = i, the following inequality holds:

(TDMG)if(s, a) ≥(TIn)if(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.

Then (TDMG)if and (TIn)if are the two functions f1, f2 that satisfy the condition in Lemma 6.
Therefore, by Lemma 6, it holds that

TDMG(TDMG)if(s, a) ≥TDMG(TIn)if(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.
(45)

22


---Page Break---
Now considering (TIn)if as the function f in Lemma 5. By Lemma 5, it holds that

TDMG(TIn)if(s, a) ≥TIn(TIn)if(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.
(46)

Combining Equations (45) and (46), we have

(TDMG)i+1f(s, a) ≥(TIn)i+1f(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.

Therefore, for all k ∈Z+, the following inequality holds:

(TDMG)kf(s, a) ≥(TIn)kf(s, a), ∀f, ∀(s, a) s.t. ˜β(a|s) > 0.
(47)

Lemma 4 states that TIn is a γ-contraction operator in the in-sample area ˆβ(a|s) > 0. Thus we have

Q∗
In(s, a) = lim
k→∞(TIn)kf(s, a), ∀(s, a) s.t. ˆβ(a|s) > 0.
(48)

Under Assumption 9, Theorem 7 states that TDMG is a γ-contraction operator in the mild generaliza-
tion area ˜β(a|s) > 0. Thus we have

Q∗
DMG(s, a) = lim
k→∞(TDMG)kf(s, a), ∀(s, a) s.t. ˜β(a|s) > 0.
(49)

As ˜β has a wider support than ˆβ, supp(ˆβ(·|s)) ⊆supp(˜β(·|s)), the following inequality holds by
combining Equations (47) to (49):

Q∗
DMG(s, a) ≥Q∗
In(s, a), ∀(s, a) s.t. ˆβ(a|s) > 0.
(50)

Therefore, for any s ∼D,

V π∗
DMG(s) = V ∗
DMG(s) = Q∗
DMG(s, π∗
DMG(s))
≥Q∗
DMG(s, π∗
In(s))

≥Q∗
In(s, π∗
In(s)) = V ∗
In(s) = V π∗
In(s)
where the first inequality holds because π∗
DMG(s) := argmaxa∼˜β(·|s) Q∗
DMG(s, a) and π∗
In(s) ∈
ˆβ(·|s) (thus π∗
In(s) ∈˜β(·|s)), and the second inequality holds by Equation (50).

This concludes the proof.

Theorem 8 indicates that the policy induced by the DMG operator can behave better than the in-sample
optimal policy under the oracle generalization condition.

B.3
Proofs under Worst-case Generalization

In this section, we focus on the analyses in the worst-case generalization scenario, where the learned
value functions may exhibit poor generalization in the mild generalization area ˜β(a|s) > 0. In other
words, this section considers that TDMG is only defined in the in-sample area ˆβ(a|s) > 0 and the
learned value functions may have any generalization error at other state-action pairs. In this case, we
use the notation ˆTDMG for differentiation.

In this case, we make the following continuity assumptions about the learned Q function and the
transition dynamics P.
Assumption 10 (Lipschitz Q). The learned Q function is KQ-Lipschitz. ∀s ∼D, ∀a1, a2 ∼A,
|Q(s, a1) −Q(s, a2)| ≤KQ∥a1 −a2∥
Assumption 11 (Lipschitz P). The transition dynamics P is KP -Lipschitz. ∀s, s′ ∼S, ∀a1, a2 ∼A,
|P(s′|s, a1) −P(s′|s, a2)| ≤KP ∥a1 −a2∥

For Assumption 10, a continuous learned Q function is particularly necessary for the analysis of value
function generalization and can be relatively easily satisfied [24], since we often use neural networks
or linear models to parameterize the value function. For Assumption 11, continuous transition
dynamics is also a standard assumption in the theoretical studies of RL [13, 14, 87, 61]. Several
previous works assume the transition to be Lipschitz continuous with respect to (w.r.t) both state and
action [13, 14]. In our paper, we need the Lipschitz continuity to hold only w.r.t. action.

Before we start the proof of Theorem 4, we prove two lemmas.

23


---Page Break---
Lemma 7. Under Assumption 10, for any function f and s ∼D, the following inequality holds:

max
a∼˜β(·|s)
f(s, a) −
max
a∼ˆβ(·|s)
f(s, a) ≤ϵaKQ.
(51)

Proof. For any s ∼D, we define ˜a∗, ˆa∗, ˆa′ as follows:

˜a∗= argmax
a∼˜β(·|s)
f(s, a)
(52)

ˆa∗= argmax
a∼ˆβ(·|s)
f(s, a)
(53)

ˆa′ = argmin
a∼ˆβ(·|s)
∥˜a∗−a∥
(54)

According to the definition of mildly generalized policy ˜β (Definition 4), it holds that ∥˜a∗−ˆa′∥≤ϵa.
Further by Assumption 10, it holds that

|f(s, ˜a∗) −f(s, ˆa′)| ≤KQ∥˜a∗−ˆa′∥≤ϵaKQ, ∀s ∼D.

Therefore,
f(s, ˜a∗) −f(s, ˆa∗) ≤f(s, ˜a∗) −f(s, ˆa′) ≤ϵaKQ, ∀s ∼D.

Lemma 8. For any function f1, f2 such that f1(s, a) ≥f2(s, a), ∀(s, a) s.t. ˆβ(a|s) > 0, the
following inequality holds:

TInf1(s, a) ≥TInf2(s, a), ∀(s, a) s.t. ˆβ(a|s) > 0.
(55)

Proof. According to the definitions, for all (s, a) s.t. ˆβ(a|s) > 0,

TInf(s, a) = R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f(s′, a′)

#

(56)

f1 and f2 satisfy
f1(s, a) ≥f2(s, a), ∀(s, a) s.t. ˆβ(a|s) > 0.

Therefore, for all (s, a) s.t. ˆβ(a|s) > 0,

TInf1(s, a) −TInf2(s, a)

=γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f1(s′, a′) −
max
a′∼ˆβ(·|s′)
f2(s′, a′)

#

≥0

Now we start the proof of Theorem 4 in the main paper.

We consider the iteration starting from arbitrary function Q0: ˆQk
DMG = ˆTDMG ˆQk−1
DMG and Qk
In =
TInQk−1
In , ∀k ∈Z+. The possible value of ˆQk
DMG is upper bounded by the following results.

Theorem 9 (Limited over-estimation, Theorem 4). Under Assumption 10, the learned Q function of
DMG by iterating ˆTDMG satisfies the following inequality

Qk
In(s, a) ≤ˆQk
DMG(s, a) ≤Qk
In(s, a) + λϵaKQγ

1 −γ
(1 −γk), ∀s, a ∼D, ∀k ∈Z+.
(57)

24


---Page Break---
Proof. Under worst-case generalization, ˆTDMG is only defined in the area ˆβ(a|s) > 0, i.e., the
dataset, and may have any generalization error at other (s, a).

For any function f and any s, a ∼D,

ˆTDMGf(s, a) −TInf(s, a)

=R(s, a) + γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f(s′, a′) + (1 −λ)
max
a′∼ˆβ(·|s′)
f(s′, a′)

#

−R(s, a) −γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
f(s′, a′)

#

=γEs′∼P (·|s,a)

"

λ
max
a′∼˜β(·|s′)
f(s′, a′) −λ
max
a′∼ˆβ(·|s′)
f(s′, a′)

#

≤γλϵaKQ

where the last inequality holds by Lemma 7.

On the other hand, because ˜β has a wider support than ˆβ, we also have

ˆTDMGf(s, a) −TInf(s, a) ≥0

Therefore, for any function f, the following inequality holds:

TInf(s, a) ≤ˆTDMGf(s, a) ≤TInf(s, a) + γλϵaKQ, ∀s, a ∼D.
(58)

Let f in Equation (58) be Q0. We have

Q1
In(s, a) ≤ˆQ1
DMG(s, a) ≤Q1
In(s, a) + λϵaKQγ

1 −γ
(1 −γ), ∀s, a ∼D.
(59)

This is the same as Equation (57) with k = 1. Therefore, Equation (57) holds when k = 1.

Suppose when k = i, Equation (57) holds:

Qi
In(s, a) ≤ˆQi
DMG(s, a) ≤Qi
In(s, a) + λϵaKQγ

1 −γ
(1 −γi), ∀s, a ∼D.
(60)

Then let f in Equation (58) be ˆQi
DMG. We have

TIn ˆQi
DMG(s, a) ≤ˆQi+1
DMG(s, a) = ˆTDMG ˆQi
DMG(s, a) ≤TIn ˆQi
DMG(s, a) + γλϵaKQ, ∀s, a ∼D.
(61)

On the one hand, according to Lemma 8 and Equation (60), for any s, a ∼D, we have

TIn ˆQi
DMG(s, a)

≤TIn


Qi
In(s, a) + λϵaKQγ

1 −γ
(1 −γi)


=R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)


Qi
In(s′, a′) + λϵaKQγ

1 −γ
(1 −γi)
#

=R(s, a) + γEs′∼P (·|s,a)

"

max
a′∼ˆβ(·|s′)
Qi
In(s′, a′)

#

+ γ λϵaKQγ

1 −γ
(1 −γi)

=TInQi
In(s, a) + γ λϵaKQγ

1 −γ
(1 −γi)

=Qi+1
In (s, a) + γ λϵaKQγ

1 −γ
(1 −γi)
(62)

25


---Page Break---
Combining Equations (61) and (62), for any s, a ∼D, we have

ˆQi+1
DMG(s, a)

≤Qi+1
In (s, a) + γ λϵaKQγ

1 −γ
(1 −γi) + γλϵaKQ

=Qi+1
In (s, a) + λϵaKQγ
γ(1 −γi)

1 −γ
+ 1


=Qi+1
In (s, a) + λϵaKQγ

1 −γ
(1 −γi+1)

On the other hand, according to Lemma 8 and Equation (60), for any s, a ∼D, we have

TIn ˆQi
DMG(s, a) ≥TInQi
In(s, a) = Qi+1
In (s, a)
(63)

Combining Equations (61) and (63), for any s, a ∼D, we have

ˆQi+1
DMG(s, a) ≥Qi+1
In (s, a).
(64)

Hence, Equation (57) still holds when k = i + 1:

Qi+1
In (s, a) ≤ˆQi+1
DMG(s, a) ≤Qi+1
In (s, a) + λϵaKQγ

1 −γ
(1 −γi+1), ∀s, a ∼D.
(65)

Therefore, Equation (57) holds for all k ∈Z+, which concludes the proof.

Since in-sample training eliminates extrapolation error completely [37, 92], Qk
In can be considered a
relatively accurate estimate. Therefore, Theorem 9 indicates that DMG has limited over-estimation
under the worst generalization case. Moreover, the bound gets tighter as ϵa gets smaller (more mild
action generalization) and λ gets smaller (more mild generalization propagation). This is consistent
with our intuitions in Section 3.2.

Finally, Theorem 5 in the main paper shows that even under worst-case generalization, DMG is
guaranteed to output a safe policy with a performance lower bound.

We give a lemma before we start the proof of Theorem 5,
Lemma 9. Let π1 and π2 be two deterministic policies. Under Assumption 11, the following
inequality holds:
TV (dπ1||dπ2) ≤CKP max
s
∥π1(s) −π2(s)∥
(66)

where C is a positive constant and dπ(s) is the state occupancy induced by π.

dπ(s) = (1 −γ)

∞
X

t=0
γtEπ [I [st = s]] .
(67)

Proof. Please refer to Lemma A.5 in [61] and Lemma 1 in [87].

Theorem 10 (Performance lower bound, Theorem 5). Let ˆπDMG be the learned policy of DMG by
iterating ˆTDMG, π∗be the optimal policy, and ϵD be the inherent performance gap of the in-sample
optimal policy ϵD := J(π∗) −J(π∗
In). Under Assumptions 10 and 11, for sufficiently small ϵa, we
have

J(ˆπDMG) ≥J(π∗) −CKP Rmax

1 −γ
ϵa −ϵD.
(68)

where C is a positive constant.

Proof. Following previous works [38, 83, 37, 49], we define the in-sample optimal policy as π∗
In:

π∗
In(s) = argmax
a∼ˆβ(·|s)
Q∗
In(s, a)
(69)

26


---Page Break---
We also use ϵD to denote the performance gap between the in-sample optimal policy and the globally
optimal policy, which is fixed once the dataset is provided.

ϵD = J(π∗) −J(π∗
In).
(70)

We use ˆQDMG to denote the learned Q function of DMG with sufficient iteration steps ˆQk
DMG,
k →∞. And ˆπDMG is the output policy of ˆQDMG:

ˆπDMG(s) = argmax
a∼˜β(·|s)
ˆQDMG(s, a)
(71)

It holds that

|J(π∗) −J(ˆπDMG)|
=|J(π∗) −J(π∗
In) + J(π∗
In) −J(ˆπDMG)|
≤|J(π∗) −J(π∗
In)| + |J(π∗
In) −J(ˆπDMG)|
=ϵD + |J(π∗
In) −J(ˆπDMG)|
(72)

In the following, we bound the term |J(π∗
In) −J(ˆπDMG)|.

|J(π∗
In) −J(ˆπDMG)|

=

1
1 −γ Es∼dˆπDMG [r(s)] −
1
1 −γ Es∼dπ∗
In[r(s)]


=
1
1 −γ



X

s


dˆπDMG(s) −dπ∗
In(s)

r(s)



≤
1
1 −γ

X

s



dˆπDMG(s) −dπ∗
In(s)
 |r(s)|

≤Rmax

1 −γ TV

dˆπDMG(s)||dπ∗
In(s)


≤Rmax

1 −γ CKP max
s
∥ˆπDMG(s) −π∗
In(s)∥
(73)

where the last inequality holds by Lemma 9.

According to Theorem 9, ˆQDMG satisfies the following inequality:

Q∗
In(s, a) ≤ˆQDMG(s, a) ≤Q∗
In(s, a) + λϵaKQγ

1 −γ
, ∀s, a ∼D.
(74)

It means that for any (s, a) ∼D, with sufficiently small ϵa, ˆQDMG(s, a) sufficiently approximates
Q∗
In(s, a). By Definition 4, ˜β is a mildly generalized policy. That is, for any s ∼D, ˜β satisfies

supp(ˆβ(·|s)) ⊆supp(˜β(·|s)), and
max
a1∼˜β(·|s)
min
a2∼ˆβ(·|s)
∥a1 −a2∥≤ϵa,

As ˆπDMG(s) ∈˜β(·|s), it implies that we can find ain ∈ˆβ(·|s) (in dataset) such that ∥ˆπDMG(s) −
ain∥≤ϵa.

Now suppose ain is not the maximum point of Q∗
In(s, ·) at a certain s. We use π∗
In(s) to denote the
maximum point of Q∗
In(s, ·). Let ϵQ∗
In be the gap between Q∗
In(s, ain) and Q∗
In(s, π∗
In(s)):

ϵQ∗
In(s) := Q∗
In(s, π∗
In(s)) −Q∗
In(s, ain) > 0.
(75)

By Assumption 10 (Lipschitz Q), we have

ˆQDMG(s, ˆπDMG(s)) −ˆQDMG(s, ain) ≤KQ∥ˆπDMG(s) −ain∥≤KQϵa.
(76)

27


---Page Break---
Therefore,
ˆQDMG(s, π∗
In(s)) −ˆQDMG(s, ˆπDMG(s))

≥ˆQDMG(s, π∗
In(s)) −ˆQDMG(s, ain) −KQϵa

≥Q∗
In(s, π∗
In(s)) −Q∗
In(s, ain) −λϵaKQγ

1 −γ
−KQϵa

=ϵQ∗
In(s) −λϵaKQγ

1 −γ
−KQϵa

where the first inequality holds by Equation (76), the second inequality holds by Equation (74), and
the last equality holds by Equation (75).

Hence, for sufficiently small ϵa such that ϵQ∗
In(s) −λϵaKQγ

1−γ
−KQϵa > 0, i.e.,

ϵa <
(1 −γ)ϵQ∗
In(s)
KQ(1 −γ + λγ),
(77)

it holds that ˆQDMG(s, π∗
In(s)) −ˆQDMG(s, ˆπDMG(s)) > 0. As π∗
In(s) ∈ˆβ(·|s), it also satisfies
π∗
In(s) ∈˜β(·|s). This contradicts the definition of ˆπDMG(s) in Equation (71):

ˆπDMG(s) = argmax
a∼˜β(·|s)
ˆQDMG(s, a)

Therefore, ain is the maximum point of Q∗
In(s, ·). In other words, the maximum point of Q∗
In(s, ·)
(denoted by π∗
In(s)) is the closest neighbor of ˆπDMG(s) in the dataset (ˆβ(·|s) > 0):

π∗
In(s) = argmin
a∼ˆβ(·|s)
∥a −ˆπDMG(s)∥

As ˆπDMG(s) ∈˜β(·|s), the following inequality holds by Definition 4:

∥ˆπDMG(s) −π∗
In(s)∥≤ϵa.

Therefore, we have

|J(π∗
In) −J(ˆπDMG)| ≤Rmax

1 −γ CKP ϵa.
(78)

By combining Equations (72) and (78), we have

J(ˆπDMG) ≥J(π∗) −CKP Rmax

1 −γ
ϵa −ϵD.
(79)

This concludes the proof.

C
Experimental Details

C.1
Experimental Details in Offline Experiments

Our evaluation criteria follow those used in most previous works. For the Gym locomotion tasks,
we average returns over 10 evaluation trajectories and 5 random seeds, while for the AntMaze tasks,
we average over 100 evaluation trajectories and 5 random seeds. Following the suggestions in the
benchmark [16], we subtract 1 from the rewards for the AntMaze datasets. And following previous
works [17, 37, 83, 88], we normalize the states in Gym locomotion datasets. We choose TD3 [18]
as our base algorithm and optimize a deterministic policy. Thus we replace the log likelihood in
Eq. (14) with mean squared error in practice, which is equivalent to optimizing a Gaussian policy
with fixed variance [17]. The reported results are the normalized scores, which are offered by the
D4RL benchmark [16] to measure how the learned policy compared with random and expert policy:

D4RL score = 100 × learned policy return −random policy return

expert policy return −random policy return

28


---Page Break---
Table 5: Hyperparameters of DMG.

Hyperparameter
Value

DMG

Optimizer
Adam [34]
Critic learning rate
3 × 10−4

Actor learning rate
3 × 10−4 with cosine schedule
Batch size
256
Discount factor
0.99
Number of iterations
106
Target update rate
0.005
Number of Critics
2
Penalty coefficient ν
{0.1,10} for Gym-MuJoCo
{0.5} for Antmaze
Mixture coefficient λ
0.25

IQL Specific

Expectile τ
0.7 for Gym-MuJoCo
0.9 for Antmaze
Inverse temperature α
3.0 for Gym-MuJoCo
10.0 for Antmaze

Architecture
Actor
input-256-256-output
Critic
input-256-256-1

As we implement our main algorithm based on IQL [37], we use the hyperparameters suggested in
their paper for fair comparisons, i.e., τ = 0.7 and α = 3 for Gym locomotion tasks and τ = 0.9
and α = 10 for AntMaze tasks. For the results of XQL+DMG and SQL+DMG, we also adopt the
suggested hyperparameters in their papers [21, 88] for fair comparisons. In detail, we choose β in
XQL [21] as 5.0 in medium, medium-replay, and medium-expert datasets, and α in SQL [88] as 2.0
for medium, medium-replay datasets, and 5.0 for medium-expert datasets.

DMG has two main hyperparameters: mixture coefficient λ and penalty coefficient ν. We use
λ = 0.25 for all tasks. We use ν = 0.5 for Antmaze tasks and ν ∈{0.1, 10} for Gym locomotion
tasks (0.1 for medium, medium-replay, random datasets; 10 for expert and medium-expert datasets).
All hyperparameters of DMG are included in Table 5.

C.2
Experimental Details in Offline-to-online Experiments

For online fine-tuning experiments, we first run offline RL for 1 × 106 gradient steps. Then we
continue training while collecting data actively in the environment and adding the data to the replay
buffer. We perform online fine-tuning for 1 × 106 steps with 1 update-to-data (UTD) ratio, and
collect data with exploration noise 0.1 as suggested by TD3 [18]. During offline pre-training, we
fix the mixture coefficient λ = 0.25 and the penalty coefficient ν = 0.5, while in the online phase,
we exponentially adjust λ and ν, as DMG with λ = 1 and ν = 0 corresponds to standard online RL.
In the challenging AntMaze domains characterized by high-dimensional state and action spaces, as
well as sparse rewards, the extrapolation error remains significant even during the online phase [83].
Therefore, we decay λ from 0.25 to 0.5 and ν from 0.5 to 0.005 (1% of its initial value), employing
a decay rate of 0.99 every 1000 gradient steps. Additionally, following previous works [83, 72], we
set γ = 0.995 when fine-tuning on antmaze-large datasets, for both DMG and IQL to ensure a fair
comparison. All other training details remain consistent between the offline RL phase and the online
fine-tuning phase.

D
Additional Experimental Results

D.1
Computational Cost

We test the runtime of offline RL algorithms on halfcheetah-medium-replay-v2 on a GeForce RTX
3090. The results of DMG and other baselines are shown in Figure 3. It takes 1.7h for DMG to finish
the task, which is comparable to the fastest offline RL algorithm TD3BC [17].

29


---Page Break---
DT
MOPO
CQL
AWAC
DMG
IQL
TD3BC
0

5

10

15

16 h

15 h

4 h

2 h
1.7 h
1.5 h
1 h

Runtime of Algorithms

Figure 3: Runtime of algorithms on halfcheetah-medium-replay-v2 on a GeForce RTX 3090.

D.2
Offline Training Results of DMG on More Random Seeds

The experimental results in the main paper show the mean and standard deviation (SD) over five
random seeds. According to [56], we conduct experiments to test DMG on additional random seeds,
reporting 95% confidence interval (CI) over 10 random seeds. Table 6 shows the comparison between
the new results (10seeds/95%CI) and the previously reported results (5seeds/SD in Table 2) on the
D4RL offline training tasks. The results show that our method achieves about the same performance
as under the previous evaluation criterion.

Table 6: Comparison of DMG under different evaluation criteria on D4RL offline training tasks.

Dataset-v2
DMG (5seeds/SD)
DMG (10seeds/95%CI)

halfcheetah-m
54.9±0.2
54.9±0.3
hopper-m
100.6±1.9
100.5±1.0
walker2d-m
92.4±2.7
92.0±1.2
halfcheetah-m-r
51.4±0.3
51.4±0.4
hopper-m-r
101.9±1.4
102.1±0.6
walker2d-m-r
89.7±5.0
90.3±2.8
halfcheetah-m-e
91.1±4.2
92.9±2.1
hopper-m-e
110.4±3.4
109.0±2.6
walker2d-m-e
114.4±0.7
113.9±1.2
halfcheetah-e
95.9±0.3
95.9±0.2
hopper-e
111.5±2.2
111.8±1.3
walker2d-e
114.7±0.4
114.5±0.3
halfcheetah-r
28.8±1.3
28.7±1.2
hopper-r
20.4±10.4
21.6±6.6
walker2d-r
4.8±2.2
7.7±3.0

locomotion total
1182.8
1187.2

antmaze-u
92.4±1.8
91.8±1.6
antmaze-u-d
75.4±8.1
73.0±5.0
antmaze-m-p
80.2±5.1
80.5±2.1
antmaze-m-d
77.2±6.1
76.7±3.6
antmaze-l-p
55.4±6.2
56.7±3.6
antmaze-l-d
58.8±4.5
57.2±2.7

antmaze total
439.4
435.9

D.3
Learning Curves of DMG during Offline Training

Learning curves during offline training on Gym-MuJoCo locomotion tasks and Antmaze tasks are
presented in Figure 4 and Figure 5, respectively. The curves are averaged over 5 random seeds, with
the shaded area representing the standard deviation across seeds.

30


---Page Break---
D.4
Learning Curves of DMG during Online Fine-tuning

Learning curves during online fine-tuning on Antmaze tasks are presented in Figure 6. The curves are
averaged over 5 random seeds, with the shaded area representing the standard deviation across seeds.

E
Broader Impact

Offline reinforcement learning (RL) presents a promising avenue for enhancing and broadening the
practical applicability of RL across various domains including robotics, recommendation systems,
healthcare, and education, characterized by costly or hazardous data collection processes. However, it
is imperative to recognize the potential adverse societal ramifications associated with any offline RL
algorithm. One such concern pertains to the possibility that the offline data utilized for training may
harbor inherent biases, which could subsequently permeate into the acquired policy. Furthermore, it is
essential to contemplate the potential implications of offline RL on employment, given its contribution
to automating tasks conventionally executed by human experts, such as factory automation or au-
tonomous driving. Addressing these challenges is essential for fostering the responsible development
and deployment of offline RL algorithms, with the aim of maximizing their positive impact while
mitigating negative societal consequences.

From an academic perspective, this research scrutinizes offline RL through the lens of generalization,
balancing the need for generalization with the risk of over-generalization. The proposed approach
DMG potentially offers researchers a new perspective on appropriately exploiting generalization in
offline RL. Besides, DMG also holds the promise to be extended to safe RL [1, 26, 20], multi-agent
RL [43, 62, 65, 60, 25], and meta RL [15, 76, 77, 75, 4].

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

halfcheetah-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

halfcheetah-medium-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

halfcheetah-medium-replay-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

halfcheetah-medium-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

25

50

Episode Return

halfcheetah-random-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

hopper-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

hopper-medium-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

hopper-medium-replay-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

hopper-medium-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

25

50

Episode Return

hopper-random-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

walker2d-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

walker2d-medium-expert-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

walker2d-medium-replay-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

walker2d-medium-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

25

50

Episode Return

walker2d-random-v2

Figure 4: Learning curves of DMG on Gym locomotion tasks during offline training. The curves are
averaged over 5 random seeds, with the shaded area representing the standard deviation across seeds.

31


---Page Break---
0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-umaze-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-umaze-diverse-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-medium-play-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-medium-diverse-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-large-play-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-large-diverse-v2

Figure 5: Learning curves of DMG on Antmaze tasks during offline training. The curves are averaged
over 5 random seeds, with the shaded area representing the standard deviation across seeds.

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-umaze-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-umaze-diverse-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-medium-play-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-medium-diverse-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-large-play-v2

0.00
0.25
0.50
0.75
1.00
Gradient Steps (×106)

0

50

100

Episode Return

antmaze-large-diverse-v2

Figure 6: Learning curves of DMG on Antmaze tasks during online fine-tuning. The curves are
averaged over 5 random seeds, with the shaded area representing the standard deviation across seeds.

32


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope.
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
Justification: Please refer to Section 6.
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

33


---Page Break---
Justification: Please refer to Appendix B.
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
Justification: Please refer to Appendix C.
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

34


---Page Break---
Answer: [Yes]

Justification: Please refer to the code in the supplemental material.

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

Justification: Please refer to Appendix C.

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

Justification: The results in the paper are accompanied by standard deviations across multiple
seeds.

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

35


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

Justification: Please refer to Appendix D.1.

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

Justification: The research conducted in the paper conforms, in every respect, with the
NeurIPS Code of Ethics.

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

Justification: Please refer to Appendix E.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

36


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

Justification: The paper poses no such risks.

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

Justification: The creators or original owners of assets used in the paper are properly credited
and the license and terms of use are explicitly mentioned and properly respected.

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

37


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The code is well documented and anonymized.
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
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

38


---Page Break---
