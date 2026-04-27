Reinforcement Learning Gradients as Vitamin for
Online Finetuning Decision Transformers

Kai Yan
Alexander G. Schwing
Yu-Xiong Wang
University of Illinois Urbana-Champaign
{kaiyan3, aschwing, yxw}@illinois.edu
https://github.com/KaiYan289/RL_as_Vitamin_for_Online_Decision_Transformers

Abstract

Decision Transformers have recently emerged as a new and compelling paradigm
for offline Reinforcement Learning (RL), completing a trajectory in an autoregres-
sive way. While improvements have been made to overcome initial shortcomings,
online finetuning of decision transformers has been surprisingly under-explored.
The widely adopted state-of-the-art Online Decision Transformer (ODT) still strug-
gles when pretrained with low-reward offline data. In this paper, we theoretically
analyze the online-finetuning of the decision transformer, showing that the com-
monly used Return-To-Go (RTG) that’s far from the expected return hampers the
online fine-tuning process. This problem, however, is well-addressed by the value
function and advantage of standard RL algorithms. As suggested by our analysis,
in our experiments, we hence find that simply adding TD3 gradients to the fine-
tuning process of ODT effectively improves the online finetuning performance of
ODT, especially if ODT is pretrained with low-reward offline data. These findings
provide new directions to further improve decision transformers.

1
Introduction

While Reinforcement Learning (RL) has achieved great success in recent years [55, 31], it is known
to struggle with several shortcomings, including training instability when propagating a Temporal
Difference (TD) error along long trajectories [14], low data efficiency when training from scratch [67],
and limited benefits from more modern neural network architectures [12]. The latter point differs
significantly from other parts of the machine learning community such as Computer Vision [17] and
Natural Language Processing [11].

To address these issues, Decision Transformers (DTs) [14] have been proposed as an emerging
paradigm for RL, introducing more modern transformer architectures into the literature rather than the
still widely used Multi-Layer Perceptrons (MLPs). Instead of evaluating state and state-action pairs, a
DT considers the whole trajectory as a sequence to complete, and trains on offline data in a supervised,
auto-regressive way. Upon inception, DTs have been improved in various ways, mostly dealing with
architecture changes [37], the token to predict other than return-to-go [22], addressing the problem
of being overly optimistic [46], and the inability to stitch together trajectories [5]. Significant and
encouraging improvements have been reported on those aspects.

However, one fundamental issue has been largely overlooked by the community: offline-to-online
RL using decision transformers, i.e., finetuning of decision transformers with online interactions.
Offline-to-online RL [72, 41] is a widely studied sub-field of RL, which combines offline RL learning
from given, fixed trajectory data and online RL data from interactions with the environment. By
first training on offline data and then finetuning, the agent can learn a policy with much greater data
efficiency, while calibrating the out-of-distribution error from the offline dataset. Unsurprisingly, this
sub-field has become popular in recent years.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
While there are numerous works in the offline-to-online RL sub-field [35, 28, 62], surprisingly
few works have discussed the offline-to-online finetuning ability of decision transformers. While
there is work that discusses finetuning of decision transformers predicting encoded future trajectory
information [64], and work that finetunes pretrained decision transformers with PPO in multi-agent
RL [38], the current widely adopted state-of-the-art is the Online Decision Transformer (ODT) [74]:
the decision transformer training is continued on online data following the same supervised-learning
paradigm as in offline RL. However, this method struggles with low-reward data, as well as with
reaching expert-level performance due to suboptimal trajectories [41] (also see Sec. 4).

To address this issue and enhance online finetuning of decision transformers, we theoretically analyze
the decision transformer based on recent results [7], showing that the commonly used conditioning
on a high Return-To-Go (RTG) that’s far from the expected return hampers results. To fix, we explore
the possibility of using tried-and-true RL gradients. Testing on multiple environments, we find that
simply combining TD3 [21] gradients with the original auto-regressive ODT training paradigm is
surprisingly effective: it improves results of ODT, especially if ODT is pretrained with low-reward
offline data.

Our contributions are summarized as follows:

1) We propose a simple yet effective method to boost the performance of online finetuning of decision
transformers, especially if offline data is of medium-to-low quality;

2) We theoretically analyze the online decision transformer, explain its “policy update” mechanism
when using the commonly applied high target RTG, and point out its struggle to work well with
online finetuning;

3) We conduct experiments on multiple environments, and find that ODT aided by TD3 gradients (and
sometimes even the TD3 gradient alone) are surprisingly effective for online finetuning of decision
transformers.

2
Preliminaries

Markov Decision Process. A Markov Decision Process (MDP) is the basic framework of sequential
decision-making. An MDP is characterized by five components: the state space S, the action space A,
the transition function p, the reward r, and either the discount factor γ or horizon H. MDPs involve
an agent making decisions in discrete steps t ∈{0, 1, 2, . . . }. On step t, the agent receives the current
state st ∈S, and samples an action at ∈A according to its stochastic policy π(at|st) ∈∆(A),
where ∆(A) is the probability simplex over A, or its deterministic policy µ(st) ∈A. Executing
the action yields a reward r(st, at) ∈R, and leads to the evolution of the MDP to a new state st+1,
governed by the MDP’s transition function p(st+1|st, at). The goal of the agent is to maximize
the total reward P

t γtr(st, at), discounted by the discount factor γ ∈[0, 1] for infinite steps, or
PH
t=1 r(st, at) for finite steps. When the agent ends a complete run, it finishes an episode, and the
state(-action) data collected during the run is referred to as a trajectory τ.

Offline and Online RL. Based on the source of learning data, RL can be roughly categorized
into offline and online RL. The former learns from a given finite dataset of state-action-reward
trajectories, while the latter learns from trajectories collected online from the environment. The effort
of combining the two is called offline-to-online RL, which first pre-trains a policy using offline data,
and then continues to finetune the policy using online data with higher efficiency. Our work falls into
the category of offline-to-online RL. We focus on improving the decision transformers, instead of
Q-learning-based methods which are commonly used in offline-to-online RL.

Decision Transformer (DT). The decision transformer represents a new paradigm of offline RL,
going beyond a TD-error framework. It views a trajectory τ as a sequence to be auto-regressively
completed. The sequence interleaves three types of tokens: returns-to-go (RTG, the target total
return), states, and actions. At step t, the past sequence of context length K is given as the input, i.e.,
the input is (RTGt−K, st−k, at−k, . . . , RTGt, st), and an action is predicted by the auto-regressive
model, which is usually implemented with a GPT-like architecture [11]. The model is trained via
supervised learning, considering the past K steps of the trajectory along with the current state and
the current return-to-go as the feature, and the sequence of all actions a in a segment as the labels.
At evaluation time, a desired return RTGeval is specified, since the ground truth future return
RTGreal isn’t known in advance.

2


---Page Break---
Online Decision Transformer (ODT). ODT has two stages: offline pre-training which is identical to
classic DT training, and online finetuning where trajectories are iteratively collected and the policy is
updated via supervised learning. Specifically, the action at at step t during rollouts is computed by
the deterministic policy µDT(st−T :t, at−T :t−1, RTGt−T :t, T = Teval, RTG = RTGeval),1 or sampled
from the stochastic policy πDT(at|st−T :t, at−T :t−1, RTGt−T :t, T = Teval, RTG = RTGeval). Here, T
is the context length (which is Teval in evaluation), and RTGeval ∈R is the target return-to-go. The
data buffer, initialized with offline data, is gradually replaced by online data during finetuning.

When updating the policy, the following loss (we use the deterministic policy as an example, and
thus omit the entropy regularizer) is minimized:

Ttrain
X

t=1

µDT (s0:t, a0:t−1, RTG0:t, RTG = RTGreal, T = t) −at
2
2 .
(1)

Note, Ttrain is the training context length and RTGreal is the real return-to-go. For better readability,
we denote {sx+1, sx+2, . . . , sy}, x, y ∈N as sx:y (i.e., left exclusive and right inclusive), and
similarly {ax+1, ax+2, . . . , ay} as ax:y and {RTGx+1, . . . , RTGy} as RTGx:y. Specially, index
x = y represents an empty sequence. For example, when t = 1, a0:0 is an empty action sequence as
the decision transformer is not conditioned on any past action.

One important observation: the decision transformer is inherently off-policy (the exact policy dis-
tribution varies with the sampled starting point, context length and return-to-go), which effectively
guides our choice of RL gradients to off-policy algorithms (see Appendix C for more details).

TD3. Twin Delayed Deep Deterministic Policy Gradient (TD3) [21] is a state-of-the-art online
off-policy RL algorithm that learns a deterministic policy a = µRL(s). It is an improved version of
an actor-critic (DDPG [32]) with three adjustments to improve its stability: 1) Clipped double Q-
learning, which maintains two critics (estimators for expected return) Qϕ1, Qϕ2 : |S| × |A| →R and
uses the smaller of the two values (i.e., min (Qϕ1, Qϕ2)) to form the target for TD-error minimization.
Such design prevents overestimation of the Q-value; 2) Policy smoothing, which adds noise when
calculating the Q-value for the next action to effectively prevent overfitting; and 3) Delayed update,
which updates µRL less frequently than Qϕ1, Qϕ2 to benefit from a better Q-value landscape when
updating the actor. TD3 also maintains a set of target networks storing old parameters of the actor
and critics that are soft-updated with slow exponential moving average updates from the current,
active network. In this paper, we adapt this algorithm to fit the decision transformer architecture so
that it can be used as an auxiliary objective in an online finetuning process.

3
Method

This section is organized as follows: we will first provide intuition why RL gradients aid online
finetuning of decision transformers (Sec. 3.1), and present our method of adding TD3 gradients
(Sec. 3.2). To further justify our intuition, we provide a theoretical analysis on how ODT fails to
improve during online finetuning when pre-trained with low-reward data (Sec. 3.3).

3.1
Why RL Gradients?

In order to understand why RL gradients aid online finetuning of decision transformers, let us consider
an MDP which only has a single state s0, one step, a one dimensional action a ∈[−1, 1] (i.e., a
bandit with continuous action space) and a simple reward function r(a) = (a + 1)2 if a ≤0 and
r(a) = 1 −2a otherwise, as illustrated in Fig. 1. In this case, a trajectory can be represented
effectively by a scalar, which is the action. If the offline dataset for pretraining is of low quality, i.e.,
all actions in the dataset are either close to −1 or 1, then the decision transformer will obviously not
generate trajectories with high RTG after offline training. As a consequence, during online finetuning,
the new rollout trajectory is very likely to be uninformative about how to reach RTGeval, since it is
too far from RTGeval. Worse still, it cannot improve locally either, which requires ∂RTG

∂a . However,
the decision transformer yields exactly the inverse, i.e.,
∂a
∂RTG. Since the transformer is not invertible
(and even if the transformer is invertible, often the ground truth RTG(a) itself is not), we cannot

1ODT uses different RTGs for evaluation and online rollouts, but we refer to both as RTGeval as they are both
expert-level expected returns.

3


---Page Break---
Unreachable target RTG

RTG

Target 

Action 𝑎

𝝏𝑹𝑻𝑮

𝝏𝒂!

Local improvement by Q-function

ODT
ODT + TD3

Returns-To-Go (RTG)

Action 𝑎
Target 

𝝏𝒂
𝝏𝑹𝑻𝑮?

Offline data

Figure 1: An overview of our work, illustrating why ODT fails to improve with low-return offline
data and RL gradients such as TD3 could help. The decision transformer yields gradient
∂a
∂RTG, but
local policy improvement requires the opposite, i.e., ∂RTG

∂a . Therefore, the agent cannot recover if the
current policy conditioning on high target RTG does not actually lead to high real RTG, which is very
likely when the target RTG is too far from the pretrained policy and out-of-distribution. By adding a
small coefficient for RL gradients, the agents can improve locally, which leads to better performance.

50
100
150
200
250
300
Gradient Steps

0.0

0.2

0.4

0.6

0.8

1.0

Average Reward

ODT
DDPG
ODT+DDPG

(a) Reward Curves

0
2
4
6
8
10
12
14
Epoch

1.0

0.8

0.6

0.4

0.2

0.0

0.2

Action

ODT
DDPG
ODT+DDPG

(b) Rollout Distribution

1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

Action

1.0

0.5

0.0

0.5

1.0

Estimated RTG

Ground Truth
DDPG
ODT+DDPG

(c) Critic

1.00
0.75
0.50
0.25 0.00
0.25
0.50
0.75
1.00
Action

0.0

0.2

0.4

0.6

0.8

1.0

Estimated RTG

ODT
ODT+DDPG

(d) DT Policy

Figure 2: An illustration of a simple MDP, showing how RL can infer the direction for improvement,
while online DT fails. Panels (a) and (b) show, DDPG and ODT+DDPG manage to maximize
reward and find the correct optimal action quickly, while ODT fails to do so. Panel (c) shows how
a DDPG/ODT+DDPG critic (from light blue/orange to dark blue/red) manages to fit ground truth
reward (green curve). Panel (d) shows that the ODT policy (changing from light gray to dark) fails to
discover the hidden reward peak near 0 between two low-reward areas (near −1 and 1 respectively)
contained in the offline data. Meanwhile, ODT+DDPG succeeds in finding the reward peak.

easily estimate the former from the latter. Thus, the hope for policy improvement relies heavily on
the generalization of RTG, i.e., policy yielded by high RTGeval indeed leads to better policy without
any data as evidence, which is not the case with our constructed MDP and dataset.

In contrast, applying traditional RL for continuous action spaces to this setting, we either learn a
value function Q(s0, a) : R →R, which effectively gives us a direction of action improvement
∂Q(s0,a)

∂a
(e.g., SAC [25], DDPG [32], TD3 [21]), or an advantage A(s0, a) that highlights whether
focusing on action a improves or worsens the policy (e.g., AWR [47], AWAC [40], IQL [28]). Either
way provides a direction which suggests how to change the action locally in order to improve the
(estimated) return. In our experiment illustrated in Fig. 2 (see Appendix F for details), we found that
RL algorithms like DDPG [32] can easily solve the aforementioned MDP while ODT fails.

Thus, adding RL gradients aids the decision transformer to improve from given low RTG trajectories.
While one may argue that the self-supervised training paradigm of ODT [74] can do the same by
“prompting” the decision transformer to generate a high RTG trajectory, such paradigm is still unable
to effectively improve the policy of the decision transformer pretrained on data with low RTGs. We
provide a theoretical analysis for this in Sec. 3.3. In addition, we also explore the possibility of
fixing this problem using other existing algorithms, such as JSRL [59] and slowly growing RTG (i.e.,
curriculum learning). However, we found that those algorithms cannot address this problem well.
See Appendix G.8 for ablations.

4


---Page Break---
3.2
Adding TD3 Gradients to ODT

In this work, we mainly consider TD3 [21] as the RL gradient for online finetuning. There are
two reasons for selecting TD3. First, TD3 is a more robust off-policy RL algorithm compared to
other off-policy RL algorithms [54]. Second, the success of TD3+BC [20] indicates that TD3 is a
good candidate when combined with supervised learning. A more detailed discussion and empirical
comparison to other RL algorithms can be found in Appendix C.

Generally, we simply add a weighted standard TD3 actor loss to the decision transformer
objective. To do this, we follow classic TD3 and additionally train two critic networks Qϕ1, Qϕ2 :
S × A →R parameterized by ϕ1, ϕ2 respectively. In the offline pretraining stage, we use the
following objective for the actor:

min
µDT Eτ∼D

"
1
Ttrain

Ttrain
X

t=1



−αQϕ1(st, µDT(s0:t, a0:t−1, RTG0:t, RTG = RTGreal, T = t))+

∥µDT(s0:t, a0:t−1, RTG0:t, RTG = RTGreal, T = t) −at∥2
2

#

.

(2)

Here, α ∈{0, 0.1} is a hyperparameter, and the loss sums over the trajectory segment. For critics
Qϕ1, Qϕ2, we use the standard TD3 critic loss

min
ϕ1,ϕ2 Eτ∼D

Ttrain
X

t=1

h
(Qϕ1(st, at) −Qmin,t)2 + (Qϕ2(st, at) −Qmin,t)2i
,
with

Qmin,t = rt + γ(1 −dt) min
i∈{1,2} Qϕi,tar
 
st, clip
 
µRL
tar (zt) + clip(ϵ, −c, c), alow, ahigh

,
(3)

where τ = {s0:Ttrain+1, a0:Ttrain, RTG0:Ttrain+1, d0:Ttrain, r0:Ttrain, RTG = RTGreal} is the trajectory seg-
ment sampled from buffer D that stores the offline dataset. Further, dt indicates whether the trajectory
ends on the t-th step (true is 1, false is 0), Qmin is the target to fit, Qϕi,tar is produced by the target net-
work (stored old parameter), zt is the context for “next state” at step t. µRL
tar is the target network for the
actor (i.e., decision transformer). For an n-dimensional action, clip(a, x, y), a ∈Rn, y ∈Rn, z ∈Rn
means clip ai to [yi, zi] for i ∈{1, 2, . . . , n}. alow ∈Rn and ahigh ∈Rn are the lower and upper
bound for every dimension respectively.

To demonstrate the impact on aiding the exploration of a decision transformer, in this work we choose
the simplest form of a critic, which is reflective, i.e., only depends on the current state. This essentially
makes the Q-value an average of different context lengths sampled from a near-uniform distribution
(see Appendix D for the detailed reason and distribution for this). The choice is based on the fact
that training a transformer-based value function estimator is quite hard [45] due to increased input
complexity (i.e., noise from the environment) which leads to reduced stability and slower convergence.
In fact, to avoid this difficulty, many recent works on Large Language Models (LLMs) [13] and
vision models [48] which finetune with RL adopt a policy-based algorithm instead of an actor-critic,
despite a generally lower variance of the latter. In our experiments, we also found such a critic to be
much more stable than a recurrent critic network (see Appendix G for ablations).

During online finetuning, we again use Eq. (2) and Eq. (3), but always use α = 0.1 for Eq. (2).

While the training paradigm resembles that of TD3+BC, our proposed method improves upon
TD3+BC in the following two ways: 1) Architecture. While TD3+BC uses MLP networks for
single steps, we leverage a decision transformer, which is more expressive and can take more context
into account when making decisions. 2) Selected instead of indiscriminated behavior cloning.
Behavior cloning mimics all data collected without regard to their reward, while the supervised
learning process of a decision transformer prioritizes trajectories with higher return by conditioning
action generation on higher RTG. See Appendix G.9 for an ablation.

3.3
Why Does ODT Fail to Improve the Policy?

As mentioned in Sec. 3.1, it is the goal of ODT to “prompt” a policy with a high RTG, i.e., to improve
a policy by conditioning on a high RTG during online rollout. However, beyond the intuition provided

5


---Page Break---
in Sec. 3.1, in this section, we will analyze more formally why such a paradigm is unable to improve
the policy given offline data filled with low-RTG trajectories.

Our analysis is based on the performance bound proved by Brandfonbrener et al. [7]. Given a dataset
drawn from an underlying policy β and given its RTG distribution Pβ (either continuous or discrete),
under assumptions (see Appendix E), we have the following tight performance bound for a decision
transformer with policy πDT(a|s, RTGeval) conditioned on RTGeval:

RTGeval −Eτ=(s1,a1,...,sH,aH)∼πDT(a|s,RTGeval)[RTGreal] ≤ϵ
 1

αf
+ 2

H2.
(4)

Here, αf = infs1 Pβ(RTGreal = RTGeval|s1) for every initial state s1, ϵ > 0 is a constant, H is
the horizon of the MDP.2 Based on this tight performance bound, we will show that with high
probability,
1
αf grows superlinearly with respect to RTGeval. If true, then the RTGreal term (i.e.,
the actual return from online rollouts) must decrease to fit into the tight bound, as RTGeval grows.

To show this, we take a two-step approach: First, we prove that the probability mass of the RTG
distribution is concentrated around low RTGs, i.e., event probability Prβ (RTG −Eβ(RTG|s) ≥c|s)
for c > 0 decreases superlinearly with respect to c. For this, we apply the Chebyshev inequality,
which yields a bound of O
  1

c2

. However, without knowledge on Pβ(RTG|s), the variance can be
made arbitrarily large by high RTG outliers, hence making the bound meaningless.

Fortunately, we have knowledge about the RTG distribution Pβ(RTG|s) from the collected data. If we
refer to the maximum RTG in the dataset via RTGβmax and if we assume all rewards are non-negative,
then all trajectory samples have an RTG in [0, RTGβmax]. Thus, with adequate prior distribution, we
can state that with high probability 1 −δ, the probability mass is concentrated in the low RTG area.
Based on this, we can prove the following lemma:
Lemma 1. (Informal) Assume rewards r(s, a) are bounded in [0, Rmax],3 and RTGeval ≥RTGβmax.
Then with probability at least 1 −δ, we have the probability of event Prβ bounded as follows:

Prβ
 
RTGeval −V β(s) ≥c|s

≤O

R2
maxT 2

c2

,
(5)

where δ depends on the number of trajectories in the dataset and prior distribution (see Appendix E
for a concrete example and a more accurate bound). V β(s) is the value function of the underlying
policy β(a|s) that generates the dataset, for which we have V β(s) = Eβ(RTG|s).

The second step uses the bound of probability mass Prβ(RTG ≥c|s) to derive the bound for αf. For
the discrete case where the possibly obtained RTGs are finite or countably infinite (note, state and
action space can still be continuous), this is simple, as we have

Pβ
 
RTG = V β(s) + c|s

= Prβ
 
RTG = V β(s) + c|s

≤Prβ
 
RTG ≥V β(s) + c|s

.
(6)

Thus αf = infs1 Pβ(RTG|s1) can be conveniently bounded by Lemma 1. For the continuous case,
the proof is more involved as probability density Pβ(RTG|s) can be very high on an extremely
short interval of RTG, making the total probability mass arbitrarily small. However, assuming that
Pβ(RTG|s) is Lipschitz when RTG ≥RTGβmax (i.e., RTG area not covered by dataset), combined
with the discrete distribution case, we can still get the following (see Appendix E for proof):
Corollary 1. (Informal) If the RTG distribution is discrete (i.e., number of possible different RTGs are
at most countably infinite), then with probability at least- 1 −δ,
1
αf grows on the order of Ω(RTG2
eval)
with respect to RTGeval. For continuous RTG distributions satisfying a Lipschitz continuous RTG
density pβ,
1
αf grows on the order of Ω(RTG1.5
eval).

Here, Ω(·) refers to the big-Omega notation (asymptotic lower bound).

4
Experiments

In this section, we aim to address the following questions: a) Does our proposed solution for decision
transformers indeed improve its ability to cope with low-reward pretraining data. b) Is improving

2Eq. (4) is very informal. See Appendix E for a more rigorous description.
3Note we use “max” instead of “βmax” as this is a property of the environment and not the dataset.

6


---Page Break---
what to predict, while still using supervised learning, the correct way to improve the finetuning ability
of decision transformers? c) Does the transformer architecture, combined with RL gradients, work
better than TD3+BC? d) Is it better to combine the use of RL and supervised learning, or better to
simply abandon the supervised loss in online finetuning? e) How does online decision transformer
with TD3 gradient perform compared to other offline RL algorithms? f) How much does TD3 improve
over DDPG which was used in Fig. 2?

Baselines. In this section, we mainly compare to six baselines: the widely recognized state-of-the-art
DT for online finetuning, Online Decision Transformer (ODT) [74]; PDT, a baseline improving over
ODT by predicting future trajectory information instead of return-to-go; TD3+BC [20], a MLP offline
RL baseline; TD3, an ablated version of our proposed solution where we use TD3 gradients only for
decision transformer finetuning (but only use supervised learning of the actor for offline pretraining);
IQL [28], one of the most popular offline RL algorithms that can be used for online finetuning;
DDPG [32]+ODT, which is the same as our approach but with DDPG instead of TD3 gradients (for
ablations using SAC [25], IQL [28], PPO [52], AWAC [40] and AWR [47], see Appendix C). Each
of the baselines corresponds to one of the questions a), b), c), d), e) and f) above.

Metrics. We use the normalized average reward (same as D4RL’s standard [19]) as the metric, where
higher reward indicates better performance. If the final performance is similar, the algorithm with
fewer online examples collected to reach that level of performance is better. We report the reward
curve, which shows the change of the normalized reward’s mean and standard deviation with 5
different seeds, with respect to the number of online examples collected. The maximum number
of steps collected is capped at 500K (for mujoco) or 1M (for other environments). We also report
evaluation results using the rliable [3] library in Fig. 7 of Appendix B.

Experimental Setup. We use the same architecture and hyperparameters such as learning rate
(see Appendix F.2 for details) as ODT [74]. The architecture is a transformer with 4 layers and
4 heads in each layer. This translates to around 13M parameters in total. For the critic, we use
Multi-Layer Perceptrons (MLPs) with width 256 and two hidden layers and ReLU [1] activation
function. Specially, for the random dataset, we collect trajectories until the total number of steps
exceeds 1000 in every epoch, which differs from ODT, where only 1 trajectory per epoch is collected.
This is because many random environments, such as hopper, have very short episodes when the agent
does not perform well, which could lead to overfitting if only a single trajectory is collected per
epoch. For fairness, we use this modified rollout for ODT in our experiments as well. Not doing
so does not affect ODT results since it does generally not work well on random datasets, but will
significantly increase the time to reach a certain number of online transitions. After rollout, we train
the actor for 300 gradient steps and the critic for 600 steps following TD3’s delayed update trick.

4.1
Adroit Environments

Environment and Dataset Setup. We test on four difficult robotic manipulation tasks [49], which
are the Pen, Hammer, Door and Relocate environment. For each environment, we test three different
datasets: expert, cloned and human, which are generated by a finetuned RL policy, an imitation
learning policy and human demonstration respectively. See Appendix F.1 for details.

Results. Fig. 3 shows the performance of each method on Adroit before and after online finetuning.
TD3+BC fails on almost all tasks and often diverges with extremely large Q-value during online
finetuning. ODT and PDT perform better but still fall short of the proposed method, TD3+ODT. Note,
IQL, TD3 and TD3+ODT all perform decently well (with similar average reward as shown in Tab. 2
in Appendix B). However, we found that TD3 often fails during online finetuning, probably because
the environments are complicated and TD3 struggles to recover from a poor policy generated during
online exploration (i.e., it has a catastrophic forgetting issue). To see whether there is a simple fix,
in Appendix G.7, we ablate whether an action regularizer pushing towards a pretrain policy similar
to TD3+BC helps, but find it to hinder performance increase in other environments. IQL is overall
much more stable than TD3, but improves much less during online finetuning than TD3+ODT. ODT
can achieve good performance when pretrained on expert data, but struggles with datasets of lower
quality, which validates our motivation. DDPG+ODT starts out well in the online finetuning stage
but fails quickly, probably because DDPG is less stable compared to TD3.

7


---Page Break---
0

50

100

150

Pen-expert-v1

0

50

100

Hammer-expert-v1

0

25

50

75

100

Relocate-expert-v1

0

25

50

75

100

125
Door-expert-v1

0

50

100

150

Pen-cloned-v1

0

25

50

75

100

125

Hammer-cloned-v1

0.5

0.0

0.5

1.0

1.5

2.0

Relocate-cloned-v1

0

20

40

60

Door-cloned-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

50

100

Pen-human-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

50

100

Hammer-human-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

1

2

3

Relocate-human-v1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

20

40

60

80

Normalized Reward

Door-human-v1

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)

Figure 3: Results on Adroit [49] environments. The proposed method, TD3+ODT, improves upon
baselines. Note that TD3, IQL, and TD3+ODT all perform decently at the beginning of online
finetuning, but TD3 fails while TD3+ODT improves much more than IQL during online finetuning.

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

Antmaze-umaze-v2

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

Antmaze-umaze-diverse-v2

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

Antmaze-medium-play-v2

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

25

50

75

100

Normalized Reward

Antmaze-medium-diverse-v2

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)

Figure 4: Reward curves for each method in Antmaze environments. IQL works best on the large
maze, while our proposed method works the best on the medium maze and umaze. DDPG+ODT
works worse than our method and IQL but much better than the rest of the baselines, which again
validates our motivation that adding RL gradients to ODT is helpful.

4.2
Antmaze Environments

Environment and Dataset Setup. We further test on a harder version of the Maze2D environment in
D4RL [19] where the pointmass is substituted by a robotic ant. We study six different variants, which
are umaze, umaze-diverse, medium-play, medium-diverse, large-play and large-diverse.

Results. Fig. 4 lists the results of each method on umaze and medium maze before and after online
finetuning (see Appendix C for reward curves and Appendix B for results summary on large antmaze).
TD3+ODT works the best on umaze and medium maze, and significantly outperforms TD3. This
shows that RL gradients alone are not enough for offline-to-online RL of the decision transformer.
Though TD3+ODT does not work on large maze, we found that IQL+ODT works decently well.
However, we choose TD3+ODT in this work because IQL+ODT does not work well on the random
datasets. This is probably because IQL aims to address the Out-Of-Distribution (OOD) estimation
problem [28], which makes it better at utilizing offline data but worse at online exploration. See
Appendix C for a detailed discussion and results. DDPG+ODT works worse than TD3+ODT but
much better than baselines except IQL.

4.3
MuJoCo Environments

Environment and Dataset Setup. We further test on four widely recognized standard environ-
ments [58], which are the Hopper, Halfcheetah, Walker2d and Ant environment. For each environ-
ment, we study three different datasets: medium, medium-replay, and random. The first and second
one contain trajectories of decent quality, while the last one is generated with a random agent.

8


---Page Break---
TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)
Ho-M-v2
60.24(+4.4)
44.72(-21.3)
97.84(+48.69)
74.43(+72.21)
88.98(+29.25)
41.7(-13.18)
89.07(+25.97)
Ho-MR-v2
99.07(+33.33)
62.76(-7.63)
83.29(+63.17)
84.53(+82.23)
93.72(+55.66)
32.36(+9.9)
95.65(+65.89)
Ho-R-v2
8.36(-0.35)
20.42(+12.36)
29.08(+26.92)
35.9(+34.67)
75.68(+73.69)
25.12(+23.14)
76.13(+74.15)
Ha-M-v2
51.29(+2.73)
37.12(-10.35)
42.27(+19.23)
39.35(+39.55)
70.9(+29.59)
55.69(+14.71)
76.91(+35.3)
Ha-MR-v2
56.5(+13.07)
49.97(+6.84)
41.45(+26.77)
31.47(+31.8)
69.87(+40.59)
53.71(+24.91)
73.27(+43.98)
Ha-R-v2
44.78(+31.12)
47.85(+40.3)
2.15(-0.09)
0.74(+0.9)
68.55(+66.3)
34.56(+32.31)
59.35(+57.1)
Wa-M-v2
85.34(+3.49)
65.55(-15.12)
75.57(+18.47)
63.37(+63.3)
90.49(+24.74)
2.01(-69.54)
97.86(+27.08)
Wa-MR-v2
83.28(+0.0)
95.99(+28.78)
77.2(+12.46)
54.49(+54.18)
100.88(+32.54)
1.04(-60.59)
100.6(+42.54)
Wa-R-v2
6.99(+5.86)
10.67(+4.96)
14.12(+9.82)
15.47(+15.32)
69.91(+66.31)
2.91(-2.47)
57.86(+53.27)
An-M-v2
129.11(+7.11)
110.36(+14.26)
88.1(-0.51)
52.08(+48.47)
125.67(+37.55)
10.81(-75.52)
132.0(+41.42)
An-MR-v2
129.33(+41.03)
113.16(+24.24)
85.64(+4.49)
36.92(+32.41)
133.58(+51.17)
4.05(-87.7)
130.23(+52.08)
An-R-v2
67.89(+33.47)
12.28(+0.97)
24.96(-6.44)
14.88(+10.38)
63.47(+32.02)
4.93(-26.55)
71.69(+40.31)

Average
68.52(+14.6)
55.9(+7.8)
55.14(+18.58)
41.97(+40.44)
87.64(+44.95)
22.87(-19.22)
88.38(+46.59)
Table 1: Average reward for each method in MuJoCo environments before and after online finetuning.
The best performance for each environment is highlighted in bold font, and any result > 90% of
the best performance is underlined. To save space, the name of the environments and datasets are
abbreviated as follows: for the environments Ho=Hopper, Ha=HalfCheetah, Wa=Walker2d, An=Ant;
for the datasets M=Medium, MR=Medium-Replay, R=Random. The format is “final(+increase after
finetuning)”. The proposed solution performs well.

Results. Fig. 6 shows the results of each method on MuJoCo before and after online finetuning.
We observe that autoregressive-based algorithms, such as ODT and PDT, fail to improve the policy
on MuJoCo environments, especially from low-reward pretraining with random datasets. With RL
gradients, TD3+BC and IQL can improve the policy during online finetuning, but less than a decision
transformer (TD3 and TD3+ODT). In particular, we found IQL to struggle on most random datasets,
which are well-solved by decision transformers with TD3 gradients. TD3+ODT still outperforms
TD3 with an average final reward of 88.51 vs. 84.23. See Fig. 6 in Appendix B for reward curves.

Ablations on α. Fig. 5 (a) shows the result of using different α (i.e., RL coefficients) on different
environments. We observe an increase of α to improve the online finetuning process. However, if α
is too large, the algorithm may get unstable.

Ablations on evaluation context length Teval. Fig. 5 (b) shows the result of using different Teval
on halfcheetah-medium-replay-v2 and hammer-cloned-v1. The result shows that Teval needs to be
balanced between more information for decision-making and potential training instability due to a
longer context length. As shown in the halfcheetah-medium-replay-v2 result, Teval too long or too
short can both lead to performance drops. More ablations are available in Appendix G.

0
1
2
3
4
5
1e5

20

40

60

Halfcheetah-medium-replay-v2

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

2.10

2.15

2.20

Normalized Reward

Hammer-human-v1

10
1
0.1
0.01
0.001

(a) Ablations on α

0
1
2
3
4
5
1e5

20

40

60

Halfcheetah-medium-replay-v2

T_eval=20
T_eval=10
T_eval=5
T_eval=1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

50

100

Normalized Reward

Hammer-cloned-v1

T_eval=1
T_eval=2
T_eval=3
T_eval=5
(b) Ablations on Teval

Figure 5: Panel (a) shows ablations on RL coefficient α. While higher α aids exploration as shown in
the halfcheetah-medium-replay-v2 case, it may sometimes introduce instability, which is shown in
the hammer-human-v1 case. Panel (b) shows ablations on Teval. Teval balances training stability and
more information for decision-making.

5
Related Work

Online Finetuning of Decision Transformers. While there are many works on generalizing decision
transformers (e.g., predicting waypoints [5], goal, or encoded future information instead of return-to-
go [22, 5, 57, 36]), improving the architecture [37, 16, 53, 65] or addressing the overly-optimistic [46]
or trajectory stitching issue [63]), there is surprisingly little work beyond online decision transformers
that deals with online finetuning of decision transformers. There is some loosely related literature:
MADT [31] proposes to finetune pretrained decision transformers with PPO. PDT [64] also studies
online finetuning with the same training paradigm as ODT [74]. QDT [66] uses an offline RL

9


---Page Break---
algorithm to re-label returns-to-go for offline datasets. AFDT [76] and STG [75] use decision
transformers offline to generate an auxiliary reward and aid the training of online RL algorithms. A
few works study in-context learning [33, 34] and meta-learning [60, 30] of decision transformers,
where improvements with evaluations on new tasks are made possible. However, none of the papers
above focuses on addressing the general online finetuning issue of the decision transformer.

Transformers as Backbone for RL. Having witnessed the impressive success of transformers in
Computer Vision (CV) [17] and Natural Language Processing (NLP) [11], numerous works also
studied the impact of transformers in RL either as a model for the agent [45, 38] or as a world
model [39, 50]. However, a large portion of state-of-the-art work in RL is still based on simple Multi-
Layer Perceptrons (MLPs) [35, 28]. This is largely because transformers are significantly harder to
train and require extra effort [45], making their ability to better memorize long trajectories [42] harder
to realize compared to MLPs. Further, there are works on using transformers as feature extractors for
a trajectory [37, 45] and works that leverage the common sense of transformer-based Large Language
Model’s for RL priors [10, 9, 70]. In contrast, our work focuses on improving the new “RL via
Supervised learning” (RvS) [7, 18] paradigm, aiming to merge this paradigm with the benefits of
classic RL training.

Offline-to-Online RL. Offline-to-online RL bridges the gap between offline RL, which heavily de-
pends on the quality of existing data while struggling with out-of-distribution policies, and online RL,
which requires many interactions and is of low data efficiency. Mainstream offline-to-online RL meth-
ods include teacher-student [51, 6, 59, 72] and out-of-distribution handling (regularization [21, 29, 62],
avoidance [28, 23], ensembles [2, 15, 24]). There are also works on pessimistic Q-value initializa-
tion [69], confidence bounds [26], and a mixture of offline and online training [56, 73]. However, all
the aforementioned works are based on Q-learning and don’t consider decision transformers.

6
Conclusion

In this paper, we point out an under-explored problem in the Decision Transformer (DT) community,
i.e., online finetuning. To address online finetuning with a decision transformer, we examine
the current state-of-the-art, online decision transformer, and point out an issue with low-reward,
sub-optimal pretraining. To address the issue, we propose to mix TD3 gradients with decision
transformer training. This combination permits to achieve better results in multiple testbeds. Our
work is a complement to the current DT literature, and calls out a new aspect of improving decision
transformers.

Limitations and Future Works. While our work theoretically analyzes an ODT issue, the conclusion
relies on several assumptions which we expect to remove in future work. Empirically, in this work we
propose a simple solution orthogonal to existing efforts like architecture improvements and predicting
future information rather than return-to-go. To explore other ideas that could further improve online
finetuning of decision transformers, next steps include the study of other environments and other
ways to incorporate RL gradients into decision transformers. Other possible avenues for future
research include testing our solution on image-based environments, and decreasing the additional
computational cost compared to ODT (an analysis for the current time cost is provided in Appendix H).

Acknowledgments

This work was supported in part by NSF under Grants 2008387, 2045586, 2106825, MRI 1725729,
NIFA Award 2020-67021-32799, the IBM-Illinois Discovery Accelerator Institute, the Toyota Re-
search Institute, and the Jump ARCHES endowment through the Health Care Engineering Systems
Center at Illinois and the OSF Foundation.

10


---Page Break---
References

[1] Agarap, A. F. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375,
2018.

[2] Agarwal, R., Schuurmans, D., and Norouzi, M. An optimistic perspective on offline reinforce-
ment learning. In ICML, 2020.

[3] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. Deep reinforce-
ment learning at the edge of the statistical precipice. In NeurIPS, 2021.

[4] Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450,
2016.

[5] Badrinath, A., Flet-Berliac, Y., Nie, A., and Brunskill, E. Waypoint transformer: Reinforcement
learning via supervised learning with intermediate targets. In NeurIPS, 2023.

[6] Bastani, O., Pu, Y., and Solar-Lezama, A. Verifiable reinforcement learning via policy extraction.
In NeurIPS, 2018.

[7] Brandfonbrener, D., Bietti, A., Buckman, J., Laroche, R., and Bruna, J. When does return-
conditioned supervised learning work for offline reinforcement learning? In NeurIPS, 2022.

[8] Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba,
W. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[9] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess,
D., Dubey, A., Finn, C., et al. Rt-2: Vision-language-action models transfer web knowledge to
robotic control. arXiv preprint arXiv:2307.15818, 2023.

[10] Brohan, A., Chebotar, Y., Finn, C., Hausman, K., Herzog, A., Ho, D., Ibarz, J., Irpan, A., Jang,
E., Julian, R., et al. Do as i can, not as i say: Grounding language in robotic affordances. In
CoRL, 2023.

[11] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A.,
Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In NeurIPS,
2020.

[12] Chebotar, Y., Vuong, Q., Hausman, K., Xia, F., Lu, Y., Irpan, A., Kumar, A., Yu, T., Herzog, A.,
Pertsch, K., et al. Q-transformer: Scalable offline reinforcement learning via autoregressive
q-functions. In CoRL, 2023.

[13] Chen, C., Wang, X., Jin, Y., Dong, V. Y., Dong, L., Cao, J., Liu, Y., and Yan, R. Semi-offline
reinforcement learning for optimized text generation. In ICML, 2023.

[14] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A.,
and Mordatch, I. Decision transformer: Reinforcement learning via sequence modeling. In
NeurIPS, 2021.

[15] Chen, X., Wang, C., Zhou, Z., and Ross, K. Randomized ensembled double q-learning: Learning
fast without a model. In ICLR, 2021.

[16] David, S. B., Zimerman, I., Nachmani, E., and Wolf, L. Decision s4: Efficient sequence-based
rl via state spaces layers. In ICLR, 2022.

[17] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani,
M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers
for image recognition at scale. In ICLR, 2021.

[18] Emmons, S., Eysenbach, B., Kostrikov, I., and Levine, S. Rvs: What is essential for offline rl
via supervised learning? In ICLR, 2022.

[19] Fu, J., Kumar, A., Nachum, O., Tucker, G., and Levine, S. D4rl: Datasets for deep data-driven
reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

11


---Page Break---
[20] Fujimoto, S. and Gu, S. S. A minimalist approach to offline reinforcement learning. In NeurIPS,
2021.

[21] Fujimoto, S., Hoof, H., and Meger, D. Addressing function approximation error in actor-critic
methods. In ICML, 2018.

[22] Furuta, H., Matsuo, Y., and Gu, S. S. Generalized decision transformer for offline hindsight
information matching. In ICLR, 2022.

[23] Garg, D., Hejna, J., Geist, M., and Ermon, S. Extreme q-learning: Maxent rl without entropy.
In ICLR, 2023.

[24] Ghasemipour, S. K. S., Schuurmans, D., and Gu, S. S. Emaq: Expected-max q-learning operator
for simple yet effective offline and online rl. In ICML, 2021.

[25] Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum
entropy deep reinforcement learning with a stochastic actor. In ICML, 2018.

[26] Hong, J., Kumar, A., and Levine, S.
Confidence-conditioned value functions for offline
reinforcement learning. In ICLR, 2023.

[27] Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In ICLR, 2015.

[28] Kostrikov, I., Nair, A., and Levine, S. Offline reinforcement learning with implicit q-learning.
arXiv preprint arXiv:2110.06169, 2021.

[29] Kumar, A., Zhou, A., Tucker, G., and Levine, S. Conservative q-learning for offline reinforce-
ment learning. In NeurIPS, 2020.

[30] Lee, J., Xie, A., Pacchiano, A., Chandak, Y., Finn, C., Nachum, O., and Brunskill, E. In-context
decision-making from supervised pretraining. In ICML Workshop on New Frontiers in Learning,
Control, and Dynamical Systems, 2023.

[31] Lee, K.-H., Nachum, O., Yang, M. S., Lee, L., Freeman, D., Guadarrama, S., Fischer, I., Xu,
W., Jang, E., Michalewski, H., et al. Multi-game decision transformers. In NeurIPS, 2022.

[32] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N. M. O., Erez, T., Tassa, Y., Silver, D., and
Wierstra, D. Continuous control with deep reinforcement learning. In ICLR, 2016.

[33] Lin, L., Bai, Y., and Mei, S. Transformers as decision makers: Provable in-context reinforcement
learning via supervised pretraining. In ICLR, 2024.

[34] Liu, H. and Abbeel, P. Emergent agentic transformer from chain of hindsight experience. In
ICML, 2023.

[35] Lyu, J., Ma, X., Li, X., and Lu, Z. Mildly conservative q-learning for offline reinforcement
learning. In NeurIPS, 2022.

[36] Ma, Y., Xiao, C., Liang, H., and Hao, J. Rethinking decision transformer via hierarchical
reinforcement learning. arXiv preprint arXiv:2311.00267, 2023.

[37] Mao, H., Zhao, R., Chen, H., Hao, J., Chen, Y., Li, D., Zhang, J., and Xiao, Z. Transformer in
transformer as backbone for deep reinforcement learning. In AAMAS, 2024.

[38] Meng, L., Wen, M., Yang, Y., Le, C., Li, X., Zhang, W., Wen, Y., Zhang, H., Wang, J., and Xu,
B. Offline pre-trained multi-agent decision transformer: One big sequence model tackles all
smac tasks. arXiv preprint arXiv:2112.02845, 2021.

[39] Micheli, V., Alonso, E., and Fleuret, F. Transformers are sample efficient world models. In
ICLR, 2023.

[40] Nair, A., Dalal, M., Gupta, A., and Levine, S. Accelerating online reinforcement learning with
offline datasets. arXiv preprint arXiv:2006.09359, 2020.

[41] Nakamoto, M., Zhai, Y., Singh, A., Mark, M. S., Ma, Y., Finn, C., Kumar, A., and Levine, S.
Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. In NeurIPS, 2023.

12


---Page Break---
[42] Ni, T., Ma, M., Eysenbach, B., and Bacon, P.-L. When do transformers shine in rl? decoupling
memory from credit assignment. In NeurIPS, 2023.

[43] Oh, J., Guo, Y., Singh, S., and Lee, H. Self-imitation learning. In ICML, 2018.

[44] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal,
S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A.,
Welinder, P., Christiano, P. F., Leike, J., and Lowe, R. Training language models to follow
instructions with human feedback. In NeurIPS, 2022.

[45] Parisotto, E., Song, F., Rae, J., Pascanu, R., Gulcehre, C., Jayakumar, S., Jaderberg, M.,
Kaufman, R. L., Clark, A., Noury, S., et al. Stabilizing transformers for reinforcement learning.
In ICML, 2020.

[46] Paster, K., McIlraith, S., and Ba, J. You can’t count on luck: Why decision transformers and rvs
fail in stochastic environments. In NeurIPS, 2022.

[47] Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and
scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[48] Pinto, A. S., Kolesnikov, A., Shi, Y., Beyer, L., and Zhai, X. Tuning computer vision models
with task rewards. In ICML, 2023.

[49] Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J., Todorov, E., and Levine, S.
Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demon-
strations. In RSS, 2018.

[50] Robine, J., Höftmann, M., Uelwer, T., and Harmeling, S. Transformer-based world models are
happy with 100k interactions. In ICLR, 2023.

[51] Schmitt, S., Hudson, J. J., Zidek, A., Osindero, S., Doersch, C., Czarnecki, W. M., Leibo, J. Z.,
Kuttler, H., Zisserman, A., Simonyan, K., et al. Kickstarting deep reinforcement learning. arXiv
preprint arXiv:1803.03835, 2018.

[52] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimiza-
tion algorithms. arXiv preprint arXiv:1707.06347, 2017.

[53] Shang, J., Kahatapitiya, K., Li, X., and Ryoo, M. S. Starformer: Transformer with state-action-
reward representations for visual reinforcement learning. In ECCV, 2022.

[54] Sharif, A. and Marijan, D. Evaluating the robustness of deep reinforcement learning for
autonomous and adversarial policies in a multi-agent urban driving environment. In QRS, 2021.

[55] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L.,
Kumaran, D., Graepel, T., Lillicrap, T. P., Simonyan, K., and Hassabis, D. Mastering chess and
shogi by self-play with a general reinforcement learning algorithm. Science, 2018.

[56] Song, Y., Zhou, Y., Sekhari, A., Bagnell, J. A., Krishnamurthy, A., and Sun, W. Hybrid rl:
Using both offline and online data can make rl efficient. In ICLR, 2023.

[57] Sudhakaran, S. and Risi, S. Skill decision transformer. Foundation Models for Decision Making
Workshop at NeurIPS, 2022.

[58] Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In
IROS, 2012.

[59] Uchendu, I., Xiao, T., Lu, Y., Zhu, B., Yan, M., Simon, J., Bennice, M., Fu, C., Ma, C., Jiao, J.,
et al. Jump-start reinforcement learning. In ICML, 2023.

[60] Wang, Z., Wang, H., and Qi, Y. J. T3gdt: Three-tier tokens to guide decision transformer for
offline meta reinforcement learning. In Robot Learning Workshop: Pretraining, Fine-Tuning,
and Generalization with Large Scale Models in NeurIPS, 2023.

13


---Page Break---
[61] Wołczyk, M., Cupiał, B., Ostaszewski, M., Bortkiewicz, M., Zaj ˛ac, M., Pascanu, R., Kuci´nski,
Ł., and Miło´s, P. Fine-tuning reinforcement learning models is secretly a forgetting mitigation
problem. In ICML, 2024.

[62] Wu, J., Wu, H., Qiu, Z., Wang, J., and Long, M. Supported policy optimization for offline
reinforcement learning. In NeurIPS, 2022.

[63] Wu, Y.-H., Wang, X., and Hamaya, M. Elastic decision transformer. In NeurIPS, 2023.

[64] Xie, Z., Lin, Z., Ye, D., Fu, Q., Wei, Y., and Li, S. Future-conditioned unsupervised pretraining
for decision transformer. In ICML, 2023.

[65] Xu, M., Lu, Y., Shen, Y., Zhang, S., Zhao, D., and Gan, C. Hyper-decision transformer for
efficient online policy adaptation. In ICLR, 2023.

[66] Yamagata, T., Khalil, A., and Santos-Rodriguez, R. Q-learning decision transformer: Leveraging
dynamic programming for conditional sequence modelling in offline rl. In ICML, 2023.

[67] Yan, K., Schwing, A., and Wang, Y.-X. Ceip: Combining explicit and implicit priors for
reinforcement learning with demonstrations. In NeurIPS, 2022.

[68] You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer,
K., and Hsieh, C.-J. Large batch optimization for deep learning: Training bert in 76 minutes. In
ICLR, 2020.

[69] Yu, Z. and Zhang, X. Actor-critic alignment for offline-to-online reinforcement learning. In
ICML, 2023.

[70] Yuan, H., Zhang, C., Wang, H., Xie, F., Cai, P., Dong, H., and Lu, Z. Plan4MC: Skill
reinforcement learning and planning for open-world Minecraft tasks. In Foundation Models for
Decision Making Workshop at NeurIPS 2023, 2023.

[71] Yue, Y., Lu, R., Kang, B., Song, S., and Huang, G. Understanding, predicting and better
resolving q-value divergence in offline-rl. NeurIPS, 2024.

[72] Zhang, H., Xu, W., and Yu, H. Policy expansion for bridging offline-to-online reinforcement
learning. In ICLR, 2023.

[73] Zheng, H., Luo, X., Wei, P., Song, X., Li, D., and Jiang, J. Adaptive policy learning for
offline-to-online reinforcement learning. In AAAI, 2023.

[74] Zheng, Q., Zhang, A., and Grover, A. Online decision transformer. In ICML, 2022.

[75] Zhou, B., Li, K., Jiang, J., and Lu, Z. Learning from visual observation via offline pretrained
state-to-go transformer. In NeurIPS, 2023.

[76] Zhu, D., Wang, Y., Schmidhuber, J., and Elhoseiny, M. Guiding online reinforcement learning
with action-free offline pretraining. arXiv preprint arXiv:2301.12876, 2023.

14


---Page Break---
Appendix: Reinforcement Learning Gradients as Vitamin for Online
Finetuning Decision Transformers

The Appendix is organized as follows. In Sec. A, we discuss the potential positive and negative social
impact of the paper. Then, we summarize the performance shown in the main paper in Sec. B. After
this, we will explain our choice of RL gradients in the paper in Sec. C, and why our critic serves as
an average of policies generated by different context lengths in Sec. D. We then provide rigorous
statements for the theroetical analysis appearing in the paper in Sec. E, and list the environment
details and hyperparameters in Sec. F. We then present more experiment and ablation results in
Sec. G. Finally, we list our computational resource usage and licenses of related assets in Sec. H and
Sec. I respectively.

A
Broader Societal Impacts

Our work generally helps automation of decision-making by improving the use of online interaction
data of a pretrained decision transformer agent. While this effort improves the efficiency of decision-
makers and has the potential to boost a variety of real-life applications such as robotics and resource
allocation, it may also cause several negative social impacts, such as potential job losses, human
de-skilling (making humans less capable of making decisions without AI), and misuse of technology
(e.g., military).

B
Performance Summary

In this section, we summarize the average reward achieve by each method on different environments
and datasets, where the result for Adroit is shown in Tab. 2, and the result for Antmaze is shown
in Tab. 3. As the summary table for MuJoCo is already presented in Sec. 4, we show the reward
curves in Fig. 6. For a more rigorous evaluation, we also report other metrics including the median,
InterQuartile Mean (IQM) and optimality gap using the rliable [3] library. See Fig. 7 for details.
Breakdown analysis for each environment can be downloaded by browsing to https://kaiyan289.
github.io/assets/breakdown_rliable.rar.

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)
P-E-v1
47.88(-84.23)
149.65(-3.63)
121.82(+5.48)
25.07(+25.56)
61.56(-69.74)
2.75(-129.63)
120.65(-11.91)
P-C-v1
3.75(-10.8)
78.12(+25.01)
22.88(-24.0)
14.05(+12.15)
58.04(-17.39)
-1.41(-81.87)
133.77(+58.05)
P-H-v1
26.77(+3.19)
96.5(+27.79)
27.55(-13.91)
4.03(+0.38)
38.58(-57.71)
2.09(-92.56)
107.1(+11.87)
H-E-v1
3.11(-0.02)
126.54(+13.28)
123.07(+12.79)
98.95(+98.94)
93.99(-31.41)
-0.24(-127.05)
129.8(+6.34)
H-C-v1
0.33(+0.03)
2.27(+0.58)
0.84(+0.32)
0.66(+0.66)
0.07(-0.69)
0.12(-0.83)
126.39(+124.59)
H-H-v1
0.17(-0.3)
16.12(+14.18)
0.97(-0.13)
32.76(+32.75)
-0.03(-1.11)
-0.06(-1.02)
116.83(+115.82)
D-E-v1
-0.34(-0.01)
97.57(-7.92)
50.26(+50.14)
59.48(+59.41)
76.92(-25.56)
26.48(-78.72)
103.13(-1.94)
D-C-v1
-0.36(-0.01)
23.8(+21.66)
5.45(+5.37)
1.38(+1.54)
0.17(-4.8)
-0.01(-4.45)
58.28(+53.31)
D-H-v1
-0.33(-0.1)
34.64(+29.65)
10.61(+6.69)
0.05(+0.22)
-0.14(-9.22)
12.39(-12.33)
65.24(+55.94)
R-E-v1
-1.37(+0.22)
105.78(+2.81)
101.16(+2.11)
66.57(+66.7)
0.44(-106.67)
0.26(-106.48)
91.38(-16.19)
R-C-v1
-0.3(+0.0)
1.1(+0.97)
0.06(+0.08)
-0.03(+0.04)
-0.19(-0.29)
-0.12(-0.32)
0.36(+0.26)
R-H-v1
-0.08(+0.1)
1.6(+1.5)
0.04(+0.05)
0.04(+0.17)
-0.17(-0.28)
-0.1(-0.27)
1.19(+0.99)
Average
6.6(-7.66)
61.14(+10.49)
38.73(+3.75)
25.25(+24.87)
31.85(-29.63)
3.51(-52.96)
87.84(+33.09)
Table 2: Average reward for each method in Adroit Environments before and after online finetuning.
The best result for each setting is marked in bold font and all results > 90% of the best performance
are underlined. To save space, the name of the environments and datasets are abbreviated as follows:
P=Pen, H=Hammer, D=Door, R=Relocate for environment, and E=Expert, C=cloned, H=Human for
the dataset. It is apparent that while both IQL, TD3 and TD3+ODT perform decently well before
online finetuning, our proposed solution significantly outperforms all baselines on the adroit testbed.
DDPG+ODT starts out well in the online stage, but fails probably due to DDPG’s training instability
compared to TD3.

C
Why Do We Choose TD3 to Provide RL Gradients?

In this section, we provide an ablation analysis on which RL gradient fits the decision transformer
architecture best. Fig. 8 illustrates the result of using a pure RL gradient for online finetuning of a
pretrained decision transformer (for those RL algorithms with stochastic policy, we adopt the same

15


---Page Break---
TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)
U-v2
0.21(+0.07)
95.99(+6.59)
15.92(-38.08)
29.94(+29.94)
0.0(+0.0)
95.62(+43.62)
99.59(+83.59)
UD-v2
0.53(+0.33)
44.83(-18.17)
0.0(-52.0)
0.0(+0.0)
25.55(+25.55)
23.41(-14.59)
80.0(+42.0)
MP-v2
0.0(+0.0)
91.59(+21.59)
0.0(+0.0)
0.0(+0.0)
70.55(+70.55)
25.66(-14.34)
96.79(+42.79)
MD-v2
0.01(+0.01)
88.63(+23.43)
0.0(+0.0)
0.0(+0.0)
20.24(+20.24)
29.14(-16.86)
96.31(+36.31)
LP-v2
0.0(+0.0)
51.4(+9.6)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
LD-v2
0.0(+0.0)
70.6(+32.6)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
0.0(+0.0)
Average
0.13(+0.07)
73.83(+12.61)
2.65(-15.01)
4.99(+4.99)
19.39(+19.39)
28.97(-0.36)
62.11(+34.1)
Avg. (U+M)
0.19(+0.08)
80.26(+8.36)
3.98(-22.52)
7.49(+7.49)
29.08(+29.08)
43.46(-0.54)
93.17(+51.17)
Table 3: Average reward for each method in Antmaze Environments before and after online finetuning.
The best result is marked in bold font and all results > 90% fo the best performance are underlined.
To save space, the name of the environments and datasets are abbreviated as follows: U=Umaze,
UD=Umaze-Diverse, MP=Medium-Play, MD=Medium-Diverse, LP=Large-Play and LD=Large-
Diverse. U+M=Umaze and Medium maze. Our method performs the best on umaze and medium
maze, while IQL performs the best on large maze. Both methods are much better than the rest on
average. TD3+BC diverges on antmaze in our experiments.

0

25

50

75

100

Hopper-medium-v2

0

50

100

Ant-medium-v2

0

25

50

75

100

Walker2d-medium-v2

0

20

40

60

80

Halfcheetah-medium-v2

0

25

50

75

100

Hopper-medium-replay-v2

0

50

100

150

Ant-medium-replay-v2

0

25

50

75

100

Walker2d-medium-replay-v2

0

20

40

60

Halfcheetah-medium-replay-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

100

Hopper-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

100
Ant-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

Walker2d-random-v2

0
100000 200000 300000 400000 500000

Online Transitions

0

20

40

60

Normalized Reward

Halfcheetah-random-v2

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)

Figure 6: Results on MuJoCo [58] Environments. The TD3 gradient significantly improves the
overall performance of the decision transformer; autoregressive algorithms, such as ODT and PDT,
fails to improve policy in most cases (especially on random dataset), while TD3+BC and IQL’s
improvement during finetuning is generally limited.

architecture as ODT which outputs a squashed Gaussian distribution with trainable mean and standard
deviation). It is apparent that TD3 [21] and SAC [25] are the RL algorithms that suit the decision
transformer best. Fig. 9 further shows the performance comparison between a decision transformer
with SAC+ODT mixed gradient and TD3+ODT mixed gradient (both with coefficient 0.1). The result
shows that TD3 is the better choice when paired with supervised learning.

Note, While PPO is generally closely related with transformers (e.g., Reinforcement Learning from
Human Feedback (RLHF) [44]), and was used in some prior work for online finetuning of decision
transformers with a small, discrete action space [38], in our experiments, we find PPO generally
does not work with the decision transformer architecture. The main reason for this is the importance
sampling issue: PPO has the following objective for an actor πθ parameterized by θ:

max
θ
min

E(s,a)∼πθold

πθ(a|s)
πθold(a|s)Aπθold(s, a), clip
 πθ(a|s)

πθold(a|s), 1 −ϵ, 1 + ϵ

Aπθold (s, a)

.
(7)

Here, πθold is the policy at the beginning of the training for the current epoch. Normally, the
denominator of the red part, πθold(a|s), would be reasonably large, as the data is sampled from that
distribution. However, because of the offline nature caused by different RTGs and context lengths at
rollout and training time, the denominator for the red part in Eq. (7) could be very small in training,
which will lead to a very small loss if Aπ
θold(s, a) > 0. This introduces significant instability during
the training process. Fig. 10 illustrates the instability and degrading performance of a PPO-finetuned
decision transformer.

16


---Page Break---
0
20
40
60
80
100

TD3+BC

IQL

ODT

PDT

TD3

DDPG+ODT

TD3+ODT (ours)

Median

0
20
40
60
80
100

IQM

20
40
60
80

Mean

30
45
60
75
90

Optimality Gap

(a) Adroit

15
30
45
60
75
90

TD3+BC

IQL

ODT

PDT

TD3

DDPG+ODT

TD3+ODT (ours)

Median

15
30
45
60
75
90

IQM

30
45
60
75
90

Mean

15
30
45
60
75

Optimality Gap

(b) MuJoCo

0
20
40
60
80
100

TD3+BC

IQL

ODT

PDT

TD3

DDPG+ODT

TD3+ODT (ours)

Median

0
20
40
60
80
100

IQM

20
40
60
80
100

Mean

20
40
60
80
100

Optimality Gap

(c) Antmaze (umaze and medium)

40
80
120
160
200

TD3+BC

IQL

ODT

PDT

TD3

DDPG+ODT

TD3+ODT (ours)

Median

40
80
120
160
200

IQM

150
175
200
225
250
275

Mean

0
10
20
30
40
50

Optimality Gap

(d) Maze2d

Figure 7: Our main final results re-evaluated using the rliable library with 10000 bootstrap replications.
The x-axes are normalized scores (optimality gap is
R 100
0
Pr(reward ≤x)dx). Our method indeed
outperforms all baselines on Adroit, MuJoCo and antmaze (umaze and medium).

In contrast, RLHF, does not exhibit such a problem: it does not use different return-to-go and context
length in evaluation and training. Thus RLHF does not encounter the problem described above.

Besides the RL gradients mentioned above, as IQL works well on large Antmazes, we also explore
the possibility of using IQL as the RL gradient for decision transformer instead of TD3. We found
that IQL gradients, when applied to the decision transformer, indeed lead to much better results on
antmaze-large. However, IQL fails to improve the policy when the offline dataset subsumes very low
reward trajectories, which does not conform with our motivation. This is probably because IQL, as
an offline RL algorithm, aims to address out-of-distribution evaluation issue, which is a much more
important source of improvement in exploration in the online case. Thus, we choose TD3 as the RL
gradient applied to decision transformer finetuning in this work. Fig. 11 shows the result of adding
TD3 gradient vs. adding IQL gradient on Antmaze-large-play-v2 and hopper-random-v2.

D
Why Our Critic Serves as an Average of Policies Generated by Different
Context Lengths?

As we mentioned in Sec. 2, When updating a deterministic DT policy, the following loss is minimized:

Ttrain
X

t=1

µDT (s0:t, a0:t−1, RTG0:t, RTG = RTGreal, T = t) −at
2
2 ,
(8)

where Ttrain is the training context length and RTGreal is the real return-to-go.

17


---Page Break---
0

25

50

75

100

Hopper-medium-v2

0

50

100

Ant-medium-v2

0

20

40

60

80

100

Walker2d-medium-v2

0

20

40

60

80

Halfcheetah-medium-v2

0

25

50

75

100

Hopper-medium-replay-v2

50

0

50

100

Ant-medium-replay-v2

0

20

40

60

80

100

Walker2d-medium-replay-v2

0

20

40

60

80 Halfcheetah-medium-replay-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

100
Hopper-random-v2

0
100000 200000 300000 400000 500000

25

0

25

50

75

Ant-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

Walker2d-random-v2

0
100000 200000 300000 400000 500000

Online Transitions

0

20

40

60

Normalized Reward

Halfcheetah-random-v2

ODT
TD3
SAC
AWAC
AWR
PPO

Figure 8: Performance comparison of different, pure RL gradients for online finetuning on standard
D4RL benchmarks, with TD3 [21], SAC [25], AWAC [40], AWR [47] and PPO [52]. We also plot
ODT’s performance as a reference. The result shows that generally, TD3 and SAC work the best,
while PPO does not work at all.

40

60

80

100

120

Hopper-medium-v2

0

50

100

Ant-medium-v2

20

40

60

80

100

120
Walker2d-medium-v2

0

20

40

60

80

Halfcheetah-medium-v2

0

25

50

75

100

Hopper-medium-replay-v2

80

100

120

140

Ant-medium-replay-v2

0

25

50

75

100

Walker2d-medium-replay-v2

20

40

60

Halfcheetah-medium-replay-v2

0
100000 200000 300000 400000 500000
0

20

40

60

80

100

Hopper-random-v2

0
100000 200000 300000 400000 500000
0

20

40

60

80

100
Ant-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

Walker2d-random-v2

0
100000 200000 300000 400000 500000

Online Transitions

0

20

40

60

Normalized Reward

Halfcheetah-random-v2

TD3+ODT
SAC+ODT

Figure 9: Performance comparison between SAC+ODT and TD3+ODT on standard D4RL bench-
marks. TD3+ODT significantly outperforms SAC+ODT.

However, if we consider a particular action at in some trajectory τ of the dataset, during training
(both offline pretraining and online finetuning), the policy generated by the decision transformer
fitting at will be

at = µDT(st−T :t, at−T :t−1, RTGt−T :t, T ∼U ′(1, Ttrain), RTG = RTGreal),
(9)

T is actually sampled from a distribution U ′(1, Ttrain) over integers between 1 and Ttrain inclusive;
this distribution U ′ is introduced by the randomized starting step of the sampled trajectory segments,
and is almost a uniform distribution on integers, except that a small asymmetry is created because the
context length will be capped at the beginning of each trajectory. See Fig. 12 for an illustration.

Therefore, online decision transformers (and plain decision transformers) are actually trained to
predict with every context length between 1 and Ttrain. During the training process, the context length
is randomly sampled according to U ′, and a critic is trained to predict an “average value” for the
policy generated with context length sampled from U ′.

18


---Page Break---
(a) Reward
(b) The ratio
πθ(a|s)
πθold (a|s)

Figure 10: An illustration of the training instability and the corresponding performance of a PPO-
finetuned decision transformer. The x axis is 300× the number of gradient steps.

0
100000
200000
300000
400000
500000
0

20

40

60

80

100

Hopper-random-v2

0.0
0.2
0.4
0.6
0.8
1.0

1e6

0

20

40

60

Antmaze-large-play-v2

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

10

20

30

40

50

Normalized Reward

Antmaze-large-diverse-v2

IQL+ODT
TD3+ODT

Figure 11: Performance comparison between IQL+ODT and TD3+ODT. While IQL gradient is good
at both large antmaze environments, it is much easier to fall into local minima in low-reward offline
dataset such as Hopper-random-v2.

E
Mathematical Proofs

In this section, we will state the theoretical analysis summarized in Sec. 3.3 more rigorously. We
will first provide an explanation on how the decision transformer improves its policy during online
finetuning, linking it to an existing RL method in Sec. E.1 and Sec. E.2. We will then bound its
performance in Sec. E.3.

E.1
Preliminaries

Advantage-Weighted Actor Critic (AWAC) [40] is an offline-to-online RL algorithm, where the
replay buffer is filled with offline data during offline pretraining and then supplemented with online
experience during online finetuning. AWAC uses standard Q-learning to train the critic Q : |S| ×
|A| →R, and update the actor using weighted behavior cloning, where the weight is exponentiated

advantage (i.e., exp

A(s,a)

λ

where λ > 0 is some constant).

E.2
Connection between Decision Transformer and AWAC

We denote β as the underlying policy of the dataset, and Pβ as the distribution over states, actions
or returns induced by β. Note such Pβ can be either discrete or continuous. By prior work [7], for
decision transformer policy πDT, we have the following formula holds for any return-to-go RTG ∈R
of the future trajectory:

πDT(a|s, RTG) = Pβ(a|s, RTG) = Pβ(a|s)Pβ(RTG|s, a)

Pβ(RTG|s)
= β(a|s)Pβ(RTG|s, a)

Pβ(RTG|s) .
(10)

Based on Eq. (10), we have the following lemma:

Lemma 2. For state-action pair (s, a) in an MDP, RTG ∈R, assume the distributions of return-to-go
(RTG) Pβ(RTG|s, a) and Pβ(RTG|s) are Laplace distributions with scale σ, then for any RTG large

19


---Page Break---
RTG1
𝑠𝑠1
𝑎𝑎1
RTG2
𝑠𝑠2
𝑎𝑎2
RTG3
𝑠𝑠3
𝑎𝑎3
RTG4
𝑠𝑠4
𝑎𝑎4

𝑇𝑇eval

𝑇𝑇train

𝑇𝑇2

𝑎𝑎2
𝑎𝑎3
𝑎𝑎4

Causal Transformer

(a) Context length T2 during training

𝑇𝑇2
′
𝑇𝑇2

Trajectory 𝜏𝜏

𝑇𝑇train

Step 𝑗𝑗≥𝑇𝑇train

𝑇𝑇train

Step 𝑖𝑖< 𝑇𝑇train

𝑇𝑇train

(b) Distribution U ′ of T2

Figure 12: a) illustrates the context length T2 during training; Teval is the context length of a2 upon
sampling and evaluation. It is easy to see that T2 is randomized during training due to the left endpoint
of the sampled trajectory segment. b) shows the distribution U ′ of T2; while T ′
2 for step j ≥Ttrain is
uniformly sampled between 1 and Ttrain because the start of the segment is uniformly sampled, T2 for
step i < Ttrain will be capped at the start of the trajectory. Thus U ′ is not exactly uniform.

𝑉𝛽(𝑠)

𝑅𝑇𝐺

𝑃𝛽(𝑅𝑇𝐺)

𝑄𝛽(𝑠, 𝑎)

𝑃𝛽(𝑅𝑇𝐺|𝑠)
𝑃𝛽(𝑅𝑇𝐺|𝑠, 𝑎)

𝑃𝛽𝑅𝑇𝐺𝑠, 𝑎

𝑃𝛽(𝑅𝑇𝐺|𝑠) ∝exp 𝑅𝑇𝐺−𝑄𝛽𝑠, 𝑎

𝑅𝑇𝐺−𝑉𝛽𝑠, 𝑎
= exp(𝐴𝛽𝑠, 𝑎)

𝑅𝑇𝐺

Figure 13: An illustration of how decision transformer policy update at RTGeval is related to AWAC.
Though the assumption is strong to form the exact same formula, it shows the basic idea of how we
link P β(RTG) to its distance to V β(s) and Qβ(s, a) in this paper.

enough (more rigorously, RTG ≥max{Qβ(s, a), V β(s)}), π(a|s, RTG) is updated in the same way
as AWAC.4

Proof. If return-to-go are Laplace distributions, then by the symmetric property of such distributions,
the mean of RTG given s or (s, a) would be the expected future return, which by definition are
value functions for β, i.e., V β(s) for Pβ(RTG|s) and Qβ(s, a) for Pβ(RTG|s, a). See Fig. 13 as an
illustration. As RTGeval ≥max{Qβ(s, a), V β(s)}, we have

Pβ(RTG|s, a) = pβ(RTG|s, a) = 1

2σ exp

−RTG −Qβ(s, a)

σ


,

Pβ(RTG|s) = pβ(RTG|s, a) = 1

2σ exp

−RTG −V β(s)

σ


.
(11)

And thus we have Pβ(RTG|s,a)

Pβ(RTG|s) = exp

Qβ(s,a)−V β(s)

σ

= exp( Aβ(s,a)

σ
), where Aβ is the advantage
function.

While the Laplace distribution assumption in this lemma is strict and impractical for real-life applica-
tions, it gives us three crucial insights for a decision transformer:

• For any state s or state-action pair (s, a), we have Eβ[RTG|s, a] = Qβ(s, a), Eβ[RTG|s] =
V β(s), which is an important property of Pβ on RTG;

4Decision transformers have a sequence of past RTGs as input, but the past sequence can be augmented into
the state space to fit into such form.

20


---Page Break---
• For decision transformer, the ability to improve the policy by collecting rollouts with high
RTG, similar to AWAC, is closely related to advantage. In the above lemma, for example, if
the two Laplace distributions have different scales σV , σQ, we will have the ratio Pβ(RTG|s,a)

Pβ(RTG|s)
being exp

Qβ(s,a)

σQ
−V β(s)

σV


; if the distributions are Gaussian, we will get similar results

but with quadratic terms of Qβ and V β.

• Different from AWAC, the “policy improvement” of online decision transformers heavily
relies on the global property of the return-to-go as RTGeval moves further away from Qβ
and V β. If the return-to-go is far away from the support of the data, we will have almost no
data to evaluate Pβ, and its estimation can be very uncertain (let alone ratios). In this case,
it is very unlikely for the decision transformer to collect rollouts with high RTGtrue and get
further improvement. This is partly supported by the corollary of Brandfonbrener et al. [7],
where the optimal conditioning return they found on evaluation satisfies RTGeval = V β(s1)
at the initial state s1. This is also supported by our single-state MDP experiment discussed
in Sec. 3.1 and illustrated in Fig. 2.

Those important insights lead to the intuition that decision transformers, when finetuned online, lack
the ability to improve “locally” from low RTGtrue data, and encourages to study the scenario where
Pβ(RTG|s) and Pβ(RTG|s, a) are small.

E.3
Failure of ODT Policy Update with Low Quality Data

As mentioned in Sec. E.2, we study the performance of decision transformers in online finetuning
when Pβ(RTG|s) and Pβ(RTG|s, a) is small. Specially, in this section, r(s, a) is not a reward
function, but a reward distribution conditioned on (s, a); ri ∼r(si, ai) is the reward obtained
on i-th step. Such notation takes the noise of reward into consideration and forms a more general
framework. Also, as the discrete or continuous property of β is important in this section, we will use
Prβ to represent probability mass (for discrete distribution or cumulative distribution for continuous
distribution) and pβ to represent probability density (for probability density function for continuous
distribution).

By prior work [7], we have the following performance bound tight up to a constant factor for decision
transformer for every iteration of updates:

Theorem E.1. For any MDP with transition function p(·|s, a) and reward random variable r(s, a),
and any condition function f, assume the following holds:

• Return coverage: Pβ(g = f(s1)|s1) ≥αf for any initial state s1;

• Near determinism: for any state-action pair (s, a), ∃s′ such that Pr(s′|s, a) ≥1 −ϵ, and
∃r0(s, a) such that Pr(r(s, a) = r0(s, a)) ≥1 −ϵ;

• Consistency of f: f(s) = f(s′) + r for all s when transiting to next state s′.

Then we have

Es1∼pini [f(s1)] −Eτ=(s1,a1,...,sH,aH)∼πDT(·|s,f(s))

" H
X

i=1
Eri∼r(si,ai)ri

#

≤ϵ
 1

αf
+ 2

H2, (12)

where αf > 0, ϵ > 0 are constants, pini is the initial state distribution, and H is the horizon of the
MDP. πDT is the learned policy by Eq. (10).

Proof. See Brandfonbrener et al. [7].

In our case, we define f(s) as follows:

Definition E.2. f(s1) = RTGeval for all initial states s1, f(si+1) = f(si) −ri for the (i + 1)-th step
following i-th step (i ∈{1, 2, . . . , T −1}).

21


---Page Break---
Further, we enforce the third assumption in Thm. E.1 by including the cumulative reward so far in the
state space (as described in the paper of Brandforbrener et al. [7]). Under such definition, we have a
tight bound on the regret between our target RTGeval and the true return-to-go RTGtrue = PH
i=1 ri by
our learned policy at optimal, based on current replay buffer in online finetuning.

We will now prove that under certain assumptions,
1
αf grows superlinearly with respect to RTG; as

the bound is tight, the expected cumulative return term Eτ=(s1,a1,...,sH,aH)∼β
hPH
i=1 Eri∼r(si,ai)ri
i

will be decreasing to meet the bounds.

To do this, we start with the following assumptions:

Assumption E.3. We assume the following statements to be true:

• (Bounded reward) We assume the reward is bounded in [0, Rmax] for any state-action pairs.

• (High evaluation RTG) RTGeval ≥RTGβmax, where RTGβmax is the largest RTGtrue in the
dataset of n trajectories generated by β.

• (Beta prior) We assign the prior distribution of RTG generated by policy β to be a
Beta distribution Beta(1, 1) for the binomial likelihood of RTG falling on [0, RTGβmax]
or [RTGβmax, TRmax].

Remark E.4. The Beta distribution can be changed to any reasonable distribution; we use Beta
distribution only for a convenient derivation. Considering the fact that by common sense, trajectories
with high return are very hard to obtain, we can further strengthen the conclusion by changing the
prior distribution.

We then prove the following lemma:

Lemma 3. Under the Assumption E.3, given underlying policy β of the dataset, for any state s
with value function V β(s) and any state-action pair (s, a) with Q-function Qβ(s, a), any c ≥0 and
RTGeval ∈R, with probability 1 −δ , we have

Prβ(RTGeval −V β(s) ≥c|s) ≤(1 −ϵ)RTGβmax + ϵR2
maxT 2 −

V β(s)
2

c2
,

Prβ(RTGeval −Qβ(s, a) ≥c|s, a) ≤(1 −ϵ)RTGβmax + ϵR2
maxT 2 −

Qβ(s, a)
2

c2
,

(13)

where δ = 1 −CDFBeta(n+1,1)(ϵ), and the CDF is the cumulative distribution function.

Proof. With the beta prior assumption in Assumption E.3, we know that with n samples where
RTG ≤RTGβmax, we have the posterior distribution to be Beta(n + 1, 1), i.e., with probability
1 −CDFBeta(n+1,1)(ϵ), we have Prβ(RTG ≥RTGβmax) ≤ϵ for ϵ > 0.

Thus, by Chebyshev inequality, we know

Prβ
 
RTG −V β(s) ≥c|s

≤
Eβ

RTG2
−E2
β [RTG]
c2

≤(1 −ϵ)RTGβmax + ϵR2
maxT 2 −

V β(s)
2

c2
,

(14)

and a similar conclusion holds for Prβ
 
RTG −Qβ(s, a) ≥c|s, a

. Thus, the probability decays
superlinearly with respect to RTG.

Given this lemma, it remains to connect the bound of Prβ(RTG ≥c0) to Pβ(RTG = c0) on RTG,
c0 ∈R. For discrete distribution, the connection is straightforward: Prβ(RTG ≥c0) ≥Prβ(RTG =
c0) = Pβ(RTG = c0) for any condition s or (s, a).

Thus, we immediately get the following corollary:

22


---Page Break---
Corollary 2. Assume the reward is bounded in [0, Rmax] for any state and action, and the number of
possible different return-to-go one can get is finite or countably infinite. Then for the f condition func-
tion defined in Def. E.2, with probability of at least 1−δ, we have αf ≤(1−ϵ)RTGβmax+ϵR2
maxT 2−[V β(s)]2

(RTG−V β(s))2
,

i.e.,
1
αf grows in the order of Ω(RTG2
eval).

Remark E.5. While the limitation on return-to-go seems strong theoretically, it is very easy to satisfy
such assumption in practice because it has no requirement on the discreteness of state and action
space. Such corollary can be applied on reward discretization with arbitrary precision (including
implicit ones by float precision).

For continuous distribution, to bound pβ(RTG = c0) with Prβ(RTG ≥c0) on RTG, we would need
to assume that “peak” does not exist (see Fig. 14 for illustration), i.e. there does not exist cases where
pβ is large but Prβ is small. Thus, we made the following assumption:
Assumption E.6. (Lipschitzness on uncovered RTG distribution) pβ(RTG|s) is KV -Lipschitz
where RTG is larger than RTGβmax.
Remark E.7. The assumption is reasonable because when RTG is larger than any of the RTGtrue in
the dataset, we have no data coverage for the performance of the underlying policy β under such
RTG, and thus we can choose any inductive bias for β.
Remark E.8. Note the Lipschitzness of pβ does not rely on the Lipschitzness of the reward function.
For example, consider a single-state, single-step MDP where we have a uniformly random policy
a ∼U(0, 1) with reward r(a) = 2 −1

a. The reward is clearly not Lipschitz on a ∈(0, 1), but the

distribution of r is pr(r0) = pa(r−1(r0)) · ∂r−1(r0)

∂r
=
1
(r−2)2 , which is Lipschitz on (−∞, 1) as
a ∈(0, 1).

With such assumption, we have the following corollary:
Corollary 3. Under assumption E.6, for the f condition function defined in Def. E.2, we have
αf ≤√2KV Ω
 
RTG−1.5
eval

, i.e.,
1
αf grows in the order of Ω(RTG1.5
eval) with probability of at least
1 −δ.

Proof. Under assumption E.6, by Lipschitzness, for any RTG where pβ(RTG|s) > p0, we have
pβ(RTG + c|s) > p0 −KV · c for any c ∈[0, p0

KV ].

Thus, we know that if ∃RTG0 > RTGβmax such that pβ(RTG0|s) > p0, then we have Prβ(RTG ≥

RTG0|s) >
p2
0
2KV (See Fig. 14 for an illustration). By the contra-positive statement of the above
conclusion, we know that

Prβ (RTG ≥RTG0|s) ≤
p2
0
2KV ⇒∀RTG ≥RTG0, pβ(RTG|s) ≤p0,
(15)

and RTGeval is applicable to the inequality above by the high evaluation RTG assumption in Assump-
tion E.3. We then apply the proof lemma 3, but apply the inequality P (x −E[x] ≥c) ≤E(x−E[x])3

c3
instead of Chebyshev inequality which leads to the conclusion.

F
Experimental Details

F.1
Environment and Dataset Details

F.1.1
Single-State MDP

The single-state MDP studied in Sec. 3.1 motivates why RL gradients are useful for online finetuning.
It has a single state, a single action a ∈[−1, 1], and a reward function r(a) = (a + 1)2 if a ≤0 and
r(a) = 1 −2a otherwise.

Datasets. The dataset has a size of 128, with 100 actions uniformly sampled in (−1, 0.95), and the
remaining 28 actions uniformly sampled in (0.5, 1). The dataset is designed to conceal the reward
peak in the middle. DDPG and ODT+DDPG successfully recognized the reward peak but ODT
failed.

23


---Page Break---
𝑝0

𝑉𝛽𝑠+ 𝑐

𝑉𝛽(𝑠)

𝑅𝑇𝐺

𝑝𝛽(𝑅𝑇𝐺|𝑠)

Probability mass 

𝑝02

2𝐾𝑉

Figure 14: An illustration of how Lipschitzness on the distribution of pβ(RTG|s) could link the
bound between Pr(RTG ≥V β(s) + c|s) and pβ(RTG|s). Note we do not take the left-hand side
probability mass of p0 into account because the triangle of probability mass could be truncated by
V β(s).

F.1.2
Adroit Environments

Environments. Adroit is a set of more difficult benchmark than Mujoco in D4RL, and is becoming
increasingly popular in recent offline and offline-to-online RL works [28, 23]. We test four environ-
ments in adroit in our experiments: pen, hammer, door and relocate. Fig. 15 shows an illustration of
the four environments.

(a) Pen
(b) Hammer
(c) Door
(d) Relocate

Figure 15: Illustration of Adroit environments used in Sec. 4 based on OpenAI Gym [8] and
D4RL [19].

1. Pen. Pen is a locomotion environment where the agent needs to control a robotic hand
to manipulate a pen, such that its orientation matches the target. It has a 24-dimensional
action space, each of which controls a joint on the wrist or fingers. The state space is
45-dimensional, which contains the pose of the palm, the angular position of the joints, and
the pose of the target and current pen.
2. Hammer. Hammer is an environment where the agent needs to control a robotic hand to
pick up a hammer and use it to drive a nail into a board. The action space is 26-dimensional,
each of which corresponds to a joint on the hand. The state space is 46-dimensional, which
describes the angular position of the fingers, the pose of the palm, and the status of hammer
and nail.
3. Door. In the door environment, the agent needs to use a robotic hand to open a door by
undoing the latch and swinging it. The environment has a 28-dimensional action space,
which are the absolute angular positions of the hand joints. It also has 39-dimensional
observation space which describes each joint, the pose of the palm, and the door with its
latch.
4. Relocate. In the relocate environment, the agent needs to control a robotic hand to move a
ball from its initial location towards a goal, both of which are randomized in the environment.
The environment has a 30-dimensional action space which describes the angular position of
the joints on the hand, and a 39-dimensional space which describes the hand as well as the
ball and target.

24


---Page Break---
Dataset
Size
Normalized Reward
Pen-expert-v1
499106
107.40 ± 55.65
Pen-cloned-v1
499886
108.63 ± 122.43
Pen-human-v1
4800
202.69 ± 154.48
Hammer-expert-v1
999800
96.95 ± 50.65
Hammer-cloned-v1
999872
8.11 ± 23.35
Hammer-human-v1
10948
23.80 ± 33.36
Door-expert-v1
999800
101.19 ± 16.31
Door-cloned-v1
999939
12.29 ± 18.35
Door-human-v1
6504
28.35 ± 13.88
Relocate-expert-v1
999800
102.25 ± 19.83
Relocate-cloned-v1
999724
28.99 ± 42.88
Relocate-human-v1
9614
87.22 ± 21.28
Table 4: The size and the average and standard deviation of the normalized reward of the Adroit
datasets from D4RL [19] used in our experiments.

Datasets. For each of the four environments, we test our method across three different qualities of
datasets: expert, cloned and human, all of which provided by the DAPG [49] repository. The expert
dataset is generated by a fine-tuned RL policy; the cloned dataset is collected from an imitation policy
on the demonstrations from the other two datasets; and the human dataset is collected from human
demonstrations. Tab. 4 shows the size and average reward of each dataset.

F.1.3
Antmaze Environments

Environments. Antmaze is a more difficult version of Maze2D, where the agent controls a robotic
ant instead of a point mass through the maze. It has a 27 dimensional-state space and a 8-dimensional
action space. We test our method on six variants of antmaze: Umaze, Umaze-Diverse, Medium-Play,
Medium-Diverse, Large-Play and Large-Diverse, where “Umaze”, “Medium” and “Large” describes
the size of the maze (see Fig. 16 for an illustration), and the “Diverse” and “Play” describes the type
of the dataset. More specifically, “Diverse” means that in the offline dataset, the starting point and
the goal of the agent are randomly generated, while “Play” means that the goal is generated by a
handcraft design. “Umaze” without suffix is the simplest environment where both the starting point
and the goal are fixed.

(a) Open
(b) Umaze
(c) Medium
(d) Large

Figure 16: Illustration of mazes in antmaze and maze2D environment, where the red point is the goal
and the green point is the current location of the agent.

Datasets. Similar to Adroit and MuJoCo, we test our method on datasets provided by D4RL. Tab. 5
shows the size and normalized reward of each dataset. Note, following IQL [28] and CQL [29], we
conduct reward shaping: subtracting from all rewards in the dataset and environment the value 1
during training of both our method and baselines to provide denser reward signal for all antmaze
environments. However, we still count original sparse reward when comparing the performance.

F.1.4
MuJoCo

Environments. We test our method on four widely used environments: Hopper, Halfcheetah,
Walker2d and Ant. Fig. 17 shows an illustration of the four environments.

25


---Page Break---
Dataset
Size
Normalized Reward
Antmaze-Umaze-v2
998573
86.14 ± 34.55
Antmaze-Umaze-Diverse-v2
999000
3.48 ± 18.32
Antmaze-Medium-Play-v2
999000
90.85 ± 28.83
Antmaze-Medium-Diverse-v2
999000
66.29 ± 47.27
Antmaze-Large-Play-v2
999000
92.73 ± 25.97
Antmaze-Large-Diverse-v2
999000
86.17 ± 34.52
Table 5: The size and the average and standard deviation of the normalized reward of the Antmaze
datasets from D4RL [19] used in our experiments.

(a) Hopper
(b) Halfcheetah
(c) Ant
(d) Walker2d

Figure 17: Illustration of MuJoCo environments used in Sec. 4 based on OpenAI Gym [8] and
D4RL [19].

1. Hopper. Hopper is a locomotion task on a 2D vertical plane, where the agent manipulates a
single-legged robot to hop forward. Its state is 11-dimensional, which describes the angle
and velocity for the robot’s joints. Its action is 3-dimensional, which corresponds to the
torques applied on the three joints for the current time step respectively.

2. Halfcheetah. Halfcheetah is also a 2D environment which requires the agent to control a
cheetah-like robot to run forward. The states are 17-dimensional, containing the coordinate
and velocity of the joints The actions are 6-dimensional, which control the torques on the
joints of the robot.

3. Ant. In Ant, the agent controls a four-legged 8-DoF robotic ant to walk in a 3D environment
and tries to move forward. It has a 111-dimensional state space describing the coordinates
and velocities of the joints.

4. Walker2d. Walker2d is a 2D environment in which the agent needs to manipulate a 8-DoF
two-legged robot to walk forward under the agent’s control. Its state space is 27-dimensional.

Datasets. We test our method across three different qualities of datasets: medium, medium-replay
and random. The medium dataset contains trajectories collected by an agent trained with RL,
but early-stopped at medium-level performance. The medium-replay dataset is the collection of
trajectories sampled in the training process of the agent mentioned above. The random dataset
contains trajectories collected by an agent with random policy. Tab. 6 shows the size and normalized
reward of each dataset.

F.1.5
Maze2D Environments

Environments. Maze2D is another set of D4RL environment, where the agent needs to control a
point mass to navigate through a 2D maze and arrive at a fixed goal. It has a 4-dimensional state
space describing its coordinate and velocity, and a 2-dimensional action describing its acceleration.
The reward is determined by its current distance to the goal. We test our method on four variants
of maze: Open, Umaze, Medium and Large with increasing difficulty. The map of each maze is
illustrated in Fig. 16. Maze2D environment is tested in Sec. G.2.

Datasets. We again test our method on datasets provided by D4RL. Tab. 7 shows the size and
normalizeed reward of each dataset.

26


---Page Break---
Dataset
Size
Normalized Reward
Hopper-medium-v2
999906
44.32 ± 12.27
Hopper-medium-replay-v2
402000
14.98 ± 16.32
Hopper-random-v2
999996
1.19 ± 1.16
HalfCheetah-medium-v2
1000000
40.68 ± 5.12
HalfCheetah-medium-replay-v2
202000
27.17 ± 15.79
HalfCheetah-random-v2
1000000
0.07 ± 2.90
Walker2d-medium-v2
999995
62.09 ± 23.83
Walker2d-medium-replay-v2
302000
14.84 ± 19.48
Walker2d-medium-random-v2
999997
0.01 ± 0.09
Ant-medium-v2
999946
80.30 ± 35.82
Ant-medium-replay-v2
302000
30.95 ± 31.66
Ant-medium-random-v2
999930
6.36 ± 10.07
Table 6: The size and the average and standard deviation of the normalized reward of the MuJoCo
datasets from D4RL [19] used in our experiments.

Dataset
Size
Normalized Reward
Maze2D-Open-v0
999999
30.08 ± 50.17
Maze2D-Umaze-v1
999869
−12.55 ± 9.82
Maze2D-Medium-v1
1999733
−3.46 ± 3.95
Maze2D-Large-v1
3999692
−1.71 ± 2.87
Table 7: The size and the average and standard deviation of the normalized reward of the Maze2D
datasets from D4RL [19] used in our experiments.

F.2
Hyperparameters

F.2.1
Single-State MDP

For all networks, we use a simple MDP with two hidden layers of width 128, and ReLU [1] as
activation function. We add a Tanh activation function to limit the output for ODT and the actor of
DDPG to [−1, 1]. For both methods, we use Adam [27] as the optimizer, and the learning rate is set
to 10−3. We pretrain 5 epochs on offline data (20 gradient steps) and 16 epochs for online finetuning,
with a batch size of 32 for gradient update and collect 64 new rollout states for each epoch (thus
we train 2n + 4 steps for the n-th online finetuning epoch). RTGeval is set at 1, which serves as the
(constant) input for ODT rollout and DDPG actor. Both DDPG and ODT uses deterministic actor
with an exploration noise uniform in [−0.01, 0.01] during online rollouts.

F.2.2
Other Experiments

Tab. 8 summarizes hyperparameters that are common across all environments, and Tab. 9 summarizes
hyperparameters that are different across environments. For environments that exist in ODT [74], we
follow the hyperparameters from ODT medium environments. We did not use positional embedding
as suggested by ODT [74]. Specially, for antmaze, we remove most (all but 10) 1-step trajectories,
because the size of the replay buffer for decision transformers is controlled by the number of
trajectories, and antmaze dataset contains a large number of 1-step trajectories due to its data
generation mechanism (immediately terminate an episode when the agent is close to the goal, but do
not reset the agent location). Also, we add Layernorm [4] after each hidden layer of the critic for
Adroit, Maze and Antmaze environments, according to Yue et al. [71]’s advice. We found that such
practice stabilizes the training process (see Sec. G.5 for ablation).

For ODT and TD3 baseline, we use the same code as our TD3+ODT, while setting coefficients for RL
and supervised gradients accordingly. For PDT baseline, we use the default hyperparameter in PDT
paper, and pretrain PDT for 40K steps for all experiments. For TD3+BC and IQL, we use the default
hyperparameter in their codebase, and pretrain them for 1M steps for all experiments (remaining the
same as that in the codebase).

27


---Page Break---
Hyperparam
Value
# dim of embedding dimensions
512
# of attention heads per layer
4
# of transformer layers
4
Dropout
0.1
Actor Optimizer
LAMB [68]
# steps collected per epoch
≥1000 (random dataset), 1 trajectory (others)
Actor activation function
ReLU
Scheduler
104 steps, linear warmup
Critic layer
2
Critic width
256
Critic activation function
ReLU
Batch size
256
# actor update per epoch
300
Online exploration noise
0.1
TD3 policy noise
0.2
TD3 noise clip
0.5
TD3 target update ratio
0.005
Table 8: The common hyperparameters across all environments used in our experiments.

Ttrain Teval RTGeval RTGonline Pretrain α Online α
γ
lrc
lra
Weight decay # pretrain steps Buffer size
Hopper
20
5
3600
7200
0
0.1
0.99
10−3 10−4
0.0005
5K
1K
HalfCheetah
20
5
6000
12000
0
0.1
0.99
10−3 10−4
10−4
5K
1K
Walker2d
20
5
5000
10000
0
0.1
0.99
10−3 10−3
10−3
10K
1K
Ant
20
1
6000
12000
0
0.1
0.99
10−3 10−3
10−4
5K
1K
Pen
5
1
12000
12000
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
Hammer
5
5
16000
16000
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
Door
5
1
4000
4000
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
Relocate
5
1
5000
5000
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
Maze2D-Open
1
1
120
120
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
-Umaze
1
1
60
60
0
0.1
0.99
0.0002 10−4
10−4
40K
2.5K
-Medium
1
1
60
60
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
-Large
5
1
60
60
0
0.1
0.99
0.0002 10−4
10−4
40K
5K
Antmaze-Umaze
5
1
-100
-100
0.1
0.1
0.998 0.0002 10−4
10−4
40K
2K
-Medium
1
1
-200
-200
0.1
0.1
0.998 0.0002 10−4
10−4
200K
2K
-Large
5
1
-500
-500
0.1
0.1
0.998 0.0002 10−4
10−4
200K
2K
Table 9: Environment-specific hyperparameters, where Ttrain and Teval stands for training and eval-
uation context length, RTGeval and RTGonline represents RTG during evaluation and online rollout
respectively, α is the coefficient for RL gradient, γ is the discount factor, lrc is the critic learning rate,
and lra is the actor learning rate. Buffer size is counted in the number of trajectories. Note RTGs of
antmaze have been modified according to our reward shaping.

G
More Results

G.1
Delayed Reward

Though we have tested MuJoCo environments in Sec. 4, it is worth noting that many offline RL
algorithms have addressed the MuJoCo benchmark quite well [28, 23]. Thus, we also tested settings
where RL struggles to obtain good performance to further analyze the performance of using RL
gradients for decision transformers.

Environment and Experimental Setup. In this experiment, we use the same experiment and dataset
as in Sec. 4 except for one major difference: the rewards are not given immediately after each step.
Instead, the cumulative reward during a short period of M steps is given only at the end of the period,
while the rewards observed by the agents within a period are all zero. We adopt such a setting from
prior influential work [43], which creates a sparse-reward setting where RL algorithms struggle. We
test M = 5 in this experiment.

Results. We use the same baselines as that in Sec. 4; Tab. 10 summarizes the performance of each
method. Generally, DT with TD3 gradient still works very well, much better than ODT. While

28


---Page Break---
TD3+BC
IQL
ODT
PDT
TD3
TD3+ODT (ours)
Ho-M-v2
50.2(-1.43)
43.09(-21.28)
92.9(+43.03)
83.56(+81.73)
82.11(+17.59)
82.91(+18.61)
Ho-MR-v2
99.85(+42.33)
79.63(-4.72)
85.07(+67.6)
82.2(+80.21)
90.81(+48.79)
98.55(+67.1)
Ho-R-v2
8.8(+0.3)
18.49(+10.62)
30.58(+28.37)
25.1(+23.86)
75.55(+73.57)
52.38(+50.4)
Ha-M-v2
50.45(+2.56)
40.31(-6.88)
42.07(+20.49)
38.71(+38.66)
65.92(+24.91)
66.33(+26.1)
Ha-MR-v2
53.59(+8.82)
47.01(+3.46)
39.45(+22.36)
32.55(+32.76)
57.42(+28.48)
49.95(+20.23)
Ha-R-v2
42.11(+28.73)
34.87(+28.34)
2.16(-0.07)
-0.85(+0.96)
52.24(+49.99)
48.79(+46.54)
Wa-M-v2
84.73(+2.19)
62.01(-16.59)
76.21(+3.65)
59.79(+59.52)
86.42(+18.88)
87.83(+19.57)
Wa-MR-v2
61.1(-11.8)
86.67(+13.93)
76.96(+5.25)
53.73(+37.17)
96.38(+34.42)
91.93(+27.92)
Wa-R-v2
6.78(+4.37)
7.13(+1.22)
7.93(+3.73)
18.35(+18.2)
57.43(+53.32)
56.72(+51.72)
An-M-v2
114.21(+18.09)
103.34(+6.4)
80.29(-0.83)
49.89(+45.92)
118.21(+30.11)
110.67(+23.75)
An-MR-v2
122.11(+30.56)
103.5(+24.91)
84.61(-2.13)
42.58(+39.17)
112.2(+23.98)
116.6(+28.59)
An-R-v2
77.41(+21.34)
10.77(+0.3)
22.55(-8.82)
17.37(+13.69)
49.3(+17.87)
53.06(+21.54)
Average
64.28(+12.17)
53.07(+3.31)
53.4(+15.22)
41.92(+39.32)
78.68(+35.15)
76.31(+33.51)
Table 10: Average reward for each method in MuJoCo Environments before and after online finetuning
with delayed rewards. To save space, the name of the environments and datasets are compressed,
where Ho=Hopper, Ha=HalfCheetah, Wa=Walker2d, An=Ant for the environment, and M=Medium,
MR=Medium-Replay, R=Random for the dataset. The format is "final(+increase)". The best result for
each setting is highlighted in bold font, and any result > 90% of the best performance is underlined.

0

25

50

75

100

Hopper-medium-v2

0

50

100

150

Ant-medium-v2

0

20

40

60

80

100

Walker2d-medium-v2

0

20

40

60

Halfcheetah-medium-v2

0

25

50

75

100

Hopper-medium-replay-v2

0

25

50

75

100

125

Ant-medium-replay-v2

0

25

50

75

100

Walker2d-medium-replay-v2

0

20

40

60

Halfcheetah-medium-replay-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

100

Hopper-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

Ant-random-v2

0
100000 200000 300000 400000 500000

0

20

40

60

80

Walker2d-random-v2

0
100000 200000 300000 400000 500000

Online Transitions

0

20

40

60

Normalized Reward

Halfcheetah-random-v2

TD3+BC
IQL
ODT
PDT
TD3
TD3+ODT (ours)

Figure 18: Reward curves for MuJoCo environments with delayed reward.

TD3+BC works well in several scenarios, it struggles on random environments. See Fig. 18 for
reward curves.

G.2
Maze2D Environment

We test on navigation tasks in D4RL [19] where the agents need to control a pointmass through
four different mazes: Open, Umaze, Medium and Large with different dataset respectively. Fig. 19
lists the performance of each method on Maze2D before and after online finetuning, and Tab. 11
summarizes the performance before and after online finetuning. The result shows that our method
again significantly outperforms autoregressive-based algorithms such as ODT and PDT, which
validates our motivation in Sec. 3.1. DDPG+ODT works similarly well as TD3+ODT in this
environment with simple state and action space.

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)
Open-v0
350.58(-0.22)
576.06(+24.58)
574.4(+306.56)
515.97(+485.99)
430.61(-73.36)
574.03(+96.02)
574.32(+29.57)
Umaze-v2
90.91(+63.15)
159.97(+114.77)
43.19(+6.37)
140.84(+153.47)
162.04(+130.44)
139.45(+113.4)
140.1(+107.76)
Medium-v2
96.91(+63.2)
177.61(+97.31)
26.11(+12.77)
80.36(+81.49)
184.55(+138.57)
133.03(+95.32)
136.63(+91.69)
Large-v2
132.66(+38.21)
214.02(+143.21)
20.37(+17.25)
38.74(+39.14)
232.9(+218.72)
170.09(+151.44)
189.71(+170.17)
Average
167.77(+41.09)
281.92(+94.97)
166.02(+85.74)
193.98(+190.02)
252.53(+103.59)
254.15(+114.05)
260.19(+99.8)
Table 11: Average reward for each method in Maze2D Environments before and after online finetuning.
Our method works slightly worse than IQL but better than all other baselines.

29


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

200

400

600

Maze2d-open-v0

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

50

100

150

Maze2d-umaze-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

50

100

150

200
Maze2d-medium-v1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

100

200

Normalized Reward

Maze2d-large-v1

TD3+BC
IQL
ODT
PDT
TD3
DDPG+ODT
TD3+ODT (ours)

Figure 19: Reward curve for each method in Maze2D Environments. Again, autoregressive algorithms
such as ODT and PDT does not perform well in this case.

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

50

100

Normalized Reward

Hammer-cloned-v1

T_train=20
T_train=10
T_train=5
T_train=3
T_train=2
T_train=1

Figure 20: The reward curves on hammer-cloned-v1 with different Ttrain and Teval = 1. While longer
Ttrain leads to faster convergence in this environment, runs with too long Ttrain are also unstable.

G.3
Ablations on training context length Ttrain

Fig. 20 shows the result of using different context lengths on hammer-cloned-v1 environment (in this
experiment, we use Teval = 1 to demonstrate the effect of more different Ttrain). It is shown from
the experiment that the selection of Ttrain needs to be balanced between more information taken into
account and training stability; while longer Ttrain brings faster convergence when growing from 1 to
5, the reward curves with Ttrain ∈{10, 20} oscillates more than that with Ttrain = 5.

G.4
Longer Training Process

In some environments, such as hopper-random-v2, walker2d-random-v2 and ant-random-v2, our
proposed method still seems to be improving after 500K online samples. In Fig. 21, we show the
finetuning result of our proposed method with more online transitions, which effectively shows that
our method has greater potential in online finetuning when finetuned for more gradient steps.

G.5
The Effect of Layernorm

As we have mentioned in Sec. F.2, as suggested by Yue et al. [71], we apply Layernorm [4] to critic
networks for environment other than MuJoCo for better stability in training. In our experiment, we
found that it greatly stabilizes the critic on complicated environments such as Adroit, but makes online
policy improvement less efficient on easier MuJoCo environments. Fig. 22 shows the performance
and critic MSE loss comparison on some environments with and without layernorm; it is clearly
shown that layernorm helps stabilizes online finetuning in some cases such as pen-cloned-v1, but
hinders performance increase on other environments such as halfcheetah-medium-v2.

G.6
Recurrent Critic

As mentioned in Sec. 3.2, we use reflexive critics (i.e., critics that only take the current state nand
action) to add RL gradients to decision transformer, and this creates an average effect among policies
generated by different context lengths (see Sec. D in the appendix for detail). In this section, we
explore recurrent critic by substituting the MLP critic using a LSTM, such that for a trajectory segment,
the evaluation for the i-th action ai is based on all state-action pairs (s1, a1), . . . , (si−1, ai−1) and
current state si. As shown in Fig. 23, we found that recurrent critics are much less stable than

30


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4

1e6

0

20

40

60

80

100

120

Hopper-random-v2

0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4

1e6

0

20

40

60

80

100

120

Walker2d-random-v2

0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Online Transitions

1e6

0

25

50

75

100

125

Normalized Reward

Ant-random-v2

TD3+ODT

Figure 21: The reward curves of our method when finetuned for more steps (we only report curves
until the black line in Fig. 6). It is clearly shown that our method has greater potential for improvement
when finetuned for more steps.

0
1
2
3
4
5
1e5

40

50

60

70

Normalized Reward

Halfcheetah-medium-replay-v2

0
1
2
3
4
5
1e5

0

200

400

600

Critic MSE Loss

Halfcheetah-medium-replay-v2

0
1
2
3
4
5
1e5

0

50

100

150

Normalized Reward

Pen-cloned-v1

0
1
2
3
4
5
Online Transitions
1e5

0

1

2

3

Critic MSE Loss

1e6
Pen-cloned-v1

No Layernorm
Layernorm

Figure 22: The reward curve and critic MSE loss comparison between runs with and without layer-
norm. Layernorm effectively stabilizes online finetuning in pen-cloned-v1, but hinders performance
increase in halfcheetah-medium-v2.

reflexive critics, and the instability increases as the training context length Ttrain grows; on the
contrary, reflexive critic can well-handle the case where Ttrain is long.

G.7
Regularizer for Pure TD3 Gradients

In the Adroit environment results discussed in Sec. 4, we found that the baseline of ODT finetuned
using pure TD3 gradients struggles due to catastrophic forgetting. Inspired by Wołczyk et al. [61],
we test whether adding a KL regularizer can fix the forgetting problem. Though our policy is
deterministic, we can approximately interpret the policy as Gaussian with a very small variance.
Thus, a KL regularizer can be simply added using c0 · ∥a −aold∥2, where a is the current action and
aold is the action predicted by the pretrained policy. We set c0 = 0.05 and test this method on the
Adroit cloned and expert dataset. We illustrate the result in Fig. 24. We find that the KL regularizer
effectively addresses the issue on expert environments for both TD3 and TD3+ODT. But it can
sometimes hinder the policy improvement of TD3+ODT with low return during online finetuning.

0
20000
40000
60000
80000
100000
Online Transitions

0

20

40

60

80

100

120

Normalized Reward

T_train = 1
T_train = 2
T_train = 20
Reflexive (T_train = 20)

Figure 23: The performance of reflexive critic vs. recurrent critic on hopper-medium-v2. It is clearly
shown that recurrent critic is much harder to train, and its performance decreases as the training
context length Ttrain grows.

31


---Page Break---
0

25

50

75

100

125

150

Pen-cloned-v1

0

20

40

60

80

100

120

140
Hammer-cloned-v1

0

10

20

30

40

50

60

Relocate-cloned-v1

0

20

40

60

80

100

Door-cloned-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

125

150

Pen-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

20

40

60

80

100

120

140

Hammer-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

20

40

60

80

100

Relocate-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

20

40

60

80

100

Normalized Reward

Door-expert-v1

ODT+JSRL(oracle)
ODT (curriculum)

ODT
TD3

TD3+KL
TD3+JSRL(pretrain)

TD3+ODT (ours)+KL
TD3+ODT (ours)

Figure 24: The result of ODT with better exploration (only in cloned dataset) and TD3/TD3+ODT
forgetting mitigation. The result shows that 1) ODT with curriculum RTG does not work; 2) even
with exploration supported by an oracle, ODT can still fail on some environments such as hammer; 3)
JSRL with the pretrained policy does not work for forgetting mitigation; 4) KL regularizer effectively
addresses the issue on expert environments for both TD3 and TD3+ODT, but it can hinder the
improvement of TD3+ODT with low return.

G.8
Other possible exploration improvement techniques

In Sec. 3.1, we state that ODT cannot explore the high-RTG region when pretrained with low-quality
offline data, and we ran a simple experiment to verify this (Fig. 2). In this section, we test two
potential alternatives for addressing the exploration problem: JSRL [59] and curriculum learning.

For JSRL, an expert policy is used for the first n steps in an episode, before ODT takes over. We
set n = 100 (100% max episode length for adroit pen, 50% max episode length for other adroit
environments) initially, and apply an exponential decay rate of 0.99 for every episode. We test two
settings of JSRL: the expert policy being the offline pretrained policy, and the expert policy being
oracle, i.e., an IQL policy trained on the Adroit expert dataset.

For curriculum learning, we use ODT with a gradually increasing target RTG with the current RTG
for rollouts being RTGeval −0.99N(RTGeval −RTGdata). Here, N is the number of episodes sampled
in online stage, and RTGdata is the average RTG of the offline dataset.

Results are summarized in Fig. 24. We found that curriculum RTG does not work, probably because
the task is too hard and cannot be improved by random exploration without gradient guidance. Further,
even with oracle exploration, ODT is not guaranteed to succeed: it fails on the hammer environment
where TD3+ODT succeeds, probably because of insufficient expert-level data and an inability to
improve with random exploration.

G.9
Ablations on the Architecture

In this section, we further examine the source of the performance gain of our method compared to
TD3+BC. There are two key differences as stated in Sec. 3.2: The architecture and RL via Supervised
(RvS) learning [18]. We can hence assess two more baselines: TD3+BC with our transformer
architecture and TD3+RvS using TD3+BC’s architecture. We present the ablation result on the Adroit
cloned environment in Fig. 25. The result shows that only TD3+BC with our architecture works
(albeit still worse than our method). We hypothesize that this is because a simple MLP is hard to
model the complicated policy which takes both RTG and state into account.

To further assess if simply adding more layers to the MLP works, we conduct an ablation on the
number of layers for TD3+RvS. The result is illustrated in Fig. 26. It shows that simply adding a
few layers to the MLP does not aid performance. We speculate that it is probably the transformer
architecture that helps modeling the state-and-RTG-conditioned policy.

32


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

125

150

Pen-cloned-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

20

40

60

80

100

120

140
Hammer-cloned-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0.0

0.5

1.0

1.5

2.0

Relocate-cloned-v1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

20

40

60

Normalized Reward

Door-cloned-v1

ODT
TD3+BC
TD3+RVS
TD3+BC (our arch)
TD3+ODT (ours)

Figure 25: The result of ODT and TD3+BC ablations (TD3+RVS, DDPG+ODT, TD3+BC with
our architecture and curriculum RTG for ODT) on Adroit environments. The result shows that only
TD3+BC with our architecture works. However, it remains worse than our method.

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

25

50

75

100

125

150

Pen-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

40

60

80

100

120

140

Hammer-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
1e6

0

20

40

60

80

100

Relocate-expert-v1

0.0
0.2
0.4
0.6
0.8
1.0
Online Transitions
1e6

0

20

40

60

80

100

Normalized Reward

Door-expert-v1

TD3+RVS (1 layer)
TD3+RVS (2 layers)
TD3+RVS (3 layers)
ODT
TD3+ODT (ours)

Figure 26: Results of adding more layers to TD3+RvS. The result shows that simply adding MLP
layers does not help TD3+RvS match the performance of ODT and TD3+ODT.

H
Computational Resources

We conduct all experiments with a single NVIDIA RTX 2080Ti GPU on an Ubuntu 18.04 server
equipped with 72 Intel Xeon Gold 6254 CPUs @ 3.10GHz. Mujoco experiments takes about 6 −8
hours, and the bottleneck is the gradient update; about 50% time is spent on backpropagation and
update of parameters. Our critic appended to ODT only takes up about 20% time to train, in which
90% of the critic training time is spent on decision transformer inference to get action for “next state”.
For the actor, the training overhead of our method is negligible since it only contains an MLP critic
inference to get the Q-value. Therefore, overall our method only uses 20% extra time compared to
ODT for training, but attains much better results.

I
Dataset and Algorithm Licenses

Our code is developed upon multiple algorithm repositories and environment testbeds.

Algorithm Repositories. We implement our method on the basis of online decision transformer
repository, which has a CC BY-NC 4.0 license. We also refer to IQL [28], PDT [64] and TD3+BC [20]
repository when running baselines, all of which have MIT licenses.

Environment Testbeds. We utilize OpenAI gym [8], MuJoCo [58], and D4RL [19] as testbed, which
have an MIT license, an Apache-2.0 license, and an Apache-2.0 license respectively.

33


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Please refer to the abstract and introduction.

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

Justification: See Sec. 6.

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

34


---Page Break---
Justification: See Appendix E.
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
Justification: See Appendix F.
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

35


---Page Break---
Answer: [Yes]

Justification: See https://github.com/KaiYan289/RL_as_Vitamin_for_Online_
Decision_Transformers for the code. The data is available in public D4RL dataset.

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

Justification: See Appendix F.

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

Justification: See Sec. 4 and Appendix G.

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

36


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

Justification: See Appendix H.

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

Justification: Our research in the paper fully conforms with the NeurIPS Code of Ethics.

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

Justification: See Appendix A.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

37


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

Justification: See Appendix I.

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

38


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification:
We attach our code in the supplementary material, and has pub-
lished it on Github (https://github.com/KaiYan289/RL_as_Vitamin_for_Online_
Decision_Transformers).
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

39


---Page Break---
