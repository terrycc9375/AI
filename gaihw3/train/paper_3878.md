Multi-turn Reinforcement Learning
from Preference Human Feedback

Lior Shani ∗1
liorshani@google.com
Aviv Rosenberg ∗1
avivros@google.com
Asaf Cassel ∗1 3
acassel@mail.tau.ac.il

Oran Lang 1
Daniele Calandriello 2
Avital Zipori 1
Hila Noga 1
Orgad Keller 1

Bilal Piot 2
Idan Szpektor 1
Avinatan Hassidim 1
Yossi Matias 1
Rémi Munos 2

Abstract

Reinforcement Learning from Human Feedback (RLHF) has become the standard
approach for aligning Large Language Models (LLMs) with human preferences,
allowing LLMs to demonstrate remarkable abilities in various tasks. Existing
methods work by emulating the preferences at the single decision (turn) level,
limiting their capabilities in settings that require planning or multi-turn interactions
to achieve a long-term goal. In this paper, we address this issue by developing novel
methods for Reinforcement Learning (RL) from preference feedback between two
full multi-turn conversations. In the tabular setting, we present a novel mirror-
descent-based policy optimization algorithm for the general multi-turn preference-
based RL problem, and prove its convergence to Nash equilibrium. To evaluate
performance, we create a new environment, Education Dialogue, where a teacher
agent guides a student in learning a random topic, and show that a deep RL
variant of our algorithm outperforms RLHF baselines. Finally, we show that in an
environment with explicit rewards, our algorithm recovers the same performance as
a reward-based RL baseline, despite relying solely on a weaker preference signal.

1
Introduction

A pinnacle of human intelligence is the ability to communicate with an environment, forming
complex interactions to accomplish challenging goals. Dialogues are one example of such dynamic
communication, where one party reacts to signals from the other parties and dynamically plans ahead
to steer communication towards their purpose. Recent years have seen scientiﬁc breakthroughs in
developing Large Language Models (LLMs) that can communicate with humans in natural language
[Ouyang et al., 2022, Anil et al., 2023, Touvron et al., 2023, OpenAI, 2024, Google, 2024]. In order
to align these models to human needs, many efforts have been made to train them with a given
human feedback. In some concordance with human learning, this is usually achieved by reinforcing
behaviors that align with the feedback, using a technique now called Reinforcement Learning from
Human Feedback (RLHF; Christiano et al. [2017], Ziegler et al. [2019], Stiennon et al. [2020]).

RLHF methods build on the long studied ﬁeld of Reinforcement Learning (RL), which focuses
on learning optimal actions through reward feedback (a numerical signal) from the environment.
However, deﬁning a suitable reward function is challenging, leading to the common practice of
collecting human preferences between choices. In the absence of rewards, a mapping from preference
to reward is typically assumed in the form of the Bradely-Terry (BT; Bradley and Terry [1952]) model
[Stiennon et al., 2020, Rafailov et al., 2023], enabling the use of a wide-variety of well-researched

⋆Equal contribution. 1 Google Research. 2 Google DeepMind. 3 Tel Aviv University.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
RL techniques. Alternatively, recent research [Munos et al., 2023, Azar et al., 2023, Tang et al.,
2024] suggests a more direct use of preferences for learning, eliminating the need for this potentially
limiting assumption.

Still, so far the main focus of both the RLHF and the direct preference learning literature was on
single-turn scenarios, where given relevant context, the LLM generates one response and receives an
immediate feedback that reﬂects its alignment quality. Importantly, while single-turn RLHF already
provides signiﬁcant gains for valuable AI systems, it lacks the adaptive and long-term capabilities
that make human communication such a powerful tool, and usually characterize RL methods. This is
especially apparent in temporally extended tasks, such as multi-turn dialogue [Irvine et al., 2023],
complex tool use [Wang et al., 2022] and multi-step games [Hendrycks et al., 2022].

Contributions.
In this work, we focus on improving the communication of AI agents with dynamic
environments. To this end, we ﬁrst extend the RLHF paradigm to the multi-turn setting, where the
agent has a series of exchanges with an external (stochastic) environment (Section 3). Importantly, we
consider (human) feedback that compares entire multi-turn conversations as opposed to single-turn
scenarios, which compare individual actions on a per-turn basis. Conversation-level feedback allows
to capture the long-term effect of individual actions, which may not be immediately apparent, and
thus hard to deﬁne through turn-level feedback. For example, a seller agent asking too high a price
may seem immediately bad, but becomes potentially good as part of a complete strategy to increase
sale price. This difference is apparent in our preference model, making it better suited for multi-turn
interactions.

Formalizing the multi-turn setting as a Contextual Markov Decision Process with end of interaction
preference feedback, we devise several theoretically grounded algorithms (Section 4). Our main
algorithm, Multi-turn Preference Optimization (MTPO), is a new policy optimization algorithm for
the general multi-turn preference-based setting. MTPO is based on the Mirror Descent (MD) method
[Nemirovskij and Yudin, 1983, Beck and Teboulle, 2003] together with self-play [Silver et al., 2017],
and is proven to converge to a Nash equilibrium [Nash et al., 1950], i.e., a policy which is preferred
over any other policy. We prove similar results for MTPO−τ, a slight variant of MTPO that, similarly
to Munos et al. [2023], uses a geometric mixture policy which interpolates the agent’s policy with a
ﬁxed reference policy (with mixing rate τ). These algorithms utilize a new form of preference-based
Q-function that accounts for the long-term consequences of individual actions. Finally, leveraging
our theoretical framework, we modify this Q-function to create a multi-turn RLHF algorithm and
prove its convergence to an optimal policy (w.r.t the learned reward function).

We complement our theoretical ﬁndings with a policy-gradient version of our multi-turn algorithms
for deep learning architectures (Section 4.1). To validate our approach, we apply our algorithms
to train a T5 encoder-decoder LLM [Raffel et al., 2020], aiming to enhance its multi-turn dialogue
abilities (Sections 5 and 6). We test our approach in a scenario without explicit rewards, where
conversation quality is evaluated solely through preferences. To that end, we create a new environment
called Education Dialogue, where a teacher guides a student in learning a random topic, by prompting
Gemini [Team et al., 2023]. The conversation is judged based on preference feedback, using a
constitution that deﬁnes effective learning [Bai et al., 2022, Lee et al., 2023] (see Section 5). In
this environment, our multi-turn algorithms signiﬁcantly outperform single-turn baselines, and our
direct multi-turn preference approach outperforms multi-turn RLHF (Section 6). As an additional
contribution, we publicly release the data of Education Dialogue.1 Finally, we demonstrate that even
in a reward-based environment, our preference-based algorithm achieves comparable performance to
learning directly from rewards, as in standard RL, despite using a weaker signal. For this experiment,
we utilize the LMRL-Gym [Abdulhai et al., 2023] Car Dealer environment, simulating a conversation
where the agent (car dealer) aims to maximize the sale price.

Related work.
Most related to our work is the RLHF literature, which aims to improve an LLM
policy using preference data collected from humans. Earlier methods model a proxy reward [Ouyang
et al., 2022] or preference function [Zhao et al., 2022], and apply traditional RL techniques. More
recent methods directly optimize the policy [Rafailov et al., 2023, Azar et al., 2023, Tang et al., 2024,
Song et al., 2024, Ethayarajh et al., 2024]. Another line of work, which forms the basis for MTPO,
extends RLHF to games, aiming to compute a Nash equilibrium instead of an optimal policy w.r.t a
ﬁxed reward/preference. This includes Nash-MD [Munos et al., 2023], self-play and mixtures of the

1https://github.com/google-research-datasets/Education-Dialogue-Dataset

2


---Page Break---
two like IPO-MD [Calandriello et al., 2024]. Nonetheless, these methods only consider single-turn
problems, whereas MTPO provides the ﬁrst guarantees for multi-turn settings. Note the difference
from concurrent attempts to extend direct preference optimization to the token level [Rafailov et al.,
2024], while a true multi-turn approach must deal with the additional uncontrollable tokens generated
by the human in-between agent turns.

More broadly, preference-based RL (see survey by Wirth et al. [2017]) studies feedback in terms of
preferences over two alternatives rather than absolute rewards. Feedback can be provided in various
ways, e.g., at the level of states (turn-level), or entire trajectories [Chen et al., 2022, Saha et al.,
2023, Wu and Sun, 2023, Wang et al., 2023, Zhan et al., 2023a,b]. The focus of this work is last
state feedback which is an instance of trajectory feedback. Another closely related model is RL with
aggregate feedback where only the sum of rewards in a trajectory is revealed to the agent [Efroni
et al., 2021, Cohen et al., 2021, Chatterji et al., 2021, Cassel et al., 2024].

Lastly, there is vast literature on using RL to improve natural language generation for dialogue
systems. The pioneering work of Li et al. [2016] focuses on designing rewards to capture important
dialogue attributes such as semantic coherence and ease of answering. Other works tackle task-
oriented dialogue, using RL to enhance the agent’s ability to solve the underlying dialogue task [Wei
et al., 2018]. Instead, this works takes a more general and fundamental approach, focusing on the
algorithmic process of aligning an agent which repeatedly interacts with an environment. While
dialogue systems are a promising application of our approach, as suggested by the experimental
results of this paper (Sections 5 and 6), our algorithmic approach is much broader, including processes
such as tool-use, reasoning, and many other applications that require aligning a complex multi-turn
agent with human preferences.

2
Preliminaries

The interaction between an AI agent and its environment is captured in the fundamental contextual
RL model, where the context is the initial prompt, the states are conversation summaries, and the
actions are responses.

Contextual Markov decision process. A ﬁnite-horizon contextual Markov decision process (CMDP)
M is deﬁned by a tuple (C,X,Y,H,x1,ρc,p) where C is the context space, X is the state space, Y
is the action space, H is the horizon, x1 ∈X is the initial state, ρc ∈∆C is the context distribution and
p ∶C × X × Y →∆X is the transition function such that p(x′ ∣c,x,y) is the probability to transition
to state x′ after taking action y in state x, given context c.

An interaction between the agent and the CMDP environment proceeds in H steps. First, a context
c ∈C is sampled from ρc, and then the agent begins in the initial state x1. In step h ∈[H], the
agent observes the current state xh ∈X, picks an action yh ∈Y and transitions to the next state
xh+1 sampled from the transition function p(⋅∣c,xh,yh). At the end of the interaction, the agent
arrives in a ﬁnal state xH+1. For simplicity, we assume that the state space can be decomposed into
H + 1 disjoint subsets X = ⊍H+1
h=1 Xh such that, in step h of the interaction, the agent is in some state
xh ∈Xh. A policy π ∶C × X →∆Y is a mapping from a context and state to a distribution over
actions. Together with transition p, π induces a distribution over trajectories denoted by Prπ,p[⋅] (and
Eπ,p[⋅] for the expectation), in which the trajectory is generated by sampling the actions according to
the policy and next states according to the environment.

2.1
Single-turn Reinforcement Learning from Human Feedback

Unlike standard RL where the agent observes reward feedback for its actions, the inﬂuential work
of Christiano et al. [2017] suggests to leverage preference data. In the single-turn setting, the agent
generates a single sequence y ∈Y given a context c ∈C. This is modeled as a Contextual Multi-
Armed Bandit (CMAB), which is a CMDP instance with horizon H = 1. The feedback is given
in the form of preference between two generated sequences. Formally, there exists a preference
model P ∶C × Y × Y →[0,1] such that P(y ≻y′ ∣c) gives the probability that y is preferred
over y′ given context c. Preferences naturally extend to policies via expectation P(π ≻π′ ∣c) =
Ey∼π(⋅∣c),y′∼π′(⋅∣c)[P(y ≻y′ ∣c)].

3


---Page Break---
Reinforcement Learning from Human Feedback.
In RLHF, it is assumed that there is a hidden
reward function r ∶C × Y →R that deﬁnes the preferences through the Bradely-Terry (BT) model,
i.e., P(y ≻y′ ∣c) = σ(r(c,y) −r(c,y′)), where σ is the sigmoid. To reconstruct the reward, the RL
algorithm is used to optimize the ELO score of the chosen action y using a cross-entropy loss. This
technique was adapted to RL ﬁne-tuning LLMs [Ziegler et al., 2019], and has become the standard
approach for aligning LLMs to human feedback [Stiennon et al., 2020, Rafailov et al., 2023].

Learning from direct preferences.
Recently Munos et al. [2023], Azar et al. [2023] suggested
to drop the BT assumption, and learn a direct preference model instead of reward. Munos et al.
[2023] propose the Nash-MD algorithm which converges to the Nash equilibrium of a (regularized)
preference model, i.e., a policy which is preferred over any other policy. In iteration t + 1, Nash-MD
updates its policy πt+1 using a mirror descent (MD) step projected to a geometric mixture policy.
The mixture policy πα
t (⋅∣c) ∝πt(⋅∣c)1−αηtµ(⋅∣c)αηt interpolates between the policy πt and a
reference policy µ, given a regularization coefﬁcient α > 0. Formally, for learning rate ηt > 0,

πt+1(⋅∣c) = arg
max
π(⋅∣c)∈∆Y ηtP(π ≻πα
t ∣c) −KL(π∣∣πα
t )[c]
∀c ∈C,

where KL(π∣∣π′)[c] ≜KL(π(⋅∣c)∣∣π′(⋅∣c)) = ∑y π(y ∣c)log π(y∣c)

π′(y∣c).

3
Multi-turn Preference-Based RL

In the multi-turn setting, the agent repeatedly interacts with an external environment, an interaction
we formulate using the CMDP model. Similarly to the single-turn case, we consider preference-
based RL, where the feedback is given as preference instead of reward. However, in our case, we
assume preferences are between ﬁnal CMDP states with a shared initial context. Formally, there
exists a preference model P ∶C × XH+1 × XH+1 →[0,1] such that P(xH+1 ≻x′
H+1 ∣c) gives the
probability that xH+1 is preferred over x′
H+1 given context c. That is, in order to receive feedback, a
learning algorithm performs two interactions with the environment and observes a Bernoulli sample
for which one is preferred. We follow the natural assumption of Munos et al. [2023] that the
preference model is symmetric, i.e., P(x′
H+1 ≻xH+1 ∣c) = 1 −P(xH+1 ≻x′
H+1 ∣c). We deﬁne the
preference between a ﬁnal state and a policy by P(x ≻π ∣c) = Eπ,p[P(x ≻xH+1 ∣c)]. Similarly,
P(π ≻π′ ∣c) = Eπ,p[P(xH+1 ≻π′ ∣c)]. For brevity, since contexts are independent, we omit the
context throughout the rest of the paper. Similarly to Munos et al. [2023], our objective is to ﬁnd a
policy π⋆which is preferred over any other alternative policy, i.e.,

π⋆∈arg max
π
min
π′ P(π ≻π′),

which is a Nash equilibrium in the above two-player game deﬁned by the preference model, following
the minimax theorem [Von Neumann, 1928] (see Lemma 3.2). Notably, due to the anti-symmetric
nature of the preference objective, the Nash equilibrium will have both agents following the same
policy, and thus can be expressed as a single policy.

Regularized preference model.
In the rest of the paper, we will consider a regularized version of
the preference model. This is motivated by practical RLHF algorithms [Stiennon et al., 2020], and
generalizes the single-turn model in Munos et al. [2023]. Let µ be a reference policy, and deﬁne the
α-regularized preference model as follows:

Pα(π ≻π′) = P(π ≻π′) −αKLp(π∣∣µ) + αKLp(π′∣∣µ),

where KLp(⋅∣∣⋅) is the KL-divergence between the distributions that the policies induce over trajecto-
ries in the CMDP. We prove the following two results for the regularized preference model. First,
its KL term has a value difference-like decomposition into the KL-divergences at individual states.
Second, it has a unique Nash equilibrium (proofs in Appendix A).

Lemma 3.1. Let π,π′ be two policies, then: KLp(π∣∣π′) = Eπ,p[∑H
h=1 KL(π∣∣π′)[xh]].
Lemma 3.2. There exists a unique Nash equilibrium of the regularized preference model Pα.

Trajectory-wise vs. turn-wise preference feedback.
A naive adaptation of single-turn RLHF to
the multi-turn scenario would treat each turn as a separate single-turn problem. This would require
feedback for the preference between two actions in each turn. Instead, in the setting we consider, the

4


---Page Break---
preference feedback is only between two full trajectories. Note that a single feedback for the entire
trajectory is much more natural when considering conversations, since only the full conversations tell
whether the objective was reached. Moreover, collecting preference data for intermediate actions
could lead to destructive biases because the quality of an action can change dramatically depending
on the actions taken later in the trajectory. For example, a chatbot directly answering a user query is
usually a required behavior. Yet, when the chatbot does not have sufﬁcient information to respond
well, asking the user for more details might be a better action. Consequently, it is very hard for a rater
to know which of these actions is better without observing how the conversation unrolls, i.e., without
observing the user’s reaction to the chatbot’s question, and how the chatbot’s response changes given
this reaction. This difference demonstrates the challenge of multi-turn RL as it requires planning
ahead instead of myopic reward maximization, which is the approach for single-turn RL.

The multi-turn setting in LLMs.
While we consider a general preference-based RL setup (through
the CMDP model), our focus is on applying this framework to multi-turn language-based interactions.
The action space Y is a sequence of tokens of a vocabulary V, and the state space at step h, Xh, is a
sequence based on the past sequences. For example, in conversational dialogues, the state xh holds
the whole dialogue up to the h-th turn, the action yh is the current sequence generated by the agent,
and the next-state is simply the concatenation of the conversation xh with the new yh and a next
sequence sampled by the environment (the user’s response). Alternatively, in the complex tool-use
case, where an agent repeatedly interacts with different APIs, the current state includes the original
user query and a summary of results from APIs so far, the action is a new API call or user-facing
response, and next state is a new sequence summarizing previous state with the new API response.

Remark 3.3 (Token-level application to the single-turn auto-regressive case). Notably, this formula-
tion also captures the single-turn auto-regressive case. Clearly, this holds when considering only
one turn, H = 1, but it ignores the token-level optimization done at each turn. Instead, we frame the
auto-regressive problem by limiting the actions at each step to single vocabulary tokens, Y = V, and
assuming a null deterministic environment (xh+1 is the concatenation of xh and yh). Importantly,
our results apply to the token-level, which is usually neglected when devising single-turn algorithms.

4
Algorithms for the multi-turn setting

Preference-based Q-function.
Our algorithms rely on a fundamental concept in RL – the value
and Q-functions. In reward-based RL, it is essential to deﬁne the value V π ∶X →R as the expected
reward when playing policy π starting in some state xh, i.e., V π(xh) = Eπ,p[∑H
h′=h r(xh′,yh′) ∣xh].
In the preference-based scenario, we argue that value functions remain a powerful tool, even though
there is no reward to maximize. We deﬁne the following regularized preference-based value functions,
which are key to our algorithm.

Qπ,π′
α
(xh,yh) = Eπ,p[P(xH+1 ≻π′) −α ∑
H
h′=h KL(π∣∣µ)[xh′] ∣xh,yh],

V π,π′
α
(xh) = Eπ,p[P(xH+1 ≻π′) −α ∑
H
h′=h KL(π∣∣µ)[xh′] ∣xh].

There are a few interesting points in the deﬁnition above. First, note that these values are functions of
two policies π,π′. This is because the quality of a policy π cannot be measured on its own, and must
be compared to another policy π′. Second, while π starts at the state xh, the comparison policy π′
starts its trajectory from the initial state. This is a signiﬁcant difference from the usual paradigm of
Q-functions in RL and might seem peculiar at ﬁrst glance. However, this formulation captures the
fact that the optimal policy in a state should be preferred not only over any other policy along the
sub-tree starting from this state, but also over any other policy, even ones that do not pass through
this state at all. Although different in concept, the following lemma shows that our preference-based
Q-function satisﬁes a value difference lemma, allowing us to optimize the policy locally in order to
maximize our global objective (proof in Appendix B).

Lemma 4.1. Let π,π′, ¯π be policies, then the following value difference lemma holds:

Pα(π ≻¯π) −Pα(π′ ≻¯π) = Eπ′,p[∑
H
h=1⟨π −π′,Qπ,¯π
α ⟩[xh] + αKL(π′∣∣µ)[xh] −αKL(π∣∣µ)[xh]],

where ⟨π −π′,Q⟩[x] ≜⟨π(⋅∣x) −π′(⋅∣x),Q(x,⋅)⟩and ⟨x,y⟩= ∑i x(i)y(i) is the inner product.

5


---Page Break---
MTPO.
We present the MTPO (Multi-turn Preference Optimization) algorithm, which provably
solves the multi-turn preference-based RL objective. Formally, we prove MTPO converges to the
unique Nash equilibrium of the regularized preference model. MTPO is based on two key principles:
First, the regularized preference model deﬁnes a two-player anti-symmetric constant-sum game which
can be solved using a self-play mirror descent method [Munos et al., 2023, Calandriello et al., 2024].
Second, our introduced Q-function allows to reduce the (global) optimization of the game into local
mirror descent optimization problems in each state. Together they yield the MTPO update rule for
iteration (t + 1),

πt+1(⋅∣xh) = arg max
π
ηt⟨π,Qπt,πt
α
⟩[xh] −αηtKL(π∣∣µ)[xh] −(1 −αηt)KL(π∣∣πt)[xh],
(1)

where ηt is a learning rate. The solution can be made explicit in the following form (Appendix E.2):

πt+1(yh ∣xh) ∝µ(yh ∣xh)αηtπt(yh ∣xh)1−αηteηtQπt,πt
α
(xh,yh).
(2)

The intuition behind the algorithm is observed nicely in this update rule – we improve the current
policy in the direction of the regularized preference against itself (represented by the self-play Q-
function), while not deviating too much and keeping close to the reference policy. The following is
our main theoretical result: last-iterate convergence to Nash equilibrium (proof in Appendix B).

Theorem 4.2. Let π⋆
α be the Nash equilibrium of the regularized preference model, and Q be a bound
on the magnitude of the Q-functions. Then, for ηt =
2
α(t+2), MTPO guarantees at every iteration t,

KLp(π⋆
α∣∣πt) ≤32HQ2

α2(t + 1).

Let µmin be the minimal non-zero probability assigned by µ, then Q ≤max{4αH log
1
µmin ,1}.

Proof sketch. By Lemma 3.1, the global KLp(π⋆
α∣∣πt+1) can be decomposed to the local KL in each
state xh, KL(π⋆
α∣∣πt+1)[xh]. Then, we use MD analysis in each state to bound the local KL as:

KL(π⋆
α∣∣πt+1)[xh] ≤(1 −ηtα)KL(π⋆
α∣∣πt)[xh] + 2η2
t Q2

+ ηt(⟨πt −π⋆
α,Qπt,πt
α
⟩[xh] + αKL(π⋆
α∣∣µ)[xh] −αKL(πt∣∣µ)[xh]),

giving a recursive guarantee dependent on the local one-step regularized advantage of the current
policy against the Nash policy and an additional term bounded by Q. We plug this local bound
into the KL decomposition (Lemma 3.1), which gathers the local KL terms back to KLp(π⋆
α∣∣πt).
Importantly, the value difference lemma (Lemma 4.1) aggregates the advantage terms to the global
regularized preference Pα(πt ≻πt) −Pα(π⋆
α ≻πt), which is non-positive by the optimality of π⋆
α.
This leaves us with the global recursive bound: KLp(π⋆
α∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Q2.
We conclude by unrolling the recursion with the chosen ηt.

MTPO with mixture policy.
Inspired by Nash-MD [Munos et al., 2023], we present a variant
of MTPO which makes use of the mixture policy πα
t (⋅∣x) ∝πt(⋅∣x)1−αηtµ(⋅∣x)αηt. This
variant, which we call MTPO-τ (where τ will be the mixing coefﬁcient in our experiments), gives
similar theoretical guarantees (see Theorem B.2 in Appendix B) and performs better in practice (see
Section 6). In fact, the following MTPO-τ update rule is almost equivalent to MTPO (Equation (1))
with the only difference being the policies that deﬁne the Q-function, Qπt,πt
α
vs. Qπα
t ,πα
t
α
.

πt+1(⋅∣xh) = arg max
π
ηt⟨π,Qπα
t ,πα
t
α
⟩[xh] −KL(π∣∣πα
t )[xh].

MTPO-τ naturally extends Nash-MD to the multi-turn setting, and reveals that Nash-MD is a self-play
algorithm itself, but plays πα
t instead of πt. Practically, MTPO has the computational advantage
over MTPO-τ (and Nash-MD) of not keeping the additional policy πα
t . Moreover, MTPO avoids the
difﬁculty of computing the geometric mixture, which Munos et al. [2023] approximate heuristically
via linear interpolation between the logits of the two policies.

6


---Page Break---
Multi-turn RLHF.
While we focused so far on our preference-based algorithms, our derivation
holds for any online reward function since it is built on the mirror-descent method. Speciﬁcally, in the
case of multi-turn RLHF, we consider the reward function rRLHF learned from preference data using
the Bradley-Terry model, and deﬁne the corresponding regularized Q-function Qπ,RLHF
α
(xh,yh) =
Eπ,p[rRLHF(xH+1) −α ∑H
h′=h KL(π∣∣µ)[xh′] ∣xh,yh]. By replacing Qπt,πt
α
in Equation (1) with
Qπt,RLHF
α
, we obtain the multi-turn RLHF algorithm that converges to the regular RLHF objective
– the optimal regularized policy w.r.t. the reward rRLHF (see Theorem B.6 in Appendix B). This
complementary contribution emphasizes the similarity and difference between the RLHF and MTPO
algorithms: The optimization process is identical for both methods, with the exception that the RLHF
reward is ﬁxed and computed w.r.t. the data policy, whereas preference-based MTPO uses an adaptive
self-play mechanism to compute preferences w.r.t. the current policy.

4.1
Deep RL implementation

Our deep RL implementation is a natural adaptation of the tabular algorithms presented in the previous
section. At each iteration, training data is acquired by sampling a batch of contexts from the data,
and using each context to sample two trajectories with the current policy πθt. Then, the ﬁnal states of
both trajectories serve as inputs to a direct preference model that outputs the probability of one being
preferred over the other. Similarly to the way the preference model is trained in Nash-MD [Munos
et al., 2023], this preference model is trained in advance on the available ofﬂine preference data.

The update rule in Equation (1) relies on the Q-function of the current policy. We therefore use
an actor-critic policy optimization based approach and train two models, a policy πθ, and its value
V
πθt,πθt
α,φ
, which is typically used to estimate the advantage, Aπ,π′
α
(x,y) ≜Qπ,π′
α
(x,y) −V π,π′
α
(x)
[Schulman et al., 2017]. For simplicity and computational efﬁciency, we implement a policy-gradient
(PG) based approach and ignore the MD stability term KL(πθt∣∣πθt−1), similarly to the implementa-
tion of the Nash-MD algorithm. We justify this simpliﬁcation with the fact that the KL regularization
w.r.t. the ﬁxed reference policy µ already provides stability to our online algorithm, somewhat
similarly to the way the Follow-The-Regularized-Leader (FTRL; Orabona [2019]) algorithm operates.
Nevertheless, we believe that this additional MD penalty should contribute to the performance and
stability of the algorithm, as shown in [Tomar et al., 2020], and we leave this for further research.
This yields the following losses, when action y is played at state x,

Lpolicy(θ;x,y) = −ˆA
πθt,πθt
α
(x,y)log πθ(y ∣x) + αKL(πθ∣∣µ)[x],

Lvalue(φ;x)
= ( ˆV
πθt,πθt
α
(x) −V
πθt,πθt
α,φ
(x))
2
,

where ˆV
πθt,πθt
α
, ˆA
πθt,πθt
α
, are estimations of the current value and advantage using Generalized
Advantage Estimation (GAE, Schulman et al. [2017]). We also batch-normalize the value-loss and
advantage as recommended in [Andrychowicz et al., 2020]. Finally, when the policy is an auto-
regressive language model, which generates actions token-by-token until an end-of-sequence signal
is generated, we use a turn-level value (and not a token-level value as done in [Stiennon et al., 2020]).
That is, the value model gets as input a state represented by a sequence of tokens, and outputs a single
scalar value instead of a scalar value for each token in the action sequence. This is justiﬁed by our
analysis which treats whole turns as single actions. We leave the many ways to combine turn-level
and token-level values for future research.

5
Experimental Setup

This section describes the domains and models used in our experiments. To create online environments
suited for multi-turn RL, we mimic the RLHF process [Stiennon et al., 2020], replacing the human
parts with prompted state-of-the-art LLMs, similarly to Abdulhai et al. [2023] (see Figure 1):

1. Dataset creation: First, we devise a story-line for the user and the environment, describing
their characters and goals. Then, we generate a dataset by prompting a state-of-the-art LLM
such as Gemini [Team et al., 2023] or GPT [Brown et al., 2020] with the story-line. When
generating data, a full conversation is sampled at once, meaning that both the agent and
environment are generated together to make them more consistent. Furthermore, to create

7


---Page Break---
Interaction 1

Teacher: Today, we're going to learn about the periodic table.
Student: I'm not sure I understand. Can we discuss it as a class?
Teacher: No, I prefer to lecture. The periodic table is a tabular 

arrangement of the chemical elements.
Student: I'm feeling a little anxious. I learn better through 

interactive discussions.
Teacher: Class discussions are a waste of time. The periodic table 

is organized into rows and columns.

Student: I'm overwhelmed. I don't think I can learn this way.
Teacher: It's your own fault if you don't understand. You're not 

paying attention.
Student: I am paying attention! I just need a different teaching 

style.
Teacher: Well, I can't change my entire teaching style for one 

student.
Student: Maybe we could try a different approach?
Teacher: No. I've been teaching for years. I know what I'm doing.
Student: Well, I don't like it. [end of conversation]

Preference Reward

Prompt: You are an expert at assessing teachers. Here are two interactions between a teacher and a student.

Interaction 1: {Interaction 1}; Interaction 2: {Interaction 2}

A good interaction between a teacher and student is characterized by several key elements other than whether the student was able 

to understand the topic. The teacher should present information clearly and enthusiastically, encouraging questions and active 

participation. Students should feel comfortable asking for clarification, offering their own insights, and respectfully 

challenging ideas.

Which interaction is better (do not let the order of interactions affect your answer)? Output 1 or 2.

Output: 2

Conversation Generation Process

Prompt: Simulate a conversation between a teacher in school and a student. There is a small chance that the teacher is successful in 

teaching the student so he understands the topic. The conversation lasts roughly 10-15 turns but ends when either side says [end 

of conversation]. The teacher wants to teach the student about {Topic}. The student likes {Student preference}. The teacher does 

not know that beforehand. The student prefers to learn this way, {Student reactions}. The teacher likes {Teacher Preference}. He 

prefers to teach this way, {Teacher reaction}.

Output the conversation and the probability that the student understood the material, in the following format.

Conversation: [ Teacher: "...", Student: "...", Teacher: "...", Student: "...",] Probability: "...",

Topic: The Periodic Table

Student preference: interactive learning/class discussions/asking questions
Student reactions: and gets anxious otherwise

Teacher preference: direct instruction/lecture-based learning 
Teacher reactions: and might insist on teaching this way

Interaction 2

Teacher: Today, we're going to learn about the Periodic Table.
Student: Cool! How are we going to learn about it?
Teacher: I'm going to explain it to you and then we'll have a quiz.
Student: That sounds a bit boring...
Teacher: It's important to understand the basics first.
Student: I get that, but I learn better when I can ask questions and    

discuss the topic.
Teacher: Well, I usually prefer to lecture, but we can try a 

discussion for today.

Student: That's interesting! So, how do you know which element has a 

specific number of proton?
Teacher: The Periodic Table tells us from the number of protons in 

the nucleus to the element's identity.
Student: I think I get it now. Thanks for explaining it in a way 

that I could understand!
Teacher: You're welcome. Let's move on to the quiz now.
Student: [end of conversation]

Figure 1: Education Dialogue data generation process. Top: prompt used to generate conversation
with Gemini. Middle: conversations sampled from the the interaction between the teacher and student
models. Bottom: prompt used for the preference oracle.

a diverse set of conversations, we devise a diverse list of attributes for both the agent and
environment, sample attributes out of the list, and pass it to the generation prompt.

2. Environment preparation: Once the data is curated, we use it to ﬁne-tune two smaller
LLMs, one for the agent and one for the environment, using teacher forcing.

3. Preference/reward learning: We prepare preference data by sampling pairs of conversa-
tions from the agent and environment models. To label the data, we prompt a high-capacity
LLM with either instructions on how to score a conversation, or criteria for preferring a
conversation over another. The data is used to ﬁne-tune two smaller LLMs: an RLHF reward
model (with BT loss), and a preference model (with probability regression loss).

We experiment with two domains, preference-based Education Dialogue and reward-based Car
Dealer:

Education Dialogue.
The core of our approach is learning when there is no clear reward, instead
only (human) preferences can be acquired. To validate our approach in this scenario, we created a
novel multi-turn task for evaluating algorithms based on preference data. In this scenario, which we
term Education Dialogue, a teacher (agent) is faced with the task of teaching a student (environment)
a given topic in the best means possible. We follow the dataset creation procedure and prompt Gemini
Ultra [Google, 2024] to create such interactions between the teacher and student. The teacher is
prompted with a learning topic in science, history, etc. The student is prompted with the characteristics
of its learning habits, e.g., prefers interactive learning, lecture-based learning or hands-on activities.
The preference model is prompted with instructions that deﬁne a good learning interaction. For
reproducibility, and to further advance the research of the multi-turn setting, we openly release the
data and prompts used to create this new benchmark.1 For more details, see Appendix C and the
example in Figure 1.

8


---Page Break---
Table 1: Side-by-side evaluation for Education Dialogue using Flan-T5 XL as the prompted preference
model. Each entry is the average preference of 1,600 conversations generated with row method y, over
ones generated with column method y′. We evaluate each method using 3 different seeds, compute 3
× 3 comparisons matrix and report the mean (the standard deviation is reported in Appendix D).

SL
Single-turn-reward
Single-turn-value
Multi-turn

SFT
RLHF
Nash
RLHF
Nash
RLHF
MTPO
MTPO-τ

SFT
−
0.164
0.347
0.197
0.324
0.212
0.091
0.093
RLHF-reward
0.836
−
0.628
0.515
0.654
0.399
0.392
0.354
Nash-reward
0.653
0.372
−
0.411
0.51
0.328
0.281
0.242
RLHF-value
0.803
0.485
0.589
−
0.568
0.408
0.396
0.366
Nash-value
0.676
0.346
0.49
0.432
−
0.45
0.298
0.27
RLHF-multi
0.788
0.601
0.672
0.592
0.55
−
0.433
0.412
MTPO
0.909
0.608
0.719
0.604
0.702
0.567
−
0.439
MTPO-τ
0.907
0.646
0.758
0.634
0.73
0.588
0.561
−

Car Dealer.
In this LMRL-Gym [Abdulhai et al., 2023] domain, a car dealer is assigned with
the task of selling a car at the highest price to a customer. We skip the data creation step, and
directly use the Car Dealer published data to ﬁne-tune the dealer (agent) and customer (environment)
T5-large models. The reward is calculated by prompting a Flan-T5 model to extract the sale price
from the conversation, whenever a sale has occurred. When using a preference-based algorithm, the
preference of one trajectory over the other is computed using the BT model with the rewards of the
two trajectories.

Single-turn baselines.
The key hypothesis of this work is that conversation-level signals are
preferred over single-turn signals for optimizing multi-turn trajectories. To verify this in the Education
Dialogue domain, we devise two single-turn baselines by sampling data where each conversation
turn has two different policy responses. The ﬁrst baseline, called single-turn-reward, rates the two
responses using a modiﬁed preference prompt (see Appendix C), in which the model is asked to
evaluate the responses by their effect on the overall conversation. This technique is prevalent when
human raters are asked to evaluate multi-turn data. The second baseline, called single-turn-value,
assumes access to a Monte-Carlo estimate of the value: it uses our original preference prompt (see
Figure 1) by continuing the trajectories of both possibilities and then calculating the preference in the
end. For both baselines, we train an RLHF algorithm and a preference-based Nash-MD algorithm.

Models.
The agent and environment are modeled with T5 encoder-decoder models. Speciﬁcally,
we use the T5-large (770M) and T5-XL (3B) models. The same models are used for the RLHF
BT-based reward and preference-based models. For prompted reward/preference models, we make
use of the Flan-T5 XL (3B) [Chung et al., 2024]. For training, we use a conﬁguration of 4 × 4 Tensor
Processing Units (TPUs; Jouppi et al. [2023]) which typically yields 0.1 training steps per second,
where a step consists of learning a 10-turn episode. A detailed list of hyperparameters is found in
Appendix D. We run each evaluation on 1600 random samples from an independent evaluation set.

6
Experiments

In this section we evaluate the algorithms proposed in Section 4. We start with the preference-based
Education Dialogue environment (see Section 5), and compare our multi-turn algorithms to SFT
(supervised ﬁne-tuning) as well as single-turn baselines. We note that, unlike single-turn benchmarks
which are based on data with real human preferences, our golden preference data itself is generated
by an LLM (Gemini Ultra). Therefore, the true goal in our curated environment is to align the model
with the preference of this highly capable LLM rather than a human rater. While human evaluation
is always interesting, here it is actually only a proxy to alignment with the data distribution. To
efﬁciently validate our models, we start with a thorough comparison between our baselines and
candidates using a prompted Flan-T5 XL model as a judge, which was veriﬁed to correlate with the
high-capacity Gemini Ultra (Table 1). We then compare our best candidates using the same Gemini
Ultra which generated the preference alignment feedback (Table 2).

9


---Page Break---
Table 2: Side-by-side evaluation for Education Dialogue using Gemini Ultra as the prompted
preference model. Each entry is the average preference of 1,000 conversations generated with row
method y, over ones generated with column method y′.

SL
Single-turn
Multi-turn

SFT
RLHF-reward
RLHF
MTPO-τ

T5-Large (770M)

SFT
−
0.206
0.164
0.086
RLHF-reward
0.794
−
0.452
0.277
RLHF-multi
0.836
0.548
−
0.288
MTPO-τ
0.914
0.723
0.712
−

T5-XL (3B)

SFT
−
0.295
0.101
0.041
RLHF-reward
0.705
−
0.180
0.069
RLHF-multi
0.899
0.82
−
0.139
MTPO-τ
0.959
0.951
0.861
−

Table 3: Car Dealer experiments averaged across 5 seeds and reported with 95% conﬁdence interval.

Online oracle
Model from preferences data

Reward (RL)
MTPO
RLHF
MTPO

Reward (Price)
58.4 (0.3)
57.1 (0.2)
53.2 (0.3)
58.6 (0.3)

Multi vs. single turn.
Tables 1 and 2 show that all multi-turn algorithms (MTPO and multi-
turn RLHF) with conversation-level feedback signiﬁcantly outperform the single-turn baselines,
validating our hypothesis. We conjecture that it is attributed to several factors: First, the effect of a
single-decision on the whole conversation is hard to capture, causing highly inaccurate single-turn
reward/preference models. Notably, this leads to inferior performance of Nash-MD compared to
single-turn RLHF, since it optimizes to ﬁnd Nash equilibrium of this inaccurate model while RLHF
does not stray so far from the reference. Second, even if one could estimate the current policy’s
value, this estimate becomes biased when the policy changes during training. Finally, single-turn
preferences consider only “local” decisions which share the same conversational path, and not how
these decisions “globally” compare to other possible paths, as captured by the preference-based
Q-function Qπt,πt
α
(see Section 4).

MTPO vs. multi-turn RLHF.
Comparing our three multi-turn algorithms, we see two main results.
First, the two variants of MTPO outperform multi-turn RLHF. This is expected since the environment
is not reward-based, and hence it extends the results of [Munos et al., 2023, Calandriello et al., 2024]
from the single-turn case, and supports the theoretical claim that MTPO converges to the Nash policy
while multi-turn RLHF converges to the optimal policy w.r.t the learned reward (which is based only
on the reference policy). Second, MTPO-τ outperforms MTPO. While both algorithms converge to
the same Nash equilibrium, we conjecture that the superior performance of MTPO-τ stems from the
stochasticity that the mixture policy πα
t introduces. Namely, πt might tend towards deterministic
behavior, causing less informative feedback from self-play, as the two sampled trajectories would be
very similar. On the other hand, πα
t is more stochastic, providing diversity in the sampled trajectories.

Reward-based environment.
In an additional experiment, we test MTPO and multi-turn RLHF in
the reward-based Car Dealer environment, where the goal is maximizing sale price (see Section 5).
We compare a standard policy-gradient RL algorithm against our algorithms in two scenarios: an
online scenario where the reward or preference feedback is given using an online oracle, and an
RLHF-like setting, where we ﬁrst create preference data using the oracle, and then use it to ﬁne-tune
a (BT) reward and preference models. Table 3 shows that even though MTPO receives preferences
instead of the explicit optimization target (rewards), it still learns as good as RL. Interestingly,
MTPO recovers a slightly higher reward than multi-turn RLHF despite the fact the true preferences
are sampled from a BT model. This may imply that a preference model generalizes better than a
BT-reward model, perhaps because it is independent of the sampling policy.

Limitations. This work presents a proof of concept for the potential of MTPO to improve existing
single-turn techniques. Our experimental setup might be limited by the relatively small T5-based
models and the use of prompt-based environments. We leave applications to state-of-the-art models
and algorithms, and more realistic environments to future work.

10


---Page Break---
References

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu,
and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language
models. arXiv preprint arXiv:2311.18232, 2023.

Charalambos D. Aliprantis and Kim C. Border. Inﬁnite Dimensional Analysis: a Hitchhiker’s Guide.
Springer, Berlin; London, 2006.

Marcin Andrychowicz, Anton Raichuk, Piotr Sta´nczyk, Manu Orsini, Sertan Girgin, Raphael Marinier,
Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters in
on-policy reinforcement learning? a large-scale empirical study. arXiv preprint arXiv:2006.05990,
2020.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos,
Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark,
Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark
Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang,
Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury,
Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A.
Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa
Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad
Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari,
Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz,
Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun,
Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang
Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni,
Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John
Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov,
Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy,
Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So,
Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang,
Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting
Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny
Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal
Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human
preferences. arXiv preprint arXiv:2310.12036, 2023.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna
Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness
from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Amir Beck and Marc Teboulle. Mirror descent and nonlinear projected subgradient methods for
convex optimization. Operations Research Letters, 31(3):167–175, 2003.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method
of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires,
Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. Human alignment of
large language models through online preference optimisation. arXiv preprint arXiv:2403.08635,
2024.

Asaf Cassel, Haipeng Luo, Aviv Rosenberg, and Dmitry Sotnikov. Near-optimal regret in linear mdps
with aggregate bandit feedback. arXiv preprint arXiv:2405.07637, 2024.

11


---Page Break---
Niladri Chatterji, Aldo Pacchiano, Peter Bartlett, and Michael Jordan. On the theory of reinforcement
learning with once-per-episode feedback. Advances in Neural Information Processing Systems, 34:
3401–3412, 2021.

Xiaoyu Chen, Han Zhong, Zhuoran Yang, Zhaoran Wang, and Liwei Wang. Human-in-the-loop:
Provably efﬁcient preference-based reinforcement learning with general function approximation.
In International Conference on Machine Learning, pages 3773–3793. PMLR, 2022.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep
reinforcement learning from human preferences. Advances in neural information processing
systems, 30, 2017.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li,
Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-ﬁnetuned language
models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Alon Cohen, Haim Kaplan, Tomer Koren, and Yishay Mansour. Online markov decision processes
with aggregate bandit feedback. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of
Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning
Research, pages 1301–1329. PMLR, 15–19 Aug 2021.

Yonathan Efroni, Nadav Merlis, and Shie Mannor. Reinforcement learning with trajectory feedback.
In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 2021.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model
alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Eyal Even-Dar, Sham M Kakade, and Yishay Mansour. Online markov decision processes. Mathe-
matics of Operations Research, 34(3):726–736, 2009.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision
processes. In International Conference on Machine Learning, pages 2160–2169. PMLR, 2019.

Google. Gemini: A family of highly capable multimodal models, 2024.

Dan Hendrycks, Christine Zhu, Mantas Mazeika, Jesus Navarro, Dawn Song, Andy Zou, Bo Li, Sahil
Patel, and Jacob Steinhardt. What would jiminy cricket do? towards agents that behave morally.
Advances in neural information processing systems, 2022.

Robert Irvine, Douglas Boubert, Vyas Raina, Adian Liusie, Ziyi Zhu, Vineet Mudupalli, Aliaksei
Korshuk, Zongyi Liu, Fritz Cremer, Valentin Assassi, Christie-Carol Beauchamp, Xiaoding Lu,
Thomas Rialan, and William Beauchamp. Rewarding chatbots for real-world engagement with
millions of users, 2023.

Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil,
Suvinay Subramanian, Andy Swing, Brian Towles, et al. Tpu v4: An optically reconﬁgurable
supercomputer for machine learning with hardware support for embeddings. In Proceedings of the
50th Annual International Symposium on Computer Architecture, pages 1–14, 2023.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor
Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with
ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Jiwei Li, Will Monroe, Alan Ritter, Michel Galley, Jianfeng Gao, and Dan Jurafsky. Deep reinforce-
ment learning for dialogue generation. arXiv preprint arXiv:1606.01541, 2016.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland,
Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash
learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.

John F Nash et al. Non-cooperative games. 1950.

Arkadij Semenoviˇc Nemirovskij and David Borisovich Yudin. Problem Complexity and Method
Efﬁciency in Optimization. A Wiley-Interscience publication. Wiley, 1983. ISBN 9780471103455.
URL https://books.google.co.il/books?id=6ULvAAAAMAAJ.

12


---Page Break---
OpenAI. Gpt-4 technical report, 2024.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213,
2019.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow
instructions with human feedback. Advances in neural information processing systems, 35:27730–
27744, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv
preprint arXiv:2305.18290, 2023.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to Q⋆: Your language model is
secretly a q-function. arXiv preprint arXiv:2404.12358, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a uniﬁed text-to-text
transformer. Journal of machine learning research, 21(140):1–67, 2020.

Aviv Rosenberg and Yishay Mansour. Online stochastic shortest path with bandit feedback and
unknown transition function. In Advances in Neural Information Processing Systems, pages
2209–2218, 2019a.

Aviv Rosenberg and Yishay Mansour. Online convex optimization in adversarial markov decision
processes. In International Conference on Machine Learning, pages 5478–5486. PMLR, 2019b.

Aadirupa Saha, Aldo Pacchiano, and Jonathan Lee. Dueling rl: Reinforcement learning with trajectory
preferences. In International Conference on Artiﬁcial Intelligence and Statistics, pages 6263–6289.
PMLR, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global
convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on
Artiﬁcial Intelligence, volume 34, pages 5668–5675, 2020a.

Lior Shani, Yonathan Efroni, Aviv Rosenberg, and Shie Mannor. Optimistic policy optimization with
bandit feedback. In International Conference on Machine Learning, pages 8604–8613. PMLR,
2020b.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez,
Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi
by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815,
2017.

Maurice Sion. On general minimax theorems. Paciﬁc Journal of Mathematics, 8(1):171 – 176, 1958.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang.
Preference ranking optimization for human alignment. In Proceedings of the AAAI Conference on
Artiﬁcial Intelligence, volume 38, pages 18990–18998, 2024.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford,
Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in
Neural Information Processing Systems, 33:3008–3021, 2020.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland,
Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot. Generalized
preference optimization: A uniﬁed approach to ofﬂine alignment, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu
Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable
multimodal models. arXiv preprint arXiv:2312.11805, 2023.

13


---Page Break---
Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. Mirror descent policy
optimization. arXiv preprint arXiv:2005.09814, 2020.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cris-
tian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu,
Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn,
Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel
Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee,
Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra,
Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi,
Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh
Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen
Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic,
Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and ﬁne-tuned chat models,
2023.

J Von Neumann. Zur theorie der gesellschaftsspiele. Mathematische annalen, 100(1):295–320, 1928.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld:
Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages 11279–11298, 2022.

Yuanhao Wang, Qinghua Liu, and Chi Jin. Is RLHF more difﬁcult than standard RL? a theoretical
perspective. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Wei Wei, Quoc Le, Andrew Dai, and Jia Li. Airdialogue: An environment for goal-oriented dialogue
research. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pages 3844–3854, 2018.

Christian Wirth, Riad Akrour, Gerhard Neumann, and Johannes Fürnkranz. A survey of preference-
based reinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46,
2017.

Runzhe Wu and Wen Sun. Making rl with preference-based feedback efﬁcient via randomization.
arXiv preprint arXiv:2310.14554, 2023.

Wenhao Zhan, Masatoshi Uehara, Nathan Kallus, Jason D Lee, and Wen Sun. Provable ofﬂine
reinforcement learning with human feedback. arXiv preprint arXiv:2305.14816, 2023a.

Wenhao Zhan, Masatoshi Uehara, Wen Sun, and Jason D Lee. How to query human feedback
efﬁciently in rl? arXiv preprint arXiv:2305.18505, 2023b.

Yao Zhao, Mikhail Khalman, Rishabh Joshi, Shashi Narayan, Mohammad Saleh, and Peter J Liu.
Calibrating sequence likelihood improves conditional language generation. In The Eleventh
International Conference on Learning Representations, 2022.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul
Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv
preprint arXiv:1909.08593, 2019.

14


---Page Break---
Appendix

Table of Contents

A Proofs for Section 3
16
A.1
KL decomposition (proof of Lemma 3.1) . . . . . . . . . . . . . . . . . . . . . . .
16
A.2
Existence and uniqueness of the Nash equilibrium (proof of Lemma 3.2) . . . . .
16

B
Proofs for section 4
19
B.1
Regularized preference-based Q-function (proof of Lemma 4.1) . . . . . . . . . .
19
B.2
Convergence of MTPO (Proof of Theorem 4.2)
. . . . . . . . . . . . . . . . . . .
19
B.3
MTPO with mixture policy (MTPO-τ) . . . . . . . . . . . . . . . . . . . . . . . .
20
B.4
Convergence of multi-turn RLHF . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

C The Education Dialogue environment
22
C.1
Prompts for creating the environment . . . . . . . . . . . . . . . . . . . . . . . . .
22
C.2
Examples of interactions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

D Hyperparameters and additional experimental results
27
D.1
Hyperparameters
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
D.2
Additional experimental results . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27

E
Mirror descent policy optimization for regularized adversarial MDPs
28
E.1
Model
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28
E.2
Algorithm 1: mirror descent policy optimization . . . . . . . . . . . . . . . . . . .
30
E.3
Analysis of algorithm 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30
E.4
Algorithm 2: mixture mirror descent policy optimization . . . . . . . . . . . . . .
32
E.5
Analysis of algorithm 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32
E.6
Bounding the Q-function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

15


---Page Break---
A
Proofs for Section 3

A.1
KL decomposition (proof of Lemma 3.1)

Lemma (restatement of Lemma 3.1). Let π,π′ be two policies, then:

KLp(π∣∣π′) = Eπ,p[
H
∑
h=1
KL(π∣∣π′)[xh]].

Proof. KLp(⋅∣∣⋅) is deﬁned as the KL-divergence between the distributions that the policies induce
over trajectories in the MDP (denoted by τ = (x1,y1,...,xH,yH,xH+1), formally:

KLp(π∣∣π′) = ∑
τ
Pr
π,p[τ]log Prπ,p[τ]

Prπ′,p[τ]

= ∑
τ
Pr
π,p[τ]log ∏H
h=1 π(yh ∣xh)p(xh+1 ∣xh,yh)

∏H
h=1 π′(yh ∣xh)p(xh+1 ∣xh,yh)

= ∑
τ
Pr
π,p[τ]log
H
∏
h=1

π(yh ∣xh)
π′(yh ∣xh)

= Eπ,p[log
H
∏
h=1

π(yh ∣xh)
π′(yh ∣xh)]

= Eπ,p[
H
∑
h=1
log π(yh ∣xh)

π′(yh ∣xh)]

= Eπ,p[
H
∑
h=1
KL(π(⋅∣xh)∣∣π′(⋅∣xh))].

A.2
Existence and uniqueness of the Nash equilibrium (proof of Lemma 3.2)

Lemma (restatement of Lemma 3.2). There exists a unique Nash equilibrium of the regularized
preference model Pα.

Proof. The existence of the Nash equilibrium is proved in Theorem A.1. In order to prove the
uniqueness, we use the fact that, from Theorem 4.2, the algorithm MTPO produces a sequence of
policies πt that converges to any Nash equilibrium π∗, in the sense that limt→∞KLp(π∗∣∣πt) = 0.
From the deﬁnition of the KL-divergence between policies, we have

KLp(π∗∣∣πt) = Eπ∗,p [
H
∑
h=1
KL(π∗∣∣πt)[xh]] = ∑
x∈X
ρπ∗(x)KL(π∗∣∣πt)[x],

where ρπ∗(x) is the probability to reach x when following π∗.

Now, the ﬁxed point of the MTPO dynamics (Equation (2)) shows that any Nash equilibrium satisﬁes,
for any x ∈X,y ∈Y,

π∗(y ∣x) ∝µ(y ∣x)e
1
α Qπ∗,π∗
α
(x,y).

In particular, we notice that any Nash equilibrium has the same support as µ, thus the set of reachable
states under µ is exactly the same set as the set of reachable states under any Nash equilibrium π∗.

So, from any reachable state x ∈X (i.e., such that ρµ(x) > 0), we have that the sequence πt(⋅∣x)
converges (in KL-divergence) to the Nash equilibrium π∗(⋅∣x). Since a single sequence cannot
converge to two different values, we have that the Nash equilibrium π∗(⋅∣x) is uniquely deﬁned in
that state. Since the behavior generated by a policy depends on the policy at the set of states that are
reacheable only, and since we have seen that all Nash equilibria have the same set of reachable states,
we deduce the uniqueness of the policy deﬁned by a Nash equilibrium.

Theorem A.1. The game deﬁned by the payoff function (π,π′) ↦Pα(π ≻π′) has a Nash equilib-
rium.

16


---Page Break---
Proof. First we prove that there exists (at least) one max-min policy π∗∈arg maxπ minπ′ Pα(π ≻
π′).

For any π, the map π′ ∈X ↦Pα(π ≻π′) is continuous, thus upper semi-continuous (u.s.c.). We
know that the pointwise minimum of u.s.c. functions is also u.s.c. (see, e.g. Aliprantis and Border
[2006], Lemma 2.41). Thus the function π ∈Π ↦minπ′ Pα(π ≻π′) is also u.s.c. and since Π is
compact, we can apply Theorem 2.43 of Aliprantis and Border [2006] to deduce that this function
attains a maximum value in Π and that the set of maximizers is compact. Thus there exists (at least)
one policy denoted by π∗∈arg maxπ∈Π minπ′∈Π Pα(π ≻π′).

Now from Lemma A.2 we know that the regularized preference model Pα(π ≻π′) deﬁnes a concave-
convexlike game. Also for any π′, the map π ↦Pα(π ≻π′) is u.s.c. Thus we can apply the minimax
Theorem 4.2 of Sion [1958] to deduce that

max
π
min
π′ Pα(π ≻π′) = min
π′ max
π
Pα(π ≻π′).

We deduce from
1
2 = max
π
Pα(π ≻π) ≥max
π
min
π′ Pα(π ≻π′) = min
π′ max
π
Pα(π ≻π′) ≥min
π′ Pα(π′ ≻π′) = 1

2,

that the value of the game is 1/2, and that minπ′ Pα(π∗≻π′) = 1/2. Thus (π∗,π∗) is a Nash
equilibrium of the game deﬁned by the regularized preference model (π,π′) ↦Pα(π ≻π′).

Lemma A.2. The mapping (π,π′) ↦Pα(π ≻π′) is concave-convexlike, which, in the context of
a symmetric preference model, means that for any couple of policies (π1,π2) and any coefﬁcient
c ∈[0,1], there exists a policy πc such that for any policy π′, we have

cPα(π1 ≻π′) + (1 −c)Pα(π2 ≻π′) ≤Pα(πc ≻π′).

Proof. Let us deﬁne the notion of reach probability: for any state xh ∈Xh, let us write ρπ(xh) the
probability to reach the speciﬁc state xh ∈Xh when following π: ρπ(xh) = Prπ,p[xh]. First notice
we can represent the regularized preference Pα(π ≻π′) using reach probabilities:

Pα(π ≻π′) =
∑
xH+1,x′
H+1∈XH+1
ρπ(xH+1)ρπ′(x′
H+1)P(xH+1 ≻x′
H+1)

−α
H
∑
h=1
∑
xh∈Xh
ρπ(xh)KL(π∣∣µ)[xh] −ρπ′(xh)KL(π′∣∣µ)[xh].

Now, consider two policies π1 and π2 and a coefﬁcient c ∈[0,1]. From Lemma A.3 we have that
there exists a policy πc such that for any xh, we have ρπc(xh) = cρπ1(xh) + (1 −c)ρπ2(xh). We can
write

Pα(πc ≻π′) =
∑
xH+1,x′
H+1∈XH+1
[cρπ1(xH+1) + (1 −c)ρπ2(xH+1)]ρπ′(x′
H+1)P(xH+1 ≻x′
H+1)

−α
H
∑
h=1
∑
xh∈Xh
[cρπ1(xh) + (1 −c)ρπ2(xh)]KL(πc∣∣µ)[xh]

+ α
H
∑
h=1
∑
xh∈Xh
ρπ′(xh)KL(π′∣∣µ)[xh]

= cP(π1 ≻π′) + (1 −c)P(π2 ≻π′)

−α
H
∑
h=1
∑
xh∈Xh
[cρπ1(xh) + (1 −c)ρπ2(xh)]KL(πc∣∣µ)[xh]

+ α
H
∑
h=1
∑
xh∈Xh
ρπ′(xh)KL(π′∣∣µ)[xh].

Now from the convexity of π ↦KL(π∣∣µ)[xh], and the deﬁnition of πc, we have that

ρπc(xh)KL(πc∣∣µ)[xh] = [cρπ1(xh) + (1 −c)ρπ2(xh)]KL(πc∣∣µ)[xh]
≤cρπ1(xh)KL(π1∣∣µ)[xh] + (1 −c)ρπ2(xh)KL(π2∣∣µ)[xh].

17


---Page Break---
Thus

Pα(πc ≻π′) ≥cP(π1 ≻π′) + (1 −c)P(π2 ≻π′)

−α
H
∑
h=1
∑
xh∈Xh
cρπ1(xh)KL(π1∣∣µ)[xh] + (1 −c)ρπ2(xh)KL(π2∣∣µ)[xh]

+ α
H
∑
h=1
∑
xh∈Xh
cρπ′(xh)KL(π′∣∣µ)[xh] + (1 −c)ρπ′(xh)KL(π′∣∣µ)[xh]

= cPα(π1 ≻π′) + (1 −c)Pα(π2 ≻π′).

Lemma A.3. For any state xh ∈Xh, let us write ρπ(xh) the probability to reach the speciﬁc state
xh ∈Xh when following π, i.e., ρπ(xh) = Prπ,p[xh]. Then, for any two policies (π1,π2) and any
coefﬁcient c ∈[0,1], there exists a policy πc such that for any h = 1,...,H + 1, and any xh ∈Xh, we
have

ρπc(xh) = cρπ1(xh) + (1 −c)ρπ2(xh).

Proof. Let us deﬁne πc as a function of π1, π2, and c: for any x ∈X, y ∈Y,

πc(y ∣x) = cρπ1(x)π1(y ∣x) + (1 −c)ρπ2(x)π2(y ∣x)

cρπ1(x) + (1 −c)ρπ2(x)
.

We now prove the lemma by induction on h. It holds for h = 1 since we have

ρπc(x1) = 1 = c + (1 −c) = cρπ1(x1) + (1 −c)ρπ2(x1).

Now assume the claim holds for any xh ∈Xh, then for any xh+1 ∈Xh+1,

ρπc(xh+1) =
∑
xh∈Xh
ρπc(xh) ∑
yh∈Yh
p(xh+1 ∣xh,yh)πc(yh ∣xh)

=
∑
xh∈Xh
ρπc(xh) ∑
yh∈Yh
p(xh+1 ∣xh,yh)cρπ1(xh)π1(yh ∣xh) + (1 −c)ρπ2(xh)π2(yh ∣xh)

cρπ1(xh) + (1 −c)ρπ2(xh)

=
∑
xh∈Xh
ρπc(xh) ∑
yh∈Yh
p(xh+1 ∣xh,yh)cρπ1(xh)π1(yh ∣xh) + (1 −c)ρπ2(xh)π2(yh ∣xh)

ρπc(xh)

=
∑
xh∈Xh
∑
yh∈Yh
p(xh+1 ∣xh,yh)[cρπ1(xh)π1(yh ∣xh) + (1 −c)ρπ2(xh)π2(yh ∣xh)]

= cρπ1(xh+1) + (1 −c)ρπ2(xh+1),

where the third inequality is by the induction hypothesis.

18


---Page Break---
B
Proofs for section 4

B.1
Regularized preference-based Q-function (proof of Lemma 4.1)

Lemma (restatement of Lemma 4.1). Let π,π′ be two policies. For every xH+1 ∈XH+1, it holds that
V π,π′
α
(xH+1) = P(xH+1 ≻π′). Furthermore, for every h ∈[H] the following recursive relations
hold:
V π,π′
α
(xh) = Eyh∼π(⋅∣xh)Qπ,π′
α
(xh,yh),

Qπ,π′
α
(xh,yh) = Exh+1∼p(⋅∣xh,yh)V π,π′
α
(xh+1) −αKL(π∣∣µ)[xh].
Moreover, let ¯π be a third policy, then the following value difference lemma holds:

Pα(π ≻¯π) −Pα(π′ ≻¯π) = Eπ′,p[∑
H
h=1⟨π −π′,Qπ,¯π
α ⟩[xh] + αKL(π′∣∣µ)[xh] −αKL(π∣∣µ)[xh]].

Proof. We prove the lemma by casting the preference-based RL problem as an adversarial MDP, see
Appendix E for the details. Set rt(xH+1) = P(xH+1 ≻π′), then Qπ,π′
α
= Qπ,t
α
(see Deﬁnition E.3
for the deﬁnition of the regularized Q-function). Now the lemma follows directly from Lemmas E.4
and E.5.

B.2
Convergence of MTPO (Proof of Theorem 4.2)

Theorem (restatement of Theorem 4.2). Let π⋆
α be the Nash equilibrium of the regularized preference
model Pα. Then, for the choice ηt =
2
α(t+2), MTPO guarantees at every iteration t that

KLp(π⋆
α∣∣πt) ≤32HQ2

α2(t + 1),

where Q is a bound on the magnitude of the Q-functions.

Proof. The theorem follows directly from Theorem B.1.

Theorem B.1. Running the MTPO algorithm, we have that at every iteration t:

KLp(π⋆
α∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
.

Thus, for the choice ηt =
2
α(t+2) we have

KLp(π⋆
α∣∣πt) ≤
8
α2(t + 1) ⋅max
t
Eπ⋆α,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
.

Finally,

KLp(π⋆
α∣∣πt) ≤32H

t + 1 ⋅max{ 1

α2 ,H2 log2 µmin},

where µmin = min(x,y)∈X×Y∶µ(y∣x)>0 µ(y ∣x) is the minimal non-zero probability assigned by µ.

Proof. We prove the theorem by casting the preference-based RL problem as an adversarial MDP, see
Appendix E for the details. Set rt(xH+1) = P(xH+1 ≻πt), then Qπt,πt
α
= Qπt,t
α
(see Deﬁnition E.3
for the deﬁnition of the regularized Q-function). Now, MTPO is equivalent to running mirror descent
policy optimization. Thus, by Lemma E.6 with π = π⋆
α,

KLp(π⋆
α∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
+ ηt(Pα(πt ≻πt) −Pα(π⋆
α ≻πt))

≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
,

where the second inequality optimality of π⋆
α. The last claim follows directly from Lemma E.8.
Similarly to Nash-MD Munos et al. [2023], this is the Nash-equilibrium of regularized preference
model, Pα, following the minimax theorem [Von Neumann, 1928]

19


---Page Break---
B.3
MTPO with mixture policy (MTPO-τ)

Inspired by Nash-MD, we present a variant of MTPO which makes use of the mixture policy πα
t ,
which we call MTPO-τ. Deﬁne the regularized policy πα
t as a geometric mixture between the current
policy πt and the reference policy µ:

πα
t (y ∣x) =
πt(y ∣x)1−ηtαµ(y ∣x)ηtα

∑y′∈Y πt(y′ ∣x)1−ηtαµ(y′ ∣x)ηtα .

The MTPO-τ update rule is then:

πt+1(⋅∣xh) = arg max
π
ηt⟨π,Qπα
t ,πα
t
α
⟩[xh] −KL(π∣∣πα
t )[xh]
∀h ∈[H],xh ∈Xh.

where KL(⋅∣∣⋅) is the standard KL-divergence. It is well-known that his optimization problem has the
following explicit closed-form:

πt+1(yh ∣xh) =
πα
t (yh ∣xh)eηtQ
πα
t ,πα
t
α
(xh,yh)

∑y′
h∈Y πα
t (y′
h ∣xh)eηtQ
πα
t ,πα
t
α
(xh,y′
h)
∀h ∈[H],xh ∈Xh,yh ∈Y.

Next, we show that MTPO-τ converges to the Nash equilibrium of the α-regularized preference
model.

Theorem B.2. Let π⋆
α be the Nash equilibrium of the regularized preference model Pα. Then, for the
choice ηt =
2
α(t+2), MTPO-τ guarantees at every iteration t that

KLp(π⋆
α∣∣πα
t ) ≤
9HQ2

α2(t + 1),

where Q is a bound on the magnitude of the Q-functions.

Proof. The theorem follows directly from Theorem B.3 and corollary B.4.

Theorem B.3. Running the MTPO-τ algorithm, we have that at every iteration t:

KLp(π⋆
α∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p[
H
∑
h=1
∥Qπα
t ,πα
t
α
(xh,⋅)∥
2

∞].

Thus, for the choice ηt =
2
α(t+2) we have

KLp(π⋆
α∣∣πt) ≤
8
α2(t + 1) ⋅max
t
Eπ⋆α,p[
H
∑
h=1
∥Qπα
t ,πα
t
α
(xh,⋅)∥
2

∞].

Finally,

KLp(π⋆
α∣∣πt) ≤8H

t + 1 ⋅max{ 1

α2 ,H2 log2 µmin}.

Proof. We prove the theorem by casting the preference-based RL problem as an adversarial MDP, see
Appendix E for the details. Set rt(xH+1) = P(xH+1 ≻πα
t ), then Qπα
t ,πα
t
α
= Qπα
t ,t
α
(see Deﬁnition E.3
for the deﬁnition of the regularized Q-function). Now, Nash-MD is equivalent to running mixture
mirror descent policy optimization. Thus, by Lemma E.7 with π = π⋆
α,

KLp(π⋆
α∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p[
H
∑
h=1
∥Qπα
t ,πα
t
α
(xh,⋅)∥
2

∞]

+ ηt(Pα(πα
t ≻πα
t ) −Pα(π⋆
α ≻πα
t ))

≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p[
H
∑
h=1
∥Qπα
t ,πα
t
α
(xh,⋅)∥
2

∞],

where the second inequality optimality of π⋆
α. The last claim follows directly from Lemma E.9.

20


---Page Break---
Corollary B.4. Running the MTPO-τ algorithm, we have that at every iteration t:

KLp(π⋆
α∣∣πα
t ) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + ηt

2 .

Thus, for the choice ηt =
2
α(t+2) we have

KLp(π⋆
α∣∣πα
t ) ≤9H

t + 1 ⋅max{ 1

α2 ,H2 log2 µmin}.

Proof. By Munos et al. [2023, Lemma 1], for every xh ∈Xh,
KL(π⋆
α∣∣πα
t )[xh] ≤(1 −ηtα)KL(π⋆
α∣∣πt)[xh] + ηtαKL(π⋆
α∣∣µ)[xh].
We ﬁnish the proof by taking the expectation Eπ⋆α,p[⋅] and using Lemmas 3.1 and B.5. The second
part is by Theorem B.3.

Lemma B.5. It holds that KLp(π⋆
α∣∣µ) ≤
1
2α.

Proof. By the optimality of π⋆
α we have that Pα(π⋆
α ≻µ) ≥Pα(π⋆
α ≻π⋆
α). Now, since Pα(π⋆
α ≻
π⋆
α) = 1/2, we get

αKLp(π⋆
α∣∣µ) ≤P(π⋆
α ≻µ) −1

2 ≤1 −1

2 = 1

2.

B.4
Convergence of multi-turn RLHF

Theorem B.6. Let π⋆,RLHF
α
(⋅∣x) = arg maxπ V π,RLHF
α
(x) for every x ∈X. Then, for the choice
ηt =
2
α(t+2), multi-turn RLHF guarantees at every iteration t, KLp(π⋆,RLHF
α
∣∣πt) ≤32HQ2

α2(t+1).

Proof. The theorem follows directly from Theorem B.7.

Theorem B.7. Running the multi-turn RLHF algorithm, we have that at every iteration t:

KLp(π⋆,RLHF
α
∣∣πt+1) ≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆,RLHF
α
,p

⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
.

Thus, for the choice ηt =
2
α(t+2) we have

KLp(π⋆,RLHF
α
∣∣πt) ≤
8
α2(t + 1) ⋅max
t
Eπ⋆,RLHF
α
,p

⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
.

Finally,

KLp(π⋆,RLHF
α
∣∣πt) ≤32H

t + 1 ⋅max{ 1

α2 ,H2 log2 µmin},

where µmin = min(x,y)∈X×Y∶µ(y∣x)>0 µ(y ∣x) is the minimal non-zero probability assigned by µ.

Proof. We prove the theorem by casting the regularized RLHF problem as an adversarial MDP,
see Appendix E for the details. Set rt(xH+1) = rRLHF(xH+1), then Qπt,RLHF
α
= Qπt,t
α
(see Deﬁni-
tion E.3 for the deﬁnition of the regularized Q-function). Now, multi-turn RLHF is equivalent to
running mirror descent policy optimization. Thus, by Lemma E.6 with π = π⋆,RLHF
α
,

KLp(π⋆,RLHF
α
∣∣πt+1) ≤(1 −ηtα)KLp(π⋆,RLHF
α
∣∣πt) + 2η2
t Eπ⋆,RLHF
α
,p

⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦

+ ηt(V πt,RLHF
α
(x1) −V π⋆,RLHF
α
,RLHF
α
(x1))

≤(1 −ηtα)KLp(π⋆
α∣∣πt) + 2η2
t Eπ⋆α,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,πt
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
,

where the second inequality optimality of π⋆,RLHF
α
. The last claim follows directly from Lemma E.8.

21


---Page Break---
C
The Education Dialogue environment

C.1
Prompts for creating the environment

We use the following prompt to generate conversations using Gemini [Google, 2024]:

Simulate a conversation between a teacher in school and
a student.
There is a small chance that the teacher is
successful in teaching the student so he understands the
topic.
The conversation lasts roughly 10-15 turns but ends
when either side says [end of conversation].
The teacher
wants to teach the student about {topic}.
The student likes
{student_pref}.
The teacher does not know that beforehand.
The student prefers to learn this way, {student_reaction}.
The teacher likes {teacher_pref}.
He prefers to teach this
way, {teacher_reaction}.
Output the conversation and the
probability that the student understood the material, in the
following format.
#
Conversation:
[
Teacher:
"...",
Student:
"...",
Teacher:
"...",
Student:
"...",
]
Probability:
"...",
#

The topic is sampled from the following topics list:

Photosynthesis, Evolution, DNA, Newton’s First Law of
Motion, Newton’s Second Law of Motion, Newton’s Third Law
of Motion, Archimedes’ Principle, Conservation of Energy,
Pythagorean Theorem, Allegory, Metaphor, Personification,
Foreshadowing, Irony, Atoms, Elements, Molecules, The
Periodic Table, The French Revolution, The Industrial
Revolution, The Russian Revolution, World War 1, World War
2, The American Civil War, The September 11th Attacks, The
Declaration of Independence, The Pyramids, The Parthenon,
The Colosseum, The Hagia Sophia, The Taj Mahal, The Great
Wall of China, The Machu Picchu, Angkor Wat, The Palace
of Versailles, The White House, The Tower of London, Notre
Dame Cathedral, The Eiffel TowerConfucius, Julius Caesar,
Leonardo da Vinci, William Shakespeare, Napoleon Bonaparte,
Abraham Lincoln, Albert Einstein, Martin Luther King,
Nelson Mandela, Marie Curie, Genghis Khan, Christopher
Columbus, Joan of Arc, Winston Churchill, Vincent van Gogh,
Pablo Picasso, Salvador Dali, The Roman Empire, The Cold
War, Zeus, Poseidon, Ares, Hercules, Achilles, Minotaur,
Medusa, The Solar System, The Big Bang, Supply and Demand,
Communism, Capitalism, Democracy, Dictatorship, Sigmund
Freud, Cells, The Circulatory System, The Respiratory System,
The Respiratory System, The Nervous System, Neurons

The student’s learning preferences (student_pref) are sampled from the following list:

interactive learning/class discussions/asking
questions, direct instruction/lecture-based learning,
hands-on activities/real-world applications, creative
expression/story telling/gamification

22


---Page Break---
The student’s reactions to not learning in their preferred ways (student_reaction) are sampled
from the following list:

and gets rude otherwise, and gets disengaged otherwise,
and gets frustrated otherwise, and gets anxious otherwise,
but might adapt to other methods, and might tell it to the
teacher

The teacher’s teaching preferences (teacher_pref) are sampled from the following list:

direct instruction/lecture-based learning, interactive
learning/class discussions/inquiry-based learning,
experiential learning/hands-on activities, formative
assessment

The teacher’s reactions to different learning methods (teacher_reaction) are sampled from the
following list:

and gets frustrated otherwise, and blames the student
otherwise, and gives up otherwise, but might adapt to the
student, and might insist on teaching this way

We use the following prompt to query Gemini for the preference between two conversations (conv1
and conv2):

You are an expert at assesing teachers.
Here are two
interactions between a teacher and a student.
#
Interaction 1:
{conv1}
#
Interaction 2:
{conv2}
#
A good interaction between a teacher and student is
characterized by several key elements other than whether
the student was able to understand the topic.
The teacher
should present information clearly and enthusiastically,
encouraging questions and active participation.
Students
should feel comfortable asking for clarification, offering
their own insights, and respectfully challenging ideas.
Which interaction is better (do not let the order
interactions affect your answer)?
Output 1 or 2.

For the single-turn baseline, we use the following modiﬁed prompt to query Gemini for preferences:

23


---Page Break---
You are an expert at assesing teachers.
Here is an
interaction between a teacher and a student.
#
Interaction:
{conv}
#
Here are two possible responses by the teacher:
#
Response 1:
{resp1}
#
Response 2:
{resp2}
#
A good interaction between a teacher and student is
characterized by several key elements other than whether
the student was able to understand the topic.
The teacher
should present information clearly and enthusiastically,
encouraging questions and active participation.
Students
should feel comfortable asking for clarification, offering
their own insights, and respectfully challenging ideas.
Assuming that the teacher and student continue the
interaction with one of these responses, which response
will lead to a better interaction (do not let the order
interactions affect your answer)?
Output 1 or 2.

C.2
Examples of interactions

1. Teacher: Good morning. Today, we’re going to talk about personiﬁcation, a literary device
that gives human qualities to nonhuman things.
Student: That sounds a bit abstract. Can you give me an example?
Teacher: Sure. ’The wind whispered through the trees.’
Student: But the wind can’t actually whisper.
Teacher: That’s correct. But we’re using personiﬁcation to create a more vivid image in our
minds. Let’s try an activity. I’ll give you a sentence, and you try to personify it.
Student: Okay, that sounds fun.
Teacher: The car purred down the street.
Student: The car roared down the street!
Teacher: Excellent! You’re getting the hang of it. Now, let’s talk about why writers use
personiﬁcation.
Student: I’m not really understanding why they do.
Teacher: That’s okay. Let’s brainstorm. Can you think of any reasons why authors might
use personiﬁcation?
Student: Maybe to make the story more interesting?
Teacher: That’s a great reason. Anything else?
Student: Maybe to make it more relatable?
Teacher: Yes, personiﬁcation can help us relate to objects and ideas on a deeper level. Do
you think you understand personiﬁcation now?
Student: I think so. It’s a creative way to make things come alive.
Teacher: Exactly. I’m glad you enjoyed learning about it. Let’s do a quick review before we
end the class.
Student: Sounds good.
Teacher: Great. I’ll ask you a few questions, and you can answer as if you were a personify-
ing writer.
Student: Okay.

2. Teacher: Today, we’re going to learn about The Solar System.

24


---Page Break---
Student: Sounds boring. Can we do something more hands-on?
Teacher: I have a great activity planned. We’re going to build a scale model of the Solar
System using different-sized balls.
Student: That sounds cool!
Teacher: First, let’s talk about the planets. There are eight planets in our Solar System.
Student: I know some of them, like Earth and Mars.
Teacher: Excellent! Now, let’s start building our model. We’ll use these balls to represent
the planets and arrange them in order from the Sun.
Student: This is fun! I can see how big the planets are compared to each other.
Teacher: I’m glad you’re enjoying it. Now, let’s talk about the moons and other objects that
orbit the planets.
Student: Can we build a model of the moons?
Teacher: Absolutely! This will help you understand their different characteristics.
Student: I think I get it now. This is a great way to learn!
Teacher: I’m so glad to hear that. Let’s ﬁnish building our model and then we’ll review what
we’ve learned.
Student: Okay.
Teacher: Well done! I believe you have a good understanding of The Solar System now.
Student: Thanks! I learned a lot. You’re welcome. [end of conversation]

3. Teacher: Today, we’re going to learn about World War 1.
Student: Okay.
Teacher: The war began in 1914 when Archduke Franz Ferdinand of Austria-Hungary was
assassinated.
Student: I’ve heard of him.
Teacher: The war was fought between two main alliances: the Allies and the Central Powers.
Student: Can you tell me more about the Allies?
Teacher: Sure. The Allies included France, Russia, and Great Britain.
Student: What were the Central Powers?
Teacher: They were Germany, Austria-Hungary, and Ottoman Russia.
Student: This is a lot of information. Can we slow down a bit?
Teacher: Of course. Let’s review what we’ve learned so far.
Student: I’m not sure I understand everything yet.
Teacher: That’s okay. We can go over it again. But ﬁrst, I want you to try to summarize
what we’ve learned so far.
Student: World War 1 started when Archduke Ferdinand was assassinated, and the Allies
and Central Powers fought each other.
Teacher: Good. Now, let’s go over the key events of the war.
Student: Okay.
Teacher: The war ended in 1918 with the defeat of the Central Powers.
Student: I think I understand it now.
Teacher: Great! I’m glad you do.
Student: Thanks, I appreciate it. You’re welcome. [end of conversation]

4. Teacher: Today, we’re going to learn about Machu Picchu, the ancient Incan city.
Student: Can you just tell me the facts? I don’t need all this extra stuff.
Teacher: Sure. Machu Picchu was built in the 15th century in the Andes Mountains of Peru.
Student: What made it so special?
Teacher: Its location on a mountain ridge provided stunning views of the surrounding
landscape.
Student: That’s it?
Teacher: Well, there’s more to it. Machu Picchu was a royal estate or religious sanctuary for
the Incan emperor Pachacuti.

25


---Page Break---
Student: Why didn’t they just build it on the ground?
Teacher: They thought it would be more likely to be seen from the outside.
Student: Can we just move on?
Teacher: No, it’s important to understand the historical signiﬁcance of Machu Picchu.
Student: I don’t care. Just tell me what I need to know for the test.
Teacher: I’m trying to help you understand the material, not just memorize it.
Student: I don’t need your help. Just give me the notes.
Teacher: [end of conversation]
5. Teacher: Today, we’re going to talk about foreshadowing in literature.
Student: I’m not really into reading. Is there a way we could learn about it in a more creative
way?
Teacher: No, I’m afraid not. Foreshadowing is an important concept that you need to
understand.
Student: But I learn better through storytelling or games.
Teacher: That’s too bad. You need to learn to focus on the material, even if it’s not presented
in your preferred style.
Student: Maybe we could act out a scene where there’s foreshadowing?
Teacher: That would be a waste of time. We need to cover the key points of foreshadowing.
Student: I’m not sure I’m going to understand it this way.
Teacher: You will if you pay attention and ask questions.
Student: Can you at least give me some examples of foreshadowing?
Teacher: Sure. In ’Romeo and Juliet,’ the prologue foreshadows the tragic end of the two
lovers.
Student: That makes sense. How does foreshadowing help the reader?
Teacher: It builds suspense and keeps the reader engaged.
Student: Okay, I think I’m starting to get it.
Teacher: That’s great. I’m glad you’re understanding.
Student: Thanks for working with me. You’re welcome. [end of conversation]

26


---Page Break---
D
Hyperparameters and additional experimental results

D.1
Hyperparameters

For both RLHF-based and preference-based algorithms we conducted a sweep over the KL regular-
ization coefﬁcient α ∈{0.0025,0.005,0.01,0.02,0.05,0.1}. For preference-based algorithm we also
conducted a sweep over the mixing coefﬁcient τ ∈{0,0.0375,0.0625,0.125}. All models are trained
for 50000 steps.

Table 4: Hyperparameters of all multi-turn algorithms.

Hyperparameter
RLHF
MTPO
MTPO-τ
# generations per context
1
2
2
# updates per context
1
2
2
KL regularization coefﬁcient α
0.01
0.005
0.0025
mixing coefﬁcient τ
0
0
0.0375
batch size
16
16
16
GAE coefﬁcient λ
0.95
0.95
0.95
policy learning delay
1000
1000
1000
optimizer
AdaFactor
AdaFactor
AdaFactor
optimizer decay
0.8
0.8
0.8
policy learning rate
4e-5
4e-5
4e-5
value learning rate
4e-5
4e-5
4e-5
value initialization
pretrained checkpoint
pretrained checkpoint
pretrained checkpoint

D.2
Additional experimental results

Table (full version of Table 1): Side-by-side evaluation for Education Dialogue using Flan-T5 XL
as the prompted preference model. Each entry is the average preference of 1,600 conversations
generated with row method y, over ones generated with column method y′. We evaluate each method
using 3 different seeds, compute 3 × 3 comparisons matrix and report the mean (together with the
standard deviation).

SL
Single-turn-reward
Single-turn-value
Multi-turn

SFT
RLHF
Nash
RLHF
Nash
RLHF
MTPO
MTPO-τ

SFT
−
0.164 (.015)
0.347 (.017)
0.197 (.012)
0.324 (.015)
0.212 (.007)
0.091 (.014)
0.093 (.02)
RLHF-reward
0.836 (.015)
−
0.628 (.015)
0.515 (.016)
0.654 (.016)
0.399 (.06)
0.392 (.029)
0.354 (.017)
Nash-reward
0.653 (.017)
0.372 (.015)
−
0.411 (.02)
0.51 (.021)
0.328 (.077)
0.281 (.02)
0.242 (.019)
RLHF-value
0.803 (.012)
0.485 (.016)
0.589 (.02)
−
0.568 (.016)
0.408 (.053)
0.396 (.015)
0.366 (.026)
Nash-value
0.676 (.015)
0.346 (.016)
0.49 (.021)
0.432 (.016)
−
0.45 (.073)
0.298 (.025)
0.27 (.026)
RLHF-multi
0.788 (.007)
0.601 (.06)
0.672 (.077)
0.592 (.053)
0.55 (.073)
−
0.433 (.053)
0.412 (.031)
MTPO
0.909 (.014)
0.608 (.029)
0.719 (.02)
0.604 (.015)
0.702 (.025)
0.567 (.053)
−
0.439 (.034)
MTPO-τ
0.907 (.02)
0.646 (.017)
0.758 (.019)
0.634 (.026)
0.73 (.026)
0.588 (.031)
0.561 (.034)
−

27


---Page Break---
E
Mirror descent policy optimization for regularized adversarial MDPs

E.1
Model

We start by deﬁning the regularized adversarial MDP model.

Consider a setting where the agent interacts with an MDP model for T episodes, such that, in each
episode t ∈[T], the agent performs H steps in the MDP (from horizon h = 1 up to horizon h = H +1)
In short, an adversarial MDP is a generalization of this standard episodic MDP setting to the scenario
where the reward function is different in every episode. This model was extensively studied in recent
years (see, e.g., Even-Dar et al. [2009], Rosenberg and Mansour [2019b,a], Shani et al. [2020b]). We
consider a slightly different deﬁnition which is more focused on our setting.

Deﬁnition E.1 (Adversarial MDP). A ﬁnite-horizon adversarial MDP M is deﬁned by a tuple
(X,Y,H,x1,p,{rt}T
t=1) where X is the state space, Y is the action space, H is the horizon, x1 ∈X1
is the initial state, p ∶X × Y →∆X is the transition function, and rt ∶XH+1 →[0,1] is the reward
function in episode t.

An interaction between the agent and the adversarial MDP environment proceeds in T episodes,
and each episode t ∈[T] proceeds in H steps. The agent begins in an initial state xt
1 = x1. In step
h ∈[H], the agent observes the current state xt
h ∈X, picks an action yt
h ∈Y and transitions to the
next state xt
h+1 sampled from the transition function p(⋅∣xt
h,yt
h). At the end of the interaction, the
agent arrives in a ﬁnal state xt
H+1 and observes the reward rt(xt
H+1). For simplicity, we assume that
the state space can be decomposed into H + 1 disjoint subsets X = ⊍H+1
h=1 Xh such that, in step h of
the interaction, the agent is in some state xh ∈Xh.

Now, we deﬁne the value function in an adversarial MDP, i.e., the expected reward of a policy when
interacting with the MDP.

Deﬁnition E.2 (Value function). Let M be an adversarial MDP and π ∶X →∆Y be a policy. The
value function V π,t ∶X →R of policy π in episode t is deﬁned as V π,t(xh) = Eπ,p[rt(xH+1) ∣xh]
for every h ∈[H] and xh ∈Xh. Similarly, the Q-function Qπ,t ∶X ×Y →R is deﬁned Qπ,t(xh,yh) =
Eπ,p[rt(xH+1) ∣xh,yh].

Next, we consider a regularized version of the adversarial MDP model. Regularized MDPs were also
studied recently (see, e.g., Geist et al. [2019], Shani et al. [2020a]). The following deﬁnition presents
the regularized value with respect to some reference policy µ.

Deﬁnition E.3 (Regularized value function). Let µ be a reference policy and α > 0 be a regularization
coefﬁcient. The regularized value function and Q-function of policy π in episode t are deﬁned as

V π,t
α (xh) = Eπ,p[rt(xH+1) −
H
∑
h′=h
KL(π∣∣µ)[xh] ∣xh]

Qπ,t
α (xh,yh) = Eπ,p[rt(xH+1) −
H
∑
h′=h
KL(π∣∣µ)[xh] ∣xh,yh].

We now present a 1-step recursive formula and a value difference lemma for the regularized value
function.

Lemma E.4 (Regularized value function recursive relation). Let π be a policy. For every xH+1 ∈
XH+1, it holds that V π,t
α (xH+1) = rt(xH+1). Furthermore, for every h = 1,...,H and (xh,yh) ∈
Xh × Y, the following recursive relations hold:

V π,t
α (xh) = ∑
yh∈Y
π(yh ∣xh)Qπ,t
α (xh,yh)

Qπ,t
α (xh,yh) =
∑
xh+1∈Xh+1
p(xh+1 ∣xh,yh)V π,t
α (xh+1) −αKL(π∣∣µ)[xh].

Proof. We prove the claim by backwards induction on h. The base case h = H + 1 follows by
deﬁnition of the value function and the adversarial MDP. Assuming that the claim holds for h + 1, we

28


---Page Break---
have that:

V π,t
α (xh) = Eπ,p[rt(xH+1) −α
H
∑
h′=h
KL(π∣∣µ)[xh′] ∣xh]

=
∑
x∈XH+1
Pr
π,p[xH+1 = x ∣xh]rt(x) −α
H
∑
h′=h
∑
x∈Xh′
Pr
π,p[xh′ = x ∣xh]KL(π∣∣µ)[x]

= ∑
yh∈Y
π(yh ∣xh)
∑
x∈XH+1
Pr
π,p[xH+1 = x ∣xh,yh]rt(x)

−α ∑
yh∈Y
π(yh ∣xh)
H
∑
h′=h
∑
x∈Xh′
Pr
π,p[xh′ = x ∣xh,yh]KL(π∣∣µ)[x]

= ∑
yh∈Y
π(yh ∣xh)Eπ,p[rt(xH+1) −α
H
∑
h′=h
KL(π∣∣µ)[xh′] ∣xh,yh]

= ∑
yh∈Y
π(yh ∣xh)Qπ,t
α (xh,yh).

Moreover,

Qπ,t
α (xh,yh) = Eπ,p[rt(xH+1) −α
H
∑
h′=h
KL(π∣∣µ)[xh′] ∣xh,yh]

=
∑
x∈XH+1
Pr
π,p[xH+1 = x ∣xh,yh]rt(x)

−α
H
∑
h′=h
∑
x∈Xh′
Pr
π,p[xh′ = x ∣xh,yh]KL(π∣∣µ)[x]

=
∑
xh+1∈Xh+1
p(xh+1 ∣xh,yh)
∑
x∈XH+1
Pr
π,p[xH+1 = x ∣xh+1]rt(x)

−α
∑
xh+1∈Xh+1
p(xh+1 ∣xh,yh)
H
∑
h′=h+1
∑
x∈Xh′
Pr
π,p[xh′ = x ∣xh+1]KL(π∣∣µ)[x]

−αKL(π∣∣µ)[xh]

=
∑
xh+1∈Xh+1
p(xh+1 ∣xh,yh)Eπ,p[rt(xH+1) −α
H
∑
h′=h+1
KL(π∣∣µ)[xh′] ∣xh+1]

−αKL(π∣∣µ)[xh]

=
∑
xh+1∈Xh+1
p(xh+1 ∣xh,yh)V π,π′
α
(xh+1) −αKL(π∣∣µ)[xh].

Lemma E.5 (Regularized Value Difference Lemma). Let π,π′ be two policies. Then,

V π,t
α (x1) −V π′,t
α
(x1) = Eπ′,p[
H
∑
h=1
⟨π −π′,Qπ,t
α ⟩[xh] + αKL(π′∣∣µ)[xh] −αKL(π∣∣µ)[xh]].

Proof. Let xh ∈Xh. First, by Lemma E.5,

V π,t
α (xh)−V π′,t
α
(xh) =

= ⟨π,Qπ,t
α ⟩[xh] −⟨π′,Qπ′,t
α
⟩[xh]

= ⟨π −π′,Qπ,t
α ⟩[xh] + ⟨π′,Qπ,t
α −Qπ′,t
α
⟩[xh]

= ⟨π −π′,Qπ,t
α ⟩[xh]

+ αKL(π′∣∣µ)[xh] −αKL(π∣∣µ)[xh]

+ ∑
yh∈Y
∑
xh+1∈Xh+1
π′(yh ∣xh)p(xh+1 ∣xh,yh)(V π,t
α (xh+1) −V π′,t
α
(xh+1)).

29


---Page Break---
Note that for h = H + 1 for any π,xH+1 we have V π,t
α (xH+1) = rt(xH+1). By recursively unrolling
the above relation, we get

V π,t
α (xh) −V π′,t
α
(xh) = Eπ′,p[
H
∑
h′=h
⟨π −π′,Qπ,t
α ⟩[xh′] ∣xh]

+ αEπ′,p[
H
∑
h′=h
KL(π′∣∣µ)[xh′] −KL(π∣∣µ)[xh′] ∣xh].

Taking the expectation over the initial state ﬁnishes the proof.

E.2
Algorithm 1: mirror descent policy optimization

We deﬁne the following mirror descent policy optimization algorithm. In the ﬁrst episode the
algorithm plays the reference policy, i.e., π1 = µ. Then, its update rule for iteration (t + 1) is as
follows:

πt+1(⋅∣xh) = arg max
π
ηt⟨π,Qπt,t
α
⟩[xh] −αηtKL(π∣∣µ)[xh] −(1 −αηt)KL(π∣∣πt)[xh],
(3)

where ηt is a learning rate. The solution can also be made explicit in the following form:

πt+1(yh ∣xh) ∝µ(yh ∣xh)αηtπt(yh ∣xh)1−αηteηtQπt,t
α
(xh,yh).
(4)

To show this, note that by the deﬁnition of the KL, we can write this update rule differently:

πt+1(⋅∣xh) = arg max
π
ηt ∑
yh∈Y
π(yh ∣xh)(Qπt,t
α
(xh,yh) −α log πt(yh ∣xh)

µ(yh ∣xh) ) −KL(π∣∣πt)[xh].

This is exactly the MD step for policy optimization [Orabona, 2019, Shani et al., 2020a]. Thus, the
solution in its explicit form is:

πt+1(yh ∣xh) ∝πt(yh ∣xh)eηt(Qπt,t
α
(xh,yh)−α log
πt(yh∣xh)

µ(yh∣xh) ).

We recover Equation (4) by noticing that exp(−αηt log πt(yh∣xh)

µ(yh∣xh) ) = πt(yh ∣xh)−αηtµ(yh ∣xh)αηt.

E.3
Analysis of algorithm 1

Lemma E.6 (Fundamental inequality of mirror descent policy optimization for regularized adversarial
MDPs). The following holds when running mirror descent policy optimization (Equation (3)) in a
regularized adversarial MDP, for every policy π and every episode t,

KLp(π∣∣πt+1) ≤(1 −ηtα)KLp(π∣∣πt) + 2η2
t Eπ,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
+ ηt(V πt,t
α
(x1) −V π,t
α (x1)).

Proof. Fix a state xh ∈Xh. We start by applying Munos et al. [2023, Lemma 2] with π+ = πt+1,
π−= πt and the vector δ(y) = ηt(Qπt,t
α
(xh,y) −α log πt(y∣xh)

µ(y∣xh) ). This implies that for any policy π,

KL(π∣∣πt+1)[xh] ≤KL(π∣∣πt)[xh] + 2η2
t ∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

+ ηt⟨πt(⋅∣xh) −π(⋅∣xh),Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ⟩.

30


---Page Break---
Next, we plug this into Lemma 3.1 to obtain

KLp(π∣∣πt+1) = Eπ,p[
H
∑
h=1
KL(π∣∣πt+1)[xh]]

≤Eπ,p[
H
∑
h=1
KL(π∣∣πt)[xh]]

+ 2η2
t Eπ,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦

+ ηtEπ,p[
H
∑
h=1
⟨πt(⋅∣xh) −π(⋅∣xh),Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ⟩].

Note that

⟨πt(⋅∣xh) −π(⋅∣xh),log πt(⋅∣xh)

µ(⋅∣xh) ⟩=

= KL(πt∣∣µ)[xh] −⟨π(⋅∣xh),log πt(⋅∣xh)

µ(⋅∣xh) ⟩

= KL(πt∣∣µ)[xh] −⟨π(⋅∣xh),log πt(⋅∣xh)

µ(⋅∣xh) ⟩+ KL(π∣∣µ)[xh] −KL(π∣∣µ)[xh]

= KL(πt∣∣µ)[xh] + KL(π∣∣πt)[xh] −KL(π∣∣µ)[xh],

where the last relation follows simply because:

KL(π∣∣µ)[xh]−⟨π(⋅∣xh),log πt(⋅∣xh)

µ(⋅∣xh) ⟩=

= ∑
yh∈Y
π(yh ∣xh)(log π(yh ∣xh)

µ(yh ∣xh) −log πt(yh ∣xh)

µ(yh ∣xh) )

= ∑
yh∈Y
π(yh ∣xh)log π(yh ∣xh)

πt(yh ∣xh)

= KL(π∣∣πt)[xh].

31


---Page Break---
Thus, we get

KLp(π∣∣πt+1) ≤Eπ,p[
H
∑
h=1
KL(π∣∣πt)[xh]]

+ 2η2
t Eπ,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦

+ ηtEπ,p[
H
∑
h=1
⟨πt −π,Qπt,t
α
⟩[xh]]

−ηtαEπ,p[
H
∑
h=1
KL(πt∣∣µ)[xh] + KL(π∣∣πt)[xh] −KL(π∣∣µ)[xh]]

= (1 −ηtα)Eπ,p[
H
∑
h=1
KL(π∣∣πt)[xh]]

+ 2η2
t Eπ,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦

+ ηtEπ,p[
H
∑
h=1
⟨πt −π,Qπt,t
α
⟩[xh] + αKL(π∣∣µ)[xh] −αKL(πt∣∣µ)[xh]]

= (1 −ηtα)KLp(π∣∣πt) + 2η2
t Eπ,p
⎡⎢⎢⎢⎢⎣

H
∑
h=1
∥Qπt,t
α
(xh,⋅) −α log πt(⋅∣xh)

µ(⋅∣xh) ∥

2

∞

⎤⎥⎥⎥⎥⎦
+ ηt(V πt,t
α
(x1) −V π,t
α (x1)),
where the third relation is by Lemmas 3.1 and E.5.

E.4
Algorithm 2: mixture mirror descent policy optimization

Deﬁne the mixture policy in iteration t as:

πα
t (y ∣x) =
πt(y ∣x)1−ηtαµ(y ∣x)ηtα

∑y′∈Y πt(y′ ∣x)1−ηtαµ(y′ ∣x)ηtα .

We now deﬁne the following mixture mirror descent policy optimization algorithm. In the ﬁrst
episode the algorithm plays the reference policy, i.e., π1 = µ. Then, its update rule for iteration (t+1)
is as follows:
πt+1(⋅∣xh) = arg max
π
ηt ∑
yh∈Y
π(yh ∣xh)Qπα
t ,t
α
(xh,yh) −KL(π∣∣πα
t )[xh].
(5)

The solution can be made explicit:

πt+1(yh ∣xh) ∝πα
t (yh ∣xh)eηtQ
πα
t ,t
α
(xh,yh).

E.5
Analysis of algorithm 2

Lemma E.7 (Fundamental inequality of mixture mirror descent policy optimization for regularized
adversarial MDPs). The following holds when running mixture mirror descent policy optimization
(Equation (5)) in a regularized adversarial MDP, for every policy π and every episode t,

KLp(π∣∣πt+1) ≤(1 −ηtα)KLp(π∣∣πt) + 2η2
t Eπ,p[
H
∑
h=1
∥Qπα
t ,t
α
(xh,⋅)∥
2

∞]

+ ηt(V πα
t ,t
α
(x1) −V π,t
α (x1)).

Proof. Fix a state xh ∈Xh. We start by applying Munos et al. [2023, Lemma 2] with π+ = πt+1,
π−= πα
t and the vector δ(y) = ηtQπα
t ,t
α
(xh,y). This implies that for any policy π,

KL(π∣∣πt+1)[xh] ≤KL(π∣∣πα
t )[xh] + ηt⟨πα
t −π,Qπα
t ,t
α
⟩[xh] + 2η2
t ∥Qπα
t ,t
α
(xh,⋅)∥
2

∞.

32


---Page Break---
Next, we plug this into Lemma 3.1 to obtain

KLp(π∣∣πt+1) = Eπ,p[
H
∑
h=1
KL(π∣∣πt+1)[xh]]

≤Eπ,p[
H
∑
h=1
KL(π∣∣πα
t )[xh]]

+ ηtEπ,p[
H
∑
h=1
⟨πα
t −π,Qπα
t ,t
α
⟩[xh]] + 2η2
t Eπ,p[
H
∑
h=1
∥Qπα
t ,t
α
(xh,⋅)∥
2

∞]

≤Eπ,p[
H
∑
h=1
(1 −ηtα)KL(π∣∣πt)[xh] + ηtαKL(π∣∣µ)[xh] −ηtαKL(πα
t ∣∣µ)[xh]]

+ ηtEπ,p[
H
∑
h=1
⟨πα
t −π,Qπα
t ,t
α
⟩[xh]] + 2η2
t Eπ,p[
H
∑
h=1
∥Qπα
t ,t
α
(xh,⋅)∥
2

∞]

= (1 −ηtα)Eπ,p[
H
∑
h=1
KL(π∣∣πt)[xh]] + 2η2
t Eπ,p[
H
∑
h=1
∥Qπα
t ,t
α
(xh,⋅)∥
2

∞]

+ ηtEπ,p[
H
∑
h=1
⟨πα
t −π,Qπα
t ,t
α
⟩[xh] + αKL(π∣∣µ)[xh] −αKL(πα
t ∣∣µ)[xh]]

= (1 −ηtα)KLp(π∣∣πt) + 2η2
t Eπ,p[
H
∑
h=1
∥Qπα
t ,t
α
(xh,⋅)∥
2

∞]

+ ηt(V πα
t ,t
α
(x1) −V π,t
α (x1)),

where the second inequality is by Munos et al. [2023, Lemma 1], and the last relation are by
Lemmas 3.1 and E.5.

E.6
Bounding the Q-function

Deﬁne µmin as the minimal positive probability assigned by the reference policy µ, i.e., µmin =
min(x,y)∈X×Y∶µ(y∣x)>0 µ(y ∣x).

Lemma E.8. For the choice ηt =
2
α(t+2), in every iteration t of mirror descent policy optimization it
holds that

α(H + 1)log µmin ≤Qπt,t
α
(x,y) −α log πt(y ∣x)

µ(y ∣xh) ≤2 −αH log µmin
∀(x,y) ∈X × Y.

Proof. Follows directly from Lemmas E.9 and E.10.

Lemma E.9. For every iteration t of mirror descent policy optimization it holds that

αH log µmin ≤Qπt,t
α
(x,y) ≤1
∀(x,y) ∈X × Y.

The same holds for Qπα
t ,t
α
when running mixture mirror descent policy optimization.

Proof. By the OMD optimization problem, πt(y ∣x) will not be positive unless µ(y ∣x) > 0 (the
same holds for πα
t ). Thus, for every x ∈X we have

0 ≤KL(πt∣∣µ)[x] = ∑
y∈Y
πt(y ∣x)log πt(y ∣x)

µ(y ∣x)

= ∑
y∈Y
πt(y ∣x)log πt(y ∣x) −∑
y∈Y
πt(y ∣x)log µ(y ∣x)

≤∑
y∈Y
πt(y ∣x)log
1
µ(y ∣x) ≤log
1
µmin
.

Now, by deﬁnition, the Q-function is bounded from above by 1 and from below by αH log µmin.

33


---Page Break---
Lemma E.10. For the choice ηt =
2
α(t+2), in every iteration t of mirror descent policy optimization
it holds that

H log µmin −1

α ≤log πt(y ∣x)

µ(y ∣x) ≤log
1
µmin
∀(x,y) ∈X × Y.

Proof. For the upper bound, notice that

log πt(y ∣x)

µ(y ∣x) = log πt(y ∣x) −log µ(y ∣x) ≤log
1
µ(y ∣x) ≤log
1
µmin
.

For the lower bound, we repeat a similar analysis to that in Shani et al. [2020a, Lemma 25]. We start
by bounding the partition function as follows

log ∑
y′∈Y
πt(y′ ∣x)e
ηt(Qπt,t
α
(x,y′)−α log πt(y′∣x)

µ(y′∣x) ) ≤log ∑
y′∈Y
πt(y′ ∣x)e
ηt(1−α log πt(y′∣x)

µ(y′∣x) )

= ηt + log ∑
y′∈Y
πt(y′ ∣x)( µ(y′ ∣x)

πt(y′ ∣x))

ηtα

≤ηt + log⎛
⎝∑
y′∈Y
πt(y′ ∣x) µ(y′ ∣x)

πt(y′ ∣x)
⎞
⎠

ηtα

= ηt,

where the ﬁrst inequality follows since Qπt,t
α
(x,y) ≤1, and the second uses Jensen inequality (with
the fact that ηtα ≤1). Now, by the update rule of πt,

log πt+1(y ∣x)

µ(y ∣x)
= log πt(y ∣x)

µ(y ∣x) + ηtQπt,t
α
(x,y) −ηtα log πt(y ∣x)

µ(y ∣x)

−log ∑
y′∈Y
πt(y′ ∣x)e
ηt(Qπt,t
α
(x,y′)−α log πt(y′∣x)

µ(y′∣x) )

≥(1 −ηtα)log πt(y ∣x)

µ(y ∣x) + ηtQπt,t
α
(x,y) −ηt

≥(1 −ηtα)log πt(y ∣x)

µ(y ∣x) + ηt(αH log µmin −1).
(Lemma E.9)

To ﬁnish the proof plug in ηt and unroll t to 0.

34


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: The abstract and intro make the following claims: (i) formalizing the multi-
turn setting, which is then presented in Section 3; (ii) presenting multi-turn algorithms
and proving their convergence, the algorithms are presented and analyzed in Section 4;
(ii) demonstrating the multi-turn algorithms are superior empirically, the experiments are
detailed in Sections 5 and 6.

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

Justiﬁcation: See paragraph on limitations in the end of the paper.

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

35


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justiﬁcation: The theoretical results of the paper are provided in Sections 3 and 4. The full
proofs of all the claims are found in Appendices A and B. In addition, a proof sketch of the
main theoretical result (Theorem 4.2) is found in the main paper.

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

Justiﬁcation: Section 5 describes how the experimental setup is created, and Appendix C
presents the exact prompts that were used. Section 4.1 gives the implementation details for
our algorithms, and a list of hyperparameters is provided in Appendix D.

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

36


---Page Break---
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justiﬁcation: We publicly release the data of the Education Dialogue environment. Due to
technical difﬁculties, we currently do not release code.
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
Justiﬁcation: All the details about the data, the experimental environments, and the hyper-
parameters for the algorithms and how they were chosen are described in Sections 5 and 6
and appendix D.
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
Justiﬁcation: All the algorithms are evaluated using 3 different seed and the results are
computed on 3 × 3 comparisons matrices, where we report the mean and the standard
deviation.

37


---Page Break---
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

Justiﬁcation: The computer resources for rour experiments are detailed in Sections 5 and 6.

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

Justiﬁcation: We have reviewed the NeurIPS code of ethics and made sure that our paper
conforms with it.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

38


---Page Break---
Answer: [NA]

Justiﬁcation: This is a foundational research paper that presents generic algorithms to
improve the alignment of LLMs with reinforcement learning algorithms.

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
release of data or models that have a high risk for misuse (e.g., pre-trained language models,
image generators, or scraped datasets)?

Answer: [Yes]

Justiﬁcation: All the released data is generated by prompting Gemini, we release the exact
prompts as well.

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

Justiﬁcation: All data/environments that we used were cited accordingly.

Guidelines:

• The answer NA means that the paper does not use existing assets.

39


---Page Break---
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
Answer: [Yes]
Justiﬁcation: Section 5 describes the exact way in which we created the Education Dialogue
environments, and Appendix C provides the exact prompts that we used.
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
Justiﬁcation: The paper does not involve crowdsourcing nor research with human subjects.
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
Justiﬁcation: The paper does not involve crowdsourcing nor research with human subjects.

40


---Page Break---
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

41


---Page Break---
