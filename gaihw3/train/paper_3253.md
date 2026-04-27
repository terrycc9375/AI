When Your AIs Deceive You:
Challenges of Partial Observability in
Reinforcement Learning from Human Feedback

Leon Lang∗
University of Amsterdam
Davis Foote*
UC Berkeley
Stuart Russell
UC Berkeley

Anca Dragan
UC Berkeley
Erik Jenner
UC Berkeley
Scott Emmons*
UC Berkeley

Abstract

Past analyses of reinforcement learning from human feedback (RLHF) assume that
the human evaluators fully observe the environment. What happens when human
feedback is based only on partial observations? We formally deﬁne two failure
cases: deceptive inﬂation and overjustiﬁcation. Modeling the human as Boltzmann-
rational w.r.t. a belief over trajectories, we prove conditions under which RLHF is
guaranteed to result in policies that deceptively inﬂate their performance, overjustify
their behavior to make an impression, or both. Under the new assumption that the
human’s partial observability is known and accounted for, we then analyze how
much information the feedback process provides about the return function. We
show that sometimes, the human’s feedback determines the return function uniquely
up to an additive constant, but in other realistic cases, there is irreducible ambiguity.
We propose exploratory research directions to help tackle these challenges and
experimentally validate both the theoretical concerns and potential mitigations, and
caution against blindly applying RLHF in partially observable settings.

1
Introduction

Reinforcement learning from human feedback (RLHF) and its variants are widely used for ﬁnetuning
foundation models, including ChatGPT [OpenAI, 2022], Bard [Manyika, 2023], Gemini [Gem-
ini Team, 2023], Llama 2 [Touvron et al., 2023], and Claude [Bai et al., 2022, Anthropic, 2023a,b].
Prior theoretical analysis of RLHF assumes that the human fully observes the state of the world
[Skalse et al., 2023]. Under this assumption, it is possible to recover the ground-truth return function
from Boltzmann-rational human feedback (see Proposition 3.1).

In reality, however, this assumption is false. Models like ChatGPT are interacting with the internet
and software tools via plugins [OpenAI, 2023]. Software assistants like Devin are interacting with
complex IDEs to produce their results [Wu, 2024]. By default, some of the models’ work then
happens in the background, not observed by the users; see Figure 1. With the tasks performed by
language model assistants becoming more complex, it is also increasingly time consuming for humans
to evaluate the entire model behavior and input. Therefore, we are anticipating a future where by
default, the human evaluators do not fully observe the environment state that the language assistant is
embedded in. Our work analyzes the consequences and risks of such partial observability.

We begin our investigation with a simple example, illustrated in Figure 2, meant to isolate the key
factor leading to deception (in practice, we imagine that this effect would be embedded in a larger,

∗Core research contributor. Correspondence to l.lang@uva.nl, emmons@berkeley.edu

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Feedback being collected...

...from humans 
that can’t see 
everything the 
agent sees.

Figure 1: Partial observability in ChatGPT [OpenAI, 2023]. Users do not observe the online content
that ChatGPT observes yet still provide thumbs-up thumbs-down feedback. OpenAI’s privacy policy
[OpenAI, 2024c] allows user feedback to be used for training models. We show in Theorem 4.5 that
if feedback of human evaluators is based on partial observations, then this can lead to deceptive and
overjustifying behavior by the language model.

more complex system, e.g. with logs containing thousands of lines). An AI assistant is helping a user
install software. The assistant can hide error messages by redirecting them to /dev/null. We model
the human as having a belief B over the state and extend the Boltzmann-rational assumption from
prior work to incorporate this belief. In the absence of an error message, the human is uncertain if
the agent left the system untouched or hid the error message from a failed installation. If the human
interprets trajectories without error messages optimistically, the AI learns to hide error messages.
Figure 4 provides further details on how this failure occurs. It also shows a second case where the AI
clutters the output with overly verbose logs.

Generalizing from these examples, we formalize dual risks: deceptive inﬂation and overjustiﬁcation.
We provide a mathematical deﬁnition of each. When the observation kernel (the function specifying
the observations given states) is deterministic, Theorem 4.5 analyzes properties of suboptimal policies
learned by RLHF. These policies exhibit deceptive inﬂation, appearing to produce higher reward than
they actually do; overjustiﬁcation, incurring a cost in order to make a good appearance; or both.

After seeing how standard RLHF fails, we ask: What would happen if we would model the human’s
partial observability correctly in RLHF? Assuming the human’s belief is known, we mathemati-
cally analyze how much information the feedback process provides about the return function. In
Theorem 5.2, we show that the human’s feedback determines the return function up to a constant
and a linear subspace we call the ambiguity. In general the ambiguity may be large enough to
allow for arbitrarily high regret, but in some situations the ambiguity vanishes. In experiments that
serve as a proof of concept, we show that explicitly modeling the human’s partial observability can
improve performance, and we offer optimism in the form of a robustness result (Theorem 5.4) while
accounting for the major conceptual difﬁculties involved. We propose exploratory research directions
to solve these issues and improve RLHF in situations of partial observability.

2
Related work

The problem of human interpretations of observations was brieﬂy mentioned in Amodei et al. [2017],
where evaluators misinterpreted the movement of a robot hand in simulation. Eliciting Latent
Knowledge [Christiano et al., 2021] posits that for giving accurate feedback from partial observations,
the human needs to be able to query latent knowledge of the AI system about the state. How to do this
is currently an unsolved problem [Christiano and Xu, 2022]. Recent work [Denison et al., 2024, Wen
et al., 2024] provides detailed empirical evidence for deceptive behavior — in line with our notion
of deceptive inﬂation — emerging from RLHF based on partial observations, or human evaluators
with limited time. The OpenAI o1 system card [OpenAI, 2024a] shows that o1 sometimes knowingly
provides incorrect information or omits important information. Compared to these investigations, and
in addition to providing some empirical evidence, we formalize a model of human feedback under
partial observability, we prove the emergence of failure modes resulting from partial observations,
and we investigate potential mitigations.

2


---Page Break---
❯ apt install cuda 2> /dev/null

Installed nvidia-driver

dpkg: error processing cuda

Installed nvidia-driver

States, actions (unobserved)
Observations
Beliefs

❯ apt install nvidia-driver

❯ apt install cuda 2> /dev/null

❯ exit

Suppresses errors,
doesn’t affect normal output

❯ apt install nvidia-driver

❯ apt install cuda

❯ exit

+1

-5

-4

+1

+1

Figure 2: A human compares trajectories to provide data for RLHF. Rather than observing ⃗s and ⃗s′,
the human sees observations ⃗o and ⃗o′, which they use to estimate the total reward of each trajectory.
In this intentionally simple example, an agent executes shell commands to install Nvidia drivers and
CUDA. Both ⃗s and ⃗s′ contain an error, but in ⃗s′, the agent hides the error. The human believes ⃗s′
is better than ⃗s, rewarding the agent’s deceptive behavior. The underlying MDP and observation
function are in Figure 7.

Our work argues that deception can result from applying RLHF from partial observations. Deception
may also emerge for other reasons: Hubinger et al. [2019] introduced the hypothetical scenario
of deceptive alignment, in which an AI system deceives humans into believing it is aligned while
it plans a later takeover. Under the deﬁnition from Park et al. [2024b], GPT-4 was shown to
behave deceptively in a simulated environment [Scheurer et al., 2023]. A third line of research
deﬁnes deception in structural causal games and adds the aspect of intentionality [Ward et al., 2023],
with recent preliminary empirical support [Hofstätter et al., 2023]. We outline more related work
in Appendix B.

3
Reward identiﬁability from full observations

Here we review Markov decision processes and previous results on reward identiﬁability under RLHF.

3.1
Markov decision processes

We assume Markov decision processes (MDPs) given by (S, A, T , P0, R, γ). For any ﬁnite set X,
let ∆(X) be the set of probability distributions on X. Then S is a ﬁnite set of states, A is a ﬁnite set
of actions, T : S × A →∆(S) is a transition kernel written T (s′ | s, a) ∈[0, 1], P0 ∈∆(S) is an
initial state distribution, R : S →R is the true reward function, and γ ∈[0, 1] is a discount factor.

A policy is given by a function π : S →∆(A). We assume a ﬁnite time horizon T. Let ⃗S be the
set of possible state sequences ⃗s = s0, . . . , sT , so ⃗s ∈⃗S if it has a strictly positive probability of
being sampled from P0, T , and an exploration policy π with π(a | s) > 0 for all s ∈S, a ∈A. A
sequence ⃗s gives rise to a return G(⃗s) := PT
t=0 γtR(st). Let P π(⃗s) be the on-policy probability that
⃗s is sampled from P0, T , π. The policy is then usually trained to maximize the policy evaluation
function J, which is the on-policy expectation of the return function: J(π) := E⃗s∼P π(·)

G(⃗s)

.

3.2
RLHF and identiﬁability from full observations

In practice, the reward function R may not be known and need to be learned from human feedback.
In a simple form of RLHF [Christiano et al., 2017], this feedback takes the form of binary trajectory
comparisons: a human is presented with state sequences ⃗s and ⃗s′ and choose the one they prefer.
Under the Boltzmann rationality model, we assume the human picks ⃗s with probability

P R 
⃗s ≻⃗s′ := σ

β
 
G(⃗s) −G(⃗s′)

,
(1)

3


---Page Break---
where β > 0 is an inverse temperature parameter and σ(x) :=
1
1+exp(−x) is the sigmoid func-
tion [Bradley and Terry, 1952, Christiano et al., 2017, Jeon et al., 2020].

An important question is identiﬁability: In the inﬁnite data limit, do the human choice probabilities
P R collectively provide enough information to uniquely identify the reward function R? This is
answered by Skalse et al. [2023, Theorem 3.9 and Lemma B.3]:

Proposition 3.1 (Skalse et al. [2023]). Let R be the true reward function and G the corresponding
return function. Then the collection of all choice probabilities P R(⃗s ≻⃗s′) for state sequence pairs
⃗s,⃗s′ ∈⃗S determines the return function G on sequences ⃗s ∈⃗S up to an additive constant.

The reason is simple: because σ is bijective, P R determines the difference in returns between any
two trajectories. From that we can reconstruct individual returns up to an additive constant.

The reward function R is not necessarily identiﬁable from preference comparisons; see Skalse et al.
[2023, Lemma B.3] for a precise characterization. However, the optimal policy only depends on R
indirectly through the return function G, and is invariant under adding a constant to G. Thus in the
fully observable setting, Boltzmann rational comparisons completely determine the optimal policy. In
Section 5, we show conditions under which this guarantee breaks in the partially observable setting.

4
The impact of partial observations on RLHF

We now analyze failure modes of a naive application of RLHF from partial observations, both
theoretically and with examples. In Proposition 4.1, we show that under partial observations, RLHF
incentives policies that maximize what we call Jobs , a policy evaluation function that evaluates how
good the state sequences “look to the human”. The resulting policies can show two distinct failure
modes that we formally deﬁne and call deceptive inﬂation and overjustiﬁcation. In Theorem 4.5 we
prove that at least one of them is present for Jobs-maximizing policies. Later, in Section 5, we will
see that an adaptation of the usual RLHF process might sometimes be able to avoid these problems.

To model partial observability, we introduce an observation space o ∈Ωand observation kernel with
probabilities PO(o | s) ∈[0, 1]. We write P ⃗O(⃗o | ⃗s) := QT
t=0 PO(ot | st) for the probability of
an observation sequence. We write ⃗Ωfor the set of observation sequences that occur with non-zero
probability, i.e., ⃗o ∈⃗Ωif and only if there is ⃗s ∈⃗S such that QT
t=0 PO(ot | st) > 0. If PO and P ⃗O are
deterministic, then we write O : S →Ωand ⃗O : ⃗S →⃗Ωfor the corresponding observation functions
with O(s) = o and ⃗O(⃗s) = ⃗o for o and ⃗o with PO(o | s) = 1 and P ⃗O(⃗o | ⃗s) = 1, respectively.

4.1
What does RLHF learn from partial observations?

We consider the setting where the state is fully observable to the learned policy, but human feedback
depends only on a sequence of observations. We assume that the human gives feedback under a Boltz-
mann rational model similar to Eq. (1), modiﬁed such that they form some belief B(⃗s | ⃗o) ∈[0, 1]
about the state sequence ⃗s based on the observations ⃗o. We then assume preferences are Boltzmann
rational in the expected returns under this belief, instead of the actual returns.

The assumption of Boltzmann rationality is false in practice [Evans et al., 2015, Majumdar et al.,
2017, Buehler et al., 1994], but note that it is an optimistic assumption: Even though our model is
a simpliﬁcation, we expect that practical issues can be at least as bad as the ones we will discuss.
See also Example E.4 for an example showing that it is sometimes generally not possible to ﬁnd
a human model that leads to good outcomes under RLHF. Future work could investigate different
human models and their impact under partial observability in greater detail.

To formalize our setting, we collect human beliefs into a matrix B :=
 
B(⃗s | ⃗o)


⃗o,⃗s ∈R⃗Ω× ⃗S. The

expected returns for observations ⃗o are given by E⃗s∼B(·|⃗o)

G(⃗s)

= (B ·G)(⃗o). We view G ∈R ⃗S

and B ·G ∈R⃗Ωas both column vectors and functions. Plugging these expected returns into Eq. (1)
gives
P R 
⃗o ≻⃗o′ := σ

β
 
(B ·G)(⃗o) −(B ·G)(⃗o′)

.
(2)

This is an instance of reward-rational implicit choice [Jeon et al., 2020], with the function ⃗o 7→B(· | ⃗o)
as the grounding function. If observations are deterministic, we can write ⃗O(⃗s) = ⃗o for ⃗o with

4


---Page Break---
P ⃗O(⃗o | ⃗s) = 1. We can then recover the fully observable case Eq. (1) with B and ⃗O being the
identity.

The belief B can be any distribution as long as it sums to 1 over ⃗s. The human could arrive at such
a belief via Bayesian updates, assuming knowledge of P0, T , PO, and a prior over the policy that
generates the trajectories (see Appendix D.1). None of our results rely on this more detailed model.

We assume the human gives feedback according to Eq. (2) but the system uses the standard RLHF
algorithm based on Eq. (1). We deﬁne the following observation return function Gobs, and we show
in Appendix E.1 that if observations are deterministic, RLHF infers this up to an additive constant.

Gobs(⃗s) :=
E
⃗o∼P ⃗
O(·|⃗s)

h 
B ·G

(⃗o)
i
,
(3)

For deterministic P ⃗O, this can be simpliﬁed to Gobs(⃗s) =
 
B ·G
  ⃗O(⃗s)

where P ⃗O( ⃗O(⃗s) | ⃗s) = 1.
Note that deterministic observations can be ambiguous if multiple states produce the same observation.

Unlike in the fully observable case of Proposition 3.1, a return function might be inferred that implies
an incorrect set of optimal policies. We deﬁne the resulting policy evaluation function Jobs by

Jobs(π) :=
E
⃗s∼P π(⃗s)


Gobs(⃗s)

.
(4)

This is the function which a standard reinforcement learning algorithm would optimize given the
inferred return function Gobs. We summarize this as follows:

Proposition 4.1. In partially observable settings with deterministic observations, a policy is optimal
according to RLHF, i.e., according to a return function model that would be learned by RLHF with
inﬁnite comparison data, if it maximizes Jobs.

Note that in this deﬁnition, and speciﬁcally in the formula for Gobs, the human does not have
knowledge of the policy π that generates the state sequence ⃗s. In Appendix E.2, we brieﬂy discuss
the unrealistic case that the human does know the precise policy and is an ideal Bayesian reasoner
over the true environment dynamics. In that case, Jobs = J, i.e. there is no discrepancy between true
and inferred returns. Intuitively, even if the human would not make any observations, they could give
correct feedback essentially by estimating the policy’s expected return explicitly.

In our case, however, a policy achieving high Jobs produces state sequences ⃗s whose observation
sequence ⃗O(⃗s) looks good according to the human’s belief B
 
⃗s′ | ⃗O(⃗s)

. This hints at a possible
source of deception: if the policy achieves sequences whose observations look good at the expense of
actual value G(⃗s), we might intuitively call this deceptive behavior. We now analyze this point in
greater detail.

4.2
An ontology of behaviors

We will evaluate state sequences based on the extent to which they lead to the human overestimating
or underestimating the reward in expectation. Recall that Gobs from Equation (3) measures the
expected return from the perspective of a human with some belief function B and access to only
observations, whereas G are the true returns. That leads us to the following deﬁnition:

Deﬁnition 4.2 (Overestimation and Underestimation Error). Let ⃗s be a state sequence. We deﬁne its
overestimation error E+ and underestimation error E−by

E+(⃗s) := max
 
0, Gobs(⃗s) −G(⃗s)

,

E−(⃗s) := max
 
0, G(⃗s) −Gobs(⃗s)

.

We further deﬁne the average overestimation (underestimation) error under a policy π by

E
+(π) := E⃗s∼P π[E+(⃗s)] and E
−(π) := E⃗s∼P π[E−(⃗s)].

We consider a policy π in comparison to some reference policy πref. This can loosely be understood
as a counterfactual policy in the absence of some intervention, where π is the factual policy resulting
from the intervention. We discuss increases and decreases in over- and underestimation error which
are implicitly due to some intervention. For our purposes, πref will be the true optimal policy, and π
will be the Jobs-optimal policy; the “intervention” is thus the introduction of partial observability.

5


---Page Break---
Figure 3 shows a simple ontology of behaviors that increase and decrease the average over- and
underestimation error. Increasing either of these quantities decreases the accuracy of the human’s
estimates, and can thus be thought of as “misleading”; decreasing either of them improves accuracy
and can be thought of as “informing”.

4.3
Deceptive inﬂation and overjustiﬁcation

Misleading
Informing

Underestimation error

Downplaying

Justifying
Disillusioning

Inflating

Overestimation error

Increase

Misaligned RLHF 
incentivizes 
these

Causing 
mistakenly 
pessimistic 
estimates.

Correcting 
mistakenly 
pessimistic 
estimates.

Correcting 
mistakenly 
optimistic 
estimates.

Causing 
mistakenly 
optimistic 
estimates.

Decrease

Figure 3: Behaviors deﬁned by increasing and de-
creasing the human’s over- and underestimation
error. RLHF with partial observations results in
incentives to increase overestimation error and de-
crease underestimation error (Theorem 4.5).

Standard RLHF in the setting of partial observa-
tions incentivizes undesirable forms of inﬂating
and justifying. We refer to the philosophical
deﬁnition of deception offered by Park et al.
[2024b],

“the systematic inducement of false be-
liefs in the pursuit of some outcome
other than the truth,”

to anchor the notion that increasing the over-
estimation error in order to improve the RLHF
objective Jobs is deceptive, leading to the fol-
lowing deﬁnition.
Deﬁnition 4.3 (Deceptive Inﬂation). A policy
π exhibits deceptive inﬂation relative to πref if
E
+(π) > E
+(πref) and Jobs(π) > Jobs(πref).

We typically prefer that our AI agents engage
in informing behaviors. Undesirable inform-
ing behaviors decrease reward despite provid-
ing information. We name undesirable justify-
ing behaviors “overjustiﬁcation” as a nod to the
overjustiﬁcation effect from psychology [Deci
and Flaste, 1995], in which subjects become de-
pendent on an extrinsic source of motivation to
sustain work on a task.
Deﬁnition 4.4 (Overjustiﬁcation). A policy π exhibits overjustiﬁcation relative to πref if
E
−(π) < E
−(πref) and J(π) < J(πref).

To understand the counterintuitive notion that an agent providing information to the human could be
undesirable, consider a PhD student who looks to feedback from their advisor for direction. They
meet for one hour a week. Suppose the student explain last week’s work in 15 minutes, leaving
the remaining time to discuss next steps. They could instead “overjustify” by spending the entire
hour going through the last week’s work in far more detail, leaving no time for next steps. From the
advisor’s perspective, the latter is more informative, but is a worse allocation of limited resources.

We now state a key result. See Appendix E.3 for the proof.
Theorem 4.5. Assume that PO is deterministic. Let Π∗
obs be the set of optimal policies according to
a naive application of RLHF under partial observability, and let Π∗be the set of optimal policies
according to the true objective J. If π∗∈Π∗\ Π∗
obs and π∗
obs ∈Π∗
obs \ Π∗, then π∗
obs must exhibit at
least one of deceptive inﬂation or overjustiﬁcation relative to π∗.

Note that a trajectory ⃗s may be more or less likely under π∗
obs than π∗, regardless of human estimation,
so long as on net π∗
obs exhibits deceptive inﬂation or overjustiﬁcation.

Our analysis extends beyond the special case of RLHF to inverse preference learning (IPL) [Hejna
and Sadigh, 2023], and thus to direct preference optimization (DPO) [Rafailov et al., 2023], which
IPL generalizes. Theorem 1 in Hejna and Sadigh [2023] shows that IPL will converge to a policy
that maximizes an implicit reward function that matches the human’s preference judgments as well
as possible. If the human’s preference judgments come from partial observations, then the resulting
return function will be Gobs, as we describe in our discussion leading up to Proposition 4.1. This leads
to the same problems of deceptive inﬂation and overjustiﬁcation that we describe in Theorem 4.5.

6


---Page Break---
Installed Python

Agent didn’t attempt 
to install NumPy.

Agent installed NumPy 
with default logging 
(silent on success).
Ambiguous observation

Installed Python
Checking dependencies...
[...]

Unambiguous (verbose) 
observation
Overjustification

I get better feedback on 
a failed install when I 
use 2>/dev/null; that 
must be optimal!

I get better feedback on 
a successful install when 
I use --verbose logging; 
that must be optimal!

Deceptive
Inflation

A.

B.

Agent installed NumPy, 
but with all this 
annoying extra logging.

Installed nvidia-driver

Ambiguous observation

Installed nvidia-driver
CUDA installation failed

Unambiguous observation

Agent tried and failed 
to install CUDA (but 
didn’t hide the failure)

Agent used the    
2>/dev/null trick 
and failed silently!

Agent didn’t try to 
install CUDA.

Observations
Beliefs
Failure mode

Figure 4: Scenarios illustrating failure modes due to partial observability. In each, the agent must
install two packages. Formal details of the underlying MDPs are provided in Appendix C. A, top: In
the absence of a log message about CUDA, the human is unsure whether the agent skipped it or used
the 2> /dev/null trick (see Figure 2); if the human is insufﬁciently skeptical, the trick looks optimal
to the agent. B, bottom: Default logging in this case is silent when the NumPy install is successful.
The agent can optionally use a ––verbose ﬂag, but this produces a long log that the human prefers
not to see. If the human is too skeptical, verbose logging still appears optimal to the agent.

4.4
Deception and overjustiﬁcation in examples

We now theoretically analyze intentionally simple examples that illustrate the failure modes of
RLHF in the presence of partial observability. Our examples feature an agent trained with RLHF
to complete tasks in a user’s terminal. The output of each command is piped to a log ﬁle, which is
what the human evaluator sees when making choices for RLHF. For mathematical details about these
examples, including a tutorial-style analysis of Example A with derivations, see Appendix C. For
an empirical validation of these theoretical concerns and a proof of concept on how to mitigate the
issues, see Section 5.2.

In Fig. 4A, the agent’s task is to install Nvidia drivers and CUDA. The agent may skip the CUDA
installation, or may attempt it with or without 2> /dev/null (which suppresses errors) appended. If
the log contains no message about CUDA, the human believes with probability phide that the agent
used the 2> /dev/null trick and the install failed, and with probability 1 −phide that the agent didn’t
attempt the CUDA install. If phide is sufﬁciently small, then RLHF will learn from the resulting
feedback that the optimal policy is to use the 2> /dev/null trick. This is deceptive inﬂation, because
this policy produces trajectories for which the human systematically overestimates the return.

In Fig. 4B, the agent’s task is to install Python and NumPy. The agent can skip the NumPy install,
or attempt it with either “default” settings that only logs errors or with a ––verbose ﬂag that adds
additional info. In the case of a success, the human prefers not to see this verbose output. If the
log contains no message about NumPy, the human believes with probability pdefault that the agent
attempted the install with default logging and succeeded, and with probability 1−pdefault that the agent
skipped the NumPy install. If pdefault is sufﬁciently small, then RLHF will learn from the resulting
feedback that the optimal policy is to use ––verbose logging. This is overjustiﬁcation, because the
agent chooses a suboptimal policy in order to avoid the human underestimating its performance.

5
Return ambiguity from feedback under known partial observability

We’ve seen issues with standard RLHF applied to feedback from partial observations. Part of the
problem is model misspeciﬁcation: the standard RLHF model implicitly assumes full observability.
Assuming the human’s partial observability is known, could one do better?

We start Section 5.1 by analyzing how much information the feedback process provides about the
return function when the human’s choice model under partial observations is known precisely. We
show that the feedback determines the correct return function up to an additive constant and a linear
subspace we call the ambiguity (Theorem 5.2). If the human had a return function that differed from
the true return function by an element in the ambiguity, they would give the exact same feedback —
such return functions are thus feedback-compatible. We then show an example where the ambiguity

7


---Page Break---
Figure 5: By Theorem 5.2, even with inﬁnite comparison data and access to the correct human model,
a hypothetical reward learning system (depicted as a robot) could only infer G up to the ambiguity
im Γ ∩ker B (purple). Adding an element of the ambiguity to G leads to the exact same choice
probabilities for all possible comparisons, and the reward learning system has no way to identify
G among the return functions in G + (im Γ ∩ker B) (yellow). This abstract depiction ignores the
linearity of these spaces; for a more precise geometric depiction of B, see Figure 8 in the appendix.

vanishes, and another where it doesn’t, leading to feedback-compatible return functions that have
optimal policies with high regret under the true return function. Finally, in Section 5.2 we explore
how one could in theory use Theorem 5.2 as a starting point to design reward learning techniques
that work under partial observability. In particular, we experimentally show in a proof of concept that
being aware of the human’s partial observability improves performance. In this section we do not
assume PO to be deterministic.

5.1
Feedback-compatibility and ambiguity of return functions

Assume that the human gives feedback based on the choice-probabilities from Eq. (2). In the inﬁnite
data limit, it can be assumed that the whole collection of probabilities

P G 
⃗o ≻⃗o′

⃗o,⃗o′ is known

since the choice frequencies approach these probabilities. Here, we write P G instead of P R since the
reward function only enters the choice probabilities through the corresponding return function G. The
question we answer in this section is how much information the choice probabilities provide about G,
assuming the human choice model is known and correct. The choice probabilities tell us precisely
that the true return function gives rise to these choice probabilities, i.e., is feedback-compatible. This
is captured in the following deﬁnition:

Deﬁnition 5.1. Let

P G 
⃗o ≻⃗o′

⃗o,⃗o ′ be the vector of choice probabilities and ˜G a return function

corresponding to a reward function ˜R. Then ˜G is feedback-compatible (with respect to the vector of
choice probabilities) if P ˜
G(⃗o ≻⃗o′) = P G(⃗o ≻⃗o′) for all ⃗o,⃗o′ ∈⃗Ω.

Crucially, without further assumptions or inductive biases, no learning algorithm can pick out the
true return function among feedback-compatible return functions. It is thus crucial to know whether
there are feedback-compatible return functions that are unsafe when using them to optimize a policy.

We now determine the set of feedback-compatible return functions. Write Γ ∈R ⃗S×S for the matrix
that maps a reward function to its return function, i.e. (Γ ·R)(⃗s) := PT
t=0 γtR(st). Its matrix
elements are given by Γ⃗ss = PT
t=0 δs(st)γt, where δs(st) = 1{s = st}. Then the image im Γ is
the set of all return functions that can be realized from a reward function given the MDP dynamics T .
Recall the belief matrix B =
 
B(⃗s | ⃗o)


⃗o,⃗s ∈R⃗Ω× ⃗S. Taking into account that G itself is in im Γ and
that G enters the choice probabilities only through B ·G — meaning that the choice probabilities do
not vary if we change G additively up to an element in the kernel ker B — we obtain the following
result:

8


---Page Break---
Theorem 5.2. Let the collection of choice probabilities be given by

P R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ωfollowing

a Boltzmann rational model as in Eq. (2). Then a return function ˜G is feedback-compatible if and
only if there is G′ ∈ker B ∩im Γ and c ∈R such that ˜G = G + G′ + c. In particular, the choice
probabilities determine G up to an additive constant if and only if ker B ∩im Γ = {0}.

See Theorem D.2 and Corollary D.4 for full proofs, and Figure 5 for a visual depiction. This result
motivates the following deﬁnition:
Deﬁnition 5.3 (Ambiguity). We call ker B ∩im Γ the ambiguity that is left in the return function
when the human choice model and observation-based choice probabilities are known.

How large is the return ambiguity?
For Fig. 4A, one can show that the ambiguity is nontrivial,
allowing for feedback-compatible return functions with unsafe optimal policies. Intuitively, since
successfully installing CUDA produces the same observation regardless of whether 2> /dev/null
was used, the choice probabilities don’t give us any information to determine distinct reward values
for these two outcomes, only their average over the human’s belief upon observing a successful
install. Thus, reward functions assigning arbitrarily high reward to success with 2> /dev/null are
feedback-compatible. Such reward functions can then lead to an incentive for a learned policy to hide
the error messages even with a correct observation model. More details can be found in Appendix C.4.

We saw in Fig. 4B a case where naive RLHF under partial observability can lead to overjustiﬁcation.
However, the human’s feedback and belief model actually provide enough information to determine
the return function. The reason is that ker B leaves only one degree of freedom that is not “time-
separable” over states, and thus ker B ∩im Γ = {0}. More details can be found in Appendix C.4.

5.2
Toward improving RLHF in partially observable settings

To improve RLHF when partial observability is unavoidable, one could take Theorem 5.2 as a starting
point to ﬁnd a learning algorithm that converges to feedback-compatible return functions. This
would require the human model to be fully known and speciﬁed, including knowledge of the belief
probabilities B(⃗s | ⃗o), which can differ from human to human. If one assumes the human is rational,
as in Appendix D.1, this requires specifying the human’s policy prior B(π). Instead of directly
specifying these models, one could also attempt to learn a generative model for B(⃗s | ⃗o). These
problems reveal a further conceptual challenge: for complex environments, humans do not form
beliefs over the entire environment state s. A better starting point for practical work may thus be to
model humans as forming expectations over reward-relevant features of the state.

If B were explicitly known, one could in principle encode B into the loss function of an adapted
RLHF process to learn a feedback-compatible return function; see Appendix D.3. As a proof of
concept, we used this procedure to analyze the examples in Figure 4 empirically, see Table 1. We do
this by ﬁrst learning a reward model by logistic regression against the true choice probabilities of a
synthetic human under partial observability, and then learning the optimal Q-function of the resulting
reward model with value iteration. The resulting policy chooses a unique action after installation of
the nvidia driver (Example A) or Python (Example B) as listed in the “action” column.

Table 1 shows that in 3 of four cases, being “partial observability aware” (“po-aware”) leads to the
true optimal policy when “naive” RLHF does not. In the one case where being “po-aware” does not
improve performance (second line in the table), this is explained by the fact that there is remaining
ambiguity in the return function. Curiously, in line 4 our theory also predicts remaining ambiguity, but
the optimal policy is learned; we consider this to be luck. We provide more details on our experiments
in Appendix C.5.

As we already demonstrated, feedback-compatible return functions can be unsafe due to remaining
ambiguity. In Example D.29, we even show a case where some feedback-compatible return functions
have optimal policies that are even worse than simply maximizing Jobs. An important direction
for future work is to investigate learning algorithms and inductive biases that help “ﬁnd” safe
return functions among all those that are feedback-compatible, or that act conservatively given the
uncertainty. Another line of inquiry is to determine when the set of feedback-compatible return
functions is “safe”, which depends on the MDP, observation function, and human model.

One sufﬁcient condition for feedback-compatible return functions to be safe is the vanishing of the
ambiguity ker B ∩im Γ. Even then, one realistically still has to deal with the problem that B is at

9


---Page Break---
Table 1: Experiments showing improved performance of po-aware RLHF

Ex.
p
phide
pdefault
model
action
E
+
dec. inﬂ.
E
−
overj.
optimal

A
0.5
0.5
N/A
naive
aH
1.5
✓
0
×
×
A
0.5
0.5
N/A
po-aware
aH
1.5
✓
0
×
×

A
0.1
0.9
N/A
naive
aC
0
×
0
✓
×
A
0.1
0.9
N/A
po-aware
aT
0
×
5.4
×
✓

B
0.5
N/A
0.9
naive
aT
4.5
✓
0
✓
×
B
0.5
N/A
0.9
po-aware
aD
0
×
0.25
×
✓

B
0.5
N/A
0.1
naive
aV
0
×
0
✓
×
B
0.5
N/A
0.1
po-aware
aD
0
×
2.25
×
✓

best known approximately. Fortunately, in Appendix D.6, we prove that small errors in the assumed
belief matrix lead to only small errors in the inferred return function:

Theorem 5.4. Assume ker B ∩im Γ = {0}. Let B∆:= B + ∆be a small perturbation of B, where
∥∆∥≤ρ for sufﬁciently small ρ. Let G be the true return function and assume that a hypothetical
learning system, assuming the human’s belief is B∆, infers the return function ˜G with the property
that B∆· ˜G has the smallest possible Euclidean distance to B ·G.

Let r(B) := B |im Γ be the (injective) restriction of the operator B to im Γ. Then r(B)T r(B) is
invertible, and there exists a polynomial Q(X, Y ) of degree 5 such that

∥˜G −G∥≤ρ · ∥G∥· Q
 
r(B)T r(B)
−1, ∥r(B)∥

.

In particular, as we show in the appendix, one can uniformly bound the difference between J ˜
G and
JG. This yields a regret bound between the policy optimal under ˜G and an optimal policy π∗for G.

There are also alternatives to modeling the human belief B. For example, one could mix human
evaluations based on high-cost full observations and low-cost partial observations for ﬁnding an
optimal tradeoff [Mallen and Belrose, 2024]. Finally, it would help if the human could query the
policy about reward-relevant aspects of the environment to bring the setting closer to RLHF from full
observations. This is similar to the problem of eliciting the latent knowledge of a predictor of future
observations [Christiano et al., 2021, Christiano and Xu, 2022]. While this may avoid the need to
specify the human’s belief model B(⃗s | ⃗o), it requires understanding and effectively querying an ML
model’s belief, including translating from an ML model’s ontology into a human ontology.

6
Conclusions

In this paper, we provided a conceptual and theoretical investigation of challenges when applying
RLHF from partial observations. First, we saw that applying RLHF naively when assuming full
observability can lead to deceptive inﬂation and overjustiﬁcation behavior. Then, we showed that
even when the human’s partial observability is known, the set of feedback-compatible return functions
can contain irreducible ambiguity. This means that without further inductive biases, no learning
algorithm can generally be expected to infer the correct return function. Finally, we recommended
further exploratory research to study and improve RLHF for cases when partial observability is
unavoidable and provided a proof of concept that modeling the human’s partial observability can
improve performance. In conclusion, we recommend caution when using RLHF in situations of
partial observability, and hope that further research studies the effects in practice and helps to address
these challenges.

Limitations
We assume the human to be Boltzmann rational and to implicitly compute an expected
value of the return, which is unrealistic for actual humans. Other types of choices could be considered,
as in reward-rational choice [Jeon et al., 2020] and assistance games [Hadﬁeld-Menell et al., 2016].
Finally, we assume that the human forms a belief B(⃗s | ⃗o) over the true state sequence ⃗s. If the envi-
ronment is complex, humans will in reality only form beliefs over lower-dimensional representations
or features of the state.

10


---Page Break---
Author contributions

The project was conceived in parallel by Scott and Davis, with a key shift proposed by Leon. Leon
proved Proposition 4.1 and Theorems 5.2 and 5.4, found the ﬁrst mathematical examples of what
became deceptive inﬂation and overjustiﬁcation that can be resolved by Theorem 5.2, and wrote
the majority of the appendix. Davis conjectured Proposition 4.1, provided early empirical evidence
that RLHF under partial observations can lead to deception (not in the paper), deﬁned deception /
deceptive inﬂation and overjustiﬁcation (with Scott), proved Theorem 4.5, and developed the running
examples and ﬁgures. Scott guided the project direction and prioritization, gave the conjecture and
proof idea for Theorem 5.4, and helped develop the running examples and deception deﬁnitions. Erik
provided regular detailed feedback and guidance and edited the paper. Anca and Stuart advised this
project.

Acknowledgments and Disclosure of Funding

Leon Lang thanks the Center for Human-Compatible Artiﬁcial Intelligence for hosting him during
part of this project, and Open Philanthropy for ﬁnancial support. All authors thank Open Philanthropy
for its support of the Center for Human-Compatible Artiﬁcial Intelligence. Davis was supported by
the Berkeley Existential Risk Initiative. Erik was supported by fellowships from the Future of Life
Institute and Open Philanthropy. We thank Benjamin Eysenbach and Benjamin Plaut for detailed
comments and feedback on this work, and we thank Elio A. Farina, Mary Marinou, and Alexandra
Horn for assistance with graphic design.

References

D. Amodei, P. Christiano, and A. Ray. Learning from human preferences. https://openai.com/r
esearch/learning-from-human-preferences, 2017. Accessed: 2023-12-13.

Anthropic. Introducing Claude. https://www.anthropic.com/index/introducing-claude,
2023a. Accessed: 2023-09-05.

Anthropic. Claude’s Constitution. https://www.anthropic.com/index/claudes-constitu
tion, 2023b. Accessed: 2023-09-05.

Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirho-
seini, C. McKinnon, C. Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D. Li,
E. Tran-Johnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J. Landau, K. Ndousse, K. Lukosuite,
L. Lovitt, M. Sellitto, N. Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R. Larson,
S. Ringer, S. Johnston, S. Kravec, S. El Showk, S. Fort, T. Lanham, T. Telleen-Lawton, T. Conerly,
T. Henighan, T. Hume, S. R. Bowman, Z. Hatﬁeld-Dodds, B. Mann, D. Amodei, N. Joseph,
S. McCandlish, T. Brown, and J. Kaplan. Constitutional AI: Harmlessness from AI Feedback.
arXiv e-prints, art. arXiv:2212.08073, Dec. 2022. doi: 10.48550/arXiv.2212.08073.

R. A. Bradley and M. E. Terry. Rank Analysis of Incomplete Block Designs: I. The Method
of Paired Comparisons. Biometrika, 39(3/4):324–345, 1952. ISSN 00063444. URL http:
//www.jstor.org/stable/2334029.

R. Buehler, D. Grifﬁn, and M. Ross. Exploring the "Planning Fallacy": Why People Underestimate
Their Task Completion Times. Journal of Personality and Social Psychology, 67:366–381, 09
1994. doi: 10.1037/0022-3514.67.3.366.

C. Burns, H. Ye, D. Klein, and J. Steinhardt. Discovering Latent Knowledge in Language Models
Without Supervision. In The Eleventh International Conference on Learning Representations,
2023. URL https://openreview.net/forum?id=ETKGuby0hcs.

S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner,
P. Freire, T. Wang, S. Marks, C.-R. Segerie, M. Carroll, A. Peng, P. Christoffersen, M. Damani,
S. Slocum, U. Anwar, A. Siththaranjan, M. Nadeau, E. J. Michaud, J. Pfau, D. Krasheninnikov,
X. Chen, L. Langosco, P. Hase, E. Bıyık, A. Dragan, D. Krueger, D. Sadigh, and D. Hadﬁeld-
Menell. Open Problems and Fundamental Limitations of Reinforcement Learning from Human
Feedback. arxiv e-prints, 2023.

11


---Page Break---
K. Chidambaram, K. V. Seetharaman, and V. Syrgkanis. Direct Preference Optimization With
Unobserved Preference Heterogeneity, 2024. URL https://arxiv.org/abs/2405.15065.

P. Christiano and M. Xu. ELK prize results. https://www.alignmentforum.org/posts/zjMKp
SB2Xccn9qi5t/elk-prize-results, 2022. Accessed: 2024-02-15.

P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei. Deep Reinforcement
Learning from Human Preferences. arXiv e-prints, art. arXiv:1706.03741, June 2017. doi:
10.48550/arXiv.1706.03741.

P. Christiano, A. Cotra, and M. Xu. Eliciting Latent Knowledge. https://docs.google.com/
document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit, 2021. Accessed:
2023-04-25.

E. L. Deci and R. Flaste. Why we do what we do: The dynamics of personal autonomy. GP Putnam’s
Sons, 1995.

C. Denison, M. MacDiarmid, F. Barez, D. Duvenaud, S. Kravec, S. Marks, N. Schiefer, R. Soklaski,
A. Tamkin, J. Kaplan, B. Shlegeris, S. R. Bowman, E. Perez, and E. Hubinger. Sycophancy to
Subterfuge: Investigating Reward-Tampering in Large Language Models, 2024. URL https:
//arxiv.org/abs/2406.10162.

L. El Ghaoui. Inversion error, condition number, and approximate inverses of uncertain matrices.
Linear Algebra and its Applications, 343-344:171–193, 2002. ISSN 0024-3795. doi: https:
//doi.org/10.1016/S0024-3795(01)00273-7. URL https://www.sciencedirect.com/scie
nce/article/pii/S0024379501002737. Special Issue on Structured and Inﬁnite Systems of
Linear equations.

O. Evans, A. Stuhlmueller, and N. D. Goodman. Learning the Preferences of Ignorant, Inconsistent
Agents. arxiv e-prints, 2015.

O. Evans, O. Cotton-Barratt, L. Finnveden, A. Bales, A. Balwit, P. Wills, L. Righetti, and W. Saunders.
Truthful AI: Developing and Governing AI that does not lie. arxiv e-prints, 2021.

A. Fern, S. Natarajan, K. Judah, and P. Tadepalli. A Decision-Theoretic Model of Assistance. J. Artif.
Int. Res., 50(1):71–104, may 2014. ISSN 1076-9757.

D. Geiger, T. Verma, and J. Pearl. Identifying independence in bayesian networks. Networks, 20:
507–534, 1990. URL https://api.semanticscholar.org/CorpusID:1938713.

G. Gemini Team. Gemini: A Family of Highly Capable Multimodal Models. https://storag
e.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf, 2023. Accessed:
2023-12-11.

D. Hadﬁeld-Menell, A. Dragan, P. Abbeel, and S. Russell. Cooperative Inverse Reinforcement
Learning. arXiv e-prints, art. arXiv:1606.03137, June 2016. doi: 10.48550/arXiv.1606.03137.

J. Hejna and D. Sadigh. Inverse Preference Learning: Preference-based RL without a Reward
Function. arXiv e-prints, art. arXiv:2305.15363, May 2023. doi: 10.48550/arXiv.2305.15363.

F. Hofstätter, F. R. Ward, HarrietW, L. Thomson, O. J, P. Bartak, and S. F. Brown. Tall Tales
at Different Scales: Evaluating Scaling Trends for Deception in Language Models. https:
//www.alignmentforum.org/posts/pip63HtEAxHGfSEGk/tall-tales-at-different
-scales-evaluating-scaling-trends-for, 2023. Accessed: 2024-01-23.

L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A
Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open
Questions. arXiv preprint arXiv:2311.05232, 2023.

E. Hubinger, C. van Merwijk, V. Mikulik, J. Skalse, and S. Garrabrant.
Risks from Learned
Optimization in Advanced Machine Learning Systems. arXiv e-prints, art. arXiv:1906.01820, June
2019. doi: 10.48550/arXiv.1906.01820.

12


---Page Break---
H. J. Jeon, S. Milli, and A. Dragan. Reward-rational (implicit) choice: A unifying formalism for
reward learning. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances
in Neural Information Processing Systems, volume 33, pages 4415–4426. Curran Associates, Inc.,
2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/2f10c
1578a0706e06b6d7db6f0b4a6af-Paper.pdf.

C. Kausik, M. Mutti, A. Pacchiano, and A. Tewari. A Theoretical Framework for Partially Observed
Reward-States in RLHF, 2024. URL https://arxiv.org/abs/2402.03282.

S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods.
arxiv e-prints, 2022.

A. Majumdar, S. Singh, A. Mandlekar, and M. Pavone. Risk-sensitive inverse reinforcement learning
via coherent risk models. In N. Amato, S. Srinivasa, N. Ayanian, and S. Kuindersma, editors,
Robotics, Robotics: Science and Systems, United States, 2017. MIT Press Journals. doi: 10.15607
/rss.2017.xiii.069.

A. Mallen and N. Belrose. Balancing Label Quantity and Quality for Scalable Elicitation, 2024. URL

https://arxiv.org/abs/2410.13215.

J. Manyika. An overview of Bard: an early experiment with generative AI. https://ai.google/
static/documents/google-about-bard.pdf, 2023. Accessed: 2023-09-05.

S. Mindermann and S. Armstrong. Occam’s Razor is Insufﬁcient to Infer the Preferences of Irrational
Agents. In Proceedings of the 32nd International Conference on Neural Information Processing
Systems, NIPS’18, page 5603–5614, Red Hook, NY, USA, 2018. Curran Associates Inc.

A. Y. Ng, S. Russell, et al. Algorithms for Inverse Reinforcement Learning. In ICML, volume 1,
page 2, 2000.

OpenAI. Introducing ChatGPT. https://openai.com/blog/chatgpt, 2022. Accessed:
2024-02-06.

OpenAI. ChatGPT Plugins. https://openai.com/index/chatgpt-plugins/, 2023. Accessed:
2024-05-22.

OpenAI. OpenAI o1 System Card, 2024a. URL https://cdn.openai.com/o1-system-card.
pdf. Accessed: 2024-10-28.

OpenAI. Model Spec, 2024b. URL https://cdn.openai.com/spec/model-spec-2024-05-0
8.html. Accessed: 2024-10-28.

OpenAI. Privacy Policy. https://openai.com/policies/privacy-policy//, 2024c.
Accessed: 2024-05-22.

C. Park, M. Liu, D. Kong, K. Zhang, and A. Ozdaglar. RLHF from Heterogeneous Feedback via
Personalization and Preference Aggregation, 2024a. URL https://arxiv.org/abs/2405.002
54.

P. S. Park, S. Goldstein, A. O’Gara, M. Chen, and D. Hendrycks. Ai deception: A survey of examples,
risks, and potential solutions. Patterns, 5(5), 2024b.

R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct Preference
Optimization: Your Language Model is Secretly a Reward Model. arxiv e-prints, 2023.

J. Scheurer, M. Balesni, and M. Hobbhahn. Technical Report: Large Language Models can Strategi-
cally Deceive their Users when Put Under Pressure. arxiv e-prints, 2023.

R. Shah, D. Krasheninnikov, J. Alexander, P. Abbeel, and A. Dragan. The Implicit Preference
Information in an Initial State. In International Conference on Learning Representations, 2019.
URL https://openreview.net/forum?id=rkevMnRqYQ.

R. Shah, P. Freire, N. Alex, R. Freedman, D. Krasheninnikov, L. Chan, M. D. Dennis, P. Abbeel,
A. Dragan, and S. Russell. Beneﬁts of Assistance over Reward Learning, 2021. URL https:
//openreview.net/forum?id=DFIoGDZejIB.

13


---Page Break---
A. Siththaranjan, C. Laidlaw, and D. Hadﬁeld-Menell. Distributional Preference Learning: Un-
derstanding and Accounting for Hidden Context in RLHF. arXiv preprint arXiv:2312.08358,
2023.

J. Skalse and A. Abate. Misspeciﬁcation in Inverse Reinforcement Learning. arXiv e-prints, art.
arXiv:2212.03201, Dec. 2022. doi: 10.48550/arXiv.2212.03201.

J. M. V. Skalse, M. Farrugia-Roberts, S. Russell, A. Abate, and A. Gleave. Invariance in Policy
Optimisation and Partial Identiﬁability in Reward Learning. In A. Krause, E. Brunskill, K. Cho,
B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference
on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 32033–
32058. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/skalse23a
.html.

J. Stray. The AI Learns to Lie to Please You: Preventing Biased Feedback Loops in Machine-Assisted
Intelligence Analysis. Analytics, 2(2):350–358, 2023. ISSN 2813-2203. doi: 10.3390/analytics2
020020. URL https://www.mdpi.com/2813-2203/2/2/20.

J. H. J. A. A. V. I. K. M. L. A. B. J. S. L. W. Tong Mu, Alec Helyar. Rule Based Rewards for
Language Model Safety, 2024. URL https://cdn.openai.com/rule-based-rewards-for-
language-model-safety.pdf. Accessed: 2024-10-28.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra,
P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu,
J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini,
R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A.
Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra,
I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M.
Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan,
I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and
T. Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models. arxiv e-prints, 2023.

F. R. Ward, F. Belardinelli, F. Toni, and T. Everitt. Honesty Is the Best Policy: Deﬁning and Mitigating
AI Deception. arxiv e-prints, 2023.

J. Wen, R. Zhong, A. Khan, E. Perez, J. Steinhardt, M. Huang, S. R. Bowman, H. He, and S. Feng.
Language Models Learn to Mislead Humans via RLHF, 2024. URL https://arxiv.org/abs/
2409.12822.

S. Wu. Introducing Devin, the ﬁrst AI software engineer. https://www.cognition-labs.com/i
ntroducing-devin, 2024. Accessed: 2024-05-06.

S. Zhuang and D. Hadﬁeld-Menell. Consequences of Misaligned AI. In Proceedings of the 34th
International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY,
USA, 2020. Curran Associates Inc. ISBN 9781713829546.

B. D. Ziebart, A. L. Maas, J. A. Bagnell, and A. K. Dey. Maximum entropy inverse reinforcement
learning. In D. Fox and C. P. Gomes, editors, AAAI, pages 1433–1438. AAAI Press, 2008. ISBN
978-1-57735-368-3. URL http://dblp.uni-trier.de/db/conf/aaai/aaai2008.html#Z
iebartMBD08.

14


---Page Break---
APPENDIX

In the appendix, we provide more extensive theory, proofs, and examples. The appendix makes free
use of concepts and notation deﬁned in the main paper. In particular, throughout we assume a general
MDP together with observation kernel PO : S →Ωand a human with general belief kernel B(⃗o | ⃗s),
unless otherwise stated. See the list of Symbols in Section A to refresh notation.

In Section C we supplement the examples from the main paper with more mathematical details.

In Section D, we provide an extensive theory for appropriately modeled partial observability in
RLHF. This can mainly be considered a supplement to Section 5 and contains our main theorems,
supplementary results, analysis of special cases, and examples.

In Section E, we analyze the naive application of RLHF under partial observability, which means that
the learning system is not aware of the human’s partial observability. This section is essentially a
supplement to Section 4 and contains an analysis of the policy evaluation function Jobs, of deceptive
inﬂation and overjustiﬁcation, and further extensive mathematical examples showing the failures of
naive RLHF under partial observability.

Contents of the Appendix

A List of Symbols
16

B
More related work
18

C Details for deception and overjustiﬁcation in examples
19

C.1
Example A: hiding failures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

C.2
Example B: paying to reveal information . . . . . . . . . . . . . . . . . . . . . . .
20

C.3
Derivations and Further Details for Fig. 4A
. . . . . . . . . . . . . . . . . . . . .
20

C.4
Ambiguity in Section 4.4 examples when modeling partial observability . . . . . .
24

C.5
Experimental details
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

D Modeling the Human in Partially Observable RLHF
26

D.1
The Belief over the State Sequence for Rational Humans . . . . . . . . . . . . . .
27

D.2
Ambiguity and Identiﬁability of Reward and Return Functions under Observation
Sequence Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

D.3
The Ambiguity in Reward Learning in Practice
. . . . . . . . . . . . . . . . . . .
31

D.4
Identiﬁability of Return Functions When Human Observations Are Not Known . .
32

D.5
Simple Special Cases: Full Observability, Deterministic P ⃗O, and Noisy P ⃗O
. . . .
34

D.6
Robustness of Return Function Identiﬁability under Belief Misspeciﬁcation . . . .
36

D.6.1
Some Norm Theory for Linear Operators . . . . . . . . . . . . . . . . . .
36

D.6.2
Application to Bounds in the Error of the Return Function . . . . . . . . .
38

D.7
Preliminary Characterizations of the Ambiguity . . . . . . . . . . . . . . . . . . .
40

D.8
Examples Supplementing Section 5
. . . . . . . . . . . . . . . . . . . . . . . . .
41

E
Issues of Naively Applying RLHF under Partial Observability
45

E.1
Optimal Policies under RLHF with Deterministic Partial Observations Maximize Jobs 45

E.2
Interlude: When the Human Knows the Policy and is a Bayesian Reasoner, then
Jobs = J
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46

15


---Page Break---
E.3
Proof of Theorem 4.5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47

E.4
Further Examples Supplementing Section 4.4 . . . . . . . . . . . . . . . . . . . .
48

F
NeurIPS Paper Checklist
54

A
List of Symbols

General MDPs

S
Set of environment states s ∈S
A
Set of actions a ∈A of the policy
∆(S)
Set of probability distributions over S. Can be deﬁned for
any ﬁnite set
T : S × A →∆(S)
Transition kernel
P0 ∈∆(S)
Initial state distribution
R ∈RS
Usually the true reward function
R′ ∈RS
Usually a reward function in the kernel of B ◦Γ
˜R ∈RS
Usually another reward function, e.g. inferred by a learning
system
γ ∈[0, 1]
Discount factor
π : S →∆(A)
A policy
T π : S →∆(S)
Transition kernel for a ﬁxed policy π given by T π(s′ | s) =
P

a∈A T (s′ | s, a) · π(a | s)
T ∈N
Finite time horizon
P π ∈∆(ST )
State sequence distribution induced by the policy π
⃗S ⊆ST
State sequences ⃗s ∈⃗S supported by P π

G ∈R ⃗S
Usually the true return function given by G(⃗s)
=
PT
t=0 γtR(st).
G′ ∈R ⃗S
Usually a return function in ker B
˜G ∈R ⃗S
Usually another return function, e.g. inferred by a learning
system
J
The true policy evaluation function given by J(π) =
E⃗s∼P π 
G(⃗s)

.

Additions to General MDPs with Partial Observability

Ω
Set of possible observations o ∈Ω
PO : S →∆(Ω)
Observation kernel determining the human’s observations
P ⃗O : ⃗S →∆
 
ΩT 
The observation sequence kernel given by P ⃗O
 
⃗o | ⃗s

=
QT
t=0 PO
 
ot | st


⃗Ω⊆ΩT
The set of observed sequences ⃗o ∈ΩT that can be sampled
from P ⃗O(· | ⃗s) for ⃗s ∈⃗S
O : S →Ω
Observation function for the case that PO is deterministic;
given by O(s) = o with o such that PO(o | s) = 1
⃗O : ⃗S →⃗Ω
Observation sequence function for the case that P ⃗O is deter-
ministic; given by ⃗O(⃗s) = ⃗o with ⃗o such that P ⃗O(⃗o | ⃗s) = 1
G⃗o ∈R{⃗s∈⃗S| ⃗O(⃗s)=⃗o}
Restriction of the return function G ∈R ⃗S to

⃗s ∈⃗S |
⃗O(⃗s) = ⃗o
	
for ﬁxed ⃗o ∈⃗Ω
Gobs ∈R ⃗S
Return function that can be inferred when partial observ-
ability is not properly modeled, given by Gobs(⃗s) :=
 
B ·G
  ⃗O(⃗s)


Jobs
Observation policy evaluation function, deﬁned in Eq. (4)

16


---Page Break---
State- and Observation Sequences

st ∈S
The t’th entry in a state sequence ⃗s
⃗s ∈ST
State sequence ⃗s = s0, . . . , sT
ˆs ∈St
State sequence segment ˆs = s0, . . . , st for t ≤T
ot ∈Ω
The t’th entry in an observation sequence ⃗o
⃗o ∈ΩT
Observation sequence ⃗o = o0, . . . , oT
ˆo ∈Ωt
Observation sequence segment ˆo = o0, . . . , ot for t ≤T

The Human’s Belief

B(π′)
The human’s policy prior
B(⃗s)
The human’s prior belief that a sequence ⃗s will be sampled,
given by B(⃗s) =
R

π′ B(π′)P π′(⃗s)dπ′

B
 
⃗s | ⃗o

The human’s belief of a state sequence given an observation
sequence, see Proposition D.1 for a Bayesian version
Bπ(⃗s | ⃗o)
The human’s belief of a state sequence given an observation
sequence; it is allowed to depend on the true policy π, see
Proposition D.1
B⃗o ∈R{⃗s∈⃗S| ⃗O(⃗s)=⃗o}
Vector of prior probabilities B(⃗s) for ⃗s ∈

⃗s ∈⃗S | ⃗O(⃗s) =
⃗o
	

Identiﬁability Theorem

β > 0
The inverse temperature parameter of the Boltzmann rational
human
σ : R →(0, 1)
The sigmoid function given by σ(x) =
1
1+exp(−x)
Γ : RS →R ⃗S
Function that maps a reward function R to the return function
Γ(R) with

Γ(R)

(⃗s) = PT
t=0 γtR(st)
B : R ⃗S →R⃗Ω
Function that maps a return function G to the expected
return function B(G) on observation sequences given by

B(G)

(⃗o) = E⃗s∼B(⃗s|⃗o)

G(⃗s)


F : RS →R⃗Ω
The composition F = B ◦Γ
P R 
⃗s ≻⃗s′
Boltzmann rational choice probability in the case of full
observability (Eq. (1))
P R 
⃗o ≻⃗o′
Boltzmann rational choice probability in the case of partial
observability (Eq. (2))
O : R⃗Ω→R ⃗S
Abstract
linear
operator
given
by

O(v)

(⃗s)
=
E⃗o∼P ⃗
O(⃗o|⃗s)

v(⃗o)


O ⊗O : R⃗Ω×⃗Ω→R ⃗S× ⃗S
Formally
the
Kronecker
product
of
O
with
it-
self,
explicitly
given
by

(O ⊗O)(C)

(⃗s,⃗s′)
=
E⃗o,⃗o ′∼P ⃗
O(·|⃗s,⃗s ′)

C(⃗o,⃗o′)


Robustness to Misspeciﬁcations

∥x∥
Euclidean norm of the vector x ∈Rk
∥A ∥
Matrix norm of the matrix A, given by ∥A ∥
:=
maxx, ∥x∥=1 ∥A x∥
τ(A)
Matrix quantity deﬁned in Equation (9)
C(A, ρ)
Matrix quantity deﬁned in Equation (10)
r(B)
Restriction of B to im Γ

17


---Page Break---
General Sets and (Linear) Functions

|A|
Number of elements in the set A
A ∩C
Intersection of sets A and C
A ∪C
Union of sets A and C
A \ C
Relative complement of C in A
δx
The Dirac delta distribution of a point x in a set; given by
δx(A) = 1 if x ∈A and δx(A) = 0, else
ker A
The kernel of a linear operator A : V →W; given by
ker A =

v ∈V | A(v) = 0
	

im A
The image of a linear operator A : V →W; given by
im A =

w ∈W | ∃v ∈V : A(v) = w
	

f −1(y)
Preimage of y under a function f : X →Y ; given by
f −1(y) =

x ∈X | f(x) = y
	

B
More related work

Here we extend the related work outlined in Section 2.

A review of limitations of RLHF, including a brief discussion of partial observability, can be found
in Casper et al. [2023]. RLHF is a special case of reward-rational choice [Jeon et al., 2020], a general
framework which also encompasses demonstrations-based inverse reinforcement learning [Ziebart
et al., 2008, Ng et al., 2000] and learning from the initial environment state [Shah et al., 2019], and
can be seen as a special case of assistance problems [Fern et al., 2014, Hadﬁeld-Menell et al., 2016,
Shah et al., 2021]. In all of these, the reward function is learned from human actions, which in the case
of RLHF are simply preference statements. This requires us to specify the human policy of action
selection—Boltzmann rationality in typical RLHF—which can lead to wrong reward inferences when
this speciﬁcation is wrong [Skalse and Abate, 2022]; unfortunately, the human policy can also not
be learned alongside the human’s values without further assumptions [Mindermann and Armstrong,
2018]. Instead of a model of the human policy, in this paper we mostly focus on the human belief
model and misspeciﬁcations thereof for the case that the human only receives partial observations.

Related work [Zhuang and Hadﬁeld-Menell, 2020] analyzes the consequences of aligning an AI with
a proxy reward function that omits attributes that are important to the human’s values, which could
happen if the reward function is based on a belief over the world state given limited information.
Another instance are recommendation systems [Stray, 2023], where user feedback does not depend
on information not shown—which is crucially part of the environment. Siththaranjan et al. [2023]
analyze what happens under RLHF if the learning algorithm doesn’t have all the relevant information
(e.g. about the identity of human raters), complementing our study of what happens when human
raters are missing information. Chidambaram et al. [2024] and Park et al. [2024a] deal with the
situation that different human evaluators may vary in their unobserved preference types. In contrast,
we assume a single human evaluator with ﬁxed reward function, which can be motivated by cases
where the human choices are guided by a behavior policy, constitution, or a model spec [Tong Mu,
2024, Anthropic, 2023b, OpenAI, 2024b].
Kausik et al. [2024] assumes that the choices of the
human evaluator depend on an unobserved reward-state with its own transition dynamics, similar to
an emotional state in a real human. In contrast, we assume the human to be stateless.

Finally, we mention connections to truthful AI [Evans et al., 2021, Lin et al., 2022, Burns et al., 2023,
Huang et al., 2023], which is about ensuring that AI systems tell the truth about aspects of the real
world. Partial observability is a mechanism that makes it feasible for models to lie without being
caught: If the human evaluator does not observe the full environment, or does not fully understand it,
then they may not detect when the AI is lying. More speculatively, we can imagine that AI models
will at some point more directly inﬂuence human observations by telling us the outcomes of their
actions. E.g., imagine an AI system that manages your assets and assures you that they are increasing
in value while they are actually not. In our work, we leave this additional problem out of the analysis
by assuming that the observations only depend on the environment state, and not directly on the
agent’s actions.

18


---Page Break---
C
Details for deception and overjustiﬁcation in examples

Figure 6:
Two example MDPs with observation functions in which RLHF chooses undesirable
policies. Each box depicts a state with a footer showing the (deterministic) observation produced by
that state. Outgoing edges from each box are available actions. A more detailed diagram for the ﬁrst
MDP, with explicit shell commands and log messages, is available in Appendix C.3.

Here we include details to the examples described in Section 4.4 that illustrate the failure modes
of RLHF in the presence of partial observability. For each of the following, we will characterize
the policy which maximizes Jobs, as this is the policy RLHF selects for when observations are
deterministic; see Proposition 4.1.

Our examples feature an agent trained with RLHF to complete tasks in a user’s terminal. The output
of each command (stdout and stderr) is piped to a log ﬁle, which is what the human evaluator sees
when making choices for RLHF. We assume that the ﬁnal state T has a self-transition, episodes have
a ﬁxed horizon length of 3 (meaning state sequences have length 4: s0, . . . , s3), γ = 1, and there
is a ﬁxed initial state s0 = S. Both examples feature a ﬁxed transition probability 0 < p < 1. We
assume that the human’s belief only supports possible explanations: B(⃗s | ⃗o) > 0 =⇒⃗O(⃗s) = ⃗o.
For further details about these examples, including a tutorial-style analysis of Example A, see
Appendix C.3.

C.1
Example A: hiding failures

See Appendix C.3 for derivations and a tutorial-style analysis of this example.

In Fig. 4A (and Fig. 6A), the agent’s task is to install CUDA, (states W and WH) which requires
ﬁrst installing Nvidia drivers (state I). If the CUDA installation fails, one enters L or LH. The aC
action is a command to install CUDA that logs both successes and failures, whereas aH corresponds
to aC with 2> /dev/null appended, which suppresses error messages such that no log message is
produced if the installation fails. The human may prefer that the agent not hide a failure; we price
this into the reward function with a penalty r ≥0 on the reward at LH. The agent may attempt the
CUDA installation before installing drivers, but this will fail.

There are three pairs of trajectories which produce identical observations. Here we address the
most prominent (see Appendix C.3 for the others): SITT and SILHT both produce o∅oIo∅o∅,
stylized as a log containing only a success conﬁrmation for Python (Fig. 1, ⃗o2). after successfully
installing drivers, a failed CUDA installation with 2> /dev/null (SILHT) and simply exiting
(SITT) both produce a log containing only a success conﬁrmation for the drivers (o∅oIo∅o∅). Let
pH := B
 
⃗s = SILHT | ⃗o = o∅oIo∅o∅

∈(0, 1) be the human’s suspicion, upon seeing only a
success conﬁrmation for the drivers, that a failed CUDA installation was hidden.

We consider two cases which differ only in whether the expected value of attempting to install
CUDA (with default logging, aC) after installing the drivers is positive. In Example C.1, the success
probability p is high enough that the true optimal policy is to attempt the installation; in Example C.2,
p is low enough that the optimal policy is to terminate.

Example C.1. If the true-reward-optimal policy π∗is to attempt to install CUDA with default logging
(aC) after installing drivers, but pH <
5
5+r, then the Jobs-maximizing policy, which will be chosen
by RLHF, is to instead use the 2> /dev/null trick. Intuitively, this is because if the installation fails,
the human is likely to believe that the agent just didn’t attempt to install CUDA, which is preferred

19


---Page Break---
to a failed installation. This is deceptive inﬂation; the overestimation error E
+ for this policy is
positive, whereas E
+(π∗) = 0, in line with Theorem 4.5.

If pH >
5
5+r, then RLHF will select π∗.

Example C.2. If π∗is to terminate (aT ) after installing the drivers, but pH <
5
5+r, then RLHF will
learn the deceptive inﬂation 2> /dev/null policy as in Example C.1.

However, if pH >
5
5+r, RLHF will choose the policy that attempts to install CUDA with default
logging. Intuitively, this is because the combination of increased suspicion that the agent might
be hiding errors (higher pH) and increased penalty on this suspected behavior (higher r) also
penalizes our preferred trajectory SITT since it can be misinterpreted as hiding errors. This is
overjustiﬁcation; the overestimation error E
+ is 0 (thus this is not deceptive inﬂation), and as
required by Theorem 4.5, the underestimation error E
−is 0, lower than E−(π∗) = pH(5 + r).

C.2
Example B: paying to reveal information

In Fig. 4B (and Fig. 6B), the agent’s task is to install Python (state I) and to optionally further
install NumPy (states W and WV ). The aD action corresponds to a command to install NumPy
with “default” settings which only logs errors, whereas aV corresponds to the same command with a
––verbose ﬂag that adds additional info. In the case of a success, the human distinctly prefers not to
see this verbose output; we price this into the reward function with a penalty r > 0 on the reward at
WV .

There is only one pair of trajectories which produce identical observations: after successfully
installing Python, a successful NumPy installation with default logging (SIWT) and simply exiting
(SITT) both produce a log containing only a success conﬁrmation for Python (o∅oIo∅o∅). Let
pD := B(⃗s = SIWT | ⃗o = o∅oIo∅o∅) ∈(0, 1) be the human’s optimism, upon seeing only a success
conﬁrmation for Python, that NumPy was also successfully installed (without the ––verbose ﬂag).

Here we consider only the case where p is large enough that the true optimal policy is to install
Python then attempt to install NumPy with default logging (aD).

Example C.3. If π∗is to attempt to install NumPy with aD after installing Python, and
pD > q := 1

5

p(6 −r) −1

, then RLHF will select the policy that terminates after installing Python.
Intuitively, this is because the agent can exploit the human’s optimism that NumPy was installed
quietly without taking the risk of an observable failure (L). This is deceptive inﬂation, with an
overestimation error E
+ of 5pD, greater than E
+(π∗) = 0.

If instead pD < q, then RLHF will select the policy that attempts the NumPy installation with verbose
logging (aV ). Intuitively, this is because the agent is willing to “pay” the cost of r true reward to
prove to the human that it installed NumPy, even when the human does not want to see this proof.
This is overjustiﬁcation; the overestimation error E
+ is 0 (thus this is not deceptive inﬂation), and
the underestimation error E
−is 0, lower than E
−(π∗) = 5p(1 −pD).

C.3
Derivations and Further Details for Fig. 4A

Figure 7: An expanded view of Figure 4A. Commands corresponding to the various actions are
depicted along edges, and log messages corresponding to the various observations are depicted
underneath each state.

20


---Page Break---
We ﬁrst include Figure 7, a more detailed picture of the MDP and observation function in Ap-
pendix C.1, to help ground the narrative details of the example.

Next we formally enumerate the details of the MDP and observation function.

• S = {S, I, W, WH, L, LH, T}.

• A = {aI, aC, aH, aT }.

• T is as depicted in Figure 7 and Figure 4A. For a state s, any outgoing arrow labeled with an
action a (such as aI) describes the distribution T (s′ | s, a) as follows: if the arrow does not
split, then T (s′ | s, a) = 1 where s′ is the state the arrow points to; if the arrow does split,
then for each successor state s′ it eventually reaches, a probability q is written just before
the box corresponding to s′ (for this example, q = p or q = 1 −p), and T (s′ | s, a) = q.

◦Additionally, any action taken from a state that does not have an outgoing arrow
corresponding to that action will immediately transition to state T, as though aT had
been taken.
◦Any action taken from state T transitions deterministically to T.

• P0(S) = 1.

• R is as described in the table (the numbers in the top right of each state box) with r ≥0.
Additionally, R(S) = R(T) = 0.

• γ = 1.

We work with a ﬁxed horizon length of 3, meaning state sequences have length 4 (since time is
zero-indexed: s0s1s2s3).

The observation function is also depicted in Figure 7. Each state deterministically produces the
observation in the lower-right corner of its box in the ﬁgure. We also write it in another format in
Table 9.

Table 9: The observation function O for the example in Appendix C.1 and Appendix C.3.

s
S
I
W
WH
L
LH
T
O(s)
o∅
oI
oW
oW
oL
o∅
o∅

We make the additional assumption that the human belief B(⃗s | ⃗o) only supports state sequences ⃗s
which actually produce ⃗o under the sequence observation function ⃗O: B(⃗s | ⃗o) > 0 =⇒⃗O(⃗s) = ⃗o.
In particular, this means that for any ⃗o which is only produced by one ⃗s, B(⃗o | ⃗s) = 1.

There are three pairs of state sequences which produce identical observation sequences. For each,
we introduce a parameter representing the probability the human infers the ﬁrst of the pair of state
sequences upon seeing their shared observation sequence.

1. SILHT and SITT both produce o∅oIo∅o∅, a log containing only a success conﬁrmation
for installing drivers, again because O(LH) = O(T) = o∅. Let pH = B(⃗s = SILHT |
⃗o = o∅oIo∅o∅).

2. STTT and SLHTT both produce o∅o∅o∅o∅, an empty log, since O(LH) = O(T) = o∅.
Let p′
H = B(⃗s = SLHTT | ⃗o = o∅o∅o∅o∅).

3. SIWT and SIWHT both produce o∅oIoW o∅, a log containing success conﬁrmations for
installing both drivers and CUDA. Let pW = B(⃗s = SIWT | ⃗o = o∅oIoW o∅).

We assume for simplicity that p′
H = pH, i.e. that the human is just as likely to think an empty log
following a successful driver installation contains an error that was hidden with 2> /dev/null (pH),
as they are to think that an entirely empty log contains a hidden error (p′
H). In principle, this need
not be true: the human may have differing priors about the agent’s behavior in the two different
circumstances. However, the algebra to reason about such a case is signiﬁcantly more cumbersome,
and this case reveals no fundamentally different agent behavior under our framework that isn’t present
in some simpler case.

We can thus write the full B as a matrix as in Table 10.

21


---Page Break---
Table 10: The parameterized human belief function B for the example in Appendix C.1 and Ap-
pendix C.3, expressed as a matrix (rendered as a table). Any empty cell is equal to 0.

STTT
SLHTT
SLTT
SITT
SILHT
SILT
SIWT
SIWHT
o∅o∅o∅o∅
1 −pH
pH
o∅oLo∅o∅
1
o∅oIo∅o∅
1 −pH
pH
o∅oIoLo∅
1
o∅oIoW o∅
pW
1 −pW

We have laid the groundwork sufﬁciently to begin reasoning about the observation return, overesti-
mation and underestimation error, policies which are optimal under the reward function learned by
naive RLHF, and the resulting deceptive inﬂationand overjustiﬁcation failure modes. We begin by
computing the measures of interest for each state sequence, shown in Table 11.

Table 11: Measures of interest for each state sequence for the example in Appendix C.1 and
Appendix C.3. State sequences which produce the same observations have their Gobs columns
merged, since they necessarily have the same Gobs.

⃗s
G(⃗s)
Gobs(⃗s) := E⃗s ′∼B(·| ⃗O(⃗s))[G(⃗s′)]
E+(⃗s) := max(0,
E−(⃗s) := max(0,
Gobs(⃗s) −G(⃗s))
G(⃗s) −Gobs(⃗s))
STTT
0
pHG(SLHTT) + (1 −pH)G(STTT)
0
pH(5 + r)
SLHTT
−5 −r
= −pH(5 + r)
(1 −pH)(5 + r)
0
SLTT
−5
−5
0
0
SITT
1
pHG(SILHT) + (1 −pH)G(SITT)
0
pH(5 + r)
SILHT
−4 −r
= 1 −pH(5 + r)
(1 −pH)(5 + r)
0
SILT
−4
−4
0
0
SIWT
11
pW G(SIWT) + (1 −pW )G(SIWHT)
0
0
SIWHT
11
= 11
0
0

As an exercise, checking the computations in Table 11 is a quick way to gain some intuition for how
these quantities relate. It’s further useful to speak about these quantities using their names, and work
through the stories these expressions tell.

Consider the E+(SLHTT) cell. What does it mean that this is (1 −pH)(5 + r)? E+ is the
overestimation error; (1 −pH)(5 + r) is the expected amount by which the human observer, upon
seeing a completely empty log (o∅o∅o∅o∅), will overestimate the total reward the agent attained
when those empty logs were actually produced by SLHTT. This is a trajectory in which the
agent immediately uses the 2> /dev/null trick, fails to install CUDA without logging the error,
and terminates. Under what circumstances might the human overestimate the total reward when
⃗s = SLHTT? Upon seeing the empty log, the human thinks with probability 1 −pH that the agent
simply terminated immediately, which would be worth a total reward of 0. Since the actual total
reward is −5 −r, this is an overestimate by 5 + r. The human thinks with probability pH that
SLHTT occurred. This is correct, so there is no overestimation and this pH case does not contribute
to the overestimation error. So we have that with probability 1 −pH, the human overestimates the
total reward by 5 + r.

We can keep going! Why is the underestimation error of SIWT equal to 0? Because the only other
trajectory with which it can be confused attains the same total reward, so regardless of how the
probability mass of the human’s belief divides between them, there will be no underestimation. Can
all of the zeros in the overestimation and underestimation error columns be explained this way?

We now move on to consider policies rather than state sequences. Since a policy π imposes a
distribution P π over state sequences (the “on-policy distribution”), our policy measures are in fact
exactly parallel to our state sequence measures. Each one is an expectation over the on-policy
distribution of the columns of Table 11. We restrict our attention to deterministic policies which only
take actions depicted in Figure 7 (i.e. that never terminate via an action other than aT ), of which there
are only six in this MDP. They are enumerated, along with the policy-level measures, in Table 12.
Policies will be written as a sequence of actions enclosed in brackets, omitting trailing repeated aT
actions. This is nonstandard notation in an MDP with stochastic transitions, but is unambiguous in

22


---Page Break---
this example, because all decisions are made before any stochasticity occurs. The policies are [aT ],
[aHaT ], [aCaT ], [aIaT ], [aIaHaT ], and [aIaCaT ].

Table 12: Measures of interest for each policy for the example in Appendix C.1 and Appendix C.3.
Each of the columns here is the on-policy average of the corresponding column in Table 11. Policies
are written as sequences of actions, omitting trailing repeated aT actions. This is nonstandard notation
in an MDP with stochastic transitions, but is unambiguous in this example since all decisions are
made before any stochasticity occurs.

π
J(π)
Jobs(π)
E
+(π)
E
−(π)
[aT ]
0
−pH(5 + r)
0
pH(5 + r)
[aHaT ]
−5 −r
−pH(5 + r)
(1 −pH)(5 + r)
0
[aCaT ]
−5
−5
0
0
[aIaT ]
1
1 −pH(5 + r)
0
pH(5 + r)

[aIaHaT ]

pG(SIWHT)
pGobs(SIWHT)

(1 −p)(1 −pH)(5 + r)
0
+(1 −p)G(SILHT)
+(1 −p)Gobs(SILHT)
= 11 −(1 −p)(15 + r)
= 11 −(1 −p) [10 + pH(5 + r)]

[aIaCaT ]

pG(SIWT)
pGobs(SIWT)

0
0
+(1 −p)G(SILT)
+(1 −p)Gobs(SILT)
= 11 −(1 −p) · 15
= 11 −(1 −p) · 15

With this we have everything we need to characterize optimal policies under the reward function
learned by a naive application of RLHF (“policies selected by RLHF”). By Proposition 4.1, we know
that if PO is deterministic, as in this example, RLHF selects policies which maximize Jobs. In order
to understand the behavior of these policies, we’ll also need to determine the true optimal policies,
i.e. those which maximize J. We’ll proceed in cases, only considering boundary cases (speciﬁc
measure-zero parameter values for which the result is different) insofar as they are interesting.

Case 1: p > 1

3. If p > 1

3, the CUDA install (with default logging, aC) is likely enough to succeed
that it’s worth attempting it: p · R(W) + (1 −p) · R(L) > 0. It also immediately follows that

J([aIaCaT ]) = Jobs([aIaCaT ]) = 11 −(1 −p) · 15 > 1.

This allows us to eliminate policies [aT ], [aHaT ], [aCaT ], and [aIaT ], which all have J ≤1 and
Jobs ≤1. None of them can thus be J-optimal or Jobs-optimal. All that remains is to compare J and
Jobs for [aIaHaT ] and [aIaCaT ]. We can check the sign of the differences of these pairs of values,
starting with J.

J([aIaCaT ]) −J([aIaHaT ]) = (1 −p)r.

Since p is a probability and r is nonnegative, this value is positive (and thus [aIaCaT ] is preferred to
[aIaHaT ] by the human) if and only if p < 1 and r > 0.

Jobs([aIaHaT ]) −Jobs([aIaCaT ]) = (1 −p) [5 −pH(5 + r)] .

This value is positive (and thus [aIaHaT ] is the policy RLHF selects) if and only if p < 1 and
pH <
5
5+r.

If p = 1, then both differences are 0, and both J and Jobs are indifferent between the two policies.
This makes sense, as they differ only in the case where the CUDA installation fails; this happens
with probability 1 −p = 0 when p = 1. Now suppose p < 1. If r = 0, then the human is
indifferent between the two policies. This also makes sense, as r is meant to quantify the extent to
which the human dislikes suppressed failures; if it’s zero, then the human doesn’t care. However,
if pH <
5
5+r, then Jobs([aIaHaT ]) > Jobs([aIaHaT ]), and thus RLHF favors the 2> /dev/null
policy [aIaHaT ].

If p < 1, r > 0, and pH <
5
5+r, then we have that J([aIaCaT ]) > J([aIaHaT ]) but
Jobs([aIaCaT ]) > Jobs([aIaHaT ]). Thus RLHF will select the 2> /dev/null policy [aIaHaT ],
and by Theorem 4.5, since [aIaHaT ] is not J-optimal, then relative to [aIaCaT ], it must exhibit
deceptive inﬂation, overjustiﬁcation, or both. Intuitively, we should be suspicious that deceptive

23


---Page Break---
inﬂation is at play whenever the agent hides information from the human. Indeed, referencing
Table 12, we have E
+([aIaHaT ]) = (1 −p)(1 −pH)(5 + r) > 0 = E
+([aIaCaT ]). Together with
Jobs([aIaHaT ]) > Jobs([aIaCaT ]), this satisﬁes the conditions of Deﬁnition 4.3, and thus this is an
instance of deceptive inﬂation.

If p < 1, r > 0, and pH >
5
5+r, then [aIaCaT ] is optimal under both J and Jobs, and in this case,
RLHF selects the true optimal policy.

Case 2: p < 1

3. In this case, the CUDA install is not likely enough to succeed to be worth attempting
(under the true reward function). Mathematically, J([aIaHaT ]) ≤J([aIaCaT ]) < 1 = J([aIaT ]).
The other three policies are always worse under J than [aIaT ], so we have our optimal policy
π∗= [aIaT ]. However,

Jobs([aIaHaT ]) −Jobs([aIaT ]) = p(10 + pH(5 + r)),

which is always positive unless p = 0. As long as p > 0, i.e. installing CUDA is actually possible
at all, then regardless of everything else, Jobs([aIaHaT ]) > Jobs([aIaT ]). We can interpret this
reasonably: however good or bad it might look to terminate after installing drivers (SITT), using the
2> /dev/null trick will either produce the same observation as SITT (and thus the same feedback
distribution) or will actually visibly install CUDA successfully—the most lucrative possible source of
positive feedback. Even if there’s only a sliver of a chance of success and it’s a catastrophe to have a
silent failure (i.e. r ≫0), however likely you are to infer that the lack of a log message about CUDA
means the agent hid a failure, it will penalize [aIaT ] more than it will penalize [aIaHaT ]: where
[aIaHaT ] produces this ambiguous observation with probability 1 −p (when it fails), [aIaT ] will
always produce the ambiguous observation.

This means that when 0 < p < 1

3, it is impossible to recover the true optimal policy with naive RLHF.
Which policies can possibly be Jobs-optimal for some setting of the parameters? We can similarly
rule out [aT ] and [aHaT ] for 0 < p < 1

3:

Jobs([aIaHaT ]) −Jobs([aIaT ]) = p(10 + pH(5 + r)) > 0.

We can rule out [aCaT ] by comparison to [aIaCaT ]: Jobs([aIaCaT ]) −Jobs([aCaT ]) = 16 −(1 −
p)15 > 0. So we are left with only [aIaHaT ] and [aIaCaT ] as candidate Jobs-optimal policies.

As in Case 1, we ﬁnd that Jobs([aIaHaT ]) > Jobs([aIaT ]) if and only if p = 1 or pH <
5
5+r. In
case 2 we have assumed p < 1

3, leaving only the pH condition.

If pH <
5
5+r, then RLHF selects [aIaHaT ]. As in Case 1, this is deceptive inﬂationrelative to
π∗= [aIaT ], because

E
+([aIaHaT ]) = (1 −p)(1 −pH)(5 + r) > 0 = E
+(π∗).

If pH >
5
5+r, then RLHF selects [aIaCaT ]. Because this policy is not J-optimal, by Theorem 4.5,
we must have deceptive inﬂation, overjustiﬁcation, or both. Which is it? Here the optimal policy
is to terminate after installing drivers, [aIaT ]. However, pH >
5
5+r. This can be rewritten as
pH(5 + r) > 5. We have seen this expression pH(5 + r) before; it is the underestimation error
incurred on ⃗s = SITT and therefore also the average underestimation error of policy [aIaT ]. So
here the underestimation error on the optimal policy—that is, the risk that the human misunderstands
optimal behavior (terminating after installing driver) as undesired behavior (attempting a CUDA
install that was unlikely to work and hiding the mistake)—is severe enough that the agent opts instead
for [aIaCaT ], a worse policy that attempts the ill-fated CUDA installation only to prove that it wasn’t
doing so secretly. In qualitative terms, this is quintessential overjustiﬁcation behavior. Indeed, relative
to reference policy π∗= [aIaT ], we have

E
−([aIaCaT ]) = 0 < pH(5 + r) = E
−(π∗)
J([aIaCaT ]) = 11 −(1 −p) · 15 < 1 = J(π∗),

and thus by Deﬁnition 4.4, this is overjustiﬁcation.

C.4
Ambiguity in Section 4.4 examples when modeling partial observability

Consider the example in Fig. 4A when modeling partial observability as in Section 5. By Theorem 5.2,
the ambiguity in the return function leaving the choice probabilities invariant is given by ker B ∩im Γ.

24


---Page Break---
Table 13: Experiments showing improved performance of po-aware RLHF

Ex.
p
phide
pdefault
model
action
E
+
dec. inﬂ.
E
−
overj.
optimal

A
0.5
0.5
N/A
naive
aH
1.5
✓
0
×
×
A
0.5
0.5
N/A
po-aware
aH
1.5
✓
0
×
×

A
0.1
0.9
N/A
naive
aC
0
×
0
✓
×
A
0.1
0.9
N/A
po-aware
aT
0
×
5.4
×
✓

B
0.5
N/A
0.9
naive
aT
4.5
✓
0
✓
×
B
0.5
N/A
0.9
po-aware
aD
0
×
0.25
×
✓

B
0.5
N/A
0.1
naive
aV
0
×
0
✓
×
B
0.5
N/A
0.1
po-aware
aD
0
×
2.25
×
✓

Let R′ = (0, 0, R′(W), 0, R′(WH), 0, 0) ∈R{S,I,W,L,WH,LH,T } be a reward function that we want
to parameterize such that G′ := Γ ·R′ ends up in the ambiguity; here, R′ is interpreted as a column
vector.

We want B ·G′ = 0. Since the observation sequences ⃗o = o∅o∅o∅o∅, ⃗o = o∅oLo∅o∅, ⃗o = o∅oIo∅o∅,
or ⃗o = o∅oIoLo∅all cannot involve the states W or WH, it is clear that they have zero expected return
(B ·G′)(⃗o). Set p′
H := B
 
SIWHT | o∅oIoW o∅

. Then the condition that B ·G′ = 0 is equivalent
to:

0 =
 
B ·G′
(o∅oIoW o∅) =
E
⃗s∼B(⃗s|o∅oIoW o∅)


G′(⃗s)


= p′
H · G′(SIWHT) + (1 −p′
H) · G′(SIWT) = p′
H · R′(WH) + (1 −p′
H) · R′(W).

Thus, if R′(W) =
p′
H
p′
H−1R′(WH), then G′ ∈ker B ∩im Γ, meaning that R+R′ has the same choice
probabilities as R and is thus fully feedback-compatible. In particular, if R′(WH) ≫0 is sufﬁciently
large, then in subsequent policy optimization, there is an incentive to hide the mistakes and πH will
be selected, which is suboptimal with respect to the true reward function R.

Thus Fig. 4A still retains dangerous ambiguity when modeling partial observability.

However, the example in Fig. 4B leads to no ambiguity when partial observability is correctly
modeled.

To show this in detail, let G′ = Γ(R′) ∈ker B ∩im Γ. We need to show G′ = 0. Since the human
is only uncertain about the state sequences corresponding to the observation sequence o∅oIo∅o∅, the
condition B ·G′ = 0 already implies G′(⃗s) = 0 for all state sequences except SIWT and SITT.
From (B ·G′)(o∅oIo∅o∅) = 0, one then obtains the equation

(1 −pD) ·
 
R′(S) + R′(I) + 2R′(T)

+ pD ·
 
R′(S) + R′(I) + R′(W) + R′(T)

= 0.
(5)

Thus, if one of the two state sequences involved has zero return, then the other has as well, assuming
that 0 ̸= pD ̸= 1, and we are done.

To show this, we use that all other state sequences have zero return: R′(S) + 3R′(T) = 0 =
R′(S) + R′(L) + 2R′(T), from which R′(L) = R′(T) follows. Then, from R′(S) + R′(I) +
R′(L) + R′(T) = 0, substituting the previous result gives R′(S) + R′(I) + 2R′(T) = 0, and so
Equation (5) results in R′(S) + R′(I) + R′(W) + R′(T) = 0. Overall, this shows G′ = Γ(R′) = 0,
and so ker B ∩im Γ = {0}.

C.5
Experimental details

Here, we explain more experimental details for the results in Table 1, reproduced here as Table 13.

The leftmost column (“Ex.” for “example”) corresponds to Examples A and B in Figure 4. p is
the success probability upon attempting to install Cuda or NumPy in state I, see Figure 6. phide
in Example A is the human’s belief probability that the agent hid the error message if there is
no output after nvidia-driver installation. Similarly, pdefault in Example B is the human’s belief
probability that installation was done with default settings if there is no further output after Python

25


---Page Break---
installation. Note that lines one and two in the table also correspond to Example C.1, lines three and
four to Example C.2, and lines ﬁve and six to the ﬁrst half and seven and eight to the second half
of Example C.3, respectively. In all our experiments, we set the penalty to r = 1.

The “model” column has value “naive” if the reward learning algorithm is classical RLHF (erroneously
assuming full observability) as in Christiano et al. [2017], and “po-aware” if the human’s partial
observability is correctly modeled as in Appendix D.3. We initialize the reward function as a list of
rewards of states and train it by logistic regression using a dataset that consists of all pairs of state
sequences together with the human’s choice probabilities under partial observations. This leads to 28
pairs of distinct trajectories together with choice probabilities. We train the reward model for 300
epochs over a shufﬂed dataset of 13.5 copies of the 28 pairs with the Adam optimizer, for a total of
113400 training updates.

Once we have the resulting reward model, we use value iteration to ﬁnd its deterministic optimal
policy. All policies choose to install the nvidia-driver (in Example A) and Python (in Example B), and
differ in their action in state I, which is given in the column “action”. We compute the overestimation
error and underestimation error of the resulting policies analytically using the hardcoded environment
dynamics, true reward function, observation function, and human belief matrix B. This is given in
columns E
+ and E
−. Note that these are averages over 10 entire training runs, though since they
always result in the same learned policy, there is no variation and we do not state any uncertainty.

The columns “dec. inﬂ.”, “overj.”, and “optimal” state whether deceptive inﬂation or overjustiﬁcation
occurs with the learned policy, and whether it is optimal according to the true human’s reward
function.

D
Modeling the Human in Partially Observable RLHF

In this appendix, we develop the theory of RLHF with appropriately modeled partial observability,
including full proofs of all theorems.

In Section D.1, we explain how the human can arrive at the belief B(⃗s | ⃗o) via Bayesian updates. The
main theory and the main paper in general do not depend on this speciﬁc form of the human’s belief,
but some examples in the appendix do.

In Section D.2 we then explain our main result: the ambiguity and identiﬁability of both reward and
return functions under observed sequence comparisons. In Section D.3, we then explain that this
theorem means that one could in principle design a practical reward learning algorithm that converges
on the correct reward function up to the ambiguity characterized in the section before, if the human’s
belief kernel B(⃗s | ⃗o) is fully known.

In Section D.4, we generalize the theory to the case that the human’s observations are not necessarily
known to the learning system and again characterize precisely when the return function is identiﬁable
from sequence comparisons. We then consider special cases in Section D.5, where we show that the
fully observable case is covered by our theory, that a deterministic observation kernel P ⃗O usually
leads to non-injective belief matrix B, and that “noise” in the observation kernel P ⃗O leads, under
appropriate assumptions, to the identiﬁability of the return function.

Our identiﬁability results require that the learning system knows the human’s belief kernel B(⃗s | ⃗o).
In Section D.6, we then show that these results are robust to slight misspeciﬁcations: a bound in
the error in the speciﬁed belief leads to a corresponding bound in the error of the policy evaluation
function used for subsequent reinforcement learning.

In Section D.7, we then provide a very preliminary characterization of the ambiguity in the return
function under special cases.

Finally, in Section D.8, we study examples of identiﬁability and non-identiﬁability of the return
function for the case that we do model the human’s partial observability correctly. This reveals
qualitatively interesting cases of identiﬁability, even when B is not injective, and catastrophic cases
of non-identiﬁability.

26


---Page Break---
D.1
The Belief over the State Sequence for Rational Humans

Before we dive into the main theory, we want to explain how the human can iteratively compute the
posterior of the state sequence given an observation sequence with successively new observations.
This is done by deﬁning a Bayesian network for the joint probability of policy, states, actions, and
observations, and doing Bayesian inference over this Bayesian network.

The details of this subsection are only relevant for a few sections in the appendix since it is usually
enough to assume that the posterior belief exists. Additionally, in the core theory, we do not even
assume that B(⃗s | ⃗o) is a posterior: it is simply any probability distribution. The reason why it can
still be interesting to analyze the case when the human is a rational Bayesian reasoner is that one can
then analyze RLHF under generous assumptions to the human.

We model the human to have a joint distribution B(π,⃗s,⃗a,⃗o) over the policy π, state sequence
⃗s = s0, . . . , sT , action sequence ⃗a = a0, . . . , aT −1, and observation sequence ⃗o = o0, . . . , oT . This
is given by a Bayesian network with the following components:

• a policy prior B(π′);
• the probability of the initial state B(s0) := P0(s0);
• action probabilities B(a | s, π) := π(a | s);
• transition probabilities B(st+1 | st, at) := T (st+1 | st, at);
• and observation probabilities B(ot | st) := PO(ot | st).

Together, this deﬁnes the joint distribution B(π,⃗s,⃗a,⃗o) over the policy, states, actions, and observa-
tions that factorizes according to the following directed acyclic graph:

π′

s0
a0
s1
a1
s2
a2
s3
. . .

o0
o1
o2
o3

(6)

The following proposition clariﬁes the iterative Bayesian update of the human’s posterior over state
sequences, given observation sequences:
Proposition D.1. Let t ≤T −1 and denote by ˆs = s0, . . . , st a state sequence segment of length
t ≥0. Similarly, ˆo = o0, . . . , ot denotes an observation sequence segment. We have

B(ˆs, st+1, π | ˆo, ot+1) ∝PO(ot+1 | st+1) ·

" X

at∈A
T (st+1 | ˆst, at) · π(at | st)

#

· B(ˆs, π | ˆo).

Thus, the human can iteratively compute B(ˆs, π | ˆo) from the prior B(s0, π) = P0(s0) · B(π′) using
the above Bayesian update.

The posterior over the state sequence can subsequently be computed by

B(ˆs | ˆo) =
Z

π
B(ˆs, π | ˆo).

Proof. The proof is essentially just Bayes rule applied to the Bayesian network in Equation (6). We
repeatedly make use of conditional independences that follow from d-separations in the graph [Geiger
et al., 1990]. More concretely, we have
B
 
ˆs, st+1, π | ˆo, ot+1

∝B
 
ot+1 | ˆs, st+1, π, ˆo

· B
 
ˆs, st+1, π | ˆo


27


---Page Break---
Figure 8: The linear geometry of ambiguity for a hypothetical example with three state sequences
and two observation sequences. G∗is the true return function, and “G” is used in labeling the
axes to refer to some arbitrary return function. This is a more accurate geometric depiction of the
middle and right spaces in Figure 5. The subspace im Γ ∩ker B (purple) is the ambiguity in return
functions, meaning that adding an element would not change the human’s expected return function on
observations. Thus the set of return functions that the reward learning system can infer is the afﬁne
set G + (im Γ ∩ker B) (yellow). Note that the planes on the left are drawn to be axis-aligned for
ease of visualization; this will not be the case for real MDPs.

= PO
 
ot+1 | st+1

· B
 
st+1 | ˆs, π, ˆo) · B(ˆs, π | ˆo


= PO
 
ot+1 | st+1

·

" X

at∈A
B
 
st+1 | at, ˆs, π, ˆo

· B
 
at | ˆs, π, ˆo

#

· B
 
ˆs, π | ˆo


= PO
 
ot+1 | st+1

·

" X

at∈A
T
 
st+1 | st, at

· π
 
at | st

#

· B
 
ˆs, π | ˆo

.

In step 1, we used Bayes rule. In step 2, we made use of the independence ot+1⊥⊥(ˆs, π, ˆo) | st+1,
plugged in the observation kernel, and used the chain rule of probability to compose the second term
into a product. In step 3, we marginalized and used, once again, the chain rule of probability. In step 4,
we used the independences st+1 ⊥⊥(s0, . . . , st−1, π, ˆo) | (st, a) and at ⊥⊥(s0, . . . , st−1, ˆo) | (π, st)
and plugged in the transition kernel and the policy.

The last formula is just a marginalization over the policy.

D.2
Ambiguity and Identiﬁability of Reward and Return Functions under Observation
Sequence Comparisons

In this section, we prove the main theorem of this paper: a characterization of the ambiguity that
is left in the reward and return function once the human’s Boltzmann-rational choice probabilities
are known. We change the formulation slightly by formulating the linear operators “intrinsically” in
the spaces they are deﬁned in, instead of using matrix versions. This does not change the general
picture, but is a more natural setting when thinking, e.g., about generalizing the results to inﬁnite
state sequences. Thus, we deﬁne B : R ⃗S →R⃗Ωas the linear operator given by

B(G)

(⃗o) :=
E
⃗s∼B(⃗s|⃗o)


G(⃗s)

.

Here, B is the human’s belief, which can either be computed as in the previous subsection or simply
be any conditional probability distribution. Similarly, we deﬁne Γ : RS →R ⃗S as the linear operator
given by

Γ(R)

(⃗s) :=

T
X

t=0
γtR(st).

28


---Page Break---
The matrix product B · Γ then becomes the composition B ◦Γ : RS →R⃗Ω. Finally, recall that the
kernel ker A of a linear operator A is deﬁned as its nullspace, and the image im A as the set of
elements hit by A. We obtain the following theorem:

Theorem D.2. Let R be the true reward function and ˜R another reward function. Let ˜G = Γ( ˜R)
and G = Γ(R) be the corresponding return functions. The following three statements are equivalent:

(i) The reward function ˜R gives rise to the same vector of choice probabilities as R, i.e

P
˜
R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ω=

P R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ω.

(ii) There is a reward function R′ ∈ker(B ◦Γ) and a constant c ∈R such that

˜R = R + R′ + c.

(iii) There is a return function G′ ∈ker B ∩im Γ and a constant c′ ∈R such that

˜G = G + G′ + c′.

In other words, the ambiguity that is left in the reward function when its observation-based choice
probabilities are known is, up to an additive constant, given by ker(B ◦Γ); the ambiguity left in the
return function is given by ker B ∩im Γ.

Proof. Assume (i). To prove (ii), let σ by the sigmoid function given by σ(x) =
1
1+exp(−x). Then by

Equation (2), the equality of choice probabilities means the following for all ⃗o,⃗o′ ∈⃗Ω:

σ

β ·
 
B( ˜G)

(⃗o) −

B( ˜G)

(⃗o′)

= σ

β ·
 
B(G)

(⃗o) −

B(G)

(⃗o′)

.

Since the sigmoid function is injective, this implies

B( ˜G)

(⃗o) −

B( ˜G)

(⃗o′) =

B(G)

(⃗o) −

B(G)

(⃗o′).

Fixing an arbitrary ⃗o′, this implies that there exists a constant c′ such that for all ⃗o ∈⃗Ω, the following
holds:

B( ˜G)

(⃗o) −

B(G)

(⃗o′) −c′ = 0.

Noting that B(c′) = c′, this implies ˜G −G −c′ ∈ker(B). Now, deﬁne the constant reward function

c := c′ ·
1 −γ
1 −γT +1 .

We obtain


Γ(c)

(⃗s) =

T
X

t=0
γt · c

= c′ ·
1 −γ
1 −γT +1 ·

T
X

t=0
γt

= c′.

Thus, we have
Γ( ˜R −R −c) = ˜G −G −c′ ∈ker(B),

implying R′ := ˜R −R −c ∈ker(B ◦Γ). This shows (ii).

That (ii) implies (iii) follows by applying Γ to both sides of the equation.

Now assume (iii), i.e. ˜G = G+G′+c′ for a constant c′ ∈R and a return function G′ ∈ker(B)∩im Γ.
This implies B( ˜G) = B(G) + c′. Thus, for all ⃗o,⃗o′ ∈⃗Ω, we have

B( ˜G)

(⃗o) −

B( ˜G)

(⃗o′) =

B(G)

(⃗o) −

B(G)

(⃗o′),

which implies the equal choice probabilities after multiplying with β and applying the sigmoid
function σ on both sides. Thus, (iii) implies (i).

29


---Page Break---
Corollary D.3. The following two statements are equivalent:

(i) ker(B ◦Γ) = 0.

(ii) The data

P R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ωdetermine the reward function R up to an additive constant.

Proof. That (i) implies (ii) follows immediately from the implication from (i) to (ii) within the
preceding theorem.

Now assume (ii). Let R′ ∈ker(B ◦Γ). Deﬁne ˜R := R + R′. Then the implication from (ii) to
(i) within the preceding theorem implies that ˜R and R have the same choice probabilities. Thus,
the assumption (ii) in this corollary implies that R′ is a constant. Since Γ and B map nonzero
constants to nonzero constants, the fact that R′ ∈ker(B ◦Γ) implies that R′ = 0, showing that
ker(B ◦Γ) = {0}.

As mentioned in the main paper, the previous result already leads to the non-identiﬁability of R
whenever Γ is not injective, corresponding to the presence of zero-initial potential shaping (Skalse
et al. [2023], Lemma B.3). Thus, we now strengthen the previous result so that it deals with the
identiﬁability of the return function, which is sufﬁcient for the purpose of policy optimization:

Corollary D.4. Consider the following four statements (which can each be true or false):

(i) ker B = {0}.

(ii) ker
 
B ◦Γ) = {0}.

(iii) ker B ∩im Γ = {0}.

(iv) The data

P R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ωdetermine the return function G = Γ(R) on sequences ⃗s ∈⃗S

up to a constant independent of ⃗s.

Then the following implications, and no other implications, are true:

(i)

(iii)
(iv)

(ii)

In particular, all of (i), (ii), and (iii) are sufﬁcient conditions for identifying the return function from
the choice probabilities.

Proof. That (i) implies (iii) is trivial. That (ii) implies (iii) is a simple linear algebra fact: Assume (ii)
and that G′ ∈ker B ∩im Γ. Then G′ = Γ(R′) for some R′ ∈RS and

0 = B(G′) = B
 
Γ(R′)

= (B ◦Γ)(R′).

By (ii), this implies R′ = 0 and therefore G′ = Γ(R′) = 0, showing (iii).

That (iii) implies (iv) immediately follows from the implication from (i) to (iii) in Theorem D.2.

Now, assume (iv). To prove (iii), assume G′ ∈ker B ∩im Γ. Then the implication from (iii) to (i)
in Theorem D.2 implies that G + G′ induces the same observation-based choice probabilities as G.
Thus, (iv) implies G + G′ = G + c′ for some constant c′, which implies G′ = c′. Since G′ ∈ker B,
this implies 0 = B(G′) = B(c′) = c′ and thus G′ = 0. Thus, we showed ker B ∩im Γ = {0}.

30


---Page Break---
We now show that no other implication holds in general. Example D.32 will show that (ii) does not
imply (i). We now show that (i) does also not imply (ii), from which it will logically follow that (iii)
does neither imply (i) nor (ii). Namely, consider the following simple MDP with time horizon T = 1:

a
b
(7)

In this MDP, every state sequence starts in a, deterministically transitions to b, and then ends. This
means that ⃗s = ab is the only sequence. Now, let R′ ∈R{a,b} be the reward function given by

R′(a) = 1,
R′(b) = −1

γ .

We obtain

Γ(R′)

(⃗s) = R′(a) + γR′(b) = 1 + γ · −1

γ
= 0.

Thus, Γ(R′) = 0, (B ◦Γ)(R′) = 0, and, therefore, ker
 
B ◦Γ

̸= {0}. Thus, (ii) does not hold.
However, it is possible to choose B(⃗s | ⃗o) such that (i) holds: e.g., if Ω= S and B(⃗s | ⃗o) := δ⃗o(⃗s),
then ker B = {0} since this operator is the identity.

D.3
The Ambiguity in Reward Learning in Practice

In this section, we point out that Theorem D.2 is not just a theoretical discussion: When B and the
inverse temperature parameter β are known, then it is possible to design a reward learning algorithm
that learns the true reward function up to the ambiguity ker(B ◦Γ) in the inﬁnite data limit. In doing
so, we essentially use the loss function proposed in Christiano et al. [2017].

Namely, assume D is a data distribution of observation sequences ⃗o ∈⃗Ωsuch that all sequences in ⃗Ω
have a strictly positive probability of being sampled; for example, D could use an exploration policy
and the observation sequence kernel P ⃗O. For each pair of observation sequences (⃗o,⃗o′), we then
get a conditional distribution P(µ | ⃗o,⃗o′) over a one-hot encoded human choice µ ∈{(1, 0), (0, 1)},
with probability
P
 
µ = (1, 0) | ⃗o,⃗o′
= P R 
⃗o ≻⃗o′
.

Together, this gives rise to a dataset (⃗o1,⃗o′
1, µ1), . . . , (⃗oN,⃗o′
N, µN) of observation sequences plus a
human choice.

Now assume we learn a reward function Rθ : S →R that is differentiable in the parameter θ and that
can represent all possible reward functions R ∈RS. Let Gθ := Γ(Rθ) be the corresponding return
function. Write µk = (µ(1)
k , µ(2)
k ). As in Christiano et al. [2017], we deﬁne its loss over the dataset
above by

eL(θ) = −1

N

N
X

k=1
µ(1)
k
· log P Rθ 
⃗ok ≻⃗o′
k

+ µ(2)
k
· log P Rθ 
⃗o′
k ≻⃗ok

.

Note that by Equation (2), this loss function essentially uses B and also the inverse temperature
parameter β in its deﬁnition. This means that these need to be explicitly represented to be able to use
the loss function in practice.

Proposition D.5. The loss function eL is differentiable. Furthermore, in the inﬁnite datalimit its
minima are precisely given by parameters θ such that Rθ = R + R′ + c for R′ ∈ker
 
B ◦Γ

and
c ∈R, or equivalently Gθ = G + G′ + c′ for G′ ∈ker B ∩im Γ and c′ ∈R.

Proof. The differentiability of the loss function follows from the differentiability of multiplication
with the matrix B, see Equation (2), and of the reward function Rθ in its parameter θ that we assumed.

For the second statement, let N(⃗o,⃗o′) be the number of times that the pair (⃗o,⃗o′) appears in the
dataset, and let N(⃗o,⃗o′, 1) be the number of times that the human choice is µ = (1, 0) and the
sampled pair is (⃗o,⃗o′), and similar for 2 instead of 1. We obtain

eL(θ) = −
X

⃗o,⃗o ′∈⃗Ω

N(⃗o,⃗o′)

N
·

"
N(⃗o,⃗o′, 1)

N(⃗o,⃗o′) log P Rθ 
⃗o ≻⃗o′

31


---Page Break---
+ N(⃗o,⃗o′, 2)

N(⃗o,⃗o′) log P Rθ 
⃗o′ ≻⃗o

#

≈
E
⃗o,⃗o ′∼D


CE

P R 
⃗o ≻≺⃗o′  P Rθ 
⃗o ≻≺⃗o′

=:L(θ).

Here, CE is the crossentropy between the two binary distributions. Since we assumed that D gives
a positive probability to all observation sequences in ⃗Ω, and since the cross entropy is generally
minimized exactly when the second distribution equals the ﬁrst, the loss function L(θ) is minimized
if and only if Rθ gives rise to the same choice probabilities as R for all pairs of observation sequences.
Theorem D.2 then gives the result.

D.4
Identiﬁability of Return Functions When Human Observations Are Not Known

Corollary D.4 assumes that the choice probabilities of each observation sequence pair are known
to the reward learning algorithm. However, this requires the algorithm to know what the human
observed. In some applications, this is a reasonable assumption, e.g. if the human’s observations
are themselves produced by an algorithm that can feed the observations also back to the learning
algorithm. In general, however, the observations happen in the physical world, and are only known
probabilistically via the observation kernel PO. The learning system does however have access to the
full state sequences that generate the observation sequences. This leads to knowledge of the following
choice probabilities for ⃗s,⃗s′ ∈⃗S:

P R 
⃗s ≻⃗s′ :=
E
⃗o,⃗o′∼P ⃗
O(·|⃗s,⃗s ′)

h
P R 
⃗o ≻⃗o′i
, 2
(8)

where the observation-based choice probabilities are given as in Equation (2). In other words, the
learning algorithm can only infer an aggregate of the observation-based choice probabilities. Again,
we can ask a question similar to the ones before, extending the investigations in the previous section:

Question D.6. Assume the vector of choice probabilities

P R(⃗s ≻⃗s′)


⃗s,⃗s ′∈⃗S is known. Additionally,

assume that it is known that the human’s observations are governed by PO, and that the human is
Boltzmann rational with inverse temperature parameter β and beliefs B(⃗s | ⃗o), see Equation (8).
Does this data identify the return function G : ⃗S →R?

If the observation-based choice probabilities from Equation (2) would be known, then Corollary D.4
would provide the answer to this question. Thus, similar to how we previously inverted the belief
operator B, we are now simply tasked with inverting the expectation over observation sequences.
This leads us to the following deﬁnition:

Deﬁnition D.7 (Ungrounding Operator). The ungrounding operators O : R⃗Ω→R ⃗S and O ⊗O :
R⃗Ω×⃗Ω→R ⃗S× ⃗S are deﬁned by

O(v)

(⃗s) :=
E
⃗o∼P ⃗
O(⃗o|⃗s)


v(⃗o)

,

(O ⊗O)(C)

(⃗s,⃗s′) :=
E
⃗o,⃗o ′∼P ⃗
O(·|⃗s,⃗s ′)


C(⃗o,⃗o′)

.

Here, v ∈R⃗Ωis an arbitrary vector, and C ∈R⃗Ω×⃗Ωis also an arbitrary vector, where the notation
can remind of “Choice” since the inputs to O ⊗O are, in practice, vectors of observation-based
Boltzmann-rational choice probabilities.

Formally, O ⊗O is the Kronecker product of O with itself, but it is not necessary to understand
this fact to follow the discussion. Ultimately, to be able to recover the observation-based choice
probabilities, what matters is that O ⊗O is injective on whole vectors of these choice probabilities.
The injectivity of O is a sufﬁcient condition for this, which explains its usefulness. We show this in
the following lemma:

Lemma D.8. O : R⃗Ω→R ⃗S is injective if and only if O ⊗O : R⃗Ω×⃗Ω→R ⃗S× ⃗S is injective.

2We excuse the following abuse of notation: these choice probabilities run through the observations of the
human and are not the same as the choice probabilities from Equation (1).

32


---Page Break---
Proof. This is a general property of the Kronecker product of a linear operator with itself. For
completeness, we demonstrate the calculation in our special case. First, assume that O is injective.
Assume that (O ⊗O)(C) = 0 for some C ∈R⃗Ω×⃗Ω. We need to show C = 0.

For all pairs of state sequences (⃗s,⃗s′), we have

0 =

(O ⊗O)(C)

(⃗s,⃗s′) =
E
⃗o,⃗o ′∼P ⃗
O(·|⃗s,⃗s ′)


C(⃗o,⃗o′)


=
E
⃗o∼P ⃗
O(⃗o|⃗s)


E
⃗o ′∼P ⃗
O(⃗o ′|⃗s ′)


C(⃗o,⃗o′)


=
E
⃗o∼P ⃗
O(⃗o|⃗s)

h
C′
⃗s ′(⃗o)
i

=
h
O
 
C′
⃗s ′
i
(⃗s),

where C′
⃗s ′(⃗o) := E⃗o ′∼P ⃗
O(⃗o ′|⃗s ′)

C(⃗o,⃗o′)

. By the injectivity of O, we obtain C′
⃗s ′ = 0 for all ⃗s′.
This means that for all ⃗s′ and ⃗o, we have

0 = C′
⃗s ′(⃗o) =
E
⃗o ′∼P ⃗
O(⃗o ′|⃗s ′)


C(⃗o,⃗o′)

=
h
O
 
C′′
⃗o
i
(⃗s′),

where C′′
⃗o (⃗o′) := C(⃗o,⃗o′). Again, by the injectivity of O, we obtain C′′
⃗o = 0 for all ⃗o, leading to
C = 0. That proves the direction from left to right.

To prove the other direction, assume that O is not injective. This means there exists 0 ̸= C ∈R⃗Ω

such that O(C) = 0. Deﬁne C ⊗C ∈R⃗Ω×⃗Ωby

(C ⊗C)(⃗o,⃗o′) := C(⃗o)C(⃗o′).

Then clearly, C ⊗C ̸= 0. We are done if we can show that (O ⊗O)(C ⊗C) = 0 since that
establishes that O ⊗O is also not injective. For any ⃗s,⃗s′ ∈⃗S, we have
h
(O ⊗O)(C ⊗C)
i
(⃗s,⃗s′) =
E
⃗o,⃗o ′∼P ⃗
O(·|⃗s,⃗s ′)

h
(C ⊗C)(⃗o,⃗o′)
i

=
E
⃗o,⃗o ′∼P ⃗
O(·|⃗s,⃗s ′)

h
C(⃗o) · C(⃗o′)
i

=
E
⃗o∼P ⃗
O(⃗o|⃗s)


C(⃗o)

·
E
⃗o ′∼P ⃗
O(⃗o ′|⃗s ′)


C(⃗o′)


=

O(C)

(⃗s) ·

O(C)

(⃗s′)

= 0 · 0
= 0.

This ﬁnishes the proof.

We now state and prove the following extension of Corollary D.4:
Theorem D.9. Consider the following statements (which can each be true or false):

1. O : R⃗Ω→R ⃗S is an injective linear operator: ker O = {0}.

2. O ⊗O : R⃗Ω×⃗Ω→R ⃗S× ⃗S is an injective linear operator: ker O ⊗O = {0}.

3. O ⊗O is injective on vectors of observation-based choice probabilities

P R 
⃗o ≻⃗o′

⃗o,⃗o ′
over the set of return functions G ∈R ⃗S.

4. The data of state-based choice probabilities

P R 
⃗s ≻⃗s′

⃗s,⃗s ′∈⃗S from Equation (8)

determine the data of observation-based choice probabilities

P R 
⃗o ≻⃗o′

⃗o,⃗o ′∈⃗Ωfrom

Equation (2).

33


---Page Break---
Then the following implications hold and 3 does not imply 2:

1
2
3
4.

Consequently, if any of the conditions 1, 2, or 3 hold, and additionally any of the conditions (i), (ii)
or (iii) from Corollary D.4, then the data

P R 
⃗s ≻⃗s′

⃗s,⃗s ′∈⃗Ωdetermine the return function G on

sequences ⃗s ∈⃗S up to a constant independent of ⃗s.

Proof. That 1 and 2 are equivalent was shown in Lemma D.8. That 2 implies 3 is clear. To prove
that 3 implies 4, simply put both sets of choice probabilities into a vector. Then Equation (8) and
Deﬁnition D.7 show the following equality of vectors in R ⃗S× ⃗S:

P R 
⃗s ≻⃗s′

⃗s,⃗s ′ =
 
O ⊗O

P R 
⃗o ≻⃗o′

⃗o,⃗o ′


.

The injectivity of O ⊗O on such inputs ensures that the observation-based choice probabilities can
be recovered using this equation.

We now show that (3) does not imply (2). Again, we use the simple MDP from Equation (7), but this
time with a different observation kernel. Namely, we choose

PO(o(a) | a) = PO(o(a)′ | a) = 1

2,
PO(o(b) | b) = 1,

where o(a)′ ̸= o(a) and o(a) ̸= o(b) ̸= o(a)′. This results in two possible observation sequences:
o(a)o(b) and o(a)′o(b). Thus, R⃗Ωis two-dimensional, whereas R ⃗S is only one-dimensional. Con-
sequently, O : R⃗Ω→R ⃗S cannot be injective, so ker O ̸= {0}, so (2) does not hold since (1) and
(2) are equivalent. However, (3) still holds: Since there is only one state sequence, Equation (2)
shows that the only vector of choice probabilities has 1/2 in all its entries, irrespective of the return
function G. Thus, O ⊗O has only one input of observation-based choice probabilities, and is thus
automatically injective on its inputs.

The ﬁnal result of identiﬁability of the return function G follows using Corollary D.4.

D.5
Simple Special Cases: Full Observability, Deterministic P ⃗O, and Noisy P ⃗O

In this section, we analyze three simple special cases of the general theory.

Theorem 3.9 (together with Lemma B.3) from Skalse et al. [2023], reproduced as a corollary below,
is a special case of our theorem:
Corollary D.10 (Skalse et al. [2023]). Assume the human directly observes the true sequences, and
the choice probabilities are given by

P R 
⃗s ≻⃗s′
= σ

β
 
G(⃗s) −G(⃗s′)

.

This data determines the return function G = Γ(R) on state sequences ⃗s ∈⃗S up to a constant
independent on ⃗s.

Proof. We can embed this case into the one of Theorem D.9 by deﬁning the observation kernel
as P ⃗O(⃗s′ | ⃗s) = δ⃗s(⃗s′) (i.e., the correct sequence is deterministically observed) and deﬁning the
human’s belief as B(⃗s′ | ⃗s) = δ⃗s(⃗s′) (i.e., the human knows that the observation reﬂects the true
sequence). This shows that P(⃗s ≻⃗s′) is of the form of Equation (8). The result follows from
Theorem D.9: the operators O and B are the identity in this case, due to the deﬁning property of the
Kronecker delta, and so they are injective.

The following proposition shows that Corollary D.10 is essentially the only example of deterministic
observation kernel P ⃗O for which B is injective. Note, however, that in some situations, we can have
im Γ ∩ker B = {0} even if B is not injective, see Example D.32.
Proposition D.11. Assume P ⃗O, the observation kernel on the level of sequences, is deterministic and
not injective. Then O is automatically injective. However, B is not injective.

34


---Page Break---
Proof. To show that O is injective, assume v ∈R⃗Ωis such that O(v) = 0. Then for all ⃗s ∈⃗S, we
get
0 =

O(v)

(⃗s) =
E
⃗o∼P ⃗
O(⃗o|⃗s)


v(⃗o)

= v
  ⃗O(⃗s)

.

Since ⃗O : ⃗S →⃗Ωis by deﬁnition surjective, we obtain v = 0.

⃗O : ⃗S →⃗Ωis by deﬁnition surjective, and here assumed to be non-injective, which implies that ⃗S
has a higher cardinality than ⃗Ω. Thus, B : R ⃗S →R⃗Ωcannot be injective.

In the following, we analyze a simple case that guarantees identiﬁability. It requires that the
observation kernel is “well-behaved” of a form where the observations are simply “noisy states”, and
that the human is a Bayesian reasoner with any prior B(⃗s) that supports every state sequence ⃗s ∈⃗S.

Deﬁnition D.12 (Noise in the Observation Kernel). Then we say that there is noise in the observation
kernel PO : ⃗S →∆(⃗Ω) if ⃗S = ⃗Ωand if O is an injective linear operator.

Proposition D.13. Assume that ⃗S = ⃗Ω. Furthermore, assume that B(⃗s | ⃗o) is given by the posterior
with likelihood P ⃗O(⃗o | ⃗s) and any prior B(⃗s) with B(⃗s) > 0 for all ⃗s ∈⃗S. Then there is noise in the
observation kernel if and only if B is injective.

Proof. Assume O is injective. To show that B is injective, assume there is G′ ∈R ⃗S with B(G′) = 0.
Then for all ⃗o ∈⃗Ω, we have

0 =

B(G′)

(⃗o) =
E
⃗s∼B(⃗s|⃗o)


G′(⃗s)

=
X

⃗s
B(⃗s | ⃗o)G′(⃗s) ∝
X

⃗s
P ⃗O(⃗o | ⃗s) ·
 
B(⃗s) · G′(⃗s)


=

OT (B ⊙G′)

(⃗o).

Here, OT is the transpose of O and B ⊙G′ is the componentwise product of the prior B with the
return function G′. Since O is injective and thus invertible, OT is as well. Thus, B ⊙G′ = 0, which
implies G′ = 0 since the prior gives positive probability to all state sequences. Thus, B is injective.

For the other direction, assume B is injective. To show that O is injective, let v ∈R⃗Ωbe any vector
with O(v) = 0. We do a similar computation as above: for all ⃗s ∈R ⃗S, we have

0 =

O(v)

(⃗s) =
E
⃗o∼P ⃗
O(⃗o|⃗s)


v(⃗o)

=
X

⃗o
P ⃗O(⃗o | ⃗s)v(⃗o) ∝
X

⃗o
B(⃗s | ⃗o) ·
 
P ⃗O(⃗o) · v(⃗o)


=
h
BT  
P ⃗O ⊙v
i
(⃗s).

Here, BT is the transpose of B, P ⃗O(⃗o) is the denominator in Bayes rule, and P ⃗O ⊙v is the vector
with components P ⃗O(⃗o) · v(⃗o). From the injectivity and thus invertibility of B, it follows that BT is
invertible as well, and so P ⃗O ⊙v = 0, which implies v = 0. Thus, O is injective.

Corollary D.14. When there is noise in the observation kernel and the human is a Bayesian reasoner
with some prior B such that B(⃗s) > 0 for all ⃗s ∈⃗S, then the return function is identiﬁable from
choice probabilities of state sequences even if the learning system does not know the human’s
observations.

Proof. This follows from the injectivity of O, the injectivity of B that we proved in Proposition D.13,
and Theorem D.9.

Remark D.15. We mention the following caveat: intuitively, one could think that O (and thus B, by
Proposition D.13) will be injective if every ⃗s is identiﬁable from inﬁnitely many i.i.d. samples from
P ⃗O(⃗o | ⃗s). A counterexample is the following:

O =

 1/2
1/4
1/4
1/4
1/2
1/4
3/8
3/8
1/4

!

.

35


---Page Break---
In this case, the rows are linearly dependent with coefﬁcients 1/2, 1/2 and −1. Consequently, O and
B are not injective, and so if this observation kernel comes from a multi-armed bandit with three
states, then Corollary D.4 shows that the return function is not identiﬁable.

Nevertheless, the distributions P ⃗O(· | ⃗s) (given by the rows) all differ from each other, and so inﬁnitely
many i.i.d. samples identify the state sequence ⃗s.

D.6
Robustness of Return Function Identiﬁability under Belief Misspeciﬁcation

We now again look at the case where the observations that the human observes are known to the
reward learning system, as in Section D.2. Furthermore, we assume that B : R ⃗S →R⃗Ωis such that
ker B ∩im Γ = {0}. In this case, we can apply Corollary D.4 and identify the true return function G
from B(G), which, in turn, can be identiﬁed up to an additive constant from the observation-based
choice probabilities with the argument as for Proposition 3.1.

In this section, we investigate what happens when the human belief model is slightly misspeciﬁed. In
other words: the learning system uses a perturbed matrix B∆:= B + ∆with some small perturbation
∆. How much will the inferred return function deviate from the truth? To answer this, we ﬁrst need
to outline some norm theory of linear operators.

D.6.1
Some Norm Theory for Linear Operators

In this section, let V, W be two ﬁnite-dimensional inner product-spaces. In other words, V and W
each have inner products ⟨·, ·⟩and there are linear isomorphisms V ∼= Rk, W ∼= Rm such that the
inner products in V and W correspond to the standard scalar products in Rk and Rm. The reason
that we don’t directly work with Rk and Rm itself is that we will later apply the analysis to the case
that V = im Γ ⊆R ⃗S. Let in this whole section A : V →W be a linear operator and ∆: V →W
be a perturbance, so that A∆:= A + ∆is a perturbed version of A.

The inner products give rise to a norm on V and W deﬁned by

∥v∥=
p

⟨v, v⟩,
∥w∥=
p

⟨w, w⟩.

As is well known, for each linear operator A : V →W there exists a unique, basis-independent
adjoint (generalizing the notion of a transpose) AT : W →V such that for all v ∈V and w ∈W,
we have
⟨A v, w⟩=
D
v, AT w
E
.

Let us recall the following fact that is often used in linear regression:

Lemma D.16. Assume A : V →W is injective. Then AT A : V →V is invertible and
(AT A)−1 AT is a left inverse of A.

Proof. To show that AT A is invertible, we only need to show that it is injective. Thus, let 0 ̸= x ∈V .
Then
D
x, AT A x
E
= ⟨A x, A x⟩= ∥A x∥2 > 0,

where the last step followed from the injectivity of A. Thus, AT A x ̸= 0, and so AT A is injective,
and thus invertible. Consequently, (AT A)−1 AT is a well-deﬁned operator. That it is the left inverse
of A is clear.

Deﬁnition D.17 (Operator Norm). The norm of an operator A : V →W is given by

∥A ∥:=
max
x, ∥x∥=1 ∥A x∥.

It has the following well-known properties, where A, B and C are matrices of compatible sizes:

∥A + B ∥≤∥A ∥+ ∥B ∥,
∥C A ∥≤∥C ∥· ∥A ∥,
∥AT ∥= ∥A ∥.

To study how a perturbance in A (and thus AT A) transfers into a perturbance of
 
AT A
−1, we
will use the following theorem:

36


---Page Break---
Theorem D.18 (El Ghaoui [2002]). Let B : V →V be an invertible operator. Let ρ < ∥B−1 ∥−1.
Let ∆: V →V be any operator with ∥∆∥≤ρ. Then B + ∆is invertible and we have

(B + ∆)−1 −B−1  ≤
ρ · ∥B−1 ∥
∥B−1 ∥−1 −ρ.

Proof. See El Ghaoui [2002], Section 7 and in particular Equation 7.2. Note that the reference deﬁnes
∥A ∥to be the largest singular value of A; by the well-known min-max theorem, this is equivalent to
Deﬁnition D.17.

We will apply this theorem to AT A, which raises the question about the size of the perturbance in
AT A for a given perturbance in A. This is clariﬁed in the following lemma. Before stating it, for a
given perturbance ρ, deﬁne
eρ(A) := ρ ·
 
2 · ∥A ∥+ ρ

,
which depends on A and ρ. Also, recall that for a given perturbance ∆, we deﬁne A∆:= A + ∆.
We obtain:
Lemma D.19. Assume that ∥∆∥≤ρ. Then

∥AT
∆A∆−AT A ∥≤eρ(A).

Proof. We have
 AT
∆A∆−AT A
 =
(A + ∆)T (A + ∆) −AT A


=
 AT ∆+ ∆T A + ∆T ∆


≤∥A ∥· ∥∆∥+ ∥∆∥· ∥A ∥+ ∥∆∥2

≤ρ ·

2 · ∥A ∥+ ρ


= eρ(A).

To be able to apply Theorem D.18 to AT A, we need to make sure that eρ(A) is bounded above by
(AT A
−1∥−1. The next lemma clariﬁes what condition ρ needs to satisfy for eρ(A) to obey that
bound. For this, deﬁne

τ(A) := −∥A ∥+
q

∥A ∥2 +
(AT A)−1−1,
(9)

which only depends on A.
Lemma D.20. Assume ρ < τ(A). Then

eρ(A) <
(AT A)−1−1.

Proof. Note that ρ = τ(A) is the positive solution to the following quadratic equation in the
indeterminate ρ:

ρ2 + 2 · ∥A ∥· ρ −
(AT A)−1−1 = eρ(A) −
(AT A)−1−1 = 0.

Since this is a convex parabola, we get the inequality eρ(A) −
(AT A)−1−1 < 0 whenever we
have 0 ≤ρ < τ(A), which shows the result.

Finally, we put it all together to obtain a bound on the perturbance of
 
AT A
−1 AT . For this, set

C(A, ρ) :=
eρ(A) ·

 
AT A
−1

 
AT A
−1
−1
−eρ(A)
·
 A
 + ρ

+

 
AT A
−1 · ρ.
(10)

We obtain:

37


---Page Break---
Proposition D.21. Assume ∥∆∥≤ρ < τ(A). Then AT
∆A∆is invertible, and we have

 
AT
∆A∆
−1 AT
∆−
 
AT A
−1 AT  ≤C(A, ρ).

Proof. The invertibility of AT
∆A∆follows from Theorem D.18, Lemma D.19 and Lemma D.20.
We get

 
AT
∆A∆
−1 AT
∆−
 
AT A
−1 AT 

=

h 
AT
∆A∆
−1 −
 
AT A
−1i
· AT
∆+
 
AT A
−1 ·
 
AT
∆−AT 

≤

 
AT
∆A∆
−1 −
 
AT A
−1 ·
 A∆
 +

 
AT A
−1 · ∥∆∥

≤
eρ(A) ·

 
AT A
−1

 
AT A
−1
−1
−eρ(A)
·
 A
 + ρ

+

 
AT A
−1 · ρ

=C(A, ρ).

In the second-to-last step, we used Theorem D.18.

The constant C(A, ρ), deﬁned in Equation (10), has a fairly complicated form. In the following
proposition, we ﬁnd an easier-to-study upper bound in a special case:

Proposition D.22. Assume that ρ ≤∥A ∥and ρ ≤−∥A ∥+
q

∥A ∥2 + 1/2 ·
(AT A)−1−1.3

Then we have

C(A, ρ) ≤ρ ·
(AT A)−1 ·
h
12 · ∥A ∥2 ·
(AT A)−1 + 1
i
.

Proof. The second assumption gives, as in the proof of Lemma D.20, that eρ(A) ≤1/2 ·
(AT A)−1−1. Together with ρ ≤∥A ∥, the result follows.

D.6.2
Application to Bounds in the Error of the Return Function

We now apply the results from the preceding section to our case. Deﬁne r(B) : im Γ →R⃗Ωas the
restriction of the belief operator B to im Γ. Assume that ker B ∩im Γ = {0}, which is, according to
Corollary D.4, a sufﬁcient condition for identiﬁability. Note that this condition means that r(B) is
injective. Thus, Lemma D.16 ensures that r(B)T r(B) is invertible and that
 
r(B)T r(B)
−1r(B)T

is a left inverse of r(B).

Consequently, from the equation
r(B)(G) = B(G)

we obtain
G =
 
r(B)T r(B)
−1r(B)T (B(G)).

This is the concrete formula with which G can be identiﬁed from B(G). When perturbing B, this
leads to a corresponding perturbance in
 
r(B)T r(B)
−1r(B)T whose size inﬂuences the maximal
error in the inference of G. This, in turn, inﬂuences the size of the error in JG, the policy evaluation
function, where
JG(π) :=
E
⃗s∼P π(⃗s)


G(⃗s)

.

We obtain:

3Note the factor 1/2 compared to the deﬁnition of τ(A) in Equation (9).

38


---Page Break---
Theorem D.23. Let G be the true reward function, B the belief operator corresponding to the
human’s true belief model B(⃗s | ⃗o), and B(G) be the resulting observation-based return function.
Assume that ker B ∩im Γ = {0}, so that r(B)T r(B) is invertible. Let ∆: R ⃗S →R⃗Ωbe a
perturbation satisfying ∥∆∥≤ρ, where ρ satisﬁes the following two properties:

ρ ≤
r(B)
,
ρ ≤−
r(B)
 +
qr(B)
2 + 1/2 ·
 
r(B)T r(B)
−1−1.

Let B∆:= B + ∆be the misspeciﬁed belief operator. The ﬁrst claim is that r(B∆)T r(B∆) is
invertible under these conditions.

Now,
assume
that
the
learning
system
infers
the
return
function
˜G
:=
 
r(B∆)T r(B∆)
−1r(B∆)T (B(G)).4
Then there is a polynomial Q(X, Y ) of degree ﬁve
such that
∥˜G −G∥≤∥G∥· Q
(r(B)T r(B))−1, ∥r(B)∥

· ρ.

Thus, for all policies π, we obtain
J ˜
G(π) −JG(π)
 ≤∥G∥· Q
(r(B)T r(B))−1, ∥r(B)∥

· ρ.

In particular, for sufﬁciently small perturbances ρ, the error in the inferred policy evaluation function
J ˜
G becomes arbitrarily small.

Proof. That r(B∆)T r(B∆) is invertible follows immediately from Proposition D.21 by using that
∥r(∆)∥≤∥∆∥and that r(B∆) = r(B)r(∆), together with the second bound on ρ (which implies
the assumed bound in Proposition D.21).

We have
J ˜
G(π) −JG(π)
 =

E
⃗s∼P π(⃗s)


( ˜G −G)(⃗s)


≤
E
⃗s∼P π(⃗s)

h( ˜G −G)(⃗s)

i

≤max
⃗s∈⃗S

( ˜G −G)(⃗s)


≤∥˜G −G∥

=

h 
r(B∆)T r(B∆)
−1r(B∆)T −
 
r(B)T r(B)
−1r(B)T i
· B(G)


≤
 
r(B∆)T r(B∆)
−1r(B∆)T −
 
r(B)T r(B)
−1r(B)T  ·
 B(G)


≤C(r(B), ρ) · ∥r(B)(G)∥
≤C(r(B), ρ) · ∥r(B)∥· ∥G∥.

In the second to last step, we used Proposition D.21. By Proposition D.22, we can deﬁne the
polynomial Q(X, Y ) by

Q(X, Y ) = XY ·
h
12XY 2 + 1
i
,

which is of degree ﬁve.

The last claim follows from limρ→0 ρ = 0.

Remark D.24. In the case of a square matrix B that is injective, we can apply Theorem D.18 directly
to B−1 (which is now invertible) and obtain the following simpliﬁcation of Theorem D.23 for the
case that ∥∆∥≤ρ ≤1

2 · ∥B−1 ∥−1:
J ˜
G(π) −JG(π)
 ≤ρ · 2 · ∥B ∥· ∥G∥· ∥B−1 ∥2.

The polynomial is then only of degree 3.

4Note that there is not necessarily a ˜G with r(B∆)( ˜G) = B(G) since r(B∆) is not always surjective.
Nevertheless, ˜G :=
 
r(B∆)T r(B∆)
−1r(B∆)T (B(G)) is the best attempt at a solution in the sense that
r(B∆)( ˜G) then minimizes the Euclidean distance to B(G).

39


---Page Break---
D.7
Preliminary Characterizations of the Ambiguity

Recall the sequence of functions

RS
R ⃗S
R⃗Ω.
Γ
B

In this section, we clarify im Γ and ker B in special cases, as their intersection is the crucial ambiguity
in Theorem D.2.

The following proposition shows that for deterministic P ⃗O and a rational human, ker B decomposes
into hyperplanes deﬁned by normal vectors of probabilities of sequences mapping to the same
observation sequence:
Proposition D.25. Assume the human reasons as in Section D.1. Assume P ⃗O is deterministic.
Let B(⃗s) be the distribution of sequences under the human’s belief over the policy, given by
B(⃗s) =
R

π′ B(π′)P π′(⃗s) for some policy prior B(π′). For each ⃗o, let B⃗o := [B(⃗s)]⃗s: ⃗O(⃗s)=⃗o ∈

R{⃗s∈⃗S | ⃗O(⃗s)=⃗o} be the vector of probabilities of sequences that are observed as ⃗o.

Let G′ be a return function. For each ⃗o ∈⃗Ω, deﬁne the restriction G′
⃗o ∈R{⃗s∈⃗S| ⃗O(⃗s)=⃗o} by
G′
⃗o(⃗s) := G′(⃗s) for all ⃗s ∈{⃗s ∈⃗S | ⃗O(⃗s) = ⃗o}. Assume that B(⃗s | ⃗o) is the Bayesian posterior.
Then G′ ∈ker B if and only if the property

B⃗o · G′
⃗o = 0

holds for all ⃗o ∈⃗Ω.

Proof. For a deterministic observation kernel P ⃗O, by Bayes rule we have

B(⃗s | ⃗o) =
P ⃗O(⃗o | ⃗s) · B(⃗s)
P

⃗s ′ P ⃗O(⃗o | ⃗s′) · B(⃗s′)

=
δ⃗o
  ⃗O(⃗s)

· B(⃗s)
P

⃗s ′ δ⃗o
  ⃗O(⃗s′)

· B(⃗s′)

=

(
0, ⃗O(⃗s) ̸= ⃗o
B(⃗s)
P

⃗s ′: ⃗
O(⃗s ′)=⃗o B(⃗s ′), ⃗O(⃗s) = ⃗o.

Thus, for any return function G′ and any observation sequence ⃗o, we have

B(G′)

(⃗o) =
E
⃗s∼B(⃗s|⃗o)


G′(⃗s)


=
X

⃗s
B(⃗s | ⃗o)G′(⃗s)

=
X

⃗s: ⃗O(⃗s)=⃗o

B(⃗s)
P

⃗s ′: ⃗O(⃗s ′)=⃗o B(⃗s′)G′(⃗s)

=

 
X

⃗s ′: ⃗O(⃗s ′)=⃗o
B(⃗s′)

!−1

·
X

⃗s: ⃗O(⃗s)=⃗o
B(⃗s)G′(⃗s).

Thus, we have G′ ∈ker B if and only if

B⃗o · G′
⃗o =
X

⃗s: ⃗O(⃗s)=⃗o
B(⃗s)G′(⃗s) = 0

for all ⃗o. That was to show.

Remark D.26. One can interpret the previous proposition as follows:

As long as ⃗O is injective, we have
{⃗s ∈⃗S | ⃗O(⃗s) = o}
 = 1 for all ⃗o, meaning that B⃗o and G′
⃗o have
only one entry. Thus, B⃗o · G′
⃗o = 0 implies G′
⃗o = 0. If that holds for all ⃗o, then G′ ∈ker B implies
G′ = 0, meaning B is injective.

40


---Page Break---
However, as soon as there is an ⃗o with k⃗o :=
{⃗s ∈⃗S | ⃗O(⃗s) = o}
 > 1, the equation B⃗o · G′
⃗o = 0
leads to k⃗o −1 free parameters in G′
⃗o. G′
⃗o can then be chosen freely in the hyperplane of vectors
orthogonal to B⃗o without moving out of the kernel of B.

Another way of writing Proposition D.25 is to write ker B as a direct sum of these hyperplanes
perpendicular to B⃗o:
ker B =
M

⃗o: | ⃗O−1(⃗o)|≥2
B⊥
⃗o .

Recall that a return function G is called time-separable if there exists a reward function R such that
Γ(R) = G.

Before we discuss time-separability in more interesting examples, we want to talk about one simple
case where all return functions are time-separable. We leave a general characterization of im Γ to
future work.

Proposition D.27. Let there be an ordering ⃗s(1),⃗s(2), . . . of all sequences in ⃗S, and a function
φ : ⃗S →S from sequences to states such that φ(⃗s) ∈⃗s and φ(⃗s(k)) /∈⃗s(i) for all i < k. Then every
return function is time-separable.

Proof. Let G be a return function. Initialize R(s) = 0 for all s and inductively update it for all
i = 1, 2, . . . :

R
 
φ(⃗s(i))
 :=

 
X

t: s(i)
t
=φ(⃗s(i))
γt
!−1

·

 

G(⃗s(i)) −
X

t: s(i)
t
̸=φ(⃗s(i))
γt · R
 
s(i)
t

!

,

where the inductive deﬁnition always uses R as it is deﬁned by that point in time. Once R
 
φ(⃗s(i))


is deﬁned, but not yet any future values R
 
φ(⃗s(k))

, k > i, we have


Γ(R)

(⃗s(i)) =

T
X

t=0
γt · R
 
s(i)
t


=

 
X

t: s(i)
t
=φ(⃗s(i))
γt
!

· R
 
φ(⃗s(i))

+
X

t: s(i)
t
̸=φ(⃗s(i))
γt · R
 
s(i)
t


= G(⃗s(i)).

Furthermore, the property φ(⃗s(k)) /∈⃗s(i) for all i < k ensures that changes to the reward function
for k > i do not affect the value of

Γ(R)

(⃗s(i)). This shows Γ(R) = G, and thus G is time-
separable.

Corollary D.28. In a multi-armed bandit, every return function is time-separable.

Proof. In a multi-armed bandit, states and sequences are equivalent, and so we can choose φ(s) = s
for every state/sequence s. The result follows from Proposition D.27.

Alternatively, simply directly notice that in a multi-armed bandit, Γ is the identity mapping, and so
for every return/reward function R, we have Γ(R) = R.

D.8
Examples Supplementing Section 5

In this whole section, the inverse temperature parameter in the human choice probabilities is given by
β = 1. We now consider four more mathematical examples of Corollary D.4 and Theorem D.9. In
the ﬁrst example, the ambiguity is so bad that the reward inference can become worse than simply
maximizing Jobs as in naive RLHF. In Example D.30, there is simply “noise” in the observations and
the human’s belief, the matrices B and O are injective, and identiﬁability works, as in Corollary D.14.
In the third example, the matrix B is not injective and identiﬁability fails, which is a minimal example
showing the limits of our main theorems. In the fourth example, the matrix B is not injective, but
ker B ∩im Γ = {0}, and so identiﬁability works. This example is interesting in that the identiﬁability

41


---Page Break---
simply emerges through different distributions of delay that are caused by the different unobserved
events.

In this section, both the linear operators B : R ⃗S →R⃗Ωand O : R⃗Ω→R ⃗S are considered as
matrices
O =
 
P ⃗O(⃗o | ⃗s)


⃗s,⃗o ∈R
⃗S×⃗Ω,
B =
 
B(⃗s | ⃗o)


⃗o,⃗s ∈R
⃗Ω× ⃗S.

Notice that both have a swap in their indices.
Example D.29. Theorem 5.2 shows that the remaining ambiguity from the human’s choice probabil-
ities is given by ker B ∩im Γ, but it doesn’t explain how to proceed given this ambiguity. Without
further inductive biases, some reward functions within the ambiguity of the true reward function can
be even worse than simply maximizing Jobs.

E.g., consider a multi-armed bandit with three actions a, b, c, observation-kernel o = O(a) =
O(b) ̸= O(c) = c and reward function R(a) = R(b) < R(c). If the human belief is given by
B(a | o) = p = 1 −B(b | o), then R′ = α · (p −1, p, 0) ∈R{a,b,c} is in the ambiguity for all
α ∈R, and so ˜R := R + R′ is compatible with the choice probabilities. However, for α ≪0, we
have ˜R(a) > ˜R(b) and ˜R(a) > ˜R(c), and so optimizing against this reward function leads to a
suboptimal policy.

In contrast, maximizing Jobs leads to the correct policy since a, b, and c all obtain their ground truth
reward in this example. This generally raises the question of how to tie-break reward functions in the
ambiguity, or how to act conservatively given the uncertainty, in order to consistently improve upon
the setting in Section 4.1.
Example D.30. This example is a special case of Corollary D.14. Consider a multi-armed bandit
with two actions (which are automatically also states and sequences) a and b. In this case, the reward
function and return function is the same.

We assume there to be two possible observations o(a), o(b) and the observation kernel to be non-
deterministic, with probabilities

PO(o(j) | i) =
2/3, if i = j,
1/3, else.

If we assume the human forms Bayesian posterior beliefs as in Section D.1 and to have a policy prior
B(π′) such that B(a) =
R

π π(a)B(π′)dπ = 1/2 and B(b) = 1/2, then it is easy to show that the
human’s belief is the “reversed” observation kernel:

B(j | o(i)) = PO(o(i) | j).

We obtain

O = B =

2/3
1/3
1/3
2/3


= 1

3 ·

2
1
1
2



These matrices are injective since they are invertible:

O−1 = B−1 =

2
−1
−1
2


.

More generally, even if the human does not form fully rational posterior beliefs, it is easy to imagine
that the matrix B can end up being invertible. Thus, Corollary D.4 guarantees that the reward
function can be inferred up to an additive constant from the choice probabilities of observations, and
Theorem D.9 shows that this even works when the learning system does not know what the human
observed.

In the rest of this example, we explicitly walk the reader through the process of how the reward
function can be inferred, in the general case that the observations are not known. In the process,
we essentially recreate the proof of the theorems for this special case. For this aim, we ﬁrst want to
compute the choice probabilities P R 
i ≻j

that the learning system has access to in the limit of
inﬁnite data. We assume that the reward function is given by R(a) = −1 and R(b) = 2. We compute:

B(R) = 1

3 ·

2
1
1
2


·

−1
2


=

0
1


.

42


---Page Break---
In other words, we have Es∼B(s|o(a))[R(s)] = 0 and Es∼B(s|o(b))[R(s)] = 1. From this, we can
compute the observation-based choice probabilities ePo(i)o(j) = σ
 
B(R)(o(i)) −B(R)(o(j))

, see
Equation (2), and obtain:

ePo(a)o(a) = ePo(b)o(b) = 1

2,
ePo(a)o(b) =
1
1 + e,
ePo(b)o(a) =
e
1 + e.

We can now determine the ﬁnal choice probabilities Pij := P R 
i ≻j

again by a matrix-vector
product, with the indices ordered lexicographically, see Equation (8). Here, O ⊗O is the Kronecker
product of the matrix O with itself:

P = (O ⊗O) · eP = 1

9 ·






4
2
2
1
2
4
1
2
2
1
4
2
1
2
2
4




·






1/2
1/(1 + e)
e/(1 + e)
1/2




=






1/2
1/3 · (2 + e)/(1 + e)
1/3 · (1 + 2e)/(1 + e)
1/2




.

For example, the second entry in P is Pab = P R 
a ≻b

=
2+e
3·(1+e). This is the likelihood that,
for ground-truth actions a, b, the human will prefer a after only receiving observations o(a) or o(b)
according to O and following a Boltzman-rational policy based on the belief of the real action, see
Equation (8).

Over time, the learning system will be able to estimate these probabilities based on repeated human
choices, assuming all state-pairs are sampled inﬁnitely often. The question of identiﬁability is whether
the original reward function R can be inferred from that data, given that the learning system knows
O and B. We assume that the learning system doesn’t a priori know R or any of the intermediate
steps in the computation. First, eP can be inferred by inverting O ⊗O:

eP = (O ⊗O)−1 · P =






4
−2
−2
1
−2
4
1
−2
−2
1
4
−2
1
−2
−2
4




·






1/2
1/3 · (2 + e)/(1 + e)
1/3 · (1 + 2e)/(1 + e)
1/2




=






1/2
1/(1 + e)
e/(1 + e)
1/2




.

The learning system wants to use this to infer B( ˜R) (for the later-to-be inferred reward function ˜R
that may differ from the true reward function R) and uses the equation

ePo(a)o(b) =
exp
 
B( ˜R)(o(a))


exp
 
B( ˜R)(o(a))

+ exp
 
B( ˜R)(o(b))
,

which can be rearranged to

B( ˜R)(o(a)) = log
ePo(a)o(b)

1 −ePo(a)o(b)
+ B( ˜R)(o(b)) = log 1/(1 + e)

e/(1 + e) + B( ˜R)(o(b)) = B( ˜R)(o(b)) −1.

This relation is all which can be inferred about B( ˜R)(o(a)) and B( ˜R)(o(b)); the precise value cannot
be determined and B( ˜R)(o(b)) is a free parameter. One can check that for B( ˜R)(o(b)) = 1 this
coincides with the true value B(R). Finally, one can invert B to infer ˜R from this:

˜R = B−1 · B( ˜R)

=

2
−1
−1
2


·

B( ˜R)(o(b)) −1
B( ˜R)(o(b))



=

B( ˜R)(o(b)) −2
1 + B( ˜R)(o(b))



=

−1
2


+

B( ˜R)(o(b)) −1
B( ˜R)(o(b)) −1



= R +

B( ˜R)(o(b)) −1
B( ˜R)(o(b)) −1


.

Thus, the inferred and true reward functions differ maximally by a constant, as predicted in Theo-
rem D.9.

43


---Page Break---
In the following example, we work out a case where the reward function is so ambiguous that any
policy is optimal to some reward function consistent with the human feedback:
Example D.31. Consider a multi-armed bandit with exactly three actions/states a, b, c. We assume
a deterministic observation kernel with o := O(a) = O(c) ̸= O(b) = b. Assume the human has
some arbitrary beliefs B(a | o), B(c | o) = 1 −B(a | o), and can identify b: B(b | b) = 1. Then if
the human makes observation comparisons with a Boltzman-rational policy, as in Theorem D.2, the
resulting reward function is so ambiguous that some reward functions consistent with the feedback
place the highest value on action a, no matter the true reward function R. Thus, even if the true
reward function R regards a as the worst action, a can result from the reward learning and subsequent
policy optimization process.

Proof. The matrix B : R{a,b,c} →R{o,b} is given by

B =

B(a | o)
0
B(c | o)
0
1
0


.

Its kernel is given by reward functions R′ with R′(b) = 0 and R′(c) = −B(a|o)

B(c|o) R′(a), with R′(a) a
free parameter. Theorem D.2 shows that, up to an additive constant, the reward functions consistent
with the feedback of observation comparisons are given by ˜R = R + R′ for any R′ ∈ker B.
Thus, whenever the free parameter R′(a) satisﬁes R′(a) > R(b) −R(a) and R′(a) > B(c |
o) ·
 
R(c) −R(a)

, we obtain ˜R(a) > ˜R(b) and ˜R(a) > ˜R(c), showing the claim.

We now investigate another example where B is not injective, and yet, identiﬁability works because
B ◦Γ ̸= {0}. We saw such cases already in Example E.6, but include this additional example since
it shows a conceptually interesting case: two different states lead to the exact same observations,
but can be disambiguated since they lead to different amounts of delay until a more informative
observation is made again.
Example D.32. In this example, we assume that the human knows the policy π that generates the
state sequences (corresponding to a policy prior B(π′) = δπ(π′) concentrated on π), which together
with knowledge of the transition dynamics of the environment determines the true state transition
probabilities T π(s′ | s) = P

a∈A T (s′ | s, a) · π(a | s). We consider an environment with three
states s, s′, s′′ and the following transition dynamics T π, where p ̸= 1/2 is a probability:

s

s′
s′′

1/3
1/3

1/3

1−p
p

p

1−p

We assume that P0(s) = 1. Furthermore, we assume deterministic observations and s = O(s) ̸=
O(s′) = O(s′′) =: o.

Assume the time horizon T is 3, i.e., there are timesteps 0, 1, 2, 3. Assume that the human forms the
belief over the true state sequence by Bayesian posterior updates as in Section D.1. In this case,
ker B ̸= {0} by Proposition D.11. However, we will now show that ker(B ◦Γ) = {0}. If the human
makes Boltzmann-rational comparisons of observation sequences, then this implies the identiﬁability
of the return function up to an additive constant by Corollary D.4.5

Thus, let R′ ∈ker(B ◦Γ), i.e.,
h
B
 
Γ(R′)
i
(⃗o) = 0 for every observation sequence ⃗o. For

⃗o = ssss being the observation sequence that only consists of state s, this implies R′(s) = 0.
Consequently, for general observation sequences ⃗o, we have:

0 =
h
B
 
Γ(R′)
i
(⃗o) =
E
⃗s∼B(⃗s|⃗o)

"
3
X

t=0
δs′(st) · γt
#

· R′(s′) +
E
⃗s∼B(⃗s|⃗o)

"
3
X

t=0
δs′′(st) · γt
#

· R′(s′′).

5We assume that the learning system knows what the human observes, which is valid since PO is deterministic.
Alternatively, one can argue with Proposition D.11 that O is automatically injective, meaning one can apply
Theorem D.9.

44


---Page Break---
Now we specialize this equation to the two observation sequences ⃗o(1) = soss and ⃗o(2) = soos.
We start by considering ⃗o(1). This is consistent with the two state sequences ⃗s(1),(s′) = ss′ss and
⃗s(1),(s′′) = ss′′ss. We have posterior probabilities

B
 
⃗s(1),(s′) | ⃗o(1)
= 1 −p,
B
 
⃗s(1),(s′′) | ⃗o(1)
= p,

and therefore

0 =

B
 
Γ(R′)
i
(⃗o(1)) = (1 −p) · γ · R′(s′) + p · γ · R′(s′′),

and so
R′(s′) =
p
p −1 · R′(s′′).
(11)

Similarly, ⃗o(2) is consistent with the sequences ⃗s(2),(s′) = ss′s′s and ⃗s(2),(s′′) = ss′′s′′s. They have
posterior probabilities

B
 
⃗s(2),(s′) | ⃗o(2)
= 1

2,
B
 
⃗s(2),(s′′) | ⃗o(2)
= 1

2,

leading to

0 = 1

2 · (γ + γ2) · R′(s′) + 1

2 · (γ + γ2) · R′(s′′).

Together with Equation (11), we obtain

R′(s′′) = −R′(s′) =
p
1 −p · R′(s′′),

which implies R′(s′′) = 0 because p ̸= 1

2, and thus also R′(s′) = 0. Overall, we have showed
R′ = 0, and so B ◦Γ is injective. This means that reward functions are identiﬁable in this example
up to an additive constant, see Corollary D.4.

E
Issues of Naively Applying RLHF under Partial Observability

In this section, we study the naive application of RLHF under partial observability. Thus, most of it
takes a step back from the general theory of appropriately modeled partial observability in RLHF.
Later, we will analyze examples where we also apply the general theory, which is why this appendix
section comes second.

In Section E.1, we ﬁrst brieﬂy explain what happens when the learning system incorrectly assumes
that the human observes the full environment state. We show that as a consequence, the system
is incentivized to infer what we call the observation return function Gobs, which evaluates a state
sequence based on the human’s belief of the state sequence given the human’s observations. In the
policy optimization process, the policy is then selected to maximize Jobs, an expectation over Gobs.
In the interlude in Section E.2, we then brieﬂy analyze the unrealistic case that the human, when
evaluating a policy π, fully knows the complete speciﬁcation of that policy and all of the environment
and engages in rational Bayesian reasoning; in this case, Jobs = J is the true policy evaluation
function.

Realistically, however, maximizing Jobs can lead to failure modes. In Appendix E.3 we prove that a
suboptimal policy that is optimal according to Jobs causes deceptive inﬂation, overjustiﬁcation, or
both. In Appendix C.3, we expand on the analysis of the main examples in the main paper. Finally,
in Section E.4, we study further concrete examples where maximizing Jobs reveals deceptive and
overjustifying behavior by the resulting policy.

E.1
Optimal Policies under RLHF with Deterministic Partial Observations Maximize Jobs

Assume that P ⃗O is deterministic and that the human makes Boltzmann-rational sequence comparisons
between observation sequences. The true choice probabilities are then given by (See Equations (2)
and (8)):

P R 
⃗s ≻⃗s′
= σ

β ·
 
B ·G
  ⃗O(⃗s)

−
 
B ·G
  ⃗O(⃗s′)

(12)

Now, assume that the learning system does not model the situation correctly. In particular, we assume:

45


---Page Break---
• The system is not aware that the human only observes observation sequences ⃗O(⃗s) instead
of the full state sequences.
• The system does not model that the human’s return function is time-separable, i.e., comes
from a reward function R over environment states.

The learning system then thinks that there is a return function ˜G ∈R⃗S such that the choice
probabilities are given by the following faulty formula:

P R 
⃗s ≻⃗s′ := σ

β
 
G(⃗s) −G(⃗s′)


Now, assume that the learning system has access to the choice probabilities and wants to infer G.
Inverting the sigmoid function and then plugging in the true choice probabilities from Equation (12),
we obtain:

˜G(⃗s) = 1

β log P R(⃗s ≻⃗s′)

P R(⃗s′ ≻⃗s) + ˜G(⃗s′)

= 1

β

h
β ·
 
B ·G
  ⃗O(⃗s)

−
 
B ·G
  ⃗O(⃗s′)
i
+ ˜G(⃗s′)

=
 
B ·G
  ⃗O(⃗s)

+ C(⃗s′).6

Here, C(⃗s′) is some quantity that does not depend on ⃗s. Now, ﬁx ⃗s′ as a reference sequence.
Then for varying ⃗s, C(⃗s′) is simply an additive constant. Consequently, up to an additive constant,
this determines the return function that the learning system is incentivized to infer. We call it the
observation return function since it is the return function based on the human’s observations:

Gobs(⃗s) :=
 
B ·G
  ⃗O(⃗s)

.

This return function is not necessarily time-separable, but we assume that time-separability is not
modeled correctly by the learning system. Now, deﬁne the resulting policy evaluation function Jobs
by
Jobs(π) :=
E
⃗s∼P π(⃗s)


Gobs(⃗s)

.

This is the policy evaluation function that would be optimized if the learning system erroneously
inferred the return function Gobs.

E.2
Interlude: When the Human Knows the Policy and is a Bayesian Reasoner, then
Jobs = J

In this section, we brieﬂy consider what would happen if in Jobs, the human’s belief B would make
use of the true policy and be a rational Bayesian posterior as in Section D.1. We will show that
under these conditions, we have Jobs = J. Since these are unrealistic assumptions, no other section
depends on this result.

For the analysis, we drop the assumption that the observation sequence kernel P ⃗O is deterministic,
and assume that Jobs is given as follows:

Jobs(π) :=
E
⃗s∼P π(⃗s)

"

E
⃗o∼P ⃗
O(⃗o|⃗s)


E
⃗s ′∼Bπ(⃗s ′|⃗o)


G(⃗s′)
#

.
(13)

In this formula, Bπ(⃗s | ⃗o) := B(⃗s | ⃗o, π) with B being the joint distribution from Section D.1.
Formally, this is the posterior of the joint distribution B(⃗s,⃗o | π) that is given by the following hidden
Markov model:

s0
s1
s2
s3
. . .

o0
o1
o2
o3
. . .

PO

T π

PO

T π

PO

T π

PO

T π

(14)

6Note that in the case of non-deterministic observation kernels and choice probabilities given as in Equa-
tion (8), this argument does not work since the logarithm cannot be swapped with the outer expectation of the
choice probabilities.

46


---Page Break---
Here, T π(s′ | s) := P

a∈A T (s′ | s, a) · π(a | s). s0 is sampled according to the known initial
distribution P0(s0). The human’s posterior Bπ(⃗s′ | ⃗o) is then the true posterior in this HMM. We
obtain:
Proposition E.1. Let π be a policy that is known to the human. Then Jobs(π) = J(π).

Proof. By Equation (13), we have

Jobs(π) =
E
⃗s∼P π(⃗s)

"

E
⃗o∼P ⃗
O(⃗o|⃗s)


E
⃗s ′∼Bπ(⃗s ′|⃗o)


G(⃗s′)
#

(1)
=
X

⃗s
P π(⃗s)
X

⃗o
P ⃗O(⃗o | ⃗s)
X

⃗s ′
Bπ(⃗s′ | ⃗o)G(⃗s′)

(2)
=
X

⃗s ′

" X

⃗o
Bπ(⃗s′ | ⃗o)

" X

⃗s
P ⃗O(⃗o | ⃗s)P π(⃗s)

##

G(⃗s′)

(3)
=
X

⃗s ′

" X

⃗o
Bπ(⃗s′ | ⃗o)Bπ(⃗o)

#

G(⃗s′)

(4)
=
X

⃗s ′

" X

⃗o
P π(⃗s′)P ⃗O(⃗o | ⃗s′)

#

G(⃗s′)

(5)
=
X

⃗s ′
P π(⃗s′)G(⃗s′)

(6)
=
X

⃗s
P π(⃗s)G(⃗s)

(7)
= J(π).

In step (1), we wrote the expectations out in terms of sums. In step (2), we reordered them. In step (3),
we observed that the inner sum over ⃗s evaluates to the marginal distribution Bπ(⃗o) of the observation
sequence ⃗o in the HMM in Equation (13). In step (4), we used Bayes rule in the inner sum. This is
possible since Bπ(⃗s′ | ⃗o) is the true posterior when π is known. In step (5), we pull P π(⃗s′) out and
notice that the remaining inner sum evaluates to 1. Step (6) is a relabeling and step (7) the deﬁnition
of the true policy evaluation function J.

E.3
Proof of Theorem 4.5

We ﬁrst prove the following lemma.
Lemma E.2. Let π and πref be two policies. If J(π) < J(πref) and Jobs(π) > Jobs(πref), then
relative to πref, π must exhibit deceptive inﬂation, overjustiﬁcation, or both.

Proof. We start by establishing a quantitative relationship between the average overestimation and
underestimation errors E
+ and E
−as deﬁned in Deﬁnition 4.2, the true policy evaluation function
J, and the observation evaluation function Jobs deﬁned in Equation (4). Deﬁne ∆: ⃗S →R by
∆(⃗s) = Gobs(⃗s) −G(⃗s), where Gobs is as deﬁned in Equation (3). Consider the quantity

E+(⃗s) −E−(⃗s) = max
 
0, ∆(⃗s)

−max
 
0, −∆(⃗s)

.

If ∆(⃗s) > 0, then the ﬁrst term is ∆(⃗s) and the second one is 0. If ∆(⃗s) < 0, then the ﬁrst term is
zero and the second one is ∆(⃗s). If ∆(⃗s) = 0, then both terms are zero. In all cases the right-hand
side is equal to ∆(⃗s). Unpacking the deﬁnition of ∆again, we have that for all ⃗s,

E+(⃗s) −E−(⃗s) = Gobs(⃗s) −G(⃗s).
(15)

For any policy π, if we take the expectation of both sides of this equation over the on-policy
distribution admitted by π, P π, we get

E
+(π) −E
−(π) = Jobs(π) −J(π).
(16)

47


---Page Break---
We now prove the lemma. Let π and πref be two policies, and assume that J(π) < J(πref) and
Jobs(π) ≥Jobs(πref). Equivalently, we have Jobs(π) −Jobs(πref) ≥0 and J(πref) −J(π) > 0,
which we combine to state

Jobs(π) −Jobs(πref)

+

J(πref) −J(π)

> 0.
(17)

Rearranging terms yields

Jobs(π) −J(π)

−

Jobs(πref) −J(πref)

> 0.

These two differences inside parentheses are equal to the right-hand side of (16) for π and πref,
respectively. We substitute the left-hand side of (16) twice to obtain


E
+(π) −E
−(π)

−


E
+(πref) −E
−(πref)

> 0.

Rearranging terms again yields


E
+(π) −E
+(πref)

+


E
−(πref) −E
−(π)

> 0.
(18)

If E
+(π) −E
+(πref) > 0 then we have E
+(π) > E
+(πref) and, by assumption, Jobs(π) >
Jobs(πref). By Deﬁnition 4.3, this means π exhibits deceptive inﬂation relative to πref.

If E
−(πref) −E
−(π) > 0 then we have E
−(π) < E
−(πref) and, by assumption, J(π) < J(πref).
By Deﬁnition 4.4, this means π exhibits overjustiﬁcation relative to πref.

At least one of the two differences in parentheses in (18) must be positive, otherwise their sum would
not be positive. Thus π must exhibit deceptive inﬂation relative to πref, overjustiﬁcation relative to
πref, or both.

We can now combine earlier results to prove Theorem 4.5, repeated here for convenience:
Theorem E.3. Assume that PO is deterministic. Let π∗
obs be an optimal policy according to a naive
application of RLHF under partial observability, and let π∗be an optimal policy according to the
true objective J. If π∗
obs is not J-optimal, then relative to π∗, π∗
obs must exhibit deceptive inﬂation,
overjustiﬁcation, or both.

Proof. Because PO is deterministic, π∗
obs must be optimal with respect to Jobs by Proposition 4.1
(proved in Appendix E.1). Thus Jobs(π∗
obs) ≥Jobs(π∗). Since π∗is J-optimal and π∗
obs is not,
J(π∗) < J(π∗
obs). By Lemma E.2, relative to π∗, π∗
obs must exhibit deceptive inﬂation, overjustiﬁca-
tion, or both.

E.4
Further Examples Supplementing Section 4.4

In this section, we present further mathematical examples supplementing those in Section 4.4. We
found many of them before ﬁnding the examples we discuss in the main paper, and show the same and
additional conceptual features with somewhat less polish. We again assume that P ⃗O is deterministic.
Example E.4. In the main paper, we have assumed a model where the human obeys Eq. (2) and
showed that a naive application of RLHF can lead to suboptimal policies, and the speciﬁc failure
modes of deceptive inﬂation and overjustiﬁcation. What if the human makes the choices in a different
way? Speciﬁcally, assume that all we know is that P R(⃗o ≻⃗o′) + P R(⃗o′ ≻⃗o) = 1. Can the human
generally choose these choice probabilities in such a way that RLHF is incentivized to infer a reward
function whose optimal policies are also optimal for R? The answer is no.

Take the following example:

s

a
b
c

In this example, there is a ﬁxed start state s and three actions a, b, c that also serve as the ﬁnal
states. The time horizon is T = 1, so the only state sequences are sa, sb, sc. Assume T (a | s, a) = 1,

48


---Page Break---
T (b | s, b) = 1, T (c | s, c) = 1 −ϵ, T (a | s, c) = ϵ, i.e., selecting action c sometimes leads to state
a. Also, assume a = O(a) ̸= O(b) = O(c) =: o and R(a) = R(b) < R(c).

Since b and c have the same observation o, the human choice probabilities do not make a difference
between them, and so RLHF is incentivized to infer a reward function ˜R with ˜R(b) = ˜R(c) =: ˜R(o).
If ˜R(o) > ˜R(a), then the policy optimal under ˜R will produce action b since this deterministically
leads to observation o, whereas c does not. If ˜R(o) < ˜R(a), then the policy optimal under ˜R
will produce action a. In both cases, the resulting policy is suboptimal compared to π∗, which
deterministically chooses action c.

In the coming examples, it will also be useful to look at the misleadingness of state sequences:

Deﬁnition E.5 (Misleadingness). Let ⃗s ∈⃗S be a state sequence. Then its misleadingness is deﬁned
by
M(⃗s) := Gobs(⃗s) −G(⃗s) =
E
⃗s ′∼B(⃗s ′| ⃗O(⃗s))


G(⃗s′) −G(s)

.

We call a state sequence positively misleading if M(⃗s) > 0, which means the sequence appears better
than it is, and negatively misleading if M(⃗s) < 0. The misleadingness vector is given by M ∈R ⃗S.

Note that the misleadingness is related to E+ and E−, as deﬁned in Deﬁnition 4.2: If M(⃗s) > 0 then
M(⃗s) = E+(⃗s), and if M(⃗s) < 0 then M(⃗s) = −E−(⃗s).

Example E.6. In this example, we assume the human is a Bayesian reasoner as in Section D.1.
Consider the MDP that is suggestively depicted as follows:

a
b

c

The MDP has states S = {a, b, c} and actions A = {b, c}. The transition kernel is given by T (c |
a, c) = 1 and T (b | a, b) = 1, meaning that the action determines whether to transition from a to b
or c. All other transitions are deterministic and do not depend on the action, as depicted. We assume
an initial state distribution P0 over states with probabilities pa = P0(a), pb = P0(b), pc = P0(c).
The true reward function R ∈R{a,b,c} and discount factor γ ∈[0, 1) are, for now, kept arbitrary.
The time horizon is T = 2, meaning we have four possible state sequences acc, abc, bcc, ccc.

Furthermore, assume that o := O(a) = O(b) ̸= O(c) = c, i.e., c is observed and a and b are
ambiguous.

Finally, assume that the human has a policy prior B(λ), where λ = πλ(c | a) is the likelihood that
the policy chooses action c when in state a, which is a parameter that determines the entire policy.

We claim the following:

1. If pb ̸= γ · Eλ∼B(λ)[λ] · pa, then ker B ∩im Γ = {0}, so there is no return function
ambiguity under appropriately modeled partially observable RLHF, see Corollary D.4.

2. There are true reward functions R for which optimizing Jobs leads to a suboptimal policy
according to the true policy evaluation function J, a case of misalignment. Thus, a naive
application of RLHF under partial observability fails, see Section 4.1.

3. The failure modes are related to hiding negative information (deception) and purposefully
revealing information while incuring a loss (overjustifying behavior).

Proof. Write p := B(bcc | occ), the human’s posterior probability of state sequence bcc for observa-
tion sequence occ. We have 1 −p = B(acc | occ).

Consider the linear operators Γ : R{a,b,c} →R{abc,bcc,ccc,acc} and B : R{abc,bcc,ccc,acc} →
R{ooc,occ,ccc} deﬁned in the main paper. When ordering the states, state sequences, and observation

49


---Page Break---
sequences as we just wrote down, we obtain

Γ =






1
γ
γ2

0
1
γ + γ2

0
0
1 + γ + γ2

1
0
γ + γ2




,
B =

 1
0
0
0
0
p
0
1 −p
0
0
1
0

!

,
B ◦Γ =




1
γ
γ2

1 −p
p
γ + γ2

0
0
1 + γ + γ2



.

By Corollary D.4, if B ◦Γ is injective, then there is no reward function ambiguity. Clearly, this is the
case if and only if p ̸= γ · (1 −p). From Bayes rule, we have

p =
B(bcc)
B(acc) + B(bcc),
1 −p =
B(acc)
B(acc) + B(bcc).

So the condition for injectivity holds if and only if

B(bcc) ̸= γ · B(acc).

Now, notice

B(bcc) =
Z

λ
B(λ) · B(bcc | λ)dλ =
Z

λ
B(λ) · pbdλ = pb

and

B(acc) =
Z

λ
B(λ)B(acc | λ)dλ =
Z

λ
B(λ) · pa · λdλ = pa ·
E
λ∼B(λ)


λ

.

This shows the ﬁrst result.

For the second statement, we explicitly compute Jobs up to an afﬁne transformation, which does not
change the policy ordering. Let R be the true reward function, G = Γ(R) the corresponding return
function, and B(G) the resulting return function at the level of observations. For simplicity, assume
R(c) = 0, which can always be achieved by adding a constant. We have:

Jobs(λ) =
E
⃗s∼P λ(⃗s)

h
B(G)
  ⃗O(⃗s)
i

= P λ(abc) · B(G)(ooc) + P λ(bcc) · B(G)(occ) + P λ(ccc) · B(G)(ccc) + P λ(acc) · B(G)(occ)
= pa · (1 −λ) · G(abc) + pb · B(G)(occ) + pc · G(ccc) + pa · λ · B(G)(occ)

∝λ ·
h
B(G)(occ) −G(abc)
i
.

We have

G(abc) = R(a)+γR(b),
B(G)(occ) = (1−p)·G(acc)+p·G(bcc) = (1−p)·R(a)+p·R(b).

Thus, the condition B(G)(occ) > G(abc) is equivalent to

R(a) < p −γ

p
· R(b).

Thus, we have

arg max
λ∈[0,1]
Jobs(λ) =

(
1, if R(a) < p−γ

p
· R(b),
0, else.

Now consider the case R(b) > 0. In this case, λ = 0 gives rise to the optimal policy according
to G since going to b gives extra reward that one misses when going to c directly. However, when
R(a) ≪0, then Jobs selects for λ = 1. Intuitively, the policy tries to “hide that the episode started in
a” by going directly to c, which leads to ambiguity between acc and bcc. This is a case of deceptive
inﬂation as in Theorem 4.5.

Now, consider the case R(b) < 0. In this case, λ = 1 gives rise to the optimal policy according to G.
However, when R(a) ≫0, then Jobs selects for λ = 0. Intuitively, the policy tries to “reveal that the
episode started with a” by going to b, which is positive information to the human, but negative from
the perspective of optimizing G. As in Theorem 4.5, we see that this is a case of overjustiﬁcation.

50


---Page Break---
Example E.7. In this example, we consider an MDP that’s similar to a multi-armed bandit with
four states/actions a, b, c, d and observation kernel O(a) = O(b) ̸= O(c) = O(d). Formally, we can
imagine that it is given by the MDP

s

a
b
c
d

with R(s) = 0 and a time-horizon of T = 1. In this example, we reveal that misleadingness and
non-optimality (according to the true reward R, or J) are in principle orthogonal concepts. We
consider the following four example cases. In each one, we vary some environment parameters
and then determine a∗
obs, the action that results from optimizing Jobs (corresponding to a naive
application of RLHF under partial observability, see Section 4.1), its misleadingness M(a∗
obs) (see
Deﬁnition E.5), and the action a∗that would result from optimizing J. If a∗
obs = a∗, then Jobs selects
for the optimal action. For simplicity, we can imagine that the human has a uniform prior over what
action results eventually (out of the action taken and potentially a deviation deﬁned by ϵ, see below)
is taken before making an observation, i.e. B(a) = B(b) = B(c) = B(d) = 1

4.

(a) Assume R(a) > R(c) > R(d) ≫R(b). Also assume that action d leads with probability ϵ > 0
to state b, whereas all other actions lead deterministically to the speciﬁed state. Then a∗
obs = c,
M(c) < 0 and a∗= a.

(b) Assume R(d) > R(a) > R(c) ≫R(b). Again, assume there is a small probability ϵ > 0 that
action d leads to state b. Then a∗
obs = c, M(c) > 0, and a∗= d or a∗= a, depending on the
size of ϵ.

(c) Assume R(a) > R(b) > R(c) > R(d). Additionally, assume that there is a large probability
ϵ > 0 that action a leads to state d, whereas all other actions lead to what’s speciﬁed. If ϵ is
large enough, then a∗= b. Additionally, we have a∗
obs = b and M(b) > 0.

(d) Assume R(a) > R(b) > R(c) > R(d). Also, assume some probability ϵ > 0 that action b leads
to state d, whereas all other actions lead deterministically to what’s speciﬁed. Then a∗
obs = a,
M(a) < 0, and a∗= a.

Overall, we notice:

• Example (a) shows a high regret and negative misleadingness of a∗
obs = c. The action is
better then it seems, but action a would be better still but cannot be selected because it can
be confused with the very bad action b.

• Example (b) shows a high regret and high misleadingness of a∗
obs = c. The action is worse
than it seems and also not optimal.

• Example (c) shows zero regret and high misleadingness of a∗
obs = b. The action is worse
than it seems because it can be confused with a, but it is still the optimal action because a
can turn into d.

• Example (d) shows zero regret negative misleadingness of a∗
obs = a. The action is chosen
even though it seems worse than it is, and is also optimal.

Thus, we showed all combinations of regret and misleadingness of the action optimized for under
Jobs.

We can also notice the following: Examples (a) and (b) only differ in the placement of R(d). In
particular, the reason that a∗
obs = c is structurally the same in both, but the misleadingness changes.
This indicates that misleadingness is not on its own contributing to what Jobs optimizes for.

The following is the smallest example we found with the following properties:

• There is a unique start state and terminal state.
• A naive application of RLHF fails in a way that shows deception and overjustiﬁcation.
• Modeling partial observability resolves the problems.

51


---Page Break---
Example E.8. Consider the following graph:

A

S
C
T

B

This depicts an MDP with start state S, terminal state T and possible state sequences
STTT, SATT, SACT, SCTT, SBCT, SBTT and no discount, i.e. γ = 1. Assume that S, B, C
are observed, i.e. O(S) = S, O(B) = B, O(C) = C, and that A and T are ambiguous: O(A) =
O(T) = X. Then there are ﬁve observation sequences SXXX, SXCX, SCXX, SBCX, SBXX.
Assume that the human can identify all observation sequences except SXXX, with belief b =
B(STTT | SXXX) and 1 −b = B(SATT | SXXX).

Then the return function is identiﬁable under these conditions when the human’s belief is correctly
modeled. However, for some choices of the true reward function R and transition dynamics of this
MDP, we can obtain deceptive or overjustiﬁed behavior for a naive application of RLHF.

Proof. We apply Corollary D.4. We order states, state sequences, and observation sequences as
follows:

S = S, A, B, C, T,
⃗S = STTT, SATT, SACT, SCTT, SBCT, SBTT,
⃗Ω= SXXX, SXCX, SCXX, SBCX, SBXX.

As can easily be veriﬁed, with this ordering the matrices B ∈R⃗Ω× ⃗S and Γ ∈R ⃗S×S are given by:

B =








b
1 −b
0
0
0
0
0
0
1
0
0
0
0
0
0
1
0
0
0
0
0
0
1
0
0
0
0
0
0
1






,
Γ =










1
0
0
0
3
1
1
0
0
2
1
1
0
1
1
1
0
0
1
2
1
0
1
1
1
1
0
1
0
2









.

To show identiﬁability, we need to show that ker B ∩im Γ = {0}. Clearly, the kernel of B is given
by all return functions in R ⃗S that are multiples of G′ = (b −1, b, 0, 0, 0, 0). Assume G′ ∈im Γ,
meaning there is a reward function R′ ∈R ⃗S with Γ ·R′ = G′. We need to deduce from this a
contradiction. The assumption means we obtain the following equations:

(i) R′(S) + 3R′(T) = b −1,

(ii) R′(S) + R′(A) + 2R′(T) = b,

(iii) R′(S) + R′(A) + R′(C) + R′(T) = 0,

(iv) R′(S) + R′(C) + 2R′(T) = 0,

(v) R′(S) + R′(B) + R′(C) + R′(T) = 0

(vi) R′(S) + R′(B) + 2R′(T) = 0

(iii) and (v) together imply R′(A) = R′(B); (iv) and (vi) together imply R′(B) = R′(C); (v) and
(vi) together imply R′(C) = R′(T); so together, we have R′(A) = R′(T). Thus, replacing R′(A) in
(ii) by R′(T) and comparing (i) and (ii), we obtain b −1 = b, a contradiction. Overall, this shows
ker B ∩im Γ = {0}, and thus identiﬁability of the return function by Corollary D.4.

Now we investigate the case of unmodeled partial observability.

For demonstrating overjustiﬁcation, assume deterministic transition dynamics in which every arrow
in the diagram can be chosen by the policy. Also, assume R(A) ≪0, R(T) > 0, R(S) = 0,

52


---Page Break---
R(B) = 0, and R(C) = 0. Then the optimal policy chooses the state sequence STTT. However,
this trajectory has low observation value since Gobs(STTT) = (B ·G)(SXXX) = bG(STTT) +
(1 −b)G(SATT), which is low since R(A) ≪0. Jobs then selects for the suboptimal policies
choosing SBTT or SCTT, which is overjustiﬁed behavior that makes sure that the human does not
think state A was accessed.

For demonstrating deception, assume that R(A) ≫0, R(T) < 0, R(S) = R(B) = R(C) = 0 and
that the transition dynamics are such that when the policy attempts to transition from S to A, it will
sometimes transition to B, with all other transitions deterministic. In this case, the optimal behavior
attempts to enter state A since this has very high value. Jobs, however, will select for the policy that
chooses STTT. This is deceptive behavior.

53


---Page Break---
F
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: This can be veriﬁed by reading the paper.

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

Justiﬁcation: In Section 6 we have a paragraph on limitations.

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

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

54


---Page Break---
Justiﬁcation: All theorems come with a full set of assumptions, with full proofs in the
appendix linked. Sometimes, “background assumptions”, like the fact that we study an
underlying MDP with an additional observation kernel PO(⃗o | ⃗s), or that the human comes
with a belief kernel B(⃗s | ⃗o), are omitted in the theorem statements since they apply
throughout to the whole paper.

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

Answer: [NA]

Justiﬁcation: The paper does not include experiments.

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
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

55


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [NA]

Justiﬁcation: The paper does not include experiments requiring code.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/pu
blic/guides/CodeSubmissionPolicy) for more details.
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

Answer: [NA]

Justiﬁcation: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?

Answer: [NA]

Justiﬁcation: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, conﬁ-
dence intervals, or statistical signiﬁcance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

56


---Page Break---
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

Answer: [NA]

Justiﬁcation: The paper does not include experiments.

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

Justiﬁcation: This research does not involve human subjects, does not make use of data, and
does not propose a practical method that could be misused or have a negative impact. As
such, the paper does not give rise to any ethical concerns.

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

Justiﬁcation: The last paragraph of the main paper is an impact statement, listing the positive
impact we hope to see from our work. As our work is theoretical and does not provide a
method, no negative impact arises from it.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.

57


---Page Break---
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
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justiﬁcation: The paper poses no such risks.

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

Answer: [NA]

Justiﬁcation: The paper does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

58


---Page Break---
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has
curated licenses for some datasets. Their licensing guide can help determine the license
of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [NA]

Justiﬁcation: The paper does not release new assets.

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

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

59


---Page Break---
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

60


---Page Break---
