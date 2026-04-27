Randomized Exploration for Reinforcement Learning
with Multinomial Logistic Function Approximation

Wooseong Cho∗
Seoul National University
Seoul, South Korea
wooseong_cho@snu.ac.kr

Taehyun Hwang∗
Seoul National University
Seoul, South Korea
th.hwang@snu.ac.kr

Joongkyu Lee
Seoul National University
Seoul, South Korea
jklee0717@snu.ac.kr

Min-hwan Oh†
Seoul National University
Seoul, South Korea
minoh@snu.ac.kr

Abstract

We study reinforcement learning with multinomial logistic (MNL) function approx-
imation where the underlying transition probability kernel of the Markov decision
processes (MDPs) is parametrized by an unknown transition core with features of
state and action. For the finite horizon episodic setting with inhomogeneous state
transitions, we propose provably efficient algorithms with randomized exploration
having frequentist regret guarantees. For our first algorithm, RRL-MNL, we adapt
optimistic sampling to ensure the optimism of the estimated value function with
sufficient frequency. We establish that RRL-MNL achieves a e
O(κ−1d
3
2 H
3
2 √

T) fre-
quentist regret bound with constant-time computational cost per episode. Here, d is
the dimension of the transition core, H is the horizon length, T is the total number
of steps, and κ is a problem-dependent constant. Despite the simplicity and practi-
cality of RRL-MNL, its regret bound scales with κ−1, which is potentially large in
the worst case. To improve the dependence on κ−1, we propose ORRL-MNL, which
estimates the value function using the local gradient information of the MNL transi-
tion model. We show that its frequentist regret bound is e
O(d
3
2 H
3
2 √

T +κ−1d2H2).
To the best of our knowledge, these are the first randomized RL algorithms for
the MNL transition model that achieve statistical guarantees with constant-time
computational cost per episode. Numerical experiments demonstrate the superior
performance of the proposed algorithms.

1
Introduction

Reinforcement learning (RL) is a sequential decision-making problem in which an agent tries to
maximize its expected cumulative reward by interacting with an unknown environment over time.
Despite significant empirical progress in RL algorithms for various applications [47, 52, 65, 66, 25],
the theoretical understanding of RL algorithms had long been limited to tabular methods [40, 56,
10, 77, 79], which explicitly enumerate the entire state and action spaces and learn the value (or
the policy) for each state and action. Recently, there has been an increasing body of research in
RL with function approximation to extend beyond the tabular problem setting. In particular, linear
function approximation has served as a foundational model [43, 73, 22, 9, 37]. On the other hand,

∗Equal contribution
†Corresponding author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
the linear transition model assumption poses significant constraints: 1) the output of the function
must be within [0, 1], and 2) the sum of the probabilities for all possible next states must be exactly 1.
These constraints make it challenging to apply RL with linear function approximation to real-world
applications [35]. To overcome such challenges, there has been literature on RL with general function
approximation [21, 28, 37, 44, 4, 18]. Despite the guarantee of sample efficiency achieved by their
algorithms, this accomplishment might be impeded by computational intractability or the necessity to
rely on stronger assumptions. As a result, the resulting methods may not be as general or practical.

On the other hand, Hwang and Oh [35] introduce specific non-linear parametric MDPs called MNL-
MDPs (Assumption 1) where the transition probability of MDPs is given by an MNL model. They
consider an upper confidence bound (UCB) approach to balance exploration and exploitation. Since
it is costly or even intractable to compute UCB explicitly, randomized exploration methods such
as Thompson Sampling (TS) are widely studied in RL with linear function approximation as well
as tabular MDPs. This is because, in various decision-making problems ranging from multi-armed
bandits to RL, randomized exploration algorithms have been shown to perform better than UCB
methods in empirical evaluations [16, 57, 64, 49]. Furthermore, randomized exploration can be
easily integrated with linear function approximation. This is because the value function in linear
MDPs can be linearly parameterized, allowing perturbations of the estimator to directly control the
perturbations of the value function. However, although there has been some literature aiming to
propose randomized algorithms for general function classes [37, 4, 5, 75], these methods do not
discuss how to define the posterior distribution supported by the given function class and how to draw
the optimistic sample from the posterior [4, 5, 75], or they require stronger assumptions on stochastic
optimism [37], which is one of the most challenging elements in frequentist regret analysis. Thus, the
design of a tractable randomized exploration RL algorithm and the feasibility of frequentist regret
analysis for randomized exploration remain open challenges. Hence, the following question arises:

Can we design a provably efficient and tractable randomized algorithm for RL with MNL function
approximation?

We answer the above question by proposing the first randomized algorithm, RRL-MNL, achieving
e
O(κ−1d
3
2 H
3
2 √

T) frequentist regret with constant-time computational cost per episode. RRL-MNL is
not only the first algorithm with randomized exploration for MNL-MDPs, but also, to the best of our
knowledge, it provides the first frequentist regret analysis for a non-linear model-based algorithm
with randomized exploration without assuming stochastic optimism [37].

While RRL-MNL is statistically efficient, the current method used to analyze the regret of MNL
function approximation introduces a problem-dependent constant κ (Assumption 4), which reflects
the level of non-linearity of the MNL transition model. This constant κ originates from the use
of generalized linear models (GLMs) for contextual bandit settings [26, 51, 45] and MNL bandit
settings [54, 17, 55]. The magnitude of the constant κ can be exponentially small with respect to
the size of the decision set, hence the regret bound scaling with κ−1 could be prohibitively large in
the worst case [23]. Worse yet, the situation is even more challenging in RL, as in the worst case,
κ−1 can be much larger than in the case of bandits. To overcome the prohibitive dependence on
κ, algorithms based on new Bernstein-like inequalities and the self-concordant-like property of the
log-loss have been proposed for logistic bandits [23, 3, 24] and for MNL bandits [61, 6, 50]. As an
extension of these works, the following fundamental question remains open:

Is it possible for RL algorithms with MNL function approximation to have a sharper dependence on
the problem-dependent constant κ?

For the above question, we propose the second randomized algorithm referred to as ORRL-MNL, which
establishes a regret bound of e
O(d
3
2 H
3
2 √

T + κ−1d2H2) with constant-time computational cost per
episode. We summarize our main contributions as follows:

• We propose computationally tractable randomized algorithms for RL with MNL function
approximation: RRL-MNL and ORRL-MNL. To the best of our knowledge, these are the first
randomized model-based RL algorithms with MNL function approximation that achieve the
frequentist regret bounds with constant-time computational cost per episode.

• We establish that RRL-MNL enjoys e
O(κ−1d
3
2 H
3
2 √

T) frequentist regret bound with constant-
time computational cost per episode, where d is the dimension of the transition core, H is
horizon length, T is the total number of rounds, and κ is a problem-dependent constant. We

2


---Page Break---
derive the stochastic optimism of RRL-MNL, and to our knowledge, this is the first frequentist
regret analysis for a non-linear model-based algorithm with randomized exploration without
assuming stochastic optimism.
• To achieve a regret bound with improved dependence on κ, we introduce ORRL-MNL, which
constructs the optimistic randomized value functions by taking into account the effects
of the local gradient information for the MNL transition model at each reachable state.
We prove that ORRL-MNL enjoys an e
O(d
3
2 H
3
2 √

T + κ−1d2H2) regret with constant-time
computational cost per episode, significantly improving the regret of RRL-MNL without
requiring prior knowledge of κ.
• We evaluate our algorithms on tabular MDPs and demonstrate the superior performance of
our proposed algorithms compared to the existing state-of-the-art MNL-MDP algorithm [35].
The experiments provide evidence that our proposed algorithms are both computationally
and statistically efficient.

Related works on RL with function approximation and MNL contextual bandits are provided in
Appendix A.

2
Problem Setting

We consider the episodic Markov decision processes (MDPs) denoted by M(S, A, H, {P}H
h=1, r),
where S is the state space, A is the action space, H is the horizon length of each episode, {P}H
h=1
is the collection of probability distributions, and r is the reward function. Every episodes start
from the initial state s1 and for every step h ∈[H] := {1, ..., H} in an episode, the learning agent
interacts with the environment represented as M. The agent observes the state sh ∈S, chooses
an action ah ∈A, receives a reward r(sh, ah) ∈[0, 1] and the next state sh+1 is given by the
transition probability distribution Ph(·|sh, ah). Then this process is repeated throughout the episode.
A policy π : S × [H] →A is a function that determines the action of the agent at state sh, i.e.,
ah = π(sh, h) := πh(sh).

We define the value function of the policy π, denoted by V π
h (s), as the expected sum of re-
wards under the policy π until the end of the episode starting from sh = s, i.e., V π
h (s) =

Eπ

" H
X

h′=h
r(sh′, πh′(sh′)) | sh = s

#

. Similarly, we define the action-value function Qπ
h(s, a) =

r(s, a) + Es′∼Ph(·|s,a)

V π
h+1(s′)

. We define an optimal policy π∗to be a policy that achieves
the highest possible value at every (s, h) ∈S × [H]. We denote the optimal value function by
V ∗
h (s) = V π∗
h (s) and the optimal action-value function by Q∗
h(s, a) = Qπ∗
h (s, a). To simplify, we
introduce the notation PhVh+1(s, a) = Es′∼Ph(·|s,a)[Vh+1(s′)]. Recall that the Bellman equations
are,
Qπ
h(s, a) = r(s, a) + PhV π
h+1(s, a) ,
Q∗
h(s, a) = r(s, a) + PhV ∗
h+1(s, a) ,
where V π
H+1(s) = V ∗
H+1(s) = 0 and V ∗
h (s) = maxa∈A Q∗
h(s, a) for all s ∈S.

The goal of the agent is to maximize the sum of rewards for K episodes. In other words, the goal is to
minimize the cumulative regret of the policy π over K episodes where π = {πk}K
k=1 is a collection
of policies πk at k-th episode. The regret is defined as

Regretπ(K) :=

K
X

k=1
(V ∗
1 −V πk
1
)(sk
1)

where sk
1 is the initial state at the k-th episode.

2.1
Multinomial Logistic Markov Decision Processes (MNL-MDPs)

Even though a lot of provable RL algorithms for linear MDPs are proposed, there is a simple but
fundamental problem with the linear transition model assumption on the linear MDPs. In other
words, the output of a linear function approximating the transition model must be in [0, 1] and the
probability of all possible following states must sum to 1 exactly. Such restrictive assumption can
affect the regret performances of algorithm suggested under the linearity assumption. To resolve

3


---Page Break---
these challenges, Hwang and Oh [35] propose a setting of a multinomial logistic Markov decision
processes (MNL-MDPs), where the state transition model is given by a multinomial logistic model.
We introduce the formal definition for MNL-MDP as follows:
Assumption 1 (MNL-MDPs [35]). An MDP M(S, A, H, {Ph}H
h=1, r) is an MNL-MDP with a
feature map φ : S × A × S →Rd, if for each h ∈[H], there exists θ∗
h ∈Rd, such that for any
(s, a) ∈S × A and s′ ∈Ss,a := {s′ ∈S : P(s′ | s, a) ̸= 0}, the state transition kernel of s′ when
an action a is taken at a state s is given by,

Ph(s′ | s, a) =
exp(φ(s, a, s′)⊤θ∗
h)
P

es∈Ss,a exp(φ(s, a, es)⊤θ∗
h) .
(1)

We call each unknown vector θ∗
h transition core. Furthermore, we denote the maximum cardinality of
the set of reachable states as U, i.e., U := maxs,a |Ss,a|.
Remark 1. While Hwang and Oh [35] assume a homogeneous transition kernel, we assume an
inhomogeneous transition kernel, in which the probability varies depending on the current time step
h even for the same state transition, which is a more general setting. Also, for notational simplicity,
we denote the true transition kernel Ph as Pθ∗
h, and the estimated transition kernel by θ as Pθ.

2.2
Assumptions

We introduce some standard regularity assumptions.
Assumption 2 (Boundedness). We assume ∥φ(s, a, s′)∥2 ≤Lφ for all (s, a, s′) ∈S × A × Ss,a,
and ∥θ∗
h∥2 ≤Lθ for all h ∈[H].
Assumption 3 (Known reward). We assume that the reward function r is known to the agent.
Assumption 4 (Problem-dependent constant). Let Bd(Lθ) := {θ ∈Rd : ∥θ∥2 ≤Lθ}. There exists
κ > 0 such that for any (s, a) ∈S × A and s′, es ∈Ss,a with s′ ̸= es,

inf
θ∈Bd(Lθ) Pθ(s′ | s, a)Pθ(es | s, a) ≥κ .

Discussion of assumptions
Assumption 2 is common in the literature on RL with function approxi-
mation [43, 72, 73, 37, 35] to make the regret bounds scale-free. Assumption 3 is used to focus on
the main challenge of model-based RL that learning about P of the environment is more difficult
than learning r. In the model-based RL literature [71, 9, 72, 81, 35], the known reward r assumption
is widely used. Assumption 4 is typical in generalized linear contextual bandit [26, 51, 23, 3, 24]
and MNL contextual bandit literature [54, 8, 55, 61, 6, 76, 50] to guarantee non-singular Fisher
information matrix.

3
Randomized Algorithm for MNL-MDPs having constant-time
computational cost

Previous work for MNL-MDPs [35] proposed a UCB-based exploration algorithm. Constructing
a UCB-based optimistic value function is not only computationally intractable but also tends to
overly optimistically estimate the true optimal value function. Additionally, their algorithm incurs
increasing computation costs as episodes progress, as it requires all samples from the previous episode
to estimate the transition core. In this section, we present a novel model-based RL algorithm that
incorporates randomized exploration and online parameter estimation for MNL-MDPs.

3.1
Algorithm: RRL-MNL

Online transition core estimation
While Hwang and Oh [35] estimate the transition core using
maximum likelihood estimation over all samples from previous episodes, we employ an efficient
online parameter estimation method by exploiting the particular structure of the MNL transition
model. The key insight is that the negative log-likelihood function for the MNL model in each
episode is strongly convex over a bounded domain. This property allows us to utilize a variation of
the online Newton step [30, 31], which inspired online algorithms for logistic bandits [74] and MNL
contextual bandits [55]. Specifically, for (k, h) ∈[K] × [H], we define the response variable yk
h =

4


---Page Break---
Algorithm 1 RRL-MNL (Randomized RL for MNL-MDPs)

1: Inputs: Episodic MDP M, Feature map φ : S × A × S →Rd, Number of episodes K,
Regularization parameter λ, Exploration variance {σk}K
k=1, Sample size M, Problem-dependent
constant κ
2: Initialize: θ1
h = 0d, A1,h = λId for h ∈[H]
3: for episode k = 1, 2, · · · , K do

4:
Observe sk
1 and sample i.i.d. noise vector ξ(m)
k,h ∼N(0d, σ2
kA−1
k,h) for m ∈[M] and h ∈[H]

5:
Set

Qk
h(·, ·)
	

h∈[H] as described in (4)

6:
for horizon h = 1, 2, · · · , H do
7:
Select ak
h = argmaxa∈A Qk
h(sk
h, a) and observe sk
h+1
8:
Update Ak+1,h = Ak,h + κ

2
P

s′∈Sk,hφ(sk
h, ak
h, s′)φ(sk
h, ak
h, s′)⊤and θk+1
h
as in (2)
9:
end for
10: end for


yk
h(s′)


s′∈Sk,h such that yk
h(s′) = 1(sk
h+1 = s′) for s′ ∈Sk,h := Ssk
h,ak
h. Then, yk
h is sampled from

the following multinomial distribution: yk
h ∼multinomial(1,

Pθ∗
h(s′ | sk
h, ak
h)


s′∈Sk,h), where 1

represents that yk
h is a single-trial sample. We define the per-episode loss ℓk,h(θ) as follows:

ℓk,h(θ) := −
X

s′∈Sk,h
yk
h(s′) log Pθ(s′ | sk
h, ak
h) .

Then, the estimated transition core for θ∗
h is given by

θk
h = argmin
θ∈Bd(Lθ)

1
2∥θ −θk−1
h
∥2
Ak,h + (θ −θk−1
h
)⊤∇ℓk−1,h(θk−1
h
) ,
(2)

where θ1
h can be initialized as any point in Bd(Lθ) and Ak,h is the Gram matrix defined by

Ak,h := λId + κ

2

k−1
X

i=1

X

s′∈Si,h
φ(si
h, ai
h, s′)φ(si
h, ai
h, s′)⊤.
(3)

Stochastically optimistic value function
First of all, we introduce the key challenges of regret
analysis for randomized algorithms, explain how previous works have overcome these challenges, and
then describe why the techniques from previous works cannot be applied to MNL-MDPs. Ensuring
that the estimated value function is optimistic with sufficient frequency is a crucial challenge in
analyzing the frequentist regret of randomized algorithms. A common way to promote sufficient
exploration in randomized algorithms is by perturbing the estimated value function or by performing
posterior sampling in the transition model class. Frequentist regret analysis of randomized exploration
in an RL setting has been conducted for tabular [59, 7, 62, 60, 67], linear MDPs [73, 37], and general
function classes [37, 4, 5, 75]. In the case of linear MDPs [73, 37], since the property that the
action-value function is linear in the feature map allows perturbing the estimated parameter directly to
control the perturbation of the estimated value function. Also, even though Ishfaq et al. [37] presented
a randomized algorithm for the general function class using eluder dimension, they assume stochastic
optimism (anti-concentration), which is in fact one of the most challenging aspects of frequentist
analysis. Other posterior sampling algorithms in RL for the general function class such as [4, 5, 75],
except for very limited examples, do not discuss how to define the posterior distribution supported by
the given function class and how to draw the optimistic sample from the posterior. That is why even
after there exists a so-called general function class-based result, it is often the case that results in
specific parametric models are still needed.

Note that in episodic RL, the perturbed estimated value functions are propagated back through
horizontal steps, requiring careful adjustment of the perturbation scheme to maintain a sufficient
probability of optimism without decaying too quickly with the horizon. For example, if the probability
of the estimated value function being optimistic at horizon h is denoted as p, this would result
in the probability that the estimated value function in the initial state is optimistic being on the
order of pH, implying that the regret can increase exponentially with the length of the horizon H.

5


---Page Break---
Additionally, the non-linearity and substitution effect of the next state transition in the MNL-MDPs
make applying the existing TS techniques infeasible to guarantee optimism in MNL-MDPs with
sufficient frequency. Instead, we design the stochastically optimistic value function by exploiting
the structure of the MNL transition model. In other words, the prediction error of MNL transition
model (Definition 1) can be bounded by the weighted norm of the dominant feature ˆφ (Lemma 4).
Based on such dominant feature, we perturb the estimated value function by injecting Gaussian noise
whose variance is proportional to the inverse of the Gram matrix to encourage the perturbation with
higher variance in less explored directions. To guarantee the optimism with fixed probability, we
adapt optimistic sampling technique [7, 54, 37, 36]. For each m ∈[M], sample i.i.d. Gaussian noise
vector ξ(m)
k,h ∼N(0d, σ2
kA−1
k,h) where σk is an exploration parameter, and add the most optimistic

inner product value maxm∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h to the estimated value function. To summarize for
any (s, a) ∈S × A, set Qk
H+1(s, a) = 0 and for h ∈[H],

Qk
h(s, a) = min

r(s, a) +
X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h , H

,
(4)

where
V k
h (s)
=
maxa′ Qk
h(s, a′)
and
ˆφk,h(s, a)
:=
φ(s, a, ˆs)
for
ˆs
=
argmaxs′∈Ss,a ∥φ(s, a, s′)∥A−1
k,h.
Based on these stochastically optimistic value function,

the agent plays a greedy action ak
h = argmaxa′ Qk
h(sk
h, a′). We layout the procedure in Algorithm 1.

Remark 2. Note that RRL-MNL only requires constant-time computational cost and storage cost per
episode, as it does not require storing all samples from previous episodes, and the Gram matrix Ak,h
can be updated incrementally.

3.2
Regret bound of RRL-MNL

We present the regret upper bound of RRL-MNL. The complete proof is deferred to Appendix C.

Theorem 1 (Regret Bound of RRL-MNL). Suppose that Assumption 1- 4 hold. For any 0 < δ < Φ(−1)

2
,
if we set the input parameters in Algorithm 1 as λ = L2
φ, σk = e
O(H
√

d) and M = ⌈1 −
log H
log Φ(1)⌉
where Φ is the normal CDF, then with probability at least 1 −δ, the cumulative regret of the RRL-MNL
policy π is upper-bounded as follows:

Regretπ(K) = e
O

κ−1d
3
2 H
3
2 √

T

,

where T = KH is the total number of steps.

Discussion of Theorem 1
To our best knowledge, this is the first result to provide a frequentist
regret bound for the MNL-MDPs. Among the previous RL algorithms using function approxima-
tion, the most comparable techniques to our method are model-free algorithms with randomized
exploration [73, 37]. To guarantee stochastic optimism, Zanette et al. [73] established a lower bound
on the difference between the estimated value and the optimal value by the summation of linear
terms with respect to the average feature (Lemma F.1 in [73]). This property is achievable due to the
linear expression of the value function in linear MDPs. Instead, we established a lower bound on
the difference between value functions by the summation of the Bellman errors (Definition 1) along
the sample path obtained through the optimal policy (Lemma 7). Hence, our analysis significantly
differs from that of Zanette et al. [73] since the value function in MNL-MDPs is no longer linearly
parametrized, and there is no closed-form expression for it.

Compared to [37], they also used an optimistic sampling technique; however, our theoretical sampling
size M = O(log H) is much tighter than that of [37], i.e., O(d) for the linear function class,
O(log(T|S||A|)) for the general function class. While Ishfaq et al. [37] extend the results of
the linear function class to general function class under the assumption of stochastic optimism
(Assumption C in [37]), we provide the frequentist regret analysis for a non-linear model-based
algorithm with randomized exploration without assuming stochastic optimism.

Compared to the optimistic exploration algorithm for MNL-MDPs [35], our randomized exploration
requires a more involved proof technique to ensure that the perturbation of the estimated value
function has enough variance to maintain optimism with sufficient frequency (Lemma 6). As a result,

6


---Page Break---
the established regret of RRL-MNL differs by a factor of
√

d, which aligns with the difference in the
existing bounds of linear bandits between a TS-based algorithm [2] and a UCB-based algorithm [1].
Additionally, we achieve statistical efficiency for the inhomogeneous transition model, which is a
more general setting than that of Hwang and Oh [35]. Our computation cost per episode is O(1)
while the computation cost per episode of Hwang and Oh [35] is O(K).

Proof Sketch of Theorem 1
We provide the proof sketch of Theorem 1. By decomposing the regret
into the estimation part and the pessimism part, we have

K
X

k=1
(V ∗
1 −V πk
1 )(sk
1) =

K
X

k=1


V ∗
1 −V k
1
|
{z
}
Pessimism

+ V k
1 −V πk
1
|
{z
}
Estimation


(sk
1) .

We bound these two parts separately. For the estimation part, for each k ∈[K], h ∈[H], we first
show that the online estimated transition core θk
h (2) concentrates around the unknown transition
core parameter θ∗
h with high probability (Lemma 1). Then, we show that the prediction error induced
by the estimated transition core can be bounded by the weighted norm of the dominant feature
ˆφ, multiplied by the confidence radius of the estimated transition core (Lemma 4). The bounded
prediction error, together with the concentration of Gaussian noise, implies the desired bound on the
estimation part (Lemma 10). For the pessimism part, we first show that the stochastically optimistic
value function V k
1 is optimistic than the true optimal value function V ∗
1 with sufficient frequency
(Lemma 6). In the next step, we show that the pessimism part is upper bounded by a bound of the
estimation part times the inverse probability of being optimistic (Lemma 11). Combining all the
results, we can conclude the proof. Refer to Appendix C for detailed proofs.

4
Statistically Improved Algorithm for MNL-MDPs

Although RRL-MNL is provably efficient and achieves constant-time computational cost per episode,
the current analysis makes its regret bound scale with κ−1. Recall that the problem-dependent
constant κ introduced in Assumption 4 indicates the curvature of the MNL function, i.e., how difficult
it is to learn the true transition core parameter. It is required to ensure the non-singular Fisher
information matrix, hence is typically used in GLM or MNL bandit algorithms that use the maximum
likelihood estimator. As introduced in Faury et al. [23], κ−1 can be exponentially large in the worst
case. The appearance of κ in existing bounds originates in the connection between the difference of
estimators and the difference of gradients of negative log-likelihood, usually denoted as G in Filippi
et al. [26]. Without considering local information at all, using a loose lower bound for G incurs
κ−1 in regret bound (see Section 4.1 in Agrawal et al. [6]). Recently, improved dependence on κ
has been achieved in bandit literature [23, 3, 61, 6, 76, 50] through the use of generalization of the
Bernstein-like tail inequality [23] and the self-concordant-like property of the log loss [11]. However,
a direct adaptation of the MNL bandit technique would result in sub-optimal dependence on the
assortment size in MNL bandit, which corresponds to the size of the set of reachable states, such as
U. In this section, we introduce a new randomized algorithm for MNL-MDPs, equipped with a tight
online parameter estimation and feature centralization technique that achieves a regret bound with
improved dependence on κ and U.

4.1
Algorithms: ORRL-MNL

Tight online transition core estimation
Zhang and Sugiyama [76] presented a jointly efficient
UCB-based MNL contextual bandit algorithm using online mirror descent algorithm. Adapting the
update rule from [76], the estimated transition core run by the online mirror descent is given by

eθk+1
h
= argmin
θ∈Bd(Lθ)

1
2η ∥θ −eθk
h∥2
eBk,h + θ⊤∇ℓk,h(eθk
h) ,
(5)

where eθ1
h can be initialized as any point in Bd(Lθ), η is a step size, and eBk,h is defined as

eBk,h := Bk,h + η∇2ℓk,h(eθk
h) ,
Bk,h := λId +

k−1
X

i=1
∇2ℓi,h(eθi+1
h
) .
(6)

7


---Page Break---
Algorithm 2 ORRL-MNL (Optimistic Randomized RL for MNL-MDPs)

1: Inputs: Episodic MDP M, Feature map φ : S × A × S →Rd, Number of episodes K,
Regularization parameter λ, Exploration variance {σk}K
k=1, Confidence radius {βk}K
k=1, Sample
size M, Step size η
2: Initialize: eθ1
h = 0d, B1,h = λId for all h ∈[H]
3: for episode k = 1, 2, · · · , K do

4:
Observe sk
1 and sample i.i.d. noise vector ξ(m)
k,h ∼N(0d, σ2
kB−1
k,h) for m ∈[M] and h ∈[H]

5:
Set
n
eQk
h(·, ·)
o

h∈[H] as described in (7)

6:
for horizon h = 1, 2, · · · , H do
7:
Select ak
h = argmaxa∈A eQk
h(sk
h, a) and observe sk
h+1
8:
Update eBk,h = Bk,h + η∇2ℓk,h(eθk
h) and eθk+1
h
as in (5)

9:
Update Bk+1,h = Bk,h + ∇2ℓk,h(eθk+1
h
)
10:
end for
11: end for

Note that the MNL model in Zhang and Sugiyama [76] operates in a multiple-parameter setting,
where there are multiple unknown choice parameters and one given context feature. In contrast,
our MNL model operates in a single-parameter setting, where there is one unknown transition
core and features for up to U reachable states. This difference results in variations in applying the
self-concordant-like property of the log-loss for the MNL model. For instance, Zhang and Sugiyama
[76] utilized the fact that the log-loss for the multiple parameter MNL model is
√

6-self-concordant-
like (Lemma 2 in Zhang and Sugiyama [76]). On the other hand, Lee and Oh [50] revisit the
self-concordant-like property and demonstrate that the log-loss of the single-parameter MNL model
is 3
√

2-self-concordant-like (Proposition B.1 in Lee and Oh [50]). This results in a concentration
bound that is independent of κ and U, introduced in Lemma 12.

Remark 3. Note that the online estimated parameters θk
h (2) and eθk
h (5) do not aim to minimize
the sum of negative log-likelihoods, Pk
k′=1 ℓk′,h(θ). Instead, we show that the online estimated
parameter concentrates around the unknown transition core θ∗
h with high probability (Lemma 1
& 12). This online update approach allows us to estimate the transition core with constant-time
computational cost per episode, as the agent does not need to store all samples from previous
episodes.

Optimistic randomized value function
To achieve improved dependence on κ, a crucial point
is to utilize the local gradient information of MNL transition probabilities for each reachable state
when constructing the Gram matrix. In MNL bandit problems [61, 76], this can be accomplished by
substituting the Hessian of the negative log-likelihood with the Gram matrix using global gradient
information κ. However, there are fundamental differences between the settings in Perivier and
Goyal [61], Zhang and Sugiyama [76] and ours. Perivier and Goyal [61] address the case where
the reward for each product is uniform (i.e., all products have a reward of 1), and the reward for not
selecting a product from the given assortment (also known as the outside option) is 0. On the other
hand, Zhang and Sugiyama [76] deal with non-uniform rewards where the reward for each product
may vary; however, the rewards for individual products are known a priori to the agent. In contrast, in
MNL-MDPs, the value for each reachable state may vary (non-uniform) and is not known beforehand.
Due to these differences, the analysis techniques in MNL bandits [61, 76] cannot be directly applied
to our setting. Instead, we adapt the feature centralization technique [50]. Then, the Hessian of the
per-round loss ℓk,h(θ) is expressed in terms of the centralized feature as follows:

∇2ℓk,h(θ) =
X

s′∈Sk,h
Pθ(s′ | sk
h, ak
h)¯φ(sk
h, ak
h, s′; θ)¯φ(sk
h, ak
h, s′; θ)⊤.

where ¯φ(s, a, s′; θ) := φ(s, a, s′)−Ees∼Pθ(·|s,a)[φ(s, a, es)] is the centralized feature by θ. For more
details, please refer to Appendix D.2.

Now we introduce the optimistic randomized value function eQk
h(·, ·) for ORRL-MNL. The key point is
that when perturbing the estimated value function, we use the centralized feature by the estimated

8


---Page Break---
transition parameter eθk
h. For any (s, a) ∈S × A, set eQk
H+1(s, a) = 0 and for each h ∈[H],

eQk
h(s, a) := min

r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) + νrand
k,h (s, a) , H

,
(7)

where eV k
h (s) := maxa∈A eQk
h(s, a) and νrand
k,h (s, a) is the randomized bonus term defined by

νrand
k,h (s, a) :=
X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φ(s, a, s′; eθk
h)⊤ξs′
k,h + 3Hβ2
k max
s′∈Ss,a ∥φ(s, a, s′)∥2
B−1
k,h .

Here we sample i.i.d. Gaussian noise ξ(m)
k,h ∼N(0d, σ2
kB−1
k,h) for each m ∈[M] and set ξs′
k,h :=

ξm(s′)
k,h
where m(s′) := argmaxm∈[M] ¯φ(s, a, s′; eθk
h)⊤ξm
k,h is the most optimistic sampling index
for a reachable state s′. Based on these optimistic randomized value function, at each episode the
agent plays a greedy action with respect to eQk
h as summarized in Algorithm 2.

Remark 4. Note that the second term in the randomized bonus always has a positive value, but it
rapidly decreases as episode proceeds. While due to the randomness of ξ, the randomized bonus
νrand
k,h
itself cannot be guaranteed to always have a positive value. Consequently, the constructed

value function eQk
h(·, ·) can be optimistic or pessimistic. However, as shown in Lemma 18, optimistic
sampling technique ensures that the optimistic randomized value function eQk
h has at least a constant
probability of being optimistic than the true optimal value function.

Remark 5. As with RRL-MNL, since the transition core is estimated in an online manner and the
Gram matrices with local gradient information Bk,h and eBk,h are updated incrementally, ORRL-MNL
also requires constant-time computational cost and storage cost per-episode. Although ORRL-MNL
requires an additional O(U) computation cost for feature centralization, the computation complexity
order is the same as that of RRL-MNL because it also needs to go over reachable states to calculate
the dominant feature ˆφ, which also incurs a O(U) computation cost. On the other hand, ORRL-MNL
does not require prior knowledge of κ and achieves a regret with a better dependence on κ.

4.2
Regret Bound of ORRL-MNL

We present the regret upper bound of ORRL-MNL. The complete proof is deferred to Appendix D.

Theorem 2 (Regret Bound of ORRL-MNL). Suppose that Assumption 1- 4 hold. For any 0 <
δ <
Φ(−1)

2
, if we set the input parameters in Algorithm 2 as λ = O(L2
φd log U), βk
=
O(
√

d log U log(kH)), σk = Hβk, M = ⌈1 −log(HU)

log Φ(1) ⌉, and η = O(log U), then with probability
at least 1 −δ, the cumulative regret of the ORRL-MNL policy π is upper-bounded as follows:

Regretπ(K) = e
O

d3/2H3/2√

T + κ−1d2H2
,

where T = KH is the total number of time steps.

Dicussion of Theorem 2
Theorem 2 establishes that the leading term in the regret bound does
not suffer from the problem-dependent constant κ−1 and the second term of the regret bound is
independent of the size of set of reachable states. To the extent of our knowledge, this is the first
algorithm that provides a frequentist regret guarantee with improved dependence on κ−1 in MNL-
MDPs. Compared to RRL-MNL, the technical challenge lies in ensuring the stochastic optimism of
the estimated value for ORRL-MNL. Note that the prediction error (Definition 1) for ORRL-MNL is
characterized by two components: one related to the gradient information of the MNL transition
model at each reachable state, and the other related to the dominant feature with respect to the
Gram matrix Bk,h (Lemma 16). Hence, the probability of the Bellman error at each horizon, when
following the optimal policy, being negative can depend on the size of the reachable states. This
implies that the probability of stochastic optimism can be exponentially small, not only in the horizon
H but also in the size of the reachable states U. However, as shown in Lemma 18, this challenge
has been overcome by using a sample size M that logarithmically increases with U, effectively
addressing the issue.

9


---Page Break---
0
2000
4000
6000
8000
10000
The Number of Episodes

1.0

1.5

2.0

2.5

3.0

Episodic Returns

RRL-MNL
UCRL-MNL

ORRL-MNL
Optimal Policy

UCRL-MNL+

(a) S = 4, H = 12

0
2000
4000
6000
8000
10000
The Number of Episodes

1.0

1.5

2.0

2.5

3.0

Episodic Returns

RRL-MNL
UCRL-MNL

ORRL-MNL
Optimal Policy

UCRL-MNL+

(b) S = 8, H = 24

RRL-MNL
UCRL-MNL
ORRL-MNL
UCRL-MNL+
Algorithms

0

1000

2000

3000

4000

Time (seconds)

20.3

4302.3

76.2
70.1

RRL-MNL
UCRL-MNL

ORRL-MNL
UCRL-MNL+

(c) Runtime for 1,000 episodes

Figure 1: Riverswim experiment results

Proof Sketch of Theorem 2
The overall proof pipeline for Theorem 2 is similar to that of Theorem 1.
The main differences lie in the concentration of the estimated transition core (Lemma D.2), the bound
on the prediction error (Lemma D.2), and the stochastic optimism (Lemma 18). Please refer to
Appendix D for detailed proofs.

Optimistic exploration extension
In general, since TS-based randomized exploration requires
a more rigorous proof technique than UCB-based algorithms, our technical ingredients enable the
use of optimistic exploration in a straightforward manner. We introduce UCRL-MNL+ (Algorithm 3)
in the Appendix E, an optimism-based algorithm for MNL-MDPs. It is both computationally and
statistically efficient compared to UCRL-MNL [35], achieving the tightest regret bound for MNL-
MDPs.

Corollary 1. UCRL-MNL+ (Algorithm 3) has e
O(dH3/2√

T + κ−1d2H2) regret with high probability.

5
Numerical Experiments

We perform a numerical evaluation on a variant of RiverSwim [58] to demonstrate practicality of
our proposed algorithms. We compare our algorithms (RRL-MNL, ORRL-MNL, UCRL-MNL+) with the
state-of-the-art UCRL-MNL [35] for MNL-MDPs. For each configuration, we report the averaged
results over 10 independent runs. Figure 1a and 1b show the episodic return of each algorithm, which
is the sum of all the rewards obtained in one episode. First, our proposed algorithms (RRL-MNL,
ORRL-MNL, UCRL-MNL+) outperform UCRL-MNL [35] for both cases of |S| = 4, 8. Second, ORRL-MNL
and UCRL-MNL+ reach the optimal values quickly compared to the other algorithms, demonstrating
improved statistical efficiency. Figure 1c illustrates the comparison in running time of the algorithms
for the first 1,000 episodes. Our proposed algorithms are at least 50 times faster than UCRL-MNL.
These differences become more pronounced as the episodes progress because our algorithms have a
constant computation cost, whereas the computation cost of UCRL-MNL increases over time.

6
Conclusions

We propose randomized algorithms with provable efficiency and constant-time computational cost for
MNL-MDPs. For the first algorithm, RRL-MNL, we use an optimistic sampling technique to ensure
the stochastic optimism of the estimated value functions and provide the frequentist regret analysis.
This is the first frequentist regret analysis for a non-linear model-based algorithm with randomized
exploration without assuming stochastic optimism. To achieve a statistically improved regret bound,
we propose ORRL-MNL by constructing the optimistic randomized value function using the effects of
the local gradient of the MNL transition model equipped with the centralized feature. As a result, we
achieve a frequentist regret guarantee with improved dependence on κ in RL with the MNL transition
model, which is a significant contribution. The effectiveness and practicality of our methods are
supported by numerical experiments.

10


---Page Break---
Acknowledgements

We sincerely thank the anonymous reviewers for their constructive feedback. This work was supported
by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)
(No. 2022R1C1C1006859, 2022R1A4A1030579, and RS-2023-00222663) and by AI-Bio Research
Grant through Seoul National University.

References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. Advances in neural information processing systems, 24:2312–2320, 2011.

[2] Marc Abeille and Alessandro Lazaric. Linear Thompson Sampling Revisited. In Aarti Singh and
Jerry Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 176–184.
PMLR, PMLR, 20–22 Apr 2017.

[3] Marc Abeille, Louis Faury, and Clément Calauzènes. Instance-wise minimax-optimal algorithms
for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pages
3691–3699. PMLR, 2021.

[4] Alekh Agarwal and Tong Zhang. Model-based rl with optimistic posterior sampling: Structural
conditions and sample complexity. Advances in Neural Information Processing Systems, 35:
35284–35297, 2022.

[5] Alekh Agarwal and Tong Zhang. Non-linear reinforcement learning in large action spaces:
Structural conditions and sample-efficiency of posterior sampling. In Conference on Learning
Theory, pages 2776–2814. PMLR, 2022.

[6] Priyank Agrawal, Theja Tulabandhula, and Vashist Avadhanula. A tractable online learning
algorithm for the multinomial logit contextual bandit.
European Journal of Operational
Research, 2023.

[7] Shipra Agrawal and Randy Jia. Posterior sampling for reinforcement learning: worst-case regret
bounds. In Advances in Neural Information Processing Systems, pages 1184–1194, 2017.

[8] Sanae Amani and Christos Thrampoulidis. Ucb-based algorithms for multinomial logistic
regression bandits. Advances in Neural Information Processing Systems, 34:2913–2924, 2021.

[9] Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based rein-
forcement learning with value-targeted regression. In International Conference on Machine
Learning, pages 463–474. PMLR, 2020.

[10] Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for
reinforcement learning. In International Conference on Machine Learning, pages 263–272.
PMLR, 2017.

[11] Francis Bach. Self-concordant analysis for logistic regression. Electronic Journal of Statistics,
4(2):384 – 414, 2010.

[12] Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. The
Annals of Statistics, 2005.

[13] Steven J Bradtke and Andrew G Barto. Linear least-squares algorithms for temporal difference
learning. Machine learning, 22(1):33–57, 1996.

[14] Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy
optimization. In International Conference on Machine Learning, pages 1283–1294. PMLR,
2020.

[15] Nicolo Campolongo and Francesco Orabona. Temporal variability in implicit online learning.
Advances in neural information processing systems, 33:12377–12387, 2020.

11


---Page Break---
[16] Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. Advances in
neural information processing systems, 24, 2011.

[17] Xi Chen, Yining Wang, and Yuan Zhou. Dynamic assortment optimization with changing
contextual information. Journal of machine learning research, 2020.

[18] Zixiang Chen, Chris Junchi Li, Huizhuo Yuan, Quanquan Gu, and Michael Jordan. A general
framework for sample-efficient function approximation in reinforcement learning. In The
Eleventh International Conference on Learning Representations, 2023.

[19] Christoph Dann, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and
Robert E Schapire. On oracle-efficient pac rl with rich observations. In Advances in Neural
Information Processing Systems, volume 31, 2018.

[20] Simon Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dudik, and John
Langford. Provably efficient rl with rich observations via latent state decoding. In International
Conference on Machine Learning, pages 1665–1674. PMLR, 2019.

[21] Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong
Wang. Bilinear classes: A structural framework for provable generalization in rl. In International
Conference on Machine Learning, pages 2826–2836. PMLR, 2021.

[22] Simon S. Du, Sham M. Kakade, Ruosong Wang, and Lin F. Yang. Is a good representation
sufficient for sample efficient reinforcement learning? In 8th International Conference on
Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, 2020.

[23] Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic
algorithms for logistic bandits. In International Conference on Machine Learning, pages
3052–3060. PMLR, 2020.

[24] Louis Faury, Marc Abeille, Kwang-Sung Jun, and Clément Calauzènes. Jointly efficient and
optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence
and Statistics, pages 546–580. PMLR, 2022.

[25] Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes,
Mohammadamin Barekatain, Alexander Novikov, Francisco J R Ruiz, Julian Schrittwieser,
Grzegorz Swirszcz, et al. Discovering faster matrix multiplication algorithms with reinforcement
learning. Nature, 610(7930):47–53, 2022.

[26] Sarah Filippi, Olivier Cappé, Aurélien Garivier, and Csaba Szepesvári. Parametric bandits:
The generalized linear case. In Proceedings of the 23rd International Conference on Neural
Information Processing Systems - Volume 1, NIPS’10, page 586–594, Red Hook, NY, USA,
2010. Curran Associates Inc.

[27] Dylan J Foster, Satyen Kale, Haipeng Luo, Mehryar Mohri, and Karthik Sridharan. Logistic
regression: The importance of being improper. In Conference On Learning Theory, pages
167–208. PMLR, 2018.

[28] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity
of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.

[29] David A Freedman. On tail probabilities for martingales. the Annals of Probability, pages
100–118, 1975.

[30] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex
optimization. Machine Learning, 69(2):169–192, 2007.

[31] Elad Hazan, Tomer Koren, and Kfir Y Levy. Logistic regression: Tight bounds for stochastic
and online optimization. In Conference on Learning Theory, pages 197–209. PMLR, 2014.

[32] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in
Optimization, 2(3-4):157–325, 2016.

12


---Page Break---
[33] Jiafan He, Dongruo Zhou, and Quanquan Gu. Logarithmic regret for reinforcement learning
with linear function approximation. In International Conference on Machine Learning, pages
4171–4180. PMLR, 2021.

[34] Jiafan He, Heyang Zhao, Dongruo Zhou, and Quanquan Gu. Nearly minimax optimal reinforce-
ment learning for linear markov decision processes. In International Conference on Machine
Learning, pages 12790–12822. PMLR, 2023.

[35] Taehyun Hwang and Min-hwan Oh. Model-based reinforcement learning with multinomial
logistic function approximation. In Proceedings of the AAAI conference on artificial intelligence,
pages 7971–7979, 2023.

[36] Taehyun Hwang, Kyuwook Chai, and Min-Hwan Oh. Combinatorial neural bandits. In
Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023.

[37] Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina
Precup, and Lin Yang. Randomized exploration in reinforcement learning with general value
function approximation. In International Conference on Machine Learning, volume 139, pages
4607–4616. PMLR, PMLR, 2021.

[38] Haque Ishfaq, Qingfeng Lan, Pan Xu, A. Rupam Mahmood, Doina Precup, Anima Anandkumar,
and Kamyar Azizzadenesheli. Provable and practical: Efficient exploration in reinforcement
learning via langevin monte carlo. In The Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.net/forum?id=nfIAEJFiBZ.

[39] Haque Ishfaq, Yixin Tan, Yu Yang, Qingfeng Lan, Jianfeng Lu, A Rupam Mahmood, Doina
Precup, and Pan Xu. More efficient randomized exploration for reinforcement learning via
approximate sampling. Reinforcement Learning Journal, 3(1), 2024.

[40] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement
learning. Journal of Machine Learning Research, 11(4), 2010.

[41] Zeyu Jia, Lin Yang, Csaba Szepesvari, and Mengdi Wang. Model-based reinforcement learning
with value-targeted regression. In Learning for Dynamics and Control, pages 666–686. PMLR,
2020.

[42] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire.
Contextual decision processes with low bellman rank are pac-learnable. In International
Conference on Machine Learning, pages 1704–1713. PMLR, 2017.

[43] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement
learning with linear function approximation. In Conference on Learning Theory, pages 2137–
2143. PMLR, 2020.

[44] Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes
of rl problems, and sample-efficient algorithms. Advances in neural information processing
systems, 34:13406–13418, 2021.

[45] Kwang-Sung Jun, Aniruddha Bhargava, Robert Nowak, and Rebecca Willett. Scalable gen-
eralized linear bandits: Online computation and hashing. Advances in Neural Information
Processing Systems, 30, 2017.

[46] Yeoneung Kim, Insoon Yang, and Kwang-Sung Jun. Improved regret analysis for variance-
adaptive linear bandits and horizon-free linear mixture mdps. Advances in Neural Information
Processing Systems, 35:1060–1072, 2022.

[47] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey.
The International Journal of Robotics Research, 32(11):1238–1274, 2013.

[48] Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Pac reinforcement learning with
rich observations. Advances in Neural Information Processing Systems, 29:1840–1848, 2016.

[49] Branislav Kveton, Csaba Szepesvári, Mohammad Ghavamzadeh, and Craig Boutilier. Perturbed-
history exploration in stochastic linear bandits. In Uncertainty in Artificial Intelligence, pages
530–540. PMLR, 2020.

13


---Page Break---
[50] Joongkyu Lee and Min-hwan Oh. Nearly minimax optimal regret for multinomial logistic
bandit. arXiv preprint arXiv:2405.09831, 2024.

[51] Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear
contextual bandits. In International Conference on Machine Learning, pages 2071–2080.
PMLR, 2017.

[52] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G
Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

[53] Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder Singh. Sample complexity of rein-
forcement learning using linearly combined model ensembles. In International Conference on
Artificial Intelligence and Statistics, pages 2010–2020. PMLR, 2020.

[54] Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits.
Advances in Neural Information Processing Systems, 32:3151–3161, 2019.

[55] Min-hwan Oh and Garud Iyengar. Multinomial logit contextual bandits: Provable optimality and
practicality. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9205–9213,
2021.

[56] Ian Osband and Benjamin Van Roy. Model-based reinforcement learning and the eluder
dimension. In Advances in Neural Information Processing Systems, pages 1466–1474, 2014.

[57] Ian Osband and Benjamin Van Roy. Why is posterior sampling better than optimism for
reinforcement learning? In International conference on machine learning, pages 2701–2710.
PMLR, 2017.

[58] Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via
posterior sampling. Advances in Neural Information Processing Systems, 26, 2013.

[59] Ian Osband, Benjamin Van Roy, and Zheng Wen. Generalization and exploration via randomized
value functions. In International Conference on Machine Learning, pages 2377–2386. PMLR,
2016.

[60] Aldo Pacchiano, Philip Ball, Jack Parker-Holder, Krzysztof Choromanski, and Stephen Roberts.
Towards tractable optimism in model-based reinforcement learning. In Uncertainty in Artificial
Intelligence, pages 1413–1423. PMLR, 2021.

[61] Noemie Perivier and Vineet Goyal. Dynamic pricing and assortment under a contextual mnl
demand. Advances in Neural Information Processing Systems, 35:3461–3474, 2022.

[62] Daniel Russo.
Worst-case regret bounds for exploration via randomized value functions.
Advances in Neural Information Processing Systems, 32, 2019.

[63] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic
exploration. In Advances in Neural Information Processing Systems, pages 2256–2264, 2013.

[64] Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al. A tutorial
on thompson sampling. Foundations and Trends® in Machine Learning, 11(1):1–96, 2018.

[65] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur
Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of
go without human knowledge. nature, 550(7676):354–359, 2017.

[66] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur
Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general
reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science,
362(6419):1140–1144, 2018.

[67] Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey
Naumov, Mark Rowland, Michal Valko, and Pierre Ménard. Optimistic posterior sampling for
reinforcement learning with few samples and tight guarantees. Advances in Neural Information
Processing Systems, 35:10737–10751, 2022.

14


---Page Break---
[68] Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general
value function approximation: Provably efficient approach via bounded eluder dimension.
Advances in Neural Information Processing Systems, 33, 2020.

[69] Yining Wang, Ruosong Wang, Simon Shaolei Du, and Akshay Krishnamurthy. Optimism in
reinforcement learning with generalized linear function approximation. In 9th International
Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021,
2021.

[70] Gellért Weisz, Philip Amortila, and Csaba Szepesvári. Exponential lower bounds for planning in
mdps with linearly-realizable optimal action-value functions. In Algorithmic Learning Theory,
pages 1237–1264. PMLR, 2021.

[71] Lin Yang and Mengdi Wang. Sample-optimal parametric q-learning using linearly additive
features. In International Conference on Machine Learning, pages 6995–7004. PMLR, 2019.

[72] Lin Yang and Mengdi Wang. Reinforcement learning in feature space: Matrix bandit, kernels,
and regret bound. In International Conference on Machine Learning, pages 10746–10756.
PMLR, 2020.

[73] Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro
Lazaric. Frequentist regret bounds for randomized least-squares value iteration. In International
Conference on Artificial Intelligence and Statistics, pages 1954–1964. PMLR, 2020.

[74] Lijun Zhang, Tianbao Yang, Rong Jin, Yichi Xiao, and Zhi-Hua Zhou. Online stochastic linear
optimization under one-bit feedback. In International Conference on Machine Learning, pages
392–401. PMLR, 2016.

[75] Tong Zhang. Feel-good thompson sampling for contextual bandits and reinforcement learning.
SIAM Journal on Mathematics of Data Science, 4(2):834–857, 2022.

[76] Yu-Jie Zhang and Masashi Sugiyama. Online (multinomial) logistic bandit: Improved regret
and constant computation cost. In Thirty-seventh Conference on Neural Information Processing
Systems, 2023.

[77] Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learn-
ingvia reference-advantage decomposition. In Advances in Neural Information Processing
Systems, volume 33, pages 15198–15207, 2020.

[78] Zihan Zhang, Jiaqi Yang, Xiangyang Ji, and Simon S Du. Improved variance-aware confidence
sets for linear bandits and linear mixture mdp. Advances in Neural Information Processing
Systems, 34:4342–4355, 2021.

[79] Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Model-free reinforcement learning: from clipped
pseudo-regret to sample complexity. In International Conference on Machine Learning, pages
12653–12662. PMLR, 2021.

[80] Dongruo Zhou and Quanquan Gu. Computationally efficient horizon-free reinforcement learning
for linear mixture mdps. Advances in neural information processing systems, 35:36337–36349,
2022.

[81] Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement
learning for linear mixture markov decision processes. In Conference on Learning Theory,
pages 4532–4576. PMLR, 2021.

[82] Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for
discounted mdps with feature mapping. In International Conference on Machine Learning,
pages 12793–12802. PMLR, 2021.

15


---Page Break---
Contents of Appendix

A Related Work
16

B
Notations & Definitions
18

C Detailed Regret Analysis for RRL-MNL (Theorem 1)
22

C.1
Concentration of Estimated Transition Core θk
h
. . . . . . . . . . . . . . . . . . .
23

C.2
Bound on Prediction Error
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32

C.3
Good Events with High Probability . . . . . . . . . . . . . . . . . . . . . . . . . .
33

C.4
Stochastic Optimism
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

C.5
Bound on Estimation Part . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36

C.6
Bound on Pessimism Part . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
38

C.7
Regret Bound of RRL-MNL
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
41

D Detailed Regret Analysis for ORRL-MNL (Theorem 2)
42

D.1
Concentration of Estimated Transition Core eθk
h
. . . . . . . . . . . . . . . . . . .
42

D.2
Bound on Prediction Error
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50

D.3
Good Events with High Probability . . . . . . . . . . . . . . . . . . . . . . . . . .
55

D.4
Stochastic Optimism
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
55

D.5
Bound on Estimation Part . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
57

D.6
Bound on Pessimism Part . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
66

D.7
Regret Bound of ORRL-MNL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
67

E
Optimistic Exploration Extension
67

E.1
Optimism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
69

F
Experiment Details
70

G Auxiliary Lemmas
70

H Limitations
72

A
Related Work

RL with linear function approximation
There has been a growing interest in studies that extend
beyond tabular MDPs and focus on function approximation methods with provable guarantees [42,
71, 43, 73, 53, 22, 14, 9, 68, 70, 33, 81, 82, 37, 35, 38]. In particular, for minimizing regret in
linear MDPs, Jin et al. [43] propose an optimistic variant of the Least-Squares Value Iteration
(LSVI) algorithm [13, 59] under the assumption that the transition model and reward function of
the MDPs are linear function of a d-dimensional feature mapping and they guarantee e
O(d
3
2 H
3
2 √

T)
regret. Zanette et al. [73] propose a randomized LSVI algorithm that incorporates exploration by
perturbing the least-square approximation of the action-value function, and this algorithm guarantees
e
O(d2H2√

T) regret. Ishfaq et al. [37] propose a variant of the randomized LSVI algorithm that
combines optimism and TS by perturbing the training data with i.i.d. scalar noise, achieving a regret
bound of e
O(d
3
2 H
3
2 √

T). Similarly, Ishfaq et al. [38] introduce a randomized RL algorithm that
employs Langevin Monte Carlo (LMC) to approximate the posterior distribution of the action-value

16


---Page Break---
Table 1: This table compares the problem settings, online update, performance of the this paper with
those of other methods in provable RL with function approximation. For computation cost, we only
keep the dependence on the number of episode K.

Algorithm
Model-based
Transition model
Reward
Computation cost
Regret
LSVI-UCB [43]
✗
Linear
Linear
O(K)
e
O(d
3
2 H
3
2 √

T)
OPT-RLSVI [73]
✗
Linear
Linear
O(K)
e
O(d2H2√

T)
LSVI-PHE [37]
✗
Linear
Linear
O(K)
e
O(d
3
2 H
3
2 √

T)
UC-MatrixRL [72]
✓
Linear
Known
O(K)
e
O(d
3
2 H2√

T)
UCRL-VTR [9]
✓
Linear mixture
Known
O(K)
e
O(dH
3
2 √

T)
UCRL-MNL [35]
✓
MNL
Known
O(K)
e
O(κ−1dH
3
2 √

T)
RRL-MNL (this work)
✓
MNL
Known
O(1)
e
O(κ−1d
3
2 H
3
2 √

T)
ORRL-MNL (this work)
✓
MNL
Known
O(1)
e
O

d
3
2 H
3
2 √

T + κ−1d2H2

UCRL-MNL+ (this work)
✓
MNL
Known
O(1)
e
O

dH
3
2 √

T + κ−1d2H2

function, also ensuring a regret bound of e
O(d
3
2 H
3
2 √

T). Also, there have been studies on model-
based methods with function approximation in linear MDPs, such as Yang and Wang [72], which
assume that the transition probability kernel is a bilinear model parametrized by a matrix and propose
a UCB-based algorithm with an upper bound of e
O(d
3
2 H2√

T) for regret. He et al. [34] propose an
algorithm achieving nearly minimax optimal regret e
O(dH
√

T). Jia et al. [41] consider a specific type
of MDPs called linear mixture MDPs in which the transition probability kernel is a linear combination
of different basis kernels. This model encompasses various types of MDPs studied previously in Modi
et al. [53], Yang and Wang [72]. For this model, Jia et al. [41] propose a UCB-based RL algorithm
with value-targeted model parameter estimation that guarantees an upper bound of e
O(dH
3
2 √

T) for
regret. The same linear mixture MDPs have been used in other studies such as Ayoub et al. [9], Zhou
et al. [81, 82]. Specifically, in Zhou et al. [81], a variant of the method proposed by Jia et al. [41]
is suggested and proved that the algorithm guarantees an upper bound of e
O(dH
√

T) regret with a
matching lower bound of Ω(dH
√

T) for linear mixture MDPs. More recently, there are also works
achieving horizon-free regret bounds for linear mixture MDPs [78, 46, 80].

RL with non-linear function approximation
Studies have been conducted on extending function
approximation beyond linear models. Ayoub et al. [9], Wang et al. [68], Ishfaq et al. [37] provide
upper bound for regret based on eluder dimension [63]. Also, there has been an effort to develop
sample-efficient methods with more “general” function approximation [48, 42, 19–21, 28, 37, 44,
4, 5, 75, 18, 39] However, these attempts may have been hindered by the difficulty of solving
computationally intractable problems [48, 42, 19, 21, 28, 44, 18], the necessity of relying on stronger
assumptions [20, 37], or the lack of discussion on how to define the posterior distribution supported
by a given function class and how to draw the optimistic sample from the posterior [4, 5, 75]. That
is why even after there exists a so-called “general function class”-based result, it is often the case
that the results in specific parametric models are still needed. Despite the large number of studies
on RL with linear function approximation, there is limited research on extending beyond linear
models to other parametric models. Wang et al. [69] use generalized linear function approximation,
where the Bellman backup of any value function is assumed to be a generalized linear function of
feature mapping. Hwang and Oh [35] discuss the limitations of linear function approximation and
propose a UCB-based algorithm for MNL transition model in feature space achieving e
O(dH
3
2 √

T).
Ishfaq et al. [39] present TS-based RL algorithms that utilize approximate samplers, such as LMC or
Underdamped LMC, to enhance the implementation and computational tractability of TS for RL with
general function classes.

Contextual bandits
Faury et al. [23] first provide a UCB-based algorithm with κ-independent
regret for binary logistic bandit and Abeille et al. [3] present UCB & TS based algorithms achieving
nearly minimax optimal regret for the same setting. Faury et al. [24] propose a jointly efficient
UCB-based algorithm that achieve κ-independent regret bound with O(log t) computation cost. In
the context of MNL model, Oh and Iyengar [54] employ TS approach, while Oh and Iyengar [55]
incorporate a combination of UCB exploration and online parameter updates for MNL bandits.
Both of the methods have O(κ−1√

T) regret. Amani and Thrampoulidis [8] propose an optimistic
algorithm with better dependence on κ.
Agrawal et al. [6] design a UCB-based algorithm with

17


---Page Break---
O(
√

T) regret bound without κ in its leading term, and Perivier and Goyal [61] establish O(
p

T/κ∗)
regret for the uniform reward setting. Zhang and Sugiyama [76] develop jointly efficient UCB-based
algorithm for non-uniform MNL bandit problem. Lee and Oh [50] propose nearly minimax optimal
MNL bandit algorithm for both uniform and non-uniform reward structures.

B
Notations & Definitions

In this section, we formally summarize some definitions and notations used to analyze the proposed
algorithm.

Inhomogeneous MNL transition model

For h ∈[H], the probability of state transition to s′ ∈Ss,a when an action a is taken at a state s is
given by

Ph(s′ | s, a) := Pθ∗
h(s′ | s, a) =
exp(φ(s, a, s′)⊤θ∗
h)
P

es∈Ss,a exp(φ(s, a, es)⊤θ∗
h) .

The estimated transition probability parameterized by θ is denoted as

Pθ(s′ | s, a) :=
exp(φ(s, a, s′)⊤θ)
P

es∈Ss,a exp(φ(s, a, es)⊤θ) .

Feature vector

We abbreviate the feature vector as follows:

φs,a,s′ := φ(s, a, s′) for (s, a, s′) ∈S × A × Ss,a ,

φk,h,s′ := φ(sk
h, ak
h, s′) for (k, h) ∈[K] × [H] and s′ ∈Sk,h := Ssk
h,ak
h ,

ˆφk,h(s, a) := φ(s, a, ˆs) for ˆs := argmax
s′∈Ss,a
∥φ(s, a, s′)∥A−1
k,h ,

¯φs,a,s′(θ) := ¯φ(s, a, s′; θ) = φ(s, a, s′) −Ees∼Pθ(·|s,a)[φ(s, a, es)] ,

¯φk,h,s′(θ) := ¯φ(sk
h, ak
h, s′; θ) .

Response variable & per-episode loss

The response variable yk
h is given by

yk
h := [yk
h(s′)]s′∈Sk,h where yk
h(s′) := 1(sk
h+1 = s′) for s′ ∈Sk,h .

The per-episode loss ℓk,h(θ) is given by

ℓk,h(θ) := −
X

s′∈Sk,h
yk
h(s′) log Pθ(s′ | sk
h, ak
h) ,

Gk,h(θ) := ∇ℓk,h(θ) =
X

s′∈Sk,h
(Pθ(s′ | sk
h, ak
h) −yk
h(s′))φk,h,s′ ,

Hk,h(θ) := ∇2ℓk,h(θ)

=
X

s′∈Sk,h
Pθ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′ −
X

s′∈Sk,h

X

es∈Sk,h
Pθ(s′ | sk
h, ak
h)Pθ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,es .

18


---Page Break---
Regularity constants

H : Horizon length
K : Episode number
T = KH : Total number of interactions

Lφ : ℓ2-norm upper bound of φ(s, a, s), i.e., ∥φ(s, a, s′)∥2 ≤Lφ ,

Lθ : ℓ2-norm upper bound of θ∗
h, i.e., ∥θ∗
h∥2 ≤Lθ ,

κ : Problem-dependent constant such that
inf
θ∈Bd(Lθ) Pθ(s′ | s, a)Pθ(es | s, a) ≥κ ,

U : Maximum cardinality of the set of reachable states, i.e., U := max
s,a |Ss,a| .

Estimated transition core

The estimated transition core for RRL-MNL is given by

θk
h = argmin
θ∈Bd(Lθ)

1
2∥θ −θk−1
h
∥2
Ak,h + (θ −θk−1
h
)⊤∇ℓk−1,h(θk−1
h
) ,

and the estimated transition core for ORRL-MNL is given by

eθk+1
h
= argmin
θ∈Bd(Lθ)

1
2η

θ −eθk
h

2

eBk,h
+ θ⊤∇ℓk,h(eθk
h) .

Gram matrices

The Gram matrix with global gradient information κ is given by

Ak,h := λId + κ

2

k−1
X

i=1

X

s′∈Si,h
φ(si
h, ai
h, s′)φ(si
h, ai
h, s′)⊤.

The Gram matrices with local gradient information are given by

eBk,h := Bk,h + η∇2ℓk,h(eθk
h) and Bk,h := λId +

k−1
X

i=1
∇2ℓi,h(eθi+1
h
) .

Confidence radius

For some absolute constants Cβ, Cξ > 0,
αk := αk(δ)

=

s

8d

κ log

1 + kUL2φ

dλ


+
32LφLθ

3
+ 16

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+ 2
√

2 + 2λL2
θ

= e
O(κ−1/2d1/2) ,

βk := βk(δ) = Cβ

s

log U

λ log(Uk) + log(Uk) log
H
√

1 + 2k

δ


+ d log

1 + k

dλ


+ λL2
θ

= O(
√

d log U log(kH)) ,

γk := γk(δ) = Cξσk
p

d log(Md/δ) .

Filtration

For an arbitrary set X, we denote the Σ-algebra generated by X as Σ(X). Then we define the
following filtrations

Fk := Σ

si
j, ai
j, r(si
j, ai
j) | i < k, j ≤H
	
∪
n
ξ(m)
i,j
| i < k, j ≤H, 1 ≤m ≤M
o
,

Fk,h := Σ

Fk ∪

sk
j , ak
j , r(sk
j , ak
j ) | j ≤h
	
∪
n
ξ(m)
k,j | j ≥h, 1 ≤m ≤M
o
.

19


---Page Break---
Pseudo-noise

For RRL-MNL, the pseudo-noise is sampled as

ξ(m)
k,h ∼N(0d, σ2
kA−1
k,h) ,

and for ORRL-MNL, the pseudo-noise is sampled as

ξ(m)
k,h ∼N(0d, σ2
kB−1
k,h) ,

for M times independently.

Estimated value functions

The stochastically optimistic value function for RRL-MNL is defined as follows:

Qk
H+1(s, a) = 0 ,

Qk
h(s, a) = min

r(s, a) +
X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h , H

for h ∈[H] .

The optimistic randomized value function for ORRL-MNL is defined as follows:

eQk
H+1(s, a) = 0 ,

eQk
h(s, a) := min

r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) + νrand
k,h (s, a) , H

for h ∈[H] ,

where

νrand
k,h (s, a) :=
X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φ(s, a, s′; eθk
h)⊤ξs′
k,h + 3Hβ2
k max
s′∈Ss,a ∥φ(s, a, s′)∥2
B−1
k,h ,

ξs′
k,h := ξm(s′)
k,h
for m(s′) := argmax
m∈[M]
¯φ(s, a, s′; eθk
h)⊤ξm
k,h .

Prediction error & Bellman error

Definition 1 (Prediction error & Bellman error). For any (s, a) ∈S × A and (k, h) ∈[K] × [H],
we define the prediction error about θk
h as

∆k
h(s, a) :=
X

s′∈Ss,a


Pθk
h(s′ | s, a) −Pθ∗
h(s′|s, a)

V k
h+1(s′) .

Also we define the Bellman error as follows:

ιk
h(s, a) := r(s, a) + PhV k
h+1(s, a) −Qk
h(s, a) .

Good events

For any δ ∈(0, 1), we define the following good events:

For RRL-MNL,

G∆
k,h(δ) :=
n
|∆k
h(s, a)| ≤Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h

o
,

Gξ
k,h(δ) :=

max
m∈[M]∥ξ(m)
k,h ∥Ak,h ≤γk(δ)

,

Gk,h(δ) :=
n
G∆
k,h(δ) ∩Gξ
k,h(δ)
o
,

Gk(δ) :=
\

h∈[H]
Gk,h(δ) ,

G(K, δ) :=
\

k≤K
Gk(δ) .

20


---Page Break---
For ORRL-MNL,

G∆
k,h(δ) :=

|∆k
h(s, a)| ≤Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

+ 3Hβk(δ)2 max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h


,

Gξ
k,h(δ) :=

max
m∈[M]∥ξ(m)
k,h ∥Bk,h ≤γk(δ)

,

Gk,h(δ) :=
n
G∆
k,h(δ) ∩Gξ
k,h(δ)
o
,

Gk(δ) :=
\

h∈[H]
Gk,h(δ) ,

G(K, δ) :=
\

k≤K
Gk(δ) .

Derivative of MNL transition model

Proposition 1 (Derivative of MNL transition model). The gradient and Hessian of Pθ(· | ·, ·) can be
calculated as follows:

∇Pθ(s′ | s, a) = Pθ(s′ | s, a)



φs,a,s′ −
X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′





= Pθ(s′ | s, a)¯φs,a,s′(θ) ,

(8)

and
∇2Pθ(s′ | s, a)

= Pθ(s′ | s, a)φs,a,s′φ⊤
s,a,s′

−Pθ(s′ | s, a)
X

s′′∈Ss,a
Pθ(s′′ | s, a)
 
φs,a,s′φ⊤
s,a,s′′ + φs,a,s′′φ⊤
s,a,s′ + φs,a,s′′φ⊤
s,a,s′′


+ 2Pθ(s′ | s, a)



X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′







X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′





⊤

.

(9)

Proof of Proposition 1. Let θ = (θ1, . . . , θd) and [φs,a,s′]i be the i-th component of φs,a,s′. Then,
we have
∂
∂θj
Pθ(s′ | s, a)

=
exp
 
φ⊤
s,a,s′θ

[φs,a,s′]j
P

s′′∈Ss,a exp

φ⊤
s,a,s′′θ
 −
exp
 
φ⊤
s,a,s′θ
 P

s′′∈Ss,a exp
 
φ⊤
s,a,s′′θ

[φs,a,s′′]j
P

s′′∈Ss,a exp

φ⊤
s,a,s′′θ
2

= Pθ(s′ | s, a)



[φs,a,s′]j −
X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]j



.

Then, the gradient of Pθ(s′ | s, a) is given by

∇Pθ(s′ | s, a) = Pθ(s′ | s, a)φs,a,s′ −Pθ(s′ | s, a)
X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′

= Pθ(s′ | s, a)



φs,a,s′ −
X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′





= Pθ(s′ | s, a)¯φs,a,s′(θ) .

21


---Page Break---
On the other hand, the second derivative
∂
∂θi∂θj Pθ(s′ | s, a) can be obtained as follows:

∂
∂θi∂θj
Pθ(s′ | s, a)

= Pθ(s′ | s, a)



[φs,a,s′]i −
X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]i





·



[φs,a,s′]j −
X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]j





+ Pθ(s′ | s, a)



−
X

s′′∈Ss,a
Pθ(s′′ | s, a)



[φs,a,s′′]i −
X

es∈Ss,a
Pθ(es | s, a)[φs,a,es]i



[φs,a,s′′]j





= Pθ(s′ | s, a)

(

[φs,a,s′]i[φs,a,s′]j

−
X

s′′∈Ss,a
Pθ(s′′ | s, a)
 
[φs,a,s′′]i[φs,a,s′]j + [φs,a,s′]i[φs,a,s′′]j


+



X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]i







X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]j





−
X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]i[φs,a,s′′]j

+



X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]j







X

es∈Ss,a
Pθ(es | s, a)[φs,a,es]i










= Pθ(s′ | s, a)

[φs,a,s′]i[φs,a,s′]j

−
X

s′′∈Ss,a
Pθ(s′′ | s, a)
 
[φs,a,s′′]i[φs,a,s′]j + [φs,a,s′]i[φs,a,s′′]j


−
X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]i[φs,a,s′′]j

+2



X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]i







X

s′′∈Ss,a
Pθ(s′′ | s, a)[φs,a,s′′]j








.

Thus, we get the desired result as follows:
∇2Pθ(s′ | s, a)

= Pθ(s′ | s, a)φs,a,s′φ⊤
s,a,s′

−Pθ(s′ | s, a)
X

s′′∈Ss,a
Pθ(s′′ | s, a)
 
φs,a,s′φ⊤
s,a,s′′ + φs,a,s′′φ⊤
s,a,s′ + φs,a,s′′φ⊤
s,a,s′′


+ 2Pθ(s′ | s, a)



X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′







X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′





⊤

.

C
Detailed Regret Analysis for RRL-MNL (Theorem 1)

In this section, we provide the complete proof of Theorem 1. First, we introduce all the technical
lemmas needed to prove Theorem 1 along with their proofs. At the end of this section, we present the
proof of Theorem 1.

22


---Page Break---
C.1
Concentration of Estimated Transition Core θk
h

In this section, we provide the concentration inequality for the estimated transition core run by the
approximate online Newton step. The proof is similar to that given by Oh and Iyengar [55]. For
completeness, we provide the detailed proof.
Lemma 1 (Concentration of online estimated transition core). For each h ∈[H], if λ ≥L2
φ, then we
have
P

∀k ≥1, ∥θk
h −θ∗
h∥Ak,h ≤αk(δ)

≥1 −δ .

where αk(δ) is given by
αk(δ)

:=

s

8d

κ log

1 + kUL2φ

dλ


+
32LφLθ

3
+ 16

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+ 2
√

2 + 2λL2
θ .

Proof of lemma 1. Recall that the per-round loss ℓk,h(θ) and its gradient Gk,h(θ) is defined as
follows:
ℓk,h(θ) := −
X

s′∈Sk,h
yk
h(s′) log Pθ(s′ | sk
h, ak
h) ,
Gk,h(θ) := ∇θℓk,h(θ) .

For the analysis, we define the conditional expectations of ℓk,h(θ) & Gk,h(θ) as follows:
¯ℓk,h(θ) := Eyk
h [ℓk,h(θ) | Fk,h] ,
¯Gk,h(θ) := Eyk
h[Gk,h(θ) | Fk,h] .

By Taylor expansion with ¯θ = νθk
h + (1 −ν)θ∗
h for some ν ∈(0, 1), we have

ℓk,h(θ∗
h) = ℓk,h(θk
h) + Gk,h(θk
h)⊤(θ∗
h −θk
h) + 1

2(θ∗
h −θk
h)⊤Hk,h(¯θ)(θ∗
h −θk
h) ,
(10)

where Hk,h(θ) is the Hessian of the per-round loss evaluated at θ, i.e.,

Hk,h(θ) := ∇2ℓk,h(θ)
(11)

=
X

s′∈Sk,h
Pθ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−
X

s′,es∈Sk,h
Pθ(s′ | sk
h, ak
h)Pθ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,es .

Note that for ¯θ = νθk
h + (1 −ν)θ∗
h with ν ∈(0, 1), we have

Hk,h(¯θ) =
X

s′∈Sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−
X

s′∈Sk,h

X

es∈Sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,es

=
X

s′∈Sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−1

2

X

s′∈Sk,h

X

es∈Sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)(φk,h,s′φ⊤
k,h,es + φk,h,esφ⊤
i,h,s′)

⪰
X

s′∈Sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−1

2

X

s′∈Sk,h

X

es∈Sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)(φk,h,s′φ⊤
k,h,s′ + φk,h,esφ⊤
k,h,es)

=
X

s′∈Sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−
X

s′∈Sk,h

X

es∈Sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′ ,

23


---Page Break---
where the inequality utilizes the fact that xx⊤+ yy⊤⪰xy⊤+ yx⊤for any x, y ∈Rd. Therefore,
we have
Hk,h(¯θ) ⪰
X

s′∈Sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−
X

s′∈Sk,h

X

es∈Sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

=
X

s′̸= ˙sk,h
P¯θ(s′ | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

−
X

s′̸= ˙sk,h

X

es̸= ˙sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ(es | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

=
X

s′̸= ˙sk,h
P¯θ(s′ | sk
h, ak
h)



1 −
X

es̸= ˙sk,h
P¯θ(es | sk
h, ak
h)



φk,h,s′φ⊤
k,h,s′

=
X

s′̸= ˙sk,h
P¯θ(s′ | sk
h, ak
h)P¯θ( ˙sk,h | sk
h, ak
h)φk,h,s′φ⊤
k,h,s′

⪰
X

s′̸= ˙sk,h
κφk,h,s′φ⊤
k,h,s′

=
X

s′∈Sk,h
κφk,h,s′φ⊤
k,h,s′ ,

where ˙sk,h is the state satisfying φ(sk
h, ak
h, ˙sk,h) = 0d and the last inequality comes from the
Assumption 4.

Using the lower bound of the Hessian of the per-round loss evaluated at ¯θ, from (10) we have

ℓk,h(θ∗
h) ≥ℓk,h(θk
h)+Gk,h(θk
h)⊤(θ∗
h −θk
h)+ κ

2 (θ∗
h −θk
h)⊤



X

s′∈Sk,h
φk,h,s′φ⊤
k,h,s′



(θ∗
h −θk
h) .

By rearranging, we have

ℓk,h(θk
h) ≤ℓk,h(θ∗
h) + Gk,h(θk
h)⊤(θk
h −θ∗
h) −κ

2 (θ∗
h −θk
h)⊤Wk,h(θ∗
h −θk
h) ,

where we denote Wk,h := P

s′∈Sk,h φk,h,s′φ⊤
k,h,s′. By taking expectation over yk
h, we have

¯ℓk,h(θk
h) ≤¯ℓk,h(θ∗
h) + ¯Gk,h(θk
h)⊤(θk
h −θ∗
h) −κ

2 (θ∗
h −θk
h)⊤Wk,h(θ∗
h −θk
h) .
(12)

On the other hand, for any θ ∈Rd, since we have
¯ℓk,h(θ) −¯ℓk,h(θ∗
h)

= −
X

s′∈Sk,h
Pθ∗
h(s′ | sk
h, ak
h) log Pθ(s′ | sk
h, ak
h) +
X

s′∈Sk,h
Pθ∗
h(s′ | sk
h, ak
h) log Pθ∗
h(s′ | sk
h, ak
h)

=
X

s′∈Sk,h
Pθ∗
h(s′ | sk
h, ak
h)
 
log Pθ∗
h(s′ | sk
h, ak
h) −log Pθ(s′ | sk
h, ak
h)


=
X

s′∈Sk,h
Pθ∗
h(s′ | sk
h, ak
h) log Pθ∗
h(s′ | sk
h, ak
h)

Pθ(s′ | sk
h, ak
h)

= DKL(Pθ∗
h ∥Pθ)

≥0 ,
where DKL(P ∥Q) is the Kullback-Leibler divergence of P from Q, from (12) we have
0 ≤¯ℓk,h(θk
h) −¯ℓk,h(θ∗
h)

≤¯Gk,h(θk
h)⊤(θk
h −θ∗
h) −κ

2 ∥θ∗
h −θk
h∥2
Wk,h

= Gk,h(θk
h)⊤(θk
h −θ∗
h) −κ

2 ∥θ∗
h −θk
h∥2
Wk,h +

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h) .
(13)

24


---Page Break---
To get an upper bound of Gk,h(θk
h)⊤(θk
h −θ∗
h), recall that the estimated transition core is given by

θk+1
h
= argmin
θ∈Bd(Lθ)

1
2∥θ −θk
h∥2
Ak+1,h + (θ −θk
h)⊤Gk,h(θk
h) .
(14)

Since the objective function in (14) is convex, by the first-order optimality condition for any θ ∈
Bd(Lθ), we have


Gk,h(θk
h) + Ak+1,h(θk+1
h
−θk
h)
⊤
(θ −θk+1
h
) ≥0,

which gives

θ⊤Ak+1,h(θk+1
h
−θk
h) ≥(θk+1
h
)⊤Ak+1,h(θk+1
h
−θk
h) −Gk,h(θk
h)⊤(θ −θk+1
h
) .
(15)

Then, we have

∥θk
h −θ∗
h∥2
Ak+1,h −∥θk+1
h
−θ∗
h∥2
Ak+1,h
= (θk
h)⊤Ak+1,hθk
h −(θk+1
h
)⊤Ak+1,hθk+1
h
+ 2(θ∗
h)⊤Ak+1,h(θk+1
h
−θk
h)

≥(θk
h)⊤Ak+1,hθk
h −(θk+1
h
)⊤Ak+1,hθk+1
h
+ 2(θk+1
h
)⊤Ak+1,h(θk+1
h
−θk
h)

−2Gk,h(θk
h)⊤(θ∗
h −θk+1
h
)
(by (15))

= (θk
h)⊤Ak+1,hθk
h + (θk+1
h
)⊤Ak+1,hθk+1
h
−2(θk+1
h
)⊤Ak+1,hθk
h −2Gk,h(θk
h)⊤(θ∗
h −θk+1
h
)

= ∥θk
h −θk+1
h
∥2
Ak+1,h −2Gk,h(θk
h)⊤(θ∗
h −θk+1
h
)

= ∥θk
h −θk+1
h
∥2
Ak+1,h + 2Gk,h(θk
h)⊤(θk+1
h
−θk
h) + 2Gk,h(θk
h)⊤(θk
h −θ∗
h)

≥−∥Gk,h(θk
h)∥2
A−1
k+1,h + 2Gk,h(θk
h)⊤(θk
h −θ∗
h) ,
(16)

where the last inequality follows by the fact that

∥θk
h −θk+1
h
∥2
Ak+1,h + 2Gk,h(θk
h)⊤(θk+1
h
−θk
h) ≥
min
θ∈Bd(Lθ)

n
∥θ∥2
Ak+1,h + 2Gk,h(θk
h)⊤θ
o

= −∥Gk,h(θk
h)∥2
A−1
k+1,h .

Therefore, from (16) we have

Gk,h(θk
h)⊤(θk
h −θ∗
h) ≤1

2∥Gk,h(θk
h)∥2
A−1
k+1,h + 1

2∥θk
h −θ∗
h∥2
Ak+1,h −1

2∥θk+1
h
−θ∗
h∥2
Ak+1,h . (17)

By substituting (17) into (13), we have

0 ≤1

2∥Gk,h(θk
h)∥2
A−1
k+1,h + 1

2∥θk
h −θ∗
h∥2
Ak+1,h −1

2∥θk+1
h
−θ∗
h∥2
Ak+1,h

−κ

2 ∥θ∗
h −θk
h∥Wk,h +

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h) .
(18)

25


---Page Break---
Note that since we have

∥Gk,h(θk
h)∥2
A−1
k+1,h

=
X

s′,es∈Sk,h


Pθk
h(s′ | sk
h, ak
h) −yk
h(s′)
 
Pθk
h(es | sk
h, ak
h) −yk
h(es)

φ⊤
k,h,s′A−1
k+1,hφk,h,es

= 1

2

X

s′,es∈Sk,h


Pθk
h(s′ | sk
h, ak
h) −yk
h(s′)
 
Pθk
h(es | sk
h, ak
h) −yk
h(es)


· (φ⊤
k,h,s′A−1
k+1,hφk,h,es + φ⊤
k,h,esA−1
k+1,hφk,h,s′)

≤1

2

X

s′,es∈Sk,h

 
Pθk
h(s′ | sk
h, ak
h) −yk
h(s′)
2
φ⊤
k,h,s′A−1
k+1,hφk,h,s′

+

Pθk
h(es | sk
h, ak
h) −yk
h(es)
2
φ⊤
k,h,esA−1
k+1,hφk,h,es



=
X

s′∈Sk,h


Pθk
h(s′ | sk
h, ak
h) −yk
h(s′)
2
φ⊤
k,h,s′A−1
k+1,hφk,h,s′

≤
X

s′∈Sk,h

Pθk
h(es | sk
h, ak
h) −yk
h(es)
 φ⊤
k,h,s′A−1
k+1,hφk,h,s′

≤
X

s′∈Sk,h


Pθk
h(es | sk
h, ak
h) + yk
h(s′)

φ⊤
k,h,s′A−1
k+1,hφk,h,s′

=
X

s′∈Sk,h
Pθk
h(es | sk
h, ak
h)φ⊤
k,h,s′A−1
k+1,hφk,h,s′ +
X

s′∈Sk,h
yk
h(s′)φ⊤
k,h,s′A−1
k+1,hφk,h,s′

≤2 max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k+1,h ,
(19)

where the first inequality utilizes the inequality x⊤Ay + y⊤Ax ≤x⊤Ax + y⊤Ay for any positive-
semidefinite matrix A, and the last inequality holds since 0 ≤Pθk
h(s′ | sk
h, ak
h) ≤1 and P

s′ Pθk
h(s′ |
sk
h, ak
h) = 1.

Combining the results of (18) and (19), we have

0 ≤max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k+1,h + 1

2∥θk
h −θ∗
h∥2
Ak+1,h −1

2∥θk+1
h
−θ∗
h∥2
Ak+1,h

−κ

2 ∥θ∗
h −θk
h∥Wk,h +

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h)

= max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k+1,h + 1

2∥θk
h −θ∗
h∥2
Ak,h + κ

4 ∥θk
h −θ∗
h∥2
Wk,h −1

2∥θk+1
h
−θ∗
h∥2
Ak+1,h

−κ

2 ∥θ∗
h −θk
h∥Wk,h +

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h)

= max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k+1,h + 1

2∥θk
h −θ∗
h∥2
Ak,h −κ

4 ∥θk
h −θ∗
h∥2
Wk,h −1

2∥θk+1
h
−θ∗
h∥2
Ak+1,h

+

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h) ,

where for the first equality we use Ak+1,h = Ak,h + κ

2 Wk,h. By rearranging the terms, we have

∥θk+1
h
−θ∗
h∥2
Ak+1,h ≤∥θk
h −θ∗
h∥2
Ak,h + 2 max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k+1,h −κ

2 ∥θk
h −θ∗
h∥2
Wk,h

+ 2

¯Gk,h(θk
h) −Gk,h(θk
h)
⊤
(θk
h −θ∗
h) .

26


---Page Break---
Then summing over k gives

∥θk+1
h
−θ∗
h∥2
Ak+1,h ≤∥θ1,h −θ∗
h∥2
A1,h + 2

k
X

i=1
max
s′∈Si,h ∥φi,h,s′∥2
A−1
i+1,h −κ

2

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h

+ 2

k
X

i=1

  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)

≤2λL2
θ + 2

k
X

i=1
max
s′∈Si,h ∥φi,h,s′∥2
A−1
i+1,h −κ

2

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h

+ 2

k
X

i=1

  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h) .

For the final step, note that
  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h) is a martingale difference sequence.
To bound this term, we invoke the following lemmas:

Lemma 2. For δ ∈(0, 1) and (k, h) ∈[K] × [H], with a probability at least 1 −δ we have

k
X

i=1

  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗)

≤κ

4

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h +
16LφLθ

3
+ 8

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+
√

2 .

Lemma 3 (Generalized elliptical potential). Let St := {xt,1, . . . , xt,K} ⊂Rd. For any 1 ≤t ≤T
and i ∈[K], suppose ∥xt,i∥2 ≤L. Let Vt := λId + Pt−1
τ=1
P

i∈Sτ xτ,ix⊤
τ,i for some λ > 0. If
λ ≥L2, then we have

T
X

t=1
max
i∈[K] ∥xt,i∥2
V−1
t
≤2d log

1 + TKL

dλ


.

By Lemma 2, with probability at least 1 −δ, we have

∥θk+1
h
−θ∗
h∥2
Ak+1,h

≤2λL2
θ + 2

k
X

i=1
max
s′∈Si,h ∥φi,h,s′∥2
A−1
i+1,h

+
32LφLθ

3
+ 16

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+ 2
√

2

≤2λL2
θ + 8

κd log

 

1 + kUL2
φ
dλ

!

+
32LφLθ

3
+ 16

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+ 2
√

2 ,

where the second inequality comes from Lemma 3. Note that the Gram matrix Ak,h in Algorithm 1
and the Gram matrix V in Lemma 3 are different by the factor of κ

2 , which results in additional 2

κ
factor for the bound of Pk
i=1 maxs′∈Si,h ∥φi,h,s′∥2
A−1
i+1,h.

In the following, we provide all the proofs of the lemmas used to prove Lemma 1.

27


---Page Break---
C.1.1
Proof of Lemma 2

Proof of Lemma 2. Note that
  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h) is a martingale difference se-
quence, i.e.,

E
h  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h) | Fi,h
i

=
  ¯Gi,h(θi
h) −E

Gi,h(θi
h) | Fi,h
⊤(θi
h −θ∗
h)

= 0 .

On the other hand, for any θ ∈Rd, since we have

∥Gi,h(θ)∥2 =



X

s′∈Si,h

 
Pθ(s′ | si
h, ai
h) −yi
h(s′)

φi,h,s′


2
≤
X

s′∈Si,h

Pθ(s′ | si
h, ai
h) −yi
h(s′)
 ∥φi,h,s′∥2

≤Lφ



X

s′∈Si,h
Pθ(s′ | si
h, ai
h) +
X

s′∈Si,h
yi
h(s′)





= 2Lφ ,

then, it follows by


  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)


≤

  ¯Gi,h(θi
h)
⊤(θi
h −θ∗
h)
 +

 
Gi,h(θi
h)
⊤(θi
h −θ∗
h)


≤∥¯Gi,h(θi
h)∥2∥θi
h −θ∗
h∥2 + ∥Gi,h(θi
h)∥2∥θi
h −θ∗
h∥2
≤4Lφ∥θi
h −θ∗
h∥2
≤8LφLθ ,
(20)

where the last inequality follows by ∥θi
h −θ∗
h∥2 ≤∥θi
h∥2 + ∥θ∗
h∥2 ≤2Lθ. Hence, if we denote
Mk,h := Pk
i=1
  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h), then Mk,h is a martingale. Note that we also

28


---Page Break---
have

Σk,h =

k
X

i=1
Eyi
h

 ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)
2

=

k
X

i=1
Eyi
h

" 
Gi,h(θi
h)
⊤(θi
h −θ∗
h)
2
#

−Eyi
h

"  ¯Gi,h(θi
h)
⊤(θi
h −θ∗
h)
2
#

≤

k
X

i=1
Eyi
h


Gi,h(θi
h)
⊤(θi
h −θ∗
h)
2

=

k
X

i=1
Eyi
h







X

s′∈Si,h


Pθi
h(s′ | si
h, ai
h) −yi
h(s′)

φ⊤
i,h,s′(θi
h −θ∗
h)





2



≤

k
X

i=1
Eyi
h







X

s′∈Si,h


Pθi
h(s′ | si
h, ai
h) −yi
h(s′)
2






X

s′∈Si,h

 
φ⊤
i,h,s′(θi
h −θ∗
h)
2








(21)

=

k
X

i=1
Eyi
h



X

s′∈Si,h


Pθi
h(s′ | si
h, ai
h) −yi
h(s′)
2






X

s′∈Si,h

 
φ⊤
i,h,s′(θi
h −θ∗
h)
2




≤2

k
X

i=1

X

s′∈Si,h

 
φ⊤
i,h,s′(θi
h −θ∗
h)
2
(22)

= 2

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h =: Bk,h ,

where (21) holds by the Cauchy–Schwarz inequality, (22) holds because

X

s′∈Si,h


Pθi
h(s′ | si
h, ai
h) −yi
h(s′)
2

=
X

s′∈Si,h

n
Pθi
h(s′ | si
h, ai
h)
o2
−2Pθi
h(s′ | si
h, ai
h)yi
h(s′) +

yi
h(s′)
	2

≤2 .

However, if we denote Bk,h := 2 Pk
i=1 ∥θi
h −θ∗
h∥2
Wi,h, since Bk,h is itself a random variable, to
apply Freedman’s inequality to Mk,h, we consider two cases depending on the values of Bk,h.

Case 1 : Bk,h ≤
4
kU

29


---Page Break---
Suppose that Bk,h = 2 Pk
i=1 ∥θi
h −θ∗
h∥2
Wi,h ≤
4
kU . Then we have

Mk,h =

k
X

i=1

  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)

=

k
X

i=1

X

s′∈Si,h

 
yi
h(s′) −E[yi
h(s′)]

φ⊤
i,h,s′(θi
h −θ∗
h)

=

k
X

i=1

X

s′∈Si,h

 
yi
h(s′) −Pθ∗
h(s′ | si
h, ai
h)

φ⊤
i,h,s′(θi
h −θ∗
h)

≤

k
X

i=1

X

s′∈Si,h
|φ⊤
i,h,s′(θi
h −θ∗
h)|

≤

v
u
u
tkU

k
X

i=1

X

s′∈Si,h


φ⊤
i,h,s′(θi
h −θ∗
h)
2

=

r

kU Bk,h

2
≤
√

2 .

Case 2 : Bk,h >
4
kU

Suppose that Bk,h = 2 Pk
i=1 ∥θi
h −θ∗
h∥2
Wi,h >
4
kU . Then, we have both a lower and upper bound
for Bk,h as follows:

4
kU < Bk,h ≤2

k
X

i=1

X

s′∈Si,h
∥φi,h,s′∥2
2∥θi
h −θ∗
h∥2
2 ≤8kUL2
φL2
θ .

Then by the peeling process from Bartlett et al. [12], for any ηk > 0, we have

P

Mk,h ≥2
p

ηkBk,h + 16ηkLφLθ

3



= P

Mk,h ≥2
p

ηkBk,h + 16ηkLφLθ

3
, 4

kU < Bk,h ≤8kUL2
φL2
θ



= P

Mk,h ≥2
p

ηkBk,h + 16ηkLφLθ

3
, 4

kU < Bk,h ≤8kUL2
φL2
θ, Σk,h ≤Bk,h



≤

m
X

j=1
P

Mk,h ≥2
p

ηkBk,h + 16ηkLφLθ

3
, 4 · 2j−1

kU
< Bk,h ≤4 · 2j

kU , Σk,h ≤Bk,h



≤

m
X

j=1
P

 

Mk,h ≥

r

ηk
8 · 2j

kU
+ 16ηkLφLθ

3
, Σk,h ≤4 · 2j

kU

!

|
{z
}
Ij

,
(23)

where m = 1 + ⌈2 log2 kULφLθ⌉. For Ij, note that from (20) we have


  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)
 ≤8LφLθ .

30


---Page Break---
By Freedman’s inequality (Lemma 29), we have

P

 

Mk,h ≥

r

ηk
8 · 2j

kU
+ 16ηkLφLθ

3
, Σk,h ≤4 · 2j

kU

!

≤exp








−
q

ηk 8·2j

kU + 16ηkLφLθ

3

2

8·2j

kU + 2

3 · 8LφLθ

q

ηk 8·2j

kU + 16ηkLφLθ

3










= exp








−ηk

q

8·2j

kU + 16√ηkLφLθ

3

2

8·2j

kU + 16LφLθ

3

q

ηk 8·2j

kU +
162ηkL2φL2
θ
32








≤exp








−ηk

q

8·2j

kU + 16√ηkLφLθ

3

2

8·2j

kU + 32LφLθ

3

q

ηk 8·2j

kU +
162ηkL2φL2
θ
32








= exp(−ηk) .
(24)

By substituting Eq. (24) into Eq. (23), we have

P

Mk,h ≥2
p

ηkBk,h + 16ηkLφLθ

3


≤m exp(−ηk) .

Then, combining with the result of Case 1 & 2, letting ηk = log
m
δ/k2 = log (1+⌈2 log2 kULφLθ⌉)k2

δ
and taking union bound over k, with probability at least 1 −δ, we have

Mk,h ≤2

v
u
u
t2ηk

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h + 16ηkLφLθ

3
+
√

2 .
(25)

By applying 2
√

ab ≤a + b to the first term on the right hand side, we have

2

v
u
u
t2ηk

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h ≤8ηk

κ + κ

4

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h .
(26)

Combining the results of Eq. (25) & Eq. (26), we have

Mk,h =

k
X

i=1

  ¯Gi,h(θi
h) −Gi,h(θi
h)
⊤(θi
h −θ∗
h)

≤κ

4

k
X

i=1
∥θi
h −θ∗
h∥2
Wi,h +
16LφLθ

3
+ 8

κ


log (1 + ⌈2 log2 kULφLθ⌉) k2

δ
+
√

2 .

31


---Page Break---
C.1.2
Proof of Lemma 3

Proof of Lemma 3. By definition of Vt, we have

det(Vt+1) = det

 

Vt +
X

i∈St
xt,ix⊤
t,i

!

= det(Vt) det

 

Id +
X

i∈St
V
−1

2
t
xt,ix⊤
t,iV
−1

2
t

!

= det(Vt)

 

1 +
X

i∈St
∥xt,i∥2
V−1
t

!

= det(λId)

tY

τ=1

 

1 +
X

i∈Sτ
∥xτ,i∥2
V−1
τ

!

≥det(λId)

tY

τ=1


1 + max
i∈Sτ ∥xτ,i∥2
V−1
t


.
(27)

Since λ ≥L2, we have

max
i∈Sτ ∥xτ,i∥2
V−1
τ
≤L2

λ ≤1 .

Since for any z ∈[0, 1], it follows that z ≤2 log(1 + z). Hence, we have

T
X

t=1
max
i∈St ∥xt,i∥2
V−1
t
≤2

T
X

t=1
log

1 + max
i∈St ∥xt,i∥2
V−1
t



= 2 log

T
Y

t=1


1 + max
i∈St ∥xt,i∥2
V−1
t



≤2 log det(VT +1)

det(λId)

≤2d log

1 + TKL2

dλ


,

where the second inequality comes from Eq. (27) and the last inequality follows by the determinant-
trace inequality (Lemma 28).

C.2
Bound on Prediction Error

In this section, we provide the bound on the prediction error induced by estimated transition core θk
h.

Lemma 4 (Bound on Prediction Error). For any δ ∈(0, 1), suppose that Lemma 1 holds. Then for
any (s, a) ∈S × A, we have

|∆k
h(s, a)| ≤Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h .

Proof of Lemma 4. Recall that

∆k
h(s, a) =
X

s′∈Ss,a


Pθk
h(s′ | s, a) −Pθ∗
h(s′ | s, a)

V k
h+1(s′)

=
X

s′∈Ss,a

exp(φ⊤
s,a,s′θk
h)V k
h+1(s′)
P

es∈Ss,a exp(φ⊤
s,a,es θk
h)
−
X

s′∈Ss,a

exp(φ⊤
s,a,s′θ∗
h)V k
h+1(s′)
P

es∈Ss,a exp(φ⊤
s,a,es θ∗
h) .

32


---Page Break---
Then by the mean value theorem, there exists ¯θ = ρθk
h + (1 −ρ)θ∗
h for some ρ ∈[0, 1] satisfying
that

∆k
h(s, a) =

P

s′∈Ss,a exp(φ⊤
s,a,s′¯θ)V k
h+1(s′)φ⊤
s,a,s′(θk
h −θ∗
h)
 P

es∈Ss,a exp(φ⊤
s,a,es¯θ)


P

es∈Ss,a exp(φ⊤
s,a,es ¯θ)
2

−

P

s′∈Ss,a exp(φ⊤
s,a,s′¯θ)V k
h+1(s′)
 P

es∈Ss,a exp(φ⊤
s,a,es ¯θ)φ⊤
s,a,es (θk
h −θ∗
h)


P

es∈Ss,a exp(φ⊤
s,a,es ¯θ)
2

=
X

s′∈Ss,a
P¯θ(s′ | s, a)V k
h+1(s′)φ⊤
s,a,s′(θk
h −θ∗
h)

−

 P

s′∈Ss,a exp(φ⊤
s,a,s′¯θ)V k
h+1(s′)
P

es∈Sk,h exp(φ⊤
s,a,es ¯θ)

! X

s′∈Ss,a
P¯θ(s′ | s, a)φ⊤
s,a,s′(θk
h −θ∗
h)

=
X

s′∈Ss,a

 

V k
h+1(s′) −

P
s′∈Ss,a exp(φ⊤
s,a,s′¯θ)V k
h+1(s′)
P

es∈Ss,a exp(φ⊤
s,a,es ¯θ)

!

P¯θ(s′ | s, a)φ⊤
s,a,s′(θk
h −θ∗
h) .

Since V k
h (s′) ≤H for all s′ ∈S, k ∈[K], and h ∈[H], we have

∆k
h(s, a) ≤H
X

s′∈Ss,a
P¯θ(s′ | s, a)φ⊤
s,a,s′(θk
h −θ∗
h)

≤H max
s′∈Ss,a |φ⊤
s,a,s′(θk
h −θ∗
h)|

≤H max
s′∈Ss,a ∥φs,a,s′∥A−1
k,h∥θk
h −θ∗
h∥Ak,h

≤Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h ,

where the second inequality comes from the fact that P¯θ(s′ | s, a) ≤1 is a multinomial
probability, the third inequality holds due to the Cauchy-Schwarz inequality, and the last in-
equality follows from Lemma 1 and the definition of ˆφk,h, i.e., ˆφk,h(s, a) := φ(s, a, ˆs) for
ˆs = argmaxs′∈Ss,a ∥φ(s, a, s′)∥A−1
k,h.

C.3
Good Events with High Probability

Lemma 5 (Good event probability). For any K ∈N and δ ∈(0, 1), the good event G(K, δ′) holds
with probability at least 1 −δ where δ′ = δ/(2KH).

Proof of Lemma 5. For any δ′ ∈(0, 1), we have

G(K, δ′) =
\

k≤K

\

h≤H
Gk,h(δ′) =
\

k≤K

\

h≤H

n
G∆
k,h(δ′) ∩Gξ
k,h(δ′)
o
.

On the other hand, for any (k, h) ∈[K] × [H], by Lemma 30, Gξ
k,h(δ′) holds with probability at least
1 −δ′. Then, for δ′ = δ/(2KH) by taking union bound, we have the desired result as follows:

P(G(K, δ′)) ≥(1 −δ′)2KH ≥1 −2KHδ′ = 1 −δ .

C.4
Stochastic Optimism

Lemma 6 (Stochastic optimism). For any δ with 0 < δ < Φ(−1)/2, let σk = Hαk(δ) = e
O(H
√

d).
If we take multiple sample size M = ⌈1 −
log H
log Φ(1)⌉, then for any k ∈[K], we have

P
 
(V k
1 −V ∗
1 )(sk
1) ≥0 | sk
1, Fk

≥Φ(−1)/2 .

33


---Page Break---
Proof of lemma 6. Before presenting the proof, we introduce the following lemmas.

Lemma 7. For any k ∈[K], it holds

V k
1 (sk
1) −V ∗
1 (sk
1) ≥Eπ∗

" H
X

h=1
−ιk
h(xh, ah) | x1 = sk
1

#

,

where ιk
h(s, a) := r(s, a) + PhV k
h+1(s, a) −Qk
h(s, a).

Lemma 8. Let δ ∈(0, 1) be given. For any (k, h) ∈[K] × [H], let σk = Hαk(δ). If we define the
event G∆
k,h(δ) as

G∆
k,h(δ) :=
n
∆k
h(s, a) ≤Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h

o
,

then conditioned on G∆
k,h(δ), for any (s, a) ∈S × A, we have

P
 
−ιk
h(s, a) ≥0 | G∆
k,h(δ)

≥1 −Φ(1)M .

Lemma 9. Let δ ∈(0, 1) be given. For any (h, k) ∈[H] × [K], let σk = Hαk(δ). If we take
multiple sample size M = ⌈1 −
log H
log Φ(1)⌉, then conditioned on the event G∆
k (δ) := T

h∈[H] G∆
k,h(δ),
we have
P
 
−ιk
h(sh, ah) ≥0, ∀h ∈[H] | G∆
k (δ)

≥Φ(−1) .

Now, we define the event of the estimated value function being optimistic at the start of the k-th
episode as
Xk :=

(V k
1 −V ∗
1 )(sk
1) ≥0
	
.
Then for the event Gk(δ) =: Gk, we have

P(Xk) = 1 −P(X c
k)
= 1 −P(X c
k ∩Gk) −P(X c
k ∩Gc
k)
≥1 −P(X c
k ∩Gk) −P(Gc
k)
≥1 −P(X c
k ∩Gk) −δ

where the last inequality comes from lemma 5.

On the other hand, by Lemma 7, we have

V k
1 (sk
1) −V ∗
1 (sk
1) ≥Eπ∗

" H
X

h=1
−ιk
h(xh, ah) | x1 = sk
1

#

=

H
X

h=1
Eπ∗
−ιk
h(xh, ah) | x1 = sk
1

.

If we define an event

Yk =

( H
X

h=1
Eπ∗
−ιk
h(xh, ah) | x1 = sk
1

≥0

)

,

then, by Lemma 9, we have

P(Yk | Gk) ≥Φ(−1) ⇐⇒P(Yc
k | Gk) ≤1 −Φ(−1)
=⇒P(Yc
k ∩Gk) ≤(1 −Φ(−1)) P(Gk) ≤1 −Φ(−1)

Note that since X c
k ∩Gk ⊂Yc
k ∩Gk, we can conclude that

P(Xk) ≥1 −P(X c
k ∩Gk) −δ
≥1 −P(Yc
k ∩Gk) −δ
≥1 −(1 −Φ(−1)) −δ
= Φ(−1) −δ
≥Φ(−1)/2

where the last inequality comes from the choice of δ.

In the following, we provide all the proofs of the lemmas used to prove Lemma 6.

34


---Page Break---
C.4.1
Proof of Lemma 7

Proof of lemma 7. In this proof, we use xk
h as the states sampled under the π∗to distinguish with sk
h.
Since we have,

V k
1 (sk
1) −V ∗
1 (sk
1)

≥Qk
1(sk
1, π∗(sk
1)) −Q∗
1(sk
1, π∗(sk
1))

= r(sk
1, π∗(sk
1)) + P1V k
2 (sk
1, π∗(sk
1)) −ιk
1(sk
1, π∗(sk
1)) −
 
r(sk
1, π∗(sk
1)) + P1V ∗
2 (sk
1, π∗(sk
1))


= P1(V k
2 −V ∗
2 )(sk
1, π∗(sk
1)) −ιk
1(sk
1, π∗(sk
1))

= Ex|sk
1,π∗(sk
1)

(V k
2 −V ∗
2 )(x)

−ιk
1(sk
1, π∗(sk
1))

≥Exk
2|sk
1,π∗(sk
1)

(Qk
2 −Q∗
2)(xk
2, π∗(xk
2))

−ιk
1(sk
1, π∗(sk
1))

= Exk
2∼sk
1,π∗(sk
1)
h
Ex|xk
2,π∗(xk
2)

(V k
3 −V ∗
3 )(x)

−ιk
2(xk
2, π∗(xk
2))
i
−ιk
1(sk
1, π∗(sk
1))

= Exk
2∼sk
1,π∗(sk
1)
h
Ex|xk
2,π∗(xk
2)

(V k
3 −V ∗
3 )(x)
i

|
{z
}

Exk
3 ∼π∗|sk
1[(V k
3 −V ∗
3 )(xk
3)]

−Exk
2∼sk
1,π∗(sk
1)

ιk
2(xk
2, π∗(xk
2))

−ιk
1(sk
1, π∗(sk
1))

then by applying this argument recursively, we finally have

V k
1 (sk
1) −V ∗
1 (sk
1) ≥Eπ∗

" H
X

h=1
−ιk
h(xh, ah) | x1 = sk
1

#

.

C.4.2
Proof of Lemma 8

Proof of Lemma 8. Since we have

−ιk
h(s, a) = Qk
h(s, a) −
 
r(s, a) + PhV k
h+1(s, a)


= min




r(s, a) +
X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h , H






−
 
r(s, a) + PhV k
h+1(s, a)


≥min






X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h −PhV k
h+1(s, a), 0




,

it is enough to show that
X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h −PhV k
h+1(s, a) ≥0

at least with constant probability.

On the other hand, under the event Gk,h(δ), by Lemma 4 we have
X

s′∈Ss,a
Pθk
h(s′ | s, a)V k
h+1(s′) + max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h −PhV k
h+1(s, a)

≥max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h −Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h .

Now, for ∀m ∈[M], since ξ(m)
k,h ∼N(0d, σ2
kA−1
k,h), we have

ˆφk,h(s, a)⊤ξ(m)
k,h ∼N(0, σ2
k∥ˆφk,h(s, a)∥2
A−1
k,h) ,

35


---Page Break---
which means,

P

ˆφk,h(s, a)⊤ξ(m)
k,h ≥Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h


≥Φ(−1) ,

by setting σk = Hαk(δ). Then, finally we have the desired results as follows:

P
 
−ιk
h(s, a) ≥0 | G∆
k,h(δ)


≥P

max
m∈[M] ˆφk,h(s, a)⊤ξ(m)
k,h ≥Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h | G∆
k,h(δ)


= 1 −P

ˆφk,h(s, a)⊤ξ(m)
k,h < Hαk(δ)∥ˆφk,h(s, a)∥A−1
k,h, ∀m ∈[M] | G∆
k,h(δ)


≥1 −(1 −Φ(−1))M

= 1 −Φ(1)M .

C.4.3
Proof of Lemma 9

Proof of Lemma 9. For each h ∈[H] and k ∈[K], define an event Ek
h := {−ιk
h(sh, ah) ≥0} Then
it holds

P
 
−ιk
h(sh, ah) ≥0, ∀h ∈[H] | G∆
k (δ)

= P

 H
\

h=1
Ek
h | G∆
k (δ)

!

= 1 −P

 H
[

h=1
(Ek
h)c | G∆
k (δ)

!

≥1 −

H
X

h=1
P
 
(Ek
h)c | G∆
k,h(δ)


≥1 −HΦ(1)M

≥Φ(−1)

where the first inequality uses the union bound, the second inequality comes from the Lemma 8 and
the last inequality holds due to the choice of M = ⌈1 −
log H
log Φ(1)⌉.

C.5
Bound on Estimation Part

We decompose the regret into the estimation part and the pessimism part as follows:

K
X

k=1
(V ∗
1 −V πk
1
)(sk
1) =

K
X

k=1


V ∗
1 −V k
1
|
{z
}
Pessimism

+ V k
1 −V πk
1
|
{z
}
Estimation


(sk
1) ,

and we bound these two parts in the following sections, respectively.
Lemma 10 (Bound on estimation part). For any δ ∈(0, 1), if λ ≥L2
φ, then with probability at least
1 −δ/2, we have
K
X

k=1
(V k
1 −V πk
1
)(sk
1) = e
O

κ−1d
3
2 H
3
2 √

T

.

Proof of lemma 10. For any given k ∈[K],

(V k
1 −V πk
1
)(sk
1) = (Qk
1 −Qπk
1 )(sk
1, ak
1) + ιk
1(sk
1, ak
1) −ιk
1(sk
1, ak
1)

= (Qk
1 −Qπk
1 )(sk
1, ak
1) + P1(V k
2 −V πk
2
)(sk
1, ak
1)
(28)

+ (Qπk
1
−Qk
1)(sk
1, ak
1) −ιk
1(sk
1, ak
1)

= P1(V k
2 −V πk
2
)(sk
1, ak
1) −(V k
2 −V πk
2
)(sk
2)
|
{z
}
˙ζk
1

+(V k
2 −V πk
2
)(sk
2) −ιk
1(sk
1, ak
1)

36


---Page Break---
where the second equality holds due to the variant of ιk
h(sk
h, ak
h) as follows:

ιk
h(sk
h, ak
h) = r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h) −Qk
h(sk
h, ak
h) + Qπk
h (sk
h, ak
h) −Qπk
h (sk
h, ak
h)

= r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h) −Qk
h(sk
h, ak
h)

+ Qπk
h (sk
h, ak
h) −

r(sk
h, ak
h) + PhV πk
h+1(sk
h, ak
h)


= Ph(V k
h+1 −V πk
h+1)(sk
h, ak
h) + (Qπk
h −Qk
h)(sk
h, ak
h) .

Then, by applying this argument recursively for whole horizon, we have

(V k
1 −V πk
1
)(sk
1) =

H
X

h=1
−ιk
h(sk
h, ak
h) +

H
X

h=1
˙ζk
h ,
(29)

where ˙ζk
h := Ph(V k
h+1 −V πk
h+1)(sk
h, ak
h) −(V k
h+1 −V πk
h+1)(sk
h+1).

Let δ′ = δ/(8KH). By Lemma 5, the good event G(K, δ′) holds with probability at least 1 −δ/4.
Then under the event G(K, δ′), for any h ∈[H] we have

−ιk
h(sk
h, ak
h)

= Qk
h(sk
h, ak
h) −
 
r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h)


= min




r(sk
h, ak
h) +
X

s′∈Sk,h
Pθk
h(s′ | sk
h, ak
h)V k
h+1(s′) + max
m∈[M] ˆφk,h(sk
h, ak
h)⊤ξ(m)
k,h , H






−
 
r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h)


≤
X

s′∈Sk,h
Pθk
h(s′ | sk
h, ak
h)V k
h+1(s′) + max
m∈[M] ˆφk,h(sk
h, ak
h)⊤ξ(m)
k,h −PhV k
h+1(sk
h, ak
h)

≤



X

s′∈Sk,h
Pθk
h(s′ | sk
h, ak
h)V k
h+1(s′) −PhV k
h+1(sk
h, ak
h)


+ max
m∈[M]

ˆφk,h(sk
h, ak
h)⊤ξ(m)
k,h


≤|∆k
h(sk
h, ak
h)| + max
m∈[M] ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h∥ξ(m)
k,h ∥Ak,h
(30)

≤(Hαk(δ′) + γk(δ′)) ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h ,
(31)

where (30) comes from the Cauchy-Schwarz inequality and (31) holds due the the Lemma 4 & 30.
Then, with probability at least 1 −δ/4, we have

H
X

h=1
−ιk
h(sk
h, ak
h) ≤

H
X

h=1
(Hαk(δ′) + γk(δ′)) ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h .
(32)

On the other hand, for ˙ζk
h, we have | ˙ζk
h| ≤2H and E[ ˙ζk
h | Fk,h] = 0, which means { ˙ζk
h | Fk,h}k,h
is a martingale difference sequence for any k ∈[K] and h ∈[H]. Hence, by applying the Azuma-
Hoeffding inequality with probability at least 1 −δ/4, we have

K
X

k=1

H
X

h=1
˙ζk
h ≤2H
p

2KH log(4/δ) .
(33)

37


---Page Break---
Combining the results of (32) and (33), with probability at least 1 −δ/2, we have

(V k
1 −V πk
1
)(sk
1)

≤2H
p

2T log(4/δ) +

K
X

k=1

H
X

h=1
(Hαk(δ′) + γk(δ′)) ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h

≤2H
p

2T log(4/δ) + (HαK(δ′) + γK(δ′))

K
X

k=1

H
X

h=1
∥ˆφk,h(sk
h, ak
h)∥A−1
k,h
(34)

≤2H
p

2T log(4/δ) + (HαK(δ′) + γK(δ′))

H
X

h=1

v
u
u
tK

K
X

k=1
∥ˆφk,h(sk
h, ak
h)∥2
A−1
k,h
(35)

≤2H
p

2T log(4/δ) + (HαK(δ′) + γK(δ′))

H
X

h=1

s

4κ−1Kd log

1 + KUL2φ

dλ


(36)

= 2H
p

2T log(4/δ) + (HαK(δ′) + γK(δ′))

s

4κ−1THd log

1 + KUL2φ

dλ


,

= e
O

κ−1d
3
2 H
3
2 √

T + H
√

T

,

where (34) follows from the fact that both αk(δ) and γk(δ) are increasing in k, (35) comes from
Cauchy-Schwarz inequality and (36) holds by the generalized elliptical potential lemma (Lemma 3).

C.6
Bound on Pessimism Part

Lemma 11 (Bound on pessimism). For any δ with 0 < δ < Φ(−1)/2, let σk = Hαk(δ). If λ ≥L2
φ
and we take multiple sample size M = ⌈1 −
log H
log Φ(1)⌉, then with probability at least 1 −δ/2, we have

K
X

k=1
(V ∗
1 −V k
1 )(sk
1) = e
O

κ−1d
3
2 H
3
2 √

T

.

Proof of lemma 11. Similar to the techniques used in [73], we show that the difference between the
optimal value function V ∗
1 and the estimated value function V k
1 can be controlled by constructing an
upper bound on V ∗
1 and a lower bound on V k
1 . In this proof, we consider three kinds of pseudo-noises,
ξ, ¯ξ and ξ that we define later in the proof. Also, for δ′ = δ/10, we denote G(K, δ′), ¯G(K, δ′) and
G(K, δ′) as the good events induced by ξ, ¯ξ and ξ respectively. From now on, we denote G(K, δ′)
by the event G(K, δ′) ∩¯G(K, δ′) ∩G(K, δ′). Then, by Lemma 5, the event G(K, δ′) holds with high
probability at least 1 −3δ/10.

First, we construct the lower bound of V k
1 . For any given k ∈[K], let eξ := {eξ
(m)
k,h }m∈[M] ⊂Rd be
a set of vectors for h ∈[H] and V k
h (· ; eξ) be the value function obtained by the Algorithm 1 with

non-random eξ
(m)
k,h in place of ξ(m)
k,h . Then consider the following minimization problem:

min
{eξ
(m)
k,h }h∈[H],m∈[M]

V k
1 (sk
1; eξ)

s.t.
max
m∈[M] ∥eξ
(m)
k,h ∥Ak,h ≤γk(δ),
∀h ∈[H]

And we denote ξ := {ξ(m)
k,h }h∈[H],m∈[M] by a minimizer and V k
1(sk
1) by the minimum of the

above minimization problem, i.e., V k
h(·) := V k
h (· ; ξ). Then, under the event G(K, δ′), since

{ξ(m)
k,h }h∈[H],m∈[M] is also a feasible solution of the above optimization problem, and since V k
h =
V k
h ( ; ξ), thus we have
V k
1(sk
1) ≤V k
1 (sk
1) .
(37)

38


---Page Break---
Second, to find an upper bound for V ∗, considering i.i.d copies {¯ξ
(m)
k,h }h∈[H],m∈[M] of

{ξ(m)
k,h }h∈[H],m∈[M] and run Algorithm 1 to get a corresponding value function ¯V k
h and ¯Qk
h for
all h ∈[H]. Define the event that ¯V k
1 (sk
1) is optimistic in the k-th episode as

¯
Xk = {( ¯V k
1 −V ∗
1 )(sk
1) ≥0} .

Then by Lemma 6, for given δ, we have

P( ¯
Xk | sk
1, Fk) ≥Φ(−1)/2 .

Then by the definition of optimism, under the event G(K, δ′), we have

(V ∗
1 −V k
1 )(sk
1) ≤E¯ξ| ¯
Xk

( ¯V k
1 −V k
1 )(sk
1)


≤E¯ξ| ¯
Xk
h
( ¯V k
1 −V k
1)(sk
1)
i
,
(38)

where the expectations are over the ¯ξ’s conditioned on the event ¯
Xk and the second inequality comes
from (37). On the other hand, under the event ¯G(K, δ′) by the law of the total expectation, we have

E¯ξ
h
( ¯V k
1 −V k
1)(sk
1)
i
= E¯ξ| ¯
Xk
h
( ¯V k
1 −V k
1)(sk
1)
i
P( ¯
Xk) + E¯ξ| ¯
X c
k

h
( ¯V k
1 −V k
1)(sk
1)
i
P( ¯
X c
k)

≥E¯ξ| ¯
Xk
h
( ¯V k
1 −V k
1)(sk
1)
i
P( ¯
Xk) ,
(39)

where (39) comes from the fact that {¯ξ
(m)
k,h }h∈[H],m∈[M] is also a feasible solution of the above
optimization problem under the event ¯G(K, δ′), i.e., ¯V k
1 (sk
1) ≥V k
1(sk
1). Then, by combining the
results of (39) and (38), under the event G(K, δ′), we have

(V ∗
1 −V k
1 )(sk
1) ≤E¯ξ| ¯
Xk
h
( ¯V k
1 −V k
1)(sk
1)
i

≤E¯ξ
h
( ¯V k
1 −V k
1)(sk
1)
i
/P( ¯
Xk)

≤
2
Φ(−1)E¯ξ
h
( ¯V k
1 −V k
1 + V k
1 −V k
1)(sk
1)
i

=
2
Φ(−1)


(V k
1 −V k
1)(sk
1)

+ ¨ζk ,
(40)

where we denote
¨ζk :=
2
Φ(−1)
 
E¯ξ
 ¯V k
1 (sk
1)

−V k
1 (sk
1)

.

Note that since ¯ξ is the i.i.d copy of ξ, therefore ¯Vk,1 and Vk,1 are independent, which means
{¨ζk | Fk−1}K
k=1 is a martingale difference sequence with |¨ζk| ≤
2H
Φ(−1). Therefore by applying
Azuma-Hoeffiding inequality under the event G(K, δ′), with probability at least 1 −δ′, we have

K
X

k=1
¨ζk ≤
2H
Φ(−1)

p

2K log(1/δ′) .
(41)

On the other hand, by dividing the first term in (40) into two terms we have

(V k
1 −V k
1)(sk
1) = (V k
1 −V πk
1
)(sk
1)
|
{z
}
I1

+ (V πk
1
−V k
1)(sk
1)
|
{z
}
I2

.

For I1, note that since it is related to the estimation error, under the event G(K, δ′) we can bound the
sum of I1 for the total episode number using Lemma 10 as follows:

K
X

k=1
(V k
1 −V πk
1
)(sk
1) ≤(HαK(δ′) + γK(δ′))

s

4κ−1THd log

1 + KUL2φ

dλ



+ 2H
p

2T log(1/δ′) .
(42)

39


---Page Break---
For I2, since we have

I2 = Qπk
1 (sk
1, ak
1) −V k
1(sk
1)

≤Qπk
1 (sk
1, ak
1) −Qk
1(sk
1, ak
1)
(43)

= Qπk
1 (sk
1, ak
1) −Qk
1(sk
1, ak
1) −ιk
1(sk
1, ak
1) + ιk
1(sk
1, ak
1)

= P1(V πk
2
−V k
2)(sk
1, ak
1) + ιk
1(sk
1, ak
1)
(44)

= P1(V πk
2
−V k
2)(sk
1, ak
1) −(V πk
2
−V k
2)(sk
2)
|
{z
}
...
ζ k
1

+(V πk
2
−V k
2)(sk
2) + ιk
1(sk
1, ak
1)

where (43) comes from ak
1 = argmaxa Qk
1(sk
1, a) and (44) holds by the following definition of
ιk
h(sk
h, ak
h):

ιk
h(sk
h, ak
h) := r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h) −Qk
h(sk
h, ak
h)

= r(sk
h, ak
h) + PhV k
h+1(sk
h, ak
h) −Qk
h(sk
h, ak
h) + Qπk
h (sk
h, ak
h) −Qπk
h (sk
h, ak
h)

= Ph(V k
h+1 −V πk
h+1)(sk
h, ak
h) + (Qπk
h −Qk
h)(sk
h, ak
h) .

Then by applying the same argument recursively for the whole horizon, we have

I2 ≤

H
X

h=1
ιk
h(sk
h, ak
h) +

H
X

h=1

...
ζ
k
h ,

where we denote
...
ζ
k
h := Ph(V πk
h+1 −V k
h+1)(sk
h, ak
h) −(V πk
h+1 −V k
h+1)(sk
h+1) .

Note that
n ...
ζ
k
h | Fk,h
o

k,h is a martingale difference sequence with |
...
ζ
k
h| ≤2H. Then, under the

event G(K, δ′) by applying the Azuma-Hoeffding inequality with probability at least 1 −δ′, we have

K
X

k=1

H
X

h=1

...
ζ
k
h ≤2H
p

2T log(1/δ′) .
(45)

To bound PH
h=1 ιk
h(sk
h, ak
h), we divide the whole horizon index set into two groups as follows:

H+

=




j ∈[H] : r(sk
j , ak
j ) +
X

s′∈Sk,j
Pθk
h(s′ | sk
j , ak
j )V k
j+1(s′) + max
m∈[M] ˆφk,j(sk
j , ak
j )⊤ξ(m)
k,j > H






H−= [H]\H+ .

Then, for j ∈H+ since Qk
j (sk
j , ak
j ) = H −j + 1, V k
j+1 ≤H −j and r(sk
j , ak
j ) ≤1, we have

ιk
j (sk
j , ak
j ) = r(sk
j , ak
j ) + PjV k
j+1(sk
j , ak
j ) −(H −j + 1) ≤0 .
(46)

On the other hand, for j ∈H−, under the event G(K, δ′) we have

ιk
j (sk
j , ak
j ) = PjV k
j+1(sk
j , ak
j ) −
X

s′∈Sk,j
Pθk
h(s′ | sk
j , ak
j )V k
j+1(s′) −max
m∈[M] ˆφk,j(sk
j , ak
j )⊤ξ(m)
k,j

≤


PjV k
j+1(sk
j , ak
j ) −
X

s′∈Sk,j
Pθk
h(s′ | sk
j , ak
j )V k
j+1(s′)


+
 max
m∈[M] ˆφk,j(sk
j , ak
j )⊤ξ(m)
k,j



≤Hαk(δ′)∥ˆφk,j(sk
j , ak
j )∥A−1
k,j + max
m∈[M]

ˆφk,j(sk
j , ak
j )⊤ξ(m)
k,j


(47)

≤Hαk(δ′)∥ˆφk,j(sk
j , ak
j )∥A−1
k,j + max
m∈[M] ∥ˆφk,j(sk
j , ak
j )∥A−1
k,j∥ξ(m)
k,j ∥Ak,j

≤(Hαk(δ′) + γk(δ′)) ∥ˆφk,j(sk
j , ak
j )∥A−1
k,j ,
(48)

40


---Page Break---
where (47) holds by Lemma 4.

By combining the result of (46) and (48), we have

I2 ≤
X

j∈H−
(Hαk(δ′) + γk(δ′)) ∥ˆφk,j(sk
j , ak
j )∥A−1
k,j +

H
X

h=1

...
ζ
k
h

≤

H
X

h=1
(Hαk(δ′) + γk(δ′)) ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h +

H
X

h=1

...
ζ
k
h .

Then summing I2 over the total number of episodes, under the event G(K, δ′), we have

K
X

k=1
(V πk
1
−V k
1)(sk
1) ≤

K
X

k=1

H
X

h=1
(Hαk(δ′) + γk(δ′)) ∥ˆφk,h(sk
h, ak
h)∥A−1
k,h +

K
X

k=1

H
X

h=1

...
ζ
k
h

≤(HαK(δ′) + γK(δ′))

K
X

k=1

H
X

h=1
∥ˆφk,h(sk
h, ak
h)∥A−1
k,h +

K
X

k=1

H
X

h=1

...
ζ
k
h

≤(HαK(δ′) + γK(δ′))

s

4κ−1THd log

1 + KUL2φ

dλ


(49)

+ 2H
p

2T log(1/δ′) ,
(50)

where the last inequality holds due to the Lemma 3 and (45).

Finally, by summing (40) over k and plugging the results of (42), (50) and (41) then, we have

K
X

k=1
(V ∗
1 −V k
1 )(sk
1)

≤
4
Φ(−1)

"

(HαK(δ′) + γK(δ′))

s

4κ−1THd log

1 + KUL2φ

dλ


+ 2H
p

2T log(1/δ′)

#

+
2H
Φ(−1)

p

2K log(1/δ′)

≤e
O

κ−1d3/2H3/2√

T + H
√

T + H
√

K

.

To conclude the proof, by setting δ′ = δ/10 and we take a union bound over the two applications of
Azuma-Hoeffding (¨ζk,
...
ζ
k
h) and the event G(K, δ′), we get the desired result with probability at least
1 −δ/2.

C.7
Regret Bound of RRL-MNL

Proof of Theorem 1. We can decompose the regret with estimation part and pessimism part as follows:

Regretπ(K) =

K
X

k=1
(V ∗
1 −V πk
1
)(sk
1)

=

K
X

k=1
(V ∗
1 −V k
1 )(sk
1) +

K
X

k=1
(V k
1 −V πk
1
)(sk
1) .

Since both Lemma 10 and Lemma 11 holds with probability at least 1 −δ/2 respectively, by taking
the union bound the following holds with probability at least 1 −δ:

Regretπ(K) = e
O

κ−1d
3
2 H
3
2 √

T + H
√

T + H
√

K

+ e
O

κ−1d
3
2 H
3
2 √

T + H
√

T


= e
O

κ−1d
3
2 H
3
2 √

T

.

41


---Page Break---
D
Detailed Regret Analysis for ORRL-MNL (Theorem 2)

D.1
Concentration of Estimated Transition Core eθk
h

In this section, we provide the detailed proof of Lemma 12, which demonstrates the concentration
result for eθk
h independently of κ and U. Note that we adapt the proof provided by Zhang and
Sugiyama [76] in the MNL contextual bandit setting to MNL-MDPs and improve the result, making
it independent of U. We provide the lemmas for the concentration of the online transition core for
completeness, noting that there are slight differences compared to their work, which stem from the
different problem setting.
Lemma 12 (Concentration of online estimated transition core). Let η = O(log U) and λ =
O(d log U). Then, for any δ ∈(0, 1] and for any h ∈[H], we have

P

∀k ≥1,
eθk
h −θ∗
h

Bk,h
≤βk(δ)

≥1 −δ ,

where βk(δ) = O(
√

d log U log(kH)).

Proof of Lemma 12. Recall that the transition core updated by the online mirror descent is represented
by
eθk+1
h
= argmin
θ∈B(Lθ)
eℓk,h(θ) + 1

2η

θ −eθk
h

2

Bk,h
,

where eℓk,h(θ) = ℓk,h(eθk
h) + (θ −eθk
h)⊤∇ℓk,h(eθk
h) + 1

2
θ −eθk
h

∇2ℓk,h(eθk
h) . We introduce the

following lemma providing that the estimation error of the online estimator eθk
h can be bounded by
the regret.

Lemma 13 (Lemma 12 in [76]). Let α = log U + 2(1 + LθLφ) and λ > 0. If we set the step size
η = α/2, then we have

eθk
h −θ∗
h

2

Bk,h
≤α

k
X

i=1


ℓi,h(θ∗
h) −ℓi,h(eθi+1
h
)

+ λL2
θ

+ 3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 −

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h
.

(51)

Now, we bound the first term of (51). To simplify the presentation, for all (k, h) ∈[K] × [H], we
define the softmax function σk,h : R|Sk,h| →[0, 1]|Sk,h| as follows:

[σk,h(z)]s′ =
exp([z]s′)
P

s′′∈Sk,h exp([z]s′′),

where [·]s′ denote the element corresponding to s′ ∈S of the input vector. We also define the
pseudo-inverse of the softmax function σk,h via [σ+
k,h(p)]s′ = log([p]s′) which has the property

that for all p ∈∆|Sk,h|, we have σk,h(σ+
k,h(p)) = p and P
s∈Sk,h exp

[σ+
k,h(p)]s

= 1.

We denote Φk,h = [φk,h,s′]s′∈Sk,h ∈Rd×|Sk,h| for simplicity.
Then, the transition model
can also be written as Pθ(s′
|
sk
h, ak
h)
=
[σk,h(Φ⊤
k,hθ∗
h)]s′.
We further define ezi,h
=

σ+
i,h

Eθ∼N(eθi
h,cB−1
i,h)[σi,h(Φ⊤
i,hθ)]

for our analysis. Then, we have

k
X

i=1


ℓi,h(θ∗
h) −ℓi,h(eθi+1
h
)

=

k
X

i=1


ℓi,h(θ∗
h) −ℓ(ezi,h, yi
h)

+

k
X

i=1


ℓ(ezi,h, yi
h) −ℓi,h(eθi+1
h
)

.

(52)

We can bound the first term of (52) by the following lemma.

42


---Page Break---
Lemma 14. Let δ ∈(0, 1]. Then, for all (k, h) ∈[K] × [H], with probability at least 1 −δ, we have

k
X

i=1

 
ℓi,h(θ∗
h) −ℓ(ezi,h, yi
h)

≤ΓA
k (δ),

where ΓA
k (δ) = 5

4(3 log(Uk) + LφLθ)λ + 4(3 log(Uk) + LφLθ) log

H√1+2k

δ

+ 2.

Furthermore, we can bound the second term of (52) by the following lemma.

Lemma 15. Let λ ≥72L2
φcd. Then, for any c > 0 and all (k, h) ∈[K] × [H], we have

k
X

i=1


ℓ(ezi,h, yi
h) −ℓi,h(eθi+1
h
)

≤1

2c

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h
+ ΓB
k (δ).

where ΓB
k (δ) =
√

6cd log

1 +
2kL2
φ
dλ

.

Combining Lemma 13, Lemma 14, and Lemma 15, and by setting η = α/2, c = 2α/3 and
λ ≥max{12
√

2L3
φα, 48L2
φdα}, we derive that

eθk+1
h
−θ∗
h

2

Bk,h

≤αΓA
k (δ) + αΓB
k (δ) + λL2
θ + 3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 +
 α

2c −1

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h

≤αΓA
k (δ) + αΓB
k (δ) + λL2
θ

≤C log U

λ log(Uk) + log(Uk) log
H
√

1 + 2k

δ


+ d log

1 + k

dλ


+ λL2
θ

=: βk(δ)2
(53)

where C > 0 is an absolute constant. In the above, we choose λ = O(d log U), α = O(log U). The
second inequality of (53) is derived from the fact that

3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 +
 α

2c −1

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h

= 3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 −1

4

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h

≤3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 −λ

4

k
X

i=1

eθi+1
h
−eθi
h

2

2

≤0.

The first inequality holds from Bi,h ⪰λId, and the second inequality is obvious from our setting of
λ. Therefore, we can conclude that
eθk
h −θ∗
h

Bk,h
≤βk(δ) = O(
√

d log U log(kH)) .

In the following section, we provide the proofs of the lemmas used in Lemma 12.

43


---Page Break---
D.1.1
Proof of Lemma 13

Proof of Lemma 13. Let eℓi,h(θ) = ℓi,h(eθi
h) + ∇ℓi,h(eθi
h)⊤
θ −eθi
h

+ 1

2
θ −eθi
h

2

∇2ℓi,h(eθi
h) be a

second-order Taylor expansion of ℓi,h(θ) at eθi
h. Since we have

eθk+1
h
= argmin
θ∈Bd(Lθ)

1
2η

θ −eθk
h

2

eBk,h
+ ∇ℓk,h(eθk
h)⊤θ =
argmin
θ∈B(0d,Lθ)
eℓk,h(θ) + 1

2η

θ −eθk
h

2

Bk,h
,

by Lemma 31, if we define ψ(θ) = 1

2∥θ∥2
Bi,h we obtain

∇eℓi,h(eθi+1
h
)⊤(eθi+1
h
−θ∗
h) ≤1

2η

eθi
h −θ∗
h

2

Bi,h
−
eθi+1
h
−θ∗
h

2

Bi,h
−
eθi+1
h
−eθi
h

Bi,h


.

(54)

By applying Lemma 33, we have

ℓi,h(eθi+1
h
) −ℓi,h(θ∗
h) ≤
D
∇ℓi,h(eθi+1
h
), eθi+1
h
−θ∗
h
E
−
1
αi,h

eθi+1
h
−θ∗
h

∇2ℓi,h(eθi+1
h
) ,
(55)

where αi,h = log |Si,h| + 2(1 + LφLθ).

By setting η = αi,h/2 and merging equations (54) and (55), we arrive at

ℓi,h(eθi+1
h
) −ℓi,h(θ∗
h) ≤
D
∇ℓi,h(eθi+1
h
) −∇eℓi,h(eθi+1
h
), eθi+1
h
−θ∗
h
E

+
1
αi,h

eθi
h −θ∗
h

2

Bi,h
−
eθi+1
h
−θ∗
h

2

Bi+1,h
−
eθi+1
h
−eθi
h

Bi,h


.

(56)

Meanwhile, we obtain

∇eℓi,h(θ) = ∇ℓi,h(eθi
h) + ∇2ℓi,h(eθi
h)(θ −eθi
h)
(57)

by taking the gradient over both sides of the Taylor approximation of ℓi,h(θ). Using (57), we proceed
to bound the first term of (56) as follows:
D
∇ℓi,h(eθi+1
h
) −∇eℓi,h(eθi+1
h
), eθi+1
h
−θ∗
h
E

=
D
∇ℓi,h(eθi+1
h
) −∇ℓi,h(eθi
h) −∇2ℓi,h(eθi
h)(eθi+1
h
−eθi
h), eθi+1
h
−θ∗E

=
D
D3ℓi,h(¯θ
i+1
h
)
h
eθi+1
h
−eθi
h
i
(eθi+1
h
−eθi
h), eθi+1
h
−θ∗E

≤3
√

2Lφ
eθi+1
h
−θ∗
h

2

eθi+1
h
−eθi
h

2

∇2ℓi,h(¯θi+1
h
)

≤3
√

2Lφ
eθi+1
h
−eθi
h

2

∇2ℓi,h(¯θi+1
h
)

≤3
√

2L3
φ
eθi+1
h
−eθi
h

2

2

where ¯θ
i+1
h
is a convex combination of eθi
h and eθi+1
h
. The second equality arises from the Taylor
expansion, the first inequality is due to the self-concordant property, and the final inequality is justified
by the following:

∇2ℓi,h(¯θ
i+1
h
)

=
X

s′∈Si,h
P¯θi+1
h
(s′ | si
h, ai
h)φi,h,s′φ⊤
i,h,s′

−
X

s′∈Si,h

X

s′′∈Si,h
P¯θi+1
h
(s′ | si
h, ai
h)P¯θi+1
h
(s′′ | si
h, ai
h)φi,h,s′φ⊤
i,h,s′′

⪯
X

s′∈Si,h
P¯θi+1
h
(s′ | si
h, ai
h)φi,h,s′φ⊤
i,h,s′

⪯L2
φId.

44


---Page Break---
By summing over i and reorganizing the terms, we arrive at the final result as follows:
eθk+1
h
−θ∗
h

2

Bk+1,h

≤

k
X

i=1
αi,h

ℓi,h(θ∗
h) −ℓi,h(eθi+1
h
)

+
eθ1
h −θ∗
h

2

B1,h

+ 3
√

2L3
φ

k
X

i=1
αi,h
eθi+1
h
−eθi
h

2

2 −

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h

≤α

k
X

i=1


ℓi,h(θ∗
h) −ℓi,h(eθi+1
h
)

+ λL2
θ + 3
√

2L3
φα

k
X

i=1

eθi+1
h
−eθi
h

2

2 −

k
X

i=1

eθi+1
h
−eθi
h

2

Bi,h
.

where the first inequality holds by Assumption 2 and the last inequality holds since α = log U +
2(1 + LφLθ) ≥αi,h for all i ∈[k].

D.1.2
Proof of Lemma 14

Proof of Lemma 14. The norm of ezi,h = σ+
i,h

Eθ∼N(eθi
h,cB−1
i,h)[σi,h(Φ⊤
i,hθ)]

is generally un-

bounded [27]. In this proof, we utilize the smoothed version of ezi,h, defined as follows:

ezu
i,h = σ+
i,h

smoothu
i,h Eθ∼N(eθi
h,cB−1
i,h)[σi,h(Φ⊤
i,hθ)]


where the smooth function smoothu
i,h(p) = (1 −u)p + (u/U)1 with u ∈[0, 1/2], and 1 ∈R|Si,h|
is an all-one vector.

Exploiting the property of σ+
i,h such that σi,h(σ+
i,h(p)) = p for any p ∈∆|Si,h|, it is straightforward
to show that ezu
i,h = σ+
i,h(smoothu
i,h(σi,h(ezi,h))). Then, by Lemma 34, we have

k
X

i=1
ℓ(ezu
i,h, yi
h) −

k
X

i=1
ℓ(ezi,h, yi
h) ≤2uk,
and
∥ezu
i,h∥∞≤log(U/u).
(58)

Given the definition of ℓi,h, we know that ℓ(z∗
i,h, yi
h) = ℓi,h(θ∗
h), where z∗
i,h = Φ⊤
i,hθ∗
h. We can
bound the gap between the loss of θ∗
h and ezu
i,h as follows:

k
X

i=1

 
ℓi,h(θ∗
h) −ℓ(ezu
i,h, yi
h)


=

k
X

i=1

 
ℓ(z∗
i,h, yi
h) −ℓ(ezu
i,h, yi
h)


≤

k
X

i=1
⟨∇zℓ(z∗
i,h, yi
h), z∗
i,h −ezu
i,h⟩−

k
X

i=1

1
Mi,h
∥z∗
i,h −ezu
i,h∥2
∇2
zℓ(z∗
i,h,yi
h)

=

k
X

i=1
⟨∇zℓ(z∗
i,h, yi
h), z∗
i,h −ezu
i,h⟩−

k
X

i=1

1
Mi,h
∥z∗
i,h −ezu
i,h∥2
∇σi,h(z∗
i,h),
(59)

where Mi,h = log(|Si,h|) + 2 log(U/u), and the second equality holds by a direct calculation of the
first order and Hessian of the logistic loss.

Now, we first bound the first term of the right-hand side. Let di,h = (z∗
i,h −ezu
i,h)/(M + LφLθ),
where M = log U + 2 log(U/u). Then, one can check that ∥di,h∥∞≤1 since ∥z∗
i,h∥∞≤
maxs′∈Si,h ∥φi,h,s′∥2∥θ∗
h∥2 ≤LφLθ and ∥ezu
i,h∥∞≤log(U/u). Moreover, since z∗
i,h and ezu
i,h
are independent of yi
h, di,h is Fi,h-measurable. Since E[(σi,h(z∗
i,h) −yi
h)(σi,h(z∗
i,h) −yi
h)⊤|
Fi,h] = ∇σi,h(z∗
i,h) and ∥σi,h(z∗
i,h)−yi
h∥1 ≤2, we can apply Lemma 32. For any k and δ ∈(0, 1],

45


---Page Break---
with probability at least 1 −δ/H, we have

k
X

i=1
⟨∇zℓ(z∗
i,h, yi
h), z∗
i,h −ezu
i,h⟩

= (M + LφLθ)

k
X

i=1
⟨∇zℓ(z∗
i,h, yi
h), di,h⟩

≤(M + LφLθ)

v
u
u
tλ +

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

·

v
u
u
u
t

√

λ
4
+
4
√

λ
log




H
q

1 + 1

λ
Pk
i=1 ∥di,h∥2
∇σi,h(z∗
i,h)
δ





≤(M + LφLθ)

v
u
u
tλ +

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

s√

λ
4
+ 4 log
H
√

1 + 2k

δ


,
(60)

where the second inequality holds since ∥di,h∥2
∇σi,h(z∗
i,h) = d⊤
i,h∇σi,h(z∗
i,h)di,h ≤2 and λ ≥1.
Plugging (60) into (59) and rearranging the term, we get

k
X

i=1

 
ℓi,h(θ∗) −ℓ(ezu
i,h, yi
h)


≤(M + LφLθ)

v
u
u
tλ +

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

s√

λ
4
+ 4 log

H
√

1 + 2k

δ



−

k
X

i=1

1
Mi,h
∥z∗
i,h −ezu
i,h∥2
∇σi,h(z∗
i,h)

≤(M + LφLθ)

v
u
u
tλ +

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

s√

λ
4
+ 4 log
H
√

1 + 2k

δ



−(M + LφLθ)

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

≤(M + LφLθ)

 

λ +

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

!

+ (M + LφLθ)

 √

λ
4
+ 4 log
H
√

1 + 2k

δ

!

−(M + LφLθ)

k
X

i=1
∥di,h∥2
∇σi,h(z∗
i,h)

≤5

4(M + LφLθ)λ + 4(M + LφLθ) log
H
√

1 + 2k

δ


.
(61)

Finally, combining (58) and (61), by setting u = 1/k, we derive that

k
X

i=1

 
ℓi,h(θ∗
h) −ℓ(ezi,h, yi
h)


≤5

4(M + LφLθ)λ + 4(M + LφLθ) log
H
√

1 + 2k

δ


+ 2uk

≤5

4(3 log(Uk) + LφLθ)λ + 4(3 log(Uk) + LφLθ) log
H
√

1 + 2k

δ


+ 2

where the last inequality holds by the definition of M = log U +2 log(U/u). Taking the union bound
over h ∈[H], we conclude the proof.

46


---Page Break---
D.1.3
Proof of Lemma 15

Proof of Lemma 15. We start the proof from the observation of Proposition 2 in Foster et al. [27],
stating that ezi,h represents the mixed prediction, which adheres to the following property:

ℓ(ezi,h, yi
h) ≤−log

Eθ∼N(eθi
h,cB−1
i,h) [exp (−ℓi,h(θ))]

= −log
 1

Zi,h

Z

Rd exp (−Li,h(θ)) dθ

,

(62)

where Li,h(θ) := ℓi,h(θ) + 1

2c
θ −eθi
h

2

Bi,h
and Zi,h :=
q

(2π)dc|B−1
i,h| .

Consider the quadratic approximation

eLi,h(θ) = Li,h(eθi+1
h
) +
D
∇Li,h(eθi+1
h
), θ −eθi+1
h
E
+ 1

2c

θ −eθi+1
h

2

Bi,h
.

Using the property that ℓi,h is 3
√

2Lφ-self-concordant-like function as asserted by Proposition B.1
in [50], and applying Lemma 35, we obtain

Li,h(θ) ≤eLi,h(θ) + exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇ℓi,h(eθi+1
h
) .

Also, we have

1
Zi,h

Z

Rd exp(−Li,h(θ)) dθ

≥
1
Zi,h

Z

Rd exp

−eLi,h(θ) −exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇ℓi,h(eθi+1
h
)


dθ

=
exp

−Li,h(eθi+1
h
)


Zi,h

Z

Rd
efi+1,h(θ) · exp

−
D
∇Li,h(eθi+1
h
), θ −eθi+1
h
E
dθ,
(63)

where we define the function efi,h : B(0d, 1) →R as

efi+1,h(θ) = exp

−1

2c

θ −eθi+1
h

2

Bi,h
−exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)


.

We denote eZi+1,h =
R

Rd efi+1,h(θ) dθ ≤+∞and define eΘi+1,h as the distribution whose density
function is efi+1,h(θ)/ eZi+1,h. Then, we can rewrite (63) as follows:

1
Zi,h

Z

Rd exp(−Li,h(θ)) dθ

≥
exp

−Li,h(eθi+1
h
)

eZi+1,h

Zi,h
Eθ∼eΘi+1,h

h
exp

−
D
∇Li,h(eθi+1
h
), θ −eθi+1
h
Ei

≥
exp

−Li,h(eθi+1
h
)

eZi+1,h

Zi,h
exp

−Eθ∼eΘi+1,h

hD
∇Li,h(eθi+1
h
), θ −eθi+1
h
Ei

=
exp

−Li,h(eθi+1
h
)

eZi+1,h

Zi,h
,
(64)

where the second inequality is by Jensen’s inequality and the last inequality holds because eΘi+1,h is

symmetric around eθi+1
h
and thus Eθ∼eΘi+1,h

hD
∇Li,h(eθi+1
h
), θ −eθi+1
h
Ei
= 0.

Combining (62) and (64), we get

ℓi,h(ez) ≤Li,h(eθi+1
h
) + log Zi,h −log eZi+1,h.
(65)

47


---Page Break---
Moreover, we have

−log eZi+1,h

= −log
Z

Rd exp

−1

2c

θ −eθi+1
h

2

Bi,h

−exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)


dθ


= −log

bZi+1,h · Eθ∼bΘi+1,h


exp

−exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)



≤−log bZi+1,h + Eθ∼bΘi+1,h


exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)



= −log Zi,h + Eθ∼bΘi+1,h


exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)


,
(66)

where bΘi+1,h = N(eθi+1
h
, cB−1
i,h) and bZi+1,h =
R

Rd exp

−1

2c
θ −eθi+1
h

2

Bi,h


dθ, and the last

inequality holds because bZi+1,h and Zi,h are identical normalizing factors. Integrating (65) and (66)
and summing over k, yields

k
X

i=1
ℓ(ezi,h, yi
h)

=

k
X

i=1
Li,h(eθi+1
h
) +

k
X

i=1
Eθ∼bΘi+1,h


exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)


.

Moreover, we can further bound the second term on the right-hand side of (66). By Cauchy-Schwarz
inequality, we get

Eθ∼bΘi+1,h


exp

18L2
φ
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)



≤

s

Eθ∼bΘi+1,h


exp

36L2φ
θ −eθi+1
h

2

2



|
{z
}
(I)

s

Eθ∼bΘi+1,h

θ −eθi+1
h

4

∇2ℓi,h(eθi+1
h
)



|
{z
}
(II)

.

Since bΘi+1,h = N

eθi+1
h
, cB−1
i,h

, θ −eθi+1
h
follows the same distribution as

d
X

j=1

r

cλj

B−1
i,h

Xjej,
where Xj
i.i.d.
∼N(0, 1), ∀j ∈[d],
(67)

where λj

B−1
i,h

denotes the j-th largest eigenvalue of B−1
i,h and {e1, . . . , ed} are orthogonal basis

of Rd. Furthermore, since we know that B−1
i,h ≤λ−1Id, we can bound the term (I) by

(I) ≤

v
u
u
u
tEXj




d
Y

j=1
exp
 
36L2φcλ−1X2
j



=

v
u
u
t

d
Y

j=1
EXj

exp
 
36L2φcλ−1X2
j


≤
 
EW ∼χ2 
exp
 
36L2
φcλ−1W
 d

2 ≤EW ∼χ2 
exp
 
18L2
φcλ−1Wd


where χ2 is the chi-square distribution and the last inequality holds due to Jensen’s inequality. By
choosing λ ≥72L2
φcd, we arrive that

(I) ≤EW ∼χ2

exp
W

4


≤
√

2,
(68)

48


---Page Break---
where the last inequality holds because the moment-generating function for χ2-distribution is bounded
by EW ∼χ2[exp(tW)] ≤1/√1 −2t for all t ≤1/2. Now, we bound the term (II).

(II) =

s

Eθ∼bΘi+1,h

θ −eθi+1
h

4

∇2ℓi,h(eθi+1
h
)


=

s

Eθ∼N(0,cB−1
i,h)


∥θ∥4
∇2ℓi,h(eθi+1
h
)



=
q

Eθ∼N(0,c ¯B−1
i,h) [∥θ∥4
2],

where ¯Bi,h =

∇2ℓi,h(eθi+1
h
)
−1/2
Bi,h

∇2ℓi,h(eθi+1
h
)
−1/2
. Let ¯λj := λj

c ¯B−1
i,h

be the j-th
largest eigenvalue of the matrix. Then, a similar analysis as (67) gives that

(II) =

v
u
u
u
u
tEXj∼N(0,1)







d
X

j=1

q

¯λjXjej



4

2



=

v
u
u
u
u
tEXj∼N(0,1)








d
X

j=1
¯λjX2
j





2



=

v
u
u
t

d
X

j=1

d
X

j′=1
¯λj¯λj′EXj,Xj′∼N(0,1)[X2
j X2
j′] ≤

v
u
u
t3

d
X

j=1

d
X

j′=1
¯λj¯λj′ =
√

3c tr

¯B−1
i,h

,

where the last inequality holds due to EXj,Xj′∼N(0,1)[X2
j X2
j′] ≤3 when considering the case where

j = j′ and the last equality is derived from the fact that
Pd
j=1 ¯λj
2
= tr

c ¯B−1
i,h

. Here, we denote

tr(A) as the trace of the matrix A.

We define matrix Ri+1,h := λId/2 + Pi
τ=1 ∇2ℓτ,h(θτ+1,h). Under the condition λ ≥2L2
φ, we
have ∇2ℓi,h(θi+1,h) ⪯L2
φId ≤λ

2 Id. Then, we have Bi,h ⪰Ri+1,h. Therefore, we can bound the
trace by

tr

¯B−1
i,h

= tr

B−1
i,h∇2ℓi,h(θi+1,h)

≤tr

R−1
i+1,h∇2ℓi,h(θi+1,h)


= tr

R−1
i+1,h(Ri+1,h −Ri,h)

≤log det(Ri+1,h)

det(Ri,h) ,

where the last inequality holds due to Lemma 4.7 of Hazan et al. [32]. Therefore we can bound the
term (II) as

(II) ≤
√

3c log det(Ri+1,h)

det(Ri,h) .
(69)

Combining (68) and (69), we get

Eθ∼bΘi+1,h


exp

6
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇2ℓi,h(eθi+1
h
)


≤
√

6c log det(Ri+1,h)

det(Ri,h) .
(70)

Plugging (66) and (70) into (65), and taking summation over k, we derive that

k
X

i=1
ℓ(ezi,h, yi
h) ≤

k
X

i=1
Li,h(eθi+1
h
) +
√

6c

k
X

i=1
log det(Ri+1,h)

det(Ri,h)

=

k
X

i=1


ℓi,h(eθi+1
h
) + 1

2c

eθi+1
h
−eθi
h

2

Bi,h


+
√

6c

k
X

i=1
log det(Ri+1,h)

det(Ri,h)

≤

k
X

i=1


ℓi,h(eθi+1
h
) + 1

2c

eθi+1
h
−eθi
h

2

Bi,h


+
√

6cd log

 

1 + 2kL2
φ
dλ

!

,

where the last inequality holds because Pk
i=1 log det(Ri+1,h)

det(Ri,h)
= log(det(Rk+1,h)/ det(λ/2Id)) ≤

d log

1 +
2kL2
φ
dλ

. By rearranging the terms, we conclude the proof.

49


---Page Break---
D.2
Bound on Prediction Error

In this section, we present the bound on the prediction error of parameters updated by ORRL-MNL.
First, we compare the problem setting of MNL contextual bandits with ours and introduce the
challenges of applying their analysis to our setting.

MNL dynamic assortment optimization (single-parameter & uniform reward) [61]
Perivier
and Goyal [61] consider an assortment selection problem where the user choice is given by a
MNL choice model with the single-parameter. At each time t, the agent observes context features
{xt,i}M
i=1 ⊂Rd. Then the agent decides on the set St ⊂[M] to offer to a user, with |St| ≤N.
Without loss of generality, we may assume |St| = N. Then the user purchases one single product
j ∈St ∪{0} and the probability of each product j is purchased by a user follows the MNL model
parametrized by a unknown fixed parameter θ∗∈Rd,

qt,j(St, θ∗) :=






exp(x⊤
t,jθ∗)
1+P

k∈St exp(x⊤
k θ∗)
if j ∈St

1
1+P

k∈St exp(x⊤
k θ∗)
if j = 0 .

Then the difference between the revenue induced by θ∗and that by an estimator θ in Perivier and
Goyal [61] is expressed as follows:
X

j∈St
qt,j(St, θ∗) −
X

j∈St
qt,j(St, θ) .
(71)

If we define Q : RN →R, such that for all u = (u1, . . . , uN) ∈RN, Q(u) := PN
i=1
exp(ui)
1+PN
j=1 exp(uj)
and let v∗= (x⊤
t,i1θ∗, . . . , x⊤
t,iN θ∗) and v = (x⊤
t,i1θ, . . . , x⊤
t,iN θ), then Eq. (71) can be expressed
as follows:
X

j∈St
qt,j(St, θ∗) −
X

j∈St
qt,j(St, θ) = Q(v∗) −Q(v)

= ∇Q(v∗)⊤(v∗−v) + 1

2(v∗−v)⊤∇2Q(¯v)(v∗−v) , (72)

where ¯v is a convex combination of v∗and v. For the first term in Eq. (72), we have

∇Q(v∗)⊤(v∗−v)

=

P
i∈St exp(x⊤
t,jθ∗)(vj −v∗
j )

1 + P
j∈St exp(x⊤
t,jθ∗)
−

P

i∈St exp(x⊤
t,jθ∗) P

i∈St exp(x⊤
t,iθ∗)(vj −v∗
j )

1 + P
j∈St exp(x⊤
t,jθ∗)
2

=
X

j∈St
qt,j(St, θ∗)x⊤
t,j(θ∗−θ) −
X

j∈St

X

i∈St
qt,j(St, θ∗)qt,j(St, θ∗)x⊤
t,i(θ∗−θ)

=
X

j∈St
qt,j(St, θ∗)

 

1 −
X

i∈St
qt,i(St, θ∗)

!

x⊤
t,j(θ∗−θ)

=
X

j∈St
qt,j(St, θ∗)qt,0(St, θ∗)x⊤
t,j(θ∗−θ)

≤
X

j∈St
qt,j(St, θ∗)qt,0(St, θ∗)∥xt,j∥H−1
t
(θ∗)∥θ∗−θ∥Ht(θ∗) ,
(73)

where Ht(θ) is the Gram matrix used in [61] defined by

Ht(θ∗) :=

t−1
X

τ=1

X

j∈Sτ
qτ,j(Sτ, θ∗)xτ,jx⊤
τ,j −
X

j∈Sτ

X

i∈Sτ
qτ,j(Sτ, θ∗)qτ,i(Sτ, θ∗)xτ,jx⊤
τ,i .

Note that the term ∥θ∗−θ∥Ht(θ∗) can be bounded by the concentration result of the es-
timated parameter.
On the other hand, to apply the elliptical potential lemma to the term

50


---Page Break---
P
j∈St qt,j(St, θ∗)qt,0(St, θ∗)∥xt,j∥H−1
t
(θ∗), note that Ht(θ∗) can be bounded as follows:

Ht(θ∗)

= Ht−1(θ∗) +
X

j∈St
qt,j(St, θ∗)xt,jx⊤
t,j −1

2

X

i,j∈St
qt,j(St, θ∗)qt,i(St, θ∗)
 
xt,jx⊤
t,i + xt,ix⊤
t,j


⪰Ht−1(θ∗) +
X

j∈St
qt,j(St, θ∗)xt,jx⊤
t,j −1

2

X

i,j∈St
qt,j(St, θ∗)qt,i(St, θ∗)
 
xt,jx⊤
t,j + xt,ix⊤
t,i


= Ht−1(θ∗) +
X

j∈St
qt,j(St, θ∗)

 

1 −
X

i∈St
qt,i(St, θ∗)

!

xt,jx⊤
t,j

= Ht−1(θ∗) +
X

j∈St
qt,j(St, θ∗)qt,0(St, θ∗)xt,jx⊤
t,j .
(74)

Now since the coefficient qt,j(St, θ∗)qt,0(St, θ∗) of ∥x∥H−1
t
(θ∗) in Eq. (73) aligns with the coeffi-
cients of the lower bound of Ht(θ∗) in Eq. (74), the elliptical potential lemma can be applied. Note
that such a lower bound in Eq. (74) holds since Perivier and Goyal [61] deals with the uniform reward,
i.e., 1 −P

i∈St qt,i(St, θ∗) = qt,0(St, θ∗).

Mulitinomial logistic bandit problem [76]
Zhang and Sugiyama [76] address the multiple-
parameter MNL contextual bandit problem where at each time step t the agent selects an action
xt ∈Rd and receives response feedback yt ∈{0} ∪[N] with N + 1 possible outcomes. Each
outcome i ∈[N] is associated with a ground-truth parameter θ∗
i ∈Rd, and the probability of the
outcome P(yt = i | xt) follows the MNL model,

P(yt = i | xt) =
exp(x⊤
t θ∗
i )

1 + PN
j=1 exp(x⊤
t θ∗
j)
,
P(yt = 0 | xt) = 1 −

N
X

j=1
P(yt = j | xt) .

In this model, there are N unknown choice parameter Θ∗:= [θ∗
1, . . . , θ∗
N] ∈Rd×N and the agent
chooses one context feature xt, that is why we call multiple-parameter MNL model. Then, the
expected revenue of an action xt in [76] is given by

N
X

i=1

exp(x⊤
t θ∗
i )ρi
1 + PN
j=1 exp(x⊤
t θ∗
j)
:= ρ⊤σ(Θ∗xt) ,

where we define the softmax function σ : RN →[0, 1]N by

[σ(z)]k =
exp([z]i)

1 + PN
j=1 exp([z]j)
∀k ∈[N]
and
[σ(z)]0 =
1

1 + PN
j=1 exp([z]j)
∀k ∈[N] ,

and ρ := [ρ1, . . . , ρN] ∈RN+1
+
represents the reward for each outcome j ∈[N] with ρ0 = 0. Then,
the difference between the revenue induced by Θ∗and that by an estimator ˆΘ in [76] is expressed by

ρ⊤
σ(Θ∗xt) −σ( ˆΘxt)


=

N
X

k=1
ρk

[σ(Θ∗xt)]k −[σ( ˆΘxt)]k


=

N
X

k=1
ρk

∇[σ( ˆΘxt)]k
⊤
(Θ∗−ˆΘ)xt +

N
X

k=1
ρk∥(Θ∗−Θ)xt∥Ξk ,
(75)

51


---Page Break---
where Ξk =
R 1
0 (1 −ν)∇2[σ( ˆΘxt + ν(Θ∗−ˆΘ)xt)]kdν. Then for the first term in Eq. (75), we
have

N
X

k=1
ρk

∇[σ( ˆΘxt)]k
⊤
(Θ∗−ˆΘ)xt

≤
ρ⊤∇σ( ˆΘxt)(Θ∗−ˆΘ)xt


=
ρ⊤∇σ( ˆΘxt)(IN ⊗x⊤
t )(vec(Θ∗) −vec( ˆΘ))


≤∥vec(Θ∗) −vec( ˆΘ)∥Ht∥H
−1

2
t
(IN ⊗x⊤
t )∇σ( ˆΘxt)ρ∥2
(76)

where Ht is the Gram matrix used in [76] defined by

Ht := λIN +

t−1
X

s=1
∇σ( ˆΘs+1xs) ⊗xsx⊤
s .

Note that the term ∥vec(Θ∗) −vec( ˆΘ)∥Ht in Eq. (76) can be bounded by the concentration result

of the estimated parameter, and the term ∥H
−1

2
t
(IN ⊗x⊤
t )∇σ( ˆΘxt)ρ∥2 also can be bounded as
follows:
∥H
−1

2
t
(IN ⊗x⊤
t )∇σ( ˆΘxt)ρ∥2 ≤∥ρ∥2∥H
−1

2
t
(IN ⊗x⊤
t )∇σ( ˆΘxt)∥2 .

Here Zhang and Sugiyama [76] bound the term ∥H
−1

2
t
(IN ⊗x⊤
t )∇σ( ˆΘxt)∥2 using a matrix version
of elliptical lemma. However, they assume ∥ρ∥2 ≤R (Assumption 2 in [76]).

Now, regarding the prediction error in our setting, the estimated values (eV k
h+1(·)) for each reachable
state are typically distinct, and we do not assume a constant upper bound on the ℓ2-norm of the
estimated value vector for all reachable states. Instead, we can bound the ℓ2-norm of the estimated
value vector for all reachable states as follows:

∥eVk
h+1(s, a)∥2 ≤max
s′∈Ss,a

eV k
h+1(s′)

q

|Ss,a| ≤H
√

U ,

where eVk
h+1(s, a) :=
h
eV k
h+1(s′)
i

s′∈Ss,a ∈R|Ss,a|. However, such a bound leads to a looser regret

by a factor of
√

U. To address, we adapt the feature centralization technique [50] to bound the
prediction error independently of U, without making any additional assumptions. The key point is
that the Hessian of per-round loss ℓk,h(θ) is expressed in terms of the centralized feature as follows:

∇2ℓk,h(θ) =
X

s′∈Sk,h
Pθ(s′ | sk
h, ak
h)¯φ(sk
h, ak
h, s′; θ)¯φ(sk
h, ak
h, s′; θ)⊤.

where ¯φ(s, a, s′; θ) := φ(s, a, s′) −Ees∼Pθ(·|s,a)[φ(s, a, es)] is the centralized feature by θ. Now, we
provide the bound on prediction error of the estimated parameter updated by ORRL-MNL.
Lemma 16 (Bound on the prediction error). For any δ ∈(0, 1), suppose that Lemma 12 holds. Let
us denote the prediction error about eθk
h by

∆k
h(s, a) :=
X

s′∈Ss,a


Peθk
h(s′ | s, a) −Pθ∗
h(s′ | s, a)

eV k
h+1(s′) .

Then, for any (s, a) ∈S × A, we have

|∆k
h(s, a)| ≤Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h
+ 3Hβk(δ)2 max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h .

Proof of Lemma 16. Let us define F(θ) := P

s′∈Ss,a Pθ(s′ | s, a)eV k
h+1(s′). Then, by Taylor expan-
sion we have

F(θ∗
h) = F(eθk
h) + ∇F(eθk
h)⊤(θ∗
h −eθk
h) + 1

2(θ∗
h −eθk
h)⊤∇2F(¯θ)(θ∗
h −eθk
h) ,

52


---Page Break---
where ¯θ = (1 −v)θ∗
h + v eθk
h for some v ∈(0, 1). By Proposition 1, we have

∇F(θ) =
X

s′∈Ss,a
∇Pθ(s′ | s, a)eV k
h+1(s′)

=
X

s′∈Ss,a
Pθ(s′ | s, a)



φs,a,s′ −
X

es∈Ss,a
Pθ(es | s, a)φs,a,es



eV k
h+1(s′)

=
X

s′∈Ss,a
Pθ(s′ | s, a)¯φs,a,s′(θ)eV k
h+1(s′) ,

and

∇2F(θ) =
X

s′∈Ss,a
∇2Pθ(s′ | s, a)eV k
h+1(s′)

=
X

s′∈Ss,a
Pθ(s′ | s, a)eV k
h+1(s′)φs,a,s′φ⊤
s,a,s′

−
X

s′∈Ss,a
Pθ(s′ | s, a)eV k
h+1(s′)

·
X

s′′∈Ss,a
Pθ(s′′ | s, a)
 
φs,a,s′φ⊤
s,a,s′′ + φs,a,s′′φ⊤
s,a,s′ + φs,a,s′′φ⊤
s,a,s′′


+ 2
X

s′∈Ss,a
Pθ(s′ | s, a)eV k
h+1(s′)

·



X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′







X

s′′∈Ss,a
Pθ(s′′ | s, a)φs,a,s′′





⊤

.

Then, the prediction error can be bounded as follows:

|∆k
h(s, a)| = |F(θ∗
h) −F(eθk
h)|

≤
∇F(eθk
h)⊤(eθk
h −θ∗
h)
 + 1

2

(eθk
h −θ∗
h)⊤∇2F(¯θ)(eθk
h −θ∗
h)
 .
(77)

For the first term in Eq. (77),

∇F(eθk
h)⊤(eθk
h −θ∗
h)
 =



X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φs,a,s′(eθk
h)⊤(eθk
h −θ∗
h)eV k
h+1(s′)



≤H
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

eθk
h −θ∗
h

Bk,h

≤Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h
,
(78)

where in the first inequality we use eV k
h+1(s′) ≤H and Cauchy-Scharwz inequality, and the second
inequality follows by the concentration result of Lemma 12.

53


---Page Break---
For the second term in Eq. (77), since 0 ≤eV k
h+1(s′) ≤H,

(eθk
h −θ∗
h)⊤∇2F(¯θ)(eθk
h −θ∗
h)


≤H
X

s′∈Ss,a
P¯θ(s′ | s, a)

(eθk
h −θ∗
h)⊤φs,a,s′
2

+ H
X

s′∈Ss,a
P¯θ(s′ | s, a)

·
X

s′′∈Ss,a
P¯θ(s′′ | s, a)
(eθk
h −θ∗
h)⊤ 
φs,a,s′φ⊤
s,a,s′′ + φs,a,s′′φ⊤
s,a,s′

(eθk
h −θ∗
h)


+ H
X

s′∈Ss,a
P¯θ(s′ | s, a)
X

s′′∈Ss,a
P¯θ(s′′ | s, a)

(eθk
h −θ∗
h)⊤φs,a,s′′
2

+ 2H



(eθk
h −θ∗
h)⊤

X

s′′∈Ss,a
P¯θ(s′′ | s, a)φs,a,s′′




2

≤H
X

s′∈Ss,a
P¯θ(s′ | s, a)∥φs,a,s′∥2
B−1
k,h

eθk
h −θ∗
h

2

Bk,h

+ H
X

s′∈Ss,a
P¯θ(s′ | s, a)

·
X

s′′∈Ss,a
P¯θ(s′′ | s, a)
(eθk
h −θ∗
h)⊤ 
φs,a,s′φ⊤
s,a,s′ + φs,a,s′′φ⊤
s,a,s′′

(eθk
h −θ∗
h)


+ H
X

s′′∈Ss,a
P¯θ(s′′ | s, a)∥φs,a,s′′∥2
B−1
k,h

eθk
h −θ∗
h

2

Bk,h

+ 2H

X

s′′∈Ss,a
P¯θ(s′′ | s, a)∥φs,a,s′′∥B−1
k,h

eθk
h −θ∗
h

Bk,h

2
,
(79)

where for the second inequality we use Cauchy-Schwarz inequality, xx⊤+ yy⊤⪰xy⊤+ yx⊤for
any x, y ∈Rd, and triangle inequality. Note that

H
X

s′∈Ss,a
P¯θ(s′ | s, a)

·
X

s′′∈Ss,a
P¯θ(s′′ | s, a)
(eθk
h −θ∗
h)⊤ 
φs,a,s′φ⊤
s,a,s′ + φs,a,s′′φ⊤
s,a,s′′

(eθk
h −θ∗
h)


= H
X

s′∈Ss,a
P¯θ(s′ | s, a)

(eθk
h −θ∗
h)⊤φs,a,s′
2

+ H
X

s′′∈Ss,a
P¯θ(s′′ | s, a)

(eθk
h −θ∗
h)⊤φs,a,s′′
2

≤2H
X

s′∈Ss,a
P¯θ(s′ | s, a)∥φs,a,s′∥2
B−1
k,h

eθk
h −θ∗
h

2

Bk,h
.
(80)

54


---Page Break---
By substituting Eq. (80) into Eq. (79) we have
(eθk
h −θ∗
h)⊤∇2F(¯θ)(eθk
h −θ∗
h)


≤4H
X

s′∈Ss,a
P¯θ(s′ | s, a)∥φs,a,s′∥2
B−1
k,h

eθk
h −θ∗
h

2

Bk,h

+ 2H

X

s′′∈Ss,a
P¯θ(s′′ | s, a)∥φs,a,s′′∥B−1
k,h

eθk
h −θ∗
h

Bk,h

2

≤4Hβ2
k max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h + 2H

βk max
s′∈Ss,a ∥φs,a,s′∥B−1
k,h

2

≤6Hβ2
k max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h ,
(81)

where for the second inequality follows by Lemma 12 and P

s′∈Ss,a P¯θ(s′ | s, a) = 1. Combining
the results of Eq. (78) and Eq. (81) and , we conclude the proof.

D.3
Good Events with High Probability

In this section, we introduce the good events used to prove Theorem 2 and show that the good events
happen with high probability.

Lemma 17 (Good event probability). For any K ∈N and δ ∈(0, 1), the good event G(K, δ′) holds
with probability at least 1 −δ where δ′ = δ/(2KH).

Proof of Lemma 17. For any δ′ ∈(0, 1), we have

G(K, δ′) =
\

k≤K

\

h≤H
Gk,h(δ′) =
\

k≤K

\

h≤H

n
G∆
k,h(δ′) ∩Gξ
k,h(δ′)
o
.

On the other hand, for any (k, h) ∈[K] × [H], by Lemma 30 Gξ
k,h(δ′) holds with probability at least
1 −δ′. Then, for δ′ = δ/(2KH) by taking union bound, we have the desired result as follows:

P(G(K, δ′)) ≥(1 −δ′)2KH ≥1 −2KHδ′ = 1 −δ .

D.4
Stochastic Optimism

Lemma 18 (Stochastic optimism). For any δ with 0 < δ < Φ(−1)/2, let σk = Hβk(δ). If we take
multiple sample size M = ⌈1 −log(HU)

log Φ(1) ⌉, then for any k ∈[K], we have

P

(eV k
1 −V ∗
1 )(sk
1) ≥0 | sk
1, Fk

≥Φ(−1)/2 .

Proof of Lemma 18. First, we introduce the following lemmas.

Lemma 19. Let δ ∈(0, 1) be given. For any (k, h) ∈[K] × [H], let σk = Hβk(δ). If we define the
event G∆
k,h(δ) as

G∆
k,h(δ) :=

|∆k
h(s, a)| ≤Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

+ 3Hβk(δ)2 max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h


,

then conditioned on G∆
k,h(δ), for any (s, a) ∈S × A, we have

P
 
−ιk
h(s, a) ≥0 | G∆
k,h(δ)

≥1 −Φ(1)M .

55


---Page Break---
Lemma 20. Let δ ∈(0, 1) be given. For any (h, k) ∈[H] × [K], let σk = Hβk(δ). If we take
multiple sample size M = ⌈1 −log(HU)

log Φ(1) ⌉, then conditioned on the event G∆
k (δ) := ∩h∈[H]G∆
k,h(δ),
we have

P
 
−ιk
h(sh, ah) ≥0, ∀h ∈[H] | G∆
k (δ)

≥Φ(−1) .

Based on the result of Lemma 20, using the same argument as in Lemma 6 we obtain the desired
result.

In the following section, we provide the proofs of the lemmas used in Lemma 18.

D.4.1
Proof of Lemma 19

Proof of Lemma 19. Recall the definition of Bellman error (Definition 1), we have

−ιk
h(s, a)

= eQk
h(s, a) −

r(s, a) + Ph eV k
h+1(s, a)


= min

r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) + νrand
k,h (s, a)

−

r(s, a) + Ph eV k
h+1(s, a)


≥min
 X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) −Ph eV k
h+1(s, a) + νrand
k,h (s, a), 0

.

Then, it is enough to show that

X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) −Ph eV k
h+1(s, a) + νrand
k,h (s, a) ≥0

at least with constant probability. On the other hand, under the event G∆
k,h(δ), by Lemma 16 we have

X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) −Ph eV k
h+1(s, a) + νrand
k,h (s, a)

= ∆k
h(s, a) + νrand
k,h (s, a)

≥−Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

−3Hβk(δ)2 max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h + νrand
k,h (s, a)

=
X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φs,a,s′(eθk
h)⊤ξs′
k,h −Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h
.

Note that since ξ(m)
k,h ∼N(0, σ2
kB−1
k,h), it follows that

¯φs,a,s′(eθk
h)⊤ξ(m)
k,h ∼N

0, σ2
k
¯φs,a,s′(eθk
h)

2

B−1
k,h


,
∀m ∈[M] .

Therefore, by setting σk = Hβk(δ), we have for m ∈[M] and s′ ∈Ss,a,

P

¯φs,a,s′(eθk
h)⊤ξ(m)
k,h ≥Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h


= Φ(−1) .

56


---Page Break---
Recall that ξs′
k,h := ξm(s′)
k,h
where m(s′) := argmaxm∈[M] ¯φs,a,s′(eθk
h)⊤ξ(m)
k,h . Then, we can deduce

P

¯φs,a,s′(eθk
h)⊤ξs′
k,h ≥Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h



= P

max
m∈[M] ¯φs,a,s′(eθk
h)⊤ξ(m)
k,h ≥Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h



= 1 −P

¯φs,a,s′(eθk
h)⊤ξ(m)
k,h < Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h
, ∀m ∈[M]


≥1 −(1 −Φ(−1))M

= 1 −Φ(1)M .
(82)
Consequently, we arrive at the conclusion as follows:

P(−ιk
h(s, a) ≥0 | G∆
k,h(δ))

≥P



X

s′∈Ss,a
Peθk
h(s′ | s, a)eV k
h+1(s′) −Ph eV k
h+1(s, a) + νrand
k,h (s, a) ≥0 | G∆
k,h(δ)





≥P



X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φs,a,s′(eθk
h)⊤ξs′
k,h
(83)

≥Hβk(δ)
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h
| G∆
k,h(δ)





≥P

¯φs,a,s′(eθk
h)⊤ξs′
k,h ≥Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h
, ∀s′ ∈Ss,a | G∆
k,h(δ)


= 1 −P

∃s′ ∈Ss,a s.t. ¯φs,a,s′(eθk
h)⊤ξs′
k,h < Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h
| G∆
k,h(δ)


≥1 −UP

¯φs,a,s′(eθk
h)⊤ξs′
k,h < Hβk(δ)
¯φs,a,s′(eθk
h)

B−1
k,h
| G∆
k,h(δ)

(84)

≥1 −UΦ(1)M ,
(85)
where (84) comes from the fact that maxs,a |Ss,a| = U and the union bound, and (85) follows
by (82).

D.4.2
Proof of Lemma 20

Proof of Lemma 20. It holds

P
 
−ιk
h(sh, ah) ≥0, ∀h ∈[H]

= 1 −P
 
∃h ∈[H] s.t. −ιk
h(sh, ah) < 0


≥1 −HP
 
−ιk
h(sh, ah) < 0


≥1 −HUΦ(1)M

≥Φ(−1)
where the first inequality uses the Bernoulli’s inequality, the second inequality follows by Lemma 19,
and the last inequality holds due to the choice of M = ⌈1 −log(UH)

log Φ(1) ⌉.

D.5
Bound on Estimation Part

In this section, we provide the upper bound on the estimation part of the regret: PK
k=1(eV k
1 −V ∗
1 )(sk
1).
Lemma 21 (Bound on estimation). For any δ ∈(0, 1), if λ = O(L2
φd log U), then with probability
at least 1 −δ/2, we have

K
X

k=1
(eV k
1 −V πk
1
)(sk
1) = e
O

d3/2H3/2√

T + κ−1d2H2
.

57


---Page Break---
Proof of Lemma 21. With the same argument in Lemma 10, we have

(eV k
1 −V πk
1
)(sk
1) =

H
X

h=1
−ιk
h(sk
h, ak
h) +

H
X

h=1
˙ζk
h ,
(86)

where ˙ζk
h := Ph(eV k
h+1 −V πk
h+1)(sk
h, ak
h) −(eV k
h+1 −V πk
h+1)(sk
h+1). Note that

−ιk
h(sk
h, ak
h) = eQk
h(sk
h, ak
h) −

r(sk
h, ak
h) + Ph eV k
h+1(sk
h, ak
h)


≤
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)eV k
h+1(s′) −Ph eV k
h+1(sk
h, ak
h) + νrand
k,h (sk
h, ak
h)

≤
∆k
h(sk
h, ak
h)
 + νrand
k,h (sk
h, ak
h)

≤Hβk
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)
¯φk,h,s′(eθk
h)

B−1
k,h
+ 3Hβ2
k max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h

+ νrand
k,h (sk
h, ak
h) ,
(87)

where the last inequality follows by Lemma 16. Now we introduce the following lemma.

Lemma 22. For any (k, h) ∈[K] × [H] and (s, a) ∈S × A, it holds

X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

≤
X

s′∈Ss,a
Peθk+1
h
(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h
+ 16ηLφ
√

λ
max
s′∈Ss,a

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h
.

By plugging the result of Lemma 22 into Eq. (87), we have

−ιk
h(sk
h, ak
h)

≤Hβk
X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+ Hβk
16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
+ 3Hβ2
k max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h + νrand
k,h (sk
h, ak
h)

≤Hβk
X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)¯φk,h,s′(eθk
h)⊤ξs′
k,h

+ Hβk
16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
+ 6Hβ2
k max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h .

By letting us denote

Υk
h(s, a) := Hβk
16ηLφ
√

λ
max
s′∈Ss,a

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h
+ 6Hβ2
k max
s′∈Ss,a ∥φs,a,s′∥2
B−1
k,h ,
(88)

58


---Page Break---
and summing over all episodes, we have

K
X

k=1
(eV k
1 −V πk
1
)(sk
1) =

K
X

k=1

H
X

h=1
−ιk
h(sk
h, ak
h) +

K
X

k=1

H
X

h=1
˙ζk
h

≤HβK

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h
|
{z
}
(i)

+

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)¯φk,h,s′(eθk
h)⊤ξs′
k,h

|
{z
}
(ii)

+

K
X

k=1

H
X

h=1
Υk
h(sk
h, ak
h)

|
{z
}
(iii)

+

K
X

k=1

H
X

h=1
˙ζk
h
|
{z
}
(iv)

.
(89)

Note that PK
k=1
PH
h=1
P

s′∈Sk,h is hereafter abbreviated as P

k,h,s′.

For term (i), we have

X

k,h,s′
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

≤
s X

k,h,s′
Peθk+1
h
(s′ | sk
h, ak
h)

s X

k,h,s′
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

2

B−1
k,h

=
√

T

v
u
u
t

H
X

h=1

K
X

k=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

2

B−1
k,h

≤
√

T

s

2Hd log

1 + KUL2φ

dλ


,
(90)

where the last inequality follows by the following lemma:

Lemma 23. For each h ∈[H], if λ ≥L2
φ, then we have

K
X

k=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
≤2d log

 

1 + KUL2
φ
dλ

!

.

Then, term (i) can be bounded as follows:

(i) = HβK

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

≤HβK
√

T

s

2Hd log

1 + KUL2φ

dλ



= e
O(dH3/2√

T) .
(91)

For term (ii), we introduce the following lemma:

59


---Page Break---
Lemma 24. Let δ ∈(0, 1) be given. For any (k, h) ∈[K]×[H] and (s, a) ∈S ×A, with probability
at least 1 −δ, it holds
X

s′∈Ss,a
Peθk
h(s′ | s, a)¯φs,a,s′(eθk
h)⊤ξs′
k,h

≤γk(δ)

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+ 16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h


,

where γk(δ) := Cξσk
p

d log(Md/δ) for an absolute constant Cξ > 0.

By Lemma 24, we have
X

k,h,s′
Peθk
h(s′ | sk
h, ak
h)¯φk,h,s′(eθk
h)⊤ξs′
k,h

≤γK(δ)
 X

k,h,s′
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+ 16ηLφ
√

λ

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h



≤γK(δ)
√

T

s

2Hd log

1 + KUL2φ

dλ


+ 16ηLφ
√

λ

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h


,

(92)

where the last inequality follows by Eq. (90). Note that

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h

≤

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

A−1
k,h

=

K
X

k=1

H
X

h=1
max
s′∈Sk,h


φk,h,s′ −
X

es∈Sk,h
Peθk+1
h
(es | sk
h, ak
h)φk,h,es



2

A−1
k,h

≤

K
X

k=1

H
X

h=1
max
s′∈Sk,h




2∥φk,h,s′∥2
A−1
k,h + 2



X

es∈Sk,h
Peθk+1
h
(es | sk
h, ak
h)φk,h,es



2

A−1
k,h






≤2

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k,h + 2

K
X

k=1

H
X

h=1

X

es∈Sk,h
Peθk+1
h
(es | sk
h, ak
h)∥φk,h,es∥2
A−1
k,h

≤4

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k,h

≤16κ−1dH log

 

1 + KUL2
φ
dλ

!

,
(93)

where the first inequality holds since B−1
k,h ⪯A−1
k,h, the second inequality follows from (x + y)2 ≤
2x2 + 2y2, and the third inequality uses the triangle inequality, and the fourth inequality uses
P
es∈Sk,h Peθk+1
h
(es | sk
h, ak
h) = 1, and the last inequality follows by Lemma 3. By substituting

60


---Page Break---
Eq. (93) into Eq. (92), we have

(ii) ≤γK(δ)
√

T
q

2Hd log
 
1 + KUL2φ/(dλ)

+ 256ηLφ
√

λ
κ−1dH log
 
1 + KUL2
φ/(dλ)
 

= e
O(d3/2H3/2√

T + κ−1d3/2H2) .
(94)

For term (iii),

K
X

k=1

H
X

h=1
Υk
h(sk
h, ak
h)

=

K
X

k=1

H
X

h=1


Hβk
16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
+ 6Hβ2
k max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h



≤HβK
16ηLφ
√

λ

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
+ 6Hβ2
K

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k,h

≤βK
256ηLφ
√

λ
κ−1dH2 log
 
1 + KUL2
φ/(dλ)

+ 24κ−1dH2β2
K log
 
1 + KUL2
φ/(dλ)


= e
O(κ−1d2H2) ,
(95)

where for the second inequality we use the same argument used to derive Eq. (93) and Lemma 3.

For term (iv), since we have | ˙ζk
h| ≤2H and E[ ˙ζk
h | Fk,h] = 0, which means { ˙ζk
h | Fk,h}k,h is
a martingale difference sequence for any k ∈[K] and h ∈[H]. Hence, by applying the Azuma-
Hoeffding inequality with probability at least 1 −δ/4, we have

K
X

k=1

H
X

h=1
˙ζk
h ≤2H
p

2KH log(4/δ) .
(96)

Combining all results of Eq. (91), (94), (95), and (96), we have the desired result.

K
X

k=1
(eV k
1 −V πk
1
)(sk
1) = e
O(dH3/2√

T + d3/2H3/2√

T + κ−1d3/2H2 + κ−1d2H2 + H
√

T)

= e
O(d3/2H3/2√

T + κ−1d2H2) .

In the following, we provide the proof of the lemmas used in Lemma 21.

61


---Page Break---
D.5.1
Proof of Lemma 22

Proof of Lemma 22. Note that
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

≤
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h

+
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h) −¯φs,a,s′(eθk+1
h
)

B−1
k,h

≤
X

s′∈Ss,a
Peθk+1
h
(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h

+
X

s′∈Ss,a


Peθk
h(s′ | s, a) −Peθk+1
h
(s′ | s, a)
 ¯φs,a,s′(eθk+1
h
)

B−1
k,h
|
{z
}
(i)

+
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h) −¯φs,a,s′(eθk+1
h
)

B−1
k,h
|
{z
}
(ii)

,

where the first inequality holds by triangle inequality.

For (i), we have

(i) =
X

s′∈Ss,a
∇Pϑk
h(s′ | s, a)⊤(eθk
h −eθk+1
h
)
¯φs,a,s′(eθk+1
h
)

B−1
k,h

≤
X

s′∈Ss,a
∥∇Pϑk
h(s′ | s, a)∥B−1
k,h

eθk
h −eθk+1
h

Bk,h

¯φs,a,s′(eθk+1
h
)

B−1
k,h
(97)

where in the equality we apply the mean value theorem with ϑk
h = v eθk
h + (1 −v)eθk+1
h
for some
v ∈[0, 1], and the inequality follows by Cauchy-Schwarz inequality. Meanwhile, since we have

Pϑk
h(s′ | s, a)

¯φs,a,s′(eθk+1
h
) −
X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)¯φs,a,s′′(eθk+1
h
)

(98)

= Pϑk
h(s′ | s, a)

 

φs,a,s′ −
X

es∈Ss,a
Peθk+1
h
(es | s, a)φs,a,es

−
X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)

"

φs,a,s′′ −
X

es
Peθk+1
h
(es | s, a)φs,a,es

# !

= Pϑk
h(s′ | s, a)φs,a,s′ −Pϑk
h(s′ | s, a)
X

es∈Ss,a
Peθk+1
h
(es | s, a)φs,a,es

−Pϑk
h(s′ | s, a)
X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)φs,a,s′′

+ Pϑk
h(s′ | s, a)

X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)

|
{z
}
1

 X

es
Peθk+1
h
(es | s, a)φs,a,es

= Pϑk
h(s′ | s, a)φs,a,s′ −Pϑk
h(s′ | s, a)
X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)φs,a,s′′

= ∇Pϑk
h(s′ | s, a) ,

62


---Page Break---
by substituting (98) into (97) we have

(i)

≤
X

s′∈Ss,a

nPϑk
h(s′ | s, a)¯φs,a,s′(eθk+1
h
)

−Pϑk
h(s′ | s, a)
X

s′′∈Ss,a
Pϑk
h(s′′ | s, a)¯φs,a,s′′(eθk+1
h
)

B−1
k,h

·
eθk
h −eθk+1
h

Bk,h

¯φs,a,s′(eθk+1
h
)

B−1
k,h

)

≤
X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

2

B−1
k,h

eθk
h −eθk+1
h

Bk,h

+
 X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h

2 eθk
h −eθk+1
h

Bk,h
.
(99)

Note that by Jensen’s inequality, we have

 X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h

2
=

Es′∼Pϑk
h(·|s,a)

¯φs,a,s′(eθk+1
h
)

B−1
k,h

2

≤Es′∼Pϑk
h(·|s,a)

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h



=
X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

2

B−1
k,h
.

(100)

Also, we introduce the following lemma:

Lemma 25. For any k ∈[K] and h ∈[H], the following holds:

eθk+1
h
−eθk
h

Bk,h
≤4ηLφ
√

λ
.

Then, substituting (100) into (99), we have

(i) ≤2
X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

2

B−1
k,h

eθk
h −eθk+1
h

Bk,h

≤8ηLφ
√

λ

X

s′∈Ss,a
Pϑk
h(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

2

B−1
k,h

≤8ηLφ
√

λ
max
s′∈Ss,a

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h
,
(101)

where the second inequality comes from Lemma 25, and the last inequality holds due to
P

s′∈Ss,a Pϑk
h(s′ | s, a) = 1.

63


---Page Break---
For (ii), we have

(ii) =
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h) −¯φs,a,s′(eθk+1
h
)

B−1
k,h

=
X

s′∈Ss,a
Peθk
h(s′ | s, a)
Ees∼P e
θk
h
(·|s,a)

φs,a,es

−Ees∼P e
θk+1
h
(·|s,a)

φs,a,es

B−1
k,h

=



X

es∈Ss,a


Peθk
h(es | s, a) −Peθk+1
h
(es | s, a)

φs,a,es


B−1
k,h

=



X

es∈Ss,a


Peθk
h(es | s, a) −Peθk+1
h
(es | s, a)
 
φs,a,es −Es′∼P e
θk+1
h
(·|s,a)

φs,a,s′

B−1
k,h

=



X

es∈Ss,a


Peθk
h(es | s, a) −Peθk+1
h
(es | s, a)

¯φs,a,es(eθk+1
h
)


B−1
k,h

≤8ηLφ
√

λ
max
s′∈Ss,a

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h
,
(102)

where the last inequality is obtained through the same argument as used to bound (i). Combining the
results of Eq. (101) and Eq. (102), we have
X

s′∈Ss,a
Peθk
h(s′ | s, a)
¯φs,a,s′(eθk
h)

B−1
k,h

≤
X

s′∈Ss,a
Peθk+1
h
(s′ | s, a)
¯φs,a,s′(eθk+1
h
)

B−1
k,h
+ 16ηLφ
√

λ
max
s′∈Ss,a

¯φs,a,s′(eθk+1
h
)

2

B−1
k,h

D.5.2
Proof of Lemma 23

Proof of Lemma 23. Note that

Bk+1,h = Bk,h +
X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)¯φk,h,s′(eθk+1
h
)¯φk,h,s′(eθk+1
h
)⊤

= Bk,h +
X

s′∈Sk,h
eφk,h,s′(eθk+1
h
)eφk,h,s′(eθk+1
h
)⊤,

where we define eφk,h,s′(eθk+1
h
) :=
q

Peθk+1
h
(s′ | sk
h, ak
h)¯φk,h,s′(eθk+1
h
). Then, we have

det(Bk+1,h) = det(Bk,h) det



Id + B
−1

2
k,h
X

s′∈Sk,h
eφk,h,s′(eθk+1
h
)eφk,h,s′(eθk+1
h
)⊤B
−1

2
k,h





= det(Bk,h)



1 +
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h





= det(λId)

K
Y

k=1



1 +
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h



.

Taking the logarithm on both sides yields

log det(Bk+1,h)

det(λId)
=

K
X

k=1
log



1 +
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h



.

64


---Page Break---
On the other hand, since λ ≥L2
φ,
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h
≤
X

s′∈Sk,h

1
λ

eφk,h,s′(eθk+1
h
)

2

2

=
X

s′∈Sk,h

1
λPeθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

2

2

≤L2
φ
λ

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)

≤1 ,

where the last inequality uses P

s′∈Sk,h Peθk+1
h
(s′ | sk
h, ak
h) = 1. From the fact that z ≤2 log(1 + z)

for any z ∈[0, 1], it follows that

K
X

k=1
log



1 +
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h



≥

K
X

k=1

1
2

X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h
.

Finally, we obtain

K
X

k=1

X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h
≤2

K
X

k=1
log



1 +
X

s′∈Sk,h

eφk,h,s′(eθk+1
h
)

2

B−1
k,h





= 2 log det(BK+1,h)

det(λId)

≤2d log

 

1 + KUL2
φ
dλ

!

,

where the last inequality follows by the determinant-trace inequality (Lemma 28).

D.5.3
Proof of Lemma 24

Proof of Lemma 24. Since ξ(m)
k,h ∼N(0, σ2
kB−1
k,h), by Lemma 30 for each m ∈[M], we have

∥ξ(m)
k,h ∥Bk,h ≤Cξσk
p

d log(Md/δ) .

Following the result of Lemma 22, we have
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)
¯φk,h,s′(eθk
h)

B−1
k,h
≤
X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+ 16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
.

Then, we obtain
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)¯φk,h,s′(eθk
h)⊤ξs′
k,h

≤
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)
¯φk,h,s′(eθk
h)

B−1
k,h
∥ξs′
k,h∥Bk,h

≤Cξσk
p

d log(Md/δ)
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)∥¯φk,h,s′(eθk
h)∥B−1
k,h

≤γk(δ)

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h

+ 16ηLφ
√

λ
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h


.

65


---Page Break---
D.5.4
Proof of Lemma 25

Proof of Lemma 25. We provide a proof for Lemma 25 since it is slight modification of Lemma 20
of [76]. From the definition, we know that

eθk+1
h
⊤
∇ℓk,h(eθk
h) + 1

2η

eθk+1
h
−eθk
h

2

eBk,h
≤

eθk
h
⊤
∇ℓk,h(eθk
h) .

By rearranging the terms, the following holds:
1
2η

eθk+1
h
−eθk
h

2

eBk,h
≤

eθk
h −eθk+1
h
⊤
∇ℓk,h(eθk
h)

≤
eθk
h −eθk+1
h
eBk,h

∇ℓk,h(eθk
h)
eB−1
k,h
Thus, we get
eθk+1
h
−eθk
h
eBk,h
≤2η
∇ℓk,h(eθk
h)
eB−1
k,h
.

Since Bk,h ⪯eBk,h and eB−1
k,h ⪯λ−1Id, we obtain
eθk+1
h
−eθk
h

Bk,h
≤
eθk+1
h
−eθk
h
eBk,h
≤2η
∇ℓk,h(eθk
h)
eB−1
k,h
≤2η
√

λ

∇ℓk,h(eθk
h)

2 ≤4ηLφ
√

λ
.

(103)

For the last inequality of (103), we provide the upper bound of l2-norm of ∇ℓk,h(θ). Since

ℓk,h(θ) = −
X

s′∈Sk,h
yk
h(s′) log Pθ(s′ | sk
h, ak
h) ,

the gradient of the loss function is given by

∇ℓk,h(θ) = −
X

s′∈Sk,h
yk
h(s′)



φs,a,s′ −
X

s′′∈Sk,h
Pθ(s′′ | sk
h, ak
h)φs,a,s′′





=
X

s′∈Sk,h
yk
h(s′)
X

s′′∈Sk,h
Pθ(s′′ | sk
h, ak
h)φs,a,s′′ −
X

s′∈Sk,h
yk
h(s′)φs,a,s′

=
X

s′′∈Sk,h
Pθ(s′′ | sk
h, ak
h)φs,a,s′′ −
X

s′∈Sk,h
yk
h(s′)φs,a,s′

=
X

s′∈Sk,h

 
Pθ(s′ | sk
h, ak
h) −yk
h(s′)

φs,a,s′ .

Therefore, we have

∥∇ℓk,h(θ)∥2 =



X

s′∈Sk,h

 
Pθ(s′ | sk
h, ak
h) −yk
h(s′)

φs,a,s′


2
≤
X

s′∈Sk,h

Pθ(s′ | sk
h, ak
h) −yk
h(s′)
 ∥φs,a,s′∥2

≤2Lφ
and this concludes the proof.

D.6
Bound on Pessimism Part

In this section, we provide the upper bound on the pessimism part of the regret: PK
k=1(V ∗
1 −eV k
1 )(sk
1).
Lemma 26 (Bound on pessimism). For any δ with 0 < δ < Φ(−1)/2, let σk = Hβk. If λ =
O(L2
φd log U) and we take multiple sample size M = ⌈1 −log(HU)

log Φ(1) ⌉, then with probability at least
1 −δ/2, we have

K
X

k=1
(V ∗
1 −V k
1 )(sk
1) = e
O

d3/2H3/2√

T + κ−1d2H2
.

66


---Page Break---
Proof of Lemma 26. As seen in Lemma 18, by using multiple sampling technique we show that
the optimistic randomized value function eV of ORRL-MNL is optimistic than the true optimal value
with constant probability Hence, with the same argument used in Lemma 11, we can show that the
pessimism term of ORRL-MNL is upper bounded by a bound of the estimation term times the inverse
probability of being optimistic, i.e.,

K
X

k=1

 
V ∗
1 −V k
1

(sk
1) ≤e
O

 
1
Φ(−1)

K
X

k=1


V k
1 −V πk
1

(sk
1)

!

.

D.7
Regret Bound of ORRL-MNL

Proof of Theorem 2. Since both Lemma 21 and Lemma 26 holds with probability at least 1 −δ/2
respectively, by taking the union bound we conclude the proof.

E
Optimistic Exploration Extension

In this section, we introduce UCRL-MNL+ (Algorithm 3), which is both computationally and statis-
tically efficient for MNL-MDPs with UCB-based exploration. The main difference compared to
ORRL-MNL is that UCRL-MNL+ constructs an optimistic value function that is greater than the optimal
value function with high probability. At each episode k ∈[K], with the estimated transition core
parameter eθk
h (5), for (s, a) ∈S × A, set ˆQk
H+1(s, a) = 0. For each h ∈[H],

ˆQk
h(s, a) := r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a) ˆV k
h+1(s′) + νopt
k,h(s, a) ,
(104)

where ˆV k
h (s) := min{maxa∈A ˆQk
h(s, a), H} and νopt
k,h(s, a) is the optimistic bonus term defined by

νopt
k,h(s, a) := Hβk
X

s′∈Ss,a
Peθk
h(s′ | s, a)∥¯φ(s, a, s′; eθk
h)∥B−1
k,h + 3Hβ2
k max
s′∈Ss,a ∥φ(s, a, s′)∥2
B−1
k,h .

Based on these optimistic value function ˆQk
h, at each episode the agent plays a greedy action with
respect to ˆQk
h as summarized in Algorithm 3.

Algorithm 3 UCRL-MNL+ (Upper Confidence RL for MNL-MDPs)

1: Inputs: Episodic MDP M, Feature map φ : S × A × S →Rd, Number of episodes K,
Regularization parameter λ, Confidence radius {βk}K
k=1, Step size η

2: Initialize: eθ1
h = 0d, B1,h = λId for all h ∈[H]
3: for episode k = 1, 2, · · · , K do

4:
Observe sk
1 and set
n
ˆQk
h(·, ·)
o

h∈[H] as described in (104)

5:
for horizon h = 1, 2, · · · , H do
6:
Select ak
h = argmaxa∈A ˆQk
h(sk
h, a) and observe sk
h+1
7:
Update eBk,h = Bk,h + η∇2ℓk,h(eθk
h) and eθk+1
h
as in (5)

8:
Update Bk+1,h = Bk,h + ∇2ℓk,h(eθk+1
h
)
9:
end for
10: end for

The main difference in regret analysis lies in ensuring the optimism of the estimated value function
ˆQk
h (Lemma 27). In the following statement (formal statement of Corollary 1), we provide a regret
guarantee for UCRL-MNL+, which enjoys the tightest regret bound for MNL-MDPs.

Theorem 3 (Regret Bound of UCRL-MNL+). Suppose that Assumption 1- 4 hold. For any δ ∈(0, 1),
if we set the input parameters in Algorithm 3 as λ = O(L2
φd log U), βk = O(
√

d log U log(kH))

67


---Page Break---
η = O(log U), then with probability at least 1 −δ, the cumulative regret of the UCRL-MNL+ policy π
is upper-bounded by

Regretπ(K) = e
O

dH3/2√

T + κ−1d2H2
,

where T = KH is the total number of time steps.

Proof of Theorem 3. By Lemma 17, suppose that the good event G(K, δ′) holds with probability at
least 1 −δ. Then, we show that the optimistic value function ˆQk
h is deterministically greater than the
true optimal value function as follows:

Lemma 27 (Optimism). Suppose that the event G∆
k,h(δ) holds for all k ∈[K] and h ∈[H]. Then
for any (s, a) ∈S × A, we have
Q∗
h(s, a) ≤ˆQk
h(s, a) .

Conditioned on G(K, δ′), by Lemma 27 we have

(V ∗
1 −V πk
1
)(sk
1) = Q∗
1(sk
1, π∗(sk
1)) −Qπk
1 (sk
1, ak
1)

≤ˆQk
1(sk
1, π∗(sk
1)) −Qπk
1 (sk
1, ak
1)

≤ˆQk
1(sk
1, ak
1) −Qπk
1 (sk
1, ak
1) = νopt
k,1 (sk
1, ak
1) + P1( ˆV k
2 −V πk
2
)(sk
1, ak
1) .

Note that

P1( ˆV k
2 −V πk
2
)(sk
1, ak
1) = Ees|sk
1,ak
1

h
( ˆV k
2 −V πk
2
)(es)
i
= ( ˆV k
2 −V πk
2
)(sk
2) + ˙ζk
1 ,

where we denote ζk
h := ( ˆV k
h+1 −V πk
h+1)(sk
h+1) −Ees|sk
h,ak
h

h
( ˆV k
h+1 −V πk
h+1)(es)
i
. Then, with the same
argument, we have

(V ∗
1 −V πk
1
)(sk
1) ≤

H
X

h=1
νopt
k,h(sk
h, ak
h) +

H
X

h=1
˙ζk
h .

By summing over all episodes, we have

Regretπ(K) ≤

K
X

k=1

H
X

h=1
νopt
k,h(sk
h, ak
h) +

K
X

k=1

H
X

h=1
˙ζk
h .
(105)

On the other hand, note that

K
X

k=1

H
X

h=1
νopt
k,h(sk
h, ak
h)

=

K
X

k=1

H
X

h=1
Hβk
X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)∥¯φk,h,s′(eθk
h)∥B−1
k,h +

K
X

k=1

H
X

h=1
3Hβ2
k max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h

≤HβK

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk
h(s′ | sk
h, ak
h)∥¯φk,h,s′(eθk
h)∥B−1
k,h

+ 3Hβ2
K

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h

≤HβK

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h
|
{z
}
(i)

+ 16ηLφ
√

λ
HβK

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
|
{z
}
(ii)

+ 3Hβ2
K

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h
|
{z
}
(iii)

,

68


---Page Break---
where the last inequality follows by Lemma 22.

Term (i) can be bounded as in Eq. (91):

HβK

K
X

k=1

H
X

h=1

X

s′∈Sk,h
Peθk+1
h
(s′ | sk
h, ak
h)
¯φk,h,s′(eθk+1
h
)

B−1
k,h
= e
O(dH3/2√

T) .
(106)

For term (ii), recall that as in Eq. (93) we have

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
≤16κ−1dH log

 

1 + KUL2
φ
dλ

!

.

Then, we have

16ηLφ
√

λ
HβK

K
X

k=1

H
X

h=1
max
s′∈Sk,h

¯φk,h,s′(eθk+1
h
)

2

B−1
k,h
= e
O(κ−1dH2) .
(107)

For term (iii), since we have

3Hβ2
K

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
B−1
k,h ≤3Hβ2
K

K
X

k=1

H
X

h=1
max
s′∈Sk,h ∥φk,h,s′∥2
A−1
k,h

≤12κ−1dH2β2
K log
 
1 + KUL2
φ/(dλ)


= e
O(κ−1d2H2) .
(108)

Combining the results of Eq. (106), (107), and (108), we have

K
X

k=1

H
X

h=1
νopt
k,h(sk
h, ak
h) = e
O(dH3/2√

T + κ−1d2H2) .

Finally, by Azuma-Hoeffiding inequality as in Eq. (96) we have

K
X

k=1

H
X

h=1
˙ζk
h = e
O(H
√

T) .

This concludes the proof.

In the following, we provide the proof of Lemma 27.

E.1
Optimism

Proof of Lemma 27. We prove this by backwards induction on h. For the base case h = H, since
V ∗
H+1(s) = ˆV k
H+1(s) = 0 for all s ∈S, we have

ˆQk
H(s, a) = r(s, a) = Q∗
H(s, a) .

69


---Page Break---
Suppose that the statement holds for h+1 where h ∈[H −1]. Then, for h and for any (s, a) ∈S ×A,

ˆQk
h(s, a)

= r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a) ˆV k
h+1(s′) + νopt
k,h(s, a)

≥r(s, a) +
X

s′∈Ss,a
Peθk
h(s′ | s, a)V ∗
h+1(s′) + νopt
k,h(s, a)

= r(s, a) +
X

s′∈Ss,a
Pθ∗
h(s′ | s, a)V ∗
h+1(s′)

+
X

s′∈Ss,a


Peθk
h(s′ | s, a) −Pθ∗
h(s′ | s, a)

V ∗
h+1(s′) + νopt
k,h(s, a)

≥r(s, a) +
X

s′∈Ss,a
Pθ∗
h(s′ | s, a)V ∗
h+1(s′)

= Q∗
h(s, a) ,

where the first inequality follows from the induction hypothesis and the second inequality holds by
Lemma 16.

F
Experiment Details

s1
s2
...
sn−1
sn

0.4

0.6

0.05

0.6

0.35

0.05

0.6

0.35

0.05

0.35

0.4
(1, r =
5
1000)

1
1
1
1

(0.6, r = 1)

Figure 2: The “RiverSwim” environment with n states [58]

The RiverSwim environment (Figure 2) consists of n states that are arranged in a chain. The agent
starts in the leftmost state with a relatively small reward of 0.005 and aims to reach the rightmost
state, which has a relatively large reward of 1. Choosing to swim to the left moves the agent
deterministically to the left, while swimming to the right has a probability of transitioning the agent
toward the right state, but also a high chance of remaining in the current state or even moving left due
to the strong current of river. Therefore, efficient exploration is crucial in order to learn the optimal
policy for this environment.

We fine-tuned the hyperparameters for each algorithm within specific ranges. Figures 1a and 1b show
the episodic returns in the RiverSwim environment over 10 independent runs with |S| = 4, H = 12,
and K = 10, 000 and |S| = 8, H = 24, and K = 10, 000, respectively. The shaded areas represent
the standard deviations (1-sigma error). Figure 1c compares the running time of the algorithms over
the first 1,000 episodes. All experiments were conducted on a Xeon(R) Gold 6226R CPU @ 2.90GHz
(16 cores).

G
Auxiliary Lemmas

Lemma 28 (Determinant-trace inequality [1]). Suppose x1, . . . , xt ∈Rd and for any 1 ≤τ ≤t,
∥xτ∥2 ≤L. Let Vt = λId + Pt
τ=1 xτx⊤
τ for some λ > 0. Then,

det(Vt) ≤(λ + tL2/d)d .

Lemma 29 (Freedman’s inequality [29]). Consider a real-valued martingale {Yk : k = 0, 1, 2, . . .}
with difference sequence {Xk : k = 0, 1, 2, 3, . . .}. Assume that the difference sequence is uniformly

70


---Page Break---
bounded, Xk ≤R almost surely for k = 1, 2, 3, . . .. Define the predictable quadratic variation
process of the martingale:

Wk :=

k
X

j=1
Ej−1[X2
j ]
for k = 1, 2, 3, . . . .

Then, for all t ≥0 and σ2 > 0,

P
 
∃k ≥0 : Yk ≥t and Wk ≤σ2
≤exp

−
−t2/2
σ2 + Rt/3


.

Lemma 30 (Gaussian noise concentration (Lemma D.2 in [37])). Let ξ(1), ξ(2), . . . , ξ(M) be M
independent d-dimensional multivariate normal distributed vector with mean 0d and covariance
σ2A−1 for some σ > 0 and a positive definite matrix A−1, i.e., ξ(m) ∼N(0d, σ2A−1) for m ∈[M].
Then for any δ ∈(0, 1), with probability at least 1 −δ, we have

max
m∈[M] ∥ξ(m)∥A ≤Cξσ
p

d log(Md/δ) := γ(δ) ,

where Cξ is an absolute constant.
Lemma 31 (Proposition 4.1 of 15). Let the wt+1 be the solution of the update rule
wt+1 = arg min
w∈V ηℓt(w) + Dψ(w, wt),

where V ⊆W ⊆Rd is a non-empty convex set and Dψ(w1, w2) = ψ(w1)−ψ(w2)−⟨∇ψ(w2), w1−
w2⟩is the Bregman Divergence w.r.t. a strictly convex and continuously differentiable function
ψ : W →R. Further supposing ψ(w) is 1-strongly convex w.r.t. a certain norm ∥· ∥in W, then
there exists a gt ∈∂ℓt(wt+1) such that
⟨ηtg′
t, wt+1 −u⟩≤⟨∇ψ(wt) −∇ψ(wt+1), wt+1 −u⟩
for any u ∈W.
Lemma 32. Let {Ft}∞
t=1 be a filtration. Let {zt}∞
t=1 be a stochastic process in B2(U) = {z ∈RU |
∥z∥∞≤1} such that zt is Ft measurable. Let {εt}∞
t=1 be a martingale difference sequence such
that εt ∈RU is Ft+1 measurable. Furthermore, assume that conditional on Ft, we have ∥εt∥1 ≤2
almost surely, and denote by Σt = E[εtε⊤
t |Ft]. Let λ > 0 and for any t ≥1 define

Ut =

t−1
X

i=1
⟨εi, zi⟩
and
Bt = λ +

t−1
X

i=1
∥zi∥2
Σi,

Then, for any δ ∈(0, 1], we have

Pr

"

∃t ≥1, Ut ≥
p

Bt

 √

λ
4
+
4
√

λ
log

 r

Bt

λ

!

+
4
√

λ
log
2

δ

!#

≤δ.

Lemma 33 (Lemma 1 of 76). Let ℓ(z, y) = PK
k=0 1{y = k} · log

1
[σ(z)]k


, a ∈[−C, C]K,

y ∈{0} ∪[K] and b ∈RK where C > 0. Then, we have

ℓ(a, y) ≥ℓ(b, y) + ∇ℓ(b, y)⊤(a −b) +
1
log(K + 1) + 2(C + 1)(a −b)⊤∇2ℓ(b, y)(a −b).

Lemma 34 (Lemma 17 of 76). Let ℓ(z, y) = PK
k=0 1{y = k} · log

1
[σ(z)]k


and z ∈RK be

a K-dimensional vector. Define zµ ≜σ+ (smoothµ(σ(z))), where smoothµ(p) = (1 −µ)p +
µ1/(K + 1). Then, for µ ∈[0, 1/2], we have
ℓ(zµ, y) −ℓ(z, y) ≤2µ
for any y ∈{0} ∪[K]. We also have ∥zµ∥∞≤log(K/µ).

Lemma 35 (Lemma 18 of 76). Let Li,h(θ) := ℓi,h(θ) + 1

2c∥θ −θi
h∥2
Bi,h. Assume that ℓi,h is a
√

N-self-concordant-like function. Then, for any θ, θi
h ∈B(0d, 1), the quadratic approximation
eLi,h(θ) = Li,h(eθi+1
h
) + ⟨∇Li,h(eθi+1
h
), θ −eθi+1
h
⟩+ 1

2c
θ −eθi+1
h

2

Bi,h
satisfies

Li,h(θ) ≤eLi,h(θ) + exp

N
θ −eθi+1
h

2

2

 θ −eθi+1
h

2

∇ℓi,h(eθi+1
h
) .

71


---Page Break---
H
Limitations

We make an assumption about the transition model of MDPs by using the MNL model, which is
a specific parametric model. This assumption implies that we assume the realizability of the MNL
model. It’s worth noting that the realizability assumption has also been commonly made in previous
literature on provable reinforcement learning with function approximation, including works such as
[72, 43, 73, 53, 22, 14, 9, 68, 70, 33, 81, 82, 37, 35]. However, we hope that this condition can be
relaxed in the future work.

72


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract is to propose provably efficient random-
ized algorithms for MNL-MDPs. In Section 1 (Introduction), we provide the motivation and
main contributions of this paper.

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

Justification: We discuss the limitation of this work in Appendix H.

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

73


---Page Break---
Answer: [Yes]
Justification: We provide the full set of assumptions in Section 2.2 and a complete proof of
main results in Appendix C and D.
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
Justification: We provide numerical experiments that support our main claims in Section 5
and the detailed information of experiments in Appendix F.
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

74


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We have attached the data and code with sufficient instructions to reproduce
the main experimental results in the supplementary material.
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
Justification: We provide the detailed explanation for the experimental setting in Appendix F.
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
Justification: We report error bars (standard deviation) in our numerical experiment results
shown in Section 5.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

75


---Page Break---
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

Justification: We provide the detailed information on the computer resources used to conduct
numerical experiments in Appendix F.
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

Justification: The research conducted in this paper adheres to the NeurIPS Code of Ethics in
all aspects.
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
Justification: There is no negative societal impacts of the work performed because this
research focuses on theoretical aspects.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

76


---Page Break---
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
Justification: The research conducted in this paper does not pose any such risks.
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
Justification: This paper does not use any external assets such as code, data, or models.
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

77


---Page Break---
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper does not release new assets.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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

78


---Page Break---
