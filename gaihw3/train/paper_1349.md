Differentially Private Reinforcement Learning with
Self-Play

Dan Qiao
Department of Computer Science & Engineering
University of California, San Diego
San Diego, CA 92093
d2qiao@ucsd.edu

Yu-Xiang Wang
Halıcıo˘glu Data Science Institute
University of California, San Diego
San Diego, CA 92093
yuxiangw@ucsd.edu

Abstract

We study the problem of multi-agent reinforcement learning (multi-agent RL) with
differential privacy (DP) constraints. This is well-motivated by various real-world
applications involving sensitive data, where it is critical to protect users’ private
information. We first extend the definitions of Joint DP (JDP) and Local DP
(LDP) to two-player zero-sum episodic Markov Games, where both definitions
ensure trajectory-wise privacy protection. Then we design a provably efficient
algorithm based on optimistic Nash value iteration and privatization of Bernstein-
type bonuses. The algorithm is able to satisfy JDP and LDP requirements when
instantiated with appropriate privacy mechanisms. Furthermore, for both notions of
DP, our regret bound generalizes the best known result under the single-agent RL
case, while our regret could also reduce to the best known result for multi-agent RL
without privacy constraints. To the best of our knowledge, these are the first results
towards understanding trajectory-wise privacy protection in multi-agent RL.

1
Introduction

This paper considers the problem of multi-agent reinforcement learning (multi-agent RL), wherein
several agents simultaneously make decisions in an unfamiliar environment with the goal of maximiz-
ing their individual cumulative rewards. Multi-agent RL has been deployed not only in large-scale
strategy games like Go [Silver et al., 2017], Poker [Brown and Sandholm, 2019] and MOBA games
[Ye et al., 2020], but also in various real-world applications such as autonomous driving [Shalev-
Shwartz et al., 2016], negotiation [Bachrach et al., 2020], and trading in financial markets [Shavandi
and Khedmati, 2022]. In these applications, the learning agent analyzes users’ private feedback in
order to refine its performance, where the data from users usually contain sensitive information. Take
autonomous driving as an instance, here a trajectory describes the interaction between the cars in a
neighborhood during a fixed time window. At each timestamp, given the current situation of each
car, the system (central agent) will send a command for each car to take (e.g. speed up, pull over),
and finally the system gathers the feedback from each car (e.g. whether the driving is safe, whether
the customer feels comfortable) and enhances its policy. Here, (situation, command, feedback)
corresponds to (state, action, reward) in a Markov Game where the state and reward of each user
are considered as sensitive information. Therefore, leakage of such information is not acceptable.
Regrettably, it has been demonstrated that without the implementation of privacy safeguards, learning
agents tend to inadvertently memorize details from individual training data points [Carlini et al.,
2019], regardless of their relevance to the learning process [Brown et al., 2021]. This susceptibility
exposes multi-agent RL agents to potential privacy threats.

To handle the above privacy issue, Differential privacy (DP) [Dwork et al., 2006] has been widely
considered. The output of a differentially private reinforcement learning algorithm cannot be discerned

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Algorithms for Markov Games
Regret without privacy
Regret under ϵ-JDP
Regret under ϵ-LDP
DP-Nash-VI (Our Algorithm 1)
eO(
√

H2SABT)
eO(
√

H2SABT + H3S2AB/ϵ)
eO(
√

H2SABT + S2AB
√

H5T/ϵ)
Nash VI [Liu et al., 2021]
eO(
√

H2SABT)∗
N.A.
N.A.

Lower bounds
Ω(
p

H2S(A + B)T) [Bai and Jin, 2020]
eΩ
p

H2S(A + B)T + HS(A+B)

ϵ


eΩ
p

H2S(A + B)T +
√

HS(A+B)T

ϵ



Algorithms for MDPs (B = 1)
Regret without privacy
Regret under ϵ-JDP
Regret under ϵ-LDP
PUCB [Vietri et al., 2020]
eO(
√

H3S2AT)
eO(
√

H3S2AT + H3S2A/ϵ)⋆
N.A.
LDP-OBI [Garcelon et al., 2021]
eO(
√

H3S2AT)
N.A.
eO(
√

H3S2AT + S2A
√

H5T/ϵ)†

Private-UCB-VI [Chowdhury and Zhou, 2022]
eO(
√

H3SAT)
eO(
√

H3SAT + H3S2A/ϵ)
eO(
√

H3SAT + S2A
√

H5T/ϵ)
DP-UCBVI ‡ [Qiao and Wang, 2023]
eO(
√

H2SAT)
eO(
√

H2SAT + H3S2A/ϵ)
eO(
√

H2SAT + S2A
√

H5T/ϵ)

Table 1: Comparison of our results (in blue) to existing work regarding regret without privacy (i.e.
the privacy budget is infinity), regret under ϵ-Joint DP and regret under ϵ-Local DP. In the above, S is
the number of states, A, B are the number of actions for both players, H is the planning horizon and
K is the number of episodes (T = HK is the number of steps). Markov decision processes (MDPs)
is a special case of Markov Games where B = 1. ∗: This result is the best known regret bound when
there is no privacy concern. ⋆: More discussions about this bound can be found in Chowdhury and
Zhou [2022]. †: The original regret bound in Garcelon et al. [2021] is derived under the setting of
stationary MDP, and can be directly transferred to the bound here by adding
√

H to the first term. ‡:
This algorithm achieved the best known results under single-agent MDPs, and our Algorithm 1 can
obtain the same regret bounds under this setting.

from its output in an alternative reality where any specific user is substituted, which effectively
mitigates the privacy risks mentioned earlier. However, it is shown [Shariff and Sheffet, 2018] that
standard DP will lead to linear regret even under contextual bandits. Therefore, Vietri et al. [2020]
considered a relaxed surrogate of DP: Joint Differential Privacy (JDP) [Kearns et al., 2014] for RL.
Briefly speaking, JDP protects the information about any specific user even given the output of all
other users. Meanwhile, another variant of DP: Local Differential Privacy (LDP) [Duchi et al., 2013]
has also been extended to RL by Garcelon et al. [2021] due to its stronger privacy protection. LDP
requires that the raw data of each user is privatized before being sent to the agent. Although following
works [Chowdhury and Zhou, 2022, Qiao and Wang, 2023] established near optimal results under
these two notions of DP, all of the previous works focused on the single-agent RL setting while the
solution to multi-agent RL with differential privacy is still unknown. Therefore we question:
Question 1.1. Is it possible to design a provably efficient self-play algorithm to solve Markov games
while satisfying the constraints of differential privacy?

Our contributions. In this paper, we answer the above question affirmatively by proposing a general
algorithm for DP multi-agent RL: DP-Nash-VI (Algorithm 1). Our contributions are threefold.

• We first extend the definitions of Joint DP (Definition 2.2) and Local DP (Definition 2.3) to
the multi-agent RL setting. Both notions of DP focus on protecting the sensitive information
of each trajectory, which is consistent with the counterparts under single-agent RL.
• We design a new algorithm DP-Nash-VI (Algorithm 1) based on optimistic Nash value
iteration and privatization of Bernstein-type bonuses. The algorithm can be combined with
any Privatizer (for JDP or LDP) that possesses a corresponding regret bound (Theorem 4.1).
Moreover, when there is no privacy constraint (i.e. the privacy budget is infinity), our regret
reduces to the best known regret for non-private multi-agent RL.

• Under the constraint of ϵ-JDP, DP-Nash-VI achieves a regret of eO(
√

H2SABT +
H3S2AB/ϵ) (Theorem 5.2). Compared to the regret lower bound (Theorem 5.3), the
main term is nearly optimal while the additional cost due to JDP has optimal dependence
on ϵ. Under the ϵ-LDP constraint, DP-Nash-VI achieves a regret of eO(
√

H2SABT +
S2AB
√

H5T/ϵ) (Theorem 5.5), where the dependence on K, ϵ is optimal according to the
lower bound (Theorem 5.6). The pair of results strictly generalizes the best known results
for single-agent RL with DP [Qiao and Wang, 2023].

1.1
Related work

We compare our results with existing works on differentially private reinforcement learning [Vietri
et al., 2020, Garcelon et al., 2021, Chowdhury and Zhou, 2022, Qiao and Wang, 2023] and regret
minimization under Markov Games [Liu et al., 2021] in Table 1, while more discussions about
differentially private learning algorithms are deferred to Appendix A. Notably, all existing DP RL

2


---Page Break---
algorithms focus on the single-agent case. In comparison, our algorithm works for the more general
two-player setting and our results directly match the best known regret bounds [Qiao and Wang,
2023] when applied to the single-agent setting.

Recently, several works provide non-asymptotic theoretical guarantees for learning Markov Games.
Bai and Jin [2020] developed the first provably-efficient algorithms in MGs based on optimistic value
iteration, and the result is improved by Liu et al. [2021] using model-based approach. Meanwhile,
model-free approaches are shown to break the curse of multiagency and improve the dependence on
action space [Bai et al., 2020, Jin et al., 2021, Mao et al., 2022, Wang et al., 2023, Cui et al., 2023].
However, all these algorithms base on the original data from users, and thus are vulnerable to various
privacy attacks. While several works [Hossain and Lee, 2023, Hossain et al., 2023, Zhao et al., 2023b,
Gohari et al., 2023] study the privatization of communications between multiple agents, none of them
provide regret guarantees. In comparison, we design algorithms that provably protect the sensitive
information in each trajectory, while achieving near-optimal regret bounds simultaneously.

Technically speaking, we follow the idea of optimistic Nash value iteration and privatization of
Bernstein-type bonuses. Optimistic Nash value iteration aims to construct both upper bounds and
lower bounds for value functions, which could guide the exploration. Such idea has been applied by
previous model-based approaches [Bai and Jin, 2020, Liu et al., 2021] to derive tight regret bounds.
To satisfy the privacy guarantees, we are required to construct the UCB and LCB privately. In
this work, we privatize the transition kernel estimate and construct a private bonus function for our
purpose. Among different bonuses, we generalize the approach in Qiao and Wang [2023] and directly
operate on the Bernstein-type bonus, which could enable tight regret analysis while the privatization is
more technically demanding due to the variance term. To handle this, we first privatize the visitation
counts such that they satisfy several nice properties, then we use these counts to construct private
transition estimates and private bonuses. Lastly, we manage to prove UCB and LCB, and bound the
private terms by their non-private counterparts to complete the regret analysis.

2
Problem Setup

We consider reinforcement learning under Markov Games (MGs) [Shapley, 1953] with Differential
Privacy (DP) [Dwork et al., 2006]. Below we introduce MGs and define DP under multi-agent RL.

2.1
Markov Games and Regret

Markov Games (MGs) are the generalization of Markov Decision Processes (MDPs) to the multi-
player setting, where each player aims to maximize her own reward. We consider two-player zero-sum
episodic MGs, denoted by a tuple MG = (S, A, B, H, {Ph}H
h=1, {rh}H
h=1, s1), where S is the state
space with S = |S|, A and B are the action space for the max-player (who aims to maximize
the total reward) and the min-player (who aims to minimize the total reward) respectively with
A = |A|, B = |B|. Besides, H is the horizon while the non-stationary transition kernel Ph(·|s, a, b)
gives the distribution of the next state if action (a, b) is taken at state s and time step h. In addition,
we assume that the reward function rh(s, a, b) ∈[0, 1] is deterministic and known1. For simplicity,
we assume each episode starts from a fixed initial state s1. Then at each time step h ∈[H], two
players observe sh and choose their actions ah ∈A and bh ∈B simultaneously, after which both
players observe the action of their opponent and receive reward rh(sh, ah, bh), the environment will
transit to sh+1 ∼Ph(·|sh, ah, bh).

Markov policy, value function. A Markov policy µ of the max-player can be seen as a series
of mappings µ = {µh}H
h=1, where each µh maps each state s ∈S to a probability distribu-
tion over actions A, i.e. µh : S →∆(A). A Markov policy ν for the min-player is defined
similarly. Given a pair of policies (µ, ν) and time step h ∈[H], the value function V µ,ν
h
(·) is
defined as V µ,ν
h
(s) = Eµ,ν[PH
t=h rt|sh = s] while the Q-value function Qµ,ν
h (·, ·, ·) is defined as
Qµ,ν
h (s, a, b) = Eµ,ν[PH
t=h rt|sh, ah, bh = s, a, b] for all s, a, b. According to the definitions, the
following Bellman equation holds:

Qµ,ν
h (s, a, b) = [rh + PhV µ,ν
h+1](s, a, b), V µ,ν
h
(s) = [Eµ,νQµ,ν
h ](s), ∀(h, s, a, b).

1This assumption is wlog since the uncertainty of reward is dominated by that of transition kernel.

3


---Page Break---
Best responses, Nash equilibrium. For any policy µ of the max-player, there exists a best response
policy ν†(µ) of the min-player such that V µ,ν†(µ)
h
(s) = infν V µ,ν
h
(s) for all (s, h). For simplicity,

we denote V µ,†
h
:= V µ,ν†(µ)
h
. Also, µ†(ν) and V †,ν
h
can be defined by symmetry. It is shown [Filar
and Vrieze, 2012] that there exists a pair of policies (µ⋆, ν⋆) that are best responses against each
other, i.e., V µ⋆,†
h
(s) = V µ⋆,ν⋆

h
(s) = V †,ν⋆

h
(s), ∀(s, h) ∈S × [H]. The pair of policies (µ⋆, ν⋆)
is called the Nash equilibrium of the Markov game, which further satisfies the following minimax
property: for all (s, h) ∈S ×[H], supµ infν V µ,ν
h
(s) = V µ⋆,ν⋆

h
(s) = infν supµ V µ,ν
h
(s). The value

functions of (µ⋆, ν⋆) are called Nash value functions and we denote V ⋆
h = V µ⋆,ν⋆

h
, Q⋆
h = Qµ⋆,ν⋆

h
for
simplicity. Nash equilibrium means that no player could gain more from updating her own policy.

Learning objective: regret. Following previous works [Bai and Jin, 2020, Liu et al., 2021], we aim
to minimize the regret, which is defined as below:

Regret(K) =

K
X

k=1

h
V †,νk
1
(s1) −V µk,†
1
(s1)
i
,

where K is the number of episodes the agent interacts with the environment and (µk, νk) are the
policies executed by the agent in the k-th episode. Note that any sub-linear regret bound can be
transferred to a PAC guarantee according to the standard online-to-batch conversion [Jin et al., 2018].

2.2
Differential Privacy in Multi-agent RL

For RL with self-play, each trajectory corresponds to the interaction between a pair of users and the
environment. The interaction generally follows the protocol below. At time step h of the k-th episode,
the users send their state sk
h to a central agent M, then M sends back a pair of actions (ak
h, bk
h) for the
users to take, and finally the users send their reward rk
h to M. Following previous works [Vietri et al.,
2020, Chowdhury and Zhou, 2022, Qiao and Wang, 2023], here we let U = (u1, · · · , uK) denote the
sequence of K unique 2 pairs of users who participate in the above RL protocol. Besides, each pair of
users uk is characterized by the {sk
h, rk
h}H
h=1 information they would respond to all (AB)H3 possible
sequences of actions from the agent. Let M(U) = {(ak
h, bk
h)}H,K
h,k=1,1 denote the whole sequence of
actions suggested by the agent M. Then a direct adaptation of differential privacy [Dwork et al.,
2006] is defined below, which says that M(U) and all other pairs excluding uk together will not
disclose much information about user uk.
Definition 2.1 (Differential Privacy (DP)). For any ϵ > 0 and δ ∈[0, 1], a mechanism M : U →
(A × B)KH is (ϵ, δ)-differentially private if for any possible user sequences U and U′ that is different
on one pair of users and any subset E of (A × B)KH,

P[M(U) ∈E] ≤eϵ · P[M(U′) ∈E] + δ.

If δ = 0, we say that M is ϵ-differentially private (ϵ-DP).

Unfortunately, privately recommending actions to the pair of users uk while protecting their own
state and reward information is shown to be impractical even for the single-player setting. Therefore,
we consider a relaxed version of DP, known as Joint Differential Privacy (JDP) [Kearns et al., 2014].
JDP says that for all pairs of users uk, the recommendation to all other pairs excluding uk will not
disclose the sensitive information about uk. Although being weaker than DP, JDP could still provide
meaningful privacy protection by ensuring that even if an adversary can observe the interactions
between all other users and the environment, it is statistically hard to reconstruct the interaction
between uk and the environment. JDP is first studied by Vietri et al. [2020] under single-agent
reinforcement learning, and we extend the definition to the two-player setting.
Definition 2.2 (Joint Differential Privacy (JDP)). For any ϵ > 0, a mechanism M : U →(A×B)KH
is ϵ-joint differentially private if for any k ∈[K], any user sequences U and U′ that is different on
the k-th pair of users and any subset E of (A × B)(K−1)H,

P[M−k(U) ∈E] ≤eϵ · P[M−k(U′) ∈E],

2Uniqueness is assumed wlog, as for a returning user pair one can group them with their previous occurrences.
3At each time step h ∈[H], the agent suggests actions to both players, and thus there are AB possibilities
for each time step h.

4


---Page Break---
where M−k(U) ∈E means the sequence of actions sent to all pairs of users excluding uk belongs
to set E.

In the example of autonomous driving, JDP ensures that even if an adversary observes the interactions
between cars within all time windows except one, it is hard to know what happens during the
specific time window. While providing strong privacy protection, JDP requires the central agent
M to have access to the real trajectories from users. However, in various scenarios the users are
not even willing to directly share their data with the agent. To address such circumstances, Duchi
et al. [2013] developed a stronger notion of privacy named Local Differential Privacy (LDP). Now
that when considering LDP, the agent can not observe the state of users, we consider the following
protocol specific for LDP: at the beginning of the k-th episode, the agent M first sends a policy pair
πk = (µk, νk) to the pair of users uk, after running πk and getting a trajectory Xk, uk privatizes
their trajectory to X′
k and sends it back to M. We present the definition of Local DP below, which
generalizes the LDP under single-agent reinforcement learning by Garcelon et al. [2021]. Briefly
speaking, Local DP ensures that it is impractical for an adversary to reconstruct the whole trajectory
of uk even if observing their whole response.

Definition 2.3 (Local Differential Privacy (LDP)). For any ϵ > 0, a mechanism f
M is ϵ-
local differentially private if for any possible trajectories X, X′ and any possible set E ⊆
{ f
M(X)|X is any possible trajectory},

P[ f
M(X) ∈E] ≤eϵ · P[ f
M(X′) ∈E].

In the example of autonomous driving, LDP ensures that the system can only observe a private
version of the interactions between cars instead of the raw data.
Remark 2.4. Note that here our definitions of JDP and LDP both provide trajectory-wise privacy
protection, which is consistent with previous works [Chowdhury and Zhou, 2022, Qiao and Wang,
2023]. Moreover, under the special case where the min-player plays a fixed and known deterministic
policy (or equivalently, B only contains a single action and B = 1), the Markov Game setting
reduces to a single-agent Markov decision process while our JDP and LDP directly matches previous
definitions for the MDP setting. Therefore, our setting strictly generalizes previous works and requires
novel techniques to handle the min-player.
Remark 2.5. In the following sections we will show that LDP is consistent with sub-linear regret
bounds, while it is known that we can not derive sub-linear regret bounds under the constraint of DP.
We remark that there is no contradictory since here the RL protocols for DP and LDP are different.
As a result, here a guarantee of LDP does not directly imply a guarantee of DP and the two notions
are indeed not directly comparable.

3
Algorithm

In this part, we introduce DP-Nash-VI (Algorithm 1). Note that the algorithm takes Privatizer as an
input. We analyze the regret of Algorithm 1 for all Privatizers satisfying the Assumption 3.1 below,
which includes the cases where the Privatizer is chosen as Central (for JDP) or Local (for LDP).

We first introduce the definition of visitation counts, where N k
h(s, a, b) = Pk−1
i=1 1(si
h, ai
h, bi
h =
s, a, b) denotes the visitation count of (s, a, b) at time step h until the beginning of the k-th episode.
Similarly, we let N k
h(s, a, b, s′) = Pk−1
i=1 1(si
h, ai
h, bi
h, si
h+1 = s, a, b, s′) be the visitation count of
(h, s, a, b, s′) before the k-th episode. In multi-agent RL without privacy constraints, such visitation
counts are sufficient for estimating the transition kernel {Ph}H
h=1 and updating the exploration policy,
as in previous model-based approaches [Liu et al., 2021]. However, these counts base on the original
trajectories from the users, which could reveal sensitive information. Therefore, with the concern of
privacy, we can only incorporate these counts after a privacy-preserving step. In other words, we use
a Privatizer to transfer the original counts to the private version e
N k
h(s, a, b), e
N k
h(s, a, b, s′). We make
the following Assumption 3.1 for Privatizer, which says that the private counts are close to real ones.
Privatizers for JDP and LDP that satisfy Assumption 3.1 will be proposed in Section 5.
Assumption 3.1 (Private counts). For any privacy budget ϵ > 0 and failure probability β ∈[0, 1],
there exists some Eϵ,β > 0 such that with probability at least 1 −β/3, for all (h, s, a, b, s′, k) ∈
[H] × S × A × B × S × [K], the e
N k
h(s, a, b, s′) and e
N k
h(s, a, b) from Privatizer satisfies:

5


---Page Break---
Algorithm 1 Differentially Private Optimistic Nash Value Iteration (DP-Nash-VI)

1: Input: Number of episodes K, privacy budget ϵ, failure probability β and a Privatizer (can be
either Central or Local).
2: Initialize: Private counts e
N 1
h(s, a, b) = e
N 1
h(s, a, b, s′) = 0 for all (h, s, a, b, s′). Set up the
confidence bound Eϵ,β w.r.t the Privatizer, the minimal gap ∆= H and universal constants
C1, C2 > 0. ι = log(30HSABK/β).
3: for k = 1, 2, · · · , K do

4:
V
k
H+1(·) = V k
H+1(·) = 0.
5:
for h = H, H −1, · · · , 1 do
6:
for (s, a, b) ∈S × A × B do
7:
Compute private transition kernel eP k
h (·|s, a, b) as in (1).

8:
Compute γk
h(s, a, b) = C1

H · eP k
h (V
k
h+1 −V k
h+1)(s, a, b).

9:
Compute Γk
h(s, a, b) = C2

v
u
u
t Var e
P k
h (·|s,a,b)

" 

V k
h+1+V k
h+1
2

!

(·)

#

·ι

e
Nk
h(s,a,b)
+ C2HSEϵ,β·ι

e
Nk
h(s,a,b) + C2H2Sι

e
N k
h(s,a,b).

10:
UCB Q
k
h(s, a, b) = min{P

s′ eP k
h (s′|s, a, b) · V
k
h+1(s′) + [rh + γk
h + Γk
h](s, a, b), H}.

11:
LCB Qk
h(s, a, b) = max{P

s′ eP k
h (s′|s, a, b) · V k
h+1(s′) + [rh −γk
h −Γk
h](s, a, b), 0}.
12:
end for
13:
for s ∈S do
14:
Compute the policy πk
h(·, ·|s) = CCE(Q
k
h(s, ·, ·), Qk
h(s, ·, ·)).

15:
Compute the value functions V
k
h(s) = Eπk
hQ
k
h(s), V k
h(s) = Eπk
hQk
h(s).
16:
end for
17:
end for
18:
Deploy policy πk = (πk
1, · · · , πk
H) and get trajectory (sk
1, ak
1, bk
1, rk
1, · · · , sk
H+1).

19:
Update the private counts to e
N k+1 via Privatizer.

20:
if (V
k
1 −V k
1)(s1) < ∆then

21:
∆= (V
k
1 −V k
1)(s1) and πout = πk = (πk
1, · · · , πk
H).
22:
end if
23: end for
24: Return: The marginal policies of πout: (µout, νout).

(1) | e
N k
h(s, a, b, s′)−N k
h(s, a, b, s′)| ≤Eϵ,β, | e
N k
h(s, a, b)−N k
h(s, a, b)| ≤Eϵ,β. e
N k
h(s, a, b, s′) > 0.
(2) e
N k
h(s, a, b) = P

s′∈S e
N k
h(s, a, b, s′) ≥N k
h(s, a, b).

Given the private counts satisfying Assumption 3.1, the private estimate of transition kernel is defined
as below.

eP k
h (s′|s, a, b) =
e
N k
h(s, a, b, s′)

e
N k
h(s, a, b)
, ∀(h, s, a, b, s′, k).
(1)

Remark 3.2. Assumption 3.1 is a generalization of Assumption 3.1 of Qiao and Wang [2023] to the
two-player setting. The assumption (2) guarantees that the private transition kernel eP k
h (·|s, a, b) is a
valid probability distribution, which enables our usage of Bernstein-type bonus. Besides, eP is close
to the empirical transition kernel based on original visitation counts according to Assumption (1).

Algorithmic design. Following previous non-private approaches [Liu et al., 2021], DP-Nash-VI
(Algorithm 1) maintains a pair of value functions Q and Q which are the upper bound and lower
bound of the Q value of the current policy when facing best responses (with high probability). More
specifically, we use private visitation counts e
N k
h to construct a private estimate of transition kernel
eP k
h (line 7) and a pair of private bonus γk
h (line 8) and Γk
h (line 9). Intuitively, the first term of Γk
h is
derived from Bernstein’s inequality while the second term is the additional bonus due to differential
privacy. Next we do value iteration with bonuses to construct the UCB function Q
k
h (line 10) and the
LCB function Qk
h (line 11). The policy πk for the k-th episode is calculated using the CCE function
(discussed below) and we run πk to collect a trajectory (line 14,18). Finally, the Privatizer transfers the

6


---Page Break---
non-private counts to private ones for the next episode (line 19). The output policy πout is chosen as
the policy πk with minimal gap (V
k
1 −V k
1)(s1) (line 21). Decomposing the output policy, the output
policy (µout, νout) for both players are the marginal policies of πout, i.e. µout
h (·|s) = P

b∈B πout
h (·, b|s)
and νout
h (·|s) = P

a∈A πout
h (a, ·|s) for all (h, s) ∈[H] × S.

Coarse Correlated Equilibrium (CCE). Intuitively speaking, CCE of a Markov Game is a potentially
correlated policy where no player could benefit from unilateral unconditional deviation. As a
computationally friendly relaxation of Nash Equilibrium, CCE has been applied by previous works
[Xie et al., 2020, Liu et al., 2021] to design efficient algorithms. Formally, for any two functions
Q(·, ·), Q(·, ·) : A × B →[0, H], CCE(Q, Q) returns a policy π ∈∆(A × B) such that

E(a,b)∼πQ(a, b) ≥max
a′ E(a,b)∼πQ(a′, b), E(a,b)∼πQ(a, b) ≤min
b′ E(a,b)∼πQ(a, b′).

Since Nash Equilibrium (NE) is a special case of CCE and a NE always exists, a CCE always exists.
Moreover, a CCE can be derived in polynomial time via linear programming. Note that the policies
given by CCE can be correlated for the two players, therefore deploying such policy requires the
cooperation of both players (line 18).

4
Main results

We first state the regret analysis of DP-Nash-VI (Algorithm 1) based on Assumption 3.1, which can
be combined with any Privatizers. The proof of Theorem 4.1 is sketched in Appendix B with details
in the Appendix. Note that (µk, νk) denote the marginal policies of πk for both players.
Theorem 4.1. For any privacy budget ϵ > 0, failure probability β ∈[0, 1] and any Privatizer
satisfying Assumption 3.1, with probability at least 1 −β, the regret of DP-Nash-VI (Algorithm 1) is

Regret(K) =

K
X

k=1

h
V †,νk
1
(s1) −V µk,†
1
(s1)
i
≤eO
√

H2SABT + H2S2ABEϵ,β

,
(2)

where K is the number of episodes and T = HK.

Under the special case where the privacy budget ϵ →∞(i.e. there is no privacy concern), plugging
Eϵ,β = 0 in Theorem 4.1 will imply a regret bound of eO(
√

H2SABT). Such result directly
matches the best known result for regret minimization without privacy constraints [Liu et al., 2021]
and nearly matches the lower bound of Ω(
p

H2S(A + B)T) [Bai and Jin, 2020]. Furthermore,
under the special case of single-agent MDP (where B = 1), our result reduces to Regret(K) ≤
eO(
√

H2SAT + H2S2AEϵ,β). Such result matches the best known result under the same set of
conditions (Theorem 4.1 of Qiao and Wang [2023]). Therefore, Theorem 4.1 is a generalization of
the best known results under MARL [Liu et al., 2021] and Differentially Private (single-agent) RL
[Qiao and Wang, 2023] simultaneously.

PAC guarantee. Recall that we output a policy πout whose marginal policies are (µout, νout). We
highlight that the output policy for each player is a single Markov policy that is convenient to store
and deploy. Moreover, as a corollary of the regret bound, we give a PAC bound for the output policy.
Theorem 4.2. For any privacy budget ϵ
>
0, failure probability β
∈
[0, 1] and any
Privatizer that satisfies Assumption 3.1, if the number of episodes satisfies that K
≥
eΩ

H3SAB

α2
+ min
n
K′| H2S2ABEϵ,β

K′
≤α
o
, with probability 1 −β, (µout, νout) is α-approximate

Nash, i.e., V †,νout
1
(s1) −V µout,†
1
(s1) ≤α.

The proof is deferred to Appendix C.4. Here the second term of the sample complexity bound4
ensures that the additional cost due to DP is bounded by O(α). The detailed PAC guarantees under
the special cases where the Privatizer is either Central or Local will be provided in Section 5.

5
Privatizers for JDP and LDP

In this section, we propose Privatizers that provide DP guarantees (JDP or LDP) while satisfying
Assumption 3.1. The proofs for this section can be found in Appendix D.

4The presentation here is because the term Eϵ,β is indeed dependent of the number of episodes K.

7


---Page Break---
5.1
Central Privatizer for Joint DP

Given the number of episodes K, the Central Privatizer applies K-bounded Binary Mechanism [Chan
et al., 2011] to privatize all the visitation counter streams N k
h(s, a, b), N k
h(s, a, b, s′), thus protecting
the information of all single users. Briefly speaking, Binary mechanism takes a stream of partial
sums as input and outputs a surrogate stream satisfying differential privacy, while the error for each
item scales only logarithmically on the length of the stream5. Here in multi-agent RL, for each
(h, s, a, b), the stream {N k
h(s, a, b) = Pk−1
i=1 1(si
h, ai
h, bi
h = s, a, b)}k∈[K] can be considered as the
partial sums of {1(si
h, ai
h, bi
h = s, a, b)}. Therefore, after observing 1(sk
h, ak
h, bk
h = s, a, b) at the
end of episode k, the Binary Mechanism will output a private version of Pk
i=1 1(si
h, ai
h, bi
h = s, a, b).
However, Binary Mechanism alone does not satisfy (2) of Assumption 3.1, and a post-processing
step is required. To sum up, we let the Central Privatizer follow the workflow below:

Given the privacy budget for JDP ϵ > 0,

(1) For all (h, s, a, b, s′), we apply Binary Mechanism (Algorithm 2 in Chan et al. [2011]) with
input parameter ϵ′ =
ϵ
2H log K to privatize all the visitation counter streams {N k
h(s, a, b)}k∈[K] and

{N k
h(s, a, b, s′)}k∈[K]. We denote the output of Binary Mechanism by b
N k
h.

(2) The private counts e
N k
h are derived through Section 5.3 with Eϵ,β = O( H

ϵ log(HSABK/β)2).

Our Central Privatizer satisfies the privacy guarantee below.
Lemma 5.1. For any possible ϵ, β, the Central Privatizer satisfies ϵ-JDP and Assumption 3.1 with
Eϵ,β = eO( H

ϵ ).

Combining Lemma 5.1 with Theorem 4.1 and Theorem 4.2, we have the following regret & PAC
guarantee under ϵ-JDP.
Theorem 5.2 (Results under JDP). For any possible ϵ, β, with probability 1 −β, the regret from
running DP-Nash-VI (Algorithm 1) instantiated with Central Privatizer satisfies:

Regret(K) ≤eO(
√

H2SABT + H3S2AB/ϵ).
(3)

Moreover, if the number of episodes K is larger than eΩ( H3SAB

α2
+ H3S2AB

ϵα
), with probability 1 −β,
the output policy (µout, νout) is α-approximate Nash.

Similar to the single-agent (MDP) setting (B = 1), the additional cost due to JDP is a lower order
term under the most prevalent regime where the privacy budget ϵ is a constant. When applied to the
single-agent case, our regret matches the best known regret eO(
√

H2SAT + H3S2A/ϵ) [Qiao and
Wang, 2023]. Moreover, when compared to the regret lower bound below, our main term is nearly
optimal while the lower order term has optimal dependence on ϵ.
Theorem 5.3. For any algorithm Alg satisfying ϵ-JDP, there exists a Markov Game such that the
expected regret from running Alg for K episodes (T = HK steps) satisfies:

E [Regret(K)] ≥eΩ(
p

H2S(A + B)T + HS(A + B)

ϵ
).

The regret lower bound results from the lower bound for the non-private learning [Bai and Jin, 2020]
and an adaptation of the lower bound under JDP guarantees [Vietri et al., 2020] to the multi-player
setting. Details are deferred to the appendix.

5.2
Local Privatizer for Local DP

At the end of episode k, the Local Privatizer perturbs the statistics calculated from the new tra-
jectory before sending it to the agent. Since the set of original visitation counts {σk
h(s, a, b) =
1(sk
h, ak
h, bk
h = s, a, b)}(h,s,a,b) has ℓ1 sensitivity H, we can achieve ϵ

2-LDP by directly adding
Laplace noise, i.e., eσk
h(s, a, b) = σk
h(s, a, b) + Lap( 2H

ϵ ). Similarly, repeating the above perturbation
to {1(sk
h, ak
h, bk
h, sk
h+1 = s, a, b, s′)}(h,s,a,b,s′) will lead to identical results. Therefore, the Local
Privatizer with budget ϵ is as below:

5More details in Chan et al. [2011] and Kairouz et al. [2021].

8


---Page Break---
(1) We perturb σk
h(s, a, b) = 1(sk
h, ak
h, bk
h = s, a, b) and σk
h(s, a, b, s′) = 1(sk
h, ak
h, bk
h, sk
h+1 =
s, a, b, s′) by adding independent Laplace noises: for all (h, s, a, b, s′, k),

eσk
h(s, a, b) = σk
h(s, a, b) + Lap
2H

ϵ


, eσk
h(s, a, b, s′) = σk
h(s, a, b, s′) + Lap
2H

ϵ


.
(4)

(2) Then the noisy counts are derived according to

b
N k
h(s, a, b) =

k−1
X

i=1
eσi
h(s, a, b),
b
N k
h(s, a, b, s′) =

k−1
X

i=1
eσi
h(s, a, b, s′),
(5)

and the private counts e
N k
h are solved through Section 5.3 with Eϵ,β = O( H

ϵ
p

K log(HSABK/β)).

Our Local Privatizer satisfies the privacy guarantee below.

Lemma 5.4. For any possible ϵ, β , the Local Privatizer satisfies ϵ-LDP and Assumption 3.1 with
Eϵ,β = eO( H

ϵ
√

K).

Combining Lemma 5.4 with Theorem 4.1 and Theorem 4.2, we have the following regret & PAC
guarantee under ϵ-LDP.

Theorem 5.5 (Results under LDP). For any possible ϵ, β, with probability 1 −β, the regret from
running DP-Nash-VI (Algorithm 1) instantiated with Local Privatizer satisfies:

Regret(K) ≤eO
√

H2SABT + S2AB
√

H5T/ϵ

.
(6)

Moreover, if the number of episodes K is larger than eΩ

H3SAB

α2
+ H6S4A2B2

ϵ2α2

, with probability

1 −β, the output policy (µout, νout) is α-approximate Nash.

Similar to the single-agent case, the additional cost due to LDP is a multiplicative factor to the
regret bound. When applied to the single-agent case, our regret matches the best known regret
eO
√

H2SAT + S2A
√

H5T/ϵ

[Qiao and Wang, 2023]. Moreover, we state the lower bound.

Theorem 5.6. For any algorithm Alg satisfying ϵ-LDP, there exists a Markov Game such that the
expected regret from running Alg for K episodes (T = HK steps) satisfies:

E [Regret(K)] ≥eΩ

 
p

H2S(A + B)T +

p

HS(A + B)T

ϵ

!

.

The lower bound is adapted from Garcelon et al. [2021]. While our regret has optimal dependence on
ϵ and K, the optimal dependence on H, S, A, B remains open.

5.3
The post-processing step

Now we introduce the post-processing step. At the end of episode k, given the noisy counts b
N k
h(s, a, b)
and b
N k
h(s, a, b, s′) for all (h, s, a, b, s′), the private visitation counts are constructed as following: for
all (h, s, a, b),
n
e
N k
h(s, a, b, s′)
o

s′∈S = argmin
{xs′}s′∈S
max
s′∈S

xs′ −b
N k
h(s, a, b, s′)


such that



X

s′∈S
xs′ −b
N k
h(s, a, b)

 ≤Eϵ,β

4
and xs′ ≥0, ∀s′.
e
N k
h(s, a, b) =
X

s′∈S
e
N k
h(s, a, b, s′).

(7)
Lastly, we add a constant term to each count to ensure no underestimation (with high probability).

e
N k
h(s, a, b, s′) = e
N k
h(s, a, b, s′) + Eϵ,β

2S ,
e
N k
h(s, a, b) = e
N k
h(s, a, b) + Eϵ,β

2 .
(8)

9


---Page Break---
Remark 5.7. Solving problem (7) is equivalent to solving:

min t, s.t.
xs′ −b
N k
h(s, a, b, s′)
 ≤t, xs′ ≥0, ∀s′ ∈S,



X

s′∈S
xs′ −b
N k
h(s, a, b)

 ≤Eϵ,β

4 ,

which is a Linear Programming problem with O(S) variables and O(S) linear constraints. This
can be solved in polynomial time [Nemhauser and Wolsey, 1988]. Note that the computation of CCE
(line 14 in Algorithm 1) is also a LP problem, therefore the computational complexity of DP-Nash-VI
is dominated by O(HSABK) Linear Programming problems, which is computationally friendly.

We summarize the properties of private counts e
N k
h below, which says that the post-processing
step ensures that our private transition kernel estimate is a valid probability distribution while only
enlarging the error by a constant factor.

Lemma 5.8. Suppose b
N k
h satisfies that with probability 1 −β

3 , uniformly over all (h, s, a, b, s′, k),
 b
N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤Eϵ,β

4 ,
 b
N k
h(s, a, b) −N k
h(s, a, b)
 ≤Eϵ,β

4 ,

then the e
N k
h derived above satisfies Assumption 3.1.

5.4
Some discussions

In this part, we generalize the Privatizers in Qiao and Wang [2023] (for single-agent case) to the
two-player setting, which enables our usage of Bernstein-type bonuses. Such techniques lead to a
tight regret analysis and a near-optimal “non-private part” of the regret bound eventually.

Meanwhile, the additional cost due to DP has sub-optimal dependence on parameters regarding the
Markov Game. The issue appears even in the single-agent case and is considered to be inherent to
model-based algorithms due to the explicit estimation of private transitions [Garcelon et al., 2021].
The improvement requires new algorithmic designs (e.g., private Q-learning) and we leave those as
future works.

Lastly, the Laplace Mechanism can be replaced with other mechanisms, such as Gaussian Mechanism
[Dwork et al., 2014] with approximate DP guarantee (or zCDP). The regret and PAC guarantees are
readily derived by plugging in the corresponding Eϵ,β to Theorem 4.1 and Theorem 4.2.

6
Conclusion

We take the initial steps to study trajectory-wise privacy protection in multi-agent RL. We extend
the definitions of Joint DP and Local DP to multi-player RL. In addition, we design a provably-
efficient algorithm: DP-Nash-VI (Algorithm 1) that could satisfy either of the two DP constraints with
corresponding regret guarantee. Moreover, our regret bounds strictly generalize the best known results
under DP single-agent RL. There are various interesting future directions, such as improving the
additional cost due to DP via model-free approaches and considering Markov Games with function
approximations. We believe the techniques in this paper could serve as basic building blocks.

Acknowledgments

The research is partially supported by NSF Awards #2007117 and #2048091. The work was done
while DQ and YW were with the Department of Computer Science at UCSB.

References

Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement
learning with value-targeted regression. In International Conference on Machine Learning, pages
463–474. PMLR, 2020.

Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforce-
ment learning. In Proceedings of the 34th International Conference on Machine Learning-Volume
70, pages 263–272. JMLR. org, 2017.

10


---Page Break---
Yoram Bachrach, Richard Everett, Edward Hughes, Angeliki Lazaridou, Joel Z Leibo, Marc Lanctot,
Michael Johanson, Wojciech M Czarnecki, and Thore Graepel. Negotiating team formation using
deep reinforcement learning. Artificial Intelligence, 288:103356, 2020.

Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In
International Conference on Machine Learning, pages 551–560. PMLR, 2020.

Yu Bai, Chi Jin, and Tiancheng Yu. Near-optimal reinforcement learning with self-play. Advances in
neural information processing systems, 33:2159–2170, 2020.

Borja Balle, Maziar Gomrokchi, and Doina Precup. Differentially private policy evaluation. In
International Conference on Machine Learning, pages 2130–2138. PMLR, 2016.

Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of
irrelevant training data necessary for high-accuracy learning? In ACM SIGACT Symposium on
Theory of Computing, pages 123–132, 2021.

Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365(6456):
885–890, 2019.

Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and
lower bounds. In Theory of Cryptography Conference, pages 635–658. Springer, 2016.

Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evalu-
ating and testing unintended memorization in neural networks. In USENIX Security Symposium
(USENIX Security 19), pages 267–284, 2019.

T-H Hubert Chan, Elaine Shi, and Dawn Song. Private and continual release of statistics. ACM
Transactions on Information and System Security (TISSEC), 14(3):1–24, 2011.

Sayak Ray Chowdhury and Xingyu Zhou. Differentially private regret minimization in episodic
markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence,
2022.

Sayak Ray Chowdhury, Xingyu Zhou, and Ness Shroff. Adaptive control of differentially private
linear quadratic systems. In 2021 IEEE International Symposium on Information Theory (ISIT),
pages 485–490. IEEE, 2021.

Sayak Ray Chowdhury, Xingyu Zhou, and Nagarajan Natarajan. Differentially private reward
estimation with preference feedback. arXiv preprint arXiv:2310.19733, 2023.

Qiwen Cui, Kaiqing Zhang, and Simon Du. Breaking the curse of multiagents in a large state space:
Rl in markov games with independent linear function approximation. In The Thirty Sixth Annual
Conference on Learning Theory, pages 2651–2652. PMLR, 2023.

Chris Cundy and Stefano Ermon. Privacy-constrained policies via mutual information regularized
policy gradients. arXiv preprint arXiv:2012.15019, 2020.

Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds
for episodic reinforcement learning. In Advances in Neural Information Processing Systems, pages
5713–5723, 2017.

John C Duchi, Michael I Jordan, and Martin J Wainwright. Local privacy and statistical minimax
rates. In 2013 IEEE 54th Annual Symposium on Foundations of Computer Science, pages 429–438.
IEEE, 2013.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in
private data analysis. In Theory of cryptography conference, pages 265–284. Springer, 2006.

Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Found. Trends
Theor. Comput. Sci., 9(3-4):211–407, 2014.

Jerzy Filar and Koos Vrieze. Competitive Markov decision processes. Springer Science & Business
Media, 2012.

11


---Page Break---
Evrard Garcelon, Vianney Perchet, Ciara Pike-Burke, and Matteo Pirotta. Local differential privacy
for regret minimization in reinforcement learning. Advances in Neural Information Processing
Systems, 34, 2021.

Parham Gohari, Matthew Hale, and Ufuk Topcu. Privacy-engineered value decomposition networks
for cooperative multi-agent reinforcement learning. In 2023 62nd IEEE Conference on Decision
and Control (CDC), pages 8038–8044. IEEE, 2023.

Md Tamjid Hossain and John WT Lee. Hiding in plain sight: Differential privacy noise exploitation
for evasion-resilient localized poisoning attacks in multiagent reinforcement learning. In 2023
International Conference on Machine Learning and Cybernetics (ICMLC), pages 209–216. IEEE,
2023.

Md Tamjid Hossain, Hung Manh La, Shahriar Badsha, and Anton Netchaev. Brnes: Enabling security
and privacy-aware experience sharing in multiagent robotic and autonomous systems. In 2023
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9269–9276.
IEEE, 2023.

Justin Hsu, Zhiyi Huang, Aaron Roth, Tim Roughgarden, and Zhiwei Steven Wu. Private matchings
and allocations. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing,
pages 21–30, 2014.

Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement
learning. Journal of Machine Learning Research, 11(4), 2010.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient?
In Advances in Neural Information Processing Systems, pages 4863–4873, 2018.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement
learning with linear function approximation. In Conference on Learning Theory, pages 2137–2143.
PMLR, 2020.

Chi Jin, Qinghua Liu, Yuanhao Wang, and Tiancheng Yu. V-learning–a simple, efficient, decentralized
algorithm for multiagent rl. arXiv preprint arXiv:2110.14555, 2021.

Peter Kairouz, Brendan McMahan, Shuang Song, Om Thakkar, Abhradeep Thakurta, and Zheng Xu.
Practical and private (deep) learning without sampling or shuffling. In International Conference
on Machine Learning, pages 5213–5225. PMLR, 2021.

Michael Kearns, Mallesh Pai, Aaron Roth, and Jonathan Ullman. Mechanism design in large games:
Incentives and privacy. In Proceedings of the 5th conference on Innovations in theoretical computer
science, pages 403–410, 2014.

Jonathan Lebensold, William Hamilton, Borja Balle, and Doina Precup. Actor critic with differentially
private critic. arXiv preprint arXiv:1910.05876, 2019.

Chonghua Liao, Jiafan He, and Quanquan Gu. Locally differentially private reinforcement learning
for linear mixture markov decision processes. In Asian Conference on Machine Learning, pages
627–642. PMLR, 2023.

Qinghua Liu, Tiancheng Yu, Yu Bai, and Chi Jin. A sharp analysis of model-based reinforcement
learning with self-play. In International Conference on Machine Learning, pages 7001–7010.
PMLR, 2021.

Paul Luyo, Evrard Garcelon, Alessandro Lazaric, and Matteo Pirotta. Differentially private explo-
ration in reinforcement learning with linear representation. arXiv preprint arXiv:2112.01585,
2021.

Weichao Mao, Lin Yang, Kaiqing Zhang, and Tamer Basar. On improving model-free algorithms
for decentralized multi-agent reinforcement learning. In International Conference on Machine
Learning, pages 15007–15049. PMLR, 2022.

George Nemhauser and Laurence Wolsey. Polynomial-time algorithms for linear programming.
Integer and Combinatorial Optimization, pages 146–181, 1988.

12


---Page Break---
Dung Daniel T Ngo, Giuseppe Vietri, and Steven Wu. Improved regret for differentially private
exploration in linear mdp. In International Conference on Machine Learning, pages 16529–16552.
PMLR, 2022.

Hajime Ono and Tsubasa Takahashi. Locally private distributed reinforcement learning. arXiv
preprint arXiv:2001.11718, 2020.

Dan Qiao and Yu-Xiang Wang. Near-optimal deployment efficiency in reward-free reinforcement
learning with linear function approximation. arXiv preprint arXiv:2210.00701, 2022a.

Dan Qiao and Yu-Xiang Wang. Offline reinforcement learning with differential privacy. arXiv
preprint arXiv:2206.00810, 2022b.

Dan Qiao and Yu-Xiang Wang. Near-optimal differentially private reinforcement learning. In
International Conference on Artificial Intelligence and Statistics, pages 9914–9940. PMLR, 2023.

Dan Qiao and Yu-Xiang Wang. Near-optimal reinforcement learning with self-play under adaptivity
constraints. arXiv preprint arXiv:2402.01111, 2024.

Dan Qiao, Ming Yin, Ming Min, and Yu-Xiang Wang. Sample-efficient reinforcement learning with
loglog(T) switching cost. In International Conference on Machine Learning, pages 18031–18061.
PMLR, 2022.

Dan Qiao, Ming Yin, and Yu-Xiang Wang. Logarithmic switching cost in reinforcement learning
beyond linear mdps. arXiv preprint arXiv:2302.12456, 2023.

Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. Safe, multi-agent, reinforcement
learning for autonomous driving. arXiv preprint arXiv:1610.03295, 2016.

Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences, 39(10):
1095–1100, 1953.

Roshan Shariff and Or Sheffet. Differentially private contextual linear bandits. Advances in Neural
Information Processing Systems, 31, 2018.

Ali Shavandi and Majid Khedmati. A multi-agent deep reinforcement learning framework for
algorithmic trading in financial markets. Expert Systems with Applications, 208:118124, 2022.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez,
Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without
human knowledge. nature, 550(7676):354–359, 2017.

Imdad Ullah, Najm Hassan, Sukhpal Singh Gill, Basem Suleiman, Tariq Ahamed Ahanger, Zawar
Shah, Junaid Qadir, and Salil S Kanhere. Privacy preserving large language models: Chatgpt case
study based vision and framework. arXiv preprint arXiv:2310.12523, 2023.

Giuseppe Vietri, Borja Balle, Akshay Krishnamurthy, and Steven Wu. Private reinforcement learning
with pac and regret guarantees. In International Conference on Machine Learning, pages 9754–
9764. PMLR, 2020.

Baoxiang Wang and Nidhi Hegde. Privacy-preserving q-learning with functional noise in continuous
spaces. Advances in Neural Information Processing Systems, 32, 2019.

Yuanhao Wang, Qinghua Liu, Yu Bai, and Chi Jin. Breaking the curse of multiagency: Provably effi-
cient decentralized multi-agent rl with function approximation. arXiv preprint arXiv:2302.06606,
2023.

Fan Wu, Huseyin A Inan, Arturs Backurs, Varun Chandrasekaran, Janardhan Kulkarni, and
Robert Sim. Privately aligning language models with reinforcement learning. arXiv preprint
arXiv:2310.16960, 2023a.

Yulian Wu, Xingyu Zhou, Sayak Ray Chowdhury, and Di Wang. Differentially private episodic
reinforcement learning with heavy-tailed rewards. arXiv preprint arXiv:2306.01121, 2023b.

13


---Page Break---
Qiaomin Xie, Yudong Chen, Zhaoran Wang, and Zhuoran Yang. Learning zero-sum simultaneous-
move markov games using function approximation and correlated equilibrium. In Conference on
learning theory, pages 3674–3682. PMLR, 2020.

Tengyang Xie, Philip S Thomas, and Gerome Miklau. Privacy preserving off-policy evaluation. arXiv
preprint arXiv:1902.00174, 2019.

Deheng Ye, Guibin Chen, Wen Zhang, Sheng Chen, Bo Yuan, Bo Liu, Jia Chen, Zhao Liu, Fuhao
Qiu, Hongsheng Yu, et al. Towards playing full moba games with deep reinforcement learning.
Advances in Neural Information Processing Systems, 33:621–632, 2020.

Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, and Shuai Li. Differentially private temporal
difference learning with stochastic nonconvex-strongly-concave optimization. In Proceedings of
the Sixteenth ACM International Conference on Web Search and Data Mining, pages 985–993,
2023a.

Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, and Shuai Li. Dpmac: differentially private com-
munication for cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2308.09902,
2023b.

Fuheng Zhao, Dan Qiao, Rachel Redberg, Divyakant Agrawal, Amr El Abbadi, and Yu-Xiang Wang.
Differentially private linear sketches: Efficient implementations and applications. arXiv preprint
arXiv:2205.09873, 2022.

Xingyu Zhou. Differentially private reinforcement learning with linear function approximation.
Proceedings of the ACM on Measurement and Analysis of Computing Systems, 6(1):1–27, 2022.

14


---Page Break---
A
Extended related works

Differentially private reinforcement learning. The stream of research on DP RL started from the
offline setting. Balle et al. [2016] first studied privately evaluating the value of a fixed policy from
running it for several episodes (the on policy setting). Later, Xie et al. [2019] considered a more
general setting of DP off policy evaluation. Recently, Qiao and Wang [2022b] provided the first
results for offline reinforcement learning with DP guarantees.

More efforts focused on solving regret minimization. Under the setting of tabular MDP, Vietri et al.
[2020] designed PUCB by privatizing UBEV [Dann et al., 2017] to satisfy Joint DP. Besides, under
the constraints of Local DP, Garcelon et al. [2021] designed LDP-OBI based on UCRL2 [Jaksch
et al., 2010]. Chowdhury and Zhou [2022] designed a general framework for both JDP and LDP
based on UCBVI [Azar et al., 2017], and improved upon previous results. Finally, the best known
results are obtained by Qiao and Wang [2023] via incorporating Bernstein-type bonuses. Meanwhile,
Wu et al. [2023b] studied the case with heavy-tailed rewards. Under linear MDP, the only algorithm
with JDP guarantee: Private LSVI-UCB [Ngo et al., 2022] is a private and low switching 6 version of
LSVI-UCB [Jin et al., 2020], while LDP under linear MDP still remains open. Under linear mixture
MDP, LinOpt-VI-Reg [Zhou, 2022] generalized UCRL-VTR [Ayoub et al., 2020] to guarantee JDP,
while Liao et al. [2023] also privatized UCRL-VTR for LDP guarantee. In addition, Luyo et al.
[2021] provided a unified framework for analyzing joint and local DP exploration.

There are several other works regarding DP RL. Wang and Hegde [2019] proposed privacy-preserving
Q-learning to protect the reward information. Ono and Takahashi [2020] studied the problem of
distributed reinforcement learning under LDP. Lebensold et al. [2019] presented an actor critic
algorithm with differentially private critic. Cundy and Ermon [2020] tackled DP-RL under the policy
gradient framework. Chowdhury et al. [2021] considered the adaptive control of differentially private
linear quadratic (LQ) systems. Zhao et al. [2023a] studied differentially private temporal difference
(TD) learning. Chowdhury et al. [2023] analyzed reward estimation with preference feedback under
the constraints of DP. Hossain and Lee [2023], Hossain et al. [2023], Zhao et al. [2023b], Gohari et al.
[2023] focused on the privatization of communications between multiple agents in multi-agent RL.
For applications, DP RL was applied to protect sensitive information in natural language processing
and large language models (LLM) [Ullah et al., 2023, Wu et al., 2023a]. Meanwhile, Zhao et al.
[2022] considered linear sketches with DP.

B
Proof overview

In this section, we provide a proof sketch of Theorem 4.1, which can further imply the PAC guarantee
(Theorem 4.2) and the regret bounds under JDP (Theorem 5.2) or LDP (Theorem 5.5). The proof
consists of the following steps:

(1) Bound the difference between the private statistics and their non-private counterparts.
(2) Prove that UCB and LCB hold with high probability.
(3) Bound the regret via telescoping over time steps and replace the private terms by non-private
ones.

Below we explain the key steps in detail. Recall that N k
h denotes the real visitation counts, while
e
N k
h, eP k
h are the private visitation counts and private transition kernel respectively.

Step (1).
According to Assumption 3.1 and standard concentration inequalities, we pro-
vide high probability upper bounds for ∥eP k
h (·|s, a, b) −Ph(·|s, a, b)∥1 and | eP k
h (s′|s, a, b) −
Ph(s′|s, a, b)|. Besides, we upper bound the following key term |( eP k
h −Ph) · V ⋆
h+1(s, a, b)| by
eO
q

Var b
P k
h (·|s,a,b)V ⋆
h+1(·)/ e
N k
h(s, a, b) + HSEϵ,β/ e
N k
h(s, a, b)

. Details are deferred to Appendix
C.1.

Step (2). Then we prove that UCB and LCB hold with high probability via backward induction over
timesteps (Appendix C.2). More specifically, the variance term of Γk
h is the private Bernstein-type

6For low switching RL, please refer to Qiao et al. [2022], Qiao and Wang [2022a], Qiao et al. [2023], Qiao
and Wang [2024].

15


---Page Break---
bonus, while the difference between the private variance and its non-private counterpart can be
bounded by γk
h and the lower order terms in Γk
h.

Step (3). Lastly, the regret can be bounded by telescoping:

Regret(K) ≤O

 K
X

k=1

H
X

h=1
Γk
h(sk
h, ak
h, bk
h)

!

|
{z
}
bound by non-private terms

≤eO









K
X

k=1

H
X

h=1

s

VarPh(·|sk
h,ak
h,bk
h)V πk
h+1
N k
h(sk
h, ak
h, bk
h)
|
{z
}
bound by Cauchy-Schwarz inequality and L.T.V.

+

K
X

k=1

H
X

h=1

HSEϵ,β
N k
h(sk
h, ak
h, bk
h)
|
{z
}
≤H2S2ABEϵ,βι









≤eO(
√

H2SABT + H2S2ABEϵ,β).

The details about each inequality above and the lower order terms we ignore are deferred to Appendix
C.3.

C
Proof of main theorems

In this section, we prove Theorem 4.1 and Theorem 4.2.

C.1
Properties of private estimations

We begin with some concentration results about our private transition kernel estimate eP that will be
useful for the proof. Throughout the paper, let the non-private empirical transition kernel be:

bP k
h (s′|s, a, b) = N k
h(s, a, b, s′)
N k
h(s, a, b) , ∀(h, s, a, b, s′, k).
(9)

In addition, recall that our private transition kernel estimate is defined as below.

eP k
h (s′|s, a, b) =
e
N k
h(s, a, b, s′)

e
N k
h(s, a, b)
, ∀(h, s, a, b, s′, k).
(10)

Now we are ready to list the properties below. Note that ι = log(30HSABK/β) throughout the
paper.

Lemma C.1. With probability 1 −β

15, for all (h, s, a, b, k) ∈[H] × S × A × B × [K], it holds that:

 eP k
h (·|s, a, b) −Ph(·|s, a, b)

1 ≤2

s

Sι
e
N k
h(s, a, b)
+
2SEϵ,β
e
N k
h(s, a, b)
,
(11)

 eP k
h (·|s, a, b) −bP k
h (·|s, a, b)

1 ≤
2SEϵ,β
e
N k
h(s, a, b)
.
(12)

Proof of Lemma C.1. The proof is a direct generalization of Lemma B.2 and Remark B.3 in Qiao
and Wang [2023] to the two-player setting.

Lemma C.2. With probability 1 −2β

15 , for all (h, s, a, b, s′, k) ∈[H] × S × A × B × S × [K], it
holds that:

 eP k
h (s′|s, a, b) −Ph(s′|s, a, b)
 ≤2

v
u
u
tmin{Ph(s′|s, a, b), eP k
h (s′|s, a, b)}ι
e
N k
h(s, a, b)
+
2Eϵ,βι
e
N k
h(s, a, b)
,
(13)

 eP k
h (s′|s, a, b) −bP k
h (s′|s, a, b)
 ≤
2Eϵ,β
e
N k
h(s, a, b)
.
(14)

16


---Page Break---
Proof of Lemma C.2. The proof is a direct generalization of Lemma B.4 and Remark B.5 in Qiao
and Wang [2023] to the two-player setting.

Lemma C.3. With probability 1 −2β

15 , for all (h, s, a, b, k) ∈[H] × S × A × B × [K], it holds that:



eP k
h −Ph

· V ⋆
h+1(s, a, b)
 ≤min






s

2VarPh(·|s,a,b)V ⋆
h+1(·) · ι
e
N k
h(s, a, b)
,

v
u
u
t2Var b
P k
h (·|s,a,b)V ⋆
h+1(·) · ι

e
N k
h(s, a, b)




+ 2HSEϵ,βι

e
N k
h(s, a, b)
,

(15)


eP k
h −bP k
h

· V ⋆
h+1(s, a, b)
 ≤2HSEϵ,β

e
N k
h(s, a, b)
.
(16)

Proof of Lemma C.3. The proof is a direct generalization of Lemma B.6 and Remark B.7 in Qiao
and Wang [2023] to the two-player setting.

According to a union bound, the following lemma holds.

Lemma C.4. Under the high probability event that Assumption 3.1 holds, with probability at least
1 −β

3 , the conclusions in Lemma C.1, Lemma C.2, Lemma C.3 hold simultaneously.

Throughout the proof, we will assume that Assumption 3.1 and Lemma C.4 hold, which will happen
with high probability. Before we prove the main theorems, we present the following lemma which
bounds the two variances.

Lemma C.5 (Lemma C.5 of Qiao and Wang [2022b]). For any function V ∈RS such that ∥V ∥∞≤
H, it holds that

q

Var e
P k
h (·|s,a,b)(V ) −
q

Var b
P k
h (·|s,a,b)(V )
 ≤
√

3H ·
r eP k
h (·|s, a, b) −bP k
h (·|s, a, b)

1.
(17)

In addition, according to Lemma C.1, the left hand side can be further bounded by


q

Var e
P k
h (·|s,a,b)(V ) −
q

Var b
P k
h (·|s,a,b)(V )
 ≤3H

s

SEϵ,β
e
N k
h(s, a, b)
.
(18)

C.2
Proof of UCB and LCB

For notational simplicity, for V ∈RS such that ∥V ∥∞≤H, we define

eV k
h V (s, a, b) = Var e
P k
h (·|s,a,b)V (·),
VhV (s, a, b) = VarPh(·|s,a,b)V (·).
(19)

Then the bonus term Γ can be represented as below (C2 is the universal constant in Algorithm 1).

Γk
h(s, a, b) = C2

v
u
u
u
t

eV k
h



V
k
h+1+V k
h+1
2


(s, a, b) · ι

e
N k
h(s, a, b)
+ C2HSEϵ,β · ι

e
N k
h(s, a, b)
+
C2H2Sι
e
N k
h(s, a, b)
.
(20)

We state the following lemma that can bound the lower order term, which is helpful for proving UCB
and LCB.

Lemma C.6. Suppose Assumption 3.1 and Lemma C.4 hold, then there exists a universal constant
c1 > 0 such that: if function g(s) satisfies |g|(s) ≤(V
k
h+1 −V k
h+1)(s), then it holds that:
( eP k
h −Ph)g(s, a, b)
 ≤c1

H min
n
Ph(V
k
h+1 −V k
h+1)(s, a, b), eP k
h (V
k
h+1 −V k
h+1)(s, a, b)
o

+
c1H2Sι
e
N k
h(s, a, b)
+ c1HSEϵ,βι

e
N k
h(s, a, b)
.
(21)

17


---Page Break---
Proof of Lemma C.6. If |g|(s) ≤(V
k
h+1 −V k
h+1)(s), it holds that:
( eP k
h −Ph)g(s, a, b)
 ≤
X

s′



eP k
h −Ph

(s′|s, a, b)
 · |g|(s′)

≤
X

s′



eP k
h −Ph

(s′|s, a, b)
 ·


V
k
h+1 −V k
h+1

(s′)

≤
X

s′

 

2

s

Ph(s′|s, a, b)ι

e
N k
h(s, a, b)
+
2Eϵ,βι
e
N k
h(s, a, b)

!

·


V
k
h+1 −V k
h+1

(s′)

≤
X

s′

 
Ph(s′|s, a, b)

H
+
Hι
e
N k
h(s, a, b)
+
2Eϵ,βι
e
N k
h(s, a, b)

!

·


V
k
h+1 −V k
h+1

(s′)

≤c1

H Ph(V
k
h+1 −V k
h+1)(s, a, b) +
c1H2Sι
e
N k
h(s, a, b)
+ c1HSEϵ,βι

e
N k
h(s, a, b)
,

(22)

where the third inequality is because of Lemma C.2. The forth inequality results from AM-GM
inequality. The last inequality holds for some universal constant c1.

The empirical part with the R.H.S to be eP k
h can be proven using identical proof according to (13).

Then we prove that the UCB and LCB functions are actually upper and lower bounds of the best
responses. Recall that πk is the (correlated) policy executed in the k-th episode and (µk, νk) for
both players are the marginal policies of πk. In other words, µk
h(·|s) = P

b∈B πk
h(·, b|s) and
νk
h(·|s) = P

a∈A πk
h(a, ·|s) for all (h, s) ∈[H] × S.
Lemma C.7. Suppose Assumption 3.1 and Lemma C.4 hold, then there exist universal constants
C1, C2 > 0 (in Algorithm 1) such that for all (h, s, a, b, k) ∈[H] × S × A × B × [K], it holds that:
(

Q
k
h(s, a, b) ≥Q†,νk

h
(s, a, b) ≥Qµk,†
h
(s, a, b) ≥Qk
h(s, a, b),

V
k
h(s) ≥V †,νk

h
(s) ≥V µk,†
h
(s) ≥V k
h(s).
(23)

Proof of Lemma C.7. We prove by backward induction. For each k ∈[K], the conclusion is obvious
for h = H + 1. Suppose UCB and LCB hold for Q value functions in the (h + 1)-th time step, we
first prove the bounds for V functions in the (h+1)-th step and then prove the bounds for Q functions
in the h-th step. For all s ∈S, it holds that

V
k
h+1(s) =Eπk
h+1Q
k
h+1(s)

≥sup
µ Eµ,νk
h+1Q
k
h+1(s)

≥sup
µ Eµ,νk
h+1Q†,νk

h+1(s)

=V †,νk

h+1 (s).

(24)

The conclusion V k
h+1(s) ≤V µk,†
h+1 (s) can be proven by symmetry. Therefore, it holds that

V
k
h+1(s) ≥V †,νk

h+1 (s) ≥V ⋆
h+1(s) ≥V µk,†
h+1 (s) ≥V k
h+1(s).
(25)

Next we prove the bounds for Q value functions at the h-th step. For all (s, a, b), it holds that


Q
k
h −Q†,νk

h

(s, a, b) ≥min
n
eP k
h V
k
h+1 −PhV †,νk

h+1 + γk
h + Γk
h

(s, a, b), 0
o

≥min
n
eP k
h V †,νk

h+1 −PhV †,νk

h+1 + γk
h + Γk
h

(s, a, b), 0
o

= min











eP k
h −Ph
 
V †,νk

h+1 −V ⋆
h+1

(s, a, b)
|
{z
}
(i)

+

eP k
h −Ph

V ⋆
h+1(s, a, b)
|
{z
}
(ii)

+γk
h(s, a, b) + Γk
h(s, a, b), 0









.

(26)

18


---Page Break---
The absolute value of term (i) can be bounded as below.

|(i)| ≤c1

H
eP k
h (V
k
h+1 −V k
h+1)(s, a, b) +
c1H2Sι
e
N k
h(s, a, b)
+ c1HSEϵ,βι

e
N k
h(s, a, b)
,
(27)

for some universal constant c1 according to Lemma C.6.

The absolute value of term (ii) can be bounded as below.

|(ii)| ≤

v
u
u
t2Var b
P k
h (·|s,a,b)V ⋆
h+1(·) · ι

e
N k
h(s, a, b)
+ 2HSEϵ,βι

e
N k
h(s, a, b)
≤

v
u
u
t2Var e
P k
h (·|s,a,b)V ⋆
h+1(·) · ι

e
N k
h(s, a, b)
+ 8HSEϵ,βι

e
N k
h(s, a, b)
,

(28)
where the first inequality is because of Lemma C.3 while the second inequality holds due to Lemma
C.5.

We further bound the term Var e
P k
h (·|s,a,b)V ⋆
h+1(·) as below.


eV k
h

 

V
k
h+1 + V k
h+1
2

!

−eV k
h V ⋆
h+1(·)

 (s, a, b)

≤


eP k
h ·

 

V
k
h+1 + V k
h+1
2

!2

−eP k
h ·
 
V ⋆
h+1
2

(s, a, b) +



"
eP k
h ·

 

V
k
h+1 + V k
h+1
2

!

(s, a, b)

#2

−
h
eP k
h V ⋆
h+1(s, a, b)
i2


≤4H eP k
h ·


V
k
h+1 −V k
h+1

(s, a, b).

(29)

Therefore, the term (ii) can be further bounded as below.

|(ii)| ≤

v
u
u
t2Var e
P k
h (·|s,a,b)V ⋆
h+1(·) · ι

e
N k
h(s, a, b)
+ 8HSEϵ,βι

e
N k
h(s, a, b)

≤

v
u
u
u
t2ι · eV k
h



V
k
h+1+V k
h+1
2


(s, a, b) + 2ι · 4H eP k
h ·


V
k
h+1 −V k
h+1

(s, a, b)

e
N k
h(s, a, b)
+ 8HSEϵ,βι

e
N k
h(s, a, b)

≤

v
u
u
u
t2eV k
h



V
k
h+1+V k
h+1
2


(s, a, b)ι

e
N k
h(s, a, b)
+
eP k
h ·


V
k
h+1 −V k
h+1

(s, a, b)

H
+
2H2ι
e
N k
h(s, a, b)
+ 8HSEϵ,βι

e
N k
h(s, a, b)
,

(30)

where the second inequality results from (29) and the third inequality is due to AM-GM inequality.

Combining the upper bounds of |(i)| and |(ii)|, there exist universal constants C1, C2 > 0 such that

(i) + (ii) + γk
h(s, a, b) + Γk
h(s, a, b) ≥0.
(31)

The inequality implies that


Q
k
h −Q†,νk

h

(s, a, b)
≥
0.
By symmetry,
we have

Qk
h −Qµk,†
h

(s, a, b) ≤0. As a result, it holds that Q
k
h(s, a, b) ≥Q†,νk

h
(s, a, b) ≥Q⋆
h(s, a, b) ≥

Qµk,†
h
(s, a, b) ≥Qk
h(s, a, b).

According to backward induction, the conclusion holds for all (h, s, a, b, k).

C.3
Proof of Theorem 4.1

Given the UCB and LCB property, we are now ready to prove our main results. We first state the
following lemma that controls the error of the empirical variance estimator.

19


---Page Break---
Lemma C.8. Suppose Assumption 3.1 and Lemma C.4 hold, then there exists a universal constant
c2 > 0 such that for all (h, s, a, b, k) ∈[H] × S × A × B × [K], it holds that

eV k
h

 

V
k
h+1 + V k
h+1
2

!

−VhV πk
h+1

 (s, a, b)

≤4HPh


V
k
h+1 −V k
h+1

(s, a, b) + c2H2SEϵ,β

e
N k
h(s, a, b)
+ c2H2
s

Sι
e
N k
h(s, a, b)
.

(32)

Proof of Lemma C.8. According to Lemma C.7, V
k
h(s) ≥V πk
h (s) ≥V k
h(s) always holds. Then it
holds that

eV k
h

 

V
k
h+1 + V k
h+1
2

!

−VhV πk
h+1

 (s, a, b)

≤


eP k
h

 

V
k
h+1 + V k
h+1
2

!2

−Ph

V πk
h+1
2
−

"
eP k
h

 

V
k
h+1 + V k
h+1
2

!#2

+

PhV πk
h+1
2

(s, a, b)

≤
 eP k
h


V
k
h+1
2
−Ph

V k
h+1
2
−

eP k
h V k
h+1
2
+

PhV
k
h+1
2 (s, a, b)

≤


eP k
h −Ph
 

V
k
h+1
2 (s, a, b)
|
{z
}
(i)

+
Ph



V
k
h+1
2
−

V k
h+1
2 (s, a, b)
|
{z
}
(ii)

+


eP k
h V k
h+1
2
−

PhV k
h+1
2 (s, a, b)
|
{z
}
(iii)

+


PhV k
h+1
2
−

PhV
k
h+1
2 (s, a, b)
|
{z
}
(iv)

.

(33)

The term (i) can be bounded as below due to Lemma C.1.

(i) ≤2H2
s

Sι
e
N k
h(s, a, b)
+ 2H2SEϵ,β

e
N k
h(s, a, b)
.
(34)

The term (ii) can be directly bounded as below.

(ii) ≤2HPh


V
k
h+1 −V k
h+1

(s, a, b).
(35)

The term (iii) can be bounded as below due to Lemma C.1.

(iii) ≤2H


eP k
h −Ph

V k
h+1
 (s, a, b) ≤4H2
s

Sι
e
N k
h(s, a, b)
+ 4H2SEϵ,β

e
N k
h(s, a, b)
.
(36)

The term (iv) can be directly bounded as below.

(iv) ≤2HPh


V
k
h+1 −V k
h+1

(s, a, b).
(37)

The conclusion holds according the upper bounds of term (i), (ii), (iii) and (iv).

Finally we prove the regret bound of Algorithm 1.

Proof of Theorem 4.1. Our proof base on Assumption 3.1 and Lemma C.4. We define the following
notations.











∆k
h =


V
k
h −V k
h

(sk
h),

ζk
h = ∆k
h −


Q
k
h −Qk
h


(sk
h, ak
h, bk
h),

ξk
h = Ph


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h) −∆k
h+1.

(38)

20


---Page Break---
Then it holds that ζk
h and ξk
h are martingale differences bounded by H. In addition, we use the
following abbreviations for notational simplicity.











γk
h = γk
h(sk
h, ak
h, bk
h),
Γk
h = Γk
h(sk
h, ak
h, bk
h),
N k
h = N k
h(sk
h, ak
h, bk
h),
e
N k
h = e
N k
h(sk
h, ak
h, bk
h).

(39)

Then we have the following analysis about ∆k
h.

∆k
h = ζk
h +


Q
k
h −Qk
h


(sk
h, ak
h, bk
h)

≤ζk
h + 2γk
h + 2Γk
h + eP k
h


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h)

≤ζk
h + 2Γk
h +

1 + 2C1

H


·

"
1 + c1

H


· Ph


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h) + c1H2Sι

e
N k
h
+ c1HSEϵ,βι

e
N k
h

#

≤ζk
h +

1 + c3

H


· Ph


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h) + c3H2Sι

e
N k
h
+ c3HSEϵ,βι

e
N k
h

+ c3

v
u
u
u
t

eV k
h



V
k
h+1+V k
h+1
2


(sk
h, ak
h, bk
h)ι

e
N k
h
|
{z
}
(i)

,

(40)

where the first inequality holds because of the definition of Q and Q. The second inequality holds due
to the definition of γk
h and Lemma C.6. The last inequality holds for some universal constant c3 > 0.

The term (i) can be further bounded as below according to Lemma C.8 and AM-GM inequality.

(i) ≤

v
u
u
tVhV πk
h+1(sk
h, ak
h, bk
h)ι
e
N k
h
+

v
u
u
t4HPh


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h)ι

e
N k
h
+ H
p

c2SEϵ,βι

e
N k
h
+ c2

s

ι
e
N k
h
+ H2ι√c2S

e
N k
h

≤

v
u
u
tVhV πk
h+1(sk
h, ak
h, bk
h)ι
e
N k
h
+
c4Ph


V
k
h+1 −V k
h+1

(sk
h, ak
h, bk
h)

H
+ c4H2√

Sι
e
N k
h
+ c4H
p

SEϵ,βι
e
N k
h
+ c4

s

ι
e
N k
h
,

(41)
where the first inequality results from Lemma C.8 and AM-GM inequality on the last term of (32).
The second inequality holds for some universal constant c4 > 0 according to AM-GM inequality.

Plugging in the upper bound of term (i), for some universal constant c5 > 0, it holds that:

∆k
h ≤ζk
h +

1 + c5

H


ξk
h +

1 + c5

H


∆k
h+1 + c5

v
u
u
tVhV πk
h+1(sk
h, ak
h, bk
h)ι
e
N k
h
+ c5

s

ι
e
N k
h
+ c5H2Sι

e
N k
h
+ c5HSEϵ,βι

e
N k
h
.

(42)

Summing ∆k
1 over k ∈[K], we have for some universal constant c6 > 0, it holds that:

K
X

k=1
∆k
1 ≤

K
X

k=1

H
X

h=1


1 + c5

H

h−1
ζk
h
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


1 + c5

H

h
ξk
h
|
{z
}
(iii)

+c6

K
X

k=1

H
X

h=1

v
u
u
tVhV πk
h+1(sk
h, ak
h, bk
h)ι
e
N k
h
|
{z
}
(iv)

+ c6

K
X

k=1

H
X

h=1

s

ι
e
N k
h
|
{z
}
(v)

+c6

K
X

k=1

H
X

h=1

H2Sι + HSEϵ,βι

e
N k
h
|
{z
}
(vi)

.

(43)

21


---Page Break---
The term (ii) and term (iii) can be bounded by Azuma-Hoeffding inequality. With probability 1 −2β

9 ,
it holds that
|(ii)| ≤O
√

H3Kι

,
|(iii)| ≤O
√

H3Kι

.
(44)

The main term (iv) is bounded as below.

(iv) ≤

K
X

k=1

H
X

h=1

s

VhV πk
h+1(sk
h, ak
h, bk
h)ι
N k
h

≤

v
u
u
t

K
X

k=1

H
X

h=1
VhV πk
h+1(sk
h, ak
h, bk
h)ι ·

K
X

k=1

H
X

h=1

1
N k
h

≤
p

O (H2K + H3ι) ι · O(HSABι)

= eO
√

H3SABK + H2√

SAB

.

(45)

The first inequality is because e
N k
h ≥N k
h (Assumption 3.1). The second inequality holds due to
Cauchy-Schwarz inequality. The third inequality holds with probability 1 −β

9 because of Law of
total variance and standard concentration inequalities (for details please refer to Lemma 8 of Azar
et al. [2017]).

The term (v) is bounded as below due to pigeon-hole principle.

(v) ≤

K
X

k=1

H
X

h=1

r ι

N k
h
≤O(
√

H2SABKι),
(46)

where the first inequality is because e
N k
h ≥N k
h (Assumption 3.1). The last one results from pigeon-
hole principle.

The term (vi) can be bounded as below.

(vi) ≤

K
X

k=1

H
X

h=1

H2Sι + HSEϵ,βι

N k
h
≤O(H3S2ABι2) + O(H2S2ABEϵ,βι2).
(47)

Combining the upper bounds for term |(ii)|, |(iii)|, (iv), (v) and (vi). The regret of Algorithm 1 can
be bounded as below.

Regret(K) =

K
X

k=1

h
V †,νk
1
(s1) −V µk,†
1
(s1)
i
≤

K
X

k=1

h

V
k
1(s1) −V k
1(s1)
i

=

K
X

k=1
∆k
1 ≤eO
√

H2SABT + H3S2AB + H2S2ABEϵ,β

,

(48)

where T = HK is the number of steps.

The failure probability is bounded by β ( β

3 for Assumption 3.1, β

3 for Lemma C.4, β

3 for terms (ii),
(iii) and (iv)). The proof of Theorem 4.1 is complete.

C.4
Proof of Theorem 4.2

In this part, we provide a proof of the PAC guarantee: Theorem 4.2. The proof directly follows from
the proof of the regret bound (Theorem 4.1).

Proof of Theorem 4.2. Recall that we choose πout = πk such that k = argmink


V
k
1 −V k
1

(s1).
Therefore, we have

V †,νout
1
(s1) −V µout,†
1
(s1) ≤V

k
1(s1) −V k
1(s1) ≤1

K
eO
√

H3SABK + H2S2ABEϵ,β

, (49)

if ignoring the lower order term of the regret bound.

Therefore, choosing K ≥eΩ

H3SAB

α2
+ min
n
K′| H2S2ABEϵ,β

K′
≤α
o
bounds the R.H.S by α.

22


---Page Break---
D
Missing proof in Section 5

In this section, we provide the missing proof for results in Section 5. Recall that N k
h is the real
visitation count, b
N k
h is the intermediate noisy count calculated by both Privatizers and e
N k
h is the final
private count after the post-processing step. Note that most of the proof here are generalizations of
Appendix D in Qiao and Wang [2023] to the multi-player setting, and here we state the proof for
completeness.

Proof of Lemma 5.1. Due to Theorem 3.5 of Chan et al. [2011] and Lemma 34 of Hsu
et al. [2014], the release of { b
N k
h(s, a, b)}(h,s,a,b,k) satisfies
ϵ
2-DP. Similarly, the release of
{ b
N k
h(s, a, b, s′)}(h,s,a,b,s′,k) also satisfies ϵ

2-DP. Therefore, the release of the following private
counters { b
N k
h(s, a, b)}(h,s,a,b,k), { b
N k
h(s, a, b, s′)}(h,s,a,b,s′,k) satisfy ϵ-DP. Due to post-processing
(Lemma 2.3 of Bun and Steinke [2016]), the release of both private counts { e
N k
h(s, a, b)}(h,s,a,b,k)
and { e
N k
h(s, a, b, s′)}(h,s,a,b,s′,k) also satisfies ϵ-DP. Then it holds that the release of all πk is ϵ-DP
according to post-processing. Finally, the guarantee of ϵ-JDP results from Billboard Lemma (Lemma
9 of Hsu et al. [2014]).

For utility analysis, because of Theorem 3.6 of Chan et al. [2011], our choice ϵ′ =
ϵ
2H log K in Binary

Mechanism and a union bound, with probability 1 −β

3 , for all (h, s, a, b, s′, k),
 b
N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤O
H

ϵ log(HSABK/β)2

,

 b
N k
h(s, a, b) −N k
h(s, a, b)
 ≤O
H

ϵ log(HSABK/β)2

.
(50)

Together with Lemma 5.8, the Central Privatizer satisfies Assumption 3.1 with Eϵ,β = eO
  H

ϵ

.

Proof of Theorem 5.2. The proof directly results from plugging Eϵ,β = eO
  H

ϵ

into Theorem 4.1
and Theorem 4.2.

Proof of Theorem 5.3. The
first
term
results
from
the
non-private
regret
lower
bound
Ω(
p

H2S(A + B)T) [Bai and Jin, 2020]. The second term is a direct adaptation of the Ω(HSA/ϵ)
lower bound for any algorithms with ϵ-JDP guarantee under single-agent MDP [Vietri et al.,
2020].

Proof of Lemma 5.4. The privacy guarantee directly results from properties of Laplace Mechanism
and composition of DP [Dwork et al., 2014].

For utility analysis, because of Corollary 12.4 of Dwork et al. [2014] and a union bound, with
probability 1 −β

3 , for all possible (h, s, a, b, s′, k),
 b
N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤O
H

ϵ

p

K log(HSABK/β)

,

 b
N k
h(s, a, b) −N k
h(s, a, b)
 ≤O
H

ϵ

p

K log(HSABK/β)

.
(51)

Together with Lemma 5.8, the Local Privatizer satisfies Assumption 3.1 with Eϵ,β = eO

H

ϵ
√

K

.

Proof of Theorem 5.5. The proof directly results from plugging Eϵ,β = eO

H

ϵ
√

K

into Theorem
4.1 and Theorem 4.2.

Proof of Theorem 5.6. The
first
term
results
from
the
non-private
regret
lower
bound
Ω(
p

H2S(A + B)T) [Bai and Jin, 2020].
The second term is a direct adaptation of the
Ω(
√

HSAT/ϵ) lower bound for any algorithms with ϵ-LDP guarantee under single-agent MDP
[Garcelon et al., 2021].

23


---Page Break---
Proof of Lemma 5.8. For clarity, we denote the solution of (7) by ¯N k
h and therefore e
N k
h(s, a, b, s′) =
¯N k
h(s, a, b, s′) + Eϵ,β

2S , e
N k
h(s, a, b) = ¯N k
h(s, a, b) + Eϵ,β

2 .

When the condition (two inequalities) in Lemma 5.8 holds, the original counts {N k
h(s, a, b, s′)}s′∈S
is a feasible solution to the optimization problem, which means that

max
s′

 ¯N k
h(s, a, b, s′) −b
N k
h(s, a, b, s′)
 ≤max
s′

N k
h(s, a, b, s′) −b
N k
h(s, a, b, s′)
 ≤Eϵ,β

4 .

Combining with the condition in Lemma 5.8 with respect to b
N k
h(s, a, b, s′), it holds that

 ¯N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤
 ¯N k
h(s, a, b, s′) −b
N k
h(s, a, b, s′)
+
 b
N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤Eϵ,β

2 .

Since e
N k
h(s, a, b, s′) = ¯N k
h(s, a, b, s′) + Eϵ,β

2S and ¯N k
h(s, a, b, s′) ≥0, we have

e
N k
h(s, a, b, s′) > 0,
 e
N k
h(s, a, b, s′) −N k
h(s, a, b, s′)
 ≤Eϵ,β.
(52)

For ¯N k
h(s, a, b), according to the constraints in the optimization problem (7), it holds that
 ¯N k
h(s, a, b) −b
N k
h(s, a, b)
 ≤Eϵ,β

4 .

Combining with the condition in Lemma 5.8 with respect to b
N k
h(s, a, b), it holds that

 ¯N k
h(s, a, b) −N k
h(s, a, b)
 ≤
 ¯N k
h(s, a, b) −b
N k
h(s, a, b)
 +
 b
N k
h(s, a, b) −N k
h(s, a, b)
 ≤Eϵ,β

2 .

Since e
N k
h(s, a, b) = ¯N k
h(s, a, b) + Eϵ,β

2 , we have

N k
h(s, a, b) ≤e
N k
h(s, a, b) ≤N k
h(s, a, b) + Eϵ,β.
(53)

According to the last line of the optimization problem (7),
we have
¯N k
h(s, a, b)
=
P

s′∈S ¯N k
h(s, a, b, s′) and therefore,

e
N k
h(s, a, b) =
X

s′∈S
e
N k
h(s, a, b, s′).
(54)

The proof is complete by combining (52), (53) and (54).

24


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract claims that this paper is about differentially private reinforcement
learning with self-play.

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
Justification: We discuss in Section 5 that the additional cost due to DP does not have
optimal dependence on H, S, A, B.

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

25


---Page Break---
Answer: [Yes]

Justification: The paper provides the full set of assumptions and a complete proof.

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

Justification: This is a theory paper and we do not conduct experiments.

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

26


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: This is a theory paper and we do not conduct experiments.
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
Answer: [NA]
Justification: This is a theory paper and we do not conduct experiments.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: This is a theory paper and we do not conduct experiments.
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

27


---Page Break---
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
Answer: [NA]
Justification: This is a theory paper and we do not conduct experiments.
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
Justification: The research conducted in the paper conform, in every respect, with the
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
Justification: This is a theory paper regarding privacy protection, which does not have
negative societal impact.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

28


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

Justification: There is no risk of misuse of the algorithm in this paper.

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

Justification: This is a theory paper and we do not use other assets.

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

29


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not introduce any new assets.
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
Justification: This is a theory paper and we do not conduct any experiments.
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

30


---Page Break---
