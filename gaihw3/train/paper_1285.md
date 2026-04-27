Queueing Matching Bandits with Preference Feedback

Jung-hun Kim
Seoul National University
Seoul, South Korea
junghunkim@snu.ac.kr

Min-hwan Oh
Seoul National University
Seoul, South Korea
minoh@snu.ac.kr

Abstract

In this study, we consider multi-class multi-server asymmetric queueing systems
consisting of N queues on one side and K servers on the other side, where jobs ran-
domly arrive in queues at each time. The service rate of each job-server assignment
is unknown and modeled by a feature-based Multi-nomial Logit (MNL) function.
At each time, a scheduler assigns jobs to servers, and each server stochastically
serves at most one job based on its preferences over the assigned jobs. The primary
goal of the algorithm is to stabilize the queues in the system while learning the
service rates of servers. To achieve this goal, we propose algorithms based on
UCB and Thompson Sampling, which achieve system stability with an average
queue length bound of O(min{N, K}/ϵ) for a large time horizon T, where ϵ is
a traffic slackness of the system. Furthermore, the algorithms achieve sublinear
regret bounds of e
O(min{
√

TQmax, T 3/4}), where Qmax represents the maximum
queue length over agents and times. Lastly, we provide experimental results to
demonstrate the performance of our algorithms.

1
Introduction

Multi-class multi-server queueing systems, which have been extensively studied by [20, 19, 37, 8, 4],
are motivated by real-world applications such as ride-hailing platforms, where riders are assigned to
drivers. Other examples are online job markets, where applicants are recommended for employment,
and online labor service markets, where tasks are recommended to freelance workers. In these
systems, there are two sides: queues (agents) on one side and servers (arms) on the other. At each
time step, jobs stochastically arrive in multiple queues and are then assigned to multiple servers by a
matchmaking or scheduler algorithm to stabilize the systems. Importantly, the previous work assumes
that the service rates for each server are known for scheduling jobs.

However, real-world observations reveal that the service rate may not be known beforehand. Therefore,
learning the service rates associated with the stochastic behavior of servers is essential for real-world
applications. Furthermore, the behavior of servers may depend on their (relative) preferences
over assigned jobs. For example, in ride-hailing platforms where there are riders and drivers, the
preferences of drivers over riders (not necessarily based on personal information but rather a rider’s
pick-up location, destination, etc.) may not be known in advance to riders or even the systems,
necessitating the learning of drivers’ preferences through preference feedback over assigned riders.

Learning service rates in an online manner, based on partial feedback from each scheduling, is
highly associated with multi-armed bandit problems [33]. These problems are fundamental sequential
learning tasks where, for queueing systems, a job is assigned to a server and receives feedback on
whether the job is served or not. The utilization of bandit strategies for queueing systems has been
recently studied by Choudhury et al. [12], Stahlbuhk et al. [48], Gaitonde and Tardos [17]; Freund
et al. [15], Hsu et al. [22], Krishnasamy et al. [30, 31, 32], Sentenac et al. [46], Yang et al. [52], Huang
et al. [23], which primary focus is naturally on establishing stability of the systems while learning
service rates.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Utility

Server/Arm
Job

Queue/Agent

(a) Jobs randomly arrive in queues
(agents) and there are unknown
different values of utility between
queues and servers (arms) closely
related to preferences.

Server/Arm
Job

Queue/Agent

Assigned

(b) Each queue is assigned to a
server based on a scheduler.

Accepted/Served job

Queue/Agent

Rejected/
Remaining job

Accepted

Rejected

(c) Each server accepts at most
one of the assigned jobs based
on its preference. A job of each
accepted queue is served while
jobs of rejected queues remain.

Figure 1: Illustration of queueing process with 4 queues/agents (N = 4) and 3 servers/arms (K = 3)

However, there are still gaps between the previous models and the real-world applications. In all
previous work, the inherent service rates for each job-server assignment are determined regardless of
other jobs assigned to the same server (or it is not allowed to assign multiple jobs to the same server).
This differs from real-world scenarios where service rate may depend on (relative) preference over
multiple assigned jobs, as seen in online labor markets or ride-hailing platforms. Additionally, in
Freund et al. [15], Krishnasamy et al. [31, 32], which is closely related work to our study regarding
multi-class multi-server queueing systems with asymmetric service rates, it was allowed to assign
a job in an empty queue to a server to obtain feedback (null request), which may not be realistic.
Furthermore, all of the previous work considers a simple model without a generalizable structure or
utilizing features for service rates.

Another line of related work is matching bandits, in which there are two sides (agents and arms), and
the behavior of arms is based on their preferences [36, 45, 35, 9, 54, 29]. However, their focus is on
static settings, whereas our work considers dynamic job arrivals in queues. Additionally, it is worth
mentioning online matching problems [27, 39, 40, 18, 16, 28], where the sole focus is on optimizing
matching rather than learning underlying models from bandit feedback. In contrast, our study
concentrates on bandit problems related to learning latent utilities by managing the tradeoff between
exploration and exploitation while establishing the stability of the queueing systems. Importantly,
neither previous work on matching bandits nor online matching problems addresses the stability of
queueing systems, which is our primary focus.

In this work, we propose a novel and practical framework for queueing matching bandits under
preference feedback. In the following, we describe our setting accompanied by an illustration in
Figure 1. (a) Multiple queues (agents) and servers (arms) are involved, with jobs arriving randomly
in queues. Additionally, there are unknown utility values between queues and servers regarding
preferences. (b) Nonempty queue are then assigned to servers based on a scheduling policy. (c)
Subsequently, each server stochastically accepts at most one of the assigned queues based on its
unknown preference and serves one unit of job of the accepted queue. The rejected jobs remain in the
queues. The processes of (a), (b), and (c) are repeated over time.

Summary of Our Contributions

• We propose a novel and practical framework for queueing matching bandits, where N
agents and K arms are involved, and jobs randomly arrive in agents’ queues each round.
Subsequently, a scheduler assigns agents to arms, and the service rates for the assigned
agents depend on the arms’ unknown preferences over them. These service rates are modeled
using a feature-based multinomial logit function. To the best of our knowledge, our paper is
the first to investigate either a feature-based service function or preference feedback using a
multinomial logit function for service rates.

• We propose algorithms based on Upper Confidence Bound (UCB) and Thompson Sampling
(TS), which achieve stability with average queue length bounds of O( min{N,K}

ϵ
) for a large
time horizon T, under a traffic slackness of ϵ.

2


---Page Break---
• Furthermore, the algorithms achieve sublinear regret bounds of e
O(min{
√

TQmax, T 3/4}),
where Qmax represents the maximum queue length over agents and times. It is worth
mentioning that regret analysis has not been studied in the closely related queueing bandit
literature such as Sentenac et al. [46], Freund et al. [15], Yang et al. [52].

• Finally, we present experimental results demonstrating the stability and regret performance
of our algorithms with comparison to previously suggested methods.

2
Related Work

Bandits for Queues.
Learning the unknown service rates for queueing systems in an online manner
under bandit feedback has recently gained widespread attention [12, 47, 31, 48, 22, 32, 46, 52, 23].
While Stahlbuhk et al. [47], Krishnasamy et al. [30], Choudhury et al. [12] focused on a single queue
with multiple servers, aiming to efficiently assign the queue to the optimal server while learning the
service rates, subsequent study of Krishnasamy et al. [32] expanded this scope to encompass multiple
queues and multiple servers. However, this extension relied on a strong structural assumption that
each agent possesses a unique and distinctly optimal server without sharing the same optimal server.
Consequently, the algorithms employed in such scenarios do not necessarily consider the dynamic
queue lengths when scheduling jobs; instead, they concentrate solely on learning service rates to
achieve optimal matching. As the closest work to our study, another line of work of Sentenac et al.
[46], Freund et al. [15], Yang et al. [52] has focused more on dynamic queue lengths for multi-queue
and multi-server scenarios without assuming the unique and distinct optimal servers for each agent.
In this context, all of the previous work aimed to achieve queue length stability.

However, to the best of our knowledge, none of the previous works has focused on a structured model
with features for service rates. Furthermore, previous studies either did not allow the assignment
of different types of jobs to a server simultaneously [32, 52], or, if they did, as in Sentenac et al.
[46], Freund et al. [15], the inherent values of service rates for each job-server assignment remained
constant regardless of the entire assignments. Additionally, the server’s behavior given multiple
assigned jobs was either simple or deterministic, based on the uniform-randomly selected job among
the assigned jobs [46], or the job with the highest bid generated from the algorithm [15], respectively.
However, these approaches may not reflect real-world scenarios, where the service rates depend on
the relative preferences among the assigned jobs. Lastly, in [15, 32], it was allowed to assign a job to
a server from an empty queue for obtaining feedback, which may not be realistic.

In our study, we adopt a structured model incorporating features for unknown asymmetric service
rates. We enable the assignment of multiple agents to the same server, where the server stochastically
selects at most one of the assigned agents based on its undisclosed (relative) preference. Also, the
assignment of an agent is available only when the queue of the agent is not empty. This scenario
mirrors common situations in real-world applications such as ride-hailing platforms and online labor
markets where available multiple riders or tasks can be assigned to a driver or a worker, respectively,
and the driver or worker behaves stochastically depending on its unknown preference.

It is noteworthy that we focus not only on the stability of the systems regarding queue lengths but
also on regret against an oracle, while the closely related works by [46, 15, 52], which considered
multi-class multi-server queueing systems, only addressed stability analysis. Regret analysis has only
been studied under some limited scenarios, such as a single-server [23], a single-queue [48], or the
strong assumption that each agent has a unique and well-separated optimal server [32].

Matching Bandits.
Next, we investigate two-sided matching bandits, a topic initially explored
by Liu et al. [36] and subsequently studied by Sankararaman et al. [45], Liu et al. [35], Basu et al.
[9], Zhang et al. [54], Kong and Li [29]. The primary goal is to minimize regret by attaining an
optimal stable matching by learning agents’ side preferences through stochastic reward feedback
under static settings and deterministic behavior of arms with known preferences. Our study aligns with
the model, wherein agents select arms based on preferences. However, our study sets itself apart from
prior work on matching bandits in several key aspects. Firstly, we consider dynamic environments
with job arrivals for agents. Secondly, we propose that arm behavior is stochastic, with unknown
preferences, necessitating the learning of arms’ preferences. Lastly, our main target is to stabilize the
queue lengths in the systems rather than find stable matching for stable marriage problems.

MNL Bandits.
Lastly, we examine MNL bandits which were initially proposed by Agrawal et al.
[6] and followed by Agrawal et al. [7], Chen et al. [11], Oh and Iyengar [43, 44]. In MNL bandits,

3


---Page Break---
the goal is to select assortments of arms to maximize reward, which is based on preferences over the
arms in the selected assortment. In our study, we adopt the MNL model for arms’ choice preferences
in dynamic systems, which, to the best of our knowledge, is the first consideration of bandits for
queueing systems.

3
Problem Statement

There are N agents (queues) and K arms (servers). At each time, a job for each agent n ∈[N] arrives
randomly following a Bernoulli distribution with an unknown arrival rate λn ∈[0, 1] 1. Then at each
time t ∈[T] where T is the time horizon, each agent n ∈[N] is assigned to an arm kπ
n,t ∈[K] by
a policy π. For notational simplicity, we use kn,t for kπ
n,t when there is no confusion. Let Qn(t)
be the length of the queue for jobs of agent n ∈[N] at the beginning of time slot t in the system.
We consider that at most L agents can be assigned to each arm k ∈[K] at each time and N ≤KL
to ensure that all agents can be assigned to arms. Then, we define the set of agents (assortment)
assigned to arm k by policy π at time t as Sk,t = {n ∈[N] : kn,t = k, Qn(t) ̸= 0}, considering
only available agents with nonempty queues.

Now we explain the structure of our model. Each agent n has known d-dimensional feature informa-
tion of xn ∈Rd, and each arm k has latent (unknown) parameter θk ∈Rd for preference utilities
over agents. We adopt the Multi-nomial Logit (MNL) function commonly studied for preference
feedback [43, 44, 6, 7, 11]. Then given assortment Sk,t, arm k serves a job of agent n ∈Sk,t with a
service probability (service rate) determined by the MNL model as

µ(n|Sk,t, θk) =
exp(x⊤
n θk)
1 + P

m∈Sk,t exp(x⊤
mθk).

We note that µ(n|Sk,t, θk) represents the preference of arm k to agent n over the assigned agents Sk,t.
The service rate of each n depends on the assigned agents rather than static by reflecting real-world
scenarios, which is different from the conventional queueing problems. Following the MNL function,
the arm is allowed not to serve any agents (or allowed to serve null agent n0) with the probability
of µ(n0|Sk,t, θk) =
1
1+P

m∈Sk,t exp(x⊤
mθk). Under the MNL model for the service rate, at each time,

each arm accepts at most one agent and serves that agent’s job. The queue lengths for the accepted
agents are each reduced by one, while the queue lengths for the refused agents remain the same.

Objective Function.
Here we provide the goal of this problem. For describing the stochastic
process in the systems, at each time t, let An(t) ∈{0, 1} be a random variable with mean λn, which
denotes whether a new job arrives in the queue of agent n ∈[N] (here, 1 denotes arrival). Also,
given that n is assigned to arm kn,t, let Dn(t|Skn,t,t) ∈{0, 1} be a random variable with mean
µ(n|Skn,t,t, θkn,t), which represents whether the assigned agent n, given Skn,t,t, is accepted by arm
kn,t (here 1 denotes acceptance). Then the queue length of the agent n evolves as Qn(t + 1) =
(Qn(t) + An(t) −Dn(t|Skn,t,t))+ where x+ denotes max{x, 0} for x ∈R.

As in the previous work of queueing bandits for multiple types of agents and servers [46, 15, 52, 23],
which are closely related work to ours, our primary focus is on stabilizing the dynamic systems
regarding queue lengths. For analyzing the stability of the systems, we define the average queue
lengths over horizon time T as

Q(T) = 1

T

X

t∈[T ]

X

n∈[N]
E [Qn(t)] .

Then the goal of this problem is to design an online matching algorithm to assign the agents to arms
for stabilizing the systems by bounding Q(T). We define the system stability as follows.

Definition 1. The systems are denoted to be stable when limT →∞Q(T) < ∞.

1Our study can be easily extended to the multi-unit case where random variable An(t) lies in [0, M] for
some M > 0 with mean λn ∈[0, M].

4


---Page Break---
The same stability definition was considered in Neely [41, 42], Freund et al. [15], Yang et al.
[52], Huang et al. [23]. It is noteworthy that according to [42, 41, 15], the system satisfying this
stability condition of Definition 1 is denoted to be strongly stable.2

For the analyses, we first present the regularity conditions.

Assumption 1. ∥xn∥2 ≤1 for all n ∈[N] and ∥θk∥2 ≤1 for all k ∈[K].

Assumption 2. There exists κ > 0 such that infθ∈Rd:∥θ∥2≤1 µ(n|S, θ)µ(n0|S, θ) ≥κ for any n ∈S
and S ⊂[N].

These regularity conditions are commonly taken into account in the logistic and MNL bandit literature
[13, 3, 43, 44]. We note that, in the worst-case, 1/κ = O(L2).

We
define
the
set
of
feasible
disjoint
assortments
given
any
N
⊆
[N]
as
M(N) = {(S1, ..., SK) : Sk ⊆N, |Sk| ≤L ∀k ∈[K], Sk ∩Sl = ∅∀k ̸= l, S

k∈[K] Sk = N}.
Then, we consider the condition of traffic slackness for stability as follows.

Assumption 3. For some traffic slackness 0 < ϵ < 1, there exists {Sk}k∈[K] ∈M([N]) such that
λn + ϵ ≤µ(n|Sk, θk) for all n ∈Sk and k ∈[K].

We note that similar slackness assumptions have been commonly considered in queueing bandits
[15, 52, 23]. As ϵ decreases, achieving stability in our setting becomes more challenging. A discussion
for a refined version of ϵ can be found in Appendix A.2.

4
Preliminary Study: An Oracle Algorithm of MaxWeight

For queueing systems, MaxWeight [50] has been proven to have optimal throughput keeping the
queues in networks stable under the known service rates [51, 38, 37, 49]. Here, we analyze the
oracle policy using MaxWeight under known θk’s. We define the set of agents having non-empty
queues at time t as Nt = {n ∈[N]] : Qn(t) ̸= 0}. Then, the oracle policy using MaxWeight in our
setting is defined as {Sk,t}k∈[K] = argmax{Sk}k∈[K]∈M(Nt)
P

k∈[K]
P

n∈Sk Qn(t)µ(n|Sk, θk),
where priority in the assortment scheduling is on queues with either large queue lengths or high
service rates. We provide an analysis of stability of the MaxWeight oracle for known θk’s as follows.

Proposition 1. Given the prior knowledge of θk for all k ∈[K], the average queue length of
MaxWeight is bounded as Q(T) = O

min{N,K}

ϵ

, which implies that the algorithm achieves
stability.

Proof. The proof is provided in Appendix A.3

We note that the term of min{N, K} in the average queue length is a result of the system’s variance.
Additionally, as ϵ decreases, the stability of the system deteriorates due to the reduced traffic slackness.
Further discussion regarding ϵ can be found in Appendix A.2.

Since the oracle algorithm requires prior knowledge of θk’s for the service rate, we cannot use it
directly in our bandit setting. In the following section, we propose algorithms incorporating an
efficient learning procedure to achieve stability in our setting.

5
Algorithms and Analyses

5.1
UCB-based Algorithm
We first propose an algorithm based on the UCB strategy, UCB-QMB (Algorithm 1). We define the
negative log-likelihood as fk,t(θ) := −P

n∈Sk,t∪{n0} yn,t log µ(n|Sk,t, θ) where yn,t ∈{0, 1} is
observed preference feedback (1 denotes service acceptance, and 0 denotes refusal) and define the
gradient of the likelihood as

gk,t(θ) := ∇θfk,t(θ) =
X

n∈Sk,t
(µ(n|Sk,t, θ) −yn,t)xn.
(1)

2As mentioned in [42, 15], in general, this stability condition is stronger than the mean rate stability of

limT →∞
E[P
n∈[N] Qn(T )]

T
= 0 but weaker than the uniform stability of E[P

n∈[N] Qn(t)] ≤C for all t > 0
for some constant C > 0.

5


---Page Break---
Then, we construct the estimator of ˆθk,t from online updates applying online newton step studied by
[21, 53, 24, 14, 44] as ˆθk,t = argminθ∈Θ gk,t−1(ˆθk,t−1)⊤θ + 1

2∥θ −ˆθk,t−1∥2
Vk,t, where Θ = {θ ∈

Rd : ∥θ∥2 ≤1} and Vk,t = λId + κ

2
Pt−1
s=1
P

n∈Sk,s xnx⊤
n . Using the estimator, we define the UCB
index for agent n in assortment Sk as

eµUCB
t
(n|Sk, ˆθk,t) =
exp(hUCB
n,k,t )

1 + P

m∈Sk exp(hUCB
m,k,t),
(2)

where hUCB
n,k,t := x⊤
n ˆθk,t + βt∥xn∥V −1
k,t with βt = C1
q

λ + d

κ log(1 + tLK

dλ ) for some constant
C1 > 0. We utilize the MaxWeight with the UCB indexes by setting λ = 1.

Algorithm 1 UCB-Queueing Matching Bandit (UCB-QMB)
Input: λ, κ, C1 > 0
for t = 1, . . . , T do

for k ∈[K] do

ˆθk,t ←argminθ∈Θ gk,t−1(ˆθk,t−1)⊤θ + 1

2∥θ −ˆθk,t−1∥2
Vk,t with (1)

{Sk,t}k∈[K] ←
argmax
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
Qn(t)eµUCB
t
(n|Sk, ˆθk,t) with (2)

Offer {Sk,t}k∈[K] and observe preference feedback yn,t ∈{0, 1} for all n ∈Sk,t, k ∈[K]

5.1.1
Stability Analysis of UCB-QMB
Here we provide an analysis for the stability of UCB-QMB (Algorithm 1).

Theorem
1.
The
average
queue
length
of
Algorithm
1
is
bounded
as
Q(T)
=

O

min{N,K}

ϵ
+ d2N2K2

κ4ϵ6
polylog(T )

T

, which implies that the algorithm achieves stability as

lim
T →∞Q(T) = O
min{N, K}

ϵ


.

We note that Algorithm 1 achieves the same average queue length bound of O( min{N,K}

ϵ
) with the
oracle of MaxWeight (Proposition 1) when T is large enough.

Proof sketch. Here we provide a proof sketch and the full version is provided in Appendix A.4.
We define the set of queues Q(t) = [Qn(t) : n ∈[N]] and a Lyapunov function as V(Q(t)) =
P

n∈[N] Qn(t)2. For simplicity, we use Dn(t) for Dn(t|Skn,t,t) and D∗
n(t) for Dn(t|Sk∗
n,t,t) when
there is no confusion. Then we analyze the Lyapunov drift as follows:
X

t∈[T ]
V(Q(t + 1)) −V(Q(t))

=
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2 −Qn(t)2

+
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2.
(3)

For the first two terms in Eq.(3), with Assumption 3, we can show that
X

t∈[T ]

X

n∈[N]
E[(Qn(t) + An(t) −D∗
n(t))2 −Qn(t)2] ≤−
X

t∈[T ]

X

n∈[N]
2ϵE[Qn(t)] + 2 min{N, K}T,

(4)
For the last two terms in Eq.(3), we have

E
h X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2i

≤2E
h X

t∈[T ]

X

n∈[N]
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)
i
+ 5 min{N, K}T,
(5)

6


---Page Break---
where the first term of the last inequality is closely related to the regret analysis of the bandit
strategy. In the following, we focus on analyzing the last term in Eq.(5). We define events E1
t =
{∥ˆθk,t−θ∗
k∥Vk,t ≤βt for all k ∈[K]} and E2
n,t = {maxm∈Sk,t ∥xm∥V −1
k,t ≤C2ϵ/2βt for k = kπ
n,t}

for some constant C2 > 0. We can show that E1
t holds with high probability so here we only consider
the case when E1
t holds. Then we have

E
h X

t∈[T ]

X

n∈[N]
µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)
i

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1(E2
n,t) + 1((E2
n,t)c))].
(6)

Then for the first term of Eq.(6), from the UCB strategy and E2
n,t, we can show that
X

t∈[T ]
E
h X

n∈[N]
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E2
n,t)
i
≤C2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] .

(7)

Now we provide a bound for the second term of Eq.(6). For some constant C3 > 0, we can show that
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
(ϵ/C3)E [Qn(t)] + O
N 2K2β4
T
κ2ϵ5


.
(8)

By putting the results of Eqs. (3), (4), (5), (6), (7), (8) altogether, we can obtain

E
h X

t∈[T ]
V(Q(t + 1)) −V(Q(t))
i

≤7 min{N, K}T + 2(C2 + (1/C3) −1)ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + O(N log(T)) + O
N 2K2β4
T
κ2ϵ5


.

Finally, with positive constants C2, C3 > 0 satisfying C2 + (1/C3) < 1, from V(Q(1)) = 0 and
V(Q(T + 1)) ≥0, by using telescoping for the above inequality and rearrangement, we can conclude
the proof by 1

T
P

t∈[T ]
P

n∈[N] E[Qn(t)] = O( min{N,K}

ϵ
+ d2N2K2

κ4ϵ6
polylog(T )

T
).

5.1.2
Regret Analysis of UCB-QMB
In addition to the stability analysis, we examine the cumulative regret of UCB-QMB (Algorithm 1).
The regret is defined as the discrepancy between the performance of the oracle policy of MaxWeight
π∗, which operates with the knowledge of the true parameters θk’s, and that of our policy π. Given
the queue lengths at each time t, we denote the oracle assignments as

{S∗
k,t}k∈[K] =
argmax
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
Qn(t)µ(n|Sk, θk).

We show that this oracle policy achieves stability in Proposition 1. For simplicity, we use k∗
n,t for
kπ∗
n,t. Then, the cumulative regret under π is defined as

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E
h
(µ(n|S∗
k∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)
i
.
(9)

We define Qmax = E[maxt∈[T ],n∈[N] Qn(t)]. Then the algorithm achieves the following regret
bound.

Theorem 2. The policy π of Algorithm 1 achieves a regret bound of

Rπ(T) = e
O

min
d

κ

√

KTQmax, N
 dK

κ2ϵ3

1/4
T 3/4

.

7


---Page Break---
We emphasize that our algorithms achieve a sublinear regret bound, even in the worst-case scenario
regarding queue lengths from the minimum in regret. In contrast, [23] achieves a regret bound of
eO(max{
√

TQmax, T 3/4}) for a stationary setting, where the worst-case bound is not guaranteed to
be sublinear from the maximum in regret.

Proof sketch. Here we provide a proof sketch and the full version is provided in Appendix A.5.
We first provide the proof for regret bound of Rπ(T)
=
e
O( d

κ
√

KTQmax).
We define
event E1
t
=
{∥ˆθk,t −θ∗
k∥Vk,t
≤
βt ∀k
∈
[K]} which holds with a high probabil-
ity.
Therefore, here we only consider the case when Et holds.
Then we can show that
P

n∈[N] E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t)−µ(n|Skn,t,t, θkn,t))] ≤P

n∈[N] E[Qn(t)(eµUCB
t
(n|Skn,t,t)−
µ(n|Skn,t,t, θkn,t))] ≤P
k∈[K] E[2βt maxn∈Sk,t ∥xn∥V −1
k,t Qn(t)].
From the inequality, with
PT
t=1 maxn∈Sk,t ∥xn∥2
V −1
k,t ≤(4d/κ) log(1 + (TL/dλ)), we have

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))]

≤2E
h
max
t∈[T ],n∈[N] Qn(t)βT
s

KT
X

t∈[T ]

X

k∈[K]
max
l∈Sk,t ∥xl∥2
V −1
k,t

i
= e
O
d

κ

√

KTQmax

.(10)

Now we provide the proof for the worst-case regret bound of Rπ(T) = e
O

N
  dK

κ2ϵ3
1/4 T 3/4
in

the following. We additionally define event E2
n,t = {maxm∈Sk,t ∥xm∥V −1
k,t ≤ζ for k = kn,t} for

some constant C2 > 0. Under E1
t , we have

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1(E2
n,t) + 1((E2
n,t)c))].

(11)

Then for the first term of Eq.(11), we have

X

t∈[T ]
E



X

n∈[N]
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
2βt∥xn∥V −1
kn,t,tQn(t)1(E2
n,t)



≤
X

t∈[T ]

X

n∈[N]
2βtζE [Qn(t)] ,
(12)

where the last inequality is obtained from E2
n,t. By analyzing the selected number of agent n with
(E2
n,t)c, we can show that
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
ζβT E [Qn(t)] + O
 NK

κζ3βT


.
(13)

By putting the results of Eqs. (11), (12), (13), and Theorem 1, by setting ζ = (ϵK/Tκβ2
T )1/4, for
large enough T, we have

Rπ(T) = O



ζβT
X

t∈[T ]

X

n∈[N]
E[Qn(t)] + NK

κζ3βT
+ N log(T)



= e
O

 

N
 dK

κ2ϵ3

1/4
T 3/4
!

,

(14)

which conclude the proof combined with Eq.(10).

8


---Page Break---
Algorithm 2 Thompson Sampling-Queueing Matching Bandit (TS-QMB)
Input: λ, M, κ, C1 > 0
for t = 1, . . . , T do

for k ∈[K] do

ˆθk,t ←argminθ∈Θ gk,t−1(ˆθk,t−1)⊤θ + 1

2∥θ −ˆθk,t−1∥2
Vk,t with (1)

Sample {eθ(i)
k,t}i∈[M] independently from N(ˆθk,t, β2
t V −1
k,t )

{Sk,t}k∈[K] ←
argmax
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
Qn(t)eµT S
t
(n|Sk, {eθ(i)
k,t}i∈[M]) with (15)

Offer {Sk,t}k∈[K] and observe preference feedback yn,t ∈{0, 1} for all n ∈Sk,t, k ∈[K]

5.2
Thompson Sampling-based Algorithm
Here, we propose an algorithm based on Thompson Sampling, TS-QMB (Algorithm 2). As in the
previous algorithm, we construct the estimator as ˆθk,t = argminθ∈Θ gk,t−1(ˆθk,t−1)⊤θ + 1

2∥θ −
ˆθk,t−1∥2
Vk,t. To facilitate exploration, we sample several eθ(i)
k,t for i ∈[M] from a Gaussian distribution

of N(ˆθk,t, β2
t V −1
k,t ) and construct the Thompson Sampling (TS) index for assortment Sk as

eµT S
t
(n|Sk, {eθ(i)
k,t}i∈[M]) =
exp(hT S
n,k,t)

1 + P

m∈Sk exp(hT S
m,k,t),
(15)

where hT S
n,k,t = maxi∈[M] x⊤
n eθ(i)
k,t and βt = C1
q

λ + d

κ log(1 + tLK

dλ ) for some constant C1 > 0.

Then we utilize the MaxWeight with TS indexes. We set λ = 1 and M = ⌈1 −
log(KL)
log(1−1/4√eπ)⌉.

5.2.1
Stability Analysis of TS-QMB
Here we provide stability analysis for TS-QMB (Algorithm 2).

Theorem
3.
The
average
queue
length
of
Algorithm
2
is
bounded
as
Q(T)
=

O

min{N,K}

ϵ
+ d4N2K2

κ4ϵ6
polylog(T )

T

, which implies that the algorithm achieves stability as

lim
T →∞Q(T) = O
min{N, K}

ϵ


.

Proof. The proof is provided in Appendix A.6

5.2.2
Regret Analysis of TS-QMB
We provide a regret analysis of TS-QMB (Algorithm 2) for the regret definition of (9) in the following.

Theorem 4. The policy π of Algorithm 2 achieves a regret bound of

Rπ(T) = e
O

 

min

(
d3/2

κ

√

KTQmax, N
d2K

κ2ϵ3

1/4
T 3/4
)!

.

Proof. The proof is provided in Appendix A.7

We note that the performance of Algorithm 1 and Algorithm 2 in the analysis results shows similar
trends. However, the TS-based Algorithm 2 incurs a loss with respect to d compared to the UCB-based
Algorithm 1, as commonly seen in previous TS-based algorithms [5, 2, 43].

Here, we briefly discuss the combinatorial optimization of argmax{Sk}k∈[K]∈M(Nt)
P

k∈[K] fk(Sk)
for some function fk : S ⊂[N] →R in our algorithms. The exact optimization can be expensive
due to its NP-hard nature. To address this, we can utilize the technique of α-approximation oracle
with 0 ≤α ≤1, first introduced in Kakade et al. [25], which is deferred to Appendix A.9.

9


---Page Break---
0
5000
10000 15000 20000
Time step t

0.0

0.2

0.4

0.6

0.8

1.0

(t)

1e2
Average Queue Length

0
5000
10000 15000 20000
Time step t

0.0

0.5

1.0

1.5

2.0

(t)

1e5
Regret

ETC-GS
UCB-QMB (Algorithm 1)

MaxWeight-UCB
TS-QMB (Algorithm 2)

Q-UCB
Oracle (MaxWeight)

DAM-UCB

Figure 2: Experimental results for (left) average queue length and (right) regret

6
Experiments

Here, we provide experimental results to demonstrate the performance of our algorithms.3 For the
synthetic experiments, we consider N = 4, K = 2, L = 2, and d = 2. Each element in xn and
θk is uniformly generated from [0, 1] and then normalized, and λn’s are determined with ϵ = 0.1.
Even though no dedicated benchmark exists for our queueing matching scenario, we compare our
algorithms with previously suggested ones for queueing bandits or matching bandits: Q-UCB [32],
DAM-UCB [15], and MaxWeight-UCB [52] for multi-queue multi-server bandits with asymmetric
service rates and ETC-GS [34] for matching bandits. In Figure 2, we can observe that our algorithms
(Algorithms 1 and 2) outperform the previously suggested ones except for the oracle (MaxWeight)
operated under known (latent) service rates. We demonstrate that our algorithms achieve stability,
similar to the oracle in the left figure, which matches the results of our stability analysis (Theorems 1
and 3). Regarding regret shown in the right figure, the previously suggested algorithms exhibit
superlinear performance due to the increasing Qn(t), while our algorithms show relatively small
regret (Theorems 2 and 4). Additional experiments can be found in Appendix A.10.

7
Conclusion

In this paper, we introduce a novel framework for queueing matching bandits with preference
feedback. To achieve stability in this framework, we propose UCB and TS-based algorithms uti-
lizing the MaxWeight strategy. The algorithms achieve system stability with an average queue
length bound of O(min{N, K}/ϵ). Furthermore, the algorithms achieve sublinear regret bounds of
e
O(min{
√

TQmax, T 3/4}). Lastly, we demonstrate our algorithms using synthetic datasets.

8
Acknowledgements

JK was supported by the Global-LAMP Program of the National Research Foundation of Korea (NRF)
grant funded by the Ministry of Education (No. RS-2023-00301976). MO was supported by the NRF
grant funded by the Korea government(MSIT) (No. 2022R1C1C1006859 and 2022R1A4A1030579)
and by AI-Bio Research Grant through Seoul National University.

References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. Advances in neural information processing systems, 24, 2011.

[2] Marc Abeille and Alessandro Lazaric. Linear thompson sampling revisited. In Artificial
Intelligence and Statistics, pages 176–184. PMLR, 2017.

3The
source
code
is
available
at
https://github.com/junghunkim7786/
Queueing-Matching-Bandits-with-Preference-Feedback

10


---Page Break---
[3] Marc Abeille, Louis Faury, and Clément Calauzènes. Instance-wise minimax-optimal algorithms
for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pages
3691–3699. PMLR, 2021.

[4] Philipp Afeche, Rene Caldentey, and Varun Gupta. On the optimal design of a bipartite matching
queueing system. Operations Research, 70(1):363–401, 2022.

[5] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear
payoffs. In International conference on machine learning, pages 127–135. PMLR, 2013.

[6] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Mnl-bandit: A dynamic
learning approach to assortment selection. arXiv preprint arXiv:1706.03880, 2017.

[7] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Thompson sampling for
the mnl-bandit. In Conference on learning theory, pages 76–78. PMLR, 2017.

[8] Urtzi Ayesta, Peter Jacko, and Vladimir Novak. Scheduling of multi-class multi-server queueing
systems with abandonments. Journal of Scheduling, 20:129–145, 2017.

[9] Soumya Basu, Karthik Abinav Sankararaman, and Abishek Sankararaman. Beyond log2(t)
regret for decentralized bandits in matching markets. In International Conference on Machine
Learning, pages 705–715. PMLR, 2021.

[10] Gruia Calinescu, Chandra Chekuri, Martin Pal, and Jan Vondrák. Maximizing a monotone
submodular function subject to a matroid constraint. SIAM Journal on Computing, 40(6):
1740–1766, 2011.

[11] Xi Chen, Akshay Krishnamurthy, and Yining Wang. Robust dynamic assortment optimization
in the presence of outlier customers. Operations Research, 2023.

[12] Tuhinangshu Choudhury, Gauri Joshi, Weina Wang, and Sanjay Shakkottai. Job dispatching
policies for queueing systems with unknown service rates. In Proceedings of the Twenty-second
International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile
Networks and Mobile Computing, pages 181–190, 2021.

[13] Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic
algorithms for logistic bandits. In International Conference on Machine Learning, pages
3052–3060. PMLR, 2020.

[14] Louis Faury, Marc Abeille, Kwang-Sung Jun, and Clément Calauzènes. Jointly efficient and
optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence
and Statistics, pages 546–580. PMLR, 2022.

[15] Daniel Freund, Thodoris Lykouris, and Wentao Weng. Efficient decentralized multi-agent
learning in asymmetric queuing systems. In Conference on Learning Theory, pages 4080–4084.
PMLR, 2022.

[16] Bernhard Fuchs, Winfried Hochstättler, and Walter Kern. Online matching on a line. Theoretical
Computer Science, 332(1-3):251–264, 2005.

[17] Jason Gaitonde and Éva Tardos. Stability and learning in strategic queuing systems. In
Proceedings of the 21st ACM Conference on Economics and Computation, pages 319–347,
2020.

[18] Buddhima Gamlath, Michael Kapralov, Andreas Maggiori, Ola Svensson, and David Wajc.
Online matching with general arrivals. In 2019 IEEE 60th Annual Symposium on Foundations
of Computer Science (FOCS), pages 26–37. IEEE, 2019.

[19] Kevin D Glazebrook and José Niño-Mora. Parallel scheduling of multiclass m/m/m queues:
Approximate and heavy-traffic optimization of achievable performance. Operations Research,
49(4):609–623, 2001.

[20] J Michael Harrison. Heavy traffic analysis of a system with parallel servers: asymptotic
optimality of discrete-review policies. The Annals of Applied Probability, 8(3):822–848, 1998.

11


---Page Break---
[21] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex
optimization. Machine Learning, 69(2):169–192, 2007.

[22] Wei-Kang Hsu, Jiaming Xu, Xiaojun Lin, and Mark R Bell. Integrated online learning and
adaptive control in queueing systems with uncertain payoffs. Operations Research, 70(2):
1166–1181, 2022.

[23] Jiatai Huang, Leana Golubchik, and Longbo Huang. When lyapunov drift based queue schedul-
ing meets adversarial bandit learning. IEEE/ACM Transactions on Networking, 2024.

[24] Kwang-Sung Jun, Aniruddha Bhargava, Robert Nowak, and Rebecca Willett. Scalable gen-
eralized linear bandits: Online computation and hashing. Advances in Neural Information
Processing Systems, 30, 2017.

[25] Sham M Kakade, Adam Tauman Kalai, and Katrina Ligett. Playing games with approximation
algorithms. In Proceedings of the thirty-ninth annual ACM symposium on Theory of computing,
pages 546–555, 2007.

[26] Michael Kapralov, Ian Post, and Jan Vondrák. Online submodular welfare maximization:
Greedy is optimal. In Proceedings of the twenty-fourth annual ACM-SIAM symposium on
Discrete algorithms, pages 1216–1225. SIAM, 2013.

[27] Richard M Karp, Umesh V Vazirani, and Vijay V Vazirani. An optimal algorithm for on-line
bipartite matching. In Proceedings of the twenty-second annual ACM symposium on Theory of
computing, pages 352–358, 1990.

[28] Thomas Kesselheim, Klaus Radke, Andreas Tönnis, and Berthold Vöcking. An optimal online
algorithm for weighted bipartite matching and extensions to combinatorial auctions. In European
symposium on algorithms, pages 589–600. Springer, 2013.

[29] Fang Kong and Shuai Li. Player-optimal stable regret for bandit learning in matching markets.
In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA),
pages 1512–1522. SIAM, 2023.

[30] Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Regret of queueing
bandits. Advances in Neural Information Processing Systems, 29, 2016.

[31] Subhashini Krishnasamy, Ari Arapostathis, Ramesh Johari, and Sanjay Shakkottai. On learning
the cµ rule: Single and multi-server settings. Available at SSRN 3123545, 2018.

[32] Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Learning unknown
service rates in queues: A multiarmed bandit approach. Operations research, 69(1):315–330,
2021.

[33] Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.

[34] Lydia T Liu, Horia Mania, and Michael Jordan. Competing bandits in matching markets. In
International Conference on Artificial Intelligence and Statistics, pages 1618–1628. PMLR,
2020.

[35] Lydia T Liu, Feng Ruan, Horia Mania, and Michael I Jordan. Bandit learning in decentralized
matching markets. The Journal of Machine Learning Research, 22(1):9612–9645, 2021.

[36] Nan Liu, Yuhang Ma, and Huseyin Topaloglu. Assortment optimization under the multinomial
logit model with sequential offerings. INFORMS Journal on Computing, 32(3):835–853, 2020.

[37] Avishai Mandelbaum and Alexander L Stolyar. Scheduling flexible servers with convex delay
costs: Heavy-traffic optimality of the generalized cµ-rule. Operations Research, 52(6):836–855,
2004.

[38] Nick McKeown, Adisak Mekkittikul, Venkat Anantharam, and Jean Walrand. Achieving
100% throughput in an input-queued switch. IEEE Transactions on Communications, 47(8):
1260–1267, 1999.

12


---Page Break---
[39] Aranyak Mehta, Amin Saberi, Umesh Vazirani, and Vijay Vazirani. Adwords and generalized
online matching. Journal of the ACM (JACM), 54(5):22–es, 2007.

[40] Aranyak Mehta et al.
Online matching and ad allocation.
Foundations and Trends® in
Theoretical Computer Science, 8(4):265–368, 2013.

[41] Michael J Neely. Queue stability and probability 1 convergence via lyapunov optimization.
arXiv preprint arXiv:1008.3519, 2010.

[42] Michael J Neely. Stability and capacity regions or discrete time queueing networks. arXiv
preprint arXiv:1003.3396, 2010.

[43] Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits.
Advances in Neural Information Processing Systems, 32, 2019.

[44] Min-hwan Oh and Garud Iyengar. Multinomial logit contextual bandits: Provable optimality
and practicality. In Proceedings of the AAAI conference on artificial intelligence, volume 35,
pages 9205–9213, 2021.

[45] Abishek Sankararaman, Soumya Basu, and Karthik Abinav Sankararaman. Dominate or delete:
Decentralized competing bandits with uniform valuation. arXiv preprint arXiv:2006.15166,
2020.

[46] Flore Sentenac, Etienne Boursier, and Vianney Perchet. Decentralized learning in online
queuing systems. Advances in Neural Information Processing Systems, 34:18501–18512, 2021.

[47] Thomas Stahlbuhk, Brooke Shrader, and Eytan Modiano. Learning algorithms for scheduling
in wireless networks with unknown channel statistics. In Proceedings of the Eighteenth ACM
International Symposium on Mobile Ad Hoc Networking and Computing, pages 31–40, 2018.

[48] Thomas Stahlbuhk, Brooke Shrader, and Eytan Modiano. Learning algorithms for minimizing
queue length regret. IEEE Transactions on Information Theory, 67(3):1759–1781, 2021.

[49] Alexander L Stolyar. Maxweight scheduling in a generalized switch: State space collapse and
workload minimization in heavy traffic. The Annals of Applied Probability, 14(1):1–53, 2004.

[50] Leandros Tassiulas and Anthony Ephremides. Stability properties of constrained queueing
systems and scheduling policies for maximum throughput in multihop radio networks. In 29th
IEEE Conference on Decision and Control, pages 2130–2132. IEEE, 1990.

[51] Leandros Tassiulas and Anthony Ephremides. Dynamic server allocation to parallel queues
with randomly varying connectivity. IEEE Transactions on Information Theory, 39(2):466–478,
1993.

[52] Zixian Yang, R Srikant, and Lei Ying. Learning while scheduling in multi-server systems with
unknown statistics: Maxweight with discounted ucb. In International Conference on Artificial
Intelligence and Statistics, pages 4275–4312. PMLR, 2023.

[53] Lijun Zhang, Tianbao Yang, Rong Jin, Yichi Xiao, and Zhi-Hua Zhou. Online stochastic linear
optimization under one-bit feedback. In International Conference on Machine Learning, pages
392–401. PMLR, 2016.

[54] Yirui Zhang, Siwei Wang, and Zhixuan Fang. Matching in multi-arm bandit with collision.
Advances in Neural Information Processing Systems, 35:9552–9563, 2022.

13


---Page Break---
A
Appendix

A.1
Limitations & Discussion
We leave several questions open for future research. Firstly, it remains an open problem to establish
lower bounds for queue lengths and regret. However, we believe that constructing a lower bound
would be much more challenging compared to other parametric bandit problems. Additionally,
improving the dependency on κ for queue length and regret bounds from the structure of MNL would
be an interesting avenue for future work.

Regarding computation efficiency, we note that updating estimators in our algorithms is computa-
tionally efficient because they are based on online updates with convex optimization for updating
estimators. Concerning combinatorial optimization in our algorithms, we can alleviate computational
costs using α-approximation oracle algorithms, as discussed in Appendix A.9. Note that almost
all combinatorial bandit problems, including our proposed matching bandit framework, involve a
combinatorial optimization step which often relies on some type of approximation optimization
oracles. Hence, this is not a particular limitation specific to our study. However, developing an
improved approximation oracle would also be an interesting direction for future research.

A.2
Discussion on Traffic Slackness Parameter ϵ
As in the closely related works of [15, 52, 23], the traffic slackness remains constant regardless of the
number of agents N and the number of servers K as ϵ = ϵ0 for some 0 < ϵ0 < 1. However, this may
not align with intuition, as a large K might be beneficial in terms of traffic slackness, while a large
N could have the opposite effect. To address this, we can consider ϵ = ϵ0
min(N,K)

N
. This reflects
that when N ≥K, due to the lack of servers, increasing K is critical for increasing traffic slackness,
while increasing N could decrease it. When N < K implying there are enough servers, however, the
value of N doesn’t impact the traffic slackness because each agent can be assigned to at most one
server at each time, and there are sufficient servers to handle the agents.

If we consider the traffic slackness as ϵ = ϵ0
min{N,K}

N
, the oracle strategy of MaxWeight achieves
Q(T) = O( N

ϵ0 ) from Proposition 1. This result shows that as N increases, causing the traffic
slackness to decrease, the average queue length increases. Meanwhile, the positive influence of K on
traffic slackness is neutralized in the average queue length bound due to system variance.

A.3
Proof of Proposition 1
Define the set of queues Q(t) = [Qn(t) : n ∈[N]] and a Lyapunov function as V(Q(t)) =
P

n∈[N] Qn(t)2. For simplicity, we use Dn(t) for Dn(t|Skn,t,t) when there is no confusion. We
observe that P

n∈[N] E[An(t)] ≤P

n∈[N] λn ≤P

n∈[N] µ(n|Sk, θk) ≤K for some Sk from
Assumption 3 and P

n∈[N] λn ≤N. This implies P

n∈[N][An(t)] ≤min{N, K}. We also have
P
n∈[N] E[Dn(t)] = P
n∈[N] E[µ(n|Sk,t, θk)] ≤min{K, N}. Then we have

E[V(Q(t + 1)) −V(Q(t))]

= E



X

n∈[N]
(Qn(t) −An(t) + Dn(t))+2 −Qn(t)2





≤E



X

n∈[N]
(Qn(t) −An(t) + Dn(t))2 −Qn(t)2





≤E



2
X

n∈[N]
Qn(t)An(t) −2
X

n∈[N]
Qn(t)Dn(t) −2
X

n∈[N]
An(t)Dn(t) +
X

n∈[N]
An(t)2 +
X

n∈[N]
Dn(t)2





≤E



2
X

n∈[N]
Qn(t)An(t) −2
X

n∈[N]
Qn(t)Dn(t) +
X

n∈[N]
An(t) +
X

n∈[N]
Dn(t)





≤E



2
X

n∈[N]
Qn(t)An(t) −2
X

n∈[N]
Qn(t)Dn(t) + 2 min{N, K}




(16)

14


---Page Break---
From Assumption 3, we define the corresponding assortments as {S′
k}k∈[K] ∈M(N), which satisfies
λn + ϵ ≤µ(n|S′
k, θk) for all n ∈S′
k and k ∈[K]. Then we define the set of non-empty queues in S′
k
at time t as S′
k,t = {n ∈S′
k : Qn(t) ̸= 0}. From the property of the MNL function, we can observe
that µ(n|S′
k, θk) ≤µ(n|S′
k,t, θk) for all n ∈S′
k,t. We also note that {S′
k,t}k∈[K] ∈M(Nt). Then
we have
X

n∈[N]
(λn + ϵ)Qn(t) ≤
X

k∈[K]

X

n∈S′
k
µ(n|S′
k, θk)Qn(t)

≤
X

k∈[K]

X

n∈S′
k,t
µ(n|S′
k,t, θk)Qn(t)

≤
X

k∈[K]

X

n∈Sk,t
µ(n|Sk,t, θk)Qn(t),

where the second inequality is obtained from µ(n|S′
k, θk)Qn(t) ≤µ(n|S′
k,t, θk)Qn(t) when n ∈Nt,
and otherwise µ(n|S′
k, θk)Qn(t) = µ(n|S′
k,t, θk)Qn(t) = 0.

This implies

E



X

k∈[K]

X

n∈Sk,t
Qn(t)Dn(t)



= E



E



X

k∈[K]

X

n∈Sk,t
Qn(t)Dn(t)

Q(t)









= E



X

k∈[K]

X

n∈Sk,t
Qn(t)µ(n|Sk,t, θk)





≥E



X

k∈[K]

X

n∈S′
k,t
Qn(t)µ(n|S′
k,t, θk)





≥
X

n∈[N]
(λn + ϵ)E[Qn(t)].
(17)

From Eqs.(16) and (17), we have

E[V(Q(t + 1)) −V(Q(t))]

≤E



2
X

n∈[N]
Qn(t)An(t) −2
X

n∈[N]
Qn(t)Dn(t)



+ 2 min{N, K}

≤E







2
X

n∈[N]
Qn(t)An(t) −2
X

n∈[N]
Qn(t)Dn(t)
Q(t)







+ 2 min{N, K}

≤E



2
X

n∈[N]
λnQn(t) −2(λn + ϵ)Qn(t)



+ 2 min{N, K}

≤−E



2ϵ
X

n∈[N]
Qn(t)



+ 2 min{N, K},

which implies from V(Q(T + 1)) ≥0 and V(Q(1)) = 0,
X

t∈[T ]
E[V(Q(t + 1) −V(Q(t))] = V(Q(T + 1))

≤−
X

t∈[T ]
E[2ϵ
X

n∈[N]
Qn(t)] + 2 min{N, K}T.

Finally, we can conclude that Q(T) = (1/T) P

t∈[T ] E[P

n∈[N] Qn(t)] ≤2 min{N, K}/ϵ.

15


---Page Break---
A.4
Proof of Theorem 1
We first define the set of queues Q(t) = [Qn(t) : n ∈[N]] and a Lyapunov function as V(Q(t)) =
P

n∈[N] Qn(t)2. For simplicity, we use Dn(t) for Dn(t|Skn,t,t) and D∗
n(t) for Dn(t|Sk∗
n,t,t) when
there is no confusion. Then we analyze the Lyapunov drift as follows.
X

t∈[T ]
V(Q(t + 1)) −V(Q(t))

=
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −Qn(t)2

=
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2 −Qn(t)2

+
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2. (18)

We observe that P

n∈[N] E[An(t)] ≤P

n∈[N] λn ≤P

n∈[N] µ(n|Sk, θk) ≤K for some Sk from
Assumption 3 and P

n∈[N] λn ≤N. This implies P

n∈[N][An(t)] ≤min{N, K}. We also have
P

n∈[N] E[Dn(t)] = P

n∈[N] E[µ(n|Sk,t, θk)] ≤min{K, N}. For the first two terms in Eq.(18), by
following the same procedure of Eqs.(16) and (17), we can obtain
X

t∈[T ]

X

n∈[N]
E[(Qn(t) + An(t) −D∗
n(t))2 −Qn(t)2]

≤
X

t∈[T ]

X

n∈[N]
2E[(λn −µ(n|S∗
k,t, θk))Qn(t)] + 2 min{N, K}T

≤−
X

t∈[T ]

X

n∈[N]
2ϵE[Qn(t)] + 2 min{N, K}T,
(19)

where the last inequality is obtained using Assumption 3. For the last two terms in Eq.(18), we have
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −D∗
n(t))2

≤
X

t∈[T ]

X

n∈[N]
(D∗
n(t) −Dn(t))(2Qn(t) + 2An(t) −D∗
n(t) −Dn(t))

≤2
X

t∈[T ]

X

n∈[N]
An(t)(D∗
n(t) −Dn(t)) −
X

t∈[T ]

X

n∈[N]
D∗
n(t)2 +
X

t∈[T ]

X

n∈[N]
Dn(t)2

+ 2
X

t∈[T ]

X

n∈[N]
(D∗
n(t) −Dn(t))Qn(t)

≤4
X

t∈[T ]

X

n∈[N]
An(t) + 2
X

t∈[T ]

X

n∈[N]
(D∗
n(t) −Dn(t))Qn(t) + min{N, K}T

≤2
X

t∈[T ]

X

n∈[N]
(D∗
n(t) −Dn(t))Qn(t) + 5 min{N, K}T.
(20)

Now we provide a bound for Eq.(20). We first provide some lemmas for the concentration of
estimators.

Using the above lemma, we can show the following lemma of the concentration.

Lemma 1 (Lemma 9 in Oh and Iyengar [44]). For t ≥1 and some constant C1 > 0, with probability
at least 1 −1/t2, for all k ∈[K] we have

∥ˆθk,t −θk∥Vk,t ≤βt.

Proof. For the completeness, we provide the proof in Appendix A.8.

16


---Page Break---
Define event E1
t
=
{∥ˆθk,t −θ∗
k∥Vk,t
≤
βt
for all k
∈
[K]} where βt
=

C1
q

λ + d

κ log(1 + tLK/dλ), which holds with high probability as P(E1
t ) ≥1 −1/t2 from the
above lemma. We also define E2
n,t = {maxm∈Sk,t ∥xm∥V −1
k,t ≤C2ϵ/2βt for k = kn,t} for some
constant C2 > 0. Then, we have

X

t∈[T ]

X

n∈[N]
E[(D∗
n(t) −Dn(t)Qn(t)]

=
X

t∈[T ]

X

n∈[N]
E[E[(D∗
n(t) −Dn(t)Qn(t)|Qn(t)]]

= E[
X

t∈[T ]

X

n∈[N]
µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
n,t)c))].(21)

Now we provide a lemma for bounding the first term in Eq.(21).

Lemma 2. Under E1
t , for any n ∈[N], we have eµUCB
t
(n|Skn,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t) ≤
2βt∥xn∥V −1
kn,t,t.

Proof. Let un,k,t = x⊤
n θk. Under E1
t , for any n ∈[N] and k ∈[K] we have x⊤
n ˆθk,t −βt∥xn∥V −1
k,t ≤

x⊤
n θk ≤x⊤
n ˆθk,t + βt∥xn∥V −1
k,t , which implies 0 ≤hUCB
n,k,t −un,k,t ≤2βt∥xn∥V −1
k,t . Then by the

mean value theorem, there exists ¯un,k,t = (1 −c)hUCB
n,k,t + cun,k,t for some c ∈(0, 1) satisfying, for
any n ∈Sk, Sk ⊂[N], and k ∈[K],

eµUCB
t
(n|Sk, ˆθk,t) −µ(n|Sk, θk) =
exp(hUCB
n,k,t )

1 + P

m∈Sk exp(hUCB
m,k,t) −
exp(un,k,t)
1 + P

m∈Sk exp(um,k,t)

= ∇vn

 
exp(vn)
1 + P

m∈Sk exp(vm)

! 
vn=¯un,k,t
(hUCB
n,k,t −un,k,t)

≤
exp(¯un,k,t)(hUCB
n,k,t −un,k,t)
1 + P

n∈Sk exp(¯un,k,t)

≤hUCB
n,k,t −un,k,t
≤2βt∥xn∥V −1
k,t .

Since x/(1 + x) is a non-decreasing function for x > −1 and x⊤
n θk∗
n,t ≤x⊤
n ˆθk∗
n,t,t + βt∥xn∥V −1
k∗
n,t,t
under E1
t , we have µ(n|Sk∗
n,t,t, θk∗
n,t) ≤eµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t). Then for the first term of Eq.(21),

17


---Page Break---
we have

X

t∈[T ]
E



X

n∈[N]
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(eµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(eµUCB
t
(n|Skn,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
2βt∥xn∥V −1
kn,t,tQn(t)1(E1
t ∩E2
n,t)





≤C2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] ,
(22)

where the second inequality comes from the UCB strategy of the algorithm, the last second inequality
is obtained from Lemma 2, and the last inequality is obtained from E2
n,t.

Now we provide a bound for the second term of Eq.(21). We first have

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E1
t )c)
i

≤
X

t∈[T ]

X

n∈[N]
E

Qn(t)1((E1
t )c)


≤
X

t∈[T ]

X

n∈[N]
tP((E1
t )c)

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)),
(23)

where the second inequality is obtained from Qn(t) ≤t.

Here we utilize some techniques introduced in Freund et al. [15]. Let Tn be the set of time steps
t ∈[T] such that Qn(t) ̸= 0 and let eT = P

n∈[N]
P

t∈Tn 1((E2
n,t)c) and h = ⌈C3eT /ϵ⌉for some
constant C3 > 0. Then if t ≤h, we have Qn(t) ≤t ≤h. Otherwise, we have

Qn(t) ≤

t
X

s=t−h+1
(1/h)(Qn(s) + (t −s)) ≤(1/h)

t
X

s=1
Qn(s) + (1/h)h2 = (1/h)

t
X

s=1
Qn(s) + h.

18


---Page Break---
Then, from the above inequality, we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

=
X

n∈[N]

X

t∈Tn
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

n∈[N]

X

t∈Tn
E[Qn(t)1((E2
n,t)c)]

≤
X

n∈[N]

X

t∈Tn
E[((ϵ/C3eT )

T
X

s=1
Qn(s) + 2C3eT /ϵ)1((E2
n,t)c)]

≤E



(ϵ/C3)
X

n∈[N]

T
X

s=1
Qn(s)



+ E

2C3e2
T /ϵ


=
X

t∈[T ]

X

n∈[N]
(ϵ/C3)E [Qn(t)] + 2C3E[e2
T ]/ϵ.
(24)

Now we provide a bound for E[e2
T ].
Define Nn,k(t) = Pt−1
s=1 1(n ∈Sk,s) and eVkn,t,t =
κ
2
Pt−1
s=1
P

n∈Sk,s xnx⊤
n . Then, we have

eT =
X

n∈[N]

X

t∈Tn
1((E2
n,t)c)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥V −1
kn,t,t ≥C2ϵ/2βt)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥eV −1
kn,t,t ≥C2ϵ/2βt)

≤
X

n∈[N]

X

t∈Tn
1(1/Nn,kn,t(t) ≥(κ/2)(C2ϵ/2βt)2)

≤
X

n∈[N]

X

t∈Tn
1(Nn,kn,t(t) ≤(2/κ)(2βt/C2ϵ)2)

≤
X

n∈[N]

X

k∈[K]

X

t∈Tn
1(Nn,k(t) ≤(2/κ)(2βT /C2ϵ)2 and kn,t = k)

≤NK(2/κ)(2βT /C2ϵ)2.

From the above we have E[e2
T ] ≤64N 2K2β4
T /κ2(C2ϵ)4. Then from Eq.(24), we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
(ϵ/C3)E [Qn(t)] + O(N 2K2β4
T /κ2ϵ5).
(25)

19


---Page Break---
By putting the results of Eqs. (18), (19), (20), (21), (22), (23), (25) altogether, we can obtain

E



X

t∈[T ]
V(Q(t + 1)) −V(Q(t))





≤7 min{N, K}T −2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + 2
X

t∈[T ]

X

n∈[N]
E
h
(Dn(t|Sk∗
n,t,t) −Dn(t|Skn,t,t))Qn(t)
i

≤7 min{N, K}T −2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + 2C2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)]

+ O(N log(T)) + (2ϵ/C3)
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + O(N 2K2β4
T /κ2ϵ5)

≤7 min{N, K}T + 2(C2 + (1/C3) −1)ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + O(N log(T)) + O(N 2K2β4
T /κ2ϵ5).

Finally, with positive constants C2, C3 > 0 satisfying C2 + (1/C3) < 1, from V(Q(1)) = 0 and
V(Q(T + 1)) ≥0, by using telescoping for the above inequality, we can conclude the proof by
1
T

X

t∈[T ]

X

n∈[N]
E[Qn(t)] = O
min{N, K}

ϵ
+ 1

T
d2N 2K2

κ4ϵ6
polylog(T)

.
(26)

A.5
Proof of Theorem 2

We first provide the proof for regret bound of Rπ(T) = e
O

d
κ
√

KTQmax

.

We define event Et = {∥ˆθk,t −θ∗
k∥Vk,t ≤βt ∀k ∈[K]} which holds at least probability of 1 −1/t2
from Lemma 1.

Lemma 3. Under E1
t , for any Sk ⊂[N], we have
X

n∈Sk
(eµUCB
t
(n|Sk, ˆθk,t) −µ(n|Sk, θk))Qn(t) ≤2βt max
n∈Sk ∥xn∥V −1
k,t Qn(t).

Proof. Let un,k,t = x⊤
n θk. Under E1
t , for any n ∈[N] and k ∈[K] we have x⊤
n ˆθk,t −βt∥xn∥V −1
k,t ≤

x⊤
n θk ≤x⊤
n ˆθk,t + βt∥xn∥V −1
k,t , which implies 0 ≤hUCB
n,k,t −un,k,t ≤2βt∥xn∥V −1
k,t . Then by the

mean value theorem, there exists ¯un,k,t = (1 −c)hUCB
n,k,t + cun,k,t for some c ∈(0, 1) satisfying, for
any S ⊂[N],X

n∈Sk
(eµUCB
t
(n|Sk, ˆθk,t) −µ(n|Sk, θk))Qn(t)

=
X

n∈Sk

 
exp(hUCB
n,k,t )

1 + P
m∈Sk exp(hUCB
m,k,t) −
exp(un,k,t)
1 + P

m∈Sk exp(um,k,t)

!

Qn(t)

=
X

n∈Sk
∇vn

 
exp(vn)
1 + P

m∈Sk exp(vm)

! 
vn=¯un,k,t
(hUCB
n,k,t −un,k,t)Qn(t)

=
(1 + P

n∈Sk exp(¯un,k,t))(P

n∈Sk exp(¯un,k,t)(hUCB
n,k,t −un,k,t)Qn(t))
(1 + P

n∈Sk exp(¯un,k,t))2

−
(P
n∈Sk exp(¯un,k,t))(P
n∈Sk exp(¯un,k,t)(hUCB
n,k,t −un,k,t)Qn(t))
(1 + P

n∈Sk exp(¯un,k,t))2

≤
X

n∈Sk

exp(¯un,k,t)
1 + P

m∈Sk exp(¯um,k,t)(hUCB
n,k,t −un,k,t)Qn(t)

≤max
n∈Sk(hUCB
n,k,t −un,k,t)Qn(t)

≤2βt max
n∈Sk ∥xn∥V −1
k,t Qn(t).

20


---Page Break---
Under Et, from Lemma 3, we have
X

n∈Sk,t
(eµUCB
t
(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))Qn(t) ≤2βt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t),
(27)

and since x/(1+x) is a non-decreasing function for x > −1 and x⊤
n θk∗
n,t ≤x⊤
n ˆθ′
k∗
n,t,t+βt∥xn∥V −1
k∗
n,t,t
under E1
t , we have

µ(n|Sk∗
n,t,t, θk∗
n,t) ≤eµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t).
(28)

Now we provide an elliptical potential lemma.

Lemma 4. For any k ∈[K], we have

T
X

t=1
max
n∈Sk,t ∥xn∥2
V −1
k,t ≤(4d/κ) log(1 + (TL/dλ)).

Proof. First, we can show that

det(Vk,t)

= det



Vk,t−1 + (κ/2)
X

n∈Sk,t−1
xnx⊤
n





= det(Vk,t−1) det



Id + (κ/2)
X

n∈Sk,t−1
V −1/2
k,t−1xn(V −1/2
k,t−1xn)⊤





≥det(Vk,t−1)



1 + (κ/2)
X

n∈Sk,t−1
∥xn∥2
V −1
k,t−1





≥det(λId)

t−1
Y

s=1



1 + (κ/2)
X

n∈Sk,s
∥xn∥2
V −1
k,s



= λd
t−1
Y

s=1



1 + (κ/2)
X

n∈Sk,s
∥xn∥2
V −1
k,s





From the above, using the fact that x ≤2 log(1 + x) for any x ∈[0, 1] and (κ/2)∥xn∥2
eV −1
k,s ≤

(κ/2)∥xn∥2
2/λ ≤1 from κ < 1, we have

X

t∈[T ]
max
n∈Sk,t(κ/2)∥xn∥2
V −1
k,t ≤
X

t∈[T ]
min

max
n∈Sk,t(κ/2)∥xn∥2
V −1
k,t−1, 1


≤2
X

t∈[T ]
log



1 +
X

n∈Sk,t
(κ/2)∥xn∥2
V −1
k,t−1





= 2 log
Y

t∈[T ]



1 +
X

n∈Sk,t
(κ/2)∥xn∥2
V −1
k,t−1





≤2 log
det(Vk,T +1)

λd


.
(29)

From Lemma 10 in Abbasi-Yadkori et al. [1] with κ < 1, we can show that

det(Vk,T +1) ≤(λ + (TL/d))d.

Then from the above inequality and Eq.(29), we can conclude the proof.

21


---Page Break---
Then from Eqs.(27), (28), and Lemma 4, we can conclude that

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))]

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))1(E1
t )]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))1((E1
t )c)]

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))1(E1
t )] +
X

t∈[T ]

X

n∈[N]
tP((E1
t )c)]

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)(eµUCB
t
(n|Sk∗
n,t,t, ˆθk∗
n,t,t) −µ(n|Skn,t,t, θkn,t))1(E1
t )] + O(N/T)

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)(eµUCB
t
(n|Skn,t,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t))1(E1
t )] + O(N/T)

=
X

t∈[T ]
E[
X

k∈[K]

X

n∈Sk,t
Qn(t)(eµUCB
t
(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))1(E1
t )] + O(N/T)

≤2E



βT
X

t∈[T ]

X

k∈[K]
max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)



+ O(N/T)

≤2E




max
t∈[T ],n∈[N] Qn(t)βT
X

t∈[T ]

X

k∈[K]
max
n∈Sk,t ∥xn∥V −1
k,t



+ O(N/T)

≤2E




max
t∈[T ],n∈[N] Qn(t)βT
s

KT
X

t∈[T ]

X

k∈[K]
max
n∈Sk,t ∥xn∥2
V −1
k,t



+ O(N/T)

= e
O
d

κ

√

KTQmax


,

where the last equality comes from Lemma 4.

Now we provide the worst-case regret bound of Rπ(T) = e
O

N
  dK

κ2ϵ3
1/4 T 3/4
in the following.

We define event E1
t
=
{∥ˆθk,t −θ∗
k∥Vk,t
≤
βt for all k
∈
[K]} where βt
=

C1
q

λ + d

κ log(1 + tLK/dλ), which holds with high probability as P(E1
t ) ≥1 −1/t2 from
Lemma 1. We also define E2
n,t = {maxm∈Sk,t ∥xm∥V −1
k,t ≤ζ for k = kn,t} for some constant
C2 > 0.

Then we have

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
n,t)c))].

(30)

Since x/(1 + x) is a non-decreasing function for x > −1 and x⊤
n θk∗
n,t ≤x⊤
n ˆθk∗
n,t,t + βt∥xn∥V −1
k∗
n,t,t
under E1
t , we have µ(n|Sk∗
n,t,t, θk∗
n,t) ≤eµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t). Then for the first term of Eq.(30),

22


---Page Break---
we have

X

t∈[T ]
E



X

n∈[N]
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(eµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(eµUCB
t
(n|Skn,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
2βt∥xn∥V −1
kn,t,tQn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]

X

n∈[N]
2βtζE [Qn(t)] ,
(31)

where the second inequality comes from the UCB strategy of the algorithm, the last second inequality
is obtained from Lemma 2, and the last inequality is obtained from E2
n,t.

Now we provide a bound for the second term of Eq.(30). From Eq. (23), we have

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E1
t )c)
i

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)).
(32)

Let Tn be the set of time steps t ∈[T] such that Qn(t) ̸= 0 and let eT = P
n∈[N]
P
t∈Tn 1((E2
n,t)c)
and h = ⌈1/ζβT ⌉. Then if t ≤h, we have Qn(t) ≤t ≤h. Otherwise, we have

Qn(t) ≤

t
X

s=t−h+1
(1/h)(Qn(s) + (t −s)) ≤(1/h)

t
X

s=1
Qn(s) + (1/h)h2 = (1/h)

t
X

s=1
Qn(s) + h.

Then, by following the steps in Eq.(24), we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
ζβT E [Qn(t)] + 2E[eT ]/ζβT .
(33)

23


---Page Break---
Now we provide a bound for E[eT ].
Define Nn,k(t) = Pt−1
s=1 1(n ∈Sk,s) and eVk,t =
κ
2
Pt−1
s=1
P

n∈Sk,s xnx⊤
n . Then, we have

eT =
X

n∈[N]

X

t∈Tn
1((E2
n,t)c)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥V −1
kn,t,t ≥ζ)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥eV −1
kn,t,t ≥ζ)

≤
X

n∈[N]

X

t∈Tn
1(1/Nn,kn,t(t) ≥(κ/2)ζ2)

≤
X

n∈[N]

X

t∈Tn
1(Nn,kn,t(t) ≤2/κζ2)

≤
X

n∈[N]

X

k∈[K]

X

t∈Tn
1(Nn,k(t) ≤2/κζ2 and kn,t = k)

≤2NK/κζ2.
(34)

Then from Eqs.(33), (34), we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
ζβT E [Qn(t)] + O(NK/κζ3βT ).
(35)

By putting the results of Eqs. (30), (31), (32), (35), and Theorem 1, by setting ζ = (ϵK/Tκβ2
T )1/4,
for large enough T, we can obtain

Rπ(T) = O



ζβT
X

t∈[T ]

X

n∈[N]
E[Qn(t)] + NK

κζ3βT
+ N log(T)





= O
ζβT NT

ϵ
+ NK

κζ3βT
+ N log(T)


= O

 
β1/2
T
NT 3/4K1/4

κ1/4ϵ3/4
+ N log(T)

!

= e
O

 

N
 dK

κ2ϵ3

1/4
T 3/4
!

(36)

A.6
Proof of Theorm 3

We first define the set of queues Q(t) = [Qn(t) : n ∈[N]] and a Lyapunov function as V(Q(t)) =
P

n∈[N] Qn(t)2. For simplicity, we use Dn(t) for Dn(t|Skn,t,t), D∗
n(t) for Dn(t|Sk∗
n,t,t), and

eµT S
t
(n|Sk,t) for eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M]) when there is no confusion. Then we analyze the Lya-

24


---Page Break---
punov drift as follows.
X

t∈[T ]
E [V(Q(t + 1)) −V(Q(t))]

=
X

t∈[T ]

X

n∈[N]
E
h
(Qn(t) + An(t) −Dn(t))+2 −Qn(t)2i

=
X

t∈[T ]

X

n∈[N]
E

(Qn(t) + An(t) −D∗
n(t))2 −Qn(t)2

+
X

t∈[T ]

X

n∈[N]
E
h
(Qn(t) + An(t) −Dn(t))+2 −(Qn(t) + An(t) −D∗
n(t))2i

≤7 min{N, K}T −
X

t∈[T ]

X

n∈[N]
2ϵE[Qn(t)] + 2
X

t∈[T ]

X

n∈[N]
E [(D∗
n(t) −Dn(t))Qn(t)] , (37)

where the last inequality can be obtained by following Eqs.(19) and (20).

Define event E1
t
=
{∥ˆθk,t −θk∥Vk,t
≤
βt
for all k
∈
[K]} where βt
=

C1
q

λ + d

κ log(1 + tLK/dλ), which holds with high probability as P(E1
t ) ≥1 −1/t2 from

Lemma 1. We let γt = βt
p

d log(MKt) and filtration Ft−1 be the σ-algebra generated by random
variables before time t.

Lemma 5 (Lemma 10 in Oh and Iyengar [43]). For any given Ft−1, with probability at least
1 −O(1/t2), for all n ∈[N] and k ∈[K], we have

|hT S
n,k,t −x⊤
n ˆθk,t| ≤γt∥xn∥V −1
k,t .

Lemma 6. With probability at least 1 −O(1/t2), for all n ∈[N] and k ∈[K], we have

eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t) ≤γt∥xn∥V −1
k,t .

Proof. From Lemma 5, with probability at least 1 −O(1/t2), we have |hT S
n,k,t −x⊤
n ˆθk,t| ≤
γt∥xn∥V −1
k,t . Let un,k,t = x⊤
n ˆθk,t.
Then by the mean value theorem, there exists ¯un,k,t =

(1 −c)hT S
n,k,t + cun,k,t for some c ∈(0, 1) satisfying, for any n ∈Sk,t and k ∈[K],

eµT S
t
(n|Sk,t) −µ(n|Sk, ˆθk,t) =
exp(hT S
n,k,t)

1 + P

m∈Sk,t exp(hT S
m,k,t) −
exp(un,k,t)
1 + P

m∈Sk,t exp(um,k,t)

= ∇vn

 
P

m∈Sk,t exp(vm)

1 + P

m∈Sk,t exp(vm)

! 
vn=¯un,k,t
(hT S
n,k,t −un,k,t)

≤
exp(¯un,k,t)|hT S
n,k,t −un,k,t|
1 + P

n∈Sk,t exp(¯un,k,t)

≤|hT S
n,k,t −un,k,t|

≤γt∥xn∥V −1
k,t .

Then we define E2
t = {eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t) ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈[K]}, which

holds with probability at least 1 −O(1/t2) from Lemma 6. We also define E3
n,t = {∥xn∥V −1
k,t ≤

25


---Page Break---
ϵ/C2(γt + βt); ∀k ∈[K]} for some constant C2 ≥17√eπ. Then, for bounding Eq.(37), we have

X

t∈[T ]

X

n∈[N]
E[(D∗
n(t) −Dn(t)Qn(t)]

=
X

t∈[T ]

X

n∈[N]
E[E[(D∗
n(t) −Dn(t)Qn(t)|Qn(t)]]

= E[
X

t∈[T ]

X

n∈[N]
µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
t )c) + 1((E3
n,t)c))].

(38)

We provide a bound for the first term of Eq.(38). We first have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t)

+ eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)].

(39)

Recall that eµUCB
t
(n|Sk, ˆθk,t) =
exp(hUCB
n,k,t)
1+P

m∈Sk exp(hUCB
m,k,t). Then we can show that since x/(1 + x) is

a non-decreasing function for x > −1 and x⊤
n ˆθkn,t,t ≤x⊤
n ˆθkn,t,t + βt∥xn∥V −1
kn,t,t under E1
t , with

Lemma 2, we have

µ(n|Skn,t,t, ˆθkn,t,t)−µ(n|Skn,t,t, θkn,t) ≤eµUCB
t
(n|Skn,t,t, ˆθkn,t,t)−µ(n|Skn,t,t, θkn,t) ≤2βt∥xn∥V −1
kn,t,t.

From the above inequality, the last two terms in Eq.(39) are bounded as

X

t∈[T ]

X

n∈[N]
E

eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)


≤
X

t∈[T ]

X

n∈[N]
E

(eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, ˆθkn,t,t)

+ µ(n|Skn,t,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)


≤
X

t∈[T ]

X

n∈[N]
E[(eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, ˆθkn,t,t) + 2βt∥xn∥V −1
kn,t,t)Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

=
X

t∈[T ]

X

k∈[K]
E



X

n∈Sk,t
(eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t) + 2βt∥xn∥V −1
k,t )Qn(t)1(E1
t ∩E2
t ∩E3
n,t)





26


---Page Break---
≤
X

t∈[T ]

X

k∈[K]
E



X

n∈Sk,t
(γt + 2βt)∥xn∥V −1
k,t Qn(t)1(E1
t ∩E2
t ∩E3
n,t)





≤
X

t∈[T ]

X

k∈[K]
E



X

n∈Sk,t

(γt + 2βt)ϵ
C2(γt + βt)Qn(t)1(E1
t ∩E2
t ∩E3
n,t)





≤
X

t∈[T ]

X

k∈[K]
E



X

n∈Sk,t

(γt + 2βt)ϵ
C2(γt + βt)Qn(t)1(E1
t )





=
X

t∈[T ]

X

k∈[K]
E



E



X

n∈Sk,t

(γt + 2βt)ϵ
C2(γt + βt)Qn(t)|E1
t , Ft−1



P(E1
t |Ft−1)



.
(40)

Now we provide a bound for the first two terms in Eq.(39).

We define sets

eΘt =

{θ(i)
k }i∈[M],k∈[K] :
max
i∈[M] x⊤
n θ(i)
k −x⊤
n ˆθk,t

 ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈[K]

and

eΘopt
t
=

{θ(i)
k }i∈[M],k∈[K] :
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])Qn(t)

>
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)

∩eΘt.

Then we define event Et(eθ) = {{eθ(i)
k,t}i∈[M],k∈[K] ∈eΘopt
t
}. Recall hT S
n,k,t = maxi∈[M] x⊤
n eθ(i)
k,t.
Then we have

E



E



X

k∈[K]



X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t) −
X

n∈Sk,t
eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M])Qn(t)



1(E1
t ∩E2
t ∩E3
n,t)|Ft−1









≤E



E







X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt
max
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
eµT S
t
(n|Sk, {θ(i)
k }i∈[M])Qn(t)



1(E1
t ∩E2
t ∩E3
n,t)|Ft−1









= E



E







X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt
max
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
eµT S
t
(n|Sk, {θ(i)
k }i∈[M])Qn(t)



1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)









≤E



E







X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M])

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])



Qn(t)1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)









27


---Page Break---
= E



E




sup

{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t


eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M])

−eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])

Qn(t)1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)


≤E



E




sup

{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t

hT S
n,k,t −max
j∈[M] x⊤
n θ(j)
k

 Qn(t)1(E1
t ∩E2
t ∩E3
n,t)
Ft−1, Et(eθ)









= E



E




sup

{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t

hT S
n,k,t −x⊤
n ˆθk,t + x⊤
n ˆθk,t −max
j∈[M] x⊤
n θ(j)
k



×Qn(t)1(E1
t ∩E2
t ∩E3
n,t)
Ft−1, Et(eθ)
ii

≤2γtE



X

k∈[K]

X

n∈Sk,t
E
h
∥xn∥V −1
k,t Qn(t)1(E1
t ∩E2
t ∩E3
n,t)
Ft−1, Et(eθ)
i




≤2γtE



X

k∈[K]

X

n∈Sk,t
E

ϵ
C2(γt + βt)Qn(t)1(E1
t ∩E2
t ∩E3
n,t)
Ft−1, Et(eθ)




≤2γtE



X

k∈[K]

X

n∈Sk,t
E

ϵ
C2(γt + βt)Qn(t)1(E1
t )
Ft−1, Et(eθ)




=
2γtϵ
C2(γt + βt)E



X

k∈[K]

X

n∈Sk,t
E
h
Qn(t)
Ft−1, Et(eθ), E1
t
i
× P(E1
t |Et(eθ), Ft−1)





=
2γtϵ
C2(γt + βt)E



X

k∈[K]

X

n∈Sk,t
E
h
Qn(t)
Ft−1, Et(eθ), E1
t
i
P(E1
t |Ft−1)



,
(41)

where the first inequality is obtained by the event E2
t , the second inequality is obtained from Et(eθ),
the third inequality can be easily obtained by following some of the proof steps in Lemma 2, the
third last inequality is obtained from the definition of eΘt and event Et(eθ), and the last equality comes
from independence between E1
t and Et(eθ) given Ft−1.

We provide a lemma below for further analysis.

Lemma 7. For all t ∈[T], we have

P



X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t



≥1/4√eπ.

Proof. Given Ft−1, x⊤
n eθ(i)
k,t follows Gaussian distribution with mean x⊤
n ˆθk,t and standard deviation
βt∥xn∥V −1
k,t . Then we have

P

max
i∈[M] x⊤
n eθ(i)
k,t > x⊤
n θk|Ft−1, E1
t


= 1 −P

x⊤
n θ(i)
k,t ≤x⊤
n θk; ∀i ∈[M]|Ft−1, E1
t


= 1 −P

 

Zi ≤x⊤
n θk −x⊤
n ˆθk,t
βt∥xn∥V −1
k,t
; ∀i ∈[M]|Ft−1, E1
t

!

≥1 −P (Z ≤1)M ,

28


---Page Break---
where Zi and Z are standard normal random variables. Then we can show that

P



X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





≥P



X

k∈[K]

X

n∈S∗
k,t
eµT S
t
(n|S∗
k,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





≥P

max
i∈[M] x⊤
n eθ(i)
k,t > x⊤
n θk; ∀n ∈S∗
k,t, ∀k ∈[K]|Ft−1, E1
t



≥1 −LKP(Z ≤1)M

≥1 −LK(1 −1/4√eπ)M

≥
1
4√eπ ,

where the second last inequality is obtained from P(Z ≤1) ≤1 −1/4√eπ using the anti-
concentration of standard normal distribution, and the last inequality comes from M = ⌈1 −
log KL
log(1−1/4√eπ)⌉.

From Lemmas 5 and 7, for t ≥t0 for some constant t0 > 0, we have

P(Et(eθ)|Ft−1, E1
t )

= P



X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk) and {eθ(i)
k,t}i∈[M],k∈[K] ∈eΘt|Ft−1, E1
t





= P



X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





−P({eθ(i)
k,t}i∈[M],k∈[K] /∈eΘt|Ft−1, E1
t )

≥1/4√eπ −O(1/t2)

≥1/8√eπ.

For simplicity of the proof, we ignore the time steps before (constant) t0, which does not affect our
final result. Hence, we have

E[
X

k∈[K]

X

n∈Sk,t
Qn(t)|Ft−1, E1
t ]

≥E[
X

k∈[K]

X

n∈Sk,t
Qn(t)|Ft−1, E1
t , Et(eθ)]P(Et(eθ)|Ft−1, E1
t )

≥E[
X

k∈[K]

X

n∈Sk,t
Qn(t)|Ft−1, E1
t , Et(eθ)]1/8√eπ.
(42)

With (41) and (42), we have

E[(
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk) −
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)|Ft−1]

≤
2γtϵ
C2(γt + βt)E



X

k∈[K]

X

n∈Sk,t
Qn(t)
Ft−1, Et(eθ), E1
t



P(E1
t |Ft−1)

≤16√eπγtϵ

C2(γt + βt)E[
X

k∈[K]

X

n∈Sk,t
Qn(t)|Ft−1, E1
t ]P(E1
t |Ft−1).
(43)

29


---Page Break---
Then for the first term of Eq.(38), from Eqs.(39), (40), (43), for some C3 > 0 we have
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t)

+ eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

k∈[K]
E[E[
X

n∈Sk,t

(17√eπγt + 2βt)ϵ

C2(γt + βt)
Qn(t)|Ft−1, E1
t ]P(E1
t |Ft−1)]

≤
X

t∈[T ]

X

n∈[N]
C3ϵE[E[Qn(t)|E1
t , Ft−1]P(E1
t |Ft−1)]

=
X

t∈[T ]

X

n∈[N]
C3ϵE[Qn(t)1(E1
t )]

≤
X

t∈[T ]

X

n∈[N]
C3ϵE[Qn(t)],
(44)

where the second last inequality comes from C2 ≥17√eπ.

For the second term of Eq.(38), we first have

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E1
t )c)
i

≤
X

t∈[T ]

X

n∈[N]
E

Qn(t)1((E1
t )c)


≤
X

t∈[T ]

X

n∈[N]
tP((E1
t )c)

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)),
(45)

and
X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
t )c)
i

≤
X

t∈[T ]

X

n∈[N]
E

Qn(t)1((E2
t )c)


≤
X

t∈[T ]

X

n∈[N]
tP((E2
t )c)

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)).
(46)

Let Tn be the set of time steps t ∈[T] such that Qn(t) ̸= 0 and let eT = P

n∈[N]
P

t∈Tn 1((E3
n,t)c)
and h = ⌈C4eT /ϵ⌉for some constant C4 > 0. Then if t ≤h, we have Qn(t) ≤t ≤h. Otherwise,
we have

Qn(t) ≤

t
X

s=t−h+1
(1/h)(Qn(s) + (t −s)) ≤(1/h)

t
X

s=1
Qn(s) + (1/h)h2 = (1/h)

t
X

s=1
Qn(s) + h.

30


---Page Break---
From the above, we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E3
n,t)c)]

=
X

n∈[N]

X

t∈Tn
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E3
n,t)c)]

≤
X

n∈[N]

X

t∈Tn
E[Qn(t)1((E3
n,t)c)]

≤
X

n∈[N]

X

t∈Tn
E[((ϵ/C4eT )

T
X

s=1
Qn(s) + 2C4eT /ϵ)1((E3
n,t)c)]

≤E



(ϵ/C4)
X

n∈[N]

T
X

s=1
Qn(s)



+ E

2C4e2
T /ϵ


≤
X

t∈[T ]

X

n∈[N]
(ϵ/C4)E [Qn(t)] + 2C4E[e2
T ]/ϵ.
(47)

Now we provide a bound for E[e2
T ].
Define Nn,k(t) = Pt−1
s=1 1(n ∈Sk,s) and eVk,t =
(κ/2) Pt−1
s=1
P

n∈Sk,s xnx⊤
n . Then, we have

eT =
X

n∈[N]

X

t∈Tn
1((E2
n,t)c)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥V −1
kn,t,t ≥ϵ/C2(γt + βt))

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥eV −1
kn,t,t ≥ϵ/C2(γt + βt))

≤
X

n∈[N]

X

t∈Tn
1(1/Nn,kn,t(t) ≥(κ/2)(ϵ/C2(γt + βt))2)

≤
X

n∈[N]

X

t∈Tn
1(Nn,kn,t(t) ≤(2/κ)(C2(γt + βt)/ϵ)2)

≤
X

n∈[N]

X

k∈[K]

X

t∈Tn
1(Nn,k(t) ≤(2/κ)(C2(γt + βt)/ϵ)2 and kn,t = k)

≤NK(2/κ)(C2(γT + βT )/ϵ)2.

From the above we have E[e2
T ] ≤N 2K2(4/κ2)(C2(γT + βT )/ϵ)4. Then from Eq.(47) we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
(ϵ/C4)E [Qn(t)] + O(N 2K2(βT + γT )4/κ2ϵ5).
(48)

31


---Page Break---
By putting the results of Eqs. (37), (38), (39), (40), (44), (45), (46), (48) altogether, we can obtain

E



X

t∈[T ]
V(Q(t + 1)) −V(Q(t))





≤7 min{N, K}T −2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + 2
X

t∈[T ]

X

n∈[N]
E [(D∗
n(t) −Dn(t))Qn(t)]

≤7 min{N, K}T −2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + 2C3ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)]

+ O(N log(T)) + (2ϵ/C4)
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + O(N 2K2(βT + γT )4/ϵ5)

≤7 min{N, K}T + 2(C3 + (1/C4) −1)ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] + O(N 2K2(βT + γT )4/ϵ5).

Finally, with positive constants C3, C4 > 0 satisfying C3 + (1/C4) < 1, from V(Q(1)) = 0 and
V(Q(T + 1)) ≥0, by using telescoping for the above inequality, we can conclude the proof by

1
T

X

t∈[T ]

X

n∈[N]
E[Qn(t)] = O
min{N, K}

ϵ
+ 1

T
d4N 2K2

κ4ϵ6
polylog(T)

.
(49)

A.7
Proof of Theorem 4

We first provide the proof for the regret bound of e
O

d3/2

κ
√

KTQmax

. We define event E1
t =

{∥ˆθk,t −θk∥Vk,t ≤βt; ∀k ∈[K]} which holds at least probability of 1−1/t2 from Lemma 1. We let
γt = βt
p

d log(Mt). Then we also define E2
t = {|hT S
n,k,t −x⊤
n ˆθk,t| ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈

[K]}, which holds at least probability of 1 −1/t2 from Lemma 5.

Then we have

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))]

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t) + eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))]

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t) + eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))1(E1
t ∩E2
t )]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t) + eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))1((E1
t )c)]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t) + eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))1((E2
t )c)].

(50)

Lemma 8. Under E2
t , for all k ∈[K], we have


X

n∈Sk,t
(eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t))Qn(t)


≤γt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t).
(51)

Proof. Let un,k,t = x⊤
n ˆθk,t. Under E2
t , for any k ∈[K] we have |hT S
n,k,t −un,k,t| ≤γt∥xn∥V −1
k,t .

Then by the mean value theorem, there exists ¯un,k,t = (1 −c)hT S
n,k,t + cun,k,t for some c ∈(0, 1)

32


---Page Break---
satisfying,



X

n∈Sk,t
(eµT S
t
(n|Sk) −µ(n|Sk, θk))Qn(t)



=



X

n∈Sk,t

 
exp(hT S
n,k,t)

1 + P

m∈Sk exp(hT S
m,k,t) −
exp(un,k,t)
1 + P

m∈Sk exp(um,k,t)

!

Qn(t)



=



X

n∈Sk,t
∇vn

 P
m∈Sk,t exp(vm)

1 + P

m∈Sk exp(vm)

! 
vn=¯un,k,t
(hT S
n,k,t −un,k,t)Qn(t)



=

(1 + P

n∈Sk exp(¯un,k,t))(P

n∈Sk exp(¯un,k,t)(hT S
n,k,t −un,k,t)Qn(t))
(1 + P

n∈Sk exp(¯un,k,t))2

−
(P

n∈Sk,t exp(¯un,k,t))(P

n∈Sk,t exp(¯un,k,t)(hT S
n,k,t −un,k,t)Qn(t))

(1 + P

n∈Sk exp(¯un,k,t))2



≤
X

n∈Sk,t

exp(¯un,k,t)
1 + P

m∈Sk,t exp(¯um,k,t)|hT S
n,k,t −un,k,t|Qn(t)

≤max
n∈Sk,t |hT S
n,k,t −un,k,t|Qn(t) ≤γt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t).

Lemma 9. Under E1
t , for all k ∈[K] we have

X

n∈Sk,t
(µ(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))Qn(t) ≤2βt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t).

Proof. Since x/(1 + x) is a non-decreasing function for x > −1 and x⊤
n ˆθkn,t,t ≤x⊤
n ˆθkn,t,t +
βt∥xn∥V −1
kn,t,t under E1
t , with the definition of eµUCB
t
(n|Sk,t), we have

X

n∈Sk,t
(µ(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))Qn(t) ≤
X

n∈Sk,t
(eµUCB
t
(n|Sk,t) −µ(n|Sk,t, θk))Qn(t).

Then let un,k,t = x⊤
n θk. Under E1
t , for any n ∈[N] and k ∈[K] we have x⊤
n ˆθk,t −βt∥xn∥V −1
k,t ≤

x⊤
n θk ≤x⊤
n ˆθk,t + βt∥xn∥V −1
k,t , which implies 0 ≤hUCB
n,k,t −un,k,t ≤2βt∥xn∥V −1
k,t . Then by the

33


---Page Break---
mean value theorem, there exists ¯un,k,t = (1 −c)hUCB
n,k,t + cun,k,t for some c ∈(0, 1) satisfying,
X

n∈Sk,t
(eµUCB
t
(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))Qn(t)

=
X

n∈Sk

 
exp(hUCB
n,k,t )

1 + P
m∈Sk,t exp(hUCB
m,k,t) −
exp(un,k,t)
1 + P

m∈Sk,t exp(um,k,t)

!

Qn(t)

=
X

n∈Sk,t
∇vn

 
exp(vn)
1 + P

m∈Sk,t exp(vm)

! 
vn=¯un,k,t
(hUCB
n,k,t −un,k,t)Qn(t)

=
(1 + P

n∈Sk exp(¯un,k,t))(P

n∈Sk exp(¯un,k,t)(hUCB
n,k,t −un,k,t)Qn(t))
(1 + P

n∈Sk exp(¯un,k,t))2

−
(P

n∈Sk,t exp(¯un,k,t))(P

n∈Sk,t exp(¯un,k,t)(hUCB
n,k,t −un,k,t)Qn(t))

(1 + P

n∈Sk,t exp(¯un,k,t))2

≤
X

n∈Sk,t

exp(¯un,k,t)
1 + P

m∈Sk,t exp(¯um,k,t)(hUCB
n,k,t −un,k,t)Qn(t)

≤max
n∈Sk(hUCB
n,k,t −un,k,t)Qn(t)

≤2βt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t).

Then we first focus on the first term in the above. From Lemmas 8 and 9, we have
X

t∈[T ]

X

n∈[N]
E[eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t )]

≤
X

t∈[T ]

X

n∈[N]
E[(eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, ˆθkn,t,t)

+ µ(n|Skn,t,t, ˆθkn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t )]

≤
X

t∈[T ]
E[
X

k∈[K]

X

n∈Sk,t
(eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t)

+ µ(n|Sk,t, ˆθk,t) −µ(n|Sk,t, θk))Qn(t)1(E1
t ∩E2
t )]

≤
X

t∈[T ]

X

k∈[K]
E[(γt + 2βt) max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)1(E1
t )]

=
X

t∈[T ]

X

k∈[K]
E[E[(γt + 2βt) max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)|E1
t , Ft−1]P(E1
t |Ft−1)].
(52)

We define sets

eΘt =

{θ(i)
k }i∈[M],k∈[K] :

X

n∈Sk,t
(eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M]) −µ(n|Sk,t, ˆθk,t))Qn(t)


≤γt max
n∈Sk,t ∥xn∥V −1
k,t Qn(t); ∀k ∈[K]


and

eΘopt
t
=

{θ(i)
k }i∈[M],k∈[K] :
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])Qn(t)

>
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)

∩eΘt.

34


---Page Break---
We define event Et(eθ) = {{eθ(i)
k,t}i∈[M],k∈[K] ∈eΘopt
t
}. Then by following the proof steps in Eq.(41),
we have

E



E







X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t) −
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M])Qn(t)



1(E1
t ∩E2
t )|Ft−1









≤E

E
 X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])Qn(t)

1(E1
t ∩E2
t )|Ft−1, Et(eθ)


≤E

E

sup

{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t


eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M])

−eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])

Qn(t)1(E1
t ∩E2
t )|Ft−1, Et(eθ)


≤E

E

sup

{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sk,t


eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M]) −µ(n|Sk,t, ˆθk,t)

+ µ(n|Sk,t, ˆθk,t) −eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])

Qn(t)1(E1
t ∩E2
t )|Ft−1, Et(eθ)


≤2γtE



X

k∈[K]
E

max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)1(E1
t )
Ft−1, Et(eθ)




= 2γtE



X

k∈[K]
E

max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)
Ft−1, Et(eθ), E1
t


× P(E1
t |Et(eθ), Ft−1)





= 2γtE



X

k∈[K]
E

max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)
Ft−1, Et(eθ), E1
t


P(E1
t |Ft−1)





≤32√eπγtE



E



X

k∈[K]
max
n∈Sk,t ∥xn∥V −1
k,t Qn(t)|Ft−1, E1
t



P(E1
t |Ft−1)



,

(53)

where the last inequality is obtained from (42). Then from Eqs.(52) and (53), for some constant
C2 > 0, we have

X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t) + eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))1(E1
t ∩E2
t )]

≤
X

t∈[T ]

X

k∈[K]
E[E[ max
n∈Sk,t(33√eπγt + 2βt)∥xn∥V −1
k,t Qn(t)|E1
t , Ft−1]P(E1
t |Ft−1)]

≤C2
X

t∈[T ]

X

k∈[K]
E

max
n∈Sk,t(γt + βt)∥xn∥V −1
k,t Qn(t)1(E1
t )


≤C2
X

t∈[T ]

X

k∈[K]
E

max
n∈Sk,t(γt + βt)∥xn∥V −1
k,t Qn(t)

.

(54)

35


---Page Break---
For the second and third terms of Eq.(50), we have

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E1
t )c)
i

≤
X

t∈[T ]

X

n∈[N]
E

Qn(t)1((E1
t )c)


≤
X

t∈[T ]

X

n∈[N]
tP((E1
t )c)

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)),
(55)

and

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
t )c)
i

≤
X

t∈[T ]

X

n∈[N]
E

Qn(t)1((E2
t )c)


≤
X

t∈[T ]

X

n∈[N]
tP((E2
t )c)

= O



X

t∈[T ]

X

n∈[N]
t(1/t2)



= O(N log(T)).
(56)

Finally from Eqs.(50), (54), (55), and (56), we can conclude that

Rπ(T) = O



X

t∈[T ]

X

k∈[K]
E[ max
n∈Sk,t(γt + βt)∥xn∥V −1
k,t Qn(t)] + N log(T)





= O



γT E[
max
t∈[T ],n∈[N] Qn(t)
X

t∈[T ]

X

k∈[K]
max
n∈Sk,t ∥xn∥V −1
k,t ] + N log(T)





= O



γT E[
max
t∈[T ],n∈[N] Qn(t)
s

KT
X

t∈[T ]

X

k∈[K]
max
n∈Sk,t ∥xn∥2
V −1
k,t ] + N log(T)





= e
O
d3/2

κ

√

KTQmax


,

where the last equality is obtained from Lemma 4.

Here, we provide the proof for the worst-case regret bound of e
O

N

d2K
κ2ϵ3
1/4
T 3/4

.

We define event E1
t
=
{∥ˆθk,t −θk∥Vk,t
≤
βt for all k
∈
[K]} where βt
=

C1
q

λ + d

κ log(1 + tLK/dλ), which holds with high probability as P(E1
t ) ≥1 −1/t2 from

Lemma 1. We also define E2
t = {eµT S
t
(n|Sk,t) −µ(n|Sk,t, ˆθk,t) ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈

[K]}, which holds with high probability as P(E2
t ) ≥1 −O(1/t2), and define E3
n,t = {∥xn∥V −1
k,t ≤

ζ; ∀k ∈[K]}.

36


---Page Break---
Then we have

Rπ(T)

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
t )c) + 1((E3
n,t)c))].

(57)

We provide a bound for the first term of Eq.(57). We first have
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t)

+ eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)].

(58)

By following the steps in Eq.(40), the last two terms in Eq.(58) are bounded as

X

t∈[T ]

X

n∈[N]
E[eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

k∈[K]
E



E



X

n∈Sk,t
(γt + 2βt)ζQn(t)|E1
t , Ft−1



P(E1
t |Ft−1)



.
(59)

Now we provide a bound for the first two terms in Eq.(58).

We define sets

eΘt =

{θ(i)
k }i∈[M],k∈[K] :
max
i∈[M] x⊤
n θ(i)
k −x⊤
n ˆθk,t

 ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈[K]

and

eΘopt
t
=




{θ(i)
k }i∈[M],k∈[K] :
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {θ(i)
k }i∈[M])Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)




∩eΘt.

By following the steps in Eq.(41) with Eq. (42), we have

E[E[(
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk) −
X

k∈[K]

X

n∈Sk,t
eµT S
t
(n|Sk,t, {eθ(i)
k,t}i∈[M]))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)|Ft−1]]

= 2γtζE



X

k∈[K]

X

n∈Sk,t
E
h
Qn(t)
Ft−1, {eθ(i)
k,t}i∈[M],k∈[K] ∈eΘopt
t
, E1
t
i
P(E1
t |Ft−1)





≤16√eπγtζE



X

k∈[K]

X

n∈Sk,t
Qn(t)|Ft−1, E1
t



P(E1
t |Ft−1).

(60)

37


---Page Break---
Then for the first term of Eq.(57), from Eqs.(58), (59), for some C3 > 0 we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Skn,t,t)

+ eµT S
t
(n|Skn,t,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

k∈[K]
E



E



X

n∈Sk,t
(17√eπγt + 2βt)ζQn(t)|Ft−1, E1
t



P(E1
t |Ft−1)





≤
X

t∈[T ]

X

n∈[N]
(17√eπγt + 2βt)ζE[E[Qn(t)|E1
t , Ft−1]P(E1
t |Ft−1)]

=
X

t∈[T ]

X

n∈[N]
(17√eπγt + 2βt)ζE[Qn(t)1(E1
t )]

≤(17√eπγT + 2βT )ζ
X

t∈[T ]

X

n∈[N]
E[Qn(t)].
(61)

For the second term of Eq.(57), by following the steps in Eqs.(45), (46) we have

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E1
t )c)
i
= O(N log(T)),
(62)

and

X

t∈[T ]

X

n∈[N]
E
h
(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
t )c)
i
= O(N log(T)).
(63)

Let Tn be the set of time steps t ∈[T] such that Qn(t) ̸= 0 and let eT = P

n∈[N]
P

t∈Tn 1((E3
n,t)c)
and h = ⌈1/(17√eπγT + 2βT )ζ⌉. Then if t ≤h, we have Qn(t) ≤t ≤h. Otherwise, we have

Qn(t) ≤

t
X

s=t−h+1
(1/h)(Qn(s) + (t −s)) ≤(1/h)

t
X

s=1
Qn(s) + (1/h)h2 = (1/h)

t
X

s=1
Qn(s) + h.

Then, by following the steps in Eq.(47), we have

X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤
X

t∈[T ]

X

n∈[N]
(17√eπγT + 2βT )ζE [Qn(t)] + 2E[eT ]/(17√eπγT + 2βT )ζ.
(64)

38


---Page Break---
Now we provide a bound for E[eT ].
Define Nn,k(t) = Pt−1
s=1 1(n ∈Sk,s) and eVk,t =
(κ/2) Pt−1
s=1
P

n∈Sk,s xnx⊤
n . Then, we have

eT =
X

n∈[N]

X

t∈Tn
1((E2
n,t)c)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥V −1
kn,t,t ≥ζ)

≤
X

n∈[N]

X

t∈Tn
1(∥xn∥eV −1
kn,t,t ≥ζ)

≤
X

n∈[N]

X

t∈Tn
1(1/Nn,kn,t(t) ≥(κ/2)ζ2)

≤
X

n∈[N]

X

t∈Tn
1(Nn,kn,t(t) ≤2/κζ2)

≤
X

n∈[N]

X

k∈[K]

X

t∈Tn
1(Nn,k(t) ≤2/κζ2 and kn,t = k)

≤2NK/κζ2.
(65)

Then from Eqs.(64), (65), we have
X

t∈[T ]

X

n∈[N]
E[(µ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)1((E2
n,t)c)]

≤O



X

t∈[T ]

X

n∈[N]
ζ(βT + γT )E [Qn(t)] +
NK
κζ3(βT + γT )



.
(66)

By putting the results of Eqs. (57), (58), (59), (61), (62), (63), (66), and Theorem 3, by setting
ζ = (ϵK/κT(βT + γT )2)1/4, for large enough T, we can obtain

Rπ(T) = O



ζ(βT + γT )
X

t∈[T ]

X

n∈[N]
E[Qn(t)] +
NK
κζ3(βT + γT ) + N log(T)





= O
ζ(βT + γT )NT

ϵ
+
NK
κζ3(βT + γT ) + N log(T)


= O
(βT + γT )1/2NT 3/4K1/4

κ1/4ϵ3/4
+ N log(T)


= e
O

 

N
d2K

κ2ϵ3

1/4
T 3/4
!

.

A.8
Proof of Lemma 1

We first define
¯fk,t(θ) = Ey[fk,t(θ)|Ft−1] and ¯gk,t(θ) = Ey[gk,t(θ)|Ft−1],

where Ft−1 is the filtration contains outcomes for time s such that s ≤t−1 and y = {yn,t : n ∈Sk,t}.
From Lemma 10 in Oh and Iyengar [44], by taking expectation over y gives

¯fk,t(ˆθk,t) ≤¯fk,t(θk) + ¯gk,t(ˆθk,t)⊤(ˆθk,t −θk) −κ

2 (θk −ˆθk,t)⊤Wk,t(θk −ˆθk,t),

39


---Page Break---
where Wk,t = P
n∈Sk,t xnx⊤
n . Then with ¯fk,t(θk) ≤¯fk,t(ˆθk,t) from Lemma 12 in Oh and Iyengar
[44], we have

0 ≤¯fk,t(ˆθk,t) −¯fk,t(θk)

≤¯gk,t(ˆθk,t)⊤(ˆθk,t −θk) −κ

2 ∥θk −ˆθk,t∥2
Wk,t

= gk,t(ˆθk,t)⊤(ˆθk,t −θk) −κ

2 ∥θk −ˆθk,t∥2
Wk,t + (¯gk,t(ˆθk,t) −gk,t(ˆθk,t))⊤(ˆθk,t −θk)

≤1

2∥gk,t(ˆθk,t)∥2
V −1
k,t+1 + 1

2∥ˆθk,t −θk∥2
Vk,t+1 −1

2∥ˆθk,t+1 −θk∥2
Vk,t+1

−κ

2 ∥θk −ˆθk,t∥2
Wk,t + (¯gk,t(ˆθk,t) −gk,t(ˆθk,t))⊤(ˆθk,t −θk)

≤2 max
n∈Sk,t ∥xn∥2
V −1
k,t+1 + 1

2∥ˆθk,t −θk∥2
Vk,t+1 −1

2∥ˆθk,t+1 −θk∥2
Vk,t+1

−κ

2 ∥θk −ˆθk,t∥2
Wk,t + (¯gk,t(ˆθk,t) −gk,t(ˆθk,t))⊤(ˆθk,t −θk)

≤2 max
n∈Sk,t ∥xn∥2
V −1
k,t+1 + 1

2∥ˆθk,t −θk∥2
Vk,t + κ

4 ∥ˆθk,t −θk∥2
Wk,t −1

2∥ˆθk,t+1 −θk∥2
Vk,t+1

−κ

2 ∥θk −ˆθk,t∥2
Wk,t + (¯gk,t(ˆθk,t) −gk,t(ˆθk,t))⊤(ˆθk,t −θk)

≤2 max
n∈Sk,t ∥xn∥2
V −1
k,t+1 + 1

2∥ˆθk,t −θk∥2
Vk,t −κ

4 ∥ˆθk,t −θk∥2
Wk,t −1

2∥ˆθk,t+1 −θk∥2
Vk,t+1

+ (¯gk,t(ˆθk,t) −gk,t(ˆθk,t))⊤(ˆθk,t −θk),
(67)

where the third inequality comes from Lemma 11 in Oh and Iyengar [44] and the fourth inequality
comes from Lemma 13 in Oh and Iyengar [44]. We note that, although our estimator lies in Θ, we
can still utilize Lemma 11 from Oh and Iyengar [44] by following the same proof steps.

Hence, from the above, we have

∥ˆθk,t+1−θk∥2
Vk,t+1 ≤4 max
n∈Sk,t ∥xn∥2
V −1
k,t+1+∥ˆθk,t−θk∥2
Vk,t−κ

2 ∥ˆθk,t−θk∥2
Wk,t+2(¯gk,t(ˆθk,t)−gk,t(ˆθk,t))⊤(ˆθk,t−θk).

Then using telescoping by summing the above over t, with at least probability 1 −δ, we have

∥ˆθk,t+1 −θk∥2
Vk,t+1 ≤λ + 4

t
X

s=1
max
n∈Sk,s ∥xn∥2
V −1
k,s+1 −κ

2

t
X

s=1
∥ˆθk,s −θk∥2
Wk,s

+ 2

t
X

s=1
(¯gk,s(ˆθk,s) −gk,s(ˆθk,s))⊤(ˆθk,s −θk)

≤λ + 4

t
X

s=1
max
n∈Sk,s ∥xn∥2
V −1
k,s+1 −κ

2

t
X

s=1
∥ˆθk,s −θk∥2
Wk,s

+ κ

2

t
X

s=1
∥θk −ˆθk,t∥2
Wk,s + 2C1

κ
log((log tL)t2K/δ)

≤λ + 4

t
X

s=1
max
n∈Sk,s ∥xn∥2
V −1
k,s+1 + 2C1

κ
log((log tL)t2K/δ)

≤λ + 16(d/κ) log(1 + (tL/dλ)) + 2C1

κ
log((log tL)t2K/δ),
(68)

where the second inequality is obtained from Lemma 14 in Oh and Iyengar [44] and the last one is
obtained from Lemma 4

Then with δ = 1/t2, we can conclude that

∥ˆθk,t −θk∥Vk,t ≤C1

r

λ + d

κ log(1 + tLK/dλ).

40


---Page Break---
A.9
α-approximation Oracle
In this section, we provide a detailed explanation for α-approxiamtion oracle to reduce the computa-
tion. Instead of obtaining the exact solution, the α-approximation oracle, denoted by Oα, outputs
{Sα
k }k∈[K] satisfying P

k∈[K] fk(Sα
k ) ≥max{Sk}k∈[K]∈M(Nt)
P

k∈[K] αfk(Sk). Such an oracle
can be constructed using a straightforward greedy policy as outlined in prior work [26, 10]. Then
for assortments {Sα
k,t}k∈[K] for t ∈[T] from Algorithms 1 and 2 using Oα, we can obtain the same
queue length bounds and regret bounds for α-regret in Theorems 1, 2, 3, and 4 under an α-slackness
assumption, respectively.

We first consider the following traffic slackness assumption with 0 < α < 1 instead of Assumption 3.

Assumption 4. For some traffic slackness 0 < ϵ < 1, for each t ∈[T], there exists {Sk,t}k∈[K] ∈
M(N) which satisfies λn + ϵ ≤αµ(n|Sk,t, θk) for all n ∈Sk,t and k ∈[K].

A.9.1
α-approximation Oracle for Algorithm 1
Here we introduce an algorithm (Algorithm 3) by modifying Algorithm 1 using an α-approximation
oracle. We explain the distinct parts of the algorithm as follows. We define an oracle Oα, which
outputs {Sα
k,t}k∈[K] satisfying

max
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
αQn(t)eµUCB
t
(n|Sk, ˆθk,t) ≤
X

k∈[K]

X

n∈Sα
k,t
Qn(t)eµUCB
t
(n|Sα
k,t, ˆθk,t).

(69)

Algorithm 3 α-approximated UCB-Queueing Matching Bandit
Input: λ, κ, C1 > 0
for t = 1, . . . , T do

for k ∈[K] do

ˆθk,t ←argminθ∈Θ gk,t(ˆθk,t−1)⊤(θ −ˆθk,t−1) + 1

2∥θ −ˆθk,t−1∥2
Vk,t
{Sα
k,t}k∈[K] ←Oα from (69)
Offer {Sα
k,t}k∈[K] and observe preference feedback yn,t ∈{0, 1} for all n ∈Sk,t, k ∈[K]

For the stability analysis, we provide the following theorem.

Theorem 5. The time average expected queue length of Algorithm 3 is bounded as

Q(T) = O
min{N, K}

ϵ
+ 1

T
d2N 2K2

κ4ϵ6
polylog(T)

,

which implies that the algorithm achieves stability as

lim
T →∞Q(T) = O
min{N, K}

ϵ


.

Proof. Here we provide only the proof parts which are different from Theorem 1. We analyze the
Lyapunov drift as follows.
X

t∈[T ]
V(Q(t + 1)) −V(Q(t))

=
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −Qn(t)2

=
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −αD∗
n(t))2 −Qn(t)2

+
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −αD∗
n(t))2. (70)

41


---Page Break---
For the first two terms in Eq.(70), by following the same procedure of Eqs.(16) and (17), under
Assumption 4, we can obtain
X

t∈[T ]

X

n∈[N]
E[(Qn(t) + An(t) −αD∗
n(t))2 −Qn(t)2]

≤
X

t∈[T ]

X

n∈[N]
2E[(λn −αµ(n|S∗
k,t, θk))Qn(t)] + 2NT

≤−
X

t∈[T ]

X

n∈[N]
2ϵE[Qn(t)] + 2 min{N, K}T,
(71)

For the last two terms in Eq.(18), by following the steps for Eq.(20), we have
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −Dn(t))+2 −
X

t∈[T ]

X

n∈[N]
(Qn(t) + An(t) −αD∗
n(t))2

≤4
X

t∈[T ]

X

n∈[N]
An(t) + 2
X

t∈[T ]

X

n∈[N]
(αD∗
n(t) −Dn(t))Qn(t) + min{N, K}T

≤2
X

t∈[T ]

X

n∈[N]
(αD∗
n(t) −Dn(t))Qn(t) + 5 min{N, K}T.
(72)

We also have
X

t∈[T ]

X

n∈[N]
E[(αD∗
n(t) −Dn(t)Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
n,t)c))].

(73)

Then for the first term of Eq.(73), we have

X

t∈[T ]
E



X

n∈[N]
(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(αeµUCB
t
(n|Sk∗
n,t, ˆθk∗
n,t,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
(eµUCB
t
(n|Sα
kn,t,t, ˆθkn,t,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
n,t)





≤
X

t∈[T ]
E



X

n∈[N]
2βt∥xn∥V −1
kn,t,tQn(t)1(E1
t ∩E2
n,t)





≤C2ϵ
X

t∈[T ]

X

n∈[N]
E [Qn(t)] .
(74)

The rest of the proofs can be easily obtained from the proof steps in Theorem 1.

Now, we investigate the regret of Algorithm 3. The α-regret regarding policy π is defined as

Rα,π(T) =
X

t∈[T ]

X

n∈[N]
E
h
(αµ(n|S∗
k∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)
i
.

The algorithm achieves the following regret bound.

42


---Page Break---
Theorem 6. The policy π of Algorithm 3 achieves a regret bound of

Rα,π(T) = e
O

 

min

(
d
κ

√

KTQmax, N
 dK

κ2ϵ3

1/4
T 3/4
)!

.

Proof. In this proof, we provide only the parts that are different from the proof of Theorem 2.

We first provide the proof for regret bound of Rα,π(T) = e
O

d
κ
√

KTQmax

.

We can show that

Rα,π(T) =
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))]

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t )]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))1((E1
t )c)]

=
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t )] +
X

t∈[T ]

X

n∈[N]
tP((E1
t )c)]

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αeµUCB
t
(n|Sk∗
n,t,t, ˆθk∗
n,t,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t )] + O(N/T)

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)((eµUCB
t
(n|Sα
kn,t,t, ˆθkn,t,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t )] + O(N/T)

≤
X

t∈[T ]
E[
X

k∈[K]

X

n∈Sk,t
Qn(t)(eµUCB
t
(n|Sα
k,t) −µt(n|Sα
k,t))1(E1
t )] + O(N/T)

≤2E



βT
X

t∈[T ]

X

k∈[K]
max
n∈Sα
k,t
∥xn∥V −1
k,t Qn(t)



+ O(N/T)

≤2E




max
t∈[T ],n∈[N] Qn(t)βT
X

t∈[T ]

X

k∈[K]
max
n∈Sα
k,t
∥xn∥V −1
k,t



+ O(N/T)

≤2E




max
t∈[T ],n∈[N] Qn(t)βT
s

KT
X

t∈[T ]

X

k∈[K]
max
n∈Sα
k,t
∥xn∥2
V −1
k,t



+ O(N/T)

= e
O
d

κ

√

KTQmax


.

By following the similar steps above and the proofs for Theorem 2, we can easily obtain the worst-case
regret bound of Rα,π(T) = e
O

N
  dK

κ2ϵ3
1/4 T 3/4
, which conclude the proof.

A.9.2
α-approximation Oracle for Algorithm 2
We can obtain similar results for Algorithm 2 as in the case of Algorithm 3. Here we introduce an
algorithm (Algorithm 4) by modifying Algorithm 2 using an α-approximation oracle. We explain
the distinct parts of the algorithm as follows. We define an oracle Oα, which outputs {Sα
k,t}k∈[K]
satisfying

max
{Sk}k∈[K]∈M(Nt)

X

k∈[K]

X

n∈Sk
αQn(t)eµT S
t
(n|Sk, {eθ(i)
k,t}i∈[M]) ≤
X

k∈[K]

X

n∈Sα
k,t
Qn(t)eµT S
t
(n|Sα
k,t, {eθ(i)
k,t}i∈[M]).

(75)

43


---Page Break---
Algorithm 4 α-approximated Thompson Sampling-Queueing Matching Bandit
Input: λ, M, κ, C1 > 0
for t = 1, . . . , T do

for k ∈[K] do

ˆθk,t ←argminθ∈Θ gk,t(ˆθk,t−1)⊤(θ −ˆθk,t−1) + 1

2∥θ −ˆθk,t−1∥2
Vk,t
Sample {eθ(i)
k,t}i∈[M] independently from N(ˆθk,t, β2
t V −1
k,t )

{Sα
k,t}k∈[K] ←Oα from (75)
Offer {Sα
k,t}k∈[K] and observe preference feedback yn,t ∈{0, 1} for all n ∈Sk,t, k ∈[K]

For the stability analysis, we provide the following theorem.

Theorem 7. The time average expected queue length of Algorithm 4 is bounded as

Q(T) = O
min{N, K}

ϵ
+ 1

T
d4N 2K2

κ4ϵ6
polylog(T)

,

which implies that the algorithm achieves stability as

lim
T →∞Q(T) = O
min{N, K}

ϵ


.

Proof. Here we provide only the proof parts which are different from Theorem 3. We analyze the
Lyapunov drift as follows with Assumption 4.
X

t∈[T ]
E [V(Q(t + 1)) −V(Q(t))]

=
X

t∈[T ]

X

n∈[N]
E
h
(Qn(t) + An(t) −Dn(t))+2 −Qn(t)2i

=
X

t∈[T ]

X

n∈[N]
E

(Qn(t) + An(t) −αD∗
n(t))2 −Qn(t)2

+
X

t∈[T ]

X

n∈[N]
E
h
(Qn(t) + An(t) −Dn(t))+2 −(Qn(t) + An(t) −αD∗
n(t))2i

≤7 min{N, K}T −
X

t∈[T ]

X

n∈[N]
2ϵE[Qn(t)] + 2
X

t∈[T ]

X

n∈[N]
E [(αD∗
n(t) −Dn(t))Qn(t)] , (76)

Then, for bounding Eq.(76), we have
X

t∈[T ]

X

n∈[N]
E[(αD∗
n(t) −Dn(t)Qn(t)]

≤
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

+
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)(1((E1
t )c) + 1((E2
t )c) + 1((E3
n,t)c))].

(77)

We provide a bound for the first term of Eq.(77). We first have
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)]

≤
X

t∈[T ]

X

n∈[N]
E[(αµ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Sα
kn,t,t)

+ eµT S
t
(n|Sα
kn,t,t) −µ(n|Sα
kn,t,t, θkn,t))Qn(t)1(E1
t ∩E2
t ∩E3
n,t)].

(78)

44


---Page Break---
Now we provide a bound for the first two terms in Eq.(78).

We define sets

eΘt =

{θ(i)
k }i∈[M],k∈[K] :
max
i∈[M] x⊤
n θ(i)
k −x⊤
n ˆθk,t

 ≤γt∥xn∥V −1
k,t ; ∀n ∈[N], ∀k ∈[K]

and

eΘopt
t
=

{θ(i)
k }i∈[M],k∈[K] :
X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t, {θ(i)
k }i∈[M])Qn(t)

>
X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)

∩eΘt.

Then we define event Et(eθ) = {{eθ(i)
k,t}i∈[M],k∈[K] ∈eΘopt
t
}. Recall hT S
n,k,t = maxi∈[M] x⊤
n eθ(i)
k,t.
Then we have

E



E



X

k∈[K]



X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t) −
X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t, {eθ(i)
k,t}i∈[M])Qn(t)



1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)









≤E



E







X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t, {θ(i)
k }i∈[M])Qn(t)



1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)









≤E



E







X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t, {eθ(i)
k,t}i∈[M])

−
inf
{θ(i)
l
}i∈[M],l∈[K]∈eΘt

X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t, {θ(i)
k }i∈[M])



Qn(t)1(E1
t ∩E2
t ∩E3
n,t)|Ft−1, Et(eθ)









=
2γtϵ
C2(γt + βt)E



X

k∈[K]

X

n∈Sα
k,t
E
h
Qn(t)
Ft−1, Et(eθ), E1
t
i
P(E1
t |Ft−1)



,

(79)

We provide a lemma below for further analysis.

Lemma 10. For all t ∈[T], we have

P



X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sk,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t



≥1/4√eπ

Proof. Given Ft−1, x⊤
n eθ(i)
k,t follows Gaussian distribution with mean x⊤
n ˆθk,t and standard deviation
βt∥xn∥V −1
k,t . Then we have

P

max
i∈[M] x⊤
n eθ(i)
k,t > x⊤
n θk|Ft−1, E1
t


= 1 −P

x⊤
n θ(i)
k,t ≤x⊤
n θk; ∀i ∈[M]|Ft−1, E1
t


= 1 −P

 

Zi ≤x⊤
n θk −x⊤
n ˆθk,t
βt∥xn∥V −1
k,t
; ∀i ∈[M]|Ft−1, E1
t

!

≥1 −P (Z ≤1)M ,

45


---Page Break---
where Zi and Z are standard normal random variables. Then we can show that

P



X

k∈[K]

X

n∈Sα
k,t
eµT S
t
(n|Sα
k,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





≥P



X

k∈[K]

X

n∈Sα
k,t
αeµT S
t
(n|Sα
k,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





≥P



X

k∈[K]

X

n∈S∗
k,t
αeµT S
t
(n|S∗
k,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
αµ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





= P



X

k∈[K]

X

n∈S∗
k,t
eµT S
t
(n|S∗
k,t)Qn(t) >
X

k∈[K]

X

n∈S∗
k,t
µ(n|S∗
k,t, θk)Qn(t)|Ft−1, E1
t





≥P

max
i∈[M] x⊤
n eθ(i)
k,t > x⊤
n θk; ∀n ∈S∗
k,t, ∀k ∈[K]|Ft−1, E1
t



≥1 −LKP(Z ≤1)M

≥1 −LK(1 −1/4√eπ)M

≥
1
4√eπ ,

where the second last inequality is obtained from P(Z ≤1) ≤1 −1/4√eπ using the anti-
concentration of standard normal distribution, and the last inequality comes from M = ⌈1 −
log KL
log(1−1/4√eπ)⌉.

The rest of the proof can be easily obtained by following the proof steps in Theorem 3.

Now, we investigate the regret of Algorithm 4. The α-regret regarding policy π is defined as

Rα,π(T) =
X

t∈[T ]

X

n∈[N]
E
h
(αµ(n|S∗
k∗
n,t,t, θk∗
n,t) −µ(n|Skn,t,t, θkn,t))Qn(t)
i
.

The algorithm achieves the following regret bound.

Theorem 8. The policy π of Algorithm 4 achieves a regret bound of

Rα,π(T) = e
O

 

min

(
d3/2

κ

√

KTQmax, N
d2K

κ2ϵ3

1/4
T 3/4
)!

.

Proof. Here we provide only the proof parts which are different from Theorem 4.

We first provide the proof for the regret bound of e
O

d3/2

κ
√

KTQmax

. We have

Rπ(T) =
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −µ(n|Sα
kn,t,t, θkn,t))]

≤
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Sα
kn,t,t) + eµT S
t
(n|Sα
kn,t,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t ∩E2
t )]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Sα
kn,t,t) + eµT S
t
(n|Sα
kn,t,t) −µ(n|Sα
kn,t,t, θkn,t))1((E1
t )c)]

+
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Sα
kn,t,t) + eµT S
t
(n|Sα
kn,t,t) −µ(n|Sα
kn,t,t, θkn,t))1((E2
t )c)].

46


---Page Break---
By following the similar proof steps in Theorem 7, we can obtain
X

t∈[T ]

X

n∈[N]
E[Qn(t)(αµ(n|Sk∗
n,t,t, θk∗
n,t) −eµT S
t
(n|Sα
kn,t,t)

+ eµT S
t
(n|Sα
kn,t,t) −µ(n|Sα
kn,t,t, θkn,t))1(E1
t ∩E2
t )]

≤C2
X

t∈[T ]

X

k∈[K]
E

"

max
n∈Sα
k,t
(γt + βt)∥xn∥V −1
k,t Qn(t)

#

.

The rest of the proof can be easily obtained from the proof steps in Theorem 4 by incorporating
techniques in the proof of Theorem 7.

A.10
Additional Experiments

0
5000
10000 15000 20000
Time step t

0.0

0.2

0.4

0.6

0.8

1.0

(t)

1e2
Average Queue Length

0
5000
10000 15000 20000
Time step t

0.0

0.5

1.0

1.5

2.0

(t)

1e5
Regret

ETC-GS
UCB-QMB (Algorithm 1)

MaxWeight-UCB
TS-QMB (Algorithm 2)

Q-UCB
Oracle (MaxWeight)

DAM-UCB

Figure 3: Experimental results with N = 4, K = 3, L = 2, d = 2 for (left) average queue length
and (right) regret

A.11
Notation Table

Table 1: Notation Table.

T
Time horizon
N
Number of agents (queues)
K
Number of arms (servers)
d
Dimension of model parameters and features
xn
Feature vector of agent n
θk
Model parameter vector of arm k
Qn(t)
Length of agent n at time t
ϵ
Traffic slackness parameter
µ(n|Sk, θk)
Service rate of arm k with θk for a job in agent n given assortment
Sk
n0
Null agent
λn
Job arrival rate for agent n
kπ
n,t(= kn,t)
A server that agent n is assigned at time t according to π (simply
kn,t)
An(t)
The number of arrival jobs in agent n at time t; Random variable
with mean λn
Dn(t|Sk)
The number of departure job in agent n by arm k given Sk; Random
variable with mean µ(n|Sk, θk)
Q(T)
Average queue lengths over horizon time T
Rπ(T)
Cumulative regret under π over T
κ
Regularity parameter for MNL
Nt
Set of non-empty agents at time t
M()
Set of feasible disjoint assortments given a set of agents
Sk,t
Set of agents assigned to arm k by policy π
S∗
k,t
Set of agents assigned to arm k by the oracle π∗

kπ∗
n,t(= k∗
n,t)
A server that agent n is assigned by π∗at time t
yn,t
Preference feedback, {0, 1}, for assigned agent n at time t

47


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Through the abstract and introduction, we explain our setting with providing
motivation examples and summarize our contributions.

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

Justification: In the limitations section (Section A.1), we provide several avenues for
interesting future work. We also discuss the computational efficiency of our algorithms.

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

48


---Page Break---
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: We provide assumptions in Section 3 and proof sketches of some of the main
theorems in the main text, with complete proofs for theorems in the appendix.

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

Justification: In Section 6, we provide all the information necessary for conducting the
synthetic experiments.

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

49


---Page Break---
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

Justification: There is a link to our code in Section 6.

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

Justification: We use synthetic datasets and provide details on how to generate them in
Section 6.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.

50


---Page Break---
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In our experimental results in Section 6, we include error bars of standard
deviation along with expectation values.

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

Answer: [No]

Justification: The conducted experiments do not require significant computing power.

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

51


---Page Break---
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics.

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

Justification: Given that this study primarily focuses on theoretical analysis, we do not
foresee any negative social consequences.

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

Justification: This paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.

52


---Page Break---
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

Justification: This paper does not use existing assets.

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

53


---Page Break---
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

54


---Page Break---
