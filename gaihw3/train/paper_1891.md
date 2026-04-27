Decentralized Noncooperative Games with Coupled
Decision-Dependent Distributions

Wenjing Yan
Xuanyu Cao ∗
Department of Electronic and Computer Engineering
The Hong Kong University of Science and Technology
wj.yan@connect.ust.hk, eexcao@ust.hk

Abstract

Distribution variations in machine learning, driven by the dynamic nature of deploy-
ment environments, signiﬁcantly impact the performance of learning models. This
paper explores endogenous distribution shifts in learning systems, where deployed
models inﬂuence environments, which in turn alters the data distributions that the
learning models rely on. This phenomenon is formulated by a decision-dependent
distribution mapping within the recently introduced framework of performative
prediction (PP) (Perdomo et al., 2020). Our study investigates the performative
effect in a decentralized noncooperative game, where players aim to minimize
private cost functions while simultaneously managing coupled inequality con-
straints. In this context, we examine two equilibrium concepts for the studied game:
performative stable equilibrium (PSE) and Nash equilibrium (NE), and establish
sufﬁcient conditions for their existence and uniqueness. Notably, we provide the
ﬁrst upper bound on the distance between the PSE and NE in the literature, which
is challenging to evaluate due to the absence of strong convexity on the joint cost
function. Furthermore, we develop a decentralized stochastic primal-dual algorithm
for efﬁciently computing the PSE point. By rigorously bounding the performative
effect, we prove that the proposed algorithm achieves sublinear convergence rates
for both performative regret and constraint violations and maintains the same order
of convergence rate as the case without performativity. Numerical experiments
further conﬁrm the effectiveness of our algorithm and theoretical results.

1
Introduction

Machine learning aims to generalize models trained on given datasets to make accurate predic-
tions or decisions on new, unseen data (El Naqa and Murphy, 2015). The effectiveness of those
models depends on the alignment between the training datasets and deployment environments
(Quinonero-Candela et al., 2008). However, real-world environments are seldom static and often
exhibit ﬂuctuations that can severely degrade model performance (Zhou, 2022). In particular, shifts
in data-generating distributions, driven by the dynamic nature of real-world conditions, present
signiﬁcant challenges for model deployment.

Distribution shifts in machine learning can occur exogenously or endogenously. Exogenous distri-
bution shifts are driven by external factors beyond the control of the learning platforms, such as
environmental changes (Chan et al., 2020) or policy amendments (Wu et al., 2021). In contrast,
endogenous shifts arise from the system’s inherent dynamics and interactions, where the deployed
models affect environments, which in turn alters the data distributions that the learning models rely
on (Dong et al., 2018). For instance, an increase in commodity prices may decrease user interest,
thereby impacting sales. The key distinction lies in the controllability of endogenous shifts, providing

∗Corresponding Author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
an opportunity for designers to either exploit these shifts for improved performance or mitigate
unintended consequences (Dean et al., 2023).

While substantial efforts have been made to address exogenous distribution changes, such as covariate
shift (Chan et al., 2020), label shift (Wu et al., 2021), and concept drift (Lu et al., 2018), relatively
little attention has been paid to the challenges posed by endogenous distribution shifts. Tackling
these endogenous shifts is particularly challenging as data distributions are intrinsically linked to
the decisions made by the learning model itself (Perdomo et al., 2020). As a result, addressing
endogenous shifts may require the explicit modeling of feedback loops, consideration of causal
relationships, and the adaptation of models to dynamic environments.

A notable advancement in this area is the recently proposed framework of “performative prediction
(PP)” (Perdomo et al., 2020), also referred to as “decision-dependent learning” (Drusvyatskiy and
Xiao, 2023). This framework elegantly captures the dynamic interplay between decisions and
data distributions through a decision-dependent mapping, denoted by D(θ) where θ represents the
decision variable. By linking θ to the data distribution, this formulation bridges the gap between
model deployment and parameter optimization. Following the seminal work of (Perdomo et al., 2020),
a growing body of research has emerged, focusing on stability and optimality analysis (Piliouras
and Yu, 2023; Miller et al., 2021), as well as algorithmic design for various settings, including
reinforcement learning (Mandal et al., 2023), online learning (Wood et al., 2021), bandit problems
(Jagadeesan et al., 2022), and bilevel optimization (Lu, 2023).

This paper investigates endogenous distribution shifts in a decentralized noncooperative game, where
players aim to minimize private cost functions while simultaneously managing coupled inequality
constraints. To contextualize this setting, consider scenarios where strategic responses exhibit in
learning environments and competitive interactions occur among players. For example, in autonomous
vehicular networks, multiple vehicles compete to select their routes under constraints such as road
capacities, trafﬁc congestion, and travel costs. The route choices of each vehicle inﬂuence trafﬁc
patterns and consequently affect the travel times experienced by other vehicles (Mori et al., 2015).
Similarly, in ﬁnance, traders compete to maximize proﬁts under constraints like market capacities
and inventory levels. The trading strategies of these participants impact market volatility and the
distribution of asset prices, creating a dynamic pricing landscape (Fattouh and Mahadeva, 2014).
These dynamics extend to other domains, such as electricity market competition (Moshari et al.,
2010), ride-sharing platforms (Narang et al., 2023), natural resource extraction (Cust and Poelhekke,
2015), and online advertising auctions (Varian, 2009).

Despite its pervasiveness, this performative phenomenon has largely been overlooked in the studies of
decentralized noncooperative games. This paper addresses the problem by formulating performativity
using coupled decision-dependent distributions, following the PP framework of (Perdomo et al.,
2020). However, the intricate interplay between decentralized players and endogenous distribution
shifts presents challenging theoretical and algorithmic questions: How do strategic responses in
learning environments inﬂuence the game’s equilibrium? How can players adapt their strategies
effectively when confronted with coupled decision-dependent distributions? How can we design
algorithms to exploit these dynamics for optimal decision-making? These questions form the core
of our investigation, guiding us toward more resilient, adaptive, and efﬁcient learning outcomes in
decentralized games, especially in environments characterized by continuously evolving data and
decision-making processes. Our main contributions are summarized below:

• We initially formulate the problem of decentralized noncooperative games with data perfor-
mativity, where selﬁsh players seek to minimize individual costs while managing coupled
inequality constraints. Under this setting, we examine two equilibrium concepts: performa-
tive stable equilibrium (PSE) and Nash equilibrium (NE), and establish sufﬁcient conditions
for their existence and uniqueness. Compared to conventional games, this examination is
more complicated due to the interplay between decision-making and distribution changes.
Notably, we make a signiﬁcant contribution by providing the ﬁrst upper bound on the
distance between the PSE and NE in the literature. Computing this distance in PP games is
challenging due to the absence of strong convexity on the joint cost function, an essential
property for determining the optimality gap of performative stable points in previous work.
Instead, we characterize the distance by leveraging relations from strong duality and derive
a result comparable to the ﬁndings of the prior work (Perdomo et al., 2020; Lu, 2023).

2


---Page Break---
• To compute the PSE point of the PP-game, we propose a decentralized stochastic primal-dual
algorithm based on repeated risk minimization (RRM). The development and convergence
analysis of this algorithm face two primary challenges. First, there is a complex interaction
between decentralized competition and endogenous distribution shifts. Second, players only
have partial observation, as they communicate solely with neighbors, despite their private
cost functions being inﬂuenced by the strategies of all players. We evaluate the performance
of our algorithm by two commonly used metrics: performative regret, which measures the
suboptimality of the strategy sequence generated by RRM relative to the PSE point, and
constraint violation. By rigorously bounding the performative effect, we prove that the
proposed algorithm achieves sublinear convergence rates for both metrics. Furthermore,
our results show that while the performative effect slows down convergence, it does not
degrade the order of performative regret compared to the case without performativity (Lu
et al., 2020).

Finally, we conduct numerical experiments on a networked Cournot game and a ride-share market.
The simulation results conﬁrm the sublinear convergence of our algorithm. Furthermore, the results
demonstrate that while greater performative strength leads to a wider gap between the PSE and NE,
the discrepancy between these two equilibria remains marginal. This veriﬁes both the effectiveness
of the PSE solutions and the accuracy of our distance analysis between the PSE and NE.

Related Work: Among the numerous existing studies, two closely related works (Narang et al.,
2023) and (Wang et al., 2023) have considered performative behaviors in games. A key distinction in
our work is that our model requires all players’ collective strategies to adhere to the constraints of the
learning system, whereas both (Narang et al., 2023) and (Wang et al., 2023) address unconstrained
settings. This difference results in fundamentally distinct algorithmic designs and convergence
analyses. Our approach employs a primal-dual technique and requires consensus, whereas their
methods only rely on local stochastic gradient descent. Additionally, we consider a mathematically
richer model compared to (Wang et al., 2023), whose framework is structured in a speciﬁc form
involving local costs dependent solely on individual strategies and a regularizer quantifying similarity
among neighboring strategies. Furthermore, our algorithm design accounts for practical constraints
where players can only communicate with their immediate neighbors, while (Narang et al., 2023)
assumes full accessibility to all players’ strategies across the entire network. Importantly, our work
makes a signiﬁcant contribution by providing the ﬁrst upper bound on the distance between the
performative stable equilibrium (PSE) and Nash equilibrium (NE)—a gap not previously addressed.
Other related works such as (Li et al., 2022) and (Piliouras and Yu, 2023), have studied performative
prediction in decentralized multi-agent optimization. The former focuses on consensus-seeking
agents, while the latter is restricted to location-scale families. Finally, (Yan and Cao, 2024b)
considers the constrained performative prediction problem in a single-agent setting, whereas our
paper addresses decentralized noncooperative games. A more comprehensive literature review is
provided in Appendix A.

2
Problem Formulation

Consider a decentralized noncooperative game with n players. Each player i selects a strategy (or,
interchangeably, decision, action), denoted as θi, from its feasible set Ωi ⊆Rd. Let the collective
decisions of all players be denoted as θ := col (θ1, · · · , θn), and the collective decisions of all players
except player i be represented as θ−i := col (θ1, · · · , θi−1, θi+1, · · · , θn), for any i ∈[n], where
[n] denotes the set of integers {1, 2, . . . , n}. Each player i has a private cost function Ji(ξi; θi, θ−i),
which depends on the random variable ξi ∈Ξi, the player’s private decision θi, and the decisions of
all other players θ−i. This paper considers a scenario where the underlying populations strategically
respond to the players’ decisions, causing shifts in data distributions. This interplay is modeled by
a decision-dependent distribution mapping ξi ∼Di (θi, θ−i) for all i ∈[n]. The objective of each
player i is to selﬁshly minimize its performative risk Eξi∼Di(θi,θ−i)Ji(ξi; θi, θ−i) (abbreviated as
PRi(θi, θ−i)), subject to a coupled constriant Pn
i=1 gi(θi) ⪯0, i.e.,

min
θi∈Ωi
Eξi∼Di(θi,θ−i)Ji (ξi; θi, θ−i)

subject to
gi(θi) + P

j̸=i gj(θj) ⪯0.
(1)

Both Ji(·) and gi(·) are only locally accessible to player i for all i ∈[n]. In the game (1), each player
solves its private optimization problem to determine the best strategy, given the current strategies

3


---Page Break---
of all the other players. An equilibrium of the game (1) corresponds to a set of strategies where no
player can improve its performance by deviating unilaterally from its strategy.

Denote by ξ := col (ξ1, · · · , ξn) the concatenation of the variables ξi and by J (ξ; θ) :=
col (J1 (ξ1; θ) , · · · , Jn (ξn; θ)) the concatenation of the cost functions Ji(·) for all i
∈
[n].
A stochastic pseudogradient mapping of J (ξ; θ) is deﬁned as ∇J (ξ; θ)
:=
col (∇θ1J1 (ξ1; θ) , · · · , ∇θnJn (ξn; θ)). We have the following assumption on ∇J (ξ; θ).
Assumption 2.1. There exists a constant µ > 0 such that the stochastic gradient mapping ∇J (ξ; θ)
is µ-strongly monotone, i.e.,

∇J (ξ; θ) −∇J
 
ξ; θ′
, θ −θ′
≥µ∥θ −θ′∥2
2, ∀ξ ∈Ξ, θ, θ′ ∈Ω,
where Ξ := Ξ1 × · · · × Ξn and Ω:= Ω1 × · · · × Ωn.

Assumption 2.1 is commonly made in the literature of game theory. It sufﬁces to guarantee the
existence of Nash equilibrium for a stochastic game with ﬁxed data distributions (Facchinei and Pang,
2003, Theorem 2.3.3(b)). However, in our paper, since the data distributions are decision-dependent,
Assumption 2.1 does not imply the monotonicity of the gradient mapping of the joint performative
risk, denoted by PR(·) := col (PR1(·), · · · , PRn(·)). Therefore, the existence and uniqueness
(E&U) conditions for the Nash equilibrium of the game (1) need further investigation.

We deﬁne a graph G(P) to represent the impact of players’ decisions on the data distributions of
different players. In G(P), the weight pij > 0 if player j’s decision affects player i’s data distribution,
and pij = 0 otherwise. Particularly, pii represents the weight of self-inﬂuence. These weights are
normalized as Pn
j=1 pij = 1, for all i ∈[n]. Clearly, the larger the weight pij, the stronger the effect
of player j’s decision on the data distribution of player i.

Let W1 (D, D′) represent the Wasserstein-1 distance between two probability measures D and D′.
Following (Wang et al., 2023), we impose the following assumption on the distributions {Di}i∈[n].

Assumption 2.2. For any i ∈[n], there exists a constant εi ≥0 such that, ∀θ, θ′ ∈Ω, the

distribution mapping Di is constrained by W1
 
Di (θ) , Di
 
θ′
≤εi
qPn
j=1 pij
θj −θ′
j
2
2.

For any i ∈[n], the parameter εi bounds the sensitivity of player i’s distribution with respect to (w.r.t.)
the decision variations of all players. This ε-sensitivity property of distributions is conceptually akin
to the Lipschitz continuity of functions that quantiﬁes the variation of function values w.r.t argument
changes. We also require the following assumptions.
Assumption 2.3. For any i ∈[n], the non-empty feasible set Ωi is closed, convex, and bounded, i.e.,
there exists a constant C ≥0 such that, ∀θi ∈Ωi, ∥θi∥2 ≤C.
Assumption 2.4. For any i ∈[n] and θi ∈Ωi, the cost function Ji(ξi; θi, θ−i) is convex
w.r.t.
θi.
Moreover, there exists a constant Li ≥0 such that Ji (ξi; θ) is Li-smooth, i.e,
∇Ji (ξi; θ) −∇Ji
 
ξ′
i; θ′
2 ≤Li
 ξi −ξ′
i

2 +
θ −θ′
2

, ∀ξi, ξ′
i ∈Ξi, θ, θ′ ∈Ω.

Assumption 2.5. For any i ∈[n] and θi ∈Ωi, the constraint function gi(θi) is convex w.r.t. θi.
Moreover, there exist a constant Gg ≥0 such that gi(·) is Gg-Lipschitz, i.e.,
gi(θi) −gi(θ′
i)

2 ≤
Gg∥θi −θ′
i∥2, ∀θi, θ′
i ∈Ωi.

Assumptions 2.3 and 2.5 are widely used in constrained optimization (Bertsekas, 2014; Yan and Cao,
2024a), and Assumption 2.4 is standard in the PP literature. From Yan and Cao (2024a, Proposition 1),
under Assumptions 2.3 and 2.4, the cost function Ji(ξi; θ), ∀i ∈[n] is Lipschitz continuous, i.e., there
exist a constant Gi ≥0 such that |Ji(ξi; θ)−Ji(ξ′
i; θ′)| ≤Gi
 ξi −ξ′
i

2 +
θ −θ′
2

, ∀ξi, ξ′
i ∈
Ξi, θ, θ′ ∈Ω. Moreover, Assumptions 2.3 and 2.5 imply the boundedness of ∥gi(θi)∥2, i.e., there
exists a constant B ≥0 such that ∥gi(θi)∥2 ≤B, ∀θi ∈Ωi, i ∈[n].

3
Equilibrium of the PP-Game

This section examines two fundamental equilibrium concepts of the performative game (1): Nash
equilibrium (NE) and performative stable equilibrium (PSE), as deﬁned below.
Deﬁnition 3.1 (Nash Equilibrium). A vector θne := col (θne
1 , . . . , θne
n ) achieves an NE of the game
(1) if it holds for any i ∈[n] that

θne
i
∈arg min
θi∈Ωi
Eξi∼Di(θi,θne
−i)Ji
 
ξi; θi, θne
−i


subject to
gi(θi) + P

j̸=i gj(θne
j ) ⪯0.

4


---Page Break---
Deﬁnition 3.2 (Performative Stable Equilibrium). A vector θpse := col (θpse
1 , . . . , θpse
n ) achieves a
PSE of the game (1) if it holds for any i ∈[n] that

θpse
i
∈arg min
θi∈Ωi
Eξi∼Di(θpse)Ji
 
ξi; θi, θpse
−i


subject to
gi(θi) + P

j̸=i gj(θpse
j
) ⪯0.

NE is a fundamental concept in game theory. At NE, each player’s strategy optimally aligns with
its own interest, given the strategies of other players. Hence, no player has an incentive to deviate
from its strategy unilaterally. In the case of performative games, the computation of NE needs to take
into account the data distributions Di(·) for all i ∈[n], as they are parameterized by the optimization
variable θ. However, this information is often unavailable in practice. Instead, at PSE, the data
distribution of each player i ∈[n] is ﬁxed at Di (θpse) and the PSE point achieves an NE of the game
(1) under the ﬁxed data distribution of its own deployment. This formulation draws benign properties
akin to problems with ﬁxed data distributions, facilitating the adaptation of existing algorithms.
Therefore, PSE is more frequently chosen as a performance metric in the literature of PP.

3.1
Existence and Uniqueness of PSE

We ﬁrst establish the condition for the E&U of the PSE of the game (1). Our approach relies on
repeated risk minimization (RRM) for closed-loop retraining. First, we deﬁne a mapping T (θ) :=
{Ti(θ)}i∈[n] that, for any i ∈[n],

θ′
i = Ti(θ) := arg min
ui∈Ωi
Eξi∼Di(θi,θ−i)Ji
 
ξi; ui, θ′
−i


subject to
gi(ui) + P

j̸=i gj
 
θ′
j

⪯0.

The mapping T (θ) outputs the NE of the game (1) under the ﬁxed data distributions Di(θi, θ−i) for
all i ∈[n]. With Assumption 2.1, the E&U of this NE is guaranteed, thereby ensuring the validity of
the mapping T (θ). Based on T (θ), the RRM updates θt
i at each iteration t by

θt+1
i
= Ti(θt), ∀i ∈[n].
(2)

Clearly, θt+1 is an NE of the game (1) under the deployment of θt. Additionally, we have that
any ﬁxed point of (2) achieves an PSE for the game (1), i.e., θpse = T (θpse). By investigating the
convergence the iterative equation (2), we have the following sufﬁcient condition for the E&U of the
PSE of the game (1).
Theorem 3.3. Suppose that Assumptions 2.1-2.5 hold. Then, for any θ, δ ∈Ω, the mapping T (θ)
satisﬁes

∥T (θ) −T (δ)∥2 ≤1

µ
qPn
i=1 L2
i ε2
i maxj∈[n] pij ∥θ −δ∥2 .

Thus, if it is satisﬁed that

1
µ
qPn
i=1 L2
i ε2
i maxj∈[n] pij < 1,
(3)

the sequence generated by the RRM (2) converges to a unique PSE point θpse at a linear rate that

∥θt+1 −θpse∥2 ≤

1
µ
qPn
i=1 L2
i ε2
i maxj∈[n] pij
t θ1 −θpse
2 .

The proof of Theorem 3.3 is provided in Appendix B. According to Theorem 3.3, under Assumptions
2.1-2.5, when condition (3) holds, we have that: (i) the game (1) admits a unique PSE, and (ii) the
RRM method (2) converges linearly to the PSE.

Since the inﬂuence weights {pij}j∈[n] are normalized, with Pn
j=1 pij = 1 for all i ∈[n], we
generally have that pij = O( 1

n). Therefore, the contraction condition (3) exhibits good scalability
w.r.t. the number of players. Moreover, according to the proof in Appendix B, if for any player
i ∈[n], its distribution Di(·) depends only on its own decision θi, i.e., pij = 0 for all j ̸= i, then we
have

∥T (θ) −T (δ)∥2 ≤1

µ maxi∈[n] Liεi ∥δ −θ∥2 .

5


---Page Break---
The contraction of the above iterative equation only requires that 1

µ maxi∈[n] Liεi < 1. Furthermore,
if all players exhibit equivalent model parameters that L1 = · · · = Ln = L and ε1 = · · · = εn = ε
and pij = 1

n for all i, j ∈[n], condition (3) reduces to Lε

µ < 1, recovering the contraction requirement
of (Perdomo et al., 2020) for a single-agent PP case.

3.2
Existence and Uniqueness of NE

First, we deﬁne a gradient mapping G(i)
θ (δi, δ−i) := Eξi∼Di(θ)∇δiJi (ξi; δi, δ−i) for any i ∈[n],

and Gθ(δ) := col

G(1)
θ (δ), · · · , G(n)
θ (δ)

. Moreover, for any i ∈[n], deﬁne

H(i)
θi,θ−i(δ) := ∇uiEξi∼Di(ui,θ−i) [Ji (ξi; δ)]

ui=θi

and Hθ(δ) := col

H(1)
θ1,θ−1(δ), · · · , H(n)
θn,θ−n(δ)

. Then, for any i ∈[n], the gradient of the

performative risk PRi(θi, θ−i) w.r.t. θi is given by

∇θiPRi(θi, θ−i) = G(i)
θi,θ−i(θi, θ−i) + H(i)
θi,θ−i(θi, θ−i).

Deﬁne ∇PR(θ) := col (∇θ1PRi(θ), · · · , ∇θnPRn(θ)), we further have

∇PR(θ) = Gθ(θ) + Hθ(θ).

From Facchinei and Pang (2003, Theorem 2.3.3(b)), to prove the E&U of the NE of the (1), we
require the strongly monotonivity of the gradient mapping ∇PR(θ). Therefore, we have the following
sufﬁcient condition for the E&U of the NE of the game (1).
Theorem 3.4. Suppose that Assumptions 2.1-2.5 hold. If it is satisﬁed that

µ −Pn
i=1 Liεi maxj∈[n] √pij −
pPn
i=1 L2
i ε2
i pii > 0,
(4)

then, the PP-game (1) is strongly monotone and admits a unique NE.

The proof of Theorem 3.4 is presented in Appendix C. Since pij characterizes the inﬂuence of
player j’s decision on the data distribution of player i, we typically have pij ≤pii for j ̸= i and
thus maxj∈[n] pij = pii for all i ∈[n]. Then, the condition (4) reduces to µ −Pn
i=1 Liεipii −
pPn
i=1 L2
i ε2
i pii > 0. Similarly, when L1 = · · · = Ln = L, ε1 = · · · = εn = ε, and pij = 1

n for
all i, j ∈[n], we require that µ −2Lε > 0, i.e., ε ≤
µ
2L, which recovers the condition to guarantee
the convexity of the performative risk PR(·), and thereby the E&U of the performative optimal point
of (Miller et al., 2021) for single-agent PP.

3.3
Distance Between PSE and NE

Theorem
3.5.
Deﬁne
eµ
:=
µ
−
Pn
i=1 Liεi maxj∈[n] √pij
and
α
:=
Pn
i=1 Gi
 
1 + εi maxj∈[n] √pij

.
Suppose that Assumptions 2.1-2.5 hold and eµ > 0.
Then,
for every PSE point and NE point, we have the following relations:

∥θpse −θne∥2 ≤1

eµ
pPn
i=1 G2
i ε2
i pii
and
|PR(θpse) −PR(θne)| ≤α

eµ
pPn
i=1 G2
i ε2
i pii.

The proof of Theorem 3.5 is presented in Appendix D. According to Theorem 3.5, the distance
between the PSE and NE of the game (1) depends on the cost functions’ parameters µ, {Gi}, {Li},
as well as the sensitivity of the data distributions {εi}. Larger sensitivity parameters widen the gap
between the PSE and NE, while a bigger monotonicity parameter µ reduces it. Notably, when the
sensitivity parameter εi = 0 for all i ∈[n], the game (1) reduces to a conventional stochastic game
with ﬁxed data distributions, and as a result, the PSE and NE converge to the same point.

To the best of our knowledge, this is the ﬁrst result on the distance between PSE and NE of PP-games.
Characterizing this distance is challenging in games due to the lack of strong convexity on the joint
cost function J(·), which is an essential property for determining the optimality gap of performative
stable points in previous work (Perdomo et al., 2020; Lu, 2023). In this paper, we characterize
this gap by leveraging relations from strong duality (Boyd and Vandenberghe, 2004; Facchinei and
Pang, 2010). Our result is comparable to the ﬁndings in (Perdomo et al., 2020) for single-agent PP
problems wherein this optimality gap is bounded by 2Lε

µ . In our case, when G1 = · · · = Gn = G,
ε1 = · · · = εn = ε and pij = 1

n for all i, j ∈[n], we have ∥θpse −θne∥2 ≤
Gε
µ−Lε.

6


---Page Break---
Algorithm 1 Decentralized Stochastic Primal-Dual Algorithm: The Procedures at Player i, ∀i ∈[n]:

1: Initialize θ1
i ∈Ξi arbitrarily. Set λ1
i = 0 and bθ
1
ih = 0 for all h ̸= i.
2: for t = 1 to T do
3:
Exchange θt
i, bθ
t
i, and λt
i with all neighbors;

4:
Update the estimate bθ
t
ih for all h ̸= i by: bθ
t+1
ih
= P

k̸=h aikbθ
t
kh + aihθt
h;

5:
Deploy the model θt
i and sample ξt
i ∼Di(θt
i, θt
−i);

6:
Update the primal variable by: θt+1
i
= PΩi
h
θt
i −γt

∇θiJi

ξt
i; θt
i, bθ
t
i

+ γt∇gi(θt
i)⊤λt
i
i
;

7:
Update the dual variable by: λt+1
i
=
h 
1 −γ2
t
 P
j∈Ni aijλt
j + γtgi
 
θt
i
i

+.

8: end for

4
Computation of the PSE

Although RRM theoretically has the capability to ﬁnd a PSE point, how to perform risk minimization
at its each update remains unknown. Moreover, RRM requires the computation of an NE for each
deployment, which is computationally intensive. In this section, we present a decentralized stochastic
primal-dual algorithm for efﬁciently computing the PSE of the game (1). Theoretical analysis is also
provided on the convergence of the proposed algorithm.

4.1
Algorithm Development

For each player i ∈[n], deﬁne a regularized Lagrangian as

L(i)
δ (θi, θ−i, λ) = Eξi∼Di(δ)Ji (ξi; θi, θ−i) +
D
λ, gi(θi) + P

j̸=i gj (θj)
E
,

where λ ∈Rm
+ is the dual variable. Denote by ∇gi(·) the Jacobian matrix of gi(·). From the
primal-dual theory (Boyd and Vandenberghe, 2004; Facchinei and Pang, 2010), for any γ > 0, there
exists a bounded Lagrangian multiplier λpse such that the following condition holds:

θpse
i
=PΩi
h
θpse
i
−γ

G(i)
θpse (θpse, λpse) + γ∇gi(θpse
i
)⊤λpsei
,
∀i ∈[n],

λpse =
h
λpse + γ

gi(θpse
i
) + P

j̸=i gj
 
θpse
j
i

+ ,

where γ is a control parameter. Thus, given θpse
−i and under ξi ∼Di(θpse), (θpse
i
, λpse) is a saddle

point of the Lagrangian L(i)
θpse(θi, θpse
−i , λ) for any i ∈[n]. The joint saddle point (θpse, λpse) achieve
the PSE of the game (1) under strong duality (Boyd and Vandenberghe, 2004).

In the decentralized noncooperative game (1), each player can only communicate with its neighbors.
We use G(A) to denote the communication graph of the network, where A = (aij)n×n represents a
weight matrix. In G(A), aij = aji > 0 if there is a communication link between player i and play j,
and aij = aji = 0 otherwise. Let Ni be the set containing player i and all its neighbors such that
j ∈Ni if aij > 0. We assume that the communication graph G(A) is connected and the weight
matrix A is doubly stochastic.

To ﬁnd the saddle point (θpse, λpse), we develop a decentralized stochastic primal-dual algorithm,
as presented in Algorithm 1. The basic idea of Algorithm 1 is to perform gradient update on the
primal variables θi for all i ∈[n] and the dual variable λ. In the decentralized noncooperative game,
each player i ∈[n] only observes information from its neighbors. However, its private cost funtion
Ji(ξi; θi, θ−i) involves all players’ strategies. To solve this problem, we let each player i store an
estimate for the strategies of all the other players, denoted by bθih, for all h ̸= i. Deﬁne a vector bθi
that concatenates all the estimates bθih. In each iteration t, neighbors exchange strategy θt
i, estimate
bθ
t
i, and dual varible λt
i with each other. Then, each player i updates the estimates bθih, for all h ̸= i
by weighted average in Step 4. The primal variable θt
i is updated by gradient descent by Step 6, and
the dual variable λt
i is updated by gradient ascent by Step 7. The coefﬁcient γt is the stepsize at the
tth iteration for all t ∈[T].

7


---Page Break---
4.2
Performance Analysis

Before analyzing the performance of Algorithm 1, we deﬁne the performance metrics adopted in this
paper. The ﬁrst metric is performative regret. For any player i ∈[n], its performative regret over T
iterations is deﬁned as

Ri(T):=PT
t=1
 
Eξi∼Di(θpse)Ji
 
ξi; θt
i, θpse
−i

−PRi (θpse)

.

The regret Ri(T) measures the suboptimality of the sequence of decisions {θ1
i , · · · , θT
i } taken by
play i relative to θpse
i
. Besides, since the decisions of all players are subject to constraints, another
performance metric of constraint violation, denoted by Rg(T), is required, deﬁned as

Rg(T) =

hPT
t=1
Pn
i=1 gi
 
θt
i
i

+


2
.

Any online or learning algorithm is regarded as “good” if both the time-average regret and the
time-average constraint violation are sublinear, i.e., limT →∞Ri(T)/T ≤o(1) for any i ∈[n] and
limT →∞Rg(T)/T ≤o(1).

For analysis, we make the following assumption on the variance of the stochastic gradient
∇θiJi (ξi; δ), ∀i ∈[n].
Assumption
4.1.
The
stochastic
gradient
∇δiJi (ξi; δi, δ−i)
is
unbiased
that
Eξi∼Di(θ)∇δiJi (ξi; δi, δ−i)
=
G(i)
θ (δi, δ−i) and there exist constants σ0, σ1
≥
0 such

that Pn
i=1 Eξi∼Di(θ)
∇δiJi (ξi; δi, δ−i) −G(i)
θ (δi, δ−i)

2

2 ≤σ2
0 + σ2
1 ∥θ −θpse∥2
2 , ∀θ, δ ∈Ω.

Theorem
4.2.
Deﬁne
eµ
:=
µ
−
Pn
i=1 Liεi maxj∈[n] √pij
and
ν
:=
3
 
σ2
1 + 3 Pn
i=1 L2
i
 
1 + ε2
i maxj∈[n] pij

.
Suppose that Assumptions 2.1-2.5 and 4.1 hold
and eµ > 0. By Algorithm 1, if the stepsize satisﬁes supt∈[T ] γt ≤eµ

ν , then, the performative regret of
the game (1) is bounded by

Ri(T) ≤O
r

T

eµ

1
γT + PT
t=1 γt

, ∀i ∈[n].

Further, the constraint violation is bounded by

Rg(T) ≤O

1
γT

r
1
γT + PT
t=1 γt
 
1 + PT
t=1 γ2
t

.

For a sequence of diminishing stepsize γt = τ η
1 (τ2t + τ1)−η, where τ1, τ2 > 0 and 0 < η < 1, we
have that: 1) PT
t=1 γt ≤O
 
T 1−η
; 2) PT
t=1 γ2(t) ≤O
 
T 1−2η
. Plugging the above results into
Theorem 4.2 yields

Ri(T) ≤O

T
1+η

2
+ T 1−η

2

, i ∈[n]
and
Rg(T) ≤O

T
3
2 η + T
1+η

2
+ T 1−η

2

.

Based on the above two inequalities, the best choice of η is 1

2 such that Ri(T) ≤O(T
3
4 ), ∀i ∈[n]
and Rg(T) ≤O(T
3
4 ). This convergence speed matches that of the decentralized noncooperative
game without performativity (Lu et al., 2020).

The proof of Theorem 4.2 is provided in Appendix E. According to Theorem 4.2, the performative
effect reduces the convergence rate by amplifying the coefﬁcient 1

eµ in the regret bounds. Speciﬁcally,
as the sensitivity parameters εi increase, the coefﬁcient eµ decreases, leading to a slower convergence
rate of Ri(T) for all i ∈[n]. This occurs because a larger εi indicates a stronger performative
inﬂuence, which more signiﬁcantly impacts the algorithm’s convergence. Nevertheless, the perfor-
mative effect does not degrade the convergence order of Algorithm 1 compared to the case without
performativity (Lu et al., 2020).

5
Numerical Experiments

In this section, we evaluate the effectiveness of our algorithm and theoretical results by conducting
numerical experiments on a networked Cournot game (Abolhassani et al., 2014), which is a founda-
tional model in economic theory (Allaz and Vila, 1993) for analyzing oligopolistic competitions. We

8


---Page Break---
Figure 1: Convergence of time-average regrets and time-average constraint violations.

Figure 2: (a). Normalized distance between θt and θne. (b). Total revenue at PSE and NE.

consider a networked Cournot game with ﬁve ﬁrms selling a single commodity across three markets.
Each ﬁrm aims to maximize its proﬁt by determining the quantities it serves in all markets. The total
accommodated quantity in each market is limited by its market capacity. The simulation details and
additional numerical results are presented in Appendix F.1. We also provide an additional experiment
on a ride-share market in Appendix F.2.

Fig. 1 illustrates the convergence of the time-average regrets of ﬁve ﬁrms, denoted by Ri(t)/t,
∀i ∈[5], and the convergence of the time-average constraint violations of three markets, de-
noted by
1
t
Pt
t′=1
Pn
i=1 gij(θt′
i ), ∀j ∈[3].
The results demonstrate that both Ri(t)/t and

1
t
Pt
t′=1
Pn
i=1 gij(θt′
i ) approach zero as the iterations increase. This veriﬁes the sublinear con-
vergence of the regrets and constraint violations in Theorem 4.2.

Fig. 2 (a) compares the normalized distance between θt, generated by Algorithm 1, and the NE
point θne, denoted as ∥θt −θne∥2/∥θt∥2. The NE point is computed based on perfect knowledge
of {Di}i∈[n]. We consider three different performative strengths: ε = 0.2, 0.4, and 0.6. It is
observed that ∥θt −θne∥2/∥θt∥2 stabilizes at values approximately equal to or smaller than 10−1
with iterations, varifying the effectiveness of Algorithm 1. Additionally, a larger performative strength
leads to a wider normalized distance between the convergent point of θt and θne. In Fig. 2 (b),
we compare the total revenues, denoted by −P5
i=1 PRi(θt) under the same three ε settings. We
consider two scenarios: 1). “pse”, where θt is generated by Algorithm 1; 2). “ne”, where θt is
generated by performing the same procedures as Algorithm 1 but with perfect information on the
distributions {Di(θ)}i∈[n]. The result demonstrates the close performance of the “pse” approach and
the “ne” approach. More numerical results can be found in Appendix F.

Conclusions: We have studied the performative phenomenon in a decentralized noncooperative game
where selﬁsh players seek to maximize their individual proﬁts while adhering to coupled inequality
constraints. We have derived sufﬁcient conditions for the E&U of both PSE and NE and provided the
ﬁrst upper bound on the distance between these two equilibria. Furthermore, we have developed a
decentralized stochastic primal-dual algorithm for efﬁciently computing of the PSE point. Theoretical
analysis has demonstrated the same order of convergence speed of our algorithm as the case without
performativity. Finally, numerical simulations have been provided to verify the effectiveness of our
algorithm and theoretical results.

9


---Page Break---
References

Melika Abolhassani, Mohammad Hossein Bateni, MohammadTaghi Hajiaghayi, Hamid Mahini, and Anshul
Sawant. 2014. Network cournot competition. In International Conference on Web and Internet Economics.
Springer, 15–29.

Blaise Allaz and Jean-Luc Vila. 1993. Cournot competition, forward markets and efﬁciency. Journal of Economic
theory 59, 1 (1993), 1–16.

Dimitri P Bertsekas. 2014. Constrained optimization and Lagrange multiplier methods. Academic press.

Stephen P Boyd and Lieven Vandenberghe. 2004. Convex optimization. Cambridge university press.

Alex Chan, Ahmed Alaa, Zhaozhi Qian, and Mihaela Van Der Schaar. 2020. Unlabelled data improves bayesian
uncertainty calibration under covariate shift. In International conference on machine learning. PMLR,
1392–1402.

James Cust and Steven Poelhekke. 2015. The local economic impacts of natural resource extraction. Annu. Rev.
Resour. Econ. 7, 1 (2015), 251–268.

Sarah Dean, Mihaela Curmei, Lillian J. Ratliff, Jamie Morgenstern, and Maryam Fazel. 2023. Emergent
segmentation from participation dynamics and multi-learner retraining. arXiv preprint arXiv:2206.02667
(2023).

Jinshuo Dong, Aaron Roth, Zachary Schutzman, Bo Waggoner, and Zhiwei Steven Wu. 2018. Strategic
classiﬁcation from revealed preferences. In Proceedings of the 2018 ACM Conference on Economics and
Computation. 55–70.

Dmitriy Drusvyatskiy and Lin Xiao. 2023. Stochastic optimization with decision-dependent distributions.
Mathematics of Operations Research 48, 2 (2023), 954–998.

Issam El Naqa and Martin J Murphy. 2015. What is machine learning? Springer.

Francisco Facchinei and Jong-Shi Pang. 2003. Finite-dimensional variational inequalities and complementarity
problems. Springer.

Francisco Facchinei and Jong-Shi Pang. 2010. Nash equilibria: the variational approach. Convex optimization in
signal processing and communications (2010), 443.

Bassam Fattouh and Lavan Mahadeva. 2014. Causes and implications of shifts in ﬁnancial participation in
commodity markets. Journal of Futures Markets 34, 8 (2014), 757–787.

Yiguang Hong, Jiangping Hu, and Linxin Gao. 2006. Tracking control for multi-agent consensus with an active
leader and variable topology. Automatica 42, 7 (2006), 1177–1182.

Roger A Horn and Charles R Johnson. 2012. Matrix analysis. Cambridge university press.

Zachary Izzo, Lexing Ying, and James Zou. 2021. How to learn when data reacts to your model: performative
gradient descent. In International Conference on Machine Learning. PMLR, 4641–4650.

Meena Jagadeesan, Tijana Zrnic, and Celestine Mendler-Dünner. 2022. Regret minimization with performative
feedback. In International Conference on Machine Learning. PMLR, 9760–9785.

Qiang Li, Chung-Yiu Yau, and Hoi-To Wai. 2022. Multi-agent performative prediction with greedy deployment
and consensus seeking agents. Advances in Neural Information Processing Systems 35 (2022), 38449–38460.

Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, and Guangquan Zhang. 2018. Learning under concept drift:
A review. IEEE transactions on knowledge and data engineering 31, 12 (2018), 2346–2363.

Kaihong Lu, Guangqi Li, and Long Wang. 2020. Online distributed algorithms for seeking generalized Nash
equilibria in dynamic environments. IEEE Trans. Automat. Control 66, 5 (2020), 2289–2296.

Songtao Lu. 2023. Bilevel optimization with coupled decision-dependent distributions. In International Confer-
ence on Machine Learning. PMLR, 22758–22789.

Debmalya Mandal, Stelios Triantafyllou, and Goran Radanovic. 2023. Performative reinforcement learning. In
International Conference on Machine Learning. PMLR, 23642–23680.

John P Miller, Juan C Perdomo, and Tijana Zrnic. 2021. Outside the echo chamber: Optimizing the performative
risk. In International Conference on Machine Learning. PMLR, 7710–7720.

10


---Page Break---
Usue Mori, Alexander Mendiburu, Maite Álvarez, and Jose A Lozano. 2015. A review of travel time estimation
and forecasting for advanced traveller information systems. Transportmetrica A: Transport Science 11, 2
(2015), 119–157.

Amir Moshari, GR Youseﬁ, Akbar Ebrahimi, and Saeid Haghbin. 2010. Demand-side behavior in the smart
grid environment. In 2010 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT Europe).
IEEE, 1–7.

Adhyyan Narang, Evan Faulkner, Dmitriy Drusvyatskiy, Maryam Fazel, and Lillian J Ratliff. 2023. Multiplayer
performative prediction: Learning in decision-dependent games. Journal of Machine Learning Research 24,
202 (2023), 1–56.

Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. 2020. Performative prediction. In
Proceedings of the 37th International Conference on Machine Learning (ICML 2020). PMLR, 7599–7609.

Georgios Piliouras and Fang-Yi Yu. 2023. Multi-agent performative prediction: From global stability and
optimality to chaos. In Proceedings of the 24th ACM Conference on Economics and Computation. 1047–
1074.

Joaquin Quinonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. 2008. Dataset
shift in machine learning. Mit Press.

Jia-Wei Shan, Peng Zhao, and Zhi-Hua Zhou. 2023. Beyond Performative Prediction: Open-environment
Learning with Presence of Corruptions. In International Conference on Artiﬁcial Intelligence and Statistics.
PMLR, 7981–7998.

Hal R Varian. 2009. Online ad auctions. American Economic Review 99, 2 (2009), 430–434.

Xiaolu Wang, Chung-Yiu Yau, and Hoi To Wai. 2023. Network effects in performative prediction games. In
International Conference on Machine Learning. PMLR, 36514–36540.

Killian Wood, Gianluca Bianchin, and Emiliano Dall’Anese. 2021. Online projected gradient descent for
stochastic optimization with decision-dependent distributions. IEEE Control Systems Letters 6 (2021),
1646–1651.

Ruihan Wu, Chuan Guo, Yi Su, and Kilian Q Weinberger. 2021. Online adaptation to label distribution shift.
Advances in Neural Information Processing Systems 34 (2021), 11340–11351.

Wenjing Yan and Xuanyu Cao. 2024a. Decentralized Multi-Task Online Convex Optimization Under Random
Link Failures. IEEE Transactions on Signal Processing (2024).

Wenjing Yan and Xuanyu Cao. 2024b. Zero-regret performative prediction under inequality constraints. Advances
in Neural Information Processing Systems 36 (2024).

Zhi-Hua Zhou. 2022. Open-environment machine learning. National Science Review 9, 8 (2022), nwac123.

11


---Page Break---
A
Related Work

In recent years, the exploration of distribution shifts in machine learning systems has been extended
beyond traditional exogenous shifts (Quinonero-Candela et al., 2008), such as covariate (Chan
et al., 2020), label (Wu et al., 2021), and concept (Lu et al., 2018) drifts, to include endogenous
shifts resulting from strategic behaviors within the learning platforms themselves. Perdomo et al.
(2020) introduced the framework of performative prediction, which captures the platform’s strategic
responses using decision-dependent distribution mappings. Following this seminal work, signiﬁcant
research effort has been dedicated to investigating the phenomenon of performativity in various
scenarios. In particular, (Shan et al., 2023) studied the endogenous distribution change in open
environments, where data are obtained from a corrupted decision-dependent distribution. They
proposed an effective algorithm with theoretical guarantees by decoupling the two sources of effects.
Lu (2023) investigated the presence of performativity in bilevel optimization. They ﬁrst established
sufﬁcient conditions for the existence of performatively stable solutions and then developed a
stochastic algorithm to ﬁnd the PS point. In (Mandal et al., 2023), the authors examined the
performative effect in a regularized reinforcement learning problem and showed that repeatedly
optimizing this objective converges to a performatively stable policy under reasonable assumptions
on the transition dynamics. It is demonstrated in (Drusvyatskiy and Xiao, 2023) that typical gradient-
based stochastic algorithms can be applied to ﬁnd performative stable equilibria with a biased gradient
oracle.

While most existing work focused on ﬁnding performative stable points, there are studies aimed at
identifying the optimal solutions for performative prediction problems (Miller et al., 2021; Izzo et al.,
2021; Jagadeesan et al., 2022). The optimality gap of performative stable points was ﬁrst presented
in (Perdomo et al., 2020), where their bound is proportional to the strong convexity parameter and
inversely proportional to the smoothness parameter of cost functions and the sensitivity parameter
of the decision-dependent distributions. The primary challenges in computing optimal points in
performative prediction problems lie in the unknown decision-dependent data distributions. To
address this challenge, a commonly used method is to make parametric assumptions on the data
distributions and then design algorithms to estimate them. For instance, (Miller et al., 2021) proposed
a two-stage algorithm to ﬁnd the performative optima for distribution maps in the location family. Izzo
et al. (2021) proposed a PerfGD algorithm by exploiting the exponential structure of the underlying
distribution maps.

Among the numerous existing studies, (Narang et al., 2023) and (Wang et al., 2023) are, at a
conceptual level, the closest papers to our own since they have considered performative behaviors in
games. On a technical level, however, these two works are quite distinct from ours since we study
completely different problem settings. One deﬁning distinction is that, in our model, the collective
strategies of all players must adhere to the learning system’s constraints, whereas both (Narang
et al., 2023) and (Wang et al., 2023) are unconstrained. Constraints are unavoidable in certain game
scenarios, such as safety and cost constraints in transportation, relevance and diversity constraints in
advertising, and risk tolerance and portfolio constraints in ﬁnancial trading. The constrained problem
in our work results in a fundamentally different algorithm design and convergence analysis from
these two papers. Our work utilizes the primal-dual technique and necessitates consensus, whereas
their approach only requires local stochastic gradient descent. Additionally, there are distinctions in
the problem settings. In (Wang et al., 2023), the private cost function of each player is structured
in a speciﬁc form, involving a local cost depending solely on its own strategy and a regularizer
quantifying the similarity of strategies among neighbors. In contrast, we consider a mathematically
richer setting where each player’s private cost function depends on the strategies of all players in the
game, thus encompassing the model in (Wang et al., 2023). Moreover, our algorithm design takes
into account the practical implementation where players can only communicate with their neighbors,
while (Narang et al., 2023) assumes that the strategies of all players are publicly accessible across the
entire network. This more practical setting poses challenges for each player in observing the entire
network. More importantly, although (Narang et al., 2023) and (Wang et al., 2023) demonstrated the
existence and uniqueness of the PSE and NE for their respective game settings, neither of them offers
insights into the distance between these two equilibria. This paper makes a signiﬁcant contribution
by presenting the ﬁrst upper bound on this distance.

Furthermore, there are works on decentralized optimization of multiagent performative prediction
(Li et al., 2022; Piliouras and Yu, 2023). Speciﬁcally, (Li et al., 2022) focused on decentralized

12


---Page Break---
optimization with consensus-seeking agents, where the data distribution of each agent depends only
on its own decision. Although (Piliouras and Yu, 2023) considers multiagent, their study is in a
centralized fashion and their data distributions are restricted to location-scale families. Lastly, it is
worth mentioning that one paper (Yan and Cao, 2024b) has considered constrained optimization in
the context of performative prediction. However, (Yan and Cao, 2024b) studied the single-agent case,
while this work considers a more complex model with decentralized noncooperative players and
partially observed information about competitors’ strategies. Additionally, this paper contributes to
the evaluation of equilibria, whereas such analysis has not been involved in (Yan and Cao, 2024b).

B
Existence and Uniqueness of Performative Stable Equilibrium

From the deﬁnition of the mapping T (θ), we have that

θ′
i = Ti(θ) = arg min
ui∈Ωi
Eξi∼Di(θ)Ji
 
ξi; ui, θ′
−i

s.t.
gi(ui) +
X

j̸=i
gj
 
θ′
j

≤0,
∀i ∈[n],

δ′
i = Ti(δ) = arg min
ui∈Ωi
Eξi∼Di(δ)Ji
 
ξi; ui, δ′
−i

s.t.
gi(ui) +
X

j̸=i
gj
 
δ′
j

≤0,
∀i ∈[n].

Deﬁne Eξi∼Di(θ)∇θiJi
 
ξi; θ′
i, θ′
−i

:= G(i)
θ (θ′
i, θ′
−i). From the optimality condition of constrained
optimization, we have
D
G(i)
θ
 
θ′
, θ′
i −δ′
i
E
≤0,
∀i ∈[n].

Deﬁne a vector Gθ(θ′) := col

G(1)
θ (θ′), · · · , G(n)
θ (θ′)

that concatenates all the G(i)
θ (θ′), i ∈[n].
Then, we have

Gθ
 
θ′
, θ′ −δ′
≤0.
(A1)

Similarly, we have

Gδ
 
δ′
, θ′ −δ′
≥0.
(A2)

Further, from the monotoniticy of the gradient mapping ∇J (ξ; θ) in Assumption 2.1, we have

Gθ(θ′) −Gθ(δ′), θ′ −δ′
= Eξ∼D(θ)

∇J
 
ξ; θ′
−∇J
 
ξ; δ′
, θ′ −δ′
≥µ∥θ′ −δ′∥2
2,
(A3)

where D(θ) := D1(θ) × · · · × Dn(θ). Plugging (A1) and (A2) into (A3) gives

µ∥θ′ −δ′∥2
2 ≤

−Gθ
 
δ′
, θ′ −δ′

≤

Gδ
 
δ′
−Gθ
 
δ′
, θ′ −δ′

≤
Gδ
 
δ′
−Gθ
 
δ′
2
θ′ −δ′
2 .
(A4)

From Assumption 2.2, W1
 
Di (θ) , Di
 
θ′
≤εi
qPn
j=1 pij
θj −θ′
j
2
2. Along with Assumption
2.4, we have that

Gδ
 
δ′
−Gθ
 
δ′2
2 ≤

n
X

i=1

n
X

j=1
L2
i ε2
i pij ∥δj −θj∥2
2

≤

n
X

i=1
L2
i ε2
i max
j∈[n] pij ∥δ −θ∥2
2 .

Plugging the above result into (A4) yields

∥θ′ −δ′∥2 ≤1

µ

v
u
u
t

n
X

i=1
L2
i ε2
i max
j∈[n] pij ∥δ −θ∥2 .

13


---Page Break---
From the RRM procedure, we know that θt+1 = T (θt) and the PSE satisﬁes θpse = T (θpse). Then,
we have

θt+1 −θpse
2 ≤1

µ

v
u
u
t

n
X

i=1
L2
i ε2
i max
j∈[n] pij
θt −θpse
2

≤



1

µ

v
u
u
t

n
X

i=1
L2
i ε2
i max
j∈[n] pij





t
θ1 −θpse
2 .

Further, if for any player i, its distribution Di depends only on its own decision θi, i.e., pij = 0 and
pii = 1 for all i, j ∈[n] and j ̸= i, then, we have

 
Gδ
 
δ′
−Gθ
 
δ′
2 ≤

v
u
u
t

n
X

i=1
L2
i ε2
i ∥δi −θi∥2
2 ≤max
i∈[n] Liεi ∥δ −θ∥2 .
(A5)

Plugging (A5) into (A4) yields

∥θ′ −δ′∥2 ≤1

µ max
i∈[n] Liεi ∥δ −θ∥2 .

Correspondingly, we have

θt+1 −θpse
2 ≤
 1

µ max
i∈[n] Liεi

t θ1 −θpse
2 .

C
Existence and Uniqueness of Nash Equilibrium

Based on the results in Facchinei and Pang (2003, Theorem 2.3.3(b)), to show the existence and
uniqueness of NE, we need to prove that the gradient mapping ∇PR(θ) of the performative game
(1) is strongly monotone, i.e., there exists a α > 0 such that ⟨∇PR(θ) −∇PR(θ), θ −δ⟩≥
α ∥θ −δ∥2
2, where α denotes the strongly-monotone parameter. Since ∇PR(θ) = Gθ(θ) + Hθ(θ),
we have

⟨∇PR(θ) −∇PR(δ), θ −δ⟩= ⟨Gθ(θ) −Gδ(δ), θ −δ⟩+ ⟨Hθ(θ) −Hδ(δ), θ −δ⟩.

From Assumption 2.2, we have

⟨Gθ(θ) −Gδ(θ), θ −δ⟩≥−

n
X

i=1
Liεi max
j∈[n]
√pij ∥θ −δ∥2
2 .

Moreover, from the monotonicity of the gradient mapping ∇J (ξ; θ) in Assumption 2.1, we have

⟨Gδ(θ) −Gδ(δ), θ −δ⟩= Eξ∼D(δ) ⟨∇J(ξ; θ) −∇J(ξ; δ), θ −δ⟩≥µ ∥θ −δ∥2
2 .

Combining the above two inequalities yields

⟨Gθ(θ) −Gδ(δ), θ −δ⟩= ⟨Gθ(θ) −Gδ(θ), θ −δ⟩+ ⟨Gδ(θ) −Gδ(δ), θ −δ⟩

≥

 

µ −

n
X

i=1
Liεi max
j∈[n]
√pij

!

∥θ −δ∥2
2 .
(A6)

Further, let γ(s) = θ′ + s
 
θ −θ′
for s ∈(0, 1). Then, we have

Ji (ξi; θ) −Ji
 
ξi; θ′
=
Z 1

0


∇Ji
 
ξi; θ′ + s
 
θ −θ′
, θ −θ′
ds

=
Z 1

0


∇Ji (ξi; γ(s)) , θ −θ′
ds.
(A7)

14


---Page Break---
From the deﬁnition of H(i)
θ (δ) that H(i)
θ (δ) := ∇uiEξi∼Di(ui,θ−i) [Ji (ξi; δ)]

ui=θi, we have that

H(i)
θ (θ) −H(i)
θ (θ′) = ∇uiEξi∼Di(ui,θ−i)

Z 1

0


∇Ji (ξi; γ(s)) , θ −θ′
ds

ui=θi

=
Z 1

0
∇uiEξi∼Di(ui,θ−i)

∇Ji (ξi; γ(s)) , θ −θ′
ui=θi
ds.
(A8)

From Assumption 2.4, we have
Eξi∼Di∇Ji (ξi; θ) −Eξ′
i∼D′
i∇Ji
 
ξ′
i; θ

2 ≤LiW1(Di, D′
i).

Along with Assumption 2.2, we know that the function Eξi∼Di(θi,θ−i)∇Ji
 
ξi; θ′
is Liεipii-
Lipschitz continuous w.r.t θi, and thus its gradient satisﬁes
∇uiEξi∼Di(ui,θ−i) [∇Ji (ξi; γ(s))]

ui=θi


2 ≤Liεipii.
(A9)

Combing (A8) and (A9) gives
H(i)
θ (θ) −H(i)
θ (θ′)

2 ≤
Z 1

0

∇uiEξi∼Di(ui,θ−i) [∇Ji (ξi; γ(s))]

ui=θi


2

θ −θ′
2 ds

≤Liεipii
θ −θ′
2 ,

where the ﬁrst inequality holds due to the Cauchy-Schwartz inequality. This further implies that

Hθ(θ) −Hθ(θ′)

2 =

v
u
u
t

n
X

i=1

H(i)
θ (θ) −H(i)
θ (θ′)

2

2

≤

v
u
u
t

n
X

i=1
L2
i ε2
i pii
θ −θ′
2 .

Following prior work (Narang et al., 2023) and (Wang et al., 2023) on performative games, we assume
that the mapping Hδ(θ) is monotone w.r.t δ, i.e., ⟨Hθ(θ) −Hδ(θ), θ −δ⟩≥0. Then, we have that

⟨∇PR(θ) −∇PR(δ), θ −δ⟩= ⟨Gθ(θ) −Gδ(δ), θ −δ⟩+ ⟨Hθ(θ) −Hδ(δ), θ −δ⟩
= ⟨Gθ(θ) −Gδ(θ), θ −δ⟩+ ⟨Hθ(θ) −Hδ(θ), θ −δ⟩
+ ⟨Gδ(θ) −Gδ(δ), θ −δ⟩+ ⟨Hδ(θ) −Hδ(δ), θ −δ⟩

≥



µ −

n
X

i=1
Liεi max
j∈[n]
√pij −

v
u
u
t

n
X

i=1
L2
i ε2
i pii



∥θ −δ∥2
2 .

Based on the classical result that a strongly monotone game over a non-empty, closed, and convex set
admits a unique NE Facchinei and Pang (2003, Theorem 2.3.3(b)), we have the E&U condition for
the NE of the game (1) as given in theorem 3.4.

D
Distance Between PSE and NE

The computation on the distance between the PSE and NE of the game (1) is based on the strong
duality (Boyd and Vandenberghe, 2004; Facchinei and Pang, 2010). Recall the deﬁnitions in Section
4.1 that

L(i)
δ (θi, θ−i, λ) := Eξi∼Di(δ)Ji (ξi; θi, θ−i) +

*

λ, gi(θi) +
X

j̸=i
gj (θj)

+

.

Moreover, deﬁne a gradient mapping φi(ξi; θ, λ) := ∇θiJi (ξi; θ)+∇gi(θi)⊤λ and a concatenation
vector φ := [φ1, · · · , φn]⊤. For any i ∈[n], since (θpse
i
, λpse) is a saddle point of the Lagrangian
L(i)
θpse(θi, θpse
−i , λ) under ξi ∼Di(θpse), we have that

L(i)
θpse
 
θpse
i
, θpse
−i , λ

≤L(i)
θpse
 
θpse
i
, θpse
−i , λpse
≤L(i)
θpse
 
θi, θpse
−i , λpse
∀θi ∈Ωi, λ ∈Rm
+.

15


---Page Break---
Similarly, for any i
∈
[n], (θne
i , λne) the saddle point of the regularized Lagrangian
L(i)
θi,θne
−i(θi, θne
−i, λ) with decision-dependent distribution ξi ∼Di(θi, θne
−i). Setting λ = λne

in the ﬁrst part of the proceeding inequality, we obtain

0 ≤L(i)
θpse
 
θpse
i
, θpse
−i , λpse
−L(i)
θpse
 
θpse
i
, θpse
−i , λne
= (λpse −λne)⊤g (θpse) , ∀i ∈[n],

where (λpse −λne)⊤g (θpse) = Pm
j=1
 
λpse
j
−λne
j

(Pn
i=1 gji (θpse
i
)). By the convexity of gji(·)
for all j ∈[m], i ∈[n], we have that

n
X

i=1
gji (θpse
i
) ≤

n
X

i=1
(gji (θne
i ) + ⟨∇gji (θpse
i
) , θpse
i
−θne
i ⟩)

≤

n
X

i=1
⟨∇gji (θpse
i
) , θpse
i
−θne
i ⟩, ∀j ∈[m],

where the last inequality follows from that gj(θne) = Pn
i=1 gji (θne
i ) ≤0. Multiplying the preceding
inequality with λpse
j
and adding over all j ∈[m], we obtain

m
X

j=1

n
X

i=1
λpse
j
gji (θpse
i
) = (λpse)⊤g (θpse) ≤

n
X

i=1

* m
X

j=1
λpse
j
∇gji (θpse
i
) , θpse
i
−θne
i

+

=

n
X

i=1


∇gi(θpse
i
)⊤λpse, θpse
i
−θne
i

.
(A10)

By the deﬁnition of the mapping φi(·), for any ξi ∈Ξi, we have that,

∇gi(θpse
i
)⊤λpse = φi(ξi; θpse, λpse) −∇θiJi (ξi; θpse) , ∀i ∈[n].
(A11)

Plugging (A11) into (A10) gives

(λpse)⊤g (θpse) ≤

n
X

i=1
⟨φi(ξi; θpse, λpse) −∇θiJi (ξi; θpse) , θpse
i
−θne
i ⟩, ∀i ∈[n].
(A12)

Likewise, we have the following inequality based on the convexity of the functions {gji(·)}:

gji (θpse
i
) ≥gji (θne
i ) + ⟨∇gji (θne
i ) , θpse
i
−θne
i ⟩, ∀j ∈[m], i ∈[n].

Multiplying the preceding inequality with −λne
j and summing over j ∈[m], we obtain

−

m
X

j=1
λne
i

n
X

i=1
gji (θpse
i
) ≤−

m
X

j=1
λne
j

n
X

i=1
gji (θne
i ) −

n
X

i=1

* m
X

j=1
λne
j ∇gji (θne
i ) , θpse
i
−θne
i

+

=

n
X

i=1

D
∇gi (θne
i )⊤λne, θne
i −θpse
i
E
,

where the equality follows from that Pm
j=1 λne
j
Pn
i=1 gji (θne
i ) = (λne)⊤g (θne) = 0, which holds

by the complementary slackness condition of the Lagrangian L(i)
θi,θne
−i(θi, θne
−i, λ) for all i ∈[n].
Similar to (A12), we have

−(λne)⊤g (θpse) ≤

n
X

i=1
⟨φi(ξi; θne, λne) −∇θiJi (ξi; θne) , θne
i −θpse
i
⟩.
(A13)

Combining (A12) and (A13) yields

(λpse −λne)⊤g (θpse) ≤

n
X

i=1
⟨φi(ξi; θpse, λpse) −φi(ξi; θne, λne), θpse
i
−θne
i ⟩

−

n
X

i=1
⟨∇θiJi (ξi; θpse) −∇θiJi (ξi; θne) , θpse
i
−θne
i ⟩.

16


---Page Break---
Taking expectation on both sides of the above inequality over the distribution Di(θpse) for all i ∈[n]
gives

(λpse −λne)⊤g (θpse) ≤

n
X

i=1
Eξi∼Di(θpse) ⟨φi(ξi; θpse, λpse) −φi(ξi; θne, λne), θpse
i
−θne
i ⟩

−

n
X

i=1

D
G(i)
θpse (θpse) −G(i)
θpse (θne) , θpse
i
−θne
i
E
.
(A14)

Since (θpse
i
, λpse) is a saddle point of the Lagrangian L(i)
θpse(θpse, λpse) given ξi ∼Di(θpse), we
have that

Eξi∼Di(θpse) ⟨φi(ξi; θpse, λpse), θpse
i
−θne
i ⟩≤0, ∀i ∈[n].
(A15)

Furthermore, for any i ∈[n], we have

Eξi∼Di(θpse)φi(ξi; θne, λne) = G(i)
θpse (θne) + ∇gi(θne
i )⊤λne

+ ∇θiPRi(θne
i , θne
−i) −∇θiPRi(θne
i , θne
−i).
(A16)

Since (θne
i , λne) is a saddle point of the Lagrangian L(i)
θi,θne
−i(θi, θne
−i, λne) with decision-dependent
distribution Di(θi, θne
−i), we have that

−

∇θiPRi(θne
i , θne
−i) + ∇gi(θne
i )⊤λne, θpse
i
−θne
i

≤0, ∀i ∈[n].
(A17)

Plugging (A15), (A16), and (A17) into (A14) yields

0 ≤(λpse −λne)⊤g (θpse)

≤

n
X

i=1

D
∇iPRi(θne
i , θne
−i) −G(i)
θpse (θne) , θpse
i
−θne
i
E

−

n
X

i=1

D
G(i)
θpse (θpse) −G(i)
θpse (θne) , θpse
i
−θne
i
E

=

n
X

i=1

D
H(i)
θne(θne) + G(i)
θne (θne) −G(i)
θpse (θpse) , θpse
i
−θne
i
E
.

Then, we have

⟨Gθpse (θpse) −Gθne (θne) , θpse −θne⟩≤⟨Hθne(θne), θpse −θne⟩.

From the result in (A6) and the Cauchy-Schwarz inequality, we have
 

µ −

n
X

i=1
Liεi max
j∈[n]
√pij

!

∥θpse −θne∥2
2 ≤∥Hθne(θne)∥2∥θpse −θne∥2.

Since the cost function Ji(·) is Gi Lipschitz for any i ∈[n], along with Assumption 2.2, we have

∥Hθne(θne)∥2 =

v
u
u
t

n
X

i=1
∥H(i)
θne
i ,θne
−i(θne
i , θne
−i)∥2
2 ≤

v
u
u
t

n
X

i=1
G2
i ε2
i pii.

Combining the above results yields

∥θpse −θne∥2 ≤

pPn
i=1 G2
i ε2
i pii
µ −Pn
i=1 Liεi maxj∈[n] √pij
.

Further, from Assumption 2.2, we have

|PRi(θpse) −PRi(θne)| ≤Gi∥θpse −θne∥2 + Giεi

v
u
u
t

n
X

j=1
pij
θpse
j
−θne
j
2

2

≤Gi


1 + εi max
j∈[n]
√pij


∥θpse −θne∥2.

17


---Page Break---
Then, we have

|PR(θpse) −PR(θne)| =

n
X

i=1
|PRi(θpse) −PRi(θne)|

≤

 n
X

i=1
Gi


1 + εi max
j∈[n]
√pij

!
pPn
i=1 G2
i ε2
i pii
µ −Pn
i=1 Liεi maxj∈[n] √pij
.

E
Convergence of the Decentralized Stochastic Primal-Dual Algorithm

The proof of this section utilizes the following supporting lemmas.
Lemma E.1. Based on the update rule of the dual variable λ in Algorithm 1, for any γt ≥0,
λt
i ∈Rm
+, i ∈[n], and t ∈[T], we have that Pn
i=1 ∥γtλt
i∥2
2 ≤nB2.

Lemma E.2. Deﬁne λ
t := 1

n
Pn
i=1 λt
i the average of the dual variable over all players at the tth
iteration. Then, for any γt ≥0 and t ∈[T], we have the following relationship:

−

T
X

t=1

n
X

i=1
γt(λt
i)⊤gi(θt
i) ≤−

T
X

t=1

n
X

i=1
γtλ⊤gi
 
θt
i

+ n

2

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2 + 9

2

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2

+ 2(1 + √n)B

T
X

t=1
γt

n
X

i=1

λt
i −λ
t
2 + 4nB2
T
X

t=1
γ2
t .

Moreover, we require the following Lemma on the weight matrix A.
Lemma E.3. Let σ2(A) denote the second-largest eigenvalue of the weight matrix A. Since A is
assumed to be doubly stochastic, it holds that σ2(A) < 1 (Horn and Johnson, 2012). Furthermore,
for any i ∈[n], we construct a weight matrix A−
i by removing the ith row and column of A. Let β
represent the maximum eigenvalue of A−
i for all i ∈[n]. It has been established in Hong et al. (2006,
Lemma 3) that β < 1.

With Lemma E.3, we have the following results.

Lemma E.4. Deﬁne et
ih := bθ
t
ih −θt
h the estimation error of player i on the decision of player
h at the tth iteration, for all i, h ∈[n] and t ∈[T]. Let et
h denote the concatenation of et
ih that

et
h := col

et
1h, · · · , et
(h−1)h, et
(h+1)h, · · · , et
nh

. Then, the sum of ∥et
h∥2 over h ∈[n] and t ∈[T]
satisﬁes

T
X

t=1

n
X

h=1
E∥et
h∥2 ≤
nC
1 −β + n√n −1(G + √nBGg)

1 −β

T
X

t=1
γt = O

 T
X

t=1
γt

!

.

Moreover, the sum of ∥et
ih∥2
2 over h ∈[n] and t ∈[T] satisﬁes

T
X

t=1

n
X

h=1
E∥et
h∥2
2 ≤2nC2

1 −β + 2n(n −1)(G + √nBGg)2

(1 −β)2

T
X

t=1
γt = O

 T
X

t=1
γt

!

.

Lemma E.5. With the deﬁnition λ
t :=
1
n
Pn
i=1 λt
i, we have the following relationship on the

consensus error of the dual variable λt
i, given by λt
i −λ
t, for all i ∈[n] and t ∈[T]:

T
X

t=1

n
X

i=1

λt
i −λ
t
2 ≤2(n + √n)B

1 −σ2(A)

T
X

t=1
γt = O

 T
X

t=1
γt

!

,

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2 ≤4(n + √n)2B2

(1 −σ2(A))2

T
X

t=1
γt = O

 T
X

t=1
γt

!

.

18


---Page Break---
Next, we start the proof of Theorem 4.2. For ease of proposition, we deﬁne the following gra-
dient mappings: for any t ∈[T], φt
i(ξi; θi, θ−i, λ) := ∇iJi (ξi; θi, θ−i, θ) + γt∇gi(θi)⊤λ,
φt(·)
:=
[φt
1(·), · · · , φt
n(·)]⊤, Φi,t
δ (θ, λ)
:=
G(i)
δ (θ) + γt∇gi(θi)⊤λ, and Φt
δ(θ, λ)
:=
h
Φ1,t
δ (θ, λ), · · · , Φn,t
δ (θ, λ)
i⊤
. Then, we have

E
θt+1 −θpse2

2 =

n
X

i=1
E
PΩi
h
θt
i −γtφt
i

ξt
i; θt
i, bθ
t
i, λt
i
i
−PΩi
h
θpse
i
−γtΦi,t
θpse (θpse, λpse)
i
2

2

≤E
θt −θpse2
2 + γ2
t

n
X

i=1
E
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)

2

2

−2γt

n
X

i=1
E
D
θt
i −θpse
i
, φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)
E
.
(A18)

The second term on the right side of (A18) is handled as follows.

γ2
t

n
X

i=1
E
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)

2

2

= γ2
t

n
X

i=1
E
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θt

θt
i, bθ
t
i, λt
i

+ Φi,t
θt

θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)

2

2

≤3γ2
t

n
X

i=1
E
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θt

θt
i, bθ
t
i, λt
i

2

2
|
{z
}
(a)

+ 3γ2
t

n
X

i=1
E
G(i)
θt

θt
i, bθ
t
i

−G(i)
θpse (θpse)

2

2
|
{z
}
(b)

+ 3γ4
t

n
X

i=1
E
∇gi
 
θt
i
⊤λt
i −∇gi (θpse
i
)⊤λpse
2

2
|
{z
}
(c)

.
(A19)

We have the following results on these three terms in the last inequality of (A19).

(a) = 3γ2
t

n
X

i=1
E
∇θiJi

ξt
i; θt
i, bθ
t
i

−G(i)
θt

θt
i, bθ
t
i

2

2

≤3γ2
t

σ2
0 + σ2
1E
θt −θpse2
2


.

(b) = 3γ2
t

n
X

i=1
E
G(i)
θt

θt
i, bθ
t
i

−G(i)
θt
 
θt
+ G(i)
θt
 
θt
−G(i)
θt (θpse) + G(i)
θt (θpse) −G(i)
θpse (θpse)

2

2

≤9γ2
t

n
X

i=1
E
G(i)
θt

θt
i, bθ
t
i

−G(i)
θt
 
θt
2

2 +
G(i)
θt
 
θt
−G(i)
θt (θpse)

2

2 +
G(i)
θt (θpse) −G(i)
θpse (θpse)

2

2



≤9γ2
t

n
X

i=1
E

L2
i
bθ
t
i −θt
−i

2

2 + L2
i
θt −θpse2
2 + L2
i ε2
i max
j∈[n] pij
θt −θpse2
2


,

where the last inequality is based on Assumptions 2.2 and 2.4. Further, since the constriant function
gi(·) is Gg Lipschitz for all i ∈[n], we have that

(c) ≤6γ4
t

n
X

i=1
E
∇gi
 
θt
i
⊤λt
i

2

2 + 6γ4
t

n
X

i=1
E
∇gi (θpse
i
)⊤λpse
2

2

≤6γ2
t G2
g

n
X

i=1
E∥γtλt
i∥2
2 + 6γ4
t nG2
g∥λpse∥2
2

≤6γ2
t nB2G2
g + 6γ4
t nG2
g∥λpse∥2
2,

19


---Page Break---
where the last inequality is based on Lemma E.1.

Plugging the results of (a), (b), and (c) into (A19) gives

γ2
t

n
X

i=1
E
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)

2

2

≤3γ2
t σ2
0 + 3γ2
t

 

σ2
1 + 3

n
X

i=1
L2
i


1 + ε2
i max
j∈[n] pij

!

E
θt −θpse2
2

+ 9γ2
t

n
X

i=1
L2
i E
bθ
t
i −θt
−i

2

2 + 6γ2
t nB2G2
g + 6γ4
t nG2
g∥λpse∥2
2.
(A20)

Next, we deal with the last term on the right side of (A18). First, we have the following inequality:

E
h
φt
i

ξt
i; θt
i, bθ
t
i, λt
i

−Φi,t
θpse (θpse, λpse)
i

= E
h
G(i)
θt

θt
i, bθ
t
i

−G(i)
θpse (θpse)
i
+ γtE
h
∇gi
 
θt
i
⊤λt
i −∇gi (θpse
i
)⊤λpsei
.

Moreover, we have

−2γt

n
X

i=1
E
D
θt
i −θpse
i
, G(i)
θt

θt
i, bθ
t
i

−G(i)
θpse (θpse)
E

= −2γt

n
X

i=1
E
D
θt
i −θpse
i
, G(i)
θt

θt
i, bθ
t
i

−G(i)
θt
 
θtE
−2γtE

θt −θpse, Gθt  
θt
−Gθt (θpse)


−2γtE

θt −θpse, Gθt (θpse) −Gθpse (θpse)


≤4Cγt

n
X

i=1
LiE
bθ
t
i −θt
−i

2 −2µγtE
θt −θpse2
2 + 2γt

n
X

i=1
Liεi max
j∈[n]
√pijE
θt −θpse2
2 ,

(A21)

where the last inequality is from Assumptions 2.2, 2.3, 2.4 and the Cauchy-Schwarz inequality.

Further, we have

−2γ2
t

n
X

i=1
E
D
θt
i −θpse
i
, ∇gi
 
θt
i
⊤λt
i −∇gi (θpse
i
)⊤λpseE

≤2γ2
t

n
X

i=1
E
D
θpse
i
−θt
i, ∇gi
 
θt
i
⊤λt
i
E
+ 4γ2
t CGg∥λpse∥2

≤2γ2
t

n
X

i=1
E
h 
gi (θpse
i
) −gi
 
θt
i
⊤λt
i
i
+ 4γ2
t CGg∥λpse∥2

≤2γ2
t E

" n
X

i=1
gi (θpse
i
)⊤
λt
i −λ
t
+ g (θpse)⊤λ
t −

n
X

i=1
gi
 
θt
i
⊤λt
i

#

+ 4γ2
t CGg∥λpse∥2

≤2γ2
t

n
X

i=1
E
h
∥gi (θpse
i
)∥2
λt
i −λ
t
2 −gi
 
θt
i
⊤λt
i
i
+ 4γ2
t CGg∥λpse∥2,
(A22)

where the last inequality uses the fact that g (θpse)⊤λ
t ≤0.

Deﬁne eµ := µ −Pn
i=1 Liεi maxj∈[n] √pij, ν := 3
 
σ2
1 + 3 Pn
i=1 L2
i
 
1 + ε2
i maxj∈[n] pij

, and
π := 3σ2
0 + 6nB2G2
g + 6nG2
g∥λpse∥2
2 + 4CGg∥λpse∥. Plugging the results in (A20), (A21), and

20


---Page Break---
(A22) into (A18) yields

E
θt+1 −θpse2

2

≤
 
1 −2γteµ + νγ2
t

E
θt −θpse2
2 + 4Cγt

n
X

i=1
LiE
bθ
t
i −θt
−i

2 + 9γ2
t

n
X

i=1
L2
i E
bθ
t
i −θt
−i

2

2

+ 2γ2
t

n
X

i=1
E
h
B
λt
i −λ
t
2 −gi
 
θt
i
⊤λt
i
i
+ πγ2
t .
(A23)

Let supt≥1 γt ≤eµ

ν , then 1 −2eµγt + νγ2
t ≤1 −eµγt. Thus, we have

E
θt −θpse2
2

≤
1
eµγt


E
θt −θpse2
2 −E
θt+1 −θpse2

2


+ 4C

eµ

n
X

i=1
LiE
bθ
t
i −θt
−i

2

+ 9γt

eµ

n
X

i=1
L2
i E
bθ
t
i −θt
−i

2

2 + 2γt

eµ

n
X

i=1
E
h
B
λt
i −λ
t
2 −gi
 
θt
i
⊤λt
i
i
+ πγt

eµ .

Summing the above inequality over t ∈[T] and plugging into the result of Lemma E.2 yields

T
X

t=1
E
θt −θpse2
2 ≤

T
X

t=1

1
eµγt


E
θt −θpse2
2 −E
θt+1 −θpse2

2


+ 4C

eµ

T
X

t=1

n
X

i=1
LiE
bθ
t
i −θt
−i

2

+ 9

eµ

T
X

t=1
γt

n
X

i=1
L2
i E
bθ
t
i −θt
−i

2

2 + 2

eµ
 
3 + 2√n

B

T
X

t=1
γt

n
X

i=1
E
λt
i −λ
t
2

+ 9

eµ

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2 + π

eµ

T
X

t=1
γt + 8nB2

eµ

T
X

t=1
γ2
t

−2

eµ

T
X

t=1

n
X

i=1
γtλ⊤gi(θt
i) + n

eµ

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2.
(A24)

Since
θt −θpse2
2 ≤4C2, we have that

T
X

t=1

1
γt


E
θt −θpse2
2 −E
θt+1 −θpse2

2



= 1

γ1
E
θ1 −θpse2

2 −1

γT
E
θT +1 −θpse
2

2 +

T
X

t=2

 1

γt
−
1
γt−1


E
θt −θpse2
2

≤1

γ1
4C2 +

T
X

t=2

 1

γt
−
1
γt−1


4C2 ≤4C2

γT
,
(A25)

where in the last inequality is based on the fact that 1

γt −
1
γt−1 ≥0 because γt is a non-increasing
sequence. Further, we have the following relations:
n
X

i=1
L2
i E
bθ
t
i −θt
−i

2

2 ≤max
i
Li

n
X

i=1

X

h̸=i

bθ
t
ih −θt
h

2

2 = max
i
Li

n
X

h=1
∥et
h∥2
2,
(A26)

n
X

i=1
LiE
bθ
t
i −θt
−i

2 ≤max
i
Li

n
X

i=1

sX

h̸=i

bθ
t
ih −θt
h

2

2

≤max
i
Li

v
u
u
tn

n
X

i=1

X

h̸=i

bθ
t
ih −θt
h

2

2

≤max
i
Li
√n

n
X

h=1
∥et
h∥2,
(A27)

21


---Page Break---
where the last inequality is based on the fact that
√

a + b + c ≤√a +
√

b + √c for any a, b, c ≥0.
Plugging (A25), (A26) and (A27) into (A24) and utilizing the results in Lemmas E.4 and E.5, we
have that

T
X

t=1
E
θt −θpse2
2 + 2

eµ

T
X

t=1

n
X

i=1
γtλ⊤gi
 
θt
i

−n

eµ

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2

≤O

 
1
eµγT
+ 1

eµ

T
X

t=1
γt

!

.
(A28)

Since any λ ∈Rm
+ satisﬁes the above inequality, by setting λ = [
PT
t=1 γt
Pn
i=1 gi(θt
i)]+
n(1+PT
t=1 γ2
t )
, we have that

2
eµλ⊤
 T
X

t=1
γt

n
X

i=1
gi
 
θt
i

!

−n

eµ

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2 =


hPT
t=1 γt
Pn
i=1 gi
 
θt
i
i

+



2

2
eµn

1 + PT
t=1 γ2
t

.
(A29)

As the terms in (A29) is non-negative, omitting it in (A28) gives

T
X

t=1
E
θt −θpse2
2 ≤O

 
1
eµγT
+ 1

eµ

T
X

t=1
γt

!

.

Furthermore, since Eξi∼D(θpse)|Ji(ξi; θt
i, θpse
−i ) −Ji(ξi; θpse)| ≤Gi
θt
i −θpse
i

2, for any i ∈[n],
we have that

Ri(T) =

T
X

t=1

 
Eξi∼D(θpse)

J
 
ξi; θt
i, θpse
−i

−J (ξi; θpse)


≤Gi

T
X

t=1

θt
i −θpse
i

2

≤Gi

v
u
u
tT

T
X

t=1

θt
i −θpse
i
2
2

≤O





v
u
u
tT

eµ

 
1
γT
+

T
X

t=1
γt

!

, ∀i ∈[n].

On
the
other
hand,
plugging
(A29)
into
(A28)
and
omitting
the
non-negtive
term
PT
t=1 E
θt −θpse2
2, we have


hPT
t=1 γt
Pn
i=1 gi
 
θt
i
i

+



2

2
eµn

1 + PT
t=1 γ2
t

≤O

 
1
eµγT
+ 1

eµ

T
X

t=1
γt

!

.



" T
X

t=1
γt

n
X

i=1
gi
 
θt
i

#

+


2
≤O





v
u
u
t

 
1
γT
+

T
X

t=1
γt

!  

1 +

T
X

t=1
γ2
t

!

.

Then, we prove that

Rg(T) ≤O



1

γT

v
u
u
t

 
1
γT
+

T
X

t=1
γt

!  

1 +

T
X

t=1
γ2
t

!

.

22


---Page Break---
E.1
Proof of Lemma E.1

From the update rule of the dual variables, for any λt
i ∈Rm
+, i ∈[n], and t ∈[T], we have that

n
X

i=1

λt+1
i
2

2 ≤

n
X

i=1



n
X

j=1
aij
 
1 −γ2
t

λt
j + γtgi(θt
i)



2

2

≤

n
X

i=1

n
X

j=1
aij

(1 −γ2
t )λt
j + γ2
t
gi(θt
i)
γt



2

2

≤

n
X

i=1

n
X

j=1
aij
h
(1 −γ2
t )
λt
j
2

2 +
gi(θt
i)
2
2

i

≤(1 −γ2
t )

n
X

i=1

λt
i
2
2 +

n
X

i=1

gi(θt
i)
2
2

≤(1 −γ2
t )

n
X

i=1

λt
i
2
2 + nB2.

We next bound Pn
i=1
λt
i
2
2, ∀t ∈[T] by deduction. First, since λ1
i = 0, γ1 ≤1, and ∥gi(θ1
i )∥2
2 ≤

B2, ∀i ∈[n], we have that Pn
i=1
λ2
i
2

2 ≤Pn
i=1
gi(θ1
i )
2

2 ≤nB2 ≤
nB2

γ2
1 . Assume that
Pn
i=1 ∥λt
i∥2
2 ≤nB2

γ2
t−1 . Since {γt}t∈[T ] is a non-incerasing sequence, Pn
i=1 ∥λt
i∥2
2 ≤nB2

γ2
t−1 ≤nB2

γ2
t
and thus Pn
i=1
λt+1
i
2

2 ≤(1 −γ2
t ) nB2

γ2
t
+ nB2 =
nB2

γ2
t . Therefore, for any t ∈[T], we have
Pn
i=1 ∥λt+1
i
∥2
2 ≤nB2

γ2
t
≤nB2

γ2
t+1 , i.e., Pn
i=1 ∥γtλt
i∥2
2 ≤nB2, which completes the proof.

E.2
Proof of Lemma E.2

From the update rule of the dual variables λi, for any λ ∈Rm
+, we have that

n
X

i=1

λt+1
i
−λ
2

2 =

n
X

i=1





 
1 −γ2
t
 X

j∈Ni
aijλt
j + γtgi
 
θt
i





+

−λ



2

2

≤

n
X

i=1



 
1 −γ2
t
 X

j∈Ni
aij
 
λt
j −λt
i

+
 
λt
i −λ

+ γt
 
gi
 
θt
i

−γtλt
i



2

2

≤

n
X

i=1



X

j∈Ni
aij
λt
j −λt
i
2

2 +
λt
i −λ
2
2 + γ2
t
gi
 
θt
i

−γtλt
i
2
2

+ 2
X

j∈Ni
aij

λt
j −λt
i, λt
i −λ

+ 2γt

λt
i −λ, gi
 
θt
i

−γtλt
i


+2γt
X

j∈Ni
aij
λt
j −λt
i

2
gi
 
θt
i

−γtλt
i

2



,
(A30)

where we use the fact 1 −γ2
t ≤1 in (A30). Next, we simplify the terms in (A30). First, based on the
inequality (a −b)2 ≤2
 
a2 + b2
for any a, b ≥0, we have that

n
X

i=1

n
X

j=1
aij
λt
j −λt
i
2

2 =

n
X

i=1

n
X

j=1
aij



λt
j −λ
t
−

λt
i −λ
t
2

2


≤4

n
X

i=1

λt
i −λ
t
2

2 .

23


---Page Break---
In addition, with the result in Lemma E.1, we know that

n
X

i=1

gi
 
θt
i

−γtλt
i
2
2 ≤2

n
X

i=1

gi
 
θt
i
2
2 + 2

n
X

i=1

γtλt
i
2
2 ≤2nB2 + 2nB2 = 4nB2,

gi
 
θt
i

−γtλt
i

2 ≤
gi
 
θt
i

2 +
γtλt
i

2 ≤B + √nB = (1 + √n)B.

Moreover, based on the fact that Pn
i=1
Pn
j=1 aij

λt
j −λt
i, z

= 0 for any z ∈Rm, we have that

n
X

i=1

n
X

j=1
aij

λt
j −λt
i, λt
i −λ

=

n
X

i=1

n
X

j=1
aij
D
λt
j −λt
i, λt
i −λ
tE

≤1

2

n
X

i=1

n
X

j=1
aij

λt
j −λt
i
2

2 +
λt
i −λ
t
2

2



≤5

2

n
X

i=1

λt
i −λ
t
2

2 .

Furthermore, notice that

λt
i −λ, gi
 
θt
i

−γtλt
i

=

λt
i −λ, gi
 
θt
i

−γt∥λt
i∥2
2 + γtλ⊤λt
i

≤

λt
i −λ, gi
 
θt
i

+ γt

2


∥λ∥2
2 −
λt
i
2
2


,

where the last inequality follows that λ⊤λt
i = 1

2(∥λ∥2
2 +
λt
i
2
2). We also have

n
X

i=1

n
X

j=1
aij
λt
j −λt
i

2 ≤2

n
X

i=1

λt
i −λ
t
2 .

Plugging all the above results into (A30), we obtain

n
X

i=1

λt+1
i
−λ
2

2 ≤

n
X

i=1

λt
i −λ
2
2 + 9
λt
i −λ
t
2

2

+ 2γt

λt
i −λ, gi
 
θt
i

+ γ2
t

∥λ∥2
2 −
λt
i
2
2



+4γt(1 + √n)B
λt
i −λ
t
2


+ 4nB2γ2
t .

Rearranging the terms in the above inequality and summing over t ∈[T] gives

T
X

t=1

n
X

i=1
γt

λt
i −λ, gi
 
θt
i

+

T
X

t=1

nγ2
t
2 ∥λ∥2
2

≥1

2

T
X

t=1

n
X

i=1

λt+1
i
−λ
2

2 −
λt
i −λ
2
2



−9

2

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2 −4nB2
T
X

t=1
γ2
t

−2(1 + √n)B

T
X

t=1
γt

n
X

i=1

λt
i −λ
t
2 +

T
X

t=1

n
X

i=1

γ2
t
2

λt
i
2
2 .

The last term on the right side of the above inequality is non-negative and can be omitted. Besides,
since λ1
i = 0 for all i ∈[n], then PT
t=1
λt+1
i
−λ
2

2 −
λt
i −λ
2
2


≥−∥λ∥2
2 for any λT +1
i
∈

24


---Page Break---
Rm
+. Thus, we have

T
X

t=1

n
X

i=1
γt

(λt
i)⊤gi(θt
i) −λ⊤gi(θt
i)


≥−n

2

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2 −9

2

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2

−2(1 + √n)B

T
X

t=1
γt

n
X

i=1

λt
i −λ
t
2 −4nB2
T
X

t=1
γ2
t .

Rearranging the terms in the above inequality yields

−

T
X

t=1

n
X

i=1
γt(λt
i)⊤gi(θt
i) ≤−

T
X

t=1

n
X

i=1
γtλ⊤gi(θt
i) + n

2

 

1 +

T
X

t=1
γ2
t

!

∥λ∥2
2 + 9

2

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2

+ 2(1 + √n)B

T
X

t=1
γt

n
X

i=1

λt
i −λ
t
2 + 4nB2
T
X

t=1
γ2
t ,

which completes the proof.

E.3
Proof of Lemma E.4

Based on the update rule of bθ
t
ih that bθ
t+1
ih
= P

k̸=h aikbθ
t
kh + aihθt
h, ∀h ̸= i and i, h ∈[n], we have
that

et+1
ih
:= bθ
t+1
ih
−θt+1
h
=
X

k̸=h
aikbθ
t
kh + aihθt
h −θt+1
h
+ θt
h −θt
h

=
X

k̸=h
aiket
kh −
 
θt+1
h
−θt
h

.

Recall that A−
h is the weight matrix formed by removing the hth row and hth column of the weight

matrix A for any h ∈[n], and et
h := col

et
1h, · · · , et
(h−1)h, et
(h+1)h, · · · , et
nh

. Then,

et+1
h
= (A−
h ⊗Id)et
h + 1n−1 ⊗
 
θt+1
h
−θt
h

.

Since β is the maximium eigenvalue of A−
h for all h ∈[n], we have that

E∥et+1
h
∥2 ≤E
(A−
h ⊗Id)et
h

2 + E
1n−1 ⊗
 
θt+1
h
−θt
h

2

≤βE∥et
h∥2 +
√

n −1γtE
φt
h

ξt
h; θt
h, bθ
t
h, λt
h

2

≤βE∥et
h∥2 +
√

n −1γtE
∇θhJh

ξt
h; θt
h, bθ
t
h

2 +
√

n −1γ2
t E
∇gh(θh)⊤λt
h

2

≤βE∥et
h∥2 +
√

n −1γt(G + GgE∥γtλt
h∥2)

≤βtE∥e1
h∥2 +
√

n −1

t−1
X

k=0
βkγt−k(G + √nBGg).
(A31)

Further, since θ1
ih = 0 for any i, h ∈[n], then, from Assumption 2.3, E∥e1
ih∥2 = ∥θ1
h∥2 ≤C.
Summing the above inequality over t ∈[T] and h ∈[n], we obtain

T
X

t=1

n
X

h=1
E∥et
h∥2 ≤nC

T
X

t=1
βt−1 + n
√

n −1(G + √nBGg)

T
X

t=1

t−2
X

k=0
βkγt−k−1

≤
nC
1 −β + n
√

n −1(G + √nBGg)

T
X

k=1

T
X

t=k+1
βt−k−1γk

≤
nC
1 −β + n√n −1(G + √nBGg)

1 −β

T
X

k=1
γk.

25


---Page Break---
On the other hand, taking square on both sides of (A31), we have

E∥et
h∥2
2 ≤2βtE
e1
h
2
2 + 2(n −1)(G + √nBGg)2
 t−2
X

k=0
βkγt−k−1

!2

.
(A32)

Using the Cauchy-Schwarz inequality yields

 t−2
X

k=0
βkγt−k−1

!2

≤

 t−2
X

k=0
βk
!  t−2
X

k=0
βkγ2
t−k−1

!

≤
Pt−2
k=0 βkγt−k−1

1 −β
.
(A33)

Plugging (A33) into (A32) and summing over t ∈[T], we have that

T
X

t=1

n
X

h=1
E∥et
h∥2
2 ≤2nC2
T
X

t=1
βt + 2n(n −1)(G + √nBGg)2

1 −β

 T
X

t=1

t−2
X

k=0
βkγt−k−1

!

≤2nC2

1 −β + 2n(n −1)(G + √nBGg)2

(1 −β)2

T
X

k=1
γk.

E.4
Proof of Lemma E.5

Let ωt
i :=
h 
1 −γ2
t
 P

j∈Ni aijλt
j + γtgi
 
θt
i
i

+ −P

j∈Ni aijλt
j. Then, for any i ∈[n], we have

that

ωt
i

2 =





 
1 −γ2
t
 X

j∈Ni
aijλt
j + γtgi
 
θt
i





+

−
X

j∈Ni
aijλt
j


2

≤


−γt
X

j∈Ni
aijγtλt
j + γtgi
 
θt
i


2
≤γt
X

j∈Ni
aij
γtλt
j

2 + γt
gi
 
θt
i

2

≤γt(√n + 1)B.
(A34)

The ﬁrst inequality in (A34) results from the nonexpansive property of projection, and the third
inequality holds by using Lemma E.1. By the update rule of λi for any i ∈[n], we have that

λt+1
i
=
X

j∈Ni
aijλt
j + ωt
i.

Deﬁne concatenation vectors λt
o = col
 
λt
1, · · · , λt
n

and ωt
o = col (ωt
1, · · · , ωt
n). Then, for any
t ∈[T], we have

λt+1
o
= (A ⊗Im) λt
o + ωt
o.
(A35)

Since λ
t = 1

n
Pn
i=1 λt
i, we have that

∆t := λt
o −(1n ⊗Im) λ
t =

In −1n1T
n
n


⊗Im


λt
o, ∀t ∈[T].
(A36)

Combining (A35) and (A36) yields

∆t+1 = (A ⊗Im) ∆t +

I −11T

n


⊗Im


ωt
o, ∀t ∈[T].

26


---Page Break---
Firm 5
Firm 2

Firm 3

Firm 1
Firm 4

Figure 3: A networked Cournot game with ﬁve ﬁrms and three markets.

Since λ1
i = 0 for all i ∈[n], then ∆1 = 0. Based on the fact that
I −11T

n

F ≤2, we have that

n
X

i=1

λt+1
i
−λ
t+1
2 =
∆t+1
2 =
(A ⊗Im) ∆t +

I −11T

n


⊗Im


ωt
o


2

≤
(A ⊗Im) ∆t
2 +



I −11T

n


⊗Im


ωt
o


2
≤σ2(A)
∆t
2 + 2
ωt
o

2

≤2

t−1
X

k=0
σ2(A)k ωt−k
o

2

≤2(n + √n)B

t−1
X

k=0
σ2(A)kγt−k,

where the last inequality is based on the result in (A34). Summing the above inequality over t ∈[T]
yields

T
X

t=1

n
X

i=1

λt
i −λ
t
2 ≤2(n + √n)B

T
X

t=1

t−2
X

k=0
σ2(A)kγt−1−k

≤2(n + √n)B

1 −σ2(A)

T
X

k=1
γk.

Similarly to the calculation of PT
t=1
Pn
h=1 ∥et
h∥2
2 in Section E.3, we have that

T
X

t=1

n
X

i=1

λt
i −λ
t
2

2 ≤4(n + √n)2B2

(1 −σ2(A))2

T
X

k=1
γk.

F
Simulation Details

F.1
Networked Cournot Game

The Cournot game is a foundational model in economic theory (Allaz and Vila, 1993) for analyzing
oligopolistic competition, where a limited number of ﬁrms dominate a speciﬁc market. In Cournot
games, all ﬁrms sell a homogeneous commodity and aim to maximize their individual proﬁts by
independently and simultaneously determining optimal production quantities. The total quantity
produced by all ﬁrms is constrained by factors such as market capacity, raw material availability, and
environmental considerations. The proﬁt of each ﬁrm depends not only on its own production quantity
but also on the quantities chosen by its competitors, as they inﬂuence the demand price determined

27


---Page Break---
Figure 4: The demand prices of three markets.

Figure 5: The serving quantities of ﬁve ﬁrms to three markets.

by the market’s demand curve and the total production quantity. There are strategic interactions
between ﬁrms and markets in the Cournot game. According to the law of supply and demand, an
increased production quantity drives down the demand price, and vice versa. The Cournot game model
has diverse applications in various ﬁelds, including supply chain management, electricity market
competition, natural resource extraction, online advertising auctions, and the telecommunications
industry.

In this experiment, we consider a networked Cournot game comprising n ﬁrms selling a single
commodity across m markets, as illustrated in Fig. 3. Each ﬁrm i ∈[n] determines its output quantity
θi = col (θi1, · · · , θim) subject to the constraint of its production capacity Qi that Pm
j=1 θij ≤Qi.
Here, θij denotes the quantity of player i sold to the jth market. The total quantity allocated to market
j is limited by its market capacity Bj, satisfying the condition that Pn
i=1 θij ≤Bj ∀j ∈[m]. Thus,
the local constraint of player i associated with market j is

gij(θi) = θij −Bj/n, ∀i ∈[n], j ∈[m].

Let gi(θi) := col (gi1(θi), · · · , gim(θi)), ∀i ∈[n].

28


---Page Break---
The cost function of ﬁrm i is deﬁned as

Ji = d⊤
i θi −

m
X

j=1
pjθij,

where di = col (di1, · · · , dim) and dij represents the cost that ﬁrm i sells a unit of its product to the
jth market, ∀i ∈[n], j ∈[m]. di includes the cost of raw material, transportation, maintenance, etc.
In Ji, the term pj denotes the unit demand price of market j determined by its market demand curve
and the total production quantity, given by

pj = ξj + Λj

 

cj + 1

dj

n
X

i=1
θij

!−1

τj
, ∀j ∈[m],
(A37)

where cj, dj Λj, and τj > 0 are constants, ξj is a random variable. Due to the interaction between
ﬁrms and markets, the demand price can ﬂuctuate with production quantities, represented by ξj ∼
Dj(θ). Note that the quantity-dependent distributions Dj(θ) for all j ∈[m] are unknown by players.
For any j ∈[m], the variable ξj is deﬁned as

ξj = ξo
j + ε
αj
Pm
j′=1 αj′

 n
X

i=1
θij

!

,

where ξo
j is the random base component, ε ≥0 represents the performative strength of markets, and
αj is the relative strength of market j for any j ∈[m]. According to the law of supply and demand,
an increased production quantity generally decreases a market’s demand price, which corresponds to
the setup that αj ≤0 for all j ∈[m]. Thus, the objective of each play i ∈[n] in the network Cournot
game is formulated by

min
θi∈Ωi
Epj∼Dj(θij,∀i∈[n]),j∈[m]



d⊤
i θi −

m
X

j=1
pjθij





subject to
θij +
X

i′̸=i
θi′j ≤Bj, ∀j ∈[m].

In the simulation, we set n = 5 and m = 3. The network structure is as depicted in Fig. 3. Each
element of the communication weight matrix A = (aij)n×n is set to be aij =
1
|Ni|, and |Ni| is the
cardinality of Ni. The production capacity Qi is randomly and uniformly drawn from [10, 12] for all
i ∈[5], and the market’s capacity Bj is randomly and uniformly drawn from [10, 15] for all j ∈[m].
All entries in di, ∀i ∈[n] are randomly and uniformly drawn from [1, 1.5]. The distribution of ξo
j is
set to min(max(N(2.5, 1), 2.5), 7.5). The performative power αj is randomly and uniformly drawn
from (−1, 0], for all j ∈[3]. Other settings are: Λj = 10, cj = 10, dj = 5 and τj = 2, ∀j ∈[3].

Fig. 4 compares the demand prices of three markets at PSE and NE with performative strength
ε = 0.2, 0.4, and 0.6 and Fig. 5 compares the corresponding serving quantities of ﬁve ﬁrms to these
three markets. The results suggest that, although a larger performative strength leads to a wider gap,
the difference in these two indicators between the PSE and NE remains insigniﬁcant. This conﬁrms
the effectiveness of PSE solutions and our distance analysis between the PSE and NE as stated in
Theorem 3.5.

F.2
Ride-Share Market

We further examine an example of a ride-share market, where multiple platforms compete to maximize
their individual revenue by offering shared rides in competitive areas, taking into account opera-
tional constraints and market demands. This experiment builds upon the semi-synthetic simulation
conducted in (Narang et al., 2023), adapting it to our constrained noncooperative game setting.

Consider a ride-share market with n platforms competing in m areas. Each platform i ∈[n] aims to
maximize its revenue by determining the quantities it offers at the jth area, denoted as θij, for all
j ∈[m]. Let θi = [θi1, · · · , θim]⊤. The total number of rides provided by each platform i cannot
exceed a predeﬁned limit Qi, given by Pm
j=1 θij ≤Qi, ∀i ∈[n]. Let pj denote the demand price

29


---Page Break---
Figure 6: Convergence of the time-average revenues of three platforms.

Figure 7: Convergence of the time-average constraint violations at eight areas.

at the jth location, which ﬂuctuates with the total offered quantity at the area following the law of
supply and demand. We adopt the same model for {pj} as in the network Cournot game, given by
(A37). Additionally, the maintenance costs associated with platform operations may vary across
locations due to factors such as distance or labor costs. Let di ∈Rm represent the cost vector of
platform i at all areas. Then, the inverse of the revenue function for each platform can be expressed as

Ji = −

m
X

j=1
pjθij + d⊤
i θi, ∀i ∈[n].

Assume that each platform only offers one type of ride. Considering the diverse ride characteristics,
such as shape and speed, we use hi to denote the spatial occupancy of each ride offered by platform i.
The accommodated ride quantity at each location is constrained by Bj due to parking availability
and road conditions, such that Pn
i=1 hiθij ≤Bj. Then, the objective of each platform i ∈[n] in the
ride-share market is formulated as

min
θi∈Ωi
Epj∼Dj(θij,∀i∈[n]),∀j∈[m]



−

m
X

j=1
pjθij + d⊤
i θi





subject to
hiθij +
X

i′̸=i
hi′θi′j ≤Bj, ∀j ∈[m].

(A38)

The simulation setup is based on dataset from a prior Kaggle competition.2 Our study focuses on
three ride-share platforms (Uber, Lyft, and Via) and eight competing areas within New York. We
randomly and uniformly assign the total number of rides, Qi, from the range [200, 400] for each
platform i ∈[3]. Similarly, the accommodated capacity, Bj, is randomly and uniformly drawn from
[50, 150] for all j ∈[8]. All entries in di, ∀i ∈[n] are randomly and uniformly drawn from [0.2, 2.2].
The distribution of ξo
j is set as min(max(N(1, 1), 1), 5). Additionally, we set the following values
for all areas j ∈[8]: Λj = 5, cj = 5, dj = 5, and τj = 2.

Fig. 6 compares the convergence of the time-average revenues of these three platforms: Uber, Lyft,
and Via, denoted by −1

t
Pt
t′=1 Ept∼D(θt)[Ji(pt; θt′)]. We consider three performative strengths:

2The data is publicly available at https://www.kaggle.com/brllrb/uber-and-lyft-dataset-boston-ma

30


---Page Break---
Figure 8: The normalized distance between θt and θne.

ε = 0.1, 0.2, and 0.3. Similarly to Fig. 2 (b), we compare the performance of Algorithm 1, represented
by “pse”, and the performance of Algorithm 1with perfect knowledge of data distributions Dj(θ)
for all j ∈[m]. It is observed that, with a mild performative strength ε, the revenues achieved by
the “pse” are close to those of the “ne” for all three platforms. However, as ε increases, the gap
between the two approaches widens, although it remains relatively small. This observation conﬁrms
the analytical result presented in Theorem 3.5.

Fig. 7 shows the convergence of the time-average constraint violations at eight areas by Algorithm 1,
denoted by 1

t
Pt
t′=1
P3
i=1 gij(θt′
i ), j = 1, · · · , 8, with performative strengths of ε = 0.1, 0.2, and
0.3. The constraints hold for all three performative strengths. However, as ε increases, the platform
tends to allocate fewer rides. This may be attributed to larger market ﬂuctuations associated with a
higher ε, leading to a more conservative allocation.

Fig. 8 compares the normalized distance between θt and the NE point θne, denoted as ∥θt −
θne∥2/∥θt∥2, with performative strengths: ε = 0.1, 0.2, and 0.3. The result is quantitatively
analogous to the ﬁndings presented in Fig. 8. Firstly, θt gradually approaches θne with iterations.
Secondly, a higher performative strength leads to a wider normalized distance between the convergent
point of θt and θne.

Fig. 9 compares the demand prices of eight areas and the ride quantities offered to them by three
platforms at PSE and NE. We consider performative strengths ε = 0.1 and ε = 0.3. It is observed
that the values of these indicators at the PSE and NE are close to each other when ε = 0.1. However,
a noticeable discrepancy arises when ε = 0.3.

Additionally, we display the demand prices of eight areas in New York in Fig. 10, with different
performative strengths: ε = 0.1, 0.2, and 0.3. It can be observed that, while prices vary by location,
smaller values of ε generally correspond to higher prices. The offered quantities of these three
platforms to the eight locations are illustrated in Fig. 11. The results indicate a conservative allocation
as the performative strength increases. Furthermore, with the cost of these three platforms at different
locations in Fig. 12, we obtain the revenues of the platforms Uber, Lyft, and Via in different areas, as
illustrated in Fig. 13. Clearly, performativity has an inverse effect on revenues, and the stronger the
performative strength, the lower the revenues.

31


---Page Break---
Figure 9: The demand prices of eight areas and the ride quantities offered to them by three platforms.

Figure 10: The demand prices of different areas.

32


---Page Break---
Figure 11: The quantities of platforms offered to different areas.

Figure 12: The cost of platforms in different areas.

33


---Page Break---
Figure 13: The revenues of platforms in different areas.

34


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?
Answer: [Yes]
2. Limitations:

Does the paper discuss the limitations of the work performed by the authors? [No]
Answer: [No]
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [No]
6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
7. Experiment Statistical Signiﬁcance

Question: Does the paper report error bars suitably and correctly deﬁned or other appropriate
information about the statistical signiﬁcance of the experiments?
Answer: [Yes]
8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufﬁcient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [No]
11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

35


---Page Break---
Answer: [NA]
12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]

36


---Page Break---
