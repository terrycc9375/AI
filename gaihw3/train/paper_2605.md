Warm-up Free Policy Optimization:
Improved Regret in Linear Markov Decision
Processes

Asaf Cassel
Tel Aviv University
acassel@mail.tau.ac.il

Aviv Rosenberg
Google Research
avivros@google.com

Abstract

Policy Optimization (PO) methods are among the most popular Reinforcement
Learning (RL) algorithms in practice. Recently, Sherman et al. [2023a] proposed a
PO-based algorithm with rate-optimal regret guarantees under the linear Markov
Decision Process (MDP) model. However, their algorithm relies on a costly pure
exploration warm-up phase that is hard to implement in practice. This paper
eliminates this undesired warm-up phase, replacing it with a simple and efﬁcient
contraction mechanism. Our PO algorithm achieves rate-optimal regret with im-
proved dependence on the other parameters of the problem (horizon and function
approximation dimension) in two fundamental settings: adversarial losses with
full-information feedback and stochastic losses with bandit feedback.

1
Introduction

Policy Optimization (PO) is a widely used method in Reinforcement Learning (RL) that achieved
tremendous empirical success, with applications ranging from robotics and computer games [Schul-
man et al., 2015, 2017, Mnih et al., 2015, Haarnoja et al., 2018] to Large Language Models (LLMs;
Stiennon et al. [2020], Ouyang et al. [2022]). Theoretical work on policy optimization algorithms
initially considered tabular Markov Decision Processes (MDPs; Even-Dar et al. [2009], Neu et al.
[2010b], Shani et al. [2020], Luo et al. [2021]), where the number of states is assumed to be ﬁnite and
small. In recent years the theory was generalized to inﬁnite state spaces under function approximation,
speciﬁcally under linear function approximation in the linear MDP model [Luo et al., 2021, Dai et al.,
2023, Sherman et al., 2023b,a, Liu et al., 2023].

Recently, Sherman et al. [2023a] presented the ﬁrst policy optimization algorithm that achieves rate-
optimal regret in linear MDPs, i.e., a regret bound of eO(poly(H, d)
√

K), where K is the number of
interaction episodes, H is the horizon, and d is the dimension of the linear function approximation.
However, their algorithm requires a pure exploration warm-up phase to obtain an initial estimate
of the transition dynamics. To that end, they utilize the algorithm of Wagenmaker et al. [2022b]
for reward-free exploration which is not based on the policy optimization paradigm. Moreover,
although this algorithm is computationally efﬁcient, it relies on intricate estimation techniques that
are hard to implement in practice and unlikely to generalize beyond linear function approximation
(see discussion in section 4).

In this paper, we propose a novel contraction mechanism to avoid this costly warm-up phase. Both our
contraction mechanism and the warm-up phase serve a similar purpose – ensuring that the Q-value
estimates are bounded and yield “simple” policies. But, unlike the warm-up, our method is integrated
directly into the PO algorithm, implemented using a simple conditional truncation of the Q-estimates,
and only contributes a lower-order term to the ﬁnal regret bound. Moreover, our approach is much

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
more efﬁcient in practice since it does not rely on any reward-free methods, which explore the state
space uniformly without taking the reward into account.

Based on this contraction mechanism, we build a new policy optimization algorithm that is simpler,
more computationally efﬁcient, easier to implement, and most importantly, improves upon the
best-known regret bounds for policy optimization in linear MDPs. Our regret bound holds in two
fundamental settings:

1. Adversarial losses with full-information feedback, where the loss function changes arbitrarily
between episodes and is revealed to the agent entirely at the end of each episode.
2. Stochastic losses with bandit feedback, where the loss function in each episode is sampled
i.i.d from some unknown ﬁxed distribution and the agent only observes instantaneous losses
in the state-action pairs that she visits.

In these settings, the best-known regret bound (by Sherman et al. [2023b]) was eO(
√

H7d4K). Our
algorithm, Contracted Features Policy Optimization (CFPO), achieves eO(
√

H4d3K) regret, yielding
a
√

H3d improvement over any algorithm for the adversarial setting and matching the value iteration
based approach of Jin et al. [2020b] in the stochastic setting. We conjecture that this is the best
regret we can hope for without more sophisticated variance reduction techniques [Azar et al., 2017,
Zanette and Brunskill, 2019, He et al., 2023, Zhang et al., 2024], that have not yet been applied to PO
algorithms even in the tabular setting.1 Ignoring logarithmic factors, the regret of CFPO leaves a gap
of only
√

Hd from the Ω(
√

H3d2K) lower bound for linear MDPs [Zhou et al., 2021a]. Finally, our
analysis relies on a novel regret decomposition that uses a notion of contracted (sub) MDP and may
be of separate interest (see section 5).

1.1
Related work

Policy optimization in tabular MDPs.
The regret analysis of PO methods in tabular MDPs was
introduced by Even-Dar et al. [2009], which considered the case of known transitions and adversarial
losses under full-information feedback. Neu et al. [2010a,b] extended their algorithms to adversarial
losses under bandit feedback. Then, Shani et al. [2020] presented the ﬁrst PO algorithms for the
case of unknown transitions (for both stochastic and adversarial losses), and ﬁnally Luo et al. [2021]
devised a PO algorithm with rate-optimal regret for the challenging case of unknown transitions with
adversarial losses under bandit feedback. Since then, PO was studied in more challenging cases, e.g.,
delayed feedback [Lancewicki et al., 2022, 2023] and best-of-both-worlds [Dann et al., 2023].

Other regret minimization methods in tabular MDPs.
An alternative popular method for regret
minimization in tabular MDPs with adversarial losses is O-REPS [Zimin and Neu, 2013, Rosenberg
and Mansour, 2019a,b, Jin et al., 2020a], which optimizes over the global state-action occupancy
measures instead of locally over the policies in each state. However, this method is hard to implement
in practice and does not generalize to the function approximation setting (without restrictive assump-
tions). For stochastic losses, optimistic methods based on Value Iteration (VI; Jaksch et al. [2010],
Azar et al. [2017], Zanette and Brunskill [2019]) and Q-learning [Jin et al., 2018, Zhang et al., 2020]
are known to guarantee optimal regret, which has not been established yet for adversarial losses.

Policy optimization in linear MDPs.
While Sherman et al. [2023a] established rate-optimal regret
for PO methods in linear MDPs with stochastic losses, most of the recent research focused on the
case of adversarial losses with bandit feedback [Luo et al., 2021, Neu and Olkhovskaya, 2021, Dai
et al., 2023, Sherman et al., 2023b, Kong et al., 2023, Liu et al., 2023, Zhong and Zhang, 2023],
where rate-optimality has not been achieved yet.

Other regret minimization methods in linear MDPs and other models for function approxima-
tion.
Unlike O-REPS methods that do not generalize to linear function approximation, value-based
methods (operating under the stochastic loss assumption) are also popular in linear MDPs and have
been shown to yield optimal regret [Jin et al., 2020a, Zanette et al., 2020, Wagenmaker et al., 2022a,

1Wu et al. [2022] apply variance reduction techniques to get better regret bounds in the tabular setting, but
they use L2-regularization instead of KL-regularization which does not align with practical PO algorithms
Schulman et al. [2015, 2017].

2


---Page Break---
Hu et al., 2022, He et al., 2023, Agarwal et al., 2023]. Another line of works [Ayoub et al., 2020,
Modi et al., 2020, Cai et al., 2020, Zhang et al., 2021, Zhou et al., 2021a,b, He et al., 2022, Zhou
and Gu, 2022] study linear mixture MDP which is a different model that is incomparable with linear
MDP [Zhou et al., 2021b]. Finally, there is a rich line of works studying statistical properties of RL
with more general function approximation [Munos, 2005, Jiang et al., 2017, Dong et al., 2020, Jin
et al., 2021, Du et al., 2021], but these usually do not admit computationally efﬁcient algorithms.

2
Problem setup

Episodic Markov Decision Process (MDP).
A ﬁnite-horizon episodic MDP M is de-
ﬁned by a tuple (X, A, x1, {ℓk}K
k=1, P, H) with X, a set of states, A, a set of actions,
H, decision horizon, x1
∈
X, an initial state (assumed to be ﬁxed for simplicity),
P = (Ph)h∈[H], Ph : X × A →∆(X), the transition probabilities, and {ℓk}K
k=1, sequence of loss
functions such that ℓk = (ℓk
h)h∈[H], ℓk
h : X × A →[0, 1], is a horizon dependent immediate loss
function for taking action a at state x and horizon h of episode k. A single episode k of an MDP is a
sequence (xk
h, ak
h, ℓk
h(xk
h, ak
h))h∈[H] ∈(X × A × [0, 1])H such that

Pr[xk
h+1 = x′ | xk
h = x, ak
h = a] = Ph(x′ | x, a).

For the losses, we consider two settings: stochastic and adversarial. In the stochastic setting, there
exists a ﬁxed loss function ℓ= (ℓh)h∈[H], ℓh : X × A →[0, 1] such that ℓk is sampled i.i.d from
a distribution whose expected value is deﬁned by ℓ, i.e., E

ℓk
h(x, a) | x, a

= ℓh(x, a). In the
adversarial setting, the loss function sequence {ℓk}K
k=1 is chosen by an adaptive adversary.

Linear MDP.
A linear MDP Jin et al. [2020b] satisﬁes all the properties of the above MDP but has
the following additional structural assumptions. There is a known feature mapping φ : X × A →Rd
such that Ph(x′ | x, a) = φ(x, a)Tψh(x′) where ψh : X →Rd are unknown parameters. Moreover,
for all h ∈[H], k ∈[K], there is an unknown vector θk
h ∈Rd such that, in the adversarial case,
ℓk
h(x, a) = φ(x, a)Tθk
h, while in the stochastic case, θk
h = θh and ℓh(x, a) = φ(x, a)Tθh. We make
the following normalization assumptions, common throughout the literature:

1. ∥φ(x, a)∥≤1 for all x ∈X, a ∈A;

2. ∥θk
h∥≤
√

d for all h ∈[H], k ∈[K];

3. ∥|ψh|(X)∥= ∥P

x∈X |ψh(x)|∥≤
√

d for all h ∈[H];

where |ψh(x)| is the entry-wise absolute value of ψh(x) ∈Rd. We follow the standard assumption
in the literature that the action space A is ﬁnite. In addition, for ease of mathematical exposition (e.g.
Cassel et al. [2024]), we also assume that the state space X is ﬁnite. This allows for simple matrix
notation and avoids technical measure theoretic deﬁnitions. Importantly, our results are completely
independent of the state space size |X|, both computationally and in terms of regret. Thus, there is no
particular loss of generality.

Policy and value.
A stochastic Markov policy π = (πh)h∈[H] : [H] × X 7→∆(A) is a mapping
from a step and a state to a distribution over actions. Such a policy induces a distribution over
trajectories ι = (xh, ah)h∈[H], i.e., sequences of H state-action pairs. For f : (X × A)H →R,
which maps trajectories to real values, we denote the expectation with respect to ι under dynamics P
and policy π as EP,π[f(ι)]. Similarly, we denote the probability under this distribution by PP,π[·].
We denote the class of stochastic Markov policies as ΠM. For any policy π ∈ΠM, horizon h ∈[H]
and episode k ∈[K] we deﬁne its loss-to-go, as

V k,π
h
(x) = EP,π

" H
X

h′=h
E[ℓk
h′(xh′, ah′) | xh′, ah′]
 xh = x

#

,

which is the expected loss if one starts from state x ∈X at horizon h of episode k and follows policy
π onwards. Note that the inner expectation is only relevant for stochastic losses as its argument is
deterministic in the adversarial setup. The performance of a policy in episode k, also known as its
value, is measured by its expected cumulative loss V k,π
1
(x1).

3


---Page Break---
Interaction protocol and regret.
We consider a standard episodic regret minimization setting
where an algorithm performs K interactions with an MDP M. For stochastic losses we consider
bandit feedback, where the agent observes only the instantaneous losses along its trajectory, while
for adversarial losses we consider full-information feedback, where the agent observes the full loss
function ℓk in the end of episode k ∈[K]. Concretely, at the start of each interaction/episode
k ∈[K], the agent speciﬁes a stochastic Markov policy πk = (πk
h)h∈[H]. Subsequently, it observes
the trajectory ιk sampled from the distribution PP,πk, and, either the individual episode losses
ℓk
h(xk
h, ak
h), h ∈[H] in the case of bandit feedback, or the entire loss function ℓk in the case of
full-information feedback.

We measure the quality of any algorithm via its regret – the difference between the value of the
policies πk generated by the algorithm and that of the best policy in hindsight, i.e.,

Regret =

K
X

k=1
V k,πk

1
(x1) −min
π∈ΠM

K
X

k=1
V k,π
1
(x1) =

K
X

k=1
V k,πk

1
(x1) −V k,π⋆

1
(x1),

where the best policy in hindsight is denoted by π⋆(known to be optimal even among the class of
stochastic history-dependent policies).

Notation.
Throughout the paper φk
h = φ(xk
h, ak
h) ∈Rd denote the state-action features at horizon
h of episode k. In addition, ∥v∥A =
√

vTAv. Hyper-parameters follow the notations βz and ηz for
some z, and δ ∈(0, 1) denotes a conﬁdence parameter. Finally, in the context of an algorithm, ←
signs refer to compute operations whereas = signs deﬁne operators, which are evaluated at speciﬁc
points as part of compute operations.

3
The role of value clipping

Before presenting our contraction technique and main results, we discuss the role that value clipping
plays in regret minimization and its apparent necessity for linear MDPs. As a starting point, it is
important to note that, while commonly used [Azar et al., 2017, Luo et al., 2021], value clipping is not
strictly necessary in tabular MDPs. To demonstrate this, consider a fairly standard optimistic Value
Iteration (VI) algorithm that constructs sample-based estimates ˆℓ, ˆP with empirical error estimates
∆ℓ, ∆P , deﬁnes exploration bonuses b = (∆ℓ+ H · ∆P ), and chooses a policy ˆπ⋆that is optimal in
the empirical MDP whose dynamics are ˆP and losses are ˆℓ−b. Then its single-episode regret may
be decomposed as

V ˆπ⋆
1 (x1) −V π⋆

1 (x1) = V ˆπ⋆
1 (x1) −ˆV ˆπ⋆
1 (x1)
|
{z
}
(i)−bias / cost of optimism

+ ˆV ˆπ⋆
1 (x1) −ˆV π⋆

1 (x1)
|
{z
}
(ii)−FTL / ERM

+ ˆV π⋆

1 (x1) −V π⋆

1 (x1)
|
{z
}
(iii)−optimism

,

where ˆV is the value under the empirical MDP. Now, by deﬁnition of ˆπ⋆, we have that (ii) ≤0. Now,
let ∆ℓ= ˆℓ−ℓ, ∆P = ˆP −P. Using a standard value difference lemma (lemma 14 in appendix B)
we have that (i) ≲b and

(iii) = E ˆ
P ,π⋆



X

h∈[H]
∆ℓ(xh, ah) −b(xh, ah) +
X

x′∈X
∆P(x′ | xh, ah)V π⋆

h+1(x′)




(1)

≤E ˆ
P ,π⋆



X

h∈[H]
∆ℓ(xh, ah) + H∆P (xh, ah) −b(xh, ah)



= 0,

where the inequality also used that V π⋆

h
∈[0, H]. The ﬁnal regret bound is concluded by summing
over k ∈[K] and using a bound on harmonic sums. We note that a similar clipping-free method also
works for tabular PO (see Cassel et al. [2024]).

Moving on to Linear MDPs, one might expect a similar approach to work. Unfortunately, the standard
approach that estimates the dynamics backup operators ψh, h ∈[H] using regularized least-squares
presents a signiﬁcant challenge. This is because, unlike the tabular setting, the resulting estimate

4


---Page Break---
ˆPh(· | x, a) = φ(x, a)T bψh(·) (eq. (2)) is not guaranteed to yield a valid probability distribution, i.e.,
there could exist x ∈X, a ∈A, h ∈[H] such that

∥ˆPh(· | x, a)∥1 = c > 1
and/or
min
x′∈X
ˆPh(x′ | x, a) < 0.

ˆP is still a ﬁnite signed-measure, which is enough for the ﬁrst equality in eq. (1) to hold. However,
since E ˆ
P ,π⋆could contain negative probability terms, the inequality in eq. (1) does not hold. These
negative probabilities also seem to make calculating ˆπ⋆computationally hard. Finally, the ℓ1−norm
exceeding 1 may cause term (i) to depend on H exponentially. While some of these issues could be
mitigated without clipping, we are not aware of a method that resolves all simultaneously.

The use of value clipping opens the path for an alternative value decomposition that replaces E ˆ
P ,π⋆
in eq. (1) with EP,π⋆at the cost of also replacing V π⋆

h+1 with ˆV π⋆

h+1. We thus need that | ˆV π⋆

h+1| ≲H for
the inequality in eq. (1) to work. This is made possible using a clipping mechanism that decouples
the scale of ˆV π⋆

h+1 from the magnitude of the bonuses b, which may be much larger when the error
estimates ∆ℓ, ∆P are large. This is typically achieved by adding max{0, ·} to the recursive formula
for the value function. A similar clipping approach also works for tabular PO and VI [Azar et al.,
2017, Luo et al., 2021], and even for VI in linear MDPs [Jin et al., 2020b].

However, this is not the case for PO in linear MDPs where Sherman et al. [2023a] explain that this
type of value clipping leads to prohibitive complexity of the policy and value function classes, and
thus sub-optimal regret. Concretely, the complexity of the soft-max policy class roughly corresponds
to the number of parameters required to represent P

k∈[K] ˆQk
h. If ˆQk
h(x, a) = φ(x, a)Twk
h are
linear, then the sum remains linear and depends on d parameters (with slightly larger magnitude). If
ˆQk
h(x, a) = max{0, φ(x, a)Twk
h}, the sum may, in general, have dK parameters thus degrading the
regret. Sherman et al. [2023a] overcome this issue using a warm-up based truncation technique. In
what follows, we suggest an alternative solution that uses a novel notion of contracted features and
has several advantages over their approach (see discussion at the end of section 4).

4
Algorithm and main result

We present Contracted Features Policy Optimization (CFPO; algorithm 1), a policy optimization
routine for regret minimization in linear MDPs. The algorithm operates in epochs, each beginning
when the uncertainty of the dynamics estimation shrinks by a multiplicative factor, as expressed by
the determinant of the covariance matrices Λk
h, h ∈[H] (see line 13 for the deﬁnition of Λk
h and
line 4 for the epoch change condition). At the start of each epoch e, we reset the policy to its initial
(uniform) state, and deﬁne the contracted features ¯φke
h , h ∈[H] (line 6) by multiplying the original
features with coefﬁcients in the range [0, 1], and thus shrinking their distance to the origin. Inspired
by ideas from Zanette et al. [2020], these coefﬁcients are chosen inversely proportional to the current
uncertainty of the least squares estimators in each state-action pair, essentially degenerating the MDP
in areas of high uncertainty. Inside an epoch, at episode k, we compute the estimated reward vector
bθk (line 14) and estimated dynamics backup operators bψk
h (eq. (2)). Then, we use these bθk and bψk
h
to compute our Q-value estimates with the contracted features (eq. (3)), and run an online mirror
descent (OMD) update over them (eq. (5)), i.e., run a policy optimization step with respect to the
contracted empirical MDP (more on this in section 5.1).

We note that the computational complexity of algorithm 1 is comparable to other algorithms for regret
minimization in linear MDPs, such as LSVI-UCB [Jin et al., 2020b]. The following is our main result
for algorithm 1 (see the full analysis appendix A).

Theorem 1. Suppose that we run CFPO (algorithm 1) with the parameters deﬁned in theorem 9 (in
appendix A). Then, with probability at least 1 −δ, we have

Regret = O
p

H4d3K log(K) log(KH/δ) +
p

H5dK log(K) log|A|

.

Discussion.
Policy optimization algorithms typically entail running OMD over estimates ˆQ of the
state-action value function Q, as in eq. (5). The crux of the algorithm is in obtaining such estimates
that satisfy an optimistic condition similar to eq. (1), while also keeping the complexity of the policy
class bounded. As discussed in Sherman et al. [2023a], the latter depends on P

k′∈[k] ˆQk′
h (eq. (3))

5


---Page Break---
Algorithm 1 Contracted Features PO for linear MDPs

1: input: d, H, K, A, δ, βw, βb, ηo > 0.
2: initialize: e ←−1, Λ1
h ←I, h ∈[H].
3: for episode k = 1, 2, . . . , K do
4:
if k = 1 or ∃h ∈[H], det(Λk
h) ≥2 det(Λke
h ) then
5:
e ←e + 1 and ke ←k.

6:
¯φke
h (x, a) = φ(x, a) · σ

−βw∥φ(x, a)∥(Λke
h )−1 + log K

.
{σ(z) = 1/(1 + exp(−z))}

7:
πk
h(a | x) = 1/|A| for all h ∈[H], a ∈A, x ∈X.
8:
end if
9:
Play πk and observe losses (ℓk
h(xk
h, ak
h))h∈[H] and trajectory ιk = (xk
h, ak
h)h∈[H].
10:
In the case of full-information feedback: observe θk
h.
11:
Deﬁne ˆV k
H+1(x) = 0 for all x ∈X.
12:
for h = H, . . . , 1 do
13:
Λk+1
h
←I + P

τ∈[k] φτ
h(φτ
h)T.

14:
bθk
h ←

(
(Λk
h)−1 P

τ∈[k−1] φτ
hℓτ
h(xτ
h, aτ
h),
feedback = bandit
θk
h,
feedback = full.
15:
For any V : X →R, x ∈X, a ∈A deﬁne:

bψk
hV
= (Λk
h)−1
X

τ∈[k−1]
φτ
hV (xτ
h+1),
(2)

ˆQk
h(x, a)
= ¯φke
h (x, a)T[bθk
h + bψk
h ˆV k
h+1] −βb∥¯φke
h (x, a)∥(Λke
h )−1,
(3)

ˆV k
h (x)
=
X

a∈A
πk
h(a | x) ˆQk
h(x, a),
(4)

πk+1
h
(a | x) ∝πk
h(a | x) exp(−ηo ˆQk
h(x, a)).
(5)

16:
end for
17: end for

having a low dimensional representation nearly independent of k. Although standard unclipped
estimates admit such a representation, they lack other essential properties (see discussion in section 3).
On the other hand, the standard clipping method, which restricts the value to [0, H] between each
backup operation (see, e.g., Jin et al. [2020b]), does not admit the desired representation.

Sherman et al. [2023a] overcame this issue by employing a warm-up phase based on a reward-
free pure exploration algorithm by Wagenmaker et al. [2022b] to obtain initial backup operators
bψ0
h, h ∈[H] and subsets ¯
Xh ⊆X, h ∈[H] such that: (i) for every x, a ∈¯
Xh × A the bonuses (b in
section 3), which are proportional to the estimation uncertainty of the value backup estimates, are
small (≤1); and (ii) for all policies π ∈ΠM, the probability of reaching any x, a /∈∪h∈[H] ¯
Xh × A
is small (≲K−1/2). To ensure that the overall value estimates remain bounded, they truncate (zero
out) the Q-value estimate of these nearly unreachable state-action pairs, an operation that allows for
a low-dimensional representation of the policies. Nonetheless, their warm-up approach has several
drawbacks.

• It runs for K0 = poly(d, H)
√

K episodes, contributing the leading term in their regret
guarantee;

• It relies on a ﬁrst-order regret algorithm by Wagenmaker et al. [2022a] that is not PO-
based and uses a computationally hard variance-aware Catoni estimator for robust mean
estimation of the value backups, instead of the standard least-squares estimator. To maintain
computational efﬁciency, they use an approximate version of the estimator, losing a factor
of
√

d in the regret;

• Still, to the best of our knowledge, even the approximate estimator must be computed using
binary search methods, making it hard to apply in practical methods that typically rely on
gradient-based continuous optimization techniques;

6


---Page Break---
• It runs separate algorithms for each horizon h ∈[H], using only 1 out of H samples during
the warm-up phase;
• It is not reward-aware, and thus has to explore the space uniformly to ensure that the
uncertainty is small for all policies, which could be highly prohibitive in practice.

Our feature contraction approach obtains the desired bounded Q-value estimates and low-complexity
policy class without relying on a dedicated warm-up phase. Crucially, it only contributes a lower
order term of poly(d, H) log K to the regret guarantee, thus improving the overall dependence on d
and H. Additionally, it uses all samples, is easy to implement, and is reward-aware. To understand
the beneﬁt of reward-awareness, consider an MDP where at the initial state the agent has two actions,
each leading to a distinct MDP. Now, suppose that both MDPs have only a single state and action
for the ﬁrst H/2 steps with one MDP incurring a loss of 1 in these steps while the other incurring 0
loss. Notice that regardless of the last H/2 steps, the 0 loss MDP will outperform the 1 loss MDP.
Nonetheless, the reward-free warm-up, which does not observe the losses, will have to fully explore
both MDPs. In contrast, our reward-aware approach would quickly stop exploring the inferior MDP,
leading to better performance in practice.

5
Analysis

In this section, we prove the main claims of our result. For full details see appendix A. We begin by
introducing the main technical tool for our contraction mechanism – the contracted MDP.

5.1
Contracted (sub) MDP

For any MDP M = (X, A, x1, {ℓk}K
k=1, P, H) and contraction coefﬁcients ρ : [H] × X × A →
[0, 1] we deﬁne a contracted (sub) MDP ¯
M(ρ) = (X, A, x1, {¯ℓk}K
k=1, ¯P, H) where as ¯ℓk
h(x, a) =
ρh(x, a)ℓk
h(x, a) ∈[0, 1] are the contracted losses and ¯Ph(x′ | x, a) = ρh(x, a)Ph(x′ | x, a) ∈[0, 1]
are the contracted (sub) probability transitions. Notice that the transitions being a sub-probability
measure implies that P

x′∈X Ph(x′ | x, a) ≤1 as compared with a probability measure where
this holds with equality. For any Markov policy π ∈ΠM, let ¯V k,π
h
(·; ρ) : X →R, h ∈[H] be the
loss-to-go (or value) functions of the contracted MDP. In particular, these may be deﬁned by the usual
backward recursion

¯V k,π
h
(x; ρ) = Ea∼π(·|x)

"

E[¯ℓk
h(x, a) | x, a] +
X

x′∈X
¯Ph(x′ | x, a) ¯V k,π
h+1(x′; ρ)

#

,

with ¯V k,π
H+1(x; ρ) = 0 for all x ∈X. The following result shows that the value of any contracted
MDP lower bounds its non-contracted variant.

Lemma 2. For any ρ : [H] × X × A →[0, 1], π ∈ΠM, h ∈[H], k ∈[K], and x ∈X we have that
¯V k,π
h
(x; ρ) ≤V k,π
h
(x).

Proof. The proof follows by backward induction on h ∈[H + 1]. For the base case h = H + 1, both
values are 0 and the claim holds trivially. Now suppose the claim holds for h + 1, then we have that
for all x ∈X

¯V k,π
h
(x; ρ) = Ea∼π(·|x)

"

E[¯ℓk
h(x, a) | x, a] +
X

x′∈X
¯Ph(x′ | x, a) ¯V k,π
h+1(x′; ρ)

#

≤Ea∼π(·|x)

"

E[ℓk
h(x, a) | x, a] +
X

x′∈X
Ph(x′ | x, a)V k,π
h+1(x′)

#

= V k,π
h
(x).
■

Next, for any epoch e ∈[E], consider its contracted linear MDP (line 6 in algorithm 1) whose
contraction coefﬁcients are ρke
h (x, a) = σ

−βw∥φ(x, a)∥(Λke
h )−1 + log K

. The following result
gives an upper bound on the performance gap between the contracted and non-contracted variants.

Lemma 3. For any e ∈[E] and v ∈Rd we have that

(φ(xh, ah) −¯φke
h (xh, ah))Tv ≤(4β2
w∥φ(xh, ah)∥2
(Λk
h)−1 + 2K−1)
φ(xh, ah)Tv
.

7


---Page Break---
Proof. We have that

(φ(xh, ah) −¯φke
h (xh, ah))Tv = σ(βw∥φ(xh, ah)∥(Λke
h )−1 −log K) · φ(xh, ah)Tv

≤2(β2
w∥φ(xh, ah)∥2
(Λke
h )−1 + K−1)
φ(xh, ah)Tv


≤(4β2
w∥φ(xh, ah)∥2
(Λk
h)−1 + 2K−1)
φ(xh, ah)Tv
,

where the ﬁrst relation is by the property of the sigmoid 1−σ(x) = σ(−x), the second is by a simple
algebric argument that a quadratic function bounds the sigmoid (lemma 19 in appendix B), and the
last relation uses det(Λk
h) ≤2 det(Λke
h ) by line 4 in algorithm 1 (see lemma 16 in appendix B).
■

We note that the analogous claim in Sherman et al. [2023a] shows that for all π ∈ΠM

EP,π[(φ(xh, ah) −1{xh∈Zh}φ(xh, ah))Tv] ≤Pr(xh /∈Zh) max
x,a
φ(x, a)Tv
,
(6)

where Zh is an outcome of the reward-free warmup phase and Pr(xh /∈Zh) ≈K−1/2. Summing
this over k ∈[K] yields a term that scales as
√

K. In contrast, we use a standard bound on elliptical
potentials (lemma 15 in appendix B) to get that
X

k∈[K]
(4β2
w∥φ(xk
h, ak
h)∥2
(Λk
h)−1 + 2K−1) ≲log K.

This implies that the cost of our contraction is signiﬁcantly lower than the truncation of Sherman
et al. [2023a]. We achieve this reduced cost by using a quadratic (rather than linear) bound on the
logistic function. The challenge in our approach is that the above bound only holds for the observed
trajectories rather than for all policies as in Sherman et al. [2023a]. In what follows, we overcome
this challenge using a novel regret decomposition.

5.2
Regret bound

For any epoch e ∈[E], let Ke be the set of episodes that it contains, and let ¯V k,π
1
(x1; ρke) denote the
value of its contracted MDP as deﬁned above and in line 6 of algorithm 1. We bound the regret as

Regret =
X

k∈[K]
V k,πk
1
(x1) −V k,π⋆

1
(x1)

≤
X

e∈[E]

X

k∈Ke
V k,πk
1
(x1) −¯V k,π⋆

1
(x1; ρke)
(lemma 2)

=
X

k∈[K]
V k,πk
1
(x1) −ˆV k
1 (x1) +
X

e∈[E]

X

k∈Ke

ˆV k
1 (x1) −¯V k,π⋆

1
(x1; ρke)

=
X

k∈[K]
V k,πk
1
(x1) −ˆV k
1 (x1)

|
{z
}
(i)−Bias / Cost of optimism

+
X

e∈[E]

X

h∈[H]
E ¯
P ke,π⋆

" X

k∈Ke

X

a∈A
ˆQk
h(xh, a)(πk
h(a | xh) −π⋆
h(a | xh))

#

|
{z
}
(ii)−OMD regret

+
X

e∈[E]

X

k∈Ke

X

h∈[H]
E ¯
P ke,π⋆
h
ˆQk
h(xh, ah) −¯φke
h (xh, ah)T(θk
h + ψh ˆV k
h+1)
i

|
{z
}
(iii)−Optimism

,

where the last relation is by the extended value difference lemma (see Shani et al. [2020] and lemma 14
in appendix B). This decomposition is very similar to the standard one for PO algorithms, but with
the crucial difference that term (iii) depends on the contracted features ¯φke
h (xh, ah) instead of the
true features φ(xh, ah). As a by-product, the expectation in terms (ii) and (iii) is taken with respect

8


---Page Break---
to the contracted MDP instead of the true one. The purpose of this modiﬁcation will be made clear in
the proof of optimism (see lemma 4).

In what follows, we bound each term deterministically, conditioned on the following “good event”:

E1 =
n
∀k ∈[K], h ∈[H] : ∥θk
h −bθk
h∥Λk
h ≤βr
o
;
(7)

E2 =
n
k ∈[K], h ∈[H] : ∥(ψh −bψk
h) ˆV k
h+1∥Λk
h ≤βp, ∥ˆQk
h+1∥∞≤2H
o
.
(8)

E1 and E2 are error bounds on the loss and dynamics estimation, respectively. In the full feedback
setting, E1 holds trivially with βr = 0. In the bandit setting, it holds with high probability with βr =
O(
p

d log(KH/δ)) by well-established bounds for regularized least-squares estimation [Abbasi-
Yadkori et al., 2011]. Showing that E2 holds with high probability follows similarly to Sherman et al.
[2023a], again using least-squares arguments but also using the contraction to ensure that ˆQk
h are
bounded (see sketch at the end of this section and lemma 6 in appendix A for full details), speciﬁcally
βp = O(Hd
p

log(KH/δ)). The proof of theorem 1 is concluded by bounding each of the terms in
the regret decomposition, summing over k ∈[K] and using a standard bound on elliptical potentials
(lemma 15 in appendix B). Term (ii) is bounded using a standard Online Mirror Descent (OMD)
argument (lemma 7 in appendix A).

Optimism and its cost.
The following lemmas bound terms (iii) and (i), respectively.

Lemma 4 (Optimism). Suppose that eqs. (7) and (8) hold, then

ˆQk
h(x, a) −¯φke
h (x, a)T(θk
h + ψh ˆV k
h+1) ≤0
, ∀h ∈[H], k ∈[K], x ∈X, a ∈A.

Proof. We have that
ˆQk
h(x, a) −¯φke
h (x, a)T(θh + ψh ˆV k
h+1) = ¯φke
h (x, a)T(bθk
h −θh + ( bψk
h −ψh) ˆV k
h+1)

−βb∥¯φke
h (x, a)∥(Λke
h )−1

≤(βr + βp)∥¯φke
h (x, a)∥Λk
h
−1 −βb∥¯φke
h (x, a)∥(Λke
h )−1

≤(βr + βp −βb)∥¯φke
h (x, a)∥(Λke
h )−1 = 0,

where the ﬁrst relation is by deﬁnition of ˆQk
h (eq. (3) in algorithm 1), the second relation is by eqs. (7)
and (8) together with Cauchy-Schwarz, the third relation follows since Λke
h ⪯Λk
h and the last one is
by our choice βb = βr + βp (see theorem 9 in appendix A for hyper-parameter choices).
■

Notice that the standard PO decomposition would have required that we bound the non-contracted
expression EP,π⋆[ ˆQk
h(x, a) −φ(x, a)T(θk
h + ψh ˆV k
h+1)]. In Sherman et al. [2023a] the gap between
this argument and that of lemma 4 can be bounded using eq. (6). However, the equivalent argument
for our contraction is lemma 3, which is bounded only for πk and not for any policy π ∈ΠM.

Lemma 5 (Cost of optimism). Suppose that eqs. (7) and (8) hold, then for every k ∈[K]

V k,πk

1
(x1) −ˆV k
1 (x1) ≤3(βr + βp)EP,πk



X

h∈[H]
∥φ(xh, ah)∥(Λk
h)−1





+ 16Hβ2
wEP,πk



X

h∈[H]
∥φ(xh, ah)∥2
(Λk
h)−1



+ 16H2K−1.

Proof. First, by lemma 14 in appendix B, a value difference lemma by Shani et al. [2020],

V k,πk

1
(x1) −ˆV k
1 (x1) = EP,πk



X

h∈[H]
φ(xh, ah)T
θh + ψh ˆV k
h+1

−ˆQk
k(xh, ah)



.

Now, using lemma 3 with v = θk
h + ψh ˆV k
h+1 we have that |φ(x, a)Tv| ≤4H (by eq. (8)) and thus

[φ(xh, ah) −¯φke
h (xh, ah)]T
θh + ψh ˆV k
h+1

≤16Hβ2
w∥φ(xh, ah)∥2
(Λk
h)−1 + 16H2K−1.

9


---Page Break---
We can thus conclude the proof using standard arguments to show that

¯φke
h (xh,ah)T
θh + ψh ˆV k
h+1

−ˆQk
k(xh, ah)

= ¯φke
h (xh, ah)T
θk
h −bθk
h + (ψh −bψk
h) ˆV k
h+1

+ βb∥¯φke
h (xh, ah)∥(Λke
h )−1
(eq. (3))

≤(βr + βp)∥¯φke
h (xh, ah)∥(Λk
h)−1 + βb∥¯φke
h (xh, ah)∥(Λke
h )−1
(Cauchy-Schwarz, eqs. (7) and (8))

≤3(βr + βp)∥¯φke
h (xh, ah)∥(Λk
h)−1
(det(Λk
h) ≤2 det(Λke
h ), βb = βr + βp)

≤3(βr + βp)∥φ(xh, ah)∥(Λk
h)−1,
(σ(x) ∈[0, 1], ∀x ∈R)

as desired.
■

Bounding the Q-values (proof sketch).
The following are the main ideas in showing that E2
(eq. (8)) holds with high probability. First, we deﬁne appropriate value classes bVh that contain
all value functions Vh of the form in eq. (4) whose underlying Qh function (eq. (3)) satisﬁes
∥Qh∥∞≤2(H + 1 −h). Because both the bonus and contraction operator are kept ﬁxed during
each epoch, the log covering number of this class is logarithmic in K (similarly to Sherman et al.
[2023a]). Thus, we can use standard least squares arguments (lemma 22) to show that with high
probability ∥(ψh −bψk
h)V ∥Λk
h ≤βp for all k ∈[K], h ∈[H], V ∈bVh. The proof is concluded by

showing that ∥ˆQk
h∥≤βQ,h = 2(H + 1 −h), and thus ˆV k
h ∈bVh, which implies that eq. (8) holds.
We prove this by backward induction on h ∈[H + 1].

The base case h = H + 1 is satisﬁed because, by deﬁnition, ˆQk
H+1 = 0. Now, suppose the claim
holds for h + 1 and we show it also holds for h. Recalling the deﬁnition of ˆQ in eq. (3), we have that

| ˆQk
h(x, a)| = |¯φke
h (x, a)T(bθk
h + bψk
h ˆV k
h+1) −βb∥¯φke
h (x, a)∥(Λke
h )−1|

≤|¯φke
h (x, a)T(θh + (bθk
h −θh) + ( bψk
h −ψh) ˆV k
h+1 + ψh ˆV k
h+1)| + βb∥¯φke
h (x, a)∥(Λke
h )−1

≤1 + ∥ˆV k
h+1∥∞+ ∥¯φke
h (x, a)∥(Λke
h )−1
h
∥bθk
h −θh∥Λk
h + ∥( bψk
h −ψh) ˆV k
h+1∥Λk
h + βb
i
,

where the last inequality also used the triangle and Cauchy-Schwarz inequalities, and that Λke
h ⪯Λk
h.
By the induction hypothesis, ∥ˆQk
h+1∥∞, ∥ˆV k
h+1∥∞≤βQ,h+1 and thus ˆV k
h+1 ∈bVh+1. Combining
with E1 (eq. (7)) and plugging into the above we get that

| ˆQk
h(x, a)| ≤1 + βQ,h+1 + (βr + βp,h + βb)∥¯φke
h (x, a)∥(Λke
h )−1.

Now, using a technical algebraic argument (lemma 18), we show that

∥¯φke
h (x, a)∥(Λke
h )−1 ≤max
y≥0 [y · σ(−βwy + log K)] ≤2β−1
w log(eK).

Finally, plugging this into the above and choosing βw ≥2(βr + βp,h + βb) log(eK), we get

| ˆQk
h(x, a)| ≤1 + βQ,h+1 + 2β−1
w (βr + βp,h + βb) log(eK) ≤2 + βQ,h+1 = βQ,h,
concluding the induction.

6
Conclusions

In this paper we presented a simple and efﬁcient contraction mechanism for policy optimization in
linear MDPs, yielding an overall algorithm with improved regret guarantees under both stochastic
(bandit feedback) and adversarial (full feedback) losses. We note that, in the stochastic setting, there
are value iteration based methods (He et al. [2023]) that use variance reduction techniques to achieve
better regret bounds. We conjecture that such techniques could be applicable to PO, however, this is
highly non-trivial and thus left for future research. Finally, regarding practical implementations, we
note that our bonuses and contraction technique are computationally feasible, especially compared
to the reward-free warmup phase in Sherman et al. [2023a]. Nonetheless, it remains open whether
our techniques could be applied heuristically to drive exploration in practical deep RL methods. In
particular, it would be interesting to examine the necessity of the contraction mechanism. These are
challenging questions on exploration in deep RL that we leave for future research.

10


---Page Break---
Acknowledgments and Disclosure of Funding

This project has received funding from the European Research Council (ERC) under the European
Union’s Horizon 2020 research and innovation program (grant agreement No. 101078075). Views
and opinions expressed are however those of the author(s) only and do not necessarily reﬂect those
of the European Union or the European Research Council. Neither the European Union nor the
granting authority can be held responsible for them. This work received additional support from the
Israel Science Foundation (ISF, grant number 2549/19), the Len Blavatnik and the Blavatnik Family
Foundation, and the Israeli VATAT data science scholarship.

References

Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. In
Advances in Neural Information Processing Systems, pages 2312–2320, 2011.

A. Agarwal, Y. Jin, and T. Zhang. Vo q l: Towards optimal regret in model-free rl with nonlinear
function approximation. In The Thirty Sixth Annual Conference on Learning Theory, pages
987–1063. PMLR, 2023.

A. Ayoub, Z. Jia, C. Szepesvari, M. Wang, and L. Yang. Model-based reinforcement learning with
value-targeted regression. In International Conference on Machine Learning, pages 463–474.
PMLR, 2020.

M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In
International Conference on Machine Learning, pages 263–272. PMLR, 2017.

Q. Cai, Z. Yang, C. Jin, and Z. Wang. Provably efﬁcient exploration in policy optimization. In
International Conference on Machine Learning, pages 1283–1294. PMLR, 2020.

A. Cassel, H. Luo, A. Rosenberg, and D. Sotnikov. Near-optimal regret in linear mdps with aggregate
bandit feedback. arXiv preprint arXiv:2405.07637, 2024.

A. Cohen, T. Koren, and Y. Mansour. Learning linear-quadratic regulators efﬁciently with only
√

T
regret. In International Conference on Machine Learning, pages 1300–1309, 2019.

Y. Dai, H. Luo, C.-Y. Wei, and J. Zimmert. Reﬁned regret for adversarial mdps with linear function
approximation. In International Conference on Machine Learning, pages 6726–6759. PMLR,
2023.

C. Dann, C.-Y. Wei, and J. Zimmert. Best of both worlds policy optimization. In International
Conference on Machine Learning, pages 6968–7008. PMLR, 2023.

K. Dong, J. Peng, Y. Wang, and Y. Zhou. Root-n-regret for learning in markov decision processes
with function approximation and low bellman rank. In Conference on Learning Theory, pages
1554–1557. PMLR, 2020.

S. Du, S. Kakade, J. Lee, S. Lovett, G. Mahajan, W. Sun, and R. Wang. Bilinear classes: A structural
framework for provable generalization in rl. In International Conference on Machine Learning,
pages 2826–2836. PMLR, 2021.

E. Even-Dar, S. M. Kakade, and Y. Mansour. Online markov decision processes. Mathematics of
Operations Research, 34(3):726–736, 2009.

T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep
reinforcement learning with a stochastic actor. In International Conference on Machine Learning,
pages 1861–1870. PMLR, 2018.

J. He, D. Zhou, and Q. Gu. Near-optimal policy optimization algorithms for learning adversarial
linear mixture mdps. In International Conference on Artiﬁcial Intelligence and Statistics, pages
4259–4280. PMLR, 2022.

J. He, H. Zhao, D. Zhou, and Q. Gu. Nearly minimax optimal reinforcement learning for linear
markov decision processes. In International Conference on Machine Learning, pages 12790–12822.
PMLR, 2023.

11


---Page Break---
P. Hu, Y. Chen, and L. Huang. Nearly minimax optimal reinforcement learning with linear function
approximation. In International Conference on Machine Learning, pages 8971–9019. PMLR,
2022.

T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning. Journal of
Machine Learning Research, 11:1563–1600, 2010.

N. Jiang, A. Krishnamurthy, A. Agarwal, J. Langford, and R. E. Schapire. Contextual decision
processes with low bellman rank are pac-learnable. In International Conference on Machine
Learning, pages 1704–1713. PMLR, 2017.

C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is q-learning provably efﬁcient? Advances in
neural information processing systems, 31, 2018.

C. Jin, T. Jin, H. Luo, S. Sra, and T. Yu. Learning adversarial markov decision processes with
bandit feedback and unknown transition. In International Conference on Machine Learning, pages
4860–4869. PMLR, 2020a.

C. Jin, Z. Yang, Z. Wang, and M. I. Jordan. Provably efﬁcient reinforcement learning with linear
function approximation. In Conference on Learning Theory, pages 2137–2143. PMLR, 2020b.

C. Jin, Q. Liu, and S. Miryooseﬁ. Bellman eluder dimension: New rich classes of rl problems, and
sample-efﬁcient algorithms. Advances in neural information processing systems, 34:13406–13418,
2021.

F. Kong, X. Zhang, B. Wang, and S. Li. Improved regret bounds for linear adversarial mdps via linear
optimization. arXiv preprint arXiv:2302.06834, 2023.

T. Lancewicki, A. Rosenberg, and Y. Mansour. Learning adversarial markov decision processes with
delayed feedback. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 36,
pages 7281–7289, 2022.

T. Lancewicki, A. Rosenberg, and D. Sotnikov. Delay-adapted policy optimization and improved
regret for adversarial MDP with delayed bandit feedback. In A. Krause, E. Brunskill, K. Cho,
B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning,
ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine
Learning Research, pages 18482–18534. PMLR, 2023.

H. Liu, C.-Y. Wei, and J. Zimmert. Towards optimal regret in adversarial linear mdps with bandit
feedback. arXiv preprint arXiv:2310.11550, 2023.

H. Luo, C.-Y. Wei, and C.-W. Lee. Policy optimization in adversarial mdps: Improved exploration
via dilated bonuses. Advances in Neural Information Processing Systems, 34, 2021.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried-
miller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement
learning. nature, 518(7540):529–533, 2015.

A. Modi, N. Jiang, A. Tewari, and S. Singh. Sample complexity of reinforcement learning using
linearly combined model ensembles. In International Conference on Artiﬁcial Intelligence and
Statistics, pages 2010–2020. PMLR, 2020.

R. Munos. Error bounds for approximate value iteration. In Proceedings of the National Conference
on Artiﬁcial Intelligence, volume 20, page 1006. Menlo Park, CA; Cambridge, MA; London;
AAAI Press; MIT Press; 1999, 2005.

G. Neu and J. Olkhovskaya. Online learning in mdps with linear function approximation and bandit
feedback. Advances in Neural Information Processing Systems, 34:10407–10417, 2021.

G. Neu, A. György, C. Szepesvári, and A. Antos. Online markov decision processes under bandit
feedback. In J. D. Lafferty, C. K. I. Williams, J. Shawe-Taylor, R. S. Zemel, and A. Culotta,
editors, Advances in Neural Information Processing Systems 23: 24th Annual Conference on
Neural Information Processing Systems 2010. Proceedings of a meeting held 6-9 December 2010,
Vancouver, British Columbia, Canada, pages 1804–1812. Curran Associates, Inc., 2010a.

12


---Page Break---
G. Neu, A. György, C. Szepesvári, et al. The online loop-free stochastic shortest-path problem. In
COLT, volume 2010, pages 231–243. Citeseer, 2010b.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,
A. Ray, et al. Training language models to follow instructions with human feedback. Advances in
Neural Information Processing Systems, 35:27730–27744, 2022.

A. Rosenberg and Y. Mansour. Online stochastic shortest path with bandit feedback and unknown
transition function. In Advances in Neural Information Processing Systems, pages 2209–2218,
2019a.

A. Rosenberg and Y. Mansour. Online convex optimization in adversarial markov decision processes.
In International Conference on Machine Learning, pages 5478–5486. PMLR, 2019b.

A. Rosenberg, A. Cohen, Y. Mansour, and H. Kaplan. Near-optimal regret bounds for stochastic
shortest path. In International Conference on Machine Learning, pages 8210–8219. PMLR, 2020.

J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In
International conference on machine learning, pages 1889–1897. PMLR, 2015.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization
algorithms. arXiv preprint arXiv:1707.06347, 2017.

L. Shani, Y. Efroni, A. Rosenberg, and S. Mannor. Optimistic policy optimization with bandit
feedback. In International Conference on Machine Learning, pages 8604–8613. PMLR, 2020.

U. Sherman, A. Cohen, T. Koren, and Y. Mansour. Rate-optimal policy optimization for linear markov
decision processes. arXiv preprint arXiv:2308.14642, 2023a.

U. Sherman, T. Koren, and Y. Mansour. Improved regret for efﬁcient online reinforcement learning
with linear function approximation. In International Conference on Machine Learning, pages
31117–31150. PMLR, 2023b.

N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F.
Christiano. Learning to summarize with human feedback. Advances in Neural Information
Processing Systems, 33:3008–3021, 2020.

A. J. Wagenmaker, Y. Chen, M. Simchowitz, S. Du, and K. Jamieson. First-order regret in reinforce-
ment learning with linear function approximation: A robust estimation approach. In International
Conference on Machine Learning, pages 22384–22429. PMLR, 2022a.

A. J. Wagenmaker, Y. Chen, M. Simchowitz, S. Du, and K. Jamieson. Reward-free rl is no harder
than reward-aware rl in linear markov decision processes. In International Conference on Machine
Learning, pages 22430–22456. PMLR, 2022b.

T. Wu, Y. Yang, H. Zhong, L. Wang, S. Du, and J. Jiao. Nearly optimal policy optimization with stable
at any time guarantee. In International Conference on Machine Learning, pages 24243–24265.
PMLR, 2022.

A. Zanette and E. Brunskill. Tighter problem-dependent regret bounds in reinforcement learning
without domain knowledge using value function bounds. In International Conference on Machine
Learning, pages 7304–7312. PMLR, 2019.

A. Zanette, D. Brandfonbrener, E. Brunskill, M. Pirotta, and A. Lazaric. Frequentist regret bounds
for randomized least-squares value iteration. In International Conference on Artiﬁcial Intelligence
and Statistics, pages 1954–1964. PMLR, 2020.

Z. Zhang, Y. Zhou, and X. Ji. Almost optimal model-free reinforcement learningvia reference-
advantage decomposition. Advances in Neural Information Processing Systems, 33:15198–15207,
2020.

Z. Zhang, J. Yang, X. Ji, and S. S. Du. Improved variance-aware conﬁdence sets for linear bandits and
linear mixture mdp. Advances in Neural Information Processing Systems, 34:4342–4355, 2021.

13


---Page Break---
Z. Zhang, J. D. Lee, Y. Chen, and S. S. Du. Horizon-free regret for linear markov decision processes.
arXiv preprint arXiv:2403.10738, 2024.

H. Zhong and T. Zhang. A theoretical analysis of optimistic proximal policy optimization in linear
markov decision processes. Advances in Neural Information Processing Systems, 36, 2023.

D. Zhou and Q. Gu. Computationally efﬁcient horizon-free reinforcement learning for linear mixture
mdps. Advances in neural information processing systems, 35:36337–36349, 2022.

D. Zhou, Q. Gu, and C. Szepesvari. Nearly minimax optimal reinforcement learning for linear
mixture markov decision processes. In Conference on Learning Theory, pages 4532–4576. PMLR,
2021a.

D. Zhou, J. He, and Q. Gu. Provably efﬁcient reinforcement learning for discounted mdps with
feature mapping. In International Conference on Machine Learning, pages 12793–12802. PMLR,
2021b.

A. Zimin and G. Neu. Online learning in episodic markovian decision processes by relative entropy
policy search. In Advances in Neural Information Processing Systems 26: 27th Annual Conference
on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8,
2013, Lake Tahoe, Nevada, United States, pages 1583–1591, 2013.

14


---Page Break---
A
Analysis

We begin by deﬁning a so-called “good event”, followed by optimism, cost of optimism, and Policy
Optimization cost. We conclude with the proof of theorem 9.

Good event.
We deﬁne the following good event Eg = T3
i=1 Ei, over which the regret is determin-
istically bounded:

E1 =
n
∀k ∈[K], h ∈[H] : ∥θk
h −bθk
h∥Λk
h ≤βr
o
;
(eq. (7))

E2 =
n
k ∈[K], h ∈[H] : ∥(ψh −bψk
h) ˆV k
h+1∥Λk
h ≤βp, ∥ˆQk
h+1∥∞≤βQ
o
;
(eq. (8))

E3 =






X

k∈[K]
EP,πk[Yk] ≤
X

k∈[K]
2Yk + 4H(3(βr + βp) + 4βQβ2
w) log 6

δ




.
(9)

where Yk = P

h∈[H] 3(βr + βp)∥φ(xh, ah)∥(Λk
h)−1 + 4βQβ2
w∥φ(xh, ah)∥2
(Λk
h)−1.

Lemma 6 (Good event). Consider the parameter setting of theorem 9. If ηo ≤1, β2
w ≤K/(32Hd)
then Pr[Eg] ≥1 −δ.

Proof in appendix A.1.

Policy online mirror descent.
We use standard online mirror descent arguments to bound the local
regret in each state.

Lemma 7 (OMD). Suppose that the good event Eg holds (eqs. (7), (8) and (9)) and set ηo ≤1/βQ,
then
X

k∈Ke

X

a∈A
ˆQk
h(x, a)(π⋆
h(a | x) −πk
h(a | x)) ≤log|A|

ηo
+ ηo
X

k∈Ke
β2
Q
, ∀e ∈[E], h ∈[H], x ∈X.

Proof. Notice that the policy πk is reset at the beginning of every epoch. Then, the lemma follows
directly by lemma 13 with yt(a) = −ˆQk
h(x, a), xt(a) = πk
h(a | x) and noting that | ˆQk
h(x, a)| ≤βQ
by eq. (8).
■

Epoch schedule.
The algorithm operates in epochs. At the beginning of each epoch, the policy is
reset to be uniformly random. We denote the total number of epochs by E, the ﬁrst episode within
epoch e by ke, and the set of episodes within epoch e by Ke. The following lemma bounds the
number of epochs.

Lemma 8. The number of epochs E is bounded by (3/2)dH log(2K).

Proof. Let Th = {e1
h, e2
h, . . .} be the epochs where the condition det(Λk
h) ≥2 det(Λke
h ) was trig-
gered in line 4 of algorithm 1. Then we have that

det(Λke
h ) ≥

(
2 det(Λke−1
h
)
, e ∈Th
det(Λke−1
h
)
, otherwise.

Unrolling this relation, we get that

det(ΛK
h ) ≥2|Th|−1 det I = 2|Th|−1,

and changing sides, and taking the logarithm we get that

|Th| ≤1 + log2 det
 
ΛK
h


≤1 + d log2∥ΛK
h ∥
(det(A) ≤∥A∥d)

≤1 + d log2

 

1 +

K−1
X

k=1
∥φk
h∥2
!

(triangle inequality)

≤1 + d log2 K
(∥φk
h∥≤1)
≤(3/2)d log 2K.

15


---Page Break---
We conclude that

E = |
 
∪h∈[H]Th

| ≤
X

h∈[H]
|Th| ≤(3/2)dH log(2K).
■

Regret bound.

Theorem 9. Suppose that we run algorithm 1 with parameters

ηo =

s

3dH log(2K) log|A|

Kβ2
Q
, βb = βr + βp, βw = 4(βr + βp) log(eK),

where βr = 2
p

2d log(6KH/δ), βp = 28Hd
p

log(10K5H/δ), βQ = 2H. Then with probability
at least 1 −δ we incur regret at most

Regret ≤264
p

Kd3H4 log(2K) log(10K5H/δ) + 8
p

KdH5 log(2K) log|A|

+ 64H2d max{β2
w, log|A|} log 12K

δ
= O(
p

Kd3H4 log(K) log(KH/δ) +
p

KdH5 log(K) log|A|).

Proof. First, if β2
w > K/(32Hd) or η ≥1/βQ then

Regret ≤KH ≤32H2d max{β2
w, log|A|} log(2K),

and the proof is concluded. Otherwise, if β2
w ≤K/(32Hd) then suppose that the good event Eg
holds (eqs. (7), (8) and (9)). By lemma 6, this holds with probability at least 1 −δ. For any epoch
e ∈[E], let Ke be the set of episodes that it contains, and let ¯V k,π
1
(x1; ρke) denote the value of its
contracted MDP as deﬁned in section 5.1 and line 6 of algorithm 1. We bound the regret as

Regret =
X

k∈[K]
V k,πk
1
(x1) −V k,π⋆

1
(x1)

≤
X

e∈[E]

X

k∈Ke
V k,πk
1
(x1) −¯V k,π⋆

1
(x1; ρke)
(lemma 2)

=
X

k∈[K]
V k,πk
1
(x1) −ˆV k
1 (x1) +
X

e∈[E]

X

k∈Ke

ˆV k
1 (x1) −¯V k,π⋆

1
(x1; ρke)

=
X

k∈[K]
V k,πk
1
(x1) −ˆV k
1 (x1)

|
{z
}
(i)−Bias / Cost of optimism

+
X

e∈[E]

X

h∈[H]
E ¯
P ke,π⋆

" X

k∈Ke

X

a∈A
ˆQk
h(xh, a)(πk
h(a | xh) −π⋆
h(a | xh))

#

|
{z
}
(ii)−OMD regret

+
X

e∈[E]

X

k∈Ke

X

h∈[H]
E ¯
P ke,π⋆
h
ˆQk
h(xh, ah) −¯φke
h (xh, ah)T(θk
h + ψh ˆV k
h+1)
i

|
{z
}
(iii)−Optimism

,

where the last relation is by the extended value difference lemma (see Shani et al. [2020] and lemma 14
in appendix B).

16


---Page Break---
For term (i), we use lemma 5 as follows.

(i) ≤
X

k∈[K]
EP,πk



X

h∈[H]
3(βr + βp)∥φ(xh, ah)∥(Λk
h)−1 + 8βQβ2
w∥φ(xh, ah)∥2
(Λk
h)−1



+ 8HβQ

≤
X

k∈[K]



X

h∈[H]
6(βr + βp)∥φ(xk
h, ak
h)∥(Λk
h)−1 + 16βQβ2
w∥φ(xk
h, ak
h)∥2
(Λk
h)−1



+ 20HβQβ2
w log 6

δ

(eq. (9), βw ≥4(βr + βp) ≥32)

≤6(βr + βp)H
p

2Kd log(2K) + 32βQβ2
wHd log(2K) + 20HβQβ2
w log 6

δ
(lemma 15)

≤6(βr + βp)H
p

2Kd log(2K) + 32HdβQβ2
w log 12K

δ
.

By lemmas 7 and 8 (with our choice of ηo) we have

(ii) ≤
X

h∈[H]

X

e∈[E]
E ¯
P ke,π⋆

"
log A

ηo
+ ηo
X

k∈Ke
β2
Q

#

≤4HβQ
p

KdH log(2K) log|A|.

By lemma 4 (iii) ≤0. Putting all bounds together, we get that

Regret ≤6(βr + βp)H
p

2Kd log(2K) + 32HdβQβ2
w log 12K

δ
+ 4HβQ
p

KdH log(2K) log|A|

≤264
p

Kd3H4 log(2K) log(10K5H/δ) + 8
p

KdH5 log(2K) log|A| + 64H2dβ2
w log 12K

δ
= O(
p

Kd3H4 log(K) log(KH/δ) +
p

KdH5 log(K) log|A|).
■

A.1
Proofs of good event

We begin by deﬁning function classes and properties necessary for the uniform convergence arguments
over the value functions. We then proceed to deﬁne a proxy good event, whose high probability
occurrence is straightforward to prove. We then show that the proxy event implies the desired good
event.

Value and policy classes.
We deﬁne the following class of restricted Q-functions:

bQ(Cβ, Cw, CQ)

=
n
ˆQ(·, ·; β, w, Λ, Z) | β ∈[0, Cβ], ∥w∥≤Cw, (2K)−1I ⪯Λ ⪯I, ∥ˆQ(·, ·; w, Λ, Z)∥∞≤CQ
o
,

where ˆQ(x, a; β, w, Λ) = [wTφ(x, a) −β∥φ(x, a)∥Λ] · σ(−βw∥φ(x, a)∥Λ + log K). Next, we de-
ﬁne the following class of soft-max policies:

Π(Cβ, Cw) =
n
π(· | ·; ˆQ) | ˆQ ∈bQ(Cβ, Cw, ∞)
o
,

where π(a | x; ˆQ) =
exp( ˆ
Q(x,a))
P

a′∈A exp( ˆ
Q(x,a′)). Finally, we deﬁne the following class of restricted value

functions:

bV(Cβ, Cw, CQ) =
n
ˆV (·; π, ˆQ) | π ∈Π(CβK, CwK, CQ), ˆQ ∈bQ(Cβ, Cw, CQ)
o
,
(10)

where ˆV (x; π, ˆQ) = P

a∈A π(a | x) ˆQ(x, a). The following lemma provides the bound on the
covering number of the value function class deﬁned above.

Lemma 10. For any ϵ, Cw > 0, Cβ, CQ ≥1, we have

log Nϵ

bV(Cβ, Cw, CQ)

≤6d2 log(1 + 4(
√

192K3CQCββw)(KCβ + KCw +
√

d)/ϵ),

where Nϵ is the covering number of a class in supremum distance.

17


---Page Break---
Proof. We begin by showing that the class of Q function is Lipschitz in its parameters. For ease of
notation, denote y = φ(x, a). Then
∥∇βQ(x, a; β, w, Λ)∥= ∥y∥Λ · σ(−βw∥y∥Λ + log K) ≤1
(σ(·) ∈[0, 1], ∥y∥≤1, Λ ⪯I)

∥∇θ ˆQ(x, a; β, w, Λ)∥= ∥y · σ(−βw∥y∥Λ + log K)∥≤1
(σ(·) ∈[0, 1], ∥y∥≤1)

|Q(x, a;β, w, Λ) −Q(x, a; β, w, Λ′)|
≤β|∥y∥Λ −∥y∥Λ′| · σ(−βw∥y∥Λ + log K)
+ β∥y∥Λ′|σ(−βw∥y∥Λ + log K) −σ(−βw∥y∥Λ′ + log K)|

≤β∥(Λ1/2 −(Λ′)1/2)y∥+ ββw∥y∥Λ′∥(Λ1/2 −(Λ′)1/2)y∥
(∥·∥, σ(·) 1-Lipschitz, σ ∈[0, 1])

≤2ββw∥Λ1/2 −(Λ′)1/2∥
(∥y∥≤1, Λ ⪯I, βw ≥1)

≤
√

2Kββw∥Λ −Λ′∥
(lemma 17, Λ, Λ′ ⪰(2K)−1I)

≤
√

2Kββw∥Λ −Λ′∥F .
(∥·∥≤∥·∥F )
We thus have that for any such y
|Q(x, a; β, w,Λ) −Q(x, a; β′, w′, Λ′)|

≤|Q(x, a; β, w, Λ) −Q(x, a; β′, w, Λ)| + |Q(x, a; β′, w, Λ) −Q(x, a; β′, w′, Λ)|

+ |Q(x, a; β′, w′, Λ) −Q(x, a; β′, w′, Λ′)|

≤|β −β′| + ∥w −w′∥+
√

2Kββw∥Λ −Λ′∥F

≤
q

3(∥w −w′∥2 + |β −β′|2 + (
√

2Kββw)2∥Λ −Λ′∥2
F )

≤max{3,
√

6Kββw}
q

(∥w −w′∥2 + |β −β′|2 + ∥Λ −Λ′∥2
F )

= max{3,
√

6Kββw}∥(β, w, Λ) −(β′, w′, Λ′)∥,
where (β, w, Λ) is a vectorization of the parameters. Assuming that Cβ ≥1, we conclude that
bQ(Cβ, Cw, CQ) is
√

6KCββw−Lipschitz in supremum norm, i.e.,

∥ˆQ(·, ·; β, w, Λ) −ˆQ′(·, ·; β′, w′, Λ′)∥∞≤
√

6KCββw∥(β, w, Λ) −(β′, w′, Λ′)∥.
Next, notice that our policy class Π(CβK, CwK) is a soft-max over the Q functions thus ﬁtting
Lemma 12 of Sherman et al. [2023a]. We conclude that the policy class is
√

24K3Cββw−Lipschitz,
in ℓ1−norm, i.e.,

∥π(· | x; β, w, Λ) −π(· | x; β′, w′, Λ′)∥1 ≤
√

24K3Cββw∥(β, w, Λ) −(β′, w′, Λ′)∥.

Now, let V, V ′ ∈bV(Cβ, Cw, CQ) and θ = (β1, w1, Λ1, β2, w2, Λ2), θ′ = (β′
1, w′
1, Λ′
1, β′
2, w′
2, Λ2) ∈
R2(1+d+d2) be their respective parameters. We have that for all x ∈X

|V (x; π, ˆQ) −V (x; π′, ˆQ′)| ≤|V (x; π, ˆQ) −V (x; π, ˆQ′)|
|
{z
}
(i)

+ |V (x; π, ˆQ′) −V (x; π′, ˆQ′)|
|
{z
}
(ii)

.

For the ﬁrst term

(i) =



X

a∈A
π(a | x)( ˆQ(x, a; β2, w2, Λ2) −ˆQ(x, a; β′
2, w′
2, Λ′
2))



≤
X

a∈A
π(a | x)
 ˆQ(x, a; β2, w2, Λ2) −ˆQ(x, a; β′
2, w′
2, Λ′
2)

(triangle inequality)

≤
√

6KCββw∥(β2, w2, Λ2) −(β′
2, w′
2, Λ′
2)∥. ( ˆQ is
√

6KCββw-Lipschitz, Cauchy-Schwarz)
For the second term

(ii) =



X

a∈A
ˆQ′(x, a)(π(a | x) −π′(a | x))

 ≤CQ∥π(· | x) −π(· | x)∥1

≤
√

96K3CQCββw∥(β1, w1, Λ1) −(β′
1, w′
1, Λ′
1)∥,

18


---Page Break---
where the ﬁrst transition used that ∥Q∥∞≤CQ for all Q ∈bQ(Cβ, Cw, CQ) and the second used the
Lipschitz property of the policy class shown above. Combining the terms and assuming that CQ ≥1
we get that

|V (x; π, ˆQ) −V (x; π′, ˆQ′)| ≤
√

96K3CQCββw∥(β1, w1, Λ1) −(β′
1, w′
1, Λ′
1)∥

+
√

96K3CQCββw∥(β2, w2, Λ2) −(β′
2, w′
2, Λ′
2)∥

≤
√

192K3CQCββw∥θ −θ′∥,

implying that bV(Cβ, Cw, CQ) is
√

192K3CQCββw−Lipschitz in supremum norm. Finally, notice
that

∥θ∥≤|β1| + |β2| + ∥w1∥+ ∥w2∥+ ∥Λ1∥F + ∥Λ2∥F ≤2KCβ + 2KCw + 2
√

d,

and applying lemma 24 concludes the proof.
■

Proxy good event.
We deﬁne a proxy good event ¯Eg = E1 ∩¯E2 ∩E3 where

¯E2 =
n
k ∈[K], h ∈[H], V ∈bV(βr + βp, 2βQK, βQ,h+1) : ∥(ψh −bψk
h)V ∥Λk
h ≤βp
o
,
(11)

where βQ,h = 2(H + 1 −h), h ∈[H + 1]. Then we have the following result.

Lemma 11 (Proxy good event). Consider the parameter setting of lemma 6. Then Pr[ ¯Eg] ≥1 −δ.

Proof. First, by lemma 21 and our choice of parameters, E1 (eq. (7)) holds with probability at least
1−δ/3. Next, applying lemmas 10 and 22, we get that with probability at least 1−δ/3 simultaneously
for all k ∈[K], h ∈[H], V ∈bV(βr + βp, 2βQK, βQ,h+1)

∥(ψh −bψk
h)V ∥Λk
h

≤4βQ,h+1
q

d log(2K) + 2 log(6H/δ) + 12d2 log(1 + 8K(
√

192K3Cββw)(KCβ + KCw + 1))

≤4βQ

r

d log(2K) + 2 log(6H/δ) + 12d2 log(1 + 2K(
√

192K3K/(32Hd))(1

4K
p

K/(32Hd) + 2βQK2 + 1))

≤4βQ
q

d log(2K) + 2 log(6H/δ) + 12d2 log(7K9/2)

≤4βQd
p

12 log(10K5H/δ)

≤28Hd
p

log(10K5H/δ)
= βp,

implying ¯E2 (eq. (11)). Finally, notice that ∥φk
h∥(Λk
h)−1 ≤1, thus 0 ≤Yk ≤H(3(βr+βp)+4βQβ2
w).
Using lemma 20, a Bernstein-type inequality for martingales, we conclude that E3 (eq. (9)) holds
with probability at least 1 −δ/3.
■

The good event.
The following results show that the proxy good event implies the good event.

Lemma 12. Suppose that ¯Eg holds. If πk
h ∈Π(K(βr + βp), 2βQK2) for all h ∈[H] then ˆQk
h ∈
bQ(βr + βp, 2βQK, βQ,h), ˆV k
h ∈bV(βr + βp, 2βQK, βQ,h) for all h ∈[H + 1].

Proof. We show that the claim holds by backward induction on h ∈[H + 1].
Base case h = H + 1: Since ˆV k
H+1 = 0 it is also implied that ˆQk
H+1 = 0. Because β, w = 0 ∈
bQ(βr + βp, 2βQK, βQ,H+1 = 0) we have that ˆQk
H+1 ∈bQ(βr + βp, 2βQK, βQ,H+1 = 0), and
similarly V k
H+1 ∈bV(βr + βp, 2βQK, βQ,H+1 = 0).

19


---Page Break---
Induction step: Now, suppose the claim holds for h + 1 and we show it also holds for h. We have
that

| ˆQk
h(x, a)| = |¯φke
h (x, a)Twk
h −βb∥¯φke
h (x, a)∥(Λke
h )−1|

≤|¯φke
h (x, a)T(θh + (bθk
h −θh) + ( bψk
h −ψh) ˆV k,i
h+1 + ψh ˆV k,i
h+1)| + βb∥¯φke
h (x, a)∥(Λke
h )−1

≤1 + ∥ˆV k,i
h+1∥∞+ ∥¯φke
h (x, a)∥(Λke
h )−1
h
∥bθk
h −θh∥Λk
h + ∥( bψk
h −ψh) ˆV k,i
h+1∥Λk
h + βb
i

(triangle inequality, Cauchy-Schwarz, Λke
h ⪯Λk
h)

≤1 + βQ,h+1 + (βr + βp,h + βb)∥¯φke
h (x, a)∥(Λke
h )−1
(induction hypothesis, eqs. (7) and (11))

≤1 + βQ,h+1 + (βr + βp,h + βb) max
y≥0 [y · σ(−βwy + log K)]
(¯φke
h deﬁnition)

≤1 + βQ,h+1 + 2 log(eK)

βw
(βr + βp,h + βb)
(lemma 18)

≤2 + βQ,h+1
(βw ≥2(βr + βp,h + βb) log(eK))
= βQ,h.

Additionally, βb = βr + βp, (Λke
h )−1 ⪯I, ∥Λke
h ∥≤1 + P
k∈[K]∥φk
h∥≤2K, thus (Λke
h )−1 ⪰
(2K)−1I, and

∥wk
h∥= ∥bθk
h + bψk
h ˆV k,i
h+1∥≤K + βQK ≤2βQK = Cw.

We conclude that ˆQk
h ∈bQ(βr + βp, 2βQK, βQ,h). Since πk
h ∈Π(K(βr + βp), 2βQK2), we also
conclude that ˆV k
h ∈bV(βr +βp, 2βQK, βQ,h), proving the induction step and ﬁnishing the proof.
■

Lemma (restatement of lemma 6). Consider the parameter setting of theorem 9. If ηo ≤1, β2
w ≤
K/(32Hd) then Pr[Eg] ≥1 −δ.

Proof. Suppose that ¯Eg holds. By lemma 11, this occurs with probability at least 1 −δ. We show
that ¯Eg implies Eg, thus concluding the proof. Notice that

πk
h(a|x) ∝exp

 

η

k−1
X

k′=ke

ˆQk′
h (x, a)

!

= exp

 

σ(−βw∥φ(x, a)∥(Λke
h )−1 + log K) ·

"

φ(x, a)T
k−1
X

k′=ke
ηwk
h −ηβb(k −ke)∥φ(x, a)∥(Λke
h )−1

#!

.

We show by induction on k ∈Ke that πk
h ∈Π(K(βr + βp), 2βQK2) for all h ∈[H]. For the base
case, k = ke, πk
h are uniform, corresponding to w, β = 0 ∈Π(K(βr + βp), 2βQK2). Now, suppose
the claim holds for all k′ < k. Then by lemma 12 we have that ˆQk′
h ∈bQ(βr + βp, 2βQK, βQ,h)
for all k′ < k and h ∈[H]. This implies that ∥Pk−1
k′=ke ηwk
h∥≤2βQK2 for all h ∈[H], thus
πk
h ∈Π(K(βr + βp), 2βQK2) for all h ∈[H], concluding the induction step.

Now, since πk
h ∈Π(K(βr + βp), 2βQK2) for all k ∈[K], h ∈[H], we can apply lemma 12 to
get that ˆQk
h ∈bQ(βr + βp, 2βQK, βQ,h), ˆV k
h ∈bV(βr + βp, 2βQK, βQ,h) for all k ∈[K], h ∈[H].
Using ¯E2 (eq. (11)) we conclude that E2 (eq. (8)) holds, thus concluding the proof.
■

20


---Page Break---
B
Technical tools

B.1
Online Mirror Descent

We begin with a standard regret bound for entropy regularized online mirror descent (hedge). See
[Sherman et al., 2023a, Lemma 25].

Lemma 13. Let y1, . . . , yT ∈RA be any sequence of vectors, and η > 0 such that ηyt(a) ≥−1 for
all t ∈[T], a ∈[A]. Then if xt ∈∆A is given by x1(a) = 1/A ∀a, and for t ≥1:

xt+1(a) =
xt(a)e−ηyt(a)
P

a′∈[A] xt(a′)e−ηyt(a′) ,

then,

max
x∈∆A

T
X

t=1

X

a∈[A]
yt(a)(xt(a) −x(a)) ≤log A

η
+ η

T
X

t=1

X

a∈[A]
xt(a)yt(a)2.

B.2
Value difference lemma

We use the following extended value difference lemma by Shani et al. [2020]. We note that the lemma
holds unchanged even for MDP-like structures where the transition kernel P is a sub-stochastic
transition kernel, i.e., one with non-negative values that sum to at most one (instead of exactly one).

Lemma 14 (Extended Value difference Lemma 1 in Shani et al. [2020]). Let M be a (sub) MDP,
π, ˆπ ∈ΠM be two policies, ˆQh : X × A →R, h ∈[H] be arbitrary function, and ˆVh : X →R be
deﬁned as ˆVh(x) = P

a∈A ˆπh(a | x) ˆQh(x, a). Then

V π
1 (x1) −ˆV1(x1) = EP,π



X

h∈[H]

X

a∈A
ˆQh(xh, a)(π(a | xh) −ˆπ(a | xh))





+ EP,π



X

h∈[H]
ℓh(xh, ah) +
X

x′∈X
P(x′ | xh, ah) ˆVh+1(x′) −ˆQh(xh, ah)



.

We note that, in the context of linear MDP ℓh(xh, ah) + P

x′∈X P(x′ | xh, ah) ˆVh+1(x′) =
φ(xh, ah)T(θh + ψh ˆVh+1).

B.3
Algebraic lemmas

Next, is a well-known bound on harmonic sums [see, e.g., Cohen et al., 2019, Lemma 13]. This is
used to show that the optimistic and true losses are close on the realized predictions.

Lemma 15. Let zt ∈Rd′ be a sequence such that ∥zt∥2 ≤λ, and deﬁne Vt = λI + Pt−1
s=1 zszT
s .
Then

T
X

t=1
∥zt∥V −1
t
≤

v
u
u
tT

T
X

t=1
∥zt∥2
V −1
t
≤
p

2Td′ log(T + 1).

Next, we need the following well-known matrix inequality.

Lemma 16 (Cohen et al. [2019], Lemma 27). If N ⪰M ≻0 then for any vector v

∥v∥2
N ≤det N

det M ∥v∥2
M

Next, we need a bound on the Lipschitz constant of the spectral norm of a square-root matrix.

Lemma 17. For any λ > 0 and matrices Λ, Λ′ ∈Rd×d satisfying Λ, Λ′ ⪰λI we have that

∥Λ1/2 −Λ′1/2∥≤
1
2
√

λ
∥Λ −Λ′∥.

21


---Page Break---
Proof. Let µ be an eigenvalue of Λ1/2 −Λ′1/2 with eigenvector x ∈Rd. Then we have that

|xT(Λ −Λ′)x| = |xT(Λ1/2 −Λ′1/2)Λ1/2x + xTΛ′1/2(Λ1/2 −Λ′1/2)x|

= |µ|xT(Λ1/2 + Λ′1/2)x.

Next, notice that |xT(Λ −Λ′)x| ≤∥x∥2∥Λ −Λ′∥, and xT(Λ1/2 + Λ′1/2) ≥2
√

λ∥x∥2. We thus
therefore change sides to get that

|µ| ≤
1
2
√

λ
∥Λ −Λ′∥,

and since we can take µ = ±∥Λ1/2 −Λ′1/2∥, the proof is concluded.
■

Finally, we need the following bounds on the logistic function.

Lemma 18. For any K ≥1, β > 0 we have that

max
y≥0 [y · σ(−βy + log K)] ≤2 log(eK)

β

Proof. First, if y′ ≤(2 log K)/β then using σ(y) ∈[0, 1] we have that

y′σ(−βy′ + log K) ≤y′ ≤(2 log K)/β,

as desired. Now, if y′ ≥(2 log K)/β then

y′σ(−βy′ + log K) ≤y′σ(−βy′/2) =
y′

1 + eβy′/2 ≤
y′

βy′/2 = 2

β ,

where the ﬁrst inequality also used that σ(y) is increasing and the last inequality used that 1 + ey ≥y
for all y ≥0.
■

Lemma 19. For any K ≥1, z ≥0 we have that σ(z −log K) ≤2(z2 + K−1).

Proof. Recall the logistic function σ(z) = 1/(1 + e−x) and deﬁne the function g(z) = σ(z −
log K) −(z + K−1/2)2. We show that g(z) ≤0 for all z ≥0. First, notice that

g(0) = σ(−log K) −K−1 = (K + 1)−1 −K−1 ≤0.

Next, recall that σ′(x) = σ(x)σ(−x) and thus

g′(z) = σ(z −log K)σ(−z + log K) −2(z + K−1/2).

Examining z = 0 we further have that

g′(0) = σ(−log K)σ(log K) −2K−1/2

= (K + 1)−1(1 + K−1)−1 −2K−1/2

≤2[(K + 1)−1 −K−1/2] ≤0,

where the last two inequalities used K ≥1. Now, we have that

g′′(z) = σ(z −log K)σ(−z + log K)2 −σ(z −log K)2σ(−z + log K) −2 ≤0,

where the inequality is since σ(z) ∈[0, 1] for all z ∈R. Since g(0), g′(0) ≤0 and g′′(z) ≤0 for
all z ≥0, we conclude that g(z) ≤0 for all z ≥0. The proof is concluded using the AM-GM
inequality.
■

B.4
Concentration bounds

We give the following Bernstein type tail bound (see e.g., [Rosenberg et al., 2020, Lemma D.4].

Lemma 20. Let {Xt}t≥1 be a sequence of random variables with expectation adapted to a ﬁltration
Ft. Suppose that 0 ≤Xt ≤1 almost surely. Then with probability at least 1 −δ

T
X

t=1
E[Xt | Ft−1] ≤2

T
X

t=1
Xt + 4 log 2

δ

22


---Page Break---
We state the well-known self normalized error bounds for regularized least squares estimation of the
rewards and dynamics (see e.g., Abbasi-Yadkori et al. [2011], Jin et al. [2020b]).

Lemma 21 (reward error bound). Let bθk
h be as in line 14 of algorithm 1. With probability at least
1 −δ, for all k ≥1, h ∈[H]

∥θh −bθk
h∥Λk
h ≤2
p

2d log(2KH/δ).

Lemma 22 (dynamics error uniform convergence). Let bψk
h : RX →Rd be the linear operator
deﬁned in eq. (2) inside algorithm 1. For all h ∈[H], let Vh ⊆RX be a set of mappings V : X →R
such that ∥V ∥∞≤β and β ≥1. With probability at least 1 −δ, for all h ∈[H], V ∈Vh+1 and
k ≥1

∥(ψh −bψk
h)V ∥Λk
h ≤4β
p

d log(K + 1) + 2 log(HNϵ/δ),

where ϵ ≤β
√

d/2K, Nϵ = P

h∈[H] Nh,ϵ, and Nh,ϵ is the ϵ−covering number of Vh with respect to
the supremum distance.

B.5
Covering numbers

The following results are (mostly) standard bounds on the covering number of several function
classes.

Lemma 23. For any ϵ > 0, the ϵ-covering of the Euclidean ball in Rd with radius R ≥0 is upper
bounded by (1 + 2R/ϵ)d.

Lemma 24. Let V = {V (·; θ) : ∥θ∥≤W} denote a class of functions V : X →R. Suppose that
any V ∈V is L-Lipschitz with respect to θ and supremum distance, i.e.,

∥V (·; θ1) −V (·; θ2)∥∞≤L∥θ1 −θ2∥,
∥θ1∥, ∥θ2∥≤W.

Let Nϵ be the ϵ−covering number of V with respect to the supremum distance. Then

log Nϵ ≤d log(1 + 2WL/ϵ)

Proof. Let Θϵ/L be an (ϵ/L)-covering of the Euclidean ball in Rd with radius W. Deﬁne Vϵ =
{V (·; θ) : θ ∈Θϵ/L}. By lemma 23 we have that log|Vϵ| ≤d log(1 + 2WL/ϵ). We show that Vϵ
is an ϵ-cover of Vϵ, thus concluding the proof. Let V ∈V and θ be its associated parameter. Let
θ′ ∈Θϵ/L be the point in the cover nearest to θ and V ′ ∈V its associated function. Then we have
that

∥V (·) −V ′(·)∥∞= ∥V (·; θ) −V (·; θ′)∥∞≤L∥θ −θ′∥≤L(ϵ/L) = ϵ.
■

23


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?
Answer: [Yes]
Justiﬁcation: The abstract and introduction clearly state the accurate contribution of this
paper: a new algorithm for linear MDPs with new state-of-the-art regret guarantees. They
review existing algorithms and accurately describe the differences to our approach.
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
Justiﬁcation: The paper clearly states the setting in which the algorithm operates (linear
MDPs) and describes the assumptions made in Section 2.
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

24


---Page Break---
Answer: [Yes]

Justiﬁcation: Section 2 clearly describes the setup and assumptions. Our theoretical result in
Theorem 1 is fully proved in the appendix (proof sketch is found in the main text).

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

5. Open access to data and code

25


---Page Break---
Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justiﬁcation: The paper does not include experiments requiring code.
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
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)

26


---Page Break---
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

Justiﬁcation: We have reviewed the NeurIPS Code of Ethics and veriﬁed that our research
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

Answer: [NA]

Justiﬁcation: This is a theoretical paper that advances state-of-the-art theoretical guaran-
tees in general reinforcement learning settings. It is not directly related to any practical
application or practical algorithm.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

27


---Page Break---
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
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

28


---Page Break---
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
• We recognize that the procedures for this may vary signiﬁcantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

29


---Page Break---
