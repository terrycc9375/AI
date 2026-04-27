Improved Bayes Regret Bounds for Multi-Task
Hierarchical Bayesian Bandit Algorithms

Jiechao Guan1
Hui Xiong1,2,∗

1AI Thrust , The Hong Kong University of Science and Technology (Guangzhou), China
2Department of Computer Science and Engineering, HKUST, China
{jiechaoguan, xionghui}@hkust-gz.edu.cn

Abstract

Hierarchical Bayesian bandit refers to the multi-task bandit problem in which
bandit tasks are assumed to be drawn from the same distribution. In this work,
we provide improved Bayes regret bounds for hierarchical Bayesian bandit algo-
rithms in the multi-task linear bandit and semi-bandit settings. For the multi-task
linear bandit, we ﬁrst analyze the preexisting hierarchical Thompson sampling
(HierTS) algorithm, and improve its gap-independent Bayes regret bound from
O(m
p

n log n log (mn)) to O(m√n log n) in the case of inﬁnite action set, with
m being the number of tasks and n the number of iterations per task. In the case
of ﬁnite action set, we propose a novel hierarchical Bayesian bandit algorithm,
named hierarchical BayesUCB (HierBayesUCB), that achieves the logarithmic but
gap-dependent regret bound O(m log (mn) log n) under mild assumptions. All
of the above regret bounds hold in many variants of hierarchical Bayesian linear
bandit problem, including when the tasks are solved sequentially or concurrently.
Furthermore, we extend the aforementioned HierTS and HierBayesUCB algorithms
to the multi-task combinatorial semi-bandit setting. Concretely, our combinatorial
HierTS algorithm attains comparable Bayes regret bound O(m√n log n) with
respect to the latest one. Moreover, our combinatorial HierBayesUCB yields a
sharper Bayes regret bound O(m log (mn) log n). Experiments are conducted to
validate the soundness of our theoretical results for multi-task bandit algorithms.

1
Introduction

A stochastic bandit [26, 6, 27] is a sequential decision-making problem where at each round, an agent
has to choose an action, and receives a stochastic reward without knowing its expected value. The gap
between the cumulative reward of optimal actions in hindsight and the cumulative reward of agent
is deﬁned as regret. The goal is to minimize regret, through a combination of exploring different
actions and exploiting those with high rewards in the past. Typical applications of bandit algorithms
include news article recommendation [28], computational advertisement [20], and dynamic pricing
[24]. For example, in news article recommendation, the agent must choose a news article for a user.
The actions in this bandit setting are articles and the reward could be an indicator of a click from user.

When the agent has to solve multiple bandit tasks, many machine learning researchers resort to
multi-task learning/meta-learning paradigm [8, 34] to beneﬁt task adaptation. The existing works
focused on the multi-task bandit problem can be categorized into three main groups: (1) The ﬁrst
group attempts to learn a low-dimensional representation shared by different bandit tasks, to derive a
sharper cumulative regret bound than that derived by learning each task independently [19, 10]. (2)
The second group leverages the similarity of contexts (e.g. the feature of actions) in bandit tasks
to improve agent’s ability to predict rewards in a new task [14, 36]. (3) The third group chooses to

∗Corresponding Author

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
maintain a meta-distribution over the hyper-parameters of within-task bandit algorithms (like Tsallis-
INF [23], OFUL [9], and Thompson sampling [25, 7, 17]), and draws informative hyper-parameters
from the meta-distribution for efﬁcient regret minimization. Our work falls into the third group and
formulates the problem of learning similar bandit tasks in a hierarchical Bayesian bandit model [17].

Speciﬁcally, in hierarchical Bayesian bandit setting, each bandit task is characterized by a task pa-
rameter. Different bandit task parameters are assumed to be independently and identically distributed
according to the same distribution. At each round, the learning agent interacts with one or several
bandit tasks, which correspond to the sequential and concurrent bandit settings respectively. Many ex-
isting works considered hierarchical Bayesian bandit problem, and proposed Thompson sampling [33]
type algorithms to solve it [25, 7, 36]. The latest work [17] proposed hierarchical Thompson sampling
(HierTS) algorithm and developed a gap-independent Bayes regret bound O(m
p

n log n log (mn))
in the Gaussian linear bandit setting, where m is the number of bandit tasks and n the number of
iterations per task. However, it is still unclear for us whether we can derive sharper regret bounds or
how to extend hierarchical Bayesian bandit algorithms to the more general multi-task bandit setting.

In this work, we attempt to tackle the above two issues, by providing improved Bayes regret bounds
for hierarchical Bayesian bandit algorithms in the multi-task Gaussian linear bandit and semi-bandit
setting. Firstly, in the linear bandit setting, we improve the multi-task Bayes regret bound of HierTS
to O(m√n log n) in the case of inﬁnite action set, strengthening the latest bound in [17, Thm 3] by a
factor of O(
p

log (mn)). In the case of ﬁnite action set, we propose a novel hierarchical Bayesian
bandit algorithm, named hierarchical BayesUCB (HierBayesUCB), that achieves the logarithmic but
gap-dependent regret bound O(m log (mn) log n) under mild assumptions. All of the above regret
bounds for linear bandit hold in both the sequential and concurrent setting. Secondly, we extend the
aforementioned HierTS and HierBayesUCB algorithms to the multi-task Gaussian combinatorial
semi-bandit setting. Concretely, our combinatorial HierTS algorithm attains comparable Bayes regret
bound O(m√n log n) with respect to the latest one in [7, Thm 6]. Moreover, our combinatorial
HierBayesUCB yields a sharper but gap-dependent regret bound O(m log (mn) log n). Extensive
experiments in the Gaussian linear bandit setting are conducted to support our theoretical results.

Overall, our theoretical contributions are four-fold: (1) In the case of inﬁnite action set, we provide
a tighter Bayes regret bound O(m√n log n) for HierTS. This bound improves the latest result by
a factor of O(
p

log (mn)). (2) In the case of ﬁnite action set, we propose a novel HierBayesUCB
algorithm, and provide gap-dependent logarithmic Bayes regret bound O(m log (mn) log n) for it.
(3) We generalize the above regret bounds for linear bandit from sequential setting to the more
challenging concurrent setting. (4) We extend both HierTS and HierBayesUCB algorithms to the
more general multi-task combinatorial semi-bandit setting and derive improved Bayes regret bounds.

2
Related Work

Frequentist Regret Bounds for Stochastic Linear Bandit. In the frequentist stochastic bandit
setting, we do not assume the bandit task parameter is sampled from a ﬁxed distribution. The
frequentist regret is thus for any ﬁxed task parameter, without taking expectation over the distribution
of task parameter. (1) In the case of ﬁnite action set: [5] for the ﬁrst time investigated the stochastic
linear bandit problem and proposed an algorithm with a frequentist regret of O(
√

dn log3/2 n), where
d is the dimension of action space and n is the number of rounds. [29] developed a new algorithm
and improved the regret bound to O(√dn log n). [12] showed that the lower frequentist regret bound
in the ﬁnite action set is Ω(
√

dn). (2) In the case of inﬁnite action set: Both [13] and [30] proposed
algorithms that achieve O(d√n log3/2 n) regret. The regret bound was further improved in [1, 15] to
O(d√n log n), by designing novel linear bandit algorithms or utilizing advanced martingale methods.

Bayes Regret Bounds for Bayesian Linear Bandit. In the Bayesian stochastic bandit setting, the
Bayes regret is the expected cumulative regret whose expectation is taken over the draw of task
parameter from a distribution. It is not difﬁcult to see that the frequentist regret upper bound implies
a Bayes regret upper bound, because the former holds for any task parameter. (1) When the action set
is inﬁnite: [30] showed that in the Gaussian linear bandit setting, the Bayes regret of any Bayesian
bandit algorithm is lower bounded by Ω(√n). [31] for the ﬁrst time gave the Bayes regret bound
of O(d√n log n) for both Thompson sampling (TS) and BayesUCB [22] algorithms. Recently,
[21] provided an improved Bayes regret O(d√n log n) for TS algorithm with a concise proof. (2)
When the action set is ﬁnite: [32] derived a tight regret bound of O(
√

dn) for TS algorithm with

2


---Page Break---
sub-Gaussian reward noise, via a novel information-theoretic approach. Recently, [3] developed a
logarithmic Bayes regret bound O(d2 log2 n) for BayesUCB algorithm in the Gaussian bandit setting.

Frequentist Regret Bounds for Multi-Task Linear Bandit Problems. Under the representation
learning paradigm, the frequentist regret bounds in [19, 10] for multi-task linear bandits scales as
O(m
√

nk), where k is the dimension of low-dimensional representation. The expected frequentist
regret upper bound for multi-task adversarial linear bandit in [23] is O(m
p

n log (1 + nV )), with V
being the similarity among multiple adversarial bandit tasks. Nevertheless, we should mention that all
of these frequentist regret bounds for multi-task linear bandit problem are not tighter than Ω(m√n).

Bayes Regret Bounds for Multi-Task Bayesian Linear Bandit/Semi-Bandit. The most related
works to ours are [25, 7, 17], which provided hierarchical-type Thompson sampling algorithms for
multi-task bandit and derived Bayes regret bounds in the Gaussian reward setting. We list these
Bayes regret bounds in Table 1 for direct comparisons. Among them, the latest work [17] proposed
the HierTS algorithm and obtained its regret bound of O(m
p

n log n log (mn)), [7] derived the ﬁrst
Bayes regret bound for multi-task hierarchical Bayesian semi-bandit algorithm. In this work, we
provide for HierTS improved Bayes regret bound of O(m√n log n) in Theorem 5.1. We also propose
a novel HierBayesUCB algorithm that achieves logarithmic regret bound O(m log n log (nm)). We
ﬁnally extend HierTS and HierBayesUCB to the semi-bandit setting and derive improved theoretical
results. Other works utilized action features or structure information to derive Bayes regret bounds
for multi-task bandit [36, 37], e.g. the Bayes regret bound in [36] is O(m√n log n + m log2 (mn)).

Hierarchical Bayesian Bandit Algorithms. Hierarchical Bayesian bandit algorithm was ﬁrst pro-
posed by [25] to solve multi-task bandit problems. More hierarchical-type Thompson sampling
algorithms based on multi-task/meta learning frameworks were developed with improved theoretical
guarantees and empirical performance [7, 17, 36, 37]. There also existed other works investigating
hierarchical Bayesian bandit algorithms within the single-task bandit setting. For example, [16] ex-
tended the two-level hierarchical Bayesian bandit framework to the deeper multiple-level hierarchial
Bayesian bandit framework. [2] generalized the single-effect-parameter HierTS algorithm (i.e. the
action parameter is centered at a single latent variable) to the mixed-effect bandit framework where
each action is associated with a parameter that depends upon one or multiple effect parameters.

3
Problem Setting

For any positive integer n, denote [n] = {1, 2, ..., n} for brevity. For any square matrix M ∈Rd×d,
denote λ1(M), λd(M) as its maximum and minimum eigenvalues respectively, denote κ(M) =
λ1(M)/λd(M) as its condition number. The action set A ⊆B(0, B) ⊆Rd is assumed to be compact
for some positive constant B > 0, where B(0, B) is the closed ball centered at the origin. We use ⟨, ⟩
to denote the inner-product between vectors, use w(a) or wa to denote the a-th element of vector w.

Single-Task Bandit. A stochastic bandit problem is characterized by an unknown parameter θ
with an action set A. Each action a ∈A under the bandit instance θ is associated with a reward
distribution P(·|a, θ). The reward mean of action a under θ is denoted as r(a; θ) = EY ∼P(·|a;θ)[Y ],
and the optimal action under θ is denoted as A∗= arg maxa∈A r(a; θ). In the stochastic linear
bandit setting, the mean reward of action a ∈A is r(a, θ) = a⊤θ. In Bayesian bandit problem, we
further assume that the task parameter θ is independently and identically distributed (i.i.d.) according
to a task parameter distribution P(·|µ∗), which is characterized by an unknown hyper-parameter µ∗.

Single-Task Semi-Bandit. In the semi-bandit setting, the action set A = [K] is a set of ﬁnite items.
A = {A ⊆A : |A| ≤L} is a family of subsets of A with up to L items, where L ≤K. w ∈RK is
a weight vector. The weight of a set A ∈A is deﬁned as P

a∈Aw(a). We assume that the weights w
are drawn i.i.d. from a distribution, and the mean weight is denoted as ¯w=E[w]. Following previous
work [38], we focus on the coherent case [39] which assumes that the agent knows a feature matrix
Φ ∈RK×d, such that ¯w = Φθ, where θ is the task parameter drawn from P(·|µ∗). The reward of a
subset A ∈A under the bandit instance θ is deﬁned as r(A; θ) = P

a∈A(Φθ)(a) = P

a∈A⟨Φa, θ⟩,
where Φa is the transpose of the a-th row of matrix Φ. We further assume that ∥Φa∥≤B, ∀a ∈A .

Hierarchical Bayesian Multi-Task Bandit/Semi-Bandit. In this setting, the agent interacts with
m tasks sequentially or concurrently. First, sample the hyper-parameter µ∗from a hyper-prior Q.
Then, for each task s ∈[m], sample the task parameter θs,∗independently from distribution P(·|µ∗).
The learning process can be detailed as follows. At round t ≥1, the agent interacts with a set

3


---Page Break---
of tasks St ⊆[m], takes a series of actions At = (As,t)s∈St, and receives a series of rewards
Yt = (Ys,t)s∈St. In the bandit setting, Ys,t ∼P(·|As,t; θs,∗) is a stochastic reward obtained by taking
action As,t in task s ∈St; in the semi-bandit setting, Ys,t = { ˆws,t(a)}a∈As,t is a series of stochastic
rewards, where ˆws,t = ¯ws + ηs,t, ¯ws = Φθs,∗, and ηs,t is a K-dimensional random noise. The full
hierarchical Bayesian bandit/semi-bandit model in the m-task learning setting is exhibited as follow:
(1) µ∗∼Q; (2) θs,∗|µ∗∼P(·|µ∗), ∀s ∈[m]; (3) Ys,t|As,t, θs,∗∼P(·|As,t; θs,∗), ∀t ≥1, s ∈St.
Therefore, the goal of the agent in hierarchical Bayesian multi-task bandit/semi-bandit setting is to
interact with m tasks efﬁciently and minimize the following cumulative multi-task Bayes regret:

BR(m, n) = E
 X

t≥1

X

s∈St
r(As,∗; θs,∗) −r(As,t; θs,∗)

,
(1)

where As,∗= arg maxa∈A r(a; θs,∗) is the optimal action for task s ∈[m] in the bandit setting,
and As,∗∈arg maxA∈A r(A; θs,∗) is the optimal subset for task s ∈[m] in the semi-bandit setting.
The expectation is taken over µ∗, all task parameters (θs,∗)s∈[m], all actions (At)t≥1, all stochastic
rewards (Yt)t≥1. We further assume that the action set A is the same across different tasks for ease of
exposition, and assume that the learning agent interacts with any task s ∈[m] for at most n rounds for
convenient comparison with exiting regret upper bounds for multi-task bandit/semi-bandit problem.

4
Algorithm

Denote Hs,t=((As,ℓ, Ys,ℓ))ℓ<t,s∈St as the history of all interactions of agent with task s∈[m], and
Ht=(Hs,t)s∈[m] as the whole interaction history up to round t. We next introduce the speciﬁc form of
Hierarchical Thompson Sampling (HierTS) and Hierarchical BayesUCB (HierBayesUCB) algorithms
in the multi-task Bayesian linear bandit and semi-bandit settings, and instantiate these two algorithms
to the multi-task Gaussian linear bandit (Algorithm 1) and semi-bandit (Algorithm 2) problems.

4.1
Hierarchical Thompson Sampling and Hierarchical BayesUCB

At round t, hierarchical Bayesian bandit algorithm samples a hyper-parameter µt from the hyper-
posterior Qt deﬁned as Qt(µ) = P(µ∗= µ|Ht), and then interacts with tasks St ⊂[m]. Next, we
give details of bandit algorithms, and details of semi-bandit algorithms are deferred to Section 5.4.

Hierarchical Thompson Sampling. For any task s ∈St, HierTS samples task parameter θs,t
from the distribution Ps,t(θ|µt) ≜P(θs,∗= θ|µ∗= µt, Hs,t) and takes the action As,t =
arg maxa∈A a⊤θs,t, where Ps,t(θ|µt) is only conditioned on Hs,t due to the independence be-
tween task parameter θs,∗and other task histories. This process clearly samples bandit instance
θs,t from the true posterior P(θs,∗= θ|Ht), which is equivalent to the form:
R
P(θs,∗= θ, µ∗=
µ|Ht)dµ =
R
Ps,t(θ|µ)Qt(µ)dµ, where Ps,t(θ|µ) ∝Ls,t(θ)P(θ|µ) is the posterior probability,
Ls,t(θ)=Q

(a,y)∈Hs,tP(y|a; θ) is the likelihood function, P(θ|µ) is the prior probability by Bayes rule.

Hierarchical BayesUCB. For any task s ∈St in round t, HierBayesUCB computes the upper

conﬁdence bound Ut,s,a = a⊤ˆµs,t +
q

2 log 1

δ ∥a∥ˆΣs,t for any a ∈A, where ˆµs,t and ˆΣs,t are the
expectation and covariance of the distribution (i.e. P(θs,∗= θ|Ht)) of θs,∗conditioned on the history
Ht, and then takes action with the highest upper conﬁdence bound : As,t ←arg maxa∈A Ut,s,a.

4.2
Multi-Task Gaussian Linear Bandit and Semi-Bandit

The hierarchical Gaussian environment is generated as follow.
In the multi-task linear ban-
dit setting: (1) µ∗∼N(µq, Σq), (2) θs,∗|µ∗∼N(µ∗, Σ0), ∀s ∈[m], (3) Ys,t|As,t, θs,∗∼
N(A⊤
s,tθs,∗, σ2), ∀t ≥1, s ∈St; In the semi-bandit setting, the only difference lies in step (3)
where Ys,t,a|As,t, θs,∗∼N(⟨Φa, θs,∗⟩, σ2) for any a ∈As,t. Here, µq, µ∗, θs,∗are d-dimensional
vectors; Σq, Σ0 ∈Rd×d are positive semi-deﬁnite covariance matrices. In the above two settings, the
reward noise can be regarded as N(0, σ2). In the following theoretical analysis sections, we assume
that all of µq, Σq, Σ0 and σ are known by the agent to guarantee an analytically tractable posterior.

Concretely, using some basic algebraic computations in hierarchical Gaussian model (e.g. see [25,
Appendix D]), we can obtain the closed-form hyper-posterior in round t as Qt(µ) = N(µ; ¯µt, ¯Σt),
where the expectation ¯µt and the covariance matrix ¯Σt of Qt(µ) have the following explicit forms:

¯µt = ¯Σt
 
Σ−1
q µq +
X

s∈[m]
(Σ0 + G−1
s,t )−1G−1
s,t Bs,t

,
¯Σ−1
t
= Σ−1
q
+
X

s∈[m]
(Σ0 + G−1
s,t )−1. (2)

4


---Page Break---
Algorithm 1 Hierarchical Bayesian Algorithms
for Multi-Task Linear Bandit Setting

1: Input: Hyper-prior Q
2: Initialize Q1 ←Q
3: for t = 1, 2, . . . do
4:
Sample hyper-parameter µt ∼Qt
5:
Observe tasks St ⊆[m]
6:
for s ∈St do
7:
Option I (HierTS):
Compute Ps,t(θ | µt) ∝Ls,t(θ)P(θ | µt)
Sample task parameter θs,t∼Ps,t(· | µt)
Take action As,t ←arg maxa∈A a⊤θs,t
Option II (HierBayesUCB):

Set Ut,s,a =a⊤ˆµs,t +
q

2 log 1

δ ∥a∥ˆΣs,t,
for any a ∈A
Take action As,t ←arg maxa∈A Ut,s,a
8:
Observe reward Ys,t
9:
end for
10:
Update Qt+1
11: end for

Algorithm 2 Hierarchical Bayesian Algorithms
for Multi-Task Combinatorial Semi-Bandit Setting

1: Input: Hyper-prior Q, features Φ ∈RK×d

2: Initialize Q1 ←Q
3: for t = 1, 2, . . . do
4:
Sample hyper-parameter µt ∼Qt
5:
Observe tasks St ⊆[m]
6:
for s ∈St do
7:
Option I (HierTS):
Compute Ps,t(θ | µt) ∝Ls,t(θ)P(θ | µt)
Sample task parameter θs,t ∼Ps,t(· | µt)
Compute As,t=ORACLE(A,A ,Φθs,t)
Option II (HierBayesUCB):
Compute Ut,s(A) = P

a∈A(a⊤ˆµs,t +
q

2 log 1

δ ∥a∥ˆΣs,t), for all A ∈A
Compute As,t = arg maxA∈A Ut,s(A)
8:
ChoooseAs,tand observe{ˆws,t(a)}a∈As,t
9:
end for
10:
Update Qt+1
11: end for

Here, in the bandit setting Gs,t = σ−2 P
ℓ<t 1{s ∈Sℓ}As,ℓA⊤
s,ℓand Bs,t = σ−2 P
ℓ<t 1{s ∈
Sℓ}As,ℓYs,ℓ; in the semi-bandit setting Gs,t = σ−2 P
ℓ<t 1{s ∈Sℓ}(P
a∈As,ℓΦaΦ⊤
a ) and Bs,t =
σ−2 P

ℓ<t 1{s ∈Sℓ}(P

a∈As,ℓΦa ˆws,t(a)). After the hyper-parameter µt is sampled from Qt(µ),

we sample task parameter θs,t ∼N(θ; ˜µs,t, ˜Σs,t) for task s, where ˜µs,t = ˜Σs,t(Σ−1
0 µt + Bs,t) is the
posterior mean, ˜Σ−1
s,t = Σ−1
0
+ Gs,t the posterior covariance matrix. Such posterior of a linear model
is obtained with a Gaussian prior N(µt, Σ0) and Gaussian observations (Ys,ℓ)ℓ<t,s∈Sℓby Bayes rule.

On the other hand, we also need to handle P(θs,∗= θ|Ht). It is not difﬁcult to see that, in the
multi-task Gaussian linear bandit/semi-bandit setting, θs,∗|Ht is Gaussian and denoted as P(θs,∗=
θ|Ht)=N(θ; ˆµs,t, ˆΣs,t). According to Lemma B.1, ˆµs,t and ˆΣs,t have the following explicit forms:

ˆµs,t=˜Σs,t(Σ−1
0 ¯µt+Bs,t),
ˆΣs,t = ˜Σs,t + ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t.
(3)

5
Bayes Regret Bounds

In this section, we provide improved regret bounds of hierarchical Bayesian bandit algorithms for
multi-task Gaussian linear bandit/semi-bandit problem. Concretely, we provide improved analysis
for HierTS in the sequential linear bandit setting (Sections 5.1), propose a novel HierBayesUCB
bandit algorithm with logarithmic regret guarantee (Section 5.2), develop regret bounds for these
two algorithms in the concurrent linear bandit setting (Section 5.3), and ﬁnally extend these two
algorithms to the semi-bandit setting (Section 5.4) with improved regret bounds. In the proof for
our theoretical results, the most important step is to give an upper bound on the so-called posterior
variance Vm,n, which in the multi-task linear bandit setting is deﬁned and upper bounded as follow:

Vm,n ≜E
h X

t≥1

X

s∈St
∥As,t∥2
ˆΣs,t

i
≤O
 
md log (n

d ) + d log (m

d )

.
(4)

Although the above bound on Vm,n achieves the same order (w.r.t. m, n and d) as that in the latest
bound of [17, Sect B], our bound has a smaller multiplicative factor (see more details in Table 4). In the
multi-task semi-bandit setting, the posterior variance is Vm,n ≜E P

t≥1
P

s∈St
P

a∈As,t∥Φa∥2
ˆΣs,t
and can be bounded in a similar way. To ﬁnish the whole proof, our strategy consists of two main
steps: (1) The ﬁrst step is to transform the multi-task Bayes regret BR(m, n) into an intermediate
regret upper bound that involves the posterior variance Vm,n as the dominant term. (2) The second
step is to bound Vm,n with Eq. (4). Combining the results in steps (1) and (2) yields Bayes regret
bound for multi-task hierarchical Bayesian bandit/semi-bandit algorithms. Detailed comparisons
between our regret bounds and others in the bandit setting are shown in Table 1. Next, we deﬁne
c1 =σ2+B2λ1(Σ0), c2 = σ2+B2λ1(Σ0)+B2λ1(Σq)κ(Σ0) to be used through the whole Section 5.

5


---Page Break---
Table 1: Different Bayes regret bounds for multi-task d-dimensional linear (or K-armed) bandit
problem in the sequential setting. m is the number of tasks, n the number of iterations per task, A is
the action set. Bayes Regret Bound =Bound I + Bound II + Negligible Terms, where Bound I is
the regret bound for solving m tasks, Bound II the regret bound for learning hyper-parameter µ∗.

Bayes Regret Bound
|A|
Bound I
Bound II

[25, Theorem 3]
Finite
O
 
m√Kn log n


O
 
n2K
p

m log (n) log (K)


[7, Theorem 5]
Finite
O
 
m
p

dn(log n) log (n2|A|)


O
 p

dmn(log m) log (n|A|)


[17, Theorem 3]
Inﬁnite
O
 
md
p

n log ( n

d ) log (mn)


O
 
d
p

mn log (m) log (mn)


Our Theorem 5.1
Inﬁnite
O
 
md
p

n log ( n

d )


O
 
d
p

mn log ( m

d )


Our Theorem 5.2
Finite
O
 
md log ( n

d ) log (mn)


O
 
d log ( m

d ) log (mn)


5.1
Improved Regret Bound for HierTS in the Sequential Bandit Setting
In the sequential bandit setting, |St| = 1. Then, conditioned on Ht, it is not difﬁcult to see that in
Bayes regret, each term E[θ⊤
s,∗As,∗−θ⊤
s,∗As,t|Ht] = E

(θs,∗−ˆµs,t)⊤As,∗
Ht

, and we use a novel
Cauchy-Schwartz type inequality from [21, Prop 2] to bound E

(θs,∗−ˆµs,t)⊤As,∗
Ht

, leading

to BR(m, n) ≤E
hP

t≥1
P

s∈St

q

dE
 
(θs,∗−ˆµs,t)⊤As,t
2Ht
i
. Expand the expression in the

right hand side of the above inequality, we then have BR(m, n) ≤E P

t,s∈St

q

dA⊤
s,t ˆΣs,tAs,t ≤
p

dmnVm,n, reducing the Bayes regret bound to the posterior variance bound problem. Recalling
Eq. (4) achieves our ﬁrst improved Bayes regret upper bound in the sequential linear bandit setting.

Theorem 5.1 (Near-Optimal Sequential Regret) Let |St|= 1 for any round t. Then in the multi-task
Gaussian linear bandit setting, the Bayes regret upper bound of HierTS is as follow:

BR(m, n) ≤d
√

2mn

s

mc1 log (1 + n

d ) + c2 log (1 + m Tr(ΣqΣ−1
0 )
d
).

Our explanations for the above sequential regret bound are three-fold:
(1) The term
md
p

nc1 log (1 + n/d) represents the regret bound for solving m bandit tasks, whose parame-
ters θs,∗are drawn i.i.d. from the prior distribution N(µ∗, Σ0). Under this assumption, no task
provides information for any other task, and hence this bound is linear in m. Similar observation was

also pointed out by [25, 7, 17]. (2) The term d
q

mnc2 log
 
1+mTr(ΣqΣ−1
0 )/d

represents the regret
bound for learning the hyper-parameter µ∗. Such bound is sublinear in m and is not a dominant
term when m is large. (3) For a large m, the averaged Bayes regret bound across m tasks is of
BR(m, n)/m = O(d√n log n), and strengthens the latest averaged bound O(d√n log n) in [17,
Thm 3] by a factor √log n. Besides, since the lower Bayes regret bound for any Bayesian bandit
algorithm is Ω(d√n) [30], our task-averaged Bayes regret bound is within O(√log n) of optimality
and hence is called ‘Near-Optimal’ sequential regret bound. We further make a detailed comparison
between our regret bound in Theorem 5.1 and the regret bound [17, Thm 3] in the following remark.

Remark 5.1 (Improvements of Our Theorem 5.1 over the Latest One) Our sequential regret bound
has two improvements over the latest one in [17, Thm 3, shown in Table 1]: (1) We remove the
additional
p

log (mn) factor in both the regret bound for solving m bandit tasks and the regret bound
for learning the hyper-parameter µ∗. (2) In the regret bound for learning hyper-parameter µ∗, [17]
has a multiplicative factor κ2(Σ0), whereas our multiplicative factor is κ(Σ0). Such improvement is
achieved by using technical matrix analysis proposed in Lemma C.1. and explained in Remark A.1.

5.2
Logarithmic Regret Bound for HierBayesUCB in the Sequential Bandit Setting

In this section, we attempt to provide further improved Bayes regret bounds for hierarchical bandit
algorithms in the sequential bandit setting. Because the task averaged Bayes regret bound in
Theorem 5.1 is near optimal, it is not easy to derive improved Bayes regret bounds under the same
assumptions. Therefore, we further assume that the action set A is ﬁnite, and propose a novel

6


---Page Break---
hierarchical Bayesian bandit algorithm, named Hierarchical BayesUCB (HierBayesUCB), for multi-
task linear bandit problem. The pseudo-code of our proposed algorithm is shown in Algorithm 1.

Next, we introduce some necessary notations.
Let ∆s,t
= θ⊤
s,∗(As,∗−As,t), ∆s,min
=
mina∈A\{As,∗}
 
θ⊤
s,∗As,∗−θ⊤
s,∗a

, ∆min = mins∈[m] ∆s,min.
For any ϵ > 0, let ∆ϵ
min =

max{ϵ, ∆min}. Deﬁne the event Es,t = {∀a ∈A : |a⊤(θs,∗−ˆµs,t)|≤
q

2 log 1

δ ∥a∥ˆΣs,t}. Then,
analogous to [3], we decompose the Bayes regret BR(m, n) = EP

t≥1
P

s∈St∆s,t into three terms:
E P

t≥1,s∈St ∆s,t

1{∆s,t ≥ϵ, Es,t} + 1{∆s,t < ϵ, Es,t} + 1{ ¯Es,t}

. We can bound the last
two terms trivially with mn[ϵ + 2δ
 
maxt,s|∆s,t|

· |A|]. For the ﬁrst term, we use the fact that

As,t|Ht
i.i.d.
∼As,∗|Ht, as well as the Upper Conﬁdence Bound (UCB) technique to reduce it to an
intermediate upper bound
  P

t≥1,s∈St∥As,t∥2
ˆΣs,t log 1

δ

/ mins,t|∆s,t|. Combining the upper bound
over Vm,n in Eq. (4), HierBayesUCB can achieve the following logarithmic Bayes regret bound in
the sequential bandit setting (the logarithmic bound can be extended to the concurrent setting).

Theorem 5.2 (Logarithmic Sequential Regret of HierBayesUCB) Let |St| = 1 for any round t, and
the action set A is ﬁnite with |A| < ∞. Then in the multi-task Gaussian linear bandit setting, for any
δ ∈(0, 1), ϵ > 0, the Bayes regret BR(m, n) of HierBayesUCB is upper bounded by

mn
h
ϵ+4Bδλ

1
2
1 (Σ0+Σq)
 
d
1
2 +∥µq∥ˆΣ−1
s,1

|A|
i
+E[16d log 1

δ
∆ϵ
min
]
h
mc1 log (1+n

d )+c2 log (1+m Tr(ΣqΣ−1
0 )
d
)
i
.

We give more explanations for the above sequential regret in terms of the following ﬁve aspects:
(1) If let δ = 1/(mn), ϵ = 1/(mn) and ∆min >> ϵ, the above sequential regret bound is of
O
 
log (mn)(md log ( n

d ) + d log ( m

d ))

. The term O(md log (mn) log ( n

d )) represents the regret
bound for solving m bandit tasks and is linear in m. Such bound is sharper than the corresponding
bound O(md
p

n log ( n

d )) in our Theorem 5.1 by a multiplicative factor O(
p

log (n/d)/n log (mn)),
which is less than 1 especially when m ≤n. (2) The term O(d log (mn) log ( m

d )) represents
the regret bound for learning the hyper-parameter µ∗, and its contribution to the Bayes regret
bound can be negligible. Besides, this bound is sharper than the bound O(d
p

mn log (m/d))
in our Theorem 5.1. (3) The averaged Bayes regret bound across m tasks can be regarded as
BR(m, n)/m = O(d log (mn) log n), which is logarithmic in n. Therefore, we call our regret bound
as ‘Logarithmic’ sequential regret bound. Moreover, if there exists a ﬁxed positive integer i << n,
such that m ≤ni, then our task-averaged Bayes regret BR(m, n)/m = O(dE[
1
∆ϵ
min ] log2 n) matches
the latest single-task Bayes regret bound in [3, Thm 5] and is remarkably similar to the frequentist
regret O(d∆−1
min log2 n) in [1, Thm 5] . (4) We can obtain sharper bounds by setting δ, ϵ as different
values. For example, by setting δ = 1/n, our regret bound becomes O([mnϵ + m] + log n

∆ϵ
min m log n),

which is of order O(m log2 n) if we set ϵ = 1/(mn) and the gap ∆min >> ϵ is large. (5) We
also need to point out that, the Bayes regret bound in Theorem 5.2 scales with E[
1
∆ϵ
min ]. If the gap
∆min ≤1/(mn), then ∆ϵ
min = 1/(mn) and this may cause a large Bayes regret upper bound.

5.3 Improved Regret Bounds of HierTS and HierBayesUCB in the Concurrent Bandit Setting

In the concurrent bandit setting, there exists a positive integer L ≤m, such that 1 ≤|St| ≤L. The
concurrent bandit setting is thus more challenging than the sequential bandit setting, because the
agent in the concurrent setting needs to interact with multiple bandit tasks in parallel at each round
t ≥1, and the hyper-posterior Qt will not be updated until the end of round t. Therefore, we need to
make an additional assumption on the action space A as follow to facilitate our theoretical analysis.

Assumption 5.1 There exist actions {ai}d
i=1 ⊆A, a constant β >0, such that λd(Pd
i=1 aia⊤
i )≥β.

This assumption is also used in previous works [7, 17] for hierarchical Bayesian linear bandit. It
indicates that Pd
i=1 aia⊤
i is a positive deﬁnite matrix, and does not weaken the generality of our
theoretical results. Actually, if Rd is not spanned by actions in A, we can project A into a subspace
where the assumption holds. We also need to modify the HierTS algorithm to let the agent take the
basic actions {ai}d
i=1 for the ﬁrst d interactions in any task s ∈[m]. This modiﬁcation guarantees that
the agent explores all directions within the task. Such exploration is very similar to the initialization
method in UCB type K-arm bandit algorithms [6, 4], which choose to pull each arm in the ﬁrst K
rounds. Deﬁne c3 = 1 + B2σ−2κ(Σ0)

λ1(Σ0) + σ2/β

that will be used throughout the concurrent

7


---Page Break---
Table 2: Different Bayes regret bounds for multi-task semi-bandit problem. Bayes Regret Bound
=Bound I + Bound II + Negligible Terms. m is the number of tasks, n the number of iterations per
task, K the size of action set, L the number of pulled actions at each round (1 ≤L ≤K). Bound I
is the regret bound for solving m tasks, Bound II the regret bound for learning hyper-parameter µ∗.

Bayes Regret Bound
A
Bound I
Bound II

[7, Theorem 6]
[K]
O
 
m
p

nKL log n log (nK)


O
 p

mnKL log m log (nK)


Our Theorem 5.4
[K]
O
 
m
p

nL log (nL) log (nK)


O
 
L
3
2 p

mn log m log (nK)


Our Theorem 5.5
[K]
O
 
mL log (nL) log (mnK)


O
 
L3 log (m) log (mnK)


setting. Then, analogous to the proof for Theorem 5.1, we bound
p

mnVm,n with a more reﬁned
analysis, achieving the following improved Bayes regret bound for HierTS in the concurrent setting.

Theorem 5.3 Under Assumption 5.1, let 1 ≤|St|≤L for any round t ≥1. Then in the multi-task
Gaussian linear bandit setting, the Bayes regret BR(m, n) of HierTS is upper bounded by

2Bmd
q

λ1(Σ0 + Σq)(
√

d + ∥µq∥ˆΣ−1
s,1)+d√mn

s

2mc1 log (1+n

d )+2c2c3 log (1+m Tr(ΣqΣ−1
0 )
d
).

The concurrent regret bound in Theorem 5.3 achieves almost the same order (w.r.t. m, n, d) as the
sequential regret bound in Theorem 5.1, but differs in two aspects: (1) The bound for learning m i.i.d.
bandit tasks has an additional term Bmd
p

λ1(Σ0 + Σq)(
√

d + ∥µq∥ˆΣ−1
s,1). This is due to the fact

that we take the basic actions {ai}d
i=1 ﬁrst for each task s ∈[m] in the modiﬁed HierTS algorithm.
(2) The bound for learning the hyper-parameter µ∗has an additional multiplicative factor c3. This is
the price for deriving regret bounds in the concurrent setting. Nevertheless, when compared with the
latest concurrent regret bound in [17, Thm 4] for HierTS, our concurrent regret bound in Theorem 5.3
removes the
p

log (mn) factor in both the regret bound for learning m bandit tasks and the regret
bound for learning hyper-parameter µ∗. Detailed comparisons between different concurrent regret
bounds for multi-task linear bandit setting are listed in Table 3. Furthermore, utilizing the proof
strategy to demonstrate the logarithmic sequential regret for HierBayesUCB in our Theorem 5.2, we
can analogously develop a logarithmic concurrent regret upper bound for HierBayesUCB algorithm,
which is deferred to our Theorem C.2 in Appendix C due to the limited space of the main paper.

5.4
Improved Regret Bounds for HierTS and HierBayesUCB in the Semi-Bandit Setting

In this section, we extend the HierTS and HierBayesUCB algorithms to the multi-task Gaussian com-
binatorial semi-bandit setting. The pseudo-code of them is shown in Algorithm 2. Algorithm 2 is very
similar to Algorithm 1 (i.e. the multi-task linear bandit algorithms), except that the combinatorial Hi-
erTS in Algorithm 2 uses the approximation/randomized algorithm ORACLE to solve combinatorial
problem A∗∈arg maxA∈A
P

a∈A w(a) and denotes the solution as A∗= ORACLE(A, A , w).
We adopt the ORACLE operator as in the seminal works [11, 38] to guarantee the efﬁciency of com-
binatorial HierTS semi-bandit algorithm. In this section, we only consider the sequential semi-bandit
setting (i.e. |St| = 1) for ease of presentation, and our results can be extended to the concurrent
semi-bandit setting. Then, deﬁne c4 = σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0), we ﬁrst derive the
Bayes regret upper bound for combinatorial HierTS algorithm in the sequential semi-bandit setting.

Theorem 5.4 Let |St| = 1 for any t ≥1. Let c ≥
q

2 ln
  nKBλ1(Σ0)
√

2π

, then in the multi-task
Gaussian semi-bandit setting, the Bayes regret upper bound of combinatorial HierTS is:

BR(m, n) ≤m + c
√

mnL

s

2c1m log (1 + nL

d ) + 2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
).

Detailed comparisons between different Bayes regret bounds for multi-task semi-bandit problem
are listed in Table 2. We can see that, in our Theorem 5.4, both the regret bound O(m√n log n) for
learning m tasks and the regret bound O(√mn log m log n) for learning hyper-parameter µ∗can
achieve the same order (w.r.t. m and n) when compared with the latest bound in [7, Thm 6]. Besides,
our Bayes regret bound is logarithmic in the number K of items, whereas the Bayes regret bound in

8


---Page Break---
[7, Thm 6] is sublinear in K. Therefore, our regret bound becomes sharper when the size of action
set is very large, e.g. K >>L. Next, we derive a gap-dependent logarithmic multi-task Bayes regret
bound for our proposed combinatorial HierBayesUCB algorithm in the sequential semi-bandit setting.

Theorem 5.5 Let |St| = 1 for any t ≥1. Then for any ϵ > 0, δ ∈(0, 1), in the multi-task Gaussian
semi-bandit setting, the Bayes regret BR(m, n) of combinatorial HierBayesUCB is bounded by

mn

ϵ+4LBKδλ

1
2
1(Σ0+Σq)(d
1
2 +∥µq∥ˆΣ−1
s,1)

+E
8L log 1

δ
∆ϵ
min

h
2c1m log (1+nL

d )+2c4Ld log(1+m Tr(Σ−1
0 Σq)
d
)
i
.

In Theorem 5.5, if we set δ = 1/(mnK), ϵ = 1/(mn), and ∆min >> ϵ, then the regret bound
O
 
m log n log (mn)

for learning m tasks is logarithmic in n. Such bound is sharper than the latest
one O(m√n log n) in [7, Thm 6] for multi-task semi-bandit. The regret bound O(log m log (mn))
for learning hyper-parameter µ∗is also sharper than that of O(√mn log m log n) in [7, Thm 6].
Besides, since δ=1/(mnK), the whole Bayes regret bound is also logarithmic in the number K of
items. Nevertheless, we should point out that our bounds hold for the multi-task semi-bandits with
linear generalization, but [7] focuses on the multi-task K-arm semi-bandits without feature matrix Φ.

5.5
Technical Novelties for Deriving Improved Regret Bounds

In this section, we summarize our technical novelties in terms of the following three aspects:

(1) For the improved regret bound for HierTS in Theorem 5.1: our proof has three novelties: (i) We
apply a novel Cauchy-Schwartz type inequality in Lemma A.2 to bound E

(θs,∗−ˆµs,t)⊤As,∗
Ht

≤
q

dE
 
(θs,∗−ˆµs,t)⊤As,t
2Ht

, leading to a sharper bound without
p

log (mn) factor:

BR(m, n) ≤E
X

t,s∈St

q

dA⊤
s,t ˆΣs,tAs,t ≤
p

dmnVm,n ≤O(m
p

n log n).

(ii) We use a more technical positive semi-deﬁnite matrix decomposition analysis (i.e. our Lemma A.1)
to reduce the multiplicative factor κ2(Σ0) to κ(Σ0). (iii) Deﬁne a new matrix ˜Xs,t such that the
denominator in the regret is σ2 + B2λ1(Σ0), not just σ2, avoiding the case that the variance serves
alone as the denominator. Such technical novelties are also listed explicitly in Table 4.

(2) For the improved regret bound for HierBayesUCB in Theorem 5.2 in the sequential bandit setting:
our novelty lies in decomposing the Bayes regret BR(m, n) = E P

t≥1
P

s∈St ∆s,t into three terms:

E
X

t≥1

X

s∈St
∆s,t = E
X

t≥1,s∈St
∆s,t

1{∆s,t ≥ϵ, Es,t} + 1{∆s,t < ϵ, Es,t} + 1{ ¯Es,t}

,

and bounding the ﬁrst term with a new method as well as the property of BayesUCB algorithm as

E∆s,t1{∆s,t ≥ϵ, Es,t} = E∆2
s,t
∆s,t
1{∆s,t ≥ϵ, Es,t} ≤E
C2
t,s,As,t
∆ϵ
min
,

resulting in the ﬁnal improved gap-dependent regret bound for HierBayesUCB as follows
  X

t≥1,s∈St
∥As,t∥2
ˆΣs,t log 1

δ

/∆ϵ
min ≤O
 
m log (n) log 1

δ

δ=1/mn
======= O
 
m log (n) log (mn)

.

(3) For the improved regret bounds for HierTS and HierBayesUCB in the concurrent setting and in
the semi-bandit setting: besides the aforementioned technical novelties in (1) and (2), the additional
technical novelty lies in leveraging more reﬁned analysis (e.g. using Woodbury matrix identity) to
bound the gap between matrices ¯Σ−1
t+1 and ¯Σ−1
t
(more details is shown in Lemma C.1 and Eq. (6)).

6
Experiments

In this section, we conduct experiments in the linear bandit setting to verify our theoretical results.
Speciﬁcally, we show the inﬂuence of hyper-parameters (e.g. m, n, L) to the multi-task Bayes regret
of HierTS and HierBayesUCB, to validate the consistency between their regret bounds and practical
performance. Besides, we compare the performance between our algorithms and other baselines, to
show the effectiveness of hierarchical Bayesian bandit algorithms in the multi-task bandit setting.

9


---Page Break---
0
2
4
6
8
10
Number of Tasks m

0

50

100

150

200

Regret

Linear Bandit (d = 4, σq = 1.0, L=1)

n=100
n=200
n=300
n=400

(a) Regrets w.r.t. different m

2
4
6
8
10
Number of Concurrent Tasks L

0

100

200

300

400

Regret

Linear Bandit (m = 10, σq = 0.5)

d=2
d=4
d=8

(b) Regrets w.r.t. different L

0
100
200
300
400
Round t

0

250

500

750

1000

Regret

Linear Bandit (d = 4, m = 10, L=5)

σq=4
σq=3
σq=2
σq=1
σq=0.1

(c) Regrets w.r.t. different σq

0
100
200
300
400
Round t

0

200

400

Regret

Linear Bandit (d = 4, m = 10, L=5)

σ0=0.5
σ0=0.4
σ0=0.3
σ0=0.2
σ0=0.1

(d) Regrets w.r.t. different σ0

0
100
200
300
400
Round t

0

500

1000

Regret

Linear Bandit (d = 4, m = 10, L=5)

σ=10
σ=7
σ=4
σ=1
σ=0.1

(e) Regrets w.r.t. different σ

0
100
200
300
400
Round t

0

100

200

300

Regret

Linear Bandit (d = 4, σq = 1.0)

OracleTS
TS
HierTS
HierBayesUCB

(f) Regrets of different algorithms
Figure 1: Regrets of HierTS algorithm with respect to (w.r.t.) different hyper-parameters.

Experimental Setting. We follow the same experimental setting as that in [7, 17]. Concretely, we
conduct linear bandit experiments with Gaussian reward. The synthetic problem is deﬁned as follows.
In most experiments, we set the number of total tasks as m = 10, the dimension of action space as
d = 4, the number of concurrent tasks as L = 5, the number of rounds as n = 200m/L. We focus
on the ﬁnite action space with |A| = 10, and each action is sampled uniformly from [−0.5, 0.5]d. In
hierarchical Bayesian model, we set the hyper-prior as zero-mean isotropic Gaussian distribution
N(µq, Σq) = N(0, Σq), where Σq = σ2
qId; and set the task variance Σ0 = σ2
0Id. Unless otherwise
stated, we set σq = 1, σ0 = 0.1, σ2 = 0.5 for each task in most experiments. We exhibit the regret
performance of HierTS algorithm with respect to ﬁve hyper-parameters m, L, σq, σ0, σ in Figure 1
(a)-(e) respectively. The regret performance of HierBayesUCB is shown in Figure 2 of Appendix F.

Besides, we compare HierTS/HierBayesUCB with other two TS type algorithms that do not learn the
hyper-parameter µ∗in a hierarchical Bayesian model. The ﬁrst baseline is the vanilla TS algorithm
that samples task parameter θs,∗from the marginal prior N(µq, Σq + Σ0). The second baseline is an
idealized TS algorithm that knows µ∗exactly and uses the true prior N(µ∗, Σ0). We call the second
baseline as OracleTS, since this TS algorithm accesses more information of µ∗than HierTS and
vanilla TS algorithm. We show the regret performance of these four bandit algorithms in Figure 1 (f).

Experimental Results. From Figure 1, we can observe that: (1) In plot (a), the multi-task regret
becomes larger with the increase of m and n, which is consistent with our regret upper bound in
Theorems 5.1. (2) In plot (b), the regret increases with a higher dimension d. The number L of the
concurrent tasks seems do not have a large impact on regret. (3) In plots (c)-(e), the regret decreases
with a smaller variance (e.g. σq, σ0 and σ) in hierarchical Bayesian model, validating the provable
beneﬁts of variance-reduction in regret minimization, which is revealed in our multi-task Bayes regret
upper bounds. (4) The task-averaged regret of HierTS is tighter than that of single-task TS algorithm,
empirically demonstrating the advantages of multi-task Bayesian bandit optimization paradigm over
single-task bandit learning. (5) Our proposed HierBayesUCB achieves lower regret than HierTS.

7
Conclusions

This paper provides improved Bayes regret bounds for hierarchical Bayesian bandit algorithms in the
multi-task Gaussian linear bandit and semi-bandit setting. For linear bandit problem: in the case of
inﬁnite action set, we strengthen the preexisting regret bound O(m
p

n log n log (mn)) of HierTS
to O(m√n log n) by a factor of O(
p

log (mn)); in the case of ﬁnite action set, we propose a novel
HierBayesUCB algorithm that achieves logarithmic regret bound O(m log (mn) log n) under mild
conditions. Our regret bounds in the bandit setting hold when the agent solves tasks sequentially or
concurrently. Then, we extend the above HierTS and HierBayesUCB algorithms to the multi-task
semi-bandit setting and derive improved regret bounds. The synthetic experiments further support our
theoretical results. Our future work aims to extend our bounds to the sub-exponential bandit setting.

10


---Page Break---
Acknowledgments and Disclosure of Funding

Jiechao sincerely appreciates the ﬁnancial support from the People’s Government of Guangzhou
Municipality for his postdoctoral project. We thank all reviewers for their constructive suggestions
to improve the quality of this paper. This work was supported in part by the National Key R&D
Program of China (Grant No.2023YFF0725001), in part by the National Natural Science Foundation
of China (Grant No.92370204), in part by the guangdong Basic and Applied Basic Research Foun-
dation(Grant No.2023B1515120057), in part by Guangzhou-HKUST(GZ) Joint Funding Program
(Grant No.2023A03J0008), Education Bureau of Guangzhou Municipality.

References

[1] Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits.
In NeurIPS, pages 2312–2320, 2011.

[2] I. Aouali, B. Kveton, and S. Katariya. Mixed-effect thompson sampling. In AISTATS, pages
2087–2115, 2023.

[3] A. Atsidakou, B. Kveton, S. Katariya, C. Caramanis, and sujay sanghavi. Finite-time logarithmic
bayes regret upper bounds. In NeurIPS, 2023.

[4] J. Audibert and S. Bubeck.
Minimax policies for adversarial and stochastic bandits.
In
Conference on Learning Theory (COLT), 2009.

[5] P. Auer. Using conﬁdence bounds for exploitation-exploration trade-offs. Journal of Machine
Learning Research (JMLR), 3:397–422, 2002.

[6] P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem.
Machine Learning, 47(2-3):235–256, 2002.

[7] S. Basu, B. Kveton, M. Zaheer, and C. Szepesvári. No regrets for learning the prior in bandits.
In NeurIPS, pages 28029–28041, 2021.

[8] R. Caruana. Multitask learning. Machine Learning, 28(1):41–75, 1997.

[9] L. Cella, A. Lazaric, and M. Pontil. Meta-learning with stochastic linear bandits. In ICML,
pages 1360–1370, 2020.

[10] L. Cella, K. Lounici, G. Pacreau, and M. Pontil. Multi-task representation learning with
stochastic linear bandits. In AISTATS, pages 4822–4847, 2023.

[11] W. Chen, Y. Wang, and Y. Yuan. Combinatorial multi-armed bandit: General framework and
applications. In ICML, pages 151–159, 2013.

[12] W. Chu, L. Li, L. Reyzin, and R. E. Schapire. Contextual bandits with linear payoff functions.
In AISTATS, pages 208–214, 2011.

[13] V. Dani, T. P. Hayes, and S. M. Kakade. Stochastic linear optimization under bandit feedback.
In Conference on Learning Theory (COLT), pages 355–366, 2008.

[14] A. A. Deshmukh, Ü. Dogan, and C. Scott. Multi-task learning for contextual bandits. In
NeurIPS, pages 4848–4856, 2017.

[15] H. Flynn, D. Reeb, M. Kandemir, and J. Peters. Improved algorithms for stochastic linear
bandits using tail bounds for martingale mixtures. In NeurIPS, 2023.

[16] J. Hong, B. Kveton, S. Katariya, M. Zaheer, and M. Ghavamzadeh. Deep hierarchy in bandits.
In ICML, pages 8833–8851, 2022.

[17] J. Hong, B. Kveton, M. Zaheer, and M. Ghavamzadeh. Hierarchical bayesian bandits. In
AISTATS, pages 7724–7741, 2022.

[18] R. A. Horn and C. R. Johnson. Matrix Analysis. Cambridge University Press, 2012.

11


---Page Break---
[19] J. Hu, X. Chen, C. Jin, L. Li, and L. Wang. Near-optimal representation learning for linear
bandits and linear RL. In ICML, pages 4349–4358, 2021.

[20] S. Kale, L. Reyzin, and R. E. Schapire. Non-stochastic bandit slate problems. In NeurIPS,
pages 1054–1062, 2010.

[21] C. Kalkanli and A. Özgür. An improved regret bound for thompson sampling in the gaussian
linear bandit setting. In IEEE International Symposium on Information Theory (ISIT), pages
2783–2788, 2020.

[22] E. Kaufmann, O. Cappe, and A. Garivier. On bayesian upper conﬁdence bounds for bandit
problems. In AISTATS, pages 592–600, 2012.

[23] M. Khodak, I. Osadchiy, K. Harris, M. Balcan, K. Y. Levy, R. Meir, and Z. S. Wu. Meta-learning
adversarial bandit algorithms. In NeurIPS, 2023.

[24] R. D. Kleinberg and F. T. Leighton. The value of knowing a demand curve: Bounds on regret
for online posted-price auctions. In Symposium on Foundations of Computer Science (FOCS),
pages 594–605, 2003.

[25] B. Kveton, M. Konobeev, M. Zaheer, C. Hsu, M. Mladenov, C. Boutilier, and C. Szepesvári.
Meta-thompson sampling. In ICML, pages 5884–5893, 2021.

[26] T. L. Lai and H. Robbins. Asymptotically efﬁcient adaptive allocation rules. Advances in
Applied Mathematics, 6:4—-22, 1985.

[27] T. Lattimore and C. Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[28] L. Li, W. Chu, J. Langford, and R. E. Schapire. A contextual-bandit approach to personalized
news article recommendation. In International Conference on World Wide Web (WWW), pages
661–670, 2010.

[29] Y. Li, Y. Wang, and Y. Zhou. Nearly minimax-optimal regret for linearly parameterized bandits.
In Conference on Learning Theory (COLT), pages 2173–2174, 2019.

[30] P. Rusmevichientong and J. N. Tsitsiklis. Linearly parameterized bandits. Mathematics of
Operations Research, 35(2):395–411, 2010.

[31] D. Russo and B. V. Roy. Learning to optimize via posterior sampling. Mathematics of Operations
Research, 39(4):1221–1243, 2014.

[32] D. Russo and B. V. Roy. An information-theoretic analysis of thompson sampling. Journal of
Machine Learning Research (JMLR), 17:68:1–68:30, 2016.

[33] W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of
the evidence of two samples. Biometrika, 25:285–294, 1933.

[34] S. Thrun and L. Pratt. Learning to Learn. Kluwer Academic Publishers, 1998.

[35] M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge
University Press, 2019.

[36] R. Wan, L. Ge, and R. Song. Metadata-based multi-task bandits with bayesian hierarchical
models. In NeurIPS, pages 29655–29668, 2021.

[37] R. Wan, L. Ge, and R. Song. Towards scalable and robust structured bandits: A meta-learning
framework. In AISTATS, pages 1144–1173, 2023.

[38] Z. Wen, B. Kveton, and A. Ashkan. Efﬁcient learning in large-scale combinatorial semi-bandits.
In ICML, pages 1113–1122, 2015.

[39] Z. Wen and B. V. Roy. Efﬁcient exploration and value function generalization in deterministic
systems. In NeurIPS, pages 3021–3029, 2013.

12


---Page Break---
APPENDIX

A
Proofs for Regret Bound of HierTS in the Sequential Bandit Setting

We ﬁrst give the following proposition to bound the posterior variance E P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t in

the sequential setting. We choose to give the worst-case upper bound on P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t.

Proposition A.1 Let c1 = σ2 + B2λ1(Σ0), c2 = σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0), then

X

t≥1

X

s∈St
∥As,t∥2
ˆΣs,t ≤2mdc1 log
 
1 + n

d

+ 2dc2 log (1 + m Tr
 
Σ−1
0 Σq


d
).

Proof.
Note that ∥As,t∥2
ˆΣs,t
=
A⊤
s,t
 ˜Σs,t + ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t, then we bound
P

t≥1
P

s∈St∥As,t∥2
˜Σs,t and P

t≥1
P

s∈St A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t respectively.

(1) Bounding P

t≥1
P

s∈St∥As,t∥2
˜Σs,t. Note that ˜Σs,t = (Σ−1
0
+ Gs,t)−1 ≤Σ0, then we have

A⊤
s,t ˜Σs,tAs,t ≤B2λ1(Σ0) < B2λ1(Σ0) + σ2. Accordingly, we deﬁne a new matrix ˜Xs,t ≜(Σ−1
0
+
1
B2λ1(Σ0)+σ2
P

ℓ<t 1{s ∈St}As,ℓA⊤
s,ℓ)−1, and notice that ˜Xs,t ≥˜Σs,t = (Σ−1
0
+
1
σ2
P

ℓ<t 1{s ∈

St}As,ℓA⊤
s,ℓ)−1, and that ˜Xs,1 = Σ0. Then recall c1 = σ2 + B2λ1(Σ0), we have

X

t≥1

X

s∈St
A⊤
s,t ˜Σs,tAs,t

=c1
X

t≥1

X

s∈St

A⊤
s,t ˜Σs,tAs,t
σ2 + B2λ1(Σ0)

≤2c1
X

t≥1

X

s∈St
log
 
1 +
A⊤
s,t ˜Σs,tAs,t
σ2 + B2λ1(Σ0)


=2c1
X

t≥1
log
 
1 +
A⊤
St,t ˜ΣSt,tASt,t
σ2 + B2λ1(Σ0)


=2c1
X

t≥1

m
X

s=1
1{St = s} log
 
1 +
A⊤
s,t ˜Σs,tAs,t
σ2 + B2λ1(Σ0)


≤2c1
X

t≥1

m
X

s=1
1{St = s} log
 
1 +
A⊤
s,t ˜Xs,tAs,t
σ2 + B2λ1(Σ0)


=2c1

m
X

s=1

X

t≥1
1{St = s} log det
 
I +
˜X

1
2
s,tAs,tA⊤
s,t ˜X

1
2
s,t
σ2 + B2λ1(Σ0)


=2c1

m
X

s=1

X

t≥1
1{St = s}

log det
  ˜X−1
s,t +
As,tA⊤
s,t
σ2 + B2λ1(Σ0)

−log det ˜X−1
s,t


=2c1

m
X

s=1


log det
  ˜X−1
s,mn+1

−log det ˜X−1
s,1


=2c1

m
X

s=1
log det
 
I +
1
σ2 + B2λ1(Σ0)

X

t≤mn
1{s ∈St}Σ

1
2
0 As,tA⊤
s,tΣ

1
2
0


≤2dc1

m
X

s=1
log
Tr
 
I +
1
σ2+B2λ1(Σ0)
P

t≤mn 1{s ∈St}Σ

1
2
0 As,tA⊤
s,tΣ

1
2
0


d

13


---Page Break---
=2dc1

m
X

s=1
log
 
1 +

P

t≤mn 1{s ∈St}A⊤
s,tΣ0As,t
d(σ2 + B2λ1(Σ0))


≤2mdc1 log
 
1 + n

d

,

where the ﬁrst inequality holds because the basic inequality x ≤2 log (1 + x), ∀x ∈[0, 1]; the third

inequality holds due to the mean-value inequality (Qd
i=1 λi)
1
d ≤

Pd
i=1 λi

d
, for any λi ≥0; the last
inequality holds due to the fact that the agent interacts with each task s ∈[m] at most n times.

Before bounding the remaining P

t≥1
P

s∈St A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t, we introduce the follow-
ing lemma.

Lemma A.1 If the square matrices A > 0, B ≥0, then λ1
 
(I + AB)(I + BA)
−1
≤λ1(A)

λd(A).

Proof. According to Theorem 7.6.1 in Page 485 of [18], there exists a non-singular matrix S, such that
A = SS⊤, and B = S−⊤ΛS−1, in which Λ ≥0 is a diagonal matrix. Then we have AB = SΛS−1,
BA = S−⊤ΛS⊤. Therefore, applying Weyl’s inequality we have

λd
 
(I + AB)(I + BA)


=λd
 
(I + SΛS−1)(I + S−⊤ΛS⊤)


=λd
 
S(I + Λ)S−1S−⊤(I + Λ)S⊤

=λd
 
S⊤S(I + Λ)S−1S−⊤(I + Λ)


≥λd
 
S⊤S

λd
 
(I + Λ)S−1S−⊤(I + Λ)


≥λd
 
S⊤S

λd
 
S−1S−⊤
λd
 
(I + Λ)2

≥λd
 
S⊤S

/λ1
 
S1S⊤
= λd(A)/λ1(A).
□

Remark A.1 (Smaller multiplicative factor than the latest one in [17]) The improvement lies in
our sharper upper bound on λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 ), and detailed explanations are two-fold:
(1) Previous work [17, Appendix B] directly used Weyl’s inequality to upper bound

λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 ) ≤λ2
1(Σ−1
0 )λ2
1(˜Σs,t) ≤λ2
1(Σ−1
0 )λ2
1(Σ0) = κ2(Σ0).

(2) Instead of directly using Weyl’s inequality, we ﬁrst propose Lemma A.1 which uses positive
semi-deﬁnite matrix diagonalization technique to bound

λ1
 
(I + AB)(I + BA)
−1
≤λ1(A)

λd(A).

Then we apply Lemma A.1 to upper bound

λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 ) = λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 ) ≤λ1
 
(I + Σ0 ˜Σs,t)(I + ˜Σs,tΣ0)
−1
≤κ(Σ0),

resulting in a smaller multiplicative factor than that in [17].
□

(2) Bounding P

t≥1
P

s∈St A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t. First recall that

¯µt =¯Σt
 
Σ−1
q µq +
X

s∈[m]
Bs,t −Gs,t(Σ−1
0
+ Gs,t)−1Bs,t

= ¯Σt
 
Σ−1
q µq +
X

s∈[m]
(Σ0 + G−1
s,t )−1G−1
s,t Bs,t

,

¯Σ−1
t
=Σ−1
q
+
X

s∈[m]
Gs,t −Gs,t(Σ−1
0
+ Gs,t)−1Gs,t = Σ−1
q
+
X

s∈[m]
(Σ0 + G−1
s,t )−1.

Therefore ¯Σt ≤Σq. Then applying Lemma A.1 and Weyl’s inequality, we have

A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t ≤B2λ1(˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t) ≤B2λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 )λ1(¯Σt)

= B2λ1
 
(I + Σ0 ˜Σs,t)(I + ˜Σs,tΣ0)
−1
λ1(¯Σt) ≤B2 λ1(Σq)λ1(Σ0)

λd(Σ0)
≤B2 λ1(Σq)λ1(Σ0)

λd(Σ0)
+ B2λ1(Σ0) + σ2.

14


---Page Break---
Meanwhile, we estimate the gap between matrix ¯Σ−1
t+1 and matrix ¯Σ−1
t
as follow

¯Σ−1
t+1 −¯Σ−1
t
= (Σ0 + (Gs,t + σ−2As,tA⊤
s,t)−1)−1 −(Σ0 + G−1
s,t )−1

= Σ−1
0
−Σ−1
0 (˜Σ−1
s,t + σ−2As,tA⊤
s,t)−1Σ−1
0
−(Σ−1
0
−Σ−1
0 ˜Σs,tΣ−1
0 )

= Σ−1
0
˜Σs,t −(˜Σ−1
s,t + σ−2As,tA⊤
s,t)−1
Σ−1
0

= Σ−1
0 ˜Σ

1
2
s,t

I −(I + σ−2 ˜Σ

1
2
s,tAs,tA⊤
s,t ˜Σ

1
2
s,t)−1˜Σ

1
2
s,tΣ−1
0

= Σ−1
0 ˜Σ

1
2
s,t

I −(I −σ−2
˜Σ

1
2
s,tAs,tA⊤
s,t ˜Σ

1
2
s,t
1 + σ−2A⊤
s,t ˜Σs,tAs,t
)
˜Σ

1
2
s,tΣ−1
0

= Σ−1
0 ˜Σs,tAs,tA⊤
s,t ˜Σs,tΣ−1
0
σ2 + A⊤
s,t ˜Σs,tAs,t

≥
Σ−1
0 ˜Σs,tAs,tA⊤
s,t ˜Σs,tΣ−1
0
σ2 + B2λ1(Σ0) + B2λ1(Σq)λ1(Σ0)/λd(Σ0) ,
(5)

where the second equality holds due to the Woodbury matrix identity, and the ﬁfth equality
holds due to the Sherman-Morrison formula. Then analogous to the proof for (1) Bounding
P

t≥1
P

s∈St∥As,t∥2
˜Σs,t, recall c2 =

σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0)

we have

X

t≥1

X

s∈St
A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t

≤2c2
X

t≥1

X

s∈St
log
 
1 +
A⊤
s,t ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tAs,t
σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0)


=2c2
X

t≥1

X

s∈St
log det
 
I +
¯Σ

1
2
t Σ−1
0 ˜Σs,tAs,tA⊤
s,t ˜Σs,tΣ−1
0 ¯Σ

1
2
t
σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0)


=2c2
X

t≥1

X

s∈St


log det
 ¯Σ−1
t
+
Σ−1
0 ˜Σs,tAs,tA⊤
s,t ˜Σs,tΣ−1
0
σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0)

−log det
 ¯Σ−1
t


≤2c2
X

t≥1

X

s∈St


log det
 ¯Σ−1
t+1

−log det
 ¯Σ−1
t


≤2c2

log det
 ¯Σ−1
mn+1

−log det
 ¯Σ−1
1


=2c2 log det
 
I +
X

s∈[m]
Σ

1
2q (Σ0 + G−1
s,mn+1)−1Σ

1
2q


≤2dc2 log
Tr
 
I

+ Tr
  P

s∈[m] Σ

1
2q (Σ0 + G−1
s,mn+1)−1Σ

1
2q


d

≤2dc2 log (1 + m Tr
 
Σ−1
0 Σq


d
),

where the second inequality holds due to Eq. (5). Combining (1) and (2) ﬁnishes the whole proof. □

Remark A.2 Actually, we can replace the term Tr(Σ−1
0 Σq) in the above regret bound with
O(nλ1(Σq)), at the cost of a slightly larger regret upper bound, by bounding (Σ0 + G−1
s,mn+1)−1 ≤
Gs,mn+1 in the last but one step in the above (2), instead of bounding (Σ0 + G−1
s,mn+1)−1 ≤Σ−1
0 .
Speciﬁcally, we have the following estimation:

log
Tr(I + P
s∈[m] Σ

1
2q (Σ0 + G−1
s,mn+1)−1Σ

1
2q )

d



≤log
Tr(I + P

s∈[m] Σ

1
2q Gs,mn+1Σ

1
2q )

d



15


---Page Break---
= log

1 +
Tr(σ−2 P

s∈[m]
P

ℓ<mn+1 1[s ∈Sℓ]A⊤
s,ℓΣqAs,ℓ)

d



≤log

1 + σ−2mnλ1(Σq)

d


,

which is O(log (mn)), slightly larger than the regret bound of O(log (m)) in our Proposition A.1.

Then we can begin proving our ﬁrst Bayes regret bound for HierTS in the multi-task Gaussian linear
bandit setting. We ﬁrst give a lemma as follow, which is useful to prove our multi-task Bayes regret
bound in the sequential setting.

Lemma A.2 (Proposition 2 in [21]) Let X1 and X2 be arbitrary i.i.d. Rm valued random variables
and f1, f2 measurable maps such that f1, f2 : Rm →Rd with E∥f1(X1)∥2
2, E∥f2(X1)∥2
2 < ∞,
then |E[f1(X1)⊤f2(X1)]| ≤
p

dE[(f1(X1)⊤f2(X2))2].

Theorem A.1 (Theorem 5.1 in the main text). Let |St| = 1 for all rounds t ≥1. Then in the
multi-task Gaussian linear bandit setting, the Bayes regret upper bound of HierTS is as follow:

BR(m, n) ≤
√

mnd

s

2mdc1 log (1 + n

d ) + 2dc2 log (1 + m Tr(ΣqΣ−1
0 )
d
)

Proof. Recall that Ht = (Hs,ℓ)ℓ<t,s∈Sℓis the history up to round t, then

BR(m, n) = E
h X

t≥1

X

s∈St
E[θ⊤
s,∗As,∗−θ⊤
s,∗As,t|Ht]
i

= E
h X

t≥1

X

s∈St
E

θ⊤
s,∗As,∗−E[θs,∗|Ht]⊤E[As,t|Ht]
Ht
i

= E
h X

t≥1

X

s∈St
E

(θs,∗−ˆµs,t)⊤As,∗
Ht
i

≤E
h X

t≥1

X

s∈St

q

dE
 
(θs,∗−ˆµs,t)⊤As,t
2Ht
i

= E
h X

t≥1

X

s∈St

q

dE

A⊤
s,t(θs,∗−ˆµs,t)(θs,∗−ˆµs,t)⊤As,t
Ht
i

= E
h X

t≥1

X

s∈St

q

dE

A⊤
s,t ˆΣs,tAs,t
i

≤
√

mnd
s

E
X

t≥1

X

s∈St
∥As,t∥2
ˆΣs,t,

where both the second and the ﬁfth equality hold due to the independence between As,t and θs,∗
conditioned on Ht; the ﬁrst inequality holds by applying Lemma A.2 with functions f1(y1, y2) = y1,
f2(y1, y2) = y2 for any y1, y2 ∈Rd, and the random variable X1 = (θs,∗−ˆµs,t, As,∗)|Ht, X2 (with
the second element as As,t) is the i.i.d. copy of X1; the second inequality holds due to Jensen’s
inequality. Plugging the upper bound over E P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t in Proposition A.1 into the
above result obtains the Bayes regret bound for HierTS.
□

B
Proofs for Regret Bound of HierBayesUCB in the Sequential Bandit
Setting

Lemma B.1 Let θ | µ ∼N(µ, Σ0) and H = (xt, Yt)n
t=1 be n observations generated as Yt |
θ, xt ∼N(x⊤
t θ, σ2). Let P(µ | H) = N(µ; ¯µ, ¯Σ), and G = σ−2 Pn
t=1 xtx⊤
t . Then

E[θ | H] = (Σ−1
0
+ G)−1(Σ−1
0 ¯µ + B),
cov[θ | H] =(Σ−1
0
+ G)−1 + (Σ−1
0
+ G)−1Σ−1
0 ¯ΣΣ−1
0 (Σ−1
0
+ G)−1.

16


---Page Break---
Proof. By deﬁnition, we have cov[θ | µ, H] = (Σ−1
0
+ G)−1, E[θ | µ, H] = cov[θ | µ, H](Σ−1
0 µ +
B) where B = σ−2 Pn
t=1 xtYt. Hence cov[θ | µ, H] does not depend on µ. Then we have

E[θ | H] = E[E[θ | µ, H] | H] = cov[θ | µ, H](Σ−1
0 E[µ | H] + B) = (Σ−1
0
+ G)−1(Σ−1
0 ¯µ + B).

On the other hand, because cov[θ | µ, H] does not depend on µ, E[cov[θ | µ, H] | H] = cov[θ |
µ, H]. In addition, since B is a constant conditioned on H, then according to [17, Lemma 2], we
have the following result:

cov[E[θ | µ, H] | H] = cov[cov[θ | µ, H]Σ−1
0 µ | H] = (Σ−1
0
+ G)−1Σ−1
0 ¯ΣΣ−1
0 (Σ−1
0
+ G)−1.□

Theorem B.1 (Theorem 5.2 in the main text). Suppose the action set A is ﬁnite with |A| < ∞. Let
|St| = 1 for all rounds t ≥1. Then in the multi-task Gaussian linear bandit setting, the Bayes regret
upper bound of Hierarchical BayesUCB is as follow:

BR(m, n) ≤mnϵ + 4B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mn|A|δ

+ E
8 log 1

δ
∆ϵ
min

n
2mdc1 log
 
1 + n

d

+ 2dc2 log (1 + m Tr
 
Σ−1
0 Σq


d
)
o
.

In Theorem 5.2 in the main text, we replace
r

d +
q

8d ln 1

ζ in the right hand side of the above

inequality with
√

d for ease of exposition.
Proof. Deﬁne ∆s,t = θ⊤
s,∗As,∗−θ⊤
s,∗As,t, the event Es,t = {∀a ∈A : |a⊤(θs,∗−ˆµs,t)| ≤
q

2 log 1

δ ∥a∥ˆΣs,t}, and Ct,s,a =
q

2 log 1

δ ∥a∥ˆΣs,t. Then we can rewrite the multi-task Bayes regret
BR(m, n) as the following equivalent form:

E
 X

t≥1

X

s∈St
∆s,t

=
X

t≥1

X

s∈St
E

∆s,t1{∆s,t ≥ϵ, Es,t}

+
X

t≥1

X

s∈St
E

∆s,t1{∆s,t < ϵ, Es,t}

+
X

t≥1

X

s∈St
E

∆s,t1{ ¯Es,t}

.

Then we will bound the three terms in the RHS of the above equality respectively.
(1) Bounding P
t≥1
P
s∈St E

∆s,t1{∆s,t
≥
ϵ, Es,t}

.
Recall that Ut,s,a
=
a⊤ˆµs,t +
q

2 log 1

δ ∥a∥ˆΣs,t, then we have

X

t≥1

X

s∈St
E

∆s,t1{∆s,t ≥ϵ, Es,t}


=
X

t≥1

X

s∈St
E

E
 
θ⊤
s,∗As,∗−θ⊤
s,∗As,t
2

∆s,t
1{∆s,t ≥ϵ, Es,t}
Ht


≤
X

t≥1

X

s∈St
E

E
 
θ⊤
s,∗As,∗−Ut,s,As,∗+ Ut,s,As,t −θ⊤
s,∗As,t
2

∆s,t
1{∆s,t ≥ϵ, Es,t}
Ht


≤
X

t≥1

X

s∈St
E

E
 
Ut,s,As,t −θ⊤
s,∗As,t
2

∆s,t
1{∆s,t ≥ϵ, Es,t}
Ht


≤
X

t≥1

X

s∈St
E
4C2
t,s,As,t
∆ϵ
min



=
X

t≥1

X

s∈St
E
(8 log 1

δ )∥As,t∥2
ˆΣs,t
∆ϵ
min



≤E
8 log 1

δ
∆ϵ
min

n
2mdc1 log
 
1 + n

d

+ 2dc2 log (1 + m Tr
 
Σ−1
0 Σq


d
)
o
.

where the ﬁrst inequality holds due to the fact that Ut,s,As,t ≥Ut,s,As,∗in the BayesUCB algorithm;
the second inequality holds because when event Es,t occurs, θ⊤
s,∗As,∗≤Ut,s,As,∗the third inequality

17


---Page Break---
holds due to the deﬁnition of Ut,s,a; and the last inequality due to the result in Proposition A.1.
(2)
Bounding
P
t≥1
P
s∈St E

∆s,t1{∆s,t
<
ϵ, Es,t}

.
We
trivially
have
P

t≥1
P

s∈St E

∆s,t1{∆s,t < ϵ, Es,t}

≤mnϵ.

(3) Bounding P

t≥1
P

s∈St E

∆s,t1{ ¯Es,t}

.

First we give an upper bound of ∆s,t = θ⊤
s,∗(As,∗−As,t). Using Schwartz’s inequality, we have

θ⊤
s,∗(As,∗−As,t) ≤∥θs,∗∥ˆΣ−1
s,1∥As,∗−As,t∥ˆΣs,1

≤2B
q

λ1(ˆΣs,1)∥θs,∗∥ˆΣ−1
s,1 ≤2B
q

λ1(ˆΣs,1)

∥θs,∗−µq∥ˆΣ−1
s,1 + ∥µq∥ˆΣ−1
s,1


.

Besides, we also have θs,∗−µq = θs,∗−µ∗+ µ∗−µq ∼N(0, Σ0 + Σq) = N(0, ˆΣs,1), then

E

∥θs,∗−µq∥ˆΣ−1
s,1

≤
q

E∥ˆΣ
−1

2
s,1 (θs,∗−µq)∥2
2 =
√

d. According to [35, Exp 2.11], we have with

probability 1 −ζ, ∥θs,∗−µq∥ˆΣ−1
s,1 ≤
r

d +
q

8d ln 1

ζ . Therefore, with probability 1 −ζ over the

draw of {θs,∗}s∈[m], we have
X

t≥1

X

s∈St
E

∆s,t1{ ¯Es,t}


=
X

t≥1

X

s∈St
E

E

∆s,t1{ ¯Es,t}
Ht


≤2B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1
 X

t≥1

X

s∈St
E

1{ ¯Es,t}
Ht


=2B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1
 X

t≥1

X

s∈St
P
  ¯Es,t
Ht


≤4B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mn|A|δ,

where the last inequality holds because P
  ¯Es,t
Ht

≤2δ. Combining (1), (2) and (3), we achieve
the ﬁnal Bayes regret bound for any δ ∈(0, 1), ϵ > 0, ζ ∈(0, 1):

BR(m, n) ≤mnϵ + 4B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mn|A|δ

+ E
8 log 1

δ
∆ϵ
min

n
2mdc1 log
 
1 + n

d

+ 2dc2 log (1 + m Tr
 
Σ−1
0 Σq


d
)
o
.
□

C
Proofs for Regret Bounds of HierTS and HierBayesUCB in the
Concurrent Bandit Setting

Let Ct = {s ∈St : λd(Gs,t) ≥β/σ2} be the set of sufﬁciently-explored tasks at round t. We ﬁrst
give the following proposition to bound the posterior variance E P
t≥1
P
s∈St 1{s ∈Ct}∥As,t∥2
ˆΣs,t
in the concurrent setting. Analogous to the proof for Proposition A.1, we choose to give the worst-case
upper bound on P

t≥1
P

s∈St 1{s ∈Ct}∥As,t∥2
ˆΣs,t as follow.

Proposition C.1 Let c1 = σ2 + B2λ1(Σ0), c2 = σ2 + B2λ1(Σ0) + B2λ1(Σq)κ(Σ0), c3 = 1 +
B2σ−2κ(Σ0)

λ1(Σ0) + σ2/β

, then we have

X

t≥1

X

s∈St
1{s ∈Ct}∥As,t∥2
ˆΣs,t ≤2mdc1 log (1 + n

d ) + 2dc2c3 log (1 + m Tr
 
Σ−1
0 Σq


d
).

18


---Page Break---
Proof. Note that we have
X

t≥1

X

s∈St
1{s ∈Ct}∥As,t∥2
ˆΣs,t =
X

t≥1

X

s∈St
1{s ∈Ct}A⊤
s,t
 ˜Σs,t + ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t.

On event {s ∈Ct}, the modiﬁed HierTS samples from the posterior and actually behaves the same as
the original HierTS algorithm in Algorithm 1. Then we bound the two terms in the right hand side of
the above equality respectively .

(1) Bounding P

t≥1
P

s∈St 1{s ∈Ct}A⊤
s,t ˜Σs,tAs,t

Similar to the proof for Theorem A.1, we have A⊤
s,t ˜Σs,tAs,t ≤B2λ1(Σ0) + σ2 and ˜Xs,t ≜
(Σ−1
0
+
1
B2λ1(Σ0)+σ2
P

ℓ<t 1{s ∈St}As,ℓA⊤
s,ℓ)−1 ≥˜Σs,t. Then we can analogously obtain
X

t≥1

X

s∈St
1{s ∈Ct}A⊤
s,t ˜Σs,tAs,t

≤2

σ2 + B2λ1(Σ0)
 X

t≥1

X

s∈St
1{s ∈Ct} log
 
1 +
A⊤
s,t ˜Σs,tAs,t
σ2 + B2λ1(Σ0)


=2

σ2 + B2λ1(Σ0)
 X

t≥1

X

s∈St
1{s ∈Ct}

log det
  ˜X−1
s,t +
As,tA⊤
s,t
σ2 + B2λ1(Σ0)

−log det ˜X−1
s,t


≤2

σ2 + B2λ1(Σ0)
 X

t≥1

m
X

s=1
1{s ∈Ct}

log det
  ˜X−1
s,t +
As,tA⊤
s,t
σ2 + B2λ1(Σ0)

−log det ˜X−1
s,t


≤2

σ2 + B2λ1(Σ0)
 m
X

s=1

X

t≥1
1{s ∈St}

log det
  ˜X−1
s,t +
As,tA⊤
s,t
σ2 + B2λ1(Σ0)

−log det ˜X−1
s,t


=2

σ2 + B2λ1(Σ0)
 m
X

s=1


log det
  ˜X−1
s,mn+1

−log det ˜X−1
s,1


≤2d

σ2 + B2λ1(Σ0)
 m
X

s=1
log
 
1 +

P

t≥1 1{s ∈St}A⊤
s,tΣ0As,t
d(σ2 + B2λ1(Σ0))


≤2md

σ2 + B2λ1(Σ0)

log
 
1 + n

d

= 2mdc1 log
 
1 + n

d

,

where the second inequality holds due to the fact that, if square matrix A ≥B ≥0, then det (A) ≥
det (B), and |St| ≤m; the last inequality holds because the agent interact with each task at most n
times.

(2) Bounding P

t≥1
P

s∈St 1{s ∈Ct}A⊤
s,t
 ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t

Analysis. The real difference of the proof for the concurrent regret from the sequential regret lies in
bounding P

t≥1
P

s∈St 1{s ∈Ct}A⊤
s,t
 ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t, because |St| ≥1 and the result in
Eq. (5) (which only holds for the case |St| = 1) does not hold. To tackle this difference, we reduce the
concurrent setting to the sequential setting. Let St = {Itj}|St|
j=1 and deﬁne St,1:i = {Itj}i
j=1. Then at
round t, let s = Iti and deﬁne ¯Σ−1
s,t ≜Σ−1
q
+ P

z∈St,1:i−1(Σ0 + G−1
z,t+1)−1 + P

z∈[m]\St,1:i−1(Σ0 +
G−1
z,t)−1, we estimate the gap between ¯Σ−1
s,t and ¯Σ−1
t
as follow:
¯Σ−1
s,t −¯Σ−1
t
=
X

z∈St,1:i−1

h
(Σ0 + G−1
z,t+1)−1 −(Σ0 + G−1
z,t)
i

=
X

z∈St,1:i−1

h
(Σ0 + (Gz,t + Az,tA⊤
z,t
σ2
)−1)−1 −(Σ0 + G−1
z,t)
i

=
X

z∈St,1:i−1
Σ−1
0 ˜Σs,t
Az,tA⊤
z,t
σ2 + A⊤
z,t ˜Σz,tAz,t
˜Σs,tΣ−1
0 .

Thus we can bound λ1(¯Σ−1
s,t −¯Σ−1
t ) and tackle ¯Σ−1
s,t instead of ¯Σ−1
t
to reduce the concurrent setting
to the sequential setting. We ﬁrst give a useful lemma as follow.

19


---Page Break---
Lemma C.1 For any ﬁxed t ≥1 and i ∈[L], suppose λd(Gs,t) ≥β/σ2 and let s = It,i. Then

λ1(¯Σ−1
s,t ¯Σt) ≤1 + B2λ1(Σ0)

σ2λd(Σ0)

λ1(Σ0) + σ2

β

.

Proof. Applying Weyl’s inequality, we have

λ1
 
(¯Σ−1
s,t −¯Σ−1
t )¯Σt + I

= λ1
 ¯Σ

1
2
t (¯Σ−1
s,t −¯Σ−1
t )¯Σ

1
2
t + I


≤1 + λ1(¯Σ−1
s,t −¯Σ−1
t )λ1(¯Σt) = 1 + λ1(¯Σ−1
s,t −¯Σ−1
t )

λd(¯Σ−1
t )

We ﬁrst lower bound λd(¯Σ−1
t ). According to Weyl’s inequality, we have

λd(¯Σ−1
t ) ≥λd(Σ−1
q ) +
X

z∈[m]
λd
 
(Σ0 + G−1
z,t)−1
≥λd(Σ−1
q ) +
X

z∈[m]

1
λ1(Σ0) + λ1(G−1
z,t)

=λd(Σ−1
q ) +
X

z∈[m]

1
λ1(Σ0) +
1
λd(Gz,t)
≥λd(Σ−1
q ) +
i −1
λ1(Σ0) + σ2/β ,

where the last inequality holds because the tasks St,1:i−1 have been sufﬁciently explored. On the
other hand, using our Lemma A.1 we can bound λ1(¯Σ−1
s,t −¯Σ−1
t ) as follow

λ1(¯Σ−1
s,t −¯Σ−1
t ) ≤
X

z∈St,1:i−1
λ1(
Az,tA⊤
z,t
σ2 + A⊤
z,t ˜Σz,tAz,t
)λ1(Σ−1
0 ˜Σs,t ˜Σs,tΣ−1
0 ) ≤(i −1)B2

σ2
λ1(Σ0)
λd(Σ0).

Combining the above results, we have

λ1(¯Σ−1
s,t ¯Σt) ≤1 +
(i −1) B2

σ2
λ1(Σ0)
λd(Σ0)
λd(Σ−1
q ) +
i−1
λ1(Σ0)+σ2/β
≤1 + B2

σ2
λ1(Σ0)
λd(Σ0)

λ1(Σ0) + σ2

β

.
□

Then, recall c3 = 1 + B2σ−2κ(Σ0)

λ1(Σ0) + σ2/β

, we can bound P

t≥1
P

s∈St 1{s ∈
Ct}A⊤
s,t
 ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t as follows:
X

t≥1

X

s∈St
1{s ∈Ct}A⊤
s,t
 ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,t

As,t

=
X

t≥1

X

s∈St
1{s ∈Ct}A⊤
s,t ˜Σs,tΣ−1
0 ¯Σ

1
2
s,t
 ¯Σ
−1

2
s,t ¯Σt ¯Σ
−1

2
s,t
¯Σ

1
2
s,tΣ−1
0 ˜Σs,tAs,t

≤
X

t≥1

X

s∈St
1{s ∈Ct}λ1
 ¯Σ
−1

2
s,t ¯Σt ¯Σ
−1

2
s,t

A⊤
s,t ˜Σs,tΣ−1
0 ¯Σ

1
2
s,t ¯Σ

1
2
s,tΣ−1
0 ˜Σs,tAs,t

≤
n
1 + B2

σ2
λ1(Σ0)
λd(Σ0)

λ1(Σ0) + σ2

β
o
E
X

t≥1

X

s∈St
1{s ∈Ct}A⊤
s,t ˜Σs,tΣ−1
0 ¯Σs,tΣ−1
0 ˜Σs,tAs,t

≤2c3c2

log det
 ¯Σ−1
mn+1

−log det
 ¯Σ−1
1


≤2dc3c2 log (1 + m Tr
 
Σ−1
0 Σq


d
),

where the second inequality holds due to Lemma C.1, the third and the fourth inequality hold in the
same way as that in the proof of Bounding (2) in Proposition A.1. Combining the results in (1) and
(2) ﬁnishes the whole proof.
□

Remark C.1 In the last step of proof for Lemma C.1, we bound

λ1(¯Σ−1
s,t ¯Σt) ≤1 +
(i −1) B2

σ2
λ1(Σ0)
λd(Σ0)
λd(Σ−1
q ) +
i−1
λ1(Σ0)+σ2/β
≤1 +
(i −1) B2

σ2
λ1(Σ0)
λd(Σ0)
i−1
λ1(Σ0)+σ2/β
.

Thus our upper bound is independent of the number L of the concurrent tasks. Actually, ∀i ∈[L]:

λ1(¯Σ−1
s,t ¯Σt) ≤1 +

B2

σ2
λ1(Σ0)
λd(Σ0)

λd(Σ−1
q
)
(i−1)
+
1
λ1(Σ0)+σ2/β
≤1 +

B2

σ2
λ1(Σ0)
λd(Σ0)

λd(Σ−1
q
)
L
+
1
λ1(Σ0)+σ2/β
,

20


---Page Break---
the sharper bound 1 +

B2

σ2
λ1(Σ0)
λd(Σ0)

λd(Σ−1
q
)
L
+
1
λ1(Σ0)+σ2/β
is L-dependent. If
λd(Σ−1
q
)
L
<<
1
λ1(Σ0)+σ2/β , the

inﬂuence of L to the regret may be large; otherwise, the inﬂuence of L to the regret may be negligible.

Next, we prove the concurrent regret bound for HierTS and HierBayesUCB.

Theorem C.1 (Theorem 5.3 in the main text). Let |St| ≤L ≤m for all rounds t ≥1. Then in the
multi-task Gaussian linear bandit setting, the Bayes regret bound of HierTS is as follow:

BR(m, n) ≤2Bmd
q

λ1(Σ0 + Σq)(
√

d + ∥µq∥ˆΣ−1
s,1) + md
r

2nc1 log (1 + n

d )

+
√

mnd

s

2dc2c3 log (1 + m Tr
 
Σ−1
0 Σq


d
).

Proof. Recall that Ct = {s ∈St : λd(Gs,t) ≥β/σ2} is the set of sufﬁciently-explored tasks at round
t. Then, due to the modiﬁcation of HierTS algorithm, we decompose Bayes regret BR(m, n) into
two terms and bound them respectively:

BR(m, n) = E
X

t≥1

X

s∈St
1{s /∈Ct}θ⊤
s,∗(As,∗−As,t) + E
X

t≥1

X

s∈St
1{s ∈Ct}θ⊤
s,∗(As,∗−As,t)

(1) Bounding E P

t≥1
P

s∈St 1{s /∈Ct}θ⊤
s,∗(As,∗−As,t). Similar to the proof for Theorem B.1
(3), we have

θ⊤
s,∗(As,∗−As,t) ≤∥θs,∗∥ˆΣ−1
s,1∥As,∗−As,t∥ˆΣs,1 ≤2c
q

λ1(ˆΣs,1)

∥θs,∗−µq∥ˆΣ−1
s,1 + ∥µq∥ˆΣ−1
s,1


,

and E

∥θs,∗−µq∥ˆΣ−1
s,1

≤
q

E∥ˆΣ
−1

2
s,1 (θs,∗−µq)∥2
2=
√

d. Recalling the independence between θs,∗
and actions As,t yields

E
X

t≥1

X

s∈St
1{s /∈Ct}θ⊤
s,∗(As,∗−As,t)

≤2B
q

λ1(Σ0 + Σq)E
X

t≥1

X

s∈St
1{s /∈Ct}E

∥θs,∗−µq∥ˆΣ−1
s,1 + ∥µq∥ˆΣ−1
s,1



≤2B
q

λ1(Σ0 + Σq)(
√

d + ∥µq∥ˆΣ−1
s,1)md,

The last inequality holds because in the modiﬁed HierTS, event {s /∈Ct} occurs at most d times for
any task s ∈[m].

(2) Bounding E P

t≥1
P

s∈St 1{s ∈Ct}∥As,t∥2
ˆΣs,t. It sufﬁces to apply the upper bound in Proposi-
tion C.1.

Combining the upper bounds in steps (1) and (2) obtains the ﬁnal Bayes regret bound for HierTS in
the concurrent setting.
□

Theorem C.2 (Logarithmic Regret Bound for HierBayesUCB in the Concurrent Bandit Setting).
Suppose the action set A is ﬁnite with |A| < ∞. Let |St| ≤L ≤m for all rounds t ≥1. Then in the
multi-task Gaussian linear bandit setting, the Bayes regret bound of HierTS is as follow:

BR(m, n) ≤mnϵ + 4B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mn|A|δ

+ 2B
q

λ1(Σ0 + Σq)
 √

d + ∥µq∥ˆΣ−1
s,1

md + E
8 log 1

δ
∆ϵ
min

n
2mdc1 log (1 + n

d ) + 2dc3c2 log (1 + m Tr
 
Σ−1
0 Σq


d
)
o
.

Proof. Similar to the proof of Theorem C.1, we decompose the Bayes regret as BR(m, n) =
E P

t≥1
P

s∈St 1{s /∈Ct}θ⊤
s,∗(As,∗−As,t) + E P

t≥1
P

s∈St 1{s ∈Ct}θ⊤
s,∗(As,∗−As,t). Then

21


---Page Break---
Table 3: Different Bayes regret bounds for multi-task d-dimensional linear bandit problem in the
concurrent setting. m is the number of tasks, n is the number of iterations per task, A is the action
set. Bayes Regret Bound =Bound I + Bound II + Negligible Terms, where Bound I is the regret
bound for solving m tasks, Bound II the regret bound for learning hyper-parameter µ∗.

Bayes Regret Bound
|A|
Bound I
Bound II

[17, Theorem 4]
Inﬁnite
O
 
md
p

n log ( n

d ) log (mn)


O
 
d
p

mn log (m) log (mn)


Our Theorem 5.3
Inﬁnite
O
 
md
p

n log ( n

d )


O
 
d
p

mn log ( m

d )


Our Theorem C.2
Finite
O
 
md log ( n

d ) log (mn)


O
 
d log ( m

d ) log (mn)


we bound the ﬁrst term with the proof for Theorem C.1 (1), bound the second term with the proof for
our Theorem B.1. Then with probability 1 −ζ over the draw of {θs,∗}s∈[m],

BR(m, n) ≤2B
q

λ1(Σ0 + Σq)(
√

d + ∥µq∥ˆΣ−1
s,1)md + E[8 log 1

δ
∆ϵ
min
]
X

t≥1

X

s∈St
E

1{s ∈Ct}∥As,t∥2
ˆΣs,t


+ mnϵ + 4B
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mn|A|δ.

Plugging the upper bound on P

t≥1
P

s∈St E

1{s ∈Ct}∥As,t∥2
ˆΣs,t

in Proposition C.1 into the right
hand side of the above inequality ﬁnishes the whole proof.
□

D
Proofs for Regret Bounds of HierTS and HierBayesUCB in the
Semi-Bandit Setting

We also choose to give the worst-case upper bound on P

t≥1
P

s∈St
P

a∈As,t Φ⊤
a ˆΣs,tΦa as follow.

Proposition D.1 Let c1 = σ2 + B2λ1(Σ0), c4 = σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0), then

X

t≥1

X

s∈St

X

a∈As,t
Φ⊤
a ˆΣs,tΦa ≤2c1m log (1 + nL

d ) + 2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
).

Proof. Recall that Gs,t = σ−2 P

ℓ<t 1{s ∈Sℓ}(P

a∈As,t ΦaΦ⊤
a ), Bs,t = σ−2 P

ℓ<t 1{s ∈

Sℓ}(P

a∈As,t Φa ¯ws(a)), ˜Σ−1
s,t = Σ−1
0 +Gs,t, ¯Σ−1
t
= Σ−1
q +P

s∈[m](Σ0+G−1
s,t )−1. Then, analogous

to the proof for Proposition A.1, we introduce the matrix ˜Xs,t ≜
 
Σ−1
0
+
1
B2λ1(Σ0)+σ2
P

ℓ<t 1{s ∈

Sℓ}(P

a∈As,t ΦaΦ⊤
a )
−1.
We
next
bound
P

t≥1
P

s∈St
P

a∈As,t Φ⊤
a ˜Σs,tΦa
and
P

t≥1
P

s∈St
P

a∈As,t Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa.

(1) Bounding P

t≥1
P

s∈St
P

a∈As,t Φ⊤
a ˜Σs,tΦa.

X

t≥1

X

s∈St

X

a∈As,t
Φ⊤
a ˜Σs,tΦa

=[σ2 + B2λ1(Σ0)]
X

t≥1

m
X

s=1
1{St = s}
X

a∈As,t

Φ⊤
a ˜Σs,tΦa
σ2 + B2λ1(Σ0)

≤2[σ2 + B2λ1(Σ0)]
X

t≥1

m
X

s=1
1{St = s}
X

a∈As,t
log (1 +
Φ⊤
a ˜Σs,tΦa
σ2 + B2λ1(Σ0))

≤2[σ2 + B2λ1(Σ0)]
X

t≥1

m
X

s=1
1{St = s}
X

a∈As,t
log det(I +
˜Σ

1
2
s,tΦaΦ⊤
a ˜Σ

1
2
s,t
σ2 + B2λ1(Σ0))

22


---Page Break---
=2[σ2 + B2λ1(Σ0)]

m
X

s=1

X

t≥1

X

a∈As,t
1{St = s}

log det( ˜X−1
s,t +
ΦaΦ⊤
a
σ2 + B2λ1(Σ0)) −log det( ˜X−1
s,t )


=2[σ2 + B2λ1(Σ0)]

m
X

s=1


log det( ˜X−1
s,mn+1) −log det( ˜X−1
s,1)


=2[σ2 + B2λ1(Σ0)]

m
X

s=1
log det
 
I +
1
σ2 + B2λ1(Σ0)

X

t≤mn
1{St = s}
X

a∈As,t

ˆΣ

1
2
s,tΦaΦ⊤
a ˆΣ

1
2
s,t


=2[σ2 + B2λ1(Σ0)]

m
X

s=1
log
Tr
 
I +
1
σ2+B2λ1(Σ0)
P

t≤mn 1{St = s} P

a∈As,t ˆΣ

1
2
s,tΦaΦ⊤
a ˆΣ

1
2
s,t


d

=2[σ2 + B2λ1(Σ0)]

m
X

s=1
log
 
1 +

P
t≤mn 1{St = s} P
a∈As,t Φ⊤
a ˆΣs,tΦa
d(σ2 + B2λ1(Σ0))


≤2[σ2 + B2λ1(Σ0)]m log (1 + nL

d ) = 2c1m log (1 + nL

d ).

(2) Bounding P
t≥1
P
s∈St
P
a∈As,t Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa.

∀t ≥1, s ∈St, ∀a ∈As,t, we have Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa ≤B2λ1(Σq)κ(Σ0) + LB2λ1(Σ0) +

σ2. Meanwhile, deﬁne the matrix M ≜

˜Σ

1
2
s,tΦa1, ˜Σ

1
2
s,tΦa2, . . . , ˜Σ

1
2
s,tΦa|As,t|

∈Rd×|As,t|, we have
P

a∈As,t ˜Σ

1
2
s,tΦaΦ⊤
a ˜Σ

1
2
s,t = MM ⊤. Using the Wely’s inequality, we further have

λ1(I + σ−2M ⊤M) ≤λ1(I) + λ1(σ−2MM ⊤) ≤1 + σ−2 X

a∈As,t
λ1(˜Σ

1
2
s,tΦaΦ⊤
a ˜Σ

1
2
s,t) = 1 + σ−2 X

a∈As,t
Φ⊤
a ˜Σs,tΦa.

Then we can estimate the gap between matrix ¯Σ−1
t+1 and ¯Σ−1
t
as follow:

¯Σ−1
t+1 −¯Σ−1
t

=
 
Σ0 + (Gs,t + σ−2 X

a∈As,t
ΦaΦ⊤
a )−1−1 −
 
Σ0 + G−1
s,t
−1

=Σ−1
0
−Σ−1
0 (Σ−1
0
+ Gs,t + σ−2 X

a∈As,t
ΦaΦ⊤
a )−1Σ−1
0
−[Σ−1
0
−Σ−1
0 (Σ−1
0
+ Gs,t)−1Σ−1
0 ]

=Σ−1
0 [˜Σs,t −(˜Σ−1
s,t + σ−2 X

a∈As,t
ΦaΦ⊤
a )−1]Σ−1
0

=Σ−1
0 ˜Σ

1
2
s,t

I −(I + σ−2 X

a∈As,t

˜Σ

1
2
s,tΦaΦ⊤
a ˜Σ

1
2
s,t)−1˜Σ

1
2
s,tΣ−1
0

=Σ−1
0 ˜Σ

1
2
s,t

I −(I + σ−2MM ⊤)−1˜Σ

1
2
s,tΣ−1
0

=Σ−1
0 ˜Σ

1
2
s,t

σ−2M(I + σ−2M ⊤M)−1M ⊤˜Σ

1
2
s,tΣ−1
0

≥Σ−1
0 ˜Σ

1
2
s,t

σ−2Mλd
 
(I + σ−2M ⊤M)−1
M ⊤˜Σ

1
2
s,tΣ−1
0

=Σ−1
0 ˜Σ

1
2
s,t

σ−2
1
λ1(I + σ−2M ⊤M)MM ⊤˜Σ

1
2
s,tΣ−1
0

≥Σ−1
0 ˜Σ

1
2
s,t

σ−2
1
1 + σ−2 P

a∈As,t Φ⊤
a ˜Σs,tΦa
MM ⊤˜Σ

1
2
s,tΣ−1
0

=
Σ−1
0 ˜Σs,t
  P

a∈As,t ΦaΦ⊤
a
˜Σs,tΣ−1
0
σ2 + P

a∈As,t Φ⊤
a ˜Σs,tΦa

≥
Σ−1
0 ˜Σs,t
  P

a∈As,t ΦaΦ⊤
a
˜Σs,tΣ−1
0
σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0),
(6)

23


---Page Break---
where the second and the sixth equality hold due to the Woodbury matrix identity. The proof for Eq. (6)
is similar to the proof for Eq. (5), but requires more reﬁned analysis (i.e. the ﬁrst inequality in Eq. (6))
to estimate the lower bound of ¯Σ−1
t+1 −¯Σ−1
t . Then recall c4 = σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0)
for brevity, we can bound P

t≥1
P

s∈St
P

a∈As,t Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa as follow:
X

t≥1

X

s∈St

X

a∈As,t
Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa

≤2c4
X

t≥1

X

s∈St

X

a∈As,t
log (1 +
Φ⊤
a ˜Σs,tΣ−1
0 ¯ΣtΣ−1
0 ˜Σs,tΦa
σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0))

=2c4
X

t≥1

X

s∈St

X

a∈As,t
log det(I +
¯Σ

1
2
t Σ−1
0 ˜Σs,tΦaΦ⊤
a ˜Σs,tΣ−1
0 ¯Σ

1
2
t
σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0))

=2c4
X

t≥1

X

s∈St

X

a∈As,t


log det(¯Σ−1
t
+
Σ−1
0 ˜Σs,tΦaΦ⊤
a ˜Σs,tΣ−1
0
σ2 + B2Lλ1(Σ0) + B2λ1(Σq)κ(Σ0)) −log det(¯Σ−1
t )


≤2c4L
X

t≥1

X

s∈St


log det(¯Σ−1
t+1) −log det(¯Σ−1
t )


=2c4L

log det(¯Σ−1
mn+1) −log det(¯Σ−1
1 )


=2c4L

log det(I +
X

s∈[m]
Σ

1
2q (Σ0 + G−1
s,mn+1)−1Σ

1
2q )


≤2c4Ld

log
Tr(I + P

s∈[m] Σ

1
2q (Σ0 + G−1
s,mn+1)−1Σ

1
2q )

d


≤2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
),

where the second inequality holds due to Eq. (6).
□

Lemma D.1 If a Gaussian random variable X ∼N(µ, σ2), then E[X1{X ≥0}] = µ

1 −

ΦG(−µ

σ)

+
σ
√

2π exp{−µ2

2σ2 }. If further µ ≤0, then E[X1{X ≥0}] =
σ
√

2π exp{−µ2

2σ2 }.

Theorem D.1 (Theorem 5.4 in the main text, Regret Bound of HierTS in the Semi-Bandit Setting).

Let |St| = 1 for any t ≥1. Let c ≥
q

2 ln
  nKBλ1(Σ0)
√

2π

, c1 = σ2 + B2λ1(Σ0), c4 = σ2 +

B2Lλ1(Σ0)+B2λ1(Σq)κ(Σ0), then in the multi-task Gaussian semi-bandit setting, the Bayes regret
bound of combinatorial HierTS is:

BR(m, n) ≤m + c
√

mnL

s

2c1m log (1 + nL

d ) + 2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
).

Proof. Note that ¯ws = Φθs,∗, then deﬁne g(A, θ) = P
a∈A⟨Φa, θ⟩for brevity, we have the following
result:

BR(m, n) =E
X

t≥1

X

s∈St

 X

a∈As,∗
¯ws(a) −
X

a∈As,t
¯ws(a)


=E
X

t≥1

X

s∈St

 X

a∈As,∗
⟨Φa, θs,∗⟩−
X

a∈As,t
⟨Φa, θs,∗⟩


=E
X

t≥1

X

s∈St


g(As,∗, θs,∗) −g(As,t, θs,∗)

.

Deﬁne upper conﬁdence bound Ut,s(A) = P

a∈A

⟨Φa, ˆµs,t⟩+c
q

Φ⊤
a ˆΣs,tΦa

, where c is a constant

to be speciﬁed. Notice that As,∗|Ht
i.i.d.
∼As,t|Ht and Ut,s(·) is a deterministic function, thus

24


---Page Break---
E[Ut,s(As,∗)|Ht] = E[Ut,s(As,t)|Ht]. Then we can decompose Bayes regret BR(m, n) as follow:

BR(m, n) =E
X

t≥1

X

s∈St
E

g(As,∗, θs,∗) −Ut,s(As,∗) + Ut,s(As,t) −g(As,t, θs,∗)|Ht


=E
X

t≥1

X

s∈St


g(As,∗, θs,∗) −Ut,s(As,∗)

+ E
X

t≥1

X

s∈St


Ut,s(As,t) −g(As,t, θs,∗)

.

(1) Bounding E P

t≥1
P

s∈St

g(As,∗, θs,∗) −Ut,s(As,∗)


.

For any t ≥1, s ∈St, a ∈A, deﬁne random variable Xt,s,a = ⟨Φa, θs,∗−ˆµs,t⟩−c
q

Φ⊤
a ˆΣs,tΦa,

then we have Xt,s,a|Ht ∼N(−c
q

Φ⊤
a ˆΣs,tΦa, Φ⊤
a ˆΣs,tΦa) since E[θs,∗−ˆµs,t|Ht] = 0. Then

E
X

t≥1

X

s∈St


g(As,∗, θs,∗) −Us,t(As,∗)


=E
X

t≥1

X

s∈St

X

a∈As,∗
Xt,s,a

≤E
X

t≥1

X

s∈St

X

a∈As,∗
Xt,s,a1{Xt,s,a ≥0}

≤E
X

t≥1

X

s∈St

X

a∈[K]
Xt,s,a1{Xt,s,a ≥0}

=E
X

t≥1

X

s∈St

X

a∈[K]
E

Xt,s,a1{Xt,s,a ≥0}|Ht


≤E
X

t≥1

X

s∈St

X

a∈[K]

q

Φ⊤
a ˆΣs,tΦa
√

2π
exp{−c2

2 }

≤E
X

t≥1

X

s∈St

X

a∈[K]

Bλ1(ˆΣs,t)
√

2π
exp{−c2

2 }

≤nmK Bλ1(Σ0)
√

2π
exp{−c2

2 }.

If let nmK Bλ1(Σ0)
√

2π
exp{−c2

2 } ≤m, then c ≥
q

2 ln
  nKBλ1(Σ0)
√

2π

.

(2) Bounding E P
t≥1
P
s∈St

Ut,s(As,t) −g(As,t, θs,∗)


.

E
X

t≥1

X

s∈St


Ut,s(As,t) −g(As,t, θs,∗)


=E
X

t≥1

X

s∈St

X

a∈As,t
⟨Φa, ˆµs,t −θs,∗⟩+ c
q

Φ⊤
a ˆΣs,tΦa

=E
X

t≥1

X

s∈St

X

a∈[K]
E

1a∈As,t|Ht

E

⟨Φa, ˆµs,t −θs,∗⟩|Ht] + cE
X

t≥1

X

s∈St

X

a∈As,t

q

Φ⊤
a ˆΣs,tΦa

=cE
X

t≥1

X

s∈St

X

a∈As,t

q

Φ⊤
a ˆΣs,tΦa

≤c
√

mnL
s

E
X

t≥1

X

s∈St

X

a∈As,t
Φ⊤
a ˆΣs,tΦa,

where the second equality holds because of the mutual independence between As,t|Ht and θs,∗|Ht,
and E[ˆµs,t −θs,∗|Ht] = 0; the last inequality holds due to the Jensen inequality. Then applying

25


---Page Break---
Proposition D.1 to bound E P
t≥1
P
s∈St
P
a∈[As,t] Φ⊤
a ˆΣs,tΦa, we can obtain

E
X

t≥1

X

s∈St


Ut,s(As,t) −g(As,t, θs,∗)

≤c
√

mnL

s

2c1m log (1 + nL

d ) + 2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
).

Combining the above results ﬁnishes the whole proof.
□

Theorem D.2 (Theorem 5.5 in the main text, Regret Bound of HierBayesUCB in the Semi-Bandit
Setting). Let |St| = 1 for all rounds t ≥1. Let c1 = σ2 + B2λ1(Σ0), c4 = σ2 + B2Lλ1(Σ0) +
B2λ1(Σq)κ(Σ0), Then for any ϵ > 0, δ ∈(0, 1), ζ ∈(0, 1), in the multi-task Gaussian semi-bandit
setting, the Bayes regret upper bound of combinatorial HierBayesUCB is as follow:

BR(m, n) ≤E
8L log 1

δ
∆ϵ
min


2c1m log (1 + nL

d ) + 2c4Ld log(1 + m Tr(Σ−1
0 Σq)
d
)

+ mnϵ

+ 4LB
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mnKδ.

In Theorem 5.5 in the main text, we replace
r

d +
q

8d ln 1

ζ in the right hand side of the above

inequality with
√

d for ease of exposition.

Proof. Deﬁne the event Es,t = {∀a ∈A : |Φ⊤
a (θs,∗−ˆµs,t)| ≤
q

2 log 1

δ ∥Φa∥ˆΣs,t}, and the upper

conﬁdence bound Ut,s(A) = P
a∈A⟨Φa, ˆµs,t⟩+
q

2 log 1

δ ∥Φa∥ˆΣs,t. Let ∆s,t = g(As,∗, θs,∗) −
g(As,t, θs,∗), then we decompose the Bayes regret into three parts as follow:

E
X

t≥1

X

s∈St
∆s,t

=
X

t≥1

X

s∈St
E

∆s,t1{∆s,t ≥ϵ, Es,t}

+
X

t≥1

X

s∈St
E

∆s,t1{∆s,t < ϵ, Es,t}

+
X

t≥1

X

s∈St
E

∆s,t1{ ¯Es,t}

.

(1) Bounding P
t≥1
P
s∈St E

∆s,t1{∆s,t ≥ϵ, Es,t}

.

X

t≥1

X

s∈St
E

∆s,t1{∆s,t ≥ϵ, Es,t}


=
X

t≥1

X

s∈St
E
 
g(As,∗, θs,∗) −g(As,t, θs,∗)
2

∆s,t
1{∆s,t ≥ϵ, Es,t}


≤
X

t≥1

X

s∈St
E
 
g(As,∗, θs,∗) −Ut,s(As,∗) + Ut,s(As,t) −g(As,t, θs,∗)
2

∆s,t
1{∆s,t ≥ϵ, Es,t}


≤
X

t≥1

X

s∈St
E
 
Ut,s(As,t) −g(As,t, θs,∗)
2

∆s,t
1{∆s,t ≥ϵ, Es,t}


=
X

t≥1

X

s∈St
E

  P

a∈As,t⟨Φa, ˆµs,t −θs,∗⟩+
q

2 log 1

δ ∥Φa∥ˆΣs,t
2

∆s,t
1{∆s,t ≥ϵ, Es,t}


≤
X

t≥1

X

s∈St
E

  P
a∈As,t 2
q

2 log 1

δ ∥Φa∥ˆΣs,t
2

∆s,t
1{∆s,t ≥ϵ, Es,t}


≤
X

t≥1

X

s∈St
E

  P

a∈As,t 8 log 1

δ
  P

a∈As,t∥Φa∥2
ˆΣs,t


∆s,t
1{∆s,t ≥ϵ, Es,t}


26


---Page Break---
≤E
8L log 1

δ
∆ϵ
min

X

t≥1

X

s∈St

X

a∈As,t
∥Φa∥2
ˆΣs,t

,

where the ﬁrst and the second inequality hold due to the deﬁnition of the upper conﬁdence bound
Ut,s(As,t), the fourth inequality holds due to the Cauchy-Schwartz inequality. Utilizing the upper
bound on P

t≥1
P

s∈St
P

a∈As,t∥Φa∥2
ˆΣs,t in Proposition D.1 completes the proof for the ﬁrst part.

(2) Bounding P

t≥1
P

s∈St E

∆s,t1{∆s,t < ϵ, Es,t}

.
We trivially have P

t≥1
P

s∈St E

∆s,t1{∆s,t < ϵ, Es,t}

≤mnϵ.

(3) Bounding P
t≥1
P
s∈St E

∆s,t1{ ¯Es,t}

.

Note that θs,∗−µq ∼N(0, ˆΣs,1), and E

∥θs,∗−µq∥ˆΣ−1
s,1

≤
q

E∥ˆΣ
−1

2
s,1 (θs,∗−µq)∥2
2 =
√

d. Then

according to [35, Exp 2.11], we have with probability 1 −ζ, ∥θs,∗−µq∥ˆΣ−1
s,1 ≤
r

d +
q

8d ln 1

ζ
Therefore, with probability 1 −ζ over the draw of {θs,∗}s∈[m], we have

∆s,t =g(As,∗, θs,∗) −g(As,t, θs,∗)

=
X

a∈As,∗
⟨Φa, θs,∗⟩−
X

a∈As,t
⟨Φa, θs,∗⟩

≤
X

a∈As,∗
∥Φa∥· ∥θs,∗∥+
X

a∈As,t
∥Φa∥· ∥θs,∗∥

≤2LB∥θs,∗∥

1 −ζ
≤2LB
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

,

where the ﬁrst inequality holds due to the Schwartz inequality. Then we have with probability 1 −ζ:
X

t≥1

X

s∈St
E

∆s,t1{ ¯Es,t}


=
X

t≥1

X

s∈St
E

E

∆s,t1{ ¯Es,t}
Ht


≤2LB
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1
 X

t≥1

X

s∈St
E

1{ ¯Es,t}
Ht


= 2LB
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1
 X

t≥1

X

s∈St
P
  ¯Es,t
Ht


≤4LB
q

λ1(Σ0 + Σq)
 
s

d +
r

8d ln 1

ζ + ∥µq∥ˆΣ−1
s,1

mnKδ.

Combining the results in (1) (2) and (3) ﬁnishes the whole proof.
□

E
Technical Overview and Limitations of this Work

In this section, we explain our technical novelties for deriving near-optimal sequential regret bound
for HierTS and logarithmic sequential regret bound for HierBayesUCB as follow, when compared
with the latest bound in [17] (More detailed explanations can be found in Table 4):
(1) The Technical Overview for Deriving Near-Optimal Regret Bound in Theorem 5.1. The
biggest novelty lies in bounding each term E

(θs,∗−ˆµs,t)⊤As,∗
Ht

in Bayes regret BR(m, n).
Existing work [17] chose Cauchy-Schwartz inequality to directly bound (θs,∗−ˆµs,t)⊤As,∗≤
∥θs,∗−ˆµs,t∥ˆΣ−1
s,t∥As,∗∥ˆΣs,t, used UCB technique to bound ∥θs,∗−ˆµs,t∥ˆΣ−1
s,t (which caused an

additional multiplicative factor log 1

δ), leveraged the fact that As,∗|Ht
i.i.d.
∼As,t|Ht to transform

27


---Page Break---
P
t,s∥As,∗∥ˆΣs,t into Vm,n, and obtained an intermediate regret upper bound
p

mnVm,n log (1/δ).
Instead of using UCB technique, our Theorem 5.1 applies a novel Cauchy-Schwartz type inequality

(i.e. Lemma A.2) to bound E(θs,∗−ˆµs,t)⊤As,∗≤
q

dE
 
(θs,∗−ˆµs,t)⊤As,t
2 ≈
√

d∥As,t∥ˆΣs,t,

and ﬁnally achieves the regret bound
p

mnVm,n, removing the
p

log (1/δ) factor. Besides, when
bounding the posterior variance Vm,n, we use a different matrix analysis to prevent variance terms
(e.g. σ2, λ1(Σ0)) solely appearing in the denominator of regret bound. Moreover, we employ a
matrix decomposition technique (in our Lemma A.1) to reduce the multiplicative factor κ2(Σ0) in
[17, Theorems 3-4] to κ(Σ0) in our bounds (see more details in Table 4).
(2) The Technical Overview for Deriving Logarithmic Regret Bound in Theorem 5.2. To obtain
sharper sequential regret bound than the near-optimal regret bound in Theorem 5.1, our Theorem 5.2
chooses the Bayes regret decomposition strategy shown above Theorem 5.2, uses UCB technique
to bound the ﬁrst term in the regret decomposition as P

t,s E∆s,t1{∆s,t ≥ϵ,Es,t} ≤Vm,n log 1

δ,
and ﬁnally combines the upper bound on posterior variance Vm,n in Eq. (4) to achieve a logarithmic
Bayes regret upper bound of (log 1

δ )md log n

d .

Nevertheless, we also need to point out the limitations of our Bayes regret bounds:
The Limitations of the Multi-Task Bayes Regret Bounds. Honestly speaking, our regret bounds
have two main limitations, because they are: (i) Not advantageous when compared with single-task
regret bound. This is because our Bayes regret bounds (e.g. O(m√n log n) in Theorem 5.1) for
hierarchical Bayesian bandit problem are almost the same as the summation of regret bounds of
learning m Bayesian bandit task independently. This is also the limitation of existing bounds in
this ﬁeld (see [25, 7, 17]). (ii) Unable to shed more light on the advantages of multi-task bandit
optimization. The existing regret bound O(m
√

nk) for multi-task representation which demonstrated
that multi-task regret bound can be smaller for learning a low-dimensional representation (i.e.
k << d) than the regret bound of O(m
√

nd) for learning each task independently, and the existing
regret bound O(m
p

n log (1 + nV )) for multi-task adversarial linear bandit which proved that the
regret bound decreases with more similarity (i.e. smaller V ) among bandit tasks. Our hierarchical
Bayesian bandit model has assumed that different bandit instances are sampled the same meta-
distribution, and hence fails to reveal the inﬂuence of task similarity to the multi-task Bayes regret.

Remark E.1 (The Underlying Causes for the Limitation of Multi-Task Bayes Regret Bound.)
The underlying causes for the shortcoming of the multi-task Bayes regret bound is that the upper bound
on the posterior variance Vm,n = E P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t may be not tight enough. Detailed
explanations lie in the following three aspects:

(1) Recall that in the proof for our Theorem 5.1, we can upper bound the multi-task Bayes regret as
BR(m, n) ≤
√

mnd
p

Vm,n. Then in Proposition A.1 we use a purely algebraic technique to bound
the posterior-variance Vm,n ≤O(m log n), resulting in the ﬁnal Bayes regret bound of
√

mnd
p

Vm,n = O(
√

mnd
p

m log n) = O(m
p

n log n),

which is almost the same as the summation of the regret bounds for learning m bandit tasks indepen-
dently. Therefore, if we can upper bound the posterior-variance Vm,n with a bound that is sublinear
with respect to m and logarithmic w.r.t. n (e.g. a bound of O(√m log n)), then the ﬁnal Bayes regret
bound will be much sharper. The upper bounds on the posterior-variance Vm,n in existing works (e.g.
see [25, 7, 17] in our Table 1 in the main text) are also obtained via purely algebraic techniques and
are not sharp either (or even worse).

(2) In the proof for Proposition A.1,
we only give the worst-case upper bound on
P
t≥1
P
s∈St∥As,t∥2
ˆΣs,t via purely algebraic technique (thus leading to a worst-case upper bound

on the posterior variance E P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t). Such worst-case upper bound is obtained

via purely algebraic techniques, ignoring the expectation over the randomness of As,t and ˆΣs,t.
Therefore, we may achieve sharper regret bound by considering the expectation over the randomness
in the posterior variance.

(3) To derive a sharper and meaningful upper bound on the posterior-variance Vm,n
=
E P

t≥1
P

s∈St∥As,t∥2
ˆΣs,t, we need to consider other bounding technique like concentration in-
equality, or more technical matrix analysis, to achieve an upper bound that is sublinear w.r.t. the

28


---Page Break---
number m of tasks and sublinear w.r.t. the number n of iterations per task. Only in this way can we
obtain a multi-task regret bound o(mn) that is sublinear w.r.t. m and sublinear w.r.t. n.

(4) On the other hand, we also consider ﬁnding the lower bound of posterior-variance Vm,n to
show that our upper bound on Vm,n is tight, or ﬁnding the lower bound of multi-task Bayes regret
BR(m, n) to show that our multi-task Bayes regret upper bound could not be improved. This serves
as one of our ongoing research directions.

F
Additional Experiments and Computer Resources

0
2
4
6
8
10
Number of Tasks m

0

50

100

150

Regret

Linear Bandit (d = 4, σq = 1.0, L=1)

n=100
n=200
n=300
n=400

(a) Regrets w.r.t. different m

2
4
6
8
10
Number of Concurrent Tasks L

0

50

100

150

Regret

Linear Bandit (m = 10, σq = 1.0)

d=8
d=4
d=2

(b) Regrets w.r.t. different L

0
100
200
300
400
Round t

0

50

100

150

Regret

Linear Bandit (d = 4, m = 10, L=5)

σq=5
σq=4
σq=3

σq=2
σq=1

(c) Regrets w.r.t. different σq

0
100
200
300
400
Round t

0

100

200

300

Regret

Linear Bandit (d = 4, m = 10, L=5)

σ0=0.5
σ0=0.4
σ0=0.3
σ0=0.2
σ0=0.1

(d) Regrets w.r.t. different σ0

0
100
200
300
400
Round t

0

250

500

750

1000

Regret

Linear Bandit (d = 4, m = 10, L=5)

σ=10
σ=7
σ=4
σ=1
σ=0.1

(e) Regrets w.r.t. different σ

0
100
200
300
400
Round t

0

100

200

300

400

Regret

Linear Bandit (d = 4, σq = 0.5)

OracleTS
TS
HierTS
HierBayesUCB

(f) Regrets of different algorithms
Figure 2: Regrets of HierBayesUCB algorithm with respect to (w.r.t.) different hyper-parameters.

Experimental Results. From Figure 2, we have the similar observations as that in Figure 1: (1) In
plot (a), the multi-task regret of HierBayesUCB becomes larger with the increase of m and n, which
is consistent with our regret upper bound in Theorems 5.2. (2) In plot (b), the regret increases with
a higher dimension d, and increases with a larger number L of the concurrent tasks. (3) In plots
(c)-(e), the regret decreases with a smaller variance (e.g. σq, σ0 and σ) in hierarchical Bayesian
model, validating the provable beneﬁts of variance-reduction in Bayes regret minimization. (4)
The task-averaged regret of our proposed HierBayesUCB is smaller than that of HierTS, and such
improvement becomes larger with the increase of σq (when compared with σq = 1.0 in Figure 1 (f)).

Computer Resources. Our implementations are based on Python. We run all bandit algorithms on
a platform with 8 NVIDIA RTX 6000 GPUs and 2 AMD EPYC 7543 Processors. Each GPU has
48G memory, and each CPU has 64 cores. The CUDA version is 12.1, the Python version 3.7.16,
the matplotlib version 3.5.3, and the tensorﬂow version 1.15. The source code for reproducing all
experimental results of HierTS and HierBayesUCB is provided in the supplementary material.

29


---Page Break---
Table 4: The technical novelties for deriving our improved sequential regret bound when compared with the latest regret bound in [17, Thm 3]. m is the number of
bandit tasks, n is the number of iterations per task, and d is the dimension of action a ∈A.

Regret Bound
[17, Thm 3]
Existing Problems
Improvement Motivations
Our Theorem 5.1
Our Improvements

Bound I

dm
q

n log (mn) log (1 + nλ1(Σ0)

dσ2
)

×
r

λ1(Σ0)

log (1+ λ1(Σ0)

σ2
)

(1) There exists an
additional factor
p

log (mn).

(1) Use a Shwartz-type
inequality in Lemma A.2,
instead of UCB strategy,
to bound per-task regret
to avoid additional term
log 1

δ (where δ = nm).

(2) Deﬁne a new matrix
˜Xs,t s.t. the denominator
in the regret is
σ2 + B2λ1(Σ0), not only
σ2. Avoid the case that
the variance serves alone
as the denominator.

(3) Give a more technical
analysis in Lemma A.1
to improve λ2
1(Σ0)

λ2
d(Σ0) to

λ1(Σ0)

λd(Σ0)

dm
q

n log
 
1 + n

d


×
q 
σ2 + B2λ1(Σ0)


(1) Our regret bounds

remove the
p

log (mn)

factor.

(2) To minimize our

bound, it sufﬁces

to decrease the

variances σ2, λ1(Σ0),

λ1(Σq).

(3) Our regret bounds

also show that

we should decrease

the condition number

λ1(Σ0)

λd(Σ0)

of the variance

matrix Σ0

to minimize the

Bayes regret.

Bound II

d
q

mn log (mn) log (1 + mλ1(Σq)

λd(Σ0) )

×

v
u
u
t
λ2
1(Σ0)λ1(Σq)
 
1+ λ1(Σ0)

σ2


λ2
d(Σ0) log
 
1+
λ2
1(Σ0)λ1(Σq)

λ2
d(Σ0)σ2


(1) There also exists an
additional factor
p

log (mn).

(2) There exists a paradox in
this bound, i.e. variance σ2
exists in both the denominator
and numerator. Then
whether we should increase
or decrease σ2
to minimize the regret bound?

d
q

mn log
 
1 + m Tr(ΣqΣ−1
0
)

d


×
q

σ2 + B2λ1(Σ0) + B2 λ1(Σ0)λ1(Σq)

λd(Σ0)

30


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reﬂect the
paper’s contributions and scope?

Answer: [Yes]

Justiﬁcation: see Section 5.

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

Justiﬁcation: see Section E.

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

31


---Page Break---
Justiﬁcation: see assumptions in Section 5, and see proof sketch in Section E.
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
Justiﬁcation: see implementation details in Section 6 and Section F, and see the source code
in the supplementary material.
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

Question: Does the paper provide open access to the data and code, with sufﬁcient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

32


---Page Break---
Answer: [Yes]
Justiﬁcation: see the source code in our supplementary material.
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
Justiﬁcation: see the implementation details in Section 6.
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
Justiﬁcation: see our Figures 1-2.
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

33


---Page Break---
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

Justiﬁcation: see details in Section F.

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

Justiﬁcation: this paper is a purely theoretical paper and has no negative social impact. Be-
sides, we release the source code to implement our proposed algorithms in the supplementary
material.

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

Justiﬁcation: there is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake proﬁles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact speciﬁc
groups), privacy considerations, and security considerations.

34


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
feedback over time, improving the efﬁciency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justiﬁcation: the paper poses no such risks.

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

Justiﬁcation: the paper does not use existing assets.

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

35


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justiﬁcation: the paper does not involve crowdsourcing nor research with human subjects.
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
Justiﬁcation: the paper does not involve crowdsourcing nor research with human subjects.
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
Justiﬁcation: the paper does not involve crowdsourcing nor research with human subjects.
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

36


---Page Break---
