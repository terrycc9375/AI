Minimax Optimal and Computationally Efficient
Algorithms for Distributionally Robust Offline
Reinforcement Learning

Zhishuai Liu
Duke University
zhishuai.liu@duke.edu

Pan Xu
Duke University
pan.xu@duke.edu

Abstract

Distributionally robust offline reinforcement learning (RL), which seeks robust
policy training against environment perturbation by modeling dynamics uncertainty,
calls for function approximations when facing large state-action spaces. However,
the consideration of dynamics uncertainty introduces essential nonlinearity and
computational burden, posing unique challenges for analyzing and practically
employing function approximation. Focusing on a basic setting where the nominal
model and perturbed models are linearly parameterized, we propose minimax
optimal and computationally efficient algorithms realizing function approximation
and initiate the study on instance-dependent suboptimality analysis in the context of
robust offline RL. Our results uncover that function approximation in robust offline
RL is essentially distinct from and probably harder than that in standard offline
RL. Our algorithms and theoretical results crucially depend on a novel function
approximation mechanism incorporating variance information, a new procedure
of suboptimality and estimation uncertainty decomposition, a quantification of
the robust value function shrinkage, and a meticulously designed family of hard
instances, which might be of independent interest.

1
Introduction

Offline reinforcement learning (RL) [17, 18], which aims to learn an optimal policy achieving
maximum expected cumulative reward from a pre-collected dataset, plays an important role in critical
domains where online exploration is infeasible due to high cost or ethical issues, such as precision
medicine [49, 11, 22, 21] and autonomous driving [32, 43]. The foundational assumption of offline RL
[18, 15, 53] is that the offline dataset is collected from the same environment where learned policies
are intended to be deployed. However, this assumption can be violated in practice due to temporal
changes in dynamics. In such cases, standard offline RL could face catastrophic failures [10, 31, 64].
To address this issue, the robust offline RL [28, 30] focuses on robust policy training against the
environment perturbation, which serves as a promising solution. Existing empirical successes of
robust offline RL rely heavily on expressive function approximations [37, 36, 25, 45, 63, 16], as the
omnipresence of applications featuring large state and action spaces necessitates powerful function
representations to enhance generalization capability of decision-making in RL.

To theoretically understand robust offline RL with function approximation, the distributionally robust
Markov decision process (DRMDP) [39, 30, 13] provides an established framework. In stark contrast
to the standard MDP, DRMDP specifically tackles the model uncertainty by forming an uncertainty
set around the nominal model, and takes a max-min formulation aiming to maximize the value
function corresponding to a policy, uniformly across all perturbed models in the uncertainty set
[55, 52, 57, 35, 41, 59, 40]. The core of DRMDPs lies in achieving an amenable combination
of uncertainty set design and corresponding techniques to solve the inner optimization over the

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
uncertainty set. However, this consideration of model uncertainty introduces fundamental challenges
to function approximation in terms of computational and statistical efficiency, particularly given the
need to maximally exploit essential information in the offline dataset. For instance, in cases where
the state and action spaces are large, the commonly used (s, a)-rectangular uncertainty set can make
the inner optimization computationally intractable for function approximation [66]. Additionally, the
distribution shifts, arising from the mismatch between the behavior policy and the target policy, as well
as the mismatch between the nominal model and perturbed models, complicate the statistical analysis
[41, 4]. Several recent works attempt to conquer these challenges. Panaganti et al. [35] studied the
(s, a)-rectangularity, and their algorithm may suffer from the above mentioned computational issue.
Additionally, the (s, a)-rectangular uncertainty set may contain transitions that would never happen in
reality, and thus leads to conservative policies; Blanchet et al. [4] proposed a novel double pessimism
principle, while their algorithm requires strong oracles, which is not practically implementable.
Meanwhile, a line of works study function approximation in the online setting [44, 38, 51, 3, 20] or
with a simulator [66], which are not applicable to offline RL. Thus, the following question arises:

Is it possible to design a computationally efficient and minimax optimal algorithm for
robust offline RL with function approximation?

To answer the above question, we focus on a basic setting of d-rectangular linear DRMDP, where
the nominal model is a standard linear MDP, and all perturbed models are parameterized in a linearly
structured uncertainty set. We provide the first instance-dependent suboptimality analysis in the
DRMDP literature with function approximation, which offers insights into the problem’s intrinsic
characteristics and challenges. Concretely, our contributions are summarized as follows.

• We propose a computationally efficient algorithm, Distributionally Robust Pessimistic Value
Iteration (DRPVI), based on the pessimism principle [15, 53, 41] with a new function approximation
mechanism explicitly devised for d-rectangular linear DRMDPs. We show that DRPVI achieves
the following instance-dependent upper bound on the suboptimality gap:

β1 · supP ∈Uρ(P 0)
PH
h=1 Eπ⋆,P  Pd
i=1 ∥ϕi(sh, ah)1i∥Λ−1
h |s1 = s
1,

This bound resembles those established in offline RL within standard linear MDPs [15, 62, 54].
However, there are two significant differences in our results. First, our bound depends on the
supremum over the uncertainty set of transition kernels instead of one single transition kernel.
Second, our result relies on a diagonal-based normalization, instead of the Mahalanobis norm of
the feature vector, ∥ϕ(sh, ah)∥Λ−1
h . See Table 1 for a clearer comparison. These two distinctions
are unique to DRMDPs with function approximation, which we discuss in more details in Section 4.
Moreover, our analysis provides a novel pipeline for studying instance-dependent upper bounds of
computationally efficient algorithms under d-rectangular linear DRMDPs.
• We improve DRPVI by incorporating variance information into the new function approximation
mechanism, resulting in the VA-DRPVI algorithm, which achieves a smaller upper bound:

β2 · supP ∈Uρ(P 0)
PH
h=1 Eπ⋆,P  Pd
i=1 ∥ϕi(sh, ah)1i∥Σ⋆−1
h
|s1 = s
2.

This improves the result of DRPVI due to the fact that Σ⋆−1
h
⪯H2Λ−1
h
by definition [60, 54].
Furthermore, when the uncertainty level ρ = O(1), we show that the robust value function attains
a Range Shrinkage property, leading to an improvement in the upper bound by an order of H. This
explicit improvement is new in variance-aware algorithms, and is unique to DRMDPs.
• We further establish an information-theoretic lower bound. We prove that the upper bound of
VA-DRPVI matches the information-theoretic lower bound up to β2, which implies that VA-DRPVI
is near-optimal in the sense of information theory. Importantly, both DRPVI and VA-DRPVI are
computationally efficient and do not suffer from the high computational burden, as discussed
above in settings with the (s, a)-rectangular uncertainty set, due to a decoupling property of the
d-rectangular uncertainty set (see Remark 4.1 for more details). Thus, we confirm that, for robust
offline RL with function approximation, both the computational efficiency and minimax optimality
are achievable under the setting of d-rectangular linear DRMDPs.

1Here, d is the feature dimension, H is the horizon length, β1 = ˜O(
√

dH) is a tunning parameter in DRPVI,
Uρ(P 0) is the uncertainty set with radius ρ, π⋆is the optimal robust policy, ϕ(·, ·) : S × A →Rd is the
instance-dependent feature vector, and Λh is the covariance matrix defined in (4.3).
2β2 = ˜O(
√

d) is a hyperparameter in VA-DRPVI; Σ⋆
h is the variance-weighted covariance matrix, see (5.5).

2


---Page Break---
Table 1: Summary of instance-dependent results in offline RL with linear function approximation.
Λh and Σ⋆
h are the empirical covariance matrix defined in (4.3) and (5.5) respectively. Note that
π⋆means the optimal policy in standard MDPs and the optimal robust policy in DRMDPs. The
definition of Σ⋆
h also depends on the corresponding definition of π⋆.

Algorithm
Setting
Instance-dependent upper bound on the suboptimality gap

PEVI [15]
MDP
dH · PH
h=1 Eπ⋆,P 
∥ϕ(sh, ah)∥Λ−1
h |s1 = s


LinPEVI-ADV
[54]
MDP
√

dH · PH
h=1 Eπ⋆,P 
∥ϕ(sh, ah)∥Λ−1
h |s1 = s


LinPEVI-ADV+
[54]
MDP
√

d · PH
h=1 Eπ⋆,P 
∥ϕ(sh, ah)∥Σ⋆−1
h
|s1 = s


DRPVI (ours)
DRMDP
√

dH · supP ∈Uρ(P 0)
PH
h=1 Eπ⋆,P  Pd
i=1 ∥ϕi(sh, ah)1i∥Λ−1
h |s1 = s


VA-DRPVI (ours)
DRMDP
√

d · supP ∈Uρ(P 0)
PH
h=1 Eπ⋆,P  Pd
i=1 ∥ϕi(sh, ah)1i∥Σ⋆−1
h
|s1 = s


Our algorithm design and theoretical analysis draw inspiration from two crucial ideas proposed in
standard linear MDPs: the reference-advantage decomposition [54] and the variance-weighted ridge
regression [65]. However, the unique challenges in DRMDPs necessitate novel treatments that go
far beyond a combination of existing techniques. Specifically, existing analysis of standard linear
MDPs highly relies on the linear dependency of the Bellman equation on the (nominal) transition
kernel. This linear dependency is disrupted by the consideration of model uncertainty, which induces
essential nonlinearity that significantly complicates the statistical analysis of estimation error. To
obtain our instance-dependent upper bounds, we establish a new theoretical analysis pipeline. This
pipeline starts with a nontrivial decomposition of the suboptimality, and employs a new uncertainty
decomposition that transforms the estimation uncertainty over all perturbed models to estimation
uncertainty under the nominal model.

The information-theoretic lower bound in our paper is the first of its kind in the linear DRMDP
setting, which could be of independent interest to the community. Previous lower bounds, which
are based on the commonly used Assouad’s method and established under the standard linear MDP,
do not consider model uncertainty. In particular, one prerequisite for applying Assouad’s method
is switching the initial minimax objective to a minimax risk in terms of Hamming distance. The
intertwining of this prerequisite with the nonlinearity induced by the model uncertainty makes the
analysis significantly more challenging. To this end, we construct a novel family of hard instances,
carefully designed to (1) mitigate the nonlinearity caused by the model uncertainty, (2) fulfil the
prerequisite for Assouad’s method, and (3) be concise enough to admit matrix analysis.

Notations:
We denote ∆(S) as the set of probability measures over some set S. For any num-
ber H ∈Z+, we denote [H] as the set of {1, 2, · · · , H}. For any function V : S →R, we
denote [PhV ](s, a) = Es′∼Ph(·|s,a)[V (s′)] as the expectation of V with respect to the transition
kernel Ph, [VarhV ](s, a) = [PhV 2](s, a) −([PhV ](s, a))2 as the variance of V , [VhV ](s, a) =
max{1, [Varh V ](s, a)} as the truncated variance of V , and [V (s)]α = min{V (s), α}, given a scalar
α > 0, as the truncated value of V . For a vector x, we denote xj as its j-th entry. And we denote
[xi]i∈[d] as a vector with the i-th entry being xi. For a matrix A, denote λi(A) as the i-th eigenvalue
of A. For two matrices A and B, we denote A ⪯B as the fact that B −A is a positive semidefinite
matrix. For any function f : S →R, we denote ∥f∥∞= sups∈S f(s). Given P, Q ∈∆(S), the
total variation divergence of P and Q is defined as D(P||Q) = 1/2
R

S |P(s) −Q(s)|ds.

2
Most Related Work

DRMDPs.
The DRMDP framework has been extensively studied under different settings. The
works of [55, 52, 61, 26, 12] assumed precise knowledge of the environment and formulated the
DRMDP as classic planning problems. The works of [67, 57, 33, 56, 42, 58] assumed access to a
generative model and studied the sample complexities of DRMDPs. The works of [35, 41, 4] studied
the offline setting assuming access to only an offline dataset, and established sample complexities
under data coverage or concentrability assumptions. The works of [51, 3, 8, 19, 20] studied the online
setting where the agent can actively interact with the nominal environment to learn the robust policy.

3


---Page Break---
DRMDPs with linear function approximation.
Tamar et al. [44], Badrinath and Kalathil [3]
proposed to use linear function approximation to solve DRMDPs with large state and action spaces
and established asymptotic convergence guarantees. Zhou et al. [66] studied the natural Actor-
Critic with function approximation, assuming access to a simulator. Their function approximation
mechanisms depend on two novel uncertainty sets, one based on double sampling and the other on
an integral probability metric. Ma et al. [24] first combined the linear MDP with the d-rectangular
uncertainty set [12], and proposed the setting dubbed as the d-rectangular linear DRMDP, which
naturally admits linear representations of the robust Q-functions3. Panaganti et al. [34] leverages the
d-rectangular linear DRMDP framework to address the distribution shift problem in offline linear
MDPs. Blanchet et al. [4] studied the offline d-rectangular linear DRMDP setting, for which the
provable efficiency is established under a double pessimism principle. Liu and Xu [20] then studied
the online d-rectangular linear DRMDP setting and pointed out that the intrinsic nonlinearity of
DRMDPs might pose additional challenges for linear function approximation. After the release of our
work, a concurrent study [48] emerged, which independently investigated offline DRMDPs with linear
function approximation. Their algorithms attained the same instance-dependent suboptimalities as our
proposed algorithms DRPVI and VA-DRPVI. Their algorithm DROP also achieved the same order of
worst-case suboptimality, ˜O(dH2/
√

K), as our DRPVI. However, we further demonstrated that our
algorithm VA-DRPVI can strictly improve this result to ˜O(dH min{1/ρ, H}/
√

K). Moreover, we
introduced a novel hard instance and established the first information-theoretic lower bound for offline
DRMDPs with linear function approximation. We also note that there is a line of works [4, 35] studied
general function approximation under DRMDPs with the commonly studied (s, a)-rectangularity
uncertainty sets, where no further structure is applied except the rectangularity.

3
Problem Formulation

In this section, we provide the preliminary of d-rectangular linear DRMDPs, and describe the dataset
as well as the learning goal in offline reinforcement learning.

Standard MDPs.
We start with the standard MDP, which constitutes the basic of DRMDPs. A
finite horizon Markov decision process is denoted by MDP(S, A, H, P, r), where S and A are
the state and action spaces, H ∈Z+ is the horizon length, P = {Ph}H
h=1 denotes the set of
probability transition kernels, r = {rh}H
h=1 denotes the reward functions. More specifically, for
any (h, s, a) ∈[H] × S × A, the transition kernel Ph(·|s, a) is a probability function over the
state space S, and the reward function rh : S × A →[0, 1] is assumed to be deterministic for
simplicity. A sequence of deterministic policies is denoted as π = {πh}H
h=1, where πh : S →A is
the policy for step h ∈[H]. Given any policy π and transition P, for all (s, a, h) ∈S × A × [H],
the corresponding value function V π,P
h
(s) := Eπ,P  PH
t=h rt(st, at)
sh = s

and Q-function
Qπ,P
h
(s, a) := Eπ,P  PH
t=h rt(st, at)
sh = s, ah = a

characterize the expected cumulative rewards
starting from step h, and both of them are bounded in [0, H].

Distributionally robust MDPs.
A finite horizon distributionally robust Markov decision process
is denoted by DRMDP(S, A, H, Uρ(P 0), r), where P 0 = {P 0
h}H
h=1 is the set of nominal transition
kernels, and Uρ(P 0) = N

h∈[H] Uρ
h(P 0
h) is the uncertainty set of transitions, where each Uρ
h(P 0
h) is
usually defined as a ball centered at P 0 with radius/uncertainty level ρ ≥0 based on some probability
divergence measures [13, 57, 56]. To account for the model uncertainty, the robust value function
V π,ρ
h
(s) := infP ∈Uρ(P 0) V π,P
h
(s), ∀(h, s) ∈[H] × S is defined as the value function under the
worst possible transition kernel within the uncertainty set Uρ(P 0). Similarly, the robust Q-function
is defined as Qπ,ρ
h (s, a) = infP ∈Uρ(P 0) Qπ,P
h
(s, a), for any (h, s, a) ∈[H] × S × A. Further, we
define the optimal robust value function and the optimal robust Q-function as

V ⋆,ρ
h
(s) = supπ∈Π V π,ρ
h
(s),
Q⋆,ρ
h (s, a) = supπ∈Π Qπ,ρ
h (s, a),
∀(h, s, a) ∈[H] × S × A.

3Ma et al. [24] study the offline d-rectangular linear DRMDPs with Kullback-Leibler (KL) uncertainty sets.
We remark that 1) the proofs of their main lemmas (Lemma D.1 and Lemma D.2) related to suboptimality
decomposition and the proof of theorems have technique flaws; 2) The formulation of their assumption 4.4 on
the dual variable of the dual formulation of the KL-divergence is ambiguous and may be too strong to be realistic.
Thus, the fundamental challenges of d-rectangular linear DRMDPs remain unresolved.

4


---Page Break---
where Π is the set of all policies. The optimal robust policy π⋆= {π⋆
h}H
h=1 is defined as the policy
that achieves the optimal robust value function: π⋆
h(s) = arg supπ∈Π V π,ρ
h
(s), ∀(h, s) ∈[H] × S.

d-rectangular linear DRMDPs.
A d-rectangular linear DRMDP is a DRMDP where the nominal
environment is a special case of linear MDP with a simplex feature space [14, Example 2.2] and the
uncertainty set Uρ
h(P 0
h) is defined based on the linear structure of the nominal transition kernel P 0
h.
In particular, we make the following assumption about the nominal environment.

Assumption 3.1. Let ϕ : S×A →Rd be a state-action feature mapping such that Pd
i=1 ϕi(s, a) = 1,
ϕi(s, a) ≥0, for any (i, s, a) ∈[d] × S × A.
For any (h, s, a) ∈[H] × S × A, the
reward function and the nominal transition kernels have a linear representation: rh(s, a) =
⟨ϕ(s, a), θh⟩, and P 0
h(·|s, a) = ⟨ϕ(s, a), µ0
h(·)⟩, where ∥θh∥2 ≤
√

d, and µ0
h = (µ0
h,1, . . . , µ0
h,d)⊤

are unknown probability measures over S.

With notations in Assumption 3.1, we define the factor uncertainty sets as Uρ
h,i(µ0
h,i) =

µ : µ ∈
∆(S), D(µ||µ0
h,i) ≤ρ
	
, ∀(h, i) ∈[H] × [d], where D(·||·) is specified as the total variation (TV)
divergence in this work. The uncertainty set is defined as Uρ
h(P 0
h) = N

(s,a)∈S×A Uρ
h(s, a; µ0
h),

where Uρ
h(s, a; µ0
h) = {Pd
i=1 ϕi(s, a)µh,i(·) : µh,i(·) ∈Uρ
h,i(µ0
h,i), ∀i ∈[d]}. A notable feature of
this design is that the factor uncertainty sets {Uρ
h,i(µ0
h,i)}H,d
h,i=1 are decoupled from the state-action
pair (s, a) and also independent with each other. As demonstrated later, this decoupling property
results in a computationally efficient regime for function approximation.

Robust Bellman equation.
Under the setting of d-rectangular linear DRMDPs, it is proved that the
robust value function and the robust Q-function satisfy the robust Bellman equations [20]:

Qπ,ρ
h (s, a) = rh(s, a) + infPh(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV π,ρ
h+1](s, a),
(3.1a)

V π,ρ
h
(s) = Ea∼πh(·|s)

Qπ,ρ
h (s, a)

,
(3.1b)

and the optimal robust policy π⋆is deterministic. Thus, we can restrict the policy class Π to the
deterministic one. This leads to the robust Bellman optimality equations:

Q⋆,ρ
h (s, a) = rh(s, a) + infPh(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV ⋆,ρ
h+1](s, a),
(3.2a)

V ⋆,ρ
h
(s) = maxa∈A Q⋆
h(s, a).
(3.2b)

Offline Dataset and the Learning Goal.
Let D denote an offline dataset consisting of K i.i.d
trajectories generated from the nominal environment MDP(S, A, H, P 0, r) by a behavior policy
πb = {πb
h}H
h=1. In concrete, for each τ ∈[K], the trajectory {(sτ
h, aτ
h, rτ
h)}H
h=1 satisfies that
aτ
h ∼πb
h(·|sτ
h), rτ
h = rh(sτ
h, aτ
h), and sτ
h+1 ∼P 0
h(·|sτ
h, aτ
h) for any h ∈[H]. The goal of the
robust offline RL is to learn the optimal robust policy π⋆using the offline dataset D. We define the
suboptimality gap between any policy ˆπ and the optimal robust policy π⋆as

SubOpt(ˆπ, s1, ρ) := V ⋆,ρ
1
(s1) −V ˆπ,ρ
1
(s1).
(3.3)

Then the goal of an algorithm in distributionally robust offline reinforcement learning is to learn a
robust policy ˆπ that minimizes the suboptimality gap SubOpt(ˆπ, s, ρ), for any s ∈S.

4
Warmup: Robust Pessimistic Value Iteration

In this section, we first propose a simple algorithm in Algorithm 1 as a warm start, and provide an
instance-dependent upper bound on its suboptimality gap in Theorem 4.4.

The optimal robust Bellman equation (3.2) implies that the optimal robust policy π⋆is greedy with
respect to the optimal robust Q-function. Therefore, it suffices to estimate Q⋆,ρ
h
to approximate π⋆.
To this end, we estimate the optimal robust Q-function by iteratively performing an empirical version
of the optimal robust Bellman equation similar to (3.2). In concrete, given the estimators at step h+1,
denoted by bQh+1(s, a) and bVh+1(s) = maxa∈A bQh+1(s, a), Liu and Xu [20] show that applying
one step backward induction similar to (3.2) leads to

Qh(s, a) = rh(s, a) + infPh(·|s,a)∈Uρ
h(s,a;µ0
h)

Ph bVh+1

(s, a) =

ϕ(s, a), θh + νρ
h

,
(4.1)

5


---Page Break---
where νρ
h,i := maxα∈[0,H]{zh,i(α) −ρ(α −mins′[bVh+1(s′)]α)}, zh,i(α) := Eµ0
h,i[bVh+1(s′)]α,

∀i ∈[d], [bVh+1(s′)]α = min{bVh+1(s′), α}, and α is a dual variable stemming from the dual
formulation (see Proposition H.1). To estimate Qh(s, a), it suffices to estimate vectors zh(α) =
[zh,1(α), . . . , zh,d(α)] and νρ
h as follows.

• Estimate zh(α): note that [P0
h[Vh+1]α](s, a) = ⟨ϕ(s, a), zh(α)⟩by Assumption 3.1, where the
expectation is taken with respect to the nominal kernel P 0
h(·|s, a). Given the estimator bVh+1(s), it
is natural to estimate zh(α) by solving the following ridge regression on the offline dataset D.

ˆzh(α) = argminz∈Rd PK
τ=1
 bVh+1(sτ
h+1)


α −ϕτ⊤
h z
2 + λ∥z∥2
2

= Λ−1
h
 PK
τ=1 ϕτ
h[bVh+1(sτ
h+1)]α

,
(4.2)

where λ > 0, ϕτ
h is a shorthand notation for ϕ(sτ
h, aτ
h), and Λh is the covariance matrix:

Λh = PK
τ=1 ϕτ
h(ϕτ
h)⊤+ λI.
(4.3)

• Estimate ˆνρ
h: based on ˆzh,i(α), we can estimate ˆνρ
h,i as follows.

ˆνρ
h,i = maxα∈[0,H]{ˆzh,i(α) −ρ(α −mins′[bV ρ
h+1(s′)]α}, ∀i ∈[d].
(4.4)

After these two steps, we immediately obtain the estimated robust Q-function at step h,

bQh(s, a) =

ϕ(s, a), θh + ˆνρ
h

.
(4.5)

Note that these estimations are constructed based on an offline dataset D, which is known to cause
distributional shift. We propose to incorporate a penalty term in the estimator (4.5) following the
pessimism principle in the face of uncertainty [15, 53, 41].

Algorithm 1 Distributionally Robust Pessimistic Value Iteration (DRPVI)

Require: Input dataset D and parameter β1; bV ρ
H+1(·) = 0.
1: for h = H, · · · , 1 do
2:
Λh ←PK
τ=1 ϕτ
hϕτ⊤
h
+ λI
3:
for i = 1, · · · , d do
4:
Update ˆνρ
h,i according to (4.4)
5:
end for
6:
Γh(·, ·) ←β1
Pd
i=1
ϕi(·, ·)1i

Λ−1
h
7:
bQρ
h(·, ·) ←

ϕ(·, ·)⊤(θh + ˆνρ
h) −Γh(·, ·)
	

[0,H−h+1]
8:
ˆπh(·|·) ←argmaxπh

 bQρ
h(·, ·), πh(·|·)


A, and bV ρ
h (·) ←⟨bQρ
h(·, ·), ˆπh(·|·)⟩A
9: end for

Remark 4.1. In Algorithm 1, the pessimism is achieved by subtracting a robust penalty term,
Pd
i=1
ϕi(·, ·)1i

Λ−1
h , from the robust Q-function estimation, which is derived from bounding the

robust estimation uncertainty arising from d ridge regressions. In particular, at step h ∈[H], denoting
α⋆
i = argmax[0,H]

ˆzh,i(α) −ρ
 
α −mins′ bV ρ
h+1(s′)


α
	
, ∀i ∈[d], we solve d separate ridge
regressions to obtain different coordinates of ˆνρ
h. This design is tailored for the d-rectangular linear
DRMDP, as we will see, leading to a distinct instance-dependent upper bound in Theorem 4.4.

Remark 4.2. Notably, to solve the optimization problem with respect to α ∈[0, H] in (4.4), one will
repeatedly invoke the closed form solution (4.2) for different values of α. Moreover, the optimization
is decoupled from the state-action pair, due to the decoupling property of d-rectangular uncertainty
set. Similar algorithm designs have also appeared in [24] for Kullback-Leibler divergence based
linear DRMDPs and in [20] for online linear DRMDPs. As for the computational tractability, we
note that the minimization over α in (4.4) has been implemented in [20] using the minimize function
in the Nelder-Mead method [29] in the Python module scipy.optimize. The minimization over the
state space is avoided under a ‘fail-state’ assumption, common in applications such as robotics and
healthcare (see Assumption 4.1 and Remark 4.2 in their paper). Without this assumption, we can also
use the Nelder-Mead method to solve it. Thus, Algorithm 1 is in general computationally tractable.

6


---Page Break---
Before presenting the theoretical guarantee of DRPVI, we make the following data coverage assump-
tion, which is standard for offline linear MDPs [50, 9, 60, 54].

Assumption 4.3. We assume κ := minh∈[H] λmin(Eπb,P 0[ϕ(sh, ah)ϕ(sh, ah)⊤]) > 0 for the
behavior policy πb and the nominal transition kernel P 0.

Assumption 4.3 requires the behavior policy to sufficiently explore the state-action space under the
nominal environment. Indeed, it implicitly assumes that the nominal and perturbed environments
share the same state-action space, and that the full information of this space is accessible through the
nominal environment and the behavior policy πb. Assumption 4.3 rules out cases where new states
emerge in perturbed environments that can never be queried under the nominal environment as a
result of the distribution shift. Now we present the theoretical guarantee for Algorithm 1.
Theorem 4.4. Under Assumptions 3.1 and 4.3, ∀K > max{512 log(2dH2/δ)/κ2, 20449d2H2/κ}
and δ ∈(0, 1), if we set λ = 1 and β1 = ˜O(
√

dH) in Algorithm 1, then with probability at least
1 −δ, ∀s ∈S, the suboptimality of DRPVI satisfies

SubOpt(ˆπ, s, ρ) ≤β1 ·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P

d
X

i=1
∥ϕi(sh, ah)1i∥Λ−1
h
s1 = s

,
(4.6)

where Λh is the empirical covariance matrix defined in (4.3).

The result in Theorem 4.4 resembles existing instance-dependent bounds for standard linear MDPs [15,
54] (see Table 1 for a detailed comparison). However, there are two major distinctions between these
results. First, our result depends on the weighted sum of diagonal elements Pd
i=1 ∥ϕi(sh, ah)1i∥Λ−1
h ,
dubbed as the d-rectangular robust estimation error, instead of the Mahalanobis norm of the feature
vector ∥ϕ(sh, ah)∥Λ−1
h . As discussed in Remark 4.1, this term primarily arises due to the necessity
to solve d distinct ridge regressions in each step, which presents a unique challenge in our analysis.
Second, we consider the supremum expectation of d-rectangular robust estimation error with respect to
all transition kernels in the uncertainty set, which measures the worst case coverage of the covariance
matrix Λh under the optimal robust policy π⋆.

To connect with existing literature [4], we further show that under Assumption 4.3, the instance-
dependent suboptimality bound can be simplified as follows.
Corollary 4.5. Under the same assumptions and settings as Theorem 4.4, with probability at least
1 −δ, for all s ∈S, the suboptimality of DRPVI satisfies SubOpt(ˆπ, s, ρ) = ˜O(
√

dH2/(
√

κ · K)).
Remark 4.6. Since ∥ϕ(·, ·)∥2 ≤1 by Assumption 3.1, the coverage parameter κ is trivially upper
bounded by 1/d. Assuming that κ = c†/d for a constant 0 < c† < 1, then we have SubOpt(ˆπ, s, ρ) =
˜O(dH2/(c† ·
√

K)). This bound improves the state-of-the-art, [4, Theorem 6.3], by O(d).

5
Distributionally Robust Variance-Aware Pessimistic Value Iteration

The instance-dependent bound in Theorem 4.4 has an explicit dependency on H, which arises from
the fact that Qρ
h(s, a) ∈[0, H] for any (h, ρ) ∈[H] × (0, 1] and the Hoeffding-type self-normalized
concentration inequality used in our analysis. We will show in this section that the range of any
robust value function could be much smaller under a refined analysis. Consequently, we can leverage
variance information to improve Algorithm 1 and achieve a strengthened upper bound.

Intuition
In the robust Bellman equation (3.1), the worst-case transition kernel would put as much
mass as possible on the minimizer of V π,ρ
h+1(s), denoted by smin. Based on this observation, we
conjecture that the robust Bellman equation (3.1) recursively reduces the maximal value of robust
value functions, and thus shrinks its range. To see this, we define ˇµh,i = (1 −ρ)µ0
h,i + ρδsmin, where
δsmin is the Dirac measure at smin, and we assume V π,ρ
h+1(smin) = 0 for any (π, h) ∈Π × [H] just for
illustration. It is easy to verify that ˇµh,i ∈Uρ
h,i(µ0
h,i) and is indeed the worst-case factor kernel. Then
by (3.1) we have V π,ρ
h
(s) = Ea∼π[rh(s, a) + (1 −ρ)[P0
hV π,ρ
h+1](s, a)], which immediately implies
maxs∈S V π,ρ
h
(s) ≤1 + (1 −ρ) maxs′∈S V π,ρ
h+1(s′). This justifies our conjecture that the range of
the robust value functions shrinks over stage. We dub this phenomenon as Range Shrinkage and
summarize it in the following lemma, with a more formal proof postponed to Appendix G.5.

7


---Page Break---
Algorithm 2 Distributionally Robust and Variance Aware Pessimistic Value Iteration (VA-DRPVI)

Require: Input dataset D, D′ and β2; bV ρ
H+1(·) = 0

1: Run Algorithm 1 using dataset D′ to get {bV

′ρ
h }h∈[H]
2: for h = H, · · · , 1 do
3:
Construct variance estimator bσ2
h(·, ·; α) using D′ by (5.2) and (5.3)

4:
Σh(α) = PK
τ=1 ϕτ
hϕτ⊤
h /bσ2
h(sτ
h, aτ
h; α) + λI

5:
ˆzh(α) = Σ−1
h (α)
 PK
τ=1 ϕτ
h
bV ρ
h+1(sτ
h+1)


α/bσ2
h(sτ
h, aτ
h; α)


6:
αi = argmaxα∈[0,H]{ˆzh,i(α) −ρ(α −mins′[bV ρ
h+1(s′)]α)}, ∀i ∈[d]

7:
ˆνρ
h,i = ˆzh,i(αi) −ρ(αi −mins′[bV ρ
h+1(s′)]αi), ∀i ∈[d]

8:
Γh(·, ·) ←β2
Pd
i=1 ∥ϕi(·, ·)1i∥Σ−1
h (αi)
9:
bQρ
h(·, ·) = {ϕ(·, ·)⊤(θh + ˆνρ
h) −Γh(·, ·)}[0,H−h+1]
10:
ˆπh(·|·) ←argmaxπh⟨bQρ
h(·, ·), πh(·|·)⟩A, bV ρ
h (·) ←⟨bQρ
h(·, ·), ˆπh(·|·)⟩A
11: end for

Lemma 5.1 (Range Shrinkage). For any (ρ, π, h) ∈(0, 1] × Π × [H], we have

max
s∈S V π,ρ
h
(s) −min
s∈S V π,ρ
h
(s) ≤1 −(1 −ρ)H−h+1

ρ
.
(5.1)

This phenomenon only appears in DRMDPs since the range of value function is generally [0, H] in
standard MDPs. A similar phenomenon is first observed in infinite horizon tabular DRMDPs [42,
Lemma 7]. One important implication of Lemma 5.1 is that the conditional variance of any value
function shrinks accordingly. In particular, when ρ = O(1), the range of any robust value function
would shrink to constant order, which leads to constant order conditional variances. This motivates
us to leverage the variance information in both algorithm design and theoretical analysis. Inspired
by the variance-weighted ridge regression in standard linear MDPs [65, 27, 60, 54], we propose to
improve the vanilla ridge regression in (4.2) by incorporating variance weights. To this end, we first
propose an appropriate variance estimator, whose form is specifically motivated by our theoretical
analysis framework, to quantify the variance information.

Variance estimation
We first run Algorithm 1 using an offline dataset D′ that is independent of
D to obtain estimators of the optimal robust value functions {bV

′ρ
h }h∈[H]. By Assumption 3.1, the

variance of [bV

′ρ
h+1]α under the nominal environment is [Varh[bV

′ρ
h+1]α](s, a) = [P0
h[bV

′ρ
h+1]2
α](s, a) −

([P0
h[bV

′ρ
h+1]α](s, a))2 = ⟨ϕ(s, a), zh,2⟩−(⟨ϕ(s, a), zh,1⟩)2. We estimate zh,1 and zh,2 via ridge
regression similarly as in (4.2):

˜zh,2(α) = argminz∈Rd PK
τ=1
 bV

′ρ
h+1(sτ
h+1)
2
α −ϕτ⊤
h z
2 + λ∥z∥2
2,
(5.2a)

˜zh,1(α) = argminz∈Rd PK
τ=1
 bV

′ρ
h+1(sτ
h+1)


α −ϕτ⊤
h z
2 + λ∥z∥2
2.
(5.2b)

We then construct the following truncated variance estimator

bσ2
h(s, a; α) := max

1,

ϕ(s, a)⊤˜zh,2(α)


[0,H2] −

ϕ(s, a)⊤˜zh,1(α)
2
[0,H] −˜O
 dH3

√

Kκ


,
(5.3)

where the last term is a penalty to achieve pessimistic estimations of conditional variances.

Variance-Aware Function Approximation Mechanism
Similar to the two-step estimation proce-
dure of Algorithm 1, we first estimate zh(α) by the following variance-weighted ridge regression
under the nominal environment:

ˆzh(α) = argmin
z∈Rd

K
X

τ=1

 bV ρ
h+1(sτ
h+1)


α −ϕτ⊤
h z
2

bσ2
h(sτ
h, aτ
h; α)
+ λ∥z∥2
2

= Σ−1
h (α)

K
X

τ=1

ϕτ
h[bV ρ
h+1(sτ
h+1)]α
bσ2
h(sτ
h, aτ
h; α)


,
(5.4)

8


---Page Break---
where Σh(α) = PK
τ=1 ϕτ
hϕτ⊤
h /bσ2
h(sτ
h, aτ
h; α) + λI is the empirical variance-weighted covariance
matrix, which can be deemed as an estimator of the following variance-weighted covariance matrix

Σ⋆
h = PK
τ=1 ϕτ
hϕτ⊤
h /[VhV ⋆,ρ
h+1](sτ
h, aτ
h) + λI.
(5.5)

In the second step, we estimate νρ
h,i, ∀i ∈[d] in the same way as (4.4). We then add a pessimism
penalty based on Σh(α). We present the full algorithm details in Algorithm 2.

Theorem 5.2. Under Assumptions 3.1 and 4.3, for K > max{ ˜O(d2H6/κ), ˜O(H4/κ2)} and δ ∈
(0, 1), if we set λ = 1/H2 and β2 = ˜O(
√

d) in Algorithm 2, then with probability at least 1 −δ, the
suboptimality of VA-DRPVI satisfies

SubOpt(ˆπ, s, ρ) ≤β2 ·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P

d
X

i=1
∥ϕi(sh, ah)1i∥Σ⋆−1
h
s1 = s

,
(5.6)

where Σ⋆
h is the population variance-weighted covariance matrix defined as in (5.5).

Note that the bound in Theorem 5.2 does not explicitly depend on H anymore compared with that in
Theorem 4.4. A naive observation is that [VhV ⋆,ρ
h+1](s, a) ∈[1, H2]. By comparing the definitions in
(4.3) and (5.5), we have Σ⋆−1
h
⪯H2Λ−1
h . Thus the upper bound of Algorithm 2 is never worse than
that of Algorithm 1. This improvement brought by variance information is similar to that in standard
linear MDPs [54, Theorem 2]. However, thanks to the range shrinkage phenomenon, we can further
show that VA-DRPVI is strictly better than DRPVI when the uncertainty level is of constant order.
Corollary 5.3. Under the same assumptions and settings as Theorem 5.2, given the uncertainty level
ρ, we have with probability at least 1 −δ, for all s ∈S, the suboptimality of VA-DRPVI satisfies

SubOpt(ˆπ, s, ρ) ≤β2 · (1 −(1 −ρ)H)

ρ
·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P

d
X

i=1
∥ϕi(sh, ah)1i∥Λ−1
h |s1 = s

.

Remark 5.4. Note that (1 −(1 −ρ)H)/ρ = Θ(min{1/ρ, H}). When ρ = O(1), the suboptimality
of Algorithm 2 is strictly smaller than that of Algorithm 1 by H. With a similar argument as in
Remark 4.6, if we assume there exist a constant 0 < c† < 1, such that κ = c†/d in Assumption 4.3,
then the instance-dependent upper bound can be simplified to ˜O(dH min{1/ρ, H}/(c† ·
√

K)),
which improves the state-of-the-art [4, Theorem 6.3] by O(dH) when ρ = O(1).

6
Information-Theoretic Lower Bound

For a matrix A ∈Rd×d and a state s ∈S, we define function Φ(·, ·) : Rd×d × S →R as follows.

Φ(A, s) =
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P

d
X

i=1
∥ϕi(sh, ah)1i∥A
s1 = s

.
(6.1)

It can be seen our upper bounds in previous sections primarily depend on quantities such as Φ(Λ−1
h , s)
and Φ(Σ⋆−1
h
, s). Roughly speaking, these quantities characterize the discrepancy between the
(weighted) covariance matrix of the offline dataset and the state action pairs generated from the
transition probability in the uncertainty set. Hence we call Φ(·, ·) the uncertainty function.

We now establish an information-theoretic lower bound to show that the uncertainty function is
unavoidable for d-rectangular linear DRMDPs. Let M be a class of DRMDPs and we define
SubOpt(M, ˆπ, s, ρ) as the suboptimality gap specific to one DRMDP instance M ∈M.
Theorem 6.1. Given uncertainty level ρ ∈(0, 3/4], dimension d, horizon length H and sample size
K > ˜O(d6), there exists a class of d-rectangular linear DRMDPs M and an offline dataset D of
size K such that for all s ∈S, with probability at least 1 −δ, inf ˆπ supM∈M SubOpt(M, ˆπ, s, ρ) ≥
c · Φ(Σ⋆−1
h
, s), where c is a universal constant.

Theorem 6.1 shows that the uncertainty function Φ(Σ⋆−1
h
, s) is intrinsic to the information-theoretic
lower bound, and thus is inevitable. It is noteworthy that the lower bound in Theorem 6.1 aligns
with the upper bound in Theorem 5.2 up to a factor of β2, which implies that VA-DRPVI is minimax

9


---Page Break---
optimal in the sense of information theory, but with a small gap of ˜O(
√

d). Consequently, we affirm
that, in the context of robust offline reinforcement learning with function approximation, both the
computational efficiency and minimax optimality are achievable under the setting of d-rectangular
linear DRMDPs with TV uncertainty sets. Moreover, Theorem 6.1 also suggests that achieving a good
robust policy necessitates the worst case coverage of the offline dataset over the entire uncertainty
set of transition models, which is significantly different from standard linear MDPs where a good
coverage under the nominal model is enough [15, 60, 54]. Such a distinction indicates that learning
in linear DRMDPs may be more challenging in comparison to standard linear MDPs.

Further, we highlight that the hard instances we constructed also satisfy Assumption 4.3. It remains an
interesting direction to explore what would happen if the nominal and perturbed environments don’t
share exactly the same state-action space. We conjecture that since there could be absolutely new
states emerging in perturbed environments that can never be explored in the nominal environment, the
policy learned merely using data collected from the nominal environment could be arbitrarily bad.

Challenges and novelties in construction of hard instances
Existing tight lower bound analysis
in standard linear MDPs [62, 60, 54] generally depends on the Assouad’s method and a family of hard
instances indexed by ξ ∈{−1, 1}dH. However, they do not consider model uncertainty, which largely
hinders the derivation of explicit formulas for the robust value functions. Further, one prerequisite of
the Assouad’s method is switching the initial minimax suboptimality inf ˆπ maxM∈M SubOpt(ˆπ, s, ρ)
to a risk of the form infξ′ maxξ DH(ξ, ξ′), where DH(·, ·) is the Hamming distance. The model
uncertainty significantly complicates this procedure, as the nonlinearity involved disrupts the linear
dependency between the value function and the index ξ. At the core of Theorem 6.1 is a novel
class of hard instances M. At a high-level, the hard instances should (1) fulfill the d-rectangular
linear DRMDP conditions, (2) mitigate the nonlinearity caused by model uncertainty, (3) achieve the
prerequisite for Assouad’s method, and (4) be concise enough to admit matrix analysis. We postpone
details on the construction of hard instances and the proof of Theorem 6.1 to Appendix F.

As a side product of Theorem 6.1, we show in the following corollary an information-theoretic lower
bound in terms of the instance-dependent uncertainty function Φ(Λ−1
h , s) in Theorem 4.4.

Corollary 6.2. Under the same setting in Theorem 6.1, the class of hard instances M
and offline dataset D in Theorem 6.1 also suggests that, with probability at least 1 −δ,
inf ˆπ supM∈M SubOpt(ˆπ, s, ρ) ≥c · Φ(Λ−1
h , s), where c is a universal constant.

This implies that the uncertainty function Φ(Λ−1
h , s) in Theorem 4.4 also arises from the information-
theoretic lower bound. We note the lower bound in Corollary 6.2 matches the upper bound in
Theorem 4.4 up to β1, thus DRPVI is also minimax optimal in the sense of information theory, but
with a larger gap of ˜O(
√

dH). Moreover, the only difference between Theorem 6.1 and Corollary 6.2
is the covariance matrix. Due to the fact that Λ−1
h
⪯Σ⋆,−1
h
, the information-theoretic lower bound
in Theorem 6.1 is indeed tighter than that in Corollary 6.2.

7
Conclusions

We studied robust offline RL with function approximation under the setting of d-rectangular linear
DRMDPs with TV uncertainty sets. We first proposed the DRPVI algorithm and built up a theoretical
analysis pipeline to establish the first instance-dependent upper bound on the suboptimality gap in
the context of robust offline RL. We then showed an interesting range shrinkage phenomenon specific
to DRMDPs, and we proposed the VA-DRPVI algorithm, which leverages the conditional variance
information of the optimal robust value function. Based on the analysis pipeline built above, we
show that the upper bound of VA-DRPVI achieves sharp dependence on the horizon length H. In
addition, we found that an uncertainty function consisting of two crucial quantities–a supremum
over uncertainty set and a diagonal-based normalization–appears in all upper bounds. We further
established an information-theoretic lower bound to prove that the uncertainty function is unavoidable
for robust offline RL under the setting of d-rectangular linear DRMDPs.

It remains an interesting future research question whether the computational and provable efficiency
can be achieved in other settings for robust offline RL with function approximation. Another interest-
ing future direction is to explore the unique challenges of applying general function approximation
techniques in standard offline RL [6] to DRMDPs.

10


---Page Break---
Acknowledgments

We would like to thank the anonymous reviewers for their helpful comments. ZL and PX was
supported in part by the National Science Foundation (DMS-2323112) and the Whitehead Scholars
Program at the Duke University School of Medicine. The views and conclusions in this paper are
those of the authors and should not be interpreted as representing any funding agency.

References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. Advances in neural information processing systems, 24, 2011. (p. 46.)

[2] Alekh Agarwal, Yuda Song, Wen Sun, Kaiwen Wang, Mengdi Wang, and Xuezhou Zhang.
Provable benefits of representational transfer in reinforcement learning. In The Thirty Sixth
Annual Conference on Learning Theory, pages 2114–2187. PMLR, 2023. (p. 16.)

[3] Kishan Panaganti Badrinath and Dileep Kalathil. Robust reinforcement learning using least
squares policy iteration with provable performance guarantees. In International Conference on
Machine Learning, pages 511–520. PMLR, 2021. (pp. 2, 3, and 4.)

[4] Jose Blanchet, Miao Lu, Tong Zhang, and Han Zhong. Double pessimism is provably efficient
for distributionally robust offline reinforcement learning: Generic algorithm and robust partial
coverage. arXiv preprint arXiv:2305.09659, 2023. (pp. 2, 3, 4, 7, and 9.)

[5] Avinandan Bose, Simon Shaolei Du, and Maryam Fazel. Offline multi-task transfer rl with
representational penalization. arXiv preprint arXiv:2402.12570, 2024. (p. 16.)

[6] Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement
learning. In International Conference on Machine Learning, pages 1042–1051. PMLR, 2019.
(p. 10.)

[7] Yuan Cheng, Songtao Feng, Jing Yang, Hong Zhang, and Yingbin Liang. Provable benefit of
multitask representation learning in reinforcement learning. Advances in Neural Information
Processing Systems, 35:31741–31754, 2022. (p. 16.)

[8] Jing Dong, Jingwei Li, Baoxiang Wang, and Jingzhao Zhang. Online policy optimization for
robust mdp. arXiv preprint arXiv:2209.13841, 2022. (p. 3.)

[9] Yaqi Duan, Zeyu Jia, and Mengdi Wang. Minimax-optimal off-policy evaluation with linear
function approximation. In International Conference on Machine Learning, pages 2701–2709.
PMLR, 2020. (p. 7.)

[10] Jesse Farebrother, Marlos C Machado, and Michael Bowling. Generalization and regularization
in dqn. arXiv preprint arXiv:1810.00123, 2018. (p. 1.)

[11] Omer Gottesman, Fredrik Johansson, Matthieu Komorowski, Aldo Faisal, David Sontag, Finale
Doshi-Velez, and Leo Anthony Celi. Guidelines for reinforcement learning in healthcare. Nature
medicine, 25(1):16–18, 2019. (p. 1.)

[12] Vineet Goyal and Julien Grand-Clement. Robust markov decision processes: Beyond rectangu-
larity. Mathematics of Operations Research, 48(1):203–226, 2023. (pp. 3 and 4.)

[13] Garud N Iyengar. Robust dynamic programming. Mathematics of Operations Research, 30(2):
257–280, 2005. (pp. 1 and 4.)

[14] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement
learning with linear function approximation. In Conference on Learning Theory, pages 2137–
2143. PMLR, 2020. (pp. 5 and 43.)

[15] Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In
International Conference on Machine Learning, pages 5084–5096. PMLR, 2021. (pp. 1, 2, 3, 6,
7, 10, 16, 17, 27, and 28.)

11


---Page Break---
[16] Yufei Kuang, Miao Lu, Jie Wang, Qi Zhou, Bin Li, and Houqiang Li.
Learning robust
policy against disturbance in transition dynamics via state-conservative policy optimization. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 7247–7254,
2022. (p. 1.)

[17] Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In
Reinforcement learning: State-of-the-art, pages 45–73. Springer, 2012. (p. 1.)

[18] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning:
Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.
(p. 1.)

[19] Zhipeng Liang, Xiaoteng Ma, Jose Blanchet, Jun Yang, Jiheng Zhang, and Zhengyuan Zhou.
Single-trajectory distributionally robust reinforcement learning. In Forty-first International
Conference on Machine Learning. (p. 3.)

[20] Zhishuai Liu and Pan Xu. Distributionally robust off-dynamics reinforcement learning: Prov-
able efficiency with linear function approximation. In International Conference on Artificial
Intelligence and Statistics, pages 2719–2727. PMLR, 2024. (pp. 2, 3, 4, 5, 6, 17, and 43.)

[21] Zhishuai Liu, Zishu Zhan, Cunjie Lin, and Baqun Zhang. Estimation in optimal treatment
regimes based on mean residual lifetimes with right-censored data. Biometrical Journal, 65(8):
2200340, 2023. (p. 1.)

[22] Zhishuai Liu, Zishu Zhan, Jian Liu, Danhui Yi, Cunjie Lin, and Yufei Yang. On estimation of
optimal dynamic treatment regimes with multiple treatments for survival data-with application
to colorectal cancer study. arXiv preprint arXiv:2310.05049, 2023. (p. 1.)

[23] Rui Lu, Andrew Zhao, Simon S Du, and Gao Huang. Provable general function class repre-
sentation learning in multitask bandits and mdp. Advances in Neural Information Processing
Systems, 35:11507–11519, 2022. (p. 16.)

[24] Xiaoteng Ma, Zhipeng Liang, Li Xia, Jiheng Zhang, Jose Blanchet, Mingwen Liu, Qianchuan
Zhao, and Zhengyuan Zhou. Distributionally robust offline reinforcement learning with linear
function approximation. arXiv preprint arXiv:2209.06620, 2022. (pp. 4 and 6.)

[25] Ajay Mandlekar, Yuke Zhu, Animesh Garg, Li Fei-Fei, and Silvio Savarese. Adversarially robust
policy learning: Active construction of physically-plausible perturbations. In 2017 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS), pages 3932–3939. IEEE,
2017. (p. 1.)

[26] Shie Mannor, Ofir Mebel, and Huan Xu. Robust mdps with k-rectangular uncertainty. Mathe-
matics of Operations Research, 41(4):1484–1509, 2016. (p. 3.)

[27] Yifei Min, Tianhao Wang, Dongruo Zhou, and Quanquan Gu. Variance-aware off-policy
evaluation with linear function approximation. Advances in neural information processing
systems, 34:7598–7610, 2021. (pp. 8 and 46.)

[28] Jun Morimoto and Kenji Doya. Robust reinforcement learning. Neural computation, 17(2):
335–359, 2005. (p. 1.)

[29] John A Nelder and Roger Mead. A simplex method for function minimization. The computer
journal, 7(4):308–313, 1965. (p. 6.)

[30] Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain
transition matrices. Operations Research, 53(5):780–798, 2005. (p. 1.)

[31] Charles Packer, Katelyn Gao, Jernej Kos, Philipp Krähenbühl, Vladlen Koltun, and Dawn Song.
Assessing generalization in deep reinforcement learning. arXiv preprint arXiv:1810.12282,
2018. (p. 1.)

[32] Yunpeng Pan, Ching-An Cheng, Kamil Saigol, Keuntak Lee, Xinyan Yan, Evangelos Theodorou,
and Byron Boots. Agile autonomous driving using end-to-end deep imitation learning. In
Robotics: science and systems, 2018. (p. 1.)

12


---Page Break---
[33] Kishan Panaganti and Dileep Kalathil. Sample complexity of robust reinforcement learning
with a generative model. In International Conference on Artificial Intelligence and Statistics,
pages 9582–9602. PMLR, 2022. (p. 3.)

[34] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Bridging
distributionally robust learning and offline rl: An approach to mitigate distribution shift and
partial data coverage. In ICML 2024 Workshop: Foundations of Reinforcement Learning and
Control–Connections and Perspectives. (p. 4.)

[35] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust rein-
forcement learning using offline data. Advances in neural information processing systems, 35:
32211–32224, 2022. (pp. 1, 2, 3, and 4.)

[36] Anay Pattanaik, Zhenyi Tang, Shuijing Liu, Gautham Bommannan, and Girish Chowdhary.
Robust deep reinforcement learning with adversarial attacks. In Proceedings of the 17th
International Conference on Autonomous Agents and MultiAgent Systems, pages 2040–2042,
2018. (p. 1.)

[37] Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial
reinforcement learning. In International Conference on Machine Learning, pages 2817–2826.
PMLR, 2017. (p. 1.)

[38] Aurko Roy, Huan Xu, and Sebastian Pokutta. Reinforcement learning under model mismatch.
Advances in neural information processing systems, 30, 2017. (p. 2.)

[39] Jay K Satia and Roy E Lave Jr.
Markovian decision processes with uncertain transition
probabilities. Operations Research, 21(3):728–740, 1973. (p. 1.)

[40] Yi Shen, Pan Xu, and Michael Zavlanos. Wasserstein distributionally robust policy evaluation
and learning for contextual bandits. Transactions on Machine Learning Research, 2024. ISSN
2835-8856. URL https://openreview.net/forum?id=NmpjDHWIvg. Featured Certifica-
tion. (p. 1.)

[41] Laixi Shi and Yuejie Chi. Distributionally robust model-based offline reinforcement learning
with near-optimal sample complexity. Journal of Machine Learning Research, 25(200):1–91,
2024. (pp. 1, 2, 3, and 6.)

[42] Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, and Yuejie Chi. The curious price
of distributional robustness in reinforcement learning with a generative model. Advances in
Neural Information Processing Systems, 36, 2024. (pp. 3, 8, and 40.)

[43] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul
Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for
autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 2446–2454, 2020. (p. 1.)

[44] Aviv Tamar, Shie Mannor, and Huan Xu. Scaling up robust mdps using function approximation.
In International conference on machine learning, pages 181–189. PMLR, 2014. (pp. 2 and 4.)

[45] Chen Tessler, Yonathan Efroni, and Shie Mannor. Action robust reinforcement learning and
applications in continuous control. In International Conference on Machine Learning, pages
6215–6224. PMLR, 2019. (p. 1.)

[46] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer, New York, 2009.
(p. 37.)

[47] Roman Vershynin. High-dimensional probability: An introduction with applications in data
science, volume 47. Cambridge university press, 2018. (p. 28.)

[48] He Wang, Laixi Shi, and Yuejie Chi. Sample complexity of offline distributionally robust linear
markov decision processes. arXiv preprint arXiv:2403.12946, 2024. (p. 4.)

13


---Page Break---
[49] Lu Wang, Wei Zhang, Xiaofeng He, and Hongyuan Zha. Supervised reinforcement learning
with recurrent neural network for dynamic treatment recommendation. In Proceedings of the
24th ACM SIGKDD international conference on knowledge discovery & data mining, pages
2447–2456, 2018. (p. 1.)

[50] Ruosong Wang, Dean Foster, and Sham M Kakade. What are the statistical limits of offline rl
with linear function approximation? In International Conference on Learning Representations,
2021. (p. 7.)

[51] Yue Wang and Shaofeng Zou. Online robust reinforcement learning with model uncertainty.
Advances in Neural Information Processing Systems, 34:7193–7206, 2021. (pp. 2 and 3.)

[52] Wolfram Wiesemann, Daniel Kuhn, and Berç Rustem. Robust markov decision processes.
Mathematics of Operations Research, 38(1):153–183, 2013. (pp. 1 and 3.)

[53] Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-
consistent pessimism for offline reinforcement learning. Advances in neural information
processing systems, 34:6683–6694, 2021. (pp. 1, 2, 6, and 16.)

[54] Wei Xiong, Han Zhong, Chengshuai Shi, Cong Shen, Liwei Wang, and Tong Zhang. Nearly
minimax optimal offline reinforcement learning with linear function approximation: Single-
agent mdp and markov game. In International Conference on Learning Representations (ICLR),
2023. (pp. 2, 3, 7, 8, 9, 10, 16, 28, and 32.)

[55] Huan Xu and Shie Mannor. The robustness-performance tradeoff in markov decision processes.
Advances in Neural Information Processing Systems, 19, 2006. (pp. 1 and 3.)

[56] Zaiyan Xu, Kishan Panaganti, and Dileep Kalathil. Improved sample complexity bounds
for distributionally robust reinforcement learning. In International Conference on Artificial
Intelligence and Statistics, pages 9728–9754. PMLR, 2023. (pp. 3 and 4.)

[57] Wenhao Yang, Liangyu Zhang, and Zhihua Zhang. Toward theoretical understandings of robust
markov decision processes: Sample complexity and asymptotics. The Annals of Statistics, 50
(6):3223–3248, 2022. (pp. 1, 3, and 4.)

[58] Wenhao Yang, Han Wang, Tadashi Kozuno, Scott M Jordan, and Zhihua Zhang. Robust markov
decision processes without model estimation. arXiv preprint arXiv:2302.01248, 2023. (p. 3.)

[59] Zhouhao Yang, Yihong Guo, Pan Xu, Anqi Liu, and Animashree Anandkumar. Distributionally
robust policy gradient for offline contextual bandits. In International Conference on Artificial
Intelligence and Statistics, pages 6443–6462. PMLR, 2023. (p. 1.)

[60] Ming Yin, Yaqi Duan, Mengdi Wang, and Yu-Xiang Wang. Near-optimal offline reinforcement
learning with linear representation: Leveraging variance information with pessimism. arXiv
preprint arXiv:2203.05804, 2022. (pp. 2, 7, 8, 10, and 16.)

[61] Pengqian Yu and Huan Xu. Distributionally robust counterpart in markov decision processes.
IEEE Transactions on Automatic Control, 61(9):2538–2543, 2015. (p. 3.)

[62] Andrea Zanette, Martin J Wainwright, and Emma Brunskill. Provable benefits of actor-critic
methods for offline reinforcement learning. Advances in neural information processing systems,
34:13626–13640, 2021. (pp. 2, 10, and 16.)

[63] Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane Boning, and Cho-
Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on state
observations. Advances in Neural Information Processing Systems, 33:21024–21037, 2020. (p.
1.)

[64] Wenshuai Zhao, Jorge Peña Queralta, and Tomi Westerlund. Sim-to-real transfer in deep
reinforcement learning for robotics: a survey. In 2020 IEEE symposium series on computational
intelligence (SSCI), pages 737–744. IEEE, 2020. (p. 1.)

[65] Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement
learning for linear mixture markov decision processes. In Conference on Learning Theory,
pages 4532–4576. PMLR, 2021. (pp. 3, 8, and 46.)

14


---Page Break---
[66] Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, PR Kumar, and Chao Tian. Natural actor-
critic for robust reinforcement learning with function approximation. Advances in neural
information processing systems, 36, 2024. (pp. 2 and 4.)

[67] Zhengqing Zhou, Zhengyuan Zhou, Qinxun Bai, Linhai Qiu, Jose Blanchet, and Peter Glynn.
Finite-sample regret bound for distributionally robust offline tabular reinforcement learning. In
International Conference on Artificial Intelligence and Statistics, pages 3331–3339. PMLR,
2021. (p. 3.)

15


---Page Break---
A
Additional Related Work

Offline Linear MDPs.
Our work focuses on the offline linear MDP setting where the nominal
transition kernel, from which the offline dataset is collected, admits the linear MDP structure.
Numerous works have studied the provable efficiency and statistical limits of algorithms under
this setting [15, 62, 53, 60, 54]. The most relevant study to ours is the recent work of [54], which
established the minimax optimality of offline linear MDPs. At the core of their analysis is an
advantage-reference technique designed for offline RL under linear function approximation, together
with a variance aware pessimism-based algorithm. However, the offline linear MDP setting still
remains understudied in the context of DRMDPs.

Transfer-Learning in Low Rank MDPs.
Besides the distributionally robust perspective to solve
the planning problem in a nearly unknown target environment, another line of work focuses on transfer
learning in low-rank MDPs [7, 23, 2, 5]. Specifically, the problem setup assumes that the agent has
access to information of several source tasks. The agent learns a common representation from the
source domains and then leverages the learned representation to learn a policy performing well in
the target tasks with limited information. This setting is in stark contrast to DRMDPs, where the
agent only has access to the information of a single source domain, without any available information
of the target domain, assuming the same task is being performed. This motivates the pessimistic
principle of the distributionally robust perspective. Among the aforementioned works, Bose et al. [5]
studied the offline multi-task RL, which is the most closely related to our setting. In particular, they
investigate the representation transfer error in their Theorem 1, stating that the learned representation
can lead to a transition kernel that is close to the target kernel in terms of the TV divergence. Note
that the uncertainty is induced by the representation estimation error, which is different from our
setting assuming that the uncertainty comes from perturbations on underlying factor distributions.
Nevertheless, this work provides evidence that TV divergence is a reasonable measure to quantify the
uncertainty in transition kernels and motivates a future research direction in learning robust policies
that are robust to the uncertainty induced by the representation estimation error.

B
A More Computationally Efficient Variant of VA-DRPVI

In this section, we propose a modified version of Algorithm 2, which reduces the computation cost in
the ridge regressions for variance estimation and achieves the same theoretical guarantees.

Variance Estimator.
In Section 5, we estimate the variance of the truncated robust value function
[bV

′ρ
h+1]α. Thus, for different α, we need to establish different variance estimators, which signifi-
cantly increases the computational burden. The theoretical analysis of Algorithm 2 suggests that it
suffices to estimate the the variance of bV

′ρ
h+1, instead of the truncated one. In particular, we know

[Varh bV

′ρ
h+1](s, a) = [P0
h(bV

′ρ
h+1)2](s, a) −([P0
h bV

′ρ
h+1](s, a))2 = ⟨ϕ(s, a), zh,2⟩−(⟨ϕ(s, a), zh,1⟩)2.
Then we estimate zh,1 and zh,2 via ridge regression:

˜zh,2 = argmin
z∈Rd

K
X

τ=1

  bV

′ρ
h+1(sτ
h+1)
2 −ϕτ⊤
h z
2 + λ∥z∥2
2,
(B.1a)

˜zh,1 = argmin
z∈Rd

K
X

τ=1

 bV

′ρ
h+1(sτ
h+1) −ϕτ⊤
h z
2 + λ∥z∥2
2.
(B.1b)

We construct the following truncated variance estimator:

bσ2
h(s, a) := max
n
1,

ϕ(s, a)⊤˜zh,2


[0,H2] −

ϕ(s, a)⊤˜zh,1
2
[0,H] −˜O
 dH3

√

Kκ

o
.
(B.2)

The modified variance-aware algorithm is presented in Algorithm 3 and the theoretical guarantee is
presented in Theorem B.1.

Theorem B.1. Under Assumptions 3.1 and 4.3, for K > max{ ˜O(d2H6/κ), ˜O(H4/κ2)} and δ ∈
(0, 1), if we set λ = 1/H2 and β2 = ˜O(
√

d) in Algorithm 3, then with probability at least 1 −δ, for

16


---Page Break---
Algorithm 3 Modified VA-DRPVI

Require: Input dataset D, D′ and β2; bV ρ
H+1(·) = 0

1: Run Algorithm 1 using dataset D′ to get {bV

′ρ
h }h∈[H]
2: for h = H, · · · , 1 do
3:
Construct variance estimator bσ2
h(·, ·) using D′ by (B.1) and (B.2)

4:
Σh = PK
τ=1 ϕτ
hϕτ⊤
h /bσ2
h(sτ
h, aτ
h) + λI

5:
ˆzh(α) = Σ−1
h
 PK
τ=1 ϕτ
h
bV ρ
h+1(sτ
h+1)


α/bσ2
h(sτ
h, aτ
h)


6:
αi = argmaxα∈[0,H]{ˆzh,i(α) −ρ(α −mins′[bV ρ
h+1(s′)]α)}, ∀i ∈[d]

7:
ˆνρ
h,i = ˆzh,i(αi) −ρ(αi −mins′[bV ρ
h+1(s′)]αi), ∀i ∈[d]

8:
Γh(·, ·) ←β2
Pd
i=1 ∥ϕi(·, ·)1i∥Σ−1
h
9:
bQρ
h(·, ·) = {ϕ(·, ·)⊤(θh + ˆνρ
h) −Γh(·, ·)}[0,H−h+1]
10:
ˆπh(·|·) ←argmaxπh⟨bQρ
h(·, ·), πh(·|·)⟩A, bV ρ
h (·) ←⟨bQρ
h(·, ·), ˆπh(·|·)⟩A
11: end for

all s ∈S, the suboptimality of VA-DRPVI satisfies

SubOpt(ˆπ, s, ρ) ≤β2 ·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Σ⋆−1
h
|s1 = s
i
,
(B.3)

where Σ⋆
h = PK
τ=1 ϕτ
hϕτ⊤
h /[VhV ⋆
h+1](sτ
h, aτ
h) + λI.
Remark B.2. The computation cost of Algorithm 3 is much smaller than Algorithm 2, as the
variance estimators are not related to α anymore. Notably, Algorithm 3 shares the same upper bound
as Algorithm 2. According to Theorem 6.1, we know the modified algorithm is also minimax optimal.

C
Experiments

We conduct numerical experiments to illustrate the performances of our proposed algorithms, DRPVI
and VA-DRPVI, and compare it with the their non-robust counterpart, PEVI [15]. All numerical exper-
iments were conducted on a MacBook Pro with a 2.6 GHz 6-Core Intel CPU. The implementation of
our DRPVI algorithm is available at https://github.com/panxulab/Offline-Linear-DRMDP.

Construction of the simulated linear MDP
We leverage the simulated linear MDP setting pro-
posed by Liu and Xu [20] and modify it as an offline RL problem. In particular, the source and
target linear MDP environment are shown in Figure 1(a) and Figure 1(b). The state space is set to
be S = {x1, · · · , x5} and the action space is to be A = {−1, 1}4 ⊂R4. At each episode, the state
always starts with x1, and then transits to x2, x4, x5 with probability defined in the figures. x2 is an
intermediate state, and it can transit to x3, x4, x5 with probability defined on the lines. Moreover,
Both x4 and x5 are absorbing states. x4 (x5) is the fail state (goal state), and the reward starting from
which is always 0 (1). The reward functions and transition probabilities are designed to depend on
the hyperparameter ξ ∈R4 as shown in the figure. The target environment is constructed by only
perturbing the transition probability at x1 of the source environment, and the extend of perturbation
is controlled by the hyperparameter q ∈(0, 1). We refer more details on the construction of the
simulated linear DRMDP to the Supplementary A.1 of [20].

Implementation
We simply use the random policy that chooses actions uniformly at random
at any (s, a, h) ∈S × A × [H] to collect offline dataset. The offline dataset containing 100
trajectories collected by the behavior policy from the source environment. We conduct ablation
study by setting the hyperpameter ξ = (1/∥ξ∥1, 1/∥ξ∥1, 1/∥ξ∥1, 1/∥ξ∥1)⊤and consider different
choices of ∥ξ∥1 ∈{0.1, 0.2, 0.3}. Following [20], we use heterogeneous uncertainty level for our
two algorithms. Specifically, we set ρ1,4 = 0.5 and ρh,i = 0 for all other cases. The experiment
results are shown in Figure 2.

Figure 2 shows the performances of the learned policies of three algorithms. We conclude that both of
our proposed algorithms are robust to environmental perturbation compared to the non-robust PEVI.

17


---Page Break---
x1
x2
x3

x4

x5

(1 −p)(1 −δ −⟨ξ, a⟩)

p(1 −δ −⟨ξ, a⟩)

δ + ⟨ξ, a⟩

(1 −p)(1 −δ −⟨ξ, a⟩)

p(1 −δ −⟨ξ, a⟩)

δ + ⟨ξ, a⟩

1 −δ −⟨ξ, a⟩

δ + ⟨ξ, a⟩

1

1

(a) The source MDP environment.

x1
x2
x3

x4

x5

(1 −δ −⟨ξ, a⟩)

q(δ + ⟨ξ, a⟩)

(1 −q)(δ + ⟨ξ, a⟩)

(1 −p)(1 −δ −⟨ξ, a⟩)

p(1 −δ −⟨ξ, a⟩)

δ + ⟨ξ, a⟩

1 −δ −⟨ξ, a⟩

δ + ⟨ξ, a⟩

1

1

(b) The target MDP environment.

Figure 1: The source and the target linear MDP environments. The value on each arrow represents
the transition probability. For the source MDP, there are five states and three steps, with the initial
state being x1, the fail state being x4, and x5 being an absorbing state with reward 1. The target MDP
on the right is obtained by perturbing the transition probability at the first step of the source MDP,
with others remaining the same.

Furthermore, VA-DRPVIslightly outperforms DRPVI in most settings. These numerical results are
consistent with our theoretical findings.

D
Proof of Theorem 4.4

Our analysis mainly deals with the challenges induced by the model uncertainty, infP ∈Uρ(P 0), and
the need to maximally exploit the information in the offline dataset. More specifically, the proof of
Theorem 4.4 mainly constitutes of two steps.

Step 1: suboptimality decomposition.
We first decompose the suboptimality gap in the following
lemma to connect it with the estimation error, the full proof of which can be found in Appendix G.1.

Lemma D.1 (Suboptimality Decomposition for DRMDP). If the following holds

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −ϕ(s, a)ˆνρ
h
 ≤Γh(s, a), ∀(s, a, h) ∈S × A × [H],
(D.1)

then we have SubOpt(ˆπ, s, ρ) ≤2 supP ∈Uρ(P 0)
PH
h=1 Eπ⋆,P 
Γh(sh, ah)|s1 = s

.

The main challenge in deriving Lemma D.1 lies in the dependency of the robust Bellman equation
(3.1) on the nominal kernel P 0, which is not linear and does not even have an explicit form. It
should be noted that the term
 infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −ϕ(s, a)⊤ˆνρ
h
 in condition
(D.1) stands for the estimation error of the estimated robust Q-function in (4.5), which we refer to
as the robust estimation uncertainty. Lemma D.1 shows that under the condition that the robust
estimation uncertainty is bounded by Γh(s, a), the suboptimality gap can be upper bounded in terms
of Γh(s, a). To conclude the proof, it remains to derive Γh(s, a) and then substitute it back into the
result in Lemma D.1.

Step 2: bounding the robust estimation uncertainty.
We now bound the robust estimation uncer-
tainty in Lemma D.1 by the following result, the full proof of which can be found in Appendix G.2.

18


---Page Break---
0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

Average reward

PEVI
DRPVI
VA-DRPVI

(a) ∥ξ∥1 = 0.1, ρ1,4 = 0.3

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

Average reward

PEVI
DRPVI
VA-DRPVI

(b) ∥ξ∥1 = 0.1, ρ1,4 = 0.4

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

Average reward

PEVI
DRPVI
VA-DRPVI

(c) ∥ξ∥1 = 0.1, ρ1,4 = 0.5

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

1.4

1.5

Average reward

PEVI
DRPVI
VA-DRPVI

(d) ∥ξ∥1 = 0.2, ρ1,4 = 0.3

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

1.4

1.5

Average reward

PEVI
DRPVI
VA-DRPVI

(e) ∥ξ∥1 = 0.2, ρ1,4 = 0.4

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

0.9

1.0

1.1

1.2

1.3

1.4

1.5

Average reward

PEVI
DRPVI
VA-DRPVI

(f) ∥ξ∥1 = 0.2, ρ1,4 = 0.5

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

1.0

1.2

1.4

1.6

Average reward

PEVI
DRPVI
VA-DRPVI

(g) ∥ξ∥1 = 0.3, ρ1,4 = 0.3

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

1.0

1.2

1.4

1.6

Average reward

PEVI
DRPVI
VA-DRPVI

(h) ∥ξ∥1 = 0.3, ρ1,4 = 0.4

0.0
0.2
0.4
0.6
0.8
1.0
Perturbation

0.8

1.0

1.2

1.4

1.6

Average reward

PEVI
DRPVI
VA-DRPVI

(i) ∥ξ∥1 = 0.3, ρ1,4 = 0.5

Figure 2: Simulation results under different source domains. The x-axis represents the perturbation
level corresponding to different target environments. ρ1,4 is the input uncertainty level for our
VA-DRPVI algorithm. ∥ξ∥1 is the hyperparameter of the linear DRMDP environment.

Lemma D.2 (Robust Estimation Uncertainty Bound). For any sufficiently large sample size K
satisfying K > max{512 log(2dH2/δ)/κ2, 20449d2H2/κ}, and any fixed δ ∈(0, 1), if we set
λ = 1 in Algorithm 1, then with probability at least 1 −δ, for all (s, a, h) ∈S × A × [H], we have


inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −ϕ(s, a)⊤ˆνρ
h
 ≤Γh(s, a),
(D.2)

where Γh(s, a) = 4
√

dH√ι Pd
i=1 ∥ϕi(s, a)1i∥Λ−1
h
and ι = log(2dH2K/δ).

Γh(s, a) provides an explicit bound for the robust estimation uncertainty, which also serves as the
penalty term in Line 6 of Algorithm 1. The main challenge of deriving Lemma D.2 lies in inferring
the worst-case behavior using information merely from the nominal environment. Our idea is to
first transform the robust estimation uncertainty to the estimation uncertainty of ridge regressions
(4.2) on the nominal model P 0, where the samples are collected and statistical control is available.
We then adopt a reference-advantage decomposition technique, which is new in the linear DRMDP
literature, to further decompose the estimation uncertainty on the nominal model into the reference
uncertainty and the advantage uncertainty. The remaining proof is to bound the reference uncertainty
and advantage uncertainty respectively using concentration and union bound arguments under an
induction framework to address the temporal dependency. We highlight that all these arguments are
specifically designed for the unique problem of DRMDP, which is novel and nontrivial.

19


---Page Break---
E
Proof of the Suboptimality Upper Bounds

In this section, we prove the main results in Corollary 4.5, Remark 4.6, Theorem 5.2, and Corollary 5.3,
which give out the instance-dependent upper bounds of the proposed algorithms. Before the proof,
we introduce some useful notations. For any function f : S →[0, H −1], define

\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Phf](s, a) := ϕ(s, a)⊤ˆνρ
h(f),
(E.1)

where for each i ∈[d], we have

ˆνρ
h,i(f) = max
α∈[0,H]

n
ˆEµ0
h,i[f(s)]α −ρ(α −min
s′∈S[f(s′)]α)
o
,

ˆEµ0
h,i[f(s)]α =
h
Λ−1
h

K
X

τ=1
ϕτ
h[f(sτ
h+1)]α
i

i.

E.1
Proof of Corollary 4.5

The proof of Corollary 4.5 is straightforward given our result in Theorem 4.4.

Proof. Define ˜Λh = Eπb,P 0[ϕ(sh, ah)ϕ(sh, ah)⊤], ∀h ∈[H]. By Assumption 4.3, we have ˜Λh ⪰
κ · I. We further bound (6.1) as follows,

sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P

d
X

i=1
∥ϕi(sh, ah)1i∥Λ−1
h

s1 = s


≤
sup
P ∈Uρ(P 0)

2
√

K
Eπ⋆,P
 H
X

h=1

d
X

i=1
∥ϕi(sh, ah)1i∥˜Λ−1
h

s1 = s

(E.2)

=
sup
P ∈Uρ(P 0)

2
√

K
Eπ⋆,P
 H
X

h=1

d
X

i=1
ϕi(s, a)
q

1⊤
i ˜Λ−1
h 1i
s1 = s


≤
sup
P ∈Uρ(P 0)

2
√

K
Eπ⋆,P
 H
X

h=1

d
X

i=1
ϕi(s, a)
q

λmax( ˜Λ−1
h )
s1 = s

(E.3)

=
sup
P ∈Uρ(P 0)

2
√

K
Eπ⋆,P
 H
X

h=1

d
X

i=1
ϕi(s, a)

s

1
λmin( ˜Λh)

s1 = s


≤
sup
P ∈Uρ(P 0)

2
√

K
Eπ⋆,P
 H
X

h=1

r

1
κ


(E.4)

=
2H
√

K · κ
,

where (E.2) is due to Lemma I.3, (E.3) is due to the fact that for any matrix A, λmin ≤Aii ≤λmax,
where Aii is the i-th diagonal element of A. (E.3) holds due to Assumption 4.3 and the fact that
Pd
i=1 ϕi(s, a) = 1. We conclude the proof by invoking Theorem 4.4.

E.2
Proof of Theorem 5.2

The proof idea is similar to that of Theorem 4.4, except that we additionally analyze the variance
estimation and apply the Bernstein-type self-normalized concentration inequality to bound the
reference uncertainty, which is the dominant term. We start from analyzing the estimation error of
conditional variances in the following lemma.

Lemma E.1. Under Assumptions 3.1 and 4.3, when K ≥˜O(H4/κ2), then with probability at least
1 −δ, for all (s, a, h) ∈S × A × [H] and any fixed α, we have


Vh[V ⋆,ρ
h+1]α

(s, a) −˜O
 dH3

√

Kκ


≤bσ2
h(s, a; α) ≤

Vh[V ⋆,ρ
h+1]α

(s, a).

20


---Page Break---
The following lemma bounds the estimation error by reference-advantage decomposition.
Lemma E.2 (Variance-Aware Reference-Advantage Decomposition). There exist {αi}i∈[d], where
αi ∈[0, H], ∀i ∈[d], such that

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Σ−1
h (αi)
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Σ−1
h (αi)
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
iv

,

where ητ
h([f]αi) =
 
P0
h[f]αi

(sτ
h, aτ
h) −[f(sτ
h+1)]αi

, for any function f : S →[0, H −1].

Now we are ready to prove Theorem 5.2

Proof of Theorem 5.2. To prove this theorem, we bound the estimation error by Γh(s, a), then invoke
Lemma D.1 to get the result. First, we bound terms i-iv in Lemma E.2 to deduce Γh(s, a) at each
step h ∈[H], respectively.

Bound i and iii:
We set λ = 1/H2 to ensure that for all (s, a, h) ∈S × A × [H], we have

i + iii ≤
√

λ
√

dH

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi) =
√

d

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi).
(E.5)

Bound ii:
For all (s, a, α) ∈S × A × [0, H], by definition we have bσh(s, a; α) ≥1. Thus, for
all (h, τ, i) ∈[H] × [K] × [d], we have |ητ
h([V ⋆,ρ
h+1]αi)/bσh(sτ
h, aτ
h, αi)| ≤H. Note that V ⋆,ρ
H+1 is
independent of D, we can directly apply Bernstein-type self-normalized concentration inequality
Lemma I.2 and a union bound to obtain the upper bound. In concrete, we define the filtration
Fτ−1,h = σ({(sj
h, aj
h)}τ
j=1 ∪{sj
h+1}τ−1
j=1). Since V ⋆,ρ
h+1 and bσh(s, a; α) are independent of D, thus
ητ
h([V ⋆,ρ
h+1]αi)/bσh(sτ
h, aτ
h, αi) is mean-zero conditioned on the filtration Fτ−1,h. Further, we have

E
h ητ
h([V ⋆,ρ
h+1]αi)
bσh(sτ
h, aτ
h; αi)

2Fτ−1,h
i
= [Var[V ⋆,ρ
h+1]αi](sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h; αi)
(E.6)

≤[V[V ⋆,ρ
h+1]αi](sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h; αi)

= [V[V ⋆,ρ
h+1]αi](sτ
h, aτ
h) −˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h; αi)
+
˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h; αi)

≤1 +
˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h; αi) −˜O(dH3/
√

Kκ)
(E.7)

≤1 + 2 ˜O
 dH3

√

Kκ


,
(E.8)

21


---Page Break---
where (E.6) holds by the fact that bσ2
h(·, ·; ·) is independent of D and (sτ
h, aτ
h) is Fτ−1,h measurable.
(E.7) holds by Lemma E.1, and (E.8) holds by setting K ≥˜Ω(d2H6/κ) such that bσ2
h(sτ
h, aτ
h; αi) −
˜O(dH3/
√

Kκ) ≥1 −˜O(dH3/
√

Kκ) ≥1/2. Further, by (E.8), our choice of K also ensures that
E
 
ητ
h([V ⋆,ρ
h+1]αi)
2|Fτ−1,h

= O(1). Then by Lemma I.2, we have


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
≤˜O(
√

d).

This implies

ii ≤˜O(
√

d)

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi).
(E.9)

Bound iv:
Following the same induction analysis procedure, we have ∥[bV ρ
h+1]αi −[V ⋆,ρ
h+1]αi∥≤
˜O(
√

dH2/
√

Kκ). Then, using standard ϵ-covering number argument and Lemma I.1, we have

iv ≤˜O
d3/2H2

√

Kκ


d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi).
(E.10)

To make it non-dominant, we require K ≥˜Ω(d2H4/κ). By Lemma E.1, for any α ∈[0, H], we have

bσ2
h(sτ
h, aτ
h; α) ≤[Vh[V ⋆,ρ
h+1]α](sτ
h, aτ
h) ≤[VhV ⋆,ρ
h+1](sτ
h, aτ
h),

this implies that
 K
X

τ=1

ϕτ
hϕτ⊤
h
bσ2
h(sτ
h, aτ
h; αi) + λI
−1
⪯
 K
X

τ=1

ϕτ
hϕτ⊤
h
[VhV ⋆,ρ
h+1](sτ
h, aτ
h) + λI
−1
:= Σ⋆−1
h
.

Combining (E.5), (E.9) and (E.10), we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤˜O(
√

d)

d
X

i=1
∥ϕi(s, a)1i∥Σ⋆−1
h
.

Define Γh(s, a) = ˜O(
√

d) Pd
i=1 ∥ϕi(s, a)1i∥Σ⋆−1
h
, we concludes the proof by invoking Lemma D.1.

E.3
Proof of Corollary 5.3

In this section, we prove Corollary 5.3. We start with an interesting phenomenon, we call ‘range
shrinkage’, stated in the following lemma.
Lemma E.3 (Range Shrinkage). For any (ρ, π, h) ∈(0, 1] × Π × [H], we have

max
s∈S V π,ρ
h
(s) −min
s∈S V π,ρ
h
(s) ≤1 −(1 −ρ)H−h+1

ρ
.
(E.11)

Proof of Corollary 5.3. By the fact that the variance of a random variable can be upper bounded by
the square of its range and Lemma E.3, for all (s, a, h) ∈S × A × [H], we have

[VV ⋆
h+1](s, a) ≤
1 −(1 −ρ)H−h+1

ρ

2
≤
1 −(1 −ρ)H

ρ

2
.

Then we have

K
X

τ=1

ϕτ
hϕτ⊤
h
[VhV ⋆
h+1](sτ
h, aτ
h) + 1

H2 I ⪰

K
X

τ=1

ϕτ
hϕτ⊤
h
( 1−(1−ρ)H

ρ
)2 + 1

H2 I.

22


---Page Break---
Thus we have

Σ⋆−1
h
=
 K
X

τ=1

ϕτ
hϕτ⊤
h
[VhV ⋆
h+1](sτ
h, aτ
h) + 1

H2 I
−1
⪯
1 −(1 −ρ)H

ρ

2 K
X

τ=1
ϕτ
hϕτ⊤
h
+ 1

H2 I
−1
.

By Theorem 5.2, we have

SubOpt(ˆπ, s, ρ) ≤˜O(
√

d) ·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Σ⋆−1
h
s1 = s
i

≤˜O(
√

d) · 1 −(1 −ρ)H

ρ
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Λ−1
h
s1 = s
i
.

This concludes the proof.

F
Proof of the Information-Theoretic Lower Bound

In this section, we prove the information-theoretic lower bound. We first introduce the construction of
hard instances in Appendix F.1, then we prove Theorem 6.1 in Appendix F.2, and prove Corollary 6.2
in Appendix F.3.

F.1
Construction of Hard Instances

We design a family of d-rectangular linear DRMDPs parameterized by a Boolean vector ξ =
{ξh}h∈[H], where ξh ∈{−1, 1}d. For a given ξ and uncertainty level ρ ∈(0, 3/4], the corresponding
d-rectangular linear DRMDP M ρ
ξ has the following structure. The state space S = {x1, x2} and the
action space A = {0, 1}d. The initial state distribution µ0 is defined as

µ0(x1) = d + 1

d + 2
and
µ0(x2) =
1
d + 2.

The feature mapping ϕ : S × A →Rd+2 is defined as

ϕ(x1, a)⊤=
a1

d , a2

d , · · · , ad

d , 1 −

d
X

i=1

ai

d , 0


ϕ(x2, a)⊤=
 
0, 0, · · · , 0, 0, 1

,

which satisfies ϕi(s, a) ≥0 and Pd
i=1 ϕi(s, a) = 1. The factor distributions {µh}h∈[H] are defined
as

µ⊤
h =
 
δx1, δx1, · · · , δx1, δx1
|
{z
}
d + 1

, δx2

, ∀h ∈[H],

so the transition is homogeneous and does not depend on action but only on state. The reward
parameters {θh}h∈[H] are defined as

θ⊤
h = δ ·
ξh1 + 1

2
, ξh2 + 1

2
, · · · , ξhd + 1

2
, 1

2, 0

, ∀h ∈[H],

where δ is a parameter to control the differences among instances, which is to be determined
later. The reward rh is generated from the normal distribution rh ∼N(rh(sh, ah), 1), where
rh(s, a) = ϕ(s, a)⊤θh. Note that

rh(x1, a) = ϕ(x1, a)⊤θh = δ

2d
 
⟨ξh, a⟩+ d

≥0
and
rh(x2, a) = ϕ(x2, a)⊤θh = 0, ∀a ∈A,

which means that x2 is a worst state in terms of the mean reward. Thus, the worst case transition
kernel should have the highest possible transition probability to x2. This construction is pivotal in
achieving a concise expression of robust value function. Further, we only consider model uncertainty

23


---Page Break---
x1
x2
1
1

(a) The nominal environment.

x1
x2
1 −ρ

ρ

1

(b) The worst case transition at the first step.

Figure 3: The nominal environment and the worst case environment. The value on each arrow
represents the transition probability. The MDP has two states and H steps. For the nominal
environment, both x1 and x2 are absorbing states, which means that the state will always stay at
the initial state in the nominal environment. The worst case environment on the right is obtained by
perturbing the transition probability at the first step of the nominal environment, with others remain
the same.

in the first step. By the fact that x2 is the worse state, we know the worst case factor distribution for
the first step is

ˇµ⊤
1 =
 
(1 −ρ)δx1 + ρδx2, (1 −ρ)δx1 + ρδx2, · · · , (1 −ρ)δx1 + ρδx2, (1 −ρ)δx1 + ρδx2, δx2

.

We illustrate the designed d-rectangular linear DRMDP M ρ
ξ in Figure 3(a) and Figure 3(b).

Finally, we design the procedure for collecting the offline dataset. We assume the K trajectories are
collected by a behavior policy πb = {πb
h}h∈[H] defined as

πb
h ∼Unif
 
{e1, · · · , ed, 0}

, ∀h ∈[H],

where {ei}i∈[d] are the canonical basis vectors in Rd. The initial state is generated according to µ0.
It is straightforward to check that the constructed hard instances satisfy Assumption 4.3. We denote
the offline dataset as D.

F.2
Proof of Theorem 6.1

With this family of hard instances, we are ready to prove the information-theoretic lower bound.
First, we define some notations. For any ξ ∈{−1, 1}dH, let Qξ denote the distribution of dataset D
collected from the MDP Mξ. Denote the family of parameters as Ω= {−1, 1}dH and the family of
hard instances as M = {Mξ : ξ ∈Ω}.

Proof of Theorem 6.1. The proof constitutes three steps. In the first step, we lower bound the minimax
suboptimality gap by testing error in the following Lemma F.1, the full proof of which can be found
in Appendix G.6.
Lemma F.1 (Reduction to testing). For the given family of d-rectangular linear DRMDPs, we have

inf
ˆπ
sup
M∈M
SubOpt(ˆπ, x1, ρ) ≥(1 −ρ) · δdH

8d ·
min
ξ,ξ′∈Ω
DH(ξ,ξ′)=1

inf
ψ

h
Qξ(ψ(D) ̸= ξ) + Qξ′(ψ(D) ̸= ξ′)
i
,

(F.1)

where for fixed indices ξ and ξ′, ψ is any test function taking value in {ξ, ξ′}.

In the second step, we lower bound the testing error on the right hand side of (F.1) in the following
Lemma F.2, the full proof of which can be found in Appendix G.7.
Lemma F.2 (Lower bound on testing error). For the given family of d-rectangular linear DRMDPs,
let δ = d3/2/
√

2K, then we have

min
ξ,ξ′

DH(ξ,ξ′)=1

inf
ψ

h
Qξ(ψ(D) ̸= ξ) + Qξ′(ψ(D) ̸= ξ′)
i
≥1

2.

By Lemma F.1 and Lemma F.2, we have

inf
ˆπ
sup
M∈M
SubOpt(ˆπ, x1, ρ) ≥d3/2H

128
√

K
.
(F.2)

In the last step, we upper bound the uncertainty function Φ(Σ⋆
h, s) in the following Lemma F.3, the
full proof of which can be found in Appendix G.8.

24


---Page Break---
Lemma F.3. For all Mξ ∈M, when K ≥˜O(d4), then with probability at least 1 −δ, we have

sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Σ⋆−1
h
s1 = x1
i
≤4d3/2H
√

K
.

By Lemma F.3 and (F.2), we know that with probability at least 1 −δ, there exist a universal constant
c, such that

inf
ˆπ
sup
M∈M
SubOpt(ˆπ, x1, ρ) ≥c ·
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Σ⋆−1
h
|s1 = x1
i
.

This concludes the proof.

F.3
Proof of Corollary 6.2

Proof. The result in Corollary 6.2 directly follows from the fact shown in (G.38): for the constructed
hard instances, we have Σ⋆
h = Λh. Thus, we complete the proof by directly substituting Σ⋆
h in the
result of Theorem 6.1 by Λh.

G
Proof of Technical Lemmas

G.1
Proof of Lemma D.1

Proof. First, we decompose SubOpt(ˆπ, s, ρ) as follows

SubOpt(ˆπ, s, ρ) = V π⋆,ρ
1
(s) −bV ρ
1 (s)
|
{z
}
I

+ bV ρ
1 (s) −V ˆπ,ρ
1
(s)
|
{z
}
II

,

then we bound term I and term II, respectively.

Bounding term I:
Note that

V π⋆,ρ
h
(s) −bV ρ
h (s) = Qπ⋆,ρ
h
(s, π⋆
h(s)) −bQρ
h(s, ˆπh(s))

= Qπ⋆,ρ
h
(s, π⋆
h(s)) −bQρ
h(s, π⋆
h(s)) + bQρ
h(s, π⋆
h(s)) −bQρ
h(s, ˆπh(s))

≤Qπ⋆,ρ
h
(s, π⋆
h(s)) −bQρ
h(s, π⋆
h(s)).
(G.1)

Here (G.1) holds by the fact that ˆπh(s) is the greedy policy corresponding to bQρ
h(s, a), which leads
to bQρ
h(s, π⋆
h(s)) −bQρ
h(s, ˆπh(s)) ≤0. Further, by the robust Bellman equation (3.1), we have

Qπ⋆,ρ
h
(s, π⋆
h(s)) −bQρ
h(s, π⋆
h(s))

= rh(s, π⋆
h(s)) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV π⋆,ρ
h+1 ](s, π⋆
h(s)) −bQρ
h(s, π⋆
h(s))

= rh(s, π⋆
h(s)) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV π⋆,ρ
h+1 ](s, π⋆
h(s)) −rh(s, π⋆
h(s))

−
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆
h(s)) + rh(s, π⋆
h(s)) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆
h(s))

−bQρ
h(s, π⋆
h(s)).

To proceed, we define the robust Bellman update error as follows

ζρ
h(s, a) = rh(s, a) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −bQρ
h(s, a),

and denote the worst case transition kernel with respect to the estimated robust value function as
bP = { bPh}h∈[H], where bPh(·|s, a) = arg infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a), ∀(s, a) ∈S × A.
Then we have

Qπ⋆,ρ
h
(s, π⋆
h(s)) −bQρ
h(s, π⋆
h(s))

25


---Page Break---
=
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV π⋆,ρ
h+1 ](s, π⋆
h(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆
h(s)) + ζρ
h(s, π⋆
h(s))

≤
bPh(V π⋆,ρ
h+1 −bV ρ
h+1)

(s, π⋆
h(s)) + ζρ
h(s, π⋆
h(s)).
(G.2)

Combining (G.1) and (G.2), we have for any h ∈[H],

V π⋆,ρ
h
(s) −bV ρ
h (s) ≤
bPh(V π⋆,ρ
h+1 −bV ρ
h+1)

(s, π⋆
h(s)) + ζρ
h(s, π⋆
h(s)).
(G.3)

Recursively applying (G.3), we have

V π⋆,ρ
1
(s) −bV ρ
1 (s) ≤

H
X

h=1
Eπ⋆, b
P 
ζρ
h(sh, ah)|s1 = s

.
(G.4)

Bounding term II:
Note that bV ρ
h (s) −V ˆπ,ρ
h
(s) = bQρ
h(s, ˆπh(s)) −Qˆπ,ρ
h (s, ˆπh(s)), by the robust
Bellman equation (3.1), we have

bV ρ
h (s) −V ˆπ,ρ
h
(s)

= bQρ
h(s, ˆπh(s)) −rh(s, ˆπh(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, ˆπh(s))

+
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, ˆπh(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ˆπ,ρ
h+1](s, ˆπh(s))

= −ζρ
h(s, ˆπh(s)) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, ˆπh(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ˆπ,ρ
h+1](s, ˆπh(s)).

To proceed, we denote the worst case transition kernel with respect to the robust value function of ˆπ
as P ˆπ = {P ˆπ
h }h∈[H], where P ˆπ
h (·|s, a) = arg infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ˆπ,ρ
h+1](s, a), then we have

bV ρ
h (s) −V ˆπ,ρ
h
(s) ≤−ζρ
h(s, ˆπh(s)) +

Pˆπ
h(bV ρ
h+1 −V ˆπ,ρ
h+1)

(s, ˆπh(s)).
(G.5)

Applying (G.5) recursively, we have

bV ρ
1 (s) −V ˆπ,ρ
1
(s) ≤

H
X

h=1
Eˆπ,P ˆπ
−ζρ
h(sh, ah)|s1 = s

.
(G.6)

Now it remains to bound the robust Bellman error ζρ
h(·, ·). In particular, we aim to show that for all
(s, a, h) ∈S × A × [H],
0 ≤ζρ
h(s, a) ≤2Γh(s, a).

Note that ζρ
h(s, a) = rh(s, a) + infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −bQρ
h(s, a). Recall the defi-

nition of bQρ
h(s, a) in Algorithm 1 and the notation in (E.1), and we have

ζρ
h(s, a) = rh(s, a) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

−max
n
rh(s, a) +
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −Γh(s, a), 0
o
.

If rh(s, a) + c
infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −Γh(s, a) ≤0, then ζρ
h(s, a) = rh(s, a) +

infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) ≥0. If rh(s, a) + c
infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −

Γh(s, a) > 0, then we have ζρ
h(s, a) = rh(s, a)+infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)−rh(s, a)−
c
infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) + Γh(s, a) ≥−Γh(s, a) + Γh(s, a) = 0, where we used the
condition in (D.1). In conclusion, we have ζρ
h(s, a) ≥0.

On the other hand, we always have

ζρ
h(s, a) ≤
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) + Γh(s, a)

≤2Γh(s, a).

26


---Page Break---
Thus, for all (s, a, h) ∈S × A × [H], we have

0 ≤ζρ
h(s, a) ≤2Γh(s, a).
(G.7)

Combining (G.4), (G.6) and (G.7), we have

SubOpt(ˆπ, s, ρ) ≤

H
X

h=1
Eπ⋆, b
P 
ζh(sh, ah)|s1 = s

+

H
X

h=1
Eˆπ,P ˆπ
−ζρ
h(sh, ah)|s1 = s


≤

H
X

h=1
Eπ⋆, b
P 
ζρ
h(sh, ah)|s1 = s


≤
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P 
ζρ
h(sh, ah)|s1 = s


≤2
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P 
Γh(sh, ah)|s1 = s

.

This concludes the proof.

G.2
Proof of Lemma D.2

In this section, we prove Lemma D.2. Before the proof, we first present several auxiliary lemmas.
Lemma G.1 (Reference-Advantage Decomposition). There exist real values {αi}i∈[d], where αi ∈
[0, H], ∀i ∈[d], such that

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Λ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]αi)

Λ−1
h
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h

Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Λ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)

Λ−1
h
|
{z
}
iv

,

where ητ
h([f]αi) =
 
P0
h[f]αi

(sτ
h, aτ
h) −[f(sτ
h+1)]αi

, for any function f : S →[0, H −1].

Lemma G.2 (Bound of Weights). For any h ∈[H], denote the weight wρ
h = θh + ˆνρ
h in Algorithm 1,
then wρ
h satisfies

∥wρ
h∥2 ≤2H
p

dK/λ.

Lemma G.3. [15, Lemma B.2] Let f : S →[0, R −1] be any fixed function. For any δ ∈(0, 1),
we have

P


K
X

τ=1
ϕτ
h · ητ
h(f)

2

Λ−1
h
≥R2
2 log
1

δ


+ d log

1 + K

λ


≤δ.

Lemma G.4 (Covering number of function class Vh). For any h ∈[H], let Vh denote a class of
functions mapping from S to R with the following parametric form

Vh(s) = max
a∈A

n
ϕ(s, a)⊤θ −β

d
X

i=1

q

ϕi(s, a)1⊤
i Σ−1
h ϕi(s, a)1i
o

[0,H−h+1],

27


---Page Break---
where the parameters (θ, β, Σh) satisfy ∥θ∥≤L, β ∈[0, B], λmin(Σh) ≥λ. Assume ∥ϕ(s, a)∥≤
1 for all (s,a) pairs, and let Nh(ϵ) be the ϵ-covering number of V with respect to the distance
dist(V1, V2) = supx |V1(x) −V2(x)|. Then

log Nh(ϵ) ≤d log(1 + 4L/ϵ) + d2 log

1 + 8d1/2B2/(λϵ2)

.

Lemma G.5. [47, Covering number of an interval] Denote the ϵ-covering number of the closed
interval [a, b] for some real number b > a with respect to the distance metric d(α1, α2) = |α1 −α2|
as Nϵ([a, b]). Then we have Nϵ([a, b]) ≤3(b −a)/ϵ.

Proof of Lemma D.2. To prove this lemma, we bound terms i-iv in Lemma G.1 at each step h ∈[H],
respectively. To deal with the temporal dependency, we follow the induction procedure proposed in
[54] and make essential adjustments to adapt to the robust setting.

The base case.
We start from the last step H. By the fact that any robust value function is upper
bounded by H, then with λ = 1, for all (s, a) ∈S × A, we have

i + iii ≤2H

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h .
(G.8)

Next, we bound term ii. Note that V ⋆,ρ
H+1 is independent of D, we can directly apply Hoeffding-type
self-normalized concentration inequality Lemma I.1 and a union bound to obtain the upper bound.
In concrete, we define the filtration Fτ−1,h = σ({(sj
h, aj
h)}τ
j=1 ∪{sj
h+1}τ−1
j=1). Since V ⋆,ρ
H+1 is
independent of D and is upper bounded by H, thus we have ητ
H([V ⋆,ρ
H+1]αi)|Fτ−1,H is mean zero,
i.e., E[ητ
H([V ⋆,ρ
H+1]αi)|Fτ−1,H] = 0 and H-subGaussian. By Lemma I.1, for any fixed index i ∈[d],
with probability at least 1 −δ/2dH2, we have



K
X

τ=1
ϕτ
Hητ
H([V ⋆,ρ
H+1]αi)

2

Λ−1
h
≤2H2 log
2dH2 det(Λh)1/2

δ det(λI)1/2


.

By the proof of Lemma B.2 in [15], we know det(Λh) ≤(λ + K)d. Thus, we have



K
X

τ=1
ϕτ
Hητ
H([V ⋆,ρ
H+1]αi)

2

Λ−1
h
≤2H2d

2 log λ + K

λ
+ log 2dH2

δ


≤dH2 log 2dH2K

δ
.

Then by a union bound over i ∈[d], with probability at least 1 −δ/2H2, we have

ii ≤
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ,
(G.9)

where ι = log(2dH2K/δ) ≥1. As for the term iv, by construction we have V ⋆,ρ
H+1 = bV ρ
H+1 = 0
with probability 1. Thus, we trivially have

iv ≤
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h .
(G.10)

Combining (G.8), (G.9) and (G.10), for all (s, a) ∈S × A, with probability at least 1 −δ/2H2, we
have

inf
PH(·|s,a)∈Uρ
H(s,a;µ0
H,i)[PH bV ρ
H+1](s, a) −
\
inf
PH(·|s,a)∈Uρ
H(s,a;µ0
H,i)[PH bV ρ
H+1](s, a)


≤4
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h .
(G.11)

Thus, we define ΓH(s, a) := 4
√

dH√ι Pd
i=1 ∥ϕi(s, a)1i∥Λ−1
h . By the definition of bQρ
H(s, a) in
Algorithm 1, we have

bQρ
H(s, a) =

rH(s, a) −ΓH(s, a)
	

[0,1] ≤rH(s, a) = Q⋆,ρ
H (s, a),

28


---Page Break---
which implies that a pessimistic estimation is achieved at step H, i.e., V ⋆,ρ
H (s) ≥bV ρ
H(s), ∀s ∈S.
Next, we study V ⋆,ρ
H (s) −bV ρ
H(s). The intuition is that given the estimation error bound in (G.11),
with sufficient data, the difference between V ⋆,ρ
H (s) and bV ρ
H(s) should be small. Specifically, we have

V ⋆,ρ
H (s) −bV ρ
H(s) = Q⋆,ρ
H (s, π⋆
H(s)) −bQρ
H(s, π⋆
H(s)) + bQρ
H(s, π⋆
H(s)) −bQρ
H(s, ˆπ(s))

≤rH(s, π⋆
H(s)) +
inf
PH(·|s,a)∈Uρ
H(s,a;µ0
H,i)[PH bV ρ
H+1](s, a)−

rH(s, π⋆
H(s)) −
\
inf
PH(·|s,a)∈Uρ
H(s,a;µ0
H,i)[PH bV ρ
H+1](s, a) + ΓH(s, π⋆
H(s))

(G.12)
≤2ΓH(s, π⋆
H(s)),

where (G.12) holds by the robust Bellman equation (3.1) and the fact that bQρ
H(s, π⋆
H(s)) −
bQρ
H(s, ˆπ(s)) ≤0. Then we bound the pessimism term ΓH(s, a) in terms of the sample size K.
By Lemma I.3, when K ≥max{512 log(2dH2/δ)/κ2, 4/κ}, with probability at least 1 −δ/2H2,
we have

2ΓH(s, a) = 8
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h
≤16
√

dH√ι
√

K

d
X

i=1
ϕi(s, a)
  ˜Λ−1
H
1/2
ii ,

where ˜ΛH = Eπb,P 0[ϕ(sH, aH)ϕ(sH, aH)⊤]. Note that for any positive definite matrix A, we know
λmin(A) ≤Aii ≤λmax(A). Thus, by Assumption 4.3, we have

2ΓH(s, a) ≤16
√

dH · 1√ι
√

Kκ
:= RH.
(G.13)

To summarize, we define the event

EH =

0 ≤V ⋆,ρ
H (s) −bV ρ
H(s) ≤RH, ∀s ∈S
	
.

Then by a union bound over (G.11) and (G.13), we know EH holds with probability at least 1 −δH =
1 −δ/H2. This concludes the proof of the base case.

Inductive Hypothesis.
Suppose with probability at least 1 −δh+1, we have

inf
Ph+1(·|s,a)∈Uρ
h+1(s,a;µ0
h+1,i)[Ph+1 bV ρ
h+2](s, a) −
\
inf
Ph+1(·|s,a)∈Uρ
h+1(s,a;µ0
h+1,i)[Ph+1 bV ρ
h+2](s, a)


≤4
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h+1 := Γh+1(s, a),
(G.14)

and

Eh+1 =

0 ≤V ⋆
h+1(s) −bVh+1(s) ≤Rh+1 := 16
√

dH(H −h)√ι
√

Kκ
, ∀s ∈S
	
.
(G.15)

Inductive Step.
Next, we establish the result for step h. First, terms i, ii and iii at step h can be
similarly bounded as in the base case, i.e., we have

i + ii + iii ≤3
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ,
(G.16)

with probability at least 1 −δ/3H2. It remains to bound the term iv and ensure it is non-dominating.
Here, we need to deal with the temporal dependency, as [bV ρ
h+1(s)]α −[V ⋆,ρ
h+1(s)]α is correlated to
{(sτ
h, aτ
h, sτ
h+1)}K
τ=1, thus we need a uniform concentration argument. Consider the function class

Vh(D, B, λ) = {Vh(s; θ, β, Σ) : S →[0, H] with ∥θ∥≤D, β ∈[0, B], Σ ⪰λI},

29


---Page Break---
where Vh(s; θ, β, Σ) = maxa∈A{ϕ(s, a)⊤θ −β Pd
i=1
p

ϕi(s, a)1⊤
i Σ−1ϕi(s, a)1i}[0,H−h+1]. For
simplicity, we denote fαi(s) := [bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi, then fαi ∈Fh+1(αi), where

Fh+1(α) :=

[bV ρ
h+1(s)]α −[V ⋆,ρ
h+1(s)]α : bV ρ
h+1(s) ∈Vh+1(D0, B0, λ)
	
.

Note that for any fixed α, the covering number of Fh+1(α) is the same as that of Vh(D0, B0, λ).
By Lemma G.2, we have D0 = H
p

Kd/λ. By the induction assumption (G.14), we have B0 =
4
√

dH√ι. Denote the ϵ-covering of the interval [0, H] with respect to the distance dist(α1, α2) =
|α1 −α2| as N[0,H](ϵ), and its ϵ-covering number as |N[0,H](ϵ)|. For each α ∈[0, H], we can find
αϵ ∈N[0,H](ϵ) such that |α −αϵ| ≤ϵ. For any fixed α ∈[0, H], we denote the ϵ-covering of
Fh+1(α) with respect to the distance dist(f1, f2) = supx |f1(x) −f2(x)| as Nh+1(ϵ) (short for
Nh+1(ϵ; D, B, λ)) and its ϵ-covering number as |Nh+1(ϵ)|. For each fα ∈Fh+1(α), we can find
f ϵ
α ∈Nh+1(ϵ) such that sups |fα(s) −f ϵ
α(s)| ≤ϵ. It follows that



K
X

k=1
ϕτ
hητ
h(fαi)

2

Λ−1
h
· 1

∥fαi∥∞≤Rh+1
	

≤2


K
X

τ=1
ϕτ
hητ
h(fαiϵ)

2

Λ−1
h
· 1

∥fαiϵ∥∞≤Rh+1 + ϵ
	
+ 2


K
X

k=1
ϕτ
h
 
ητ
h(fαi) −ητ
h(fαiϵ)

2

Λ−1
h
.

Note that

2


K
X

k=1
ϕτ
h
 
ητ
h(fαi) −ητ
h(fαiϵ)

2

Λ−1
h
≤2ϵ2
K
X

τ,τ ′=1

ϕτ
hΛ−1
h ϕτ ′
h
 ≤2ϵ2K2/λ.

Then we have


K
X

k=1
ϕτ
hητ
h(fαi)

2

Λ−1
h
· 1

∥fαi∥∞≤Rh+1
	

≤4


K
X

τ=1
ϕτ
hητ
h(f ϵ
αiϵ)

2

Λ−1
h
· 1

∥f ϵ
αiϵ∥∞≤Rh+1 + 2ϵ
	

+ 4


K
X

k=1
ϕτ
h
 
ητ
h(fαiϵ) −ητ
h(f ϵ
αiϵ)

2

Λ−1
h
+ 2ϵ2K2

λ

≤4


K
X

τ=1
ϕτ
hητ
h(f ϵ
αiϵ)

2

Λ−1
h
· 1

∥f ϵ
αiϵ∥∞≤Rh+1 + 2ϵ
	
+ 6ϵ2K2

λ
,

where the last inequality holds by the fact that

4


K
X

k=1
ϕτ
h
 
ητ
h(fαiϵ) −ητ
h(f ϵ
αiϵ)

2

Λ−1
h
≤4ϵ2
K
X

τ,τ ′=1

ϕτ
hΛ−1
h ϕτ ′
h
 ≤4ϵ2K2/λ.

With a union bound over Nh+1(ϵ) and N[0,H](ϵ) and by Lemma G.3, we have

P

(

sup
αiϵ∈N[0,H](ϵ)
f ϵ
αiϵ∈Nh+1(ϵ)



K
X

τ=1
ϕτ
hητ
h(f ϵ
αiϵ)

2

Λ−1
h
· 1

∥f ϵ
αiϵ∥∞≤Rh+1 + 2ϵ
	

> (Rh+1 + 2ϵ)2
2 log 3dH2|Nh+1(ϵ)||N[0,H](ϵ)|

δ
+ d log

1 + K

λ

)

≤
δ
3dH2 .

Then with probability at least 1 −δ/3dH2, for all fαi ∈Fh+1(αi), we have



K
X

k=1
ϕτ
hητ
h(fαi)

2

Λ−1
h
· 1

∥fαi∥∞≤Rh+1
	

30


---Page Break---
≤4 inf
ϵ>0

n
(Rh+1 + 2ϵ)2
2 log
3dH2|Nh+1(ϵ)||N[0,H](ϵ)|

δ


+ d log

1 + K

λ


+ 6ϵ2K2

λ

o
.

By Lemma G.4 and Lemma G.5 together with D0 = H
p

Kd/λ and B0 = 4
√

dH√ι, setting ϵ =
d3/2H2/(K3/2√κ) and K ≥
√

dH/(32√κι), we have log |Nh+1(ϵ)| ≤2d2 log(512K3ι/d3/2H2).
Thus, we have



K
X

k=1
ϕτ
hητ
h(fαi)

2

Λ−1
h
· 1

∥fαi∥∞≤Rh+1
	
≤512dH4ι

Kκ


2 log 2dH2

δ
+ 4d2 log 512K3ι

d3/2H2



≤20480d3H4ι2

Kκ
.

Then, with a union bound over i ∈[d], we have

P

sup
i∈[d]



K
X

τ=1
ϕτ
hητ
h
 
[bV ρ
h+1]αi −[V ⋆,ρ
h+1]αi

Λ−1
h
> 143d3/2H2ι
√

Kκ



≤P

sup
i∈[d]



K
X

τ=1
ϕτ
hητ
h
 
[bV ρ
h+1]αi −[V ⋆,ρ
h+1]αi

Λ−1
h
1
[bV ρ
h+1]αi −[V ⋆,ρ
h+1]αi

∞≤Rh+1
	

> 143d3/2H2ι
√

Kκ


+ P
 
1
[bV ρ
h+1]αi −[V ⋆,ρ
h+1]αi

∞> Rh+1
	

≤
δ
3H2 + δh+1,

which implies with probability at least 1 −δ/3H2 −δh+1, the term iv at step h can be bounded as

iv ≤143d3/2H2ι
√

Kκ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h .
(G.17)

Then by a union bound over (G.16) and (G.17), if K > 20449d2H2/κ, then with probability at least
1 −2δ/3H2 −δh+1 we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤4
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h
:= Γh(s, a).
(G.18)

Further, when K > max{512 log(3H2/δ)/κ2, 4/κ}, by Lemma I.3, with probability at least 1 −
δ/3H2, we have

Γh(s, a) ≤4
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h
≤8
√

dH√ι
√

Kκ
.
(G.19)

Then by a union bound over (G.18) and (G.19), under the event Eh+1, with probability at least
1 −δ/H2 −δh+1, we have

V ⋆,ρ
h
(s) −bV ρ
h (s)

= Q⋆,ρ
h (s, π⋆(s)) −bQρ
h(s, π⋆(s)) + bQρ
h(s, π⋆(s)) −bQρ
h(s, ˆπ(s))

≤
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ⋆,ρ
h+1](s, π⋆(s)) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) + Γh(s, π⋆(s))

=
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ⋆,ρ
h+1](s, π⋆(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆(s))

+
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆(s)) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) + Γh(s, π⋆(s))

≤Rh+1 + 2Γh(s, π⋆(s))
(G.20)

31


---Page Break---
≤16
√

dH(H −h)√ι
√

Kκ
+ 16
√

dH√ι
√

Kκ

= 16
√

dH(H −h + 1)√ι
√

Kκ
:= Rh,

where (G.20) holds by the following argument

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[PhV ⋆,ρ
h+1](s, π⋆(s)) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, π⋆(s))

≤[ˆPhV ⋆,ρ
h+1](s, π⋆(s)) −[ˆPh bV ρ
h+1](s, π⋆(s))

≤sup
s |V ⋆,ρ
h+1(s) −bV ρ
h+1(s)|

≤Rh+1,
(G.21)

where ˆPh(·|s, a) = arg infPh(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a), ∀(s, a) ∈S × A, and (G.21) is due
to the induction assumption (G.15). Finally, denote

Eh = {0 ≤V ⋆,ρ
h+1(s) −bV ρ
h+1(s) ≤Rh, ∀s ∈S},

then we have P(Eh) ≤δh+1 + δ/H2 := δh.

Generalization.
By induction and a union bound over h ∈[H], setting

Γh(s, a) = 4
√

dH√ι

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ,

then with probability at least 1−(δ/H2+2δ/H2+· · ·+Hδ/H2) = 1−dH(H +1)δ/2H2 > 1−δ,
for all (s, a, h) ∈S × A × [H], we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)
 ≤Γh(s, a).

This concludes the proof.

G.3
Proof of Lemma E.1

Proof. Note that the conditional variance estimation does not involve any element of model uncer-
tainty, and thus the proof follows from Lemma 5 of [54]. Recall that we estimate [Vh[V ρ
h+1]α](s, a)
based on D′ as

bσ2
h(s, a; α) = max
n
1,

ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −˜O
√

dH3
√

Kκ

o
.

Note that


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −[Ph[bV

′ρ
h+1]2
α](s, a) −([Ph[bV

′ρ
h+1]α](s, a))2

≤


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −[Ph[bV

′ρ
h+1]2
α](s, a)
 +


ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −([Ph[bV

′ρ
h+1]α](s, a))2

≤
ϕ(s, a)⊤˜βh,2(α) −[Ph[bV

′ρ
h+1]2
α](s, a)

|
{z
}
i

+2H
ϕ(s, a)⊤˜βh,1(α) −[Ph[bV

′ρ
h+1]α](s, a)

|
{z
}
ii

.

Note that the estimation error i and ii both come from regular ridge regressions with targets [bV

′ρ
h+1(s)]2
α
and [bV

′ρ
h+1(s)]α, respectively. Thus, the analysis is standard and for simplicity we omit the details
here and focus on the results: with probability at least 1 −δ/2, we have


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −[Ph[bV

′ρ
h+1]2
α](s, a) −([Ph[bV

′ρ
h+1]α](s, a))2

≤˜O
 dH2

√

Kκ


.
(G.22)

32


---Page Break---
Then by Theorem 4.4 and Lemma I.3, for all (s, a, h) ∈S × A × [H], with probability at least
1 −δ/2, we have
[Varh[bV

′ρ
h+1]α](s, a) −[Varh[V ⋆,ρ
h+1]α](s, a)


≤
[Ph[bV

′ρ
h+1]2
α](s, a) −[Ph[V ⋆,ρ
h+1]2
α](s, a)
 +
 
[Ph[bV

′ρ
h+1]α](s, a)
2 −
 
[Ph[V ⋆,ρ
h+1]α](s, a)
2

≤2H

Ph([bV ρ
h+1]α −[V ⋆,ρ
h+1]α)

(s, a)
 + 2H

Ph
 
[V ⋆,ρ
h+1]α −[V ⋆,ρ
h+1]α

(s, a)


≤˜O
√

dH3
√

Kκ


.
(G.23)

By (G.22) and (G.23) and a union bound, we know that with probability at least 1 −δ, we have


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −[Varh[V ⋆,ρ
h+1]α](s, a)


≤


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −[Varh[bV

′ρ
h+1]α](s, a)


+
[Varh[bV

′ρ
h+1]α](s, a) −[Varh[V ⋆,ρ
h+1]α](s, a)


≤˜O
 dH3

√

Kκ


,

which implies that

ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −˜O
 dH3

√

Kκ


≤[Varh[V ⋆,ρ
h+1]α](s, a).

By the fact that the operator min{1, ·} is order preserving, thus we have

bσ2
h(s, a; α) ≤[Vh[V ⋆,ρ
h+1]α](s, a).

Further, by the fact that the operator min{1, ·} is a contraction map, (G.22) and (G.23), we have
bσ2
h(s, a; α) −

Vh[V ⋆,ρ
h+1]α

(s, a)


≤
bσ2
h(s, a; α) −

Vh[bV

′ρ
h+1]α

(s, a)
 +

Vh[bV

′ρ
h+1]α

(s, a) −

Vh[V ⋆,ρ
h+1]α

(s, a)


≤


ϕ(s, a)⊤˜βh,2(α)


[0,H2] −

ϕ(s, a)⊤˜βh,1(α)
2
[0,H] −˜O
 dH3

√

Kκ


−[Varh[bV

′ρ
h+1]α](s, a)


+
[Varh[bV

′ρ
h+1]α](s, a) −[Varh[V ⋆,ρ
h+1]α](s, a)


≤˜O
 dH2

√

Kκ


+ ˜O
 dH3

√

Kκ


+ ˜O
√

dH3
√

Kκ



= ˜O
 dH3

√

Kκ


.

This concludes the proof.

G.4
Proof of Lemma E.2

Proof. Note that the reference-advantage decomposition is exactly the same as that in the proof of
Lemma G.1, thus we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h[V ⋆,ρ
h+1(s)]αi −bEµ0
h[V ⋆,ρ
h+1(s)]αi


|
{z
}
reference uncertainty

+

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

−bEµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi


|
{z
}
advantage uncertainty

.

Next, we further decompose the reference uncertainty and the advantage uncertainty, respectively.

33


---Page Break---
The Reference Uncertainty.
Specifically, we have

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h[V ⋆,ρ
h+1(s)]αi −bEµ0
h[V ⋆,ρ
h+1(s)]αi


=

d
X

i=1
ϕi(s, a)1⊤
i

Eµ0
h[V ⋆,ρ
h+1(s)]αi −Σ−1
h (αi)

K
X

τ=1

ϕτ
h

P0
h[V ⋆,ρ
h+1]αi

(sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h; αi)

+ Σ−1
h (αi)

K
X

τ=1

ϕτ
h

P0
h[V ⋆,ρ
h+1]αi

(sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h; αi)
−Σ−1
h (αi)

K
X

τ=1

ϕτ
h[V ⋆,ρ
h+1(sτ
h+1)]αi
bσ2
h(sτ
h, aτ
h; αi)



= λ

d
X

i=1
ϕi(s, a)1⊤
i Σ−1
h (αi)Eµ0
h[V ⋆,ρ
h+1(s)]αi +

d
X

i=1
ϕi(s, a)1⊤
i Σ−1
h (αi)

K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)

≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Σ−1
h (αi)
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
ii

.

The Advantage Uncertainty.
Similar to the argument in decomposing the reference uncertainty,
we have

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

−bEµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Σ−1
h (αi)
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
iv

.

Put terms i-iv together, we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Σ−1
h (αi)
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Σ−1
h (αi)
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)


K
X

τ=1

ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (αi)
|
{z
}
iv

.

34


---Page Break---
By similar argument as Lemma G.1, we know there exist {˜αi}i∈[d] such that

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (˜αi)∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Σ−1
h (˜αi)
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (˜αi)


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (˜αi)
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (αi)
Eµ0
h
[bV ρ
h+1(s)]˜αi −[V ⋆,ρ
h+1(s)]αi

Σ−1
h (˜αi)
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h (˜αi)


K
X

τ=1

ϕτ
hητ
h([bV ρ
h+1(s)]˜αi −[V ⋆,ρ
h+1(s)]αi)
bσ2
h(sτ
h, aτ
h; αi)


Σ−1
h (˜αi)
|
{z
}
iv

.

This concludes the proof.

G.5
Proof of Lemma E.3

Proof. By the robust bellman equation (3.1), we know

V π,ρ
h
(s) = Ea∼π(·|s)
h
r(s, a) +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV π,ρ
h+1](s, a)
i
.
(G.24)

Then, we can trivially bound maxs∈S V π,ρ
h
(s) as

max
s∈S V π,ρ
h
(s) ≤max
s,a


1 +
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV π,ρ
h+1](s, a)

.
(G.25)

Further, by the definition of the d-rectangular uncertainty set, we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV π,ρ
h+1](s, a) =

d
X

i=1
ϕi(s, a)
inf
µh,i∈Uρ
h,i(µ0
h,i) Es∼µh,i[V π,ρ
h+1(s)].
(G.26)

Denoting smax = argmaxs∈S V π,ρ
h+1(s) and smin = argmins∈S V π,ρ
h+1(s), and for all i ∈[d], we
construct a distribution ˇµh,i = (1 −ρ)µh,i + ρδsmin, where δx is the Dirac Delta distribution with
mass on x. Note that ˇµh,i ∈Uρ
h,i(µ0
h,i), thus we have

inf
µh,i∈Uρ
h,i(µ0
h,i) Es∼µh,i[V π,ρ
h+1(s)] ≤Es∼ˇµh,i[V π,ρ
h+1(s)] ≤(1 −ρ) max
s∈S V π,ρ
h+1(s) + ρ min
s
V π,ρ
h+1(s).

(G.27)

Combining (G.25), (G.26) and (G.27), we have

max
s∈S V π,ρ
h
(s) ≤(1 −ρ) max
s∈S V π,ρ
h+1(s) + ρ min
s∈S V π,ρ
h+1(s) + 1.
(G.28)

On the other hand, by (G.24), we can trivially bound mins V π,ρ
h
(s) as

min
s
V π,ρ
h
(s) ≥min
s,a
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h)[PhV π,ρ
h+1](s, a).
(G.29)

By the fact that

inf
µh,i∈Uρ
h,i(µ0
h,i) Es∼µh,i[V π,ρ
h+1(s)] ≥min
s∈S V π,ρ
h+1(s),
(G.30)

35


---Page Break---
combining (G.26), (G.29) and (G.30), we have

min
s
V π,ρ
h
(s) ≥min
s∈S V π,ρ
h+1(s).
(G.31)

For any h ∈[H], by (G.28) and (G.31), we have

max
s∈S V π,ρ
h
(s) −min
s∈S V π,ρ
h
(s)

≤1 + (1 −ρ) max
s∈S V π,ρ
h+1(s) −min
s∈S V π,ρ
h+1(s) + ρ min
s∈S V π,ρ
h+1(s)

= 1 + (1 −ρ)

max
s∈S V π,ρ
h+1(s) −min
s∈S V π,ρ
h+1(s)

.
(G.32)

For step H, by the definition of the value function, we have 0 ≤V π,ρ
H (s) ≤1, ∀s ∈S. Applying
(G.32) with h = H −1 leads to maxs∈S V π,ρ
H−1(s) −mins∈S V π,ρ
H−1(s) ≤1 + (1 −ρ) · 1. We finish
the proof by recursively applying (G.32).

G.6
Proof of Lemma F.1

Proof. The proof of Lemma F.1 consists of the following two steps:

Step 1: lower bound the suboptimality by Hamming distance.
For any ξ ∈{−1, 1}dH, denote
V ⋆,ρ
ξ
(s) as the optimal robust value function for the MDP instance Mξ. For any function π, denote
V π,ρ
ξ
as the robust value function corresponding to a policy π. Then by definition, we have

V ⋆,ρ
ξ
(x1) = max
π
inf
P ∈Uρ(P 0) Eπ,P 
r1(s1, a1) + · · · + rH(sH, aH)|s1 = x1

,

V π,ρ
ξ
(x1) =
inf
P ∈Uρ(P 0) Eπ,P 
r1(s1, a1) + · · · + rH(sH, aH)|s1 = x1

.

For any given ξ, the optimal action at step h is

a⋆
h = ((1 + ξh1)/2, · · · , (1 + ξhd)/2).

The worst case transition at the first step is known as

P1(x1|x1, a) = (1 −ρ), P1(x2|x1, a) = ρ, P1(x2|x2, a) = 1, ∀a ∈A,

and from the second step on, the state always stays at s2. With these facts in mind, we have

V ⋆,ρ
ξ
(x1)

= δ
nh1

2 +

d
X

i=1

1 + ξ1i

4d

i
+ (1 −ρ)
h1

2 +

d
X

i=1

1 + ξ2i

4d

i
+ · · · + (1 −ρ)
h1

2 +

d
X

i=1

1 + ξHi

4d

io

= δ

2d

nh
d +

d
X

i=1

1 + ξ1i

2

i
+ (1 −ρ)
h
d +

d
X

i=1

1 + ξ2i

2

i
+ · · · + (1 −ρ)
h
d +

d
X

i=1

1 + ξHi

2

io
,

and

V π,ρ
ξ
(x1)

= δ

2dEπnh
d +

d
X

i=1
ξ1ia1i
i
+ (1 −ρ)
h
d +

d
X

i=1
ξ2ia2i
i
· · · + (1 −ρ)
h
d +

d
X

i=1
ξHiaHi
io
.

Then we have

V ⋆,ρ
ξ
(x1) −V π,ρ
ξ
(x1)

= δ

2d

nh
d
X

i=1

1 + ξ1i

2
−ξ1iEπa1i
i
+ (1 −ρ)

H
X

h=2

d
X

i=1

1 + ξhi

2
−ξhiEπahi
o

≥δ

2d(1 −ρ)

H
X

h=1

d
X

i=1

1 + ξhi

2
−ξhiEπahi


36


---Page Break---
= δ

2d(1 −ρ)

H
X

h=1

d
X

i=1

1

2 + ξhiEπ1

2 −ahi


= δ

4d(1 −ρ)

H
X

h=1

d
X

i=1
(1 −ξhiEπ(2ahi −1)).
(G.33)

Note that for any (h, i) ∈[H] × [d], by design we have 1 = ξ2
hi, thus

δ
4d(1 −ρ)

H
X

h=1

d
X

i=1
(1 −ξhiEπ(2ahi −1)) = δ

4d(1 −ρ)

H
X

h=1

d
X

i=1
(ξhi −Eπ(2ahi −1))ξhi

= δ

4d(1 −ρ)

H
X

h=1

d
X

i=1
|ξhi −Eπ(2ahi −1)|,
(G.34)

where (G.34) holds due to the fact that Eπ(2ahi −1) ∈[−1, 1]. To continue, we have

δ
4d(1 −ρ)

H
X

h=1

d
X

i=1
|ξhi −Eπ(2ahi −1)|

≥δ

4d(1 −ρ)

H
X

h=1

d
X

i=1
|ξhi −Eπ(2ahi −1)| 1{ξhi ̸= sign(Eπ(2ah,i −1))}

≥δ

4d(1 −ρ)

H
X

h=1

d
X

i=1
1{ξhi ̸= sign(Eπ(2ah,i −1))}

≥δ

4d(1 −ρ)DH(ξ, ξπ),
(G.35)

where DH(·, ·) is the Hamming distance, ξπ = {ξπ
h}h∈[H], and ξπ
hi := sign(Eπ(2ahi −1)), ∀i ∈[d].
Combining (G.33), (G.34), (G.35) and the definition of the suboptimality gap, we have

SupOpt(Mξ, x1, π, ρ) ≥δ

4d(1 −ρ)DH(ξ, ξπ).
(G.36)

Step 2: lower bound the hamming distance by testing error.
Applying Assouad’s method [46,
Lemma 2.12], we have

inf
π sup
ξ∈Ω
Eξ

DH(ξ, ξ′)

≥dH

2
min
ξ,ξ′∈Ω
DH(ξ,ξ′)=1

inf
ψ

h
Qξ(ψ(D) ̸= ξ) + Qξ′(ψ(D) ̸= ξ′)
i
,
(G.37)

where infψ denotes the infimum over all test functions taking values in {ξ, ξ′}. We conclude the
proof by combining (G.36) and (G.37).

G.7
Proof of Lemma F.2

Proof. By the Theorem 2.12 in [46], we lower bound the testing error as follows

min
ξ,ξ′:DH(ξ,ξ′)=1 inf
ψ

h
Qξ(ψ(D) ̸= ξ) + Qξ′(ψ(D) ̸= ξ′)
i

≥1 −
1

2
max
ξ,ξ′:DH(ξ,ξ′)=1 DKL
 
Qξ||Qξ′1/2
,

where DKL(·||·) is the Kullback-Leibler divergence. Then it remains to bound DKL
 
Qξ||Qξ′
.
According to the definition of Qξ(D), we have

Qξ(D) =

K
Y

k=1

H
Y

h=1
πb
h(ak
h|sk
h)Ph(sk
h+1|sk
h, ak
h)R(sk
h, ak
h; rk
h),

37


---Page Break---
where R(sk
h, ak
h; rk
h) is the density function of N(rh(sk
h, ak
h), 1) at rk
h. Note that the difference
between the two distribution Qξ(D) and Qξ′(D) lies only in the reward distribution corresponding to
the index where ξ and ξ′ differ. Then, by the chain rule of Kullback-Leibler divergence, we have

DKL
 
Qξ(D)||Qξ′(D)

=

K
d+2
X

k=1
DKL

N
d + 1

2d δ, 1

N
d −1

2d δ, 1

=
K
d + 2
δ2

d2 .

Then by our choice of δ, we have

min
ξ,ξ′:DH(ξ,ξ′)=1 inf
ψ

h
Qξ(ψ(D) ̸= ξ) + Qξ′(ψ(D) ̸= ξ′)
i
≥1 −

Kδ2

2(d + 2)d2

1/2

≥1 −
Kδ2

2d3

1/2

= 1

2.

This completes the proof.

G.8
Proof of Lemma F.3

Proof. Recall that

Σ⋆−1
h
=

K
X

k=1

ϕτ
hϕτ⊤
h
[VhV ⋆,ρ
h
](sτ
h, aτ
h) + λI.

We first show that with sufficiently large K, the clipped conditional variances of the optimal robust
value functions are always 1. Note that V ⋆,ρ
h
(x2) = 0, ∀h ∈[H], and

V ⋆,ρ
H (x1) = δ

2d


d
X

i=1

1 + ξHi

2
+ d

≤δ,

V ⋆,ρ
H−1(x1) = δ

2d


d
X

i=1

1 + ξH−1i

2
+ d

+ V ⋆,ρ
H (x1) ≤2δ,

· · ·

V ⋆,ρ
2
(x1) = δ

2d


d
X

i=1

1 + ξ2i

2
+ d

+ V ⋆,ρ
3
(x1) ≤(H −1) · δ.

Then, when K ≥Ω(H2d3), we have

Var1 V ⋆,ρ
2

(x1, a) =

P0
1(V ⋆,ρ
2
)2
(x1, a) −
 
P0
1(V ⋆,ρ
2
)2
(x1, a)
2 ≤(1 −ρ)ρH2δ2 ≤1,

and by design we have,

[Var1 V ⋆,ρ
2
](x2, a) = 0 and [Varh V ⋆,ρ
h+1](s, a) = 0, ∀(s, a, h) ∈S × A × [H]/{1}.

Thus, we have [VhV ⋆,ρ
h
](sτ
h, aτ
h) = 1, which implies

Σ⋆
h = Λh.
(G.38)

Define
˜Λh = Eπb,P 0[ϕ(sh, ah)ϕ(sh, ah)⊤],

then by definition we have

˜Λh =
1
d + 2





1
d2
0
· · ·
0
1
d(1 −1

d)
0
0
0
· · ·
0
0
0
...
...
...
...
...
0
0
· · ·
0
0
0
1
d(1 −1

d)
0
· · ·
0
(1 −1

d)2
0
0
0
· · ·
0
0
0





+
1
d + 2





0
0
· · ·
0
0
0
0
1
d2
· · ·
0
1
d(1 −1

d)
0
...
...
...
...
...
0
0
· · ·
0
0
0
0
1
d(1 −1

d)
· · ·
0
(1 −1

d)2
0
0
0
· · ·
0
0
0





38


---Page Break---
+ · · · +
1
d + 2





0
0
· · ·
0
0
0
0
0
· · ·
0
0
0
...
...
...
...
...
0
0
· · ·
1
d2
1
d(1 −1

d)
0
0
0
· · ·
1
d(1 −1

d)
(1 −1

d)2
0
0
0
· · ·
0
0
0





+
1
d + 2





0
0
· · ·
0
0
0
0
0
· · ·
0
0
0
...
...
...
...
...
0
0
· · ·
0
0
0
0
0
· · ·
0
1
0
0
0
· · ·
0
0
0





+
1
d + 2





0
0
· · ·
0
0
0
0
0
· · ·
0
0
0
...
...
...
...
...
0
0
· · ·
0
0
0
0
0
· · ·
0
0
0
0
0
· · ·
0
0
1





=
d
d + 2





1
d3
0
· · ·
0
1
d2 (1 −1

d)
0
0
1
d3
· · ·
0
1
d2 (1 −1

d)
0
...
...
...
...
...
0
0
· · ·
1
d3
1
d2 (1 −1

d)
0
1
d2 (1 −1

d)
1
d2 (1 −1

d)
· · ·
1
d2 (1 −1

d)
(1 −1

d)2 + 1

d
0
0
0
· · ·
0
0
1
d





.

Denote

D =





1
d3
0
· · ·
0
1
d2 (1 −1

d)
0
1
d3
· · ·
0
1
d2 (1 −1

d)
...
...
...
...
0
0
· · ·
1
d3
1
d2 (1 −1

d)
1
d2 (1 −1

d)
1
d2 (1 −1

d)
· · ·
1
d2 (1 −1

d)
(1 −1

d)2 + 1

d




,

then by Gaussian elimination, we have

D−1 =





2d3 −2d2 + d
d3 −2d2 + d
· · ·
d3 −2d2 + d
d −d2

d3 −2d2 + d
2d3 −2d2 + d
· · ·
d3 −2d2 + d
d −d2
...
...
...
...
d3 −2d2 + d
d3 −2d2 + d
· · ·
2d3 −2d2 + d
d −d2

d −d2
d −d2
· · ·
d −d2
d




.

Note that

˜Λh =
d
d + 2

D
0
0
1
d


,

then we have

˜Λ−1
h
= d + 2

d


D−1
0
0
d


.

Note that λmin(D) = O(1/d3), thus ∥˜Λ−1
h ∥= O(d3). Then when K > ˜O(d6), for any (s, a, i, h) ∈
S × A × [d] × [H], with probability at least 1 −δ, we have

∥ϕi(s, a)1i∥Λ−1
h
≤
2
√

K
∥ϕi(s, a)1i∥˜Λ−1
h .
(G.39)

With this in mind, we have

sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(s, a)1i∥Σ⋆−1
h
|s1 = x1
i

=
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h
d
X

i=1
∥ϕi(sh, ah)1i∥Λ−1
h |s1 = x1
i

39


---Page Break---
≤
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h 2
√

K

d
X

i=1
∥ϕi(sh, ah)1i∥˜Λ−1
h |s1 = x1
i
(G.40)

=
sup
P ∈Uρ(P 0)

H
X

h=1
Eπ⋆,P h 2
√

K

d
X

i=1
ϕi(sh, ah)
  ˜Λ−1
h
1/2
ii |s1 = x1
i

≤4Hd3/2

√

K
,

where (G.40) is due to (G.39). This concludes the proof.

H
Proof of Supporting Lemmas

H.1
Proof of Lemma G.1

To prove Lemma G.1, we need the following proposition on the dual formulation under the TV
uncertainty set.

Proposition H.1. (Strong duality for TV [42, Lemma 4]). Given any probability measure µ0 over
S, a fixed uncertainty level ρ, the uncertainty set Uρ(µ0) = {µ : µ ∈∆(S), DT V (µ||µ0) ≤ρ}, and
any function V : S →[0, H], we obtain

inf
µ∈Uρ(µ0) Es∼µV (s) =
max
α∈[Vmin,Vmax]


Es∼µ0[V (s)]α −ρ
 
α −min
s′ [V (s′)]α
	
,
(H.1)

where [V (s)]α = min{V (s), α}, Vmin = mins V (s) and Vmax = maxs V (s). Notably, the range of
α can be relaxed to [0, H] without impacting the optimization.

Proof of Lemma G.1. By Assumption 3.1 and Proposition H.1, we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

=

d
X

i=1
ϕi(s, a)
h
max
α∈[0,H]{Eµ0
h,i[bV ρ
h+1(s)]α −ρ(α −min
s′ [bV ρ
h+1(s′)]α)}

−max
α∈[0,H]{bEµ0
h,i[bV ρ
h+1(s)]α −ρ(α −min
s′ [bV ρ
h+1(s′)]α)}
i
.

Denote αi = argmaxα∈[0,H]{Eµ0
h,i[bV ρ
h+1(s)]α −ρ(α −mins′[bV ρ
h+1(s′)]α)}, then we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤

d
X

i=1
ϕi(s, a)
 
Eµ0
h,i[bV ρ
h+1(s)]αi −bEµ0
h,i[bV ρ
h+1(s)]αi


=

d
X

i=1
ϕi(s, a)

1⊤
i Eµ0
h[bV ρ
h+1(s)]αi −1⊤
i bEµ0
h[bV ρ
h+1(s)]αi

.

Here we do reference-advantage decomposition by using the optimal robust value function as the
reference function. Specifically, we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤

d
X

i=1
ϕi(s, a)

1⊤
i
 
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi + [V ⋆,ρ
h+1(s)]αi


−1⊤
i
 bEµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi + [V ⋆,ρ
h+1(s)]αi


40


---Page Break---
=

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h[V ⋆,ρ
h+1(s)]αi −bEµ0
h[V ⋆,ρ
h+1(s)]αi


|
{z
}
reference uncertainty

+

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

−bEµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi


|
{z
}
advantage uncertainty

.

(H.2)

The Reference Uncertainty.
First, we bound the reference uncertainty. Specifically, we have

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h[V ⋆,ρ
h+1(s)]αi −bEµ0
h[V ⋆,ρ
h+1(s)]αi


=

d
X

i=1
ϕi(s, a)1⊤
i

Eµ0
h[V ⋆,ρ
h+1(s)]αi −Λ−1
h

K
X

τ=1
ϕτ
h

P0
h[V ⋆,ρ
h+1]αi

(sτ
h, aτ
h)

+ Λ−1
h

K
X

τ=1
ϕτ
h

P0
h[V ⋆,ρ
h+1]αi

(sτ
h, aτ
h) −Λ−1
h

K
X

τ=1
ϕτ
h[V ⋆,ρ
h+1(sτ
h+1)]αi


=

d
X

i=1
ϕi(s, a)1⊤
i

Eµ0
h[V ⋆,ρ
h+1(s)]αi −Λ−1
h

K
X

τ=1
ϕτ
hϕτ⊤
h Eµ0
h[V ⋆,ρ
h+1(s)]αi

+ Λ−1
h

K
X

τ=1
ϕτ
h
 
P0
h[V ⋆,ρ
h+1]αi

(sτ
h, aτ
h) −[V ⋆,ρ
h+1(sτ
h+1)]αi

.

For any function f : S →[0, H −1], we define ητ
h([f]αi) =
 
P0
h[f]αi

(sτ
h, aτ
h) −[f(sτ
h+1)]αi

.
Then, we have

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h[V ⋆,ρ
h+1(s)]αi −bEµ0
h[V ⋆,ρ
h+1(s)]αi


= λ

d
X

i=1
ϕi(s, a)1⊤
i Λ−1
h Eµ0
h[V ⋆,ρ
h+1(s)]αi +

d
X

i=1
ϕi(s, a)1⊤
i Λ−1
h

K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]αi)

≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Λ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]αi)

Λ−1
h
|
{z
}
ii

.

(H.3)

The Advantage Uncertainty.
Next, we bound the advantage uncertainty. By similar argument in
bounding the reference uncertainty, we have

d
X

i=1
ϕi(s, a)1⊤
i
 
Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

−bEµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h

Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Λ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)

Λ−1
h
|
{z
}
iv

.
(H.4)

41


---Page Break---
Combining (H.2), (H.3) and (H.4), we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Λ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]αi)

Λ−1
h
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h

Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Λ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)

Λ−1
h
|
{z
}
iv

.

On the other hand, we can similarly deduce

\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)

≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]α′
i∥Λ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]α′
i)

Λ−1
h
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h

Eµ0
h
[bV ρ
h+1(s)]α′
i −[V ⋆,ρ
h+1(s)]α′
i

Λ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([bV ρ
h+1(s)]α′
i −[V ⋆,ρ
h+1(s)]α′
i)

Λ−1
h
|
{z
}
iv

,

where α′
i = argmaxα∈[0,H]{bEµ0
h,i[bV ρ
h+1(s)]α −ρ(α −mins′[bV ρ
h+1(s′)]α)}. Then for all i ∈[d],
there exist ˜αi ∈{αi, α′
i}, such that

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]˜αi∥Λ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([V ⋆,ρ
h+1]˜αi)

Λ−1
h
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h

Eµ0
h
[bV ρ
h+1(s)]˜αi −[V ⋆,ρ
h+1(s)]˜αi

Λ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Λ−1
h



K
X

τ=1
ϕτ
hητ
h([bV ρ
h+1(s)]˜αi −[V ⋆,ρ
h+1(s)]˜αi)

Λ−1
h
|
{z
}
iv

,

This concludes the proof.

H.2
Proof of Lemma G.2

The proof of Lemma G.2 will use the following fact.

42


---Page Break---
Lemma H.2. [14, Lemma D.1] Let Λt = λI + Pt
i=1 ϕiϕ⊤
i , where ϕi ∈Rd and λ > 0. Then:

t
X

i=1
ϕ⊤
i (Λt)−1ϕi ≤d.

Proof of Lemma G.2. The proof of Lemma G.2 is similar to that of Lemma E.1 in [20]. Denote
αi = argmaxα∈[0,H]{ˆzh,i(α)−ρ(α−mins′[bV ρ
h+1(s′)]α)}, i ∈[d]. For any vector v ∈Rd, we have
v⊤wρ
h
 =
v⊤θh + v⊤h
max
α∈[0,H]{ˆzh,i(α) −ρ(α −min
s′ [bV ρ
h+1(s′)]α)}
i

i∈[d]



≤
v⊤θh
 +
v⊤h
max
α∈[0,H]{ˆzh,i(α) −ρ(α −min
s′ [bV ρ
h+1(s′)]α)}
i

i∈[d]



≤
√

d∥v∥2 + H∥v∥1 +
v⊤

1⊤
i


Λ−1
h

K
X

τ=1
ϕτ
h[max
a
bQρ
h+1(sτ
h+1, a)]αi



i∈[d]


(H.5)

≤
√

d∥v∥2 + H
√

d∥v∥2 +

v
u
u
t

 K
X

τ=1
v⊤Λ−1
h v
 K
X

τ=1
(ϕτ
h)⊤Λ−1
h ϕτ
h


· H
(H.6)

≤2H∥v∥2
p

dK/λ.
(H.7)

We note that the term [(Λ−1
h
PK
τ=1 ϕτ
h[maxa bQρ
h+1(sτ
h+1, a)]αi)i]i∈[d] in (H.5) is constructed by first
taking out the i-th coordinate of the ridge solution vector, Λ−1
h
PK
τ=1 ϕτ
h[maxa bQρ
h+1(sτ
h+1, a)]αi ∈
Rd, ∀i ∈[d], and then concatenating all d values into a vector. Inequality (H.5) is due to the fact that
ρ ≤1, (H.6) is due to the fact that bQρ
h+1 ≤H, and (H.7) is due to Lemma H.2 with t = K and the
fact that the minimum eigenvalue of Λh is lower bounded by λ. The remainder of the proof follows
from the fact that ∥wρ
h∥2 = maxv:∥v∥2=1 |v⊤wρ
h|.

H.3
Proof of Lemma G.4

The proof of Lemma G.4 will use the following fact.
Lemma H.3. [14, Covering Number of Euclidean Ball] For any ϵ > 0, the ϵ-covering number of the
Euclidean ball in Rd with radius R > 0 is upper bounded by (1 + 2R/ϵ)d.

Proof of Lemma G.4. The proof is similar to the proof of Lemma E.3 in [20]. Denote A = β2Σ−1
h ,
so we have

Vh(·) = max
a∈A

n
ϕ(s, a)⊤θ −

d
X

i=1

q

ϕi(s, a)1⊤
i Aϕi(s, a)1i
o

[0,H−h+1],
(H.8)

for ∥θ∥≤L, ∥A∥≤B2λ−1. For any two functions V1, V2 ∈V, let them take the form in (H.8)
with parameters (θ1, A1) and (θ2, A2), respectively. Then since both {·}[0,H−h+1] and maxa are
contraction maps, we have

dist(V1, V2) ≤sup
x,a




θ⊤
1 ϕ(x, a) −

d
X

i=1

q

ϕi(x, a)1⊤
i A1ϕi(x, a)1i



−

θ⊤
2 ϕ(x, a) −

d
X

i=1

q

ϕi(x, a)1⊤
i A2ϕi(x, a)1i



≤
sup
ϕ:∥ϕ∥≤1




θ⊤
1 ϕ −

d
X

i=1

q

ϕi1⊤
i A1ϕi1i


−

θ⊤
2 ϕ −

d
X

i=1

q

ϕi1⊤
i A2ϕi1i



≤
sup
ϕ:∥ϕ∥≤1

(θ1 −θ2)⊤ϕ
 +
sup
ϕ:∥ϕ∥≤1

d
X

i=1

q

ϕi1⊤
i (A1 −A2)ϕi1i
(H.9)

43


---Page Break---
≤∥θ1 −θ2∥+
p

∥A1 −A2∥
sup
ϕ:∥ϕ∥≤1

d
X

i=1
∥ϕi1i∥

≤∥θ1 −θ2∥+
p

∥A1 −A2∥F ,
(H.10)

where (H.9) follows from triangular inequlaity and the fact that |√x −√y| ≤
p

|x −y|, ∀x, y ≥0.
For matrices, ∥· ∥and ∥· ∥F denote the matrix operator norm and Frobenius norm respectively.

Let Cθ be an ϵ/2-cover of {θ ∈Rd|∥θ∥2 ≤L} with respect to the 2-norm, and CA be an ϵ2/4-cover
of {A ∈Rd×d|∥A∥F ≤d1/2B2λ−1} with respect to the Frobenius norm. By Lemma H.3, we know:
Cθ
 ≤
 
1 + 4L/ϵ
d,
CA
 ≤

1 + 8d1/2B2/(λϵ2)
d2
.

By (H.10), for any V1 ∈V, there exists θ2 ∈Cθ and A2 ∈CA such that V2 parametrized by (θ2, A2)
satisfies dist(V1, V2) ≤ϵ. Hence, it holds that Nϵ ≤|Cθ| · |CA|, which leads to

log Nϵ ≤log |Cw| + log |CA| ≤d log(1 + 4L/ϵ) + d2 log

1 + 8d1/2B2/(λϵ2)

.

This concludes the proof.

H.4
Proof of Theorem B.1

In this section, we give the proof of Theorem B.1, which largely follows the proof of Theorem 5.2,
only with minor modifications of the argument of the variance estimation.

The following lemma bounds the estimation error by reference-advantage decomposition.
Lemma H.4 (Modified Variance-Aware Reference-Advantage Decomposition). There exist
{αi}i∈[d], where αi ∈[0, H], ∀i ∈[d], such that

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h ∥Eµ0
h[V ⋆,ρ
h+1(s)]αi∥Σ−1
h
|
{z
}
i

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h



K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h)


Σ−1
h
|
{z
}
ii

+ λ

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h

Eµ0
h
[bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi

Σ−1
h
|
{z
}
iii

+

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h



K
X

τ=1

ϕτ
hητ
h([bV ρ
h+1(s)]αi −[V ⋆,ρ
h+1(s)]αi)
bσ2
h(sτ
h, aτ
h)


Σ−1
h
|
{z
}
iv

,

where ητ
h([f]αi) =
 
P0
h[f]αi

(sτ
h, aτ
h) −[f(sτ
h+1)]αi

, for any function f : S →[0, H −1].

Proof of Theorem B.1. To prove this theorem, we bound the estimation error by Γh(s, a), then invoke
Lemma D.1 to get the results. First, we bound terms i-iv in Lemma H.4 at each step h ∈[H]
respectively to deduce Γh(s, a).

Bound i and iii:
We set λ = 1/H2 to ensure that for all (s, a, h) ∈S × A × [H], we have

i + iii ≤
√

λ
√

dH

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h
=
√

d

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h .
(H.11)

Bound ii:
For all (s, a, α) ∈S × A × [0, H], by definition we have bσh(s, a) ≥1. Thus, for
all (h, τ, i) ∈[H] × [K] × [d], we have ητ
h([V ⋆,ρ
h+1]αi)/bσh(sτ
h, aτ
h) ≤H. Note that V ⋆,ρ
H+1 is in-
dependent of D, we can directly apply Bernstein-type self-normalized concentration inequality

44


---Page Break---
Lemma I.2 and a union bound to obtain the upper bound. In concrete, we define the filtration
Fτ−1,h = σ({(sj
h, aj
h)}τ
j=1 ∪{sj
h+1}τ−1
j=1). Since V ⋆,ρ
h+1 and bσh(s, a) are independent of D, thus
ητ
h([V ⋆,ρ
h+1]αi)/bσh(sτ
h, aτ
h) is mean-zero conditioned on the filtration Fτ−1,h. By Lemma E.1 with
α = H, we have

VhV ⋆,ρ
h+1

(s, a) −˜O
 dH3

√

Kκ


≤bσ2
h(s, a) ≤

VhV ⋆,ρ
h+1

(s, a),
(H.12)

thus, for any αi ∈[0, H], we have

Vh[V ⋆,ρ
h+1]αi

(s, a) −˜O
 dH3

√

Kκ


≤

VhV ⋆,ρ
h+1

(s, a) −˜O
 dH3

√

Kκ


≤bσ2
h(s, a).
(H.13)

Further, we have

E
hητ
h([V ⋆,ρ
h+1]αi)
bσh(sτ
h, aτ
h)

2Fτ−1,h
i
= [Var[V ⋆,ρ
h+1]αi](sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h)
(H.14)

≤[V[V ⋆,ρ
h+1]αi](sτ
h, aτ
h)
bσ2
h(sτ
h, aτ
h)

= [V[V ⋆,ρ
h+1]αi](sτ
h, aτ
h) −˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h)
+
˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h)

≤1 +
˜O(dH3/
√

Kκ)
bσ2
h(sτ
h, aτ
h) −˜O(dH3/
√

Kκ)
(H.15)

≤1 + 2 ˜O
 dH3

√

Kκ


,
(H.16)

where (H.14) holds by the fact that bσ2
h(·, ·) is independent of D and (sτ
h, aτ
h) is Fτ−1,h measurable.
(H.15) holds by (H.13), and (H.16) holds by setting K ≥˜Ω(d2H6/κ) such that bσ2
h(sτ
h, aτ
h) −
˜O(dH3/
√

Kκ) ≥1 −˜O(dH3/
√

Kκ) ≥1/2. Further, by (H.16), our choice of K also ensures that
E
 
ητ
h([V ⋆,ρ
h+1]αi)
2|Fτ−1,h

= O(1). Then by Lemma I.2, we have


K
X

τ=1

ϕτ
hητ
h([V ⋆,ρ
h+1]αi)
bσ2
h(sτ
h, aτ
h)


Σ−1
h
≤˜O(
√

d).

This implies

ii ≤˜O(
√

d)

d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h .
(H.17)

Bound iv:
Following the same induction analysis procedure, we know that ∥[bV ρ
h+1]αi−[V ⋆,ρ
h+1]αi∥≤
˜O(
√

dH2/
√

Kκ). Using standard ϵ-covering number argument and Lemma I.1, we have

iv ≤˜O
d3/2H2

√

Kκ


d
X

i=1
∥ϕi(s, a)1i∥Σ−1
h .
(H.18)

To make it non-dominant, we require K ≥˜Ω(d2H4/κ). By (H.12), we have bσ2
h(sτ
h, aτ
h) ≤
[VhV ⋆
h+1](sτ
h, aτ
h), which implies that
 K
X

τ=1

ϕτ
hϕτ⊤
h
bσ2
h(sτ
h, aτ
h) + λI
−1
⪯
 K
X

τ=1

ϕτ
hϕτ⊤
h
[VhV ⋆
h+1](sτ
h, aτ
h) + λI
−1
.

Combining (H.11), (H.17) and (H.18), we have

inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a) −
\
inf
Ph(·|s,a)∈Uρ
h(s,a;µ0
h,i)[Ph bV ρ
h+1](s, a)


≤˜O(
√

d)

d
X

i=1
∥ϕi(s, a)1i∥Σ⋆−1
h
.

Define Γh(s, a) = ˜O(
√

d) Pd
i=1 ∥ϕi(s, a)1i∥Σ⋆−1
h
, we concludes the proof by invoking Lemma D.1.

45


---Page Break---
I
Auxiliary Lemmas

Lemma I.1 (Concentration of Self-Normalized Processes). [1, Theorem 1] Let {ϵt}∞
t=1 be a real-
valued stochastic process with corresponding filtration {Ft}∞
t=0. Let ϵt|Ft−1 be mean-zero and
σ-subGaussian; i.e. E[ϵt|Ft−1] = 0, and

∀λ ∈R,
E[eλϵt|Ft−1] ≤eλ2σ2/2.

Let {ϕt}∞
t=1 be an Rd-valued stochastic process where ϕt is Ft−1 measurable. Assume Λ0 is a d×d
positive definite matrix, and let Λt = Λ0 + Pt
s=1 ϕsϕ⊤
s . Then for any δ > 0, with probability at
least 1 −δ, we have for all t ≥0:


t
X

s=1
ϕsϵs



2

Λ−1
t
≤2σ2 log
det(Λt)1/2 det(Λ0)−1/2

δ


.

Lemma I.2 (Bernstein inequality for self-normalized martingales). [65, Theorem 2] Let {ηt}∞
t=1 be
a real-valued stochastic process. Let {Ft}∞
t=0 be a filtration, such that ηt is Ft-measurable. Assume
ηt also satisfies

|ηt| ≤R, E[ηt|Ft−1] = 0, E[η2
t |Ft−1] ≤σ2.

Let {xt}∞
t=1 be an Rd-valued stochastic process where xt is Ft−1 measurable and ∥xt∥≤L. Let
Λt = λId + Pt
s=1 xsx⊤
s . Then for any δ > 0, with probability at least 1 −δ, for all t > 0,


t
X

s=1
xsηs


Λ−1
t
≤8σ

r

d log

1 + tL2

λd


· log
4t2

δ


+ 4R log
4t2

δ


.

Lemma I.3. [27, Lemma H.5] Let ϕ : S×A →Rd satisfying ∥ϕ(x, a)∥≤C for all (x, a) ∈S×A.
For any K > 0 and λ > 0, define GK = PK
k=1 ϕ(xk, ak)ϕ(xk, ak)⊤+ λId where (xk, ak) ’s are
i.i.d. samples from some distribution ν over S × A. Let G = Ev[ϕ(x, a)ϕ(x, a)⊤]. Then, for any
δ ∈(0, 1), if K satisfies that

K ≥max
n
512C4G−12 log
2d

δ


, 4λ
G−1
o
,

then with probability at least 1 −δ, it holds simultaneously for all u ∈Rd that

∥u∥G
−1
K ≤
2
√

K
∥u∥G−1.

46


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We accurately summarize the paper’s contributions and scope in the abstract
and introduction (Section 1).

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

Justification: In Section 6, we claim that there are small gaps between the upper bounds
in Theorem 4.4 and Theorem 5.2 and lower bound in Theorem 6.1. The computation
tractability is discussed in Remark 4.2 in Section 4. The assumptions are formally stated in
Assumption 3.1 and Assumption 4.3.

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

47


---Page Break---
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: The assumptions are formally stated in Assumption 3.1 and Assumption 4.3.
Complete proofs are provided in the appendix.
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
Justification: The code of our implementation is available at https://github.com/
panxulab/Offline-Linear-DRMDP.
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

48


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: The code of our implementation is available at https://github.com/
panxulab/Offline-Linear-DRMDP.

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

Justification: The code of our implementation is available at https://github.com/
panxulab/Offline-Linear-DRMDP.

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

Justification: Not applicable to our experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

49


---Page Break---
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

Answer: [Yes]

Justification: All information is provided in the experiment section and the code of our imple-
mentation is available at https://github.com/panxulab/Offline-Linear-DRMDP.

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
Justification: the authors had reviewed the NeurIPS Code of Ethics and confirm that the
research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

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
Justification: this work focuses on the theoretical side of robust RL, and methods in this
paper do not lead to a direct path to any negative applications.

Guidelines:

50


---Page Break---
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
Justification: the paper poses no such risks.
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
Justification: the paper does not use existing assets.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

51


---Page Break---
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

Justification: the paper does not release new assets.

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

Justification: the paper does not involve crowdsourcing nor research with human subjects.

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

Justification: the paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

52


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

53


---Page Break---
