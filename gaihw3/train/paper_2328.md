An Adaptive Approach for Infinitely Many-armed
Bandits under Generalized Rotting Constraints

Jung-hun Kim
Seoul National University
Seoul, South Korea
junghunkim@snu.ac.kr

Milan Vojnovi´c
London School of Economics
London, United Kingdom
m.vojnovic@lse.ac.uk

Se-Young Yun
KAIST AI
Seoul, South Korea
yunseyoung@kaist.ac.kr

Abstract

In this study, we consider the infinitely many-armed bandit problems in a rested
rotting setting, where the mean reward of an arm may decrease with each pull,
while otherwise, it remains unchanged. We explore two scenarios regarding the
rotting of rewards: one in which the cumulative amount of rotting is bounded by
VT , referred to as the slow-rotting case, and the other in which the cumulative
number of rotting instances is bounded by ST , referred to as the abrupt-rotting
case. To address the challenge posed by rotting rewards, we introduce an algorithm
that utilizes UCB with an adaptive sliding window, designed to manage the bias
and variance trade-off arising due to rotting rewards. Our proposed algorithm
achieves tight regret bounds for both slow and abrupt rotting scenarios. Lastly, we
demonstrate the performance of our algorithm using numerical experiments.

1
Introduction

We consider multi-armed bandit problems [15], which are fundamental sequential learning problems
where an agent plays an arm at each time and receives a corresponding reward. The core challenge
lies in balancing the exploration-exploitation trade-off. Bandit problems have significant implications
across diverse real-world applications, such as recommendation systems [17] and clinical trials [23].
In a recommendation system, each arm could represent an item, and the objective is to maximize the
click-through rate by making effective recommendations.

In practice, the mean rewards associated with arms may decrease over repeated plays. For instance,
in content recommendation systems, the click rates for each item (arm) may diminish due to user
boredom with repeated exposure to the same content. Another example is evident in clinical trials,
where the efficacy of a medication can decline over time due to drug tolerance induced by repeated
administration. The decline in mean rewards resulting from playing arms, referred to as (rested)
rotting bandits, has been studied by Levine et al. [16], Seznec et al. [20, 21]. The previous work
focuses on finite K arms, in which Seznec et al. [20] proposed algorithms achieving ˜O(
√

KT) regret.
This suggests that rotting bandits with a finite number of arms are no harder than the stationary case.

However, in real-world scenarios like recommendation systems, where the content items such as
movies or articles are numerous, prior methods encounter limitations as the parameter K becomes
large, resulting in trivial regret. This emphasizes the necessity of studying rotting scenarios with
infinitely many arms, particularly when there is a lack of information about the features of each item.
The consideration of infinitely many arms for rested rotting bandits fundamentally distinguishes these
problems from those with a finite number of arms, as we will explain later.

The study of multi-armed bandit problems with an infinite number of arms has been extensively
conducted in the context of stationary rewards [6, 24, 8, 10, 5], where the agent has no chance to
play all the arms at least once until horizon time T. Initially, the distribution of the mean rewards for

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
Table 1: Summary of our regret bounds.

Type
Regret upper bounds
for β ≥1

Regret upper bounds
for 0 < β < 1

Regret lower bounds
for β > 0

Slow rotting (VT )

Theorem 3.1:
˜O

max
n
V

1
β+2
T
T

β+1
β+2 , T

β
β+1
o
Theorem 3.1:
˜O

max
n
V

1
3
T T
2
3 ,
√

T
o
Theorem 4.1:

Ω

max
n
V

1
β+2
T
T

β+1
β+2 , T

β
β+1
o

Abrupt rotting (ST )

Theorem 3.3:
˜O

max
n
S

1
β+1
T
T

β
β+1 , ¯VT
o
Theorem 3.3:
˜O

max
n√ST T, ¯VT
o
Theorem 4.2:

Ω

max
n
S

1
β+1
T
T

β
β+1 , ¯VT
o

the arms was assumed to be uniform over the interval [0, 1] [6, 8]. This assumption was expanded
to include a much wider range of distributions satisfying P(µ(a) > µ∗−x) = Θ(xβ), for a
parameter β > 0, where µ(a) represents the mean reward of arm a and µ∗is the mean reward of
the best-performing arm [24, 10, 5]. Additionally, feature information for each arm is not required
for multi-armed bandit problems with infinitely many arms, which differs from linear bandits [1] or
continuum-armed bandits [3, 14], where feature information for each arm, either for the Lipschitz
or linear structure, is involved. While Kim et al. [13], as the closest work, explores the concept of
diminishing rewards in the context of bandits with infinitely many arms, their focus is restricted to
the case of the maximum rotting rate constraint, where the amount of rotting at each time step is
bounded by ρ (= o(1)). This naturally directs focus towards regret regarding the maximum rotting
rate rather than the total rotting rate over the time horizon. Furthermore, their focus is limited to the
case where the initial mean rewards are uniformly distributed (β = 1).

In our study, we explore rotting bandits with infinitely many arms, subject to generalized initial mean
reward distribution with β > 0 and, importantly, generalized constraints on the rate at which the mean
reward of an arm declines. Our investigation into diminishing, or ‘rotting,’ rewards encompasses two
scenarios: one with the total amount of rotting bounded by VT , and the other with the total number of
rotting instances bounded by ST . This allows us to capture characteristics of entire rotting rates over
the time horizon. Similar constraints of VT or ST regarding nonstationarity have been explored in the
context of nonstationary finite K-armed bandit problems [7, 4, 19], where the reward distribution
changes over time independently of the agent. Following established terminology for nonstationary
bandits, we denote the environment with a bounded total amount of rotting as the slow rotting (VT )
case and the one with a bounded total number of rotting instances as the abrupt rotting (ST ) case.

Here we discuss why (rested) rotting bandits for infinitely many arms are fundamentally different
from those for finite arms. In the case of finite arms, rested rotting is known to be no harder than
stationary case [20, 21]. This result arises from the confinement of mean rewards of optimal arms and
played arms within confidence bounds, even under rested rotting (as demonstrated in Lemma 1 of
Seznec et al. [20, 21]). However, in the case of infinite arms under distribution for initial mean reward
that allows for an infinite number of near-optimal arms, there always exist near-optimal arms outside
of explored arms. Therefore, the mean reward gap may not be confined within confidence bounds.
This fundamental difference from finite-armed rotting bandits introduces additional challenges. In
our setting of infinite arms, there exists an additional cost for exploring new (unexplored) arms to
find near-optimal arms while eliminating explored suboptimal arms. If the total rotting effect on
explored arms is significant, then the frequency at which new near-optimal arms must be sought
increases substantially, resulting in a large regret. This is why the rested rotting significantly affects
the exploration cost regarding VT or ST in our setting, which differs from the case of finite arms.

To solve our problem, we introduce algorithms that employ an adaptive sliding window mechanism,
effectively managing the tradeoff between bias and variance stemming from rotting rewards. Notably,
to the best of our knowledge, this is the first work to consider slow and abrupt rotting scenarios, in the
context of infinitely many-armed bandits. Furthermore, it is the first work to consider the generalized
initial mean reward distribution for rotting bandits with infinitely many arms.

Summary of our Contributions.
The key contributions of this study are summarized in the
following points. Please refer to Table 1 for a summary of our regret bounds.

• To address the slow and abrupt rotting scenarios, we propose a UCB-based algorithm using an
adaptive sliding window and a threshold parameter. This algorithm allows for effectively managing
the bias and variance trade-off arising from rotting rewards.

2


---Page Break---
• In the context of slow rotting (VT ) or abrupt rotting (ST ), for any β > 0, we present regret upper
bounds achieved by our algorithm with an appropriately tuned threshold parameter. It is noteworthy
that VT , ST , and β are being considered for the first time in the context of rotting bandits with
infinitely many arms.

• We establish regret lower bounds for both slow rotting and abrupt rotting scenarios. These regret
lower bounds imply the tightness of our upper bounds when β ≥1. In the other case, when
0 < β < 1, there is a gap between our upper bounds and the corresponding lower bounds, similar to
what can be found in related literature, which is discussed in the paper.

• Lastly, we demonstrate the performance of our algorithm through numerical experiments on
synthetic datasets, validating our theoretical results.

2
Problem Statement

We consider rotting bandits with infinitely many arms where the mean reward of an arm may decrease
when the agent pulls the arm. Let A be the set of infinitely many arms and let µt(a) denote the
unknown mean reward of arm a ∈A at time t. At each time t, an agent pulls arm aπ
t ∈A
according to policy π and observes stochastic reward rt given by rt = µt(aπ
t ) + ηt, where ηt is a
noise term following a 1-sub-Gaussian distribution. To simplify, we use at for aπ
t when there is
no confusion about the policy. We assume that initial mean rewards {µ1(a)}a∈A are i.i.d. random
variables on [0, 1], a widely accepted assumption in the context of infinitely many-armed bandits
[8, 6, 24, 10, 5, 13].

As in Wang et al. [24], Carpentier and Valko [10], Bayati et al. [5], we consider, to our best knowledge,
the most general condition on the distribution of the initial mean reward of an arm, satisfying the
following condition: there exists a constant β > 0 such that for every a ∈A and all x ∈[0, 1],

P(µ1(a) > 1 −x) = P(∆1(a) < x) = Θ(xβ),
(1)
where ∆1(a) = 1 −µ1(a) is the initial sub-optimality gap. As noted in [24, 10, 5], Eq.(1) is
a non-trivial condition only when x approaches 0, as for any constant x ∈(0, 1], it becomes
P(∆1(a) < x) = Θ(1), which may accommodate a wide range of distributions. It is noteworthy
that the larger the value of β, the smaller the probability of sampling a good arm. Furthermore, the
uniform distribution is a special case when β = 1. Importantly, our work allows for a wider range of
distributions satisfying (1) for any constant β > 0 than the uniform distribution (β = 1) considered
in Kim et al. [13]. Additional discussion is deferred to Appendix A.2.

The rotting of arms is defined as follows. At each time t ≥1, the mean rewards of arms are updated as
µt+1(a) = µt(a) −ρt(a),
where ρt(at) ≥0 for the pulled arm at and ρt(a) = 0 for every a ∈A/{at}, which implies that the
rotting may occur only for the pulled arm at each time. Note that, for every a ∈A and t ≥2, it holds
µt(a) = µ1(a) −Pt−1
s=1 ρs(a), allowing µt(a) to take negative values. For notation simplicity, in
what follows, we write ρt for ρt(at) when there is no confusion. We refer to ρ1, ρ2, . . . as rotting
rates. We also use the notation [m] := {1, . . . , m}, for any integer m ≥1.

We consider two cases for rotting rates: (a) slow rotting case where, for given VT ≥0, the cumulative
amount of rotting is required to satisfy the slow rotting constraint PT −1
t=1 ρt ≤VT , and (b) abrupt
rotting case where, for given ST ∈[T], the cumulative number of rotting instances (plus one) is
required to satisfy the abrupt rotting constraint 1 + PT −1
t=1 1(ρt ̸= 0) ≤ST . The values of rotting
rates of pulled arms, {ρt}t∈[T −1], are assumed to be determined by an adversary, described as follows.
Assumption 2.1 (Adaptive Adversary). At each time t ∈[T], the value of the rotting rate ρt ≥0 is
arbitrarily determined immediately after the agent pulls at, subject to the constraint of either slow
rotting for a given VT or abrupt rotting for a given ST .
Remark 2.2. The adaptive adversary under the slow rotting constraint (VT ) is more general than that
in Kim et al. [13], in which the adversary is under a maximum rotting rate constraint; that is, for
given ρ = o(1), ρt ≤ρ for all t ∈[T −1]. This is because our adversary is under a weaker constraint
bounding the total sum of the rotting rates rather than each individual rotting rate. Additionally, the
abrupt rotting constraint (ST ) is fundamentally different from the maximum rotting constraint [13]
because the adversary for abrupt rotting is under a constraint on the total number of rotting instances
rather than the magnitude of rotting rates.

3


---Page Break---
Our problem’s objective is to find a policy that minimizes the expected cumulative regret over a time
horizon of T time steps. For a given policy π, the regret is defined as E[Rπ(T)] = E[PT
t=1(1 −
µt(aπ
t ))]. The use of 1 in the regret definition for the optimal mean reward is justified because among
the infinite arms with initial mean rewards following the distribution specified in (1), there always
exists an arm whose mean reward is sufficiently close to 1.1

We note that while we have ST ≤T because the number of rotting instances is at most T −1,
the upper bound for VT may not exist due to the lack of a constraint on the values of ρt’s. Here
we discuss an assumption for the cumulative amount of rotting. In the case of PT −1
t=1 ρt > T, the
problem becomes trivial as shown in the following proposition.

Proposition 2.3. In the case of PT −1
t=1 ρt > T, there always exists a rotting adversary that incurs
regret of Ω(T) and a simple policy that samples a new arm every round achieves the optimal regret
of Θ(T).

Proof. The proof is provided in Appendix A.3

From the above proposition, when PT −1
t=1 ρt > T, the regret lower bound of this problem is Ω(T),
which can be achieved by a simple policy. Therefore, we consider the following assumption for the
region of non-trivial problems.

Assumption 2.4. PT −1
t=1 ρt ≤T.

Notably, from the above assumption, we consider VT ≤T for the slow rotting case. We also note
that the assumption is not strong, as it frequently arises in real-world scenarios and is more general
than the assumption made in prior work, as described in the following remarks.

Remark 2.5. The assumption of PT −1
t=1 ρt ≤T is satisfied if mean rewards are under the constraint
of 0 ≤µt(at) ≤1 for all t ∈[T], because this condition implies ρt ≤1 for all t ∈[T]. Such a
scenario is frequently encountered in real-world applications, where reward is represented by metrics
like click rates or (normalized) ratings in content recommendation systems.

Remark 2.6. Our rotting scenario with PT −1
t=1 ρt ≤T is more general in scope than the one with the
maximum rotting rate constraint where ρt ≤ρ = o(1) for all t ∈[T −1], which was explored in
Kim et al. [13]. This is because for our setting, ρt is not necessarily bounded by o(1), and under the
maximum rotting constraint, the condition PT −1
t=1 ρt ≤T is always satisfied.

3
Algorithms and Regret Analysis

We propose an algorithm (Algorithm 1) utilizing an adaptive sliding window for delicately controlling
bias and variance tradeoff of the mean reward estimator from rotting rewards, drawing on insights
from [4, 21]. This is why our algorithm can adapt to varying rotting rates ρt and achieve tight regret
bounds with respect to VT or even ST . Furthermore, our algorithm accommodates the general mean
reward distribution with β > 0 by employing a carefully optimized threshold parameter.

Here we describe our proposed algorithm in detail. We define bµ[t1,t2](a) = Pt2
t=t1 rt1(at =
a)/n[t1,t2](a) where n[t1,t2](a) = Pt2
t=t1 1(at = a) for t1 ≤t2. Then for window-UCB index of the
algorithm, we define WUCB(a, t1, t2, T) = bµ[t1,t2](a) +
p

12 log(T)/n[t1,t2](a). In what follows,
‘selecting an arm’ means that a policy chooses an arm before pulling it. In Algorithm 1, we first
select an arbitrary new arm a ∈A′ without prior knowledge regarding the arms in A′, denoting
the corresponding time as t(a). We define Tt(a) as the set of starting times for sliding windows of
doubling lengths, defined as Tt(a) = {s ∈[T] : t(a) ≤s ≤t −1 and s = t −2i−1 for some i ∈N}.
Then the algorithm pulls the arm consecutively until the following threshold condition is satisfied:
mins∈Tt(a) WUCB(a, s, t −1, T) < 1 −δ, in which the sliding window having minimized window-
UCB is utilized for adapting nonstationarity. If the threshold condition holds, then the algorithm
considers the arm to be a sub-optimal (bad) arm and withdraws the arm. Then it selects a new arm
and repeats this procedure.

1This assertion follows from the fact that for any ϵ > 0, there exists an arm a in A excluding rotted arms
such that ∆1(a) < ϵ with probability 1, as limn→∞(1 −P(∆1(a) ≥ϵ)n) = 1.

4


---Page Break---
Algorithm 1 UCB-Threshold with Adaptive Sliding Window
Input: T, δ, A; Initialize: A′ ←A
Select a new arm a ∈A′; Pull arm a and get reward r1
t(a) ←1
for t = 2, . . . , T do

if mins∈Tt(a) WUCB(a, s, t −1, T) < 1 −δ then

A′ ←A′/{a}
Select a new arm a ∈A′; Pull arm a and get reward rt
t(a) ←t
else

Pull arm a and get reward rt

Figure 1: Illustrations for the adaptive sliding window: (left) the effect of the sliding window length
on the mean reward estimation, (right) sliding window candidates with doubling lengths.

Utilizing the adaptive sliding window having minimized window UCB index enhances the algorithm’s
ability to dynamically identify poorly-performing arms across varying rotting rates. This adaptability
is achieved by managing the tradeoff between bias and variance. The concept is depicted in Figure 1
(left), where an arm a undergoes multiple rotting events. WUCB with a smaller window exhibits min-
imal bias with the arm’s most recent mean reward but introduces higher variance. Conversely, WUCB
with a larger window displays increased bias but reduced variance. In this visual representation, the
value of WUCB with a small window reaches a minimum, enabling the algorithm to compare this
value with 1 −δ to identify the suboptimal arm. Moreover, as illustrated in Figure 1 (right), by taking
into account the constraint of s = t −2i−1 for the size of the adaptive windows, we can reduce the
computation time for determining the appropriate window and reduce the required memory from
O(t) to O(log t), respectively, for each time t.

Having introduced our algorithm, we compare it with the previously proposed algorithm UCB-TP [13],
which is tailored for the maximum rotting rate constraint ρt ≤ρ (= o(1)) for all t > 0 and the
uniform initial mean reward distribution (β = 1). The mean reward estimator in UCB-TP considers
the worst-case scenario with the maximum rotting rate ρ as ˜µo
t(a) −ρnt(a) where ˜µo
t is an estimator
for the initial mean reward, nt(a) is the number of times arm a is pulled until time t −1, and ρnt(a)
is for reducing the bias from the worst-case rotting, which leads to achieving a regret bound of
˜O(max{ρ1/3T,
√

T}). This estimator is not appropriate for dealing with our generalized rotting
constraints because it aims to attain the regret bound regarding the maximum rotting rate ρ without
adequately addressing individual ρt values. Our algorithm resolves this by using an adaptive sliding
window estimator, which can handle rotting rates carefully. Furthermore, it can accommodate any
constant β > 0 by using a carefully optimized δ, as shown below.

Slow Rotting (VT ).
Here we consider the case of slow rotting, where, recall, the adaptive adversary
is constrained such that the total amount of rotting is bounded by VT . We analyze the regret of
Algorithm 1 with tuned δ using β and VT . We define δV (β) = max{(VT /T)1/(β+2), 1/T 1/(β+1)}
when β ≥1 and δV (β) = max{(VT /T)1/3, 1/
√

T} when 0 < β < 1. The algorithm with δV (β)
achieves a regret bound in the following theorem.

5


---Page Break---
Theorem 3.1. The policy π of Algorithm 1 with δ = δV (β) achieves:

E[Rπ(T)] =

(
˜O(max{V

1
β+2
T
T

β+1
β+2 , T

β
β+1 })
for β ≥1,
˜O(max{V

1
3
T T
2
3 ,
√

T})
for 0 < β < 1.

We observe that when β increases above 1, the regret bound becomes worse because the likeli-
hood of sampling a good arm decreases. However, when β decreases below 1, the regret bound
remains the same due to the inability to avoid a certain level of regret arising from estimat-
ing the mean reward. Further discussion will be provided later. Also, we observe that when
VT = O(max{1/T 1/(β+1), 1/
√

T}) where the problem becomes near-stationary, the regret bound
in Theorem 3.1 matches the previously known regret bound for stationary infinitely many-armed
bandits, ˜O(max{T β/(β+1),
√

T}), as shown in Wang et al. [24], Bayati et al. [5].

Proof sketch. The full proof is provided in Appendix A.4. Here we outline the main ideas of the
proof. There are several technical challenges involved in regret analysis, such as dealing with varying
ρt individually with respect to the total rotting budget of VT , adaptive estimation in our algorithm,
and the generalized distributions of initial mean rewards of arms with parameter β > 0, none of
which appear in Kim et al. [13].

We separate the regret into two components: one associated with pulling initially good arms and
another with pulling initially bad arms. An arm a is said to be good if µ1(a) ≥1 −2δ and, otherwise,
it is said to be bad. The reason why the separation is required is that our adaptive algorithm has
different behaviors depending on the category of arms. Good arms may be pulled repeatedly when
rotting rates are sufficiently small but bad arms are not. We write Rπ(T) = RG(T) + RB(T), where
RG(T) is the regret from good arms and RB(T) is the regret from bad arms.

We first provide a bound for E[RG(T)]. For analyzing regret from good arms, we analyze the
cumulative amount of rotting while pulling a selected good arm before withdrawing the arm by the
algorithm. Let AG
T be a set of distinct good arms selected until T, t1(a) be the initial time step at
which arm a is pulled, and t2(a) be the final time step at which the arm is pulled by the algorithm so
that the threshold condition holds when t = t2(a) + 1. For simplicity, we use t1 and t2 for t1(a) and
t2(a), when there is no confusion. For any time steps n ≤m, we define V[n,m](a) = Pm
t=n ρt(a)
and ρ[n,m](a) = V[n,m](a)/n[n,m](a). We show that the regret is decomposed as

RG(T) =
X

a∈AG
T


∆1(a)n[t1,t2](a) +

t2
X

t=t1+1
V[t1,t−1](a)

,
(2)

which consists of regret from the initial mean reward and the cumulative amount of rotting for each
arm. For the first term of P

a∈AG
T ∆1(a)n[t1,t2](a) in (2), since ∆1(a) = O(δ) from the definition

of good arms a ∈AG
T , we have E[P

a∈AG
T ∆1(a)n[t1,t2](a)] = O(δT).

The main difficulty in (2) lies in dealing with the second term, P
a∈AG
T
Pt2
t=t1+1 V[t1,t−1](a) , where
we need to analyze the amount of cumulative rotting until the arm is eliminated by using the adaptive
threshold condition. A careful analysis of the adaptive threshold policy is required to limit the total
variation of rotting. By examining the estimation errors arising from variance and bias due to the
adaptive threshold condition, we can establish an upper bound for the cumulative amount of rotting as

X

a∈AG
T

t2
X

t=t1+1
V[t1,t−1](a) = ˜O

Tδ + VT +
X

a∈AG
T

V[t1,t2−2](a)
1
3 n[t1,t2−2](a)
2
3

.
(3)

Therefore, from δ = δV (β), VT ≤T, and Eqs. (2) and (3), using Hölder’s inequality, we have

E[RG(T)] =

(
˜O(max{V

1
β+2
T
T

β+1
β+2 , T

β
β+1 })
for β ≥1,
˜O(max{V

1
3
T T
2
3 ,
√

T})
for 0 < β < 1.
(4)

Next, we provide a bound for E[RB(T)]. We employ episodic regret analysis, defining an episode
as the time steps between consecutively selected distinct good arms by the algorithm. By analyzing

6


---Page Break---
bad arms within each episode, we can derive an upper bound for the overall regret arising from
bad arms. We define the regret from bad arms over mG episodes as RB
mG. We first consider the
case of VT = ω(max{1/
√

T, 1/T 1/(β+1)}). In this case, by setting mG = ⌈2VT /δ⌉, we can show
that RB(T) ≤RB
mG with a high probability. By analyzing RB
mG with the episodic analysis, we can

show that E[RB(T)] ≤E[RB
mG] = ˜O(max{T

β+1
β+2 V

1
β+2
T
, T
2
3 V

1
3
T }). As in the similar manner, when
VT = O(max{1/
√

T, 1/T 1/(β+1)}), by setting mG = C3 for some constant C3 > 0, we can show
that E[RB(T)] ≤E[RB
mG] = ˜O(max{T

β
β+1 ,
√

T}). From the above two inequalities, we have

E[RB(T)] =

(
˜O(max{V

1
β+2
T
T

β+1
β+2 , T

β
β+1 })
for β ≥1,
˜O(max{V

1
3
T T
2
3 ,
√

T})
for 0 < β < 1.
(5)

Finally, from (4) and (5), we can conclude the proof from E[Rπ(T)] = E[RG(T)] + E[RB(T)].

Remark 3.2. We compare our result in Theorem 3.1 with that in Kim et al. [13], which, recall, is
under the maximum rotting rate constraint ρt ≤ρ = o(1) for all t and uniform distribution of initial
mean rewards (β = 1). For a fair comparison, we consider an oblivious adversary for rotting rates
where the values of ρt’s are determined before an algorithm is run, which may imply VT = PT −1
t=1 ρt
and ρ = maxt∈[T −1] ρt. Then with β = 1, from VT ≤Tρ, we can observe that the regret bound of

Algorithm 1 is tighter than that of UCB-TP [13] as ˜O(max{V

1
3
T T
2
3 ,
√

T}) ≤˜O(max{ρ
1
3 T,
√

T}),
where the latter is the regret bound of UCB-TP. We will demonstrate this in our numerical results.

Abrupt Rotting (ST ).
Here we consider abruptly rotting reward distribution under the constraint of
ST . We consider Algorithm 1 with δ newly tuned by ST and β. We define δS(β) = (ST /T)1/(β+1)

for β ≥1 and δS(β) = (ST /T)1/2 for 0 < β ≤1. We also define ¯VT = PT −1
t=1 E[ρt]. In the
following theorem, we present a regret upper bound for Algorithm 1 with δS(β).

Theorem 3.3. The policy π of Algorithm 1 with δ = δS(β) achieves:

E[Rπ(T)] =

(
˜O(max{S

1
β+1
T
T

β
β+1 , ¯VT })
for β ≥1,
˜O(max{√ST T, ¯VT })
for 0 < β < 1.

As in the slow rotting case, for the abrupt rotting case (ST ), we observe that when β increases above
1, the regret bound in the above theorem worsens as the likelihood of sampling a good arm decreases.
When β decreases below 1, the regret bound remains the same because we cannot avoid a certain
level of regret arising from estimating the mean reward of an arm. Additionally, we observe that
the regret bound is linearly bounded by ¯VT , which is attributed to the algorithm’s necessity to pull a
rotted arm at least once to determine its status as bad. Later, in the analysis of regret lower bounds,
we will establish the impossibility of avoiding ¯VT regret in the worst-case. Notably, in the typical
cases where 0 ≤ρt ≤1 for all t > 0, as discussed in Remark 2.5, ¯VT is negligible in the regret
bound from ¯VT ≤ST ≤T. Furthermore, we observe that for the case of ST = 1, where the problem
becomes stationary (implying ¯VT = 0), the regret bound matches the previously known regret bound
of ˜O(max{T β/(β+1),
√

T}) for the stationary infinitely many-armed bandits [24, 5].

Proof sketch. The full proof is provided in Appendix A.5. Here we provide a proof outline. We
follow the proof framework of Theorem 3.1 but the main difference lies in carefully dealing with
substantially rotted arms. For the ease of presentation, we consider each arm that experiences abrupt
rotting as if it were newly selected by the algorithm, treating the arm before and after abrupt rotting
as distinct arms. The definition of a good arm and a bad arm is based on the mean reward at the
time when it is newly selected. Then we divide the regret into regret from good and bad arms as
Rπ(T) = RG(T) + RB(T). From the definition of good arms, we can easily show that

E[RG(T)] = O(δS(β)T) =

(
˜O(S

1
β+1
T
T

β
β+1 )
for β ≥1,
˜O(√ST T)
for 0 < β < 1.

For dealing with RB(T), we partition the regret into two scenarios: one where the bad arm is initially
bad sampled from the distribution of (1) and another where it becomes bad after rotting. This can be

7


---Page Break---
Figure 2: Adaptive sliding window for abrupt rotting.

expressed as RB(T) = RB,1(T) + RB,2(T). Then for the former regret, RB,1(T), as in the proof of
Theorem 3.1, by using the episodic analysis with mG = ST , we can show that

E[RB,1(T)] ≤E[RB
mG] =

(
˜O(S

1
β+1
T
T

β
β+1 )
for β ≥1,
˜O(√ST T)
for 0 < β < 1.

For the regret from rotted bad arms, RB,2(T), it is critical to analyze significant rotting instances to
obtain a tight bound with respect to ST , a factor not addressed in the regret analysis of slow rotting
(VT ) in Theorem 3.1. We analyze that when there exists significant rotting, then the algorithm can
efficiently detect it as a bad arm and eliminate it by pulling it at once. From this analysis, we have

E[RB,2(T)] =

(
˜O(max{S

β
β+1
T
T
1
β+1 , ¯VT })
for β ≥1,
˜O(max{√ST T, ¯VT })
for 0 < β < 1.

Putting all the results together with E[Rπ(T)] = E[RG(T)]+E[RB,1(T)]+E[RB,2(T)] and ST ≤T,
we can conclude the proof.

Remarkably, our proposed method, utilizing an adaptive sliding window, yields a tight bound (lower
bounds will be presented later) not only for slow rotting but also for abrupt rotting (ST ) scenarios
characterized by a limited number of rotting instances. The rationale behind the effectiveness of the
adaptive sliding window in controlling the bias and variance tradeoff with respect to abrupt rotting is
as follows. It can be observed that the adaptive threshold condition of mins∈Tt(a) WUCB(a, s, t −
1, T) < 1 −δ is equivalent to the condition of WUCB(a, s, t −1, T) < 1 −δ for some s such that
t1(a) ≤s ≤t −1 (ignoring the computational reduction trick). The latter expression represents
the threshold condition tested for every time step before t, encompassing the time step immediately
following an abrupt rotting event. Consequently, as illustrated in Figure 2, this adaptive threshold
condition can identify substantially rotted arms by mitigating bias and variance using the window
starting from the time step following the occurrence of rotting.

Slow rotting (VT ) and abrupt rotting (ST ).
In what follows, we study the case of rotting under both
slow rotting and abrupt rotting constraints. In this case, Algorithm 1, with δ = min{δV (β), δS(β)},
can achieve a tighter regret bound as noted in the following corollary, which can be obtained from
Theorems 3.1 and 3.3.
Corollary 3.4. Let RV and RS be defined as

RV :=

(
max{V

1
β+2
T
T

β+1
β+2 , T

β
β+1 }
for β ≥1,
max{V 1/3
T
T 2/3,
√

T} for 0 < β < 1
and RS :=

(
max{S

1
β+1
T
T

β
β+1 , VT }
for β ≥1,
max{√ST T, VT }
for 0 < β < 1.

The policy π of Algorithm 1 with δ = min{δV (β), δS(β)} achieves the regret bound of E[Rπ(T)] =
˜O (min {RV , RS}) .

Case without Prior Knowledge of VT , ST , and β.
Here we study the case when the algorithm
does not have prior information about the values of VT , ST , and β under the constraints of VT
and ST . These parameters play a crucial role in determining the optimal threshold parameter δ in
Algorithm 1. We propose an algorithm based on estimating the optimal threshold parameter δ directly
(Algorithm 2), rather than estimating each unknown parameter separately, employing the Bandit-
over-Bandit (BoB) approach [11]. Under assumptions concerning the bounds for the cumulative

8


---Page Break---
amount of rotting and a constrained version of the adaptive adversary for rotting rates, which are
less general than Assumptions 2.1 and 2.4 but still more general than those in Kim et al. [13], the
algorithm achieves a regret bound of E[Rπ(T)] = ˜O(min {RV , RS}+max{T (2β+1)/(2β+2),
√

T}).
The additional cost arises from learning δ compared to the regret bound of Corollary 3.4. Further
details of the algorithm and regret analysis are provided in Appendix A.6.

4
Regret Lower Bounds

In this section, we present regret lower bounds for our problem under Assumptions 2.1 and 2.4
to provide guidance on the tightness of our regret upper bounds. For the regret lower bounds, we
consider worst-case instances of rotting rates. In the following theorems, we provide regret lower
bounds for slow rotting (VT ) and abrupt rotting (ST ), respectively.

Theorem 4.1. For the slow rotting case with the constraint VT and β > 0, for any policy π, there
always exists a rotting rate adversary such that the regret of π satisfies

E[Rπ(T)] = Ω

max
n
V

1
β+2
T
T

β+1
β+2 , T

β
β+1
o
.

Proof. The proof is provided in Appendix A.8.

Theorem 4.2. For the abrupt rotting case with the constraint ST and β > 0, for any policy π, there
always exists a rotting rate adversary such that the regret of π satisfies

E[Rπ(T)] = Ω

max
n
S

1
β+1
T
T

β
β+1 , ¯VT
o
.

Proof. The proof is provided in Appendix A.9.

For the abrupt rotting (ST ) case, it is unavoidable to incur a Ω( ¯VT ) regret because an arm may only
be rotted once and any algorithm pulls this rotted arm at least once in the worst case. From Table 1,
we can observe that Algorithm 1 achieves near-optimal regret when β ≥1. The optimality proven
only for β ≥1 has also been observed for stationary infinitely many-armed bandits [5, 24]. We
believe that our regret upper bounds are near-optimal across the entire range of β. Achieving tighter
regret lower bounds when β < 1 is left for future research; see Appendix A.1 for further discussion.

5
Experiments

In this section, we present numerical results validating some claims of our theoretical analysis.2 We
use randomly generated datasets under a uniform distribution for initial mean rewards (β = 1).

We first compare the performance of our Algorithms 1 and 2 with UCB-TP [13], the state-of-the-art
algorithm for the rotting setting, and SSUCB [5], a near-optimal algorithm for stationary infinitely
many-armed bandits. For comparison with UCB-TP, recall our discussion in Remark 3.2. We set
the rotting rates such that ρt = 1/(t log(T)) for all t, for which ρ = ρ1 = 1/ log(T) = o(1),
VT = O(1), and ST = T. In Figure 5 (a), we can observe that Algorithms 1 and 2 perform better
than UCB-TP and SSUCB (and Algorithm 1 outperforms Algorithm 2), which is in agreement with our
theoretical analysis for the case β = 1. In this case, the regret bounds for Algorithms 1 and 2 are
˜O(T 2/3) and ˜O(T 3/4) from Corollary 3.4 and Theorem A.15, respectively, which are tighter than
the regret bound of ˜O(T/ log(T)1/3) for UCB-TP.

We next compare the performance of our Algorithms with benchmarks for smaller or larger β. In
Figure 5 (b,c), we can observe that our algorithms outperform the benchmarks for β = 0.5 and β = 2.
Additional experiments can be found in Appendix A.10.

2The
source
code
is
available
at
https://github.com/junghunkim7786/
An-Adaptive-Approach-for-Infinitely-Many-armed-Bandits-under-Generalized-Rotting-Constraints

9


---Page Break---
0
1
2
3
4
5
T
1e6

0

1

2

3

E[R (T)]

1e6

SSUCB
UCB-TP
Algorithm 2
Algorithm 1

(a) β = 1

0
1
2
3
4
5
T
1e6

0.0

0.5

1.0

1.5

2.0

2.5

3.0

E[R (T)]

1e6

SSUCB
UCB-TP
Algorithm 2
Algorithm 1

(b) β = 0.5

0
1
2
3
4
5
T
1e6

0

1

2

3

4

E[R (T)]

1e6

SSUCB
UCB-TP
Algorithm 2
Algorithm 1

(c) β = 2

Figure 3: Regret Performance comparison between our algorithms and benchmarks.

6
Conclusion

We explore the challenges of infinitely many-armed bandit problems with rotting rewards, focusing
on slow rotting (VT ) and abrupt rotting (ST ) scenarios. To address these challenges, we propose an
algorithm incorporating an adaptive sliding window, which achieves tight regret bounds for both
cases. We also provide regret lower bounds for both slow rotting and abrupt rotting cases. Lastly, we
demonstrate our algorithm using synthetic datasets.

7
Acknowledgements

JK was supported by the Global-LAMP Program of the National Research Foundation of Korea (NRF)
grant funded by the Ministry of Education (No. RS-2023-00301976). SY was supported by Institute
of Information & communications Technology Planning & Evaluation (IITP) grant funded by the
Korea government(MSIT) (No. RS-2022-II220311, Development of Goal-Oriented Reinforcement
Learning Techniques for Contact-Rich Robotic Manipulation of Everyday Objects)

References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. Advances in neural information processing systems, 24, 2011.

[2] Peter Auer, Nicoló Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic
multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002.

[3] Peter Auer, Ronald Ortner, and Csaba Szepesvári. Improved rates for the stochastic continuum-
armed bandit problem. In International Conference on Computational Learning Theory, pages
454–468. Springer, 2007.

[4] Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an
unknown number of distribution changes. In Conference on Learning Theory, pages 138–158.
PMLR, 2019.

10


---Page Break---
[5] Mohsen Bayati, Nima Hamidi, Ramesh Johari, and Khashayar Khosravi. The unreasonable
effectiveness of greedy algorithms in multi-armed bandit with many arms. arXiv preprint
arXiv:2002.10121, 2020.

[6] Donald A. Berry, Robert W. Chen, Alan Zame, David C. Heath, and Larry A. Shepp. Bandit
problems with infinitely many arms. The Annals of Statistics, 25(5):2103 – 2116, 1997.

[7] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with
non-stationary rewards. Advances in neural information processing systems, 27, 2014.

[8] Thomas Bonald and Alexandre Proutière. Two-target algorithms for infinite-armed bandits with
bernoulli rewards. In Proceedings of the 26th International Conference on Neural Information
Processing Systems - Volume 2, NIPS’13, page 2184–2192, Red Hook, NY, USA, 2013. Curran
Associates Inc.

[9] Daniel G Brown. How I wasted too long finding a concentration inequality for sums of geometric
variables. Found at https://cs. uwaterloo. ca/˜ browndg/negbin. pdf, 8(4), 2011.

[10] Alexandra Carpentier and Michal Valko. Simple regret for infinitely many armed bandits. In
Proceedings of the 32nd International Conference on International Conference on Machine
Learning - Volume 37, ICML’15, page 1133–1141. JMLR.org, 2015.

[11] Wang Chi Cheung, David Simchi-Levi, and Ruihao Zhu. Learning to optimize under non-
stationarity. In The 22nd International Conference on Artificial Intelligence and Statistics, pages
1079–1087. PMLR, 2019.

[12] Alexander Rakhlin Dylan J. Foster. Statistical reinforcement learning and decision making:
Course notes. MIT Lecture notes for course 9.S915 Fall 2022, 2022.

[13] Jung-hun Kim, Milan Vojnovic, and Se-Young Yun. Rotting infinitely many-armed bandits. In
International Conference on Machine Learning, pages 11229–11254. PMLR, 2022.

[14] Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. Advances in
Neural Information Processing Systems, 17, 2004.

[15] Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[16] Nir Levine, Koby Crammer, and Shie Mannor. Rotting bandits. Advances in neural information
processing systems, 30, 2017.

[17] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to
personalized news article recommendation. In Proceedings of the 19th international conference
on World wide web, pages 661–670, 2010.

[18] Phillippe Rigollet and Jan-Christian Hütter. High dimensional statistics. Lecture notes for
course 18S997, 813:814, 2015.

[19] Yoan Russac, Claire Vernade, and Olivier Cappé. Weighted linear bandits for non-stationary
environments. arXiv preprint arXiv:1909.09146, 2019.

[20] Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, and Michal Valko.
Rotting bandits are no harder than stochastic ones. In The 22nd International Conference on
Artificial Intelligence and Statistics, pages 2564–2572. PMLR, 2019.

[21] Julien Seznec, Pierre Menard, Alessandro Lazaric, and Michal Valko. A single algorithm for
both restless and rested rotting bandits. In Silvia Chiappa and Roberto Calandra, editors, Pro-
ceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics,
volume 108 of Proceedings of Machine Learning Research, pages 3784–3794. PMLR, 26–28
Aug 2020.

[22] Alex Tsun. Probability & statistics with applications to computing, 2020. URL https:
//www.alextsun.com/files/Prob_Stat_for_CS_Book.pdf.

11


---Page Break---
[23] Sofía S Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal
design of clinical trials: benefits and challenges. Statistical science: a review journal of the
Institute of Mathematical Statistics, 30(2):199, 2015.

[24] Yizao Wang, Jean-Yves Audibert, and Rémi Munos. Algorithms for infinitely many-armed
bandits. Advances in Neural Information Processing Systems, 21, 2008.

[25] Chen-Yu Wei and Haipeng Luo. Non-stationary reinforcement learning without prior knowledge:
An optimal black-box approach. In Conference on learning theory, pages 4300–4354. PMLR,
2021.

12


---Page Break---
A
Appendix

A.1
Limitations & Discussion

As we summarize our results in Table 1, Algorithm 1 achieves near-optimal regret only when β ≥1.
Here, we discuss the discrepancies between lower and upper bounds when 0 < β < 1. From (1), we
can observe that as β decreases below 1, the probability to sample good arms may increase, which
appears to be beneficial with respect to regret. However, the regret upper bounds for 0 < β < 1 in
Theorems 3.1 and 3.3 remain the same as the case when β = 1 while the regret lower bounds in
Theorems 4.1 and 4.2 decrease as β decreases, resulting in a gap between the regret upper and lower
bounds. The phenomenon that the regret upper bound remains the same when β decreases has also
been observed in previous literature on infinitely many-armed bandits [5, 24, 10]. As mentioned in
Carpentier and Valko [10], although there are likely to be many good arms when β is small, it is not
possible to avoid a certain amount of regret from estimating mean rewards to distinguish arms under
sub-Gaussian reward noise. Therefore, we believe that our regret upper bounds are near-optimal
across the entire range of β, and achieving tighter regret lower bounds when β < 1 is left for future
research. Notably, the optimality proven only for β ≥1 has also been observed for stationary
infinitely many-armed bandits [5, 24].

A.2
Additional Explanations for Eq. (1)

1

1
𝛽= 1
𝛽> 1
𝛽< 1
𝑥

ℙΔ! 𝑎< 𝑥

(Uniform distribution)
(Good arm ¯)
(Good arm ­)

Figure 4: P(∆1(a) < x) = xβ for different values of β.

To discuss the effect of β on the distribution of ∆1(a) and the probability of sampling a good arm
(having small ∆1(a)), we consider the case when P(∆1(a) < x) = xβ, which is shown in Figure 4
for some values of β. It is noteworthy that the uniform distribution is a special case when β = 1.
Importantly, the larger the value of β, the smaller the probability of sampling a good arm.

A.3
Proof of Proposition 2.3

Recall ∆1(a) = 1 −µ1(a). We first show that E[µ1(a)] = Θ(1). For any randomly sampled
a ∈A, we have E[µ1(a)] ≥yP(µ1(a) ≥y) = yP(∆1(a) < 1 −y) for y ∈[0, 1]. With y = 1/2,
we have E[µ1(a)] ≥(1/2)P(∆1(a) < (1/2)) = Θ(1) from constant β > 0 and (1). Then with
E[µ1(a)] ≤1, we can conclude E[µ1(a)] = Θ(1) (Especially when P(∆(a) < x) = xβ, we have
E[∆1(a)] =
R 1
0 P(∆1(a) ≥x)dx = 1 −
R 1
0 P(∆1(a) < x)dx = 1 −
R 1
0 xβdx = 1 −
1
β+1, which
implies E[µ1(a)] = Θ(1) with constant β > 0). We then think of a policy π′ that randomly samples
a new arm and pulls it only once every round. Since E[µ1(a)] = Θ(1) for any randomly sampled a,
we have E[Rπ′(T)] = Θ(T).

Next we show that the policy π′ is optimal for the worst case of PT −1
t=1 ρt > T. We think of any
policy π′′ except π′. For any policy π′′, there always exists an arm a such that the policy must
pull arm a at least twice. Let t′ and t′′ be the rounds when the policy pulls arm a. If we consider
ρt′ > 0 and ρt = 0 for t ∈[T −1]/{t′} such that ρt′ = PT −1
t=1 ρt then such policy has Ω(PT −1
t=1 ρt)
regret bound. Since PT −1
t=1 ρt > T, for any algorithm π′′ except π′, there always exist a rotting rate
adversary such that E[Rπ′′(T)] = Ω(PT −1
t=1 E[ρt]) = Ω(T). Therefore we can conclude that π′ is
the optimal algorithm for achieving the optimal regret of Θ(T).

13


---Page Break---
A.4
Proof of Theorem 3.1: Regret Upper Bound of Algorithm 1 for Slow Rotting with VT

Let ∆t(a) = 1−µt(a). Using a threshold parameter δ, we classify an arm a as good if ∆1(a) ≤δ/2,
near-good if δ/2 < ∆1(a) ≤2δ, and otherwise, we classify a as a bad arm. In A, let ¯a1, ¯a2, . . . , be
a sequence of arms, which have i.i.d. mean rewards with uniform distribution on [0, 1]. Without loss
of generality, we assume that the policy samples arms, which are pulled at least once, according to
the sequence of ¯a1, ¯a2, . . . , . Let AT be the set of sampled arms over the horizon of T time steps,
which satisfies |AT | ≤T. Let AG
T be a set of good or near good arms in AT . WLOG, the following
proofs proceed under the given AT , since the proofs hold for any AT .

Let µ[s1,s2](a) = Ps2
t=s1 µt(a)/n[s1,s2](a) for the time steps 0 < s1 ≤s2. We define event
E1 = {|bµ[s1,s2](a) −µ[s1,s2](a)| ≤
p

12 log(T)/n[s1,s2](a) for all 1 ≤s1 ≤s2 ≤T, a ∈AT }. By
following the proof of Lemma 35 in Dylan J. Foster [12], from Lemma A.29 we have

P

 bµ[s1,s2](a) −µ[s1,s2](a)
 ≤

s

12 log T
n[s1,s2](a)

!

≤

T
X

n=1
P

 
1
n

n
X

i=1
Xi

 ≤
p

12 log(T)/n

!

≤2

T 5 ,
(6)

where Xi = rτi −µτi(a) and τi is the i-th time that the policy pulls arm a starting from s1. We
note that even though Xi’s seem to depend on each other from τi’s, each value of Xi is independent
of each other. Then using union bound for s1, s2, and a ∈AT , we have P(Ec
1) ≤2/T 2. From
the cumulative amount of rotting VT , we note that ∆t(a) = O(VT + 1) for any a and t, which
implies E[Rπ(T)|Ec
1] = o(T 2) from VT ≤T. For the case where E1 does not hold, the regret is
E[Rπ(T)|Ec
1]P(Ec
1) = O(1), which is negligible compared to the regret when E1 holds, which we
show later. Therefore, for the rest of the proof, we assume that E1 holds.

For regret analysis, we divide Rπ(T) into two parts, RG(T) and RB(T) corresponding to regret of
good or near-good arms, and bad arms over time T, respectively, such that Rπ(T) = RG(T)+RB(T).
We first provide a bound of RG(T) in the following lemma.
Lemma A.1. Under E1 and policy π, we have

E[RG(T)] = ˜O

Tδ + T 2/3V 1/3
T

.

Proof. Here we consider arms a ∈AG
T .
Let V[n,m](a) = Pm
l=n ρl(a) and ρ[n,m](a) =
Pm
l=n ρl(a)/n[n,m](a) for time steps n ≤m. For ease of presentation, for time steps r > q,
we define V[r,q](a) = n[r,q](a) = ρ[r,q](a) = Pq
t=r x(t) = 0 for x(t) ∈R and 1/0 = ∞. Then, for
any s such that n ≤s ≤m, under E1 we have

bµ[s,m](a)
≤
¯µ[s,m](a) +
q

12 log(T)/n[s,m](a)

≤
µm(a) +

m−1
X

l=s
ρl1(al = a) +
q

12 log(T)/n[s,m](a)

=
µn(a) −

m−1
X

l=n
ρl1(al = a) +

m−1
X

l=s
ρl1(al = a) +
q

12 log(T)/n[s,m](a)

≤
µn(a) −V[n,m−1](a) + ρ[s,m−1](a)n[s,m](a) +
q

12 log(T)/n[s,m](a).

Therefore, from µn(a) ≤1 we obtain

bµ[s,m](a) +
q

12 log(T)/n[s,m](a)

≤1 −V[n,m−1](a) + ρ[s,m−1](a)n[s,m](a) + 2
q

12 log(T)/n[s,m](a).
(7)

14


---Page Break---
Let t1(a) be the initial time when the arm a is sampled and pulled and t2(a) be the final time when
the policy pulls the arm. For simplicity, we use t1 and t2 instead of t1(a) and t2(a), respectively,
when there is no confusion. We define A0 as a set of arms a ∈AG
T such that t2(a) = t1(a)
and define A1 as a set of arms a ∈AG
T such that t2(a) = t1(a) + 1. We also define a set of
arms A
G
T = {a ∈AG
T /{A0 ∪A1} : n[t1,t2−1](a) > ⌈(log T)1/3/ρ[t1,t2−2](a)2/3⌉}. Let w(a) =
⌈(log T)1/3/ρ[t1,t2−2](a)2/3⌉. For simplicity, we use w for w(a) when there is no confusion. Then
with the fact that µt(a) = µt1(a) −Pt−1
t=t1(a) ρt(a) = µt1(a) −V[t1,t−1](a) for t1(a) ≤t ≤t2(a),
we have

E[RG(T)] = E



X

a∈AG
T

t2(a)
X

t=t1(a)
(1 −µt(a))





= E



X

a∈AG
T



∆1(a)n[t1,t2](a) +

t2(a)
X

t=t1(a)+1
V[t1,t−1](a)









≤E



2Tδ +
X

a∈A1
ρt1(a) +
X

a∈AG
T /{A
G
T ∪A0∪A1}

t2(a)
X

t=t1(a)+1
V[t1,t−1](a)

+
X

a∈A
G
T





t1(a)+w(a)
X

t=t1(a)+1
V[t1,t−1](a) +

t2(a)
X

t=t1(a)+w(a)+1
V[t1,t−1](a)







,

(8)

where the first inequality comes from ∆1(a) ≤2δ for any a ∈AG
T . For the second term in the right
hand side of the last inequality (8),

X

a∈A1
ρt1(a) ≤VT .
(9)

For the third term in (8), from the fact that n[t1+1,t2](a) = n[t1,t2−1](a) < w(a) for any a ∈AG
T /A
G
T
from the definition of A
G
T , we have

X

a∈AG
T /{A
G
T ∪A0∪A1}

t2(a)
X

t=t1(a)+1
V[t1,t−1](a)

≤
X

a∈AG
T /{A
G
T ∪A0∪A1}
n[t1+1,t2](a)V[t1,t2−2](a) + ρt2(a)−1

= O




VT +
X

a∈AG
T /{A
G
T ∪A0∪A1}
w(a)V[t1,t2−2](a)






= ˜O




VT +
X

a∈AG
T /{A
G
T ∪A0∪A1}
n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3




.

(10)

15


---Page Break---
Now, we focus on the fourth term in (8). From t1(a) + w(a) + 1 ≤t2(a) for a ∈A
G
T from the
definition of A
G
T and (10), we first have

X

a∈A
G
T

t1(a)+w(a)
X

t=t1(a)+1
V[t1,t−1](a) =
X

a∈A
G
T

t1(a)+w(a)
X

t=t1(a)+1

t−1
X

s=t1
ρs

≤
X

a∈A
G
T

t1(a)+w(a)
X

t=t1(a)+1

t2(a)−2
X

s=t1(a)
ρs

≤
X

a∈A
G
T

w(a)V[t1,t2−2](a)

= ˜O





X

a∈A
G
T

n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3




.
(11)

Now we focus on P

a∈A
G
T
Pt2(a)
t=t1(a)+w(a)+1 V[t1,t−1](a) in (8). From the definition of t2 and the
threshold condition in the algorithm with (7), for any t1 ≤t ≤t2 and any t1 ≤s ≤t −1 s.t.
s = t −2l−1 for l ∈Z+, we have

1 −V[t1,t−2](a) + n[s,t−1](a)ρ[s,t−2](a) + 2
q

12 log(T)/n[s,t−1](a) ≥1 −δ.
(12)

For t ≥t1+w(a)+1, there always exists t1 ≤s(t) ≤t−1 such that w(a)/2 ≤n[s(t),t−1](a) ≤w(a)
and s(t) = t −2l−1 for l ∈Z+. Then from (12) with s = s(t), we have

V[t1,t−2](a) = ˜O

δ + ρ[s(t),t−2](a)/ρ[t1,t2−2](a)2/3 + ρ[t1,t2−2](a)1/3
.
(13)

Using the facts that n[s(t),t−2](a) ≥n[s(t),t−1](a)/2 ≥w(a)/4 and t −s(t) ≤w(a) from
n[s(t),t−1](a) ≤w(a), we can obtain that

t2(a)
X

t=t1(a)+1+w(a)

ρ[s(t),t−2](a) ≤

t2(a)
X

t=t1(a)+1+w(a)

Pt−2
k=t−w(a) ρk
n[s(t),t−2](a)

≤

t2(a)−2
X

t=t1(a)+1

w(a)ρt
n[s(t),t−2](a)

≤4

t2(a)−2
X

t=t1(a)
ρt,
(14)

where the second inequality is obtained from the fact that the number of times that ρt is duplicated for
each t ∈[t1(a) + 1, t2(a) −2] in the expression Pt2(a)
t=t1(a)+1+w(a)
Pt−2
k=t−w(a) ρk is at most w(a).
Then with (13) and (14), using the fact that

t2(a)
X

t1(a)+1+w(a)

ρ[t1,t2−2](a)1/3 ≤n[t1,t2−2](a)ρ[t1,t2−2](a)1/3 = O(n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3),

16


---Page Break---
we have

X

a∈A
G
T

t2(a)
X

t=t1(a)+1+w(a)
V[t1,t−1](a)

≤
X

a∈A
G
T

t2(a)
X

t=t1(a)+1+w(a)
V[t1,t−2](a) + ρt2(a)−1

= ˜O




δT + VT +
X

a∈A
G
T

t2(a)
X

t=t1(a)+1+w(a)

ρ[s(t),t−2](a)/ρ[t1,t2−2](a)2/3 +
X

a∈A
G
T

t2(a)
X

t=t1(a)+1+w(a)

ρ[t1,t2−2](a)1/3






= ˜O




δT + VT +
X

a∈A
G
T

t2(a)
X

t=t1(a)+1+w(a)

ρ[s(t),t−2](a)/ρ[t1,t2−2](a)2/3 +
X

a∈A
G
T

n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3






= ˜O




δT + VT +
X

a∈A
G
T

t2(a)−2
X

t=t1(a)
ρt/ρ[t1,t2−2](a)2/3 +
X

a∈A
G
T

n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3






= ˜O




δT + VT +
X

a∈A
G
T

n[t1,t2−2](a)2/3V[t1,t2−2](a)1/3




.
(15)

Then putting the results from (8),(10),(11), and (15) altogether, we have

E[RG(T)]

≤E



X

a∈AG
T



∆1(a)n[t1,t2](a) +

t2(a)
X

t=t1(a)+1
V[t1,t−1](a)









= ˜O



Tδ + VT + E




X

a∈AG
T /{A0∪A1}
V[t1,t2−2](a)1/3n[t1,t2−2](a)2/3









= ˜O

Tδ + V 1/3
T
T 2/3
,
(16)

where the last equality comes from Hölder’s inequality and VT ≤T. This concludes the proof.

Now, we provide a bound for RB(T). We note that the initially bad arms can be defined only when
2δ < 1. Otherwise when 2δ ≥1, we have R(T) = RG(T), which completes the proof. Therefore,
for the regret from bad arms, we consider the case of 2δ < 1. We adopt the episodic approach in
Kim et al. [13] for the remaining regret analysis. The episodic approach is reformulated using the
cumulative amount of rotting instead of the maximum rotting rate. In the following, we define some
notation.

Given a policy sampling arms in the sequence order, let mG be the number of samples of distinct good
arms and mB
i be the number of consecutive samples of distinct bad arms between the i −1-st and
i-th sample of a good arm among mG good arms. We refer to the period starting from sampling the
i −1-st good arm before sampling the i-th good arm as the i-th episode. Observe that mB
1 , . . . , mB
mG
are i.i.d. random variables with geometric distribution with parameter 2δ, given a fixed value of mG.
Therefore, for non-negative integer k we have P(mB
i = k) = (1−2δ)k2δ, for i = 1, . . . , mG. Define
˜mT to be the number of episodes from the policy π over the horizon T, ˜mG
T to be the total number of
samples of a good arm by the policy π over the horizon T such that ˜mG
T = ˜mT or ˜mG
T = ˜m −1, and
˜mB
i,T to be the number of samples of a bad arm in the i-th episode by the policy π over the horizon T.

17


---Page Break---
Under a policy π, let RB
i,j be the regret (summation of mean reward gaps) contributed by pulling

the j-th bad arm in the i-th episode. Then let RB
mG = PmG

i=1
P
j∈[mB
i ] RB
i,j, which is the regret from
initially bad arms over the period of mG episodes.

Let a(i) be a good arm in the i-th episode and a(i, j) be a j-th bad arm in the i-th episode. We define
VT (a) = PT
t=1 ρt1(at = a). Then excluding the last episode ˜mT over T, we provide lower bounds
of the total rotting variation over T for a(i), denoted by VT (a(i)), in the following lemma.

Lemma A.2. Under E1, given ˜mT , for any i ∈[ ˜mG
T ]/{ ˜mT } we have

VT (a(i)) ≥δ/2.

Proof. Suppose that VT (a(i)) < δ/2, then we have

min
t1(a(i))≤s≤t2(a(i))

n
bµ[s,t2(a(i))](a(i)) +
q

12 log(T)/n[s,t2(a(i))](a(i))
o

≥
min
t1(a(i))≤s≤t2(a(i)){µ[s,t2(a(i))](a(i))}

≥µt2(a(i))(a(i))

≥µ1(a(i)) −VT (a(i))
> 1 −δ,

where the first inequality is obtained from E1, and the last inequality is from VT (a(i)) < δ/2 and
µ1(a(i)) ≥1 −δ/2. Therefore, from the threshold condition, policy π must pull arm a(i) until its
total rotting amount is greater than (or equal to) δ/2, which implies VT (a(i)) ≥δ/2.

In the following, we consider two different cases with respect to VT ; large and small VT .

Case 1: We consider VT > max{1/
√

T, 1/T 1/(β+1)} in the following.

In this case, we have δ = δV (β) = max{(VT /T)1/(β+2), (VT /T)1/3}. Here, we define the policy π
after time T such that it pulls a good arm until its total rotting variation is equal to or greater than
δ/2 and does not pull a sampled bad arm. We note that defining how π works after T is only for the
proof to get a regret bound over time horizon T. For the last arm ˜a over the horizon T, it pulls the
arm until its total variation becomes max{δ/2, VT (˜a)} if ˜a is a good arm. For i ∈[mG], j ∈[mB
i ]
let V G
i and V B
i,j be the total rotting variation of pulling the good arm in i-th episode and j-th bad arm
in i-th episode from the policy, respectively. Here we define V G
i ’s and V B
i,j’s as follows:

If ˜a is a good arm,

V G
i =
VT (a(i))
for i ∈[ ˜mG
T −1]
max{δ/2, VT (a(i))}
for i ∈[mG]/[ ˜mG
T −1] , V B
i,j =
VT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ].

Otherwise,

V G
i =
VT (a(i))
for i ∈[ ˜mG
T ]
δ/2
for i ∈[mG]/[ ˜mG
T ] , V B
i,j =
VT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T −1], j ∈[mB
i ]/[ ˜mB
i,T ].

For i ∈[mG], j ∈[mB
i ] let nB
i,j be the number of pulling the j-th bad arm in i-th episode from
the policy. We define nT (a) be the total amount of pulling arm a over T. Here we define nB
i,j’s as
follows:

nB
i,j =
nT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ].

Then we provide mG such that RB(T) ≤RB
mG in the following lemma.

Lemma A.3. Under E1, when mG = ⌈2VT /δ⌉we have

RB(T) ≤RB
mG.

18


---Page Break---
Proof. From Lemma A.2, we have
X

i∈[mG]
V G
i ≥mG δ

2 ≥VT ,

which implies that RB(T) ≤RB
mG.

From the result of Lemma A.3, we set mG = ⌈2VT /δ⌉. We analyze RB
mG for obtaining a bound for
RB(T) in the following.
Lemma A.4. Under E1 and policy π, we have

E[RB
mG] = ˜O

max{T (β+1)/(β+2)V 1/(β+2)
T
, T 2/3V 1/3
T
}

.

Proof. Let a(i, j) be a sampled arm for j-th bad arm in the i-th episode and ˜mT be the number of
episodes from the policy π over the horizon T. Suppose that the algorithm samples arm a(i, j) at time
t1(a(i, j)). Then the algorithm stops pulling arm a(i, j) at time t2(a(i, j)) + 1 if bµ[s,t2(a(i,j))](a) +
p

12 log(T)/n[s,t2(a(i,j))](a) < 1 −δ for some s such that t1(a(i, j)) ≤s ≤t2(a(i, j)) and
s = t2(a(i, j)) + 1 −2l−1 for l ∈Z+. For simplicity, we use t1 and t2 instead of t1(a(i, j)) and
t2(a(i, j)) when there is no confusion. We first consider the case where the algorithm stops pulling
arm a(i, j) because the threshold condition is satisfied. For the regret analysis, we consider that for
t > t2, arm a is virtually pulled. We note that under E1, we have

bµ[s,t2](a(i, j)) +
q

12 log(T)/n[s,t2](a(i, j)) ≤µ[s,t2](a(i, j)) + 2
q

12 log(T)/n[s,t2](a(i, j))

≤µ1(a(i, j)) + 2
q

12 log(T)/n[s,t2](a(i, j)).

Then we assume that ˜t2(≥t2) is the smallest time that there exists t1 ≤s ≤˜t2 with s = ˜t2 +1−2l−1
for l ∈Z+ such that the following threshold condition is met:

µ1(a(i, j)) + 2
q

12 log(T)/n[s,˜t2](a(i, j)) < 1 −δ.
(17)

From the definition of ˜t2, we observe that for given ˜t2, the time step s = s′ which satisfying
(17) equals to t1 (i.e. s′ = t1). Then, we can observe that n[s′,˜t2](a(i, j)) = n[t1,˜t2](a(i, j)) =
⌈C2 log(T)/(∆t1(a(i, j)) −δ)2⌉for some constant C2 > 0, which satisfies (17). Then from
n[t1,t2](a(i, j)) ≤n[t1,˜t2](a(i, j)), for all i ∈[ ˜mT ], j ∈[ ˜mB
i,T ] we have nB
i,j = ˜O(1/(∆1(a(i, j)) −
δ)2). Then with the facts that nB
i,j = 0 for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ]/[ ˜mB
i,T ], we have, for any
i ∈[mG] and j ∈[mB
i ],
nB
i,j = ˜O(1/(∆t1(a(i, j)) −δ)2).

For 2δ < x ≤1, let b(x) = P(∆1(a) = x|a is a bad arm). Then we have

b(x) = P(∆1(a) = x|∆1(a) > 2δ)
= P(∆1(a) = x)/P(∆1(a) > 2δ)

= P(∆1(a) = x)/(1 −C(2δ)β).

We note that 2δ < ∆t1(a(i, j)) = ∆1(a(i, j)) ≤1. Since nB
i,j = ˜O(1/(∆t1(a(i, j)) −δ)2) =
˜O(1/δ2), we have

E[RB
i,j] = E





t2(a(i,j))
X

t=t1(a(i,j))
∆t1(a(i, j)) +

t2(a(i,j))−1
X

t=t1(a(i,j))

t
X

s=t1(a(i,j))
ρs





≤E[∆1(a(i, j))nB
i,j + V B
i,jnB
i,j]

≤E[∆1(a(i, j))nB
i,j + V B
i,j(1/δ2)]

= ˜O
Z 1

2δ

1
(x −δ)2 xb(x)dx + E[V B
i,j(1/δ2)]

.
(18)

19


---Page Break---
Recall that we consider 2δ < 1 for regret from bad arms. We adopt some techniques introduced in
Appendix D of Bayati et al. [5] to deal with the generalized mean reward distribution with β. Let
K = (1 −2δ)/δ, aj =
2
jδ, and pj =
R (j+1)δ
jδ
b(t + δ)dt. Then for obtaining a bound of the last
equality in (18) we have

Z 1

2δ


1
(x −δ)2 x

b(x)dx =
Z 1−δ

δ

1

t + δ

t2


b(t + δ)dt

=

K
X

j=1

Z (j+1)δ

jδ

1

t + δ

t2


b(t + δ)dt

≤

K
X

j=1

2
jδ

Z (j+1)δ

jδ
b(t + δ)dt

=

K
X

j=1
ajpj.
(19)

We note that Pj
i=1 pi ≤C0(jδ)β for all j ∈[K] for some constant C0 > 0. Then for getting a
bound of the last equality in (19), we have

K
X

j=1
ajpj =

K−1
X

j=1
(aj −aj+1)

 j
X

i=1
pi

!

+ aK

K
X

i=1
pi

≤

K−1
X

j=1
(aj −aj+1)C0(jδ)β + aKC0(Kδ)β

= C0δβa1 +

K
X

j=2
C0(jβ −(j −1)β)δβaj

= O




1

δ


δβ +

K
X

j=2

 1

jδ

  
(jδ)β −((j −1)δ)β




= O



δβ−1 +

K
X

j=2

1

j δβ−1
  
jβ −(j −1)β


.
(20)

Now we analyze the term in the last equality in (20) according to the criteria for β. For β = 1, we
can obtain

O



δβ−1 +

K
X

j=2

1

j δβ−1
  
jβ −(j −1)β


= ˜O(1).
(21)

For β > 1, we have jβ −(j −1)β ≤βjβ−1 using the mean value theorem. Therefore, we obtain the
following.

O



δβ−1 +

K
X

j=2

1

j δβ−1
  
jβ −(j −1)β


= O




K
X

j=1

1

j δβ−1

jβ−1





= O




K
X

j=2
δβ−1jβ−2





= O

δβ−1
1
β −1
 
(K + 1)β−1 −1


= O(1).
(22)

20


---Page Break---
For β < 1, when j > 1 we have jβ −(j −1)β ≤β(j −1)β−1 using the mean value theorem.
Therefore, we obtain

O



δβ−1 +

K
X

j=1

1

j δβ−1
  
jβ −(j −1)β


= O



δβ−1 +

K
X

j=2

1

j δβ−1

(j −1)β−1





= O



δβ−1 +

K
X

j=2
δβ−1(j −1)β−2





= O

δβ−1 + δβ−1
1
β −1
 
(K + 1)β−1 −1


= O

δβ−1 + δβ−1 1 −((1 −δ)/δ)β−1

1 −β


= O(δβ−1).
(23)

From (19),(20),(21),(22), and (23), we have

Z 1

2δ


1
(x −δ)2 x

b(x)dx = ˜O(max{1, δβ−1}).

Then for any i ∈[mG], j ∈[mB
i ], we have

E[RB
i,j] ≤E

∆(a(i, j))nB
i,j + V B
i,jnB
i,j


= ˜O
 
max{1, δβ−1} + E[V B
i,j]/δ2
.
(24)

Recall that RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j. With δ = max{(VT /T)1/(β+2), (VT /T)1/3} and mG =
⌈2VT /δ⌉, from the fact that mB
i ’s are i.i.d. random variables with geometric distribution with
E[mB
i ] = 1/(2δ)β −1, we have

E[RB
mG] = O



E




mG
X

i=1

X

j∈[mB
i ]
RB
i,j









= ˜O

(VT /δ) 1

δβ max{1, δβ−1} + VT /δ2


= ˜O

max{T (β+1)/(β+2)V 1/(β+2)
T
, T 2/3V 1/3
T
}

.
(25)

From
Rπ(T)
=
RG(T) + RB(T)
and
Lemmas
A.1,
A.3,
A.4,
with
δ
=
max{(VT /T)1/(β+2), (VT /T)1/3} we have

E[Rπ(T)] = ˜O

max{T (β+1)/(β+2)V 1/(β+2)
T
, T 2/3V 1/3
T
}

.
(26)

Case 2: Now we consider VT ≤max{1/
√

T, 1/T 1/(β+1)} in the following. In this case, we have
δ = max{1/T
1
β+1 , 1/
√

T}. For getting RB
mG, here we define the policy π after time T such that it
pulls VT amount of rotting variation for a good arm and 0 for a bad arm. We note that defining how
π works after T is only for the proof to get a regret bound over time horizon T. For the last arm
˜a over the horizon T, it pulls the arm up to VT amount of rotting variation if ˜a is a good arm. For
i ∈[mG], j ∈[mB
i ] let V G
i and V B
i,j be the amount of rotting variation from pulling the good arm in
i-th episode and j-th bad arm in i-th episode from the policy, respectively. Here we define V G
i ’s and
V B
i,j’s as follows:

If ˜a is a good arm,

V G
i =
VT (a(i))
for i ∈[ ˜mG
T −1]
VT
for i ∈[mG]/[ ˜mG
T −1] , V B
i,j =
VT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ].

21


---Page Break---
Otherwise,

V G
i =
VT (a(i))
for i ∈[ ˜mG
T ]
VT
for i ∈[mG]/[ ˜mG
T ] , V B
i,j =
VT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T −1], j ∈[mB
i ]/[ ˜mB
i,T ].

For i ∈[mG], j ∈[mB
i ] let nB
i,j be the number of pulling the j-th bad arm in i-th episode from
the policy. We define nT (a) be the total amount of pulling arm a over T. Here we define nB
i,j’s as
follows:

nB
i,j =
nT (a(i, j))
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ].

Then we provide mG such that RB(T) ≤RB
mG in the following lemma.

Lemma A.5. Under E1, when mG = 3, we have
RB(T) ≤RB
mG.

Proof. From Lemma A.2, under E1 we can find that V G
i
≥min{δ/2, VT } for i ∈[mG]. Then if
mG = 3, then with δ = max{1/T 1/(β+1), 1/
√

T} and VT ≤max{1/T 1/(β+1), 1/
√

T}, we have
X

i∈[mG]
V G
i ≥C3 min{δ/2, VT } > VT ,

which implies RB(T) ≤RB
mG.

We analyze RB
mG for obtaining a bound for RB(T) in the following.
Lemma A.6. Under E1 and policy π, we have

E[RB
mG] = ˜O

max{T β/(β+1),
√

T}

.

Proof. From (24), for any i ∈[mG], j ∈[mB
i ], we have

E[RB
i,j] ≤E

∆(a(i, j))nB
i,j + V B
i,jnB
i,j


= ˜O
 
max{1, δβ−1} + E[V B
i,j]/δ2
.

Recall that RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j. With δ = max{(1/T)1/(β+1), 1/T 1/2} and mG = C3,
from the fact that mB
i ’s are i.i.d. random variables with geometric distribution with E[mB
i ] =
1/(2δ)β −1, we have

E[RB
mG] = O



E




mG
X

i=1

X

j∈[mB
i ]
RB
i,j









= ˜O
 1

δβ max{1, δβ−1} + VT /δ2


= ˜O

max{T β/(β+1),
√

T}

.

From Lemma A.1, with δ = max{1/T
1
β+1 , 1/
√

T} we have

E[RG(T)] = ˜O

max{T β/(β+1),
√

T}

.

From Rπ(T) = RG(T) + RB(T) and Lemmas A.1, A.5, A.6 with δ = max{1/T
1
β+1 , 1/
√

T} we
have

E[Rπ(T)] = ˜O

max{T β/(β+1),
√

T}

.
(27)

22


---Page Break---
Conclusion:
Overall, from (26) and (27), we have

E[Rπ(T)] = ˜O

max{V 1/(β+2)
T
T (β+1)/(β+2), V 1/3
T
T 2/3, T β/(β+1),
√

T}

.

A.5
Proof of Theorem 3.3: Regret Upper Bound of Algorithm 1 for Abrupt Rotting (ST )

Using the threshold parameter δ in the algorithm, we define an arm a as a good arm if ∆t(a) ≤δ/2,
a near-good arm if δ/2 < ∆t(a) ≤2δ, and otherwise, a is a bad arm at time t. For analysis, we
consider abrupt change as sampling a new arm. In other words, if a sudden change occurs to an
arm a by pulling the arm a, then the arm is considered to be two different arms; before and after the
change. The type of abruptly rotted arms (good, near-good, or bad) after the change is determined
by the current value of rotted mean reward. Without loss of generality, we assume that the policy
samples arms, which are pulled at least once, in the sequence of ¯a1, ¯a2, . . . , . Let AT be the set
of sampled arms, which are pulled at least once, over the horizon of T time steps, which satisfies
|AT | ≤T. We also define AS as a set of arms that have been rotted and pulled at least once, which
satisfies |AS| ≤ST . To better understand the definitions, we provide an example. If an arm a suffers
abrupt rotting at first, then the arm a is considered to be a different arm a′ after the rotting. If the arm
a′ again suffers abrupt rotting, then it is considered to be a′′ after the rotting. If arms a, a′, a′′ are
pulled at least once, then {a, a′, a′′} ∈AT and {a′, a′′} ∈AS but a /∈AS. If arm a′′ is not pulled
at least once but a and a′ are pulled at least once, then {a, a′} ∈AT and a′ ∈AS but a′′ /∈AS.

WLOG, the following proofs proceed under the given AT , since the proofs hold for any AT .
Let µ[t1,t2](a) = Pt2
t=t1 µt(a)1(at = a)/n[t1,t2](a). We define the event E1 = {|bµ[s1,s2](a) −
µ[s1,s2](a)| ≤
p

12 log(T)/n[s1,s2](a) for all 1 ≤s1 ≤s2 ≤T, a ∈AT }. By following the proof
of Lemma 35 in Dylan J. Foster [12], from Lemma A.29 we have

P

 bµ[s1,s2](a) −µ[s1,s2](a)
 ≤

s

12 log T
n[s1,s2](a)

!

≤

T
X

n=1
P

 
1
n

n
X

i=1
Xi

 ≤
p

12 log(T)/n

!

≤2

T 5 ,
(28)

where Xi = rτi −µτi(a) and τi is the i-th time that the policy pulls arm a starting from s1. We note
that even though Xi’s seem to depend on each other from τi’s, each value of Xi is independent of
each other. Then using union bound for s1, s2, and a ∈AT , we have

P(Ec
1) ≤2/T 2.

Let t(s) be the time when s-th abrupt rotting occurs with ρt(s) for s ∈[ST ]. Then we have ∆t(a) =
O(1 + PST
s=1 ρt(s)) = O(1 + VT ) for any a and t, which implies E[Rπ(T)|Ec
1] = O(T + TVT ).
For the case that E1 does not hold, the regret is E[Rπ(T)|Ec
1]P(Ec
1) = O((1 + VT )/T), which is
negligible comparing with the regret when E1 holds true which we show later. Therefore, in the rest
of the proof we assume that E1 holds true.

Recall that Rπ(T) = PT
t=1(1 −µt(at)). For regret analysis, we divide Rπ(T) into two parts,
RG(T) and RB(T) corresponding to regret of good or near-good arms, and bad arms over time
T, respectively, such that Rπ(T) = RG(T) + RB(T). Recall that we consider abrupt change as
sampling a new arm in this analysis. Then, from ∆t(a) ≤2δ for any good or near-good arms a at
time t, we can easily obtain that

E[RG(T)] = O(δT) = O(max{S1/(β+1)
T
T β/(β+1),
p

ST T}).
(29)

Now we analyze RB(T). We divide regret RB(T) into two regret from bad arms in AT /AS,
denoted by RB,1(T), and regret from bad arms in AS, denoted by RB,2(T) such that RB(T) =
RB,1(T) + RB,2(T). We denote bad arms in AS by AB
S. We first analyze RB,1(T) in the following.
For regret analysis, we adopt the episodic approach suggested in Kim et al. [13]. The main difference
lies in analyzing our adaptive window UCB and a more generalized mean-reward distribution with

23


---Page Break---
β. In the following, we introduce some notation. Here we only consider arms in AT /AS so that
the following notation is defined without considering (rotted) arms in AS. We note that from the
definition of AT , arms a before having undergone rotting are contained in AT /AS. Here we consider
the case of 2δS(β) < 1 since otherwise when 2δS(β) ≥1, bad arms are not defined in AT /AS.

Given a policy sampling arms in the sequence order, let mG be the number of samples of distinct good
arms and mB
i be the number of consecutive samples of distinct bad arms between the i −1-st and
i-th sample of a good arm among mG good arms. We refer to the period starting from sampling the
i −1-st good arm before sampling the i-th good arm as the i-th episode. Observe that mB
1 , . . . , mB
mG
are i.i.d. random variables with geometric distribution with parameter 2δ, given a fixed value of mG.
Therefore, for non-negative integer k we have P(mB
i = k) = (1 −2δ)k2δ, for i = 1, . . . , mG.

Define ˜mG
T to be the total number of samples of a good arm by the policy π over the horizon T and
˜mB
i,T to be the number of samples of a bad arm in the i-th episode by the policy π over the horizon T.
For i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ], let ˜nG
i be the number of pulls of the good arm in the i-th episode and ˜nB
i,j
be the number of pulls of the j-th bad arm in the i-th episode by the policy π over the horizon T. Let
˜a be the last sampled arm over time horizon T by π.

With a slight abuse of notation, we use π for a modified strategy after T. Under a policy π, let RB
i,j
be the regret (summation of mean reward gaps) contributed by pulling the j-th bad arm in the i-th
episode. Then let RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j, which is the regret from initially bad arms over the
period of mG episodes. For getting RB
mG, here we define the policy π after T such that it pulls T
amounts for a good arm and zero for a bad arm. After T we can assume that there are no abrupt
changes. For the last arm ˜a over the horizon T, it pulls the arm up to T amounts if ˜a is a good arm
and ˜nG
˜mG
T < T. For i ∈[mG], j ∈[mB
i ] let nG
i and nB
i,j be the number of pulling the good arm in i-th

episode and j-th bad arm in i-th episode under π, respectively. Here we define nG
i ’s and nB
i,j’s as
follows:

If ˜a is a good arm,

nG
i =






˜nG
i
for i ∈[ ˜mG
T −1]
T
for i = ˜mG
T
0
for i ∈[mG]/[ ˜mG
T ]
, nB
i,j =
˜nB
i,j
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T ], j ∈[mB
i ]/[ ˜mB
i,T ].

Otherwise,

nG
i =






˜nG
i
for i ∈[ ˜mG
T ]
T
for i = ˜mG
T + 1
0
for i ∈[mG]/[ ˜mG
T + 1]
, nB
i,j =
˜nB
i,j
for i ∈[ ˜mG
T ], j ∈[ ˜mB
i,T ]
0
for i ∈[mG]/[ ˜mG
T −1], j ∈[mB
i ]/[ ˜mB
i,T ].

Using the above notation and newly defined π after T, we show that if mG = ST + 1, then
RB(T) ≤RB
mG in the following.

Lemma A.7. Under E1, when mG = ST we have

RB,1(T) ≤RB
mG.

Proof. There are ST −1 number of abrupt changes over T. We consider two cases; there are ST
abrupt changes before sampling ST -th good arm or there are not. For the former case, if π samples
the ST -th good arm and there are ST −1 number of abrupt changes before sampling the good arm,
then it continues to pull the good arm until T. This is because when the algorithm samples a good
arm a at time t′, from E1 and the stationary period, we have

bµ[t′,t](a) +
q

12 log(T)/n[t′,t](a) ≥µt′(a) ≥1 −δ.

This implies that from the threshold condition, the algorithm does not stop pulling the good arm
a. After T, from the definition of π for the case when ˜a is a good arm, nG
˜mG
T = T. Therefore, the
algorithm pulls the good arm for T rounds.

Now we consider the latter case, such that π samples the ST -th good arm before the ST −1-st abrupt
change over T. Before sampling the ST -th good arm, there must exist two consecutive good arms

24


---Page Break---
such that there is no abrupt change between the two sampled good arms. This is a contraction because
π must pull the first good arm among the two up to T under E1 and ST −1-st abrupt change must
occur after T.

Therefore, it is enough to consider the former case. When mG = ST , we have
X

i∈[mG]
nG
i ≥T,

which implies RB,1(T) ≤RB
mG.

From the above lemma, we set mG = ST . We analyze RB
mG to get a bound for RB,1(T) in the
following lemma.

Lemma A.8. Under E1 and policy π, we have

E

RB
mG

= ˜O

max{S1/(β+1)
T
T β/(β+1),
p

ST T}

.

Proof. Recall that we consider arms in AT /AS. Let a(i, j) be a sampled arm for j-th bad arm in the
i-th episode and ˜mT be the number of episodes from the policy π over the horizon T. Suppose that
the algorithm samples arm a(i, j) at time t1(a(i, j)). Then the algorithm stops pulling arm a(i, j) at
time t2(a(i, j)) + 1 if bµ[s,t2(a(i,j))](a) +
p

12 log(T)/n[s,t2(a(i,j))](a) < 1 −δ for some s such that
t1(a(i, j)) ≤s ≤t2(a(i, j)) and s = t2(a(i, j)) + 1 −2l−1 for l ∈Z+. For simplicity, we use t1
and t2 instead of t1(a(i, j)) and t2(a(i, j)) when there is no confusion. For the regret analysis, we
consider that for t > t2, arm a is virtually pulled. With E1, we assume that ˜t2(≥t2) is the smallest
time that there exists t1 ≤s ≤˜t2 with s = ˜t2 +1−2l−1 for l ∈Z+ such that the following condition
is met:

µt1(a(i, j)) + 2
q

12 log(T)/n[s,˜t2](a(i, j)) < 1 −δ.
(30)

From the definition of ˜t2, we observe that for given ˜t2, the time step s = s′ satisfying (30)
equals to t1 (i.e. s′ = t1). Then, we can observe that n[s′,˜t2](a(i, j)) = n[t1,˜t2](a(i, j)) =
⌈C2 log(T)/(∆t1(a(i, j)) −δ)2⌉for some constant C2 > 0, which satisfies (30). Then from
n[t1,t2](a(i, j)) ≤n[t1,˜t2](a(i, j)), for all i ∈[ ˜mT ], j ∈[ ˜mB
i,T ] we have nB
i,j = ˜O(1/(∆t1(a(i, j))−
δ)2). We note that this bound for the number of pulling an arm holds for not only the case where
the arm stops being pulled from the threshold condition but also the case where the arm stops being
pulled from meeting an abrupt change (recall that abrupt changes are considered as sampling a new
arm) or T. Then with the facts that nB
i,j = 0 for i ∈[mG]/[ ˜mT ], j ∈[mB
i ]/[ ˜mB
i,T ], we have, for any
i ∈[mG] and j ∈[mB
i ],
nB
i,j = ˜O(1/(∆t1(a(i, j)) −δ)2).

For 2δ < x ≤1, let b(x) = P(∆t1(a) = x|a is a bad arm).
Then we have P(∆t1(a) =
x|a is a bad arm) = P(∆t1(a) = x|∆t1(a) > 2δ) = P(∆t1(a) = x)/P(∆1(a) > 2δ) =
P(∆t1(a) = x)/(1 −C(2δ)β) = O(P(∆t1(a) = x)), where the last equality comes from small δ
with ST = o(T). For any i ∈[mG], j ∈[mB
i ], we have

E[RB
i,j] ≤E

∆t1(a(i,j))(a(i, j))nB
i,j


= ˜O
Z 1

2δ

1
(x −δ)2 xb(x)dx

.
(31)

From the above results in (31),(19),(20),(21),(22),(23), for β > 0 we have

E[RB
i,j] = ˜O(max{1, δβ−1}).

Recall that RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j. With δ = max{(ST /T)1/(β+1), (ST /T)1/2} and mG =
ST , from Lemma A.7 and the fact that mB
i ’s are i.i.d. random variables following geometric

25


---Page Break---
distribution with E[mB
i ] = 1/(2δ)β −1, we have

E[RB
mG] = O



E




mG
X

i=1

X

j∈[mB
i ]
RB
i,j









= ˜O

ST
1
δβ max{1, δβ−1}


= ˜O

max{S1/(β+1)
T
T β/(β+1),
p

ST T}

.

From Lemma A.8, we have E[RB,1(T)] = E[RB
mG] = ˜O

max{S1/(β+1)
T
T β/(β+1), √ST T}

.

Now we analyze RB,2(T) in the following lemma. Here, we consider arms in AB
S, which is allowed
to have negative mean rewards.

Lemma A.9. Under E1 and policy π, we have

E

RB,2(T)

= ˜O
 
max{ST /δ, ¯VT }

.

Proof. Recall that we consider arms a ∈AB
S so that ∆t1(a) > 2δ from definition. Suppose that
the arm a is sampled and pulled for the first time at time t1(a). Then the algorithm stops pulling
arm a at time t2(a) + 1 if bµ[s,t2(a)](a) +
p

12 log(T)/n[s,t2(a)](a) < 1 −δ for some s such that
s ≤t2(a) and s = t2(a) + 1 −2l−1 for l ∈Z+. For simplicity, we use t1 and t2 instead of t1(a) and
t2(a) when there is no confusion. For regret analysis, we consider that for t > t2, arm a is virtually
pulled. With E1, we assume that ˜t2(≥t2) is the smallest time that there exists t1 ≤s ≤˜t2 with
s = ˜t2 + 1 −2l−1 for l ∈Z+ such that the following condition is met:

µt1(a) + 2
q

12 log(T)/n[s,˜t2](a) < 1 −δ.
(32)

From the definition of ˜t2, we observe that for given ˜t2, the time step s, which satisfies (32), equals to
t1. Then, we can observe that n[t1,˜t2](a) = max{⌈C2 log(T)/(∆t1(a) −δ)2⌉, 1} for some constant
C2 > 0, which satisfies (32). From the above, for any a ∈AB
S satifying ∆t1(a) ≥
p

C2 log(T) + δ,
we have n[t1,˜t2](a) = 1. This implies that after pulling the arm a once, the arm is eliminated and
after that, the arm is not pulled anymore. Therefore, for any arm a′ which was rotted to a, we have
∆t1(a′)(a′) <
p

C2 log(T) + δ. This is because otherwise such that ∆t1(a′)(a′) ≥
p

C2 log(T) + δ,
the arm a′ is eliminated and a cannot be pulled which means a /∈AB
S, which is a contradiction. Then
for any arm a ∈AB
S, we have ∆t1(a) ≤
p

C2 log(T) + δ + ρt1(a)−1. Recall that we consider abrupt
rotting of an arm as sampling a new arm. Let t(s) be the time step when the s-th abrupt rotting
occurs. Then we note that ρt1(a)−1 = ρt(s) when arm a is a sampled arm from s-th abrupt rotting for
s ∈[ST ].

From n[t1,t2](a) ≤n[t1,˜t2](a), we have n[t1,t2](a) = ˜O(max{1/(∆t1(a) −δ)2, 1}). We note that
this bound for number of pulling an arm holds for not only the case where the arm stops to be pulled
from the threshold condition, but also the case where the arm stops to be pulled from meeting an
abrupt change (recall that abrupt changes are considered as sampling a new arm) or T. From the
definition of bad arms, we have ∆t1(a) ≥2δ. Then the regret from arm a, denoted by R(a), is
bounded as follows: R(a) = ∆t1(a)n[t1,t2](a) = ˜O(max{∆t1(a)/(∆t1(a) −δ)2, ∆t1(a)}). Since
x/(x −δ)2 ≤2/δ for any x ≥2δ, we have R(a) = ˜O(max{1/δ, ∆t1(a)}). Therefore, with the fact
that ∆t1(a) ≤
p

C2 log(T) + δ + ρt(s) for the corresponding s ∈[ST ] such that ρt1(a)−1 = ρt(s),

26


---Page Break---
we have

E



X

a∈AB
S

R(a)



= ˜O



max




ST /δ, E



X

a∈AB
S

∆t1(a)














= ˜O(max{ST /δ, ST +

ST
X

s=1
E[ρt(s)]})

= ˜O(max{ST /δ,

ST
X

s=1
E[ρt(s)]})

= ˜O(max{ST /δ, ¯VT }),

where the second last equality comes from ST /δ ≥ST .

Finally, from Rπ(T) = RG(T) + RB(T), (29), and Lemmas A.8, A.9, we have

E[Rπ(T)] = ˜O

max{S1/(β+1)
T
T β/(β+1),
p

ST T, ¯VT }

.

A.6
Details for the Case of Unknown Parameters

Algorithm 2 Adaptive UCB-Threshold with Adaptive Sliding Window

Given: T, H, B, A, α, κ, C
Initialize: A′ ←A, w(δ′) ←1 for δ′ ∈B
for i = 1, 2, . . . , ⌈T/H⌉do

t′ ←(i −1)H + 1
Select a new arm a ∈A′
Pull arm a and get reward r(i−1)H+1
p(δ′) ←(1 −α)
w(δ′)
P

k∈B w(k) + α 1

B for δ′ ∈B
Select δ ←δ′ with probability p(δ′) for δ′ ∈B
for t = (i −1)H + 2, . . . , i · H ∧T do

if mins∈Tt(a) WUCB(a, s, t −1, H) < 1 −δ then

A′ ←A′/{a}
Select a new arm a ∈A′
Pull arm a and get reward rt
t′ ←t
else

Pull arm a and get reward rt
end if
end for

w(δ) ←w(δ) exp

α
Bp(δ)


1
2 +

Pi·H∧T
t=(i−1)H rt
CH log(H)+4√H log T



end for

Here, we consider the case where there are constraints for both ST and VT , and parameters of VT ,
ST , and β are unknown to the agent. We note that ¯VT ≤VT from the constraint. The parameters of
β, VT , and ST are used to set the optimal threshold parameter δ in Algorithm 1. Therefore, when
the parameters are not given, the procedure to find the optimal value δ is required. We adopt the
Bandit-over-Bandit (BoB) approach in Cheung et al. [11], Kim et al. [13] by additionally considering
adaptive window. In Algorithm 2, the algorithm consists of a master and several base algorithms with
B. For the master, we use EXP3 [2] to find a nearly best base in B. Each base represents Algorithm 1
with a candidate threshold δ′ ∈B. The algorithm divides the time horizon into several blocks of
length H. At each block, the algorithm samples a base in B from the EXP3 strategy and runs the base
over the time steps of the block. Using the feedback from the block, the algorithm updates EXP3 and
samples a new base for the next block. By block time passes, the master is likely to find an optimized
δ in B. Let B = |B|. Then for Algorithm 2, we set α = min{1,
p

B log B/((e −1)⌈T/H⌉)} and
C > 0 to be a large enough constant.

27


---Page Break---
We
define
δ†
V
=
max{(VT /T)1/(β+2), (VT /T)1/3, 1/H1/(β+1), 1/
√

H}
and
δ†
S
=
max{(ST /T)1/(β+1), (ST /T)1/2, 1/H1/(β+1), 1/
√

H}.
Then the optimized threshold pa-
rameter is δ†
V S = min{δ†
S, δ†
V }. The optimized threshold parameter can be derived from the
theoretical analysis in Appendix A.7. The target of the master is to find the parameter. From the
above, we can observe that 1/
√

H ≤δ†
V S ≤1. Therefore, we set B = {1/2, . . . , 1/2log2
√

H} which
is the candidate values for unknown δ†.

The regret is composed of two factors from the master and bases. To ensure that the regret bound from
each base concerning VT and ST remains guaranteed irrespective of the bases chosen, we consider a
constrained adaptive adversary. For the following, we consider that ϱt for all t > 0 are arbitrarily
determined before an algorithm is run, under the constraints of VT and ST where PT −1
t=1 ϱt ≤VT
and 1 + PT −1
t=1 1(ϱt ̸= 0) ≤ST . Then under ϱt for all t > 0, we consider the following adversary
for rotting rates.
Assumption A.10 (Constrained Adaptive Adversary). At each time t > 0, the value of rotting
rate ρt is determined arbitrarily immediately after the agent pulls an arm at under the constraint of
0 ≤ρt ≤ϱt for given ϱt.
Remark A.11. Assumption A.10 is still more general than that for the maximum rotting rate constraint
in Kim et al. [13] where ϱt = ρ for all t > 0. We also observe that the constrained adaptive adversary
in Assumption A.10 is milder than the adaptive adversary in Assumption 2.1. Additionally, we note
that a special case of constraint ρt = ϱt for all t > 0 in Assumption A.10 represents an oblivious
adversary because ϱt’s are determined before an algorithm is run.

With a time block size of H (where H = ⌈
√

T⌉), the algorithm operates over ⌈T/H⌉blocks. Denote
by Ti the set of time steps in the i-th block containing time steps of (i −1)H + 1 ≤t ≤iH ∧T. It
is possible to encounter large rotting for some i block, potentially resulting in an arm’s mean reward
having a significantly low negative value, leading to suboptimal behavior by the master incurring
large regret from the master. To address this, we introduce the assumption of equally distributed
cumulative rotting for blocks, stated as follows:
Assumption A.12. P
t∈Ti ρt ≤H for all i ∈[⌈T/H⌉]
Remark A.13. As similarly highlighted in Remark 2.5, this assumption is satisfied when mean
rewards are under the constraint of 0 ≤µt(at) ≤1 for all t ∈[T], which is frequently encountered
in real-world applications where reward is represented by metrics like click rates or (normalized)
ratings in content recommendation systems.
Remark A.14. Our rotting scenario withP

t∈Ti ρt ≤H for all i ∈[⌈T/H⌉] is more general in scope
than the one with a maximum rotting rate constraint where ρt ≤ρ = o(1) for all t ∈[T −1],
which was explored in Kim et al. [13]. This is because for our setting, ρt is not necessarily bounded
by o(1), and for the maximum rotting constraint setting with ρt ≤ρ = o(1), the condition of
P

t∈Ti ρt ≤H for all i ∈[⌈T/H⌉] is always satisfied. It is noteworthy that Assumption A.12 implies
Assumption 2.4.

We provide a regret bound of Algorithm 2 under Assumption A.10 and Assumption A.12 in the
following.
Theorem A.15. Let R′
V and R′
S be defined as

R′
V :=

(
V

1
β+2
T
T

β+1
β+2 + T

2β+1
2β+2
for β ≥1,

V

1
3
T T
2
3 + T
3
4
for 0 < β < 1
and R′
S :=

(
max{S

1
β+1
T
T

β
β+1 + T

2β+1
2β+2 , VT }
for β ≥1,
max{√ST T + T
3
4 , VT }
for 0 < β < 1.

Then, the policy π of Algorithm 2 with H = ⌈
√

T⌉achieves the following regret bound:

E[Rπ(T)] = ˜O(min{R′
V , R′
S})

Proof. The proof is provided in Appendix A.7.

We can observe that there is the additional regret cost of T (2β+1)/(2β+2) for β ≥1 or T 3/4 for
0 < β < 1 compared to Algorithm 1. This additional cost originates from the additional procedure to
learn the optimal value of δ in Algorithm 2, which is negligible when VT and ST are large enough.
We also note that the well-known black-box framework proposed for addressing nonstationarity

28


---Page Break---
[25] is not applicable to this problem because the UCB of the chosen arm in this context does not
consistently surpass the optimal mean reward, violating the necessary assumption for the framework.
Attaining the optimal regret bound under a parameter-free algorithm remains an unresolved issue.

A.7
Proof of Theorem A.15: Regret Upper Bound of Algorithm 2

In the following, we deal with the cases of (a) δ†
V ≤δ†
S so that δ†
V S = δ†
V and (b) δ†
V > δ†
S so that
δ†
V S = δ†
S, separately.

A.7.1
Case of δ†
V ≤δ†
S

Let πi(δ′) for δ′ ∈B denote the base policy for time steps between (i −1)H + 1 and i · H ∧T in
Algorithm 2 using 1 −δ′ as a threshold. Denote by aπi(δ′)
t
the pulled arm at time step t by policy
πi(δ′). Then, for δ† ∈B, which is set later for a near-optimal policy, we have

E[Rπ(T)] = E




T
X

t=1
1 −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπ
t )



= E[Rπ
1 (T)] + E[Rπ
2 (T)].
(33)

where

Rπ
1 (T) =

T
X

t=1
1 −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπi(δ†)
t
)

and

Rπ
2 (T) =

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπi(δ†)
t
) −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπ
t ).

Note that Rπ
1 (T) accounts for the regret caused by the near-optimal base algorithm πi(δ†)’s
against the optimal mean reward and Rπ
2 (T) accounts for the regret caused by the master al-
gorithm by selecting a base with δ ∈B at every block against the base with δ†. In what fol-
lows, we provide upper bounds for each regret component. We first provide an upper bound
for E[Rπ
1 (T)] by following the proof steps in Theorem 3.1. Then we provide an upper bound
for E[Rπ
2 (T)]. We set H = ⌈T 1/2⌉and δ† to be a smallest value in B which is larger than
δ†
V = max{(VT /T)1/(β+2), (VT /T)1/3, 1/H1/(β+1), 1/H1/2}.
Remark A.16. One might wonder whether Rπ
1 (T), regret from the near-optimal base of δ†, satisfies
the constraint of VT and ST even though the master may not select the near-optimal base in the
algorithm for each block. From Assumption A.10, we can guarantee the constraints of VT and ST for
rotting rates from each base regardless of the selected bases from the master for each block because
the rotting upper bound ϱt’s are determined before staring the game regardless of the behavior of
the master. Therefore, we can utilize VT and ST for bounding Rπ
1 (T), which is the regret from the
near-optimal base of δ†.

Upper Bounding E[Rπ
1 (T)]. We refer to the period starting from time step (i −1)H + 1 to time
step i · H ∧T as the i-th block. For any i ∈⌈(T/H) −1⌉, policy πi(δ†) runs over H time steps
independent to other blocks so that each block has the same expected regret and the last block has
a smaller or equal expected regret than other blocks. Therefore, we focus on finding a bound on
the regret from the first block equal to PH
t=1 1 −µt(aπ1(δ†)
t
). We define an arm a as a good arm
if ∆(a) ≤δ†/2, a near-good arm if δ†/2 < ∆(a) ≤2δ†, and otherwise, a is a bad arm. In A, let
¯a1, ¯a2, . . . , be a sequence of arms, which have i.i.d. mean rewards following (1). Without loss of
generality, we assume that the policy samples arms in the sequence of ¯a1, ¯a2, . . . , .

Denote by A(i) the set of selected (explored) arms in the i-th block, which satisfies |A(i)| ≤H.
WLOG, we consider the case of given A(i) for the following because the proof can be applied to any
given A(i). Let µ[t1,t2](a) = Pt2
t=t1 µt(a)/n[t1,t2](a). We define the event E1 = {|bµ[s1,s2](a) −
µ[s1,s2](a)| ≤
p

12 log(H)/n[s1,s2](a) for all 1 ≤s1 ≤s2 ≤H, a ∈A(i)}. As in (6), we have

P(Ec
1) ≤2/H2.

29


---Page Break---
We denote by VH,i = P
t∈Ti ρt the cumulative amount of rotting in the time steps in the i-th block.
From the cumulative amount of rotting, we note that ∆t(a) = O(VH,i + 1) for any a and t in i-th
block, which implies E[Rπ(T)|Ec
1] = O(H2) from VH,i ≤H under Assumption A.12. For the case
where E1 does not hold, the regret is E[Rπ(T)|Ec
1]P(Ec
1) = O(1), which is negligible compared
to the regret when E1 holds, which we show later. For the case that E1 does not hold, the regret
is E[Rπ(H)|Ec
1]P(Ec
1) = O(1), which is negligible compared with the regret when E1 holds true
which we show later. Therefore, in the rest of the proof we assume that E1 holds true.

In the following, we first provide a regret bound over the first block.

For regret analysis, we divide Rπ1(δ†)(H) into two parts, RG(H) and RB(H) corresponding to
regret of good or near-good arms, and bad arms over time H, respectively, such that Rπ1(δ†)(H) =
RG(H) + RB(H). We denote by VH,i the cumulative amount of rotting in the time steps in the i-th
block. We first provide a bound of RG(H) in the following lemma.

Lemma A.17. Under E1 and policy π, we have

E[RG(H)] = ˜O

Hδ† + H2/3E[V 1/3
H,1 ]

.

Proof. We can easily prove the theorem by following the proof steps in Lemma A.1

Now, we provide a regret bound for RB(H). We note that the initially bad arms can be defined only
when 2δ† < 1. Otherwise when 2δ† ≥1, we have R(T) = RG(T), which completes the proof.
Therefore, for the regret from bad arms, we consider the case of 2δ† < 1. For the proof, we adopt the
episodic approach in Kim et al. [13] for regret analysis.

Given a policy sampling arms in the sequence order, let mG be the number of samples of distinct good
arms and mB
i be the number of consecutive samples of distinct bad arms between the i −1-st and
i-th sample of a good arm among mG good arms. We refer to the period starting from sampling the
i −1-st good arm before sampling the i-th good arm as the i-th episode. Observe that mB
1 , . . . , mB
mG
are i.i.d. random variables with geometric distribution with parameter 2δ, given a fixed value of mG.
Therefore, for non-negative integer k we have P(mB
i = k) = (1 −2δ†)k2δ†, for i = 1, . . . , mG.
Define ˜mH to be the number of episodes from the policy π over the horizon H, ˜mG
H to be the total
number of samples of a good arm by the policy π over the horizon H such that ˜mG
H = ˜mH or
˜mG
H = ˜m −1, and ˜mB
i,H to be the number of samples of a bad arm in the i-th episode by the policy
π1(δ†) over the horizon H.

Under a policy π1(δ†), let RB
i,j be the regret (summation of mean reward gaps) contributed by pulling

the j-th bad arm in the i-th episode. Then let RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j, which is the regret from
initially bad arms over the period of mG episodes.

For obtaining a regret bound, we first focus on finding a required number of episodes, mG, such that
RB(T) ≤RB
mG. Then we provide regret bounds for each bad arm and good arm in an episode. Lastly,
we obtain a regret bound for E[RB(T)] using the episodic regret bound.

Let a(i) be a good arm in the i-th episode and a(i, j) be a j-th bad arm in the i-th episode. We define
VH(a) = PH
t=1 ρt1(at = a). Then excluding the last episode ˜mH over H, we provide lower bounds
of the total rotting variation over H for a(i), denoted by VH(a(i)), in the following lemma.

Lemma A.18. Under E1, given ˜mH, for any i ∈[ ˜mG
H]/{ ˜mH} we have

VH(a(i)) ≥δ†/2.

Proof. We can easily prove the theorem by following the proof steps in Lemma A.2

We first consider the case where VT > max{T/H3/2, T/H(β+2)/(β+1)}. In this case, we have
δ† = max{(VT /T)1/(β+2), (VT /T)1/3}. Here, we define the policy π after time H such that it pulls
a good arm until its total rotting variation is equal to or greater than δ†/2 and does not pull a sampled
bad arm. We note that defining how π works after H is only for the proof to get a regret bound

30


---Page Break---
over time horizon H. For the last arm ˜a over the horizon H, it pulls the arm until its total variation
becomes max{δ†/2, VH(˜a)} if ˜a is a good arm. For i ∈[mG], j ∈[mB
i ] let V G
i and V B
i,j be the total
rotting variation of pulling the good arm in i-th episode and j-th bad arm in i-th episode from the
policy, respectively. Here we define V G
i ’s and V B
i,j’s as follows:

If ˜a is a good arm,

V G
i =
VH(a(i))
for i ∈[ ˜mG
H −1]
max{δ†/2, VH(a(i))}
for i ∈[mG]/[ ˜mG
H −1] , V B
i,j =
VH(a(i, j))
for i ∈[ ˜mG
H], j ∈[ ˜mB
i,H]
0
for i ∈[mG]/[ ˜mG
H], j ∈[mB
i ].

Otherwise,

V G
i =
VH(a(i))
for i ∈[ ˜mG
H]
δ†/2
for i ∈[mG]/[ ˜mG
H] , V B
i,j =
VH(a(i, j))
for i ∈[ ˜mG
H], j ∈[ ˜mB
i,H]
0
for i ∈[mG]/[ ˜mG
H −1], j ∈[mB
i ]/[ ˜mB
i,H].

For i ∈[mG], j ∈[mB
i ] let nB
i,j be the number of pulling the good arm in i-th episode and j-th bad
arm in i-th episode from the policy, respectively. We define nH(a) be the total amount of pulling arm
a over H. Here we define nB
i,j’s as follows:

nB
i,j =
nH(a(i, j))
for i ∈[ ˜mG
H], j ∈[ ˜mB
i,H]
0
for i ∈[mG]/[ ˜mG
H], j ∈[mB
i ].

Then we provide mG such that RB(H) ≤RB
mG in the following lemma.

Lemma A.19. Under E1, when mG = ⌈2VH,1/δ†⌉we have

RB(H) ≤RB
mG.

Proof. We can easily show the theorem by following the proof steps of Lemma A.3

From the result of Lemma A.19, we set mG = ⌈2VH,1/δ†⌉. In the following, we anlayze RB
mG for
obtaining a regret bound for RB(H).
Lemma A.20. Under E1 and policy π, we have

E[RB
mG] = ˜O

max{VH,1(T/VT )(β+1)/(β+2) + (T/VT )β/(β+2), VH,1(T/VT )2/3 + (T/VT )1/3}

.

Proof. We can easily prove the theorem by following proof steps in Lemma A.4. From (24), for any
i ∈[mG], j ∈[mB
i ], we have

E[RB
i,j] ≤E

∆(a(i, j))nB
i,j + V B
i,jnB
i,j


= ˜O
 
max{1, (δ†)β−1} + E[V B
i,j]/(δ†)2
.

Recall that RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j. With δ† = max{(VT /T)1/(β+2), (VT /T)1/3} and mG =
⌈2VH,1/δ†⌉, from the fact that mB
i ’s are i.i.d. random variables with geometric distribution with
E[mB
i ] = 1/(2δ†)β −1, we have

E[RB
mG] = O



E




mG
X

i=1

X

j∈[mB
i ]
RB
i,j









= ˜O

(E[VH,1]/δ† + 1)
1
(δ†)β max{1, (δ†)β−1} + E[VH,1]/(δ†)2


= ˜O

max
 E[VH,1]

(δ†)β+1 , E[VH,1]

(δ†)2


+ max

1
(δ†)β , 1

δ†



= ˜O

max{E[VH,1](T/VT )(β+1)/(β+2) + (T/VT )β/(β+2), E[VH,1](T/VT )2/3 + (T/VT )1/3}

.

31


---Page Break---
From Rπ1(δ†)(H)
=
RG(H) + RB(H) and Lemmas A.17, A.19, A.20, with δ†
=
max{(VT /T)1/(β+2), (VT /T)1/3} we have

E[Rπ1(δ†)(H)]

= ˜O

max
n
E[VH,1](T/VT )(β+1)/(β+2) + H(VT /T)1/(β+2) + (T/VT )β/(β+2),

E[VH,1](T/VT )2/3 + H(VT /T)1/3 + (T/VT )1/3o
+ H2/3E[V 1/3
H,1 ]

.

The above regret bound is for the first block. Therefore, by summing regrets from ⌈T/H⌉number of
blocks, from VT > max{T/H(β+2)/(β+1), T/H3/2}, H = ⌈T 1/2⌉and the fact that E[PT −1
t=1 ρt] ≤
VT , using Hölder’s inequality we have shown that

E[Rπ
1 (T)] = ˜O

max{T (β+1)/(β+2)V 1/(β+2)
T
, T 2/3V 1/3
T
} + T

H max{(T/VT )β/(β+2), (T/VT )1/3}


= ˜O

max{T (β+1)/(β+2)V 1/(β+2)
T
, T 2/3V 1/3
T
} + max{T (2β+1)/(2β+2), T 3/4}

.

(34)

Now, we consider the case where VT ≤max{T/H3/2, T/H(β+2)/(β+1)}. In this case, we have
δ† = max{1/
√

H, 1/H
1
β+1 }. From the result of Lemma A.19, by setting mG = ⌈2VH,1/δ†⌉we
have RB(H) ≤RB
mG.

Lemma A.21. Under E1 and policy π, we have

E[RB
mG] = ˜O

max{VH,1(T/VT )(β+1)/(β+2) + (T/VT )β/(β+2), VH,1(T/VT )2/3 + (T/VT )1/3}

.

Proof. We can easily prove the theorem by following proof steps in Lemma A.4. From (24), for any
i ∈[mG], j ∈[mB
i ], we have

E[RB
i,j] ≤E

∆(a(i, j))nB
i,j + V B
i,jnB
i,j


= ˜O
 
max{1, δβ−1} + E[V B
i,j]/δ2
.

Recall that RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j. With δ† = max{1/H1/2, 1/H1/(β+1)} and mG =
⌈2VH,1/δ†⌉, from the fact that mB
i ’s are i.i.d. random variables with geometric distribution with
E[mB
i ] = 1/(2δ†)β −1, we have

E[RB
mG] = O



E




mG
X

i=1

X

j∈[mB
i ]
RB
i,j









= ˜O

(E[VH,1]/δ† + 1)
1
(δ†)β max{1, (δ†)β−1} + E[VH,1]/(δ†)2


= ˜O

max
 E[VH,1]

(δ†)β+1 , E[VH,1]

(δ†)2


+ max

1
(δ†)β , 1

δ†



= ˜O

E[VH,1]H + max{Hβ/(β+1), H1/2}

.

From Rπ1(δ†)(H)
=
RG(H) + RB(H) and Lemmas A.17, A.19, A.21, with δ†
=
Θ(max{1/H1/2, 1/H1/(β+1)}) we have

E[Rπ1(δ†)(H)] = ˜O

max{Hβ/(β+1), H1/2} + H2/3E[V 1/3
H,1 ] + E[VH,1]H

.

32


---Page Break---
Therefore,
by
summing
regrets
from
⌈T/H⌉
number
of
blocks
and
from
VT
=
O(max{T/H3/2, T/H(β+2)/(β+1)}), H = ⌈T 1/2⌉, and the fact that length of time steps in each
block is bounded by H, we have

E[Rπ
1 (T)] = ˜O



T

H max{Hβ/(β+1), H1/2} +

⌈T/H⌉
X

i=1
H2/3E[V 1/3
H,i ] +

⌈T/H⌉
X

i=1
E[VH,i]H





= ˜O
 T

H max{Hβ/(β+1), H1/2} + T 2/3V 1/3
T
+ VT H


= ˜O

max{T/H1/(β+1), T/H1/2}


= ˜O

max{T (2β+1)/(2β+2), T 3/4}

,
(35)

where the second equality comes from Hölder’s inequality.

From (34) and (35), we have

E[Rπ
1 (T)] = ˜O(max{T (β+1)/(β+2)V 1/(β+2)
T
+ T (2β+1)/(2β+2), T 2/3V 1/3
T
+ T 3/4}).
(36)

Upper Bounding E[Rπ
2 (T)]. We observe that the EXP3 is run for ⌈T/H⌉decision rounds and the
number of policies (i.e. πi(δ′) for δ′ ∈B) is B. Denote the maximum absolute sum of rewards of any
block with length H by a random variable Q′. We first provide a bound for Q′ using concentration
inequalities. For any block i, we have


i·H∧T
X

t=(i−1)H+1
µt(aπ
t ) + ηt


≤



i·H∧T
X

t=(i−1)H+1
µt(aπ
t )


+



i·H∧T
X

t=(i−1)H+1
ηt


.
(37)

Denote by Ti the set of time steps in the i-th block. We define the event E2(i) = {|bµ[s1,s2](a) −
µ[s1,s2](a)| ≤
p

14 log(H)/n[s1,s2](a), for all s1, s2 ∈Ti, s1 ≤s2, a ∈A(i)} and E2 =
T
i∈[⌈T/H⌉] E2(i). From Lemma A.29, with H = ⌈
√

T⌉we have

P(Ec
2) ≤
X

i∈[⌈T/H⌉]

2H3

H6 ≤2

T .

By assuming that E2 holds true, we can get a lower bound for µt(aπ
t ), which may be a neg-
ative value from rotting, for getting an upper bound for | Pi·H∧T
t=(i−1)H+1 µt(aπ
t )|.
We can ob-

serve that Pi·H∧T
t=(i−1)H+1 µt(aπ
t ) ≤H. Therefore the remaining part is to get a lower bound for
Pi·H∧T
t=(i−1)H+1 µt(aπ
t ). For the proof simplicity, we consider that when an arm is rotted, then the arm
is considered as a different arm after rotting. For instance, when arm a is rotted at time s, then arm a
is considered as a different arm a′ after s. Therefore, each arm can be considered to be stationary.
The set of arms is denoted by L. We denote by L+ the set of arms having µt(a) ≥0 for a ∈L. We
first focus on the arms in L/L+.

Let δmax denote the maximum value in B so that δmax = 1/2. With E2 and a ∈L/L+, we assume
that ˜t2(≥t2) is the smallest time that there exists t1 ≤s ≤˜t2 with s = ˜t2 + 1 −2l−1 for l ∈Z+
such that the following condition is met:

µt1(a) +
q

12 log(H)/n[s,˜t2](a) +
q

14 log(H)/n[s,˜t2](a) < 1 −δmax.
(38)

From the definition of ˜t2, we observe that for given ˜t2, the time step s, which satisfies (38),
equals to t1. Then, we can observe that n[t1,˜t2](a) = max{⌈C2 log(H)/(∆t1(a) −δmax)2⌉, 1}
for some constant C2 > 0, which satisfies (38).
From n[t1,t2](a) ≤n[t1,˜t2](a), we have
n[t1,t2](a) ≤max{C3 log(H)/(∆t1(a) −δmax)2, 1} for some constant C3 > 0.
Then the
regret from arm a, denoted by R(a), is bounded as follows: R(a) = ∆t1(a)n[t1,t2](a) ≤
max{C3 log(H)∆t1(a)/(∆t1(a) −δmax)2, ∆t1(a)}. Since x/(x −δmax)2 < 1/(1 −δmax)2 =
4 for any x
>
1, we have ∆t1(a)/(∆t1(a) −δmax)2
≤
4.
Then we have R(a)
≤

33


---Page Break---
max{C4 log(H), ∆t1(a)} for some constant C4
>
0.
Then from |L|
≤
H, we have
P

a∈L/{L+} R(a) ≤max{C4H log(H), H + VH,i}.

Since P

a∈L+ R(a) ≤H, we have P

a∈L R(a) ≤H + max{C4H log(H), H + VH,i}. Therefore
from R(a) = Pt2(a)
t=t1(a)(1 −µt(a)), we have

iH∧T
X

t=(i−1)H+1
µt(at) ≥−max{C4H log(H), H + VH,i},

which implies that from VH,i ≤H under Assumption A.12, for some C5 > 0, we have


i·H∧T
X

t=(i−1)H+1
µt(aπ
t )


≤max{C4H log(H), H + VH,i} ≤C5H log(H).

Next we provide a bound for | Pi·H∧T
t=(i−1)H+1 ηt|. We define the event E3(i) = {| Pi·H∧T
t=(i−1)H+1 ηt| ≤

2
p

H log(T)} and E3 = T

i∈[⌈T/H⌉] E3(i). From Lemma A.29, for any i ∈[⌈T/H⌉], we have

P (E3(i)c) ≤2

T 2 .

Then, under E2 ∩E3, with (37), we have

Q′ ≤max{C5H log H, H} + 2
p

H log(T) ≤C5H log H + 2
p

H log(T),

which implies 1/2+Pi·H∧T
t=(i−1)H rt/(C5H log H+4√H log T) ∈[0, 1] or some large enough C > 0.
With the rescaling and translation of rewards in Algorithm 2, from Corollary 3.2. in Auer et al. [2],
we have

E[Rπ
2 (T)|E2 ∩E3] = ˜O

(C5H log H + 2
p

H log T)
p

BT/H

= ˜O
√

HBT

.
(39)

Remark A.22. Regarding the utilization of the regret analysis of Corollary 3.2 (EXP3) in Auer et al.
[2], we note that the reward for each base can be defined independently of the actual master’s action.
One might wonder whether the regret analysis for EXP3 can be utilized, considering the fact that
the reward from a selected base may depend on the master’s action due to the adaptive rotting rates.
However, we highlight that the critical aspect of applying EXP3 analysis is whether the rewards
from each base are defined independently of the actual action of the master, rather than whether the
received (observed) reward from the selected base depends on the master’s action. We can construct
rewards for each base δ ∈B at time t when a block starts, denoted as xt(δ), as the reward obtained
when the master selects base δ (even though base δ is not actually selected from the algorithm). Then,
we can define xt(δ) for each δ regardless of the master’s actual action. In other words, irrespective of
the actually selected base, we define xt(δ) for all δ ∈B as the reward that the master can obtain by
selecting δ. In such a case, whatever the selected base by the master is at time t, xt(δ)’s remain the
same, respectively. This construction is feasible because it’s solely for analytical purposes and not
necessary for the algorithm’s functioning. With this construction of reward for each base, we can
utilize EXP3 analysis to obtain a regret bound regarding the master (Rπ
2 (T)).

Note that the expected regret from EXP3 is trivially bounded by o(H2(T/H)) = o(TH) and
B = O(log(T)). Then, with (39), we have

E[Rπ
2 (T)] = E[Rπ
2 (T)|E2 ∩E3]P(E2 ∩E3) + E[Rπ
2 (T)|Ec
2 ∪Ec
3]P(Ec
2 ∪Ec
3)

= ˜O
√

HT

+ o (TH) (4/T 2)

= ˜O
√

HT

.
(40)

Finally, from (33), (36), and (40), with H = T 1/2, we have

E[Rπ(T)] = ˜O

max

V

1
β+2
T
T

β+1
β+2 + T

2β+1
2β+2 , V

1
3
T T
2
3 + T
3
4

,

which concludes the proof.

34


---Page Break---
A.7.2
Case of δ†
V > δ†
S

Let πi(δ′) for δ′ ∈B denote the base policy for time steps between (i −1)H + 1 and i · H ∧T in
Algorithm 2 using 1 −δ′ as a threshold. Denote by aπi(δ′)
t
the pulled arm at time step t by policy
πi(δ′). Then, for δ† ∈B, which is set later for a near-optimal policy, we have

E[Rπ(T)] = E




T
X

t=1
1 −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπ
t )



= E[Rπ
1 (T)] + E[Rπ
2 (T)].
(41)

where

Rπ
1 (T) =

T
X

t=1
1 −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπi(δ†)
t
)

and

Rπ
2 (T) =

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπi(δ†)
t
) −

⌈T/H⌉
X

i=1

i·H∧T
X

t=(i−1)H+1
µt(aπ
t ).

Note that Rπ
1 (T) accounts for the regret caused by the near-optimal base algorithm πi(δ†)’s against
the optimal mean reward and Rπ
2 (T) accounts for the regret caused by the master algorithm by
selecting a base with δ ∈B at every block against the base with δ†. In what follows, we provide upper
bounds for each regret component. We first provide an upper bound for E[Rπ
1 (T)] by following the
proof steps in Theorem 3.3. Then we provide an upper bound for E[Rπ
2 (T)]. We set δ† to be a smallest
value in B which is larger than δ†
S = max{(ST /T)1/(β+1), 1/H1/(β+1), (ST /T)1/2, 1/H1/2} such
that we have δ† = Θ(max{(ST /T)1/(β+1), 1/H1/(β+1), (ST /T)1/2, 1/H1/2}).

Upper Bounding E[Rπ
1 (T)]. We refer to the period starting from time step (i −1)H + 1 to time
step i · H ∧T as the i-th block. For any i ∈⌈T/H −1⌉, policy πi(δ†) runs over H time steps
independent to other blocks so that each block has the same expected regret and the last block has
a smaller or equal expected regret than other blocks. Therefore, we focus on finding a bound on
the regret from the first block equal to PH
t=1 1 −µt(aπ1(δ†)
t
). We define an arm a as a good arm if
∆t(a) ≤δ†/2, a near-good arm if δ†/2 < ∆t(a) ≤2δ†, and otherwise, a is a bad arm at time t. In
A, let ¯a1, ¯a2, . . . , be a sequence of arms, which have i.i.d. mean rewards following (1). For analysis,
we consider abrupt change as sampling a new arm. In other words, if a sudden change occurs to an
arm a by pulling the arm a, then the arm is considered to be two different arms; before and after the
change. The type of abruptly rotted arms (good, near-good, or bad) after the change is determined by
the rotted mean reward. Without loss of generality, we assume that the policy samples arms, which
are pulled at least once, in the sequence of ¯a1, ¯a2, . . . , .

Denote by A(i) the set of sampled arms, which are pulled at least once, in the i-th block, which
satisfies |A(i)| ≤H. We also define AS(i) as a set of arms that have been rotted and pulled at
least once in the i-th block, which satisfies |AS(i)| ≤Si, where Si is defined as the number of
abrupt changes in the i-th block. Let µ[t1,t2](a) = Pt2
t=t1 µt(a)/n[t1,t2](a). We define the event
E1 = {|bµ[s1,s2](a) −µ[s1,s2](a)| ≤
p

12 log(H)/n[s1,s2](a) for all 1 ≤s1 ≤s2 ≤H, a ∈A(i)}.
From Lemma A.29, as in (6), we have

P(Ec
1) ≤2/H2.

For the case that E1 does not hold, the regret is E[Rπ(H)|Ec
1]P(Ec
1) = O(1), which is negligible
comparing with the regret when E1 holds true which we show later. Therefore, in the rest of the
proof we assume that E1 holds true.

In the following, we first provide a regret bound over the first block.

For regret analysis, we divide Rπ1(δ†)
1
(H) into two parts, RG(H) and RB(H) corresponding to
regret of good or near-good arms, and bad arms over time T, respectively, such that Rπ
1 (H) =
RG(H) + RB(H). We can easily obtain that

E[RG(H)] = O(δ†H),
(42)

35


---Page Break---
from ∆(a) ≤2δ† for any good or near-good arms a.

Now we analyze RB(H). We divide regret RB(H) into two regret from bad arms in A(1)/AS(1),
denoted by RB,1(H), and regret from bad arms in AS(1), denoted by RB,2(H) such that RB(H) =
RB,1(H) + RB,2(H). We first analyze RB,1(H) in the following. We consider arms in A(1)/AS(1).
For the proof, we adopt the episodic approach in Kim et al. [13] for regret analysis. In the following,
we introduce some notation. Here we only consider arms in A(1)/AS(1) so that the following
notation is defined without considering (rotted) arms in AS(1). Given a policy sampling arms in
the sequence order, let mG be the number of samples of distinct good arms and mB
i be the number
of consecutive samples of distinct bad arms between the i −1-st and i-th sample of a good arm
among mG good arms. We refer to the period starting from sampling the i −1-st good arm before
sampling the i-th good arm as the i-th episode. Observe that mB
1 , . . . , mB
mG’s are i.i.d. random
variables with geometric distribution with parameter 2δ†, conditional on the value of mG. Therefore,
P(mB
i = k) = (1 −2δ†)k2δ†, for i = 1, . . . , mG.

Define ˜mG
H to be the total number of samples of a good arm by the policy π1(δ†) over the horizon
H and ˜mB
i,H to be the number of selections of a bad arm in the i-th episode by the policy π over
the horizon H. For i ∈[ ˜mG
H], j ∈[ ˜mB
i,H], let ˜nG
i be the number of pulls of the good arm in the i-th
episode and ˜nB
i,j be the number of pulls of the j-th bad arm in the i-th episode by the policy π1(δ†)
over the horizon H. Let ˜a be the last sampled arm over time horizon H by π1(δ†).

With a slight abuse of notation, we use π1(δ†) for a modified strategy after H. Under a policy π1(δ†),
let RB
i,j be the regret (summation of mean reward gaps) contributed by pulling the j-th bad arm in

the i-th episode. Then let RB
mG = PmG

i=1
P

j∈[mB
i ] RB
i,j, which is the regret from initially bad arms
over the period of mG episodes. For getting RB
mG, here we define the policy π1(δ†) after H such that
it pulls H amounts for a good arm and zero for a bad arm. After H we can assume that there are
no abrupt changes. For the last arm ˜a over the horizon H, it pulls the arm up to H amounts if ˜a is a
good arm and ˜nG
˜mG
H < H. For i ∈[mG], j ∈[mB
i ] let nG
i and nB
i,j be the number of pulling the good

arm in i-th episode and j-th bad arm in i-th episode under π, respectively. Here we define nG
i ’s and
nB
i,j’s as follows:

If ˜a is a good arm,

nG
i =






˜nG
i
for i ∈[ ˜mG
H −1]
H
for i = ˜mG
H
0
for i ∈[mG]/[ ˜mG
H]
, nB
i,j =
˜nB
i,j
for i ∈[ ˜mG
H], j ∈[ ˜mB
i,H]
0
for i ∈[mG]/[ ˜mG
H], j ∈[mB
i ]/[ ˜mB
i,H].

Otherwise,

nG
i =






˜nG
i
for i ∈[ ˜mG
H]
H
for i = ˜mG
H + 1
0
for i ∈[mG]/[ ˜mG
H + 1]
, nB
i,j =
˜nB
i,j
for i ∈[ ˜mG
H], j ∈[ ˜mB
i,H]
0
for i ∈[mG]/[ ˜mG
H −1], j ∈[mB
i ]/[ ˜mB
i,H].

With a slight abuse of notation, we define Si to be the number of abrupt changes in i-th block. Then,
we show that if mG = S1, then RB,1(H) ≤RB
mG.

Lemma A.23. Under E1, when mG = S1 we have

RB,1(H) ≤RB
mG.

Proof. There are at most S1 −1 number of abrupt changes over the first block H. We consider two
cases; there are S1 −1 abrupt changes before sampling S1-th good arm or not. For the first case, if
π1(δ†) samples the S1-th good arm and there are S1 −1 number of abrupt changes before sampling
the good arm, then it continues to pull the good arm for H rounds from E1 and the definition of
π1(δ†) after H.

Now we consider the second case. If π1(δ†) samples the S1-th good arm before T and there is at least
one abrupt change after sampling the arm, then before sampling the S1-th good arm, there must exist
two consecutive good arms such that there is no abrupt change between sampling the two good arms.

36


---Page Break---
This is a contraction because π1(δ†) must pull the first good arm up to H under E1 and S1 −1-st
abrupt change must occur after H.

Therefore, considering the first case, when mG = S1 + 1, we have
X

i∈[mG]
nG
i ≥H,

which implies RB(H) ≤RB
mG.

From the above lemma, we set mG = S1 and analyze RB
mG to get a bound for RB,1(H) in the
following lemma.

Lemma A.24. Under E1 and policy π1(δ†), we have

E[RB
mG] = ˜O
 
E[S1 max{1/(δ†)β, 1/δ†}]

.

Proof. We can show this theorem by following the proof steps in Lemma A.8.

Now we analyze RB,2(H) in the following lemma. We denote by VH a cumulative amount of rotting
rates in the first block.

Lemma A.25. Under E1 and policy π, we have

E

RB,2(H)

= ˜O

 

E

"

max{S1/δ†,

S1
X

s=1
ρt(s)}

#!

.

Proof. We can show this theorem by following the proof steps in Lemma A.9.

From Lemmas A.23, A.24, A.25, we have

E[RB(H)] = E[RB,1(H)] + E[RB,2(H)] = ˜O

 

E

"

S1 max{1/(δ†)β, 1/δ†} +

S1
X

s=1
ρt(s)

#!

(43)

From Rπ
1 (H) = RG(H) + RB(H), (42), and (43), we have

E[Rπ1(δ†)
mG
] = ˜O

 

Hδ† + E

"

S1 max{1/(δ†)β, 1/δ†} +

S1
X

s=1
ρt(s)

#!

.

The above regret is for the first block. Therefore, by summing regrets over ⌈T/H⌉number of blocks,
we have shown that

E[Rπ
1 (T)] = ˜O(Tδ† + (T/H + ST ) max{1/(δ†)β, 1/δ†} +

ST
X

s=1
E[ρt(s)]).
(44)

Upper bounding E[Rπ
2 (T)]. By following the proof steps in Theorem A.15, we have

E[Rπ
2 (T)] = ˜O
√

HT

.
(45)

37


---Page Break---
Finally, from (41), (44), and (45), with the fact that PST
s=1 E[ρt(s)] ≤VT , H = T 1/2, and δ† =
Θ(max{(ST /T)1/(β+1), 1/H1/(β+1), (ST /T)1/2, 1/H1/2}), we have

E[Rπ(T)] = ˜O

 

Tδ† + (T/H + ST ) max{1/(δ†)β, 1/δ†} +
√

HT +

ST
X

s=1
E[ρt(s)]

!

= ˜O

 

Tδ† + max{T/H, ST } max{1/(δ†)β, 1/δ†} +
√

HT +

ST
X

s=1
E[ρt(s)]

!

= ˜O

 

2Tδ† +
√

HT +

ST
X

s=1
E[ρt(s)]

!

= ˜O

max{S1/(β+1)
T
T β/(β+1) + T (2β+1)/(2β+2),
p

ST T + T 3/4, VT }

,

which concludes the proof.

A.8
Proof of Theorem 4.1: Regret Lower Bound for Slowly Rotting Rewards

We first consider the case when VT = Θ(T). Recall that ∆1(a) = 1 −µ1(a). Then for any randomly
sampled a ∈A, we have E[µ1(a)] ≥yP(µ1(a) ≥y) = yP(∆1(a) < 1 −y) for y ∈[0, 1]. Then
with y = 1/2, we have E[µ1(a)] ≥(1/2)P(∆1(a) < (1/2)) = Θ(1) from constant β > 0 and (1).
Then with E[µ1(a)] ≤1, we have E[µ1(a)] = Θ(1). We then think of a policy π′ that randomly
samples a new arm and pulls it once every round. Since E[µ1(a)] = Θ(1) for any randomly sampled
a, we have E[Rπ′(T)] = Θ(T). Next, we think of any policy π′′ except π′. Then any policy π′′ must
pull an arm a at least twice. Let t′ and t′′ be the rounds when the policy pulls arm a. If we consider
ρt′ = VT then such policy has Ω(VT ) regret bound. Since VT = Θ(T), any algorithm has Ω(T) in
the worst case. Therefore we can conclude that any algorithm including π′ has a regret bound of
Ω(T) in the worst case, which concludes the proof for VT = Θ(T).

Now we think of the case where VT = o(T). For the lower bound, we adopt the proof methodology
of Theorem 1 in Kim et al. [13] by making necessary adjustments to accommodate VT and β. We note
that since higher β implies a reduced chance of sampling a near-optimal arm, the criteria for defining
the mean rewards of near-optimal arms becomes less stringent for higher β, which does not appear
in the previous work. We first categorize arms as either bad or good according to their initial mean
reward values. For the categorization, we utilize two thresholds in the proof as follows. Consider
0 < γ < c < 1 for γ, which will be specified, and a constant c. Then the value of 1 −γ represents a
threshold value for identifying good arms, while 1 −c serves as the threshold for identifying bad
arms. We refer to arms a satisfying µ1(a) ≤1−c as ‘bad’ arms and arms a satisfying µ1(a) > 1−γ
as ‘good’ arms. We also consider a sequence of arms in A denoted by ¯a1, ¯a2, . . . . Given a policy π,
without loss of generality, we can assume that π selects arms according to the order of ¯a1, ¯a2, . . . .
For the rotting rates, we define ϱ = VT /(T −1). Then we consider ρt = ϱ for all t ∈[T −1] so that
PT −1
t=1 ρt = VT .

Case of VT = O(1/T 1/(β+1)): When VT = O(1/T 1/(β+1)), the lower bound of order T

β
β+1 for the
stationary case, from Theorem 3 in Wang et al. [24], is tight enough for the non-stationary case. From
Theorem 3 in Wang et al. [24], we have

E[Rπ(T)] = Ω(T

β
β+1 ).
(46)

We note that even though the mean rewards are rotting in our setting, Theorem 3 in Wang et al. [24]
remains applicable without requiring any alterations in the proofs providing a tight regret bound for
the near-stationary case. For the sake of completeness, we provide the proof of the theorem in the
following. Let K1 denote the number of bad arms a that satisfy µ1(a) ≤1 −c before sampling the
first good arm, which satisfies µ1(a) > 1 −γ, in the sequence of arms ¯a1, ¯a2, . . . . Let µ be the initial
mean reward of the best arm among the sampled arms by π over time horizon T. Then for some
κ > 0, we have

Rπ(T) = Rπ(T)1(µ ≤1 −γ) + Rπ(T)1(µ > 1 −γ)
≥Tγ1(µ ≤1 −γ) + K1c1(µ > 1 −γ)
≥Tγ1(µ ≤1 −γ) + κc1(µ > 1 −γ, K1 ≥κ).
(47)

38


---Page Break---
By taking expectations on the both sides in (47) and setting κ = Tγ/c, we have

E[Rπ(T)] ≥TγP(µ ≤1 −γ) + κc(P(µ > 1 −γ) −P(K1 < κ)) = cκP(K1 ≥κ).

We observe that K1 follows a geometric distribution with success probability P(µ1(a) > 1 −
γ)/p(µ1(a) /∈(1 −c, 1 −γ]) = γ ≤C1γβ/(1 + C2γβ −C3cβ) for some constants C1, C2, C3 > 0
from (1), in which the success probability is the probability of sampling a good arm given that the
arm is either a good or bad arm. Here we set a constant 0 < c < 1 satisfying 1 −C3cβ > 0. Then by
setting γ = 1/T
1
β+1 with κ = T

β
β+1 /c, for some constant C > 0 we have

E[Rπ(T)] ≥cκ(1 −γ)κ = Ω

T

β
β+1 (1 −Cγβ)T
β
β+1 /c

= Ω(T

β
β+1 ),

where the last equality is obtained from log x ≥1 −1/x for all x > 0.

Case of VT = ω(1/T 1/(β+1)) and VT = o(T): When VT = ω(1/T 1/(β+1)), however, the lower
bound of the stationary case is not tight enough. Here we provide the proof for the lower bound of
V 1/(β+2)
T
T (β+1)/(β+2) for the case of VT = ω(1/T 1/(β+1)). Let Km denote the number of “bad"
arms a that satisfy µ1(a) ≤1 −c before sampling m-th “good" arm, which satisfies µ1(a) > 1 −γ,
in the sequence of arms ¯a1, ¯a2, . . . . Let NT be the number of sampled good arms a such that
µ1(a) > 1 −γ until T.

We can decompose Rπ(T) into two parts as follows:

Rπ(T) = Rπ(T)1(NT < m) + Rπ(T)1(NT ≥m).
(48)

We set m = ⌈(1/2)T 1/(β+2)V (β+1)/(β+2)
T
⌉and γ = (VT /T)1/(β+2) with VT = o(T). For the first
term in (48), Rπ(T)1(NT < m), we consider the fact that the minimal regret is obtained from the
situation where there are m −1 arms whose mean rewards are 1. In such a case, the optimal policy
must sample the best m −1 arms until their mean rewards become below the threshold 1 −γ (step
1) and then samples the best arm at each time for the remaining time steps (step 2). The number
of times each arm needs to be pulled for the best m −1 arms until their mean reward falls below
1 −γ is bounded from above by γ/ρ + 1 = γ((T −1)/VT ) + 1. Therefore, the regret from step 2 is
R = Ω((T −mγ(T/VT ))γ) = Ω(T (β+1)/(β+2)V 1/(β+2)
T
) in which the optimal policy pulls arms
which mean rewards are below 1 −γ for the remaining time after step 1. Therefore, we have

Rπ(T)1(NT < m) = Ω(R1(NT < m)) = Ω(T (β+1)/(β+2)V 1/(β+2)
T
1(NT < m)).
(49)

For getting a lower bound of the second term in (48), Rπ(T)1(NT ≥m), we use the minimum
number of sampled arms a that satisfy µ1(a) ≤1 −c. When NT ≥m and Km ≥κ, the policy
samples at least κ number of distinct arms a satisfying µ1(a) ≤1 −c until T. Therefore, we have

Rπ(T)1(NT ≥m) ≥cκ1(NT ≥m, Km ≥κ).
(50)

We have γ = Θ(γβ) from (1) with constant β > 0. By setting κ = m/γ −m −√m/γ, with
VT = o(T) and constant β > 0, we have

κ = Θ(T (β+1)/(β+2)V 1/(β+2)
T
).
(51)

Then from (49), (50), and (51), we have

E[Rπ(T)] = Ω(T (β+1)/(β+2)V 1/(β+2)
T
P(NT < m) + T (β+1)/(β+2)V 1/(β+2)
T
P(NT ≥m, Km ≥κ))

≥Ω(T (β+1)/(β+2)V 1/(β+2)
T
P(Km ≥κ)).
(52)

Next we provide a lower bound for P(Km ≥κ). Observe that Km follows a negative binomial
distribution with m successes and the success probability P(µ1(a) > 1 −γ)/P(µ1(a) /∈(1 −c, 1 −
γ]) = γ, in which the success probability is the probability of sampling a good arm given that the arm
is either a good or bad arm. In the following lemma, we provide a concentration inequality for Km.
Lemma A.26. For any 1/2 + γ/m < α < 1,

P(Km ≥αm(1/γ) −m) ≥1 −exp(−(1/3)(1 −1/α)2(αm −γ)).

39


---Page Break---
Proof. Let Xi for i > 0 be i.i.d. Bernoulli random variables with success probability γ. From Section
2 in Brown [9], we have

P

Km ≤

αm 1

γ


−m

= P






⌊αm 1

γ ⌋
X

i=1
Xi ≥m




.
(53)

From (53) and Lemma A.28, for any 1/2 + γ/m < α < 1 we have

P

Km ≤αm 1

γ −m

= P

Km ≤

αm 1

γ


−m


= P






⌊αm 1

γ ⌋
X

i=1
Xi ≥m






≤exp

−(1 −1/α)2

3


αm 1

γ



γ


≤exp

−(1 −1/α)2

3
(αm −γ)

,

in which the first inequality comes from Lemma A.28, which concludes the proof.

From Lemma A.26 with α = 1 −1/√m and large enough T, we have

P(Km ≥κ) ≥1 −exp

 

−1

3(m −√m −γ)

1
√m −1

2!

≥1 −exp

 

−1

6(m −√m)

1
√m −1

2!

= 1 −exp

−1

6

√m
√m −1



≥1 −exp(−1/6).
(54)

Therefore, from (52) and (54), we have

E[Rπ(T)] = Ω(T (β+1)/(β+2)V 1/(β+2)
T
).
(55)

Finally, from (46) and (55), we conclude that for any policy π, we have

E[Rπ(T)] = Ω

max
n
T (β+1)/(β+2)V 1/(β+2)
T
, T

β
β+1
o
.

A.9
Proof of Theorem 4.2: Regret Lower Bound for Abruptly Rotting Rewards

First, we deal with the case when ST = 1 or ST = Θ(T). When ST = 1 (implying VT =
0), from the definition, the problem becomes stationary without rotting instances, which implies
E[Rπ(T)] = Ω(
√

T) from Theorem 3 in Wang et al. [24]. When ST = Θ(T), we consider that
rotting occurs for the first ST −1 rounds with ρt = 1 for all t ∈[ST −1]. Then it is always
beneficial to pull new arms every round until ST −1 rounds because the mean rewards of rotted
arms are below 0 and those of non-rotted arms lie in [0, 1]. This means that any ideal policy samples
a new arm and pulls it every round until ST −1. Then for any randomly sampled a ∈A, we
have E[µ1(a)] ≥yP(µ1(a) ≥y) = yP(∆1(a) < 1 −y) for y ∈[0, 1]. Then with y = 1/2,
we have E[µ1(a)] ≥(1/2)P(∆1(a) < (1/2)) = Θ(1) from constant β > 0 and (1). Then with
E[µ1(a)] ≤1, we have E[µ1(a)] = Θ(1). Since E[µ1(a)] = Θ(1) for any randomly sampled a ∈A,
any ideal policy has E[Rπ(T)] ≥PST
i=1 E[µ1(a)] = Ω(ST ) = Ω(T), which concludes the proof for
ST = Θ(T).

40


---Page Break---
Now we consider the case of ST = o(T) and ST ≥2. We initially provide a regret bound
with respect to the cumulative rotting amount of ¯VT . We first think of a policy π that randomly
samples a new arm and pulls it once every round. Then for any randomly sampled a ∈A, we have
E[µ1(a)] = Θ(1). Then from constant β > 0, E[Rπ(T)] = Ω(T). Then there always exists ρt’s
satisfying PT −1
t=1 ρt = T, which implies E[Rπ(T)] = Ω(T) = Ω( ¯VT ).

Now we think of any nontrivial algorithm which must pull an arm a at least twice. Let t′ and
t′′ be the rounds when the policy pulls arm a (t′ < t′′). If we consider ρt′ > 0 and ρt = 0 for
t ∈[T −1]/{t′} in which ρt′ = PT −1
t=1 ρt and 1 + PT −1
t=1 ρt1(ρt ̸= 0) ≤ST , then such policy has
Rπ(T) = Ω(PT −1
t=1 ρt) regret bound because, at time t′′, it pulls the arm a rotted by ρt′. Therefore,
for any policy π, there always exist a rotting rate adversary satisfying the following expected regret
bound of

E[Rπ(T)] = Ω( ¯VT ).
(56)

Next, for the regret bound with respect to ST , we follow the proof steps in Theorem 4.1. However,
the regret bound of ST does not depend on the magnitude of rotting rates but on the number of
rotting instances. To address this, we need to design a new worst-case in which an adversary makes
near-optimal arms rotted to be sub-optimal arms abruptly rather than gradually. We first categorize
arms as either bad or good according to their initial mean reward values. For the categorization,
we utilize two thresholds in the proof as follows. Consider 0 < γ < c < 1 for γ, which will be
specified, and a constant c. Then the value of 1 −γ represents a threshold value for identifying good
arms, while 1 −c serves as the threshold for identifying bad arms. We refer to arms a satisfying
µ1(a) ≤1 −c as ‘bad’ arms and arms a satisfying µ1(a) > 1 −γ as ‘good’ arms. We also consider
a sequence of arms in A denoted by ¯a1, ¯a2, . . . . Given a policy π, without loss of generality, we can
assume that π selects arms according to the order of ¯a1, ¯a2, . . . .

Let Km denote the number of bad arms a that satisfy µ1(a) ≤1 −c before sampling m-th good
arm, which satisfies µ1(a) > 1 −γ, in the sequence of arms ¯a1, ¯a2, . . . . Let NT be the number of
sampled good arms a such that µ1(a) > 1 −γ until T.

We can decompose Rπ(T) into two parts as follows:

Rπ(T) = Rπ(T)1(NT < m) + Rπ(T)1(NT ≥m).
(57)

We set m = ST and γ = (ST /T)1/(β+1) with ST = o(T). For getting a lower bound for the first
term in (57), Rπ(T)1(NT < m), we consider the fact that the minimal regret is obtained from the
situation where there are m −1 arms whose mean rewards are 1. In such a case, the optimal policy
must sample the best m −1 arms until their mean rewards become equal to or below the threshold
value of 1 −γ (step 1) and then samples the best arm at each time for the remaining time steps (step
2). In step 1, when the optimal policy pulls an optimal arm, we can think of the case when the mean
reward of the arm is abruptly rotted to the value of 1 −γ. This implies that the required number of
rounds for step 1 is m−1. The regret from step 2 is R = Ω((T −m+1)γ) = Ω(S1/(β+1)
T
T β/(β+1)),
in which the optimal policy pulls arms which mean rewards are below or equal to 1 −γ for the
remaining time after step 1. Therefore, we have

Rπ(T)1(NT < m) = Ω(R1(NT < m)) = Ω(S1/(β+1)
T
T β/(β+1)1(NT < m)).
(58)

For getting the above, we note that there always exists ρt’s satisfying PT −1
t=1 ρt = O(γm) = o(T),
which implies PT −1
t=1 ρt ≤T. Such ρt’s can be considered for the below. For getting a lower bound
of the second term in (57), Rπ(T)1(NT ≥m), we use the minimum number of sampled arms a that
satisfy µ1(a) ≤1 −c. When NT ≥m and Km ≥κ, the policy samples at least κ number of distinct
arms a satisfying µ1(a) ≤1 −c until T. Therefore, we have

Rπ(T)1(NT ≥m) ≥cκ1(NT ≥m, Km ≥κ).
(59)

We set γ = P(µ1(a) > 1 −γ)/p(µ1(a) /∈(1 −c, 1 −γ]). Then we have γ = Θ(γβ) from (1) with
constant β > 0. By setting κ = m/γ −m −m/(γ√m + 3), with ST = o(T) and constant β > 0,
we have

κ = Θ(S1/(β+1)
T
T β/(β+1)).
(60)

41


---Page Break---
Then from (58), (59), and (60), we have

E[Rπ(T)] = Ω(S1/(β+1)
T
T β/(β+1)P(NT < m) + S1/(β+1)
T
T β/(β+1)P(NT ≥m, Km ≥κ))

≥Ω(S1/(β+1)
T
T β/(β+1)P(Km ≥κ)).
(61)

Next we provide a lower bound for P(Km ≥κ). Observe that Km follows a negative binomial
distribution with m successes and the success probability P(µ1(a) > 1 −γ)/P(µ1(a) /∈(1 −c, 1 −
γ]) = γ, in which the success probability is the probability of sampling a good arm given that the
arm is either a good or bad arm. We recall Lemma A.26 for a concentration inequality for Km in the
following.
Lemma A.27. For any 1/2 + γ/m < α < 1,

P(Km ≥αm(1/γ) −m) ≥1 −exp(−(1/3)(1 −1/α)2(αm −γ)).

From Lemma A.27 with α = 1 −1/√m + 3 and large enough T, we have

P(Km ≥κ) ≥1 −exp

 

−1

3(m −
m
√m + 3 −γ)

1
√m + 3 −1

2!

≥1 −exp

 

−1

6(m −
m
√m + 3)

1
√m + 3 −1

2!

= 1 −exp

−1

6
m
m + 3

√m + 3
√m + 3 −1



≥1 −exp(−1/24),
(62)

where the last inequality comes from m/(m+3) = (ST )/(ST +3) ≥1/4 and √m + 3/(√m + 3−
1) ≥1. Therefore, from (61) and (62), we have

E[Rπ(T)] = Ω(S1/(β+1)
T
T β/(β+1)).
(63)

Overall
from
(56)
and
(63),
for
any
π,
there
exist
ρt’s
such
that
E[Rπ(T)]
=
Ω(max{S1/(β+1)
T
T β/(β+1), VT }).

A.10
Additional Experiments

0
1
2
3
4
5
T
1e6

0.0

0.5

1.0

1.5

2.0

E[R (T)]

1e6

Regret upper bound (Alg2)
Regret upper bound (Alg1)
Algorithm 2
Algorithm 1

Figure 5: Performance of our algorithms with theoretical bounds.

We compare the experiment results of our algorithm with the theoretical results in Corollary 3.4 and
Theorem A.15. We follow the same setting in Fig 5 (a) so that the theoretical bounds are T 2/3 log(T)
and T 2/3 log(T) + T 3/4 log(T) for Algorithms 1 and 2, respectively, including logarithmic factors
with constant terms set to 1. In Figure 5, the experiment results outperform the theoretical bounds,
which is expected since the theoretical bounds represent worst-case scenarios, while the experiments
reflect a specific instance.

42


---Page Break---
A.11
Lemmas for Concentration Inequalities

Lemma A.28 (Theorem 6.2.35 in Tsun [22]). Let X1, . . . , Xn be identical independent Bernoulli
random variables. Then, for 0 < ν < 1, we have

P

 n
X

i=1
Xi ≥(1 + ν)E

" n
X

i=1
Xi

#!

≤exp

−ν2E[Pn
i=1 Xi]
3


.

Lemma A.29 (Corollary 1.7 in Rigollet and Hütter [18]). Let X1, . . . , Xn be independent random
variables with σ-sub-Gaussian distributions. Then, for any a = (a1, . . . , an)⊤∈Rn and t ≥0, we
have

P

 n
X

i=1
aiXi > t

!

≤exp

−
t2

2σ2∥a∥2
2


and P

 n
X

i=1
aiXi < −t

!

≤exp

−
t2

2σ2∥a∥2
2


.

43


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

Justification: In the limitations section (Section A.1), we discuss an avenue for future work
regarding regret lower bounds.

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

44


---Page Break---
Answer: [Yes]
Justification: We provide assumptions in Section 2 and, for the case of unknown parameters,
in Appendix A.6. In addition, proof sketches of some of our main theorems (Theorems 3.1,
3.3) are included in the main text, with complete proofs provided for all theorems in the
appendix.
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
Justification: In Section 5, we provide all the information necessary for conducting the
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

45


---Page Break---
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: There is a link to our code in Section 5.

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
Section 5.

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

Justification: In our experimental results in Section 5, we include error bars of standard
deviation along with expectation values.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

46


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

47


---Page Break---
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

48


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

49


---Page Break---
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

50


---Page Break---
