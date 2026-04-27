e-COP : Episodic Constrained Optimization of Policies

Akhil Agnihotri
University of Southern California
agnihotri.akhil@gmail.com

Rahul Jain
Google DeepMind and USC
rahulajain@google.com

Deepak Ramachandran
Google DeepMind
ramachandrand@google.com

Sahil Singla
Google DeepMind
sasingla@google.com

Abstract

In this paper, we present the e-COP algorithm, the first policy optimization algorithm
for constrained Reinforcement Learning (RL) in episodic (finite horizon) settings. Such
formulations are applicable when there are separate sets of optimization criteria and
constraints on a system’s behavior. We approach this problem by first establishing a pol-
icy difference lemma for the episodic setting, which provides the theoretical foundation
for the algorithm. Then, we propose to combine a set of established and novel solution
ideas to yield the e-COP algorithm that is easy to implement and numerically stable,
and provide a theoretical guarantee on optimality under certain scaling assumptions.
Through extensive empirical analysis using benchmarks in the Safety Gym suite, we
show that our algorithm has similar or better performance than SoTA (non-episodic)
algorithms adapted for the episodic setting. The scalability of the algorithm opens
the door to its application in safety-constrained Reinforcement Learning from Human
Feedback for Large Language or Diffusion Models.

1
Introduction

RL problems may be formulated in order to satisfy multiple simultaneous objectives. These can include
performance objectives that we want to maximize, and physical, operational or other objectives that
we wish to constrain rather than maximize. For example, in robotics, we often want to optimize a task
completion objective while obeying physical safety constraints. Similarly, in generative AI models, we
want to optimize for human preferences while ensuring that the output generations remain safe (expressed
perhaps as a threshold on an automatic safety score that penalizes violent or other undesirable content).
Scalable policy optimization algorithms such as TRPO [31], PPO [33], etc have been central to the
achievements of RL over the last decade [34, 38, 7]. In particular, these have found utility in generative
models, e.g., in the training of Large Language Models (LLMs) to be aligned to human preferences through
the RL with Human Feedback (RLHF) paradigm [2]. But these algorithms are designed primarily for the
unconstrained infinite-horizon discounted setting: Their use for constrained problems via optimization of
the Lagrangian often gives unsatisfactory constraint satisfaction results. This has prompted development
of a number of constrained policy optimization algorithms for the infinite-horizon discounted setting
[3, 39, 43, 41, 19, 6, 5], and for the average setting [4].

However, many RL problems are more accurately formulated as episodic, i.e., having a finite time horizon.
For instance, in image diffusion models [10, 21], the denoising sequence is really a finite step trajectory
optimization problem, better suited to be solved via RL algorithms for the episodic setting. When existing
algorithms for infinite horizon discounted setting are used for such problem, they exhibit sub-optimal
performance or fail to satisfy task-specific constraints by prioritizing short-term constraint satisfaction
over episodic goals [11, 28, 17]. Furthermore, the episodic setting allows for objective functions to be
time-dependent which the infinite-horizon formulations do not. Even when the objective functions are

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
time-invariant, there is a key difference: for non-episodic settings, a stationary policy that is optimal
exists whereas for episodic settings, the optimal policy is always non-stationary and time-dependent.
This necessitates development of policy optimization algorithms specifically for the episodic constrained
setting. We note that such policy optimization algorithms do not exist even for the unconstrained episodic
setting.

In this paper, we introduce e-COP, a policy optimization algorithm for episodic constrained RL problems.
Inspired by PPO, it uses deep learning-based function approximation, a KL divergence-based proximal
trust region and gradient clipping, along with several novel ideas specifically for the finite horizon
episodic and constrained setting. The algorithm is based on a technical lemma (Lemma 3.1) on the
difference between two policies, which leads to a new loss function for the episodic setting. We also
introduce other ideas that obviate the need for matrix inversion thus improving scalability and numerical
stability. The resulting algorithm improves performance over state-of-the-art baselines (after adapting
them for the episodic setting). In sum, the e-COP algorithm has the following advantages: (i) Solution
equivalence: We show that the solution set of our e-COP loss function is same as that of the original
CMDP problem, leading to precise cost control during training and avoidance of approximation errors (see
Theorem 3.3); (ii) Stable convergence: The e-COP algorithm converges tightly to safe optimal policies
without the oscillatory behavior seen in other algorithms like PDO [13] and FOCOPS [43]; (iii) Easy
integration: e-COP follows the skeleton structure of PPO-style algorithms using clipping instead of steady
state distribution approximation, and hence can be easily integrated into existing RL pipelines; and (iv)
Empiricial performance: the e-COP algorithm demonstrates superior performance on several benchmark
problems for constrained optimal control compared to several state-of-the-art baselines.

Our Contributions and Novelty. We introduce the first policy optimization algorithm for episodic RL
problems, both with and without constraints (no constraints is a special case). While some of the policy
optimization algorithms can be adapted for the constrained setting via a Lagrangian formulation, as we
show, they don’t work so well empirically in terms of constraint satisfaction and optimality. The algorithm
is based on a policy difference (technical) lemma, which is novel. We have gotten rid of Hessian matrix
inversion, a common feature of policy optimization algorithms (see, for example, PPO [33], CPO [3],
P3O [41], etc.) and replaced it with a quadratic penalty term which also improves numerical stability
near the edge of the constraint set boundary - a problem unique to constrained RL problems. We provide
an instance-dependent hyperparameter tuning routine that generalizes to various testing scenarios. And
finally, our extensive empirical results against an extensive suite of baseline algorithms (e.g., adapted PPO
[33], FOCOPS [43], CPO [3], PCPO [39], and P3O [41]) show that e-COP performs the best or near-best
on a range of Safety Gym [12] benchmark problems.

Related Work. A broad view of planning and model-free RL techniques for Constrained MDPs is provided
in [27] and [18]. The development of SOTA policy optimization started with the TRPO algorithm [31],
which used a trust region to update the current policy, and was further improved in PPO by use of proximal
ideas [33]. This led to works like CPO [3], RCPO [36], and PCPO [39] for constrained RL problems in
the infinite-horizon discounted setting. ACPO [4] extended CPO to the infinite-horizon average setting.
These methods typically require inversion of a computationally-expensive Fischer information matrix
at each update step, thus limiting scalability. Lagrangian-based algorithms [29, 30] showed that you
could incorporate constraints but constrained satisfaction remained a concern. Algorithms like PDO [13]
and RCPO [36] also use Lagrangian duality but to solve risk-constrained RL problems and suffer from
computational overhead. Some other notable algorithms include IPO [26], P3O [41], APPO [15], etc. that
use penalty terms, and hence do not suffer from computational overhead but have other drawbacks. For
example, IPO assumes feasible intermediate iterations, which cannot be fulfilled in practice, P3O requires
arbritarily large penalty factors for feasibility which can lead to significant estimation errors. We note that
all the above algorithms are for the infinite-horizon discounted (non-episodic) setting (except ACPO [4],
which is for the average setting). We are not aware of any policy optimization algorithm for the episodic
RL problem, with or without constraints.

2
Preliminaries

An episodic, or fixed horizon Markov decision process (MDP) is a tuple, M :“ pS, A, r, P, µ, Hq,
where S is the set of states, A is the set of actions, r : S ˆ A ˆ S Ñ R is the reward function,
P : S ˆ A ˆ S Ñ r0, 1s is the transition probability function such that Pps1|s, aq is the probability of
transitioning to state s1 from state s by taking action a, µ : S Ñ r0, 1s is the initial state distribution, and
H is the time horizon for each episode (characterized by a terminal state sH).

2


---Page Break---
A policy π : S Ñ ∆pAq is a mapping from states to probability distributions over the actions, with πpa|sq
denoting the probability of selecting action a in state s, and ∆pAq is the probability simplex over the
action space A. However, due to the temporal nature of episodic RL, the optimal policies are generally
not stationary, and we index the policy at time h by πh, and denote π1:H “ pπhqH
h“1. Then, the total
undiscounted reward objective within an episode is defined as

Jpπ1:Hq :“
E
τ„π1:H

« H
ÿ

h“1
rpsh, ah, sh`1q

ff

where τ refers to the sample trajectory ps1, a1, s2, a2, . . . , sHq generated when following a policy se-
quence, i.e., ah „ πhp¨|shq, sh`1 „ Pp¨|sh, ahq, and s1 „ µ.

Let Rh:Hpτq denote the total reward of a trajectory τ starting from time h until episode terminal time
H generated by following the policy sequence πh:H. We also define the state-value function of a
state s at step h as V π
h psq :“
E
τ„πrRh:Hpτq | sh “ ss and the action-value function as Qπ
h ps, aq :“

E
τ„πrRh:Hpτq | sh “ s, ah “ as. The advantage function is Aπ
h ps, aq :“ Qπ
h ps, aq ´ V π
h psq. We also

define Pπ
h ps | s1q “ ř
aPA Pπ
h ps, a | s1q, where the term Pπ
h ps, a | s1q is the probability of reaching ps, aq
at time step h following π and starting from s1.

Constrained MDPs. A constrained Markov decision process (CMDP) is an MDP augmented with
constraints that restrict the set of allowable policies for that MDP. Specifically, we have m cost functions,
C1, ¨ ¨ ¨ , Cm (with each function Ci : S ˆ A ˆ S Ñ R mapping transition tuples to costs, similar to the
reward function), and bounds d1, ¨ ¨ ¨ , dm. And similar to the value function for the reward objective, we
define the expected total cost objective for each cost function Ci (called cost value for the constraint) as

JCipπ1:Hq :“
E
τ„π1:H

« H
ÿ

h“1
Cipsh, ah, sh`1q

ff

.

The goal then, in each episode, is to find a policy sequence π‹
1:H such that

Jpπ‹
1:Hq :“
max
π1:HPΠC Jpπ1:Hq, where ΠC :“ tπ1:H P Π : JCipπ1:Hq ď di, @ i P r1 : msu
(1)

is the set of feasible policies for a CMDP for some given class of policies Π. Lastly, analogous to V π
h ,
Qπ
h , and Aπ
h , we can also define quantities for the cost functions Cip¨q by replacing, and denote them by
V π
Ci,h, Qπ
Ci,h, and Aπ
Ci,h. Proofs of theorems and statements, if not given, are available in Appendix A.

Notation. rNs denotes t1, . . . , Nu for some N P N. πh refers to the policy at time step h within an
episode. Denote πs:t :“ pπs, πs`1, . . . , πtq for some s ď t with s, t P rHs. We shall only write πk,h
when it is necessary to differentiate policies from different episodes but at the same time h. It then
naturally follows to define πk,s:t to be the sequence πs:t in episode k. We will denote πk ” πk,1:H, and
where not needed drop the index for the episode so that π ” πk.

3
Episodic Constrained Optimization of Policies (e-COP)

In this section, we propose a constrained policy optimization algorithm for episodic MDPs. Policy
optimization algorithms for MDPs have proven remarkably effective given their ability to computationally
scale up to high dimensional continuous state and action spaces [31–33]. Such algorithms have also
been proposed for infinite-horizon constrained MDPs with discounted criterion [1] as well as the average
criterion [4] but not for the finite horizon (or as it is often called, the episodic) setting.

We note that finite horizon is not simply a special case of infinite-horizon discounted setting since the
reward/cost functions in the former can be time-varying while the latter only allows for time-invariant
objectives. Furthermore, even with time-invariant objectives, the optimal policy is time-dependent, while
for the latter setting there an optimal policy that is stationary exists.

A Policy Difference Lemma for Episodic MDPs. Most policy optimization RL algorithms are based on
a value or policy difference technical lemma [23]. Unfortunately, the policy difference lemmas that have
been derived previously for the infinite-horizon discounted [31] and average case [4] are not applicable
here and hence, we derive a new policy difference lemma for the episodic setting.

3


---Page Break---
Lemma 3.1. For an episode of length H and two policies, π and π1, the difference in their performance
assuming identical initial state distribution µ (i.e., s1 „ µ) is given by

Jpπq ´ Jpπ1q “

H
ÿ

h“1
E
sh,ah„Pπ
h p¨,¨ | sq
s1„µ

“
Aπ1
h psh, ahq
‰
.
(2)

The proof can be found in Appendix A.1. A key difference to note between the above and similar
results for infinite-horizon settings [31, 4] is that considering stationary policies (and hence corresponding
occupation measures) is not enough for the episodic setting since, in general, such a policy may be far
from optimal. This explains why Lemma 3.1 looks so different (e.g., see (2) in [31], and Lemma 3.1 in
[4]). Indeed, the lemma above indicates that policy updates do not have to recurse backwards from the
terminal time as dynamic programming algorithms do for episodic settings, which is somewhat surprising.

A Constrained Policy Optimization Algorithm for Episodic MDPs. Iterative policy optimization
algorithms achieve state of the art performance [33, 36, 39] on RL problems. Most such algorithms
maximize the advantage function based on a suitable policy difference lemma, solving an unconstrained
RL problem. Some additionally ensure satisfaction of infinite horizon expectation constraints [3, 4].
However, given that our policy lemma for the episodic setting (Lemma 3.1) is significantly different,
we need to re-design the algorithm based on it. A first attempt is presented as Algorithm 1, where each
iteration corresponds to an update with a full horizon H episode.

Algorithm 1 Iterative Policy Optimization for Constrained Episodic (IPOCE) RL

1: Input: Initial policy π0, number of episodes K, episode horizon H.
2: for k “ 1, 2, . . . , K do
3:
Run πk´1 to collect trajectories τ.
4:
Evaluate Aπk´1
h
and Aπk´1
Ci,h for h P rHs from τ.
5:
for t “ H, H ´ 1, . . . , 1 do

6:
π‹
k,t “ arg min
πk,t

H
ÿ

h“t
E
s„ρπk,h
a„πk,h
r´Aπk´1
h
ps, aqs
s.t.
JCipπk´1q `

H
ÿ

h“t
E
s„ρπk,h
a„πk,h

“
Aπk´1
Ci,h ps, aq
‰
ď di, @i
(3)

7:
end for
8:
Set πk Ð
´
π‹
k,1, π‹
k,2, . . . , π‹
k,H
¯
.

9: end for

The iterative constrained policy optimization algorithm introduced above uses the current iterate of the
policy πk to collect a trajectory τ, and use them to evaluate Aπk´1
h
and Aπk´1
Ci,h for h P rHs. At the end of
the episode, we solve H optimization problems (one for each h P rHs) that result in a new sequence of
policies π. As is natural in episodic problems, we do backward iteration in time, i.e., solve the problem in
step (6) at h “ H, and then go backwards towards h “ 1.

Note that the expectation of advantage functions in equation (3) is with respect to the policy π (the
optimization variable) and its corresponding time-dependent state occupation distribution ρπh. In the
infinite-horizon settings, the expectation is with respect to the steady state stationary distribution, but that
does not exist in the episodic setting.

Using current policy for action selection. Algorithm 1 represents an exact principled solution tothe
constrained episodic MDP, but the intractable optimization performed in (3) makes it impractical (as in the
case of infinite horizon policy optimization algorithms [31, 3, 39]). We proceed to introduce a sequence
of ideas that make the algorithm practical (e.g., by avoiding computationally expensive Hessian matrix
inversion for use with trust-region methods [33, 14, 41]). However, getting rid of trust regions leads to
large updates on policies, but PPO [33] and P3O [41] successfully overcome this problem by clipping the
advantage function and adding a ReLU-like penalty term to the minimization objective. Motivated by this,
we rewrite the optimization problem in (3) as follows by parameterizing the policy πk,t in episode k and
time step t by θk,t:

πk,t “ arg min
πk,t

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,i max
"
0,

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
Ci,h ps, aq
‰
` JCipπk´1q ´ di

*
,

(4)

4


---Page Break---
where ρpθhq “
πθk,h
πθk´1,h is the importance sampling ratio, λt,i is a penalty factor for constraint Ci, and

πk,θh ” πk,h ” θk,h. Note that the ReLU-like penalty term above is different from the traditional
first-order and second-order gradient approximations that are employed in trust-region methods [3, 39].
In essence, the penalty is applied when the agent breaches the associated constraint, while the objective
remains consistent with standard policy optimization when all constraints are satisfied.

Introducing quadratic damping penalty.
It has been noted in such iterative policy optimization
algorithms that the behaviour of the learning agent when it nears the constraint threshold is quite volatile
during training [3, 39, 42]. This is because the penalty term is active only when the constraints are violated
which results in sharp behavior change for the agent. To alleviate this problem, we introduce an additional
quadratic damping term to the objective above, which provides stable cost control to compliment the
lagged Lagrangian multipliers. This has proved effective in physics-based control applications [16, 25, 20]
resulting in improved convergence since the damping term provides stability, while keeping the solution
set the same as for the original Problem (3) and the adapted Problem (4) (as we prove later).

For brevity, we denote the constraint term in Problem (4) as

ΨCi,tpπk´1, πkq :“ řH
h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
Ci,h ps, aq
‰
` JCipπk´1q ´ di.

Now introduce a slack variable xt,i ě 0 for each constraint to convert the inequality constraint
(ΨCi,tp¨, ¨q ď 0) to equality by letting

wt,ipπkq :“ ΨCi,tpπk´1, πkq ` xt,i “ 0.

With this notation, we restate Problem (4) as:

π‹
k,t “ minπk,t Ltpπk, λq :“ řH
h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
` řm
i λt,i maxt0, ΨCi,tpπk´1, πkqu.

Now we introduce the quadratic damping term and the intermediate loss function then takes the form,

Ltpπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x Ltpπk, λ, x, βq ,

(5)

where β is the damping factor, λt “ pλt,iqm
i“1, and xt “ pxt,iqm
i“1. We can then construct a primal-dual
solution to the max-min optimization problem. The need for a slack variable x can be obviated by
setting the partial derivative of Ltp¨q with respect to x equal to 0. This leads to a ReLU-like solution:
x‹
t,i “ max
`
0, ´ΨCi,tpπk´1, πkq ´ λt,i

β
˘
. The intermediate problem then takes the form as below.

Proposition 3.2. The inner optimization problem in (5) with respect to x is a convex quadratic program
with non-negative constraints, which can be solved to yield the following intermediate problem:

pπ‹
k,t, λ‹
tq “ max
λě0 min
πk,t Ltpπk, λ, βq,
where

Ltpπk, λ, βq “

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
` β

2

m
ÿ

i

ˆ
max
"
0, ΨCi,tpπk´1, πkq ` λt,i

β

*2
´ λ2
t,i
β2

˙
.
(6)

The proof can be found in Appendix A.1.
One can see that the cost penalty is active when
ΨCi,tpπk´1, πkq ě ´ λt,i

β
rather than when ΨCi,tpπk´1, πkq ě 0. This allows the agent to act in a
constrained manner even before the constraint is violated. Further, as we show next, the introduction of
the damping factor and the RELU-like penalty does not change the solution of the problem (under some
suitable assumptions):

Theorem 3.3. Let π(3)‹ be a solution to Problem (3), and let
`
π(6)‹, λ(6)‹˘
be a solution to Problem (6).
Then, for sufficiently large β ą ¯β and λt,i ą ¯λ @ i, π(3)‹ is a solution to Problem (6), and π(6)‹ is a
solution to Problem (3).

We refer the reader to Appendix A.1 for the proof. This theorem implies that we can search for the optimal
feasible policies of the CMDP Problem (1) by iteratively solving Problem (6). Next, we make some
further modifications to Problem (6) that give us our final tractable algorithm.

5


---Page Break---
Removing Lagrange multiplier dependency. Problem (6) requires a primal-dual algorithm that will
iteratively solve over the policies and the dual variable λ. But from the Lagrangian, we can actually take a
derivative with respect to λ, and then solve for it, which yields the following update rule for it:

λpkq
t,i “ max
`
0, λpk´1q
t,i
` βΨCi,tpπk´1, πk´1q
˘
.
(7)

This update rule simplifies the optimization problem and updates the Lagrange multipliers in the kth
episode based on the constraint violation in the pk ´ 1qth episode.

Clipping the advantage functions. Solving the optimization problem presented in equation (6) is
intractable since we do not know ρπ beforehand. Hence, we replace ρπ by the empirical distribution
observed with the policy of the previous episode, πk´1, i.e., ρπk,h « ρπk´1,h @ h. Similar to [33] for
PPO, we also use clipped surrogate objective functions for both the reward and cost advantage functions.
Thus, the final problem combining equation (4) and equation (6) can be constructed as follows.

If we let

Ltpθq “

H
ÿ

h“t
E
s„ρπk´1,h
a„πk´1,h

“
´ min
␣
ρpθhqAπk´1
h
ps, aq, clippρpθhq, 1 ´ ϵ, 1 ` ϵqAπk´1
h
ps, aq
(‰
and,

LCi,tpθq “

H
ÿ

h“t
E
s„ρπk´1,h
a„πk´1,h

“
´ min
␣
ρpθhqAπk´1
Ci,h ps, aq, clippρpθhq, 1 ´ ϵ, 1 ` ϵqAπk´1
Ci,h ps, aq
(‰

then, the final loss function r
Ltp¨q of the final problem is:

π‹
k,t “ arg min
πk,t
r
Ltpπθ, λ, βq :“ arg min
πk,t
Ltpθq `

m
ÿ

i
λt,i max
␣
0, LCi,tpθq `
`
JCipπk´1q ´ di
˘(

` β

2

m
ÿ

i

ˆ
max
"
0, LCi,tpθq `
`
JCipπk´1q ´ di
˘
` λt,i

β

*2
´ λ2
t,i
β2

˙
(8)

Usually for experiments, Gaussian policies with means and variances predicted from neural networks
are used [31, 3, 33, 39]. We employ the same approach and since we work in the finite horizon setting,
the reward and constraint advantage functions can easily be calculated from any trajectory τ „ π. The
surrogate problem in equation (8) then includes the pessimistic bounds on Problem (6), which is unclipped.

Adaptive parameter selection. The value of β is required to be larger than the unknown ¯β according to
Theorem 3.3, but we also know that too large a β decays the performance (as seen in harmonic oscillator
kinetic energy formulations [25, 20, 40]). To manage this tradeoff, we provide an instance-dependent
adaptive way to adjust the damping factor as a hyperparameter. In each episode k, we update the damping
parameter whenever a secondary constraint cost value denoted by Cpπkq is larger than some threshold ck.
Using Proposition 3.2, we provide the following definitions.

Cpπkq :“

H
ÿ

t“1

m
ÿ

i
max
"
JCipπkq ´ di, ´
λpkq
t,i
β

*
and
ck :“
?m

β
¨ max
tPrHs

››λpkq
t
››
8

Algorithm 2 Episodic Constrained Optimization of Policies (e-COP)

1: Input: Initial policy θ0 :“ π0 :“ πθ0, critic networks V ϕ0 and V ψ0
Ci
@ i, penalty factor β, number
of episodes K, episode horizon H, learning rate α, penalty update factor p.
2: for k “ 1, 2, . . . , K do
3:
Collect a set of trajectories Dk´1 with policy πk´1 and update the critic network.
4:
Get updated λpkq using equation (7).
5:
for t “ H, H ´ 1, . . . , 1 do
6:
Update the policy θk,t Ð θk,t`1 ´ α∇θ r
Ltpθ, λpkq, βq using equation (8).
7:
end for
8:
if Cpθkq ě ck then
9:
β “ minpβmax, pβq
10:
end if
11: end for

6


---Page Break---
(a) Humanoid
(b) Circle
(c) Reach
(d) Grid
(e) Bottleneck
(f) Navigation

Figure 1: The Humanoid, Circle, Reach, Grid, Bottleneck, and Navigation tasks. See Appendix A.2.1 for details.

Hence, we increase β by a constant factor p ą 1 after every episode if Cpπkq ě ck until a stopping
condition is fulfilled, typically a constant βmax. This leads to constraint-satisfying iterations that are more
stable, and we show that it enables a fixed β to generalize well across various tasks. The initial β can
simply be selected by a quantified line-search to obtain a feasible β ą ¯β [3, 4].

We note that the final loss function in equation (8) is differentiable almost everywhere, so we could easily
solve it via any first-order optimizer [24]. The final practical algorithm, e-COP is given in Algorithm 2.

4
Experiments

We conducted extensive experimental evaluation on the relative empirical performance of the e-COP
algorithm to arrive at the following conclusions: (i) The e-COP algorithm performs better or nearly as
well as all baseline algorithms for infinite-horizon discounted safe RL tasks in maximizing episodic return
while satisfying given constraints. (ii) It is more robust to stochastic and complex environments [30], even
where previous methods struggle. (iii) It has stable behavior and more accurate cost control as compared
to other baselines near the constraint threshold due to the damping term.

Environments. For a comprehensive empirical evaluation, we selected eight scenarios from well-
known safe RL benchmark environments - Safe MuJoCo [43] and Safety Gym [30], as well as MuJoCo
environments. These include: Humanoid, PointCircle, AntCircle, PointReach, AntReach, Grid,
Bottleneck, and Navigation. See Figure 1 for an overview of the tasks and scenarios. Note that
Navigation is a multi-constraint task and for the Reach environment, we set the reward as a function of
the Euclidean distance between agent’s position and goal position. In addition, we make it impossible for
the agent to reach the goal before the end of the episode. For more information see Appendix A.2.1.

Baselines. We compare our e-COP algorithm with the following baseline algorithms: CPO [3], PCPO
[39], FOCOPS [43], PPO with Lagrangian relaxation [33, 35], and penalty-based P3O [41]. Since the
above state-of-the-art baseline algorithms are already well understood to outperform other algorithms
such as PDO [13], IPO [26], and CPPO-PID [35] in prior benchmarking studies, we do not compare
against them. Moreover, since PPO does not originally incorporate constraints, for fair comparison, we
introduce constraints using a Lagrangian relaxation (called PPO-L). In addition, for each algorithm, we
report its performance with the discount factor that achieves the best performance. See Appendix A.3.1
for more details.

Evaluation Details and Protocol. For the Circle task, we use a a point-mass with S Ď R9, A Ď R2 and
for the Reach task, an ant robot with S Ď R16, A Ď R8. The Grid task has S Ď R56, A Ď R4. We use
two hidden layer neural networks to represent Gaussian policies for the tasks. For Circle and Reach, size
is (32,32) for both layers, and for Grid and Navigation the layer sizes are (16,16) and (25,25). We set
the step size δ to 10´4, and for each task, we conduct 5 independent runs of K “ 500 episodes each of
horizon H “ 200. Since there are two objectives (rewards in the objective and costs in the constraints),
we show the plots which maximize the reward objective while satisfying the cost constraint.

4.1
Performance Analysis

Table 1 lists the numerical performance of all tested algorithms in seven single constraint scenarios, and
one multiple constraint scenario. We find that overall, the e-COP algorithm in most cases outperforms
(green) all other baseline algorithms in finding the optimal policy while satisfying the constraints, and in
other cases comes a close second (light green).

From Figure 2, we can see how the e-COP algorithm is able to improve the reward objective over the
baselines while having approximate constraint satisfaction. We also see that updates of e-COP are
faster and smoother than other baselines due to the added damping penalty, which ensures smoother

7


---Page Break---
Task
e-COP
FOCOPS [43]
PPO-L [30]
PCPO [39]
P3O [41]
CPO [3]
APPO [15]
IPO [26]

Humanoid R
1652.5 ˘ 13.4 1734.1 ˘ 27.4 1431.2 ˘ 25.2 1602.3 ˘ 10.1 1669.4 ˘ 13.7 1465.1 ˘ 55.3 1488.2 ˘ 29.3 1578.6 ˘ 25.2
C (20.0)
17.3 ˘ 0.3
19.7 ˘ 0.6
18.8 ˘ 1.5
16.3 ˘ 1.4
20.1 ˘ 3.3
18.5 ˘ 2.9
20.0 ˘ 1.3
19.1 ˘ 2.5

PointCircle R
110.5 ˘ 9.3
81.6 ˘ 8.4
57.2 ˘ 9.2
68.2 ˘ 9.1
89.1 ˘ 7.1
65.3 ˘ 5.3
91.2 ˘ 9.6
68.7 ˘ 15.2
C (10.0)
9.8 ˘ 0.9
10.0 ˘ 0.4
9.8 ˘ 0.5
9.9 ˘ 0.4
9.9 ˘ 0.3
9.5 ˘ 0.9
10.2 ˘ 0.6
9.3 ˘ 0.5

AntCircle R
198.6 ˘ 7.4
161.9 ˘ 22.2
134.4 ˘ 10.3
168.3 ˘ 13.3
182.6 ˘ 18.7
127.1 ˘ 12.1
155.5 ˘ 19.4
149.3 ˘ 33.6
C (10.0)
9.8 ˘ 0.6
9.9 ˘ 0.5
9.6 ˘ 1.6
9.5 ˘ 0.6
9.8 ˘ 0.2
10.1 ˘ 0.7
10.0 ˘ 0.5
9.5 ˘ 1.0

PointReach R
81.5 ˘ 10.2
65.1 ˘ 9.6
46.1 ˘ 14.8
73.2 ˘ 7.4
76.3 ˘ 6.4
89.2 ˘ 8.1
74.3 ˘ 6.7
49.1 ˘ 10.6
C (25.0)
24.5 ˘ 6.1
24.8 ˘ 7.6
25.1 ˘ 6.1
24.9 ˘ 5.6
26.3 ˘ 6.9
33.3 ˘ 10.7
26.3 ˘ 8.1
24.7 ˘ 11.3

AntReach R
70.8 ˘ 14.6
48.3 ˘ 5.6
54.2 ˘ 9.5
39.4 ˘ 5.3
73.6 ˘ 5.1
102.3 ˘ 7.1
61.5 ˘ 10.4
45.2 ˘ 13.3
C (25.0)
24.2 ˘ 8.4
25.1 ˘ 11.9
21.9 ˘ 10.7
27.9 ˘ 12.2
24.8 ˘ 7.3
35.1 ˘ 10.9
24.5 ˘ 6.4
24.9 ˘ 9.2
R
258.1 ˘ 33.1
215.4 ˘ 45.6
276.3 ˘ 57.9
226.5 ˘ 29.2
201.5 ˘ 39.2
178.1 ˘ 23.8
184.4 ˘ 21.5
229.4 ˘ 32.8
Grid
C (75.0)
71.3 ˘ 26.9
76.6 ˘ 29.8
71.8 ˘ 25.1
72.6 ˘ 16.5
79.3 ˘ 19.3
69.3 ˘ 19.8
79.5 ˘ 35.8
74.2 ˘ 24.6

Bottleneck R
345.1 ˘ 52.6
251.3 ˘ 59.1
298.3 ˘ 71.2
264.2 ˘ 43.8
291.1 ˘ 26.7
388.1 ˘ 36.6
220.1 ˘ 30.1
279.3 ˘ 43.8
C (50.0)
49.7 ˘ 15.1
46.6 ˘ 19.8
41.4 ˘ 17.6
49.8 ˘ 10.5
45.3 ˘ 8.2
54.3 ˘ 13.5
47.4 ˘ 12.3
48.2 ˘ 14.6

Navigation
R
217.6 ˘ 11.5
175.1 ˘ 3.7
153.5 ˘ 25.2
135.7 ˘ 19.2
164.1 ˘ 12.8
C1 (10.0)
9.6 ˘ 1.5
n/a
9.9 ˘ 1.9
n/a
9.9 ˘ 1.7
n/a
9.9 ˘ 2.1
10.0 ˘ 0.5
C2 (25.0)
23.7 ˘ 4.1
22.3 ˘ 2.1
24.5 ˘ 4.1
23.9 ˘ 3.8
24.6 ˘ 3.1

Table 1: Mean performance with normal 95% confidence interval over 5 independent runs on some tasks.

Episodic Rewards:

100
200
300
400
500

2000

1600

1200

800

400

0

100
200
300
400
500

250

200

150

100

50

0

0
100
200
300
400
500

100

80

60

40

20

0

0
100
200
300
400
500

400

320

240

160

80

0

Constraint Costs:

100
200
300
400
500

25

20

15

10

5

0

(a) Humanoid

0
100
200
300
400
500

20

16

12

8

4

0

(b) Ant Circle

0
100
200
300
400
500

40

32

24

16

8

0

(c) Point Reach

0
100
200
300
400
500

90

72

54

36

18

0

(d) Bottleneck

Figure 2: The cumulative episodic reward and constraint cost function values vs episode learning curves
for some algorithm-task pairs. Solid lines in each figure are the means, while the shaded area represents 1
standard deviation, all over 5 runs. The dashed line in constraint plots is the constraint threshold.

convergence with only a few constraint-violating behaviors during training. In particular, e-COP is the
only algorithm that best learns almost-constraint-satisfying maximum reward policies across all tasks: in
simple Humanoid and Circle environments, e-COP is able to almost exactly track the cost constraint
values to within the given threshold. However, for the high dimensional Grid environment we have more
constraint violations due to complexity of the policy behavior, leading to higher variance in episodic
rewards as compared to other environments. Regardless, overall in these environments, e-COP still
outperforms all other baselines with the least episodic constraint violation. For the multiple constraint
Navigation environment, see Figure 3.

4.2
Secondary Evaluation

In this section, we take a deeper dive into the empirical performance of e-COP. We discuss its dependence
on various factors, and try to verify its merits.

Generalizability. From the discussion above, it’s clear that e-COP demonstrates accurate safety satis-
faction across tasks of varying difficulty levels. From Figure 4, we further see that e-COP satisfies the

8


---Page Break---
0
100
200
300
400
500

300

250

200

150

100

50

0

(a) Rewards

0
100
200
300
400
500

18

15

12

9

6

3

0

(b) Cost1 (hazards)

0
100
200
300
400
500

42

35

28

21

14

7

0

(c) Cost2 (pillars)

Figure 3: Navigation environment with multiple constraints: Episodic Rewards (left), Cost1 (center, for hazards)
and Cost2 (right, for pillars) of e-COP . The dashed line in the cost plots is the cost threshold (10 for Cost1 and 25 for
Cost2). C1/C2 constrained means only taking Cost1/Cost2 into the e-COP loss function and ignoring the other one.

constraints in all cases and precisely converges to the specified cost limit. Furthermore, the fluctuation
observed in the baseline Lagrangian-based algorithms is shown not to be tied to a specific cost limit.

We also conducted a set of experiments wherein we study how e-COP effectively adapts to different cost
thresholds. For this, we use the hyperparameters of a pre-trained e-COP agent, which is trained with a
particular cost threshold in an environment, for learning on different cost thresholds within the same
environment. Figure 6 in Appendix A.3.4 illustrates the training curves of these pre-trained agents, and
we see that while e-COP can generalize well across different cost thresholds, other baseline algorithms
may require further tuning to accommodate different constraint thresholds.

0
100
200
300
400
500

300

250

200

150

100

50

0

(a) Ant Circle Rewards

0
100
200
300
400
500

18

15

12

9

6

3

0

(b) Ant Circle Costs

0
100
200
300
400
500

130

125

100

75

50

25

0

(c) Point Reach Rewards

0
100
200
300
400
500

60

50

40

30

20

10

0

(d) Point Reach Costs

Figure 4: Cumulative episodic rewards and costs of baselines in two environments with two constraint thresholds.

Task
e-COP
P3O [41]
β “ 5, fixed
β “ 5, adaptive
β “ 10, fixed
β “ 10, adaptive

PointCircle
R
150.5 ˘ 11.1
168.6 ˘ 14.3
145.2 ˘ 12.2
165.3 ˘ 11.4
162.4 ˘ 14.7
C (20.0)
17.3 ˘ 1.3
19.7 ˘ 0.6
18.8 ˘ 1.5
18.3 ˘ 1.4
19.1 ˘ 3.3

AntReach
R
48.2 ˘ 3.5
58.6 ˘ 5.1
53.2 ˘ 5.3
65.2 ˘ 8.1
61.1 ˘ 5.6
C (20.0)
19.8 ˘ 5.9
20.0 ˘ 4.4
20.6 ˘ 4.5
19.2 ˘ 6.2
18.9 ˘ 7.3

Table 2: Performance of e-COP for different β settings on two tasks. Values are given with normal 95% confidence
interval over 5 independent runs.

Sensitivity. The effectiveness and performance of e-COP would not be justified if it was not robust to
the damping hyperparameter β, which varies across tasks depending on the values of Cp¨q and ch. Since
this damping penalty enables e-COP to have stable continuous cost control, we update it adaptively as
described in Algorithm 2. As seen in Table 2, damping penalty indeed stabilizes the training process and
helps in converging to an optimal safe policy.

5
Conclusion

In this paper, we have introduced an easy to implement, scalable policy optimization algorithm for episodic
RL problems with constraints due to safety or other considerations. It is based on a policy difference
lemma for the episodic setting, which surprisingly has quite a different form than the ones for infinite-
horizon discounted or average settings. This provides the theoretical foundation for the algorithm, which
is designed by incorporating several time-tested, practical as well as novel ideas. Policy optimization
algorithms for Constrained MDPs tend to be numerical unstable and non-scalable due to the need for
inverting the Fisher information matrix. We sidestep both of these issues by introducing a quadratic
damping penalty term that works remarkably well. The algorithm development is well supported by
theory, as well as with extensive empirical analysis on a range of Safety Gym and Safe MuJoco benchmark
environments against a suite of baseline algorithms adapted from their non-episodic roots.

9


---Page Break---
References

[1] J. Achiam. UC Berkeley CS 285 (Fall 2017), Advanced Policy Gradients, 2017. URL: http://rail.
eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf. Last
visited on 2020/05/24.
[2] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,
S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
[3] J. Achiam, D. Held, A. Tamar, and P. Abbeel. Constrained policy optimization. In Proceedings of
the 34th International Conference on Machine Learning-Volume 70, pages 22–31. JMLR. org, 2017.
[4] A. Agnihotri, R. Jain, and H. Luo. ACPO: A policy optimization algorithm for average MDPs with
constraints. In Proceedings of the 41st International Conference on Machine Learning, volume 235
of Proceedings of Machine Learning Research, pages 397–415. PMLR, 21–27 Jul 2024.
[5] A. Agnihotri, R. Jain, D. Ramachandran, and Z. Wen. Online bandit learning with offline preference
data. arXiv preprint arXiv:2406.09574, 2024.
[6] A. Agnihotri, P. Saraf, and K. R. Bapnad. A convolutional neural network approach towards self-
driving cars. In 2019 IEEE 16th India Council International Conference (INDICON), pages 1–4,
2019.
[7] I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert,
G. Powell, R. Ribas, et al. Solving rubik’s cube with a robot hand. arXiv preprint arXiv:1910.07113,
2019.
[8] D. P. Bertsekas. Constrained optimization and Lagrange multiplier methods. Academic press, 2014.
[9] E. G. Birgin and J. M. Martínez. Practical augmented Lagrangian methods for constrained opti-
mization. SIAM, 2014.
[10] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. Training diffusion models with reinforcement
learning. arXiv preprint arXiv:2305.13301, 2023.
[11] H. Bojun. Steady state analysis of episodic reinforcement learning. Advances in Neural Information
Processing Systems, 33:9335–9345, 2020.
[12] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. Openai
gym, 2016.
[13] Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone. Risk-constrained reinforcement learning
with percentile risk criteria. The Journal of Machine Learning Research, 18(1):6070–6120, 2017.
[14] Y. Chow, O. Nachum, A. Faust, E. Duenez-Guzman, and M. Ghavamzadeh. Lyapunov-based safe
policy optimization for continuous control. arXiv preprint arXiv:1901.10031, 2019.
[15] J. Dai, J. Ji, L. Yang, Q. Zheng, and G. Pan. Augmented proximal policy optimization for safe
reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 37:7288–
7295, Jun. 2023.
[16] A. P. Dowling and A. S. Morgans. Feedback control of combustion oscillations. Annu. Rev. Fluid
Mech., 37:151–182, 2005.
[17] I. Greenberg and S. Mannor. Detecting rewards deterioration in episodic reinforcement learning. In
International Conference on Machine Learning, pages 3842–3853. PMLR, 2021.
[18] S. Gu, L. Yang, Y. Du, G. Chen, F. Walter, J. Wang, Y. Yang, and A. Knoll. A review of safe
reinforcement learning: Methods, theory and applications. arXiv preprint arXiv:2205.10330, 2022.
[19] H. Hu, Z. Liu, S. Chitlangia, A. Agnihotri, and D. Zhao. Investigating the impact of multi-lidar
placement on object detection for autonomous driving. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages 2550–2559, June 2022.
[20] I. Imayoshi, A. Isomura, Y. Harima, K. Kawaguchi, H. Kori, H. Miyachi, T. Fujiwara, F. Ishidate,
and R. Kageyama. Oscillatory control of factors determining multipotency and fate in mouse neural
progenitors. Science, 342(6163):1203–1208, 2013.
[21] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with diffusion for flexible behavior
synthesis. arXiv preprint arXiv:2205.09991, 2022.
[22] J. Ji, J. Zhou, B. Zhang, J. Dai, X. Pan, R. Sun, W. Huang, Y. Geng, M. Liu, and Y. Yang.
Omnisafe: An infrastructure for accelerating safe reinforcement learning research. arXiv preprint
arXiv:2305.09304, 2023.

10


---Page Break---
[23] S. Kakade and J. Langford. Approximately optimal approximate reinforcement learning. In
International Conference on Machine Learning, volume 2, pages 267–274, 2002.
[24] D. P. Kingma and J. Ba.
Adam:
A method for stochastic optimization.
arXiv preprint
arXiv:1412.6980, 2014.
[25] W. Klimesch. Alpha-band oscillations, attention, and controlled access to stored information. Trends
in cognitive sciences, 16(12):606–617, 2012.
[26] Y. Liu, J. Ding, and X. Liu. Ipo: Interior-point policy optimization under constraints. In Proceedings
of the AAAI conference on artificial intelligence, volume 34, pages 4940–4947, 2020.
[27] Y. Liu, A. Halev, and X. Liu. Policy learning with constraints in model-free reinforcement learning:
A survey. In The 30th international joint conference on artificial intelligence (ijcai), 2021.
[28] G. Neu and C. Pike-Burke. A unifying view of optimism in episodic reinforcement learning.
Advances in Neural Information Processing Systems, 33:1392–1403, 2020.
[29] A. Ray, J. Achiam, and D. Amodei. Benchmarking Safe Exploration in Deep Reinforcement
Learning. arXiv preprint arXiv:1910.01708, 2019.
[30] A. Ray, J. Achiam, and D. Amodei. Benchmarking safe exploration in deep reinforcement learning.
arXiv preprint arXiv:1910.01708, 7, 2019.
[31] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In
International Conference on Machine Learning, pages 1889–1897, 2015.
[32] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control
using generalized advantage estimation. International Conference on Learning Representations
(ICLR), 2016.
[33] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization
algorithms. arXiv preprint arXiv:1707.06347, 2017.
[34] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker,
M. Lai, A. Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–
359, 2017.
[35] A. Stooke, J. Achiam, and P. Abbeel. Responsive safety in reinforcement learning by pid lagrangian
methods. In International Conference on Machine Learning, pages 9133–9143. PMLR, 2020.
[36] C. Tessler, D. J. Mankowitz, and S. Mannor. Reward constrained policy optimization. International
Conference on Learning Representation (ICLR), 2019.
[37] E. Vinitsky, A. Kreidieh, L. Le Flem, N. Kheterpal, K. Jang, F. Wu, R. Liaw, E. Liang, and A. M.
Bayen. Benchmarks for reinforcement learning in mixed-autonomy traffic. In Proceedings of
Conference on Robot Learning, pages 399–409, 2018.
[38] O. Vinyals, I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi, R. Powell,
T. Ewalds, P. Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement
learning. Nature, 575(7782):350–354, 2019.
[39] T.-Y. Yang, J. Rosca, K. Narasimhan, and P. J. Ramadge. Projection-based constrained policy
optimization. In International Conference on Learning Representation (ICLR), 2020.
[40] J. Yuan and A. Lamperski. Online convex optimization for cumulative constraints. Advances in
Neural Information Processing Systems, 31, 2018.
[41] L. Zhang, L. Shen, L. Yang, S. Chen, B. Yuan, X. Wang, and D. Tao. Penalized proximal policy
optimization for safe reinforcement learning. arXiv preprint arXiv:2205.11814, 2022.
[42] S. Zhang, Y. Wan, R. S. Sutton, and S. Whiteson. Average-reward off-policy policy evaluation with
function approximation. arXiv preprint arXiv:2101.02808, 2021.
[43] Y. Zhang, Q. Vuong, and K. W. Ross. First order constrained optimization in policy space. arXiv
preprint arXiv:2002.06506, 2020.

11


---Page Break---
A
Appendix

A.1
Proofs

Lemma A.1. For an episode of length H and two policies, π and π1, the difference in their performance
assuming identical initial state distribution µ (i.e., s1 „ µ) is given by

Jpπq ´ Jpπ1q “

H
ÿ

h“1
E
sh,ah„Pπ
h p¨,¨ | sq
s1„µ

“
Aπ1
h psh, ahq
‰
.
(2)

Proof. Let us consider s1 „ µ and categorize the value function difference between the two policies.
Also define Pπ
h ps | s1q “ ř

aPA Pπ
h ps, a | s1q, where the term Pπ
h ps, a | s1q is the probability of reaching
ps, aq at time step h following π and starting from s1.

V π
1 ps1q ´ V π1
1 ps1q “
E
a1,s2
“
rps1, a1q ` V π
2 ps2q|s1
‰
` E
s2
“
V π1
2 ps2q ´ V π1
2 ps2q|s1
‰
´ V π1
1 ps1q

“ E
s2
“
V π
2 ps2q ´ V π1
2 ps2q|s1
‰
`
E
a1,s2
“
rps1, a1q ` V π1
2 ps2q ´ V π1
1 ps1q|s1
‰

“ E
s2
“
V π
2 ps2q ´ V π1
2 ps2q|s1
‰
` E
a1
“
Qπ1
1 ps1, a1q ´ V π1
1 ps1q|s1
‰

“ E
s2
“
V π
2 ps2q ´ V π1
2 ps2q|s1
‰
` E
a1
“
Aπ1
1 ps1, a1q|s1
‰
,

where a1 „ π1p¨|s1q, s2 „ Pp¨|s1, π1ps1qq and s1 „ µ, the initial state distribution.

Now recursively apply the same procedure to the term V π
h pshq ´ V π1
h pshq @ h P t2, . . . , Hu to obtain
the following:

V π
1 psq ´ V π1
1 psq “

H
ÿ

h“1
E
sh,ah„Pπ
h p¨,¨ | sq

“
Aπ1
h psh, ahq|s
‰

Now we know that Jpπq “ E
s„µrV π
1 psqs, this means that we combine this with the above to obtain the

final result.

Proposition A.2. The inner optimization problem in (5) with respect to x is a convex quadratic program
with non-negative constraints, which can be solved to yield the following intermediate problem:

pπ‹
k,t, λ‹
tq “ max
λě0 min
πk,t Ltpπk, λ, βq,
where

Ltpπk, λ, βq “

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
` β

2

m
ÿ

i

ˆ
max
"
0, ΨCi,tpπk´1, πkq ` λt,i

β

*2
´ λ2
t,i
β2

˙
.
(6)

Proof. As in Equation (4), we have the following equivalent problem:

π‹
k,t “ arg min
πk,t

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,i max
"
0,

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
Ci,h ps, aq
‰
` pJCipπk´1q ´ diq
*
.

Letting ΨCi,tpπk´1, πkq :“ řH
h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
Ci,h ps, aq
‰
` pJCipπk´1q ´ diq, and introducing

slack variables xt,i ě 0 and defining wt,ipπkq :“ ΨCi,tpπk´1, πkq ` xt,i “ 0, we get the quadratic
damped problem same as Equation (5) below.

pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x Ltpπk, λ, x, βq

“ max
λě0 min
πk,t,x

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

(9)

Like the Lagrangian method, we can alternately update π, λ, and x to find the optimal triplet. Consider
updating π and x by minimizing Ltpπ, λ, x, βq at any iteration:

12


---Page Break---
`
π‹
k,t, x‹
t
˘
“ arg min
πk,t
min
xią0

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i“1
λt,i
`
ΨCi,tpπk´1, πkq ` xt,i
˘
` β

2

m
ÿ

i“1

`
ΨCi,tpπk´1, πkq ` xt,i
˘2

The inner optimization problem with respect to x is a convex quadratic programming problem with
non-negative constraints,

x‹
t “ arg min
xią0

m
ÿ

i“1
λt,i
`
ΨCi,tpπk´1, πkq ` xt,i
˘
` β

2

m
ÿ

i“1

`
ΨCi,tpπk´1, πkq ` xt,i
˘2

The optimal solution is x‹
t,i “ max
!
0, ´ λt,i

β ´ ΨCi,tpπk´1, πkq
)
. Then,

wt,ipπkq “ ΨCi,tpπk´1, πkq ` x‹
t,i “ ΨCi,tpπk´1, πkq ` max
"
0, ´λt,i

β
´ ΨCi,tpπk´1, πkq
*

“ λt,i

β
` ΨCi,tpπk´1, πkq ` max
"
0, ´λt,i

β
´ ΨCi,tpπk´1, πkq
*
´ λt,i

β

“ max
"
0, λt,i

β
` ΨCi,tpπk´1, πkq
*
´ λt,i

β

Substituting back into Equation (9), we get

Ltpπk, λ, x, βq “

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i“1
λt,i

ˆ
max
"
0, λt,i

β
` ΨCi,tpπk´1, πkq
*
´ λt,i

β

˙

` β

2

m
ÿ

i“1

ˆ
max
"
0, λt,i

β
` ΨCi,tpπk´1, πkq
*
´ λt,i

β

˙2

“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
` β

2

m
ÿ

i“1

˜

max
"
0, λt,i

β
` ΨCi,tpπk´1, πkq
*2
´ λ2
t,i
β2

¸

Hence, we finally get

pπ‹
k,t, λ‹
tq “ max
λě0 min
πk,t Ltpπk, λ, βq

“ max
λě0 min
πk,t

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
` β

2

m
ÿ

i“1

ˆ
max
"
0, ΨCi,tpπk´1, πkq ` λt,i

β

*2
´ λ2
t,i
β2

˙

Lemma A.3. Consider two problems, Problem (P) and Problem (Q) below. For sufficiently large
λi ą ¯λ @ i and β ą ¯β for some finite ¯λ and finite ¯β, the optimal solution set of Problem (Q) (equivalent
version of Problem (6)) is identical to the optimal solution set of Problem (P).

Problem (P) :

LP
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LP
t pπk, λ, x, βq
(P)

Problem (Q) :

13


---Page Break---
LQ
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, θq

` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, θq2

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LQ
t pπk, λ, x, βq
(Q)

, where x` :“ maxp0, xq, and

ΨCi,tpπk´1, πk, θq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
Ci,h ps, aq
‰
` pJCipπk´1q ´ diq.

Proof. This proof uses some ideas given in [41] for Part 1 below.

Part 1 - Solution of Problem (P) is solution of Problem (Q).

Suppose ¯θt is the optimum of the constrained Problem (P) augmented with the quadratic penalty. Let ¯λt be
the corresponding Lagrange multiplier vector for its dual problem, and ¯β be the additive quadratic penalty
coefficient. Then for λt,i ě ||¯λ||8 @ i and β ě ||¯β||8, ¯θ is also a minimizer of its ReLU-penalized
optimization Problem (Q) as below. Let Ωpθtq :“ řH
h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
. Then it follows

that:

Ωpθtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, θq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, θq2 ě Ωpθtq `

m
ÿ

i
¯λiΨ`
Ci,tpπk´1, πk, θq `
¯β
2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, θq2

ě Ωpθtq `

m
ÿ

i
¯λiΨCi,tpπk´1, πk, θq `
¯β
2

m
ÿ

i
ΨCi,tpπk´1, πk, θq2

By assumption, ¯θt is a Karush-Kuhn-Tucker point in the constrained Problem (P), at which KKT conditions
are satisfied with the Lagrange multiplier vector ¯λ and ¯β. We then have:

Ωpθtq `

m
ÿ

i
¯λiΨCi,tpπk´1, πk, θq `
¯β
2

m
ÿ

i
ΨCi,tpπk´1, πk, θq2 ě Ωp¯θtq `

m
ÿ

i
¯λiΨCi,tpπk´1, πk, ¯θq `
¯β
2

m
ÿ

i
ΨCi,tpπk´1, π, ¯θq2

“ Ωp¯θtq `

m
ÿ

i
¯λiΨ`
Ci,tpπk´1, πk, ¯θq `
¯β
2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, ¯θq2

“ Ωp¯θtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, ¯θq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, ¯θq2

, where the first line holds because ¯θt minimizes the Lagrange function, and the second line is derived
from the complementary slackness. Thus, we conclude that for the objective function of Problem (Q), call
it LQpθtq, we have LQpθtq ě LQp¯θtq for all θt P Θ, which means ¯θt is a minimizer of the quadratic
damped optimization Problem (Q).

Part 2 - Solution of Problem (Q) is solution of Problem (P).

Let rθt be an optimal point of the quadratic damped Problem (Q), with ¯θt and ¯λ being the same as defined
above. Then, if rθt is in the set of feasible solutions Sfeasible “ tθ | ΨCi,tpπk´1, πk, θq ď 0
@ iu, we
have:

Ωprθtq “ Ωprθtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, rθq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, rθq

ď Ωpθtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, θq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, θq

“ Ωpθtq

14


---Page Break---
The inequality above indicates rθt is also optimal in the constrained Problem (P). Now, if rθ is not feasible,
we have:

Ωp¯θtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, ¯θq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, ¯θq2 “ Ωp¯θtq `

m
ÿ

i
¯λiΨ`
Ci,tpπk´1, πk, ¯θq `
¯β
2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, ¯θq2

“ Ωp¯θtq `

m
ÿ

i
¯λiΨCi,tpπk´1, πk, ¯θq `
¯β
2

m
ÿ

i
ΨCi,tpπk´1, πk, ¯θq2

ď Ωprθtq `

m
ÿ

i
¯λiΨCi,tpπk´1, πk, rθq `
¯β
2

m
ÿ

i
ΨCi,tpπk´1, πk, rθq2

ď Ωprθtq `

m
ÿ

i
¯λiΨ`
Ci,tpπk´1, πk, rθq `
¯β
2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, rθq2

ď Ωprθtq `

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, rθq ` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, rθq2

, which is a contradiction to the assumption that rθt is a minimizer of the penalized optimization Problem
(Q). Thus, rθt can only be the feasible optimal solution for Problem (P).

Lemma A.4. Consider two problems, Problem (P’) and Problem (R). For sufficiently large β ą ¯β for
some finite ¯β, the feasible optimal solution set of Problem (R) (equivalent version of Problem (3)) is
identical to the solution set of Problem (P’).

Problem (P’) :

LP 1
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LP 1
t pπk, λ, x, βq
(P’)

Problem (R) :

LR
t pπk, λ, xq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LR
t pπk, λ, xq
(R)

Proof. Recall that we are using parameterized policies, hence we overload notation as θ ” π frequently.
For brevity, denote Ωtpπq :“ řH
h“t
E
s„ρπk,h
a„πk´1,h

“
ρpθhqAπk´1
h
ps, aq
‰
. We will also go back and forth between

the equivalent problems of Problem (P’) and Problem (6) of the main paper in Section 3.

Part 1. Solution of Problem (R) is solution of Problem (P’).

Suppose that π‹ is the optimal feasible policy for the primal Problem (R), which is a Lagrangian version
of Problem (3). Consider the corresponding Langrangian dual parameter λ‹ of π‹, which satisfies the
KKT conditon,

∇πLR
t
`
π‹
k,t, λ‹, x‹˘
“ ´∇πΩt
`
π‹
k,t
˘
`

m
ÿ

i“1
λ*
t,i∇πwt,i pπ‹
kq “ 0

and the second-order sufficient condition that for all non-zero vectors u that satisfy uT ∇πwt,i pπ‹
kq “ 0 ,
we have

uT ∇2
πLR
t
`
π‹
k,t, λ‹, x‹˘
u ą 0
(A)

Compare Equation (R) and Equation (P’), we have,

15


---Page Break---
∇πLP 1
t
`
π‹
k,t, λ‹, x‹, β
˘
“ ´∇πΩt
`
π‹
k,t
˘
`

m
ÿ

i“1
λ‹
t,i∇πwt,i pπ‹
kq ` β

m
ÿ

i“1
wt,i pπ‹
kq ∇πwt,i pπ‹
kq

“ ∇πLR
t
`
π‹
k,t, λ‹, x‹˘
` β

m
ÿ

i“1
wt,i pπ‹
kq ∇πwt,i pπ‹
kq

“ 0

, where we use wt,ipπ‹
kq :“ ΨCi,tpπk´1, π‹q ` x‹
t,i “ 0 with the feasible policy π‹. Moreover,

∇2
πLP 1
t
`
π‹
k,t, λ‹, x‹, β
˘
“ ´ ∇2
πΩt
`
π‹
k,t
˘

`

m
ÿ

i“1
λ‹
t,i∇2
πwt,i pπ‹
kq ` β∇πwt pπ‹
kq ∇πwt pπ‹
kqT

“∇2
πLR
t
`
π‹
k,t, λ‹, x‹˘
` β∇πwt pπ‹
kq ∇πwt pπ‹
kqT .

To prove that pπ‹
k,t, λ‹q is a strict minimum solution to LP 1
t pπk, λ, x, βq, we only need to prove the
following is true for sufficiently large β,

∇2
πLP 1
t
pπ‹
k, λ‹, x‹, βq ą 0.

If the above is not true, then for any large β, there exists ut such that }ut} “ 1 and satisfies

uT
t ∇2
πLP 1
t
pπ‹
k, λ‹, x‹, βq ut “ uT
t ∇2
πLR
t pπ‹
k, λ‹, x‹q ut ` β
›››∇πwt pπ‹
kqT ut
›››
2
ď 0

ñ
›››∇πwt pπ‹
kqT ut
›››
2
ď ´ 1

β uT
t ∇2
πLR
t pπ‹
k, λ‹, x‹q ut Ñ 0, as β Ñ 8.

Therefore, tuhu is a bounded sequence and there must be a limit point, denoted by 8u. Then

∇πwt pπ‹
kqT 8u “ 0

8uT ∇2
πLR
t pπ‹
k, λ‹, x‹q 8u ď 0.

The above contradicts Equation (A), so the conclusion. Hence, π‹
k,t is also the optimal feasible policy for
the primal-dual Problem (P’).

Part 2. Solution of Problem (P’) is solution of Problem (R).

This part is straightforward since it is a standard result. Please see Chapter 2 and Chapter 9 of [9], and
Chapter 2 and Chapter 4 of [8] for the proof. For completeness, we provide the result below.

Suppose π‹
k,t in the feasible optimal solution set of the primal-dual Problem (P’). Let λ‹ be the corre-
sponding dual parameter of π‹
k,t. Consider Problem (6), which is an equivalent version of Problem (P’).
For any feasible πk, we have

Lt pπ‹
k, λ‹, βq ď Lt pπk, λ‹, βq .

Now we have two cases:

Case 1. When
λ‹
t,i
β ` ΨCi,tpπk´1, π‹
kq ą 0, we have

16


---Page Break---
Lt pπk, λ‹, βq “ ´Ωtpπk,tq `

m
ÿ

i“1
λ‹
t,iΨCi,tpπk´1, πkq ` β

2

m
ÿ

i“1
Ψ2
Ci,tpπk´1, πkq

“ ´Ωtpπk,tq `

m
ÿ

i“1
βΨCi,tpπk´1, πkq
ˆλ‹
t,i
β
` ΨCi,tpπk´1, πkq
˙
´ β

2

m
ÿ

i“1
Ψ2
Ci,tpπk´1, πkq

ď ´Ωtpπk,tq

where the last step uses β ą 0, ΨCi,tpπk´1, πkq ă 0, and
λ‹
t,i
β ` ΨCi,tpπk´1, πkq ą 0.

Case 2. When
λ‹
t,i
β ` ΨCi,tpπk´1, π‹
kq ď 0, we have

Lt pπk, λ‹, βq “ ´Ωtpπk,tq ´ 1

2β

m
ÿ

i“1
λ‹2
t,i ď ´Ωtpπk,tq.

Now, combining both cases above, we have Lt pπk, λ‹, βq ď ´Ωtpπk,tq. On the other hand,
Lt pπ‹
k, λ‹, βq “ Lt pπ‹
k, λ‹, x‹, βq “ ´Ωtpπ‹
k,tq. Thus, the combining all of the above we get

´Ωtpπ‹
k,tq “ Lt pπ‹
k, λ‹, βq ď Lt pπk, λ‹, βq ď ´Ωtpπk,tq.

Theorem 3.3. Let π(3)‹ be a solution to Problem (3), and let
`
π(6)‹, λ(6)‹˘
be a solution to Problem (6).
Then, for sufficiently large β ą ¯β and λt,i ą ¯λ @ i, π(3)‹ is a solution to Problem (6), and π(6)‹ is a
solution to Problem (3).

Proof. We prove this result as a two-step process.

First, we show that the solution sets of the below problems are identical. See Lemma A.3 for the proof.

Problem (P).

LP
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LP
t pπk, λ, x, βq
(P)

Problem (Q).

LQ
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iΨ`
Ci,tpπk´1, πk, θq

` β

2

m
ÿ

i
Ψ`
Ci,tpπk´1, πk, θq2

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LQ
t pπk, λ, x, βq
(Q)

Second, we show that the solution sets of the below problems are identical. See Lemma A.4 for the proof.

Problem (P’) :

LP 1
t pπk, λ, x, βq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq ` β

2

m
ÿ

i
w2
t,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LP 1
t pπk, λ, x, βq
(P’)

17


---Page Break---
(a) Humanoid
(b) Circle
(c) Reach
(d) Grid
(e) Bottleneck
(f) Navigation

Figure 5: The Humanoid, Circle, Reach, Grid, Bottleneck, and Navigation tasks. (a) Humanoid: The agent is to run
as fast as possible on a flat surface, while not exceeding a specified speed limit i.e. the cost constraint. (b) Circle: The
agent is rewarded for moving in a specified circle but is penalized if the diameter of the circle is larger than some value
[3]. (c) Reach: The agent is rewarded for reaching a goal while avoiding obstacles (cost constraints) that are placed
to hinder the agent [30]. (d) Grid: The agent controls traffic lights in a 3x3 road network and is rewarded for high
traffic throughput but is constrained to let lights be red for at most 5 consecutive seconds [37]. (e) Bottleneck: The
agent controls vehicles (red) in a merging traffic situation and is rewarded for maximizing the number of vehicles that
pass through but is constrained to ensure that white vehicles (not controlled by agent) have “low” speed for no more
than 10 seconds [37]. (f) Navigation: The agent is rewarded for reaching the target area (green) but is constrained to
avoid hazards (light purple) and impassible pillars (dark purple). The cost for hazards and pillars is different [30].

Problem (R) :

LR
t pπk, λ, xq :“

H
ÿ

h“t
E
s„ρπk,h
a„πk´1,h

“
´ ρpθhqAπk´1
h
ps, aq
‰
`

m
ÿ

i
λt,iwt,ipπkq

Then,
pπ‹
k,t, λ‹
t, x‹
tq “ max
λě0 min
πk,t,x LR
t pπk, λ, xq
(R)

Now, it follows from equivalency that the optimal solution of Problem (Q) and Problem (R), and hence
Problem (6) and Problem (3), is the same.

A.2
Experiments Revisited

Below we detail the experimental attributes that we used in benchmarking. See Figure 5 for the environ-
ment details. All our experiments are run in the omnisafe module [22].

A.2.1
Environment Details

Comprehensively, our experiments consist of eight tasks ranging from more superficial (Run and Circle
tasks) to relatively more stochastic and sophisticated (Bottleneck and Grid tasks), each training different
robots. They come from three well-known safe RL benchmark environments, Safe MuJoCo, Bullet-Safety-
Gym, and Safety-Gym. For agents maneuvering on a two-dimensional plane, the cost is calculated as
Cps, aq “
b

v2x ` v2y. For agents moving along a straight line, the cost is calculated as Cps, aq “ |vx|,
where vx and vy are the velocities of the agent in the x and y directions.

Circle
This environment is inspired by [3]. Reward is maximized by moving along a circle of radius d:

R “
vTr´y, xs

1 `
ˇˇa

x2 ` y2 ´ d
ˇˇ,

but the safety region xlim is smaller than the radius d : C “ 1rx ą xlims.

Navigation
This environment is inspired by [30]. Reward is maximized by getting close to the des-
tination R “ Distptarget, st´1q ´ Distptarget, stq, but it yields a cost of +1 when the agent hits the
hazard or the pillar. The two different types of cost functions are returned separately and have different
thresholds. In out setting, d1 “ 25 for the hazard constraint and d2 “ 20 for the pillar constraint.

Since the main goal in MuJoCo is to train the robot to locomote on the plane, we call it the "Run"
task in our article. Our chosen two robots are the relatively complex types in MuJoCo: Ant and
Humanoid. OpenAI Gym is open source at https://github.com/openai/gym, and has a documentation

18


---Page Break---
Hyperparameter
APPO
PDO
FOCOPS
CPPO-PID
IPO
P3O
CPO
TRPO-L
PCPO

Actor Net layers
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q

Critic Net layers
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q
p32, 32q

Activation
tanh
tanh
tanh
tanh
tanh
tanh
tanh
tanh
tanh

Initial log std
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5

Discount γ
0.99
0.95
0.995
0.995
0.99
0.99
0.99
0.99
0.99

Policy lr
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4
3e ´ 4

Critic Net lr
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3
1e ´ 3

No. of episodes
500
500
500
500
500
500
500
500
500

Steps per epochs
300
300
300
300
300
300
300
300
300

Target KL
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01
0.01

KL early stop
True
True
True
True
True
True
False
False
False

Line Search Times
N/A
N/A
N/A
N/A
N/A
N/A
25
25
25

Line Search Decay
N/A
N/A
N/A
N/A
N/A
N/A
0.8
0.8
0.8

Proximal clip
0.2
0.2
0.2
0.2
0.2
0.2
N/A
N/A
N/A

Max horizon
200
200
200
200
200
200
200
200
200

Table 3: Hyperparameters used for each baseline.

at https://www.gymlibrary.ml/. Bullet Safety Gym. The implementation of the Circle task comes from
Bullet-safety-Gym (Gronauer 2022), which Stooke, Achiam, and Abbeel (2020) first proposed. The
reward is dense and increases by the agent’s velocity and the proximity to the boundary of the circle.
Costs are received when the agent leaves the safety zone defined by the two yellow boundaries. The
environment is open source at https://github.com/SvenGronauer/Bullet-Safety-Gym.

Safety Gym. The remaining two tasks, Goal and Button, are from Safety-Gym (Ray, Achiam, and Amodei
2019). Compared to Run and Circle tasks, they are more stochastic and sophisticated in that agents are
challenged to maximize the return while satisfying the constraints.

The environment is open source at https://github.com/openai/safety-gym, and readers can see OpenAI’s
blog at https://openai.com/blog/safety-gym/ for more details.

A.3
Agents

For single-constraint scenarios, Point agent is a 2D mass point(A Ď R2) and Ant is an quadruped
robot(A Ď R8). For the multi-constraint scenario which is modified from OpenAI SafetyGym [30],
S Ď R28`16¨m where m is the number of pseudo-radar (one for each type of obstacles and we set two
different types of obstacles in the Navigation task) and A Ď R2 for a mass point or a wheeled car.

A.3.1
Experimental Details

To be fair in comparison, the proposed e-COP algorithm and FOCOPS [43] are implemented with same
rules and tricks on the code-base of [30].

A.3.2
Hyperparameters

Table 3 shows the hyperparameters of baseline algorithms.

A.3.3
Runtime Environment

All experiments were implemented in Pytorch 1.7 .0 with CUDA 11.0 and conducted on an Ubuntu
20.04.2 LTS with 8 CPU cores (AMD Ryzen Threadripper PRO 3975WX 8-Coresz), 127G memory and 2
GPU cards (NVIDIA GeForce RTX 4060 Ti Cards).

A.3.4
Robustness to Cost Thresholds

We conducted a set of experiments wherein we study how e-COP effectively adapts to different cost
thresholds. For this, we use a pre-trained e-COP agent, which is trained with a particular cost threshold in

19


---Page Break---
an environment, and test its performance on different cost thresholds within the same environment. Figure
6 illustrates the training curves of these pre-trained agents, and we see that while e-COP can generalize
well across different cost thresholds, other baseline algorithms may require further tuning to accommodate
different constraint thresholds.

0
100
200
300
400
500

3000

2500

2000

1500

1000

500

0

(a) Humanoid Rewards

0
100
200
300
400
500

60

50

40

30

20

10

0

(b) Humanoid Costs

0
100
200
300
400
500

60

50

40

30

20

10

0

(c) Point Circle Rewards

0
100
200
300
400
500

18

15

12

9

6

3

0

(d) Point Circle Costs

Figure 6: Cumulative episodic rewards and costs of baselines in two environments with two different constraint cost
thresholds: 10 and 50 in Humanoid, and 3 and 15 in Point Circle. The hyperparameters are tuned at constraint limit
of 20 in Humanoid and 10 in Point Circle.

20


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope?
Answer: [Yes]
Justification: We make claims on developing the theory for episodic policy optimization for
CMDPs and on presenting experimental validation of our approach. Please see Sections 3 and 4.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in
the paper.
• The abstract and/or introduction should clearly state the claims made, including the contri-
butions made in the paper and important assumptions and limitations. A No or NA answer
to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much
the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are
not attained by the paper.
2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: While there are no algorithmic limitations, potential limitations can occur while
applying our work to very high dimensional real-life scenarios, for instance, in large language
models or the denoising process in image and video diffusion tasks. We have developed this work
with the above two applications in mind, and plan to explore that direction of future research.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the
paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings, model
well-specification, asymptotic approximations only holding locally). The authors should
reflect on how these assumptions might be violated in practice and what the implications
would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only
tested on a few datasets or with a few runs. In general, empirical results often depend on
implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For
example, a facial recognition algorithm may perform poorly when image resolution is low
or images are taken in low lighting. Or a speech-to-text system might not be used reliably
to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and
how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address
problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important
role in developing norms that preserve the integrity of the community. Reviewers will be
specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a
complete (and correct) proof?

21


---Page Break---
Answer: [Yes]
Justification: All our lemmas and theorems have proofs, some which are available in the main
text, and some in Appendix A.
Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they
appear in the supplemental material, the authors are encouraged to provide a short proof
sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by
formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We provide the evaluation protocol and baseline comparison metrics in Section 4.
More information is also available in Appendix A.2.
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well
by the reviewers: Making the paper reproducible is important, regardless of whether the
code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to
make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For
example, if the contribution is a novel architecture, describing the architecture fully might
suffice, or if the contribution is a specific model and empirical evaluation, it may be
necessary to either make it possible for others to replicate the model with the same dataset,
or provide access to the model. In general. releasing code and data is often one good way
to accomplish this, but reproducibility can also be provided via detailed instructions for
how to replicate the results, access to a hosted model (e.g., in the case of a large language
model), releasing of a model checkpoint, or other means that are appropriate to the research
performed.
• While NeurIPS does not require releasing code, the conference does require all submissions
to provide some reasonable avenue for reproducibility, which may depend on the nature of
the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to
reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe the
architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct the
dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors
are welcome to describe the particular way they provide for reproducibility. In the case
of closed-source models, it may be that access to the model is limited in some way (e.g.,
to registered users), but it should be possible for other researchers to have some path to
reproducing or verifying the results.
5. Open access to data and code

22


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instructions
to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We have provided the code.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including
code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to
access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed
method and baselines. If only a subset of experiments are reproducible, they should state
which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions
(if applicable).
• Providing as much information as possible in supplemental material (appended to the paper)
is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters,
how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We detail our experimental procedure and details in Sections 4 and A.2.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that
is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All our results are reported with 1 standard deviation across multiple seeds. Please
see Section 4.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the main
claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall run
with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to
a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).

23


---Page Break---
• It should be clear whether the error bar is the standard deviation or the standard error of the
mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably
report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality
of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures
symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they
were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer
resources (type of compute workers, memory, time of execution) needed to reproduce the
experiments?

Answer: [Yes]

Justification: In Section A.2 we detail the compute resources used.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or
cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the
experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make
it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have read the code of ethics and our work satisfies it in every respect.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration
due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal
impacts of the work performed?

Answer: [Yes]

Justification: In the Introduction Section 1 we detail the potential applications of our work.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact
or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g.,
deployment of technologies that could make decisions that unfairly impact specific groups),
privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to
particular applications, let alone deployments. However, if there is a direct path to any
negative applications, the authors should point it out. For example, it is legitimate to point

24


---Page Break---
out that an improvement in the quality of generative models could be used to generate
deepfakes for disinformation. On the other hand, it is not needed to point out that a generic
algorithm for optimizing neural networks could enable people to train models that generate
Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being
used as intended and functioning correctly, harms that could arise when the technology is
being used as intended but gives incorrect results, and harms following from (intentional or
unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks, mecha-
nisms for monitoring misuse, mechanisms to monitor how a system learns from feedback
over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release
of data or models that have a high risk for misuse (e.g., pretrained language models, image
generators, or scraped datasets)?

Answer: [NA]

Justification: Our work needs no such safeguards for responsible release of assets.

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring that
users adhere to usage guidelines or restrictions to access the model or implementing safety
filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not
require this, but we encourage authors to take this into account and make a best faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the
paper, properly credited and are the license and terms of use explicitly mentioned and properly
respected?

Answer: [NA]

Justification: We do not use existing assets. Proper references for existing results and algorithms
are provided throughout the main text.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service
of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the
derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the
asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

25


---Page Break---
Answer: [NA]
Justification: We do not release new assets. Our results, both theoretical, algorithmic, and
empirical, are properly referenced and explained in the main text.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their sub-
missions via structured templates. This includes details about training, license, limitations,
etc.
• The paper should discuss whether and how consent was obtained from people whose asset
is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as well as
details about compensation (if any)?
Answer: [NA]
Justification: We do not use crowdsourcing or research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribution
of the paper involves human subjects, then as much detail as possible should be included in
the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or
other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether such
risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or
an equivalent approval/review based on the requirements of your country or institution) were
obtained?
Answer: [NA]
Justification: Our work does not involve crowdsourcing or research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you should
clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and
locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines
for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

26


---Page Break---
