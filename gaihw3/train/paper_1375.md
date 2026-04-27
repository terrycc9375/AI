Online Control with Adversarial Disturbance for
Continuous-time Linear Systems

Jingwei Li
IIIS, Tsinghua University
Shanghai Qizhi Institute
ljw22@mails.tsinghua.edu.cn

Jing Dong
The Chinese University of Hong Kong, Shenzhen
jingdong@link.cuhk.edu.cn

Can Chang
IIIS, Tsinghua University
cc22@mails.tsinghua.edu.cn

Baoxiang Wang
The Chinese University of Hong Kong, Shenzhen
bxiangwang@cuhk.edu.cn

Jingzhao Zhang
IIIS, Tsinghua University
Shanghai Qi zhi Institute
jingzhaoz@mail.tsinghua.edu.cn

Abstract

We study online control for continuous-time linear systems with finite sampling
rates, where the objective is to design an online procedure that learns under non-
stochastic noise and performs comparably to a fixed optimal linear controller. We
present a novel two-level online algorithm, by integrating a higher-level learning
strategy and a lower-level feedback control strategy. This method offers a practical
and robust solution for online control, which achieves sublinear regret. Our work
provides the first nonasymptotic results for controlling continuous-time linear
systems with finite number of interactions with the system. Moreover, we examine
how to train an agent in domain randomization environments from a non-stochastic
control perspective. By applying our method to the SAC (Soft Actor-Critic)
algorithm, we achieved improved results in multiple reinforcement learning tasks
within domain randomization environments. Our work provides new insights into
non-asymptotic analyses of controlling continuous-time systems. Furthermore,
our work brings practical intuition into controller learning under non-stochastic
environments.

1
Introduction

A major challenge in robotics is to deploy simulated controllers into real-world. This process,
known as sim-to-real transfer, can be difficult due to misspecified dynamics, unanticipated real-world
perturbations, and non-stationary environments. Various strategies have been proposed to address
these issues, including domain randomization, meta-learning, and domain adaptation [20, 10, 21].
Although they have shown great effectiveness in experimental results, training agents within these
setups poses a significant challenge. To accommodate different environments, the strategies developed
by agents tend to be overly conservative [26, 4] or lead to suboptimal outcomes [43, 27].

In this work, we provide an analysis of the sim-to-real transfer problem from an online control
perspective. Online control focuses on iteratively updating the controller after deployment (i.e.,

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
online) based on collected trajectories. Significant progress has been made in this field by applying
insights from online learning to linear control problems [2, 1, 12, 19, 11, 8, 6, 16].

Following this line of work, we approach the sim-to-real transfer issue for continuous-time linear
systems as a non-stochastic control problem, as explored in previous works [19, 11, 8]. These studies
provide regret bounds for an online controller that lacks prior knowledge of system perturbations.
However, a gap remains as no previous analysis has specifically investigated continuous-time systems,
but real world systems often evolve continuously in time.

Existing literature on online continuous control is limited [42, 22, 13, 32]. Most continuous control
research emphasizes the development of model-free algorithms, such as policy iteration, under the
assumption of noise absence. Recently, [8] examined online continuous-time linear quadratic control
problem and achieves sublinear regret. However, it relies on the assumption of standard Brownian
noise instead of non-stochastic noise that may not always hold true in real-world applications. This
leads us to the crucial question:

Is it possible to design an online non-stochastic control algorithm
in a continuous-time setting that achieves sublinear regret?

Our work addresses this question by proposing a two-level online controller. The higher-level
controller symbolizes the policy learning process and updates the policy at a low frequency to
minimize regret. Conversely, the lower-level controller delivers high-frequency feedback control
input to reduce discretization error. Our proposed algorithm results in regret bounds for continuous-
time linear control in the face of non-stochastic disturbances.

Furthermore, we implement the ideas from our theoretical analysis and test them in several experi-
ments. Note that the key difference between our algorithm and traditional online policy optimization
is that we utilize information from past states with some skips to enable faster adaptation to en-
vironmental changes. Although the aforementioned concepts are often adopted experimentally as
frame stacking and frame skipping, there is relatively little known about the appropriate scenarios
for applying these techniques. Our analysis and experiments demonstrate that these techniques are
particularly effective in developing adaptive policies for uncertain environments. We choose the task
of training agents in a domain randomization environment to evaluate our method, and the results
confirm that these techniques substantially improve the agents’ performance.

2
Related Works

The control theory of linear dynamical systems under disturbances has been thoroughly examined
in various contexts, such as linear quadratic stochastic control [7], robust control [37, 23], system
identification [17, 24, 9, 25]. However, most of these problems are investigated in non-robust settings,
with robust control being the sole exception where adversarial perturbations in the dynamic are
permitted. In this scenario, the controller solves for the optimal linear controller in the presence of
worst-case noise. Nonetheless, the algorithms designed in this context can be overly conservative as
they optimize over the worst-case noise, a scenario that is rare in real-world applications. We will
elaborate on the difference between robust control and online non-stochastic control in Section 3.

Online Control
There has been a recent surge of interest in online control, as demonstrate by
studies such as [2, 1, 12]. In online control, the player interacts with the environment and updates
the policy in each round aiming to achieve sublinear regret. In scenarios with stochastic Gaussian
noise, [12] provides the first efficient algorithm with an O(
√

T) regret bound. However, in real-world
applications, the assumption of Gaussian distribution is often unfulfilled.

[3] pioneers research on non-stochastic online control, where the noises can be adversarial. Under
general convex costs, they introduce the Disturbance-Action Policy Class. Using an online convex
optimization (OCO) algorithm with memory, they achieve an O(
√

T) regret bound. Subsequent
studies extend this approach to other scenarios, such as quadratic costs [8], partial observations [36,
35] or unknown dynamical systems [19, 11]. Other works yield varying theoretical guarantees like
online competitive ratio [15, 33].

Online Continuous Control
Compared to online control, there has been relatively little research on
model-based continuous-time control. Most continuous control works focus on developing model-free

2


---Page Break---
algorithms such as policy iteration (e.g. [42, 22, 32]), typically assuming zero-noise. This is because
analyzing the system when transition dynamics are represented by differential equations, rather than
recurrence formulas, poses a significant challenge.

Recently, [8] studies online continuous-time linear quadratic control with standard Brownian noise
and unknown system dynamics. They propose an algorithm based on the least-square method, which
estimates the system’s coefficients and solves the corresponding Riccati equation. The papers [34, 14]
also focus on online control setups with continuous-time stochastic linear systems and unknown
dynamics. They achieve O(
√

T log T) regret by different approaches. [34] uses the Thompson
sampling algorithm to learn optimal actions. [14] takes a randomized-estimates policy to balance
exploration and exploitation. The main difference between [8, 34, 14] and our paper is that they
consider stochastic noise of Brownian motion which can be quite stringent and may fail in real-world
applications, while the noise in our setup is non-stochastic. This makes our analysis completely
different from theirs.

Domain Randomization
Domain randomization, which is proposed by [39], is a commonly used
technique for training agents to adapt to different (real) environments by training in randomized
simulated environments. From the empirical perspective, many previous works focus on designing
efficient algorithms for learning in a randomized simulated environment (by randomizing environmen-
tal settings, such as friction coefficient) such that the algorithm can adapt well in a new environment,
[29, 44, 26, 28, 30]. Other works study how to effectively randomize the simulated environment
so that the trained algorithm would generalize well in other environments [43, 27, 38]. However,
prior research has not explored how to apply certain theoretical analysis ideas to train agents in
domain-randomized environments. Limited previous works, such as [10] and [21], concentrate on
theoretically analyzing the sim-to-real gap within specific domain randomization models but they do
not test their algorithms in real domain randomization environments.

3
Problem Setting

In this paper, we consider the online non-stochastic control for continuous-time linear systems.
Therefore, we provide a brief overview below and define our notations.

3.1
Continuous-time Linear Systems

The Linear Dynamical System can be considered a specific case of a continuous Markov decision
process with linear transition dynamics. The state transitions are governed by the following equation:

˙xt = Axt + But + wt ,

where xt is the state at time t, ut is the action taken by the controller at time t, and wt represents
the disturbance at time t. Follow the setup of [3], we assume x0 = 0. We do not make any strong
assumptions about the distribution of wt, and we also assume that the distribution of wt is unknown
to the learner beforehand. This implies that the disturbance sequence wt can be selected adversarially.

When the action ut is applied to the state xt, a cost ct(xt, ut) is incurred. Here, we assume that the
cost function ct is convex. However, this cost is not known in advance and is only revealed after
the action ut is implemented at time t. In the system described above, an online policy π is defined
as a function that maps known states to actions, i.e., ut = π({xξ|ξ ∈[0, t]}). Our goal, then, is to
design an algorithm that determines such an online policy to minimize the cumulative cost incurred.
Specifically, for any algorithm A, the cost incurred over a time horizon T is:

JT (A) =
Z T

0
ct(xt, ut)dt .

In scenarios where the policy is linear (i.e., a linear controller), such that ut = −Kxt, we use J(K)
to denote the cost of a policy K ∈K from a certain class K.

3.2
Difference between Robust and Online Non-stochastic Control

While both robust and online non-stochastic control models incorporate adversarial noise, it’s crucial
to understand that their objectives differ significantly.

3


---Page Break---
The objective function for robust control, as seen in [37, 23], is defined as:
min
u1 max
w1:T min
u2 . . . min
ut max
wT JT (A) ,

Meanwhile, the objective function for online non-stochastic control, as discussed in [3], is:
min
A max
w1:T (JT (A) −min
K∈K JT (K)) .

Note that the robust control approach seeks to directly minimize the cost function, while online
non-stochastic control targets the minimization of regret, which is the discrepancy between the actual
cost and the cost associated with a baseline policy. Additionally, in robust control, the noise at each
step can depend on the preceding policy, whereas in online non-stochastic control, all the noise is
predetermined (though unknown to the player).

3.3
Assumptions

We operate under the following assumptions throughout this paper. To be concise, we denote ∥· ∥as
the L2 operator norm of the vector and matrix. Firstly, we make assumptions concerning the system
dynamics and noise:
Assumption 1. The matrices that govern the dynamics are bounded, meaning ∥A∥≤κA and
∥B∥≤κB, where κA and κB are constants. Moreover, the perturbation and its derivative are both
continuous and bounded: ∥wt∥, ∥˙wt∥≤W, with W being a constant.

These assumptions ensure that we can bound the states and actions, as well as their first and second-
order derivatives. Next, we make assumptions regarding the cost function:
Assumption 2. The costs ct(x, u) are convex in x and u. Additionally, if there exists a constant
D such that ∥x∥, ∥u∥≤D, then we have the following inequalities of the costs: |ct(x, u)| ≤
βD2, ∥∇xct(x, u)∥, ∥∇uct(x, u)∥≤GD, |ct1(x, u) −ct2(x, u)| ≤L|t1 −t2|D2,

where β,G and L are constants corresponding to the cost function. This assumption implies that if
the differences between states and actions are small, then the error in their cost will also be relatively
small.

3.4
Strongly Stable Policy

We next describe our baseline policy class introduced in [12]. Note that the continuous system and
the discrete system are different. If we consider the approximation over a relatively small interval h,
we get

xt+h = xt +
Z t+h

s=t
˙xsds =xt +
Z t+h

s=t
Axs + Bus + wsds

≈xt + h(Axt + But + wt) = (I + hA)xt + hBut + hwt .

Therefore, if we consider the transition of a discrete system xi+1 = ˜Axi + ˜Bui + ˜wi, we get
the approximation ˜A ≈I + hA, ˜B ≈hB. Hence, we extend the definition of a strongly stable
policy [12, 3] in the discrete system to the continuous system as follows:
Definition 1. A linear policy K is (κ, γ)-strongly stable if, for any h > 0 that is sufficiently small,
there exist matrices Lh, P such that I + h(A −BK) = PLhP −1, with the following two conditions:

1. The norm of Lh is strictly smaller than unity and dependent on h, i.e., ∥Lh∥≤1 −hγ.

2. The controller and transforming matrices are bounded, i.e., ∥K∥≤κ and ∥P∥,
P −1 ≤κ.

The above definition ensures the system can be stabilized by a linear controller K.

3.5
Regret Formulation

To evaluate the designed algorithm, we follow the setup in [12, 3] and use regret, which is defined
as the cumulative difference between the cost incurred by the policy of our algorithm and the cost
incurred by the best policy in hindsight. Let K denotes the class of strongly stable linear policies, i.e.
K = {K : K is (κ, γ)-strongly stable}. Then we try to minimize the regret of algorithm:
min
A max
w1:T Regret(A) = min
A max
w1:T (JT (A) −min
K∈K JT (K)) .

4


---Page Break---
4
Algorithm Design

In this section, we outline the design of our algorithm and formally define the concepts involved in
deriving our main theorem. We summarize our algorithm design as follows:

First, we discretize the total time period T into smaller intervals of length h. We use the information at
each point xh, x2h, . . . and uh, u2h, . . . to approximate the actual cost of each time interval, leveraging
the continuity assumption. This process does introduce some discretization errors.

Next, we employ the Disturbance-Action policy (DAC) [3]. This policy selects the action based on
the current time step and the estimations of disturbances from several past steps. This policy can
approximate the optimal linear policy in hindsight when we choose suitable parameters. However,
the optimal policy K∗is unknown, so we cannot directly acquire the optimal choice. To overcome
this, we employ the OCO with memory framework [5] to iteratively adjust the DAC policy parameter
Mt to approximate the optimal solution M ∗.

After that, we introduce the concept of the ideal state yt and ideal action vt that approximate the
actual state xt and action ut. Note that both the state and policy depend on all DAC policy parameters
M1, M2, . . . , Mt. Yet, the OCO with memory framework only considers the previous H steps.
Therefore, we need to consider ideal state and action. yt and vt represent the state the system would
reached if it had followed the DAC policy {Mt−H, . . . , Mt} at all time steps from t −H to t, under
the assumption that the state xt−H was 0.

From all the analysis above, we can decompose the regret as three parts: the discretization error R1,
the regret of the OCO with memory R2, and the approximation error between the ideal cost and the
actual cost R3.

Then we will formally introduce out method and define all the concepts. In the subsequent discussion,
we use shorthand notation to denote the cost, state, control, and disturbance variables cih, xih, uih,
and wih as ci, xi, ui, and wi, respectively.

First, we need to define the Disturbance-Action Policy Class(DAC) for continuous systems:
Definition 2. The Disturbance-Action Policy Class(DAC) is defined as:

ut = −Kxt +

l
X

i=1
M i
t ˆwt−i ,

where K is a fixed strongly stable policy, l is a parameter that signifies the dimension of the policy
class, Mt = {M 1
t , . . . , M l
t} is the weighting parameter of the disturbance at step t, and ˆwt is the
estimated disturbance:

ˆwt = xt+1 −xt −h(Axt + But)

h
.
(1)

We note that this definition differs from the DAC policy in discrete systems [3] as we utilize the
estimation of disturbance over an interval [t, t + h] instead of only the noise in time t. It counteracts
the second-order residue term of the Taylor expansion of xt and is also an online policy as it only
requires information from the previous state.

Our higher-level controller adopts the OCO with memory framework. A technical challenge lies in
balancing the approximation error and OCO regret. To achieve a low approximation error, we desire
the policy update interval H to be inversely proportional to the sampling distance h. However, this
relationship may lead to large OCO regret. To mitigate this issue, we introduce a new parameter
m = Θ( 1

h), representing the lookahead window. We update the parameter Mt only once every m
iterations, further reducing the OCO regret without negatively impacting the approximation error:

Mt+1 =
ΠM (Mt −η∇gt(M))
if t mod m == 0 ,
Mt
otherwise .

Where gt is a function corresponding to the loss function ct and we will introduce later in Algorithm 1.
For notational convenience and to avoid redundancy, we denote ˜
M[t/m] = Mt. We can then define
the ideal state and action. Due to the properties of the OCO with memory structure, we need to
consider only the previous Hm states and actions, rather than all states. As a result, we introduce
the definition of the ideal state and action. During the interval t ∈[im, (i + 1)m −1], the learning
policy remains unchanged, so we could define the ideal state and action follow the definition in [3]:

5


---Page Break---
Definition 3. The ideal state yt and action vt at time t ∈[im, (i + 1)m −1] are defined as

yt = xt( ˜
Mi−H, ..., ˜
Mi), vt = −Kyt +

l
X

j=1
M j
i wt−i .

where the notation indicates that we assume the state xt−H is 0 and that we apply the DAC policy

˜
Mi−H, . . . , ˜
Mi

at all time steps from t −Hm to t.

We can also define the ideal cost in this interval follow the definition in [3]:

Definition 4. The ideal cost function during the interval t ∈[im, (i + 1)m −1] is defined as follows:

fi

˜
Mi−H, . . . , ˜
Mi

=

(i+1)m−1
X

t=im
ct

yt

˜
Mi−H, . . . , ˜
Mi

, vt

˜
Mi−H, . . . , ˜
Mi

.

With all the concepts presented above, we are now prepared to introduce our algorithm:

Algorithm 1 Continuous two-level online control algorithm

Input: step size η, sample distance h, policy update parameters H, m, parameters κ, γ, T.
Define sample numbers n = ⌈T/h⌉, OCO policy update times p = ⌈n/m⌉.

Define DAC policy update class M =
n
˜
M =
n
˜
M 1 . . . ˜
M Hmo
:
 ˜
M i ≤2hκ3(1 −γ)i−1o
.
Initialize M0 ∈M arbitrarily.
for k = 0, . . . , p −1 do

for s = 0, . . . , m −1 do

Denote the discretization time r = km + s.
Use the action ut = −Kxr + h PHm
i=1 ˜
M i
k ˆwr−i during the time period t ∈[rh, (r + 1)h].
Observe the new state xr+1 at time (r + 1)h and record ˆwr according to Equation (1).
end for
Define the function gk(M) = fk(M, . . . , M).

Update OCO policy ˜
Mk+1 = ΠM

˜
Mk −η∇gk( ˜
Mk)

.
end for

5
Main Result

In this section, we present the primary theorem of online continuous control regret analysis:

Theorem 1. Under Assumption 1, 2, a step size of η = Θ(p m

T h), and a DAC policy update frequency
m = Θ( 1

h), Algorithm 1 attains a regret bound of

JT (A) −min
K∈K JT (K) ≤O(nh(1 −hγ)
H

h ) + O(
√

nh) + O(Th) .

With the sampling distance h = Θ( 1
√

T ), and the OCO policy update parameter H = Θ(log(T)),
Algorithm 1 achieves a regret bound of

JT (A) −min
K∈K JT (K) ≤O
√

T log (T)

.

Theorem 1 demonstrates a regret that matches the regret of a discrete system [3]. Despite the analysis
of a continuous system differing from that of a discrete system, we can balance discretization error,
approximation error, and OCO with memory regret by selecting an appropriate update frequency for
the policy. Here, O(·) and Θ(·) are abbreviations for the polynomial factors of universal constants in
the assumption.

While we defer the detailed proof to the appendix, we outline the key ideas and highlight them below.

6


---Page Break---
Figure 1: Bounding the states and their derivatives
separately. We employ Gronwall’s inequality with
the induction method to bound the states.

Challenge and Proof Sketch
We first explain why
we cannot directly apply the methods for discrete
nonstochastic control from [3] to our work. To utilize
Assumption 2, it is necessary first to establish a union
bound over the states. In a discrete-time system, it
can be easily proved by applying the dynamics in-
equality ∥xt+1∥≤a∥xt∥+ b (where a < 1) and the
induction method presented in [3]. However, for a
continuous-time system, a different approach is nec-
essary because we only have the differential equation
instead of the state recurrence formula.

To overcome this challenge, we employ Gronwall’s
inequality to bound the first and second-order derivatives in the neighborhood of the current state.
We then use these bounded properties, in conjunction with an estimation of previous noise, to bound
the distance to the next state. Through an iterative application of this method, we can argue that all
states and actions are bounded.

Another challenge is that we need to discretize the system but we must overcome the curse of
dimensionality caused by discretization. In continuous-time systems, the number of states is inversely
proportional to the discretization parameter h, which also determines the size of the OCO memory
buffer. Our regret is primarily composed of three components: the error caused by discretization R1,
the regret of OCO with memory R2 and the difference between the actual cost and the approximate
cost R3. The discretization error R1 is O(hT), therefore if we achieve O(
√

T) regret, we must
choose h no more than O( 1
√

T ).

If we update the OCO with memory parameter at each timestep follow the method in [3], we will
incur the regret of OCO with memory R2 = O(H2.5√

T). The difference between the actual cost
and the approximate cost R3 = O(T(1 −hγ)H). To achieve sublinear regret for the third term,
we must choose H = O( log T

hγ ), but since h is no more than O( 1
√

T ), H will be larger than Θ(
√

T),

therefore the second term R2 will definitely exceed O(
√

T).

Therefore, we adjust the frequency of updating the OCO parameters by introducing a new parameter
m, using a two-level approach and update the OCO parameters once in every m steps. This will incur
the third term R3 = O(T(1−hγ)Hm) but keep the OCO with memory regret R2 = O(H2.5√

T), so
we can choose H = O( log T

γ
) and m = O( 1

h). Then the term of R2 is O(
√

T log T) and we achieve
the same regret compare with the discrete system.

6
Experiments

In this section, we apply our theoretical analysis to the practical training of agents. First we highlight
the key difference between our algorithm and traditional online policy optimization.

1. Stack: While standard online policy optimization learns the optimal policy from the current
state ut = ϕ(xt), an optimal non-stochastic controller employs the DAC policy as outlined
in Definition 2. Leveraging information from past states aids the agent in adapting to
dynamic environments.
2. Skip: Different from the analysis in [3], in a continuous-time system we update the state
information every few steps, rather than updating it at every step. This solves the curse of
dimensionality caused by discretization in continuous-time system.

The above inspires us with an intuitive strategy for training agents by stacking past observations
with some observations to skip. We denote this as Stack & skip for convenience. Stack & skip is
frequently used as a heuristic in reinforcement learning, yet little was known about when and why
such a technique could boost agent performance.

How should we evaluate our algorithm in a non-stochastic environment? We opt for learning
an optimal policy within a domain randomization environment. In this context, each model’s
parameters are randomly sampled from a predetermined task distribution. We train policies to
optimize performance across various simulated models [41, 29].

7


---Page Break---
Figure 2: Leverage past observation of states with some skip.

We observe that learning in Do-
main Randomization (DR) sig-
nificantly differs from stochas-
tic or robust learning problems.
In DR, sampling from environ-
mental variables occurs at the be-
ginning of each episode, rather
than at every step, distinguish-
ing it from stochastic learning
where randomness is step-wise
independent and identically dis-
tributed. This episodic sampling
approach allows agents in DR to
exploit environmental conditions
and adapt to episodic changes
within an episode. On the other hand, robust learning focuses on worst-case scenarios depend-
ing on an agent’s policy. DR, in contrast, is concerned with the distribution of conditions aimed at
broad applicability rather than worst-case perturbations.

In the context of non-stochastic control, the disturbance, while not disclosed to the learner beforehand,
remains fixed throughout the episode and does not adaptively respond to the control policy. This
setup in non-stochastic control shows a clear parallel to domain randomization: fixed yet unknown
disturbances in non-stochastic control mirror the unknown training environments in DR. As the agent
continually interacts with these environments, it progressively adapts, mirroring the adaptive process
observed in domain randomization. Therefore, we propose evaluating our algorithm within a domain
randomization training task. Subsequently, we introduce the details of our experimental setup:

Environment
Parameters
DR distribution

Hopper

Joint damping
[0.5, 1.5]
Foot friction
[1, 3]
Height of head
[1.2, 1.7]
Torso size
[0.025, 0.075]

Half-Cheetah

Joint damping
[0.005, 0.015]
Foot friction
[3, 7]
Torso size
[0.04, 0.06]

Walker2D

Joint damping
[0.05, 0.15]
Density
[500, 1500]
Torso size
[0.025, 0.075]

Table 1: The DR distributions of environment.

Environment Setting
We conduct
experiments on the hopper, half-
cheetah, and walker2d benchmarks us-
ing the MuJoCo simulator [40]. The
randomized parameters include envi-
ronmental physical parameters such
as damping and friction, as well as
the agent properties such as torso size.
We set the range of our domain ran-
domization to follow a distribution
with default parameters as the mean
value, shown in Table 1. When train-
ing in the domain randomization en-
vironment, the parameter is uniformly
sampled from this distribution. To an-
alyze the result of generalization, we only change one of the parameters and keep the other parameters
as the mean of its distribution in each test environment. We conducted experiments using NVIDIA
A40 graphics card.

Algorithm Design and Baseline
We design a practical meta-algorithm that converts any standard
deep RL algorithm into a domain-adaptive algorithm, shown in Figure 2. In this algorithm, we
augment the original state observation oold
t
at time t with past observations, resulting in onew
t
=
[oold
t
, oold
t−m, . . . , oold
t−(h−1)m]. Here h is the number of past states we leverage and m is the number
of states we skip when we get each of the past states. For clarity in our results, we selected the
SAC algorithm for evaluation. We use a variant of Soft Actor-Critic (SAC) [31] and leverage past
states with some skip as our algorithm. We compare our algorithm with the standard SAC algorithm
training on domain randomization environments as our baseline.

Impact of Frame Stack and Frame Skip
To understand the effects of the frame stack number h
and frame skip number m, we carried out experiments in the hopper environment with different h
and m. For each parameter we train with 3 random seeds and take the average. Figure 3 shows that
the performance increases significantly when the frame stack number is increased from 1 to 3, and

8


---Page Break---
remains roughly unchanged when the frame stack number continues to climb up. Figure 4 shows
that the optimal frame skip number is 3, while both too large or too small frame skip numbers result
in sub-optimal results. Therefore, in the following experiments we fix the parameter h = 3, m = 3.
We train our algorithm with this parameter and standard SAC on hopper and test the performance on
more environments. Figure 5 shows that our algorithm outperforms the baseline in all environments.

Figure 3: Impact of frame stack number.
Figure 4: Impact of frame skip number.

Figure 5: Agents’ reward in various test environments of hopper.

Results on Other Environments
Each algorithm was trained using three distinct random seeds in
the half-cheetah and walker2d domain randomization (DR) environments. Consistent with previous
experiments, we employed a frame stack number of h = 3 and frame skip number of m = 3.
The comparative performance of our algorithm and the baseline algorithm, across various domain
parameters, is presented in Figure 6. The result clearly demonstrates that our algorithm consistently
outperforms the baseline in all evaluated test environments.

Figure 6: Performance in half-cheetah(Top) and walker2d(Bottom).

9


---Page Break---
7
Conclusion, Limitations and Future Directions

In this paper, we propose a two-level online controller for continuous-time linear systems with adver-
sarial disturbances, aiming to achieve sublinear regret. This approach is grounded in our examination
of agent training in domain randomization environments from an online control perspective. At the
higher level, our controller employs the Online Convex Optimization (OCO) with memory framework
to update policies at a low frequency, thus reducing regret. The lower level uses the DAC policy to
align the system’s actual state more closely with the idealized setting.

In our empirical evaluation, applying our algorithm’s core principles to the SAC (Soft Actor-Critic)
algorithm led to significantly improved results in multiple reinforcement learning tasks within domain
randomization environments. This highlights the adaptability and effectiveness of our approach in
practical scenarios.

It is important to note that our theoretical analysis depends on the known dynamics of the system
and the assumption of convex costs. This reliance could represent a limitation to our method, as it
may not adequately address scenarios where these conditions do not hold or where system dynamics
are incompletely understood. For future research, there are several promising directions in online
non-stochastic control of continuous-time systems. These include extending our methods to systems
with unknown dynamics, exploring the impact of assuming strong convexity in cost functions, and
shifting the focus from regret to the competitive ratio. Further research can also explore how to
utilize historical information more effectively to enhance agent training in domain randomization
environments. This might involve employing time series analysis instead of simply incorporating
parameters into neural network training.

References

[1] Yasin Abbasi-Yadkori, Peter Bartlett, and Varun Kanade. Tracking adversarial targets. In
International Conference on Machine Learning, 2014.

[2] Yasin Abbasi-Yadkori and Csaba Szepesvári. Regret bounds for the adaptive control of linear
quadratic systems. In Proceedings of the 24th Annual Conference on Learning Theory, 2011.

[3] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control
with adversarial disturbances. In International Conference on Machine Learning, 2019.

[4] Artemij Amiranashvili, Max Argus, Lukas Hermann, Wolfram Burgard, and Thomas Brox.
Pre-training of deep rl agents for improved learning under domain randomization. arXiv preprint
arXiv:2104.14386, 2021.

[5] Oren Anava, Elad Hazan, and Shie Mannor. Online learning for adversaries with memory: price
of past mistakes. Advances in Neural Information Processing Systems, 2015.

[6] Lachlan Andrew, Siddharth Barman, Katrina Ligett, Minghong Lin, Adam Meyerson, Alan
Roytman, and Adam Wierman. A tale of two metrics: Simultaneous bounds on competitiveness
and regret. In Conference on Learning Theory, pages 741–763. PMLR, 2013.

[7] Michael Athans. The role and use of the stochastic linear-quadratic-gaussian problem in control
system design. IEEE transactions on automatic control, 16(6):529–552, 1971.

[8] Matteo Basei, Xin Guo, Anran Hu, and Yufei Zhang. Logarithmic regret for episodic continuous-
time linear-quadratic reinforcement learning over a finite-time horizon. Journal of Machine
Learning Research, 2022.

[9] Marco C Campi and PR Kumar. Adaptive linear quadratic gaussian control: the cost-biased
approach revisited. SIAM Journal on Control and Optimization, 36(6):1890–1907, 1998.

[10] Xiaoyu Chen, Jiachen Hu, Chi Jin, Lihong Li, and Liwei Wang. Understanding domain
randomization for sim-to-real transfer. In International Conference on Learning Representations,
2022.

[11] Xinyi Chen and Elad Hazan. Black-box control for linear dynamical systems. In Conference on
Learning Theory, 2021.

10


---Page Break---
[12] Alon Cohen, Avinatan Hasidim, Tomer Koren, Nevena Lazic, Yishay Mansour, and Kunal
Talwar. Online linear quadratic control. In International Conference on Machine Learning,
2018.

[13] Tyrone E Duncan, Petr Mandl, and Bo˙zenna Pasik-Duncan. On least squares estimation in
continuous time linear stochastic systems. Kybernetika, 28(3):169–180, 1992.

[14] Mohamad Kazem Shirani Faradonbeh and Mohamad Sadegh Shirani Faradonbeh. Online
reinforcement learning in stochastic continuous-time systems. In The Thirty Sixth Annual
Conference on Learning Theory, pages 612–656. PMLR, 2023.

[15] Gautam Goel, Naman Agarwal, Karan Singh, and Elad Hazan. Best of both worlds in online
control: Competitive ratio and policy regret. arXiv preprint arXiv:2211.11219, 2022.

[16] Gautam Goel and Adam Wierman. An online algorithm for smoothed regression and lqr control.
In The 22nd International Conference on Artificial Intelligence and Statistics, pages 2504–2513.
PMLR, 2019.

[17] Graham C Goodwin, Peter J Ramadge, and Peter E Caines. Discrete time stochastic adaptive
control. SIAM Journal on Control and Optimization, 19(6):829–853, 1981.

[18] Elad Hazan. Introduction to online convex optimization. CoRR, abs/1909.05207, 2019.

[19] Elad Hazan, Sham Kakade, and Karan Singh. The nonstochastic control problem. In Algorithmic
Learning Theory, 2020.

[20] Sebastian Höfer, Kostas Bekris, Ankur Handa, Juan Camilo Gamboa, Melissa Mozifian, Florian
Golemo, Chris Atkeson, Dieter Fox, Ken Goldberg, John Leonard, et al. Sim2real in robotics
and automation: Applications and challenges. IEEE transactions on automation science and
engineering, 2021.

[21] Jiachen Hu, Han Zhong, Chi Jin, and Liwei Wang. Provable sim-to-real transfer in continuous
domain with partial observations. arXiv preprint arXiv:2210.15598, 2022.

[22] Yu Jiang and Zhong-Ping Jiang. Computational adaptive optimal control for continuous-time
linear systems with completely unknown dynamics. Automatica, 2012.

[23] IS Khalil, JC Doyle, and K Glover. Robust and optimal control. Prentice hall, 1996.

[24] PR Kumar. Optimal adaptive control of linear-quadratic-gaussian systems. SIAM Journal on
Control and Optimization, 21(2):163–178, 1983.

[25] Lennart Ljung. System identification. Springer, 1998.

[26] Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher J Pal, and Liam Paull. Active
domain randomization. In Conference on Robot Learning, pages 1162–1176. PMLR, 2020.

[27] Melissa Mozian, Juan Camilo Gamboa Higuera, David Meger, and Gregory Dudek. Learning
domain randomization distributions for training robust locomotion policies. In 2020 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS), pages 6112–6117. IEEE,
2020.

[28] Fabio Muratore, Christian Eilers, Michael Gienger, and Jan Peters. Data-efficient domain
randomization with bayesian optimization. IEEE Robotics and Automation Letters, 6(2):911–
918, 2021.

[29] Fabio Muratore, Michael Gienger, and Jan Peters. Assessing transferability from simulation
to reality for reinforcement learning. IEEE transactions on pattern analysis and machine
intelligence, 43(4):1172–1183, 2019.

[30] Fabio Muratore, Theo Gruner, Florian Wiese, Boris Belousov, Michael Gienger, and Jan Peters.
Neural posterior domain randomization. In Conference on Robot Learning, pages 1532–1542.
PMLR, 2022.

11


---Page Break---
[31] Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, and Aaron Courville.
The primacy bias in deep reinforcement learning. In International Conference on Machine
Learning. PMLR, 2022.

[32] Syed Ali Asad Rizvi and Zongli Lin. Output feedback reinforcement learning control for
the continuous-time linear quadratic regulator problem. In 2018 Annual American Control
Conference (ACC), 2018.

[33] Guanya Shi, Yiheng Lin, Soon-Jo Chung, Yisong Yue, and Adam Wierman. Online optimization
with memory and competitive control. Advances in Neural Information Processing Systems,
33:20636–20647, 2020.

[34] Mohamad Kazem Shirani Faradonbeh, Mohamad Sadegh Shirani Faradonbeh, and Mohsen
Bayati. Thompson sampling efficiently learns to control diffusion processes. Advances in
Neural Information Processing Systems, 35:3871–3884, 2022.

[35] Max Simchowitz. Making non-stochastic control (almost) as easy as stochastic. Advances in
Neural Information Processing Systems, 33:18318–18329, 2020.

[36] Max Simchowitz, Karan Singh, and Elad Hazan. Improper learning for non-stochastic control.
In Conference on Learning Theory, pages 3320–3436. PMLR, 2020.

[37] Robert F Stengel. Optimal control and estimation. Courier Corporation, 1994.

[38] Gabriele Tiboni, Karol Arndt, and Ville Kyrki. Dropo: Sim-to-real transfer with offline domain
randomization. Robotics and Autonomous Systems, 166:104432, 2023.

[39] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel.
Domain randomization for transferring deep neural networks from simulation to the real world.
In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages
23–30. IEEE, 2017.

[40] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based
control. 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages
5026–5033, 2012.

[41] Jonathan Tremblay, Aayush Prakash, David Acuna, Mark Brophy, Varun Jampani, Cem Anil,
Thang To, Eric Cameracci, Shaad Boochoon, and Stan Birchfield. Training deep networks with
synthetic data: Bridging the reality gap by domain randomization. In Proceedings of the IEEE
conference on computer vision and pattern recognition workshops, pages 969–977, 2018.

[42] Draguna Vrabie, O Pastravanu, Murad Abu-Khalaf, and Frank L Lewis. Adaptive optimal
control for continuous-time linear systems based on policy iteration. Automatica, 2009.

[43] Quan Vuong, Sharad Vikram, Hao Su, Sicun Gao, and Henrik I Christensen. How to pick the
domain randomization parameters for sim-to-real transfer of reinforcement learning policies?
arXiv preprint arXiv:1903.11774, 2019.

[44] Sergey Zakharov, Wadim Kehl, and Slobodan Ilic. Deceptionnet: Network-driven domain
randomization. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pages 532–541, 2019.

12


---Page Break---
In the appendix we define n as the smallest integer greater than or equal to T

h , and we use the
shorthand cih, xih, uih, and wih as ci, xi, ui, and wi, respectively. First we provide the proof of our
main theorem there.

A
Proof of Theorem 1

Theorem 1. Under Assumption 1, 2, a step size of η = Θ(p m

T h), and a DAC policy update frequency
m = Θ( 1

h), Algorithm 1 attains a regret bound of

JT (A) −min
K∈K JT (K) ≤O(nh(1 −hγ)
H

h ) + O(
√

nh) + O(Th) .

With the sampling distance h = Θ( 1
√

T ), and the OCO policy update parameter H = Θ(log(T)),
Algorithm 1 achieves a regret bound of

JT (A) −min
K∈K JT (K) ≤O
√

T log (T)

.

Proof. We denote u∗
t = K∗x∗
t as the optimal state and action that follows the policy specified by
K∗, where K∗= arg maxK∈K JT (K).

We then discretize and decompose the regret as follows:

JT (A) −min
K∈K JT (K) =
Z T

0
ct(xt, ut)dt −
Z T

0
ct(x∗
t , u∗
t )dt

=

n−1
X

i=0

Z (i+1)h

ih
ct(xt, ut)dt −

n−1
X

i=0

Z (i+1)h

ih
ct(x∗
t , u∗
t )dt

= h

 n−1
X

i=0
ci(xi, ui) −

n−1
X

i=0
ci(x∗
i , u∗
i )

!

+ R0 ,

where R0 represents the discretization error.

We define p as the smallest integer greater than or equal to n

m, then the first term can be further
decomposed as

n−1
X

i=0
ci(xi, ui) −

n−1
X

i=0
ci(x∗
i , u∗
i )

=

p−1
X

i=0

(i+1)m−1
X

j=im
ci(xi, ui) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i )

=

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −

(i+1)m−1
X

j=im
ci(yi, vi)



+

p−1
X

i=0

(i+1)m−1
X

j=im
ci(yi, vi) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i )

=

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −fi( ˜
Mi−H, . . . , ˜
Mi)



+

p−1
X

i=0
fi( ˜
Mi−H, . . . , ˜
Mi)

−min
M∈M

p−1
X

i=0
fi(M, . . . , M) + min
M∈M

p−1
X

i=0
fi(M, . . . , M) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i ) ,

where the last equality is by the definition of the idealized cost function.

13


---Page Break---
Let us denote

R1 =

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −fi( ˜
Mi−H, . . . , ˜
Mi)



,

R2 =

p−1
X

i=0
fi( ˜
Mi−H, . . . , ˜
Mi) −min
M∈M

p−1
X

i=0
fi(M, . . . , M) ,

R3 = min
M∈M

p−1
X

i=0
fi(M, . . . , M) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i ) .

Then we have the regret decomposition as

Regret(T) = h(R1 + R2 + R3) + O(hT) .

We then separately upper bound each of the four terms.

The term R0 represents the error caused by discretization, which decreases as the number of sampling
points increases and the sampling distance h decreases. This is because more sampling points make
our approximation of the continuous system more accurate. Using Lemma 3, we get the following
upper bound: R0 ≤O(hT).

The term R1 represents the difference between the actual cost and the approximate cost. For a fixed
h, this error decreases as the number of sample points looked ahead m increases, while it increases
as the sampling distance h decreases. This is because the closer adjacent points are, the slower the
convergence after approximation. By Lemma 4 we can bound it as R1 ≤O(n(1 −hγ)Hm).

The term R2 is incurred due to the regret of the OCO with memory algorithm. Note that this term is
determined by learning rate η and the policy update frequency m. Choosing suitable parameters and
using Lemma 5, we can obtain the following upper bound: R2 ≤O(
p

n/h).

The term R3 represents the difference between the ideal optimal cost and the actual optimal cost.
Since the accuracy of the DAC policy approximation of the optimal policy depends on its degree of
freedom l, a higher degree of freedom leads to a more accurate approximation of the optimal policy.
We use Lemma 6 and choose l = Hm to bound this error: R3 ≤O(n(1 −hγ)Hm).

By summing up these four terms and taking m = Θ( 1

h), we get:

Regret(T) ≤O(nh(1 −hγ)
H

h ) + O(
√

nh) + O(hT) .

Finally, we choose h = Θ

1
√

T


, m = Θ
  1

h

, H = Θ(log(T)), the regret is bounded by

Regret(T) ≤O(
√

T log(T)) .

B
Key Lemmas

In this section, we will primarily discuss the rationale behind the proof of our key lemmas. First, we
need to prove all the states and actions are bounded.

Lemma 2. Under Assumption 1 and 2, choosing arbitrary h in the interval [0, h0] where h0 is
a constant only depends on the parameters in the assumption, we have for any t and policy Mi,
∥xt∥, ∥yt∥, ∥ut∥, ∥vt∥≤D, ∥˙xt∥≤D, ∥xt −yt∥, ∥ut −vt∥≤κ2(1 + κ)(1 −hγ)Hm+1D. In
particular, taking all the Mt = 0 and K = K∗, we can also obtain the inequality of the optimal
solution: ∥x∗
t ∥, ∥u∗
t ∥≤D.

The proof of this Lemma mainly use the Gronwall inequality and the induction method. Then we
analyze the discretization error of the system.

14


---Page Break---
Lemma 3. Under Assumption 2, Algorithm 1 attains the following bound of R0:

R0 =

n−1
X

i=0

Z (i+1)h

ih
(ct(xt, ut) −ct(x∗
t , u∗
t ))dt −h

n−1
X

i=0
(ci(xi, ui) −ci(x∗
i , u∗
i )) ≤(G + L)D2hT .

This lemma indicates that the discretization error is directly proportional to the sample distance h. In
other words, increasing the number of sampling points leads to more accurate estimation of system.

Then we analysis the difference between ideal cost and actual cost. The following lemma describes
the upper bound of the error by approximating the ideal state and action:

Lemma 4. Under Assumption 1 and 2, Algorithm 1 attains the following bound of R1:

R1 =

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −fi

˜
Mi−H, . . . , ˜
Mi



≤nGD2κ2(1 + κ)(1 −hγ)Hm+1 .

From this lemma, it is evident that for a fixed sample distance h, the error diminishes as the number
of sample points looked ahead m increases. However, as the sampling distance h decreases, the
convergence rate of this term becomes slower. Therefore, it is not possible to select an arbitrarily
small value for h in order to minimize the discretization error R0.

We need to demonstrate that the discrepancy between xt and yt, as well as ut and vt, is sufficiently
small, given assumption 1. This can be proven by analyzing the state evolution under the DAC policy.

By utilizing Assumption 2 and Lemma 2, we can deduce the following inequality:

|ct (xt, ut) −ct (yt, vt)| ≤|ct (xt, ut) −ct (yt, ut)| + |ct (yt, ut) −ct (yt, vt)|
≤GD∥xt −yt∥+ GD∥ut −vt∥.

Summing over all the terms and use Lemma 2, we can derive an upper bound for R1.

Next, we analyze the regret of Online Convex Optimization (OCO) with a memory term. To analyze
OCO with a memory term, we provide an overview of the framework established by [5] in online
convex optimization. The framework considers a scenario where, at each time step t, an online player
selects a point xt from a set K ⊂Rd. At each time step, a loss function ft : KH+1 →R is revealed,
and the player incurs a loss of ft (xt−H, . . . , xt). The objective is to minimize the policy regret,
which is defined as

PolicyRegret =

T
X

t=H
ft (xt−H, . . . , xt) −min
x∈K

T
X

t=H
ft(x, . . . , x) .

In this setup, the first term corresponds to the DAC policy we choose, while the second term is used
to approximate the optimal strongly stable linear policy.

Lemma 5. Under Assumption 1 and 2, choosing m = C

h and η = Θ( m

T h), Algorithm 1 attains the
following bound of R2:

R2 =

p−1
X

i=0
fi( ˜
Mi−H, . . . , ˜
Mi) −min
M∈M

p−1
X

i=0
fi(M, . . . , M)

≤4a

γ

s

GDC2κ2(κ + 1)W0κB

γ
(GDCκ2(κ + 1)W0κB

γ
+ C2κ3κBW0H2)n

h .

To analyze this term, we can transform the problem into an online convex optimization with memory
and utilize existing results presented by [5] for it. By applying their results, we can derive the
following bound:

T
X

t=H
ft (xt−H, . . . , xt) −min
x∈K

T
X

t=H
ft(x, . . . , x) ≤O

D
q

Gf (Gf + LH2) T

.

15


---Page Break---
Taking into account the bounds on the diameter, Lipschitz constant, and the gradient, we can ultimately
derive an upper bound for R2.

Lastly, we aim to establish a bound on the approximation error between the optimal DAC policy and
the unknown optimal linear policy.

Lemma 6. Under Assumption 1 and 2, Algorithm 1 attains the following bound of R3:

R3 = min
M∈M

p−1
X

i=0
fi(M, ..., M) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i ) ≤3n(1 −hγ)HmGDW0κ3a(lhκB + 1) .

The intuition behind this lemma is that the evolution of states leads to an approximation of the
optimal linear policy in hindsight, where u∗
t = −K∗xt if we choose M ∗= {M i}, where M i =
(K −K∗)(I + h(A −BK∗))i. Although the optimal policy K∗is unknown, such an upper bound
is attainable because the left-hand side represents the minimum of M ∈M.

C
The evolution of the state

In this section we will prove that using the DAC policy, the states and actions are uniformly bounded.
The difference between ideal and actual states and the difference between ideal and actual action is
very small.

We begin with expressions of the state evolution using DAC policy:

Lemma 7. We have the evolution of the state and action:

xt+1 = Ql+1
h
xt−l + h

2l
X

i=0
Ψt,i ˆwt−i ,

yt+1 = h

2Hm
X

i=0
Ψt,i ˆwt−i ,

vt = −Kyt + h

Hm
X

j=1
M j
t ˆwt−j .

where Ψt,i represent the coefficients of ˆwt−i:

Ψt,i = Qi
h1i≤l + h

l
X

j=0
Qj
hBM i−j
t−j 1i−j∈[1,l] .

Proof. Define Qh = I + h(A −BK). Using the Taylor expansion of xt and denoting rt as the
second-order residue term, we have

xt+1 = xt + h ˙xt + h2rt = xt + h(Axt + But + wt) + h2rt .

Then we calculate the difference between wi and ˆwi:

ˆwt −wt = xt+1 −xt −h(Axt + But + wt)

h
= hrt .

16


---Page Break---
Using the definition of DAC policy and the difference between disturbance, we have

xt+1 = xt + h

 

Axt + B

 

−Kxt + h

l
X

i=1
M i
t ˆwt−i

!

+ ˆwt −hrt

!

+ h2rt

= (I + h(A −BK))xt + h

 

Bh

l
X

i=1
M i
t ˆwt−i + ˆwt

!

= Qhxt + h

 

Bh

l
X

i=1
M i
t ˆwt−i + ˆwt

!

= Q2
hxt−1 + h

 

Qh

 

Bh

l
X

i=1
M i
t−1 ˆwt−1−i + ˆwt−1

!!

+ h

 

Bh

l
X

i=1
M i
t ˆwt−i + ˆwt

!

= Ql+1
h
xt−l + h

2l
X

i=0
Ψt,i ˆwt−i ,

where the last equality is by recursion and Ψt,i represent the coefficients of ˆwt−i.

Then we calculate the coefficients of wt−i and get the following result:

Ψt,i = Qi
h1i≤l + h

l
X

j=0
Qj
hBM i−j
t−j 1i−j∈[1,l] .

By the ideal definition of yt+1 and vt(only consider the effect of the past Hm steps while planning,
assume xt−Hm = 0), taking l = Hm we have

yt+1 = h

2Hm
X

i=0
Ψt,i ˆwt−i,

vt = −Kyt + h

Hm
X

j=1
M j
t ˆwt−j .

Then we prove the norm of the transition matrix is bounded.

Lemma 8. We have the following bound of the transition matrix:

∥Ψt,i∥≤a(lhκB + 1)κ2(1 −hγ)i−1 .

Proof. By the definition of strongly stable policy, we know

∥Qi
h∥= ∥(PLhP −1)i∥= ∥P(Lh)iP −1∥≤∥P∥∥Lh∥i∥P −1∥≤aκ2(1 −hγ)i .
(2)

By the definition of Ψt,i, we have

∥Ψt,i∥=


Qi
h1i≤l + h

l
X

j=0
Qj
hBM i−j
t−j 1i−j∈[1,l]



≤κ2(1 −hγ)i + ah

l
X

j=1
κBκ2(1 −hγ)j(1 −hγ)i−j−1

≤κ2(1 −hγ)i + alhκBκ2(1 −hγ)i−1 ≤a(lhκB + 1)κ2(1 −hγ)i−1 ,

where the first inequality is due to equation 2, assumption 1 and the condition of
M i
t
 ≤a(1 −
hγ)i−1.

After that, we can uniformly bound the state xt and its first and second-order derivative.

17


---Page Break---
Lemma 9. For any t ∈[0, T], choosing arbitrary h in the interval [0, h0] where h0 is a constant only
depends on the parameters in the assumption, we have ∥xt∥≤D1, ∥˙xt∥≤D2, ∥¨xt∥≤D3 and the
estimatation of disturbance is bounded by ∥ˆwt∥≤W0. Moreover, D1, D2, D3 are only depend on
the parameters in the assumption.

Proof. We prove this lemma by induction. When t = 0, it is clear that x0 satisfies this condition.
Suppose xt ≤D1, ˙xt ≤D2, ¨xt ≤D3, ˆwt ≤W0 for any t ≤t0, where t0 = kh is the k-th
discretization point. Then for t ∈[t0, t0 + h], we first prove that ˙xt ≤D2, ¨xt ≤D3.

By Assumption 1 and our definition of ut, we know that for any t ∈[t0, t0 + h]. Thus, we have

∥˙xt∥= ∥Axt + But + wt∥

= ∥Axt + B(−Kxt0 + h

l
X

i=1
M i
k ˆwk−i) + wt∥

≤κA∥xt∥+ κBκ∥xt0∥+ h

l
X

i=1
(1 −hγ)i−1W0 + W

≤κA∥xt∥+ κBκD1 + W0

γ
+ W ,

where the first inequality is by the induction hypothesis ˆwt ≤W0 for any t ≤t0 and M i
k ≤
(1 −hγ)i−1, the second inequality is by the induction hypothesis xt ≤D1 for any t ≤t0.

For any t ∈[t0, t0 + h], because we choose the fixed policy ut ≡ut0, so we have ˙ut = 0 and

∥¨xt∥= ∥A ˙xt + B ˙ut + ˙wt∥= ∥A ˙xt + ˙wt∥≤κA∥˙xt∥+ W .

By the Newton-Leibniz formula, we have for any ζ ∈[0, h],

˙xt0+ζ −˙xt0 =
Z ζ

0
¨xt0+ξdξ .

Then we have

∥˙xt0+ζ∥≤∥˙xt0∥+
Z ζ

0
∥¨xt0+ξ∥dξ

≤∥˙xt0∥+
Z ζ

0
(κA∥˙xt0+ξ∥+ W)dξ

= ∥˙xt0∥+ Wζ + κA

Z ζ

0
∥˙xt0+ξ∥dξ .

By Gronwall inequality, we have

∥˙xt0+ζ∥≤∥˙xt0∥+ Wζ +
Z ζ

0
(∥˙xt0∥+ Wξ) exp(κA(ζ −ξ))dξ .

Then we have

∥˙xt0+ζ∥≤∥˙xt0∥+ Wζ +
Z ζ

0
(∥˙xt0∥+ Wζ) exp(κAζ))dξ

= (∥˙xt0∥+ Wζ)(1 + ζ exp(κAζ))

≤

κA∥xt0∥+ κBκD1 + W0

γ
+ W + Wh

(1 + h exp(κAh))

≤

(κA + κBκ)D1 + W0

γ
+ W + Wh

(1 + h exp(κAh))

≤

(κA + κBκ)D1 + W0

γ
+ 2W

(1 + exp(κA)) ,

18


---Page Break---
where the first inequality is by the relation ξ ≤ζ, the second inequality is by the relation ζ ≤h and
the bounding property of first-order derivative, the third inequality is by the induction hypothesis and
the last inequality is due to h ≤1.

By the relation ∥¨xt∥≤κA∥˙xt∥+ W, we have
∥¨xt0+ζ∥≤κAD2 + W .
So we choose D3 = κAD2 + W. By the equation, we have

∥ˆwt −wt∥=

xt+1 −xt −h(Axt + But + wt)

h



=

xt+1 −xt −h ˙xt

h

 =



R h
0 ( ˙xt+ξ −˙xt)dξ

h

 =



R h
0
R ξ
0 ¨xt+ζdζdξ

h



≤

R h
0
R ξ
0 ∥¨xt+ζ∥dζdξ

h
≤hD3 ,

where in the second line we use the Newton-Leibniz formula, the inequality is by the conclusion
∥¨xt∥≤D3 which we have proved before. By Assumption 1, we have
∥ˆwt∥≤W + hD3 .

Choosing D3 = κAD2 + W, W0 = W + hD3 = W + h(κAD2 + W), we get

∥˙xt0+ζ∥≤((κA + κBκ)D1 + W0

γ
+ 2W)(1 + exp(κA))

≤((κA + κBκ)D1 + W + h(κAD2 + W)

γ
+ 2W)(1 + exp(κA))

≤D2

hκA

γ (1 + exp(κA))

+

(κA + κBκ)D1 + (1 + h + 2γ)W

γ


(1 + exp(κA))) .

Using the notation

β1 = hκA

γ (1 + exp(κA)) ,

β2 =

(κA + κBκ)D1 + 2(1 + γ)W

γ


(1 + exp(κA)) .

When h <
γ
2κA(1+exp(κA)), we have β1 < 1

2. Taking D2 = 2β2 we get

∥˙xt0+ζ∥≤β1D2 + β2 ≤D2 .

So we have proved that for any t ∈[t0, t0 + h], ∥˙xt∥≤D2, ∥¨xt∥≤D3, ∥ˆwt∥≤W0.

Then we choose suitable D1 and prove that for any t ∈[t0, t0 + h], ∥xt∥≤D1.

Using Lemma 7, we have

xt+1 = h

t
X

i=0
Ψt,i ˆwt−i .

By the induction hypothesis of bounded state and estimation noise in [0, t0] together with Lemma 8,
we have

∥xt+1∥≤h

t
X

i=0
(lhκB + 1)κ2(1 −hγ)i(W + hD3)

≤(lhκB + 1)κ2(W + hD3)

γ
.

19


---Page Break---
Then, by the Taylor expansion and the inequality ˙xt ≤D2 , we have for any ζ ∈[0, h],

∥xt+1 −xt+ζ∥= ∥
Z h

ζ
˙xt+ξdξ∥≤(h −ζ)D2 ≤hD2 .

Therefore we have

∥xt+ζ∥≤∥xt+1∥+ hD2 ≤(lhκB + 1)κ2(W + hD3)

γ
+ hD2

= (lhκB + 1)κ2W(1 + h)

γ
+ hD2

(lhκB + 1)κ2κA

γ
+ 1


≤(lκB + 1)2κ2W

γ
+ hD2

(lκB + 1)κ2κA

γ
+ 1

.

In the last inequality we use h ≤1.

By the relation D2 = β2/(1 −β1) and β1 ≤1

2, we know that

D2 ≤2

(κA + κBκ)D1 + 2(1 + γ)W

γ


(1 + exp(κA)).

Using the notation

γ1 = 2h(κA + κBκ)(1 + exp(κA)) ,

γ2 = (lκB + 1)2κ2W

γ
+ 4(1 + γ)W

γ
(1 + exp(κA))
(lκB + 1)κ2κA

γ
+ 1

.

We have ∥xt+ζ∥≤γ1D1 + γ2.

From the equation of γ1 we know that when h ≤
1
4(κA+κBκ)(1+exp(κA)) we have γ1 ≤1

2. Then we
choose D1 = 2γ2, we finally get

∥xt+ζ∥≤γ1D1 + γ2 ≤D1 .

Finally, set

h0 = min

1,
γ
κA(1 + exp(κA)),
1
4(κA + κBκ)(1 + exp(κA))


,

By the relationship D1 = 2γ2, D2 = 2β2, D3 = κAD2 + W, W0 = W + hD3,

we can verify the induction hypothesis. Moreover, we know that D1, D2, D3 are not depend on h.
Therefore we have proved the claim.

The last step is then to bound the action and the approximation errors of states and actions.
Lemma 2. Under Assumption 1 and 2, choosing arbitrary h in the interval [0, h0] where h0 is
a constant only depends on the parameters in the assumption, we have for any t and policy Mi,
∥xt∥, ∥yt∥, ∥ut∥, ∥vt∥≤D, ∥˙xt∥≤D, ∥xt −yt∥, ∥ut −vt∥≤κ2(1 + κ)(1 −hγ)Hm+1D. In
particular, taking all the Mt = 0 and K = K∗, we can also obtain the inequality of the optimal
solution: ∥x∗
t ∥, ∥u∗
t ∥≤D.

Proof. By Lemma 8, we have

∥Ψt,i∥≤a(lhκB + 1)κ2(1 −hγ)i−1 .

By Lemma 9 we know that for any h in [0, h0], where

h0 = min

1,
γ
κA(1 + exp(κA)),
1
4(κA + κBκ)(1 + exp(κA))


,

20


---Page Break---
we have ∥xt∥≤D1.

By Lemma 7, Lemma 8 and Lemma 9, we have

∥yt+1∥= ∥h

2Hm
X

i=0
Ψt,i ˆwt−i∥

≤hW0

2Hm
X

i=0
a(lhκB + 1)κ2(1 −hγ)i−1

≤aW0(lhκB + 1)κ2

γ
= ˜D1 .

Via the definition of xt, yt, we have

∥xt −yt∥≤κ2(1 −hγ)Hm+1 ∥xt−Hm∥≤κ2(1 −hγ)Hm+1D1 .

For the actions

ut = −Kxt + h

Hm
X

i=1
M i
t ˆwt−i ,

vt = −Kyt + h

Hm
X

i=1
M i
t ˆwt−i ,

we can derive the bound

∥ut∥≤∥Kxt∥+ h

Hm
X

i=1

M i
t ˆwt−i
 ≤κ ∥xt∥+ W0h

Hm
X

i=1
a(1 −hγ)i−1 ≤κD1 + aW0

γ
,

∥vt∥≤∥Kyt∥+ h

Hm
X

i=1

M i
t ˆwt−i
 ≤κ ∥yt∥+ W0h

Hm
X

i=1
a(1 −hγ)i−1 ≤κ ˜D1 + aW0

γ
,

∥ut −vt∥≤∥K∥∥xt −yt∥≤κ3(1 −hγ)Hm+1D1 .

By Lemma 9, taking D = max{D1, D2, ˜D1, κD1+ W0

γ , κ ˜D1+ W0

γ }, we get the following inequality:
∥xt∥, ∥yt∥, ∥ut∥, ∥vt∥≤D, ∥˙xt∥≤D.

We also have

∥xt −yt∥+∥ut −vt∥≤κ2(1−hγ)Hm+1D1 +κ3(1−hγ)Hm+1D1 ≤κ2(1+κ)(1−hγ)Hm+1D .

In particular, the optimal policy can be recognized as taking the DAC policy with all the Mt equal to
0 and the fixed strongly stable policy K = K∗. So we also have ∥x∗
t ∥, ∥u∗
t ∥≤D.

Now we have finished the analysis of evolution of the states. It will be helpful to prove the key
lemmas in this paper.

D
Proof of Lemma 3

In this section we will prove the following lemma:

Lemma 3. Under Assumption 2, Algorithm 1 attains the following bound of R0:

R0 =

n−1
X

i=0

Z (i+1)h

ih
(ct(xt, ut) −ct(x∗
t , u∗
t ))dt −h

n−1
X

i=0
(ci(xi, ui) −ci(x∗
i , u∗
i )) ≤(G + L)D2hT .

21


---Page Break---
Proof. By Assumption 2 and Lemma 2, since we use the unchanged policy ut in the interval
t ∈[ih, (i + 1)h], we have
|ct(xt, ut) −cih(xih, uih)| ≤|ct(xt, ut) −ct(xih, uih)| + |ct(xih, uih) −cih(xih, uih)|

≤max
x
∥∇xct(x, u)∥∥xt −xih∥+ L(t −ih)D2

≤GD∥
Z t

ih
˙xsds∥+ L(t −ih)D2

≤(G + L)D2(t −ih) .

Therefore we have

|

n−1
X

i=0

Z (i+1)h

ih
ct(xt, ut)dt −h

n−1
X

i=0
ci(xi, ui)|

=|

n−1
X

i=0

Z (i+1)h

ih
(ct(xt, ut) −cih(xih, uih))dt|

≤(G + L)D2
n−1
X

i=0

Z (i+1)h

ih
(t −ih)dt = 1

2(G + L)D2nh2 = 1

2(G + L)D2hT .

A similar bound can easily be established by lemma 2 about the optimal state and policy:

|

n−1
X

i=0

Z (i+1)h

ih
ct(x∗
t , u∗
t )dt −

n−1
X

i=0
ci(x∗
i , u∗
i )| ≤1

2(G + L)D2hT .

Taking sum of the two terms we get R0 ≤(G + L)D2hT.

E
Proof of Lemma 4

In this section we will prove the following lemma:
Lemma 4. Under Assumption 1 and 2, Algorithm 1 attains the following bound of R1:

R1 =

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −fi

˜
Mi−H, . . . , ˜
Mi



≤nGD2κ2(1 + κ)(1 −hγ)Hm+1 .

Proof. Using Lemma 2 and Assumption 2, have the approximation error between ideal cost and
actual cost bounded as,
|ct (xt, ut) −ct (yt, vt)| ≤|ct (xt, ut) −ct (yt, ut)| + |ct (yt, ut) −ct (yt, vt)|
≤GD∥xt −yt∥+ GD∥ut −vt∥

≤GD2κ2(1 + κ)(1 −hγ)Hm+1 ,
where the first inequality is by triangle inequality, the second inequality is by Assumption 2, Lemma
2, and the third inequality is by Lemma 2.

With this, we have

R1 =

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −fi( ˜
Mi−H, ..., ˜
Mi)





=

p−1
X

i=0





(i+1)m−1
X

j=im
ci(xi, ui) −

(i+1)m−1
X

j=im
ci(yi, vi)





≤

p−1
X

i=0

(i+1)m−1
X

j=im
GD2κ2(1 + κ)(1 −hγ)Hm+1 ≤nGD2κ2(1 + κ)(1 −hγ)Hm+1 .

22


---Page Break---
F
Proof of Lemma 5

Before we start the proof of Lemma 5, we first present an overview of the online convex optimization
(OCO) with memory framework. Consider the setting where, for every t, an online player chooses
some point xt ∈K ⊂Rd, a loss function ft : KH+1 7→R is revealed, and the learner suffers a loss
of ft (xt−H, . . . , xt). We assume a certain coordinate-wise Lipschitz regularity on ft of the form
such that, for any j ∈{1, . . . , H}, for any x1, . . . , xH, ˜xj ∈K

|ft (x1, . . . , xj, . . . , xH) −ft (x1, . . . , ˜xj, . . . , xH)| ≤L ∥xj −˜xj∥.

In addition, we define ˜ft(x) = ft(x, . . . , x), and we let

Gf =
sup
t∈{1,...,T },x∈K

∇˜ft(x)
 ,
Df = sup
x,y∈K
∥x −y∥.

The resulting goal is to minimize the policy regret, which is defined as

Regret =

T
X

t=H
ft (xt−H, . . . , xt) −min
x∈K

T
X

t=H
ft(x, . . . , x) .

Algorithm 2 Online Gradient Descent with Memory (OGD-M)

Input: Step size η, functions {ft}T
t=m.
Initialize x0, . . . , xH−1 ∈K arbitrarily.
for t = H, . . . , T do

Play xt, suffer loss ft (xt−H, . . . , xt).

Set xt+1 = ΠK

xt −η∇˜ft(x)

.
end for

To minimize this regret, a commonly used algorithm is the Online Gradient descent. By running the
Algorithm 2, we may bound the policy regret by the following lemma:

Lemma 10. Let {ft}T
t=1 be Lipschitz continuous loss functions with memory such that ˜ft are convex.
Then by runnning algorithm 2 itgenerates a sequence {xt}T
t=1 such that

T
X

t=H
ft (xt−H, . . . , xt) −min
x∈K

T
X

t=H
ft(x, . . . , x) ≤
D2
f
η + TG2
fη + LH2ηGfT .

Furthermore, setting η =
Df
√

Gf (Gf +LH2)T implies that

PolicyRegret ≤2Df
q

Gf (Gf + LH2) T .

Proof. By the standard OGD analysis [18], we know that

T
X

t=H
˜ft (xt) −min
x∈K

T
X

t=H
˜ft(x) ≤
D2
f
η + TG2η.

In addition, we know by the Lipschitz property, for any t ≥H, we have

|ft (xt−H, . . . , xt) −ft (xt, . . . , xt)| ≤L

H
X

j=1
∥xt −xt−j∥≤L

H
X

j=1

j
X

l=1
∥xt−l+1 −xt−l∥

≤L

H
X

j=1

j
X

l=1
η
∇˜ft−l (xt−l)
 ≤LH2ηG,

and so we have that

T
X

t=H
ft (xt−H, . . . , xt) −

T
X

t=H
ft (xt, . . . , xt)

 ≤TLH2ηG.

23


---Page Break---
It follows that
T
X

t=H
ft (xt−H, . . . , xt) −min
x∈K

T
X

t=H
ft(x, . . . , x) ≤
D2
f
η + TG2
fη + LH2ηGfT .

In this setup, the first term corresponds to the DAC policy we make, and the second term is used to
approximate the optimal strongly stable linear policy. It is worth noting that the cost of OCO with
memory depends on the update frequency H. Therefore, we propose a two-level online controller.
The higher-level controller updates the policy with accumulated feedback at a low frequency to reduce
the regret, whereas a lower-level controller provides high-frequency updates of the DAC policy to
reduce the discretization error. In the following part, we define the update distance of the DAC policy
as l = Hm, where m is the ratio of frequency between the DAC policy update and OCO memory
policy update. Formally, we update the value of Mt once every m transitions, where gt represents a
loss function.

Mt+1 =
ΠM (Mt −η∇gt(M))
if t%m == 0
Mt
otherwise .

From now on, we denote ˜
Mt = Mtm for the convenience to remove the duplicate elements. By the
definition of ideal cost, we know that it is a well-defined definition.

By Lemma 7 we know that

yt+1 = h

2Hm
X

i=0
Ψt,i ˆwt−i,

vt = −Kyt + h

Hm
X

j=1
M j
t ˆwt−j ,

where

Ψt,i = Qi
h1i≤l + h

l
X

j=0
Qj
hBM i−j
t−j 1i−j∈[1,l] .

So we know that yt and yt are linear combination of Mt, therefore

fi

˜
Mi−H, . . . , ˜
Mi

=

(i+1)m−1
X

t=im
ct

yt

˜
Mi−H, . . . , ˜
Mi

, vt

˜
Mi−H, . . . , ˜
Mi

.

is convex in Mt. So we can use the OCO with memory structure to solve this problem.

By Lemma 9 we know that yt and vt are bounded by D. Then we need to calculate the diameter,
Lipchitz constant, and gradient bound of this function fi. In the following, we choose the DAC policy
parameter l = Hm.
Lemma 11. (Bounding the diameter) We have

Df =
sup
Mi,Mj∈M
∥Mi −Mj∥≤2a

hγ
.

Proof. By the definition of M, taking l = Hm we know that

sup
Mi,Mj∈M
∥Mi −Mj∥≤

Hm
X

k=1
∥M k
i −M k
j ∥

≤

Hm
X

k=1
2a(1 −hγ)k−1

≤2a

hγ .

24


---Page Break---
Lemma
12.
(Bounding
the
Lipschitz
Constant)
Consider
two
policy
sequences
n
˜
Mi−H . . . ˜
Mi−k . . . ˜
Mi
o
and
n
˜
Mi−H . . . ˆ
Mi−k . . . ˜
Mi
o
which differ in exactly one policy

played at a time step t −k for k ∈{0, . . . , H}. Then we have that

fi

˜
Mi−H . . . ˜
Mi−k . . . ˜
Mi

−fi

˜
Mi−H . . . ˆ
Mi−k . . . ˜
Mi
 ≤C2κ3κBW0

Hm
X

j=0
∥˜
M j
i−k−ˆ
M j
i−k∥,

where C is a constant.

Proof. By the definition we have

∥yt −˜yt∥= ∥h

2Hm
X

i=0
h

Hm
X

j=0
Qj
hB(M i−j
t−j −˜
M i−j
t−j )1i−j∈[1,Hm] ˆwt−i∥

≤h2κ2κBW0

2Hm
X

i=0

Hm
X

j=0
∥M i−j
t−j −˜
M i−j
t−j ∥1i−j∈[1,Hm]

≤h2κ2κBW0m

Hm
X

j=0
∥˜
M j
i−k −ˆ
M j
i−k∥

= hCκ2κBW0

Hm
X

j=0
∥˜
M j
i−k −ˆ
M j
i−k∥.

Where the first inequality is by ∥Qj
h∥≤κ2(1 −hγ)j−1 ≤κ2 and lemma 9 of bounded estimation
disturbance, the second inequality is by the fact that Mi−k have taken m times, the last equality is by
m = C

h . Furthermore, we have that

∥vt −˜vt∥= ∥−K (yt −˜yt) ∥≤hCκ3κBW0

Hm
X

j=0

 ˜
M j
i−k −ˆ
M j
i−k
 .

Therefore using Assumption 2, Lemma 9 and Lemma 2 we immediately get that

fi

˜
Mi−H . . . ˜
Mi−k . . . ˜
Mi

−fi

˜
Mi−H . . . ˆ
Mi−k . . . ˜
Mi
 ≤C2κ3κBW0

Hm
X

j=0
∥˜
M j
i−k−ˆ
M j
i−k∥.

Lemma 13. (Bounding the Gradient) We have the following bound for the gradient:

∥∇Mft(M . . . M)∥F ≤GDCκ2(κ + 1)W0κB

γ

Proof. Since M is a matrix, the ℓ2 norm of the gradient ∇Mft corresponds to the Frobenius norm of
the ∇Mft matrix. So it will be sufficient to derive an absolute value bound on ∇M [r]
p,qft(M, . . . , M)
for all r, p, q. To this end, we consider the following calculation. Using lemma 9 we get that
yt(M . . . M), vt(M . . . M) ≤D. Therefore, using Assumption 2 we have that

∇M [r]
p,qct(M . . . M)
 ≤GD

 
∂yt(M)

∂M [r]
p,q
+ ∂vt(M . . . M)

∂M [r]
p,q



!

.

25


---Page Break---
We now bound the quantities on the right-hand side:


δyt(M . . . M)

δM [r]
p,q

 =


h

2Hm
X

i=0
h

Hm
X

j=1

"
∂Qj
hBM [i−j]

∂M [r]
p,q

#

ˆwt−i1i−j∈[1,H]



≤h2
r+Hm
X

i=r



"
∂Qi−r
h
BM [r]

∂M [r]
p,q

#

wt−i



≤h2κ2W0κB
1
hγ = hκ2W0κB

γ
.

Similarly,

∂vt(M . . . M)

∂M [r]
p,q

 ≤κ


δyt(M . . . M)

δM [r]
p,q

 ≤κhκ2W0κB

γ
≤hκ3W0κB

γ
.

Combining the above inequalities with

fi

˜
Mi−H, . . . , ˜
Mi

=

(i+1)m−1
X

t=im
ct

yt

˜
Mi−H, . . . , ˜
Mi

, vt

˜
Mi−H, . . . , ˜
Mi

.

gives the bound that

∥∇Mft(M . . . M)∥F ≤GDCκ2(κ + 1)W0κB

γ
.

Finally we prove Lemma 5:

Lemma 5. Under Assumption 1 and 2, choosing m = C

h and η = Θ( m

T h), Algorithm 1 attains the
following bound of R2:

R2 =

p−1
X

i=0
fi( ˜
Mi−H, . . . , ˜
Mi) −min
M∈M

p−1
X

i=0
fi(M, . . . , M)

≤4a

γ

s

GDC2κ2(κ + 1)W0κB

γ
(GDCκ2(κ + 1)W0κB

γ
+ C2κ3κBW0H2)n

h .

Proof. By Lemma 10 we have

R2 ≤2Df
q

Gf (Gf + LH2) p

By Lemma 11, Lemma 12, and Lemma 13 we have

R2 ≤2Df
q

Gf (Gf + LH2) p

≤2 2a

hγ

s

GDCκ2(κ + 1)W0κB

γ
(GDCκ2(κ + 1)W0κB

γ
+ C2κ3κBW0H2) n

m

≤4a

γ

s

GDC2κ2(κ + 1)W0κB

γ
(GDCκ2(κ + 1)W0κB

γ
+ C2κ3κBW0H2)n

h .

26


---Page Break---
G
Proof of Lemma 6

In this section, we will prove the approximation value of DAC policy and optimal policy is sufficiently
small. First, we introduce the following:

Lemma 14. For any two (κ, γ)-strongly stable matrices K∗, K, there exists M =
 
M 1, . . . , M Hm

where

M i = (K −K∗) (I + h(A −BK∗))i−1 ,

such that

ct(xt(M), ut(M)) −ct(x∗
t , u∗
t ) ≤GDW0κ3a(lhκB + 1)(1 −hγ)Hm .

Proof. Denote Qh(K) = I + h(A −BK), Qh(K∗) = I + h(A −BK∗). By Lemma 7 we have

x∗
t+1 = h

t
X

i=0
Qi
h(K∗) ˆwt−i .

Consider the following calculation for i ≤Hm and M i = (K −K∗) (I + h(A −BK∗))i−1:

Ψt,i (M, . . . , M) = Qi
h(K) + h

i
X

j=1
Qi−j
h
(K)BM j

= Qi
h(K) + h

i
X

j=1
Qi−j
h
(K)B (K −K∗) Qj−1
h
(K∗)

= Qi
h(K) +

i
X

j=1
Qi−j
h
(K)(Qh(K∗) −Qh(K))Qj−1
h
(K∗)

= Qi
h(K∗) ,

where the final equality follows as the sum telescopes. Therefore, we have that

xt+1(M) = h

Hm
X

i=0
Qi
h(K∗) ˆwt−i + h

t
X

i=Hm+1
Ψt,i ˆwt−i .

Then we obtain that

xt+1(M) −x∗
t+1
 ≤hW0

t
X

i=Hm+1
(∥Ψt,i (M∗)∥+ ∥Qi
h(K∗)∥) .

Using Definition 1 and Lemma 7 we finally get

xt+1(M) −x∗
t+1
 ≤hW0(

t
X

i=Hm+1
((lhκB + 1)aκ2(1 −hγ)i−1) + κ2(1 −hγ)i)

≤W0(lhκB + 2)aκ2(1 −hγ)Hm .

27


---Page Break---
We also have

∥u∗
t −ut (M)∥=

−K∗x∗
t + Kxt (M) −h

Hm
X

i=0
M i ˆwt−i



= ∥(K −K∗)x∗
t + K(xt(M) −x∗
t ) −h

Hm
X

i=0
M i ˆwt−i∥

= ∥(K −K∗)h

t−1
X

i=0
Qi
h(K∗) ˆwt−i + K(xt(M) −x∗
t ) −h

Hm
X

i=0
M i ˆwt−i∥

= ∥K(xt(M) −x∗
t ) −h

t−1
X

i=Hm+1
(K −K∗)Qi−1
h
(K∗) ˆwt−i∥

= ∥Kh

t−1
X

i=Hm+1
(Ψt,i −Qi−1
h
(K∗)) ˆwt−i −h

t−1
X

i=Hm+1
(K −K∗)Qi−1
h
(K∗) ˆwt−i∥

=

h

t−1
X

i=Hm+1
K∗ 
Qi−1
h
(K∗) + Ψt,i

ˆwt−i



≤W0κ((1 −hγ)Hm + a(lhκB + 1)κ2(1 −hγ)Hm)

= W0κ(a(lhκB + 1)κ2 + 1)(1 −hγ)Hm) ,
where the inequality is by Definition 1 and Lemma 8.

Finally, we have
|ct (xt(M), ut(M)) −ct (x∗
t , u∗
t )|
≤|ct (xt(M), ut(M)) −ct (x∗
t , ut(M))| + |ct (x∗
t , ut(M)) −ct (x∗
t , u∗
t )|
≤GD|xt(M) −x∗
t | + GD|ut(M) −u∗
t |

≤GDW0κ3a(lhκB + 1)(1 −hγ)Hm ,
where the second inequality is by Assumption 2.

Then we can prove our main lemma:
Lemma 6. Under Assumption 1 and 2, Algorithm 1 attains the following bound of R3:

R3 = min
M∈M

p−1
X

i=0
fi(M, ..., M) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i ) ≤3n(1 −hγ)HmGDW0κ3a(lhκB + 1) .

Proof. By choosing
M i = (K −K∗) (I + h(A −BK∗))i−1 .
We know that
∥M i∥= ∥(K −K∗) (I + h(A −BK∗))i−1 ∥≤2κ3(1 −γ)i−1 .
Therefore choose a = 2κ3 we have M = {M i} in the DAC policy update class M.

Then we have the analysis of the regret:

R3 = min
M∈M

p−1
X

i=0
fi(M, ..., M) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i )

≤min
M∈M

p−1
X

i=0

(i+1)m−1
X

j=im
ci(xi(M), ui(M)) −

p−1
X

i=0

(i+1)m−1
X

j=im
ci(x∗
i , u∗
i ) + nκ2(1 + κ)(1 −hγ)Hm+1D

≤3n(1 −hγ)HmGDW0κ3a(lhκB + 1) ,
where the first inequality is by Lemma 2 and the second inequality is by Lemma 14.

28


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We clarify our contributions and basic problem setups in both abstract and
introduction.
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
Justification: We discuss the limitation of our paper in Section 7.
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
Answer: [Yes]

29


---Page Break---
Justification: We provide the full set of assumptions and a complete proof.
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
Justification: We disclose the experiment details in Section 6.
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

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

30


---Page Break---
Answer: [No]

Justification: Our code is very simple, just use the traditional SAC algorithm with one line
implement. Our main contribution is the theoretical analysis.

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

Justification: We specify all the training and test details in 6.

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

Justification: For each experiment we use 3 random seeds and take the average.

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

31


---Page Break---
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

Justification: We specify all the computational resources in 6.

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

Justification: We conform with the NeurIPS Code of Ethics.

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

Justification: Our work is about the theory on online control, which does not seem to have
evident societal impacts.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

32


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

Justification: The paper poses no such risks.

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

Answer: [Yes]

Justification: We add citations for all datasets we used.

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

33


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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
Justification: The paper does not involve crowdsourcing nor research with human subjects.
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

34


---Page Break---
