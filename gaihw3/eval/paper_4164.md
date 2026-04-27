Gradient-based Discrete Sampling with
Automatic Cyclical Scheduling

Patrick Pynadath
Department of Computer Science
Purdue University
West Lafayette, IN, 47907
ppynadat@purdue.edu.edu

Riddhiman Bhattacharya
Department of Management
Purdue University
West Lafayette, IN, 47907
bhatta76@purdue.edu

Arun Hariharan
Department of Computer Science
Purdue University
West Lafayette, IN, 47907
harihar4@purdue.edu

Ruqi Zhang
Department of Computer Science
Purdue University
West Lafayette, IN, 47907
ruqiz@purdue.edu

Abstract

Discrete distributions, particularly in high-dimensional deep models, are often
highly multimodal due to inherent discontinuities. While gradient-based discrete
sampling has proven effective, it is susceptible to becoming trapped in local modes
due to the gradient information. To tackle this challenge, we propose an automatic
cyclical scheduling, designed for efficient and accurate sampling in multimodal
discrete distributions. Our method contains three key components: (1) a cyclical
step size schedule where large steps discover new modes and small steps exploit
each mode; (2) a cyclical balancing schedule, ensuring “balanced” proposals for
given step sizes and high efficiency of the Markov chain; and (3) an automatic
tuning scheme for adjusting the hyperparameters in the cyclical schedules, al-
lowing adaptability across diverse datasets with minimal tuning. We prove the
non-asymptotic convergence and inference guarantee for our method in general
discrete distributions. Extensive experiments demonstrate the superiority of our
method in sampling complex multimodal discrete distributions.

1
Introduction

Discrete variables are common in many machine learning problems, highlighting the crucial need
for efficient discrete samplers. Recent advances [Grathwohl et al., 2021, Zhang et al., 2022b, Sun
et al., 2021, 2023b,a, Xiang et al., 2023] have leveraged gradient information in discrete distributions
to improve proposal distributions, significantly boosting their efficiency. These advancements have
set new benchmarks in discrete sampling tasks across graphical models, energy-based models, and
combinatorial optimization [Goshvadi et al., 2023].

However, one major limitation of gradient-based methods is their susceptibility to becoming trapped in
local modes [Ruder, 2016, Ziyin et al., 2021], which significantly reduces the accuracy and efficiency
of sampling results. In continuous spaces, several strategies such as cyclical step sizes [Zhang et al.,
2020], parallel tempering [Swendsen and Wang, 1986, Deng et al., 2020a], and flat histograms [Berg
and Neuhaus, 1991, Deng et al., 2020b], have been proposed to address this issue. When it comes to
discrete distributions, which are inherently more multimodal due to their discontinuous nature, the
problem becomes even more severe. Despite the pressing need, there is a lack of methodology for
gradient-based discrete samplers to effectively explore multimodal distributions. Current methods

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
(a) Ground Truth
(b) RW
(c) DMALA
(d) AB
(e) ACS (Ours)

Figure 1: Sampling on a 2d distribution with multiple modes. (a): ground truth. (b): results from
a random walk sampler. (c): results from DMALA [Zhang et al., 2022b] with a manually tuned
step size. (d): results from AB [Sun et al., 2023a]. (e): results from our method ACS. While the
random walk sampler can find all modes, its characterization is noisy and lacks details for each mode.
Gradient-based samplers (b) and (c) effectively characterize a specific mode but are easily trapped
in some local modes. Our method (d) can find all modes efficiently and characterize each mode
accurately.

often fall far short in traversing the complex landscapes of multimodal distributions, as illustrated in
Figure 1.

In this paper, we propose automatic cyclical scheduling for gradient-based discrete sampling to
efficiently and accurately sample from multimodal distributions. To balance between uncovering new
modes and characterizing the current mode, we parameterize a family of gradient-based proposals
that span a spectrum from local to global proposals. The parameterized proposal dynamically adjusts
according to cyclical schedules of both step size and the balancing parameter, smoothly transitioning
from global exploratory moves to more localized moves within each cycle. These cyclical schedules
are automatically tuned by a specially designed algorithm, which identifies optimal step sizes and
balancing parameters for discrete distributions. Our contributions are summarized as follows:

• We present the first gradient-based discrete sampling method that targets multimodal dis-
tributions. Our method incorporates cyclical schedules for both step size and balancing
parameter to facilitate the exploration and exploitation in discrete distributions.

• We propose an automatic tuning algorithm to configure the cyclical schedule, enabling effort-
less and customized adjustments across various datasets without much manual intervention.

• We offer non-asymptotic convergence and inference guarantees for our method in general
discrete distributions. To our knowledge, this is the first non-asymptotic convergence bound
of gradient-based discrete sampling to the target distribution with inference guarantees,
which could be of independent interest.

• We demonstrate the superiority of our method for both sampling and learning tasks including
restricted Boltzmann machines, deep energy-based models, and large language models.

2
Related Work

Gradient-based Discrete Sampling
Zanella [2017] introduced a family of locally informed pro-
posals, laying the foundation for recent developments in efficient discrete sampling. Building upon
this, Grathwohl et al. [2021] further incorporates gradient approximation, significantly reducing
computational costs. Following these pioneering efforts, numerous studies have proposed various
gradient-based discrete sampling techniques [Rhodes and Gutmann, 2022, Sun et al., 2021, 2022,
2023b, Xiang et al., 2023]. Zhang et al. [2022b] develops a discrete Langevin proposal, translating
the powerful Langevin algorithm to discrete spaces. Sansone [2022] introduces a self-balancing
method to optimize the balancing functions in locally balanced proposals. While our work also
utilizes an adaptive phase, it differs in that our parameterization extends beyond the local regime, and
our proposal parameterization is considerably simpler.

Perhaps the most closely related study is the any-scale balanced sampler [Sun et al., 2023a]. This
method uses a non-local balancing proposal and adaptively tunes it. Our work, however, differs in
several key aspects: (1) We focus on combining both local and non-local proposals to effectively
characterize multimodal discrete distributions, as opposed to focusing on a single optimal proposal.
(2) Our automatic tuning algorithm adjusts the step size and balancing parameter by considering

2


---Page Break---
the special discrete structures and targets a specific Metropolis-Hastings acceptance rate, rather than
maximizing the average coordinates changed per step. (3) Our method can be applied to learning
energy-based models (EBM) and sampling large language models, whereas their approach cannot.

Sampling on Multimodal Distributions
There exist several sampling methods targeting discrete
multimodal distributions, such as simulated tempering [Marinari and Parisi, 1992], the Swendsen-
Wang algorithm [Swendsen and Wang, 1987], and the Wolff algorithm [Wolff, 1989]. However, these
methods usually use random walk or Gibbs sampling as their proposals. It is unclear how these
methods can be adapted for gradient-based discrete sampling.

In continuous spaces, various gradient-based methods have been developed specifically for multi-
modal distributions [Zhang et al., 2020, Deng et al., 2020a,b]. Our method distinguishes from the
cyclical step size in Zhang et al. [2020] by incorporating an additional cyclical balancing parameter
schedule and an automatic tuning scheme, which are crucial for efficient exploration in discrete
distributions. Furthermore, our theoretical analysis of convergence is different from that in Zhang
et al. [2020] which relies on continuous stochastic processes.

3
Preliminaries

3.1
Problem Definition

We consider the task of sampling from some target distribution defined over a discrete space

π(θ) = 1

Z exp(U(θ)),
θ ∈Θ.

Here, θ is a d dimensional discrete variable in domain Θ, U is the energy function, and Z is the
normalizing constant. We make the following assumptions of the domain and the energy function,
following the literature of gradient-based discrete sampling [Grathwohl et al., 2021, Sun et al., 2021,
Zhang et al., 2022b]: (1) The domain is coordinatewisely factorized, Θ = Πd
i=1Θi. (2) The energy
function U can be extended to a differentiable function in Rd.

3.2
Locally Balanced Proposals

Zanella [2017] introduces a family of informed proposals, which is defined below:

Qg,α(θ′|θ) =
g

π(θ′)

π(θ)

Kα(θ′ −θ)

Zg,α(θ)
(1)

Here, Kα is a kernel that determines the scale of the proposal where α plays a similar role as the
step size. g(t) is a balancing function that determines how to incorporate the information about π. If
g(t) = tg( 1

t ), the proposal becomes a locally balanced proposal, which is asymptotically optimal in
the local regime, that is, when the step size α →0.

4
Automatic Cyclical Sampler

We aim to develop a sampler capable of escaping local modes in general multimodal discrete
distributions, including those that appear in deep energy-based models and large language models.
First, we motivate using the cyclical schedule by demonstrating the issue of gradient-based samplers
getting stuck in local modes on a toy dataset. We then present our sampler’s parameterization of the
step size and balancing function. Next, we introduce a cyclical schedule for the proposal distribution
that enables effective exploration and characterization of discrete multimodal distributions. Finally,
we develop an automatic tuning method that simplifies the process of identifying hyperparameters in
cyclical schedules.

4.1
Motivating Example: A Synthetic Multimodal Discrete Distribution

To demonstrate the crucial issue of local modes trapping gradient-based samplers, we construct a 2-
dimensional dataset consisting of integers. We define Θ = {0, 1, · · · N}2, where N is the maximum

3


---Page Break---
value for each coordinate. Given a set of modes {µ1, µ2, . . . µl}, we define the energy as follows:

U(θ) = log

 
l
X

i=1
exp
||θ −µi||2

2σ

!

.
(2)

This distribution enables easy comparison between different methods in terms of their ability to both
explore and exploit the target distribution. We demonstrate the results of various samplers in Figure 1.
More experimental details can be found in Appendix D.1.

A visual comparison reveals that while gradient-based samplers (DMALA [Zhang et al., 2022b] and
AB [Sun et al., 2023a]) are very effective at characterizing a given mode, they tend to get trapped in
some small neighborhood, preventing a proper characterization of the distribution as a whole.

We can understand this behavior of gradient-based samplers by comparing them to a random walk
sampler (RW), which is able to explore all the modes but unable to fully characterize the detail of each
one. While the RW sampler proposes movements uniformly over the sample space, gradient-based
samplers propose movement based on the geometry of the distribution as captured by the gradient.
Because the proposed movements are in the direction of increasing density, these proposals are able
to characterize a given mode in detail. At the same time, these proposals hinder escape to more
distant modes as the gradient points away from their direction. For this reason, we observe that local
modes are able to “trap” gradient-based samplers.

4.2
Parameterized Proposal Distribution

To derive an automatic schedule for the proposal, we need to parameterize the proposal first. We
define Kα and g(t) in the informed proposal [Zanella, 2017] as follows:

Kα(θ′ −θ) = exp −||θ′−θ||2

2α
Z
,
α ∈(0, ∞);
g(t) = tβ,
β ∈[0.5, 1)
(3)

where β is called a balancing parameter. α →0, β = 0.5 correspond to a locally-balanced proposal
and α →∞, β = 1 correspond to a globally-balanced proposal. Values in between result in
interpolations between locally-balanced and globally-balanced proposals. Note that β ∈(0, 1) in
Sun et al. [2023a] while our range is narrower.

We substitute these definitions into Equation (1) and apply the first-order Taylor expansion:

Qα,β(θ′|θ) ∝exp

β(U(θ′) −U(θ)) −||θ′ −θ||2

2α


≈exp

β(∇θU(θ)(θ′ −θ)) −||θ′ −θ||2

2α


.

(4)

As in Zhang et al. [2022b], we use the assumption of coordinate-wise factorization to obtain the
following coordinate-wise proposal function:

Qi
α,β(θ′
i|θ) = Cat

Softmax

β∇U(θ)i(θ′
i −θi) −(θ′
i −θi)2

2α


.
(5)

In order to make the resulting Markov chain reversible, we apply the Metropolis-Hastings correction,
where we accept the proposed step with the following probability:

A(θ′|θ, α, β) = min

1, exp(U(θ′) −U(θ)))Qα,β(θ|θ′)

Qα,β(θ′|θ)


.
(6)

In summary, we parameterize our proposal as in Equation (5) which includes a spectrum of local
and global proposals. Our proposal is determined by two hyperparameters, the step size α and the
balancing parameter β.

4.3
Cyclical Hyperparameter Schedules

Cyclical Step Size Schedule
In order to effectively explore the whole target distribution while
retaining the ability to exploit local modes, we adopt the cyclical step size schedule from Zhang et al.
[2020]. The definition of step size α for iteration k is as follows:

αk = max

αmax · cos
πmod(k, s)

s


+ 1, αmin


,
(7)

4


---Page Break---
0
20
40
60
80
100
Sampling Step

0.5

1.0

1.5

2.0

2.5

Step Size

0
20
40
60
80
100
Sampling Step

0.5

0.6

0.7

0.8

0.9

Balancing Parameter

(a)

0
1
2
3
4
5
Step Size

0.0

0.2

0.4

0.6

0.8

1.0

MH Acceptance Rate

0.19 0.74

acceptance rate with beta=.8
target acceptance rate

(b)

Figure 2: (a) α-schedule along with the corresponding β schedule. The initial large steps enable
the sampler to explore different regions of the distribution, while the smaller steps enable good
characterization of each region. The balancing parameter β varies with the step size to enable high
acceptance rates for all step sizes. (b) Acceptance rate v.s. step size on EBM sampling on MNIST
shows a non-monotonic relationship.

Algorithm 1 Cyclical Sampling Algorithm

Require: step size schedule {αk}s
k=1, balancing parameter schedule {βk}s
k=1, cycle number n,
steps per cycle s
1: samples ←[ ]
2: for cycle c in range n do
3:
for step k in range s do
4:
θ ←samples[-1]
5:
for coordinate i in range d do
6:
construct Qi
αk,βk(·|θ) as in (5)
7:
sample θ′
i ∼Qi
αk,βk(·|θ)
8:
end for
9:
samples ←θ′ with probability (6)
10:
end for
11: end for
12: return samples

where αmax is the initial step size, αmin is the minimum step size, and s is the number of sampling
steps per cycle. Differing from the cyclical schedule in Zhang et al. [2020], we additionally add αmin
to make sure that even the smallest step size remains effective in discrete spaces.

Cyclical Balancing Schedule
Using large step sizes in (7) can easily result in very low acceptance
rates, removing any benefit of exploration. To address this issue, we introduce a balancing parameter
schedule, which enables reasonable acceptance rates for large step sizes. As discussed in Zanella
[2017], Sun et al. [2023a], the balancing parameter should vary with different step sizes to achieve a
“balanced” proposal. A balanced proposal ensures that the Markov chain is reversible with respect
to a certain distribution, which will converge weakly to the target distribution. For example, when
the step size α →0, the optimal balancing parameter is β = 0.5, whereas for α →∞, the ideal
balancing parameter becomes β = 1.

Thus for a schedule of step sizes, each αi requires a different βi ∈[.5, 1), with larger step sizes
having βi closer to 1 and smaller step sizes having βi closer to 0.5. Using the Metropolis-Hastings
acceptance rate to characterize the quality of a given α, β pair, we define the value of βi as follows:

βi = argmaxβ∈[.5,βi−1]
 
Eθ∼π,θ′∼Qα,β [{A(θ′|θ, αi, β)]

(8)

Intuitively, this definition means that the best βi for a given step size αi maximizes the average
acceptance rate for the proposal function Qα,β. It also conveys that larger step sizes will have larger
balancing parameters.

We include a visualization of the resulting schedules in Figure 2a and outline our algorithm using
the α, β schedules in Algorithm 1. Note that it incurs no extra overhead compared to previous
gradient-based discrete sampling methods as it only adjusts hyperparameters α and β. By using
a combination of large and small α and β, we enable the sampler to explore the distribution fully
without sacrificing the ability to characterize each mode. This is demonstrated in Figure 1e.

5


---Page Break---
Algorithm 2 Automatic Schedule Tuning Algorithm

Require: βmin = .5, βmax, target acceptance rate ρ∗, initial state θinit, steps per cycle s, initial largest
step size αceil = 60, initial smallest step size αfloor = .05
1: θ ←InitBurnin(αceil, βmax, θinit)
2: αmin ←EstAlpha(αfloor, βmin, θ, ρ∗, MAX=False)
3: αmax ←EstAlpha(αceil, βmax, θ, ρ∗, MAX=True)
4: Construct α-sched of length s using (7)
5: β-sched ←EstBalSched(α-sched, βmax, βmin, θ)
6: return α-sched, β-sched

4.4
Automatic Schedule Tuning

For schedules in Equations (7) and (8), we have parameters αmax, αmax, and {β1, β2 . . . βs} to be
decided. In this section, we will introduce an automatic tuning algorithm to easily find suitable values.

Main Idea
Our automatic tuning algorithm depends on the initial balancing parameter βmax, the
final balancing parameter βmin, a target acceptance rate ρ∗, and the number of steps per cycle s. These
values are relatively easy to select, as detailed in Appendix A. Below, we assume they are already
determined. The tuning algorithm first estimates the optimal choices for αmax and αmin based on
ρ∗, which can then be used to construct the full step-size schedule using (7). We then construct the
balancing parameter schedule using (8). The method is summarized in Algorithm 2 with details
regarding subroutines in Appendix A. Our automatic tuning introduces minimal overhead relative to
the more expensive sampling process. For example, in Section 6, we use 500 steps as the budget for
Algorithm 2 where the total number of sampling steps is at least 5000. We further demonstrate that
our algorithm is relatively robust to hyperparameters in Appendix A.1.

In short, our tuning algorithm adopts an alternative optimization strategy, leveraging existing knowl-
edge about hyperparameter values (e.g. βmin and βmax should be around 0.5 and 1 respectively).
While estimating the best pair α, β is challenging due to their interdependence, it is much easier to
fix one and optimize the other [Sun et al., 2023a].

Estimating αmax, αmin
For a given βmax, βmin, our goal is to find step sizes αmax, αmin that enable
an acceptance rate close to ρ∗. We can formally state this goal as follows:

J(α, β) = Eθ∼π

Eθ′∼Qα,β(·|θ) |ρ∗−A(θ′|θ, α, β)|

.
(9)

Given βmax, βmin, we construct the following objectives to pick the corresponding αmax, αmin:

αmax = max{α s.t J(α, βmax) ≈0}
αmin = min{α s.t J(α, βmin) ≈0}.
(10)

By defining the initial and final step sizes in this manner, we ensure that our cyclical schedule includes
a wide range of hyperparameter pairs with different trade-offs in exploration and exploitation.

To solve (10), we estimate αmax by starting with a large step size and gradually decreasing it to find
the step size that yields ρ∗. Unlike existing works that start with small step sizes, we observed that
multiple α values can yield the same acceptance rate for a given β, as shown in Figure 2b. Therefore,
we start with an upper limit αceil and reduce the step size to avoid missing any larger α values that
meet our criteria. Detailed implementation is provided in Algorithm 4 in the Appendix. αmin can be
obtained similarly.

Estimating Balancing Schedule
After setting the start and end pairs for the α and β schedules,
we now define intermediate β values. As the entire step size schedule is fixed by (7), the problem is
to determine the best balancing parameter for each step size. A simple strategy is to test different β
spaced out evenly throughout the interval [.5, βi−1] and select the best value in terms of acceptance
rate. This approach leverages the observation that smaller step sizes tend to have smaller optimal
balancing parameters. Detailed implementation is given in Algorithm 5 in Appendix.

6


---Page Break---
5
Theoretical Analysis

In this section, we present a convergence rate analysis for Algorithm 1. For general step size and
balancing parameter schedules, i.e., at each cycle, the algorithm will go through s steps in which it
will use step sizes α1, α2, · · · , αs and balancing parameters β1, β2, · · · , βs. Note that for each pair
(αi, βi), we have a Markov transition operator which we label Pi for i = 1, 2, · · · , s. The Markov
operator for a single cycle is given by ˆP = P1P2 · · · Ps. We have the following two assumptions:
Assumption 5.1. The function U(·) ∈C2(Rd) has M-Lipschitz gradient. That is

∥∇U(θ) −∇U(θ′)∥≤M ∥θ −θ′∥.

Note that it implicitly assumes that the set in domain Θ is finite. We define conv(Θ) as the convex
hull of the set Θ.
Assumption 5.2. For each θ ∈Rd, there exists an open ball containing θ of some radius rθ, denoted
by B(θ, rθ), such that the function U(·) is mθ-strongly concave in B(θ, rθ) for some mθ > 0.

Assumptions 5.1 and 5.2 are standard in optimization and sampling literature [Bottou et al., 2018,
Dalalyan, 2017]. Under Assumption 5.2, U(·) is m-strongly concave on conv(Θ), following Lemma
C.3 in Appendix.

We define diam(Θ) = supθ,θ′∈Θ ∥θ −θ′∥and ϵαi,βi to be

exp

−
 1

2αi
+ βiM −βim

2


diam(Θ)2 −∥∇U(a)∥diam(Θ)

.

The Markov kernel corresponding to each Pi in each step of the cycle in Algorithm 1 is

pi(θ′|θ) = A(θ′|θ, αi, βi)Qαi,βi(θ′|θ) + (1 −L(θ)) δθ(θ′)
(11)

where

L(θ) =
X

θ′∈Θ

π(θ′)Qαi,βi(θ|θ′)

π(θ)Qαi,βi(θ′|θ) ∧1

Qαi,βi(θ′|θ)

is the total rejection probability from θ. Finally, recall that the total variation distance between two
probability measures µ and ν, defined on some space Θ ⊂Rd is

∥µ −ν∥T V =
sup
A∈B(Θ)
|µ(A) −ν(A)|

where B(Θ) is the set of all measurable sets in Θ.

Constant Step Size and Balancing Parameter
To analyze Algorithm 1 with step size and balancing
parameter schedules, we first solve a simpler problem where the step size and balancing parameter
are fixed and then extend the analysis to the setting of Algorithm 1.

Our main method of proof is to establish uniform ergodicity of the Markov chain P, for a single α, β,
by establishing a uniform minorization for P. We denote the transition kernel for this Markov chain
P as p(· | ·), which is given in (11) with αi, βi replaced by a fixed α, β.
Lemma 5.3. Let Assumptions 5.1-5.2 with α <
1
βM hold. Then for the Markov chain P we have, for
any θ, θ′ ∈Θ,

p(θ | θ′) ≥ϵβ,α
exp {βU(θ′)}
P

θ′∈Θ exp {βU(θ′)},

where

ϵβ,α = exp

−
 1

2α + βM −β m

2


diam(Θ)2

−∥∇U(a)∥diam(Θ)}

with a ∈arg minθ∈Θ ∥∇U(θ)∥.

Proof. The proof is provided in Appendix C.1.

7


---Page Break---
Theorem 5.4. Let Assumptions 5.1-5.2 hold with α < 1/βM. Then for the Markov chain P, the
following hold:
i. P is uniformly ergodic with

∥P n −π∥T V ≤(1 −ϵβ,α)n .

ii. For any real-valued function f and samples X1, X2, X3, · · · , Xn from P, one has

√n

 
1
n

n
X

i=1
f(Xi) −
X

θ∈Θ
f(θ)π(θ)

!

d→N(0, ˜σ2
∗)

for some ˜σ∗> 0 as n →∞.

Proof. The proof directly follows from our Lemma 5.3 and Jones [2004][Corollary 5].

Note that as α →0, we have ϵβ,α →1 which implies that small step sizes result in low convergence
rates. This is intuitive as the algorithm could not explore much in this case. Furthermore, our results
suggest that large β restricts α to small values. Given that large β generally requires large α, our
findings imply an upper bound for the step size.

Adaptive Step Size and Balancing Parameter
Now we tackle the case of varying step sizes and
balancing parameters. Each cycle has s steps with step sizes α1, α2, · · · , αs and balancing parameters
β1, β2, · · · , βs. Note that this case is more challenging as at each step the transition operator changes
and the Markov chain is no longer homogeneous. However, the marginal chain for each cycle is
indeed homogeneous and can be analyzed. We present our results in this setting as follows:

Theorem 5.5. Let Assumptions 5.1 and 5.2 hold with αi < 1/βiM , i = 1, 2, · · · s. Then for the
Markov chain ˆP, the following hold
i. ˆP is uniformly ergodic with
 ˆP n −π

T V ≤(1 −ϵβs,αs)n.

ii. For any real-valued function f and samples X1, X2, X3, · · · , Xn from ˆP, one has

√n

 
1
n

n
X

i=1
f(Xi) −
X

θ∈Θ
f(θ)π(θ)

!

d→N(0, ˜σ2
∗)

for some ˜σ∗> 0 as n →∞, where,

ϵβs,αs = exp

−
 1

2αs
+ βsM −βs m

2


diam(Θ)2


· exp {−∥∇U(a)∥diam(Θ)}

with a ∈arg minθ∈Θ ∥∇U(θ)∥.

Proof. The proof follows from our Lemma 5.3, Proposition C.1 and Jones [2004][Corollary 5].

Both Theorems 5.4 and 5.5 hold uniformly over all functions in the class of functions with at least a
local minima in Θ. The Central Limit Theorem results in Theorems 5.4 and 5.5 imply that we may
perform inference on the target distribution π(·) even though the asymptotic variances are unknown,
as we may perform batch-means to estimate these variances Vats et al. [2019].

In summary, we have established a geometric convergence rate to the target distribution for our
sampler. Previous research has only established asymptotic convergence [Zhang et al., 2022b] or
relative convergence rate bounds [Grathwohl et al., 2021] for gradient-based discrete samplers. To the
best of our knowledge, our results present the first non-asymptotic convergence bounds that explicitly
quantify the distance between the estimated and target distributions. Further, our convergence bound
also shows that discrete spaces play a fundamental part in the ergodic nature of these algorithms.

8


---Page Break---
0
5000
10000
Iteration

6.5

6.0

5.5

5.0

4.5

4.0

Log MMD

mnist

GWG
DMALA
AB
ACS (Ours)

0
5000
10000

6.0

5.5

5.0

4.5

4.0

3.5

3.0

kmnist

0
5000
10000
5.0

4.5

4.0

3.5

3.0

2.5

emnist

0
5000
10000

6.6

6.4

6.2

6.0

5.8

omniglot

0
5000
10000

5.0

4.5

4.0

3.5

3.0

caltech

0
1000
2000
3000
4000
5000
Iteration

600

550

500

450

400

Average Energy

static_mnist

GWG
DMALA
AB
ACS (Ours)

0
1000
2000
3000
4000
5000

600

550

500

450

400

dynamic_mnist

0
1000
2000
3000
4000
5000

650

600

550

500

450

400

omniglot

0
1000
2000
3000
4000
5000

1400

1300

1200

1100

1000

caltech

Figure 3: Sampling performance of various methods. Top row demonstrates convergence to ground
truth on RBMs, bottom row demonstrates convergence speed on deep EBMs. We report the average
performance across 11 random seeds within 1 standard error for the top row, and we show the average
performance for the bottom row, as the error area is not visibly clear. For both distribution types,
ACS demonstrates competitive performance with all baselines.

6
Experiments

We call our method that combines Algorithm 1 and 2 Automatic Cyclical Sampler (ACS). For RBM
and EBM sampling tasks, we compare our method to Gibbs-with-Gradient (GWG) [Grathwohl et al.,
2021], Any-scale sampler (AB) [Sun et al., 2023a], and Discrete Metropolis Adjusted Langevin
Algorithm (DMALA) [Zhang et al., 2022b], which are popular and recent gradient-based discrete
samplers. For learning tasks, we omit AB sampler as it is not originally applied to the model learning
tasks. More experimental details are in Appendix D. We released our code at the following link:
https://github.com/patrickpynadath1/automatic_cyclical_sampling.

6.1
Sampling Tasks

We evaluate our sampling method on both Restricted Boltzmann Machines (RBMs) and deep
convolutional Energy-Based Models (EBMs). For RBMs, we measure accuracy by comparing the
Maximum Mean Divergence (MMD) between samples generated by our method and Block Gibbs,
which can be considered the ground truth. We sample on EBMs to demonstrate our method’s
scalability to more complex distributions. Experimental details are provided in Appendices D.2 and
D.3 for RBM and EBM sampling, respectively.

Results
In Figure 3, our proposed ACS method performs competitively for both RBMs and EBMs
across all datasets. For RBM sampling, ACS is able to converge to the ground truth quicker than
other methods due to the ability to capture the multi-modal nature of the target distribution. We see
that this performance generalizes to more complex distributions as represented by deep EBMs.

6.2
Learning RBMs and EBMs
Table 1: Deep Convolution EBM Log likelihood scores on
test data as estimated by AIS. GWG results are taken from
[Grathwohl et al., 2021]. ACS is able to achieve better results
than the baselines.

GWG*
DMALA
ACS

Static MNIST
−80.01
−80.031 ± 0.038
−79.905 ± 0.057
Dynamic MNIST
−80.51
−80.120 ± 0.036
−79.634 ± 0.024
Omniglot
−94.72
−99.243 ± 2.101
−91.487 ± 0.128
Caltech
−96.20
−98.001 ± 0.371
−89.262 ± 0.290

One common application of MCMC
techniques is learning energy-based
models (EBMs), where a neural net-
work parameterized by ϕ represents
an energy function Eϕ. These mod-
els are typically trained using Per-
sistent Contrastive Divergence (PCD)
and evaluated with Annealed Impor-
tance Sampling (AIS). Details on ACS for EBM learning are in Appendix B. We test our algorithm on

9


---Page Break---
learning deep convolutional EBMs, with experimental details in Appendix D.5. We include additional
experimentation with learning RBMs in Appendix D.4.

Results Table 1 demonstrates that ACS is capable of learning better quality EBMs given the same com-
putational budget as DMALA. Furthermore, ACS learns better quality models with less computational
budget than GWG.

6.3
Text Infilling

One challenging application of discrete MCMC methods is text-infilling, where the goal is to complete
a sentence with some missing words. Given a dataset of sentences, we randomly mask our 50% of
the words and fill them in using the distribution given by a pretrained RoBERTa model. We include
experiment details in Appendix D.6.

Results
Table 2 demonstrates that ACS is capable of generating more diverse sentences, as ACS
has a lower self-BLEU and higher percentage of unique n-grams. While the perplexity results seem
to imply that ACS generates lower quality than DMALA, we note that the ACS generations are more
likely to be predicted as linguistically acceptable as shown by the CoLA scores. We discuss the
results more extensively in Appendix D.6.

Dataset
Method
Perplexity (↓)
CoLA (↑)
Self-Bleu (↓)
Unique n-gram (↑)

n=2
n=3

Grimm
DMALA
280.82 ± 27.26
50.46 ± 1.25
41.83 ± 6.85
48.55
70.56
ACS
369.44 ± 30.85
53.42 ± 1.26
36.70 ± 6.42
53.91
74.70

SST2
DMALA
256.66 ± 10.53
42.62 ± 1.14
37.47 ± .79
57.68
75.21
ACS
307.05 ± 14.84
47.12 ± 1.20
32.42 ± .75
62.54
78.87
Table 2: Empirical evaluation of the generated sentences. ACS outperforms DMALA for all metrics
related to diversity.

7
Conclusion and Limitations

In this work, we propose Automatic Cyclical Sampler (ACS) to more effectively characterize
multimodal distributions in discrete spaces. First, we demonstrate that gradient-based samplers are
prone to getting trapped in local modes, preventing a full characterization of target distributions. To
address this issue, we combine a cyclical step size schedule with a cyclical balancing parameter
schedule along with an automatic tuning algorithm to configure these schedules. We also theoretically
establish the non-asymptotic convergence bound of our method to the target distribution in addition
to providing extensive experimental results.

While our proposed ACS method generates impressive results on a wide range of experiments, there
are some limitations to our work that should be mentioned. Specifically, though we have proven
a geometric convergence rate and the relationship between α and β in our theoretical analysis, we
require U(·) to be twice differentiable as well as locally strongly concave and the proof is not based
on the specific tuning algorithm implemented. This is why we provide extensive experimentation to
demonstrate that our algorithm is capable of picking good α, β schedules.

10


---Page Break---
References

Bernd A Berg and Thomas Neuhaus. Multicanonical algorithms for first order phase transitions.
Physics Letters B, 267(2):249–253, 1991.

L´eon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine
learning. Siam Review, 60(2):223–311, 2018.

Arnak S Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave
densities. Journal of the Royal Statistical Society Series B: Statistical Methodology, 79(3):651–676,
2017.

Wei Deng, Qi Feng, Liyao Gao, Faming Liang, and Guang Lin. Non-convex learning via replica
exchange stochastic gradient mcmc. In International Conference on Machine Learning, pages
2474–2483. PMLR, 2020a.

Wei Deng, Guang Lin, and Faming Liang. A contour stochastic gradient langevin dynamics algorithm
for simulations of multi-modal distributions. Advances in neural information processing systems,
33:15725–15736, 2020b.

Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances
in Neural Information Processing Systems, 32, 2019.

Katayoon Goshvadi, Haoran Sun, Xingchao Liu, Azade Nova, Ruqi Zhang, Will Sussman Grathwohl,
Dale Schuurmans, and Hanjun Dai. Discs: A benchmark for discrete sampling. In Thirty-seventh
Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris J. Maddison. Oops I
took A gradient: Scalable sampling for discrete distributions. CoRR, abs/2102.04509, 2021. URL
https://arxiv.org/abs/2102.04509.

Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural
computation, 14(8):1771–1800, 2002.

Galin L Jones. On the markov chain central limit theorem. 2004.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining
approach. arXiv preprint arXiv:1907.11692, 2019.

Enzo Marinari and Giorgio Parisi. Simulated tempering: a new monte carlo scheme. Europhysics
letters, 19(6):451, 1992.

Radford M Neal. Annealed importance sampling. Statistics and computing, 11:125–139, 2001.

Benjamin Rhodes and Michael Gutmann. Enhanced gradient-based mcmc in discrete spaces, 2022.

Sebastian Ruder.
An overview of gradient descent optimization algorithms.
arXiv preprint
arXiv:1609.04747, 2016.

Emanuele Sansone. Lsb: Local self-balancing mcmc in discrete spaces. In International Conference
on Machine Learning, pages 19205–19220. PMLR, 2022.

Haoran Sun, Hanjun Dai, Wei Xia, and Arun Ramamurthy. Path auxiliary proposal for mcmc in
discrete space. In International Conference on Learning Representations, 2021.

Haoran Sun, Hanjun Dai, and Dale Schuurmans. Optimal scaling for locally balanced proposals in
discrete spaces, 2022.

Haoran Sun, Bo Dai, Charles Sutton, Dale Schuurmans, and Hanjun Dai. Any-scale balanced samplers
for discrete space. In The Eleventh International Conference on Learning Representations, 2023a.
URL https://openreview.net/forum?id=lEkl0jdSb7B.

Haoran Sun, Hanjun Dai, Bo Dai, Haomin Zhou, and Dale Schuurmans. Discrete langevin samplers
via wasserstein gradient flow. In International Conference on Artificial Intelligence and Statistics,
pages 6290–6313. PMLR, 2023b.

11


---Page Break---
Robert H Swendsen and Jian-Sheng Wang. Replica monte carlo simulation of spin-glasses. Physical
review letters, 57(21):2607, 1986.

Robert H Swendsen and Jian-Sheng Wang. Nonuniversal critical dynamics in monte carlo simulations.
Physical review letters, 58(2):86, 1987.

Tijmen Tieleman. Training restricted boltzmann machines using approximations to the likelihood
gradient. In Proceedings of the 25th international conference on Machine learning, pages 1064–
1071, 2008.

Dootika Vats, James M Flegal, and Galin L Jones. Multivariate output analysis for markov chain
monte carlo. Biometrika, 106(2):321–337, 2019.

Ulli Wolff. Collective monte carlo updating for spin systems. Physical Review Letters, 62(4):361,
1989.

Yue Xiang, Dongyao Zhu, Bowen Lei, Dongkuan Xu, and Ruqi Zhang. Efficient informed proposals
for discrete distributions via newton’s series approximation. In International Conference on
Artificial Intelligence and Statistics, pages 7288–7310. PMLR, 2023.

Giacomo Zanella. Informed proposals for local mcmc in discrete spaces, 2017.

Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, and Andrew Gordon Wilson. Cyclical
stochastic gradient mcmc for bayesian deep learning. In International Conference on Learning
Representations, 2020. URL https://openreview.net/forum?id=rkeS1RVtPS.

Ruqi Zhang, Xingchao Liu, and Qiang Liu. A langevin-like sampler for discrete distributions. In
International Conference on Machine Learning, pages 26375–26396. PMLR, 2022a.

Ruqi Zhang, Xingchao Liu, and Qiang Liu. A langevin-like sampler for discrete distributions, 2022b.

Liu Ziyin, Botao Li, James B Simon, and Masahito Ueda. Sgd can converge to local maxima. In
International Conference on Learning Representations, 2021.

12


---Page Break---
A
Details of Automatic Cyclical Sampler Algorithm

Here we include more details regarding the Automatic Cyclical Sampler algorithm. We discuss all the
individual sub-routines that compose the algorithm shown in Algorithm 2. We also include an ablation
study to demonstrate the robustness of our algorithm to various hyper-parameter configurations.

InitialBurnin
We find that in order to produce meaningful estimates for the objective in (9), it is
necessary to burn in the MCMC sampling chain. This is due to the dependence of the acceptance
rate on current sample θ. If we use θ very low in density with respect to the target distribution, the
acceptance rates estimated by the tuning algorithm will lose accuracy as the sampler converges to the
target distribution. In order to avoid this issue, we run a quick burn-in stage with two distinct stages.

The first stage uses the gradient information to move the sampler away from the initialized point as
quickly as possible. We use the parameterized proposal from Equation (4) with stepsize αceil, βmax
without any Metropolis-Hastings correction as this enables very large movements from the initial
sample.

For some datasets, this enables a very quick burn-in.
This can be noticed in Figure 3 for
Static/Dynamic MNIST and Omniglot. We hypothesize that this is due to the distribution hav-
ing a relatively simple structure that enables the gradient to provide meaningful information for very
large sampling steps. It is impossible to determine a priori whether a given distribution will have this
property, so we include a following stage that uses a Metropolis-Hastings correction to increase the
chance of arriving at a reasonable sample θ.

For this stage, we construct a naive step size schedule and balancing constant schedule using the
values of αceil, αfloor, βmax, βmin. We then run the parameterized sampler from Equation (4) with the
Metropolis-Hastings correction. Our goal is to move the sampler to samples θ that are more likely in
the target distribution. This will enable the acceptance rates computed during the tuning algorithm to
be closer to the acceptance rates for the steady-state chain.

For all the sampling experiments, these two stages combined use 100 sampling steps.

EstimateAlpha
Here we discuss the algorithm used to calculate both αmax, αmin as defined in
Equation (10). When calculating αmax, the goal is to pick the largest stepsize αmax that acheives the
acceptance rate ρ∗for a given βmax. When calculating αmin, the goal is to determine the smallest
step-size capable of acheiving the target acceptance rate. We put the full pseudo-code in Algorithm 4.

For calculating αmax and αmin, the algorithm follows the general pattern of automatically shifting the
range of potential α based on the best values calculated from the previous iteration. When calculating
αmax, the algorithm starts with an upper-bound initialized to αbound = αceil and iteratively decreases
the range of proposed α. For αmin, the algorithm starts with a lower bound αbound = αfloor and
iteratively increases the range. For both, the other bound is calculated by the following learning rule:

αprop = αbound ± ζ|ρ −ρ∗|.

Here, ζ is the learning rate that determines how much we can adjust the step size in one tuning
step. We found ζ insensitive and set ζ = .5 in all tasks. Additionally, ρ is the best acceptance rate
computed from the previous iteration of the algorithm. For the first step of the algorithm, we set
ρ = 0.

The algorithm uses αprop, αbound to determine the range of α to test. For calculating αmax, the
algorithm searches in the range of [αprop, αbound]. For calculating αmin, the range is [αbound, αprop].

Given the appropriate range of α and an initial θ, we test t potential α and calculate their respective
acceptance rates using Equation (6). Once we have computed all the acceptance rates, we set αbound
to the value that resulted in the most optimal acceptance rate as determined by Equation (9), θ to the
corresponding θ′, and ρ to the corresponding acceptance rate.

Choice of βmax, βmin, ρ∗, s
The automatic tuning algorithm depends on an initial choice of
βmax, , βmin, ρ∗, s that enable it to automatically configure an effective hyper-parameter schedule.
Here we describe the general approach to picking these values.

For some target distributions, it is possible that the best possible acceptance rate with a very high
βmax, such as βmax = .95, will not be close to the target acceptance rate ρ∗. In this case, the

13


---Page Break---
EstimateAlphaMax algorithm will keep on decreasing the proposed αmax, which will result in a very
small αmax. In order to avoid this behavior, we recommend starting with βmax = .95, and decreasing
it by .05 if the resulting αmax is reasonable.

We always set βmin = 0.5 which is the smallest value β can take.

We determine the target ρ∗by starting with a value of .5 and increasing it by .05 until desirable
performance metrics are obtained. While this process is essentially the same as a grid search, we
note that we only needed to apply this process in the specific case of training a deep EBM on the
Caltech Silhouettes dataset. For all other tasks and datasets, the target acceptance rate of ρ∗= .5 was
effective. We discuss the unique difficulty presented within the Caltech Silhouettes dataset in D.5.

To determine the steps per cycle s, we required a similar approach to determine the optimal value.
In our experiments, we only look at two different values: either s = 8, or s = 20. Having a longer
cycle length tends to enable more exploitation of the target distribution, whereas having a shorter
cycle enables more exploration. While we do not have an algorithm for automatically configuring
this value, we were able to achieve good results across all tasks and datasets by choosing either of
these two values. For more details on the resulting hyper-parameters used for each experiment, see
Appendix D.

A.1
Hyper-parameter Sensitivity

Our method introduces the following hyperparameters: βmax, βmin, αceil, αfloor, learning rate for tuning
γ, steps per cycle s, target acceptance rate ρ∗, and budget B. This may seem like many additional
hyperparameters, but the majority of these are introduced due to the automatic tuning mechanism and
are not changed across all tasks and datasets in the paper: γ = .5, αfloor = .05, αceil = 5, βmin = .5,
B = 200. Thus the only hyperparameters requiring tuning in practice are βmax, ρ∗, and s. Note that
the existing adaptive discrete sampler, any-scale sampler introduced in [Sun et al., 2023a], has a
similar number of hyper-parameters: initial step size σ, initial balancing parameter α, update rate
γ, decay rate β, buffer size N, initial Hessian matrix W, and initial diagonal matrix D. Like our
method, most of these hyperparameters are fixed across experiments.

We conduct an ablation study to evaluate the sensitivity of our tuning algorithm to these hyperparam-
eters choices. We choose one hyperparameter at a time to ablate and keep the rest at default values of
the hyperparameters at their default setting. We run the RBM sampling experiment over multiple
datasets, each for 10 random seeds, and report the average results in Figure 4. We omit the standard
error as that would harm the interpretability of the graph as many of the plots are quite close together.

We can summarize the main takeaways as follows:

1. The sensitivity of our algorithm to the hyperparameters depends on the dataset. For example,
the sensitivity of our algorithm is low on MNIST, kMNIST, eMNIST, and Omniglot while
the sensitivity is relatively high on Caltech.

2. The optimal hyperparameter values depend on the dataset. For example, high values of s
generally yield superior results, except for Caltech, where lower values excel. Similarly, low
βmax values are usually less effective, though Caltech is an exception, showcasing decent
outcomes. In general, the hyperparameter values we selected to generate the final results in
the experiment section were the ones that generalized across the datasets.

3. For each ablation, the values tested demonstrate reasonable results when compared with the
baselines. While not all hyperparameter values result in equally competitive performance,
all of them outperform the Gibbs-With-Gradient sampler Grathwohl et al. [2021]. This
demonstrates that our method performs well with a wide range of hyperparameters and can
achieve even better performance with careful hyperparameter tuning.

In conclusion, we believe these results demonstrate that our algorithm is relatively robust to choice in
hyperparameters.

14


---Page Break---
Algorithm 3 InitBurnin

Require: αceil, αfloor, βmax, βmin, steps per cycle s, steps to take without MH correction l = 50, steps
to take with MH correction lMH = 50, initial state θ
1: for step i in range(l) do
2:
θ ∼Qαceil,βmax(·|θ) ▷Run burnin steps without MH correction
3: end for
4: {α0, α1 · · · αs−1} ←values from Equation (7) using αceil, αfloor.
5: {β0, β1 · · · βs−1} ←values from Equation (7) using βmax, βmin ▷We can use Equation (7) to get
interpolations of β
6: number of cycles n = floor( lMH

s )
7: Obtain θ by running Algorithm 1 using the calculated α, β schedule ▷Run burnin steps with MH
correction
8: return θ

Algorithm 4 EstimateAlpha

Require: αbound, BUDGET, initial state θ, Balancing parameter β, target acceptance rate ρ∗, learning
rate ζ, number of proposals per step t = 5, flag MAX
1: ρcur ←0
2: while iteration i ≤BUDGET do
3:
if MAX then
4:
αprop = α(1 −ζ|ρ∗−ρcur|) ▷adaptively decrease the range of potential α
5:
proposed-params ←LinSpace(αprop, αbound, t) ▷we use αbound = αceil as the ceiling for
proposed α
6:
else
7:
αprop = α(1 + ζ|ρ∗−ρcur|) ▷For AlphaMin, adaptively increase the range of potential α
8:
proposed-params ←LinSpace(αbound, αprop, t) ▷For AlphaMin, use αbound = αfloor as the
floor for proposed α
9:
end if
10:
initialize bookkeeping to keep track of proposed states and acceptance rates
11:
for α ∈proposed-params do
12:
θ′ ∼Qαprop,β(·|θ) ▷Use proposed α to take sampling step
13:
ρ = A(θ′|θ, αprop, β) ▷Compute acceptance rate for proposed α
14:
i = i + 1
15:
end for
16:
Set ρcur to the acceptance rate closest to the target a∗

17:
Set αbound to the corresponding α ▷Update αbound to shift the range of proposed α for the next
step
18:
set θcur to the corresponding θ
19: end while
20: if MAX then
21:
return αmax = αbound
22: else
23:
return αmin = αbound
24: end if

15


---Page Break---
0
500
1000
Iteration

6.5

6.0

5.5

5.0

4.5

4.0

3.5

Log MMD

mnist

rho*=0.3
rho*=0.4
rho*=0.5
rho*=0.6
rho*=0.7
rho*=0.8
rho*=0.9

0
500
1000

6.0

5.5

5.0

4.5

4.0

3.5

3.0

kmnist

0
500
1000
5.0

4.5

4.0

3.5

3.0

2.5

2.0
emnist

0
500
1000

6.6

6.4

6.2

6.0

omniglot

0
500
1000
6

5

4

3

caltech

0
500
1000
Iteration

6.5

6.0

5.5

5.0

4.5

4.0

3.5

3.0

Log MMD

bmax=0.6
bmax=0.7
bmax=0.8
bmax=0.9
bmax=1.0

0
500
1000

6.0

5.5

5.0

4.5

4.0

3.5

3.0

0
500
1000

4.5

4.0

3.5

3.0

2.5

2.0

0
500
1000

6.6

6.4

6.2

6.0

5.8

0
500
1000

5.0

4.5

4.0

3.5

3.0

2.5

0
500
1000
Iteration

6.5

6.0

5.5

5.0

4.5

4.0

3.5

3.0

Log MMD

s=50
s=100
s=250
s=500
s=1000
s=5000

0
500
1000

6.0

5.5

5.0

4.5

4.0

3.5

3.0

0
500
1000

4.5

4.0

3.5

3.0

2.5

2.0

0
500
1000

6.6

6.4

6.2

6.0

0
500
1000

5.5

5.0

4.5

4.0

3.5

3.0

2.5

Figure 4: Average performance across multiple seeds for various hyper-parameter settings. We
note that all configurations to exhibit convergence to the ground truth as indicated by the maximum
mean discrepancy (log MMD), albeit with varying convergence speeds. In some cases, specific
hyper-parameter configurations are able to achieve better performance than what we report in the
RBM sampling experiment. Overall, we can observe that our algorithm is reasonably robust to
various hyper-parameter configurations as it will still demonstrate convergent behavior towards the
ground truth.

B
ACS for EBM Learning

B.1
Background

Energy Based Models (EBMs) are a class of generative models that learn some unnormalized
distribution over a sample space. As discussed in Hinton [2002], these models can be trained via the
following Maximum Likelihood objective:

L(ϕ) = Ex∼pdata [−log pϕ(x)]
(12)

The gradient updates for this loss function are known to be as follows:

∇ϕL(ϕ) = Ex∼pdata [∇ϕEϕ(x)] −Ex∼pϕ [∇ϕEϕ(x)]
(13)

While the expectation on the left is straight forward to calculate given a dataset, calculating the right
expectation is not as clear. Here we will mention the two methods that are relevant towards our
experiments with EBMs.

Contrastive Divergence (CD)
In order to estimate the second term, we initialize some sampler
using the x in the first term and run it for a set number of sampling steps. For a more detailed
description, refer to Hinton [2002].

Persistent Contrastive Divergence (PCD)
The expectation on the right can be calculated using
samples from a persistent Markov Chain that approximates the true distribution Tieleman [2008].

16


---Page Break---
Algorithm 5 EstimateBalSched

Require: Step size schedule {αmax, α1, . . . αmin}, βmax, βmin, number of proposals per step t = 10,
initial state θ, target acceptance rate ρ∗

1: βfloor = βmin, βceil = βmax
2: β-sched ←{βmax}
3: for i in {1, 2, . . . s −1} do
4:
proposed-params ←LinSpace(βfloor, βmax, t) ▷Create t potential balance parameters for
index i in the schedule
5:
initialize bookkeeping to keep track of proposed states and acceptance rates
6:
for β ∈proposed-params do
7:
θ′ ∼Qαi,β(·|θ) ▷Use current proposed β to take a sampling step
8:
ρ = A(θ′|θ, αi, β) ▷Evaluate the acceptance rate of proposed β for current αi
9:
bookkeeping[β] ←θ′, ρ
10:
end for
11:
pick βi as β ∈bookkeeping largest ρ
12:
βceil ←βi ▷Shrink the range of potential balancing parameters by using assumption βi >
βi+1
13:
θ = θ′ correspending to βi
14: end for
15: β-sched.append(βmin)
16: return β-sched

Instead of resetting the chain each training iteration, we maintain a buffer of the generated samples
that we use to calculate the second expectation. This method relies on the intuition that the model
distribution does not vary too widely within one iteration. Using the intuition provided by [Du and
Mordatch, 2019], we can view this process as updating the model parameters ϕ to put more weight on
true samples and less weight on fake samples. By doing so, the model will in turn generate samples
that closer to those from the true distribution.

B.2
Persistent Contrastive Divergence with ACS

Main Idea
We can apply the ACS algorithm combining the automatic tuning of the cyclical
schedule with the original PCD learning algorithm. Our goal in doing so is to improve PCD through
better characterization of the entire model distribution. During training, we can view PCD as adjusting
the model parameters to “push down” the probability of samples from the model distribution while
“pushing up” samples from the true data distribution. Because our sampling method is able to explore
the model’s distribution more effectively than other samplers, we can adjust more regions of the
model distribution at a quicker rate than previous sampling methods, which should improve the
quality of gradient updates and thus lead to better model parameters. We adapt ACS to work within
PCD by having the step size depend on the training iteration as opposed to the sampling iteration,
with the corresponding α, β pair being used for all the sampling steps within the iteration. We include
the complete learning algorithm in Algorithm 6.

Cyclical Scheduling
We find that the learning task requires a different approach to the cyclical
scheduling than the sampling task. Rather than having a relative equal amount of exploration and
exploitation, we find that it is more effective to use a cyclical schedule biased towards exploitation.
However, exploration is still important as it enables the model to better represent the distribution as a
whole rather than a few local modes. Given this, we construct a cyclical schedule consisting of one
iteration that uses αmax, βmax with the rest using αmin, βmin.

Tuning
One of the advantages of using the simplified cyclical schedule is that it only requires
two pairs of hyper-parameters to be optimized. Thus we can leverage the EstimateAlphaMax and
EstimateAlphaMin algorithm to both tune the respective α, β pair while also updating the persistent
buffer. Not only does this reduce the additional overhead of the tuning component, but it allows the
hyper-parameters to adapt to the changing EBM distribution.

17


---Page Break---
Algorithm 6 ACS for Persistent Contrastive Divergence

Require: Number Iterations N, EBM Eϕ, data-loader D, sampler Q, small sampling steps Ssmall,
big sampling steps Sbig , initial buffer Xf, cycle length s, αfloor, αceil, adaptive learning rate ζ,
adaptive budget BUDGET
1: while iteration i ≤N do
2:
for Xt ∼D do
3:
cycle number c = floor( i

C )
4:
if c mod K = 0 then
5:
if i mod s = 0 then
6:
Xf, αmax ←EstAlphaMax(αceil, budget=BUDGET, learning-rate =γ)
7:
else
8:
Xf, αmin ←EstAlphaMin(αfloor, budget=BUDGET, learning-rate=γ)
9:
end if
10:
Update Sampler Step Schedule ▷Update the buffer by running either the AlphaMax or
AlphaMin estimation algorithm
11:
else
12:
if i mod s = 0 then
13:
S = Sbig
14:
α = αmax, β = βmax ▷Use the α, β pair that best enables exploration
15:
else
16:
S = Ssmall
17:
α = αmin, β = βmin ▷Use the α, β pair that best enables exploitation
18:
end if
19:
Construct Q = Qα,β(·|Xf) using (4)
20:
for sampling step in range(Sbig) do
21:
X ∼Q(·|Xf)
22:
if i mod s = 0 then
23:
Xf ←X
24:
continue ▷If i is the first step of the cycle, omit the MH correction
25:
end if
26:
Xf ←X with acceptance probability as calculated in (6)
27:
end for
28:
end if
29:
Calculate Ex∼pϕ[∇ϕEϕ(x)] using Xf
30:
Calculate Ex∼pdata[∇ϕEϕ(x)] using Xt
31:
∇L(ϕ) = Ex∼pϕ[∇ϕEϕ(x)]−Ex∼pdata[∇ϕEϕ(x)] ▷Estimate the gradient of the Maximum-
Likelihood objective as in (12)
32:
ϕ = ϕ −γϕ∇L(ϕ)
33:
i+ = 1
34:
end for
35: end while

18


---Page Break---
C
Theoretical Results

We define the problem setting in more detail. We have a target that is of the form

π(θ) = 1

Z exp(U(θ)).

We consider the proposal kernel as

Qα,β(θ′|θ) ∝exp

β∇U(θ)T (θ′ −θ) −1

2α∥θ′ −θ∥2


and consider the transition kernel as

p(θ′ | θ) =
π(θ′)Qα,β(θ | θ′)

π(θ)Qα,β(θ′ | θ) ∧1

Qα,β(θ′ | θ) + (1 −L(θ)) δθ(θ′)

where δθ(θ′) is the Kronecker delta function and L(θ) is the total acceptance probability from the
point θ with

L(θ) =
X

θ′∈Θ

π(θ′)Qα,β(θ|θ′)

π(θ)Qα,β(θ′|θ) ∧1

Qα,β(θ′|θ).

We also define

Zα,β(θ) =
X

x∈Θ
exp

β ∇U(θ)T (x −θ) −1

2α∥x −θ∥2


which is the normalizing constant for the proposal kernel.

C.1
Proof of Lemma 5.3

Proof. By including the balancing parameter, we start by noting that

Qα,β(θ′|θ) =
exp

β∇U(θ)T (θ′ −θ) −
1
2α∥θ′ −θ∥2	

P

θ∈Θ exp

β ∇U(θ)T (θ −θ) −
1
2α∥θ −θ∥2	
(14)

Consider the term,

β ∇U(θ)T (θ′ −θ) = β (−U(θ) + U(θ′)) −β

2 (θ −θ′)T (
Z 1

0
∇2U((1 −s)θ + sθ′) ds)(θ −θ′)

(15)

Substituting (15) in (14), the numerator of Qα,β(θ, θ′)

β∇U(θ)T (θ′ −θ) −1

2α∥θ′ −θ∥2 =β (−U(θ) + U(θ′))

−β

2 (θ −θ′)T
Z 1

0
∇2U((1 −s)θ + sθ′) ds

(θ −θ′)

−1

2α(θ −θ′)T I(θ −θ′)

=β (−U(θ) + U(θ′))

−1

2(θ −θ′)T

β
Z 1

0
∇2U((1 −s)θ + sθ′) ds + 1

αI

(θ −θ′)

.

From Assumption 5.1 (U is M-gradient Lipschitz), we have

β
Z 1

0
∇2U((1 −s)θ + sθ′) ds)(θ −θ′) + 1

αI ≥
 1

α −βM

I

19


---Page Break---
Since α < 1/βM, the matrix
  1

2α −βM

I is positive definite. We note that

p(θ′|θ) =
π(θ′)Qα,β(θ|θ′)

π(θ)Qα,β(θ′|θ) ∧1

Qα,β(θ′|θ) + (1 −L(θ)) δθ(θ′)
(16)

≥
π(θ′)Qα,β(θ|θ′)

π(θ)Qα,β(θ′|θ) ∧1

Qα,β(θ′|θ)
(17)

=
 Zα,β(θ)

Zα,β(θ′) ∧1

Qα,β(θ′|θ).
(18)

Zα,β(θ) =
X

x∈Θ
exp

β ∇U(θ)T (x −θ) −1

2α∥x −θ∥2


=
X

x∈Θ
exp

−β (U(θ) −U(x)) −1

2(θ −x)T (β
Z 1

0
∇2U((1 −s)θ + sx) ds)(θ −x) + 1

αI)(θ −x)

.

This can be seen as

π(θ)Qα,β(θ′|θ) =
1
Z Zα,β(θ) exp

β (U(θ) + U(θ′)) −(θ′ −θ)T
 1

2αI + β

2

Z 1

0
∇2U((1 −s)θ + sθ′)ds

(θ′ −θ)

.

Since Assumption 5.2 holds true in this setting, we have an m > 0 such that for any θ ∈conv(Θ)

−∇2U(θ) ≥m I.
From this, one notes that

exp

−βU(θ) −1

2

 1

α −β m

diam(Θ)2
 X

x∈Θ
exp (βU(x)) ≤Zα,β(θ) ≤exp (−βU(θ))
X

x∈Θ
exp (βU(x))

where the right-hand side follows from the fact that α < 1/(βM). Therefore,
Zα,β(θ)
Zα,β(θ′) ≥
exp {β (−U(θ) + U(θ′))}
exp
 1

2
  1

α −βm

diam(Θ)2	

Also note that

Qα,β(θ′|θ) =
exp
n
β (−U(θ) + U(θ′)) −(θ −θ′)T 
1
2αI + β

2
R 1
0 ∇2U((1 −s)θ + sθ′)

(θ −θ′)
o

P

θ′∈Θ exp
n
β (−U(θ) + U(θ′)) −(θ −θ′)T 
1
2αI + β

2
R 1
0 ∇2U((1 −s)θ + sθ′)

(θ −θ′)
o

≥exp

β ⟨∇U(θ), θ′ −θ⟩−
1
2α∥θ −θ′∥2	
P

θ′Θ exp {β (−U(θ) + U(θ′))}
.

We also note that

−β ⟨∇U(θ), θ′ −θ⟩+ 1

2α∥θ −θ′∥2 = β ⟨−∇U(θ) + ∇U(a), θ′ −θ⟩+ β ⟨−∇U(a), θ′ −θ⟩+ 1

2α∥θ −θ′∥2

≤β ⟨−∇U(θ) + ∇U(a), θ′ −θ⟩+ β ⟨−∇U(a), θ′ −θ⟩+ 1

2αdiam(Θ)2

≤β ∥−∇U(θ) + ∇U(a)∥∥θ′ −θ∥+ β ∥∇U(a)∥∥θ′ −θ∥+ 1

2αdiam(Θ)2

≤β∥−∇U(θ) + ∇U(a)∥diam(Θ) + β∥∇U(a)∥diam(Θ) + 1

2αdiam(Θ)2

≤

βM + 1

2α


diam(Θ)2 + β∥∇U(a)∥diam(Θ).

Combining, we get

p(θ′|θ) ≥ϵβ,α
exp {βU(θ′)}
P

θ′Θ exp {βU(θ′)}
where

ϵβ,α = exp

−
 1

α + βM −β m

2


diam(Θ)2 −∥∇U(a)∥diam(Θ)

.

20


---Page Break---
C.2
Proofs of Proposition C.1 and Corollary C.2

Proposition C.1. Let P1, P2, · · · Ps be Markov transition operators with kernels p1, p2, · · · ps with
respect to a reference measure η. Also, let pi(θ′|θ) ≥ϵiνi(θ′) for some density νi on Θ and ϵi > 0
with respect to a reference measure η. Then, for the Markov operator ˆPi defined with respect to the
kernel as

ˆpi(θ′|θ) =
Z

ΘS−1pi+1(θ1|θ)pi+2(θ2|θ1) · · · ps(θs−i+1|θs−i)

· · · pi(θ′|θs−1)dη(θ1)dη(θ2) · · · dη(θs−1),

we have

ˆpi(θ′|θ) ≥ϵiνi(θ′), ∀θ ∈Θ·

Proof. The proof is straightforward by using the minorization of pi. Indeed, one has

ˆp(θ′|θ) =
Z

ΘS−1 pi+1(θ1|θ)pi+2(θ2|θ1) · · · ps(θs−i+1|θs−i) · · · pi(θ′|θs−1)dη(θ1)dη(θ2) · · · dη(θs−1)

≥ϵiνi(θ′)
Z

ΘS−1 pi+1(θ1|θ)pi+2(θ2|θ1) · · · ps(θs−i+1|θs−i) · · · pi−1(θs−1|θs−2) dη(θ1) · · · dη(θs−1)

≥ϵiνi(θ′)

which establishes the result.

Note that in Algorithm 1, for each cycle, we go through s steps corresponding to the step size and
balancing parameter schedules ({α1, α2, · · · αs}) and ({β1, β2, · · · βs}). Let P1, P2, · · · , Ps be the
Markov operators corresponding to them.
Corollary C.2. Let Assumptions 5.1 and 5.2 hold. Then

P1P2P3 · · · Ps(θ, A) ≥ϵsνs(A)

for any measurable subset A of Θ.

Proof. The proof is immediate from Proposition C.1.

C.3
Additional Lemma

Lemma C.3. Let Assumption 5.2 hold with Θ compact. Then, there exists some m > 0 such that for
any θ ∈conv(Θ), λmin(∇2 −U(θ)) > m.

Proof. Note that since Θ is compact conv(Θ) is also compact. This is easy to see as we only need
to establish that conv(Θ) is closed and bounded by the Heine-Borel Theorem. Take any element
in θ ∈conv(Θ). By definition, θ = αθ1 + (1 −α)θ2 for some θ1, θ2 ∈Θ and 0 ≤α ≤1. Since
Θ is compact, we know that there exists M > 0 such that ∥θi∥< M for i = 1, 2. Therefore
∥θ∥< M by triangle inequality. Thus the set is bounded. The fact that it is closed is also easy
to see. Take any sequence xn in conv(Θ). This implies there exists αn, θ1,n, θ2,n such that xn =
αn θ1,n + (1 −αn)θ2,n. Since xn converges as our assumption, it is Cauchy which in turn implies
each of αn, θ1,n, θ2,n is Cauchy as Θ is bounded. Thus the proof immediately follows. Now, consider
each θ ∈conv(Θ). There exits a B(θ, rθ) such that ∇2 −U(θ′) ≥mθI for all θ′ ∈B(θ, rθ). Since
conv(Θ) ⊂∪θ∈ΘB(θ, rθ), this is an open cover of conv(Θ). Since conv(Θ) is compact, there exists
θ1, θ2, · · · , θk such that conv(Θ) ⊂∪k
i=1B(θi, rθi). Thus for each i we have ∇2 −U(θ′) ≥mθiI
when θ ∈B(θi, rθi). Thus ∇2 −U(θ) ≥min1≤i≤k mθiI for all θ ∈conv(Θ). Hence we are
done.

D
Additional Experimental Results and Details

Here, we include the full details for all the experiments we include in this paper, as well as some
additional results. All experiments were run on a single RTX A6000.

21


---Page Break---
D.1
Multi-modal Experiment Design

Synthetic Distribution
In order to construct a distribution that is easy to visualize, we first must
define a few experiment parameters. We must define the space between the modes, the total number
of modes, and the variance of each mode σ. For convenience, we have the number of modes as 25,
which is a perfect square. We define the space between modes as 75, and the variance for each mode
σ2 as .15. Given this, we can calculate the maximum value for each coordinate as follows:

MaxVal = (
√

NumModes + 1) ∗SpaceBetweenModes

We can calculate the coordinate value for each mode µi,j as follows:

µi,j[0] =
MaxVal
√

NumModes + 2
(i + 1)

µi,j[1] =
MaxVal
√

NumModes + 2
(j + 1)

Sampler Configuration
Our goal in this experiment is to demonstrate how gradient-based samplers
typically behave when faced with a distribution with modes that are far apart. In order for this
experiment to be meaningful, it is important that the representation of each sample respect the notion
of distance between the integer values. For this reason, we cannot use a categorical distribution or
represent each coordinate with a one-hot encoding, as every sample in this representation would be
within a 2-hamming ball of every other point.

Figure 5: Uneven multi-modal target distribution.
While the top-left mode does have the most mass,
only sampling from this mode will result in an
inaccurate representation of the target distribution.

In order to determine the step sizes for the base-
lines, we tune each until we reach an acceptance
rate around .574. For DMALA, this ends up be-
ing around α = 53. For the any-scale sampler,
we set the initial step size to be the same and use
their implemented adaptive algorithm.

For the cyclical sampler, we set αmax = 1575,
αmin = 3, and steps per cycle s = 20. Because
the goal of the experiment is to demonstrate the
need for larger step sizes along with smaller step
sizes, we do not use the automatic tuning algo-
rithm on this example as restricting the space
to be ordinal changes the optimal setting for
αceil. In most practical cases, the samples would
be represented by a categorical or binary form,
which the proposed tuning algorithm is able to
handle as demonstrated by the performance on
real data distributions.

Uneven Multi-modal Distributions
Not only
does a cyclical step-size enable more accurate
sampling in highly multi-modal distributions,
but it is also able to handle distributions where
the modes are weighted unevenly. This problem
is more difficult since this requires not only exploring all the modes of a distribution, but ensuring
that the less likely modes are not over represented in the generated samples. We provide a visual com-
parison between the target distribution, the estimated distribution from DMALA, and the estimated
distribution from ACS in Figure 5.

Since the modes may not be clear due to the nature of this specific problem, we also include a
quantitative comparison between DMALA and ACS in Table 3 by computing the KL divergence
between the estimated distribution and the target distribution in addition to the average energy of the
generated samples. Through both Table 3 and Figure 5, we observe that a cyclical step-size enables
accurate sampling from uneven multi-modal distributions.

22


---Page Break---
Table 3: Quantitative comparison of sampler performance
on uneven multi-modal distribution. ACS retains the ability
to accurately capture all the modes within the distribution
despite the uneven weighting.

DMALA
ACS

KL Divergence
0.70
0.13
Average Energy
−2.66 ± 1.68
−3.39 ± 1.63

Furthermore, it is interesting to ob-
serve that generating more high-
probability samples does not necessar-
ily correspond to accurate sampling,
highlighting the difference in goals be-
tween generating very likely samples
and accurately sampling the target dis-
tribution.

D.2
RBM Sampling

RBM Overview
We will give a brief overview of the Block-Gibbs sampler used to represent the
ground truth of the RBM distribution. For a more in-depth explanation, see Grathwohl et al. [2021].
Given the hidden units h and the sample x, we define the RBM distribution as follows:

log p(x, h) = hT Wx + bT x + cT −log Z
(19)

As before, Z is the normalizing constant for the distribution. The sample x is represented by the
visible layer with units corresponding to the sample space dimension and h represents the model
capacity. It can be shown that the marginal distributions are as follows:
p(x|h) = Bernoulli(Wx + c)

p(h|x) = Bernoulli(W th + b)

The Block-Gibbs sampler updates x and h alternatively, allowing for many of the coordinates to get
changed at the same time, due to utilizing the specific structure of the RBM model.

Experiment Setup
Similar to the experimental setup of Zhang et al. [2022a], we use RBM models
with 500 hidden units and 784 visible units. We adopt the same training protocol, except we train
the RBM with 100 steps of Contrastive Divergence as opposed to 10. We also train the models for
1000 iterations as opposed to a single pass through the dataset. We find that this enables the RBMs to
generate more realistic samples. We include the generated images in Figure 6 to demonstrate that
these models have learned the dataset reasonably well.

(a) MNIST
(b) eMNIST
(c) kMNIST
(d) Omniglot
(e) Caltech

Figure 6: Images sampled from RBMs trained by Contrastive-Divergence with Block Gibbs. We use
Block Gibbs as the sampling algorithm to produce these images as well.

Sampler Configuration
For GWG, we use the same settings as Grathwohl et al. [2021], for
DMALA, we set step size to .2, and for AB we use the default hyper-parameters for the first order
sampler.

For ACS, we use ρ∗= .5, βmax = .95, ζ = .5, cycle length s = 20 for all the datasets. We also fix
the total overhead of the tuning algorithm to 10% of the total sampling steps.

Escape from Local Modes
In addition to using the same initialization as Zhang et al. [2022a],
Grathwohl et al. [2021], we extend the experiment to measure the ability of a sampler to escape from
local modes. We initialize the sampler within the most likely mode, as measured by unnormalized
energy of the RBM. Samplers that are less prone to getting trapped in local modes will be able
to converge quicker to the ground truth, as measured by log MMD. We include the performance
of the various samplers across 11 random seeds in 7. ACS demonstrates superior robustness to
mode-specific initialization due to its capability to escape from local modes.

23


---Page Break---
0
5000
10000
Time (s)

6

5

4

3

2

Log MMD

mnist

GWG
DMALA
AB
ACS (Ours)

0
5000
10000
6

5

4

3

2

kmnist

0
5000
10000

4.0

3.5

3.0

2.5

2.0

1.5

1.0
emnist

0
5000
10000

6

5

4

3

2

omniglot

0
5000
10000
5.0

4.5

4.0

3.5

3.0

2.5

2.0

1.5

1.0

caltech

Figure 7: Log MMDs v.s sampling iteration across various datasets. ACS demonstrates more robust
sampling behavior across the datasets than other methods, as evidenced by superior convergence on
all datasets except KMNIST. We do note that ACS performance is still competitive on KMNIST with
the added benefit of a smaller standard error.

Generated Images
We found that a visual inspection of the generated images demonstrates the
ability of ACS to escape local modes. We include the generated images in Figure 8.

(a) GWG
(b) AB
(c) DMALA
(d) ACS

Figure 8: Images sampled from RBM trained on MNIST when the sampler is initialized to most
likely mode. ACS is able to generate a diverse range of digits, demonstrating its ability to escape
from modes. It should also noted that while AB is able to generate a diverse range of digits as well,
the images are slightly less clear than those generated by ACS.

We can make two primary inferences from the generated images: the first being that ACS is able to
escape from local modes and explore the distribution as a whole, as demonstrated by the wide range
of generated images; and that ACS does not compromise on the ability to characterize each mode as
evidenced by the quality of generated samples.

Sampling Speed
While the run time can vary depending on the specific implementation of a given
sampling algorithm, we illustrate the efficiency of ACS in Figure 9. ACS is able to outperform
DMALA in terms of convergence with respect to time, even including the overhead of the tuning
algorithm.

D.3
EBM Sampling

0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
Time (s)

5.5

5.0

4.5

4.0

3.5

3.0

Log MMD

kmnist

DMALA
ACS (Ours)

Figure 9: Log MMD for the RBM sampling task
across time for both DMALA and ACS for the
kMNIST dataset. We observe that while ACS is
offset slightly in the beginning due to the initial
tuning algorithm, it is quickly able to demonstrate
superior convergence.

Base EBM Training
In order to train the
EBMs, we use Gibbs-with-Gradient to sample
the EBM distribution during PCD, following
the same training protocol as Grathwohl et al.
[2021]. We train these models for 50,000 itera-
tions total with 40 sampling steps per iteration
and use the parameters corresponding to the best
log likelihood scores on the validation dataset.

Experimental Design
For each of the trained
models, we evaluate the samplers based on how
quickly the average energy of the generated sam-
ples rises. This gives an estimate of the speed
at which a sampler is able to reach a stationary
distribution.

24


---Page Break---
Sampler Configuration
For GWG, we use the same settings as Grathwohl et al. [2021], for
DMALA, we set step size to .2, and for AB we use the default hyper-parameters for the diagonal
variant of the AB sampler. We choose this variant as this is what they evaluate for their experiments
when measuring mixing speed of samplers on EBMs.

For ACS, we use ρ∗= .5, βmax = .8, ζ = .5, cycle length s = 20 for all the datasets. As in RBM
Sampling, we fix the total overhead of the tuning algorithm to 10% of the total sampling steps.

Sampler Performance
It is worth commenting on the similarity in performance between ACS and
DMALA when sampling from Caltech. We find that when sampling from the EBM trained on the
Caltech dataset, ACS finds a αmax similar to αmin, thus making the ACS sampler similar to DMALA
for this specific case. We hypothesize that small step sizes are most effective for this dataset. The
results in Figure 3 demonstrate that ACS can handle such cases automatically: while the step size
for DMALA must be hand-tuned, the ACS method can automatically adapt to a suitable step size
schedule.

Generated Images
We include the images generated by ACS when sampling from deep EBMs in
Figure 10.

(a) Static
(b) Dynamic
(c) Omniglot
(d) Caltech

Figure 10: Generated Images from applying ACS sampling to deep EBMs trained with GWG. These
samples capture multiple different modes while retaining good sample quality, demonstrating the
benefit of our ACS method.

D.4
RBM Learning

Experiment Design
We use the same RBM structure as the sampling task, with 500 hidden units
and 784 visible units. However, we apply the samplers of interest to the PCD algorithm introduced
by Tieleman [2008]. The model parameters are tuned via the Adam optimizer with a learning rate of
.001.

In order to evaluate the learned RBMs, we run AIS with Block-Gibbs as the sampler to calculate the
log likelihood values for the models Neal [2001]. We run AIS for 100,000 steps, which is adequate
given the efficiency of Block Gibbs for this specific model.

Sampler Configuration
For DMALA, we use a step size of .2. For the ACS algorithm, we set
βmax = .9, ρ∗= .5 for all the data-sets. We do modify the number of cycles for each data-set as
different distributions require different amounts of exploration and exploitation. We use cycle length
of 8 for MNIST, eMNIST, and kMNIST; we use 20 for Omniglot and Caltech silhouettes. This
difference reflects the specific needs for each dataset in terms of exploration and exploitation – more
complex datasets tend to need longer cycles in order to better exploit each region, while simpler
datasets tend to need shorter cycles in order to capture all the modes of the learned distribution. In
Figure 11, we show the samples generated from AIS for 100,000 steps as opposed to the persistent
buffer as this forms a longer MCMC chain, thus giving a better visual of what the learned distribution
represents.

In order to ensure that the overhead for the tuning algorithm does not add to the overall computational
cost, we spread out the computations of the EstimateAlphaMin algorithm throughout the training
process. We keep a running list of αmin and set αfloor to be one standard deviation below the mean of
this list. By doing this, we start closer to what the ideal αmin. For EstimateAlphaMax, we simply call

25


---Page Break---
the tuning function every 50 cycles containing 50 training iterations, with αceil = 5. As the initial
step does not use the Metropolis-Hastings correction and has half the sampling steps, the budget for
each call of EstimateAlphaMax can be seen as coming in part by the computation saved.

Results
We include the AIS results for RBMs trained with different sampling methods in Table
4. We see that ACS achieves superior log likelihood results when compared to other sampling
methods across all datasets. Furthermore, the AIS results are consistently close to those achieved by
Block-Gibbs, which can be considered close to ideal for RBMs since it leverages the known structure
of the model.

Generated Images
We include the generated images from the RBMs trained using different
samplers in Figure 11.

Figure 11: Generated images from RBMs trained with different samplers. First row corresponds
to GWG, second row corresponds to DMALA, and final row corresponds to ACS. First column
represents models trained on MNIST, second on eMNIST, third on kMNIST, fourth on Omniglot,
and fifth on Caltech Silhouettes. Images are generated via AIS for 100,000 steps.

Table 4: Log likelihood scores for RBM learning on test data
as estimated by AIS. ACS outperforms all gradient-based
baselines across all datasets.

GB
GWG
DMALA
ACS

MNIST
-191.98
-387.34
-278.35
-249.55
eMNIST
-317.78
-590.97
-324.34
-304.96
kMNIST
-357.69
-681.28
-436.3538
-407.39
Omniglot
-161.73
-276.81
-222.61
-220.71
Caltech
-511.65
-827.45
-427.29
-396.04

In general, the images generated from
the ACS-trained RBM capture more
modes than other methods, except for
the Caltech Silhouettes dataset.
In
particular, all the methods struggle to
generate reasonable images for this
dataset. We hypothesize that this is
due to the increased complexity of
the distribution relative to the other
datasets – Caltech Silhouettes is com-
posed of the silhouettes from real ob-
jects, whereas the other datasets are
hand-written symbols. This hypothesis is supported by the generated images in Figure 6, where the
images generated when using Block-Gibbs on Caltech Silhouettes also seem less reasonable than
the samples obtained from different datasets. Since Block-Gibbs is the best sampler for this specific
model as it leverages the known structure of the RBM, this appears to be unavoidable as a result of
limited model capacity. This motivates our experiments with deep convolutional EBMs, where we
can understand how our method does when using a model architecture with sufficient capacity.

26


---Page Break---
D.5
EBM Learning

Experiment Design
We use the same EBM model architecture as Zhang et al. [2022a], Grathwohl
et al. [2021] and follow the same experimental design, with the only change being to the number of
sampling steps alotted for each sampler.

In order to determine the number of sampling steps that we could use for ACS-PCD, we tested
different sampling steps. For Static/Dynamic MNIST and Omniglot, we found that we only needed
to use 10 sampling steps to achieve good models. However, we observed divergence when training
models on Caltech. In order to determine what number of sampling steps to use for ACS, we do a
grid search over the number of sampling steps and ρ∗with all other values remaining the same. We
test sampling steps of 10, 20, and 30; and we use ρ∗.5, .6, .7, .8. We decide which hyper-parameters
to use based on when training diverged the latest; and we use the best model parameters as indicated
by validation log likelihood. We use 10 sampling steps for Static, Dynamic MNIST, and Omniglot,
while we found 30 was the minimum we could use for Caltech Silhouettes and obtain reasonable
results. We apply this number of sampling steps for both DMALA and ACS to demonstrate how the
methods compare when facing a similar budget constraint.

In order to evaluate these learned models, we use the same evaluation protocol as Zhang et al.
[2022a], Grathwohl et al. [2021]. We run AIS for 300,000 iterations using Gibbs-With-Gradient as
the evaluation sampler. By following the same experimental design as previous works, we can draw
meaningful comparisons from previous results in Grathwohl et al. [2021].

Sampler Configuration
For DMALA, we use a step size of .15 as used in Zhang et al. [2022b]. For
ACS, we use 200 sampling steps for EstimateAlphaMax and EstimateAlphaMin. For Static MNIST,
Dynamic MNIST, and Omniglot, we set the algorithm to tune αmax and αmin every 25 cycles, where
each cycle has 50 training iterations. The additional overhead of this is 16,000 extra sampling steps,
which is a 3.2% of the total budget of 500,000 sampling steps. For Caltech Silhouettes, we have to
adapt every 10 cycles with the same number of training iterations. This results in 40,000 additional
sampling steps due to the tuning algorithm. For this specific dataset, because we use 30 sampling
steps, the additional cost is 2.6% of the total sampling steps 1,500,000.

In terms of the final parameters for cycle length and sampling steps, we find that we can use the same
ρ∗across all datasets, with the exception of Caltech Silhouettes. For Static/Dynamic MNIST and
Omniglot, we were able to use ρ∗= .5 and For this dataset, we found good results by setting ρ∗= .7.
We hypothesize that the need for a higher acceptance rate is due to the fundamental difference between
Caltech Silhouettes and the other datasets, as previously mentioned. Because Caltech Silhouettes
contain samples are derived from real objects, they are more complex than the hand-written figures.

Experimental Results
In addition to the empirical results in Table 1, we provide some qualitative
data in the form of the generated images from the PCD buffer when using ACS. We choose to
include the buffer images for this experiment as the chain from the persistent buffer is much longer
than the chain from AIS due to the increased training duration: the chain from AIS is obtained
using 300,000 sampling steps whereas the persistent buffer is obtained from 500,000 sampling steps
on Static/Dynamic MNIST and Omniglot, 1,500,000 sampling steps for Caltech Silhouettes. By
visualizing the generated images from the longer chain, we get a better understanding of the quality
of the trained distribution. We put the images in Figure 12.

We also observe that this behavior is not unique to ACS and does occur when Gibbs-With-Gradient
and DMALA are used with 40 sampling steps as indicated. Instability is common when training
deep EBMs, and this is most likely why the original experimental design included check-pointing
throughout the training process as well as comparisons based on the validation set. We also note that
despite this behavior, the trained models are able to generate fairly realistic images. We present the
images from the PCD buffer for ACS below in figure.

When the images in Figure 12 are taken in context of the improvements in log likelihoods as
presented in Table 1, the results indicate the benefits of using ACS when learning multi-modal
discrete distributions.

27


---Page Break---
(e) Static
(f) Dynamic
(g) Omniglot
(h) Caltech

Figure 12: The example images from the representative datasets, along with the samples generated
from the persistent buffer when using ACS as the sampler for PCD. The images on the top row are
examples from the dataset, while the bottom row are from the trained EBM. The images generated
from ACS are remarkably similar to those from the dataset, demonstrating that the model is capable
of generating high-quality samples.

D.6
Text Infilling

Experimental Design
For both datasets, we sample 100 sentences randomly and mask 50% of
the tokens. We use a pretrained RoBERTa model available through the Hugging Face API Liu et al.
[2019]. We run 25 separate chains for each example and take the final state of each chain to be a
sample. We then take the top-5 most likely samples and use these for empirical comparisons.

We define the energy function the same as in Zhang et al. [2022a]. Let us define a sentence of length
d θ = {θ1, θ2, . . . θd}, where θi is a one hot vector over vocabulary V . Let M ⊂{1, 2, . . . d} the set
of indices we wish to sample. We define the function f(θi|θ¬i) to be the log probability distribution
over V for the i position conditioned on all other positions. Given this, we define the energy function
for the sentence θ to be as follows:

U(θ) =
X

m∈M
f(θm|θ¬m)
(20)

Sampler Configuration
For DMALA, we tune the step-size to achieve an acceptance rate of 50%,
which ends up being α = .5. For ACS, we use a cycle length of 20. We use the same hyper-parameters
for the tuning algorithms as in previous tasks, demonstrating that our algorithm can be applied across
domains and tasks with little modification. We include example generations for both ACS and
DMALA in Figure 13.

Perplexity v.s Sampling Accuracy
In Table 2, we observe that the ACS generations have higher
perplexity than the DMALA generations. While perplexity is a popular means of evaluating language
generations, it is important to recognize that perplexity is based on the likelihood of the generation
under the language model. This metric is biased towards frequent patterns and does not account
for diverse modes. Therefore, it does not completely align with the goal of MCMC, which is to
accurately characterize the target distribution.

Minimizing the average perplexity of the sample corresponds to maximizing the average likelihood
of the generations, which can be at odds with the goal of accurately capturing the target distribution.
We illustrate this in Appendix D.1, where we compare the performance of DMALA and ACS when
sampling from a uneven multi-modal distribution.

28


---Page Break---
• The comedy that follows feels hackneyed or just plain crude, calculated to provoke shocked stares, without opening up to a deeper

truth.
• This comedy could either be hacky, or just plain crude, calculated to provoke shocked curiosity, without opening up to a deeper

insight.
• A comedy that feels slightly hacky, or just plain crude, calculated to achieve shocked results, without coming up with a deeper

message.
• The comedy was either unnecessarily hacky, or just plain crude, calculated to create shocked humor, without leading up to a deeper

plot.
• Most comedy is flat or hack-ish or just plain crude, calculated to evoke shocked laughs, without opening up to a deeper audience.

(a) ACS

• This comedy is either plain hacky, or just plain crude, calculated to provoke shocked discussion, without linking up to a deeper

message.
• And comedy that can be hacky, or just plain crude, calculated to provoke shocked discussion, without opening up to a deeper topic.
• Modern comedy can be deliberately hacklish, or just plain crude, calculated to provoke shocked disbelief, without opening up to a

deeper meaning.
• Simple comedy has all things hacky, or just plain crude, calculated to be shocked away, without opening up to a deeper meaning.
• A comedy usually ranges from hacky, or just plain crude, calculated to provoke shocked surprise, without opening up to a deeper

reality.

(b) DMALA

Figure 13: Text Generations using ACS and DMALA. As demonstrated empirically in 2, the ACS
examples demonstrate higher diversity than the DMALA generations.

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: in the introduction, we make four primary claims. We introduce our discrete
gradient based sampler in Section 4.2. We introduce the tuning algorithm in Section 4.4.
We introduce the theoretical results, along with the necessary assumptions in Section 5. We
include all the experimental results on RBMs, EBMs and LLMs in Section 6
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
Justification: We include the limitations in the conclusion section.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
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

30


---Page Break---
Answer: [Yes]

Justification: We include the assumptions in Section 5 as well as in Appendix C.

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

Justification: We include a lengthy appendix that details all the experimental configuration,
along with the hyper-parameters used.

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

31


---Page Break---
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We will include a link to an anonomyzed repository containing all the code to
run the necessary experiments. All experimental results can be generated by running a bash
script.
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
Justification: We include the experimental details in the appendix.
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
Justification: For the experimental results on RBMs, which are relatively noisy, we include
the performance within one standard error across different random seeds. We also include
standard error measures for the text infilling experiment. For RBM training, EBM training,
and EBM sampling, we do not include standard error bars as these experiments are not
significantly affected by random seeds, at least within our observations.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

32


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

Justification: We include the specific GPU at the beginning of Appendix D.

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

Justification: We have read the Ethics Guidelines, and our submission aligns with all the
points listed.

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

Justification: This is a MCMC sampling technique which does not have a direct societal
impact. The research presented is either theoretical, or based on common benchmarks such
as MNIST. The most relevant impact would be due to text infilling, which can be used for
LLMs text generation.

33


---Page Break---
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

Justification: This foundational research that does not directly have a societal impact, as it is
primarily an MCMC algorithm for discrete spaces.
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
Answer: [No]

Justification: We were unable to find the license for the datasets we used. However, they are
fairly popular and well known datasets, and we cite the relevant paper where necessary.
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.

34


---Page Break---
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

Justification: We do not release new assets.

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

Justification: This research does not involve crowdsourcing or human subjects.

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

Justification: There are no study participants.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

35


---Page Break---
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

36


---Page Break---
