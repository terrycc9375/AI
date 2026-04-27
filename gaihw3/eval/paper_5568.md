Robust Neural Contextual Bandit against
Adversarial Corruptions

Yunzhe Qi, Yikun Ban, Arindam Banerjee, Jingrui He
University of Illinois at Urbana-Champaign
Champaign, IL 61820
{yunzheq2,yikunb2,arindamb,jingrui}@illinois.edu

Abstract

Contextual bandit algorithms aim to identify the optimal arm with the highest
reward among a set of candidates, based on the accessible contextual information.
Among these algorithms, neural contextual bandit methods have shown generally
superior performances against linear and kernel ones, due to the representation
power of neural networks. However, similar to other neural network applications,
neural bandit algorithms can be vulnerable to adversarial attacks or corruptions on
the received labels (i.e., arm rewards), which can lead to unexpected performance
degradation without proper treatments. As a result, it is necessary to improve the ro-
bustness of neural bandit models against potential reward corruptions. In this work,
we propose a novel neural contextual bandit algorithm named R-NeuralUCB, which
utilizes a novel context-aware Gradient Descent (GD) training strategy to improve
the robustness against adversarial reward corruptions. Under over-parameterized
neural network settings, we provide regret analysis for R-NeuralUCB to quantify
reward corruption impacts, without the commonly adopted arm separateness as-
sumption in existing neural bandit works. We also conduct experiments against
baselines on real data sets under different scenarios, in order to demonstrate the
effectiveness of our proposed R-NeuralUCB.

1
Introduction

Contextual bandits refer to one specific type of multi-armed bandit (MAB) problems, where the
learner can access the arm context information during the decision-making process. Contextual bandit
algorithms have been commonly applied in various real-world applications, including online content
recommendation [59, 79, 8], and medical experiments [30, 73, 7]. While these algorithms have been
proved effective for numerous online learning tasks, they can be susceptible to the malicious feedback
from the environment, such as malicious user feedback in recommender systems [64], and corrupted
labels under active learning settings [61]. This can potentially impair the model performance and
interfere with the internal decision-making logic. One renowned research direction formulates this
problem as contextual bandits with adversarial corruptions [57, 42, 16], where received arm rewards
can be potentially “corrupted” by the unknown adversary. In this case, bandit algorithms need to be
robust against such adversarial corruptions, otherwise they can lead to sub-optimal results. Existing
works on contextual bandits with corruptions are mainly based on linear [17, 29, 57] and kernelized
bandits [15, 16], where the unknown reward mapping function is assumed to be linear, or lies in
a specified Reproducing Kernel Hilbert Space (RKHS). However, one key challenge is that these
assumptions can evidently fail under real-world application scenarios [86], when we have little prior
knowledge regarding this mapping function, or it becomes increasingly complex.

In the face of this challenge, neural bandit algorithms [86, 84, 11, 12] have been proposed to relax the
assumptions on reward functions. By leveraging the representation power of neural networks, neural
contextual bandit algorithms are able to deal with complex reward functions irrespective of whether

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
they are linear or non-linear, along with suitable exploration strategies for tackling the exploitation-
exploration dilemma [5, 59]. While neural bandit algorithms have been proved effective [9, 68, 60],
they can be sensitive to adversarial corruptions as well. From perspectives of trustworthiness, it is
well known that neural models can be susceptible to “label attacks” [71, 62, 65], which is akin to
reward corruptions in bandit settings. Failure to comply with robustness requirements can impair the
feasibility under real-world application scenarios like recommender systems [82, 27, 87, 78], and
therefore it is necessary for neural bandit methods to be robust against potential adversarial corruptions.
Furthermore, existing neural bandit works generally require arm separateness assumptions (e.g.,
assuming a positive-definite Neural Tangent Kernel [NTK] Gram matrix [86, 84, 26], or positive
arm Euclidean distances [11, 67]), which will require no duplicate arm contexts are observed (or
chosen) by the learner. This can lead to additional vulnerabilities when arm contexts are intentionally
chosen by the adversary (e.g., assigning duplicate arms in the candidate pool across different rounds),
making the arm separateness assumption fail in such adversarial environments.

Motivated by aforementioned challenges, in this paper, we propose a novel neural contextual bandit
algorithm named Robust Neural-UCB (R-NeuralUCB), which can model the discrepancy among
candidate arms and adopt arm-specific context-aware Gradient Descent (GD) to enhance model
robustness against reward corruptions. Instead of applying ordinary GD to update the network
parameters, R-NeuralUCB utilizes a fine-grained GD strategy by modeling the importance level of
training samples, to reduce the impact of potential adversarial reward corruptions. Meanwhile, to
improve the model performance from both the theoretical and empirical perspectives, R-NeuralUCB
simultaneously perceives the uncertainty levels of candidate arms, and adaptively customizes network
parameters for each of these candidates. To deal with the exploitation-exploration dilemma, R-
NeuralUCB is equipped with an informative exploration mechanism based on Upper Confidence
Bound (UCB) to achieve principled exploration. In addition, we present regret analysis without
the commonly adopted arm separateness assumption, which reinforces R-NeuralUCB’s theoretical
robustness under adversarial scenarios. Our contributions can be summarized as follows:

• Problem Settings and Proposed algorithm: We study a novel neural bandit problem, where the
received arm reward can be potentially corrupted by the unknown adversary. To deal with this
problem, we propose a novel neural bandit algorithm called R-NeuralUCB, which leverages a
refined context-aware Gradient Descent training strategy to improve the model robustness against
potential arm reward corruptions. While we consider all the observed arms are governed by the
same unknown reward mapping function as in (1) similar to existing neural bandit works, our
R-NeuralUCB interestingly maintains separate model parameters specific to the different candidate
arms, for a fine-grained way of improving the robustness. This casts lights on our contributions of
novel algorithmic designs, compared to related existing methods without an arm-specific modeling
(e.g., [42] and our base algorithm NeuralUCB-WGD in Appendix E).
• Theoretical Analysis: With over-parameterized neural networks, we present the regret analysis for
R-NeuralUCB. Given finite horizon T, effective dimension of NTK Gram matrix ed, and corruption
level C, R-NeuralUCB enjoys a data-dependent regret bound of e
O(ed
√

T + C ed). In addition, to
ensure R-NeuralUCB is capable of handling contexts specified by the adversary (e.g., duplicate
arms across different rounds), our analysis removes the arm separateness assumption dependency,
a widely adopted assumption for neural bandit literature, which can be of independent interest.
• Experiments: We conduct experiments on publicly available real-world data sets with various
specifications. Under different types of reward corruptions, our R-NeuralUCB can achieve better
performance, and is less vulnerable to reward corruptions than baselines.

2
Related Works

Contextual Bandits with Adversarial Corruptions. To begin with, there have been numerous
studies [76, 17, 29, 57, 63, 31, 22, 54] working on tackling adversarial reward corruptions under
linear contextual bandit settings [59, 24]. A related topic is bandits with mis-specifications [36, 55,
33, 53, 83, 75], where the deviation of reward estimation comes from problem modeling instead of
the adversary. On the other hand, kernelized bandits [15, 40, 16] extend the adversarial corruption
problem to non-linear cases by assuming the reward mapping is a functional in the specified RKHS
[72], while comparable ideas are also applicable for robust Bayesian Optimization [52]. Adversarial
corruptions are also studied for other formulations, such as Lipschitz bandits [49, 89] and MAB
without contexts [18, 80]. However, compared with neural bandit methods, these works generally
require assumptions on the reward function prior, which may not be satisfied in real-world scenarios.

2


---Page Break---
Neural Contextual Bandits. Neural contextual bandits algorithms are proposed to leverage the
representation power of neural networks, and relax the assumptions on the reward mapping functions
that can be linear or non-linear. Neural-UCB [86] applies a fully-connected (FC) neural network for
reward estimation and utilizes corresponding network gradients for principled exploration. Com-
parable ideas have been leveraged by other neural bandit works [84, 50, 25, 10, 37, 46, 9, 60, 11],
and adopted under various application scenarios such as active learning [74, 13, 6], and bandit-based
graph learning [66, 67, 51] with graph neural networks [77, 34, 35]. Alternatively, [81] utilizes
the neural network to embed original arm contexts for regression. [26] utilizes inverse reward gap
for exploration, and [48] achieves exploration with the reward perturbation. However, as these
methods are not designed to defend against reward corruptions and widely require arm separateness
assumptions, they can fail to meet the robustness requirements in an adversarial environment.
3
Problem Definition

Let T be the finite horizon. In round t ∈[T], the learner receives K candidate arms Xt, |Xt| = K,
and each arm xi,t ∈Xt with arm index i ∈[K] is described by a d-dimensional vector xi,t ∈Rd.
The learner will then choose one arm xt ∈Xt and receive its reward rt. The index of xt is denoted
by it ∈[K], s.t. xt = xit,t. Here, similar to existing works (e.g., [42, 16, 15, 86, 84]), we define
corruption-free arm reward eri,t for each candidate arm xi,t ∈Xt, as well as corrupted arm reward rt
for chosen arm xt ∈Xt, as

eri,t = h(xi,t) + ϵi,t,
rt = ert + ct = h(xt) + ϵt + ct,
(1)

where h : Rd 7→R is an unknown reward mapping function that can be either linear or non-
linear. ϵi,t ∈R stands for zero-mean ν-sub-Gaussian random noise which is standard for stochastic
contextual bandit works (e.g., [24, 72, 86]), and ct ∈R is the unknown adversarial corruption
imposed by the adversary. While kernelized bandit works (e.g., [16, 72]) assume h(·) belongs to the
RKHS induced by specified kernels, we alternatively consider h(·) as an arbitrary unknown function,
and utilize the neural model to learn this mapping with flexibility.

Taking expectation w.r.t. zero-mean noise ϵ, for the chosen arm xt ∈Xt, we denote its expected
perturbed reward E[rt] = h(xt) + ct; meanwhile, for each candidate arm xi,t ∈Xt, its expected
corruption-free reward E[eri,t] = h(xi,t). Here, we consider E[r] and E[er] both fall into value range
[0, 1], analogous to existing works (e.g., [86, 84, 11, 50]). This is intuitive as numerous real-world
applications work with bounded rewards (e.g., online recommendation tasks with normalized rating
[67] or binary feedback [24]); and the adversary also needs its attack to be stealthy, by ensuring
perturbed rewards fall into the normal value range. With previously chosen arms {xτ}τ∈[t] up to
round t, we denote received context-reward tuples with perturbed rewards as Pt := {xτ, rτ}τ∈[t] =
{xiτ ,τ, riτ ,τ}τ∈[t], and the corresponding context-reward tuples with corruption-free rewards as
ePt := {xτ, erτ}τ∈[t] = {xiτ ,τ, eriτ ,τ}τ∈[t], where each corruption-free but imaginary unobserved
reward is erτ = h(xτ) + ϵτ, τ ∈[t] based on (1).

Learning Objective. Our objective is to minimize cumulative pseudo-regret for T rounds:

R(T) =
XT

t=1 E[er∗
t −ert],
(2)

where E[ert] = h(xt) is the expected corruption-free reward of the chosen arm xt ∈Xt, and
E[er∗
t ] = maxxi,t∈Xt[h(xi,t)] stands for that of the optimal arm x∗
t ∈Xt.

Corruption Level. If the adversary determines the reward corruption ci,t for each candidate arm
xi,t ∈Xt beforehand, without observing the learner’s choice xt, some works (e.g., [38]) formulate
the corruption level measurement as C′ = P

t∈[T ][maxi∈[K]|ci,t|]. In this work, similar to [42, 16],
we alternatively consider reward corruptions are determined w.r.t. particular chosen arms {xt}t∈[T ],
and formulate the corruption level as C = P

t∈[T ]|ct|. This leads to C ≤C′.
4
Proposed Algorithm: Robust Neural-UCB (R-NeuralUCB)

Recall that in (1), arm rewards under neural bandit settings are governed by the unknown reward
mapping function h(·), where h(·) can be an arbitrary function. For our proposed R-NeuralUCB, we
adopt a neural network f(·) to approximate h(·) for reward estimation.

Network Structure. We use f(·; θ) to denote an FC network with depth L ≥2 and width m ∈N+:

f(x; θ) := √mθLσ(θL−1σ(θL−2 . . . σ(θ1x)))
(3)

3


---Page Break---
Table 1: Comparison of T-round regret bounds with adversarial corruption level C.

Algorithm
Reward Function
Corruption C
Regret Bound*

Robust OFUL [76]
Linear
Known
e
O(d
√

T +
q

T PT
t=1 c2
t)
CW-OFUL [42]
Linear
Known
e
O(d
√

T + Cd)

COBE + OFUL [76]
Linear
Unknown
e
O(d
√

T +
q

T PT
t=1 c2
t)
CW-OFUL ( ¯C =
√

T) [42]
Linear
Unknown
e
O(d
√

T), if C ≤
√

T. Otherwise O(T)

Fast-slow GP-UCB [15]
Kernelized
Known
e
O(ed
√

T + C ed
√

T)
RGB-PE [16]
Kernelized
Known
e
O(ed
√

T + C ed3/2)
Fast-slow GP-UCB [15]
Kernelized
Unknown
e
O(ed
√

T + C ed
√

T)

NeuralUCB-WGD (Thm. E.1)
Arbitrary
Known
e
O(ed
√

T + C ed3/2)
R-NeuralUCB (Thm. 5.6)
Arbitrary
Unknown
e
O(ed
√

T + Cβ−1 ed)

* d: context dimension; e
d: NTK matrix effective dimension or kernel information gain; β: data-dependent gradient deviation term.

where σ(·) is element-wise ReLU activation, and we have trainable weight matrices θ1 ∈Rm×d,
θl ∈Rm×m, 2 ≤l ≤L −1, θL ∈R1×m. For the ease of notation, we denote vectorized parameters

θ := [vec(θ1)⊺, vec(θ2)⊺, . . . , θL]⊺∈Rp,

with the dimensionality of p, and randomly initialized parameters are denoted by θ0. Then, we let
g(x; θ) = vec(∇θf(x; θ)) ∈Rp be vectorized network gradients w.r.t. input x and parameters θ.

We motivate our proposed R-NeuralUCB by first mentioning a base algorithm named Neural-
UCB with Weighted GD (NeuralUCB-WGD), which is elaborated in Appendix E. To begin with,
NeuralUCB-WGD measures the uncertainty level of training samples (i.e., previously received arm-
reward pairs) through their UCB values, as the UCB essentially measures arm uncertainty levels in
terms of reward estimation [24, 72, 86]. Then, different from conventional neural bandit methods that
treat all training samples equally [86, 84], inspired by [42], NeuralUCB-WGD utilizes a weighted
GD process to train neural model f(·) for estimating arm rewards, where training samples with high
uncertainty levels will be downplayed. The main idea is that although we do not know which training
samples are corrupted, we instead aim to reduce potentially severe impacts caused by adversarial
corruptions, by paying relatively more attention on the training samples (arm-reward pairs) with low
uncertainty levels, for a stable GD training process. We also present corresponding regret analysis for
NeuralUCB-WGD in Appendix E.2, as well as experiments in Section 6.

However, notice that the neural model will also perceive varying uncertainty levels for different
candidate arms in terms of reward estimation. In this case, simply applying the same exploitation-
exploration strategy across all candidate arms can overlook this discrepancy, leading to insufficient
granularity w.r.t. reward estimation. For instance, regarding candidate arms with low uncertainty
levels, it can be more beneficial to adequately leverage existing training samples for estimating their
rewards, instead of sharing an identical exploitation-exploration strategy with other high-uncertainty
candidate arms. Meanwhile, analogous to existing works (e.g., [42, 16, 15]), NeuralUCB-WGD
supposes a known corruption level C for regret analysis (Theorem E.1), which can be difficult to
satisfy if we have limited knowledge regarding the unknown adversary. With the above motivations,
we propose R-NeuralUCB as a refined solution to further enhance neural model robustness against
potential reward corruptions. For readers’ reference, we also compare our proposed R-NeuralUCB
and NeuralUCB-WGD with some regret results from existing works in Table 1.

4.1
R-NeuralUCB: Robust Neural-UCB

Our R-NeuralUCB formulates a novel context-aware GD process, by taking the uncertainty infor-
mation of both candidate arms and training samples into account, for neural network training and
decision making. Here, R-NeuralUCB customizes individual sets of network parameters θi,t−1 for
each candidate arm xi,t ∈Xt, i ∈[K], before the actual arm recommendation. Afterwards, these
arm-specific networks are applied for arm reward estimation, along with an informative arm-specific
UCB-based exploration mechanism. The pseudo-code is presented in Algorithm 1.

Arm Weight Formulation. Inspired by NTK-based exploration mechanisms [86, 51, 66, 11, 84],
we measure arm uncertainty levels with the weighted gradient norm of arms. With a regular-
ization parameter λ > 0, we first define a weight-free gradient covariance matrix ¯Σt−1 =
λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m. Here, θτ−1 is the shorthand of θiτ ,τ−1, representing
the network parameters of the previously chosen arm xτ = xiτ ,τ, τ ∈[t −1]. Then, for each

4


---Page Break---
Algorithm 1 Robust Neural-UCB (R-NeuralUCB)

1: Input: Time horizon T. GD iterations J. Learning rate η. Exploration coefficient ν. Scaling
parameter α. Norm parameter S. Regularization parameter λ.
2: Initialization: Parameters θ0. Weight-free covariance matrix ¯Σ0 = λI. Records P0 = ∅.
3: for each round t ∈[T] do
4:
Observe a collection of K candidate arms Xt = {xi,t}i∈[K].
5:
for each candidate arm xi,t ∈Xt do
6:
if t equals to 1 then
7:
θi,t−1 ←θ0.
8:
else
9:
With arm weights {w(τ)
i,t }τ∈[t−1] in (4), train parameters θi,t−1 with GD and the arm-
specific loss function in (5) based on received records Pt−1.
10:
end if
11:
For candidate arm xi,t, calculate its benefit score U(xi,t) based on (6).
12:
end for
13:
Choose arm xt = arg maxxi,t∈Xt

U(xi,t)

with the highest benefit score.
14:
Receive arm reward rt, and update the records, such that Pt ←Pt−1 ∪{(xt, rt)}.
15:
Update the shorthand θt−1 ←θit,t−1, and matrix ¯Σt ←¯Σt−1+g(xt; θt−1)g(xt; θt−1)⊺/m.
16: end for

candidate arm xi,t ∈Xt, we formulate its weight w.r.t. previously chosen arm xτ, τ ∈[t −1] as

w(τ)
i,t = min




1,
α · min
x∈Xt∥g(x; θt−1)/√m∥2
¯Σ−1
t−1
gτ · ∥g(xi,t; θt−1)/√m∥( ¯Σ(κ)
t−1)−1




,
(4)

with κ2-scaled covariance matrix being ¯Σ(κ)
t−1 := λI + κ2 · P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m,

for a constant κ ∈(0, 1). Alternatively, we also denote w(τ)
i,t = min

1, α · fracτ(xi,t; Xt, ¯Σt−1)
	
,
with fracτ(·) being a shorthand that integrally represents the fraction term in (4). A tunable parameter
α > 0 and the squared round-wise minimum weighted norm minx∈Xt ∥g(x; θt−1)/√m∥2
¯Σ−1
t−1 in the

numerator are applied for scaling purposes. We also include complementary discussions for arm
weight scaling in Appendix B.5. Meanwhile, we have gτ = ∥g(xτ; θτ−1)/√m∥( ¯Σ(κ)
τ−1)−1, τ ∈[t−1]

quantifying uncertainty levels of previously chosen arms (training samples) motivated by UCB-based
exploration strategies (e.g., [86, 42]). Since previous {gτ}τ∈[t−1] values can be reused, we only need
to compute and store gt for current round t. As a result, if candidate arm xi,t is of high uncertainty
(i.e., large ∥g(xi,t; θt−1)/√m∥¯Σ−1
t−1 value), its arm weights {w(τ)
i,t }τ∈[t−1] will become small.

Model Training with Context-aware GD. According to line 9 in Algorithm 1, we perform model
training before the actual arm recommendation in each round t ∈{2, . . . , T}. For each candidate
xi,t ∈Xt, we train its arm-specific parameters θi,t−1 with J iterations of GD and received records
Pt−1 = {(xτ, rτ)}τ∈[t−1]. Starting from initialization θ(0)
i,t−1 = θ0, we have j-th GD iteration

(j ∈[J]) being θ(j)
i,t−1 = θ(j−1)
i,t−1 −η∇θLi,t(Pt−1; θ(j−1)
i,t−1 ), where η > 0 refers to the learning rate.
We formulate a loss function Li,t(·; ·), i ∈[K] specified to candidate arm xi,t ∈Xt as

Li,t(Pt−1; θ) =
X

(xτ ,rτ )∈Pt−1

w(τ)
i,t
2
·
f(xτ; θ) −rτ
2 + mλ

2
· ∥θ −θ0∥2
2,
(5)

where the L2 loss is scaled by arm weights w(τ)
i,t , τ ∈[t −1] from (4). Intuitively, if arm weights

w(τ)
i,t are large (i.e., low uncertainty level), we proceed to train a neural model that adequately fits the
collected training data (i.e., previously received records) Pt−1, instead of staying around the random
initialization θ0 given the L2 regularization. On the other hand, if arm weights w(τ)
i,t are small, it
means that the uncertainty level in terms of reward estimation is high. In this case, we prefer being
relatively conservative to prevent potentially large impacts caused by adversarial corruptions. As a
result, R-NeuralUCB will focus more on the training samples in Pt−1 with low uncertainty levels,
and stay relatively close to the random initialization θ0 due to the regularization term in (5).

5


---Page Break---
In practice, instead of starting from θ0 in each round t ∈{2, . . . , T}, we can alternatively initiate
the GD process from the existing trained parameters to reduce computational cost, inspired by the
concept of warm-start GD [13]. Here, we can start from θt−2, the parameters of the previously
chosen arm xt−1, and fine-tune arm-specific parameters θi,t−1 for each candidate arm xi,t ∈Xt,
based on its loss function Li,t(·; ·) and a small batch of samples from Pt−1. Further details are
elaborated in Appendix B.6, and this approach is also applied for the experiments in Section 6.

Arm Selection. For candidate arm xi,t ∈Xt and arm weights {w(τ)
i,t }τ∈[t−1], we formulate its arm-

specific gradient covariance matrix Σi,t−1 = λI + P

τ∈[t−1] w(τ)
i,t · g(xτ; θτ−1)g(xτ; θτ−1)⊺/m.
Here, if the variance proxy value ν in (1) is unknown, similar to existing works (e.g., [86, 84]),
we deem ν ≥0 as a tunable parameter to control the exploration intensity. With our UCB-type
exploration motivated by Appendix Lemma C.11, we formulate the benefit score for arm xi,t ∈Xt as

U(xi,t) = f(xi,t; θi,t−1) + γi,t−1 ·
q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1)/m,
(6)

where the confidence coefficient γi,t−1 = ζ ·
 
ν
q

log det(Σi,t−1)

det(λI)
−2 log(δ) +
√

λS

, along with a

constant ζ > 0 from Lemma C.11. Afterwards, we choose xt = arg maxxi,t∈Xt

U(xi,t)

(line 13,
Algorithm 1), based on calculated arm benefit scores in (6). After receiving reward rt, the collected
records will be updated by Pt ←Pt−1 ∪{(xt, rt)} (line 14, Algorithm 1). We also update the
shorthand for model parameters of the chosen arm as θt−1 ←θit,t−1, and the weight-free covariance
matrix ¯Σt ←¯Σt−1 + g(xt; θt−1)g(xt; θt−1)⊺/m for next round t + 1 (line 15, Algorithm 1).

In summary, the primary goal of R-NeuralUCB is to customize individual learning objectives (i.e.,
loss functions) for different candidate arms by leveraging arm uncertainty information before pulling
an arm. For candidate arms with high uncertainty levels, the neural model may lack confidence in
estimating rewards based on current records, due to potential reward corruptions, which can lead to
significant estimation errors. In this situation, by using the regularization term mλ

2 ∥θ −θ0∥2
2, we
prefer to adopt a relatively conservative approach, training a model close to random initialization to
mitigate the potentially large impacts of adversarial corruptions. This approach is inspired by existing
work on enhancing model robustness through regularization techniques (e.g., [69, 23]). On the other
hand, for candidate arms with low uncertainty, we aim to train neural models that fully utilize the
received records for reward estimation. Since the model is confident in its estimation, the received
samples can provide adequate reference. With larger arm weights, the loss function can focus more
on the training samples, instead of staying closely around θ0.

5
Theoretical Analysis

To the best of our knowledge, we provide the first theoretical results under the neural bandit settings
with adversarial reward corruptions, and our proof flow is distinct from those of linear and kernelized
bandit works. In particular, as our ReLU activation in (3) is not Lipschitz smooth [3, 21], it leads to
additional challenges for our theoretical analysis, since a small perturbation on rewards can lead to
drastic changes of network gradients. As a result, even with a small corruption level C, the corrupted
model parameters trained by GD can significantly deviate from the imaginary network parameters
trained with corresponding corruption-free rewards. Therefore, it is non-trivial to quantify corruption
impacts from theoretical perspectives, which simultaneously makes our proof flow differ significantly
from that of the vanilla Neural-UCB [86]. We include additional discussions on analysis distinctions
and our contributions in Appendix B.3. To begin with, we first introduce some preliminaries.

Parameter Initialization. Analogous to existing works [86, 21, 3, 9, 84], for an L-layer network of
width m in (3), we let its intermediate-layer matrices θl =
  Λ 0
0 Λ

, l ∈[L −1], where each element
of matrix Λ is drawn from Gaussian distribution N(0, 4/m). Similarly, let θL = (w⊺, −w⊺), where
each element of vector w is drawn from N(0, 2/m).

Arm Context Normalization. To ensure arm contexts are of unit length (i.e., ∥xi,t∥= 1, ∀i ∈
[K], t ∈[T]) as in existing neural bandit works [86, 84, 11, 67, 50], we can apply the following trans-
formation inspired by existing works [3, 86, 84] without loss of generality: with unprocessed context
exi,t, we formulate the corresponding normalized arm context xi,t = [
exi,t
2·∥exi,t∥2 , 1

2,
exi,t
2·∥exi,t∥2 , 1

2]. It
can be verified that we have three properties: (i) ∥xi,t∥2 = 1; (ii) no two normalized arm contexts
will be in opposite directions; and (iii) f(xi,t; θ0) = 0 with the randomly initialized θ0.

6


---Page Break---
Definitions of NTK Matrices. First, we denote imaginary corruption-free models as f(·; eθt−1), t ∈
[T] which are trained on corruption-free records ePt = {xτ, erτ}τ∈[t], t ∈[T], for the sake of
theoretical analysis, and the learner does not need to own the imaginary model in practice. Let
{xt}t∈[T ] = {xit,t}t∈[T ] be arms chosen by the corrupted model f(·; θ), and {ext}t∈[T ] =
{x˜it,t}t∈[T ] be those chosen by the corruption-free model f(·; eθ) respectively. Then, define a
union set ˘
AT := ({xt}T
t=1 ∪{x∗
t }T
t=1 ∪{ext}T
t=1), based on: (i) the chosen arms {xt}T
t=1, (ii) the op-
timal arms {x∗
t }T
t=1 according to (2), and (iii) arms {ext}T
t=1 chosen by the imaginary corruption-free
models. Here, ˘
AT naturally contains unique arms from these three arm collections, with cardinality
| ˘
AT | ≤3T. Meanwhile, we simply merge these three arm collections to form AT (with cardinality
|AT | = 3T), which allows duplicate arms. Afterwards, we have the following two formulations
of the NTK Gram matrix: (i) The NTK Gram matrix H with possibly duplicate arms based on the
collection AT ; (ii) the NTK matrix for non-duplicate arms ˘H built upon the set ˘
AT .

Definition 5.1 (NTK Gram Matrix with Possibly Duplicate Arms). Let N be the Gaussian distribution.
With layer index l ∈[L] and subscripts i, j ∈{1, . . . , |AT |} for enumerating across arms, comparable
to [47, 86], define the following recursive process

H0
i,j = Ψ0
i,j = ⟨xi, xj⟩,
Nl
i,j =
Ψl
i,i
Ψl
i,j
Ψl
j,i
Ψl
j,j


,

Ψl
i,j = 2Ea,b∼N(0,Nl−1
i,j )[σ(a)σ(b)],
Hl
i,j = 2Hl−1
i,j Ea,b∼N(0,Nl−1
i,j )[σ′(a)σ′(b)] + Ψl
i,j.
(7)

With AT containing possibly duplicate arms, we denote the NTK Gram matrix H = (HL+ΨL)/2 ∈
R3T ×3T , and expected reward vector h = [h(x)]x∈AT ∈R3T . Existing works with the arm
separateness assumption (e.g., [86, 84, 9, 25, 81]) generally assume H ≻0, while we do not.
Definition 5.2 (NTK Gram Matrix with Non-duplicate Arms). Follow the recursive process in (7).
With set ˘
AT containing non-duplicate arms, we denote the corresponding NTK matrix ˘H = ( ˘HL +
˘ΨL)/2 ∈R| ˘
AT |×| ˘
AT |, and expected reward vector ˘h = [h(x)]x∈˘
AT ∈R| ˘
AT |, with | ˘
AT | ≤3T.

Remark 5.3 (No Arm Separateness Assumption). Existing neural bandit works generally impose
separateness assumptions regarding the arm contexts: NTK-based approaches (e.g., [86, 84, 9, 51,
50]) commonly assume H ≻0 which requires no two arms are parallel among {xi,t}i∈[K],t∈[T ];
meanwhile, some other works (e.g., [11, 67]) assume the Euclidean separateness: ∥xi,t −xi′,t′∥2 > 0
if (i, t) ̸= (i′, t′), ∀i, i′ ∈[K], t, t′ ∈[T]. To avoid the arm separateness assumption, since ˘
AT
contains all the unique arms from AT , we alternatively build the confidence ellipsoid upon the NTK
matrix ˘H, and the ellipsoid will also hold for all the arms in AT for regret analysis (Lemma C.1).
This also leads to our tighter definition of NTK norm term S (Theorem 5.6, Remark 5.8).

Fact 5.4. Let ˘λ0 be the minimum eigenvalue of matrix ˘H, and λ0 be that of NTK matrix H. We
have (i) ˘λ0 = λmin( ˘H) > 0; and, (ii) ˘λ0 ≥λ0 ≥0.

For (i) in Fact 5.4, since ˘
AT contains no parallel arms, matrix ˘H will be full-rank, leading to
˘λ0 > 0. For (ii), if ˘
AT ̸= AT , then AT contains duplicate arms and matrix H will be singular, s.t.
˘λ0 > λ0 = 0. Otherwise, if ˘
AT = AT , it will naturally lead to ˘H = H and ˘λ0 = λ0. Next, similar
to existing neural bandit works (e.g., [86, 84]), we define the NTK Gram matrix effective dimension
ed, which essentially measures the vanishing speed of NTK Gram matrix eigenvalues.
Definition 5.5 (Effective Dimension of NTK Matrix [86, 84]). Given the NTK matrix H with
possibly duplicate arms (Def. 5.1), its effective dimension is defined as ed = log det(I+H/λ)

log(1+T K/λ) .

5.1
Regret Analysis for R-NeuralUCB

We follow the pseudo-regret R(T) = PT
t=1 E[er∗
t −ert] in (2), which is defined based on the expected
corruption-free reward of chosen arms and optimal arms across T rounds.

Instance-dependent Gradient Deviation Term β. Recall that for a candidate arm xi,t ∈Xt in round
t ∈[T], its arm weight w.r.t. previously chosen arm xτ, τ ∈[t−1] in (4) can be represented by w(τ)
i,t =
min

1, α · fracτ(xi,t; Xt, ¯Σt−1)
	
, with the scaling parameter α > 0. Here, we define a minimum
fraction value as β = mint∈[T ],τ∈[t−1]

min{fracτ(xt; Xt, ¯Σt−1), fracτ(ext; Xt, ¯Σt−1)}

, which

7


---Page Break---
is formulated to quantify the gradient deviation among arms. Here, the learner is not required to
know β, and we can adjust the scaling parameter α in each round t ∈[T] to constrain the round-wise
minimum weight value min{w(τ)
i,t }i∈[K],τ∈[t−1] (Subsection B.5), which leads to Theorem 5.6.

Theorem 5.6. With finite horizon T ∈N+, denote S ≥
p

2˘h
⊺˘H−1˘h, β > 0. Suppose λ ≥S−2, η ≤
O((TmL+mλ)−1), J ≥e
O(TL/λ). Let f(·) be an L-layer FC network with width m, and adjust the
scaling parameter α, s.t. min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2, ∀t ∈[T], for a tunable constant κ ∈(0, 1)
from (4). With δ ∈(0, 1), let network width m ≥Ω(poly(T, L, κ−1, ˘λ−1
0 , λ−1, S−1) log(δ−1)). With
probability at least 1 −δ, R-NeuralUCB achieves the regret bound of

R(T) ≤O

ν

r

ed log(λ + TK

λ
) −2 log(δ)+S
√

λ

e
O
q

T ed/κ2

+O

C edβ−1κ2 log(λ + TK

λ
)

.

The proof of Theorem 5.6 is in Appendix C. The first term on the RHS refers to the corruption-

independent regret upper bound, which comparably matches the bound e
O(ed
√

T +S
p

edT) in existing
corruption-free neural bandit works [86, 84]. Here, our corruption-dependent term is free of the NTK
norm S, which measures the complexity of reward mapping h(·) (Appendix B.4). This is different
from existing works (e.g., [15]) that include a parameter norm (similar to our NTK norm S) in their
corruption-dependent terms, as the estimation error of confidence ellipsoids. In addition, inspired by
[84], we can derive a T-independent upper bound for the β−1 term, when the arm contexts are nearly
spreading within some low-dimensional subspace of the NTK-induced RKHS (Appendix C.9), as it
will lead to small effective dimension ed and small eigenvalues of NTK matrix H [84]. Meanwhile,
compared with the regret bound of our base algorithm NeuralUCB-WGD (Theorem E.1), Theorem
5.6 removes the assumption of known corruption C; and, reduces the order of effective dimension ed
as well as the dependency of NTK norm S for corruption-dependent terms.
Remark 5.7 (Unknown corruption level C). For Theorem 5.6, we do not assume C is known to the
learner in advance, as practitioners can have little prior knowledge regarding the unknown adversary.
This makes our regret analysis more challenging, compared with the existing works (e.g., [16]) where
C is assumed known for setting hyper-parameters to achieve tight regret bounds.

Remark 5.8 (Tighter definition for NTK norm S). For existing works (e.g., [86, 84]), the NTK Gram
matrix is generally defined with all the TK observed candidate arms, i.e., {xi,t}i∈[K],t∈[T ], while
our NTK matrices (Def. 5.1 and 5.2) only rely on arm collection AT and set ˘
AT , with cardinality
| ˘
AT | ≤|AT | = 3T. This results in our parameter norm S that can be tighter compared to existing
works (e.g., [84, 86]), because when constructing the confidence ellipsoid around the initialization
Θ0 in Lemma C.1, our ellipsoid is intuitively tighter, as it only needs to ensure Eq. C.1 holds for
arms in ˘
AT (with cardinality | ˘
AT | ≤3T), rather than for all TK candidate arms.

Remark 5.9 (Reducing the order of ed and removing the dependency of S for corruption-dependent
terms). For corruption-dependent terms involving C, we have e
O(Cβ−1 ed). Using NTK to align the
information gain definition [16] with our effective dimension ed, our result improves latest kernelized
bandit works from e
O(ed3/2) to e
O(ed) for corruption-dependent terms, given the NTK-induced RKHS
and an indefinite arm space (Corollary 7 in [16]). Meanwhile, our corruption-dependent term is free
of the NTK norm S, while for some existing works with UCB-type exploration (e.g., [15]), they
involve comparable parameter norms in their corruption-dependent regret terms, in order to quantify
corruption impacts w.r.t. the reward mapping function complexity.
6
Experiments

We evaluate R-NeuralUCB and the base algorithm NeuralUCB-WGD (Appendix E) with experiments
on three real data sets, under different adversarial corruption scenarios. Following definition in (2), we
record the cumulative regret in terms of corruption-free rewards R(T) = P
t∈[T ]

er∗
t −ert

. Note that
the learner will still only have access to the potentially corrupted rewards rt, t ∈[T]. Our baselines
consist of linear algorithms: Lin-UCB [24], CW-OFUL [42]; and conventional neural algorithms:
Neural-UCB [86], Neural-TS [84]. Complementary experiment details are in Appendix A.

MovieLens and Amazon Data Sets. From “MovieLens 20M rating data set” [41], we choose 5,000
movies and 10,000 users with most reviews to form the user-movie matrix, and the entries are user
ratings. Then, we consider the arm (user-item pair) features as the concatenation of corresponding

8


---Page Break---
0
2500
5000
7500
10000
Time step

0

500

1000

1500

2000

2500

3000

3500

Cumulative regret

MovieLens dataset

Lin-UCB
CW-OFUL
Neural-TS
Neural-UCB
NeuralUCB-WGD
R-NeuralUCB

0
2500
5000
7500
10000
Time step

0

1000

2000

3000

4000

Amazon dataset

0
2500
5000
7500
10000
Time step

0

1000

2000

3000

4000

5000

6000
MNIST dataset

0
2500
5000
7500
10000
Time step

0

1000

2000

3000

4000

5000

Cumulative regret

MovieLens dataset

Lin-UCB
CW-OFUL
Neural-TS
Neural-UCB
NeuralUCB-WGD
R-NeuralUCB

0
2500
5000
7500
10000
Time step

0

2000

4000

6000

8000

Amazon dataset

0
2500
5000
7500
10000
Time step

0

1000

2000

3000

4000

5000

6000

7000

8000
MNIST dataset

Figure 1: Regret results on real data sets. (Left three figures: For MovieLens and Amazon, corrupt the chosen arm reward with 20% probability.
For MNIST, consider C = 2000 and randomly sample 2000 rounds for attack); (Right three figures: For MovieLens and Amazon: we corrupt
reward with 50% probability; For MNIST: C = 4000 and randomly sample 4000 corrupted rounds).

user features and item features, which are obtained by singular value decomposition (SVD) and
extracting item genome-scores respectively, with K = 10 and d = 41. The corruption-free arm
rewards eri,t are user ratings normalized into range [0, 1]. Here, we consider the “exaggerated reward
corruption”. If one pulled arm xt is attacked and its corruption-free reward ert ≥0.5, we exaggerate
its reward to rt = 1. Otherwise, if one pulled arm xt is attacked and ert < 0.5, we set its reward
rt = 0. Amazon Recommendation data set [43] consists of user reviews and corresponding ratings.
With each piece of review (user-item pair) as an arm, we vectorize the review as the arm features
using the “Sentire” package [85, 58], with K = 10 and d = 41. Similarly, the corruption-free arm
rewards eri,t are normalized user ratings with the value range [0, 1]. Different from MovieLens data
set, we here consider the “reverse exaggerated corruption”: if the pulled arm xt is attacked and its
corruption-free reward ert ≥0.5, we downplay its reward to rt = 0; or if the pulled arm xt is attacked
and ert < 0.5, we alternatively set the corrupted reward rt = 1.
MNIST Data Set. To perform online classification with bandit feedback experiment, we adopt
the MNIST data set [56] which consists of 10 image classes. Similar to previous works (e.g.,
[86, 84]), given a sample x ∈Rd′ in each round, we transform it into K = 10 arms, denoted by
x1 = (x, 0, . . . , 0), x2 = (0, x, . . . , 0), . . . , x10 = (0, 0, . . . , x) ∈R10×d′, s.t. d = 10 × d′. The
arm index that the learner chooses will be its predicted class, and the reward is 1 if the sample x
belongs to this class; otherwise, the reward will be 0. Here, we consider the symmetric “label-flipping”
attack [39]. For example, when a sample from digit class 2 is attacked, its corrupted label will be
switched to digit class 9 −2 = 7, and the corrupted arm rewards will also change accordingly.

Experiment Results. The experiment results are shown in Fig. 1, and we also include a parameter
study in Appendix A.2. Due to the representation power of neural networks, neural algorithms
generally perform better than linear ones. In particular, for the MNIST data set, since the reward
mapping can be relatively more complex, neural algorithms manage to achieve more significant
improvements over the linear algorithms. Here, compared with conventional neural methods, our
proposed NeuralUCB-WGD and R-NeuralUCB are more robust against adversarial reward corrup-
tions. In particular, we see that R-NeuralUCB outperforms NeuralUCB-WGD on these three data
sets, which helps support our claim that it is beneficial to involve the uncertainty information in
terms of both training samples and candidate arms. When we increase the corruption intensity (three
figures on the right), the overall results tend to be consistent with previous findings. Notice that
the performance gap among algorithms on the Amazon data set tends to be smaller, as this setting
becomes significantly more difficult (i.e., with up to ∼8000 regret) when we increase the corruption
probability to 50%. Meanwhile, for MNIST, when we increase C to 4000, the performance gap
between our proposed algorithms and the conventional neural methods tends to increase, as the task
becomes increasingly more complex. We also see that R-NeuralUCB still outperforms NeuralUCB-
WGD given the increased corruption intensity, showing the benefit of involving the candidate arm
information and customizing arm-specific model parameters.

7
Conclusion and Future Direction

In this paper, we propose a novel neural bandit algorithm named R-NeuralUCB to address poten-
tial adversarial corruption issues on arm rewards. To enhance model robustness against reward
corruptions, R-NeuralUCB applies a refined, context-aware Gradient Descent procedure that incor-
porates arm uncertainty information. To demonstrate its effectiveness, we present a regret analysis
of R-NeuralUCB to quantify the impacts of adversarial corruption. Furthermore, to ensure that
R-NeuralUCB can handle arm contexts deliberately chosen by an adversary (e.g., duplicate arms
across different rounds), our analysis avoids the commonly adopted arm separateness assumption in
neural bandit literature, which can be of independent interest. Empirical evaluations on real datasets
with varied specifications show the effectiveness of our proposed solution over baseline methods. A
challenging future direction is to derive the theoretical lower bound for neural bandits with corruption,
and we provide complementary discussions in Appendix B.7.

9


---Page Break---
Acknowledgments and Disclosure of Funding

This work is supported by National Science Foundation under Award No. IIS-2117902, and Agricul-
ture and Food Research Initiative (AFRI) grant no. 2020-67021-32799/project accession no.1024178
from the USDA National Institute of Food and Agriculture. The work is also supported in part by
the National Science Foundation through awards IIS 21-31335, OAC 21-30835, DBI 20-21898, as
well as a C3.ai research award. The views and conclusions are those of the authors and should not be
interpreted as representing the official policies of the funding agencies or the government.

References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear
stochastic bandits. Advances in neural information processing systems, 24:2312–2320, 2011.

[2] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear
payoffs. In ICML, pages 127–135. PMLR, 2013.

[3] Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via
over-parameterization. In International Conference on Machine Learning, pages 242–252.
PMLR, 2019.

[4] Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang.
On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955,
2019.

[5] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed
bandit problem. Machine learning, 47(2-3):235–256, 2002.

[6] Yikun Ban, Ishika Agarwal, Ziwei Wu, Yada Zhu, Kommy Weldemariam, Hanghang Tong, and
Jingrui He. Neural active learning beyond bandits. arXiv preprint arXiv:2404.12522, 2024.

[7] Yikun Ban and Jingrui He. Generic outlier detection in multi-armed bandit. In Proceedings of
the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining,
pages 913–923, 2020.

[8] Yikun Ban and Jingrui He. Local clustering in contextual multi-armed bandits. In Proceedings
of the Web Conference 2021, pages 2335–2346, 2021.

[9] Yikun Ban, Jingrui He, and Curtiss B Cook. Multi-facet contextual bandits: A neural network
perspective. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &
Data Mining, pages 35–45, 2021.

[10] Yikun Ban, Yunzhe Qi, Tianxin Wei, Lihui Liu, and Jingrui He. Meta clustering of neural
bandits. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, pages 95–106, 2024.

[11] Yikun Ban, Yuchen Yan, Arindam Banerjee, and Jingrui He. Ee-net: Exploitation-exploration
neural networks in contextual bandits. arXiv preprint arXiv:2110.03177, 2021.

[12] Yikun Ban, Yuchen Yan, Arindam Banerjee, and Jingrui He. Neural exploitation and exploration
of contextual bandits. arXiv preprint arXiv:2305.03784, 2023.

[13] Yikun Ban, Yuheng Zhang, Hanghang Tong, Arindam Banerjee, and Jingrui He. Improved
algorithms for neural active learning. Advances in Neural Information Processing Systems,
35:27497–27509, 2022.

[14] Alberto Bietti and Julien Mairal. On the inductive bias of neural tangent kernels. Advances in
Neural Information Processing Systems, 32, 2019.

[15] Ilija Bogunovic, Andreas Krause, and Jonathan Scarlett. Corruption-tolerant gaussian process
bandit optimization. In International Conference on Artificial Intelligence and Statistics, pages
1071–1081. PMLR, 2020.

10


---Page Break---
[16] Ilija Bogunovic, Zihan Li, Andreas Krause, and Jonathan Scarlett. A robust phased elimination
algorithm for corruption-tolerant gaussian process bandits. Advances in Neural Information
Processing Systems, 35:23951–23964, 2022.

[17] Ilija Bogunovic, Arpan Losalka, Andreas Krause, and Jonathan Scarlett. Stochastic linear
bandits robust to adversarial attacks. In International Conference on Artificial Intelligence and
Statistics, pages 991–999. PMLR, 2021.

[18] Sébastien Bubeck, Nicolo Cesa-Bianchi, et al. Regret analysis of stochastic and nonstochastic
multi-armed bandit problems. Foundations and Trends® in Machine Learning, 5(1):1–122,
2012.

[19] Xu Cai and Jonathan Scarlett. On lower bounds for standard and robust gaussian process bandit
optimization. In International Conference on Machine Learning, pages 1216–1226. PMLR,
2021.

[20] Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, and Quanquan Gu. Towards understanding
the spectral bias of deep learning. arXiv preprint arXiv:1912.01198, 2019.

[21] Yuan Cao and Quanquan Gu. Generalization bounds of stochastic gradient descent for wide and
deep neural networks. Advances in Neural Information Processing Systems, 32:10836–10846,
2019.

[22] Vasileios Charisopoulos, Hossein Esfandiari, and Vahab Mirrokni. Robust and private stochastic
linear bandits. In International Conference on Machine Learning, pages 4096–4115. PMLR,
2023.

[23] Christopher A Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. Label-
only membership inference attacks. In International conference on machine learning, pages
1964–1974. PMLR, 2021.

[24] Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff
functions. In AISTATS, pages 208–214, 2011.

[25] Zhongxiang Dai, Yao Shu, Arun Verma, Flint Xiaofeng Fan, Bryan Kian Hsiang Low, and
Patrick Jaillet. Federated neural bandit. arXiv preprint arXiv:2205.14309, 2022.

[26] Rohan Deb, Yikun Ban, Shiliang Zuo, Jingrui He, and Arindam Banerjee. Contextual bandits
with online neural regression. arXiv preprint arXiv:2312.07145, 2023.

[27] Yashar Deldjoo, Tommaso Di Noia, and Felice Antonio Merra. A survey on adversarial
recommender systems: from attack/defense strategies to generative adversarial networks. ACM
Computing Surveys (CSUR), 54(2):1–38, 2021.

[28] Aniket Anand Deshmukh, Urun Dogan, and Clay Scott. Multi-task learning for contextual
bandits. In NeurIPS, pages 4848–4856, 2017.

[29] Qin Ding, Cho-Jui Hsieh, and James Sharpnack. Robust stochastic linear contextual bandits
under adversarial attacks. In International Conference on Artificial Intelligence and Statistics,
pages 7111–7123. PMLR, 2022.

[30] Audrey Durand, Charis Achilleos, Demetris Iacovides, Katerina Strati, Georgios D Mitsis,
and Joelle Pineau. Contextual bandits for adapting treatment in a mouse model of de novo
carcinogenesis. In Machine learning for healthcare conference, pages 67–82. PMLR, 2018.

[31] Li Fan, Ruida Zhou, Chao Tian, and Cong Shen. Federated linear bandits with finite adversarial
actions. Advances in Neural Information Processing Systems, 36, 2024.

[32] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adap-
tation of deep networks. In International conference on machine learning, pages 1126–1135.
PMLR, 2017.

[33] Dylan J Foster, Claudio Gentile, Mehryar Mohri, and Julian Zimmert. Adapting to mis-
specification in contextual bandits.
Advances in Neural Information Processing Systems,
33:11478–11489, 2020.

11


---Page Break---
[34] Dongqi Fu, Liri Fang, Ross Maciejewski, Vetle I. Torvik, and Jingrui He. Meta-learned metrics
over multi-evolution temporal graphs. In KDD 2022, 2022.

[35] Dongqi Fu and Jingrui He. SDG: A simplified and dynamic graph neural network. In SIGIR
2021, 2021.

[36] Avishek Ghosh, Sayak Ray Chowdhury, and Aditya Gopalan. Misspecified linear bandits. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.

[37] Quanquan Gu, Amin Karbasi, Khashayar Khosravi, Vahab Mirrokni, and Dongruo Zhou.
Batched neural bandits. ACM/IMS Journal of Data Science, 1(1):1–18, 2024.

[38] Anupam Gupta, Tomer Koren, and Kunal Talwar. Better algorithms for stochastic bandits with
adversarial corruptions. In Conference on Learning Theory, pages 1562–1578. PMLR, 2019.

[39] B Han, Q Yao, X Yu, G Niu, M Xu, W Hu, I Tsang, and M Sugiyama. Robust training of deep
neural networks with extremely noisy labels. In Thirty-fourth Conference on Neural Information
Processing Systems (NeurIPS), volume 2, page 4, 2020.

[40] Eric Han and Jonathan Scarlett. Adversarial attacks on gaussian process bandits. In International
Conference on Machine Learning, pages 8304–8329. PMLR, 2022.

[41] F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm
transactions on interactive intelligent systems (tiis), 5(4):1–19, 2015.

[42] Jiafan He, Dongruo Zhou, Tong Zhang, and Quanquan Gu. Nearly optimal algorithms for linear
contextual bandits with adversarial corruptions. arXiv preprint arXiv:2205.06811, 2022.

[43] Ruining He and Julian McAuley. Ups and downs: Modeling the visual evolution of fashion
trends with one-class collaborative filtering. In proceedings of the 25th international conference
on world wide web, pages 507–517, 2016.

[44] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural
collaborative filtering. In WWW, pages 173–182, 2017.

[45] Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 2012.

[46] Taehyun Hwang, Kyuwook Chai, and Min-hwan Oh. Combinatorial neural bandits. In Interna-
tional Conference on Machine Learning, pages 14203–14236. PMLR, 2023.

[47] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. Advances in neural information processing systems, 31,
2018.

[48] Yiling Jia, Weitong ZHANG, Dongruo Zhou, Quanquan Gu, and Hongning Wang. Learning
neural contextual bandits through perturbed rewards. In International Conference on Learning
Representations, 2021.

[49] Yue Kang, Cho-Jui Hsieh, and Thomas Chun Man Lee. Robust lipschitz bandits to adversarial
corruptions. Advances in Neural Information Processing Systems, 36, 2024.

[50] Parnian Kassraie and Andreas Krause. Neural contextual bandits without regret. In International
Conference on Artificial Intelligence and Statistics, pages 240–278. PMLR, 2022.

[51] Parnian Kassraie, Andreas Krause, and Ilija Bogunovic. Graph neural network bandits. arXiv
preprint arXiv:2207.06456, 2022.

[52] Johannes Kirschner and Andreas Krause. Bias-robust bayesian optimization via dueling bandits.
In International Conference on Machine Learning, pages 5595–5605. PMLR, 2021.

[53] Sanath Kumar Krishnamurthy, Vitor Hadad, and Susan Athey. Adapting to misspecification
in contextual bandits with offline regression oracles. In International Conference on Machine
Learning, pages 5805–5814. PMLR, 2021.

12


---Page Break---
[54] Yuko Kuroki, Alberto Rumi, Taira Tsuchiya, Fabio Vitale, and Nicolò Cesa-Bianchi. Best-of-
both-worlds algorithms for linear contextual bandits. In International Conference on Artificial
Intelligence and Statistics, pages 1216–1224. PMLR, 2024.

[55] Tor Lattimore, Csaba Szepesvari, and Gellert Weisz. Learning with good feature representations
in bandits and in rl with a generative model. In International conference on machine learning,
pages 5662–5670. PMLR, 2020.

[56] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning
applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[57] Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, Mengxiao Zhang, and Xiaojin Zhang. Achieving
near instance-optimality and minimax-optimality in stochastic and adversarial linear bandits
simultaneously. In International Conference on Machine Learning, pages 6142–6151. PMLR,
2021.

[58] Lei Li, Yongfeng Zhang, and Li Chen. Generate neural template explanations for recommenda-
tion. In Proceedings of the 29th ACM International Conference on Information & Knowledge
Management, pages 755–764, 2020.

[59] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to
personalized news article recommendation. In WWW, pages 661–670, 2010.

[60] Xiaoqiang Lin, Zhaoxuan Wu, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick
Jaillet, and Bryan Kian Hsiang Low. Use your instinct: Instruction optimization using neural
bandits coupled with transformers. arXiv preprint arXiv:2310.02905, 2023.

[61] Brad Miller, Alex Kantchelian, Sadia Afroz, Rekha Bachwani, Edwin Dauber, Ling Huang,
Michael Carl Tschantz, Anthony D Joseph, and J Doug Tygar. Adversarial active learning. In
Proceedings of the 2014 workshop on artificial intelligent and security workshop, pages 3–14,
2014.

[62] David J Miller, Zhen Xiang, and George Kesidis. Adversarial learning targeting deep neural
network classification: A comprehensive review of defenses against attacks. Proceedings of the
IEEE, 108(3):402–433, 2020.

[63] Aritra Mitra, Arman Adibi, George J Pappas, and Hamed Hassani. Collaborative linear bandits
with adversarial agents: Near-optimal regret bounds. Advances in neural information processing
systems, 35:22602–22616, 2022.

[64] Bamshad Mobasher, Robin Burke, Runa Bhaumik, and Chad Williams. Toward trustworthy rec-
ommender systems: An analysis of attack models and algorithm robustness. ACM Transactions
on Internet Technology (TOIT), 7(4):23–es, 2007.

[65] Rui Ning, Jiang Li, Chunsheng Xin, and Hongyi Wu. Invisible poison: A blackbox clean
label backdoor attack to deep neural networks. In IEEE INFOCOM 2021-IEEE Conference on
Computer Communications, pages 1–10. IEEE, 2021.

[66] Yunzhe Qi, Yikun Ban, and Jingrui He. Neural bandit with arm group graph. arXiv preprint
arXiv:2206.03644, 2022.

[67] Yunzhe Qi, Yikun Ban, and Jingrui He. Graph neural bandits. In Proceedings of the 29th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, page 1920–1931,
New York, NY, USA, 2023. Association for Computing Machinery.

[68] Yunzhe Qi, Yikun Ban, Tianxin Wei, Jiaru Zou, Huaxiu Yao, and Jingrui He. Meta-learning
with neural bandit scheduler. Advances in Neural Information Processing Systems, 36, 2024.

[69] Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, and Zico Kolter. Certified robustness to
label-flipping attacks via randomized smoothing. In International Conference on Machine
Learning, pages 8230–8241. PMLR, 2020.

[70] Jonathan Scarlett, Ilija Bogunovic, and Volkan Cevher. Lower bounds on regret for noisy
gaussian process bandit optimization. In Conference on Learning Theory, pages 1723–1742.
PMLR, 2017.

13


---Page Break---
[71] Ruixiang Tang, Mengnan Du, Ninghao Liu, Fan Yang, and Xia Hu. An embarrassingly simple
approach for trojan attack in deep neural networks. In Proceedings of the 26th ACM SIGKDD
international conference on knowledge discovery & data mining, pages 218–228, 2020.

[72] Michal Valko, Nathan Korda, Rémi Munos, Ilias Flaounas, and Nello Cristianini. Finite-time
analysis of kernelised contextual bandits. In Uncertainty in Artificial Intelligence, 2013.

[73] Sofía S Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal
design of clinical trials: benefits and challenges. Statistical science: a review journal of the
Institute of Mathematical Statistics, 30(2):199, 2015.

[74] Zhilei Wang, Pranjal Awasthi, Christoph Dann, Ayush Sekhari, and Claudio Gentile. Neural
active learning with performance guarantees. Advances in Neural Information Processing
Systems, 34:7510–7521, 2021.

[75] Zhiyong Wang, Jize Xie, Xutong Liu, Shuai Li, and John Lui. Online clustering of bandits with
misspecified user models. Advances in Neural Information Processing Systems, 36, 2024.

[76] Chen-Yu Wei, Christoph Dann, and Julian Zimmert. A model selection approach for corruption
robust reinforcement learning. In International Conference on Algorithmic Learning Theory,
pages 1043–1096. PMLR, 2022.

[77] Max Welling and Thomas N Kipf. Semi-supervised classification with graph convolutional
networks. In International Conference on Learning Representations, 2017.

[78] Longfeng Wu, Yao Zhou, and Dawei Zhou. Towards high-order complementary recommen-
dation via logical reasoning network. In Xingquan Zhu, Sanjay Ranka, My T. Thai, Takashi
Washio, and Xindong Wu, editors, IEEE International Conference on Data Mining, ICDM 2022,
Orlando, FL, USA, November 28 - Dec. 1, 2022, pages 1227–1232. IEEE, 2022.

[79] Qingyun Wu, Huazheng Wang, Quanquan Gu, and Hongning Wang. Contextual bandits in a
collaborative environment. In SIGIR, pages 529–538, 2016.

[80] Yulian Wu, Xingyu Zhou, Youming Tao, and Di Wang. On private and robust bandits. Advances
in Neural Information Processing Systems, 36, 2024.

[81] Pan Xu, Zheng Wen, Handong Zhao, and Quanquan Gu. Neural contextual bandits with deep
representation and shallow exploration. arXiv preprint arXiv:2012.01780, 2020.

[82] Fuzhi Zhang and Quanqiang Zhou. Ensemble detection model for profile injection attacks in
collaborative recommender systems based on bp neural network. IET Information Security,
9(1):24–31, 2015.

[83] Weitong Zhang, Jiafan He, Zhiyuan Fan, and Quanquan Gu. On the interplay between misspeci-
fication and sub-optimality gap in linear contextual bandits. In International Conference on
Machine Learning, pages 41111–41132. PMLR, 2023.

[84] Weitong Zhang, Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural thompson sampling. In
International Conference on Learning Representations, 2021.

[85] Yongfeng Zhang, Haochen Zhang, Min Zhang, Yiqun Liu, and Shaoping Ma. Do users rate
or review? boost phrase-level sentiment labeling with review-level sentiment classification. In
Proceedings of the 37th international ACM SIGIR conference on Research & development in
information retrieval, pages 1027–1030, 2014.

[86] Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based
exploration. In International Conference on Machine Learning, pages 11492–11502. PMLR,
2020.

[87] Yao Zhou, Haonan Wang, Jingrui He, and Haixun Wang. From intrinsic to counterfactual: On
the explainability of contextualized recommender systems. CoRR, abs/2110.14844, 2021.

[88] Yao Zhou, Jianpeng Xu, Jun Wu, Zeinab Taghavi, Evren Korpeoglu, Achan Kannan, and Jingrui
He. Pure: Positive-unlabeled recommendation with generative adversarial network. In KDD,
2021.

14


---Page Break---
[89] Shiliang Zuo. Corruption-robust lipschitz contextual search. In International Conference on
Algorithmic Learning Theory, pages 1234–1254. PMLR, 2024.

15


---Page Break---
A
Experiment Settings and Additional Experiments

A.1
Experiment settings

For all UCB-based baselines, we choose the exploration parameter through grid search over the
range {0.01, 0.1, 1}. We set L = 2 for all deep learning models, including our proposed NeuralUCB-
WGD and R-NeuralUCB, and set the network width to m = 200. The learning rate for all neural
algorithms is chosen by grid search from the range {0.0001, 0.001, 0.01}. For all methods, we select
the regularization parameter λ from the range {0.0001, 0.001, 0.01}. The scaling parameter α for
NeuralUCB-WGD and R-NeuralUCB is chosen from {0.2, 0.5, 1}. All experiments are conducted
on a server with an Intel Xeon CPU and NVIDIA V100 GPUs. Additionally, we provide further
details on our baseline methods, which include two linear algorithms and two conventional neural
algorithms:

• Lin-UCB [24, 59] uses linear regression as the reward estimation model and employs a UCB-based
strategy for exploration.

• CW-OFUL [42] applies weighted linear ridge regression in instead of the standard one from
Lin-UCB, with weights assigned to selected samples in proportion to reward estimation confidence.

• Neural-UCB [86] employs a single neural network to estimate arm rewards and calculates the UCB
based on network gradients for exploration.

• Neural-TS [84] utilizes a fully connected network for arm reward estimation, along with the
Thompson Sampling strategy [2] for exploration.

Additional Data Processing Details for Recommendation Data Sets. Here, we provide additional
details about our data processing procedure. For the first data set, MovieLens 20M rating data set
(https://grouplens.org/datasets/movielens/20m/), we initially select 5,000 movies and
10,000 users with the most reviews to form a user-movie matrix, where the entries represent user
ratings. The user features xu ∈Rd′ are derived via singular value decomposition (SVD) with a
dimensionality of d′ = 20. Using the genome scores provided for each movie, we select the 20
tags with the highest variance and used their corresponding scores as movie features vi ∈Rd′. At
each time step t, given a user ut, we encode user information into the arm contexts following the
Generalized Matrix Factorization (GMF) approach [44, 88] by concatenating the features xi,t =
[xut; vi] ∈R2d′, where c ∈Ct and i ∈[K], with K = 10. Finally, we concatenate a constant 0.01 to
each xi,t and normalize the entire vector to obtain xi,t ∈Rd, where d = 41. The corruption-free arm
rewards eri,t are user ratings normalized to the range [0, 1]. We consider the scenario of "exaggerated
reward corruption": If a pulled arm xt is attacked and its corruption-free reward ert ≥0.5, we
exaggerate its reward to rt = 1. Conversely, if a pulled arm xt is attacked and its corruption-free
reward ert < 0.5, we downplay its reward to rt = 0.

For the Amazon Recommendation data set (https://jmcauley.ucsd.edu/data/amazon/
index_2014.html), each user-item pair is associated with a review and the corresponding user
rating. We transform the review text into vector representations to derive the arm contexts, following
the text processing procedure in the "Sentires" package [85, 58]. We then set d = 41 and apply L2
normalization, with an arm pool size of K = 10. Similarly, the corruption-free arm rewards eri,t
are normalized user ratings in the range [0, 1]. Unlike the MovieLens data set, we apply a "reverse
exaggerated corruption" approach here: If the pulled arm xt is attacked and its corruption-free reward
ert ≥0.5, we downplay its reward to rt = 0. Conversely, if the pulled arm xt is attacked and its
corruption-free reward ert < 0.5, we set the corrupted reward to rt = 1.

A.2
Additional experiments: parameter study

We also include additional experiments with different regularization parameter values λ and ex-
ploration parameter values ν. On the MNIST data set (corruption level C = 2000), we conduct
experiments for NeuralUCB-WGD and R-NeuralUCB. We present the parameter study results in
Tables 2 and 3. For both of our proposed algorithms, setting ν ∈(0.1, 0.5] generally yields the
best performance. However, with an overly small exploration coefficient (e.g., ν = 0.05), optimal
empirical performance may not be achievable. Meanwhile, setting λ to smaller values, such as 0.001
or 0.0001, tends to result in the best performance. Increasingly large regularization parameter values

16


---Page Break---
Algorithm \ λ value
λ = 0.1
λ = 0.01
λ = 0.001
λ = 0.0001

NeuralUCB-WGD
3244 ±95
2782 ±90
2156 ±105
2103 ±119
R-NeuralUCB
2933 ±101
2501 ±93
1989 ±67
2127 ±92

Table 2: Regret results for different exploration regularization parameter values λ (with std.)

Algorithm \ ν value
ν = 1
ν = 0.5
ν = 0.1
ν = 0.05

NeuralUCB-WGD
2510 ±102
2154 ±121
2103 ±119
2278 ±85
R-NeuralUCB
2793 ±104
2163 ±88
1989 ±67
2197 ±65

Table 3: Regret results for different exploration parameter values ν (with std.)

can cause the trained model parameters θ to remain close to their random initialization θ0, which can
overly constrain the neural network’s capacity to fit the underlying reward mapping function. Thus,
practitioners can adjust the λ value based on specific application needs, as is common in other neural
bandit studies (e.g., [86]). In practice, starting with small values like 10−4 and performing a grid
search to identify the optimal λ is a reasonable approach for R-NeuralUCB and NeuralUCB-WGD.

B
Complementary Discussions on the Content of the Main Body

In this section, we provide additional discussion to complement the main body content.

B.1
Boarder impacts

Since our objective is to deal with the potential adversarial attacks in machine learning applications,
this work can contribute to the goal of achieving trustworthy machine learning for general practitioners.
Therefore, we do not perceive significant negative societal impacts that can be generated by this work.

B.2
Limitations

One limitation of this work is the absence of a theoretical lower bound for neural bandits with
adversarial corruptions. We would like to mention that this problem is significantly challenging and
non-trivial. Given that we deal with an arbitrary reward function h(·), which is considerably different
from linear [42] and kernelized bandits [16], deriving the lower bound itself can lead to substantial
contributions, potentially leading to a separate line of research works (e.g., [70] under kernelized
bandit settings). Therefore, we consider the derivation of such a lower bound for neural bandits with
adversarial corruptions as an interesting and challenging future direction of this work. Additional
discussions on the lower bound can be found in Subsec. B.7.

B.3
Theoretical contributions and comparisons with vanilla Neural-UCB

Recall that we propose deriving the regret bound using NTK-based regression techniques. Unlike
linear bandit approaches (e.g., [42]) and kernel bandit methods (e.g., [16]), our regression is conducted
on the network gradients g(·; θ) := vec(∇θf(·; θ)), which serve as the mapping for gradient-based
NTK. In this framework, even for the same arm x, the NTK-embedded arm contexts can differ due
to the corrupted parameters g(x; θ) and the corruption-free parameters g(x; eθ). To address this
challenge, we define two sets of regression parameters corresponding to the corrupted model and
the corruption-free model respectively. Then, using the corruption-free model f(·; eθ), we derive the
confidence ellipsoid around its parameters eθ. This serves as a proxy to quantify the parameter shift of
the trained corrupted model parameters θ, enabling us to establish the regret upper bound.

For the theoretical analysis of R-NeuralUCB presented in Theorem 5.6, we considerably modify
the regret analysis workflow due to the following reasons: (i) To achieve improved performance,
R-NeuralUCB differs from conventional neural bandit approaches by tuning separate sets of network
parameters for each candidate arm after perceiving arm context information; (ii) To achieve a tighter
regret bound and eliminate the assumption of a known corruption level C, unlike in Theorem E.1,

17


---Page Break---
we cannot quantify the impact of C using the confidence ellipsoid. To address this, let x∗
t and xt
represent the optimal arm and the chosen arm by the corrupted model, respectively. We decompose the
single-round pseudo-regret Rt = min{h(x∗
t ) −h(xt), 1} into three components: (i) The prediction
error of the corruption-free model; (ii) The reward estimation discrepancy between the corruption-free
model and the corrupted model on the same arm; (iii) The arm selection discrepancy of the corrupted
model induced by adversarial corruptions. Next, we apply carefully designed arm weights in (4) to
guide the gradient descent process and mitigate the impact of adversarial corruptions. As a result,
R-NeuralUCB achieves non-trivial theoretical improvements: (i) removing the assumption of a known
corruption level C in regret analysis; (ii) eliminating the dependency on the NTK norm term S for
corruption-dependent terms in the regret bound; (iii) reducing the order of corruption-dependent
terms in the regret bound to the effective dimension ed, from O(ed3/2) to O(ed), compared with our
base algorithm NeuralUCB-WGD and existing kernelized bandit algorithms (e.g., [16]).

In addition, as mentioned in Remark 5.3, existing neural bandit approaches typically impose sepa-
rateness assumptions on observed arm contexts, whereas we do not. For example, Neural-UCB [86]
assumes H ≻0, which requires that no two arms are parallel among {xi,t}i∈[K],t∈[T ]. In contrast,
by formulating our NTK Gram matrices (Definitions 5.1 and 5.2), we complete our proof without the
separateness assumption, reinforcing the theoretical robustness of our approach against possible arm
contexts selected by an adversary (e.g., duplicate arm contexts across time steps). As in Remark 5.8,
existing methods, including Neural-UCB [86, 84], generally define their NTK Gram matrices over all
TK observed arms, i.e., {xi,t}i∈[K],t∈[T ]. However, our NTK matrices (Definitions 5.1 and 5.2) are
based on AT and ˘
AT , where | ˘
AT | ≤|AT | = 3T. This formulation can result in a tighter NTK norm
S compared with existing methods.

Upper bound for vanilla Neural-UCB. Meanwhile, to provide insights into the regret bound of
Neural-UCB, one possible approach is to follow a similar analysis to the regret bound of NeuralUCB-
WGD. The key idea here is to quantify the impact of adversarial corruptions on the confidence
ellipsoid around the trained parameters. Referring to the derivations in Lemma F.1, and denoting the
corruption-free confidence radius in round t as ˜γt−1, we obtain the corrupted confidence ellipsoid for
Neural-UCB as Ct−1 =

θ : ∥θ −θt−1∥Γt−1 ≤γt−1/√m
	
, where γt−1 = ˜γt−1 + O(CLλ−1/2).
This result is derived by setting wτ = 1 for τ ∈[t −1] and applying the fact that P

τ∈[t] cτ ≤C,
along with Lemma G.2 and the initialization of the gradient covariance matrix Γ. Following the proof

flow of Lemma 5.3 in [86], we obtain a regret upper bound of ˜O( ˜d
√

T +
p

S ˜dT + CL
q

˜dT/λ),

which introduces an additional ˜O(
√

T) to the corruption-dependent term.

Over-parameterization. For most neural bandit works with experiments (e.g., [84, 86, 25, 9, 11]), a
gap exists between experiments and theoretical analysis. On one hand, as the number of layers L
and hidden dimension m increase, neural networks become progressively harder to train, more time-
consuming in inference, and more resource-intensive. To make neural bandits feasible for practical
applications, these works generally use a neural network of ordinary size for experiments. It has
been shown that even with ordinary-sized neural networks, neural bandit algorithms achieve notable
performance gains over linear and kernel-based methods [86, 84, 9, 11, 67]. On the other hand, from
a theoretical standpoint, neural networks need to be over-parameterized, with m ≥O(poly(T)), to
approximate any arbitrary reward mapping function h(·). Additionally, with over-parameterization,
the difference between NTK-based regression models and neural networks becomes sufficiently small
for regret analysis, which is essential in neural bandit research. Therefore, we use a two-layer fully
connected network for experiments while performing theoretical analysis under over-parameterized
settings, as in most existing neural bandit works (e.g., [86, 84, 9, 11, 67]).

B.4
The definition and order of NTK norm parameter S

Recall that we have the NTK norm S defined as the upper bound of the weighted norm S ≥∥˘h∥˘H−1,
where h refers to the vector of expected rewards and ˘H refers to the NTK Gram matrix (Definition
5.2). We can follow existing neural bandit works [86, 50, 51, 84, 48] by considering that the reward
mapping function h(·) in (1) belongs to the Reproducing Kernel Hilbert Space (RKHS) H induced
by NTK. In this case, we can upper bound S with the RKHS norm, such that ∥h∥H ≥S, and the
RKHS norm ∥h∥H will not grow along with the finite horizon T (Remark 4.8 in [86]).

18


---Page Break---
Meanwhile, we also would like to mention that this is a common formulation, and nearly all the
neural bandit works (e.g., [86, 50, 51, 84, 9, 48]) will include a comparable NTK norm term in
the regret bound. This is because the regret analysis of neural bandits is generally depending on
the NTK regression approach. In this case, when constructing the confidence ellipsoid within the
NTK-induced RKHS, we will need to involve the RKHS norm as the cost. Analogously, for the
kernelized contextual bandits works (e.g., [72, 28, 16]), they inevitably involve the RKHS norm into
the regret bound. For linear bandit works (e.g., [42]), they will also need to include an assumed
upper bound of the true parameter θ∗norm in the Euclidean space, such that ∥θ∗∥2 is bounded by a
constant. Meanwhile, since the NTK norm term S stays invariant across candidate arms xi,t ∈Xt,
we can treat S as a constant in practice (e.g., setting S = 1), and control the exploration intensity by
tuning the exploration parameter ν.

B.5
Details regarding the scaling of arm weights w

Recall that when defining the sample weights w(τ)
i,t in (4), we use the minimum gradient norm
in the numerator to scale weights across the current candidate arms Xt, while introducing the
scaling parameter α > 0 to provide additional control from the practitioner’s perspective. Under
the stochastic contextual bandit settings, the learner receives the candidate arm pool Xt in each
round t from the environment, having little control over the minimum gradient norm, as Xt is only
revealed at round t. To address this, we introduce a tunable parameter α to control the minimum
value of w(τ)
i,t , aiming for a more stable learning process. Additionally, when deriving the regret
bound for R-NeuralUCB (Theorem 5.6), we scale the α values to ensure that the minimum weight
value is κ2. Without this scaling, such as by setting α = 1, an extra corruption-independent term

O(
q

β−1T ed log(1 + TK/λ)) would be added to the current regret bound, making the overall bound
less tight. Therefore, the scaling parameter α is essential for R-NeuralUCB.

Furthermore, the denominator of (4) consists of the product of two gradient norms: (i) the norm of
the previously chosen arm gτ, and (ii) the norm of the candidate arm ∥g(xi,t; θ)/√m∥Σ−1. Here,
we use the squared norm in the numerator to balance with the norm product in the denominator. This
design is also critical for deriving the regret bound in Theorem 5.6. Without using the squared norm,
our current derivation would yield a corruption-dependent term of e
O(ed
√

Tβ−1C), rather than the
current e
O(edβ−1C).

To be specific, for scaling the arm weight based on κ, we first recall that in round t ∈[T], we have arm
weights w(τ)
i,t , τ ∈[t −1], i ∈[K]. We can also denote w(τ)
i,t = min

1, α · fracτ(xi,t; Xt, ¯Σt−1)
	
.
Here, instead of deeming α as a fixed value across horizon T, we can consider α to be varying
across different rounds, denoted by αt, t ∈[T]. With a shorthand for minimum fraction value
fracmin
t
= mini∈[K],τ∈[t−1]

fracτ(xi,t; Xt, ¯Σt−1)

, we can set each αt = κ2/fracmin
t
, κ ∈(0, 1).

As a result, we can consequently have min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2, ∀t ∈[T].

B.6
Warm-start training for candidate arms

Recall that in each round t ∈{2, . . . , T}, we need to train different sets of arm-specific parameters
θi,t−1, i ∈[K] according to Algorithm 1, for each of the candidate arms xi,t ∈Xt, i ∈[K]. As
we have mentioned in the main body, we can adopt the idea of warm-start GD [13] in practice.
Here, instead of training each set of parameters θi,t−1 from the randomly initialized θ0, we tune
arm-specific parameters for each xi,t ∈Xt with a small number of samples from current received
records Pt−1. With the formulated arm weights w(τ)
i,t in (4), we first recall the arm-specific loss
function associated with arm xi,t as

Li,t(Pt−1; θ) =
X

(xτ ,rτ )∈Pt−1

w(τ)
i,t
2
·
f(xτ; θ) −rτ
2 + mλ

2
· ∥θ −θ0∥2
2.

The idea is that instead of starting from θ0, we can initiate the GD process from the existing
network parameters θt−2 from the previous round t −1, where θt−2 = θit−1,t−2 represents the
parameters associated with the chosen arm xt−1 = xit−1,t−1 in round t −1. The pseudo-code for
this arm-specific warm-start GD process is provided in Algorithm 2.

19


---Page Break---
Algorithm 2 Warm-start training for R-NeuralUCB

1: Input: Candidate arm xi,t ∈Xt. Training steps ¯J. Learning rates η. Batch size B. Regulariza-
tion parameter λ. Network parameters θt−2 from round t −1. Received records Pt−1.
2: Output: Trained arm-specific network parameters θi,t−1 for arm xi,t ∈Xt.

3: Sample a batch of training samples from Pt−1, denoted by bPt−1 ⊆Pt−1, where | bPt−1| = B.
Following (4), calculate arm weights w(τ)
i,t for samples in bPt−1.

4: θ(0)
i,t−1 ←θt−2.
5: for each training step j ∈¯J do

6:
θ(j)
i,t−1 = θ(j−1)
i,t−1 −η∇θLi,t( bPt−1; θ(j−1)
i,t−1 )
7: end for
8: θi,t−1 ←θ( ¯
J)
i,t−1.
9: Return arm-specific network parameters θi,t−1.

Hidden = 200
Hidden = 300
Hidden = 500
Index

0

5000

10000

15000

20000

Parameters

Number of parameters

0
2000
4000
6000
8000
10000
Round

0.6

0.7

0.8

0.9

1.0

1.1

1.2

1.3

Inference Time

Inference time for MovieLens

Hidden = 200
Hidden = 300
Hidden = 500

0
2000
4000
6000
8000
10000
Round

0.6

0.7

0.8

0.9

1.0

1.1

1.2

1.3

Inference Time (s)

Inference time for MNIST

Figure 2: Number of parameters with hidden dimensions m. Inference time with warm-start.

As a result, in our experiments, to balance computational costs and model performance, we implement
the following strategies: (1) Inspired by meta-learning approaches [32], we apply warm-start gradient
descent (GD) by adapting previously trained network parameters θt−2 for each candidate arm, using
a small number of training samples rather than starting from θ0 with a large sample size; (2) Based
on our formulation of the warm-start GD process, we sample a fixed number of mini-batch training
samples (i.e., received arm-reward pairs) for each candidate arm to compute arm weights and perform
GD. Using a fixed number of training samples helps keep round-wise inference time relatively stable,
avoiding a drastic increase with T. Figure 2 illustrates the parameter count and inference time across
different hidden dimensions. As shown, inference time remains relatively stable due to the fixed
number of adaptation samples used for warm-start GD.

B.7
Additional discussions on the lower bound

Under linear bandit settings, there is a model-agnostic lower bound of corruption-dependent term
Ω(Cd) with probability at least 1/2 [17], which will also hold for neural bandit works as our h(·)
can be an arbitrary function. Meanwhile, the lower bounds for kernelized bandits tend to vary
depending on kernel characteristics, e.g., Ω(C(log(T))d/2) for the SE kernel and Ω(C
v
d+v T
v
d+v ) for
the v-Matérn kernel [70, 16]. In this case, the order of term C and whether the lower bound depends
on non-logarithmic T, will both depend on the kernel properties. Therefore, given close connections
between NTK-based regression and over-parameterized networks, we hypothesize that such a lower
bound for neural bandits with corruption can depend on NTK properties. However, it will require
significant efforts and a well-established existing knowledge base (e.g., number of functions M
needed for the functional separateness condition [70, 19] regarding specified kernels) to obtain such
a lower bound for non-linear cases, especially considering few restrictions are imposed for reward
mapping h(·) for neural bandits. Since there are no existing works from neural bandits or NTK
perspectives, it can lead to a different line of research work by proving these results. Therefore, we
consider providing such a corruption-dependent regret lower bound as a challenging future direction.

Meanwhile, when C = 0, we obtain a corruption-free regret of ˜O( ˜d
√

T + S
p

˜dT). By setting

C = Ω(RT /d), the regret bound becomes ˜O

( ˜d2√

T + S ˜d3/2√

T) · Cβ−1d−1
. We note that,
following the proof flow of Theorem 4.12 in [42] and the learning problem defined in Assumption

20


---Page Break---
2.1 of [42], our effective dimension term ˜d2 may depend on the horizon T and can grow with T [26].
Consequently, although the regret bound contains only
√

T terms, the overall order of the regret
bound could reach or exceed O(T) due to the effective dimension ˜d2, as discussed in [26]. This
behavior differs from that of linear bandits, where regret bounds generally depend on the horizon
T and other T-independent terms, such as context dimension d and a fixed T-independent linear
parameter norm ∥θ∗∥2. Therefore, our Theorem 5.6 will not contradict Theorem 4.12 in [42].

21


---Page Break---
C
Regret Analysis for R-NeuralUCB

To begin with, recall that we aim to minimize the pseudo-regret for T rounds, denoted by

R(T) =

T
X

t=1
Rt =

T
X

t=1


h(x∗
t ) −h(xt)


where the second equality is due to the definition of reward mapping h in (1). We denote f(·) as
the bandit model we currently possess, which is trained with corrupted records Pt−1 up to round t.
Similarly, we can also suppose a corresponding imaginary corruption-free bandit model, which is
trained with corruption-free records ePt−1. Similarly, the model parameters of our possessed f(·) are
denoted as θ, while the parameters of the imaginary corruption-free model will be denoted as eθ.

Subsections outline and proof sketch. The content in this section is organized into the following
sub-components: In Subsection C.1, we first present theoretical properties related to our definitions of
the NTK Gram matrix (Definitions 5.1 and 5.2); In Subsection C.2, we decompose the single-round
objective Rt, t ∈[T] into its sub-components: (i) the first component represents the prediction
error of the corruption-free model; (ii) the second component measures the potential arm selection
discrepancy of the corrupted model caused by adversarial corruptions; and (iii) the third component
captures the estimation discrepancy between the corruption-free model and the corrupted model.
Next, in Subsection C.3, we bound the cumulative pseudo-regret R(T) (proof of Theorem 5.6). Using
the auxiliary sequence introduced in Subsection C.4, we bound the components of the single-round
regret in Subsections C.5 through C.8. Finally, we discuss bounding the minimum fraction term β
in Subsection C.9, particularly when the observed arm contexts lie nearly within a low-dimensional
subspace of the RKHS induced by NTK.

C.1
Theoretical Results with NTK Gram Matrices

We will first introduce some results, in order to link the NTK matrices (Def. 5.1 and Def. 5.2) with
the reward mapping function h(·) and the gradient covariance matrix Σ (Algorithm 1).
Lemma C.1. With probability at least 1 −δ, if network width m satisfies the condition in Theorem
5.6, for any x ∈AT , there exists a set of parameters θ∗such that

h(x) = ⟨g(x; θ0), θ∗−θ0⟩
(C.1)

where parameters θ∗satisfy ∥θ∗−θ0∥≤S/√m, along with the NTK norm S ≥
p

2˘h
⊺˘H−1˘h.

Proof.
The proof of this lemma is inspired by that of Lemma 5.1 in [86]. However, we build
our proof upon the non-duplicate arms ˘
AT and the corresponding NTK Gram matrix ˘H (Def. 5.2),
instead of imposing the full-rank assumption on the conventional NTK matrix H (Def. 5.1). Here,
we recall that the matrix ˘H is naturally positive definite (˘λ0 = λmin( ˘H) > 0), as it is the NTK Gram
matrix built upon a set of distinct arms that are not parallel (Fact 5.4).

Then, consider the gradient matrix with no-duplicate arms ˘G = [g(x; θ0)]x∈˘
AT /√m ∈Rp×| ˘
AT |,
where p represents the total number of parameters in the neural network. As a result, by applying
conclusion from Lemma C.3 and due to the fact that | ˘
AT | ≤3T, with the network width m ≥
Ω(L6 log(TL/δ)/ϵ), ∀ϵ > 0 and the probability at least 1 −δ, we will have

∥˘G⊺˘G −˘H∥F ≤| ˘
AT | · ϵ.

By setting ϵ =
˘λ0
2| ˘
AT |, we will have

˘G⊺˘G ⪰˘H −∥˘G⊺˘G −˘H∥F I ⪰˘H −˘λ0/2I ⪰˘H/2 ≻0.

where the last two inequalities are due to Fact 5.4 that ˘H ⪰˘λ0I ≻0. Analogous to Lemma 5.1 in
[86], we consider the singular value decomposition of ˘G being ˘G = ˘P ˘A ˘Q⊺, where we naturally
have ˘A ≻0 since ˘H is positive definite. Then, with the expected reward vector ˘h (Def. 5.2), we have

˘h = ( ˘Q ˘A˘P⊺) · (˘P ˘A−1 ˘Q⊺) · ˘h = √m · ˘G⊺(θ∗−θ0),

22


---Page Break---
by considering there exists a set of parameters θ∗= θ0 + (˘P ˘A−1 ˘Q⊺) · ˘h/√m.

Therefore, since ˘h = √m · ˘G⊺(θ∗−θ0), we will have ∀x ∈˘
AT ,

h(x) = ⟨g(x; θ0), θ∗−θ0⟩.

Meanwhile, since ˘G⊺˘G ⪰˘H/2, by applying Lemma G.8, we will have the distance

m · ∥θ∗−θ0∥2
2 = h⊺˘Q ˘A−1 ˘P⊺· ˘P ˘A−1 ˘Q⊺h = h⊺( ˘G⊺˘G)−1h ≤2h⊺˘H−1h.

Finally, since the above results holds ∀x ∈˘
AT , due to the fact that ˘
AT contains all the unique
arms of the collection AT , we will directly have the above results regarding parameters θ∗feasible
∀x ∈AT . This completes the proof.

Lemma C.2. Suppose m satisfies the conditions in Theorem 5.6. Suppose the gradient matrix with
randomly initialized parameters is Σ(0) = λI + P

x∈A g(x; θ0) · g(x; θ0)⊺/m, upon an arbitrary
subset A ⊆AT of arm collection AT . With probability at least 1 −δ over the initialization, the
result holds:

log
det Σ(0)

det λI


≤ed log(1 + TK/λ) + 1.

Proof. First, recall that we have AT as the arm collection of: (i) the chosen arms {xt}T
t=1; (ii) the
optimal arms {x∗
t }T
t=1; (iii) and the imaginary ones {ext}T
t=1 chosen by the corruption-free model.
This makes its cardinality |AT | = 3T. Thus, for the left hand side, we have

log det(Σ(0))

det(λI)
≤log det(λI +
X

x∈AT
g(x; θ0)g(x; θ0)⊺/m) = det(λI + G0G⊺
0),

where we define gradient matrix G0 =

g(x; θ0)/√m


x∈AT ∈Rp×(3T ) based on arm collection
AT . Here, based on Lemma C.3, we can bound the distance between the Gradient matrix product
G⊺
0G0 and the NTK matrix H (Def. 5.1), as

∥G⊺
0G0 −H∥≤3T ·
1
3T · O(
√

T/λ)
=
1
O(
√

T/λ)
,

by setting ϵ =
1
3T ·O(
√

T /λ). The above results will hold, as long as we have the network width

m ≥Ω((TL)6 log(TL/δ)/λ4), matching the conditions in Theorem 5.6. As a result, we can have

log det(I + G⊺
0G0/λ)

= log det(I + H/λ + (G⊺
0G0 −H)/λ)

≤log det(I + H/λ) + ⟨(I + H/λ)−1, (G⊺
0G0 −H)/λ⟩

≤log det(I + H/λ) + ∥(I + H/λ)−1∥F ∥G⊺
0G0 −H∥F /λ

≤log det(I + H/λ) + O(
√

T/λ) · ∥G⊺
0G0 −H∥F
≤log det(I + H/λ) + 1

= ed log(1 + TK/λ) + 1.

The first inequality is because the concavity of log det(·) function; The third inequality is due to
∥(I + Hλ)−1∥F ≤∥I−1∥F ≤
√

T; The fourth inequality is by applying the above distance upper
bound ∥G⊺
0G0 −H∥≤
1
O(
√

T /λ). The last inequality is because of the choice the m; The last

equality is because of the Definition of ed. The proof is completed.

Lemma C.3. With the randomly initialized network parameters θ0 ∈Rp and a collection of arms
A ⊂Rd, define the gradient matrix GA = [g(x; θ0)]x∈A ∈Rp×|A|. Following the recursive
procedure in Def. 5.1 (7), construct the NTK Gram matrix HA based on arms A. Then, with the
probability at least 1 −δ, we will have

∥G⊺
AGA −HA∥F ≤|A| · ϵ,

with the network width m ≥Ω(L6 log(|A|L/δ)/ϵ4).

23


---Page Break---
Proof. The proof of this lemma is analogous to the proof of Lemma B.1 in [86]. Based on Theorem 3.1
from [4], we have that for any two arms x, x′ ∈A, as the network width m ≥Ω(L6 log(L/δ)/ϵ4),
we will have |⟨g(x; θ0), g(x′; θ0)⟩/m −HA[x, x′]| ≤ϵ, where HA[x, x′] represents the element
in NTK Gram matrix HA that corresponds to arms x and x′.

Next, taking the union bound over all the arms in A, we will have

∥G⊺
AGA −HA∥F =
sX

x∈A

X

x′∈A
|⟨g(x; θ0), g(x′; θ0)⟩/m −HA[x, x′]|2 ≤|A| · ϵ,

as long as the network width m ≥Ω(L6 log(|A|L/δ)/ϵ4).

C.2
Bounding single-round regret

With the above results linking the NTK to the neural model, we proceed to bound the single-round
regret Rt for t ∈[T]. This single-round regret will then be aggregated to obtain the cumulative regret
R(T) = P

t∈[T ] Rt.

Based on Lemma C.1, we have the expected reward of an arm x ∈Xt being

E[r|x] = h(x) = ⟨g(x; θ0), θ∗−θ0⟩

where there exist parameters θ∗such that ∥θ∗−θ0∥≤S/√m. Meanwhile, apart from the trained
parameters θt−1 based on chosen arms as well as the corresponding received rewards {xτ, rτ}τ∈[t−1],
we also denote the imaginary corruption-free parameters eθt−1, which is trained with the chosen arms
along with their unknown corruption-free rewards {xτ, erτ}τ∈[t−1], for the sake of analysis.

C.2.1
Decomposing the single-round regret

To bound the single-round regret Rt, we first decompose the objective into several individual terms,
and then bound them individually. For reference, the parameters θt−1, covariance matrix Σt−1,
confidence ellipsoid Ct−1, and weights wt pertain to the chosen arm xt, with the arm index i ∈[K]
omitted for simplicity of notation.

Arm selection scores. First, recall that based on the arm pulling mechanism (line 13, Algorithm 1)
and the benefit score (6) (Lemma C.11), the chosen arm xt ∈Xt is selected by

xt = arg max
xi,t∈Xt


f(xi,t; θi,t−1) + γi,t−1 ·
q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1)/m


= arg max
xi,t∈Xt U(xi,t),

where we denote the corresponding score shorthand as

U(xi,t) = f(xi,t; θi,t−1) + γi,t−1 ·
q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1).
(C.2)

Analogously, with eθi,t−1 being the parameters trained on same set of chosen arms and the corre-
sponding corruption-free rewards, we denote

eU(xi,t) = f(xi,t; eθi,t−1) + eγi,t−1 ·
q

g(xi,t; eθi,t−1)⊺eΣ
−1
i,t−1g(xi,t; eθi,t−1),
(C.3)

with corresponding covariance matrix eΣi,t−1 = λI+P

τ∈[t−1] w(τ)
i,t ·g(xτ; eθτ−1)g(xτ; eθτ−1)⊺/m,

and the coefficient eγi,t−1 based on eΣi,t−1 following the definition from (6). It is obvious that the
arm selection depends on the trained network parameters, and thus if the corruption makes the makes
the trained network parameters θi,t−1 deviate from the corruption-free ones eθi,t−1, it will lead to
discrepancy in terms of arm selection decisions.

Alternative forms of selection scores. On the other hand, to maintain the consistency with the form
of (C.1) in Lemma C.1, we can consider an alternative form of arm selection being

V (xi,t) =

g(xi,t; θ0), θi,t−1 −θ0

+ γi,t−1 ·
q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1)/m

=
max
θ∈Ci,t−1


g(xi,t; θ0), θ −θ0

(C.4)

24


---Page Break---
where we have Ci,t−1 := {θ : ∥θ −θi,t−1∥Σi,t−1 ≤γi,t−1/√m, γi,t−1 > 0} being the confi-
dence ellipsoid of for the actual trained parameters θi,t−1, trained with possibly corrupted records.
γi,t−1 > 0 represents the radius of the confidence ellipsoid; and the last equality in (C.4) is due
to maxx:∥x−b∥A≤c⟨a, x⟩= ⟨a, b⟩+ c ·
√

a⊺A−1a [86]. Analogously, for the corruption-free
parameters eθi,t−1, we also define its confidence ellipsoid eCi,t−1 := {θ : ∥θ −eθi,t−1∥eΣi,t−1 ≤
eγi,t−1/√m, eγi,t−1 > 0}, with eγi,t−1 > 0 being the radius of the confidence ellipsoid, and by Lemma
C.10, we will have θ∗∈eCi,t−1. As a result, we can also formulate an alternative form for arm
selection as

eV (xi,t) =

g(xi,t; θ0), eθi,t−1 −θ0

+ eγi,t−1 ·
q

g(xi,t; eθi,t−1)⊺eΣ
−1
i,t−1g(xi,t; eθi,t−1)/m

=
max
eθ∈e
Ci,t−1


g(xi,t; θ0), eθ −θ0

.
(C.5)

Here, the radius of the confidence ellipsoid for the corruption-free parameters eγi,t−1, is provided in
Lemma C.10, ensuring that θ∗∈eCi,t−1. On the other hand, deriving the radius for the trained model,
γi,t−1 in face of potential corruptions, is considerably more challenging. With our carefully designed
arm weights in (4), we manage to establish the updated confidence ellipsoid in Lemma C.12, along
with the corresponding UCB for reward estimation (Lemma C.11).

Note that our UCB-based exploration score (6) is also motivated by Lemma C.11. For a specific arm
xi,t ∈Xt, our UCB-type exploration score includes only terms related to xi,t, omitting constant
and arm-invariant parts. This approach is intuitive, as arm-independent terms do not influence arm
selection (line 13, Algorithm 1), and they will remain the same across candidate arms Xt. Moreover,
with a sufficiently large network width m as in Theorem 5.6, a majority of these terms can be further
reduced to O(1).

Decomposing the single-round objective. Up to the time step t ∈[T], denoting xt, x∗
t ∈Xt being
the chosen arm and the optimal arm in each round respectively, we will have the corresponding regret
bound for step t as:

Rt = min

h(x∗
t ) −h(xt), 1


= min

h(x∗
t ) −eV (xt) + eV (xt) −h(xt)
|
{z
}
]
UCBt(xt)

, 1


= min

]
UCBt(xt) + h(x∗
t ) −V (xt)
|
{z
}
IR1

+ V (xt) −eV (xt)
|
{z
}
IR2

, 1

(C.6)

where the first equality is because we have the expected rewards in bounded value range [0, 1]. Here,
with the three terms after decomposing the single-round regret Rt, we can bound the first term
]
UCBt(xt) with Lemma C.9, while bounding the two error terms IR1 and IR2 with Lemma C.6 and
Lemma C.4 respectively.

C.3
Bounding cumulative regret R(T) (Proof of Theorem 5.6)

First, we would like to mention that since both the expected corrupted reward E[r] and the expected
corruption-free reward E[er] fall within the range [0, 1], as defined in (1), it follows that C ≤T. In
this case, as long as the network width requirement m ≥Ω(poly(T)) in Theorem 5.6 is adequately
satisfied, this naturally implies m ≥Ω(poly(C, T)) with a sufficiently large network width m. This
also indicates that knowledge of C is not mandatory for setting m before the online learning process
begins. The above clarification will also be used to establish the final regret bound.

Then, with the derived results in terms of single-round regret Rt, t
∈
[T], we can then
proceed to bound the cumulative regret over T rounds.
By definition in (C.6), we have

R(T) = PT
t=1 min

]
UCBt(xt)+h(x∗
t ) −V (xt)
|
{z
}
IR1

+ V (xt) −eV (xt)
|
{z
}
IR2

, 1

. Recall that the first term

]
UCBt(xt) can be bounded with Lemma C.9, while we bound the other two error terms IR1 and IR2

25


---Page Break---
with Lemma C.6 and Lemma C.4 respectively. Next, with the conclusions from above lemmas, we
will have

R(T) =

T
X

t=1
min

]
UCBt(xt) + h(x∗
t ) −V (xt)
|
{z
}
IR1

+ V (xt) −eV (xt)
|
{z
}
IR2

, 1


≤

T
X

t=1
min

2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1

+ O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1
+ O(
√

λS) ·
g(ext; θ˜it,t−1)/√m

(Σ′
t−1)−1

+ O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1
+ O
 √

λS

·
g(xt; θt−1)/√m

( ¯Σt−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(m−1/6p

log(m)t1/6λ−7/6L2/7) + O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γt−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2) + O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)), 1


where "weight-free" covariance matrices are ¯Σt−1 = λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m,
and Σ′
t−1 = λI + P
τ∈[t−1] g(exτ; θ˜iτ ,τ−1)g(exτ; θ˜iτ ,τ−1)⊺/m. Afterwards, with sufficient network
width m that satisfies the conditions in Theorem 5.6, the majority of the terms on the RHS of (C.7),
which contain m to the negative order, can be reduced to O(1). Thus, we will then have

R(T) ≤O(1) +

T
X

t=1
min

2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1

+ O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(
√

λS)
g(ext; θ˜it,t−1)/√m

(Σ′
t−1)−1

+ O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O
 √

λS

·
g(xt; θt−1)/√m

( ¯Σt−1)−1, 1


≤O(1) +

T
X

t=1
min

2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1 1


+ O(αC) ·

T
X

t=1
min
g(xt; θt−1)/√m


2

( ¯Σt−1)−1, 1


+ O(
√

λS)

v
u
u
tT

T
X

t=1
min
g(ext; θ˜it,t−1)/√m


2

(Σ′
t−1)−1
, 1


+ O
 √

λS

·

v
u
u
tT

T
X

t=1
min
g(xt; θt−1)/√m


2

( ¯Σt−1)−1, 1


(C.7)

where the second inequality is by applying the triangular inequality.
Then, denote wmin =
min[{w(τ)
t
, w(τ)
˜it,t}t∈[T ],τ∈[t−1]]. Analogously, we also have the round-wise minimum weight value

being wmin
t
= min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T], which will be used to determine our

scaling parameter α. With our notation from Theorem 5.6, this intuitively leads to α ≤κ2

β . As a

26


---Page Break---
result, we can therefore have

R(T) ≤O(1) +

T
X

t=1
min

2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1, 1


+ O
 
Cβ−1κ2
· log det( ¯ΣT )
det(λI) + O
 √

λS

·
√

T log det( ¯ΣT )

det(λI) + O(
√

λS) ·
√

T log det(Σ′
T )
det(λI)

≤O(1) +

T
X

t=1
min

2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1, 1


+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O
 √

λS

·
q

T ed log(1 + TK/λ)

≤O(1) +

T
X

t=1
min

2 eγt−1
√wmin
· ∥√wming(xt; eθt−1)/√m∥eΣ−1
t−1, 1


+

T
X

t=1
min
 γt−1
√wmin
· ∥√wming(xt; θt−1)/√m∥Σ−1
t−1, 1


+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O
 √

λS

·
q

T ed log(1 + TK/λ)
(C.8)

where the first inequality is because the auxiliary matrices ¯Σ(0)
t−1 and Σ′
t−1 do not involve arm weights,
thus we can directly applying the Lemma G.7 and Lemma G.6. The second inequality is by applying
Lemma C.2. Meanwhile, since Σ′
t−1 is not defined w.r.t. randomly initialized θ0, we additional apply
the Lemma G.6 in terms of the matrix determinant difference to derive the results, where the extra
term will also be reduced to O(1) with sufficiently large m. Afterwards, for the rest of the summation
terms in (C.8), we can have

R(T) ≤O(
eγT
√wmin
)

v
u
u
tT

T
X

t=1
min

∥√wming(xt; eθt−1)/√m∥2
eΣ−1
t−1, 1


+ O(
γT
√wmin
)

v
u
u
tT

T
X

t=1
min

∥√wming(xt; θt−1)/√m∥2
Σ−1
t−1, 1


+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O
 √

λS

·
q

T ed log(1 + TK/λ) + O(1)

≤
1
√wmin
O

ν

s

log det(eΣT )

det(λI) −2 log(δ) + λ1/2S + ν

s

log det(ΣT )

det(λI) −2 log(δ) + λ1/2S


·
q

T ed log(1 + TK/λ)

+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O
 √

λS

·
q

T ed log(1 + TK/λ) + O(1)

≤
1
√wmin
O

λ1/2S + ν

s

log det(eΣ(0)
T )
det(λI) + O(m−1/6p

log(m)L4T 5/3λ−1/6) −2 log(δ)

+ ν

s

log det(Σ(0)
T )
det(λI) + O(m−1/6p

log(m)L4T 5/3λ−1/6) −2 log(δ)
q

T ed log(1 + TK/λ)

+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O
 √

λS

·
q

T ed log(1 + TK/λ) + O(1),
(C.9)

where the second inequality is by applying Lemma G.7 and Lemma C.2, as well as the definition
of coefficients γT , eγT , along with the fact that γT ≥γt, eγT ≥eγt, t ∈[T]. The last inequality is
by applying Lemma G.6. With sufficiently large network width m as in Theorem 5.6, we can have
O(m−1/6p

log(m)L4T 5/3λ−1/6) ≤O(1). Then, due to the fact that ¯Σ(0)
T
⪰eΣ(0)
T , Σ(0)
T , applying

27


---Page Break---
the Lemma C.2, it will then lead to

R(T) ≤
1
√wmin
O

ν
q

ed log(1 + TK/λ) −2 log(δ) + λ1/2S
q

T ed log(1 + TK/λ)

+ O
 
Cβ−1κ2
· ed log(1 + TK/λ) + O(1)

≤O

ν
q

ed log(1 + TK/λ) −2 log(δ) + λ1/2S
q

T ed log(1 + TK/λ)/κ2

+ O
 
Cβ−1 edκ2 log(1 + TK/λ)


(C.10)

where the second inequality is by setting the tunable parameter α in each round accordingly, in
order to ensure wmin = κ2 < 1. Here, we remind that our regret analysis does not require
the learner to know the minimum fraction value β before the online learning process, where
β = mint∈[T ],τ∈[t−1]

min{fracτ(xt; Xt, ¯Σt−1), fracτ(ext; Xt, ¯Σt−1)}

. As we have mentioned,
in practice, the learner can scale the α values in each round to make sure the round-wise minimum
weight value wmin
t
= min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T] (Subsec. B.5).

C.4
Auxiliary sequences: Regression parameters and gradient descent parameters

To bridge neural models with NTK regression, we have two different routes to decouple the effects
of adversarial corruptions from the received arm rewards. First, we define the gradient-based ridge
regression parameters specific to a candidate arm xi,t ∈Xt in round t ∈[T], as

Σ(0)
i,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t g(xτ; θ0) · g(xτ; θ0)⊺/m,

Σi,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t g(xτ; θτ−1) · g(xτ; θτ−1)⊺/m,

eΣi,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t g(xτ; eθτ−1) · g(xτ; eθτ−1)⊺/m,

b(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
i,t g(xτ; θ0) · rτ/√m,
bi,t−1 =
X

τ∈[t−1]
w(τ)
i,t g(xτ; θτ−1) · rτ/√m,

eb
(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
t
g(xτ; θ0) · erτ/√m,
ebi,t−1 =
X

τ∈[t−1]
w(τ)
i,t g(xτ; θτ−1) · erτ/√m,

(C.11)

where {xτ, rτ}, τ ∈[t] respectively stand for the chosen arms as well as their rewards, while
{xτ, erτ}, τ ∈[t] refer to chosen arms and their imaginary corruption-free rewards. For notation
simplicity, we use w(τ)
t
, τ ∈[t −1] to denote the arm weights for xt.

Meanwhile, given an candidate arm xi,t ∈Xt with arm weight w(τ)
i,t defined in (4), we can try
to bound the I2 term by decomposing the adversarial corruptions with a series auxiliary gradient
sequences {θ0, Θ(1), . . . , Θ(J)} as in Lemma D.1, such that for j-th iteration

Θ(j+1) = Θ(j) −η ·

J(0) · W ·
 
[J(0)]⊺(Θ(j) −θ0) −y

+ mλ(Θ(j) −θ0)


as well as an analogous sequence for corruption-free parameters

eΘ
(j+1) = eΘ
(j) −η ·

J(0) · W ·
 
[J(0)]⊺( eΘ
(j) −θ0) −ey

+ mλ( eΘ
(j) −θ0)


where J(0) :=
 
g(x1; θ0), g(x2; θ0), . . . , g(xt−1; θ0)

∈Rp×(t−1), and W refers to the diagonal
matrix of arm weights {w(τ)
i,t }τ∈[t−1], along with the reward vectors y, ey ∈Rt−1 separately being
the vector of received rewards and corruption-free rewards. In particular, with [J(0)]τ being the τ-th
column of matrix J(0), the auxiliary sequence Θ(j) can be deemed as applying Gradient Descent to

28


---Page Break---
solve the following optimization problem

min
Θ L(Θ) =
X

τ∈[t−1]

1
2 · w(τ)
i,t ·
[J(0)]⊺
τ(Θ −θ0) −yτ



2

2
+ 1

2 · mλ ·
Θ −θ0



2

2
(C.12)

Analogously, we can also derive the optimization problem for the sequence of corruption-free

auxiliary parameters eΘ
(j), by applying the same definition of weight matrix W. Since the arm
weights w ≤1 by definition, we will also have the diagonal matrix norm ∥W∥2 ≤1.

Notation simplicity. For reference, we remind that the parameters θt−1, covariance matrix Σt−1,
confidence ellipsoid Ct−1, and weights wt pertain to the chosen arm xt, with the arm index i ∈[K]
omitted for simplicity of notation. Meanwhile, the gradinet covariance matrix Σi,t−1, confidence
ellipsoid Ci,t−1, and weights wi,t are associated with each candidate arm xi,t for i ∈[K].

C.5
Bounding the error term IR2 in (C.6)

Recall that to derive the upper bound for the single-round regret Rt, we need to respectively bound
the three error terms on the RHS of (C.6). Here, we first bound term IR2 with the following Lemma
C.4.
Lemma C.4. Suppose the imaginary neural network f(·; eθt−1) in round t ∈[T] has been trained on
corruption-free rewards {xτ, erτ}τ∈[t−1]. Meanwhile, f(·) is an L-layer FC network with width m.
Suppose we have m, J, η satisfying the conditions in Theorem 5.6. Then, for the chosen arm xt ∈Xt,
with the probability at least 1 −δ, we will have

IR2 =V (xt) −eV (xt)

≤γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1 + O(m−1/6p

log(m)t1/6λ−7/6L2/7)

+ O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1
+ O(
√

λS)
g(xt; θt−1)/√m

( ¯Σt−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4)

+ O(m−1/6p

log(m)t2/3λ−2/3L7/2).

By definition, we have γt−1 = O
 
ν
q

log det(Σt−1)

det(λI)
−2 log(δ) + λ1/2S

, as well as the gradient

covariance matrix ¯Σt−1 = λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m. The minimum round-wise

arm weight wmin
t
= min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T] by scaling parameter α.

Proof. For the error term IR2, we have

IR2 = V (xt) −eV (xt)

= max
θ∈Ct−1⟨g(xt; θ0), θ −θ0⟩−max
eθ∈e
Ct−1
⟨g(xt; θ0), eθ −θ0⟩

≤max
θ∈Ct−1⟨g(xt; θ0), θ −θ0⟩−⟨g(xt; θ0), eθt−1 −θ0⟩

≤max
θ∈Ct−1⟨g(xt; θ0), θ −θ0⟩−⟨g(xt; θt−1), eθt−1 −θ0⟩+ O(m−1/6p

log(m)t2/3λ−2/3L7/2)

= max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩−⟨g(xt; θt−1), eθt−1 −θt−1⟩+ O(m−1/6p

log(m)t2/3λ−2/3L7/2)

= max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩
|
{z
}
Projection difference

+| ⟨g(xt; θt−1), eθt−1 −θt−1⟩
|
{z
}
I2

| + O(m−1/6p

log(m)t2/3λ−2/3L7/2)

(C.13)

where the first inequality is because of the definition of confidence ellipsoids eCt−1 and Ct−1. The
second inequality is by applying Lemma G.3 and Lemma G.4. Here, we have the first term on

29


---Page Break---
the RHS is bounded by Corollary C.5, where this term is used to represent the gradient projection
difference, between the confidence ellipsoid center parameters θt−1 and the other parameters in this
confidence ellipsoid θ ∈Ct−1. Meanwhile, term I2 will be bounded by Corollary C.8.

Corollary C.5. Suppose the imaginary neural network f(·; eθt−1) in round t ∈[T] has been trained
on corruption-free rewards {xτ, erτ}τ∈[t−1]. Meanwhile, f(·) is an L-layer FC network with width
m. Suppose we have m, J, η satisfying the conditions in Theorem 5.6. Then, for the chosen arm
xt ∈Xt, with the probability at least 1 −δ, we will have

max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩≤γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1 + O(m−1/6p

log(m)t1/6λ−7/6L2/7),

and the corresponding summation value being
X

t∈[T ]
min

max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩, 1
	

≤
1
p

wmin
t
O

ν

s

log det(ΣT )

det(λI) −2 log(δ) + λ1/2S

·

s

2T · log det(ΣT )

det(λI) .

+ O(Sm−1/6p

log(m)T 7/6λ−1/6L2/7) + O(m−1/6p

log(m)T 7/6λ−7/6L2/7).

By definition, we have the radius of Ct−1 being γt−1 = O
 
ν ·
q

log det(Σt−1)

det(λI)
−2 log(δ) + λ1/2S

,

and the gradient covariance matrix Σt−1 = λI + P

τ∈[t−1] w(τ)
t
· g(xτ; θτ−1)g(xτ; θτ−1)⊺/m.

Proof. The proof of this corollary follows an analogous approach as Lemma C.9. By definition, we
have the gradient inner product for the chosen arm xt in round t as

max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩

≤max
θ∈Ct−1 ∥θ −θt−1∥Σt−1 · ∥g(xt; θ0)∥Σ−1
t−1

≤γt−1 · ∥g(xt; θt−1)/√m∥Σ−1
t−1 + O(m−1/6p

log(m)t1/6λ−7/6L2/7)

(C.14)

where the first inequality is due to Holder’s inequality. The second inequality is by applying the
definition of confidence ellipsoid Ct−1, as well as Lemma G.4 and Lemma G.6. Then, similarly,
we will also need to bound the summation over T rounds. Following an analogous procedure as in
Lemma C.9, we will have
X

t∈[T ]
min{ max
θ∈Ct−1⟨g(xt; θ0), θ −θt−1⟩, 1}

≤
1
p

wmin
t
O

ν

s

log det(ΣT )

det(λI) −2 log(δ) + λ1/2S

·

s

2T · log det(ΣT )

det(λI) .

+ O(Sm−1/6p

log(m)T 7/6λ−1/6L2/7) + O(m−1/6p

log(m)T 7/6λ−7/6L2/7).

By the definition β = mint∈[T ],τ∈[t−1][min{fracτ(xt; Xt, ¯Σt−1), fracτ(ext; Xt, ¯Σt−1)}], we have
the lower bound of arm weights being wmin
t
≥α · β. Note that we also scale the α parameter to
ensure wmin
t
= κ2 as indicated in the Subsec. B.5, which requires no prior knowledge for β.

C.6
Bounding the error term IR1 in (C.6)

In this subsection, we bound the error term IR1 in (C.6), with the following Lemma C.6. Meanwhile,
we denote an extra weight-free gradient covariance matrix for the chosen arms {exτ}τ∈[t] of the
corruption-free model. The corresponding arm index is denoted as ˜it, t ∈[T], such that x˜it,t = ext.

Lemma C.6. Suppose the imaginary neural network f(·; eθt−1) in round t ∈[T] has been trained on
corruption-free rewards {xτ, erτ}τ∈[t−1]. Meanwhile, f(·) is an L-layer FC network with width m.

30


---Page Break---
Suppose we have m, J, η satisfying the conditions in Theorem 5.6. Then, for the chosen arm xt ∈Xt,
with the probability at least 1 −δ, we will have
IR1 = h(x∗
t ) −V (xt)

≤O(αC)
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(
√

λS) ·
g(ext; θ˜it,t−1)/√m

(Σ′
t−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γ˜it,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2).

By definition, we have the coefficient γ˜it,t−1 = O
 
ν
q

log
det(Σ˜it,t−1)

det(λI)
−2 log(δ) + λ1/2S

, as well
as the gradient covariance matrices Σ′
t−1 = λI + P

τ∈[t−1] g(exτ; θ˜iτ ,τ−1)g(exτ; θ˜iτ ,τ−1)⊺/m,
and ¯Σt−1 = λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m. The minimum round-wise arm weight

wmin
t
= min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T] by scaling parameter α.

Proof. By definition, we can have IR1 = h(x∗
t ) −U(xt) + U(xt) −V (xt). Here, in terms of the
distance between U(xi,t) and V (xi,t) given a candidate arm xi,t ∈Xt, we have

|U(xi,t) −V (xi,t)| = |f(xi,t; θi,t−1) −

g(xi,t; θ0), θi,t−1 −θ0

|

≤|f(xi,t; θi,t−1) −

g(xi,t; θi,t−1), θi,t−1 −θ0

|

+ |

g(xi,t; θi,t−1), θi,t−1 −θ0

−

g(xi,t; θ0), θi,t−1 −θ0

|

≤O(m−1/6p

log(m)t2/3λ−2/3L3) + O(m−1/6t2/3λ−2/3p

log(m)L7/2).
where the first inequality is by triangular inequality, and the second inequality is by applying a similar
approach as in (B.12) from [86] as well as Lemma G.4. Next, we proceed to bound term IR1. Here,
for a candidate arm xi,t ∈Xt and the associated corruption-free confidence ellipsoid eCi,t−1, we have
θ∗∈eCi,t−1 based on Lemma C.10. Thus, with ext = arg maxxi,t∈Xt eV (xi,t) being the arm chosen
by the corruption-free neural model, with the highest score eV (ext), there will be
IR1 = h(x∗
t ) −U(xt) + U(xt) −V (xt)

= ⟨g(x∗
t ; θ0), θ∗−θ0⟩−U(xt) + U(xt) −V (xt)

≤
max
eθ∈e
C˜it,t−1
⟨g(x∗
t ; θ0), eθ −θ0⟩−U(xt) + U(xt) −V (xt)

≤
max
eθ∈e
C˜it,t−1
⟨g(ext; θ0), eθ −θ0⟩−U(xt) + |U(xt) −V (xt)|,

where eC˜it,t−1 refers to the confidence ellipsoid of corruption-free parameters eθ˜it,t−1, associated to
arm ext ∈Xt. Afterwards, it further leads to

IR1 ≤
max
eθ∈e
C˜it,t−1
⟨g(ext; θ0), eθ −θ0⟩−U(xt)

+ O(m−1/6p

log(m)t2/3λ−2/3L3) + O(m−1/6t2/3λ−2/3p

log(m)L7/2)

≤
max
eθ∈e
C˜it,t−1
⟨g(ext; θ0), eθ −θ0⟩−V (xt)

+ O(m−1/6p

log(m)t2/3λ−2/3L3) + O(m−1/6t2/3λ−2/3p

log(m)L7/2)

≤
max
eθ∈e
C˜it,t−1
⟨g(ext; θ0), eθ −θ0⟩−V (ext)

+ O(m−1/6p

log(m)t2/3λ−2/3L3) + O(m−1/6t2/3λ−2/3p

log(m)L7/2)

= eV (ext) −V (ext) + O(m−1/6p

log(m)t2/3λ−2/3L3) + O(m−1/6t2/3λ−2/3p

log(m)L7/2)

31


---Page Break---
To bound the output difference between V (xi,t) and eV (xi,t) for an arm xi,t, we can decompose
them into separate terms. Recall the definition of V (·) in (C.4), and we can also define the analogous
eV (·) by applying the corruption-free parameters eθ and covariance matrix eΣ. Therefore, for arm
xi,t ∈Xt, we have

V (xi,t) −eV (xi,t) ≤|V (xi,t) −V (0)(xi,t)| + |V (0)(xi,t) −eV (0)(xi,t)| + |eV (0)(xi,t) −eV (xi,t)|,
(C.15)

where for the sake of analysis, we define two variants with randomly initialized parameters θ0

being: V (0)(xi,t) =

g(xi,t; θ0), θt−1 −θ0

+ γt−1 ·
q

g(xi,t; θ0)⊺(Σ(0)
t−1)−1g(xi,t; θ0)/m, and

eV (0)(xi,t) =

g(xi,t; θ0), eθt−1 −θ0

+ eγt−1 ·
q

g(xi,t; θ0)⊺(Σ(0)
t−1)−1g(xi,t; θ0)/m. With this
result, our objective then is to derive the upper bounds for the three terms on the RHS of (C.15).

Bounding the second term |V (0)(xi,t) −eV (0)(xi,t)| in (C.15). For the second term, we have

|V (0)(xi,t) −eV (0)(xi,t)|

≤|

g(xi,t; θ0), θi,t−1 −eθi,t−1

|

+ |γi,t−1 ·
q

g(xi,t; θ0)⊺(Σ(0)
i,t−1)−1g(xi,t; θ0)/m −eγi,t−1 ·
q

g(xi,t; θ0)⊺(Σ(0)
i,t−1)−1g(xi,t; θ0)/m|

= |

g(xi,t; θ0), θi,t−1 −eθi,t−1

| + |γi,t−1 −eγi,t−1| ·
q

g(xi,t; θ0)⊺(Σ(0)
i,t−1)−1g(xi,t; θ0)/m

≤|

g(xi,t; θ0), θi,t−1 −eθi,t−1

| + |γi,t−1 −eγi,t−1| · O(L/
√

λ).

Due to the fact that |√a −
√

b| ≤
p

|a −b|, based on the definition of γi,t−1 and eγi,t−1, we will have

|γi,t−1 −eγi,t−1| ≤ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6)

·

v
u
u
t

log(det Σi,t−1

det λI
) −log(det Σ0

det λI ) + log(det Σ0

det λI ) −log(det eΣi,t−1

det λI
)



≤ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12),

based on Lemma G.6. Therefore, we will end up with

|V (0)(xi,t) −eV (0)(xi,t)| ≤|

g(xi,t; θ0), θi,t−1 −eθi,t−1

|

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12).

Bounding Term |V (xi,t) −V (0)(xi,t)| and term |eV (0)(xi,t) −eV (xi,t)| in (C.15). On the other
hand, for the first term on the RHS, |V (xi,t) −V (0)(xi,t)|, we can bound this difference term by

V (xi,t) −V (0)(xi,t)

= γi,t−1
√m

q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1) −
q

g(xi,t; θ0)⊺(Σ(0)
i,t−1)−1g(xi,t; θ0)


= γi,t−1
√m

g(xi,t; θi,t−1)

Σ−1
i,t−1 −
g(xi,t; θ0)

(Σ(0)
i,t−1)−1



≤γi,t−1
√m

g(xi,t; θi,t−1)

Σ−1
i,t−1 −
g(xi,t; θi,t−1)

(Σ(0)
i,t−1)−1 +
g(xi,t; θ0) −g(xi,t; θi,t−1)

(Σ(0)
i,t−1)−1



≤O(m−1/6t1/6λ−7/6p

log(m)L7/2) + γi,t−1
√m ·
g(xi,t; θi,t−1)

Σ−1
i,t−1 −
g(xi,t; θi,t−1)

(Σ(0)
i,t−1)−1



32


---Page Break---
≤O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γi,t−1
√m ·
q

g(xi,t; θi,t−1)⊺Σ−1
i,t−1g(xi,t; θi,t−1) −g(xi,t; θi,t−1)⊺(Σ(0)
i,t−1)−1g(xi,t; θi,t−1)


≤O(m−1/6t1/6λ−7/6p

log(m)L7/2) + γi,t−1
√m ·
q

⟨g(xi,t; θi,t−1), (Σ−1
i,t−1 −(Σ(0)
i,t−1)−1)g(xi,t; θi,t−1)⟩


= O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γi,t−1
√m ·
s
g(xi,t; θi,t−1),

Σ−1
i,t−1 · (Σ(0)
i,t−1 −Σi,t−1) · (Σ(0)
i,t−1)−1

· g(xi,t; θi,t−1)


≤O(m−1/6t1/6λ−7/6p

log(m)L7/2) + γi,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2)
where the first inequality is because of the triangular inequality. The second inequality is by applying
Lemma G.4. The third inequality is again the application of triangular inequality, and the last
inequality is the application of Lemma G.6. Since the similar procedure can also be applied to bound
the third term |eV (0)(xi,t) −eV (xi,t)|, after summing up the results, we will have

|V (xi,t) −V (0)(xi,t)|, |eV (0)(xi,t) −eV (xi,t)|

≤O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γi,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2).

Summing up results. Combining the results, we finally have the upper bound for IR1 being

IR1 ≤|

g(ext; θ0), θ˜it,t−1 −eθ˜it,t−1

|

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γ˜it,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2)

≤|

g(ext; θ˜it,t−1), θ˜it,t−1 −eθ˜it,t−1

|

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γ˜it,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2),

and therefore we can formulate our objective to bound as

IR1 ≤|

g(ext; θ˜it,t−1), θ˜it,t−1 −eθ˜it,t−1


|
{z
}
I1

|

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L3t5/6λ−7/12)

+ O(m−1/6t2/3λ−2/3p

log(m)L7/2) + O(m−1/6t1/6λ−7/6p

log(m)L7/2)

+ γ˜it,t−1 · O(m−1/12t7/12λ−13/12 log1/4(m)Lt/2),
(C.16)

where we recall that ν is the pre-defined exploration parameter that echoes the sub-Gaussian noise
variance proxy. Finally, using the upper bound for term I1 from Lemma C.7 will finish the proof.

C.7
Bounding the terms I1, I2

Here, we see that there are still two terms in Rt, t ∈[T] that need to be bounded, which are

I1 =

g(ext; θ˜it,t−1), θ˜it,t−1 −eθ˜it,t−1

,
I2 = ⟨g(xt; θt−1), eθt−1 −θt−1⟩,

33


---Page Break---
where we recall xt ∈Xt is the actual chosen arm that is selected by the corrupted model θt−1,
while we have ext ∈Xt being the imaginary arm chosen by the corruption-free model eθt−1. In this
subsection, the error term I1 will be bounded by Lemma C.7, while term I2 will be bounded by
Corollary C.8.

Recap of auxiliary parameter definitions. With definitions in Subsection C.4, we can have two
different alternatives to decouple the adversarial corruptions from arm rewards: (i) the gradient-based
regression parameters; and, (ii) the auxiliary sequence of gradient descent. For reference, recall that
we denote the a series of gradient-based regression parameters, specified to candidate arm xi,t ∈Xt,
as

Σ(0)
i,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · g(xτ; θ0)⊺/m,

Σi,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · g(xτ; θτ−1)⊺/m,

b(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · rτ/√m,
bi,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · rτ/√m,

eb
(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · erτ/√m,
ebi,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · erτ/√m,

where {xτ, rτ}, τ ∈[t] respectively stands for the chosen arms as well as their received rewards,
while {xτ, erτ}, τ ∈[t] refer to the imaginary corruption-free rewards.

We also recall that with the chosen arm xt ∈Xt in round t with arm weights w(τ)
t
, τ ∈[t−1], defined
in (4), we will have a series of auxiliary gradient sequences {θ0, Θ(1), . . . , Θ(J)} as in Lemma D.1,
such that for j-th, j ∈[J], iteration

Θ(j+1) = Θ(j) −η ·

J(0) · W
 
[J(0)]⊺(Θ(j) −θ0) −y

+ mλ(Θ(j) −θ0)


where the diagonal weight matrix is made up with the arm weights, as W = diag
 
[w(τ)
t
]τ∈[t−1]

∈
R(t−1)×(t−1). Similarly, we will also have the sequence definition for corruption-free parameters

eΘ
(j+1) = eΘ
(j) −η ·

J(0) ·W
 
[J(0)]⊺( eΘ
(j) −θ0)−ey

+mλ( eΘ
(j) −θ0)

, along with the Jacobian

matrix J(0) :=
 
g(x1; θ0), g(x2; θ0), . . . , g(xt−1; θ0)

∈Rp×(t−1). Here, we have y, ey ∈Rt−1
separately being the vector of received rewards and that of the imaginary corruption-free rewards.
In particular, with [J(0)]τ being the τ-th column of matrix J(0), the auxiliary sequence Θ(j) can be
deemed as applying Gradient Descent to solve the following optimization problem

min
Θ L(Θ) =
X

τ∈[t−1]

1
2 · w(τ)
i,t

[J(0)]⊺
τ(Θ −θ0) −yτ



2

2
+ 1

2 · mλ
Θ −θ0



2

2

Consequently, we can follow an analogous approach as in Lemma D.1 with the above optimization
problem, to bound the difference between gradient-based parameters and the auxiliary sequence.

C.7.1
Bounding the error term I1

We bound the error term using Lemma C.7. Following the notation in the main body, let θt−1
denote the trained parameters associated with the chosen arm xt, and let θ˜it,t−1 represent the trained
parameters of the arm ext = x˜it,t, t ∈[T], which corresponds to the chosen arm of the hypothetical

corruption-free model f(·; eθ). Additionally, we define a weight-free gradient covariance matrix for
the arm collection {exτ}τ∈[t] containing arms selected by the corruption-free model, which will be
denoted by Σ′
t−1 = λI + P
τ∈[t−1] g(exτ; θ˜iτ ,τ−1)g(exτ; θ˜iτ ,τ−1)⊺/m.

Lemma C.7. Suppose the imaginary corruption-free neural network f(·; eθi,t−1) for arm xi,t ∈
Xt has been trained on corruption-free rewards {xτ, erτ}τ∈[t−1], while the other trained network
f(·; θi,t−1) is trained on the received records {xτ, rτ}τ∈[t−1]. Suppose f(·) is an L-layer FC

34


---Page Break---
network with width m that satisfies the conditions in Theorem 5.6. Then, for the arms xt, ext ∈Xt,
with the probability at least 1 −δ, we will have

I1 =
⟨g(ext; θ˜it,t−1), θ˜it,t−1 −eθ˜it,t−1⟩


≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(
√

λS) ·
g(ext; θ˜it,t−1)/√m

(Σ′
t−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4).

where θt−1 are the parameters associated to chosen arm xt, and θ˜it,t−1 are those parameters of arm
ext. The gradient matrix is defined as Σ′
t−1 = λI + P

τ∈[t−1] g(exτ; θ˜iτ ,τ−1)g(exτ; θ˜iτ ,τ−1)⊺/m,
and we also have ¯Σt−1 = λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m.

Proof. Since the only difference between term I1 and term I2 is the gradients w.r.t. different arms,
we begin with term I1 = ⟨g(ext; θt−1), eθt−1 −θt−1⟩, and the results can be readily generalized
to term I2. Here, recall that we have ∥θ∗−eθt∥eΣt−1 ≤eγt/√m based on Lemma C.10, as well as

∥θt −θ0∥2, ∥eθt −θ0∥2 ≤O(
p

t/mλ) based on lemma G.3.

Simplifying the notation. For the following proof, for the sake of notation simplicity, we directly

use eΘ
(J), Θ(J) to respectively represent eΘ
(J)
˜it,t−1, Θ(J)
˜it,t−1, which are the gradient descent based
parameters associated with the arm ext. On the other hand, for the terms that belong to arm ext = x˜it,t,
we use θ˜it, Σ˜it and b˜it to separately represent θ˜it,t−1, Σ˜it,t−1 and b˜it,t−1 by omitted the subscript
of time step t −1 to simplify the notation.

Afterwards, it can further lead to

I1 =
⟨g(ext; θ˜it), θ˜it −eθ˜it⟩


≤
⟨g(ext; θ˜it), θ˜it −Θ(J)⟩
 +
⟨g(ext; θ˜it), eΘ
(J) −eθ˜it⟩
 +
⟨g(ext; θ˜it), eΘ
(J) −Θ(J)⟩

|
{z
}
I1.1

≤O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) +
⟨g(ext; θ˜it), Θ(J) −eΘ
(J)⟩

|
{z
}
I1.1

,

where the first inequality is due to triangular inequality, and the last inequality is by applying Lemma
C.4 in [86], with the optimization problem being (C.12) which bounds the difference between
gradient-based parameters and the auxiliary sequence. Here, term I1.1 measures the distance between
the auxiliary sequence trained with corrupted records, and that trained by imaginary corruption-free
records.

Next, recall that with randomly initialized network parameters θ0, we can formulate the least
square parameters as (Σ(0)
˜it )−1b(0)
˜it /√m, while the least square parameters trained by corruption-free

rewards are analogously denoted by (Σ(0)
˜it )−1eb
(0)
˜it /√m. In this case, the term I1.1 can be alternatively
transformed to

I1.1 =



g(ext; θ˜it), Θ(J) −eΘ
(J)

≤



g(ext; θ˜it), (Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m) −( eΘ
(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m)


≤



g(ext; θ˜it), Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m


+



g(ext; θ˜it), eΘ
(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m


35


---Page Break---
which further leads to

I1.1 ≤



g(ext; θ˜it), Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m


+



g(ext; θ˜it), eΘ
(J) −θ0 −(Σ(0)
˜it )−1eb
(0)
˜it /√m


+



g(ext; θ˜it), (Σ(0)
˜it )−1(
X

τ∈[t−1]
w(τ)
˜it,t · g(xτ; θ0) · cτ)/m


|
{z
}
I1.2

(C.17)

where the inequality is due to the definition of gradient-based regression parameters and triangular
inequality. It is obvious that for the third term on the RHS, the only thing we have control on
is θ˜it. When trying to bound the first two terms on the RHS for the arm ext, we can first apply
Holder’s inequality with Σ′
t−1 = λI + P
τ∈[t−1] g(exτ; θ˜iτ ,τ−1)g(exτ; θ˜iτ ,τ−1)⊺/m. Note that
for the purpose of analysis and different from previous ones defined in (C.11), the new matrix
(Σ′
t−1) contains gradients of the sequence of arms {exτ}τ∈[t−1] chosen by the corruption-free model
f(·; eθτ−1), τ ∈[t −1], with the parameters eθτ−1, τ ∈[t −1].

In this case, we use the Holder’s inequality to the first term on the RHS of (C.17) as

g(ext; θ˜it), Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m


≤
g(ext; θ˜it)/√m

(Σ′
t−1)−1 · √m ·
Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m

(Σ′
t−1)

≤
g(ext; θ˜it)/√m

(Σ′
t−1)−1 · √m ·
Θ(J) −θ0 −(Σ(0)
˜it )−1b(0)
˜it /√m

2 ·
(Σ′
t−1)

2

≤
g(ext; θ˜it)/√m

(Σ′
t−1)−1 · O(λ + tL) ·
√m(Θ(J) −θ0) −(Σ(0)
˜it )−1b(0)
˜it

2

≤
g(ext; θ˜it)/√m

(Σ′
t−1)−1 · O(
√

λ + tL)

· O

(1 −ηmλ)J/2p

t/λ + m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)


≤
g(ext; θ˜it)/√m

(Σ′
t−1)−1 · O(
√

λS)

where the first two inequalities are by the Holder’s inequality and Cauchy-Schwartz inequality.
The third inequality is by Lemma G.6. The fourth inequality is by applying Lemma D.1, with
the optimization problem being (C.12) bounding the difference between gradient-based parameters
and the auxiliary sequence. Finally, with wt ≤1 and the conditions in Theorem 5.6, applying the
conclusion from Remark 4.7 in [86] will give the last inequality. Following a similar approach can
also lead to the identical upper bound for the second term on the RHS of (C.17).

Bounding Term I1.2 Based on Arm Weights. Next, we proceed to bound term I1.2. Recall that we
need to derive the upper bound for the following term

I1.2 =



g(ext; θ˜it), (Σ(0)
˜it )−1(
X

τ∈[t−1]
w(τ)
˜it,t · g(xτ; θ0) · cτ)/m


≤

X

τ∈[t−1]


g(ext; θ˜it), (Σ(0)
˜it )−1 · w(τ)
˜it,t · g(xτ; θ0) · cτ/m


≤

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · (Σ(0)
˜it )−1g(xτ; θτ−1)cτ/m


+

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · (Σ(0)
˜it )−1(g(xτ; θ0) −g(xτ; θτ−1))cτ/m


≤

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · (Σ(0)
˜it )−1g(xτ; θτ−1) · cτ/m
 + O(Cm−1/6p

log(m)t1/6λ−7/6L4)

36


---Page Break---
≤

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · (Σ˜it)−1g(xτ; θτ−1) · cτ/m
 + O(Cm−1/6p

log(m)t1/6λ−7/6L4)

+

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · ((Σ(0)
˜it )−1 −(Σ˜it)−1)g(xτ; θτ−1) · cτ/m


≤

X

τ∈[t−1]


g(ext; θ˜it), w(τ)
˜it,t · (Σ˜it)−1g(xτ; θτ−1) · cτ/m


|
{z
}
I1.3

+O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4)
where the second and fourth inequality is due to triangular inequality. The third inequality is by
Lemma G.4, and the last inequality is due to Lemma G.6.

In particular, we aim to train separate neural models f(·; θi,t) for each candidate arm xi,t ∈
Xt, such that the term I1.2 can be minimized.
Recall that we denote ¯Σt−1
=
λI +
P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m as the "vanilla" gradient covariance matrix without the arm
weights, in terms of the potentially corrupted network parameters θτ−1.

In this case, with w(τ)
˜it,t referring to the weight of arm ext, the above formulation of term I1.3 can be
further transformed into

I1.3 ≤

X

τ∈[t−1]

w(τ)
˜it,t · cτ

m
·

g(ext; θ˜it), (Σ˜it)−1 · g(xτ; θτ−1)


≤

X

τ∈[t−1]
w(τ)
˜it,t · cτ ·
g(ext; θ˜it)/√m

(Σ˜it)−1 ·
(Σ˜it)−1 · g(xτ; θτ−1)/√m

Σ˜it



≤

X

τ∈[t−1]
w(τ)
˜it,t · cτ ·
g(ext; θ˜it)/√m

(Σ˜it)−1 ·
g(xτ; θτ−1)/√m

(Σ˜it)−1



≤

X

τ∈[t−1]
w(τ)
˜it,t · cτ ·
g(ext; θ˜it)/√m

( ¯Σ(κ)
t−1)−1 ·
g(xτ; θτ−1)/√m

( ¯Σ(κ)
τ−1)−1



≤αC · min
x∈Xt

g(x; θt−1)/√m


2

( ¯Σt−1)−1

≤αC ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1

where the first inequality follows from applying Holder’s inequality, and the second and third
inequalities are obtained by using Lemma G.8, along with the fact that Σ˜it ⪰¯Σ(κ)
t−1 ⪰¯Σ(κ)
τ−1 by

definition. The last two inequalities are derived from the definition of the arm weight w(τ)
˜it,t and
the corruption level C = P

t∈[T ] |ct|. Recall that for each arm xi,t ∈Xt, we define its weight as

w(τ)
i,t = min

(

1,

α· min
x∈Xt
∥g(x;θt−1)/√m∥2
¯
Σ−1
t−1
gτ ·∥g(xi,t;θt−1)/√m∥
( ¯
Σ(κ)
t−1)−1

)

, where α > 0 is the tunable parameter. As a result,

we will have the upper bound for I1.2, being

I1.2 ≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4)

In this case, since the weights are lower bounded by κ2, the only way to ensure this is by adjusting
the tunable parameter α > 0 accordingly. We remind that this does not require the learner to have a
global view of the minimum fraction value β. In practice, the learner can adjust the α values in each
round to ensure that the round-wise minimum weight value wmin
t
= min{w(τ)
i,t }i∈[K] = κ2 < 1, for

37


---Page Break---
all t ∈[T] and τ ∈[t −1], by tuning the parameter α. By summing up all the results, we obtain the
single-round bound for term I1, which leads to

I1 ≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1
+ O(
√

λS) ·
g(ext; θ˜it,t−1)/√m

(Σ′
t−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4).
(C.18)

C.7.2
Bounding the error term I2

Similarly, for the term I2 related to the chosen arm xt ∈Xt, we can follow the below procedure to
obtain a comparable bound as for term I1.

Corollary C.8. Suppose the imaginary corruption-free neural network f(·; eθi,t−1) for arm xi,t ∈Xt
has been trained on corruption-free rewards {xτ, erτ}τ∈[t−1]. f(·) is an L-layer FC network with
width m that satisfy the conditions in Theorem 5.6. Then, for the chosen arm xt ∈Xt, with the
probability at least 1 −δ, we will have

I2 ≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(
√

λS) ·
g(xt; θt−1)/√m

( ¯Σt−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4).

where θt−1 are the trained parameters associated to chosen arm xt ∈Xt in round t, and θ˜it,t−1 are
the trained parameters of arm ext, along with the corresponding weight-free gradient covariance
matrix ¯Σt−1 = λI + P
τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m.

Proof. The proof of this corollary follows an analogous procedure as in Lemma C.7.

Simplifying the notation. For notation simplicity, we apply eΘ
(J) and Θ(J) to represent the gradient
descent-based parameters associated with the arm xt. For terms specific to the arm xt = xit,t, we
simplify notation by using θt−1, Σt−1, and bt−1 to denote θit,t−1, Σit,t−1, and bit,t−1, respectively.

Then, with the simplified notation, it leads to

I2 =
⟨g(xt; θt−1), θt−1 −eθt−1⟩


≤
⟨g(xt; θt−1), θt−1 −Θ(J)⟩
 +
⟨g(xt; θt−1), eΘ
(J) −eθt−1⟩
 +
⟨g(xt; θt−1), eΘ
(J) −Θ(J)⟩

|
{z
}
I2.1

≤O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) +
⟨g(xt; θt−1), Θ(J) −eΘ
(J)⟩

|
{z
}
I2.1

where the first inequality is due to triangular inequality, and the last inequality is by applying
Lemma C.4 in [86], with the optimization problem being (C.12) bounding the difference between
gradient-based parameters and the auxiliary sequence.

Next, recall that with randomly initialized network parameters θ0, we can formulate the least square
parameters as (Σ(0)
t−1)−1b(0)
t−1/√m, while the least square parameters trained by corruption-free

38


---Page Break---
rewards are (Σ(0)
t−1)−1eb
(0)
t−1/√m. In this case, the term I2.1 can be alternatively transformed to

I2.1 =



g(xt; θt−1), Θ(J) −eΘ
(J)

≤



g(xt; θt−1), Θ(J) −θ0 −(Σ(0)
t−1)−1b(0)
t−1/√m


+



g(xt; θt−1), eΘ
(J) −θ0 −(Σ(0)
t−1)−1eb
(0)
t−1/√m


+



g(xt; θt−1), (Σ(0)
t−1)−1(
X

τ∈[t−1]
w(τ)
t
g(xτ; θ0) · cτ)/m


|
{z
}
I2.2

(C.19)

where the inequality is due to the definition of gradient-based regression parameters.

When trying to bound the first two terms on the RHS, we first apply Holder’s inequality with
¯Σt−1 = λI + P

τ∈[t−1] g(exτ; θτ−1)g(exτ; θτ−1)⊺/m, and it further leads to

g(xt; θt−1), Θ(J) −θ0 −(Σ(0)
t−1)−1b(0)
t−1/√m


≤
g(xt; θt−1)/√m

( ¯Σt−1)−1 · √m ·
Θ(J) −θ0 −(Σ(0)
t−1)−1b(0)
t−1/√m

( ¯Σt−1)

≤
g(xt; θt−1)/√m

( ¯Σt−1)−1 · √m ·
Θ(J) −θ0 −(Σ(0)
t−1)−1b(0)
t−1/√m

2 ·
( ¯Σt−1)

2

≤
g(xt; θt−1)/√m

( ¯Σt−1)−1 · O(λ + tL) ·
√m(Θ(J) −θ0) −(Σ(0)
t−1)−1b(0)
t−1

2

≤
g(xt; θt−1)/√m

( ¯Σt−1)−1 · O(
√

λ + tL)

· O

(1 −ηmλ)J/2p

t/λ + m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)


≤
g(xt; θt−1)/√m

( ¯Σt−1)−1 · O(
√

λS)

where the first two inequalities is by Holder’s inequality and Cauchy-Schwartz inequality. The third
inequality is by Lemma G.6. The fourth inequality is by applying Lemma D.1, with the optimization
problem being (C.12), in terms of the difference between gradient-based parameters and the auxiliary
sequence. Finally, with wt ≤1 and the conditions in Theorem 5.6, applying the conclusion from
Remark 4.7 in [86] will give the last inequality. Following a similar approach can also lead to the
upper bound for the second term on the RHS of (C.19).

Bounding Term I2.2 Based on Arm Weights. Next, we need to bound term I2.2. Recall that we
want to have an upper bound for the following term

I2.2 =



g(xt; θt−1), (Σ(0)
t−1)−1(
X

τ∈[t−1]
g(xτ; θ0) · cτ)/m


≤

X

τ∈[t−1]


g(xt; θt−1), (Σ(0)
t−1)−1g(xτ; θτ−1) · cτ/m
 + O(Cm−1/6p

log(m)t1/6λ−7/6L4)

≤

X

τ∈[t−1]


g(xt; θt−1), (Σt−1)−1g(xτ; θτ−1) · cτ/m
 + O(Cm−1/6p

log(m)t1/6λ−7/6L4)

+

X

τ∈[t−1]


g(xt; θt−1), ((Σ(0)
t−1)−1 −(Σt−1)−1)g(xτ; θτ−1) · cτ/m


≤

X

τ∈[t−1]


g(xt; θt−1), (Σt−1)−1g(xτ; θτ−1) · cτ/m


|
{z
}
I2.3

+O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4)

39


---Page Break---
where the first inequality follows from applying the triangle inequality and Lemma G.4. The second
inequality is also due to the triangle inequality, and the last inequality follows from Lemma G.6.
In particular, we aim to train separate neural models f(·; θi,t) for each candidate arm xi,t ∈Xt to
minimize the term I2.2. Similarly, recall that ¯Σt−1 = λI + P

τ∈[t−1] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m

is the gradient covariance matrix without arm weights, and w(τ)
t
is the weight of arm xt, defined as

w(τ)
t
= min{1,

α·minx∈Xt ∥g(x;θt−1)/√m∥2
¯
Σ−1
t−1
gτ ·∥g(xt;θt−1)/√m∥
( ¯
Σ(κ)
t−1)−1 }, where α > 0 is the tunable parameter. Similar to

the derivation of term I1.3, it can further lead to

I2.3 ≤

X

τ∈[t−1]

w(τ)
t
· cτ
m
·

g(xt; θt−1), (Σt−1)−1 · g(xτ; θτ−1)


≤

X

τ∈[t−1]
w(τ)
t
· cτ ·
g(xt; θt−1)/√m

(Σt−1)−1 ·
(Σt−1)−1 · g(xτ; θτ−1)/√m

Σt−1



≤

X

τ∈[t−1]
w(τ)
t
· cτ ·
g(xt; θt−1)/√m

(Σt−1)−1
·
g(xτ; θτ−1)/√m

(Σt−1)−1



≤

X

τ∈[t−1]
w(τ)
t
· cτ ·
g(xt; θt−1)/√m

( ¯Σ(κ)
t−1)−1
·
g(xτ; θτ−1)/√m

( ¯Σ(κ)
τ−1)−1



≤αC · min
x∈Xt

g(x; θt−1)/√m


2

( ¯Σt−1)−1

≤αC ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1

where the first inequality is by applying the Holder’s inequality, and the second and third inequalities
are by applying Lemma G.8 with the fact that Σt−1 ⪰¯Σ(κ)
t−1 ⪰¯Σ(κ)
τ−1 by definition. The last two
inequalities are due to the definition of arm weight w(τ)
t
, as well as the definition of corruption level
C = P
t∈[T ] |ct|. As a result, by combining the upper bounds for the terms I2.4 and I2.5, we obtain
the following upper bound for I2.2:

I2.2 ≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4).

In this case, since we need to ensure that the minimum weight is κ2, we scale the tunable parameter
α > 0 accordingly. Recall that this does not require the learner to have the prior knowledge of the
minimum fraction value β, and the learner can adjust the α values in each round to ensure that the
round-wise minimum weight wmin
t
= min{w(τ)
i,t }i∈[K] < 1, for all t ∈[T] and τ ∈[t −1]. By
summing up all the results, we have

I2 ≤O(αC) ·
g(xt; θt−1)/√m


2

( ¯Σt−1)−1 + O(
√

λS) ·
g(xt; θt−1)/√m

( ¯Σt−1)−1

+ O(m−2/3 log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)) + O(Cm−1/6p

log(m)t7/6λ−1/6L5)

+ O(Cm−1/6p

log(m)t1/6λ−7/6L4).

C.8
Deriving the UCB and confidence ellipsoid for corruption-free parameters and corrupted
parameters

In this subsection, we provide upper bounds for ]
UCBt(xt) in terms of the corruption-free param-
eters eθt−1. Recall that without the arm index i ∈[K], the parameters θ, covariance matrix Σt−1,

40


---Page Break---
confidence ellipsoid Ct−1, and weights wt pertain to the chosen arm xt. For the hypothetical
corruption-free parameters eθ, trained with corruption-free rewards, Lemma C.9 provides the cor-
responding UCB result, and Lemma C.10 introduces the associated confidence ellipsoid. For the
trained parameters θi,t−1, based on the received records Pt−1 with potentially corrupted rewards,
Lemma C.11 presents the corresponding UCB for each candidate arm xi,t ∈Xt, while Lemma C.12
provides the corresponding confidence ellipsoid. Note that Lemmas C.11 and C.12 are used solely to
motivate the design of our UCB-type exploration strategy and are not applied to derive the cumulative
regret analysis result.

Lemma C.9. Suppose the imaginary neural network f(·; eθt−1) in round t ∈[T] has been trained on
corruption-free rewards {xτ, erτ}τ∈[t−1], and f(·) is an L-layer FC network with width m. Suppose
we have m, J, η satisfying the conditions in Theorem 5.6. Then, for the chosen arm xt ∈Xt, with the
probability at least 1 −δ, we will have

]
UCBt(xt) = eV (xt) −h(xt) ≤2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

+ O(m−1/6p

log(m)t1/6λ−7/6L2/7),

and the corresponding summation value across T rounds will be
X

t∈[T ]
min
]
UCBt(xt), 1
	
=
X

t∈[T ]
min
eV (xt) −h(xt), 1
	

≤
1
p

wmin
t
O

ν

s

log det(eΣT )

det(λI) −2 log(δ) + λ1/2S

·

s

2T · log det(eΣT )

det(λI)

+ O(Sm−1/6p

log(m)T 7/6λ−1/6L2/7) + O(m−1/6p

log(m)T 7/6λ−7/6L2/7)

By definitions in Lemma C.10, we have the corresponding radius term for confidence ellipsoid eCt−1

as eγt−1 = O

ν ·
q

log det(eΣt−1)

det(λI)
−2 log(δ) + λ1/2S

, and the gradient covariance matrix as

eΣt−1 = λI + P

τ∈[t−1] w(τ)
t
· g(xτ; eθτ−1)g(xτ; eθτ−1)⊺/m.

Proof. With the confidence ellipsoid around the corruption-free parameters eCt−1 := {θ : ∥θ −
eθt−1∥eΣt−1 ≤eγt−1/√m, eγt−1 > 0}, we have θ∗∈eCt−1 according to Lemma C.10. In this case,

with the coefficient eγt−1 = O
 
ν ·
q

log det(eΣt−1)

det(λI)
−2 log(δ) + λ1/2S

and the gradient covariance

matrix eΣt−1 = λI + P

τ∈[t−1] w(τ)
t
· g(xτ; eθτ−1)g(xτ; eθτ−1)⊺/m, we have

]
UCBt(xt) = eV (xt) −h(xt)

= max
θ∈e
Ct−1


g(xt; θ0), θ −θ0

−

g(xt; θ0), θ∗−θ0


≤max
θ∈e
Ct−1


g(xt; eθ0), θ −eθt−1

−

g(xt; eθt−1), θ∗−eθt−1


+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

≤max
θ∈e
Ct−1
∥θ −eθt−1∥eΣt−1∥g(xt; eθ0)∥eΣ−1
t−1 + ∥g(xt; eθt−1)∥eΣ−1
t−1∥θ∗−eθt−1∥eΣt−1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

≤2eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1 + O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

+ O(m−1/6p

log(m)t1/6λ−7/6L2/7)

(C.20)

where the first inequality is by Lemma G.4, second inequality is by Holder’s inequality, and the
last inequality is by applying the definition of confidence ellipsoid eCt−1, Lemma G.4, and Lemma
G.6. Next, recall that the above ]
UCBt is one term composing the single-round regret Rt, and for the
cumulative regret, it will be summed up across T rounds. In this case, for the first term on the RHS

41


---Page Break---
above, we have
X

t∈[T ]
2 · min

eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1, 1
	
=
X

t∈[T ]
2 · min
eγt−1
√wt
· ∥√wtg(xt; eθt−1)/√m∥eΣ−1
t−1, 1
	

≤2(1 +
eγT
p

wmin
t
) ·
X

t∈[T ]
min

∥√wtg(xt; eθt−1)/√m∥eΣ−1
t−1, 1
	

≤2(1 +
eγT
p

wmin
t
) ·
s

T ·
X

t∈[T ]
min

∥√wtg(xt; eθt−1)/√m∥2
eΣ−1
t−1, 1
	

where we have wmin
t
< 1. Then, applying Lemma G.7 and by the definition of γT as well as supposing
that
eγT
√

wmin
t
≥1, we will have

X

t∈[T ]
2 min

eγt−1 · ∥g(xt; eθt−1)/√m∥eΣ−1
t−1, 1
	

≤
1
p

wmin
t
O

ν

s

log det(eΣT )

det(λI) −2 log(δ) + λ1/2S

·

s

2T · log det(eΣT )

det(λI) .

Next, for the corrupted parameters θi,t−1, we consider the confidence ellipsoid Ci,t−1 := {θ :
∥θ −θi,t−1∥Σi,t−1 ≤γi,t−1/√m} constructed around the corrupted parameters θi,t−1 , with the

coefficient γi,t−1 = O
 
ν ·
q

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S

and the gradient covariance matrix

Σi,t−1 = λI + P

τ∈[t−1] w(τ)
i,t · g(xτ; θτ−1)g(xτ; θτ−1)⊺/m.

Lemma C.10. In round t, with the notation and conditions in Theorem 5.6, suppose the corruption-
free parameters eθi,t−1 associated with a candidate arm xi,t ∈Xt are trained by L(θ) =
1
2
P

τ∈[t−1] w(τ)
i,t · |f(xτ; θ) −erτ|2 + mλ

2
· ∥θ −θ0∥2
2. Then, we have the corresponding confi-
dence ellipsoid
eCi,t−1 := {θ : ∥θ −eθi,t−1∥eΣi,t−1 ≤eγi,t−1/√m},

such that θ∗∈eCi,t−1, where eγi,t−1 = O
 
ν ·

r

log det(eΣi,t−1)

det(λI)
−2 log(δ)+λ1/2S

, and the gradient

covariance matrix eΣi,t−1 = λI + P

τ∈[t−1] w(τ)
i,t · g(xτ; eθτ−1)g(xτ; eθτ−1)⊺/m.

Proof. The proof of this lemma follows an analogous approach as in Lemma 5.2 in [86]. Recall
that based on Lemma C.1, we have the expected reward of an arm x ∈Xt being E[r|x] = h(x) =
⟨g(x; θ0), θ∗−θ0⟩where there exist parameters θ∗such that ∥θ∗−θ0∥≤S/√m, S > 0.
Intuitively, for each previously chosen arm xτ, τ ∈[t −1], we can consider an alternative form being

E
q

w(τ)
i,t · r
 xτ


=
q

w(τ)
i,t · h(xτ) =
q

w(τ)
i,t · g(xτ; θ0)/√m, √m(θ∗−θ0)


where w > 0 refers to the weight associated with arm x in our settings of R-NeuralUCB. Afterwards,
with the weighted sequence of chosen arm gradients as well as their expected corruption-free rewards,
we can have

∥√m(θ∗−θ0) −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1∥eΣ(0)
i,t−1 ≤ν ·

s

log
det(eΣ(0)
i,t−1)
det(λI)
−2 log(δ) + λ1/2S,

by applying the conclusion of Theorem 2 from [1]. In this case, by triangular inequality, we also have

∥θ∗−θ0∥eΣi,t−1

≤∥θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥eΣi,t−1 + ∥eθi,t−1 −θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥eΣi,t−1.

42


---Page Break---
Then, for the first term on the right hand side, we have

∥θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥2
eΣi,t−1

= (θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)⊺eΣi,t−1(θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)

= (θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)⊺eΣ(0)
i,t−1(θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)

+ (θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)⊺· (eΣi,t−1 −eΣ(0)
i,t−1) · (θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)

≤(θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)⊺eΣ(0)
i,t−1(θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)

+
∥eΣi,t−1 −eΣ(0)
i,t−1∥2
λ
· (θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)⊺eΣ(0)
i,t−1(θ∗−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m)

≤(1 +
∥eΣi,t−1 −eΣ(0)
i,t−1∥2
λ
) ·

ν ·

s

log
det(eΣ(0)
i,t−1)
det(λI)
−2 log(δ) + λ1/2S

/m

≤m−1/2 ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6)

·

ν ·

s

log det(eΣi,t−1)

det(λI)
+ O(m−1/6p

log(m)L4t5/3λ−1/6) −2 log(δ) + λ1/2S


where the first inequality is because x⊺Ax ≤x⊺Bx · ∥A∥2/λmin(B) for some 0 ≺B and the fact
that the minimum eigenvalue λmin(eΣ(0)
i,t−1) ≥λ, and the last inequality is by Lemma G.6 as well as

due to the fact w(τ)
i,t ≤1. Afterwards, for the second term, we have

∥eθi,t−1−θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥eΣi,t−1 ≤
q

∥eΣi,t−1∥2 · ∥eθi,t−1 −θ0 −(eΣ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥2

≤O(
√

λ + tL) · O

(1 −ηmλ)J/2p

t/mλ + m−2/3p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)

.

The first inequality follows from applying Lemma G.6. For the second inequality, we apply a similar
approach as in Lemma D.1, with the optimization problem given by (C.12). Since we scale the α
parameter to ensure 1 > w(τ)
i,t ≥κ2, we can follow the proof of Lemma B.2 in [86] to bound the
difference between GD-based optimization and gradient-based regression. With the minimum weight
value lower bounded by κ2 and under the conditions in Theorem 5.6, applying Remark 4.7 in [86]
completes the proof.

Lemma C.11. For candidate arm xi,t ∈Xt, suppose its associated neural network f(·; θi,t−1) has
been trained on received records {xτ, rτ}τ∈[t−1], with J iterations of GD and learning rate η. Let
f(·; θi,t−1) be an L-layer FC network with width m. Suppose conditions in Theorem 5.6 are satisfied.
Then, given the candidate arm xi,t ∈Xt, with a constant ζ > 0 and probability at least 1 −δ, we
will have
f(xi,t; θi,t−1) −h(xi,t)


≤∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1 ·

ζ ·

ν

s

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S


+ (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−1/6p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12)


+ O(αC) · min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3),
(C.21)

43


---Page Break---
with notation and definitions in Theorem 5.6. We also have the covariance matrix Σi,t−1 = λI +
P

τ∈[t] w(τ)
i,t · g(xτ; θτ−1)g(xτ; θτ−1)⊺, with gradient vector g(xi,t; θ) = vec(∇θf(xi,t; θ)).

Proof. Applying the Lemma C.1, we can transform the objective by substituting the reward mapping
function h(·), as

f(xi,t;θi,t−1) −h(xi,t)
 =
f(xi,t; θi,t−1) −

g(xi,t; θ0), θ∗−θ0


≤
f(xi,t; θi,t−1) −

g(xi,t; θi,t−1), θ∗−θ0
 + O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

≤
f(xi,t; θi,t−1) −

g(xi,t; θi,t−1), θi,t−1 −θ0


+

g(xi,t; θi,t−1), θi,t−1 −θ0

−

g(xi,t; θi,t−1), θ∗−θ0


+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

where the first equality is due to Lemma C.1, while the first inequality is due to Lemma G.4 and
Lemma C.1, and the last inequality is because of the triangular inequality. Then, we proceed to
separately bound the first and second term on the RHS. For the first term, we will have

f(xi,t; θi,t−1) −

g(xi,t; θi,t−1), θi,t−1 −θ0


=
f(xi,t; θi,t−1) −f(xi,t; θ0) −

g(xi,t; θi,t−1), θi,t−1 −θ0


≤O(m−1/6p

log(m)t2/3λ−2/3L3)

where the first equality is due to the fact that f(xi,t; θ0) = 0 based on our parameter initialization
approach, and the inequality is by applying Lemma G.5 and Lemma G.3. Then, for the second term,
we will have


g(xi,t; θi,t−1),θi,t−1 −θ0

−

g(xi,t; θi,t−1), θ∗−θ0


=

g(xi,t; θi,t−1), θ∗−θi,t−1


≤∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1 · √m · ∥θ∗−θi,t−1∥Σi,t−1

≤γi,t−1 · ∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1

where the first inequality is by applying the Holder’s inequality, and the last inequality is by applying
Lemma 5.2 in [86], Lemma C.1 in terms of the confidence set Ci,t−1 and the fact that θ∗∈Ci,t−1.
Then, with the confidence ellipsoid introduced and discussed in Lemma C.12, summing up the results
above, we will then have

f(xi,t; θi,t−1) −h(xi,t)


≤∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1

·

O

ν

s

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S

+ O(αC) ·
min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1
∥g(xi,t; θi,t−1)/√m∥( ¯Σ(κ)
t−1)−1

+ (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12)


+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

44


---Page Break---
= ∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1

·

O

ν

s

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S


+ (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−1/6p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12)


+ ∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1 · O(αC) ·
min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1
∥g(xi,t; θi,t−1)/√m∥( ¯Σ(κ)
t−1)−1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3).

where we naturally have Hermitian matrices Σi,t−1 ⪰
¯Σ(κ)
t−1 by definition, which leads to
(Σi,t−1)−1 ⪯( ¯Σ(κ)
t−1)−1 by applying the conclusion from Lemma G.8. Afterwards, due to the
fact that the minimum round-wise arm weight wmin
t
= min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T]
by scaling parameter α, we can further have
f(xi,t; θi,t−1) −h(xi,t)


≤∥g(xi,t; θi,t−1)/√m∥Σ−1
i,t−1

·

O

ν

s

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S


+ (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12)


+ O(αC) · min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3),

which completes the proof for this lemma.

Lemma C.12. In round t ∈[T], with the notation and conditions from Theorem 5.6, suppose the
corrupted parameters θi,t−1 associated with a candidate arm xi,t ∈Xt are trained by L(θ) =
1
2
P

τ∈[t−1] w(τ)
i,t · |f(xτ; θ) −rτ|2 + mλ

2 · ∥θ −θ0∥2
2. Then, we have the confidence ellipsoid

Ci,t−1 =

θ : ∥θ −θi,t−1∥Σi,t−1 ≤γi,t−1/√m


where we have the unknown parameter θ∗∈Ci,t−1, and we denote

γi,t−1 = O

ν

s

log det(Σi,t−1)

det(λI)
−2 log(δ) + λ1/2S

+ O(αC) ·
min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1
∥g(xi,t; θi,t−1)/√m∥( ¯Σ(κ)
t−1)−1

+ (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−1/6p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

+ ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−1/12 log1/4(m)L2t5/6λ−1/12).

45


---Page Break---
Proof. Recall that for the imaginary corruption-free parameters eθi,t−1, trained on corruption-free
records ePt−1, we can construct the confidence interval eCi,t−1 := {θ : ∥θ −eθi,t−1∥eΣi,t−1 ≤

eγi,t−1/√m, eγi,t−1 > 0}, ensuring that the unknown θ∗in Lemma C.1 satisfies θ∗∈eCi,t−1. The
corresponding confidence ellipsoid is presented in Lemma C.10.

With the ellipsoid centered at θi,t−1 and the gradient covariance matrix defined as Σi,t−1 = λI +
P

τ∈[t−1] w(τ)
i,t · g(xτ; θτ−1) · g(xτ; θτ−1)⊺/m, we proceed to derive the corresponding radius.

Recall that w(τ)
i,t denotes the sample weight associated with the chosen arm xτ. For reference, we first

recall the preliminary bounds: ∥Σi,t−1 −eΣi,t−1∥F ≤O(m−1/6p

log(m)L4t7/6λ−1/6) by Lemma
G.6 and ∥θi,t−1 −eθi,t−1∥2 ≤O(
p

t/(mλ)) as shown in Lemma G.3. Next, as we already have
∥θ −eθi,t−1∥eΣi,t−1 ≤eγi,t−1/√m, we then proceed to transform the objective to

∥θ∗−θi,t−1∥Σi,t−1 ≤∥θ∗−eθi,t−1∥Σi,t−1 + ∥eθi,t−1 −θi,t−1∥Σi,t−1
≤∥θ∗−eθi,t−1∥Σi,t−1 + ∥eθi,t−1 −θi,t−1∥Σi,t−1
≤∥θ∗−eθi,t−1∥Σi,t−1−eΣi,t−1+eΣi,t−1 + ∥eθi,t−1 −θi,t−1∥Σi,t−1

≤∥θ∗−eθi,t−1∥eΣi,t−1 + ∥θ∗−eθi,t−1∥Σi,t−1−eΣi,t−1 + ∥eθi,t−1 −θi,t−1∥Σi,t−1

≤eγi,t−1/√m + ∥eθi,t−1 −θi,t−1∥Σi,t−1 + O(
p

t/(mλ)) · O(m−1/6p

log(m)L4t7/6λ−1/6)

≤eγi,t−1/√m + ∥eθi,t−1 −θi,t−1∥Σi,t−1 + O(m−2/3p

log(m)L4t13/6λ−2/3).

For the first term on the RHS, we can follow the proof flow in Lemma C.6, by applying Lemma
G.4 to substitute eΣi,t−1 with corresponding Σi,t−1, which will consequently lead to |γi,t−1/√m −

eγi,t−1/√m| ≤ν ·
q

1 + O(m−1/6p

log(m)L4t7/6λ−7/6) · O(m−7/12 log1/4(m)L2t5/6λ−1/12).
Meanwhile, for the second term on the RHS, we first define the gradient-based regression parameters
as

Σ(0)
i,t−1 = λI +
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · g(xτ; θ0)⊺/m,

b(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · rτ/√m,

eb
(0)
i,t−1 =
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · erτ/√m,

Then, we can proceed to have

∥eθi,t−1 −θi,t−1∥Σi,t−1

≤∥eθi,t−1 −θ0 −(Σ(0)
i,t−1)−1b(0)
i,t−1/√m + (Σ(0)
i,t−1)−1b(0)
i,t−1/√m + θ0 −θi,t−1∥Σi,t−1

≤∥θi,t−1 −θ0 −(Σ(0)
i,t−1)−1b(0)
i,t−1/√m∥Σi,t−1 + ∥eθi,t−1 −θ0 −(Σ(0)
i,t−1)−1b(0)
i,t−1/√m∥Σi,t−1

≤∥θi,t−1 −θ0 −(Σ(0)
i,t−1)−1b(0)
i,t−1/√m∥Σi,t−1 + ∥eθi,t−1 −θ0 −(Σ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥Σi,t−1

+ m−1∥(Σ(0)
i,t−1)−1 · (
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1.

(C.22)

Bounding the first two term in Inequality C.22. Here, for the first term on the RHS, we can
individually apply Lemma D.1, by considering the auxiliary sequence in j-th iteration (j ∈[J]) with
Θ(0) = θ0, as

Θ(j+1) = Θ(j) −η ·

J(0) · W ·
 
[J(0)]⊺(Θ(j) −θ0) −y

+ mλ(Θ(j) −θ0)


where the Jacobian matrix J(0) :=
 
g(x1; θ0), g(x2; θ0), . . . , g(xi,t−1; θ0)

∈Rp×(t−1), vector
y ∈Rt−1 contains the received arm rewards rτ, τ ∈[t −1], and matrix W ∈R(t−1)×(t−1) is the

46


---Page Break---
diagonal matrix that contains sample weights w(τ)
i,t , τ ∈[t −1]. In particular, we have its norm
∥W∥2 ≤1 by definition. Here, with [J(0)]τ being the τ-th column of matrix J(0), the above sequence
is expected to solve the following problem

min
Θ L(Θ) =
X

τ∈[t−1]

w(τ)
i,t
2
·
[J(0)]⊺
τ(Θ −θ0) −rτ



2

2
+ 1

2 · mλ ·
Θ −θ0



2

2
.

As a result, following an analogous approach as in Lemma C.4 in [86], we can have ∥Θ(j) −θ0 −
(Σ(0)
i,t−1)−1b(0)
i,t−1/√m∥Σi,t−1 ≤(1 −ηmλ)j/2p

t/(mλ). Furthermore, by applying the conclusion
of Lemma D.1, we can have

∥θi,t−1 −θ0−(Σ(0)
i,t−1)−1b(0)
i,t−1/√m∥2

≤(1 −ηmλ)J/2p

t/(mλ) + O(m−2/3p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)).

Similarly, for the second term in Inequality C.22, we also can apply a comparable approach by solving
the problem:

min
Θ L(Θ) =
X

τ∈[t−1]

w(τ)
i,t
2

[J(0)]⊺
τ(Θ −θ0) −erτ



2

2
+ 1

2mλ ·
Θ −θ0



2

2
,

and constructing the corresponding auxiliary sequence. This will lead to a similar bound for the

second term on the RHS of inequality C.22, such that ∥eθi,t−1 −θ0 −(Σ(0)
i,t−1)−1eb
(0)
i,t−1/√m∥2 ≤
(1 −ηmλ)J/2p

t/(mλ) + O(m−2/3p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)).

Bounding the third term in Inequality C.22. Then, for the third term on the RHS, we first have

m−1 · ∥(Σ(0)
i,t−1)−1 · (
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1

≤m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1

+ m−1∥
 
(Σi,t−1)−1 −(Σ(0)
i,t−1)−1
· (
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1

≤m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1

+ m−1∥(Σi,t−1)−1 
Σi,t−1 −Σ(0)
i,t−1

(Σ(0)
i,t−1)−1 · (
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1

≤m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θ0) · cτ)∥Σi,t−1 + O(Cm−2/3p

log(m)t7/6λ−13/6L9/2)

≤m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · cτ)∥Σi,t−1 + O(Cm−2/3p

log(m)t7/6λ−13/6L9/2)

+ m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · (g(xτ; θτ−1) −g(xτ; θ0)) · cτ)∥Σi,t−1

≤m−1∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · cτ)∥Σi,t−1 + O(Cm−7/6p

log(m)t1/6λ−7/6L7/2)

+ O(Cm−2/3p

log(m)t7/6λ−13/6L9/2),

where the third inequality is due to Lemma G.6, and the last inequality is due to Lemma G.4. Then,
recall that the weight w(τ)
i,t from (4) for each previously chosen arm xτ. In this case, we can further

47


---Page Break---
have

m−1/2∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · cτ)/√m∥Σi,t−1

= m−1/2∥
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · cτ/√m∥(Σi,t−1)−1

≤m−1/2
X

τ∈[t−1]
w(τ)
i,t cτ · ∥g(xτ; θτ−1)/√m∥(Σi,t−1)−1

where the first inequality is by applying the triangular inequality. By definition, we naturally
have Σi,t−1 ⪰¯Σ(κ)
τ−1, ∀τ ∈[t −1]. Next, we utilize Lemmas G.8 and the fact that wmin
t
=
min{w(τ)
i,t }i∈[K],τ∈[t−1] = κ2 < 1, ∀t ∈[T] by scaling the parameter α, which will lead to

m−1/2∥(Σi,t−1)−1(
X

τ∈[t−1]
w(τ)
i,t · g(xτ; θτ−1) · cτ)/√m∥Σi,t−1

≤O(m−1/2)
X

τ∈[t−1]
w(τ)
i,t cτ · ∥g(xτ; θτ−1)/√m∥(Σi,t−1)−1

≤O(m−1/2)
X

τ∈[t−1]
w(τ)
i,t cτ · ∥g(xτ; θτ−1)/√m∥( ¯Σ(κ)
τ−1)−1

≤O(m−1/2)
X

τ∈[t−1]
cτ ·
α · min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1
∥g(xi,t; θi,t−1)/√m∥( ¯Σ(κ)
t−1)−1

≤O(m−1/2) · C ·
α · min
x∈Xt∥g(x; θi,t−1)/√m∥2
¯Σ−1
t−1
∥g(xi,t; θi,t−1)/√m∥( ¯Σ(κ)
t−1)−1
,

where the third and the last inequalities are due to the definition of w(τ)
i,t and the definition of
corruption level C. Finally, summing up all the results will give the lemma.

C.9
Discussion on the Minimum Fraction Value β

In this subsection, we provide an exemplary upper bound of O(1/β). Inspired by [84], the following
analysis is based on a scenario where the arm contexts nearly lie within a low-dimensional subspace
of the RKHS, induced by the NTK defined in Definition 5.2. To begin, we recall that, as defined in
(4), each candidate arm xi,t ∈Xt is associated with the corresponding arm weight:

w(τ)
i,t = min




1,
α · minx∈Xt ∥g(x; θt−1)/√m∥2
¯Σ−1
t−1
gτ · ∥g(xi,t; θt−1)/√m∥( ¯Σ(κ)
t−1)−1




= min

1, α · fracτ(xi,t; Xt, ¯Σt−1)
	
,

where α > 0 is a tunable scaling parameter to control the arm weight value range, and frac(·)
is a shorthand for the fraction term. We denote the data-dependent minimum fraction value as
β = mint∈[T ],τ∈[t−1]

min

fracτ(xt; Xt, ¯Σt−1), fracτ(ext; Xt, ¯Σt−1)
	
. In this case, the lower
bound for fracτ(xi,t; Xt, ¯Σt−1), represented by β, can be expressed as

minx∈Xt ∥g(x; θt−1)/√m∥2
¯Σ−1
t−1
∥g(xτ; θτ−1)/√m∥( ¯Σ(κ)
τ−1)−1 · ∥g(xi,t; θt−1)/√m∥( ¯Σ(κ)
t−1)−1
≥
minx∈Xt ∥g(x; θt−1)/√m∥2
¯Σ−1
t−1
O(L · λ−1)

≥minx∈Xt ∥g(x; θt−1)/√m∥2
2 · λmin( ¯Σ−1
t−1)
O(L · λ−1)

where the first inequality is because of Lemma G.2 and Lemma G.6. The second inequality is by
applying Rayleigh-Ritz theorem.

48


---Page Break---
Then, we let ˘Pt−1 ⊆Pt−1 being the collection of received records that only consists of unique
chosen arms from Pt−1. With ˘θt−1 be the parameters trained on ˘Pt−1, applying Lemma G.4 and
Lemma G.3, we have ∥g(x′; θt−1)/√m−g(x′; ˘θt−1)/√m∥2 ≤O(m−1/6λ−1/6t1/6L7/2 log(m)),
where x′ = arg minx∈Xt ∥g(x; θt−1)/√m∥2. Given the over-parameterization settings in Theorem
5.6 with sufficiently large m, we can have ∥g(x′; θt−1)/√m −g(x′; ˘θt−1)/√m∥2 ≪O(1). As a
result, it leads to

minx∈Xt ∥g(x; θt−1)/√m∥2
¯Σ−1
t−1
∥g(xτ; θτ−1)/√m∥( ¯Σ(κ)
τ−1)−1 · ∥g(xi,t; θt−1)/√m∥( ¯Σ(κ)
t−1)−1
≥minx∈Xt ∥g(x; θt−1)/√m∥2
2 · λmin( ¯Σ−1
t−1)
O(L · λ−1)

≥

 
∥g(x′; ˘θt−1)/√m∥2
2 + ∥g(x′; θt−1)/√m∥2
2 −∥g(x′; ˘θt−1)/√m∥2
2

· λmin( ¯Σ−1
t−1)
O(L · λ−1)

≥O(1) · λmin( ¯Σ−1
t−1)
O(L · λ−1)

where the second inequality is derived based on ∥g(x′; θt−1)/√m∥2
2 −∥g(x′; ˘θt−1)/√m∥2
2 ≥
−1 · (∥g(x′; θt−1)/√m −g(x′; ˘θt−1)/√m∥2 · ∥g(x′; θt−1)/√m + g(x′; ˘θt−1)/√m∥2), and we
bound the two multipliers separately with choice of m. The last inequality is by applying Theorem 3
of [3] with the fact that f(·; θt−1) = O(1) (Lemma B.2 in [21]).

Afterwards, regarding the lower bound of λmin( ¯Σ−1
t−1), since λmin( ¯Σ−1
t−1) =
1
λmax( ¯Σt−1), we need to

find the upper bound for ∥¯Σt−1∥2. Denoting G(0)
t−1 = [g(x1; Θ0), g(x2; Θ0), . . . , g(xt−1; Θ0)]⊺∈
R(t−1)×p as the gradient matrix, as well as Ht−1 ∈R(t−1)×(t−1) as the NTK matrix constructed
from the chosen arms {xτ}τ∈[t−1], with sufficient network width m (Theorem 5.6), we first have

∥G(0)
t−1(G(0)
t−1)⊺/m −Ht−1∥2 ≤O(1)

and this inequality is because of Lemma B.1 in [86] by setting ϵ = t −1. In this case, we will have

∥¯Σt−1∥2 −λmax(Ht−1) ≤∥¯Σt−1∥2 −∥(G(0)
t−1)⊺G(0)
t−1/m∥2 + ∥G(0)
t−1(G(0)
t−1)⊺/m∥2 −λmax(Ht−1)

≤∥¯Σt−1 −(G(0)
t−1)⊺G(0)
t−1/m∥2 + ∥G(0)
t−1(G(0)
t−1)⊺/m −Ht−1∥2

≤O(m−1/6p

log(m)L4t7/6λ−1/6) + O(λ) + O(1)

where the last inequality is by applying Lemma G.6. This leads to the result that ∥¯Σt−1∥2 ≤
λmax(Ht−1) + O(m−1/6p

log(m)L4t7/6λ−1/6) + O(λ) + O(1).

Then, inspired by Section D in [84], based on the formulation from [14] and [20], we can consider
that each entry of Ht−1 is generated by

[Ht−1]i,i′ =

∞
X

k=0
µk

N(d,k)
X

j=1
Yk,j(xi)Yk,j(xi′),

where Yk,j are linearly independent spherical harmonics, w.r.t. degree k and d variables. N(d, k) =
2k+d−2

k
· Cd−2
k+d−3, µk = Θ(max{k−d, (d −1)1−k}). With the above feature mapping, if we have a
subspace of the RKHS, such that the feature mapping in the RKHS is close enough to its projection
onto this subspace, we can have λmax(Ht−1) = ∥Ht−1∥2 ≤O(1). As a result, summing up the
results and with sufficiently large m, we can have 1

β ≤O(Lλ−1).

D
Bounding the Difference of Trained Parameters and Regression
Parameters

In section, with weighted Gradient Descent, we provide the upper bound in terms of the distance
between GD trained parameters θt−1 and the regression parameters (Σ(0)
t−1)−1b(0)
t−1. The results will
be applied to the proof flow of both NeuralUCB-WGD and R-NeuralUCB. First, with the definitions

49


---Page Break---
in Subsec. C.2, we have the gradient-based regression parameters specified to arm xt ∈Xt as

Σ(0)
t−1 = λI +
X

τ∈[t−1]
w(τ)
t
g(xτ; θ0)g(xτ; θ0)⊺/m,

Σt−1 = λI +
X

τ∈[t−1]
w(τ)
t
g(xτ; θτ−1)g(xτ; θτ−1)⊺/m,

b(0)
t−1 =
X

τ∈[t−1]
w(τ)
t
g(xτ; θ0) · rτ/√m,
bt−1 =
X

τ∈[t−1]
w(τ)
t
g(xτ; θτ−1) · rτ/√m

where {xτ, rτ}, τ ∈[t] respectively stands for the chosen arms as well as their rewards. For notation
simplicity, we also use w(τ)
t
, τ ∈[t −1] to denote the arm weights for the chosen arm xt respectively.

Analogously, given an candidate arm xi,t ∈Xt with arm weight w(τ)
i,t defined in (4), we have a series
auxiliary gradient sequences {Θ(0), Θ(1), . . . , Θ(J)} as in G.3, such that for j-th iteration

Θ(j+1) = Θ(j) −η ·

J(0) · W ·
 
[J(0)]⊺(Θ(j) −θ0) −y

+ mλ(Θ(j) −θ0)


where J(0) :=
 
g(x1; θ0), g(x2; θ0), . . . , g(xt−1; θ0)

∈Rp×(t−1), and W refers to the diagonal
matrix of arm weights {w(τ)
i,t }τ∈[t−1], along with the reward vectors y ∈Rt−1 separately being the
vector of received rewards and corruption-free rewards. In particular, the auxiliary sequence Θ(j)

can be deemed as applying Gradient Descent to solve the following optimization problem

min
Θ L(Θ) =
X

τ∈[t−1]

1
2 · w(τ)
i,t ·
[J(0)]⊺
τ(Θ −θ0) −yτ



2

2
+ 1

2 · mλ ·
Θ −θ0



2

2

where [J(0)]τ refers to the τ-th column of the matrix J(0). Analogously, we can also derive the

optimization problem for the sequence of corruption-free auxiliary parameters eΘ
(j), by applying the
same definition of weight matrix W. By the definition of arm weights, we will also have ∥W∥2 ≤1.
Lemma D.1 (Lemma B.2 of [86]). With the notation and conditions in Theorem 5.6, consider
m ≥Ω(poly(T, L, ˘λ−1
0 , λ−1) and η ≤O(
1
mλ+tmL). For round t ∈[T], we have

∥θt−1 −θ0 −(Σ(0)
t−1)−1b(0)
t−1∥2

≤(1 −ηmλ)J/2p

t/(mλ) + O(m−2/3t5/3p

log(m)L7/2λ−5/3(1 +
p

t/λ))

with the J iterations of Gradient Descent process.

Proof. The proof of this lemma follows an analogous approach as in Lemma B.2 of [86]. Here, similar
to the previous sequence {Θ(0), Θ(1), . . . , Θ(J)}, we denote another set of auxiliary sequences to
simulate the J iterations of GD, by

θ(j+1) = θ(j) −η ·

J(j) · W ·
 
f (j) −y

+ mλ(θ(j) −θ0)


where we denote the corresponding gradient matrix at the j-th iteration as J(j)
=
 
g(x1; θ(j)), g(x2; θ(j)), . . . , g(xt−1; θ(j))

∈Rp×(t−1) , as well as the vector of network out-
puts f (j) =
 
f(x1; θ(j)), f(x2; θ(j)), . . . , f(xt−1; θ(j))

∈Rt−1. In this case, we can have the
difference between parameter sequences as

∥θ(j+1) −Θ(j+1)∥

= ∥(1 −ηmλ) · (θ(j) −Θ(j)) −ηW(J(j) −J(0))(f (j) −y) −ηW(f (j) −[J(0)]⊺(Θ(j) −θ0))∥

≤∥(1 −ηmλ) · (θ(j) −Θ(j))∥+ η∥W(J(j) −J(0))(f (j) −y)∥+ η∥J(0)W(f (j) −[J(0)]⊺(Θ(j) −θ0))∥

≤∥(1 −ηmλ)(θ(j) −Θ(j))∥+ η∥W∥∥(J(j) −J(0))(f (j) −y)∥+ η∥J(0)W∥∥f (j) −[J(0)]⊺(Θ(j) −θ0)∥

≤∥(1 −ηmλ) · (θ(j) −Θ(j))∥
|
{z
}
I4

+ η · ∥(J(j) −J(0))(f (j) −y)∥
|
{z
}
I5

+ η · ∥J(0)∥∥f (j) −[J(0)]⊺(Θ(j) −θ0)∥
|
{z
}
I6

50


---Page Break---
where the first inequality is by applying the triangular inequality. The second inequality is by using
Cauchy-Schwartz inequality, and the third inequality is due to the fact that ∥W∥2 ≤1. Here, the first
term I4 can be bounded recursively. For the second term I5 on the RHS, we have

I5 = η∥(J(j) −J(0))(f (j) −y)∥≤η∥(J(j) −J(0))∥∥(f (j) −y)∥≤O(ηt7/6m1/3p

log(m)L7/2λ−1/6)

by extending Lemma G.4 to the matrix J. Since we have the arm weights w(τ)
i,t ≤O(1) due to the
maximum cap as well as the choice of parameter α, we can also directly apply the conclusion of
Lemma C.3 in [86] for the second inequality. Meanwhile, for the third term I6, we will have

I6 = η · ∥J(0)∥∥f (j) −[J(0)]⊺(Θ(j) −θ0)∥

≤η · ∥J(0)∥· max
τ∈[t−1]

√

t · |f(xτ; θj) −f(xτ; θ0) −⟨g(xτ; θ0), θj −θ0⟩|

≤O(ηt5/3m1/3p

log(m)L7/2λ−2/3)

by extending Lemma G.2 to the gradient matrix setting, as well as applying the Lemma G.5 on the
absolute value. Afterwards, we can integrate these three terms, which will lead to

∥θ(j+1) −Θ(j+1)∥≤∥(1 −ηmλ) · (θ(j) −Θ(j))∥

+ O(ηt7/6m1/3p

log(m)L7/2λ−1/6) + O(ηt5/3m1/3p

log(m)L7/2λ−2/3)

≤(1 −ηmλ)∥θ(j) −Θ(j)∥+ O(ηt7/6m1/3p

log(m)L7/2λ−1/6) + O(ηt5/3m1/3p

log(m)L7/2λ−2/3)

≤O(m−2/3t5/3p

log(m)L7/2λ−5/3(1 +
p

t/λ))

where for term I4, the last inequality is obtained by recursively applying the process to ∥θ(0) −
Θ(0)∥= 0 and substituting the chosen upper bound for the learning rate η. Then, with a sufficiently
large network width m as indicated in the lemma, we have

∥θt−1 −θ0 −(Σ(0)
t−1)−1b(0)
t−1∥2 ≤∥θ(j+1) −Θ(j+1)∥+ ∥Θ(j+1) −θ0 −(Σ(0)
t−1)−1b(0)
t−1∥2

≤∥Θ(j+1) −θ0 −(Σ(0)
t−1)−1b(0)
t−1∥2 + O(m−2/3t5/3p

log(m)L7/2λ−5/3(1 +
p

t/λ))

≤(1 −ηmλ)j/2p

t/(mλ) + O(m−2/3t5/3p

log(m)L7/2λ−5/3(1 +
p

t/λ))

where the first inequality is by applying triangular inequality, and the second inequality is by applying
the previous conclusion. The last inequality is by applying Lemma C.4 in [86] with the fact that
∥θ(j+1) −Θ(j+1)∥≤
p

t/(mλ). Since we have the arm weights w(τ)
i,t ≤1 due to its maximum
bound, we apply the conclusion of Lemma C.4 in [86] for the last inequality.

51


---Page Break---
E
A Base Algorithm: NeuralUCB-WGD

Recall that for each candidate arm xi,t ∈Xt, its corruption-free expected reward is generated by
an unknown reward mapping function h(·). Following existing neural bandit approaches, we use a
neural network f(·) to approximate h(·) for estimating arm rewards. Consistent with the main text
and R-NeuralUCB, we consider the network f(·; θ) to be a fully connected (FC) network with depth
L ≥2 and width m ∈N+:

f(x; θ) := √mθLσ(θL−1σ(θL−2 . . . σ(θ1x))),

where σ(·) denotes the ReLU activation function, and the trainable weight matrices are θ1 ∈Rm×d,
θl ∈Rm×m for 2 ≤l ≤L −1, and θL ∈R1×m. For simplicity, we also denote the vectorized
parameters as
θ := [vec(θ1)⊺, vec(θ2)⊺, . . . , vec(θL)]⊺∈Rp,
with dimensionality p and randomly initialized parameters θ0. Similar to R-NeuralUCB, we also
define g(x; θ) = vec(∇θf(x; θ)) ∈Rp as the vectorized network gradients, for input x and
parameters θ.

E.1
NeuralUCB-WGD: Neural-UCB with Weighted GD

Then, we introduce the workflow of our base algorithm NeuralUCB-WGD (Algorithm 3), which
stands for Neural-UCB with Weighted GD. Here, NeuralUCB-WGD can be considered as a simplified
version of R-NeuralUCB, where all the candidate arms Xt in round t will share the same neural
network for decision making. The idea is that although we do not know which training samples are
corrupted, we can reduce the effects caused by the potential corruption instead, by paying relatively
more attention on the samples with low uncertainty for a stable training process.

Algorithm 3 Neural-UCB with Weighted GD (NeuralUCB-WGD)

1: Input: Time horizon T. GD steps J. Learning rate η. Exploration coefficient ν ≥0. Scaling
coefficient α > 0. Norm parameter S, regularization parameter λ.
2: Initialization: Initialized parameters θ0. Covariance matrix Γ0 = λI. Received records P0 = ∅.
3: for each round t ∈[T] do
4:
Observe candidate arms Xt = {xi,t}i∈[K].
5:
for each arm xi,t ∈Xt do
6:
Calculate its benefit score U(xi,t), based on reward estimation f(xi,t; θi,t−1) and the
UCB-type exploration score for arm xi,t, (E.1).
7:
end for
8:
Recommend arm based on benefit scores xt = arg maxxi,t∈Xt

U(xi,t)

.
9:
Receive arm reward rt, and update the records, such that Pt = Pt−1 ∪{(xt, rt)}. Then, save
the corresponding weight for chosen arm xt, as wt = min{1, α/∥g(xt; θt−1)/√m∥Γ−1
t−1}.

10:
Update the gradient covariance matrix Γt = Γt−1 + wt · g(xt; θt−1) · g(xt; θt−1)⊺/m.
11:
Starting from random initialization θ0, update the network parameters to θt, based on J
iterations of Gradient Descent and training data Pt.
12: end for

Arm Selection.
In each round t ∈[T], after observing the candidate arms Xt for selection, we
calculate the reward estimation and the UCB score for arm selection (lines 5-7, Algorithm 3). Similar
to R-NeuralUCB, in terms of the arm selection (line 8, Algorithm 3), we determine the chosen arm
xt ∈Xt with the highest benefit score, by xt = arg maxxi,t∈Xt

U(xi,t)

. Here, with the NTK
norm parameter S > 0 and the probability at least 1 −δ given probability parameter δ ∈(0, 1),
we formulate the benefit score U(xi,t), along with a UCB-type exploration strategy (motivated by
Lemma F.1), as

U(xi,t) = f(xi,t; θt−1) + O
 
ν

s

log det(Γt−1)

det(λI)
−2 log(δ) + λ1/2S

· ∥g(xi,t; θt−1)/√m∥Γ−1
t−1
(E.1)

where θt−1 refer to network parameters in round t before GD, which have been trained with
received records Pt−1, and λ > 0 is regularization parameter.
Different from conventional

52


---Page Break---
neural bandit works, our gradient covariance matrix is defined as Γt−1 = λI + P
τ∈[t−1] wτ ·
g(xτ; θτ−1)g(xτ; θτ−1)⊺/m. Here, to quantify the arm uncertainty level, inspired by [42], we
define the sample weight as wτ = min{1, α/∥g(xτ; θτ−1)/√m∥Γ−1
τ−1} based on the gradient vector

g(xτ; θτ−1) = vec
 
∇θf(xτ; θτ−1)

∈Rp, and it is scaled by a tunable parameter α > 0. Notice
that the arm weight wτ is inversely proportional to our UCB-type exploration score in (E.1). Since
the UCB-based exploration score can be considered to quantify reward estimation uncertainty levels
[24, 72, 86], we thus assign small weights to training samples with high uncertainty. Recall that in
(1), we apply ν to characterize the random noise ϵ. Similar to R-NeuralUCB, when this value is
unknown, we alternatively deem ν ≥0 as a tunable exploration parameter to control the exploration
intensity analogous to existing works (e.g., [86]).

After receiving the reward rt for the chosen arm xt, we update the records to Pt. The arm
context xt and its received reward rt are added to the collection Pt, along with their weight
wt = min{1, α/∥g(xt; θt−1)/√m∥Γ−1
t−1} (line 9, Algorithm 3).

Model Training.
Afterwards, we perform J iterations of GD to update the network parameters
(line 11, Algorithm 3). With Pt = {xτ, rτ}τ∈[t] up to round t, we train the model parameters θt,
through θ(j)
t
= θ(j−1)
t
−η∇θL(Pt; θ(j−1)
t
). Here, θ(j)
t , j ∈[J] are the parameters after the j-th
GD iteration, starting from randomly initialized ones θ(0)
t
= θ0, and η > 0 refers to the learning rate.
Different from existing neural bandit works (e.g., [86, 84]) which consider all received records to
be equally important with the ordinary L2 loss function, we alternatively define the weighted loss
function as

L(P; θ) =
X

(xτ ,rτ )∈P

wτ

2

f(xτ; θ) −rτ
2 + mλ

2 ∥θ −θ0∥2
2

where λ > 0 is the regularization parameter as in (E.1), and we have previously defined sample
weights wτ, τ ∈[t] associated with each chosen arm-reward pair (xτ, rτ) ∈Pt. In summary, the
intuition is that, when defining the loss function for the received records Pt = {xτ, rτ}τ∈[t] up
to round t, we aim to give extra emphasis to samples with low estimation uncertainty. Intuitively,
if samples with high uncertainty are indeed corrupted by the adversary, they are more likely to
significantly disrupt the internal decision-making process, thereby impacting the stability of reward
estimation. To mitigate this risk, even though we cannot identify which training samples are corrupted,
we conservatively assign smaller weights to high-uncertainty samples, supporting a stable and robust
GD training process.

E.2
Regret Analysis for NeuralUCB-WGD

For the theoretical analysis, different from that of R-NeuralUCB (Subsection 5.1), we focus on the
case where the corruption level C is known, a common setting in existing works [16, 42, 76, 17]. This
assumption allows us to appropriately select the parameter α in Algorithm 3 to achieve a tighter regret
bound. Additionally, we briefly discuss potential outcomes if C is unknown. Recall that our objective
(2) is to minimize the overall pseudo-regret in terms of the corruption-free expected reward over a
finite horizon of T rounds: R(T) = PT
t=1 E[er∗
t −ert], where E[ert] = h(xt) denotes the expected
corruption-free reward from the chosen arm xt, and E[er∗
t ] = maxxi,t∈Xt[h(xi,t)] represents the
expected reward of the optimal arm.

To address challenges in regret analysis, we define two sets of regression parameters corresponding
to the corrupted model and the corruption-free model. Using the corruption-free model f(·; eθ), we
derive the confidence ellipsoid around its parameters eθ, which serves as a proxy for updating the
confidence ellipsoid around the trained corrupted model parameters θ. With the updated confidence
ellipsoid and concentration results, we then finalize the regret upper bound. Here, without carefully
designing the arm weights wτ, τ ∈[t] (Algorithm 3) and structuring the regret analysis workflow,
deriving the regret upper bound under adversarial corruption settings would be impractical. The
following Theorem E.1 provides a bound on the cumulative pseudo-regret for NeuralUCB-WGD.

Theorem E.1. Given the finite horizon T ∈N+, denote S ≥
p

2˘h
⊺˘H−1˘h. Suppose prob-
ability parameter δ ∈(0, 1), network width m ≥Ω(poly(T, L, C, ˘λ−1
0 , λ−1, S−1) · log(1/δ)),
η ≤O((TmL + mλ)−1), J ≥e
O(TL/λ), and λ ≥S−2. Let f(·) be the L-layer FC network with

53


---Page Break---
width m, and set α = 1/C without the prior knowledge of ed. Then, with probability at least 1 −δ
over random initialization, NeuralUCB-WGD achieves the regret upper bound:

R(T) ≤e
O
p

edT

· e
O

ν
q

ed −2 log(δ) + λ1/2S

+ O

C ed log(1 + TK/λ)

+ O
 
C ed · λ1/2S


+ O

C ed ·
 
ν
q

ed log(1 + TK/λ) −2 log(δ)


The proof of Theorem E.1 is provided in Appendix F. The first term on the RHS represents the
corruption-independent regret upper bound, which matches the bound e
O(ed
√

T) in corruption-
free neural bandit studies [86, 84]. For terms that depend on the corruption level C, we obtain
e
O(C edλ1/2S + C ed3/2) by omitting logarithmic terms. When aligning the definition of information
gain [16] with the effective dimension ed, our results are consistent with the latest kernelized bandit
research in terms of the horizon T and effective dimension ed, given the NTK-induced RKHS and an
indefinite arm space (Corollary 7 in [16]). Additionally, we can bound the NTK norm term S by a
constant if h(·) belongs to the RKHS norm induced by NTK (Subsection B.4). The regularization
parameter λ can also be tuned to account for the NTK norm S. Different from the vanilla Neural-
UCB [86], we quantify the impact of corruption by deriving a new confidence ellipsoid around the
corrupted parameters θt−1, ensuring that the corruption-related terms remain independent of the
non-logarithmic T term.

Inspired by [42], when C is unknown to the learner, an estimated corruption level ¯C > 0 can be
utilized based on prior knowledge of the adversary. In this case, we can set the scaling parameter
α = 1/ ¯C. If the actual C ≤¯C, then the corresponding regret upper bound in Theorem 5.6 still
holds. Conversely, if C > ¯C, the regret bound will no longer hold, leading to a trivial upper bound
of R(T) ≤O(T), similar to existing works (e.g., [42]). Here, practically, one approach is to set
¯C =
√

T. For C ≤
√

T, the overall regret is then bounded by e
O(ed3/2√

T), which matches [16]
and improves upon e
O(edT) from [15]. When C >
√

T, our trivial regret bound of O(T) also
aligns with the state-of-the-art kernelized method [15] with unknown C, which has a bound of
e
O(ed
√

T + C ed
√

T) =⇒
e
O(ed
√

T + edT). Thus, the regret bound for NeuralUCB-WGD comparably
matches the latest theoretical results from the kernelized bandit research [16, 15] under the indefinite
arm space setting. While the problem definition of neural contextual bandits (1) is more general and
reduce restrictions from the reward mapping function aspect, it is also significantly distinct from the
problem definitions of linear or kernelized bandits, which makes lots of techniques from existing
works on tackling adversarial corruptions (e.g., [42, 16]) infeasible.

F
Proof of Regret Bound for NeuralUCB-WGD (Proof of Theorem E.1)

By definition in (1), recall that we aim to minimize the pseudo-regret for T rounds, denoted by

R(T) =

T
X

t=1


h(x∗
t ) −h(xt)

=

T
X

t=1
Rt

=

T
X

t=1


⟨g(x∗
t ; θ0), θ∗−θ0⟩−⟨g(xt; θ0), θ∗−θ0⟩


where xt is the chosen arm and x∗
t = arg maxxi,t∈Xt[h(xi,t)] being the optimal arm in round t.
The third equality is due to Lemma C.1. Then, we denote f(·) as the bandit model we currently
possess, which is trained with corrupted records Pt−1 up to round t, and also suppose an imaginary
corruption-free bandit model accordingly, which is trained with corruption-free records ePt−1. The
corresponding model parameters of f(·) will be denoted as θ, while the parameters of the imaginary
corruption-free model will be denoted as eθ.

Proof sketch. To begin with, we first analyze the single-round pseudo-regret Rt for t ∈[T], where the
cumulative regret is given by R(T) = P

t∈[T ] Rt. Here, we demonstrate that the single-round regret
with corruption can be upper bounded by using the updated confidence ellipsoid (Lemma F.1) around

54


---Page Break---
the corrupted network parameters θt. Since we cannot directly apply the self-regularized martingale
concentration results from existing studies [1, 86], we instead derive updated concentration results
that account for adversarial corruptions, as shown in Lemma F.2. These results are then combined to
establish the cumulative regret over T rounds. Additionally, we provide the optimal value for the
scaling parameter α based on the known corruption level C, demonstrating why setting α = 1/C
yields the desired regret bound.

F.1
Bounding the Single-round Regret

Following analogous approach as the proof of Lemma 5.3 in [86], we can transform the regret for a
single round t ∈[T] to

Rt = ⟨g(x∗
t ; θ0), θ∗−θ0⟩−⟨g(xt; θ0), θ∗−θ0⟩

≤⟨g(x∗
t ; θt−1), θ∗−θ0⟩−⟨g(xt; θt−1), θ∗−θ0⟩+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

≤max
θ∈Ct−1⟨g(x∗
t ; θt−1), θ −θ0⟩−⟨g(xt; θt−1), θ∗−θ0⟩+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7).

where the first inequality is by the gradient difference in Lemma G.4, and the bound of ∥θ∗−θ0∥
in Lemma C.1. Here, based on Lemma F.1, we have the unknown parameter θ∗∈Ct−1 with the
confidence ellipsoid Ct−1 = {θ : ∥θ −θt−1∥Γ−1
t−1 ≤γt−1/√m} induced by our currently possessed
parameters θt−1 as well as the chosen arms {xτ}τ∈[t−1]. Here, with eγt−1 being the corresponding
corruption-free radius term, we have

γt−1 = eγt−1 + α · C + (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2).

Then, based on the arm pulling mechanism, we denote the estimated arm benefit score as U(xi,t) =

f(xi,t; θt−1) + γt−1 ·
q

g(xi,t; θt−1)⊺Γ−1
t−1g(xi,t; θt−1), and we also define its alternative based
on the confidence ellipsoid as

V (xi,t) =

g(xi,t; θt−1), θt−1 −θ0

+ γt−1 ·
q

g(xi,t; θt−1)⊺Γ−1
t−1g(xi,t; θt−1)/m

= max
θ∈Ct−1


g(xi,t; θt−1), θ −θ0


based on the confidence interval Ct−1 induced by the corrupted parameters θt−1. Regarding their
distance, we can further derive |U(xi,t)−V (xi,t)| ≤O(m−1/6p

log(m)t2/3λ−2/3L3) by applying
Lemma G.5, as well as the fact that f(x; θ0) = 0 based on random initialization. It then leads to

Rt ≤max
θ∈Ct−1⟨g(x∗
t ; θt−1), θ −θ0⟩−⟨g(xt; θt−1), θ∗−θ0⟩+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

= V (x∗
t ) −⟨g(xt; θt−1), θ∗−θ0⟩+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

≤U(x∗
t ) −⟨g(xt; θt−1), θ∗−θ0⟩

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

≤U(xt) −⟨g(xt; θt−1), θ∗−θ0⟩

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

≤V (xt) −⟨g(xt; θt−1), θ∗−θ0⟩

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

≤max
θ∈Ct−1⟨g(xt; θt−1), θ −θ0⟩−⟨g(xt; θt−1), θ∗−θ0⟩

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

where the third inequality is due to the arm pulling mechanism. The second and the fourth inequality
is due to the distance between V (·) and U(·). Since we have θ∗∈Ct−1, by Holder’s inequality, it

55


---Page Break---
will further lead to

Rt ≤max
θ∈Ct−1⟨g(xt; θt−1), θ −θ0⟩−⟨g(xt; θt−1), θ∗−θ0⟩+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

+ O(m−1/6p

log(m)t2/3λ−2/3L3)

≤max
θ∈Ct−1 ∥g(xt; θt−1)∥Γ−1
t−1 · ∥θ −θ0∥Γt−1 + ∥g(xt; θt−1)∥Γ−1
t−1 · ∥θ∗−θ0∥Γt−1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

≤2γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1 + O(Sm−1/6p

log(m)t1/6λ−1/6L2/7) + O(m−1/6p

log(m)t2/3λ−2/3L3)

where the last inequality is due to the definition of confidence ellipsoid Ct−1. This gives the upper
bound for our single-round regret.

F.2
Bounding the cumulative regret

On the other hand, based on the conclusion from Subsec. F.1, the cumulative regret upper bound can
be transformed to

R(T) =

T
X

t=1


h(x∗
t ) −h(xt)

=

T
X

t=1
Rt

≤

T
X

t=1


2 · min

γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1, 1

+ O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

+ O(m−1/6p

log(m)t2/3λ−2/3L3)


≤

T
X

t=1


2 · min

γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1, 1

+ O(1)

where the last inequality is because of sufficiently large network width m that satisfies conditions in
Theorem E.1. Then, for the first term on the RHS, we can further have

T
X

t=1
2 · min

γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1, 1


≤2γT ·
q

edT log(1 + TK/λ) + (1 + γT

α ) · ed log(1 + TK/λ)

= ed log(1 + TK/λ) + 2γT ·
q

edT log(1 + TK/λ) + 1

α · ed log(1 + TK/λ)


≤ed log(1 + TK/λ) +
q

edT log(1 + TK/λ) + 1

α · ed log(1 + TK/λ)


·

2eγt−1 + 2α · C + 2(1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)


≤ed log(1 + TK/λ) +
q

edT log(1 + TK/λ) + 1

α · ed log(1 + TK/λ)


·
 
ν
q

ed log(1 + TK/λ) −2 log(δ) + λ1/2S

+ 2α · C

+ 2(1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)


≤
q

edT log(1 + TK/λ) +
ed
α log(1 + TK/λ)

· O

ν
q

ed log(1 + TK/λ) −2 log(δ) + λ1/2S + 2αC

+ O(1)

56


---Page Break---
where the first inequality is due to Lemma F.2, and the second inequality is due to Lemma F.1. The
third inequality can be derived following an analogous approach as Lemma 5.2 in [86], and the last
inequality is due to the sufficiently large network width m as mentioned in Theorem E.1, as well as
the sufficient number of GD iterations J = e
O(TL/λ).

Discussion on the value of α. Here, notice that we have the tunable parameter α > 0 nested in the
regret bound. Taking some more steps, we can have

T
X

t=1
2 min

γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1, 1


≤
q

edT log(1 + TK/λ) + 1

α · ed log(1 + TK/λ)


· O

ν
q

ed log(1 + TK/λ) −2 log(δ) + λ1/2S + 2α · C


≤eO
p

edT + 1

α · ed

·

ν
q

ed −2 log(δ) + λ1/2S + 2α · C


≤e
O
p

edT + 1

α · ed

·

ν
q

ed −2 log(δ) + λ1/2S

+ e
O

αC
p

edT + C ed


Since we have no prior knowledge of effective dimension ed, we can set α = 1

C . It will then lead to

T
X

t=1
2 min

γt−1 · ∥g(xt; θt−1)/√m∥Γ−1
t−1, 1


≤e
O
p

edT + 1

α · ed

·

ν
q

ed −2 log(δ) + λ1/2S

+ e
O

αC
p

edT + C ed


≤e
O
p

edT

·

ν
q

ed −2 log(δ) + λ1/2S

+ e
O

C ed + C ed ·
 
ν
q

ed −2 log(δ) + λ1/2S

.

Finally, summing up all the results above will give the conclusion.

F.3
Confidence ellipsoid for corrupted parameters

Lemma F.1. With the notation and conditions in Theorem E.1, train the network parameters θt−1
based on received records Pt−1. The confidence ellipsoid around the corrupted network parameters
θt−1 can be defined as

Ct−1 =

θ : ∥θ −θt−1∥Γt−1 ≤γt−1/√m


where we have the unknown parameter θ∗∈Ct−1, and we denote

γt−1 = eγt−1 + α · C + (1 −ηmλ)J/2p

t/λ + O(m−1/6p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ))

+ O(Cm−2/3p

log(m)t1/6λ−7/6L7/2) + O(Cm−1/6p

log(m)t7/6λ−13/6L9/2)

Proof. The proof follows an analogous approach as in Lemma C.12. For the imaginary corruption-free
parameters eθt−1, which are trained on corruption-free records ePt−1, we can construct the confidence
interval eCt−1 := {θ : ∥θ−eθt−1∥eΓt−1 ≤eγt−1/√m, eγt−1 > 0}, such that the unknown θ∗in Lemma

C.1 satisfies θ∗∈eCt−1. However, since we do not possess eθt−1, we need to alternatively derive the
confidence ellipsoid Ct−1 around the possessed corrupted parameters θt−1, such that θ∗∈Ct−1.

With the ellipsoid center θt−1 as well as the weighted gradient covariance matrix Γt−1 = λI +
P

τ∈[t−1] wτ · g(xτ; θτ−1) · g(xτ; θτ−1)⊺/m, we need to derive the corresponding radius. Recall
that wτ is the sample weight associated with chosen arm xτ. Here, we first recall some preliminaries
that ∥Γt−1 −eΓt−1∥F ≤O(m−1/6p

log(m)L4t7/6λ−1/6) due to Lemma G.6, as well as ∥θt−1 −
eθt−1∥2 ≤O(
p

t/(mλ)) due to Lemma G.3.

57


---Page Break---
Next, as we already have ∥θ −eθt−1∥eΓt−1 ≤eγt−1/√m, we proceed to transform the objective to

∥θ∗−θt−1∥Γt−1 ≤∥θ∗−eθt−1∥Γt−1 + ∥eθt−1 −θt−1∥Γt−1
≤∥θ∗−eθt−1∥Γt−1 + ∥eθt−1 −θt−1∥Γt−1
≤∥θ∗−eθt−1∥Γt−1−eΓt−1+eΓt−1 + ∥eθt−1 −θt−1∥Γt−1

≤∥θ∗−eθt−1∥eΓt−1 + ∥θ∗−eθt−1∥Γt−1−eΓt−1 + ∥eθt−1 −θt−1∥Γt−1

≤eγt−1/√m + ∥eθt−1 −θt−1∥Γt−1 + O(
p

t/(mλ)) · O(m−1/6p

log(m)L4t7/6λ−1/6)

≤eγt−1/√m + ∥eθt−1 −θt−1∥Γt−1 + O(m−2/3p

log(m)L4t13/6λ−2/3).

For the second term on the RHS, we first define the gradient-based regression parameters as

Γ(0)
t−1 = λI +
X

τ∈[t−1]
wτ · g(xτ; θ0) · g(xτ; θ0)⊺/m,

b(0)
t−1 =
X

τ∈[t−1]
wτ · g(xτ; θ0) · rτ/√m,

eb
(0)
t−1 =
X

τ∈[t−1]
wτ · g(xτ; θ0) · erτ/√m,

Then, with the triangular inequality, we can proceed to have

∥eθt−1−θt−1∥Γt−1 ≤∥eθt−1 −θ0 −(Γ(0)
t−1)−1b(0)
t−1/√m + (Γ(0)
t−1)−1b(0)
t−1/√m + θ0 −θt−1∥Γt−1

≤∥θt−1 −θ0 −(Γ(0)
t−1)−1b(0)
t−1/√m∥Γt−1 + ∥eθt−1 −θ0 −(Γ(0)
t−1)−1b(0)
t−1/√m∥Γt−1

≤∥θt−1 −θ0 −(Γ(0)
t−1)−1b(0)
t−1/√m∥Γt−1 + ∥eθt−1 −θ0 −(Γ(0)
t−1)−1eb
(0)
t−1/√m∥Γt−1

+ m−1∥(Γ(0)
t−1)−1 · (
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1.

(F.1)

Bounding the first two term on the RHS of Inequality F.1. Here, for the first term on the RHS, we
can individually apply Lemma D.1, by considering the auxiliary sequence in j-th iteration (j ∈[J])
with Θ(0) = θ0, as

Θ(j+1) = Θ(j) −η ·

J(0) · W ·
 
[J(0)]⊺(Θ(j) −θ0) −y

+ mλ(Θ(j) −θ0)


where the Jacobian matrix J(0) :=
 
g(x1; θ0), g(x2; θ0), . . . , g(xt−1; θ0)

∈Rp×(t−1), and vector
y ∈Rt−1 contains the received arm rewards rτ, τ ∈[t −1], while matrix W ∈R(t−1)×(t−1) is
the diagonal matrix that contains sample weights wτ, τ ∈[t −1]. We have its norm ∥W∥2 ≤1 by
definition. Here, the above sequence is expected to solve the following problem with gradient descent

min
Θ L(Θ) =
X

τ∈[t−1]

wτ

2 ·
[J(0)]⊺
τ(Θ −θ0) −rτ



2

2
+ 1

2 · mλ ·
Θ −θ0



2

2
.

As a result, following an analogous approach as in Lemma C.4 in [86], we can have ∥Θ(j) −θ0 −
(Γ(0)
t−1)−1b(0)
t−1/√m∥Γt−1 ≤(1 −ηmλ)j/2p

t/(mλ). Furthermore, by applying the conclusion of
Lemma D.1, we can have

∥θt−1−θ0−(Γ(0)
t−1)−1b(0)
t−1/√m∥2 ≤(1−ηmλ)J/2p

t/(mλ)+O(m−2/3p

log(m)L7/2t5/3λ−5/3(1+
p

t/λ)).

Similarly, for the second term in Inequality F.1, we also can apply a comparable approach by solving
the problem:

min
Θ L(Θ) =
X

τ∈[t−1]

wτ

2

[J(0)]⊺
τ(Θ −θ0) −erτ



2

2
+ 1

2mλ ·
Θ −θ0



2

2
,

58


---Page Break---
and constructing the corresponding auxiliary sequence. Following an analogous approach, it will

lead to a similar bound for the second term, such that ∥eθt−1 −θ0 −(Γ(0)
t−1)−1eb
(0)
t−1/√m∥2 ≤
(1 −ηmλ)J/2p

t/(mλ) + O(m−2/3p

log(m)L7/2t5/3λ−5/3(1 +
p

t/λ)).

Bounding the third term on the RHS of Inequality F.1. Afterwards, for the third term on the RHS,
we first have

m−1 · ∥(Γ(0)
t−1)−1 · (
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1

≤m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1

+ m−1∥
 
(Γt−1)−1 −(Γ(0)
t−1)−1
· (
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1

≤m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1

+ m−1∥(Γt−1)−1 
Γt−1 −Γ(0)
t−1

(Γ(0)
t−1)−1 · (
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1

≤m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θ0) · cτ)∥Γt−1 + O(Cm−2/3p

log(m)t7/6λ−13/6L9/2)

≤m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θτ−1) · cτ)∥Γt−1 + O(Cm−2/3p

log(m)t7/6λ−13/6L9/2)

+ m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · (g(xτ; θτ−1) −g(xτ; θ0)) · cτ)∥Γt−1

≤m−1∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θτ−1) · cτ)∥Γt−1

+ O(Cm−7/6p

log(m)t1/6λ−7/6L7/2) + O(Cm−2/3p

log(m)t7/6λ−13/6L9/2)

where the third inequality is due to Lemma G.6, and the last inequality is due to Lemma G.4. Then,
recall that the weight wτ = {1, α/∥g(xτ; θτ−1)/√m∥Γ−1
τ−1} for each chosen arm xτ. In this case,
we can further have

m−1/2∥(Γt−1)−1(
X

τ∈[t−1]
wτ · g(xτ; θτ−1) · cτ)/√m∥Γt−1

= m−1/2∥
X

τ∈[t−1]
wτ · g(xτ; θτ−1) · cτ/√m∥(Γt−1)−1

≤m−1/2
X

τ∈[t−1]
wτ|cτ| · ∥g(xτ; θτ−1)/√m∥(Γt−1)−1

≤m−1/2
X

τ∈[t−1]
wτ|cτ| · ∥g(xτ; θτ−1)/√m∥(Γτ−1)−1

≤α · C/√m.

where the first inequality is by applying the Cauchy-Schwartz inequality, while the second inequality
is due to Γτ−1 ⪯Γt−1 and Lemma G.8. The last inequality is by the definition of corruption level C.
Finally, summing up all the results will give the desired lemma.

F.4
Self-regularized Martingale Sequence with Weighted Matrix

Recall that we have the weighted gradient covariance matrix Γt−1 = λI+P

τ∈[t−1] wτ ·g(xτ; θτ−1)·
g(xτ; θτ−1)⊺/m, where g(xτ; θτ−1) is the vectorized gradient vector, and wτ ≤1 refers to the
sampled associated with chosen arm xτ.

59


---Page Break---
By existing works [1, 86], we can have the self-normalized martingale such that
X

τ∈[t]
min
g(xτ; θτ−1)/√m
2
Γ−1
τ−1, 1

≤2 log det(Γt)

det(λI).

if weights in Γt are all set to 1. However, since in our settings the gradient covariance matrix
Γt involves sample weights, we will need to further discuss the upper bound for this sequence
summation.
Lemma F.2. With the definition of γt−1 from Lemma F.1 as well as the notation and conditions in
Theorem E.1, we have the following inequality
X

t∈[T ]
min

γt−1 ·
g(xt; θt−1)/√m

Γ−1
t−1, 1

≤γT
q

edT log(1 + TK/λ) + (1 + γT

α ) · ed log(1 + TK/λ),

where Γt−1 = λI + P

τ∈[t−1] wτ · g(xτ; θτ−1) · g(xτ; θτ−1)⊺/m.

Proof. The proof of this lemma follows an analogous approach as Theorem 4.2 in [42]. Recall that
sample weights wτ ≤1, τ ∈[T]. In this case, we separately consider two scenarios when (i) wτ = 1,
then τ ∈T (T )
w=1; and (ii) the scenario when wτ < 1, then τ ∈T (T )
w<1. In this case, the original objective
will become the following inequality
X

t∈[T ]
min
g(xt; θt−1)/√m

Γ−1
t−1, 1


=
X

t∈T (T )
w=1

min
g(xt; θt−1)/√m

Γ−1
t−1, 1

+
X

t∈T (T )
w<1

min
g(xt; θt−1)/√m

Γ−1
t−1, 1


First, for the scenario where wτ = 1, τ ∈T (T )
w=1, we have
X

t∈T (T )
w=1

min
g(xt; θt−1)/√m

Γ−1
t−1, 1

≤

v
u
u
t

X

t∈T (T )
w=1

min
g(xt; θt−1)/√m
2
Γ−1
t−1, 1


≤

v
u
u
t

X

t∈T (T )
w=1

min
g(xt; θt−1)/√m
2
(Γw=1
t−1 )−1, 1

.

Here, we define an extra auxiliary matrix Γw=1
t−1
=
λI + P

τ∈T (t−1)
w=1 wτ · g(xτ; θτ−1) ·
g(xτ; θτ−1)⊺/m. Compared with the original gradient covariance matrix Γt−1, since we have
Γw=1
t−1 ⪯Γt−1 and they are both Hermitian matrices, we can derive the last inequality based on
Lemma G.8. Then, by applying Lemma G.7 and Lemma C.2, it will lead to
X

t∈T (T )
w=1

min
g(xt; θt−1)/√m

Γ−1
t−1, 1

≤

v
u
u
t|T (T )
w=1| ·
X

t∈T (T )
w=1

min
g(xt; θt−1)/√m
2
(Γw=1
t−1 )−1, 1


≤
q

|T (T )
w=1| · ed log(1 + TK/λ)

≤
q

edT log(1 + TK/λ).
Then, since we have γt−1 ≤γT , plugging in the γT will complete the proof.

Afterwards, for the second scenario when wτ < 1, τ ∈T (T )
w<1, we will have
X

t∈T (T )
w<1

min

γt−1 ·
g(xt; θt−1)/√m

Γ−1
t−1, 1

=
X

t∈T (T )
w<1


γt−1 · wt

α ·
g(xt; θt−1)/√m
2
Γ−1
t−1, 1


≤(1 + γT

α ) ·
X

t∈T (T )
w<1


wt ·
g(xt; θt−1)/√m
2
Γ−1
t−1, 1


≤(1 + γT

α ) ·
X

t∈T (T )
w<1


wt ·
g(xt; θt−1)/√m
2
(Γw<1
t−1 )−1, 1


60


---Page Break---
where the inequality is because we have γt−1 ≤γT , ∀t ∈[T].

Following the previous approach, we define the auxiliary matrix Γw<1
t−1 = λI + P

τ∈T (t−1)
w=1 wτ ·

g(xτ; θτ−1)·g(xτ; θτ−1)⊺/m. Here, since we also have Γw<1
t−1 ⪯Γt−1 and they are both Hermitian
matrices, we can derive the last inequality based on Lemma G.8. In addition, we consider an
alternative form of the original gradient vector as g′(xτ; θτ−1) = √wτ · g(xτ; θτ−1), τ ∈T (t−1)
w=1 .
In this case, the auxiliary gradient covariance matrix can be alternatively represented as Γw<1
t−1 =
λI + P

τ∈T (t−1)
w=1 g′(xτ; θτ−1) · g′(xτ; θτ−1)⊺/m, and the RHS will become

X

t∈T (T )
w<1

min

γt−1 ·
g(xt; θt−1)/√m

Γ−1
t−1, 1

≤(1 + γT

α ) ·
X

t∈T (T )
w<1

g′(xt; θt−1)/√m
2
(Γw<1
t−1 )−1, 1


≤(1 + γT

α ) · ed log(1 + TK/λ).

where the last inequality is by applying Lemma G.7 and Lemma C.2. Summing up the results will
finish the proof.

G
Lemmas for Over-parameterized FC Neural Networks

Given the input arm context vector x ∈Rd, we denote the L-layer FC neural network with width m
as

f(x; θ) = θL(

L−1
Y

l=1
Dlθl) · x,
(G.1)

where with σ being the ReLU activation, we define the intermediate hidden representations hl, l ∈
{0, . . . , L −1} as
h0 = x,
hl = σ(θlhl−1), l ∈[L −1].
and we also have the binary diagonal matrix functioning as the ReLU activation being

Dl = diag(I{(θlhl−1)1}, . . . , I{(θlhl−1)m}), l ∈[L −1].

where I(·) is the indicator function. Afterwards, the corresponding gradients will become

∇θlf(x; θ) =

(
[hl−1θL(QL−1
τ=l+1 Dτθτ)]⊺, l ∈[L −1]
h⊺
L−1, l = L.
(G.2)

Lemma G.1. There exists a positive constant C > 0 such that with probability at least 1 −δ, if
m ≥CT 4L6 log(T 2L/δ)/λ4 for each arbitrary xτ ∈S

τ∈[t] Xτ, there exists a set of parameters θ∗

such that with the neural network parameters θt−1 trained on {xτ}t−1
τ=1, we have

|⟨g(x; θ0), θ∗−θ0⟩−⟨g(x; θt−1), θ∗−θ0⟩| ≤O(Sm−1/6p

log(m)t1/6λ−1/6L2/7)

where parameters θ∗satisfy ∥θ∗−θ0∥≤S/√m, S > 0 as shown in Lemma C.1.

Proof. This lemma is based on Lemma C.1. Here, our objective can be reformed into

|⟨g(x; θ0), θ∗−θ0⟩−⟨g(x; θt−1), θ∗−θ0⟩| = ∥θ∗−θ0∥2 ·
 
g(x; θ0) −g(x; θt−1)


≤S/√m ·
 
g(x; θ0) −g(x; θt−1)


≤O(Sm−1/6p

log(m)t1/6λ−1/6L2/7),

where the first inequality is due to Lemma C.1, and the second inequality is due to Lemmas G.2, G.3,
and Lemma G.4.
Lemma G.2 (Lemma B.3 in [21] ). There exist constants {C1, C2} such that for any δ > 0, if we
have
ω ≤C1L−6(log m)−3/2,
then with probability at least 1−δ, for any ∥θ−θ0∥≤ω and for x ∈{Xt}T
t=1 we have ∥g(x; θ)∥2 ≤
C2
√

mL.

61


---Page Break---
Proof. In terms of the gradient upper bound, directly applying Lemma B.3 in [21] will give the
desired result that ∥g(x; θ)∥2 ≤O(
√

mL).

Lemma G.3 (Lemma B.2 in [86] ). For the L-layer full-connected network f trained with J iterations
of GD, there exist constants {Ci}5
i=1 ≥0 such that for δ > 0, if for all t ∈[T], η, m satisfy

2
p

t/(mλ) ≥C1m−3/2L−3/2[log(TL2/δ)]3/2,

2
p

t/(mλ) ≤C2 min{L−6[log m]−3/2, (m(λη)2L−6t−1(log m)−1)3/8},

η ≤C3(mλ + tmL)−1,

m1/6 ≥C4
p

log mL7/2t7/6λ−7/6(1 +
p

t/λ),

then, with probability at least 1 −δ, we have

∥θt −θ0∥≤2
p

t/(mλ)

∥θt −θ0 −¯Σ
−1
t ¯bt/√m∥≤(1 −ηmλ)J/2p

t/(mλ) + C5m−2/3p

log mL7/2t5/3λ−5/3(1 +
p

t/λ).

where the unweighted regression parameters are ¯Σt = λI + P

τ∈[t] g(xτ; θτ−1)g(xτ; θτ−1)⊺/m
and ¯bt = P

τ∈[t] g(xτ; θτ−1)rτ/√m.

Lemma G.4 (Theorem 5 in [3]). With probability at least 1 −δ, there exist constants C1, C2 such
that if ω ≤C1L−9/2 log−3 m, for ∥θt −θ0∥2 ≤ω, we have

∥g(x; θt) −g(x; θ0)∥2 ≤C2
p

log mω1/3L3∥g(x; θ0)∥2.

Lemma G.5 (Lemma 4.1 in [21]). There exist constants { ¯C3
i=1} ≥0 such that for any δ ≥0, if τ
satisfies that
τ ≤¯C2L−6[log m]−3/2,

then with probability at least 1 −δ, for all θ1, θ2 satisfying ∥θ1 −θ0∥≤τ, ∥θ2 −θ0∥≤τ and for
any x ∈{xt}T
t=1, we have

|f(x; θ1) −f(x; θ2) −⟨(g(x; θ2), θ1 −θ2)⟩| ≤¯C3τ 4/3L3p

m log m.

Lemma G.6. Suppose m satisfies the conditions in Theorem 5.6. Suppose the gradient matrix can be
represented by Σ = λI + P

t∈[T ] g(xi,t; θt−1) · g(xi,t; θt−1)⊺/m, with an arbitrary arm xi,t ∈Xt
from each time step t. With probability at least 1 −δ over the initialization, the following results
hold:
∥Σ∥2 ≤λ + O(TL),

∥Σ −Σ(0)∥F ≤O(m−1/6p

log(m)L4t7/6λ−1/6)

∥log det(Σ)

det(λI) −log det(Σ(0))

det(λI) ∥F ≤O(m−1/6p

log(m)L4t5/6λ−1/6),

where the gradient matrix defined based on randomly initialized parameters θ0 is Σ(0) = λI +
P

t∈[T ] g(xi,t; θ0) · g(xi,t; θ0)⊺/m.

Proof. Based on the Lemma G.2, for any t ∈[T], ∥g(xi,t; Θ0)∥2 ≤O(
√

mL). Then, for the first
inequality:

∥Σ(0)∥2 = ∥λI +

T
X

t=1
g(xi,t; Θ0)g(xi,t; Θ0)⊺/m∥2

≤∥λI∥2 + ∥

T
X

t=1
g(xi,t; Θ0)g(xi,t; Θ0)⊺/m∥2

≤λ +

T
X

t=1
∥g(xi,t; Θ0)/√m∥2
2 ≤λ + O(TL).

Then, the second and third inequalities in this lemma are the direct application of Lemma B.3 of [86].

62


---Page Break---
Lemma G.7 (Lemma 11 in [1], Lemma B.7 in [86]). Suppose a sequence of arms {x′
τ}τ∈[t],
with an arbitrary arm x′
τ ∈Xτ from each time step τ ∈[t]. The gradient matrix is denoted by
Σt = λI+P

τ∈[t] g(x′
τ; θτ−1)·g(x′
τ; θτ−1)⊺/m, where θτ−1 refer to network parameters in round
τ ∈[t]. We can have

X

τ∈[t]
min

∥g(xτ; θτ−1)/√m∥2
Σ−1
τ−1, 1
	
≤2 log det(Σt)

det(λI) .

G.1
Auxiliary Lemma

Lemma G.8 ((Corollary 7.7.4. (a) from [45]). Let A, B be Hermitian matrices of the same shape,
and suppose they are positive semi-definite. Then, we have A ⪰B iff A−1 ⪯B−1.

63


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We have included discussion for our contributions in the Abstract and Intro-
duction.
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
Justification: Please see appendix for the discussion of the limitation.
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

64


---Page Break---
Justification: Please see theoretical analysis section and appendix for details.
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
Justification: Please see appendix and our submitted source code.
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

65


---Page Break---
Answer: [Yes]

Justification: We include the source code along with our submission.

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

Justification: We have shown our way of selecting the parameters and included the parameter
study.

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

Justification: Standard deviation is given for experiment results.

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

66


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

Justification: Please see appendix where we mention our system specifications.

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

Justification: We confirm this perform conform with the NeurIPS Code of Ethics.

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

Justification: We have included a subsection in appendix discussing this.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

67


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
Justification: This paper is mainly about defensing adversarial attacks under neural bandit
settings with no new datasets.

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

Justification: We include the URLs for the datasets used instead.

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

68


---Page Break---
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: No new datasets are introduced.
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
Justification: No human subjects are involved.
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
Justification: No human subjects are involved.
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

69


---Page Break---
