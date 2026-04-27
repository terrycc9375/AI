SureMap: Simultaneous mean estimation
for single-task and multi-task disaggregated evaluation

Mikhail Khodak∗
Princeton University
mkhodak@cs.cmu.edu

Lester Mackey, Alexandra Chouldechova, Miroslav Dudík
Microsoft Research
{lmackey,alexandrac,mdudik}@microsoft.com

Abstract

Disaggregated evaluation—estimation of performance of a machine learning
model on different subpopulations—is a core task when assessing performance
and group-fairness of AI systems. A key challenge is that evaluation data is scarce,
and subpopulations arising from intersections of attributes (e.g., race, sex, age) are
often tiny. Today, it is common for multiple clients to procure the same AI model
from a model developer, and the task of disaggregated evaluation is faced by each
customer individually. This gives rise to what we call the multi-task disaggregated
evaluation problem, wherein multiple clients seek to conduct a disaggregated
evaluation of a given model in their own data setting (task). In this work we develop
a disaggregated evaluation method called SureMap that has high estimation
accuracy for both multi-task and single-task disaggregated evaluations of blackbox
models. SureMap’s efficiency gains come from (1) transforming the problem into
structured simultaneous Gaussian mean estimation and (2) incorporating external
data, e.g., from the AI system creator or from their other clients. Our method
combines maximum a posteriori (MAP) estimation using a well-chosen prior to-
gether with cross-validation-free tuning via Stein’s unbiased risk estimate (SURE).
We evaluate SureMap on disaggregated evaluation tasks in multiple domains,
observing significant accuracy improvements over several strong competitors.

1
Introduction

Evaluation is a key challenge in modern AI, with much effort spent deciding what metrics to measure,
with which methods, and on what data. This challenge is especially acute in fairness assessment,
which requires not only high-quality data to run a model and score its outputs but also demographic in-
formation for defining groups. Due to the high cost of obtaining high-quality evaluation data, the issue
of sample complexity—sample size needed to get a good performance estimate—remains salient, espe-
cially when we want to release not just one overall measure but instead to output a disaggregated eval-
uation that captures variation among demographic subpopulations of the data [Barocas et al., 2021].
For instance, we might want to assess group fairness by examining the variation in performance across
groups of users defined by intersections of the demographic attributes age, race, and sex. The naive ap-
proach of independently evaluating each group’s performance on its own data can fail because the sam-
ple sizes of intersectional groups rapidly decrease as we consider more attributes [Herlihy et al., 2024].
Recent work has shown how to improve upon naive methods by combining data from multiple sub-
populations to inform their individual performance estimates [Miller et al., 2021, Herlihy et al., 2024].

In today’s technology landscape it is common for multiple clients to procure the same model (e.g., an
automated speech recognition or language model) from an AI developer, with each client performing
a disaggregated evaluation of the same model on their own data. We refer to this problem as the
multi-task disaggregated evaluation. We formalize and study this problem, showing that one can
improve the disaggregated evaluations of individual clients by using multi-task data in the form of

∗Work done in part while at Microsoft Research and supported in part by a CMU TCS Presidential Fellowship.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).


---Page Break---
summary statistics from other clients or from the model provider. Our approach uses the (out-of-
distribution) multi-task data to set the parameters of a multivariate normal prior and then performs
maximum a posteriori (MAP) inference on the (in-distribution) client data. Formally, we model the
problem as Gaussian mean estimation and design a simple-yet-expressive additive prior that can
capture many different relationships between subpopulations. Drawing upon classical statistics, we fit
prior parameters by minimizing Stein’s unbiased risk estimator [SURE, Stein, 1981]. While motivated
by multi-task considerations, we show that our method also performs well in the single-task setting.

1.1
Contributions

1. SureMap: We introduce a method that uses SURE to tune the parameters of a well-chosen
Gaussian prior before applying MAP estimation. The prior is motivated by its attainment of a
good efficiency–expressivity tradeoff, requiring only a linear (in the number of subpopulations)
number of parameters to recover several natural baselines for disaggregated evaluation.
2. Datasets: Disaggregated evaluation has few benchmarks [Herlihy et al., 2024], so we intro-
duce new ones for both the single-task and multi-task settings, covering automated speech
recognition (ASR) and also tabular domains (with linear models and also in-context LLMs).
3. Single-task: We find that SureMap is always competitive with strong baselines from prior work,
while improving significantly in some settings with intersectional sensitive attributes.
4. Multi-task: Incorporating data from multiple clients into SureMap yields significant improve-
ments across all evaluated settings. This multi-task approach is more accurate even with just one
additional task and is the only method to consistently outperform the naive and pooling baselines.

1.2
Related work

Disaggregated evaluation is a core task in the fairness assessment of AI systems [Barocas et al.,
2021]. Past work has sought to improve estimation accuracy by combining information across
different groups, e.g., via Bayesian modeling [Miller et al., 2021], Gaussian process approximation
of loss surfaces [Piratla et al., 2021], and structured regression [Herlihy et al., 2024]. The last work
found that classical James–Stein-type mean estimation [James and Stein, 1961, Bock, 1975] is
often competitive, and so we adopt it as our first non-naive baseline. We also compare to structured
regression itself, which turns out to have a tight mathematical connection to SureMap; indeed, apart
from our use of Gaussian (ridge) rather than Laplace (lasso) priors (regularization)—as well as our
use of a more flexible tuning based on SURE rather than cross-validation—the method of Herlihy et al.
[2024] can be viewed as the discriminative counterpart to our generative approach (see §E for details).

Within the disaggregated evaluation literature we are the first to formulate and study multi-task
disaggregated evaluation. This is an important direction because (a) model providers often have their
own data or data from multiple clients that can inform the evaluation and (b) transferring information
across distributions is a key way to handle very low-sample regimes. We also contribute several
datasets that we hope will spur further development in disaggregated evaluation.

SureMap relies on applying classical mean estimation tools to quantities modeled as Gaussian means.
Notably, Miller et al. [2021] model scores via well-studied distributions—e.g., Gaussians—but since
scores are related non-linearly to metrics it is unclear if this can lead to similarly simple estimators. To
tune parameters, we use SURE, a popular statistical approach [Li, 1985, Donoho and Johnstone, 1995].
Specifically, in the empirical Bayes tradition, we use it to set the MAP estimator of a hierarchical
model. Using SURE to tune the scale of an isotropic Gaussian prior was shown to be asymptotically
(in the dimension) optimal in the case of heteroskedastic data distributions [Xie et al., 2012]. Since
disaggregated evaluation data is highly heteroskedastic due to variation in group size, this is positive
evidence for our approach, although our prior is non-isotropic and has many more variance parameters.

2
Setup

We first describe the disaggregated evaluation problem (§2.1), recast it as a Gaussian mean estima-
tion (§2.2), and motivate a multi-task variant (§2.3), all while introducing several baselines estimators.

2.1
Setting and baselines

We want to assess a predictive model p : X →Y under some distribution D over input space X and
output space Y using error measure ℓ: Y × Y →R. For example, in image classification, D is a dis-
tribution over (image, label) pairs and ℓis the 0-1 error. To simplify notation, we mainly deal with the
composite function f(z) = ℓ(y, p(x)) acting on points z = (x, y) in the product space Z = X × Y.

2


---Page Break---
In disaggregated evaluation, Z is assumed to be a union of d ≥1 disjoint subsets Zg, each associated
to some subpopulation or group g ∈[d], where we use [k] to denote the set {1, . . . , k}. As a running
example, suppose each point z ∈Z is an individual whose sex s and age a are categorical variables
with d1 and d2 possible values, respectively. Then d = d1d2 and each point has an associated index
g = (s −1)d2 + a denoting its intersection of sex s ∈[d1] and age a ∈[d2]. The task is then to
use a set S ∼Dn of n ≥1 i.i.d. samples from the distribution D to estimate the vector of true
subpopulation errors µ ∈Rd, with components µg = Ez∼D|Zg[f(z)].1 We write ˆµ(S) ∈Rd for an
estimator of µ with components denoted as ˆµg(S).

Empirically, we measure estimation accuracy, i.e., the performance of the performance estimate, via
mean absolute error (MAE) LMAE
µ
(ˆµ(S)) = 1

d∥ˆµ(S) −µ∥1, which is easy to interpret and less
sensitive to outliers than mean squared error (MSE). Our method development, however, is based
on a count-weighted version of MSE, where we denote group counts as ng = |S ∩Zg|:

Ln
µ(ˆµ(S)) = 1

d

d
X

g=1
ng(ˆµg(S) −µg)2.
(1)

We conclude the setup with two baselines. The first is the naive estimator returning group means:2

ˆµnaive
g
(S) = 1

ng

X

z∈S∩Zg
f(z).
(2)

While unbiased, ˆµnaive can perform poorly on groups with few samples. The second baseline, the
pooled estimator, returns an identical quantity—the overall sample mean—for all groups:

ˆµpooled
g
(S) = 1

n

X

z∈S
f(z) = 1

n

d
X

g=1
ng ˆµnaive
g
(S).
(3)

This estimator is generally biased (unless all group means are equal), but it can perform well in
low-sample regimes thanks to a much lower variance.

2.2
A Gaussian model for disaggregated evaluation

We use a simple but natural model to aid in the design of disaggregated evaluation methods. Specif-
ically, denoting the naive estimator by y = ˆµnaive(S) ∈Rd, we model group g’s entry yg as being
drawn from a Gaussian with (unknown) mean µg and (known) variance σ2/ng, where σ2 is shared
across groups. This reduces the problem of disaggregated evaluation, as defined in §2.1, to that of es-
timating the mean of a multivariate Gaussian with known diagonal covariance Σg,g = σ2/ng given a
single sample y ∼N(µ, Σ). Our model has many advantages in the disaggregated evaluation setting:
1. By the central limit theorem, y is asymptotically normal with mean µ and diagonal covariance Σ
for many distributions D of interest, even when the underlying data is non-Gaussian. Furthermore,
because the methods derived from our model only take y and Σ as input, they can be applied even
when the evaluated statistic is not the pointwise average assumed by the setup in §2.1, so long
as y ∼N(µ, Σ) holds asymptotically. An example of this is when yg = ˆµnaive
g
(S) corresponds
to the area under the ROC curve (AUC) computed over group g’s data S ∩Zg [Lehmann, 1951];
we demonstrate SureMap’s applicability to AUC empirically in §G (Figures 10 & 12).
2. While a shared variance is a strong assumption, it is perhaps the simplest way of incorporating the
inductive bias that Σg,g will be highly correlated with the inverse of ng, the number of samples
from group g. In practice, we set σ2 to be the pooled estimate
1
n−d
Pd
g=1
P

z∈S∩Zg(f(z) −yg)2.
3. Gaussian mean estimation is one of the best-studied problem in statistics, with numerous
well-tested baselines and approaches for developing new methods. In particular, we make
significant use of the classic James–Stein approach [James and Stein, 1961, Bock, 1975],
SURE [Stein, 1981], and empirical Bayesian estimation methods [?].
4. In the multi-task setting, clients are likely to be unwilling to share their actual data but possibly
more willing to share group summary statistics. Thus methods developed for our Gaussian
model—which only require the group means y, group counts n, and an estimate of σ2—will
be more broadly applicable than methods that act directly on the dataset S ⊂Z.

1We assume existence of the first (and, in §2.2, of the second) moments of f(z), z ∼D | Zg, across all g ∈[d].
2If ng = 0 we let ˆµnaive
g
fall back to pooling (Eq. 3), i.e., ˆµnaive
g
= ˆµpooled
g
; in the next section, assume ng > 0 ∀g.

3


---Page Break---
This model can also be naturally extended—using a non-diagonal covariance Σ—to disaggregated
evaluation with non-disjoint groups, e.g., to simultaneously estimate performance both for all women
and for only women in their forties. In the interest of brevity we focus on the disjoint group setting.

2.3
The multi-task setting

We can easily extend this model to study multi-task disaggregated evaluation, in which for each task
t = 1, . . . , T (e.g., a client of the model provider) we observe a set St ⊂Z of nt samples from
the task distribution Dt. The goal is then to output T vectors ˆµt that are close on-average to the
tasks’ subpopulation errors µt;g = Ez∼Dt[f(z)|z ∈Zg]. Converting to our Gaussian model, we
observe T vectors yt ∼N(µt, Σt)—where we set Σt;g,g = σ2/nt;g for some globally shared σ2 and
task-specific group count vectors nt ∈Zd
≥0—and must output T mean estimates ˆµt({yt}T
t=1) ∈Rd.

We consider two natural multi-task baseline estimators. The first is the global naive estimator (or
global estimator for short), which combines the data from all tasks, computes a single global vector
of group averages, and uses it as the estimate for each task:

ˆµglobal
t
({St}T
t=1) = ˆµnaive
 T[

t=1
St

!

or
ˆµglobal
t
({yt}T
t=1) =

 T
X

t=1
Σ−1
t

!−1
T
X

t=1
Σ−1
t yt.
(4)

While low variance, the global estimator ignores variation across tasks. Our second baseline—the
multi-task offset estimator—shifts the global estimate on each task to ensure that the task’s pooled
mean is preserved (thus accounting for the variation in pooled means across individual tasks):

ˆµoffset
t
({yt}T
t=1) = θ + ˆµpooled(yt −θ),
where
θ = ˆµglobal
t
({yt}T
t=1).
(5)

3
Methods

In the last section we reduced the problem of disaggregated evaluation to that of estimating a mean
µ ∈Rd given a sample y ∼N(µ, Σ), where Σ is known and diagonal. We now design a method,
SureMap, for the latter problem. Our technical approach involves the following two steps:
1. Choosing a parameterized mean estimator. We use the MAP estimator under a multivariate
normal prior that we design specifically for intersectional subpopulations.
2. Tuning the estimator’s hyperparameters. We use SURE to estimate the quality of our estimator,
which we then optimize over the choice of hyperparameters using the L-BFGS-B algorithm.

3.1
Designing a parameterized estimator

As mean estimation is a vast area, we use three criteria for designing an estimator: it should
(1) dominate baselines such as ˆµnaive and ˆµpooled; (2) have relatively few hyperparameters; and
(3) handle heteroskedasticity stemming from variation in group sizes. One natural source of candidates
are James–Stein-type shrinkage estimators: the original James–Stein estimator famously dominates
ˆµnaive in MSE and has no hyperparameters to tune [James and Stein, 1961], satisfying our first two
desiderata. Furthermore, while James and Stein [1961] assumed an isotropic Σ, subsequent estimators
such as the following variant of an estimator due to Bock [1975] do handle heteroskedastic Σ:3

ˆµBock
θ
(y) = θ +

1 −
d −2
(y −θ)⊤Σ−1(y −θ)



+
(y −θ),
(6)

where (·)+ = max{·, 0}, and θ ∈Rd is a default estimate towards which y is shrunk.4

However, empirically we find that this often underperforms the pooled estimator in low-sample
regimes; further, the form of ˆµBock
θ
shows that the amount of shrinkage towards θ is the same for each
coordinate g ∈[d], despite intuition suggesting that we should shrink less for groups g with more
samples. Corrections to this tend to be involved and difficult to generalize [Efron and Morris, 1973].

We thus turn to a different family of well-known Gaussian mean estimators: those that return the
mode of the posterior distribution assuming µ is sampled from the conjugate prior N(θ, Λ) with
mean θ ∈Rd and positive-definite covariance Λ ∈Rd×d (e.g., Gelman et al. [2014, Equation 3.12]):

ˆµMAP
θ,Λ (y) = (Λ−1 + Σ−1)−1(Λ−1θ + Σ−1y).
(7)

3Feldman et al. [2014] show that using d in the numerator instead of Bock’s Tr Σ

∥Σ∥performs better; in §B we
show that their modified form can be derived by minimizing an upper bound on the Σ−1-weighted MSE.
4We typically use θ = 0, but in §B we derive a variant shrinking towards θ = ˆµpooled(y).

4


---Page Break---
Algorithm 1: Single-task SureMap.
(For multi-task SureMap see §D.)
Input: target f : Z →R, samples S ⊂Z,
partition {Zg}d
g=1 of Z, each group g
an intersection of k ∈Z>0 attributes
// compute naive group means
for group g ∈[d] do

ng ←|S ∩Zg|
yg ←
1
ng
P

z∈S∩Zg f(z)

// estimate group variances
σ2 ←
1
|S|−d
Pd
g=1
P

z∈S∩Zg(f(z) −yg)2

Σ−1 ←diag(n)/σ2

// compute auxiliary matrix
Method A(τ):

// compute prior covariance
(matrices CA are defined in
§3.1.1 and §C.1)
Λ ←P

A⊆[k] τ 2
ACA
Output: (Id + ΛΣ−1)−1

// optimize SURE using L-BFGS-B
ˆτ ←arg min

τ∈R2k
≥0
∥A(τ)y∥2
Σ−1 −2 Tr(A(τ))

// estimate group means using MAP
Output: y −A(ˆτ)y

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

F,<25

M,<25

F,25-64

M,25-64

F,>64

M,>64

+ ω 2

sex
+ ω 2

age

lRL/Gq36EX+EneBOvXpysOfhqGCiqurumK0ylMOh5L87Q8Mjo2PjEZG1qemZ2rj6/cGqSTHNo8UQm+jxkBqSIoYUCJZynGpgKJZyF1welfnYD2ogkPsF+Ch3FLmMRCc7QUt360sZdgCzrBgi3qFVu4HaTXUJxsdWtu17DGxT9C/wKuKSqo+68Uw96Cc8UxMglM6bteyl2cqZRcAlFLcgMpIxf2/1tC2OmwHTywQ0FXbVMj0aJti9GOmC/T+RMGdNXoe1UDK/Mb60k/9PaGUZ7nVzEaYQ8y+jKJMUE1oGQntCA0fZt4BxLexfKb9imnG0sf1wCZW9oQdRsO76eVA6hVEehIq6flEpTauUVF5SRc1m6P9O7C843Wr4O42d423uV+lOUGWyQpZIz7ZJU1ySI5Ii3ByTx7I3lynp1X5815/2odcqZRfKjnI9Pv2WuCA=</latexit>+ ω 2

sex,age
!(ω) = ω 2

→

Figure 1: Matrices CA ∈{0, 1}d×d, whose linear
combination defines the covariance Λ(τ) of our
additive intersectional effects prior (Eq. 9). In this
example, there are two categories for sex (F,M) and
three for age (<25,25-64,>64), yielding d = 6
groups. Shaded squares are 1s, unshaded are 0s.

MAP naturally handles heteroskedasticity by weighting low-variance coordinates g more heavily,
satisfying our third criterion; however, its most general form has O(d2) hyperparameters, violating
the second. We next use problem structure to restrict the number of hyperparameters needed to define
Λ while still allowing ˆµMAP
θ,Λ to express every baseline introduced in §2.1, satisfying our first criterion.

3.1.1
An additive intersectional effects prior
For brevity, we build up our estimator somewhat informally; a full description is in §C.1. We return
to our simple example where each group g ∈[d] corresponds to an intersection (s, a) ∈[d1] × [d2]
of two attributes: a sex s ∈[d1] and an age a ∈[d2]. A simple prior that additively incorporates
individual attribute effects into intersectional group means is the following:

µg = τ∅ζ + τsexζsex
a
+ τageζage
a
+ τsex,ageζg + θg
(8)

where θ ∈Rd and τ∅, τsex, τage, τsex,age ∈R≥0 are hyperparameters, ζ ∼N(0, 1) is a scalar effect
shared across all groups, the vector ζsex ∈Rd1 has its sth entry ζsex
s
∼N(0, 1) shared by all groups
g whose sex is s, the vector ζage ∈Rd2 has its ath entry ζage
a
∼N(0, 1) shared by all groups g whose
age is a, and the vector ζ ∼N(0d, Id) contains an independent noise term ζg for each group g.

The hyperparameter τ∅quantifies how much we expect all of the means to be shifted (by a shared
positive or negative value) from the default θ. Hyperparameters τsex and τage express how large we
expect contributions of sex and age alone to be towards the means. And finally, non-zero τsex,age gives
the prior flexibility to model heterogeneity across all intersectional groups.

Given the vector of hyperparameters τ = (τ∅, . . . , τsex,age) ∈R4
≥0, the prior can be written more
compactly as µ ∼N(θ, Λ(τ)), where the covariance is

Λ(τ) =
X

A⊆{sex,age}
τ 2
ACA = τ 2
∅C∅+ τ 2
sexCsex + τ 2
ageCage + τ 2
sex,ageCsex,age
(9)

for matrices CA ∈{0, 1}d×d s.t. each entry CA;g,h is one iff groups g and h agree on the attributes
included in A. In particular, we have that C∅= 1d×d is the all-ones matrix, the entries Csex;g,h of
Csex ∈{0, 1}d×d are one iff groups g and h share the same sex attribute, the matrix Cage is analogous,
and Csex,age = Id is the d × d identity. This structure is visualized in Figure 1.

5


---Page Break---
3.1.2
Efficiency and expressivity

As detailed in §C.1, this prior can be naturally extended to any number of attributes k using a
covariance matrix Λ(τ) ∈Rd×d specified by a vector τ ∈R2k
≥0 of 2k hyperparameters. Since
k ≤⌊log2 d⌋, the total number of hyperparameters (including θ ∈Rd) is d + 2k = O(d), which is
much smaller than the O(d2) complexity of the general case. We can further reduce this by fixing the
entries of θ, constraining them to be identical, or setting them using external (e.g., multi-task) data.

Despite this reduction in hyperparameters, we can show that for a suitable choice of τ, the estimator
ˆµMAP
θ,Λ(τ) recovers many estimators of interest, including the naive estimator and the (possibly offset)
pooled estimator (see §C.2). This means that MAP with our structured prior should be able to
outperform all four baselines from the previous section, if appropriately tuned.

3.2
Tuning by minimizing expected risk

Having specified a parameterized estimator, there remains the question of setting its parameters θ
and τ. One might want to treat this as a hyperparameter tuning problem and use a data-splitting
approach; however, the dimensionality of the problem makes standard techniques either expensive
or noisy, and data splitting introduces additional randomness and design decisions into an already
data-poor environment. We instead make continued use of our Gaussian assumption and turn to
SURE, which given a differentiable estimator ˆµ : Rd →Rd returns an unbiased estimate of its
weighted MSE Ln
µ using sample data y ∼N(µ, Σ):

ˆRn
µ(y) = σ2

d


∥ˆµ(y) −y∥2
Σ−1 −d + 2∇y · ˆµ(y)

,
(10)

where given any W ≻0d×d we denote ∥x∥2
W = ⟨x, Wx⟩∀x ∈Rd. Using SURE we can now tune
the parameters of ˆµ by minimizing ˆRn
µ(y) in a manner similar to empirical risk minimization.

3.2.1
Single-task SureMap

In the single-task setting, we fix θ = 0d and tune the variance parameters τ ∈R2k
≥0. Letting
A(τ) = (Λ−1(τ) + Σ−1)−1Λ−1(τ), we define the single-task SureMap estimator as

ˆµSM(y) = ˆµMAP
0d,Λ(ˆτ)(y) = (Id −A(ˆτ))y
(11)

for ˆτ = arg min
τ∈R2k
≥0
∥A(τ)y∥2
Σ−1 −2 Tr(A(τ)).
(12)

The optimization problem in the second line comes from substituting ˆµMAP
0d,Λ(ˆτ) into SURE (Eq. 10).
It is nonconvex, but we find that it can be quickly solved to sufficient accuracy with L-BFGS-B [Byrd
et al., 1995], a standard method for bound-constrained optimization of differentiable functions.

3.2.2
Multi-task SureMap

To generalize SureMap to the multi-task setting we propose to specify ˆθ and ˆτ by minimizing SURE
aggregated across tasks, i.e., PT
t=1 ˆRnt
µt(yt). While setting both parameters via direct optimization
of this objective is the most straightforward approach, we find that it performs worse than single-task
SureMap when there are only a few tasks (T ≤5) and rarely improves significantly above the
multi-task global and offset estimators. This can be explained by observing the few-task limit—i.e.
T = 1—in which case optimizing the aggregated SURE objective results in setting ˆθ = y1 and thus
makes the multi-task estimator equivalent to the naive estimator.

We find that a better approach is to treat the choice of ˆθ as its own simultaneous mean estimation
problem and apply the SureMap approach to it. In particular, our model yt ∼N(µt, Σt) and our prior
µt ∼N(θ, Λ) imply that the samples yt ∼N(θ, Λ+Σt) have mean θ and known covariances (apart
from tuning parameters). Therefore, the MAP estimator of θ itself given a hyperprior θ ∼N(0d, Γ)
with covariance Γ ≻0 will have the form ˆθ =
 
Γ−1 + PT
t=1(Λ + Σt)−1−1 PT
t=1(Λ + Σt)−1yt.

To reduce the number of tuning parameters, we use a prior of the same form as before by specifying
Γ = Λ(υ) for υ ∈R2k
≥0, i.e., the same structured covariance as described in §3.1.1 but with separately
tuned parameters (see §3.1.1 and §C.1). Substituting the meta-level MAP estimator of θ into the
MAP estimator of µt and tuning the parameters τ and υ by optimizing the sum of SUREs (Eq. 10)

6


---Page Break---
across tasks yields our multi-task SureMap estimator (see §C.3 for details):
ˆµSM
t ({yt}T
t=1) = ˆµMAP
ˆθ(ˆτ,ˆυ),Λ(ˆτ)(yt) = yt + At(ˆτ)(ˆθ(ˆτ, ˆυ) −yt)
(13)

for ˆθ(τ, υ) =

T
X

t=1
Mt(τ, υ)yt,
At(τ) =

Λ−1(τ) + Σ−1
t
−1
Λ−1(τ),

Mt(τ, υ) =

Λ−1(υ) +

T
X

s=1
(Λ(τ) + Σs)−1
−1
(Λ(τ) + Σt)−1,

ˆτ, ˆυ =arg min

τ,υ∈R2k
≥0

T
X

t=1

hAt(τ)(ˆθ(τ, υ) −yt)
2
Σ−1
t
+ 2 Tr
 
At(τ)(Mt(τ, υ) −Id)
i
.
(14)

The optimization problem on the last line can again be approximately solved using L-BFGS-B.

3.3
Limitations

Various modeling assumptions impact performance of SureMap. For instance, the Gaussian error
assumption is less appropriate when errors are heavy-tailed. In §G, Figure 9, we consider MSE as an
example of a target metric with heavy-tailed observation errors. We find that SureMap still performs
well, but is no longer superior to previous approaches. One avenue for improvement would be to use
a variance-stabilizing transformation (e.g., Hawkins and Wixley [1986]) prior to applying SureMap.

Note that SureMap achieves its improved accuracy by shrinking naive estimates towards a less granular
estimator (e.g., a pooled mean). As a result the estimation is biased towards less disparity, which
could lead to overly optimistic conclusions about fairness. For this reason it is extremely important to
examine not just the point estimates, but also confidence intervals. These can be obtained, for example,
by viewing SureMap as a regression approach (§E) and leveraging inference techniques for regression.

4
Datasets

We evaluate our approach in several representative settings for disaggregated evaluation, including
two tabular settings appearing in previous works [Miller et al., 2021, Herlihy et al., 2024, Liu et al.,
2024], and three new settings: a multi-task tabular setting based on state-level U.S. census data and
both a single-task and a multi-task ASR evaluation setting.

4.1
Tabular datasets

We consider three tabular datasets, two for the single-task and one for the multi-task setting, covering
two important domains where fairness concerns can arise: healthcare records and demographic data.
While we focus on the classification task and 0-1 error, in §G we also report results for regression
(Figures 8 & 9) and for classification with AUC as the target metric (Figures 10 & 12).

Diabetes. This is a tabular dataset of Strack et al. [2014], containing around 100K patient records with
six race, two sex, and three age categories. We evaluate a logistic regression classifier trained to predict
patient disposition after a hospital stay (discharged or otherwise). The target metric is the 0-1 error.

Adult. We use the classic Adult census dataset [Kohavi, 1996] to evaluate performance of an
in-context LLM learner—specifically llama-3-70b—in predicting whether a person makes more
or less than $50K after being provided with eight examples via a modification of the prompt template
of Liu et al. [2024]. The target metric is the 0-1 error, disaggregation is by race, sex, and age.

State-Level ACS (SLACS). This is a tabular dataset for multi-task setting derived from the census
data for all U.S. states and Puerto Rico assembled by Ding et al. [2021]. Each datapoint corresponds to
a person in one of nine race and two sex categories; we consider three age categories: below 25, 25–64,
and over 64. The underlying task is to classify each person as earning either more or less than $50K.
We train a regularized logistic model on the data from California, and seek to evaluate its performance
on the other 50 states/territories, which comprise the tasks. The target metric is the 0-1 error.

4.2
ASR datasets

We also introduce both single-task and multi-task speech recognition datasets, based on applying
the popular Whisper ASR model [Radford et al., 2023]—specifically whisper-tiny—on the En-
glish part of the Common Voice (CV) dataset [Ardila et al., 2020], which contains utterances from
individuals in one of nine age and three sex categories.

7


---Page Break---
Figure 2: Single-task evaluations on Diabetes (top, disaggregating by race, sex, and age), Adult (mid-
dle, disaggregating by race, sex, and age), and Common Voice (bottom, disaggregating by sex and
age). The MAE is averaged across all groups (left), large groups (center), or small groups (right).
Large and small groups are defined as above and below median group size.

Common Voice. This is a single-task dataset obtained by combining the validation and test partitions
of the CV dataset. We calculate the word-error rate (WER) across all the utterances of each individual,
which becomes the target metric to be predicted.

Common Voice Clusters (CVC). This is a multi-task ASR dataset. To construct it, we first cluster
the utterances in the train partition of the CV dataset into 20 clusters by applying k-means to the sums
of GloVe word embeddings [Pennington et al., 2014] of their corresponding text strings. To model
task relatedness, we then randomly reassign each utterance to a random cluster with probability
α ∈[0, 1]. The resulting clusters are the tasks. The target metric is the word-error rate (WER) across
all the utterances of each individual in a given cluster. In most experiments we use α = 1

2, but
we also investigate what happens when α varies between zero, i.e., the original clusters, and one,
corresponding to identically distributed tasks.

5
Evaluation

Our main metric is MAE relative to a ground truth vector, which we take to be the mean of all
available data for each subpopulation g ∈[d], except those with fewer than 40 samples. In our main
results we subsample with replacement from the entire dataset at different rates and track performance
as a function of the sizes of the resulting datasets. To obtain 95% confidence intervals we conduct 200
and 40 random trials at each subsampling rate in the single-task and multi-task settings, respectively.

5.1
Single-task

We compare SureMap to the naive (Eq. 2) and pooled (Eq. 3) baselines, as well as to the Bock
estimator with shrinkage towards the pooled estimator (Eq. 17) and the structured regression
estimator of Herlihy et al. [2024]. On both Diabetes and Adult, SureMap significantly outperforms
all competitors (Figure 2, top & middle), the greatest improvement is on subpopulations with limited
data. In §G, we consider a regression variant of Diabetes and observe similar results (Figure 8).
On the Common Voice task, SureMap performs roughly similarly to Bock, while outperforming
structured regression at some subsampling rates (Figure 2, bottom); here again the gains are driven

8


---Page Break---
Figure 3: Multi-task evaluations on SLACS (top, disaggregating by race, sex, and age) and CVC
(bottom, disaggregating by sex and age). Left: Performance across different subsampling rates. Right:
Multiplicative improvement in MAE over naive estimator on individual tasks; subsampling rate=0.1.

by better performance on small groups. Pooling performs best when data is extremely limited, but
it is not competitive with even modestly more data, and also underperforms on small groups.

5.2
Multi-task

In the multi-task setting, we use all the single-task methods as baselines while adding multi-task (MT)
ones, including MT global (Eq. 4), MT offset (Eq. 5), and an MT extension of Bock (Eq. 6) in which
θg is set using the average across the group g data on all other tasks. We first consider the SLACS
task, for which Figure 3 (top left) shows that MT SureMap significantly outperforms other methods in
the low-data regime while matching the best one (MT Bock) in the high-data regime. At subsampling
rate 0.1, Figure 3 (top right) also shows that using multi-task data leads to improvement on all but two
of the fifty tasks, that the reduction in MAE over the naive estimator on a typical task is 2x, and that
this improvement only loosely correlates with the task’s ground truth distance from the multi-task
median. On the other hand, it also shows that while MT Bock’s improvements are typically smaller,
on SLACS it improves performance for every state (including the two where MT SureMap is worse).

On the CVC task, the MT offset baseline is the most competitive, except at the lowest subsampling rate
where pooling is better and at higher subsampling rates where it stops improving with additional data
(Figure 3, bottom left). SureMap outperforms it and all other methods across all subsampling rates
and its advantage is greatest in low-data regimes, where it even outperforms pooling. In the task-level
evaluation in Figure 3 (bottom right) we see that on every task, MT SureMap attains an improvement
of 2–3.5x over the naive baseline and almost always outperforms MT Bock. Furthermore, the latter
performs substantially worse on tasks whose ground truth vectors are far away from the multi-task
center while MT SureMap is not affected.

5.3
Ablations

We next look at how the degree of included intersectional effects and multi-task structure affect
performance. In Figure 4 (left), we evaluate utility of including higher-order interactions in the struc-
tured prior. We implement SureMap variants with up to ℓth-order interactions by setting τA for |A| ∈
{ℓ+1, . . . , k−1} to zero (but not for |A| = k). We observe that including zeroth-order effects (ℓ= 0)
in single-task SureMap (i.e., shrinkage to pooling) improves upon the usual single-parameter Gaussian
prior (ℓ= −1) in low-data regimes. Adding first-order effects (ℓ= 1) leads to substantial further im-

9


---Page Break---
Figure 4: Left: Comparison of SureMap variants on SLACS. The SureMap (ℓ) variant sets to zero the
entries of τ corresponding to interactions of size > ℓ(except for the highest-order interactions). Right:
Evaluation of different methods as the interpolation coefficient that defines the CVC tasks is varied.

Figure 5: Performance as the number of tasks varies, evaluated on SLACS (left) and CVC (right).

provement, but second-order effects (ℓ= 2) make performance slightly worse. A similar effect can be
observed among the multi-task variants, except there is no loss (but also no gain) to using the highest-
order variant (ℓ= 2). Overall, this study suggests that using the full-order SureMap is a reasonable
default but that most of the method’s effectiveness comes from zeroth- and first-order effects.

Figure 4 (right) tracks performance as the task-similarity parameter defining the CVC task is varied.
MT SureMap outperforms all methods at all settings and is also not as strongly affected by the task
similarity, at least as it is defined for the CVC data. This suggests that the structured prior we use
may be useful even if the dataset means are quite different but the underlying evaluation problem (in
this case estimating WER of ASR models) is the same.

Lastly, in Figure 5 we study how the number of tasks affects multi-task performance. On both SLACS
and CVC, MT SureMap outperforms all single-task baselines (the best one being the single-task
SureMap) at T = 2 tasks, i.e., it can take advantage of even very little external information. In contrast,
on CVC, the competitor multi-task methods (e.g., MT offset and MT Bock) do not even outperform
single-task methods until T ≥5 tasks. These results demonstrate that, unlike these comparators, multi-
task SureMap can be confidently used even when only one additional client’s worth of data is available.

6
Conclusion

We have introduced SureMap, a disaggregated evaluation approach, which combines MAP estima-
tion under a structured Gaussian prior with hyperparameter tuning via SURE. SureMap achieves
substantial empirical improvements over strong baselines in both single-task and multi-task settings.
Valuable future directions include improving robustness to heavy-tailed data and developing multi-
task methods that can handle client privacy concerns. More broadly, we hope our work will have
a positive impact by allowing model users to more accurately identify fairness-related harms, the
first step towards mitigating them. However, using SureMap and any other disaggregated evaluation
approach must be done with care, so as to not risk overconfidence in a model’s fairness.

10


---Page Break---
Acknowledgments

We thank Nicolas Le Roux for discussions and feedback at the beginning of this effort.

References

Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben
Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common Voice: A massively-
multilingual speech corpus. In Proceedings of the 12th Language Resources and Evaluation
Conference, 2020.

Solon Barocas, Anhong Guo, Ece Kamar, Jacquelyn Krones, Meredith Ringel Morris, Jennifer Wort-
man Vaughan, W. Duncan Wadsworth, and Hanna Wallach. Designing disaggregated evaluations
of AI systems: Choices, considerations, and tradeoffs. In Proceedings of the 2021 AAAI/ACM
Conference on AI, Ethics, and Society, 2021.

Mary Ellen Bock. Minimax estimators of the mean of a multivariate normal distribution. The Annals
of Statistics, 3(1):209–218, 1975.

James R. Bunch, Christopher P. Nielsen, and Danny C. Sorensen. Rank-one modification of the
symmetric eigenproblem. Numerische Mathematik, 31:31–48, 1978.

Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for
bound constrained optimization. SIAM Journal on Scientific Computing, 16(5):1190–1208, 1995.

Corinna Cortes and Mehryar Mohri. AUC optimization vs. error rate minimization. In Advances in
Neural Information Processing Systems, 2003.

Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring Adult: New datasets for fair
machine learning. In Advances in Neural Information Processing Systems, 2021.

David L. Donoho and Iain M. Johnstone. Adapting to unknown smoothness via wavelet shrinkage.
Journal of the American Statistical Association, 90(432):1200–1224, 1995.

Bradley Efron and Carl Morris. Stein’s estimation rule and its competitors–an empirical Bayes
approach. Journal of the American Statistical Association, 68(341):117–130, 1973.

Sergey Feldman, Maya R. Gupta, and Bela A. Frigyik. Revisiting Stein’s paradox: Multi-task
averaging. Journal of Machine Learning Research, 15:3621–3662, 2014.

Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin.
Bayesian Data Analysis. Electronic Edition, 2014.

Douglas M. Hawkins and R. A. J. Wixley. A note on the transformation of chi-squared variables to
normality. The American Statistician, 40(4):296–298, 1986.

Christine Herlihy, Kimberly Truong, Alexandra Chouldechova, and Miroslav Dudík. A structured
regression approach for evaluating model performance across intersectional subgroups. In Pro-
ceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 2024.

Roger A. Horn and Charles R Johnson. Matrix Analysis. Cambridge University Press, second edition,
2013.

Willard James and Charles M. Stein. Estimation with quadratic loss. In Proceedings of the Fourth
Berkeley Symposium on Mathematical Statistics and Probability, 1961.

Ron Kohavi.
Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid.
In
Proceedings of the Second International Conference on Knowledge Discovery and Data Mining,
1996.

Sören Laue, Matthias Mitterreiter, and Joachim Giesen. Computing higher order derivatives of matrix
and tensor expressions. In Advances in Neural Information Processing Systems, 2018.

11


---Page Break---
Erich Leo Lehmann. Consistency and unbiasedness of certain nonparametric tests. Annals of
Mathematical Statistics, 22(1):165–179, 1951.

Ker-Chau Li. From Stein’s unbiased risk estimates to the method of generalized cross validation. The
Annals of Statistics, 13(4):1352–1377, 1985.

Jun S. Liu. Siegel’s formula via Stein’s identities. Statistics & Probability Letters, 21:247–251, 1994.

Yanchen Liu, Srishti Gautam, Jiaqi Ma, and Himabindu Lakkaraju. Confronting LLMs with traditional
ML: Rethinking the fairness of large language models in tabular classifications. In Proceedings of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, 2024.

Andrew C. Miller, Leon A. Gatys, Joseph Futoma, and Emily Fox. Model-based metrics: Sample-
efficient estimates of predictive model subpopulation performance. In Proceedings of the 6th
Machine Learning for Healthcare Conference, 2021.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global vectors for word
representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing, 2014.

Vihari Piratla, Soumen Chakrabarti, and Sunita Sarawagi. Active assessment of prediction services
as accuracy surface over attribute combinations. In Advances in Neural Information Processing
Systems, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Ro-
bust speech recognition via large-scale weak supervision. In Proceedings of the 40th International
Conference on Machine Learning, 2023.

Sidney Siegel. Nonparametric Statistics for the Behavioral Sciences. McGraw-Hill, 1956.

Charles M. Stein. Estimation of the mean of a multivariate normal distribution. The Annals of
Statistics, 9(6):1135–1151, 1981.

Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J.
Cios, , and John N. Clore. Impact of HbA1c measurement on hospital readmission rates: Analysis
of 70,000 clinical database patient records. BioMed Research International, 2014, 2014.

Qing Wang and Alexandria Guo. An efficient variance estimator of AUC and its applications to
binary classification. Statistics in Medicine, 39:4107–4349, 2020.

Hilde Weerts, Miroslav Dud´k, Richard Edgar, Adrin Jalali, Roman Lutz, and Michael Madaio.
Fairlearn: Assessing and improving fairness of AI systems. Journal of Machine Learning Research,
24:1–8, 2023.

Xianchao Xie, S. C. Kou, and Lawrence D. Brown. SURE estimates for a heteroscedastic hierarchical
model. Journal of the American Statistical Association, 107(500):1465–1479, 2012.

12


---Page Break---
A
Notation

• For p ∈[1, ∞] we use ∥·∥p to denote the p-norm on Rd.
• We use ∥·∥to denote the spectral norm on Rm×n.
• For any positive semi-definite matrix A ⪰0d×d we use the notation ∥·∥A : Rd →R≥0 to denote
the vector norm ∥x∥A =
p

⟨x, Ax⟩= ∥
√

Ax∥2 on x ∈Rd.
• For any positive integer k we use [k] to denote the set {1, . . . , k}.
• For any set S we use 2S to denote its powerset.
• For any vector a ∈Rd we use ai to denote its ith entry. If a is only defined as an expression then
we will abuse notation and use (a)i to refer to its ith entry.
• For any vector a ∈Rk and any subset S ⊂[k] with elements s1 < · · · < sm we define
aS = (as1
· · ·
asm) to be the vector of the entries of a whose indices correspond to the
elements of S sorted in ascending order.
• For any subset S ⊂[k] we assume×s∈S iterates over the elements in ascending order.
• For any matrix A ∈Rm×n we use Ai,j to denote its (i, j)th entry and Ai,: its ith row. If A is only
defined as an expression then we will abuse notation and use (A)i,j to refer to its (i, j)th entry.
• For any k-tensor Z ∈R×k
a=1 da with dimensions d1, . . . , dk ∈Z>0 and any vector c ∈×
k
a=1[da]
of indices we use Zc to refer the the (c1, . . . , ck)th entry of Z.
• We use 0m, 1m ∈Rm and 0m×n, 1m×n ∈Rm×n to refer to all-zero and all-one vectors and
matrices, and In to refer the n × n identity matrix.

B
Heteroskedastic James–Stein-type shrinkage estimation

For reference we state a weighted version of Stein’s unbiased risk estimate (SURE) [Stein, 1981]:

Lemma B.1. Suppose y ∼N(µ, Σ) for mean µ ∈Rd and diagonal p.s.d. covariance Σ ∈Rd×d,
and consider a function ˆµ : Rd →Rd s.t. for every i ∈[d] the function ˆµi is almost differentiable
in yi and we have E∥∇y · ˆµ(y)∥1 < ∞. Then for any diagonal matrix W ∈Rd×d we have
E∥ˆµ(y) −µ∥2
W = E∥ˆµ(y) −y∥2
W −Tr(WΣ) + 2E[∇y · (WΣˆµ(y))].

Proof. Expanding the squared norm and applying Stein’s Lemma [Stein, 1981, Lemma 2] to the last
term yields

E∥ˆµ(y) −µ∥2
W = E∥ˆµ(y) −y∥2
W + E∥y −µ∥2
W + 2E⟨ˆµ(y) −y, W(y −µ)⟩

= E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E[∇y · (WΣˆµ(y)) −2 Tr(WΣ)]
(15)

We now turn to James–Stein-type estimators of the form ˆµ(y) = y −
c
∥P(y−b)∥2
Σ−1 P(y −b) for

c ∈R, θ ∈Rd, and P ∈Rd×d, roughly extending the one considered by Bock [1975] (in a more
general form) to P ̸= Id and b ̸= 0d. Setting x = P(y −θ), we apply Lemma B.1 to show that the
expected Σ−1-weighted MSE is

E∥ˆµ(y) −µ∥2
Σ−1 = d + E
c2 −2c Tr(P)

∥x∥2
Σ−1
+ 4c Tr(xx⊤Σ−1P)

∥x∥4
Σ−1



= d + E

"
c2 −2c Tr(P)

∥x∥2
Σ−1
+ 4c Tr(Σ−1

2 xx⊤Σ−1

2 Σ−1

2 PΣ
1
2 )
∥x∥4
Σ−1

#

≤d + E

"
c2∥Σ∥−2c Tr(ΣP) + 4c∥Σ−1

2 PΣ
1
2 ∥
∥x∥2
Σ−1

#
(16)

where to compute the derivative in SURE we made use of the matrix calculus tool of Laue et al.
[2018] and the last line follows by von Neumann’s trace inequality. This upper bound is minimized
at c = Tr(P) −2∥Σ−1

2 PΣ
1
2 ∥. Note that for P = Id this setting of c exactly recovers the Bock
estimator ˆµBock
θ
in Equation 6, apart from the positive-part correction.

13


---Page Break---
On the other hand, for shrinking to the pooled mean ˆµpooled(y) = 1d×dΣ−1y

Tr(Σ−1) we can apply a well-
known fact about the eigenvalues of symmetric rank-one updates (e.g. Bunch et al. [1978, Theorem 1])

to diagonal matrices to obtain ∥Σ−1

2 PΣ
1
2 ∥=
Id −Σ−1

2 1d×dΣ−1

2
Tr(Σ−1)

 = 1, which yields the setting

in Feldman et al. [2014] of c = Tr(P) −2∥Σ−1

2 PΣ
1
2 ∥= d −3:

ˆµBock(y) = ˆµpooled(y) +

1 −
d −3
∥y −ˆµpooled(y)∥2
Σ−1



+
(y −ˆµpooled(y))
(17)

Lastly, we note that it is straightforward to extend SureMap to the case of non-diagonal covariance
matrices Σ, i.e., when we want to simultaneously release performance estimates for groups with
different numbers of intersections (e.g., just race, just age, and both race and age). In this case there
will be nonzero covariance between elements of the vector y, e.g., those corresponding to a specific
age bracket and an intersection of that bracket with a specific race. To handle this, one only needs to
derive the SURE objective estimating the (non-diagonal) Σ−1-weighted risk of the MAP estimator
with (the same, non-diagonal) covariance Σ; this can be done with the following generalization of
Lemma B.1:
Lemma B.2. Suppose y ∼N(µ, Σ) for mean µ ∈Rd and p.s.d. covariance Σ ∈Rd×d, and
consider a function ˆµ : Rd →Rd s.t. for every i ∈[d] the function ˆµi is a.e. differentiable in yi and
E∥∇y · ˆµ(y)∥1 < ∞. Then for any p.s.d. W ∈Rd×d we have

E∥ˆµ(y) −µ∥2
W = E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E

d
X

i=1

d
X

j=1
(Σ ⊙(W∇y ˆµ(y) −W))i,j (18)

where ∇y ˆµ(y) is the Jacobian of ˆµ with entries ∇y;i,j ˆµ(y) = ∂yj ˆµi(y).

Proof. Note that since Ey = µ we have by a multivariate version of Stein’s lemma (e.g., Liu
[1994, Lemma 1]) that the following identity holds for any a.e. continuous f : Rd →R s.t.
E|∂yif(y)| < ∞∀i ∈[d]:

E[(y −µ)if(y)] = Cov((y −µ)i, f(y)) = ⟨Σi,:, ∇yf(y)⟩
(19)

Using this equality with f(y) = (W(ˆµ −y))i yields

E∥ˆµ(y) −µ∥2
W
= E∥ˆµ(y) −y∥2
W + E∥y −µ∥2
W + 2E⟨ˆµ(y) −y, W(y −µ)⟩

= E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E

d
X

i=1
⟨Σi,:, ∇y((Wˆµ(y))i −(Wy)i)⟩

= E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E

d
X

i=1

d
X

j=1
Σi,j∂yj(⟨Wi,:, ˆµ(y)⟩−Wi,:y)

= E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E

d
X

i=1

d
X

j=1
Σi,j(⟨Wi,:, ∇yj ˆµ(y)⟩−Wi,j)

= E∥ˆµ(y) −y∥2
W + Tr(WΣ) + 2E

d
X

i=1

d
X

j=1
(Σ ⊙(W∇y ˆµ(y) −W))i,j

(20)

14


---Page Break---
C
SureMap estimator

In this section we describe the prior in full for any number of attributes, prove expressivity results
reference in §3.1.2, and describe how to compute the estimator using coordinate descent.

C.1
A linear prior

Suppose each group g ∈[d] corresponds to an intersection g ∈×
k
a=1[da] of k attributes, with
attribute a having da possible classes.5 For example, the first attribute could be one of d1 different
age brackets and the second could be one of d2 different racial categories. For each subset A ⊂[k]
of attributes define a random tensor ZA ∈R×a∈Ada with i.i.d. entries ZA;c ∼N(0, 1), where
c ∈×a∈A[da] is a specific class combination of the attribute A. Then the additive intersectional
effects prior on the mean µg of group g is the weighted sum

µg = θg +
X

A∈2[k]
τAZA;gA
(21)

with coefficients τA ∈R corresponding to each A ∈2[k]. These hyperparameters thus determine the
strength of the effect of attribute intersection A on the group means, with τAZA;c being added to
the means of all groups g s.t. gA = c, i.e. whose attribute intersection results in the specific class
combination c.

Letting τ = (τ∅, . . . , τ[k]) ∈R2k
≥0 be the vector of hyperparameters τA and assuming τ[k] > 0, this
prior is equivalent to a Gaussian prior µ ∼N(θ, Λ(τ)) with covariance

Λ(τ) =
X

A∈2[k]
τ 2
AUAU⊤
A =
X

A∈2[k]
τ 2
ACA
(22)

for matrices UA ∈{0, 1}d×Q

a∈A da with orthogonal column vectors, each one corresponding to a
different attribute intersection in×a∈A[da] and its gth entry indicating whether group g is a subset
of that intersection. Their outer product matrices CA = UAU⊤
A have entries CA;g,h that are one if
gA = hA and zero otherwise, i.e. they indicate if groups g and h have the same type (e.g. (senior,
female)) of attribute intersection A (e.g. (age, sex)). In the simplest case, if we are disaggregating
across just one attribute, i.e. k = 1 and d1 = d, then Λ(τ) = τ 2
∅1d×d + τ 2
[k]Id. Since k ≤⌊log2 d⌋,
using this prior to define the covariance matrix reduces the number of hyperparameters to d+2k ≤2d
for ˆµMAP
θ,Λ(τ), compared to O(d2) for a general covariance Λ.

C.2
Expressivity of the linear prior

Theorem C.1. Consider a disaggregated setting with k attributes with d1, . . . , dk possible categories,
respectively, and total number of groups d = Qk
a=1 da. If we use a diagonal covariance Σ ∈Rd×d

satisfying Σh,h = σ2/nh ∀h then the following holds ∀y ∈Rd:

1. If τA = 0 ∀A ̸= [k] and θ ∈Rd then
lim
τ[k]→∞ˆµMAP
θ,Λ(τ)(y) = ˆµnaive(y)

2. If τA = 0 ∀A ̸∈{∅, [k]} then
lim
τ∅→∞
τ[k]→0
ˆµMAP
0d,Λ(τ)(y) = ˆµpooled(y)

3. If τA = 0 ∀A ̸= [k] and θ ∈Rd then lim
τ[k]→0 ˆµMAP
θ,Λ(τ)(y) = θ

4. If τA = 0 ∀A ̸∈{∅, [k]} and θ ∈Rd then
lim
τ∅→∞
τ[k]→0
ˆµMAP
θ,Λ(τ)(y) = ˆµpooled(y −θ) + θ

as desired.

5Note that this does not reduce the generality of our basic setup, which can be recovered with k = 1 and d1 = d.

15


---Page Break---
Proof. The first and third results can be easily shown using the fact that Σ is diagonal:

lim
τ[k]→∞ˆµMAP
0d,Λ(τ)(y) =
lim
τ[k]→∞

 
Id
τ 2
[k]
+ Σ−1
!−1  
θ
τ 2
[k]
+ Σ−1y

!

= y = ˆµnaive(y)
(23)

lim
τ[k]→0 ˆµMAP
θ,Λ(τ)(y) = lim
τ[k]→0

 
Id
τ 2
[k]
+ Σ−1
!−1  
θ
τ 2
[k]
+ Σ−1y

!

= θ
(24)

The second result follows from the last by substituting θ = 0d, so we just need to prove the latter
one. First note that

ˆµMAP
θ,Λ(τ)(y) =
 
Λ−1(τ) + Σ−1−1  
Λ−1(τ)θ + Σ−1y

(25)

= θ +
 
Λ−1(τ) + Σ−1−1 Σ−1(y −θ)
(26)

so we just need to show that the second term approaches ˆµpooled(y−θ). We use the Sherman-Morrison
formula to compute

Λ−1(τ) = (τ 2
[k]Id + τ 2
∅1d1⊤
d )−1 = Id

τ 2
[k]
−
τ 2
∅1d1⊤
d
τ 4
[k] + τ 2
[k]τ 2
∅d
(27)

and apply it again to compute

(Λ−1(τ) + Σ−1)−1 =

 
Id
τ 2
[k]
−τ 2
∅1d1⊤
d
τ 2
[k] + τ 2
∅d + Σ−1
!−1

=

 
Id
τ 2
[k]
+ Σ−1
!−1

+
τ 2
∅


Id
τ 2
[k] + Σ−1
−1
1d1⊤
d


Id
τ 2
[k] + Σ−1
−1

τ 4
[k] + τ 2
[k]τ 2
∅d −τ 2
∅Tr

 
Id
τ 2
[k] + Σ−1
−1!

(28)

Defining δ = y −θ, we have by L’Hôpital’s rule that ∀g ∈[d]

lim
τ∅→∞
τ[k]→0

 
(Λ−1(τ) + Σ−1)−1Σ−1δ


g

=
lim
τ∅→∞
τ[k]→0

δg/Σg,g
1
τ 2
[k] +
1
Σg,g
+

τ 2
[k]τ 2
∅

1+
τ2
[k]
Σg,g

dP

h=1

δh/Σh,h

1+
τ2
[k]
Σh,h

τ 2
[k] + τ 2
∅d −
dP

h=1

τ 2
∅

1+
τ2
[k]
Σh,h

= lim
τ∅→∞

lim
τ[k]→∞

τ 2
∅

1+
τ2
[k]
Σg,g





dP

h=1

δh/Σh,h

1+
τ2
[k]
Σh,h
−
τ 2
[k]δh/Σ2
h,h
 

1+
τ2
[k]
Σh,h

!2 −
τ 2
[k]/Σg,g
 

1+
τ2
[k]
Σg,g

!2

dP

h=1

δh/Σh,h

1+
τ2
[k]
Σh,h






lim
τ[k]→∞1 + τ 2
∅

dP

h=1

Σh,h

τ 2
[k]+Σh,h
2

= lim
τ∅→∞

τ 2
∅

dP

h=1

δh
Σh,h

1 +
dP

h=1

τ 2
∅
Σh,h

= 1

n

d
X

h=1
nh(yh −θh) = ˆµpooled
g
(y −θ)

(29)

as desired.

16


---Page Break---
Figure 6: Performance as the number of tasks varies, evaluated on SLACS (left) and CVC (right).
This plot is similar to Figure 5 but demonstrates the superiority of the multi-task version of SureMap
that we choose (MetaMap) relative to alternatives.

C.3
How to set the multi-task center

In the single-task case we set the prior mean θ = 0d; this can also be done in the multi-task setting.
Alternatively, we can use multi-task data to construct estimator ˆθ of some true underlying multi-task
mean θ and substitute the former for θ; for simplicity we will restrict to the ourselves to linear
estimators ˆθ = PT
t=1 Mtyt, where the matrices M1, . . . , MT are independent of y1, . . . , yT . To
determine these matrices, we will assign them some parametric form and set those parameters by
minimizing the sum of the Σ−1
t -weighted risks across tasks, as estimated by SURE:

T
X

t=1
Et∥ˆµMAP
ˆθ,Λ (yt) −µt∥2
Σ−1
t
=

T
X

t=1
Et

∥ˆµMAP
ˆθ,Λ (yt) −yt∥2
Σ−1
t
−d + 2∇yt · ˆµMAP
ˆθ,Λ (yt)


= dT +

T
X

t=1
Et∥At(ˆθ −yt)∥2
Σ−1
t
+ 2 Tr(AtMt −At)

(30)

for At = (Λ−1 + Σ−1
t )−1Λ−1, i.e. ˆµMAP
ˆθ,Λ (yt) = yt + At(ˆθ −yt) = yt −At(yt −PT
s=1 Msys).

C.3.1
SureSolve

This approach sets the prior mean by finding the θ that minimizes the above SURE objective, which
is equivalent to solving an overconstrained linear system:

ˆθ = dT + arg min
θ∈Rd

T
X

t=1
∥At(θ −yt)∥2
Σ−1
t
−2 Tr(At) =

 T
X

t=1
A⊤
t Σ−1
t At

!−1
T
X

t=1
A⊤
t Σ−1
t Atyt

(31)
Note that this estimator can be expressed in the desired linear form ˆθ = PT
t=1 Mtyt, with the

matrices Mt =
PT
s=1 A⊤
s Σ−1
s As
−1
A⊤
t Σ−1
t At depending on the parameters determining Λ.

C.3.2
MetaMap

Alternatively, if we assume the prior is correct for some θ and Λ, i.e. yt ∼N(θ, Λ + Σt), then a
natural estimator to use for θ given a fixed Λ is a MAP estimator of ˆθΓ with prior N(0d, Γ) for some
covariance Γ. Define ΣΛ = (PT
t=1(Λ + Σt)−1)−1 and note that yΛ = ΣΛ
PT
t=1(Λ + Σt)−1yt is
distributed as N(θ, ΣΛ). Then the MAP estimator of θ

ˆθΓ =
 
Γ−1 + Σ−1
Λ
−1 Σ−1
Λ yΛ =

T
X

t=1

 
Γ−1 + Σ−1
Λ
−1 (Λ + Σt)−1yt
(32)

has the necessary form ˆθ = PT
t=1 Mtyt for Mt = (Γ−1 + Σ−1
Λ )−1(Λ + Σt)−1. In this case the
matrices depend on both Λ and Γ, i.e. the covariance matrices of the prior and the meta-prior.

17


---Page Break---
Algorithm 2: Multi-task SureMap.

Input: target f : Z →R, samples St ⊂Z for each task t = 1, . . . , T, partition {Zg}d
g=1 of Z,
each group g an intersection of k ∈Z>0 attributes
// compute naive group means
for task t ∈[T] do

for group g ∈[d] do

nt;g ←|St ∩Zg|
yt;g ←
1
nt;g
P

z∈St∩Zg f(z)

// estimate group variances
n = PT
t=1 |St|
σ2 ←
1
n−dT
PT
t=1
Pd
g=1
P

z∈St∩Zg(f(z) −yt;g)2

for task t ∈[T] do

Σ−1
t
←diag(nt)/σ2

// compute auxiliary matrices (avoids inverting Λ)
Method At(τ):

// compute prior covariance (matrices CA are defined in Appendix C.1)
Λ ←P

A∈2[k] τ 2
ACA
Output: (Id + ΛΣ−1
t )−1

// compute auxiliary matrices (avoids inverting Σ−1
t
and Γ)
Method Mt(τ, υ):

// compute prior and meta-prior covariances
Λ ←P

A∈2[k] τ 2
ACA
Γ ←P

A∈2[k] υ2
ACA

Output:


Id + Γ PT
s=1 Σ−1
s (ΛΣ−1
s
+ Id)−1−1
ΓΣ−1
t (ΛΣ−1
t
+ Id)−1

// estimates prior mean using MAP
Method ˆθ(τ, υ):

Output: PT
t=1 Mt(τ, υ)yt

// optimize the sum of SUREs across tasks using L-BFGS-B
ˆτ, ˆυ = arg min

τ,υ∈R2k
≥0

PT
t=1 ∥At(τ)(ˆθ(τ, υ) −yt)∥2
Σ−1
t
+ 2 Tr (At(τ)(Mt(τ, υ) −Id))

// return an estimate of the group means of each task t using MAP
for task t ∈[T] do

Output: yt + At(ˆτ)(ˆθ(ˆτ, ˆυ) −yt)

D
Computation

Here we note additional computational details.

D.1
Nonnegativity of SureMap

In both the single and multi-task cases there is no guarantee that θ is in the convex hull of the values
in y or {yt}T
t=1, and in fact it can be negative. Since the quantities we are estimating are usually
nonnegative, in practice we do a post-hoc correction forcing θ to be nonnegative.

D.2
Optimization of SureMap

Both the single-task and multi-task variants of SureMap require solving optimization problems, the
former over τ 2 = (τ 2
A)A⊆[k] (Eq. 12) and the latter also over υ2 = (υ2
A)A⊆[k] (Eq. 14), both of
which are vectors in R2k
≥0. We find that both problems can be quickly and efficiently optimized using

L-BFGS-B over the entire domain R2k
≥0 and R2·2k
≥0 , respectively, using the default settings provided in

18


---Page Break---
its SciPy implementation.6 To initialize the algorithm we set all entries of τ 2 and υ2 to zero except
those corresponding to the entire set [k], which we set to one.7 In the remainder of this section, we
describe how to compute gradients (passed to L-BFGS-B) of the multi-task SureMap objective

dT +

T
X

t=1
Et

At

 

yt −

T
X

s=1
Msys

!

2

Σ−1
t

+ 2 Tr(AtMt −At)
(33)

where the matrices Mt differ based on whether we are using the SureSolve or MetaMap variant. The
gradients are taken w.r.t. the tuning parameters, which are the coefficients τ 2
1 , . . . , τ 2
2k used to define

the prior covariance Λ = P2k

i=1 τ 2
i UiU⊤
i and, in the second case, the coefficients υ2
1, . . . , υ2
2k used to

define the meta-prior covariance Γ = P2k

i=1 υ2
i UiU⊤
i .8 Noting that At = (Λ+Σ−1
t )−1Λ−1 = (Id+
ΛΣ−1
t )−1 and (Λ+Σt)−1 = Λ−1(Id+ΣtΛ−1)−1 = Σ−1
t (Id+ΛΣ−1
t )−1 = Σ−1
t At, we have the
derivative ∂iAt = −AtUiU⊤
i Σ−1
t At w.r.t. τ 2
i , which yields ∂i Tr(At) = Tr(−AtUiU⊤
i Σ−1
t At)
and

∂i∥At(θ −yt)∥2
Σ−1
t
= 2(θ −yt)⊤A⊤
t Σ−1
t ∂iAt(θ −yt)

= −2(θ −yt)⊤A⊤
t Σ−1
t AtUiU⊤
i Σ−1
t At(θ −yt)
(34)

D.2.1
SureSolve

First note that

∂i[A⊤
t Σ−1
t At] = ∂i[Σ−1
t A2
t] = Σ−1
t (At∂iAt + ∂iAtAt)

= −Σ−1
t (A2
t + At)UiU⊤
i Σ−1
t (A2
t + At)
(35)

where the first step follows by Σ−1
t At = Σ−1
t (Id + ΛΣ−1
t )−1 = (Id + Σ−1
t Λ)−1Σ−1
t
= A⊤
t Σ−1
t .

Then for Mt =
PT
s=1 A⊤
s Σ−1
s As
−1
A⊤
t Σ−1
t At we have

∂iMt =

 T
X

s=1
A⊤
s Σ−1
s As

!−1

∂i[A⊤
t Σ−1
t At] −

 T
X

s=1
A⊤
s Σ−1
s As

!−1
T
X

s=1
∂i[A⊤
s Σ−1
s As]Mt

= −2

 T
X

s=1
A⊤
s Σ−1
s As

!−1

Σ−1
t (A2
t + At)UiU⊤
i Σ−1
t (A2
t + At)

+ 2

 T
X

s=1
A⊤
s Σ−1
s As

!−1
T
X

s=1
Σ−1
s (A2
s + As)UiU⊤
i Σ−1
s (A2
s + As)Mt

= −2Bi,t + 2

T
X

s=1
Bi,sMt

(36)

where Bi,t =
PT
s=1 A⊤
s Σ−1
s As
−1
Σ−1
t (A2
t +At)UiU⊤
i Σ−1
t (A2
t +At). We can then compute

∂i Tr(AtMt) = −2 Tr

 

At

 

Bi,t −

T
X

s=1
Bi,sMt

!!

−Tr(AtUiU⊤
i Σ−1
t AtMt)

= −2 Tr

 

At

 

Bi,t +

 

UiU⊤
i Σ−1
t At −

T
X

s=1
Bi,s

!

Mt

!!
(37)

6https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
7This initialization corresponds to setting the prior covariance to be the d × d identity.
8We use indices i instead of subsets A here for simplicity.

19


---Page Break---
and

∂i

At

 

yt −

T
X

s=1
Msys

!

2

Σ−1
t

= ∂i

T
X

s=1

T
X

r=1
(yt −Msys)⊤A⊤
t Σ−1
t At(yt −Mryr)

= −2

T
X

s=1
(∂iMsys)⊤A⊤
t Σ−1
t At(yt −ˆθ)

+ 2(yt −ˆθ)⊤A⊤
t Σ−1
t ∂iAt(yt −ˆθ)

= 4

T
X

s=1
y⊤
s B⊤
i,sA⊤
t Σ−1
t At(yt −ˆθ)

−4ˆθ⊤
T
X

s=1
B⊤
i,sA⊤
t Σ−1
t At(yt −ˆθ)

−2(yt −ˆθ)⊤A⊤
t Σ−1
t AtUiU⊤
i Σ−1
t At(yt −ˆθ)

= 4

T
X

s=1
(ys −ˆθ)⊤B⊤
i,sA⊤
t Σ−1
t At(yt −ˆθ)

−2(yt −ˆθ)⊤A⊤
t Σ−1
t AtUiU⊤
i Σ−1
t At(yt −ˆθ)

(38)

D.3
MetaMap

Note that (Λ + Σt)−1 = Σ−1
t (ΛΣ−1
t
+ Id) = Σ−1
t At so we can write the matrices representing
the estimator as Mt = (Γ−1 + Σ−1
Λ )−1(Λ + Σt)−1 = (Γ−1 + PT
s=1 Σ−1
s As)−1Σ−1
t At. Thus

∂iMt = (Γ−1 + Σ−1
Λ )−1Σ−1
t ∂iAt −(Γ−1 + Σ−1
Λ )−1
T
X

s=1
Σ−1
s ∂iAsMt

= −MtUiU⊤
i Σ−1
t At +

T
X

s=1
MsUiU⊤
i Σ−1
s AsMt

(39)

We can then compute

∂i Tr(AtMt) = −Tr(At(MtAt + AtMt)UiU⊤
i Σ−1
t ) +

T
X

s=1
Tr(AtMsUiU⊤
i Σ−1
s AsMt)

(40)

20


---Page Break---
and

∂i

At

 

yt −

T
X

s=1
Msys

!

2

Σ−1
t

= ∂i

T
X

s=1

T
X

r=1
(yt −Msys)⊤A⊤
t Σ−1
t At(yt −Mryr)

= −2

T
X

s=1
(∂iMsys)⊤A⊤
t Σ−1
t At(yt −ˆθΓ)

+ 2(yt −ˆθΓ)⊤A⊤
t Σ−1
t ∂iAt(yt −ˆθΓ)

= 2

T
X

s=1
y⊤
s (MsUiU⊤
i Σ−1
s As)⊤AtΣ−1
t At(yt −ˆθΓ)

−2

T
X

s=1

T
X

r=1
y⊤
s (MrUiU⊤
i Σ−1
r ArMs)⊤AtΣ−1
t At(yt −ˆθΓ)

−2(yt −ˆθΓ)⊤A⊤
t Σ−1
t AtUiU⊤
i Σ−1
t At(yt −ˆθΓ)

= 2

T
X

s=1
(ys −ˆθΓ)⊤A⊤
s Σ−1
s UiU⊤
i M⊤
s A⊤
t Σ−1
t At(yt −ˆθΓ)

−2(yt −ˆθΓ)⊤A⊤
t Σ−1
t AtUiU⊤
i Σ−1
t At(yt −ˆθΓ)
(41)

Lastly, note that Mt = (Γ−1 + Σ−1
Λ )−1Σ−1
t At = (Id + ΓΣ−1
Λ )−1ΓΣ−1
t At and redefine the
derivative ∂i to be w.r.t. υ2
i , so that

∂iMt = −(Id + ΓΣ−1
Λ )−1UiU⊤
i Σ−1
Λ Mt + (Id + ΓΣ−1
Λ )−1UiU⊤
i Σ−1
t At
(42)

Then we have ∂i Tr(AtMt) = Tr(At(Id + ΓΣ−1
Λ )−1UiU⊤
i (Σ−1
t At −Σ−1
Λ Mt)) and

∂i
At

yt −ˆθΓ

2

Σ−1
t

= −2

T
X

s=1
(∂iMsys)⊤A⊤
t Σ−1
t At(yt −ˆθΓ)

= 2

 

Σ−1
Λ ˆθΓ −

T
X

s=1
Σ−1
s Asys

!⊤

UiU⊤
i (Id + ΓΣ−1
Λ )−⊤A⊤
t Σ−1
t At(yt −ˆθΓ)

(43)

D.4
Handling groups with no data

It is frequently the case that a specific group g may not have any examples, i.e., ng = 0, and so we
cannot define Σg,g = σ2/ng. At the same time, we may need to handle the dimension associated
with this group, either because we still need to report a value for it or because we are in the multi-task
setting and other tasks do have examples of that group that may allow us to get an estimate. To handle
this issue, in all computations we only use the precision matrix Σ−1, which can be easily defined in the
case of ng = 0 using Σ−1
g,g = ng/σ2 = 0. Note in particular that (Λ+Σ)−1 = Σ−1(ΛΣ−1 +Id)−1.

D.5
Handling near-singular covariances

Since we would like to optimize over the entire domain τ 2 ∈R2k
≥0 but Λ(02k) = 0d×d is singular,
we avoid inverting prior covariance matrices (i.e. Λ, Γ) in all computations. Note in particular that
At = (Λ−1 + Σ−1)−1Λ−1 = (Id + ΛΣ−1)−1.

21


---Page Break---
D.6
Handling group variances for AUC

While obtaining unbiased variance estimates for averages is straightforward, it is more involved
for multi-point statistics such as AUC. For simplicity we just use (ng + 1)/(12ngn(0)
g n(1)
g ), where
n(i)
g
is the number of members of group g with label i; this estimate is derived from the variance
estimate of the related Mann-Whitney U-statistic [Siegel, 1956]. However, future work may consider
improvements based on more complicated approaches [Cortes and Mohri, 2003, Wang and Guo,
2020], or using bootstrapping techniques as is done by Herlihy et al. [2024] for structured regression.

E
Derivation of SureMap estimators via ridge regression

In this appendix we show that ˆµSM and ˆµSM
t
can be derived as ridge regression estimators.

Similar to the notation in Appendix C, we consider k sensitive attributes (like sex, age, etc.) indexed
by a ∈[k]. The ath sensitive attribute is denoted ga and takes values in [da]. The joint sensitive
attribute g takes values in G = [d1] × [d2] × · · · × [dk] with the cardinality d = |G| = d1d2 · · · dk.
For A ⊆[k], write gA for the tuple (ga)a∈A.

E.1
Single-task regression model

Each joint attribute g ∈G identifies an intersectional group (of order k). We seek to jointly fit
means of all these groups, represented as a vector µ ∈R|G|, based on the vector of empirical means
y ∈R|G|. We posit a regression model
µ = Φβ

where Φ ∈R|G|×|J | is a feature matrix and β ∈R|J | is a vector of regression coefficients. The
columns of Φ are referred to as features and indexed by j ∈J , where J is some index set. We
assume a Gaussian prior on the parameter β ∼N(0, K) and a Gaussian distribution over observation
errors, so y ∼N(Φβ, Σ), where K and Σ are, respectively, the prior covariance matrix and error
covariance matrix.

The error covariance matrix Σ is assumed fixed and positive definite, the prior covariance matrix
K is viewed as a hyperparameter (with a specific structure to reduce its dimension); for simplicity,
we assume that K is positive definite (but that assumption is not necessary). Any generic forms of
Φ, K and Σ can be considered. Here we exhibit a specific form of Φ and K under which the MAP
regression estimator is the same as ˆµSM, when provided with the same error covariance matrix Σ.

We consider a structured form of matrix Φ, with features being indicators of tuples of sensitive
attribute values. The features are indexed by j ∈J , where

J =

(A, c) : A ⊆[k], c ∈Q

a∈A[da]
	
,

and defined as
Φg,(A,c) = 1{gA = c}.
(44)

We will also consider subsets of features that focus on a specific subset of sensitive attributes A ⊆[k],

JA =

(A, c) : c ∈Q

a∈A[da]
	
,

and write ΦA ∈R|G|×|JA| for the submatrix composed of features indexed by j ∈JA.

The prior matrix K is diagonal, specified using a set of hyperparameters τ = (τA)A⊆[k], with
diagonal entries
K(A,c),(A,c) = τ 2
A.
(45)

The MAP solution under the model described above is obtained by solving

ˆβ = arg min
β

h
(y −Φβ)⊤Σ−1(y −Φβ) + β⊤K−1β
i
,

which is a weighted ridge regression problem. Setting the gradient to zero, the solution must satisfy

−2Φ⊤Σ−1(y −Φβ) + 2K−1β = 0.

22


---Page Break---
Hence,
ˆβ = (Φ⊤Σ−1Φ + K−1)−1Φ⊤Σ−1y.

This yields a regression-based estimator

ˆµreg = Φ ˆβ = Φ(Φ⊤Σ−1Φ + K−1)−1Φ⊤Σ−1y.
(46)

In contrast, by Eqs. (11) and (7), the ˆµSM estimator is obtained as

ˆµSM = (Λ−1 + Σ−1)−1Σ−1y,

where the matrix Λ (following Eq. 22 in Appendix C.1) depends on the hyperparameter vector τ as

Λ =
X

A⊆[k]
τ 2
AUAU⊤
A,

where UA is precisely the submatrix ΦA defined above. Thus, writing ϕA,c ∈R|G| for the column
of Φ indexed by (A, c), we obtain

Λ =
X

A⊆[k]
τ 2
AΦAΦ⊤
A =
X

A⊆[k]

X

c∈Q

a∈A[da]
τ 2
AϕA,cϕ⊤
A,c = ΦKΦ⊤.

Hence, the ˆµSM estimator can be rewritten as

ˆµSM =
 
(ΦKΦ⊤)−1 + Σ−1−1Σ−1y.
(47)

Theorem E.1. With Φ and K defined as above (Eqs. 44 and 45), we have ˆµreg = ˆµSM.

In the proof we use a variant of Sherman–Morrison–Woodbury formula [Horn and Johnson, 2013,
Eq. 0.7.4.1], specialized to symmetric positive definite matrices:

Proposition E.1 (Symmetric SMW Formula). Let A ∈Rn×n and R ∈Rm×m be symmetric positive
definite matrices and let X ∈Rn×m. Then

(A + XRX⊤)−1 = A−1 −A−1X(R−1 + X⊤A−1X)−1X⊤A−1.

Proof of Theorem E.1. It suffices to show that

Φ(Φ⊤Σ−1Φ + K−1)−1Φ⊤=
 
(ΦKΦ⊤)−1 + Σ−1−1.

We do this by direct calculation, using the symmetric SMW formula:

Φ(Φ⊤Σ−1Φ + K−1)−1Φ⊤= Φ
h
K −KΦ⊤(Σ + ΦKΦ⊤)−1ΦK
i
Φ⊤

= (ΦKΦ⊤) −(ΦKΦ⊤)

(Σ + (ΦKΦ⊤)
−1
(ΦKΦ⊤)

=
 
(ΦKΦ⊤)−1 + Σ−1−1.

The first equality follows by the SMW formula with A = K−1, R = Σ−1 and X = Φ⊤, the second
by multiplying out the terms, and the third by the SMW formula with A−1 = ΦKΦ⊤, R−1 = Σ
and X equal to the d × d identity matrix.

E.2
Multi-task regression model

Consider multi-task setting with tasks indexed by t = 1, . . . , T. We write T = [T] for the set of tasks.
Multi-task setting can be reduced to single-task setting by viewing the task id as an additional sensitive
attribute, and performing the same analysis as for the single-task setting, but with G′ = T × G, with
the dimension d′ = |G′| = Td.

Formally, we seek to fit µ′ ∈R|G′| based on the vector of empirical means y′ ∈R|G′|. The entries
of µ′ and y′ are denoted as µ′
t,g and y′
t,g for the task t and the intersectional group g. We denote
task-specific slices of these vectors as µ′
t = µ′
{t}×G and y′
t = y′
{t}×G.

23


---Page Break---
Features are indexed by elements of J ′ = S × J , where S = T ∪{glob}, so there are task-
specific and global features. The feature matrix Φ′ ∈R|G′|×|J ′| uses the single-task feature matrix
Φ ∈R|G|×|J | (defined in Eq. 44) as a building block. The matrix Φ′ has a block structure, with
|T | × |S| blocks of size |G| × |J | defined as

Φ′
t,s =
Φ
if s = t or s = glob,
0|G|×|J |
otherwise.

We posit the regression model
µ′ = Φ′β′,

where β′ ∈RJ ′. With the definition of Φ′ as above, this boils down to

µ′
t = Φ(β′
t + β′
glob)
for all t ∈T .

As before, we assume a Gaussian prior and Gaussian errors, β′ ∼N(0, K′), y′ ∼N(Φ′β′, Σ′).

The error covariance Σ′ ∈R|G′|×|G′| has a block-diagonal form with single-task error covariance
matrices Σt ∈R|G|×|G|, for t ∈[T], along the diagonal.

We consider a structured form of the prior covariance matrix K′ specified by two vectors of hyperpa-
rameters: τ = (τ 2
A)A⊆[k] and υ = (υA)A⊆[k]. The matrix K′ is diagonal and positive definite, with
entries

K′
(s,A,c),(s,A,c) =
τ 2
A
if s ∈T ,
υA
if s = glob.

It can also be viewed as a block-diagonal matrix with |S| diagonal blocks of size |J |× |J | defined as

K′
s,s =
K
if s ∈T ,
V
if s = glob,

where K is the single-task prior matrix defined in Eq. (45), and V ∈R|J |×|J | is an analogous matrix
based on the vector of hyperparameters υ rather than τ, with diagonal entries

V(A,c),(A,c) = υA.
(48)

The multi-task regression-based estimator is obtained by solving the resulting MAP regression
problem. Similarly to the single-task case, the estimator takes form

ˆµ′reg = Φ′
Φ′⊤Σ′−1Φ′ + K′−1−1
Φ′⊤Σ′−1y′,
(49)

with the individual task estimates denoted ˆµ′
t
reg.

The multi-task SureMap estimator, introduced in §3.2.2, takes form

ˆµSM
t
= y′
t +

Λ−1 + Σ−1
t
−1
Λ−1 ˆθ −y′
t

,
(50)

where

ˆθ =

Γ−1 +

T
X

t=1
(Λ + Σt)−1
−1
T
X

t=1
(Λ + Σt)−1y′
t,
(51)

and
Λ = ΦKΦ⊤
and
Γ = ΦVΦ⊤.

Theorem E.2. With Φ′, K′ and Σ′ defined as above, we have ˆµ′
t
reg = ˆµSM
t
for all t ∈[T].

In the proof we use the following identities:

Proposition E.2. Let A, B ∈Rn×n be symmetric positive definite matrices and let I be the n × n
identity matrix. Then

(A−1 + B−1)−1A−1 = I −A(A + B)−1 = B(A + B)−1.

24


---Page Break---
Proof. We have

(A−1 + B−1)−1A−1 =

A −A(A + B)−1A

A−1

= I −A(A + B)−1

= (A + B)(A + B)−1 −A(A + B)−1

= B(A + B)−1,

where the first equality follows by the SMW formula (Proposition E.1), and the remaining equalities
follow by simple algebraic manipulations.

Proof of Theorem E.2. Starting with Eq. (49), we have

ˆµ′reg = Φ′
Φ′⊤Σ′−1Φ′ + K′−1−1
Φ′⊤Σ′−1y′

=
 
Φ′K′Φ′⊤−1 + Σ′−1−1
Σ′−1y′

=

I −Σ′ 
Φ′K′Φ′⊤+ Σ′−1
y′,
(52)

where the second equality follows by the same reasoning as in the proof of Theorem E.1, and the
third equality follows by Proposition E.2.

We next evaluate Φ′K′Φ′⊤. Note that the matrices Φ′, K′ and Σ′ have the following block structure:

Φ′ =




Φ
Φ
...
...
Φ
Φ



,
K′ =






K
...
K
V




,
Σ′ =




Σ1
...
ΣT



.

Thus,

Φ′K′Φ′⊤=




Φ
...
Φ








K
...
K








Φ⊤
...
Φ⊤



+




Φ...
Φ



V
 
Φ⊤· · · Φ⊤

=




ΦKΦ⊤
...
ΦKΦ⊤



+




I...
I



ΦVΦ⊤(I · · · I)

=




Λ
...
Λ



+




I...
I



Γ (I · · · I) = Λ′ + XΓX⊤,

where in the last line we introduced the notation Λ′ for the block-diagonal matrix with T copies of
matrix Λ along diagonal, and notation X for the matrix obtained by stacking T copies of the |G|×|G|
identity matrix I on top of each other. Plugging the last expression back into Eq. (52), we obtain

ˆµ′reg =
h
I −Σ′
Λ′ + XΓX⊤+ Σ′−1i
y′

= y′ −Σ′h
(Λ′ + Σ′)−1

−(Λ′ + Σ′)−1X

Γ−1 + X⊤(Λ′ + Σ′)−1X
−1
X⊤(Λ′ + Σ′)−1i
y′,
(53)

where the second equality follows by the SMW formula (Proposition E.1) with A = Λ′ +Σ′, R = Γ,
and X = X. We next focus on simplifying the last term in the bracket in Eq. (53).

Since Λ′ and Σ′ are block-diagonal, the matrix (Λ′ +Σ′)−1 is also block-diagonal with blocks along
the diagonal equal to (Λ + Σt)−1 for t = 1, . . . , T. Thus,

X⊤(Λ′ + Σ′)−1X = (I · · · I)




(Λ + Σ1)−1
...
(Λ + ΣT )−1








I...
I



=

T
X

t=1
(Λ + Σt)−1,

25


---Page Break---
and similarly,

X⊤(Λ′ + Σ′)−1y′ =

T
X

t=1
(Λ + Σt)−1y′
t.

Therefore,

Γ−1 + X⊤(Λ′ + Σ′)−1X
−1
X⊤(Λ′ + Σ′)−1y′

=

Γ−1 +

T
X

t=1
(Λ + Σt)−1
−1
T
X

t=1
(Λ + Σt)−1y′
t = ˆθ.

Plugging this back in Eq. (53), we obtain

ˆµ′reg = y′ −Σ′h
(Λ′ + Σ′)−1y′ −(Λ′ + Σ′)−1Xˆθ
i
= y′ −Σ′(Λ′ + Σ′)−1



y′ −




I...
I



ˆθ



.

Using, again, the fact that matrices Λ′ and Σ′ are block-diagonal, the task-specific blocks of ˆµ′reg

must be equal to

ˆµ′
t
reg = y′
t −Σt(Λ + Σt)−1(y′
t −ˆθ)

= y′
t −
 
Λ−1 + Σ−1
t
−1Λ−1(y′
t −ˆθ)

= y′
t +
 
Λ−1 + Σ−1
t
−1Λ−1(ˆθ −y′
t) = ˆµSM
t ,

where the second equality follows by Proposition E.2.

F
Resources

F.1
Data and model resources

We make use of data / models with the following sources / licenses:
1. Strack et al. [2014]: CC Attribution License.
2. Weerts et al. [2023]: MIT License.
3. Ardila et al. [2020]: CC0 License.
4. Radford et al. [2023]: Apache-2.0 License.
5. https://archive.ics.uci.edu/dataset/2/adult: CC BY 4.0 License
6. https://huggingface.co/JaaackXD/Llama-3-70B-GGUF: Meta Llama 3 License
We use the third and fourth resources to create a dataset of Whisper model evaluations on Common
Voice utterances, which result in the Common Voice and CVC tasks described in §4.2; we also use
the last two resources to create a dataset of in-context evaluations of Llama 3 on the Adult dataset,
which results in the Adult task described in §4.1. Both resources are released under a CC BY 4.0
License and are available at https://github.com/mkhodak/SureMap.

F.2
Computational

By far the most computation was required to generate the Common Voice, CVC, and Adult tasks,
which was done on a machine with two RTX-8000 GPUs and took about a week. As described
above, the corresponding datasets are made publicly available and easy to re-use without any GPU
access. Given these dataset, the main experiments were run on a 40-core machine and take a couple
hours, with the vast majority of this time spent running the structured regression approach of Herlihy
et al. [2024]; see Figure 7 for a summary of the costs associated with each method evaluated in this
paper. Code for both generating the task data and reproducing the method evaluations is available at
https://github.com/mkhodak/SureMap.

26


---Page Break---
Figure 7: Cost of running the methods evaluated in this paper as a function of the number of tasks,
with both axes scaled logarithmically. While they are 1-2 orders of magnitude more expensive than
the baselines—which all have closed form expressions—SureMap and multi-task SureMap are also
1-2 orders of magnitude than structured regression [Herlihy et al., 2024]. Note that for the most part
the runtime of all methods will usually be dwarfed by the cost of inference.

Figure 8:
Evaluations on the regression variant of Diabetes, using MAE as the target metric,
disaggregating by race, sex, and age. On the left the MAE is taken across all groups, while in the
center it is only over large groups and on the left over small groups. Large and small are defined as
the top and bottom half of all groups, respectively.

Figure 9:
Evaluations on the regression variant of Diabetes, using MSE as the target metric,
disaggregating by race, sex, and age. On the left the MAE is taken across all groups, while in the
center it is only over large groups and on the left over small groups. Large and small are defined as
the top and bottom half of all groups, respectively.

G
Additional evaluations

In Figures 8 & 9, we compare evaluation methods on the regression variant of the Diabetes task, where
the goal is to predict a patient’s length of stay using ridge regression. In Figures 10 & 12, we compare
evaluation methods on Diabetes and SLACS with AUC as the target metric. In Figures 11 & 13, we
compare evaluation methods on Common Voice and CVC with the character error rate (CER) as the
target metric. And finally, in Figures 14 & 15, we compare evaluation methods on Diabetes, Adult,
Common Voice, SLACS, and CVC according to RMSE instead of MAE.

27


---Page Break---
Figure 10: Single-task evaluations on the Diabetes classification setting (disaggregating by race,
sex, and age) when using AUC as the target metric. In the left column the RMSE is taken across
all groups, while in the center it is only over large groups and on the right over small groups. Large
and small are defined as the top and bottom half of all groups, respectively.

Figure 11: Single-task evaluation on the Common Voice ASR setting (bottom, disaggregating by
sex and age) when using CER as the target metric. In the left column the MAE is taken across all
groups, while in the center it is only over large groups and on the right over small groups. Large
and small are defined as the top and bottom half of all groups, respectively.

Figure 12: Multi-task evaluations on state-level ACS data (disaggregating by race, sex, and age)
when using AUC as the target metric. On the left is the performance across different subsampling
rates while on the right we show (multiplicative) performance improvement over the naive estimator
on different tasks at subsampling rate 0.1.

Figure 13: Multi-task evaluations on Common Voice clusters (disaggregating by sex and age) when
using CER as the target metric. On the left is the performance across different subsampling rates
while on the right we show (multiplicative) performance improvement over the naive estimator on
different tasks at subsampling rate 0.1.

28


---Page Break---
Figure 14: Single-task evaluations on the Diabetes classification setting (top, disaggregating by race,
sex, and age), the Adult in-context classification setting (middle, disaggregating by race, sex, and age),
and the Common Voice ASR setting (bottom, disaggregating by sex and age); these are the same evalu-
ations as Figure 2 except with RMSE instead of MAE as the performance measure. In the left column
the RMSE is taken across all groups, while in the center it is only over large groups and on the right
over small groups. Large and small are defined as the top and bottom half of all groups, respectively.

Figure 15: Multi-task evaluations on state-level ACS data (top, disaggregating by race, sex, and age)
and Common Voice clusters (bottom, disaggregating by sex and age); these plots visualize the same
evaluations as Figure 3 except they use RMSE instead of MAE as the performance measure. On the
left is the performance across different subsampling rates while on the right we show (multiplicative)
performance improvement over the naive estimator on different tasks at subsampling rate 0.1.

29


---Page Break---
NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: our abstract and introduction both describe the introduction of a new method
for disaggregated evaluation, which we do.
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
Justification: c.f. Sections 3.3 and 5.3.
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

30


---Page Break---
Justification: c.f. Appendix C.2.
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
Justification: c.f. Appendix D
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

31


---Page Break---
Answer: [Yes]
Justification: https://github.com/mkhodak/SureMap
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
Justification: c.f. Section 5, Appendix D, and the linked code.
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
Justification: all figures except scatter plots have 95% confidence intervals.
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

32


---Page Break---
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
Justification: c.f. Appendix F.2.
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
Justification: we have reviewed the Code of Ethics and our work conforms.
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
Justification: c.f. Section 6.
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

33


---Page Break---
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

Justification: The one asset released is just evaluations of an open-source model on open-
source data and so unlikely to be misused.

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

Justification: c.f. Appendix F.1.

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

34


---Page Break---
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: we release new datasets in conjunction with three of the tasks described in
Section 4; see that section and Appendix F.1 for further details. This data is provided at the
code link.
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
Justification:
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
Justification:
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

35


---Page Break---
